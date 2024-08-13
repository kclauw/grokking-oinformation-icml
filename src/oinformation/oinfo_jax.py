from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from src.oinformation.hoi.metrics.base_hoi import HOIEstimator
from src.oinformation.hoi.core.entropies import get_entropy, prepare_for_entropy, entropy_gcmi
from src.oinformation.hoi.utils.progressbar import get_pbar
from src.oinformation.hoi.core.combinatory import combinations
from src.oinformation.hoi.utils.logging import logger, set_log_level
from src.oinformation.oinfo_jax_original import OinfoOriginal
import pandas as pd 
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"


@partial(jax.jit, static_argnums=(2, 3))
def _oinfo_no_ent_cluster(inputs, index, entropy_3d=None, entropy_4d=None):
    data, acc = inputs
    
    msize = len(index)
 
    # tuple selection
    
    
    x = jnp.hstack([c for c in data[index]]).T
    
    # force x to be 3d
    
    assert x.ndim >= 2
    if x.ndim == 2:
        x = x[jnp.newaxis, ...]
 
    x_c = data[:, index, :]

    # compute h(x^{n})
    h_xn = entropy_3d(x_c)

    # compute \sum_{j=1}^{n} h(x_{j}
    h_xj_sum = entropy_4d(x_c[:, :, jnp.newaxis, :]).sum(0)

    # compute \sum_{j=1}^{n} h(x_{-j}
    h_xmj_sum = entropy_4d(x_c[:, acc, :]).sum(0)

    # compute oinfo
    oinfo = (msize - 2) * h_xn + h_xj_sum - h_xmj_sum

    return inputs, oinfo

@partial(jax.jit, static_argnums=(2, 3))
def _oinfo_no_ent(inputs, index, entropy_3d=None, entropy_4d=None):
    data, acc = inputs
    msize = len(index)
    
    # tuple selection
    x_c = data[:, index, :]

    # compute h(x^{n})
    h_xn = entropy_3d(x_c)

    # compute \sum_{j=1}^{n} h(x_{j}
    h_xj_sum = entropy_4d(x_c[:, :, jnp.newaxis, :]).sum(0)

    # compute \sum_{j=1}^{n} h(x_{-j}
    h_xmj_sum = entropy_4d(x_c[:, acc, :]).sum(0)

    # compute oinfo
    oinfo = (msize - 2) * h_xn + h_xj_sum - h_xmj_sum

    return inputs, oinfo



class Oinfo():
    r"""O-information.

    The O-information is defined as the difference between the total
    correlation (TC) minus the dual total correlation (DTC):

    .. math::

        \Omega(X^{n})  &=  TC(X^{n}) - DTC(X^{n}) \\
                       &=  (n - 2)H(X^{n}) + \sum_{j=1}^{n} [H(X_{j}) - H(
                        X_{-j}^{n})]

    .. warning::

        * :math:`\Omega(X^{n}) > 0 \Rightarrow Redundancy`
        * :math:`\Omega(X^{n}) < 0 \Rightarrow Synergy`

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1, 2), (2, 7, 8, 9)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Rosas et al., 2019 :cite:`rosas2019oinfo`
    """

    __name__ = "O-Information"

    def __init__(self, clustered_features, y=None, multiplets=None, verbose=None):
        #clustered_features = convert_to_jax(clustered_features)
        self.x = []
        #Convert features to jax format
        for c in clustered_features:
            label, i, feat, size = c
            self.x.append(jnp.array(feat))
        self.x = jnp.array(self.x)
        
        #self.clustered_features = np.array(clustered_features, dtype=object)
        
        self.n_features = len(clustered_features)
        
        # compute only selected multiplets
        self._custom_mults = None
        if isinstance(multiplets, (list, np.ndarray)):
            self._custom_mults = multiplets
            
        # additional variable along feature dimension
        self._task_related = isinstance(y, (list, np.ndarray, tuple))
        if self._task_related:
            y = np.asarray(y)
            if y.ndim == 1:
                assert len(y) == x.shape[0]
                y = np.tile(y.reshape(-1, 1, 1), (1, 1, x.shape[-1]))
            elif y.ndim == 2:
                assert y.shape[0] == x.shape[0]
                assert y.shape[-1] == x.shape[-1]
                y = y[:, np.newaxis, :]
            x = np.concatenate((x, y), axis=1)
            
        
        """
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )
        """

    def fit_exhaustive(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute the O-information.

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gcmi', 'binning', 'knn', 'kernel}
            Name of the method to compute entropy. Use either :

                * 'gcmi': gaussian copula entropy [default]. See
                  :func:`hoi.core.entropy_gcmi`
                * 'binning': binning-based estimator of entropy. Note that to
                  use this estimator, the data have be to discretized. See
                  :func:`hoi.core.entropy_bin`
                * 'knn': k-nearest neighbor estimator. See
                  :func:`hoi.core.entropy_knn`
                * 'kernel': kernel-based estimator of entropy
                  see :func:`hoi.core.entropy_kernel`

        kwargs : dict | {}
            Additional arguments are sent to each entropy function
        """
        # ________________________________ I/O ________________________________
        # check min and max sizes
        minsize, maxsize = self._check_minmax(minsize, maxsize)

        # prepare the x for computing entropy
        
       
        # get entropy function
        entropy = jax.vmap(get_entropy(method=method, **kwargs))
        
        oinfo_no_ent = partial(
            _oinfo_no_ent_cluster,
            entropy_3d=entropy,
            entropy_4d=jax.vmap(entropy, in_axes=1),
        )
        
        # prepare output
        kw_combs = dict(maxsize=maxsize, astype="jax")
        h_idx = self.get_combinations(minsize, **kw_combs)
   
        order = self.get_combinations(minsize, order=True, **kw_combs)

       
        # subselection of multiplets
        self._multiplets = self.filter_multiplets(h_idx, order)
        order = (self._multiplets >= 0).sum(1)
       
        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # ______________________________ ENTROPY ____________________________
        
        
        offset = 0
        hoi = jnp.zeros((len(order), self.n_features), dtype=jnp.float32)
        
        combs = []
        for msize in pbar:
            pbar.set_description(desc="Oinfo (%i)" % msize, refresh=False)

            # combinations of features
            keep = order == msize
            _h_idx = self._multiplets[keep, 0:msize]
           
            # generate indices for accumulated entropies
            acc = jnp.mgrid[0:msize, 0:msize].sum(0) % msize
  
            multiplets, _hoi = jax.lax.scan(oinfo_no_ent, (self.x, acc[:, 1:]), _h_idx)
            print("---")
            print(_hoi)
            exit(0)
            #print(entropy_gcmi(_hoi[0]))
            exit(0)
            
            combs.extend(_h_idx.tolist())
            # fill variables
            n_combs, n_feat = _h_idx.shape
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)
            
            # updates
            offset += n_combs

        
        df = pd.DataFrame({
            'multiplet': combs,
            'metric_value': np.array(hoi).flatten().tolist(),
            'size': [len(c) for c in combs],
            "metric_name": ["o_information" for c in combs]
        })
        
        df = df.dropna(subset=['metric_value'])
        df.sort_values('metric_value', inplace=True, ascending=False)
        
        df_syn = df.loc[df.groupby("size")["metric_value"].idxmin()]
        df_red = df.loc[df.groupby("size")["metric_value"].idxmax()]

        df_syn["metric_name"].replace({"o_information": "synergy"}, inplace=True)
        df_red["metric_name"].replace({"o_information": "redundancy"}, inplace=True)

        return df, df_syn, df_red
    
    def get_combinations(
        self, minsize, maxsize=None, astype="jax", order=False
    ):
        """Get combinations of features.

        Parameters
        ----------
        minsize : int
            Minimum size of the multiplets
        maxsize : int | None
            Maximum size of the multiplets. If None, minsize is used.
        astype : {'jax', 'numpy', 'iterator'}
            Specify the output type. Use either 'jax' get the data as a jax
            array [default], 'numpy' for NumPy array or 'iterator'.
        order : bool, optional
            If True, return the order of each multiplet. Default is False.

        Returns
        -------
        combinations : array_like
            Combinations of features.
        """
        return combinations(
            self.n_features,
            minsize,
            maxsize=maxsize,
            astype=astype,
            order=order,
        )
    
    def _check_minmax(self, minsize, maxsize):
        """Define min / max size of the multiplets."""

        # check minsize / maxsize
        if not isinstance(minsize, int):
            minsize = 1
        if not isinstance(maxsize, int):
            maxsize = self.n_features
        assert isinstance(maxsize, int)
        assert isinstance(minsize, int)
        assert maxsize >= minsize
        maxsize = max(1, min(maxsize, self.n_features))
        minsize = max(1, minsize)

        self.minsize, self.maxsize = minsize, maxsize

        return minsize, maxsize
    
    def filter_multiplets(self, mults, order):
        """Filter multiplets.

        Parameters
        ----------
        mults : array_like
            Multiplets of shape (n_mult, maxsize)
        order : array_like
            Order of each multiplet of shape (n_mult,)

        Returns
        -------
        keep : array_like
            Boolean array of shape (n_mult,) indicating which multiplets to
            keep.
        """
        # order filtering
        if self._custom_mults is None:
            keep = jnp.ones((len(order),), dtype=bool)

            if self.minsize > 1:
                logger.info(f"    Selecting order >= {self.minsize}")
                keep = jnp.where(order >= self.minsize, keep, False)

            # task related filtering
            if self._task_related:
                logger.info("    Selecting task-related multiplets")
                keep_tr = (mults == self.n_features - 1).any(1)
                keep = jnp.logical_and(keep, keep_tr)

            return mults[keep, :]
        else:
            logger.info("    Selecting custom multiplets")
            _orders = [len(m) for m in self._custom_mults]
            mults = jnp.full((len(self._custom_mults), max(_orders)), -1)
            for n_m, m in enumerate(self._custom_mults):
                mults = mults.at[n_m, 0 : len(m)].set(m)

            return mults

def exhaustive_loop_zerolag(x, cluster_features = False, minsize=3, maxsize=20, verbose="debug"):
    #if x.shape[1] > x.shape[0]:
    #    x = np.transpose(x)
    
    if cluster_features:
        model = Oinfo(x, verbose=verbose)
    else:
        model = OinfoOriginal(x, verbose=verbose)
    #x = x.astype(float)
   
    return model.fit_exhaustive(minsize=minsize, maxsize=maxsize, method="gcmi")
    