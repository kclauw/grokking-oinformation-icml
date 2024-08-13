from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from oinformation.hoi.metrics.base_hoi import HOIEstimator
from oinformation.hoi.core.entropies import get_entropy, prepare_for_entropy
from oinformation.hoi.utils.progressbar import get_pbar
from oinformation.hoi.core.combinatory import combinations
import pandas as pd 

@partial(jax.jit, static_argnums=(2, 3))
def _oinfo_no_ent(inputs, index, entropy_3d=None, entropy_4d=None):
    data, acc = inputs
    msize = len(index)

    # tuple selection
    x_c = data[:, index, :]

    #jax.debug.print("data : {}", x_c[:, :, 0:5])
    
    # compute h(x^{n})
    h_xn = entropy_3d(x_c)
    #jax.debug.print("joint : {}", h_xn)

    # compute \sum_{j=1}^{n} h(x_{j}
    h_xj_sum = entropy_4d(x_c[:, :, jnp.newaxis, :]).sum(0)
    #jax.debug.print("individual : {}", h_xj_sum)

    # compute \sum_{j=1}^{n} h(x_{-j}
    #jax.debug.print("h_xmj_sum data : {}", x_c[:, acc, 0:5])
    h_xmj_sum = entropy_4d(x_c[:, acc, :]).sum(0)
    #jax.debug.print("complement : {}", h_xmj_sum)

    # compute oinfo
    oinfo = (msize - 2) * h_xn + h_xj_sum - h_xmj_sum
    #jax.debug.print("oinfo : {}", oinfo)
    return inputs, oinfo


class Oinfo(HOIEstimator):
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

    def __init__(self, x, y=None, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )

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
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)
       
        # get entropy function
        entropy = jax.vmap(get_entropy(method=method, **kwargs))
        
        oinfo_no_ent = partial(
            _oinfo_no_ent,
            entropy_3d=entropy,
            entropy_4d=jax.vmap(entropy, in_axes=1),
        )

        # prepare output
        kw_combs = dict(maxsize=maxsize, astype="jax")
        #h_idx = self.get_combinations(minsize, **kw_combs)[-1:]
        #order = self.get_combinations(minsize, order=True, **kw_combs)[-1:]
       
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
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)

        combs = []
        for msize in pbar:
            pbar.set_description(desc="Oinfo (%i)" % msize, refresh=False)

            # combinations of features
            
            keep = order == msize
            _h_idx = self._multiplets[keep, 0:msize]
            
            #_h_idx = jnp.array([jnp.array([7, 14, 19])])
            # generate indices for accumulated entropies
            acc = jnp.mgrid[0:msize, 0:msize].sum(0) % msize
            acc = acc[:, 1:]
            
            #acc = acc[:1]
            #_h_idx = jnp.array([[6, 15, 19]])
            #_h_idx_clustered_size = jnp.array([[3]])
            
            # compute oinfo
            multiplets, _hoi = jax.lax.scan(oinfo_no_ent, (x, acc), _h_idx)
            
            
           
            combs.extend(_h_idx)
            
            # fill variables
            n_combs = _h_idx.shape[0]
           
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)
            
            
            # updates
            offset += n_combs

        df = pd.DataFrame({
            'multiplet': combs,
            'metric_value': np.array(hoi).flatten().tolist(),
            'size': [len(c) for c in combs],
            "metric_name": ["o_information" for c in combs]
        })
        
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
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.hoi.utils import landscape
    from matplotlib.colors import LogNorm

    plt.style.use("ggplot")

    path = "/home/god/Projects/2_o-information/hoi/test_features.npy"
    x = np.transpose(np.load(path, allow_pickle=True)) #(n_samples, n_features) 
    x = x.astype(float)
    
    #, multiplets=[(0, 1, 2), (4, 5, 7, 8)]
    model = Oinfo(x, verbose="debug")
    hoi = model.fit(minsize=3, maxsize=16, method="gcmi")

   
    print(hoi.shape)
    print(model.order.shape)
    print(model.multiplets.shape)

    lscp = landscape(hoi.squeeze(), model.order, output="xarray")
    lscp.plot(x="order", y="bins", cmap="jet", norm=LogNorm())
    plt.axvline(model.undersampling, linestyle="--", color="k")
    plt.grid(True)
    plt.show()
