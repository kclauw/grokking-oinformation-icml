from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import get_entropy, prepare_for_entropy
from hoi.utils.progressbar import get_pbar
import pandas as pd
from tqdm import tqdm

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
        The feature of shape (n_samples,) for estimating task-related O-info
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1, 2), (2, 7, 8, 9)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Rosas et al., 2019 :cite:`rosas2019oinfo`
    """

    __name__ = "O-Information"
    _encoding = False
    _positive = "redundancy"
    _negative = "synergy"
    _symmetric = True

    def __init__(self, x, y=None, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )

    def fit(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
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

        Returns
        -------
        hoi : array_like
            The NumPy array containing values of higher-rder interactions of
            shape (n_multiplets, n_variables)
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

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _________________________________ HOI _______________________________
        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        combs = []
        for msize in pbar:
            pbar.set_description(desc="Oinfo (%i)" % msize, refresh=False)

            # get the number of features when considering y
            n_feat_xy = msize + self._n_features_y

            # combinations of features
            _h_idx = h_idx[order == msize, 0:n_feat_xy]

            # indices for X_{-j} and skip first column
            acc = jnp.mgrid[0:n_feat_xy, 0:n_feat_xy].sum(0) % n_feat_xy
            acc = acc[:, 1:]
            
           
            # compute oinfo
            _, _hoi = jax.lax.scan(oinfo_no_ent, (x, acc), _h_idx)
        
            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

            combs.extend(_h_idx)

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


class GreedyOinfo(HOIEstimator):
    __name__ = "O-Information"
    _encoding = False
    _positive = "redundancy"
    _negative = "synergy"
    _symmetric = True

    def __init__(self, x, y=None, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )
        
    def exhaustive_fit(self, minsize=2, maxsize_greedy=None, maxsize_exhaustive=None, method="gcmi", **kwargs):
        # ________________________________ I/O ________________________________
        # check min and max sizes
        minsize, maxsize_exhaustive = self._check_minmax(minsize, maxsize_exhaustive)
       
        # prepare the x for computing entropy
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)

        # get entropy function
        entropy = jax.vmap(get_entropy(method=method, **kwargs))
        oinfo_no_ent = partial(
            _oinfo_no_ent,
            entropy_3d=entropy,
            entropy_4d=jax.vmap(entropy, in_axes=1),
        )

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize_exhaustive)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _________________________________ HOI _______________________________
        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        combs = []
        for msize in pbar:
            pbar.set_description(desc="Oinfo (%i)" % msize, refresh=False)

            # get the number of features when considering y
            n_feat_xy = msize + self._n_features_y

            # combinations of features
            _h_idx = h_idx[order == msize, 0:n_feat_xy]

            # indices for X_{-j} and skip first column
            acc = jnp.mgrid[0:n_feat_xy, 0:n_feat_xy].sum(0) % n_feat_xy
            acc = acc[:, 1:]
            
        
            # compute oinfo
            _, _hoi = jax.lax.scan(oinfo_no_ent, (x, acc), _h_idx)
        
            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

            combs.extend(_h_idx.tolist())

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
        
        return df_syn, df_red

    def fit(self, minsize=2, maxsize_greedy=None, maxsize_exhaustive=None, method="gcmi", **kwargs):
        
        
        # prepare the x for computing entropy
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)
        
        # get entropy function
        entropy = jax.vmap(get_entropy(method=method, **kwargs))
        oinfo_no_ent = partial(
            _oinfo_no_ent,
            entropy_3d=entropy,
            entropy_4d=jax.vmap(entropy, in_axes=1),
        )
        
        #df_syn = None 
        #df_red = None
        df_syn, df_red = self.exhaustive_fit(minsize=2, maxsize_greedy=None, maxsize_exhaustive=maxsize_exhaustive, method="gcmi", **kwargs)
        
        
        df_syn = self.greedy_search_synergy(df_syn, x, maxsize_greedy = maxsize_greedy, oinfo_no_ent = oinfo_no_ent)
        df_red = self.greedy_search_redundancy(df_red, x, maxsize_greedy = maxsize_greedy, oinfo_no_ent = oinfo_no_ent)
        
        return df_syn, df_red

    def greedy_search_synergy(self, df_syn, x, maxsize_greedy, oinfo_no_ent):
        
        _, nvars, nsamp = x.shape
        
      
        best_multiplet, best_synergy, syn_size, _ = df_syn.loc[df_syn['metric_value'].idxmin()].to_numpy()
        results_synergy = list(df_syn.to_numpy())
        nd = len(best_multiplet)
        others = np.setdiff1d(np.arange(0, nvars), np.array(best_multiplet))
        nvar = len(others)
        dbest = best_multiplet
       
 
        for nd in tqdm(range(nd, nvars)):
           
            if nd == maxsize_greedy:
                break

            o1 = np.zeros(nvar)

            multiplet_size = nd + 1
            acc = jnp.mgrid[0:multiplet_size, 0:multiplet_size].sum(0) % multiplet_size
            acc = acc[:, 1:]
            
          
            multiplets = np.stack([np.concatenate([[others[k]], dbest]) for k in range(nvar)])
            
            
            _, o1 = jax.lax.scan(oinfo_no_ent, (x, acc), multiplets)
          
        
            o_syn = float(np.min(o1))
            o_syn_i = np.argmin(o1)
        
            dbest = np.concatenate((dbest, [others[o_syn_i]]))
        
            print([len(dbest), o_syn])
            results_synergy.append([dbest, o_syn, len(dbest), "synergy"])
            #print(results_synergy)
            
            del o1
        
            others = np.setdiff1d(others, others[o_syn_i])
            nvar -= 1
            
        
        
        df_syn = pd.DataFrame(results_synergy, columns=list(df_syn.columns),)
      
        del results_synergy
        return df_syn
    
    def greedy_search_redundancy(self, df_red, x, maxsize_greedy, oinfo_no_ent):
        best_multiplet, best_redundancy, red_size, _ = df_red.loc[df_red['metric_value'].idxmax()].to_numpy()
       
        _, nvars, nsamp = x.shape
        
        results_redundancy = list(df_red.to_numpy())

      
        nd = len(best_multiplet)
        others = np.setdiff1d(np.arange(0, nvars), best_multiplet)
        nvar = len(others)
        dbest = best_multiplet
        
        for nd in tqdm(range(nd, nvars)):
            if nd == maxsize_greedy:
                break

            o1 = np.zeros(nvar)

            multiplet_size = nd + 1
            acc = jnp.mgrid[0:multiplet_size, 0:multiplet_size].sum(0) % multiplet_size
            acc = acc[:, 1:]
            
            multiplets = np.stack([np.concatenate([[others[k]], dbest]) for k in range(nvar)])
            
            _, o1 = jax.lax.scan(oinfo_no_ent, (x, acc), multiplets)
          
        
            o_red = np.max(o1)
            o_red_i = np.argmax(o1)
            dbest = np.concatenate((dbest, [others[o_red_i]]))
            del o1
            results_redundancy.append([dbest, o_red, len(dbest), "redundancy"])
        
            others = np.setdiff1d(others, others[o_red_i])
            nvar -= 1
          
            
        df_red = pd.DataFrame(results_redundancy, columns=list(df_red.columns),)
        del results_redundancy
        return df_red
