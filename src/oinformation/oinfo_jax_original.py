from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from oinformation.hoi.metrics.base_hoi import HOIEstimator
from oinformation.hoi.core.entropies import get_entropy, prepare_for_entropy
from oinformation.hoi.utils.progressbar import get_pbar
from oinformation.hoi.core.combinatory import combinations, combinations_from_clustered_neurons
from copy import deepcopy
import pandas as pd 
import itertools


@partial(jax.jit, static_argnums=(2, 3))
def _oinfo_no_ent(inputs, index, entropy_3d=None, entropy_4d=None, entropy_4d_nansum = None):

    data, acc, multiplets, cluster_sizes, combs_per_multiplet = inputs
    msize = cluster_sizes[index]
    #jax.debug.print("data : {}", data.shape)
    l = 3
    #acc_test = jnp.mgrid[0:l, 0:l].sum(0) % l
    #acc_data = data.at[:, acc[index, :, :], :].get(mode='fill', fill_value=jnp.nan) #Select subset of variables from data
    subset_data = data.at[:, multiplets[index], :].get(mode='fill', fill_value=jnp.nan) #Select subset of variables from data
    #subset_data = data.at[jnp.arange(len(multiplets[index])),:].set(data[:, multiplets[index],:])
 
    acc_data = data.at[:, acc[index, :, :], :].get(mode='fill', fill_value=jnp.nan) #Select subset of variables from data
    #acc_data = data.at[:, acc[index, 0:1, :], :].get(mode='fill', fill_value=jnp.nan) #Select subset of variables from data
    #jax.debug.print("multiplets : {}", multiplets[index])
    #jax.debug.print("acc : {}", acc_data.shape)
    
    #jax.debug.print("subset_data1 : {}", subset_data)
    #jax.debug.print("acc data : {}", acc_data)
    
    #subset_data = subset_data.at[jnp.arange(len(multiplets[index])),:].set(subset_data[multiplets[index],:])
    #jax.debug.print("subset_data2 : {}", subset_data)
    #x = x.at[indices[0], :].set(jnp.full(x.shape[1], jnp.nan))

    
    #First set the first rows to the values we want
    
    #x = x.at[indices[0], :].set(jnp.full(x.shape[1], jnp.nan))

    #acc = acc[index, :, :] #
    
    #jax.debug.print("data[index] : {}", data[index])
    #jax.debug.print("multiplets[index] : {}", multiplets[index])
    #jax.debug.print("acc[index] : {}", multiplets[index])
    #jax.debug.print("acc_data : {}", acc_data.shape)
    #jax.debug.print("subset_data : {}", subset_data.shape)
    
    #jax.debug.print("subset_data : {}", subset_data[0, :, 0:5])
    #acc = data.at[:, acc[index], :].get(mode='fill', fill_value=jnp.nan)
    
    #jax.debug.print("acc : {}", acc)
    
    #jax.debug.print("acc[index] : {}", jax.lax.dynamic_slice(acc, (index, 0, msize), (2,2,2)))
    #acc = acc.at[index, :, :].get(mode='fill', fill_value=jnp.nan)[0:3, 0:5]
    
    #acc = reduce_matrix(acc[index])
    #nan_rows_mask = ~jnp.all(jnp.isnan(acc), axis=1)
    #nan_rows_mask = jnp.any(~jnp.isnan(acc), axis=1)
    #indices = jnp.where(nan_rows_mask)[0]
    
    #jax.debug.print("individual : {}", acc.shape)
    
    
    x_c = subset_data
    
    _, n_features, _ = x_c.shape
    h_xn = entropy_3d(x_c, msize, True)
    
    
    # compute \sum_{j=1}^{n} h(x_{j}
    h_xj_data = x_c[:, :, jnp.newaxis, :]
    h_xj_sum = jnp.nansum(entropy_4d(h_xj_data, 1, False))
    #jax.debug.print("h_xj_sum : {}", h_xj_sum)
    

    
    #jax.debug.print("combs_per_multiplet : {}", combs_per_multiplet[index])
    # compute \sum_{j=1}^{n} h(x_{-j}
    mask = jnp.arange(n_features) < combs_per_multiplet[index]
    #jax.debug.print("mask : {}", mask.reshape(-1, 1).shape)
    #jax.debug.print("acc_data : {}", acc_data.shape)
    h_xmj_sum = entropy_4d(acc_data, msize - 1, True) * mask.reshape(-1, 1)
    # Compare the array with the threshold
    #jax.debug.print("h_xmj_sum : {}", h_xmj_sum)
    h_xmj_sum = jnp.nansum(h_xmj_sum)
    
    # compute oinfo
    oinfo = (msize - 2) * h_xn + h_xj_sum - h_xmj_sum
    """
    jax.debug.print("joint : {}", h_xn)
    jax.debug.print("individual : {}", h_xj_sum)
    jax.debug.print("h_xmj_sum data : {}", acc_data)
    jax.debug.print("complement : {}", h_xmj_sum)
    jax.debug.print("oinfo : {}", oinfo)
    """
    return inputs, oinfo



class OinfoOriginal(HOIEstimator):
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

    def __init__(self, x, y=None, multiplets=None, verbose=None, clustered_labels_to_neurons = None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, clustered_labels_to_neurons = clustered_labels_to_neurons, verbose=verbose
        )

    
    def fit_exhaustive(self, minsize=2, maxsize=None, method="gcmi", batch_size = None, n_clusters = None, **kwargs):
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
        entropy = jax.vmap(get_entropy(method=method, **kwargs), in_axes=(0, None, None))
        #entropy_nansum = jax.vmap(get_entropy(method=method, **kwargs), in_axes=(0, None))
        
    
    
        # prepare output
        
        
        
        oinfo_no_ent = partial(
            _oinfo_no_ent,
            entropy_3d=entropy,
            entropy_4d=jax.vmap(entropy, in_axes=(1, None, None))
        )
       
        #h_idx = self.get_combinations(minsize, **kw_combs)

        #kw_combs = dict(astype="jax")
       
        
        offset = 0
        
      
        #hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        hoi = []
        total_clusters, total_multiplets = [], []
        n = self.n_features
        
        for multiplet_size in range(minsize, maxsize):
            
            
            #Generate cluster combinations (i.e. 1 2 3 - N) and multiplet combinations which maps each 1 .. N to the original size multiplets
            clusters, multiplets, multiplets_size, total_combs_per_multiplet = self.get_combinations(multiplet_size, 
                                            clustered_labels_to_neurons=self.clustered_labels_to_neurons, 
                                            maxsize = multiplet_size,
                                            n_features = n,
                                            #n_features=n_clusters
                                            )
            
            if len(multiplets_size) == 0:
                break
          
         
            #print("cluster %d max multiplets %d min muliplet %d" % (multiplet_size, max_multiplet_size, min_multiplet_size))
                
            #Loop over clusters and multiplets in batches
            for i in range(0, clusters.shape[0], batch_size):
            
                #batch_x = x[i:i+batch_size]
                batch_clusters = clusters[i:i+batch_size]
                batch_multiplets = multiplets[i:i+batch_size]
              
            
                batch_multiplets_size = multiplets_size[i:i+batch_size]
                batch_total_combs_per_multiplet = total_combs_per_multiplet[i:i+batch_size]
            
                padding = n + 1  # out-of-bound index
                n_cluster_combs = len(batch_clusters)
                
                #generate the combinations of each joint variable - 1 variable to exclude (complement)
                batch_accs_multiplets = np.full((n_cluster_combs, n, n), padding, dtype=int)
                batch_total_combs_per_multiplet = np.zeros(n_cluster_combs, dtype=int)
                for n_c, c in enumerate(batch_multiplets):
                    
                    c = c[c != self.n_features + 1]
                    multiplet_combinations = np.array(list(itertools.combinations(c.tolist(), len(c) - 1)))
                    
                    for j in range(multiplet_combinations.shape[0]):
                        comb = multiplet_combinations[j]
                        batch_accs_multiplets[n_c, j, 0:len(comb)] = comb
                    
                
                    batch_total_combs_per_multiplet[n_c] = j + 1
                    #batch_accs_multiplets[n_c, :, multiplet_combinations.shape[1]] = multiplet_combinations
                batch_total_combs_per_multiplet = jnp.array(batch_total_combs_per_multiplet)
                batch_accs_multiplets = jnp.array(batch_accs_multiplets)
                
                n_indices = jnp.arange(len(batch_multiplets))
                
              
                
                carry, _hoi = jax.lax.scan(oinfo_no_ent, (x, batch_accs_multiplets, batch_multiplets, batch_multiplets_size, batch_total_combs_per_multiplet), n_indices)
                
                for multiplet, cluster in zip(batch_multiplets, batch_clusters):
                    total_multiplets.append([multiplet[multiplet < self.n_features + 1] for i in multiplet][0])
                    total_clusters.append(cluster)

                # fill variables
                n_combs = batch_multiplets.shape[0]
                hoi.extend(_hoi)
                #hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)
            
                # updates
                offset += n_combs
                
                
                
                    
                
                #print(accs_multiplets.shape)
            """
            for c in clusters:
                multiplet_combinations = np.array(list(itertools.combinations(multiplet_clustered_neurons.tolist(), len(multiplet_clustered_neurons) - 1)))
                
                #print(c)
                #print(clustered_c)
                
                #print(multiplet_combinations)
                # Iterate over each row and set values at corresponding indices
            
                for i in range(multiplet_combinations.shape[0]):
                    comb = multiplet_combinations[i]
                    #accs_multiplets[n_c, i, 0:len(comb)] = comb
                
                total_combs_per_multiplet[n_c] = i + 1
            """
            #carry, _hoi = jax.lax.scan(oinfo_no_ent, (x, _acc_clustered, _h_idx_clustered, _h_idx_clustered_size, _total_combs_per_multiplet), n_indices)
            
       
        
        df = pd.DataFrame({
            'clusters': total_clusters,
            'multiplets': total_multiplets,
            'metric_value': np.array(hoi).flatten().tolist(),
            'size_clusters': [len(c) for c in total_clusters],
            'size_multiplets': [len(c) for c in total_multiplets],
            "metric_name": ["o_information" for c in total_multiplets]
        })
       
        df.sort_values('metric_value', inplace=True, ascending=False)
        print(df)
        df_syn = df.loc[df.groupby("size_clusters")["metric_value"].idxmin()]
        df_red = df.loc[df.groupby("size_clusters")["metric_value"].idxmax()]
     
        df_syn["metric_name"].replace({"o_information": "synergy"}, inplace=True)
        df_red["metric_name"].replace({"o_information": "redundancy"}, inplace=True) 
       
            
        return df, df_syn, df_red
        
        """
        h_idx, h_idx_clustered, h_idx_clustered_size, acc_clustered, total_combs_per_multiplet = self.get_combinations(minsize, clustered_labels_to_neurons=self.clustered_labels_to_neurons, **kw_combs)
        order = self.get_combinations(minsize, clustered_labels_to_neurons=self.clustered_labels_to_neurons, order=True, **kw_combs)
       
        print(h_idx_clustered.shape)
        exit(0)
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
        
    
       
        combs, combs_clustered = [], []
        for msize in pbar:
            pbar.set_description(desc="Oinfo (%i)" % msize, refresh=False)
            
            # combinations of features
            keep = order == msize
            
            # generate indices for accumulated entropies
            _h_idx = self._multiplets[keep, 0:msize]
            
            n_data, n_variables = h_idx_clustered.shape
            #_h_idx_clustered = h_idx_clustered[keep][1:2, :]
            #_h_idx_clustered_size = h_idx_clustered_size[keep][1:2]
            #_h_idx_clustered = h_idx_clustered[keep][105:106, :]
            #_h_idx_clustered_size = h_idx_clustered_size[keep][105:106]
            #acc_clustered = acc_clustered[keep][105:106, :, :]
            
            _h_idx_clustered_size = h_idx_clustered_size[keep]
            _h_idx_clustered = h_idx_clustered[keep]
            _acc_clustered = acc_clustered[keep]
            _total_combs_per_multiplet = total_combs_per_multiplet[keep]
            combs.extend(_h_idx)
        
            n_indices = jnp.arange(len(_h_idx_clustered))
            
            #x = x[:, h_idx_clustered, 0:5]
            
          
         
            
            carry, _hoi = jax.lax.scan(oinfo_no_ent, (x, _acc_clustered, _h_idx_clustered, _h_idx_clustered_size, _total_combs_per_multiplet), n_indices)
            
            clustered_comb = [row[row < self.n_features + 1] for row in _h_idx_clustered]
            
            combs_clustered.extend(clustered_comb)
            
             # fill variables
            n_combs = _h_idx_clustered.shape[0]
           
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)
          
            # updates
            offset += n_combs

        
        df = pd.DataFrame({
            'clusters': combs,
            'multiplets': combs_clustered,
            'metric_value': np.array(hoi).flatten().tolist(),
            'size_clusters': [len(c) for c in combs],
            'size_multiplets': [len(c) for c in combs_clustered],
            "metric_name": ["o_information" for c in combs]
        })
        
        df.sort_values('metric_value', inplace=True, ascending=False)
        
        df_syn = df.loc[df.groupby("size_clusters")["metric_value"].idxmin()]
        df_red = df.loc[df.groupby("size_clusters")["metric_value"].idxmax()]
      
        df_syn["metric_name"].replace({"o_information": "synergy"}, inplace=True)
        df_red["metric_name"].replace({"o_information": "redundancy"}, inplace=True)
        
        return df, df_syn, df_red
        """
    
    
    def get_combinations(
        self, minsize, n_features = None, maxsize=None, astype="jax", order=False, clustered_labels_to_neurons = None
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
        if not n_features:
            n_features = self.n_features
        if clustered_labels_to_neurons:
            
            return combinations_from_clustered_neurons(
                n_features,
                minsize,
                maxsize=maxsize,
                astype=astype,
                order=order,
                clustered_labels_to_neurons = clustered_labels_to_neurons,
            )
        else:
            return combinations(
                n_features,
                minsize,
                maxsize=maxsize,
                astype=astype,
                order=order
            )
            

