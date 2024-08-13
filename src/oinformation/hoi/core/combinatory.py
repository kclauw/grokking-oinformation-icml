import jax.numpy as jnp
import numpy as np
import itertools
from math import comb as ccomb



def _combinations(n, k, order):
    for c in itertools.combinations(range(n), k):
        # convert to list
        c = list(c)

        # deal with order
        if order:
            c = len(c)

        yield c



def combinations(
    n, minsize, maxsize=None, astype="iterator", order=False, fill_value=-1, clustered_neurons = None
):
    """Get combinations.

    Parameters
    ----------
    n : int
        Represents the total number of elements in the set
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
    combs : jnp.array
        An array of shape (n_combinations, k) representing all possible
        combinations of k elements.
    """
    # ________________________________ ITERATOR _______________________________
    if not isinstance(maxsize, int):
        maxsize = minsize
    assert maxsize >= minsize
    
    iterators = []
    for msize in range(minsize, maxsize + 1):
        iterators.append(_combinations(n, msize, order))
    iterators = itertools.chain(*tuple(iterators))

    if astype == "iterator":
        return iterators

    # _________________________________ ARRAYS ________________________________
    if order:
        combs = np.asarray([c for c in iterators]).astype(int)
    else:
        # get the number of combinations
        
        n_mults = sum([ccomb(n, c) for c in range(minsize, maxsize + 1)])

        # prepare output
        combs = np.full((n_mults, maxsize), fill_value, dtype=int)
    
        # fill the array
        for n_c, c in enumerate(iterators):
            combs[n_c, 0 : len(c)] = c

    """    
    # jax conversion (f required)
    if astype == "jax":
        combs = jnp.asarray(combs)
    """
    return combs


def pad_and_concatenate(arrays, max_length):
    padded_arrays = []
    for array in arrays:
        padding_length = max_length - len(array)
        padded_array = jnp.pad(array, (0, padding_length), mode='constant', constant_values=-1)
 
        padded_arrays.append(padded_array)
    return jnp.array(padded_arrays)

from scipy.linalg import pascal

def stretch(a, k):
    l = a.sum()+len(a)*(-k)
    out = np.full(l, -1, dtype=int)
    out[0] = a[0]-1
    idx = (a-k).cumsum()[:-1]
    out[idx] = a[1:]-1-k
    return out.cumsum()

def numpy_combinations(data):
    #n, k = data #benchmark version
    n, k = data
    x = np.array([n])
    P = pascal(n).astype(int)
    C = []
    for b in range(k-1,-1,-1):
        x = stretch(x, b)
        r = P[b][x - b]
        C.append(np.repeat(x, r))
    return n - 1 - np.array(C).T

def generate_combinations_matrix(lst):
    n = len(lst)
    combinations_lst = list(combinations(lst.tolist(), n-1))
    return np.array(combinations_lst)

def combinations_from_clustered_neurons(
    n, minsize, clustered_labels_to_neurons, maxsize=None, astype="iterator", order=False, fill_value=-1
):
    """Get combinations.

    Parameters
    ----------
    n : int
        Represents the total number of elements in the set
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
    combs : jnp.array
        An array of shape (n_combinations, k) representing all possible
        combinations of k elements.
    """
    # ________________________________ ITERATOR _______________________________

    n_clusters = len(clustered_labels_to_neurons)
  
    if not isinstance(maxsize, int):
        maxsize = minsize
    assert maxsize >= minsize
    
    iterators = []
    
    for msize in range(minsize, maxsize + 1):
        iterators.append(_combinations(n_clusters, msize, order)) #We operate on cluster size (not the original features size)
  
    iterators = itertools.chain(*tuple(iterators))

    if astype == "iterator":
        return iterators

    # _________________________________ ARRAYS ________________________________
    if order:
        combs = np.asarray([c for c in iterators]).astype(int)
        return combs
    else:
        
       
        # get the number of combinations
        n_cluster_combs = sum([ccomb(n_clusters, c) for c in range(minsize, maxsize + 1)])
        #print("n %d total combs %d" % (minsize, n_cluster_combs))
        # prepare output
        
        clusters = np.full((n_cluster_combs, maxsize), fill_value, dtype=int)
        
        padding = n + 1  # out-of-bound index
        multiplets = np.full((n_cluster_combs, n), padding, dtype=int)
        
        #accs_multiplets = np.full((n_cluster_combs, n, n), padding, dtype=int)
       
        multiplets_size = np.zeros(n_cluster_combs, dtype=int)
        total_combs_per_multiplet = np.zeros(n_cluster_combs, dtype=int)
        
        
        for n_c, c in enumerate(iterators):
            clusters[n_c, 0 : len(c)] = c
            
            
            multiplet_clustered_neurons = np.concatenate([clustered_labels_to_neurons[i] for i in c])
            size_cluster = len(multiplet_clustered_neurons)
            multiplets_size[n_c] = size_cluster
  
            #test = generate_combinations_matrix(clustered_c)
        
            multiplets[n_c, 0:len(multiplet_clustered_neurons)] = multiplet_clustered_neurons
           
    
    
    return clusters, multiplets, multiplets_size, total_combs_per_multiplet
    

if __name__ == "__main__":
    print(combinations(10, minsize=2, maxsize=None, astype="jax", order=False))

    # print(np.array(list(itertools.combinations(np.arange(10), 3))).shape)
