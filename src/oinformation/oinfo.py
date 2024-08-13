from src.oinformation.gcmi import copnorm, ent_g, mi_gg
from joblib import Parallel, delayed
import tqdm
import numpy as np
import itertools
from numpy.linalg import LinAlgError
import pandas as pd

def make_matrix_positive_definite(matrix, noise_factor=1e-6):
    # Create a copy of the diagonal and add noise
    diagonal = np.diag(matrix).copy()
    diagonal += noise_factor * np.random.rand(diagonal.shape[0])
    # Create the new matrix with modified diagonal
    matrix = matrix.copy()
    np.fill_diagonal(matrix, diagonal)
    return matrix

# @jit("f4(f4[:, :])")
def nb_ent_g(x, noise_factor=1e-6):
    """Numba implementation of the entropy of a Gaussian variable in bits.
    """
    nvarx, ntrl = x.shape

    # covariance
    c = np.dot(x, x.T) / float(ntrl - 1)
    #print(np.linalg.eigvalsh(c))
    try:
        chc = np.linalg.cholesky(c)
    except LinAlgError as e:
        chc = make_matrix_positive_definite(c)
    
    # entropy in nats
    hx = np.sum(np.log(np.diag(chc))) + 0.5 * nvarx * (
        np.log(2 * np.pi) + 1.0)
    return hx

def _o_info(x, comb, return_comb=True):
    nvars, _ = x.shape

    # (n - 2) * H(X^n)
    o = (nvars - 2) * nb_ent_g(x)
    
   
    for j in range(nvars):
        #print(nb_ent_g(x[[j], :]) - nb_ent_g(np.delete(x, j, axis=0)))
        # sum_{j=1...n}( H(X_{j}) - H(X_{-j}^n) )
        o += nb_ent_g(x[[j], :]) - nb_ent_g(np.delete(x, j, axis=0))
        
 
    if return_comb:
        return o, comb
    else:
        return o
    
def combinations(n, k, groups=None):
    assert isinstance(n, int)
    if isinstance(k, int): k = [k]
    assert isinstance(k, (list, tuple, np.ndarray))

    iterable = np.arange(n)

    combs = []
    for i in k:
        combs += [itertools.combinations(iterable, i) for i in k]
    comb = itertools.chain(*tuple(combs))

    for i in comb:
        if isinstance(groups, (list, tuple)):
            if all([k in i for k in groups]):
                yield i
        else:
            yield i
            
def exhaustive_loop_zerolag(ts, clustered_neurons = None, y = None, maxsize=5, n_best=10, groups=None, n_jobs=4,
                            n_boots=None, alpha=0.05, debug = True):
    """Simple implementation of the Oinfo.

    Parameters
    ----------
    ts : array_like
        Time-series of shape (n_variables, n_samples) (e.g (n_roi, n_trials))
    """
    # copnorm and demean the data
    
    x = copnorm(ts)
    x = (x - x.mean(axis=1)[:, np.newaxis]).astype(np.float32)
    nvars, nsamp = x.shape
    
    if clustered_neurons:
        maxsize = len(clustered_neurons.keys())
    
    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = nvars
    maxsize = max(1, maxsize)
    
    # get all possible combinations and size
    iterable = np.arange(3, maxsize + 1)
    n_mults = sum(1 for _ in combinations(maxsize, iterable, groups=groups))
   
    all_comb = combinations(maxsize, iterable, groups=groups)
    
    
    
    if n_jobs == -1:
        n_jobs = None
        
  
    if debug:
        # progressbar definition
        pbar = tqdm.trange(n_mults, mininterval=3)
        pbar.set_description(f'Computation of the Oinfo (n_multiplets={n_mults})')
        
        outs = Parallel(n_jobs=n_jobs)(delayed(_o_info)(
                x[np.concatenate([clustered_neurons[c] for c in comb]), :], comb) for comb, _ in zip(all_comb, pbar))
    else:
        outs = Parallel(n_jobs=n_jobs)(delayed(_o_info)(
            x[np.concatenate([clustered_neurons[c] for c in comb]), :], comb) for comb in all_comb)
    
    # unwrap outputs
    oinfo, combs = zip(*outs)

    # dataframe conversion

    df = pd.DataFrame({
        'multiplet': combs,
        'metric_value': oinfo,
        'size': [len(c) for c in combs],
        "metric_name": ["o_information" for c in combs]
    })
   
    df.sort_values('metric_value', inplace=True, ascending=False)
    df_syn = df.loc[df.groupby("size")["metric_value"].idxmin()]
    df_red = df.loc[df.groupby("size")["metric_value"].idxmax()]
    
    
    df_syn["metric_name"].replace({"o_information": "synergy"}, inplace=True)
    df_red["metric_name"].replace({"o_information": "redundancy"}, inplace=True)

    return df, df_syn, df_red