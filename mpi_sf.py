"""
Calculates the second order structure function:
$$
S_{D}(\vec{\ell}) = \langle | s_{D}(\vec{x}) - s_{D}(\vec{x}+\vec{\ell}) |^2 \rangle
$$
for arbitrary dimension $D$.

NOTE: $\vec{\ell}$ will be only the *positive* lags.

SINGLE-PROCESSED USAGE:
>>> import mpi_sf
>>> ar = ... # load data
>>> N, D = np.min(ar.shape), ar.ndim
>>> ell, sf2 = mpi_sf.mpi_sf(ar)
>>> ell = ell.reshape((N//2 for _ in range(D)))
>>> sf2 = sf2.reshape((N//2 for _ in range(D)))
"""


import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    print('NO MPI')
    mpi4py = None
    comm = None

def process_lags(ar, lags):
    """process_lags(ar, lags)

    Calculate the 2nd order structure function for the given lags    
    """
    out = np.zeros(len(lags))
    N = np.shape(ar)
    D = len(N)
    for i, l in enumerate(lags):
        _D = len(l)
        s1, s2 = [slice(0,N[d]) for d in range(D)], [slice(0,N[d]) for d in range(D)]
        for d in range(1,_D+1):
            s1[d-1] = slice(l[_D-d],N[D-d])
            s2[d-1] = slice(0,N[D-d]-l[_D-d])
        out[i] = np.nanmean(np.abs(ar[tuple(s1)] - ar[tuple(s2)])**2)
    return out

def get_all_lagvecs(shape):
    """get_all_lagvecs(shape)
    
    Get all the lagvectors as a list required to compute the positive lag quadrant of the SF
    """
    return np.transpose(np.indices(shape).reshape((-1, np.prod(shape))))

def mpi_sf(ar, max_lag=None, longitudinal=False):
    """mpi_sf(ar, max_lag)

    Calculates the second order SF by naively distributing the lags amongst the available
        processors. Returns the lags, and SF computed by the current processor
    You will need to save the outputs, and recombine into one large SF later on.

    Args:
        ar (np.ndarray): Numpy ND array structure to compute the SF for
        max_lag (int): Maximum lag to go to
        longitudinal (bool): If true, compute the longitudinal SF only
    Returns:
        l (np.ndarray): Lags corresponding to the SF for the current processor
        sf (np.ndarray): SF computed by the current processor
    """
    if max_lag is None:
        max_lag = np.min(np.shape(ar))//2
    D = ar.ndim
    if longitudinal:
        D = 1
    max_lag = tuple([max_lag for _ in range(D)])
    p_func = process_lags
    # Not running in mpi, so just compute the statfunc
    if (comm is None) or comm.Get_size() == 1:
        lags = get_all_lagvecs(max_lag)
        return lags, p_func(ar, lags)
    # Compute the structure function without any communication whatsoever
    rank = comm.Get_rank()
    size = comm.Get_size()
    lags = get_all_lagvecs(max_lag)[rank::size]
    print('starting %s:%s' % (comm.Get_rank(), len(lags)))
    # Compute lag function
    statfunc = p_func(ar, lags)
    return lags, statfunc
