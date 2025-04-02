"""
Provides the functions to calculate the equivalent spectrum (Uncorrected, as well as Debiased).
Also provides additional helper functions for the required calculations.

USAGE:
>>> import equiv_spectrum
>>> ell, sf2 = ... # Calculate the angle-averaged second order structure function
>>> ke, BfekS, fekS = equiv_spectrum.equiv_spectrum(ell, sf2, grid_dims, [L for _ in range(D)])
"""

import numpy as np

import sympy as sp


##################################
##    INTERPOLATION ROUTINES    ##
##################################
from scipy.interpolate import interp1d
def log_log_interpolate(x, y, x_interp):
    """log_log_interpolate(x, y, x_interp)

    Interpolates in log-log-space
    """
    f_int = interp1d(np.log10(x), np.log10(y), kind='linear', fill_value='extrapolate')
    y_new = f_int(np.log10(x_interp))
    return 10**y_new

def logx_interpolate(x, y, x_interp):
    """logx_interpolate(x, y, x_interp)

    Interpolates in log(x)-space
    """
    f_int = interp1d(np.log10(x), y, kind='linear', fill_value='extrapolate')
    y_new = f_int(np.log10(x_interp))
    return y_new

##################################
##       BINNING ROUTINES       ##
##################################
from scipy.stats import binned_statistic

def per_spectrum(ar1, phys_dims=None):
    """per_spectrum(ar, phys_dims)

    Computes the spectrum of the given array via Fourier transform.

    `$P_{n}(k) = (dk/2pi)**n |dx**n FFTn{ar}|^2$`\n
    `$k_m = 2 pi m / L$`\n
    `$m \in [-N/2, N/2]$`

    This is the classical/Schuster periodogram

    Args:
        ar1 (np.ndarray): Array to compute the spectrum of
        lenx,leny,lenz (float): System size in x,y,z directions
    Returns:
        kvec (tuple): Wavenumber arrays
        fek (np.ndarray): Spectrum of the array
    """
    if phys_dims is None:
        phys_dims = [2. * np.pi for _ in range(len(ar1.shape))]
    kvec = []
    dx = []
    dk = []
    for i, N in enumerate(ar1.shape):
        dx.append(phys_dims[i]/N)
        dk.append(2.*np.pi/(N*dx[i]))
        kvec.append(np.fft.fftshift(np.fft.fftfreq(N))*2.*np.pi/dx[i])
    far1 = np.prod(dx)*np.fft.fftshift(np.fft.fftn(ar1))
    fek = np.abs(far1)**2
    # Normalize the power spectrum
    ## NOTE: With BOTH the normalizations, we obtain the expected DFT formula.
    fek = fek / np.sum(np.ones_like(ar1) * np.prod(dx))
    ## NOTE: This normalization is not going to be applied.
    ##      Apply this during the integration.
    #fek = fek * np.prod(dk) / (2.*np.pi)**ar1.ndim
    return tuple(kvec), fek

def integrate_spectrum(kvec, mspec, phys_dims):
    """integrate_spectrum(kvec, mspec, phys_dims)

    Calculates the angle-integrated spectrum

    Args:
        kvec (tuple): Tuple of wavenumbers associated with each dimension
        mspec (np.ndarray): Modal spectrum
        phys_dims (tuple): Physical size of each dimension
    Returns:
        ko (np.ndarray): Wavenumber magnitude of the spectrum
        feko (np.ndarray): Angle-integrated spectrum   
    """
    norm_bin_size = True
    km = np.meshgrid(*kvec, indexing='xy')
    kmesh = np.linalg.norm(km, axis=0)
    min_k = 2.*np.pi / phys_dims[0]
    bins, ispec, istd = bin_data(kmesh, mspec, bin_func=np.nansum,
        cut_excess=True, nan_small=False, min_bin=min_k, bin_loc='true_center',
        norm_bin_size=norm_bin_size, log_space=False, num_bins=None, ignore_nan=False,
        max_bin=None, max_half_bin_width=None)
    return bins, ispec

def bin_data(nar, ar, bin_func=np.nanmean,
             cut_excess=False, nan_small=False, min_bin=None,
             max_bin=None, bin_loc='center', norm_bin_size=False, log_space=False,
             num_bins=None, ignore_nan=False, max_half_bin_width=None):
    """bin_data(...)

    Bin `ar` with the domain `nar` using the `bin_func`.
    This reduces an ND array down to 1 dimension through the `bin_func`.

    Args:
        nar (np.ndarray): The position array to bin magnitudes of
        ar (np.ndarray): The ND array to bin down to 1D
        bin_func (func): The function used to bin `ar`
        cut_excess (bool): If true, remove binned lags greater than one of the basis directions
        nan_small (bool): If true, set the binned function to nan for when the number of points is small
        min_bin (float): Smallest bin value, if None, then automatically pick
        max_bin (float): Largest bin value.
            if None, then get the largest of `nar`
            if `basis`, then choose the largest in the basis
        bin_loc (str):
            center: Places the bin in the center of the bin range.
            true_center: Places the bins in the center of the available data,
                this fixes problems close to 0.
            left: Places the bins at the left bin-edge.
        norm_bin_size (bool): If true, divide the binned functions by the size of their respective bins
        log_space (bool): If true, use log-spacings
        num_bins (float): Number of bins
        ignore_nan (bool): Remove nan values
        max_half_bin_width (None/float): Default None. If set, this is the maximum (half) bin width allowed
    Returns:
        bins (np.array): The bins of the computed statistics
        ar1D (np.array): Mean statistic of the binning from ND to 1D on the bins
        width (np.array): Bin widths
    """
    nar = (nar.copy()).round(decimals=10)
    # Find the basis of the position array
    pos = np.where(nar == 0)
    if len(pos[0]) == 0:
        basis_index = 0
        narbasis = nar
        nn = nar
    else:
        basis_index = [pos[i][0] for i in range(len(pos))]
        basis_index[0] = Ellipsis
        narbasis = nar[tuple(basis_index)]
        nn = narbasis[pos[0][0]:]
    # Calculate the bin space
    # Assume that the array is evenly spaced (this is an assumption made with everything)
    if min_bin is None:
        # If no minimum bin specified, then automatically choose one
        min_bin = np.nanmin(nn)
    if max_bin is None:
        max_bin = np.nanmax(nar)
    if isinstance(max_bin, str) and max_bin == 'basis':
        max_bin = np.nanmax(narbasis)
    if num_bins is None:
        binsize = np.diff(nn)[0]
        nbins = int(np.round(max_bin/binsize) + 1)
    else:
        nbins = num_bins
    bins, bin_widths, be = get_bins(min_bin, max_bin, nbins, max_half_bin_width, log_space)
    # Set bad, the regions outside the binning domain because scipy includes them into the binnings
    car = ar.copy()
    car[nar < min_bin] = np.nan
    car[nar > max_bin] = np.nan
    # Compute the binnings
    ar1d, bin_edges, _ = binned_statistic(nar.ravel(), car.ravel(), bins=be, statistic=bin_func)
    if bin_loc == 'center':
        # Set the bins to the mid point of the bin edges
        bins1d = (bin_edges[1:] + bin_edges[:-1])/2.
    elif bin_loc == 'true_center':
        # Set the bins to the mid point of the actual `nar` data
        new_bins = []
        for i in range(len(bin_edges)-1):
            bmin, bmax = bin_edges[i].round(decimals=10), bin_edges[i+1].round(decimals=10)
            mask = np.where(np.logical_and(nar >= bmin, nar < bmax))
            if nar[mask].size == 0:
                new_bins.append((bmin + bmax)/2.)
            else:
                new_bins.append((np.nanmin(nar[mask]) + np.nanmax(nar[mask]))/2.)
        bins1d = np.array(new_bins)
    else:
        # Set the bins to the start of the bin edges
        bins1d = bin_edges[:-1]
    if max_half_bin_width is not None:
        # We should remove the bad bin edges (for the bins we don't want)
        mask = np.isin(bins1d, bins)
        ar1d = ar1d[mask]
        bins1d = bins1d[mask]
    # Divide the functions by their bin sizes
    # NOTE: The bin widths are actually half the bin widths
    if max_half_bin_width:
        # We have already calculate this
        width = 2.*bin_widths
    else:
        # Otherwise, calculate from binned_statistic bin_edges
        lower_bins, upper_bins = bin_edges[:-1], bin_edges[1:]
        width = (upper_bins - lower_bins)
        if max_half_bin_width is not None:
            width = np.minimum(width, 2.*max_half_bin_width)
    if norm_bin_size:
        ar1d = ar1d / width
    if nan_small:
        # Compute the counts, so we can ignore bad statistics
        cts, _, _ = binned_statistic(nar.ravel(), car.ravel(), bins=be, statistic='count')
        mask = cts <= 1
        ar1d[mask] = np.nan
    if cut_excess:
        # Cut off lags above the basis directions
        ar1d = ar1d[bins1d <= nn[-1]]
        width = width[bins1d <= nn[-1]]
        bins1d = bins1d[bins1d <= nn[-1]]
    if ignore_nan:
        # Remove nan values
        mask = np.isfinite(ar1d)
        ar1d = ar1d[mask]
        width = width[mask]
        bins1d = bins1d[mask]
    return bins1d, ar1d, width

def get_bins(min_bin, max_bin, nbins, max_half_bin_width=None, log_space=False):
    """get_bins(min_bin, max_bin, nbins, max_half_bin_width)

    Generates the bin centers, widths and edges for use with the binning functions

    Args:
        min_bin (float): Minimum bin number
        max_bin (float): Maximum bin number
        nbins (int): The number of bins
        max_half_bin_width (float): The maximum (half) width of the bins
        log_space (bool): If true, use log separations for the bins
    Returns:
        bins (np.ndarray): List of bin centers
        half_bin_width (np.ndarray): List of (half) bin widths for each bin center
        bin_edges (np.ndarray): Unique bin edges
    """
    if log_space:
        bin_range = np.exp(np.linspace(np.log(min_bin), np.log(max_bin), nbins+1))
    else:
        bin_range = np.linspace(min_bin, max_bin, nbins+1)
    valid_bins = np.unique(bin_range.round(decimals=10))
    lower_bins, upper_bins = valid_bins[:-1], valid_bins[1:]
    bin_width = (upper_bins - lower_bins) / 2.
    if max_half_bin_width is not None:
        bin_width = np.minimum(bin_width, max_half_bin_width)
    bins = upper_bins - bin_width
    bin_edges = np.unique(np.concatenate((bins-bin_width, bins+bin_width)).round(decimals=10))
    return bins, bin_width, bin_edges

##################################
## EQUIVALENT SPECTRUM ROUTINES ##
##################################
def get_powerlaw(k, fek, num_bins=16, log_space=False):
    """get_powerlaw(k, fek, num_bins, log_space)
    
    Calculates the local power law slope

    Args:
        k (np.ndarray): Wavenumbers
        fek (np.ndarray): Spectrum
        num_bins (int/None): Number of bins for binning and interpolation
        log_space (bool): If true, bin equally in log space
    Returns:
        est_alpha (np.narray): Estimate of the local power law slope at each k
    """
    est_alpha = np.gradient(np.log(fek), np.log(k))
    if num_bins is None:
        return est_alpha
    b_k, b_est_alpha, _ = bin_data(
        k, est_alpha, bin_func=np.nanmean, log_space=log_space,
        bin_loc='center', num_bins=num_bins, min_bin=np.nanmin(k), max_bin=np.nanmax(k),
        ignore_nan=True)
    est_alpha = logx_interpolate(b_k, b_est_alpha, k)
    return est_alpha

def filter_bad(kk, fek, ko=None):
    """filter_bad(kk, fek, ko)

    Helper function to remove bad/unwanted values from the spectrum

    Args:
        kk (np.ndarray): wavenumbers to clean
        fek (np.ndarray): Spectrum to clean
        ko (np.ndarray): 'true' wavenumbers
    Returns:
        kk (np.ndarray): Cleaned wavenumbers
        fek (np.ndarray): Cleaned spectrum
    """
    # If we provide a 'ko', make sure we don't extend outside of its range
    if ko is not None:
        kmin = np.nanmin(ko)
        kmax = np.nanmax(ko)
        fek = fek[kk >= kmin]
        kk = kk[kk >= kmin]
        fek = fek[kk <= kmax]
        kk = kk[kk <= kmax]
    kk = kk[np.isfinite(fek)]
    fek = fek[np.isfinite(fek)]
    # Remove "un-physical" negative values
    if np.sum(fek <= 0) > 0:
        last_0 = np.where(fek<0)[0][-1]
        kk = kk[last_0:]
        fek = fek[last_0:]
        kk = kk[fek > 0]
        fek = fek[fek > 0]
    # Interpolate onto 'ko' if provided
    if ko is not None:
        fek = log_log_interpolate(kk, fek, ko[ko <= kmax])
        kk = ko[ko <= kmax]
    return kk, fek

def sf_to_spectrum(ell, sf2, b, ko=None):
    """sf_to_spectrum(ell, sf2, phys_dims, grid_dims, b, ko)
    
    Args:
        ell (np.ndarray): Lags
        sf2 (np.ndarray): Angle-averaged second order structure function
        b (float): Wavenumber bias factor
        ko (np.ndarray/None): Discrete wavenumbers for FFT
    Returns:
        ke (np.ndarray): Equivalent wavenumbers
        BfekS (np.ndarray): Uncorrected equivalent spectrum
    """
    dSdell = np.gradient(sf2, ell)
    BfekS = (1./2.) * ell**2 * dSdell / b
    ke = b / ell
    ke, BfekS = ke[::-1], BfekS[::-1]
    ke, BfekS = filter_bad(ke, BfekS, ko)
    return ke, BfekS

def debias(est_alpha, b, D):
    """debias(est_alpha, b, D)

    The power law bias factor B^{pow}

    Args:
        est_alpha (np.ndarray): Local power law estimate (betas)
        b (float): Wavenumber bias factor
        D (float): Number of dimensions
    Returns:
        Bpow (np.narray): Local power law based bias/correction factor
    """
    sp_beta = sp.symbols('beta', positive=True, real=True)
    sp_D = sp.symbols('D', positive=True, real=True)
    sp_b = sp.symbols('b', positive=True, real=True)
    spec_bias = sp.S(2)*sp.S(2)**(-sp_beta) * sp_b**(sp_beta - sp.S(1)) * (sp.gamma(sp_D/sp.S(2)) * sp.gamma((sp.S(3) - sp_beta) / sp.S(2))/sp.gamma(sp_D/sp.S(2) + sp_beta/sp.S(2) - sp.Rational(1,2)))
    spec_bias_f = sp.lambdify((sp_D, sp_b, sp_beta), spec_bias)
    # clamp the local beta
    est_alpha[est_alpha > -1.01] = -1.01
    est_alpha[est_alpha < -2.99] = -2.99
    bias_factor = spec_bias_f(float(D), b, -est_alpha)
    return bias_factor

def equiv_spectrum(ell, sf2, D, b=None, ko=None):
    """equiv_spectrum(ell, sf2, D, b, ko)

    Calculates the equivalent spectrum

    Args:
        ell (np.ndarray): Lags
        sf2 (np.ndarray): Angle-averaged second-order structure function
        D (int): Number of dimensions
        b (float/None): The wavenumber bias factor (little-b). If None, uses $b^{est}$.
        ko (np.ndarray): FFT based wavenumbers to interpolate the equivalent spectrum onto
    Returns:
        ke (np.ndarray): Equivalent wavenumbers
        BfekS (np.ndarray): Uncorrected equivalent spectrum
        fekS (np.ndarray): Debiased equivalent spectrum
    """
    if b is None:
        if D == 1:
            b = 1.
        else:
            b = np.sqrt(2.*float(D) - 2.)
    ke, BfekS = sf_to_spectrum(ell, sf2, b, ko)
    a_fekS = get_powerlaw(ke, BfekS)
    B = debias(a_fekS, b, float(D))
    return ke, BfekS, BfekS/B
