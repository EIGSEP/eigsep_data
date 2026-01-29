"""
Methods for flagging RFI in autocorrelation data. Many of these methods
are adapted from hera_qm/xrfi.py, but without a lot of the
array-level logic.
"""

import hera_filters
import numpy as np
from scipy.ndimage import binary_dilation, convolve, median_filter

# --------- general methods ---------


# from hera_qm
def robust_divide(num, den):
    """
    Prevent division by zero.
    This function will compute division between two array-like objects
    by setting values to infinity when the denominator is small for the
    given data type. This avoids floating point exception warnings that
    may hide genuine problems in the data.

    Parameters
    ----------
    num : array
        The numerator.
    den : array
        The denominator.

    Returns
    -------
    out : array
        The result of dividing num / den. Elements where den is small
        (or zero) are set to infinity.

    """
    thresh = np.finfo(den.dtype).eps
    out = np.true_divide(num, den, where=(np.abs(den) > thresh))
    out = np.where(np.abs(den) > thresh, out, np.inf)
    return out


def grow_flags(flags, axes=None):
    """
    Grow RFI flags by 1 pixel along `axes`.

    Parameters
    ----------
    flags : ndarray of bool
        2D array of RFI flags.
    axes : int, tuple or None
        Which axis or axes to grow the flags in. None means both time and
        frequency.

    Returns
    -------
    new_flags : ndarray of bool

    """
    new_flags = binary_dilation(flags, axes=axes)
    return new_flags


def broadcast_flags(flags, time_thresh=0.5, freq_thresh=0.5):
    """
    Flag an entire integration or entire channel if the flag occupancy
    is greater than the given threshold.

    Parameters
    ----------
    flags : ndarray of bool
        2D array of RFI flags
    time_thresh : float
        Threshold in time axis (axis 0)
    freq_thresh : float
        Threshold in frequency axis (axis 1)

    Returns
    -------
    new_flags : ndarray of bool

    """
    new_flags = flags.copy()

    ch_means = flags.mean(axis=0)  # avg flag in each channel
    int_means = flags.mean(axis=1)  # avg flag in each integration

    new_flags[:, ch_means > freq_thresh] = True
    new_flags[int_means > time_thresh, :] = True
    return new_flags


# ------ noise estimation methods ------


def radiometer_noise(data, dt, df, auto_corr=True):
    """
    Estimate the radiometer noise level in the data.

    Parameters
    ----------
    data : ndarray
        2D array of data (or data model) to estimate noise from.
    dt : float
        Integration time in seconds.
    df : float
        Channel width in Hz.
    auto_corr : bool
        Whether the data is from autocorrelations or from
        cross-correlations.

    Returns
    -------
    noise : ndarray
        2D array of estimated noise standard deviation.

    """
    noise = np.abs(data) / np.sqrt(dt * df)
    if not auto_corr:
        noise /= np.sqrt(2)
    return noise


def median_absolute_deviation(data, axis=None, kernel_len=None):
    """
    Compute the median absolute deviation (MAD) of the data.

    Parameters
    ----------
    data : ndarray
        Input 2D data array.
    axis : {0, 1} or None
        Axis along which to compute the MAD. If None, compute over
        the raveled array.
    kernel_len : int or None
        If given, compute the MAD in a rolling window of this length
        along the specified axis.

    Returns
    -------
    sigma : ndarray
        The MAD of the data, scaled to be an estimate of the standard
        deviation.

    """
    if kernel_len is not None:
        if axis is None:
            raise ValueError("Must specify axis when using kernel_len")
        shape = [1] * data.ndim
        shape[axis] = kernel_len
        med = median_filter(data, size=tuple(shape), mode="mirror")
        res = np.abs(data - med)
        mad = median_filter(res, size=tuple(shape), mode="mirror")
    else:
        med = np.median(data, axis=axis, keepdims=True)
        res = np.abs(data - med)
        mad = np.median(res, axis=axis, keepdims=True)

    sigma = 1.4826 * mad
    return sigma


# ------ flagging methods ------


def median_flagger(data, nsig=8, kernel_half_width=5, return_z=False):
    """
    Flag data samples using a 1D median filter model and a sigma threshold.

    A rolling median filter is applied along the second axis of the data
    using a window of width ``2 * kernel_half_width + 1``. Samples whose
    residuals from this median model exceed ``nsig`` times an estimate of
    the noise level (from the median absolute deviation) are flagged as RFI.

    Parameters
    ----------
    data : ndarray
        Input data array. Typically 2D with shape ``(ntime, nfreq)``.
    nsig : float, optional
        Sigma threshold used to define outliers in units of the estimated
        standard deviation. Default is 8.
    kernel_half_width : int, optional
        Half-width of the rolling median window along the second axis.
        The full window size is ``2 * kernel_half_width + 1``. Default is 5.
    return_z : bool, optional
        If True, also return the z-score array used for thresholding.
        Default is False.

    Returns
    -------
    flags : ndarray of bool
        Boolean array of the same shape as ``data`` indicating flagged
        samples (True for flagged).
    z_score : ndarray, optional
        Array of z-scores with the same shape as ``data``. Only returned
        if ``return_z`` is True.

    """
    width = 2 * kernel_half_width + 1
    kernel = np.ones((1, width))  # no need to set center to 0 for median

    # median calculation
    model = median_filter(data, footprint=kernel, mode="mirror")
    residuals = data - model

    # estimate noise from data
    mad = np.median(np.abs(residuals))  # median abs deviation
    sigma = 1.4826 * mad

    z_score = robust_divide(residuals, sigma)

    flags = np.where(np.isnan(z_score), True, np.abs(z_score) > nsig)

    if return_z:
        return flags, z_score
    else:
        return flags


# adapted from hera_qm (channel_diff_flagger)
def mean_flagger(data, noise, nsig=6, kernel_widths=[3, 4, 5], flags=None):
    """
    Identify RFI in data using channel differencing kernels. Returns a
    boolean array of flags with values of True indicating channels
    flagged for RFI

    Parameters:
    ----------
    data: np.ndarray
        2D data array of the shape (time, frequency)
    noise: np.ndarray
        2D array for containing an estimate of the noise standard
        deviation. Must be the same shape as the data
    nsig: float, default=6
        The number of sigma in the metric above which to flag pixels.
    kernel_widths: list, default=[3, 4, 5]
        Half-width of the convolution kernels used to produce model.
        True kernel width is (2 * kernel_width + 1)
    flags: np.ndarray, default=None
        2D array of boolean flags to be interpreted as mask for data.
        Must be the same shape as data.

    Returns:
    -------
    flags: np.ndarray
        Array of boolean flags that has the same shape as the data,
        where values of True indicate flagged channels
    """
    if flags is None:
        wgts = np.ones_like(data)
    elif flags.dtype != bool:
        raise TypeError("Input flag array must be type bool")
    else:
        wgts = np.array(np.logical_not(flags), dtype=np.float64)

    # Iterate through kernel widths
    for kw in kernel_widths:
        # Build convolution kernel
        width = 2 * kw + 1
        kernel = np.ones((1, width))
        kernel[0, width // 2] = 0

        # Convolve kernel with data and weights
        _data = convolve(data * wgts, kernel)
        _wgts = convolve(wgts, kernel)

        # Calculate smooth model
        model = robust_divide(_data, _wgts)

        # Estimate noise level in absence of RFI
        sigma = np.abs(model) * (noise / data)
        res = data - model

        # Identify outlier channels (both positive and negative deviations)
        wgts = np.where(np.abs(res) > sigma * nsig, 0.0, 1.0)

    return np.isclose(wgts, 0)


# adapted from hera_qm
def dpss_flagger(
    data,
    noise,
    freqs,
    filter_centers,
    filter_half_widths,
    flags=None,
    nsig=6,
    mode="dpss_solve",
    eigenval_cutoff=[1e-9],
    suppression_factors=[1e-9],
    cache=None,
    return_models=False,
):
    """
    Identify RFI in visibilities by filtering data with discrete
    prolate spheroidal sequences. Returns a boolean array of flags with
    values of True indicating channels flagged for RFI

    Parameters:
    ----------
    data: np.ndarray
        2D data array of the shape (time, frequency)
    noise: np.ndarray
        2D array for containing an estimate of the noise standard
        deviation of the data.
        Must be the same shape as the data.
    freqs: np.ndarray
        1D array of frequencies present in the data in units of Hz
    filter_centers: array-like
        list of floats of centers of delay filter windows in nanosec
    filter_half_widths: array-like
        list of floats of half-widths of delay filter windows in nanosec
    flags: np.ndarray
        2D array of boolean flags to be interpreted as mask for data.
        Must be the same shape as data.
    nsig: float, default=6
        The number of sigma in the metric above which to flag pixels.
    mode: str, default='dpss_solve'
        Method used to solve for DPSS model components. Options are
        'dpss_matrix', 'dpss_solve', and 'dpss_leastsq'.
    eigenval_cutoff: array-like, default=[1e-9]
        List of sinc_matrix eigenvalue cutoffs to use for included
        DPSS modes.
    suppression_factors: array-like, default=[1e-9]
        Specifies the fractional residuals of model to leave in the
        data. For example, 1e-6 means that the filter
        will leave in 1e-6 of data fitted by the model.
    cache: dictionary, default=None
        Dictionary for caching fitting matrices. By default this value
        is None to prevent the size of the cached matrices from getting
        too large. By passing in a cache dictionary, this function could
        be much faster, but the memory requirement will also increase.
    return_models: bool, default=False
        If True, also return the fitted data model and noise (sigma)
        arrays in addition to the flags.

    Returns:
    -------
    flags: np.ndarray
        Array of boolean flags that has the same shape as the data,
        where values of True indicate flagged channels
    model : np.ndarray, optional
        Data model. Returned only if ``return_models`` is True.
    sigma : np.ndarray, optional
        Noise model. Returned only if ``return_models`` is True.
    """
    if len(suppression_factors) == 1 and len(filter_centers) > 1:
        suppression_factors = len(filter_centers) * suppression_factors

    if len(eigenval_cutoff) == 1 and len(filter_centers) > 1:
        eigenval_cutoff = len(filter_centers) * eigenval_cutoff

    if flags is None:
        wgts = np.ones_like(data)
    elif flags.dtype != bool:
        raise TypeError("Input flag array must be type bool")
    else:
        wgts = np.array(np.logical_not(flags), dtype=np.float64)

    # Compute model and residuals
    model, _, _ = hera_filters.dspec.fourier_filter(
        freqs,
        data,
        wgts,
        filter_centers,
        filter_half_widths,
        mode=mode,
        suppression_factors=suppression_factors,
        eigenval_cutoff=eigenval_cutoff,
        cache=cache,
    )
    res = data - model

    # Use smooth model to noise standard deviation without RFI
    sigma = np.abs(model) * (noise / data)

    # Determine weights (flag both positive and negative outliers)
    weights = np.where(np.abs(res) > sigma * nsig, True, False)
    if not return_models:
        return weights
    return weights, model, sigma


def dpss2d(data, At, Af, flags=None):
    """
    Fit a 2D DPSS model to the data.

    Parameters
    ----------
    data : ndarray
        2D array of data to fit. Shape (time, frequency).
    At : ndarray
        Design matrix for time axis.
    Af : ndarray
        Design matrix for frequency axis.
    flags : ndarray of bool
        2D array of flags. Shape (time, frequency).

    Returns
    -------
    model : ndarray
        2D array of the fitted DPSS model. Shape (time, frequency).

    Notes
    -----
    Can use hera_filters.dspec.dpss_operators to generate At and Af.

    """
    if flags is None:
        flags = np.zeros_like(data, dtype=bool)

    wgts = np.logical_not(flags).astype(float)
    fit, _ = hera_filters.dspec.sparse_linear_fit_2D(data, wgts, At, Af)
    dmdl = At @ fit @ Af.T
    return dmdl
