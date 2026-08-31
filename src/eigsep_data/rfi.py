"""
Methods for flagging RFI in autocorrelation data. Many of these methods
are adapted from hera_qm/xrfi.py, but without a lot of the
array-level logic.
"""

import hera_filters
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation, convolve, median_filter
from scipy.optimize import least_squares, minimize
from scipy.signal.windows import dpss

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
    good = np.abs(den) > thresh
    shape = np.broadcast_shapes(np.shape(num), np.shape(den))
    out = np.full(shape, np.inf)
    np.true_divide(num, den, where=good, out=out)
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

    # estimate noise from data; exact-zero residuals (window median
    # equal to the sample, common in quantized or mostly-zero data)
    # would deflate the MAD to the degenerate all-flagged limit
    nonzero = residuals[residuals != 0]
    if nonzero.size:
        mad = np.median(np.abs(nonzero))
    else:
        mad = np.float64(0.0)
    sigma = 1.4826 * mad

    z_score = robust_divide(residuals, sigma)
    # a sample equal to its model is never an outlier, even if sigma=0
    z_score = np.where(residuals == 0, 0.0, z_score)

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


# -----------------------------------------------------------------------
# Beam-mapping DPSS pipeline
# -----------------------------------------------------------------------

def flag_rfi_time(data, window=21, sigma=6.0, floor_fraction=0.05):
    """
    Flag short-duration positive RFI independently for each frequency.

    Uses a sliding median baseline to identify positive excursions above
    a local robust threshold, with a per-frequency global floor to prevent
    the threshold from collapsing near beam nulls.

    Parameters
    ----------
    data : np.ndarray, shape (nsample, nfreq)
    window : int
        Length of the median filter window in time (forced odd).
    sigma : float
        Detection threshold in units of local robust sigma.
    floor_fraction : float
        Minimum threshold as a fraction of the per-frequency global MAD.

    Returns
    -------
    rfi_mask : np.ndarray of bool, shape (nsample, nfreq)
    baseline : np.ndarray, shape (nsample, nfreq)
    residual : np.ndarray, shape (nsample, nfreq)
    """
    data = np.asarray(data, dtype=float)
    if window % 2 == 0:
        window += 1

    baseline = median_filter(data, size=(window, 1), mode="nearest")
    residual = data - baseline

    local_mad = median_filter(np.abs(residual), size=(window, 1), mode="nearest")
    local_sigma = 1.4826 * local_mad

    freq_scale = 1.4826 * np.nanmedian(
        np.abs(data - np.nanmedian(data, axis=0)), axis=0
    )
    sigma_floor = floor_fraction * freq_scale[None, :]
    effective_sigma = np.maximum(local_sigma, sigma_floor)

    rfi_mask = (residual > sigma * effective_sigma) & np.isfinite(data)
    return rfi_mask, baseline, residual


def build_dpss_basis(nchan, nw=4.0, nterms=16, include_poly=True, poly_order=2):
    """
    Build a smooth spectral basis using DPSS tapers plus optional polynomials.

    Parameters
    ----------
    nchan : int
        Number of spectral channels.
    nw : float
        DPSS time-bandwidth product.
    nterms : int
        Number of DPSS basis vectors.
    include_poly : bool
        If True, append low-order polynomial basis vectors.
    poly_order : int
        Highest polynomial order to include.

    Returns
    -------
    A : np.ndarray, shape (nchan, nbasis)
        Column-normalized basis matrix.
    """
    basis = list(dpss(M=nchan, NW=nw, Kmax=nterms, sym=False))
    if include_poly:
        x = np.linspace(-1, 1, nchan)
        for p in range(poly_order + 1):
            basis.append(x ** p)
    A = np.vstack(basis).T
    A = A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-30)
    return A


def normalize_each_freq(x, valid=None, eps=1e-30):
    """
    Normalize each frequency column by its valid-sample mean.

    Parameters
    ----------
    x : np.ndarray, shape (ntime, nfreq)
    valid : np.ndarray of bool, optional
        Same shape as x. Defaults to isfinite(x).
    eps : float
        Columns whose mean is below eps are left as NaN.

    Returns
    -------
    out : np.ndarray, shape (ntime, nfreq)
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if valid is None:
        valid = np.isfinite(x)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(x)
    for fi in range(x.shape[1]):
        good = valid[:, fi]
        if np.sum(good) > 0:
            m = np.nanmean(x[good, fi])
            if np.isfinite(m) and np.abs(m) > eps:
                out[good, fi] = x[good, fi] / m
    return out


def build_frequency_trust_mask(freqs, min_freq=50.0, fm_low=87.0, fm_high=108.0):
    """
    Return a boolean mask of trusted frequency channels.

    Excludes channels below min_freq and the FM band [fm_low, fm_high].
    All frequencies assumed to be in MHz.

    Parameters
    ----------
    freqs : array-like, shape (nchan,)
    min_freq : float
        Low-frequency cutoff in MHz.
    fm_low, fm_high : float
        FM band edges in MHz to exclude.

    Returns
    -------
    trusted : np.ndarray of bool, shape (nchan,)
    """
    freqs = np.asarray(freqs, dtype=float)
    trusted = np.isfinite(freqs)
    if min_freq is not None:
        trusted &= freqs >= min_freq
    if fm_low is not None and fm_high is not None:
        trusted &= ~((freqs >= fm_low) & (freqs <= fm_high))
    return trusted


def make_dpss_fit_mask(
    nchan,
    chs,
    guard_bins=1,
    fit_min_chan=None,
    fit_max_chan=None,
    freqs=None,
    min_freq=50.0,
    fm_low=87.0,
    fm_high=108.0,
):
    """
    Build a boolean mask of channels used in the DPSS bandpass fit.

    Comb channels and their guard bins are excluded; optionally also
    channels outside a frequency range or in the FM band.

    Parameters
    ----------
    nchan : int
    chs : array-like of int
        Comb channel indices to mask out.
    guard_bins : int
        Number of channels to exclude on either side of each comb channel.
    fit_min_chan, fit_max_chan : int, optional
        Channel range to fit within (default: full range).
    freqs : array-like, shape (nchan,), optional
        If provided, also applies build_frequency_trust_mask.
    min_freq, fm_low, fm_high : float
        Passed to build_frequency_trust_mask when freqs is given.

    Returns
    -------
    fit_mask : np.ndarray of bool, shape (nchan,)
    """
    chs = np.asarray(chs, dtype=int)
    fit_min_chan = max(0, int(fit_min_chan or 0))
    fit_max_chan = min(nchan, int(fit_max_chan or nchan))

    fit_mask = np.zeros(nchan, dtype=bool)
    fit_mask[fit_min_chan:fit_max_chan] = True

    if freqs is not None:
        freqs = np.asarray(freqs, dtype=float)
        if freqs.shape != (nchan,):
            raise ValueError(f"freqs must have shape ({nchan},), got {freqs.shape}")
        fit_mask &= build_frequency_trust_mask(freqs, min_freq, fm_low, fm_high)

    for c in chs:
        lo = max(0, c - guard_bins)
        hi = min(nchan, c + guard_bins + 1)
        fit_mask[lo:hi] = False

    return fit_mask


def fit_smooth_model_one_spectrum(
    spectrum,
    basis,
    fit_mask,
    constraint_chs=None,
    use_log=True,
    ridge=1e-6,
    min_good_extra=2,
    enforce_comb_upper=True,
    maxiter=300,
    ftol=1e-10,
    constraint_tol=1e-8,
    return_info=False,
):
    """
    Fit one smooth DPSS bandpass model with two-pass IRLS RFI rejection.

    Optionally enforces that the model does not exceed the measured comb
    channel values (useful when the comb tones are reliable upper bounds).

    Parameters
    ----------
    spectrum : array-like, shape (nchan,)
    basis : np.ndarray, shape (nchan, nbasis)
        From build_dpss_basis.
    fit_mask : np.ndarray of bool, shape (nchan,)
        Channels used in the fit.
    constraint_chs : array-like of int, optional
        Comb channels used as upper-bound constraints.
    use_log : bool
        Fit in log-power space (recommended).
    ridge : float
        Ridge regularization strength.
    min_good_extra : int
        Minimum extra channels beyond nbasis required for a valid fit.
    enforce_comb_upper : bool
        If True and constraint_chs is given, apply upper-bound constraints.
    maxiter, ftol : int, float
        SLSQP optimizer settings (used only when constraints are active).
    constraint_tol : float
        Tolerance for constraint satisfaction.
    return_info : bool
        If True, return (model, info_dict).

    Returns
    -------
    model : np.ndarray, shape (nchan,)
        NaN where the fit failed.
    info : dict, only if return_info=True
    """
    y = np.asarray(spectrum, dtype=float)
    nchan, nbasis = basis.shape

    good = fit_mask & np.isfinite(y)
    if use_log:
        good &= y > 0

    def _fail(msg):
        info = {"success": False, "message": msg,
                "constraint_active": False, "max_constraint_violation": np.nan}
        m = np.full(nchan, np.nan)
        return (m, info) if return_info else m

    if np.sum(good) < nbasis + min_good_extra:
        return _fail("Not enough valid fit channels")

    Ag = basis[good]
    if use_log:
        floor_val = np.percentile(y[good], 1)
        yy = np.log(np.clip(y[good], floor_val, None))
    else:
        yy = y[good]

    # First pass: ridge fit
    lhs = Ag.T @ Ag + ridge * np.eye(nbasis)
    rhs = Ag.T @ yy
    try:
        coeff0 = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeff0 = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    # Identify positive outliers (RFI) via MAD and refit
    residuals = yy - Ag @ coeff0
    med_res = np.median(residuals)
    std_robust = 1.4826 * np.median(np.abs(residuals - med_res))
    rfi_mask = residuals > (med_res + 3.0 * (std_robust + 1e-8))
    Ag_fit, yy_fit = Ag[~rfi_mask], yy[~rfi_mask]

    lhs2 = Ag_fit.T @ Ag_fit + ridge * np.eye(nbasis)
    rhs2 = Ag_fit.T @ yy_fit
    try:
        coeff0 = np.linalg.solve(lhs2, rhs2)
    except np.linalg.LinAlgError:
        coeff0 = np.linalg.lstsq(lhs2, rhs2, rcond=None)[0]

    def _build_model(coeff):
        m = basis @ coeff
        return np.exp(m) if use_log else m

    # No constraints
    if not enforce_comb_upper or constraint_chs is None or len(constraint_chs) == 0:
        info = {"success": True, "message": "Unconstrained ridge fit",
                "constraint_active": False, "max_constraint_violation": 0.0}
        return (_build_model(coeff0), info) if return_info else _build_model(coeff0)

    # Check which constraint channels are valid
    constraint_chs = np.asarray(constraint_chs, dtype=int)
    yc = y[constraint_chs]
    ok = np.isfinite(yc) & (yc > 0 if use_log else np.ones_like(yc, dtype=bool))
    chs_ok = constraint_chs[ok]

    if len(chs_ok) == 0:
        info = {"success": True, "message": "No valid constraint channels",
                "constraint_active": False, "max_constraint_violation": np.nan}
        return (_build_model(coeff0), info) if return_info else _build_model(coeff0)

    Ac = basis[chs_ok]
    upper = np.log(y[chs_ok]) if use_log else y[chs_ok]

    if np.max(Ac @ coeff0 - upper) <= constraint_tol:
        viol = float(max(0.0, np.max(Ac @ coeff0 - upper)))
        info = {"success": True, "message": "Unconstrained solution already feasible",
                "constraint_active": False, "max_constraint_violation": viol}
        return (_build_model(coeff0), info) if return_info else _build_model(coeff0)

    # Constrained SLSQP
    def objective(c):
        r = Ag_fit @ c - yy_fit
        return 0.5 * np.dot(r, r) + 0.5 * ridge * np.dot(c, c)

    def gradient(c):
        return Ag_fit.T @ (Ag_fit @ c - yy_fit) + ridge * c

    result = minimize(
        objective, coeff0, jac=gradient,
        constraints={"type": "ineq", "fun": lambda c: upper - Ac @ c, "jac": lambda c: -Ac},
        method="SLSQP",
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )

    viol = float(max(0.0, np.max(Ac @ result.x - upper)))
    success = result.success and viol <= constraint_tol
    if not success:
        return _fail(result.message)

    info = {"success": True, "message": result.message,
            "constraint_active": True, "max_constraint_violation": viol}
    m = _build_model(result.x)
    return (m, info) if return_info else m


def fit_dpss_model_per_time(
    raw,
    chs,
    freqs,
    nw=4.0,
    nterms=8,
    include_poly=True,
    poly_order=2,
    guard_bins=1,
    fit_min_chan=None,
    fit_max_chan=None,
    min_freq=50.0,
    fm_low=87.0,
    fm_high=108.0,
    use_log=True,
    ridge=1e-6,
    enforce_comb_upper=True,
    mask_untrusted_output=True,
    constraint_tol=1e-8,
    optimizer_maxiter=300,
    optimizer_ftol=1e-10,
    verbose=True,
):
    """
    Fit a smooth DPSS bandpass model independently at each time sample and
    return the comb-channel residuals.

    Parameters
    ----------
    raw : np.ndarray, shape (ntime, nchan)
        Raw spectra.
    chs : array-like of int
        Comb channel indices.
    freqs : array-like, shape (nchan,)
        Frequency axis in MHz.
    nw, nterms, include_poly, poly_order : see build_dpss_basis.
    guard_bins : int
        Channels around comb tones excluded from the fit.
    fit_min_chan, fit_max_chan : int, optional
        Channel range restriction.
    min_freq, fm_low, fm_high : float
        Frequency trust mask parameters (MHz).
    use_log : bool
        Fit in log-power space.
    ridge : float
        Ridge regularization.
    enforce_comb_upper : bool
        Enforce upper bounds at comb channels.
    mask_untrusted_output : bool
        Set untrusted comb frequencies to NaN in the output.
    constraint_tol, optimizer_maxiter, optimizer_ftol :
        Passed to fit_smooth_model_one_spectrum.
    verbose : bool

    Returns
    -------
    dict with keys:
        model_raw, reduced, test_data, fit_mask, trusted_freq_mask,
        trusted_comb_mask, trusted_constraint_chs, basis, chs, freqs,
        fit_success
    """
    raw = np.asarray(raw, dtype=float)
    chs = np.asarray(chs, dtype=int)
    freqs = np.asarray(freqs, dtype=float)
    ntime, nchan = raw.shape

    if freqs.shape != (nchan,):
        raise ValueError(f"freqs must have shape ({nchan},), got {freqs.shape}")
    if np.any(chs < 0) or np.any(chs >= nchan):
        raise ValueError("Some entries of chs are outside the raw spectral axis.")

    trusted_freq_mask = build_frequency_trust_mask(freqs, min_freq, fm_low, fm_high)
    trusted_comb_mask = trusted_freq_mask[chs]
    trusted_constraint_chs = chs[trusted_comb_mask]

    basis = build_dpss_basis(nchan, nw=nw, nterms=nterms,
                             include_poly=include_poly, poly_order=poly_order)
    fit_mask = make_dpss_fit_mask(nchan, chs, guard_bins=guard_bins,
                                  fit_min_chan=fit_min_chan, fit_max_chan=fit_max_chan,
                                  freqs=freqs, min_freq=min_freq,
                                  fm_low=fm_low, fm_high=fm_high)
    if verbose:
        print("number of fit channels:", np.sum(fit_mask))

    model_raw = np.full_like(raw, np.nan)
    fit_success = np.zeros(ntime, dtype=bool)
    constraint_active = np.zeros(ntime, dtype=bool)
    max_violation = np.full(ntime, np.nan)

    for ti in range(ntime):
        model_t, info = fit_smooth_model_one_spectrum(
            spectrum=raw[ti], basis=basis, fit_mask=fit_mask,
            constraint_chs=trusted_constraint_chs, use_log=use_log, ridge=ridge,
            enforce_comb_upper=enforce_comb_upper, maxiter=optimizer_maxiter,
            ftol=optimizer_ftol, constraint_tol=constraint_tol, return_info=True,
        )
        model_raw[ti] = model_t
        fit_success[ti] = info["success"]
        constraint_active[ti] = info["constraint_active"]
        max_violation[ti] = info["max_constraint_violation"]

        if verbose and ti % 500 == 0:
            print(f"fit {ti}/{ntime} (success={fit_success[ti]}, "
                  f"constraint={constraint_active[ti]})")

    reduced = raw[:, chs] - model_raw[:, chs]
    if mask_untrusted_output:
        reduced[:, ~trusted_comb_mask] = np.nan

    test_data = normalize_each_freq(reduced, valid=np.isfinite(reduced))

    if verbose:
        print(f"\nDPSS fit diagnostics")
        print(f"successful fits: {np.sum(fit_success)}/{ntime}")
        print(f"fraction successful: {np.mean(fit_success):.3f}")

    return {
        "model_raw": model_raw,
        "reduced": reduced,
        "test_data": test_data,
        "fit_mask": fit_mask,
        "trusted_freq_mask": trusted_freq_mask,
        "trusted_comb_mask": trusted_comb_mask,
        "trusted_constraint_chs": trusted_constraint_chs,
        "basis": basis,
        "chs": chs,
        "freqs": freqs,
        "fit_success": fit_success,
    }


def robust_dpss_fit(y, B, sigma=4.0, max_iter=4, min_points=None):
    """
    Robust iterative DPSS fit to one spectrum with MAD clipping.

    Parameters
    ----------
    y : np.ndarray, shape (nfreq,)
    B : np.ndarray, shape (nfreq, nmodes)
    sigma : float
        MAD clipping threshold.
    max_iter : int
        Maximum clipping iterations.
    min_points : int, optional
        Minimum channels to retain (default: nmodes + 3).

    Returns
    -------
    model : np.ndarray, shape (nfreq,)
    good : np.ndarray of bool, shape (nfreq,)
    """
    y = np.asarray(y, dtype=float)
    nmodes = B.shape[1]
    if min_points is None:
        min_points = nmodes + 3

    good = np.isfinite(y)
    if np.sum(good) < min_points:
        return np.full_like(y, np.nan), good

    coeff0 = np.linalg.lstsq(B[good], y[good], rcond=None)[0]
    result = least_squares(lambda c: B[good] @ c - y[good], coeff0,
                           loss="soft_l1", f_scale=1.0)
    coeff = result.x

    for _ in range(max_iter):
        model = B @ coeff
        residual = y - model
        r = residual[good]
        center = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - center))
        robust_sigma = 1.4826 * mad

        if not np.isfinite(robust_sigma) or robust_sigma <= 0:
            break

        new_good = np.isfinite(y) & (np.abs(residual - center) <= sigma * robust_sigma)
        if np.sum(new_good) < min_points:
            break
        if np.array_equal(new_good, good):
            good = new_good
            break
        good = new_good
        coeff = np.linalg.lstsq(B[good], y[good], rcond=None)[0]

    if np.sum(good) < min_points:
        return np.full_like(y, np.nan), good

    coeff = np.linalg.lstsq(B[good], y[good], rcond=None)[0]
    return B @ coeff, good


def dpss_fit_logspace(y, B, eps=1e-12):
    """
    Fit a DPSS model to one spectrum in log-power space.

    Parameters
    ----------
    y : np.ndarray, shape (nfreq,)
        Power measurements (must be positive for valid channels).
    B : np.ndarray, shape (nfreq, nmodes)
    eps : float
        Floor applied before log.

    Returns
    -------
    model : np.ndarray, shape (nfreq,)
        Positive DPSS model in linear power units. NaN if fit fails.
    """
    y = np.asarray(y, dtype=float)
    good = np.isfinite(y) & (y > 0)
    if np.sum(good) < B.shape[1] + 2:
        return np.full_like(y, np.nan)

    ylog = np.log(np.maximum(y[good], eps))
    coeff, *_ = np.linalg.lstsq(B[good], ylog, rcond=None)
    return np.exp(B @ coeff)


def fit_dpss_interleaved_fix(
    P_meas,
    freqs,
    rfi_mask=None,
    nmodes=8,
    NW=4.0,
    max_bad_fraction=0.35,
):
    """
    Separate and reconstruct two interleaved tone sets (even/odd columns)
    with DPSS fits, preserving clean measured values.

    Good measured tones are kept; missing or RFI-flagged tones are
    reconstructed from the DPSS model interpolated onto the full grid.

    Parameters
    ----------
    P_meas : np.ndarray, shape (nsample, nfreq)
    freqs : np.ndarray, shape (nfreq,)
    rfi_mask : np.ndarray of bool, shape (nsample, nfreq), optional
    nmodes : int
        Number of DPSS modes per tone set.
    NW : float
        DPSS time-bandwidth product.
    max_bad_fraction : float
        Time samples with more than this fraction of RFI-flagged channels
        are skipped entirely.

    Returns
    -------
    P1_hat : np.ndarray, shape (nsample, nfreq)  — even-indexed tone set
    P6_hat : np.ndarray, shape (nsample, nfreq)  — odd-indexed tone set
    G_hat  : np.ndarray, shape (nsample, nfreq)  — P1_hat + P6_hat
    bad_time : np.ndarray of bool, shape (nsample,)
    rfi_occupancy : np.ndarray, shape (nsample,)
    """
    P_meas = np.asarray(P_meas, dtype=float)
    freqs  = np.asarray(freqs, dtype=float)
    nsample, nfreq = P_meas.shape

    if rfi_mask is None:
        rfi_mask = np.zeros_like(P_meas, dtype=bool)
    else:
        rfi_mask = np.asarray(rfi_mask, dtype=bool)
    if rfi_mask.shape != P_meas.shape:
        raise ValueError("rfi_mask must have the same shape as P_meas")

    idx1, idx6 = np.arange(0, nfreq, 2), np.arange(1, nfreq, 2)
    f1, f6 = freqs[idx1], freqs[idx6]

    nm1, nm6 = min(nmodes, len(idx1)), min(nmodes, len(idx6))
    B1 = dpss(len(idx1), NW=NW, Kmax=nm1).T
    B6 = dpss(len(idx6), NW=NW, Kmax=nm6).T

    P1_hat = np.full_like(P_meas, np.nan)
    P6_hat = np.full_like(P_meas, np.nan)

    rfi_occupancy = np.mean(rfi_mask, axis=1)
    bad_time = rfi_occupancy > max_bad_fraction

    for si in range(nsample):
        if bad_time[si]:
            continue

        for y_sub, bad_sub, B_sub, f_sub, idx_sub, P_hat in [
            (P_meas[si, idx1], rfi_mask[si, idx1], B1, f1, idx1, P1_hat),
            (P_meas[si, idx6], rfi_mask[si, idx6], B6, f6, idx6, P6_hat),
        ]:
            good = ~(bad_sub | ~np.isfinite(y_sub))
            if np.sum(good) < B_sub.shape[1] + 2:
                continue
            coeff, *_ = np.linalg.lstsq(B_sub[good], y_sub[good], rcond=None)
            smooth = B_sub @ coeff
            P_hat[si, :] = interp1d(f_sub, smooth, kind="linear",
                                    bounds_error=False, fill_value="extrapolate")(freqs)
            P_hat[si, idx_sub[good]] = y_sub[good]

    return P1_hat, P6_hat, P1_hat + P6_hat, bad_time, rfi_occupancy


def fit_dpss_interleaved_log(P_meas, freqs, nmodes=4, NW=4.0, eps=1e-12):
    """
    Reconstruct two interleaved tone sets (even/odd columns) using DPSS
    fits in log-power space, then interpolate onto the full frequency grid.

    Parameters
    ----------
    P_meas : np.ndarray, shape (nsample, nfreq)
        RFI-cleaned interleaved power. Even columns = set 1, odd = set 6.
    freqs : np.ndarray, shape (nfreq,)
    nmodes : int
    NW : float
    eps : float
        Floor before log.

    Returns
    -------
    P1_hat : np.ndarray, shape (nsample, nfreq)
    P6_hat : np.ndarray, shape (nsample, nfreq)
    G_hat  : np.ndarray, shape (nsample, nfreq)  — P1_hat + P6_hat
    """
    P_meas = np.asarray(P_meas, dtype=float)
    freqs  = np.asarray(freqs, dtype=float)
    nsample, nfreq = P_meas.shape

    if freqs.ndim != 1 or len(freqs) != nfreq:
        raise ValueError(f"freqs must be 1D with length {nfreq}, got {freqs.shape}")

    idx1, idx6 = np.arange(0, nfreq, 2), np.arange(1, nfreq, 2)
    f1, f6 = freqs[idx1], freqs[idx6]

    nm1, nm6 = min(nmodes, len(idx1)), min(nmodes, len(idx6))
    B1 = dpss(len(idx1), NW=NW, Kmax=nm1).T
    B6 = dpss(len(idx6), NW=NW, Kmax=nm6).T

    P1_hat = np.full((nsample, nfreq), np.nan)
    P6_hat = np.full((nsample, nfreq), np.nan)

    for si in range(nsample):
        for y_sub, B_sub, f_sub, idx_sub, P_hat in [
            (P_meas[si, idx1], B1, f1, idx1, P1_hat),
            (P_meas[si, idx6], B6, f6, idx6, P6_hat),
        ]:
            smooth = dpss_fit_logspace(y_sub, B_sub, eps=eps)
            if not np.any(np.isfinite(smooth)):
                continue
            P_hat[si, :] = interp1d(f_sub, smooth, kind="linear",
                                    bounds_error=False, fill_value="extrapolate")(freqs)
            good = np.isfinite(y_sub) & (y_sub > 0)
            P_hat[si, idx_sub[good]] = y_sub[good]

    return P1_hat, P6_hat, P1_hat + P6_hat
