"""
Spherical-harmonic beam fitting.

This is unique to eigsep_data (not part of healjax): fit sparse HEALPix
maps to a regularized real spherical-harmonic basis, and synthesize
full maps back out.
"""

import healpy as hp
import numpy as np
from scipy.linalg import lstsq
from scipy.special import sph_harm_y


def build_real_design_matrix_from_angles(theta, phi, lmax):
    """
    Build design matrix A of shape (n_obs, (lmax+1)^2) mapping real
    coefficient vector x to real values at (theta, phi).

    Coefficient layout: a_l0, then for m=1..l: Re(a_lm), Im(a_lm).
    """
    npar = (lmax + 1) ** 2
    A = np.empty((theta.size, npar), dtype=np.float64)
    k = 0
    for l in range(lmax + 1):
        Y = sph_harm_y(l, 0, theta, phi)
        A[:, k] = Y.real
        k += 1
        for m in range(1, l + 1):
            Y = sph_harm_y(l, m, theta, phi)
            A[:, k]     =  2.0 * Y.real
            A[:, k + 1] = -2.0 * Y.imag
            k += 2
    return A


def x_to_alm(x, lmax):
    """Convert real coefficient vector x to complex healpy alm array."""
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    k = 0
    for l in range(lmax + 1):
        alm[hp.Alm.getidx(lmax, l, 0)] = x[k]
        k += 1
        for m in range(1, l + 1):
            alm[hp.Alm.getidx(lmax, l, m)] = x[k] + 1j * x[k + 1]
            k += 2
    return alm


def _make_penalty_diag(lmax, p=2):
    """
    Return penalty weight vector for Laplacian regularization.

    pen[k] = (l(l+1))^p for each coefficient in x; zero for l=0.
    """
    npar = (lmax + 1) ** 2
    pen = np.zeros(npar, dtype=np.float64)
    k = 0
    for l in range(lmax + 1):
        w = (l * (l + 1)) ** p
        pen[k] = w
        k += 1
        for m in range(1, l + 1):
            pen[k] = w
            pen[k + 1] = w
            k += 2
    return pen


def _solve_regularized_ls(A, y, lam, pen, w=None):
    """
    Solve argmin ||W^(1/2)(Ax - y)||^2 + lam * ||diag(pen)^(1/2) x||^2
    via row augmentation.

    Parameters
    ----------
    A : (n_obs, npar)
    y : (n_obs,)
    lam : float
    pen : (npar,) — penalty weights from _make_penalty_diag
    w : (n_obs,) optional observation weights
    """
    npar = A.shape[1]
    if w is not None:
        w  = np.asarray(w, dtype=np.float64)
        sw = np.sqrt(np.clip(w, 0.0, np.inf))
        A  = A * sw[:, None]
        y  = y * sw

    sqrt_lam_pen = np.sqrt(lam * pen)
    cols = np.where(sqrt_lam_pen > 0)[0]
    if cols.size > 0:
        R = np.zeros((cols.size, npar), dtype=np.float64)
        R[np.arange(cols.size), cols] = sqrt_lam_pen[cols]
        A = np.vstack([A, R])
        y = np.concatenate([y, np.zeros(cols.size, dtype=np.float64)])

    x, *_ = lstsq(A, y)
    return x


def fit_alms_from_maps(
    maps,
    nside,
    lmax=5,
    lam=1e-2,
    p=2,
    fit_log=False,
    eps=1e-12,
    user_weights=None,
    peak_weight_alpha=0.0,
    peak_weight_gamma=2.0,
    nest=False,
):
    """
    Fit spherical-harmonic coefficients (alms) to sparse HEALPix maps.

    Parameters
    ----------
    maps : (n_maps, npix) or (npix,)
        Input maps. Pixels equal to hp.UNSEEN or non-finite are ignored.
    nside : int
        HEALPix nside.
    lmax : int
        Maximum spherical-harmonic degree.
    lam : float
        Regularization strength (Laplacian penalty).
    p : int
        Regularization order: penalty = (l(l+1))^p.
    fit_log : bool
        If True, fit log(values). Values must be positive; eps prevents
        log(0).
    eps : float
        Floor applied before taking log when fit_log=True.
    user_weights : (npix,) optional
        Per-pixel weights applied to observed pixels.
    peak_weight_alpha : float
        Extra upweighting strength for bright pixels.
    peak_weight_gamma : float
        Power applied to normalized pixel value for peak upweighting.
    nest : bool
        HEALPix ordering (False = RING).

    Returns
    -------
    alms : np.ndarray, shape (n_maps, hp.Alm.getsize(lmax)), complex128
    """
    maps = np.asarray(maps)
    if maps.ndim == 1:
        maps = maps[None, :]
    n_maps, npix = maps.shape
    assert npix == hp.nside2npix(nside)

    pen = _make_penalty_diag(lmax, p=p)
    A_cache = {}

    alms = np.empty((n_maps, hp.Alm.getsize(lmax)), dtype=np.complex128)

    for i in range(n_maps):
        m = maps[i]
        known = (m != hp.UNSEEN) & np.isfinite(m)
        ipix = np.where(known)[0]
        if ipix.size == 0:
            raise ValueError(f"Map {i} has no observed pixels.")

        y_lin = m[ipix].astype(np.float64)
        y = np.log(np.maximum(y_lin, eps)) if fit_log else y_lin

        w = None
        if user_weights is not None:
            w = np.asarray(user_weights, dtype=np.float64)[ipix].copy()

        if peak_weight_alpha > 0:
            f    = np.maximum(y_lin, 0.0)
            fmax = f.max() if f.size else 1.0
            extra = 1.0 + peak_weight_alpha * (f / max(fmax, 1e-30)) ** peak_weight_gamma
            w = extra if w is None else (w * extra)

        key = tuple(ipix.tolist())
        if key not in A_cache:
            theta, phi = hp.pix2ang(nside, ipix, nest=nest)
            A_cache[key] = build_real_design_matrix_from_angles(theta, phi, lmax)

        x = _solve_regularized_ls(A_cache[key], y, lam=lam, pen=pen, w=w)
        alms[i] = x_to_alm(x, lmax)

    return alms


def alms_to_filled_maps(
    alms,
    nside,
    lmax,
    clamp_known=False,
    original_maps=None,
    fit_log=False,
    log_offset=None,
):
    """
    Synthesize full HEALPix maps from alm coefficients.

    Parameters
    ----------
    alms : (n_maps, hp.Alm.getsize(lmax)) or (alm_size,)
    nside : int
    lmax : int
    clamp_known : bool
        If True and original_maps is provided, restore observed pixels
        to their original values.
    original_maps : array-like, optional
        Original sparse maps used to identify observed pixels for clamping.
    fit_log : bool
        If True, exponentiate the synthesized map (undoes log-domain fit).
    log_offset : (n_maps,) optional
        Additive offsets applied in log-domain before exponentiation.

    Returns
    -------
    out : np.ndarray, shape (n_maps, npix)
    """
    alms = np.asarray(alms)
    if alms.ndim == 1:
        alms = alms[None, :]
    n_maps = alms.shape[0]

    out = np.empty((n_maps, hp.nside2npix(nside)), dtype=np.float64)
    for i in range(n_maps):
        pred = hp.alm2map(alms[i].astype(np.complex128), nside=nside, lmax=lmax)
        if fit_log:
            if log_offset is not None:
                pred = pred + float(log_offset[i])
            pred = np.exp(pred)
        if clamp_known and original_maps is not None:
            m0 = np.asarray(
                original_maps[i]
                if np.asarray(original_maps).ndim == 2
                else original_maps
            )
            known = (m0 != hp.UNSEEN) & np.isfinite(m0)
            pred[known] = m0[known]
        out[i] = pred

    return out


def sph_fit(maps, nside, lmax, lam, peak_weight_alpha=0.0, peak_weight_gamma=2.0):
    """
    Convenience wrapper: fit alms then synthesize full maps.

    Parameters
    ----------
    maps : (n_maps, npix) or (npix,)
    nside : int
    lmax : int
    lam : float
        Regularization strength.
    peak_weight_alpha, peak_weight_gamma : float
        Optional peak upweighting parameters passed to fit_alms_from_maps.

    Returns
    -------
    filled_maps : np.ndarray, shape (n_maps, npix)
    """
    alms = fit_alms_from_maps(
        maps=maps,
        nside=nside,
        lmax=lmax,
        lam=lam,
        p=2,
        fit_log=True,
        eps=1e-12,
        peak_weight_alpha=peak_weight_alpha,
        peak_weight_gamma=peak_weight_gamma,
    )
    return alms_to_filled_maps(
        alms,
        nside=nside,
        lmax=lmax,
        fit_log=True,
        clamp_known=False,
        original_maps=maps,
    )
