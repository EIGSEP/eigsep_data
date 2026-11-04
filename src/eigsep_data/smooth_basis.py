"""Smooth models of a spectrum across frequency, and their held-out score.

A bench spectrum (the noise source's excess, for one) is fitted with a
small linear basis by weighted least squares. The bases:

- :func:`poly_logf_design`: a polynomial in ``ln(f / f0)``;
- :func:`dpss_design`: discrete prolate spheroidal sequences over a band,
  concentrated within a delay half-width, beside such a polynomial (the
  sequences taper to the band edges, so a bright smooth spectrum needs the
  polynomial to be fitted near them).

:func:`block_cv` scores a basis by holding out contiguous blocks of
channels in turn and predicting them from the rest; it is how a basis is
chosen, so that the choice rests on prediction, not on the fit's own
residuals.
"""

import numpy as np


def poly_logf_design(f_mhz, deg, f0_mhz=120.0):
    """Design matrix of a polynomial of degree ``deg`` in ``ln(f / f0)``,
    columns ``1, u, ..., u**deg``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.log(np.asarray(f_mhz, float) / f0_mhz)
        return np.vander(u, deg + 1, increasing=True)


def dpss_design(
    f_mhz,
    f_lo_mhz,
    f_hi_mhz,
    half_width_ns,
    df_mhz=None,
    poly_deg=None,
    f0_mhz=120.0,
):
    """DPSS basis over ``[f_lo, f_hi]`` with delay half-width
    ``half_width_ns``, evaluated at the channels ``f_mhz``.

    The sequences are computed on a regular grid of spacing ``df_mhz``
    (default: the median spacing of ``f_mhz``) spanning the band and
    interpolated to ``f_mhz``; the first ``ceil(2 NW) + 1`` are kept
    (``NW = half_width * bandwidth``: the Slepian count of
    well-concentrated vectors, plus one). With ``poly_deg``, the columns of
    :func:`poly_logf_design` of that degree come first. Returns
    (len(f_mhz), ncol)."""
    from scipy.signal.windows import dpss

    f = np.asarray(f_mhz, float)
    df = float(np.median(np.diff(np.sort(f)))) if df_mhz is None else df_mhz
    grid = np.arange(f_lo_mhz, f_hi_mhz + 0.5 * df, df)
    nw = half_width_ns * 1e-9 * (grid.size * df * 1e6)
    kmax = max(1, int(np.ceil(2 * nw)) + 1)
    vecs = np.atleast_2d(dpss(grid.size, nw, Kmax=kmax))
    if vecs[0].sum() < 0:
        vecs[0] = -vecs[0]
    out = np.stack([np.interp(f, grid, v) for v in vecs], axis=1)
    if poly_deg is not None:
        out = np.hstack([poly_logf_design(f, poly_deg, f0_mhz), out])
    return out


def wls_fit(design, y, w):
    """Weighted least squares ``y ~ design @ c`` with weights ``w``
    (inverse variances; zero leaves a point out). Returns the
    coefficients, the prediction on every row of ``design`` and the
    coefficients' covariance ``(A^T W A)^-1``."""
    design = np.asarray(design, float)
    w = np.asarray(w, float)
    use = w > 0
    sw = np.sqrt(w[use])
    a = design[use] * sw[:, None]
    y = np.asarray(y, float)[use] * sw
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    cov = np.linalg.pinv(a.T @ a, rcond=1e-15)
    with np.errstate(invalid="ignore", over="ignore"):
        pred = design @ coef
    return coef, pred, cov


def block_cv(
    f_mhz, y, sigma, good, design_fn, block_mhz=12.0, offsets_mhz=(0.0,)
):
    """Held-out-block score of a smooth model.

    The band is cut into contiguous blocks of ``block_mhz``, starting at
    the lowest good channel plus each of ``offsets_mhz``; each block's good
    channels are predicted by ``design_fn(f)`` fitted (weights
    ``1/sigma**2``) to the good channels outside it. Only blocks with good
    channels on both sides are scored: holding out an end block would test
    extrapolation.

    Returns a dict: ``rms`` (in the units of ``y``) and ``chi2`` (per
    point, in ``sigma``) of all held-out residuals, their count ``n``, the
    number of scored blocks ``nblock`` and the standard error ``rms_se``
    of ``rms`` from the block-to-block scatter of the mean square (delta
    method)."""
    f = np.asarray(f_mhz, float)
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    good = np.asarray(good, bool) & np.isfinite(y) & np.isfinite(sigma)
    design = design_fn(f)
    w_all = np.where(good, 1.0 / np.where(good, sigma, 1.0) ** 2, 0.0)
    f_lo, f_hi = f[good].min(), f[good].max()
    sq, chi, msb = [], [], []
    for off in offsets_mhz:
        lo = f_lo + off
        while lo + block_mhz <= f_hi:
            held = good & (f >= lo) & (f < lo + block_mhz)
            below = good & (f < lo)
            above = good & (f >= lo + block_mhz)
            if held.any() and below.any() and above.any():
                _, p, _ = wls_fit(design, y, np.where(held, 0.0, w_all))
                r = (y - p)[held]
                sq.append(r**2)
                chi.append((r / sigma[held]) ** 2)
                msb.append(np.mean(r**2))
            lo += block_mhz
    sq, chi, msb = np.concatenate(sq), np.concatenate(chi), np.array(msb)
    rms = float(np.sqrt(np.mean(sq)))
    se_ms = float(np.std(msb, ddof=1) / np.sqrt(msb.size))
    return {
        "rms": rms,
        "rms_se": se_ms / (2 * rms),
        "chi2": float(np.mean(chi)),
        "n": int(sq.size),
        "nblock": int(msb.size),
    }
