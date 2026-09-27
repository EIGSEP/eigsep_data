"""Background under the transmitter comb teeth.

A transmitter comb puts a tooth on every ``spacing``-th channel. To isolate a
tooth, the sky, instrument and transmitter broadband-noise background under it
must be interpolated from the channels around it. Two methods:

* ``method='dpss'`` (default, lower noise). For each spectrum, the channels in
  a window are fitted with DPSS modes of a set delay half-width, with each
  tooth and its ``guard`` neighbours (filterbank leakage) and any ``exclude``
  channels masked. Outliers on either side are clipped symmetrically, and the
  fit is evaluated at the tooth. Modes are kept down to a spectral
  concentration of ``concentration_min`` (default 1e-6, about 2NW + 6 modes),
  which represents a constant or a power law across the window to ~1e-4.
  ``concentration_min=None`` keeps floor(2NW) + 1 modes, the rule used for beam
  fits v0009-v0011. It cannot represent a constant near the window edges
  (errors of several percent) and is kept only to reproduce them.
* ``method='gap'``. The mean of the two channels ``gap_offset`` either side of
  the tooth (half the comb spacing by default). It is exact for any background
  linear across ``2 * gap_offset`` channels, but it carries sqrt(1.5) of the
  per-channel noise, and a ripple of delay tau is misestimated by the fraction
  ``1 - cos(2 pi tau gap_offset dnu)`` of its amplitude. That fraction is
  largest near the comb-mask alias, delay 1 / (2 gap_offset dnu).

Symmetric clipping matters. The transmitter's broadband amplifier noise is part
of the background under a tooth. A fit that rejects only positive excess, as
the v3-beta RFI flagger does, sits on the lower envelope of unmodelled ripple
and is biased low by several percent. The default half-width (150 ns) and
window (48 MHz) were chosen for Marjum 2026-07 in
``data-analysis/notebooks/arp/marjum-2026-07/debug/tooth_background_*debug.ipynb``:
150 ns covers the ~110 ns instrument ripple, and 48 MHz separates structure near
550 ns from the 512 ns alias of an 8-channel comb mask. Wider bases extrapolate
badly across the masked hole.

Public API
----------
dpss_modes
robust_fit
background_windows
gap_background
tooth_background
"""

import warnings

import numpy as np
from scipy.signal.windows import dpss


def dpss_modes(n, halfwidth_ns, channel_mhz, concentration_min=1e-6):
    """DPSS modes over ``n`` contiguous channels of width ``channel_mhz`` with delay
    half-width ``halfwidth_ns``: ``(n, K)``.

    Modes are kept while their spectral concentration is at least
    ``concentration_min``. ``None`` keeps K = floor(2 N W) + 1 instead (the
    v0009-v0011 rule, which under-represents smooth functions near the edges).
    """
    nw = halfwidth_ns * n * channel_mhz * 1e-3          # ns x GHz
    if concentration_min is None:
        k = int(np.floor(2 * nw)) + 1
        return dpss(n, nw, Kmax=k).T if k > 1 else dpss(n, nw)[:, None]
    kmax = min(n, int(np.floor(2 * nw)) + 40)
    modes, ratios = dpss(n, nw, Kmax=kmax, return_ratios=True)
    return modes[ratios >= concentration_min].T


def robust_fit(y, A, base, clip=4.0, iters=6):
    """Least squares of ``y`` on ``A`` over the channels in ``base``, dropping
    ``|residual| > clip * sigma`` (1.4826 MAD) on either side until the mask stops
    changing. Returns the prediction on all channels and the final fit mask."""
    use = base.copy()
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[use], y[use], rcond=None)
        r = y - A @ coef
        sigma = 1.4826 * np.median(np.abs(r[use]))
        new = base & (np.abs(r) <= clip * sigma)
        if np.array_equal(new, use):
            break
        use = new
    return A @ coef, use


def background_windows(freqs_mhz, teeth, window_mhz=48.0, step_mhz=24.0, limits_mhz=None):
    """Fit windows and the teeth each evaluates: a list of ``(channel indices, teeth)``.

    Window centres step by ``step_mhz`` within ``limits_mhz`` (default the full
    band); each tooth is evaluated in the window whose centre is nearest.
    """
    freqs_mhz = np.asarray(freqs_mhz, float)
    teeth = np.asarray(teeth)
    lo, hi = (freqs_mhz[0], freqs_mhz[-1]) if limits_mhz is None else limits_mhz
    centres = np.arange(lo + window_mhz / 2, hi - window_mhz / 2 + step_mhz, step_mhz)
    centres = np.clip(centres, lo + window_mhz / 2, hi - window_mhz / 2)
    nearest = np.argmin(np.abs(freqs_mhz[teeth][:, None] - centres[None, :]), axis=1)
    out = []
    for k, c in enumerate(centres):
        ev = teeth[nearest == k]
        if len(ev):
            idx = np.flatnonzero((freqs_mhz >= c - window_mhz / 2) & (freqs_mhz < c + window_mhz / 2))
            out.append((idx, ev))
    return out


def gap_background(spectra, teeth, gap_offset=4, exclude=None, return_residual_rms=False):
    """Mean of the channels ``gap_offset`` either side of each tooth.

    Where one of the two is excluded (or off the grid), the other is used alone;
    where both are, the result is NaN. With ``return_residual_rms``, also returns
    ``|D[c-g] - D[c+g]| / (D[c-g] + D[c+g])``, the local asymmetry of the two
    gap channels (a slope-and-noise measure comparable in role to the DPSS
    residual rms).
    """
    spectra = np.asarray(spectra, float)
    teeth = np.asarray(teeth)
    nchan = spectra.shape[1]
    usable = np.ones(nchan, bool) if exclude is None else ~np.asarray(exclude, bool)
    lo, hi = teeth - gap_offset, teeth + gap_offset
    ok_lo = (lo >= 0) & usable[np.clip(lo, 0, nchan - 1)]
    ok_hi = (hi < nchan) & usable[np.clip(hi, 0, nchan - 1)]
    a = np.where(ok_lo[None, :], spectra[:, np.clip(lo, 0, nchan - 1)], np.nan)
    b = np.where(ok_hi[None, :], spectra[:, np.clip(hi, 0, nchan - 1)], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # both sides excluded -> NaN
        out = np.nanmean(np.stack([a, b]), axis=0)
    if not return_residual_rms:
        return out
    return out, np.abs(a - b) / (a + b)


def tooth_background(spectra, freqs_mhz, teeth, spacing=8, guard=1, exclude=None,
                     halfwidth_ns=150.0, window_mhz=48.0, step_mhz=24.0, limits_mhz=None,
                     clip=4.0, return_residual_rms=False, method='dpss',
                     concentration_min=1e-6, gap_offset=None):
    """Background at each tooth for each spectrum.

    Parameters
    ----------
    spectra : (nrow, nchan) power, on the uniform channel grid ``freqs_mhz``.
    teeth : channel indices of the teeth to evaluate (each a multiple of ``spacing``).
    spacing : comb spacing in channels; every multiple of it is masked as a tooth.
    guard : channels either side of every tooth also masked.
    exclude : (nchan,) bool, channels never used in fits (e.g. an FM band).
    method : 'dpss' or 'gap' (see the module docstring).
    concentration_min : DPSS mode concentration cutoff (``None``: the v0009-v0011 rule).
    gap_offset : channels either side for ``method='gap'`` (default ``spacing // 2``).

    Returns
    -------
    background : (nrow, len(teeth)), NaN where no window covers a tooth or a fit fails.
    With ``return_residual_rms``, also (nrow, len(teeth)): the rms of the fractional
    fit residual on the off-tooth channels 2..4 either side of each tooth, a
    per-tooth measure of how well the background fits locally.
    """
    if method == 'gap':
        return gap_background(spectra, teeth, spacing // 2 if gap_offset is None else gap_offset,
                              exclude, return_residual_rms)
    if method != 'dpss':
        raise ValueError("method must be 'dpss' or 'gap'")
    spectra = np.asarray(spectra, float)
    freqs_mhz = np.asarray(freqs_mhz, float)
    teeth = np.asarray(teeth)
    nchan = freqs_mhz.size
    channel_mhz = float(np.median(np.diff(freqs_mhz)))
    offset = (np.arange(nchan) + spacing // 2) % spacing - spacing // 2
    masked = np.abs(offset) <= guard
    if exclude is not None:
        masked = masked | np.asarray(exclude, bool)
    col = {c: i for i, c in enumerate(teeth)}
    out = np.full((len(spectra), len(teeth)), np.nan)
    local = np.full_like(out, np.nan)
    for idx, ev in background_windows(freqs_mhz, teeth, window_mhz, step_mhz, limits_mhz):
        A = dpss_modes(len(idx), halfwidth_ns, channel_mhz, concentration_min)
        base0 = ~masked[idx]
        pos = [c - idx[0] for c in ev]
        near = [np.r_[p - 4:p - 1, p + 2:p + 5] for p in pos]
        for r, row in enumerate(spectra[:, idx]):
            base = base0 & np.isfinite(row) & (row > 0)
            if base.sum() < A.shape[1] + 5:
                continue
            scale = np.median(row[base])
            pred, _ = robust_fit(row / scale, A, base, clip)
            for c, p, nb in zip(ev, pos, near):
                out[r, col[c]] = pred[p] * scale
                if return_residual_rms:
                    nb = nb[(nb >= 0) & (nb < len(idx))]
                    x = row[nb] / scale / pred[nb] - 1
                    local[r, col[c]] = np.sqrt(np.mean(x ** 2))
    return (out, local) if return_residual_rms else out
