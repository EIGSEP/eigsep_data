"""Phase a cross-correlation to a point source and find its delay.

Three steps, in the order a caller uses them:

1. :func:`digital_delay_ns`: the SNAP's digital delay per integration,
   from ``header/pol_delay`` with the deployment-5 header fixes.
2. :func:`direct_delay_ns`: the delay of a source's direct path,
   ``tau(t) = tau_instr - D(t) - b . s(t) / c``.
3. :func:`matched_filter`: the coherent matched filter along that
   track, which returns the remaining constant delay (``tau_instr``),
   its phase and a jackknife error.

Sign convention: ``V_ab = x_a conj(x_b)`` (``data/04`` is
``x_0 conj(x_4)``), so a signal that reaches input ``b`` a time ``tau``
after input ``a`` gives ``V_ab ~ exp(+2 pi i nu tau)``. Delaying input
``a`` by ``D`` multiplies ``V_ab`` by ``exp(-2 pi i nu D)``. With the
baseline ``b = r_b - r_a`` (true ENU, m), a plane wave from the unit
vector ``s`` reaches ``a`` later by ``b . s / c``, which acts like a
digital delay on ``a``. ``tau_instr`` is input ``b``'s signal-chain
delay minus input ``a``'s. In D5 phase C, ``a`` = 0 (box-gnd), ``b`` =
4 (box-air), and the digital delay is on pair 01, i.e. on box-gnd.
"""

import json
from pathlib import Path

import h5py
import numpy as np
from eigsep_base.const import MARJUM_PASS
from eigsep_base.const import c as C_LIGHT

#: Digital delay per ``pol_delay`` count: one 250-MHz FPGA fabric cycle
#: (2 ADC samples), applied to the ADC samples before the PFB.
NS_PER_COUNT = 4.0

#: Integrations whose ``header/pol_delay`` is wrong in D5, found from
#: the steps in V04's delay across all phase-C files.
#: ``(file, pair) -> ((first_row, counts), ...)``: rows from
#: ``first_row`` on ran at ``counts`` (None: unknown), until the next
#: entry. Rows before the first entry keep the header's value.
D5_POL_DELAY_FIXES = {
    # The delay was set from 0 to 250 counts during this file, near
    # integration 205; the header reads 0. Rows 200-209 bracket the
    # change.
    ("corr_20260715_044951Z.h5", "01"): ((200, None), (210, 250)),
    # Restart files whose headers fell back to the corr_config.yaml
    # defaults (0). V04's delay does not step at either boundary, so
    # the hardware stayed at 250.
    ("corr_20260716_012224Z.h5", "01"): ((0, 250),),
    ("corr_20260717_032454Z.h5", "01"): ((0, 250),),
}

#: Cyg A (ICRS), the position of Perley & Butler (2017, ApJS 230, 7).
CYG_A = ("19h59m28.35s", "40d44m02.1s")


def digital_delay_ns(path, pair="01", fixes=D5_POL_DELAY_FIXES):
    """Digital delay of ``pair`` per integration of a corr file [ns].

    Reads ``header/pol_delay`` (JSON, e.g. ``{"01": 250, ...}``),
    applies ``fixes`` for this file's name and pair, and multiplies by
    :data:`NS_PER_COUNT`. Returns ``(n_int,)`` floats, NaN where the
    delay is unknown.
    """
    with h5py.File(path, "r") as h5:
        counts = json.loads(h5["header"]["pol_delay"][()])[pair]
        n = h5["header"]["times"].shape[0]
    d = np.full(n, float(counts))
    for first, value in fixes.get((Path(path).name, pair), ()):
        d[first:] = np.nan if value is None else float(value)
    return NS_PER_COUNT * d


def source_enu(ra, dec, times_unix, site=MARJUM_PASS):
    """True-ENU unit vectors toward an ICRS position, ``(n_time, 3)``.

    ``ra`` and ``dec`` are anything ``SkyCoord`` reads in degrees or
    with explicit units (e.g. :data:`CYG_A`); ``site`` is (lat deg, lon
    deg, height m). Astropy AltAz without refraction.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    loc = EarthLocation(
        lat=site[0] * u.deg, lon=site[1] * u.deg, height=site[2] * u.m
    )
    t = Time(np.asarray(times_unix, float), format="unix")
    # The bundled IERS tables, however old their predictions: polar
    # motion and UT1 errors move a direction by < 15 arcsec, < 0.01 ns
    # on 100 m, and a download would need the network.
    with (
        iers.conf.set_temp("auto_download", False),
        iers.conf.set_temp("auto_max_age", None),
    ):
        aa = SkyCoord(ra, dec, unit="deg", frame="icrs").transform_to(
            AltAz(location=loc, obstime=t)
        )
    alt, az = aa.alt.rad, aa.az.rad
    return np.stack(
        [np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)],
        -1,
    )


def direct_delay_ns(baseline_m, s_enu, d_ns=0.0, tau_instr_ns=0.0):
    """Delay of the direct path, ``tau_instr - D - b . s / c`` [ns].

    ``baseline_m`` is ``r_b - r_a`` (true ENU, m), ``s_enu`` the source
    direction(s) ``(..., 3)`` and ``d_ns`` the digital delay on input
    ``a`` (minus any on ``b``), broadcast against ``s_enu[..., 0]``.
    """
    geo = np.asarray(s_enu) @ np.asarray(baseline_m, float) / C_LIGHT
    return tau_instr_ns - np.asarray(d_ns) - geo * 1e9


def highpass(x, w, nwin):
    """Remove the weighted running mean over ``nwin`` bins, per channel.

    ``x`` is ``(n_bin, n_chan)``, real or complex; ``w`` its weights,
    ``(n_bin, n_chan)`` or ``(n_bin,)``. The window is centred and
    truncated at the edges. Bins without weight take no part in the
    mean; their values are returned shifted but carry no weight later.
    """
    x = np.asarray(x)
    w = np.broadcast_to(np.asarray(w, float).reshape(len(x), -1), x.shape)
    xw = np.where(w > 0, x, 0) * w
    cx = np.concatenate([np.zeros_like(xw[:1]), np.cumsum(xw, 0)])
    cw = np.concatenate([np.zeros_like(w[:1]), np.cumsum(w, 0)])
    i = np.arange(len(x))
    lo = np.clip(i - nwin // 2, 0, len(x))
    hi = np.clip(i + nwin // 2 + 1, 0, len(x))
    den = cw[hi] - cw[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return x - np.where(den > 0, (cx[hi] - cx[lo]) / den, 0)


def _peak(tau, s):
    """Delay and ``|S|`` of the largest ``|S|``, parabolically refined."""
    a = np.abs(s)
    i = int(np.argmax(a))
    if not 0 < i < a.size - 1:
        return float(tau[i]), float(a[i])
    y0, y1, y2 = a[i - 1], a[i], a[i + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float(tau[i] + d * (tau[1] - tau[0])), float(
        y1 - 0.25 * (y0 - y2) * d
    )


def matched_filter(c, w, nu_hz, track_ns, tau_ns, nwin, blocks=None):
    """Coherent matched filter of a cross along a delay track.

    The template ``m = exp(2 pi i nu track(t))`` and the data ``c`` (a
    coherence ``V_ab / sqrt(P_a P_b)``, NaN where flagged) go through
    the same :func:`highpass` (window ``nwin`` bins), giving ``c'`` and
    ``m'``. Then

        S(tau) = sum_k exp(-2 pi i nu_k tau) sum_t w c' conj(m')
                 / sum_tk w |m'|^2,

    so ``c = A exp(i phi) exp(2 pi i nu (track + tau0))`` gives
    ``S(tau0) = A exp(i phi)``. With ``track_ns`` from
    :func:`direct_delay_ns` at ``tau_instr_ns=0``, ``tau0`` is
    ``tau_instr``.

    Parameters
    ----------
    c, w : (n_bin, n_chan) arrays
        Data and weights (e.g. integration seconds after masks).
    nu_hz : (n_chan,) array
    track_ns : (n_bin,) or (n_bin, n_chan) array
        Model delay per bin; NaN bins get no weight.
    tau_ns : (n_tau,) array
        Uniform delay grid to search.
    nwin : int
        High-pass window in bins.
    blocks : (n_bin,) int array, optional
        Block of each bin for the delete-one-block jackknife (e.g.
        30-min blocks); without it, the errors are NaN.

    Returns
    -------
    dict
        ``tau`` (peak delay, ns), ``amp`` (``|S|`` there), ``phase``
        (``arg S`` there, rad), ``sigma_tau``, ``sigma_amp`` (jackknife)
        and ``profile`` (complex ``S`` on ``tau_ns``).
    """
    c = np.asarray(c)
    nu = np.asarray(nu_hz, float)
    track = np.asarray(track_ns, float)
    if track.ndim == 1:
        track = track[:, None]
    w = np.where(np.isfinite(c) & np.isfinite(track), w, 0.0)
    cp = highpass(c, w, nwin)
    mp = highpass(
        np.exp(2j * np.pi * nu * np.nan_to_num(track) * 1e-9), w, nwin
    )
    blocks = np.zeros(len(c), int) if blocks is None else np.asarray(blocks)
    nblock = int(blocks.max()) + 1
    z = np.zeros((nblock, nu.size), complex)
    n = np.zeros(nblock)
    np.add.at(
        z, blocks, np.where(w > 0, w * np.nan_to_num(cp), 0) * np.conj(mp)
    )
    np.add.at(n, blocks, (w * np.abs(mp) ** 2).sum(1))
    blocks_all = np.arange(nblock)
    tau = np.asarray(tau_ns, float)
    e = np.exp(-2j * np.pi * np.outer(tau * 1e-9, nu))

    def fit(keep):
        return _peak(tau, e @ z[keep].sum(0) / n[keep].sum())

    used = n > 0
    prof = e @ z[used].sum(0) / n[used].sum()
    t0, a0 = _peak(tau, prof)
    s0 = np.exp(-2j * np.pi * nu * t0 * 1e-9) @ z[used].sum(0)
    # Delete-one-block jackknife, re-maximising in each replicate.
    sig = np.full(2, np.nan)
    idx = np.flatnonzero(used)
    if idx.size > 1:
        reps = np.array([fit(used & (blocks_all != j)) for j in idx])
        f = (idx.size - 1) / idx.size
        sig = np.sqrt(f * ((reps - reps.mean(0)) ** 2).sum(0))
    return dict(
        tau=t0,
        amp=a0,
        phase=float(np.angle(s0)),
        sigma_tau=float(sig[0]),
        sigma_amp=float(sig[1]),
        profile=prof,
    )
