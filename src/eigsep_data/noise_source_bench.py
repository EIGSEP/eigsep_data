"""Measure the noise source's temperature from a bench run.

A bench run switches a hot load, a reference load and the noise source (on
and off) through the RF switch while the hot load steps through temperature
plateaus, with VNA sweeps of every path. :func:`measure` turns its files
into the quantities :func:`eigsep_data.products.noise_source.write` stores:

1. per switch state, dwells (:mod:`eigsep_data.switched`) with RFI flags and
   the rows around VNA sweeps at zero weight;
2. switch cycles, each steady, heating or cooling by the hot load's rate of
   change; only steady cycles are fitted (the thermistor lags the load);
3. the switched ratio Q per cycle and channel
   (:func:`eigsep_cal.dicke.calc_Tant_star`);
4. the mismatch factors at P: the loads' and the receiver's reflections from
   the run's own S11 sweeps (``eigsep_cal.S11``, interpolated to each cycle),
   and the noise source's from a lab measurement;
5. a channel mask: outside the band, a persistent line in any state, fit
   residuals beyond ``chi2_factor`` times the in-band median in radiometer
   units, or a slope error beyond ``max_rel_err``;
6. ``T_NON`` per channel (:mod:`eigsep_cal.noise_source`) and the checks a
   product's ``validation.json`` records.

Which state is hot, which thermistor sits where and the thresholds are the
caller's: they belong to a run, and a script records them.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import switched as sw


@dataclass(frozen=True)
class BenchSetup:
    """How a bench run was wired.

    ``hot``, ``ref``, ``non``, ``noff``: switch states of the hot load, the
    reference load and the noise source on and off. ``hot_s11``, ``ref_s11``:
    the ``eigsep_cal.S11`` DUT names of the loads' ports. ``hot_therm``,
    ``ref_therm``: ``(metadata column of the logged resistance, measured
    pull-up in Ohm)``. ``board_therm_c``: metadata column, deg C, of the
    noise-diode board's thermistor."""

    hot: str
    ref: str
    non: str
    noff: str
    hot_s11: str
    ref_s11: str
    hot_therm: tuple
    ref_therm: tuple
    board_therm_c: str
    antenna: str = "box-air"


def _gamma_at(s11, dut, plane, freqs, t):
    """Reflection of ``dut`` at ``plane`` on channels ``freqs`` (MHz) at
    times ``t``: linear in frequency (NaN outside the sweep), then linear in
    time between sweeps (held before the first and after the last)."""
    ts, g = s11.get_all_s11s(dut, plane)
    ts, g = np.asarray(ts, float), np.asarray(g)
    fv = np.asarray(s11.freqs_mhz, float)
    gc = np.array([np.interp(freqs, fv, x.real, left=np.nan, right=np.nan)
                   + 1j * np.interp(freqs, fv, x.imag, left=np.nan, right=np.nan) for x in g])
    u = np.interp(t, ts, np.arange(ts.size))
    lo = np.floor(u).astype(int)
    hi = np.minimum(lo + 1, ts.size - 1)
    w = (u - lo)[:, None]
    return (1 - w) * gc[lo] + w * gc[hi]


def _on_channels(freqs_mhz, f_meas_mhz, gamma):
    return (np.interp(freqs_mhz, f_meas_mhz, np.real(gamma), left=np.nan, right=np.nan)
            + 1j * np.interp(freqs_mhz, f_meas_mhz, np.imag(gamma), left=np.nan, right=np.nan))


def _weighted_mean(x, w):
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(w[ok] * x[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def measure(index, s11, gamma_ns, setup, *, steady_per_min=1.0, dwell_gap_s=5.0,
            sweep_guard=sw.SWEEP_GUARD_S, band_mhz=(50.0, 245.0), chi2_factor=3.0,
            max_rel_err=0.05, enbw_hz=250e6 / 1024):
    """The noise source's temperature at P from a bench run.

    Parameters
    ----------
    index : eigsep_data.MetadataIndex
        Over the run's correlator files.
    s11 : eigsep_cal.S11
        The run's sweeps, calibrated (``s11.calibrate()``).
    gamma_ns : dict
        Lab measurement of the noise-source path's reflection at P:
        ``freqs_mhz``, ``on`` and ``off`` (complex). ``on`` is used; ``off``
        enters the validation (they should agree, the diode being behind a
        pad).
    setup : BenchSetup
    steady_per_min, dwell_gap_s, sweep_guard, band_mhz, chi2_factor,
    max_rel_err, enbw_hz
        The run's choices (module docstring).

    Returns
    -------
    dict
        ``freqs``, ``t_non_k``, ``t_non_sigma_k`` (NaN off ``good``),
        ``t_off_k``, ``good``, ``gamma_ns`` (on the channels), ``cycles`` (a
        table of every cycle), ``mask`` (per reason) and ``validation`` (a
        JSON-ready dict).
    """
    from eigsep_cal.dicke import calc_Tant_star, mismatch
    from eigsep_cal.noise_source import (
        fit_source_temperature,
        source_temperature_terms,
        switched_ratio_noise,
    )

    s = setup
    order = (s.ref, s.hot, s.non, s.noff)
    allmeta = index.select().meta
    sweeps = sw.vna_sweeps(allmeta)

    # 1. Dwells, with RFI flags and the sweep guard.
    dwells, persistent, flagged, guarded, freqs = {}, None, {}, {}, None
    for st in order:
        meta, f, spec, _ = sw.state_dwells(index, st, s.antenna, gap_s=dwell_gap_s)
        freqs = f if freqs is None else freqs
        if not np.array_equal(f, freqs):
            raise ValueError(f"{st}: a different frequency axis")
        meta["T_hot"] = sw.thermistor_k(meta[s.hot_therm[0]], s.hot_therm[1])
        meta["T_ref"] = sw.thermistor_k(meta[s.ref_therm[0]], s.ref_therm[1])
        meta["T_board"] = meta[s.board_therm_c].astype(float) + 273.15
        flags, pc = sw.rfi_flags(spec, f)
        near = sw.near_sweep(meta, sweeps, sweep_guard)
        flags[near] = True
        persistent = pc if persistent is None else persistent | pc
        inb = (f >= band_mhz[0]) & (f <= band_mhz[1])
        flagged[st] = float(flags[:, inb].mean())
        guarded[st] = int(near.sum())
        dwells[st] = sw.dwell_means(meta, spec, flags, columns=("T_hot", "T_ref", "T_board"))

    # 2. Cycles, classified by the hot load's rate of change.
    cyc, cp = sw.cycles(dwells, order)
    if len(cyc) == 0:
        raise ValueError("no complete switch cycle")
    for role, st, col in (("T_hot", s.hot, "T_hot"), ("T_ref", s.ref, "T_ref"),
                          ("T_board_off", s.noff, "T_board")):
        cyc[role] = dwells[st][0][col].to_numpy()[cyc[f"i_{st}"].to_numpy()]
    for st in order:
        cyc[f"n_{st}"] = dwells[st][0]["n"].to_numpy()[cyc[f"i_{st}"].to_numpy()]
    cyc["rate_per_min"], cyc["kind"] = sw.classify_rate(cyc["t"], cyc["T_hot"], steady_per_min)
    steady = (cyc["kind"] == "steady").to_numpy()

    # 3-4. Q and the mismatch factors at P.
    q = calc_Tant_star(cp[s.hot], cp[s.non], cp[s.ref], 1.0, 0.0)
    t = cyc["t"].to_numpy(float)
    g_rec = _gamma_at(s11, "rec", "default", freqs, t)
    g_hot = _gamma_at(s11, s.hot_s11, "lna", freqs, t)
    g_ref = _gamma_at(s11, s.ref_s11, "lna", freqs, t)
    g_ns = _on_channels(freqs, gamma_ns["freqs_mhz"], gamma_ns["on"])
    g_ns_off = _on_channels(freqs, gamma_ns["freqs_mhz"], gamma_ns["off"])
    m_hot, m_ref = np.real(mismatch(g_hot, g_rec)), np.real(mismatch(g_ref, g_rec))
    m_ns = np.real(mismatch(g_ns[None, :], g_rec))
    th, tr = cyc["T_hot"].to_numpy()[:, None], cyc["T_ref"].to_numpy()[:, None]

    # 5. Channel mask, from the mismatch-corrected Q fit.
    fit_q = fit_source_temperature(m_hot * th - m_ref * tr, q, steady)
    tau = float(np.median(index.select().meta["integration_time"]))
    sig_q = switched_ratio_noise(cp[s.hot], cp[s.ref], cp[s.non],
                                 cyc[f"n_{s.hot}"].to_numpy()[:, None],
                                 cyc[f"n_{s.ref}"].to_numpy()[:, None],
                                 cyc[f"n_{s.non}"].to_numpy()[:, None], enbw_hz, tau)
    inband = (freqs >= band_mhz[0]) & (freqs <= band_mhz[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        z2 = np.where(steady[:, None], (fit_q["res"] / sig_q) ** 2, np.nan)
        chi2 = np.array([np.nanmean(c) if np.isfinite(c).any() else np.nan for c in z2.T])
        rel = fit_q["sb"] / fit_q["b"]
    mask = {"band": ~inband, "persistent": inband & persistent}
    ref_chi2 = np.nanmedian(chi2[inband & ~persistent])     # median: flagging only
    mask["chi2"] = inband & ~(chi2 <= chi2_factor * ref_chi2)
    mask["rel_err"] = inband & ~((fit_q["b"] > 0) & (rel <= max_rel_err))
    good = ~(mask["band"] | mask["persistent"] | mask["chi2"] | mask["rel_err"])

    # 6. T_NON, and the ramp check: heating and cooling cycles fitted alone.
    x, y = source_temperature_terms(q, m_hot, m_ref, m_ns, th, tr)
    fits = {k: fit_source_temperature(x, y, (cyc["kind"] == k).to_numpy())
            for k in ("steady", "heating", "cooling")}
    f_st = fits["steady"]
    t_off = _weighted_mean(cyc.loc[steady, "T_board_off"].to_numpy(float),
                           np.ones(int(steady.sum())))
    ramp = {}
    for k in ("heating", "cooling"):
        fk = fits[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            zk = (fk["b"] - f_st["b"]) / np.hypot(fk["sb"], f_st["sb"])
        zk = zk[good & np.isfinite(zk)]
        ramp[k] = {"cycles": int((cyc["kind"] == k).sum()),
                   "mean_z": float(np.mean(zk)) if zk.size else None,
                   "p95_abs_z": float(np.percentile(np.abs(zk), 95)) if zk.size else None}

    def band_wmean(lo, hi):
        k = good & (freqs >= lo) & (freqs < hi)
        w = 1.0 / f_st["sb"][k] ** 2
        return {"channels": int(k.sum()),
                "t_non_k": _weighted_mean(f_st["b"][k], w),
                "excess_k": _weighted_mean(f_st["b"][k] - t_off, w),
                "min_excess_k": float(np.min(f_st["b"][k]) - t_off) if k.any() else None,
                "max_excess_k": float(np.max(f_st["b"][k]) - t_off) if k.any() else None}

    validation = {
        "cycles": {k: int((cyc["kind"] == k).sum()) for k in ("steady", "heating", "cooling")},
        "steady_per_min": steady_per_min,
        "vna_sweeps": int(len(sweeps)),
        "rows_excluded_around_sweeps": guarded,
        "sweep_guard_s": list(sweep_guard),
        "flagged_in_band_fraction": flagged,
        "channels_masked": {k: int(v.sum()) for k, v in mask.items()},
        "channels_good": int(good.sum()),
        "median_chi2_in_band": float(ref_chi2),
        "gamma_ns": {
            "max_abs_on_minus_off": float(np.nanmax(np.abs(g_ns - g_ns_off)[inband])),
            "abs_range_in_band": [float(np.nanmin(np.abs(g_ns[inband]))),
                                  float(np.nanmax(np.abs(g_ns[inband])))],
        },
        "loads_at_their_ports_max_abs_gamma": {
            dut: float(np.nanmax(np.abs(np.asarray(s11.get_all_s11s(dut, "dut")[1])[
                :, (np.asarray(s11.freqs_mhz) >= band_mhz[0]) & (np.asarray(s11.freqs_mhz) <= band_mhz[1])])))
            for dut in (s.hot_s11, s.ref_s11)},
        "mismatch_factor_ns_range": [float(np.nanmin(m_ns[:, inband])), float(np.nanmax(m_ns[:, inband]))],
        "ramp_check_heating_cooling_vs_steady": ramp,
        "t_off_k": t_off,
        "bands": {f"{lo:.0f}-{hi:.0f}": band_wmean(lo, hi)
                  for lo, hi in ((50, 88), (108, 150), (150, 200), (200, 245))},
    }
    return {"freqs": freqs,
            "t_non_k": np.where(good, f_st["b"], np.nan),
            "t_non_sigma_k": np.where(good, f_st["sb"], np.nan),
            "t_off_k": t_off, "good": good, "gamma_ns": g_ns,
            "cycles": pd.DataFrame(cyc), "mask": mask, "validation": validation}
