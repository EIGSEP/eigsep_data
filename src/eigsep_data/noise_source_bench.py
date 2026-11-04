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

Options, each off by default (``noise_source`` v0001's behaviour):

- ``load_paths``: the loads' thermistors give their temperatures at their
  own ports, but their noise reaches P through their switch paths. With the
  paths' S-parameters, each load's noise temperature at P is
  ``G T_load + (1 - G) T_path`` (:func:`loads_at_p`, ``eigsep_cal.network
  .embed``), ``G`` the path's available gain for the load's port
  reflection and ``T_path`` the switch board's temperature in that load's
  dwell;
- ``hot_below_k``: fit only steady cycles with the hot load below this
  temperature (the thermistors' specified range);
- ``s11_average``: one reflection per DUT, the mean over the sweeps, instead
  of interpolating between sweeps in time (for S11 sweeps taken in another
  session of the same bench).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import switched as sw
from .numerics import interp_complex, weighted_mean
from .units import celsius_to_kelvin


@dataclass(frozen=True)
class BenchSetup:
    """How a bench run was wired.

    ``hot``, ``ref``, ``non``, ``noff``: switch states of the hot load, the
    reference load and the noise source on and off. ``hot_s11``, ``ref_s11``:
    the ``eigsep_cal.S11`` DUT names of the loads' ports. ``hot_therm``,
    ``ref_therm``: ``(metadata column of the logged resistance, measured
    pull-up in Ohm)``. ``board_therm_c``: metadata column, deg C, of the
    noise-diode board's thermistor. ``path_therm_c``: metadata columns,
    deg C, whose mean is the switch paths' physical temperature (used only
    with ``measure(..., load_paths=...)``)."""

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
    path_therm_c: tuple = ()


def _on_cycles(ts, g, fv, freqs, t, average=False):
    """Per-sweep curves ``g`` (nsweep, len(fv)) on channels ``freqs`` (MHz)
    at times ``t``: linear in frequency (NaN outside the sweep), then linear
    in time between sweeps (held before the first and after the last), or,
    with ``average``, the mean over the sweeps for every time."""
    ts, g = np.asarray(ts, float), np.asarray(g)
    if average:
        gc = interp_complex(freqs, fv, g.mean(axis=0))
        return np.broadcast_to(gc, (np.size(t), gc.size))
    gc = interp_complex(freqs, fv, g)
    u = np.interp(t, ts, np.arange(ts.size))
    lo = np.floor(u).astype(int)
    hi = np.minimum(lo + 1, ts.size - 1)
    w = (u - lo)[:, None]
    return (1 - w) * gc[lo] + w * gc[hi]


def _gamma_at(s11, dut, plane, freqs, t, average=False):
    """Reflection of ``dut`` at ``plane`` on channels ``freqs`` (MHz) at
    times ``t`` (:func:`_on_cycles`)."""
    ts, g = s11.get_all_s11s(dut, plane)
    fv = np.asarray(s11.freqs_mhz, float)
    return _on_cycles(ts, g, fv, freqs, t, average)


def _gain_at(s11, dut, sparams, freqs, t, average=False):
    """Available gain of ``dut``'s switch path (:func:`path_gain`) for its
    port reflection (``plane="dut"``), on channels ``freqs`` at times
    ``t``: per sweep, then as :func:`_on_cycles`; with ``average``, from
    the mean port reflection over the sweeps. Returns (len(t), nchan)."""
    ts, g = s11.get_all_s11s(dut, "dut")
    g = np.asarray(g)
    fv = np.asarray(s11.freqs_mhz, float)
    if average:
        gain = path_gain(g.mean(axis=0), sparams)
        on = np.interp(freqs, fv, gain, left=np.nan, right=np.nan)
        return np.broadcast_to(on, (np.size(t), on.size))
    gain = path_gain(g, sparams)
    return _on_cycles(ts, gain, fv, freqs, t).real


def path_gain(gamma_port, sparams):
    """Available gain of a switch path from its port (port 2) to P (port
    1) for a termination of reflection ``gamma_port``
    (``eigsep_cal.network.embed``; Monsalve et al. 2024 eqs. 16-17).

    ``sparams`` is the path's ``(S11, S12S21, S22)``, the rows of a
    ``switch_sparams*.npz`` entry, on the same frequencies as
    ``gamma_port`` (..., nfreq). Returns ``G``, real."""
    from eigsep_cal.network import SParams, embed

    s11, s12s21, s22 = (np.asarray(x, complex) for x in sparams)
    sp = SParams(s11=s11, s12s21=s12s21, s22=s22, name="path")
    _, g = embed(np.asarray(gamma_port, complex), 1.0, sp, 0.0)
    return np.asarray(g, float)


def loads_at_p(t_load_k, gain, t_path_k):
    """A load's noise temperature at P through a path of available gain
    ``gain``: ``G T_load + (1 - G) T_path`` (Monsalve et al. 2017 eq. 8).
    ``t_load_k`` and ``t_path_k`` (ncycle,) are per cycle; ``gain`` is
    (nchan,) or (ncycle, nchan). Returns (ncycle, nchan)."""
    t_load = np.asarray(t_load_k, float)[:, None]
    t_path = np.asarray(t_path_k, float)[:, None]
    return gain * t_load + (1.0 - gain) * t_path


def fitted_cycles(kind, t_hot_k, hot_below_k=None):
    """The cycles a fit uses: steady ones, and with ``hot_below_k`` only
    those whose hot load is below it."""
    use = np.asarray(kind) == "steady"
    if hot_below_k is not None:
        use &= np.asarray(t_hot_k, float) < hot_below_k
    return use


def measure(index, s11, gamma_ns, setup, *, steady_per_min=1.0, dwell_gap_s=5.0,
            sweep_guard=sw.SWEEP_GUARD_S, band_mhz=(50.0, 245.0), chi2_factor=3.0,
            max_rel_err=0.05, enbw_hz=None,
            report_bands_mhz=((50.0, 88.0), (108.0, 150.0), (150.0, 200.0),
                              (200.0, 245.0)),
            load_paths=None, hot_below_k=None, s11_average=False):
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
        The run's choices (module docstring). ``enbw_hz`` defaults to
        ``eigsep_cal.conventions.D5_ENBW_HZ``.
    report_bands_mhz : sequence of (lo, hi)
        Sub-bands whose weighted-mean temperatures the validation reports
        (``validation["bands"]``, keyed ``"lo-hi"``); they affect nothing else.
    load_paths : dict, optional
        ``{"hot": sparams, "ref": sparams}``, each the switch path's
        ``(S11, S12S21, S22)`` on ``s11.freqs_mhz`` (P on port 1, the load's
        port on port 2): embed the loads to P through their paths
        (:func:`loads_at_p`), with the paths at the mean of
        ``setup.path_therm_c``. Default: the loads' thermistors are taken
        as their temperatures at P (no path loss).
    hot_below_k : float, optional
        Fit only steady cycles with the hot load below this (K).
    s11_average : bool
        Use each DUT's mean reflection over the sweeps for every cycle.

    Returns
    -------
    dict
        ``freqs``, ``t_non_k``, ``t_non_sigma_k`` (NaN off ``good``),
        ``t_off_k``, ``good``, ``gamma_ns`` (on the channels), ``cycles`` (a
        table of every cycle), ``mask`` (per reason), ``validation`` (a
        JSON-ready dict); for diagnostics, the fits' inputs ``q`` and
        ``mismatch`` (``hot``, ``ref``, ``ns``), each cycles x channels,
        and with ``load_paths`` the paths' available gains ``path_gain``
        (``hot``, ``ref``), cycles x channels.
    """
    from eigsep_cal.conventions import D5_ENBW_HZ
    from eigsep_cal.dicke import calc_Tant_star, mismatch
    from eigsep_cal.noise_source import (
        fit_source_temperature,
        source_temperature_terms,
        switched_ratio_noise,
    )

    enbw_hz = D5_ENBW_HZ if enbw_hz is None else enbw_hz
    s = setup
    if load_paths is not None and not s.path_therm_c:
        raise ValueError("load_paths needs setup.path_therm_c")
    order = (s.ref, s.hot, s.non, s.noff)
    columns = ("T_hot", "T_ref", "T_board")
    if load_paths is not None:
        columns += ("T_path",)
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
        meta["T_board"] = celsius_to_kelvin(meta[s.board_therm_c])
        if load_paths is not None:
            board = [pd.to_numeric(meta[c], errors="coerce").to_numpy(float)
                     for c in s.path_therm_c]
            meta["T_path"] = celsius_to_kelvin(np.mean(board, axis=0))
        flags, pc = sw.rfi_flags(spec, f)
        near = sw.near_sweep(meta, sweeps, sweep_guard)
        flags[near] = True
        persistent = pc if persistent is None else persistent | pc
        inb = (f >= band_mhz[0]) & (f <= band_mhz[1])
        flagged[st] = float(flags[:, inb].mean())
        guarded[st] = int(near.sum())
        dwells[st] = sw.dwell_means(meta, spec, flags, columns=columns)

    # 2. Cycles, classified by the hot load's rate of change.
    cyc, cp = sw.cycles(dwells, order)
    if len(cyc) == 0:
        raise ValueError("no complete switch cycle")
    for role, st, col in (("T_hot", s.hot, "T_hot"), ("T_ref", s.ref, "T_ref"),
                          ("T_board_off", s.noff, "T_board")):
        cyc[role] = dwells[st][0][col].to_numpy()[cyc[f"i_{st}"].to_numpy()]
    if load_paths is not None:
        for role, st in (("T_path_hot", s.hot), ("T_path_ref", s.ref)):
            i = cyc[f"i_{st}"].to_numpy()
            cyc[role] = dwells[st][0]["T_path"].to_numpy()[i]
    for st in order:
        cyc[f"n_{st}"] = dwells[st][0]["n"].to_numpy()[cyc[f"i_{st}"].to_numpy()]
    cyc["rate_per_min"], cyc["kind"] = sw.classify_rate(cyc["t"], cyc["T_hot"], steady_per_min)
    steady = (cyc["kind"] == "steady").to_numpy()
    use = fitted_cycles(cyc["kind"], cyc["T_hot"], hot_below_k)

    # 3-4. Q and the mismatch factors at P.
    q = calc_Tant_star(cp[s.hot], cp[s.non], cp[s.ref], 1.0, 0.0)
    t = cyc["t"].to_numpy(float)
    avg = s11_average
    g_rec = _gamma_at(s11, "rec", "default", freqs, t, avg)
    g_hot = _gamma_at(s11, s.hot_s11, "lna", freqs, t, avg)
    g_ref = _gamma_at(s11, s.ref_s11, "lna", freqs, t, avg)
    g_ns = interp_complex(freqs, gamma_ns["freqs_mhz"], gamma_ns["on"])
    g_ns_off = interp_complex(freqs, gamma_ns["freqs_mhz"], gamma_ns["off"])
    m_hot, m_ref = np.real(mismatch(g_hot, g_rec)), np.real(mismatch(g_ref, g_rec))
    m_ns = np.real(mismatch(g_ns[None, :], g_rec))
    th, tr = cyc["T_hot"].to_numpy()[:, None], cyc["T_ref"].to_numpy()[:, None]
    gains = None
    if load_paths is not None:
        gains = {
            role: _gain_at(s11, dut, load_paths[role], freqs, t, avg)
            for role, dut in (("hot", s.hot_s11), ("ref", s.ref_s11))
        }
        th = loads_at_p(cyc["T_hot"], gains["hot"], cyc["T_path_hot"])
        tr = loads_at_p(cyc["T_ref"], gains["ref"], cyc["T_path_ref"])

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
            for k in ("heating", "cooling")}
    fits["steady"] = fit_source_temperature(x, y, use)
    f_st = fits["steady"]
    t_off = float(weighted_mean(cyc.loc[use, "T_board_off"].to_numpy(float),
                                np.ones(int(use.sum()))))
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
                "t_non_k": float(weighted_mean(f_st["b"][k], w)),
                "excess_k": float(weighted_mean(f_st["b"][k] - t_off, w)),
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
                  for lo, hi in report_bands_mhz},
    }
    if hot_below_k is not None:
        validation["hot_below_k"] = float(hot_below_k)
        validation["cycles_fitted"] = int(use.sum())
    if s11_average:
        validation["s11"] = "mean over each DUT's sweeps"
    if gains is not None:
        validation["load_paths"] = {
            role: {
                "path_temperature_k": [
                    float(np.nanmin(cyc[col][use])),
                    float(np.nanmax(cyc[col][use])),
                ],
                "available_gain_in_band": [
                    float(np.nanmin(gains[role][:, inband])),
                    float(np.nanmax(gains[role][:, inband])),
                ],
            }
            for role, col in (("hot", "T_path_hot"), ("ref", "T_path_ref"))
        }
    return {"freqs": freqs,
            "t_non_k": np.where(good, f_st["b"], np.nan),
            "t_non_sigma_k": np.where(good, f_st["sb"], np.nan),
            "t_off_k": t_off, "good": good, "gamma_ns": g_ns,
            "cycles": pd.DataFrame(cyc), "mask": mask, "validation": validation,
            "q": q, "mismatch": {"hot": m_hot, "ref": m_ref, "ns": m_ns},
            **({"path_gain": gains} if gains is not None else {})}
