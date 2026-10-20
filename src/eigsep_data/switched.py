"""Switched calibration measurements: dwells, cycles and their sensors.

The RF switch steps through a fixed sequence of states, a few integrations
each. These helpers turn that into the units a calibration uses, for the
field calibration visits and for bench runs alike:

- :func:`thermistor_k`: a tempctrl thermistor's temperature from its logged
  resistance and the *measured* pull-up (the firmware assumes 10 kOhm);
- :func:`state_dwells`: one switch state's integrations, grouped into dwells
  (contiguous runs, :meth:`Selection.visits` with a short gap);
- :func:`rfi_flags`: per-sample flags from :mod:`eigsep_data.flagging`'s
  detectors (int32 wraps, transients) and per-channel persistent lines;
- :func:`vna_sweeps` and :func:`near_sweep`: the VNA sweeps, and the rows
  around them that the VNA service disturbs;
- :func:`dwell_means`: each dwell's integration-time-weighted mean spectrum
  with flagged samples at zero weight, and weighted means of chosen
  metadata columns;
- :func:`cycles`: dwells of several states grouped into switch cycles;
- :func:`classify_rate`: each cycle steady, heating or cooling by a
  temperature's rate of change.

Which state is which load, and which thermistor sits where, are arguments:
they differ between the field and each bench run, and are recorded with
each run, not here.
"""

import numpy as np
import pandas as pd

from .flagging.detectors import overflow_mask, persistent_track, transient_track

#: Steinhart-Hart coefficients (1/T = A + B ln R + C ln^3 R, T in K, R in
#: Ohm) of the YSI 44909 (30 kOhm at 25 C, curve H), fit to YSI's table over
#: 0-110 C (residual <= 0.019 K); pico-firmware tempctrl.h LOAD_THERM_SH_*.
SH_YSI_44909 = (9.362744897e-4, 2.209138866e-4, 1.280636863e-7)

#: The pull-up the tempctrl firmware assumes when it computes `resistance`.
FIRMWARE_PULLUP_OHM = 10000.0

#: Rows this long before a VNA sweep starts and after it ends are disturbed:
#: on the 2026-10-08 bench run every integration in the ~17 s before many
#: sweeps read +120 % to +280 % in whatever state the switch was on.
SWEEP_GUARD_S = (20.0, 5.0)


def thermistor_k(resistance_logged, pullup_ohm, sh=SH_YSI_44909):
    """Temperature (K) of a tempctrl thermistor channel.

    The firmware reports ``resistance`` computed with a 10 kOhm pull-up;
    scaling by ``pullup_ohm / 10000`` gives the resistance for the measured
    pull-up exactly, and Steinhart-Hart gives the temperature."""
    r = np.asarray(resistance_logged, dtype=float) * pullup_ohm / FIRMWARE_PULLUP_OHM
    lr = np.log(r)
    a, b, c = sh
    return 1.0 / (a + b * lr + c * lr ** 3)


def state_dwells(index, state, antenna, gap_s=5.0, products=()):
    """One switch state's integrations, with dwell ids, and their spectra.

    Rows of ``state`` closer than ``gap_s`` belong to one dwell (integrations
    within a dwell are ~1 s apart; a state recurs a cycle later). Returns
    ``(meta, freqs_mhz, spec, bundle)``: ``meta`` is the selection's metadata
    with ``dwell`` (-1 for rows with no time) and ``pos_in_dwell`` columns and
    a 0..n-1 index matching ``spec``'s rows."""
    sel = index.select(rfswitch=state)
    b = sel.load_bundle(antenna=antenna, products=list(products))
    meta = sel.meta.reset_index(drop=True)
    if not np.array_equal(meta["time_best"].to_numpy(), np.asarray(b.t)):
        raise ValueError(f"{state}: bundle rows are not in the metadata's order")
    meta["dwell"] = sel.visits(gap_s=gap_s)
    meta["pos_in_dwell"] = meta.groupby("dwell").cumcount()
    return meta, np.asarray(b.freqs_mhz, float), np.asarray(b.data, float), b


def rfi_flags(spec, freqs_mhz):
    """Per-sample flags (int32 wraps | transients) and per-channel persistent
    lines for one state's integrations, in time order, from
    :mod:`eigsep_data.flagging.detectors`.

    Wrapped samples are replaced by their channel's median log power before
    the two tracks run, so they do not drag the baselines; they stay flagged.
    ``persistent_track`` checks only its analysis band (45-235 MHz)."""
    spec = np.asarray(spec, dtype=float)
    wrap = overflow_mask(spec)
    valid = ~np.all(spec == 0, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log10(np.where(spec > 0, spec, np.nan))
    fill = np.nanmedian(logp[valid], axis=0)
    logp = np.where(np.isfinite(logp), logp, fill)
    trans, _, _ = transient_track(logp, valid)
    persist, _ = persistent_track(logp, valid, np.asarray(freqs_mhz, float))
    return wrap | trans, persist


def vna_sweeps(meta, gap_s=60.0):
    """Start and end times (unix) of each VNA sweep: runs of rows in a
    ``VNA*`` switch state, split where rows are more than ``gap_s`` apart.
    Returns an (nsweep, 2) array."""
    state = meta["rfswitch"].astype(str)
    t = np.sort(meta.loc[state.str.startswith("VNA"), "time_best"].to_numpy(float))
    if t.size == 0:
        return np.empty((0, 2))
    cut = np.flatnonzero(np.diff(t) > gap_s)
    return np.c_[np.r_[t[0], t[cut + 1]], np.r_[t[cut], t[-1]]]


def near_sweep(meta, sweeps, guard=SWEEP_GUARD_S):
    """Rows of ``meta`` from ``guard[0]`` s before a sweep's start to
    ``guard[1]`` s after its end."""
    t = meta["time_best"].to_numpy(float)
    out = np.zeros(t.size, dtype=bool)
    for t0, t1 in np.asarray(sweeps).reshape(-1, 2):
        out |= (t > t0 - guard[0]) & (t < t1 + guard[1])
    return out


def dwell_means(meta, spec, flags=None, skip_first=0, columns=()):
    """Each dwell's integration-time-weighted mean spectrum.

    Flagged samples (``flags``, per sample; default: int32 wraps only) and
    all-zero rows get zero weight; the first ``skip_first`` integrations of
    each dwell are dropped. ``columns`` are metadata columns averaged with the
    same per-row weights. Returns a table (one row per dwell: ``dwell``,
    ``t``, ``n`` and the columns) and the (ndwell, nchan) mean powers, NaN
    where a channel has no weight."""
    spec = np.asarray(spec, dtype=float)
    flags = overflow_mask(spec) if flags is None else np.asarray(flags, bool)
    keep = (meta["pos_in_dwell"].to_numpy() >= skip_first) & (meta["dwell"].to_numpy() >= 0)
    good = ~flags & ~np.all(spec == 0, axis=1)[:, None] & keep[:, None]
    w = np.where(good, meta["integration_time"].to_numpy(float)[:, None], 0.0)
    rows, means = [], []
    for d, g in meta[keep].groupby("dwell"):
        i = g.index.to_numpy()
        ws = w[i].sum(axis=0)
        num = (w[i] * np.where(good[i], spec[i], 0.0)).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            means.append(np.where(ws > 0, num / ws, np.nan))
        tw = g["integration_time"].to_numpy(float)
        row = {"dwell": d, "t": np.average(g["time_best"], weights=tw), "n": len(i)}
        for c in columns:
            v = g[c].to_numpy(float)
            ok = np.isfinite(v)
            row[c] = np.average(v[ok], weights=tw[ok]) if ok.any() else np.nan
        rows.append(row)
    nchan = spec.shape[1]
    return pd.DataFrame(rows), (np.array(means) if means else np.empty((0, nchan)))


def cycles(dwells, order, max_span_s=90.0):
    """Group dwells of several states into switch cycles.

    ``dwells`` maps each state to its ``(table, powers)`` from
    :func:`dwell_means`; ``order[0]`` starts a cycle. A cycle takes the first
    dwell of each other state after the starting dwell, if it begins before
    the next starting dwell and within ``max_span_s`` of it; cycles missing a
    state are dropped (a VNA sweep cutting a cycle short, for one). Returns a
    table (one row per cycle: the start time ``t`` and each state's dwell
    index ``i_<state>``) and a dict of (ncycle, nchan) powers per state."""
    first, rest = order[0], order[1:]
    t_first = dwells[first][0]["t"].to_numpy(float)
    rows, out = [], {s: [] for s in order}
    for k, t0 in enumerate(t_first):
        t_next = t_first[k + 1] if k + 1 < t_first.size else np.inf
        pick = {first: k}
        for s in rest:
            t = dwells[s][0]["t"].to_numpy(float)
            j = np.flatnonzero((t > t0) & (t < t_next) & (t - t0 < max_span_s))
            if j.size:
                pick[s] = int(j[0])
        if len(pick) < len(order):
            continue
        rows.append({"t": t0, **{f"i_{s}": pick[s] for s in order}})
        for s in order:
            out[s].append(dwells[s][1][pick[s]])
    nchan = dwells[first][1].shape[1]
    arrays = {s: (np.array(v) if v else np.empty((0, nchan))) for s, v in out.items()}
    return pd.DataFrame(rows), arrays


def classify_rate(t, temperature, steady_per_min):
    """Each point steady, heating or cooling by the rate of change of
    ``temperature`` between neighbouring points (per minute), ``steady``
    when its magnitude is below ``steady_per_min``. Returns the rates and
    the labels."""
    rate = np.gradient(np.asarray(temperature, float), np.asarray(t, float)) * 60.0
    kind = np.where(np.abs(rate) < steady_per_min, "steady",
                    np.where(rate > 0, "heating", "cooling"))
    return rate, kind
