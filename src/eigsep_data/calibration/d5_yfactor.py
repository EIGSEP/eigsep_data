# Vendored unchanged from data-analysis/notebooks/arp/marjum-2026-07/d5_yfactor.py
# (commit 58c4c64): Christian's minimal Y-factor chain, run as written to check
# tcal against an independent calibration. Not used to build any product.
"""Minimal three-state Y-factor calibration for EIGSEP Deployment 5.

Reads the correlator files and the field S11 files directly, with only
numpy, h5py and cmt_vna. Each step is a plain function that can be
used on its own:

1. :func:`load_corr` -- every integration of one correlator input in a
   time window, with its switch state and load thermistor reading.
2. :func:`average_blocks` -- one average spectrum per contiguous
   switch-state block (a "visit").
3. :func:`nearest` -- nearest-in-time pairing, used for load and
   noise-source visits and for S11 sweeps alike.
4. :func:`tant_star` -- the three-state Y factor.
5. :func:`calibrate_s11` -- field S11 sweeps calibrated to plane P, the
   receiver input, with the chain of ``calibrate_field_s11.py`` in
   EIGSEP/data-analysis.
6. :func:`correct_antenna_s11` and :func:`correct_receiver_s11` -- the
   two reflection corrections.

Nothing is flagged, cut, smoothed or interpolated in time. The one
exclusion a caller will want, dropped integrations (all-zero rows), is
left to the caller through ``use=`` in :func:`average_blocks`.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from cmt_vna import calkit

#: Assumed excess temperature of the noise source over the ambient load,
#: in K: the nameplate ENR of 35 dB behind a 30 dB pad, so
#: 290 K * 10**(5/10) = 917 K. It has not been measured, and the kelvin
#: scale of every temperature below is only as good as this number.
T_NS = 290.0 * 10 ** ((35.0 - 30.0) / 10)

#: ``rfswitch`` states of the three Y-factor measurements.
SKY, NOISE, LOAD = "RFANT", "RFNON", "RFAMB"

#: A file whose last header time is further than this from its filename
#: stamp has a stale clock (about 12 % of D5 files, off by ~54 days).
SYNC_TOLERANCE_S = 3600.0

#: A time step longer than this starts a new switch block, even when
#: the state is unchanged. Consecutive integrations are 0.27 or 0.54 s
#: apart.
MAX_GAP_S = 5.0

#: Switch paths from the VNA's internal reference plane to plane P, per
#: DUT in the S11 files: (VNA-leg path de-embedded, RF-leg path
#: embedded). The receiver has no RF leg. ``load`` and ``noise`` have no
#: path to P and are not calibrated here.
S11_PATHS = {
    "ant": ("VNAANT", "RFANT"),
    "amb": ("VNAAMB", "RFAMB"),
    "sp1_open": ("VNASP1", "RFSP1"),
    "sp1_short": ("VNASP1", "RFSP1"),
    "rec": ("VNARF", None),
}

_STAMP = re.compile(r"_(\d{8}_\d{6})Z")


def file_time(path):
    """Unix time of the UTC stamp in a D5 filename.

    For correlator files this is the time the file was closed, after its
    last integration.
    """
    stamp = _STAMP.search(Path(path).name).group(1)
    t = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    return t.replace(tzinfo=timezone.utc).timestamp()


def _entries(meta, name, n):
    """Per-integration entries of one metadata stream, padded to ``n``."""
    if name not in meta:
        return [None] * n
    entries = json.loads(meta[name][()])[:n]
    return entries + [None] * (n - len(entries))


def read_corr_file(path, key):
    """One correlator file: spectra of ``key`` and per-row metadata.

    Parameters
    ----------
    path : str or Path
    key : str
        Data key: an auto such as ``"4"`` or a cross such as ``"04"``.
        In phase C (from Jul 15), ``"4"`` is the suspended (box-air)
        antenna, which the calibration switch serves, and ``"0"`` is the
        ground (box-gnd) antenna.

    Returns
    -------
    dict
        ``spec`` (n_row, n_chan): as stored (int32 counts) for an auto;
        complex for a cross, built from the stored (real, imag) pair.
        ``time`` (n_row,): unix time of each integration.
        ``time_from_filename`` (bool): the header clock was stale, so
        ``time`` counts back from the filename stamp by the integration
        time.
        ``state`` (n_row,): the ``rfswitch`` state, ``"MISSING"`` where
        none was recorded.
        ``t_load`` (n_row,): load thermistor ``tempctrl_load.T_now`` in
        K, NaN where none was recorded.
        ``freqs`` (n_chan,): channel frequencies in MHz.
    """
    with h5py.File(path, "r") as h:
        spec = h["data"][key][()]
        header_times = h["header/times"][()]
        freqs = h["header/freqs"][()]
        dt = float(h["header"].attrs["integration_time"])
        meta = h["metadata"]
        n = len(header_times)
        states = _entries(meta, "rfswitch", n)
        loads = _entries(meta, "tempctrl_load", n)
    if spec.ndim == 3:
        spec = spec[..., 0] + 1j * spec[..., 1]
    t_name = file_time(path)
    stale = abs(t_name - header_times[-1]) > SYNC_TOLERANCE_S
    if stale:
        time = t_name - (n - 1 - np.arange(n)) * dt
    else:
        time = header_times
    t_load = [
        (
            e["T_now"] + 273.15
            if isinstance(e, dict) and isinstance(e.get("T_now"), (int, float))
            else np.nan
        )
        for e in loads
    ]
    return {
        "spec": spec,
        "time": np.asarray(time, dtype=float),
        "time_from_filename": stale,
        "state": np.array(
            [s if isinstance(s, str) else "MISSING" for s in states]
        ),
        "t_load": np.array(t_load, dtype=float),
        "freqs": freqs,
    }


def load_corr(corr_dir, t_start, t_stop, key, margin_s=1200.0):
    """Every integration of ``key`` with time in ``[t_start, t_stop)``.

    Files are chosen by their filename stamp, which is the close time and
    can trail the last integration by the write backlog (up to ~16 min):
    every file stamped in ``[t_start, t_stop + margin_s]`` is read and
    its rows selected by time.

    Returns
    -------
    dict
        The arrays of :func:`read_corr_file` joined over files and sorted
        by time, plus ``file`` (n_row,) naming each row's file and
        ``time_from_filename`` (n_row,) per row.
    """
    files = [
        p
        for p in sorted(Path(corr_dir).glob("corr_*.h5"))
        if t_start <= file_time(p) <= t_stop + margin_s
    ]
    parts = []
    for p in files:
        d = read_corr_file(p, key)
        keep = (d["time"] >= t_start) & (d["time"] < t_stop)
        if not keep.any():
            continue
        parts.append(
            {
                "spec": d["spec"][keep],
                "time": d["time"][keep],
                "state": d["state"][keep],
                "t_load": d["t_load"][keep],
                "file": np.full(keep.sum(), p.name),
                "time_from_filename": np.full(
                    keep.sum(), d["time_from_filename"]
                ),
            }
        )
        freqs = d["freqs"]
    if not parts:
        raise ValueError(f"no {key!r} integrations in the window")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(out["time"], kind="stable")
    out = {k: v[order] for k, v in out.items()}
    out["freqs"] = freqs
    return out


def switch_blocks(state, time, max_gap_s=MAX_GAP_S):
    """Contiguous runs of one switch state.

    A new block starts where the state changes or where consecutive
    integrations are more than ``max_gap_s`` apart.

    Returns
    -------
    starts, stops : np.ndarray
        Row indices, so block ``i`` is rows ``starts[i]:stops[i]``.
    """
    new = np.ones(len(state), dtype=bool)
    new[1:] = (state[1:] != state[:-1]) | (np.diff(time) > max_gap_s)
    starts = np.flatnonzero(new)
    stops = np.r_[starts[1:], len(state)]
    return starts, stops


def average_blocks(
    data,
    states=(SKY, NOISE, LOAD),
    use=None,
    max_gap_s=MAX_GAP_S,
    stat=np.median,
):
    """One spectrum per switch block, for each state in ``states``.

    Parameters
    ----------
    data : dict
        From :func:`load_corr`.
    states : tuple of str
    use : np.ndarray of bool, optional
        Rows to include. Excluded rows still belong to their block, so
        they do not split it; a block with no included rows is omitted.
    max_gap_s : float
        See :func:`switch_blocks`.
    stat : callable
        Reduction over rows, called as ``stat(x, axis=0)``. The default
        is the median because strong narrow lines overflow the int32
        accumulator and wrap to about -2e9 in a few integrations, which
        the median ignores and a mean (``np.mean``) does not.

    Returns
    -------
    dict
        Per state, arrays over its blocks: ``t`` (mean time of the rows
        used), ``t_start``, ``t_stop`` (first and last row used), ``n``
        (rows used), ``spec`` (n_block, n_chan) and ``t_load`` (mean
        thermistor temperature in K over the rows used).
    """
    use = np.ones(len(data["time"]), bool) if use is None else use
    starts, stops = switch_blocks(data["state"], data["time"], max_gap_s)
    out = {
        s: {k: [] for k in ("t", "t_start", "t_stop", "n", "spec", "t_load")}
        for s in states
    }
    for a, b in zip(starts, stops):
        state = data["state"][a]
        rows = a + np.flatnonzero(use[a:b])
        if state not in out or rows.size == 0:
            continue
        t = data["time"][rows]
        block = out[state]
        block["t"].append(t.mean())
        block["t_start"].append(t[0])
        block["t_stop"].append(t[-1])
        block["n"].append(rows.size)
        block["spec"].append(stat(data["spec"][rows].astype(float), axis=0))
        loads = data["t_load"][rows]
        block["t_load"].append(
            loads[np.isfinite(loads)].mean()
            if np.isfinite(loads).any()
            else np.nan
        )
    return {s: {k: np.array(v) for k, v in b.items()} for s, b in out.items()}


def nearest(t, t_ref):
    """Nearest-in-time neighbour.

    Returns
    -------
    index : np.ndarray of int
        For each time in ``t``, the index of the nearest time in
        ``t_ref``.
    age : np.ndarray
        ``|t - t_ref[index]|`` in the units of ``t`` (s for unix times).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    t_ref = np.asarray(t_ref, dtype=float)
    index = np.abs(t[:, None] - t_ref[None, :]).argmin(axis=1)
    return index, np.abs(t - t_ref[index])


def tant_star(p_ant, p_ns, p_load, t_load, t_ns=None):
    """Three-state Y factor (Monsalve et al. 2017, eq. 1).

    ``T* = T_load + T_NS (P_ant - P_load) / (P_ns - P_load)``

    Parameters
    ----------
    p_ant, p_ns, p_load : array_like
        Powers in the sky, noise-source and ambient-load states.
    t_load : array_like
        Physical temperature of the ambient load in K, broadcastable
        against the powers (shape ``(n, 1)`` for ``(n, n_chan)`` powers).
    t_ns : float, optional
        Excess temperature of the noise-source state over the load state
        in K. Defaults to the module-level :data:`T_NS`, read at call
        time.

    Returns
    -------
    np.ndarray
        ``T*`` in K. Channels where ``P_ns == P_load`` come out inf or
        NaN.
    """
    t_ns = T_NS if t_ns is None else t_ns
    with np.errstate(divide="ignore", invalid="ignore"):
        return t_load + t_ns * (p_ant - p_load) / (p_ns - p_load)


def characterize_internal_osl(path):
    """True reflection of the VNA's internal open/short/load standards.

    ``path`` is the lab capture ``vna_internal_osl_*.npz``, which holds
    uncalibrated traces of the S911T manual kit (``MANUALO/S/L``) and of
    the internal standards (``VNAO/S/L``). The VNA error network is
    solved from the manual traces against the ``calkit.S911T`` model and
    de-embedded from the internal traces.

    Returns
    -------
    freqs_hz : np.ndarray
    osl : np.ndarray
        (3, n_freq), in O, S, L order.
    """
    raw = np.load(path)
    freqs = np.asarray(raw["freqs"], dtype=float)
    manual = np.array([raw["MANUALO"], raw["MANUALS"], raw["MANUALL"]])
    internal = np.array([raw["VNAO"], raw["VNAS"], raw["VNAL"]])
    model = calkit.S911T(freq_Hz=freqs).std_gamma
    vna = calkit.network_sparams(model, manual)
    return freqs, calkit.de_embed_sparams(vna, internal)


def read_s11_file(path):
    """One raw S11 file (``ants11_*`` or ``recs11_*``).

    Returns
    -------
    dict
        ``time``: measurement time (``metadata_snapshot_unix``; the
        filename is the write time, not the measurement time).
        ``mode``: ``"ant"`` or ``"rec"``.
        ``freqs_hz`` (n_freq,).
        ``traces``: DUT name to uncalibrated complex trace.
        ``osl`` (3, n_freq): the internal open, short and load traces
        recorded with this sweep.
    """
    with h5py.File(path, "r") as h:
        traces = {k: h["data"][k][()] for k in h["data"]}
        attrs = h["header"].attrs
        time, mode = float(attrs["metadata_snapshot_unix"]), attrs["mode"]
        freqs = np.array(json.loads(h["header/freqs"][()]), dtype=float)
    osl = np.array([traces.pop(f"cal:VNA{s}") for s in "OSL"])
    return {
        "time": time,
        "mode": mode,
        "freqs_hz": freqs,
        "traces": traces,
        "osl": osl,
    }


def calibrate_s11(
    s11_dir, internal_osl_path, switch_sparams_path, borrow_osl=True
):
    """Every field S11 trace in ``s11_dir``, calibrated to plane P.

    Per trace: remove the VNA error network solved from the sweep's own
    internal OSL traces against :func:`characterize_internal_osl`; then
    de-embed the VNA-leg switch path and embed the RF-leg path
    (:data:`S11_PATHS`). This is the chain of ``calibrate_field_s11.py``.

    Some sweeps are zero-filled (cmt_vna issue #54). A trace containing
    an exact zero is not calibrated. A sweep whose internal OSL contains
    a zero cannot calibrate itself; with ``borrow_osl=True`` it uses the
    valid OSL nearest in time from a sweep of the same mode, as
    ``calibrate_field_s11.py`` does, and with ``False`` its traces are
    not calibrated. Both cases are reported in ``log``.

    Returns
    -------
    freqs_hz : np.ndarray
    s11 : dict
        DUT name to ``t`` (n_sweep,), ``gamma`` (n_sweep, n_freq) at P,
        sorted by time, and ``osl_t`` (n_sweep,): time of the sweep
        whose internal OSL calibrated it (equal to ``t`` when its own).
    log : list of dict
        One entry per trace: ``file``, ``dut``, ``t``, ``osl_t`` and
        ``status`` (``"own OSL"``, ``"borrowed OSL"``,
        ``"zero-filled trace"`` or ``"zero-filled OSL"``).
    """
    freqs, osl_true = characterize_internal_osl(internal_osl_path)
    sparams = dict(np.load(switch_sparams_path))
    sweeps = [
        (p.name, read_s11_file(p))
        for p in sorted(Path(s11_dir).glob("*s11_*.h5"))
    ]
    for _, s in sweeps:
        if not np.array_equal(s["freqs_hz"], freqs):
            raise ValueError("S11 and OSL frequency grids differ")
    valid_osl = {}
    for _, s in sweeps:
        if not np.any(s["osl"] == 0):
            valid_osl.setdefault(s["mode"], []).append((s["time"], s["osl"]))
    s11, log = {}, []
    for name, s in sweeps:
        own = not np.any(s["osl"] == 0)
        if own:
            osl_t, osl = s["time"], s["osl"]
        elif borrow_osl and valid_osl.get(s["mode"]):
            bank = valid_osl[s["mode"]]
            i, _ = nearest(s["time"], [t for t, _ in bank])
            osl_t, osl = bank[i[0]]
        else:
            osl_t, osl = np.nan, None
        network = (
            None if osl is None else calkit.network_sparams(osl_true, osl)
        )
        for dut, trace in s["traces"].items():
            if dut not in S11_PATHS:
                continue
            entry = {"file": name, "dut": dut, "t": s["time"], "osl_t": osl_t}
            log.append(entry)
            if np.any(trace == 0):
                entry["status"] = "zero-filled trace"
                continue
            if network is None:
                entry["status"] = "zero-filled OSL"
                continue
            entry["status"] = "own OSL" if own else "borrowed OSL"
            vna_leg, rf_leg = S11_PATHS[dut]
            g = calkit.de_embed_sparams(network, trace)
            g = calkit.de_embed_sparams(sparams[vna_leg], g)
            if rf_leg is not None:
                g = calkit.embed_sparams(sparams[rf_leg], g)
            out = s11.setdefault(dut, {"t": [], "gamma": [], "osl_t": []})
            out["t"].append(s["time"])
            out["gamma"].append(g)
            out["osl_t"].append(osl_t)
    for dut, out in s11.items():
        order = np.argsort(out["t"])
        s11[dut] = {k: np.asarray(v)[order] for k, v in out.items()}
    return freqs, s11, log


def s11_to_channels(gamma, freqs_s11_mhz, freqs_mhz):
    """Resample reflection coefficients onto correlator channels.

    Real and imaginary parts are interpolated linearly in frequency (not
    magnitude and phase). Channels outside the sweep are NaN.
    """
    g = np.atleast_2d(gamma)

    def part(x):
        return np.interp(
            freqs_mhz, freqs_s11_mhz, x, left=np.nan, right=np.nan
        )

    return np.array([part(row.real) + 1j * part(row.imag) for row in g])


def mismatch(gamma, gamma_rec):
    """Fraction of a source's available power delivered to the receiver,
    relative to a matched source: ``(1 - |G|^2) / |1 - G G_rec|^2``.

    The receiver's own ``1 - |G_rec|^2`` is common to every source and
    left in the gain.
    """
    return (1 - np.abs(gamma) ** 2) / np.abs(1 - gamma * gamma_rec) ** 2


def correct_antenna_s11(t_star, gamma_ant):
    """``T* / (1 - |G_ant|^2)``: the antenna S11 correction alone.

    Treats the receiver and the ambient load as matched
    (``G_rec = G_load = 0``).
    """
    return t_star / (1 - np.abs(gamma_ant) ** 2)


def correct_receiver_s11(t_star, t_load, gamma_ant, gamma_load, gamma_rec):
    """Antenna, ambient-load and receiver S11 corrections together.

    With ``M_s`` from :func:`mismatch`, the powers are taken as
    ``P_ant = g (M_ant T_ant + T_r)``, ``P_load = g (M_load T_load + T_r)``
    and ``P_ns - P_load = g T_NS``, so that
    ``T* = M_ant T_ant + (1 - M_load) T_load`` and

    ``T_ant = [T* - (1 - M_load) T_load] / M_ant``.

    Assumes equal gain on every switch path, no receiver noise waves
    (``T_r`` does not depend on the source reflection), and ``T_NS`` as
    the noise-source excess delivered to the receiver. Reduces to
    :func:`correct_antenna_s11` for ``G_rec = G_load = 0``.
    """
    m_ant = mismatch(gamma_ant, gamma_rec)
    m_load = mismatch(gamma_load, gamma_rec)
    return (t_star - (1 - m_load) * t_load) / m_ant
