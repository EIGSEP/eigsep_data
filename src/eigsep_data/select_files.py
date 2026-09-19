#!/usr/bin/env python3
"""
Pick a clean subset of Marjum Pass 2026-07 correlator files.

The campaign spans three antenna wirings, two accumulator lengths, and a dozen
instrument excursions. Handing a pipeline all 5,120 files is almost always
wrong. This script applies the masking rules documented in
INDEX.md § "Selecting data (for analysts and agents)" and reports what it
dropped and why.

Usage
-----
As a library -- point the package at a campaign once, then select::

    import eigsep_data
    eigsep_data.set_campaign_root("~/data/marjum-2026-07")
    kept, dropped, warnings = eigsep_data.select_files.select(phase="C")

or pass ``root=`` for one call without changing the global setting.

From the shell (``--campaign`` overrides the configured root)::

    eigsep-select-files --phase C
    eigsep-select-files --phase C --start 2026-07-16T02:00:00Z \
                        --end 2026-07-17T20:00:00Z --inputs 0
    eigsep-select-files --phase C --json > files.json
    eigsep-select-files --phase C --explain          # per-mask detail
    eigsep-select-files --start ... --no-conditional # mandatory masks only

This module lived at ``marjum-2026-07/curation/select_files.py`` until
2026-09-19 and anchored on its own ``__file__``; it now resolves the
campaign through :mod:`eigsep_data.paths`. The mask catalog and the
meaning of every mask are unchanged.

Exit status is 0 even when the selection is empty; check the report.

Masks are declared as data at the top of this file. If you learn something new
about the campaign, add a mask here rather than filtering ad hoc downstream —
that way every consumer inherits the correction.

Provenance: phase pivots and outage windows come from data/README.md; SNAP
test-flips, mux-tuning windows, and undocumented gaps come from the boundary
scan (see detect_boundaries.py, boundaries.jsonl, events.jsonl).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .paths import campaign_data_dir, get_campaign_root


def _root():
    """Campaign root, resolved at call time -- never at import.

    This module used to anchor on its own ``__file__`` (it lived in
    the campaign's ``curation/``). Inside the package there is no
    such anchor, so the campaign must be named first, by
    ``eigsep_data.set_campaign_root()``, ``EIGSEP_CAMPAIGN_ROOT``, or
    the ``root=`` argument on :func:`select`.
    """
    return get_campaign_root(required=True)


def _data():
    return campaign_data_dir()

# --------------------------------------------------------------------------
# Campaign constants (authoritative: data/README.md)
# --------------------------------------------------------------------------

# Phase pivots. The named file is the FIRST file of the new phase.


PHASE_PIVOTS = {
    "A->B": "2026-07-14T04:10:43Z",   # corr_20260714_041043Z.h5
    "B->C": "2026-07-15T00:32:17Z",   # corr_20260715_003217Z.h5
}
CAMPAIGN_START = "2026-07-12T13:31:46Z"
CAMPAIGN_END = "2026-07-18T03:22:48Z"

# Which correlator inputs carry real sky signal in each phase.
PHASE_INPUTS = {
    "A": ("0", "2", "3", "4"),   # 0/2 early window unconfirmed; 3 (+4) later
    "B": ("3", "4", "5"),        # 5 is the mux copy of 4
    "C": ("0", "1", "4", "5"),   # 1 and 5 are mux copies of 0 and 4
}

# The accumulator length doubles here. Not a window to drop — a split point.
ACC_LEN_SPLIT = {
    "t_utc": "2026-07-15T15:54:59Z",
    "file": "corr_20260715_155459Z.h5",
    "before": 67108864,
    "after": 134217728,
}

# --------------------------------------------------------------------------
# Mandatory masks — never valid science data, excluded unconditionally
# --------------------------------------------------------------------------

MANDATORY_MASKS = [
    # Eight brief SNAP board swaps (C000122 flight -> C000069 scratch -> back).
    # Different board, different antenna set, 4x accumulator. Not flight config.
    *[
        {
            "name": "snap-test-flip",
            "start": s,
            "end": e,
            "reason": "SNAP swapped to scratch board C000069 (not a flight configuration)",
        }
        for s, e in [
            ("2026-07-12T21:15:40Z", "2026-07-12T21:16:59Z"),
            ("2026-07-13T22:36:01Z", "2026-07-13T22:37:06Z"),
            ("2026-07-14T02:07:57Z", "2026-07-14T02:09:23Z"),
            ("2026-07-14T21:22:33Z", "2026-07-14T21:23:38Z"),
            ("2026-07-15T00:25:50Z", "2026-07-15T00:26:55Z"),
            ("2026-07-15T04:48:26Z", "2026-07-15T04:49:51Z"),
            ("2026-07-16T01:22:24Z", "2026-07-16T01:24:33Z"),
            ("2026-07-17T03:24:54Z", "2026-07-17T03:27:03Z"),
        ]
    ],
    {
        "name": "antenna-move-allzero",
        "start": "2026-07-15T00:32:17Z",
        "end": "2026-07-15T01:16:59Z",
        "reason": "Phase B->C antenna move; spectra are identically zero",
    },
]

# Files the filter dropped as corrupt. Listed so a caller who sees them
# referenced in a log knows they are intentionally absent, not lost.
CORRUPT_FILES = {
    "corr_20260715_044343Z.h5",
    "corr_20260715_213105Z.h5",
    "corr_20260718_032419Z.h5",
}

# --------------------------------------------------------------------------
# Conditional masks — real data, but compromised for some measurements.
# `inputs` limits the mask to callers actually using those inputs.
# --------------------------------------------------------------------------

CONDITIONAL_MASKS = [
    {
        "name": "box-air-outage",
        "windows": [
            ("2026-07-15T18:55:00Z", "2026-07-15T21:30:00Z"),
            ("2026-07-16T17:32:00Z", "2026-07-16T17:42:00Z"),
            ("2026-07-16T18:42:00Z", "2026-07-16T23:27:00Z"),
            ("2026-07-17T23:35:00Z", "2026-07-17T23:45:00Z"),
        ],
        "inputs": {"4", "5"},
        "reason": "box-air (input 4) outage",
    },
    {
        "name": "phaseB-mux-off",
        "windows": [("2026-07-14T21:22:00Z", "2026-07-14T23:31:00Z")],
        "inputs": {"5"},
        "reason": "Phase B mux copy 4->5 disabled; the `35` cross is invalid",
    },
    {
        "name": "phaseC-mux-absent",
        "windows": [("2026-07-15T01:17:00Z", "2026-07-15T03:29:00Z")],
        "inputs": {"1", "5"},
        "reason": "Phase C mux copies not yet online; keys 1/5/15 unpopulated "
                  "(inputs 0 and 4 themselves are valid here)",
    },
    {
        "name": "mux-tuning",
        "windows": [
            ("2026-07-15T05:39:58Z", "2026-07-15T06:14:18Z"),
            ("2026-07-15T06:47:39Z", "2026-07-15T07:16:56Z"),
            ("2026-07-15T08:04:57Z", "2026-07-15T08:13:39Z"),
            ("2026-07-15T08:51:12Z", "2026-07-15T10:53:33Z"),
            ("2026-07-15T11:48:14Z", "2026-07-15T12:18:23Z"),
            ("2026-07-15T13:34:39Z", "2026-07-15T14:50:46Z"),
            ("2026-07-15T15:25:02Z", "2026-07-15T15:56:24Z"),
        ],
        "inputs": None,
        "reason": "ADC mux configuration actively being iterated",
    },
    {
        "name": "comb-rfi-1p25mhz",
        "windows": [("2026-07-16T01:00:00Z", "2026-07-16T01:30:00Z")],
        "inputs": None,
        "reason": "RF resonance: comb of spikes every 1.25 MHz plus half-spacing spurs",
    },
    {
        "name": "digital-self-comb",
        "windows": [("2026-07-17T15:37:00Z", "2026-07-18T03:24:00Z")],
        "inputs": None,
        "reason": "Self-generated comb at 1.953125 MHz = exactly 8 channels "
                  "(250/128 MHz, an ADC-clock subharmonic; tones at channels "
                  "= 0 mod 8, including DC). Onset is sharp at "
                  "corr_20260717_153744Z; intermittent until ~17:00 then "
                  "continuous. Present on internal loads and on BOTH boxes, "
                  "so it is instrumental, not sky. This is a per-CHANNEL "
                  "defect -- prefer masking channels = 0 mod 8 over dropping "
                  "files. NOTE: it is NOT the TX comb (1.000 MHz, 4.096 ch, "
                  "walks); the TX was off after 07-16 16:51, so anything in "
                  "this window selected 'every 8th channel' is self-RFI",
    },
    {
        "name": "thunderstorm",
        "windows": [("2026-07-16T00:00:00Z", "2026-07-16T06:00:00Z")],
        "inputs": None,
        "reason": "Thunderstorm with lightning visible in the data "
                  "(end time estimated; field notes give onset only)",
    },
    {
        "name": "laptop-rfi",
        "windows": [("2026-07-13T00:00:00Z", "2026-07-13T23:59:59Z")],
        "inputs": None,
        "reason": "Laptop RFI 145-160 MHz with 2 MHz spacing during Phase A checks",
    },
    {
        "name": "end-of-campaign-writestall",
        "windows": [("2026-07-18T03:20:00Z", "2026-07-18T03:22:48Z")],
        "inputs": None,
        "reason": "Last 3 files flush-written 838-931 s late after a 1,061 s write "
                  "stall; filenames lag their data badly. Data itself is valid and "
                  "contiguous (ends 03:08:49). Mask only if you key on filename time",
    },
    {
        "name": "accumulator-overflow",
        "windows": [],          # data-driven: see curation/overflow_channels.jsonl
        "inputs": None,         # scoped per-file to the inputs that actually wrapped
        "reason": "int32 accumulator wrapped in at least one channel (negative "
                  "auto power). Per-CHANNEL defect -- median 1 bad channel of "
                  "1024. Prefer curation/overflow_channels.jsonl and mask "
                  "channels; this drops whole files and is off by default",
    },
    {
        "name": "bad-header-clock",
        "windows": [
            # Sustained F-engine desync: header clock ~54.2 days in the past and
            # non-monotonic. Filenames are fine; per-sample header/times are not.
            ("2026-07-13T05:05:00Z", "2026-07-13T15:57:00Z"),
            # Smaller desync blocks.
            ("2026-07-12T13:31:00Z", "2026-07-12T20:39:00Z"),
            ("2026-07-12T22:55:00Z", "2026-07-12T23:05:00Z"),
            ("2026-07-15T03:19:00Z", "2026-07-15T03:28:00Z"),
            ("2026-07-15T15:54:00Z", "2026-07-15T15:57:00Z"),
        ],
        "inputs": None,
        "reason": "header/times unusable (unsynced F-engine); the DATA is fine. "
                  "Mask only if you need per-sample timing -- otherwise use the "
                  "filename close-time and keep these files",
    },
]

DEFAULT_CONDITIONAL = {
    "box-air-outage",
    "phaseB-mux-off",
    "phaseC-mux-absent",
    "mux-tuning",
    "comb-rfi-1p25mhz",
}
"""Conditional masks applied unless --no-conditional or --skip-mask.

Deliberately excludes thunderstorm, laptop-rfi, end-of-campaign-writestall and
bad-header-clock: those are band-, cadence- or clock-specific and dropping them
by default would silently discard good data for most callers. Enable them with
--add-mask.

bad-header-clock in particular masks files whose DATA is perfectly good -- only
their per-sample `header/times` are unusable. Add it if and only if you are
building a per-sample timeline; see INDEX.md "Which clock to trust".
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def parse_iso(t: str) -> datetime:
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def file_close_time(name: str) -> datetime:
    """UTC close time from the filename. Trust this over in-file header/times,
    which can carry unsynced May-2026 values."""
    stem = name.removeprefix("corr_").removesuffix(".h5")
    if "-" in stem:                       # corr_..Z-1.h5 duplicate suffix
        stem = stem.split("-", 1)[0]
    return datetime.strptime(stem, "%Y%m%d_%H%M%SZ").replace(tzinfo=timezone.utc)


def phase_of(t: datetime) -> str:
    if t < parse_iso(PHASE_PIVOTS["A->B"]):
        return "A"
    if t < parse_iso(PHASE_PIVOTS["B->C"]):
        return "B"
    return "C"


def in_window(t: datetime, start: str, end: str) -> bool:
    return parse_iso(start) <= t <= parse_iso(end)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Observing-mode selection
#
# Modes come from curation/mode_table.jsonl (tracked), NOT from the gitignored
# per-file expansion, so a fresh checkout can select by mode without first
# re-running the scan. Each window names its file range; membership is by
# position in the sorted file list.
# --------------------------------------------------------------------------

def _mode_table():
    return _root() / "curation" / "mode_table.jsonl"


def _overflow_table():
    return _root() / "curation" / "overflow_channels.jsonl"


def load_overflow():
    """file -> {input: {chan: n_samples}} from curation/overflow_channels.jsonl.

    int32 accumulator wrap. This is a per-CHANNEL defect: the median affected
    (file, input) has ONE bad channel out of 1024 and the worst has under 50,
    so masking whole files throws away good data to fix ~0.01% of cells.
    Prefer the per-channel product; the file-level mask below exists only for
    callers who cannot apply channel masks, and is OFF by default.
    """
    overflow_table = _overflow_table()
    if not overflow_table.is_file():
        return {}
    out = {}
    with overflow_table.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out.setdefault(r["file"], {})[r["input"]] = r["channels"]
    return out


MODE_FIELDS = {
    "era": "height_era",
    "rotation": "rot_state",
    "orientation": "orient_bin",
    "tx-comb": "tx_comb",
    "acc-len": "corr_acc_len",
    "switch": "rfswitch_dominant",
}
"""CLI name -> mode_table.jsonl field. `--phase` is handled by the existing
phase logic and is deliberately not duplicated here."""


def load_mode_windows():
    """Return the mode windows, or None if the table has not been built."""
    mode_table = _mode_table()
    if not mode_table.is_file():
        return None
    with mode_table.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def file_mode_map(windows, all_files):
    """Map each file name -> its mode window dict.

    Windows are contiguous in file order, so expand each [file_first, file_last]
    span across the sorted file list.
    """
    pos = {name: i for i, name in enumerate(all_files)}
    out = {}
    for w in windows:
        i, j = pos.get(w["file_first"]), pos.get(w["file_last"])
        if i is None or j is None:
            continue
        for name in all_files[i:j + 1]:
            out[name] = w
    return out


def select(phase=None, start=None, end=None, inputs=None,
           conditional=None, explain=False, modes=None, root=None):
    """Return (kept, dropped, warnings).

    kept     : sorted list of file names
    dropped  : list of {file, mask, reason}
    warnings : list of strings the caller must not ignore

    modes    : optional {cli_name: [accepted values]}; a file is kept only if
               its mode window matches every constraint. Mode selection and
               masking compose -- modes choose a configuration, masks drop bad
               data within it.

    root     : optional campaign root for this call only; overrides
               ``set_campaign_root()`` and ``EIGSEP_CAMPAIGN_ROOT``
               without changing them.
    """
    data = (
        Path(root) / "data" if root is not None else _data()
    )
    if not data.is_dir():
        raise SystemExit(f"error: no data directory at {data}")

    conditional = DEFAULT_CONDITIONAL if conditional is None else conditional
    inputs = set(inputs) if inputs else None

    t_start = parse_iso(start) if start else parse_iso(CAMPAIGN_START)
    t_end = parse_iso(end) if end else parse_iso(CAMPAIGN_END)

    active_conditional = [m for m in CONDITIONAL_MASKS if m["name"] in conditional]
    overflow = load_overflow() if any(
        m["name"] == "accumulator-overflow" for m in active_conditional) else {}

    kept, dropped, warnings = [], [], []

    all_files = [p.name for p in sorted(data.glob("corr_*.h5"))]

    mode_of = {}
    if modes:
        windows = load_mode_windows()
        if windows is None:
            raise SystemExit(
                f"error: --era/--rotation/--orientation/--tx-comb/--acc-len/--switch "
                f"need {_mode_table()}, which is missing.\n"
                f"       Build it with: "
                f"data-analysis/scripts/marjum-2026-07/build_mode_table.py"
            )
        mode_of = file_mode_map(windows, all_files)
        unmapped = len(all_files) - len(mode_of)
        if unmapped:
            warnings.append(
                f"{unmapped} file(s) are not covered by the mode table and were "
                f"excluded by mode selection; rebuild with build_mode_table.py if "
                f"the data directory changed."
            )

    for path in sorted(data.glob("corr_*.h5")):
        name = path.name
        try:
            t = file_close_time(name)
        except ValueError:
            warnings.append(f"unparseable filename skipped: {name}")
            continue

        if not (t_start <= t <= t_end):
            continue
        if phase and phase_of(t) != phase:
            continue

        # Mode selection: a configuration filter, not a quality mask. Files
        # outside the requested mode are simply not in scope, so they are not
        # reported as "dropped" -- that column is reserved for bad data.
        if modes:
            w = mode_of.get(name)
            if w is None:
                continue
            if not all(str(w.get(MODE_FIELDS[k])) in {str(v) for v in vals}
                       for k, vals in modes.items()):
                continue

        hit = None

        for m in MANDATORY_MASKS:
            if in_window(t, m["start"], m["end"]):
                hit = (m["name"], m["reason"])
                break

        if hit is None:
            for m in active_conditional:
                # A mask scoped to specific inputs only applies if the caller
                # asked for at least one of them.
                if m["inputs"] and inputs and not (m["inputs"] & inputs):
                    continue
                if m["name"] == "accumulator-overflow":
                    # Data-driven and per-file input-scoped: only a concern if
                    # an input the caller actually uses wrapped in THIS file.
                    ov = overflow.get(name)
                    if not ov:
                        continue
                    bad_inputs = set(ov)
                    if inputs and not (bad_inputs & inputs):
                        continue
                    n_ch = sum(len(c) for i, c in ov.items()
                               if not inputs or i in inputs)
                    hit = (m["name"],
                           f"{m['reason']} (inputs {sorted(bad_inputs)}, "
                           f"{n_ch} channel(s))")
                    break
                if any(in_window(t, s, e) for s, e in m["windows"]):
                    hit = (m["name"], m["reason"])
                    break

        if hit:
            dropped.append({"file": name, "mask": hit[0], "reason": hit[1]})
        else:
            kept.append(name)

    # ---- warnings the caller must not ignore -----------------------------
    acc_t = parse_iso(ACC_LEN_SPLIT["t_utc"])
    if kept:
        k_first, k_last = file_close_time(kept[0]), file_close_time(kept[-1])
        if k_first < acc_t <= k_last:
            warnings.append(
                f"selection straddles the corr_acc_len change at "
                f"{ACC_LEN_SPLIT['t_utc']} ({ACC_LEN_SPLIT['before']} -> "
                f"{ACC_LEN_SPLIT['after']}). Integration time doubles. Split the "
                f"analysis here or renormalize."
            )
        if phase is None and phase_of(k_first) != phase_of(k_last):
            warnings.append(
                f"selection spans phases {phase_of(k_first)}->{phase_of(k_last)}; "
                f"live inputs differ between them. Pass --phase unless you are "
                f"deliberately crossing the boundary."
            )

    # Overflow is sparse and per-channel, so the mask is off by default -- but a
    # caller must not silently integrate wrapped channels. Warn whenever the
    # kept set contains any, naming the worst channels so they can be masked.
    if kept and "accumulator-overflow" not in conditional:
        ov_all = load_overflow()
        chan_hits, n_files = {}, 0
        for name in kept:
            ov = ov_all.get(name)
            if not ov:
                continue
            rel = {i: c for i, c in ov.items() if not inputs or i in inputs}
            if not rel:
                continue
            n_files += 1
            for c in rel.values():
                for ch, n in c.items():
                    chan_hits[int(ch)] = chan_hits.get(int(ch), 0) + n
        if n_files:
            worst = sorted(chan_hits.items(), key=lambda kv: -kv[1])[:8]
            warnings.append(
                f"{n_files} of {len(kept)} kept file(s) contain int32 "
                f"accumulator wrap (negative auto power). This is per-channel, "
                f"not per-file -- most-affected channels: "
                f"{', '.join(str(c) for c, _ in worst)}. Mask those channels "
                f"using curation/overflow_channels.jsonl, or pass "
                f"--add-mask accumulator-overflow to drop the files entirely."
            )

    if inputs and phase:
        live = set(PHASE_INPUTS[phase])
        bad = inputs - live
        if bad:
            warnings.append(
                f"inputs {sorted(bad)} are not live in phase {phase} "
                f"(live: {sorted(live)}). Check data/README.md."
            )

    return kept, dropped, warnings


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Masks are declared as data in this file; add new ones there so "
               "every consumer inherits the correction.",
    )
    p.add_argument("--campaign", metavar="PATH", default=None,
                   help="campaign root (or its data/ dir); overrides "
                        "eigsep_data.set_campaign_root() and "
                        "EIGSEP_CAMPAIGN_ROOT")
    p.add_argument("--phase", choices=["A", "B", "C"],
                   help="restrict to one wiring phase (usually C for flight data)")
    p.add_argument("--start", metavar="ISO", help="earliest close time, e.g. 2026-07-16T02:00:00Z")
    p.add_argument("--end", metavar="ISO", help="latest close time")
    p.add_argument("--inputs", metavar="LIST",
                   help="comma-separated correlator inputs you intend to use, "
                        "e.g. 0,4 — scopes input-specific masks and validates "
                        "them against the phase")
    p.add_argument("--no-conditional", action="store_true",
                   help="apply mandatory masks only")
    p.add_argument("--skip-mask", metavar="LIST", default="",
                   help="comma-separated conditional masks to NOT apply")
    p.add_argument("--add-mask", metavar="LIST", default="",
                   help="comma-separated non-default conditional masks to apply "
                        "(thunderstorm, laptop-rfi, end-of-campaign-writestall, "
                        "bad-header-clock)")

    g = p.add_argument_group(
        "observing mode",
        "Select by instrument configuration, from curation/mode_table.jsonl. "
        "Each takes a comma-separated list; values OR within a flag and AND "
        "across flags. Mode selection composes with masking: modes choose the "
        "configuration, masks drop bad data inside it.")
    g.add_argument("--era", metavar="LIST",
                   help="height era, e.g. ~30m or ~87.5m,~91m")
    g.add_argument("--rotation", metavar="LIST",
                   help="parked, az-moving, el-moving, az+el-moving")
    g.add_argument("--orientation", metavar="LIST",
                   help="orientation bin, e.g. ~30m/C/az005")
    g.add_argument("--tx-comb", metavar="LIST", choices=None,
                   help="on or off — whether the 4 MHz transmitter comb is present")
    g.add_argument("--acc-len", metavar="LIST",
                   help="correlator accumulator length, e.g. 67108864")
    g.add_argument("--switch", metavar="LIST",
                   help="dominant RF switch state, e.g. RFANT or RFAMB,RFNON")
    g.add_argument("--list-modes", action="store_true",
                   help="print the observing-mode catalog and exit")

    p.add_argument("--json", action="store_true",
                   help="emit a JSON object on stdout instead of a file list")
    p.add_argument("--explain", action="store_true",
                   help="list every dropped file and its mask on stderr")
    p.add_argument("--list-masks", action="store_true",
                   help="print the mask catalog and exit")
    a = p.parse_args(argv)

    if a.campaign:
        from .paths import set_campaign_root

        set_campaign_root(a.campaign)

    if a.list_modes:
        windows = load_mode_windows()
        if windows is None:
            raise SystemExit(
                f"error: {_mode_table()} is missing. Build it with: "
                f"data-analysis/scripts/marjum-2026-07/build_mode_table.py")
        print(f"{len(windows)} mode windows in {_mode_table()}\n")
        for cli, field in MODE_FIELDS.items():
            counts = {}
            for w in windows:
                v = w.get(field)
                if v is None:
                    continue
                counts[str(v)] = counts.get(str(v), 0) + w["n_files"]
            if not counts:
                continue
            shown = sorted(counts.items(), key=lambda kv: -kv[1])
            print(f"--{cli}")
            for v, n in shown[:12]:
                print(f"    {v:28s} {n:5d} files")
            if len(shown) > 12:
                print(f"    ... and {len(shown) - 12} more "
                      f"(see {_mode_table()})")
            print()
        return 0

    if a.list_masks:
        print("MANDATORY (always applied):")
        for name in sorted({m["name"] for m in MANDATORY_MASKS}):
            n = sum(1 for m in MANDATORY_MASKS if m["name"] == name)
            print(f"  {name}  ({n} window{'s' if n > 1 else ''})")
        print("\nCONDITIONAL (* = applied by default):")
        for m in CONDITIONAL_MASKS:
            star = "*" if m["name"] in DEFAULT_CONDITIONAL else " "
            scope = f"inputs {sorted(m['inputs'])}" if m["inputs"] else "all inputs"
            print(f" {star} {m['name']:26} {len(m['windows']):2d} window(s)  {scope}")
            print(f"     {m['reason']}")
        return 0

    if a.no_conditional:
        conditional = set()
    else:
        conditional = set(DEFAULT_CONDITIONAL)
        conditional |= {s for s in a.add_mask.split(",") if s}
        conditional -= {s for s in a.skip_mask.split(",") if s}

    known = {m["name"] for m in CONDITIONAL_MASKS}
    for name in conditional - known:
        print(f"warning: unknown mask {name!r} ignored "
              f"(see --list-masks)", file=sys.stderr)
    conditional &= known

    inputs = [s.strip() for s in a.inputs.split(",")] if a.inputs else None

    modes = {}
    for cli in MODE_FIELDS:
        val = getattr(a, cli.replace("-", "_"))
        if val:
            modes[cli] = [v.strip() for v in val.split(",") if v.strip()]

    kept, dropped, warnings = select(
        phase=a.phase, start=a.start, end=a.end,
        inputs=inputs, conditional=conditional, explain=a.explain,
        modes=modes or None,
    )

    by_mask = {}
    for d in dropped:
        by_mask[d["mask"]] = by_mask.get(d["mask"], 0) + 1

    if a.json:
        json.dump({
            "files": kept,
            "n_kept": len(kept),
            "n_dropped": len(dropped),
            "dropped_by_mask": by_mask,
            "dropped": dropped,
            "warnings": warnings,
            "query": {
                "phase": a.phase, "start": a.start, "end": a.end,
                "inputs": inputs, "conditional_masks": sorted(conditional),
            },
            "live_inputs": list(PHASE_INPUTS[a.phase]) if a.phase else None,
            "corrupt_files_absent_by_design": sorted(CORRUPT_FILES),
        }, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for name in kept:
            print(name)

    # Report always goes to stderr so it never pollutes a piped file list.
    r = sys.stderr
    print(f"\n  kept {len(kept)} file(s), dropped {len(dropped)}", file=r)
    if a.phase:
        print(f"  phase {a.phase}; live inputs {list(PHASE_INPUTS[a.phase])}", file=r)
    for mask, n in sorted(by_mask.items(), key=lambda kv: -kv[1]):
        print(f"    -{n:4d}  {mask}", file=r)
    if a.explain:
        print("", file=r)
        for d in dropped:
            print(f"    {d['file']}  [{d['mask']}] {d['reason']}", file=r)
    for w in warnings:
        print(f"\n  WARNING: {w}", file=r)
    if not kept:
        print("\n  Selection is empty. Widen --start/--end or relax masks.", file=r)
    print("", file=r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
