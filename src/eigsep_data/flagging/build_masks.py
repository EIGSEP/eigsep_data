"""Build versioned, category-tagged RFI flag masks for marjum-2026-07.

Product contract (agreed with data-archivist, 2026-09-12):

  flags/v0/flags_YYYYMMDD.h5   one per campaign day, gzipped, gitignored
  flags/v0/manifest.json       provenance + selection basis   (tracked)
  flags/v0/flag_bits.json      bit -> category mapping as data (tracked)
  flags/v0/summary.json        kept-fraction per band/day      (tracked)

**Orthogonality.** These masks carry channel x time RFI *within an
otherwise-valid file*.  They deliberately do NOT re-encode the
file-level campaign masks (SNAP test-flips, box-air outages, the
all-zero antenna move, mux windows).  `select_files.py` owns those.
Encoding them in both layers would let consumers double-apply and
would make it impossible to tell which layer removed what.

Times come from filenames; `header/times` is corrupt in 642/5120 files.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import h5py
import numpy as np

from ..paths import get_campaign_root
from . import detectors as D

VERSION = "v0"


def _root():
    """The campaign root, resolved at call time, never at import."""
    return str(get_campaign_root(required=True))


REPORT_BANDS = {
    "50-88 (DTV-lo)": (50.0, 88.0),
    "88-108 (FM)": (88.0, 108.0),
    "108-137": (108.0, 136.5),
    "137-138 (Orbcomm)": (136.5, 138.5),
    "138-174": (138.5, 174.0),
    "174-216 (DTV-hi)": (174.0, 216.0),
    "216-235": (216.0, 235.0),
}


def load_mode_table(path):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r["file_first"])
    return rows


def mode_for(modes, fname):
    for m in modes:
        if m["file_first"] <= fname <= m["file_last"]:
            return m
    return None


def process_file(args):
    path, tx_on = args
    fname = os.path.basename(path)
    try:
        with h5py.File(path, "r") as h:
            freqs = h["header/freqs"][:]
            keys = sorted(k for k in h["data"] if len(k) == 1)
            rfsw = None
            if "metadata" in h and "rfswitch" in h["metadata"]:
                rfsw = h["metadata/rfswitch"][()]
            per_input = {}
            for k in keys:
                raw = h["data/" + k][:]
                nt = raw.shape[0]
                ant = D.antenna_mask(rfsw, nt)

                # int32 accumulator wrap: label it, and repair it before
                # detection so it neither fires the transient detector
                # nor drags the robust scale. An instrumental wrap
                # counted as RFI corrupts both error directions.
                ovf = D.overflow_mask(raw)
                logp = np.log10(np.maximum(raw.astype(np.float64), 1.0))
                if ovf.any():
                    with np.errstate(invalid="ignore"):
                        chan_med = np.nanmedian(
                            np.where(ovf, np.nan, logp), axis=0)
                    chan_med = np.where(np.isfinite(chan_med), chan_med, 0.0)
                    logp = np.where(ovf, chan_med[None, :], logp)
                if np.median(logp) <= 0:
                    continue  # identically-zero file; file-level, not ours

                pix, _resid, _scale = D.transient_track(logp, ant)
                chan_flags, fresid = D.persistent_track(logp, ant, freqs)
                med = np.median(logp[ant], axis=0) if ant.sum() >= 4 \
                    else np.median(logp, axis=0)
                combs = D.identify_combs(med, freqs)

                bb = D.broadband_times(pix, freqs)
                ms = D.meteor_scatter_times(pix, freqs)
                cat = D.categorise(pix, chan_flags, freqs, combs, tx_on,
                                   bb, ms)
                # Calibration samples: not sky. Marked, never counted as RFI.
                cat[~ant, :] |= D.CAL
                # Accumulator wrap: instrumental, never counted as RFI.
                # Applied last and as an OR so it overrides any category
                # the detectors may already have assigned there.
                if ovf.any():
                    cat[ovf] = (cat[ovf] & ~np.uint8(D.RFI_BITS)) | D.OVERFLOW
                per_input[k] = {
                    "cat": cat,
                    "combs": {n: {kk: vv for kk, vv in v.items()
                                  if kk != "teeth"}
                              for n, v in combs.items()},
                    "n_ant": int(ant.sum()),
                    "n_time": int(nt),
                    "n_broadband": int(bb.sum()),
                    "n_meteor": int(ms.sum()),
                }
            return fname, per_input, freqs, None
    except Exception as exc:
        return fname, {}, None, f"{type(exc).__name__}: {exc}"


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_root(),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--campaign",
        default=None,
        help="campaign root (or its data/ dir); overrides "
             "eigsep_data.set_campaign_root() and EIGSEP_CAMPAIGN_ROOT",
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if args.campaign:
        from ..paths import set_campaign_root

        set_campaign_root(args.campaign)
    root = _root()
    data_dir = args.data or os.path.join(root, "data")
    out_dir = args.out or os.path.join(root, "flags", VERSION)

    os.makedirs(out_dir, exist_ok=True)
    modes = load_mode_table(os.path.join(root, "curation", "mode_table.jsonl"))
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if args.limit:
        files = files[: args.limit]

    tasks = []
    for p in files:
        m = mode_for(modes, os.path.basename(p))
        tasks.append((p, bool(m and m.get("tx_comb") == "on")))

    by_day = defaultdict(dict)
    stats = defaultdict(lambda: defaultdict(float))
    comb_log = []
    errors = []
    freqs_ref = None
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fname, per_input, freqs, err in ex.map(process_file, tasks,
                                                   chunksize=4):
            done += 1
            if done % 500 == 0:
                print(f"# {done}/{len(files)}", file=sys.stderr, flush=True)
            if err:
                errors.append({"file": fname, "error": err})
                continue
            if freqs is not None and freqs_ref is None:
                freqs_ref = freqs
            day = fname[5:13]
            for k, rec in per_input.items():
                by_day[day][f"{fname}/{k}"] = rec["cat"]
                comb_log.append({
                    "file": fname, "input": k,
                    **{f"contrast_{n}": v["contrast"]
                       for n, v in rec["combs"].items()},
                    **{f"det_{n}": v["detected"]
                       for n, v in rec["combs"].items()},
                    "n_broadband": rec["n_broadband"],
                    "n_meteor": rec["n_meteor"],
                })
                _accumulate(stats, day, k, rec["cat"], freqs)

    # ---- write per-day mask files -----------------------------------
    written = []
    for day, entries in sorted(by_day.items()):
        out = os.path.join(out_dir, f"flags_{day}.h5")
        with h5py.File(out, "w") as h:
            h.attrs["campaign"] = "marjum-2026-07"
            h.attrs["product"] = "flags"
            h.attrs["version"] = VERSION
            h.attrs["day"] = day
            h.attrs["note"] = (
                "uint8 category bitfield, axes (time, channel). "
                "File-level campaign masks are NOT encoded here; "
                "apply select_files.py for those.")
            if freqs_ref is not None:
                h.create_dataset("freqs_mhz", data=freqs_ref)
            g = h.create_group("mask")
            for key, cat in entries.items():
                g.create_dataset(key, data=cat, compression="gzip",
                                 compression_opts=6, shuffle=True)
        written.append(os.path.basename(out))

    with open(os.path.join(out_dir, "flag_bits.json"), "w") as f:
        json.dump({
            "encoding": "uint8 bitfield per (time, channel) sample",
            "axes": ["time", "channel"],
            "n_channels": D.N_CHAN,
            "channel_width_mhz": D.CHAN_WIDTH_MHZ,
            "clean_value": 0,
            "bits": [
                {"bit": 0, "value": int(D.CAL), "name": "cal",
                 "rfi": False,
                 "meaning": "receiver on load/noise/VNA, not on antenna; "
                            "not sky and not RFI"},
                {"bit": 1, "value": int(D.TX_COMB), "name": "tx_comb",
                 "rfi": False,
                 "meaning": "beam-mapping transmitter comb, 1.953125 MHz "
                            "= 8 channels exactly, clock-locked; wanted "
                            "signal, not interference"},
                {"bit": 2, "value": int(D.SELF_RFI), "name": "self-RFI",
                 "rfi": True,
                 "meaning": "self-generated: Panda EMI comb (1.000 MHz, "
                            "walks, box-air only), fan, laptop comb"},
                {"bit": 3, "value": int(D.FM_DTV_MS), "name": "FM-scatter",
                 "rfi": True,
                 "meaning": "FM and DTV bands rising together: "
                            "meteor-scatter propagation"},
                {"bit": 4, "value": int(D.AIRPLANE), "name": "airplane",
                 "rfi": True,
                 "meaning": "broadband short transient in FM/DTV bands; "
                            "morphological, NOT corroborated by ADS-B"},
                {"bit": 5, "value": int(D.ORBCOMM), "name": "orbcomm",
                 "rfi": True, "meaning": "136.5-138.5 MHz satellite downlink"},
                {"bit": 6, "value": int(D.UNKNOWN), "name": "unknown",
                 "rfi": True,
                 "meaning": "detector fired, cause not attributable"},
                {"bit": 7, "value": int(D.OVERFLOW), "name": "overflow",
                 "rfi": False,
                 "meaning": "int32 auto accumulator wrapped (raw sample "
                            "negative); instrumental, not interference. "
                            "Excluded from the RFI denominator."},
            ],
        }, f, indent=2)

    manifest = {
        "provenance": {
            "product": "flags",
            "campaign": "marjum-2026-07",
            "version": VERSION,
            "generated_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "generator": "eigsep_data.flagging.build_masks",
            "generator_commit": git_commit(),
            "inputs": [
                {"path": "data/*.h5", "n_files": len(files)},
                {"path": "curation/mode_table.jsonl",
                 "sha256": sha256(os.path.join(
                     root, "curation", "mode_table.jsonl"))},
            ],
            "params": {
                "transient_clip_sigma": 5.0,
                "transient_median_width": 9,
                "persistent_clip_sigma": 6.0,
                "persistent_smooth_width": 31,
                "comb_snr_threshold": 8.0,
                "broadband_frac": 0.15,
                "meteor_min_frac": 0.05,
                "analysis_band_mhz": list(D.BAND_ANALYSIS),
            },
        },
        "selection_basis": {
            "generated_over": "ALL files in data/, not a select_files.py subset",
            "reason": "masks are orthogonal to file-level campaign masks; "
                      "consumers apply select_files.py for file validity and "
                      "these flags for channel x time RFI within surviving "
                      "files",
            "file_level_masks_encoded_here": False,
        },
        "files_written": written,
        "n_input_files": len(files),
        "n_errors": len(errors),
        "errors": errors[:50],
        "compact": f"marjum-2026-07/flags@{VERSION}+{git_commit()}",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(out_dir, "comb_detections.jsonl"), "w") as f:
        for r in comb_log:
            f.write(json.dumps(r) + "\n")

    _write_summary(out_dir, stats)
    print(f"wrote {len(written)} day files to {out_dir}; "
          f"{len(errors)} errors", file=sys.stderr)


def _accumulate(stats, day, inp, cat, freqs):
    key = f"{day}|{inp}"
    nt = cat.shape[0]
    is_cal = (cat & D.CAL) != 0
    is_ovf = (cat & D.OVERFLOW) != 0
    rfi_bits = D.RFI_BITS
    # "Sky" samples are the ones a science analysis could actually use:
    # neither a calibration state nor an instrumental wrap. RFI
    # fractions are quoted against this denominator, so an accumulator
    # overflow can never inflate the RFI rate.
    sky = ~(is_cal | is_ovf)
    stats[key]["n_sample"] += float(sky.sum())
    stats[key]["n_cal"] += float(is_cal.sum())
    stats[key]["n_overflow"] += float(is_ovf.sum())
    stats[key]["n_rfi"] += float(((cat & rfi_bits) != 0)[sky].sum())
    stats[key]["n_tx"] += float(((cat & D.TX_COMB) != 0).sum())
    for name, bit in [("self", D.SELF_RFI), ("ms", D.FM_DTV_MS),
                      ("airplane", D.AIRPLANE), ("orbcomm", D.ORBCOMM),
                      ("unknown", D.UNKNOWN)]:
        stats[key][f"n_{name}"] += float(((cat & bit) != 0)[sky].sum())
    for bname, (lo, hi) in REPORT_BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        sub = cat[:, sel]
        subsky = sky[:, sel]
        stats[key][f"band_{bname}_n"] += float(subsky.sum())
        stats[key][f"band_{bname}_rfi"] += float(
            ((sub & rfi_bits) != 0)[subsky].sum())
    stats[key]["n_time"] += nt


def _write_summary(out_dir, stats):
    summary = {"per_day_input": {}, "per_band_day": {}}
    for key, s in sorted(stats.items()):
        day, inp = key.split("|")
        n = max(s["n_sample"], 1.0)
        summary["per_day_input"][key] = {
            "n_sky_samples": int(s["n_sample"]),
            "n_cal_samples": int(s["n_cal"]),
            "n_overflow_samples": int(s["n_overflow"]),
            "frac_flagged_rfi": round(s["n_rfi"] / n, 5),
            "frac_kept": round(1.0 - s["n_rfi"] / n, 5),
            "frac_tx_comb": round(s["n_tx"] / n, 5),
            "by_category": {
                c: round(s[f"n_{c}"] / n, 5)
                for c in ("self", "ms", "airplane", "orbcomm", "unknown")
            },
        }
        bands = {}
        for bname in REPORT_BANDS:
            bn = s.get(f"band_{bname}_n", 0.0)
            if bn > 0:
                bands[bname] = round(s[f"band_{bname}_rfi"] / bn, 5)
        summary["per_band_day"][key] = bands
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
