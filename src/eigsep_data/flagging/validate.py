"""Validate the v0 flag masks in both error directions.

PROGRAM.md §8: "If it's a foreground-separation result: where is the
signal-loss number?"  A flagger without one is not a result either, so
this reports three things:

1. **Labelled-window recall.** Does the detector fire where CAMPAIGN.md
   says an emitter was, and stay quiet in matched control windows?

2. **Sky removed (false positives).** Two measurements: the flagged
   fraction in RFI-quiet data, which bounds over-flagging; and a direct
   injection test showing that a smooth 21-cm-like absorption feature
   passes through the detector untouched.  Over-flagging is not
   conservative -- it is signal loss.

3. **Leakage (false negatives).** Narrowband spikes of known amplitude
   are injected into quiet data and the detection probability is
   measured as a function of amplitude, giving the completeness curve
   and the 50%/90% detection thresholds.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import h5py
import numpy as np

from ..paths import campaign_data_dir, get_campaign_root
from . import detectors as D


def _root():
    """The campaign root, resolved at call time, never at import."""
    return str(get_campaign_root(required=True))


def _data():
    return str(campaign_data_dir())

# Windows CAMPAIGN.md labels, plus matched controls on the same day.


LABELLED = {
    "comb-1.25MHz (claimed)": ("2026-07-16T01:00", "2026-07-16T01:30"),
    "laptop-comb 145-160": ("2026-07-13T00:00", "2026-07-13T23:59"),
    "thunderstorm": ("2026-07-16T00:00", "2026-07-16T06:00"),
    "lidar-sweep": ("2026-07-18T01:37", "2026-07-18T03:00"),
    "digital-comb (found)": ("2026-07-17T15:37", "2026-07-18T03:00"),
}
CONTROLS = {
    "control 07-16 midday": ("2026-07-16T12:00", "2026-07-16T15:00"),
    "control 07-17 early": ("2026-07-17T06:00", "2026-07-17T10:00"),
}


def files_between(t0, t1):
    out = []
    for p in sorted(glob.glob(os.path.join(_data(), "*.h5"))):
        try:
            t = D.file_close_time(os.path.basename(p))
        except ValueError:
            continue
        if t0 <= t.strftime("%Y-%m-%dT%H:%M") <= t1:
            out.append(p)
    return out


def load_logp(path, key=None):
    with h5py.File(path, "r") as h:
        freqs = h["header/freqs"][:]
        keys = sorted(k for k in h["data"] if len(k) == 1)
        if key is None:
            key = "4" if "4" in keys else keys[0]
        if key not in keys:
            return None, None, None
        raw = h["data/" + key][:]
        rfsw = None
        if "metadata" in h and "rfswitch" in h["metadata"]:
            rfsw = h["metadata/rfswitch"][()]
        ant = D.antenna_mask(rfsw, raw.shape[0])
    logp = np.log10(np.maximum(raw.astype(np.float64), 1.0))
    if np.median(logp) <= 0:
        return None, None, None
    return logp, freqs, ant


def flag_fraction(logp, freqs, ant, tx_on=False):
    """Fraction of in-band samples flagged as *interference*.

    Runs the full categorisation, then counts only the RFI bits. The
    TX comb, calibration states and accumulator overflow are excluded:
    counting the beam-mapping transmitter as interference would make
    every TX-on window look catastrophically contaminated, which is
    how an earlier version of this check reported a quiet control
    window as dirtier than the thunderstorm.
    """
    pix, _, _ = D.transient_track(logp, ant)
    chan, _ = D.persistent_track(logp, ant, freqs)
    med = np.median(logp[ant], axis=0) if ant.sum() >= 4 \
        else np.median(logp, axis=0)
    combs = D.detect_combs(med, freqs, tx_on=tx_on)
    bb = D.broadband_times(pix, freqs)
    ms = D.meteor_scatter_times(pix, freqs)
    cat = D.categorise(pix, chan, freqs, combs, tx_on, bb, ms)
    cat[~ant, :] |= D.CAL
    sel = (freqs >= D.BAND_ANALYSIS[0]) & (freqs <= D.BAND_ANALYSIS[1])
    sub = cat[:, sel]
    sky = (sub & (D.CAL | D.OVERFLOW)) == 0
    if sky.sum() == 0:
        return float("nan")
    return float(((sub & D.RFI_BITS) != 0)[sky].mean())


# ------------------------------------------------------------- 1. recall

def labelled_recall(max_files=24):
    rows = []
    for name, (t0, t1) in {**LABELLED, **CONTROLS}.items():
        fs = files_between(t0, t1)
        if not fs:
            rows.append((name, 0, float("nan"), float("nan")))
            continue
        step = max(len(fs) // max_files, 1)
        fs = fs[::step][:max_files]
        fracs, combs = [], []
        for p in fs:
            logp, freqs, ant = load_logp(p)
            if logp is None:
                continue
            fracs.append(flag_fraction(logp, freqs, ant))
            med = np.median(logp[ant], axis=0) if ant.sum() >= 4 \
                else np.median(logp, axis=0)
            combs.append(D.detect_combs(med, freqs))
        fracs = np.asarray(fracs, dtype=float)
        if fracs.size == 0 or not np.isfinite(fracs).any():
            continue
        # An all-calibration file yields NaN by design; skip those
        # rather than letting one poison the window median.
        rows.append((name, int(np.isfinite(fracs).sum()),
                     float(np.nanmedian(fracs)), combs))
    return rows


# --------------------------------------------------- 2. sky removed

def sky_removal(quiet_window=("2026-07-17T06:00", "2026-07-17T10:00"),
                n_files=12):
    """False-positive rate, and a smooth-signal pass-through test."""
    fs = files_between(*quiet_window)[:n_files]
    base, injected, amps = [], [], []
    for p in fs:
        logp, freqs, ant = load_logp(p)
        if logp is None:
            continue
        base.append(flag_fraction(logp, freqs, ant))

        # A 21-cm-like absorption trough: -200 mK, 78 MHz centre,
        # ~15 MHz width, on a foreground of order 10^3-10^4 K. Injected
        # multiplicatively in power, i.e. additively in log space.
        t21 = -0.200 / 3000.0  # fractional depth against a ~3000 K sky
        prof = t21 * np.exp(-0.5 * ((freqs - 78.0) / 7.5) ** 2)
        logp2 = logp + np.log10(1.0 + prof)[None, :]
        injected.append(flag_fraction(logp2, freqs, ant))
        amps.append(float(np.abs(prof).max()))
    return {
        "n_files": len(base),
        "frac_flagged_clean": round(float(np.median(base)), 6),
        "frac_flagged_with_21cm": round(float(np.median(injected)), 6),
        "delta": round(float(np.median(injected) - np.median(base)), 8),
        "injected_fractional_depth": round(float(np.max(amps)), 8)
        if amps else None,
    }


# ------------------------------------------------------ 3. leakage

def leakage_curve(quiet_window=("2026-07-17T06:00", "2026-07-17T10:00"),
                  n_files=8, amps_sigma=(1, 2, 3, 4, 5, 6, 8, 12),
                  n_inject=40, seed=0):
    """Detection probability vs injected narrowband spike amplitude.

    Amplitudes are in units of the per-channel temporal MAD, which is
    the quantity the detector actually thresholds on, so the curve is
    directly interpretable as "what survives the cut".
    """
    rng = np.random.default_rng(seed)
    fs = files_between(*quiet_window)[:n_files]
    hits = {a: [0, 0] for a in amps_sigma}
    for p in fs:
        logp, freqs, ant = load_logp(p)
        if logp is None:
            continue
        sel = np.where((freqs >= D.BAND_ANALYSIS[0])
                       & (freqs <= D.BAND_ANALYSIS[1]))[0]
        _, _, scale = D.transient_track(logp, ant)
        good_t = np.where(ant)[0]
        if good_t.size < 20:
            continue
        for a in amps_sigma:
            for _ in range(n_inject):
                ch = int(rng.choice(sel))
                ti = int(rng.choice(good_t))
                s = scale[ch]
                if not np.isfinite(s) or s <= 0:
                    continue
                test = logp.copy()
                test[ti, ch] += a * s
                pix, _, _ = D.transient_track(test, ant)
                hits[a][1] += 1
                if pix[ti, ch]:
                    hits[a][0] += 1
    return {
        f"{a}sigma": {
            "n_trials": hits[a][1],
            "detected": hits[a][0],
            "detection_prob": round(hits[a][0] / hits[a][1], 4)
            if hits[a][1] else None,
        }
        for a in amps_sigma
    }


def main():
    out = {}
    print("=== 1. labelled-window recall ===", file=sys.stderr)
    rows = labelled_recall()
    rec = {}
    for name, n, frac, combs in rows:
        if isinstance(combs, list) and combs:
            best = {k: round(float(np.median([c[k]["snr"] for c in combs])), 2)
                    for k in combs[0]}
        else:
            best = {}
        rec[name] = {"n_files": n, "median_flag_frac": round(frac, 5)
                     if frac == frac else None, "median_comb_snr": best}
        print(f"  {name:28s} n={n:3d} flagfrac={frac:.4f} {best}",
              file=sys.stderr)
    out["labelled_recall"] = rec

    print("=== 2. sky removed ===", file=sys.stderr)
    out["sky_removal"] = sky_removal()
    print(" ", out["sky_removal"], file=sys.stderr)

    print("=== 3. leakage ===", file=sys.stderr)
    out["leakage"] = leakage_curve()
    for k, v in out["leakage"].items():
        print(f"  {k:8s} {v}", file=sys.stderr)

    dest = os.path.join(_root(), "flags", "v0", "validation.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
