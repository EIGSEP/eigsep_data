"""Bounded raw width/control diagnostic for the established 1MHz comb mask."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eigsep_data.index import scan_corr_file
from eigsep_observing.io import read_hdf5

plt.switch_backend("Agg")

NAMES = [
    "corr_20260716_011800Z.h5",
    "corr_20260717_085328Z.h5",
    "corr_20260717_085537Z.h5",
]
ROW_TABLE_SHA = (
    "17c27df153a7810b79754a1d4a6af02ccb4da7cce8fa2f6f69122b8ea2a66ea5"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(a):
    valid = np.isfinite(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, a, 0).sum(0) / valid.sum(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--candidate-rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha(args.candidate_rows) != ROW_TABLE_SHA:
        raise ValueError("authoritative row-table bytes differ")
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("commit bounded diagnostic source before execution")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = pd.read_parquet(args.candidate_rows)
    data, inputs = {}, {str(args.candidate_rows): ROW_TABLE_SHA}
    for name in NAMES:
        path = args.raw_dir / name
        inputs[str(path)] = sha(path)
        powers, header, _ = read_hdf5(path)
        metadata = scan_corr_file(path, streams=["rfswitch"])
        p = np.asarray(powers["4"], float)
        p[p <= 0] = np.nan
        selected = rows[(rows.file == name) & (rows.key.astype(str) == "4")]
        mask = np.isin(np.arange(len(p)), selected.row.to_numpy())
        assert np.array_equal(metadata.row.to_numpy(), np.arange(len(p)))
        data[name] = dict(
            power=p,
            selected=mask,
            state=metadata.rfswitch.to_numpy(),
            time_best=metadata.time_best.to_numpy(),
            frequency_mhz=np.asarray(header["freqs"], float),
        )
    cases = [
        (NAMES[0], "RFANT", NAMES[0]),
        (NAMES[1], "RFNON", NAMES[1]),
        (NAMES[2], "RFNON", NAMES[1]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), constrained_layout=True)
    records = []
    edges = np.linspace(-0.5, 0.5, 21)
    arrays = {}
    for k, (name, state, reference) in enumerate(cases):
        d, c = data[name], data[reference]
        f = d["frequency_mhz"]
        np.testing.assert_array_equal(f, c["frequency_mhz"])
        test = d["selected"] & (d["state"] == state)
        control = ~c["selected"] & (c["state"] == state)
        if not test.any() or not control.any():
            raise ValueError("missing same-state non-detected control")
        pt, pc = d["power"][test], c["power"][control]
        st, sc = mean(pt), mean(pc)
        band = (f >= 30) & (f <= 245)

        # Compare independent row-group means at the SAME native channel.
        # A local neighbour denominator can itself contain another comb tooth.
        with np.errstate(divide="ignore", invalid="ignore"):
            contrast = st / sc - 1
        offset = f - np.round(f)
        use = band
        core = use & (np.abs(offset) < 0.15)
        neighbour = use & (np.abs(offset) >= 0.15) & (np.abs(offset) < 0.4)
        rec = dict(
            file=name,
            state=state,
            control_file=reference,
            selected_rows=int(test.sum()),
            control_rows=int(control.sum()),
            all_detected_rows=int(d["selected"].sum()),
            control="same switch state, not detected; not certified clean",
            core_mean_contrast=float(np.nanmean(contrast[core])),
            neighbour_mean_contrast=float(np.nanmean(contrast[neighbour])),
            neighbour_max_contrast=float(np.nanmax(contrast[neighbour])),
            neighbour_peak_frequency_mhz=float(
                f[neighbour][np.nanargmax(contrast[neighbour])]
            ),
        )
        records.append(rec)
        arrays["frequency_mhz"] = f
        arrays[f"{k}_detected_rows"] = np.flatnonzero(test)
        arrays[f"{k}_control_rows"] = np.flatnonzero(control)
        arrays[f"{k}_detected_time_best"] = d["time_best"][test]
        arrays[f"{k}_control_time_best"] = c["time_best"][control]
        folded = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = use & (offset >= lo) & (offset < hi)
            folded.append(
                float(np.nanmean(contrast[m])) if m.any() else np.nan
            )
        axes[k, 1].scatter(
            offset[use],
            contrast[use],
            s=4,
            alpha=0.25,
            color="0.3",
            label="native channels",
        )
        axes[k, 1].plot(
            (edges[:-1] + edges[1:]) / 2,
            folded,
            "o-",
            ms=3,
            label="mean contrast",
        )
        axes[k, 1].axhline(0, color="0.5", lw=0.6)
        arrays[f"{k}_contrast_folded"] = folded
        arrays[f"{k}_native_contrast"] = contrast
        axes[k, 0].plot(f[band], st[band], lw=0.8, label="detected")
        axes[k, 0].plot(f[band], sc[band], lw=0.8, label="control")
        axes[k, 0].set_yscale("log")
        axes[k, 0].set_title(
            f"{name[5:20]} {state}: {test.sum()}/{control.sum()} rows"
        )
        axes[k, 0].set(xlabel="frequency [MHz]", ylabel="raw power [counts]")
        axes[k, 1].axvspan(
            -0.15,
            0.15,
            color="0.8",
            alpha=0.5,
            label="existing mask" if k == 0 else None,
        )
        axes[k, 1].set(
            xlim=(-0.5, 0.5),
            xlabel="offset from integer MHz [MHz]",
            ylabel="detected mean / control mean − 1",
        )
        axes[k, 1].set_yscale("symlog", linthresh=0.05)
        axes[k, 0].legend(fontsize=8)
        axes[k, 1].legend(fontsize=8)
    limits = [ax.get_ylim() for ax in axes[:, 0]]
    for ax in axes[:, 0]:
        ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
    limits = [ax.get_ylim() for ax in axes[:, 1]]
    for ax in axes[:, 1]:
        ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
    fig.savefig(args.output / "raw_comb_width.pdf")
    fig.savefig(args.output / "raw_comb_width.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(
        args.output / "profiles.npz", **arrays, edges_mhz=edges
    )
    report = dict(
        label="diagnostic",
        input_sha256=inputs,
        cases=records,
        commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        mask="exact detected rows, input4, all states, abs(f-round(f))<.15MHz",
        limitations="No flags applied to raw proof except nonpositive power; "
        "controls not certified clean, cross-file third control, "
        "descriptive means do not establish false positives "
        "or full comb support",
    )
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output / "CAPTION.txt").write_text(
        "Three bounded raw input4 comparisons, 30–245 MHz, using the pinned "
        "native-row comb detections. Left: detected and same-state "
        "non-detected "
        "mean raw spectra; right: their ratio at the same native frequency, "
        "minus one, folded around integer MHz. Shading is the "
        "unchanged +/-0.15MHz window. Nonpositive power is omitted, no RFI "
        "flags or fills applied. The third control is the earlier file. "
        "This tests observed widths and neighbours, "
        "not complete detector validation.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
