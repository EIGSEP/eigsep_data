"""Validate pointing_line (bit 10) on the 07-17 raster batches.

Clean run: v3 vs pointing scatter, bit-10 flag fraction, its spread over
elevation bins (flags piling up at particular elevations would be model
failure, not RFI). Injection run: 300 single-cell lines each at 30/60/100
radiometer sigma; recovery by the v3 bits alone vs the default mask with
bit 10. Disproof conditions: scatter not reduced; bit-10 fraction >~1% or
strongly elevation-dependent; no recovery gain.
"""
import json, sys, time
import numpy as np
import pyarrow.parquet as pq
from dataclasses import replace
from eigsep_data import MetadataIndex
from eigsep_data.antenna_policy import AntennaResolutionPolicy
from eigsep_data.rfi_supported import (RFIConfig, BIT_BY_REASON, load_selection_inputs,
                                       flag_arrays, switch_states)

C = "/mnt/data02/eigsep/marjum-2026-07/"
BATCHES = dict(a=("corr_20260717_203032Z.h5", "corr_20260717_204952Z.h5"),
               b=("corr_20260717_205410Z.h5", "corr_20260717_211329Z.h5"))
idx = MetadataIndex(C + "data")
pol = AntennaResolutionPolicy.load(C + "curation/antenna_resolution.json")
pt = pq.read_table(C + "curation/pointing_table.parquet", columns=["file", "sample_idx", "el_deg"]).to_pandas()
cfg = replace(RFIConfig(), pointing_background=True)
bit10 = 1 << BIT_BY_REASON["pointing_line"]
out = {}
rng = np.random.default_rng(17)
for label, files in BATCHES.items():
    sel = idx.select(files=files); sel = idx.select(files=sel.files)
    B = load_selection_inputs(sel, resolution_policy=pol)
    a = B["air"]
    states, _ = switch_states(a.meta, cfg)
    el = a.meta[["file", "row"]].merge(pt, left_on=["file", "row"], right_on=["file", "sample_idx"], how="left").el_deg.to_numpy(float)
    dt = a.meta.integration_time.to_numpy(float)
    args = (B["ground"].data, B["cross"].data, a.t, a.freqs_mhz, dt, states)
    t0 = time.time()
    r = flag_arrays(a.data, *args, config=cfg, elevation_deg=el)
    secs = time.time() - t0
    seg = [s["pointing"] for s in r.diagnostics["segments"] if s.get("pointing")]
    sky = (states == "RFANT") & np.isfinite(el)
    line = r.reasons["pointing_line"]
    v3only = (r.flags & ~np.uint16(bit10 | (1 << BIT_BY_REASON["high_scatter"]))) != 0
    new = line & ~v3only
    bins = np.arange(-90, 91, 15)
    which = np.digitize(el, bins)
    by_el = {f"{bins[k-1]}..{bins[k]}": float(new[(which == k) & sky].mean())
             for k in range(1, len(bins)) if ((which == k) & sky).sum() > 50}
    chan = new[sky].mean(0)
    top = np.argsort(chan)[::-1][:10]
    res = dict(segments=seg, seconds=round(secs),
               sky_rows=int(sky.sum()),
               v3_excluded=float(v3only[sky].mean()),
               bit10=float(line[sky].mean()), bit10_new=float(new[sky].mean()),
               new_by_elevation=by_el,
               top_channels={f"{r.freqs_mhz[c]:.3f}": float(chan[c]) for c in top},
               high_scatter=float(r.reasons["high_scatter"][sky].mean()))
    print(label, json.dumps(res), flush=True)
    # injections on cells that are clean in the clean run
    clean = sky[:, None] & ~r.mask
    clean &= np.isfinite(r.model)
    rr, cc = np.nonzero(clean)
    pick = rng.choice(len(rr), 900, replace=False)
    amps = np.repeat([30.0, 60.0, 100.0], 300)
    data = a.data.astype(float).copy()
    norm = np.sqrt(2 * dt * np.median(np.diff(a.freqs_mhz)) * 1e6)
    data[rr[pick], cc[pick]] += amps / norm[rr[pick]] * r.model[rr[pick], cc[pick]]
    ri = flag_arrays(data, *args, config=cfg, elevation_deg=el)
    v3i = (ri.flags & ~np.uint16(bit10 | (1 << BIT_BY_REASON["high_scatter"]))) != 0
    rec = {}
    for A in (30.0, 60.0, 100.0):
        m = amps == A
        cells = (rr[pick][m], cc[pick][m])
        rec[f"{A:g}"] = dict(v3=float(v3i[cells].mean()), with_bit10=float(ri.mask[cells].mean()),
                             bit10=float(ri.reasons["pointing_line"][cells].mean()))
    res["recovery"] = rec
    print(label, "recovery", json.dumps(rec), flush=True)
    out[label] = res
json.dump(out, open(sys.argv[1], "w"), indent=1)
