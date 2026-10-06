"""What if, in raster rows, the pointing background replaced v3's auto
detectors and support check (bits 2, 3, 7) instead of adding bit 10?

Replacement mask on moving sky rows: bits 1, 4, 5, 6 from v3; positive excess
= bit 10; negative = zp/spread < -negative_cut or nonfinite; support = the
pointing model finite and > 0; then the v3 time guard. Reports the excluded
fraction against v3, its elevation dependence, and recovery of planted
30/60/100-sigma lines in cells clean under each mask. Disproof: excluded
fraction not much below v3's 73%, flags piling up by elevation, or recovery
worse than v3 at the same amplitude.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
from dataclasses import replace
from scipy.ndimage import maximum_filter
import eigsep_data.rfi_supported as rs
from eigsep_data import MetadataIndex
from eigsep_data.antenna_policy import AntennaResolutionPolicy

captured = {}
orig = rs._pointing_background
def capture(*a, **k):
    model, stats = orig(*a, **k)
    captured["model"] = model
    return model, stats
rs._pointing_background = capture

C = "/mnt/data02/eigsep/marjum-2026-07/"
BATCHES = dict(a=("corr_20260717_203032Z.h5", "corr_20260717_204952Z.h5"),
               b=("corr_20260717_205410Z.h5", "corr_20260717_211329Z.h5"))
idx = MetadataIndex(C + "data")
pol = AntennaResolutionPolicy.load(C + "curation/antenna_resolution.json")
pt = pq.read_table(C + "curation/pointing_table.parquet", columns=["file", "sample_idx", "el_deg"]).to_pandas()
cfg = replace(rs.RFIConfig(), pointing_background=True)
rng = np.random.default_rng(23)
B10, B9 = 1 << rs.BIT_BY_REASON["pointing_line"], 1 << rs.BIT_BY_REASON["high_scatter"]


def masks(r, data, el, states, dt):
    sky = (states == "RFANT") & np.isfinite(el)
    v3 = ((r.flags & ~np.uint16(B10 | B9)) != 0)
    norm = np.sqrt(2 * dt[:, None] * np.median(np.diff(r.freqs_mhz)) * 1e6)
    model = captured["model"]
    zp = rs.residual_z(data, model, norm)
    R = r.reasons
    tested = ~R["invalid_input_or_domain"] & np.isfinite(zp) & np.isfinite(model) & (model > 0)
    _, spread = rs._temporal_center_scale(zp, tested & sky[:, None])
    point = zp / spread
    base = (R["invalid_input_or_domain"] | R["cross_change"] | R["band_group_trigger"]
            | R["comb_group_trigger"] | ~tested
            | (point > rs._point_thresholds(r.freqs_mhz, cfg))
            | (point < -cfg.negative_cut))
    rep = maximum_filter(base, size=(2 * cfg.time_guard + 1, 1), mode="constant")
    new = v3.copy()
    new[sky] = rep[sky]
    return sky, v3, new, point, spread


out = {}
for label, files in BATCHES.items():
    sel = idx.select(files=files); sel = idx.select(files=sel.files)
    Bn = rs.load_selection_inputs(sel, resolution_policy=pol)
    a = Bn["air"]
    states, _ = rs.switch_states(a.meta, cfg)
    el = a.meta[["file", "row"]].merge(pt, left_on=["file", "row"], right_on=["file", "sample_idx"], how="left").el_deg.to_numpy(float)
    dt = a.meta.integration_time.to_numpy(float)
    args = (Bn["ground"].data, Bn["cross"].data, a.t, a.freqs_mhz, dt, states)
    data = a.data.astype(float)
    r = rs.flag_arrays(data, *args, config=cfg, elevation_deg=el)
    sky, v3, new, point, spread = masks(r, data, el, states, dt)
    bins = np.arange(-90, 91, 30); which = np.digitize(el, bins)
    res = dict(v3_excluded=float(v3[sky].mean()), replace_excluded=float(new[sky].mean()),
               both_clean=float((~v3 & ~new)[sky].mean()),
               v3_clean_replace_flags=float((~v3 & new)[sky].mean()),
               spread_median=float(np.median(spread)), spread_p90=float(np.percentile(spread, 90)),
               replace_by_el={f"{bins[k-1]}..{bins[k]}": float(new[(which == k) & sky].mean()) for k in range(1, len(bins))},
               v3_by_el={f"{bins[k-1]}..{bins[k]}": float(v3[(which == k) & sky].mean()) for k in range(1, len(bins))})
    print(label, json.dumps(res), flush=True)
    norm = np.sqrt(2 * dt * np.median(np.diff(a.freqs_mhz)) * 1e6)
    rec = {}
    pools = {"clean_in_v3": sky[:, None] & ~v3 & np.isfinite(r.model_raw),
             "clean_in_replace_only": sky[:, None] & v3 & ~new & np.isfinite(r.model_raw)}
    inj = data.copy(); planted = {}
    for pname, pool in pools.items():
        rr, cc = np.nonzero(pool)
        pick = rng.choice(len(rr), 600, replace=False)
        amps = np.repeat([30.0, 60.0, 100.0], 200)
        inj[rr[pick], cc[pick]] += amps / norm[rr[pick]] * r.model_raw[rr[pick], cc[pick]]
        planted[pname] = (rr[pick], cc[pick], amps)
    ri = rs.flag_arrays(inj, *args, config=cfg, elevation_deg=el)
    _, v3i, newi, _, _ = masks(ri, inj, el, states, dt)
    for pname, (rr, cc, amps) in planted.items():
        for A in (30.0, 60.0, 100.0):
            m = amps == A
            rec[f"{pname} {A:g}"] = dict(v3=float(v3i[rr[m], cc[m]].mean()), replace=float(newi[rr[m], cc[m]].mean()))
    res["recovery"] = rec
    print(label, "recovery", json.dumps(rec), flush=True)
    out[label] = res
json.dump(out, open(sys.argv[1], "w"), indent=1)
