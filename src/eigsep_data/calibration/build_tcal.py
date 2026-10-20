"""Build ``derived/tcal/<version>/``: the field temperature calibration.

Collects, for the stretches of a campaign where the RF switch was cycling
through its calibrators, everything :mod:`eigsep_data.products.tcal` needs
to put an integration of the switched input on a kelvin scale at
reference plane P:

- one masked, integration-time-weighted mean spectrum per ambient-load
  (RFAMB) and noise-source (RFNON) visit, labelled with its receiver
  regime from the curated calibration windows;
- every field S11 sweep of the antenna, the ambient load and the receiver,
  calibrated to plane P with ``eigsep_cal.S11`` and put on the correlator
  channels;
- every load-thermistor reading in the same span;
- the noise-source excess as the receiver takes it, ``t_ns_k``: an assumed
  value, or one value per channel built from a bench-measured
  ``derived/noise_source/<version>/`` (:func:`noise_source_tns`), applied
  to every integration.

The campaign's choices (span, switched input, flag product, visit
rejection, interpolation limits, the noise source) are the version's
``config.json``; the manifest records it. It writes ``solutions.npz`` and
``manifest.json``; the hand-written ``README.md`` beside them says what the
product is for. Run::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 eigsep-build-tcal v0004

The calibration itself (interpolation, limits, equations) is applied when a
row is loaded, by :mod:`eigsep_data.products.tcal` and ``eigsep_cal.dicke``;
nothing here computes a temperature.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..index import MetadataIndex
from ..paths import get_campaign_root
from ._common import code_provenance, load_config, product_dir, sha256, unix


def regime_of(files, windows):
    """Receiver regime of each visit, from the curated cal windows.

    Matched on file, not time: the windows are bounded by filenames, which
    are file close times and trail the rows by a minute or two. A visit
    takes the label of the window whose ``file_first`` .. ``file_last``
    range holds its first file; a visit in no window gets none and is
    dropped rather than guessed.
    """
    out = []
    for f in files:
        hit = windows[(windows.file_first <= f) & (windows.file_last >= f)]
        out.append(hit.receiver_regime.iloc[0] if len(hit) else "")
    return np.array(out, dtype=str)


def select_span(index, cfg, **kw):
    sel = index.select(time=tuple(cfg["span_utc"]), **kw)
    return sel.select(files=tuple(cfg["files"]))


def visits(index, state, windows, cfg):
    key = cfg["key"]
    sel = select_span(index, cfg, rfswitch=state)
    loaded = sel.load(keys=[key])
    p = np.asarray(loaded.data[key], dtype=float)
    t = loaded.meta.time_best.to_numpy(dtype=float)
    tau = loaded.meta.integration_time.to_numpy(dtype=float)
    files = loaded.meta.file.to_numpy(str)
    order = np.argsort(t, kind="stable")
    p, t, tau, files = p[order], t[order], tau[order], files[order]
    rfi = rfi_on_full_axis(
        sel, t, np.asarray(loaded.freq, dtype=float), index, cfg
    )
    masked_frac = []
    usable = ~np.all(p == 0, axis=1)
    starts = np.flatnonzero(np.r_[True, np.diff(t) > cfg["max_row_gap_s"]])
    stops = np.r_[starts[1:], t.size]
    rows = []
    for a, b in zip(starts, stops):
        use = a + np.flatnonzero(usable[a:b])
        spec = None
        if use.size:
            spec, frac = masked_mean(p[use], tau[use], rfi[use])
            masked_frac.append(frac)
        rows.append(
            {
                "t": (
                    np.average(t[use], weights=tau[use])
                    if use.size
                    else np.nan
                ),
                "file": files[a],
                "n": use.size,
                "p": spec,
            }
        )
    v = pd.DataFrame(rows)
    v["regime"] = regime_of(v.file, windows)
    regimes = tuple(cfg["regimes"])
    keep = (v.n >= cfg["min_visit_rows"]) & v.regime.isin(regimes)
    log = {
        "visits_found": int(len(v)),
        "dropped_partial": int((v.n < cfg["min_visit_rows"]).sum()),
        "dropped_regime": int(
            ((v.n >= cfg["min_visit_rows"]) & ~v.regime.isin(regimes)).sum()
        ),
        "kept": int(keep.sum()),
        "rows_per_visit": [int(v.n[keep].min()), int(v.n[keep].max())],
        "masked_sample_fraction": {
            "median_over_visits": float(np.median(masked_frac)),
            "max_over_visits": float(np.max(masked_frac)),
        },
    }
    v = v[keep].reset_index(drop=True)
    freqs = np.asarray(loaded.freq, dtype=float)
    bad, v = contaminated(v, freqs, cfg["visit_rejection"])
    log["dropped_contaminated"] = bad
    log["kept"] = int(len(v))
    return v, freqs, log


def rfi_on_full_axis(sel, t, freqs, index, cfg):
    """The flag product's RFI bits on the full correlator axis.

    The flag cubes cover only their tested domain; channels outside it carry
    no RFI information and are left unflagged. Rows are matched to the raw
    load by time, which must agree exactly.
    """
    fb = sel.load_bundle(
        key=cfg["key"], root=index_root(index), products=[cfg["flags"]]
    )
    skipped = fb.provenance["products"]["flags"]["skipped"]
    if skipped:
        raise SystemExit(f"{cfg['flags']} missing for {skipped}")
    order = np.argsort(fb.t, kind="stable")
    if not np.array_equal(fb.t[order], t):
        raise SystemExit("flag rows do not align with the raw rows")
    lo = int(np.argmin(np.abs(freqs - fb.freqs_mhz[0])))
    assert np.allclose(freqs[lo : lo + fb.freqs_mhz.size], fb.freqs_mhz)
    rfi = np.zeros((t.size, freqs.size), dtype=bool)
    rfi[:, lo : lo + fb.freqs_mhz.size] = (
        fb.flags[order] & cfg["rfi_mask"]
    ) != 0
    return rfi


def index_root(index):
    """The campaign tree the index's raw data (and flag cubes) live in."""
    return Path(index.data_dir).resolve().parent


def masked_mean(p, tau, rfi):
    """Integration-time-weighted mean of one visit, masked per sample.

    Masks the flag product's RFI and wrapped (negative) samples. Returns the
    spectrum (NaN in a channel with no unmasked sample) and the fraction of
    samples masked.
    """
    bad = rfi | (p < 0)
    w = np.where(bad, 0.0, tau[:, None])
    wsum = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        spec = np.where(wsum > 0, (w * p).sum(axis=0) / wsum, np.nan)
    return spec, float(bad.mean())


def contaminated(v, freqs, rej):
    """Drop visits that disagree with their neighbours in many channels.

    A visit is dropped when more than ``max_bad_channels`` channels in
    ``band_mhz`` sit more than ``frac`` from the median of its neighbouring
    visits (up to two either side, same regime). Returns the dropped visits
    (UTC and channel count) and the rest. (The median is a flagging
    statistic here, not an estimate.)
    """
    band = (freqs >= rej["band_mhz"][0]) & (freqs < rej["band_mhz"][1])
    p = np.stack(v.p.to_list())
    n_bad = np.zeros(len(v), dtype=int)
    for i in range(len(v)):
        nb = [
            j
            for j in range(i - 2, i + 3)
            if j != i and 0 <= j < len(v) and v.regime[j] == v.regime[i]
        ]
        if len(nb) < 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = p[i][band] / np.median(p[nb][:, band], axis=0)
        n_bad[i] = int(np.sum(np.abs(r - 1) > rej["frac"]))
    drop = n_bad > rej["max_bad_channels"]
    dropped = [
        {
            "utc": datetime.fromtimestamp(t, timezone.utc).strftime(
                "%m-%d %H:%M"
            ),
            "channels_off": int(n),
        }
        for t, n in zip(v.t[drop], n_bad[drop])
    ]
    return dropped, v[~drop].reset_index(drop=True)


def s11_on_channels(s11_dir, switch_path, osl_path, freqs_mhz, cfg):
    from eigsep_cal import S11

    s = S11(s11_dir).calibrate(switch_path, osl_path)
    fs = s.freqs / 1e6
    split = unix(cfg["s11_epoch_split_utc"])
    span = [unix(x) for x in cfg["span_utc"]]
    out, log = {}, {"skipped_files": [list(x) for x in s.skipped]}
    for dut in ("ant", "amb", "rec"):
        t, g = s.get_all_s11s(dut, "default")
        t = np.asarray(t, dtype=float)
        keep = (t >= span[0]) & (t < span[1])
        t, g = t[keep], np.asarray(g)[keep]
        on = np.array(
            [
                np.interp(freqs_mhz, fs, x.real, left=np.nan, right=np.nan)
                + 1j
                * np.interp(freqs_mhz, fs, x.imag, left=np.nan, right=np.nan)
                for x in g
            ]
        )
        epoch = np.where(t < split, "pre-gap", "post-gap")
        out[dut] = (t, on, epoch)
        log[dut] = {
            "sweeps": int(t.size),
            "plane": "lna (P)" if dut != "rec" else "dut (P, receiver)",
            "utc": [
                datetime.fromtimestamp(x, timezone.utc).strftime("%m-%d %H:%M")
                for x in t
            ],
        }
    # The hold limit must keep a held sweep inside its own epoch.
    for dut, (t, _, epoch) in out.items():
        pre, post = t[epoch == "pre-gap"], t[epoch == "post-gap"]
        if pre.size and post.size:
            assert (
                cfg["limits"]["s11_hold_s"] < (post.min() - pre.max()) / 2
            ), dut
    return out, log


def thermistor(index, cfg):
    sel = index.select(time=tuple(cfg["span_utc"]))
    m = sel.meta
    t = m.time_best.to_numpy(dtype=float)
    c = pd.to_numeric(m[cfg["load_thermistor_c"]], errors="coerce").to_numpy(
        float
    )
    ok = np.isfinite(t) & np.isfinite(c)
    order = np.argsort(t[ok], kind="stable")
    tt, kk = t[ok][order], c[ok][order] + 273.15
    # Collapse duplicate timestamps (one row per file per integration).
    tt, first = np.unique(tt, return_index=True)
    return (
        tt,
        kk[first].astype(np.float32),
        {
            "readings": int(tt.size),
            "rows_without_reading": int((~ok).sum()),
            "range_k": [float(kk.min()), float(kk.max())],
        },
    )


def noise_source_tns(ns, freqs, s11, therm_k, check_band_mhz):
    """``t_ns_k`` per channel from a ``noise_source`` product.

    ``T_NS = M_NS (T_load + T_ex) - M_amb T_load``
    (``eigsep_cal.noise_source.tns_as_received``): the bench excess at P (the
    product's smooth fit), the noise-source path's reflection at P (lab) and
    the field receiver's and ambient load's reflections at P, averaged over
    the campaign's sweeps, with T_load the mean load thermistor. One spectrum
    for every integration; the check records how far it would move if each
    receiver sweep (with its nearest ambient-load sweep) were used."""
    from eigsep_cal.noise_source import tns_as_received

    t_ex = ns.on(freqs, smooth=True)
    idx = np.searchsorted(ns.freqs, freqs)
    g_ns = ns.gamma_ns[np.clip(idx, 0, ns.freqs.size - 1)]
    (tr, gr, _), (ta, ga, _) = s11["rec"], s11["amb"]
    t_load = float(np.mean(therm_k))
    t_ns = tns_as_received(
        t_ex, g_ns, np.mean(gr, axis=0), np.mean(ga, axis=0), t_load
    )
    near = np.array([np.argmin(np.abs(ta - t)) for t in tr])
    per_sweep = tns_as_received(t_ex, g_ns, gr, ga[near], t_load)
    b = (freqs >= check_band_mhz[0]) & (freqs <= check_band_mhz[1])
    dev = (per_sweep[:, b] - t_ns[b]) / t_ns[b]
    check = {
        "band_mhz": list(check_band_mhz),
        "receiver_sweeps": int(tr.size),
        "max_abs_frac_change_if_per_sweep": float(np.nanmax(np.abs(dev))),
        "rms_frac_change_if_per_sweep": float(np.sqrt(np.nanmean(dev**2))),
        "t_load_k": t_load,
        "t_ns_k_range_in_band": [
            float(np.nanmin(t_ns[b])),
            float(np.nanmax(t_ns[b])),
        ],
    }
    return t_ns.astype(float), check


def main(argv=None):
    import eigsep_cal.s11

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version", help="product version, e.g. v0004")
    ap.add_argument(
        "--products",
        help="campaign tree to read the config from and write derived/tcal/ "
        "in (default: the campaign root, which also supplies the raw data)",
    )
    ap.add_argument(
        "--config", help="build config (default: <version dir>/config.json)"
    )
    args = ap.parse_args(argv)

    root = get_campaign_root(required=True)
    proot = Path(args.products) if args.products else root
    out_dir = product_dir(proot, "tcal", args.version)
    cfg, cfg_path = load_config(proot, "tcal", args.version, args.config)
    out_dir.mkdir(parents=True, exist_ok=True)
    switch = root / cfg["switch_sparams"]
    osl = eigsep_cal.s11.PACKAGED_FILES[cfg["osl_year"]]["osldata"]
    windows = pd.DataFrame(
        [json.loads(x) for x in open(root / cfg["cal_windows"])]
    )
    limits = cfg["limits"]

    index = MetadataIndex(root / "data")
    amb, freqs, amb_log = visits(index, "RFAMB", windows, cfg)
    non, freqs_n, non_log = visits(index, "RFNON", windows, cfg)
    assert np.array_equal(freqs, freqs_n)
    s11, s11_log = s11_on_channels(
        root / "data" / "s11", switch, osl, freqs, cfg
    )
    therm_t, therm_k, therm_log = thermistor(index, cfg)

    tns_cfg = cfg["t_ns"]
    ns_input = None
    if "noise_source" in tns_cfg:
        from ..products import noise_source

        ver = tns_cfg["noise_source"]
        ns_root = next(
            r
            for r in (proot, root)
            if (
                product_dir(r, "noise_source", ver) / "solutions.npz"
            ).is_file()
        )
        ns = noise_source.load(ver, root=ns_root)
        t_ns, check = noise_source_tns(
            ns, freqs, s11, therm_k, tns_cfg["check_band_mhz"]
        )
        t_ns_model = {
            "t_ns_k": "spectrum on freqs (solutions.npz)",
            "t_ns_source": f"derived/noise_source/{ver}: bench-measured excess at P "
            "(smooth fit), converted to the field receiver with "
            "eigsep_cal.noise_source.tns_as_received",
            "t_ns_formula": "M_NS (T_load + T_ex) - M_amb T_load, campaign-mean field Gamma_rec "
            "and Gamma_amb at P, lab Gamma_NS at P, mean load thermistor",
            "t_ns_check": check,
        }
        ns_input = {
            "version": ver,
            "sha256": sha256(
                product_dir(ns_root, "noise_source", ver) / "solutions.npz"
            ),
            "provenance": ns.manifest.get("provenance", {}).get("code"),
        }
    else:
        t_ns = float(tns_cfg["assumed_k"])
        t_ns_model = {"t_ns_k": t_ns, "t_ns_source": tns_cfg["source"]}

    sols = dict(
        freqs=freqs,
        keys=np.array([cfg["key"]]),
        t_ns_k=t_ns,
        **limits,
        amb_t=amb.t.to_numpy(float),
        amb_p=np.stack(amb.p.to_list()),
        amb_regime=amb.regime.to_numpy(str),
        non_t=non.t.to_numpy(float),
        non_p=np.stack(non.p.to_list()),
        non_regime=non.regime.to_numpy(str),
        therm_t=therm_t,
        therm_k=therm_k,
    )
    for dut, (t, g, epoch) in s11.items():
        sols[f"s11_{dut}_t"] = t
        sols[f"s11_{dut}"] = g.astype(np.complex64)
        sols[f"s11_{dut}_epoch"] = epoch
    np.savez_compressed(out_dir / "solutions.npz", **sols)

    manifest = {
        "provenance": {
            "product": "tcal",
            "version": args.version,
            "built_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "builder": "eigsep_data.calibration.build_tcal (eigsep-build-tcal)",
            "argv": sys.argv[1:],
            "config": {
                "path": (
                    str(Path(cfg_path).resolve().relative_to(proot.resolve()))
                    if Path(cfg_path).resolve().is_relative_to(proot.resolve())
                    else str(cfg_path)
                ),
                "sha256": sha256(cfg_path),
            },
            "code": code_provenance(),
        },
        "applies_to": {
            "keys": [cfg["key"]],
            "antenna": cfg["antenna"],
            "plane": "P",
        },
        "model": {
            "equations": "eigsep_cal.dicke.tstar_coefficients, receiver_s11_coefficients",
            **t_ns_model,
            "visit_average": "integration-time-weighted mean; per-sample mask: flag-product RFI bits, wrapped samples, dropped integrations",
            "flags": cfg["flags"],
            "rfi_mask": cfg["rfi_mask"],
            "visit_rejection": cfg["visit_rejection"],
            "omits": [
                "receiver noise waves",
                *(
                    [
                        "noise-source port mismatch (VNANON/RFNON paths uncharacterized)"
                    ]
                    if ns_input is None
                    else []
                ),
                "loss in balun, coax and RFANT switch path (temperatures stay at P)",
            ],
        },
        "limits": limits,
        "inputs": {
            "switch_sparams": {
                "path": cfg["switch_sparams"],
                "sha256": sha256(switch),
            },
            "internal_osl": {
                "path": "eigsep_cal package data/" + Path(osl).name,
                "sha256": sha256(osl),
            },
            "s11_manifest_sha256": sha256(
                root / "data" / "s11" / "manifest.csv"
            ),
            "cal_windows_sha256": sha256(root / cfg["cal_windows"]),
            **({"noise_source": ns_input} if ns_input else {}),
            "span_utc": cfg["span_utc"],
            "regimes": cfg["regimes"],
            "s11_epoch_split_utc": cfg["s11_epoch_split_utc"],
        },
        "contents": {
            "amb_visits": amb_log,
            "non_visits": non_log,
            "s11": s11_log,
            "thermistor": therm_log,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["contents"], indent=1)[:3000])
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
