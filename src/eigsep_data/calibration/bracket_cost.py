"""What wider tcal limits would cost, in temperature.

Measured on one receiver regime of ``derived/tcal/<version>/`` (the
config's ``validation.bracket_regime``), where the calibrator cycles every
~13 min and S11 is swept hourly:

1. **Calibration brackets.** For every pair of cycles m apart, each cycle
   between them is predicted by linear interpolation and compared with
   what was measured. The errors in P_load and in the gain proxy
   G = P_NS - P_load become an error in T* through
   dT* = -T_NS dP_load / G - (T* - T_load) dG / G, on the median T*
   spectrum of the reference night.
2. **S11 holds.** Each sweep is used in place of the one dt later, and the
   calibrated T (antenna + load + receiver correction) is compared.
3. **The hole.** The longest gap between the regime's cycles; on
   marjum-2026-07 the antenna was raised inside it (the config's
   ``validation.raise_utc``). Normal drift over a span cannot see a step,
   so the two edges of the hole are compared directly with the
   distribution of changes over the same span elsewhere.

Writes ``bracket_cost.json`` beside the product. Run::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 eigsep-tcal-bracket-cost v0004
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..paths import get_campaign_root
from ._common import load_config, product_dir, unix
from .validate_tcal import reference_night


def median_night_tstar(root, vcfg, key):
    """Median T* over the reference night's sky visits (Christian's chain)."""
    from . import d5_yfactor as yf

    np.seterr(divide="ignore", invalid="ignore")
    d = yf.load_corr(root / "data", *reference_night(vcfg), key)
    v = yf.average_blocks(d, use=~np.all(d["spec"] == 0, axis=1))
    sky, non, amb = (v[s] for s in (yf.SKY, yf.NOISE, yf.LOAD))
    il, _ = yf.nearest(sky["t"], amb["t"])
    inn, _ = yf.nearest(sky["t"], non["t"])
    tl = amb["t_load"][il][:, None]
    ts = yf.tant_star(sky["spec"], non["spec"][inn], amb["spec"][il], tl)
    return np.median(ts, axis=0), float(np.nanmedian(tl))


def cycles(z, regime, max_pair_s=300.0):
    """Pair each noise-source visit of ``regime`` with its load visit.

    A visit pairs with the nearest load visit when they are under
    ``max_pair_s`` apart: in one cycle, not the next (~13 min later)."""
    keep_n = z["non_regime"] == regime
    tn, pn = z["non_t"][keep_n], z["non_p"][keep_n]
    ka = z["amb_regime"] == regime
    ta, pa = z["amb_t"][ka], z["amb_p"][ka]
    j = np.array([np.argmin(np.abs(ta - t)) for t in tn])
    ok = np.abs(ta[j] - tn) < max_pair_s
    return tn[ok], pa[j][ok], pn[ok] - pa[j][ok]


def stats(x):
    x = np.asarray(x)
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def bracket_errors(
    t, pl, g, t_ns, tstar, t_load, b, brackets=(2, 3, 4, 5, 7, 9, 11, 14)
):
    """Fractional T* error of interpolating calibration across ``m`` cycles.

    ``t``, ``pl``, ``g`` are cycle times, load powers and gain proxies of a
    stretch with no gap; ``brackets`` the spans ``m`` to score, in cycles.
    Each row reports the span in minutes at the stretch's median cadence."""
    cadence = float(np.median(np.diff(t)))
    rows = []
    for m in brackets:
        errs = []
        for i in range(t.size - m):
            for k in range(i + 1, i + m):
                w = (t[k] - t[i]) / (t[i + m] - t[i])
                dpl = (1 - w) * pl[i] + w * pl[i + m] - pl[k]
                dg = (1 - w) * g[i] + w * g[i + m] - g[k]
                dts = -t_ns * dpl / g[k] - (tstar - t_load) * dg / g[k]
                errs.append(np.nanmedian(np.abs(dts / tstar)[b]))
        rows.append({"bracket_min": round(m * cadence / 60, 1), **stats(errs)})
    return rows


def s11_hold_errors(z, before, tstar, t_load, b, holds=(1, 2, 3, 4)):
    """Fractional error in calibrated T of holding an S11 sweep ``m``
    sweeps too long, over the sweeps up to ``before``; ``holds`` are the
    ``m`` to score. Each row reports the hold in hours (median)."""
    from eigsep_cal.dicke import receiver_s11_coefficients

    st = z["s11_ant_t"]
    pre = st <= before
    g_ant, g_amb, g_rec = (z[f"s11_{d}"][pre] for d in ("ant", "amb", "rec"))
    st = st[pre]

    def t_cal(i):
        a, o = 1.0, 0.0  # T* is given, so the T* line is the identity
        sc, of = receiver_s11_coefficients(
            a, o, t_load, g_ant[i], g_amb[i], g_rec[i]
        )
        return np.real(sc * tstar + of)

    rows = []
    for m in holds:
        errs, dts = [], []
        for i in range(st.size - m):
            ref = t_cal(i + m)
            errs.append(np.nanmedian(np.abs(t_cal(i) / ref - 1)[b]))
            dts.append(st[i + m] - st[i])
        rows.append(
            {"hold_h": round(float(np.median(dts)) / 3600, 2), **stats(errs)}
        )
    return rows


def across_hole(
    z, tc, gc, ih, b, raise_t, cycle_tol_s=600.0, s11_tol_s=1800.0
):
    """Changes across the hole against the same span elsewhere.

    ``tc``, ``gc`` are the regime's cycle times and gain proxies, the hole
    lies between cycles ``ih`` and ``ih + 1``, ``b`` is the band's channel
    mask and ``raise_t`` (unix s) a time inside the hole. A pair elsewhere
    counts as spanning the same time when its span is within
    ``cycle_tol_s`` (cycles, ~13 min apart) or ``s11_tol_s`` (S11 sweeps,
    hourly) of the hole's, before the hole."""
    before, after = tc[ih], tc[ih + 1]
    ib, ia = ih, ih + 1
    span = after - before
    edge = float(np.nanmedian(np.abs(gc[ia] / gc[ib] - 1)[b]))
    ref = []
    for i in np.flatnonzero(tc <= before):
        j = np.argmin(np.abs(tc - (tc[i] + span)))
        if abs(tc[j] - tc[i] - span) < cycle_tol_s and tc[j] <= before:
            ref.append(np.nanmedian(np.abs(gc[j] / gc[i] - 1)[b]))
    s_all = z["s11_ant_t"]
    sb = np.flatnonzero(s_all < raise_t).max()
    sa = np.flatnonzero(s_all > raise_t).min()
    s_span = s_all[sa] - s_all[sb]
    dgam = float(np.nanmedian(np.abs(z["s11_ant"][sa] - z["s11_ant"][sb])[b]))
    sref = []
    for i in range(s_all.size):
        j = np.argmin(np.abs(s_all - (s_all[i] + s_span)))
        if (
            abs(s_all[j] - s_all[i] - s_span) < s11_tol_s
            and s_all[j] < s_all[sb] + 1  # sweeps up to sb (1 s slack)
        ):
            sref.append(
                np.nanmedian(np.abs(z["s11_ant"][j] - z["s11_ant"][i])[b])
            )
    return {
        "cycle_edges_utc": [
            datetime.utcfromtimestamp(x).strftime("%H:%M")
            for x in (before, after)
        ],
        "gain_change_across_hole": edge,
        "gain_change_same_span_elsewhere": stats(ref) if ref else None,
        "s11_edges_utc": [
            datetime.utcfromtimestamp(x).strftime("%H:%M")
            for x in (s_all[sb], s_all[sa])
        ],
        "s11_ant_abs_change_across_hole": dgam,
        "s11_ant_abs_change_same_span_elsewhere": (
            stats(sref) if sref else None
        ),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version")
    ap.add_argument(
        "--products",
        help="campaign tree to read/write derived/tcal/ in (default: the "
        "campaign root, which also supplies the raw data)",
    )
    ap.add_argument(
        "--config", help="build config (default: <version dir>/config.json)"
    )
    args = ap.parse_args(argv)
    root = get_campaign_root(required=True)
    proot = Path(args.products) if args.products else root
    pdir = product_dir(proot, "tcal", args.version)
    cfg, _ = load_config(proot, "tcal", args.version, args.config)
    vcfg = cfg["validation"]
    band = tuple(vcfg["band_mhz"])
    z = dict(np.load(pdir / "solutions.npz"))
    f = z["freqs"]
    b = (f >= band[0]) & (f < band[1])
    # One number or one value per channel; either broadcasts against the
    # per-channel spectra below.
    t_ns = np.asarray(z["t_ns_k"], dtype=float)
    tstar, t_load = median_night_tstar(root, vcfg, cfg["key"])
    # The hole is the longest gap between the regime's cycles.
    tc, plc, gc = cycles(z, vcfg["bracket_regime"])
    ih = int(np.argmax(np.diff(tc)))
    before = tc[ih]
    night = " to ".join(vcfg["reference_night_local"])
    out = {
        "band_mhz": band,
        "t_star_reference": f"median over the sky visits of {night} "
        f"({vcfg['reference_night_tz']})",
    }

    # 1. Calibration brackets, quiet stretch only (no bracket spans the hole).
    out["cal_bracket_frac_error_in_tstar"] = bracket_errors(
        tc[tc <= before],
        plc[tc <= before],
        gc[tc <= before],
        t_ns,
        tstar,
        t_load,
        b,
    )
    # 2. S11 hold, the regime's sweeps before the hole.
    out["s11_hold_frac_error_in_t"] = s11_hold_errors(
        z, before, tstar, t_load, b
    )

    # 3. Across the hole: edges vs the same span elsewhere.
    out["hole"] = across_hole(z, tc, gc, ih, b, unix(vcfg["raise_utc"]))
    (pdir / "bracket_cost.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
