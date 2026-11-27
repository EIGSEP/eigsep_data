"""Tests for the noise-source options added for noise_source v0002 and
tcal v0005: the loads' switch paths, the cycle selection, mean S11, the
DPSS smooth fit and the field conversion's off-path temperature."""

import json

import numpy as np
import pandas as pd
import pytest

from eigsep_data import noise_source_bench as nsb
from eigsep_data import smooth_basis as sb
from eigsep_data.calibration import build_tcal as bt
from eigsep_data.products import noise_source as ns


def lossy_path(nf, loss_db=0.5, s11=0.05 + 0.02j, s22=-0.03 + 0.01j):
    g = 10 ** (-loss_db / 10)
    return (
        np.full(nf, s11),
        np.full(nf, g * np.exp(0.3j)),
        np.full(nf, s22),
    )


def test_a_matched_lossless_path_passes_the_load_unchanged():
    nf = 5
    path = (np.zeros(nf), np.ones(nf), np.zeros(nf))
    gain = nsb.path_gain(np.full(nf, 0.01 + 0.0j), path)
    np.testing.assert_allclose(gain, 1.0)
    t = nsb.loads_at_p(np.array([300.0, 340.0]), gain, np.array([305.0] * 2))
    np.testing.assert_allclose(t, [[300.0] * nf, [340.0] * nf])


def test_path_gain_is_the_available_gain():
    nf = 4
    s11, s12s21, s22 = lossy_path(nf)
    gt = np.full(nf, 0.004 - 0.003j)
    gs = s11 + s12s21 * gt / (1 - s22 * gt)
    expect = (
        np.abs(s12s21)
        * (1 - np.abs(gt) ** 2)
        / (np.abs(1 - s22 * gt) ** 2 * (1 - np.abs(gs) ** 2))
    )
    gain = nsb.path_gain(gt, (s11, s12s21, s22))
    np.testing.assert_allclose(gain, expect)
    assert np.all((gain > 0.85) & (gain < 0.95))


def test_a_lossy_path_pulls_the_load_toward_the_path():
    gain = np.array([0.9, 0.8])
    t = nsb.loads_at_p(np.array([350.0]), gain, np.array([300.0]))
    np.testing.assert_allclose(t, [[345.0, 340.0]])


def test_fitted_cycles_keeps_steady_cycles_below_the_limit():
    kind = np.array(["steady", "heating", "steady", "steady"])
    t_hot = np.array([320.0, 330.0, 345.0, 342.0])
    assert nsb.fitted_cycles(kind, t_hot).tolist() == [1, 0, 1, 1]
    use = nsb.fitted_cycles(kind, t_hot, hot_below_k=343.15)
    assert use.tolist() == [1, 0, 0, 1]


def test_s11_average_is_the_sweep_mean_at_every_time():
    fv = np.array([50.0, 100.0, 150.0])
    g = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]) + 0j
    ts = np.array([0.0, 100.0])
    t = np.array([-10.0, 50.0, 500.0])
    f = np.array([75.0, 125.0])
    mean = nsb._on_cycles(ts, g, fv, f, t, average=True)
    np.testing.assert_allclose(mean, [[0.25, 0.35]] * 3)
    interp = nsb._on_cycles(ts, g, fv, f, t)
    # held before the first sweep and after the last, linear between
    np.testing.assert_allclose(interp[:, 0], [0.15, 0.25, 0.35])


def test_dpss_basis_size_and_polynomial_columns():
    f = np.arange(50.0, 245.0, 0.244140625)
    d = sb.dpss_design(f, 50.78, 244.0, 10.0, poly_deg=2)
    nw = 10e-9 * (
        np.arange(50.78, 244.0 + 0.122, 0.244140625).size * 0.244140625e6
    )
    assert d.shape == (f.size, 3 + int(np.ceil(2 * nw)) + 1)
    np.testing.assert_allclose(d[:, 0], 1.0)
    np.testing.assert_allclose(d[:, 1], np.log(f / 120.0))


def smooth_spectrum(f):
    return 600.0 * (f / 120.0) ** -0.4 + 5.0 * np.sin(f / 30.0)


def test_dpss_smooth_recovers_a_smooth_spectrum_and_masks():
    f = np.arange(40.0, 250.0, 0.25)
    good = (f >= 50.0) & (f <= 245.0)
    good[(f > 120.0) & (f < 121.0)] = False
    rng = np.random.default_rng(1)
    sigma = np.full(f.size, 1.0)
    y = np.where(good, smooth_spectrum(f) + rng.normal(0, 1, f.size), np.nan)
    smooth = {
        "model": "dpss",
        "half_width_ns": 10,
        "poly_deg": 2,
        "f0_mhz": 120.0,
        "exclude_mhz": [[193.0, 222.0]],
        "cv_block_mhz": 12.0,
        "cv_offsets": 2,
    }
    out, coef, f_lo, f_hi, info = ns.smooth_dpss(f, y, sigma, good, smooth)
    assert (f_lo, f_hi) == (50.0, 245.0)
    assert np.isnan(out[f < 50.0]).all() and np.isnan(out[f > 245.0]).all()
    assert np.isnan(out[(f >= 193.0) & (f <= 222.0)]).all()
    inside = np.isfinite(out)
    assert inside[(f > 120.0) & (f < 121.0)].all()  # interpolated
    resid = (out - smooth_spectrum(f))[inside]
    assert np.sqrt(np.mean(resid**2)) < 0.5
    assert info["ncol"] == coef.size
    held = info["held_out"]
    assert held["this_model"]["chi2"] < 1.5
    assert held["this_model"]["rms"] < held["cubic_in_f (v0001 form)"]["rms"]


def test_write_with_dpss_round_trips_and_says_so(tmp_path):
    f = np.arange(40.0, 250.0, 0.25)
    good = (f >= 50.0) & (f <= 245.0)
    sigma = np.full(f.size, 2.0)
    t_non = np.where(good, 300.0 + smooth_spectrum(f), np.nan)
    smooth = {
        "model": "dpss",
        "half_width_ns": 10,
        "poly_deg": 2,
        "exclude_mhz": [[193.0, 222.0]],
    }
    d = ns.write(
        tmp_path,
        "v0002",
        freqs=f,
        t_non_k=t_non,
        t_non_sigma_k=sigma,
        t_off_k=300.0,
        good=good,
        gamma_ns=np.full(f.size, 0.05 + 0j),
        provenance={},
        smooth=smooth,
    )
    n = ns.load("v0002", root=tmp_path)
    assert n.smooth_model == "dpss"
    assert np.isnan(n.on(f[(f > 200.0) & (f < 201.0)])).all()
    np.testing.assert_allclose(
        n.on(f[(f > 100.0) & (f < 101.0)]),
        smooth_spectrum(f[(f > 100.0) & (f < 101.0)]),
        atol=0.01,
    )
    m = json.loads((d / "manifest.json").read_text())
    assert m["smooth_fit"]["model"] == "dpss"
    assert m["smooth_fit"]["f_lo_mhz"] == 50.0
    with pytest.raises(ValueError, match="smooth model"):
        ns.write(
            tmp_path,
            "v0003",
            freqs=f,
            t_non_k=t_non,
            t_non_sigma_k=sigma,
            t_off_k=300.0,
            good=good,
            gamma_ns=np.zeros(f.size),
            provenance={},
            smooth={"model": "spline"},
        )


def test_a_v0001_product_reads_as_a_polynomial(tmp_path):
    f = np.linspace(50.0, 245.0, 40)
    ns.write(
        tmp_path,
        "v0001",
        freqs=f,
        t_non_k=800.0 + 0 * f,
        t_non_sigma_k=np.ones(f.size),
        t_off_k=300.0,
        good=np.ones(f.size, bool),
        gamma_ns=np.zeros(f.size),
        provenance={},
    )
    assert ns.load("v0001", root=tmp_path).smooth_model == "poly"


class FakeNS:
    def __init__(self, f, excess, gamma):
        self.freqs, self._ex, self.gamma_ns = f, excess, gamma

    def on(self, freqs, smooth=True):
        return self._ex


def field_inputs(nf=6):
    f = np.linspace(60.0, 190.0, nf)
    g_rec = np.full(nf, 0.2 + 0.1j)
    s11 = {
        "rec": (np.array([0.0, 10.0]), np.array([g_rec, g_rec * 1.01]), 0),
        "amb": (np.array([1.0, 11.0]), np.array([np.full(nf, 0.01j)] * 2), 0),
    }
    therm = np.array([301.0, 303.0])
    return (
        f,
        FakeNS(f, np.full(nf, 500.0), np.full(nf, 0.06 + 0.0j)),
        s11,
        therm,
    )


def test_field_conversion_defaults_are_tcal_v0004s():
    from eigsep_cal.noise_source import tns_as_received

    f, n, s11, therm = field_inputs()
    t_ns, check = bt.noise_source_tns(n, f, s11, therm, (50.0, 200.0))
    expect = tns_as_received(
        500.0,
        n.gamma_ns,
        s11["rec"][1].mean(0),
        s11["amb"][1].mean(0),
        302.0,
    )
    np.testing.assert_array_equal(t_ns, expect)
    assert "t_off_k" not in check


def test_field_conversion_with_the_board_and_the_load_path():
    from eigsep_cal.dicke import mismatch

    f, n, s11, therm = field_inputs()
    base, _ = bt.noise_source_tns(n, f, s11, therm, (50.0, 200.0))
    dt = (np.arange(4) * 1800.0, np.array([2.0, 3.0, 4.0, 3.0]))
    t_ns, check = bt.noise_source_tns(
        n, f, s11, therm, (50.0, 200.0), dt_board_k=dt
    )
    m_ns = np.real(mismatch(n.gamma_ns, s11["rec"][1].mean(0)))
    np.testing.assert_allclose(t_ns - base, 3.0 * m_ns)
    assert check["t_off_k"] == pytest.approx(305.0)
    assert check["board_minus_load_k"]["mean"] == pytest.approx(3.0)
    frac = check["frac_change_if_per_reading"]
    assert frac["max_abs"] == pytest.approx(
        frac["band_mean_dtns_dtoff_per_k"] * 1.0
    )
    # a lossless load path changes nothing; a lossy one lowers the load
    # term's temperature toward T_off, so it raises T_NS slightly less
    same, _ = bt.noise_source_tns(
        n,
        f,
        s11,
        therm,
        (50.0, 200.0),
        dt_board_k=dt,
        load_gain=np.ones(f.size),
    )
    np.testing.assert_allclose(same, t_ns)
    lossy, c2 = bt.noise_source_tns(
        n,
        f,
        s11,
        therm,
        (50.0, 200.0),
        dt_board_k=dt,
        load_gain=np.full(f.size, 0.9),
    )
    m_amb = np.real(mismatch(s11["amb"][1].mean(0), s11["rec"][1].mean(0)))
    np.testing.assert_allclose(t_ns - lossy, m_amb * 0.1 * 3.0)
    assert c2["load_path_gain_in_band"] == [0.9, 0.9]


class FakeIndex:
    def __init__(self, meta):
        self.meta = meta

    def select(self, time):
        t0, t1 = (pd.Timestamp(x).timestamp() for x in time)
        m = self.meta
        return FakeIndex(m[(m.time_best >= t0) & (m.time_best <= t1)])


def test_board_minus_load_skips_bad_readings_and_duplicates():
    t0 = pd.Timestamp("2026-07-17T05:00:00Z").timestamp()
    meta = pd.DataFrame(
        {
            "time_best": t0 + np.array([0.0, 0.0, 10.0, 20.0, 30.0, 9e5]),
            "rfswitch_therm_temp_therm2": [
                32.0,
                32.0,
                33.0,
                -40.0,
                31.0,
                30.0,
            ],
            "tempctrl_load_T_now": [29.0, 29.0, 29.5, 29.0, "MISSING", 29.0],
        }
    )
    cfg = {
        "span_utc": ["2026-07-17T00:00:00Z", "2026-07-18T04:00:00Z"],
        "load_thermistor_c": "tempctrl_load_T_now",
    }
    t, d = bt.board_minus_load(
        FakeIndex(meta), cfg, "rfswitch_therm_temp_therm2"
    )
    np.testing.assert_allclose(t - t0, [0.0, 10.0])
    np.testing.assert_allclose(d, [3.0, 3.5])


def test_tns_terms_reproduce_the_field_conversion():
    f, n, s11, therm = field_inputs()
    dt = (np.arange(4) * 1800.0, np.array([2.0, 3.0, 4.0, 3.0]))
    gain = np.linspace(0.85, 0.95, f.size)
    t_ns, check = bt.noise_source_tns(
        n, f, s11, therm, (50.0, 200.0), dt_board_k=dt, load_gain=gain
    )
    c0, c_off, c_load = bt.tns_terms(n, f, s11, gain)
    again = c0 + c_off * check["t_off_k"] + c_load * check["t_load_k"]
    np.testing.assert_allclose(again, t_ns, rtol=1e-12)
    # without a load path G = 1: T_load,P = T_load
    c0, c_off, c_load = bt.tns_terms(n, f, s11)
    t_ns1, check1 = bt.noise_source_tns(
        n, f, s11, therm, (50.0, 200.0), dt_board_k=dt
    )
    np.testing.assert_allclose(
        c0 + c_off * check1["t_off_k"] + c_load * check1["t_load_k"],
        t_ns1, rtol=1e-12,
    )


def test_visit_board_minus_load_is_the_dwell_mean():
    t = np.arange(0.0, 300.0, 1.0)
    d = np.where((t % 100) < 50, 4.0, 6.0) + 0.001 * t
    mean, n = bt.visit_board_minus_load((t, d), np.array([25.0, 75.0,
                                                          1e4]), 10.0)
    np.testing.assert_array_equal(n, [21, 21, 0])
    np.testing.assert_allclose(mean[:2], [4.025, 6.075])
    assert np.isnan(mean[2])


def test_realistic_sigma_is_the_running_rms_floored_by_the_formal():
    rng = np.random.default_rng(3)
    nf = 400
    fitted = np.ones(nf, bool)
    fitted[100:110] = False
    resid = np.where(np.arange(nf) < 200, 2.0, 0.5) * rng.standard_normal(nf)
    sigma = np.full(nf, 1.0)
    out = ns.realistic_sigma(resid, sigma, fitted, 49)
    assert np.isnan(out[100:110]).all()
    assert np.nanmean(out[30:90]) == pytest.approx(2.0, rel=0.15)
    assert np.all(out[fitted] >= 1.0)
    assert np.all(out[260:370] == 1.0)


def test_realistic_sigma_keeps_the_scatter_beside_an_edge():
    # A masked channel within half a window of the fitted range's edge
    # leaves the edge windows less than half fitted.
    rng = np.random.default_rng(5)
    nf = 400
    fitted = np.zeros(nf, bool)
    fitted[50:350] = True
    fitted[60] = fitted[340] = False
    resid = 2.0 * rng.standard_normal(nf)
    out = ns.realistic_sigma(resid, np.full(nf, 0.5), fitted, 49)
    assert np.isfinite(out[fitted]).all() and np.isnan(out[~fitted]).all()
    assert out[fitted].min() > 1.4
    # the edge channels take the nearest window with half its channels
    assert out[50] == out[51] and out[349] == out[348]


def test_realistic_sigma_without_a_half_fitted_window_is_nan():
    fitted = np.zeros(100, bool)
    fitted[40:50] = True
    out = ns.realistic_sigma(np.ones(100), np.ones(100), fitted, 49)
    assert np.isnan(out).all()


@pytest.mark.parametrize("window", [48, 0, 49.0, None, 401])
def test_realistic_sigma_refuses_a_bad_window(window):
    with pytest.raises(ValueError, match="window_channels"):
        ns.realistic_sigma(np.ones(400), np.ones(400), np.ones(400, bool),
                           window)


def test_write_with_realistic_adds_the_error(tmp_path):
    f = np.arange(40.0, 250.0, 0.25)
    good = (f >= 50.0) & (f <= 245.0)
    rng = np.random.default_rng(4)
    sigma = np.full(f.size, 0.5)
    t_non = np.where(
        good, 300.0 + smooth_spectrum(f) + rng.normal(0, 2.0, f.size), np.nan
    )
    smooth = {"model": "dpss", "half_width_ns": 10, "poly_deg": 2,
              "f0_mhz": 120.0, "exclude_mhz": [[193.0, 222.0]],
              "cv_block_mhz": 12.0, "cv_offsets": 2}
    kw = dict(freqs=f, t_non_k=t_non, t_non_sigma_k=sigma, t_off_k=300.0,
              good=good, gamma_ns=np.zeros(f.size, complex),
              provenance={}, smooth=smooth)
    ns.write(tmp_path, "v0002", **kw)
    ns.write(tmp_path, "v0003", realistic={"window_channels": 49}, **kw)
    old, new = ns.load("v0002", tmp_path), ns.load("v0003", tmp_path)
    assert old.excess_sigma_k is None
    np.testing.assert_array_equal(old.excess_smooth_k, new.excess_smooth_k)
    fitted = good & np.isfinite(new.excess_smooth_k)
    assert np.isfinite(new.excess_sigma_k[fitted]).all()
    assert np.isnan(new.excess_sigma_k[~fitted]).all()
    assert np.nanmean(new.excess_sigma_k) == pytest.approx(2.0, rel=0.1)
    assert "excess_sigma_k" in new.manifest["contents"]


def test_comb_hit_drops_only_visits_above_the_fraction():
    v = pd.DataFrame({"t": [1.78e9, 1.78e9 + 600, 1.78e9 + 1200],
                      "comb": [0.0, 0.2, 0.01], "n": [100, 100, 100]})
    dropped, kept = bt.comb_hit(v, 0.05)
    assert [d["comb_fraction"] for d in dropped] == [0.2]
    np.testing.assert_array_equal(kept.comb, [0.0, 0.01])


def test_on_full_axis_leaves_untested_channels_unflagged():
    words = np.array([[0, 4096, 4, 8192], [1, 0, 12288, 0]], np.uint16)
    out = bt.on_full_axis(words, slice(2, 6), 8, 12288)
    assert out.shape == (2, 8) and not out[:, :2].any()
    assert not out[:, 6:].any()
    np.testing.assert_array_equal(
        out[:, 2:6], [[0, 1, 0, 1], [0, 0, 1, 0]]
    )


def test_check_config_refuses_options_that_do_not_go_together():
    ok = {"visit_rejection": {"comb_mask": 12288, "comb_max_fraction": 0.05},
          "t_ns": {"noise_source": "v0003", "t_off": {},
                   "per_integration": {"dwell_s": 30.0}}}
    bt.check_config(ok)
    bt.check_config({"visit_rejection": {}, "t_ns": {"assumed_k": 917.0}})

    def changed(section, **kw):
        cfg = {k: dict(v) for k, v in ok.items()}
        for k, v in kw.items():
            if v is None:
                del cfg[section][k]
            else:
                cfg[section][k] = v
        return cfg

    for cfg, match in (
        (changed("visit_rejection", comb_max_fraction=None), "comb_mask"),
        (changed("visit_rejection", comb_mask=None), "comb_mask"),
        (changed("t_ns", t_off=None), "t_ns.t_off"),
        (changed("t_ns", noise_source=None), "t_ns.noise_source"),
        (changed("t_ns", per_integration={}), "dwell_s"),
    ):
        with pytest.raises(SystemExit, match=match):
            bt.check_config(cfg)
