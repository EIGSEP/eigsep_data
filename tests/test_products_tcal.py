"""Tests for the tcal product: field calibration built per row.

Every assertion is about what a row is allowed to be calibrated with:
the bracketing load and noise-source visits of its own regime, the
nearest thermistor reading, the S11 of its own epoch, and nothing
extrapolated. The arithmetic itself is eigsep_cal's and tested there.
"""

import json

import numpy as np
import pytest

from eigsep_data import MetadataIndex

from conftest import NCHAN, write_corr_file

pytest.importorskip("eigsep_cal.dicke")
from eigsep_cal.dicke import (  # noqa: E402
    receiver_s11_coefficients,
    tstar_coefficients,
)

ANTS = {"0": "box-gnd", "4": "box-air"}
T_NS = 917.0
STATES = ["RFANT"] * 4 + ["RFAMB", "RFNON"]


@pytest.fixture
def campaign(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for i in range(2):
        write_corr_file(
            data / f"corr_20260717_15{i}041Z.h5",
            ntimes=6,
            keys=("0", "4"),
            rfswitch=STATES,
            sync_time=1.7843e9 + 600 * i,
            input_to_ant=ANTS,
            seed=i,
        )
    return tmp_path


def _times(root):
    return MetadataIndex(root / "data", cache=False).select().meta.time_best.to_numpy(float)


def write_solutions(root, version="v0000", **override):
    """A solution set bracketing every row, with flat spectra so the
    expected coefficients can be written down by hand."""
    t = _times(root)
    t0, t1 = t.min() - 10.0, t.max() + 10.0
    freqs = np.linspace(0, 250, NCHAN, endpoint=False)
    flat = np.ones(NCHAN)
    sols = dict(
        freqs=freqs,
        keys=np.array(["4"]),
        t_ns_k=T_NS,
        max_cal_gap_s=3600.0,
        max_s11_gap_s=7200.0,
        s11_hold_s=1800.0,
        therm_tol_s=60.0,
        amb_t=np.array([t0, t1]),
        amb_p=np.stack([10.0 * flat, 20.0 * flat]),
        amb_regime=np.array(["rx-A", "rx-A"]),
        non_t=np.array([t0, t1]),
        non_p=np.stack([110.0 * flat, 120.0 * flat]),
        non_regime=np.array(["rx-A", "rx-A"]),
        therm_t=np.array([t0, t1]),
        therm_k=np.array([300.0, 302.0]),
    )
    for dut, g in (("ant", 0.5), ("amb", 0.05), ("rec", 0.3j)):
        sols[f"s11_{dut}_t"] = np.array([t0, t1])
        sols[f"s11_{dut}"] = np.stack([g * flat, g * flat]).astype(complex)
        sols[f"s11_{dut}_epoch"] = np.array(["pre-gap", "pre-gap"])
    sols.update(override)
    d = root / "derived" / "tcal" / version
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "solutions.npz", **sols)
    (d / "manifest.json").write_text(
        json.dumps({"provenance": {"product": "tcal", "version": version}})
    )
    return sols


def bundle(root, antenna="box-air"):
    from eigsep_data import products as P

    P.get("tcal")._cache.clear()
    index = MetadataIndex(root / "data", cache=False)
    return index.select().load_bundle(
        antenna=antenna, root=root, products=["tcal@v0000"]
    )


class TestInterpolation:
    def test_load_and_noise_source_are_linear_in_time(self, campaign):
        s = write_solutions(campaign)
        b = bundle(campaign)
        t0, t1 = s["amb_t"]
        w = (b.t - t0) / (t1 - t0)
        p_amb, p_non = 10 + 10 * w, 110 + 10 * w
        a = T_NS / (p_non - p_amb)
        np.testing.assert_allclose(b.products["tcal"]["tstar_scale"], np.repeat(a[:, None], NCHAN, 1))

    def test_the_load_temperature_is_the_nearest_reading(self, campaign):
        s = write_solutions(campaign)
        b = bundle(campaign)
        tt = s["therm_t"]
        # Every row is 10 s or more from either reading; the first file
        # is nearer the first reading, the second nearer the second.
        expect = np.where(np.abs(b.t - tt[0]) <= np.abs(b.t - tt[1]), 300.0, 302.0)
        np.testing.assert_array_equal(b.products["tcal"]["tcal_t_load_k"], expect)

    def test_coefficients_are_eigsep_cals(self, campaign):
        s = write_solutions(campaign)
        b = bundle(campaign)
        tc = b.products["tcal"]
        t0, t1 = s["amb_t"]
        w = ((b.t - t0) / (t1 - t0))[:, None]
        a, off = tstar_coefficients(110 + 10 * w, 10 + 10 * w, T_NS, tc["tcal_t_load_k"][:, None])
        sc, of = receiver_s11_coefficients(a, off, tc["tcal_t_load_k"][:, None], 0.5, 0.05, 0.3j)
        np.testing.assert_allclose(tc["scale"], np.broadcast_to(sc, tc["scale"].shape))
        np.testing.assert_allclose(tc["offset"], np.broadcast_to(of, tc["offset"].shape))


class TestNothingIsStretched:
    def test_visits_in_different_regimes_do_not_bracket(self, campaign):
        write_solutions(campaign, amb_regime=np.array(["rx-A", "rx-B"]))
        tc = bundle(campaign).products["tcal"]
        assert np.isnan(tc["tstar_scale"]).all()
        assert np.isnan(tc["tcal_amb_gap_s"]).all()

    def test_a_gap_longer_than_the_limit_is_nan(self, campaign):
        write_solutions(campaign, max_cal_gap_s=1.0)
        assert np.isnan(bundle(campaign).products["tcal"]["scale"]).all()

    def test_no_extrapolation_past_the_last_visit(self, campaign):
        t = _times(campaign)
        write_solutions(
            campaign,
            amb_t=np.array([t.min() - 10, t.min() + 1]),
            non_t=np.array([t.min() - 10, t.min() + 1]),
        )
        b = bundle(campaign)
        late = b.t > t.min() + 1
        assert late.any()
        assert np.isnan(b.products["tcal"]["tstar_scale"][late]).all()
        assert np.isfinite(b.products["tcal"]["tstar_scale"][~late]).all()

    def test_a_stale_thermistor_reading_is_nan(self, campaign):
        write_solutions(campaign, therm_tol_s=1.0)
        tc = bundle(campaign).products["tcal"]
        assert np.isnan(tc["tcal_t_load_k"]).all()
        assert np.isnan(tc["tstar_offset"]).all()

    def test_another_input_is_not_calibrated(self, campaign):
        write_solutions(campaign)
        tc = bundle(campaign, antenna="box-gnd").products["tcal"]
        assert np.isnan(tc["scale"]).all()


class TestS11:
    def test_one_sweep_is_held_within_the_limit(self, campaign):
        t = _times(campaign)
        one = {}
        for dut, g in (("ant", 0.5), ("amb", 0.05), ("rec", 0.3j)):
            one[f"s11_{dut}_t"] = np.array([t.min() - 5.0])
            one[f"s11_{dut}"] = np.full((1, NCHAN), g, dtype=complex)
            one[f"s11_{dut}_epoch"] = np.array(["post-gap"])
        write_solutions(campaign, **one)
        tc = bundle(campaign).products["tcal"]
        assert np.isfinite(tc["scale"]).all()
        assert (tc["tcal_s11_ant_dt_s"] <= 1800.0).all()

    def test_sweeps_of_different_epochs_do_not_bracket(self, campaign):
        write_solutions(
            campaign,
            s11_ant_epoch=np.array(["pre-gap", "post-gap"]),
            s11_hold_s=1.0,
        )
        assert np.isnan(bundle(campaign).products["tcal"]["scale"]).all()


class TestBundleViews:
    def test_raw_data_are_untouched(self, campaign):
        write_solutions(campaign)
        b = bundle(campaign)
        raw = MetadataIndex(campaign / "data", cache=False).select().load_bundle(
            antenna="box-air", root=campaign
        )
        np.testing.assert_array_equal(b.data, raw.data)

    def test_calibrated_is_nan_off_the_antenna(self, campaign):
        write_solutions(campaign)
        b = bundle(campaign)
        cal = b.calibrated
        ant = (b.meta.rfswitch == "RFANT").to_numpy()
        assert np.isnan(cal[~ant]).all()
        assert np.isfinite(cal[ant]).all()

    def test_t_star_returns_the_load_on_load_rows(self, campaign):
        # A calibration applied to the very visits it was built from
        # has to give back their temperatures: put nodes on the RFAMB
        # and RFNON rows, set their spectra to those rows' own powers.
        b0 = MetadataIndex(campaign / "data", cache=False).select().load_bundle(
            antenna="box-air", root=campaign
        )
        st = b0.meta.rfswitch.to_numpy()
        amb, non = st == "RFAMB", st == "RFNON"
        write_solutions(
            campaign,
            amb_t=b0.t[amb], amb_p=b0.data[amb].astype(float),
            amb_regime=np.array(["rx-A"] * amb.sum()),
            non_t=b0.t[non], non_p=b0.data[non].astype(float),
            non_regime=np.array(["rx-A"] * non.sum()),
            therm_t=b0.t, therm_k=np.full(b0.t.size, 299.0),
        )
        b = bundle(campaign)
        # The first load row precedes every noise-source visit, so it
        # has no bracket and must stay NaN; the second has one.
        first, second = np.flatnonzero(amb)
        assert np.isnan(b.t_star[first]).all()
        np.testing.assert_allclose(b.t_star[second], 299.0, rtol=1e-9)

    def test_calibrated_needs_the_product(self, campaign):
        raw = MetadataIndex(campaign / "data", cache=False).select().load_bundle(
            antenna="box-air", root=campaign
        )
        with pytest.raises(KeyError, match="tcal"):
            raw.calibrated
