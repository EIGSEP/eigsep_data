"""Tests for eigsep_data.data."""

import warnings

import numpy as np

from eigsep_data import data


def _raster(n_plat=6, plat_len=20, ramp_len=3, step_deg=10.0):
    """Synthetic az raster: stable plateaus joined by short transitions."""
    seg, az_pot = [], []
    for k in range(n_plat):
        seg += [2 * k] * plat_len
        az_pot += list(
            np.full(plat_len, step_deg * k)
            + np.random.default_rng(k).normal(0, 0.01, plat_len)
        )
        if k < n_plat - 1:
            seg += [2 * k + 1] * ramp_len
            az_pot += list(
                np.linspace(step_deg * k, step_deg * (k + 1), ramp_len + 2)[
                    1:-1
                ]
            )
    return np.array(az_pot, dtype=float), np.array(seg, dtype=float)


class TestExtractCleanPotDataV2:
    """A dropped pot sample must not take its neighbours with it."""

    def test_single_nan_does_not_destroy_plateau(self):
        # np.median propagates NaN, and the median is broadcast over the
        # whole plateau and then into the ramps on both sides -- one bad
        # sample used to cost ~26.
        az_pot, az_step = _raster()
        y = az_pot.copy()
        y[30] = np.nan
        out = data.extract_clean_pot_data_v2(y, az_step)
        assert not np.isnan(out).any()

    def test_partial_dropout_run_recovers(self):
        # Real Jul-17 dropout runs span many samples but rarely a whole
        # plateau; those must still yield a usable median.
        az_pot, az_step = _raster()
        y = az_pot.copy()
        y[23:40] = np.nan  # 17 of the 20 samples in one plateau
        out = data.extract_clean_pot_data_v2(y, az_step)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out[23:43], 10.0, atol=0.1)

    def test_nan_at_edges_does_not_flood_array(self):
        # The first/last plateau median fills everything before/after it.
        az_pot, az_step = _raster()
        for idx in (5, len(az_pot) - 5):
            y = az_pot.copy()
            y[idx] = np.nan
            out = data.extract_clean_pot_data_v2(y, az_step)
            assert not np.isnan(out).any()

    def test_fully_missing_plateau_stays_nan_but_is_contained(self):
        # No measurement exists here, so it must stay flagged missing
        # rather than being interpolated across -- but it must not
        # spread beyond its own samples.
        az_pot, az_step = _raster()
        y = az_pot.copy()
        y[23:43] = np.nan
        out = data.extract_clean_pot_data_v2(y, az_step)
        assert np.isnan(out[23:43]).all()
        assert np.isfinite(out[:23]).all()
        assert np.isfinite(out[43:]).all()

    def test_fully_missing_plateau_emits_no_warning(self):
        # np.nanmedian over an all-NaN slice warns; the plateau must be
        # skipped before that happens.
        az_pot, az_step = _raster()
        y = az_pot.copy()
        y[23:43] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data.extract_clean_pot_data_v2(y, az_step)

    def test_clean_data_unchanged(self):
        az_pot, az_step = _raster()
        base = data.extract_clean_pot_data_v2(az_pot, az_step)
        assert not np.isnan(base).any()
        np.testing.assert_allclose(base[:20], 0.0, atol=0.1)
        np.testing.assert_allclose(base[23:43], 10.0, atol=0.1)

    def test_untouched_plateaus_bit_identical(self):
        # A NaN in one plateau must not perturb any other.
        az_pot, az_step = _raster()
        base = data.extract_clean_pot_data_v2(az_pot, az_step)
        y = az_pot.copy()
        y[30] = np.nan
        out = data.extract_clean_pot_data_v2(y, az_step)
        np.testing.assert_array_equal(out[:20], base[:20])
        np.testing.assert_array_equal(out[46:], base[46:])


CPD = 62.77777777777778  # stepper counts per degree


def _el_scan(span_deg, n=800, noise=1e-3, seed=0):
    """Gravity vector for an elevation scan, plus commanded motor counts."""
    el = np.linspace(-span_deg / 2, span_deg / 2, n)
    th = np.deg2rad(el)
    accel = np.stack([np.zeros(n), np.sin(th), -np.cos(th)], axis=1)
    accel += np.random.default_rng(seed).normal(0, noise, accel.shape)
    return el, accel, el * CPD


class TestImuElFromAccel:
    """The SVD plane basis is arbitrary; el_pos must anchor it."""

    def test_recovers_truth(self):
        for span in (80, 180, 260, 360):
            el, accel, el_pos = _el_scan(span)
            out = data.imu_el_from_accel(accel, el_pos)
            assert np.max(np.abs(out - el)) < 1.0

    def test_windows_agree_on_sign_and_origin(self):
        # Two calls over different time windows of one physical scan used
        # to disagree by up to ~180 deg in origin, and could come back
        # mirrored. They must now agree wherever they overlap.
        for span in (80, 180, 260, 360):
            el, accel, el_pos = _el_scan(span)
            full = data.imu_el_from_accel(accel, el_pos)
            for lo, hi in [(5, 800), (0, 795), (20, 780), (137, 642)]:
                got = data.imu_el_from_accel(accel[lo:hi], el_pos[lo:hi])
                np.testing.assert_allclose(got, full[lo:hi], atol=1.0)

    def test_sign_follows_motor_direction(self):
        # A scan run in the opposite direction must come back with the
        # opposite sense, not the same one.
        el, accel, el_pos = _el_scan(180)
        fwd = data.imu_el_from_accel(accel, el_pos)
        rev = data.imu_el_from_accel(accel[::-1], el_pos[::-1])
        assert np.polyfit(el, fwd, 1)[0] > 0
        assert np.polyfit(el[::-1], rev, 1)[0] > 0

    def test_nan_accel_rows_stay_nan(self):
        el, accel, el_pos = _el_scan(180)
        accel[100:110] = np.nan
        out = data.imu_el_from_accel(accel, el_pos)
        assert np.isnan(out[100:110]).all()
        assert np.isfinite(out[:100]).all()
        assert np.isfinite(out[110:]).all()

    def test_partial_nan_el_pos_still_anchors(self):
        el, accel, el_pos = _el_scan(180)
        el_pos = el_pos.copy()
        el_pos[::3] = np.nan  # most of the anchor dropped
        out = data.imu_el_from_accel(accel, el_pos)
        assert np.max(np.abs(out - el)) < 1.0

    def test_fixed_elevation_scan_is_consistent(self):
        # Zero covariance leaves the sign unobservable; the offset must
        # still place the answer at the commanded elevation.
        n = 400
        th = np.full(n, np.deg2rad(30.0))
        accel = np.stack([np.zeros(n), np.sin(th), -np.cos(th)], axis=1)
        accel += np.random.default_rng(0).normal(0, 1e-3, accel.shape)
        out = data.imu_el_from_accel(accel, np.full(n, 30.0 * CPD))
        np.testing.assert_allclose(out, 30.0, atol=1.0)

    def test_warns_without_usable_motor_position(self):
        el, accel, _ = _el_scan(180)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = data.imu_el_from_accel(accel, np.full(len(el), np.nan))
        assert len(w) == 1
        assert "anchor" in str(w[0].message)
        assert np.isfinite(out).all()

    def test_all_nan_accel_returns_all_nan(self):
        accel = np.full((50, 3), np.nan)
        out = data.imu_el_from_accel(accel, np.arange(50.0) * CPD)
        assert np.isnan(out).all()


class TestCalibrateWeakArm:
    def test_no_valid_crossings_returns_nan_not_indexerror(self):
        # A fixed-elevation azimuth raster has no el=0 crossings, so
        # power_at_0 is shape (0,) rather than (0, nfreq) and the
        # per-frequency loop used to raise IndexError before reaching
        # its own validity check.
        n = 200
        el = np.full(n, 45.0)
        el[::40] = 44.0
        az = np.linspace(0, 360, n)
        dpss = np.ones((n, 8))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scale, peaks = data.calibrate_weak_arm(el, az, dpss)
        assert scale.shape == (4,)
        assert peaks.shape == (8,)
        assert np.isnan(scale).all()
        assert np.isnan(peaks).all()
        assert len(w) == 1
        assert "crossings" in str(w[0].message)

    def test_crossings_outside_ten_degrees_are_not_valid(self):
        # Crossings exist but all are steep enough that |el| >= 10 at the
        # sample before zero, so valid_crossings is still empty.
        n = 100
        el = np.tile([30.0, -30.0], n // 2)
        az = np.linspace(0, 360, n)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            scale, peaks = data.calibrate_weak_arm(el, az, np.ones((n, 4)))
        assert np.isnan(scale).all()

    def test_normal_scan_still_fits(self):
        el = 45 * np.sin(np.linspace(0, 8 * np.pi, 400))
        az = np.linspace(0, 720, 400)
        dpss = (
            np.abs(np.cos(np.deg2rad(az)))[:, None] ** 2
            * np.arange(1, 9)[None, :]
            + 0.01
        )
        scale, peaks = data.calibrate_weak_arm(el, az, dpss)
        assert np.isfinite(peaks).all()
        assert scale.shape == (4,)


class TestParseTimeFromName:
    def test_handles_deployment_naming_variants(self):
        cases = {
            "corr_20250922_160500.h5": (2025, 9, 22, 16, 5, 0),
            "corr_20260715_172825Z.h5": (2026, 7, 15, 17, 28, 25),
            "corr_20260712_235712Z-1.h5": (2026, 7, 12, 23, 57, 12),
        }
        for name, expect in cases.items():
            got = data._parse_time_from_name(name)
            assert (
                got.year,
                got.month,
                got.day,
                got.hour,
                got.minute,
                got.second,
            ) == expect

    def test_unparseable_name_raises(self):
        import pytest

        with pytest.raises(ValueError):
            data._parse_time_from_name("not_a_corr_file.h5")
