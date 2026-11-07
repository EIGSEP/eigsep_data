"""Tests for tcal's load temperature: nearest reading or window mean.

These need no eigsep_cal, unlike the rest of the tcal tests, so they run
everywhere.
"""

import numpy as np
import pytest

from eigsep_data.products.tcal import load_temperature

CADENCE = 0.537
SIGMA = 0.196
TOL = 60.0


def _nearest_old(tt, tk, t, tol_s):
    """tcal's nearest-reading rule before windows existed, verbatim."""
    i = np.clip(np.searchsorted(tt, t), 1, max(tt.size - 1, 1))
    i = np.where(np.abs(tt[i - 1] - t) <= np.abs(tt[i] - t), i - 1, i)
    therm_dt = np.abs(tt[i] - t)
    return np.where(therm_dt <= tol_s, tk[i], np.nan), therm_dt


def stream(n=20000, seed=0):
    """A white-noise thermistor stream about 300 K, stored as the builder
    stores it (float32 kelvin on sorted float64 times)."""
    rng = np.random.default_rng(seed)
    tt = 1.7843e9 + CADENCE * np.arange(n)
    tk = (300.0 + SIGMA * rng.standard_normal(n)).astype(np.float32)
    return tt, tk


class TestNearest:
    def test_default_is_the_old_rule_bit_for_bit(self):
        tt, tk = stream()
        rng = np.random.default_rng(1)
        t = np.concatenate([
            rng.uniform(tt[0] - 100, tt[-1] + 100, 500),
            tt[::997],                          # exactly on readings
            0.5 * (tt[:-1:991] + tt[1::991]),   # ties between two
        ])
        old_k, old_dt = _nearest_old(tt, tk, t, TOL)
        new_k, new_dt, n = load_temperature(tt, tk, t, TOL)
        assert n is None
        assert new_k.dtype == old_k.dtype
        np.testing.assert_array_equal(new_k, old_k)
        np.testing.assert_array_equal(new_dt, old_dt)


class TestWindowMean:
    def test_is_the_mean_of_the_readings_inside(self):
        tt, tk = stream()
        t = tt[100:-100:300] + 0.1
        start, stop = t - 30.0, t + 30.0
        k, _, n = load_temperature(tt, tk, t, TOL, (start, stop))
        for j in range(t.size):
            inside = (tt >= start[j]) & (tt <= stop[j])
            assert n[j] == inside.sum()
            assert k[j] == pytest.approx(
                np.mean(tk[inside].astype(float)), abs=1e-9
            )

    def test_edges_are_inclusive(self):
        tt, tk = stream(n=10)
        k, _, n = load_temperature(
            tt, tk, tt[[4]], TOL, (tt[[2]], tt[[6]])
        )
        assert n[0] == 5
        assert k[0] == pytest.approx(np.mean(tk[2:7].astype(float)))

    @pytest.mark.parametrize("width", [1, 4, 16, 64, 256])
    def test_scatter_falls_as_one_over_root_n(self, width):
        tt, tk = stream(n=200000, seed=2)
        # Disjoint windows of `width` readings each.
        first = np.arange(0, tt.size - width, width)
        start, stop = tt[first], tt[first + width - 1]
        k, _, n = load_temperature(tt, tk, 0.5 * (start + stop), TOL,
                                   (start, stop))
        assert np.all(n == width)
        err = np.std(k - 300.0)
        expect = SIGMA / np.sqrt(width)
        # 1/sqrt(2 * nwin) relative error on a std; nwin >= 780 here.
        assert err == pytest.approx(expect, rel=4 / np.sqrt(2 * first.size))

    def test_a_scalar_window_broadcasts(self):
        tt, tk = stream(n=100)
        t = tt[[10, 20]]
        k, _, n = load_temperature(tt, tk, t, TOL, (tt[0], tt[-1]))
        np.testing.assert_array_equal(n, [100, 100])
        np.testing.assert_allclose(k, np.mean(tk.astype(float)))


class TestWindowEdgeCases:
    def test_an_empty_window_falls_back_to_the_nearest_reading(self):
        tt, tk = stream(n=100)
        t = np.array([tt[10] + 0.2])
        start, stop = t - 0.01, t + 0.01        # between two readings
        k, dt, n = load_temperature(tt, tk, t, TOL, (start, stop))
        near, near_dt = _nearest_old(tt, tk, t, TOL)
        assert n[0] == 0
        assert k[0] == near[0]
        assert dt[0] == near_dt[0]

    def test_an_empty_window_beyond_the_tolerance_is_nan(self):
        tt, tk = stream(n=100)
        t = np.array([tt[-1] + 2 * TOL])
        k, _, n = load_temperature(tt, tk, t, TOL, (t - 1.0, t + 1.0))
        assert n[0] == 0
        assert np.isnan(k[0])

    @pytest.mark.parametrize("end", ["first", "last"])
    def test_a_window_past_the_stream_uses_what_is_there(self, end):
        tt, tk = stream(n=100)
        if end == "first":
            t, start, stop = tt[:1], tt[:1] - 50.0, tt[4:5]
            inside = slice(0, 5)
        else:
            t, start, stop = tt[-1:], tt[-5:-4], tt[-1:] + 50.0
            inside = slice(95, 100)
        k, _, n = load_temperature(tt, tk, t, TOL, (start, stop))
        assert n[0] == 5
        assert k[0] == pytest.approx(np.mean(tk[inside].astype(float)))

    def test_nan_readings_are_skipped(self):
        tt, tk = stream(n=100)
        tk[[3, 5]] = np.nan
        k, _, n = load_temperature(tt, tk, tt[[4]], TOL, (tt[[2]], tt[[6]]))
        assert n[0] == 3
        assert k[0] == pytest.approx(np.mean(tk[[2, 4, 6]].astype(float)))

    def test_a_window_of_only_nan_readings_falls_back(self):
        tt, tk = stream(n=100)
        tk[[3, 4]] = np.nan
        t = tt[[3]] + 0.1
        k, _, n = load_temperature(tt, tk, t, TOL, (tt[[3]], tt[[4]]))
        assert n[0] == 0
        # The fallback is the old rule, which takes the reading as it is.
        assert np.isnan(k[0])

    def test_a_nan_edge_is_an_empty_window(self):
        tt, tk = stream(n=100)
        t = tt[[10]]
        k, _, n = load_temperature(
            tt, tk, t, TOL, (np.array([np.nan]), t + 5.0)
        )
        assert n[0] == 0
        assert k[0] == tk[10]

    def test_a_reversed_window_is_refused(self):
        tt, tk = stream(n=100)
        with pytest.raises(ValueError, match="ends before it starts"):
            load_temperature(tt, tk, tt[[10]], TOL, (tt[[12]], tt[[8]]))
