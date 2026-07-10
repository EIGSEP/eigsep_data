"""Tests for eigsep_data.rfi."""

import warnings

import numpy as np

from eigsep_data import rfi


class TestRobustDivide:
    def test_zero_denominator_gives_inf(self):
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([1.0, 0.0, 2.0])
        out = rfi.robust_divide(num, den)
        np.testing.assert_array_equal(out, [1.0, np.inf, 1.5])

    def test_no_warnings_emitted(self):
        num = np.ones((10, 10))
        den = np.zeros((10, 10))
        den[::2] = 2.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = rfi.robust_divide(num, den)
        assert np.isinf(out[1]).all()
        np.testing.assert_array_equal(out[0], 0.5)


class TestMedianFlagger:
    def test_quantized_data_not_all_flagged(self):
        # >50% exact-zero residuals (deep attenuation, few-count
        # levels) must not collapse the MAD to zero and flag the world
        rng = np.random.default_rng(7)
        d = rng.poisson(0.3, (50, 256)).astype(float)
        d[:, 100] += 100.0
        flags = rfi.median_flagger(d, nsig=8)
        assert flags[:, 100].all()
        assert flags.mean() < 0.3
        clean = np.ones(256, dtype=bool)
        clean[95:106] = False
        assert flags[:, clean].mean() < 0.05

    def test_constant_data_unflagged(self):
        d = np.full((20, 64), 7.0)
        flags = rfi.median_flagger(d)
        assert not flags.any()
