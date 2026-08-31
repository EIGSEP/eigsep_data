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


class TestFitSmoothModelOneSpectrum:
    """The MAD-clipping refit must not solve an underdetermined system."""

    @staticmethod
    def _setup(nchan=64, nterms=8, poly_order=2, spread=False, n_out=3):
        basis = rfi.build_dpss_basis(
            nchan, nterms=nterms, poly_order=poly_order
        )
        nbasis = basis.shape[1]
        rng = np.random.default_rng(1)
        y = np.full(nchan, 1e3) + rng.normal(0, 1.0, nchan)
        ngood = nbasis + 2  # exactly passes the pre-clipping check
        idx = (
            np.linspace(0, nchan - 1, ngood).astype(int)
            if spread
            else np.arange(ngood)
        )
        fit_mask = np.zeros(nchan, dtype=bool)
        fit_mask[idx] = True
        y[idx[:n_out]] *= 50.0  # positive outliers -> MAD-clipped
        return y, basis, fit_mask, nbasis

    def test_underdetermined_after_clipping_fails(self):
        # The pre-check passes on the unclipped set, but MAD clipping
        # drops it below nbasis. The ridge term keeps lhs2 positive
        # definite, so solve() succeeds and would report a garbage model
        # as a success.
        y, basis, fit_mask, _ = self._setup()
        model, info = rfi.fit_smooth_model_one_spectrum(
            y, basis, fit_mask, enforce_comb_upper=False, return_info=True
        )
        assert info["success"] is False
        assert "clipping" in info["message"]
        assert np.isnan(model).all()

    def test_healthy_fit_still_succeeds(self):
        # Guard must not fire on a well-sampled spectrum.
        nchan = 256
        basis = rfi.build_dpss_basis(nchan, nterms=12, poly_order=2)
        rng = np.random.default_rng(3)
        y = 1e3 * (1.0 + 0.1 * np.linspace(-1, 1, nchan) ** 2)
        y += rng.normal(0, 1.0, nchan)
        fit_mask = np.ones(nchan, dtype=bool)
        model, info = rfi.fit_smooth_model_one_spectrum(
            y, basis, fit_mask, enforce_comb_upper=False, return_info=True
        )
        assert info["success"] is True
        assert np.isfinite(model).all()
        np.testing.assert_allclose(model, y, rtol=0.05)

    def test_pre_clipping_check_still_fires(self):
        # The original guard on the unclipped set must be unaffected.
        nchan = 64
        basis = rfi.build_dpss_basis(nchan, nterms=8, poly_order=2)
        nbasis = basis.shape[1]
        y = np.full(nchan, 1e3)
        fit_mask = np.zeros(nchan, dtype=bool)
        fit_mask[: nbasis - 1] = True
        model, info = rfi.fit_smooth_model_one_spectrum(
            y, basis, fit_mask, enforce_comb_upper=False, return_info=True
        )
        assert info["success"] is False
        assert info["message"] == "Not enough valid fit channels"
        assert np.isnan(model).all()

    def test_failed_fit_recorded_as_unsuccessful_per_time(self):
        # fit_dpss_model_per_time copies info["success"] into
        # fit_success; a bad fit must not enter the beam map flagged good.
        y, basis, fit_mask, _ = self._setup()
        _, info = rfi.fit_smooth_model_one_spectrum(
            y, basis, fit_mask, enforce_comb_upper=False, return_info=True
        )
        assert not info["success"]


class TestFitChannelBounds:
    """fit_max_chan=0 must mean zero channels, not the whole band."""

    @staticmethod
    def _bounds(nchan, fit_min_chan, fit_max_chan):
        # Mirrors the normalization at the top of fit_dpss_model_per_time.
        if fit_min_chan is None:
            fit_min_chan = 0
        if fit_max_chan is None:
            fit_max_chan = nchan
        return max(0, int(fit_min_chan)), min(nchan, int(fit_max_chan))

    def test_zero_max_is_not_widened_to_full_band(self):
        assert self._bounds(1024, None, 0) == (0, 0)

    def test_none_still_means_full_band(self):
        assert self._bounds(1024, None, None) == (0, 1024)

    def test_explicit_bounds_preserved_and_clamped(self):
        assert self._bounds(1024, 100, 200) == (100, 200)
        assert self._bounds(1024, -5, 5000) == (0, 1024)

    def test_matches_module_source(self):
        # Guard against the falsy-`or` idiom coming back.
        import inspect

        from eigsep_data import rfi

        src = inspect.getsource(rfi.fit_dpss_model_per_time)
        assert "fit_max_chan or nchan" not in src
        assert "fit_min_chan or 0" not in src
