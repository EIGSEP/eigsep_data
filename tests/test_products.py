"""Tests for the product plugin contract."""

import numpy as np
import pytest

from eigsep_data import products
from eigsep_data.products import base


class TestParseSpec:
    def test_splits_kind_and_version(self):
        assert base.parse_spec("flags@v2") == ("flags", "v2")

    def test_tolerates_surrounding_space(self):
        assert base.parse_spec(" flags @ v0 ") == ("flags", "v0")

    @pytest.mark.parametrize("spec", ["flags", "flags@", "@v2", "", None])
    def test_a_bare_kind_is_refused_never_defaulted(self, spec):
        # The whole point: flags@v0 is uint8 and flags@v2 is uint16, so
        # a default version silently changes the dtype of somebody's
        # arrays without one line of their call site changing.
        with pytest.raises((ValueError, TypeError)):
            base.parse_spec(spec)


class TestRegistry:
    def test_ships_the_products_with_real_consumers(self):
        assert products.registered() == [
            "flags",
            "gain",
            "pointing",
            "smooth_model",
        ]

    def test_get_returns_an_instance_not_the_class(self):
        # A class would make every product method an unbound function
        # and defeat the per-load caching the Parquet reader relies on.
        assert isinstance(products.get("flags"), base.Product)

    def test_unknown_kind_names_what_is_available(self):
        with pytest.raises(KeyError, match="smooth_model"):
            products.get("nope")


class TestAxisFingerprint:
    def test_same_axis_same_fingerprint(self):
        f = np.linspace(45, 235, 778)
        assert base.axis_fingerprint(f) == base.axis_fingerprint(f.copy())

    def test_a_rebuild_on_a_different_band_differs(self):
        a = base.axis_fingerprint(np.linspace(45, 235, 778))
        b = base.axis_fingerprint(np.linspace(50, 235, 778))
        assert a != b

    def test_is_three_scalars_not_the_axis(self):
        # The per-file check has to stay O(1); if this ever becomes the
        # array itself, every file in a 5120-file load pays for it.
        fp = base.axis_fingerprint(np.linspace(45, 235, 778))
        assert len(fp) == 3
        assert all(np.isscalar(v) for v in fp)


class TestLocateAxis:
    def setup_method(self):
        self.full = np.linspace(0, 250, 1024, endpoint=False)

    def test_finds_a_contiguous_sub_band(self):
        start, stop = base.locate_axis(self.full, self.full[184:962], "x")
        assert (start, stop) == (184, 962)

    def test_a_whole_axis_locates_at_zero(self):
        assert base.locate_axis(self.full, self.full, "x") == (0, 1024)

    def test_a_shifted_grid_raises_rather_than_snapping(self):
        # Half a channel off is exactly the failure a shape check misses
        # and an interpolating loader would paper over.
        shifted = self.full[184:962] + 0.12
        with pytest.raises(ValueError, match="not a contiguous slice"):
            base.locate_axis(self.full, shifted, "smooth_model@v0")

    def test_a_decimated_grid_raises(self):
        with pytest.raises(ValueError, match="not a contiguous slice"):
            base.locate_axis(self.full, self.full[::2], "x")

    def test_a_longer_axis_is_named_as_such(self):
        with pytest.raises(ValueError, match="more than"):
            base.locate_axis(self.full, np.linspace(0, 250, 2048), "x")

    def test_the_error_names_the_product(self):
        with pytest.raises(ValueError, match=r"flags@v2"):
            base.locate_axis(self.full, self.full[::2], "flags@v2")


class TestReadManifest:
    def test_absent_manifest_is_empty_not_fatal(self, tmp_path):
        assert base.read_manifest(tmp_path / "nope.json") == {}

    def test_unparseable_manifest_reports_rather_than_raises(self, tmp_path):
        bad = tmp_path / "manifest.json"
        bad.write_text("{not json")
        assert "error" in base.read_manifest(bad)
