"""Shared array helpers: eigsep_data.numerics."""

import numpy as np
import pytest

from eigsep_data.numerics import interp_complex, weighted_mean
from eigsep_data.products.base import derived_dir, versions_in


def test_weighted_mean_drops_bad_values_and_weights():
    x = np.array([1.0, 3.0, np.nan, 100.0, 5.0])
    w = np.array([1.0, 3.0, 2.0, 0.0, np.inf])
    assert float(weighted_mean(x, w)) == pytest.approx((1 + 9) / 4)


def test_weighted_mean_is_nan_with_nothing_left():
    assert np.isnan(weighted_mean([1.0, 2.0], [0.0, -1.0]))


def test_weighted_mean_along_an_axis_broadcasts_weights():
    x = np.array([[1.0, 2.0], [3.0, np.nan]])
    w = np.array([[1.0], [3.0]])
    np.testing.assert_allclose(weighted_mean(x, w, axis=0), [2.5, 2.0])


def test_interp_complex_one_curve_and_a_stack():
    xp = np.array([0.0, 1.0, 2.0])
    fp = np.array([0 + 0j, 1 + 2j, 2 + 4j])
    x = np.array([-1.0, 0.5, 1.5, 3.0])
    one = interp_complex(x, xp, fp)
    assert np.isnan(one[0]) and np.isnan(one[-1])
    np.testing.assert_allclose(one[1:3], [0.5 + 1j, 1.5 + 3j])
    stack = interp_complex(x, xp, np.stack([fp, 2 * fp]))
    assert stack.shape == (2, 4)
    np.testing.assert_allclose(stack[1, 1:3], 2 * one[1:3])
    assert interp_complex(x, xp, np.empty((0, 3), complex)).shape == (0, 4)


def test_derived_dir_and_versions_in(tmp_path):
    assert derived_dir(tmp_path, "tcal") == tmp_path / "derived" / "tcal"
    assert (
        derived_dir(tmp_path, "tcal", "v0002")
        == tmp_path / "derived" / "tcal" / "v0002"
    )
    assert versions_in(derived_dir(tmp_path, "tcal")) == []
    for v, payload in (("v0002", True), ("v0001", True), ("v0003", False)):
        d = derived_dir(tmp_path, "tcal", v)
        d.mkdir(parents=True)
        if payload:
            (d / "solutions.npz").write_bytes(b"")
    assert versions_in(derived_dir(tmp_path, "tcal")) == ["v0001", "v0002"]
