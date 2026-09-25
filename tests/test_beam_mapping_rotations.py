import numpy as np
import pytest

from eigsep_data.beam_mapping.beam_rotations import (
    body_to_enu,
    enu_to_body,
    mount_rotation,
    spherical_basis,
    vector_to_spherical,
)

PSI_MARJUM = 142.164      # Marjum 2026-07 highline, deg ccw from East


def compass_bearing(v):
    return np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360


def test_el_zero_points_boresight_to_zenith_for_any_roll():
    R = mount_rotation(np.linspace(-180, 180, 13), 0.0, PSI_MARJUM)
    np.testing.assert_allclose(body_to_enu([0, 0, 1], R), np.tile([0, 0, 1.0], (13, 1)), atol=1e-12)


def test_el_90_points_to_the_marjum_sweep_plane_and_180_to_nadir():
    b90 = mount_rotation(0, 90, PSI_MARJUM) @ [0, 0, 1]
    assert abs(b90[2]) < 1e-12
    assert compass_bearing(b90) == pytest.approx(37.836, abs=1e-9)
    np.testing.assert_allclose(mount_rotation(0, 180, PSI_MARJUM) @ [0, 0, 1], [0, 0, -1], atol=1e-12)


def test_arm_lies_along_the_axle_at_zero_roll():
    arm = mount_rotation(0, 0, PSI_MARJUM) @ [1, 0, 0]
    assert np.degrees(np.arctan2(arm[1], arm[0])) % 360 == pytest.approx(PSI_MARJUM)
    # Tipping in elevation does not move the axle.
    np.testing.assert_allclose(mount_rotation(0, 57, PSI_MARJUM) @ [1, 0, 0], arm, atol=1e-12)


def test_roll_is_right_handed_about_the_boresight():
    # psi = 0: axle along East. At el = 0 a +90 roll takes the arm from East to North.
    np.testing.assert_allclose(mount_rotation(90, 0, 0.0) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_rotations_are_proper_orthonormal_and_broadcast():
    rng = np.random.default_rng(1)
    az, el = rng.uniform(-180, 180, (4, 5)), rng.uniform(-180, 180, (4, 5))
    R = mount_rotation(az, el, PSI_MARJUM)
    assert R.shape == (4, 5, 3, 3)
    np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
    v = rng.normal(size=(4, 5, 3))
    np.testing.assert_allclose(enu_to_body(body_to_enu(v, R), R), v, atol=1e-12)


def test_spherical_helpers_round_trip():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(50, 3))
    th, ph = vector_to_spherical(v)
    r, thhat, phhat = spherical_basis(th, ph)
    np.testing.assert_allclose(r, v / np.linalg.norm(v, axis=1, keepdims=True), atol=1e-12)
    np.testing.assert_allclose(np.sum(r * thhat, 1), 0, atol=1e-12)
    np.testing.assert_allclose(np.sum(thhat * phhat, 1), 0, atol=1e-12)
