import healpy as hp
import numpy as np
import pytest

from eigsep_data.beam_mapping import HFSSBeamSet, simulate_hfss_coupling
from eigsep_data.beam_mapping.beam_rotations import mount_rotation
from eigsep_data.beam_mapping.tx_coupling import (
    TransmitterGeometry,
    heading_between,
    interpolate_fields,
    normalize_fields,
    tooth_gains,
    transmitter_coupling,
    transmitter_power,
)

NSIDE = 16


def dipole_fields(nfreq=3, nside=NSIDE):
    """Short-dipole-like far field along body +x: E = transverse part of x-hat."""
    v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))        # (3, npix)
    e = np.array([1.0, 0, 0])[:, None] - v[0][None, :] * v                   # x-hat minus radial part
    return np.repeat(e[None].astype(complex), nfreq, axis=0)


def test_on_axis_coupling_follows_the_roll_angle():
    fields = dipole_fields()
    # psi = 0 (axle along East), el = 0 (boresight up), transmitter overhead,
    # its arm-0 field along East (alpha = -90): full coupling at roll 0, none at roll 90.
    geom = TransmitterGeometry([0, 0, 1], alpha_deg=-90)
    az = np.array([0.0, 45.0, 90.0])
    R = mount_rotation(az, 0.0, 0.0)
    p = transmitter_power(fields, R, geom, arms=0)
    assert p[0, 0] > 0.99                        # bilinear sampling near the pole: within 1%
    np.testing.assert_allclose(p[0] / p[0, 0], np.cos(np.deg2rad(az)) ** 2, atol=1e-9)
    # Arm 1 is the orthogonal dipole.
    p1 = transmitter_power(fields, R, geom, arms=1)
    np.testing.assert_allclose(p1[0] / p[0, 0], np.sin(np.deg2rad(az)) ** 2, atol=1e-9)


def test_transmitter_direction_in_body_frame():
    geom = TransmitterGeometry(heading_between([0, 0, 0], [0, 0, -93.0]), 0.0)
    R = mount_rotation(np.array([0.0, 30.0]), np.array([0.0, 180.0]), 142.164)
    _, theta, _ = transmitter_coupling(dipole_fields(1), R, geom, arms=0)
    np.testing.assert_allclose(np.degrees(theta), [180.0, 0.0], atol=1e-9)   # nadir: -z at el 0, +z at el 180


def test_bilinear_and_nearest_agree_for_a_smooth_beam():
    fields = dipole_fields(nside=32)
    geom = TransmitterGeometry(heading_between([0, 0, 0], [7, 3, -90]), 40.0)
    rng = np.random.default_rng(0)
    R = mount_rotation(rng.uniform(-180, 180, 200), rng.uniform(-180, 180, 200), 142.164)
    a = transmitter_power(fields, R, geom, arms=0, interpolation='bilinear')
    b = transmitter_power(fields, R, geom, arms=0, interpolation='nearest')
    assert np.max(np.abs(a - b)) < 0.1


def test_normalize_and_interpolate_fields():
    rng = np.random.default_rng(3)
    npix = hp.nside2npix(4)
    base = rng.normal(size=(3, npix)) + 1j * rng.normal(size=(3, npix))
    freqs = np.array([100.0, 150.0, 200.0])
    fields = np.array([base * 2.0 * np.exp(1j * 0.3), base * 5.0 * np.exp(-1j * 1.1), base * 0.5])
    norm, amp, phase = normalize_fields(fields, freqs, reference_mhz=200.0)
    np.testing.assert_allclose(np.mean(np.sum(np.abs(norm) ** 2, axis=1), axis=1), 1.0)
    np.testing.assert_allclose(norm[0], norm[2], atol=1e-12)          # same shape, amplitude and phase removed
    np.testing.assert_allclose(interpolate_fields(norm, freqs, freqs[1]), norm[1], atol=1e-12)
    with pytest.raises(ValueError):
        interpolate_fields(norm, freqs, 250.0)


def test_tooth_gains_recover_known_gains_and_respect_masks():
    rng = np.random.default_rng(4)
    power = rng.uniform(0.1, 1.0, (3, 500))
    gains = np.array([2.0, 0.5, 7.0])
    data = gains[:, None] * power
    good = np.ones_like(data, bool)
    data[0, :10] = 1e6
    good[0, :10] = False
    np.testing.assert_allclose(tooth_gains(power, data, good), gains)
    assert tooth_gains(power, -data, good).max() == 0.0                 # non-negative


def test_tx_model_wrapper_matches_transmitter_coupling():
    fields = dipole_fields(2)
    beam = HFSSBeamSet(fields, np.zeros((2, fields.shape[-1])), np.zeros((2, fields.shape[-1])),
                       np.array([100.0, 150.0]))
    geom = TransmitterGeometry(heading_between([0, 0, 0], [7, 3, -90]), 40.0)
    az, el = np.array([10.0, 70.0, -30.0]), np.array([0.0, 120.0, -160.0])
    c1, _ = simulate_hfss_coupling(beam, az, el, geom, [0, 1, 0], psi_deg=142.164)
    c2, _, _ = transmitter_coupling(fields, mount_rotation(az, el, 142.164), geom, [0, 1, 0])
    np.testing.assert_allclose(c1, c2)
