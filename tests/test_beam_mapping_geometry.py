import numpy as np

from eigsep_data.beam_mapping import (
    MOTOR_DEG_PER_STEP,
    PolarizationBeamMapper,
    TransmitterGeometry,
    rotation_matrix,
    fuse_pointing,
    imu_elevation_deg,
    estimate_motor_slip_steps,
    simulate_pointing_streams,
    TransmitterGeometry,
)


def beam(theta, phi):
    return 0.2 + np.cos(theta) ** 2, 0.1 + np.sin(theta) ** 2


def test_pointing_fusion_matches_notebook_conventions():
    accel = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    assert np.allclose(imu_elevation_deg(accel), [0, 45])
    table = fuse_pointing([0, 11300], [0, 11300], [10, 20], accel)
    assert np.allclose(table.az_deg, [5, 100], atol=1e-10)
    assert np.isclose(MOTOR_DEG_PER_STEP * 11300, 180)


def test_heading_and_alpha_recovered_from_off_axis_synthetic_data():
    mapper = PolarizationBeamMapper(None, None, sampler=beam)
    az = np.linspace(-175, 175, 180)
    el = 20 * np.sin(np.linspace(0, 3 * np.pi, az.size))
    arms = np.arange(az.size) % 2
    heading = np.array([0.8, 0.4, -0.45]); heading /= np.linalg.norm(heading)
    truth = TransmitterGeometry(heading, 27)
    y, _ = mapper.predict(az, el, truth, arms, scale=1.8, offset=0.03)
    fit = mapper.fit_heading(az, el, arms, y, [25, -15, 20])
    assert np.linalg.norm(fit.geometry.heading_top - heading) < 1e-5
    assert abs(fit.geometry.alpha_deg - 27) < 1e-5
    assert abs(fit.scale - 1.8) < 1e-5
    assert abs(fit.offset - 0.03) < 1e-5



def test_noisy_pointing_streams_expose_encoder_slip():
    true_az = np.linspace(-80, 80, 200)
    true_el = 12 * np.sin(np.linspace(0, 3 * np.pi, 200))
    streams = simulate_pointing_streams(true_az, true_el, pot_sigma_deg=0.1,
                                        motor_sigma_steps=0.2, slip_az_steps=37,
                                        slip_el_steps=-19, seed=5)
    assert abs(estimate_motor_slip_steps(streams.motor_az_steps, streams.pot_az_deg) - 37) < 1.0
    assert abs(estimate_motor_slip_steps(streams.motor_el_steps, streams.true_el_deg) + 19) < 1.0
    fused = fuse_pointing(streams.motor_az_steps - 37, streams.motor_el_steps + 19,
                          streams.pot_az_deg, streams.imu_accel, az_weight=0.1)
    assert np.sqrt(np.mean((fused.az_deg - true_az) ** 2)) < 1.0


def test_transmitter_arms_match_dominic_e2_convention():
    tx = TransmitterGeometry([0, 0, -1], alpha_deg=60)
    np.testing.assert_allclose(tx.field_top(0), [-np.sin(np.deg2rad(60)), 0.5, 0])
    np.testing.assert_allclose(tx.field_top(1), [-0.5, -np.sin(np.deg2rad(60)), 0])


def test_directed_azimuth_shaft_axis():
    np.testing.assert_allclose(
        rotation_matrix(90, 0, az_axis=(0, 0, -1)) @ [1, 0, 0],
        [0, -1, 0], atol=1e-12,
    )
