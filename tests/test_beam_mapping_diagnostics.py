import numpy as np

from eigsep_data.beam_mapping.diagnostics import (
    channel_validity_masks,
    gross_power_time_flags,
    isolated_map_outliers,
    radiometer_difference_sigma,
    tx_arm_for_channel,
)


def test_tx_polarization_alternates_every_eight_channels():
    assert tx_arm_for_channel(536) != tx_arm_for_channel(544)
    assert tx_arm_for_channel(544) != tx_arm_for_channel(552)
    assert tx_arm_for_channel(536) == tx_arm_for_channel(552)


def test_intrinsic_channel_flags_stay_local_but_time_flags_are_shared():
    data = {
        "times": np.ones(3),
        "measured_tx": np.array([
            [20.0, 20.0],
            [0.0, 20.0],
            [20.0, 20.0],
        ]),
    }
    local = channel_validity_masks(data, [0, 1], threshold=10.0)
    assert local.tolist() == [[True, True], [False, True], [True, True]]
    shared = channel_validity_masks(
        data, [0, 1], threshold=10.0,
        shared_time_flags=np.array([False, False, True]))
    assert shared.tolist() == [[True, True], [False, True], [False, False]]


def test_adjacent_channel_subtraction_radiometer_noise():
    sigma = radiometer_difference_sigma(
        tx_power=4.0, left_power=2.0, right_power=2.0,
        bandwidth_hz=2.0, integration_s=1.0)
    assert np.isclose(sigma, 3.0)


def test_map_outliers_preserve_broad_nulls_and_flag_isolated_samples():
    residual = np.zeros(40)
    residual[8:20] = -20.0
    residual[30] = 30.0
    flagged = isolated_map_outliers(
        residual, np.ones(40, dtype=bool),
        az_deg=np.arange(40), el_deg=np.zeros(40),
        clip_sigma=5.0, neighbors=5)
    assert flagged[8:20].sum() <= 2
    assert not np.any(flagged[10:18])
    assert flagged[30]


def test_gross_power_flags_preserve_nulls_and_share_only_large_excursions():
    measured = np.ones((200, 2))
    measured[20:60, 0] = 0.0
    measured[100, 1] = 100.0
    measured[120, 0] = -3.0
    data = {
        "times": np.ones(200),
        "measured_tx": measured,
        "measured_sigma": np.ones_like(measured),
    }
    shared, by_channel, _ = gross_power_time_flags(
        data, [0, 1], reference_percentile=99.0, factor=5.0)
    assert not np.any(shared[20:60])
    assert shared[100]
    assert not by_channel[100, 0]
    assert by_channel[100, 1]
    assert shared[120]
    assert by_channel[120, 0]
    assert not by_channel[120, 1]
