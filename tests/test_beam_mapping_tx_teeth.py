import numpy as np

from eigsep_data.beam_mapping.tx_teeth import (
    isolation,
    neighbour_coherence,
    select_teeth,
    tooth_arms,
)

CHANNELS = np.arange(560, 800, 8)          # 30 teeth, alternating arms


def beam_like_signal(nrow=2000, seed=0):
    """Each arm's teeth share a smooth time series (the beam through a scan),
    scaled by a smooth per-tooth gain, plus 2% noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, nrow)
    arms = tooth_arms(CHANNELS)
    series = {a: 1.2 + np.sin(2 * np.pi * (7 + a) * t + a) * np.cos(2 * np.pi * 1.3 * t) for a in (0, 1)}
    sig = np.stack([series[a] * (1 + 0.002 * (c - 680)) for a, c in zip(arms, CHANNELS)], axis=1)
    return sig * (1 + 0.02 * rng.normal(size=sig.shape))


def test_arms_alternate_every_tooth():
    np.testing.assert_array_equal(tooth_arms([560, 568, 576, 584]), [0, 1, 0, 1])


def test_coherence_drops_an_rfi_tooth_and_keeps_its_neighbours():
    sig = beam_like_signal()
    bad = 14
    rng = np.random.default_rng(5)
    sig[:, bad] = np.where(rng.uniform(size=len(sig)) < 0.1, 50.0, 0.05)   # intermittent RFI, no beam
    good = np.ones_like(sig, bool)
    keep, dropped, first = neighbour_coherence(sig, good, CHANNELS, np.ones(len(CHANNELS), bool), 0.9)
    assert [d[0] for d in dropped] == [CHANNELS[bad]]
    assert keep.sum() == len(CHANNELS) - 1
    assert first[bad - 2] < 1.0 and all(v > 0.9 for i, v in first.items() if abs(i - bad) > 2)


def test_a_tooth_with_no_neighbour_is_dropped_and_threshold_zero_disables():
    sig = beam_like_signal()
    good = np.ones_like(sig, bool)
    keep0 = np.zeros(len(CHANNELS), bool)
    keep0[[0, 2, 4, 20]] = True                  # channel 720 (index 20) has no same-arm tooth within 48 channels
    keep, dropped, _ = neighbour_coherence(sig, good, CHANNELS, keep0, 0.9)
    assert [d[0] for d in dropped] == [CHANNELS[20]] and np.isnan(dropped[0][1])
    keep, dropped, _ = neighbour_coherence(sig, good, CHANNELS, keep0, 0.0)
    assert dropped == [] and np.array_equal(keep, keep0)


def test_select_teeth_reports_a_reason_for_every_tooth():
    sig = beam_like_signal(seed=1)
    background = np.full_like(sig, 0.5)
    good = np.ones_like(sig, bool)
    good[:, 3] = False
    good[:1500, 3] = True                         # 75% good: passes
    good[:1800, 7] = False                        # 10% good: fails good_fraction
    error = np.full(len(CHANNELS), 0.01)
    error[9] = 10.0                               # isolation ~0.3: fails
    sig[:, 11] = np.random.default_rng(2).uniform(0.4, 60.0, len(sig))   # RFI-like: fails coherence
    table = select_teeth(CHANNELS, sig + background, background, good, error)
    reason = dict(zip(table['channel'], table['reason']))
    assert reason[CHANNELS[7]] == 'good_fraction'
    assert reason[CHANNELS[9]] == 'isolation'
    assert reason[CHANNELS[11]] == 'coherence'
    assert reason[CHANNELS[3]] == 'selected'
    assert table['selected'].sum() == len(CHANNELS) - 3
    iso, s90 = isolation(sig / background, good, error)
    np.testing.assert_allclose(table['isolation'], iso)
