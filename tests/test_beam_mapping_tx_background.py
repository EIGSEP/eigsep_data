import numpy as np
import pytest

from eigsep_data.beam_mapping.tx_background import (
    background_windows,
    dpss_modes,
    tooth_background,
)

NCHAN = 1024
DNU = 250.0 / NCHAN
FREQ = np.arange(NCHAN) * DNU
TEETH = np.arange(8, NCHAN - 8, 8)
BAND = (FREQ > 55) & (FREQ < 235)
FM = (FREQ > 87) & (FREQ < 108.5)
LIMITS = (40, 249)
RIPPLE_NS = 110.0


def sky(ripple=0.0, nrow=20, seed=0):
    """Power-law sky with an optional ripple at RIPPLE_NS, varying in level and phase per row."""
    rng = np.random.default_rng(seed)
    amp = rng.uniform(0.9, 1.1, (nrow, 1))
    phase = rng.uniform(0, 2 * np.pi, (nrow, 1))
    law = 1e6 * (np.maximum(FREQ, 30.0) / 150.0) ** -2.5
    return amp * law * (1 + ripple * np.cos(2 * np.pi * RIPPLE_NS * 1e-3 * FREQ + phase)), phase


def with_comb(bg):
    """A comb 50x the sky on every tooth, leaking 0.5x into the +/-1 channels."""
    out = bg.copy()
    out[:, TEETH] += 50 * bg[:, TEETH]
    out[:, TEETH + 1] += 0.5 * bg[:, TEETH + 1]
    out[:, TEETH - 1] += 0.5 * bg[:, TEETH - 1]
    return out


def band_teeth(exclude_fm=False):
    keep = BAND[TEETH] & (~FM[TEETH] if exclude_fm else True)
    return TEETH[keep]


def test_default_dpss_modes_represent_smooth_functions():
    n = 196
    A = dpss_modes(n, 150.0, DNU)
    x = np.arange(n) / n
    for y in (np.ones(n), 1 + x, (1 + 0.9 * x) ** -2.5):
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        assert np.max(np.abs(resid)) < 1e-3
    # The v0009-v0011 rule keeps floor(2NW) + 1 modes and cannot (kept only for reproduction).
    legacy = dpss_modes(n, 150.0, DNU, concentration_min=None)
    assert legacy.shape[1] == int(2 * 150.0 * n * DNU * 1e-3) + 1 < A.shape[1]
    resid = 1 - legacy @ np.linalg.lstsq(legacy, np.ones(n), rcond=None)[0]
    assert np.max(np.abs(resid)) > 1e-2


@pytest.mark.parametrize('ripple', [0.0, 0.03])
def test_dpss_recovers_the_background_under_the_comb(ripple):
    bg, _ = sky(ripple)
    teeth = band_teeth()
    est = tooth_background(with_comb(bg), FREQ, teeth, limits_mhz=LIMITS)
    err = est / bg[:, teeth] - 1
    assert np.all(np.isfinite(est))
    assert np.max(np.abs(err)) < 1e-3


def test_gap_recovers_a_smooth_background_under_the_comb():
    bg, _ = sky(0.0)
    teeth = band_teeth()
    est = tooth_background(with_comb(bg), FREQ, teeth, method='gap')
    err = est / bg[:, teeth] - 1
    # Only the curvature across +/-4 channels remains: largest at the low end, ~1e-3.
    assert np.max(np.abs(err)) < 2e-3
    assert np.median(np.abs(err)) < 3e-4


def test_gap_misestimates_a_ripple_by_the_predicted_fraction():
    # On a flat background the gap mean of a ripple is exact: cos(2 pi tau f) * cos(2 pi tau g dnu).
    ripple, g = 0.03, 4
    phase = np.random.default_rng(0).uniform(0, 2 * np.pi, (20, 1))
    bg = 1e6 * (1 + ripple * np.cos(2 * np.pi * RIPPLE_NS * 1e-3 * FREQ + phase))
    teeth = band_teeth()
    est = tooth_background(with_comb(bg), FREQ, teeth, method='gap')
    kernel = np.cos(2 * np.pi * RIPPLE_NS * 1e-3 * g * DNU)
    predicted = 1e6 * (1 + ripple * kernel * np.cos(2 * np.pi * RIPPLE_NS * 1e-3 * FREQ[teeth] + phase))
    np.testing.assert_allclose(est, predicted, rtol=1e-12)
    # So the error at a tooth reaches ripple * (1 - kernel), 0.66% here, where DPSS is under 1e-3.
    assert np.max(np.abs(est / bg[:, teeth] - 1)) == pytest.approx(ripple * (1 - kernel), rel=0.05)
    dpss_est = tooth_background(with_comb(bg), FREQ, teeth, limits_mhz=LIMITS)
    assert np.max(np.abs(dpss_est / bg[:, teeth] - 1)) < 1e-3


def test_symmetric_clipping_ignores_one_sided_outliers():
    bg, _ = sky(0.03, nrow=5, seed=1)
    teeth = band_teeth()
    clean = tooth_background(bg, FREQ, teeth, limits_mhz=LIMITS)
    rng = np.random.default_rng(2)
    off_tooth = np.flatnonzero(BAND & (np.abs((np.arange(NCHAN) + 4) % 8 - 4) > 1))
    spiky = bg.copy()
    spiky[:, rng.choice(off_tooth, 40, replace=False)] *= 3.0          # narrowband RFI, positive only
    est = tooth_background(spiky, FREQ, teeth, limits_mhz=LIMITS)
    assert np.max(np.abs(est / clean - 1)) < 2e-3


@pytest.mark.parametrize('method', ['dpss', 'gap'])
def test_excluded_channels_are_never_used(method):
    bg, _ = sky(0.0, nrow=3, seed=3)
    teeth = band_teeth(exclude_fm=True)
    junk = bg.copy()
    junk[:, FM] = 1e9
    a = tooth_background(bg, FREQ, teeth, exclude=FM, limits_mhz=LIMITS, method=method)
    b = tooth_background(junk, FREQ, teeth, exclude=FM, limits_mhz=LIMITS, method=method)
    np.testing.assert_allclose(a, b)
    assert np.all(np.isfinite(b))       # gap: a tooth beside FM falls back to its other side


def test_windows_cover_every_tooth_once_and_residual_rms():
    teeth = band_teeth()
    covered = np.concatenate([ev for _, ev in background_windows(FREQ, teeth, limits_mhz=LIMITS)])
    assert sorted(covered) == sorted(teeth)
    bg, _ = sky(0.0, nrow=2)
    for method, limit in (('dpss', 1e-3), ('gap', 5e-2)):
        est, rms = tooth_background(bg, FREQ, teeth[:5], limits_mhz=LIMITS, method=method,
                                    return_residual_rms=True)
        assert est.shape == rms.shape == (2, 5)
        assert np.all(rms < limit)
    with pytest.raises(ValueError):
        tooth_background(bg, FREQ, teeth, method='spline')
