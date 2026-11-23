"""flagging.detectors.persistent_track on synthetic spectra."""

import warnings

import numpy as np
import pytest
from scipy.ndimage import median_filter

from eigsep_data.flagging import detectors as det

F = np.arange(0, 250, 250 / 1024)
LINE, BAND = int(np.argmin(abs(F - 120))), (F >= 45) & (F <= 235)


def spectrum(noise, line):
    """Power law plus a 6 MHz ripple: monotonic at 30-245 MHz if quiet."""
    lp = np.log10((np.maximum(F, 1) / 100) ** -2.5) + 1e-3 * np.sin(F)
    lp = lp + noise * np.random.default_rng(0).standard_normal((8, F.size))
    lp[:, LINE] += line
    return lp


def test_an_input_with_a_valid_mad_is_unchanged():
    lp = spectrum(3e-2, 0.3)
    med = np.median(lp, 0)
    resid = med - median_filter(med, size=31, mode="nearest")
    old = (resid > 6 * det._mad(resid[BAND]).item()) & BAND
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new, _ = det.persistent_track(lp, np.ones(8, bool), F)
    np.testing.assert_array_equal(new, old)
    assert new[LINE]


def test_a_monotonic_spectrum_falls_back_and_flags_its_line():
    with pytest.warns(RuntimeWarning, match="Savitzky-Golay baseline"):
        flags, _ = det.persistent_track(spectrum(0, 0.01), np.ones(8, bool), F)
    assert np.flatnonzero(flags).tolist() == [LINE]
