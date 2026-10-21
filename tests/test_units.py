"""Unit conversions."""

import numpy as np

from eigsep_data.units import from_dB, to_dB


def test_dB_round_trip():
    x = np.array([1e-3, 0.5, 1.0, 2.0, 1e6])
    assert np.allclose(from_dB(to_dB(x)), x, rtol=1e-14)
    assert to_dB(100.0) == 20.0
    assert from_dB(-10) == 0.1
    assert np.allclose(from_dB([0, 3.0103]), [1.0, 2.0], rtol=1e-5)
