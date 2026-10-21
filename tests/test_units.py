"""Unit conversions."""

import numpy as np

from eigsep_data.units import celsius_to_kelvin, from_dB, to_dB


def test_dB_round_trip():
    x = np.array([1e-3, 0.5, 1.0, 2.0, 1e6])
    assert np.allclose(from_dB(to_dB(x)), x, rtol=1e-14)
    assert to_dB(100.0) == 20.0
    assert from_dB(-10) == 0.1
    assert np.allclose(from_dB([0, 3.0103]), [1.0, 2.0], rtol=1e-5)


def test_celsius_to_kelvin():
    assert celsius_to_kelvin(0) == 273.15
    assert np.allclose(celsius_to_kelvin([-273.15, 25.0]), [0.0, 298.15])
