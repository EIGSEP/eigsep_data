"""Unit conversions shared across the package."""

import numpy as np
from scipy.constants import zero_Celsius

#: 0 degC in kelvin.
ZERO_CELSIUS_K = zero_Celsius


def celsius_to_kelvin(t):
    """Temperature ``t`` in degC, in kelvin."""
    return np.asarray(t, dtype=float) + ZERO_CELSIUS_K


def to_dB(x):
    """Power ratio ``x`` in decibels: ``10 log10(x)``."""
    return 10 * np.log10(x)


def from_dB(db):
    """Power ratio of ``db`` decibels: ``10**(db / 10)``."""
    return 10 ** (np.asarray(db) / 10)
