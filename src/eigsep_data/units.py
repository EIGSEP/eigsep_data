"""Unit conversions shared across the package."""

import numpy as np


def to_dB(x):
    """Power ratio ``x`` in decibels: ``10 log10(x)``."""
    return 10 * np.log10(x)


def from_dB(db):
    """Power ratio of ``db`` decibels: ``10**(db / 10)``."""
    return 10 ** (np.asarray(db) / 10)
