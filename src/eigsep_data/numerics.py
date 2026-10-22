"""Small array helpers shared across the package."""

import numpy as np


def weighted_mean(x, w, axis=None):
    """Weighted mean of ``x`` along ``axis`` (all of it by default).

    A sample whose value or weight is not finite, or whose weight is not
    positive, carries no weight; where none is left the result is NaN. ``x``
    and ``w`` broadcast against each other. Returns an array (0-d for
    ``axis=None``)."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    num = np.sum(np.where(ok, w * x, 0.0), axis=axis)
    den = np.sum(np.where(ok, w, 0.0), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def interp_complex(x, xp, fp):
    """``numpy.interp`` of complex ``fp`` (real and imaginary parts apart),
    NaN outside ``xp``. ``fp`` is one curve on ``xp`` or a stack of them
    (rows); the result has the same number of rows, on ``x``."""
    fp = np.asarray(fp)

    def one(f):
        return np.interp(
            x, xp, f.real, left=np.nan, right=np.nan
        ) + 1j * np.interp(x, xp, f.imag, left=np.nan, right=np.nan)

    if fp.ndim == 1:
        return one(fp)
    return np.array([one(f) for f in fp]).reshape(len(fp), np.size(x))
