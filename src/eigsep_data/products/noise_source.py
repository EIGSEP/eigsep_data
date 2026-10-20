"""The noise source's temperature, measured on the bench.

``derived/noise_source/<version>/solutions.npz`` with ``manifest.json`` and
``README.md`` beside it. Unlike the other products here it is not joined to
correlator rows: it is one characterization of the noise source, an input
to a calibration (``tcal`` from v0004 builds its ``t_ns_k`` from it). So it
is read with :func:`load`, not through ``load_bundle``.

The payload, all on one frequency axis ``freqs`` (MHz):

- ``t_non_k``, ``t_non_sigma_k``: the noise-source path's temperature at
  reference plane P when on, per channel, and its standard error;
  receiver-independent (``eigsep_cal.noise_source``);
- ``t_off_k``: the temperature taken for the path when off (the diode
  board's thermistor during the run), and ``excess_k = t_non_k - t_off_k``;
- ``good``: the channels the measurement keeps (its mask); ``t_non_k``
  and ``excess_k`` are NaN elsewhere;
- ``excess_smooth_k``: a low-order polynomial in frequency through
  ``excess_k`` over the good channels, defined on every channel, with its
  coefficients ``smooth_coef`` in ``x = (f - smooth_x0_mhz) /
  smooth_xscale_mhz`` (highest power first, as ``numpy.polyval``);
- ``gamma_ns``: the noise-source path's reflection at P, measured in the
  lab, on ``freqs``.

The manifest records the bench run, the code and the inputs' checksums; the
README says what the numbers are valid for.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..paths import get_campaign_root
from .base import read_manifest

_ARRAYS = ("freqs", "t_non_k", "t_non_sigma_k", "excess_k", "good",
           "excess_smooth_k", "smooth_coef", "gamma_ns")


def _base(root=None):
    root = Path(root) if root is not None else Path(get_campaign_root(required=True))
    return root / "derived" / "noise_source"


def versions(root=None):
    """Versions of the product on disk, sorted."""
    base = _base(root)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and (p / "solutions.npz").is_file())


@dataclass(frozen=True)
class NoiseSource:
    """One version of the noise-source characterization (module docstring)."""

    version: str
    freqs: np.ndarray
    t_non_k: np.ndarray
    t_non_sigma_k: np.ndarray
    t_off_k: float
    excess_k: np.ndarray
    good: np.ndarray
    excess_smooth_k: np.ndarray
    smooth_coef: np.ndarray
    smooth_x0_mhz: float
    smooth_xscale_mhz: float
    gamma_ns: np.ndarray
    manifest: dict = field(default_factory=dict)

    def on(self, freqs_mhz, smooth=True):
        """The excess (K) on the channels ``freqs_mhz``, which must be
        channels of this product's own axis: frequencies are matched, never
        regridded. ``smooth`` selects the polynomial (defined everywhere) or
        the per-channel values (NaN off the mask)."""
        freqs_mhz = np.asarray(freqs_mhz, dtype=float)
        idx = np.searchsorted(self.freqs, freqs_mhz)
        idx = np.clip(idx, 0, self.freqs.size - 1)
        if not np.allclose(self.freqs[idx], freqs_mhz, rtol=0, atol=1e-6):
            raise ValueError("frequencies are not channels of this product's axis")
        return (self.excess_smooth_k if smooth else self.excess_k)[idx]


def load(version, root=None):
    """Read ``derived/noise_source/<version>/`` under the campaign root."""
    d = _base(root) / version
    path = d / "solutions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"no solutions.npz under {d}")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in _ARRAYS + ("t_off_k", "smooth_x0_mhz", "smooth_xscale_mhz") if k not in z.files]
        if missing:
            raise KeyError(f"{path} lacks {missing}")
        arrays = {k: np.asarray(z[k]) for k in _ARRAYS}
        scalars = {k: float(z[k]) for k in ("t_off_k", "smooth_x0_mhz", "smooth_xscale_mhz")}
    arrays["good"] = arrays["good"].astype(bool)
    n = arrays["freqs"].size
    for k in _ARRAYS:
        if k != "smooth_coef" and arrays[k].shape != (n,):
            raise ValueError(f"{k} has shape {arrays[k].shape}, freqs has {n} channels")
    return NoiseSource(version=version, **arrays, **scalars,
                       manifest=read_manifest(d / "manifest.json"))
