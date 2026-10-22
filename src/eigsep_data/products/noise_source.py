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

import numpy as np

from ..paths import get_campaign_root
from .base import derived_dir, read_manifest, versions_in

_ARRAYS = ("freqs", "t_non_k", "t_non_sigma_k", "excess_k", "good",
           "excess_smooth_k", "smooth_coef", "gamma_ns")


def _base(root=None):
    root = root if root is not None else get_campaign_root(required=True)
    return derived_dir(root, "noise_source")


def versions(root=None):
    """Versions of the product on disk, sorted."""
    return versions_in(_base(root))


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


def write(root, version, *, freqs, t_non_k, t_non_sigma_k, t_off_k, good,
          gamma_ns, provenance, validation=None, smooth_deg=3,
          smooth_x0_mhz=150.0, smooth_xscale_mhz=100.0, overwrite=False):
    """Write ``derived/noise_source/<version>/`` under ``root``.

    Takes the measured quantities (module docstring), derives the excess
    ``t_non_k - t_off_k`` and its smooth fit (a polynomial of degree
    ``smooth_deg`` in ``(f - smooth_x0_mhz) / smooth_xscale_mhz``, weighted by
    ``1 / t_non_sigma_k**2`` over the ``good`` channels), and writes
    ``solutions.npz``, ``manifest.json`` (``provenance``, which should name
    the bench run, the inputs with their checksums and the code with its
    commits, plus a description of the contents) and, if given,
    ``validation.json``. The README beside them is written by hand. Refuses
    to overwrite an existing version unless ``overwrite``. Returns the
    version's directory."""
    import json

    d = _base(root) / version
    if (d / "solutions.npz").exists() and not overwrite:
        raise FileExistsError(f"{d} already holds a solutions.npz")
    freqs = np.asarray(freqs, dtype=float)
    good = np.asarray(good, dtype=bool) & np.isfinite(t_non_k) & np.isfinite(t_non_sigma_k)
    t_non_k = np.where(good, np.asarray(t_non_k, float), np.nan)
    t_non_sigma_k = np.where(good, np.asarray(t_non_sigma_k, float), np.nan)
    excess = t_non_k - float(t_off_k)
    x = (freqs - smooth_x0_mhz) / smooth_xscale_mhz
    coef = np.polyfit(x[good], excess[good], smooth_deg, w=1.0 / t_non_sigma_k[good])
    smooth = np.polyval(coef, x)
    resid = excess[good] - smooth[good]
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "solutions.npz", freqs=freqs, t_non_k=t_non_k, t_non_sigma_k=t_non_sigma_k,
             t_off_k=float(t_off_k), excess_k=excess, good=good, excess_smooth_k=smooth,
             smooth_coef=coef, smooth_x0_mhz=float(smooth_x0_mhz),
             smooth_xscale_mhz=float(smooth_xscale_mhz),
             gamma_ns=np.asarray(gamma_ns, dtype=complex))
    manifest = {
        "provenance": dict(provenance, product="noise_source", version=version),
        "contents": {
            "freqs": "MHz, the correlator channels",
            "t_non_k": "noise-source path's temperature at P when on, K; NaN off `good`",
            "t_non_sigma_k": "standard error of t_non_k, K",
            "t_off_k": "temperature taken for the path when off (diode-board thermistor), K",
            "excess_k": "t_non_k - t_off_k, K",
            "good": "channels kept by the measurement's mask",
            "excess_smooth_k": "smooth fit of excess_k on every channel, K",
            "smooth_coef": "its polynomial coefficients, highest power first",
            "gamma_ns": "noise-source path's reflection at P (lab), on freqs",
        },
        "smooth_fit": {
            "degree": int(smooth_deg), "x": "(f - smooth_x0_mhz) / smooth_xscale_mhz",
            "weights": "1 / t_non_sigma_k**2 over good channels",
            "n_channels": int(good.sum()),
            "residual_rms_k": float(np.sqrt(np.mean(resid ** 2))),
            "residual_max_abs_k": float(np.max(np.abs(resid))),
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    if validation is not None:
        (d / "validation.json").write_text(json.dumps(validation, indent=2, default=float) + "\n")
    return d
