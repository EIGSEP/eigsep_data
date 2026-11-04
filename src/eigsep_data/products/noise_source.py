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
- ``excess_smooth_k``: a smooth fit of ``excess_k`` over the good
  channels. By default (``smooth_model`` ``"poly"``) a low-order
  polynomial in frequency, defined on every channel, with its coefficients
  ``smooth_coef`` in ``x = (f - smooth_x0_mhz) / smooth_xscale_mhz``
  (highest power first, as ``numpy.polyval``). With ``smooth_model``
  ``"dpss"`` (from v0002) a DPSS basis beside a polynomial in ``ln f``
  (:mod:`eigsep_data.smooth_basis`), NaN outside the fitted channels'
  range and over excluded bands, with ``smooth_coef`` the basis
  coefficients (the manifest's ``smooth_fit`` says how to rebuild it);
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
    smooth_model: str = "poly"

    def on(self, freqs_mhz, smooth=True):
        """The excess (K) on the channels ``freqs_mhz``, which must be
        channels of this product's own axis: frequencies are matched, never
        regridded. ``smooth`` selects the smooth fit (NaN where it is not
        defined: nowhere for ``smooth_model == "poly"``) or the per-channel
        values (NaN off the mask)."""
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
        if "smooth_model" in z.files:
            scalars["smooth_model"] = str(z["smooth_model"])
    arrays["good"] = arrays["good"].astype(bool)
    n = arrays["freqs"].size
    for k in _ARRAYS:
        if k != "smooth_coef" and arrays[k].shape != (n,):
            raise ValueError(f"{k} has shape {arrays[k].shape}, freqs has {n} channels")
    return NoiseSource(version=version, **arrays, **scalars,
                       manifest=read_manifest(d / "manifest.json"))


def smooth_dpss(freqs, excess, sigma, good, smooth):
    """The ``"dpss"`` smooth fit of ``excess`` (module docstring).

    ``smooth`` holds ``half_width_ns`` (the DPSS delay half-width),
    ``poly_deg`` and ``f0_mhz`` (the polynomial in ``ln(f / f0)`` beside
    it) and optionally ``exclude_mhz``, a list of ``[lo, hi]`` bands left
    out of the fit and set to NaN, and ``cv_block_mhz`` and ``cv_offsets``
    for the held-out score (:func:`eigsep_data.smooth_basis.block_cv`).
    Weights are ``1 / sigma**2`` over the good channels outside the
    excluded bands; the basis spans their range, outside which the result
    is NaN. Returns ``(smooth, coef, f_lo, f_hi, info)``."""
    from .. import smooth_basis as sb

    f = np.asarray(freqs, float)
    excluded = np.zeros(f.size, bool)
    for lo, hi in smooth.get("exclude_mhz", ()):
        excluded |= (f >= lo) & (f <= hi)
    fit = good & ~excluded
    f_lo, f_hi = float(f[fit].min()), float(f[fit].max())

    def design(ff):
        return sb.dpss_design(
            ff,
            f_lo,
            f_hi,
            smooth["half_width_ns"],
            df_mhz=float(np.median(np.diff(np.sort(f)))),
            poly_deg=smooth["poly_deg"],
            f0_mhz=smooth.get("f0_mhz", 120.0),
        )

    d = design(f)
    w = np.where(fit, 1.0 / np.where(fit, sigma, 1.0) ** 2, 0.0)
    coef, pred, _ = sb.wls_fit(d, excess, w)
    inside = (f >= f_lo) & (f <= f_hi) & ~excluded
    out = np.where(inside, pred, np.nan)
    resid = (excess - pred)[fit]
    info = {
        "model": "dpss",
        "half_width_ns": float(smooth["half_width_ns"]),
        "poly_deg": int(smooth["poly_deg"]),
        "f0_mhz": float(smooth.get("f0_mhz", 120.0)),
        "basis": "eigsep_data.smooth_basis.dpss_design(f, f_lo, f_hi, "
        "half_width_ns, df_mhz=channel spacing, poly_deg, f0_mhz); "
        "excess_smooth_k = basis @ smooth_coef",
        "f_lo_mhz": f_lo,
        "f_hi_mhz": f_hi,
        "ncol": int(d.shape[1]),
        "exclude_mhz": [list(map(float, b)) for b in
                        smooth.get("exclude_mhz", ())],
        "defined": "NaN outside [f_lo, f_hi] and over exclude_mhz",
        "chi2_per_dof": float(
            np.sum((resid / sigma[fit]) ** 2) / (fit.sum() - d.shape[1])
        ),
    }
    if "cv_block_mhz" in smooth:
        b = float(smooth["cv_block_mhz"])
        n_off = int(smooth.get("cv_offsets", 1))
        offs = tuple(np.arange(n_off) * b / n_off)
        info["held_out"] = {
            "block_mhz": b,
            "offsets": n_off,
            "this_model": sb.block_cv(
                f, excess, sigma, fit, design, b, offs
            ),
            "cubic_in_f (v0001 form)": sb.block_cv(
                f,
                excess,
                sigma,
                fit,
                lambda ff: np.vander((ff - 150.0) / 100.0, 4),
                b,
                offs,
            ),
        }
    return out, coef, f_lo, f_hi, info


def write(root, version, *, freqs, t_non_k, t_non_sigma_k, t_off_k, good,
          gamma_ns, provenance, validation=None, smooth_deg=3,
          smooth_x0_mhz=150.0, smooth_xscale_mhz=100.0, overwrite=False,
          smooth=None):
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
    version's directory.

    ``smooth``, a dict with ``"model": "dpss"`` and the parameters of
    :func:`smooth_dpss`, replaces the polynomial (``smooth_deg`` and the
    ``x`` scaling are then unused)."""
    import json

    d = _base(root) / version
    if (d / "solutions.npz").exists() and not overwrite:
        raise FileExistsError(f"{d} already holds a solutions.npz")
    freqs = np.asarray(freqs, dtype=float)
    good = np.asarray(good, dtype=bool) & np.isfinite(t_non_k) & np.isfinite(t_non_sigma_k)
    t_non_k = np.where(good, np.asarray(t_non_k, float), np.nan)
    t_non_sigma_k = np.where(good, np.asarray(t_non_sigma_k, float), np.nan)
    excess = t_non_k - float(t_off_k)
    extra = {}
    if smooth is None:
        x = (freqs - smooth_x0_mhz) / smooth_xscale_mhz
        coef = np.polyfit(
            x[good], excess[good], smooth_deg, w=1.0 / t_non_sigma_k[good]
        )
        smoothed = np.polyval(coef, x)
        resid = excess[good] - smoothed[good]
        smooth_fit = {
            "degree": int(smooth_deg),
            "x": "(f - smooth_x0_mhz) / smooth_xscale_mhz",
            "weights": "1 / t_non_sigma_k**2 over good channels",
            "n_channels": int(good.sum()),
        }
    elif smooth.get("model") == "dpss":
        smoothed, coef, f_lo, f_hi, smooth_fit = smooth_dpss(
            freqs, excess, t_non_sigma_k, good, smooth
        )
        smooth_x0_mhz, smooth_xscale_mhz = f_lo, f_hi - f_lo
        fitted = np.isfinite(smoothed) & good
        resid = excess[fitted] - smoothed[fitted]
        smooth_fit.update(
            weights="1 / t_non_sigma_k**2 over good channels outside "
            "exclude_mhz",
            n_channels=int(fitted.sum()),
            smooth_x0_mhz="f_lo_mhz", smooth_xscale_mhz="f_hi_mhz - f_lo_mhz",
        )
        extra["smooth_model"] = np.str_("dpss")
    else:
        raise ValueError(f"unknown smooth model {smooth.get('model')!r}")
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "solutions.npz", freqs=freqs, t_non_k=t_non_k, t_non_sigma_k=t_non_sigma_k,
             t_off_k=float(t_off_k), excess_k=excess, good=good,
             excess_smooth_k=smoothed,
             smooth_coef=coef, smooth_x0_mhz=float(smooth_x0_mhz),
             smooth_xscale_mhz=float(smooth_xscale_mhz),
             gamma_ns=np.asarray(gamma_ns, dtype=complex), **extra)
    manifest = {
        "provenance": dict(provenance, product="noise_source", version=version),
        "contents": {
            "freqs": "MHz, the correlator channels",
            "t_non_k": "noise-source path's temperature at P when on, K; NaN off `good`",
            "t_non_sigma_k": "standard error of t_non_k, K",
            "t_off_k": "temperature taken for the path when off (diode-board thermistor), K",
            "excess_k": "t_non_k - t_off_k, K",
            "good": "channels kept by the measurement's mask",
            "excess_smooth_k": "smooth fit of excess_k on every channel, K"
            if smooth is None else
            "smooth fit of excess_k, K; NaN where not defined (smooth_fit)",
            "smooth_coef": "its polynomial coefficients, highest power first"
            if smooth is None else "its basis coefficients (smooth_fit)",
            "gamma_ns": "noise-source path's reflection at P (lab), on freqs",
        },
        "smooth_fit": {
            **smooth_fit,
            "residual_rms_k": float(np.sqrt(np.mean(resid ** 2))),
            "residual_max_abs_k": float(np.max(np.abs(resid))),
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    if validation is not None:
        (d / "validation.json").write_text(json.dumps(validation, indent=2, default=float) + "\n")
    return d
