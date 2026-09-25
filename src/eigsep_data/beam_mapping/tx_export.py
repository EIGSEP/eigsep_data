"""Writing a fitted empirical beam in the HFSS beam-map format.

The export is readable by :func:`eigsep_data.beam_sim.read_beam` (keys
``beam_cart``, ``gain_th``, ``gain_ph``, ``freqs``, ``nside``) plus metadata.
It covers every HFSS frequency and every tooth frequency. Inside the fitted band
the fields are the empirical model; outside it they are the normalized HFSS
beam. Every slice has its radial component removed and is scaled to unit
spherical-mean total power, so a tooth's gain changes by that slice's
normalization; :func:`export_beam` returns the adjusted gains.

Public API
----------
export_beam
"""

import numpy as np

from .beam_basis import real_spherical_harmonics
from .tx_coupling import interpolate_fields


def export_beam(path, hfss_freqs_mhz, hfss_fields_normalized, basis, coeff, lmax, band_mhz,
                tooth_freqs_mhz, gains, sample_count=None, metadata=None):
    """Write the empirical beam to ``path`` (npz) and return ``(export_gain, fields, freqs)``.

    Parameters
    ----------
    hfss_freqs_mhz, hfss_fields_normalized : the HFSS grid and its normalized fields
        (nfreq, 3, npix), used outside ``band_mhz``.
    basis, coeff, lmax : the fitted spectral basis, coefficients (nmodes, 3 nlm) and
        spherical-harmonic order.
    band_mhz : (lo, hi), where the empirical model replaces HFSS.
    tooth_freqs_mhz, gains : the fitted teeth and their gains (nteeth,).
    sample_count : (npix,), optional; how many samples fell in each pixel.
    metadata : dict of extra arrays or strings to store.

    ``export_gain`` is ``gains`` rescaled for the per-slice normalization, so that
    ``export_gain * |E_export . e|^2`` reproduces the fitted prediction.
    """
    import healpy as hp

    tooth_freqs_mhz = np.asarray(tooth_freqs_mhz, float)
    freqs = np.unique(np.r_[np.asarray(hfss_freqs_mhz, float), tooth_freqs_mhz])
    fields = interpolate_fields(hfss_fields_normalized, hfss_freqs_mhz, freqs)
    npix = fields.shape[-1]
    nside = hp.npix2nside(npix)
    empirical = (freqs >= band_mhz[0]) & (freqs <= band_mhz[1])
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    y = real_spherical_harmonics(theta, phi, lmax)
    fields[empirical] = (basis.evaluate(freqs[empirical]) @ coeff).reshape(empirical.sum(), 3, y.shape[1]) @ y.T
    r = np.array(hp.pix2vec(nside, np.arange(npix)))
    fields -= np.sum(fields * r[None], axis=1)[:, None] * r[None]
    norm_power = np.mean(np.sum(np.abs(fields) ** 2, axis=1), axis=1)
    fields /= np.sqrt(norm_power)[:, None, None]
    eth = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    eph = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    gain_th = np.abs(np.sum(fields * eth[None], axis=1)) ** 2
    gain_ph = np.abs(np.sum(fields * eph[None], axis=1)) ** 2
    export_gain = np.asarray(gains, float) * norm_power[np.searchsorted(freqs, tooth_freqs_mhz)]
    extra = {} if metadata is None else {k: np.asarray(v) for k, v in metadata.items()}
    if sample_count is not None:
        extra['sample_count'] = np.asarray(sample_count)
    np.savez_compressed(path, beam_cart=fields, gain_th=gain_th, gain_ph=gain_ph, freqs=freqs, nside=nside,
                        empirical_frequency_mask=empirical, fitted_frequency_limits_mhz=np.asarray(band_mhz),
                        spectral_basis=np.array(getattr(basis, 'kind', type(basis).__name__)),
                        normalization=np.array('unit spherical-average total power; independent tooth gains'),
                        **extra)
    if not (np.isfinite(fields).all() and np.allclose(np.mean(gain_th + gain_ph, axis=1), 1.0)):
        raise RuntimeError('exported beam failed its normalization checks')
    return export_gain, fields, freqs
