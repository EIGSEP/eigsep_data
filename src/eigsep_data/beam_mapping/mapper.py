"""Polarization beam mapping from alternating transmitter arms.

:class:`PolarizationBeamMapper` fits an off-axis transmitter heading and
polarization angle, then solves non-negative theta/phi beam gains at the
sampled directions from alternating transmitter arms.

Beam maps may be supplied as HEALPix arrays or as a callable returning
``(g_theta, g_phi)`` for ``(theta, phi)``.

Public API
----------
PolarizationBeamMapper
"""

import numpy as np
from scipy.optimize import least_squares, nnls

from .geometry import (
    TransmitterGeometry,
    _sph_basis,
    rotation_matrix,
    vector_to_spherical,
)


class PolarizationBeamMapper:
    """Forward model and fitting tools for alternating transmitter spikes."""

    def __init__(self, beam_theta, beam_phi, sampler=None):
        self.beam_theta = beam_theta
        self.beam_phi = beam_phi
        self.sampler = sampler

    def gains(self, theta, phi):
        if self.sampler is not None:
            return tuple(np.asarray(x, float) for x in self.sampler(theta, phi))
        try:
            import healpy as hp
        except ImportError as exc:
            raise ImportError("healpy is required for HEALPix beam arrays") from exc
        px = hp.ang2pix(hp.npix2nside(np.asarray(self.beam_theta).size), theta, phi)
        return np.asarray(self.beam_theta)[px], np.asarray(self.beam_phi)[px]

    def design(self, az_deg, el_deg, heading_top, alpha_deg, arms):
        az_deg, el_deg = np.broadcast_arrays(az_deg, el_deg)
        arms = np.asarray(arms, int)
        coeff = np.zeros((az_deg.size, 2))
        theta = np.zeros(az_deg.size); phi = np.zeros(az_deg.size)
        for i, (az, el, arm) in enumerate(zip(az_deg.flat, el_deg.flat, arms.flat)):
            R = rotation_matrix(az, el)
            h = R.T @ heading_top
            theta[i], phi[i] = vector_to_spherical(h)
            _, thhat, phhat = _sph_basis(theta[i], phi[i])
            e = R.T @ TransmitterGeometry(heading_top, alpha_deg).field_top(arm)
            coeff[i] = [(e @ thhat)**2, (e @ phhat)**2]
        gth, gph = self.gains(theta, phi)
        return coeff, theta, phi, np.asarray(gth), np.asarray(gph)

    def predict(self, az_deg, el_deg, geometry, arms, scale=1.0, offset=0.0):
        coeff, theta, phi, gth, gph = self.design(
            az_deg, el_deg, geometry.heading_top, geometry.alpha_deg, arms
        )
        return offset + scale * (gth * coeff[:, 0] + gph * coeff[:, 1]), (theta, phi)

    def fit_heading(self, az_deg, el_deg, arms, observed, initial,
                    fit_alpha=True, fit_scale=True, fit_offset=True):
        """Fit transmitter heading (az/el), polarization angle and amplitudes."""
        y = np.asarray(observed, float)
        arms = np.asarray(arms, int)
        initial = np.asarray(initial, float)
        n = 2 + int(fit_alpha) + int(fit_scale) + int(fit_offset)
        x0 = np.zeros(n); x0[:2] = initial[:2]; j = 2
        if fit_alpha: x0[j] = initial[2] if initial.size > 2 else 0; j += 1
        if fit_scale: x0[j] = 1.0; j += 1
        if fit_offset: x0[j] = 0.0
        def unpack(x):
            heading = np.array([np.cos(np.deg2rad(x[0]))*np.cos(np.deg2rad(x[1])),
                                np.sin(np.deg2rad(x[0]))*np.cos(np.deg2rad(x[1])),
                                np.sin(np.deg2rad(x[1]))])
            j = 2; alpha = x[j] if fit_alpha else (initial[2] if initial.size > 2 else 0); j += int(fit_alpha)
            scale = x[j] if fit_scale else 1.; j += int(fit_scale)
            offset = x[j] if fit_offset else 0.
            return TransmitterGeometry(heading, alpha), scale, offset
        def residual(x):
            geom, scale, offset = unpack(x)
            pred, _ = self.predict(az_deg, el_deg, geom, arms, scale, offset)
            return (pred - y) / max(np.nanstd(y), 1e-12)
        result = least_squares(residual, x0)
        geom, scale, offset = unpack(result.x)
        result.geometry, result.scale, result.offset = geom, scale, offset
        return result

    def fit_sampled_gains(self, az_deg, el_deg, geometry, arms, observed):
        """Recover non-negative theta/phi gains independently at each pixel."""
        coeff, theta, phi, _, _ = self.design(az_deg, el_deg, geometry.heading_top, geometry.alpha_deg, arms)
        if self.sampler is None:
            import healpy as hp
            pixels = hp.ang2pix(hp.npix2nside(np.asarray(self.beam_theta).size), theta, phi)
        else:
            # Callable samplers cannot define a stable pixel identity.
            pixels = np.arange(theta.size)
        y = np.asarray(observed, float)
        out = {}
        for px in np.unique(pixels):
            m = pixels == px
            out[int(px)] = nnls(coeff[m], y[m])[0]
        return out, theta, phi
