"""Polarization beam mapping from alternating transmitter arms.

:class:`PolarizationBeamMapper` fits an off-axis transmitter heading and
polarization angle, then solves non-negative theta/phi beam gains at the
sampled directions from alternating transmitter arms.

Beam maps may be supplied as HEALPix arrays or as a callable returning
``(g_theta, g_phi)`` for ``(theta, phi)``. Pointing follows
:mod:`~eigsep_data.beam_mapping.beam_rotations`; ``psi_deg`` (the
elevation-axle direction) is a site constant supplied by the caller.

Public API
----------
PolarizationBeamMapper
"""

import numpy as np
from scipy.optimize import least_squares, nnls

from .beam_rotations import enu_to_body, mount_rotation, spherical_basis, vector_to_spherical
from .tx_coupling import TransmitterGeometry


class PolarizationBeamMapper:
    """Forward model and fitting tools for alternating transmitter spikes."""

    def __init__(self, beam_theta, beam_phi, psi_deg, sampler=None):
        self.beam_theta = beam_theta
        self.beam_phi = beam_phi
        self.psi_deg = float(psi_deg)
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

    def design(self, az_deg, el_deg, heading_enu, alpha_deg, arms):
        az_deg, el_deg = np.broadcast_arrays(np.ravel(az_deg), np.ravel(el_deg))
        arms = np.broadcast_to(np.ravel(np.asarray(arms, int)), az_deg.shape)
        R = mount_rotation(az_deg, el_deg, self.psi_deg)
        tx = TransmitterGeometry(heading_enu, alpha_deg)
        theta, phi = vector_to_spherical(enu_to_body(tx.heading_enu, R))
        _, thhat, phhat = spherical_basis(theta, phi)
        e_enu = np.where(arms[:, None] == 0, tx.field_enu(0)[None, :], tx.field_enu(1)[None, :])
        e = enu_to_body(e_enu, R)
        coeff = np.stack([np.sum(e * thhat, axis=1) ** 2, np.sum(e * phhat, axis=1) ** 2], axis=1)
        gth, gph = self.gains(theta, phi)
        return coeff, theta, phi, np.asarray(gth), np.asarray(gph)

    def predict(self, az_deg, el_deg, geometry, arms, scale=1.0, offset=0.0):
        coeff, theta, phi, gth, gph = self.design(
            az_deg, el_deg, geometry.heading_enu, geometry.alpha_deg, arms
        )
        return offset + scale * (gth * coeff[:, 0] + gph * coeff[:, 1]), (theta, phi)

    def fit_heading(self, az_deg, el_deg, arms, observed, initial,
                    fit_alpha=True, fit_scale=True, fit_offset=True):
        """Fit transmitter heading, polarization angle and amplitudes.

        The heading is parameterized by ENU angles: ``initial[0]`` is its
        direction counter-clockwise from East and ``initial[1]`` its elevation
        above the horizon (deg); ``initial[2]`` is alpha.
        """
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
        coeff, theta, phi, _, _ = self.design(az_deg, el_deg, geometry.heading_enu, geometry.alpha_deg, arms)
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
