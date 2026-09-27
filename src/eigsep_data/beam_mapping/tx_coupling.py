"""Transmitter coupling through an HFSS-style beam.

The received power of one transmitter comb tooth is ``gain * |E_beam* . e_tx|^2``:
the antenna's complex far-field E vector toward the transmitter, projected on
the transmitter's field direction, both expressed in the antenna body frame
(see :mod:`~eigsep_data.beam_mapping.beam_rotations` for the frame and
rotation convention).

Beam fields are ``(nfreq, 3, npix)`` complex Cartesian components on a
HEALPix grid in the body frame, as in ``hfss_beam_maps/bowtie_beam.npz``.

Public API
----------
TransmitterGeometry
ground_heading
heading_between
normalize_fields
interpolate_fields
transmitter_frame
sample_fields
transmitter_coupling
transmitter_power
tooth_gains
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from .beam_rotations import enu_to_body, vector_to_spherical


@dataclass
class TransmitterGeometry:
    """Transmitter direction and polarization, both in ENU.

    ``heading_enu`` is the unit vector from the antenna to the transmitter.
    ``alpha_deg`` sets the transmitter's field direction for arm 0:
    ``(-sin a, cos a, 0)`` with ``a = alpha_deg`` (so alpha = 0 is North, and
    alpha increases toward West). Arm 1 is the orthogonal dipole,
    ``a = alpha_deg + 90``. This follows Dominic's HFSS workflow, where arm 0
    is the E2 drive (``E1 = 0, E2 = 1``) and arm 1 repeats that drive with the
    dipole pair rotated by 90 deg.
    """

    heading_enu: np.ndarray
    alpha_deg: float = 0.0

    def __post_init__(self):
        self.heading_enu = np.asarray(self.heading_enu, float)
        self.heading_enu = self.heading_enu / np.linalg.norm(self.heading_enu)

    def field_enu(self, arm=0):
        """Unit field direction of transmitter arm ``arm`` (0 or 1) in ENU."""
        a = np.deg2rad(self.alpha_deg + (90.0 if arm else 0.0))
        return np.array([-np.sin(a), np.cos(a), 0.0])


def ground_heading(east_m=0.0, north_m=0.0, height_m=92.5):
    """Unit antenna -> transmitter vector for a transmitter ``height_m`` below the antenna."""
    v = np.array([east_m, north_m, -height_m], dtype=float)
    return v / np.linalg.norm(v)


def heading_between(antenna_enu, transmitter_enu):
    """Unit antenna -> transmitter vector from two ENU positions (metres)."""
    v = np.asarray(transmitter_enu, float) - np.asarray(antenna_enu, float)
    return v / np.linalg.norm(v)


def normalize_fields(fields, freqs_mhz, reference_mhz=190.0):
    """Remove each slice's overall amplitude and common phase.

    With a free gain per tooth, only the beam's shape is identifiable. Each
    slice is scaled to unit spherical-mean total power and rotated in phase to
    align with the slice nearest ``reference_mhz``, so that slices can be
    interpolated in frequency. Returns ``(normalized, amplitude, phase)``.
    """
    fields = np.asarray(fields)
    freqs_mhz = np.asarray(freqs_mhz, float)
    amplitude = np.sqrt(np.mean(np.sum(np.abs(fields) ** 2, axis=1), axis=1))
    ref = fields[np.argmin(np.abs(freqs_mhz - reference_mhz))].ravel()
    phase = np.angle(fields.reshape(len(freqs_mhz), -1) @ ref.conj())
    return fields / (amplitude * np.exp(1j * phase))[:, None, None], amplitude, phase


def interpolate_fields(fields, freqs_mhz, target_mhz):
    """Linear interpolation of (normalized) fields to ``target_mhz``; no extrapolation."""
    return interp1d(np.asarray(freqs_mhz, float), fields, axis=0)(np.asarray(target_mhz, float))


def transmitter_frame(rotations, geometry, arms):
    """The transmitter as seen from the antenna body frame, for each pointing.

    Returns ``(theta, phi, rhat, e)``: the transmitter direction (radians and
    unit vector, ``(n, 3)``) and the transverse unit-less field direction of the
    arm driven at each pointing, ``(n, 3)``. ``arms`` is an int or ``(n,)``;
    pass ``arms=None`` to get both arms, ``e`` then ``(2, n, 3)``.
    """
    rotations = np.asarray(rotations, float)
    n = rotations.shape[0]
    rhat = enu_to_body(geometry.heading_enu, rotations)                       # (n, 3)
    theta, phi = vector_to_spherical(rhat)

    def transverse(e_enu):
        e = enu_to_body(e_enu, rotations)
        return e - np.sum(e * rhat, axis=-1, keepdims=True) * rhat

    if arms is None:
        e = np.stack([transverse(geometry.field_enu(a)) for a in (0, 1)])
    else:
        arms = np.broadcast_to(np.asarray(arms, int), (n,))
        e = transverse(np.where(arms[:, None] == 0, geometry.field_enu(0)[None, :],
                                geometry.field_enu(1)[None, :]))
    return theta, phi, rhat, e


def sample_fields(fields, theta, phi, interpolation='bilinear'):
    """Beam fields ``(nfreq, 3, npix)`` toward (theta, phi): ``(nfreq, n, 3)``."""
    import healpy as hp

    nside = hp.npix2nside(fields.shape[-1])
    if interpolation == 'bilinear':
        pix, weight = hp.get_interp_weights(nside, theta, phi)                # (4, n)
        return np.einsum('fckn,kn->fnc', fields[:, :, pix], weight)
    if interpolation == 'nearest':
        return np.moveaxis(fields[:, :, hp.ang2pix(nside, theta, phi)], 1, -1)
    raise ValueError("interpolation must be 'bilinear' or 'nearest'")


def transmitter_coupling(fields, rotations, geometry, arms, interpolation='bilinear'):
    """Complex coupling ``E_beam* . e_tx`` for every frequency slice and pointing.

    Parameters
    ----------
    fields : (nfreq, 3, npix) complex
        Beam E-field components in the body frame, HEALPix (RING).
    rotations : (n, 3, 3)
        Body -> ENU rotations (:func:`~eigsep_data.beam_mapping.beam_rotations.mount_rotation`).
    geometry : TransmitterGeometry
    arms : int or (n,) ints
        Transmitter arm for each pointing (0 or 1).
    interpolation : 'bilinear' or 'nearest'
        HEALPix sampling of the beam toward the transmitter.

    Returns
    -------
    coupling : (nfreq, n) complex
    theta, phi : (n,) radians, the transmitter direction in the body frame.
    """
    fields = np.asarray(fields)
    theta, phi, rhat, e = transmitter_frame(rotations, geometry, arms)
    beam = sample_fields(fields, theta, phi, interpolation)                   # (nfreq, n, 3)
    w = beam - np.sum(beam * rhat[None], axis=2, keepdims=True) * rhat[None]
    return np.einsum('fni,ni->fn', np.conj(w), e), theta, phi


def transmitter_power(fields, rotations, geometry, arms, interpolation='bilinear'):
    """``|transmitter_coupling|^2``: (nfreq, n) received power per unit gain."""
    coupling, _, _ = transmitter_coupling(fields, rotations, geometry, arms, interpolation)
    return np.abs(coupling) ** 2


def tooth_gains(power, data, good, train=None):
    """Non-negative least-squares gain per tooth: ``data[f] ~ gain[f] * power[f]``.

    ``power``, ``data``, ``good``: (nteeth, n). ``train``: (n,) bool selecting the
    samples used (default all). Masked samples do not contribute.
    """
    w = np.asarray(good, bool)
    if train is not None:
        w = w & np.asarray(train, bool)[None, :]
    num = np.sum(np.where(w, power * data, 0.0), axis=1)
    den = np.sum(np.where(w, power ** 2, 0.0), axis=1)
    return np.maximum(num / np.maximum(den, 1e-30), 0.0)
