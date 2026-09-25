"""Receiver mount rotations for the rotating-antenna beam experiment.

This is the one pointing convention in :mod:`eigsep_data.beam_mapping`
(adopted 2026-09-25). A vector in the antenna body frame maps to local
topocentric ENU (x East, y North, z Up) as

    v_ENU = Rz(psi) @ Rx(el) @ Rz(az) @ v_body

* **Body frame** is the HFSS beam frame: boresight along +z, dipole arm along +x.
* **az** rolls the antenna about its boresight, right-handed, so at az = 0 the
  arm lies along the elevation axle.
* **el** tips the boresight about the elevation axle (body/axle x): el = 0 is
  zenith, +90 deg is the horizon on the -y side of the axle frame, and 180 deg
  is nadir.
* **psi** is the direction of the elevation axle, in degrees counter-clockwise
  from East. It is a site constant, passed by the caller; the package has no
  default. For Marjum 2026-07 the axle lies along the highline, psi = 142.164
  deg, so el = +90 deg points to compass bearing 37.836 deg.

All angles are in degrees. Functions broadcast over leading dimensions and
return ``(..., 3, 3)`` rotation matrices.

Public API
----------
rz
rx
mount_rotation
enu_to_body
body_to_enu
vector_to_spherical
spherical_basis
"""

import numpy as np


def rz(angle_deg):
    """Right-handed rotation about +z by ``angle_deg``: shape ``(..., 3, 3)``."""
    a = np.deg2rad(np.asarray(angle_deg, float))
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def rx(angle_deg):
    """Right-handed rotation about +x by ``angle_deg``: shape ``(..., 3, 3)``."""
    a = np.deg2rad(np.asarray(angle_deg, float))
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def mount_rotation(az_deg, el_deg, psi_deg):
    """Body -> ENU rotation ``Rz(psi) Rx(el) Rz(az)`` for each (az, el) pointing.

    ``az_deg`` and ``el_deg`` broadcast together; ``psi_deg`` is a scalar site
    constant (the elevation-axle direction, deg counter-clockwise from East).
    """
    az_deg, el_deg = np.broadcast_arrays(np.asarray(az_deg, float), np.asarray(el_deg, float))
    return rz(float(psi_deg)) @ rx(el_deg) @ rz(az_deg)


def enu_to_body(vectors_enu, rotation):
    """Express ENU vectors in the body frame: ``R^T v``.

    ``vectors_enu`` is ``(3,)`` (one vector for every rotation) or ``(..., 3)``
    matching the leading shape of ``rotation``.
    """
    v = np.asarray(vectors_enu, float)
    return np.einsum('...ji,...j->...i', rotation, v)


def body_to_enu(vectors_body, rotation):
    """Express body-frame vectors in ENU: ``R v``."""
    v = np.asarray(vectors_body, float)
    return np.einsum('...ij,...j->...i', rotation, v)


def vector_to_spherical(v):
    """Polar angle theta (from +z) and azimuth phi (from +x, in [0, 2 pi)) of ``v``, radians."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.arccos(np.clip(v[..., 2], -1, 1)), np.mod(np.arctan2(v[..., 1], v[..., 0]), 2 * np.pi)


def spherical_basis(theta, phi):
    """Unit vectors (r-hat, theta-hat, phi-hat) at (theta, phi), each ``(..., 3)``."""
    s, c = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    r = np.stack([cp * s, sp * s, c], axis=-1)
    th = np.stack([cp * c, sp * c, -s], axis=-1)
    ph = np.stack([-sp, cp, np.zeros_like(theta)], axis=-1)
    return r, th, ph
