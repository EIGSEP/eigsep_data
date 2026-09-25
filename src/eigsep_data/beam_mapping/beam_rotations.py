"""Receiver mount rotations for beam mapping.

The convention lives in :mod:`eigsep_base.rotations`, shared with
``eigsep_sim.Beam.top2body``; this module re-exports it as the
:mod:`eigsep_data.beam_mapping` entry point. In short, body -> ENU is

    v_ENU = Rz(psi) @ Rx(el) @ Rz(az) @ v_body

a roll mount (body: boresight +z, dipole arm +x; az rolls about the boresight;
el tips about the axle, 0 = zenith; psi = axle direction, deg counter-clockwise
from East, always supplied by the caller: 142.164 for Marjum 2026-07). See
:mod:`eigsep_base.rotations` for the full statement.

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

from eigsep_base.rotations import (
    body_to_enu,
    enu_to_body,
    mount_rotation,
    rx,
    rz,
    spherical_basis,
    vector_to_spherical,
)

__all__ = [
    "body_to_enu",
    "enu_to_body",
    "mount_rotation",
    "rx",
    "rz",
    "spherical_basis",
    "vector_to_spherical",
]
