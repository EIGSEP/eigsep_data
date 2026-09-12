"""Backward-compatibility shim — use healjax.maps and healjax.coord directly."""
import warnings as _w
_w.warn("eigsep_data.hpm is deprecated; import from healjax.maps / healjax.coord instead.",
        DeprecationWarning, stacklevel=2)

from healjax.maps import HPM  # noqa: F401
from healjax.maps.hpm import vec2ang, ang2pix, vec2pix  # noqa: F401
from healjax.interp import interpolate_map, rotate_interpolate_and_sum  # noqa: F401
from healjax.coord import xyz2thphi, angles_to_coord  # noqa: F401

# Spherical-harmonic fitting moved to eigsep_data.sph_fit
from .sph_fit import (  # noqa: F401
    build_real_design_matrix_from_angles,
    x_to_alm,
    fit_alms_from_maps,
    alms_to_filled_maps,
    sph_fit,
)
