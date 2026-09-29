"""Backward-compatibility shim — use eigsep_cal directly."""
import warnings as _w
_w.warn("eigsep_data.s11 is deprecated; import from eigsep_cal.s11 instead.",
        DeprecationWarning, stacklevel=2)

from eigsep_cal import S11, RawS11
