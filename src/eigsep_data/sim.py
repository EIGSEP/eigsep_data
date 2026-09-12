"""Backward-compatibility shim — use eigsep_sim directly."""
import warnings as _w
_w.warn("eigsep_data.sim is deprecated; use eigsep_sim instead.",
        DeprecationWarning, stacklevel=2)
