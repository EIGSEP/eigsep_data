"""EIGSEP analysis package.

Submodules are imported lazily. ``import eigsep_data`` used to cost
~7 s because the package eagerly imported ``beam_sim``, which pulls
JAX, s2fft and jax_healpy; that was paid by every caller, including a
script that only wanted to read a flag mask. The names below still
resolve exactly as before -- ``from eigsep_data import MetadataIndex``,
``eigsep_data.hpm`` -- but nothing is imported until it is touched.

The JAX-backed modules (``hpm``, ``sph_fit``, ``beam_sim``,
``beam_mapping``) therefore raise on *access* rather than warning at
import time when their extras are missing, which is also when the
caller can do something about it.
"""

import importlib

__author__ = "Christian Hellum Bye"
__version__ = "0.0.1"

#: Submodule -> the extras it needs, for a useful message on failure.
_EXTRAS = {
    "hpm": "JAX and healjax",
    "sph_fit": "JAX and healjax",
    "beam_sim": "JAX and healjax",
    "beam_mapping": "JAX and healjax",
}

_SUBMODULES = {
    "beam_mapping",
    "beam_sim",
    "browse",
    "bundle",
    "clock",
    "data",
    "flagging",
    "hpm",
    "imu",
    "index",
    "metadata",
    "paths",
    "plot",
    "products",
    "quicklook",
    "rfi",
    "s11",
    "select_files",
    "sim",
    "sph_fit",
}

#: Public name -> the submodule that defines it.
_ATTRS = {
    "Bundle": "bundle",
    "Campaign": "bundle",
    "load_bundle": "bundle",
    "EigsepData": "data",
    "ImuCalibrator": "imu",
    "ImuDataset": "imu",
    "ImuSnapshot": "imu",
    "MetadataIndex": "index",
    "campaign_data_dir": "paths",
    "get_campaign_root": "paths",
    "set_campaign_root": "paths",
    "RawS11": "s11",
    "S11": "s11",
    "Selection": "index",
    "format_time": "clock",
    "to_unix_time": "clock",
}

__all__ = sorted(_SUBMODULES | set(_ATTRS) | {"__version__"})


def __getattr__(name):
    if name in _SUBMODULES:
        try:
            module = importlib.import_module(f".{name}", __name__)
        except ImportError as e:
            extras = _EXTRAS.get(name)
            if extras is None:
                raise
            raise ImportError(
                f"eigsep_data.{name} requires {extras}; install the "
                f"extras to use it ({e})"
            ) from e
        globals()[name] = module
        return module
    if name in _ATTRS:
        value = getattr(__getattr__(_ATTRS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
