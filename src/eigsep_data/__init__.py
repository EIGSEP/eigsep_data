__author__ = "Christian Hellum Bye"
__version__ = "0.0.1"

from .imu import ImuCalibrator, ImuSnapshot, ImuDataset
from .s11 import S11, RawS11
from .clock import to_unix_time, format_time
from .data import EigsepData
from .index import MetadataIndex, Selection
from . import metadata
from . import clock
from . import plot
from . import rfi

try:
    from . import hpm
    from . import sim
    from . import beam_sim
    from . import beam_fit
except ImportError:
    from warnings import warn

    warn(
        "hpm, sim, beam_sim, and beam_fit modules require additional "
        "dependencies (JAX, jax_healpy). Install them to use these modules.",
        ImportWarning,
    )
