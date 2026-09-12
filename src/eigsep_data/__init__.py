__author__ = "Christian Hellum Bye"
__version__ = "0.0.1"

from .imu import ImuCalibrator, ImuSnapshot, ImuDataset
from .s11 import S11
from .data import EigsepData, to_unix_time
from . import plot
from . import rfi

try:
    from . import hpm
    from . import sph_fit
    from . import beam_sim
    from . import beam_fit
except ImportError:
    from warnings import warn

    warn(
        "hpm, sph_fit, beam_sim, and beam_fit modules require additional "
        "dependencies (JAX, healjax). Install them to use these modules.",
        ImportWarning,
    )
