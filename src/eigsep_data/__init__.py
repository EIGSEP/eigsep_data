__author__ = "Christian Hellum Bye"
__version__ = "0.0.1"

from .imu import ImuCalibrator, ImuSnapshot, ImuDataset
from .s11 import S11
from .data import EigsepData
from . import plot

try:
    from . import hpm
    from . import sim
except ImportError:
    from warnings import warn

    warn(
        "hpm and sim modules require additional dependencies. Install them to"
        "use these modules.",
        ImportWarning,
    )
