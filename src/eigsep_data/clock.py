"""Time handling for correlator data: epoch in, rendering out.

**The implementation now lives in :mod:`eigsep_base.time`** (moved
2026-09-17, Aaron's decision). This module grew independently on the
metadata-index branch and ended up carrying a byte-identical second
copy of ``to_unix_time``, re-creating the duplication the B12
consolidation had just removed. Rather than delete the module and churn
every call site, it stays as the name this package imports through --
one implementation, one import path here.

Every timestamp a corr file stores is Unix epoch, and nothing in this
stack shifts it. Deployment 5's absolute epoch was verified against an
external anchor (the switch-board thermistor's diurnal cycle).
Timezones enter in exactly two places: rendering an epoch for a human
(:func:`format_time`), and parsing a *filename* stamp, whose zone
depends on the deployment (:func:`parse_filename_time`).
"""

from eigsep_base.time import (
    FILENAME_TZ,
    LEGACY_FILENAME_TZ,
    filename_unix,
    format_time,
    parse_filename_time,
    to_unix_time,
)

__all__ = [
    "FILENAME_TZ",
    "LEGACY_FILENAME_TZ",
    "filename_unix",
    "format_time",
    "parse_filename_time",
    "to_unix_time",
]
