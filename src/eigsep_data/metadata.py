"""Flatten a corr file's metadata group into per-integration columns.

Every stream in a corr file carries exactly one entry per integration --
the writer guarantees it, back-filling a stream that appears mid-buffer
with ``[None] * counter`` and appending ``None`` for a stream missing
from a sample, explicitly "so indices align 1:1 with samples"
(``eigsep_observing.io.File._insert_sample``). Indexing metadata by row
number is therefore safe.

``rfswitch`` is the one stream that is a list of plain strings rather
than dicts: upstream routes it to ``_avg_rfswitch_metadata``, which
returns a bare state name. Special-casing it here matches the producer.

Three kinds of missing are kept distinct, because they mean different
things to an analyst:

===========================  ==================  ==============
case                         column              ``<stream>_ok``
===========================  ==================  ==============
stream absent from the file  ``MISSING`` / NaN   False
row entry is ``None``        ``MISSING`` / NaN   False
field is ``None``            ``MISSING`` / NaN   True
===========================  ==================  ==============

(``MISSING`` for string fields, NaN for numeric ones. A string column
never holds Python ``None``, so it survives the h5 cache unchanged.)

The first row of that table is the one this module cannot keep on its
own under ``streams="all"``, which takes the streams *this file* has
and so emits no column at all for one it lacks. The gap is closed a
level up, in :func:`eigsep_data.index._finalise`, which fills an
``*_ok`` gap with ``False``.

So ``UNKNOWN`` -- the producer asserting this integration is
contaminated by a switch transition, an error status, or a
mid-integration flip -- is never confused with ``MISSING``, which means
no information reached the writer at all.
"""

import numpy as np

MISSING = "MISSING"

#: Streams and fields carried by default. Flattening every field of
#: every stream would be ~130 columns (~600 MB at 1.2M rows), dominated
#: by adc_stats' 36 floats and the two tempctrl streams' 20 each. Widen
#: with ``streams=`` when you need them.
CURATED_FIELDS = {
    "motor": (
        "az_pos",
        "az_target_pos",
        "el_pos",
        "el_target_pos",
        "boot_id",
    ),
    "potmon": ("pot_az_angle", "pot_az_near_rail", "sp1_term_name"),
    "imu_el": ("accel_x", "accel_y", "accel_z", "el_deg"),
    "tempctrl_load": ("T_now", "active"),
    "tempctrl_lna": ("T_now", "active"),
    "rfswitch_therm": ("temp_therm0", "temp_therm1", "temp_therm2"),
    "system_current": ("current_a",),
    "lidar": ("distance_m",),
}

#: Streams whose value is a bare scalar rather than a dict.
SCALAR_STREAMS = ("rfswitch",)

#: Bookkeeping keys every sensor dict carries; never a column.
_HOUSEKEEPING = ("status", "sensor_name", "app_id")

#: Curated ``(stream, field)`` pairs whose value is a string, verified
#: against the real producer (``potmon``'s ``sp1_term_name`` is the only
#: one; ``boot_id`` is int, ``pot_az_near_rail`` is bool, everything
#: else numeric). Declared statically rather than sniffed from a file's
#: values, because sniffing cannot tell "string field, no string seen
#: this file" apart from "numeric field" when the field is absent or
#: ``None`` in every row -- and Task 4's concatenation plus Task 6's
#: cache round-trip both require a curated column's dtype to be the
#: same in every file, never a function of that file's content.
_STRING_FIELDS = frozenset(
    {
        ("potmon", "sp1_term_name"),
    }
)


def _as_float(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return np.nan


def flatten_metadata(metadata, ntimes, streams=None):
    """
    Flatten a corr file's metadata dict into per-integration columns.

    Parameters
    ----------
    metadata : dict
        The ``metadata`` mapping from ``eigsep_observing.io.read_hdf5``:
        ``{stream_name: [entry_per_integration, ...]}``. Entries are
        dicts, or bare strings for ``rfswitch``, or ``None``.
    ntimes : int
        Number of integrations in the file. Short or absent streams are
        padded to this length so every column is alignable with
        ``header["times"]``.
    streams : iterable of str, "all", or None
        Streams to carry. ``None`` uses :data:`CURATED_FIELDS` plus the
        scalar streams; ``"all"`` takes every stream and field present;
        an iterable names streams explicitly (curated fields for known
        streams, every field for unknown ones).

    Returns
    -------
    cols : dict[str, np.ndarray]
        Column name to length-*ntimes* array. ``rfswitch`` is a string
        array; dict-stream fields are float arrays, except fields
        declared in :data:`_STRING_FIELDS` or that carry any string
        value in this file, which are string arrays with ``MISSING``
        for gaps. Each stream also yields a boolean ``<stream>_ok``.
    """
    metadata = metadata or {}
    if streams == "all":
        wanted = {k: None for k in metadata}
    elif streams is None:
        wanted = dict(CURATED_FIELDS)
        for name in SCALAR_STREAMS:
            wanted.setdefault(name, None)
    else:
        wanted = {name: CURATED_FIELDS.get(name) for name in streams}

    cols = {}
    for name, fields in wanted.items():
        entries = list((metadata.get(name) or [])[:ntimes])
        entries += [None] * (ntimes - len(entries))
        ok = np.array(
            [
                e is not None
                and not (isinstance(e, dict) and e.get("status") == "error")
                for e in entries
            ],
            dtype=bool,
        )
        if name in SCALAR_STREAMS:
            cols[name] = np.array(
                [e if isinstance(e, str) else MISSING for e in entries],
                dtype=object,
            )
            cols[f"{name}_ok"] = ok
            continue

        if fields is None:
            fields = tuple(
                sorted(
                    {
                        k
                        for e in entries
                        if isinstance(e, dict)
                        for k in e
                        if k not in _HOUSEKEEPING
                    }
                )
            )
        for field in fields:
            raw = [
                e.get(field) if isinstance(e, dict) else None for e in entries
            ]
            is_string = (name, field) in _STRING_FIELDS or any(
                isinstance(v, str) for v in raw
            )
            if is_string:
                cols[f"{name}_{field}"] = np.array(
                    [v if isinstance(v, str) else MISSING for v in raw],
                    dtype=object,
                )
            else:
                cols[f"{name}_{field}"] = np.array(
                    [_as_float(v) for v in raw], dtype=float
                )
        cols[f"{name}_ok"] = ok
    return cols
