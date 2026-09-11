"""A per-integration index over a directory of corr files.

Scanning reads only headers and metadata -- never spectra -- so a whole
deployment (5120 files, ~1.2M integrations) indexes in ~21-26 s with all
streams, ~5.5 s with rfswitch alone. Queries then run against the table
and only the selected rows are ever read from disk.

Row identity is the ``(file, row)`` pair. Ordering is by ``time_best``,
which is the header time when the file's clock was sane and a
filename-derived estimate otherwise; see :func:`scan_corr_file`.
"""

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .clock import filename_unix
from .metadata import MISSING, flatten_metadata

SCHEMA_VERSION = 1

#: Sidecar cache written next to the data. Gitignored -- derived from
#: untracked data and rebuilt whenever the inputs move.
CACHE_NAME = ".eigsep_index.h5"

#: A file whose last integration time differs from its filename by more
#: than this is flagged ``sync_consistent = False``. Deployment 5 splits
#: cleanly: good files land in [-1, +945] s, stale ones ~54 days out.
SYNC_TOLERANCE_S = 3600.0

#: Per-file header attrs broadcast onto every row, with the value used
#: when the attr is absent: MISSING for strings, NaN for numbers.
HEADER_ATTRS = {
    "run_tag": MISSING,
    "integration_time": np.nan,
    "adc_mux_sel": np.nan,
    "nchan": np.nan,
}


def _attr_value(value):
    """
    An h5py attribute as a plain Python scalar.

    Anything that is not a scalar -- an array- or list-valued attr --
    is stringified. Broadcasting a non-scalar with ``np.repeat`` would
    give a column of the wrong length, and the resulting ``ValueError``
    costs *every* integration in the file (see :meth:`MetadataIndex._scan`),
    which is far worse than one awkwardly rendered column.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def scan_corr_file(path, streams=None, filename_tz=None):
    """
    Index one corr file: one row per integration, spectra never read.

    This is the default ``scanner`` of :class:`MetadataIndex` and the
    seam for indexing other file kinds: an S11 index would supply its
    own scanner and reuse the cache, selection and summary machinery.
    A scanner must return a DataFrame with at least the columns
    ``file, row, time_best``.

    Identity columns, always present:

    ``file, row``
        Basename and row number: the durable identity of an integration.
    ``time, acc_cnt, sync_recovered``
        Header time, accumulation count, and
        ``time - acc_cnt * integration_time`` (the sync epoch the
        writer used; a bad clock shows up here directly).
    ``sync_consistent``
        ``|filename time - last header time| <= SYNC_TOLERANCE_S``. The
        filename is stamped from a correct clock at write time, so this
        catches the stale-``sync_time`` files -- 12.4 % of deployment 5,
        off by ~54 days -- but is blind to a whole-run clock offset,
        where filename and times would be wrong together. That is why
        it is not called ``time_ok``: it detects a clock *correction*
        during a run, not a clock that was wrong throughout.
    ``time_fname``
        ``filename time - (ntimes - 1 - row) * integration_time``: an
        estimate good to the write backlog (~16 min at worst, usually
        seconds), NaN when the name has no stamp.
    ``time_best``
        ``time`` when ``sync_consistent``, else ``time_fname``. Sort and
        window on this, never on ``time`` alone.

    Then root attrs (bare names; ``filter_corr_keys.py`` writes
    ``filter_phase``, ``filtered_keys`` and the mux-copy flags there),
    header attrs from :data:`HEADER_ATTRS` (overriding a root attr of the
    same name), ``data_keys`` (comma-joined), and the metadata columns
    from :func:`eigsep_data.metadata.flatten_metadata`.

    Parameters
    ----------
    path : str or Path
    streams : iterable of str, "all", or None
        Passed to ``flatten_metadata``.
    filename_tz : str or None
        Zone of a filename stamp with no ``Z`` suffix; see
        :func:`eigsep_data.clock.parse_filename_time`.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "header" not in h5 or "times" not in h5["header"]:
            raise KeyError("no header/times")
        times = np.asarray(h5["header"]["times"], dtype=float)
        ntimes = times.size
        if "acc_cnt" in h5["header"]:
            acc_cnt = np.asarray(h5["header"]["acc_cnt"], dtype=float)
        else:
            acc_cnt = np.full(ntimes, np.nan)
        root_attrs = {k: _attr_value(v) for k, v in h5.attrs.items()}
        header_attrs = {
            k: _attr_value(v) for k, v in h5["header"].attrs.items()
        }
        # Data keys come from the same handle: reopening the file
        # would double the scan cost.
        data_keys = sorted(h5["data"]) if "data" in h5 else []
        meta = {}
        if "metadata" in h5:
            for name in h5["metadata"]:
                raw = h5["metadata"][name][()]
                if isinstance(raw, bytes):
                    raw = raw.decode()
                try:
                    meta[name] = json.loads(raw)
                except (ValueError, TypeError):
                    continue

    integration_time = float(header_attrs.get("integration_time", np.nan))
    row = np.arange(ntimes, dtype=np.int32)
    fname_t = filename_unix(path.name, tz=filename_tz)
    lag = fname_t - times[-1] if ntimes else np.nan
    consistent = bool(np.isfinite(lag) and abs(lag) <= SYNC_TOLERANCE_S)
    time_fname = fname_t - (ntimes - 1 - row) * integration_time
    cols = {
        "file": np.repeat(path.name, ntimes),
        "row": row,
        "time": times,
        "acc_cnt": acc_cnt,
        "sync_recovered": times - acc_cnt * integration_time,
        "sync_consistent": np.repeat(consistent, ntimes),
        "time_fname": time_fname,
        "time_best": times if consistent else time_fname,
    }
    for key, value in root_attrs.items():
        if key not in cols:
            cols[key] = np.repeat(value, ntimes)
    for key, default in HEADER_ATTRS.items():
        cols[key] = np.repeat(header_attrs.get(key, default), ntimes)
    cols["data_keys"] = np.repeat(",".join(data_keys), ntimes)
    cols.update(flatten_metadata(meta, ntimes, streams=streams))
    return pd.DataFrame(cols)


def _finalise(table):
    """
    Fill gaps in object columns, normalise string dtypes, and sort.

    An attr present in one file and absent in another concatenates to an
    object column with NaN in the gaps; every such gap becomes
    ``MISSING``, so the column reads the same whether the table came
    from a scan or from the cache.

    Only columns whose values are *all* strings are then cast to ``str``.
    A boolean root attr (``mux_copy_0to1``) is left holding real
    ``True``/``False`` alongside ``MISSING``, because casting it would
    turn the flag into the strings ``"True"``/``"False"`` and make
    ``table.mux_copy_0to1 == True`` match nothing -- an empty result
    indistinguishable from an honest one. An attr present in every file
    keeps its native dtype and never reaches this branch at all.

    Sorting is on ``time_best``; ties (the ``-1`` suffix twins closed in
    the same second) break on filename then row.
    """
    for col in table.columns:
        if table[col].dtype != object:
            continue
        filled = table[col].fillna(MISSING)
        if filled.map(lambda v: isinstance(v, str)).all():
            filled = filled.astype(str)
        table[col] = filled
    return table.sort_values(
        ["time_best", "file", "row"], kind="stable"
    ).reset_index(drop=True)


class MetadataIndex:
    """
    One row per integration across a directory of corr files.

    Parameters
    ----------
    data_dir : str or Path
        Directory of corr h5 files.
    streams : iterable of str, "all", or None
        Metadata streams to carry; see
        :func:`eigsep_data.metadata.flatten_metadata`.
    patterns : tuple of str
        Filename globs to index. Dot-prefixed names and the cache file
        are always excluded, whatever the pattern.
    cache : bool
        Read and write the sidecar cache. No cache exists yet, so this
        only records the caller's intent.
    filename_tz : str or None
        Zone for filename stamps without a ``Z`` suffix (deployments
        1-4); ``None`` means Pacific.
    scanner : callable or None
        ``scanner(path, streams=..., filename_tz=...) -> DataFrame``
        replacing :func:`scan_corr_file` -- the hook for indexing a
        different file kind with the same cache and query machinery.

    Attributes
    ----------
    table : pandas.DataFrame
        Sorted by ``time_best``. Row *identity* is the ``(file, row)``
        pair, never the positional index, which shifts whenever the
        file set changes.
    from_cache : bool
        Whether the last :meth:`rebuild` was served from the sidecar.
    skipped : list of (str, str)
        ``(basename, reason)`` for every file the last :meth:`rebuild`
        could not index. Scanning warns and moves on, so this is the
        only way to tell a short table from a complete one without
        re-globbing the directory.
    """

    def __init__(
        self,
        data_dir,
        streams=None,
        patterns=("corr_*.h5",),
        cache=False,
        filename_tz=None,
        scanner=None,
    ):
        self.data_dir = Path(data_dir)
        self.streams = streams
        self.patterns = tuple(patterns)
        self.use_cache = cache
        self.filename_tz = filename_tz
        self.scanner = scanner or scan_corr_file
        self.table = None
        self.from_cache = False
        self.skipped = []
        self.rebuild()

    @property
    def cache_path(self):
        return self.data_dir / CACHE_NAME

    def _files(self):
        found = set()
        for pattern in self.patterns:
            found.update(self.data_dir.glob(pattern))
        # pathlib's glob matches dotfiles, so "*.h5" would hand back the
        # index's own cache and any editor/OS droppings next to it.
        return sorted(
            p
            for p in found
            if not p.name.startswith(".") and p.name != CACHE_NAME
        )

    def _scan(self, streams):
        frames = []
        skipped = []
        for path in self._files():
            try:
                frames.append(
                    self.scanner(
                        path, streams=streams, filename_tz=self.filename_tz
                    )
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                # One truncated or contract-violating file must not kill
                # a 5120-file scan, but the omission has to stay
                # answerable afterwards -- the table is simply shorter.
                skipped.append((path.name, str(exc)))
                warnings.warn(f"Skipping {path.name}: {exc}", stacklevel=2)
        self.skipped = skipped
        if not frames:
            raise FileNotFoundError(
                f"No indexable files matching {self.patterns} in "
                f"{self.data_dir}"
            )
        return _finalise(pd.concat(frames, ignore_index=True))

    def rebuild(self):
        """Scan every matching file and rebuild :attr:`table`."""
        self.from_cache = False
        self.table = self._scan(self.streams)
