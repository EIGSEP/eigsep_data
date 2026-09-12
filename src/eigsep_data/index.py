"""A per-integration index over a directory of corr files.

Scanning reads only headers and metadata -- never spectra. Measured on
the 5124-file deployment 5 (1.23 M integrations, the nine curated
streams): ~64 s for a cold scan, then ~5 s per session to rebuild from
the sidecar cache it leaves behind. Queries run against the table and
only the selected rows are ever read from disk.

Row identity is the ``(file, row)`` pair. Ordering is by ``time_best``,
which is the header time when the file's clock was sane and a
filename-derived estimate otherwise; see :func:`scan_corr_file`.
"""

import contextlib
import fnmatch
import hashlib
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .clock import filename_unix, to_unix_time
from .metadata import (
    CURATED_FIELDS,
    MISSING,
    SCALAR_STREAMS,
    flatten_metadata,
)

#: Version of what a scan *produces*. **Bump this whenever a scan would
#: give different columns or different values for identical inputs** --
#: a field added to :data:`eigsep_data.metadata.CURATED_FIELDS`, a
#: changed ``time_best`` or ``sync_recovered`` formula, a new identity
#: column. It is the only part of the cache fingerprint that can say so:
#: the manifest, the patterns, the zone and the scanner's qualified name
#: all stay identical when the scanner's *behaviour* changes, so without
#: a bump every existing sidecar keeps validating and keeps serving the
#: old table, in every later session, with no warning.
SCHEMA_VERSION = 1

#: Sidecar cache written next to the data. Gitignored -- derived from
#: untracked data and rebuilt whenever the inputs move.
CACHE_NAME = ".eigsep_index.h5"

#: Key under which :func:`scan_corr_file` reports metadata streams it
#: could not parse, in the returned frame's ``DataFrame.attrs``, as
#: ``[(stream, reason), ...]``. A scanner is a plain function with no
#: handle on the index and the frame is the only thing it hands back;
#: ``attrs`` carries the note without costing a column of a million
#: identical values. A scanner that sets nothing reports nothing.
LOST_STREAMS_ATTR = "lost_streams"

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
        Carrying ``attrs[LOST_STREAMS_ATTR]``: the ``(stream, reason)``
        pairs whose JSON would not parse. Each one warns as well, and
        :class:`MetadataIndex` collects them into
        :attr:`MetadataIndex.lost_streams`.
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
        lost = []
        if "metadata" in h5:
            for name in h5["metadata"]:
                raw = h5["metadata"][name][()]
                if isinstance(raw, bytes):
                    raw = raw.decode()
                try:
                    meta[name] = json.loads(raw)
                except (ValueError, TypeError) as exc:
                    # Every column of this stream now reads MISSING,
                    # which says "no information reached the writer" --
                    # but information did reach it and was destroyed on
                    # read. Nothing in the table can tell the two apart,
                    # so the loss is reported here and recorded on the
                    # index (see MetadataIndex.lost_streams).
                    lost.append((name, str(exc)))
                    warnings.warn(
                        f"{path.name}: metadata stream {name!r} could "
                        f"not be parsed, its columns read MISSING: "
                        f"{exc}",
                        stacklevel=2,
                    )

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
    frame = pd.DataFrame(cols)
    frame.attrs[LOST_STREAMS_ATTR] = lost
    return frame


def _is_boolean(values):
    """Whether an object column's non-gap values are all booleans."""
    return pd.api.types.infer_dtype(values.dropna(), skipna=True) == "boolean"


def _finalise(table):
    """
    Fill gaps in object columns, normalise string dtypes, and sort.

    An attr present in one file and absent in another concatenates to an
    object column with NaN in the gaps; every such gap becomes
    ``MISSING``, so the column reads the same whether the table came
    from a scan or from the cache.

    The ``<stream>_ok`` flags are the one exception and take ``False``
    instead, because that is what the contract says a stream absent
    from a file reads (see :mod:`eigsep_data.metadata`). A gap reaches
    them only under ``streams="all"``, which enumerates the streams
    *that file* carries and so emits no column at all for a file that
    has none; ``MISSING`` there would make ``select(potmon_ok=False)``
    return no rows for exactly the files it is asking about, and turn a
    one-byte boolean into an object column besides. Root attrs share
    this namespace, so the values are checked as well as the name:
    filling a *string* column with ``False`` and casting it would read
    ``True`` in every row.

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
        if col.endswith("_ok") and _is_boolean(table[col]):
            # eq, not fillna(False).astype(bool): a gap is NaN, which
            # is not equal to True, and fillna on an object column
            # downcasts with a pandas FutureWarning.
            table[col] = table[col].eq(True)
            continue
        filled = table[col].fillna(MISSING)
        if filled.map(lambda v: isinstance(v, str)).all():
            filled = filled.astype(str)
        table[col] = filled
    return table.sort_values(
        ["time_best", "file", "row"], kind="stable"
    ).reset_index(drop=True)


def _stream_key(streams):
    """
    Canonical form of a stream request: ``"all"`` or a sorted list.

    A bare string other than ``"all"`` is one stream name, not an
    iterable of characters. ``"all"`` being a documented bare string
    makes ``streams="rfswitch"`` a natural thing to try, and
    ``sorted(set("rfswitch"))`` would ask for eight streams named after
    letters -- a table of ``c_ok``, ``f_ok``, ``h_ok`` columns and not
    one real value in any of them.
    """
    if streams == "all":
        return "all"
    if streams is None:
        return sorted(set(CURATED_FIELDS) | set(SCALAR_STREAMS))
    if isinstance(streams, str):
        return [streams]
    return sorted(set(streams))


def _covers(have, want):
    """
    Whether a cache holding *have* can answer a request for *want*.

    Both are canonical stream keys. ``"all"`` covers everything and is
    covered by nothing else: a named set cannot promise to contain every
    stream the files happen to carry.

    The promise is about the *request*, not the columns. ``"all"`` means
    every stream the files happen to carry, so an ``"all"`` cache over
    files that carry no ``lidar`` stream covers a later curated or
    ``("lidar",)`` request and hands back a table with no
    ``lidar_distance_m`` column -- where a cold scan for that request
    would have given one, padded with NaN, because a named stream is
    always materialised. Accepted: a caller that needs the column to
    exist checks ``name in index.table.columns``, and gets it by
    rescanning for exactly its own request
    (:meth:`MetadataIndex.rebuild` with ``force=True``, or
    ``cache=False``).
    """
    if have == "all":
        return True
    if want == "all":
        return False
    return set(want) <= set(have)


def _union(a, b):
    """Canonical stream key covering both *a* and *b* (*b* may be
    ``None``, meaning nothing was cached)."""
    if b is None:
        return a
    if a == "all" or b == "all":
        return "all"
    return sorted(set(a) | set(b))


#: Encoding of an ``object`` column, keyed by what
#: :func:`pandas.api.types.infer_dtype` says about the values that are
#: not the ``MISSING`` sentinel. Strings and booleans are the two kinds
#: the scan produces: a gap upcasts a string column to object (it
#: already is one) and a boolean column to object, while a numeric
#: column just takes a NaN and stays numeric. Anything else is
#: unrepresentable; see :func:`_encode_column`.
_OBJECT_KINDS = {
    "string": "str",
    "empty": "str",
    "boolean": "bool",
}


def _encode_column(name, values):
    """
    One column as ``(array, kind)`` ready for :mod:`h5py`.

    A non-object column goes to disk in its own dtype (``"native"``),
    so a float column's NaN gaps stay NaN and an ``int32`` column does
    not come back widened. An object column is written as fixed-width
    utf-8 bytes, and *kind* names what its non-``MISSING`` values are
    converted back to on read: ``"str"`` or ``"bool"``.

    That last distinction is the point of this function. A boolean root
    attr written for only some files (``mux_copy_0to1``) lands in an
    object column holding real ``True``, real ``False`` and the string
    ``MISSING``; encoding object columns as bytes wholesale would hand
    back the *strings* ``"True"``/``"False"``, after which
    ``table.mux_copy_0to1 == True`` matches nothing -- an empty answer
    indistinguishable from an honest one.

    Raises
    ------
    TypeError
        The column mixes types the cache cannot restore exactly. A lossy
        cache would serve a different table in every later session, so
        the caller declines to write one at all.
    """
    if values.dtype != object:
        return values, "native"
    live = values[values != MISSING]
    inferred = pd.api.types.infer_dtype(live, skipna=True)
    kind = _OBJECT_KINDS.get(inferred)
    if kind is None:
        raise TypeError(f"column {name!r} holds {inferred} values")
    return np.char.encode(values.astype(str), "utf-8"), kind


def _decode_column(values, kind):
    """Invert :func:`_encode_column`, gaps and element types included."""
    if kind == "native":
        return values
    text = np.char.decode(values, "utf-8")
    # astype(object) on a numpy string array yields real Python str,
    # not np.str_, so a decoded column is indistinguishable from a
    # scanned one.
    out = text.astype(object)
    if kind == "bool":
        live = text != MISSING
        out[live] = (text[live] == "True").astype(object)
    return out


class MetadataIndex:
    """
    One row per integration across a directory of corr files.

    Parameters
    ----------
    data_dir : str or Path
        Directory of corr h5 files.
    streams : iterable of str, "all", or None
        Metadata streams to carry; see
        :func:`eigsep_data.metadata.flatten_metadata`. Kept in canonical
        form (``"all"`` or a sorted list, ``None`` spelled out as the
        curated set), so an iterable is read exactly once however many
        times :meth:`rebuild` runs.
    patterns : tuple of str
        Filename globs to index. Dot-prefixed names and the cache file
        are always excluded, whatever the pattern.
    cache : bool
        Read and write the sidecar cache (:data:`CACHE_NAME`, next to
        the data); see :meth:`rebuild` for when it is trusted. ``False``
        neither reads nor writes it, leaving any existing sidecar alone.
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
    cached_streams : list of str or "all"
        The stream set :attr:`table` actually carries, which is a
        superset of the requested one whenever the cache had more; see
        :meth:`rebuild`.
    skipped : list of (str, str)
        ``(basename, reason)`` for every file the last :meth:`rebuild`
        could not index. Scanning warns and moves on, so this is the
        only way to tell a short table from a complete one without
        re-globbing the directory. A cache hit restores the list the
        cached table was built with and re-warns, because the file
        manifest of a partial scan validates just as well as a complete
        one and would otherwise serve the short table in silence.
    lost_streams : list of (str, str, str)
        ``(basename, stream, reason)`` for every metadata stream that
        was present but unparseable. Its columns read ``MISSING`` for
        that file, which otherwise means "no information reached the
        writer" -- here it did, and was destroyed on read. Kept apart
        from :attr:`skipped` deliberately: these files *are* in the
        table, at full length, so folding them in would break the one
        thing ``skipped`` is for (telling a short table from a complete
        one). Restored and re-warned on a cache hit for the same reason
        ``skipped`` is.
    """

    def __init__(
        self,
        data_dir,
        streams=None,
        patterns=("corr_*.h5",),
        cache=True,
        filename_tz=None,
        scanner=None,
    ):
        self.data_dir = Path(data_dir)
        # Canonicalised once, not per rebuild: a one-shot iterator would
        # otherwise be consumed by the first build and canonicalise to
        # the empty set on the next, scanning no streams at all and
        # saying nothing about it.
        self.streams = _stream_key(streams)
        self.patterns = tuple(patterns)
        self.use_cache = cache
        self.filename_tz = filename_tz
        self.scanner = scanner or scan_corr_file
        self.table = None
        self.from_cache = False
        self.cached_streams = None
        self.skipped = []
        self.lost_streams = []
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
        lost = []
        for path in self._files():
            try:
                frame = self.scanner(
                    path, streams=streams, filename_tz=self.filename_tz
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                # One truncated or contract-violating file must not kill
                # a 5120-file scan, but the omission has to stay
                # answerable afterwards -- the table is simply shorter.
                skipped.append((path.name, str(exc)))
                warnings.warn(f"Skipping {path.name}: {exc}", stacklevel=2)
                continue
            frames.append(frame)
            # The scanner has already warned about each of these; this
            # is what makes them answerable later, and cacheable. A
            # scanner that reports nothing contributes nothing.
            lost.extend(
                (path.name, stream, reason)
                for stream, reason in frame.attrs.get(LOST_STREAMS_ATTR, ())
            )
        self.skipped = skipped
        self.lost_streams = lost
        if not frames:
            raise FileNotFoundError(
                f"No indexable files matching {self.patterns} in "
                f"{self.data_dir}"
            )
        return _finalise(pd.concat(frames, ignore_index=True))

    def _fingerprint(self):
        """
        Identity of the inputs the table was built from -- the stream
        set is deliberately *not* part of it (see :meth:`rebuild`).

        Everything that changes what a scan would produce is in here:
        the schema version, the patterns, the filename zone, the
        scanner, and the ``(basename, size, mtime_ns)`` manifest. The
        sidecar itself is never in the manifest (:meth:`_files` excludes
        it), or writing it would change the fingerprint it was written
        under and no build would ever hit the cache.
        """
        manifest = []
        for path in self._files():
            stat = path.stat()
            manifest.append((path.name, stat.st_size, stat.st_mtime_ns))
        payload = json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "patterns": sorted(self.patterns),
                "filename_tz": self.filename_tz,
                "scanner": f"{self.scanner.__module__}."
                f"{self.scanner.__qualname__}",
                "manifest": manifest,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _read_cache(self, fingerprint, wanted):
        """
        ``(table, streams, skipped, lost)`` from the sidecar, or
        ``None``s.

        *table* is ``None`` when there is nothing usable to read, but
        *streams* still comes back when the fingerprint matched and only
        the column set fell short, so :meth:`rebuild` can rescan for the
        union instead of dropping what was already cached.

        A sidecar that cannot be *opened* -- truncated, garbage, not HDF5
        at all -- is treated as absent without comment, since the scan
        produces the same table anyway. One that opens but does not
        decode warns instead: that is a defect in this codec rather than
        a stale cache, and it would otherwise show up only as a
        ~64 s rescan in every session for ever.

        Note that the schema version is *not* checked here. Invalidation
        is entirely by fingerprint, which :data:`SCHEMA_VERSION` is part
        of, so a schema bump never reaches this point -- the fingerprints
        simply do not match.
        """
        if not self.cache_path.exists():
            return None, None, None, None
        try:
            with h5py.File(self.cache_path, "r") as h5:
                if h5.attrs.get("fingerprint") != fingerprint:
                    return None, None, None, None
                have = json.loads(h5.attrs["streams"])
                if not _covers(have, wanted):
                    return None, have, None, None
                skipped = [
                    tuple(item)
                    for item in json.loads(h5["skipped"][()].decode())
                ]
                # Absent from sidecars written before the record
                # existed. Those tables are unchanged by it, and an
                # empty list is the only answer such a file can give;
                # they lose it for good on their next rebuild.
                lost = [
                    tuple(item)
                    for item in json.loads(
                        h5["lost_streams"][()].decode()
                        if "lost_streams" in h5
                        else "[]"
                    )
                ]
                group = h5["columns"]
                table = pd.DataFrame(
                    {
                        name: _decode_column(
                            group[name][()], group[name].attrs["kind"]
                        )
                        for name in json.loads(h5.attrs["columns"])
                    }
                )
        except OSError:
            return None, None, None, None
        except (KeyError, ValueError, TypeError) as exc:
            warnings.warn(
                f"Could not read index cache, rescanning: {exc}", stacklevel=3
            )
            return None, None, None, None
        return table, have, skipped, lost

    def _write_cache(self, fingerprint, streams):
        """
        Write :attr:`table` to the sidecar, warning rather than raising.

        The two stages are handled separately because only one of them
        owns the file. Encoding happens first and touches nothing, so a
        column the cache cannot represent leaves the path exactly as it
        was -- which matters when a sidecar is already there: under the
        widening policy a request can rescan, fail to encode the *wider*
        table, and must not take the narrower cache down with it. Past
        the open, ``h5py`` has truncated the path, so whatever is there
        is this attempt's own half-written sidecar and the handler
        removes it rather than leave several hundred MB of rubble next to
        the data. A read-only directory fails at the open with nothing
        created, and removing nothing is not an error.

        Every failure here costs a rescan and nothing else, so none of
        them is worth raising over: :attr:`table` is already built and
        correct. The schema version is not written into the file -- it is
        inside *fingerprint*, and a second copy nobody reads would only
        suggest a read-side check that does not exist.
        """
        try:
            encoded = [
                (name,) + _encode_column(name, self.table[name].to_numpy())
                for name in self.table.columns
            ]
        except (TypeError, ValueError) as exc:
            warnings.warn(f"Could not write index cache: {exc}", stacklevel=3)
            return
        try:
            with h5py.File(self.cache_path, "w") as h5:
                h5.attrs["fingerprint"] = fingerprint
                h5.attrs["streams"] = json.dumps(streams)
                h5.attrs["columns"] = json.dumps(list(self.table.columns))
                # A dataset, not an attr: a bad batch of files would put
                # thousands of skip reasons past the 64 KB attr limit.
                h5.create_dataset(
                    "skipped", data=np.bytes_(json.dumps(self.skipped))
                )
                h5.create_dataset(
                    "lost_streams",
                    data=np.bytes_(json.dumps(self.lost_streams)),
                )
                group = h5.create_group("columns")
                for name, values, kind in encoded:
                    dataset = group.create_dataset(name, data=values)
                    dataset.attrs["kind"] = kind
        except (OSError, TypeError, ValueError) as exc:
            # Removing it may itself be refused (the read-only directory
            # that failed the open in the first place), in which case
            # there was nothing there to remove anyway.
            with contextlib.suppress(OSError):
                self.cache_path.unlink(missing_ok=True)
            warnings.warn(f"Could not write index cache: {exc}", stacklevel=3)

    def rebuild(self, force=False):
        """
        Build :attr:`table`, reusing the sidecar cache when it matches.

        The cache is keyed on the file manifest (name, size, mtime), the
        schema version, the patterns, the filename zone and the scanner
        -- see :meth:`_fingerprint`. The stream set is stored alongside
        rather than in the key: a request is served from the cache when
        its streams are a subset of what was cached, and otherwise
        rescans for the union and overwrites, so widening and narrowing
        never thrash each other.

        ``patterns`` *is* in the key, and there is only one sidecar per
        directory, so two callers that glob differently over the same
        directory overwrite each other's cache and each rescan. The beam
        extraction (``file_patterns=("*.h5", "*.hdf5")``) is the one
        such caller; alternating it with a plain index over the same
        directory rebuilds every time. Accepted deliberately: the
        pattern set changes which files are seen, and a cache must not
        answer for files it never looked at.

        Parameters
        ----------
        force : bool
            Ignore any existing cache and rescan.
        """
        fingerprint = self._fingerprint()
        wanted = self.streams  # already canonical; see __init__
        self.from_cache = False
        have = None
        if self.use_cache and not force:
            cached, have, skipped, lost = self._read_cache(fingerprint, wanted)
            if cached is not None:
                self.table = cached
                self.cached_streams = have
                self.skipped = skipped
                self.lost_streams = lost
                self.from_cache = True
                for name, reason in skipped:
                    warnings.warn(
                        f"Cached index omits {name}: {reason}", stacklevel=2
                    )
                for name, stream, reason in lost:
                    warnings.warn(
                        f"Cached index could not parse stream {stream!r} "
                        f"of {name}, its columns read MISSING: {reason}",
                        stacklevel=2,
                    )
                return
        streams = _union(wanted, have)
        self.table = self._scan(streams)
        self.cached_streams = streams
        if self.use_cache:
            self._write_cache(fingerprint, streams)

    def select(self, **kwargs):
        """
        Choose integrations by metadata.

        Keyword arguments are column filters: a scalar matches equality,
        a list matches membership. Three named selectors:

        ``files=``
            A glob string, a *list* of globs, or a 2-*tuple*
            ``(lo, hi)``: an inclusive lexical range on basenames (the
            idiom the motor-scan notebook used, and the right selector
            when header times cannot be trusted).
        ``time=(lo, hi)``
            Half-open range on ``time_best``; bounds go through
            :func:`eigsep_data.clock.to_unix_time`.
        ``where=``
            A callable taking the DataFrame and returning a boolean
            mask.

        Returns
        -------
        Selection
        """
        return _apply_filters(self, self.table, [], **kwargs)


class Selection:
    """
    A set of integrations chosen from a :class:`MetadataIndex`.

    Holds only metadata rows -- no spectra. Load them with :meth:`load`
    or :meth:`eigsep_data.EigsepData.from_selection`.
    """

    def __init__(self, index, meta, provenance=None):
        self.index = index
        self.meta = meta
        self.provenance = list(provenance or [])

    @property
    def nrows(self):
        return len(self.meta)

    @property
    def files(self):
        """Filenames contributing rows, in ``time_best`` order."""
        return list(dict.fromkeys(self.meta.file))

    def file_counts(self):
        """Rows per file, as a Series in selection order."""
        return self.meta.groupby("file", sort=False).size()

    def select(self, **kwargs):
        """Narrow this selection further; same arguments as
        :meth:`MetadataIndex.select`."""
        return _apply_filters(self.index, self.meta, self.provenance, **kwargs)

    def visits(self, gap_s=600):
        """
        Group rows into contiguous visits separated by *gap_s*.

        Returns one id per row: ``0, 1, 2, ...`` in ``time_best`` order
        for rows that have a ``time_best``, and ``-1`` for rows that do
        not (an inconsistent clock and no usable filename estimate; see
        :meth:`summary`). A row with no time is never reported as a
        member of a real visit -- there is no time by which to place it
        beside one -- so a caller averaging a visit gets only rows it
        knows the time of, and the ``-1`` group is an explicit decision
        rather than padding on the last visit.

        Gaps are measured between rows that have times, so a run of
        timeless rows never joins two visits or chains one into the
        next.
        """
        times = self.meta.time_best.to_numpy(dtype=float)
        visits = np.full(times.size, -1, dtype=int)
        timed = np.isfinite(times)
        known = times[timed]
        if known.size:
            ids = np.zeros(known.size, dtype=int)
            ids[1:] = np.cumsum(np.diff(known) > gap_s)
            visits[timed] = ids
        return visits

    def summary(self):
        """Human-readable account of what each filter removed."""
        lines = [f"{len(self.index.table)} rows indexed"]
        for name, before, after in self.provenance:
            lines.append(
                f"  {name}: {before} -> {after} ({before - after} removed)"
            )
        lines.append(
            f"{self.nrows} rows selected from {len(self.files)} files"
        )
        if "sync_consistent" in self.meta:
            n_bad = int((~self.meta.sync_consistent.astype(bool)).sum())
            lines.append(
                f"  {n_bad} rows have sync_consistent=False; their "
                "time_best is a filename estimate, good to the write "
                "backlog (~16 min), not to the integration"
            )
        if "time_best" in self.meta:
            # A name with no stamp leaves nothing to fall back on, so
            # these rows sort to the end and no time= window can ever
            # reach them. Deliberate, but never silent.
            n_no_time = int(self.meta.time_best.isna().sum())
            lines.append(
                f"  {n_no_time} rows have no time_best (an "
                "inconsistent clock and no usable filename estimate); "
                "they sort last and fall outside every time= window"
            )
        return "\n".join(lines)

    def load(self, keys=None, time_avg=1, missing="raise"):
        """Read the spectra for these integrations; see
        :meth:`eigsep_data.EigsepData.from_selection`."""
        # Lazy on purpose: data.py imports from this module for
        # from_selection, so a module-scope import here would close
        # the cycle.
        from .data import EigsepData

        return EigsepData.from_selection(
            self, keys=keys, time_avg=time_avg, missing=missing
        )


def _is_range(spec):
    """A 2-tuple of plain names is a lexical range; anything holding a
    glob character is a pair of patterns, so ``("a*", "b*")`` still
    means what a caller expects."""
    return (
        isinstance(spec, tuple)
        and len(spec) == 2
        and all(isinstance(s, str) for s in spec)
        and not any(ch in s for s in spec for ch in "*?[")
    )


def _match_files(names, spec):
    """Boolean mask over the unique *names* for a ``files=`` spec."""
    if _is_range(spec):
        lo, hi = spec
        return np.array([lo <= n <= hi for n in names], dtype=bool)
    patterns = [spec] if isinstance(spec, str) else list(spec)
    return np.array(
        [any(fnmatch.fnmatch(n, p) for p in patterns) for n in names],
        dtype=bool,
    )


def _apply_filters(
    index,
    table,
    provenance,
    *,
    files=None,
    time=None,
    where=None,
    **filters,
):
    """Apply one round of filters, recording what each one removed."""
    provenance = list(provenance)
    current = table

    def step(name, mask):
        nonlocal current
        before = len(current)
        current = current[np.asarray(mask, dtype=bool)]
        provenance.append((name, before, len(current)))

    if files is not None:
        # Match the few thousand unique names, not the million rows.
        names = current.file.unique()
        keep = names[_match_files(names, files)]
        step(f"files={files!r}", current.file.isin(keep))
    if time is not None:
        lo, hi = (to_unix_time(t) for t in time)
        step(
            f"time=({lo}, {hi})",
            (current.time_best >= lo) & (current.time_best < hi),
        )
    for column, value in filters.items():
        if column not in current.columns:
            raise KeyError(
                f"No column {column!r} in the index. Available: "
                f"{sorted(current.columns)}"
            )
        if isinstance(value, (list, tuple, set)):
            mask = current[column].isin(list(value))
        else:
            mask = current[column] == value
        step(f"{column}={value!r}", mask)
    if where is not None:
        step("where=<callable>", where(current))

    return Selection(index, current, provenance)
