# Per-Integration Metadata Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `eigsep_data` select correlator integrations by RF switch state — and by any other per-integration metadata — before reading any spectra.

**Architecture:** A cheap scan of headers and metadata builds one row per integration into a pandas DataFrame (`MetadataIndex`), cached to a self-invalidating sidecar. Queries against that table produce a `Selection` of `(file, row)` pairs; only then are the chosen spectra read into an `EigsepData` carrying its aligned metadata. The four existing hand-rolled scan-parse-select implementations collapse onto this core.

**Tech Stack:** Python ≥3.11, numpy, pandas (declared but currently unused in `src/`), h5py, pytest. Reads via `eigsep_observing.io`.

**Spec:** `docs/superpowers/specs/2026-09-11-switch-state-index-design.md`

## Global Constraints

- **Line length 79** (`black` line-length 79, `flake8` max-line-length 79, `extend-ignore = E203, W503`).
- **No new runtime dependencies.** The cache is written with `h5py`; `pandas<3` and `h5py` are already declared. `ipywidgets` is added only to the existing `vis` extra.
- **Never mutate epoch.** Loaders return `header["times"]` unmodified. Timezone handling happens at render time only.
- **Never gate by a fixed row count.** Switch-state selection keys on the state string. The writer's transition guard is `ceil(0.5 s / integration_time)` samples — 2 rows at 0.2684 s, 1 at 0.5369 s, both present in deployment 5.
- **`None` and `UNKNOWN` are different and must stay so.** `UNKNOWN` = producer asserts contamination. `None`/`MISSING` = no information reached the writer.
- **`rfswitch` is an open category, never an enum.** 17 values observed; an unseen value must not raise.
- **Row identity is `(file basename, row)`**, never a position in a concatenated array.
- Tests use `pytest`, live in `tests/`, and follow the existing convention of class-per-unit with docstrings stating *why* the case matters (see `tests/test_data.py`).
- Run `pytest` from the repo root using `.venv/bin/python -m pytest`.

## Deviation from the spec

The spec proposes a new `[interactive]` extra for `ipywidgets`. This plan
instead adds it to the **existing `vis` extra**, which already carries
`jupyterlab` and `ipympl` — the same audience, and the repo's established
pattern. One fewer extra to explain.

## File Structure

| file | responsibility | status |
|---|---|---|
| `tests/conftest.py` | synthetic corr-file writer + directory fixtures | create |
| `src/eigsep_data/metadata.py` | flatten one file's metadata group → named columns; missing policy | create |
| `src/eigsep_data/index.py` | `MetadataIndex` (scan, cache, invalidate), `Selection` (query) | create |
| `src/eigsep_data/browse.py` | `StateBrowser` interactive flipbook over a `Selection` | create |
| `src/eigsep_data/data.py` | `EigsepData` + `.meta`, `from_selection`, clock helpers; beam extractor becomes a wrapper | modify |
| `src/eigsep_data/quicklook.py` | carry `.meta`, report state breakdown, `--state` gate | modify |
| `src/eigsep_data/__init__.py` | export the new surface | modify |
| `pyproject.toml` | `ipywidgets` into the `vis` extra | modify |
| `notebooks/christian/deployment5/rf_state_tools.py` | superseded | delete |
| `notebooks/christian/deployment5/rfswitch_index.npz` | superseded (tracked in git) | delete |

`data.py` imports `index.py`; never the reverse. `index.py` imports `metadata.py`.

---

### Task 1: Synthetic corr-file fixture

Everything downstream is tested against files written by the **real producer**, so the contract under test is `eigsep_observing`'s, not a mock of it.

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_conftest.py`

**Interfaces:**
- Consumes: `eigsep_observing.io.write_hdf5`
- Produces: `write_corr_file(path, **kw) -> Path`; pytest fixtures `corr_dir`, `RFSWITCH_LADDER`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conftest.py
"""The fixture must produce files the real reader accepts."""

import numpy as np
from eigsep_observing.io import read_hdf5

from conftest import RFSWITCH_LADDER, write_corr_file


class TestWriteCorrFile:
    def test_roundtrips_through_the_real_reader(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        data, header, meta = read_hdf5(p)
        assert sorted(data) == ["0", "4"]
        assert data["0"].shape == (60, 1024)
        assert len(header["times"]) == 60
        assert header["run_tag"] == "panda_observe"

    def test_none_survives_the_json_roundtrip(self, tmp_path):
        # A None row means "no reading reached the writer" and is
        # distinct from "UNKNOWN". If the fixture cannot express it,
        # none of the missing-data tests below are meaningful.
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        _, _, meta = read_hdf5(p)
        sw = meta["rfswitch"]
        assert sw[0] == "RFANT"
        assert sw[20] == "UNKNOWN"
        assert sw[45] is None

    def test_times_follow_the_writer_formula(self, tmp_path):
        # times = acc_cnt * integration_time + sync_time. The index
        # recovers sync from this, so the fixture must obey it.
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=30,
            integration_time=0.2684,
            sync_time=1.7843e9,
        )
        _, header, _ = read_hdf5(p)
        expected = header["acc_cnt"] * 0.2684 + 1.7843e9
        np.testing.assert_allclose(header["times"], expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_conftest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'conftest'` or `ImportError: cannot import name 'write_corr_file'`

- [ ] **Step 3: Write the fixture**

```python
# tests/conftest.py
"""Synthetic correlator files written by the real producer.

Tests build genuine corr files with ``eigsep_observing.io.write_hdf5``
rather than mocking the layout, so what they pin is the producer's
contract. The rfswitch ladder mirrors
``eigsep_observing._test_fixtures.CORR_METADATA``: a steady state, the
writer's UNKNOWN transition guard, a second steady state, a dropout
run of None, and recovery.
"""

import numpy as np
import pytest
from eigsep_observing.io import write_hdf5

NCHAN = 1024

RFSWITCH_LADDER = (
    ["RFANT"] * 20
    + ["UNKNOWN"] * 5
    + ["RFNOFF"] * 20
    + [None] * 5
    + ["RFNOFF"] * 10
)


def write_corr_file(
    path,
    *,
    ntimes=60,
    sync_time=1.7843e9,
    acc_cnt0=0,
    integration_time=0.5369,
    run_tag="panda_observe",
    keys=("0", "4"),
    rfswitch=None,
    streams=("motor",),
    seed=0,
):
    """Write one synthetic corr file and return its path.

    ``rfswitch=None`` omits the stream entirely, which is how the
    ~1842 deployment-5 files with no switch metadata look.
    """
    rng = np.random.default_rng(seed)
    acc_cnt = np.arange(acc_cnt0, acc_cnt0 + ntimes, dtype=np.int64)
    times = acc_cnt * integration_time + sync_time
    data = {
        k: rng.integers(1, 1000, size=(ntimes, NCHAN), dtype=np.int32)
        for k in keys
    }
    header = {
        "times": times,
        "acc_cnt": acc_cnt,
        "freqs": np.linspace(0, 250, NCHAN, endpoint=False),
        "integration_time": float(integration_time),
        "run_tag": run_tag,
        "adc_mux_sel": 5,
        "nchan": NCHAN,
    }
    md = {}
    if rfswitch is not None:
        md["rfswitch"] = list(rfswitch)[:ntimes]
    if "motor" in streams:
        md["motor"] = [
            {
                "status": "update",
                "sensor_name": "motor",
                "app_id": 1,
                "boot_id": 7,
                "az_pos": float(i),
                "az_target_pos": float(i),
                "el_pos": 0.0,
                "el_target_pos": 0.0,
            }
            for i in range(ntimes)
        ]
    write_hdf5(path, data, header, metadata=md or None)
    return path


@pytest.fixture
def corr_dir(tmp_path):
    """Three files: full ladder, no rfswitch stream, faster cadence."""
    write_corr_file(
        tmp_path / "corr_20260717_150041Z.h5",
        ntimes=60,
        rfswitch=RFSWITCH_LADDER,
        sync_time=1.7843e9,
    )
    write_corr_file(
        tmp_path / "corr_20260717_151041Z.h5",
        ntimes=60,
        rfswitch=None,
        sync_time=1.7843e9 + 600,
    )
    write_corr_file(
        tmp_path / "corr_20260717_152041Z.h5",
        ntimes=60,
        rfswitch=["RFAMB"] * 30 + ["RFNON"] * 30,
        integration_time=0.2684,
        sync_time=1.7843e9 + 1200,
        run_tag="motor_scan",
    )
    return tmp_path
```

Add to `pyproject.toml` so `from conftest import ...` resolves:

```toml
[tool.pytest.ini_options]
pythonpath = ["tests"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_conftest.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_conftest.py pyproject.toml
git commit -m "test: synthetic corr files written by the real producer"
```

---

### Task 2: `metadata.py` — flatten one file's streams to columns

Pure transformation: dicts in, arrays out. No file I/O, so it is testable in isolation and fast.

**Files:**
- Create: `src/eigsep_data/metadata.py`
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `CURATED_FIELDS: dict[str, tuple[str, ...]]`
  - `MISSING = "MISSING"`
  - `flatten_metadata(metadata: dict, ntimes: int, streams=None) -> dict[str, np.ndarray]`
  - column names are `"rfswitch"` (bare) and `"<stream>_<field>"`, plus `"<stream>_ok"` per stream

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metadata.py
"""Tests for eigsep_data.metadata."""

import numpy as np

from eigsep_data import metadata as md


class TestFlattenMetadata:
    """The three kinds of missing must stay distinguishable."""

    def test_absent_stream_is_missing_not_unknown(self):
        # The ~1842 deployment-5 files with no rfswitch metadata must
        # not masquerade as the producer's UNKNOWN contamination flag.
        cols = md.flatten_metadata({}, ntimes=4)
        assert list(cols["rfswitch"]) == [md.MISSING] * 4
        assert not cols["rfswitch_ok"].any()

    def test_none_row_is_not_a_state(self):
        # None means no reading reached the writer for that window.
        cols = md.flatten_metadata(
            {"rfswitch": ["RFANT", None, "RFANT", None]}, ntimes=4
        )
        assert list(cols["rfswitch"]) == [
            "RFANT", md.MISSING, "RFANT", md.MISSING,
        ]
        assert list(cols["rfswitch_ok"]) == [True, False, True, False]

    def test_unknown_is_preserved_verbatim(self):
        cols = md.flatten_metadata(
            {"rfswitch": ["RFANT", "UNKNOWN"]}, ntimes=2
        )
        assert list(cols["rfswitch"]) == ["RFANT", "UNKNOWN"]
        # UNKNOWN is information: the row is present, just contaminated.
        assert list(cols["rfswitch_ok"]) == [True, True]

    def test_unseen_state_does_not_raise(self):
        # The vocabulary is open: 17 values observed in deployment 5,
        # and the upstream fixtures already use RFNOFF. A closed enum
        # (as in the rf_state_tools prototype) raises here.
        cols = md.flatten_metadata(
            {"rfswitch": ["WHO_KNOWS"]}, ntimes=1
        )
        assert cols["rfswitch"][0] == "WHO_KNOWS"

    def test_dict_stream_flattens_to_prefixed_columns(self):
        meta = {
            "motor": [
                {"status": "update", "el_pos": 1.0, "az_pos": 2.0},
                {"status": "update", "el_pos": 3.0, "az_pos": 4.0},
            ]
        }
        cols = md.flatten_metadata(meta, ntimes=2)
        np.testing.assert_allclose(cols["motor_el_pos"], [1.0, 3.0])
        np.testing.assert_allclose(cols["motor_az_pos"], [2.0, 4.0])

    def test_offline_sensor_keeps_ok_true_but_values_nan(self):
        # imu_az/lidar/tempctrl_lna publish dicts whose fields are all
        # None: the stream is alive, the sensor is not. That is a
        # different failure from the row being absent, and callers
        # need to tell them apart.
        meta = {"motor": [{"status": "update", "el_pos": None}]}
        cols = md.flatten_metadata(meta, ntimes=1)
        assert np.isnan(cols["motor_el_pos"][0])
        assert cols["motor_ok"][0]

    def test_short_stream_is_padded_to_ntimes(self):
        cols = md.flatten_metadata(
            {"rfswitch": ["RFANT"]}, ntimes=3
        )
        assert len(cols["rfswitch"]) == 3
        assert list(cols["rfswitch"][1:]) == [md.MISSING] * 2

    def test_error_status_marks_stream_not_ok(self):
        meta = {"motor": [{"status": "error", "el_pos": 1.0}]}
        cols = md.flatten_metadata(meta, ntimes=1)
        assert not cols["motor_ok"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metadata.py -v`
Expected: FAIL with `ImportError: cannot import name 'metadata'`

- [ ] **Step 3: Write the implementation**

```python
# src/eigsep_data/metadata.py
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
field is ``None``            NaN                 True
===========================  ==================  ==============

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
        "az_pos", "az_target_pos", "el_pos", "el_target_pos", "boot_id",
    ),
    "potmon": ("pot_az_angle", "pot_az_near_rail", "sp1_term_name"),
    "imu_el": ("accel_x", "accel_y", "accel_z", "el_deg"),
    "tempctrl_load": ("T_now", "active"),
    "rfswitch_therm": ("temp_therm0", "temp_therm1", "temp_therm2"),
    "system_current": ("current_a",),
    "lidar": ("distance_m",),
}

#: Streams whose value is a bare scalar rather than a dict.
SCALAR_STREAMS = ("rfswitch",)


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
    streams : iterable of str or None
        Streams to carry. ``None`` uses :data:`CURATED_FIELDS` plus the
        scalar streams; ``"all"`` takes every stream and field present.

    Returns
    -------
    cols : dict[str, np.ndarray]
        Column name to length-*ntimes* array. ``rfswitch`` is a string
        array; dict-stream fields are float arrays except string fields,
        which stay object arrays. Each stream also yields a boolean
        ``<stream>_ok``.
    """
    metadata = metadata or {}
    if streams == "all":
        wanted = {k: None for k in metadata}
    elif streams is None:
        wanted = dict(CURATED_FIELDS)
        for name in SCALAR_STREAMS:
            wanted.setdefault(name, None)
    else:
        wanted = {}
        for name in streams:
            wanted[name] = CURATED_FIELDS.get(name)

    cols = {}
    for name, fields in wanted.items():
        entries = metadata.get(name) or []
        entries = list(entries[:ntimes])
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
                        if k not in ("status", "sensor_name", "app_id")
                    }
                )
            )
        for field in fields:
            raw = [
                e.get(field) if isinstance(e, dict) else None
                for e in entries
            ]
            if any(isinstance(v, str) for v in raw):
                cols[f"{name}_{field}"] = np.array(raw, dtype=object)
            else:
                cols[f"{name}_{field}"] = np.array(
                    [_as_float(v) for v in raw], dtype=float
                )
        cols[f"{name}_ok"] = ok
    return cols
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metadata.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/metadata.py tests/test_metadata.py
git commit -m "feat: flatten per-integration metadata streams to columns"
```

---

### Task 3: `MetadataIndex` — scan a directory into a table

No cache yet; that is Task 5. This task establishes the schema and the identity columns.

**Files:**
- Create: `src/eigsep_data/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `metadata.flatten_metadata`, `metadata.MISSING`
- Produces:
  - `SCHEMA_VERSION = 1`
  - `SYNC_TOLERANCE_S = 3600.0`
  - `MetadataIndex(data_dir, streams=None, patterns=("*.h5",), cache=True)`
  - attribute `.table` (`pandas.DataFrame`), method `.rebuild()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index.py
"""Tests for eigsep_data.index."""

import numpy as np

from eigsep_data.index import MetadataIndex
from eigsep_data.metadata import MISSING

from conftest import write_corr_file


class TestMetadataIndexScan:
    def test_one_row_per_integration(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        assert len(idx.table) == 180  # 3 files x 60

    def test_identity_columns_present(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        for col in ("file", "row", "time", "acc_cnt", "sync_recovered",
                    "sync_consistent", "integration_time", "run_tag"):
            assert col in idx.table.columns

    def test_row_identity_is_file_and_row_not_position(self, corr_dir):
        # Positions shift when the file set changes; annotations and
        # selections pinned to them would rot invisibly.
        idx = MetadataIndex(corr_dir, cache=False)
        first = idx.table[idx.table.file == "corr_20260717_150041Z.h5"]
        assert list(first.row) == list(range(60))

    def test_integration_time_varies_per_file(self, corr_dir):
        # Deployment 5 carries both 0.2684 s and 0.5369 s files, and the
        # writer's transition guard width depends on it. Anything that
        # assumes one value for a deployment is wrong.
        idx = MetadataIndex(corr_dir, cache=False)
        assert set(np.round(idx.table.integration_time, 4)) == {
            0.5369, 0.2684,
        }

    def test_absent_stream_reads_as_missing(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        sub = idx.table[idx.table.file == "corr_20260717_151041Z.h5"]
        assert (sub.rfswitch == MISSING).all()

    def test_sync_recovered_matches_the_writer_formula(self, corr_dir):
        # sync = times - acc_cnt * integration_time. This is the direct
        # bad-clock diagnostic and must be exact.
        idx = MetadataIndex(corr_dir, cache=False)
        sub = idx.table[idx.table.file == "corr_20260717_150041Z.h5"]
        np.testing.assert_allclose(sub.sync_recovered, 1.7843e9, atol=1e-3)


class TestSyncConsistent:
    def test_flags_a_stale_sync_time(self, tmp_path):
        # A stale sync_time poisons every time in the file while the
        # filename, stamped at write time, stays correct. Deployment 5
        # has 12.4% such files, off by ~54 days.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=10,
            sync_time=1.7843e9 - 54 * 86400,
        )
        idx = MetadataIndex(tmp_path, cache=False)
        assert not idx.table.sync_consistent.any()

    def test_accepts_the_normal_write_backlog(self, tmp_path):
        # Good deployment-5 files land within [-1, +945] s of their
        # filename; the writer backlogs by up to ~16 min. That must not
        # be flagged.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=10,
            sync_time=1784300441.0 - 900,
        )
        idx = MetadataIndex(tmp_path, cache=False)
        assert idx.table.sync_consistent.all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eigsep_data.index'`

- [ ] **Step 3: Write the implementation**

```python
# src/eigsep_data/index.py
"""A per-integration index over a directory of corr files.

Scanning reads only headers and metadata -- never spectra -- so a whole
deployment (5120 files, ~1.2M integrations) indexes in ~21-26 s with all
streams, ~5.5 s with rfswitch alone. Queries then run against the table
and only the selected rows are ever read from disk.
"""

import fnmatch
import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .metadata import MISSING, flatten_metadata

SCHEMA_VERSION = 1

#: A file whose last integration time differs from its filename by more
#: than this is flagged ``sync_consistent = False``. Deployment 5 splits
#: cleanly: good files land in [-1, +945] s, stale ones ~54 days out.
SYNC_TOLERANCE_S = 3600.0

#: Per-file header attrs broadcast onto every row of that file.
HEADER_ATTRS = ("run_tag", "integration_time", "adc_mux_sel")


def _filename_unix(name):
    """Unix time parsed from a corr filename, or NaN."""
    match = re.search(r"(\d{8})_(\d{6})", Path(name).stem)
    if match is None:
        return np.nan
    stamp = datetime.strptime(
        match.group(1) + match.group(2), "%Y%m%d%H%M%S"
    )
    return stamp.replace(tzinfo=timezone.utc).timestamp()


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
        Filename globs to index.
    cache : bool
        Read and write the sidecar cache (see :meth:`rebuild`).

    Attributes
    ----------
    table : pandas.DataFrame
        Indexed by position, but row *identity* is the ``(file, row)``
        pair -- never the positional index, which shifts whenever the
        file set changes.
    """

    def __init__(
        self, data_dir, streams=None, patterns=("*.h5",), cache=True
    ):
        self.data_dir = Path(data_dir)
        self.streams = streams
        self.patterns = tuple(patterns)
        self.use_cache = cache
        self.table = None
        self.rebuild()

    def _files(self):
        found = []
        for pattern in self.patterns:
            found.extend(self.data_dir.glob(pattern))
        return sorted(set(found))

    def _scan_file(self, path):
        with h5py.File(path, "r") as h5:
            if "header" not in h5 or "times" not in h5["header"]:
                raise KeyError("no header/times")
            times = np.asarray(h5["header"]["times"], dtype=float)
            ntimes = times.size
            acc_cnt = np.asarray(
                h5["header"].get("acc_cnt", np.full(ntimes, np.nan)),
                dtype=float,
            )
            attrs = dict(h5["header"].attrs)
            # Data keys come from the same handle: reopening the file
            # would double the scan cost, and the scan budget for a
            # whole deployment is ~21-26 s.
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

        integration_time = float(attrs.get("integration_time", np.nan))
        cols = {
            "file": np.repeat(path.name, ntimes),
            "row": np.arange(ntimes, dtype=np.int32),
            "time": times,
            "acc_cnt": acc_cnt,
            "sync_recovered": times - acc_cnt * integration_time,
        }
        fname_t = _filename_unix(path.name)
        lag = fname_t - times[-1] if ntimes else np.nan
        consistent = bool(np.isfinite(lag) and abs(lag) <= SYNC_TOLERANCE_S)
        cols["sync_consistent"] = np.repeat(consistent, ntimes)
        for key in HEADER_ATTRS:
            value = attrs.get(key, np.nan)
            if isinstance(value, bytes):
                value = value.decode()
            cols[key] = np.repeat(value, ntimes)
        cols["data_keys"] = np.repeat(",".join(data_keys), ntimes)
        cols.update(flatten_metadata(meta, ntimes, streams=self.streams))
        return pd.DataFrame(cols)

    def rebuild(self):
        """Scan every matching file and rebuild :attr:`table`."""
        frames = []
        for path in self._files():
            try:
                frames.append(self._scan_file(path))
            except (OSError, KeyError, ValueError) as exc:
                warnings.warn(f"Skipping {path.name}: {exc}")
        if not frames:
            raise FileNotFoundError(
                f"No indexable files matching {self.patterns} in "
                f"{self.data_dir}"
            )
        table = pd.concat(frames, ignore_index=True)
        if "rfswitch" in table:
            table["rfswitch"] = table["rfswitch"].fillna(MISSING)
        self.table = table.sort_values(
            ["time", "file", "row"], kind="stable"
        ).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_index.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/index.py tests/test_index.py
git commit -m "feat: per-integration metadata index over a corr directory"
```

---

### Task 4: `Selection` and the query surface

**Files:**
- Modify: `src/eigsep_data/index.py` (append `Selection`, add `MetadataIndex.select`)
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `MetadataIndex.table`
- Produces:
  - `MetadataIndex.select(*, files=None, time=None, where=None, **filters) -> Selection`
  - `Selection` with `.meta` (DataFrame), `.index`, `.nrows`, `.files`, `.select(...)`, `.summary() -> str`, `.visits(gap_s=600) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selection.py
"""Tests for eigsep_data.index.Selection."""

import numpy as np
import pytest

from eigsep_data.index import MetadataIndex


class TestSelect:
    def test_selects_by_state(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        assert sel.nrows == 20
        assert set(sel.meta.rfswitch) == {"RFANT"}

    def test_accepts_a_list_of_states(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFAMB", "RFNON"]
        )
        assert sel.nrows == 60

    def test_unseen_state_returns_empty_not_error(self, corr_dir):
        # The vocabulary is open; asking for a state this directory
        # never saw is a legitimate empty result, not a crash.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="VNAO")
        assert sel.nrows == 0

    def test_none_rows_are_never_returned_as_a_state(self, corr_dir):
        # The ladder has 5 None rows; they must not answer to RFNOFF.
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch="RFNOFF"
        )
        assert sel.nrows == 30  # 20 + 10, not 35

    def test_filters_by_time_range(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        lo = float(idx.table.time.min())
        sel = idx.select(time=(lo, lo + 10 * 0.5369))
        assert sel.nrows == 10

    def test_filters_by_filename_glob(self, corr_dir):
        # Filename selectors matter because 12% of deployment-5 files
        # have header times wrong by days -- a time range cannot reach
        # them correctly, but a glob can.
        sel = MetadataIndex(corr_dir, cache=False).select(
            files="corr_20260717_1500*"
        )
        assert set(sel.meta.file) == {"corr_20260717_150041Z.h5"}

    def test_filters_by_run_tag(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            run_tag="motor_scan"
        )
        assert sel.nrows == 60

    def test_where_escape_hatch(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            where=lambda df: df.motor_az_pos > 50
        )
        assert sel.nrows > 0
        assert (sel.meta.motor_az_pos > 50).all()

    def test_selection_composes(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        sel = idx.select(rfswitch="RFANT").select(
            where=lambda df: df.row < 5
        )
        assert sel.nrows == 5

    def test_unknown_column_raises_with_a_useful_message(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        with pytest.raises(KeyError, match="nonsense"):
            idx.select(nonsense="x")


class TestSummary:
    def test_reports_what_each_filter_removed(self, corr_dir):
        # load_gated silently dropped every None row and every file with
        # no stream. Making the exclusions visible is the point.
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch="RFANT"
        )
        text = sel.summary()
        assert "180" in text and "20" in text
        assert "rfswitch" in text


class TestVisits:
    def test_groups_contiguous_runs(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFAMB", "RFNON"]
        )
        visits = sel.visits(gap_s=60)
        assert visits[0] == 0
        assert len(visits) == sel.nrows

    def test_new_visit_after_a_gap(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        sel = idx.select(where=lambda df: df.row < 2)
        # Three files, 10 min apart, two rows each.
        assert len(np.unique(sel.visits(gap_s=60))) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_selection.py -v`
Expected: FAIL with `AttributeError: 'MetadataIndex' object has no attribute 'select'`

- [ ] **Step 3: Write the implementation**

Append to `src/eigsep_data/index.py`:

```python
class Selection:
    """
    A set of integrations chosen from a :class:`MetadataIndex`.

    Holds only metadata rows -- no spectra. Load them with
    :meth:`eigsep_data.EigsepData.from_selection`.
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
        """Filenames contributing rows, in time order."""
        return list(dict.fromkeys(self.meta.file))

    def select(self, **kwargs):
        """Narrow this selection further; same arguments as
        :meth:`MetadataIndex.select`."""
        return _apply_filters(
            self.index, self.meta, self.provenance, **kwargs
        )

    def visits(self, gap_s=600):
        """Group rows into contiguous visits separated by *gap_s*."""
        times = self.meta.time.to_numpy()
        visits = np.zeros(times.size, dtype=int)
        if times.size > 1:
            visits[1:] = np.cumsum(np.diff(times) > gap_s)
        return visits

    def summary(self):
        """Human-readable account of what each filter removed."""
        lines = [f"{len(self.index.table)} rows indexed"]
        for name, before, after in self.provenance:
            lines.append(
                f"  {name}: {before} -> {after} "
                f"({before - after} removed)"
            )
        lines.append(f"{self.nrows} rows selected "
                     f"from {len(self.files)} files")
        return "\n".join(lines)


def _apply_filters(
    index, table, provenance, *, files=None, time=None, where=None,
    **filters
):
    provenance = list(provenance)
    current = table

    def step(name, mask):
        nonlocal current
        before = len(current)
        current = current[mask]
        provenance.append((name, before, len(current)))

    if files is not None:
        patterns = [files] if isinstance(files, str) else list(files)
        step(
            f"files={files!r}",
            current.file.map(
                lambda n: any(fnmatch.fnmatch(n, p) for p in patterns)
            ),
        )
    if time is not None:
        lo, hi = time
        from .data import to_unix_time

        lo, hi = to_unix_time(lo), to_unix_time(hi)
        step(f"time=({lo}, {hi})",
             (current.time >= lo) & (current.time < hi))
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
```

And add to `MetadataIndex`:

```python
    def select(self, **kwargs):
        """
        Choose integrations by metadata.

        Keyword arguments are column filters: a scalar matches equality,
        a list matches membership. ``files=`` takes a filename glob (or
        list of them), ``time=(lo, hi)`` a half-open range accepted by
        :func:`eigsep_data.data.to_unix_time`, and ``where=`` a callable
        taking the DataFrame and returning a boolean mask.

        Returns
        -------
        Selection
        """
        return _apply_filters(self, self.table, [], **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_selection.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/index.py tests/test_selection.py
git commit -m "feat: Selection query surface with visible filter provenance"
```

---

### Task 5: Sidecar cache with self-invalidation

**Files:**
- Modify: `src/eigsep_data/index.py`
- Test: `tests/test_index_cache.py`

**Interfaces:**
- Consumes: `MetadataIndex.table`, `SCHEMA_VERSION`
- Produces: `MetadataIndex.cache_path` (property), `MetadataIndex._fingerprint()`, cache read/write inside `rebuild()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_cache.py
"""The cache must never serve a table that no longer matches the data."""

import numpy as np

from eigsep_data.index import MetadataIndex

from conftest import write_corr_file


class TestCache:
    def test_second_build_reuses_the_cache(self, corr_dir):
        first = MetadataIndex(corr_dir)
        assert first.cache_path.exists()
        second = MetadataIndex(corr_dir)
        assert second.from_cache
        np.testing.assert_array_equal(
            first.table.time.to_numpy(), second.table.time.to_numpy()
        )

    def test_new_file_invalidates(self, corr_dir):
        MetadataIndex(corr_dir)
        write_corr_file(
            corr_dir / "corr_20260717_153041Z.h5",
            ntimes=60,
            rfswitch=["RFANT"] * 60,
            sync_time=1.7843e9 + 1800,
        )
        rebuilt = MetadataIndex(corr_dir)
        assert not rebuilt.from_cache
        assert len(rebuilt.table) == 240

    def test_modified_file_invalidates(self, corr_dir):
        MetadataIndex(corr_dir)
        write_corr_file(
            corr_dir / "corr_20260717_150041Z.h5",
            ntimes=30,
            rfswitch=["RFANT"] * 30,
        )
        rebuilt = MetadataIndex(corr_dir)
        assert not rebuilt.from_cache
        assert len(rebuilt.table) == 150

    def test_widened_stream_request_invalidates(self, corr_dir):
        # A cache built with the curated set cannot answer a query about
        # a stream it never read.
        MetadataIndex(corr_dir)
        wider = MetadataIndex(corr_dir, streams="all")
        assert not wider.from_cache

    def test_schema_bump_invalidates(self, corr_dir, monkeypatch):
        MetadataIndex(corr_dir)
        monkeypatch.setattr("eigsep_data.index.SCHEMA_VERSION", 99)
        assert not MetadataIndex(corr_dir).from_cache

    def test_cache_false_never_writes(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        assert not idx.cache_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_index_cache.py -v`
Expected: FAIL with `AttributeError: 'MetadataIndex' object has no attribute 'cache_path'`

- [ ] **Step 3: Write the implementation**

Add to `src/eigsep_data/index.py` (`json` and `hashlib` are already imported at module top from Task 3):

```python
CACHE_NAME = ".eigsep_index.h5"


class MetadataIndex:
    ...

    @property
    def cache_path(self):
        """Sidecar cache location. Gitignored -- it is derived from
        untracked data and rebuilt whenever the inputs move."""
        return self.data_dir / CACHE_NAME

    def _fingerprint(self):
        """Identity of the inputs this table was built from."""
        manifest = [
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in self._files()
        ]
        payload = json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "streams": "all"
                if self.streams == "all"
                else sorted(self.streams or []),
                "patterns": sorted(self.patterns),
                "manifest": manifest,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _read_cache(self, fingerprint):
        if not self.cache_path.exists():
            return None
        try:
            with h5py.File(self.cache_path, "r") as h5:
                if h5.attrs.get("fingerprint") != fingerprint:
                    return None
                cols = {}
                for name in h5["columns"]:
                    values = h5["columns"][name][()]
                    if values.dtype.kind == "S":
                        values = values.astype(str)
                    cols[name] = values
        except (OSError, KeyError):
            return None
        return pd.DataFrame(cols)

    def _write_cache(self, fingerprint):
        try:
            with h5py.File(self.cache_path, "w") as h5:
                h5.attrs["fingerprint"] = fingerprint
                h5.attrs["schema"] = SCHEMA_VERSION
                grp = h5.create_group("columns")
                for name in self.table.columns:
                    values = self.table[name].to_numpy()
                    if values.dtype == object:
                        values = values.astype("S")
                    grp.create_dataset(name, data=values)
        except OSError as exc:
            warnings.warn(f"Could not write index cache: {exc}")
```

Rewrite `rebuild` to consult the cache:

```python
    def rebuild(self, force=False):
        """Build :attr:`table`, reusing the sidecar cache when it still
        matches the inputs.

        Parameters
        ----------
        force : bool
            Ignore any existing cache and rescan.
        """
        fingerprint = self._fingerprint()
        self.from_cache = False
        if self.use_cache and not force:
            cached = self._read_cache(fingerprint)
            if cached is not None:
                self.table = cached
                self.from_cache = True
                return

        frames = []
        for path in self._files():
            try:
                frames.append(self._scan_file(path))
            except (OSError, KeyError, ValueError) as exc:
                warnings.warn(f"Skipping {path.name}: {exc}")
        if not frames:
            raise FileNotFoundError(
                f"No indexable files matching {self.patterns} in "
                f"{self.data_dir}"
            )
        table = pd.concat(frames, ignore_index=True)
        if "rfswitch" in table:
            table["rfswitch"] = table["rfswitch"].fillna(MISSING)
        self.table = table.sort_values(
            ["time", "file", "row"], kind="stable"
        ).reset_index(drop=True)
        if self.use_cache:
            self._write_cache(fingerprint)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_index_cache.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/index.py tests/test_index_cache.py
git commit -m "feat: self-invalidating sidecar cache for the metadata index"
```

---

### Task 6: Characterization test for `extract_beam_mapping_data`

This function has **zero coverage** and feeds the beam pipeline. Pin its current behaviour *before* touching it, so the rewrite in Task 8 is constrained to reproduce it exactly.

**Files:**
- Test: `tests/test_beam_extraction_characterization.py`

**Interfaces:**
- Consumes: `eigsep_data.data.extract_beam_mapping_data` as it exists today
- Produces: nothing (a gate for Task 8)

- [ ] **Step 1: Write the characterization test**

```python
# tests/test_beam_extraction_characterization.py
"""Pins extract_beam_mapping_data's current output.

Written against the implementation as it stands, BEFORE the rewrite onto
MetadataIndex. Its job is to fail loudly if the consolidation changes any
value the beam pipeline consumes. If a value here needs to change, that
is a deliberate decision to argue for -- not a test to relax quietly.
"""

import numpy as np

from eigsep_data.data import extract_beam_mapping_data

from conftest import write_corr_file


# 1784320080.0 == 2026-07-17 20:28:00Z, matching the first filename so
# sync_consistent is True and the fixture reads coherently.
BEAM_T0 = 1784320080.0


def _beam_dir(tmp_path):
    """Two files with motor/potmon/imu_el, as the beam pipeline sees."""
    for i, name in enumerate(
        ["corr_20260717_202800Z.h5", "corr_20260717_203000Z.h5"]
    ):
        write_corr_file(
            tmp_path / name,
            ntimes=20,
            sync_time=BEAM_T0 + i * 120,
            acc_cnt0=i * 20,
            keys=("0", "4", "04"),
            streams=("motor", "potmon", "imu_el"),
            rfswitch=["RFANT"] * 20,
            seed=i,
        )
    return tmp_path


class TestExtractBeamMappingDataContract:
    def test_returns_the_documented_keys(self, tmp_path):
        out = extract_beam_mapping_data(
            _beam_dir(tmp_path),
            start_time=BEAM_T0 - 1,
            end_time=BEAM_T0 + 600,
        )
        assert sorted(out) == sorted(
            [
                "times", "freqs", "sky", "ground", "cross",
                "el_pos", "az_pos", "pot_az_angle", "imu_el_deg",
                "imu_accel",
            ]
        )

    def test_shapes_and_ordering(self, tmp_path):
        out = extract_beam_mapping_data(
            _beam_dir(tmp_path),
            start_time=BEAM_T0 - 1,
            end_time=BEAM_T0 + 600,
        )
        assert out["times"].shape == (40,)
        assert out["sky"].shape == (40, 1024)
        assert out["imu_accel"].shape == (40, 3)
        # Chronological, across file boundaries.
        assert (np.diff(out["times"]) > 0).all()

    def test_values_are_stable(self, tmp_path):
        # Golden values: any drift in selection, sorting, or metadata
        # alignment shows up here.
        out = extract_beam_mapping_data(
            _beam_dir(tmp_path),
            start_time=BEAM_T0 - 1,
            end_time=BEAM_T0 + 600,
        )
        np.testing.assert_allclose(out["az_pos"][:5], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(out["times"][0], BEAM_T0)
        assert np.isfinite(out["sky"]).all()

    def test_sweep_slice_applies_to_every_per_sample_array(self, tmp_path):
        out = extract_beam_mapping_data(
            _beam_dir(tmp_path),
            start_time=BEAM_T0 - 1,
            end_time=BEAM_T0 + 600,
            sweep_slice=slice(5, 15),
        )
        assert out["times"].shape == (10,)
        assert out["sky"].shape == (10, 1024)
        assert out["freqs"].shape == (1024,)  # never sliced
```

The fixture needs `potmon` and `imu_el` support. Extend `write_corr_file` in `tests/conftest.py`:

```python
    if "potmon" in streams:
        md["potmon"] = [
            {
                "status": "update",
                "sensor_name": "potmon",
                "app_id": 2,
                "pot_az_angle": 10.0 + i,
                "pot_az_voltage": 1.5,
                "pot_az_near_rail": False,
                "sp1_term_name": "SHORT",
            }
            for i in range(ntimes)
        ]
    if "imu_el" in streams:
        md["imu_el"] = [
            {
                "status": "update",
                "sensor_name": "imu_el",
                "app_id": 3,
                "accel_x": np.cos(0.01 * i),
                "accel_y": np.sin(0.01 * i),
                "accel_z": 0.1,
                "el_deg": 0.01 * i,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "standby": False,
            }
            for i in range(ntimes)
        ]
```

- [ ] **Step 2: Run the test — it must PASS on current code**

Run: `.venv/bin/python -m pytest tests/test_beam_extraction_characterization.py -v`
Expected: **4 passed.** This is the one test in this plan that must be green before any implementation. If it fails, the golden values are wrong — fix them to match current behaviour, do not change `data.py`.

- [ ] **Step 3: Record the baseline in the commit message**

```bash
.venv/bin/python -m pytest tests/test_beam_extraction_characterization.py -q
```

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_beam_extraction_characterization.py
git commit -m "test: characterize extract_beam_mapping_data before rewiring"
```

---

### Task 7: `EigsepData` carries metadata; `from_selection`

**Files:**
- Modify: `src/eigsep_data/data.py:75-157` (`EigsepData` dataclass, `from_path`, `slice`)
- Test: `tests/test_eigsep_data.py`

**Interfaces:**
- Consumes: `Selection`
- Produces:
  - `EigsepData(data, acc_cnt, times, freq, meta=None)`
  - `EigsepData.from_selection(selection, keys=None) -> EigsepData`
  - `Selection.load(keys=None) -> EigsepData`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eigsep_data.py
"""EigsepData must carry metadata aligned to its own time axis."""

import numpy as np

from eigsep_data.data import EigsepData
from eigsep_data.index import MetadataIndex


class TestFromSelection:
    def test_reads_only_the_selected_rows(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert d.data["0"].shape == (20, 1024)
        assert d.times.shape == (20,)

    def test_meta_is_aligned_row_for_row(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFAMB", "RFNON"]
        )
        d = EigsepData.from_selection(sel, keys=["0"])
        assert len(d.meta) == len(d.times)
        np.testing.assert_allclose(d.meta.time.to_numpy(), d.times)
        assert set(d.meta.rfswitch) == {"RFAMB", "RFNON"}

    def test_reads_only_requested_keys(self, corr_dir):
        # Reading every key is what OOMed the 16 GB laptop; the loader
        # must never pull keys the caller did not ask for.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert list(d.data) == ["0"]

    def test_times_are_not_shifted(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert abs(d.times[0] - 1.7843e9) < 1.0

    def test_empty_selection_raises_clearly(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="VNAO")
        try:
            EigsepData.from_selection(sel, keys=["0"])
        except ValueError as exc:
            assert "no integrations" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError")

    def test_selection_load_is_equivalent(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        np.testing.assert_allclose(
            sel.load(keys=["0"]).times,
            EigsepData.from_selection(sel, keys=["0"]).times,
        )

    def test_slice_carries_meta(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"]).slice(0, 5)
        assert len(d.meta) == 5
        assert d.times.shape == (5,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eigsep_data.py -v`
Expected: FAIL with `AttributeError: type object 'EigsepData' has no attribute 'from_selection'`

- [ ] **Step 3: Write the implementation**

In `src/eigsep_data/data.py`, add `meta` to the dataclass and the new constructor:

```python
@dataclass
class EigsepData:

    data: dict[str, np.ndarray] = None
    acc_cnt: np.ndarray = None
    times: np.ndarray = None
    freq: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 250, num=1024, endpoint=False)
    )
    #: Per-integration metadata, one row per entry in ``times``.
    meta: "pd.DataFrame" = None

    @classmethod
    def from_selection(cls, selection, keys=None):
        """
        Read the spectra for the integrations in *selection*.

        Only the chosen rows of the chosen keys are read, so selecting a
        few percent of a deployment costs a few percent of the I/O.

        Parameters
        ----------
        selection : eigsep_data.index.Selection
        keys : list of str or None
            Data keys to read. ``None`` reads every key common to the
            selected files.

        Returns
        -------
        EigsepData
        """
        meta = selection.meta
        if len(meta) == 0:
            raise ValueError(
                "Selection contains no integrations; nothing to load."
            )
        data_dir = selection.index.data_dir
        if keys is None:
            keys = sorted(
                set.intersection(
                    *(
                        set(str(k).split(","))
                        for k in meta.data_keys.unique()
                    )
                )
            )
        chunks = {k: [] for k in keys}
        freq = None
        for name, group in meta.groupby("file", sort=False):
            rows = group.row.to_numpy()
            with h5py.File(data_dir / name, "r") as h5:
                if freq is None and "freqs" in h5["header"]:
                    freq = np.asarray(h5["header"]["freqs"])
                lo, hi = int(rows.min()), int(rows.max()) + 1
                local = rows - lo
                for key in keys:
                    if key not in h5["data"]:
                        continue
                    block = np.asarray(h5["data"][key][lo:hi])
                    chunks[key].append(block[local])
        order = np.argsort(meta.time.to_numpy(), kind="stable")
        data = {
            k: np.concatenate(v, axis=0)[order]
            for k, v in chunks.items()
            if v
        }
        return cls(
            data=data,
            acc_cnt=meta.acc_cnt.to_numpy()[order],
            times=meta.time.to_numpy()[order],
            freq=freq,
            meta=meta.iloc[order].reset_index(drop=True),
        )
```

Update `slice` to carry `meta`:

```python
        return EigsepData(
            data=sliced_data,
            acc_cnt=sliced_acc_cnt,
            times=sliced_times,
            freq=self.freq,
            meta=None
            if self.meta is None
            else self.meta.iloc[min_index:max_index].reset_index(drop=True),
        )
```

Add `Selection.load` in `index.py`:

```python
    def load(self, keys=None):
        """Read the spectra for these integrations."""
        from .data import EigsepData

        return EigsepData.from_selection(self, keys=keys)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eigsep_data.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/data.py src/eigsep_data/index.py \
        tests/test_eigsep_data.py
git commit -m "feat: EigsepData.from_selection reads only the chosen rows"
```

---

### Task 8: Rewire the beam extractor; delete `_select_h5_in_range`

**Files:**
- Modify: `src/eigsep_data/data.py:186-297` (delete `_select_h5_in_range`), `:299-435` (rewrite body of `extract_beam_mapping_data`)
- Test: `tests/test_beam_extraction_characterization.py` (unchanged — it is the gate)

**Interfaces:**
- Consumes: `MetadataIndex`, `Selection`, `EigsepData.from_selection`
- Produces: `extract_beam_mapping_data(...)` with its **existing** signature and return dict

- [ ] **Step 1: Confirm the gate is green before touching anything**

Run: `.venv/bin/python -m pytest tests/test_beam_extraction_characterization.py -v`
Expected: 4 passed. Do not proceed otherwise.

- [ ] **Step 2: Rewrite the body onto the index**

Replace the body of `extract_beam_mapping_data` (keep the signature and docstring, adding a note that `counts_per_deg` is unchanged):

```python
    start_unix = to_unix_time(start_time)
    end_unix = to_unix_time(end_time)
    if end_unix <= start_unix:
        raise ValueError("end_time must be later than start_time.")

    from .index import MetadataIndex

    index = MetadataIndex(
        data_dir,
        streams=("motor", "potmon", "imu_el"),
        patterns=file_patterns,
        cache=False,
    )
    sel = index.select(time=(start_unix, end_unix))
    if sel.nrows == 0:
        raise ValueError(
            "No integrations found inside the requested time range."
        )
    for key in (sky_key, ground_key, cross_key):
        present = set(str(sel.meta.data_keys.iloc[0]).split(","))
        if key not in present:
            raise KeyError(
                f"Data key {key!r} not found; available keys: "
                f"{sorted(present)}"
            )

    loaded = EigsepData.from_selection(
        sel, keys=[sky_key, ground_key, cross_key]
    )
    meta = loaded.meta
    el_pos = meta.motor_el_pos.to_numpy()
    accel = np.column_stack(
        [
            meta.imu_el_accel_x.to_numpy(),
            meta.imu_el_accel_y.to_numpy(),
            meta.imu_el_accel_z.to_numpy(),
        ]
    )
    out = {
        "times": loaded.times,
        "freqs": loaded.freq,
        "sky": loaded.data[sky_key],
        "ground": loaded.data[ground_key],
        "cross": loaded.data[cross_key],
        "el_pos": el_pos,
        "az_pos": meta.motor_az_pos.to_numpy(),
        "pot_az_angle": meta.potmon_pot_az_angle.to_numpy(),
        "imu_el_deg": imu_el_from_accel(accel, el_pos, counts_per_deg),
        "imu_accel": accel,
    }
    if sweep_slice is not None:
        out = {
            k: (v if k == "freqs" else v[sweep_slice])
            for k, v in out.items()
        }
    return out
```

- [ ] **Step 3: Delete `_select_h5_in_range`**

Remove `src/eigsep_data/data.py:186-297` entirely — it is private with exactly one caller, now gone.

- [ ] **Step 4: Run the gate and the full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass, including the 4 characterization tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/data.py
git commit -m "refactor: beam extraction onto the metadata index"
```

---

### Task 9: Clock handling — stop mutating epoch

**Files:**
- Modify: `src/eigsep_data/data.py:53-72` (`_parse_time_from_name`), `:85-157` (`from_path`), add `format_time`
- Modify: `notebooks/christian/deployment4/explore.ipynb:33`
- Test: `tests/test_clock.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `format_time(t, tz="UTC", fmt="%Y-%m-%d %H:%M:%S") -> str | list[str]`
  - `FILENAME_TZ = {"deployment4": "America/Los_Angeles", "deployment5": "UTC"}`
  - `_parse_time_from_name(fname, tz="UTC") -> datetime` (now tz-aware)
  - `EigsepData.from_path(..., pacific_to_mountain=None)` — deprecated, ignored

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clock.py
"""Epoch is never mutated; timezone is a rendering concern."""

import warnings
from datetime import timezone

import numpy as np
import pytest

from eigsep_data import data


class TestFormatTime:
    def test_renders_utc_by_default(self):
        # 1784321280.0 == 2026-07-17 20:48:00Z (verified, not assumed).
        assert data.format_time(1784321280.0) == "2026-07-17 20:48:00"

    def test_renders_mountain_for_field_notes(self):
        # The legacy +3600 existed so timestamps matched watches in
        # Utah. That is a rendering job, not a correction to the data.
        # MDT is UTC-6, so 20:48Z is 14:48 local.
        assert (
            data.format_time(1784321280.0, tz="America/Denver")
            == "2026-07-17 14:48:00"
        )

    def test_accepts_an_array(self):
        out = data.format_time(np.array([1784321280.0, 1784321340.0]))
        assert out == ["2026-07-17 20:48:00", "2026-07-17 20:49:00"]


class TestParseTimeFromName:
    def test_is_timezone_aware(self):
        t = data._parse_time_from_name("corr_20260717_204800Z.h5")
        assert t.tzinfo is not None
        assert t.timestamp() == 1784321280.0

    def test_deployment4_filenames_are_pacific(self):
        # Before eigsep_observing c4ef1ee the writer used a naive
        # datetime.now(), so D1-4 filenames are Pacific local wall clock.
        t = data._parse_time_from_name(
            "corr_20250922_160500.h5", tz="America/Los_Angeles"
        )
        assert t.utcoffset().total_seconds() == -7 * 3600


class TestNoSilentShift:
    def test_from_path_does_not_shift_times(self, corr_dir):
        d = data.EigsepData.from_path(corr_dir)
        assert abs(d.times[0] - 1.7843e9) < 1.0

    def test_pacific_to_mountain_warns_and_is_ignored(self, corr_dir):
        with pytest.warns(DeprecationWarning, match="pacific_to_mountain"):
            d = data.EigsepData.from_path(
                corr_dir, pacific_to_mountain=True
            )
        # Crucially: it warns AND does nothing. Deployment-5 epochs are
        # verified correct UTC; adding 3600 s corrupts them.
        assert abs(d.times[0] - 1.7843e9) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_clock.py -v`
Expected: FAIL with `AttributeError: module 'eigsep_data.data' has no attribute 'format_time'`

- [ ] **Step 3: Write the implementation**

```python
# src/eigsep_data/data.py -- new, near to_unix_time
from zoneinfo import ZoneInfo

#: Timezone the *filename* of each deployment was stamped in. Before
#: eigsep_observing commit c4ef1ee the writer used a naive
#: datetime.now(), so deployment 1-4 filenames are local wall clock;
#: from deployment 5 they are UTC with a Z suffix.
FILENAME_TZ = {
    "deployment4": "America/Los_Angeles",
    "deployment5": "UTC",
}


def format_time(t, tz="UTC", fmt="%Y-%m-%d %H:%M:%S"):
    """
    Render Unix epoch seconds in a named timezone.

    Deployment timestamps are epoch and always correct; only their
    *display* is zone-dependent. Use ``tz="America/Denver"`` to match
    what watches read in the field, which is what the old
    ``pacific_to_mountain`` flag was really for -- it shifted the data
    to fake a rendering.

    Parameters
    ----------
    t : float or array_like
        Unix seconds.
    tz : str
        IANA zone name.
    fmt : str
        ``strftime`` format.

    Returns
    -------
    str or list of str
    """
    zone = ZoneInfo(tz)
    values = np.atleast_1d(np.asarray(t, dtype=float))
    out = [
        datetime.fromtimestamp(float(v), tz=zone).strftime(fmt)
        for v in values
    ]
    return out[0] if np.ndim(t) == 0 else out
```

Make `_parse_time_from_name` tz-aware (replacing the naive return):

```python
def _parse_time_from_name(fname: str, tz: str = "UTC") -> datetime:
    """
    Parse a timezone-aware datetime from a correlator filename.

    *tz* is the zone the filename was stamped in -- UTC from deployment
    5 onward (the ``Z`` suffix), local wall clock before it. See
    :data:`FILENAME_TZ`. Callers that guess UTC for a deployment-4 name
    are wrong by the Pacific offset.

    Note this is the file *close* time, which lags the integrations
    inside it -- by up to ~16 min on deployment-5 data, and by far more
    on the ~12% of files written before the clock synced. Use
    header["times"] whenever the actual integration time matters.
    """
    stem = Path(fname).stem
    match = re.search(r"(\d{8})_(\d{6})", stem)
    if match is None:
        raise ValueError(f"Could not parse a timestamp from {fname!r}.")
    naive = datetime.strptime(
        match.group(1) + match.group(2), "%Y%m%d%H%M%S"
    )
    return naive.replace(tzinfo=ZoneInfo(tz))
```

Rewire `from_path` onto the index and neutralise the flag:

```python
    @classmethod
    def from_path(
        cls,
        path,
        start_time=None,
        end_time=None,
        pacific_to_mountain=None,
        keys=None,
    ):
        """
        Create an EigsepData instance from a directory or a file.

        Parameters
        ----------
        path : Path
            Directory of corr files, or a single file.
        start_time, end_time : str or float, optional
            Time range, passed through :func:`to_unix_time`.
        pacific_to_mountain : bool, optional
            **Deprecated and ignored.** Times are Unix epoch and are
            never shifted. The old +3600 s was a display convention for
            deployment 1-4's Pacific-stamped filenames; use
            :func:`format_time` with ``tz="America/Denver"`` instead.
        keys : list of str, optional
            Data keys to read; ``None`` reads all keys present.

        Returns
        -------
        EigsepData
        """
        if pacific_to_mountain is not None:
            warnings.warn(
                "pacific_to_mountain is deprecated and ignored: times "
                "are Unix epoch and are never shifted. Use "
                "format_time(t, tz='America/Denver') to render Utah "
                "local time.",
                DeprecationWarning,
                stacklevel=2,
            )
        from .index import MetadataIndex

        path = Path(path)
        if path.is_file():
            index = MetadataIndex(
                path.parent, patterns=(path.name,), cache=False
            )
        else:
            index = MetadataIndex(path)
        filters = {}
        if start_time is not None or end_time is not None:
            lo = to_unix_time(start_time) if start_time else -np.inf
            hi = to_unix_time(end_time) if end_time else np.inf
            filters["time"] = (lo, hi)
        return cls.from_selection(index.select(**filters), keys=keys)
```

Update the deployment-4 notebook cell (`notebooks/christian/deployment4/explore.ipynb:33`) to render rather than shift:

```python
data = ed.EigsepData.from_path(
    DATA_DIR,
    start_time="20250719_100000",
    end_time="20250719_235959",
)
# Field notes are in Utah local time; the data stays epoch.
print(ed.format_time(data.times[0], tz="America/Denver"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_clock.py tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/data.py tests/test_clock.py \
        notebooks/christian/deployment4/explore.ipynb
git commit -m "fix: stop shifting epoch; render timezones instead"
```

---

### Task 10: Switch-state awareness in `quicklook`

**Files:**
- Modify: `src/eigsep_data/quicklook.py:203-214` (`QuickLookResult`), `:231-278` (`quicklook`), `:367-387` (`_print_report`), `:399+` (`main`)
- Test: `tests/test_quicklook_state.py`

**Interfaces:**
- Consumes: `metadata.flatten_metadata`
- Produces: `QuickLookResult.meta`, `QuickLookResult.state_counts`, `quicklook(fname, ..., state=None)`, `--state` CLI flag

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quicklook_state.py
"""quicklook must be able to report and gate on switch state."""

from eigsep_data import quicklook as ql

from conftest import RFSWITCH_LADDER, write_corr_file


class TestQuickLookState:
    def test_reports_the_state_breakdown(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        res = ql.quicklook(p)
        assert res.state_counts["RFANT"] == 20
        assert res.state_counts["UNKNOWN"] == 5
        assert res.state_counts["MISSING"] == 5

    def test_gating_restricts_the_waterfall(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        res = ql.quicklook(p, state="RFANT")
        assert res.stats["0"]["ntimes"] == 20
        assert res.times.shape == (20,)

    def test_gating_on_an_absent_state_raises(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        try:
            ql.quicklook(p, state="VNAO")
        except ValueError as exc:
            assert "VNAO" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_file_without_rfswitch_reports_all_missing(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=10,
            rfswitch=None,
        )
        res = ql.quicklook(p)
        assert res.state_counts == {"MISSING": 10}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_quicklook_state.py -v`
Expected: FAIL with `AttributeError: 'QuickLookResult' object has no attribute 'state_counts'`

- [ ] **Step 3: Write the implementation**

Add to `QuickLookResult` (after `stats`):

```python
    meta: dict = field(default=None, repr=False)

    @property
    def state_counts(self):
        """``{switch state: n integrations}`` for this file."""
        if self.meta is None or "rfswitch" not in self.meta:
            return {}
        states, counts = np.unique(
            self.meta["rfswitch"], return_counts=True
        )
        return {str(s): int(c) for s, c in zip(states, counts)}
```

In `quicklook()`, read metadata and apply the gate before flagging:

```python
    fname = Path(fname)
    data, header, raw_meta = read_hdf5(fname)
    pairs = sorted(data.keys(), key=lambda p: (len(p), p))
    nchan = next(iter(data.values())).shape[1]
    ntimes = next(iter(data.values())).shape[0]
    freqs = np.asarray(header.get("freqs", np.arange(nchan)))
    times = np.asarray(header.get("times", np.arange(ntimes)))
    meta = metadata.flatten_metadata(raw_meta, ntimes)
    if state is not None:
        keep = np.flatnonzero(meta["rfswitch"] == state)
        if keep.size == 0:
            raise ValueError(
                f"No integrations in state {state!r}; this file has "
                f"{sorted(set(meta['rfswitch']))}."
            )
        data = {p: v[keep] for p, v in data.items()}
        times = times[keep]
        meta = {k: v[keep] for k, v in meta.items()}
```

and pass `meta=meta` into the returned `QuickLookResult`. Add `from . import metadata` to the imports and `state=None` to the signature.

In `_print_report`, after the header line:

```python
    counts = res.state_counts
    if counts:
        breakdown = "  ".join(
            f"{k}={v}" for k, v in sorted(counts.items())
        )
        print(f"  switch states: {breakdown}")
```

In `main`, add the flag and thread it through:

```python
    parser.add_argument(
        "--state",
        default=None,
        help="only integrations in this RF switch state "
        "(e.g. RFANT, RFAMB, RFNON)",
    )
```

```python
            res = quicklook(
                fname, nsig=args.nsig, pad=args.pad, state=args.state
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_quicklook_state.py tests/test_quicklook.py -v`
Expected: all pass (existing quicklook tests unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/quicklook.py tests/test_quicklook_state.py
git commit -m "feat: switch-state reporting and gating in quicklook"
```

---

### Task 11: `browse.py`, exports, and removing the prototype

**Files:**
- Create: `src/eigsep_data/browse.py`
- Modify: `src/eigsep_data/__init__.py`, `pyproject.toml`
- Delete: `notebooks/christian/deployment5/rf_state_tools.py`, `notebooks/christian/deployment5/rfswitch_index.npz`
- Test: `tests/test_browse.py`

**Interfaces:**
- Consumes: `Selection`, `EigsepData.from_selection`
- Produces: `StateBrowser(selection, keys=("0", "4"), vmin=4, vmax=6.6)` with `.goto(pos)`, `.nfiles`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_browse.py
"""The browser must work headless, and fail clearly without widgets."""

import matplotlib

matplotlib.use("Agg")

from eigsep_data.browse import StateBrowser
from eigsep_data.index import MetadataIndex


class TestStateBrowser:
    def test_builds_without_widgets(self, corr_dir):
        # Importing the module must not require ipywidgets; only the
        # interactive controls do.
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch="RFANT"
        )
        b = StateBrowser(sel, keys=["0"], controls=False)
        assert b.nfiles == 1

    def test_goto_reads_one_file(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFAMB", "RFNON"]
        )
        b = StateBrowser(sel, keys=["0"], controls=False)
        b.goto(0)
        assert b.pos == 0

    def test_empty_selection_raises(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch="VNAO"
        )
        try:
            StateBrowser(sel, keys=["0"], controls=False)
        except ValueError as exc:
            assert "no integrations" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_browse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eigsep_data.browse'`

- [ ] **Step 3: Write the implementation**

```python
# src/eigsep_data/browse.py
"""Interactive flipbook over a :class:`~eigsep_data.index.Selection`.

Successor to the ``StateBrowser`` prototype in
``notebooks/christian/deployment5/rf_state_tools.py``, rebuilt on the
index so the state vocabulary stays open and the exclusions are visible.

``ipywidgets`` is imported only when the controls are built, so the
module is importable -- and testable -- without it. Install it with the
``vis`` extra.
"""

import numpy as np

from .data import EigsepData


class StateBrowser:
    """
    Flip through per-file spectra for a selection of integrations.

    Usage in a ``%matplotlib widget`` notebook::

        idx = MetadataIndex("data/deployment5_filtered")
        sel = idx.select(rfswitch="RFAMB", files="corr_20260717*")
        b = StateBrowser(sel)

    Parameters
    ----------
    selection : eigsep_data.index.Selection
    keys : sequence of str
        Data keys to plot.
    vmin, vmax : float
        log10 power limits for the waterfall.
    controls : bool
        Build the ipywidgets controls. ``False`` for headless use.
    """

    def __init__(
        self, selection, keys=("0", "4"), vmin=4, vmax=6.6, controls=True
    ):
        import matplotlib.pyplot as plt

        if selection.nrows == 0:
            raise ValueError(
                "Selection contains no integrations; nothing to browse."
            )
        self.selection = selection
        self.keys = list(keys)
        self.vmin, self.vmax = vmin, vmax
        self._files = selection.files
        self.pos = 0

        self.fig, (self.ax_w, self.ax_s) = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            layout="constrained",
            gridspec_kw={"height_ratios": [1, 1.4]},
        )
        if controls:
            self._build_controls()
        self.goto(0)

    @property
    def nfiles(self):
        return len(self._files)

    def _read(self, pos):
        name = self._files[pos]
        sub = self.selection.select(files=name)
        return name, EigsepData.from_selection(sub, keys=self.keys)

    def goto(self, pos):
        """Draw the file at index *pos*."""
        self.pos = int(np.clip(pos, 0, self.nfiles - 1))
        name, loaded = self._read(self.pos)
        states = sorted(set(loaded.meta.rfswitch))

        self.ax_w.clear()
        self.ax_s.clear()
        first = next((k for k in self.keys if k in loaded.data), None)
        if first is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                img = np.log10(np.abs(loaded.data[first]))
            self.ax_w.imshow(
                img,
                aspect="auto",
                cmap="plasma",
                interpolation="none",
                vmin=self.vmin,
                vmax=self.vmax,
                extent=[
                    loaded.freq.min(),
                    loaded.freq.max(),
                    len(loaded.times),
                    0,
                ],
            )
            self.ax_w.set_ylabel(f"row (key {first})")
        self.ax_w.set_title(
            f"[{self.pos + 1}/{self.nfiles}] {name}  "
            f"{len(loaded.times)} rows  {','.join(states)}",
            fontsize=10,
        )
        for key, spec in loaded.data.items():
            mean = np.nanmean(spec, axis=0)
            lo, hi = np.nanpercentile(spec, [10, 90], axis=0)
            (line,) = self.ax_s.plot(
                loaded.freq, mean, lw=1, label=f"key {key}"
            )
            self.ax_s.fill_between(
                loaded.freq, lo, hi, alpha=0.25,
                color=line.get_color(), lw=0,
            )
        self.ax_s.set_yscale("log")
        self.ax_s.set_xlabel("Frequency [MHz]")
        self.ax_s.set_ylabel("Power [counts]")
        self.ax_s.legend(loc="upper right", fontsize=8)
        self.fig.canvas.draw_idle()

    def _build_controls(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError(
                "StateBrowser controls need ipywidgets. Install the "
                "vis extra: pip install -e '.[vis]'"
            ) from exc

        slider = widgets.IntSlider(
            0, 0, self.nfiles - 1, description="file",
            layout=widgets.Layout(width="55%"),
        )
        play = widgets.Play(
            interval=400, min=0, max=self.nfiles - 1, step=1
        )
        widgets.jslink((play, "value"), (slider, "value"))
        slider.observe(lambda ch: self.goto(ch["new"]), names="value")
        display(widgets.HBox([play, slider]))
```

Update `src/eigsep_data/__init__.py`:

```python
from .imu import ImuCalibrator, ImuSnapshot, ImuDataset
from .s11 import S11, RawS11
from .data import EigsepData, to_unix_time, format_time
from .index import MetadataIndex, Selection
from . import metadata
from . import plot
from . import rfi
```

Add `ipywidgets` to the existing `vis` extra in `pyproject.toml`:

```toml
vis = [
    "jupyterlab",
    "jupyterlab-vim",
    "ipympl",
    "ipywidgets",
]
```

- [ ] **Step 4: Run the whole suite, then delete the prototype**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass

```bash
git rm notebooks/christian/deployment5/rf_state_tools.py \
       notebooks/christian/deployment5/rfswitch_index.npz
```

Then check nothing still imports it:

```bash
grep -rn "rf_state_tools\|rfswitch_index" --include=*.ipynb --include=*.py . \
  | grep -v '^./build'
```

Expected: no hits outside `docs/`. If a notebook still imports it, port that cell to `MetadataIndex(...).select(rfswitch=...)` in this task.

- [ ] **Step 5: Commit**

```bash
git add -A src/eigsep_data tests pyproject.toml notebooks
git commit -m "feat: StateBrowser on Selection; retire rf_state_tools"
```

---

## Verification

After Task 11:

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m flake8 src/eigsep_data tests
.venv/bin/python -m black --check src/eigsep_data tests
```

Then a real-data smoke test, which is the first thing this whole plan was for:

```python
from eigsep_data import MetadataIndex, EigsepData

idx = MetadataIndex("data/deployment5_filtered")
sel = idx.select(rfswitch="RFAMB", files="corr_20260717*")
print(sel.summary())
d = EigsepData.from_selection(sel, keys=["0", "4"])
print(d.data["4"].shape, d.meta.rfswitch.unique())
```

Expected: the ~1107 `RFAMB` rows of the Jul 17 cal cycle, `summary()` naming how many rows each filter removed, and a cold build of ~21–26 s followed by ~0.1 s on the next run.
