# Per-Integration Metadata Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `eigsep_data` select correlator integrations by RF switch state — and by any other per-integration metadata — before reading any spectra.

**Architecture:** A cheap scan of headers and metadata builds one row per integration into a pandas DataFrame (`MetadataIndex`), cached to a self-invalidating sidecar. Queries against that table produce a `Selection` of `(file, row)` pairs; only then are the chosen spectra read into an `EigsepData` carrying its aligned metadata. The four existing hand-rolled scan-parse-select implementations collapse onto this core. Time handling lives in one small `clock.py` so the filename parser exists exactly once.

**Tech Stack:** Python ≥3.11, numpy, pandas (declared but currently unused in `src/`), h5py, pytest. Reads via `eigsep_observing.io`.

**Spec:** `docs/superpowers/specs/2026-09-11-switch-state-index-design.md` (revised 2026-09-11 after review; this plan matches the revision).

## Global Constraints

- **Line length 79** (`black` line-length 79, `flake8` max-line-length 79, `extend-ignore = E203, W503`).
- **No new runtime dependencies.** The cache is written with `h5py`; `pandas<3` and `h5py` are already declared. `ipywidgets` is added only to the existing `vis` extra.
- **Never mutate epoch.** `meta.time` is `header["times"]` exactly as written; nothing adds or subtracts an offset. `EigsepData.times` carries `time_best`, which equals `meta.time` wherever the file's clock was sane and is a filename-derived estimate elsewhere. Timezone handling happens at render time only.
- **Never sort or window on header `time` alone; use `time_best`.** 12.4 % of deployment-5 files carry a stale `sync_time`, so their header times are wrong by days while their filenames are right. `time_best` is the header time when `sync_consistent`, else a filename-derived estimate.
- **The scan never opens the cache file.** `pathlib`'s glob matches dotfiles (verified on Python 3.12), so `_files()` must exclude `.eigsep_index.h5` and every other dot-prefixed name explicitly, or the cache invalidates itself on every write.
- **`cross` keys are complex exactly as `read_hdf5` returns them.** Deployment-5 files store crosses as `(ntimes, nchan, 2)` int32 (re, im); `from_selection` must apply the reader's reconstruction rule, and the fixture must write crosses that way so tests can see it.
- **Never gate by a fixed row count.** Switch-state selection keys on the state string. The writer's transition guard is `ceil(0.5 s / integration_time)` samples — 2 rows at 0.2684 s, 1 at 0.5369 s, both present in deployment 5.
- **`None` and `UNKNOWN` are different and must stay so.** `UNKNOWN` = producer asserts contamination. `None`/`MISSING` = no information reached the writer.
- **`rfswitch` is an open category, never an enum.** 17 values observed; an unseen value must not raise.
- **Row identity is `(file basename, row)`**, never a position in a concatenated array.
- **`index.py` imports from `clock` and `metadata` only.** Its single import from `data.py` is the lazy one inside `Selection.load`.
- Tests use `pytest`, live in `tests/`, and follow the existing convention of class-per-unit with docstrings stating *why* the case matters (see `tests/test_data.py`).
- Run `pytest` from the repo root using `.venv/bin/python -m pytest`.

## File Structure

| file | responsibility | status |
|---|---|---|
| `tests/conftest.py` | synthetic corr-file writer + directory fixtures | create |
| `tests/test_conftest.py` | pins that the fixture obeys the producer's contract | create |
| `src/eigsep_data/clock.py` | `to_unix_time`, `format_time`, `parse_filename_time`, `filename_unix`, `FILENAME_TZ` | create |
| `tests/test_clock.py` | clock tests | create |
| `src/eigsep_data/metadata.py` | flatten one file's metadata group → named columns; missing policy | create |
| `src/eigsep_data/index.py` | `scan_corr_file`, `MetadataIndex` (scan, cache, invalidate), `Selection` (query) | create |
| `src/eigsep_data/browse.py` | `StateBrowser` interactive flipbook over a `Selection` | create |
| `src/eigsep_data/data.py` | `EigsepData` + `.meta`, `from_selection`; re-exports `to_unix_time`; beam extractor becomes a wrapper | modify |
| `src/eigsep_data/quicklook.py` | carry `.meta`, report state breakdown, `--state` gate | modify |
| `src/eigsep_data/__init__.py` | export the new surface | modify |
| `pyproject.toml` | pytest `pythonpath`; `ipywidgets` into the `vis` extra | modify |
| `notebooks/christian/deployment4/explore.ipynb` | re-express the selection window in epoch | modify |
| `notebooks/christian/deployment5/calibration.ipynb`, `flipbook.ipynb` | port off `rf_state_tools` | modify |
| `notebooks/christian/deployment5/rf_state_tools.py` | superseded | delete |
| `notebooks/christian/deployment5/rfswitch_index.npz` | superseded (tracked in git) | delete |

Import graph: `browse.py` → `data.py` → (lazily) `index.py` → `metadata.py`, `clock.py`. `index.py` never imports `data.py` at module level.

---

### Task 1: Synthetic corr-file fixture

Everything downstream is tested against files written by the **real producer**, so the contract under test is `eigsep_observing`'s, not a mock of it. Cross keys are written as `(ntimes, NCHAN, 2)` int32 pairs, which is how deployment-5 files store them, so the reader's complex reconstruction is exercised by every test that touches a cross.

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_conftest.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `eigsep_observing.io.write_hdf5`
- Produces: `write_corr_file(path, **kw) -> Path`; `RFSWITCH_LADDER`; `NCHAN`; pytest fixture `corr_dir`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conftest.py
"""The fixture must produce files the real reader accepts."""

import h5py
import numpy as np
from eigsep_observing.io import read_hdf5

from conftest import NCHAN, RFSWITCH_LADDER, write_corr_file


class TestWriteCorrFile:
    def test_roundtrips_through_the_real_reader(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        data, header, meta = read_hdf5(p)
        assert sorted(data) == ["0", "4"]
        assert data["0"].shape == (60, NCHAN)
        assert len(header["times"]) == 60
        assert header["run_tag"] == "panda_observe"
        assert header["nchan"] == NCHAN

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

    def test_cross_key_roundtrips_as_complex(self, tmp_path):
        # Deployment-5 files store crosses as (re, im) int32 pairs and
        # read_hdf5 rebuilds complex from them. The loader written in
        # Task 8 must apply the same rule, and this fixture must give
        # the characterization test something to pin.
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=6,
            keys=("0", "4", "04"),
        )
        data, _, _ = read_hdf5(p)
        assert data["04"].dtype.kind == "c"
        assert data["04"].shape == (6, NCHAN)
        with h5py.File(p, "r") as h5:
            raw = h5["data"]["04"][()]
        assert raw.shape == (6, NCHAN, 2)
        assert raw.dtype == np.int32
        np.testing.assert_array_equal(
            data["04"], raw[..., 0] + 1j * raw[..., 1]
        )

    def test_root_attrs_are_written(self, tmp_path):
        # filter_corr_keys.py stamps filter_phase and the mux-copy
        # flags at file root, outside the producer's header group.
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=6,
            root_attrs={"filter_phase": "C", "mux_copy_4to5": True},
        )
        with h5py.File(p, "r") as h5:
            assert h5.attrs["filter_phase"] == "C"
            assert bool(h5.attrs["mux_copy_4to5"]) is True
        _, header, _ = read_hdf5(p)
        assert "filter_phase" not in header
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

import h5py
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
    root_attrs=None,
    seed=0,
):
    """
    Write one synthetic corr file and return its path.

    Keys of length 2 are cross-correlations and are stored as
    ``(ntimes, NCHAN, 2)`` int32 -- (re, im) pairs -- exactly as
    deployment-5 files are, so ``read_hdf5`` reconstructs them as
    complex. ``rfswitch=None`` omits the stream entirely, which is how
    the ~1842 deployment-5 files with no switch metadata look.
    ``root_attrs`` are written at the file root after the producer is
    done, mirroring ``filter_corr_keys.py`` (``filter_phase``,
    ``filtered_keys``, ``mux_copy_0to1``, ``mux_copy_4to5``).
    """
    rng = np.random.default_rng(seed)
    acc_cnt = np.arange(acc_cnt0, acc_cnt0 + ntimes, dtype=np.int64)
    times = acc_cnt * integration_time + sync_time
    data = {}
    for k in keys:
        if len(k) == 2:
            data[k] = rng.integers(
                -1000, 1000, size=(ntimes, NCHAN, 2), dtype=np.int32
            )
        else:
            data[k] = rng.integers(
                1, 1000, size=(ntimes, NCHAN), dtype=np.int32
            )
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
                "accel_x": float(np.cos(0.01 * i)),
                "accel_y": float(np.sin(0.01 * i)),
                "accel_z": 0.1,
                "el_deg": 0.01 * i,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "standby": False,
            }
            for i in range(ntimes)
        ]
    write_hdf5(path, data, header, metadata=md or None)
    if root_attrs:
        with h5py.File(path, "a") as h5:
            for key, value in root_attrs.items():
                h5.attrs[key] = value
    return path


@pytest.fixture
def corr_dir(tmp_path):
    """Three phase-C files: full ladder, no rfswitch stream, fast cadence.

    All three have a cross key, root attrs from the filter script, and
    filenames within the write backlog of their header times, so every
    row is ``sync_consistent`` and ``time_best == time``.
    """
    root = {"filter_phase": "C", "mux_copy_0to1": True}
    write_corr_file(
        tmp_path / "corr_20260717_150041Z.h5",
        ntimes=60,
        keys=("0", "4", "04"),
        rfswitch=RFSWITCH_LADDER,
        sync_time=1.7843e9,
        root_attrs=root,
    )
    write_corr_file(
        tmp_path / "corr_20260717_151041Z.h5",
        ntimes=60,
        keys=("0", "4", "04"),
        rfswitch=None,
        sync_time=1.7843e9 + 600,
        root_attrs=root,
        seed=1,
    )
    write_corr_file(
        tmp_path / "corr_20260717_152041Z.h5",
        ntimes=60,
        keys=("0", "4", "04"),
        rfswitch=["RFAMB"] * 30 + ["RFNON"] * 30,
        integration_time=0.2684,
        sync_time=1.7843e9 + 1200,
        run_tag="motor_scan",
        root_attrs=root,
        seed=2,
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
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_conftest.py pyproject.toml
git commit -m "test: synthetic corr files written by the real producer"
```

---

### Task 2: `clock.py` — one place for epoch, rendering, and filename zones

Time handling is pulled out of `data.py` so that `index.py` can parse filenames without importing `data.py`, and so the filename parser exists exactly once. Nothing here touches a file.

**Files:**
- Create: `src/eigsep_data/clock.py`
- Modify: `src/eigsep_data/data.py:14-72` (move `to_unix_time`, alias `_parse_time_from_name`), `:116` and `:120` (keep `from_path` behaviour-identical until Task 10)
- Test: `tests/test_clock.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `to_unix_time(value) -> float` (moved verbatim; `data.py` re-exports it)
  - `format_time(t, tz="UTC", fmt="%Y-%m-%d %H:%M:%S") -> str | list[str]`
  - `FILENAME_TZ = {"deployment4": "America/Los_Angeles", "deployment5": "UTC"}` (documentation)
  - `parse_filename_time(fname, tz=None) -> datetime` (tz-aware; `Z` suffix wins)
  - `filename_unix(fname, tz=None) -> float` (NaN when unparseable)
  - `data._parse_time_from_name(fname, tz=None)` stays as a thin alias so `tests/test_data.py` is untouched

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clock.py
"""Epoch is never mutated; timezone is a rendering concern."""

from datetime import timezone

import numpy as np
import pytest

from eigsep_data import clock, data


class TestFormatTime:
    def test_renders_utc_by_default(self):
        # 1784321280.0 == 2026-07-17 20:48:00Z (verified, not assumed).
        assert clock.format_time(1784321280.0) == "2026-07-17 20:48:00"

    def test_renders_mountain_for_field_notes(self):
        # The legacy +3600 existed so timestamps matched watches in
        # Utah. That is a rendering job, not a correction to the data.
        # MDT is UTC-6, so 20:48Z is 14:48 local.
        assert (
            clock.format_time(1784321280.0, tz="America/Denver")
            == "2026-07-17 14:48:00"
        )

    def test_accepts_an_array(self):
        out = clock.format_time(np.array([1784321280.0, 1784321340.0]))
        assert out == ["2026-07-17 20:48:00", "2026-07-17 20:49:00"]


class TestParseFilenameTime:
    def test_z_suffix_is_utc(self):
        t = clock.parse_filename_time("corr_20260717_204800Z.h5")
        assert t.tzinfo is not None
        assert t.timestamp() == 1784321280.0

    def test_z_suffix_wins_over_an_explicit_tz(self):
        # The suffix is the producer's own statement of the zone; a
        # caller's guess must not override it.
        t = clock.parse_filename_time(
            "corr_20260717_204800Z.h5", tz="America/Los_Angeles"
        )
        assert t.timestamp() == 1784321280.0

    def test_no_suffix_defaults_to_pacific(self):
        # Before eigsep_observing c4ef1ee the writer used a naive
        # datetime.now(), so deployment 1-4 filenames are Pacific wall
        # clock. September is PDT, UTC-7.
        t = clock.parse_filename_time("corr_20250922_160500.h5")
        assert t.utcoffset().total_seconds() == -7 * 3600
        assert (t.hour, t.minute) == (16, 5)

    def test_explicit_tz_is_honoured_without_suffix(self):
        t = clock.parse_filename_time(
            "corr_20250922_160500.h5", tz="America/Denver"
        )
        assert t.utcoffset().total_seconds() == -6 * 3600

    def test_disambiguating_suffix_still_parses(self):
        t = clock.parse_filename_time("corr_20260712_235712Z-1.h5")
        assert t.tzinfo == timezone.utc

    def test_unparseable_name_raises(self):
        with pytest.raises(ValueError):
            clock.parse_filename_time("not_a_corr_file.h5")

    def test_data_alias_still_works(self):
        # tests/test_data.py and any notebook using the private name
        # keep working; the alias is tz-aware.
        t = data._parse_time_from_name("corr_20260717_204800Z.h5")
        assert t.timestamp() == 1784321280.0


class TestFilenameUnix:
    def test_matches_parse(self):
        assert (
            clock.filename_unix("corr_20260717_204800Z.h5")
            == 1784321280.0
        )

    def test_nan_on_garbage(self):
        # The index calls this on every file it globs; a stray file
        # must yield NaN, not an exception.
        assert np.isnan(clock.filename_unix("notes.txt"))


class TestToUnixTime:
    def test_naive_string_is_utc(self):
        assert clock.to_unix_time("2026-07-17 20:48:00") == 1784321280.0

    def test_epoch_passthrough(self):
        assert clock.to_unix_time(1784321280.0) == 1784321280.0

    def test_reexported_from_data(self):
        # notebooks/dominic/test_beam_mapping_module.ipynb imports it
        # from eigsep_data.data.
        assert data.to_unix_time is clock.to_unix_time
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_clock.py -v`
Expected: FAIL with `ImportError: cannot import name 'clock' from 'eigsep_data'`

- [ ] **Step 3: Write the implementation**

```python
# src/eigsep_data/clock.py
"""Time handling for correlator data: epoch in, rendering out.

Every timestamp a corr file stores is Unix epoch, and this package
never shifts it. Deployment 5's absolute epoch was verified against an
external anchor (the switch-board thermistor's diurnal cycle).
Timezones enter in exactly two places: rendering an epoch for a human
(:func:`format_time`), and parsing a *filename* stamp, whose zone
depends on the deployment (:func:`parse_filename_time`).
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

#: Timezone the *filename* of each deployment was stamped in. Before
#: eigsep_observing commit c4ef1ee (2026-05-15, "stamp auto-generated
#: h5 filenames in UTC with Z suffix") the writer used a naive
#: datetime.now(), so deployment 1-4 filenames are Pacific wall clock;
#: from deployment 5 they are UTC and carry the Z suffix. Kept as
#: documentation -- parse_filename_time keys off the suffix itself.
FILENAME_TZ = {
    "deployment4": "America/Los_Angeles",
    "deployment5": "UTC",
}

#: Zone assumed for a filename stamp that has no Z suffix.
LEGACY_FILENAME_TZ = "America/Los_Angeles"

_STAMP = re.compile(r"(\d{8})_(\d{6})(Z?)")


def to_unix_time(value):
    """
    Convert a datetime string, datetime object, or Unix timestamp to Unix
    seconds (float).

    Strings without a timezone are interpreted as UTC.

    Accepted formats:
        "2026-07-17 06:00:00"
        "2026-7-17 6:00:00"
        "2026-07-17T06:00:00Z"
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"Could not interpret time {value!r}. "
                    "Use a format such as '2026-07-17 06:00:00'."
                )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


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


def parse_filename_time(fname, tz=None):
    """
    Parse a timezone-aware datetime from a correlator filename.

    A ``Z`` suffix on the stamp (``corr_20260715_172825Z.h5``) means
    UTC and wins over *tz*. Without the suffix the stamp is local wall
    clock: *tz* if given, else ``America/Los_Angeles``, the convention
    for deployments 1-4 (see :data:`FILENAME_TZ`). Handles the
    ``-1`` disambiguating suffix for files closed within one second.

    Note this is the file *close* time, which lags the integrations
    inside it -- by up to ~16 min on deployment-5 data when the writer
    backlogs. It is, however, always stamped from a correct clock,
    which header ``times`` are not on ~12 % of deployment-5 files.

    Parameters
    ----------
    fname : str or Path
    tz : str or None
        IANA zone for a stamp with no ``Z`` suffix.

    Returns
    -------
    datetime
        Timezone-aware.
    """
    match = _STAMP.search(Path(fname).stem)
    if match is None:
        raise ValueError(f"Could not parse a timestamp from {fname!r}.")
    naive = datetime.strptime(
        match.group(1) + match.group(2), "%Y%m%d%H%M%S"
    )
    if match.group(3):
        return naive.replace(tzinfo=timezone.utc)
    return naive.replace(tzinfo=ZoneInfo(tz or LEGACY_FILENAME_TZ))


def filename_unix(fname, tz=None):
    """Unix seconds from a corr filename, or NaN when it has no stamp."""
    try:
        return parse_filename_time(fname, tz=tz).timestamp()
    except ValueError:
        return np.nan
```

In `src/eigsep_data/data.py`, delete the bodies of `to_unix_time` (lines 14-50) and `_parse_time_from_name` (lines 53-72) and replace them with:

```python
from .clock import (  # noqa: F401  -- to_unix_time is re-exported
    parse_filename_time,
    to_unix_time,
)


def _parse_time_from_name(fname: str, tz=None) -> datetime:
    """Alias for :func:`eigsep_data.clock.parse_filename_time`."""
    return parse_filename_time(fname, tz=tz)
```

Drop the now-unused `re` and `timezone` imports from `data.py`. The alias is timezone-aware, and the untouched `from_path` still compares filename times against naive `strptime` results, so until Task 10 replaces `from_path` wholesale make its two list comprehensions (lines 116 and 120) strip the zone:

```python
            times = [
                _parse_time_from_name(f.name).replace(tzinfo=None)
                for f in files
            ]
```

That keeps `from_path` byte-for-byte equivalent to today's behaviour for the two intermediate commits.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_clock.py tests/test_data.py -v`
Expected: all pass, including the untouched `TestParseTimeFromName` in `tests/test_data.py`

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/clock.py src/eigsep_data/data.py tests/test_clock.py
git commit -m "feat: clock module; filename zone keyed off the Z suffix"
```

---

### Task 3: `metadata.py` — flatten one file's streams to columns

Pure transformation: dicts in, arrays out. No file I/O, so it is testable in isolation and fast. String fields never carry Python `None` -- gaps become `MISSING` -- so object columns round-trip through the cache losslessly.

**Files:**
- Create: `src/eigsep_data/metadata.py`
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `CURATED_FIELDS: dict[str, tuple[str, ...]]` (includes `tempctrl_lna`)
  - `SCALAR_STREAMS = ("rfswitch",)`
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
            "RFANT",
            md.MISSING,
            "RFANT",
            md.MISSING,
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
        cols = md.flatten_metadata({"rfswitch": ["WHO_KNOWS"]}, ntimes=1)
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
        cols = md.flatten_metadata({"rfswitch": ["RFANT"]}, ntimes=3)
        assert len(cols["rfswitch"]) == 3
        assert list(cols["rfswitch"][1:]) == [md.MISSING] * 2

    def test_error_status_marks_stream_not_ok(self):
        meta = {"motor": [{"status": "error", "el_pos": 1.0}]}
        cols = md.flatten_metadata(meta, ntimes=1)
        assert not cols["motor_ok"][0]

    def test_string_field_gap_is_missing_never_none(self):
        # A Python None in an object column would become the string
        # "None" after a trip through the h5 cache. Gaps in string
        # fields must be MISSING so cached and fresh tables are equal.
        meta = {
            "potmon": [
                {"status": "update", "sp1_term_name": "SHORT"},
                {"status": "update", "sp1_term_name": None},
                None,
            ]
        }
        cols = md.flatten_metadata(meta, ntimes=3)
        assert list(cols["potmon_sp1_term_name"]) == [
            "SHORT",
            md.MISSING,
            md.MISSING,
        ]
        assert None not in list(cols["potmon_sp1_term_name"])

    def test_tempctrl_lna_is_curated(self):
        # Calibrator work needs the LNA temperature as much as the load
        # temperature; both are two columns.
        assert md.CURATED_FIELDS["tempctrl_lna"] == ("T_now", "active")
        meta = {"tempctrl_lna": [{"status": "update", "T_now": 31.5}]}
        cols = md.flatten_metadata(meta, ntimes=1)
        np.testing.assert_allclose(cols["tempctrl_lna_T_now"], [31.5])
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
field is ``None``            ``MISSING`` / NaN   True
===========================  ==================  ==============

(``MISSING`` for string fields, NaN for numeric ones. A string column
never holds Python ``None``, so it survives the h5 cache unchanged.)

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
        array; dict-stream fields are float arrays, except fields with
        any string value, which are string arrays with ``MISSING`` for
        gaps. Each stream also yields a boolean ``<stream>_ok``.
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
                e.get(field) if isinstance(e, dict) else None
                for e in entries
            ]
            if any(isinstance(v, str) for v in raw):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metadata.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/metadata.py tests/test_metadata.py
git commit -m "feat: flatten per-integration metadata streams to columns"
```

---

### Task 4: `scan_corr_file` and `MetadataIndex` — scan a directory into a table

No cache yet; that is Task 6. This task establishes the schema, the identity columns including `time_best`, the dotfile exclusion, and the scanner hook.

**Files:**
- Create: `src/eigsep_data/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `metadata.flatten_metadata`, `metadata.MISSING`, `clock.filename_unix`
- Produces:
  - `SCHEMA_VERSION = 1`, `SYNC_TOLERANCE_S = 3600.0`, `CACHE_NAME = ".eigsep_index.h5"`
  - `HEADER_ATTRS = {"run_tag": MISSING, "integration_time": nan, "adc_mux_sel": nan, "nchan": nan}` (attr → default when absent)
  - `scan_corr_file(path, streams=None, filename_tz=None) -> pandas.DataFrame`
  - `MetadataIndex(data_dir, streams=None, patterns=("corr_*.h5",), cache=True, filename_tz=None, scanner=None)`
  - attributes `.table`, `.from_cache`; property `.cache_path`; method `.rebuild()`
  - identity columns `file, row, time, acc_cnt, sync_recovered, sync_consistent, time_fname, time_best`; then root attrs (bare names), header attrs, `data_keys`, metadata columns

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index.py
"""Tests for eigsep_data.index."""

import warnings

import numpy as np
import pandas as pd
import pytest

from eigsep_data import clock
from eigsep_data.index import CACHE_NAME, MetadataIndex, scan_corr_file
from eigsep_data.metadata import MISSING

from conftest import write_corr_file

IDENTITY = (
    "file",
    "row",
    "time",
    "acc_cnt",
    "sync_recovered",
    "sync_consistent",
    "time_fname",
    "time_best",
)


class TestMetadataIndexScan:
    def test_one_row_per_integration(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        assert len(idx.table) == 180  # 3 files x 60

    def test_identity_and_header_columns_present(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        for col in IDENTITY + (
            "integration_time",
            "run_tag",
            "adc_mux_sel",
            "nchan",
            "data_keys",
        ):
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
            0.5369,
            0.2684,
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

    def test_data_keys_are_recorded(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        assert set(idx.table.data_keys) == {"0,04,4"}

    def test_root_attrs_become_columns(self, corr_dir):
        # filter_corr_keys.py already asserts the wiring phase and the
        # mux-copy flags per file. Indexing them makes
        # select(filter_phase="C") work before any annotation layer.
        idx = MetadataIndex(corr_dir, cache=False)
        assert set(idx.table.filter_phase) == {"C"}
        assert idx.table.mux_copy_0to1.dtype == bool
        assert idx.table.mux_copy_0to1.all()

    def test_file_without_root_attrs_reads_missing(self, tmp_path):
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=4,
            root_attrs={"filter_phase": "C"},
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=4,
            sync_time=1.7843e9 + 600,
        )
        idx = MetadataIndex(tmp_path, cache=False)
        sub = idx.table[idx.table.file == "corr_20260717_151041Z.h5"]
        assert (sub.filter_phase == MISSING).all()

    def test_string_columns_never_hold_none_or_nan(self, corr_dir):
        # The cache encodes object columns as bytes; None or NaN would
        # come back as the strings "None"/"nan". The table must already
        # be free of them.
        idx = MetadataIndex(corr_dir, cache=False)
        for col in idx.table.columns:
            if idx.table[col].dtype == object:
                assert idx.table[col].map(type).eq(str).all(), col

    def test_cache_file_and_dotfiles_are_not_scanned(self, corr_dir):
        # pathlib's glob matches dotfiles, so "*.h5" would return the
        # cache itself -- which then poisons its own fingerprint.
        (corr_dir / CACHE_NAME).write_bytes(b"not an hdf5 file")
        (corr_dir / ".hidden.h5").write_bytes(b"editor dropping")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            idx = MetadataIndex(corr_dir, patterns=("*.h5",), cache=False)
        assert len(idx.table) == 180

    def test_default_pattern_is_corr_files_only(self, corr_dir):
        # Sidecars such as motor_scan_20260717_key4.h5 live next to
        # data in some directories; they are not integrations.
        write_corr_file(corr_dir / "sidecar_20260717_150041Z.h5", ntimes=4)
        idx = MetadataIndex(corr_dir, cache=False)
        assert "sidecar_20260717_150041Z.h5" not in set(idx.table.file)

    def test_scanner_hook_is_called(self, corr_dir):
        # The seam for indexing another file kind (S11 sweeps) without
        # a second cache/select/summary implementation.
        seen = []

        def scanner(path, streams=None, filename_tz=None):
            seen.append(path.name)
            return scan_corr_file(path, streams, filename_tz).head(1)

        idx = MetadataIndex(corr_dir, cache=False, scanner=scanner)
        assert len(seen) == 3
        assert len(idx.table) == 3


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

    def test_pacific_named_deployment4_file_is_consistent(self, tmp_path):
        # Deployment 1-4 filenames are Pacific wall clock with no Z
        # suffix. Parsing them as UTC would put every file 7 h off and
        # flag a whole deployment as inconsistent.
        name = "corr_20250922_160500.h5"
        t_close = clock.filename_unix(name)
        write_corr_file(tmp_path / name, ntimes=10, sync_time=t_close - 60)
        idx = MetadataIndex(tmp_path, cache=False)
        assert idx.table.sync_consistent.all()

    def test_filename_tz_is_honoured(self, tmp_path):
        name = "corr_20250922_160500.h5"
        t_close = clock.filename_unix(name, tz="America/Denver")
        write_corr_file(tmp_path / name, ntimes=10, sync_time=t_close - 60)
        assert MetadataIndex(
            tmp_path, cache=False, filename_tz="America/Denver"
        ).table.sync_consistent.all()
        # ... and the default (Pacific) puts it one hour out, which is
        # inside the tolerance, so make the tolerance the test's edge:
        t_close_pacific = clock.filename_unix(name)
        assert abs(t_close - t_close_pacific) == 3600


def _stale_pair(tmp_path):
    """One good file, then one whose sync_time is 54 days stale."""
    write_corr_file(
        tmp_path / "corr_20260717_150041Z.h5",
        ntimes=60,
        sync_time=1784300441.0 - 30,
    )
    write_corr_file(
        tmp_path / "corr_20260717_151041Z.h5",
        ntimes=60,
        sync_time=1784301041.0 - 30 - 54 * 86400,
        seed=1,
    )
    return tmp_path


class TestTimeBest:
    def test_good_file_uses_header_time(self, tmp_path):
        idx = MetadataIndex(_stale_pair(tmp_path), cache=False)
        good = idx.table[idx.table.file == "corr_20260717_150041Z.h5"]
        np.testing.assert_array_equal(good.time_best, good.time)

    def test_stale_file_uses_filename_estimate(self, tmp_path):
        # time_fname = close time - (rows after this one) x dt, so the
        # last row lands on the filename stamp and earlier rows are
        # spaced by the integration time.
        idx = MetadataIndex(_stale_pair(tmp_path), cache=False)
        bad = idx.table[idx.table.file == "corr_20260717_151041Z.h5"]
        assert not bad.sync_consistent.any()
        np.testing.assert_array_equal(bad.time_best, bad.time_fname)
        assert bad.time_best.iloc[-1] == pytest.approx(1784301041.0)
        assert np.allclose(np.diff(bad.time_best), 0.5369)

    def test_stale_file_sorts_into_its_true_slot(self, tmp_path):
        # Sorting on header time would put the stale file 54 days
        # before everything else, at the front of the table.
        idx = MetadataIndex(_stale_pair(tmp_path), cache=False)
        assert idx.table.file.iloc[0] == "corr_20260717_150041Z.h5"
        assert idx.table.file.iloc[-1] == "corr_20260717_151041Z.h5"
        assert idx.table.time_best.is_monotonic_increasing

    def test_time_fname_is_nan_without_a_stamp(self, tmp_path):
        write_corr_file(tmp_path / "corr_nostamp.h5", ntimes=4)
        idx = MetadataIndex(tmp_path, cache=False)
        assert idx.table.time_fname.isna().all()
        assert not idx.table.sync_consistent.any()
        assert isinstance(idx.table, pd.DataFrame)
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

Row identity is the ``(file, row)`` pair. Ordering is by ``time_best``,
which is the header time when the file's clock was sane and a
filename-derived estimate otherwise; see :func:`scan_corr_file`.
"""

import fnmatch
import hashlib
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .clock import filename_unix, to_unix_time
from .metadata import CURATED_FIELDS, MISSING, SCALAR_STREAMS, flatten_metadata

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
    """An h5py attribute as a plain Python scalar."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
        where filename and times would be wrong together.
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
    Normalise dtypes and sort.

    Object columns become pure ``str`` with ``MISSING`` for gaps, so a
    column that is a string in one file and absent in another (a root
    attr, say) is identical whether the table came from a scan or from
    the cache. Sorting is on ``time_best``; ties (the ``-1`` suffix
    twins closed in the same second) break on filename then row.
    """
    for col in table.columns:
        if table[col].dtype == object:
            table[col] = table[col].fillna(MISSING).astype(str)
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
        Read and write the sidecar cache (see :meth:`rebuild`).
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
        self.streams = streams
        self.patterns = tuple(patterns)
        self.use_cache = cache
        self.filename_tz = filename_tz
        self.scanner = scanner or scan_corr_file
        self.table = None
        self.from_cache = False
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
        for path in self._files():
            try:
                frames.append(
                    self.scanner(
                        path, streams=streams, filename_tz=self.filename_tz
                    )
                )
            except (OSError, KeyError, ValueError) as exc:
                warnings.warn(f"Skipping {path.name}: {exc}")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_index.py -v`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/index.py tests/test_index.py
git commit -m "feat: per-integration metadata index with time_best"
```

---

### Task 5: `Selection` and the query surface

**Files:**
- Modify: `src/eigsep_data/index.py` (append `Selection`, `_apply_filters`; add `MetadataIndex.select`)
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `MetadataIndex.table`, `clock.to_unix_time`
- Produces:
  - `MetadataIndex.select(*, files=None, time=None, where=None, **filters) -> Selection`
  - `Selection` with `.meta` (DataFrame), `.index`, `.nrows`, `.files`, `.file_counts() -> pd.Series`, `.select(...)`, `.summary() -> str`, `.visits(gap_s=600) -> np.ndarray`, `.load(keys=None, time_avg=1, missing="raise")` (body lands in Task 8)
  - `files=` is a glob string, a **list** of globs, or a 2-**tuple** `(lo, hi)` inclusive lexical range on basenames
  - `time=(lo, hi)` is half-open on `time_best`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selection.py
"""Tests for eigsep_data.index.Selection."""

import numpy as np
import pandas as pd
import pytest

from eigsep_data.index import MetadataIndex

from conftest import write_corr_file


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
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFNOFF")
        assert sel.nrows == 30  # 20 + 10, not 35

    def test_no_arguments_selects_everything(self, corr_dir):
        assert MetadataIndex(corr_dir, cache=False).select().nrows == 180

    def test_filters_by_time_range(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        lo = float(idx.table.time_best.min())
        sel = idx.select(time=(lo, lo + 9.5 * 0.5369))
        assert sel.nrows == 10

    def test_filters_by_filename_glob(self, corr_dir):
        # Filename selectors matter because 12% of deployment-5 files
        # have header times wrong by days -- a time range on header
        # time cannot reach them, but a glob can.
        sel = MetadataIndex(corr_dir, cache=False).select(
            files="corr_20260717_1500*"
        )
        assert set(sel.meta.file) == {"corr_20260717_150041Z.h5"}

    def test_filters_by_filename_range(self, corr_dir):
        # motor_scan_20260717.ipynb selected its raster block as
        # RASTER_LO <= name <= RASTER_HI. A tuple is that range,
        # inclusive at both ends.
        sel = MetadataIndex(corr_dir, cache=False).select(
            files=("corr_20260717_150041Z.h5", "corr_20260717_151041Z.h5")
        )
        assert sel.files == [
            "corr_20260717_150041Z.h5",
            "corr_20260717_151041Z.h5",
        ]

    def test_glob_list_is_not_a_range(self, corr_dir):
        # A list is a set of globs; only a tuple is a range.
        sel = MetadataIndex(corr_dir, cache=False).select(
            files=["corr_*_150041Z.h5", "corr_*_152041Z.h5"]
        )
        assert len(sel.files) == 2

    def test_filters_by_run_tag(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            run_tag="motor_scan"
        )
        assert sel.nrows == 60

    def test_filters_by_root_attr(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(filter_phase="C")
        assert sel.nrows == 180

    def test_filters_by_bool_column(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            sync_consistent=True
        )
        assert sel.nrows == 180

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


class TestTimeBestWindow:
    def test_time_window_reaches_a_stale_sync_file(self, tmp_path):
        # The stale file's header times are 54 days early; its
        # filename estimate is right. A window around the estimate must
        # find it, which a window on header time never could.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            sync_time=1784300441.0 - 30,
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=60,
            sync_time=1784301041.0 - 30 - 54 * 86400,
            seed=1,
        )
        idx = MetadataIndex(tmp_path, cache=False)
        sel = idx.select(time=(1784301041.0 - 5, 1784301041.0 + 1))
        assert sel.files == ["corr_20260717_151041Z.h5"]
        assert not sel.meta.sync_consistent.any()


class TestSummary:
    def test_reports_what_each_filter_removed(self, corr_dir):
        # load_gated silently dropped every None row and every file with
        # no stream. Making the exclusions visible is the point.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        text = sel.summary()
        assert "180" in text and "20" in text
        assert "rfswitch" in text

    def test_reports_rows_with_estimated_times(self, corr_dir):
        text = MetadataIndex(corr_dir, cache=False).select().summary()
        assert "0 rows have sync_consistent=False" in text
        assert "filename" in text


class TestFileCounts:
    def test_rows_per_file_in_selection_order(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFANT", "RFAMB"]
        )
        counts = sel.file_counts()
        assert isinstance(counts, pd.Series)
        assert counts.to_dict() == {
            "corr_20260717_150041Z.h5": 20,
            "corr_20260717_152041Z.h5": 30,
        }
        assert list(counts.index) == sel.files


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
        return _apply_filters(
            self.index, self.meta, self.provenance, **kwargs
        )

    def visits(self, gap_s=600):
        """Group rows into contiguous visits separated by *gap_s*."""
        times = self.meta.time_best.to_numpy()
        visits = np.zeros(times.size, dtype=int)
        if times.size > 1:
            visits[1:] = np.cumsum(np.diff(times) > gap_s)
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
        return "\n".join(lines)

    def load(self, keys=None, time_avg=1, missing="raise"):
        """Read the spectra for these integrations; see
        :meth:`eigsep_data.EigsepData.from_selection`."""
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
```

And add to `MetadataIndex`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_selection.py -v`
Expected: 22 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/index.py tests/test_selection.py
git commit -m "feat: Selection query surface with visible filter provenance"
```

---

### Task 6: Sidecar cache with self-invalidation and a superset stream policy

**Files:**
- Modify: `src/eigsep_data/index.py`
- Test: `tests/test_index_cache.py`

**Interfaces:**
- Consumes: `MetadataIndex.table`, `SCHEMA_VERSION`, `CACHE_NAME`
- Produces: `MetadataIndex._fingerprint()`, `MetadataIndex.cached_streams`, cache read/write inside `rebuild(force=False)`
- Policy: the fingerprint covers schema version, patterns, `filename_tz`, the scanner's qualified name, and the `(basename, size, mtime_ns)` manifest. The stream set is a separate attr. A request **hits** when the fingerprint matches and the requested streams are a subset of the cached set (or the cache holds `"all"`); the table then carries every cached column. Otherwise it rebuilds with the **union** and overwrites.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_cache.py
"""The cache must never serve a table that no longer matches the data."""

import pandas as pd

from eigsep_data.index import MetadataIndex

from conftest import write_corr_file


class TestCache:
    def test_second_build_reuses_the_cache_losslessly(self, corr_dir):
        first = MetadataIndex(corr_dir)
        assert first.cache_path.exists()
        second = MetadataIndex(corr_dir)
        assert second.from_cache
        # Whole-table equality, dtypes and column order included: the
        # h5 round trip must not turn None into "None" or NaN into
        # "nan", or reorder columns.
        pd.testing.assert_frame_equal(first.table, second.table)

    def test_narrower_stream_request_hits(self, corr_dir):
        # A cache holding the curated set can answer a motor-only
        # request; the extra columns are just along for the ride.
        MetadataIndex(corr_dir)
        narrow = MetadataIndex(corr_dir, streams=("motor",))
        assert narrow.from_cache
        assert "rfswitch" in narrow.table.columns

    def test_wider_stream_request_rebuilds_with_the_union(self, corr_dir):
        MetadataIndex(corr_dir, streams=("motor",))
        wider = MetadataIndex(corr_dir)
        assert not wider.from_cache
        assert "potmon_pot_az_angle" in wider.table.columns
        # ... and the union is what got cached, so both requests now hit.
        assert MetadataIndex(corr_dir, streams=("motor",)).from_cache
        assert MetadataIndex(corr_dir).from_cache

    def test_all_streams_covers_everything(self, corr_dir):
        MetadataIndex(corr_dir, streams="all")
        assert MetadataIndex(corr_dir).from_cache
        assert MetadataIndex(corr_dir, streams=("lidar",)).from_cache

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

    def test_schema_bump_invalidates(self, corr_dir, monkeypatch):
        MetadataIndex(corr_dir)
        monkeypatch.setattr("eigsep_data.index.SCHEMA_VERSION", 99)
        assert not MetadataIndex(corr_dir).from_cache

    def test_filename_tz_is_part_of_the_identity(self, corr_dir):
        # sync_consistent and time_best depend on it.
        MetadataIndex(corr_dir)
        assert not MetadataIndex(
            corr_dir, filename_tz="America/Denver"
        ).from_cache

    def test_corrupt_cache_is_rebuilt(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        idx.cache_path.write_bytes(b"garbage")
        rebuilt = MetadataIndex(corr_dir)
        assert not rebuilt.from_cache
        assert MetadataIndex(corr_dir).from_cache

    def test_cache_false_never_writes(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        assert not idx.cache_path.exists()

    def test_force_rescans(self, corr_dir):
        MetadataIndex(corr_dir)
        idx = MetadataIndex(corr_dir)
        assert idx.from_cache
        idx.rebuild(force=True)
        assert not idx.from_cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_index_cache.py -v`
Expected: FAIL with `AssertionError` on `second.from_cache` (no cache is written yet)

- [ ] **Step 3: Write the implementation**

Add module-level helpers to `src/eigsep_data/index.py` (`json` and `hashlib` are already imported):

```python
def _stream_key(streams):
    """Canonical form of a stream request: ``"all"`` or a sorted list."""
    if streams == "all":
        return "all"
    if streams is None:
        return sorted(set(CURATED_FIELDS) | set(SCALAR_STREAMS))
    return sorted(set(streams))


def _covers(have, want):
    if have == "all":
        return True
    if want == "all":
        return False
    return set(want) <= set(have)


def _union(a, b):
    if b is None:
        return a
    if a == "all" or b == "all":
        return "all"
    return sorted(set(a) | set(b))
```

Add to `MetadataIndex`:

```python
    def _fingerprint(self):
        """Identity of the inputs the table was built from -- the
        stream set is deliberately *not* part of it (see rebuild)."""
        manifest = [
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in self._files()
        ]
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

    def _read_cache(self, fingerprint):
        """``(table, streams)`` if the cache matches *fingerprint*,
        else ``(None, None)``; the table is ``None`` but *streams* set
        when the fingerprint matches but the columns are not wanted."""
        if not self.cache_path.exists():
            return None, None
        try:
            with h5py.File(self.cache_path, "r") as h5:
                if h5.attrs.get("fingerprint") != fingerprint:
                    return None, None
                have = json.loads(h5.attrs["streams"])
                columns = json.loads(h5.attrs["columns"])
                if not _covers(have, _stream_key(self.streams)):
                    return None, have
                cols = {}
                for name in columns:
                    values = h5["columns"][name][()]
                    if values.dtype.kind == "S":
                        values = np.char.decode(values, "utf-8")
                    cols[name] = values
        except (OSError, KeyError, ValueError):
            return None, None
        return pd.DataFrame(cols), have

    def _write_cache(self, fingerprint, streams):
        try:
            with h5py.File(self.cache_path, "w") as h5:
                h5.attrs["fingerprint"] = fingerprint
                h5.attrs["schema"] = SCHEMA_VERSION
                h5.attrs["streams"] = json.dumps(streams)
                h5.attrs["columns"] = json.dumps(list(self.table.columns))
                grp = h5.create_group("columns")
                for name in self.table.columns:
                    values = self.table[name].to_numpy()
                    if values.dtype == object:
                        values = np.char.encode(
                            values.astype(str), "utf-8"
                        )
                    grp.create_dataset(name, data=values)
        except OSError as exc:
            warnings.warn(f"Could not write index cache: {exc}")
```

Rewrite `rebuild` to consult the cache:

```python
    def rebuild(self, force=False):
        """
        Build :attr:`table`, reusing the sidecar cache when it matches.

        The cache is keyed on the file manifest (name, size, mtime),
        the schema version, the patterns, the filename zone and the
        scanner. The stream set is stored alongside: a request is
        served from the cache when its streams are a subset of what
        was cached, and otherwise rebuilds with the union, so widening
        and narrowing never thrash.

        Parameters
        ----------
        force : bool
            Ignore any existing cache and rescan.
        """
        fingerprint = self._fingerprint()
        wanted = _stream_key(self.streams)
        self.from_cache = False
        have = None
        if self.use_cache and not force:
            cached, have = self._read_cache(fingerprint)
            if cached is not None:
                self.table = cached
                self.cached_streams = have
                self.from_cache = True
                return
        streams = _union(wanted, have)
        self.table = self._scan(streams)
        self.cached_streams = streams
        if self.use_cache:
            self._write_cache(fingerprint, streams)
```

`_scan(streams)` from Task 4 already takes the stream set as an argument; `_stream_key` returns a list of names that `flatten_metadata` treats exactly like `None` (curated fields for known streams), so the curated request and its canonical form scan to the same columns.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_index_cache.py tests/test_index.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/index.py tests/test_index_cache.py
git commit -m "feat: sidecar index cache with superset stream policy"
```

---

### Task 7: Characterization test for `extract_beam_mapping_data`

This function has **zero coverage** and feeds the beam pipeline. Pin its current behaviour *before* touching it, so the rewrite in Task 9 is constrained to reproduce it exactly -- including that `cross` is complex, which the fixture's `(n, 1024, 2)` int32 storage now makes visible.

**Files:**
- Test: `tests/test_beam_extraction_characterization.py`

**Interfaces:**
- Consumes: `eigsep_data.data.extract_beam_mapping_data` as it exists today
- Produces: nothing (a gate for Task 9)

- [ ] **Step 1: Write the characterization test**

```python
# tests/test_beam_extraction_characterization.py
"""Pins extract_beam_mapping_data's current output.

Written against the implementation as it stands, BEFORE the rewrite onto
MetadataIndex. Its job is to fail loudly if the consolidation changes any
value the beam pipeline consumes. If a value here needs to change, that
is a deliberate decision to argue for -- not a test to relax quietly.
"""

import h5py
import numpy as np

from eigsep_data.data import extract_beam_mapping_data

from conftest import write_corr_file

# 1784320080.0 == 2026-07-17 20:28:00Z, matching the first filename so
# sync_consistent is True and the fixture reads coherently.
BEAM_T0 = 1784320080.0
BEAM_FILES = ["corr_20260717_202800Z.h5", "corr_20260717_203000Z.h5"]


def _beam_dir(tmp_path):
    """Two files with motor/potmon/imu_el, as the beam pipeline sees."""
    for i, name in enumerate(BEAM_FILES):
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


def _extract(tmp_path, **kw):
    return extract_beam_mapping_data(
        _beam_dir(tmp_path),
        start_time=BEAM_T0 - 1,
        end_time=BEAM_T0 + 600,
        **kw,
    )


class TestExtractBeamMappingDataContract:
    def test_returns_the_documented_keys(self, tmp_path):
        out = _extract(tmp_path)
        assert sorted(out) == sorted(
            [
                "times",
                "freqs",
                "sky",
                "ground",
                "cross",
                "el_pos",
                "az_pos",
                "pot_az_angle",
                "imu_el_deg",
                "imu_accel",
            ]
        )

    def test_shapes_and_ordering(self, tmp_path):
        out = _extract(tmp_path)
        assert out["times"].shape == (40,)
        assert out["sky"].shape == (40, 1024)
        assert out["cross"].shape == (40, 1024)
        assert out["imu_accel"].shape == (40, 3)
        # Chronological, across file boundaries.
        assert (np.diff(out["times"]) > 0).all()

    def test_cross_is_complex_from_re_im_pairs(self, tmp_path):
        # Real deployment-5 files store the cross as (n, 1024, 2) int32
        # and read_hdf5 rebuilds complex. Anything that reads raw h5py
        # must do the same, or the beam pipeline gets an int32 cube.
        out = _extract(tmp_path)
        assert out["cross"].dtype.kind == "c"
        raw = []
        for name in BEAM_FILES:
            with h5py.File(tmp_path / name, "r") as h5:
                raw.append(h5["data"]["04"][()])
        raw = np.concatenate(raw)
        np.testing.assert_array_equal(
            out["cross"], raw[..., 0] + 1j * raw[..., 1]
        )

    def test_values_are_stable(self, tmp_path):
        # Golden values: any drift in selection, sorting, or metadata
        # alignment shows up here.
        out = _extract(tmp_path)
        np.testing.assert_allclose(out["az_pos"][:5], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(out["el_pos"][:5], [0, 0, 0, 0, 0])
        np.testing.assert_allclose(
            out["pot_az_angle"][:5], [10, 11, 12, 13, 14]
        )
        np.testing.assert_allclose(out["times"][0], BEAM_T0)
        assert np.isfinite(out["sky"]).all()
        assert np.isfinite(out["imu_el_deg"]).all()

    def test_sweep_slice_applies_to_every_per_sample_array(self, tmp_path):
        out = _extract(tmp_path, sweep_slice=slice(5, 15))
        assert out["times"].shape == (10,)
        assert out["sky"].shape == (10, 1024)
        assert out["cross"].shape == (10, 1024)
        assert out["freqs"].shape == (1024,)  # never sliced
```

- [ ] **Step 2: Run the test — it must PASS on current code**

Run: `.venv/bin/python -m pytest tests/test_beam_extraction_characterization.py -v`
Expected: **5 passed.** This is the one test in this plan that must be green before any implementation. If it fails, the golden values are wrong — fix them to match current behaviour, do not change `data.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_beam_extraction_characterization.py
git commit -m "test: characterize extract_beam_mapping_data before rewiring"
```

---

### Task 8: `EigsepData` carries metadata; `from_selection`

**Files:**
- Modify: `src/eigsep_data/data.py` (`EigsepData` dataclass, new `from_selection`, `slice`; module helpers)
- Test: `tests/test_eigsep_data.py`

**Interfaces:**
- Consumes: `Selection`
- Produces:
  - `EigsepData(data, acc_cnt, times, freq, meta=None)`; `times` is `meta.time_best`, `meta.time` keeps the raw header time
  - `EigsepData.from_selection(selection, keys=None, time_avg=1, missing="raise") -> EigsepData`
  - `Selection.load(...)` (stub from Task 5) now works
  - helpers `_as_spectra`, `_avg_rows`, `_nan_block`, `_estimate_bytes`, `_mem_available`, `_warn_if_large`

Behaviour:
- **Complex reconstruction** identical to `read_hdf5`: `ndim >= 2 and shape[-1] == 2 and dtype.kind == "i"` gives `re + 1j*im` as complex128.
- **Row order** equals selection order. Positions are gathered per file while reading and the blocks are placed with the inverse permutation; nothing assumes files are contiguous in time.
- **`missing="raise"`** (default): a requested key absent from any selected file raises `KeyError` naming those files and suggesting a `data_keys` or `filter_phase` filter. **`missing="nan"`** NaN-fills that file's rows (complex NaN for a cross), which upcasts autos to float64.
- **`time_avg > 1`**: within each file, consecutive selected rows are averaged in blocks of `time_avg`; the remainder is dropped and counted in one warning. Autos become float32, crosses complex64. `meta` keeps the first row of each block with `time`, `time_best` and `acc_cnt` replaced by block means; `times` and `acc_cnt` likewise. `time_avg == 1` keeps native dtypes (int32 autos).
- **Memory guard**: estimate bytes from rows, `nchan`, per-key width (4 B auto int32, 16 B complex128; 4/8 when averaged) and warn above half of `MemAvailable` from `/proc/meminfo`; silent where that file does not exist. Peak is about twice the estimate during concatenation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eigsep_data.py
"""EigsepData must carry metadata aligned to its own time axis."""

import warnings

import h5py
import numpy as np
import pytest

from eigsep_data.data import EigsepData
from eigsep_data.index import MetadataIndex

from conftest import NCHAN, write_corr_file


class TestFromSelection:
    def test_reads_only_the_selected_rows(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert d.data["0"].shape == (20, NCHAN)
        assert d.times.shape == (20,)

    def test_meta_is_aligned_row_for_row(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFAMB", "RFNON"]
        )
        d = EigsepData.from_selection(sel, keys=["0"])
        assert len(d.meta) == len(d.times)
        np.testing.assert_allclose(d.meta.time_best.to_numpy(), d.times)
        assert set(d.meta.rfswitch) == {"RFAMB", "RFNON"}

    def test_reads_only_requested_keys(self, corr_dir):
        # Reading every key is what OOMed the 16 GB laptop; the loader
        # must never pull keys the caller did not ask for.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert list(d.data) == ["0"]

    def test_native_dtypes_at_full_resolution(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert d.data["0"].dtype == np.int32

    def test_times_are_not_shifted(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert abs(d.times[0] - 1.7843e9) < 1.0

    def test_cross_is_complex(self, corr_dir):
        # The reader's rule: (n, nchan, 2) int32 is (re, im).
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["04"])
        assert d.data["04"].dtype.kind == "c"
        assert d.data["04"].shape == (20, NCHAN)
        with h5py.File(corr_dir / "corr_20260717_150041Z.h5", "r") as h5:
            raw = h5["data"]["04"][:20]
        np.testing.assert_array_equal(
            d.data["04"], raw[..., 0] + 1j * raw[..., 1]
        )

    def test_empty_selection_raises_clearly(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="VNAO")
        with pytest.raises(ValueError, match="no integrations"):
            EigsepData.from_selection(sel, keys=["0"])

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


class TestRowOrder:
    def test_interleaved_files_come_back_in_selection_order(self, tmp_path):
        # Two files whose integrations alternate in time. Concatenating
        # per file and then reordering with a permutation computed on
        # the other axis would scramble spectra against times; this
        # pins the inverse permutation.
        t0 = 1784300441.0 - 20
        names = ["corr_20260717_150041Z.h5", "corr_20260717_150042Z.h5"]
        for i, name in enumerate(names):
            write_corr_file(
                tmp_path / name,
                ntimes=20,
                sync_time=t0 + 0.25 * i,
                rfswitch=["RFANT"] * 20,
                seed=i,
            )
        sel = MetadataIndex(tmp_path, cache=False).select()
        assert len(set(sel.meta.file.iloc[:2])) == 2  # they interleave
        d = EigsepData.from_selection(sel, keys=["0"])
        assert (np.diff(d.times) > 0).all()
        raw = {}
        for name in names:
            with h5py.File(tmp_path / name, "r") as h5:
                raw[name] = h5["data"]["0"][()]
        for i, (name, row) in enumerate(zip(d.meta.file, d.meta.row)):
            np.testing.assert_array_equal(d.data["0"][i], raw[name][row])


def _mixed_keys_dir(tmp_path):
    """Phase change inside a window: the second file has no cross."""
    write_corr_file(
        tmp_path / "corr_20260717_150041Z.h5",
        ntimes=10,
        keys=("0", "4", "04"),
    )
    write_corr_file(
        tmp_path / "corr_20260717_151041Z.h5",
        ntimes=10,
        keys=("0", "4"),
        sync_time=1.7843e9 + 600,
        seed=1,
    )
    return tmp_path


class TestMissingKeys:
    def test_raise_names_the_files(self, tmp_path):
        sel = MetadataIndex(_mixed_keys_dir(tmp_path), cache=False).select()
        with pytest.raises(KeyError, match="151041Z") as info:
            EigsepData.from_selection(sel, keys=["04"])
        assert "data_keys" in str(info.value)

    def test_nan_fills_that_files_rows(self, tmp_path):
        sel = MetadataIndex(_mixed_keys_dir(tmp_path), cache=False).select()
        d = EigsepData.from_selection(sel, keys=["04"], missing="nan")
        assert d.data["04"].shape == (20, NCHAN)
        assert d.data["04"].dtype.kind == "c"
        gap = (d.meta.file == "corr_20260717_151041Z.h5").to_numpy()
        assert np.isnan(d.data["04"][gap]).all()
        assert np.isfinite(d.data["04"][~gap]).all()

    def test_default_keys_are_the_common_ones(self, tmp_path):
        sel = MetadataIndex(_mixed_keys_dir(tmp_path), cache=False).select()
        d = EigsepData.from_selection(sel)
        assert sorted(d.data) == ["0", "4"]


class TestTimeAvg:
    def test_blocks_within_a_file(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with pytest.warns(UserWarning, match="dropped 4"):
            d = EigsepData.from_selection(
                sel, keys=["0", "04"], time_avg=8
            )
        assert d.data["0"].shape == (2, NCHAN)
        assert d.data["0"].dtype == np.float32
        assert d.data["04"].dtype == np.complex64
        assert len(d.meta) == 2
        assert list(d.meta.row) == [0, 8]
        np.testing.assert_allclose(d.times[0], 1.7843e9 + 3.5 * 0.5369)
        np.testing.assert_allclose(d.acc_cnt, [3.5, 11.5])

    def test_block_mean_matches_numpy(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        full = EigsepData.from_selection(sel, keys=["0"])
        avg = EigsepData.from_selection(sel, keys=["0"], time_avg=4)
        expected = full.data["0"].reshape(5, 4, NCHAN).mean(axis=1)
        np.testing.assert_allclose(avg.data["0"], expected, rtol=1e-6)

    def test_never_averages_across_files(self, corr_dir):
        # 20 RFANT rows in one file, 30 RFAMB in another: 2 + 3 blocks
        # of 8, never a block straddling the boundary.
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFANT", "RFAMB"]
        )
        with pytest.warns(UserWarning, match="dropped 10"):
            d = EigsepData.from_selection(sel, keys=["0"], time_avg=8)
        assert d.data["0"].shape == (5, NCHAN)
        assert d.meta.file.tolist() == (
            ["corr_20260717_150041Z.h5"] * 2
            + ["corr_20260717_152041Z.h5"] * 3
        )


class TestMemoryGuard:
    def test_warns_when_load_exceeds_half_of_available(
        self, corr_dir, monkeypatch
    ):
        monkeypatch.setattr("eigsep_data.data._mem_available", lambda: 1000)
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with pytest.warns(UserWarning, match="available"):
            EigsepData.from_selection(sel, keys=["0"])

    def test_silent_without_meminfo(self, corr_dir, monkeypatch):
        monkeypatch.setattr("eigsep_data.data._mem_available", lambda: None)
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            EigsepData.from_selection(sel, keys=["0"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eigsep_data.py -v`
Expected: FAIL with `AttributeError: type object 'EigsepData' has no attribute 'from_selection'`

- [ ] **Step 3: Write the implementation**

Add `import pandas as pd` to `src/eigsep_data/data.py`, then the module-level helpers (near the top, after the imports):

```python
#: Per-row stamps that are averaged when ``time_avg > 1``.
_AVG_COLS = ("time", "time_best", "acc_cnt")


def _as_spectra(arr):
    """Apply read_hdf5's storage rule: (re, im) int32 pairs are complex."""
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[-1] == 2 and arr.dtype.kind == "i":
        return arr[..., 0].astype(np.float64) + 1j * arr[..., 1].astype(
            np.float64
        )
    return arr


def _avg_rows(block, time_avg):
    """Mean over consecutive groups of *time_avg* rows; float32 for
    autos, complex64 for crosses. The caller trims the remainder."""
    nout = block.shape[0] // time_avg
    out = block.reshape((nout, time_avg) + block.shape[1:]).mean(axis=1)
    return out.astype(np.complex64 if out.dtype.kind == "c" else np.float32)


def _nan_block(nrows, nchan, key):
    dtype = np.complex128 if len(key) == 2 else np.float64
    return np.full((nrows, nchan), np.nan, dtype=dtype)


def _estimate_bytes(nrows, nchan, keys, time_avg):
    if time_avg > 1:
        width = sum(8 if len(k) == 2 else 4 for k in keys)
    else:
        width = sum(16 if len(k) == 2 else 4 for k in keys)
    return (nrows // time_avg) * nchan * width


def _mem_available():
    """Bytes of MemAvailable from /proc/meminfo, or None."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _warn_if_large(nbytes):
    avail = _mem_available()
    if avail is not None and nbytes > avail / 2:
        warnings.warn(
            f"Loading ~{nbytes / 1e9:.1f} GB of spectra with "
            f"{avail / 1e9:.1f} GB available; narrow the selection, "
            "drop keys, or pass time_avg."
        )
```

Then the dataclass and constructor:

```python
@dataclass
class EigsepData:

    data: dict[str, np.ndarray] = None
    acc_cnt: np.ndarray = None
    #: ``time_best`` of each row: header time where the file's clock
    #: was sane, filename-derived estimate otherwise. The raw header
    #: time is ``meta.time``.
    times: np.ndarray = None
    freq: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 250, num=1024, endpoint=False)
    )
    #: Per-integration metadata, one row per entry in ``times``.
    meta: pd.DataFrame = None

    @classmethod
    def from_selection(
        cls, selection, keys=None, time_avg=1, missing="raise"
    ):
        """
        Read the spectra for the integrations in *selection*.

        Only the chosen rows of the chosen keys are read, so selecting a
        few percent of a deployment costs a few percent of the I/O.
        Crosses come back complex, exactly as ``read_hdf5`` returns
        them. Output rows are in selection order (``time_best``).

        Parameters
        ----------
        selection : eigsep_data.index.Selection
        keys : list of str or None
            Data keys to read. ``None`` reads every key common to the
            selected files.
        time_avg : int
            Average this many consecutive selected rows *within each
            file* into one output row; a trailing remainder is dropped
            with a warning. Autos become float32 and crosses complex64,
            which is what lets a whole day fit in a laptop.
        missing : {"raise", "nan"}
            What to do when a requested key is absent from one of the
            selected files (a wiring-phase change inside the window).

        Returns
        -------
        EigsepData
        """
        if missing not in ("raise", "nan"):
            raise ValueError("missing must be 'raise' or 'nan'")
        time_avg = int(time_avg)
        if time_avg < 1:
            raise ValueError("time_avg must be >= 1")
        meta = selection.meta.reset_index(drop=True)
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
        keys = list(keys)
        nchan = 1024
        if "nchan" in meta and np.isfinite(meta.nchan.iloc[0]):
            nchan = int(meta.nchan.iloc[0])
        _warn_if_large(_estimate_bytes(len(meta), nchan, keys, time_avg))

        blocks = {k: [] for k in keys}
        positions, stamps = [], []
        absent = {}
        dropped = 0
        freq = None
        for name, group in meta.groupby("file", sort=False):
            pos = group.index.to_numpy()
            rows = group.row.to_numpy()
            keep = (len(rows) // time_avg) * time_avg
            dropped += len(rows) - keep
            if keep == 0:
                continue
            pos, rows = pos[:keep], rows[:keep]
            with h5py.File(data_dir / name, "r") as h5:
                if freq is None and "freqs" in h5["header"]:
                    freq = np.asarray(h5["header"]["freqs"])
                lo, hi = int(rows.min()), int(rows.max()) + 1
                local = rows - lo
                for key in keys:
                    if key not in h5["data"]:
                        absent.setdefault(key, []).append(name)
                        block = _nan_block(keep, nchan, key)
                    else:
                        block = _as_spectra(h5["data"][key][lo:hi])[local]
                    if time_avg > 1:
                        block = _avg_rows(block, time_avg)
                    blocks[key].append(block)
            stamp = group[list(_AVG_COLS)].to_numpy(dtype=float)[:keep]
            if time_avg > 1:
                stamp = stamp.reshape(-1, time_avg, len(_AVG_COLS))
                stamp = stamp.mean(axis=1)
                pos = pos.reshape(-1, time_avg)[:, 0]
            positions.append(pos)
            stamps.append(stamp)

        if absent and missing == "raise":
            detail = "; ".join(
                f"key {k!r} is absent from {sorted(set(v))}"
                for k, v in absent.items()
            )
            raise KeyError(
                f"{detail}. The selection straddles a change in recorded "
                "data keys: narrow it with a data_keys or filter_phase "
                "filter, or pass missing='nan'."
            )
        if not positions:
            raise ValueError(
                f"time_avg={time_avg} exceeds every file's contribution; "
                "nothing to load."
            )
        if dropped:
            warnings.warn(
                f"time_avg={time_avg}: dropped {dropped} trailing rows "
                "that did not fill a block."
            )

        pos = np.concatenate(positions)
        order = np.argsort(pos, kind="stable")  # back to selection order
        data = {
            k: np.concatenate(v, axis=0)[order] for k, v in blocks.items()
        }
        out_meta = meta.iloc[pos[order]].reset_index(drop=True)
        stamp = np.concatenate(stamps)[order]
        for i, col in enumerate(_AVG_COLS):
            out_meta[col] = stamp[:, i]
        return cls(
            data=data,
            acc_cnt=out_meta.acc_cnt.to_numpy(),
            times=out_meta.time_best.to_numpy(),
            freq=freq,
            meta=out_meta,
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

`Selection.load` from Task 5 needs no change.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eigsep_data.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/data.py tests/test_eigsep_data.py
git commit -m "feat: EigsepData.from_selection reads only the chosen rows"
```

---

### Task 9: Rewire the beam extractor; delete `_select_h5_in_range`

**Files:**
- Modify: `src/eigsep_data/data.py` (delete `_select_h5_in_range`; rewrite the body of `extract_beam_mapping_data`)
- Test: `tests/test_beam_extraction_characterization.py` (unchanged — it is the gate)

**Interfaces:**
- Consumes: `MetadataIndex`, `Selection`, `EigsepData.from_selection`
- Produces: `extract_beam_mapping_data(...)` with its **existing** signature and return dict; `cross` complex as before

- [ ] **Step 1: Confirm the gate is green before touching anything**

Run: `.venv/bin/python -m pytest tests/test_beam_extraction_characterization.py -v`
Expected: 5 passed. Do not proceed otherwise.

- [ ] **Step 2: Rewrite the body onto the index**

Replace the body of `extract_beam_mapping_data`. Keep the signature; add these paragraphs to the docstring after the first one:

```
    Selection runs through :class:`eigsep_data.index.MetadataIndex`,
    which writes a sidecar cache ``.eigsep_index.h5`` next to the data
    on first use (later calls are near-instant). A read-only directory
    just skips the write with a warning. Only files whose clock agrees
    with their filename (``sync_consistent``) are considered, which is
    what the old header-time selection did in practice: a stale-clock
    file never fell inside a real window.
```

Body:

```python
    start_unix = to_unix_time(start_time)
    end_unix = to_unix_time(end_time)
    if end_unix <= start_unix:
        raise ValueError("end_time must be later than start_time.")

    from .index import MetadataIndex

    index = MetadataIndex(data_dir, patterns=file_patterns)
    sel = index.select(time=(start_unix, end_unix), sync_consistent=True)
    if sel.nrows == 0:
        raise ValueError(
            "No integrations found inside the requested time range."
        )
    loaded = EigsepData.from_selection(
        sel, keys=[sky_key, ground_key, cross_key], missing="raise"
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

The default curated streams already include `motor`, `potmon` and `imu_el`, so the wrapper shares one cache with every other caller instead of building a private narrower one.

- [ ] **Step 3: Delete `_select_h5_in_range`**

Remove the whole function (currently `src/eigsep_data/data.py:186-297`) — it is private with exactly one caller, now gone. `h5py` stays imported: `from_selection` uses it.

- [ ] **Step 4: Run the gate and the full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass, including the 5 characterization tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/data.py
git commit -m "refactor: beam extraction onto the metadata index"
```

---

### Task 10: `from_path` onto the index; retire the epoch shift; re-express the deployment-4 window

**Files:**
- Modify: `src/eigsep_data/data.py` (`from_path`), `src/eigsep_data/__init__.py`
- Modify: `notebooks/christian/deployment4/explore.ipynb` (the `from_path` cell)
- Test: `tests/test_from_path.py`

**Interfaces:**
- Consumes: `MetadataIndex`, `EigsepData.from_selection`, `clock.to_unix_time`
- Produces:
  - `EigsepData.from_path(path, start_time=None, end_time=None, pacific_to_mountain=None, keys=None)`; selection is on `time_best` in UTC epoch, no longer on filename wall clock
  - `pacific_to_mountain` deprecated: warns and is ignored
  - package exports `to_unix_time`, `format_time`, `EigsepData`, `MetadataIndex`, `Selection`, and the `metadata` and `clock` modules

- [ ] **Step 1: Write the failing test**

```python
# tests/test_from_path.py
"""from_path is sugar over the index; it never shifts epoch."""

import pytest

import eigsep_data
from eigsep_data import data


class TestFromPath:
    def test_directory_loads_everything(self, corr_dir):
        d = data.EigsepData.from_path(corr_dir)
        assert d.times.shape == (180,)
        assert d.meta is not None

    def test_single_file(self, corr_dir):
        d = data.EigsepData.from_path(
            corr_dir / "corr_20260717_152041Z.h5", keys=["0"]
        )
        assert d.times.shape == (60,)
        assert list(d.data) == ["0"]

    def test_window_is_on_epoch(self, corr_dir):
        # Strings go through to_unix_time (naive == UTC) and select on
        # time_best. Header times, not filename wall clock.
        d = data.EigsepData.from_path(
            corr_dir,
            start_time="2026-07-17 15:03:20",  # == 1.7843e9 + 600
            end_time="2026-07-17 15:13:20",
        )
        assert set(d.meta.file) == {"corr_20260717_151041Z.h5"}

    def test_does_not_shift_times(self, corr_dir):
        d = data.EigsepData.from_path(corr_dir)
        assert abs(d.times[0] - 1.7843e9) < 1.0

    def test_pacific_to_mountain_warns_and_is_ignored(self, corr_dir):
        with pytest.warns(DeprecationWarning, match="pacific_to_mountain"):
            d = data.EigsepData.from_path(corr_dir, pacific_to_mountain=True)
        # Crucially: it warns AND does nothing. Deployment-5 epochs are
        # verified correct UTC; adding 3600 s corrupts them.
        assert abs(d.times[0] - 1.7843e9) < 1.0

    def test_package_exports(self):
        for name in (
            "EigsepData",
            "MetadataIndex",
            "Selection",
            "to_unix_time",
            "format_time",
            "metadata",
            "clock",
        ):
            assert hasattr(eigsep_data, name), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_from_path.py -v`
Expected: FAIL — `test_directory_loads_everything` fails on `d.meta is None`, `test_pacific_to_mountain_warns_and_is_ignored` on the missing warning, `test_package_exports` on `format_time`

- [ ] **Step 3: Write the implementation**

Replace `from_path` in `src/eigsep_data/data.py` wholesale (the interim `.replace(tzinfo=None)` edits from Task 2 go with it):

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

        Sugar over :class:`eigsep_data.index.MetadataIndex` and
        :meth:`from_selection`. Selection is on ``time_best`` in UTC
        epoch -- no longer on the filename wall clock, which is what
        this method filtered on before deployment 5. A directory gets
        a ``.eigsep_index.h5`` sidecar cache on first use.

        Parameters
        ----------
        path : Path
            Directory of corr files, or a single file.
        start_time, end_time : str or float, optional
            Half-open window, passed through
            :func:`eigsep_data.clock.to_unix_time`; naive strings are
            UTC.
        pacific_to_mountain : bool, optional
            **Deprecated and ignored.** Times are Unix epoch and are
            never shifted. The old +3600 s was a display convention for
            deployment 1-4's Pacific-stamped filenames; use
            :func:`eigsep_data.clock.format_time` with
            ``tz="America/Denver"`` instead.
        keys : list of str, optional
            Data keys to read; ``None`` reads every key common to the
            selected files.

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
            lo = -np.inf if start_time is None else to_unix_time(start_time)
            hi = np.inf if end_time is None else to_unix_time(end_time)
            filters["time"] = (lo, hi)
        return cls.from_selection(index.select(**filters), keys=keys)
```

Update `src/eigsep_data/__init__.py` (keep the existing JAX `try/except` block below it unchanged):

```python
from .imu import ImuCalibrator, ImuSnapshot, ImuDataset
from .s11 import S11, RawS11
from .clock import to_unix_time, format_time
from .data import EigsepData
from .index import MetadataIndex, Selection
from . import metadata
from . import clock
from . import plot
from . import rfi
```

Update the deployment-4 notebook cell (`notebooks/christian/deployment4/explore.ipynb`, the cell calling `from_path`). The old strings were Pacific wall-clock *filename* times; the same window in epoch is 17:00Z on Jul 19 to 06:59:59Z on Jul 20:

```python
# from_path now selects on epoch (header times), not on the filename
# wall clock. The old window "20250719_100000".."20250719_235959" was
# 10:00-23:59:59 PDT on the filenames, i.e. 17:00Z Jul 19 to 06:59:59Z
# Jul 20; deployment-4 filenames are Pacific (no Z suffix).
data = ed.EigsepData.from_path(
    DATA_DIR,
    start_time="2025-07-19 17:00:00",
    end_time="2025-07-20 06:59:59",
)
# Field notes are in Utah local time; the data stays epoch.
print(ed.format_time(data.times[0], tz="America/Denver"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/eigsep_data/data.py src/eigsep_data/__init__.py \
        tests/test_from_path.py \
        notebooks/christian/deployment4/explore.ipynb
git commit -m "fix: from_path on the index; stop shifting epoch"
```

---

### Task 11: Switch-state awareness in `quicklook`

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

import pytest

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
        with pytest.raises(ValueError, match="VNAO"):
            ql.quicklook(p, state="VNAO")

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

### Task 12: `browse.py`, port the two notebooks, remove the prototype

**Files:**
- Create: `src/eigsep_data/browse.py`
- Modify: `pyproject.toml`
- Modify: `notebooks/christian/deployment5/calibration.ipynb`, `notebooks/christian/deployment5/flipbook.ipynb` (the only two importers of `rf_state_tools`)
- Delete: `notebooks/christian/deployment5/rf_state_tools.py`, `notebooks/christian/deployment5/rfswitch_index.npz`
- Test: `tests/test_browse.py`

**Interfaces:**
- Consumes: `Selection`, `EigsepData.from_selection`
- Produces: `StateBrowser(selection, keys=("0", "4"), vmin=4, vmax=6.6, controls=True)` with `.goto(pos)`, `.nfiles`, `.pos`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_browse.py
"""The browser must work headless, and fail clearly without widgets."""

import matplotlib
import pytest

matplotlib.use("Agg")

from eigsep_data.browse import StateBrowser  # noqa: E402
from eigsep_data.index import MetadataIndex  # noqa: E402


class TestStateBrowser:
    def test_builds_without_widgets(self, corr_dir):
        # Importing the module must not require ipywidgets; only the
        # interactive controls do.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
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
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="VNAO")
        with pytest.raises(ValueError, match="no integrations"):
            StateBrowser(sel, keys=["0"], controls=False)
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
        return name, EigsepData.from_selection(
            sub, keys=self.keys, missing="nan"
        )

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
            mag = np.abs(spec)
            mean = np.nanmean(mag, axis=0)
            lo, hi = np.nanpercentile(mag, [10, 90], axis=0)
            (line,) = self.ax_s.plot(
                loaded.freq, mean, lw=1, label=f"key {key}"
            )
            self.ax_s.fill_between(
                loaded.freq,
                lo,
                hi,
                alpha=0.25,
                color=line.get_color(),
                lw=0,
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
            0,
            0,
            self.nfiles - 1,
            description="file",
            layout=widgets.Layout(width="55%"),
        )
        play = widgets.Play(
            interval=400, min=0, max=self.nfiles - 1, step=1
        )
        widgets.jslink((play, "value"), (slider, "value"))
        slider.observe(lambda ch: self.goto(ch["new"]), names="value")
        display(widgets.HBox([play, slider]))
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

- [ ] **Step 4: Port the two notebooks off the prototype**

`grep -rn "rf_state_tools\|rfswitch_index" --include=*.ipynb --include=*.py .` finds exactly `calibration.ipynb` and `flipbook.ipynb` (plus the prototype itself and `docs/`). Both notebooks are output-stripped by pre-commit, so editing the source cells is safe.

`flipbook.ipynb`: replace the import and the two `StateBrowser` cells with

```python
from eigsep_data import MetadataIndex
from eigsep_data.browse import StateBrowser

DATA_DIR = "data/deployment5_filtered"  # relative to the repo root
idx = MetadataIndex(DATA_DIR)  # ~25 s the first time, cached after
```

```python
b = StateBrowser(idx.select(rfswitch="RFANT"), keys=["0", "4"])
# phases A/B have other live keys, e.g.:
#   StateBrowser(idx.select(rfswitch="RFANT", filter_phase="B"),
#                keys=["3", "4"])
```

```python
# load-only / noise-only / no-metadata browsing
# b_load = StateBrowser(idx.select(rfswitch="RFAMB"), keys=["0", "4"])
# b_noise = StateBrowser(idx.select(rfswitch="RFNON"), keys=["0", "4"])
# b_gap = StateBrowser(idx.select(rfswitch="MISSING"), keys=["0", "4"])
```

`calibration.ipynb`: replace the import and the `load_gated` cell with an adapter that yields the same dict the later cells consume (`spec`, `t`, `int_t`, `t_load`, `freqs`, `visit`), so nothing below the load cell changes:

```python
from eigsep_data import MetadataIndex

DATA_DIR = "data/deployment5_filtered"
idx = MetadataIndex(DATA_DIR)
```

```python
KEYS = ["0", "4"]
PATTERNS = None          # e.g. ["corr_20260717*"] for just the big cal day
GAP_S = 600              # new visit after this many seconds without cal data


def gated(state):
    sel = idx.select(rfswitch=state, files=PATTERNS)
    print(sel.summary())
    d = sel.load(keys=KEYS)
    return {
        "spec": {k: v.astype(np.float32) for k, v in d.data.items()},
        "t": d.times,                                   # time_best
        "int_t": d.meta.integration_time.to_numpy(),
        "t_load": d.meta.tempctrl_load_T_now.to_numpy(),
        "freqs": d.freq,
        "visit": sel.visits(gap_s=GAP_S),
    }


amb = gated("RFAMB")
non = gated("RFNON")
freqs = amb["freqs"]
for name, c in [("RFAMB", amb), ("RFNON", non)]:
    t0 = datetime.fromtimestamp(c["t"][0], tz=timezone.utc)
    t1 = datetime.fromtimestamp(c["t"][-1], tz=timezone.utc)
    print(
        f"{name}: {len(c['t'])} rows, {c['visit'].max() + 1} visits, "
        f"{t0:%b %d %H:%M} - {t1:%b %d %H:%M} UTC"
    )
```

Note the improvement over the prototype: `t` is now per-integration `time_best` rather than one file-close time shared by every row of a file, so visit boundaries and the stability-vs-time plot sharpen. Update the notebook's leading markdown cell to say the rows come from `MetadataIndex`, not `rfswitch_index.npz`.

- [ ] **Step 5: Run the whole suite, then delete the prototype**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass

```bash
git rm notebooks/christian/deployment5/rf_state_tools.py \
       notebooks/christian/deployment5/rfswitch_index.npz
```

Then check nothing still imports it:

```bash
grep -rn "rf_state_tools\|rfswitch_index" --include=*.ipynb --include=*.py . \
  | grep -v '^./build\|^./.venv'
```

Expected: no hits outside `docs/`.

- [ ] **Step 6: Commit**

```bash
git add -A src/eigsep_data tests pyproject.toml notebooks
git commit -m "feat: StateBrowser on Selection; retire rf_state_tools"
```

---

## Verification

After Task 12:

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m flake8 src/eigsep_data tests
.venv/bin/python -m black --check src/eigsep_data tests
```

Then a real-data smoke test, which is the first thing this whole plan was for:

```python
from eigsep_data import MetadataIndex

idx = MetadataIndex("data/deployment5_filtered")  # ~25 s cold, then cached
sel = idx.select(rfswitch="RFAMB", files="corr_20260717*")
print(sel.summary())  # per-filter removals + the sync_consistent line
print(sel.file_counts().head())
d = sel.load(keys=["0", "4"])
print(d.data["4"].shape, d.meta.rfswitch.unique())

# root attrs from filter_corr_keys.py are queryable from day one
print(idx.select(filter_phase="C").nrows)

# a whole day at 8x averaging fits in a laptop
day = idx.select(files="corr_20260717*").load(
    keys=["0", "4", "04"], time_avg=8
)
print(day.data["04"].dtype, day.data["04"].shape)
```

Expected: the ~1107 `RFAMB` rows of the Jul 17 cal cycle; `summary()` naming how many rows each filter removed and how many rows carry estimated times; `file_counts()` listing the cal-cycle files in time order; a cold build of ~21–26 s followed by ~0.1 s on the next run; `day.data["04"]` complex64.
