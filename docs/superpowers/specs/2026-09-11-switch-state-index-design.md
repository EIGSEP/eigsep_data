# Per-integration metadata index and switch-state gating

**Date:** 2026-09-11
**Status:** design approved, revised after review 2026-09-11 (see the revision
log at the end), not yet implemented

## Problem

Deployment-5 correlator files carry per-integration RF switch state
(`metadata/rfswitch`), but nothing in `eigsep_data` can key off it. Analysis
that needs "the `RFAMB` rows from the Jul 17 cal cycle" has no way to ask for
them.

The same gap exists for every other per-integration stream. The repo currently
holds **four independent hand-rolled implementations** of scan-files →
parse-metadata-JSON → select-rows:

| copy | location |
|---|---|
| `_select_h5_in_range` + `extract_beam_mapping_data` | `src/eigsep_data/data.py` |
| `StateIndex` / `load_gated` | `notebooks/christian/deployment5/rf_state_tools.py` |
| `parse_records` / `select_files` / `extract` | `notebooks/christian/deployment5/motor_scan_20260717.ipynb` |
| phase-boundary scan | `data/deployment5_filtered/filter_corr_keys.py` |

`EigsepData.from_path` drops metadata entirely; `quicklook` ignores it.

## Decisions

1. **General per-integration metadata axis**, with switch state as a
   first-class instance — not an rfswitch-only module. The streams are
   structurally identical, so the plumbing is written once.
2. **Full consolidation.** One selection/loading core; existing loaders become
   thin wrappers over it.
3. **Index-first, two-stage** (query the table, then read only the chosen rows)
   rather than eager-load-then-mask.
4. **Auto sidecar cache**, self-invalidating, gitignored.
5. **Derived index now; asserted annotations deferred**, with three seams built
   in so the annotation layer lands later without a redesign.

## Architecture

Four modules, split so the pure logic is testable without touching a disk:

| module | responsibility | I/O |
|---|---|---|
| `clock.py` | `to_unix_time`, `format_time`, `parse_filename_time` / `filename_unix` with the per-deployment filename zone | none |
| `metadata.py` | flatten one file's metadata group → named columns; missing/None policy | none — dicts in, arrays out |
| `index.py` | `scan_corr_file`, `MetadataIndex` (scan, cache, invalidate), `Selection` (query result) | headers + metadata only |
| `data.py` | `EigsepData` carrying `.meta`; `from_selection()` reads chosen rows | spectra |

Import direction: `data.py` imports `index.py`; `index.py` imports only
`clock.py` and `metadata.py`. The one exception is `Selection.load()`, sugar
over `EigsepData.from_selection` with a function-level import. `to_unix_time`
lives in `clock.py` and is re-exported from `data.py` so existing imports keep
working; the filename parser exists exactly once.

The per-file scan is a plain function, `scan_corr_file(path, streams,
filename_tz) -> DataFrame`, and `MetadataIndex(scanner=...)` accepts a
replacement. That is the seam for indexing a different file kind later — the
S11 sweeps, whose header carries `metadata_snapshot_unix` rather than
`times` — without a second cache, query and summary implementation.

```python
idx = MetadataIndex("data/deployment5_filtered")       # ~64 s cold, ~5 s cached

sel = idx.select(rfswitch="RFANT", run_tag="motor_scan",
                 time=("2026-07-17 20:28", "2026-07-17 21:28"))
sel.summary()                    # rows in, rows out, what each filter removed,
                                 # and how many rows have estimated times
sel.visits(gap_s=600)
sel.file_counts()                # rows per file, in time order

d = EigsepData.from_selection(sel, keys=["0", "4"])    # only now are spectra read
d.meta.rfswitch                  # aligned row-for-row with d.times

# whole-day browsing that fits in memory
sel = idx.select(files="corr_20260717*")
d = sel.load(keys=["0", "4", "04"], time_avg=8)        # float32 / complex64
```

Filters accept a scalar, a list, a `(lo, hi)` time range, a filename glob or
list of globs, a `(lo, hi)` filename range (a 2-tuple of plain names with no
glob characters; inclusive, lexical on basenames — the idiom
`motor_scan_20260717.ipynb` used), or `where=<callable>`. `select()`
composes: `sel.select(...)` narrows further.

## What the files actually contain

Confirmed by reading `eigsep_observing.io` (the writer) and probing the data,
not assumed:

- **1:1 alignment is a producer contract.** `File._insert_sample` back-fills a
  stream appearing mid-buffer with `[None] * counter` and appends `None` for a
  stream missing from a sample, explicitly "so indices align 1:1 with samples".
  Indexing metadata by row number is guaranteed.
- **`rfswitch` is a list of plain strings; every other stream is a list of
  dicts.** `avg_metadata` routes `rfswitch` to `_avg_rfswitch_metadata` (returns
  a bare state name) and everything else to `_avg_sensor_values`. Special-casing
  it is correct, not defensive.
- **`None` means "no reading in this integration window"** — writer padding, not
  an error. Hence `adc_stats` at 233/240 `None` (sparse cadence). Distinct from
  an all-fields-`None` dict (`imu_az`, `lidar`, `tempctrl_lna`), which means the
  stream publishes but the sensor is offline.
- **`integration_time` is not constant**: both 0.2684 s and 0.5369 s occur
  within deployment 5 (208 and 94 of 302 sampled files).
- **Gap-fill rows exist as a writer path** (`add_data` inserts zero-spectra,
  no-metadata samples on an `acc_cnt` jump) but are **absent here**: 0 of 9600
  sampled Jul-17 rows. Cheap detector, no machinery.

### The state vocabulary is open

Counted across a 1-in-9 sample of the deployment — **17 distinct values**
(the 207 stream-absent files scale to ~1860, matching the ~1842 the prototype
docstring reports):

```
RFANT 81464 | None 2219 | RFAMB 1107 | RFNON 816 | RFSP1 244 | VNARF 214
<stream absent> 207 files | VNAL 109 | UNKNOWN 101 | VNAS 87 | VNANOFF 66
VNANON 66 | VNAO 63 | VNAANT 62 | VNASP1 62 | VNAAMB 61 | RFNOFF 16
```

`rfswitch` is therefore an **open category, never a closed enum**.
`rf_state_tools` does `self.states.index(state)`, which raises on anything its
August scan did not see — the upstream test fixtures already use `RFNOFF`.

Note also that `None` (2219 rows) outnumbers `UNKNOWN` (101) by 22×. The
prototype's docstring frames `UNKNOWN` as what swallows unusable rows; in fact
the dominant exclusion is writer `None`-padding plus 207 sampled files with no
stream at all. `load_gated(state="RFANT")` drops all of it silently. This is why
`select()` reports what each filter removed.

### Transition guards are integration-time-dependent

The writer flags `n_to_flag = ceil(0.5 s / integration_time)` samples `UNKNOWN`
after a switch change — **2 rows on the 0.2684 s files, 1 on the 0.5369 s ones**.
Measured run-lengths on Jul 17: `{1: 128, 2: 229, 3: 5, 4: 1}`; the 3s and 4s are
mid-integration flips that `_avg_rfswitch_metadata` collapses to `UNKNOWN`, plus
error statuses.

**Never gate by dropping a fixed row count. Gate on the string.**

## Index schema

**Identity (always):** `file` (basename), `row`, `time`, `acc_cnt`,
`sync_recovered` (= `time − acc_cnt × integration_time`), `sync_consistent`,
`time_fname`, `time_best` (both defined under `sync_consistent` below).

**Per-file root attrs broadcast to rows, bare names.** The filtered
deployment-5 files already carry `filter_phase`, `filtered_keys`,
`mux_copy_0to1` and `mux_copy_4to5` at file root, asserted by
`filter_corr_keys.py`. These are the first annotations, and they cost nothing
to index: `select(filter_phase="C")` works from day one. Header attrs override
root attrs on a name collision.

**Per-file header attrs broadcast to rows:** `run_tag`, `integration_time`,
`adc_mux_sel`, `nchan`, data keys present (`data_keys`, comma-joined).

**Curated metadata default:** `rfswitch`; motor positions/targets/`boot_id`;
`pot_az_angle`/`near_rail`/`sp1_term_name`; IMU-el accel + `el_deg`;
`tempctrl_load.T_now`/`active` **and** `tempctrl_lna.T_now`/`active` (any
receiver calibration needs the LNA temperature as much as the load's); the
three switch-board thermistors; `system_current.current_a`;
`lidar.distance_m`; plus a `<stream>_ok` boolean per curated stream.

Naming is `<stream>_<field>`; `rfswitch` stays bare. Flattening *everything*
would be ~130 columns (~600 MB at 1.2 M rows), so widening is opt-in via
`streams=[...]`, with `streams="all"` as the escape hatch.

**Three kinds of missing, kept distinguishable:**

| case | column | `<stream>_ok` |
|---|---|---|
| stream absent from file | `MISSING` / NaN | False |
| row entry is `None` (no reading this window) | `MISSING` / NaN | False |
| field is `None` (sensor offline) | `MISSING` / NaN | True |

Numeric columns use NaN; string columns — `rfswitch`, `sp1_term_name`, a
missing `run_tag` — use the `MISSING` sentinel and **never** Python `None`.
That is what makes the cache round trip lossless: an object column holding
`None` would come back from HDF5 as the string `"None"`.

The `<stream>_ok` column holds under every `streams=` mode, but only because
`_finalise` normalises it. `streams="all"` enumerates the streams *each file*
carries, so a file without one emits no such column at all and concatenation
leaves a gap; `_finalise` fills the gaps of an `*_ok` column with `False`,
which is the same answer a named or curated request produces directly. Without
that step the gap reads `MISSING` and `select(<stream>_ok=False)` returns no
rows for exactly the files it is asking about.

So `UNKNOWN` (producer-asserted transition) is never confused with `MISSING`
(no metadata at all).

### `None` vs `UNKNOWN` for `rfswitch`

These mean opposite things and must never be merged:

- **`UNKNOWN` — "we have information, and it says this row is contaminated."**
  Produced when any reading in the window has `status == "error"`; when two
  different `sw_state_name` values appear inside one integration (a mid-window
  flip); or by the writer's forward transition guard, which overrides even a
  missing reading ("is also applied when the sample carried no rfswitch reading
  at all").
- **`None` — "we have no information."** Produced when the `rfswitch` key is
  absent from the dict handed to `add_data` (so `_insert_sample` pads `None`);
  when the stream value is not a non-empty list, or `avg_metadata` raises (both
  logged at ERROR and dropped); when `value[0]` is not a dict; when the reading
  arrived but carries no `sw_state_name`; or on a gap-fill sample.

**Empirically, `None` is a Pico-level dropout.** `rfswitch` and
`rfswitch_therm` are fanned out of the same `PicoRFSwitch._rfswitch_redis_handler`,
so they fail together. Across 196080 sampled rows, `rfswitch` is `None` on
**2.50 %**, and on those rows `rfswitch_therm` is also missing **4865 / 4896
(99.4 %)**; where `rfswitch` holds a state, `rfswitch_therm` is present
**191145 / 191184 (99.98 %)**. Run lengths are mostly 1 sample, but reach 197 —
so both brief blips and multi-minute outages occur.

Upstream quirk worth reporting (code reading, not observed in data): in
`_avg_rfswitch_metadata`, `unique` excludes `None`, so `states == [None, "RFANT"]`
yields `len(unique) == 1` and the function returns `states[0]` — i.e. `None`,
discarding a good state because the *first* sub-reading lacked the field.

### `sync_consistent`, and what it cannot do

`times = acc_cnt × integration_time + sync_times`, elementwise. A stale
`sync_time` poisons every time in a file while the filename — stamped from
`datetime.now(timezone.utc)` at *write* time — stays correct.

Detection rule: `|filename_time − last header time| > 1 h`. Measured on 732
sampled files: good files land in **[−1, +945] s** (the write backlog), bad ones
are off by **≈54 days**, flagging **12.4 %** — bimodal with four orders of
magnitude of margin, so the threshold is uncritical.

The column is named `sync_consistent`, **not** `time_ok`, because it detects a
clock **correction mid-run** and is **blind to a whole-run offset**, where
filename and times would be wrong together. Its docstring must say so. Checked
separately: mid-file sync spread is exactly 0 across 466 files, so in practice
the flag is per-file even though the writer permits per-row.

**The filename side of the comparison must be parsed in the right zone.**
Deployment 1–4 names are Pacific wall clock (see Clock handling); parsed as
UTC they sit 7 h from their correct header epoch and every file would be
flagged. `filename_unix(name, tz)` treats a `Z` suffix as UTC regardless of
`tz`, and otherwise uses `MetadataIndex(filename_tz=...)`, defaulting to
`America/Los_Angeles`.

**Flagged rows still need a usable time.** Without one, a table sorted on
header `time` puts the 12.4 % of stale-sync files at the front in May,
`visits()` splits on them, and a `time=` window silently excludes them — the
exact hazard the flag was built to expose. So two more columns:

- `time_fname = filename_unix − (ntimes − 1 − row) × integration_time`: the
  file-close time walked back by row. Good to the write backlog (up to ~16
  min), which is enough for day boundaries, visits and windows, not for
  pointing.
- `time_best = time if sync_consistent else time_fname`.

The table is **sorted on `time_best`**; `time=` filters and `visits()` use
`time_best`; `summary()` reports how many selected rows are estimates.
`EigsepData.times` carries `time_best` too, so a loaded day plots in the
right order without the caller knowing about stale clocks; the raw header
value is always available as `meta.time` and is never modified. This is the
lexsort-by-filename-then-`acc_cnt` trick that `quicklook_multiday.ipynb` and
`rfi_meeting_20260812.ipynb` each reinvented, made durable.

## Clock handling

`EigsepData.from_path(pacific_to_mountain=True)` currently adds 3600 s to every
timestamp by default. **This is a display convention, not a data correction, and
it must be removed rather than re-spelled.**

Established this session:

- The `eigsep_observing` stack is epoch-only and UTC-explicit
  (`sync_time = time.time()`; filenames `datetime.now(timezone.utc)`). No naive
  `now()`, `localtime` or `astimezone` anywhere, so the Pi's timezone setting
  cannot affect any stored value.
- Before `eigsep_observing` commit `c4ef1ee` ("stamp auto-generated h5 filenames
  in UTC with Z suffix") the writer used naive `datetime.now()`. **Deployment 1–4
  filenames are Pacific local wall-clock**; `+3600` rendered them as Utah watch
  time for matching handwritten field notes. Header `times` were always correct
  epoch.
- **Deployment 5's absolute epoch is correct UTC**, verified against an external
  anchor: hourly-median `rfswitch_therm` PCB temperature peaks at 22:00Z and
  bottoms at 12:00Z, against a prediction of 21:00–23:00Z / 11:00–12:30Z for
  lon −113.4027 (solar noon 19:34Z). A 7-h-behind clock would peak at
  14:00–16:00Z.

  *Do not re-verify using "the motor raster was at 20:28Z" — that figure derives
  from these same header times and is circular.*

Therefore:

- Loaders return **unmodified epoch**. `pacific_to_mountain` warns and is
  ignored for one release, then is deleted. `from_selection` never shifts.
- All of this lives in `clock.py`: `to_unix_time` (moved, re-exported from
  `data.py`), `format_time(t, tz="America/Denver")` so field-note correlation
  is served at **render** time, and `parse_filename_time(fname, tz=None)`,
  which returns an **aware** datetime: a `Z` suffix on the stamp means UTC
  whatever `tz` says; without one, `tz` applies, defaulting to
  `America/Los_Angeles` — the pre-`c4ef1ee` convention that covers
  deployments 1–4. `FILENAME_TZ` documents the per-deployment mapping.
  `data._parse_time_from_name` stays as an alias so `tests/test_data.py` is
  untouched. Today it returns a naive datetime and callers guess — a real
  cross-deployment bug.
- `from_path(start_time, end_time)` therefore changes meaning: it selects on
  `time_best` in UTC epoch, no longer on filename wall clock. The one caller,
  `notebooks/christian/deployment4/explore.ipynb`, wrote its window as
  Pacific filename times; it is re-expressed in UTC with a comment, not
  silently re-interpreted.

## Rewiring

- `EigsepData.from_selection(sel, keys=None, time_avg=1, missing="raise")` is
  canonical; `from_path()` stays as sugar that builds an index and selects.
  - **Cross keys come back complex, exactly as `read_hdf5` returns them.**
    Deployment-5 files store `04` as `(n, 1024, 2)` int32; the same
    reconstruction rule (`ndim ≥ 2`, last axis 2, integer dtype →
    `re + 1j·im`) applies. Reading raw h5py blocks without it would silently
    change the shape and dtype of `cross` for the beam pipeline.
  - **Output order equals selection order.** Rows are read per file and
    placed by an inverse permutation, never re-sorted by a separately
    computed argsort — that is only right when files never interleave in
    time, and the `-1` backlog twins make interleaving plausible.
  - **A requested key missing from a selected file** raises a `KeyError`
    naming the files and pointing at `data_keys` / `filter_phase` filters.
    `missing="nan"` NaN-fills that file's rows instead (what `load_gated`
    did). Deployment-5 phase boundaries make this a routine path, and the
    old `_select_h5_in_range` error for it was clear; a bare IndexError from
    deep inside a concatenate is not.
  - **`time_avg > 1`** averages consecutive selected rows per file in blocks,
    dropping and reporting the remainder, and downcasts to float32 /
    complex64. `meta` keeps the first row of each block with `time`,
    `time_best` and `acc_cnt` replaced by block means. A day of three keys at
    full resolution is a few GB and the deployment is tens; this is what the
    multiday notebooks needed to stay inside 16 GB. A memory guard warns when
    the estimated load exceeds half of `MemAvailable`.
- `extract_beam_mapping_data` keeps its **exact** return dict (`times, freqs,
  sky, ground, cross, el_pos, az_pos, pot_az_angle, imu_el_deg, imu_accel`) and
  becomes a thin wrapper: default curated streams, cache **on** (a
  `.eigsep_index.h5` sidecar appears next to the data; a read-only directory
  just skips the write with a warning), `select(time=(lo, hi),
  sync_consistent=True)` — which reproduces the old header-time contract,
  since a stale-sync file never fell inside a real window — and
  `missing="raise"`. With `cache=False` every call would rescan all metadata
  JSON in the directory, ~20 s on deployment 5, where the old code peeked
  only `header/times`. `beam_sim`, `beam_fit`, and
  `notebooks/dominic/test_beam_mapping_module.ipynb` are untouched.
- `_select_h5_in_range` is deleted — private, one caller.
- `quicklook`: `QuickLookResult` gains `.meta`; the summary reports the per-file
  state breakdown; `--state RFANT` gates flagging and stats.
- `StateBrowser` is promoted to `eigsep_data.browse`, rebuilt on `Selection`,
  with `ipywidgets` imported only when the controls are built and added to
  the existing `vis` extra (same audience as `jupyterlab` and `ipympl`; one
  fewer extra to explain). `rf_state_tools.py` and the committed
  `rfswitch_index.npz` are then deleted. Its `files_with(state, min_rows)`
  becomes `sel.file_counts()`.

The rotation-response paper figure is **not** at risk: `build_nb.py` reads a
sidecar built by `motor_scan_20260717.ipynb` with plain h5py and never calls
`extract_beam_mapping_data`.

## Cache

`<data_dir>/.eigsep_index.h5`, written with `h5py` — already a dependency, where
parquet would pull in pyarrow and npz would mangle dtypes. Gitignored: it is
derived from untracked data.

**The scan must never see its own cache.** `pathlib.Path.glob("*.h5")`
matches dotfiles (verified on this venv's Python 3.12), so a naive glob would
put the cache into its own manifest — its mtime changes on every write, the
fingerprint never matches, and the cache never hits, with a "skipping" warning
on every build for good measure. `_files()` excludes dotfiles and the cache
name explicitly, and the default pattern is `corr_*.h5`, which is what the
old `from_path` used and which keeps sidecars such as
`motor_scan_20260717_key4.h5` out of the index.

**Fingerprint:** schema version, patterns, `filename_tz`, the scanner's
qualified name, and the sorted `(basename, size, mtime_ns)` manifest — every
input that changes the table's contents. The stream set is stored beside it,
not in it, and compared as a **subset**: a request whose streams are covered
by the cache hits and keeps all cached columns; a wider request rebuilds with
the union and overwrites. An exact-match policy would make the curated
default and the beam wrapper thrash each other's cache.

**Round trip is lossless.** Object columns are encoded to fixed-width bytes
on write and decoded on read; the `MISSING`-never-`None` rule above is what
makes that exact. The test asserts whole-table equality after a round trip,
not just the time column.

Cold-scan cost, measured on the finished implementation against the real
deployment (5124 files, 1.23 M rows, the nine curated streams): **63.7 s**
cold, and **4.78 s** for a later session to rebuild from the sidecar, of which
3.76 s is reading it. The sidecar is **407.7 MB**, 332 B/row.

The pre-implementation estimate (~21–26 s cold, ~0.1 s cached) was wrong in
both halves for one reason: it never priced the **per-row materialisation of
per-file broadcast columns**. Eleven of the 48 columns hold one value per
*file* stored once per *row* — a 240× redundancy at 5120 files against 1.23 M
rows — and nothing in the cache is compressed. The remedy is identified and
deferred to after the merge: `pd.Categorical` on the seven low-cardinality
object columns plus HDF5 compression, which requires a `SCHEMA_VERSION` bump
under the rule documented beside `SCHEMA_VERSION` in `index.py`.

Filename globs are matched against the unique filenames and then broadcast
with `isin`, not evaluated per row — per-row `fnmatch` over 1.2 M rows costs
seconds and would make the cached path slower than the promise above.

## Deferred: the annotation layer

Facts like "transmitter on", "antenna suspended", or the A/B/C wiring phases are
**asserted** — a human knows them, no sensor recorded them, and an index rebuild
must never touch them. They belong in small, hand-written, version-controlled
interval files joined onto index rows at query time, so that

```python
idx.select(rfswitch="RFANT", antenna="suspended", comb_tx="off")
```

works uniformly whether a column came from a Pico or from a logbook, with
provenance attached.

Not built now: the vocabulary, overlap/conflict policy, and confidence model
need input from the other interested parties. Guessing produces a vocabulary
nobody adopts.

**Three seams to build now — cheap now, expensive later:**

1. **Row identity is `(file basename, row)`, never array position.** Positions
   shift when the file set changes; annotations pinned to them rot invisibly.
2. **Open column namespace** — a table with named columns and per-column
   provenance, not a dataclass with fixed attributes. `pandas` is already a
   declared dependency and unused in `src/`; a DataFrame is the natural carrier.
3. **`sync_consistent`, plus filename-range selectors alongside time intervals.**
   Annotations are naturally written in wall-clock time, but 12.4 % of D5 files
   have header times wrong by days, so a pure time-interval join would
   confidently mislabel them. This is why `filter_corr_keys.py` keyed its phases
   off filenames; that prior art was right.

First candidates to migrate when the layer lands: comb transmitter on from Jul
17 18:14Z; motor-scan session windows; D4 noise-dropout events; and the
per-deployment filename timezone. The A/B/C wiring phases are already served:
`filter_corr_keys.py` wrote them as root attrs on the filtered files, and the
index broadcasts root attrs (see Index schema). The raw files on the external
drive do not carry them, which is exactly the case the annotation layer is
for.

Two primitives that later work will want, both a few lines of pandas on top
of what is built now and deliberately not built yet: a **calibration cycle
id** grouping consecutive antenna → load → noise visits (the Y-factor pairing
in `calibration.ipynb` is the first customer), and an **S11 index** over the
`s11_data_deployment5` sweeps through the `scanner=` seam, so VNA switch
states in the corr index can be joined to the sweeps they bracket.

## Testing

**Write real files with the real writer.** `eigsep_observing._test_fixtures.CORR_METADATA`
already contains the edge-case ladder:

```python
["RFANT"]*20 + ["UNKNOWN"]*5 + ["RFNOFF"]*20 + [None]*5 + ["RFNOFF"]*10
```

A pytest fixture calls `eigsep_observing.io.write_hdf5` to build a small
directory of genuine corr files, so tests pin the **producer's** contract rather
than a mock of it. The fixture writes cross keys as `(n, 1024, 2)` int32 —
the real on-disk layout — and can set root attrs, so the complex
reconstruction and the phase columns are exercised, not assumed.

Cases to pin:

- the three kinds of missing stay distinct, and string columns never hold
  `None`
- an unseen state does not raise (open vocabulary)
- `None` is never counted as a state
- per-file `integration_time` variation, and that gating never assumes a fixed
  transition width
- `sync_consistent` on a deliberately bad `sync_time`; a Pacific-named
  deployment-4 file with correct epoch is **not** flagged; a `Z` suffix wins
  over an explicit zone
- a stale-sync file gets `time_best` from its filename and sorts into its
  true slot, and a `time=` window reaches it
- root attrs appear as columns; `tempctrl_lna` is curated
- the cache file and other dotfiles are neither scanned nor warned about
- cache invalidation on a touched or added file and on a schema bump; a
  narrower stream request hits, a wider one rebuilds with the union;
  whole-table equality after a round trip
- cross keys load complex and equal `re + 1j·im` from the raw file
- output order equals selection order for two files whose rows interleave in
  time
- a key missing from one selected file raises clearly, or NaN-fills on
  request
- `time_avg` shapes, dtypes, `meta` length, and the dropped remainder
- filename range tuples, globs, and `file_counts()`
- `from_path` no longer shifts times, and warns when `pacific_to_mountain` is passed

**Characterization test first.** `extract_beam_mapping_data` has zero coverage
today, so it gets a golden-array test written against the *current*
implementation and passing **before** anything is rewired. The rewrite is then
constrained to reproduce it exactly — including that `cross` is complex, and
pinning `pot_az_angle` and `el_pos` values, not only shapes.

Existing pure-function tests in `tests/test_data.py` are untouched.

## Build order

1. Fixture: real corr files from the real writer, cross keys `(n, 1024, 2)`,
   root attrs
2. `clock.py` + tests — `to_unix_time` moved, `format_time`, zone-aware
   filename parsing
3. `metadata.py` + tests — pure, no I/O
4. `index.py` scan + tests on synthetic dirs: identity columns, `time_best`,
   root attrs, scanner hook
5. `Selection` + tests: globs, ranges, `time_best` windows, `summary()`,
   `file_counts()`
6. Cache + tests: self-exclusion, subset policy, lossless round trip
7. Characterization test for `extract_beam_mapping_data`, green on current code
8. `from_selection` + tests: complex cross, order, `missing`, `time_avg`
9. Rewire the beam wrapper — characterization test must stay green; delete
   `_select_h5_in_range`
10. `from_path` onto the index; deprecate `pacific_to_mountain`; re-express
    the deployment-4 notebook window in UTC
11. `quicklook` state awareness
12. `browse.py`, then delete `rf_state_tools.py` and `rfswitch_index.npz`

## Open questions

- **An unresolved timezone contradiction, which does not block this work.**
  Three facts do not fit together:

  1. `eigsep-field/image/pi-gen-config/config` pre-seeds `TIMEZONE_DEFAULT="Etc/UTC"`.
  2. The analysis laptop is `America/Los_Angeles`, and the `sshpi` alias sends
     `date +'%Y-%m-%d %H:%M:%S'` — a zone-less **local** string — which
     `date -s` then interprets in the Pi's own zone.
  3. Deployment-5 timestamps are nonetheless correct UTC.

  (1) + (2) predict a 7-hour error; (3) excludes one decisively — the thermal
  minimum falls in the 12:00Z bin against a computed sunrise of 12:26Z, and the
  maximum 2.3 h after solar noon (19:40Z). So one premise does not hold for the
  field Pi: either its zone was not `Etc/UTC` in July, or something other than
  the current `sshpi` last set its clock. Note `~/.bashrc` was modified
  2026-09-10, after the deployment, so today's alias text is not evidence of
  what ran in the field. A `timedatectl` on the Pi resolves it.

  This matters only for interpreting **future** deployments; D5's times are
  verified correct regardless of which premise fails. The `sshpi` hazard itself
  is known and tracked outside this repo.

Deployment 1–4's `+3600` is confirmed by the observer's field notes; it is a
display convention, and no longer an open question.

## Revision log

**2026-09-11, after review of the first plan.** Held against the
deployment-5 notebooks as use cases and against the writer and a real Jul 17
file. Changes, in order of consequence:

- `from_selection` reconstructs complex cross keys like `read_hdf5`; the
  fixture writes the real `(n, 1024, 2)` layout so the characterization test
  can see it.
- The scan excludes its own cache file and dotfiles; default pattern
  `corr_*.h5`.
- Filename parsing is zone-aware via the `Z` suffix and `filename_tz`;
  `sync_consistent` no longer flags every deployment-4 file. One parser, in
  `clock.py`.
- `time_fname` and `time_best` added; sorting, `time=` windows and `visits()`
  use `time_best`; `summary()` reports estimated rows.
- Output row order comes from an inverse permutation, not a re-sort.
- Missing-key policy is explicit: raise by default, `missing="nan"` opt-in.
- `time_avg` on load with float32 / complex64 and a memory guard.
- Filename range tuples; globs matched on unique names.
- Cache stream policy is subset-hits / union-rebuild; round trip is lossless
  because string columns hold `MISSING`, never `None`.
- Root attrs broadcast (`filter_phase`, mux-copy flags); `tempctrl_lna`
  curated; `nchan` broadcast; `file_counts()`.
- Beam wrapper uses the cache and pins `sync_consistent=True`.
- `from_path` selects on `time_best` in UTC; the deployment-4 notebook window
  is re-expressed rather than silently shifted.
- `scanner=` hook on `MetadataIndex` as the seam for an S11 index.
- `ipywidgets` goes in the existing `vis` extra, not a new one.

**2026-09-11, after the whole-branch review of the implementation.** Corrections
only, no design change: the scan and rebuild costs above are now the measured
ones and carry the diagnosis of why the estimate missed them, and the
`<stream>_ok` contract now says what makes it true under `streams="all"`.
