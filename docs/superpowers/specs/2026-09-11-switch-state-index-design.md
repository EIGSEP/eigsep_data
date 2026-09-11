# Per-integration metadata index and switch-state gating

**Date:** 2026-09-11
**Status:** design approved, not yet implemented

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

Three modules, split so the pure logic is testable without touching a disk:

| module | responsibility | I/O |
|---|---|---|
| `metadata.py` | flatten one file's metadata group → named columns; missing/None policy | none — dicts in, arrays out |
| `index.py` | `MetadataIndex` (scan, cache, invalidate), `Selection` (query result) | headers + metadata only |
| `data.py` | `EigsepData` carrying `.meta`; `from_selection()` reads chosen rows | spectra |

`data.py` imports `index.py`; never the reverse. `Selection.load()` is sugar
over `EigsepData.from_selection` with a function-level import.

```python
idx = MetadataIndex("data/deployment5_filtered")       # ~25 s cold, ~0.1 s cached

sel = idx.select(rfswitch="RFANT", run_tag="motor_scan",
                 time=("2026-07-17 20:28", "2026-07-17 21:28"))
sel.summary()                    # rows in, rows out, what each filter removed
sel.visits(gap_s=600)

d = EigsepData.from_selection(sel, keys=["0", "4"])    # only now are spectra read
d.meta.rfswitch                  # aligned row-for-row with d.times
```

Filters accept a scalar, a list, a `(lo, hi)` time range, a filename glob, or
`where=<callable>`. `select()` composes: `sel.select(...)` narrows further.

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
`sync_recovered` (= `time − acc_cnt × integration_time`), `sync_consistent`.

**Per-file header attrs broadcast to rows:** `run_tag`, `integration_time`,
`adc_mux_sel`, data keys present.

**Curated metadata default:** `rfswitch`; motor positions/targets/`boot_id`;
`pot_az_angle`/`near_rail`/`sp1_term_name`; IMU-el accel + `el_deg`;
`tempctrl_load.T_now`/`active`; the three switch-board thermistors;
`system_current.current_a`; `lidar.distance_m`; plus a `<stream>_ok` boolean per
curated stream.

Naming is `<stream>_<field>`; `rfswitch` stays bare. Flattening *everything*
would be ~130 columns (~600 MB at 1.2 M rows), so widening is opt-in via
`streams=[...]`, with `streams="all"` as the escape hatch.

**Three kinds of missing, kept distinguishable:**

| case | column | `<stream>_ok` |
|---|---|---|
| stream absent from file | `MISSING` / NaN | False |
| row entry is `None` (no reading this window) | NaN | False |
| field is `None` (sensor offline) | NaN | True |

So `UNKNOWN` (producer-asserted transition) is never confused with `MISSING`
(no metadata at all).

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
- Add `format_time(t, tz="America/Denver")` so field-note correlation is served
  at **render** time.
- `_parse_time_from_name` gains a per-deployment filename zone: D1–4 =
  `America/Los_Angeles`, D5 = UTC. It currently returns a naive datetime and
  callers guess — a real cross-deployment bug.

## Rewiring

- `EigsepData.from_selection(sel, keys=...)` is canonical; `from_path()` stays as
  sugar that builds an index and selects.
- `extract_beam_mapping_data` keeps its **exact** return dict (`times, freqs,
  sky, ground, cross, el_pos, az_pos, pot_az_angle, imu_el_deg, imu_accel`) and
  becomes a thin wrapper. `beam_sim`, `beam_fit`, and
  `notebooks/dominic/test_beam_mapping_module.ipynb` are untouched.
- `_select_h5_in_range` is deleted — private, one caller.
- `quicklook`: `QuickLookResult` gains `.meta`; the summary reports the per-file
  state breakdown; `--state RFANT` gates flagging and stats.
- `StateBrowser` is promoted to `eigsep_data.browse`, rebuilt on `Selection`,
  with `ipywidgets` imported inside `__init__` and declared as an `[interactive]`
  extra — mirroring how `__init__.py` already handles the optional JAX stack.
  `rf_state_tools.py` and the committed `rfswitch_index.npz` are then deleted.

The rotation-response paper figure is **not** at risk: `build_nb.py` reads a
sidecar built by `motor_scan_20260717.ipynb` with plain h5py and never calls
`extract_beam_mapping_data`.

## Cache

`<data_dir>/.eigsep_index.h5`, written with `h5py` — already a dependency, where
parquet would pull in pyarrow and npz would mangle dtypes. Keyed on a fingerprint
of the sorted `(basename, size, mtime_ns)` manifest plus the schema version and
the included stream set. Any mismatch, or a request for streams the cache does
not hold, triggers a silent rebuild. Gitignored: it is derived from untracked
data.

Cold-scan cost measured: **~21–26 s** for 5120 files × 11 streams (~1.2 M rows);
`rfswitch` alone ~5.5 s.

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

First candidates to migrate when the layer lands: the A/B/C wiring phases; comb
transmitter on from Jul 17 18:14Z; motor-scan session windows; D4 noise-dropout
events; and the per-deployment filename timezone.

## Testing

**Write real files with the real writer.** `eigsep_observing._test_fixtures.CORR_METADATA`
already contains the edge-case ladder:

```python
["RFANT"]*20 + ["UNKNOWN"]*5 + ["RFNOFF"]*20 + [None]*5 + ["RFNOFF"]*10
```

A pytest fixture calls `eigsep_observing.io.write_hdf5` to build a small
directory of genuine corr files, so tests pin the **producer's** contract rather
than a mock of it.

Cases to pin:

- the three kinds of missing stay distinct
- an unseen state does not raise (open vocabulary)
- `None` is never counted as a state
- per-file `integration_time` variation, and that gating never assumes a fixed
  transition width
- `sync_consistent` on a deliberately bad `sync_time`
- cache invalidation on a touched file, and on a widened stream request
- `from_path` no longer shifts times, and warns when `pacific_to_mountain` is passed

**Characterization test first.** `extract_beam_mapping_data` has zero coverage
today, so it gets a golden-array test written against the *current*
implementation and passing **before** anything is rewired. The rewrite is then
constrained to reproduce it exactly.

Existing pure-function tests in `tests/test_data.py` are untouched.

## Build order

1. `metadata.py` + tests — pure, no I/O
2. `index.py` + tests on synthetic dirs, cache invalidation included
3. Characterization test for `extract_beam_mapping_data`, green on current code
4. Rewire `data.py` — characterization test must stay green
5. Clock handling: remove the shift, add `format_time`, per-deployment filename
   zone; update `notebooks/christian/deployment4/explore.ipynb`
6. `quicklook` state awareness
7. `browse.py`, then delete `rf_state_tools.py` and `rfswitch_index.npz`
8. Delete `_select_h5_in_range`

## Open questions

- **Deployment 4's `+3600` is unverified.** Its raw data is on the T7 drive, not
  mounted. When it is, verify with an **external anchor** (the diurnal thermal
  fold above), not filename-vs-header, which is blind to whole-run offsets.
- **The Pi's timezone is inferred, not observed.** The data excludes a 7-hour
  error, and D1–4's Pacific filenames exclude a 1-hour one, which together imply
  `America/Los_Angeles`. A `timedatectl` on the Pi would make it certain.
- **`sshpi` is a standing hazard.** It sends the laptop's Pacific wall clock as a
  zone-less string; `date -s` interprets it in the Pi's zone, so the error is
  `Pi_offset − laptop_offset`. It happens to be correct only because both
  machines are Pacific. Sending epoch (`date -u +%s`) would remove the coupling.
