# eigsep_data

Analysis package for the EIGSEP experiment: a per-integration metadata
index over a deployment's correlator files, a loader that joins raw
spectra to their derived companions, plus beam mapping, RFI, S11 and
simulation code.

Split out of [`EIGSEP/data-analysis`](https://github.com/EIGSEP/data-analysis)
on 2026-09-17, with history. That repo keeps the notebooks; this one is
the importable package. **The import name did not change** — it was
always `eigsep_data`, only the repository was called `data-analysis`.

```sh
pip install -e .
```

## First: say where the campaign is

A campaign is a directory holding `data/` beside `flags/`, `derived/`,
`curation/` and `imgs/`. Point the package at one once, right after
import, and everything that needs it finds it without being handed a
path:

```python
import eigsep_data

eigsep_data.set_campaign_root("~/data/marjum-2026-07")
```

Either the campaign root or its `data/` subdirectory works — a directory
named `data` is read as "the campaign is its parent" — so pointing at the
data itself, which is how people describe the location out loud, does the
right thing. A path that does not exist is refused on the spot; pass
`must_exist=False` for a campaign not yet staged on this machine.

Resolution order, most specific first: an explicit `root=` at the call
site, then `set_campaign_root()`, then the `EIGSEP_CAMPAIGN_ROOT`
environment variable, then whatever the caller can compute (for an index,
its `data_dir` parent). The in-process setter deliberately outranks the
environment variable, which is ambient and easy to forget having
exported. `eigsep_data.get_campaign_root()` reports what is in effect and
`campaign_data_dir()` gives the `data/` path.

Nothing about the existing explicit-path API changed: passing paths
directly still works and still wins.

## Where to start: the index

Scanning reads headers and metadata only, never spectra. On the 5120-file
deployment 5 (1.23 M integrations) a cold scan is ~64 s, then ~5 s per
session from the sidecar cache it leaves behind. Queries run against the
table, and only the selected rows are ever read from disk.

```python
from eigsep_data import MetadataIndex

idx = MetadataIndex("/path/to/marjum-2026-07/data")
sel = idx.select(time=("2026-07-17T20:00Z", "2026-07-17T21:00Z"),
                 rfswitch="RFANT")
print(sel.summary())          # what each filter removed
data = sel.load(keys=["0"])   # EigsepData: spectra for those rows only
```

Row identity is the `(file, row)` pair and ordering is by `time_best` —
the header time where the file's clock was sane, a filename-derived
estimate otherwise. Note that corr filenames are file *close* times, so
a filename-ordered range is not quite a time-ordered one.

## The loader: raw data plus its companions

`load_bundle` answers the question most analyses re-implement by hand:
*this stretch of data for this antenna, with its flags and its smooth-band
model and its pointing, already lined up in time and frequency.*

```python
b = sel.load_bundle(
    antenna="box-gnd",                              # resolved per file
    products=["flags@v2", "smooth_model@v0", "pointing@v1.2"],
    band_mhz=(45.0, 235.0),
)

b.data           # (nrow, nchan) raw counts
b.freqs_mhz      # one axis; every array here is on it
b.t              # time_best per row
b.flags          # uint16 bitfield, dtype exactly as stored
b.smooth_model   # the DPSS model
b.residual       # data - model, recomputed (see below)
b.pointing       # per-row az/el/quality/flags as a DataFrame
b.provenance     # versions, manifests, and every file that was skipped
print(b.summary())
```

Four things it will not do for you, each of which has bitten this
campaign at least once:

- **No default product version.** `flags@v0` is `uint8` and `flags@v2`
  is `uint16`; code that assumes one-byte pixels misbehaves silently on
  the other. You name the version, and it is recorded in `provenance`.
- **No regridding.** Frequency axes are asserted to match. The axis is
  resolved once per product version into a slice; per file only a
  three-scalar fingerprint is compared. A shifted or decimated grid
  raises rather than snapping to the nearest channel.
- **No hardcoded input keys.** The antenna is resolved from each file's
  own header. `box-gnd` is input 0 on 2026-07-17 and input 2 on 07-13,
  and the ADC mux copies an even input onto the odd one above it.
- **No silent drops.** A file with no companion keeps its rows as NaN
  and is named in `provenance`, so eight files' worth of plot cannot be
  mistaken for twenty-eight.

`residual` is always `data - model`, recomputed. `hera_filters` zeroes
the companion's stored residual at flagged pixels, which would hide
exactly the comb and RFI spikes a residual waterfall exists to show.

## Adding a product

A product is a derived dataset keyed by the raw files —
`eigsep_data/products/`. Implement `Product.fetch`, `@register` it, and
`load_bundle` needs no changes; `gain`, the nearest-in-time join, was
~100 lines. Three join shapes exist so far:

| kind | payload | joined by |
|---|---|---|
| `flags` | one HDF5 day file, per-(time, channel) bitfields | `(file, row)` |
| `smooth_model` | one companion HDF5 per raw file | `(file, row)` |
| `pointing` | one Parquet table | `(file, sample_idx)` |
| `gain` | solutions on their own cadence | nearest time, bounded |
| `tcal` | calibration measurements on their own cadences | bracketing in time, bounded |

`products/noise_source.py` is a different kind: one bench
characterization of the noise source, an input to `tcal`, read with
`noise_source.load(version)` rather than joined to rows.

## Layout

- `index.py`, `metadata.py`, `clock.py` — the per-integration index.
- `data.py` — `EigsepData`, the spectra reader.
- `bundle.py`, `products/` — the raw-plus-companions join.
- `beam_mapping/` — beam fits, pointing fusion, TX forward model.
- `rfi.py`, `s11.py`, `quicklook.py`, `browse.py` — analysis utilities.
- `select_files.py` (`eigsep-select-files`) — file-level campaign masks
  and observing-mode selection.
- `flagging/` — writer for the `flags/v0` per-(time, channel) masks.
- `switched.py` — switched calibration measurements: dwells, cycles,
  the VNA-sweep guard, thermistors with measured pull-ups.
- `noise_source_bench.py` — the noise source's temperature at P from a
  bench run (`measure`), stored with `products.noise_source.write`.
- `calibration/` — builders of the `tcal` and `noise_source` products
  (`eigsep-build-tcal`, `eigsep-validate-tcal`, `eigsep-tcal-bracket-cost`,
  `eigsep-build-noise-source`); each version's campaign choices are its
  `config.json` in the campaign tree.
- `units.py` — unit conversions: `to_dB`/`from_dB` for power ratios,
  `celsius_to_kelvin` (with `ZERO_CELSIUS_K` from `scipy.constants`).
- `provenance.py` — file checksums and git state, for the provenance
  records that product generators write into their manifests.
- `beam_sim.py`, `sim.py`, `hpm.py`, `sph_fit.py` — simulation and
  spherical-harmonic work (needs the JAX/healjax extras).

## Related repositories

- [`eigsep_base`](https://github.com/EIGSEP/eigsep_base) — the file-format
  contract (correlator HDF5, S11, metadata), physical constants, time
  handling. Depends on nothing here; installs on hardware with no JAX.
- [`data-analysis`](https://github.com/EIGSEP/data-analysis) — notebooks
  and exploratory work. Imports this package; not installable itself.
- [`eigsep_observing`](https://github.com/EIGSEP/eigsep_observing) — field
  software that writes the files this package reads.

## Recent changes

- 2026-10-21: `noise_source_bench.measure` takes the sub-bands its validation reports as `report_bands_mhz` (default unchanged: 50-88, 108-150, 150-200, 200-245 MHz).
- 2026-10-21: `units.celsius_to_kelvin` replaces the literal 273.15 in `noise_source_bench` and `calibration/`.
- 2026-10-21: `units.to_dB` and `units.from_dB` (also `eigsep_data.to_dB`/`from_dB`) are the package's dB conversions for power ratios; `quicklook` and `calibration/d5_yfactor` use them, with identical results.
- 2026-10-21: `provenance` holds the file checksum and git helpers that `calibration/`, `flagging/`, `geometry_release` and `rfi_supported` each used to define for themselves. What each product records is unchanged.
- 2026-10-20: the `tcal` and `noise_source` builders move here from `data-analysis` as `calibration/` with four commands, so no released product depends on `data-analysis`. Each product version's campaign choices (span, flags, limits, bench wiring, thresholds) are a `config.json` beside it in the campaign tree, which the manifest records. Christian's chain (`calibration/d5_yfactor.py`), which the `tcal` validation compares against, is vendored unchanged.

- 2026-10-20: `switched` (dwells and cycles of switched calibration measurements, the VNA-sweep guard, thermistors with measured pull-ups) the `noise_source` product reader and writer, and `noise_source_bench.measure` (the noise source's temperature from a bench run) added, for the bench-measured noise source; `tcal` accepts `t_ns_k` as one value per channel (from v0004) as well as one number (v0000-v0003).
- 2026-10-08: `geometry_release` publishes releases from a `derived/geometry_posterior/vNNNN` product, numbered to match it (`v0004_marjum_geometry` is the first); the `terrain/` builder that made v0001 needs `--legacy-terrain`.
- 2026-10-05: `rfi_supported` released as `v3` (default product version `v3`,
  status `released`, CLI `eigsep-rfi-v3`; `eigsep-rfi-v3-beta` still works). The
  numerical algorithm is unchanged from revision 5.
- 2026-10-04: `rfi_supported` revision 5: high-scatter channels get an advisory
  bit 9 (`high_scatter`), not excluded by default, instead of bit 7.
- 2026-10-04: `rfi_supported` revision 4: files recorded with the switch daemon
  off are flagged as sky (the switch sat on the antenna), not left non-sky.
- 2026-10-04: `rfi_supported` revision 3: unconverged cross channels and uneven
  sampling no longer fail batches, calibration states are flagged, channels the
  smooth background cannot describe are marked unsupported, and box-gnd can be
  flagged into the same product (`docs/rfi_v3_beta.md`, memo 006).
- 2026-10-03 (Claude Code, for Aaron): comb labels corrected to memo 001
  everywhere. The 8-channel comb (07-17 onward) is the beam-mapping
  transmitter and the 07-16 1.000 MHz comb is box-air's own EMI; the old
  labels had them backwards. `select_files` masks are now `tx-comb-teeth`
  and `boxair-emi-1mhz` (old names warn), the `--tx-comb` mode is replaced
  by `--boxair-emi` and `--transmitter` (old flag errors), and `flagging`
  reads the mode table's new `transmitter` column. Mode tables built
  before the rename must be rebuilt.
- 2026-10-03: `rfi_supported` v3-beta's point threshold now scales with each
  channel's measured residual scatter (`point_scale="empirical"`, new algorithm
  revision); the radiometer-only threshold over-flagged smooth-model error as RFI
  (`docs/rfi_v3_beta.md`, memo 006).
- 2026-09-17 (`agent:eigsep-67`): `import eigsep_data` made lazy, 7.05 s
  -> 0.00 s; JAX is no longer pulled to read a flag mask.
- 2026-09-17 (`agent:eigsep-67`): added `bundle.py` and `products/`, the
  raw-plus-companions loader, and this README -- the split carried the
  package across but not a file describing it.
- 2026-09-17 (`agent:eigsep-67`): repo created by splitting the package
  half out of `EIGSEP/data-analysis`, with history.
