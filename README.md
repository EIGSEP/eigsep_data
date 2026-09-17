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

## Layout

- `index.py`, `metadata.py`, `clock.py` — the per-integration index.
- `data.py` — `EigsepData`, the spectra reader.
- `bundle.py`, `products/` — the raw-plus-companions join.
- `beam_mapping/` — beam fits, pointing fusion, TX forward model.
- `rfi.py`, `s11.py`, `quicklook.py`, `browse.py` — analysis utilities.
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

- 2026-09-17 (`agent:eigsep-67`): `import eigsep_data` made lazy, 7.05 s
  -> 0.00 s; JAX is no longer pulled to read a flag mask.
- 2026-09-17 (`agent:eigsep-67`): added `bundle.py` and `products/`, the
  raw-plus-companions loader, and this README -- the split carried the
  package across but not a file describing it.
- 2026-09-17 (`agent:eigsep-67`): repo created by splitting the package
  half out of `EIGSEP/data-analysis`, with history.
