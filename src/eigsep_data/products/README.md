# eigsep_data.products

Derived companion datasets, joined to raw correlator rows by
`(file, row)`. Each product knows how to find its own payload and how to
return it aligned to a set of integrations; `eigsep_data.bundle` knows
nothing about any of them beyond the interface in `base.py`.

Added 2026-09-17 because four independent implementations of the same
join existed — `rfi_explorer_core.load_range`, `b16_dpss_model.fit_file`,
`beam_explorer`'s offline `.npz` cache, and
`beam_mapping.diagnostics.load_v007_data` — and none knew about the
others. See `DATASET_LOADER_PLAN.md` in the parent meta-repo.

## Modules

| Module | Purpose |
|---|---|
| `base.py` | The plugin contract: `Product`, the registry, `parse_spec` (`"kind@version"`), and the frequency-axis helpers `axis_fingerprint` / `locate_axis`. |
| `flags.py` | RFI category bitmasks, one HDF5 day file per campaign day. Also `read_file`, the whole-file read the campaign's `flagging/read_flags.py` delegates to. |
| `smooth_model.py` | DPSS smooth-band model, one companion HDF5 per raw file. |
| `pointing.py` | Calibrated pointing, one Parquet table, joined on `(file, sample_idx)`. |
| `gain.py` | Receiver gain / noise temperature, solutions on their own cadence, joined by nearest time within an explicit tolerance. |

## Join shapes

Three exist, which is what the contract has to cover:

- **Per-(time, channel) cube**, one payload per raw file (`smooth_model`)
  or per day keyed by raw filename (`flags`). Row-sliced, band-matched.
- **Per-row scalars** keyed by `(file, row)` (`pointing`). An exact join,
  so no tolerance has to be chosen or defended.
- **Nearest-in-time within a bound** (`gain`, and S11 when it lands).
  This is the one that goes quietly wrong: a solution carried across a
  switch cycle attaches a calibration to a different receiver state. The
  window is explicit, out-of-window rows are NaN rather than reaching
  for the next solution, and the offset actually used comes back as a
  column.

## Adding a product

Subclass `Product`, implement `fetch` (and `freqs` if it is a cube),
decorate with `@register`. `load_bundle` needs no changes — `gain` was
~100 lines and one extra argument on `fetch`. Set `cube = True` when
`fetch` returns `(nrow, nchan)` arrays that must be aligned in
frequency.

Two rules the contract enforces rather than trusts:

- **Versions are explicit.** A spec is `kind@version`, never bare.
  `flags@v0` is `uint8` and `flags@v2` is `uint16`; a default version is
  how that difference reaches somebody's arrays with no call site
  changing.
- **Frequency axes are asserted, not regridded.** The axis is resolved
  once per product version into a slice; per file only a three-scalar
  fingerprint is compared, so the check stays O(1) in a 5120-file loop
  while still catching a payload rebuilt on a different band. A shifted
  or decimated grid raises.

## Status

`flags`, `smooth_model` and `pointing` are verified against real
campaign data. **`gain` is tested on synthetic fixtures only** — the
real `abscal` solutions are still loose `.npz` in the meta-repo's
`abscal/` rather than at `derived/gain/v0/solutions.npz`, deferred until
someone is actually calibrating with it (Aaron, 2026-09-17). Its
tolerance default and column set are untested guesses about real data;
the join logic is tested.

## Recent changes

- 2026-09-17 (`agent:eigsep-67`): added `read_file` to `flags`, so the
  campaign's `flagging/read_flags.py` could become a shim instead of a
  second implementation of the day-file layout.
- 2026-09-17 (`agent:eigsep-67`): added `gain`, the nearest-in-time join
  shape. Needed no change to `bundle.py`, which is the evidence this
  contract holds.
- 2026-09-17 (`agent:eigsep-67`): initial version — `flags`,
  `smooth_model`, `pointing`, promoted from `rfi_explorer_core.py`'s
  hardcoded `flags/v2` + `derived/smooth_model/v0` pair.
