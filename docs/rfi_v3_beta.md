# Supported-DPSS RFI products (`v3-beta`)

`eigsep_data.rfi_supported` fits a smooth background to an air
autocorrelation and generates a reason-coded exclusion mask. It uses the
ground autocorrelation and their complex cross correlation to detect changes
in coherent emission. Stable coherent structure is absorbed by a robust,
time-smooth cross background rather than being flagged solely because its
absolute coherence is bright.

The background model combines two jointly fitted components. The original
50 ns frequency by 1 mHz time DPSS basis describes the continuum across the
full analysis band. A second, 300 ns frequency basis covers 35--88 MHz and has
one constant-in-time coefficient per spectral mode. It follows stable
low-frequency structure that the broad continuum basis cannot represent,
without giving narrow structure independent freedom in every integration.
The correction is tapered over 5 MHz at both band edges and orthogonalized
against the original frequency basis. Design support and ridge-prior
diagnostics are calculated for the complete combined design.

The corresponding `RFIConfig` fields are
`spectral_correction_halfwidth_s`, `spectral_correction_band_mhz`,
`spectral_correction_taper_mhz`, and
`spectral_correction_svd_cutoff`. Set
`spectral_correction_halfwidth_s=0` to disable the correction for a baseline
comparison.

This interface is beta. Name both products explicitly as `flags@v3-beta` and
`smooth_model@v3-beta`; no default product version is implied by the readers.

## Planning and running a campaign

```sh
eigsep-rfi-v3-beta --data-dir /path/to/campaign/data \
  --files corr_20260716_031155Z.h5 corr_20260716_033323Z.h5 \
  --plan
```

`--data-dir` can be omitted when `eigsep_data.set_campaign_root(...)` or
`EIGSEP_CAMPAIGN_ROOT` configures the campaign. The older positional campaign
root remains accepted. A separately mounted raw directory defaults to writing
`flags/` and `derived/` beside that directory; use `--output-root` to put those
products elsewhere.

The selection is explicit: use inclusive `--files` endpoints, a `--time`
range, or `--all`. `--plan` indexes and prints every pending complete-file
batch without fitting or writing. Inspect that JSON on the processing machine,
then run, for example:

```sh
eigsep-rfi-v3-beta --data-dir /data/marjum-2026-07/data --all \
  --output-root /data/marjum-2026-07 \
  --resolution-policy /data/marjum-2026-07/curation/antenna_resolution.json \
  --flags-version v3-beta.1 --model-version v3-beta.1 \
  --files-per-batch 10 --workers 6 --resume
```

The Marjum filtered files explicitly warn that `header/input_to_ant` is stale.
The runner therefore requires an antenna-resolution policy by default,
discovering `OUTPUT_ROOT/curation/antenna_resolution.json` when no path is
given. The approved `marjum-2026-07-rfi-safe-v1` policy selects ground/air/cross
as 3/4/35 for the 974 phase-B files with a valid 4-to-5 mux copy, and 0/4/04
for all 2,421 phase-C files. It explicitly excludes 1,588 unresolved phase-A
files and 137 phase-B files without a valid cross. The plan reports each rule's
counts and the exact excluded filenames. Policy rejection is never converted
to `missing="skip"`.

`load_bundle(..., resolution_policy=...)` exposes the same behavior to other
consumers while retaining header resolution when no policy is supplied. Passing
`--header-resolution` to the runner is the explicit opt-in to that generic
fallback; it is unsafe for this campaign.

Use a new product version such as `v3-beta.1`. Existing `v3-beta` records were
made under the stale-header contract and cannot be mixed with policy-resolved
records. Both the runner and writer reject a version directory containing a
different or absent policy hash, even with `--overwrite`.

Files are batched only within one UTC filename day, contiguous visit,
integration time, filter phase, and raw-key set. One process owns all batches
for a day, while different days run in parallel. BLAS thread counts default to
one per process to avoid oversubscription. The writer also takes a POSIX file
lock around its atomic payload and manifest updates, so worker completion order
cannot lose records.

The physical antennas are resolved from each HDF5 file independently,
including cross-correlation orientation; the runner does not assume that one
correlator key has the same meaning throughout the range. Use `--set
NAME=VALUE` repeatedly to override `RFIConfig` fields. `--dry-run` fits without
writing. Every product records the exact flagger source hash, numerical
algorithm revision, antenna-resolution policy hash, resolved keys, and matched
rules. `--resume` skips only files recorded in both product manifests with the
same parameters, numerical revision, and resolution policy. Partial or
mismatched products stop with an error. `--overwrite` replaces payloads only
within a policy-compatible product version. Time selections expand the two
endpoint files to their complete row sets because published companions are
whole-file products.

The runner writes the product layouts already consumed by
`Selection.load_bundle`:

- `flags/v3-beta/flags_YYYYMMDD.h5`, dataset
  `mask/<raw filename>/<input key>`, dtype `uint16`
- `derived/smooth_model/v3-beta/<raw filename>`, dataset
  `input_<key>/model`, dtype `float32`

Both version directories contain `manifest.json`. Each raw filename maps to
its source SHA-256 and the SHA-256 of the full parameter set. The flags
directory also contains `flag_bits.json`.

## Flag meanings

Every nonzero value means that the cell is excluded. Bits can overlap.

| Bit | Name | RFI evidence |
| ---: | --- | :---: |
| 0 | `non_sky_switch_state` | no |
| 1 | `invalid_input_or_domain` | no |
| 2 | `positive_auto_excess` | yes |
| 3 | `negative_or_model_failure` | no |
| 4 | `cross_change` | yes |
| 5 | `band_group_trigger` | yes |
| 6 | `comb_group_trigger` | yes |
| 7 | `unsupported_background` | no |
| 8 | `guard` | no |

`RFIResult.mask` includes every reason. `RFIResult.rfi_mask` includes only the
four rows marked as RFI evidence above. A non-sky switch state is flagged over
the entire frequency axis before fitting. Missing or nonpositive inputs and
the untested domain are also excluded.

The public model is NaN wherever bit 7 is set. These cells are not filled by
interpolation: the design-noise and ridge-prior diagnostics say the fit is not
reliably constrained there. The raw extrapolation remains available in the
in-memory `model_raw` result for diagnosis, but is not published as the model.

Selections are split at time gaps and integration-time changes, and each block
is fitted independently. A block containing no sky-state row is returned with
bits 0 and 7 set and a fully NaN model.
