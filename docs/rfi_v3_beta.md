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

## Running a file range

```sh
eigsep-rfi-v3-beta /path/to/campaign \
  --files corr_20260716_031155Z.h5 corr_20260716_033323Z.h5 \
  --air-antenna box-air --ground-antenna box-gnd
```

The endpoint filenames are inclusive. The physical antennas are resolved from
each HDF5 file independently, including cross-correlation orientation; the
runner does not assume that one correlator key has the same meaning throughout
the range. Use `--set NAME=VALUE` repeatedly to
override `RFIConfig` fields, and use `--dry-run` to run without writing. The
writer accepts complete files only, refuses an existing input dataset unless
`--overwrite` is supplied, and updates files atomically.

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
