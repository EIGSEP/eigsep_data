# eigsep_data.calibration

Builders of the calibration products. The readers are
`eigsep_data.products.tcal` (joined to rows by `load_bundle`) and
`eigsep_data.products.noise_source` (`noise_source.load(version)`); these
are the writer half of the same contracts, as `eigsep_data.flagging` is for
the flag product.

| Module | Command | Purpose |
|---|---|---|
| `build_tcal.py` | `eigsep-build-tcal <version>` | Collects the field calibration inputs on their own cadences and writes `derived/tcal/<version>/solutions.npz` and `manifest.json`: RFAMB/RFNON visit spectra (weighted means, masked with the flag product's RFI bits) with their receiver regime, field S11 at plane P on correlator channels, load-thermistor readings, and the noise-source excess `t_ns_k` (assumed, or built from a `noise_source` version). It computes no temperatures. |
| `validate_tcal.py` | `eigsep-validate-tcal <version>` | Leave-one-out test of linear versus nearest interpolation, a plumbing check through `load_bundle`, comparison with Christian's chain on a reference night, and coverage; writes `validation.json`. |
| `bracket_cost.py` | `eigsep-tcal-bracket-cost <version>` | What wider interpolation limits cost in temperature, and whether the largest hole can be bridged; writes `bracket_cost.json`. The limits in the config come from it. |
| `build_noise_source.py` | `eigsep-build-noise-source <version>` | Runs `eigsep_data.noise_source_bench.measure` on a bench run and writes `derived/noise_source/<version>/` with `products.noise_source.write`. |
| `d5_yfactor.py` | — | Christian's minimal Y-factor chain, vendored from `data-analysis` (58c4c64) with changes that leave its results identical: `calkit` comes from `eigsep_cal`, not `cmt_vna`, and its dB and degC conversions use `eigsep_data.units`. The validation's independent reference, used for no product. |
| `_common.py` | — | Config loading and the packages' commits. Checksums and git state come from `eigsep_data.provenance`, product paths from `eigsep_data.products.base`, times from `to_unix_time` and averaging from `eigsep_data.numerics`. |

**Campaign choices live in the campaign tree, not here.** Each builder reads
`derived/<product>/<version>/config.json` (or `--config`) and records its
path and SHA-256 in the manifest. To start a new version, copy the previous
version's `config.json` into the new directory and edit it. The configs say
why each choice was made.

All four take `--products <campaign tree>` to read the config from and write
into a tree other than the campaign root (e.g. a branch worktree); raw data
come from `EIGSEP_CAMPAIGN_ROOT`. `eigsep_cal` is imported only when a
command runs.

## Recent changes

- 2026-10-21: the builders' own copies of the weighted mean, the complex interpolation onto channels, the product path and the time parser are gone in favour of the package's shared ones; the products they write are unchanged.
- 2026-10-21: the validation's remaining fixed choices are keyword arguments of new functions, with the old values as defaults: in `validate_tcal`, `plumbing(tol_frac=0.05, max_channels_off=50)` (the output keys are built from them, so `channel_fraction_off_5pct` keeps its name) and `coverage(min_channel_fraction=0.99)`; in `bracket_cost`, `bracket_errors(brackets=...)` and `s11_hold_errors(holds=...)`, the spans scored. Output is unchanged.
- 2026-10-21: `bracket_cost`'s matching tolerances are documented keyword arguments with the old values as defaults: `cycles(max_pair_s=300)` pairs a noise-source visit with the load visit of the same cycle, and the new `across_hole(cycle_tol_s=600, s11_tol_s=1800)` sets how closely a comparison span must match the hole's. degC to K goes through `eigsep_data.units.celsius_to_kelvin`. Results are unchanged.
- 2026-10-21: checksums and git state come from `eigsep_data.provenance`, shared with the other product generators, instead of `_common.py`; manifests are unchanged.
- 2026-10-20: added: the `tcal` builders moved from `data-analysis/scripts/marjum-2026-07/tcal/` and the `noise_source` builder from `data-analysis/scripts/marjum-2026-07/noise_source/`, with their campaign choices moved into each version's `config.json`. Rebuilding `tcal` v0004 and `noise_source` v0001 with them reproduces the arrays exactly.
