# eigsep_data.beam_mapping

Beam mapping for the rotating-receiver / transmitter-drone experiment.
Promoted from `notebooks/arp/marjum-2026-07/`'s flat script layout into this
subpackage (B5: `rotation_beam`, `tx_beam_sim`, `beam_pca`,
`fit_v007_pca_beam`, `data_space_rfi` had 9–18 importers each and were
import-shaped, not exploratory — see `REORG_PLAN.md` and
`BRANCH_MERGE_PLAN.md` for the promotion rationale). `__init__.py`
re-exports the common entry points; it contains no logic of its own.

## Layers, in dependency order

| Module | Purpose |
|---|---|
| `beam_rotations.py` | Re-exports `eigsep_base.rotations`, the one pointing convention shared with `eigsep_sim.Beam.top2body`: body -> ENU is `Rz(psi) Rx(el) Rz(az)` (boresight +z, dipole arm +x; az rolls about the boresight; el tips about the axle, 0 = zenith; psi = axle direction ccw from East, always passed in). |
| `geometry.py` | Pointing fusion from motor, potentiometer, and IMU streams. |
| `tx_coupling.py` | Transmitter geometry (ENU heading, polarization), HFSS field normalization and frequency interpolation, the `|E* . e_tx|^2` coupling, per-tooth gains. |
| `tx_background.py` | Background under the comb teeth: local DPSS fit (`method='dpss'`, default) or gap averaging (`method='gap'`). |
| `tx_teeth.py` | Model-free tooth selection: isolation and iterative neighbour coherence. |
| `mapper.py` | Non-negative theta/phi gain recovery from alternating transmitter arms (`PolarizationBeamMapper`). |
| `tx_model.py` | HFSS-backed simulation of transmitter spikes and correlator waterfalls (`HFSSBeamSet`, consumes Dominic's `hfss_beam_maps/bowtie_beam.npz` format), built on `tx_coupling`. |
| `basis.py` | Low-rank template banks (PCA eigen-beams, `BeamPCA`) packed as an `HFSSBeamSet` for linear fitting. |
| `rfi.py` | Model-independent RFI flagging from raw correlator data (not beam residuals) — flags outlier times using off-comb monitor channels. |
| `fit.py` | Transmitter position and polarization fitting (JAX). |
| `diagnostics.py` | v007 campaign loading, joint fits, and the standard diagnostic figures. `load_v007_data` is **frozen**: `Selection.load_bundle` supersedes it for new work, but v007 is withdrawn and ~35 call sites in `data-analysis/notebooks/arp/marjum-2026-07/` published numbers out of exactly that code path. Keep it working, do not build on it. |

## Recent changes

- 2026-09-25 (Claude Code, for Aaron): `beam_rotations` now re-exports
  `eigsep_base.rotations` (merged to eigsep_base main as `d6db0c0`); the
  implementation and its tests live there.
- 2026-09-25 (Claude Code, for Aaron; branch `beam-mapping-promotion`, phase 1 of
  promoting `data-analysis/scripts/marjum-2026-07/fit_beam.py`): added
  `beam_rotations`, `tx_coupling`, `tx_background` and `tx_teeth`. **Breaking.**
  The package now has one pointing convention. `geometry.rotation_matrix`
  (+Z default) and `tx_model`'s hard-coded -Z path are gone.
  `simulate_hfss*`, `fit_ground_position`, `recover_sampled_beam`,
  `PolarizationBeamMapper` and the v007 `diagnostics` fits take a required
  `psi_deg`. `TransmitterGeometry` moved to `tx_coupling`, with `heading_top`
  renamed to `heading_enu` and `field_top` to `field_enu`. v007 diagnostics now
  give different numbers (Aaron chose migration over preserving them).
  `tx_background` keeps DPSS modes to a concentration of 1e-6.
  `concentration_min=None` reproduces the floor(2NW)+1 rule of beam fits
  v0009-v0011, which cannot represent a constant near the window edges.
  Still to come: `beam_basis`, the joint fit and export (phase 2);
  `fit.py`/`beam_sim.TransmitterAntenna` still carry their own rotation.

- 2026-09-17 (`agent:eigsep-67`): marked `diagnostics.load_v007_data`
  frozen rather than migrating it onto the new `eigsep_data.bundle`
  loader (Aaron's call) -- v007 is withdrawn and its callers' published
  numbers came from this code path.
- 2026-09-15 (`software-engineer`): added this file (README-convention
  retrofit, fleet-wide consolidation pass) — the B5 promotion itself hadn't
  had one until now.
