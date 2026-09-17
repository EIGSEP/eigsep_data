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
| `geometry.py` | Pointing fusion from motor, potentiometer, and IMU streams, plus the transmitter-heading container. |
| `mapper.py` | Non-negative theta/phi gain recovery from alternating transmitter arms (`PolarizationBeamMapper`). |
| `tx_model.py` | HFSS-backed forward model of the transmitter coupling and correlator waterfall (`HFSSBeamSet`, consumes Dominic's `hfss_beam_maps/bowtie_beam.npz` format). |
| `basis.py` | Low-rank template banks (PCA eigen-beams, `BeamPCA`) packed as an `HFSSBeamSet` for linear fitting. |
| `rfi.py` | Model-independent RFI flagging from raw correlator data (not beam residuals) — flags outlier times using off-comb monitor channels. |
| `fit.py` | Transmitter position and polarization fitting (JAX). |
| `diagnostics.py` | v007 campaign loading, joint fits, and the standard diagnostic figures. |

## Recent changes

- 2026-09-15 (`software-engineer`): added this file (README-convention
  retrofit, fleet-wide consolidation pass) — the B5 promotion itself hadn't
  had one until now.
