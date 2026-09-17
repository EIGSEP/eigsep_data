# hfss_beam_maps

`bowtie_beam.npz` — Dominic's HFSS simulation of the bowtie transmitter
beam, 52 frequency slices on a HEALPix grid. Read it with
`eigsep_data.beam_sim.read_beam()`; `beam_mapping.tx_model.HFSSBeamSet`
consumes the same format.

## Contents

| Array | Shape | Meaning |
|---|---|---|
| `beam_cart` | (52, 3, 12288) | Cartesian E-field components per frequency slice. |
| `gain_th`, `gain_ph` | (52, 12288) | Spherical theta/phi gain components, summed and peak-normalized per frequency to serve as HFSS "truth" maps. |
| `freqs` | (52,) | Slice frequency in MHz: 50.78125–250.0, step 3.90625. |
| `nside` | scalar | HEALPix nside (`npix = 12 * nside**2`). |

`read_beam(drop_last=True)` removes the top slice (250.0 MHz), which is
HFSS's own Nyquist edge and has no matching correlator channel. What
remains matches `eigsep_observing`'s correlator `freqs[::16][13:]`
exactly.

## The frequency labels were wrong until 2026-09-16

`freqs` was uniformly low by one HFSS grid step (3.90625 MHz) — it had
been reverse-engineered from a stale notebook anchor rather than read
from the HFSS export filenames. Corrected in `0f8559f`, verified against
the original HFSS source CSVs and independently against deployment-5
data. **`beam_cart`, `gain_th`, `gain_ph` and `nside` were never
affected**; only the label changed, from 46.875–246.09375 to
50.78125–250.0 MHz. Anything that recorded a per-slice frequency before
that date is off by one step.

## This is the canonical copy

A byte-identical copy exists in `data-analysis/hfss_beam_maps/`, left
behind so notebook paths kept resolving across the 2026-09-17 repo
split. **If the beam map is reissued, it lands here and that copy is
deleted, not updated.**

Note that `beam_sim.py` resolves this file two directories above
`src/eigsep_data/`, which only exists in a source checkout — the path
does not survive a non-editable install. See the repo root README.

## Recent changes

- 2026-09-17 (`agent:eigsep-67`): added this file; recorded the
  canonical-copy rule after the repo split copied rather than moved the
  directory.
- 2026-09-16 (`Dominic Vazquez`): corrected the `freqs` labels, off by
  one 3.90625 MHz grid step (`0f8559f`).
