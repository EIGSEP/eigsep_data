# hfss_beam_maps

`bowtie_beam.npz` — Dominic's HFSS simulation of the beam of box-air's
bowtie antenna (the receiving antenna, not the transmitter), 52 frequency slices on a HEALPix grid. Read it with
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

## `tx_reflector_beam_with_top_ring.npz`: the transmitter antenna

Bahram's HFSS simulation of the transmitter antenna (the basket with the reflector), exported 2026-11-04,
with the **top ring** in place (the design has three rings). A run without the top ring is coming from
Bahram and will be added beside this file as `tx_reflector_beam_with_no_top_ring.npz` (or replace it).

- Excitation (confirmed by Bahram, 2026-11-05): 1 W into a 50 ohm port on the feed, one pair of blades
  driven, no balun in the model. The fields are for 1 W into the port with the port mismatch included,
  the same convention as Dominic's bowtie export.
- Same layout as `bowtie_beam.npz`, but with 51 slices instead of 52:

| Array | Shape | Meaning |
|---|---|---|
| `beam_cart` | (51, 3, 12288) | Cartesian E-field components rE (complex, mV) per frequency slice. |
| `gain_th`, `gain_ph` | (51, 12288) | 4 pi \|E_theta\|^2 / (2 eta 1 W) and the same for phi: realized gain (linear) per 1 W into the port. |
| `freqs` | (51,) | 50.78125–246.09375 MHz, step 3.90625 (no 250.0 MHz slice). |
| `nside` | scalar | 32. |

- **Read it with `read_beam(path, drop_last=False)`.** The default `drop_last=True` removes the last slice
  because the bowtie file ends at 250.0 MHz; this file ends at 246.09375 MHz, a real slice.
- Made with Dominic's HFSS-to-HEALPix converter (k = 8 nearest samples, inverse-distance-squared weights, nside 32),
  with two changes for this export: the theta < 0 half of the grid is dropped (it repeats theta > 0 at
  phi + 180, checked to rounding) and the rounded frequencies in the file names are snapped to the 3.90625 MHz grid.
- `gain_th`/`gain_ph` are computed from `beam_cart`. In the bowtie file they follow \|E_theta\|^2 and \|E_phi\|^2
  of its `beam_cart` to correlation 0.9999 but are scaled per accepted power; `read_beam` peak-normalizes
  them, so only the shape matters there.
- Checks: \|E\|^2 against the nearest exported sample at random pixels agrees within 1% (50.8 MHz), 2.4%
  (148.4 MHz) and 9.4% (246.1 MHz; sharper patterns, interpolation); realized gain straight up is
  −60.6 / −20.1 / −6.3 dBi at 50.8 / 148.4 / 246.1 MHz.
- The raw HFSS CSV export (about 620 MB) is not in git.

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

- 2026-11-06 (Richard): added `tx_reflector_beam_with_top_ring.npz`, Bahram's transmitter
  antenna simulation (with the top ring).
- 2026-10-02: corrected the description: this is box-air's bowtie antenna, not the
  transmitter's (Aaron). The file is unchanged.
- 2026-09-17 (`agent:eigsep-67`): added this file; recorded the
  canonical-copy rule after the repo split copied rather than moved the
  directory.
- 2026-09-16 (`Dominic Vazquez`): corrected the `freqs` labels, off by
  one 3.90625 MHz grid step (`0f8559f`).
