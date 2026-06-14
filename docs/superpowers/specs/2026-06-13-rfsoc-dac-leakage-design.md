# RFSoC DAC spectral leakage / phase-noise analysis

**Date:** 2026-06-13
**Branch:** `dac_test`
**Purpose:** Answer reviewer comment (9) on the EIGSEP instrument paper.

## Reviewer question

> Section 4.5: The synchronization of the RFSoC DACs and the SNAP correlator clocks
> will result in no spectral leakage for perfect delta functions. In practical cases, is
> it found that there is significant phase-noise associated with the generation of these
> delta-functions through the RFSoC board? Would the spill-over of any power resulting
> from this phase-noise be enough to provide significant contamination on scales of the
> 21cm absorption feature in the immediate neighbouring channels, or would these be
> flagged or down-weighted in any resulting fits?

## Experiment

The RFSoC outputs a comb of tones ("delta functions"). This comb was combined with a
broadband noise generator through a coupler and injected into one SNAP correlator input.
The noise generator's variable attenuator was stepped through 10, 11, 12, and 15 dB,
recorded in the filename. Higher attenuation → lower injected noise → lower correlator
noise floor → higher dynamic range. The four settings therefore form a **dynamic-range
ladder** against which the residual spillover of the (constant) comb tones can be probed.

**The comb is present in real EIGSEP sky data at the same level** — it is not a
lab-only calibration signal. The leakage characterized here is therefore a *direct*
measurement of the spillover contamination present in science observations, not a proxy
or conservative bound on it.

Data: `notebooks/christian/rfsoc_lab_test/noise_floor_{10,11,12,15}db.h5`.

## Data layout (verified)

Each HDF5 file contains:
- `data/3` — shape `(N_acc, 1024)` int32, the **auto-correlation of the comb+noise input**.
  This is the signal path we analyze (confirmed by the user).
- `data/1` — near-empty (tones only at ch 0/256/512, exact zeros elsewhere); not used.
- `data/13` — `(N_acc, 1024, 2)` cross-correlation; not used.
- `header/freqs` — 1024 channels, 0–249.76 MHz, df = 0.244 MHz/channel.
- `header/acc_cnt`, `header/times` — N_acc = 60 for 10/11/12 dB, 10 for 15 dB.

Established facts from `data/3`:
- Comb tones every **16 channels**, ch 128 → 944 (~50 tones), ~constant amplitude
  (6–10×10⁸ counts) independent of attenuator setting.
- Off-tone noise floor scales with the attenuator: ≈ 2986 / 1289 / 503 / 8.5 counts for
  10 / 11 / 12 / 15 dB → dynamic range ≈ 3.3×10⁵ → 1.2×10⁸ (≈ 55 → 80 dB).
- Stacking the wings of all tones, the immediately adjacent channel (±1) sits at the
  noise floor at 10 dB and pokes only a few counts above the floor at 15 dB; ±2 and
  beyond are at the floor. Spillover is at the ~10⁻⁸-of-tone (≈ −80 dB) level, confined
  to ±1 channel.

## Approach (chosen: A — stacked neighbor-ratio across the dynamic-range ladder)

Make the user's "dynamic range at which leakage appears" idea statistically rigorous by
stacking the wings of all comb tones to beat the noise down by ~√(N_tones). Produce a
measurement where the spillover is detected and a clean upper limit where it is not, with
the significance computed from the data (not asserted).

Rejected alternatives: (B) fitting a parametric phase-noise skirt — overreaches, the
signal is at the floor; (C) single-channel threshold without stacking — same idea as A
but with much weaker sensitivity.

## Notebook structure

`notebooks/christian/rfsoc_lab_test/dac_leakage.ipynb`

1. **Context (markdown).** Reviewer question verbatim; lab setup; analysis logic
   (sync → on-bin tones → measure residual spillover → bound phase noise).
2. **Load.** Loader for `data/3` + `freqs`/`acc_cnt` per file; time-average over
   accumulations. Note the 15 dB file has 10 accumulations vs 60.
3. **Comb & sync check.** Locate the comb channels; full-band log plot (one setting)
   showing each tone is a single-channel spike with neighbors at the floor — directly
   confirms the reviewer's premise that synchronization keeps tones on-bin.
4. **Dynamic-range ladder.** Per setting: off-tone median floor (masking ±a few channels
   around each tone and the band edges), mean tone amplitude, dynamic range. Summary table.
5. **Stacked leakage profile (core).** Per setting: extract a ±8-channel window around
   each tone, normalize each window by its own tone-channel value, average across all
   tones and time. Mean normalized power vs offset −8…+8, with error bars from the
   across-tone scatter (σ/√N_tones), plus an off-tone control at the same normalization.
   Read leakage at ±1 and upper limits at ±2, ±3, ….
6. **Leakage vs dynamic range.** Plot the ±1 excess-over-floor (normalized) vs dynamic
   range for the four settings. The 15 dB point (~80 dB DR) sets the tightest constraint.
   Headline: spillover into ±1 channel ≈ −80 dB (measurement or 3σ upper limit, whichever
   the error bars support); undetectable beyond ±1.
7. **Reviewer answer (markdown).** (a) sync confirmed, tones on-bin; (b) because the comb
   is present in real sky data at this level, the measurement applies directly to science
   observations; (c) total spillover ≈ −80 dB, confined to ±1 channel (0.24 MHz);
   (d) this is an upper bound on phase noise, since it also includes the deterministic PFB
   channel response; (e) each comb tone occupies a single channel (every 16 ch / 3.9 MHz)
   and is flagged; its spillover is confined to ±1 channel, spectrally distinct from the
   broad (~10–20 MHz) 21cm feature, and flagged/down-weighted in fits; (f) conclusion:
   negligible contamination on 21cm scales.

## Design decisions

- **`data/3` auto-correlation** is the product: phase noise manifests as a power skirt
  around the tone, which the auto captures directly.
- **Per-tone normalization** (divide each window by its own tone channel) handles the
  band-dependent tone amplitude so the stacked profile is a consistent power ratio.
- **Detection vs upper limit** is decided by the computed significance per setting, not
  asserted. The 15 dB setting (deepest dynamic range) provides the headline constraint.
- **Framing is relative** (dynamic range + spectral shape), requiring no external sky
  numbers, per the user's choice.

## Out of scope

- `data/1`, `data/13`, and the cross-correlation product.
- Absolute conversion to K / fraction of feature depth (relative argument only).
- Parametric phase-noise modeling.

## Success criteria

The notebook runs top-to-bottom and produces: the dynamic-range table, the stacked
leakage profile with error bars, the leakage-vs-dynamic-range plot, and a written
conclusion that addresses every clause of the reviewer's question (phase-noise level,
spillover magnitude, spectral scale vs the 21cm feature, and flagging/down-weighting).

## Outcome (as built)

The ~−80 dB / "confined to ±1, measurement-or-3σ-limit" figures above were the
pre-analysis design estimate; the final result, after a proper paired per-tone contrast
and a check of how the excess scales across the dynamic-range ladder, is:

- The comb is on-bin: each tone is a single channel with neighbours at the floor.
- A +/-1-channel excess over the in-gap far wing (offsets 2–8) is detected at ~7–9σ at
  every setting, measured with a **paired per-tone contrast** (`adjacent_excess`) so the
  common tone-to-tone noise-level variation cancels.
- The excess shrinks with dynamic range (≈2×10⁻⁷ of the tone at 10 dB → ≈5×10⁻⁹ at
  15 dB). It is **not** proportional to the noise floor (a pure-noise model fits poorly,
  χ²≈69/3; power-law slope ≈0.65), so the wording is kept mechanism-agnostic: part of the
  high-noise excess is the injected noise's channelizer response, with a small residual at
  the deepest setting. We do **not** claim "scales with noise" or a "still-falling
  conservative limit."
- Headline (deepest, lowest-noise 15 dB setting ≈ 80 dB DR): +/-1 spillover
  **4.96×10⁻⁹ ± 0.60×10⁻⁹ of the tone (≈ −83 dB)**, confined to ±1 channel (±2 and beyond
  show no significant excess). This total bounds the comb's phase-noise contribution.
- Conclusion to the reviewer: spillover ≲ 5×10⁻⁹ (≈ −83 dB), one channel wide — negligible
  and trivially flagged on the broad (~10–20 MHz) 21cm-feature scale.

Deliverables: `notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py` (8 passing tests),
`test_dac_leakage_lib.py`, `dac_leakage.ipynb`.
