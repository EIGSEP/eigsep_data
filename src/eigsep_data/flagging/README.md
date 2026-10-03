# eigsep_data.flagging

Writer for the campaign's per-(time, channel) RFI flag masks,
`flags/v0/flags_YYYYMMDD.h5` (uint8 category bitfields). The reader is
`eigsep_data.products.flags`; the two share one layout. File-level
campaign masks (outages, SNAP flips, mux windows) are not encoded here;
`eigsep_data.select_files` owns those.

| Module | Purpose |
|---|---|
| `detectors.py` | Detection primitives: a temporal track for transients, a frequency track for persistent emitters, comb identification (`identify_combs`), and `categorise`, which assigns a bit per flagged pixel. Bit numbers are in `flag_bits.json` and must not change. |
| `build_masks.py` | Runs the detectors over every file and writes the day files, `flag_bits.json`, `manifest.json`, `summary.json` and `comb_detections.jsonl`. Reads the comb state per file from `curation/mode_table.jsonl`. |
| `validate.py` | Recall in labelled windows, sky-removal and leakage tests; writes `flags/v0/validation.json`. |

Combs (memo 001): `transmitter` is the beam-mapping transmitter, 8 channels
(1.953125 MHz), both antennas, 07-17 15:36 onward. `boxair_emi` is box-air's
own 1.000 MHz EMI, box-air only, 07-16 01:18-16:51. `categorise` still puts
the transmitter's teeth in bit 2 (self-RFI) and never sets bit 1 (`tx_comb`),
which is how `flags/v0` was built. Moving them to bit 1 would make them
non-RFI and needs a new flags version.

`detect_combs` is known broken (NameError) and is only called by
`validate.py`; a test pins this.

## Recent changes

- 2026-10-03 (Claude Code, for Aaron): comb labels renamed to match memo
  001: `digital_self` -> `transmitter`, `panda_emi` -> `boxair_emi`, and
  the docstrings that said neither comb was the transmitter are rewritten.
  `build_masks` now takes the transmitter state from the mode table's
  `transmitter` column instead of `tx_comb` (which is the box-air EMI, so
  the old code had the state inverted) and refuses a mode table without
  the new columns. `comb_detections.jsonl` columns change name to match;
  the manifest records the old names under `renamed_from`. Flag bits are
  unchanged. This README is new.
