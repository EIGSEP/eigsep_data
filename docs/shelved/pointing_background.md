# Shelved: pointing-aware background for the rotating-antenna raster

Branch `rfi-pointing-line`, not merged (2026-10-06). `RFIConfig.pointing_background`
fits a second background in segments where the elevation moves (Fourier harmonics of
elevation times the slow DPSS time modes, times v3's frequency modes, solved exactly
from the normal equations) and sets bit 10 `pointing_line` from its residuals.

On the two 07-17 raster batches (corr_20260717_203032Z-204952Z, 205410Z-211329Z):

- As an extra line detector it adds nothing: v3 already excludes 71-76% of raster sky
  cells (mostly `unsupported_background` and `positive_auto_excess`), and on the cells
  it keeps, v3 already recovers the planted lines that bit 10 would catch
  (`validate_pointing_line.py`).
- Replacing v3's bits 2, 3 and 7 with it in raster rows would exclude 26-36% instead,
  but the freed cells cannot be tested: planted lines at 30/100 radiometer sigma are
  recovered 4-8% / 25-36% of the time (`validate_replace.py`).
- The training mask has no good choice: starting from v3's mask biases the fit low
  where v3 flagged the modulation (synthetic test, xfail); starting from every tested
  cell lets the strong raster RFI wreck the fit.

Memo 006 records the conclusion. Run the scripts with this branch on `PYTHONPATH`.
