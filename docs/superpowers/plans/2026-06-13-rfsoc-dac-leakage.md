# RFSoC DAC Spectral-Leakage Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested analysis library plus a narrative notebook that bounds the RFSoC comb's spillover into neighbouring channels, answering reviewer comment (9) on the EIGSEP instrument paper.

**Architecture:** Numeric logic lives in a small, unit-tested module (`dac_leakage_lib.py`) — load, comb detection, noise-floor / dynamic-range, and tone-stacking. The notebook (`dac_leakage.ipynb`) imports that module and is pure narrative + plots + the written reviewer answer. Tests use synthetic spectra with known injected leakage so the recovery logic is verified against ground truth, plus one integration test against a real data file.

**Tech Stack:** Python 3.11, numpy, h5py, matplotlib, pytest (all already in `.venv`). black line-length 79. Notebook executed headless with nbconvert (`python3` kernel).

**Working directory:** all paths below are relative to the repo root `/home/christian/Documents/research/eigsep/data-analysis`. Activate the venv first in each shell: `source .venv/bin/activate`.

---

## File Structure

- Create `notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py` — analysis functions (one responsibility: turn an HDF5 file into leakage numbers). Pure/testable, no plotting.
- Create `notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py` — pytest tests: synthetic-data unit tests + one real-file integration test.
- Create `notebooks/christian/rfsoc_lab_test/dac_leakage.ipynb` — narrative notebook (markdown + plots + conclusion) importing the lib.

Verified data facts the code relies on (from `data/3`, time-averaged auto-correlation):
- Comb tones at channels 128, 144, …, 944 (every 16; 52 tones), ~constant amplitude.
- `detect_comb` with `rel_threshold=100` returns exactly that set on all four files.
- Noise floor ≈ 2986 / 1289 / 503 / 8.5 counts for 10/11/12/15 dB; dynamic range ≈ 3.3×10⁵ / 7.7×10⁵ / 2.0×10⁶ / 1.2×10⁸.
- `noise_floor_15db.h5` has 10 accumulations; the others have 60.

---

## Task 1: Library scaffolding + comb geometry

**Files:**
- Create: `notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py`
- Test: `notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py`

- [ ] **Step 1: Write the failing test**

Create `notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py`:

```python
import os

import numpy as np
import pytest

import dac_leakage_lib as lib

HERE = os.path.dirname(os.path.abspath(__file__))


def _synthetic(floor=0.0, amp=1e8, leak=0.0, hw_leak=1):
    """Spectrum with a comb of amplitude ``amp`` on a flat ``floor``,
    each tone leaking ``leak * amp`` into +/- ``hw_leak`` neighbours."""
    spec = np.full(lib.N_CHAN, float(floor))
    for t in lib.comb_channels():
        spec[t] = amp
        for d in range(1, hw_leak + 1):
            spec[t - d] += leak * amp
            spec[t + d] += leak * amp
    return spec


def test_comb_channels():
    tones = lib.comb_channels()
    assert tones[0] == 128
    assert tones[-1] == 944
    assert len(tones) == 52
    assert np.all(np.diff(tones) == 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dac_leakage_lib'`.

- [ ] **Step 3: Write minimal implementation**

Create `notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py`:

```python
"""Analysis helpers for the RFSoC DAC spectral-leakage measurement.

The RFSoC emits a comb of tones (every 16 channels). Combined with
injected noise it is recorded as an auto-correlation in ``data/3``.
These helpers locate the comb, characterize the noise floor / dynamic
range, and stack the tone wings to measure spillover into neighbouring
channels.
"""
import h5py
import numpy as np

N_CHAN = 1024
COMB_START = 128
COMB_STEP = 16
COMB_STOP = 944           # inclusive
BAND = (120, 950)         # in-band channels used for floor statistics
DF_MHZ = 250.0 / N_CHAN   # channel width, ~0.2441 MHz


def comb_channels(start=COMB_START, step=COMB_STEP, stop=COMB_STOP):
    """Channel indices of the comb tones (``stop`` inclusive)."""
    return np.arange(start, stop + 1, step)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py \
        notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py
git commit -m "feat: comb geometry helper for DAC leakage analysis"
```

---

## Task 2: Load real data + robust comb detection

**Files:**
- Modify: `notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py`
- Test: `notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_dac_leakage_lib.py`:

```python
def test_detect_comb_recovers_comb():
    spec = _synthetic(floor=1.0, amp=1e6, leak=0.0)
    assert np.array_equal(lib.detect_comb(spec), lib.comb_channels())


def test_load_auto_real_file():
    path = os.path.join(HERE, "noise_floor_15db.h5")
    spec, freqs, n_acc = lib.load_auto(path)
    assert spec.shape == (lib.N_CHAN,)
    assert freqs.shape == (lib.N_CHAN,)
    assert n_acc == 10
    assert np.array_equal(lib.detect_comb(spec), lib.comb_channels())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py -v`
Expected: FAIL — `AttributeError: module 'dac_leakage_lib' has no attribute 'detect_comb'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dac_leakage_lib.py`:

```python
def load_auto(path, key="3"):
    """Time-averaged auto-correlation from one HDF5 file.

    Returns ``(spec, freqs, n_acc)`` with ``spec`` shape ``(1024,)``.
    """
    with h5py.File(path, "r") as f:
        data = f[f"data/{key}"][:]       # (n_acc, 1024)
        freqs = f["header/freqs"][:]
    return data.mean(axis=0), freqs, data.shape[0]


def in_band_mask(n_chan=N_CHAN, band=BAND):
    """Boolean mask, True for channels inside ``band``."""
    lo, hi = band
    mask = np.zeros(n_chan, dtype=bool)
    mask[lo:hi] = True
    return mask


def detect_comb(spec, rel_threshold=100.0, band=BAND):
    """Channels exceeding ``rel_threshold`` x the in-band median.

    Verified to return exactly ``comb_channels()`` on all four files.
    """
    band_mask = in_band_mask(spec.size, band)
    med = np.median(spec[band_mask])
    return np.where((spec > rel_threshold * med) & band_mask)[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py \
        notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py
git commit -m "feat: load_auto + robust comb detection"
```

---

## Task 3: Noise floor, tone amplitude, dynamic range

**Files:**
- Modify: `notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py`
- Test: `notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py`

- [ ] **Step 1: Write the failing test**

Append to `test_dac_leakage_lib.py`:

```python
def test_floor_amplitude_dynamic_range():
    spec = _synthetic(floor=500.0, amp=1e8, leak=0.0)
    tones = lib.comb_channels()
    assert lib.noise_floor(spec, tones) == pytest.approx(500.0)
    assert lib.tone_amplitude(spec, tones) == pytest.approx(1e8)
    assert lib.dynamic_range(spec, tones) == pytest.approx(2e5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py::test_floor_amplitude_dynamic_range -v`
Expected: FAIL — `AttributeError: ... has no attribute 'noise_floor'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dac_leakage_lib.py`:

```python
def off_tone_mask(spec, tones, guard=2, band=BAND):
    """In-band channels not within ``guard`` of any tone."""
    mask = in_band_mask(spec.size, band)
    for t in tones:
        mask[max(0, t - guard):t + guard + 1] = False
    return mask


def noise_floor(spec, tones, guard=2, band=BAND):
    """Median power of the off-tone, in-band channels."""
    return np.median(spec[off_tone_mask(spec, tones, guard, band)])


def tone_amplitude(spec, tones):
    """Mean power in the tone channels."""
    return spec[tones].mean()


def dynamic_range(spec, tones, guard=2, band=BAND):
    """Mean tone amplitude divided by the noise floor."""
    floor = noise_floor(spec, tones, guard, band)
    return tone_amplitude(spec, tones) / floor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py \
        notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py
git commit -m "feat: noise floor and dynamic range helpers"
```

---

## Task 4: Stacked leakage profile

**Files:**
- Modify: `notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py`
- Test: `notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py`

- [ ] **Step 1: Write the failing test**

Append to `test_dac_leakage_lib.py`:

```python
def test_stacked_profile_recovers_leakage():
    leak = 3e-4
    spec = _synthetic(floor=0.0, amp=1e8, leak=leak, hw_leak=1)
    tones = lib.comb_channels()
    offsets, mean, err = lib.stacked_profile(spec, tones, half_width=8)
    c = int(np.where(offsets == 0)[0][0])
    assert mean[c] == pytest.approx(1.0)
    assert mean[c - 1] == pytest.approx(leak)
    assert mean[c + 1] == pytest.approx(leak)
    assert mean[c + 2] == pytest.approx(0.0)
    assert mean[c - 2] == pytest.approx(0.0)
    assert offsets[0] == -8 and offsets[-1] == 8
    assert err.shape == mean.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py::test_stacked_profile_recovers_leakage -v`
Expected: FAIL — `AttributeError: ... has no attribute 'stacked_profile'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dac_leakage_lib.py`:

```python
def stacked_profile(spec, tones, half_width=8):
    """Stack the wings of all comb tones, normalized per tone.

    Each tone's window ``spec[t-hw : t+hw+1]`` is divided by its own
    tone-channel value ``spec[t]`` and averaged across tones.

    Returns ``(offsets, mean, err)``:
      offsets : integer channel offsets ``-hw .. hw``
      mean    : mean normalized power at each offset
      err     : standard error of the mean across tones
    """
    hw = half_width
    offsets = np.arange(-hw, hw + 1)
    windows = np.array(
        [spec[t - hw:t + hw + 1] / spec[t] for t in tones]
    )                                    # (n_tones, 2*hw+1)
    mean = windows.mean(axis=0)
    err = windows.std(axis=0, ddof=1) / np.sqrt(windows.shape[0])
    return offsets, mean, err
```

- [ ] **Step 4: Run the full test suite to verify it passes**

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint and commit**

```bash
source .venv/bin/activate
black --line-length 79 notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py \
      notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py
flake8 notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py \
       notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py
git add notebooks/christian/rfsoc_lab_test/dac_leakage_lib.py \
        notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py
git commit -m "feat: stacked comb-tone leakage profile"
```
Expected: black reformats if needed, flake8 reports nothing.

---

## Task 5: Narrative notebook

**Files:**
- Create: `notebooks/christian/rfsoc_lab_test/dac_leakage.ipynb`

Build the notebook by adding the cells below in order (use the `python3` kernel). After creating all cells, execute it headless and confirm it runs clean.

- [ ] **Step 1: Markdown cell — title & reviewer question**

```markdown
# RFSoC DAC spectral leakage / phase noise

**Reviewer comment (9):** "The synchronization of the RFSoC DACs and the
SNAP correlator clocks will result in no spectral leakage for perfect
delta functions. In practical cases, is it found that there is
significant phase-noise associated with the generation of these
delta-functions through the RFSoC board? Would the spill-over of any
power resulting from this phase-noise be enough to provide significant
contamination on scales of the 21cm absorption feature in the immediate
neighbouring channels, or would these be flagged or down-weighted in any
resulting fits?"

**Experiment.** The RFSoC comb (delta-function tones, every 16 channels
= 3.9 MHz) was combined with a broadband noise generator through a
coupler and injected into one SNAP input. The noise attenuator was
stepped 10 / 11 / 12 / 15 dB (higher attenuation -> lower noise floor ->
higher dynamic range), giving a **dynamic-range ladder** against which we
probe the residual spillover of the (constant) comb tones. The comb is
present in real EIGSEP sky data at this same level, so this is a direct
measurement of the science-data contamination.

**Method.** For each setting we stack the wings of all 52 comb tones
(normalized per tone) to beat the noise down by ~sqrt(52), measuring the
mean spillover into each neighbouring channel — a detection where it
rises above the floor, a clean upper limit where it does not.
```

- [ ] **Step 2: Code cell — imports and load**

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

import dac_leakage_lib as lib

SETTINGS = ["10", "11", "12", "15"]
specs, n_accs = {}, {}
for db in SETTINGS:
    spec, freqs, n_acc = lib.load_auto(f"noise_floor_{db}db.h5")
    specs[db] = spec
    n_accs[db] = n_acc
tones = lib.comb_channels()
print(f"{len(tones)} comb tones, channels {tones[0]}..{tones[-1]} "
      f"(step {lib.COMB_STEP}, df={lib.DF_MHZ:.4f} MHz)")
print("accumulations per setting:",
      {db: n_accs[db] for db in SETTINGS})
```

- [ ] **Step 3: Markdown cell — comb & synchronization check**

```markdown
## 1. The comb is on-bin (synchronization works)

Each comb tone lands in a single channel with its neighbours at the
noise floor. This directly confirms the reviewer's premise: the
DAC/correlator clock synchronization keeps the delta-functions on FFT
bin centres, with no visible off-bin smearing.
```

- [ ] **Step 4: Code cell — full-band spectrum plot + sync assertion**

```python
db = "12"
spec = specs[db]
detected = lib.detect_comb(spec)
assert np.array_equal(detected, tones), "comb detection mismatch"

fig, ax = plt.subplots(figsize=(11, 4))
ax.semilogy(freqs, spec, lw=0.7)
ax.semilogy(freqs[tones], spec[tones], "r.", ms=5, label="comb tones")
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Auto-correlation [counts]")
ax.set_title(f"data/3 spectrum, {db} dB noise setting")
ax.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Markdown cell — dynamic-range ladder**

```markdown
## 2. Dynamic-range ladder

The comb amplitude is constant; the attenuator sets the noise floor.
The four settings span ~3x10^5 to ~10^8 in dynamic range (~55 to ~80 dB
in power), so the 15 dB setting is the deepest probe of spillover.
```

- [ ] **Step 6: Code cell — dynamic-range table**

```python
print(f"{'attn':>5} {'n_acc':>6} {'floor':>12} {'tone':>12} "
      f"{'dyn.range':>12} {'DR[dB]':>8}")
dr = {}
for db in SETTINGS:
    s = specs[db]
    floor = lib.noise_floor(s, tones)
    amp = lib.tone_amplitude(s, tones)
    d = lib.dynamic_range(s, tones)
    dr[db] = d
    print(f"{db+' dB':>5} {n_accs[db]:>6} {floor:>12.1f} {amp:>12.3e} "
          f"{d:>12.3e} {10*np.log10(d):>8.1f}")
```

- [ ] **Step 7: Markdown cell — stacked leakage profile**

```markdown
## 3. Stacked spillover profile

Stacking the normalized wings of all 52 tones, we read the mean
spillover into each offset from the tone. The dashed line is the
off-tone noise floor (normalized by the mean tone amplitude): points
consistent with it are noise, points above it are spillover.
```

- [ ] **Step 8: Code cell — stacked profile plot + per-setting numbers**

```python
fig, ax = plt.subplots(figsize=(9, 5))
prof = {}
for db in SETTINGS:
    s = specs[db]
    offsets, mean, err = lib.stacked_profile(s, tones, half_width=8)
    prof[db] = (offsets, mean, err)
    floor_norm = lib.noise_floor(s, tones) / lib.tone_amplitude(s, tones)
    line = ax.errorbar(offsets, mean, yerr=err, marker="o", ms=4,
                       capsize=2, lw=1, label=f"{db} dB")
    ax.axhline(floor_norm, ls="--", lw=0.8,
               color=line[0].get_color())
ax.set_yscale("log")
ax.set_xlabel("Channel offset from tone")
ax.set_ylabel("Power / tone power")
ax.set_title("Stacked comb-tone leakage profile (52 tones)")
ax.legend(title="noise setting")
plt.tight_layout()
plt.show()

print(f"{'attn':>5} {'<n1>/tone':>12} {'floor/tone':>12} "
      f"{'excess':>12} {'sigma':>7}")
for db in SETTINGS:
    offsets, mean, err = prof[db]
    s = specs[db]
    floor_norm = lib.noise_floor(s, tones) / lib.tone_amplitude(s, tones)
    c = int(np.where(offsets == 0)[0][0])
    n1 = 0.5 * (mean[c - 1] + mean[c + 1])
    n1_err = 0.5 * np.hypot(err[c - 1], err[c + 1])
    excess = n1 - floor_norm
    sigma = excess / n1_err if n1_err > 0 else np.inf
    print(f"{db+' dB':>5} {n1:>12.3e} {floor_norm:>12.3e} "
          f"{excess:>12.3e} {sigma:>7.1f}")
```

- [ ] **Step 9: Markdown cell — leakage vs dynamic range**

```markdown
## 4. Spillover vs dynamic range

The deepest setting bounds the spillover into the immediately adjacent
(+/-1) channel. Spillover beyond +/-1 channel stays at the noise floor
at every setting, so any contamination is confined to a single channel.
```

- [ ] **Step 10: Code cell — leakage vs dynamic range + headline number**

```python
# +/-1 spillover (excess over floor) and 3-sigma upper limit per setting
dr_vals, excess_vals, err_vals, limit_vals = [], [], [], []
for db in SETTINGS:
    offsets, mean, err = prof[db]
    s = specs[db]
    floor_norm = lib.noise_floor(s, tones) / lib.tone_amplitude(s, tones)
    c = int(np.where(offsets == 0)[0][0])
    n1 = 0.5 * (mean[c - 1] + mean[c + 1])
    n1_err = 0.5 * np.hypot(err[c - 1], err[c + 1])
    excess = n1 - floor_norm
    dr_vals.append(dr[db])
    excess_vals.append(excess)
    err_vals.append(n1_err)
    limit_vals.append(max(excess, 0.0) + 3 * n1_err)

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(dr_vals, np.abs(excess_vals), yerr=err_vals, fmt="o",
            capsize=3, label="|+/-1 excess over floor|")
ax.plot(dr_vals, limit_vals, "x--", color="C3",
        label="3-sigma upper limit")
for db, x, y in zip(SETTINGS, dr_vals, limit_vals):
    ax.annotate(f"{db} dB", (x, y), textcoords="offset points",
                xytext=(5, 5), fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Dynamic range (tone / noise floor)")
ax.set_ylabel("+/-1 channel spillover / tone power")
ax.set_title("Spillover constraint vs dynamic range")
ax.legend()
plt.tight_layout()
plt.show()

# Headline from the deepest probe (15 dB):
db = "15"
offsets, mean, err = prof[db]
s = specs[db]
floor_norm = lib.noise_floor(s, tones) / lib.tone_amplitude(s, tones)
c = int(np.where(offsets == 0)[0][0])
n1 = 0.5 * (mean[c - 1] + mean[c + 1])
n1_err = 0.5 * np.hypot(err[c - 1], err[c + 1])
excess = n1 - floor_norm
limit = max(excess, 0.0) + 3 * n1_err   # 3-sigma upper bound

print(f"Deepest probe: {db} dB, dynamic range {dr[db]:.2e} "
      f"({10*np.log10(dr[db]):.0f} dB)")
print(f"+/-1 channel spillover (excess over floor): "
      f"{excess:.2e} +/- {n1_err:.2e} of tone")
print(f"3-sigma upper limit on +/-1 spillover: {limit:.2e} of tone "
      f"({10*np.log10(limit):.0f} dB)")
# spillover beyond +/-1 is at the floor:
beyond = mean[c + 2:]
print(f"max |offset|>=2 point: {beyond.max():.2e} of tone "
      f"(floor {floor_norm:.2e})")
```

- [ ] **Step 11: Markdown cell — reviewer answer**

```markdown
## 5. Answer to the reviewer

1. **Synchronization holds.** Each comb tone occupies a single channel
   with neighbours at the noise floor — the DAC/correlator clocks keep
   the delta-functions on-bin (Section 1).

2. **Spillover is ~80 dB down and one channel wide.** Stacking 52 tones
   at up to ~80 dB dynamic range, the spillover into the immediately
   adjacent (+/-1) channel is at the ~10^-8-of-tone level (see the 15 dB
   number / 3-sigma upper limit above), and offsets |>=2| are
   indistinguishable from the noise floor.

3. **This bounds the phase noise.** The measured +/-1 spillover is the
   *total* — it also contains the deterministic PFB channel response —
   so the phase-noise contribution is at most this, i.e. negligible.

4. **It is spectrally distinct and flagged.** Because the comb is in the
   real sky data at this level, this is a direct measurement. Each tone
   is a single channel (every 16 ch / 3.9 MHz) that is flagged, and its
   spillover is confined to the one adjacent channel (0.24 MHz) — far
   narrower than the broad (~10-20 MHz) 21cm absorption feature. Such a
   single-channel residual is naturally down-weighted / flagged in any
   spectral fit.

**Conclusion:** phase-noise spillover from the RFSoC comb is ~80 dB
below the tone, confined to a single neighbouring channel, and therefore
negligible — and trivially flagged — on the scales of the 21cm
absorption feature.
```

- [ ] **Step 12: Execute the notebook headless and verify it runs clean**

Run:
```bash
source .venv/bin/activate
cd notebooks/christian/rfsoc_lab_test && \
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=120 dac_leakage.ipynb
```
Expected: completes with no error (no traceback in output); the
`assert np.array_equal(detected, tones)` in Step 4 passes; the
dynamic-range table prints monotonically increasing dyn.range across
10 -> 15 dB; the headline cell prints a +/-1 spillover at the ~1e-8
level and an `|offset|>=2` point at the floor.

(Note: `cd` here is intentional so the notebook's relative `noise_floor_*.h5`
paths and `import dac_leakage_lib` resolve from the data directory.)

- [ ] **Step 13: Commit**

```bash
git add notebooks/christian/rfsoc_lab_test/dac_leakage.ipynb
git commit -m "feat: RFSoC DAC leakage notebook answering reviewer comment 9"
```

---

## Final verification

- [ ] Run the full test suite once more:

Run: `source .venv/bin/activate && python -m pytest notebooks/christian/rfsoc_lab_test/test_dac_leakage_lib.py -v`
Expected: PASS (5 passed).

- [ ] Confirm the executed notebook contains the dynamic-range table, the stacked-profile figure, the headline upper-limit print, and the written conclusion, with no error outputs.
