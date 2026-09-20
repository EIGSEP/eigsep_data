"""RFI detection primitives for the Marjum Pass 2026-07 campaign.

Design notes (why this is shaped the way it is)
------------------------------------------------

The instrument sits ~100 m above the canyon floor, so environmental
reflections land beyond ~670 ns and imprint a *static* ripple on the
bandpass with a period of ~1.5 MHz (PROGRAM.md §5, EIGSEP paper §2.1).
That ripple is real instrument response, not interference.  A naive
frequency-domain residual detector flags it, which is signal loss
dressed up as caution.

So the primary detector works along **time**, per channel: a static
bandpass ripple is constant in time and cancels exactly under temporal
detrending, while genuine RFI is transient.  That gives a clean
detector for the site's dominant interference (airplane reflections,
meteor scatter, lightning, LIDAR pulses).

Persistent emitters (box fans, laptop combs) do *not* move in time and
are invisible to the temporal track, so a second frequency-domain
track runs on the time-median spectrum.  Rather than trusting raw
residual amplitude there -- which is where the reflection ripple would
bite us -- persistent detections are qualified by band membership and
by explicit comb matched-detection.

Comb attribution is its own track, and it is where this campaign has
been hardest to get right.  Two distinct comb episodes exist:

    07-16 01:18-16:51        1.000 MHz, WALKS (4.096 ch), box-air only
    07-17 15:37-07-18 03:00  1.953125 MHz, LOCKED (8 ch), both boxes

**Neither is established as the transmitter.**  Neither shows the
pointing dependence a far-field source must have, once a smooth time
trend is removed.  The locked/walking axis does NOT carry source
attribution: the TX is a clock-locked comb by design (PROGRAM.md §5),
while an internal device with its own oscillator -- the Panda -- emits
a walking one.  See the COMB_SPACINGS_MHZ block for the measured
numbers.  The claimed 1.25 MHz LNA-feedback and 2 MHz laptop combs are
not present in the filtered data at all.

Times come from filenames.  `header/times` is corrupt in 642/5120
files (May-2026 dates, negative spans), so it is never read here.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.ndimage import median_filter

# ---------------------------------------------------------------- constants

CHAN_WIDTH_MHZ = 250.0 / 1024.0  # 0.244140625
N_CHAN = 1024

# Category bit assignments for the per-pixel uint8 mask.
CLEAN = 0
CAL = 1 << 0          # receiver on loads/noise/VNA -- not sky, not RFI
TX_COMB = 1 << 1      # beam-mapping transmitter comb -- wanted signal
SELF_RFI = 1 << 2     # fan / laptop / LNA feedback / LIDAR / Panda
FM_DTV_MS = 1 << 3    # FM+DTV co-moving: meteor-scatter propagation
AIRPLANE = 1 << 4     # broadband transient reflection
ORBCOMM = 1 << 5      # 137-138 MHz satellite downlink
UNKNOWN = 1 << 6      # detected, not attributable
OVERFLOW = 1 << 7     # int32 accumulator wrap -- instrumental, not RFI

CATEGORY_NAMES = {
    CAL: "cal",
    TX_COMB: "tx_comb",
    SELF_RFI: "self-RFI",
    FM_DTV_MS: "FM-scatter",
    AIRPLANE: "airplane",
    ORBCOMM: "orbcomm",
    UNKNOWN: "unknown",
    OVERFLOW: "overflow",
}

# Bits that are NOT interference and must never enter RFI statistics.
NON_RFI_BITS = CAL | TX_COMB | OVERFLOW
RFI_BITS = SELF_RFI | FM_DTV_MS | AIRPLANE | ORBCOMM | UNKNOWN


def overflow_mask(raw):
    """Samples where the int32 auto accumulator wrapped.

    Correlator autos are non-negative by construction, so a negative
    value is an unambiguous wrap -- no threshold, no tuning.  Affects
    601/5120 files (20,852 samples, 0.0006% overall, up to 0.66% within
    a single file), concentrated on bright channels after the 07-15
    15:55 accumulator-length doubling.

    These must be excluded from the detectors' baselines as well as
    labelled: a wrap clamps to a huge *downward* excursion in log
    space, which both drags the robust scale and fires the transient
    detector, so leaving them in would inflate `unknown` with an
    instrumental artefact and corrupt the error accounting on both
    sides.
    """
    return np.asarray(raw) < 0

# Emitter bands, MHz. Used for categorisation only, never for blind masking.
BAND_FM = (88.0, 108.0)
BAND_ORBCOMM = (136.5, 138.5)
BAND_DTV_LO = (54.0, 88.0)     # US VHF channels 2-6
BAND_DTV_HI = (174.0, 216.0)   # US VHF channels 7-13
BAND_LAPTOP = (145.0, 160.0)   # Chrofaris laptop comb
BAND_FAN = (148.0, 152.0)      # box-fan RFI ~150 MHz

# Analysis band. Below ~45 MHz and above ~235 MHz the bandpass rolls off
# and residual statistics stop being meaningful.
BAND_ANALYSIS = (45.0, 235.0)

# Comb inventory. See flags/v0/COMB_INVENTORY.md for the full evidence.
#
# Two episodes exist, and NEITHER is established as the transmitter.
#
#   digital_self  1.953125 MHz = 8 channels EXACTLY (250/128 MHz)
#                 07-17 15:37 -> 07-18 03:00, BOTH boxes
#   panda_emi     1.000 MHz, walks (4.096 ch)
#                 07-16 01:18 -> 16:51, box-air ONLY, dies at the Panda
#                 power cycle
#
# Attribution rests on BEAM RESPONSE, tested correctly: a far-field
# source must modulate with pointing. Measured over the 07-17/18 scan,
# after removing a smooth time trend (which is what an earlier version
# of this analysis failed to do, and it inverted the answer):
#
#   box-air residual vs AZ: var explained -0.139, permutation z = -1.58
#   box-gnd residual vs AZ: var explained -0.244, z = -0.86 (null control)
#
# i.e. NO pointing dependence. A source that does not modulate with
# pointing is on the platform or conducted -- self-generated. The raw
# (non-detrended) az correlation of +0.48 was a time-trend confound:
# variance explained by TIME was 0.855, higher than by AZ (0.642), and
# the non-rotating ground box showed a spurious AZ dependence of 0.364.
#
# "Walking" does NOT imply external. It implies "not referenced to OUR
# ADC clock", which includes internal devices running their own
# oscillators -- exactly the Panda. That is why panda_emi walks while
# being self-generated, and it is why the locked/walking axis cannot
# carry source attribution on its own.
#
# Consequence: no comb in this campaign is demonstrated to be the TX,
# so TX_COMB is defined but never set. This is consistent with
# beam-analyst's finding that the v007 beam fits track radiated
# self-RFI rather than the transmitter.
COMB_SPACINGS_MHZ = {
    "digital_self": 250.0 / 128.0,  # 1.953125 MHz = 8 channels exactly
    "panda_emi": 1.000,     # walks (4.096 ch); box-air only; 07-16 01:18-16:51
    "lna_feedback": 1.250,  # claimed 07-16 01:00-01:30; NOT FOUND in the data
    "laptop": 2.000,        # claimed 07-13, 145-160 MHz; NOT FOUND in the data
}

# Spacings that are an exact integer number of channels. For these the
# teeth sit at fixed channel indices; for the others they walk.
LOCKED_CHAN = {"digital_self": 8}

SELF_SPACING_CHAN = 8
PANDA_SPACING_MHZ = 1.000


def tooth_contrast(med_logp, freqs, band=BAND_ANALYSIS, spacing_chan=None,
                   spacing_mhz=None, n_phase=16):
    """Best-phase tooth contrast, in units of the residual MAD.

    Replaces the periodogram for comb identification. A periodogram
    quantises the period to an FFT bin, which over a ~550-channel span
    cannot separate the 4.000 and 4.096 channel hypotheses -- that
    ambiguity produced two rounds of contradictory comb numbers across
    the fleet. Tooth contrast scans the phase explicitly and treats
    locked and walking hypotheses identically.

    Returns ``(contrast, phase, teeth_mask)``.
    """
    chans = np.arange(med_logp.size)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if in_band.sum() < 32:
        return 0.0, None, np.zeros(med_logp.size, dtype=bool)
    width = (2 * spacing_chan + 1) if spacing_chan else \
        max(int(round(2 * spacing_mhz / CHAN_WIDTH_MHZ)) | 1, 5)
    resid = med_logp - median_filter(med_logp, size=width, mode="nearest")
    scale = 1.4826 * float(np.median(
        np.abs(resid[in_band] - np.median(resid[in_band]))))
    if scale < 1e-12:
        return 0.0, None, np.zeros(med_logp.size, dtype=bool)
    best = (-np.inf, None, np.zeros(med_logp.size, dtype=bool))
    phases = range(spacing_chan) if spacing_chan else \
        [i * spacing_mhz / n_phase for i in range(n_phase)]
    for ph in phases:
        if spacing_chan:
            teeth = in_band & (chans % spacing_chan == ph)
        else:
            off = (freqs - ph) / spacing_mhz
            teeth = in_band & (
                np.abs(off - np.round(off)) * spacing_mhz <= 0.5 * CHAN_WIDTH_MHZ)
        other = in_band & ~teeth
        if teeth.sum() < 8 or other.sum() < 8:
            continue
        c = (np.median(resid[teeth]) - np.median(resid[other])) / scale
        if c > best[0]:
            best = (float(c), ph, teeth)
    return best


def identify_combs(med_logp, freqs, thresh=3.0):
    """Which combs are present, and which channels each occupies.

    Returns ``{"tx": {...}, "panda_emi": {...}, ...}`` with a
    ``contrast``, ``detected`` flag and ``teeth`` mask per comb.

    Selection logic. The TX comb at 8 channels also lights up every
    16th, 32nd and 64th channel detector, because those teeth are
    subsets of its own -- so a "3.9 MHz comb" detection is not
    independent evidence of anything. Conversely the walking 1.000 MHz
    Panda comb lights up a 2.000 MHz detector as its second harmonic.
    Each comb is therefore credited only against its own fundamental,
    and the harmonics are not reported as separate combs.
    """
    out = {}
    c_tx, ph_tx, teeth_tx = tooth_contrast(
        med_logp, freqs, spacing_chan=SELF_SPACING_CHAN)
    out["digital_self"] = {
        "contrast": round(c_tx, 3), "phase": ph_tx,
        "detected": bool(c_tx >= thresh), "teeth": teeth_tx,
        "spacing_mhz": 250.0 / 128.0, "channel_locked": True}
    c_p, ph_p, teeth_p = tooth_contrast(
        med_logp, freqs, spacing_mhz=PANDA_SPACING_MHZ)
    # The 8-channel comb's teeth partially coincide with a 1 MHz grid,
    # so only credit the Panda comb when it is not the other being
    # re-found.
    panda = bool(c_p >= thresh and c_p > c_tx)
    out["panda_emi"] = {"contrast": round(c_p, 3), "phase": ph_p,
                        "detected": panda, "teeth": teeth_p,
                        "spacing_mhz": PANDA_SPACING_MHZ,
                        "channel_locked": False}
    for name in ("lna_feedback", "laptop"):
        c, ph, teeth = tooth_contrast(
            med_logp, freqs, spacing_mhz=COMB_SPACINGS_MHZ[name])
        det = bool(c >= thresh and c > c_tx and c > c_p)
        out[name] = {"contrast": round(c, 3), "phase": ph, "detected": det,
                     "teeth": teeth, "spacing_mhz": COMB_SPACINGS_MHZ[name],
                     "channel_locked": False}
    return out


_FNAME_RE = re.compile(r"corr_(\d{8})_(\d{6})Z")


# ------------------------------------------------------------------- timing

def file_close_time(name: str) -> datetime:
    """UTC close time parsed from the filename.

    Authoritative: `header/times` is corrupt in 642/5120 files.
    """
    m = _FNAME_RE.search(str(name))
    if not m:
        raise ValueError(f"cannot parse close time from {name!r}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def sample_times(name: str, n_time: int, integration_s: float) -> np.ndarray:
    """Per-spectrum UTC timestamps, derived from the filename close time.

    The file is stamped with its *close* time, so sample ``i`` of ``n``
    lands at ``t_close - (n - 1 - i) * dt``.  Uniform spacing within a
    file is assumed; that is a property of the correlator accumulator,
    not of the (untrustworthy) header.
    """
    t_close = file_close_time(name)
    dt = integration_s / max(n_time - 1, 1)
    base = t_close.timestamp()
    return base - (n_time - 1 - np.arange(n_time)) * dt


# -------------------------------------------------------------- rfsw states

def antenna_mask(rfswitch, n_time: int) -> np.ndarray:
    """True where the receiver is looking at the antenna (RFANT).

    Everything else -- RFAMB, RFNON, RFSP1, VNA* -- is a calibration
    state.  Those samples are not sky, so they must not enter sky-RFI
    statistics, and the noise source's own signature must not be
    allowed to set the detectors' baselines.
    """
    if rfswitch is None:
        # No rfswitch stream (early Phase A). Assume antenna; the mode
        # table's cal windows still gate these files at a coarser level.
        return np.ones(n_time, dtype=bool)
    states = rfswitch
    if isinstance(states, (bytes, str)):
        try:
            states = json.loads(
                states.decode() if isinstance(states, bytes) else states
            )
        except (ValueError, AttributeError):
            return np.ones(n_time, dtype=bool)
    states = list(states)
    if len(states) == 0:
        return np.ones(n_time, dtype=bool)
    # The metadata stream is sampled on its own cadence; resample by
    # nearest-neighbour onto the spectrum axis.
    idx = np.minimum(
        (np.arange(n_time) * len(states) // max(n_time, 1)), len(states) - 1
    )
    arr = np.asarray(states, dtype=object)[idx]
    return np.array([str(s).upper() == "RFANT" for s in arr], dtype=bool)


# ------------------------------------------------------------------- tracks

def _mad(x, axis=None):
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis, keepdims=True)


def transient_track(logp, valid_time, med_width=9, clip_sigma=5.0):
    """Per-channel departures from a smooth time trend.

    Returns ``(flags, residual, scale)``.

    A static bandpass ripple -- the ~1.5 MHz environmental-reflection
    structure that the 100 m suspension is *designed* to produce -- is
    constant in time and cancels here exactly.  That is the whole point
    of detecting along time rather than frequency.
    """
    nt, nch = logp.shape
    flags = np.zeros((nt, nch), dtype=bool)
    resid = np.zeros((nt, nch), dtype=np.float32)
    scale = np.zeros(nch, dtype=np.float32)
    if valid_time.sum() < max(med_width + 2, 12):
        return flags, resid, scale

    sub = logp[valid_time]
    model = median_filter(sub, size=(med_width, 1), mode="nearest")
    r = sub - model
    s = _mad(r, axis=0)  # (1, nch)
    s = np.where(s < 1e-9, np.inf, s)
    flags[valid_time] = np.abs(r) > clip_sigma * s
    resid[valid_time] = r.astype(np.float32)
    scale[:] = np.squeeze(s).astype(np.float32)
    return flags, resid, scale


def persistent_track(logp, valid_time, freqs, smooth_width=31, clip_sigma=6.0):
    """Narrowband features that are steady over the whole file.

    Operates on the time-median spectrum, so transients average away
    and only always-on emitters survive.  Uses a higher clip than the
    transient track because the frequency axis carries genuine
    instrumental structure that we do not want to shave off.
    """
    nch = logp.shape[1]
    chan_flags = np.zeros(nch, dtype=bool)
    if valid_time.sum() < 4:
        return chan_flags, np.zeros(nch, dtype=np.float32)
    med = np.median(logp[valid_time], axis=0)
    smooth = median_filter(med, size=smooth_width, mode="nearest")
    resid = med - smooth
    in_band = (freqs >= BAND_ANALYSIS[0]) & (freqs <= BAND_ANALYSIS[1])
    if in_band.sum() < 32:
        return chan_flags, resid.astype(np.float32)
    s = float(_mad(resid[in_band]))
    if not np.isfinite(s) or s < 1e-9:
        return chan_flags, resid.astype(np.float32)
    chan_flags = (resid > clip_sigma * s) & in_band
    return chan_flags, resid.astype(np.float32)


# --------------------------------------------------------------------- comb

def comb_tone_channels(freqs, spacing_mhz, band, tol_chan=1.0, phase_mhz=0.0):
    """Channels within ``tol_chan`` of a comb tone.

    Tones are at ``phase_mhz + n * spacing_mhz``.  Because the spacing
    is not an integer number of channels (1.000 MHz = 4.096 channels),
    this is computed in frequency, never in channel index.
    """
    lo, hi = band
    in_band = (freqs >= lo) & (freqs <= hi)
    off = (freqs - phase_mhz) / spacing_mhz
    dist_tones = np.abs(off - np.round(off)) * spacing_mhz  # MHz to nearest tone
    return in_band & (dist_tones <= tol_chan * CHAN_WIDTH_MHZ)


def comb_snr(med_logp, freqs, spacing_mhz, band=(50.0, 200.0)):
    """Periodogram SNR for a comb of the given spacing.

    Takes the *median log spectrum*, not a pre-detrended residual:
    detrending twice flattens the comb along with the continuum and
    was the reason an earlier version of this detector reported
    nothing on windows where a comb is plainly visible.
    """
    lo, hi = band
    sel = (freqs >= lo) & (freqs <= hi)
    if sel.sum() < 64:
        return 0.0
    w = med_logp[sel].astype(float)
    if not np.isfinite(w).all():
        return 0.0
    w = w - median_filter(w, size=31, mode="nearest")
    w = w - w.mean()
    n = w.size
    spec = np.abs(np.fft.rfft(w * np.hanning(n)))
    spec[0] = 0.0
    baseline = float(np.median(spec[1:]))
    if baseline < 1e-12:
        return 0.0
    k = (n * CHAN_WIDTH_MHZ) / spacing_mhz
    k0 = max(int(np.floor(k)) - 1, 1)
    k1 = min(int(np.ceil(k)) + 2, spec.size)
    if k1 <= k0:
        return 0.0
    return float(spec[k0:k1].max() / baseline)


def detect_combs(med_logp, freqs, snr_thresh=8.0, tx_on=False):
    """Matched detection of the campaign's known comb spacings.

    Returns ``{name: {"snr":…, "spacing_mhz":…, "detected":bool}}``.

    Detection is by periodicity, which is what actually distinguishes a
    1.00 MHz comb from a 1.25 MHz one -- they differ by roughly one
    channel per tone, so matching individual peak positions is not
    enough.

    **Harmonic guard.** The TX comb at 1.000 MHz puts real power at
    0.500 and 2.000 MHz, and 2.000 MHz is exactly the laptop-comb
    spacing.  Measured on TX-on Phase-C files the harmonic sits at
    roughly half the fundamental's SNR (e.g. 1.000 MHz -> 77, 2.000 MHz
    -> 42).  So when TX is on, a 2 MHz detection is only credited if it
    is strong *relative to* the fundamental; otherwise we would label
    the beam-mapping transmitter as somebody's laptop.
    """
    out = {}
    for name, spacing in COMB_SPACINGS_MHZ.items():
        # Wide band throughout: the laptop band alone (145-160 MHz, 61
        # channels) is too few samples for a periodogram, which made an
        # earlier version return exactly 0.0 for every file.
        snr = comb_snr(med_logp, freqs, spacing, band=(50.0, 200.0))
        out[name] = {
            "snr": round(snr, 2),
            "spacing_mhz": round(spacing, 6),
            "channel_locked": name in CHANNEL_LOCKED_COMBS,
            "detected": bool(snr >= snr_thresh),
        }
    # Harmonic guards. A strong fundamental leaks into its sub- and
    # super-harmonics and into neighbouring periodogram bins; without
    # these, the TX transmitter gets relabelled as somebody's laptop and
    # an LNA fault gets invented out of a sidelobe.
    tx_snr = max(out["tx"]["snr"], 1e-9)
    if tx_on or out["tx"]["detected"]:
        for child in ("laptop", "lna_feedback"):
            if out[child]["snr"] < 0.8 * tx_snr:
                out[child]["detected"] = False
                out[child]["suppressed_as_tx_harmonic"] = True
    self_snr = max(out["digital_self"]["snr"], 1e-9)
    if out["digital_self"]["detected"]:
        # The 8-channel digital comb has Fourier power at periods 8, 4 and
        # 2 channels, so it leaks into the 1.000 MHz (4.096 ch) TX
        # statistic and can push it over threshold. Left unguarded this
        # relabels self-generated RFI as `tx_comb`, which is a *non-RFI*
        # bit -- i.e. it would quietly un-flag real interference.
        # Affects 2/10939 records, both on 07-17/18.
        if out["tx"]["snr"] < self_snr:
            out["tx"]["detected"] = False
            out["tx"]["suppressed_as_digital_harmonic"] = True
        # 1.953125 and 2.000 MHz differ by <1 periodogram bin over a
        # 150 MHz span, so a channel-locked comb always drags the
        # "laptop" statistic up with it.
        if out["laptop"]["snr"] < 1.2 * self_snr:
            out["laptop"]["detected"] = False
            out["laptop"]["suppressed_as_digital_harmonic"] = True
    return out


# ----------------------------------------------------------- categorisation

def _in(freqs, band):
    return (freqs >= band[0]) & (freqs <= band[1])


def categorise(pix_flags, chan_flags, freqs, combs, tx_on, broadband_time,
               ms_times=None):
    """Assign a category bit to every flagged pixel.

    Precedence is deliberate: attributable physical causes first,
    ``unknown`` only as the residue.  A large ``unknown`` fraction is
    information, not failure -- it is the honest statement that a
    detector fired and we cannot say why.
    """
    nt, nch = pix_flags.shape
    cat = np.zeros((nt, nch), dtype=np.uint8)

    fm = _in(freqs, BAND_FM)
    dtv = _in(freqs, BAND_DTV_LO) | _in(freqs, BAND_DTV_HI)
    orb = _in(freqs, BAND_ORBCOMM)
    laptop = _in(freqs, BAND_LAPTOP)
    fan = _in(freqs, BAND_FAN)

    # --- persistent, always full-file in extent -------------------------
    # Comb teeth come from identify_combs(), which returns an explicit
    # channel mask per comb. Never recompute them from a spacing here:
    # the TX comb is channel-locked and the Panda comb walks, so a
    # single tone-position rule cannot serve both.
    self_chan = np.zeros(nch, dtype=bool)
    for name in ("digital_self", "panda_emi", "lna_feedback", "laptop"):
        rec = combs.get(name) or {}
        if rec.get("detected") and rec.get("teeth") is not None:
            self_chan |= np.asarray(rec["teeth"], dtype=bool)
    self_chan |= chan_flags & (laptop | fan)

    # --- TX comb --------------------------------------------------------
    # Deliberately never set. No comb in this campaign shows the beam
    # response a far-field transmitter must have, so labelling any of
    # them `tx_comb` -- a NON-RFI bit -- would un-flag real
    # interference. If a TX comb is later identified, set tx_chan here.
    tx_present = False
    tx_chan = np.zeros(nch, dtype=bool)

    both = pix_flags | chan_flags[None, :]

    # Assign in reverse precedence so earlier categories overwrite later.
    cat[both & orb[None, :]] = ORBCOMM
    cat[both & (fm | dtv)[None, :]] = UNKNOWN  # refined just below
    if ms_times is not None and ms_times.any():
        ms = np.zeros((nt, nch), dtype=bool)
        ms[ms_times[:, None] & (fm | dtv)[None, :]] = True
        cat[both & ms] = FM_DTV_MS
    if broadband_time is not None and broadband_time.any():
        ap = broadband_time[:, None] & (fm | dtv)[None, :]
        cat[both & ap] = AIRPLANE
    cat[both & self_chan[None, :]] = SELF_RFI
    if tx_present:
        cat[both & tx_chan[None, :]] = TX_COMB

    # anything still flagged but unlabelled
    cat[both & (cat == 0)] = UNKNOWN
    cat[~both] = CLEAN
    return cat


def broadband_times(pix_flags, freqs, frac=0.15):
    """Time samples where a large fraction of the band lit up at once.

    Airplane reflections and lightning are broadband and short; the TX
    comb and the self-RFI combs are narrowband and steady.  This is the
    morphological discriminator between them.
    """
    sel = _in(freqs, BAND_ANALYSIS)
    if sel.sum() == 0:
        return np.zeros(pix_flags.shape[0], dtype=bool)
    return pix_flags[:, sel].mean(axis=1) > frac


def meteor_scatter_times(pix_flags, freqs, min_frac=0.05):
    """Times where FM *and* DTV bands rise together.

    Under micrometeor scattering, distant transmitters flash in and the
    FM and DTV bands move up and down *coherently* -- that joint
    behaviour is the signature, and it is what separates a propagation
    event from a local broadband transient.
    """
    fm = _in(freqs, BAND_FM)
    dtv = _in(freqs, BAND_DTV_LO) | _in(freqs, BAND_DTV_HI)
    if fm.sum() == 0 or dtv.sum() == 0:
        return np.zeros(pix_flags.shape[0], dtype=bool)
    f_fm = pix_flags[:, fm].mean(axis=1)
    f_dtv = pix_flags[:, dtv].mean(axis=1)
    return (f_fm > min_frac) & (f_dtv > min_frac)
