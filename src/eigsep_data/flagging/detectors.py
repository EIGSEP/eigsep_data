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
been hardest to get right.  Two distinct combs exist (memo 001, § Combs):

    07-16 01:18-16:51        1.000 MHz, WALKS (4.096 ch), box-air only
                             -> ``boxair_emi``: box-air's own EMI
    07-17 15:36-end of data  1.953125 MHz, LOCKED (8 ch), both antennas
                             -> ``transmitter``: the beam-mapping transmitter

The 8-channel comb is the transmitter: it switches on in both antennas
at once, box-gnd (on the ground near the transmitter) sees it at up to
~27 dB, its alternate teeth (= 0 / 8 mod 16) are the transmitter's two
arms, and it follows the 07-17 drive-attenuation tests.  The 1.000 MHz
comb is self-EMI: box-air only, never on box-gnd, gone when the Panda
was power-cycled.  Labels before 2026-10-03 had these the other way
round (``digital_self`` / ``panda_emi`` and "neither is the TX").  The
claimed 1.25 MHz LNA-feedback and 2 MHz laptop combs are not present
in the filtered data at all.

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
SELF_RFI = 1 << 2     # fan / laptop / LNA feedback / LIDAR / box-air EMI
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

# Comb inventory (memo 001, § Combs; flags/v0/COMB_INVENTORY.md predates
# the identification and uses the old labels).
#
#   transmitter  1.953125 MHz = 8 channels EXACTLY (250/128 MHz)
#                07-17 15:36 -> end of data, BOTH antennas; the
#                beam-mapping transmitter. Teeth = 0 mod 16 are arm 0,
#                = 8 mod 16 arm 1.
#   boxair_emi   1.000 MHz, walks (4.096 ch)
#                07-16 01:18 -> 16:51, box-air ONLY, gone at the Panda
#                power cycle; box-air's own electronics
#
# Until 2026-10-03 these were labelled `digital_self` and `panda_emi`,
# and the 8-channel comb was taken to be self-generated because a
# detrended box-air residual showed no azimuth dependence. Memo 001
# overturned that: box-gnd, which does not move, sees the 8-channel comb
# at up to ~27 dB and tracks the transmitter's on/off and drive tests,
# and on box-air the tooth/gap ratio swings with the raster's elevation
# sweeps.
#
# "Walking" does NOT imply external. It implies "not referenced to OUR
# ADC clock", which includes internal devices running their own
# oscillators -- which is why boxair_emi walks while being self-generated.
#
# Flag bits are unchanged by the rename. TX_COMB (bit 1) is still never
# set by categorise(): the transmitter teeth are detected as the
# `transmitter` comb and land in SELF_RFI (bit 2), so on 07-17/18 bit 2
# holds the transmitter teeth. That is what flags/v0 contains; moving
# them to bit 1 is a separate decision (it would make them non-RFI).
COMB_SPACINGS_MHZ = {
    "transmitter": 250.0 / 128.0,  # 1.953125 MHz = 8 channels exactly
    "boxair_emi": 1.000,    # walks (4.096 ch); box-air only; 07-16 01:18-16:51
    "lna_feedback": 1.250,  # claimed 07-16 01:00-01:30; NOT FOUND in the data
    "laptop": 2.000,        # claimed 07-13, 145-160 MHz; NOT FOUND in the data
}

# Spacings that are an exact integer number of channels. For these the
# teeth sit at fixed channel indices; for the others they walk.
LOCKED_CHAN = {"transmitter": 8}

TRANSMITTER_SPACING_CHAN = 8
BOXAIR_EMI_SPACING_MHZ = 1.000


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

    Returns ``{"transmitter": {...}, "boxair_emi": {...}, ...}`` with a
    ``contrast``, ``detected`` flag and ``teeth`` mask per comb.

    Selection logic. The TX comb at 8 channels also lights up every
    16th, 32nd and 64th channel detector, because those teeth are
    subsets of its own -- so a "3.9 MHz comb" detection is not
    independent evidence of anything. Conversely the walking 1.000 MHz
    box-air EMI comb lights up a 2.000 MHz detector as its second harmonic.
    Each comb is therefore credited only against its own fundamental,
    and the harmonics are not reported as separate combs.
    """
    out = {}
    c_tx, ph_tx, teeth_tx = tooth_contrast(
        med_logp, freqs, spacing_chan=TRANSMITTER_SPACING_CHAN)
    out["transmitter"] = {
        "contrast": round(c_tx, 3), "phase": ph_tx,
        "detected": bool(c_tx >= thresh), "teeth": teeth_tx,
        "spacing_mhz": 250.0 / 128.0, "channel_locked": True}
    c_p, ph_p, teeth_p = tooth_contrast(
        med_logp, freqs, spacing_mhz=BOXAIR_EMI_SPACING_MHZ)
    # The 8-channel comb's teeth partially coincide with a 1 MHz grid,
    # so only credit the box-air EMI comb when it is not the other being
    # re-found.
    boxair = bool(c_p >= thresh and c_p > c_tx)
    out["boxair_emi"] = {"contrast": round(c_p, 3), "phase": ph_p,
                         "detected": boxair, "teeth": teeth_p,
                         "spacing_mhz": BOXAIR_EMI_SPACING_MHZ,
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


def detect_combs(med_logp, freqs, snr_thresh=8.0, boxair_emi_on=False):
    """Matched detection of the campaign's known comb spacings.

    Returns ``{name: {"snr":…, "spacing_mhz":…, "detected":bool}}``.

    Detection is by periodicity, which is what actually distinguishes a
    1.00 MHz comb from a 1.25 MHz one -- they differ by roughly one
    channel per tone, so matching individual peak positions is not
    enough.

    **Harmonic guard.** The 1.000 MHz comb puts real power at 0.500
    and 2.000 MHz, and 2.000 MHz is exactly the laptop-comb spacing.
    Measured on 07-16 Phase-C files the harmonic sits at roughly half
    the fundamental's SNR (e.g. 1.000 MHz -> 77, 2.000 MHz -> 42).  So
    when that comb is on, a 2 MHz detection is only credited if it is
    strong *relative to* the fundamental.  (This text was written when
    the 1.000 MHz comb was taken to be the transmitter; it is box-air's
    own EMI -- memo 001.)

    Known broken: ``CHANNEL_LOCKED_COMBS`` is defined nowhere, so this
    raises NameError on first use. Kept as
    migrated; only :mod:`.validate` calls it.
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
    # these, the 1 MHz comb gets relabelled as somebody's laptop and
    # an LNA fault gets invented out of a sidelobe.
    emi_snr = max(out["boxair_emi"]["snr"], 1e-9)
    if boxair_emi_on or out["boxair_emi"]["detected"]:
        for child in ("laptop", "lna_feedback"):
            if out[child]["snr"] < 0.8 * emi_snr:
                out[child]["detected"] = False
                out[child]["suppressed_as_boxair_emi_harmonic"] = True
    self_snr = max(out["transmitter"]["snr"], 1e-9)
    if out["transmitter"]["detected"]:
        # The 8-channel transmitter comb has Fourier power at periods 8,
        # 4 and 2 channels, so it leaks into the 1.000 MHz (4.096 ch)
        # statistic and can push it over threshold. Left unguarded this
        # relabels the transmitter as the 1 MHz comb.
        # Affects 2/10939 records, both on 07-17/18.
        if out["boxair_emi"]["snr"] < self_snr:
            out["boxair_emi"]["detected"] = False
            out["boxair_emi"]["suppressed_as_transmitter_harmonic"] = True
        # 1.953125 and 2.000 MHz differ by <1 periodogram bin over a
        # 150 MHz span, so a channel-locked comb always drags the
        # "laptop" statistic up with it.
        if out["laptop"]["snr"] < 1.2 * self_snr:
            out["laptop"]["detected"] = False
            out["laptop"]["suppressed_as_transmitter_harmonic"] = True
    return out


# ----------------------------------------------------------- categorisation

def _in(freqs, band):
    return (freqs >= band[0]) & (freqs <= band[1])


def categorise(pix_flags, chan_flags, freqs, combs, tx_on, broadband_time,
               ms_times=None):
    """Assign a category bit to every flagged pixel.

    ``tx_on`` is whether the beam-mapping transmitter is on for this
    file (mode_table column ``transmitter``). It is currently unused:
    see the TX comb block below.

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
    # the transmitter comb is channel-locked and the box-air EMI comb
    # walks, so a single tone-position rule cannot serve both.
    #
    # The transmitter's teeth are included here, i.e. labelled SELF_RFI
    # (bit 2). That is how flags/v0 was built, under the old belief that
    # the 8-channel comb was self-generated; it is kept so the code still
    # reproduces v0. See the COMB_SPACINGS_MHZ block.
    self_chan = np.zeros(nch, dtype=bool)
    for name in ("transmitter", "boxair_emi", "lna_feedback", "laptop"):
        rec = combs.get(name) or {}
        if rec.get("detected") and rec.get("teeth") is not None:
            self_chan |= np.asarray(rec["teeth"], dtype=bool)
    self_chan |= chan_flags & (laptop | fan)

    # --- TX comb --------------------------------------------------------
    # Never set, so the code reproduces flags/v0. It was left unset
    # because no comb was then believed to be the transmitter; memo 001
    # has since identified the `transmitter` comb. Setting tx_chan from
    # combs["transmitter"]["teeth"] when `tx_on` would move those teeth
    # to the non-RFI bit 1 -- a product change for a new flags version.
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

    Airplane reflections and lightning are broadband and short; the
    transmitter comb and the self-RFI combs are narrowband and steady.
    This is the morphological discriminator between them.
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
