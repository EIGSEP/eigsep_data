"""Model-independent RFI flagging from raw correlator data, not beam residuals.

Flags outlier TIMES using only the data itself: off-comb (non-TX-tone)
raw channels and the 0x4 cross-correlation should vary smoothly in
time (tracking the slow, physical variation of system temperature and
antenna pointing), so a departure from a smooth per-file DPSS fit is
evidence of transient interference, not a beam-model assumption. This
sidesteps a real failure mode found in the model-residual-based
approach used earlier: a systematically-wrong (e.g. too-low-order)
beam model produces large, coherent residuals that a residual-based
detector can misclassify as isolated RFI when it's really just model
inadequacy -- confirmed in that case by the fit's reported quality
being almost insensitive to how much genuine model freedom was
allowed, once the outlier loop got to iteratively re-select which
~quarter of the data "fit".

RFI is typically broadband in frequency but short in time (the
opposite of the narrowband, steady TX comb), so a handful of off-comb
monitor channels, checked for simultaneous departures from their own
smooth time trends, is a good broadband-transient detector. The 0x4
cross-correlation (between the ground-facing reference antenna, input
0, and the scanned science antenna, input 4) is an even more sensitive
monitor: the common-mode sky/monopole term that dominates the *auto*-
correlations mostly cancels in this cross product, so genuine
transient interference registers with much higher relative SNR.

Public API
----------
monitor_channels
smooth_time_flags
data_space_rfi_mask
"""

import glob
from pathlib import Path

import numpy as np
from hera_filters import dspec


def monitor_channels(freqs_mhz, beam_freq_min, beam_freq_max,
                     comb_spacing=8, n_monitors=24, offset=3):
    """Off-comb channels spread across the HFSS frequency support.

    Offset by ``offset`` from each comb tone so these channels never
    coincide with a TX tone (which would defeat the point).
    """
    df = float(freqs_mhz[1] - freqs_mhz[0])
    first = int(np.ceil(beam_freq_min / df / comb_spacing) * comb_spacing)
    last = int(np.floor(beam_freq_max / df / comb_spacing) * comb_spacing)
    comb = np.arange(first, last + 1, comb_spacing, dtype=int)
    stride = max(len(comb) // n_monitors, 1)
    return (comb[::stride] + offset).astype(int)


def smooth_time_flags(values, times, filter_half_width_hz=0.01, clip_sigma=5.0):
    """Flag departures from a low-order DPSS fit vs time (real-valued input).

    ``filter_half_width_hz`` is deliberately "low order" (~8 DPSS terms
    over a typical 240-sample, ~130 s file at the default 0.01 Hz):
    wide enough to track genuine slow variation (antenna pointing,
    system temperature) but not fast enough to absorb a short RFI
    transient into the "smooth" model.
    """
    design, _ = dspec.dpss_operator(
        times, [0.0], [filter_half_width_hz], eigenval_cutoff=[1e-9])
    weights = np.ones(times.size)
    solution = dspec.fit_solution_matrix(weights, design)
    model = (design @ (solution @ values)).real
    residual = values - model
    scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    if scale < 1e-12:
        return np.zeros(times.size, dtype=bool)
    return np.abs(residual) > clip_sigma * scale


def data_space_rfi_mask(data_path, beam_freq_min, beam_freq_max,
                        files_slice=(-185, -150), filter_half_width_hz=0.01,
                        clip_sigma=5.0, min_votes=1, n_monitors=24):
    """Shared, model-independent per-time RFI mask (True = clean).

    Matches ``v007_beam_diagnostic.load_v007_data``'s file selection
    and per-file time windowing exactly, so the returned mask aligns
    directly with its ``measured_tx``/``az_deg``/etc arrays -- this is
    computed once from raw data, not per TX channel, and used in place
    of any beam-model-residual-based flagging.
    """
    from eigsep_observing import io

    files = sorted(glob.glob(str(Path(data_path) / "*.h5")))[slice(*files_slice)]
    if not files:
        raise ValueError(f"No HDF5 files found in {data_path}")
    n = len(files) * 240
    clean = np.ones(n, dtype=bool)
    monitors = None
    for i, filename in enumerate(files):
        dat, header, metadata = io.read_hdf5(filename)
        nt = len(header["times"])
        sl = slice(240 * i, 240 * i + nt)
        if monitors is None:
            freqs_mhz = np.asarray(header["freqs"])
            monitors = monitor_channels(
                freqs_mhz, beam_freq_min, beam_freq_max, n_monitors=n_monitors)
        t = np.asarray(header["times"], float)
        t = t - t[0]
        auto = np.asarray(dat["4"], float)
        cross = np.asarray(dat["04"])
        votes = np.zeros(nt, dtype=int)
        for ch in monitors:
            votes += smooth_time_flags(
                auto[:, ch], t, filter_half_width_hz, clip_sigma)
            votes += smooth_time_flags(
                cross[:, ch].real, t, filter_half_width_hz, clip_sigma)
            votes += smooth_time_flags(
                cross[:, ch].imag, t, filter_half_width_hz, clip_sigma)
        clean[sl] = votes < min_votes
    return clean
