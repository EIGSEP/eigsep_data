"""Fit the HFSS TX beam model to the v007 Marjum beam-map data.

Loads a v007 campaign directory, flags bad channels and times, and fits
the transmitter heading/polarization jointly across TX channels, with
plotting helpers for the standard diagnostic figures.

Public API
----------
load_v007_data
model_v007_points
tx_arm_for_channel
radiometer_difference_sigma
channel_validity_masks
isolated_map_outliers
gross_power_time_flags
fit_v007_beam_joint
fit_v007_beam
make_diagnostic
make_joint_diagnostics
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .geometry import TransmitterGeometry
from .tx_model import HFSSBeamSet, ground_heading, simulate_hfss


MOTOR_CAL = 180.0 / 1.13e4
BEAM_CHANNEL = 600 - 4 * 16
TX_CHANNELS = np.arange(1, 1023)


def _records_to_array(records, fields, n):
    out = np.zeros((n, len(fields)), dtype=np.float32)
    for i, record in enumerate(records or []):
        if record is None:
            continue
        for j, field in enumerate(fields):
            if field in record:
                out[i, j] = record[field]
    return out[:, 0] if len(fields) == 1 else out


COMB_BAND = (480, 800)
COMB_MIN_TONES = 20


def comb_present(spectrum, band=COMB_BAND, min_tones=COMB_MIN_TONES,
                 snr=20.0):
    """True if a TX comb is on in this spectrum, whatever its spacing.

    Counts channels whose adjacent-channel second difference stands more than
    ``snr`` MADs above the band's continuum.  Deliberately does not assume a
    comb spacing: the Marjum comb was reconfigured between 2, 4, 8 and 256
    channels during the campaign, so a fixed grid would mislabel other eras.
    Measured separation on the 07-17/18 beam scan is clean -- comb-on files
    score 29-45, comb-off files 0-16.
    """
    lo, hi = band
    spectrum = np.asarray(spectrum, float)
    second = spectrum[1:-1] - 0.5 * (spectrum[:-2] + spectrum[2:])
    band_values = second[lo - 1:hi - 1]
    mad = 1.4826 * np.median(np.abs(band_values - np.median(band_values)))
    return int(np.sum(band_values > snr * max(mad, 1e-30))) >= min_tones


def load_v007_data(data_path, require_comb=True):
    """Load v007 metadata and baseline-subtracted TX comb channels.

    ``require_comb`` drops spectra from files in which the transmitter comb is
    off.  Three of the 35 files in the default slice are comb-off -- the
    transmitter was being toggled around beam-scan start (fieldnotes p181) --
    and they are *not* caught by the existing flagging, because a comb-off
    sample is small rather than a large excursion.  Including them shifts the
    fitted TX heading by 10.6 deg.  Pass False only to reproduce older results.
    """
    import glob
    from eigsep_observing import io

    files = sorted(glob.glob(str(Path(data_path) / "*.h5")))[-185:-150]
    if not files:
        raise ValueError(f"No HDF5 files found in {data_path}")
    comb_off_files = []
    n = len(files) * 240
    accel = np.zeros((n, 3), dtype=np.float32)
    pot = np.zeros(n, dtype=np.float32)
    azm = np.zeros(n, dtype=np.float32)
    elm = np.zeros(n, dtype=np.float32)
    times = np.zeros(n, dtype=np.float64)
    measured_tx = np.zeros((n, 1024), dtype=np.float32)
    measured_sigma = np.zeros((n, 1024), dtype=np.float32)
    freqs = None
    for i, filename in enumerate(files):
        dat, header, metadata = io.read_hdf5(filename)
        nt = len(header["times"])
        sl = slice(240 * i, 240 * i + nt)
        times[sl] = header["times"]
        accel[sl] = _records_to_array(metadata.get("imu_el"),
                                      ["accel_x", "accel_y", "accel_z"], nt)
        pot[sl] = _records_to_array(metadata.get("potmon"), ["pot_az_angle"], nt)
        azm[sl], elm[sl] = _records_to_array(
            metadata.get("motor"), ["az_pos", "el_pos"], nt).T
        auto = np.asarray(dat["4"], float)
        if freqs is None:
            freqs = np.asarray(header["freqs"])
        if require_comb and not comb_present(np.median(auto, axis=0)):
            comb_off_files.append(Path(filename).name)
            times[sl] = 0.0          # the loader's existing invalid sentinel
            continue
        measured_tx[sl, 1:-1] = auto[:, 1:-1] - 0.5 * (
            auto[:, :-2] + auto[:, 2:])
        bandwidth_hz = abs(float(header["freqs"][1] - header["freqs"][0])) * 1e6
        integration_s = float(header.get("integration_time", np.median(
            np.diff(np.asarray(header["times"], float)))))
        measured_sigma[sl, 1:-1] = radiometer_difference_sigma(
            auto[:, 1:-1], auto[:, :-2], auto[:, 2:],
            bandwidth_hz, integration_s)
    if comb_off_files:
        print(f"load_v007_data: dropped {len(comb_off_files)} comb-off file(s) "
              f"of {len(files)}: {', '.join(comb_off_files)}")
    return {
        "files": files, "times": times, "accel": accel, "pot": pot,
        "az_deg": azm * MOTOR_CAL, "el_deg": elm * MOTOR_CAL,
        "measured_tx": measured_tx, "measured_sigma": measured_sigma,
        "freqs": freqs, "comb_off_files": comb_off_files,
    }


def _beam_channel_indices(data, beam):
    df = float(data["freqs"][1] - data["freqs"][0])
    channels = np.rint(beam.freqs_mhz / df).astype(int)
    use = (channels >= TX_CHANNELS[0]) & (channels <= TX_CHANNELS[-1])
    data_cols = channels[use]
    return np.flatnonzero(use), data_cols


def model_v007_points(data, beam_file, height_m=92.5, alpha_deg=60.0,
                      east_m=0.0, north_m=0.0):
    """Evaluate every HFSS slice at the exact v007 motor pointings."""
    beam = beam_file if isinstance(beam_file, HFSSBeamSet) else HFSSBeamSet.from_npz(beam_file)
    geom = TransmitterGeometry(ground_heading(east_m, north_m, height_m), alpha_deg)
    arm = np.arange(beam.beam_cart.shape[0]) % 2
    model, _ = simulate_hfss(beam, data["az_deg"], data["el_deg"], geom, arm)
    fi, _ = _beam_channel_indices(data, beam)
    return model[fi], fi, beam.freqs_mhz[fi]


def _beam_slice_at_frequency(beam, frequency_mhz):
    """Linearly interpolate the complex HFSS beam at one frequency."""
    f = float(frequency_mhz)
    if not beam.freqs_mhz[0] <= f <= beam.freqs_mhz[-1]:
        raise ValueError(f"Frequency {f} MHz is outside the HFSS beam grid")
    hi = int(np.searchsorted(beam.freqs_mhz, f, side="right"))
    hi = min(max(hi, 1), beam.freqs_mhz.size - 1)
    lo = hi - 1
    weight = (f - beam.freqs_mhz[lo]) / (beam.freqs_mhz[hi] - beam.freqs_mhz[lo])
    return HFSSBeamSet(
        (1 - weight) * beam.beam_cart[lo:lo + 1] + weight * beam.beam_cart[hi:hi + 1],
        (1 - weight) * beam.gain_th[lo:lo + 1] + weight * beam.gain_th[hi:hi + 1],
        (1 - weight) * beam.gain_ph[lo:lo + 1] + weight * beam.gain_ph[hi:hi + 1],
        np.array([f]),
    )


def _unit_heading(vector):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("heading vector must be nonzero")
    return vector / norm


def _heading_to_angles(heading):
    """Convert a unit heading vector to (az_deg, el_deg).

    Matches the convention in ``rotation_beam.PolarizationBeamMapper.fit_heading``.
    """
    heading = _unit_heading(heading)
    el_deg = np.rad2deg(np.arcsin(np.clip(heading[2], -1.0, 1.0)))
    az_deg = np.rad2deg(np.arctan2(heading[1], heading[0]))
    return float(az_deg), float(el_deg)


def _angles_to_heading(az_deg, el_deg):
    """Inverse of :func:`_heading_to_angles`."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])


def _coarse_heading_grid(objective, alpha0, az0, el0,
                         az_step=30.0, el_step=20.0, alpha_step=30.0,
                         refine_levels=4):
    """Scan a coarse (az, el, alpha) grid and return the best point.

    Guards the local Powell refinement against converging to different
    basins on different runs (the source of the heading instability
    documented for the unconstrained Cartesian parameterization) by
    always starting Powell from a global, not just local, candidate.
    Alpha is included in the grid, not held fixed at the caller's
    starting value: the polarization coupling term entangles heading
    and alpha, so a coarse (az, el) scan at a poor starting alpha can
    land in a bad basin (verified: a deliberately bad start converged
    to a much higher-residual optimum without this). Alpha is scanned
    over one dipole period (180 degrees) since the coupling only enters
    as ``(e . basis)**2``, invariant under a sign flip of the dipole
    axis. The caller's own (az0, el0, alpha0) is included as a
    candidate so the grid can never do worse than the previous
    behavior.
    """
    candidates = [(az0, el0, alpha0)]
    for az in np.arange(-180.0, 180.0, az_step):
        for el in np.arange(-85.0, 86.0, el_step):
            for alpha in np.arange(0.0, 180.0, alpha_step):
                candidates.append((az, el, alpha))
    best_x, best_val = (az0, el0, alpha0), np.inf
    for az, el, alpha in candidates:
        val = objective(np.array([az, el, alpha]))
        if val < best_val:
            best_val, best_x = val, (az, el, alpha)
    # Iteratively refine around the current winner at half spacing each
    # round (a simple pattern search) so a narrow basin sitting between
    # coarse grid points -- or just outside the caller's own starting
    # point -- isn't missed (verified: without this, a start already
    # close to a slightly better nearby minimum could still beat the
    # grid-selected basin after Powell refinement, since Powell alone
    # only descends locally).
    step_az, step_el, step_alpha = az_step, el_step, alpha_step
    for _ in range(refine_levels):
        step_az, step_el, step_alpha = step_az / 2, step_el / 2, step_alpha / 2
        az_best, el_best, alpha_best = best_x
        fine_candidates = [
            (az_best + daz, el_best + del_, alpha_best + dalpha)
            for daz in (-step_az, 0.0, step_az)
            for del_ in (-step_el, 0.0, step_el)
            for dalpha in (-step_alpha, 0.0, step_alpha)
        ]
        for az, el, alpha in fine_candidates:
            val = objective(np.array([az, el, alpha]))
            if val < best_val:
                best_val, best_x = val, (az, el, alpha)
    return best_x


def tx_arm_for_channel(channel, spike_spacing=8):
    """Return the TX arm for tones spaced by eight raw FFT channels."""
    return (int(channel) // int(spike_spacing)) % 2


def radiometer_difference_sigma(tx_power, left_power, right_power,
                                bandwidth_hz, integration_s):
    """Radiometer uncertainty of TX minus the mean of two side channels."""
    variance = (np.asarray(tx_power, float) ** 2
                + 0.25 * np.asarray(left_power, float) ** 2
                + 0.25 * np.asarray(right_power, float) ** 2)
    return np.sqrt(variance / (float(bandwidth_hz) * float(integration_s)))


def channel_validity_masks(data, channels, threshold=None, shared_time_flags=None):
    """Return channel-local validity after applying shared residual time flags."""
    channels = np.asarray(channels, dtype=int)
    y = data["measured_tx"][:, channels]
    sigma = data.get("measured_sigma")
    local = (data["times"][:, None] > 0) & np.isfinite(y)
    if sigma is not None:
        sigma = sigma[:, channels]
        local &= np.isfinite(sigma) & (sigma > 0)
    if threshold is not None:
        local &= y > threshold
    if shared_time_flags is None:
        return local
    return local & ~np.asarray(shared_time_flags, dtype=bool)[:, None]


def isolated_map_outliers(residual, valid, az_deg, el_deg,
                          clip_sigma=5.0, neighbors=12):
    """Flag spatially isolated residuals while preserving coherent beam nulls."""
    residual = np.asarray(residual, float)
    valid = np.asarray(valid, bool)
    out = np.zeros(valid.shape, dtype=bool)
    good = np.flatnonzero(valid & np.isfinite(residual))
    if good.size < max(neighbors + 1, 20):
        return out
    az = np.deg2rad(np.asarray(az_deg, float)[good])
    el = np.deg2rad(np.asarray(el_deg, float)[good])
    coordinates = np.column_stack([
        np.cos(az), np.sin(az), np.cos(el), np.sin(el)])
    tree = cKDTree(coordinates)
    _, near = tree.query(coordinates, k=min(neighbors + 1, good.size))
    local = np.median(residual[good][near[:, 1:]], axis=1)
    highpass = residual[good] - local
    center = float(np.median(highpass))
    mad = 1.4826 * float(np.median(np.abs(highpass - center)))
    if mad > 1e-12:
        out[good] = np.abs(highpass - center) > clip_sigma * mad
    else:
        out[good] = np.abs(highpass - center) > 1e-12
    return out


def gross_power_time_flags(data, channels, reference_percentile=99.0,
                           factor=5.0, negative_factor=2.0):
    """Find impossible power excursions without rejecting beam nulls.

    Positive excursions use a deliberately loose cutoff because real TX beam
    peaks are positive. Negative excursions are compared to the ordinary
    positive beam scale separately: adjacent-channel subtraction can scatter
    around zero at a null, but cannot produce a large negative TX signal.
    """
    channels = np.asarray(channels, dtype=int)
    y = data["measured_tx"][:, channels].astype(float)
    valid = channel_validity_masks(data, channels, threshold=None)
    by_channel = np.zeros_like(valid)
    references = np.empty(channels.size)
    for ci in range(channels.size):
        positive = valid[:, ci] & (y[:, ci] > 0)
        if not np.any(positive):
            references[ci] = np.percentile(
                np.abs(y[valid[:, ci], ci]), reference_percentile)
        else:
            references[ci] = np.percentile(
                y[positive, ci], reference_percentile)
        scale = max(references[ci], 1e-30)
        by_channel[:, ci] = valid[:, ci] & (
            (y[:, ci] > factor * scale)
            | (y[:, ci] < -negative_factor * scale))
    return np.any(by_channel, axis=1), by_channel, references


def _fit_v007_beam_joint_once(data, beam, threshold, height_m, initial,
                               beam_channels, sample_mask=None,
                               normalization_rms=None):
    """Fit shared pointing/polarization parameters with one gain per channel."""
    channels = np.asarray(beam_channels, dtype=int)
    target_frequencies = np.asarray(data["freqs"])[channels].astype(float)
    slices = [_beam_slice_at_frequency(beam, f) for f in target_frequencies]
    y_full = data["measured_tx"][:, channels].astype(float)
    shared_flags = None if sample_mask is None else ~np.asarray(sample_mask, dtype=bool)
    valid = channel_validity_masks(data, channels, threshold, shared_flags)
    counts = valid.sum(axis=0)
    if np.any(counts < 20):
        raise ValueError("Too few valid samples in at least one TX channel")
    if normalization_rms is None:
        base = channel_validity_masks(data, channels, threshold)
        normalization_rms = np.array([
            np.sqrt(np.mean(y_full[base[:, ci], ci] ** 2))
            for ci in range(channels.size)
        ])
    normalization_rms = np.asarray(normalization_rms, float)
    arms = np.array([tx_arm_for_channel(channel) for channel in channels])

    def stack_beams(indices):
        selected = [slices[i] for i in indices]
        return HFSSBeamSet(
            np.concatenate([item.beam_cart for item in selected], axis=0),
            np.concatenate([item.gain_th for item in selected], axis=0),
            np.concatenate([item.gain_ph for item in selected], axis=0),
            np.concatenate([item.freqs_mhz for item in selected]),
        )

    arm_groups = {
        arm: (np.flatnonzero(arms == arm),
              stack_beams(np.flatnonzero(arms == arm)))
        for arm in np.unique(arms)
    }

    def evaluate(x, full=False):
        heading = _angles_to_heading(x[0], x[1])
        geom = TransmitterGeometry(heading, x[2])
        models = np.empty((channels.size, data["az_deg"].size), float)
        for arm, (indices, grouped_beam) in arm_groups.items():
            group_model, _ = simulate_hfss(
                grouped_beam, data["az_deg"], data["el_deg"], geom,
                np.full(data["az_deg"].size, arm, dtype=int))
            models[indices] = group_model
        if full:
            return models
        gains = np.empty(channels.size)
        residuals = []
        for ci in range(channels.size):
            use = valid[:, ci]
            one_model = models[ci, use]
            one_data = y_full[use, ci]
            gains[ci] = max(0.0, np.sum(one_data * one_model) /
                             max(np.sum(one_model ** 2), 1e-30))
            residuals.append(
                (one_data - gains[ci] * one_model) / normalization_rms[ci])
        return residuals, gains

    def objective(x):
        residuals, _ = evaluate(x)
        return float(np.mean([np.mean(residual ** 2) for residual in residuals]))

    initial = np.asarray(initial, float)
    az0, el0 = _heading_to_angles(initial[:3])
    alpha0 = float(initial[3])
    az0, el0, alpha0 = _coarse_heading_grid(objective, alpha0, az0, el0)
    x0 = np.array([az0, el0, alpha0])
    result = minimize(objective, x0, method="Powell",
                      bounds=[(-180, 180), (-90, 90), (-180, 180)],
                      options={"maxiter": 30, "xtol": 1e-3, "ftol": 1e-4})
    result.heading = _angles_to_heading(result.x[0], result.x[1])
    result.alpha_deg = float(result.x[2])
    result.model = evaluate(result.x, full=True)
    _, result.gains = evaluate(result.x)
    result.initial_rms = normalization_rms
    result.residual_rms = float(np.sqrt(result.fun))
    result.n_channels = int(channels.size)
    result.n_samples_by_channel = counts.astype(int)
    result.n_samples = int(counts.sum())
    result.fi = np.arange(channels.size)
    result.data_cols = channels
    result.frequency_mhz = target_frequencies
    return result


def fit_v007_beam_joint(data, beam_file, threshold=None, height_m=92.5,
                        initial=(0.0, 0.0, -1.0, 60.0),
                        beam_channels=(536, 544, 552), clip_sigma=5.0,
                        max_clip_iterations=0,
                        gross_outlier_factor=5.0,
                        gross_reference_percentile=99.0):
    """Jointly fit channels with local validity and shared residual time flags."""
    beam = beam_file if isinstance(beam_file, HFSSBeamSet) else HFSSBeamSet.from_npz(beam_file)
    channels = np.asarray(beam_channels, dtype=int)
    y_full = data["measured_tx"][:, channels].astype(float)
    sigma_full = data["measured_sigma"][:, channels].astype(float)
    base = channel_validity_masks(data, channels, threshold)
    base_counts = base.sum(axis=0)
    if np.any(base_counts < 20):
        raise ValueError("Too few valid samples in at least one TX channel")
    normalization_rms = np.array([
        np.sqrt(np.mean(y_full[base[:, ci], ci] ** 2))
        for ci in range(channels.size)
    ])
    if gross_outlier_factor is None:
        flagged = np.zeros(data["times"].shape, dtype=bool)
        flagged_by_channel = np.zeros_like(base, dtype=bool)
        gross_references = np.full(channels.size, np.nan)
    else:
        flagged, flagged_by_channel, gross_references = gross_power_time_flags(
            data, channels, gross_reference_percentile, gross_outlier_factor)
    current = np.asarray(initial, float).copy()
    fit = None
    for iteration in range(max_clip_iterations + 1):
        fit = _fit_v007_beam_joint_once(
            data, beam, threshold, height_m, current, channels,
            sample_mask=~flagged, normalization_rms=normalization_rms)
        current = np.r_[fit.heading, fit.alpha_deg]
        modeled = fit.model.T * fit.gains
        standardized = np.full_like(y_full, np.nan, dtype=float)
        np.divide(y_full - modeled, sigma_full, out=standardized,
                  where=sigma_full > 0)
        used = base & ~flagged[:, None]
        if iteration == max_clip_iterations:
            break
        outlier_by_channel = np.column_stack([
            isolated_map_outliers(
                standardized[:, ci], used[:, ci],
                data["az_deg"], data["el_deg"], clip_sigma=clip_sigma)
            for ci in range(channels.size)
        ])
        flagged_by_channel |= outlier_by_channel
        new_flagged = flagged | np.any(outlier_by_channel, axis=1)
        new_used = base & ~new_flagged[:, None]
        if (not np.any(new_flagged & ~flagged)
                or np.any(new_used.sum(axis=0) < 20)):
            break
        flagged = new_flagged
    used = base & ~flagged[:, None]
    fit.used_mask_by_channel = used
    fit.base_mask_by_channel = base
    fit.used_mask = used[:, 0] if channels.size == 1 else np.any(used, axis=1)
    fit.base_mask = base[:, 0] if channels.size == 1 else np.any(base, axis=1)
    fit.flagged_mask = flagged
    fit.flagged_by_channel = flagged_by_channel
    fit.n_flagged_by_channel = flagged_by_channel.sum(axis=0).astype(int)
    fit.gross_power_references = gross_references
    fit.n_base_samples_by_channel = base_counts.astype(int)
    fit.n_base_samples = int(base_counts.sum())
    fit.n_flagged = int(flagged.sum())
    fit.clip_sigma = float(clip_sigma)
    fit.initial_rms = normalization_rms
    fit.n_samples_by_channel = used.sum(axis=0).astype(int)
    fit.n_samples = int(fit.n_samples_by_channel.sum())
    modeled = fit.model.T * fit.gains
    standardized = np.full_like(y_full, np.nan, dtype=float)
    np.divide(y_full - modeled, sigma_full, out=standardized,
              where=sigma_full > 0)
    fit.channel_residual_rms = np.array([
        np.sqrt(np.mean(((y_full[used[:, ci], ci]
                         - modeled[used[:, ci], ci])
                        / normalization_rms[ci]) ** 2))
        for ci in range(channels.size)
    ])
    fit.channel_chisq = np.array([
        np.sum(standardized[used[:, ci], ci] ** 2)
        for ci in range(channels.size)
    ])
    fit.channel_reduced_chisq = fit.channel_chisq / np.maximum(
        fit.n_samples_by_channel - 1, 1)
    n_parameters = channels.size + 3  # one gain/channel + 2-D heading + alpha
    fit.dof = max(int(fit.n_samples_by_channel.sum()) - n_parameters, 1)
    fit.reduced_chisq = float(np.sum(fit.channel_chisq) / fit.dof)
    fit.residual_rms = float(np.sqrt(np.mean(fit.channel_residual_rms ** 2)))
    return fit


def fit_v007_beam(data, beam_file, threshold=None, height_m=92.5,
                  initial=(0.0, 0.0, -1.0, 60.0), beam_channel=BEAM_CHANNEL,
                  clip_sigma=5.0, max_clip_iterations=0):
    """Backward-compatible single-channel wrapper around the joint fitter."""
    return fit_v007_beam_joint(
        data, beam_file, threshold, height_m, initial, (beam_channel,),
        clip_sigma, max_clip_iterations)


def make_diagnostic(data_path, beam_file, output, threshold=None,
                    height_m=92.5, alpha_deg=60.0, beam_channel=BEAM_CHANNEL,
                    clip_sigma=5.0):
    data = load_v007_data(data_path)
    beam = HFSSBeamSet.from_npz(beam_file)
    fit = fit_v007_beam(data, beam, threshold, height_m, (0, 0, -1, alpha_deg),
                        beam_channel=beam_channel, clip_sigma=clip_sigma)
    display = 0
    channel = int(fit.data_cols[display])
    valid = fit.used_mask
    y = data["measured_tx"][:, channel][valid]
    modeled = fit.model[display] * fit.gains[display]
    m = modeled[valid]
    residual = (y - m) / float(np.asarray(fit.initial_rms)[display])
    flagged = fit.flagged_mask
    positive = y[y > 0]
    floor = np.percentile(positive, 1) if positive.size else 1.0
    lo, hi = np.percentile(np.log10(np.maximum(y, floor)), [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, values, title in ((axes[0], y, "v007 measured"),
                              (axes[1], m, "HFSS model, fitted gain")):
        ax.scatter(data["el_deg"][valid], data["az_deg"][valid],
                   c=np.log10(np.maximum(values, floor)),
                   s=20, vmin=lo, vmax=hi, cmap="plasma")
        if np.any(flagged):
            ax.scatter(data["el_deg"][flagged], data["az_deg"][flagged],
                       color="black", marker="x", s=26,
                       linewidths=0.7)
        ax.set_title(title); ax.set_xlabel("Elevation [deg]"); ax.set_ylabel("Azimuth [deg]")
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 0)
    axes[2].scatter(data["el_deg"][valid], data["az_deg"][valid], c=residual,
                    s=20, cmap="bwr", vmin=-1, vmax=1)
    if np.any(flagged):
        axes[2].scatter(data["el_deg"][flagged], data["az_deg"][flagged],
                        color="black", marker="x", s=26,
                        linewidths=0.7)
    axes[2].set_title(
        f"normalized residual; RMS={fit.residual_rms:.4f}; "
        f"reduced chi2={fit.reduced_chisq:.2g}")
    axes[2].set_xlabel("Elevation [deg]"); axes[2].set_ylabel("Azimuth [deg]")
    axes[2].set_xlim(-180, 180); axes[2].set_ylim(-180, 0)
    fig.suptitle(f"v007 input-4 @ {fit.frequency_mhz[display]:.3f} MHz; "
                 f"heading={np.array2string(fit.heading, precision=3)}, alpha={fit.alpha_deg:.1f} deg; "
                 f"single-channel fit; flagged={fit.n_flagged} times")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {"files": len(data["files"]), "n_channels": fit.n_channels,
            "n_samples": fit.n_samples, "n_base_samples": fit.n_base_samples,
            "n_flagged": fit.n_flagged, "frequency_mhz": float(fit.frequency_mhz),
            "heading": fit.heading.tolist(),
            "alpha_deg": float(fit.alpha_deg),
            "initial_rms": float(np.asarray(fit.initial_rms)[display]),
            "residual_rms": float(np.asarray(fit.channel_residual_rms)[display]),
            "reduced_chisq": float(fit.reduced_chisq)}


def make_joint_diagnostics(data_path, beam_file, output_prefix,
                           beam_channels=(536, 544, 552), threshold=None,
                           height_m=92.5, alpha_deg=60.0, clip_sigma=5.0,
                           max_clip_iterations=0, initial=None):
    """Fit two or more channels jointly and write one diagnostic per channel."""
    data = load_v007_data(data_path)
    beam = HFSSBeamSet.from_npz(beam_file)
    if initial is None:
        initial = (0, 0, -1, alpha_deg)
    fit = fit_v007_beam_joint(
        data, beam, threshold, height_m, initial,
        beam_channels=beam_channels, clip_sigma=clip_sigma,
        max_clip_iterations=max_clip_iterations)
    outputs = []
    for display, channel in enumerate(fit.data_cols):
        channel = int(channel)
        valid = fit.used_mask_by_channel[:, display]
        y = data["measured_tx"][:, channel][valid]
        modeled = fit.model[display] * fit.gains[display]
        m = modeled[valid]
        rms0 = float(fit.initial_rms[display])
        residual = (y - m) / rms0
        positive = y[y > 0]
        floor = np.percentile(positive, 1) if positive.size else 1.0
        lo, hi = np.percentile(np.log10(np.maximum(y, floor)), [1, 99])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, values, title in ((axes[0], y, "v007 measured"),
                                  (axes[1], m, "HFSS model, fitted gain")):
            ax.scatter(data["el_deg"][valid], data["az_deg"][valid],
                       c=np.log10(np.maximum(values, floor)), s=20,
                       vmin=lo, vmax=hi, cmap="plasma")
            ax.scatter(data["el_deg"][fit.flagged_mask],
                       data["az_deg"][fit.flagged_mask], color="black",
                       marker="x", s=26, linewidths=0.7)
            ax.set_title(title); ax.set_xlabel("Elevation [deg]"); ax.set_ylabel("Azimuth [deg]")
            ax.set_xlim(-180, 180); ax.set_ylim(-180, 0)
        axes[2].scatter(data["el_deg"][valid], data["az_deg"][valid],
                        c=residual, s=20, cmap="bwr", vmin=-1, vmax=1)
        axes[2].scatter(data["el_deg"][fit.flagged_mask],
                        data["az_deg"][fit.flagged_mask], color="black",
                        marker="x", s=26, linewidths=0.7)
        axes[2].set_title(
            f"normalized residual; RMS={fit.channel_residual_rms[display]:.4f}; "
            f"reduced chi2={fit.channel_reduced_chisq[display]:.2g}")
        axes[2].set_xlabel("Elevation [deg]"); axes[2].set_ylabel("Azimuth [deg]")
        axes[2].set_xlim(-180, 180); axes[2].set_ylim(-180, 0)
        fig.suptitle(f"v007 input-4 @ {fit.frequency_mhz[display]:.3f} MHz; "
                     f"heading={np.array2string(fit.heading, precision=3)}, "
                     f"alpha={fit.alpha_deg:.1f} deg; joint fit; "
                     f"common flagged={fit.n_flagged} times")
        output = f"{output_prefix}_ch{channel}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(output)
    return {"files": len(data["files"]), "channels": fit.data_cols.tolist(),
            "frequencies_mhz": fit.frequency_mhz.tolist(),
            "n_base_samples": fit.n_base_samples_by_channel.tolist(),
            "n_samples": fit.n_samples_by_channel.tolist(),
            "n_flagged": fit.n_flagged,
            "n_flagged_by_channel": fit.n_flagged_by_channel.tolist(),
            "heading": fit.heading.tolist(),
            "alpha_deg": float(fit.alpha_deg), "gains": fit.gains.tolist(),
            "initial_rms": fit.initial_rms.tolist(),
            "residual_rms": fit.channel_residual_rms.tolist(),
            "reduced_chisq": float(fit.reduced_chisq),
            "channel_reduced_chisq": fit.channel_reduced_chisq.tolist(),
            "outputs": outputs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("beam_file")
    ap.add_argument("-o", "--output", default="v007_beam_model_vs_measured")
    ap.add_argument("--threshold", type=float, default=None,
                    help="optional channel-local TX excess cut; default keeps nulls")
    ap.add_argument("--height-m", type=float, default=92.5)
    ap.add_argument("--alpha-deg", type=float, default=60.0)
    ap.add_argument("--clip-sigma", type=float, default=5.0)
    ap.add_argument("--clip-iterations", type=int, default=0,
                    help="optional experimental residual clipping; default disabled")
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--channels", type=int, nargs="+", default=[536, 544, 552])
    args = ap.parse_args()
    channels = [args.channel] if args.channel is not None else args.channels
    if len(channels) == 1:
        output = args.output if args.output.endswith(".png") else args.output + ".png"
        print(make_diagnostic(args.data_path, args.beam_file, output,
                              args.threshold, args.height_m, args.alpha_deg,
                              channels[0], args.clip_sigma))
    else:
        print(make_joint_diagnostics(args.data_path, args.beam_file, args.output,
                                     channels, args.threshold, args.height_m,
                                     args.alpha_deg, args.clip_sigma,
                                     args.clip_iterations))
