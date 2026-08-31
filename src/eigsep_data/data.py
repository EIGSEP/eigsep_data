from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import curve_fit

from eigsep_observing import io


def to_unix_time(value):
    """
    Convert a datetime string, datetime object, or Unix timestamp to Unix
    seconds (float).

    Strings without a timezone are interpreted as UTC.

    Accepted formats:
        "2026-07-17 06:00:00"
        "2026-7-17 6:00:00"
        "2026-07-17T06:00:00Z"
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"Could not interpret time {value!r}. "
                    "Use a format such as '2026-07-17 06:00:00'."
                )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_time_from_name(fname: str) -> datetime:
    """
    Parse datetime from filename of form 'corr_YYYYMMDD_HHMMSS.h5'
    """
    stem = Path(fname).stem  # 'corr_20250922_160500'
    _, datestr, timestr = stem.split("_")  # ['corr', '20250922', '160500']
    return datetime.strptime(datestr + timestr, "%Y%m%d%H%M%S")


@dataclass
class EigsepData:

    data: dict[str, np.ndarray] = None
    acc_cnt: np.ndarray = None
    times: np.ndarray = None
    freq: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 250, num=1024, endpoint=False)
    )

    @classmethod
    def from_path(
        cls,
        path: Path,
        start_time: str = None,
        end_time: str = None,
        pacific_to_mountain: bool = True,
    ):
        """
        Create an EigsepData instance from a directory or a file.

        Parameters
        ----------
        path : Path
            The path to the directory or file.
        start_time : str
            The start time in the format "YYYYMMDD_HHMMSS" for filtering data.
            Only used if reading from a directory.
        end_time : str
            The end time in the format "YYYYMMDD_HHMMSS" for filtering data.
            Only used if reading from a directory.
        pacific_to_mountain : bool
            If True, convert times from Pacific to Mountain time by adding
            3600 seconds.

        Returns
        -------
        EigsepData

        """
        if path.is_dir():
            files = sorted(path.glob("corr*.h5"))
            times = [_parse_time_from_name(f.name) for f in files]
            if start_time:
                start_dt = datetime.strptime(start_time, "%Y%m%d_%H%M%S")
                files = [f for f, t in zip(files, times) if t >= start_dt]
                times = [_parse_time_from_name(f.name) for f in files]
            if end_time:
                end_dt = datetime.strptime(end_time, "%Y%m%d_%H%M%S")
                files = [f for f, t in zip(files, times) if t <= end_dt]
        elif path.is_file():
            files = [path]
        if not files:
            raise ValueError(f"No data files found in {path}.")

        data = {}
        acc_cnt = []
        times = []
        freq = None
        for f in files:
            try:
                d, hdr, metadata = io.read_hdf5(f)
            except Exception as e:
                warnings.warn(f"Failed to read {f}: {e}. Skipping this file.")
                continue
            for k, v in d.items():
                data[k] = data.get(k, []) + [v]
            acc_cnt.append(hdr["acc_cnt"])
            times.append(hdr["times"])
            if freq is None:
                freq = hdr["freqs"]
            elif not np.array_equal(freq, hdr["freqs"]):
                warnings.warn(
                    f"Frequency mismatch in {f}. Using first file's "
                    "frequency array. "
                )
        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0)
        acc_cnt = np.concatenate(acc_cnt, axis=0)
        times = np.concatenate(times, axis=0)
        if pacific_to_mountain:
            times += 3600
        return cls(data=data, acc_cnt=acc_cnt, times=times, freq=freq)

    def slice(self, min_index, max_index):
        """
        Slice the data along the time axis.

        Parameters
        ----------
        min_index : int
            The minimum index (inclusive).
        max_index : int
            The maximum index (exclusive).

        Returns
        -------
        EigsepData
            A new EigsepData instance with the sliced data.

        """
        sliced_data = {k: v[min_index:max_index] for k, v in self.data.items()}
        sliced_acc_cnt = self.acc_cnt[min_index:max_index]
        sliced_times = self.times[min_index:max_index]
        return EigsepData(
            data=sliced_data,
            acc_cnt=sliced_acc_cnt,
            times=sliced_times,
            freq=self.freq,
        )


def _select_h5_in_range(data_dir, start_unix, end_unix, file_patterns):
    """
    Read HDF5 files in *data_dir* matching *file_patterns* and keep only
    the integrations with header["times"] in [start_unix, end_unix).

    Returns
    -------
    times : np.ndarray, shape (nsamples,)
        Chronologically sorted Unix times.
    freqs : np.ndarray or None
        Frequency axis from the first file's header.
    data_range : dict[str, np.ndarray]
        Per-key data arrays, sorted to match *times*.
    headers : list[dict]
        Per-file dicts with "selected_indices" (positions in the
        original file arrays) and "times" (filtered), in file order.
    metadata : list[dict]
        Per-file raw metadata dicts (as returned by io.read_hdf5), in
        file order, aligned with *headers*.
    sort_index : np.ndarray
        Index used to sort per-file-concatenated arrays chronologically;
        reused to align per-file metadata (motor/potmon/imu) extracted
        separately via *headers*.
    """
    data_dir = Path(data_dir)
    h5_files = []
    for pattern in file_patterns:
        h5_files.extend(data_dir.glob(pattern))
    h5_files = sorted(set(h5_files))
    if not h5_files:
        raise FileNotFoundError(
            f"No files matching {file_patterns} found in:\n{data_dir}"
        )

    selected_data = {}
    selected_times = []
    headers = []
    metadata = []
    freqs = None

    for filename in h5_files:
        try:
            data_file, header_file, metadata_file = io.read_hdf5(filename)
            if "times" not in header_file:
                raise KeyError(f"{filename.name} does not contain header['times'].")
        except (OSError, KeyError) as e:
            warnings.warn(f"Skipping {filename.name}: {e}")
            continue

        times_file = np.asarray(header_file["times"])
        time_mask = (times_file >= start_unix) & (times_file < end_unix)
        if not np.any(time_mask):
            continue

        if selected_times and set(data_file) != set(selected_data):
            raise KeyError(
                f"{filename.name} has data keys {sorted(data_file)}, which "
                f"differ from the keys seen so far {sorted(selected_data)}. "
                "The requested time range straddles a change in recorded "
                "data keys (e.g. a deployment reconfiguration); narrow the "
                "range to a window with consistent keys."
            )

        selected_times.append(times_file[time_mask])
        headers.append({
            "selected_indices": np.flatnonzero(time_mask),
            "times": times_file[time_mask],
        })
        metadata.append(metadata_file)
        if freqs is None:
            freqs = header_file.get("freqs")

        for key, values in data_file.items():
            values = np.asarray(values)
            if values.shape[0] != times_file.size:
                raise ValueError(
                    f"{filename.name}, key {key!r}: shape mismatch "
                    f"({values.shape[0]} vs {times_file.size})"
                )
            selected_data.setdefault(key, []).append(values[time_mask])

    if not selected_times:
        raise ValueError("No integrations found inside the requested time range.")

    times = np.concatenate(selected_times)
    sort_index = np.argsort(times)
    times = times[sort_index]
    data_range = {
        k: np.concatenate(v, axis=0)[sort_index] for k, v in selected_data.items()
    }

    return times, freqs, data_range, headers, metadata, sort_index


def extract_beam_mapping_data(
    data_dir,
    start_time,
    end_time,
    file_patterns=("*.h5", "*.hdf5"),
    sky_key="4",
    ground_key="0",
    cross_key="04",
    sweep_slice=None,
):
    """
    Extract raw beam-mapping arrays from correlator HDF5 files.

    Reads every file matching *file_patterns* in *data_dir*, keeps
    integrations within [*start_time*, *end_time*), and pulls out the
    power spectra plus the motor/potentiometer/IMU metadata needed by
    the beam-mapping pipeline (beam_sim, beam_fit, rfi). The IMU
    elevation angle is derived from the accelerometer axes via SVD, the
    same way as the reference notebook.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing correlator HDF5 files.
    start_time, end_time : str, datetime, or Unix timestamp
        Time range to select; passed through to_unix_time.
    file_patterns : tuple of str
        Glob patterns used to find HDF5 files in *data_dir*.
    sky_key, ground_key, cross_key : str
        Data-dict keys for the sky, ground, and cross-correlation power
        spectra.
    sweep_slice : slice, optional
        If given, applied to every returned per-sample array (e.g. to
        drop a calibration sweep at the start of a run).

    Returns
    -------
    dict with keys:
        times : np.ndarray, shape (nsamples,) -- Unix times
        freqs : np.ndarray -- frequency axis from the first file's header
        sky, ground, cross : np.ndarray, shape (nsamples, nchan)
        el_pos, az_pos : np.ndarray -- commanded motor positions
        pot_az_angle : np.ndarray -- raw potentiometer azimuth reading
        imu_el_deg : np.ndarray -- IMU-derived elevation angle
        imu_accel : np.ndarray, shape (nsamples, 3) -- raw accelerometer
            (x, y, z)
    """
    start_unix = to_unix_time(start_time)
    end_unix = to_unix_time(end_time)
    if end_unix <= start_unix:
        raise ValueError("end_time must be later than start_time.")

    times, freqs, data_range, headers, metadata, sort_index = _select_h5_in_range(
        data_dir, start_unix, end_unix, file_patterns
    )

    for key in (sky_key, ground_key, cross_key):
        if key not in data_range:
            raise KeyError(
                f"Data key {key!r} not found; available keys: "
                f"{sorted(data_range)}"
            )

    el_list, az_list, pot_list, accel_list = [], [], [], []
    for header_entry, meta in zip(headers, metadata):
        indices = header_entry["selected_indices"]

        motor = meta["motor"]
        file_el, file_az = [], []
        for idx in indices:
            entry = motor[idx]
            if entry is None:
                file_el.append(np.nan)
                file_az.append(np.nan)
            else:
                file_el.append(entry.get("el_pos", np.nan))
                file_az.append(entry.get("az_pos", np.nan))
        el_list.append(np.asarray(file_el, dtype=float))
        az_list.append(np.asarray(file_az, dtype=float))

        potmon = meta["potmon"]
        file_pot = []
        for idx in indices:
            entry = potmon[idx]
            file_pot.append(np.nan if entry is None else entry.get("pot_az_angle", np.nan))
        pot_list.append(np.asarray(file_pot, dtype=float))

        imu_el = meta["imu_el"]
        file_accel = []
        for idx in indices:
            entry = imu_el[idx]
            if entry is None:
                file_accel.append((np.nan, np.nan, np.nan))
            else:
                file_accel.append(
                    (entry.get("accel_x", np.nan), entry.get("accel_y", np.nan), entry.get("accel_z", np.nan))
                )
        accel_list.append(np.asarray(file_accel, dtype=float))

    el_pos = np.concatenate(el_list)[sort_index]
    az_pos = np.concatenate(az_list)[sort_index]
    pot_az_angle = np.concatenate(pot_list)[sort_index]
    accel = np.concatenate(accel_list)[sort_index]

    # IMU elevation angle: SVD of the accelerometer axes finds the plane
    # of rotation; the angle within that plane tracks elevation. NaN rows
    # (dropped IMU readings) are excluded from the SVD and left NaN in the
    # output, since np.linalg.svd raises on NaN input.
    valid_accel = ~np.any(np.isnan(accel), axis=1)
    imu_el_deg = np.full(accel.shape[0], np.nan)
    if np.any(valid_accel):
        _, _, Vt = np.linalg.svd(accel[valid_accel], full_matrices=False)
        u, v = Vt[0, :], Vt[1, :]
        proj_x = accel[valid_accel] @ u
        proj_y = accel[valid_accel] @ v
        imu_el_deg[valid_accel] = (
            np.unwrap(np.degrees(np.arctan2(proj_y, proj_x)), period=360) - 180
        )

    out = {
        "times": times,
        "freqs": freqs,
        "sky": data_range[sky_key],
        "ground": data_range[ground_key],
        "cross": data_range[cross_key],
        "el_pos": el_pos,
        "az_pos": az_pos,
        "pot_az_angle": pot_az_angle,
        "imu_el_deg": imu_el_deg,
        "imu_accel": accel,
    }
    if sweep_slice is not None:
        out = {k: (v if k == "freqs" else v[sweep_slice]) for k, v in out.items()}
    return out


def extract_clean_pot_data_v2(az_pot, az_step, min_stable_samples=10, settle_samples=3):
    """
    Clean noisy potentiometer data by isolating stable plateaus, computing
    their medians, and linearly interpolating across motor transitions.

    Parameters
    ----------
    az_pot : np.ndarray
        Raw potentiometer azimuth readings, shape (nsamples,).
    az_step : np.ndarray
        Stepper-motor step count at each sample, shape (nsamples,).
    min_stable_samples : int
        Minimum consecutive samples with the same step value to count as a
        stable plateau.
    settle_samples : int
        Samples to discard at the start of each plateau to allow the motor
        to physically settle before taking the median.

    Returns
    -------
    clean_az_pot : np.ndarray
        Cleaned azimuth array with transitions replaced by linear ramps and
        plateaus replaced by their median values.
    """
    clean_az_pot = np.copy(az_pot)

    change_indices = np.where(np.diff(az_step) != 0)[0] + 1
    boundaries = np.concatenate(([0], change_indices, [len(az_step)]))

    plateaus = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if (end - start) >= min_stable_samples:
            safe_start = min(start + settle_samples, end - 1)
            plateau_median = np.median(az_pot[safe_start:end])
            clean_az_pot[start:end] = plateau_median
            plateaus.append((start, end, plateau_median))

    for i in range(len(plateaus) - 1):
        _, curr_end, curr_val = plateaus[i]
        next_start, _, next_val = plateaus[i + 1]
        if next_start > curr_end:
            ramp = np.linspace(curr_val, next_val, next_start - curr_end + 2)
            clean_az_pot[curr_end:next_start] = ramp[1:-1]

    if len(plateaus) > 0:
        if plateaus[0][0] > 0:
            clean_az_pot[: plateaus[0][0]] = plateaus[0][2]
        if plateaus[-1][1] < len(clean_az_pot):
            clean_az_pot[plateaus[-1][1] :] = plateaus[-1][2]

    return clean_az_pot


def calibrate_weak_arm(el_deg, az_deg, dpss_red):
    """
    Compute per-frequency scale factors to normalize the weak transmitter arm
    to the strong arm.

    This function is a dataset-specific workaround for the deployment-4 data
    where one polarization arm has lower coupling. It fits a Malus's-law
    (cos²) curve to the power measured at el=0 crossings to find the expected
    peak power for each arm, then returns the ratio strong/weak so the caller
    can rescale the weak-arm channels.

    Ideally this correction should not be needed for future datasets.

    Parameters
    ----------
    el_deg : np.ndarray
        Elevation angles in degrees, shape (nsamples,).
    az_deg : np.ndarray
        Azimuth angles in degrees, shape (nsamples,).
    dpss_red : np.ndarray
        DPSS-reduced power, shape (nsamples, nfreq). Even frequency indices
        are the strong arm; odd indices are the weak arm.

    Returns
    -------
    scale_factors : np.ndarray
        Multiplicative scale factors for weak-arm channels, shape (nfreq//2,).
    peak_powers : np.ndarray
        Fitted peak power at each frequency, shape (nfreq,). NaN where the
        fit failed.
    """
    crossings = np.where(np.diff(np.sign(el_deg)))[0]
    valid_crossings = [i for i in crossings if abs(el_deg[i]) < 10]

    az_at_0 = []
    power_at_0 = []
    for i in valid_crossings:
        el0, el1 = el_deg[i], el_deg[i + 1]
        az0, az1 = az_deg[i], az_deg[i + 1]
        p0, p1 = dpss_red[i], dpss_red[i + 1]
        t = (0.0 - el0) / (el1 - el0)
        az_at_0.append(az0 + t * (az1 - az0))
        power_at_0.append(p0 + t * (p1 - p0))

    az_at_0 = np.array(az_at_0)
    power_at_0 = np.array(power_at_0)

    def _malus_law(az, peak_power, min_power, phase_offset):
        az_rad = np.deg2rad(az)
        phase_rad = np.deg2rad(phase_offset)
        return min_power + (peak_power - min_power) * np.cos(az_rad - phase_rad) ** 2

    num_freqs = dpss_red.shape[1]
    peak_powers = np.full(num_freqs, np.nan)
    for f_idx in range(num_freqs):
        p_freq = power_at_0[:, f_idx]
        valid = np.isfinite(p_freq) & np.isfinite(az_at_0)
        if np.sum(valid) < 4:
            continue
        p_v, az_v = p_freq[valid], az_at_0[valid]
        try:
            popt, _ = curve_fit(
                _malus_law, az_v, p_v,
                p0=[np.max(p_v), np.min(p_v), az_v[np.argmax(p_v)]],
            )
            peak_powers[f_idx] = popt[0]
        except RuntimeError:
            peak_powers[f_idx] = np.max(p_v)

    idx = np.arange(num_freqs)
    idx_strong, peaks_strong = idx[0::2], peak_powers[0::2]
    idx_weak, peaks_weak = idx[1::2], peak_powers[1::2]
    valid_strong = np.isfinite(peaks_strong)

    expected_strong_at_weak = np.full_like(peaks_weak, np.nan)
    if np.any(valid_strong):
        expected_strong_at_weak = np.interp(
            idx_weak, idx_strong[valid_strong], peaks_strong[valid_strong]
        )

    scale_factors = expected_strong_at_weak / peaks_weak
    return scale_factors, peak_powers
