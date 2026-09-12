from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from eigsep_observing import io

from .clock import (  # noqa: F401  -- to_unix_time is re-exported
    parse_filename_time,
    to_unix_time,
)


def _parse_time_from_name(fname: str, tz=None) -> datetime:
    """Alias for :func:`eigsep_data.clock.parse_filename_time`."""
    return parse_filename_time(fname, tz=tz)


#: Per-row stamps that are averaged when ``time_avg > 1``.
_AVG_COLS = ("time", "time_best", "acc_cnt")

#: Channel count assumed when no selected file declares one. Only ever
#: used to size a NaN stand-in for a key that no selected file carries,
#: and to estimate how much memory a load will take.
_DEFAULT_NCHAN = 1024


def _as_spectra(arr):
    """
    Apply ``read_hdf5``'s storage rule: (re, im) int32 pairs are complex.

    The rule is duplicated from ``eigsep_observing.io.read_hdf5`` -- its
    source, and the place to look if the storage convention ever changes
    -- because ``read_hdf5`` loads whole files and this reads only the
    selected rows.
    """
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[-1] == 2 and arr.dtype.kind == "i":
        return arr[..., 0].astype(np.float64) + 1j * arr[..., 1].astype(
            np.float64
        )
    return arr


def _avg_rows(block, time_avg):
    """Mean over consecutive groups of *time_avg* rows; float32 for
    autos, complex64 for crosses. The caller trims the remainder."""
    nout = block.shape[0] // time_avg
    out = block.reshape((nout, time_avg) + block.shape[1:]).mean(axis=1)
    return out.astype(np.complex64 if out.dtype.kind == "c" else np.float32)


def _nan_block(nrows, nchan, key):
    """
    An all-NaN stand-in for the rows of a file that lacks *key*.

    Whether *key* is a cross is decided from its name, not from
    :func:`_as_spectra`'s shape rule: there is no array to inspect --
    that is the whole point -- so the two-character key name is the only
    information available.
    """
    dtype = np.complex128 if len(key) == 2 else np.float64
    return np.full((nrows, nchan), np.nan, dtype=dtype)


def _estimate_bytes(nrows, nchan, keys, time_avg):
    """Bytes one load will hold. Crosses are wider, and again the key
    name is the only cross test available before anything is read."""
    if time_avg > 1:
        width = sum(8 if len(k) == 2 else 4 for k in keys)
    else:
        width = sum(16 if len(k) == 2 else 4 for k in keys)
    return (nrows // time_avg) * nchan * width


def _mem_available():
    """Bytes of MemAvailable from /proc/meminfo, or None."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _warn_if_large(nbytes):
    """Warn when a load would take more than half of free memory. Peak
    usage is about twice *nbytes*, during the concatenation."""
    avail = _mem_available()
    if avail is not None and nbytes > avail / 2:
        warnings.warn(
            f"Loading ~{nbytes / 1e9:.1f} GB of spectra with "
            f"{avail / 1e9:.1f} GB available; narrow the selection, "
            "drop keys, or pass time_avg.",
            # 3, not 2: the advice is for whoever called
            # from_selection, which is this helper's own caller.
            stacklevel=3,
        )


def _missing_keys_error(absent):
    """
    The ``KeyError`` for keys absent from some of the selected files.

    Built in one place because it is raised twice: once from the
    metadata before any spectra are read, and once from the files
    themselves as a backstop (see
    :meth:`EigsepData.from_selection`).
    """
    detail = "; ".join(
        f"key {k!r} is absent from {sorted(set(v))}" for k, v in absent.items()
    )
    return KeyError(
        f"{detail}. The selection straddles a change in recorded data "
        "keys: narrow it with a data_keys or filter_phase filter, or "
        "pass missing='nan'."
    )


@dataclass
class EigsepData:

    data: dict[str, np.ndarray] = None
    acc_cnt: np.ndarray = None
    #: ``time_best`` of each row: the header time where the file's clock
    #: was sane, a filename-derived estimate otherwise. The raw header
    #: time is ``meta.time``. As built by :meth:`from_selection` this is
    #: a *view* of ``meta.time_best`` -- the same numbers, deliberately
    #: not copied -- so writing into it edits ``meta`` too; take a copy
    #: before any in-place arithmetic.
    times: np.ndarray = None
    freq: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 250, num=1024, endpoint=False)
    )
    #: Per-integration metadata, one row per entry in ``times``.
    meta: pd.DataFrame = None

    @classmethod
    def from_selection(cls, selection, keys=None, time_avg=1, missing="raise"):
        """
        Read the spectra for the integrations in *selection*.

        Only the chosen rows of the chosen keys are read, so selecting a
        few percent of a deployment costs a few percent of the I/O.
        Crosses come back complex, exactly as ``read_hdf5`` returns
        them. Output rows are in selection order (``time_best``), even
        where two files interleave in time.

        Parameters
        ----------
        selection : eigsep_data.index.Selection
        keys : list of str or None
            Data keys to read. ``None`` reads every key common to the
            selected files.
        time_avg : int
            Average this many consecutive selected rows *within each
            file* into one output row; a trailing remainder is dropped
            with a warning. Autos become float32 and crosses complex64,
            which is what lets a whole day fit in a laptop.
        missing : {"raise", "nan"}
            What to do when a requested key is absent from one of the
            selected files (a wiring-phase change inside the window).

        Returns
        -------
        EigsepData
            ``freq`` comes from the first selected file that carries a
            ``freqs`` header, and ``meta`` is one row per entry in
            ``times``.
        """
        if missing not in ("raise", "nan"):
            raise ValueError("missing must be 'raise' or 'nan'")
        time_avg = int(time_avg)
        if time_avg < 1:
            raise ValueError("time_avg must be >= 1")
        meta = selection.meta.reset_index(drop=True)
        if len(meta) == 0:
            raise ValueError(
                "Selection contains no integrations; nothing to load."
            )
        data_dir = selection.index.data_dir
        if keys is None:
            # A file with no data group at all declares nothing, and
            # must not drag the intersection to empty: every key is
            # simply absent from it, which the missing= policy covers.
            per_file = [
                set(str(k).split(","))
                for k in meta.data_keys.unique()
                if str(k)
            ]
            if not per_file:
                raise ValueError(
                    "No selected file records any data keys; there is "
                    "nothing to read."
                )
            keys = sorted(set.intersection(*per_file))
            if not keys:
                raise ValueError(
                    "The selected files share no data keys "
                    f"({[sorted(p) for p in per_file]}); name the keys to "
                    "read explicitly, or narrow the selection to one "
                    "wiring phase."
                )
        # A bare string is one key, not an iterable of characters:
        # list("04") would read the two autos instead of the cross. A
        # repeated key is read once: blocks is keyed by name, so a
        # duplicate would append a second block per file and hand back
        # one file's rows in another file's place.
        keys = list(dict.fromkeys([keys] if isinstance(keys, str) else keys))
        # A fallback only: the real width comes from the data below.
        nchan = _DEFAULT_NCHAN
        if "nchan" in meta:
            declared = pd.to_numeric(meta.nchan, errors="coerce").dropna()
            if not declared.empty:
                nchan = int(declared.iloc[0])

        if missing == "raise" and "data_keys" in meta:
            # The same column that infers `keys` above says which files
            # carry them, so a straddled phase boundary -- a routine
            # thing in deployment 5 -- can be reported before spending
            # the I/O and the peak memory of the whole selection. The
            # per-file check in the read loop stays as the backstop for
            # an index that has gone stale against the files.
            undeclared = {}
            per_file_keys = meta.groupby("file", sort=False).data_keys.first()
            for fname, spec in per_file_keys.items():
                for key in set(keys) - set(str(spec).split(",")):
                    undeclared.setdefault(key, []).append(fname)
            if undeclared:
                raise _missing_keys_error(undeclared)

        _warn_if_large(_estimate_bytes(len(meta), nchan, keys, time_avg))

        blocks = {k: [] for k in keys}
        absent = {}
        positions, stamps, files_read = [], [], []
        dropped = 0
        freq = None
        for name, group in meta.groupby("file", sort=False):
            pos = group.index.to_numpy()
            rows = group.row.to_numpy()
            keep = (len(rows) // time_avg) * time_avg
            dropped += len(rows) - keep
            if keep == 0:
                continue
            pos, rows = pos[:keep], rows[:keep]
            files_read.append(name)
            with h5py.File(data_dir / name, "r") as h5:
                if freq is None and "freqs" in h5["header"]:
                    freq = np.asarray(h5["header"]["freqs"])
                # One contiguous span covering the selected rows, then
                # the rows out of it: h5py's own fancy indexing reads
                # element by element, and a corr file is ~60 rows.
                lo, hi = int(rows.min()), int(rows.max()) + 1
                local = rows - lo
                # A file with no data group at all is indexable (the
                # scanner tolerates it), and every key is then absent.
                datasets = h5["data"] if "data" in h5 else {}
                for key in keys:
                    if key in datasets:
                        block = _as_spectra(datasets[key][lo:hi][local])
                        if time_avg > 1:
                            block = _avg_rows(block, time_avg)
                    else:
                        # How wide the NaN stand-in must be is not known
                        # yet -- the files that do carry this key may not
                        # have been read. Note the slot it goes in and
                        # fill it once every real block is in hand.
                        absent.setdefault(key, []).append(
                            (name, len(blocks[key]), keep)
                        )
                        block = None
                    blocks[key].append(block)
            if time_avg > 1:
                # Only now is meta touched: at full resolution its
                # stamps are the header's own values, untouched.
                stamp = group[list(_AVG_COLS)].to_numpy(dtype=float)[:keep]
                stamp = stamp.reshape(-1, time_avg, len(_AVG_COLS))
                stamps.append(stamp.mean(axis=1))
                pos = pos.reshape(-1, time_avg)[:, 0]
            positions.append(pos)

        if absent and missing == "raise":
            raise _missing_keys_error(
                {k: [fname for fname, _, _ in v] for k, v in absent.items()}
            )
        if not positions:
            raise ValueError(
                f"time_avg={time_avg} exceeds every file's contribution; "
                "nothing to load."
            )
        if dropped:
            warnings.warn(
                f"time_avg={time_avg}: dropped {dropped} trailing rows "
                "that did not fill a block.",
                stacklevel=2,
            )
        for key, slots in absent.items():
            real = next((b for b in blocks[key] if b is not None), None)
            # No file carried the key, so no width was ever observed and
            # the header's channel count is all there is to go on.
            width = nchan if real is None else real.shape[-1]
            for _, slot, nrows in slots:
                gap = _nan_block(nrows, width, key)
                blocks[key][slot] = (
                    _avg_rows(gap, time_avg) if time_avg > 1 else gap
                )

        for key, parts in blocks.items():
            if len({part.shape[-1] for part in parts}) > 1:
                # np.concatenate would raise a bare dimension mismatch
                # naming neither the key nor the files.
                detail = ", ".join(
                    f"{fname}: {part.shape[-1]}"
                    for fname, part in zip(files_read, parts)
                )
                raise ValueError(
                    f"key {key!r} has a different channel count in "
                    f"different selected files ({detail}); narrow the "
                    "selection to one correlator configuration."
                )

        gathered = np.concatenate(positions)
        # gathered[i] is the selection position of read row i, so putting
        # the read rows in gathered order inverts the gather. Sorting the
        # times instead would be right only if files never interleaved.
        inverse = np.argsort(gathered, kind="stable")
        data = {
            k: np.concatenate(v, axis=0)[inverse] for k, v in blocks.items()
        }
        out_meta = meta.iloc[gathered[inverse]].reset_index(drop=True)
        if time_avg > 1:
            # The spec's one sanctioned edit to an epoch column: each
            # block keeps its first row, with the stamps it averaged.
            stamp = np.concatenate(stamps)[inverse]
            for i, col in enumerate(_AVG_COLS):
                out_meta[col] = stamp[:, i]
        return cls(
            data=data,
            acc_cnt=out_meta.acc_cnt.to_numpy(),
            times=out_meta.time_best.to_numpy(),
            freq=freq,
            meta=out_meta,
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
            times = [
                _parse_time_from_name(f.name).replace(tzinfo=None)
                for f in files
            ]
            if start_time:
                start_dt = datetime.strptime(start_time, "%Y%m%d_%H%M%S")
                files = [f for f, t in zip(files, times) if t >= start_dt]
                times = [
                    _parse_time_from_name(f.name).replace(tzinfo=None)
                    for f in files
                ]
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
            meta=(
                None
                if self.meta is None
                else self.meta.iloc[min_index:max_index].reset_index(drop=True)
            ),
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
        # Read header["times"] alone before deciding whether to load the
        # file. io.read_hdf5 pulls the full payload at ~36 ms/file, which
        # is ~77x the cost of this peek and is wasted on every file
        # outside the window -- and in a deployment directory of several
        # thousand files, that is nearly all of them. Selection still
        # uses header times, so this changes speed only, not results.
        try:
            with h5py.File(filename, "r") as h5:
                if "header" not in h5 or "times" not in h5["header"]:
                    raise KeyError(
                        f"{filename.name} does not contain header['times']."
                    )
                times_file = np.asarray(h5["header"]["times"])
        except (OSError, KeyError) as e:
            warnings.warn(f"Skipping {filename.name}: {e}")
            continue

        time_mask = (times_file >= start_unix) & (times_file < end_unix)
        if not np.any(time_mask):
            continue

        try:
            data_file, header_file, metadata_file = io.read_hdf5(filename)
        except (OSError, KeyError) as e:
            warnings.warn(f"Skipping {filename.name}: {e}")
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
        headers.append(
            {
                "selected_indices": np.flatnonzero(time_mask),
                "times": times_file[time_mask],
            }
        )
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
        raise ValueError(
            "No integrations found inside the requested time range."
        )

    times = np.concatenate(selected_times)
    sort_index = np.argsort(times)
    times = times[sort_index]
    data_range = {
        k: np.concatenate(v, axis=0)[sort_index]
        for k, v in selected_data.items()
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
    counts_per_deg=62.77777777777778,
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
    counts_per_deg : float
        Stepper counts per degree, used to convert *el_pos* to degrees
        when anchoring the IMU elevation angle. The default follows
        picohost.motor.PicoMotor (step_angle_deg=1.8, gear_teeth=113,
        microstep=1), i.e. +-11300 counts = +-180 deg.

    Returns
    -------
    dict with keys:
        times : np.ndarray, shape (nsamples,) -- Unix times
        freqs : np.ndarray -- frequency axis from the first file's header
        sky, ground, cross : np.ndarray, shape (nsamples, nchan)
        el_pos, az_pos : np.ndarray -- commanded motor positions
        pot_az_angle : np.ndarray -- raw potentiometer azimuth reading
        imu_el_deg : np.ndarray -- IMU-derived elevation angle, in
            degrees, with sign and zero point anchored to el_pos
        imu_accel : np.ndarray, shape (nsamples, 3) -- raw accelerometer
            (x, y, z)
    """
    start_unix = to_unix_time(start_time)
    end_unix = to_unix_time(end_time)
    if end_unix <= start_unix:
        raise ValueError("end_time must be later than start_time.")

    times, freqs, data_range, headers, metadata, sort_index = (
        _select_h5_in_range(data_dir, start_unix, end_unix, file_patterns)
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
            file_pot.append(
                np.nan if entry is None else entry.get("pot_az_angle", np.nan)
            )
        pot_list.append(np.asarray(file_pot, dtype=float))

        imu_el = meta["imu_el"]
        file_accel = []
        for idx in indices:
            entry = imu_el[idx]
            if entry is None:
                file_accel.append((np.nan, np.nan, np.nan))
            else:
                file_accel.append(
                    (
                        entry.get("accel_x", np.nan),
                        entry.get("accel_y", np.nan),
                        entry.get("accel_z", np.nan),
                    )
                )
        accel_list.append(np.asarray(file_accel, dtype=float))

    el_pos = np.concatenate(el_list)[sort_index]
    az_pos = np.concatenate(az_list)[sort_index]
    pot_az_angle = np.concatenate(pot_list)[sort_index]
    accel = np.concatenate(accel_list)[sort_index]

    imu_el_deg = imu_el_from_accel(accel, el_pos, counts_per_deg)

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
        out = {
            k: (v if k == "freqs" else v[sweep_slice]) for k, v in out.items()
        }
    return out


def imu_el_from_accel(accel, el_pos, counts_per_deg=62.77777777777778):
    """
    Derive elevation in degrees from accelerometer readings.

    An SVD of the accelerometer vectors finds the plane gravity sweeps
    out as the antenna tilts; the angle within that plane tracks
    elevation.

    Finding the plane is well posed, but the basis (u, v) spanning it is
    arbitrary up to a rotation within the plane and a reflection, and
    LAPACK's choice depends on the rows it is given. Two calls over
    different time windows of one scan therefore disagree on both the
    sign and the origin of the angle. Both are anchored to the commanded
    motor position: the IMU still supplies the precise angle, *el_pos*
    only resolves the two-fold sign and pins the constant offset.

    Parameters
    ----------
    accel : np.ndarray, shape (nsamples, 3)
        Accelerometer (x, y, z). Rows containing NaN (dropped IMU
        readings) are excluded from the SVD and left NaN in the output,
        since np.linalg.svd raises on NaN input.
    el_pos : np.ndarray, shape (nsamples,)
        Commanded motor elevation, in stepper counts.
    counts_per_deg : float
        Stepper counts per degree, used to convert *el_pos* to degrees.

    Returns
    -------
    imu_el_deg : np.ndarray, shape (nsamples,)
        Elevation in degrees, NaN where the IMU reading was dropped.
    """
    accel = np.asarray(accel, dtype=float)
    el_pos = np.asarray(el_pos, dtype=float)
    valid = ~np.any(np.isnan(accel), axis=1)
    imu_el_deg = np.full(accel.shape[0], np.nan)
    if not np.any(valid):
        return imu_el_deg

    _, _, Vt = np.linalg.svd(accel[valid], full_matrices=False)
    u, v = Vt[0, :], Vt[1, :]
    proj_x = accel[valid] @ u
    proj_y = accel[valid] @ v
    raw_deg = np.unwrap(np.degrees(np.arctan2(proj_y, proj_x)), period=360)

    el_motor_deg = el_pos[valid] / counts_per_deg
    anchor = np.isfinite(el_motor_deg)
    if np.count_nonzero(anchor) < 2:
        warnings.warn(
            "No usable motor el_pos to anchor the IMU elevation angle; "
            "its sign and zero point are arbitrary and may differ between "
            "calls over different time windows."
        )
        imu_el_deg[valid] = raw_deg - 180
        return imu_el_deg

    cov = np.cov(raw_deg[anchor], el_motor_deg[anchor])[0, 1]
    # cov == 0 means the scan holds one elevation, where the sign is
    # unobservable and immaterial: the offset below absorbs it.
    sign = -1.0 if cov < 0 else 1.0
    offset = np.median(el_motor_deg[anchor] - sign * raw_deg[anchor])
    imu_el_deg[valid] = sign * raw_deg + offset
    return imu_el_deg


def extract_clean_pot_data_v2(
    az_pot, az_step, min_stable_samples=10, settle_samples=3
):
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
    dropped = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if (end - start) >= min_stable_samples:
            safe_start = min(start + settle_samples, end - 1)
            window = az_pot[safe_start:end]
            if not np.any(np.isfinite(window)):
                # A dropout run can cover an entire plateau. Keep it out
                # of `plateaus` so a NaN median cannot poison the
                # neighbouring ramps, and record it so it stays NaN
                # below: there is no measurement here, and interpolating
                # would assign azimuths the motor never visited.
                dropped.append((start, end))
                continue
            # nanmedian, not median: a single missing sample would
            # otherwise turn the whole plateau NaN.
            plateau_median = np.nanmedian(window)
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

    # Applied last: the ramp and edge fills above write across these
    # spans, and a plateau with no data must stay flagged as missing.
    for start, end in dropped:
        clean_az_pot[start:end] = np.nan

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

    num_freqs = dpss_red.shape[1]
    if not valid_crossings:
        # e.g. a fixed-elevation azimuth raster. Without this, power_at_0
        # is shape (0,) rather than (0, nfreq) and the per-frequency loop
        # below raises IndexError before its own validity check runs.
        warnings.warn(
            "No el=0 crossings within +-10 deg; cannot fit the weak-arm "
            "calibration. Returning NaN scale factors."
        )
        return np.full(num_freqs // 2, np.nan), np.full(num_freqs, np.nan)

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
        return (
            min_power
            + (peak_power - min_power) * np.cos(az_rad - phase_rad) ** 2
        )

    peak_powers = np.full(num_freqs, np.nan)
    for f_idx in range(num_freqs):
        p_freq = power_at_0[:, f_idx]
        valid = np.isfinite(p_freq) & np.isfinite(az_at_0)
        if np.sum(valid) < 4:
            continue
        p_v, az_v = p_freq[valid], az_at_0[valid]
        try:
            popt, _ = curve_fit(
                _malus_law,
                az_v,
                p_v,
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
