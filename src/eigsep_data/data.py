from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np

from eigsep_observing import io


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
