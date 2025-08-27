from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np

from eigsep_observing import io

@dataclass
class EigsepData:

    data : dict[str, np.ndarray] = None
    acc_cnt : np.ndarray = None
    times : np.ndarray = None
    freq : np.ndarray = field(default_factory=lambda: np.linspace(0, 250, num=1024, endpoint=False))

    @classmethod
    def from_path(cls, path: Path, pacific_to_mountain: bool = True):
        """
        Create an EigsepData instance from a directory or a file.

        Parameters
        ----------
        path : Path
            The path to the directory or file.

        Returns
        -------
        EigsepData
        
        """
        if path.is_dir():
            files = sorted(path.glob("corr*.h5"))
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
                warnings.warn(
                    f"Failed to read {f}: {e}. Skipping this file."
                )
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
