from datetime import datetime
import numpy as np
from scipy import signal

from cmt_vna import calkit as cal
import os
from eigsep_observing import io
from pathlib import Path
import h5py
import numpy as np

from dataclasses import dataclass

"""Write / read per-DUT HDF5 files for calibrated field VNA data.
    Takes the in-memory ``caled_s11s`` structure produced by an
    after-the-fact field-calibration pipeline --

        caled_s11s[dut][timestamp][cal_plane] -> complex S11 array, (Nfreq,)
"""


def write_dut_calibration_h5(
    caled_s11s,
    *,
    save_dir,
    freqs=None,
    final_plane=None,
    fname_template="{dut}_calibrated.h5",
):
    """
    Write one HDF5 file per DUT from a nested calibration dict.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        raise ValueError(
            f"save_dir does not exist or is not a directory: {save_dir}"
        )
    if not caled_s11s:
        raise ValueError("caled_s11s is empty -- nothing to write")

    written = {}
    for dut, by_time in caled_s11s.items():
        if not by_time:
            continue  # no timestamps recorded for this DUT -- skip
        timestamps = sorted(by_time)
        path = save_dir / fname_template.format(dut=dut)
        with h5py.File(path, "w") as f:
            f.attrs["dut"] = dut
            f.attrs["n_timestamps"] = len(timestamps)
            f.create_dataset(
                "timestamps", data=np.asarray(timestamps, dtype=float)
            )
            if freqs is not None:
                f.create_dataset("freqs", data=np.asarray(freqs, dtype=float))
            for idx, ts in enumerate(timestamps):
                planes = by_time[ts]
                grp = f.create_group(f"t{idx}")
                grp.attrs["timestamp"] = float(ts)
                grp.attrs["cal_planes"] = sorted(planes.keys())
                for plane, s11 in planes.items():
                    grp.create_dataset(
                        plane, data=np.asarray(s11, dtype=complex)
                    )
                if final_plane is not None and final_plane in planes:
                    grp["default"] = h5py.SoftLink(f"/t{idx}/{final_plane}")
        written[dut] = path
    return written


def read_dut_calibration_h5(path):
    """
    Read one file written by :func:`write_dut_calibration_h5`.
    """
    with h5py.File(path, "r") as f:
        dut = f.attrs["dut"]
        timestamps = f["timestamps"][:]
        freqs = f["freqs"][:] if "freqs" in f else None
        by_time = {}
        for idx, ts in enumerate(timestamps):
            grp = f[f"t{idx}"]
            by_time[float(ts)] = {plane: grp[plane][:] for plane in grp}
    return dut, timestamps, freqs, by_time


def read_final_plane(path, final_plane_name="default"):
    """
    Convenience reader: just the commonly-used plane, per timestamp.
    """
    with h5py.File(path, "r") as f:
        timestamps = f["timestamps"][:]
        out = {}
        for idx, ts in enumerate(timestamps):
            grp = f[f"t{idx}"]
            if final_plane_name in grp:
                out[float(ts)] = grp[final_plane_name][:]
    return out


@dataclass
class S11Sample:
    """One paired (dut, receiver) lookup from :meth:`S11.get_s11`.

    ``dut_time``/``rec_time`` are the actual timestamps matched --
    each DUT (including "rec") has its own set of capture times, so
    these can differ from each other and from the timestamp that was
    requested.
    """

    dut: str
    dut_time: float
    dut_s11: np.ndarray
    rec_time: float
    rec_s11: np.ndarray


class S11:
    """Calibrated field S11 data, loaded from the per-DUT HDF5 files
    written by ``scripts/calibrate_field_s11.py`` (via
    :func:`write_dut_calibration_h5`).

    Parameters
    ----------
    caldir : str or Path
        Directory containing the ``<dut>_calibrated.h5`` files.
    pattern : str, optional
        Glob pattern (relative to ``caldir``) selecting which files
        to load. Default ``"*_calibrated.h5"``.

    Attributes
    ----------
    freqs : np.ndarray or None
        Shared frequency axis (Hz), read off whichever file has one.
    duts : list of str
        Names of the DUTs actually loaded. Each is also available as
        an instance attribute of the same name, e.g. ``self.ant``,
        ``self.rec`` -- ``{timestamp: {cal_plane: s11_array}}``, the
        same shape as :func:`read_dut_calibration_h5`'s ``by_time``.
    """

    def __init__(self, caldir, pattern="*_calibrated.h5"):
        caldir = Path(caldir)
        paths = sorted(caldir.glob(pattern))
        if not paths:
            raise ValueError(f"no files matching {pattern!r} in {caldir}")

        self.duts = []
        self.freqs = None
        for path in paths:
            dut, _timestamps, freqs, by_time = read_dut_calibration_h5(path)
            setattr(self, dut, by_time)
            self.duts.append(dut)
            if freqs is not None:
                if self.freqs is None:
                    self.freqs = freqs
                elif not np.array_equal(self.freqs, freqs):
                    raise ValueError(
                        f"freqs mismatch: {dut!r} does not share the "
                        "same frequency axis as an earlier file in "
                        f"{caldir}"
                    )

    def _nearest(self, dut, timestamp, plane):
        by_time = getattr(self, dut, None)
        if by_time is None:
            raise ValueError(
                f"no data loaded for dut={dut!r}; loaded duts are "
                f"{self.duts}"
            )
        if not by_time:
            raise ValueError(f"{dut!r} has no timestamps recorded")
        times = np.array(list(by_time.keys()))
        nearest_time = float(times[np.argmin(np.abs(times - timestamp))])
        planes = by_time[nearest_time]
        if plane not in planes:
            raise KeyError(
                f"{dut!r} at t={nearest_time} has no {plane!r} plane "
                f"(has: {sorted(planes)})"
            )
        return nearest_time, planes[plane]

    def get_s11(self, dut, timestamp, plane="default"):
        """The calibrated S11 for ``dut``, paired with the receiver
        S11, both closest in time to ``timestamp``.

        Parameters
        ----------
        dut : str
            Which DUT to look up (e.g. "ant", "amb", "sp1_open").
        timestamp : float
            Unix timestamp to match against.
        plane : str, optional
            Calibration plane to read. Default ``"default"`` -- the
            deepest plane written for that DUT (see
            ``scripts/calibrate_field_s11.py``).

        Returns
        -------
        S11Sample
            ``dut``, the matched ``dut_time``/``dut_s11``, and the
            matched ``rec_time``/``rec_s11``.
        """
        dut_time, dut_s11 = self._nearest(dut, timestamp, plane)
        rec_time, rec_s11 = self._nearest("rec", timestamp, plane)
        return S11Sample(dut, dut_time, dut_s11, rec_time, rec_s11)
    
    def get_all_s11s(self, dut, plane):
        """
                Get all the s11s of a plane for a dut. 
        """
        
        s11s = {timestamp: value[plane] for timestamp,value in getattr(self, dut).items()}
        return np.array(list(s11s.keys())), np.array(list(s11s.values()))
        
    @property
    def s11_dly(self):
        if not self._s11_dly:
            bh = signal.windows.blackmanharris(self.freqs.size, sym=False)
            self._s11_dly = {
                k: np.abs(np.fft.fft(self.s11_cal[k] * bh)) for k in self.data
            }
        return self._s11_dly


