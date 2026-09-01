"""Read and calibrate EIGSEP VNA S11 measurements.

Two entry points, for two different data products:

``RawS11``
    Wraps a single raw S11 capture (as written by
    ``eigsep_observing.io.write_s11_file``) and calibrates it in-process
    against an ideal open/short/load model. Quick-look path -- needs no
    lab-characterized switch-path or OSL files.

``S11``
    Reads the per-DUT HDF5 products written by
    ``scripts/calibrate_field_s11.py`` (via
    :func:`write_dut_calibration_h5`), which carry the full
    vna -> dut -> lna calibration chain::

        caled_s11s[dut][timestamp][cal_plane] -> complex S11 array, (Nfreq,)
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy import signal

from cmt_vna import calkit
from eigsep_observing import io


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
    rec_planes : list of str or None
        The largest set of calibration planes actually reached, at
        one loaded timestamp, by the ``"rec"`` DUT -- the
        receiver-mode measurement. ``None`` if no ``"rec"`` file was
        loaded.
    ant_planes : list of str or None
        The largest set of calibration planes actually reached, at
        one loaded timestamp of one DUT, among every DUT *other than*
        ``"rec"`` -- the antenna-mode measurements (``"ant"``,
        ``"amb"``, ``"load"``, ``"noise"``,
        ``"sp1"``/``"sp1_open"``/``"sp1_short"``, ...). ``None`` if no
        such DUT was loaded. This is the plane list of whichever
        single DUT/timestamp goes deepest (e.g. "ant"/"amb" reach
        "lna", "load"/"noise" stop at "vna" -- see
        ``scripts/calibrate_field_s11.py`` -- so this is "ant"'s/
        "amb"'s 4-plane set), *not* a set-union across DUTs -- a union
        could report plane names that no single DUT/timestamp actually
        has together.
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

        self.rec_planes = self._planes_for(["rec"])
        self.ant_planes = self._planes_for(
            [dut for dut in self.duts if dut != "rec"]
        )

    def _planes_for(self, duts):
        """The largest plane set actually reached by any one loaded
        (dut, timestamp) pair among ``duts`` -- e.g. across "ant"'s
        and "load"'s timestamps, whichever single one has the most
        planes.

        Deliberately *not* a set-union across duts/timestamps: planes
        are a nested chain per capture (raw -> vna -> dut -> lna, see
        ``scripts/calibrate_field_s11.py``), but a union would still
        merge in anything oddly present at just one timestamp and
        report it as if the whole group had it. Returning one real,
        actually-achieved plane set avoids that.

        ``duts`` not present in ``self.duts`` are silently ignored
        (not an error -- callers pass a fixed candidate list, e.g.
        every non-"rec" DUT, regardless of what actually loaded).
        Returns ``None`` if none of ``duts`` were loaded, or none of
        the ones that were have any timestamps recorded. Ties (more
        than one (dut, timestamp) reaching the same max plane count)
        resolve to whichever is encountered first, in ``duts`` order.
        """
        best = None
        for dut in duts:
            by_time = getattr(self, dut, None)
            if not by_time:
                continue
            for plane_dict in by_time.values():
                planes = sorted(plane_dict.keys())
                if best is None or len(planes) > len(best):
                    best = planes
        return best

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


class RawS11:
    """A single raw S11 capture, calibrated to the VNA's internal
    reference plane against an ideal open/short/load model.

    Quick-look counterpart to :class:`S11`: it reads one raw file
    directly and needs no lab-characterized switch-path or OSL inputs,
    so it stays usable for datasets that have not been run through
    ``scripts/calibrate_field_s11.py``. It calibrates only as deep as
    the VNA reference plane -- use :class:`S11` when the switch-path
    de-embedding to the DUT/LNA planes is needed.
    """

    def __init__(self, fpath):
        """
        Parameters
        ----------
        fpath : pathlib.Path
            File path of S11 measurement.

        """
        self.data, self.cal_data, self.hdr, self.meta = io.read_s11_file(
            fpath
        )
        self.time = datetime.fromisoformat(fpath.name[-18:-3])
        self.timestamp = self.time.timestamp()
        self.freqs = np.array(self.hdr["freqs"]) / 1e6  # in MHz
        self.dlys = (
            np.fft.fftfreq(self.freqs.size, d=self.freqs[1] - self.freqs[0])
            * 1e3
        )  # in ns

        osl_model = np.array([1, -1, 0])  # open, short, load
        osl_model.shape = (3, 1)  # second axis is freq
        self.osl_model = np.repeat(osl_model, self.freqs.size, axis=1)

        self._s11_cal = {}
        self._s11_dly = {}

    def calibrate_s11(self, key):
        """
        First stage calibration at internal reference plane.
        S-parameters of internal network must also be de-embedded.

        Parameters
        ----------
        key : str
            Which measurment to calibrate; `ant`, `noise`, or `load`
            for `ants11` files or `rec` for `recs11` files.

        Returns
        -------
        np.ndarray
            Calibrated S11 data for the given key.

        """
        o = self.cal_data["VNAO"]
        s = self.cal_data["VNAS"]
        load = self.cal_data["VNAL"]
        osl = np.array([o, s, load])
        network_sparams = calkit.network_sparams(self.osl_model, osl)
        return calkit.de_embed_sparams(network_sparams, self.data[key])

    @property
    def s11_cal(self):
        if not self._s11_cal:
            self._s11_cal = {k: self.calibrate_s11(k) for k in self.data}
        return self._s11_cal

    @property
    def s11_dly(self):
        if not self._s11_dly:
            bh = signal.windows.blackmanharris(self.freqs.size, sym=False)
            self._s11_dly = {
                k: np.abs(np.fft.fft(self.s11_cal[k] * bh)) for k in self.data
            }
        return self._s11_dly
