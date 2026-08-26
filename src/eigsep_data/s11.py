from datetime import datetime
import numpy as np
from scipy import signal

from cmt_vna import calkit
from eigsep_observing import io

# +
"""Write / read per-DUT HDF5 files for calibrated field VNA data.

Takes the in-memory ``caled_s11s`` structure produced by an
after-the-fact field-calibration pipeline --

    caled_s11s[dut][timestamp][cal_plane] -> complex S11 array, (Nfreq,)

(e.g. dut in {"ant", "rec", "amb", "load", "noise", "sp1_open",
"sp1_short", "sp1"}; timestamp a unix float; cal_plane a string
naming which calibration reference plane that trace is expressed at,
e.g. "raw" / "vna_port_corrected" / "final") -- and writes one HDF5
file per DUT, instead of one file per capture sorted by timestamp.
Each file holds every timestamp recorded for that DUT and every
calibration plane recorded at each timestamp, so nothing is thrown
away, while still keeping the common case ("give me the final
calibrated trace") a one-line lookup.

Layout of one ``<dut>_calibrated.h5`` file::

    /                       .attrs: dut, n_timestamps
    /timestamps             (n_timestamps,) float64 -- sorted, unix seconds
    /freqs                  (Nfreq,) float64 -- shared freq axis, if given
    /t0/                    .attrs: timestamp (float), cal_planes (list[str])
        /t0/<cal_plane>     (Nfreq,) complex128, one per plane recorded
        /t0/default         soft link -> <cal_plane> named by final_plane,
                             if that plane was recorded at this timestamp
    /t1/ ...
    ...

``timestamps`` is the single source of truth for "what times exist for
this DUT" -- read it once rather than listing group names. The
``t<i>`` group names are positional (index into the sorted
``timestamps`` array), not the timestamp value itself, so there's no
float-to-string encoding to get wrong or round-trip.

Assumes numeric (unix-float) timestamps, matching the ``*_unix``
convention used throughout the rest of this codebase's metadata /
provenance fields. If your timestamps are ``datetime`` objects,
convert with ``.timestamp()`` before calling.
"""

from pathlib import Path

import h5py
import numpy as np


def write_dut_calibration_h5(
    caled_s11s,
    *,
    save_dir,
    freqs=None,
    final_plane=None,
    fname_template="{dut}_calibrated.h5",
):
    """Write one HDF5 file per DUT from a nested calibration dict.

    Parameters
    ----------
    caled_s11s : dict
        ``{dut: {timestamp: {cal_plane: s11_array}}}``. Not every
        timestamp needs the same set of calibration planes, and not
        every DUT needs the same number of timestamps.
    save_dir : str or Path
        Directory the per-DUT files are written into. Must already
        exist.
    freqs : array-like, optional
        Shared frequency axis (Hz), written into every file if given.
        Omit if your S11 arrays don't share one common frequency axis
        (e.g. different npoints/fstart/fstop between captures) -- in
        that case attach a per-timestamp freq axis yourself, or pass a
        per-DUT dict of freqs and call this once per DUT instead.
    final_plane : str, optional
        Name of the calibration plane most callers want (e.g.
        ``"final"``). If given and present at a given timestamp, a
        soft link named ``"default"`` is added inside that
        timestamp's group, pointing at it -- so ``f["t0/default"]``
        (or :func:`read_final_plane`) works without knowing the plane
        name, while every plane -- including the aliased one -- stays
        reachable by its own name too.
    fname_template : str, optional
        Output filename per DUT; formatted with ``dut=<dut name>``.
        Default ``"{dut}_calibrated.h5"``.

    Returns
    -------
    dict[str, Path]
        ``{dut: written_file_path}`` -- DUTs with zero timestamps
        recorded are skipped (no empty file written) and absent from
        this dict.

    Raises
    ------
    ValueError
        If ``save_dir`` doesn't exist / isn't a directory, or if
        ``caled_s11s`` is empty.
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
    """Read one file written by :func:`write_dut_calibration_h5`.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dut : str
    timestamps : np.ndarray
        Sorted timestamps recorded in this file (float, unix seconds).
    freqs : np.ndarray or None
        Shared frequency axis, if one was written.
    by_time : dict[float, dict[str, np.ndarray]]
        ``{timestamp: {cal_plane: s11_array}}`` -- the same shape as
        one DUT's slice of the original ``caled_s11s`` input. Every
        plane that was written is included under its own name; if a
        ``final_plane`` alias was written it's also present under the
        key ``"default"``.
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
    """Convenience reader: just the commonly-used plane, per timestamp.

    The one-line lookup most callers want -- skips every other
    calibration plane in the file.

    Parameters
    ----------
    path : str or Path
    final_plane_name : str, optional
        The alias written by ``final_plane=`` in
        :func:`write_dut_calibration_h5` (default ``"default"``). A
        timestamp where that plane wasn't recorded is simply absent
        from the result rather than raising.

    Returns
    -------
    dict[float, np.ndarray]
        ``{timestamp: s11_array}``.
    """
    with h5py.File(path, "r") as f:
        timestamps = f["timestamps"][:]
        out = {}
        for idx, ts in enumerate(timestamps):
            grp = f[f"t{idx}"]
            if final_plane_name in grp:
                out[float(ts)] = grp[final_plane_name][:]
    return out


if __name__ == "__main__":
    # Self-test with synthetic data shaped like the real input, run
    # via `python field_calibration_io.py`. Not pytest -- this module
    # lives outside eigsep_observing's own test suite since it's a
    # standalone field-analysis helper, not part of the package.
    import shutil
    import tempfile

    rng = np.random.default_rng(0)
    duts = [
        "ant",
        "rec",
        "amb",
        "load",
        "noise",
        "sp1_open",
        "sp1_short",
        "sp1",
    ]
    nfreq = 16
    freqs = np.linspace(1e6, 250e6, nfreq)
    planes = ["raw", "vna_port_corrected", "final"]

    caled_s11s = {}
    for i, dut in enumerate(duts):
        n_times = (i % 3) + 1  # variable number of timestamps per DUT
        by_time = {}
        base_ts = 1_700_000_000.0
        for j in range(n_times):
            ts = base_ts + j * 3600.0
            # Not every timestamp has every plane -- exercise the
            # partial-set path too.
            these_planes = planes if j % 2 == 0 else planes[:2]
            by_time[ts] = {
                p: rng.standard_normal(nfreq) + 1j * rng.standard_normal(nfreq)
                for p in these_planes
            }
        caled_s11s[dut] = by_time
    # One DUT present in the dict but with no captures -- must be
    # skipped cleanly (no empty file).
    caled_s11s["sp1_never_measured"] = {}

    tmpdir = Path(tempfile.mkdtemp())
    try:
        written = write_dut_calibration_h5(
            caled_s11s,
            save_dir=tmpdir,
            freqs=freqs,
            final_plane="final",
        )
        assert set(written) == set(duts), (set(written), set(duts))
        assert "sp1_never_measured" not in written

        for dut in duts:
            path = written[dut]
            r_dut, r_ts, r_freqs, r_by_time = read_dut_calibration_h5(path)
            assert r_dut == dut
            np.testing.assert_allclose(r_freqs, freqs)
            expected_ts = sorted(caled_s11s[dut])
            np.testing.assert_allclose(sorted(r_ts), expected_ts)
            assert set(r_by_time) == set(expected_ts)
            for ts, orig_planes in caled_s11s[dut].items():
                got_planes = r_by_time[ts]
                # every recorded plane round-trips exactly
                for p, arr in orig_planes.items():
                    np.testing.assert_allclose(got_planes[p], arr)
                # "final" plane, when present, is also reachable via
                # the "default" alias
                if "final" in orig_planes:
                    np.testing.assert_allclose(
                        got_planes["default"], orig_planes["final"]
                    )
                else:
                    assert "default" not in got_planes

            final_only = read_final_plane(path)
            expected_final_ts = {
                ts for ts, p in caled_s11s[dut].items() if "final" in p
            }
            assert set(final_only) == expected_final_ts, (
                dut,
                set(final_only),
                expected_final_ts,
            )
            for ts in expected_final_ts:
                np.testing.assert_allclose(
                    final_only[ts], caled_s11s[dut][ts]["final"]
                )

        print(f"ALL SELF-TESTS PASSED ({len(written)} DUT files)")
    finally:
        shutil.rmtree(tmpdir)


# -


