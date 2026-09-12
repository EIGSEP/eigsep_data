"""Synthetic correlator files written by the real producer.

Tests build genuine corr files with ``eigsep_observing.io.write_hdf5``
rather than mocking the layout, so what they pin is the producer's
contract. The rfswitch ladder mirrors
``eigsep_observing._test_fixtures.CORR_METADATA``: a steady state, the
writer's UNKNOWN transition guard, a second steady state, a dropout
run of None, and recovery.
"""

import h5py
import numpy as np
import pytest
from eigsep_observing.io import write_hdf5

NCHAN = 1024

RFSWITCH_LADDER = (
    ["RFANT"] * 20
    + ["UNKNOWN"] * 5
    + ["RFNOFF"] * 20
    + [None] * 5
    + ["RFNOFF"] * 10
)


def write_corr_file(
    path,
    *,
    ntimes=60,
    sync_time=1.7843e9,
    acc_cnt0=0,
    integration_time=0.5369,
    run_tag="panda_observe",
    keys=("0", "4"),
    rfswitch=None,
    streams=("motor",),
    el_pos=0.0,
    root_attrs=None,
    seed=0,
):
    """
    Write one synthetic corr file and return its path.

    Keys of length 2 are cross-correlations and are stored as
    ``(ntimes, NCHAN, 2)`` int32 -- (re, im) pairs -- exactly as
    deployment-5 files are, so ``read_hdf5`` reconstructs them as
    complex. ``rfswitch=None`` omits the stream entirely, which is how
    the ~1842 deployment-5 files with no switch metadata look.
    ``root_attrs`` are written at the file root after the producer is
    done, mirroring ``filter_corr_keys.py`` (``filter_phase``,
    ``filtered_keys``, ``mux_copy_0to1``, ``mux_copy_4to5``).
    ``el_pos`` is the motor stream's commanded elevation (in stepper
    counts): a scalar (the default, ``0.0``, matches every caller that
    predates this parameter) or an array-like of length ``ntimes`` for
    a genuine per-sample sweep.
    """
    el_pos = np.broadcast_to(np.asarray(el_pos, dtype=float), (ntimes,))
    rng = np.random.default_rng(seed)
    acc_cnt = np.arange(acc_cnt0, acc_cnt0 + ntimes, dtype=np.int64)
    times = acc_cnt * integration_time + sync_time
    data = {}
    for k in keys:
        if len(k) == 2:
            data[k] = rng.integers(
                -1000, 1000, size=(ntimes, NCHAN, 2), dtype=np.int32
            )
        else:
            data[k] = rng.integers(
                1, 1000, size=(ntimes, NCHAN), dtype=np.int32
            )
    header = {
        "times": times,
        "acc_cnt": acc_cnt,
        "freqs": np.linspace(0, 250, NCHAN, endpoint=False),
        "integration_time": float(integration_time),
        "run_tag": run_tag,
        "adc_mux_sel": 5,
        "nchan": NCHAN,
    }
    md = {}
    if rfswitch is not None:
        md["rfswitch"] = list(rfswitch)[:ntimes]
    if "motor" in streams:
        md["motor"] = [
            {
                "status": "update",
                "sensor_name": "motor",
                "app_id": 1,
                "boot_id": 7,
                "az_pos": float(i),
                "az_target_pos": float(i),
                "el_pos": float(el_pos[i]),
                "el_target_pos": float(el_pos[i]),
            }
            for i in range(ntimes)
        ]
    if "potmon" in streams:
        md["potmon"] = [
            {
                "status": "update",
                "sensor_name": "potmon",
                "app_id": 2,
                "pot_az_angle": 10.0 + i,
                "pot_az_voltage": 1.5,
                "pot_az_near_rail": False,
                "sp1_term_name": "SHORT",
            }
            for i in range(ntimes)
        ]
    if "imu_el" in streams:
        md["imu_el"] = [
            {
                "status": "update",
                "sensor_name": "imu_el",
                "app_id": 3,
                "accel_x": float(np.cos(0.01 * i)),
                "accel_y": float(np.sin(0.01 * i)),
                "accel_z": 0.1,
                "el_deg": 0.01 * i,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "standby": False,
            }
            for i in range(ntimes)
        ]
    write_hdf5(path, data, header, metadata=md or None)
    if root_attrs:
        with h5py.File(path, "a") as h5:
            for key, value in root_attrs.items():
                h5.attrs[key] = value
    return path


def stale_pair(tmp_path):
    """
    One good file, then one whose ``sync_time`` is 54 days stale.

    Both names are stamped ~30 s after the last integration they hold,
    the way the writer stamps them; only the second file's header times
    are poisoned, so it is the pair that separates ``time`` from
    ``time_best``.
    """
    write_corr_file(
        tmp_path / "corr_20260717_150041Z.h5",
        ntimes=60,
        sync_time=1784300441.0 - 30,
    )
    write_corr_file(
        tmp_path / "corr_20260717_151041Z.h5",
        ntimes=60,
        sync_time=1784301041.0 - 30 - 54 * 86400,
        seed=1,
    )
    return tmp_path


@pytest.fixture
def corr_dir(tmp_path):
    """Three phase-C files: full ladder, no rfswitch stream, fast cadence.

    All three have a cross key, root attrs from the filter script, and
    filenames within the write backlog of their header times, so every
    row is ``sync_consistent`` and ``time_best == time``.
    """
    root = {"filter_phase": "C", "mux_copy_0to1": True}
    write_corr_file(
        tmp_path / "corr_20260717_150041Z.h5",
        ntimes=60,
        keys=("0", "4", "04"),
        rfswitch=RFSWITCH_LADDER,
        sync_time=1.7843e9,
        root_attrs=root,
    )
    write_corr_file(
        tmp_path / "corr_20260717_151041Z.h5",
        ntimes=60,
        keys=("0", "4", "04"),
        rfswitch=None,
        sync_time=1.7843e9 + 600,
        root_attrs=root,
        seed=1,
    )
    write_corr_file(
        tmp_path / "corr_20260717_152041Z.h5",
        ntimes=60,
        keys=("0", "4", "04"),
        rfswitch=["RFAMB"] * 30 + ["RFNON"] * 30,
        integration_time=0.2684,
        sync_time=1.7843e9 + 1200,
        run_tag="motor_scan",
        root_attrs=root,
        seed=2,
    )
    return tmp_path
