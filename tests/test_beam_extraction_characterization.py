# tests/test_beam_extraction_characterization.py
"""Pins extract_beam_mapping_data's current output.

Written against the implementation as it stands, BEFORE the rewrite onto
MetadataIndex. Its job is to fail loudly if the consolidation changes any
value the beam pipeline consumes. If a value here needs to change, that
is a deliberate decision to argue for -- not a test to relax quietly.
"""

import h5py
import numpy as np

from eigsep_data.data import extract_beam_mapping_data

from conftest import write_corr_file

# 1784320080.0 == 2026-07-17 20:28:00Z, matching the first filename so
# sync_consistent is True and the fixture reads coherently.
BEAM_T0 = 1784320080.0
BEAM_FILES = ["corr_20260717_202800Z.h5", "corr_20260717_203000Z.h5"]

# imu_el_deg for one 20-sample file, as extract_beam_mapping_data
# produces it today from this fixture's deterministic accel_x=cos(0.01*i),
# accel_y=sin(0.01*i), accel_z=0.1, el_pos=0.0 (no rng involved -- the
# per-key data arrays are randomised by `seed`, but the motor/imu_el
# streams are not). Recorded by running the unmodified
# extract_beam_mapping_data against this exact fixture once and reading
# off the result; file 2 reproduces the identical pattern because its
# accel/el_pos generators reset their local index at the start of every
# file, the same way az_pos and pot_az_angle do. This is a literal
# constant, not a value computed from the function under test.
_IMU_EL_DEG_GOLDEN_FILE = [
    -5.416005,
    -4.845914,
    -4.275818,
    -3.705717,
    -3.135613,
    -2.565506,
    -1.995396,
    -1.425284,
    -0.855171,
    -0.285057,
    0.285057,
    0.855171,
    1.425284,
    1.995396,
    2.565506,
    3.135613,
    3.705717,
    4.275818,
    4.845914,
    5.416005,
]


def _beam_dir(tmp_path):
    """Two files with motor/potmon/imu_el, as the beam pipeline sees."""
    for i, name in enumerate(BEAM_FILES):
        write_corr_file(
            tmp_path / name,
            ntimes=20,
            sync_time=BEAM_T0 + i * 120,
            acc_cnt0=i * 20,
            keys=("0", "4", "04"),
            streams=("motor", "potmon", "imu_el"),
            rfswitch=["RFANT"] * 20,
            seed=i,
        )
    return tmp_path


def _extract(tmp_path, **kw):
    return extract_beam_mapping_data(
        _beam_dir(tmp_path),
        start_time=BEAM_T0 - 1,
        end_time=BEAM_T0 + 600,
        **kw,
    )


class TestExtractBeamMappingDataContract:
    """Pins the beam pipeline's only entry point into corr files.

    Every value check here is derived from an independent source --
    the fixture's own generator formulas, a raw h5py read, or a golden
    literal recorded once -- never from a second call to the function
    under test, so a bug that touches selection, sorting, per-file
    metadata alignment, or key mapping has somewhere to show up.
    """

    def test_returns_the_documented_keys(self, tmp_path):
        # The beam pipeline (beam_sim, beam_fit, rfi) unpacks this dict
        # by key. Task 9's rewrite onto MetadataIndex must not add,
        # drop, or rename any of them.
        out = _extract(tmp_path)
        assert sorted(out) == sorted(
            [
                "times",
                "freqs",
                "sky",
                "ground",
                "cross",
                "el_pos",
                "az_pos",
                "pot_az_angle",
                "imu_el_deg",
                "imu_accel",
            ]
        )

    def test_shapes_and_ordering(self, tmp_path):
        # Two 20-integration files must concatenate to 40 rows in
        # chronological order across the file boundary, not per-file
        # order or filename order.
        out = _extract(tmp_path)
        assert out["times"].shape == (40,)
        assert out["sky"].shape == (40, 1024)
        assert out["cross"].shape == (40, 1024)
        assert out["imu_accel"].shape == (40, 3)
        # Chronological, across file boundaries.
        assert (np.diff(out["times"]) > 0).all()

    def test_cross_is_complex_from_re_im_pairs(self, tmp_path):
        # Real deployment-5 files store the cross as (n, 1024, 2) int32
        # and read_hdf5 rebuilds complex. Anything that reads raw h5py
        # must do the same, or the beam pipeline gets an int32 cube.
        out = _extract(tmp_path)
        assert out["cross"].dtype.kind == "c"
        raw = []
        for name in BEAM_FILES:
            with h5py.File(tmp_path / name, "r") as h5:
                raw.append(h5["data"]["04"][()])
        raw = np.concatenate(raw)
        np.testing.assert_array_equal(
            out["cross"], raw[..., 0] + 1j * raw[..., 1]
        )

    def test_values_are_stable(self, tmp_path):
        # Golden values: any drift in selection, sorting, or metadata
        # alignment shows up here -- across the file boundary, not
        # only in file 1. write_corr_file's motor/potmon generators
        # reset their per-sample index at the start of every file
        # (az_pos=float(i), pot_az_angle=10.0+i for i in
        # range(ntimes)), so the correct 40-row array is two identical
        # 20-row ramps back to back. A bug isolated to file 2's
        # indices/selected_indices alignment -- exactly the class of
        # bug a multi-file MetadataIndex rewrite is likely to
        # introduce -- breaks this tiling but would not be caught by
        # checking only the first five rows.
        out = _extract(tmp_path)
        expected_az = np.tile(np.arange(20, dtype=float), 2)
        expected_pot = np.tile(10.0 + np.arange(20, dtype=float), 2)
        expected_el = np.zeros(40)
        np.testing.assert_allclose(out["az_pos"], expected_az)
        np.testing.assert_allclose(out["pot_az_angle"], expected_pot)
        np.testing.assert_allclose(out["el_pos"], expected_el)
        np.testing.assert_allclose(out["times"][0], BEAM_T0)

    def test_sky_and_ground_match_raw_data(self, tmp_path):
        # sky/ground map into data_range by key ("4"/"0" here); a
        # swapped mapping or a wrong-key read would still be finite
        # and the right shape, so only a direct comparison against
        # the raw per-key arrays -- not a second call to the function
        # under test -- can catch it. Unlike "04", these single-char
        # keys are not (re, im) pairs, so read_hdf5 leaves them as
        # plain int32 and the raw bytes compare equal outright.
        out = _extract(tmp_path)
        raw_sky, raw_ground = [], []
        for name in BEAM_FILES:
            with h5py.File(tmp_path / name, "r") as h5:
                raw_sky.append(h5["data"]["4"][()])
                raw_ground.append(h5["data"]["0"][()])
        raw_sky = np.concatenate(raw_sky)
        raw_ground = np.concatenate(raw_ground)
        assert out["sky"].shape == raw_sky.shape
        assert out["ground"].shape == raw_ground.shape
        np.testing.assert_array_equal(out["sky"], raw_sky)
        np.testing.assert_array_equal(out["ground"], raw_ground)

    def test_imu_el_deg_matches_recorded_golden_values(self, tmp_path):
        # imu_el_from_accel's SVD-plus-anchoring result is otherwise
        # unpinned by value anywhere in this file (only isfinite), so
        # a rewrite that fed the wrong accel columns, dropped the
        # el_pos anchor, or misaligned per-file indices would go
        # undetected. atol is looser than the recorded precision, to
        # tolerate floating-point differences across platforms/BLAS
        # without losing the ability to catch a materially wrong
        # computation.
        out = _extract(tmp_path)
        expected = np.array(_IMU_EL_DEG_GOLDEN_FILE * 2)
        np.testing.assert_allclose(out["imu_el_deg"], expected, atol=1e-4)

    def test_sweep_slice_applies_to_every_per_sample_array(self, tmp_path):
        # sweep_slice exists to drop a calibration sweep at the start
        # of a run; it must apply uniformly to every per-sample array,
        # select the SAME rows an unsliced call would (not merely the
        # right count), and leave freqs (not per-sample) untouched.
        full = _extract(tmp_path)
        out = _extract(tmp_path, sweep_slice=slice(5, 15))
        assert out["times"].shape == (10,)
        assert out["sky"].shape == (10, 1024)
        assert out["cross"].shape == (10, 1024)
        assert out["freqs"].shape == (1024,)  # never sliced
        np.testing.assert_array_equal(out["times"], full["times"][5:15])
        np.testing.assert_array_equal(out["sky"], full["sky"][5:15])
        np.testing.assert_array_equal(out["cross"], full["cross"][5:15])
