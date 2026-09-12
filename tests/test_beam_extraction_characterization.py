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
        # alignment shows up here. az_pos/pot_az_angle come straight
        # from write_corr_file's per-sample generators (az_pos = i,
        # pot_az_angle = 10 + i), and el_pos is the fixture's constant
        # 0.0 motor elevation -- pinning the values, not just the
        # shapes, catches an off-by-one in index alignment that a
        # shape-only check would miss.
        out = _extract(tmp_path)
        np.testing.assert_allclose(out["az_pos"][:5], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(out["el_pos"][:5], [0, 0, 0, 0, 0])
        np.testing.assert_allclose(
            out["pot_az_angle"][:5], [10, 11, 12, 13, 14]
        )
        np.testing.assert_allclose(out["times"][0], BEAM_T0)
        assert np.isfinite(out["sky"]).all()
        assert np.isfinite(out["imu_el_deg"]).all()

    def test_sweep_slice_applies_to_every_per_sample_array(self, tmp_path):
        # sweep_slice exists to drop a calibration sweep at the start
        # of a run; it must apply uniformly to every per-sample array
        # and leave freqs (not per-sample) untouched.
        out = _extract(tmp_path, sweep_slice=slice(5, 15))
        assert out["times"].shape == (10,)
        assert out["sky"].shape == (10, 1024)
        assert out["cross"].shape == (10, 1024)
        assert out["freqs"].shape == (1024,)  # never sliced
