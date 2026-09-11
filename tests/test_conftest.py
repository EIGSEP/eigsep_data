"""The fixture must produce files the real reader accepts."""

import h5py
import numpy as np
from eigsep_observing.io import read_hdf5

from conftest import NCHAN, RFSWITCH_LADDER, write_corr_file


class TestWriteCorrFile:
    def test_roundtrips_through_the_real_reader(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        data, header, meta = read_hdf5(p)
        assert sorted(data) == ["0", "4"]
        assert data["0"].shape == (60, NCHAN)
        assert len(header["times"]) == 60
        assert header["run_tag"] == "panda_observe"
        assert header["nchan"] == NCHAN

    def test_none_survives_the_json_roundtrip(self, tmp_path):
        # A None row means "no reading reached the writer" and is
        # distinct from "UNKNOWN". If the fixture cannot express it,
        # none of the missing-data tests below are meaningful.
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        _, _, meta = read_hdf5(p)
        sw = meta["rfswitch"]
        assert sw[0] == "RFANT"
        assert sw[20] == "UNKNOWN"
        assert sw[45] is None

    def test_times_follow_the_writer_formula(self, tmp_path):
        # times = acc_cnt * integration_time + sync_time. The index
        # recovers sync from this, so the fixture must obey it.
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=30,
            integration_time=0.2684,
            sync_time=1.7843e9,
        )
        _, header, _ = read_hdf5(p)
        expected = header["acc_cnt"] * 0.2684 + 1.7843e9
        np.testing.assert_allclose(header["times"], expected)

    def test_cross_key_roundtrips_as_complex(self, tmp_path):
        # Deployment-5 files store crosses as (re, im) int32 pairs and
        # read_hdf5 rebuilds complex from them. The loader written in
        # Task 8 must apply the same rule, and this fixture must give
        # the characterization test something to pin.
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=6,
            keys=("0", "4", "04"),
        )
        data, _, _ = read_hdf5(p)
        assert data["04"].dtype.kind == "c"
        assert data["04"].shape == (6, NCHAN)
        with h5py.File(p, "r") as h5:
            raw = h5["data"]["04"][()]
        assert raw.shape == (6, NCHAN, 2)
        assert raw.dtype == np.int32
        np.testing.assert_array_equal(
            data["04"], raw[..., 0] + 1j * raw[..., 1]
        )

    def test_root_attrs_are_written(self, tmp_path):
        # filter_corr_keys.py stamps filter_phase and the mux-copy
        # flags at file root, outside the producer's header group.
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=6,
            root_attrs={"filter_phase": "C", "mux_copy_4to5": True},
        )
        with h5py.File(p, "r") as h5:
            assert h5.attrs["filter_phase"] == "C"
            assert bool(h5.attrs["mux_copy_4to5"]) is True
        _, header, _ = read_hdf5(p)
        assert "filter_phase" not in header
