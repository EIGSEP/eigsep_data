"""Tests for eigsep_data.quicklook."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from eigsep_data import quicklook as ql

REAL_FILE = (
    Path(__file__).parent.parent
    / "notebooks/christian/rfsoc_lab_test/noise_floor_10db.h5"
)


def make_synthetic_file(fname, ntimes=20, nchan=256, comb_offset=0):
    """Write a minimal corr h5 file that read_hdf5 understands."""
    rng = np.random.default_rng(0)
    auto = rng.chisquare(100, (ntimes, nchan)) * 1e4
    auto[:, comb_offset::16] *= 1e3  # comb tones
    auto[:, 77] *= 50  # narrowband RFI line
    auto[3] = 0  # dropped integration
    cross = auto * rng.normal(0.1, 0.01, (ntimes, nchan))
    with h5py.File(fname, "w") as f:
        f["data/1"] = auto.astype(np.int64)
        re_im = np.stack([cross, cross * 0.5], axis=-1)
        f["data/13"] = re_im.astype(np.int32)
        hdr = f.create_group("header")
        hdr["freqs"] = np.arange(nchan) * 0.244140625
        hdr["times"] = 1.78e9 + np.arange(ntimes) * 1.07
        hdr.attrs["integration_time"] = 1.07
        hdr.attrs["nchan"] = nchan
    return fname


class TestFindComb:
    def test_detects_comb_offset(self):
        rng = np.random.default_rng(1)
        spec = rng.chisquare(100, 1024)
        spec[5::16] *= 1e4
        offset, excess = ql.find_comb(spec)
        assert offset == 5
        assert excess > 100

    def test_no_comb_in_flat_noise(self):
        rng = np.random.default_rng(2)
        spec = rng.chisquare(100, 1024)
        offset, _ = ql.find_comb(spec)
        assert offset is None


class TestCombFlags:
    def test_flags_tones_and_adjacent(self):
        flags = ql.comb_flags(64, offset=3, pad=1)
        assert flags.shape == (64,)
        assert flags[3] and flags[2] and flags[4]
        assert flags[19] and flags[18] and flags[20]
        assert not flags[10]

    def test_pad_zero_flags_only_tones(self):
        flags = ql.comb_flags(64, offset=0, pad=0)
        assert flags[0] and flags[16]
        assert not flags[1] and not flags[15]


class TestFlagSpectra:
    def test_flags_rfi_line_and_comb(self):
        rng = np.random.default_rng(3)
        wf = rng.chisquare(100, (50, 256)) * 1e4
        wf[:, 0::16] *= 1e3
        wf[:, 77] *= 50
        flags, offset = ql.flag_spectra(wf)
        assert offset == 0
        assert flags[:, 77].all()
        assert flags[:, 16].all() and flags[:, 17].all()
        # bulk of clean channels unflagged
        clean = np.ones(256, bool)
        clean[ql.comb_flags(256, 0)] = False
        clean[77] = False
        assert flags[:, clean].mean() < 0.05

    def test_zero_rows_fully_flagged(self):
        rng = np.random.default_rng(4)
        wf = rng.chisquare(100, (30, 128)) * 1e4
        wf[7] = 0
        flags, _ = ql.flag_spectra(wf)
        assert flags[7].all()
        # the zero row must not poison the other rows
        assert flags[:7].mean() < 0.05

    def test_mostly_dead_input(self):
        # lab-style input: digital zeros except a few live channels
        rng = np.random.default_rng(6)
        wf = np.zeros((40, 256))
        for ch in (10, 50, 90):
            wf[:, ch] = rng.chisquare(100, 40) * 1e4
        wf[5, 50] *= 100  # genuine outlier on a live channel
        flags, offset = ql.flag_spectra(wf)
        assert offset is None
        assert flags[wf == 0].all()  # dead samples flagged
        assert flags[5, 50]  # outlier caught
        assert not flags[:, 10].all()  # live channel not blanket-flagged


class TestWaterfallStats:
    def test_basic_stats(self):
        rng = np.random.default_rng(5)
        wf = rng.chisquare(100, (40, 128)) * 1e4
        wf[[2, 9]] = 0
        flags = np.zeros_like(wf, bool)
        flags[:, 64] = True
        s = ql.waterfall_stats(wf, flags)
        assert s["ntimes"] == 40
        assert s["nchan"] == 128
        assert s["n_zero_rows"] == 2
        assert s["zero_frac"] == pytest.approx(2 / 40)
        assert s["flag_frac"] == pytest.approx(1 / 128)
        assert s["peak_chan"] == int(np.abs(wf).mean(axis=0).argmax())
        assert s["med_power"] > 0


class TestQuicklookFile:
    def test_synthetic_file_end_to_end(self, tmp_path):
        fname = make_synthetic_file(tmp_path / "corr_test.h5")
        res = ql.quicklook(fname)
        assert set(res.pairs) == {"1", "13"}
        assert res.freqs.shape == (256,)
        assert res.times.shape == (20,)
        s = res.stats["1"]
        assert s["n_zero_rows"] == 1
        assert s["comb_offset"] == 0
        # RFI line found on top of the comb
        assert res.flags["1"][:, 77].all()

    def test_plot_writes_png(self, tmp_path):
        fname = make_synthetic_file(tmp_path / "corr_test.h5")
        res = ql.quicklook(fname)
        out = ql.plot_quicklook(res, save=tmp_path / "out.png")
        assert (tmp_path / "out.png").exists()
        assert out is not None

    @pytest.mark.skipif(not REAL_FILE.exists(), reason="lab file absent")
    def test_real_lab_file(self):
        res = ql.quicklook(REAL_FILE)
        assert set(res.pairs) == {"1", "3", "13"}
        assert res.freqs.shape == (1024,)
        for p in res.pairs:
            assert res.stats[p]["ntimes"] == 60


class TestCli:
    def test_main_writes_png_and_returns_zero(self, tmp_path, capsys):
        fname = make_synthetic_file(tmp_path / "corr_test.h5")
        rc = ql.main([str(fname), "-o", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "corr_test_quicklook.png").exists()
        out = capsys.readouterr().out
        assert "corr_test.h5" in out
        assert "flag" in out.lower()

    def test_main_on_directory(self, tmp_path):
        make_synthetic_file(tmp_path / "a.h5")
        make_synthetic_file(tmp_path / "b.h5")
        rc = ql.main([str(tmp_path), "-o", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "a_quicklook.png").exists()
        assert (tmp_path / "b_quicklook.png").exists()
