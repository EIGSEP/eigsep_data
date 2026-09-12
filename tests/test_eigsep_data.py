"""EigsepData must carry metadata aligned to its own time axis."""

import warnings

import h5py
import numpy as np
import pytest

from eigsep_data.data import EigsepData
from eigsep_data.index import MetadataIndex
from eigsep_data.metadata import MISSING

from conftest import NCHAN, write_corr_file


class TestFromSelection:
    def test_reads_only_the_selected_rows(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert d.data["0"].shape == (20, NCHAN)
        assert d.times.shape == (20,)

    def test_meta_is_aligned_row_for_row(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFAMB", "RFNON"]
        )
        d = EigsepData.from_selection(sel, keys=["0"])
        assert len(d.meta) == len(d.times)
        np.testing.assert_allclose(d.meta.time_best.to_numpy(), d.times)
        assert set(d.meta.rfswitch) == {"RFAMB", "RFNON"}

    def test_reads_only_requested_keys(self, corr_dir):
        # Reading every key is what OOMed the 16 GB laptop; the loader
        # must never pull keys the caller did not ask for.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert list(d.data) == ["0"]

    def test_native_dtypes_at_full_resolution(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert d.data["0"].dtype == np.int32

    def test_selected_rows_are_the_ones_on_disk(self, corr_dir):
        # The whole point of the index: row 20 of the selection is row
        # 20 of the file, not row 20 of some resorted concatenation.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFNOFF")
        d = EigsepData.from_selection(sel, keys=["0"])
        with h5py.File(corr_dir / "corr_20260717_150041Z.h5", "r") as h5:
            raw = h5["data"]["0"][()]
        np.testing.assert_array_equal(d.data["0"], raw[d.meta.row.to_numpy()])

    def test_times_are_not_shifted(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        assert abs(d.times[0] - 1.7843e9) < 1.0

    def test_freq_comes_from_the_header(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["0"])
        with h5py.File(corr_dir / "corr_20260717_150041Z.h5", "r") as h5:
            np.testing.assert_array_equal(d.freq, h5["header"]["freqs"][()])

    def test_cross_is_complex(self, corr_dir):
        # The reader's rule: (n, nchan, 2) int32 is (re, im).
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys=["04"])
        assert d.data["04"].dtype.kind == "c"
        assert d.data["04"].shape == (20, NCHAN)
        with h5py.File(corr_dir / "corr_20260717_150041Z.h5", "r") as h5:
            raw = h5["data"]["04"][:20]
        np.testing.assert_array_equal(
            d.data["04"], raw[..., 0] + 1j * raw[..., 1]
        )

    def test_a_bare_key_string_is_one_key(self, corr_dir):
        # list("04") is ["0", "4"]: a bare string must not be read as
        # an iterable of characters, or asking for the cross silently
        # hands back the two autos.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        d = EigsepData.from_selection(sel, keys="04")
        assert list(d.data) == ["04"]
        assert d.data["04"].dtype.kind == "c"

    def test_a_repeated_key_is_read_once(self, tmp_path):
        # blocks is keyed by name, so a duplicate used to append a
        # second block per file while the row positions stayed one per
        # row: correct shape, right dtype, one file's rows served in
        # another file's place.
        names = ["corr_20260717_150041Z.h5", "corr_20260717_151041Z.h5"]
        write_corr_file(tmp_path / names[0], ntimes=6, keys=("0",))
        write_corr_file(
            tmp_path / names[1],
            ntimes=6,
            keys=("0",),
            sync_time=1.7843e9 + 600,
            seed=1,
        )
        sel = MetadataIndex(tmp_path, cache=False).select()
        d = EigsepData.from_selection(sel, keys=["0", "0"])
        assert list(d.data) == ["0"]
        assert d.data["0"].shape == (12, NCHAN)
        raw = {}
        for name in names:
            with h5py.File(tmp_path / name, "r") as h5:
                raw[name] = h5["data"]["0"][()]
        for i, (name, row) in enumerate(zip(d.meta.file, d.meta.row)):
            np.testing.assert_array_equal(d.data["0"][i], raw[name][row])

    def test_empty_selection_raises_clearly(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="VNAO")
        with pytest.raises(ValueError, match="no integrations"):
            EigsepData.from_selection(sel, keys=["0"])

    def test_rejects_a_bad_missing_policy(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with pytest.raises(ValueError, match="missing must be"):
            EigsepData.from_selection(sel, keys=["0"], missing="skip")

    def test_rejects_a_bad_time_avg(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with pytest.raises(ValueError, match="time_avg"):
            EigsepData.from_selection(sel, keys=["0"], time_avg=0)

    def test_selection_load_is_equivalent(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        viaload = sel.load(keys=["0", "04"], time_avg=4)
        direct = EigsepData.from_selection(sel, keys=["0", "04"], time_avg=4)
        np.testing.assert_allclose(viaload.times, direct.times)
        np.testing.assert_allclose(viaload.acc_cnt, direct.acc_cnt)
        for key in ("0", "04"):
            np.testing.assert_array_equal(viaload.data[key], direct.data[key])
        assert viaload.meta.equals(direct.meta)

    def test_slice_carries_meta(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        full = EigsepData.from_selection(sel, keys=["0"])
        d = full.slice(0, 5)
        assert len(d.meta) == 5
        assert d.times.shape == (5,)
        assert d.meta.row.tolist() == full.meta.row.tolist()[:5]

    def test_slice_without_meta_still_works(self, corr_dir):
        # from_path-built instances have meta=None until Task 10.
        d = EigsepData(
            data={"0": np.zeros((4, 8))},
            acc_cnt=np.arange(4.0),
            times=np.arange(4.0),
        ).slice(1, 3)
        assert d.meta is None
        assert d.times.tolist() == [1.0, 2.0]


class TestRowOrder:
    def test_interleaved_files_come_back_in_selection_order(self, tmp_path):
        # Two files whose integrations alternate in time. Concatenating
        # per file and then reordering with a permutation computed on
        # the other axis would scramble spectra against times; this
        # pins the inverse permutation.
        t0 = 1784300441.0 - 20
        names = ["corr_20260717_150041Z.h5", "corr_20260717_150042Z.h5"]
        for i, name in enumerate(names):
            write_corr_file(
                tmp_path / name,
                ntimes=20,
                sync_time=t0 + 0.25 * i,
                rfswitch=["RFANT"] * 20,
                seed=i,
            )
        sel = MetadataIndex(tmp_path, cache=False).select()
        assert len(set(sel.meta.file.iloc[:2])) == 2  # they interleave
        d = EigsepData.from_selection(sel, keys=["0"])
        assert (np.diff(d.times) > 0).all()
        raw = {}
        for name in names:
            with h5py.File(tmp_path / name, "r") as h5:
                raw[name] = h5["data"]["0"][()]
        for i, (name, row) in enumerate(zip(d.meta.file, d.meta.row)):
            np.testing.assert_array_equal(d.data["0"][i], raw[name][row])

    def test_interleaved_files_stay_paired_when_averaged(self, tmp_path):
        # Blocks interleave too, so the same permutation has to place
        # averaged blocks against the file they were averaged from.
        t0 = 1784300441.0 - 20
        names = ["corr_20260717_150041Z.h5", "corr_20260717_150042Z.h5"]
        for i, name in enumerate(names):
            write_corr_file(
                tmp_path / name, ntimes=20, sync_time=t0 + 0.25 * i, seed=i
            )
        sel = MetadataIndex(tmp_path, cache=False).select()
        d = EigsepData.from_selection(sel, keys=["0"], time_avg=4)
        assert len(d.meta) == 10
        assert len(set(d.meta.file.iloc[:2])) == 2
        raw = {}
        for name in names:
            with h5py.File(tmp_path / name, "r") as h5:
                raw[name] = h5["data"]["0"][()]
        for i, (name, row) in enumerate(zip(d.meta.file, d.meta.row)):
            np.testing.assert_allclose(
                d.data["0"][i],
                raw[name][row : row + 4].mean(axis=0),
                rtol=1e-6,
            )


def _mixed_keys_dir(tmp_path):
    """Phase change inside a window: the second file has no cross."""
    write_corr_file(
        tmp_path / "corr_20260717_150041Z.h5",
        ntimes=10,
        keys=("0", "4", "04"),
    )
    write_corr_file(
        tmp_path / "corr_20260717_151041Z.h5",
        ntimes=10,
        keys=("0", "4"),
        sync_time=1.7843e9 + 600,
        seed=1,
    )
    return tmp_path


class TestMissingKeys:
    def test_raise_names_the_files(self, tmp_path):
        sel = MetadataIndex(_mixed_keys_dir(tmp_path), cache=False).select()
        with pytest.raises(KeyError, match="151041Z") as info:
            EigsepData.from_selection(sel, keys=["04"])
        assert "data_keys" in str(info.value)

    def test_raise_names_every_offending_file(self, tmp_path):
        # The pre-pass reports the whole selection, not the first file
        # it trips over.
        _mixed_keys_dir(tmp_path)
        write_corr_file(
            tmp_path / "corr_20260717_152041Z.h5",
            ntimes=10,
            keys=("0", "4"),
            sync_time=1.7843e9 + 1200,
            seed=2,
        )
        sel = MetadataIndex(tmp_path, cache=False).select()
        with pytest.raises(KeyError) as info:
            EigsepData.from_selection(sel, keys=["04"])
        assert "151041Z" in str(info.value)
        assert "152041Z" in str(info.value)

    def test_raise_happens_before_any_spectra_are_read(
        self, tmp_path, monkeypatch
    ):
        # A straddled phase boundary is routine, so it must not cost the
        # I/O and peak memory of the whole selection first.
        sel = MetadataIndex(_mixed_keys_dir(tmp_path), cache=False).select()

        def _no_reads(*args, **kwargs):
            raise AssertionError("opened a file before raising")

        monkeypatch.setattr("eigsep_data.data.h5py.File", _no_reads)
        with pytest.raises(KeyError, match="151041Z"):
            EigsepData.from_selection(sel, keys=["04"])

    def test_a_stale_index_still_raises_from_the_file(self, tmp_path):
        # The pre-pass believes meta.data_keys; when the file has
        # changed since the scan, the per-file check is the backstop.
        path = tmp_path / "corr_20260717_150041Z.h5"
        write_corr_file(path, ntimes=6, keys=("0", "4"))
        sel = MetadataIndex(tmp_path, cache=False).select()
        with h5py.File(path, "a") as h5:
            del h5["data"]["4"]
        with pytest.raises(KeyError, match="150041Z"):
            EigsepData.from_selection(sel, keys=["4"])

    def test_nan_fills_that_files_rows(self, tmp_path):
        sel = MetadataIndex(_mixed_keys_dir(tmp_path), cache=False).select()
        d = EigsepData.from_selection(sel, keys=["04"], missing="nan")
        assert d.data["04"].shape == (20, NCHAN)
        assert d.data["04"].dtype.kind == "c"
        gap = (d.meta.file == "corr_20260717_151041Z.h5").to_numpy()
        assert np.isnan(d.data["04"][gap]).all()
        assert np.isfinite(d.data["04"][~gap]).all()

    def test_nan_fills_an_auto_too(self, tmp_path):
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5", ntimes=6, keys=("0", "4")
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=6,
            keys=("0",),
            sync_time=1.7843e9 + 600,
            seed=1,
        )
        sel = MetadataIndex(tmp_path, cache=False).select()
        d = EigsepData.from_selection(sel, keys=["4"], missing="nan")
        # int32 cannot hold NaN, so the whole key upcasts to float.
        assert d.data["4"].dtype == np.float64
        gap = (d.meta.file == "corr_20260717_151041Z.h5").to_numpy()
        assert np.isnan(d.data["4"][gap]).all()
        assert np.isfinite(d.data["4"][~gap]).all()

    def test_file_with_no_data_group_is_just_a_missing_key(self, tmp_path):
        # scan_corr_file indexes a file that has header/times but no
        # data group at all, so every key is absent from it and the
        # missing= policy has to cover it -- not a raw h5py KeyError.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5", ntimes=6, keys=("0",)
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=6,
            keys=("0",),
            sync_time=1.7843e9 + 600,
            seed=1,
        )
        with h5py.File(tmp_path / "corr_20260717_151041Z.h5", "a") as h5:
            del h5["data"]
        sel = MetadataIndex(tmp_path, cache=False).select()
        with pytest.raises(KeyError, match="151041Z"):
            EigsepData.from_selection(sel, keys=["0"])
        d = EigsepData.from_selection(sel, keys=["0"], missing="nan")
        assert d.data["0"].shape == (12, NCHAN)
        assert np.isnan(d.data["0"][-6:]).all()
        assert np.isfinite(d.data["0"][:6]).all()
        # ... and with keys=None too: declaring nothing must not drag
        # the intersection to empty and turn this into "no shared keys".
        with pytest.raises(KeyError, match="151041Z"):
            EigsepData.from_selection(sel)
        d = EigsepData.from_selection(sel, missing="nan")
        assert list(d.data) == ["0"]
        assert np.isnan(d.data["0"][-6:]).all()

    def test_no_file_records_any_keys(self, tmp_path):
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5", ntimes=4, keys=("0",)
        )
        with h5py.File(tmp_path / "corr_20260717_150041Z.h5", "a") as h5:
            del h5["data"]
        sel = MetadataIndex(tmp_path, cache=False).select()
        with pytest.raises(ValueError, match="nothing to read"):
            EigsepData.from_selection(sel)

    def test_nan_fill_survives_time_avg(self, tmp_path):
        sel = MetadataIndex(_mixed_keys_dir(tmp_path), cache=False).select()
        d = EigsepData.from_selection(
            sel, keys=["0", "04"], time_avg=5, missing="nan"
        )
        assert d.data["04"].shape == (4, NCHAN)
        assert d.data["04"].dtype == np.complex64
        gap = (d.meta.file == "corr_20260717_151041Z.h5").to_numpy()
        assert np.isnan(d.data["04"][gap]).all()
        assert np.isfinite(d.data["04"][~gap]).all()
        assert np.isfinite(d.data["0"]).all()

    def test_default_keys_are_the_common_ones(self, tmp_path):
        sel = MetadataIndex(_mixed_keys_dir(tmp_path), cache=False).select()
        d = EigsepData.from_selection(sel)
        assert sorted(d.data) == ["0", "4"]

    def test_disjoint_keys_say_so(self, tmp_path):
        # Nothing to intersect: an empty data dict would look like a
        # successful read of nothing.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5", ntimes=4, keys=("0",)
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=4,
            keys=("4",),
            sync_time=1.7843e9 + 600,
            seed=1,
        )
        sel = MetadataIndex(tmp_path, cache=False).select()
        with pytest.raises(ValueError, match="share no data keys"):
            EigsepData.from_selection(sel)


class TestTimeAvg:
    def test_blocks_within_a_file(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with pytest.warns(UserWarning, match="dropped 4"):
            d = EigsepData.from_selection(sel, keys=["0", "04"], time_avg=8)
        assert d.data["0"].shape == (2, NCHAN)
        assert d.data["0"].dtype == np.float32
        assert d.data["04"].dtype == np.complex64
        assert len(d.meta) == 2
        assert list(d.meta.row) == [0, 8]
        np.testing.assert_allclose(d.times[0], 1.7843e9 + 3.5 * 0.5369)
        np.testing.assert_allclose(d.acc_cnt, [3.5, 11.5])
        # The spec's one sanctioned exception to "meta.time is the raw
        # header time": an explicitly requested block mean.
        np.testing.assert_allclose(d.meta.time.to_numpy(), d.times)
        np.testing.assert_allclose(d.meta.acc_cnt.to_numpy(), [3.5, 11.5])

    def test_block_mean_matches_numpy(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        full = EigsepData.from_selection(sel, keys=["0"])
        avg = EigsepData.from_selection(sel, keys=["0"], time_avg=4)
        expected = full.data["0"].reshape(5, 4, NCHAN).mean(axis=1)
        np.testing.assert_allclose(avg.data["0"], expected, rtol=1e-6)

    def test_cross_block_mean_matches_numpy(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        full = EigsepData.from_selection(sel, keys=["04"])
        avg = EigsepData.from_selection(sel, keys=["04"], time_avg=4)
        expected = full.data["04"].reshape(5, 4, NCHAN).mean(axis=1)
        np.testing.assert_allclose(avg.data["04"], expected, rtol=1e-6)

    def test_never_averages_across_files(self, corr_dir):
        # 20 RFANT rows in one file, 30 RFAMB in another: 2 + 3 blocks
        # of 8, never a block straddling the boundary.
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFANT", "RFAMB"]
        )
        with pytest.warns(UserWarning, match="dropped 10"):
            d = EigsepData.from_selection(sel, keys=["0"], time_avg=8)
        assert d.data["0"].shape == (5, NCHAN)
        assert d.meta.file.tolist() == (
            ["corr_20260717_150041Z.h5"] * 2 + ["corr_20260717_152041Z.h5"] * 3
        )

    def test_block_that_no_file_can_fill_says_so(self, corr_dir):
        # 20 rows, blocks of 64: every row is a remainder, so there is
        # nothing to return and the error names the culprit.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with pytest.raises(ValueError, match="time_avg=64"):
            EigsepData.from_selection(sel, keys=["0"], time_avg=64)


def _narrow_file(path, nchan=512, declare=False):
    """Rewrite one file with *nchan* channels; *declare* keeps the
    ``nchan`` header attr, otherwise it is removed."""
    with h5py.File(path, "a") as h5:
        for key in list(h5["data"]):
            narrow = h5["data"][key][()][:, :nchan]
            del h5["data"][key]
            h5["data"][key] = narrow
        freqs = h5["header"]["freqs"][()][:nchan]
        del h5["header"]["freqs"]
        h5["header"]["freqs"] = freqs
        if declare:
            h5["header"].attrs["nchan"] = nchan
        else:
            del h5["header"].attrs["nchan"]


def _narrow_dir(tmp_path, nchan=512, declare=False):
    """
    A 512-channel pair whose *first* file has no cross key.

    With ``declare=False`` neither file carries an ``nchan`` header attr
    -- the shape a deployment predating it has -- so ``meta.nchan`` is
    all-NaN and any width taken from the metadata is a 1024-channel
    guess; the only honest source is then the real block read from the
    file that does carry the key.
    """
    names = ["corr_20260717_150041Z.h5", "corr_20260717_151041Z.h5"]
    write_corr_file(tmp_path / names[0], ntimes=10, keys=("0", "4"))
    write_corr_file(
        tmp_path / names[1],
        ntimes=10,
        keys=("0", "4", "04"),
        sync_time=1.7843e9 + 600,
        seed=1,
    )
    for name in names:
        _narrow_file(tmp_path / name, nchan=nchan, declare=declare)
    return tmp_path


class TestChannelWidth:
    def test_nan_width_comes_from_the_data(self, tmp_path):
        root = _narrow_dir(tmp_path)
        sel = MetadataIndex(root, cache=False).select()
        assert sel.meta.nchan.isna().all()  # nothing to read it from
        d = EigsepData.from_selection(sel, keys=["04"], missing="nan")
        assert d.freq.size == 512
        assert d.data["04"].shape == (20, 512)
        gap = (d.meta.file == "corr_20260717_150041Z.h5").to_numpy()
        assert np.isnan(d.data["04"][gap]).all()
        assert np.isfinite(d.data["04"][~gap]).all()

    def test_key_no_file_carries_falls_back_to_the_header(self, tmp_path):
        # The one path where the metadata's nchan sizes a returned
        # array: no file has the key, so no real width was ever seen.
        sel = MetadataIndex(
            _narrow_dir(tmp_path, declare=True), cache=False
        ).select()
        assert (sel.meta.nchan == 512).all()
        d = EigsepData.from_selection(sel, keys=["5"], missing="nan")
        assert d.data["5"].shape == (20, 512)  # not the 1024 default
        assert np.isnan(d.data["5"]).all()

    def test_files_that_disagree_on_nchan_say_so(self, tmp_path):
        # np.concatenate would raise a bare dimension mismatch naming
        # neither the key nor the files.
        names = ["corr_20260717_150041Z.h5", "corr_20260717_151041Z.h5"]
        write_corr_file(tmp_path / names[0], ntimes=6, keys=("0",))
        write_corr_file(
            tmp_path / names[1],
            ntimes=6,
            keys=("0",),
            sync_time=1.7843e9 + 600,
            seed=1,
        )
        _narrow_file(tmp_path / names[0], declare=True)
        sel = MetadataIndex(tmp_path, cache=False).select()
        with pytest.raises(ValueError, match="different channel count") as e:
            EigsepData.from_selection(sel, keys=["0"])
        assert names[0] in str(e.value)
        assert "512" in str(e.value) and str(NCHAN) in str(e.value)


def _gappy_attr_dir(tmp_path):
    """Only the first file carries the boolean root attr."""
    write_corr_file(
        tmp_path / "corr_20260717_150041Z.h5",
        ntimes=10,
        root_attrs={"mux_copy_0to1": True},
    )
    write_corr_file(
        tmp_path / "corr_20260717_151041Z.h5",
        ntimes=10,
        sync_time=1.7843e9 + 600,
        seed=1,
    )
    return tmp_path


class TestMetaColumnContract:
    def test_gappy_boolean_column_keeps_real_booleans(self, tmp_path):
        # A bool attr written for only some files is an object column
        # holding real True and the string MISSING. Stringifying it on
        # the way through would make meta.mux_copy_0to1 == True match
        # nothing -- an empty answer indistinguishable from an honest
        # one.
        sel = MetadataIndex(_gappy_attr_dir(tmp_path), cache=False).select()
        d = EigsepData.from_selection(sel, keys=["0"])
        col = d.meta.mux_copy_0to1
        assert col.dtype == object
        assert col.eq(True).sum() == 10
        assert col.eq(MISSING).sum() == 10

    def test_averaging_keeps_the_blocks_first_row_verbatim(self, tmp_path):
        sel = MetadataIndex(_gappy_attr_dir(tmp_path), cache=False).select()
        d = EigsepData.from_selection(sel, keys=["0"], time_avg=5)
        col = d.meta.mux_copy_0to1
        assert col.dtype == object
        assert col.eq(True).sum() == 2
        assert col.eq(MISSING).sum() == 2
        assert d.meta.row.tolist() == [0, 5, 0, 5]


class TestMemoryGuard:
    def test_warns_when_load_exceeds_half_of_available(
        self, corr_dir, monkeypatch
    ):
        monkeypatch.setattr("eigsep_data.data._mem_available", lambda: 1000)
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with pytest.warns(UserWarning, match="available"):
            EigsepData.from_selection(sel, keys=["0"])

    def test_silent_without_meminfo(self, corr_dir, monkeypatch):
        monkeypatch.setattr("eigsep_data.data._mem_available", lambda: None)
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            EigsepData.from_selection(sel, keys=["0"])

    def test_averaging_lowers_the_estimate(self):
        from eigsep_data.data import _estimate_bytes

        # 4 B per auto sample, 16 B per cross, then 4/8 averaged: the
        # numbers the guard's advice ("pass time_avg") rests on.
        assert _estimate_bytes(100, 1024, ["0", "04"], 1) == 100 * 1024 * 20
        assert _estimate_bytes(100, 1024, ["0", "04"], 4) == 25 * 1024 * 12
