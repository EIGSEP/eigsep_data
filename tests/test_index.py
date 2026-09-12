"""Tests for eigsep_data.index."""

import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

from eigsep_data import clock
from eigsep_data.index import CACHE_NAME, MetadataIndex, scan_corr_file
from eigsep_data.metadata import MISSING

from conftest import corrupt_stream, stale_pair, write_corr_file

IDENTITY = (
    "file",
    "row",
    "time",
    "acc_cnt",
    "sync_recovered",
    "sync_consistent",
    "time_fname",
    "time_best",
)


class TestMetadataIndexScan:
    def test_one_row_per_integration(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        assert len(idx.table) == 180  # 3 files x 60

    def test_identity_and_header_columns_present(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        for col in IDENTITY + (
            "integration_time",
            "run_tag",
            "adc_mux_sel",
            "nchan",
            "data_keys",
        ):
            assert col in idx.table.columns

    def test_row_identity_is_file_and_row_not_position(self, corr_dir):
        # Positions shift when the file set changes; annotations and
        # selections pinned to them would rot invisibly.
        idx = MetadataIndex(corr_dir, cache=False)
        first = idx.table[idx.table.file == "corr_20260717_150041Z.h5"]
        assert list(first.row) == list(range(60))

    def test_integration_time_varies_per_file(self, corr_dir):
        # Deployment 5 carries both 0.2684 s and 0.5369 s files, and the
        # writer's transition guard width depends on it. Anything that
        # assumes one value for a deployment is wrong.
        idx = MetadataIndex(corr_dir, cache=False)
        assert set(np.round(idx.table.integration_time, 4)) == {
            0.5369,
            0.2684,
        }

    def test_absent_stream_reads_as_missing(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        sub = idx.table[idx.table.file == "corr_20260717_151041Z.h5"]
        assert (sub.rfswitch == MISSING).all()

    def test_stream_ok_is_a_real_boolean_under_streams_all(self, corr_dir):
        # streams="all" enumerates the streams *that file* carries, so
        # the middle fixture file -- which has no rfswitch stream --
        # contributes no rfswitch_ok column at all and concatenation
        # leaves a gap. Filled with the MISSING sentinel the column
        # would stop answering the question it exists for.
        idx = MetadataIndex(corr_dir, cache=False, streams="all")
        flags = [c for c in idx.table.columns if c.endswith("_ok")]
        assert flags
        for col in flags:
            assert idx.table[col].dtype == bool, col
        # 60 rows with no stream at all, plus the 5 None-padded rows of
        # the ladder in the first file.
        assert idx.select(rfswitch_ok=False).nrows == 65

    def test_streams_all_still_finds_the_file_that_lacks_a_stream(
        self, tmp_path
    ):
        # The query the flag exists for: which rows have no potmon
        # reading. It must reach the file that has no potmon stream.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=6,
            streams=("motor", "potmon"),
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=6,
            streams=("motor",),
            sync_time=1.7843e9 + 600,
            seed=1,
        )
        sel = MetadataIndex(tmp_path, cache=False, streams="all").select(
            potmon_ok=False
        )
        assert sel.files == ["corr_20260717_151041Z.h5"]
        assert sel.nrows == 6

    def test_a_string_attr_named_like_a_flag_is_not_coerced(self, tmp_path):
        # The gaps of a <stream>_ok column are collapsed to False.
        # Root attrs share that namespace, and collapsing a string
        # column the same way would destroy it.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=4,
            root_attrs={"cal_ok": "pending"},
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=4,
            sync_time=1.7843e9 + 600,
            seed=1,
        )
        col = MetadataIndex(tmp_path, cache=False).table.cal_ok
        assert set(col) == {"pending", MISSING}

    def test_sync_recovered_matches_the_writer_formula(self, corr_dir):
        # sync = times - acc_cnt * integration_time. This is the direct
        # bad-clock diagnostic and must be exact.
        idx = MetadataIndex(corr_dir, cache=False)
        sub = idx.table[idx.table.file == "corr_20260717_150041Z.h5"]
        np.testing.assert_allclose(sub.sync_recovered, 1.7843e9, atol=1e-3)

    def test_data_keys_are_recorded(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        assert set(idx.table.data_keys) == {"0,04,4"}

    def test_root_attrs_become_columns(self, corr_dir):
        # filter_corr_keys.py already asserts the wiring phase and the
        # mux-copy flags per file. Indexing them makes
        # select(filter_phase="C") work before any annotation layer.
        idx = MetadataIndex(corr_dir, cache=False)
        assert set(idx.table.filter_phase) == {"C"}
        assert idx.table.mux_copy_0to1.dtype == bool
        assert idx.table.mux_copy_0to1.all()

    def test_file_without_root_attrs_reads_missing(self, tmp_path):
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=4,
            root_attrs={"filter_phase": "C"},
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=4,
            sync_time=1.7843e9 + 600,
        )
        idx = MetadataIndex(tmp_path, cache=False)
        sub = idx.table[idx.table.file == "corr_20260717_151041Z.h5"]
        assert (sub.filter_phase == MISSING).all()

    def test_header_attr_missing_falls_back_to_default(self, tmp_path):
        # run_tag is absent on the earliest deployment-5 files; nchan
        # and adc_mux_sel on some hand-written ones. A missing attr is
        # a gap, not a reason to drop the column or the file.
        write_corr_file(tmp_path / "corr_20260717_150041Z.h5", ntimes=4)
        with h5py.File(tmp_path / "corr_20260717_150041Z.h5", "a") as h5:
            del h5["header"].attrs["run_tag"]
            del h5["header"].attrs["nchan"]
        idx = MetadataIndex(tmp_path, cache=False)
        assert (idx.table.run_tag == MISSING).all()
        assert idx.table.nchan.isna().all()

    def test_file_without_metadata_group_still_indexes(self, tmp_path):
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5", ntimes=4, streams=()
        )
        idx = MetadataIndex(tmp_path, cache=False)
        assert len(idx.table) == 4
        assert (idx.table.rfswitch == MISSING).all()
        assert not idx.table.motor_ok.any()

    def test_boolean_root_attr_survives_a_file_that_lacks_it(self, tmp_path):
        # filter_corr_keys.py writes mux_copy_* per file, so a directory
        # mixing filtered and unfiltered files gives a bool column with
        # gaps. Stringifying it would make select(mux_copy_0to1=True)
        # return nothing and look like an honest empty result.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=4,
            root_attrs={"mux_copy_0to1": True},
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=4,
            sync_time=1.7843e9 + 600,
        )
        col = MetadataIndex(tmp_path, cache=False).table.mux_copy_0to1
        assert col.eq(True).sum() == 4
        assert col.eq(MISSING).sum() == 4

    def test_array_valued_root_attr_does_not_cost_the_file(self, tmp_path):
        # np.repeat on a non-scalar mis-lengths the column, and the
        # DataFrame it breaks would take every integration in the file
        # down with it.
        path = write_corr_file(tmp_path / "corr_20260717_150041Z.h5", ntimes=4)
        with h5py.File(path, "a") as h5:
            h5.attrs["filtered_keys"] = ["0", "4"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            idx = MetadataIndex(tmp_path, cache=False)
        assert len(idx.table) == 4
        assert idx.table.filtered_keys.map(type).eq(str).all()

    def test_string_columns_never_hold_none_or_nan(self, corr_dir):
        # The cache encodes object columns as bytes; None or NaN would
        # come back as the strings "None"/"nan". The table must already
        # be free of them.
        idx = MetadataIndex(corr_dir, cache=False)
        for col in idx.table.columns:
            if idx.table[col].dtype == object:
                assert idx.table[col].map(type).eq(str).all(), col

    def test_cache_file_and_dotfiles_are_not_scanned(self, corr_dir):
        # pathlib's glob matches dotfiles, so "*.h5" would return the
        # cache itself -- which then poisons its own fingerprint.
        (corr_dir / CACHE_NAME).write_bytes(b"not an hdf5 file")
        (corr_dir / ".hidden.h5").write_bytes(b"editor dropping")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            idx = MetadataIndex(corr_dir, patterns=("*.h5",), cache=False)
        assert len(idx.table) == 180

    def test_default_pattern_is_corr_files_only(self, corr_dir):
        # Sidecars such as motor_scan_20260717_key4.h5 live next to
        # data in some directories; they are not integrations.
        write_corr_file(corr_dir / "sidecar_20260717_150041Z.h5", ntimes=4)
        idx = MetadataIndex(corr_dir, cache=False)
        assert "sidecar_20260717_150041Z.h5" not in set(idx.table.file)

    def test_unreadable_file_is_warned_about_and_skipped(self, corr_dir):
        (corr_dir / "corr_20260717_153041Z.h5").write_bytes(b"truncated")
        with pytest.warns(UserWarning, match="corr_20260717_153041Z.h5"):
            idx = MetadataIndex(corr_dir, cache=False)
        assert len(idx.table) == 180

    def test_skipped_files_are_recorded(self, corr_dir):
        # A warning scrolls past; the table just comes back shorter.
        # Without this list a caller cannot tell a partial index from a
        # complete one without re-globbing and re-applying the
        # exclusion rules.
        (corr_dir / "corr_20260717_153041Z.h5").write_bytes(b"truncated")
        with pytest.warns(UserWarning):
            idx = MetadataIndex(corr_dir, cache=False)
        assert [name for name, _ in idx.skipped] == [
            "corr_20260717_153041Z.h5"
        ]
        assert idx.skipped[0][1]  # a reason, not an empty string

    def test_skipped_is_empty_on_a_clean_directory(self, corr_dir):
        assert MetadataIndex(corr_dir, cache=False).skipped == []

    def test_a_type_error_skips_one_file_not_the_whole_scan(self, corr_dir):
        # float() on a header attr that is not a number raises
        # TypeError, and a custom scanner may raise it for reasons of
        # its own. Either way the stated policy is skip-and-continue.
        bad = "corr_20260717_151041Z.h5"

        def scanner(path, streams=None, filename_tz=None):
            if path.name == bad:
                raise TypeError("float() argument must be a number")
            return scan_corr_file(path, streams, filename_tz)

        with pytest.warns(UserWarning, match=bad):
            idx = MetadataIndex(corr_dir, cache=False, scanner=scanner)
        assert len(idx.table) == 120
        assert [name for name, _ in idx.skipped] == [bad]

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MetadataIndex(tmp_path, cache=False)

    def test_scanner_hook_is_called(self, corr_dir):
        # The seam for indexing another file kind (S11 sweeps) without
        # a second cache/select/summary implementation.
        seen = []

        def scanner(path, streams=None, filename_tz=None):
            seen.append(path.name)
            return scan_corr_file(path, streams, filename_tz).head(1)

        idx = MetadataIndex(corr_dir, cache=False, scanner=scanner)
        assert len(seen) == 3
        assert len(idx.table) == 3


class TestMalformedMetadataStream:
    """A stream that cannot be parsed is lost, and must not be silent.

    Every column of that stream then reads ``MISSING`` -- exactly what
    a file whose sensor never published looks like -- so nothing in the
    table distinguishes the two. The file itself is still indexed, so
    ``skipped`` stays empty too.
    """

    def bad_file(self, tmp_path):
        path = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=4,
            streams=("motor", "potmon"),
        )
        return corrupt_stream(path, "potmon")

    def test_the_scan_warns(self, tmp_path):
        path = self.bad_file(tmp_path)
        with pytest.warns(UserWarning, match="potmon"):
            scan_corr_file(path)

    def test_the_index_records_the_lost_stream(self, tmp_path):
        path = self.bad_file(tmp_path)
        with pytest.warns(UserWarning, match="potmon"):
            idx = MetadataIndex(tmp_path, cache=False)
        assert [(f, s) for f, s, _ in idx.lost_streams] == [
            (path.name, "potmon")
        ]
        assert idx.lost_streams[0][2]  # a reason, not an empty string

    def test_the_file_is_indexed_not_skipped(self, tmp_path):
        self.bad_file(tmp_path)
        with pytest.warns(UserWarning):
            idx = MetadataIndex(tmp_path, cache=False)
        assert idx.skipped == []
        assert len(idx.table) == 4
        # The rows the record is about: unreadable, and indistinguishable
        # in the table from a potmon that never published.
        assert idx.table.potmon_pot_az_angle.isna().all()
        assert not idx.table.potmon_ok.any()
        # A stream that parsed is untouched by its neighbour's failure.
        assert idx.table.motor_ok.all()

    def test_a_clean_directory_loses_nothing(self, corr_dir):
        assert MetadataIndex(corr_dir, cache=False).lost_streams == []


class TestSyncConsistent:
    def test_flags_a_stale_sync_time(self, tmp_path):
        # A stale sync_time poisons every time in the file while the
        # filename, stamped at write time, stays correct. Deployment 5
        # has 12.4% such files, off by ~54 days.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=10,
            sync_time=1.7843e9 - 54 * 86400,
        )
        idx = MetadataIndex(tmp_path, cache=False)
        assert not idx.table.sync_consistent.any()

    def test_accepts_the_normal_write_backlog(self, tmp_path):
        # Good deployment-5 files land within [-1, +945] s of their
        # filename; the writer backlogs by up to ~16 min. That must not
        # be flagged.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=10,
            sync_time=1784300441.0 - 900,
        )
        idx = MetadataIndex(tmp_path, cache=False)
        assert idx.table.sync_consistent.all()

    def test_pacific_named_deployment4_file_is_consistent(self, tmp_path):
        # Deployment 1-4 filenames are Pacific wall clock with no Z
        # suffix. Parsing them as UTC would put every file 7 h off and
        # flag a whole deployment as inconsistent. The stamp is pinned
        # to its epoch here rather than read back through clock, so a
        # change to the default zone moves only one side of the test.
        name = "corr_20250922_160500.h5"
        t_close = 1758582300.0  # 2025-09-22 16:05:00 PDT
        write_corr_file(tmp_path / name, ntimes=10, sync_time=t_close - 60)
        idx = MetadataIndex(tmp_path, cache=False)
        assert idx.table.sync_consistent.all()

    def test_filename_tz_is_honoured(self, tmp_path):
        # A zone-less name stamped in Mountain time reads an hour late
        # under the Pacific default: the last integration then sits
        # 3655 s before the name, just past the 3600 s tolerance, and
        # the whole run is flagged until the right zone is passed.
        name = "corr_20250922_160500.h5"
        t_close = clock.filename_unix(name, tz="America/Denver")
        write_corr_file(tmp_path / name, ntimes=10, sync_time=t_close - 60)
        default = MetadataIndex(tmp_path, cache=False)
        assert not default.table.sync_consistent.any()
        denver = MetadataIndex(
            tmp_path, cache=False, filename_tz="America/Denver"
        )
        assert denver.table.sync_consistent.all()

    def test_z_suffix_beats_an_explicit_filename_tz(self, tmp_path):
        # 2026-07-17 15:00:41 UTC. A filename_tz set for the older
        # deployments must not drag a Z-stamped name 6 h sideways.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=10,
            sync_time=1784300441.0 - 60,
        )
        idx = MetadataIndex(
            tmp_path, cache=False, filename_tz="America/Denver"
        )
        assert idx.table.sync_consistent.all()


class TestTimeBest:
    def test_good_file_uses_header_time(self, tmp_path):
        idx = MetadataIndex(stale_pair(tmp_path), cache=False)
        good = idx.table[idx.table.file == "corr_20260717_150041Z.h5"]
        np.testing.assert_array_equal(good.time_best, good.time)

    def test_stale_file_uses_filename_estimate(self, tmp_path):
        # time_fname = close time - (rows after this one) x dt, so the
        # last row lands on the filename stamp and earlier rows are
        # spaced by the integration time.
        idx = MetadataIndex(stale_pair(tmp_path), cache=False)
        bad = idx.table[idx.table.file == "corr_20260717_151041Z.h5"]
        assert not bad.sync_consistent.any()
        np.testing.assert_array_equal(bad.time_best, bad.time_fname)
        assert bad.time_best.iloc[-1] == pytest.approx(1784301041.0)
        assert np.allclose(np.diff(bad.time_best), 0.5369)

    def test_stale_file_sorts_into_its_true_slot(self, tmp_path):
        # Sorting on header time would put the stale file 54 days
        # before everything else, at the front of the table.
        idx = MetadataIndex(stale_pair(tmp_path), cache=False)
        assert idx.table.file.iloc[0] == "corr_20260717_150041Z.h5"
        assert idx.table.file.iloc[-1] == "corr_20260717_151041Z.h5"
        assert idx.table.time_best.is_monotonic_increasing

    def test_time_fname_is_nan_without_a_stamp(self, tmp_path):
        write_corr_file(tmp_path / "corr_nostamp.h5", ntimes=4)
        idx = MetadataIndex(tmp_path, cache=False)
        assert idx.table.time_fname.isna().all()
        assert not idx.table.sync_consistent.any()
        assert isinstance(idx.table, pd.DataFrame)
