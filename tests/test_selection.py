"""Tests for eigsep_data.index.Selection."""

import numpy as np
import pandas as pd
import pytest

from eigsep_data.index import MetadataIndex
from eigsep_data.metadata import MISSING

from conftest import stale_pair, write_corr_file


class TestSelect:
    def test_selects_by_state(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        assert sel.nrows == 20
        assert set(sel.meta.rfswitch) == {"RFANT"}

    def test_accepts_a_list_of_states(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFAMB", "RFNON"]
        )
        assert sel.nrows == 60

    def test_unseen_state_returns_empty_not_error(self, corr_dir):
        # The vocabulary is open; asking for a state this directory
        # never saw is a legitimate empty result, not a crash.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="VNAO")
        assert sel.nrows == 0

    def test_empty_selection_still_answers(self, corr_dir):
        # An empty result is a normal answer, so every accessor on it
        # has to work rather than blow up on a zero-length frame.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="VNAO")
        assert sel.files == []
        assert sel.file_counts().empty
        assert sel.visits().tolist() == []
        assert "0 rows selected from 0 files" in sel.summary()

    def test_none_rows_are_never_returned_as_a_state(self, corr_dir):
        # The ladder has 5 None rows; they must not answer to RFNOFF.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFNOFF")
        assert sel.nrows == 30  # 20 + 10, not 35

    def test_missing_rows_are_addressable_as_missing(self, corr_dir):
        # None outnumbers UNKNOWN 22x in deployment 5, so "no switch
        # information" has to be reachable in its own right, not merely
        # excluded from the real states. Both flavours answer to it:
        # the ladder's 5-row None dropout and the 60 rows of the file
        # that carries no rfswitch stream at all.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch=MISSING)
        assert sel.nrows == 65
        assert sel.file_counts().to_dict() == {
            "corr_20260717_150041Z.h5": 5,
            "corr_20260717_151041Z.h5": 60,
        }

    def test_no_arguments_selects_everything(self, corr_dir):
        assert MetadataIndex(corr_dir, cache=False).select().nrows == 180

    def test_filters_by_time_range(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        lo = float(idx.table.time_best.min())
        sel = idx.select(time=(lo, lo + 9.5 * 0.5369))
        assert sel.nrows == 10

    def test_filters_by_filename_glob(self, corr_dir):
        # Filename selectors matter because 12% of deployment-5 files
        # have header times wrong by days -- a time range on header
        # time cannot reach them, but a glob can.
        sel = MetadataIndex(corr_dir, cache=False).select(
            files="corr_20260717_1500*"
        )
        assert set(sel.meta.file) == {"corr_20260717_150041Z.h5"}

    def test_filters_by_filename_range(self, corr_dir):
        # motor_scan_20260717.ipynb selected its raster block as
        # RASTER_LO <= name <= RASTER_HI. A tuple is that range,
        # inclusive at both ends.
        sel = MetadataIndex(corr_dir, cache=False).select(
            files=("corr_20260717_150041Z.h5", "corr_20260717_151041Z.h5")
        )
        assert sel.files == [
            "corr_20260717_150041Z.h5",
            "corr_20260717_151041Z.h5",
        ]

    def test_glob_list_is_not_a_range(self, corr_dir):
        # A list is a set of globs; only a tuple is a range.
        sel = MetadataIndex(corr_dir, cache=False).select(
            files=["corr_*_150041Z.h5", "corr_*_152041Z.h5"]
        )
        assert len(sel.files) == 2
        # The first and third file, not the first two a range would
        # have swept up.
        assert sel.files == [
            "corr_20260717_150041Z.h5",
            "corr_20260717_152041Z.h5",
        ]

    def test_filters_by_run_tag(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(run_tag="motor_scan")
        assert sel.nrows == 60

    def test_filters_by_root_attr(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(filter_phase="C")
        assert sel.nrows == 180

    def test_filters_by_bool_column(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(sync_consistent=True)
        assert sel.nrows == 180

    def test_bool_attr_false_is_distinct_from_missing(self, tmp_path):
        # filter_corr_keys.py writes mux_copy_* on the files it
        # touches, so a mixed directory gives an object column holding
        # real True, real False and the string MISSING. The False leg
        # must return the files that said "no", never the files that
        # said nothing at all.
        write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=4,
            root_attrs={"mux_copy_0to1": True},
        )
        write_corr_file(
            tmp_path / "corr_20260717_151041Z.h5",
            ntimes=6,
            sync_time=1.7843e9 + 600,
            root_attrs={"mux_copy_0to1": False},
            seed=1,
        )
        write_corr_file(
            tmp_path / "corr_20260717_152041Z.h5",
            ntimes=8,
            sync_time=1.7843e9 + 1200,
            seed=2,
        )
        idx = MetadataIndex(tmp_path, cache=False)
        assert idx.select(mux_copy_0to1=True).files == [
            "corr_20260717_150041Z.h5"
        ]
        assert idx.select(mux_copy_0to1=False).files == [
            "corr_20260717_151041Z.h5"
        ]
        assert idx.select(mux_copy_0to1=False).nrows == 6
        assert idx.select(mux_copy_0to1=MISSING).nrows == 8

    def test_where_escape_hatch(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            where=lambda df: df.motor_az_pos > 50
        )
        assert sel.nrows > 0
        assert (sel.meta.motor_az_pos > 50).all()
        # motor az_pos counts up 0..59 in each of the three files, so
        # the answer is 9 rows per file -- pinned independently of the
        # mask the test just handed in.
        assert sel.nrows == 27

    def test_selection_composes(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        sel = idx.select(rfswitch="RFANT").select(where=lambda df: df.row < 5)
        assert sel.nrows == 5

    def test_unknown_column_raises_with_a_useful_message(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        with pytest.raises(KeyError, match="nonsense"):
            idx.select(nonsense="x")


class TestTimeBestWindow:
    def test_time_window_reaches_a_stale_sync_file(self, tmp_path):
        # The stale file's header times are 54 days early; its
        # filename estimate is right. A window around the estimate must
        # find it, which a window on header time never could.
        idx = MetadataIndex(stale_pair(tmp_path), cache=False)
        sel = idx.select(time=(1784301041.0 - 5, 1784301041.0 + 1))
        assert sel.files == ["corr_20260717_151041Z.h5"]
        assert not sel.meta.sync_consistent.any()

    def test_unstamped_rows_fall_outside_every_window(self, tmp_path):
        # No filename stamp and no trustworthy clock leaves time_best
        # NaN. Such rows sort to the end and cannot answer a time
        # window -- deliberate, and why summary() names them.
        write_corr_file(tmp_path / "corr_20260717_150041Z.h5", ntimes=4)
        write_corr_file(tmp_path / "corr_nostamp.h5", ntimes=3, seed=1)
        idx = MetadataIndex(tmp_path, cache=False)
        assert idx.table.time_best.isna().sum() == 3
        sel = idx.select(time=(1.7843e9 - 10, 1.7843e9 + 10))
        assert set(sel.meta.file) == {"corr_20260717_150041Z.h5"}


class TestSummary:
    def test_reports_what_each_filter_removed(self, corr_dir):
        # load_gated silently dropped every None row and every file with
        # no stream. Making the exclusions visible is the point, so the
        # whole line shape is pinned -- which filter, before, after and
        # how many it removed, in that order. Bare substring checks
        # would pass on a summary with before and after swapped.
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        text = sel.summary()
        assert "180" in text and "20" in text
        assert "rfswitch" in text
        assert text.startswith("180 rows indexed\n")
        assert "  rfswitch='RFANT': 180 -> 20 (160 removed)" in text
        assert "20 rows selected from 1 files" in text

    def test_reports_rows_with_estimated_times(self, corr_dir):
        text = MetadataIndex(corr_dir, cache=False).select().summary()
        assert "0 rows have sync_consistent=False" in text
        assert "filename" in text

    def test_reports_rows_with_no_time_best(self, tmp_path):
        # A file whose name carries no stamp contributes rows no time
        # window can ever reach. Counting them next to the estimated
        # times is what keeps that from being silent.
        write_corr_file(tmp_path / "corr_20260717_150041Z.h5", ntimes=4)
        write_corr_file(tmp_path / "corr_nostamp.h5", ntimes=3, seed=1)
        text = MetadataIndex(tmp_path, cache=False).select().summary()
        assert "3 rows have no time_best" in text
        assert (
            "0 rows have no time_best"
            in MetadataIndex(tmp_path, cache=False)
            .select(files="corr_2*")
            .summary()
        )


class TestFileCounts:
    def test_rows_per_file_in_selection_order(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFANT", "RFAMB"]
        )
        counts = sel.file_counts()
        assert isinstance(counts, pd.Series)
        assert counts.to_dict() == {
            "corr_20260717_150041Z.h5": 20,
            "corr_20260717_152041Z.h5": 30,
        }
        assert list(counts.index) == sel.files


class TestVisits:
    def test_groups_contiguous_runs(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        # One file, integrations 0.2684 s apart end to end: one visit.
        sel = idx.select(rfswitch=["RFAMB", "RFNON"])
        visits = sel.visits(gap_s=60)
        assert len(visits) == sel.nrows
        assert len(np.unique(visits)) == 1
        # Add a state from a file 20 min earlier and the ~1190 s hole
        # between the two blocks has to start a second visit.
        spread = idx.select(rfswitch=["RFANT", "RFAMB"])
        np.testing.assert_array_equal(
            spread.visits(gap_s=60), [0] * 20 + [1] * 30
        )

    def test_new_visit_after_a_gap(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        sel = idx.select(where=lambda df: df.row < 2)
        # Three files, 10 min apart, two rows each.
        assert len(np.unique(sel.visits(gap_s=60))) == 3

    def test_timeless_rows_join_no_visit(self, tmp_path):
        # A row with no time_best cannot be placed beside anything. It
        # sorts to the tail, so inheriting its neighbour's id would
        # quietly pad the last visit with rows from an unrelated file
        # on an unrelated day -- which a caller then averages.
        write_corr_file(tmp_path / "corr_20260717_150041Z.h5", ntimes=4)
        write_corr_file(
            tmp_path / "corr_20260717_152041Z.h5",
            ntimes=4,
            sync_time=1.7843e9 + 1200,
            seed=1,
        )
        write_corr_file(tmp_path / "corr_nostamp.h5", ntimes=3, seed=2)
        sel = MetadataIndex(tmp_path, cache=False).select()
        visits = sel.visits(gap_s=60)
        # Two real visits 20 min apart, then three rows in no visit.
        np.testing.assert_array_equal(
            visits, [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1]
        )
        timeless = sel.meta.time_best.isna().to_numpy()
        assert set(visits[timeless]) == {-1}
        assert -1 not in visits[~timeless]
