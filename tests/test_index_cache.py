"""The cache must never serve a table that no longer matches the data."""

import os
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

from eigsep_data.index import MetadataIndex, scan_corr_file
from eigsep_data.metadata import MISSING

from conftest import corrupt_stream, write_corr_file


def gappy_dir(tmp_path, present):
    """
    One file per entry of *present*, plus one carrying no root attrs.

    Every attr named in *present* therefore lands in an object column
    holding real values beside the ``MISSING`` sentinel -- the shape a
    directory of half-filtered files really has.
    """
    for i, attrs in enumerate(list(present) + [None]):
        write_corr_file(
            tmp_path / f"corr_20260717_15{i}041Z.h5",
            ntimes=4,
            sync_time=1.7843e9 + i * 600,
            root_attrs=attrs,
            seed=i,
        )
    return tmp_path


class TestCache:
    def test_second_build_reuses_the_cache_losslessly(self, corr_dir):
        first = MetadataIndex(corr_dir)
        assert first.cache_path.exists()
        second = MetadataIndex(corr_dir)
        assert second.from_cache
        # Whole-table equality, dtypes and column order included: the
        # h5 round trip must not turn None into "None" or NaN into
        # "nan", or reorder columns.
        pd.testing.assert_frame_equal(first.table, second.table)

    def test_a_cache_hit_does_not_read_the_files(self, corr_dir):
        # from_cache is the index's own account of itself; what a caller
        # actually buys is that the 5120 files are not opened again.
        # The same scanner has to be used both times -- it is part of
        # the fingerprint.
        seen = []

        def scanner(path, streams=None, filename_tz=None):
            seen.append(path.name)
            return scan_corr_file(path, streams, filename_tz)

        first = MetadataIndex(corr_dir, scanner=scanner)
        assert len(seen) == 3
        second = MetadataIndex(corr_dir, scanner=scanner)
        assert second.from_cache
        assert len(seen) == 3
        pd.testing.assert_frame_equal(first.table, second.table)

    def test_narrower_stream_request_hits(self, corr_dir):
        # A cache holding the curated set can answer a motor-only
        # request; the extra columns are just along for the ride.
        MetadataIndex(corr_dir)
        narrow = MetadataIndex(corr_dir, streams=("motor",))
        assert narrow.from_cache
        assert "rfswitch" in narrow.table.columns

    def test_wider_stream_request_rebuilds_with_the_union(self, corr_dir):
        MetadataIndex(corr_dir, streams=("motor",))
        wider = MetadataIndex(corr_dir)
        assert not wider.from_cache
        assert "potmon_pot_az_angle" in wider.table.columns
        # ... and the union is what got cached, so both requests now hit.
        assert MetadataIndex(corr_dir, streams=("motor",)).from_cache
        assert MetadataIndex(corr_dir).from_cache

    def test_union_keeps_what_only_the_older_request_asked_for(self, corr_dir):
        # adc_stats is outside the curated set, so widening to the
        # curated default must not drop it: a union, not a replacement.
        MetadataIndex(corr_dir, streams=("adc_stats",))
        wider = MetadataIndex(corr_dir)
        assert not wider.from_cache
        assert "adc_stats_ok" in wider.table.columns
        assert "rfswitch" in wider.table.columns
        assert "adc_stats" in wider.cached_streams
        assert "lidar" in wider.cached_streams

    def test_all_streams_covers_everything(self, corr_dir):
        MetadataIndex(corr_dir, streams="all")
        assert MetadataIndex(corr_dir).from_cache
        assert MetadataIndex(corr_dir, streams=("lidar",)).from_cache

    def test_a_narrow_cache_never_answers_an_all_request(self, corr_dir):
        # "all" means every stream the files happen to carry, which a
        # named set cannot promise to contain.
        MetadataIndex(corr_dir)
        assert not MetadataIndex(corr_dir, streams="all").from_cache

    def test_new_file_invalidates(self, corr_dir):
        MetadataIndex(corr_dir)
        write_corr_file(
            corr_dir / "corr_20260717_153041Z.h5",
            ntimes=60,
            rfswitch=["RFANT"] * 60,
            sync_time=1.7843e9 + 1800,
        )
        rebuilt = MetadataIndex(corr_dir)
        assert not rebuilt.from_cache
        assert len(rebuilt.table) == 240

    def test_modified_file_invalidates(self, corr_dir):
        MetadataIndex(corr_dir)
        write_corr_file(
            corr_dir / "corr_20260717_150041Z.h5",
            ntimes=30,
            rfswitch=["RFANT"] * 30,
        )
        rebuilt = MetadataIndex(corr_dir)
        assert not rebuilt.from_cache
        assert len(rebuilt.table) == 150

    def test_schema_bump_invalidates(self, corr_dir, monkeypatch):
        MetadataIndex(corr_dir)
        monkeypatch.setattr("eigsep_data.index.SCHEMA_VERSION", 99)
        assert not MetadataIndex(corr_dir).from_cache

    def test_filename_tz_is_part_of_the_identity(self, corr_dir):
        # sync_consistent and time_best depend on it.
        MetadataIndex(corr_dir)
        assert not MetadataIndex(
            corr_dir, filename_tz="America/Denver"
        ).from_cache

    def test_scanner_is_part_of_the_identity(self, corr_dir):
        # A different scanner produces different columns from the same
        # bytes, so the file manifest alone cannot vouch for the table.
        MetadataIndex(corr_dir)

        def scanner(path, streams=None, filename_tz=None):
            return scan_corr_file(path, streams, filename_tz).head(1)

        assert not MetadataIndex(corr_dir, scanner=scanner).from_cache

    def test_patterns_are_part_of_the_identity(self, corr_dir):
        # "*.h5" indexes sidecars that "corr_*.h5" excludes, so the two
        # must never answer for each other.
        write_corr_file(corr_dir / "beam_20260717_150041Z.h5", ntimes=4)
        wide = MetadataIndex(corr_dir, patterns=("*.h5",))
        assert len(wide.table) == 184
        narrow = MetadataIndex(corr_dir)
        assert not narrow.from_cache
        assert len(narrow.table) == 180
        # The globs are in the key, not merely the files they matched,
        # and one directory holds one sidecar: two callers that glob
        # differently overwrite each other and each rescan, even when
        # the file set is identical. Deliberate -- see rebuild.
        assert not MetadataIndex(
            corr_dir, patterns=("corr_*.h5", "*.hdf5")
        ).from_cache

    def test_corrupt_cache_is_rebuilt(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        idx.cache_path.write_bytes(b"garbage")
        rebuilt = MetadataIndex(corr_dir)
        assert not rebuilt.from_cache
        assert MetadataIndex(corr_dir).from_cache

    def test_an_undecodable_cache_says_so(self, corr_dir):
        # A readable sidecar whose contents do not decode is a defect in
        # the codec, not a stale cache: silently rescanning would hide
        # it behind nothing but a ~64 s wait every session. (The
        # sidecar is outside its own manifest, so editing it here does
        # not invalidate the fingerprint.)
        idx = MetadataIndex(corr_dir)
        with h5py.File(idx.cache_path, "a") as h5:
            del h5["columns"]["rfswitch"]  # still listed in the attr
        with pytest.warns(UserWarning, match="Could not read index cache"):
            rebuilt = MetadataIndex(corr_dir)
        assert not rebuilt.from_cache
        assert len(rebuilt.table) == 180

    def test_a_one_shot_streams_iterator_survives_a_rebuild(self, corr_dir):
        # An exhausted iterator canonicalises to the empty set, which
        # would scan no streams at all and say nothing about it.
        idx = MetadataIndex(corr_dir, streams=iter(["motor"]))
        assert "motor_az_pos" in idx.table.columns
        idx.rebuild(force=True)
        assert "motor_az_pos" in idx.table.columns

    def test_a_bare_stream_name_is_a_name_not_eight_letters(self, corr_dir):
        # "all" is a documented bare string, so streams="rfswitch" is a
        # natural thing to try. Iterating it as characters would scan
        # eight streams named after letters and return nothing real.
        idx = MetadataIndex(corr_dir, streams="rfswitch", cache=False)
        assert "rfswitch" in idx.table.columns
        assert "r_ok" not in idx.table.columns

    def test_cache_false_never_writes(self, corr_dir):
        idx = MetadataIndex(corr_dir, cache=False)
        assert not idx.cache_path.exists()

    def test_force_rescans(self, corr_dir):
        MetadataIndex(corr_dir)
        idx = MetadataIndex(corr_dir)
        assert idx.from_cache
        idx.rebuild(force=True)
        assert not idx.from_cache

    def test_the_cache_does_not_invalidate_itself(self, corr_dir):
        # Task 9's beam wrapper indexes with "*.h5". If the sidecar
        # entered its own manifest, writing it would change the
        # fingerprint it was just written under and no build would ever
        # hit the cache again.
        MetadataIndex(corr_dir, patterns=("*.h5",))
        second = MetadataIndex(corr_dir, patterns=("*.h5",))
        assert second.from_cache
        assert len(second.table) == 180


class TestLosslessColumns:
    def test_gappy_boolean_column_survives_the_round_trip(self, tmp_path):
        # filter_corr_keys.py writes mux_copy_* per file, so a directory
        # mixing filtered and unfiltered files gives an object column
        # holding real True, real False and MISSING. Encoding it as
        # bytes wholesale would hand back the *strings* "True"/"False"
        # and make select(mux_copy_0to1=True) match nothing -- an empty
        # answer indistinguishable from an honest one.
        root = gappy_dir(
            tmp_path, [{"mux_copy_0to1": True}, {"mux_copy_0to1": False}]
        )
        first = MetadataIndex(root)
        assert first.table.mux_copy_0to1.dtype == object
        second = MetadataIndex(root)
        assert second.from_cache
        col = second.table.mux_copy_0to1
        assert col.eq(True).sum() == 4
        assert col.eq(False).sum() == 4
        assert col.eq(MISSING).sum() == 4
        pd.testing.assert_frame_equal(first.table, second.table)

    def test_gappy_numeric_column_keeps_nan_gaps(self, tmp_path):
        # The other way a root attr can be gappy: a numeric one takes a
        # NaN rather than turning into an object column, and the NaN has
        # to come back as a NaN, not as the string "nan" -- which would
        # read as a value and never answer isna().
        root = gappy_dir(tmp_path, [{"filter_nkeys": 3}, {"filter_nkeys": 4}])
        first = MetadataIndex(root)
        assert first.table.filter_nkeys.dtype == float
        second = MetadataIndex(root)
        assert second.from_cache
        assert second.table.filter_nkeys.isna().sum() == 4
        assert second.table.filter_nkeys.eq(3).sum() == 4
        pd.testing.assert_frame_equal(first.table, second.table)

    def test_all_string_column_comes_back_as_strings(self, tmp_path):
        # And the converse: a column that really is all strings must
        # come back as strings, sentinel included.
        root = gappy_dir(tmp_path, [{"filter_phase": "C"}] * 2)
        MetadataIndex(root)
        second = MetadataIndex(root)
        assert second.from_cache
        col = second.table.filter_phase
        assert col.map(type).eq(str).all()
        assert col.eq("C").sum() == 8
        assert col.eq(MISSING).sum() == 4

    def test_native_dtypes_are_not_widened(self, corr_dir):
        # row is int32 and sync_consistent bool in the scan; coming back
        # as int64 and object would still compare equal and quietly cost
        # memory on 1.2M rows.
        MetadataIndex(corr_dir)
        second = MetadataIndex(corr_dir)
        assert second.from_cache  # or these are the scan's dtypes
        assert second.table.row.dtype == np.int32
        assert second.table.sync_consistent.dtype == bool
        assert second.table.mux_copy_0to1.dtype == bool

    def test_a_failed_write_leaves_no_rubble(self, corr_dir):
        # "." names a group itself in HDF5, so a column called that
        # fails *inside* the write, after the file exists -- the shape a
        # full disk or an interrupted write has. The partial sidecar can
        # never be served, and at ~300 bytes a row it must not be left
        # sitting next to the data either.
        with h5py.File(corr_dir / "corr_20260717_150041Z.h5", "a") as h5:
            h5.attrs["."] = 1
        with pytest.warns(UserWarning, match="Could not write index cache"):
            idx = MetadataIndex(corr_dir)
        assert len(idx.table) == 180
        assert not idx.cache_path.exists()

    def test_a_failed_write_keeps_the_cache_it_never_touched(self, corr_dir):
        # The widening policy means a request can rescan and then fail to
        # cache the wider table -- here because the wider stream set is
        # what brings in the column that cannot be encoded. That failure
        # happens before anything is opened, so the narrower cache
        # already on disk is not this attempt's rubble: deleting it would
        # throw away a valid table and leave the next narrow request to
        # rescan for nothing.
        def scanner(path, streams=None, filename_tz=None):
            table = scan_corr_file(path, streams, filename_tz)
            if streams == "all":
                table["odd"] = [{"a": 1}] * len(table)
            return table

        narrow = MetadataIndex(corr_dir, streams=("motor",), scanner=scanner)
        assert narrow.cache_path.exists()
        with pytest.warns(UserWarning, match="Could not write index cache"):
            MetadataIndex(corr_dir, streams="all", scanner=scanner)
        assert narrow.cache_path.exists()
        assert MetadataIndex(
            corr_dir, streams=("motor",), scanner=scanner
        ).from_cache

    def test_unrepresentable_column_is_not_cached(self, corr_dir):
        # A scanner is a public seam, so a column the cache cannot
        # restore exactly is possible. Writing a lossy cache would serve
        # a different table in every later session; declining to write
        # one only costs a rescan.
        def scanner(path, streams=None, filename_tz=None):
            table = scan_corr_file(path, streams, filename_tz)
            table["odd"] = [{"a": 1}] * len(table)
            return table

        with pytest.warns(UserWarning, match="Could not write index cache"):
            idx = MetadataIndex(corr_dir, scanner=scanner)
        assert len(idx.table) == 180
        assert not idx.cache_path.exists()


class TestPartialScans:
    def test_skipped_files_are_reported_again_from_the_cache(self, corr_dir):
        # The fingerprint validates over the file set even when a file
        # was dropped, so a partial table would otherwise be served
        # silently and for ever, with .skipped coming back empty.
        (corr_dir / "corr_20260717_153041Z.h5").write_bytes(b"truncated")
        with pytest.warns(UserWarning, match="corr_20260717_153041Z.h5"):
            first = MetadataIndex(corr_dir)
        assert [name for name, _ in first.skipped] == [
            "corr_20260717_153041Z.h5"
        ]
        with pytest.warns(UserWarning, match="corr_20260717_153041Z.h5"):
            second = MetadataIndex(corr_dir)
        assert second.from_cache
        assert second.skipped == first.skipped
        assert len(second.table) == len(first.table) == 180

    def test_lost_streams_are_reported_again_from_the_cache(self, corr_dir):
        # Same hazard as a skipped file, one level down: the table is
        # full length, so nothing about it looks partial, and the rows
        # of the lost stream read MISSING for ever after.
        corrupt_stream(corr_dir / "corr_20260717_150041Z.h5", "motor")
        with pytest.warns(UserWarning, match="motor"):
            first = MetadataIndex(corr_dir)
        with pytest.warns(UserWarning, match="motor"):
            second = MetadataIndex(corr_dir)
        assert second.from_cache
        assert second.lost_streams == first.lost_streams
        assert [(f, s) for f, s, _ in second.lost_streams] == [
            ("corr_20260717_150041Z.h5", "motor")
        ]

    def test_a_clean_cache_hit_warns_about_nothing(self, corr_dir):
        # The re-warning above must be driven by what was cached, not
        # emitted on every hit.
        MetadataIndex(corr_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            second = MetadataIndex(corr_dir)
        assert second.from_cache
        assert second.skipped == []
        assert second.lost_streams == []


class TestReadOnlyDirectory:
    @pytest.mark.skipif(
        os.geteuid() == 0, reason="root writes read-only directories anyway"
    )
    def test_write_failure_warns_and_still_indexes(self, corr_dir):
        # Deployment data often sits on a read-only mount; losing the
        # cache is a slow index, not a failed one.
        corr_dir.chmod(0o500)
        try:
            with pytest.warns(
                UserWarning, match="Could not write index cache"
            ):
                idx = MetadataIndex(corr_dir)
        finally:
            corr_dir.chmod(0o700)
        assert len(idx.table) == 180
        assert not idx.cache_path.exists()
