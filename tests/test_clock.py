"""Epoch is never mutated; timezone is a rendering concern."""

from datetime import timezone

import numpy as np
import pytest

from eigsep_data import clock, data


class TestFormatTime:
    def test_renders_utc_by_default(self):
        # 1784321280.0 == 2026-07-17 20:48:00Z (verified, not assumed).
        assert clock.format_time(1784321280.0) == "2026-07-17 20:48:00"

    def test_renders_mountain_for_field_notes(self):
        # The legacy +3600 existed so timestamps matched watches in
        # Utah. That is a rendering job, not a correction to the data.
        # MDT is UTC-6, so 20:48Z is 14:48 local.
        assert (
            clock.format_time(1784321280.0, tz="America/Denver")
            == "2026-07-17 14:48:00"
        )

    def test_accepts_an_array(self):
        out = clock.format_time(np.array([1784321280.0, 1784321340.0]))
        assert out == ["2026-07-17 20:48:00", "2026-07-17 20:49:00"]


class TestParseFilenameTime:
    def test_z_suffix_is_utc(self):
        t = clock.parse_filename_time("corr_20260717_204800Z.h5")
        assert t.tzinfo is not None
        assert t.timestamp() == 1784321280.0

    def test_z_suffix_wins_over_an_explicit_tz(self):
        # The suffix is the producer's own statement of the zone; a
        # caller's guess must not override it.
        t = clock.parse_filename_time(
            "corr_20260717_204800Z.h5", tz="America/Los_Angeles"
        )
        assert t.timestamp() == 1784321280.0

    def test_no_suffix_defaults_to_pacific(self):
        # Before eigsep_observing c4ef1ee the writer used a naive
        # datetime.now(), so deployment 1-4 filenames are Pacific wall
        # clock. September is PDT, UTC-7.
        t = clock.parse_filename_time("corr_20250922_160500.h5")
        assert t.utcoffset().total_seconds() == -7 * 3600
        assert (t.hour, t.minute) == (16, 5)

    def test_explicit_tz_is_honoured_without_suffix(self):
        t = clock.parse_filename_time(
            "corr_20250922_160500.h5", tz="America/Denver"
        )
        assert t.utcoffset().total_seconds() == -6 * 3600

    def test_disambiguating_suffix_still_parses(self):
        t = clock.parse_filename_time("corr_20260712_235712Z-1.h5")
        assert t.tzinfo == timezone.utc

    def test_unparseable_name_raises(self):
        with pytest.raises(ValueError):
            clock.parse_filename_time("not_a_corr_file.h5")

    def test_data_alias_still_works(self):
        # tests/test_data.py and any notebook using the private name
        # keep working; the alias is tz-aware.
        t = data._parse_time_from_name("corr_20260717_204800Z.h5")
        assert t.timestamp() == 1784321280.0


class TestFilenameUnix:
    def test_matches_parse(self):
        assert clock.filename_unix("corr_20260717_204800Z.h5") == 1784321280.0

    def test_nan_on_garbage(self):
        # The index calls this on every file it globs; a stray file
        # must yield NaN, not an exception.
        assert np.isnan(clock.filename_unix("notes.txt"))


class TestToUnixTime:
    def test_naive_string_is_utc(self):
        assert clock.to_unix_time("2026-07-17 20:48:00") == 1784321280.0

    def test_epoch_passthrough(self):
        assert clock.to_unix_time(1784321280.0) == 1784321280.0

    def test_reexported_from_data(self):
        # notebooks/dominic/test_beam_mapping_module.ipynb imports it
        # from eigsep_data.data.
        assert data.to_unix_time is clock.to_unix_time
