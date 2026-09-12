"""from_path is sugar over the index; it never shifts epoch."""

import pytest

import eigsep_data
from eigsep_data import data


class TestFromPath:
    def test_directory_loads_everything(self, corr_dir):
        d = data.EigsepData.from_path(corr_dir)
        assert d.times.shape == (180,)
        assert d.meta is not None

    def test_single_file(self, corr_dir):
        d = data.EigsepData.from_path(
            corr_dir / "corr_20260717_152041Z.h5", keys=["0"]
        )
        assert d.times.shape == (60,)
        assert list(d.data) == ["0"]

    def test_window_is_on_epoch(self, corr_dir):
        # Strings go through to_unix_time (naive == UTC) and select on
        # time_best. Header times, not filename wall clock.
        d = data.EigsepData.from_path(
            corr_dir,
            start_time="2026-07-17 15:03:20",  # == 1.7843e9 + 600
            end_time="2026-07-17 15:13:20",
        )
        assert set(d.meta.file) == {"corr_20260717_151041Z.h5"}

    def test_does_not_shift_times(self, corr_dir):
        d = data.EigsepData.from_path(corr_dir)
        assert abs(d.times[0] - 1.7843e9) < 1.0

    def test_pacific_to_mountain_warns_and_is_ignored(self, corr_dir):
        with pytest.warns(DeprecationWarning, match="pacific_to_mountain"):
            d = data.EigsepData.from_path(corr_dir, pacific_to_mountain=True)
        # Crucially: it warns AND does nothing. Deployment-5 epochs are
        # verified correct UTC; adding 3600 s corrupts them.
        assert abs(d.times[0] - 1.7843e9) < 1.0

    def test_package_exports(self):
        for name in (
            "EigsepData",
            "MetadataIndex",
            "Selection",
            "to_unix_time",
            "format_time",
            "metadata",
            "clock",
        ):
            assert hasattr(eigsep_data, name), name
