"""quicklook must be able to report and gate on switch state."""

import pytest

from eigsep_data import quicklook as ql

from conftest import RFSWITCH_LADDER, write_corr_file


class TestQuickLookState:
    def test_reports_the_state_breakdown(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        res = ql.quicklook(p)
        assert res.state_counts["RFANT"] == 20
        assert res.state_counts["UNKNOWN"] == 5
        assert res.state_counts["MISSING"] == 5

    def test_gating_restricts_the_waterfall(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        res = ql.quicklook(p, state="RFANT")
        assert res.stats["0"]["ntimes"] == 20
        assert res.times.shape == (20,)

    def test_gating_on_an_absent_state_raises(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=60,
            rfswitch=RFSWITCH_LADDER,
        )
        with pytest.raises(ValueError, match="VNAO"):
            ql.quicklook(p, state="VNAO")

    def test_file_without_rfswitch_reports_all_missing(self, tmp_path):
        p = write_corr_file(
            tmp_path / "corr_20260717_150041Z.h5",
            ntimes=10,
            rfswitch=None,
        )
        res = ql.quicklook(p)
        assert res.state_counts == {"MISSING": 10}
