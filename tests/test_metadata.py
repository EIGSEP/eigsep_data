"""Tests for eigsep_data.metadata."""

import numpy as np

from eigsep_data import metadata as md


class TestFlattenMetadata:
    """The three kinds of missing must stay distinguishable."""

    def test_absent_stream_is_missing_not_unknown(self):
        # The ~1842 deployment-5 files with no rfswitch metadata must
        # not masquerade as the producer's UNKNOWN contamination flag.
        cols = md.flatten_metadata({}, ntimes=4)
        assert list(cols["rfswitch"]) == [md.MISSING] * 4
        assert not cols["rfswitch_ok"].any()

    def test_none_row_is_not_a_state(self):
        # None means no reading reached the writer for that window.
        cols = md.flatten_metadata(
            {"rfswitch": ["RFANT", None, "RFANT", None]}, ntimes=4
        )
        assert list(cols["rfswitch"]) == [
            "RFANT",
            md.MISSING,
            "RFANT",
            md.MISSING,
        ]
        assert list(cols["rfswitch_ok"]) == [True, False, True, False]

    def test_unknown_is_preserved_verbatim(self):
        cols = md.flatten_metadata(
            {"rfswitch": ["RFANT", "UNKNOWN"]}, ntimes=2
        )
        assert list(cols["rfswitch"]) == ["RFANT", "UNKNOWN"]
        # UNKNOWN is information: the row is present, just contaminated.
        assert list(cols["rfswitch_ok"]) == [True, True]

    def test_unseen_state_does_not_raise(self):
        # The vocabulary is open: 17 values observed in deployment 5,
        # and the upstream fixtures already use RFNOFF. A closed enum
        # (as in the rf_state_tools prototype) raises here.
        cols = md.flatten_metadata({"rfswitch": ["WHO_KNOWS"]}, ntimes=1)
        assert cols["rfswitch"][0] == "WHO_KNOWS"

    def test_dict_stream_flattens_to_prefixed_columns(self):
        meta = {
            "motor": [
                {"status": "update", "el_pos": 1.0, "az_pos": 2.0},
                {"status": "update", "el_pos": 3.0, "az_pos": 4.0},
            ]
        }
        cols = md.flatten_metadata(meta, ntimes=2)
        np.testing.assert_allclose(cols["motor_el_pos"], [1.0, 3.0])
        np.testing.assert_allclose(cols["motor_az_pos"], [2.0, 4.0])

    def test_offline_sensor_keeps_ok_true_but_values_nan(self):
        # imu_az/lidar/tempctrl_lna publish dicts whose fields are all
        # None: the stream is alive, the sensor is not. That is a
        # different failure from the row being absent, and callers
        # need to tell them apart.
        meta = {"motor": [{"status": "update", "el_pos": None}]}
        cols = md.flatten_metadata(meta, ntimes=1)
        assert np.isnan(cols["motor_el_pos"][0])
        assert cols["motor_ok"][0]

    def test_short_stream_is_padded_to_ntimes(self):
        cols = md.flatten_metadata({"rfswitch": ["RFANT"]}, ntimes=3)
        assert len(cols["rfswitch"]) == 3
        assert list(cols["rfswitch"][1:]) == [md.MISSING] * 2

    def test_error_status_marks_stream_not_ok(self):
        meta = {"motor": [{"status": "error", "el_pos": 1.0}]}
        cols = md.flatten_metadata(meta, ntimes=1)
        assert not cols["motor_ok"][0]

    def test_string_field_gap_is_missing_never_none(self):
        # A Python None in an object column would become the string
        # "None" after a trip through the h5 cache. Gaps in string
        # fields must be MISSING so cached and fresh tables are equal.
        meta = {
            "potmon": [
                {"status": "update", "sp1_term_name": "SHORT"},
                {"status": "update", "sp1_term_name": None},
                None,
            ]
        }
        cols = md.flatten_metadata(meta, ntimes=3)
        assert list(cols["potmon_sp1_term_name"]) == [
            "SHORT",
            md.MISSING,
            md.MISSING,
        ]
        assert None not in list(cols["potmon_sp1_term_name"])

    def test_tempctrl_lna_is_curated(self):
        # Calibrator work needs the LNA temperature as much as the load
        # temperature; both are two columns.
        assert md.CURATED_FIELDS["tempctrl_lna"] == ("T_now", "active")
        meta = {"tempctrl_lna": [{"status": "update", "T_now": 31.5}]}
        cols = md.flatten_metadata(meta, ntimes=1)
        np.testing.assert_allclose(cols["tempctrl_lna_T_now"], [31.5])
