"""Explicit antenna-resolution policy and bundle integration tests."""

import numpy as np
import pytest
from conftest import write_corr_file

from eigsep_data import (
    AntennaResolutionError,
    AntennaResolutionPolicy,
    MetadataIndex,
)

POLICY = {
    "schema_version": 1,
    "name": "test-campaign-safe-v1",
    "rules": [
        {
            "name": "phase-a-unresolved",
            "status": "reject",
            "match": {"filter_phase": "A"},
            "reason": "antenna transition unresolved",
        },
        {
            "name": "phase-b-valid",
            "status": "use",
            "match": {"filter_phase": "B", "mux_copy_4to5": True},
            "inputs": {"box-gnd": "3", "box-air": "4"},
            "crosses": [
                {
                    "antennas": ["box-gnd", "box-air"],
                    "key": "35",
                    "conjugated": False,
                }
            ],
        },
        {
            "name": "phase-b-invalid",
            "status": "reject",
            "match": {"filter_phase": "B", "mux_copy_4to5": False},
            "reason": "cross copy absent",
        },
        {
            "name": "phase-c",
            "status": "use",
            "match": {"filter_phase": "C"},
            "inputs": {"box-gnd": "0", "box-air": "4"},
            "crosses": [
                {
                    "antennas": ["box-gnd", "box-air"],
                    "key": "04",
                    "conjugated": False,
                }
            ],
        },
    ],
}


def test_policy_partitions_approved_and_explicitly_rejected_files():
    import pandas as pd

    meta = pd.DataFrame(
        {
            "file": ["a", "b-good", "b-bad", "c"],
            "filter_phase": ["A", "B", "B", "C"],
            "mux_copy_4to5": [False, True, False, True],
        }
    )
    approved, rejected, decisions = AntennaResolutionPolicy(POLICY).partition(
        meta
    )
    assert approved == ["b-good", "c"]
    assert [item["file"] for item in rejected] == ["a", "b-bad"]
    assert decisions["b-good"] == "phase-b-valid"
    assert decisions["c"] == "phase-c"


def test_bundle_policy_overrides_stale_headers_and_resolves_crosses(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    specifications = [
        (
            "corr_20260714_120000Z.h5",
            ("3", "4", "5", "35"),
            {"filter_phase": "B", "mux_copy_4to5": True},
            "3",
            "4",
            "35",
        ),
        (
            "corr_20260716_120000Z.h5",
            ("0", "4", "04"),
            {"filter_phase": "C", "mux_copy_4to5": True},
            "0",
            "4",
            "04",
        ),
    ]
    for number, (fname, keys, attrs, _gnd, _air, _cross) in enumerate(
        specifications
    ):
        write_corr_file(
            data / fname,
            ntimes=4,
            keys=keys,
            sync_time=1.7843e9 + 1000 * number,
            # Deliberately stale phase-A-style mapping.
            input_to_ant={"0": "box-air", "2": "box-gnd", "4": "viv-N"},
            root_attrs=attrs,
            seed=number,
        )
    selection = MetadataIndex(data, cache=False).select()
    policy = AntennaResolutionPolicy(POLICY)
    ground = selection.load_bundle(
        antenna="box-gnd", resolution_policy=policy, missing="raise"
    )
    air = selection.load_bundle(
        antenna="box-air", resolution_policy=policy, missing="raise"
    )
    cross = selection.load_bundle(
        antenna=("box-gnd", "box-air"),
        resolution_policy=policy,
        missing="raise",
    )
    for bundle, key_index in ((ground, 3), (air, 4), (cross, 5)):
        for fname, _keys, _attrs, gnd, air_key, cross_key in specifications:
            expected_key = {3: gnd, 4: air_key, 5: cross_key}[key_index]
            rows = bundle.meta.file == fname
            literal = selection.select(files=[fname]).load(keys=[expected_key])
            np.testing.assert_array_equal(
                bundle.data[rows], literal.data[expected_key]
            )
            assert set(bundle.meta.loc[rows, "input_key"]) == {expected_key}
    assert set(ground.provenance["resolution_rules"].values()) == {
        "phase-b-valid",
        "phase-c",
    }


def test_bundle_policy_rejection_is_never_missing_skip(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    fname = "corr_20260714_220000Z.h5"
    write_corr_file(
        data / fname,
        ntimes=4,
        keys=("3", "4", "5", "35"),
        root_attrs={"filter_phase": "B", "mux_copy_4to5": False},
    )
    selection = MetadataIndex(data, cache=False).select()
    with pytest.raises(AntennaResolutionError, match="cross copy absent"):
        selection.load_bundle(
            antenna="box-air",
            resolution_policy=AntennaResolutionPolicy(POLICY),
            missing="skip",
        )
