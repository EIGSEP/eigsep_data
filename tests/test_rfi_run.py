"""Planning and resume tests for the v3-beta campaign runner."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from eigsep_data.rfi_run import (
    _resume_files,
    build_batches,
    resolve_locations,
)
from eigsep_data.rfi_supported import (
    ALGORITHM_REVISION,
    LEGACY_COMPATIBLE_SOURCE_SHA256,
    RFIConfig,
    algorithm_source_sha256,
    parameter_sha256,
)


def _selection():
    rows = []
    specifications = [
        ("corr_20260716_000000Z.h5", 0.0, "C"),
        ("corr_20260716_000100Z.h5", 60.0, "C"),
        ("corr_20260716_000200Z.h5", 120.0, "C"),
        ("corr_20260716_000300Z.h5", 180.0, "D"),
        ("corr_20260717_000000Z.h5", 86400.0, "D"),
    ]
    for fname, start, phase in specifications:
        for row in range(2):
            rows.append(
                {
                    "file": fname,
                    "row": row,
                    "time_best": start + row,
                    "integration_time": 1.0,
                    "filter_phase": phase,
                    "data_keys": "0,04,4",
                }
            )
    return SimpleNamespace(meta=pd.DataFrame(rows))


def test_batches_do_not_cross_day_or_acquisition_configuration():
    batches = build_batches(_selection(), files_per_batch=2)
    assert [len(batch["files"]) for batch in batches] == [2, 1, 1, 1]
    assert [batch["block"] for batch in batches] == [0, 0, 1, 2]
    assert [batch["day"] for batch in batches] == [
        "20260716",
        "20260716",
        "20260716",
        "20260717",
    ]


def test_explicit_data_dir_defaults_products_to_its_parent(tmp_path):
    data = tmp_path / "mounted-raw"
    data.mkdir()
    assert resolve_locations(data_dir=data) == (
        data.resolve(),
        tmp_path.resolve(),
    )
    output = tmp_path / "products"
    assert resolve_locations(data_dir=data, output_root=output) == (
        data.resolve(),
        output.resolve(),
    )


def test_resume_requires_both_products_with_matching_parameters(tmp_path):
    fname = "corr_20260716_000000Z.h5"
    digest = parameter_sha256(RFIConfig())
    algorithm = algorithm_source_sha256()
    record = {
        fname: {
            "parameter_sha256": digest,
            "algorithm_source_sha256": algorithm,
            "algorithm_revision": ALGORITHM_REVISION,
        }
    }
    for path in (
        tmp_path / "flags" / "v3-beta" / "manifest.json",
        tmp_path / "derived" / "smooth_model" / "v3-beta" / "manifest.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"files": record}))
    assert _resume_files(tmp_path, "v3-beta", "v3-beta", [fname], digest) == {
        fname
    }
    with pytest.raises(ValueError, match="another parameter set"):
        _resume_files(
            tmp_path,
            "v3-beta",
            "v3-beta",
            [fname],
            "different",
        )


def test_resume_accepts_known_writer_only_legacy_source(tmp_path):
    fname = "corr_20260716_000000Z.h5"
    digest = parameter_sha256(RFIConfig())
    legacy = next(iter(LEGACY_COMPATIBLE_SOURCE_SHA256))
    record = {
        fname: {
            "parameter_sha256": digest,
            "algorithm_source_sha256": legacy,
        }
    }
    for path in (
        tmp_path / "flags" / "v3-beta" / "manifest.json",
        tmp_path / "derived" / "smooth_model" / "v3-beta" / "manifest.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"files": record}))
    assert _resume_files(tmp_path, "v3-beta", "v3-beta", [fname], digest) == {
        fname
    }
    record[fname]["algorithm_source_sha256"] = "0" * 64
    for path in (
        tmp_path / "flags" / "v3-beta" / "manifest.json",
        tmp_path / "derived" / "smooth_model" / "v3-beta" / "manifest.json",
    ):
        path.write_text(json.dumps({"files": record}))
    with pytest.raises(ValueError, match="different flagger source"):
        _resume_files(tmp_path, "v3-beta", "v3-beta", [fname], digest)
