"""Planning and resume tests for the v3-beta campaign runner."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from eigsep_data.rfi_run import (
    _output_policy_conflicts,
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
    assert _resume_files(
        tmp_path, "v3-beta", "v3-beta", [fname], digest, None
    ) == {fname}
    with pytest.raises(ValueError, match="another parameter set"):
        _resume_files(
            tmp_path,
            "v3-beta",
            "v3-beta",
            [fname],
            "different",
            None,
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
    assert _resume_files(
        tmp_path, "v3-beta", "v3-beta", [fname], digest, None
    ) == {fname}
    record[fname]["algorithm_source_sha256"] = "0" * 64
    for path in (
        tmp_path / "flags" / "v3-beta" / "manifest.json",
        tmp_path / "derived" / "smooth_model" / "v3-beta" / "manifest.json",
    ):
        path.write_text(json.dumps({"files": record}))
    with pytest.raises(ValueError, match="different flagger source"):
        _resume_files(tmp_path, "v3-beta", "v3-beta", [fname], digest, None)


def test_existing_header_resolved_products_conflict_with_policy(tmp_path):
    fname = "corr_20260716_000000Z.h5"
    for path in (
        tmp_path / "flags" / "v3-beta" / "manifest.json",
        tmp_path / "derived" / "smooth_model" / "v3-beta" / "manifest.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"files": {fname: {}}}))

    conflicts = _output_policy_conflicts(
        tmp_path, "v3-beta", "v3-beta", "a" * 64
    )

    assert len(conflicts) == 2
    assert all(item["observed"] == [None] for item in conflicts)
    assert all(item["expected"] == "a" * 64 for item in conflicts)


def _runner_task(tmp_path, *, dry_run=False):
    return dict(
        data_dir=str(tmp_path),
        output_root=str(tmp_path),
        day="20260714",
        batches=[
            {"id": 1, "files": ["bad.h5"]},
            {"id": 2, "files": ["good.h5"]},
        ],
        resolution_policy=None,
        config=RFIConfig(),
        air_antenna="box-air",
        ground_antenna="box-gnd",
        flags_version="v3-beta.1",
        model_version="v3-beta.1",
        overwrite=False,
        dry_run=dry_run,
        failure_report_dir=None if dry_run else str(tmp_path / "reports"),
    )


def _runner_result():
    import numpy as np

    return SimpleNamespace(
        times=np.arange(2),
        freqs_mhz=np.arange(3),
        mask=np.zeros((2, 3), bool),
        rfi_mask=np.zeros((2, 3), bool),
        support_ok=np.ones((2, 3), bool),
    )


@pytest.mark.parametrize("stage", ["fit", "write"])
def test_failed_batch_is_logged_and_next_batch_written(
    tmp_path, monkeypatch, stage
):
    import eigsep_data.rfi_run as runner

    task = _runner_task(tmp_path)
    monkeypatch.setattr(
        runner,
        "MetadataIndex",
        lambda path: SimpleNamespace(select=lambda files: files[0]),
    )
    result = _runner_result()
    attempted = []
    written = []

    def fit(selection, **kwargs):
        attempted.append(selection)
        if selection == "bad.h5" and stage == "fit":
            raise ValueError("x values must be equally spaced")
        result.file = selection
        return result

    def write(result, *args, **kwargs):
        if result.file == "bad.h5":
            raise OSError("write failed")
        written.append(result.file)

    monkeypatch.setattr(runner, "run_selection", fit)
    monkeypatch.setattr(runner, "write_products", write)
    output = runner._run_day(task)
    assert attempted == ["bad.h5", "good.h5"]
    assert written == ["good.h5"]
    assert [item["files"] for item in output["batches"]] == [["good.h5"]]
    failure = output["failures"][0]
    assert failure["stage"] == stage
    assert failure["files"] == ["bad.h5"]
    assert "Traceback" in failure["traceback"]
    assert json.loads(
        (tmp_path / "reports" / "20260714.jsonl").read_text()
    ) == json.loads(json.dumps(failure))
    # Failure records never enter the success manifests or count as resumable.
    assert (
        runner._resume_files(
            tmp_path,
            "v3-beta.1",
            "v3-beta.1",
            ["bad.h5"],
            parameter_sha256(RFIConfig()),
            None,
        )
        == set()
    )


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("all_fail", [False, True])
def test_worker_failure_does_not_cancel_other_days(
    tmp_path,
    monkeypatch,
    capsys,
    workers,
    all_fail,
):
    from concurrent.futures import ThreadPoolExecutor
    import eigsep_data.rfi_run as runner

    batches = [
        dict(id=i, files=[f"{i}.h5"], day=f"2026071{i+4}") for i in range(2)
    ]
    selection = SimpleNamespace(files=["0.h5", "1.h5"])
    monkeypatch.setattr(
        runner,
        "MetadataIndex",
        lambda path: SimpleNamespace(select=lambda **kwargs: selection),
    )
    monkeypatch.setattr(runner, "_selection", lambda *args: selection)
    monkeypatch.setattr(
        runner, "build_batches", lambda *args, **kwargs: batches
    )
    # Exercise the futures collection path deterministically without a process
    # importing pytest's monkeypatches; actual day work is tested separately.
    monkeypatch.setattr(
        runner,
        "ProcessPoolExecutor",
        lambda max_workers, mp_context: ThreadPoolExecutor(max_workers),
    )
    visited = []

    def day(task):
        visited.append(task["day"])
        if all_fail or task["day"] == "20260714":
            raise RuntimeError("worker initialization failed")
        return dict(
            day=task["day"],
            failures=[],
            batches=[
                dict(
                    files=["1.h5"],
                    rows=2,
                    cells=6,
                    excluded=0,
                    rfi=0,
                    supported=6,
                )
            ],
        )

    monkeypatch.setattr(runner, "_run_day", day)
    runner.main(
        [
            "--data-dir",
            str(tmp_path),
            "--all",
            "--header-resolution",
            "--workers",
            str(workers),
            "--dry-run",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert sorted(visited) == ["20260714", "20260715"]
    assert output["status"] == "completed with failures"
    assert output["failed_files"] == (2 if all_fail else 1)
    assert output["processed_files"] == (0 if all_fail else 1)
    assert output["excluded_fraction"] == (None if all_fail else 0)
    assert (
        list(tmp_path.iterdir()) == []
    )  # Dry run writes no reports/products.
