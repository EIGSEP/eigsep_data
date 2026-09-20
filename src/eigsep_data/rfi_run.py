"""Plan and run supported-DPSS ``v3-beta`` campaign products."""

from __future__ import annotations

# Each process owns the linear algebra for one campaign day. Prevent BLAS from
# multiplying that process count again; users can override these before launch.
import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields, replace
import json
import multiprocessing
from pathlib import Path
import re
import sys

import numpy as np

from .clock import to_unix_time
from .index import MetadataIndex
from .paths import campaign_data_dir, get_campaign_root
from .rfi_supported import (
    DEFAULT_VERSION,
    RFIConfig,
    parameter_sha256,
    run_selection,
    write_products,
)

_DAY = re.compile(r"^corr_(\d{8})")


def _value(field, text):
    current = getattr(RFIConfig(), field.name)
    if isinstance(current, bool):
        return text.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, tuple):
        return tuple(float(item) for item in text.split(","))
    return text


def config_with_overrides(items):
    """Return default configuration with ``NAME=VALUE`` overrides."""
    known = {field.name: field for field in fields(RFIConfig)}
    values = {}
    for item in items:
        name, separator, text = item.partition("=")
        if not separator or name not in known:
            raise ValueError(
                f"invalid parameter override {item!r}; expected one of "
                f"{sorted(known)} as NAME=VALUE"
            )
        values[name] = _value(known[name], text)
    return replace(RFIConfig(), **values)


def resolve_locations(campaign=None, data_dir=None, output_root=None):
    """Resolve raw input and product roots without requiring installation."""
    if campaign is not None and data_dir is not None:
        raise ValueError("give either CAMPAIGN or --data-dir, not both")
    if campaign is not None:
        supplied = Path(campaign).expanduser().resolve()
        if supplied.name == "data":
            raw = supplied
            root = supplied.parent
        else:
            root = supplied
            raw = root / "data"
    elif data_dir is not None:
        raw = Path(data_dir).expanduser().resolve()
        root = raw.parent
    else:
        root = get_campaign_root(required=True).resolve()
        raw = campaign_data_dir(required=True).resolve()
    if output_root is not None:
        root = Path(output_root).expanduser().resolve()
    if not raw.is_dir():
        raise NotADirectoryError(f"raw data directory does not exist: {raw}")
    return raw, root


def _file_day(fname, first_time):
    match = _DAY.match(fname)
    if match:
        return match.group(1)
    return (
        np.datetime64(int(first_time), "s")
        .astype("datetime64[D]")
        .astype(str)
        .replace("-", "")
    )


def build_batches(selection, files_per_batch=10, gap_s=600.0):
    """Group complete files by day and stable acquisition configuration."""
    if files_per_batch < 1:
        raise ValueError("files_per_batch must be positive")
    records = []
    for fname, rows in selection.meta.groupby("file", sort=False):
        times = rows.time_best.to_numpy(dtype=float)
        finite = times[np.isfinite(times)]
        if not finite.size:
            raise ValueError(f"{fname}: no finite time_best; cannot plan")
        dt = rows.integration_time.to_numpy(dtype=float)
        dt = float(np.nanmedian(dt))
        records.append(
            {
                "file": str(fname),
                "rows": int(len(rows)),
                "first_time": float(finite.min()),
                "last_time": float(finite.max()),
                "day": _file_day(str(fname), finite.min()),
                "integration_time": dt,
                "filter_phase": str(rows.filter_phase.iloc[0]),
                "data_keys": str(rows.data_keys.iloc[0]),
            }
        )
    blocks = []
    current = []
    for record in records:
        if current:
            previous = current[-1]
            changed = (
                record["day"] != previous["day"]
                or record["first_time"] - previous["last_time"] > gap_s
                or not np.isclose(
                    record["integration_time"],
                    previous["integration_time"],
                    rtol=1e-5,
                    atol=0,
                )
                or record["filter_phase"] != previous["filter_phase"]
                or record["data_keys"] != previous["data_keys"]
            )
            if changed:
                blocks.append(current)
                current = []
        current.append(record)
    if current:
        blocks.append(current)

    batches = []
    for block_id, block in enumerate(blocks):
        for offset in range(0, len(block), files_per_batch):
            chunk = block[offset : offset + files_per_batch]
            batches.append(
                {
                    "id": len(batches),
                    "block": block_id,
                    "day": chunk[0]["day"],
                    "files": [item["file"] for item in chunk],
                    "rows": sum(item["rows"] for item in chunk),
                    "start_unix": chunk[0]["first_time"],
                    "end_unix": chunk[-1]["last_time"],
                    "integration_time_s": chunk[0]["integration_time"],
                    "filter_phase": chunk[0]["filter_phase"],
                    "data_keys": chunk[0]["data_keys"],
                }
            )
    return batches


def _read_manifest(path):
    if not path.exists():
        return {}
    with open(path) as stream:
        return json.load(stream).get("files", {})


def _resume_files(root, flags_version, model_version, wanted, config_hash):
    flags = _read_manifest(root / "flags" / flags_version / "manifest.json")
    models = _read_manifest(
        root / "derived" / "smooth_model" / model_version / "manifest.json"
    )
    completed = set()
    partial = []
    for fname in wanted:
        pair = (flags.get(fname), models.get(fname))
        if pair[0] is None and pair[1] is None:
            continue
        if all(
            record is not None
            and record.get("parameter_sha256") == config_hash
            for record in pair
        ):
            completed.add(fname)
        else:
            partial.append(fname)
    if partial:
        preview = ", ".join(partial[:3])
        raise ValueError(
            "--resume found partial products or another parameter set for "
            f"{preview}; choose a new version or use --overwrite"
        )
    return completed


def _day_tasks(batches, **settings):
    days = {}
    for batch in batches:
        days.setdefault(batch["day"], []).append(batch)
    return [
        dict(settings, day=day, batches=value) for day, value in days.items()
    ]


def _run_day(task):
    index = MetadataIndex(Path(task["data_dir"]))
    summaries = []
    for batch in task["batches"]:
        selection = index.select(files=batch["files"])
        result = run_selection(
            selection,
            air_antenna=task["air_antenna"],
            ground_antenna=task["ground_antenna"],
            config=task["config"],
        )
        if not task["dry_run"]:
            write_products(
                result,
                task["output_root"],
                data_dir=task["data_dir"],
                flags_version=task["flags_version"],
                model_version=task["model_version"],
                overwrite=task["overwrite"],
            )
        summaries.append(
            {
                "id": batch["id"],
                "files": batch["files"],
                "rows": len(result.times),
                "channels": len(result.freqs_mhz),
                "cells": int(result.mask.size),
                "excluded": int(result.mask.sum()),
                "rfi": int(result.rfi_mask.sum()),
                "supported": int(result.support_ok.sum()),
            }
        )
    return {"day": task["day"], "batches": summaries}


def parser():
    out = argparse.ArgumentParser(
        description=(
            "Plan or generate supported-DPSS v3-beta flags and smooth models "
            "from complete EIGSEP correlator files."
        )
    )
    out.add_argument(
        "campaign",
        nargs="?",
        type=Path,
        help="campaign root or its data/ directory (configured root if omitted)",
    )
    out.add_argument(
        "--data-dir", type=Path, help="explicit raw data directory"
    )
    out.add_argument(
        "--output-root",
        type=Path,
        help="campaign root for flags/ and derived/ (default: data parent)",
    )
    select = out.add_mutually_exclusive_group(required=True)
    select.add_argument(
        "--all", action="store_true", help="select all indexed files"
    )
    select.add_argument("--files", nargs=2, metavar=("FIRST", "LAST"))
    select.add_argument("--time", nargs=2, metavar=("START", "END"))
    out.add_argument("--air-antenna", default="box-air")
    out.add_argument("--ground-antenna", default="box-gnd")
    out.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE"
    )
    out.add_argument("--flags-version", default=DEFAULT_VERSION)
    out.add_argument("--model-version", default=DEFAULT_VERSION)
    out.add_argument("--files-per-batch", type=int, default=10)
    out.add_argument("--batch-gap-s", type=float, default=600.0)
    out.add_argument("--workers", type=int, default=1)
    mode = out.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    out.add_argument(
        "--plan",
        action="store_true",
        help="print plan; do no fitting or writing",
    )
    out.add_argument(
        "--dry-run",
        action="store_true",
        help="fit selected data but do not write",
    )
    return out


def _selection(index, args):
    if args.files:
        selected = index.select(files=tuple(args.files))
    elif args.time:
        selected = index.select(
            time=tuple(to_unix_time(value) for value in args.time)
        )
    else:
        selected = index.select()
    if selected.nrows == 0:
        raise ValueError("the requested range selected no rows")
    # Time boundaries usually cut through endpoint files. Product writing has
    # a whole-file contract, so expand only those named files to all rows.
    return index.select(files=selected.files)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    data_dir, output_root = resolve_locations(
        args.campaign, args.data_dir, args.output_root
    )
    config = config_with_overrides(args.set)
    config_hash = parameter_sha256(config)
    index = MetadataIndex(data_dir)
    selected = _selection(index, args)
    original_files = selected.files
    completed = set()
    if args.resume:
        completed = _resume_files(
            output_root,
            args.flags_version,
            args.model_version,
            original_files,
            config_hash,
        )
    pending = [name for name in original_files if name not in completed]
    if pending:
        pending_selection = index.select(files=pending)
        batches = build_batches(
            pending_selection,
            files_per_batch=args.files_per_batch,
            gap_s=args.batch_gap_s,
        )
    else:
        batches = []

    plan = {
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "flags_version": args.flags_version,
        "model_version": args.model_version,
        "parameter_sha256": config_hash,
        "parameters": asdict(config),
        "selected_files": len(original_files),
        "completed_files": len(completed),
        "pending_files": sum(len(batch["files"]) for batch in batches),
        "workers_requested": args.workers,
        "worker_days": sorted({batch["day"] for batch in batches}),
        "batches": batches,
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if not batches:
        print(json.dumps(dict(plan, status="already complete"), indent=2))
        return

    tasks = _day_tasks(
        batches,
        data_dir=str(data_dir),
        output_root=str(output_root),
        air_antenna=args.air_antenna,
        ground_antenna=args.ground_antenna,
        config=config,
        flags_version=args.flags_version,
        model_version=args.model_version,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    results = []
    workers = min(args.workers, len(tasks))
    if workers == 1:
        for task in tasks:
            results.append(_run_day(task))
            print(f"finished day {task['day']}", file=sys.stderr, flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(_run_day, task): task["day"] for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"finished day {result['day']}",
                    file=sys.stderr,
                    flush=True,
                )
    summaries = [batch for day in results for batch in day["batches"]]
    cells = sum(item["cells"] for item in summaries)
    output = dict(plan)
    output.update(
        {
            "status": "dry run" if args.dry_run else "written",
            "workers_used": workers,
            "processed_files": sum(len(item["files"]) for item in summaries),
            "rows": sum(item["rows"] for item in summaries),
            "excluded_fraction": sum(item["excluded"] for item in summaries)
            / cells,
            "rfi_fraction": sum(item["rfi"] for item in summaries) / cells,
            "supported_fraction": sum(item["supported"] for item in summaries)
            / cells,
        }
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
