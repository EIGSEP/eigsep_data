"""Command-line range runner for supported-DPSS ``v3-beta`` products."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

from .clock import to_unix_time
from .index import MetadataIndex
from .rfi_supported import (
    DEFAULT_VERSION,
    RFIConfig,
    run_selection,
    write_products,
)


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


def parser():
    out = argparse.ArgumentParser(
        description=(
            "Generate supported-DPSS v3-beta flags and smooth models for "
            "a range of complete EIGSEP correlator files."
        )
    )
    select = out.add_mutually_exclusive_group(required=True)
    out.add_argument("campaign", type=Path)
    select.add_argument("--files", nargs=2, metavar=("FIRST", "LAST"))
    select.add_argument("--time", nargs=2, metavar=("START", "END"))
    out.add_argument("--air-antenna", default="box-air")
    out.add_argument("--ground-antenna", default="box-gnd")
    out.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE"
    )
    out.add_argument("--flags-version", default=DEFAULT_VERSION)
    out.add_argument("--model-version", default=DEFAULT_VERSION)
    out.add_argument("--overwrite", action="store_true")
    out.add_argument("--dry-run", action="store_true")
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    config = config_with_overrides(args.set)
    index = MetadataIndex(args.campaign / "data")
    if args.files:
        selection = index.select(files=tuple(args.files))
    else:
        selection = index.select(
            time=tuple(to_unix_time(value) for value in args.time)
        )
    if selection.nrows == 0:
        raise ValueError("the requested range selected no rows")
    result = run_selection(
        selection,
        air_antenna=args.air_antenna,
        ground_antenna=args.ground_antenna,
        config=config,
    )
    summary = {
        "rows": len(result.times),
        "channels": len(result.freqs_mhz),
        "files": selection.files,
        "excluded_fraction": float(result.mask.mean()),
        "rfi_fraction": float(result.rfi_mask.mean()),
        "supported_fraction": float(result.support_ok.mean()),
    }
    if not args.dry_run:
        written = write_products(
            result,
            args.campaign,
            flags_version=args.flags_version,
            model_version=args.model_version,
            overwrite=args.overwrite,
        )
        summary["written"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in written.items()
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
