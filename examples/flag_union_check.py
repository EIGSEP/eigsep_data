"""Reproduce the bounded real-mask union without shared writes or fitting."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zlib

import numpy as np

from eigsep_data.flag_union import (
    FlagSchema,
    NativeFlags,
    candidate_rows,
    compose,
    read_diagnostic,
    read_self_emi,
    self_emi_mask,
    write_diagnostic,
    calibration_comb_mask,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-rows", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("commit implementation and fixtures before execution")
    fixtures = sorted((root / "tests/data/flag_union").glob("*.h5.json"))
    if len(fixtures) != 3:
        raise ValueError("exactly three reviewed real fixtures required")
    candidate_table = None
    candidate_sha = None
    if args.candidate_rows:
        import pandas as pd

        candidate_table = pd.read_parquet(args.candidate_rows)
        candidate_sha = digest(args.candidate_rows)
        if not {"file", "row", "key"} <= set(candidate_table.columns):
            raise ValueError("candidate table lacks native row identities")
        if candidate_table.duplicated(["file", "row", "key"]).any():
            raise ValueError("duplicate native candidate identity")
    args.output.mkdir(parents=True, exist_ok=False)
    rows, blocks, inputs, entries = [], [], {}, []
    packets = [json.loads(p.read_text()) for p in fixtures]
    curation_hashes = {p["curation_sha256"] for p in packets}
    if len(curation_hashes) != 1:
        raise ValueError("fixture curation source versions differ")
    curation_path = args.output / "exact_fixture_curation.jsonl"
    curation_path.write_text(
        json.dumps(
            {
                "provenance": dict(
                    product="boxair_emi_presence",
                    original_sha256=next(iter(curation_hashes)),
                    extraction=(
                        "exact three fixture file records; "
                        "no inferred records"
                    ),
                )
            }
        )
        + "\n"
        + "\n".join(json.dumps(p["curation_entry"]) for p in packets)
        + "\n"
    )
    table = read_self_emi(curation_path)
    for fixture, packet in zip(fixtures, packets):
        inputs[str(fixture)] = digest(fixture)
        raw = zlib.decompress(base64.b85decode(packet["flags_zlib_b85"]))
        if hashlib.sha256(raw).hexdigest() != packet["flags_sha256"]:
            raise ValueError("fixture flag bytes changed")
        flags = np.frombuffer(raw, dtype="<u2").reshape(packet["shape"])
        freq = np.asarray(packet["frequencies_mhz"], dtype="<f8")
        if (
            hashlib.sha256(freq.tobytes()).hexdigest()
            != packet["frequencies_sha256"]
        ):
            raise ValueError("fixture frequencies changed")
        schema = FlagSchema.from_v3(packet["flag_bits"])
        base = NativeFlags(
            file=packet["file"],
            input_key="4",
            phase="C",
            raw_sha256=packet["released_attributes"]["source_sha256"],
            rows=np.arange(flags.shape[0]),
            channels=np.rint(freq / (250 / 1024)).astype(int),
            frequencies_mhz=freq,
            flags=flags,
            provenance=packet["released_attributes"],
        )
        candidates = []
        if candidate_table is not None:
            selected = candidate_table[
                (candidate_table.file == base.file)
                & (candidate_table.key.astype(str) == base.input_key)
            ]
            candidates = [
                candidate_rows(
                    base,
                    "calibration_comb_detection_rows",
                    selected.row.to_numpy(dtype=int),
                    candidate_sha,
                )
            ]
        sources = [self_emi_mask(base, table)]
        calibration = np.zeros(flags.shape, bool)
        if candidates:
            source = calibration_comb_mask(
                base, candidates[0]["rows"], candidate_sha
            )
            sources.append(source)
            calibration = source.mask
        result = compose(base, schema, sources, candidates=candidates)
        before = (flags & np.uint16(65535 ^ ((1 << 9) | (1 << 10)))) != 0
        emi = np.broadcast_to(
            np.array([abs(float(f) - round(float(f))) < 0.15 for f in freq]),
            flags.shape,
        )
        emi = emi & packet["curation_entry"]["boxair_emi"]
        after = schema.excluded(result.flags, "sky")
        np.testing.assert_array_equal(after, before | emi | calibration)
        rows.append(
            dict(
                file=base.file,
                input_key=base.input_key,
                shape=list(flags.shape),
                additional_exclusions=int((after & ~before).sum()),
                union_equal=True,
                native_identity_preserved=True,
                candidate_rows=len(candidates[0]["rows"]) if candidates else 0,
                candidate_pixel_exclusion=False,
                identified_calibration_comb_cells=int(calibration.sum()),
                calibration_comb_pixel_exclusion=True,
            )
        )
        blocks.append(result)
        entries.append((base.file, base.input_key))
    destination = args.output / "v4-c9-diagnostic"
    manifest = write_diagnostic(destination, schema, blocks)
    for block in blocks:
        loaded_schema, loaded = read_diagnostic(
            destination, block.file, block.input_key
        )
        np.testing.assert_array_equal(loaded.flags, block.flags)
        np.testing.assert_array_equal(
            loaded_schema.excluded(loaded.flags, "sky"),
            schema.excluded(block.flags, "sky"),
        )
    report = dict(
        label="diagnostic",
        scope="three released input4 masks; known calibration comb; no fit",
        commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        input_sha256=inputs,
        results=rows,
        candidate_table_sha256=candidate_sha,
        blocking_gates=manifest["blocking_gates"],
        versions={
            k: importlib.metadata.version(k)
            for k in ["numpy", "h5py", "pandas", "eigsep-data"]
        },
    )
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
