"""Native identity and release-policy checks, including bounded real masks."""

import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
import pytest

from eigsep_data.flag_union import (
    FlagSchema,
    MaskSource,
    NativeFlags,
    candidate_rows,
    compose,
    read_diagnostic,
    read_self_emi,
    self_emi_mask,
    write_diagnostic,
    calibration_comb_mask,
)

import eigsep_data.flag_union as union

DATA = Path(__file__).parent / "data/flag_union"
SHA = "a" * 64


def _small():
    schema = FlagSchema.from_v3(
        {
            "bits": [
                dict(bit=0, name="non_sky", rfi=False, advisory=False),
                dict(bit=2, name="auto", rfi=True, advisory=False),
                dict(bit=9, name="scatter", rfi=False, advisory=True),
            ]
        }
    )
    block = NativeFlags(
        file="corr_20260716_011800Z.h5",
        input_key="4",
        phase="C",
        raw_sha256=SHA,
        rows=np.array([0, 1]),
        channels=np.array([3, 4]),
        frequencies_mhz=np.array([0.732421875, 0.9765625]),
        flags=np.array([[0, 512], [4, 0]], dtype=np.uint16),
        provenance={"source": "bounded fixture", "sha256": SHA},
    )
    return schema, block


def test_hard_bit12_is_decoded_by_policy_and_reasons_overlap():
    schema, base = _small()
    mask = np.array([[False, True], [True, False]])
    src = MaskSource("identified_self_emi", base, mask, "curation", SHA)
    out = compose(base, schema, [src])
    np.testing.assert_array_equal(out.flags, [[0, 4608], [4100, 0]])
    np.testing.assert_array_equal(base.flags, [[0, 512], [4, 0]])
    np.testing.assert_array_equal(schema.excluded(out.flags, "sky"), mask)
    assert schema.reason(out.flags, "scatter")[0, 1]
    assert schema.reason(out.flags, "auto")[1, 0]
    # Bits 10 and 11 stay unset: the RFI prototype lineage gives them
    # meanings (retracted, residual) that existing consumers keep.
    assert not (out.flags & np.uint16((1 << 10) | (1 << 11))).any()
    # The existing consumer constant (exclude all but bits 9 and 10)
    # excludes the new hard reason without change.
    legacy = (out.flags & np.uint16(~((1 << 9) | (1 << 10)) & 65535)) != 0
    np.testing.assert_array_equal(legacy, schema.excluded(out.flags, "sky"))
    assert not schema.excluded(np.array([[512]], np.uint16), "sky").any()
    np.testing.assert_array_equal(
        schema.excluded(np.array([[1, 4, 4096]], np.uint16), "rfi"),
        [[False, True, True]],
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", np.array([1, 0])),
        ("channels", np.array([4, 3])),
        ("frequencies_mhz", np.array([0.732421875, 0.99])),
        ("file", "corr_20260716_012224Z.h5"),
        ("input_key", "0"),
        ("raw_sha256", "b" * 64),
        ("phase", "B"),
    ],
)
def test_coordinate_mismatch_cannot_shift_exclusions(field, value):
    schema, base = _small()
    wrong = replace(base, **{field: value})
    source = MaskSource(
        "identified_self_emi", wrong, np.ones((2, 2), bool), "curation", SHA
    )
    with pytest.raises(ValueError, match="coordinates"):
        compose(base, schema, [source])


def test_unknown_bits_missing_policy_and_ambiguous_sources_fail():
    schema, base = _small()
    with pytest.raises(ValueError, match="unknown bits"):
        compose(
            replace(base, flags=np.full((2, 2), 1 << 15, np.uint16)),
            schema,
            [],
        )
    with pytest.raises(ValueError, match="policy"):
        schema.excluded(base.flags, "unpublished")
    with pytest.raises(ValueError, match="duplicate source"):
        src = MaskSource(
            "identified_self_emi", base, np.zeros((2, 2), bool), "same", SHA
        )
        compose(base, schema, [src, src])
    with pytest.raises(ValueError, match="rows"):
        replace(base, rows=np.array([0, 0]))


def test_self_emi_has_exact_native_scope_and_strict_edges(tmp_path):
    schema, base = _small()
    file = tmp_path / "curation.jsonl"
    file.write_text(
        json.dumps(dict(file=base.file, phase="C", boxair_emi=True)) + "\n"
    )
    table = read_self_emi(file)
    base = replace(
        base,
        channels=np.arange(4),
        frequencies_mhz=np.array([0.85, 1.0, 1.15, 1.2]),
        flags=np.zeros((2, 4), np.uint16),
    )
    out = compose(base, schema, [self_emi_mask(base, table)])
    # Floating-point coordinates use exactly the existing strict comparison.
    expected = np.array(
        [abs(float(f) - round(float(f))) < 0.15 for f in base.frequencies_mhz]
    )
    np.testing.assert_array_equal(
        schema.excluded(out.flags, "sky"), np.broadcast_to(expected, (2, 4))
    )
    other = replace(base, input_key="0")
    assert not self_emi_mask(other, table).mask.any()
    with pytest.raises(ValueError, match="missing curation"):
        self_emi_mask(replace(base, file="other.h5"), table)
    assert not self_emi_mask(
        replace(base, file="other.h5"), read_self_emi(file, unlisted="off")
    ).mask.any()
    with pytest.raises(ValueError, match="phase"):
        self_emi_mask(replace(base, phase="B"), table)
    file.write_text(file.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_self_emi(file)


def test_row_candidate_is_evidence_without_pixel_exclusion():
    schema, base = _small()
    record = candidate_rows(base, "calibration_comb", [1], SHA)
    out = compose(base, schema, [], candidates=[record])
    np.testing.assert_array_equal(out.flags, base.flags)
    assert out.candidates[0]["channel_footprint"] is None
    assert not out.candidates[0]["pixel_exclusion"]
    with pytest.raises(ValueError, match="candidate"):
        candidate_rows(base, "calibration_comb", [2], SHA)
    with pytest.raises(TypeError, match="MaskSource"):
        compose(base, schema, [record])


@pytest.mark.parametrize(
    "change",
    [
        {"rows": [-1, 999999]},
        {"rows": [0, 0]},
        {"rows": [0.5]},
        {"name": ""},
        {"source_sha256": "invalid"},
        {"coordinate_sha256": "b" * 64},
        {"pixel_exclusion": True},
    ],
)
def test_untrusted_candidate_records_fail_every_boundary(tmp_path, change):
    schema, base = _small()
    record = candidate_rows(base, "generic_candidate", [0], SHA)
    bad = dict(record, **change)
    with pytest.raises(ValueError):
        replace(base, candidates=(bad,))
    with pytest.raises(ValueError):
        compose(base, schema, [], candidates=[bad])
    inherited = replace(base, candidates=(dict(record),))
    inherited.candidates[0].update(change)
    with pytest.raises(ValueError):
        compose(inherited, schema, [])
    with pytest.raises(ValueError):
        write_diagnostic(tmp_path / "bad-diagnostic", schema, [inherited])
    out = compose(base, schema, [], candidates=[record])
    directory = tmp_path / "read-diagnostic"
    manifest = write_diagnostic(directory, schema, [out])
    import h5py

    day = directory / "flags_20260716.h5"
    with h5py.File(day, "a") as h:
        h[f"identity/{base.file}/4"].attrs["candidates_json"] = json.dumps(
            [bad]
        )
    manifest["files"][day.name] = hashlib.sha256(day.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_diagnostic(directory, base.file, "4")


def test_identified_comb_uses_known_rows_and_channels_with_own_reason():
    schema, base = _small()
    source = calibration_comb_mask(base, [0], SHA)
    out = compose(base, schema, [source])
    expected = np.array([[False, True], [False, False]])
    np.testing.assert_array_equal(
        schema.reason(out.flags, "identified_calibration_comb"), expected
    )
    assert not calibration_comb_mask(
        replace(base, input_key="0"), [0], SHA
    ).mask.any()
    assert source.details["switch_states"] == "all"
    with pytest.raises(ValueError):
        calibration_comb_mask(base, [2], SHA)


def test_bound_producer_source_runtime_and_historical_read(
    tmp_path, monkeypatch
):
    schema, base = _small()
    first = tmp_path / "first-diagnostic"
    manifest = write_diagnostic(first, schema, [base])
    original = json.loads((first / "producer.json").read_text())
    monkeypatch.setattr(
        union, "_EXECUTED_SOURCE", union._EXECUTED_SOURCE + b"\n# revision\n"
    )
    second = tmp_path / "second-diagnostic"
    write_diagnostic(second, schema, [base])
    changed = json.loads((second / "producer.json").read_text())
    assert changed["source_sha256"] != original["source_sha256"]
    # Historical source is verified against its own snapshot, not this reader.
    read_diagnostic(first, base.file, "4")
    record = dict(original)
    record["runtime"] = {}
    data = json.dumps(record).encode()
    (first / "producer.json").write_bytes(data)
    manifest["files"]["producer.json"] = hashlib.sha256(data).hexdigest()
    manifest["producer_sha256"] = hashlib.sha256(data).hexdigest()
    (first / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="producer"):
        read_diagnostic(first, base.file, "4")
    mpath = second / "manifest.json"
    missing = json.loads(mpath.read_text())
    missing.pop("producer_sha256")
    mpath.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="producer"):
        read_diagnostic(second, base.file, "4")


@pytest.mark.parametrize(
    "filename,extra",
    [
        ("corr_20260716_011800Z.h5", 6499),
        ("corr_20260717_085537Z.h5", 0),
    ],
)
def test_released_real_union_has_exact_pixel_set_and_roundtrip(
    tmp_path, filename, extra
):
    fixture = json.loads((DATA / (filename + ".json")).read_text())
    raw = zlib.decompress(base64.b85decode(fixture["flags_zlib_b85"]))
    assert hashlib.sha256(raw).hexdigest() == fixture["flags_sha256"]
    flags = np.frombuffer(raw, dtype="<u2").reshape(fixture["shape"])
    freq = np.array(fixture["frequencies_mhz"], dtype="<f8")
    assert (
        hashlib.sha256(freq.tobytes()).hexdigest()
        == fixture["frequencies_sha256"]
    )
    schema = FlagSchema.from_v3(fixture["flag_bits"])
    base = NativeFlags(
        file=filename,
        input_key="4",
        phase="C",
        raw_sha256=fixture["released_attributes"]["source_sha256"],
        rows=np.arange(flags.shape[0]),
        channels=np.rint(freq / (250 / 1024)).astype(int),
        frequencies_mhz=freq,
        flags=flags,
        provenance=fixture["released_attributes"],
    )
    curation = tmp_path / "curation.jsonl"
    curation.write_text(json.dumps(fixture["curation_entry"]) + "\n")
    out = compose(base, schema, [self_emi_mask(base, read_self_emi(curation))])
    # Independent legacy consumer oracle on actual native coordinates.
    legacy_bad = (flags & np.uint16(65535 ^ ((1 << 9) | (1 << 10)))) != 0
    emi = np.zeros(flags.shape, bool)
    for channel, value in enumerate(freq):
        if abs(float(value) - round(float(value))) < 0.15:
            emi[:, channel] = True
    actual = schema.excluded(out.flags, "sky")
    np.testing.assert_array_equal(actual, legacy_bad | emi)
    assert int((actual & ~legacy_bad).sum()) == extra
    # The existing consumer constant now excludes the new reasons too.
    legacy_new = (out.flags & np.uint16(65535 ^ ((1 << 9) | (1 << 10)))) != 0
    np.testing.assert_array_equal(legacy_new, actual)
    np.testing.assert_array_equal(out.flags & np.uint16(1023), flags)
    directory = tmp_path / "v4-c9-diagnostic"
    manifest = write_diagnostic(directory, schema, [out])
    assert manifest["status"] == "diagnostic"
    assert "state_aware_liveness" in manifest["blocking_gates"]
    loaded_schema, loaded = read_diagnostic(directory, filename, "4")
    np.testing.assert_array_equal(loaded.flags, out.flags)
    np.testing.assert_array_equal(
        loaded_schema.excluded(loaded.flags, "sky"), actual
    )
    assert loaded.coordinate_sha256 == out.coordinate_sha256
    with pytest.raises(FileExistsError):
        write_diagnostic(directory, schema, [out])
    with pytest.raises(ValueError, match="duplicate"):
        write_diagnostic(tmp_path / "dup", schema, [out, out])
    with pytest.raises(KeyError):
        read_diagnostic(directory, filename, "0")
    day = next(directory.glob("flags_*.h5"))
    with day.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        read_diagnostic(directory, filename, "4")
