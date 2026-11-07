"""Compose identified exclusions on exact native coordinates.

This opt-in component preserves a pinned v3 base and its known defects.
Its diagnostic writer cannot declare a complete v4 release. It does not
run a detector, infer switch state, fill data or change legacy defaults.
"""

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

BLOCKING_GATES = (
    "missing_switch_state_inference",
    "state_aware_liveness",
    "per_input_validity",
    "pair_evidence_split",
    "guard_review",
    "candidate_channel_footprint",
    "held_out_detector_validation",
    "consumer_policy_migration",
    "downstream_mask_and_noise_validation",
)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase sha256")
    return value


def _array(value, kind, name):
    a = np.array(value, copy=True)
    if a.ndim != 1 or not a.size or a.dtype.kind not in kind:
        raise ValueError(f"invalid {name} axis")
    if not np.isfinite(a).all() or len(np.unique(a)) != len(a):
        raise ValueError(f"nonfinite or duplicate {name}")
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class NativeFlags:
    """One file/input's flags, tied to native rows and physical frequencies.

    ``raw_sha256`` identifies the original file, not a selected export.
    Row numbers, not header times, establish integration identity. Time,
    exposure and switch-state arrays remain upstream coordinates joined
    through that identity; no timestamp is guessed from a filename here.
    """

    file: str
    input_key: str
    phase: str
    raw_sha256: str
    rows: np.ndarray
    channels: np.ndarray
    frequencies_mhz: np.ndarray
    flags: np.ndarray
    provenance: dict
    candidates: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if Path(self.file).name != self.file or not self.file:
            raise ValueError("file must be a native basename")
        if not re.fullmatch(r"[0-9]", self.input_key):
            raise ValueError("input_key must identify one input, not a pair")
        if self.phase not in {"A", "B", "C"}:
            raise ValueError("explicit wiring phase required")
        _sha(self.raw_sha256)
        for name, kind in (
            ("rows", "iu"),
            ("channels", "iu"),
            ("frequencies_mhz", "fiu"),
        ):
            object.__setattr__(
                self, name, _array(getattr(self, name), kind, name)
            )
        if (self.rows < 0).any() or (self.channels < 0).any():
            raise ValueError("native rows/channels must be nonnegative")
        if len(self.channels) != len(self.frequencies_mhz):
            raise ValueError("channel/frequency coordinates differ")
        bits = np.array(self.flags, copy=True)
        if (
            bits.dtype.kind != "u"
            or bits.size == 0
            or bits.shape != (len(self.rows), len(self.channels))
            or bits.max() > 65535
        ):
            raise ValueError("flags must be a native unsigned uint16 bitfield")
        bits = bits.astype(np.uint16)
        bits.flags.writeable = False
        object.__setattr__(self, "flags", bits)
        if not self.provenance:
            raise ValueError("source provenance required")
        # Reject unserializable/nonfinite provenance before any output writes.
        _json(self.provenance)

    @property
    def coordinates(self):
        """Serializable exact identity; masks do not enter this key."""
        return dict(
            file=self.file,
            input_key=self.input_key,
            phase=self.phase,
            raw_sha256=self.raw_sha256,
            rows=self.rows.tolist(),
            channels=self.channels.tolist(),
            frequencies_mhz=self.frequencies_mhz.tolist(),
        )

    @property
    def coordinate_sha256(self):
        return _digest(_json(self.coordinates))


@dataclass(frozen=True)
class FlagSchema:
    """Published reason definitions and named exclusion policies.

    Policies enumerate reason names. No consumer may assume that bit9/10,
    or any bit absent from a schema, is harmless. The ``rfi`` policy keeps
    non-sky/validity reasons distinct, so calibration rows can be inspected.
    This does not by itself validate calibration-row detector applicability.
    """

    reasons: tuple
    policies: dict
    base_definition: dict = field(default_factory=dict)

    def __post_init__(self):
        bits = [r["bit"] for r in self.reasons]
        names = [r["name"] for r in self.reasons]
        if (
            not bits
            or len(set(bits)) != len(bits)
            or len(set(names)) != len(names)
            or any(type(b) is not int or b < 0 or b > 15 for b in bits)
        ):
            raise ValueError("duplicate or invalid reason definitions")
        kinds = {"rfi", "validity", "selection", "advisory", "pair"}
        if any(r["kind"] not in kinds for r in self.reasons):
            raise ValueError("unknown reason category")
        if not self.policies or any(
            set(p) - set(names) for p in self.policies.values()
        ):
            raise ValueError("policy refers to unknown reasons")
        _json(self.to_dict())

    @classmethod
    def from_v3(cls, definition):
        """Pin the supplied released mapping; append the known EMI reason.

        This retains legacy cross_change flags as such. Splitting their
        pair evidence from input validity is a remaining release gate.
        """
        reasons = []
        for item in definition["bits"]:
            kind = (
                "advisory"
                if item.get("advisory")
                else (
                    "rfi"
                    if item["rfi"]
                    else "selection" if item["bit"] == 0 else "validity"
                )
            )
            reason = dict(item, kind=kind)
            if item["name"] == "cross_change":
                reason["evidence_scope"] = "legacy_pair_evidence_per_input"
            reasons.append(reason)
        if any(r["bit"] >= 10 for r in reasons):
            raise ValueError("expected pinned v3 bits0..9 before EMI addition")
        reasons.append(
            dict(
                bit=10,
                name="identified_self_emi",
                kind="rfi",
                value=1024,
                rfi=True,
                advisory=False,
            )
        )
        return cls(
            tuple(reasons),
            dict(
                sky=[r["name"] for r in reasons if r["kind"] != "advisory"],
                rfi=[r["name"] for r in reasons if r["kind"] == "rfi"],
            ),
            base_definition=json.loads(_json(definition)),
        )

    def to_dict(self):
        return dict(
            schema="native-flag-union-v1",
            dtype="uint16",
            reasons=list(self.reasons),
            policies=self.policies,
            base_definition=self.base_definition,
        )

    def validate(self, flags):
        flags = np.asarray(flags)
        if flags.dtype.kind != "u" or (flags.size and flags.max() > 65535):
            raise ValueError("unsigned uint16 flags required")
        known = sum(1 << r["bit"] for r in self.reasons)
        if np.any(flags.astype(np.uint64) & np.uint64(65535 ^ known)):
            raise ValueError("unknown bits in native flag payload")

    def reason(self, flags, name):
        self.validate(flags)
        matches = [r for r in self.reasons if r["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"unknown reason {name}")
        return (np.asarray(flags) & (1 << matches[0]["bit"])) != 0

    def excluded(self, flags, policy):
        """Decode exclusions using the named policy."""
        self.validate(flags)
        if policy not in self.policies:
            raise ValueError(f"unknown exclusion policy {policy}")
        value = sum(
            1 << r["bit"]
            for r in self.reasons
            if r["name"] in self.policies[policy]
        )
        return (np.asarray(flags) & value) != 0


@dataclass(frozen=True)
class MaskSource:
    """A reviewed pixel reason, with exact scope and source-byte provenance."""

    reason: str
    coordinates: NativeFlags
    mask: np.ndarray
    source_id: str
    source_sha256: str
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        _sha(self.source_sha256)
        a = np.array(self.mask, copy=True)
        if a.dtype != bool or a.shape != self.coordinates.flags.shape:
            raise ValueError(
                "source must give a boolean mask on all coordinates"
            )
        if not self.source_id:
            raise ValueError("source identifier required")
        a.flags.writeable = False
        object.__setattr__(self, "mask", a)
        _json(self.details)


def candidate_rows(block, name, rows, source_sha256):
    """Retain row-only evidence without inventing a channel footprint."""
    rows = np.asarray(rows)
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or len(np.unique(rows)) != len(rows)
        or not np.isin(rows, block.rows).all()
        or not name
    ):
        raise ValueError("invalid candidate native rows")
    return dict(
        name=name,
        coordinate_sha256=block.coordinate_sha256,
        source_sha256=_sha(source_sha256),
        rows=rows.tolist(),
        channel_footprint=None,
        pixel_exclusion=False,
    )


def compose(base, schema, sources, *, candidates=()):
    """Preserve all base reasons and OR reviewed reasons without resampling."""
    schema.validate(base.flags)
    out, records, seen = base.flags.copy(), [], set()
    for source in sources:
        if not isinstance(source, MaskSource):
            raise TypeError(
                "pixel exclusions require MaskSource, not evidence"
            )
        if source.coordinates.coordinate_sha256 != base.coordinate_sha256:
            raise ValueError("source coordinates differ from native base")
        if source.source_id in seen:
            raise ValueError("duplicate source identifier")
        seen.add(source.source_id)
        reasons = [r for r in schema.reasons if r["name"] == source.reason]
        if len(reasons) != 1 or reasons[0]["kind"] == "pair":
            raise ValueError("unknown or pair-only input exclusion reason")
        out[source.mask] |= np.uint16(1 << reasons[0]["bit"])
        records.append(
            dict(
                reason=source.reason,
                source_id=source.source_id,
                source_sha256=source.source_sha256,
                mask_sha256=_digest(source.mask.tobytes()),
                details=source.details,
            )
        )
    for item in candidates:
        if (
            item.get("coordinate_sha256") != base.coordinate_sha256
            or item.get("pixel_exclusion") is not False
            or item.get("channel_footprint") is not None
        ):
            raise ValueError("candidate evidence cannot set pixel exclusions")
    return replace(
        base,
        flags=out,
        candidates=base.candidates + tuple(candidates),
        provenance=dict(base=base.provenance, sources=records),
    )


@dataclass(frozen=True)
class SelfEMICuration:
    records: dict
    source: str
    sha256: str
    unlisted: str
    header: dict


def read_self_emi(path, *, unlisted="error"):
    """Read a pinned table, rejecting duplicates and ambiguous membership.

    ``unlisted='off'`` explicitly reproduces the legacy consumer's set
    membership convention; the default refuses undeclared file coverage.
    """
    if unlisted not in {"error", "off"}:
        raise ValueError("unlisted policy must be error or off")
    path = Path(path)
    raw = path.read_bytes()
    records, header = {}, {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) == {"provenance"}:
            if (
                header
                or row["provenance"].get("product") != "boxair_emi_presence"
            ):
                raise ValueError("ambiguous self-EMI provenance header")
            header = row["provenance"]
            continue
        if "file" not in row:
            raise ValueError("curation record lacks native file identity")
        name = row["file"]
        if name in records:
            raise ValueError("duplicate self-EMI curation file")
        if type(row.get("boxair_emi")) is not bool:
            raise ValueError("explicit boolean boxair_emi required")
        records[name] = row
    return SelfEMICuration(records, str(path), _digest(raw), unlisted, header)


def self_emi_mask(block, table):
    """Legacy phase-C/input4 file-wide strict integer-MHz +/-0.15MHz rule."""
    if block.phase != "C":
        raise ValueError("this legacy overlay is scoped to wiring phase C")
    if block.file not in table.records and table.unlisted == "error":
        raise ValueError(f"missing curation coverage for {block.file}")
    entry = table.records.get(block.file, {})
    if entry.get("phase", block.phase) != block.phase:
        raise ValueError("curation wiring phase differs from native input")
    channels = (
        np.abs(block.frequencies_mhz - np.round(block.frequencies_mhz)) < 0.15
    )
    on = block.input_key == "4" and entry.get("boxair_emi", False)
    mask = np.broadcast_to(channels & on, block.flags.shape).copy()
    return MaskSource(
        "identified_self_emi",
        block,
        mask,
        "boxair_emi_presence",
        table.sha256,
        dict(
            source=table.source,
            input_key="4",
            phase="C",
            row_scope="all native rows of marked file",
            frequency_rule="abs(f_MHz-round(f_MHz)) < 0.15",
            unlisted=table.unlisted,
            listed=bool(entry),
            curation_record=entry,
            curation_provenance=table.header,
        ),
    )


def write_diagnostic(directory, schema, blocks):
    """Write an immutable opt-in diagnostic in the existing day-file layout.

    This writer always records all unfinished C9 gates. It cannot publish
    a validated/released v4 or overwrite a version. ``identity`` groups
    supplement the ordinary ``mask`` layout with exact native coordinates.
    """
    directory = Path(directory)
    blocks = list(blocks)
    if not blocks:
        raise ValueError("no native coverage supplied")
    seen, frequencies, raw_hashes = set(), {}, {}
    for b in blocks:
        schema.validate(b.flags)
        if (b.file, b.input_key) in seen:
            raise ValueError("duplicate native file/input coverage")
        seen.add((b.file, b.input_key))
        if not re.fullmatch(r"corr_[0-9]{8}_[0-9]{6}Z.h5", b.file):
            raise ValueError("native filename lacks an unambiguous day")
        if not np.array_equal(b.rows, np.arange(len(b.rows))):
            raise ValueError("day-file layout requires complete native rows")
        day = b.file[5:13]
        if day in frequencies and not np.array_equal(
            frequencies[day], b.frequencies_mhz
        ):
            raise ValueError("day-file frequency coordinates differ")
        frequencies[day] = b.frequencies_mhz
        if b.file in raw_hashes and raw_hashes[b.file] != b.raw_sha256:
            raise ValueError("native file/input raw hashes differ")
        raw_hashes[b.file] = b.raw_sha256
    if not directory.name.endswith("-diagnostic"):
        raise ValueError("this component requires a diagnostic version name")
    directory.mkdir(parents=True, exist_ok=False)
    for b in blocks:
        path = directory / f"flags_{b.file[5:13]}.h5"
        with h5py.File(path, "a") as h:
            if "freqs_mhz" not in h:
                h.create_dataset("freqs_mhz", data=b.frequencies_mhz)
            ds = h.create_dataset(
                f"mask/{b.file}/{b.input_key}",
                data=b.flags,
                compression="gzip",
            )
            ds.attrs["source_sha256"] = b.raw_sha256
            identity = h.create_group(f"identity/{b.file}/{b.input_key}")
            identity.attrs["coordinate_sha256"] = b.coordinate_sha256
            identity.attrs["phase"] = b.phase
            identity.attrs["provenance_json"] = _json(b.provenance).decode()
            identity.attrs["candidates_json"] = _json(b.candidates).decode()
            identity.create_dataset("rows", data=b.rows)
            identity.create_dataset("channels", data=b.channels)
    schema_path = directory / "flag_schema.json"
    schema_path.write_bytes(_json(schema.to_dict()) + b"\n")
    bits = dict(
        dtype="uint16",
        bits=[dict(r, value=1 << r["bit"]) for r in schema.reasons],
    )
    (directory / "flag_bits.json").write_bytes(_json(bits) + b"\n")
    files = {p.name: _digest(p.read_bytes()) for p in directory.iterdir()}
    manifest = dict(
        product="flags",
        version=directory.name,
        status="diagnostic",
        schema="native-flag-union-v1",
        blocking_gates=list(BLOCKING_GATES),
        coverage=[
            dict(
                file=b.file,
                input_key=b.input_key,
                coordinate_sha256=b.coordinate_sha256,
            )
            for b in blocks
        ],
        files=files,
    )
    (directory / "manifest.json").write_bytes(_json(manifest) + b"\n")
    return manifest


def read_diagnostic(directory, file, input_key):
    """Read only verified explicit native coverage, with the stored policy."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema") != "native-flag-union-v1"
        or manifest.get("status") != "diagnostic"
    ):
        raise ValueError("not a native-union diagnostic manifest")
    coverage = [
        r
        for r in manifest["coverage"]
        if (r["file"], r["input_key"]) == (file, input_key)
    ]
    if len(coverage) != 1:
        raise KeyError("missing or ambiguous native file/input coverage")
    required = {"flag_schema.json", "flag_bits.json", f"flags_{file[5:13]}.h5"}
    if required - set(manifest["files"]):
        raise ValueError("missing required diagnostic file hashes")
    for name, digest in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("invalid manifest file identity")
        if _digest((directory / name).read_bytes()) != digest:
            raise ValueError(f"diagnostic hash mismatch: {name}")
    definition = json.loads((directory / "flag_schema.json").read_text())
    schema = FlagSchema(
        tuple(definition["reasons"]),
        definition["policies"],
        definition["base_definition"],
    )
    with h5py.File(directory / f"flags_{file[5:13]}.h5", "r") as h:
        ds = h[f"mask/{file}/{input_key}"]
        identity = h[f"identity/{file}/{input_key}"]
        block = NativeFlags(
            file=file,
            input_key=input_key,
            phase=identity.attrs["phase"],
            raw_sha256=ds.attrs["source_sha256"],
            rows=identity["rows"][:],
            channels=identity["channels"][:],
            frequencies_mhz=h["freqs_mhz"][:],
            flags=ds[:],
            provenance=json.loads(identity.attrs["provenance_json"]),
            candidates=tuple(json.loads(identity.attrs["candidates_json"])),
        )
        if (
            block.coordinate_sha256 != identity.attrs["coordinate_sha256"]
            or block.coordinate_sha256 != coverage[0]["coordinate_sha256"]
        ):
            raise ValueError("native coordinate identity differs")
    schema.validate(block.flags)
    return schema, block
