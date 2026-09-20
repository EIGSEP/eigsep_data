"""Explicit physical-antenna resolution rules for campaign data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class AntennaResolutionError(ValueError):
    """A policy rejected or could not uniquely resolve one raw file."""


@dataclass(frozen=True)
class PolicyDecision:
    """The unique policy rule applying to one raw file."""

    name: str
    status: str
    reason: str | None
    inputs: dict
    crosses: tuple


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


class AntennaResolutionPolicy:
    """Validated, hash-addressed rules mapping physical antennas to keys.

    Rules match file-level columns from :class:`MetadataIndex`. Exactly one
    rule must match every file under consideration. A ``use`` rule declares
    auto-input and cross-product keys; a ``reject`` rule gives the reason the
    file is outside the policy's approved coverage.
    """

    schema_version = 1

    def __init__(self, document, *, source=None, encoded=None):
        self.document = document
        self.source = None if source is None else str(Path(source).resolve())
        self._validate()
        if encoded is None:
            encoded = json.dumps(
                document, sort_keys=True, separators=(",", ":")
            ).encode()
        self.sha256 = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def load(cls, value):
        """Load a policy object, JSON path, or already-decoded mapping."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value)
        path = Path(value).expanduser().resolve()
        encoded = path.read_bytes()
        return cls(json.loads(encoded), source=path, encoded=encoded)

    @property
    def name(self):
        return self.document["name"]

    @property
    def rules(self):
        return self.document["rules"]

    def _validate(self):
        doc = self.document
        if not isinstance(doc, dict):
            raise TypeError("antenna resolution policy must be a JSON object")
        required = {"schema_version", "name", "rules"}
        missing = required - set(doc)
        if missing:
            raise ValueError(f"antenna policy lacks fields {sorted(missing)}")
        if doc["schema_version"] != self.schema_version:
            raise ValueError(
                f"unsupported antenna policy schema {doc['schema_version']}; "
                f"expected {self.schema_version}"
            )
        if not isinstance(doc["name"], str) or not doc["name"]:
            raise ValueError("antenna policy name must be a nonempty string")
        if not isinstance(doc["rules"], list) or not doc["rules"]:
            raise ValueError("antenna policy rules must be a nonempty list")
        names = set()
        for rule in doc["rules"]:
            for field in ("name", "status", "match"):
                if field not in rule:
                    raise ValueError(f"antenna policy rule lacks {field!r}")
            if rule["name"] in names:
                raise ValueError(
                    f"duplicate antenna policy rule {rule['name']!r}"
                )
            names.add(rule["name"])
            if rule["status"] not in {"use", "reject"}:
                raise ValueError(
                    f"rule {rule['name']!r} status must be 'use' or 'reject'"
                )
            if not isinstance(rule["match"], dict) or not rule["match"]:
                raise ValueError(f"rule {rule['name']!r} has no match fields")
            if rule["status"] == "reject":
                if not rule.get("reason"):
                    raise ValueError(
                        f"reject rule {rule['name']!r} requires a reason"
                    )
                continue
            inputs = rule.get("inputs")
            crosses = rule.get("crosses")
            if not isinstance(inputs, dict) or not inputs:
                raise ValueError(f"use rule {rule['name']!r} requires inputs")
            if not isinstance(crosses, list):
                raise TypeError(f"use rule {rule['name']!r} requires crosses")
            for entry in crosses:
                antennas = entry.get("antennas")
                if not isinstance(antennas, list) or len(antennas) != 2:
                    raise ValueError(
                        f"rule {rule['name']!r} cross requires two antennas"
                    )
                if not isinstance(entry.get("conjugated"), bool):
                    raise TypeError(
                        f"rule {rule['name']!r} cross requires conjugated bool"
                    )
                if not isinstance(entry.get("key"), str):
                    raise TypeError(
                        f"rule {rule['name']!r} cross requires a string key"
                    )

    @staticmethod
    def _row_value(row, name):
        try:
            return _plain(row[name])
        except (KeyError, TypeError) as exc:
            raise AntennaResolutionError(
                f"antenna policy match requires metadata column {name!r}"
            ) from exc

    def decision(self, row, *, fname=None):
        """Return the one rule matching a file-level metadata row."""
        matches = []
        for rule in self.rules:
            if all(
                self._row_value(row, field) == expected
                for field, expected in rule["match"].items()
            ):
                matches.append(rule)
        label = fname or self._row_value(row, "file")
        if not matches:
            raise AntennaResolutionError(
                f"{label}: antenna policy {self.name!r} has no matching rule"
            )
        if len(matches) != 1:
            names = [rule["name"] for rule in matches]
            raise AntennaResolutionError(
                f"{label}: antenna policy {self.name!r} matches multiple "
                f"rules {names}"
            )
        rule = matches[0]
        return PolicyDecision(
            name=rule["name"],
            status=rule["status"],
            reason=rule.get("reason"),
            inputs=dict(rule.get("inputs", {})),
            crosses=tuple(rule.get("crosses", ())),
        )

    def partition(self, meta):
        """Return approved filenames and explicit rejection records."""
        approved = []
        rejected = []
        decisions = {}
        for fname, rows in meta.groupby("file", sort=False):
            decision = self.decision(rows.iloc[0], fname=fname)
            decisions[str(fname)] = decision.name
            if decision.status == "use":
                approved.append(str(fname))
            else:
                rejected.append(
                    {
                        "file": str(fname),
                        "rule": decision.name,
                        "reason": decision.reason,
                        "rows": len(rows),
                    }
                )
        return approved, rejected, decisions

    @staticmethod
    def _require_available(fname, key, available, decision):
        if key not in available:
            raise AntennaResolutionError(
                f"{fname}: policy rule {decision.name!r} selects key {key!r}, "
                f"but available keys are {sorted(available)}"
            )

    def resolve_auto(self, row, antenna, available, *, fname=None):
        decision = self.decision(row, fname=fname)
        label = fname or self._row_value(row, "file")
        if decision.status == "reject":
            raise AntennaResolutionError(
                f"{label}: policy rule {decision.name!r} rejects the file: "
                f"{decision.reason}"
            )
        key = decision.inputs.get(antenna)
        if key is None:
            raise AntennaResolutionError(
                f"{label}: policy rule {decision.name!r} does not map "
                f"antenna {antenna!r}"
            )
        key = str(key)
        self._require_available(label, key, available, decision)
        return key, decision.name

    def resolve_cross(self, row, pair, available, *, fname=None):
        decision = self.decision(row, fname=fname)
        label = fname or self._row_value(row, "file")
        if decision.status == "reject":
            raise AntennaResolutionError(
                f"{label}: policy rule {decision.name!r} rejects the file: "
                f"{decision.reason}"
            )
        pair = tuple(pair)
        for entry in decision.crosses:
            declared = tuple(entry["antennas"])
            if pair == declared:
                key = entry["key"]
                conjugated = entry["conjugated"]
            elif pair == declared[::-1]:
                key = entry["key"]
                conjugated = not entry["conjugated"]
            else:
                continue
            self._require_available(label, key, available, decision)
            return key, conjugated, decision.name
        raise AntennaResolutionError(
            f"{label}: policy rule {decision.name!r} does not map cross {pair}"
        )

    def provenance(self):
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "source": self.source,
        }
