#!/usr/bin/env python3
"""Publish an immutable, linearly versioned Marjum geometry release.

The public sequence is ``imgs/fits/vNNNN_marjum_geometry/`` under the
campaign, strictly linear, never edited or renumbered once published.
This module is the only supported way to create one; it refuses to
replace an existing release or to skip a version.

Two roots are involved. The **campaign** supplies the archived images and
the transmitter product, and receives the release; it comes from
:mod:`eigsep_data.paths`. The **terrain** checkout supplies the fitted
geometry and its git provenance; it defaults to ``terrain/`` beside the
campaign, which is where it has always been, and can be named explicitly
with ``--terrain`` or ``EIGSEP_TERRAIN_ROOT``.

Provenance paths inside a release are recorded relative to the workspace
(the campaign's parent), unchanged from the releases already published --
this module moved out of ``marjum-2026-07/curation/`` on 2026-09-19 and
that move is deliberately not visible in what a release records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .paths import get_campaign_root

#: Set by ``--terrain`` / ``EIGSEP_TERRAIN_ROOT``; None means "beside the
#: campaign", which is where it has always been.
_TERRAIN = None


def _campaign() -> Path:
    """The campaign root, resolved at call time, never at import."""
    return get_campaign_root(required=True)


def _workspace() -> Path:
    """The checkout that holds the campaign and terrain side by side.

    Release provenance paths are recorded relative to this, so it must
    keep meaning what it meant when v0001 was published.
    """
    return _campaign().parent


def _terrain() -> Path:
    if _TERRAIN is not None:
        return Path(_TERRAIN)
    env = os.environ.get("EIGSEP_TERRAIN_ROOT")
    if env:
        return Path(env)
    return _workspace() / "terrain"


def _fit_root() -> Path:
    return _campaign() / "imgs" / "fits"


EXCLUDED = {
    "2219": "Borderline registration: 67.9 px raster-horizon RMS and sky NLL 9.56; not promoted.",
    "2225": "Rejected after registration entered a false terrain minimum.",
    "2227": "Rejected after registration entered a false terrain minimum.",
    "2228": "Rejected after registration entered a false terrain minimum.",
    "2230": "Rejected after registration entered a false terrain minimum.",
    "2241": "Rejected after registration entered a false terrain minimum.",
    "2242": "Rejected after registration entered a false terrain minimum.",
    "2243": "Rejected after registration entered a false terrain minimum.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_terrain()), *args], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(_workspace().resolve()))


def source_record(path: Path) -> dict:
    return {"path": relative(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="four-digit release version, for example v0001")
    parser.add_argument(
        "--transmitter-product", type=Path, default=None,
        help="default: <campaign>/curation/transmitter_position.json",
    )
    parser.add_argument(
        "--campaign", default=None,
        help="campaign root (or its data/ dir); overrides "
             "eigsep_data.set_campaign_root() and EIGSEP_CAMPAIGN_ROOT",
    )
    parser.add_argument(
        "--terrain", default=None,
        help="terrain checkout supplying the fitted geometry; default "
             "is terrain/ beside the campaign (or EIGSEP_TERRAIN_ROOT)",
    )
    args = parser.parse_args()

    global _TERRAIN
    if args.campaign:
        from .paths import set_campaign_root

        set_campaign_root(args.campaign)
    if args.terrain:
        _TERRAIN = args.terrain
    campaign = _campaign()
    terrain = _terrain()
    fit_root = _fit_root()
    if args.transmitter_product is None:
        args.transmitter_product = (
            campaign / "curation" / "transmitter_position.json"
        )
    if not re.fullmatch(r"v\d{4}", args.version):
        raise SystemExit("version must match vNNNN, for example v0001")
    sequence = int(args.version[1:])
    if sequence < 1:
        raise SystemExit("release numbering begins at v0001")

    release_id = f"{args.version}_marjum_geometry"
    destination = fit_root / release_id
    if destination.exists():
        raise SystemExit(f"refusing to replace immutable release: {destination}")
    prior = sorted(fit_root.glob("v[0-9][0-9][0-9][0-9]_marjum_geometry"))
    existing = [int(path.name[1:5]) for path in prior]
    if existing != list(range(1, len(existing) + 1)):
        raise SystemExit(f"existing release sequence is not contiguous: {existing}")
    expected = len(prior) + 1
    if sequence != expected:
        raise SystemExit(
            f"strict sequence violation: found {len(prior)} releases; next must be v{expected:04d}"
        )

    label_path = terrain / "meta.json"
    fit_path = terrain / "cv_transmitter_joint_v4" / "fit_transmitter.npz"
    antenna_path = terrain / "cv_antenna_repick_v1" / "fit_antenna.npz"
    tx_report_path = terrain / "cv_transmitter_joint_v4" / "report.json"
    tx_acceptance_path = terrain / "cv_transmitter_joint_v4" / "acceptance.json"
    sources = [
        label_path, fit_path, antenna_path, tx_report_path, tx_acceptance_path,
        args.transmitter_product, terrain / "marjum_camera.py",
        terrain / "marjum_bundle.py", terrain / "marjum_antenna_repick.py",
        terrain / "marjum_transmitter_ray_polish_v4.py",
        terrain / "marjum_dem_sw.npz", terrain / "marjum_2026_07_exif.npz",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise SystemExit("missing release inputs:\n" + "\n".join(missing))

    labels = json.loads(label_path.read_text())
    tx_product = json.loads(args.transmitter_product.read_text())
    tx_acceptance = json.loads(tx_acceptance_path.read_text())
    with np.load(fit_path, allow_pickle=True) as fit:
        keys = [str(key) for key in fit["keys"]]
        cameras = np.asarray(fit["cameras"], float)
        distortion = np.asarray(fit["distortion"], float)
        groups = np.asarray(fit["groups"], int)
        shapes = np.asarray(fit["shapes"], int)
        antenna = np.asarray(fit["antenna"], float)
        transmitter = np.asarray(fit["transmitter"], float)
        gps_bias = np.asarray(fit["gps_bias"], float)
        provenance = [str(value) for value in fit["camera_provenance"]]
        conditioned = {str(value) for value in fit["transmitter_conditioned_keys"]}

    sizes = {len(keys), len(cameras), len(distortion), len(groups), len(shapes), len(provenance)}
    if len(sizes) != 1:
        raise SystemExit("camera arrays have inconsistent lengths")
    image_ids = sorted(labels, key=int)
    if len(image_ids) != 37 or len(keys) != 29:
        raise SystemExit(f"expected 37 labels and 29 poses; found {len(image_ids)} and {len(keys)}")
    if set(image_ids) != set(keys) | set(EXCLUDED):
        raise SystemExit("labeled-image coverage does not equal usable plus excluded poses")

    image_hashes = {}
    for key in image_ids:
        image = campaign / "imgs" / f"IMG_{key}.HEIC"
        if not image.exists():
            raise SystemExit(f"missing archived source image: {image}")
        image_hashes[key] = sha256(image)

    label_rows = []
    for key in image_ids:
        row = {"image": f"IMG_{key}.HEIC", "image_sha256": image_hashes[key]}
        row.update(labels[key])
        label_rows.append(row)
    labels_product = {
        "schema_version": 1, "release": args.version,
        "pixel_coordinates": {
            "order": ["x", "y"], "origin": "lower-left of the processed image",
            "x_direction": "right", "y_direction": "up", "units": "pixels",
        },
        "counts": {
            "images": len(image_ids),
            "antenna_labels": sum("ant_px" in labels[key] for key in image_ids),
            "transmitter_labels": sum("transmitter_px" in labels[key] for key in image_ids),
        },
        "images": label_rows,
    }

    fit_index = {key: index for index, key in enumerate(keys)}
    camera_rows = []
    for key in image_ids:
        row = {
            "release": args.version, "image": f"IMG_{key}.HEIC",
            "image_sha256": image_hashes[key], "labels": labels[key],
        }
        if key in fit_index:
            index = fit_index[key]
            p = cameras[index]
            row.update({
                "pose_status": "usable",
                "camera": {
                    "position_enu_m": p[:3].tolist(),
                    "orientation_rad": {
                        "theta": float(p[3]), "phi": float(p[4]), "tilt": float(p[5]),
                    },
                    "focal_length_px": float(p[6]),
                    "radial_distortion": {
                        "k1": float(distortion[index, 0]), "k2": float(distortion[index, 1]),
                    },
                    "lens_group": int(groups[index]),
                    "image_shape_px": {
                        "height": int(shapes[index, 0]), "width": int(shapes[index, 1]),
                    },
                },
                "fit_provenance": provenance[index],
                "transmitter_conditioned": key in conditioned,
            })
        else:
            row.update({
                "pose_status": "excluded", "camera": None,
                "exclusion_reason": EXCLUDED[key], "transmitter_conditioned": False,
            })
        camera_rows.append(row)

    failing = [
        name for name, passed in tx_acceptance.get("checks", {}).items()
        if passed is not True
    ]
    shared = {
        "schema_version": 1, "release": args.version,
        "frame": {
            "horizontal_crs": "EPSG:6341", "axes": ["east", "north", "up"],
            "units": "metres",
            "working_grid": "anchored by terrain/marjum_bundle.working_grid",
            "vertical_datum_status": "not independently verified",
        },
        "antenna_91m_era": {
            "status": "provisional deterministic fit", "position_enu_m": antenna.tolist(),
            "source": relative(antenna_path), "source_sha256": sha256(antenna_path),
            "uncertainty": None,
            "warning": "The completed legacy MCMC pilot is unconverged and supplies no uncertainty interval.",
        },
        "transmitter": {
            "working_candidate": {
                "status": "candidate; acceptance failed", "position_enu_m": transmitter.tolist(),
                "source": relative(fit_path), "source_sha256": sha256(fit_path),
                "accepted": bool(tx_acceptance.get("accepted", False)), "failing_checks": failing,
            },
            "recommended_for_propagation": {
                "position_enu_m": tx_product["best_estimate_enu_m"],
                "bound_m": tx_product["uncertainty"]["bound_m"],
                "bound_kind": tx_product["uncertainty"]["kind"],
                "stamp": tx_product["stamp"], "source": relative(args.transmitter_product),
                "source_sha256": sha256(args.transmitter_product),
            },
        },
        "gps_bias_enu_m": gps_bias.tolist(),
    }

    readme = f"""# {release_id}

This is immutable high-level geometry release `{args.version}` for the Marjum
Pass 2026 campaign. It contains the current labels for all 37 archived images,
29 usable camera solutions, eight explicitly excluded pose attempts, the 91 m
era antenna solution, and transmitter products.

The release is **candidate** because the selected transmitter fit fails
`2211_transmitter_reprojection`. Use `recommended_for_propagation` in
`shared.json`; its bound is not a posterior sigma.

- `manifest.json`: release identity, schemas, source hashes, and provenance.
- `labels.json`: immutable snapshot of antenna/transmitter pixel labels.
- `cameras.jsonl`: one row for each image, including excluded views.
- `shared.json`: antenna, transmitter, coordinate frame, and GPS bias.

Versions advance only as `vNNNN`; see the parent `README.md` and workspace
`AGENTS.md`. Never edit this directory after publication.
"""

    fit_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{release_id}.", dir=fit_root) as temp_name:
        temp = Path(temp_name)
        write_json(temp / "labels.json", labels_product)
        (temp / "cameras.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in camera_rows)
        )
        write_json(temp / "shared.json", shared)
        (temp / "README.md").write_text(readme)
        manifest = {
            "schema_version": 1, "release": args.version, "release_id": release_id,
            "sequence": sequence, "status": "candidate",
            "supersedes": prior[-1].name.split("_", 1)[0] if prior else None,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "campaign": "marjum-2026-07",
            "coverage": {"labeled_images": 37, "usable_camera_poses": 29, "excluded_camera_poses": 8},
            "camera_parameter_schema": {
                "vector_order": ["east_m", "north_m", "up_m", "theta_rad", "phi_rad", "tilt_rad", "focal_px"],
                "rotation": "body_to_ENU = Rz(phi) @ Ry(theta) @ Rz(tilt)",
                "camera_body_ray_before_rotation": "[cy-y, cx-x, focal_px]",
                "distortion": "q_distorted = q * (1 + k1*r^2 + k2*r^4), in focal-normalized coordinates",
            },
            "source_repository": {
                "path": "terrain", "commit": git_value("rev-parse", "HEAD"),
                "tracked_worktree_dirty": bool(git_value("status", "--porcelain", "--untracked-files=no")),
                "note": "File hashes pin corrected working sources that postdate the snapshot commit.",
            },
            "source_artifacts": [source_record(path) for path in sources],
            "release_artifacts": {
                name: {"sha256": sha256(temp / name)}
                for name in ["labels.json", "cameras.jsonl", "shared.json"]
            },
        }
        write_json(temp / "manifest.json", manifest)
        temp.rename(destination)
    print(f"published {destination}")


if __name__ == "__main__":
    main()
