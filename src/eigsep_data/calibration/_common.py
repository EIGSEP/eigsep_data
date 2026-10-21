"""Helpers shared by the calibration-product builders: configs and code
provenance."""

import json
from pathlib import Path

import pandas as pd

from ..provenance import package_rev, sha256

#: Each product version's choices live beside it, in the campaign tree.
CONFIG_NAME = "config.json"


def product_dir(root, product, version):
    return Path(root) / "derived" / product / version


def load_config(root, product, version, path=None):
    """The build choices of ``derived/<product>/<version>/``.

    ``path`` overrides the default, ``<that directory>/config.json``. Returns
    the parsed config and the file it came from."""
    path = (
        Path(path)
        if path
        else product_dir(root, product, version) / CONFIG_NAME
    )
    if not path.is_file():
        raise SystemExit(
            f"no build config at {path}: copy the previous version's "
            f"{CONFIG_NAME} there and edit it"
        )
    return json.loads(path.read_text()), path


def config_record(path, root):
    """The manifest's record of a config: its path relative to the campaign
    tree ``root`` when it is inside it, and its SHA-256."""
    p, r = Path(path).resolve(), Path(root).resolve()
    return {
        "path": str(p.relative_to(r)) if p.is_relative_to(r) else str(path),
        "sha256": sha256(path),
    }


def code_provenance():
    """Commits of the packages a calibration product is built with."""
    import eigsep_cal

    import eigsep_data

    return {
        "eigsep_data": package_rev(eigsep_data),
        "eigsep_cal": package_rev(eigsep_cal),
    }


def unix(iso):
    return pd.Timestamp(iso).timestamp()
