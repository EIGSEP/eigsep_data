"""Helpers shared by the calibration-product builders: configs, checksums,
code provenance."""

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(path):
    """Commit, branch and dirtiness of the checkout holding ``path``.

    Dirtiness is checked in ``path``'s own directory. An installed (not
    editable) package is not in a checkout; that is recorded instead."""

    def run(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True
        ).stdout.strip()

    commit = run("rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "note": f"{path} is not in a git checkout"}
    return {
        "commit": commit,
        "branch": run("branch", "--show-current"),
        "dirty": bool(
            run("status", "--porcelain", "--untracked-files=no", "--", ".")
        ),
    }


def code_provenance():
    """Commits of the packages a calibration product is built with."""
    import eigsep_cal

    import eigsep_data

    return {
        "eigsep_data": git_rev(Path(eigsep_data.__file__).parent),
        "eigsep_cal": git_rev(Path(eigsep_cal.__file__).parent),
    }


def unix(iso):
    return pd.Timestamp(iso).timestamp()
