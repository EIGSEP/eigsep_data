"""Checksums and git state, for the provenance records that product
generators write into their manifests."""

import hashlib
import subprocess
from pathlib import Path


def sha256(path):
    """SHA-256 hex digest of the file at ``path``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(path, *args, check=False):
    """Stripped stdout of ``git -C <path> <args>``.

    With ``check`` a failing command raises; without it, the result is ``""``.
    """
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=check,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_rev(path):
    """Commit, branch and dirtiness of the checkout holding ``path``.

    Dirtiness is checked in ``path``'s own directory. An installed (not
    editable) package is not in a checkout; that is recorded instead."""
    commit = git(path, "rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "note": f"{path} is not in a git checkout"}
    return {
        "commit": commit,
        "branch": git(path, "branch", "--show-current"),
        "dirty": bool(
            git(path, "status", "--porcelain", "--untracked-files=no", "--", ".")
        ),
    }


def package_rev(module):
    """:func:`git_rev` of an imported package's source directory."""
    return git_rev(Path(module.__file__).parent)
