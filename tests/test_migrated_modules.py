"""The campaign's select_files/flagging code, after the 2026-09-19 move.

These modules came from ``marjum-2026-07/{curation,flagging}/`` where they
anchored on their own ``__file__``. In the package there is no such
anchor, so the contract is: nothing resolves a campaign path at import
time, and everything resolves it through :mod:`eigsep_data.paths` when
called.
"""

import subprocess
import sys

import pytest

import eigsep_data
from eigsep_data import paths


@pytest.fixture(autouse=True)
def clean_setting(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    paths.set_campaign_root(None)
    yield
    paths.set_campaign_root(None)


@pytest.fixture
def campaign_dir(tmp_path):
    root = tmp_path / "marjum-2026-07"
    (root / "data").mkdir(parents=True)
    (root / "curation").mkdir()
    return root


class TestImportDoesNotNeedACampaign:
    """Importing must not explode just because no campaign is set."""

    @pytest.mark.parametrize(
        "module",
        [
            "eigsep_data.select_files",
            "eigsep_data.flagging",
            "eigsep_data.flagging.detectors",
            "eigsep_data.flagging.build_masks",
            "eigsep_data.flagging.validate",
        ],
    )
    def test_module_imports_unconfigured(self, module):
        out = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, out.stderr

    def test_reachable_as_package_attributes(self):
        assert eigsep_data.select_files is not None
        assert eigsep_data.flagging.detectors is not None


class TestPathsComeFromTheSetting:
    def test_select_files_uses_the_configured_root(self, campaign_dir):
        from eigsep_data import select_files

        paths.set_campaign_root(campaign_dir)
        assert select_files._root() == campaign_dir
        assert select_files._data() == campaign_dir / "data"
        assert select_files._mode_table() == (
            campaign_dir / "curation" / "mode_table.jsonl"
        )

    def test_select_files_refuses_when_unconfigured(self):
        from eigsep_data import select_files

        with pytest.raises(RuntimeError, match="set_campaign_root"):
            select_files._root()

    def test_build_masks_and_validate_use_it_too(self, campaign_dir):
        from eigsep_data.flagging import build_masks, validate

        paths.set_campaign_root(campaign_dir)
        assert build_masks._root() == str(campaign_dir)
        assert validate._root() == str(campaign_dir)
        assert validate._data() == str(campaign_dir / "data")

    def test_select_root_argument_overrides(self, campaign_dir, tmp_path):
        """``root=`` is per-call and must not change the global setting."""
        from eigsep_data import select_files

        other = tmp_path / "other"
        (other / "data").mkdir(parents=True)
        paths.set_campaign_root(campaign_dir)
        # No corr files anywhere, so both return empty -- what is being
        # pinned is that `root=` is accepted and the setting survives.
        select_files.select(root=other)
        assert paths.get_campaign_root() == campaign_dir


class TestDetectorsAreSelfContained:
    """detectors.py is byte-identical to what produced flags/v0."""

    def test_resolves_no_paths_of_its_own(self):
        """Naming the campaign in prose is fine; resolving a path is not."""
        from eigsep_data.flagging import detectors

        with open(detectors.__file__) as f:
            text = f.read()
        assert "__file__" not in text
        assert "sys.path" not in text
        assert "/mnt/" not in text

    def test_band_constants_survived(self):
        from eigsep_data.flagging import detectors

        assert detectors.BAND_ANALYSIS == (45.0, 235.0)
        assert len(detectors.CATEGORY_NAMES) == 8


def test_detect_combs_is_known_broken():
    """Pre-existing NameError, imported with the move, not caused by it.

    ``CHANNEL_LOCKED_COMBS`` is defined nowhere in the fleet, so
    ``detect_combs`` raises on its first loop iteration. Only
    ``flagging.validate`` calls it; ``build_masks`` uses
    ``identify_combs`` and is unaffected. This test pins the bug so that
    fixing it is a deliberate act with a visible failure here.
    """
    import numpy as np

    from eigsep_data.flagging import detectors

    freqs = np.linspace(45.0, 235.0, 1024)
    with pytest.raises(NameError, match="CHANNEL_LOCKED_COMBS"):
        detectors.detect_combs(np.zeros(1024), freqs)


class TestGeometryRelease:
    """Release publishing resolves both roots at call time."""

    def test_imports_unconfigured(self):
        out = subprocess.run(
            [sys.executable, "-c", "import eigsep_data.geometry_release"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, out.stderr

    def test_terrain_defaults_beside_the_campaign(self, campaign_dir):
        from eigsep_data import geometry_release as gr

        paths.set_campaign_root(campaign_dir)
        assert gr._campaign() == campaign_dir
        assert gr._workspace() == campaign_dir.parent
        assert gr._terrain() == campaign_dir.parent / "terrain"
        assert gr._fit_root() == campaign_dir / "imgs" / "fits"

    def test_terrain_env_override(self, campaign_dir, tmp_path, monkeypatch):
        from eigsep_data import geometry_release as gr

        paths.set_campaign_root(campaign_dir)
        monkeypatch.setenv("EIGSEP_TERRAIN_ROOT", str(tmp_path / "elsewhere"))
        assert gr._terrain() == tmp_path / "elsewhere"

    def test_workspace_is_the_campaign_parent(self, campaign_dir):
        """Release provenance is recorded relative to this.

        v0001 recorded workspace-relative source paths; this must keep
        meaning the same thing or a later release stops comparing.
        """
        from eigsep_data import geometry_release as gr

        paths.set_campaign_root(campaign_dir)
        src = campaign_dir.parent / "terrain" / "meta.json"
        assert gr.relative(src) == "terrain/meta.json"
