"""One place to say where the campaign is; everything else reads it."""

import os
from pathlib import Path

import pytest

import eigsep_data
from eigsep_data import paths


@pytest.fixture(autouse=True)
def clean_setting(monkeypatch):
    """No test may leak a campaign root into the next one."""
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    paths.set_campaign_root(None)
    yield
    paths.set_campaign_root(None)


@pytest.fixture
def campaign_dir(tmp_path):
    root = tmp_path / "marjum-2026-07"
    (root / "data").mkdir(parents=True)
    return root


class TestSetCampaignRoot:
    def test_the_root_itself_is_taken_as_given(self, campaign_dir):
        assert paths.set_campaign_root(campaign_dir) == campaign_dir
        assert paths.get_campaign_root() == campaign_dir

    def test_the_data_dir_resolves_to_its_parent(self, campaign_dir):
        """People say "the data directory"; they mean the campaign."""
        assert paths.set_campaign_root(campaign_dir / "data") == campaign_dir

    def test_a_user_path_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "camp").mkdir()
        assert paths.set_campaign_root("~/camp") == tmp_path / "camp"

    def test_a_relative_path_is_made_absolute(self, campaign_dir, monkeypatch):
        monkeypatch.chdir(campaign_dir.parent)
        assert paths.set_campaign_root("marjum-2026-07") == campaign_dir

    def test_a_missing_directory_is_refused(self, tmp_path):
        with pytest.raises(NotADirectoryError, match="no campaign directory"):
            paths.set_campaign_root(tmp_path / "nope")

    def test_a_file_is_refused(self, tmp_path):
        f = tmp_path / "corr.h5"
        f.touch()
        with pytest.raises(NotADirectoryError):
            paths.set_campaign_root(f)

    def test_must_exist_false_allows_an_unstaged_campaign(self, tmp_path):
        missing = tmp_path / "not-here-yet"
        assert paths.set_campaign_root(missing, must_exist=False) == missing

    def test_none_clears_the_setting(self, campaign_dir):
        paths.set_campaign_root(campaign_dir)
        assert paths.set_campaign_root(None) is None
        assert paths.get_campaign_root() is None


class TestResolutionOrder:
    def test_the_setting_beats_the_environment(
        self, campaign_dir, monkeypatch
    ):
        monkeypatch.setenv(paths.ENV_VAR, "/from/the/env")
        paths.set_campaign_root(campaign_dir)
        assert paths.get_campaign_root() == campaign_dir

    def test_the_environment_beats_a_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "from-env"))
        got = paths.get_campaign_root(default=tmp_path / "fallback")
        assert got.name == "from-env"

    def test_the_environment_is_not_checked_for_existence(self, monkeypatch):
        """A wrong env value must surface where it is used, not here."""
        monkeypatch.setenv(paths.ENV_VAR, "/nowhere/at/all")
        assert paths.get_campaign_root() == Path("/nowhere/at/all")

    def test_an_empty_environment_value_is_ignored(self, monkeypatch):
        monkeypatch.setenv(paths.ENV_VAR, "")
        assert paths.get_campaign_root() is None

    def test_the_fallback_is_last(self, tmp_path):
        assert paths.get_campaign_root(default=tmp_path) == tmp_path

    def test_nothing_configured_is_none_by_default(self):
        assert paths.get_campaign_root() is None

    def test_required_raises_when_nothing_resolves(self):
        with pytest.raises(RuntimeError, match="set_campaign_root"):
            paths.get_campaign_root(required=True)


class TestCampaignDataDir:
    def test_it_is_the_data_subdirectory(self, campaign_dir):
        paths.set_campaign_root(campaign_dir)
        assert paths.campaign_data_dir() == campaign_dir / "data"

    def test_it_round_trips_from_the_data_dir(self, campaign_dir):
        paths.set_campaign_root(campaign_dir / "data")
        assert paths.campaign_data_dir() == campaign_dir / "data"

    def test_it_raises_when_unconfigured(self):
        with pytest.raises(RuntimeError):
            paths.campaign_data_dir()


class TestReachableFromThePackage:
    """The whole point is one call right after ``import eigsep_data``."""

    def test_the_setter_is_a_package_level_name(self, campaign_dir):
        assert eigsep_data.set_campaign_root(campaign_dir) == campaign_dir
        assert eigsep_data.get_campaign_root() == campaign_dir
        assert eigsep_data.campaign_data_dir() == campaign_dir / "data"

    def test_the_setter_shares_state_with_the_module(self, campaign_dir):
        eigsep_data.set_campaign_root(campaign_dir)
        assert paths.get_campaign_root() == campaign_dir

    def test_it_is_listed(self):
        assert "set_campaign_root" in dir(eigsep_data)

    def test_setting_it_costs_no_heavy_import(self):
        """Configuring a path must not drag JAX in behind it."""
        import subprocess
        import sys

        code = (
            "import sys, eigsep_data; "
            "eigsep_data.set_campaign_root('/tmp', must_exist=False); "
            "assert 'jax' not in sys.modules, sorted(sys.modules)[:0] or 'jax'"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, out.stderr


class TestCampaignUsesIt:
    def test_for_index_picks_up_the_setting(self, tmp_path, monkeypatch):
        from eigsep_data.bundle import Campaign

        class FakeIndex:
            data_dir = tmp_path / "somewhere" / "data"

        elsewhere = tmp_path / "configured"
        elsewhere.mkdir()
        paths.set_campaign_root(elsewhere)
        assert Campaign.for_index(FakeIndex()).root == elsewhere.resolve()

    def test_an_explicit_root_still_wins(self, tmp_path):
        from eigsep_data.bundle import Campaign

        class FakeIndex:
            data_dir = tmp_path / "somewhere" / "data"

        configured = tmp_path / "configured"
        configured.mkdir()
        explicit = tmp_path / "explicit"
        explicit.mkdir()
        paths.set_campaign_root(configured)
        got = Campaign.for_index(FakeIndex(), root=explicit)
        assert got.root == explicit.resolve()

    def test_the_data_dir_parent_is_still_the_last_resort(self, tmp_path):
        from eigsep_data.bundle import Campaign

        class FakeIndex:
            data_dir = tmp_path / "somewhere" / "data"

        assert Campaign.for_index(FakeIndex()).root == (
            tmp_path / "somewhere"
        ).resolve()


def test_no_module_level_state_leaks_between_processes():
    """``_ROOT`` is per-process; a fresh interpreter starts unset."""
    assert os.environ.get(paths.ENV_VAR) is None
    assert paths.get_campaign_root() is None
