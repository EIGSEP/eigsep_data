"""The package must not pull JAX to read a flag mask."""

import subprocess
import sys

import pytest

import eigsep_data

PY_RUN = [sys.executable, "-c"]


def _run(code):
    out = subprocess.run(
        PY_RUN + [code], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


class TestNamesStillResolve:
    """Laziness is only acceptable if nothing about the API changed."""

    @pytest.mark.parametrize(
        "name",
        [
            "MetadataIndex",
            "Selection",
            "EigsepData",
            "Bundle",
            "Campaign",
            "load_bundle",
            "S11",
            "RawS11",
            "ImuCalibrator",
            "ImuDataset",
            "ImuSnapshot",
            "to_unix_time",
            "format_time",
        ],
    )
    def test_attribute_is_reachable(self, name):
        assert getattr(eigsep_data, name) is not None

    @pytest.mark.parametrize(
        "name", ["index", "bundle", "products", "clock", "metadata", "data"]
    )
    def test_submodule_is_reachable(self, name):
        assert getattr(eigsep_data, name) is not None

    def test_from_import_works(self):
        from eigsep_data import MetadataIndex, to_unix_time  # noqa: F401

    def test_dir_lists_the_public_names(self):
        listed = dir(eigsep_data)
        assert "MetadataIndex" in listed and "beam_sim" in listed

    def test_a_typo_is_still_an_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute"):
            eigsep_data.MetadatIndex


class TestJaxIsNotPulledByDefault:
    def test_bare_import_does_not_load_jax(self):
        # The regression this guards: beam_sim was imported eagerly in
        # __init__, so every caller paid ~7 s and a hard JAX dependency
        # to touch anything at all.
        out = _run(
            "import sys, eigsep_data; "
            "print('jax' in sys.modules)"
        )
        assert out.strip() == "False"

    def test_the_index_and_loader_do_not_load_jax(self):
        out = _run(
            "import sys, eigsep_data; "
            "eigsep_data.MetadataIndex; eigsep_data.load_bundle; "
            "eigsep_data.products; "
            "print('jax' in sys.modules)"
        )
        assert out.strip() == "False"

    def test_touching_a_jax_module_does_load_it(self):
        # The other half of the contract: lazy must not mean absent.
        out = _run(
            "import sys, eigsep_data; "
            "eigsep_data.hpm; print('jax' in sys.modules)"
        )
        assert out.strip() == "True"
