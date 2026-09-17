"""Tests for eigsep_data.bundle: the raw-plus-companions join."""

import json

import h5py
import numpy as np
import pytest

from eigsep_data import MetadataIndex
from eigsep_data.bundle import Campaign, _resolve_key

from conftest import NCHAN, write_corr_file

ANTS = {"0": "box-gnd", "1": "box-gnd", "4": "box-air", "5": "box-air"}


@pytest.fixture
def campaign(tmp_path):
    """A campaign tree: two raw files, flags for both, a model for one.

    The model covers only the first file on purpose -- a companion that
    does not cover the whole window is the normal case in this campaign,
    not an error, and the bundle has to say so rather than quietly
    returning a shorter array.
    """
    data = tmp_path / "data"
    data.mkdir()
    names = ["corr_20260717_150041Z.h5", "corr_20260717_151041Z.h5"]
    for i, name in enumerate(names):
        write_corr_file(
            data / name,
            ntimes=6,
            keys=("0", "4"),
            sync_time=1.7843e9 + 600 * i,
            input_to_ant=ANTS,
            seed=i,
        )
    freqs = np.linspace(0, 250, NCHAN, endpoint=False)

    flags = tmp_path / "flags" / "v2"
    flags.mkdir(parents=True)
    with h5py.File(flags / "flags_20260717.h5", "w") as h:
        h.create_dataset("freqs_mhz", data=freqs)
        mask = h.create_group("mask")
        for j, name in enumerate(names):
            g = mask.create_group(name)
            for key in ("0", "4"):
                bits = np.zeros((6, NCHAN), dtype=np.uint16)
                bits[:, 10 + j] = 256  # the v2-only DPSS-outlier bit
                g.create_dataset(key, data=bits)
    (flags / "manifest.json").write_text(
        json.dumps({"provenance": {"product": "flags", "version": "v2"}})
    )

    band = slice(40, 200)
    model_dir = tmp_path / "derived" / "smooth_model" / "v0"
    model_dir.mkdir(parents=True)
    with h5py.File(model_dir / names[0], "w") as h:
        h.create_dataset("freqs_mhz", data=freqs[band])
        g = h.create_group("input_0")
        g.create_dataset(
            "model", data=np.full((6, band.stop - band.start), 7.0)
        )
    return tmp_path, names, freqs, band


class TestCampaignRoot:
    def test_defaults_to_the_data_dir_parent(self, campaign):
        root, _names, _f, _b = campaign
        index = MetadataIndex(root / "data", cache=False)
        assert Campaign.for_index(index).root == root.resolve()

    def test_env_override_wins(self, campaign, tmp_path, monkeypatch):
        root, _names, _f, _b = campaign
        index = MetadataIndex(root / "data", cache=False)
        monkeypatch.setenv("EIGSEP_CAMPAIGN_ROOT", str(tmp_path / "elsewhere"))
        assert Campaign.for_index(index).root.name == "elsewhere"

    def test_explicit_root_beats_the_env(self, campaign, monkeypatch):
        root, _names, _f, _b = campaign
        index = MetadataIndex(root / "data", cache=False)
        monkeypatch.setenv("EIGSEP_CAMPAIGN_ROOT", "/nowhere")
        assert Campaign.for_index(index, root=root).root == root.resolve()


class TestResolveKey:
    def test_reads_the_antenna_map_from_the_file(self, campaign):
        root, names, _f, _b = campaign
        path = root / "data" / names[0]
        assert _resolve_key(path, "box-gnd", {"0", "4"}) == "0"
        assert _resolve_key(path, "box-air", {"0", "4"}) == "4"

    def test_prefers_the_wired_input_over_its_mux_copy(self, campaign):
        # adc_mux copies input 0's antenna onto input 1, so box-gnd
        # names both; 0 is the source and is what should be read.
        root, names, _f, _b = campaign
        path = root / "data" / names[0]
        assert _resolve_key(path, "box-gnd", {"0", "1", "4"}) == "0"

    def test_ignores_a_key_the_file_does_not_carry(self, campaign):
        root, names, _f, _b = campaign
        path = root / "data" / names[0]
        assert _resolve_key(path, "box-gnd", {"4"}) is None

    def test_unknown_antenna_is_none_not_a_guess(self, campaign):
        root, names, _f, _b = campaign
        path = root / "data" / names[0]
        assert _resolve_key(path, "viv-N", {"0", "4"}) is None


class TestLoadBundle:
    def _bundle(self, campaign, **kw):
        root, _names, _f, _b = campaign
        index = MetadataIndex(root / "data", cache=False)
        kw.setdefault("antenna", "box-gnd")
        return index.select().load_bundle(root=root, **kw)

    def test_raw_rows_match_a_plain_load(self, campaign):
        root, _names, _f, _b = campaign
        index = MetadataIndex(root / "data", cache=False)
        plain = index.select().load(keys=["0"])
        bundle = index.select().load_bundle(antenna="box-gnd", root=root)
        np.testing.assert_array_equal(bundle.data, plain.data["0"])
        np.testing.assert_array_equal(bundle.t, plain.times)

    def test_antenna_and_key_are_mutually_exclusive(self, campaign):
        with pytest.raises(ValueError, match="exactly one"):
            self._bundle(campaign, key="0")

    def test_one_of_them_is_required(self, campaign):
        root, _n, _f, _b = campaign
        index = MetadataIndex(root / "data", cache=False)
        with pytest.raises(ValueError, match="exactly one"):
            index.select().load_bundle(root=root)

    def test_flags_keep_their_stored_dtype(self, campaign):
        bundle = self._bundle(campaign, products=["flags@v2"])
        assert bundle.flags.dtype == np.uint16
        assert bundle.flags.shape == bundle.data.shape

    def test_flags_land_on_the_rows_they_belong_to(self, campaign):
        # Each file marks a different channel; if the join were off by a
        # file the marks would land on the wrong rows.
        bundle = self._bundle(campaign, products=["flags@v2"])
        first = bundle.meta.file == bundle.meta.file.iloc[0]
        assert (bundle.flags[first.to_numpy(), 10] == 256).all()
        assert (bundle.flags[~first.to_numpy(), 11] == 256).all()

    def test_an_unrequested_product_is_a_clear_error(self, campaign):
        bundle = self._bundle(campaign)
        with pytest.raises(KeyError, match="products="):
            bundle.flags

    def test_band_is_the_intersection_with_the_products(self, campaign):
        # The model covers 40:200 of the 1024-channel axis; asking for
        # the whole band must narrow to what every product covers, not
        # pad the model out to the data's width.
        _root, _n, freqs, band = campaign
        bundle = self._bundle(campaign, products=["smooth_model@v0"])
        np.testing.assert_allclose(bundle.freqs_mhz, freqs[band])
        assert bundle.data.shape[1] == band.stop - band.start

    def test_residual_is_recomputed_not_read(self, campaign):
        bundle = self._bundle(campaign, products=["smooth_model@v0"])
        np.testing.assert_allclose(
            bundle.residual, bundle.data - bundle.smooth_model
        )

    def test_a_file_without_a_companion_is_nan_not_dropped(self, campaign):
        # Only the first file has a model. The second file's rows stay,
        # so every array keeps one row per integration and nothing
        # silently shifts; the gap is NaN and the file is named.
        bundle = self._bundle(campaign, products=["smooth_model@v0"])
        assert bundle.data.shape[0] == bundle.smooth_model.shape[0]
        covered = bundle.meta.file == bundle.meta.file.iloc[0]
        assert np.isfinite(bundle.smooth_model[covered.to_numpy()]).all()
        assert np.isnan(bundle.smooth_model[~covered.to_numpy()]).all()
        skipped = bundle.provenance["products"]["smooth_model"]["skipped"]
        assert skipped == [bundle.meta.file.iloc[-1]]

    def test_missing_raise_names_the_file(self, campaign):
        with pytest.raises(FileNotFoundError, match="corr_20260717_151041Z"):
            self._bundle(
                campaign, products=["smooth_model@v0"], missing="raise"
            )

    def test_provenance_records_the_versions_asked_for(self, campaign):
        bundle = self._bundle(
            campaign, products=["flags@v2", "smooth_model@v0"]
        )
        versions = {
            k: v["version"]
            for k, v in bundle.provenance["products"].items()
        }
        assert versions == {"flags": "v2", "smooth_model": "v0"}
        assert bundle.provenance["keys"] == ["0"]

    def test_manifest_travels_with_the_product(self, campaign):
        bundle = self._bundle(campaign, products=["flags@v2"])
        manifest = bundle.provenance["products"]["flags"]["manifest"]
        assert manifest["provenance"]["version"] == "v2"

    def test_a_bare_product_name_is_refused(self, campaign):
        with pytest.raises(ValueError, match="kind@version"):
            self._bundle(campaign, products=["flags"])

    def test_band_outside_the_data_is_an_error(self, campaign):
        with pytest.raises(ValueError, match="selects no channels"):
            self._bundle(campaign, band_mhz=(400.0, 500.0))

    def test_summary_mentions_rows_files_and_products(self, campaign):
        text = self._bundle(campaign, products=["flags@v2"]).summary()
        assert "box-gnd" in text and "flags@v2" in text


class TestAntennaMovesBetweenInputs:
    """box-gnd is input 0 on 2026-07-17 and input 2 on 07-13 in the real
    campaign. The bundle must follow it, which a hardcoded key tuple
    cannot."""

    def test_rows_come_from_the_right_input_in_each_file(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        write_corr_file(
            data / "corr_20260717_150041Z.h5",
            ntimes=4,
            keys=("0", "2"),
            sync_time=1.7843e9,
            input_to_ant={"0": "box-gnd", "2": "box-air"},
            seed=0,
        )
        write_corr_file(
            data / "corr_20260717_151041Z.h5",
            ntimes=4,
            keys=("0", "2"),
            sync_time=1.7843e9 + 600,
            input_to_ant={"0": "box-air", "2": "box-gnd"},
            seed=1,
        )
        index = MetadataIndex(data, cache=False)
        bundle = index.select().load_bundle(
            antenna="box-gnd", root=tmp_path
        )
        assert sorted(bundle.provenance["keys"]) == ["0", "2"]
        with h5py.File(data / "corr_20260717_150041Z.h5", "r") as h:
            np.testing.assert_array_equal(bundle.data[:4], h["data/0"][:])
        with h5py.File(data / "corr_20260717_151041Z.h5", "r") as h:
            np.testing.assert_array_equal(bundle.data[4:], h["data/2"][:])
