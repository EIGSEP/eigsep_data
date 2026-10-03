"""select_files' comb masks and comb modes after the 2026-10-03 rename.

Memo 001 identified the two combs: the 8-channel comb from 07-17 is the
beam-mapping transmitter, and the 07-16 1.000 MHz comb is box-air's own
EMI. The old labels said the opposite. These tests pin the new names, the
deprecated aliases, and the refusal of the inverted ``tx-comb`` mode.
Synthetic, empty filenames only: select() reads names, not contents.
"""

import json
import warnings

import pytest

from eigsep_data import paths, select_files as sf


@pytest.fixture(autouse=True)
def clean_setting(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    paths.set_campaign_root(None)
    yield
    paths.set_campaign_root(None)


def make_campaign(tmp_path, names, mode_rows=None):
    root = tmp_path / "marjum-2026-07"
    (root / "data").mkdir(parents=True)
    (root / "curation").mkdir()
    for n in names:
        (root / "data" / n).touch()
    if mode_rows is not None:
        with open(root / "curation" / "mode_table.jsonl", "w") as f:
            for r in mode_rows:
                f.write(json.dumps(r) + "\n")
    # select(root=...) still resolves curation/ tables through the global
    # setting, so set it too.
    paths.set_campaign_root(root)
    return root


def dropped_by(dropped):
    return {d["file"]: d["mask"] for d in dropped}


class TestCatalog:
    def test_mask_names_unique(self):
        names = [m["name"] for m in sf.CONDITIONAL_MASKS]
        assert len(names) == len(set(names))
        mandatory = {m["name"] for m in sf.MANDATORY_MASKS}
        assert not mandatory & set(names)

    def test_new_names_present_old_names_gone(self):
        names = {m["name"] for m in sf.CONDITIONAL_MASKS}
        assert {"tx-comb-teeth", "boxair-emi-1mhz"} <= names
        assert not {"digital-self-comb", "comb-rfi-1p25mhz"} & names

    def test_aliases_point_at_real_masks(self):
        names = {m["name"] for m in sf.CONDITIONAL_MASKS}
        assert set(sf.MASK_ALIASES.values()) <= names
        assert not set(sf.MASK_ALIASES) & names

    def test_boxair_emi_definition(self):
        m = next(m for m in sf.CONDITIONAL_MASKS
                 if m["name"] == "boxair-emi-1mhz")
        assert m["windows"] == [("2026-07-16T01:18:00Z",
                                 "2026-07-16T16:51:10Z")]
        assert m["inputs"] == {"4", "5"}
        assert "boxair-emi-1mhz" in sf.DEFAULT_CONDITIONAL
        assert "tx-comb-teeth" not in sf.DEFAULT_CONDITIONAL

    def test_mode_fields(self):
        assert sf.MODE_FIELDS["boxair-emi"] == "boxair_emi"
        assert sf.MODE_FIELDS["transmitter"] == "transmitter"
        assert "tx-comb" not in sf.MODE_FIELDS


class TestAliases:
    @pytest.mark.parametrize("old,new", sorted(sf.MASK_ALIASES.items()))
    def test_resolve_warns(self, old, new):
        with pytest.warns(FutureWarning, match=new):
            assert sf.resolve_mask_names([old]) == {new}

    def test_current_names_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert sf.resolve_mask_names(["tx-comb-teeth"]) == {
                "tx-comb-teeth"}

    def test_old_name_in_select_applies_new_mask(self, tmp_path):
        root = make_campaign(tmp_path, ["corr_20260717_200000Z.h5"])
        with pytest.warns(FutureWarning, match="tx-comb-teeth"):
            kept, dropped, _ = sf.select(
                root=root, conditional={"digital-self-comb"})
        assert kept == []
        assert dropped_by(dropped) == {
            "corr_20260717_200000Z.h5": "tx-comb-teeth"}

    def test_old_name_on_cli(self, tmp_path, capsys):
        root = make_campaign(tmp_path, ["corr_20260717_200000Z.h5"])
        with pytest.warns(FutureWarning):
            sf.main(["--campaign", str(root),
                     "--add-mask", "digital-self-comb", "--json"])
        out = capsys.readouterr().out
        payload = json.loads(out[out.index("{"):])
        assert payload["dropped_by_mask"] == {"tx-comb-teeth": 1}


class TestBoxairEmiMask:
    FILE = "corr_20260716_120000Z.h5"

    def test_drops_boxair_keeps_boxgnd(self, tmp_path):
        root = make_campaign(tmp_path, [self.FILE])
        cond = {"boxair-emi-1mhz"}
        kept, dropped, _ = sf.select(root=root, conditional=cond,
                                     inputs=["4"])
        assert kept == [] and dropped_by(dropped) == {
            self.FILE: "boxair-emi-1mhz"}
        kept, dropped, _ = sf.select(root=root, conditional=cond,
                                     inputs=["0"])
        assert kept == [self.FILE] and dropped == []

    def test_applied_by_default(self, tmp_path):
        root = make_campaign(tmp_path, [self.FILE])
        kept, dropped, _ = sf.select(root=root, inputs=["4"])
        assert dropped_by(dropped) == {self.FILE: "boxair-emi-1mhz"}

    def test_window_boundaries_inclusive(self, tmp_path):
        names = ["corr_20260716_011759Z.h5", "corr_20260716_011800Z.h5",
                 "corr_20260716_165110Z.h5", "corr_20260716_165111Z.h5"]
        root = make_campaign(tmp_path, names)
        kept, dropped, _ = sf.select(root=root, inputs=["4"],
                                     conditional={"boxair-emi-1mhz"})
        assert kept == [names[0], names[3]]
        assert set(dropped_by(dropped)) == {names[1], names[2]}


class TestCombModes:
    def test_tx_comb_raises_naming_both(self, tmp_path):
        root = make_campaign(tmp_path, [])
        with pytest.raises(ValueError) as e:
            sf.select(root=root, modes={"tx-comb": ["on"]})
        assert "boxair-emi" in str(e.value)
        assert "transmitter" in str(e.value)

    def test_tx_comb_cli_errors(self, tmp_path, capsys):
        root = make_campaign(tmp_path, [])
        with pytest.raises(SystemExit) as e:
            sf.main(["--campaign", str(root), "--tx-comb", "on"])
        assert e.value.code == 2
        err = capsys.readouterr().err
        assert "--boxair-emi" in err and "--transmitter" in err

    def _rows(self, column):
        return [
            {"file_first": "corr_20260716_120000Z.h5",
             "file_last": "corr_20260716_120000Z.h5", "n_files": 1,
             column: "on"},
            {"file_first": "corr_20260717_200000Z.h5",
             "file_last": "corr_20260717_200000Z.h5", "n_files": 1,
             column: "off"},
        ]

    def test_transmitter_mode_selects(self, tmp_path):
        names = ["corr_20260716_120000Z.h5", "corr_20260717_200000Z.h5"]
        rows = self._rows("boxair_emi")
        rows[0]["transmitter"], rows[1]["transmitter"] = "off", "on"
        root = make_campaign(tmp_path, names, rows)
        paths.set_campaign_root(root)
        kept, _, _ = sf.select(modes={"transmitter": ["on"]},
                               conditional=set())
        assert kept == [names[1]]
        kept, _, _ = sf.select(modes={"boxair-emi": ["on"]},
                               conditional=set())
        assert kept == [names[0]]

    def test_old_table_refused(self, tmp_path):
        names = ["corr_20260716_120000Z.h5", "corr_20260717_200000Z.h5"]
        root = make_campaign(tmp_path, names, self._rows("tx_comb"))
        paths.set_campaign_root(root)
        with pytest.raises(SystemExit, match="tx_comb.*rebuild"):
            sf.select(modes={"boxair-emi": ["on"]}, conditional=set())
