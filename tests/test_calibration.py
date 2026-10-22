"""The calibration-product builders: configs, the pieces that need no raw
data, and the commands' wiring."""

import importlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eigsep_data.calibration import _common
from eigsep_data.calibration import build_tcal as bt
from eigsep_data.calibration import validate_tcal as vt
from eigsep_data.provenance import sha256

REPO = Path(__file__).resolve().parents[1]


def test_config_is_read_from_the_version_directory(tmp_path):
    d = tmp_path / "derived" / "tcal" / "v0009"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"key": "4"}))
    cfg, path = _common.load_config(tmp_path, "tcal", "v0009")
    assert cfg == {"key": "4"} and path == d / "config.json"


def test_a_missing_config_says_how_to_make_one(tmp_path):
    with pytest.raises(SystemExit, match="copy the previous version"):
        _common.load_config(tmp_path, "tcal", "v0009")


def test_an_explicit_config_overrides_the_default(tmp_path):
    p = tmp_path / "other.json"
    p.write_text(json.dumps({"key": "1"}))
    cfg, path = _common.load_config(tmp_path, "tcal", "v0009", p)
    assert cfg["key"] == "1" and path == p


def test_masked_mean_weights_by_integration_time_and_masks_wraps():
    p = np.array([[1.0, 2.0], [3.0, -5.0], [5.0, 4.0]])
    tau = np.array([1.0, 1.0, 2.0])
    rfi = np.array([[False, False], [False, False], [False, True]])
    spec, frac = bt.masked_mean(p, tau, rfi)
    assert spec[0] == pytest.approx((1 + 3 + 10) / 4)
    assert spec[1] == pytest.approx(2.0)  # wrap and RFI both masked
    assert frac == pytest.approx(2 / 6)


def test_a_channel_with_nothing_left_is_nan():
    spec, _ = bt.masked_mean(
        np.array([[-1.0]]), np.array([1.0]), np.array([[False]])
    )
    assert np.isnan(spec[0])


def test_regime_comes_from_the_window_holding_the_first_file():
    w = pd.DataFrame(
        {
            "file_first": ["a2", "b1"],
            "file_last": ["a9", "b9"],
            "receiver_regime": ["rx-A", "rx-B"],
        }
    )
    assert list(bt.regime_of(["a3", "b2", "c1"], w)) == ["rx-A", "rx-B", ""]


def test_a_visit_far_from_its_neighbours_is_dropped():
    f = np.linspace(20, 250, 200)
    base = np.ones(f.size)
    # The outlier sits mid-run: every visit has at least three clean
    # neighbours, so only the outlier is off their median.
    p = [base * (1 + 0.001 * k) for k in range(9)]
    p[4] = base * 3.0
    v = pd.DataFrame({"t": np.arange(9.0), "p": p, "regime": ["rx-A"] * 9})
    rej = {"band_mhz": [30.0, 245.0], "frac": 0.1, "max_bad_channels": 100}
    dropped, kept = bt.contaminated(v, f, rej)
    assert len(dropped) == 1 and len(kept) == 8
    assert 4.0 not in kept.t.to_list()


def test_leave_one_out_prefers_linear_on_a_linear_drift():
    t = np.arange(10.0)
    f = np.linspace(40, 210, 50)
    v = (1 + 0.01 * t)[:, None] * np.ones(f.size)
    r = vt.leave_one_out(t, v, np.array(["a"] * 10), 10.0, f, (50, 200))
    assert r["n_scored"] == 8
    assert r["median_linear"] == pytest.approx(0.0, abs=1e-12)
    assert r["median_nearest"] > 0


def test_reference_night_is_local_time():
    start, stop = vt.reference_night(
        {
            "reference_night_local": [
                "2026-07-16T21:30:00",
                "2026-07-17T07:55:00",
            ],
            "reference_night_tz": "America/Denver",
        }
    )
    assert start == pd.Timestamp("2026-07-17T03:30:00Z").timestamp()
    assert (stop - start) / 3600 == pytest.approx(10 + 25 / 60)


def test_every_command_names_a_main():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "scripts"
    ]
    ours = {k: v for k, v in scripts.items() if ".calibration." in v}
    assert set(ours) == {
        "eigsep-build-tcal",
        "eigsep-validate-tcal",
        "eigsep-tcal-bracket-cost",
        "eigsep-build-noise-source",
    }
    for target in ours.values():
        mod, fn = target.split(":")
        assert callable(getattr(importlib.import_module(mod), fn))


def test_importing_the_builders_does_not_import_eigsep_cal():
    code = (
        "import sys; import eigsep_data.calibration.build_tcal, "
        "eigsep_data.calibration.validate_tcal, eigsep_data.calibration.bracket_cost, "
        "eigsep_data.calibration.build_noise_source; "
        "print('eigsep_cal' in sys.modules, 'cmt_vna' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"]


def test_config_record_is_relative_to_the_campaign_tree(tmp_path):
    d = tmp_path / "derived" / "tcal" / "v0009"
    d.mkdir(parents=True)
    p = d / "config.json"
    p.write_text("{}")
    rec = _common.config_record(p, tmp_path)
    assert rec["path"] == "derived/tcal/v0009/config.json"
    assert rec["sha256"] == sha256(p)
    outside = tmp_path.parent / "elsewhere.json"
    outside.write_text("{}")
    assert _common.config_record(outside, tmp_path)["path"] == str(outside)
