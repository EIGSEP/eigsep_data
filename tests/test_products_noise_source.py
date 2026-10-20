"""Tests for the noise_source product reader."""

import json

import numpy as np
import pytest

from eigsep_data.products import noise_source as ns


def write(root, version="v0001", nchan=16, **override):
    f = np.linspace(50.0, 200.0, nchan)
    good = np.ones(nchan, bool)
    good[3] = False
    excess = np.where(good, 500.0 + f / 10, np.nan)
    x0, xs = 150.0, 100.0
    coef = np.polyfit((f[good] - x0) / xs, excess[good], 1)
    d = dict(freqs=f, t_non_k=excess + 300.0, t_non_sigma_k=np.full(nchan, 2.0),
             t_off_k=300.0, excess_k=excess, good=good,
             excess_smooth_k=np.polyval(coef, (f - x0) / xs), smooth_coef=coef,
             smooth_x0_mhz=x0, smooth_xscale_mhz=xs, gamma_ns=np.full(nchan, 0.05 + 0j))
    d.update(override)
    p = root / "derived" / "noise_source" / version
    p.mkdir(parents=True)
    np.savez(p / "solutions.npz", **d)
    (p / "manifest.json").write_text(json.dumps({"provenance": {"product": "noise_source", "version": version}}))
    return d


def test_load_round_trips(tmp_path):
    d = write(tmp_path)
    n = ns.load("v0001", root=tmp_path)
    assert ns.versions(root=tmp_path) == ["v0001"]
    np.testing.assert_allclose(n.excess_smooth_k, d["excess_smooth_k"])
    assert n.good.dtype == bool and not n.good[3]
    assert n.t_off_k == 300.0
    assert n.manifest["provenance"]["version"] == "v0001"


def test_on_selects_channels_and_never_regrids(tmp_path):
    d = write(tmp_path)
    n = ns.load("v0001", root=tmp_path)
    sub = d["freqs"][2:6]
    np.testing.assert_allclose(n.on(sub), d["excess_smooth_k"][2:6])
    assert np.isnan(n.on(sub, smooth=False)[1])           # channel 3 is masked
    with pytest.raises(ValueError, match="channels"):
        n.on(sub + 0.5)


def test_a_wrong_shape_is_refused(tmp_path):
    write(tmp_path, t_non_k=np.ones(5))
    with pytest.raises(ValueError, match="t_non_k"):
        ns.load("v0001", root=tmp_path)


def test_missing_version(tmp_path):
    with pytest.raises(FileNotFoundError):
        ns.load("v0009", root=tmp_path)
