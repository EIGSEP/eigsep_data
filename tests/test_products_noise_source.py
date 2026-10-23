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


def test_write_then_load(tmp_path):
    f = np.linspace(50.0, 245.0, 40)
    sigma = np.full(f.size, 2.0)
    good = np.ones(f.size, bool)
    good[[5, 6]] = False
    t_non = 800.0 + 0.002 * (f - 150.0) ** 2 + np.where(np.arange(f.size) % 2, 3.0, -3.0)
    d = ns.write(tmp_path, "v0001", freqs=f, t_non_k=t_non, t_non_sigma_k=sigma, t_off_k=308.0,
                 good=good, gamma_ns=np.full(f.size, 0.05 + 0.01j), provenance={"run": "test"},
                 validation={"check": 1})
    n = ns.load("v0001", root=tmp_path)
    assert np.isnan(n.t_non_k[5]) and not n.good[5]
    np.testing.assert_allclose(n.excess_k[good], t_non[good] - 308.0)
    # the smooth fit is numpy's weighted polyfit over the good channels
    x = (f - 150.0) / 100.0
    expect = np.polyval(np.polyfit(x[good], t_non[good] - 308.0, 3, w=1 / sigma[good]), x)
    np.testing.assert_allclose(n.excess_smooth_k, expect)
    assert np.isfinite(n.excess_smooth_k).all()
    assert n.manifest["provenance"]["run"] == "test"
    assert n.manifest["smooth_fit"]["n_channels"] == 38
    assert (d / "validation.json").is_file()
    with pytest.raises(FileExistsError):
        ns.write(tmp_path, "v0001", freqs=f, t_non_k=t_non, t_non_sigma_k=sigma, t_off_k=308.0,
                 good=good, gamma_ns=np.zeros(f.size), provenance={})
