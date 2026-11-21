"""Phasing a cross to a point source: eigsep_data.cross."""

import json

import h5py
import numpy as np
import pytest
from eigsep_base.const import MARJUM_PASS
from eigsep_base.const import c as C_LIGHT

from eigsep_data import cross


def _corr(path, counts, n=240):
    with h5py.File(path, "w") as h5:
        h5["header/pol_delay"] = json.dumps({"01": counts, "23": 0})
        h5["header/times"] = np.arange(n, dtype=float)
    return path


def test_digital_delay_reads_the_header(tmp_path):
    d = cross.digital_delay_ns(_corr(tmp_path / "corr_x.h5", 250))
    np.testing.assert_array_equal(d, np.full(240, 1000.0))


def test_digital_delay_applies_the_d5_fixes(tmp_path):
    mid = cross.digital_delay_ns(
        _corr(tmp_path / "corr_20260715_044951Z.h5", 0)
    )
    np.testing.assert_array_equal(mid[:200], 0.0)
    assert np.isnan(mid[200:210]).all()
    np.testing.assert_array_equal(mid[210:], 1000.0)
    path = _corr(tmp_path / "corr_20260717_032454Z.h5", 0)
    np.testing.assert_array_equal(cross.digital_delay_ns(path), 1000.0)
    np.testing.assert_array_equal(cross.digital_delay_ns(path, "23"), 0.0)


def test_direct_delay_matches_time_domain_delays():
    # V_ab = x_a conj(x_b) for one sky signal reaching x_a (input a)
    # through a geometric lag (a plane wave from the zenith, a below b
    # by h) and a digital delay, and x_b through a longer chain. The
    # delay of V_ab must be what direct_delay_ns predicts.
    rng = np.random.default_rng(1)
    nfft, dt, geo, dig, chain = 256, 2.0, 3, 5, 40  # ns, samples
    sky = rng.standard_normal((400, nfft + 64))
    k = np.arange(nfft) + 64
    xa, xb = sky[:, k - geo - dig], sky[:, k - chain]
    ch = slice(20, 100)
    nu = np.fft.rfftfreq(nfft, dt * 1e-9)[ch]
    v = (np.fft.rfft(xa)[:, ch] * np.conj(np.fft.rfft(xb)[:, ch])).mean(0)
    grid = np.arange(-50, 150, 0.25)
    lag = grid[
        np.argmax(np.abs(np.exp(-2j * np.pi * np.outer(grid * 1e-9, nu)) @ v))
    ]
    h = geo * dt * 1e-9 * C_LIGHT
    tau = cross.direct_delay_ns(
        [0, 0, h], [0, 0, 1], d_ns=dig * dt, tau_instr_ns=chain * dt
    )
    assert lag == pytest.approx(tau) == pytest.approx(64.0)


def test_source_enu_points_at_the_pole():
    s = cross.source_enu(0.0, 90.0, [1.784e9, 1.784e9 + 6 * 3600])
    lat = np.radians(MARJUM_PASS[0])
    np.testing.assert_allclose(
        s, [[0, np.cos(lat), np.sin(lat)]] * 2, atol=0.01
    )


def test_highpass_removes_a_constant_and_keeps_a_fast_fringe():
    t = np.arange(600)
    x = (2.0 + np.exp(2j * np.pi * t / 20.2))[:, None]
    y = cross.highpass(x, np.ones(600), 101)
    np.testing.assert_allclose(
        y[100:-100, 0], x[100:-100, 0] - 2.0, atol=1e-12
    )


def test_matched_filter_recovers_tau_instr_across_a_delay_step():
    # A Cyg A night on an 88-m baseline, 10-s bins, the digital delay
    # stepping 0 -> 1000 ns halfway, a stationary crosstalk term and
    # radiometer noise; the template uses the same D(t) and baseline.
    t = 1784258640.0 + 10.0 * np.arange(3000)
    nu = np.linspace(120e6, 180e6, 240)
    base = np.array([1.57, -5.25, 88.36])
    d = np.where(np.arange(t.size) < 1500, 0.0, 1000.0)
    track = cross.direct_delay_ns(base, cross.source_enu(*cross.CYG_A, t), d)
    tau0, amp, phi = 813.8, 4e-3, 0.7
    rng = np.random.default_rng(0)
    w = np.full((t.size, nu.size), 10.0)
    w[rng.random(w.shape) < 0.1] = 0.0
    c = amp * np.exp(
        1j * phi + 2j * np.pi * nu * (track[:, None] + tau0) * 1e-9
    )
    c = c + 5e-3 * np.exp(-2j * np.pi * nu * 300e-9)
    c = c + 0.02 * (
        rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    )
    c[w == 0] = np.nan
    out = cross.matched_filter(
        c,
        w,
        nu,
        track,
        np.arange(700, 900, 0.25),
        nwin=360,
        blocks=np.arange(t.size) // 180,
    )
    assert abs(out["tau"] - tau0) < 0.5
    assert out["sigma_tau"] < 0.5
    assert out["amp"] == pytest.approx(amp, rel=0.05)
    assert out["phase"] == pytest.approx(phi, abs=0.05)
