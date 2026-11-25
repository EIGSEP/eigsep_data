"""Phasing a cross to a point source: eigsep_data.cross."""

import json
import warnings

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
    # arg S is referenced to nu = 0, so it carries the delay error times
    # the band centre (150 MHz); without that term it scatters by 7 mrad.
    phase = out["phase"] + 2 * np.pi * 150e6 * (out["tau"] - tau0) * 1e-9
    assert phase == pytest.approx(phi, abs=0.03)


def test_direct_delay_in_air_follows_snell():
    # Plane-parallel air of index n between the antennas: the ray in
    # the air is at the zenith angle zr with n sin(zr) = cos(alt), and
    # the vertical baseline h sees the delay n h cos(zr) / c.
    n, h, alt = 1.00025, 88.36, np.radians(40.0)
    s = [0.0, np.cos(alt), np.sin(alt)]
    zr = np.arcsin(np.cos(alt) / n)
    assert cross.direct_delay_ns([0, 0, h], s, n_air=n) == pytest.approx(
        -n * h * np.cos(zr) / C_LIGHT * 1e9, abs=1e-9
    )
    # At the zenith the excess is h (n - 1) / c = 0.0737 ns.
    zen = [0, 0, 1]
    excess = cross.direct_delay_ns([0, 0, h], zen) - cross.direct_delay_ns(
        [0, 0, h], zen, n_air=n
    )
    assert excess == pytest.approx(0.073684, abs=1e-6)
    # A horizontal baseline sees the vacuum delay.
    assert cross.direct_delay_ns([30, 40, 0], s, n_air=n) == pytest.approx(
        cross.direct_delay_ns([30, 40, 0], s), abs=1e-12
    )


def test_direct_delay_defaults_to_the_vacuum_delay_exactly():
    s = np.random.default_rng(2).standard_normal((500, 3))
    s /= np.linalg.norm(s, axis=1)[:, None]
    b = np.array([1.57, -5.25, 88.36])
    np.testing.assert_array_equal(
        cross.direct_delay_ns(b, s), 0.0 - 0.0 - s @ b / C_LIGHT * 1e9
    )


def test_air_index_from_pressure_temperature_and_vapour():
    # 77.6 * 820 / 288 = 220.944 and 3.73e5 * 6 / 288**2 = 26.982 ppm.
    assert cross.air_index(820.0, 288.0) == pytest.approx(
        1.000220944, abs=1e-9
    )
    assert cross.air_index(820.0, 288.0, 6.0) == pytest.approx(
        1.000247927, abs=1e-9
    )


def test_highpass_gives_nonfinite_weights_and_data_no_weight():
    x = np.full((50, 1), 2.0)
    w = np.ones(50)
    w[5] = np.nan
    np.testing.assert_allclose(cross.highpass(x, w, 11), 0.0, atol=1e-12)
    x[7] = np.nan
    y = cross.highpass(x, np.ones(50), 11)
    assert np.isnan(y[7, 0])
    np.testing.assert_allclose(np.delete(y, 7, 0), 0.0, atol=1e-12)


GRID = np.arange(700, 900, 0.25)
RESULTS = ("tau", "amp", "phase", "sigma_tau", "sigma_amp")


def _fringe(n_bin=600, n_chan=64, noise=0.0):
    """A fringe of delay 813.8 ns along a 0.5 ns/bin track, unit weights."""
    nu = np.linspace(120e6, 180e6, n_chan)
    track = 0.5 * np.arange(n_bin)
    c = 4e-3 * np.exp(2j * np.pi * nu * (track[:, None] + 813.8) * 1e-9)
    rng = np.random.default_rng(3)
    c = c + noise * (
        rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    )
    return c, np.ones(c.shape), nu, track


def test_matched_filter_peak_at_a_grid_edge_is_nan():
    c, w, nu, track = _fringe(noise=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = cross.matched_filter(
            c,
            w,
            nu,
            track,
            np.arange(700, 813.5, 0.25),
            61,
            blocks=np.arange(600) // 60,
        )
    for key in RESULTS:
        assert np.isnan(out[key]), key
    assert np.isfinite(out["profile"]).all()


def test_matched_filter_without_usable_data_is_nan():
    c, w, nu, track = _fringe()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = cross.matched_filter(
            np.full_like(c, np.nan), w, nu, track, GRID, 61
        )
    for key in RESULTS:
        assert np.isnan(out[key]), key
    assert np.isnan(out["profile"]).all()


def test_matched_filter_blocks_are_labels():
    c, w, nu, track = _fringe(noise=0.02)
    idx = np.arange(600) // 60
    ref = cross.matched_filter(c, w, nu, track, GRID, 61, blocks=idx)
    assert ref["sigma_tau"] > 0
    for labels in (991000.0 + idx, idx - 1):
        out = cross.matched_filter(c, w, nu, track, GRID, 61, blocks=labels)
        assert out["sigma_tau"] == pytest.approx(ref["sigma_tau"], rel=1e-9)
        assert out["sigma_amp"] == pytest.approx(ref["sigma_amp"], rel=1e-9)


def test_matched_filter_gives_nan_weights_no_weight():
    c, w, nu, track = _fringe(noise=0.02)
    c = c + 5e-3  # stationary crosstalk, for the high-pass to remove
    bad = np.zeros(w.shape, bool)
    bad[5] = True
    ref = cross.matched_filter(c, np.where(bad, 0.0, w), nu, track, GRID, 61)
    out = cross.matched_filter(
        c, np.where(bad, np.nan, w), nu, track, GRID, 61
    )
    np.testing.assert_allclose(out["profile"], ref["profile"], rtol=1e-9)
    assert abs(out["tau"] - 813.8) < 0.5


def test_matched_filter_takes_per_bin_weights():
    # n_bin == n_chan, where weights taken along the wrong axis still
    # broadcast.
    c, _, nu, track = _fringe(n_bin=128, n_chan=128, noise=0.02)
    w = np.ones(128)
    w[:10] = 0.0
    ref = cross.matched_filter(
        c, np.repeat(w[:, None], 128, 1), nu, track, GRID, 31
    )
    out = cross.matched_filter(c, w, nu, track, GRID, 31)
    np.testing.assert_allclose(out["profile"], ref["profile"], rtol=1e-9)
