"""Tests for eigsep_data.vector_beam, the per-pixel vector-beam solve."""

import numpy as np
import jax.numpy as jnp
import pytest

from eigsep_data import vector_beam as vb
from eigsep_data.beam_sim import (
    RotatingAntennaCartesian,
    TransmitterAntenna,
    cross_sim,
)
from healjax.coord import angles_to_coord
from eigsep_data.rfi import (
    fit_dpss_interleaved_fix,
    flag_unbracketed_fills,
    freq_axis_flag_reduced,
)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _transverse_basis(r):
    """Two orthonormal vectors perpendicular to unit vector r."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(r @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = _unit(a - (a @ r) * r)
    e2 = np.cross(r, e1)
    return e1, e2


def _synthetic_pixels(rng, npix=40, visits=4, g=1.0 + 0j):
    """Exact geometry: each pixel has one fixed r_hat and its visits see
    the transmitter arms at different rolls about it."""
    r_hat, E1v, E2v, pxs, u_true = [], [], [], [], {}
    for p in range(npix):
        r = _unit(rng.normal(size=3))
        e1, e2 = _transverse_basis(r)
        u = rng.normal(size=2) @ np.array([e1, e2]) + 1j * (
            rng.normal(size=2) @ np.array([e1, e2])
        )
        u_true[p] = u
        for k in range(visits):
            psi = np.pi * k / visits + 0.1 * rng.normal()
            a1 = np.cos(psi) * e1 + np.sin(psi) * e2
            a2 = -np.sin(psi) * e1 + np.cos(psi) * e2
            r_hat.append(r)
            E1v.append(a1)
            E2v.append(a2)
            pxs.append(p)
    r_hat, E1v, E2v = map(np.asarray, (r_hat, E1v, E2v))
    pxs = np.asarray(pxs)
    U = np.array([u_true[p] for p in pxs])
    Ex = np.einsum("ij,ij->i", E1v, U)
    Ey = g * np.einsum("ij,ij->i", E2v, U)
    return pxs, r_hat, E1v, E2v, Ex, Ey, u_true


class TestSolvePixel:
    def test_exact_recovery_single_and_multi_visit(self):
        rng = np.random.default_rng(0)
        g = 3.0 * np.exp(1j * 0.7)
        pxs, r_hat, E1v, E2v, Ex, Ey, u_true = _synthetic_pixels(
            rng, visits=3, g=g
        )
        groups = vb.group_by_pixel(pxs, np.ones(len(pxs), bool))
        for p, idx in groups.items():
            for subset in (idx[:1], idx):
                u, resid, frac, cond = vb.solve_pixel(
                    subset, E1v, E2v, r_hat, Ex, Ey, g=g
                )
                np.testing.assert_allclose(u, u_true[p], atol=1e-10)
                assert frac < 1e-10
                assert cond == pytest.approx(1.0)

    def test_radial_component_comes_only_from_prior(self):
        rng = np.random.default_rng(1)
        pxs, r_hat, E1v, E2v, _, _, u_true = _synthetic_pixels(
            rng, npix=5, visits=5
        )
        # Add a radial part to the truth; the data cannot see it.
        U = np.array([u_true[p] for p in pxs]) + (2.0 + 1j) * r_hat
        Ex = np.einsum("ij,ij->i", E1v, U)
        Ey = np.einsum("ij,ij->i", E2v, U)
        groups = vb.group_by_pixel(pxs, np.ones(len(pxs), bool))
        for p, idx in groups.items():
            u, *_ = vb.solve_pixel(idx, E1v, E2v, r_hat, Ex, Ey)
            r = r_hat[idx[0]]
            assert abs(u @ r) < 1e-10
            np.testing.assert_allclose(u, u_true[p], atol=1e-10)

    def test_wrong_g_leaves_a_closure_residual(self):
        rng = np.random.default_rng(2)
        g = np.exp(1j * 1.2)
        pxs, r_hat, E1v, E2v, Ex, Ey, _ = _synthetic_pixels(rng, visits=4, g=g)
        idx = np.flatnonzero(pxs == 0)
        _, _, right, _ = vb.solve_pixel(idx, E1v, E2v, r_hat, Ex, Ey, g=g)
        _, _, wrong, _ = vb.solve_pixel(idx, E1v, E2v, r_hat, Ex, Ey)
        assert right < 1e-10
        assert wrong > 1e-2


class TestGroupByPixel:
    def test_groups_respect_valid_and_min_visits(self):
        pxs = np.array([5, 5, 7, 5, 7, 9])
        valid = np.array([1, 1, 1, 0, 1, 1], bool)
        groups = vb.group_by_pixel(pxs, valid, min_visits=2)
        assert set(groups) == {5, 7}
        np.testing.assert_array_equal(groups[5], [0, 1])
        np.testing.assert_array_equal(groups[7], [2, 4])

    def test_cond_filter_rejects_only_rank_deficient_visits(self):
        r = np.array([0.0, 0.0, 1.0])
        e1, e2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        # Pixel 0: all visits at one roll. Pixel 1: rolls 45 deg apart.
        r_hat = np.tile(r, (6, 1))
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        E1v = np.array([e1, e1, e1, e1, c * e1 + s * e2, e2])
        E2v = np.array([e2, e2, e2, e2, -s * e1 + c * e2, -e1])
        pxs = np.array([0, 0, 0, 1, 1, 1])
        groups = vb.group_by_pixel(
            pxs,
            np.ones(6, bool),
            min_visits=2,
            cond_ref=(r_hat, E1v, E2v),
            cond_thresh=30.0,
        )
        # Repeating one visit's rows does not change cond(A): the gate
        # cannot tell same-roll visits from diverse ones, both pass.
        assert set(groups) == {0, 1}
        # A pixel whose visits leave the transverse plane rank-deficient
        # (E1 and E2 parallel) is rejected.
        E2v_bad = E1v.copy()
        groups = vb.group_by_pixel(
            pxs,
            np.ones(6, bool),
            min_visits=2,
            cond_ref=(r_hat, E1v, E2v_bad),
            cond_thresh=30.0,
        )
        assert 0 not in groups


class TestClosureG:
    def test_recovers_injected_g(self):
        rng = np.random.default_rng(3)
        g = 5.0 * np.exp(1j * np.radians(-140.0))
        pxs, r_hat, E1v, E2v, Ex, Ey, _ = _synthetic_pixels(
            rng, npix=30, visits=4, g=g
        )
        g_fit = vb.fit_closure_g(
            pxs, r_hat, E1v, E2v, Ex, Ey, np.ones(len(pxs), bool)
        )
        assert abs(g_fit - g) / abs(g) < 1e-3


class TestValidationHelpers:
    def test_dwell_clusters(self):
        clusters = vb.dwell_clusters([3, 1, 2, 40, 41, 100], gap=20)
        assert [c.tolist() for c in clusters] == [[1, 2, 3], [40, 41], [100]]
        assert vb.dwell_clusters([]) == []

    def test_leave_one_out_is_exact_on_noiseless_data(self):
        rng = np.random.default_rng(4)
        g = 2.0 - 1.0j
        pxs, r_hat, E1v, E2v, Ex, Ey, _ = _synthetic_pixels(rng, visits=4, g=g)
        out = vb.leave_one_out(
            pxs, r_hat, E1v, E2v, Ex, Ey, np.ones(len(pxs), bool), g
        )
        assert len(out["ex_pred"]) == len(pxs)
        np.testing.assert_allclose(out["ex_pred"], out["ex_true"], atol=1e-9)
        np.testing.assert_allclose(out["ey_pred"], out["ey_true"], atol=1e-9)

    def test_split_half_uses_whole_dwells(self):
        rng = np.random.default_rng(5)
        pxs, r_hat, E1v, E2v, Ex, Ey, u_true = _synthetic_pixels(
            rng, npix=3, visits=4
        )
        # Sample index is time: place each pixel's four visits as two
        # dwells of two consecutive samples, 100 samples apart.
        times = np.empty(len(pxs), int)
        for p in range(3):
            times[pxs == p] = 1000 * p + np.array([0, 1, 100, 101])
        nt = times.max() + 1
        valid = np.zeros(nt, bool)
        valid[times] = True
        args = []
        for a in (pxs, r_hat, E1v, E2v, Ex, Ey):
            full = np.zeros((nt,) + a.shape[1:], a.dtype)
            full[times] = a
            args.append(full)
        pix, ua, ub, nd = vb.split_half_by_dwell(*args, valid, 1.0, gap=20)
        assert set(pix.tolist()) == {0, 1, 2}
        assert np.all(nd == 2)
        np.testing.assert_allclose(ua, ub, atol=1e-10)

    def test_compare_to_truth_ignores_global_complex_scale(self):
        rng = np.random.default_rng(6)
        u = rng.normal(size=(50, 3)) + 1j * rng.normal(size=(50, 3))
        out = vb.compare_to_truth((0.3 - 2j) * u, u)
        for k in ("power", "x", "y", "z"):
            assert out[k] == pytest.approx(1.0)
        assert out["scale"] == pytest.approx(1 / (0.3 - 2j))

    def test_compare_to_truth_drops_absurd_pixels(self):
        rng = np.random.default_rng(7)
        u = rng.normal(size=(50, 3)) + 0j
        u_hat = u.copy()
        u_hat[0] *= 100.0
        for absurd in ("mad", 15.0):
            out = vb.compare_to_truth(u_hat, u, absurd=absurd)
            assert out["n_absurd"] == 1
            assert out["n"] == 49


class TestAgainstForwardModel:
    """Solve data generated by beam_sim.cross_sim through the real
    transmitter geometry."""

    @pytest.fixture(scope="class")
    @staticmethod
    def sim():
        nside = 8
        npix = 12 * nside**2
        rng = np.random.default_rng(8)
        beam = rng.normal(size=(3, npix)) + 1j * rng.normal(size=(3, npix))
        rx = RotatingAntennaCartesian(beam_cart=jnp.asarray(beam))
        theta, phi, alpha = 70.0, 120.0, 10.0
        coord = angles_to_coord(theta, phi)
        tx1 = TransmitterAntenna(
            E1=0.0, E2=1.0, heading_top=coord, alpha=alpha
        )
        tx2 = TransmitterAntenna(
            E1=0.0, E2=1.0, heading_top=coord, alpha=alpha + 90.0
        )
        az = np.deg2rad(np.tile(np.linspace(-180, 180, 721), 9))
        el = np.deg2rad(np.repeat(np.linspace(-80, 80, 9), 721))
        g = 4.0 * np.exp(1j * np.radians(-150.0))
        Ex, Ey, pxs_sim, *_ = cross_sim(
            rx, tx1, tx2, jnp.asarray(az), jnp.asarray(el), g=g
        )
        pxs, r_hat, E1v, E2v = vb.transmitter_fields(
            theta, phi, alpha, az, el, nside
        )
        return dict(
            pxs=pxs,
            pxs_sim=np.asarray(pxs_sim),
            r_hat=r_hat,
            E1v=E1v,
            E2v=E2v,
            Ex=np.asarray(Ex),
            Ey=np.asarray(Ey),
            g=g,
        )

    def test_geometry_matches_cross_sim(self, sim):
        np.testing.assert_array_equal(sim["pxs"], sim["pxs_sim"])
        for E in (sim["E1v"], sim["E2v"]):
            dots = np.einsum("ij,ij->i", E, sim["r_hat"])
            np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_closure_g_and_held_out_prediction(self, sim):
        args = (
            sim["pxs"],
            sim["r_hat"],
            sim["E1v"],
            sim["E2v"],
            sim["Ex"],
            sim["Ey"],
            np.ones(len(sim["pxs"]), bool),
        )
        g_fit = vb.fit_closure_g(*args)
        # Within-pixel spread of r_hat makes the closure inexact; the
        # fitted g is still close.
        assert abs(g_fit - sim["g"]) / abs(sim["g"]) < 0.05
        out = vb.leave_one_out(*args, g_fit)
        corr = np.corrcoef(out["power_pred"], out["power_true"])[0, 1]
        assert corr > 0.95


class TestRfiAdditions:
    def test_trust_mask_leaves_untrusted_slots_out_of_the_fit(self):
        f = np.linspace(100, 200, 40)
        P = np.tile(np.cos(2 * np.pi * (f - 100) / 200.0), (3, 1))
        trust = f > 120
        P_bad = P.copy()
        P_bad[:, ~trust] = 1e6  # garbage outside the trusted domain
        clean = fit_dpss_interleaved_fix(P, f, nmodes=4, trust_mask=trust)
        bad = fit_dpss_interleaved_fix(P_bad, f, nmodes=4, trust_mask=trust)
        # The trusted region is untouched by what lies outside it...
        for k in (0, 1):
            np.testing.assert_array_equal(clean[k][:, trust], bad[k][:, trust])
        # ...whereas without trust_mask the garbage enters the fit.
        full = fit_dpss_interleaved_fix(P_bad, f, nmodes=4)
        assert np.nanmax(np.abs(full[0] - clean[0])[:, trust]) > 1.0
        # Measured values pass through even outside trust_mask.
        G = np.where(np.arange(40) % 2 == 0, bad[0], bad[1])
        np.testing.assert_array_equal(G[:, ~trust], P_bad[:, ~trust])

    def test_same_arm_gap_is_left_nan(self):
        f = np.linspace(100, 200, 40)
        P = np.tile(np.cos(2 * np.pi * (f - 100) / 200.0), (2, 1))
        P[0, 10] = np.nan  # arm-1 native slot, unmeasured at row 0
        P1, P6, _, _, _, m1, m6 = fit_dpss_interleaved_fix(
            P, f, nmodes=4, return_measured_mask=True
        )
        assert np.isnan(P1[0, 10])
        assert not m1[0, 10]
        assert np.isfinite(P6[0, 10])  # the routine cross-arm fill

    def test_freq_axis_flag_reduced_catches_a_tone_spike(self):
        rng = np.random.default_rng(10)
        P = 1.0 + 0.01 * rng.normal(size=(50, 40))
        base = freq_axis_flag_reduced(P)
        P[20, 16] = 50.0
        added = freq_axis_flag_reduced(P) & ~base
        assert np.argwhere(added).tolist() == [[20, 16]]


def test_reduce_cross_passes_measured_tones_through():
    rng = np.random.default_rng(11)
    ntime, ntone = 6, 40
    chs = 8 * np.arange(1, ntone + 1)
    nchan = chs.max() + 2
    freqs = np.linspace(110, 200, ntone)
    tones = (1.0 + 0.1 * rng.normal(size=(ntime, ntone))) * np.exp(
        1j * rng.uniform(0, 2 * np.pi, size=(ntime, ntone))
    )
    cross = np.zeros((ntime, nchan), complex)
    cross[:, chs] = tones
    scale = np.ones(ntone // 2)
    Ex, Ey, bad_time, mx, my = vb.reduce_cross(
        cross,
        chs,
        np.zeros((ntime, ntone), bool),
        scale,
        freqs,
        shift=0,
        tau_ns=91.65,
        nmodes=8,
        flag_fill_outliers=False,
    )
    assert not bad_time.any()
    assert mx[:, 0::2].all() and my[:, 1::2].all()
    np.testing.assert_allclose(Ex[:, 0::2], tones[:, 0::2], atol=1e-12)
    np.testing.assert_allclose(Ey[:, 1::2], tones[:, 1::2], atol=1e-12)


def test_flag_unbracketed_fills():
    # Arm measured on even columns; fills on odd columns.
    measured = np.zeros((3, 8), bool)
    measured[:, 0::2] = True
    measured[1, 4] = False  # row 1 loses the anchor between cols 3 and 5
    trust = np.ones(8, bool)
    trust[6] = False  # untrusted anchor: cols 5 and 7 lose a side
    out = flag_unbracketed_fills(measured, trust)
    # Row 0: cols 1, 3 bracketed; col 5 (6 untrusted), col 7 (edge) not.
    assert out[0].tolist() == [0, 0, 0, 0, 0, 1, 0, 1]
    # Row 1: the gap at col 4 is unmeasured with no anchor at 3 or 5,
    # and it unbrackets its neighbours 3 and 5.
    assert out[1].tolist() == [0, 0, 0, 1, 1, 1, 0, 1]
    assert not out[measured].any()
    # Without a trust mask, col 5 is bracketed again.
    assert not flag_unbracketed_fills(measured)[0, 5]


def test_reduce_cross_require_bracketed_fill_drops_extrapolations():
    rng = np.random.default_rng(13)
    ntime, ntone = 4, 40
    chs = 8 * np.arange(1, ntone + 1)
    freqs = np.linspace(110, 200, ntone)
    tones = (1.0 + 0.1 * rng.normal(size=(ntime, ntone))) * np.exp(
        1j * np.linspace(0, 1, ntone)
    )
    cross = np.zeros((ntime, chs.max() + 2), complex)
    cross[:, chs] = tones
    mask = np.zeros((ntime, ntone), bool)
    mask[0, 21] = True  # an Ey tone: Ey's fills at 20 and 22 lose a side
    scale = np.ones(ntone // 2)
    kw = dict(shift=0, tau_ns=91.65, nmodes=8, flag_fill_outliers=False)
    Ex0, Ey0, *_ = vb.reduce_cross(cross, chs, mask, scale, freqs, **kw)
    Ex1, Ey1, *_ = vb.reduce_cross(
        cross, chs, mask, scale, freqs, require_bracketed_fill=True, **kw
    )
    assert np.isfinite(Ey0[0, [20, 22]]).all()
    assert np.isnan(Ey1[0, [20, 22]]).all()
    # Edge fills (Ex at the top column, Ey at the bottom) never are.
    assert np.isnan(Ex1[:, -1]).all() and np.isnan(Ey1[:, 0]).all()
    # Everything else, measured or bracketed, is untouched.
    np.testing.assert_array_equal(Ex1[:, :-1], Ex0[:, :-1])
    np.testing.assert_array_equal(Ey1[1:, 1:], Ey0[1:, 1:])


class TestAgainstProductionModel:
    """Solve data generated by beam_mapping's own forward model."""

    @pytest.fixture(scope="class")
    @staticmethod
    def sim():
        from eigsep_data.beam_mapping.tx_coupling import transmitter_coupling
        from eigsep_data.beam_mapping.tx_fit import TxGeometryModel

        nside = 8
        npix = 12 * nside**2
        rng = np.random.default_rng(12)
        beam = rng.normal(size=(1, 3, npix)) + 1j * rng.normal(
            size=(1, 3, npix)
        )
        model = TxGeometryModel(
            psi_deg=142.164,
            antenna_enu=np.array([0.0, 0.0, 92.5]),
            transmitter_enu=np.array([12.0, -30.0, 0.0]),
            alpha0_deg=35.0,
            az_offset_deg=4.0,
        )
        params = np.array([1.0, -2.0, 0.5, 0.3, -0.4])
        az = np.tile(np.linspace(-180, 180, 721), 9)
        el = np.repeat(np.linspace(-80, 80, 9), 721)
        R = model.rotations(az, el, params)
        geo = model.transmitter(params)
        n = len(az)
        V = [
            transmitter_coupling(
                beam, R, geo, np.full(n, a), interpolation="nearest"
            )[0][0]
            for a in (0, 1)
        ]
        g = 3.0 * np.exp(1j * 0.4)
        pxs, r_hat, E1v, E2v = vb.fields_from_model(
            model, params, az, el, nside
        )
        return dict(
            pxs=pxs,
            r_hat=r_hat,
            E1v=E1v,
            E2v=E2v,
            Ex=np.asarray(V[0]),
            Ey=g * np.asarray(V[1]),
            g=g,
            beam=beam[0],
        )

    def test_fields_are_transverse_unit_arms(self, sim):
        for E in (sim["E1v"], sim["E2v"]):
            np.testing.assert_allclose(
                np.einsum("ij,ij->i", E, sim["r_hat"]), 0.0, atol=1e-12
            )

    def test_recovers_the_conjugate_beam(self, sim):
        args = (
            sim["pxs"],
            sim["r_hat"],
            sim["E1v"],
            sim["E2v"],
            sim["Ex"],
            sim["Ey"],
            np.ones(len(sim["pxs"]), bool),
        )
        g_fit = vb.fit_closure_g(*args)
        assert abs(g_fit - sim["g"]) / abs(sim["g"]) < 0.05
        sol = vb.solve_sampled_beam(*args, g=g_fit)
        # The production coupling is conj(E_beam) . e, so u = conj(beam)
        # up to its radial part, which the data never constrain.
        truth = np.conj(sim["beam"][:, sol["pix"]].T)
        r = np.array(
            [sim["r_hat"][sim["pxs"] == p].mean(0) for p in sol["pix"]]
        )
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        truth_t = truth - np.sum(truth * r, axis=1, keepdims=True) * r
        out = vb.compare_to_truth(sol["u"], truth_t)
        assert out["power"] > 0.95
        for k in "xyz":
            assert out[k] > 0.9
