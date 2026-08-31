"""Tests for eigsep_data.beam_sim."""

import healpy
import jax.numpy as jnp
import numpy as np

from eigsep_data.beam_sim import (
    DEFAULT_BEAM_PATH,
    RotatingAntennaCartesian,
    TransmitterAntenna,
    power_sim,
    read_beam,
)

NSIDE = 8
NPIX = healpy.nside2npix(NSIDE)


class TestRotationAxes:
    def test_default_axes_unchanged(self):
        rx = RotatingAntennaCartesian(beam_cart=jnp.zeros((3, NPIX)))
        np.testing.assert_array_equal(rx.el_axis, [1, 0, 0])
        np.testing.assert_array_equal(rx.az_axis, [0, 0, 1])

    def test_custom_axes_are_used(self):
        rx_default = RotatingAntennaCartesian(beam_cart=jnp.zeros((3, NPIX)))
        rx_tilted = RotatingAntennaCartesian(
            beam_cart=jnp.zeros((3, NPIX)), el_axis=(0, 1, 0)
        )
        np.testing.assert_array_equal(rx_tilted.el_axis, [0, 1, 0])
        assert not np.allclose(
            rx_default.rotation(0.3, 0.4), rx_tilted.rotation(0.3, 0.4)
        )

    def test_custom_axes_survive_pytree_roundtrip(self):
        rx = RotatingAntennaCartesian(
            beam_cart=jnp.zeros((3, NPIX)),
            el_axis=(0, 1, 0),
            az_axis=(1, 0, 0),
        )
        leaves, aux = rx.tree_flatten()
        rx2 = RotatingAntennaCartesian.tree_unflatten(aux, leaves)
        np.testing.assert_array_equal(rx2.el_axis, [0, 1, 0])
        np.testing.assert_array_equal(rx2.az_axis, [1, 0, 0])


def _write_beam_npz(path, nfreq=4, npix=12):
    rng = np.random.default_rng(0)
    beam_cart = rng.normal(size=(nfreq, 3, npix)) + 1j * rng.normal(
        size=(nfreq, 3, npix)
    )
    gain_th = rng.uniform(0.1, 1.0, size=(nfreq, npix))
    gain_ph = rng.uniform(0.1, 1.0, size=(nfreq, npix))
    freqs = np.linspace(50.0, 250.0, nfreq)
    np.savez(
        path,
        beam_cart=beam_cart,
        gain_th=gain_th,
        gain_ph=gain_ph,
        freqs=freqs,
        nside=healpy.npix2nside(npix),
    )
    return beam_cart, gain_th, gain_ph, freqs


class TestReadBeam:
    def test_default_path_exists_and_loads(self):
        # The HFSS bowtie beam ships with the repo; the default must
        # resolve to it without any override.
        assert DEFAULT_BEAM_PATH.exists()
        beam_cart, gain_sph, freqs = read_beam()
        assert beam_cart.shape[0] == gain_sph.shape[0] == freqs.shape[0]
        assert beam_cart.shape[1:] == (3, gain_sph.shape[1])

    def test_gain_sph_is_peak_normalized(self, tmp_path):
        path = tmp_path / "beam.npz"
        _, gain_th, gain_ph, _ = _write_beam_npz(path)
        _, gain_sph, _ = read_beam(path, drop_last=False)
        np.testing.assert_allclose(np.max(gain_sph, axis=1), 1.0)
        expected = (gain_th + gain_ph) / np.max(
            gain_th + gain_ph, axis=1, keepdims=True
        )
        np.testing.assert_allclose(gain_sph, expected)

    def test_drop_last_trims_every_array_together(self, tmp_path):
        path = tmp_path / "beam.npz"
        beam_cart, gain_th, gain_ph, freqs = _write_beam_npz(path, nfreq=5)
        beam_cart_kept, gain_sph_kept, freqs_kept = read_beam(
            path, drop_last=True
        )
        beam_cart_all, gain_sph_all, freqs_all = read_beam(
            path, drop_last=False
        )
        assert beam_cart_kept.shape[0] == 4
        assert freqs_kept.shape[0] == 4
        np.testing.assert_array_equal(freqs_kept, freqs[:-1])
        np.testing.assert_array_equal(beam_cart_kept, beam_cart_all[:-1])
        np.testing.assert_array_equal(gain_sph_kept, gain_sph_all[:-1])


def _power(beam_cart, conjugate_beam, E2):
    """Simulated power over a small az/el grid."""
    rx = RotatingAntennaCartesian(
        beam_cart=jnp.asarray(beam_cart), conjugate_beam=conjugate_beam
    )
    tx = TransmitterAntenna(
        E1=1.0, E2=E2, heading_top=jnp.array([0, 0, -1]), alpha=60
    )
    az = jnp.array(np.deg2rad(np.linspace(0, 350, 36)))
    el = jnp.array(np.deg2rad(np.linspace(-40, 40, 36)))
    return np.asarray(power_sim(rx, tx, az, el, normalize=False)[0])


class TestConjugateBeam:
    def test_flag_changes_elliptical_response(self):
        # conj(W).E and W.E differ whenever either field is elliptically
        # polarized; the flag exists to correct a flipped HFSS phase
        # convention, so it must actually reach the PLF.
        beam = np.zeros((3, NPIX), dtype=complex)
        beam[0] = 1.0
        beam[1] = 0.6j
        p_true = _power(beam, True, 1.0j)
        p_false = _power(beam, False, 1.0j)
        assert not np.allclose(p_true, p_false)

    def test_flag_is_noop_for_linear_polarization(self):
        # A real (up to global phase) beam and a linear TX are invariant
        # under conjugation -- guards against the branch flipping a sign
        # it should not.
        beam = np.zeros((3, NPIX), dtype=complex)
        beam[0] = 1.0
        np.testing.assert_allclose(
            _power(beam, True, 0.0), _power(beam, False, 0.0)
        )

    def test_default_is_conjugated(self):
        rng = np.random.default_rng(0)
        beam = rng.normal(size=(3, NPIX)) + 1j * rng.normal(size=(3, NPIX))
        rx = RotatingAntennaCartesian(beam_cart=jnp.asarray(beam))
        assert rx.conjugate_beam is True
        np.testing.assert_allclose(
            _power(beam, True, 1.0j), _power(beam, rx.conjugate_beam, 1.0j)
        )

    def test_flag_survives_pytree_roundtrip(self):
        # power_sim is jitted, so the flag rides in the aux data; a lost
        # round-trip would silently restore the hardcoded behaviour.
        beam = np.zeros((3, NPIX), dtype=complex)
        beam[0] = 1.0
        rx = RotatingAntennaCartesian(
            beam_cart=jnp.asarray(beam), conjugate_beam=False
        )
        leaves, aux = rx.tree_flatten()
        assert RotatingAntennaCartesian.tree_unflatten(
            aux, leaves
        ).conjugate_beam is False
