"""HFSS-backed transmitter spike simulation and beam recovery.

This module consumes Dominic's ``hfss_beam_maps/bowtie_beam.npz`` format
(``beam_cart``, ``gain_th``, ``gain_ph``, ``freqs``) from eigsep_data's
``origin/beams`` branch.  It deliberately keeps the dependency surface small
and uses the same rotation convention as
:mod:`eigsep_data.beam_mapping.geometry`.

Public API
----------
HFSSBeamSet
ground_heading
simulate_hfss_coupling
simulate_hfss
correlator_waterfall
simulate_correlator_waterfall
fit_ground_position
recover_sampled_beam
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .geometry import (
    TransmitterGeometry,
    _sph_basis,
    rotation_matrix,
    vector_to_spherical,
)
from .mapper import PolarizationBeamMapper


@dataclass
class HFSSBeamSet:
    beam_cart: np.ndarray
    gain_th: np.ndarray
    gain_ph: np.ndarray
    freqs_mhz: np.ndarray

    @classmethod
    def from_npz(cls, path, drop_last=True):
        """Load the committed HFSS beam NPZ and optionally drop its last slice."""
        with np.load(Path(path)) as z:
            out = cls(np.asarray(z["beam_cart"]), np.asarray(z["gain_th"]),
                      np.asarray(z["gain_ph"]), np.asarray(z["freqs"]))
        if drop_last:
            out = cls(out.beam_cart[:-1], out.gain_th[:-1], out.gain_ph[:-1], out.freqs_mhz[:-1])
        return out

    @property
    def nside(self):
        import healpy as hp
        return hp.npix2nside(self.beam_cart.shape[-1])


def ground_heading(east_m=0.0, north_m=0.0, height_m=92.5):
    """Unit receiver->transmitter vector for a ground TX offset."""
    v = np.array([east_m, north_m, -height_m], dtype=float)
    return v / np.linalg.norm(v)


def _sample_cartesian(beam, theta, phi):
    import healpy as hp
    px = hp.ang2pix(beam.nside, theta, phi)
    return np.moveaxis(beam.beam_cart[:, :, px], 1, -1), px


def simulate_hfss_coupling(beam, az_deg, el_deg, geometry, arms,
                           az_axis=(0, 0, -1), el_axis=(1, 0, 0)):
    """Complex per-HFSS-slice, per-pointing polarization coupling.

    Returns ``coupling[nfreq, nsample]`` (before squaring to power) and
    the HEALPix pixels sampled at each pointing.  Factored out of
    :func:`simulate_hfss` so callers that need to linearly combine
    multiple beam slices *before* squaring -- e.g. reconstructing power
    from a PCA/SVD decomposition of the complex field, where the cross
    terms between components are physically real and cannot be dropped
    by summing each component's own power -- can reuse the exact same
    rotation/projection machinery instead of duplicating it.
    """
    az_deg, el_deg = np.broadcast_arrays(az_deg, el_deg)
    arms = np.broadcast_to(arms, az_deg.shape).astype(int).ravel()
    heading = geometry.heading_top
    az = np.deg2rad(az_deg.ravel())
    el = np.deg2rad(el_deg.ravel())
    # This is the physical Marjum sequence: azimuth about +z/-z first,
    # followed by elevation about the fixed East shaft. Keep a vectorized
    # path for the production axes and a generic fallback for custom axes.
    if np.allclose(az_axis, (0, 0, -1)) and np.allclose(el_axis, (1, 0, 0)):
        ca, sa = np.cos(az), -np.sin(az)
        ce, se = np.cos(el), np.sin(el)
        Rs = np.empty((az.size, 3, 3), float)
        Rs[:, 0] = np.stack([ca, -sa, np.zeros_like(ca)], axis=1)
        Rs[:, 1] = np.stack([ce * sa, ce * ca, -se], axis=1)
        Rs[:, 2] = np.stack([se * sa, se * ca, ce], axis=1)
    else:
        Rs = np.asarray([rotation_matrix(a, e, az_axis=az_axis, el_axis=el_axis)
                         for a, e in zip(np.rad2deg(az), np.rad2deg(el))])
    rhat = np.einsum("nij,j->ni", Rs.transpose(0, 2, 1), heading)
    th, ph = vector_to_spherical(rhat)
    beam_xyz, px = _sample_cartesian(beam, th, ph)
    # Vectorize the polarization coupling over all HFSS slices and pointings.
    e0, e1 = geometry.field_top(0), geometry.field_top(1)
    e_top = np.where(arms[:, None] == 0, e0[None, :], e1[None, :])
    e = np.einsum('nij,nj->ni', Rs.transpose(0, 2, 1), e_top)
    e = e - np.sum(e * rhat, axis=1, keepdims=True) * rhat
    w = beam_xyz - np.sum(beam_xyz * rhat[None, :, :], axis=2, keepdims=True) * rhat[None, :, :]
    coupling = np.einsum('fni,ni->fn', np.conj(w), e)
    return coupling, px


def simulate_hfss(beam, az_deg, el_deg, geometry, arms, scale=1.0,
                  offset=0.0, distance_m=None, reference_distance_m=92.5,
                  az_axis=(0, 0, -1), el_axis=(1, 0, 0)):
    """Generate one spike-power vector per HFSS frequency slice.

    Returns ``power[nfreq, nsample]`` and the HEALPix pixels sampled at each
    pointing.  The output is linear power; callers may add radiometer noise
    or embed it in a correlator-like waterfall.
    """
    coupling, px = simulate_hfss_coupling(
        beam, az_deg, el_deg, geometry, arms, az_axis, el_axis)
    distance_factor = 1.0 if distance_m is None else (reference_distance_m / distance_m) ** 2
    out = offset + scale * distance_factor * np.abs(coupling) ** 2
    return out, px


def correlator_waterfall(power, freqs_mhz, nchan=1024, comb_spacing=16,
                         noise_std=0.0, baseline=None, seed=0):
    """Embed simulated TX spikes into a correlator-like frequency waterfall."""
    rng = np.random.default_rng(seed)
    power = np.asarray(power)
    out = rng.normal(0.0, noise_std, (power.shape[1], nchan))
    if baseline is not None:
        out += np.asarray(baseline)[None, :]
    df_mhz = 250.0 / nchan
    chans = np.rint(np.asarray(freqs_mhz) / df_mhz).astype(int)
    good = (chans >= 0) & (chans < nchan)
    out[:, chans[good]] += power[good].T
    return out, chans


def simulate_correlator_waterfall(beam, az_deg, el_deg, geometry, scale=1.0,
                                  noise_std=0.0, baseline_level=0.0,
                                  baseline_slope=0.0, seed=0):
    """Simulate a time/frequency waterfall with alternating TX comb arms."""
    az_deg, el_deg = np.broadcast_arrays(az_deg, el_deg)
    ntime = az_deg.size
    power = np.empty((beam.beam_cart.shape[0], ntime))
    for fi in range(power.shape[0]):
        power[fi], _ = simulate_hfss(
            beam, az_deg, el_deg, geometry, np.full(ntime, fi % 2), scale=scale
        )
    freq = np.linspace(0.0, 250.0, 1024, endpoint=False)
    baseline = baseline_level + baseline_slope * (freq - 125.0)
    return correlator_waterfall(power, beam.freqs_mhz, nchan=1024,
                                noise_std=noise_std, baseline=baseline, seed=seed)


def fit_ground_position(beam, az_deg, el_deg, arms, observed, initial=(0., 0., 0.),
                        height_m=92.5):
    """Fit east/north TX offset, polarization angle, scale, and offset."""
    y = np.asarray(observed, float)
    def residual(x):
        east, north, alpha, log_scale, offset = x
        geom = TransmitterGeometry(ground_heading(east, north, height_m), alpha)
        pred, _ = simulate_hfss(beam, az_deg, el_deg, geom, arms,
                                 scale=np.exp(log_scale), offset=offset)
        return (pred.ravel() - y.ravel()) / max(np.nanstd(y), 1e-12)
    x0 = np.array([initial[0], initial[1], initial[2], 0.0, 0.0])
    result = least_squares(residual, x0)
    result.east_m, result.north_m, result.alpha_deg = result.x[:3]
    result.scale, result.offset = np.exp(result.x[3]), result.x[4]
    return result


def recover_sampled_beam(beam_maps, az_deg, el_deg, geometry, arms, observed):
    """Recover theta/phi gains at sampled pixels from alternating arms."""
    mapper = PolarizationBeamMapper(beam_maps.gain_th[0], beam_maps.gain_ph[0])
    return mapper.fit_sampled_gains(az_deg, el_deg, geometry, arms, observed)
