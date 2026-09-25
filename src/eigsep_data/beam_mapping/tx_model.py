"""HFSS-backed transmitter spike simulation and beam recovery.

This module consumes Dominic's ``hfss_beam_maps/bowtie_beam.npz`` format
(``beam_cart``, ``gain_th``, ``gain_ph``, ``freqs``) from eigsep_data's
``origin/beams`` branch.  It deliberately keeps the dependency surface small
and uses the package's one pointing convention,
:mod:`eigsep_data.beam_mapping.beam_rotations`, through
:func:`~eigsep_data.beam_mapping.tx_coupling.transmitter_coupling`. Every
function that points the antenna takes ``psi_deg``, the elevation-axle
direction, as a required site constant.

Public API
----------
HFSSBeamSet
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

from .beam_rotations import mount_rotation
from .mapper import PolarizationBeamMapper
from .tx_coupling import TransmitterGeometry, ground_heading, transmitter_coupling


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


def simulate_hfss_coupling(beam, az_deg, el_deg, geometry, arms, psi_deg,
                           interpolation='bilinear'):
    """Complex per-HFSS-slice, per-pointing polarization coupling.

    Returns ``coupling[nfreq, nsample]`` (before squaring to power) and the
    HEALPix pixels nearest the transmitter direction at each pointing. Callers
    that combine beam slices linearly *before* squaring, such as a PCA of the
    complex field, where cross terms are physical, reuse this directly.
    """
    import healpy as hp
    az_deg, el_deg = np.broadcast_arrays(np.asarray(az_deg, float), np.asarray(el_deg, float))
    rotations = mount_rotation(az_deg.ravel(), el_deg.ravel(), psi_deg)
    arms = np.broadcast_to(arms, az_deg.shape).astype(int).ravel()
    coupling, theta, phi = transmitter_coupling(beam.beam_cart, rotations, geometry, arms, interpolation)
    return coupling, hp.ang2pix(beam.nside, theta, phi)


def simulate_hfss(beam, az_deg, el_deg, geometry, arms, psi_deg, scale=1.0,
                  offset=0.0, distance_m=None, reference_distance_m=92.5,
                  interpolation='bilinear'):
    """Generate one spike-power vector per HFSS frequency slice.

    Returns ``power[nfreq, nsample]`` and the HEALPix pixels sampled at each
    pointing.  The output is linear power; callers may add radiometer noise
    or embed it in a correlator-like waterfall.
    """
    coupling, px = simulate_hfss_coupling(beam, az_deg, el_deg, geometry, arms, psi_deg, interpolation)
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


def simulate_correlator_waterfall(beam, az_deg, el_deg, geometry, psi_deg, scale=1.0,
                                  noise_std=0.0, baseline_level=0.0,
                                  baseline_slope=0.0, seed=0):
    """Simulate a time/frequency waterfall with alternating TX comb arms."""
    az_deg, el_deg = np.broadcast_arrays(az_deg, el_deg)
    ntime = az_deg.size
    power = np.empty((beam.beam_cart.shape[0], ntime))
    for fi in range(power.shape[0]):
        power[fi], _ = simulate_hfss(
            beam, az_deg, el_deg, geometry, np.full(ntime, fi % 2), psi_deg, scale=scale
        )
    freq = np.linspace(0.0, 250.0, 1024, endpoint=False)
    baseline = baseline_level + baseline_slope * (freq - 125.0)
    return correlator_waterfall(power, beam.freqs_mhz, nchan=1024,
                                noise_std=noise_std, baseline=baseline, seed=seed)


def fit_ground_position(beam, az_deg, el_deg, arms, observed, psi_deg, initial=(0., 0., 0.),
                        height_m=92.5):
    """Fit east/north TX offset, polarization angle, scale, and offset."""
    y = np.asarray(observed, float)
    def residual(x):
        east, north, alpha, log_scale, offset = x
        geom = TransmitterGeometry(ground_heading(east, north, height_m), alpha)
        pred, _ = simulate_hfss(beam, az_deg, el_deg, geom, arms, psi_deg,
                                scale=np.exp(log_scale), offset=offset)
        return (pred.ravel() - y.ravel()) / max(np.nanstd(y), 1e-12)
    x0 = np.array([initial[0], initial[1], initial[2], 0.0, 0.0])
    result = least_squares(residual, x0)
    result.east_m, result.north_m, result.alpha_deg = result.x[:3]
    result.scale, result.offset = np.exp(result.x[3]), result.x[4]
    return result


def recover_sampled_beam(beam_maps, az_deg, el_deg, geometry, arms, observed, psi_deg):
    """Recover theta/phi gains at sampled pixels from alternating arms."""
    mapper = PolarizationBeamMapper(beam_maps.gain_th[0], beam_maps.gain_ph[0], psi_deg)
    return mapper.fit_sampled_gains(az_deg, el_deg, geometry, arms, observed)
