"""
Forward simulation of the EIGSEP beam-mapping experiment.

Models a rotating receive antenna paired with a cross-dipole transmitter
emitting two interleaved polarization combs. The core classes use JAX
for JIT compilation and vmapping over rotation angles.

Public API
----------
read_beam
tot_g
TransmitterAntenna
RotatingAntennaCartesian
power_sim
simulate_all_frequencies
simulate_both_arms_interleaved
"""

from pathlib import Path

import healpy
import numpy as np
import jax
import jax.numpy as jnp
from jax_healpy import pixelfunc as jhp

from .hpm import xyz2thphi

jax.config.update("jax_enable_x64", True)

dtype_r = jnp.float64


# -----------------------------------------------------------------------
# HFSS beam I/O
# -----------------------------------------------------------------------

# The HFSS bowtie beam map ships with the repo (hfss_beam_maps/) rather
# than being fetched from external storage, so read_beam can default to
# it directly. Single compressed npz with named keys -- freqs, bm-style
# Cartesian/spherical arrays, nside -- matching the convention used for
# other simulation inputs (see sim.py's load_beam).
DEFAULT_BEAM_PATH = (
    Path(__file__).resolve().parents[2] / "hfss_beam_maps" / "bowtie_beam.npz"
)


def read_beam(path=DEFAULT_BEAM_PATH, drop_last=True):
    """
    Load HFSS beam maps from disk.

    Reads a single npz holding two representations of the same simulated
    beam, plus the frequency each slice corresponds to:
      - beam_cart : complex Cartesian E-field beam, shape (nfreq, 3, npix);
        consumed by RotatingAntennaCartesian for forward simulation and
        fitting.
      - gain_th, gain_ph : spherical theta/phi gain components, shape
        (nfreq, npix); summed and peak-normalized per frequency to serve
        as HFSS "truth" maps for comparison against reduced data.
      - freqs : frequency of each slice, in MHz, on the same grid as
        eigsep_observing's correlator freqs (freqs[::16][12:]).
      - nside : HEALPix nside of the pixelization (npix = 12*nside**2).

    Parameters
    ----------
    path : str or Path
        Path to the beam npz file. Defaults to the bowtie beam committed
        at hfss_beam_maps/bowtie_beam.npz.
    drop_last : bool
        If True (default), drop the last frequency slice from beam_cart,
        gain_th, gain_ph, and freqs before combining -- matches the
        convention used elsewhere in this pipeline where the final HFSS
        entry is unused.

    Returns
    -------
    beam_cart : np.ndarray, shape (nfreq, 3, npix)
        Complex Cartesian E-field beam maps.
    gain_sph : np.ndarray, shape (nfreq, npix)
        Peak-normalized total-gain maps (theta + phi power), one per
        frequency.
    freqs : np.ndarray, shape (nfreq,)
        Frequency of each slice, in MHz.
    """
    with np.load(path) as npz:
        beam_cart = npz["beam_cart"]
        beam_th = npz["gain_th"]
        beam_ph = npz["gain_ph"]
        freqs = npz["freqs"]

    if drop_last:
        beam_cart = beam_cart[:-1]
        beam_th = beam_th[:-1, :]
        beam_ph = beam_ph[:-1, :]
        freqs = freqs[:-1]

    gain_sph = beam_th + beam_ph
    gain_sph = gain_sph / np.max(gain_sph, axis=1, keepdims=True)

    return beam_cart, gain_sph, freqs


def tot_g(beam_cart):
    """
    Convert a complex Cartesian HFSS E-field beam (in mV) to total gain.

    Assumes a unit incident power (Pinc = 1 W).

    Parameters
    ----------
    beam_cart : array-like, shape (..., 3, npix)
        Complex Cartesian E-field beam, e.g. as returned by read_beam,
        with axis -2 holding [Ex, Ey, Ez] in millivolts. Leading axes
        (e.g. frequency) are broadcast over.

    Returns
    -------
    gain : jnp.ndarray, shape (..., npix)
        Total gain at each pixel, in absolute (not peak-normalized)
        units. Note this differs from read_beam's *gain_sph*, which is
        divided by its per-frequency peak: the two describe the same
        beam but are not directly comparable without normalizing one of
        them.
    """
    mu0, eps0 = 12.566e-7, 8.854e-12
    eta0 = jnp.sqrt(mu0 / eps0)
    Pinc = 1.0
    K_E = 4 * jnp.pi / (2 * eta0 * Pinc)

    mv_V = 1e-3
    Ex = beam_cart[..., 0, :] * mv_V
    Ey = beam_cart[..., 1, :] * mv_V
    Ez = beam_cart[..., 2, :] * mv_V
    return (jnp.abs(Ex) ** 2 + jnp.abs(Ey) ** 2 + jnp.abs(Ez) ** 2) * K_E


# -----------------------------------------------------------------------
# Geometry helpers (module-private)
# -----------------------------------------------------------------------

def _sph_basis(th, phi):
    """Return (r_hat, th_hat, phi_hat) unit vectors at each (th, phi)."""
    s, c   = jnp.sin(th), jnp.cos(th)
    sp, cp = jnp.sin(phi), jnp.cos(phi)
    r_hat   = jnp.stack([cp * s,  sp * s,  c],                    axis=-1)
    th_hat  = jnp.stack([cp * c,  sp * c, -s],                    axis=-1)
    phi_hat = jnp.stack([-sp,      cp,     jnp.zeros_like(th)],   axis=-1)
    return r_hat, th_hat, phi_hat


def rot_m(ang, vec):
    """
    Return 3x3 rotation matrix for rotation by *ang* (radians) around *vec*
    (axis-angle), using the right-hand rule.

    Both *ang* and *vec* may be batched, returning a batch of matrices.
    The matrix scales by |vec|; pass a unit vector for a pure rotation.
    """
    c = jnp.cos(ang)
    s = jnp.sin(ang)
    C = 1 - c
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    xs, ys, zs = x * s, y * s, z * s
    xC, yC, zC = x * C, y * C, z * C
    xyC, yzC, zxC = x * yC, y * zC, z * xC
    rm = jnp.array(
        [[x * xC + c, xyC - zs,  zxC + ys],
         [xyC + zs,   y * yC + c, yzC - xs],
         [zxC - ys,   yzC + xs,  z * zC + c]],
        dtype=jnp.double,
    )
    if rm.ndim > 2:
        axes = list(range(rm.ndim))
        return rm.transpose(axes[-1:] + axes[:-1])
    return rm


def _rot_mat(alpha):
    """Return z-axis rotation matrix for angle *alpha* in degrees."""
    a  = jnp.deg2rad(alpha)
    ca, sa = jnp.cos(a), jnp.sin(a)
    return jnp.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


# -----------------------------------------------------------------------
# Antenna classes
# -----------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class TransmitterAntenna:
    """
    Cross-dipole transmitter model.

    The two orthogonal dipole axes (ax1, ax2) are derived from the x- and
    y-axes of the aperture frame, rotated by *alpha* degrees around z.
    Polarization state is controlled via E1 and E2 complex amplitudes.

    Parameters
    ----------
    E1, E2 : float or complex
        Field amplitudes on ax1 and ax2.
    heading_top : array-like, shape (3,)
        Unit vector pointing from the receiver toward the transmitter in the
        receiver's local topocentric frame.
    alpha : float
        Rotation of the dipole axes around z, in degrees. Even-indexed
        data frequencies use *alpha*; odd-indexed use *alpha* + 90.
    """

    def __init__(self, E1=1.0, E2=0.0, heading_top=jnp.array([0, 0, -1]),
                 alpha=60, dtype=dtype_r):
        self.E1 = jnp.asarray(E1)
        self.E2 = jnp.asarray(E2)
        self.heading_top = jnp.asarray(heading_top, dtype=dtype)
        RM = _rot_mat(alpha=alpha)
        self.ax1_top = RM @ jnp.array([1, 0, 0])
        self.ax2_top = RM @ jnp.array([0, 1, 0])

    def tree_flatten(self):
        return (self.E1, self.E2, self.heading_top, self.ax1_top, self.ax2_top), ()

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        E1, E2, heading_top, ax1_top, ax2_top = leaves
        obj = cls(E1=E1, E2=E2, heading_top=heading_top)
        object.__setattr__(obj, "ax1_top", ax1_top)
        object.__setattr__(obj, "ax2_top", ax2_top)
        return obj

    def set_polarization(self, p1=1.0, delta=0.0, total_amp=1.0, cir=False):
        """
        Set polarization state.

        Parameters
        ----------
        p1 : float
            Fraction of power on ax1 (0..1); ax2 gets 1 - p1.
        delta : float
            Relative phase of ax2 w.r.t. ax1, in radians.
        total_amp : float
            Overall field scale.
        cir : bool
            If True, set circular polarization (ignores p1/delta).
        """
        a1 = jnp.sqrt(jnp.clip(p1, 0.0, 1.0))
        a2 = jnp.sqrt(1.0 - jnp.clip(p1, 0.0, 1.0)) * jnp.exp(1j * delta)
        self.E1 = total_amp * a1
        self.E2 = total_amp * a2
        if cir:
            self.E1 = 1.0
            self.E2 = 1j * self.E1

    def E_top(self):
        """Complex Jones vector in the aperture-top frame."""
        return (self.E1 * self.ax1_top.astype(self.E1.dtype)
                + self.E2 * self.ax2_top.astype(self.E2.dtype))

    def E_rotated(self, R):
        """TX field rotated into the receiver's local frame."""
        return jnp.linalg.inv(R) @ self.E_top()

    def heading_rotated(self, R):
        """heading_top rotated into the receiver's local frame."""
        return jnp.linalg.inv(R) @ self.heading_top

    def heading_thphi(self, R):
        """Spherical angles (theta, phi) of the heading in the local frame."""
        v = self.heading_rotated(R)
        th, phi = xyz2thphi(v)
        return th, jnp.mod(phi, 2 * jnp.pi)


@jax.tree_util.register_pytree_node_class
class RotatingAntennaCartesian:
    """
    Receive antenna model using a complex Cartesian HFSS E-field beam.

    Parameters
    ----------
    beam_cart : array-like, shape (3, npix)
        Complex Cartesian beam: [Ex, Ey, Ez] at each HEALPix pixel.
    conjugate_beam : bool
        If True, compute V = conj(E_beam) · E_inc (standard receiving
        convention). Set False if the HFSS phase convention appears flipped.
    el_axis, az_axis : array-like, shape (3,)
        Unit vectors, in the beam's own Cartesian frame, that the gimbal
        physically rotates the antenna about for elevation and azimuth.
        Default to [1, 0, 0] and [0, 0, 1] -- the assumed-ideal mount.
        Override to test a suspected mounting misalignment (the true
        mechanical axes not matching the beam's coordinate frame); there
        is no fit that recovers these from data (see fit_multi_freq_joint
        notes on the alpha/axis-tilt degeneracy).
    """

    def __init__(
        self,
        beam_cart,
        conjugate_beam=True,
        el_axis=(1, 0, 0),
        az_axis=(0, 0, 1),
    ):
        self.beam_cart = jnp.asarray(beam_cart)
        if self.beam_cart.shape[0] != 3:
            raise ValueError(
                "beam_cart must have shape (3, npix) with axes [Ex, Ey, Ez]."
            )
        self.nside = healpy.npix2nside(int(self.beam_cart.shape[-1]))
        self.el_axis = jnp.asarray(el_axis, dtype=dtype_r)
        self.az_axis = jnp.asarray(az_axis, dtype=dtype_r)
        self._theta_flip_to_data = False
        self.conjugate_beam = bool(conjugate_beam)

    def tree_flatten(self):
        leaves = (self.beam_cart, self.el_axis, self.az_axis)
        aux    = (self.nside, self._theta_flip_to_data, self.conjugate_beam)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        nside, theta_flip, conjugate_beam = aux
        beam_cart, el_axis, az_axis = leaves
        obj = cls(beam_cart=beam_cart, conjugate_beam=conjugate_beam)
        object.__setattr__(obj, "nside",               int(nside))
        object.__setattr__(obj, "el_axis",             el_axis)
        object.__setattr__(obj, "az_axis",             az_axis)
        object.__setattr__(obj, "_theta_flip_to_data", bool(theta_flip))
        object.__setattr__(obj, "conjugate_beam",      bool(conjugate_beam))
        return obj

    def rotation(self, az, el):
        """Combined az/el rotation matrix."""
        return rot_m(el, self.el_axis) @ rot_m(az, self.az_axis)

    def _apply_theta_conv(self, th):
        return jnp.where(self._theta_flip_to_data, jnp.pi - th, th)

    def r_hat(self, th, phi):
        r, _, _ = _sph_basis(self._apply_theta_conv(th), phi)
        return r

    def th_hat(self, th, phi):
        _, thh, _ = _sph_basis(self._apply_theta_conv(th), phi)
        return thh

    def phi_hat(self, th, phi):
        _, _, phh = _sph_basis(self._apply_theta_conv(th), phi)
        return phh

    def ang2pix(self, th, phi):
        """Convert (theta, phi) to HEALPix pixel index."""
        th  = self._apply_theta_conv(th)
        phi = jnp.mod(phi, 2 * jnp.pi)
        return jhp.ang2pix(self.nside, th, phi, lonlat=False).astype(jnp.int32)

    def beam_at_pix(self, pxs):
        """
        Return complex Cartesian beam vectors at the given pixel indices.

        Parameters
        ----------
        pxs : array-like, shape (N,)

        Returns
        -------
        beam_xyz : jnp.ndarray, shape (N, 3)
        """
        pxs = pxs.astype(jnp.int32)
        return jnp.moveaxis(jnp.take(self.beam_cart, pxs, axis=-1), 0, -1)


# -----------------------------------------------------------------------
# Forward simulation
# -----------------------------------------------------------------------

def _px_th_ph_vmapped(rx, tx, az, el):
    """Vmap rotation, heading angles, and pixel lookup over az/el arrays."""
    Rs   = jax.vmap(rx.rotation)(az, el)
    ths, phis = jax.vmap(tx.heading_thphi)(Rs)
    pxs  = jax.vmap(rx.ang2pix)(ths, phis).astype(jnp.int32)
    return Rs, ths, phis, pxs


@jax.jit
def power_sim(rx, tx, az, el, K=1.0, C0=0.0, normalize=True,
              use_gain_units=False):
    """
    Compute received power for every (az, el) pointing.

    Uses the polarization loss factor (PLF) between the incident TX field
    and the complex Cartesian RX beam, with optional gain-unit output and
    mean normalization.

    Parameters
    ----------
    rx : RotatingAntennaCartesian
    tx : TransmitterAntenna
    az, el : jnp.ndarray
        Receiver rotation angles in radians.
    K : float
        Overall gain scale.
    C0 : float
        Constant power offset.
    normalize : bool
        If True, divide output by its mean.
    use_gain_units : bool
        If True, scale by antenna gain in SI units instead of beam shape.

    Returns
    -------
    P : jnp.ndarray, shape (N,)
    pxs : jnp.ndarray, shape (N,)
    Es : jnp.ndarray — TX field in RX frame
    Wxyz : jnp.ndarray — RX beam vectors
    PLF : jnp.ndarray — polarization loss factor
    pinc_xyz : jnp.ndarray — normalized incident field
    prx_xyz : jnp.ndarray — normalized RX beam
    """
    Rs, ths, phis, pxs = _px_th_ph_vmapped(rx, tx, az, el)
    pxs = pxs.astype(jnp.int32)

    Es = jax.vmap(tx.E_rotated)(Rs)

    # Remove radial (non-transverse) component
    r_hat = rx.r_hat(ths, phis).astype(Es.dtype)
    Er    = jnp.einsum("ij,ij->i", Es, r_hat)
    Es    = Es - Er[:, None] * r_hat

    Wxyz  = rx.beam_at_pix(pxs)
    Wr    = jnp.einsum("ij,ij->i", Wxyz, r_hat)
    Wxyz  = Wxyz - Wr[:, None] * r_hat

    Epow  = jnp.sum((Es.conj() * Es).real,   axis=-1)
    Wpow  = jnp.sum((Wxyz.conj() * Wxyz).real, axis=-1)

    eps   = 1e-30
    En    = jnp.sqrt(jnp.maximum(Epow, eps))
    Wn    = jnp.sqrt(jnp.maximum(Wpow, eps))
    pinc_xyz = Es   / En[:, None]
    prx_xyz  = Wxyz / Wn[:, None]

    # conjugate_beam lives in the pytree aux data, so it is static under
    # jit and a plain Python branch is fine here.
    prx_use = jnp.conj(prx_xyz) if rx.conjugate_beam else prx_xyz
    inner = jnp.einsum("ij,ij->i", prx_use, pinc_xyz)
    PLF   = (inner.conj() * inner).real

    P_shape = C0 + K * (Wpow * PLF * Epow)

    mu0, eps0 = 12.566e-7, 8.854e-12
    eta0      = jnp.sqrt(mu0 / eps0)
    K_E       = 4 * jnp.pi / (2 * eta0 * 1.0)
    P_gain    = C0 + K * (K_E * 1e-6 * Wpow * PLF * Epow)

    P = jnp.where(use_gain_units, P_gain, P_shape)
    P = jnp.where(normalize, P / jnp.maximum(jnp.mean(P), eps), P)

    return P, pxs, Es, Wxyz, PLF, pinc_xyz, prx_xyz


def simulate_all_frequencies(bowtie_beams_use, az_deg, el_deg, coord, alpha):
    """
    Simulate received power for every beam in *bowtie_beams_use* using the
    same transmitter direction and polarization angle.

    Parameters
    ----------
    bowtie_beams_use : array-like
        HFSS beam maps; first axis indexes frequency.
    az_deg, el_deg : array-like
        Receiver rotation angles in degrees.
    coord : array-like, shape (3,)
        Transmitter heading_top unit vector.
    alpha : float
        Transmitter polarization angle in degrees.

    Returns
    -------
    sim_all : np.ndarray, shape (n_freq, n_samples)
    px_all  : np.ndarray, shape (n_freq, n_samples)
    """
    az_rad = jnp.deg2rad(jnp.asarray(az_deg))
    el_rad = jnp.deg2rad(jnp.asarray(el_deg))

    sim_all, px_all = [], []
    for beam in bowtie_beams_use:
        rx_f = RotatingAntennaCartesian(beam_cart=beam, conjugate_beam=True)
        tx_f = TransmitterAntenna(E1=0.0, E2=1.0, heading_top=coord, alpha=alpha)
        sim  = power_sim(rx_f, tx_f, az_rad, el_rad)
        sim_all.append(np.asarray(sim[0]).squeeze())
        px_all.append(np.asarray(sim[1]).squeeze())

    return np.stack(sim_all, axis=0), np.stack(px_all, axis=0)


def simulate_both_arms_interleaved(bowtie_beams_use, az_deg, el_deg, coord, alpha):
    """
    Simulate received power for both transmitter arms, interleaved.

    For each beam map, generates TWO output frequencies:
      - Even index: arm 1 at *alpha*
      - Odd index:  arm 2 at *alpha* + 90°

    If *bowtie_beams_use* has N maps the output has 2N frequency slots,
    matching the interleaved data layout used in fit_multi_freq_joint.

    Parameters
    ----------
    bowtie_beams_use : array-like
        HFSS beam maps; first axis indexes frequency.
    az_deg, el_deg : array-like
        Receiver rotation angles in degrees.
    coord : array-like, shape (3,)
        Transmitter heading_top unit vector.
    alpha : float
        Transmitter polarization angle in degrees for arm 1.

    Returns
    -------
    sim_all : np.ndarray, shape (2*n_freq, n_samples)
    px_all  : np.ndarray, shape (2*n_freq, n_samples)
    """
    az_rad = jnp.deg2rad(jnp.asarray(az_deg))
    el_rad = jnp.deg2rad(jnp.asarray(el_deg))

    sim_all, px_all = [], []
    for beam in bowtie_beams_use:
        rx_f = RotatingAntennaCartesian(beam_cart=beam, conjugate_beam=True)

        for arm_alpha in (alpha, alpha + 90.0):
            tx_f = TransmitterAntenna(E1=0.0, E2=1.0, heading_top=coord,
                                      alpha=arm_alpha)
            sim  = power_sim(rx_f, tx_f, az_rad, el_rad)
            sim_all.append(np.asarray(sim[0]).squeeze())
            px_all.append(np.asarray(sim[1]).squeeze())

    return np.stack(sim_all, axis=0), np.stack(px_all, axis=0)
