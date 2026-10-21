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
cross_sim
cross_sim_ground_ref
ground_antenna_response
simulate_all_frequencies
simulate_both_arms_interleaved
"""

from pathlib import Path

import healpy
import numpy as np
import jax
import jax.numpy as jnp
from jax_healpy import pixelfunc as jhp

from healjax.coord import angles_to_coord, xyz2thphi

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
      - freqs : true frequency of each slice, in MHz (50.78125-250.0,
        step 3.90625), taken directly from the HFSS export filenames
        (Re_Ex_<freq>MHz.csv etc. under research/data/all_Efields/
        ElectricPatterns_NEW/). The top slice (250.0 MHz) is HFSS's own
        Nyquist edge and has no matching eigsep_observing correlator
        channel; after drop_last removes it, the remaining slices match
        eigsep_observing's correlator freqs (freqs[::16][13:]) exactly.
        (This array was previously off by one grid step low -- reverse-
        engineered from a mislabeled notebook anchor point rather than
        read from the source filenames -- found and fixed 2026-09-16;
        beam_cart/gain_th/gain_ph were never affected, only this label.)
      - nside : HEALPix nside of the pixelization (npix = 12*nside**2).

    Parameters
    ----------
    path : str or Path
        Path to the beam npz file. Defaults to the bowtie beam committed
        at hfss_beam_maps/bowtie_beam.npz.
    drop_last : bool
        If True (default), drop the last frequency slice from beam_cart,
        gain_th, gain_ph, and freqs before combining -- the final HFSS
        entry (250.0 MHz) is HFSS's own Nyquist edge and has no matching
        correlator channel.

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
    s, c = jnp.sin(th), jnp.cos(th)
    sp, cp = jnp.sin(phi), jnp.cos(phi)
    r_hat = jnp.stack([cp * s, sp * s, c], axis=-1)
    th_hat = jnp.stack([cp * c, sp * c, -s], axis=-1)
    phi_hat = jnp.stack([-sp, cp, jnp.zeros_like(th)], axis=-1)
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
        [
            [x * xC + c, xyC - zs, zxC + ys],
            [xyC + zs, y * yC + c, yzC - xs],
            [zxC - ys, yzC + xs, z * zC + c],
        ],
        dtype=jnp.double,
    )
    if rm.ndim > 2:
        axes = list(range(rm.ndim))
        return rm.transpose(axes[-1:] + axes[:-1])
    return rm


def _rot_mat(alpha):
    """Return z-axis rotation matrix for angle *alpha* in degrees."""
    a = jnp.deg2rad(alpha)
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

    def __init__(
        self,
        E1=1.0,
        E2=0.0,
        heading_top=jnp.array([0, 0, -1]),
        alpha=60,
        dtype=dtype_r,
    ):
        self.E1 = jnp.asarray(E1)
        self.E2 = jnp.asarray(E2)
        self.heading_top = jnp.asarray(heading_top, dtype=dtype)
        RM = _rot_mat(alpha=alpha)
        self.ax1_top = RM @ jnp.array([1, 0, 0])
        self.ax2_top = RM @ jnp.array([0, 1, 0])

    def tree_flatten(self):
        return (
            self.E1,
            self.E2,
            self.heading_top,
            self.ax1_top,
            self.ax2_top,
        ), ()

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
        return self.E1 * self.ax1_top.astype(
            self.E1.dtype
        ) + self.E2 * self.ax2_top.astype(self.E2.dtype)

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
        aux = (self.nside, self._theta_flip_to_data, self.conjugate_beam)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        nside, theta_flip, conjugate_beam = aux
        beam_cart, el_axis, az_axis = leaves
        obj = cls(beam_cart=beam_cart, conjugate_beam=conjugate_beam)
        object.__setattr__(obj, "nside", int(nside))
        object.__setattr__(obj, "el_axis", el_axis)
        object.__setattr__(obj, "az_axis", az_axis)
        object.__setattr__(obj, "_theta_flip_to_data", bool(theta_flip))
        object.__setattr__(obj, "conjugate_beam", bool(conjugate_beam))
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
        th = self._apply_theta_conv(th)
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
    Rs = jax.vmap(rx.rotation)(az, el)
    ths, phis = jax.vmap(tx.heading_thphi)(Rs)
    pxs = jax.vmap(rx.ang2pix)(ths, phis).astype(jnp.int32)
    return Rs, ths, phis, pxs


@jax.jit
def power_sim(
    rx, tx, az, el, K=1.0, C0=0.0, normalize=True, use_gain_units=False
):
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
    Er = jnp.einsum("ij,ij->i", Es, r_hat)
    Es = Es - Er[:, None] * r_hat

    Wxyz = rx.beam_at_pix(pxs)
    Wr = jnp.einsum("ij,ij->i", Wxyz, r_hat)
    Wxyz = Wxyz - Wr[:, None] * r_hat

    Epow = jnp.sum((Es.conj() * Es).real, axis=-1)
    Wpow = jnp.sum((Wxyz.conj() * Wxyz).real, axis=-1)

    eps = 1e-30
    En = jnp.sqrt(jnp.maximum(Epow, eps))
    Wn = jnp.sqrt(jnp.maximum(Wpow, eps))
    pinc_xyz = Es / En[:, None]
    prx_xyz = Wxyz / Wn[:, None]

    # conjugate_beam lives in the pytree aux data, so it is static under
    # jit and a plain Python branch is fine here.
    prx_use = jnp.conj(prx_xyz) if rx.conjugate_beam else prx_xyz
    inner = jnp.einsum("ij,ij->i", prx_use, pinc_xyz)
    PLF = (inner.conj() * inner).real

    P_shape = C0 + K * (Wpow * PLF * Epow)

    mu0, eps0 = 12.566e-7, 8.854e-12
    eta0 = jnp.sqrt(mu0 / eps0)
    K_E = 4 * jnp.pi / (2 * eta0 * 1.0)
    P_gain = C0 + K * (K_E * 1e-6 * Wpow * PLF * Epow)

    P = jnp.where(use_gain_units, P_gain, P_shape)
    P = jnp.where(normalize, P / jnp.maximum(jnp.mean(P), eps), P)

    return P, pxs, Es, Wxyz, PLF, pinc_xyz, prx_xyz


@jax.jit
def cross_sim(rx, tx1, tx2, az, el, g=1.0 + 0j):
    """
    Forward-simulate noiseless cross-correlation observables Ex, Ey.

    Unlike power_sim (which forms the polarization loss factor between
    the incident TX field and the RX beam -- an auto-correlation-style,
    phase-blind product), this returns the complex dot products
    ``Ex = E1 . u`` and ``Ey = g * (E2 . u)``, where *u* is the RX
    antenna's own complex Cartesian beam vector (``rx.beam_cart``) and
    E1/E2 are the two TX arms' incident fields in the RX frame. This is
    exactly the generative model that the per-pixel solve
    (:func:`eigsep_data.beam_mapping.vector_beam.solve_pixel`) inverts to
    recover *u* -- so noiseless output from this function, fed through
    that solver, isolates the solver's own recovery floor (geometry/
    pixelization/rank effects) from real measurement noise.

    Parameters
    ----------
    rx : RotatingAntennaCartesian
        Its beam_cart is treated as the TRUE receive beam vector u.
    tx1, tx2 : TransmitterAntenna
        The two known transmitter polarization states (e.g. alpha and
        alpha + 90 degrees), sharing the same heading_top.
    az, el : jnp.ndarray
        Receiver rotation angles in radians.
    g : complex
        Trial arm1/arm2 complex D-term; Ey's target is g * (E2 . u),
        matching solve_group's convention (which divides the Ey data by
        g before solving).

    Returns
    -------
    Ex, Ey : jnp.ndarray, shape (N,), complex
        Noiseless simulated cross-correlation data.
    pxs : jnp.ndarray, shape (N,)
    u : jnp.ndarray, shape (N, 3), complex — RX beam vector at each pxs,
        already projected transverse to r_hat (matches the convention
        used by solve_group's basis vectors).
    E1v, E2v : jnp.ndarray, shape (N, 3), complex — TX arms' incident
        fields in the RX frame, transverse-projected.
    r_hat : jnp.ndarray, shape (N, 3), real.
    """
    Rs, ths, phis, pxs = _px_th_ph_vmapped(rx, tx1, az, el)
    pxs = pxs.astype(jnp.int32)

    r_hat = rx.r_hat(ths, phis).astype(dtype_r)

    E1v = jax.vmap(tx1.E_rotated)(Rs)
    E1r = jnp.einsum("ij,ij->i", E1v, r_hat)
    E1v = E1v - E1r[:, None] * r_hat

    E2v = jax.vmap(tx2.E_rotated)(Rs)
    E2r = jnp.einsum("ij,ij->i", E2v, r_hat)
    E2v = E2v - E2r[:, None] * r_hat

    u = rx.beam_at_pix(pxs)
    ur = jnp.einsum("ij,ij->i", u, r_hat.astype(u.dtype))
    u = u - ur[:, None] * r_hat

    Ex = jnp.einsum("ij,ij->i", E1v, u)
    Ey = g * jnp.einsum("ij,ij->i", E2v, u)

    return Ex, Ey, pxs, u, E1v, E2v, r_hat


def transmitter_fields(theta_deg, phi_deg, alpha_deg, az_rad, el_rad, nside):
    """
    Geometry of every sample for one transmitter placement, in this
    module's rotation convention: the inputs
    :mod:`eigsep_data.beam_mapping.vector_beam` solves from.
    :func:`~eigsep_data.beam_mapping.vector_beam.fields_from_model`
    gives the same arrays in the production ``eigsep_base.rotations``
    convention.

    Frequency independent: compute once per geometry and reuse for every
    channel.

    Parameters
    ----------
    theta_deg, phi_deg : float
        Transmitter direction (``hpm.angles_to_coord`` convention).
    alpha_deg : float
        Polarization angle of transmitter arm 1; arm 2 is at alpha + 90.
    az_rad, el_rad : array-like, shape (nsample,)
        Sky-antenna rotation angles in radians.
    nside : int
        HEALPix nside the samples are binned on.

    Returns
    -------
    pxs : np.ndarray of int, shape (nsample,)
        Pixel of the arrival direction in the sky antenna's frame.
    r_hat : np.ndarray, shape (nsample, 3)
        Unit arrival direction (continuous, not snapped to the pixel
        centre).
    E1v, E2v : np.ndarray, shape (nsample, 3)
        Arm 1 and arm 2 incident fields in the sky antenna's frame,
        projected transverse to r_hat.
    """
    npix = 12 * nside**2
    rx = RotatingAntennaCartesian(
        beam_cart=jnp.zeros((3, npix), dtype=jnp.complex128)
    )
    coord = angles_to_coord(theta_deg, phi_deg)
    tx1 = TransmitterAntenna(
        E1=0.0, E2=1.0, heading_top=coord, alpha=alpha_deg
    )
    tx2 = TransmitterAntenna(
        E1=0.0, E2=1.0, heading_top=coord, alpha=alpha_deg + 90.0
    )
    az = jnp.asarray(az_rad)
    el = jnp.asarray(el_rad)
    Rs, ths, phis, pxs = _px_th_ph_vmapped(rx, tx1, az, el)
    r_hat = rx.r_hat(ths, phis).astype(jnp.float64)
    E1v = jax.vmap(tx1.E_rotated)(Rs)
    E2v = jax.vmap(tx2.E_rotated)(Rs)
    E1v = E1v - jnp.einsum("ij,ij->i", E1v, r_hat)[:, None] * r_hat
    E2v = E2v - jnp.einsum("ij,ij->i", E2v, r_hat)[:, None] * r_hat
    return (
        np.asarray(pxs),
        np.asarray(r_hat, dtype=float),
        np.asarray(E1v, dtype=float),
        np.asarray(E2v, dtype=float),
    )


def cross_sim_ground_ref(
    rx_sky, rx_ground, tx1, tx2, az_sky, el_sky, ground_coord, ground_alpha=0.0
):
    """
    Forward-simulate noiseless cross-correlation observables between a
    ROTATING sky receive antenna and a STATIONARY ground receive antenna,
    both illuminated by the same two-arm cross-dipole transmitter.

    ``cross_sim`` stands in for the second (ground) side of the
    correlator with an arbitrary complex D-term ``g``. This function
    instead derives that arm1/arm2 asymmetry from first principles: the
    ground antenna is a stationary copy of the SAME physical antenna as
    the sky one (pass the same ``beam_cart``/``rx_ground`` beam to model
    that), sitting at its own fixed bearing toward the TX and its own
    fixed mounting rotation -- both parameterized exactly like the sky
    antenna's own geometry (a coord + an alpha), NOT by hand-picking a
    pixel. The pixel actually sampled falls out of composing rotations
    the same way `_px_th_ph_vmapped` does for the sky side, just with a
    single fixed rotation instead of a time-varying gimbal one. Because
    the ground antenna never rotates, its own response to each TX arm is
    a single fixed complex number per frequency -- structurally exactly
    the role ``solve_group``'s per-channel D-term ``g`` plays, except now
    it is a genuine, HFSS-computable physical quantity rather than a free
    fitted scalar. Returns that value as ``g_true`` so it can be compared
    against whatever constant a pipeline currently assumes.

    This (``ground_coord``, ``ground_alpha``) pair is deliberately the
    ground-antenna analog of the sky antenna's own (``coord``, ``alpha``)
    -- 3 free parameters total (2 for the bearing, 1 for the mounting
    rotation), so that fitting the REAL ground antenna's geometry from
    data later is a straightforward extension of the existing
    theta/phi/alpha fit machinery, not a new formalism.

    Parameters
    ----------
    rx_sky : RotatingAntennaCartesian
        The rotating sky antenna; its beam_cart is the TRUE u being
        (elsewhere) recovered.
    rx_ground : RotatingAntennaCartesian
        The stationary ground antenna -- typically built from the SAME
        beam_cart as rx_sky (a physical copy), but need not be.
    tx1, tx2 : TransmitterAntenna
        The two known TX polarization arms, sharing the same heading_top
        (the TX's bearing as seen from the SKY antenna's un-rotated
        frame) -- this is a DIFFERENT vector from ``ground_coord``, since
        the two receivers are not co-located.
    az_sky, el_sky : jnp.ndarray
        Sky antenna rotation angles in radians.
    ground_coord : array-like, shape (3,)
        Unit vector: the TX's bearing as seen from the ground antenna's
        own REFERENCE (un-rotated) frame -- the ground-antenna analog of
        ``tx1.heading_top``.
    ground_alpha : float
        Rotation, in degrees, of the ground antenna's own beam pattern
        about its boresight (z) axis relative to that reference frame --
        the ground-antenna analog of ``TransmitterAntenna``'s ``alpha``,
        capturing an unknown mounting/reference-orientation offset. Uses
        the SAME rotation convention as the sky antenna's own az/el
        gimbal rotations (``RotatingAntennaCartesian.rotation``'s
        ``az_axis``), just applied once since the ground antenna never
        rotates. Default 0 (no offset from the shared reference frame).

    Returns
    -------
    Ex, Ey : jnp.ndarray, shape (N,), complex — noiseless cross-correlation
        data: sky-arm1 field dotted with u_sky, times conj of the ground
        antenna's own (fixed) arm1 response; same for arm2/Ey.
    pxs : jnp.ndarray, shape (N,) — sky antenna's sampled pixel per sample.
    u_sky : jnp.ndarray, shape (N, 3), complex — TRUE sky beam vector at
        each sample's pixel, transverse-projected (what the closure solve
        should recover).
    E1v, E2v : jnp.ndarray, shape (N, 3), complex — sky-side TX arm fields,
        transverse-projected (solve_group's basis vectors).
    r_hat : jnp.ndarray, shape (N, 3), real — sky-side line of sight.
    g_true : complex — the ground antenna's own arm2/arm1 response ratio;
        the value solve_group's ``g`` must equal for an exact (noiseless)
        inversion of this data.
    px_ground : int — the single HEALPix pixel the ground antenna samples
        (fixed for the whole simulation), for inspection/debugging.
    """
    Rs, ths, phis, pxs = _px_th_ph_vmapped(rx_sky, tx1, az_sky, el_sky)
    pxs = pxs.astype(jnp.int32)

    r_hat = rx_sky.r_hat(ths, phis).astype(dtype_r)

    E1v = jax.vmap(tx1.E_rotated)(Rs)
    E1v = E1v - jnp.einsum("ij,ij->i", E1v, r_hat)[:, None] * r_hat
    E2v = jax.vmap(tx2.E_rotated)(Rs)
    E2v = E2v - jnp.einsum("ij,ij->i", E2v, r_hat)[:, None] * r_hat

    u_sky = rx_sky.beam_at_pix(pxs)
    u_sky_r = jnp.einsum("ij,ij->i", u_sky, r_hat.astype(u_sky.dtype))
    u_sky = u_sky - u_sky_r[:, None] * r_hat

    Ex_sky_arm = jnp.einsum("ij,ij->i", E1v, u_sky)
    Ey_sky_arm = jnp.einsum("ij,ij->i", E2v, u_sky)

    # ---- Ground antenna: fixed geometry, single evaluation (no time
    # dependence -- it never rotates). ----
    V_ground_arm1, V_ground_arm2, px_g = ground_antenna_response(
        rx_ground, tx1, tx2, ground_coord, ground_alpha
    )

    Ex = jnp.conj(V_ground_arm1) * Ex_sky_arm
    Ey = jnp.conj(V_ground_arm2) * Ey_sky_arm
    g_true = jnp.conj(V_ground_arm2) / jnp.conj(V_ground_arm1)

    return Ex, Ey, pxs, u_sky, E1v, E2v, r_hat, g_true, px_g


def ground_antenna_response(
    rx_ground, tx1, tx2, ground_coord, ground_alpha=0.0
):
    """
    The stationary ground antenna's own fixed complex response to each TX
    arm -- the per-channel building block ``cross_sim_ground_ref`` uses,
    factored out so it can be evaluated on its own (e.g. inside a fit
    objective) without also simulating the sky antenna's full time series.

    Reuses ``TransmitterAntenna``'s own heading/field-rotation machinery:
    builds throwaway "views" of the SAME physical TX (same E1/E2/dipole
    axes, copied via tree leaves rather than re-deriving alpha) but with
    heading_top set to the ground antenna's OWN bearing, then pushes
    through a single fixed rotation (``ground_alpha``, about z) exactly
    the way the sky antenna's gimbal rotation is applied per sample.

    Parameters
    ----------
    rx_ground : RotatingAntennaCartesian
    tx1, tx2 : TransmitterAntenna
        The two known TX polarization arms (heading_top here is IGNORED --
        only their dipole axes / E1,E2 are reused; ``ground_coord`` is
        used as the heading instead).
    ground_coord : array-like, shape (3,)
        Unit vector: the TX's bearing as seen from the ground antenna's
        own REFERENCE (un-rotated) frame.
    ground_alpha : float
        Rotation, in degrees, of the ground antenna's own beam pattern
        about its boresight (z) axis (see ``cross_sim_ground_ref``).

    Returns
    -------
    V_ground_arm1, V_ground_arm2 : complex — the ground antenna's fixed
        response to TX arm 1 and arm 2 respectively.
    px_ground : int — the single HEALPix pixel sampled.
    """
    ground_coord = jnp.asarray(ground_coord, dtype=dtype_r)
    tx1_ground = TransmitterAntenna.tree_unflatten(
        (), (tx1.E1, tx1.E2, ground_coord, tx1.ax1_top, tx1.ax2_top)
    )
    tx2_ground = TransmitterAntenna.tree_unflatten(
        (), (tx2.E1, tx2.E2, ground_coord, tx2.ax1_top, tx2.ax2_top)
    )

    R_ground = rot_m(
        jnp.deg2rad(ground_alpha), jnp.array([0.0, 0.0, 1.0], dtype=dtype_r)
    )

    th_g, phi_g = tx1_ground.heading_thphi(R_ground)
    px_g = rx_ground.ang2pix(th_g, phi_g)
    r_hat_g = rx_ground.r_hat(th_g, phi_g)

    E1_g = tx1_ground.E_rotated(R_ground)
    E1_g = E1_g - jnp.dot(E1_g, r_hat_g) * r_hat_g
    E2_g = tx2_ground.E_rotated(R_ground)
    E2_g = E2_g - jnp.dot(E2_g, r_hat_g) * r_hat_g

    u_ground = rx_ground.beam_at_pix(px_g[None])[0]
    u_ground = (
        u_ground - jnp.dot(u_ground, r_hat_g.astype(u_ground.dtype)) * r_hat_g
    )

    V_ground_arm1 = jnp.dot(E1_g, u_ground)
    V_ground_arm2 = jnp.dot(E2_g, u_ground)

    return V_ground_arm1, V_ground_arm2, px_g


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
        tx_f = TransmitterAntenna(
            E1=0.0, E2=1.0, heading_top=coord, alpha=alpha
        )
        sim = power_sim(rx_f, tx_f, az_rad, el_rad)
        sim_all.append(np.asarray(sim[0]).squeeze())
        px_all.append(np.asarray(sim[1]).squeeze())

    return np.stack(sim_all, axis=0), np.stack(px_all, axis=0)


def simulate_both_arms_interleaved(
    bowtie_beams_use, az_deg, el_deg, coord, alpha
):
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
            tx_f = TransmitterAntenna(
                E1=0.0, E2=1.0, heading_top=coord, alpha=arm_alpha
            )
            sim = power_sim(rx_f, tx_f, az_rad, el_rad)
            sim_all.append(np.asarray(sim[0]).squeeze())
            px_all.append(np.asarray(sim[1]).squeeze())

    return np.stack(sim_all, axis=0), np.stack(px_all, axis=0)
