import aipy
import healpy as hp
import numpy as np
from functools import partial
from scipy.linalg import lstsq
from scipy.special import sph_harm_y
import jax
import jax.numpy as jnp

from healjax import get_interp_weights
import healjax
from healjax import INT_TYPE as int_dtype
from healjax import FLOAT_TYPE as float_dtype

def vec2ang(c1, c2, c3):
    return healjax.vec2ang(c1, c2, c3)

def ang2pix(scheme, nside, c1, c2):
    px_flat = jax.vmap(lambda th, ph:healjax.ang2pix(scheme, nside, th, ph))(c1.ravel(), c2.ravel())
    px_out = px_flat.reshape(c1.shape)
    return px_out
    
def vec2pix(scheme, nside, c1, c2, c3):
    px_flat = jax.vmap(lambda x, y, z: healjax.vec2pix(scheme, nside, x, y, z))(c1.ravel(), c2.ravel(), c3.ravel())
    px_out = px_flat.reshape(c1.shape)
    return px_out

@jax.jit
def interpolate_map(nside, map_data, c1, c2, c3=None):
    '''Jax accelerated map interpolation using healjax vec2ang
    and get_interp_weights.'''
    if c3 is not None:  # translate xyz to th/phi
        c1, c2 = vec2ang(c1, c2, c3)
    px, wgts = get_interp_weights(c1, c2, nside)
    slicing = (slice(None),) * wgts.ndim + (None,) * (map_data.ndim - 1)
    interp_data = jnp.sum(map_data[px] * wgts[slicing], axis=0) 
    return interp_data

@jax.jit
def rotate_interpolate_and_sum(nside, map_data, sky, crds, rot_ms):
    '''Jax accelerated rotation/interpolation/summing using interpolate_map
    and jnp vector math.'''
    ntimes, nfreq = rot_ms.shape[0], map_data.shape[-1]
    data_out = jnp.empty((ntimes, nfreq), dtype=float_dtype)
    for tind, rot_m in enumerate(rot_ms):
        tx, ty, tz = jnp.einsum('xy,yp->xp', rot_m, crds)
        wgt = interpolate_map(nside, map_data, tx, ty, tz)
        val = jnp.sum(wgt * sky, axis=0) / jnp.sum(wgt, axis=0)
        data_out = data_out.at[tind].set(val)
    return data_out    



#  _   _ ____  __  __ 
# | | | |  _ \|  \/  |
# | |_| | |_) | |\/| |
# |  _  |  __/| |  | |
# |_| |_|_|   |_|  |_|

class HPM(aipy.healpix.HealpixMap):

    def __init__(self, *args, **kwargs):
        aipy.healpix.HealpixMap.__init__(self, *args, **kwargs)
        scheme = self._scheme.lower()
        self.jax_ang2pix = jax.jit(partial(ang2pix, scheme, self._nside))
        self.jax_vec2pix = jax.jit(partial(vec2pix, scheme, self._nside))
        self.jax_vec2ang = jax.jit(vec2ang)

    def crd2px(self, c1, c2, c3=None, interpolate=False):
        """Convert 1 dimensional arrays of input coordinates to pixel indices.
        If only c1,c2 provided, then read them as th,phi. If c1,c2,c3
        provided, read them as x,y,z. If interpolate is False, return a single
        pixel coordinate. If interpolate is True, return px,wgts where each
        entry in px contains the 4 pixels adjacent to the specified location,
        and wgt contains the 4 corresponding weights of those pixels."""
        is_nest = (self._scheme == 'NEST')
        if not interpolate:
            if c3 is None: # th/phi angle mode
                px = self.jax_ang2pix(c1, c2)
            else: # x,y,z mode
                px = self.jax_vec2pix(c1, c2, c3)
            return px
        else:
            if c3 is not None:  # translate xyz to th/phi
                c1,c2 = self.jax_vec2ang(c1, c2, c3)
            assert not is_nest  # XXX not supporting this in jax right now
            px,wgts = self.jax_get_interp_weights(c1, c2, self._nside)
            return px.T, wgts.T

    def rotate_interpolate_and_sum(self, sky, crds, rot_ms, chunk_size=16):
        data_out = []
        for i in range(0, rot_ms.shape[0], chunk_size):
            data_out.append(rotate_interpolate_and_sum(int_dtype(self._nside),
                            self.map, sky, crds, rot_ms[i:i+chunk_size]))
        return np.concatenate(data_out, axis=0)

    def __getitem__(self, crd):
        """Access data on a sphere via hpm[crd].
        crd = either 1d array of pixel indices, (th,phi), or (x,y,z), where
        th,phi,x,y,z are numpy arrays of coordinates."""
        if type(crd) is tuple:
            crd = [aipy.healpix.mk_arr(c, dtype=np.double) for c in crd]
            if self._use_interpol:
                return interpolate_map(int_dtype(self._nside), self.map, *crd)
            else:
                px = self.crd2px(*crd)
        else:
            px = aipy.healpix.mk_arr(crd, dtype=np.int64)
        return self.map[px]

    def set_map(self, data, scheme="RING"):
        """Assign data to HealpixMap.map.  Infers Nside from # of pixels via
        Npix = 12 * Nside**2."""
        try:
            nside = self.npix2nside(data.shape[0])
        except(AssertionError,ValueError):
            raise ValueError("First axis of data must have 12*N**2.")
        self.set_nside_scheme(nside, scheme)
        self.map = data


# -----------------------------------------------------------------------
# Coordinate geometry helpers
# -----------------------------------------------------------------------

def xyz2thphi(xyz, *, dtype=jnp.float64, return_mask=False, masked_fill=0.0):
    """
    Convert Cartesian unit vectors to spherical angles (theta, phi).

    theta is the polar angle from +z; phi is the azimuth from +x,
    counter-clockwise.

    Parameters
    ----------
    xyz : array-like or sequence of three arrays
        Shape (3, ...) or a length-3 sequence [x, y, z].  Accepts
        np.ma.MaskedArray; the mask is taken from the x component.
    dtype : jax dtype
        Output dtype (default float64).
    return_mask : bool
        If True, also return a boolean mask shaped (2, ...) propagated
        from the x-component mask.
    masked_fill : float
        Fill value used for masked elements before the computation.

    Returns
    -------
    out : jnp.ndarray, shape (2, ...)
        Stacked [theta, phi].
    mask : jnp.ndarray, shape (2, ...), only if return_mask=True
    """
    x, y, z = xyz

    x_is_ma = isinstance(x, np.ma.MaskedArray)
    if x_is_ma:
        in_mask = jnp.asarray(getattr(x, "mask", False))
        x = np.ma.filled(x, masked_fill)
        y = np.ma.filled(y, masked_fill)
        z = np.ma.filled(z, masked_fill)
    else:
        in_mask = None

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)

    r   = jnp.hypot(x, y)
    phi = jnp.arctan2(y, x)
    th  = jnp.arctan2(r, z)

    out = jnp.stack([th, phi]).astype(dtype)

    if return_mask:
        if in_mask is None:
            mask = jnp.zeros_like(x, dtype=bool)
        else:
            mask = jnp.asarray(in_mask, dtype=bool)
        mask = jnp.broadcast_to(mask, x.shape)
        mask = jnp.stack([mask, mask])
        return out, mask

    return out


def angles_to_coord(theta_deg, phi_deg):
    """
    Convert spherical angles to a Cartesian unit vector.

    Parameters
    ----------
    theta_deg : float
        Polar angle from +z, in degrees.
    phi_deg : float
        Azimuth from +x in the xy-plane, in degrees.

    Returns
    -------
    coord : np.ndarray, shape (3,)
        Unit vector [x, y, z].
    """
    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def angles_to_coord_jax(theta_deg, phi_deg):
    """JAX version of angles_to_coord."""
    theta = jnp.deg2rad(theta_deg)
    phi   = jnp.deg2rad(phi_deg)
    return jnp.array([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta),
    ])


# -----------------------------------------------------------------------
# Spherical-harmonic beam fitting
# -----------------------------------------------------------------------

def build_real_design_matrix_from_angles(theta, phi, lmax):
    """
    Build design matrix A of shape (n_obs, (lmax+1)^2) mapping real
    coefficient vector x to real values at (theta, phi).

    Coefficient layout: a_l0, then for m=1..l: Re(a_lm), Im(a_lm).
    """
    npar = (lmax + 1) ** 2
    A = np.empty((theta.size, npar), dtype=np.float64)
    k = 0
    for l in range(lmax + 1):
        Y = sph_harm_y(l, 0, theta, phi)
        A[:, k] = Y.real
        k += 1
        for m in range(1, l + 1):
            Y = sph_harm_y(l, m, theta, phi)
            A[:, k]     =  2.0 * Y.real
            A[:, k + 1] = -2.0 * Y.imag
            k += 2
    return A


def x_to_alm(x, lmax):
    """Convert real coefficient vector x to complex healpy alm array."""
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    k = 0
    for l in range(lmax + 1):
        alm[hp.Alm.getidx(lmax, l, 0)] = x[k]
        k += 1
        for m in range(1, l + 1):
            alm[hp.Alm.getidx(lmax, l, m)] = x[k] + 1j * x[k + 1]
            k += 2
    return alm


def _make_penalty_diag(lmax, p=2):
    """
    Return penalty weight vector for Laplacian regularization.

    pen[k] = (l(l+1))^p for each coefficient in x; zero for l=0.
    """
    npar = (lmax + 1) ** 2
    pen = np.zeros(npar, dtype=np.float64)
    k = 0
    for l in range(lmax + 1):
        w = (l * (l + 1)) ** p
        pen[k] = w
        k += 1
        for m in range(1, l + 1):
            pen[k] = w
            pen[k + 1] = w
            k += 2
    return pen


def _solve_regularized_ls(A, y, lam, pen, w=None):
    """
    Solve argmin ||W^(1/2)(Ax - y)||^2 + lam * ||diag(pen)^(1/2) x||^2
    via row augmentation.

    Parameters
    ----------
    A : (n_obs, npar)
    y : (n_obs,)
    lam : float
    pen : (npar,) — penalty weights from _make_penalty_diag
    w : (n_obs,) optional observation weights
    """
    npar = A.shape[1]
    if w is not None:
        w  = np.asarray(w, dtype=np.float64)
        sw = np.sqrt(np.clip(w, 0.0, np.inf))
        A  = A * sw[:, None]
        y  = y * sw

    sqrt_lam_pen = np.sqrt(lam * pen)
    cols = np.where(sqrt_lam_pen > 0)[0]
    if cols.size > 0:
        R = np.zeros((cols.size, npar), dtype=np.float64)
        R[np.arange(cols.size), cols] = sqrt_lam_pen[cols]
        A = np.vstack([A, R])
        y = np.concatenate([y, np.zeros(cols.size, dtype=np.float64)])

    x, *_ = lstsq(A, y)
    return x


def fit_alms_from_maps(
    maps,
    nside,
    lmax=5,
    lam=1e-2,
    p=2,
    fit_log=False,
    eps=1e-12,
    user_weights=None,
    peak_weight_alpha=0.0,
    peak_weight_gamma=2.0,
    nest=False,
):
    """
    Fit spherical-harmonic coefficients (alms) to sparse HEALPix maps.

    Parameters
    ----------
    maps : (n_maps, npix) or (npix,)
        Input maps. Pixels equal to hp.UNSEEN or non-finite are ignored.
    nside : int
        HEALPix nside.
    lmax : int
        Maximum spherical-harmonic degree.
    lam : float
        Regularization strength (Laplacian penalty).
    p : int
        Regularization order: penalty = (l(l+1))^p.
    fit_log : bool
        If True, fit log(values). Values must be positive; eps prevents
        log(0).
    eps : float
        Floor applied before taking log when fit_log=True.
    user_weights : (npix,) optional
        Per-pixel weights applied to observed pixels.
    peak_weight_alpha : float
        Extra upweighting strength for bright pixels.
    peak_weight_gamma : float
        Power applied to normalized pixel value for peak upweighting.
    nest : bool
        HEALPix ordering (False = RING).

    Returns
    -------
    alms : np.ndarray, shape (n_maps, hp.Alm.getsize(lmax)), complex128
    """
    maps = np.asarray(maps)
    if maps.ndim == 1:
        maps = maps[None, :]
    n_maps, npix = maps.shape
    assert npix == hp.nside2npix(nside)

    pen = _make_penalty_diag(lmax, p=p)
    A_cache = {}

    alms = np.empty((n_maps, hp.Alm.getsize(lmax)), dtype=np.complex128)

    for i in range(n_maps):
        m = maps[i]
        known = (m != hp.UNSEEN) & np.isfinite(m)
        ipix = np.where(known)[0]
        if ipix.size == 0:
            raise ValueError(f"Map {i} has no observed pixels.")

        y_lin = m[ipix].astype(np.float64)
        y = np.log(np.maximum(y_lin, eps)) if fit_log else y_lin

        w = None
        if user_weights is not None:
            w = np.asarray(user_weights, dtype=np.float64)[ipix].copy()

        if peak_weight_alpha > 0:
            f    = np.maximum(y_lin, 0.0)
            fmax = f.max() if f.size else 1.0
            extra = 1.0 + peak_weight_alpha * (f / max(fmax, 1e-30)) ** peak_weight_gamma
            w = extra if w is None else (w * extra)

        key = tuple(ipix.tolist())
        if key not in A_cache:
            theta, phi = hp.pix2ang(nside, ipix, nest=nest)
            A_cache[key] = build_real_design_matrix_from_angles(theta, phi, lmax)

        x = _solve_regularized_ls(A_cache[key], y, lam=lam, pen=pen, w=w)
        alms[i] = x_to_alm(x, lmax)

    return alms


def alms_to_filled_maps(
    alms,
    nside,
    lmax,
    clamp_known=False,
    original_maps=None,
    fit_log=False,
    log_offset=None,
):
    """
    Synthesize full HEALPix maps from alm coefficients.

    Parameters
    ----------
    alms : (n_maps, hp.Alm.getsize(lmax)) or (alm_size,)
    nside : int
    lmax : int
    clamp_known : bool
        If True and original_maps is provided, restore observed pixels
        to their original values.
    original_maps : array-like, optional
        Original sparse maps used to identify observed pixels for clamping.
    fit_log : bool
        If True, exponentiate the synthesized map (undoes log-domain fit).
    log_offset : (n_maps,) optional
        Additive offsets applied in log-domain before exponentiation.

    Returns
    -------
    out : np.ndarray, shape (n_maps, npix)
    """
    alms = np.asarray(alms)
    if alms.ndim == 1:
        alms = alms[None, :]
    n_maps = alms.shape[0]

    out = np.empty((n_maps, hp.nside2npix(nside)), dtype=np.float64)
    for i in range(n_maps):
        pred = hp.alm2map(alms[i].astype(np.complex128), nside=nside, lmax=lmax)
        if fit_log:
            if log_offset is not None:
                pred = pred + float(log_offset[i])
            pred = np.exp(pred)
        if clamp_known and original_maps is not None:
            m0 = np.asarray(
                original_maps[i]
                if np.asarray(original_maps).ndim == 2
                else original_maps
            )
            known = (m0 != hp.UNSEEN) & np.isfinite(m0)
            pred[known] = m0[known]
        out[i] = pred

    return out


def sph_fit(maps, nside, lmax, lam, peak_weight_alpha=0.0, peak_weight_gamma=2.0):
    """
    Convenience wrapper: fit alms then synthesize full maps.

    Parameters
    ----------
    maps : (n_maps, npix) or (npix,)
    nside : int
    lmax : int
    lam : float
        Regularization strength.
    peak_weight_alpha, peak_weight_gamma : float
        Optional peak upweighting parameters passed to fit_alms_from_maps.

    Returns
    -------
    filled_maps : np.ndarray, shape (n_maps, npix)
    """
    alms = fit_alms_from_maps(
        maps=maps,
        nside=nside,
        lmax=lmax,
        lam=lam,
        p=2,
        fit_log=True,
        eps=1e-12,
        peak_weight_alpha=peak_weight_alpha,
        peak_weight_gamma=peak_weight_gamma,
    )
    return alms_to_filled_maps(
        alms,
        nside=nside,
        lmax=lmax,
        fit_log=True,
        clamp_known=False,
        original_maps=maps,
    )
