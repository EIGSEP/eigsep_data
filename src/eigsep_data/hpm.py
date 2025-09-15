import aipy
import numpy as np
from functools import partial
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
