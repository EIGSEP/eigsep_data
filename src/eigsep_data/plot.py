'''Module for generating standard plots.'''

import numpy as np
import matplotlib.pyplot as plt


def default_cmap(mode):
    if mode in ('real', 'imag', 'phs'):
        return 'bwr'
    else:
        return 'plasma'

def data_mode(data, mode='abs'):
    """ Convert data for chosen plotting mode.
    data : array_like
        Array of data to be plotted
    mode : str, optional
          - 'phs':  Phase angle.
          - 'abs':  Absolute value.
          - 'real': Real value.
          - 'imag': Imaginary value.
          - 'log':  Log (base-10) of absolute value.
        Default: 'abs'.
    Returns
    -------
    data : array_like
        Data transformed according to the value of `mode`.
    """
    if mode.startswith('phs'):
        data = np.angle(data)
    elif mode.startswith('abs'):
        data = np.absolute(data)
    elif mode.startswith('real'):
        data = data.real
    elif mode.startswith('imag'):
        data = data.imag
    elif mode.startswith('log'):
        data = np.absolute(data)
        data = np.log10(data)
    else:
        raise ValueError('Unrecognized plot mode.')
    return data


def set_vrange(d, vmin=None, vmax=None, drng=None):
    """
    Set vmin, vmax according to data and arguments.
    vmin, vmax : float, optional
        Min/max values of color scale after data_mode is applied.
        Default: min/max of data.
    drng : float, optional
        Alternate way to set vmin=vmax - drng.
    Returns:
        (vmin, vmax)
    """
    if drng is not None:
        assert vmin is None  # can only specify vmin or drng, not both
    if vmax is None:
        vmax = d.max()
    if drng is None:
        if vmin is None:
            vmin = d.min()
    else:
        vmin = vmax - drng
    return vmin, vmax

def waterfall(d, ax=None, mode='log', vmin=None, vmax=None, drng=None, 
              colorbar=False, **kwargs):
    """
    Generate a 2D waterfall plot.
    d : array_like
        2D array of data.
    mode : str, optional
        see docs for data_mode
        Default: 'log'.
    vmin, vmax, drng : float, optional
        See set_vrange.
    """
    if np.ma.isMaskedArray(d):
        d = d.filled(0)
    d = data_mode(d, mode=mode)
    vmin, vmax = set_vrange(d, vmin=vmin, vmax=vmax, drng=drng)
    new_kwargs = dict(aspect='auto', origin='lower', interpolation='nearest',
                      cmap=default_cmap(mode))
    new_kwargs.update(kwargs)  # override with any provided kwargs
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(d, vmax=vmax, vmin=vmin, **new_kwargs)
    if colorbar:
        plt.colorbar(im)
    return im


def color_str2tup(s):
    """Convert a hex color string to an rgb tuple."""
    r, g, b = s[0:2], s[2:4], s[4:6]
    return tuple(map(lambda x: int(x, base=16) / 255, (r, g, b)))


def terrain_plot(dem, ax=None, xlabel=True, ylabel=True,
             colorbar=False, cmap='terrain', erng_m=None, nrng_m=None,
             decimate=1, **kw):
    '''Generate standard terrain plot.'''
    E, N, U = dem.get_tile(erng_m=erng_m, nrng_m=nrng_m, mesh=False, decimate=decimate)
    extent = (E[0], E[-1], N[0], N[-1])
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(U, extent=extent, cmap=cmap, origin='lower',
                   interpolation='nearest', **kw)
    if colorbar:
        plt.colorbar(im)
    if xlabel:
        ax.set_xlabel('East [m]')
    if ylabel:
        ax.set_ylabel('North [m]')
    return im


def hp_orthoview(d, mode='real', vmin=None, vmax=None, drng=None,
                 colorbar=False, rots=[(0, 90, 0), (0, -90, 0)], **kwargs):
    import healpy as hp
    new_kwargs = dict(xsize=800, cmap=default_cmap(mode), half_sky=True, title=None)
    new_kwargs.update(**kwargs)
    d = data_mode(d, mode=mode)
    vmin, vmax = set_vrange(d, vmin=vmin, vmax=vmax, drng=drng)
    for cnt, rot in enumerate(rots):
        hp.orthview(
            d, rot=rot, sub=(1, len(rots), cnt + 1),
            min=vmin, max=vmax, **new_kwargs
        )
        hp.graticule(dpar=30, dmer=30)

def hp_mollview(d, mode='real', vmin=None, vmax=None, drng=None,
                colorbar=False, **kwargs):
    import healpy as hp
    new_kwargs = dict(cmap=default_cmap(mode), title=None)
    new_kwargs.update(**kwargs)
    d = data_mode(d, mode=mode)
    vmin, vmax = set_vrange(d, vmin=vmin, vmax=vmax, drng=drng)
    im = hp.mollview(d, cbar=colorbar, max=vmax, min=vmin, **new_kwargs)
    hp.graticule(dpar=30, dmer=30)
    return im
