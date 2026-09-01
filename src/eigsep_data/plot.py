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


# -----------------------------------------------------------------------
# Beam-mapping DPSS diagnostics
# -----------------------------------------------------------------------

def plot_dpss_fit_examples(raw, dpss_out, times_to_plot=(0, 1000, 2000, 3000, 4000),
                           yscale="linear"):
    """
    Plot raw spectrum, DPSS model, comb channels, and fit channels for
    selected time indices.

    Parameters
    ----------
    raw : np.ndarray, shape (ntime, nchan)
    dpss_out : dict
        Output of rfi.fit_dpss_model_per_time.
    times_to_plot : sequence of int
        Time indices to plot.
    yscale : str
        Matplotlib yscale ('linear' or 'log').
    """
    raw = np.asarray(raw)
    model = dpss_out["model_raw"]
    fit_mask = dpss_out["fit_mask"]
    chs = dpss_out["chs"]

    for ti in times_to_plot:
        if ti < 0 or ti >= raw.shape[0]:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(raw[ti], ".", markersize=3, label="raw spectrum")
        plt.plot(model[ti], "-", linewidth=2, label="per-time DPSS model")
        plt.plot(chs, raw[ti, chs], "o", label="comb channels")
        plt.plot(np.where(fit_mask)[0], raw[ti, fit_mask], ".",
                 markersize=2, alpha=0.35, label="fit channels")
        plt.xlabel("raw channel")
        plt.ylabel("power")
        plt.yscale(yscale)
        plt.title(f"Per-time DPSS fit, time index {ti}")
        plt.grid(True)
        plt.legend()


def plot_dpss_reduced_waterfalls(dpss_out, test_data_skyavg=None,
                                 rfi_bad=None, diff_lim=0.5):
    """
    Plot the DPSS-reduced normalized waterfall and optionally compare to
    a sky-average normalized waterfall.

    Parameters
    ----------
    dpss_out : dict
        Output of rfi.fit_dpss_model_per_time.
    test_data_skyavg : np.ndarray, optional
        Sky-average normalized waterfall for comparison.
    rfi_bad : iterable of int, optional
        Frequency indices to blank before plotting.
    diff_lim : float
        Color scale limit for the difference panel.
    """
    plot_dpss = np.asarray(dpss_out["test_data"], dtype=float).copy()

    if rfi_bad is not None:
        for fi in set(rfi_bad):
            if 0 <= fi < plot_dpss.shape[1]:
                plot_dpss[:, fi] = np.nan

    plt.figure(figsize=(10, 5))
    plt.imshow(plot_dpss.T, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(label="normalized DPSS-reduced data")
    plt.xlabel("time index")
    plt.ylabel("frequency index")
    plt.title("Per-time DPSS reduced data")

    if test_data_skyavg is None:
        return

    sky = np.asarray(test_data_skyavg, dtype=float)
    dps = plot_dpss
    ntime = min(sky.shape[0], dps.shape[0])
    nfreq = min(sky.shape[1], dps.shape[1])
    sky, dps = sky[:ntime, :nfreq].copy(), dps[:ntime, :nfreq].copy()

    valid = np.isfinite(sky) & np.isfinite(dps)
    if rfi_bad is not None:
        for fi in set(rfi_bad):
            if 0 <= fi < nfreq:
                valid[:, fi] = False

    sky[~valid] = np.nan
    dps[~valid] = np.nan
    diff = dps - sky
    diff[~valid] = np.nan

    plt.figure(figsize=(10, 5))
    plt.imshow(sky.T, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(label="normalized sky-average data")
    plt.xlabel("time index")
    plt.ylabel("frequency index")
    plt.title("Original sky-average reduced data")

    plt.figure(figsize=(10, 5))
    plt.imshow(diff.T, aspect="auto", origin="lower", interpolation="none",
               vmin=-diff_lim, vmax=diff_lim)
    plt.colorbar(label="DPSS - skyavg")
    plt.xlabel("time index")
    plt.ylabel("frequency index")
    plt.title("Difference: per-time DPSS - sky-average")


def plot_dpss_median_spectra(dpss_out, test_data_skyavg=None, rfi_bad=None):
    """
    Compare median normalized spectra across frequency.

    Parameters
    ----------
    dpss_out : dict
        Output of rfi.fit_dpss_model_per_time.
    test_data_skyavg : np.ndarray, optional
        Sky-average waterfall for comparison.
    rfi_bad : iterable of int, optional
        Frequency indices to blank.
    """
    dps = np.asarray(dpss_out["test_data"], dtype=float).copy()
    sky = np.asarray(test_data_skyavg, dtype=float).copy() \
        if test_data_skyavg is not None else None

    if rfi_bad is not None:
        for fi in set(rfi_bad):
            if 0 <= fi < dps.shape[1]:
                dps[:, fi] = np.nan
            if sky is not None and 0 <= fi < sky.shape[1]:
                sky[:, fi] = np.nan

    plt.figure(figsize=(9, 4))
    if sky is not None:
        plt.plot(np.nanmedian(sky, axis=0), "o-", label="sky-average")
    plt.plot(np.nanmedian(dps, axis=0), "o-", label="per-time DPSS")
    plt.xlabel("frequency index")
    plt.ylabel("median normalized amplitude")
    plt.title("Median spectra after reduction")
    plt.grid(True)
    plt.legend()


def summarize_dpss_reduction(dpss_out, test_data_skyavg=None, rfi_bad=None):
    """
    Print diagnostics for the DPSS reduction and return summary statistics.

    Parameters
    ----------
    dpss_out : dict
        Output of rfi.fit_dpss_model_per_time.
    test_data_skyavg : np.ndarray, optional
        Sky-average waterfall for comparison.
    rfi_bad : iterable of int, optional
        Frequency indices to exclude from statistics.

    Returns
    -------
    dict with key 'neg_frac' : per-frequency fraction of negative residuals.
    """
    reduced = np.asarray(dpss_out["reduced"], dtype=float)
    dps     = np.asarray(dpss_out["test_data"], dtype=float)
    valid   = np.isfinite(reduced)

    if rfi_bad is not None:
        for fi in set(rfi_bad):
            if 0 <= fi < valid.shape[1]:
                valid[:, fi] = False

    n_valid = np.sum(valid, axis=0)
    n_neg = np.sum((reduced <= 0) & valid, axis=0)
    neg_frac = np.divide(
        n_neg, n_valid, out=np.full(n_valid.shape, np.nan, dtype=float),
        where=n_valid > 0,
    )

    print("DPSS reduced shape:", reduced.shape)
    print("DPSS normalized shape:", dps.shape)
    print("median negative fraction:", np.nanmedian(neg_frac))
    print("max negative fraction:", np.nanmax(neg_frac))

    if test_data_skyavg is not None:
        sky = np.asarray(test_data_skyavg, dtype=float)
        ntime = min(sky.shape[0], dps.shape[0])
        nfreq = min(sky.shape[1], dps.shape[1])
        sky = sky[:ntime, :nfreq]
        dps = dps[:ntime, :nfreq]
        comp_valid = np.isfinite(sky) & np.isfinite(dps)
        if rfi_bad is not None:
            for fi in set(rfi_bad):
                if 0 <= fi < nfreq:
                    comp_valid[:, fi] = False
        diff = dps - sky
        print("median abs DPSS-skyavg difference:",
              np.nanmedian(np.abs(diff[comp_valid])))
        print("RMS DPSS-skyavg difference:",
              np.sqrt(np.nanmean(diff[comp_valid] ** 2)))

    return {"neg_frac": neg_frac}
