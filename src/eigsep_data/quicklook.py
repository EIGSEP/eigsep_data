"""
Quick-look summary of EIGSEP correlator h5 files.

One command in the field: load a corr file (or directory of them),
auto-detect the calibration comb, flag RFI, print summary statistics,
and save a waterfall/spectrum/time-series figure per file.

Command line::

    eigsep-quicklook path/to/file.h5 [more files or dirs] [-o OUTDIR]

Library::

    from eigsep_data import quicklook as ql
    res = ql.quicklook("corr_file.h5")
    ql.plot_quicklook(res, save="corr_file_quicklook.png")
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eigsep_observing.io import read_hdf5

from . import rfi

COMB_SPACING = 16  # cal comb lands in every 16th channel


def find_comb(spec, spacing=COMB_SPACING, min_excess=3.0):
    """
    Detect the calibration comb in a spectrum.

    Compares the median power of each channel residue class (mod
    ``spacing``) to the overall median. The comb makes one class stand
    out by orders of magnitude.

    Parameters
    ----------
    spec : ndarray
        1D power spectrum (e.g. a time-median of the waterfall).
    spacing : int
        Channel spacing of the comb tones.
    min_excess : float
        Minimum ratio of comb-class median to overall median required
        to declare a detection.

    Returns
    -------
    offset : int or None
        Channel offset of the comb tones, or None if no comb detected.
    excess : float
        Ratio of the winning class median to the overall median.

    """
    spec = np.abs(np.asarray(spec, dtype=float))
    positive = spec[spec > 0]
    if positive.size == 0:
        return None, 0.0
    base = np.median(positive)
    scores = np.array(
        [np.median(spec[off::spacing]) for off in range(spacing)]
    )
    offset = int(scores.argmax())
    excess = float(scores[offset] / base)
    if excess < min_excess:
        return None, excess
    return offset, excess


def comb_flags(nchan, offset, spacing=COMB_SPACING, pad=1):
    """
    Static channel mask for comb tones and their spillover.

    Parameters
    ----------
    nchan : int
        Number of frequency channels.
    offset : int
        Channel offset of the first comb tone.
    spacing : int
        Channel spacing of the comb tones.
    pad : int
        Also flag this many channels on each side of every tone
        (adjacent-channel spillover).

    Returns
    -------
    flags : ndarray of bool
        Length-``nchan`` mask, True on tones and their pad.

    """
    flags = np.zeros(nchan, dtype=bool)
    tones = np.arange(offset % spacing, nchan, spacing)
    for delta in range(-pad, pad + 1):
        idx = tones + delta
        idx = idx[(idx >= 0) & (idx < nchan)]
        flags[idx] = True
    return flags


def flag_spectra(wf, nsig=8, spacing=COMB_SPACING, pad=1):
    """
    Flag a waterfall: comb tones, RFI outliers, dropped integrations.

    Exact-zero samples (dropped integrations, dead inputs) are flagged
    outright. The comb is detected automatically; comb channels,
    mostly-dead channels, and all-zero rows are excluded from the
    median-filter outlier search so they do not skew its noise
    estimate.

    Parameters
    ----------
    wf : ndarray
        2D waterfall, shape ``(ntimes, nchan)``. May be complex
        (magnitude is used).
    nsig : float
        Outlier threshold for :func:`eigsep_data.rfi.median_flagger`.
    spacing : int
        Comb tone spacing in channels.
    pad : int
        Channels flagged on each side of every comb tone.

    Returns
    -------
    flags : ndarray of bool
        Same shape as ``wf``, True where flagged.
    comb_offset : int or None
        Detected comb channel offset, or None.

    """
    mag = np.abs(np.asarray(wf)).astype(float)
    dead = mag == 0
    zero_rows = dead.all(axis=1)
    live_rows = ~zero_rows
    if live_rows.any():
        med_spec = np.median(mag[live_rows], axis=0)
        dead_chan = dead[live_rows].mean(axis=0) > 0.9
    else:
        med_spec = mag.mean(axis=0)
        dead_chan = np.ones(mag.shape[1], dtype=bool)
    comb_offset, _ = find_comb(med_spec, spacing=spacing)
    static = np.zeros(mag.shape[1], dtype=bool)
    if comb_offset is not None:
        static = comb_flags(
            mag.shape[1], comb_offset, spacing=spacing, pad=pad
        )
    flags = np.zeros(mag.shape, dtype=bool)
    live_cols = np.flatnonzero(~(static | dead_chan))
    if live_cols.size and live_rows.any():
        sub = mag[np.ix_(live_rows, live_cols)]
        flags[np.ix_(live_rows, live_cols)] = rfi.median_flagger(
            sub, nsig=nsig
        )
    flags |= (static | dead_chan)[None, :]
    flags |= dead
    return flags, comb_offset


def waterfall_stats(wf, flags=None):
    """
    Summary statistics of a waterfall.

    Parameters
    ----------
    wf : ndarray
        2D waterfall, shape ``(ntimes, nchan)``. May be complex.
    flags : ndarray of bool or None
        Flag array of the same shape.

    Returns
    -------
    stats : dict
        ntimes, nchan, n_zero_rows (dropped integrations), zero_frac
        (fraction of exact-zero samples), flag_frac, med_power,
        max_power, peak_chan, dyn_range_db.

    """
    mag = np.abs(np.asarray(wf))
    ntimes, nchan = mag.shape
    zero_rows = ~mag.any(axis=1)
    positive = mag[mag > 0]
    med = float(np.median(positive)) if positive.size else 0.0
    peak = float(mag.max()) if mag.size else 0.0
    mean_spec = mag.mean(axis=0)
    return {
        "ntimes": int(ntimes),
        "nchan": int(nchan),
        "n_zero_rows": int(zero_rows.sum()),
        "zero_frac": float((mag == 0).mean()),
        "flag_frac": float(flags.mean()) if flags is not None else 0.0,
        "med_power": med,
        "max_power": peak,
        "peak_chan": int(mean_spec.argmax()),
        "dyn_range_db": 10 * np.log10(peak / med) if med > 0 else 0.0,
    }


@dataclass
class QuickLookResult:
    """Everything :func:`quicklook` extracts from one corr file."""

    path: Path
    pairs: list
    freqs: np.ndarray
    times: np.ndarray
    header: dict
    data: dict = field(repr=False)
    flags: dict = field(repr=False)
    stats: dict

    @property
    def tstart(self):
        """UTC start time as an ISO string."""
        return _iso(self.times[0])

    @property
    def tend(self):
        """UTC end time as an ISO string."""
        return _iso(self.times[-1])


def _iso(unix_t):
    t = datetime.fromtimestamp(float(unix_t), tz=timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def quicklook(fname, nsig=8, spacing=COMB_SPACING, pad=1):
    """
    Load a corr h5 file, flag it, and compute summary statistics.

    Parameters
    ----------
    fname : str or Path
        Correlator h5 file written by eigsep_observing.
    nsig : float
        Outlier threshold for the median flagger.
    spacing : int
        Comb tone spacing in channels.
    pad : int
        Channels flagged on each side of every comb tone.

    Returns
    -------
    res : QuickLookResult

    """
    fname = Path(fname)
    data, header, _ = read_hdf5(fname)
    pairs = sorted(data.keys(), key=lambda p: (len(p), p))
    nchan = next(iter(data.values())).shape[1]
    ntimes = next(iter(data.values())).shape[0]
    freqs = np.asarray(header.get("freqs", np.arange(nchan)))
    times = np.asarray(header.get("times", np.arange(ntimes)))
    flags, stats = {}, {}
    for p in pairs:
        f, comb_offset = flag_spectra(
            data[p], nsig=nsig, spacing=spacing, pad=pad
        )
        flags[p] = f
        s = waterfall_stats(data[p], f)
        s["comb_offset"] = comb_offset
        stats[p] = s
    return QuickLookResult(
        path=fname,
        pairs=pairs,
        freqs=freqs,
        times=times,
        header=header,
        data=data,
        flags=flags,
        stats=stats,
    )


def plot_quicklook(res, save=None, show=False):
    """
    Per-pair waterfall, spectrum, and band-power figure.

    Parameters
    ----------
    res : QuickLookResult
        Output of :func:`quicklook`.
    save : str or Path or None
        If given, save the figure here (PNG).
    show : bool
        If True, leave the figure open for interactive use.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    import matplotlib.pyplot as plt

    npairs = len(res.pairs)
    fig, axes = plt.subplots(
        npairs,
        3,
        figsize=(14, 3.2 * npairs),
        squeeze=False,
        constrained_layout=True,
    )
    dur = float(res.times[-1] - res.times[0])
    if dur < 120:
        t_unit, t_scale = "s", 1.0
    elif dur < 7200:
        t_unit, t_scale = "min", 60.0
    else:
        t_unit, t_scale = "hr", 3600.0
    t_ax = (res.times - res.times[0]) / t_scale
    extent = [res.freqs[0], res.freqs[-1], t_ax[-1], t_ax[0]]
    for i, p in enumerate(res.pairs):
        mag = np.abs(res.data[p]).astype(float)
        fl = res.flags[p]
        logmag = np.log10(mag, where=mag > 0, out=np.full_like(mag, np.nan))

        ax = axes[i, 0]
        im = ax.imshow(
            logmag, aspect="auto", extent=extent, interpolation="none"
        )
        fig.colorbar(im, ax=ax, label="log10(power)")
        ax.set_ylabel(f"{p}\n{t_unit} since start")
        if i == npairs - 1:
            ax.set_xlabel("freq [MHz]")

        ax = axes[i, 1]
        mean_spec = np.ma.masked_less_equal(mag, 0).mean(axis=0).filled(np.nan)
        ax.semilogy(res.freqs, mean_spec, lw=0.7, label="mean")
        chan_flagged = fl.mean(axis=0) > 0.5
        if chan_flagged.any():
            ax.semilogy(
                res.freqs[chan_flagged],
                mean_spec[chan_flagged],
                ".",
                ms=3,
                label="flagged",
            )
        ax.legend(fontsize="small", loc="upper right")
        ax.set_xlim(res.freqs[0], res.freqs[-1])
        if i == npairs - 1:
            ax.set_xlabel("freq [MHz]")
        ax.set_ylabel("power")

        ax = axes[i, 2]
        band = np.ma.masked_array(mag, mask=fl).mean(axis=1).filled(np.nan)
        ax.plot(t_ax, band, lw=0.8)
        ax.set_ylabel("band power (unflagged)")
        if i == npairs - 1:
            ax.set_xlabel(f"{t_unit} since start")

    s0 = res.stats[res.pairs[0]]
    fig.suptitle(
        f"{res.path.name} — {s0['ntimes']}x{s0['nchan']}, "
        f"{res.tstart} to {res.tend} UTC"
    )
    if save is not None:
        fig.savefig(save, dpi=150)
    if not show:
        plt.close(fig)
    return fig


def _print_report(res):
    print(f"\n{res.path.name}")
    print(f"  {res.tstart} to {res.tend} UTC")
    hdr = (
        f"  {'pair':>5} {'ntimes':>7} {'nchan':>6} {'zrows':>5} "
        f"{'zero%':>6} {'flag%':>6} {'med_power':>10} {'dynrng_dB':>9} "
        f"{'comb':>5}"
    )
    print(hdr)
    for p in res.pairs:
        s = res.stats[p]
        comb = "-" if s["comb_offset"] is None else str(s["comb_offset"])
        print(
            f"  {p:>5} {s['ntimes']:>7} {s['nchan']:>6} "
            f"{s['n_zero_rows']:>5} {100 * s['zero_frac']:>6.2f} "
            f"{100 * s['flag_frac']:>6.2f} "
            f"{s['med_power']:>10.3e} {s['dyn_range_db']:>9.1f} "
            f"{comb:>5}"
        )


def _collect_files(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.glob("*.h5")))
        else:
            files.append(p)
    return files


def main(argv=None):
    """Command-line entry point. Returns an exit code."""
    parser = argparse.ArgumentParser(
        description="Quick look at EIGSEP corr h5 files: stats, "
        "RFI flags, and a summary figure per file."
    )
    parser.add_argument("paths", nargs="+", help="h5 files or directories")
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="directory for PNGs (default: next to each file)",
    )
    parser.add_argument(
        "--nsig", type=float, default=8, help="outlier threshold"
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=1,
        help="channels flagged each side of comb tones",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="stats only, no figure"
    )
    parser.add_argument(
        "--show", action="store_true", help="open figures interactively"
    )
    args = parser.parse_args(argv)

    files = _collect_files(args.paths)
    if not files:
        print("no h5 files found")
        return 1

    n_fail = 0
    figs = []
    for fname in files:
        try:
            res = quicklook(fname, nsig=args.nsig, pad=args.pad)
        except Exception as e:
            print(f"\n{fname}: SKIPPED ({type(e).__name__}: {e})")
            n_fail += 1
            continue
        _print_report(res)
        if not args.no_plot:
            outdir = Path(args.outdir) if args.outdir else fname.parent
            outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / f"{fname.stem}_quicklook.png"
            figs.append(plot_quicklook(res, save=out, show=args.show))
            print(f"  figure: {out}")
    if args.show and figs:
        import matplotlib.pyplot as plt

        plt.show()
    return 1 if n_fail == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
