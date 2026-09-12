"""Interactive flipbook over a :class:`~eigsep_data.index.Selection`.

Successor to the ``StateBrowser`` prototype that lived in
``notebooks/christian/deployment5/rf_state_tools.py`` (deleted when this
module landed; it is in the git history), rebuilt on the index so the
state vocabulary stays open and the exclusions are visible. The
prototype looked each state up in the fixed list its own scan had seen,
which raised on any state that scan had missed.

``ipywidgets`` is imported only when the controls are built, so the
module is importable -- and testable -- without it. Install it with the
``vis`` extra.
"""

import numpy as np

from .data import EigsepData


class StateBrowser:
    """
    Flip through per-file spectra for a selection of integrations.

    Usage in a ``%matplotlib widget`` notebook::

        idx = MetadataIndex("data/deployment5_filtered")
        sel = idx.select(rfswitch="RFAMB", files="corr_20260717*")
        b = StateBrowser(sel)

    Parameters
    ----------
    selection : eigsep_data.index.Selection
    keys : sequence of str
        Data keys to plot.
    vmin, vmax : float
        log10 power limits for the waterfall.
    controls : bool
        Build the ipywidgets controls. ``False`` for headless use.

    Attributes
    ----------
    pos : int
        Position of the file on screen in ``selection.files``.
    loaded : eigsep_data.EigsepData
        The rows behind the current figure -- only the selected rows of
        the current file. Replaced on every :meth:`goto`.
    """

    def __init__(
        self, selection, keys=("0", "4"), vmin=4, vmax=6.6, controls=True
    ):
        import matplotlib.pyplot as plt

        if selection.nrows == 0:
            raise ValueError(
                "Selection contains no integrations; nothing to browse."
            )
        self.selection = selection
        self.keys = list(keys)
        self.vmin, self.vmax = vmin, vmax
        self._files = selection.files
        self.pos = 0
        self.loaded = None

        self.fig, (self.ax_w, self.ax_s) = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            layout="constrained",
            gridspec_kw={"height_ratios": [1, 1.4]},
        )
        if controls:
            self._build_controls()
        self.goto(0)

    @property
    def nfiles(self):
        return len(self._files)

    def _read(self, pos):
        name = self._files[pos]
        sub = self.selection.select(files=name)
        return name, EigsepData.from_selection(
            sub, keys=self.keys, missing="nan"
        )

    def goto(self, pos):
        """Draw the file at index *pos*."""
        self.pos = int(np.clip(pos, 0, self.nfiles - 1))
        name, loaded = self._read(self.pos)
        self.loaded = loaded
        # missing="nan" hands back every requested key, including the
        # ones this file never recorded, so "is it in loaded.data" does
        # not separate them -- having any finite sample does. A key the
        # wiring phase lacks is then left out of both panels instead of
        # drawing a panel of NaN over the key that is live.
        live = [k for k in self.keys if np.isfinite(loaded.data[k]).any()]
        # A selection built with a narrower streams= has no rfswitch
        # column; browsing is still perfectly meaningful without it.
        states = sorted(set(loaded.meta.get("rfswitch", [])))

        self.ax_w.clear()
        self.ax_s.clear()
        first = live[0] if live else None
        if first is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                img = np.log10(np.abs(loaded.data[first]))
            self.ax_w.imshow(
                img,
                aspect="auto",
                cmap="plasma",
                interpolation="none",
                vmin=self.vmin,
                vmax=self.vmax,
                extent=[
                    loaded.freq.min(),
                    loaded.freq.max(),
                    len(loaded.times),
                    0,
                ],
            )
            self.ax_w.set_ylabel(f"row (key {first})")
        self.ax_w.set_title(
            f"[{self.pos + 1}/{self.nfiles}] {name}  "
            f"{len(loaded.times)} rows  {','.join(states)}",
            fontsize=10,
        )
        for key in live:
            mag = np.abs(loaded.data[key])
            mean = np.nanmean(mag, axis=0)
            lo, hi = np.nanpercentile(mag, [10, 90], axis=0)
            (line,) = self.ax_s.plot(
                loaded.freq, mean, lw=1, label=f"key {key}"
            )
            self.ax_s.fill_between(
                loaded.freq,
                lo,
                hi,
                alpha=0.25,
                color=line.get_color(),
                lw=0,
            )
        self.ax_s.set_yscale("log")
        self.ax_s.set_xlabel("Frequency [MHz]")
        self.ax_s.set_ylabel("Power [counts]")
        self.ax_s.legend(loc="upper right", fontsize=8)
        self.fig.canvas.draw_idle()

    def _build_controls(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError(
                "StateBrowser controls need ipywidgets. Install the "
                "vis extra: pip install -e '.[vis]'"
            ) from exc

        slider = widgets.IntSlider(
            0,
            0,
            self.nfiles - 1,
            description="file",
            layout=widgets.Layout(width="55%"),
        )
        play = widgets.Play(interval=400, min=0, max=self.nfiles - 1, step=1)
        widgets.jslink((play, "value"), (slider, "value"))
        slider.observe(lambda ch: self.goto(ch["new"]), names="value")
        display(widgets.HBox([play, slider]))
