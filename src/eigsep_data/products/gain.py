"""Receiver gain and noise temperature: solutions on their own cadence.

``derived/gain/<version>/solutions.npz`` with ``manifest.json`` beside
it, following the ``<kind>/vN/`` convention the campaign already uses
for ``flags/`` and ``derived/smooth_model/``. The arrays are the ones
``abscal`` produces: ``sol_times`` (one per solution), ``freqs``, and
per-solution spectra ``gain`` and ``t_rx``.

**Status: not yet validated against real solutions.** The tests cover
the join on synthetic fixtures only. ``abscal``'s real output is still
loose ``.npz`` in ``abscal/`` rather than at ``derived/gain/v0/
solutions.npz``, and moving it was deliberately deferred until someone
is actually calibrating with this (Aaron, 2026-09-17). Treat the
tolerance default and the column set as untested guesses about real
data until that happens; the join logic itself is tested.

**This is the nearest-in-time join, and it is the one that can quietly
go wrong.** A solution is valid for the state the receiver was in when
it was measured; carrying it across a switch cycle or a hardware change
attaches somebody's calibration to a different regime. So the match is
bounded by an explicit ``tolerance_s``, rows outside it come back NaN
rather than reaching for the next solution, and the actual signed
offset used for every row is returned as ``gain_dt_s`` for the caller
to inspect. Nothing here guesses how stale is too stale.
"""

import numpy as np

from .base import Product, axis_fingerprint, register

#: Default match window, in seconds. Deliberately finite and
#: deliberately not generous: abscal's phase-C solutions land every few
#: minutes, so an hour means "the nearest solution in this observing
#: block", not "any solution in the campaign".
DEFAULT_TOLERANCE_S = 3600.0


@register
class Gain(Product):
    kind = "gain"
    cube = True

    def __init__(self, tolerance_s=DEFAULT_TOLERANCE_S):
        self.tolerance_s = float(tolerance_s)
        self._cache = {}

    def root_dir(self, campaign, version):
        return campaign.root / "derived" / "gain" / version

    def versions(self, campaign):
        base = campaign.root / "derived" / "gain"
        if not base.is_dir():
            return []
        return sorted(
            p.name
            for p in base.iterdir()
            if p.is_dir() and (p / "solutions.npz").is_file()
        )

    def _solutions(self, campaign, version):
        path = self.root_dir(campaign, version) / "solutions.npz"
        key = str(path)
        if key not in self._cache:
            if not path.is_file():
                return None
            with np.load(path, allow_pickle=False) as npz:
                self._cache[key] = {
                    name: npz[name]
                    for name in ("freqs", "sol_times", "gain", "t_rx")
                    if name in npz
                }
        return self._cache[key]

    def freqs(self, campaign, version):
        sols = self._solutions(campaign, version)
        if sols is None:
            raise FileNotFoundError(
                f"no solutions.npz under {self.root_dir(campaign, version)}"
            )
        return np.asarray(sols["freqs"], dtype=float)

    def fetch(self, campaign, version, fname, rows, key, band, times):
        sols = self._solutions(campaign, version)
        if sols is None:
            return None
        sol_times = np.asarray(sols["sol_times"], dtype=float)
        if sol_times.size == 0:
            return None
        times = np.asarray(times, dtype=float)

        # Nearest solution per row, then the tolerance decides whether
        # it is allowed to be used at all.
        idx = np.searchsorted(sol_times, times)
        left = np.clip(idx - 1, 0, sol_times.size - 1)
        right = np.clip(idx, 0, sol_times.size - 1)
        closer_left = np.abs(times - sol_times[left]) <= np.abs(
            sol_times[right] - times
        )
        pick = np.where(closer_left, left, right)
        dt = times - sol_times[pick]
        usable = np.isfinite(times) & (np.abs(dt) <= self.tolerance_s)

        out = {"gain_dt_s": np.where(usable, dt, np.nan)}
        for name in ("gain", "t_rx"):
            if name not in sols:
                continue
            block = np.asarray(sols[name], dtype=float)[pick]
            if band is not None:
                block = block[:, band]
            block[~usable] = np.nan
            out[name] = block
        out["_fp"] = axis_fingerprint(sols["freqs"])
        return out
