"""Temperature calibration: nearby field measurements, applied per row.

``derived/tcal/<version>/solutions.npz`` with ``manifest.json`` and
``README.md`` beside it. The payload is not a calibrated cube. It holds
the measurements a calibration is built from, on their own cadences:

- ``amb_t``/``amb_p`` and ``non_t``/``non_p``: one median spectrum per
  ambient-load (RFAMB) and noise-source (RFNON) visit, with the
  receiver regime of each (``amb_regime``, ``non_regime``);
- ``s11_<dut>_t``/``s11_<dut>`` for ``ant``, ``amb`` and ``rec``:
  reflection coefficients at reference plane P on the correlator
  channels, with an ``s11_<dut>_epoch`` label;
- ``therm_t``/``therm_k``: every load-thermistor reading, which is
  logged whether or not the switch is cycling;
- ``t_ns_k``, the noise-source excess, and the matching limits
  ``max_cal_gap_s``, ``max_s11_gap_s``, ``s11_hold_s``, ``therm_tol_s``;
- ``keys``: the correlator inputs the calibration applies to.

:meth:`Tcal.fetch` builds each row's calibration from them:

- the load and noise-source spectra are interpolated linearly in time
  between the visits that bracket the row, only within one receiver
  regime and only across a gap no longer than ``max_cal_gap_s``;
- each reflection coefficient is interpolated the same way between the
  bracketing sweeps of one epoch (``max_s11_gap_s``), or held from the
  nearest sweep within ``s11_hold_s`` where no pair brackets the row.
  The builder keeps ``s11_hold_s`` under half of every gap between
  epochs, so a held sweep is always from the row's own epoch;
- the load temperature is the thermistor reading nearest the row,
  within ``therm_tol_s``.

Anything that fails a limit is NaN, never extrapolated. The calibration
is linear in the row's power, ``T = scale * P + offset``
(``eigsep_cal.dicke``), so it comes back as coefficients:

- ``tstar_scale``/``tstar_offset``: the three-state Y factor T*, valid
  for every switch state;
- ``scale``/``offset``: T* with the antenna, load and receiver S11
  corrections, valid for antenna (RFANT) rows only.

:attr:`eigsep_data.bundle.Bundle.calibrated` and
:attr:`~eigsep_data.bundle.Bundle.t_star` apply them and leave
``Bundle.data`` raw.
"""

import numpy as np

from .base import Product, axis_fingerprint, register

_DUTS = ("ant", "amb", "rec")


def _bracket(node_t, label, t, max_gap_s):
    """Bracketing node pair and linear weight for each time in *t*.

    Returns ``(left, right, w, ok, width)``: the value at ``t`` is
    ``(1 - w) * v[left] + w * v[right]``. ``ok`` is False where no pair
    of nodes with the same *label* and at most *max_gap_s* apart
    brackets the time.
    """
    n = node_t.size
    right = np.searchsorted(node_t, t, side="left")
    left = right - 1
    inside = (left >= 0) & (right < n)
    left_c = np.clip(left, 0, max(n - 1, 0))
    right_c = np.clip(right, 0, max(n - 1, 0))
    # A row exactly on a node takes that node alone.
    on_node = (right < n) & (node_t[right_c] == t)
    width = np.where(inside, node_t[right_c] - node_t[left_c], np.nan)
    ok = (
        inside
        & (label[left_c] == label[right_c])
        & (width <= max_gap_s)
    ) | on_node
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(on_node, 1.0, (t - node_t[left_c]) / width)
    w = np.where(ok, w, np.nan)
    return left_c, right_c, w, ok, np.where(on_node, 0.0, width)


def _interp(values, left, right, w, ok):
    out = (1 - w)[:, None] * values[left] + w[:, None] * values[right]
    out[~ok] = np.nan
    return out


@register
class Tcal(Product):
    kind = "tcal"
    cube = True

    def __init__(self):
        self._cache = {}

    def root_dir(self, campaign, version):
        return campaign.root / "derived" / "tcal" / version

    def versions(self, campaign):
        base = campaign.root / "derived" / "tcal"
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
                self._cache[key] = {name: npz[name] for name in npz.files}
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
        from eigsep_cal.dicke import (
            receiver_s11_coefficients,
            tstar_coefficients,
        )

        t = np.asarray(times, dtype=float)
        nchan = sols["freqs"].size
        band = slice(0, nchan) if band is None else band

        def cut(a):
            return np.asarray(a)[:, band]

        out_cols = {}
        applies = str(key) in set(np.asarray(sols["keys"]).astype(str))

        # Load and noise-source spectra: bracketing visits, one regime.
        p = {}
        for name in ("amb", "non"):
            left, right, w, ok, width = _bracket(
                sols[f"{name}_t"], sols[f"{name}_regime"], t,
                float(sols["max_cal_gap_s"]),
            )
            ok &= applies
            p[name] = _interp(cut(sols[f"{name}_p"]), left, right, w, ok)
            out_cols[f"tcal_{name}_gap_s"] = np.where(ok, width, np.nan)

        # Reflection coefficients: bracketing sweeps, else the nearest
        # sweep of the same epoch within the hold limit.
        gam = {}
        for dut in _DUTS:
            st = sols[f"s11_{dut}_t"]
            ep = sols[f"s11_{dut}_epoch"]
            left, right, w, ok, _ = _bracket(
                st, ep, t, float(sols["max_s11_gap_s"])
            )
            g = _interp(cut(sols[f"s11_{dut}"]), left, right, w, ok)
            near = np.where(
                np.abs(t - st[left]) <= np.abs(st[right] - t), left, right
            )
            dt = np.abs(t - st[near])
            # The builder guarantees s11_hold_s is shorter than half of
            # every gap between epochs, so a sweep this close is always
            # in the row's own epoch.
            hold = ~ok & (dt <= float(sols["s11_hold_s"]))
            if np.any(hold):
                g[hold] = cut(sols[f"s11_{dut}"])[near[hold]]
            use = (ok | hold) & applies
            g[~use] = np.nan
            gam[dut] = g
            out_cols[f"tcal_s11_{dut}_dt_s"] = np.where(use, dt, np.nan)

        # Load temperature: the nearest thermistor reading.
        tt, tk = sols["therm_t"], sols["therm_k"]
        i = np.clip(np.searchsorted(tt, t), 1, max(tt.size - 1, 1))
        i = np.where(np.abs(tt[i - 1] - t) <= np.abs(tt[i] - t), i - 1, i)
        therm_dt = np.abs(tt[i] - t)
        t_load = np.where(
            applies & (therm_dt <= float(sols["therm_tol_s"])), tk[i], np.nan
        )
        out_cols["tcal_t_load_k"] = t_load
        out_cols["tcal_therm_dt_s"] = np.where(
            np.isfinite(t_load), therm_dt, np.nan
        )

        t_ns = float(sols["t_ns_k"])
        a, b = tstar_coefficients(p["non"], p["amb"], t_ns, t_load[:, None])
        scale, offset = receiver_s11_coefficients(
            a, b, t_load[:, None], gam["ant"], gam["amb"], gam["rec"]
        )
        out = {
            "tstar_scale": a,
            "tstar_offset": b,
            "scale": np.real(scale),
            "offset": np.real(offset),
            **out_cols,
        }
        out["_fp"] = axis_fingerprint(sols["freqs"])
        return out
