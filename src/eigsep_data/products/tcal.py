"""Temperature calibration: nearby field measurements, applied per row.

``derived/tcal/<version>/solutions.npz`` with ``manifest.json`` and
``README.md`` beside it. The payload is not a calibrated cube. It holds
the measurements a calibration is built from, on their own cadences:

- ``amb_t``/``amb_p`` and ``non_t``/``non_p``: one spectrum per
  ambient-load (RFAMB) and noise-source (RFNON) visit (a masked,
  integration-time-weighted mean since v0001; a median in v0000), with the
  receiver regime of each (``amb_regime``, ``non_regime``);
- ``s11_<dut>_t``/``s11_<dut>`` for ``ant``, ``amb`` and ``rec``:
  reflection coefficients at reference plane P on the correlator
  channels, with an ``s11_<dut>_epoch`` label;
- ``therm_t``/``therm_k``: every load-thermistor reading, which is
  logged whether or not the switch is cycling;
- ``t_ns_k``, the noise-source excess as the receiver takes it: one
  number (v0000-v0003, the 917 K of a 35 dB ENR diode behind a 30 dB
  pad), or one value per channel on ``freqs`` (from v0004, built from a
  measured noise source), applied to every row; and the matching limits
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
  within ``therm_tol_s``; or, when the caller passes a time window per
  row (``t_load_window``, e.g. a bin's first and last row times), the
  mean of the readings inside it (see :func:`load_temperature`).

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


def load_temperature(therm_t, therm_k, t, tol_s, window=None):
    """Load temperature at each time in *t* from the thermistor stream.

    Without *window*, the reading nearest each time, NaN where it is
    more than *tol_s* away. That is right for one integration, whose
    own reading it is, but not for a bin of many: the thermistor reads
    white at ~0.2 K per reading, and a single reading's error enters
    the calibrated temperature as the same kelvin offset at every
    channel of the bin.

    With ``window=(start, stop)``, arrays broadcastable to *t*, the
    load temperature is the mean of the finite readings with
    ``start <= therm_t <= stop``, its error falling as 1/sqrt(n). Where
    the window holds none (or an edge is NaN), it falls back to the
    nearest reading within *tol_s*.

    *therm_t* must be sorted, as the builder writes it.

    Returns ``(t_load, dt, n)``: ``dt`` is the distance to the nearest
    reading; ``n`` is the number of readings averaged, 0 where the
    nearest reading was used (``None`` without *window*).
    """
    tt, tk = therm_t, therm_k
    i = np.clip(np.searchsorted(tt, t), 1, max(tt.size - 1, 1))
    i = np.where(np.abs(tt[i - 1] - t) <= np.abs(tt[i] - t), i - 1, i)
    dt = np.abs(tt[i] - t)
    t_load = np.where(dt <= tol_s, tk[i], np.nan)
    if window is None:
        return t_load, dt, None

    start, stop = np.broadcast_arrays(*window, t)[:2]
    if np.any(stop < start):
        raise ValueError("t_load_window: a window ends before it starts")
    # In float64 (the builder stores float32) and relative to one
    # reading, so 300 K x 1e5 readings costs no precision.
    kk = np.asarray(tk, dtype=float)
    good = np.isfinite(kk)
    ref = kk[good][0] if good.any() else 0.0
    cs = np.concatenate([[0.0], np.cumsum(np.where(good, kk - ref, 0.0))])
    cn = np.concatenate([[0], np.cumsum(good)])
    lo = np.searchsorted(tt, start, side="left")
    hi = np.searchsorted(tt, stop, side="right")
    n = np.where(
        np.isfinite(start) & np.isfinite(stop), cn[hi] - cn[lo], 0
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ref + (cs[hi] - cs[lo]) / n
    return np.where(n > 0, mean, t_load), dt, n


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

    def fetch(
        self, campaign, version, fname, rows, key, band, times,
        t_load_window=None,
    ):
        """The calibration coefficients at *times* (see the module).

        *t_load_window* is an optional ``(start, stop)`` pair of arrays
        like *times*: each row's load temperature is then the mean of
        the thermistor readings in its window, and the output gains
        ``tcal_therm_n``, the number averaged (0 where the nearest
        reading stood in). Pass the bin's first and last row times when
        the rows are bins of binned data. Without it (the bundle's
        per-integration call) the nearest reading is used, as before.
        """
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

        # Load temperature: the nearest thermistor reading, or the mean
        # of the readings in each row's window.
        t_load, therm_dt, therm_n = load_temperature(
            sols["therm_t"], sols["therm_k"], t,
            float(sols["therm_tol_s"]), t_load_window,
        )
        t_load = np.where(applies, t_load, np.nan)
        out_cols["tcal_t_load_k"] = t_load
        out_cols["tcal_therm_dt_s"] = np.where(
            np.isfinite(t_load), therm_dt, np.nan
        )
        if therm_n is not None:
            out_cols["tcal_therm_n"] = np.where(
                np.isfinite(t_load), therm_n, 0
            )

        # One number for every channel, or one value per channel on this
        # product's axis, cut to the bundle's band like every other spectrum.
        t_ns = np.asarray(sols["t_ns_k"], dtype=float)
        if t_ns.ndim == 1:
            if t_ns.size != nchan:
                raise ValueError(
                    f"t_ns_k has {t_ns.size} channels, freqs has {nchan}"
                )
            t_ns = t_ns[band][None, :]
        elif t_ns.ndim != 0:
            raise ValueError("t_ns_k must be a number or a spectrum on freqs")
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
