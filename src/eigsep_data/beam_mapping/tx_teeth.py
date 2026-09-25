"""Choosing which transmitter comb teeth a beam fit can use.

Two model-free tests, so the choice cannot be tuned toward good fit residuals:

* **Isolation.** A tooth's strong-sample signal relative to its background
  error: the 90th percentile over good rows of ``(D - B) / B``, divided by a
  fractional background error supplied by the caller (for example the larger of
  a transmitter-off truth test and the local fit residual).
* **Neighbour coherence.** A transmitter tooth sees the beam, which changes
  smoothly with frequency, so its time series should track its nearest
  same-arm teeth. Intermittent RFI inflates the 90th-percentile signal and can
  pass isolation, but it does not track the neighbours. The test is iterative:
  the least coherent tooth below the threshold is dropped, then the rest are
  recomputed, so one bad tooth cannot drag its neighbours down. A tooth left
  with no neighbour cannot be verified and is dropped too.

Teeth alternate between the two transmitter arms: ``arm = (channel // spacing) % 2``.

Public API
----------
tooth_arms
isolation
neighbour_coherence
select_teeth
"""

import numpy as np


def tooth_arms(channels, spacing=8):
    """Transmitter arm (0 or 1) of each tooth channel."""
    return (np.asarray(channels) // spacing) % 2


def isolation(signal_over_background, good, background_error, percentile=90.0):
    """Isolation of each tooth: percentile of ``(D - B) / B`` over good rows divided by
    the fractional background error. Arrays are ``(nrow, nteeth)`` except
    ``background_error`` ``(nteeth,)``. Returns ``(isolation, signal_percentile)``."""
    sb = np.where(good, signal_over_background, np.nan)
    signal = np.nanpercentile(sb, percentile, axis=0)
    return signal / np.asarray(background_error, float), signal


def neighbour_coherence(signal, good, channels, keep, threshold=0.9, neighbour_channels=48,
                        spacing=8, n_neighbours=2):
    """Iterative neighbour-coherence selection.

    ``signal`` and ``good`` are ``(nrow, nteeth)``: the tooth signal (D - B) and its
    usable samples. For each tooth in ``keep``, the correlation over rows good for
    it and its neighbours between ``log(signal)`` and ``log(mean of the
    n_neighbours nearest same-arm kept teeth within neighbour_channels)``;
    signals are floored at 1e-3 of the tooth's maximum before the log.

    Returns ``(keep, dropped, first_pass)``: the final mask, the dropped teeth as
    ``(channel, coherence at drop)`` in drop order (NaN when no neighbour
    remained), and each initially kept tooth's first-pass coherence by index.
    ``threshold <= 0`` disables dropping.
    """
    signal = np.asarray(signal, float)
    good = np.asarray(good, bool)
    channels = np.asarray(channels)
    keep = np.asarray(keep, bool).copy()
    arms = tooth_arms(channels, spacing)

    def coherence(i):
        same = [j for j in np.flatnonzero(keep) if j != i and arms[j] == arms[i]
                and abs(int(channels[j]) - int(channels[i])) <= neighbour_channels]
        same = sorted(same, key=lambda j: abs(int(channels[j]) - int(channels[i])))[:n_neighbours]
        if not same:
            return np.nan
        ok = good[:, i] & np.all(good[:, same], axis=1)
        if ok.sum() < 3:
            return np.nan
        floor = 1e-3 * np.nanmax(signal[:, i])
        x = np.log(np.clip(signal[ok, i], floor, None))
        y = np.log(np.clip(np.mean(signal[ok][:, same], axis=1), floor, None))
        return np.corrcoef(x, y)[0, 1]

    first_pass = {int(i): coherence(i) for i in np.flatnonzero(keep)}
    dropped = []
    while threshold > 0 and keep.any():
        c = {i: coherence(i) for i in np.flatnonzero(keep)}
        worst = min(c, key=lambda i: c[i] if np.isfinite(c[i]) else -1)
        if np.isfinite(c[worst]) and c[worst] >= threshold:
            break
        keep[worst] = False
        dropped.append((int(channels[worst]), float(c[worst])))
    return keep, dropped, first_pass


def select_teeth(channels, spectra_teeth, background, good, background_error,
                 isolation_min=10.0, good_fraction_min=0.5, coherence_min=0.9,
                 neighbour_channels=48, spacing=8):
    """Apply isolation, good-fraction and neighbour-coherence rules.

    ``spectra_teeth`` and ``background`` are ``(nrow, nteeth)`` (the data at the tooth
    channels and the background under them); ``good`` marks usable samples.
    Returns a dict of per-tooth arrays: ``channel``, ``arm``, ``isolation``,
    ``signal_p90_over_bg``, ``good_fraction``, ``coherence_first_pass``,
    ``coherence_at_drop``, ``selected``, and ``reason`` ('selected', 'good_fraction',
    'isolation' or 'coherence').
    """
    channels = np.asarray(channels)
    good = np.asarray(good, bool)
    signal = np.asarray(spectra_teeth, float) - np.asarray(background, float)
    iso, sig = isolation(signal / background, good, background_error)
    good_fraction = good.mean(axis=0)
    passes = (good_fraction >= good_fraction_min) & (iso >= isolation_min)
    keep, dropped, first = neighbour_coherence(signal, good, channels, passes, coherence_min,
                                               neighbour_channels, spacing)
    at_drop = np.full(len(channels), np.nan)
    for c, v in dropped:
        at_drop[np.flatnonzero(channels == c)[0]] = v
    reason = np.where(keep, 'selected',
                      np.where(good_fraction < good_fraction_min, 'good_fraction',
                               np.where(iso < isolation_min, 'isolation', 'coherence')))
    return dict(channel=channels, arm=tooth_arms(channels, spacing), isolation=iso,
                signal_p90_over_bg=sig, good_fraction=good_fraction,
                coherence_first_pass=np.array([first.get(i, np.nan) for i in range(len(channels))]),
                coherence_at_drop=at_drop, selected=keep, reason=reason)
