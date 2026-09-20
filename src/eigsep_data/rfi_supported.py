"""Supported separable-DPSS RFI flagging and product writing.

This is the ``v3-beta`` algorithm developed in the Marjum RFI notebooks.
It fits one smooth time/frequency background without forming a separate
frequency inverse for every flag pattern.  A design-support calculation
marks places where that fit is too weakly constrained; the public model is
NaN there and the flag bitfield records why the sample was excluded.

The numerical API is campaign independent.  :func:`run_selection` is the
thin adapter from an :class:`eigsep_data.index.Selection`, and
:func:`write_products` writes the existing ``flags`` and ``smooth_model``
product layouts understood by :meth:`Selection.load_bundle`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import warnings

import h5py
import numpy as np
import pandas as pd
from hera_filters.dspec import dpss_operator
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.ndimage import maximum_filter, median_filter
from scipy.sparse.linalg import LinearOperator, cg

REPORT_BANDS = {
    "50-88 (DTV-lo)": (50.0, 88.0),
    "88-108 (FM)": (88.0, 108.0),
    "108-137": (108.0, 136.5),
    "137-138 (Orbcomm)": (136.5, 138.5),
    "138-174": (138.5, 174.0),
    "174-216 (DTV-hi)": (174.0, 216.0),
    "216-235": (216.0, 235.0),
}

FLAG_BITS = {
    "0": {"name": "non_sky_switch_state", "rfi": False},
    "1": {"name": "invalid_input_or_domain", "rfi": False},
    "2": {"name": "positive_auto_excess", "rfi": True},
    "3": {"name": "negative_or_model_failure", "rfi": False},
    "4": {"name": "cross_change", "rfi": True},
    "5": {"name": "band_group_trigger", "rfi": True},
    "6": {"name": "comb_group_trigger", "rfi": True},
    "7": {"name": "unsupported_background", "rfi": False},
    "8": {"name": "guard", "rfi": False},
}
FLAG_MEANINGS = {
    "non_sky_switch_state": "receiver switch is not RFANT; not sky and not RFI",
    "invalid_input_or_domain": "missing/nonpositive input or outside the tested 35-235 MHz domain",
    "positive_auto_excess": "positive air-auto residual above its point threshold",
    "negative_or_model_failure": "large negative residual or nonfinite residual/model",
    "cross_change": "complex cross coherence changed relative to its robust smooth background",
    "band_group_trigger": "a report band or 8-channel tile crossed its group threshold",
    "comb_group_trigger": "a clock-aligned 8/4/2-channel comb group crossed its threshold",
    "unsupported_background": "design-noise or ridge-prior check rejects the model prediction",
    "guard": "time/frequency guard grown around another exclusion reason",
}
BIT_BY_REASON = {v["name"]: int(k) for k, v in FLAG_BITS.items()}
DEFAULT_VERSION = "v3-beta"


@dataclass(frozen=True)
class RFIConfig:
    """Parameters for the supported-DPSS v3-beta algorithm."""

    band_mhz: tuple[float, float] = (35.0, 250.0)
    tested_band_mhz: tuple[float, float] = (35.0, 235.0)
    sky_state: str = "RFANT"
    freq_halfwidth_s: float = 50e-9
    spectral_correction_halfwidth_s: float = 300e-9
    spectral_correction_band_mhz: tuple[float, float] = (35.0, 88.0)
    spectral_correction_taper_mhz: float = 5.0
    spectral_correction_svd_cutoff: float = 1e-8
    time_halfwidth_hz: float = 1e-3
    eigenvalue_cutoff: float = 1e-12
    fit_clip: float = 8.0
    point_cut: float = 6.0
    other_point_cut: float = 8.0
    negative_cut: float = 8.0
    group_cut: float = 6.0
    group_cell_cut: float = 2.0
    group_fraction: float = 0.35
    tile_channels: int = 8
    comb_cut: float = 6.0
    comb_tooth_cut: float = 2.0
    comb_fraction: float = 0.10
    comb_min_teeth: int = 16
    ridge: float = 1e-6
    cg_rtol: float = 1e-8
    cg_maxiter: int = 300
    robust_rounds: int = 50
    mask_change_tol: float = 5e-4
    cross_cut: float = 8.0
    cross_clip: float = 8.0
    cross_iterations: int = 80
    cross_step_tol: float = 1e-3
    support_inflation: float = 5.0
    prior_fraction: float = 0.10
    refit_rounds: int = 2
    time_guard: int = 1
    frequency_guard: int = 0
    gap_factor: float = 3.0


@dataclass
class RFIResult:
    """Aligned output arrays and diagnostics from :func:`flag_arrays`."""

    flags: np.ndarray
    model: np.ndarray
    model_raw: np.ndarray
    residual_z: np.ndarray
    support_ok: np.ndarray
    fit_keep: np.ndarray
    reasons: dict[str, np.ndarray]
    inflation: np.ndarray
    prior_fraction: np.ndarray
    cross_background: np.ndarray
    cross_score: np.ndarray
    freqs_mhz: np.ndarray
    times: np.ndarray
    meta: pd.DataFrame | None
    config: RFIConfig
    diagnostics: dict

    @property
    def mask(self):
        """Boolean exclusion mask (every nonzero reason bit)."""
        return self.flags != 0

    @property
    def rfi_mask(self):
        """Boolean mask containing only reasons classified as RFI."""
        value = np.uint16(0)
        for bit, spec in FLAG_BITS.items():
            if spec["rfi"]:
                value |= np.uint16(1 << int(bit))
        return (self.flags & value) != 0


def _bases(times, freqs, config):
    af = dpss_operator(
        freqs * 1e6,
        [0],
        [config.freq_halfwidth_s],
        eigenval_cutoff=[config.eigenvalue_cutoff],
    )[0].real
    at = dpss_operator(
        times - times[0],
        [0],
        [config.time_halfwidth_hz],
        eigenval_cutoff=[config.eigenvalue_cutoff],
    )[0].real
    return at, af


class _TensorFit:
    def __init__(self, times, freqs, data, config):
        at, af = _bases(times, freqs, config)
        self.qt = np.linalg.qr(at)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            baseline = np.nanmedian(
                np.where(np.isfinite(data), np.maximum(data, 1), np.nan),
                axis=0,
            )
        baseline = np.nan_to_num(baseline, nan=1e4)
        self.scale = np.maximum(median_filter(baseline, size=31), 1e4)
        self.qf = np.linalg.qr(af / self.scale[:, None])[0]
        self.config = config
        self.cross_qt = self.qt.copy()
        base_nt, base_nf = self.qt.shape[1], self.qf.shape[1]
        extra = self._spectral_correction_modes(freqs)
        if extra.shape[1]:
            constant = np.ones(len(times)) / np.sqrt(len(times))
            self.qt = np.column_stack([self.qt, constant])
            self.qf = np.column_stack([self.qf, extra])
        self.shape = (self.qt.shape[1], self.qf.shape[1])
        self.active_mask = np.zeros(self.shape, dtype=bool)
        self.active_mask[:base_nt, :base_nf] = True
        if extra.shape[1]:
            self.active_mask[base_nt, base_nf:] = True
        self.active = np.flatnonzero(self.active_mask.ravel())
        self.ncoeff = len(self.active)
        self.stationary_frequency_modes = extra.shape[1]
        self.history = []

    def _spectral_correction_modes(self, freqs):
        config = self.config
        width = config.spectral_correction_halfwidth_s
        lo, hi = config.spectral_correction_band_mhz
        selected = (freqs >= lo) & (freqs < hi)
        if width <= 0 or selected.sum() < 3:
            return np.empty((len(freqs), 0))
        modes = dpss_operator(
            freqs[selected] * 1e6,
            [0],
            [width],
            eigenval_cutoff=[config.eigenvalue_cutoff],
        )[0].real
        taper_width = config.spectral_correction_taper_mhz
        if taper_width > 0:
            edge = np.minimum(
                np.clip((freqs[selected] - lo) / taper_width, 0, 1),
                np.clip((hi - freqs[selected]) / taper_width, 0, 1),
            )
            modes *= np.sin(np.pi * edge / 2)[:, None] ** 2
        extra = np.zeros((len(freqs), modes.shape[1]))
        extra[selected] = modes / self.scale[selected, None]
        for _ in range(2):
            extra -= self.qf @ (self.qf.T @ extra)
        left, values, _ = np.linalg.svd(extra, full_matrices=False)
        if not len(values) or values[0] == 0:
            return np.empty((len(freqs), 0))
        keep = values > values[0] * config.spectral_correction_svd_cutoff
        return left[:, keep]

    def _expand(self, coefficients):
        full = np.zeros(np.prod(self.shape))
        full[self.active] = coefficients
        return full.reshape(self.shape)

    def solve(self, data, keep):
        started = time.perf_counter()
        qt, qf = self.qt, self.qf
        weights = np.asarray(keep, dtype=float)
        if weights.sum() <= self.ncoeff:
            raise ValueError(
                "insufficient sky samples for the supported-DPSS fit: "
                f"{int(weights.sum())} samples for {self.ncoeff} "
                "coefficients"
            )
        y = np.where(keep, data / self.scale, 0.0)
        rhs = (qt.T @ y @ qf).ravel()[self.active]
        ridge = self.config.ridge

        def matvec(coefficients):
            block = self._expand(coefficients)
            normal = qt.T @ (weights * (qt @ block @ qf.T)) @ qf
            return normal.ravel()[self.active] + ridge * coefficients

        op = LinearOperator(
            (self.ncoeff, self.ncoeff), matvec=matvec, dtype=float
        )
        if self.stationary_frequency_modes:
            diagonal = ((qt**2).T @ weights @ (qf**2)).ravel()[
                self.active
            ] + ridge
            pre = LinearOperator(
                op.shape, matvec=lambda value: value / diagonal, dtype=float
            )
            factorizations = 0
        else:
            ct = cho_factor(
                qt.T @ (weights.mean(1)[:, None] * qt) / weights.mean()
                + ridge * np.eye(self.shape[0])
            )
            cf = cho_factor(
                qf.T @ (weights.mean(0)[:, None] * qf)
                + ridge * np.eye(self.shape[1])
            )

            def precondition(value):
                block = cho_solve(ct, value.reshape(self.shape))
                return cho_solve(cf, block.T).T.ravel()

            pre = LinearOperator(op.shape, matvec=precondition, dtype=float)
            factorizations = 2
        iterations = []
        coeff, info = cg(
            op,
            rhs.ravel(),
            M=pre,
            rtol=self.config.cg_rtol,
            maxiter=self.config.cg_maxiter,
            callback=lambda _: iterations.append(1),
        )
        relative = np.linalg.norm(matvec(coeff) - rhs.ravel()) / max(
            np.linalg.norm(rhs), 1e-30
        )
        record = {
            "seconds": time.perf_counter() - started,
            "iterations": len(iterations),
            "info": int(info),
            "relative_normal_residual": float(relative),
            "factorizations": factorizations,
            "kept_fraction": float(weights.mean()),
            "coefficients": self.ncoeff,
            "stationary_frequency_modes": self.stationary_frequency_modes,
        }
        self.history.append(record)
        if info != 0:
            raise RuntimeError(
                f"DPSS conjugate-gradient solve failed: {record}"
            )
        return (qt @ self._expand(coeff) @ qf.T) * self.scale


def residual_z(data, model, normalization):
    """Radiometer-normalized fractional residual."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            model > 0,
            (data / np.maximum(model, 1) - 1) * normalization,
            np.inf,
        )


def _robust_model(data, eligible, solver, normalization, config):
    keep = eligible.copy()
    trace = []
    for iteration in range(config.robust_rounds):
        model = solver.solve(data, keep)
        z = residual_z(data, model, normalization)
        updated = eligible & np.isfinite(z) & (z < config.fit_clip)
        change = float(np.mean(updated != keep))
        trace.append(
            {
                "round": iteration,
                "changed_fraction": change,
                "kept_fraction": float(updated.mean()),
            }
        )
        keep = updated
        if change < config.mask_change_tol:
            break
    return solver.solve(data, keep), keep, trace


def _support(solver, keep):
    started = time.perf_counter()
    qt, qf = solver.qt, solver.qf
    nt, nf = solver.shape
    tt = np.einsum("ta,tb->abt", qt, qt).reshape(nt * nt, -1)
    moments = (tt @ keep.astype(float)).reshape(nt, nt, -1)
    full_info = np.einsum(
        "abf,fi,fj->aibj", moments, qf, qf, optimize=True
    ).reshape(nt * nf, nt * nf)
    active = np.ix_(solver.active, solver.active)
    info = full_info[active]
    values, vectors = eigh((info + info.T) * 0.5)
    values = np.maximum(values, 0)
    ridge = solver.config.ridge
    inverse = (vectors / (values + ridge)) @ vectors.T
    inverse2 = (vectors / (values + ridge) ** 2) @ vectors.T

    def diagonal(covariance):
        full = np.zeros((nt * nf, nt * nf))
        full[active] = covariance
        fc = np.einsum(
            "aibj,fi,fj->abf",
            full.reshape(nt, nf, nt, nf),
            qf,
            qf,
            optimize=True,
        )
        return tt.T @ fc.reshape(nt * nt, -1)

    variance = np.maximum(diagonal(inverse), 1e-30)
    nominal = (qt**2) @ solver.active_mask.astype(float) @ (qf**2).T
    inflation = np.sqrt(variance / nominal)
    prior = np.clip(ridge * diagonal(inverse2) / variance, 0, 1)
    return inflation, prior, time.perf_counter() - started


def _cross_background(cross_z, qt, reference_valid, config):
    started = time.perf_counter()
    usable = reference_valid & np.isfinite(cross_z)
    active = np.any(usable, axis=0)
    if not active.any():
        raise ValueError("no valid sky samples for cross-background fit")
    good = usable[:, active]
    data = np.where(good, cross_z[:, active], 0)
    rank = qt.shape[1]
    qt_outer = np.einsum("ta,tb->abt", qt, qt).reshape(rank * rank, -1)
    coeff = qt.T @ data
    fitted = qt @ coeff
    history = []
    for _ in range(config.cross_iterations):
        residual = data - fitted
        weights = good * np.minimum(
            1.0, config.cross_clip / np.maximum(abs(residual), 1e-12)
        )
        rhs = qt.T @ (weights * data)
        gram = (qt_outer @ weights).reshape(rank, rank, -1).transpose(2, 0, 1)
        gram += config.ridge * np.eye(rank)[None, :, :]
        coeff = np.linalg.solve(gram, rhs.T[:, :, None])[:, :, 0].T
        updated = qt @ coeff
        change = float(np.max(abs(updated - fitted)))
        history.append(change)
        fitted = updated
        if change < config.cross_step_tol:
            break
    if history[-1] >= config.cross_step_tol:
        raise RuntimeError(
            "cross-background solve did not converge: last step "
            f"{history[-1]:.4g}"
        )
    model = np.zeros_like(cross_z)
    model[:, active] = fitted
    residual = cross_z - model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sr = 1.4826 * np.nanmedian(
            np.where(usable, abs(residual.real), np.nan), axis=0
        )
        si = 1.4826 * np.nanmedian(
            np.where(usable, abs(residual.imag), np.nan), axis=0
        )
    scale = np.maximum(np.nan_to_num(np.maximum(sr, si), nan=1.0), 1.0)
    return (
        model,
        abs(residual) / scale,
        {
            "iterations": len(history),
            "last_step": history[-1],
            "reached_step_tolerance": True,
            "systems_solved": int(active.sum() * len(history)),
            "system_rank": rank,
            "seconds": time.perf_counter() - started,
        },
    )


def _temporal_center_scale(z, usable):
    values = np.where(usable & np.isfinite(z), z, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(values, axis=0)
        scale = 1.4826 * np.nanmedian(abs(values - center), axis=0)
    return np.nan_to_num(center), np.maximum(np.nan_to_num(scale, nan=1), 1)


def _detect(
    data,
    model,
    cross_score,
    valid,
    normalization,
    freqs,
    channels,
    reference_valid,
    config,
):
    reference_valid = reference_valid & valid
    z = residual_z(data, model, normalization)
    threshold = np.full(len(freqs), config.other_point_cut)
    for lo, hi in REPORT_BANDS.values():
        threshold[(freqs >= lo) & (freqs < hi)] = config.point_cut
    reasons = {
        "invalid_input_or_domain": ~valid,
        "cross_change": cross_score > config.cross_cut,
        "positive_auto_excess": z > threshold,
        "negative_or_model_failure": (z < -config.negative_cut)
        | ~np.isfinite(z),
    }
    center, scale = _temporal_center_scale(
        z, reference_valid & (z < config.fit_clip)
    )
    standardized = np.nan_to_num(
        (z - center) / scale, nan=0, posinf=0, neginf=0
    )
    finite_z = np.where(np.isfinite(z), z, 0)
    curvature = finite_z - 0.5 * (
        np.roll(finite_z, 1, axis=1) + np.roll(finite_z, -1, axis=1)
    )
    curvature = np.nan_to_num(curvature) / np.sqrt(1.5)
    evidence = valid & np.isfinite(z)
    evidence &= np.roll(evidence, 1, axis=1)
    evidence &= np.roll(evidence, -1, axis=1)
    evidence[:, [0, -1]] = False
    evidence &= ~((freqs >= 88) & (freqs < 108))[None, :]
    _, curvature_scale = _temporal_center_scale(
        curvature, evidence & reference_valid
    )
    tooth = curvature / curvature_scale
    phase_on = np.zeros((len(data), 8), dtype=bool)
    comb_scores = np.zeros((len(data), 8))
    regions = ((35, 108), (108, 174), (174, 235))
    for phase in range(8):
        selected = channels % 8 == phase
        good = evidence[:, selected]
        count = good.sum(1)
        score = (np.clip(tooth[:, selected], -3, 8) * good).sum(1)
        score /= np.sqrt(np.maximum(count, 1))
        hits = (tooth[:, selected] > config.comb_tooth_cut) & good
        coverage = np.ones(len(data), dtype=bool)
        for lo, hi in regions:
            region = (freqs[selected] >= lo) & (freqs[selected] < hi)
            coverage &= hits[:, region].sum(1) >= 2
        on = (score > config.comb_cut) & (
            hits.sum(1) >= np.maximum(4, config.comb_fraction * count)
        )
        on &= coverage & (count >= config.comb_min_teeth)
        comb_scores[:, phase] = score
        phase_on[:, phase] = on
    comb = phase_on[:, channels % 8]
    reasons["comb_group_trigger"] = comb
    bands = np.zeros_like(valid)
    band_events = {}
    for label, (lo, hi) in REPORT_BANDS.items():
        indices = np.flatnonzero((freqs >= lo) & (freqs < hi))
        groups = [(label, indices)] + [
            (f"{label}/tile{k}", indices[k : k + config.tile_channels])
            for k in range(0, len(indices), config.tile_channels)
        ]
        for name, selected in groups:
            if len(selected) < 3:
                continue
            good = valid[:, selected] & ~comb[:, selected]
            count = good.sum(1)
            raw = (np.clip(standardized[:, selected], -3, 6) * good).sum(
                1
            ) / np.sqrt(np.maximum(count, 1))
            ref = raw[np.any(reference_valid[:, selected], axis=1)]
            median = np.median(ref) if len(ref) else 0
            sigma = (
                max(1.0, 1.4826 * np.median(abs(ref - median)))
                if len(ref)
                else 1.0
            )
            score = (raw - median) / sigma
            fraction = (
                (standardized[:, selected] > config.group_cell_cut) & good
            ).sum(1) / np.maximum(count, 1)
            on = (
                (score > config.group_cut)
                & (fraction >= config.group_fraction)
                & (count >= 3)
            )
            bands[np.ix_(on, selected)] = True
            band_events[name] = on
    reasons["band_group_trigger"] = bands
    for name in reasons:
        if name != "invalid_input_or_domain":
            reasons[name] &= valid
    return reasons, z, {"band_events": band_events, "comb_scores": comb_scores}


def _one_segment(data, ground, cross, times, freqs, dt, sky, config):
    started = time.perf_counter()
    df_hz = float(np.median(np.diff(freqs))) * 1e6
    normalization = np.sqrt(2 * dt[:, None] * df_hz)
    domain = (freqs >= config.tested_band_mhz[0]) & (
        freqs < config.tested_band_mhz[1]
    )
    valid_input = (
        np.isfinite(data)
        & np.isfinite(ground)
        & np.isfinite(cross)
        & (data > 0)
        & (ground > 0)
        & domain[None, :]
    )
    valid = valid_input & sky[:, None]
    cross_z = cross / np.sqrt(np.maximum(data, 1) * np.maximum(ground, 1))
    cross_z *= normalization
    channels = np.rint(freqs / (df_hz / 1e6)).astype(int)
    solver = _TensorFit(times, freqs, data[sky], config)
    reference = valid.copy()
    model, robust_keep, robust_trace = _robust_model(
        data, reference, solver, normalization, config
    )
    cross_model, cross_score, cross_stats = _cross_background(
        cross_z, solver.cross_qt, reference, config
    )
    keep = robust_keep
    refit_trace = []
    for iteration in range(config.refit_rounds):
        reasons, z, _ = _detect(
            data,
            model,
            cross_score,
            valid,
            normalization,
            freqs,
            channels,
            reference,
            config,
        )
        reject = (
            reasons["positive_auto_excess"]
            | reasons["negative_or_model_failure"]
            | reasons["band_group_trigger"]
            | reasons["comb_group_trigger"]
            | (reasons["cross_change"] & (z > 2))
        )
        updated = robust_keep & ~reject
        change = float(np.mean(updated != keep))
        keep = updated
        model = solver.solve(data, keep)
        refit_trace.append(
            {
                "round": iteration,
                "changed_fraction": change,
                "kept_fraction": float(keep.mean()),
            }
        )
        if change < config.mask_change_tol:
            break
    inflation, prior, support_seconds = _support(solver, keep)
    supported = (
        (inflation <= config.support_inflation)
        & (prior <= config.prior_fraction)
        & (model > 0)
        & np.isfinite(model)
        & domain[None, :]
    )
    reasons, z, event_stats = _detect(
        data,
        model,
        cross_score,
        valid & supported,
        normalization,
        freqs,
        channels,
        reference & supported,
        config,
    )
    reasons["non_sky_switch_state"] = np.broadcast_to(
        ~sky[:, None], data.shape
    ).copy()
    reasons["invalid_input_or_domain"] = ~valid_input
    reasons["unsupported_background"] = ~supported
    reasons["cross_change"] = (
        (cross_score > config.cross_cut) & valid_input & sky[:, None]
    )
    before_guard = np.logical_or.reduce(list(reasons.values()))
    expanded = maximum_filter(
        before_guard,
        size=(2 * config.time_guard + 1, 2 * config.frequency_guard + 1),
        mode="constant",
    )
    reasons["guard"] = expanded & ~before_guard
    flags = encode_reasons(reasons)
    return {
        "flags": flags,
        "model": np.where(supported, model, np.nan),
        "model_raw": model,
        "residual_z": np.where(supported, z, np.nan),
        "support_ok": supported,
        "fit_keep": keep,
        "reasons": reasons,
        "inflation": inflation,
        "prior_fraction": prior,
        "cross_background": cross_model,
        "cross_score": cross_score,
        "diagnostics": {
            "seconds": time.perf_counter() - started,
            "solver": solver.history,
            "robust_trace": robust_trace,
            "refit_trace": refit_trace,
            "cross": cross_stats,
            "support_seconds": support_seconds,
            **event_stats,
        },
    }


def encode_reasons(reasons):
    """Pack named boolean masks into the stable uint16 v3-beta bitfield."""
    unknown = set(reasons) - set(BIT_BY_REASON)
    if unknown:
        raise KeyError(f"unknown flag reasons: {sorted(unknown)}")
    shape = next(iter(reasons.values())).shape
    flags = np.zeros(shape, dtype=np.uint16)
    for name, mask in reasons.items():
        flags[np.asarray(mask, dtype=bool)] |= np.uint16(
            1 << BIT_BY_REASON[name]
        )
    return flags


def _segments(times, integration_times, gap_factor):
    if len(times) == 0:
        return []
    breaks = np.zeros(len(times), dtype=bool)
    breaks[0] = True
    if len(times) > 1:
        expected = integration_times[:-1]
        breaks[1:] = (
            ~np.isfinite(np.diff(times))
            | (np.diff(times) > gap_factor * expected)
            | ~np.isclose(
                integration_times[1:], integration_times[:-1], rtol=1e-5
            )
        )
    starts = np.flatnonzero(breaks)
    stops = np.r_[starts[1:], len(times)]
    return [slice(int(a), int(b)) for a, b in zip(starts, stops)]


def flag_arrays(
    data,
    ground,
    cross,
    times,
    freqs_mhz,
    integration_times,
    switch_states,
    *,
    config=None,
    meta=None,
):
    """Run v3-beta on aligned air auto, ground auto, and cross arrays.

    Gaps and integration-time changes are fitted independently.  Every segment
    must contain at least one sky row; non-sky rows remain in the output and
    are flagged a priori.
    """
    config = RFIConfig() if config is None else config
    data = np.asarray(data, dtype=float)
    ground = np.asarray(ground, dtype=float)
    cross = np.asarray(cross)
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs_mhz, dtype=float)
    dt = np.asarray(integration_times, dtype=float)
    states = np.asarray(switch_states, dtype=object)
    if not (data.shape == ground.shape == cross.shape):
        raise ValueError("data, ground, and cross must have identical shapes")
    if data.shape != (len(times), len(freqs)) or len(dt) != len(times):
        raise ValueError("array axes do not match times/frequencies")
    if len(freqs) < 3 or not np.allclose(np.diff(freqs), np.diff(freqs)[0]):
        raise ValueError("frequencies must be a uniformly spaced axis")
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("integration times must be finite and positive")
    sky = states == config.sky_state
    pieces = []
    diagnostics = []
    for segment in _segments(times, dt, config.gap_factor):
        if not sky[segment].any():
            ntime = segment.stop - segment.start
            shape = (ntime, len(freqs))
            reasons = {
                name: np.zeros(shape, dtype=bool) for name in BIT_BY_REASON
            }
            reasons["non_sky_switch_state"][:] = True
            reasons["unsupported_background"][:] = True
            pieces.append(
                {
                    "flags": encode_reasons(reasons),
                    "model": np.full(shape, np.nan),
                    "model_raw": np.full(shape, np.nan),
                    "residual_z": np.full(shape, np.nan),
                    "support_ok": np.zeros(shape, dtype=bool),
                    "fit_keep": np.zeros(shape, dtype=bool),
                    "reasons": reasons,
                    "inflation": np.full(shape, np.inf),
                    "prior_fraction": np.ones(shape),
                    "cross_background": np.full(shape, np.nan, dtype=complex),
                    "cross_score": np.full(shape, np.nan),
                    "diagnostics": {"skipped": "no sky-state rows"},
                }
            )
        else:
            pieces.append(
                _one_segment(
                    data[segment],
                    ground[segment],
                    cross[segment],
                    times[segment],
                    freqs,
                    dt[segment],
                    sky[segment],
                    config,
                )
            )
        diagnostics.append(pieces[-1]["diagnostics"])
    concatenate = (
        "flags",
        "model",
        "model_raw",
        "residual_z",
        "support_ok",
        "fit_keep",
        "inflation",
        "prior_fraction",
        "cross_background",
        "cross_score",
    )
    joined = {
        name: np.concatenate([p[name] for p in pieces]) for name in concatenate
    }
    reasons = {
        name: np.concatenate([p["reasons"][name] for p in pieces])
        for name in BIT_BY_REASON
    }
    return RFIResult(
        **joined,
        reasons=reasons,
        freqs_mhz=freqs,
        times=times,
        meta=meta,
        config=config,
        diagnostics={"segments": diagnostics},
    )


def load_selection_inputs(
    selection,
    *,
    air_antenna="box-air",
    ground_antenna="box-gnd",
    config=None,
):
    """Load physical antennas, resolving their input keys in every file."""
    config = RFIConfig() if config is None else config
    bundles = {
        name: selection.load_bundle(
            antenna=antenna, band_mhz=config.band_mhz, missing="raise"
        )
        for name, antenna in {
            "air": air_antenna,
            "ground": ground_antenna,
            "cross": (ground_antenna, air_antenna),
        }.items()
    }
    air = bundles["air"]
    for name, bundle in bundles.items():
        if not np.array_equal(bundle.t, air.t):
            raise ValueError(f"{name} rows do not align with the air auto")
        if not np.array_equal(bundle.freqs_mhz, air.freqs_mhz):
            raise ValueError(
                f"{name} frequencies do not align with the air auto"
            )
        if not bundle.meta[["file", "row"]].equals(air.meta[["file", "row"]]):
            raise ValueError(
                f"{name} file/row keys do not align with the air auto"
            )
    return bundles


def run_selection(
    selection,
    *,
    air_antenna="box-air",
    ground_antenna="box-gnd",
    config=None,
):
    """Resolve physical antennas per file and run supported-DPSS flagging."""
    config = RFIConfig() if config is None else config
    bundles = load_selection_inputs(
        selection,
        air_antenna=air_antenna,
        ground_antenna=ground_antenna,
        config=config,
    )
    air = bundles["air"]
    if "rfswitch" not in air.meta or "integration_time" not in air.meta:
        raise ValueError(
            "selection metadata lacks rfswitch or integration_time"
        )
    result = flag_arrays(
        air.data,
        bundles["ground"].data,
        bundles["cross"].data,
        air.t,
        air.freqs_mhz,
        air.meta.integration_time.to_numpy(dtype=float),
        air.meta.rfswitch.to_numpy(dtype=object),
        config=config,
        meta=air.meta.copy(),
    )
    result.diagnostics["antennas"] = {
        "air": str(air_antenna),
        "ground": str(ground_antenna),
    }
    resolved = {}
    for fname in air.meta.file.unique():
        resolved[fname] = {}
        for name, bundle in bundles.items():
            rows = bundle.meta.file == fname
            keys = bundle.meta.loc[rows, "input_key"].unique().tolist()
            resolved[fname][name] = str(keys[0])
            if name == "cross":
                orientations = (
                    bundle.meta.loc[rows, "conjugated"].astype(bool).unique()
                )
                resolved[fname]["cross_conjugated"] = bool(orientations[0])
    result.diagnostics["resolved_inputs"] = resolved
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parameter_sha256(config):
    """Stable hash used to identify a complete :class:`RFIConfig`."""
    value = asdict(config)
    encoded = json.dumps(_jsonable(value), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def algorithm_source_sha256():
    """Hash the exact flagger source used to generate a product."""
    return _sha256(Path(__file__).resolve())


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_h5_update(path, update):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        if path.exists():
            shutil.copy2(path, temporary)
        with h5py.File(temporary, "a") as h5:
            update(h5)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _input_key(group):
    if "input_key" not in group:
        raise ValueError(
            "writing requires per-row input_key metadata from "
            "load_bundle(antenna=...)"
        )
    keys = group.input_key.astype(str).unique()
    if len(keys) != 1:
        raise ValueError(f"one raw file resolved to multiple air keys: {keys}")
    return keys[0]


def _whole_files(result, data_dir):
    if result.meta is None:
        raise ValueError("writing requires Selection metadata")
    for fname, group in result.meta.groupby("file", sort=False):
        input_key = _input_key(group)
        rows = group.row.to_numpy(dtype=int)
        source = Path(data_dir) / fname
        with h5py.File(source, "r") as h5:
            expected = np.arange(h5["data"][str(input_key)].shape[0])
        if not np.array_equal(rows, expected):
            raise ValueError(
                f"{fname}: product writing requires complete files; selected "
                f"{len(rows)} of {len(expected)} rows"
            )


def _preflight_writes(result, root, flags_version, model_version, overwrite):
    """Refuse incompatible/existing payloads before changing any file."""
    for fname, rows in result.meta.groupby("file", sort=False):
        input_key = _input_key(rows)
        day_path = root / "flags" / flags_version / f"flags_{fname[5:13]}.h5"
        if day_path.exists():
            with h5py.File(day_path, "r") as h5:
                if "freqs_mhz" in h5 and not np.array_equal(
                    h5["freqs_mhz"][:], result.freqs_mhz
                ):
                    raise ValueError(f"{day_path}: frequency axis differs")
                dataset = f"mask/{fname}/{input_key}"
                if dataset in h5 and not overwrite:
                    raise FileExistsError(f"{day_path}:{dataset}")
        model_path = root / "derived" / "smooth_model" / model_version / fname
        if model_path.exists():
            with h5py.File(model_path, "r") as h5:
                if "freqs_mhz" in h5 and not np.array_equal(
                    h5["freqs_mhz"][:], result.freqs_mhz
                ):
                    raise ValueError(f"{model_path}: frequency axis differs")
                group = f"input_{input_key}"
                if group in h5 and not overwrite:
                    raise FileExistsError(f"{model_path}:{group}")


def _write_products_unlocked(
    result,
    campaign_root,
    *,
    data_dir=None,
    flags_version=DEFAULT_VERSION,
    model_version=DEFAULT_VERSION,
    overwrite=False,
):
    """Atomically write flags and smooth-model companions for whole files.

    Existing datasets are refused unless ``overwrite=True``.  Each payload
    carries the parameter hash and source hash; the version manifests map
    those hashes back to the full parameter set and source file.
    """
    root = Path(campaign_root).resolve()
    source_dir = (
        root / "data" if data_dir is None else Path(data_dir).resolve()
    )
    _whole_files(result, source_dir)
    _preflight_writes(result, root, flags_version, model_version, overwrite)
    config = asdict(result.config)
    parameter_hash = parameter_sha256(result.config)
    algorithm_hash = algorithm_source_sha256()
    generated = datetime.now(timezone.utc).isoformat()
    files_record = {}
    flags_root = root / "flags" / flags_version
    model_root = root / "derived" / "smooth_model" / model_version
    for fname, group in result.meta.groupby("file", sort=False):
        input_key = _input_key(group)
        # Bundle metadata normally has a RangeIndex. Resolve by file/row to
        # keep the writer correct if a caller preserved another index.
        positions = np.flatnonzero(result.meta.file.to_numpy() == fname)
        source = source_dir / fname
        source_hash = _sha256(source)
        files_record[fname] = {
            "source": (
                f"data/{fname}" if source_dir == root / "data" else str(source)
            ),
            "source_sha256": source_hash,
            "parameter_sha256": parameter_hash,
            "algorithm_source_sha256": algorithm_hash,
            "input_key": str(input_key),
            "generated_utc": generated,
        }
        day_path = flags_root / f"flags_{fname[5:13]}.h5"

        def update_flags(h5, fname=fname, positions=positions):
            if "freqs_mhz" in h5:
                if not np.array_equal(h5["freqs_mhz"][:], result.freqs_mhz):
                    raise ValueError(f"{day_path}: frequency axis differs")
            else:
                h5.create_dataset("freqs_mhz", data=result.freqs_mhz)
            group_path = f"mask/{fname}"
            out = h5.require_group(group_path)
            key = str(input_key)
            if key in out:
                if not overwrite:
                    raise FileExistsError(f"{day_path}:{group_path}/{key}")
                del out[key]
            dataset = out.create_dataset(
                key, data=result.flags[positions], compression="gzip"
            )
            dataset.attrs["parameter_sha256"] = parameter_hash
            dataset.attrs["algorithm_source_sha256"] = algorithm_hash
            dataset.attrs["source_sha256"] = source_hash

        _atomic_h5_update(day_path, update_flags)
        model_path = model_root / fname

        def update_model(h5, positions=positions):
            if "freqs_mhz" in h5:
                if not np.array_equal(h5["freqs_mhz"][:], result.freqs_mhz):
                    raise ValueError(f"{model_path}: frequency axis differs")
            else:
                h5.create_dataset("freqs_mhz", data=result.freqs_mhz)
            name = f"input_{input_key}"
            if name in h5:
                if not overwrite:
                    raise FileExistsError(f"{model_path}:{name}")
                del h5[name]
            out = h5.create_group(name)
            out.create_dataset(
                "model",
                data=result.model[positions].astype(np.float32),
                compression="gzip",
            )
            out.create_dataset(
                "support_ok",
                data=result.support_ok[positions],
                compression="gzip",
            )
            out.create_dataset(
                "fit_keep",
                data=result.fit_keep[positions],
                compression="gzip",
            )
            out.attrs["parameter_sha256"] = parameter_hash
            out.attrs["algorithm_source_sha256"] = algorithm_hash
            out.attrs["source_sha256"] = source_hash

        _atomic_h5_update(model_path, update_model)
    _atomic_json(
        flags_root / "flag_bits.json",
        {
            "encoding": "uint16 bitfield per (time, channel) sample",
            "axes": ["time", "channel"],
            "n_channels": int(len(result.freqs_mhz)),
            "channel_width_mhz": float(np.median(np.diff(result.freqs_mhz))),
            "clean_value": 0,
            "product": "flags",
            "version": flags_version,
            "status": "beta",
            "dtype": "uint16",
            "bits": [
                {
                    "bit": int(bit),
                    "value": 1 << int(bit),
                    "name": spec["name"],
                    "rfi": spec["rfi"],
                    "meaning": FLAG_MEANINGS[spec["name"]],
                }
                for bit, spec in FLAG_BITS.items()
            ],
        },
    )
    for product_root, kind, version in (
        (flags_root, "flags", flags_version),
        (model_root, "smooth_model", model_version),
    ):
        manifest_path = product_root / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as stream:
                manifest = json.load(stream)
        manifest.update(
            {
                "product": kind,
                "version": version,
                "status": "beta",
                "algorithm": "supported separable DPSS v3-beta",
                "algorithm_source_sha256": algorithm_hash,
                "axes": {
                    "rows": "raw file integration row",
                    "frequency": "MHz",
                },
            }
        )
        manifest.setdefault("parameter_sets", {})[parameter_hash] = config
        manifest.setdefault("files", {}).update(files_record)
        manifest["updated_utc"] = generated
        _atomic_json(manifest_path, manifest)
    return {
        "flags": flags_root,
        "smooth_model": model_root,
        "files": sorted(files_record),
        "parameter_sha256": parameter_hash,
        "algorithm_source_sha256": algorithm_hash,
    }


@contextmanager
def _write_lock(path):
    """Serialize product updates across local campaign-runner processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_products(
    result,
    campaign_root,
    *,
    data_dir=None,
    flags_version=DEFAULT_VERSION,
    model_version=DEFAULT_VERSION,
    overwrite=False,
):
    """Atomically write flags and smooth models for complete raw files.

    ``data_dir`` defaults to ``campaign_root / "data"``. It may point at
    separately mounted raw data while products are written below
    ``campaign_root``. A POSIX advisory lock covers preflight, payload, and
    manifest updates so campaign workers cannot lose one another's updates.
    """
    root = Path(campaign_root).resolve()
    lock = root / "flags" / flags_version / ".write.lock"
    with _write_lock(lock):
        return _write_products_unlocked(
            result,
            root,
            data_dir=data_dir,
            flags_version=flags_version,
            model_version=model_version,
            overwrite=overwrite,
        )
