"""Fitting the beam to transmitter comb teeth.

The data are one background-subtracted time series per tooth
(:class:`ToothData`). The model for tooth ``f`` at sample ``n`` is

    gain_f * |E_f(n-hat_tx) . e_tx,arm(f)|^2,

with the transmitter direction and field taken into the antenna body frame by
the mount rotation (:mod:`~eigsep_data.beam_mapping.beam_rotations`) and the
geometry in :class:`TxGeometryModel`. Three stages, as in beam fits
v0008-v0011:

1. :func:`coarse_offset_alpha`: a grid over the az offset and the transmitter
   polarization centre, with the HFSS beam and zero geometry corrections.
2. :func:`fit_geometry`: least squares over nested sets of the five geometry
   corrections (:data:`PARAM_NAMES`), each scored by held-out fractional RMS.
3. :class:`JointBeamFit`: the empirical beam, i.e. complex coefficients
   ``C (nmodes, 3 nlm)`` of a spectral x spherical-harmonic basis
   (:mod:`~eigsep_data.beam_mapping.beam_basis`), fitted jointly with one
   positive gain per tooth under a prior toward the HFSS coefficients. JAX
   autodiff with L-BFGS-B, in float64.

Samples are split into train/validation/test by stripes of azimuth
(:func:`stripe_split`), so held-out scores test directions the fit did not see.

Public API
----------
PARAM_NAMES
ToothData
stripe_split
TxGeometryModel
tooth_fields
hfss_power
score
coarse_offset_alpha
fit_geometry
JointBeamFit
"""

import dataclasses
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, minimize

from .beam_basis import real_spherical_harmonics
from .beam_rotations import mount_rotation
from .tx_coupling import (
    TransmitterGeometry,
    heading_between,
    interpolate_fields,
    normalize_fields,
    sample_fields,
    tooth_gains,
    transmitter_frame,
)
from .tx_teeth import tooth_arms

log = logging.getLogger(__name__)

PARAM_NAMES = ('polarization_delta_deg', 'az_zero_delta_deg', 'el_zero_delta_deg',
               'tx_east_delta_m', 'tx_north_delta_m')
DEFAULT_BOUNDS = (30.0, 10.0, 10.0, 1.5 / np.sqrt(2), 1.5 / np.sqrt(2))
GEOMETRY_CANDIDATES = (('fixed', ()), ('polarization', (0,)), ('az_el', (1, 2)),
                       ('tx_position', (3, 4)), ('polarization_az_el', (0, 1, 2)),
                       ('all', (0, 1, 2, 3, 4)))


@dataclass
class ToothData:
    """Background-subtracted tooth signals on a common set of samples.

    ``t``, ``az_deg``, ``el_deg``, ``split``: (n,). ``data``, ``good``: (nteeth, n).
    ``channels``, ``freqs_mhz``: (nteeth,). ``split`` is 0 train, 1 validation,
    2 test. ``fit_channels`` marks teeth that constrain the beam (default all).
    """

    t: np.ndarray
    az_deg: np.ndarray
    el_deg: np.ndarray
    data: np.ndarray
    good: np.ndarray
    channels: np.ndarray
    freqs_mhz: np.ndarray
    split: np.ndarray
    fit_channels: np.ndarray = None
    spacing: int = 8

    def __post_init__(self):
        if self.fit_channels is None:
            self.fit_channels = np.ones(len(self.channels), bool)
        self.good = np.asarray(self.good, bool) & np.isfinite(self.data)

    @property
    def arms(self):
        return tooth_arms(self.channels, self.spacing)

    def subset(self, indices):
        """The same teeth on a subset of samples."""
        return dataclasses.replace(self, t=self.t[indices], az_deg=self.az_deg[indices],
                                   el_deg=self.el_deg[indices], data=self.data[:, indices],
                                   good=self.good[:, indices], split=self.split[indices])


def stripe_split(az_deg, width_deg=12.0, period=5, validation=1, test=3):
    """Hold out whole stripes of azimuth: stripe ``k = floor(((az + 180) % 360) / width)``
    is validation when ``k % period == validation``, test when ``== test``, else train."""
    k = np.floor(((np.asarray(az_deg, float) + 180) % 360) / width_deg).astype(int)
    return np.where(k % period == validation, 1, np.where(k % period == test, 2, 0))


@dataclass
class TxGeometryModel:
    """Site geometry and the reference transmitter polarization.

    ``psi_deg``: elevation-axle direction (deg ccw from East). ``antenna_enu``,
    ``transmitter_enu``: positions (m). ``alpha0_deg``: arm-0 polarization centre
    (see :class:`~eigsep_data.beam_mapping.tx_coupling.TransmitterGeometry`).
    ``az_offset_deg``: added to the table az (the az-zero calibration). The five
    corrections in :data:`PARAM_NAMES` are applied on top by every method.
    """

    psi_deg: float
    antenna_enu: np.ndarray
    transmitter_enu: np.ndarray
    alpha0_deg: float = 0.0
    az_offset_deg: float = 0.0

    def transmitter(self, params):
        p = np.asarray(params, float)
        tx = np.asarray(self.transmitter_enu, float) + np.array([p[3], p[4], 0.0])
        return TransmitterGeometry(heading_between(self.antenna_enu, tx), self.alpha0_deg + p[0])

    def rotations(self, az_deg, el_deg, params):
        p = np.asarray(params, float)
        return mount_rotation(np.asarray(az_deg) + self.az_offset_deg + p[1], np.asarray(el_deg) + p[2],
                              self.psi_deg)

    def frame(self, data, params, indices=None):
        """Transmitter (theta, phi) and both arms' fields ``e`` (2, n, 3) in the body frame."""
        idx = slice(None) if indices is None else indices
        R = self.rotations(data.az_deg[idx], data.el_deg[idx], params)
        theta, phi, _, e = transmitter_frame(R, self.transmitter(params), None)
        return theta, phi, e


def tooth_fields(fields, freqs_mhz, tooth_freqs_mhz, reference_mhz=190.0):
    """HFSS fields normalized (unit mean power, common phase removed) and interpolated to the teeth."""
    normalized, _, _ = normalize_fields(fields, freqs_mhz, reference_mhz)
    return interpolate_fields(normalized, freqs_mhz, tooth_freqs_mhz)


def hfss_power(fields_teeth, data, model, params, indices=None, interpolation='bilinear'):
    """``|E_f . e_arm(f)|^2`` for every tooth ``f`` and sample: (nteeth, n)."""
    theta, phi, e = model.frame(data, params, indices)
    beam = sample_fields(fields_teeth, theta, phi, interpolation)            # (nteeth, n, 3)
    return np.abs(np.einsum('fnc,fnc->fn', beam, e[data.arms])) ** 2


def score(data, prediction, label):
    """Per tooth and split (train, validation, test, all): fractional RMS
    ``||d - m|| / ||d||``, R^2 and the mean residual, on good samples."""
    records = []
    for f, freq in enumerate(data.freqs_mhz):
        for s, name in ((0, 'train'), (1, 'validation'), (2, 'test'), (-1, 'all')):
            use = data.good[f] & ((data.split == s) if s >= 0 else True)
            y, pred = data.data[f, use], prediction[f, use]
            err = y - pred
            records.append(dict(model=label, freq_mhz=float(freq), channel=int(data.channels[f]),
                                arm=int(data.arms[f]), split=name, n=int(use.sum()),
                                used_for_beam_fit=bool(data.fit_channels[f]),
                                fractional_rms=float(np.sqrt(np.sum(err ** 2) / np.sum(y ** 2))),
                                r_squared=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
                                mean_residual=float(err.mean())))
    return records


def _median_score(records, split):
    return float(np.median([r['fractional_rms'] for r in records
                            if r['split'] == split and r['used_for_beam_fit']]))


def _train_rms(data, fields_teeth, model, params, idx, scale, usef):
    p = hfss_power(fields_teeth, data, model, params, idx)
    g = tooth_gains(p, data.data[:, idx], data.good[:, idx])
    good = data.good[:, idx]
    r = np.where(good, (g[:, None] * p - np.where(good, data.data[:, idx], 0.0)) / scale[:, None], 0.0)
    return r[usef]


def coarse_offset_alpha(data, fields_teeth, model, offsets=np.arange(-60.0, 60.01, 3.0),
                        alphas=np.arange(0.0, 180.0, 6.0), refine=1.0, stride=4):
    """Grid search of the az offset and the polarization centre (HFSS beam, zero
    corrections) on every ``stride``-th training sample, then a ``refine``-degree grid
    around the best point. Returns ``(model, table)``: ``model`` with the best
    ``az_offset_deg`` and ``alpha0_deg``, and every point scored."""
    idx = np.flatnonzero(data.split == 0)[::stride]
    usef = np.flatnonzero(data.fit_channels)
    scale = np.sqrt(np.nanmean(np.where(data.good[:, idx], data.data[:, idx], np.nan) ** 2, axis=1))
    rows = []

    def run(offset_values, alpha_values):
        for off in offset_values:
            for a in alpha_values:
                m = dataclasses.replace(model, az_offset_deg=float(off), alpha0_deg=float(a) % 180)
                r = _train_rms(data, fields_teeth, m, np.zeros(5), idx, scale, usef)
                rows.append(dict(az_offset_deg=float(off), alpha_deg=float(a) % 180,
                                 normalized_rms=float(np.sqrt(np.mean(r ** 2)))))

    run(offsets, alphas)
    if refine:
        best = min(rows, key=lambda x: x['normalized_rms'])
        so, sa = np.diff(offsets).min(), np.diff(alphas).min()
        run(np.arange(best['az_offset_deg'] - so, best['az_offset_deg'] + so + 1e-9, refine),
            np.arange(best['alpha_deg'] - sa, best['alpha_deg'] + sa + 1e-9, refine))
    best = min(rows, key=lambda x: x['normalized_rms'])
    return dataclasses.replace(model, az_offset_deg=best['az_offset_deg'], alpha0_deg=best['alpha_deg']), rows


def fit_geometry(data, fields_teeth, model, candidates=GEOMETRY_CANDIDATES, bounds=DEFAULT_BOUNDS,
                 stride=8):
    """Least squares over nested sets of geometry corrections.

    Each candidate frees the listed parameters (within ``bounds``), fitting on
    every ``stride``-th training sample with one gain per tooth. It is then
    scored on all samples with gains fitted on the training stripes, and the
    candidate with the lowest median validation fractional RMS is chosen.
    Returns ``(params, name, table, predictions)``: the chosen corrections
    (5,), its name, one summary dict per candidate, and each candidate's
    (nteeth, n) prediction.
    """
    train = data.split == 0
    idx = np.flatnonzero(train)[::stride]
    usef = np.flatnonzero(data.fit_channels)
    scale = np.sqrt(np.nanmean(np.where(data.good[:, idx], data.data[:, idx], np.nan) ** 2, axis=1))
    bound = np.asarray(bounds, float)
    table, predictions, parameters = [], {}, {}
    for name, free in candidates:
        free = list(free)
        x = np.zeros(5)
        condition = 1.0
        if free:
            def unpack(v, free=free):
                p = np.zeros(5)
                p[free] = v
                return p

            def residual(v, unpack=unpack):
                r = _train_rms(data, fields_teeth, model, unpack(v), idx, scale, usef)
                return r.ravel() / np.sqrt(len(idx) * len(usef))

            opt = least_squares(residual, x[free], bounds=(-bound[free], bound[free]), diff_step=1e-3,
                                max_nfev=100, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            x = unpack(opt.x)
            sv = np.linalg.svd(opt.jac, compute_uv=False)
            condition = float(sv[0] / max(sv[-1], 1e-30))
        p = hfss_power(fields_teeth, data, model, x)
        prediction = tooth_gains(p, data.data, data.good, train)[:, None] * p
        records = score(data, prediction, name)
        row = dict(model=name, **dict(zip(PARAM_NAMES, x)), jacobian_condition=condition,
                   **{split + '_median_rms': _median_score(records, split)
                      for split in ('train', 'validation', 'test')})
        table.append(row)
        predictions[name], parameters[name] = prediction, x
        log.info('geometry %s', row)
    best = min(table, key=lambda r: r['validation_median_rms'])['model']
    return parameters[best], best, table, predictions


@dataclass
class JointBeamFit:
    """Empirical beam fitted jointly with the tooth gains.

    ``basis`` is a spectral basis with ``evaluate(freqs)`` (``beam_basis``); ``c0``
    (nmodes, 3 nlm) the prior (usually the HFSS beam in this basis, from
    :func:`~eigsep_data.beam_mapping.beam_basis.initial_coefficients`).
    """

    data: ToothData
    model: TxGeometryModel
    params: np.ndarray
    lmax: int
    basis: object
    c0: np.ndarray
    design: np.ndarray = field(init=False, repr=False)
    spectral: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        theta, phi, e = self.model.frame(self.data, self.params)
        y = real_spherical_harmonics(theta, phi, self.lmax)                   # (n, nlm)
        self.design = np.array([np.concatenate([y * arm[:, c, None] for c in range(3)], axis=1)
                                for arm in e])                                 # (2, n, 3 nlm)
        self.spectral = self.basis.evaluate(self.data.freqs_mhz)               # (nteeth, nmodes)

    def power(self, coeff):
        """``|E . e|^2`` per tooth and sample for coefficients ``coeff`` (unit gain)."""
        q = np.empty(self.data.data.shape, dtype=complex)
        arms = self.data.arms
        for arm in (0, 1):
            use = arms == arm
            q[use] = self.spectral[use] @ coeff @ self.design[arm].T
        return np.abs(q) ** 2

    def fit(self, train, penalty, initial=None, initial_gain=None, maxiter=3000,
            check_gradients=True):
        """Fit on the samples in ``train`` with prior weight ``penalty``.

        The loss is the mean over fitting teeth of the weighted squared residual
        (each tooth scaled to unit rms), plus ``penalty * ||C - c0||^2 / ||c0||^2``.
        Gains are positive (fitted in log). Returns
        ``(coeff, gain, prediction, info)``: the coefficients, a gain for every
        tooth (conditional least squares for teeth outside ``fit_channels``), the
        (nteeth, n) prediction on all samples, and optimizer diagnostics.
        """
        import jax
        import jax.numpy as jnp

        d = self.data
        rows = np.flatnonzero(train)
        usef = np.flatnonzero(d.fit_channels)
        valid = d.good[usef][:, rows]
        obs = d.data[usef][:, rows]
        scale = np.sqrt(np.sum(np.where(valid, obs ** 2, 0), axis=1) / valid.sum(axis=1))
        shape, n = self.c0.shape, self.c0.size
        arms = d.arms[usef]
        groups = [np.flatnonzero(arms == a) for a in (0, 1)]
        start = self.c0 if initial is None else initial
        if initial_gain is None:
            initial_gain = tooth_gains(self.power(start), d.data, d.good, train)
        eta = np.log(np.maximum(initial_gain[usef] / scale, 1e-12))
        x0 = np.r_[start.real.ravel(), start.imag.ravel(), eta]
        started = time.monotonic()
        with jax.enable_x64(True):                                              # float64, scoped
            target = jnp.asarray(np.where(valid, obs / scale[:, None], 0))
            weight = jnp.asarray(valid / valid.sum(axis=1)[:, None] / len(usef))
            spectral = jnp.asarray(self.spectral[usef])
            design = jnp.asarray(self.design[:, rows])
            c0 = jnp.asarray(self.c0)
            norm = jnp.sum(jnp.abs(c0) ** 2)

            def objective(x):
                coeff = (x[:n] + 1j * x[n:2 * n]).reshape(shape)
                power = jnp.zeros_like(target)
                for arm, idx in enumerate(groups):
                    power = power.at[idx].set(jnp.abs(spectral[idx] @ coeff @ design[arm].T) ** 2)
                residual = jnp.exp(x[2 * n:])[:, None] * power - target
                return jnp.sum(weight * residual ** 2) + penalty * jnp.sum(jnp.abs(coeff - c0) ** 2) / norm

            value_gradient = jax.jit(jax.value_and_grad(objective))

            def scipy_objective(x):
                value, gradient = value_gradient(jnp.asarray(x))
                return float(value), np.asarray(gradient, float)

            checks = []
            if check_gradients:
                rng = np.random.default_rng(20260923)
                _, gradient = scipy_objective(x0)
                for block in ('coefficients', 'log_gains'):
                    direction = rng.normal(size=len(x0))
                    if block == 'coefficients':
                        direction[2 * n:] = 0
                    else:
                        direction[:2 * n] = 0
                    direction /= np.linalg.norm(direction)
                    eps = 1e-5
                    fd = (scipy_objective(x0 + eps * direction)[0]
                          - scipy_objective(x0 - eps * direction)[0]) / (2 * eps)
                    analytic = float(gradient @ direction)
                    if not np.isclose(fd, analytic, rtol=1e-3, atol=1e-7):
                        raise RuntimeError(f'gradient check failed for {block}: {fd} vs {analytic}')
                    checks.append(dict(block=block, finite_difference=fd, autodiff=analytic))
            opt = minimize(scipy_objective, x0, jac=True, method='L-BFGS-B',
                           bounds=[(None, None)] * (2 * n) + [(-25.0, 25.0)] * len(usef),
                           options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-6, maxcor=30, maxls=40))
        coeff = (opt.x[:n] + 1j * opt.x[n:2 * n]).reshape(shape)
        power = self.power(coeff)
        gain = tooth_gains(power, d.data, d.good, train)                      # conditional, all teeth
        gain[usef] = np.exp(opt.x[2 * n:]) * scale
        info = dict(penalty=penalty, iterations=int(opt.nit), success=bool(opt.success),
                    message=str(opt.message), loss=float(opt.fun),
                    gradient_max=float(np.max(np.abs(opt.jac))), seconds=time.monotonic() - started,
                    gain_bound_hit=bool(np.any(np.abs(opt.x[2 * n:]) >= 24.99)),
                    coefficient_real_parameters=2 * n, joint_channel_gains=len(usef),
                    gradient_checks=checks)
        log.info('joint fit %s', {k: v for k, v in info.items() if k != 'gradient_checks'})
        return coeff, gain, gain[:, None] * power, info
