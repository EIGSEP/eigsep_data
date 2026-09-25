"""Pointing fusion for the rotating beam-mapping receiver.

:func:`fuse_pointing` turns motor counts, potentiometer azimuth, and IMU
acceleration into a common pointing table. The module is NumPy/SciPy based so
it can be used from notebooks without requiring JAX.

Rotations between the antenna body frame and ENU live in
:mod:`~eigsep_data.beam_mapping.beam_rotations`, and the transmitter geometry
in :mod:`~eigsep_data.beam_mapping.tx_coupling` (both moved there 2026-09-25,
when the package adopted one pointing convention).

Public API
----------
MOTOR_DEG_PER_STEP
imu_elevation_deg
PointingStreams
simulate_pointing_streams
estimate_motor_slip_steps
PointingTable
fuse_pointing
"""

from dataclasses import dataclass

import numpy as np


MOTOR_DEG_PER_STEP = 180.0 / 1.13e4


def imu_elevation_deg(accel_xyz):
    """Convert the v007 notebook's IMU acceleration convention to elevation."""
    a = np.asarray(accel_xyz, dtype=float)
    return np.rad2deg(np.unwrap(
        np.arctan2(np.sign(a[..., 1]) * np.linalg.norm(a[..., :2], axis=-1), -a[..., 2]),
        period=2 * np.pi,
    ))


def _circular_mean_deg(values, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones(values.shape[0])
    weights = np.asarray(weights, dtype=float).reshape((-1,) + (1,) * (values.ndim - 1))
    good = np.isfinite(values)
    z = np.sum(weights * np.where(good, np.exp(1j * np.deg2rad(values)), 0), axis=0)
    norm = np.sum(weights * good, axis=0)
    return np.rad2deg(np.angle(z / np.maximum(norm, 1e-30)))


@dataclass
class PointingStreams:
    """Synthetic or measured pointing streams before fusion."""
    true_az_deg: np.ndarray
    true_el_deg: np.ndarray
    motor_az_steps: np.ndarray
    motor_el_steps: np.ndarray
    pot_az_deg: np.ndarray
    imu_accel: np.ndarray
    motor_slip_az_steps: np.ndarray
    motor_slip_el_steps: np.ndarray


def simulate_pointing_streams(true_az_deg, true_el_deg, pot_sigma_deg=0.5,
                              motor_sigma_steps=2.0, slip_az_steps=0.0,
                              slip_el_steps=0.0, imu_sigma=0.01, seed=0):
    """Generate noisy pot/encoder/IMU streams with persistent gear slip."""
    rng = np.random.default_rng(seed)
    az = np.asarray(true_az_deg, float); el = np.asarray(true_el_deg, float)
    slip_az = np.broadcast_to(slip_az_steps, az.shape).astype(float)
    slip_el = np.broadcast_to(slip_el_steps, el.shape).astype(float)
    motor_az = az / MOTOR_DEG_PER_STEP + slip_az + rng.normal(0, motor_sigma_steps, az.shape)
    motor_el = el / MOTOR_DEG_PER_STEP + slip_el + rng.normal(0, motor_sigma_steps, el.shape)
    pot = az + rng.normal(0, pot_sigma_deg, az.shape)
    # Invert the v007 IMU elevation convention: ay=sin(el), -az=cos(el).
    accel = np.stack([np.zeros_like(el), np.sin(np.deg2rad(el)),
                      -np.cos(np.deg2rad(el))], axis=-1)
    accel += rng.normal(0, imu_sigma, accel.shape)
    return PointingStreams(az, el, motor_az, motor_el, pot, accel, slip_az, slip_el)


def estimate_motor_slip_steps(motor_steps, reference_deg):
    """Robustly estimate a constant encoder offset against a reference angle."""
    residual = MOTOR_DEG_PER_STEP * np.asarray(motor_steps) - np.asarray(reference_deg)
    return float(np.nanmedian(residual) / MOTOR_DEG_PER_STEP)


@dataclass
class PointingTable:
    az_deg: np.ndarray
    el_deg: np.ndarray
    az_motor_deg: np.ndarray
    az_pot_deg: np.ndarray
    el_imu_deg: np.ndarray
    el_motor_deg: np.ndarray


def fuse_pointing(motor_az_steps, motor_el_steps, pot_az_deg=None,
                  imu_accel=None, az_weight=0.5, el_weight=0.5):
    """Fuse available motor, potentiometer, and IMU pointing streams.

    Motor counts use the calibration from ``EIGSEP_data_explore_v007``.
    Missing values may be NaN.  Azimuth is combined on the circle; elevation
    is combined linearly.  The returned table retains each input stream for
    diagnostics and fitting.
    """
    azm = MOTOR_DEG_PER_STEP * np.asarray(motor_az_steps, float)
    elm = MOTOR_DEG_PER_STEP * np.asarray(motor_el_steps, float)
    azp = np.full_like(azm, np.nan) if pot_az_deg is None else np.asarray(pot_az_deg, float)
    eli = np.full_like(elm, np.nan) if imu_accel is None else imu_elevation_deg(imu_accel)
    az = azm.copy()
    both = np.isfinite(azm) & np.isfinite(azp)
    az[both] = _circular_mean_deg(np.stack([azm[both], azp[both]]), [az_weight, 1 - az_weight])
    az[~np.isfinite(azm) & np.isfinite(azp)] = azp[~np.isfinite(azm) & np.isfinite(azp)]
    el = elm.copy()
    both = np.isfinite(elm) & np.isfinite(eli)
    el[both] = (el_weight * elm[both] + (1 - el_weight) * eli[both])
    el[~np.isfinite(elm) & np.isfinite(eli)] = eli[~np.isfinite(elm) & np.isfinite(eli)]
    return PointingTable(az, el, azm, azp, eli, elm)
