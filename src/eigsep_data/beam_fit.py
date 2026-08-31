"""
Transmitter position and polarization fitting for the EIGSEP beam-mapping
experiment.

The main entry point is fit_multi_freq_joint, which jointly fits the
transmitter heading (theta, phi) and polarization angle (alpha) across
multiple frequencies using a global differential-evolution search followed
by Nelder-Mead refinement, and always returns a 2D Cartesian loss map at
the best-fit alpha.

Public API
----------
fit_multi_freq_joint
rms_err_np
rms_err_per_freq_np
"""

import numpy as np
import jax.numpy as jnp
from scipy.optimize import differential_evolution, minimize

from .hpm import angles_to_coord
from .beam_sim import RotatingAntennaCartesian, TransmitterAntenna, power_sim


def fit_multi_freq_joint(
    P_meas,
    bowtie_beams_use,
    az_deg,
    el_deg,
    freq_map,
    mask=None,
    theta_bounds=(90.0, 180.0),
    phi_bounds=(0.0, 360.0),
    alpha_bounds=(0.0, 180.0),
    fit_offset=True,
    seed=0,
    xy_bounds=(-1.0, 1.0),
    grid_res=0.04,
):
    """
    Jointly fit transmitter heading and polarization angle across multiple
    frequencies, then generate a 2D Cartesian loss map at the best-fit alpha.

    The transmitter has two orthogonal dipole arms that emit on interleaved
    frequency combs. Arm assignment is determined by the data-frequency index
    in freq_map: even d_idx uses alpha, odd d_idx uses alpha + 90°.

    Parameters
    ----------
    P_meas : np.ndarray, shape (n_samples, n_data_freqs)
        Measured reduced power (DPSS-reduced, interleaved).
    bowtie_beams_use : array-like
        HFSS beam maps indexed by the h_idx entries in freq_map.
    az_deg, el_deg : array-like, shape (n_samples,)
        Receiver rotation angles in degrees.
    freq_map : list of (d_idx, h_idx)
        Mapping from data frequency index to HFSS beam index.
        d_idx selects a column of P_meas; h_idx selects a beam map.
    mask : array-like of bool, shape (n_samples,), optional
        True = include sample in fit. Defaults to all True.
    theta_bounds : (float, float)
        Search bounds for polar angle in degrees. Defaults to (90, 180),
        i.e. the bottom hemisphere (z <= 0), matching the loss map below
        which always evaluates z = -sqrt(1 - x^2 - y^2).
    phi_bounds : (float, float)
        Search bounds for azimuth in degrees.
    alpha_bounds : (float, float)
        Search bounds for polarization angle in degrees.
    fit_offset : bool
        If True, remove the mean residual before computing MSE (removes
        unknown absolute gain from the loss).
    seed : int
        Random seed for differential_evolution.
    xy_bounds : (float, float)
        x/y extent of the Cartesian loss map (unit disk).
    grid_res : float
        Grid spacing of the Cartesian loss map.

    Returns
    -------
    local_result : scipy.optimize.OptimizeResult
        Result of the Nelder-Mead refinement.
    best_coord : np.ndarray, shape (3,)
        Best-fit Cartesian unit vector.
    best_x, best_y : float
        x and y components of best_coord.
    best_alpha : float
        Best-fit polarization angle in degrees.
    loss_2d : np.ndarray, shape (ny, nx)
        Loss landscape on a Cartesian grid at best_alpha, indexed
        [row=y, col=x] per imshow/pcolormesh convention. Points outside
        the unit disk are set to np.inf.
    extent : list of float
        [x_min, x_max, y_min, y_max] for imshow/pcolormesh.

    Notes
    -----
    This fit assumes the RX gimbal's true mechanical el/az axes exactly
    match RotatingAntennaCartesian's el_axis/az_axis (ideal [1,0,0] and
    [0,0,1] by default) -- there is no parameter here for a mounting
    misalignment between them. A quick numerical check (3 deg synthetic
    axis tilts against an alpha-only refit, on the committed bowtie beam)
    found this is *not* an exact degeneracy: 2-3% relative RMS power
    residual remained unexplained by any alpha, and best_alpha stayed
    within about 1 deg of truth rather than absorbing the tilt. So a
    joint fit of alpha and an axis tilt should be identifiable in
    principle, given enough scan coverage and SNR -- but that residual is
    small enough that on real (noisier, RFI-flagged) data the two could
    still be poorly conditioned in practice. This was a rough check, not
    a Fisher-information/condition-number analysis; treat it as a
    starting point, not a proof either way.
    """
    az_rad = jnp.deg2rad(jnp.asarray(az_deg))
    el_rad = jnp.deg2rad(jnp.asarray(el_deg))

    # Pre-build RX beam objects (one per freq_map entry)
    rx_beams = [
        RotatingAntennaCartesian(beam_cart=bowtie_beams_use[h_idx], conjugate_beam=True)
        for (_, h_idx) in freq_map
    ]

    if mask is None:
        mask = np.ones(P_meas.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    # ------------------------------------------------------------------
    # Core loss function (shared by optimizer and loss-map generation)
    # ------------------------------------------------------------------
    def loss_from_coord(coord, alpha_deg):
        total_mse = 0.0
        for i, (d_idx, _) in enumerate(freq_map):
            arm_alpha = alpha_deg if (d_idx % 2 == 0) else alpha_deg + 90.0
            tx_f = TransmitterAntenna(E1=0.0, E2=1.0, heading_top=coord,
                                      alpha=arm_alpha)
            sim  = np.asarray(
                power_sim(rx_beams[i], tx_f, az_rad, el_rad)[0]
            ).squeeze()
            data = P_meas[:, d_idx].squeeze()
            valid = mask & np.isfinite(data) & np.isfinite(sim)
            d, s  = data[valid], sim[valid]
            residual = d - s
            if fit_offset:
                residual -= np.mean(residual)
            total_mse += np.mean(residual ** 2)
        return total_mse

    def loss(params):
        theta_deg, phi_deg, alpha_deg = params
        return loss_from_coord(angles_to_coord(theta_deg, phi_deg), alpha_deg)

    # ------------------------------------------------------------------
    # Global search + local refinement
    # ------------------------------------------------------------------
    global_result = differential_evolution(
        loss,
        bounds=[theta_bounds, phi_bounds, alpha_bounds],
        seed=seed,
        tol=1e-8,
        polish=False,
        workers=1,
    )

    local_result = minimize(
        loss,
        global_result.x,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-12},
    )

    best_theta, best_phi, best_alpha = local_result.x

    # Standardize angles to canonical range
    best_theta %= 360.0
    best_phi   %= 360.0
    best_alpha %= 180.0
    if best_theta > 180.0:
        best_theta = 360.0 - best_theta
        best_phi   = (best_phi + 180.0) % 360.0

    best_coord = angles_to_coord(best_theta, best_phi)
    best_x, best_y = float(best_coord[0]), float(best_coord[1])

    local_result.x = np.array([best_theta, best_phi, best_alpha])

    # ------------------------------------------------------------------
    # Cartesian loss map at the best-fit alpha
    # ------------------------------------------------------------------
    lo, hi = xy_bounds
    xs = np.arange(lo, hi + grid_res, grid_res)
    ys = np.arange(lo, hi + grid_res, grid_res)
    loss_2d = np.full((len(ys), len(xs)), np.inf)

    print(f"Generating {len(xs)}x{len(ys)} Cartesian loss map "
          f"at alpha = {best_alpha:.2f}°...")

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            r2 = x ** 2 + y ** 2
            if r2 <= 1.0:
                z = -np.sqrt(1.0 - r2)
                loss_2d[j, i] = loss_from_coord(
                    jnp.array([x, y, z]), best_alpha
                )

    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    return local_result, best_coord, best_x, best_y, best_alpha, loss_2d, extent


def rms_err_np(truth, sim, mask=None, eps=1e-30):
    """
    Root-mean-square error between truth and sim.

    Parameters
    ----------
    truth, sim : array-like
    mask : array-like of bool, optional
        Restricts the comparison to selected elements. Non-finite values
        are always excluded.
    eps : float
        Unused (kept for API compatibility).

    Returns
    -------
    float
        RMS error, or NaN if no valid samples remain.
    """
    truth = np.asarray(truth, dtype=float)
    sim   = np.asarray(sim,   dtype=float)
    valid = np.isfinite(truth) & np.isfinite(sim)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    n_valid = np.sum(valid)
    if n_valid == 0:
        return np.nan
    return np.sqrt(np.sum((truth[valid] - sim[valid]) ** 2) / n_valid)


def rms_err_per_freq_np(truth, sim, mask=None, eps=1e-30):
    """
    Normalized RMS error per frequency (first axis).

    Computes sqrt( sum((truth - sim)^2) / sum(truth^2) ) independently
    for each row of truth/sim.

    Parameters
    ----------
    truth, sim : np.ndarray, shape (n_freq, n_samples)
    mask : array-like of bool, shape (n_freq, n_samples), optional
    eps : float
        Floor applied to the denominator to prevent division by zero.

    Returns
    -------
    np.ndarray, shape (n_freq,)
    """
    truth = np.asarray(truth, dtype=float)
    sim   = np.asarray(sim,   dtype=float)
    valid = np.isfinite(truth) & np.isfinite(sim)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    num = np.sum(np.where(valid, (truth - sim) ** 2, 0.0), axis=1)
    den = np.sum(np.where(valid, truth ** 2,          0.0), axis=1)
    return np.sqrt(num / np.maximum(den, eps))
