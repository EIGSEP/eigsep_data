"""
Per-pixel recovery of the receive antenna's complex vector beam from
transmitter cross-correlations.

Three antennas take part: the rotating sky antenna, whose beam ``u`` is
the unknown; a stationary copy of it on the ground, the correlation
partner; and the stationary cross-dipole transmitter. Both receivers are
single-port. The transmitter alternates two orthogonal arms on an
interleaved comb, so at every sample the sky antenna's one output is
recorded once per arm::

    Ex = E1 . u
    Ey = g * (E2 . u)

E1 and E2 are the two arms' incident fields expressed in the sky
antenna's instantaneous body frame and projected transverse to the
arrival direction r_hat. g is a per-channel complex D-term that absorbs
the ground antenna's fixed response to each arm (it is never modelled
explicitly). The far-field condition ``r_hat . u = 0`` is added as one
more row, so a single visit to a pixel is an exactly determined 3x3
system and several visits are an overdetermined least squares.

The component of u along r_hat is never measured: E1 and E2 are always
transverse, so the prior row alone sets it, however many visits a pixel
has. More visits only improve the transverse plane.

Geometry comes from either of two conventions: :func:`fields_from_model`
takes a :class:`~eigsep_data.beam_mapping.TxGeometryModel` (the
``eigsep_base.rotations`` mount convention the production beam fit uses;
arm 0 at the model's alpha, arm 1 at alpha + 90), and
:func:`transmitter_fields` takes the older :mod:`eigsep_data.beam_sim`
angles (``RotatingAntennaCartesian`` / ``TransmitterAntenna``). Both
return the same four arrays, and everything downstream is convention
free.

Public API
----------
fields_from_model
transmitter_fields
group_by_pixel
solve_pixel
solve_sampled_beam
fit_closure_g
reduce_cross
dwell_clusters
leave_one_out
split_half_by_dwell
compare_to_truth
"""

import numpy as np
import jax
import jax.numpy as jnp
from healjax.coord import angles_to_coord
from scipy.optimize import minimize

from .beam_sim import (
    RotatingAntennaCartesian,
    TransmitterAntenna,
    _px_th_ph_vmapped,
)
from .rfi import (
    fit_dpss_interleaved_fix,
    flag_absurd_cells,
    flag_dpss_interleaved_outliers,
)

jax.config.update("jax_enable_x64", True)

#: Default D-term seed: the single global phase used before g was fit
#: per channel by closure.
G_SEED = np.exp(1j * np.radians(-147.4))


def fields_from_model(model, params, az_deg, el_deg, nside):
    """
    Geometry of every sample from a production beam-fit geometry.

    Parameters
    ----------
    model : eigsep_data.beam_mapping.TxGeometryModel
        Site geometry, az offset and polarization centre.
    params : array-like, shape (5,)
        The fitted geometry corrections (``beam_mapping.PARAM_NAMES``).
    az_deg, el_deg : array-like, shape (nsample,)
        Table pointing, before the model's az offset.
    nside : int
        HEALPix nside (RING) the samples are binned on.

    Returns
    -------
    pxs, r_hat, E1v, E2v : as :func:`transmitter_fields`, with E1v the
        field of transmitter arm 0 and E2v of arm 1, in the body frame
        the HFSS ``beam_cart`` is expressed in.
    """
    import healpy as hp

    from .beam_mapping.tx_coupling import transmitter_frame

    R = model.rotations(az_deg, el_deg, params)
    theta, phi, r_hat, e = transmitter_frame(
        R, model.transmitter(params), None
    )
    pxs = hp.ang2pix(nside, theta, phi)
    return (
        np.asarray(pxs),
        np.asarray(r_hat, dtype=float),
        np.asarray(e[0], dtype=float),
        np.asarray(e[1], dtype=float),
    )


def transmitter_fields(theta_deg, phi_deg, alpha_deg, az_rad, el_rad, nside):
    """
    Geometry of every sample for one transmitter placement.

    Frequency independent: compute once per geometry and reuse for every
    channel.

    Parameters
    ----------
    theta_deg, phi_deg : float
        Transmitter direction (``hpm.angles_to_coord`` convention).
    alpha_deg : float
        Polarization angle of transmitter arm 1; arm 2 is at alpha + 90.
    az_rad, el_rad : array-like, shape (nsample,)
        Sky-antenna rotation angles in radians.
    nside : int
        HEALPix nside the samples are binned on.

    Returns
    -------
    pxs : np.ndarray of int, shape (nsample,)
        Pixel of the arrival direction in the sky antenna's frame.
    r_hat : np.ndarray, shape (nsample, 3)
        Unit arrival direction (continuous, not snapped to the pixel
        centre).
    E1v, E2v : np.ndarray, shape (nsample, 3)
        Arm 1 and arm 2 incident fields in the sky antenna's frame,
        projected transverse to r_hat.
    """
    npix = 12 * nside**2
    rx = RotatingAntennaCartesian(
        beam_cart=jnp.zeros((3, npix), dtype=jnp.complex128)
    )
    coord = angles_to_coord(theta_deg, phi_deg)
    tx1 = TransmitterAntenna(
        E1=0.0, E2=1.0, heading_top=coord, alpha=alpha_deg
    )
    tx2 = TransmitterAntenna(
        E1=0.0, E2=1.0, heading_top=coord, alpha=alpha_deg + 90.0
    )
    az = jnp.asarray(az_rad)
    el = jnp.asarray(el_rad)
    Rs, ths, phis, pxs = _px_th_ph_vmapped(rx, tx1, az, el)
    r_hat = rx.r_hat(ths, phis).astype(jnp.float64)
    E1v = jax.vmap(tx1.E_rotated)(Rs)
    E2v = jax.vmap(tx2.E_rotated)(Rs)
    E1v = E1v - jnp.einsum("ij,ij->i", E1v, r_hat)[:, None] * r_hat
    E2v = E2v - jnp.einsum("ij,ij->i", E2v, r_hat)[:, None] * r_hat
    return (
        np.asarray(pxs),
        np.asarray(r_hat, dtype=float),
        np.asarray(E1v, dtype=float),
        np.asarray(E2v, dtype=float),
    )


def _design(sample_idx, E1v, E2v, r_hat):
    n = len(sample_idx)
    A = np.empty((3 * n, 3))
    A[0::3] = E1v[sample_idx]
    A[1::3] = E2v[sample_idx]
    A[2::3] = r_hat[sample_idx]
    return A


def group_by_pixel(pxs, valid, min_visits=1, cond_ref=None, cond_thresh=30.0):
    """
    Group valid sample indices by pixel.

    Parameters
    ----------
    pxs : array-like of int, shape (nsample,)
    valid : array-like of bool, shape (nsample,)
    min_visits : int
        Pixels with fewer valid samples are dropped.
    cond_ref : tuple (r_hat, E1v, E2v), optional
        If given, drop pixels whose stacked design matrix has condition
        number above ``cond_thresh``. At a fixed transmitter geometry a
        visit's rows (E1v, E2v, r_hat) are one fixed triple rotated, so
        a single visit's cond(A) is the same for every sample, and
        repeating a visit at the same roll does not change it: this gate
        does NOT detect pixels whose visits lack roll diversity. It does
        vary with a trial geometry (through the arms' foreshortening
        along the line of sight), so in a geometry search it rejects
        placements where the two arms are nearly degenerate.
    cond_thresh : float

    Returns
    -------
    groups : dict
        pixel -> np.ndarray of sample indices, in sample order.
    """
    pxs = np.asarray(pxs)
    valid = np.asarray(valid, dtype=bool)
    pxs_valid = pxs[valid]
    idx_valid = np.flatnonzero(valid)
    order = np.argsort(pxs_valid, kind="stable")
    pxs_sorted = pxs_valid[order]
    idx_sorted = idx_valid[order]
    boundaries = np.flatnonzero(np.diff(pxs_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(pxs_sorted)]))
    groups = {}
    for s, e in zip(starts, ends):
        if e - s < min_visits:
            continue
        sample_idx = idx_sorted[s:e]
        if cond_ref is not None:
            r_hat_c, E1v_c, E2v_c = cond_ref
            A = _design(sample_idx, E1v_c, E2v_c, r_hat_c)
            if np.linalg.cond(A) > cond_thresh:
                continue
        groups[int(pxs_sorted[s])] = sample_idx
    return groups


def solve_pixel(sample_idx, E1v, E2v, r_hat, Ex, Ey, g=1.0 + 0j):
    """
    Least-squares u at one pixel from all of its visits.

    Parameters
    ----------
    sample_idx : array-like of int
        Samples landing in the pixel.
    E1v, E2v, r_hat : np.ndarray, shape (nsample, 3)
        From :func:`transmitter_fields`.
    Ex, Ey : np.ndarray of complex, shape (nsample,)
        Arm 1 and arm 2 cross-correlations at one channel.
    g : complex
        D-term; Ey is divided by it before solving.

    Returns
    -------
    u : np.ndarray of complex, shape (3,)
    resid : float
        ``|A u - b|``.
    resid_frac : float
        ``resid / |b|``.
    cond : float
        Condition number of the stacked design matrix.
    """
    sample_idx = np.asarray(sample_idx)
    A = _design(sample_idx, E1v, E2v, r_hat).astype(complex)
    b = np.zeros(3 * len(sample_idx), dtype=complex)
    b[0::3] = Ex[sample_idx]
    b[1::3] = Ey[sample_idx] / g
    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = np.linalg.norm(A @ u - b)
    denom = np.linalg.norm(b) + 1e-30
    cond = np.linalg.cond(A)
    return u, resid, resid / denom, cond


def solve_sampled_beam(
    pxs,
    r_hat,
    E1v,
    E2v,
    Ex,
    Ey,
    valid,
    g=1.0 + 0j,
    min_visits=1,
    cond_thresh=30.0,
):
    """
    Solve u at every sampled pixel of one channel.

    Each pixel's own condition number is the gate: pixels above
    ``cond_thresh`` are left out. No smoothing across pixels or
    frequency, and no value is produced for an unsampled pixel.

    Returns
    -------
    dict with
        pix : np.ndarray of int, shape (npix_solved,)
        u : np.ndarray of complex, shape (npix_solved, 3)
        n_visits, cond, resid_frac : np.ndarray, shape (npix_solved,)
    """
    groups = group_by_pixel(pxs, valid, min_visits=min_visits)
    pix, us, nv, conds, fracs = [], [], [], [], []
    for p, sample_idx in groups.items():
        u, _, frac, cond = solve_pixel(
            sample_idx, E1v, E2v, r_hat, Ex, Ey, g=g
        )
        if np.isfinite(cond) and cond <= cond_thresh:
            pix.append(p)
            us.append(u)
            nv.append(len(sample_idx))
            conds.append(cond)
            fracs.append(frac)
    order = np.argsort(pix)
    return dict(
        pix=np.asarray(pix, dtype=int)[order],
        u=np.asarray(us, dtype=complex).reshape(-1, 3)[order],
        n_visits=np.asarray(nv, dtype=int)[order],
        cond=np.asarray(conds, dtype=float)[order],
        resid_frac=np.asarray(fracs, dtype=float)[order],
    )


def _closure_objective(g_reim, groups, E1v, E2v, r_hat, Ex, Ey, cond_gate):
    g = complex(g_reim[0], g_reim[1])
    if abs(g) < 1e-8:
        return 10.0
    fracs = []
    for sample_idx in groups.values():
        _, _, frac, cond = solve_pixel(
            sample_idx, E1v, E2v, r_hat, Ex, Ey, g=g
        )
        if np.isfinite(frac) and np.isfinite(cond) and cond <= cond_gate:
            fracs.append(frac)
    if not fracs:
        return 10.0
    return float(np.mean(fracs))


def fit_closure_g(
    pxs,
    r_hat,
    E1v,
    E2v,
    Ex,
    Ey,
    valid,
    min_visits=3,
    cond_thresh=30.0,
    seeds=None,
    cond_gate=1e6,
):
    """
    Fit one channel's D-term g by closure, with no beam model.

    A pixel visited n >= 2 times gives 3n equations for 3 unknowns, so
    the residual only collapses when g (and the geometry) are right.
    Minimizes the mean fractional residual over well-conditioned,
    multiply-visited pixels with Nelder-Mead from several seeds.

    Parameters
    ----------
    min_visits : int
        Minimum visits for a pixel to enter the objective.
    cond_thresh : float
        Pixel conditioning gate used when selecting the pixels.
    seeds : sequence of complex, optional
        Starting values; default ``G_SEED, 1, -1, 1j``.
    cond_gate : float
        Looser per-evaluation conditioning gate inside the objective.

    Returns
    -------
    g : complex
    """
    groups = group_by_pixel(
        pxs,
        valid,
        min_visits=min_visits,
        cond_ref=(r_hat, E1v, E2v),
        cond_thresh=cond_thresh,
    )
    if seeds is None:
        seeds = [G_SEED, 1.0, -1.0, 1j]
    best = None
    for seed in seeds:
        seed = complex(seed)
        res = minimize(
            _closure_objective,
            (seed.real, seed.imag),
            args=(groups, E1v, E2v, r_hat, Ex, Ey, cond_gate),
            method="Nelder-Mead",
            options=dict(xatol=1e-4, fatol=1e-6, maxiter=300),
        )
        if best is None or res.fun < best.fun:
            best = res
    return complex(best.x[0], best.x[1])


def reduce_cross(
    cross,
    chs,
    tone_mask,
    scale_factors,
    freqs_mhz,
    shift,
    tau_ns,
    trust_mask=None,
    nmodes=23,
    NW=4.0,
    flag_fill_outliers=True,
):
    """
    Reduce raw cross-correlation spectra to per-arm Ex_hat, Ey_hat.

    Steps, in order: subtract a three-point local baseline at each comb
    tone; scale the weak arm (odd tones) by ``sqrt(scale_factors)``
    (scale_factors equalizes power, the cross is an amplitude); drop the
    first ``shift`` tones; NaN the flagged tones; remove a fixed cable
    delay so the residual phase is slow enough for a DPSS basis; fit the
    interleaved arms with :func:`~eigsep_data.rfi.fit_dpss_interleaved_fix`
    (real and imaginary parts separately); optionally NaN implausible
    fills; restore the delay.

    Parameters
    ----------
    cross : np.ndarray of complex, shape (ntime, nchan)
    chs : array-like of int, shape (ntone,)
        Comb tone channels, alternating arm 1 / arm 2.
    tone_mask : np.ndarray of bool, shape (ntime, ntone - shift)
        Flags on the reduced tones after the shift, from whatever RFI
        flagging the caller applies.
    scale_factors : array-like, shape (ntone // 2,) or broadcastable
        Weak-arm power calibration (``data.calibrate_weak_arm``).
    freqs_mhz : np.ndarray, shape (ntone - shift,)
        Frequency of each reduced tone.
    shift : int
        Leading tones to drop.
    tau_ns : float
        Delay to remove before the DPSS fit and restore after it.
    trust_mask : np.ndarray of bool, shape (ntone - shift,), optional
        Passed to fit_dpss_interleaved_fix.
    nmodes, NW : DPSS parameters.
    flag_fill_outliers : bool
        NaN DPSS fills that fail
        :func:`~eigsep_data.rfi.flag_dpss_interleaved_outliers`.

    Returns
    -------
    Ex_hat, Ey_hat : np.ndarray of complex, shape (ntime, ntone - shift)
    bad_time : np.ndarray of bool, shape (ntime,)
    Ex_measured, Ey_measured : np.ndarray of bool
        True where the value is that arm's own measurement.
    """
    chs = np.asarray(chs)
    reduced = cross[:, chs] - 0.5 * (cross[:, chs - 1] + cross[:, chs + 1])
    reduced = reduced.copy()
    reduced[:, 1::2] *= np.sqrt(scale_factors)
    reduced = reduced[:, shift:]

    clean = reduced.astype(complex)
    # Both parts: a bare np.nan on a complex array leaves imag finite.
    clean[np.asarray(tone_mask, dtype=bool)] = np.nan + 1j * np.nan

    derot = np.exp(-2j * np.pi * freqs_mhz * 1e6 * tau_ns * 1e-9)
    clean = clean * derot[None, :]

    Exr, Eyr, _, bad_r, _, Ex_meas, Ey_meas = fit_dpss_interleaved_fix(
        clean.real,
        freqs_mhz,
        nmodes=nmodes,
        NW=NW,
        return_measured_mask=True,
        trust_mask=trust_mask,
    )
    Exi, Eyi, _, bad_i, _ = fit_dpss_interleaved_fix(
        clean.imag,
        freqs_mhz,
        nmodes=nmodes,
        NW=NW,
        trust_mask=trust_mask,
    )
    Ex_hat = Exr + 1j * Exi
    Ey_hat = Eyr + 1j * Eyi
    bad_time = bad_r | bad_i

    if flag_fill_outliers:
        _, _, Ex_out, Ey_out = flag_dpss_interleaved_outliers(
            np.abs(Ex_hat), np.abs(Ey_hat), Ex_meas, Ey_meas
        )
        Ex_hat[Ex_out] = np.nan + 1j * np.nan
        Ey_hat[Ey_out] = np.nan + 1j * np.nan

    rerot = np.conj(derot)
    return (
        Ex_hat * rerot[None, :],
        Ey_hat * rerot[None, :],
        bad_time,
        Ex_meas,
        Ey_meas,
    )


def dwell_clusters(sample_idx, gap=20):
    """
    Split sample indices into dwells: a jump of more than ``gap``
    samples starts a new dwell.

    Consecutive integrations within one dwell are near duplicates, so
    validation that needs independent subsets must split whole dwells,
    never samples.
    """
    sample_idx = np.sort(np.asarray(sample_idx))
    if len(sample_idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(sample_idx) > gap) + 1
    return np.split(sample_idx, breaks)


def leave_one_out(
    pxs,
    r_hat,
    E1v,
    E2v,
    Ex,
    Ey,
    valid,
    g,
    min_visits=3,
    cond_thresh=30.0,
):
    """
    Held-out forward check: for each visit to a multiply-visited pixel,
    solve u from the other visits and predict this visit's Ex and Ey.

    Unlike re-predicting the data a solve was fitted to, the predicted
    visit never enters its own solve.

    Returns
    -------
    dict of np.ndarray: ex_pred, ex_true, ey_pred, ey_true,
    power_pred, power_true, plus n_skipped_cond (int).
    """
    groups = group_by_pixel(
        pxs,
        valid,
        min_visits=min_visits,
        cond_ref=(r_hat, E1v, E2v),
        cond_thresh=cond_thresh,
    )
    out = {k: [] for k in ("ex_pred", "ex_true", "ey_pred", "ey_true")}
    n_skip = 0
    for sample_idx in groups.values():
        sample_idx = np.asarray(sample_idx)
        for i, t in enumerate(sample_idx):
            train = np.delete(sample_idx, i)
            u, _, _, cond = solve_pixel(train, E1v, E2v, r_hat, Ex, Ey, g=g)
            if not (np.isfinite(cond) and cond <= cond_thresh):
                n_skip += 1
                continue
            out["ex_pred"].append(E1v[t] @ u)
            out["ex_true"].append(Ex[t])
            out["ey_pred"].append(g * (E2v[t] @ u))
            out["ey_true"].append(Ey[t])
    out = {k: np.asarray(v, dtype=complex) for k, v in out.items()}
    out["power_pred"] = (
        np.abs(out["ex_pred"]) ** 2 + np.abs(out["ey_pred"]) ** 2
    )
    out["power_true"] = (
        np.abs(out["ex_true"]) ** 2 + np.abs(out["ey_true"]) ** 2
    )
    out["n_skipped_cond"] = n_skip
    return out


def split_half_by_dwell(
    pxs,
    r_hat,
    E1v,
    E2v,
    Ex,
    Ey,
    valid,
    g,
    pixels=None,
    min_dwells=2,
    gap=20,
    cond_thresh=30.0,
):
    """
    Self-consistency check: solve u twice per pixel from alternating
    whole dwells and return both halves.

    Parameters
    ----------
    pixels : iterable of int, optional
        Restrict to these pixels (e.g. those passing an absurd-value
        cut on the full solve).
    min_dwells : int
        Pixels with fewer dwells are skipped.
    gap : int
        Dwell boundary, see :func:`dwell_clusters`.

    Returns
    -------
    pix : np.ndarray of int
    u_a, u_b : np.ndarray of complex, shape (npix_used, 3)
    n_dwells : np.ndarray of int
    """
    groups = group_by_pixel(pxs, valid, min_visits=1)
    keep = None if pixels is None else set(int(p) for p in pixels)
    pix, ua, ub, nd = [], [], [], []
    for p, sample_idx in groups.items():
        if keep is not None and p not in keep:
            continue
        clusters = dwell_clusters(sample_idx, gap=gap)
        if len(clusters) < min_dwells:
            continue
        half_a = np.concatenate(clusters[0::2])
        half_b = np.concatenate(clusters[1::2])
        if len(half_a) == 0 or len(half_b) == 0:
            continue
        u1, _, _, c1 = solve_pixel(half_a, E1v, E2v, r_hat, Ex, Ey, g=g)
        u2, _, _, c2 = solve_pixel(half_b, E1v, E2v, r_hat, Ex, Ey, g=g)
        if not (
            np.isfinite(c1)
            and c1 <= cond_thresh
            and np.isfinite(c2)
            and c2 <= cond_thresh
        ):
            continue
        if not (np.all(np.isfinite(u1)) and np.all(np.isfinite(u2))):
            continue
        pix.append(p)
        ua.append(u1)
        ub.append(u2)
        nd.append(len(clusters))
    return (
        np.asarray(pix, dtype=int),
        np.asarray(ua, dtype=complex).reshape(-1, 3),
        np.asarray(ub, dtype=complex).reshape(-1, 3),
        np.asarray(nd, dtype=int),
    )


def compare_to_truth(u_hat, u_true, absurd="mad", n_mad=5.0, min_pixels=20):
    """
    Score a recovered beam against a reference vector beam on the same
    pixels.

    The recovered beam has an unknown overall complex scale (gain and
    phase), so it is first aligned by the single least-squares complex
    factor ``a = <u_hat, u_true> / <u_hat, u_hat>``. Pixels with an
    implausibly large recovered total power are dropped before that.

    Parameters
    ----------
    u_hat, u_true : np.ndarray of complex, shape (npix, 3)
    absurd : "mad" or float
        "mad" drops totals above median + n_mad * 1.4826 * MAD
        (:func:`~eigsep_data.rfi.flag_absurd_cells`); a number drops
        totals above that multiple of the median (the older rule, which
        is too loose on noisy channels).
    n_mad : float
    min_pixels : int

    Returns
    -------
    dict with power, x, y, z (Pearson correlations of the aligned
    ``|.|^2`` per component and summed), n (pixels used), n_absurd,
    scale (the complex factor a); or None if fewer than ``min_pixels``
    usable pixels.
    """
    u_hat = np.asarray(u_hat, dtype=complex)
    u_true = np.asarray(u_true, dtype=complex)
    finite = np.all(np.isfinite(u_hat), axis=1) & np.all(
        np.isfinite(u_true), axis=1
    )
    u_hat, u_true = u_hat[finite], u_true[finite]
    if len(u_hat) < min_pixels:
        return None

    total = np.sum(np.abs(u_hat) ** 2, axis=1)
    if isinstance(absurd, str):
        if absurd != "mad":
            raise ValueError(f"absurd must be 'mad' or a number: {absurd!r}")
        bad = flag_absurd_cells(total, n_mad=n_mad)
    elif np.any(total > 0):
        bad = total > float(absurd) * np.median(total[total > 0])
    else:
        bad = np.zeros(len(total), dtype=bool)
    u_hat, u_true = u_hat[~bad], u_true[~bad]
    if len(u_hat) < min_pixels:
        return None

    a = np.sum(np.conj(u_hat) * u_true) / (
        np.sum(np.conj(u_hat) * u_hat) + 1e-30
    )
    al = a * u_hat
    out = dict(
        power=float(
            np.corrcoef(
                np.sum(np.abs(al) ** 2, axis=1),
                np.sum(np.abs(u_true) ** 2, axis=1),
            )[0, 1]
        ),
        n=len(u_hat),
        n_absurd=int(bad.sum()),
        scale=complex(a),
    )
    for k, name in enumerate("xyz"):
        out[name] = float(
            np.corrcoef(np.abs(al[:, k]) ** 2, np.abs(u_true[:, k]) ** 2)[0, 1]
        )
    return out
