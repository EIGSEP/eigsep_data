"""End-to-end synthetic beam recovery through eigsep_data.beam_mapping.

A transmitter comb seen through the HFSS beam on a raster-like scan (continuous
+/-180 deg elevation sweeps, azimuth stepping 6 deg per sweep across ~227 deg,
as on Marjum 07-17), on top of a power-law sky with a 110 ns ripple, with 0.3%
radiometer noise and 0.5% leakage into the channels beside each tooth. The
pipeline must interpolate the background under the comb, find the az offset and
polarization, and, starting from a deliberately wrong beam, recover the HFSS
beam on held-out azimuth stripes.
"""

import os

import healpy as hp
import numpy as np
import pytest

import eigsep_data.beam_mapping as bm
from eigsep_data.beam_sim import DEFAULT_BEAM_PATH, read_beam

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_BEAM_PATH), reason="HFSS beam map not present")

PSI = 142.164
ANT = np.array([1655.868, 2030.923, 1777.397])
TX = np.array([1652.198, 2025.104, 1684.384])
TRUE_OFFSET, TRUE_ALPHA = -16.0, 96.0
TRUE_PARAMS = np.array([0.0, 0.0, 0.6, 0.0, 0.0])
FREQ = np.arange(1024) * 250.0 / 1024
TEETH = np.arange(600, 848, 8)                       # 31 teeth, 146.5-205 MHz, both arms


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(7)
    az, el = [], []
    for i, a in enumerate(np.arange(188.0, 416.0, 6.0) % 360):
        e = np.arange(-180.0, 180.0, 6.0) + rng.uniform(0, 6.0)
        az.append(np.full(e.size, a))
        el.append(e if i % 2 == 0 else e[::-1])
    az, el = np.concatenate(az), (np.concatenate(el) + 180) % 360 - 180
    n = az.size
    fields, _, hfreqs = read_beam()
    truth_teeth = bm.tooth_fields(fields, hfreqs, FREQ[TEETH])
    truth_model = bm.TxGeometryModel(PSI, ANT, TX, alpha0_deg=TRUE_ALPHA, az_offset_deg=TRUE_OFFSET)
    frame = bm.ToothData(t=np.arange(n) * 0.537, az_deg=az, el_deg=el, data=np.zeros((TEETH.size, n)),
                         good=np.ones((TEETH.size, n), bool), channels=TEETH, freqs_mhz=FREQ[TEETH],
                         split=bm.stripe_split(az))
    gains = 3e6 * (1 + 0.3 * np.sin(np.arange(TEETH.size) / 5.0)) * np.where(frame.arms == 0, 0.15, 1.0)
    signal = gains[:, None] * bm.hfss_power(truth_teeth, frame, truth_model, TRUE_PARAMS)

    sky = 1e6 * (FREQ.clip(30) / 150.0) ** -2.5
    level = 1 + 0.2 * rng.uniform(size=(n, 1))
    ripple = 1 + 0.02 * np.cos(2 * np.pi * 0.110 * FREQ[None, :] + rng.uniform(0, 2 * np.pi, (n, 1)))
    spectra = level * sky[None, :] * ripple
    spectra[:, TEETH] += signal.T
    spectra[:, TEETH + 1] += 0.005 * signal.T
    spectra[:, TEETH - 1] += 0.005 * signal.T
    spectra *= 1 + 0.003 * rng.normal(size=spectra.shape)

    background = bm.tooth_background(spectra, FREQ, TEETH, limits_mhz=(40, 249))
    data = bm.ToothData(t=frame.t, az_deg=az, el_deg=el, data=spectra[:, TEETH].T - background.T,
                        good=np.ones((TEETH.size, n), bool), channels=TEETH, freqs_mhz=FREQ[TEETH],
                        split=frame.split)

    # The starting beam: HFSS with a smooth angular distortion the truth does not have.
    npix = fields.shape[-1]
    x, y, z = hp.pix2vec(hp.npix2nside(npix), np.arange(npix))
    prior_fields = fields * (1 + 0.3 * x * y + 0.15 * z)[None, None, :]
    return dict(fields=fields, hfreqs=hfreqs, truth_teeth=truth_teeth, truth_model=truth_model,
                signal=signal, data=data, prior_fields=prior_fields,
                prior_teeth=bm.tooth_fields(prior_fields, hfreqs, FREQ[TEETH]))


@pytest.fixture(scope="module")
def fitted(sim):
    data = sim["data"]
    model, _ = bm.coarse_offset_alpha(data, sim["prior_teeth"], bm.TxGeometryModel(PSI, ANT, TX))
    params, _, _, _ = bm.fit_geometry(data, sim["prior_teeth"], model)
    hfreqs = sim["hfreqs"]
    band = (FREQ[TEETH].min() - FREQ[1], FREQ[TEETH].max() + FREQ[1])
    sel = np.flatnonzero((hfreqs >= band[0]) & (hfreqs <= band[1]))
    sel = np.r_[sel[0] - 1, sel, sel[-1] + 1]
    prior_norm, _, _ = bm.normalize_fields(sim["prior_fields"], hfreqs)
    lmax, sh, _ = bm.select_lmax(prior_norm[sel])
    basis = bm.pca_basis(hfreqs[sel], prior_norm[sel])
    c0 = bm.initial_coefficients(basis, sh)
    jf = bm.JointBeamFit(data, model, params, lmax, basis, c0)
    coeff, gain, prediction, info = jf.fit(data.split == 0, penalty=1e-3, maxiter=800)
    prior_power = jf.power(c0)
    prior = bm.tooth_gains(prior_power, data.data, data.good, data.split == 0)[:, None] * prior_power
    return dict(model=model, params=params, lmax=lmax, basis=basis, coeff=coeff, gain=gain,
                prediction=prediction, prior=prior, info=info, band=band, prior_norm=prior_norm)


def held_out_error(sim, prediction):
    held = sim["data"].split > 0
    return (np.linalg.norm((prediction - sim["signal"])[:, held])
            / np.linalg.norm(sim["signal"][:, held]))


def test_background_under_the_comb_is_removed(sim):
    err = sim["data"].data - sim["signal"]
    assert np.median(np.abs(err)) / np.sqrt(np.mean(sim["signal"] ** 2)) < 5e-3


def test_coarse_scan_finds_the_az_offset_and_polarization(fitted):
    assert fitted["model"].az_offset_deg == pytest.approx(TRUE_OFFSET, abs=1.0)
    assert fitted["model"].alpha0_deg == pytest.approx(TRUE_ALPHA, abs=1.0)


def test_geometry_is_recovered_with_the_true_beam(sim, fitted):
    params, _, _, _ = bm.fit_geometry(sim["data"], sim["truth_teeth"], fitted["model"],
                                      candidates=(("az_el", (1, 2)),))
    assert params[2] == pytest.approx(TRUE_PARAMS[2], abs=0.1)
    assert abs(params[1]) < 0.3


def test_joint_fit_recovers_the_hfss_beam_on_held_out_stripes(sim, fitted):
    assert fitted["info"]["success"]
    prior, recovered = held_out_error(sim, fitted["prior"]), held_out_error(sim, fitted["prediction"])
    assert prior > 0.15                         # the starting beam is clearly wrong ...
    assert recovered < 0.03                     # ... and the fit recovers HFSS to a few percent
    assert recovered < prior / 5


def test_export_round_trips_and_reproduces_the_fit(sim, fitted, tmp_path):
    path = tmp_path / "empirical_beam.npz"
    export_gain, fields, freqs = bm.export_beam(path, sim["hfreqs"], fitted["prior_norm"], fitted["basis"],
                                                fitted["coeff"], fitted["lmax"], fitted["band"], FREQ[TEETH],
                                                fitted["gain"])
    loaded, _, f = read_beam(path, drop_last=False)
    np.testing.assert_array_equal(f, freqs)
    np.testing.assert_array_equal(loaded, fields)
    idx = np.searchsorted(freqs, FREQ[TEETH])
    exported = export_gain[:, None] * bm.hfss_power(loaded[idx], sim["data"], fitted["model"], fitted["params"])
    # The export is a HEALPix map sampled by bilinear interpolation, the fit an exact
    # harmonic evaluation: they agree to the map's pixelization (the legacy 1% criterion).
    data = sim["data"].data
    per_tooth = np.sqrt(np.mean((exported - fitted["prediction"]) ** 2, axis=1) / np.mean(data ** 2, axis=1))
    assert per_tooth.max() < 0.01
