"""The package reproduces beam fit v0011 (Marjum 2026-07), which was produced by
the pre-package scripts in data-analysis/scripts/marjum-2026-07/.

From v0011's own inputs (its diagnostics, geometry and saved bases) the package
must give v0011's HFSS model, joint-fit prediction and exported-beam
prediction. Needs the campaign checkout beside eigsep_data (or
EIGSEP_CAMPAIGN_ROOT); skipped otherwise.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import eigsep_data.beam_mapping as bm
from eigsep_base.spectral_basis import SpectralBasis
from eigsep_data.beam_sim import read_beam

CAMPAIGN = Path(os.environ.get("EIGSEP_CAMPAIGN_ROOT",
                               Path(__file__).resolve().parents[2] / "marjum-2026-07"))
PRODUCT = CAMPAIGN / "derived/beam/empirical_raster_v0011"
pytestmark = pytest.mark.skipif(not (PRODUCT / "provenance.json").exists(),
                                reason="Marjum 2026-07 v0011 beam product not present")


@pytest.fixture(scope="module", params=["pca", "dpss"])
def case(request):
    kind = request.param
    z = dict(np.load(PRODUCT / kind / "diagnostics.npz"))
    data = bm.ToothData(t=z["t"], az_deg=z["az_table"], el_deg=z["el"], data=z["data"], good=z["good"],
                        channels=z["channels"], freqs_mhz=z["freqs"], split=z["split"],
                        fit_channels=z["fit_channels"])
    ant = json.loads((CAMPAIGN / "imgs/fits/v0001_marjum_geometry/shared.json").read_text())[
        "antenna_91m_era"]["position_enu_m"]
    tx = json.loads((CAMPAIGN / "curation/transmitter_position.json").read_text())["best_estimate_enu_m"]
    model = bm.TxGeometryModel(142.164, ant, tx, alpha0_deg=float(z["alpha0_deg"]),
                               az_offset_deg=float(z["az_offset_deg"]))
    return kind, z, data, model


def max_rel(a, b, good):
    return np.max(np.abs(a - b)[good]) / np.max(np.abs(b)[good])


def test_hfss_model(case):
    _, z, data, model = case
    fields, _, freqs = read_beam()
    p = bm.hfss_power(bm.tooth_fields(fields, freqs, data.freqs_mhz), data, model, z["params"])
    pred = bm.tooth_gains(p, data.data, data.good, data.split == 0)[:, None] * p
    assert max_rel(pred, z["hfss"], data.good) < 1e-10


def test_joint_fit_prediction(case):
    kind, z, data, model = case
    sb = np.load(PRODUCT / kind / "spectral_basis.npz")
    if kind == "pca":
        basis = SpectralBasis(sb["native_modes"], freqs=sb["native_freqs"])
    else:
        basis = bm.DPSSSpectralBasis(sb["native_modes"], sb["native_freqs"], sb["grid_mhz"], sb["dpss_modes"],
                                     sb["concentrations"], float(sb["half_bandwidth_cycles_per_sample"]),
                                     sb["qr_r"], None)
    lmax = json.loads((PRODUCT / "provenance.json").read_text())["lmax"]
    fit = bm.JointBeamFit(data, model, z["params"], lmax, basis, z["initial_coeff"])
    pred = z["final_gain"][:, None] * fit.power(z["coeff"])
    assert max_rel(pred, z["empirical_refit"], data.good) < 1e-10


def test_exported_beam_prediction(case):
    kind, z, data, model = case
    beam, _, freqs = read_beam(PRODUCT / kind / "empirical_beam.npz", drop_last=False)
    idx = np.searchsorted(freqs, data.freqs_mhz)
    pred = z["exported_gain"][:, None] * bm.hfss_power(beam[idx], data, model, z["params"])
    assert max_rel(pred, z["empirical_exported"], data.good) < 1e-10
