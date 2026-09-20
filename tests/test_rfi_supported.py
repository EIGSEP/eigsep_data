"""Behavior and product-contract tests for supported-DPSS v3-beta."""

from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from eigsep_data.bundle import Campaign
from eigsep_data import MetadataIndex
from eigsep_data.products import get
from eigsep_data.rfi_supported import (
    BIT_BY_REASON,
    RFIConfig,
    RFIResult,
    _support,
    _TensorFit,
    algorithm_source_sha256,
    encode_reasons,
    flag_arrays,
    load_selection_inputs,
    write_products,
)

from conftest import write_corr_file


def test_sparse_tensor_solver_and_support_match_dense_design():
    rng = np.random.default_rng(912)
    solver = _TensorFit.__new__(_TensorFit)
    solver.config = RFIConfig()
    solver.history = []
    solver.qt = np.linalg.qr(rng.normal(size=(31, 4)))[0]
    solver.qf = np.linalg.qr(rng.normal(size=(27, 6)))[0]
    solver.scale = np.ones(27)
    solver.shape = (4, 6)
    solver.active_mask = np.zeros(solver.shape, dtype=bool)
    solver.active_mask[:, :3] = True
    solver.active_mask[0, 3:] = True
    solver.active = np.flatnonzero(solver.active_mask.ravel())
    solver.ncoeff = len(solver.active)
    solver.stationary_frequency_modes = 3

    data = rng.normal(size=(31, 27))
    keep = rng.random(data.shape) > 0.35
    prediction = solver.solve(data, keep)
    design = np.einsum("ta,fb->tfab", solver.qt, solver.qf).reshape(
        data.size, -1
    )[:, solver.active]
    normal = design[keep.ravel()].T @ design[keep.ravel()]
    inverse = np.linalg.inv(
        normal + solver.config.ridge * np.eye(solver.ncoeff)
    )
    coefficients = inverse @ (design[keep.ravel()].T @ data[keep])
    np.testing.assert_allclose(
        prediction, (design @ coefficients).reshape(data.shape), atol=1e-7
    )

    inflation, _prior, _seconds = _support(solver, keep)
    expected = np.sqrt(
        np.einsum("ni,ij,nj->n", design, inverse, design)
        / np.sum(design**2, axis=1)
    ).reshape(data.shape)
    np.testing.assert_allclose(inflation, expected, rtol=1e-8)


def test_stationary_spectral_correction_fits_broad_low_band_structure():
    times = np.linspace(0, 1200, 100)
    freqs = np.linspace(35, 235, 256, endpoint=False)
    smooth = (
        1e6
        * (1 + 0.12 * np.cos(2 * np.pi * freqs[None, :] / 150))
        * (1 + 0.02 * np.cos(2 * np.pi * times[:, None] / 1800))
    )
    truth = smooth * (
        1 + 0.05 * np.exp(-0.5 * ((freqs[None, :] - 60) / 4) ** 2)
    )
    keep = np.ones_like(truth, dtype=bool)
    without = _TensorFit(
        times,
        freqs,
        truth,
        replace(RFIConfig(), spectral_correction_halfwidth_s=0),
    ).solve(truth, keep)
    solver = _TensorFit(times, freqs, truth, RFIConfig())
    corrected = solver.solve(truth, keep)
    low = (freqs >= 50) & (freqs < 70)
    baseline_error = np.sqrt(
        np.mean((without[:, low] / truth[:, low] - 1) ** 2)
    )
    corrected_error = np.sqrt(
        np.mean((corrected[:, low] / truth[:, low] - 1) ** 2)
    )
    assert solver.stationary_frequency_modes > 0
    assert corrected_error < baseline_error / 20


@pytest.mark.parametrize(
    "name",
    ["low_band_overflag_case1.npz", "low_band_overflag_case2.npz"],
)
def test_real_low_band_regressions_recover_good_data(name):
    path = Path(__file__).parent / "data" / "rfi_v3_beta" / name
    with np.load(path, allow_pickle=False) as fixture:
        arguments = (
            fixture["air"],
            fixture["ground"],
            fixture["cross"],
            fixture["times"],
            fixture["freqs_mhz"],
            fixture["integration_times"],
            fixture["switch_states"],
        )
        freqs = fixture["freqs_mhz"]
        provenance = json.loads(str(fixture["provenance_json"]))
    assert provenance["campaign"] == "marjum-2026-07"
    assert provenance["selection"]["offset_s"] == 10 * 3600
    assert all(
        len(digest) == 64 for digest in provenance["source_sha256"].values()
    )
    old = flag_arrays(
        *arguments,
        config=replace(RFIConfig(), spectral_correction_halfwidth_s=0),
    )
    corrected = flag_arrays(*arguments, config=RFIConfig())
    low = (freqs >= 50) & (freqs < 70)
    old_retained = (~old.mask[:, low]).mean()
    corrected_retained = (~corrected.mask[:, low]).mean()
    old_residual = np.nanmedian(np.abs(old.residual_z[:, low]))
    corrected_residual = np.nanmedian(np.abs(corrected.residual_z[:, low]))
    assert corrected.support_ok[:, low].mean() > 0.99
    assert corrected_retained > old_retained + 0.15
    assert corrected_residual < 0.4 * old_residual


def _synthetic():
    rng = np.random.default_rng(91)
    nt, nf = 160, 256
    times = np.linspace(0, 1200, nt)
    freqs = np.linspace(35, 235, nf, endpoint=False)
    dt = np.full(nt, times[1] - times[0])
    normalization = np.sqrt(2 * dt[0] * np.diff(freqs)[0] * 1e6)
    truth = (
        1e6
        * (1 + 0.12 * np.cos(2 * np.pi * freqs[None, :] / 150))
        * (1 + 0.02 * np.cos(2 * np.pi * times[:, None] / 1800))
    )
    air = truth * (1 + rng.normal(size=truth.shape) / normalization)
    ground = truth * 0.8
    noise = (
        rng.normal(size=truth.shape) + 1j * rng.normal(size=truth.shape)
    ) / np.sqrt(2)
    stable = 50 * np.exp(1j * 2 * np.pi * freqs[None, :] / 9)
    cross = (noise + stable) * np.sqrt(air * ground) / normalization
    states = np.full(nt, "RFANT", dtype=object)
    states[40:44] = "VNAO"
    return air, ground, cross, times, freqs, dt, states


def test_non_sky_is_whole_row_and_stable_coherence_is_background():
    air, ground, cross, times, freqs, dt, states = _synthetic()
    result = flag_arrays(
        air,
        ground,
        cross,
        times,
        freqs,
        dt,
        states,
        config=replace(RFIConfig(), time_guard=0),
    )
    nonsky = 1 << BIT_BY_REASON["non_sky_switch_state"]
    cross_bit = 1 << BIT_BY_REASON["cross_change"]
    assert np.all(result.flags[40:44] & nonsky)
    assert not np.any(result.flags[:40] & nonsky)
    assert (result.flags[:, :] & cross_bit != 0).mean() < 0.01


def test_cross_change_is_detected_on_first_changed_integration():
    air, ground, cross, times, freqs, dt, states = _synthetic()
    selected = (freqs > 180) & (freqs < 190)
    normalization = np.sqrt(2 * dt[0] * np.diff(freqs)[0] * 1e6)
    cross[90:96, selected] += (
        30
        * np.sqrt(air[90:96, selected] * ground[90:96, selected])
        / normalization
    )
    result = flag_arrays(
        air,
        ground,
        cross,
        times,
        freqs,
        dt,
        states,
        config=replace(RFIConfig(), time_guard=0),
    )
    bit = 1 << BIT_BY_REASON["cross_change"]
    assert ((result.flags[90, selected] & bit) != 0).mean() >= 0.9


def test_encode_reasons_allows_overlapping_uint16_bits():
    reasons = {name: np.zeros((2, 3), dtype=bool) for name in BIT_BY_REASON}
    reasons["positive_auto_excess"][0, 1] = True
    reasons["band_group_trigger"][0, 1] = True
    flags = encode_reasons(reasons)
    assert flags.dtype == np.uint16
    assert flags[0, 1] == (1 << 2) | (1 << 5)


def test_selection_resolves_antennas_and_cross_orientation_per_file(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    names = ["corr_20260716_000000Z.h5", "corr_20260716_000100Z.h5"]
    mappings = [
        {"0": "box-gnd", "2": "box-air"},
        {"0": "box-air", "2": "box-gnd"},
    ]
    for i, (name, mapping) in enumerate(zip(names, mappings)):
        write_corr_file(
            data / name,
            ntimes=8,
            sync_time=1.7843e9 + 100 * i,
            keys=("0", "2", "02"),
            input_to_ant=mapping,
            rfswitch=["RFANT"] * 8,
            seed=i,
        )
    selection = MetadataIndex(data, cache=False).select()
    bundles = load_selection_inputs(selection)
    for bundle, expected in (
        (bundles["air"], ["2", "0"]),
        (bundles["ground"], ["0", "2"]),
        (bundles["cross"], ["02", "02"]),
    ):
        resolved = [
            bundle.meta.loc[bundle.meta.file == name, "input_key"].iloc[0]
            for name in names
        ]
        assert resolved == expected
    orientations = [
        bool(
            bundles["cross"]
            .meta.loc[bundles["cross"].meta.file == name, "conjugated"]
            .iloc[0]
        )
        for name in names
    ]
    assert orientations == [False, True]


@pytest.mark.parametrize("external_data", [False, True])
def test_writer_round_trips_through_product_readers(tmp_path, external_data):
    root = tmp_path / "campaign"
    data_dir = tmp_path / "raw" if external_data else root / "data"
    data_dir.mkdir(parents=True)
    fname = "corr_20260716_031155Z.h5"
    freqs = np.linspace(35, 235, 8, endpoint=False)
    with h5py.File(data_dir / fname, "w") as h5:
        h5.create_group("data").create_dataset("4", data=np.ones((3, 8)))
    reasons = {name: np.zeros((3, 8), dtype=bool) for name in BIT_BY_REASON}
    reasons["positive_auto_excess"][1, 2] = True
    flags = encode_reasons(reasons)
    model = np.arange(24, dtype=float).reshape(3, 8) + 1
    result = RFIResult(
        flags=flags,
        model=model,
        model_raw=model.copy(),
        residual_z=np.zeros_like(model),
        support_ok=np.ones_like(model, bool),
        fit_keep=np.ones_like(model, bool),
        reasons=reasons,
        inflation=np.ones_like(model),
        prior_fraction=np.zeros_like(model),
        cross_background=np.zeros_like(model, dtype=complex),
        cross_score=np.zeros_like(model),
        freqs_mhz=freqs,
        times=np.arange(3.0),
        meta=pd.DataFrame(
            {"file": [fname] * 3, "row": np.arange(3), "input_key": "4"}
        ),
        config=RFIConfig(),
        diagnostics={},
    )
    write_products(result, root, data_dir=data_dir if external_data else None)
    campaign = Campaign(root)
    got_flags, got_freqs = get("flags").read_file(
        campaign, "v3-beta", fname, "4"
    )
    np.testing.assert_array_equal(got_flags, flags)
    np.testing.assert_array_equal(got_freqs, freqs)
    fetched = get("smooth_model").fetch(
        campaign,
        "v3-beta",
        fname,
        np.arange(3),
        "4",
        None,
        np.arange(3.0),
    )
    np.testing.assert_allclose(fetched["model"], model)
    metadata = get("flags").bits(campaign, "v3-beta")
    assert metadata["dtype"] == "uint16"
    assert metadata["bits"][4]["name"] == "cross_change"
    manifest = json.loads(
        (root / "flags" / "v3-beta" / "manifest.json").read_text()
    )
    assert (
        manifest["files"][fname]["algorithm_source_sha256"]
        == algorithm_source_sha256()
    )
