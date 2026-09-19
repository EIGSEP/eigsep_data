"""Behavior and product-contract tests for supported-DPSS v3-beta."""

from dataclasses import replace

import h5py
import numpy as np
import pandas as pd

from eigsep_data.bundle import Campaign
from eigsep_data.products import get
from eigsep_data.rfi_supported import (
    BIT_BY_REASON,
    RFIConfig,
    RFIResult,
    encode_reasons,
    flag_arrays,
    write_products,
)


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


def test_writer_round_trips_through_product_readers(tmp_path):
    root = tmp_path
    (root / "data").mkdir()
    fname = "corr_20260716_031155Z.h5"
    freqs = np.linspace(35, 235, 8, endpoint=False)
    with h5py.File(root / "data" / fname, "w") as h5:
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
        meta=pd.DataFrame({"file": [fname] * 3, "row": np.arange(3)}),
        config=RFIConfig(),
        diagnostics={},
    )
    write_products(result, root)
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
