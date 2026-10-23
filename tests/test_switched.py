"""Tests for eigsep_data.switched: dwells, cycles and sensors of switched
calibration measurements."""

import numpy as np
import pandas as pd
import pytest

from eigsep_data import switched as sw


def test_thermistor_matches_the_ysi_table():
    # YSI 44909 (curve H): 30 kOhm at 25 C, 10.1 kOhm-ish at 50 C; with the
    # firmware's assumed pull-up the logged resistance is the true one.
    assert sw.thermistor_k(30000.0, 10000.0) == pytest.approx(298.15, abs=0.05)
    # A 1 % smaller pull-up means the true resistance is 1 % smaller than logged.
    t_meas = sw.thermistor_k(30000.0, 9900.0)
    assert t_meas == pytest.approx(sw.thermistor_k(29700.0, 10000.0))
    assert t_meas > 298.15


def _meta(times, states, dwell, pos, tau=1.0):
    return pd.DataFrame({"time_best": times, "rfswitch": states, "dwell": dwell,
                         "pos_in_dwell": pos, "integration_time": tau})


def test_dwell_means_weight_by_time_and_skip_flagged_samples():
    meta = _meta([0, 1, 2, 10, 11], ["RFAMB"] * 5, [0, 0, 0, 1, 1], [0, 1, 2, 0, 1])
    meta["integration_time"] = [1.0, 1.0, 2.0, 1.0, 1.0]
    meta["T"] = [300.0, 301.0, 302.0, np.nan, 310.0]
    spec = np.array([[1.0, 5.0], [2.0, 5.0], [4.0, -1.0], [7.0, 7.0], [9.0, 9.0]])   # -1: a wrap
    flags = sw.overflow_mask(spec)
    flags[0, 1] = True
    table, p = sw.dwell_means(meta, spec, flags, columns=("T",))
    np.testing.assert_allclose(p[0], [(1 + 2 + 2 * 4) / 4, 5.0])
    np.testing.assert_allclose(p[1], [8.0, 8.0])
    np.testing.assert_allclose(table["T"], [(300 + 301 + 2 * 302) / 4, 310.0])
    assert list(table["n"]) == [3, 2]
    _, p_skip = sw.dwell_means(meta, spec, flags, skip_first=1)
    np.testing.assert_allclose(p_skip[0, 0], (2 + 2 * 4) / 3)


def test_sweeps_and_their_guard():
    t = np.arange(0.0, 300.0, 1.0)
    states = np.array(["RFANT"] * 300, dtype=object)
    states[100:130] = "VNAANT"
    states[200:210] = "VNARF"            # 70 s after the first: a separate sweep
    meta = pd.DataFrame({"time_best": t, "rfswitch": states})
    sweeps = sw.vna_sweeps(meta)
    np.testing.assert_allclose(sweeps, [[100, 129], [200, 209]])
    assert len(sw.vna_sweeps(meta, gap_s=80.0)) == 1      # within the gap: one sweep
    near = sw.near_sweep(meta, sweeps, guard=(20.0, 5.0))
    assert not near[80] and near[81] and near[133] and not near[134]


def _dwells(starts, nchan=3, value=1.0):
    table = pd.DataFrame({"t": np.asarray(starts, float), "n": 8})
    return table, np.full((len(starts), nchan), value)


def test_cycles_need_every_state_before_the_next_start():
    order = ("RFANT", "RFAMB", "RFNON")
    dw = {"RFANT": _dwells([0, 60, 120]), "RFAMB": _dwells([10, 70, 130], value=2.0),
          "RFNON": _dwells([20, 140], value=3.0)}            # the cycle at 60 has no RFNON
    table, p = sw.cycles(dw, order)
    np.testing.assert_allclose(table["t"], [0, 120])
    assert list(table["i_RFNON"]) == [0, 1]
    np.testing.assert_allclose(p["RFAMB"], 2.0)
    table, _ = sw.cycles(dw, order, max_span_s=15)            # RFNON 20 s after the start: too late
    assert len(table) == 0


def test_classify_rate():
    t = np.arange(0, 600, 60.0)
    temp = np.r_[np.full(4, 300.0), 305.0, 310.0, 315.0, np.full(3, 315.0)]
    rate, kind = sw.classify_rate(t, temp, steady_per_min=1.0)
    assert kind[0] == "steady" and kind[-1] == "steady"
    assert (kind[4:6] == "heating").all()
    _, kind_neg = sw.classify_rate(t, temp[::-1], steady_per_min=1.0)
    assert "cooling" in set(kind_neg)


def test_rfi_flags_find_a_transient_and_a_line():
    rng = np.random.default_rng(0)
    nt, nch = 60, 1024
    f = np.linspace(0, 250, nch, endpoint=False)
    spec = 1e6 * (1 + 1e-3 * rng.standard_normal((nt, nch)))
    spec[30, 400] *= 5.0                  # a transient in one sample
    spec[:, 600] *= 3.0                   # a steady line at ~146 MHz
    spec[10, 100] = -2e9                  # a wrap
    flags, persistent = sw.rfi_flags(spec, f)
    assert flags[30, 400] and flags[10, 100]
    assert persistent[600]
    assert flags.mean() < 0.01
