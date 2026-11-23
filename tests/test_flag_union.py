import json

import numpy as np
import pytest

from eigsep_data.flag_union import (
    FLAG_BITS_UNION,
    add_calibration_comb,
    add_self_emi,
    comb_channels,
    read_self_emi,
)
from eigsep_data.rfi_supported import ADVISORY_VALUE, FLAG_BITS

FREQS = np.array([0.84, 0.9, 1.0, 1.1, 1.16, 1.5])
COMB = np.array([False, True, True, True, False, False])


def _v3():
    flags = np.zeros((3, len(FREQS)), np.uint16)
    flags[0, 0] = 1 << 2  # an existing RFI reason
    flags[1, 1] = 1 << 9  # advisory high_scatter
    return flags


def test_new_bits_extend_v3_without_collision():
    assert set(FLAG_BITS) < set(FLAG_BITS_UNION)
    assert {"12", "13", "14", "15"} == set(FLAG_BITS_UNION) - set(FLAG_BITS)
    names = [v["name"] for v in FLAG_BITS_UNION.values()]
    assert len(set(names)) == len(names)


def test_comb_channels_strict_window():
    np.testing.assert_array_equal(comb_channels(FREQS), COMB)


def test_self_emi_sets_bit12_on_every_row_and_keeps_v3():
    base = _v3()
    out = add_self_emi(base, FREQS, "4", "C", True)
    np.testing.assert_array_equal(
        (out & (1 << 12)) != 0, np.broadcast_to(COMB, base.shape)
    )
    np.testing.assert_array_equal(out & 0x03FF, base)
    np.testing.assert_array_equal(base, _v3())  # input not modified
    # v3's default exclusion (all non-advisory bits) picks up bit 12.
    excluded = (out & ~np.uint16(ADVISORY_VALUE)) != 0
    assert excluded[:, COMB].all()


@pytest.mark.parametrize(
    "input_key,phase,marked",
    [("0", "C", True), ("4", "B", True), ("4", "C", False)],
)
def test_self_emi_out_of_scope_is_noop(input_key, phase, marked):
    base = _v3()
    out = add_self_emi(base, FREQS, input_key, phase, marked)
    np.testing.assert_array_equal(out, base)


def test_calibration_comb_sets_bit13_on_given_rows_only():
    out = add_calibration_comb(_v3(), FREQS, "4", "C", [0, 2])
    expected = np.zeros((3, len(FREQS)), bool)
    expected[[0, 2]] = COMB
    np.testing.assert_array_equal((out & (1 << 13)) != 0, expected)
    assert not (
        add_calibration_comb(_v3(), FREQS, "0", "C", [0]) & (1 << 13)
    ).any()
    with pytest.raises(ValueError, match="rows"):
        add_calibration_comb(_v3(), FREQS, "4", "C", [3])


def test_mask_shape_and_dtype_checked():
    with pytest.raises(ValueError, match="uint16"):
        add_self_emi(_v3().astype(np.uint8), FREQS, "4", "C", True)
    with pytest.raises(ValueError, match="frequency"):
        add_self_emi(_v3(), FREQS[:-1], "4", "C", True)


def test_read_self_emi(tmp_path):
    path = tmp_path / "boxair_emi_presence.jsonl"
    rows = [
        {"provenance": {"product": "boxair_emi_presence"}},
        {"file": "a.h5", "phase": "C", "boxair_emi": True},
        {"file": "b.h5", "phase": "C", "boxair_emi": False},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert read_self_emi(path) == {"a.h5": True, "b.h5": False}
    path.write_text(path.read_text() + json.dumps(rows[1]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_self_emi(path)
    path.write_text(json.dumps({"file": "c.h5", "boxair_emi": 1}) + "\n")
    with pytest.raises(ValueError, match="true or false"):
        read_self_emi(path)


def test_dead_input_is_whole_row_against_its_own_state():
    from eigsep_data.flag_union import (
        DEAD_INPUT_BIT,
        add_dead_input,
        band_power,
        state_reference,
    )

    freqs = np.linspace(60, 75, 61)
    states = np.array(["RFANT"] * 4 + ["RFSP1"] * 2, dtype=object)
    level = np.array([3.0, 3.0, 0.5, 0.0, 0.7, 0.7])  # rows 2, 3 dead
    spectra = level[:, None] * np.ones(freqs.size)
    power = band_power(spectra, freqs, np.full(6, 0.5))
    np.testing.assert_allclose(power, 2 * level)
    reference = state_reference(power, states)
    assert reference == {"RFANT": 6.0, "RFSP1": 1.4}
    flags = np.zeros((6, 5), np.uint16)
    flags[0, 0] = 1
    out = add_dead_input(flags, power, states, reference)
    dead = (out >> DEAD_INPUT_BIT) & 1
    assert dead[[2, 3]].all() and not dead[[0, 1, 4, 5]].any()
    assert out[0, 0] == 1 and FLAG_BITS_UNION["14"]["rfi"] is False
    unknown = add_dead_input(flags, power, states, {})
    np.testing.assert_array_equal(unknown, flags)


def test_recurring_line_is_one_channel_on_sky_rows():
    from eigsep_data.flag_union import RECURRING_LINE_BIT, add_recurring_line

    freqs = np.arange(612, 635) * 250 / 1024  # channel 623 = 152.0996 MHz
    flags = np.zeros((3, freqs.size), np.uint16)
    flags[1] = 1  # a non-sky row
    flags[0, 0] = 1 << 2
    out = add_recurring_line(flags, freqs)
    line = (out >> RECURRING_LINE_BIT) & 1
    assert line[[0, 2]].sum(1).tolist() == [1, 1] and line[1].sum() == 0
    assert line[0, 623 - 612] == 1
    np.testing.assert_array_equal(out & 0x7FFF, flags)
