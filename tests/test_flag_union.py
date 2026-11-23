import json

import numpy as np
import pytest

import pandas as pd

from eigsep_data.flag_union import (
    add_calibration_comb,
    add_release_bits,
    add_self_emi,
    comb_channels,
    read_self_emi,
)
from eigsep_data.rfi_supported import (
    ADVISORY_VALUE,
    FLAG_BITS,
    FLAG_BITS_UNION,
    RFIConfig,
    RFIResult,
)

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


def test_read_calibration_comb_expands_inclusive_ranges(tmp_path):
    from eigsep_data.flag_union import read_calibration_comb

    path = tmp_path / "comb.jsonl"
    records = [
        {"provenance": {"source": "test"}},
        {"file": "a.h5", "input": "4", "rows": [[0, 2], [5, 5]]},
        {"file": "b.h5", "input": 4, "rows": [[239, 239]]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records))
    rows = read_calibration_comb(path)
    np.testing.assert_array_equal(rows["a.h5"], [0, 1, 2, 5])
    np.testing.assert_array_equal(rows["b.h5"], [239])
    for bad in (
        {"file": "a.h5", "input": "4", "rows": [[1, 1]]},
        {"file": "c.h5", "input": "0", "rows": [[1, 1]]},
        {"file": "c.h5", "input": "4", "rows": [[3, 2]]},
    ):
        path.write_text("\n".join(json.dumps(r) for r in records[1:2] + [bad]))
        with pytest.raises(ValueError):
            read_calibration_comb(path)


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


def test_release_bits_follow_files_native_rows_and_phase():
    freqs = np.array([151.9, 152.1, 153.0, 153.5])  # line 623, comb 153
    flags = np.zeros((6, 4), np.uint16)
    flags[5] = 1  # a non-sky row
    meta = pd.DataFrame(
        {
            "file": ["a"] * 3 + ["b"] * 3,
            "row": [2, 0, 1, 0, 1, 2],  # time order is not raw order
            "input_key": "4",
            "filter_phase": ["C"] * 3 + ["B"] * 3,
        }
    )
    power = np.array([1.0, 1.0, 0.1, 1.0, 0.1, 1.0])
    states = np.array(["RFANT"] * 6, object)
    out = add_release_bits(
        flags, meta, freqs, power, states,
        self_emi={"a": True, "b": True},
        comb_rows={"a": np.array([0]), "b": np.array([0])},
        reference={"C": {"RFANT": 1.0}},
    )
    bit = lambda b: (out >> b) & 1  # noqa: E731
    np.testing.assert_array_equal(bit(12)[:, 2], [1, 1, 1, 0, 0, 0])
    # bit 13 on a's native row 0 (position 1); phase B is out of scope
    np.testing.assert_array_equal(bit(13)[:, 2], [0, 1, 0, 0, 0, 0])
    # bit 14 only where phase C has a reference
    np.testing.assert_array_equal(bit(14)[:, 0], [0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(bit(15)[:, 1], [1, 1, 1, 1, 1, 0])
    assert not bit(15)[:, [0, 2, 3]].any()
    with pytest.raises(KeyError, match="self-EMI"):
        add_release_bits(
            flags, meta, freqs, power, states, self_emi={"a": False},
            comb_rows={}, reference={},
        )


def test_default_masks_exclude_release_bits():
    flags = np.array([[1 << 12, 1 << 14, 1 << 15, 1 << 9]], np.uint16)
    z = np.zeros(flags.shape)
    result = RFIResult(
        flags, z, z, z, z, z, {}, z, z, z, z, np.arange(4.0),
        np.arange(1.0), None, RFIConfig(), {},
    )
    np.testing.assert_array_equal(result.mask, [[1, 1, 1, 0]])
    np.testing.assert_array_equal(result.rfi_mask, [[1, 0, 1, 0]])
