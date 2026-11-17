"""Add identified exclusion reasons to a flags/v3 mask.

Opt-in. Each function ORs one identified reason into a v3 ``uint16``
mask on its native (row, channel) axes and leaves bits 0-9 unchanged.
Both reasons are box-air's 1.000 MHz comb on input 4 in wiring phase
C, on channels within 0.15 MHz of an integer MHz:

- bit 12, ``identified_self_emi``: every row of a file that the
  ``boxair_emi_presence`` curation marks ``boxair_emi: true``;
- bit 13, ``identified_calibration_comb``: the identified native rows,
  in every switch state.

Bits 10 and 11 stay free because the RFI prototype lineage gives them
meanings (retracted, post-DPSS residual) that some consumers keep. A
consumer that excludes every non-advisory bit, v3's default, excludes
bits 12 and 13 without change.
"""

import json
from pathlib import Path

import numpy as np

from .rfi_supported import FLAG_BITS

UNION_BITS = {
    "12": {"name": "identified_self_emi", "rfi": True},
    "13": {"name": "identified_calibration_comb", "rfi": True},
}
UNION_MEANINGS = {
    "identified_self_emi": (
        "box-air's 1.000 MHz self-EMI: input 4, phase C, channels within "
        "0.15 MHz of an integer MHz, every row of a curated file"
    ),
    "identified_calibration_comb": (
        "calibration comb: input 4, phase C, channels within 0.15 MHz of "
        "an integer MHz, identified native rows in every switch state"
    ),
}
FLAG_BITS_UNION = {**FLAG_BITS, **UNION_BITS}
SELF_EMI_BIT = 12
CALIBRATION_COMB_BIT = 13
COMB_HALFWIDTH_MHZ = 0.15
COMB_INPUT = "4"
COMB_PHASE = "C"


def comb_channels(freqs_mhz, halfwidth=COMB_HALFWIDTH_MHZ):
    """Channels strictly within ``halfwidth`` MHz of an integer MHz."""
    f = np.asarray(freqs_mhz, dtype=float)
    return np.abs(f - np.round(f)) < halfwidth


def read_self_emi(path):
    """Map file name to ``boxair_emi`` from the curation JSONL.

    A provenance-only header line is skipped. Duplicate files and
    non-boolean ``boxair_emi`` values raise.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) == {"provenance"}:
            continue
        name, value = row["file"], row.get("boxair_emi")
        if name in out:
            raise ValueError(f"duplicate curation record for {name}")
        if type(value) is not bool:
            raise ValueError(f"{name}: boxair_emi must be true or false")
        out[name] = value
    return out


def _check(flags, freqs_mhz):
    flags = np.asarray(flags)
    if flags.dtype != np.uint16 or flags.ndim != 2:
        raise ValueError("flags must be a 2-D uint16 v3 mask")
    if flags.shape[1] != len(freqs_mhz):
        raise ValueError("flags and frequency axis differ in length")
    return flags.copy()


def add_self_emi(flags, freqs_mhz, input_key, phase, boxair_emi):
    """Return a copy of ``flags`` with bit 12 on the self-EMI channels.

    No-op unless ``boxair_emi`` is true and the block is input 4 in
    phase C.
    """
    out = _check(flags, freqs_mhz)
    if boxair_emi and str(input_key) == COMB_INPUT and phase == COMB_PHASE:
        out[:, comb_channels(freqs_mhz)] |= np.uint16(1 << SELF_EMI_BIT)
    return out


def add_calibration_comb(flags, freqs_mhz, input_key, phase, rows):
    """Return a copy of ``flags`` with bit 13 on the identified comb rows.

    ``rows`` are native row indices into ``flags``. No-op unless the
    block is input 4 in phase C.
    """
    out = _check(flags, freqs_mhz)
    rows = np.asarray(rows, dtype=int)
    if rows.size and (rows.min() < 0 or rows.max() >= out.shape[0]):
        raise ValueError("comb rows outside the native row axis")
    if str(input_key) == COMB_INPUT and phase == COMB_PHASE:
        sel = np.ix_(rows, np.flatnonzero(comb_channels(freqs_mhz)))
        out[sel] |= np.uint16(1 << CALIBRATION_COMB_BIT)
    return out
