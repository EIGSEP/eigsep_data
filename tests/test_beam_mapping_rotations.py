import numpy as np
import pytest

import eigsep_base.rotations as base
from eigsep_data.beam_mapping import beam_rotations


def test_beam_rotations_is_the_eigsep_base_convention():
    # One implementation: beam_mapping re-exports eigsep_base.rotations (tested there).
    for name in beam_rotations.__all__:
        assert getattr(beam_rotations, name) is getattr(base, name)


def test_marjum_sweep_plane_through_the_beam_mapping_entry_point():
    b90 = beam_rotations.mount_rotation(0, 90, 142.164) @ [0, 0, 1]
    assert np.degrees(np.arctan2(b90[0], b90[1])) % 360 == pytest.approx(37.836, abs=1e-9)
