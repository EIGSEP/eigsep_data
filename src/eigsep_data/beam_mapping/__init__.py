"""Beam mapping for the rotating-receiver / transmitter-drone experiment.

Layers, roughly in dependency order:

* :mod:`~eigsep_data.beam_mapping.beam_rotations` -- the one pointing
  convention: body -> ENU is ``Rz(psi) Rx(el) Rz(az)``.
* :mod:`~eigsep_data.beam_mapping.geometry` -- pointing fusion from motor,
  potentiometer and IMU streams.
* :mod:`~eigsep_data.beam_mapping.tx_coupling` -- transmitter geometry and
  the ``|E* . e_tx|^2`` coupling through an HFSS-style beam.
* :mod:`~eigsep_data.beam_mapping.tx_background` -- the background under the
  transmitter comb teeth (local DPSS fit).
* :mod:`~eigsep_data.beam_mapping.tx_teeth` -- model-free tooth selection
  (isolation and neighbour coherence).
* :mod:`~eigsep_data.beam_mapping.mapper` -- non-negative theta/phi gain
  recovery from alternating transmitter arms.
* :mod:`~eigsep_data.beam_mapping.tx_model` -- HFSS-backed forward model of
  the transmitter coupling and correlator waterfall.
* :mod:`~eigsep_data.beam_mapping.basis` -- low-rank template banks (PCA
  eigen-beams) packed as an ``HFSSBeamSet`` for linear fitting.
* :mod:`~eigsep_data.beam_mapping.rfi` -- data-space RFI flagging using
  out-of-band monitor channels.
* :mod:`~eigsep_data.beam_mapping.fit` -- transmitter position and
  polarization fitting (JAX).
* :mod:`~eigsep_data.beam_mapping.diagnostics` -- v007 campaign loading,
  joint fits and standard diagnostic figures.

This module re-exports the common entry points; it contains no logic.
"""

from .basis import BeamPCA, compute_beam_pca
from .beam_rotations import (
    body_to_enu,
    enu_to_body,
    mount_rotation,
    spherical_basis,
    vector_to_spherical,
)
from .geometry import (
    MOTOR_DEG_PER_STEP,
    PointingStreams,
    PointingTable,
    estimate_motor_slip_steps,
    fuse_pointing,
    imu_elevation_deg,
    simulate_pointing_streams,
)
from .mapper import PolarizationBeamMapper
from .rfi import data_space_rfi_mask, monitor_channels, smooth_time_flags
from .tx_background import tooth_background
from .tx_coupling import (
    TransmitterGeometry,
    ground_heading,
    heading_between,
    interpolate_fields,
    normalize_fields,
    tooth_gains,
    transmitter_coupling,
    transmitter_power,
)
from .tx_model import (
    HFSSBeamSet,
    correlator_waterfall,
    fit_ground_position,
    recover_sampled_beam,
    simulate_correlator_waterfall,
    simulate_hfss,
    simulate_hfss_coupling,
)
from .tx_teeth import isolation, neighbour_coherence, select_teeth, tooth_arms

# diagnostics and fit pull in matplotlib / JAX; import them explicitly as
# eigsep_data.beam_mapping.diagnostics / .fit rather than re-exporting here.
