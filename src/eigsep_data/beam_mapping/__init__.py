"""Beam mapping for the rotating-receiver / transmitter-drone experiment.

Layers, roughly in dependency order:

* :mod:`~eigsep_data.beam_mapping.geometry` -- pointing fusion from motor,
  potentiometer and IMU streams, plus the transmitter heading container.
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
from .geometry import (
    MOTOR_DEG_PER_STEP,
    PointingStreams,
    PointingTable,
    TransmitterGeometry,
    estimate_motor_slip_steps,
    fuse_pointing,
    imu_elevation_deg,
    rotation_matrix,
    simulate_pointing_streams,
    vector_to_spherical,
)
from .mapper import PolarizationBeamMapper
from .rfi import data_space_rfi_mask, monitor_channels, smooth_time_flags
from .tx_model import (
    HFSSBeamSet,
    correlator_waterfall,
    fit_ground_position,
    ground_heading,
    recover_sampled_beam,
    simulate_correlator_waterfall,
    simulate_hfss,
    simulate_hfss_coupling,
)

# diagnostics and fit pull in matplotlib / JAX; import them explicitly as
# eigsep_data.beam_mapping.diagnostics / .fit rather than re-exporting here.
