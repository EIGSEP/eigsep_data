"""Low-order PCA/POD spectral basis for the HFSS beam.

Reduces the frequency axis of a :class:`~eigsep_data.beam_mapping.tx_model.HFSSBeamSet` to a
small number of orthogonal complex-vector eigen-beams via an
(uncentered) SVD of the raw HFSS field across frequency. Component 0
captures the dominant, roughly-frequency-independent beam shape;
higher components capture the frequency-dependent corrections. Any
individual HFSS frequency slice is reconstructed as
``beam_cart[f] ~= sum_k loadings[f, k] * components.beam_cart[k]``.

This step is deliberately geometry-independent -- it only needs the
raw HFSS data, not the transmitter heading/polarization -- so it is
computed once and reused. Projecting the eigen-beams through a
specific transmitter geometry/arm at fit time reuses
:func:`~eigsep_data.beam_mapping.tx_model.simulate_hfss_coupling` unchanged: each eigen-beam
is just handed to it as if it were one more frequency slice.
Public API
----------
BeamPCA
compute_beam_pca
"""

from dataclasses import dataclass

import numpy as np

from .tx_model import HFSSBeamSet


@dataclass
class BeamPCA:
    components: HFSSBeamSet
    loadings: np.ndarray
    freqs_mhz: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray


def compute_beam_pca(beam, n_components=4):
    """Reduce ``beam``'s frequency axis to ``n_components`` eigen-beams.

    ``components`` is an :class:`HFSSBeamSet` whose "frequency" axis is
    really the component index (``freqs_mhz`` is just ``0..K-1`` and is
    not meaningful as a physical frequency). ``loadings[f, k]`` are the
    complex coefficients such that ``beam.beam_cart[f] ~=
    sum_k loadings[f, k] * components.beam_cart[k]``.
    """
    beam_cart = np.asarray(beam.beam_cart)
    nfreq, ncomp_field, npix = beam_cart.shape
    matrix = beam_cart.reshape(nfreq, ncomp_field * npix)
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    k = min(int(n_components), s.size)
    components = HFSSBeamSet(
        beam_cart=vh[:k].reshape(k, ncomp_field, npix),
        gain_th=np.zeros((k, npix)),
        gain_ph=np.zeros((k, npix)),
        freqs_mhz=np.arange(k, dtype=float),
    )
    loadings = u[:, :k] * s[None, :k]
    variance = s ** 2
    total_variance = float(np.sum(variance))
    explained = variance[:k] / total_variance if total_variance > 0 else np.zeros(k)
    return BeamPCA(
        components=components,
        loadings=loadings,
        freqs_mhz=np.asarray(beam.freqs_mhz, float),
        singular_values=s,
        explained_variance_ratio=explained,
    )
