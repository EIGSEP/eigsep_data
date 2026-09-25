import healpy as hp
import numpy as np
import pytest

from eigsep_data.beam_mapping.beam_basis import (
    DPSSSpectralBasis,
    fit_spherical_harmonics,
    initial_coefficients,
    nlm,
    pca_basis,
    real_spherical_harmonics,
    select_lmax,
)

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)
THETA, PHI = hp.pix2ang(NSIDE, np.arange(NPIX))
FREQS = np.arange(130.0, 215.0, 3.90625)


def test_real_harmonics_are_orthonormal_on_the_sphere():
    y = real_spherical_harmonics(THETA, PHI, 4)
    assert y.shape == (NPIX, nlm(4))
    gram = y.T @ y * (4 * np.pi / NPIX)
    np.testing.assert_allclose(gram, np.eye(nlm(4)), atol=5e-3)


def synthetic_fields(lmax=4, seed=0):
    """(nfreq, 3, npix) complex fields that are exactly band-limited to lmax and
    vary smoothly (two spectral functions) with frequency."""
    rng = np.random.default_rng(seed)
    y = real_spherical_harmonics(THETA, PHI, lmax)
    c = rng.normal(size=(2, 3, nlm(lmax))) + 1j * rng.normal(size=(2, 3, nlm(lmax)))
    x = (FREQS - 170.0) / 40.0
    spectral = np.stack([np.ones_like(x), x + 0.3 * x ** 2], axis=1)            # (nfreq, 2)
    return np.einsum('fk,kcl,pl->fcp', spectral, c, y)


def test_harmonic_fit_and_lmax_selection():
    fields = synthetic_fields(lmax=4)
    coeffs, err = fit_spherical_harmonics(fields, 4)
    assert coeffs.shape == (len(FREQS), 3 * nlm(4)) and err.max() < 1e-10
    lmax, _, records = select_lmax(fields, max_relative_error=1e-6)
    assert lmax == 4 and records[0]['lmax'] == 2 and records[0]['worst_relative_field_rms'] > 1e-6
    with pytest.raises(ValueError):
        select_lmax(fields, max_relative_error=1e-6, candidates=(2,))


def test_pca_basis_and_initial_coefficients_reproduce_the_fields():
    fields = synthetic_fields()
    basis = pca_basis(FREQS, fields, max_relative_error=1e-8)
    assert basis.nmodes == 2 and basis.kind == 'pca'
    _, sh, _ = select_lmax(fields, max_relative_error=1e-6)
    c0 = initial_coefficients(basis, sh)
    np.testing.assert_allclose(basis.A @ c0, sh, atol=1e-8 * np.abs(sh).max())


def test_dpss_basis_is_orthonormal_and_continues_smoothly():
    fields = synthetic_fields()
    # As in use: the teeth span the HFSS frequencies (to within one HFSS sample).
    teeth = np.arange(FREQS[0] + 2.0, FREQS[-1] - 2.0, 1.953125)
    basis = DPSSSpectralBasis.from_fields(FREQS, fields, teeth, max_relative_error=1e-3)
    np.testing.assert_allclose(basis.A.T @ basis.A, np.eye(basis.nmodes), atol=1e-10)
    np.testing.assert_allclose(basis.evaluate(FREQS), basis.A, atol=1e-10)
    # A smooth spectral function known only on the HFSS grid is continued to the teeth.
    x = lambda f: (f - 170.0) / 40.0
    g = 1 + x(FREQS) + 0.3 * x(FREQS) ** 2
    coeff = basis.A.T @ g
    np.testing.assert_allclose(basis.evaluate(teeth) @ coeff, 1 + x(teeth) + 0.3 * x(teeth) ** 2, atol=1e-3)
