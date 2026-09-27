"""Bases for an empirical beam: angular (spherical harmonics) x spectral.

A beam E-field is modelled as

    E_c(nu, n-hat) = sum_k  S_k(nu)  sum_lm  C[k, c, lm]  Y_lm(n-hat),

with ``c`` the three body-frame Cartesian components, ``Y_lm`` real spherical
harmonics up to ``lmax``, and ``S_k`` a small spectral basis
(:class:`eigsep_base.spectral_basis.SpectralBasis`). The complex coefficients
``C`` are laid out as ``(nmodes, 3 * nlm)``, component-major.

The fields are complex because the transmitter coupling ``|E* . e_tx|^2``
depends on how the components combine before squaring. A power-only basis, like
``eigsep_sim``'s, is enough for an unpolarized sky but not here.

Spectral bases:

* :func:`pca_basis`: left singular vectors of the (normalized) HFSS fields
  across frequency, with the fewest modes that reconstruct every frequency to
  ``max_relative_error``. Linearly interpolated between HFSS frequencies.
* :class:`DPSSSpectralBasis`: band-limited in delay. DPSS sequences on a
  uniform grid (the comb tooth spacing, padded by the band on each side),
  continued to any frequency by Nystrom sinc interpolation, and orthonormalized
  on the HFSS frequencies. The delay half-width is the smallest candidate whose
  span reconstructs the HFSS fields to ``max_relative_error``. This is a
  smoothness scale, not a measured path delay.

Public API
----------
real_spherical_harmonics
nlm
fit_spherical_harmonics
select_lmax
pca_basis
DPSSSpectralBasis
initial_coefficients
"""

import numpy as np
from scipy.signal.windows import dpss
from scipy.special import sph_harm_y

from eigsep_base.spectral_basis import SpectralBasis


def nlm(lmax):
    """Number of real spherical harmonics up to and including ``lmax``."""
    return (lmax + 1) ** 2


def real_spherical_harmonics(theta, phi, lmax):
    """Orthonormal real spherical harmonics, ``(n, nlm)``, ordered by (l, m = -l..l).

    ``Y_l0 = Re Y_l0``; for m > 0, ``sqrt(2) Re Y_lm``; for m < 0,
    ``sqrt(2) Im Y_l|m|``.
    """
    theta, phi = np.asarray(theta, float), np.asarray(phi, float)
    columns = []
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            y = sph_harm_y(ell, abs(m), theta, phi)
            columns.append(y.real if m == 0 else np.sqrt(2) * (y.imag if m < 0 else y.real))
    return np.column_stack(columns)


def fit_spherical_harmonics(fields, lmax):
    """Least-squares SH coefficients of HEALPix fields.

    ``fields``: (nfreq, 3, npix). Returns ``(coefficients, relative_error)``:
    coefficients (nfreq, 3 * nlm) component-major, and each frequency's relative
    rms field error of the reconstruction.
    """
    import healpy as hp

    fields = np.asarray(fields)
    nfreq, ncomp, npix = fields.shape
    theta, phi = hp.pix2ang(hp.npix2nside(npix), np.arange(npix))
    y = real_spherical_harmonics(theta, phi, lmax)
    target = fields.transpose(2, 0, 1).reshape(npix, -1)                    # (npix, nfreq * 3)
    c = np.linalg.solve(y.T @ y, y.T @ target)                                # (nlm, nfreq * 3)
    pred = (y @ c).reshape(npix, nfreq, ncomp).transpose(1, 2, 0)
    error = (np.linalg.norm((fields - pred).reshape(nfreq, -1), axis=1)
             / np.linalg.norm(fields.reshape(nfreq, -1), axis=1))
    coefficients = c.reshape(nlm(lmax), nfreq, ncomp).transpose(1, 2, 0).reshape(nfreq, -1)
    return coefficients, error


def select_lmax(fields, max_relative_error=0.01, candidates=(2, 4, 6, 8, 10, 12)):
    """Smallest ``lmax`` in ``candidates`` whose SH fit reconstructs every frequency to
    ``max_relative_error``. Returns ``(lmax, coefficients, records)``."""
    records, fits = [], {}
    for ell in candidates:
        c, err = fit_spherical_harmonics(fields, ell)
        fits[ell] = c
        records.append(dict(lmax=ell, worst_relative_field_rms=float(err.max()),
                            median_relative_field_rms=float(np.median(err))))
        if err.max() <= max_relative_error:
            return ell, c, records
    raise ValueError(f'no lmax in {candidates} reaches {max_relative_error}')


def pca_basis(freqs_mhz, fields, max_relative_error=0.01, max_modes=10):
    """PCA spectral basis of ``fields`` (nfreq, 3, npix), a :class:`SpectralBasis`
    with ``kind = 'pca'``: the fewest left singular vectors across frequency that
    reconstruct every frequency to ``max_relative_error``."""
    fields = np.asarray(fields)
    samples = fields.reshape(fields.shape[0], -1).T                           # (3 * npix, nfreq)
    basis = SpectralBasis.from_samples(freqs_mhz, samples, max_relative_error=max_relative_error,
                                       max_modes=max_modes)
    basis.kind = 'pca'
    return basis


class DPSSSpectralBasis(SpectralBasis):
    """Delay-limited spectral basis continued to any frequency (see the module docstring).

    Construct with :meth:`from_fields`. ``A`` is the basis at the HFSS frequencies
    (orthonormal columns); :meth:`evaluate` continues it to any frequency.
    """

    kind = 'dpss'

    def __init__(self, A, freqs, grid, modes, concentrations, halfwidth, r, delay_halfwidth_ns,
                 records=None):
        super().__init__(A, freqs=freqs, records=records)
        self.grid, self.modes, self.concentrations = grid, modes, concentrations
        self.halfwidth, self.r, self.delay_halfwidth_ns = halfwidth, r, delay_halfwidth_ns

    @staticmethod
    def _continuation(f, grid, halfwidth, modes, concentrations):
        df = grid[1] - grid[0]
        kernel = 2 * halfwidth * np.sinc(2 * halfwidth * (np.asarray(f)[:, None] - grid[None, :]) / df)
        return kernel @ (modes.T / concentrations)

    def evaluate(self, freqs, fill_value=None):
        """The basis at any frequencies (band-limited continuation; no edge limit)."""
        raw = self._continuation(freqs, self.grid, self.halfwidth, self.modes, self.concentrations)
        return np.linalg.solve(self.r.T, raw.T).T

    @classmethod
    def from_fields(cls, freqs_mhz, fields, tooth_freqs_mhz, max_relative_error=0.01,
                    delays_ns=(1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20), concentration_min=0.9):
        """Choose the delay half-width and build the basis.

        The grid spacing is the smallest tooth spacing, and the grid is padded by
        the number of teeth on each side so the band edges are not forced to
        taper. Only modes with concentration ``>= concentration_min`` are used.
        """
        freqs_mhz = np.asarray(freqs_mhz, float)
        tooth_freqs_mhz = np.asarray(tooth_freqs_mhz, float)
        fields = np.asarray(fields)
        target = fields.reshape(len(freqs_mhz), -1)
        denominator = np.linalg.norm(target, axis=1)
        df = float(np.diff(tooth_freqs_mhz).min())
        pad = len(tooth_freqs_mhz)
        grid = tooth_freqs_mhz[0] + np.arange(-pad, len(tooth_freqs_mhz) + pad) * df
        records = []
        for delay in delays_ns:
            w = delay * 1e-3 * df
            nw = len(grid) * w
            modes, eigen = dpss(len(grid), nw, Kmax=min(len(grid), int(np.ceil(2 * nw)) + 4),
                                return_ratios=True)
            keep = eigen >= concentration_min
            if not np.any(keep):
                continue
            raw = cls._continuation(freqs_mhz, grid, w, modes[keep], eigen[keep])
            q, r = np.linalg.qr(raw)
            error = np.linalg.norm(target - q @ (q.T @ target), axis=1) / denominator
            records.append(dict(delay_half_width_ns=delay, time_bandwidth=nw, modes=int(keep.sum()),
                                minimum_concentration=float(eigen[keep].min()),
                                worst_relative_field_rms=float(error.max()),
                                median_relative_field_rms=float(np.median(error))))
            if error.max() <= max_relative_error:
                return cls(q, freqs_mhz, grid, modes[keep], eigen[keep], w, r, delay, records)
        raise ValueError(f'no delay half-width in {delays_ns} reaches {max_relative_error}')

    def save(self, path):
        np.savez(path, kind=np.array(self.kind), native_freqs=self.freqs, native_modes=self.A,
                 grid_mhz=self.grid, dpss_modes=self.modes, concentrations=self.concentrations,
                 half_bandwidth_cycles_per_sample=self.halfwidth, qr_r=self.r,
                 delay_halfwidth_ns=self.delay_halfwidth_ns)


def initial_coefficients(basis, sh_coefficients):
    """Coefficients ``C`` (nmodes, 3 * nlm) that best reproduce per-frequency SH
    coefficients (nfreq, 3 * nlm) given at the basis's own frequencies ``basis.freqs``."""
    return np.linalg.lstsq(basis.A, np.asarray(sh_coefficients), rcond=None)[0]
