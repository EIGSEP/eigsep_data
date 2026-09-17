"""The product plugin contract: one join, many companions.

A *product* is a derived dataset that lives beside the raw correlator
files and is keyed by them -- RFI flags, a DPSS smooth-band model, a
pointing table, a gain solution. Each one knows how to find its own
payload and how to return it aligned to a set of ``(file, row)`` pairs;
:func:`eigsep_data.bundle.load_bundle` knows nothing about any of them
beyond this interface.

Versions are explicit everywhere. A product is named ``kind@version``
(``"flags@v2"``), never bare, because a version is not a detail: the
flags bitfield is ``uint8`` in v0 and ``uint16`` in v2, so code that
assumes one-byte pixels misbehaves silently on the other. The version a
caller typed is what gets read and what gets recorded in the bundle's
provenance.
"""

import json

import numpy as np

#: Tolerance for calling two frequency axes the same, in MHz. Far below
#: the 0.244 MHz channel width and far above float64 round-trip noise.
FREQ_TOL_MHZ = 1e-6

_REGISTRY = {}


def register(product):
    """
    Add a product to the registry, as a decorator or a call.

    Accepts either an instance or a :class:`Product` subclass; a class
    is instantiated once and the singleton is what the registry serves,
    so a product may cache (a parsed Parquet table, a resolved axis)
    across the files of one load. The class is handed back unchanged so
    ``@register`` above a class definition still binds the class name.
    """
    instance = product() if isinstance(product, type) else product
    _REGISTRY[instance.kind] = instance
    return product


def get(kind):
    """The registered product for *kind*."""
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise KeyError(
            f"unknown product {kind!r}; registered: "
            f"{sorted(_REGISTRY)}"
        ) from None


def registered():
    """Sorted list of registered product kinds."""
    return sorted(_REGISTRY)


def parse_spec(spec):
    """
    Split a ``"kind@version"`` string.

    A bare ``"kind"`` is refused rather than defaulted. Defaulting is
    how a caller ends up with v2's uint16 masks in code written against
    v0's uint8 without one line of the call site changing.
    """
    if not isinstance(spec, str) or "@" not in spec:
        raise ValueError(
            f"product spec must be 'kind@version', got {spec!r}. There is "
            "no default version: flags@v0 is uint8 and flags@v2 is "
            "uint16, and picking for you is how that difference goes "
            "unnoticed."
        )
    kind, _, version = spec.partition("@")
    kind, version = kind.strip(), version.strip()
    if not kind or not version:
        raise ValueError(f"product spec must be 'kind@version', got {spec!r}")
    return kind, version


def axis_fingerprint(freqs):
    """
    A cheap scalar stand-in for a frequency axis.

    Compared once per file instead of comparing 1024 floats: the axis of
    a given product version is fixed by construction, so the only thing
    a per-file check must catch is a payload rebuilt against a different
    band. Length and both endpoints do that; they cannot be changed by a
    rebuild without changing the band.
    """
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size == 0:
        return (0, float("nan"), float("nan"))
    return (int(freqs.size), float(freqs[0]), float(freqs[-1]))


def locate_axis(full, sub, what):
    """
    Where *sub* sits inside *full*, as a ``(start, stop)`` pair.

    *sub* must be a contiguous run of *full* -- the band-restricted
    products in this campaign are exactly that (45-235 MHz out of the
    full 1024 channels), so the alignment is a slice and costs nothing
    per row. Anything else raises: this layer never interpolates and
    never silently truncates, because a product on a genuinely
    different grid is a question for the caller, not something to guess
    at inside a loop.
    """
    full = np.asarray(full, dtype=float)
    sub = np.asarray(sub, dtype=float)
    if sub.size == 0:
        raise ValueError(f"{what}: empty frequency axis")
    if sub.size > full.size:
        raise ValueError(
            f"{what}: has {sub.size} channels, more than the "
            f"{full.size} of the data it must align to"
        )
    start = int(np.searchsorted(full, sub[0] - FREQ_TOL_MHZ))
    stop = start + sub.size
    if stop > full.size or not np.allclose(
        full[start:stop], sub, atol=FREQ_TOL_MHZ, rtol=0
    ):
        raise ValueError(
            f"{what}: frequency axis is not a contiguous slice of the "
            f"data's axis (product spans {sub[0]:.6f}-{sub[-1]:.6f} MHz "
            f"in {sub.size} channels; data spans "
            f"{full[0]:.6f}-{full[-1]:.6f} MHz in {full.size}). Rebuild "
            "the product against the data's grid, or ask for a band "
            "both cover."
        )
    return start, stop


def read_manifest(path):
    """Read a tracked ``manifest.json``, or return ``{}`` if absent.

    A missing manifest is not fatal -- the payload is still readable --
    but it is recorded as such in the bundle's provenance, because an
    unmanifested product has no statement of what made it.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, NotADirectoryError):
        return {}
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


class Product:
    """
    One companion dataset, joined to raw rows by ``(file, row)``.

    Subclasses set :attr:`kind` and implement :meth:`fetch`; the two
    shapes seen so far are a per-(time, channel) cube (flags, smooth
    models) and per-row scalars (the pointing table). A cube product
    also implements :meth:`freqs` so the bundle can resolve its band
    once per version rather than per file.
    """

    #: Registry name, the part before ``@`` in a spec.
    kind = None
    #: True when :meth:`fetch` returns (nrow, nchan) arrays that must be
    #: aligned in frequency; False for per-row columns.
    cube = False

    def root_dir(self, campaign, version):
        """Directory holding this product version's payload."""
        raise NotImplementedError

    def manifest(self, campaign, version):
        """The version's tracked provenance, as a dict."""
        root = self.root_dir(campaign, version)
        return read_manifest(root / "manifest.json")

    def versions(self, campaign):
        """Versions present on disk, sorted."""
        raise NotImplementedError

    def freqs(self, campaign, version):
        """The product's own frequency axis in MHz (cube products)."""
        raise NotImplementedError

    def fetch(self, campaign, version, fname, rows, key, band):
        """
        Arrays for one file's *rows*.

        Parameters
        ----------
        campaign : eigsep_data.bundle.Campaign
        version : str
        fname : str
            Raw correlator filename, the join key.
        rows : ndarray of int
            Within-file integration indices, ascending.
        key : str or None
            Correlator input key these rows were read from, for products
            that are per-input.
        band : slice or None
            Slice into this product's *own* channel axis, resolved once
            per version by the bundle.

        Returns
        -------
        dict of str -> ndarray, or None
            ``None`` means "this file has no payload here", which the
            bundle records as a skip rather than an error.
        """
        raise NotImplementedError
