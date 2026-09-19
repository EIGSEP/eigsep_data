"""Raw spectra plus their companion products, aligned on one axis pair.

:func:`load_bundle` answers the question every analysis in this campaign
has re-implemented by hand: *give me this stretch of data for this
antenna, with its flags and its smooth-band model and its pointing,
already lined up in time and frequency.* Four independent
implementations of that join existed before this module; see
``DATASET_LOADER_PLAN.md``.

What it does not do is choose for you. Every product is named with an
explicit version, the frequency axes are asserted to match rather than
regridded onto each other, and a file with a missing companion is
reported rather than quietly dropped.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import products as _products
from .products.base import axis_fingerprint, locate_axis


class Campaign:
    """
    A campaign directory: ``data/`` beside ``flags/``, ``derived/``,
    ``curation/``.

    Anchored on a path the caller passes or, by default, on the parent
    of the index's ``data_dir`` -- computed, never a hardcoded literal,
    per ``PATH_PORTABILITY_PROPOSAL.md``. ``EIGSEP_CAMPAIGN_ROOT``
    overrides it for a dataset mounted somewhere else.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()

    @classmethod
    def for_index(cls, index, root=None):
        import os

        if root is None:
            root = os.environ.get("EIGSEP_CAMPAIGN_ROOT")
        if root is None:
            root = Path(index.data_dir).resolve().parent
        return cls(root)

    def __repr__(self):
        return f"Campaign({str(self.root)!r})"


@dataclass
class Bundle:
    """Aligned raw data and companions for one antenna (or one antenna
    pair's cross correlation) over one window."""

    #: Frequency axis shared by every array here, in MHz.
    freqs_mhz: np.ndarray
    #: ``time_best`` per row, Unix seconds, ascending.
    t: np.ndarray
    #: Raw counts, ``(nrow, nchan)``.
    data: np.ndarray
    #: One row per entry in :attr:`t`, carrying ``file``, ``row`` and
    #: the index's metadata columns.
    meta: pd.DataFrame
    #: ``{"kind": {"array_name": ndarray}}`` for every product asked for.
    products: dict = field(default_factory=dict)
    #: What was read, at which version, and what was skipped.
    provenance: dict = field(default_factory=dict)

    @property
    def nrows(self):
        return self.data.shape[0]

    def _one(self, kind, name):
        if kind not in self.products:
            raise KeyError(
                f"{kind!r} was not requested; pass "
                f"products=['{kind}@<version>'] to load_bundle"
            )
        return self.products[kind][name]

    @property
    def flags(self):
        """The flag bitfield, dtype exactly as stored (see the version)."""
        return self._one("flags", "mask")

    @property
    def smooth_model(self):
        return self._one("smooth_model", "model")

    @property
    def residual(self):
        """``data - model``, recomputed here.

        Never the companion's stored ``residual``: ``hera_filters``
        zeroes that at flagged pixels, which would hide exactly the comb
        and RFI spikes a residual waterfall exists to show.
        """
        return self.data - self.smooth_model

    @property
    def pointing(self):
        """Per-row pointing columns as a DataFrame."""
        return pd.DataFrame(self.products["pointing"], index=self.meta.index)

    def summary(self):
        p = self.provenance
        lines = [
            f"{self.nrows} rows x {self.freqs_mhz.size} channels "
            f"({self.freqs_mhz[0]:.3f}-{self.freqs_mhz[-1]:.3f} MHz)",
            f"antenna {p.get('antenna')!r} via keys {p.get('keys')}",
            f"{len(p.get('used_files', []))} files used, "
            f"{len(p.get('skipped_files', []))} skipped",
        ]
        for kind, info in p.get("products", {}).items():
            line = f"  {kind}@{info['version']}"
            if info.get("skipped"):
                line += f" -- no payload for {len(info['skipped'])} files"
            lines.append(line)
        return "\n".join(lines)


def _header_item(header, name):
    """One header field, whether the writer stored it as an attr or a
    dataset.

    Both forms are real: ``eigsep_base.io.write_hdf5`` puts a string
    header field in ``header.attrs``, while the deployment-5 files on
    disk carry ``header/input_to_ant`` as a JSON string *dataset*. A
    reader that knows only one of them silently finds nothing on half
    the campaign and falls back to guessing the wiring.
    """
    if header is None:
        return None
    if name in header.attrs:
        return header.attrs[name]
    if name in header:
        return header[name][()]
    return None


def _as_json(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _input_to_ant(path):
    """``{key: antenna}`` for one raw file.

    Prefers the producer-written ``input_to_ant`` and falls back to
    deriving it from the wiring and the ADC mux, which is what
    ``eigsep_base.io.corr_pair_labels`` does for files that predate the
    field.
    """
    import h5py

    from eigsep_base.io import effective_input_to_ant

    with h5py.File(path, "r") as h:
        header = h.get("header")
        declared = _as_json(_header_item(header, "input_to_ant"))
        if declared:
            return {str(k): str(v) for k, v in declared.items()}
        wiring = _as_json(_header_item(header, "wiring")) or {}
        mux = _header_item(header, "adc_mux_sel")
        if mux is None:
            mux = h.attrs.get("adc_mux_sel", 0)
        return effective_input_to_ant(wiring, int(mux))


def _resolve_key(path, antenna, available):
    """
    The input key carrying *antenna* in this file, or ``None``.

    The ADC mux copies an even input's antenna onto the odd input above
    it, so one antenna can name two keys; the even one is the wired
    source and is preferred. Which input an antenna sits on is genuinely
    per-file in this campaign -- box-gnd is input 0 on 2026-07-17 and
    input 2 on 07-13 -- so this is resolved from the file's own header
    every time, never from a hardcoded table.
    """
    mapping = _input_to_ant(path)
    candidates = [
        k for k, ant in mapping.items() if ant == antenna and k in available
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda k: (int(k) % 2, int(k)))[0]


#: ``antenna=`` shorthand for the one cross this campaign is built
#: around. Order matters: the first antenna is unconjugated.
CROSS_ALIASES = {"cross": ("box-gnd", "box-air")}


def _resolve_cross(path, pair, available):
    """
    The cross key carrying *pair* in this file, and whether to conjugate
    it, or ``None``.

    A cross key is two input digits, lower first; nominally
    ``V_lo * conj(V_hi)``, though that sign convention is not verified
    against the gateware here. Which antenna is on the lower input is
    per-file -- box-air is input 0 and box-gnd input 2 on 2026-07-12,
    the other way round on 07-17 -- so the stored key alone does not
    say which way round the product is. The returned flag is True when
    *pair*'s first antenna sits on the higher input, i.e. when the
    stored product must be conjugated so that every file reads in the
    same orientation, *pair*[0] in the unconjugated slot.

    Wired (even) inputs are preferred over their mux copies, as in
    :func:`_resolve_key`.
    """
    a, b = pair
    mapping = _input_to_ant(path)

    def inputs(ant):
        ks = [k for k, v in mapping.items() if v == ant]
        return sorted(ks, key=lambda k: (int(k) % 2, int(k)))

    for ia in inputs(a):
        for ib in inputs(b):
            if ia == ib:
                continue
            lo, hi = sorted((ia, ib), key=int)
            if lo + hi in available:
                return lo + hi, int(ia) > int(ib)
    return None


def _as_pair(antenna):
    """``(ant_a, ant_b)`` if *antenna* names a cross, else ``None``."""
    if isinstance(antenna, str):
        return CROSS_ALIASES.get(antenna)
    pair = tuple(antenna)
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError(
            f"a cross antenna= must name two different antennas, "
            f"got {antenna!r}"
        )
    return pair


def _band_slice(freqs, band_mhz):
    if band_mhz is None:
        return 0, int(freqs.size)
    lo, hi = band_mhz
    keep = np.nonzero((freqs >= lo) & (freqs <= hi))[0]
    if keep.size == 0:
        raise ValueError(
            f"band_mhz={band_mhz} selects no channels of the data's "
            f"{freqs[0]:.3f}-{freqs[-1]:.3f} MHz axis"
        )
    return int(keep[0]), int(keep[-1]) + 1


def load_bundle(
    selection,
    antenna=None,
    key=None,
    products=(),
    band_mhz=None,
    root=None,
    missing="skip",
):
    """
    Read *selection*'s rows for one antenna, with companion products.

    Parameters
    ----------
    selection : eigsep_data.index.Selection
        Which integrations to read. A file range is expressible here
        (``index.select(files=(first, last))``) without being the only
        vocabulary: time windows and metadata filters work too, which
        filename ranges cannot express and which matter because corr
        filenames are file *close* times.
    antenna : str or (str, str), optional
        Physical antenna name (``"box-gnd"``, ``"box-air"``), resolved
        to an input key per file from that file's own header. Exactly
        one of *antenna* or *key* is required. A pair of names, or
        ``"cross"`` for ``("box-gnd", "box-air")``, selects their cross
        correlation instead, resolved to a cross key (``"04"``,
        ``"02"``, ...) per file and conjugated where needed so that
        every row has the same orientation, nominally ``V_a * conj(V_b)``. ``meta.conjugated``
        says which rows were flipped. Companion products are stored
        per input, so a cross bundle will usually report them skipped.
    key : str, optional
        A literal correlator input key, when you mean one input rather
        than one antenna.
    products : sequence of str
        ``"kind@version"`` specs, e.g. ``["flags@v2",
        "smooth_model@v0"]``. There is no default version for any
        product.
    band_mhz : (lo, hi), optional
        Restrict to this band. The result is the intersection of it and
        every product's own band, always a contiguous slice.
    root : path-like, optional
        Campaign root. Defaults to the index's ``data_dir`` parent.
    missing : {"skip", "raise"}
        What to do with a file that has no payload for a requested
        product, or no key for the requested antenna.

    Returns
    -------
    Bundle
    """
    if (antenna is None) == (key is None):
        raise ValueError("pass exactly one of antenna= or key=")
    if missing not in ("skip", "raise"):
        raise ValueError("missing must be 'skip' or 'raise'")

    campaign = Campaign.for_index(selection.index, root)
    specs = [_products.parse_spec(s) for s in products]
    for kind, version in specs:
        product = _products.get(kind)
        check = getattr(product, "check_version", None)
        if check is not None:
            check(campaign, version)

    data_dir = Path(selection.index.data_dir)
    meta = selection.meta.reset_index(drop=True)
    if len(meta) == 0:
        raise ValueError(
            "Selection contains no integrations; nothing to load."
        )

    # --- which key carries the antenna, per file -------------------
    pair = _as_pair(antenna) if antenna is not None else None
    keys_by_file = {}
    conj_by_file = {}
    no_key = []
    for fname in meta.file.unique():
        conj_by_file[fname] = False
        if key is not None:
            keys_by_file[fname] = key
            continue
        available = set(
            str(meta.loc[meta.file == fname, "data_keys"].iloc[0]).split(",")
        )
        if pair is not None:
            got = _resolve_cross(data_dir / fname, pair, available)
            resolved, conj = got if got is not None else (None, False)
            conj_by_file[fname] = conj
        else:
            resolved = _resolve_key(data_dir / fname, antenna, available)
        if resolved is None:
            no_key.append(fname)
        else:
            keys_by_file[fname] = resolved
    if no_key and missing == "raise":
        raise ValueError(
            f"{len(no_key)} selected files have no live input for "
            f"antenna {antenna!r} (first: {no_key[0]}). Pass "
            "missing='skip' to drop them."
        )

    kept = meta[meta.file.isin(keys_by_file)]
    if kept.empty:
        raise ValueError(
            f"no selected file has a live input for antenna {antenna!r}; "
            f"{len(no_key)} files scanned"
        )

    # --- raw spectra, one load per distinct key --------------------
    # One antenna can sit on different inputs in different campaign
    # phases, so the selection is split by resolved key, each part read
    # with the key it actually has, and the parts re-sorted into one
    # time-ordered block.
    # A cross key is further split by orientation: the same "04" can be
    # gnd x air* in one phase and air x gnd* in another.
    blocks, metas = [], []
    freqs_full = None
    groups = sorted({(k, conj_by_file[f]) for f, k in keys_by_file.items()})
    for this_key, conj in groups:
        files = [
            f
            for f, k in keys_by_file.items()
            if k == this_key and conj_by_file[f] == conj
        ]
        sub = selection.select(files=sorted(files))
        loaded = sub.load(keys=[this_key])
        block = np.asarray(loaded.data[this_key])
        blocks.append(np.conj(block) if conj else block)
        metas.append(loaded.meta.assign(input_key=this_key, conjugated=conj))
        freqs_full = np.asarray(loaded.freq, dtype=float)

    raw = np.concatenate(blocks, axis=0)
    rows_meta = pd.concat(metas, ignore_index=True)
    order = np.argsort(
        rows_meta.time_best.to_numpy(dtype=float), kind="stable"
    )
    raw = raw[order]
    rows_meta = rows_meta.iloc[order].reset_index(drop=True)

    # --- one frequency axis, resolved once per product version -----
    lo, hi = _band_slice(freqs_full, band_mhz)
    spans = {}
    absent = {}
    for kind, version in specs:
        product = _products.get(kind)
        if not product.cube:
            continue
        try:
            pf = np.asarray(product.freqs(campaign, version), dtype=float)
        except FileNotFoundError as e:
            # A version with no payload at all on disk. Under
            # missing="skip" that is the same kind of event as one file
            # lacking a companion -- reported, not fatal -- and it must
            # not take the whole load down before a single row is read.
            if missing == "raise":
                raise
            absent[kind] = str(e)
            continue
        start, stop = locate_axis(freqs_full, pf, f"{kind}@{version}")
        spans[kind] = (start, stop, axis_fingerprint(pf))
        lo, hi = max(lo, start), min(hi, stop)
    if lo >= hi:
        raise ValueError(
            "the requested band and the products' own bands do not "
            f"overlap (bands: {[(k, v[0], v[1]) for k, v in spans.items()]})"
        )
    freqs_mhz = freqs_full[lo:hi]
    data = raw[:, lo:hi]

    # --- companions ------------------------------------------------
    out = {}
    prov_products = {}
    for kind, version in specs:
        product = _products.get(kind)
        if kind in absent:
            out[kind] = {}
            prov_products[kind] = {
                "version": version,
                "skipped": list(rows_meta.file.unique()),
                "reason": absent[kind],
                "manifest": {},
            }
            continue
        band = None
        if product.cube:
            start, _stop, expect_fp = spans[kind]
            band = slice(lo - start, hi - start)
        gathered, skipped = {}, []
        for fname, group in rows_meta.groupby("file", sort=False):
            rows = group.row.to_numpy(dtype=int)
            got = product.fetch(
                campaign,
                version,
                fname,
                rows,
                keys_by_file[fname],
                band,
                group.time_best.to_numpy(dtype=float),
            )
            if got is None:
                skipped.append(fname)
                if missing == "raise":
                    raise FileNotFoundError(
                        f"{kind}@{version} has no payload for {fname}"
                    )
                got = {}
            if product.cube and "_fp" in got and got["_fp"] != expect_fp:
                raise ValueError(
                    f"{fname}: {kind}@{version} was built on a different "
                    f"frequency axis ({got['_fp']}) than the rest of the "
                    f"version ({expect_fp}). Rebuild it or exclude the file."
                )
            gathered.setdefault(fname, {}).update(
                {k: v for k, v in got.items() if not k.startswith("_")}
            )
        names = sorted({n for d in gathered.values() for n in d})
        assembled = {}
        for name in names:
            parts = []
            for fname, group in rows_meta.groupby("file", sort=False):
                block = gathered.get(fname, {}).get(name)
                if block is None:
                    block = _gap(len(group), hi - lo if product.cube else None)
                parts.append(np.asarray(block))
            assembled[name] = np.concatenate(parts, axis=0)
        out[kind] = assembled
        prov_products[kind] = {
            "version": version,
            "skipped": skipped,
            "manifest": product.manifest(campaign, version),
        }

    return Bundle(
        freqs_mhz=freqs_mhz,
        t=rows_meta.time_best.to_numpy(dtype=float),
        data=data,
        meta=rows_meta,
        products=out,
        provenance={
            "campaign_root": str(campaign.root),
            "antenna": antenna,
            "cross": list(pair) if pair is not None else None,
            "key": key,
            "keys": sorted(set(keys_by_file.values())),
            "band_mhz": band_mhz,
            "used_files": list(rows_meta.file.unique()),
            "skipped_files": no_key,
            "products": prov_products,
            "selection": selection.summary(),
        },
    )


def _gap(nrows, nchan):
    """NaN stand-in for a file a product has no payload for.

    The rows stay in place so every array in the bundle keeps one row
    per integration and nothing silently shifts; what is missing is
    visibly missing, and the file is named in the provenance.
    """
    if nchan is None:
        return np.full(nrows, np.nan)
    return np.full((nrows, nchan), np.nan)
