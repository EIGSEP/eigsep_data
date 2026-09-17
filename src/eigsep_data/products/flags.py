"""RFI flag masks: one HDF5 day file per campaign day.

Layout, unchanged from the campaign's ``flagging/build_masks.py`` and
its read-only accessor ``flagging/read_flags.py``:
``flags/<version>/flags_YYYYMMDD.h5``, with per-(time, channel) category
bitmasks at ``mask["<filename>/<input>"]`` and the frequency axis at
``freqs_mhz``. ``flag_bits.json`` beside them maps bit to category.

**v0 is uint8, v2 is uint16.** v2 is v0's bits unchanged plus bit 8
(``dpss_residual_outlier``, value 256) from B16's DPSS residual
flagging, which v0's byte had no room for. Code that hardcodes one-byte
pixels, or calls ``.tobytes()``, misbehaves silently on the other -- so
the version is a required part of the spec and the dtype comes back as
stored, never coerced.
"""

import numpy as np

from .base import Product, axis_fingerprint, read_manifest, register


@register
class Flags(Product):
    kind = "flags"
    cube = True

    def root_dir(self, campaign, version):
        return campaign.root / "flags" / version

    def versions(self, campaign):
        base = campaign.root / "flags"
        if not base.is_dir():
            return []
        return sorted(
            p.name
            for p in base.iterdir()
            if p.is_dir() and any(p.glob("flags_*.h5"))
        )

    def bits(self, campaign, version):
        """The version's ``flag_bits.json``: bit -> category mapping."""
        return read_manifest(
            self.root_dir(campaign, version) / "flag_bits.json"
        )

    def _day_file(self, campaign, version, fname):
        # Day is the YYYYMMDD in corr_YYYYMMDD_HHMMSSZ.h5, the same
        # slice build_masks.py writes the file under.
        return self.root_dir(campaign, version) / f"flags_{fname[5:13]}.h5"

    def freqs(self, campaign, version):
        import h5py

        day_files = sorted(self.root_dir(campaign, version).glob("flags_*.h5"))
        if not day_files:
            raise FileNotFoundError(
                f"no flags day files under {self.root_dir(campaign, version)}"
            )
        with h5py.File(day_files[0], "r") as h:
            return h["freqs_mhz"][:]

    def fetch(self, campaign, version, fname, rows, key, band):
        import h5py

        path = self._day_file(campaign, version, fname)
        if not path.is_file():
            return None
        with h5py.File(path, "r") as h:
            mask = h["mask"]
            if fname not in mask or key not in mask[fname]:
                return None
            dset = mask[fname][key]
            # Read the rows this selection actually wants. h5py needs a
            # sorted, duplicate-free list for fancy indexing; the index
            # hands rows out in ascending order per file, and the guard
            # is cheap next to the read.
            take = np.asarray(rows, dtype=int)
            if take.size and (np.any(np.diff(take) <= 0)):
                order = np.argsort(take, kind="stable")
                block = dset[np.unique(take[order])]
            else:
                block = dset[take] if take.size else dset[0:0]
            if band is not None:
                block = block[:, band]
            # The day file's own axis, checked by the bundle against the
            # version's. One small read inside an already-open file; the
            # comparison itself is three scalars.
            return {"mask": block, "_fp": axis_fingerprint(h["freqs_mhz"][:])}
