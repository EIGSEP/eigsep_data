"""Calibrated pointing: one Parquet table, one row per integration.

``curation/pointing_table.parquet`` with ``curation/pointing_table.
schema.json`` beside it. Joined on ``(file, sample_idx)``, which is the
index's own ``(file, row)`` identity -- an exact join, not a
nearest-in-time one, so no tolerance has to be chosen or defended.

Two columns are easy to confuse and the schema is explicit about it:
``quality`` rates confidence in the az/el *values*, while drive state
(``EL_STUCK``, ``EL_POST_FAILURE``) lives in ``flags``. A parked but
IMU-measured antenna is ``quality="ok"`` with ``EL_POST_FAILURE`` set,
because we know where it pointed. **Select scanning data via ``flags``,
not via ``quality``.**
"""

import numpy as np

from .base import Product, read_manifest, register

#: Columns carried by default. The table has 24; these are the ones a
#: beam or RFI analysis actually joins against. Ask for others with
#: ``columns=`` on the product spec's options.
DEFAULT_COLUMNS = (
    "az_deg",
    "el_deg",
    "az_sigma_deg",
    "el_sigma_deg",
    "height_m",
    "quality",
    "flags",
    "phase",
)


@register
class Pointing(Product):
    kind = "pointing"
    cube = False

    def root_dir(self, campaign, version):
        return campaign.root / "curation"

    def _path(self, campaign, version):
        return self.root_dir(campaign, version) / "pointing_table.parquet"

    def manifest(self, campaign, version):
        return read_manifest(
            self.root_dir(campaign, version) / "pointing_table.schema.json"
        )

    def versions(self, campaign):
        schema = read_manifest(
            campaign.root / "curation" / "pointing_table.schema.json"
        )
        version = schema.get("version")
        return [version] if version else []

    def check_version(self, campaign, version):
        """The table is a single file, so its version is a claim in the
        schema rather than a directory name. Mismatch is an error: a
        caller asking for v1.2 must not be handed v2 silently."""
        declared = self.manifest(campaign, version).get("version")
        if declared is not None and declared != version:
            raise ValueError(
                f"pointing@{version} requested but "
                f"curation/pointing_table.schema.json declares "
                f"{declared!r}. The table is one file, not a versioned "
                "directory -- check out the revision you meant."
            )

    def fetch(self, campaign, version, fname, rows, key, band, times):
        import pandas as pd

        path = self._path(campaign, version)
        if not path.is_file():
            return None
        table = getattr(self, "_cache", None)
        if table is None or table.attrs.get("path") != str(path):
            table = pd.read_parquet(path)
            table.attrs["path"] = str(path)
            # (file, sample_idx) is the join key; indexing once turns
            # every later per-file lookup into a hash hit instead of a
            # full-column scan over 1.2 M rows.
            table = table.set_index(["file", "sample_idx"], drop=False)
            table = table.sort_index()
            self._cache = table
        if fname not in table.index.get_level_values(0):
            return None
        block = table.loc[fname]
        take = np.asarray(rows, dtype=int)
        found = block.reindex(take)
        out = {}
        for col in DEFAULT_COLUMNS:
            if col not in found:
                continue
            values = found[col].to_numpy()
            out[col] = values
        return out
