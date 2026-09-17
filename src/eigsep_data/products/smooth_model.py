"""DPSS smooth-band model: one companion HDF5 per raw file.

``derived/smooth_model/<version>/<same basename as the raw file>``,
written by the campaign's ``flagging/b16_dpss_model.py``. Per-input
groups ``input_<key>`` hold ``model``, ``residual`` and ``flagged``,
band-restricted to the analysis band, with the band's ``freqs_mhz`` at
the file root.

**The stored ``residual`` is not what a waterfall wants.**
``hera_filters`` zeroes the residual at flagged pixels, so
``model + residual`` silently hides exactly the comb and RFI spikes a
data panel exists to show. Every waterfall in this campaign recomputes
``data - model`` instead, and so does the bundle -- which is why this
product hands back ``model`` and leaves the arithmetic to the caller
that has the raw data in hand.
"""

from .base import Product, axis_fingerprint, register


@register
class SmoothModel(Product):
    kind = "smooth_model"
    cube = True

    def root_dir(self, campaign, version):
        return campaign.root / "derived" / "smooth_model" / version

    def versions(self, campaign):
        base = campaign.root / "derived" / "smooth_model"
        if not base.is_dir():
            return []
        return sorted(
            p.name
            for p in base.iterdir()
            if p.is_dir() and any(p.glob("*.h5"))
        )

    def freqs(self, campaign, version):
        import h5py

        companions = sorted(self.root_dir(campaign, version).glob("corr_*.h5"))
        if not companions:
            raise FileNotFoundError(
                "no smooth-model companions under "
                f"{self.root_dir(campaign, version)}"
            )
        with h5py.File(companions[0], "r") as h:
            return h["freqs_mhz"][:]

    def fetch(self, campaign, version, fname, rows, key, band, times):
        import h5py
        import numpy as np

        path = self.root_dir(campaign, version) / fname
        if not path.is_file():
            return None
        group = f"input_{key}"
        with h5py.File(path, "r") as h:
            if group not in h:
                return None
            dset = h[group]["model"]
            take = np.asarray(rows, dtype=int)
            # A companion is written per raw file and has the same row
            # count; a selection that straddles a re-fit with fewer
            # rows must not silently read the wrong integrations.
            if take.size and take.max() >= dset.shape[0]:
                raise ValueError(
                    f"{fname}: smooth_model@{version} has "
                    f"{dset.shape[0]} rows, selection asks for row "
                    f"{int(take.max())}. The companion was built "
                    "against a different version of this file."
                )
            block = dset[take] if take.size else dset[0:0]
            if band is not None:
                block = block[:, band]
            return {
                "model": block,
                "_fp": axis_fingerprint(h["freqs_mhz"][:]),
            }
