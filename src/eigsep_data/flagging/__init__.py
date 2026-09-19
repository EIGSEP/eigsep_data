"""RFI flag production: detectors, the mask builder, and its validation.

The reader for what this writes is :mod:`eigsep_data.products.flags`;
these are the writer half of the same contract, which lived apart from
it in the campaign repository until 2026-09-19. Both halves declare the
same layout -- ``flags/<version>/flags_YYYYMMDD.h5`` with per-(time,
channel) category bitfields -- so a change to one is a change to both.

Scope note: this builds ``flags/v0``'s eight categories. The
``dpss_residual_outlier`` bit added by ``flags/v2`` is NOT produced
here; it came from the B16 DPSS trial, which is unvalidated and stayed
in ``data-analysis/`` deliberately. See ``flags/v2/README.md`` in the
campaign for that bit's known defect.

Paths come from :mod:`eigsep_data.paths`; nothing here anchors on its
own ``__file__``, so point the package at a campaign first::

    import eigsep_data
    eigsep_data.set_campaign_root("~/data/marjum-2026-07")
"""

from . import build_masks, detectors, validate

__all__ = ["build_masks", "detectors", "validate"]
