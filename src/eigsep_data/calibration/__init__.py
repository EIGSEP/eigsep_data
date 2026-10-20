"""Builders of the calibration products: ``tcal`` and ``noise_source``.

The readers are :mod:`eigsep_data.products.tcal` and
:mod:`eigsep_data.products.noise_source`; these are the writer half of the
same contracts, kept beside them as :mod:`eigsep_data.flagging` is for the
flag product. Each product version's campaign choices are its
``config.json`` in the campaign tree, so the code holds none. Commands:
``eigsep-build-tcal``, ``eigsep-validate-tcal``, ``eigsep-tcal-bracket-cost``
and ``eigsep-build-noise-source``.

``d5_yfactor`` is Christian's independent calibration chain, vendored
unchanged; the validation compares ``tcal`` with it. ``eigsep_cal`` (and,
for ``d5_yfactor``, ``cmt_vna``) is imported only when a builder runs.
"""
