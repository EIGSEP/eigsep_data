"""Derived companion products, joined to raw rows by ``(file, row)``.

Importing this package registers the products that ship with it. See
:mod:`eigsep_data.products.base` for the plugin contract and
:func:`eigsep_data.bundle.load_bundle` for the join.
"""

from .base import (  # noqa: F401
    Product,
    axis_fingerprint,
    get,
    locate_axis,
    parse_spec,
    register,
    registered,
)
from . import flags, pointing, smooth_model  # noqa: F401,E402

__all__ = [
    "Product",
    "axis_fingerprint",
    "get",
    "locate_axis",
    "parse_spec",
    "register",
    "registered",
]
