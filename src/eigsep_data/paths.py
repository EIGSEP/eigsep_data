"""Where the campaign lives, set once instead of threaded through.

A campaign is a directory holding ``data/`` beside ``flags/``,
``derived/``, ``curation/`` and ``imgs/``. Every consumer needs to know
where it is, and before this module each one answered that question its
own way: ``Campaign.for_index`` walked up from the index's ``data_dir``,
the ``b15`` scripts read a ``MARJUM_DATA_ROOT`` env var,
``b16_dpss_model.py`` hardcoded this server's mount point, and
``curation/select_files.py`` anchored on its own ``__file__``. Four
conventions, and moving a script broke it. See
``PATH_PORTABILITY_PROPOSAL.md``.

The intended usage is one call right after import::

    import eigsep_data
    eigsep_data.set_campaign_root("~/data/marjum-2026-07")

after which anything that needs the campaign finds it without being
passed a path. The setter takes either the campaign root or its
``data/`` subdirectory -- a directory literally named ``data`` is read
as "the campaign is its parent" -- so pointing at the data itself, which
is how people describe the location out loud, does the right thing.

Resolution order, most specific first:

1. an explicit ``root=`` argument at the call site,
2. whatever :func:`set_campaign_root` was last given,
3. the ``EIGSEP_CAMPAIGN_ROOT`` environment variable,
4. a caller-supplied fallback (for an index, its ``data_dir`` parent).

The programmatic setter deliberately outranks the environment: it is a
deliberate act inside the running process, while the env var is ambient
and may belong to a shell someone forgot they exported in. Setting it to
``None`` restores the env-var behaviour.
"""

import os
from pathlib import Path

#: Environment variable consulted when nothing was set in-process.
ENV_VAR = "EIGSEP_CAMPAIGN_ROOT"

#: Set by :func:`set_campaign_root`; ``None`` means "not configured".
_ROOT = None


def _normalize(path):
    """Expand, resolve, and step up from a ``data/`` subdirectory."""
    p = Path(path).expanduser()
    # Resolve without requiring existence; validation is the caller's.
    p = Path(os.path.abspath(p))
    if p.name == "data":
        p = p.parent
    return p


def set_campaign_root(path, must_exist=True):
    """
    Point the package at a campaign directory.

    Parameters
    ----------
    path : path-like or None
        The campaign root, or its ``data/`` subdirectory -- both name
        the same campaign. ``None`` clears the setting, after which
        :data:`ENV_VAR` applies again.
    must_exist : bool, optional
        Raise if `path` is not an existing directory. Default True: a
        typo here would otherwise surface much later as an empty file
        list or a missing product. Pass False to configure a campaign
        that has not been staged on this machine yet.

    Returns
    -------
    pathlib.Path or None
        The campaign root now in effect.

    Raises
    ------
    NotADirectoryError
        If `must_exist` and `path` is not an existing directory.
    """
    global _ROOT
    if path is None:
        _ROOT = None
        return None
    root = _normalize(path)
    if must_exist and not root.is_dir():
        raise NotADirectoryError(
            f"no campaign directory at {root} (from {path!r}); pass "
            f"must_exist=False to set it anyway"
        )
    _ROOT = root
    return _ROOT


def get_campaign_root(default=None, required=False):
    """
    The campaign root in effect, by the documented resolution order.

    Parameters
    ----------
    default : path-like, optional
        Fallback used only when neither :func:`set_campaign_root` nor
        :data:`ENV_VAR` supplied one -- typically something the caller
        computed, such as an index's ``data_dir`` parent.
    required : bool, optional
        Raise instead of returning ``None`` when nothing resolves.

    Returns
    -------
    pathlib.Path or None
        Not checked for existence: a value from the environment or from
        `default` is reported as given, so that a wrong path is
        recognisable in the error it eventually causes.

    Raises
    ------
    RuntimeError
        If `required` and nothing resolved.
    """
    if _ROOT is not None:
        return _ROOT
    env = os.environ.get(ENV_VAR)
    if env:
        return _normalize(env)
    if default is not None:
        return _normalize(default)
    if required:
        raise RuntimeError(
            "no campaign root configured: call "
            "eigsep_data.set_campaign_root(<path>) or set "
            f"{ENV_VAR} in the environment"
        )
    return None


def campaign_data_dir(default=None, required=True):
    """The campaign's ``data/`` directory, by the same resolution."""
    root = get_campaign_root(default=default, required=required)
    return None if root is None else root / "data"
