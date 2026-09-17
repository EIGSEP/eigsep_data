# docs

Design records, not user documentation. How to *use* the package is in
the repo root README and in the docstrings; these are the "why it is
shaped like this" documents, written before the work and kept as the
record afterwards.

| Directory | Contents |
|---|---|
| `superpowers/plans/` | Implementation plans, dated by when they were written. `2026-09-11-metadata-index.md` is the per-integration index that `index.py` came from. |
| `superpowers/specs/` | Design specs the plans were built against, e.g. `2026-09-11-switch-state-index-design.md`. |

These are historical by nature: read them for the reasoning, not as a
description of current behaviour, and trust the code and its tests where
they disagree.

The design record for the loader (`bundle.py`, `products/`) lives in the
parent meta-repo as `DATASET_LOADER_PLAN.md`, because it spans three
repositories and the repo split itself.

## Recent changes

- 2026-09-17 (`agent:eigsep-67`): added this file; noted where the
  loader's design record lives, since it is not in this directory.
