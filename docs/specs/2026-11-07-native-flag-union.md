# Native flag union: bounded C9 component

**Diagnostic, local review component. No v4 release or consumer adoption.**
The workspace C9 audit found that flags v3 omits some samples currently
excluded by separate self-EMI masks in bin_raw, 054 and 056. This component
composes that already-used exclusion with the pinned base on native
coordinates. It does not add a detector or change v3 defaults.

## Scope and assumptions

The supported overlay is the current phase-C/input4 convention: in files
marked `boxair_emi == true`, exclude every row of channels satisfying
`abs(f_MHz-round(f_MHz)) < 0.15` on the actual frequency axis. Input0 is
unaffected. The file-wide scope is inherited from existing consumers; it
does not establish that every excluded cell contains interference. The
curation's false/untested notes and source provenance remain in the record.

Source JSONL may have the existing provenance header. Duplicate files,
missing Boolean membership and unknown header records fail. The default
rejects an unlisted file. `unlisted="off"` is an explicit, recorded opt-in
to reproduce the old consumer's set-membership convention. No inference
about unlisted files is made implicitly. Other wiring phases need their
own checked applicability.

The supplied base definition and original reason bits are retained,
including v3's unresolved cross-input validity, state and guard defects.
This component must not be called a repaired base. Row-only calibration
comb candidates have no channel footprint and remain evidence: they cannot
be passed as pixel exclusions. Pair-only reasons cannot be OR-ed into an
individual input's mask. Splitting legacy cross-change evidence remains
a release gate.

## Public component

`eigsep_data.flag_union` provides:

- `NativeFlags`: exact native file/input/phase, raw-file SHA-256, native
  row/channel IDs, physical frequencies, unsigned flags and provenance.
  There is no nearest-coordinate match or frequency-array index shortcut.
  Row identity is independent of stale header times. Time, duration and
  switch state join through the original file/row identity upstream.
- `FlagSchema`: reason definitions, their RFI/validity/selection/advisory
  categories and explicit named exclusion policies. Unknown bits or policy
  names fail. The `rfi` policy separates contamination reasons from non-sky
  selection; it does not validate detector use on calibration rows.
- `MaskSource` and `compose`: require exact matching coordinates, Boolean
  coverage and source-byte hashes; preserve overlapping reasons; reject
  duplicate sources and pair-only exclusions.
- `candidate_rows`: preserves exact row evidence and its source hash with
  `channel_footprint=null` and `pixel_exclusion=false`.
- `write_diagnostic` / `read_diagnostic`: opt-in immutable day HDF5 files,
  schema/policies, all unfinished C9 gates, coordinate identities and file
  hashes. Publication status is always diagnostic. Missing input coverage,
  mixed axes, partial native-row numbering and modified products fail.

The new named self-EMI reason occupies bit10 in this opt-in schema. **Current
consumers keep bit10**, so they cannot consume this product correctly until
they use `schema.excluded(flags, policy)` instead of their keep-bit constant.
The real-fixture test demonstrates that failure. The diagnostic name/status
and `consumer_policy_migration` gate prevent a claim of ready adoption;
no existing version or consumer is changed here.

## Bounded primary checks

Two `(240,880)` released input4 mask extracts are small compressed JSON
fixtures under `tests/data/flag_union/`, together with the live definition,
frequency bytes/hash, original builder attributes and exact curation rows.
`fetch.py` reproduces the read-only intake from Limbo. It uses no index or
data producer and writes only the caller's fixture directory. The raw-file
SHA-256 is the released dataset's attestation; it is not presented as a new
raw-file hash computation. The full curation hash is retained separately
from the selected fixture curation bytes used by the bounded checker.

The independent oracle expands the old consumer's exact integer-MHz rule
and ORs it with its hard-bit interpretation. Required results are zero
unexplained pixel-set differences, **6,499 additional exclusions** in
`corr_20260716_011800Z.h5`, and **zero** in `corr_20260717_085537Z.h5`.
Original bits, input0 independence, true physical frequencies, strict
boundaries, advisory behavior, candidate non-exclusion, missing coverage,
unknown bits, corruption and round-trip identities are checked separately.
These prove operator parity in two files; they do not establish detection
completeness, DTV removal, a fit shift or a valid covariance approximation.

After committing, reproduce the two diagnostic products with:

```sh
uv run --no-sync python examples/flag_union_check.py \
  --output /private/output/c9-bounded-check \
  --candidate-rows /private/input/v4_comb_rows.parquet
```

The optional candidate table is read only, hashed and filtered by exact
file/row/input. Its selected rows remain evidence. Nothing is inferred from
rounded episode times, and no whole-row or channel exclusion is added.

## Remaining release and migration gates

`BLOCKING_GATES` is machine readable in every output manifest. It includes
MISSING-state inference, state-aware liveness, per-input validity, pair
evidence split, guard review, candidate channel footprint, held-out detector
validation, consumer-policy migration and downstream mask/noise validation.
The existing code and two mask checks cannot settle these gates.

After those gates are reviewed, migrate bin_raw/054/056 together: require an
explicit version, decode its named policy, remove separate EMI ORs only
after equality checks, retain native measured/filled/excluded identity,
and rebuild affected downstream products under new versions. Validate
actual-mask signal transfer and averaging/noise propagation. No coverage
cut, empirical-SEM repair or likelihood reweighting is introduced here.
Campaign generation, shared pushes and publication require the project's
separate review/contribution authorization.
