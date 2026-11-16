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
This component must not be called a repaired base. Generic row-only candidates remain evidence without a spectral footprint.
The identified calibration comb has known integer-MHz frequencies: exact
detected rows are crossed with the established +/-0.15MHz rule under a
separate hard reason, in all switch states; it is not a missing input. Pair-only reasons cannot be OR-ed into an
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

The new named self-EMI reason occupies bit12 and the calibration comb bit13
in this opt-in schema. Bits 10 and 11 are left unused: in the RFI prototype
lineage they mean retracted and post-DPSS residual, and existing consumers
(e.g. the raster sky/Sun scripts) keep bit10. A consumer that excludes every
non-advisory bit, v3's documented default, therefore excludes the new
reasons without change; `schema.excluded(flags, policy)` remains the
explicit route and is required for the `rfi`/`sky` distinction.
The diagnostic name/status and `consumer_policy_migration` gate prevent a
claim of ready adoption; no existing version or consumer is changed here.

## Bounded primary checks

Three `(240,880)` released input4 mask extracts are small compressed JSON
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
These prove operator parity in bounded files; they do not establish detection
completeness, DTV removal, a fit shift or a valid covariance approximation.

After committing, reproduce the two diagnostic products with:

```sh
uv run --no-sync python examples/flag_union_check.py \
  --output /private/output/c9-bounded-check \
  --candidate-rows /private/input/v4_comb_rows.parquet
```

The optional detection table is read only, hashed and filtered by exact
file/row/input. Its selected rows are retained as evidence and feed the
identified calibration-comb row-by-frequency reason. Nothing is inferred
from rounded episode times, and no whole-row exclusion is added.

## Remaining release and migration gates

`BLOCKING_GATES` is machine readable in every output manifest. It includes
MISSING-state inference, state-aware liveness, per-input validity, pair
evidence split, guard review, held-out detector
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

## 2026-11-07 amendment: conformance and identified calibration comb

Candidate records are validated at all public boundaries, including inherited
and serialized evidence. Product schema v2 binds producer.json and the actual
imported composer source snapshot, its SHA-256, numerical runtime and available
Git/clean-tree/source-match evidence into the product manifest. Readers verify
that historical snapshot rather than requiring it to equal the current source.
Original f35959e products remain unchanged: the old reader still reads them;
the new reader requires explicit allow_legacy_unbound=True for those schema-v1
products, retaining their absence of bound writer provenance without inventing
one. Missing provenance in schema v2 fails.

CHB confirmed the identified calibration comb has known integer-MHz teeth.
Its new bit13 is distinct from file-wide self-EMI bit12 and preserves overlap.
The frozen table supplies205/9/91 exact rows in the three bounded files,
independent of receiver switch state. The established strict +/-0.15MHz rule
is applied without tuning. Generic unidentified row evidence retains its
non-excluding contract. No complete-v4, full-width, false-positive, calibration
or fit-improvement claim follows.

examples/calibration_comb_width.py checks the three raw filtered files only,
using scan_corr_file time_best/native identities and the unchanged frequency
window. It shows raw detected/control spectra and folded spectral contrasts.
Controls are non-detected rows in the same switch state, not certified clean;
the third RFNON control comes from the earlier file. Reported neighbouring
contrast remains a width/contamination limitation, not an automatically widened
mask. The rest of C9 and downstream validation remain open.
