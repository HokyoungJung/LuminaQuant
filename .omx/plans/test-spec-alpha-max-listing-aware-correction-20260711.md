# Alpha-Max Revision 5.15 Official-Availability Correction Test Spec

## Root and manifest contracts

- The v2 manifest has exactly ten lexicographically ordered records and exact
  canonical UTC raw/feature start and end fields.
- Start/end maps are immutable, exact-keyed, ordered, and sealed together by
  SHA-256; every interval is non-empty and half-open.
- A symbol may have zero entries only when its declared interval does not
  intersect the root.
- Partial first and final month/day content is clipped exactly to the declared
  interval.
- Every owned raw month has exact first/last seconds, exact one-row-per-second
  count, exact 1000 ms internal spacing, and exact cross-month continuity.
- Funding output retains the official source timestamp separately from its
  canonical settlement timestamp.  Only a non-negative jitter of at most 1000
  ms may be normalized; negative jitter, cadence mismatch, duplicate canonical
  boundaries, and conflicting rows fail closed.
- The official `TONUSDT` `08:00:00Z` funding row is retained as evidence but
  excluded from the owned feature interval because it predates the official
  `12:30:00Z` onboard time.  The absent `12:00:00Z` settlement is not
  synthesized; the owned continuous four-hour feature interval begins at the
  first post-onboard official settlement, `16:00:00Z`.
- The nine eight-hour candidates normalize to 00/08/16 UTC. `TONUSDT` preserves
  its genuine four-hour cadence but can never enter the eight-hour admitted
  resolver domain.
- Out-of-interval partitions/rows, root-observed boundary inference, missing
  in-interval partitions, duplicate/nonmonotone rows, stale funding, unsafe
  links, and mapping/hash mutation all fail closed.
- Unavailable-to-available adjacent feature roots accept only a first genuine
  point within `8h + 1s`; active-to-active gaps retain the original rule.
- Available-to-unavailable adjacent feature roots accept only a final genuine
  point within the declared right-edge tolerance; post-end data is rejected.

## Admission and process boundaries

- `TONUSDT` remains in the candidate universe but fails original warmup/train
  admission requirements without synthetic rows.
- At least five other candidates must pass or prelock fails.
- Prelock seals roots only with the manifest-owned start/end mappings.
- Activation rehash uses both retained mappings.
- Historical evaluation revalidates the prelock-owned manifest and cannot
  override availability through CLI, environment, config profile, or files.
- A different historical start or end mapping fails before any
  market/funding/order event.

## Data preparation

- The preparation command requires an absent target and genuine canonical
  source roots.
- Source roots and the output parent stay bound by directory descriptors;
  ancestor-directory swaps during read or `renameat2(RENAME_NOREPLACE)` fail
  closed and cannot redirect input or publication.
- Raw monthly and feature daily partitions are clipped to the intersection of
  each phase and kind-specific availability interval.
- Output files are independent regular files with one link; symlink/hardlink
  sources and incomplete post-floor inventories are rejected.
- A canonical preparation manifest records source/output inventories and
  content hashes deterministically.

## Verification

- New targeted tests pass after first demonstrating the old implementation
  fails.
- Existing Alpha-Max section, CLI/process boundary, raw-first, materializer,
  and full repository suites remain green.
- Ruff, format, compile, architecture/static gates, checksum manifests, sealed
  bundle auditors, source/root before-after hashes, push, and hosted CI pass.
