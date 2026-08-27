# Alpha-Max Revision 5.15 Official-Availability Data Correction

## Evidence and decision

The first audit on the designated data PC found that the frozen Revision 5.14
input contract cannot be satisfied with genuine Binance USD-M data.  Nine
candidate symbols have official daily aggTrades archives at the protocol floor
(`2022-12-31`), while `TONUSDT` has no archive before `2024-03-01`.  The first
genuine `TONUSDT` archive row is `2024-03-01T12:31:10.473Z`; the first funding
record returned by the official endpoint is `2024-03-01T08:00:00Z`.  That
funding record predates the official `12:30:00Z` onboard time, the nominal
`12:00:00Z` four-hour settlement does not exist, and the next official funding
record is the exact `2024-03-01T16:00:00Z` settlement.  Therefore `16:00:00Z`
is the first post-onboard point from which the genuine four-hour sequence is
continuous; the earlier observation remains source evidence but is not owned
or synthesized into the usable feature interval.

The same audit also found an immutable right boundary inside the historical
interval.  Binance USD-M `exchangeInfo` reports `TONUSDT` with
`onboardDate=1709296200000` (`2024-03-01T12:30:00Z`) and
`deliveryDate=1782205200000` (`2026-06-23T09:00:00Z`).  The final official
funding point is `2026-06-23T08:00:00.005Z`; the final daily aggTrades archive
is `2026-06-23`, and no `TONUSDT` archive exists from `2026-06-24` onward.
Rows in the final archive after the delivery boundary are settlement activity
and are not admitted to the owned raw interval.

No performance-bearing replay was run before this defect was found.

Revision 5.15 keeps the ten-candidate universe, chronology, trial inventory,
costs, seeds, gates, and admission thresholds unchanged.  It adds immutable,
kind-specific half-open official-source availability intervals to the sole
contract manifest:

- the nine continuously available symbols use
  `[2022-12-31T00:00:00Z, 2026-07-01T00:00:00Z)` for both kinds;
- `TONUSDT` raw roots use the first genuine canonical one-second bar boundary,
  `[2024-03-01T12:31:10Z, 2026-06-23T09:00:00Z)`; the earlier exchange
  `onboardDate` is retained as source evidence but is not fabricated into bars
  before the first official aggTrade;
- `TONUSDT` feature roots use
  `[2024-03-01T16:00:00Z, 2026-06-23T09:00:00Z)`; the official pre-onboard
  `08:00:00Z` observation is retained only as evidence, the absent
  `12:00:00Z` point is never synthesized, and `16:00:00Z` is the first usable
  canonical settlement;
- root integrity permits absence only outside the declared interval;
- any out-of-interval partition or row is rejected as fabrication;
- every partition and funding cadence inside the interval remains mandatory;
- canonical raw roots contain every one-second row in their owned interval,
  including exact partition edges, with no sparse or inferred gaps;
- Binance funding timestamps retain their official `source_timestamp_ms` while
  the causal lookup key is normalized to the declared settlement boundary only
  when the observed non-negative jitter is at most one second; collisions,
  negative jitter, and out-of-cadence rows fail closed;
- the nine admissible-cadence candidates use eight-hour funding boundaries;
  `TONUSDT`'s genuine four-hour cadence is preserved in its feature root and is
  explicitly ineligible for the eight-hour resolver in addition to failing the
  original warmup/train availability gate;
- admission still requires the complete original warmup and train history, so
  `TONUSDT` must be rejected rather than excused or synthesized.

Availability is predeclared from official metadata and source observations;
it is never inferred from a phase root being tested.

## Rejected alternatives

- Remove or replace `TONUSDT`: changes the candidate/trial identity after
  observing data availability and introduces more discretion.
- Substitute `GRAMUSDT` after the rebrand: changes instrument identity and
  creates a post-hoc stitched history.
- Move the chronology: a full TON warmup/train would extend beyond currently
  available data and would redefine every fold.
- Backfill synthetic data: violates the raw-first and no-fabrication contract.
- Continue Revision 5.14 unchanged: physically impossible and guaranteed to
  fail coverage validation.

## Implementation boundary

1. Add a v2 contract manifest with per-symbol raw/feature availability
   intervals.
2. Bind both interval maps and their SHA-256 into every root seal and receipt.
3. Propagate the sealed mapping through prelock, activation rehash, and the
   historical process using the prelock-owned manifest; add no CLI override.
4. Materialize phase-owned roots from a new official-data staging root with
   exact half-open clipping and independent regular files.  Source traversal
   and final publication retain directory capabilities and use descriptor-
   relative opens/rename so an ancestor swap cannot redirect the operation.
5. Run prelock once, audit it, then run the physically separate historical
   process once.  Performance results cannot alter this correction.

## Stop condition

Completion requires passing regression/adversarial tests, complete official
phase roots, source/root checksum audits, the 816 validation folds, the 680
historical folds, immutable bundle audits, full repository verification, and a
successful pushed CI run.  Until then the allocation decision remains zero.
