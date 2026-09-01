# PRD — Independent Return-First Native-Clock Tri-Core Alpha Portfolio (Revision 5.14)

## Product statement

Create a reproducible, research-only crypto-perpetual portfolio experiment that chooses one validation-only `prelock_champion` and separately reports a fixed-gate historical-exposed return-first leaderboard. Because committed artifacts already expose outcomes in `[2025-09-07,2026-07-01)`, that evaluation is diagnostic robustness evidence only: it can challenge but cannot confirm, select, resize, retune, or deploy any row. Valid evidence comes only from matched raw-first event replay, never factory proxies, converted signal clocks, reconstructed costs, or analytically scaled returns.

## User and job

The user needs an isolated branch a separate data PC can run without edits or discretionary choices to determine whether daily trend persistence, daily near-52-week-high anchoring, 4-hour funding carry, or their portfolio combinations improve return while normally holding MDD to 30% and conditionally allowing up to 35%.

## Immutable requirements

1. **Return-first after eligibility:** cumulative return, CAGR, Sharpe/DSR, Sortino, Calmar, MDD/duration, tail loss, turnover/RPT, capacity, costs, exposure, funding, coverage, folds/regimes, liquidation, and wipeout are reported. Turnover, RPT, and capacity are diagnostics only and never gate or rank a row.
2. **Portfolio-first, controls intact:** components, equal-weight, equal-risk, shrunk-HRP, scaled siblings, LOO controls, and named incumbents remain separately measurable.
3. **Native clocks:** the outer portfolio consumes every eligible one-second decision window; trend/near-high consume completed UTC daily bars; carry consumes completed UTC 4-hour bars; common 4-hour cadence is reporting only.
4. **MDD:** target `<=30%`; `(30%,35%]` only with strict CAGR and Calmar dominance over the deterministic return-first best matched selection-eligible `<=30%` row; `>35%` rejects. No normal comparator means no exception.
5. **Ruin:** zero liquidations and zero bankruptcy/wipeout at actual-engine nominal 30 bps.
6. **Evidence:** every selectable row/cost cell is a complete actual raw-first engine bundle. Diagnostic evidence cannot select.
7. **Causality:** completed bars only, atomic admitted-universe near-high batches, explicit split-boundary bucket finalization, exact immediately-previous/current bounded funding roots with newest eligible bar-close as-of selection, per-00/08/16-UTC cash settlement from causal boundary rate/emitted-close/pre-boundary quantity, and no later-root visibility, train-only admission/initial fit, validation-only champion/gross, fixed `final_weight_refit=true`, prelock freeze, report-only historical evaluation.
8. **Warmup:** indicator-only state transfers into fresh flat scoring engines; no warmup orders/equity/fills or ghost positions.
9. **No historical evaluation influence or false confirmation:** no historical-evaluation path, inventory, content, metadata, missingness, or result affects prelock; `historical_evaluation_leader` is never a selected/deployable id. Every report records exposed provenance and requires a genuinely fresh future/withheld interval regardless of whether the leader agrees.
10. **Research-only:** explicit tiers, zero paper/live/real allocation, no promotion, invalid artifacts fail closed.
11. **No local data collection or performance claim.**
12. **Isolation/delivery:** exact baseline `252910e54e280cc593365484cbc99d6ca87893f9`, no shared-session/`.omc` contamination, existing dependencies only, bounded diff, CI/reviews/Lore commit/isolated push/hosted CI green.
13. **No implementation discretion:** exact candidate universe, contract-manifest schema, train-only notional/admission computation, interval-to-root ownership, UTC split/fold/regime dates, purge/embargo, allocation return frequency/cost cell/numerics/rounding/order/caps, terminal precedence, and scaling epsilon are frozen with no CLI override.
14. **Immutable runtime and metrics:** the runner constructs only the versioned final `AlphaMaxBacktestConfig` contract without `._rt`; it accepts no profile, environment, merge, or runtime override. Any `LQ_` environment key fails. Exact poll/window/decision clocks, strict phase-owned injected funding-lookup/resolver/raw-accessor kwargs/identity, per-boundary funding ledger, constructed strategy cadence `1`, strict constructor-only cost-sink kwargs, parity, latency/order policy, engine/risk/execution/reporting fields, common-random-number seed, complete UTC 4h metric stream, annualization, DSR/SPA/PBO inputs, and report-only turnover/RPT/capacity formulas are frozen by the architecture plan and runtime-contract hash.
15. **Closed trials:** the sole prior inventory is the exact baseline G004 manifest blob with 1466 canonical prior nodes; `.omx/plans/alpha-max-current-trial-nodes-v1.json` is the sole exact 21-node definition source (file SHA-256 `cfe3a04620c52cc235d6f1cda1cac617ba30cd7327c753fc2f620d8250d51a4e`, current-key-set SHA-256 `3a4791cf353abcb82f9717ce89ee16b9d73d84f431d5b058135046c2ba8e332b`). The canonical schema/key/set-hash contract yields prior-key-set SHA-256 `3b078011040f89e8d788b2cef9214c58f687221104381e26a688a7f8cdbddd78` from actual LF `0x0a` separators/trailer and binding DSR `num_trials=1487`; literal `\n` bytes, alternate ids/params/members/allocation/gross/omission, glob, semantic guess, status-based deletion, correlation discount, or ambient artifact cannot change it.
16. **Exact engine materialization:** every resolvable current component/full/LOO/scaled row uses the sole canonical `alpha_max_engine_portfolio_manifest.v1` builder and exact constructor mapping `{"portfolio_mode":"manifest:<immutable absolute row path>","decision_cadence_seconds":1}`. Component nodes become one-child manifests; full/LOO members resolve only through the frozen component nodes; research-control fields are never passed as constructor kwargs. The real consumer must resolve exact children/weights/gross/cash/native timeframes without fail-closing before engine start. The separately frozen incumbent-resolution artifact SHA-256 `5133bc40116399fe7af32e75a1ecc52a4f385dc8a0b5d3a4a9585e2437615ed8` predeclares all three named incumbents unavailable, so they materialize no proxy manifest. Direct leaf execution, default/ambient manifest, in-memory injection, incumbent re-resolution/proxying, fallback, overwrite, or post-construction mutation rejects.

## In scope

- exact native rows and three research-only completed-bar adapters;
- explicit raw+feature sidecars with ordered bounded cross-root causal funding lookup/content hashes;
- two-engine indicator-only warmup capsules and flat split starts;
- default-OFF immutable positive-fill pricing trace from canonical fill calculation plus post-reduce-only portfolio-application evidence, including reduce-only zero-applied fills, and a separate handler-owned no-fill-attempt ledger for zero execution;
- strict runtime-field allowlisting/read audit, exact immutable portfolio-manifest materialization and constructor binding, phase-owned injected funding-lookup construction/identity, effective one-second portfolio cadence, constructor-only cost-sink injection, sealed runtime hash, and exact common-random-number seed schedule;
- actual event-engine replay for nominal 10/15/20/30-bps cells;
- full component/control/policy/LOO/clean-incumbent matrix plus diagnostic incumbents;
- equal-weight/equal-risk/shrunk-HRP and scaled full equal-risk/HRP;
- fixed combined train+validation final weight refit among frozen members only;
- full-event plus common-4h reporting equity, coverage, validation selection metrics, and separate historical-evaluation report metrics/ranking;
- separate immutable prelock and one-touch historical evaluation commands/roots;
- closed 1487-node trial/rejection/lineage ledger with the exact canonical current-node file/hash, manifests, note, data-PC runbook;
- tests, cleanup, CI, reviews, commit, push.

## Out of scope

- downloads/new feeds or local performance testing;
- factory rewrite, daily-to-4h reinterpretation, new grid, broad factor zoo, router optimization, or OOS retuning;
- new dependencies, event-schema redesign, broad engine refactor, or original strategy behavior/tier changes;
- analytical leverage scaling, passive cost reconstruction, missing funding as zero, or carried split positions;
- paper/live/real deployment, promotion, main-branch overwrite, or merging another session's work.

## Frozen experiment matrix

Selection-eligible when complete and matched:

- `component_trend_1x`, `component_near_high_1x`, `component_carry_1x`;
- `full_equal_weight_1x`;
- `full_equal_risk_1x`, `full_equal_risk_scaled`;
- `full_shrunk_hrp_1x`, `full_shrunk_hrp_scaled`;
- nine 1x LOO rows: equal-weight/equal-risk/shrunk-HRP crossed with omission of trend/near-high/carry;
- exact Track-A id, `cross_asset_lead_lag_momentum`, and `cross_candidate_hybrid:hybrid_v3_5`, only when faithfully resolvable to matched actual-engine manifests.

Diagnostic-only: exact Track-B id, historical/router/H35 headlines, factory/proxy artifacts. Unreplayable clean incumbents remain explicit `incumbent_replay_unavailable` rows and never enter selection or MDD comparison. Thus the frozen registry contains 21 rows including the Track-B diagnostic.

## Primary data-PC flow

1. Run `run_alpha_max_prelock.py` with frozen config, required canonical contract manifest, exchange/output, and explicit raw/feature pairs for warmup, train, purge, validation, and embargo. The CLI has no profile or runtime override. Each pair owns only its fixed half-open interval; no historical-evaluation argument exists.
2. Warm each split causally, atomically finalize every completed working bucket, hash indicator-only capsules, score every row from flat, freeze train membership, fit policies on train, evaluate twelve validation folds, fix scaled gross, and select the sole `prelock_champion` without pruning the matrix.
3. Because `final_weight_refit=true`, refit only equal-risk/HRP static weights on combined train+validation among unchanged members; do not change policy/gross/params/caps/membership and do not rescore validation.
4. Build/hash every final manifest or unavailable status and the historical evaluation-boundary capsule, then seal immutable prelock.
5. Run `run_alpha_max_historical_evaluation.py` once with read-only prelock, the explicit embargo feature root re-hashed against prelock, the separately owned historical-evaluation raw/feature roots, and a new output root. Verify hashes, replay frozen rows once, append a report-only `historical_evaluation_leader`, never mutate prelock or selected fields, and identify the interval as already exposed.
6. Read exactly one ordered `terminal_outcome`: `no_demonstrated_alpha`, `historical_evaluation_incomplete`, `prelock_champion_historical_robustness_failed`, or `prelock_champion_historical_robustness_passed`. Read leader disagreement and incumbent comparison only from their orthogonal fields; `requires_fresh_confirmation=true` and `confirmation_status="not_run"` are invariant.

## Functional requirements

### FR1 — Frozen hypotheses and class safety

- Validate/version/canonicalize/hash native params, exact ten-symbol `candidate_symbols` trial identity, the separately sealed 5–10-symbol `admitted_symbols` execution subset, exact split/fold/regime chronology, calendars, warmup, allocator numerics, costs, matrix, gates, prelock selection, historical-evaluation report ranking, refit, trials, comparators.
- The exact adapters are `ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy`, `ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy`, and `ResearchOnlyFourHourFundingHarvestCarryStrategy`. They inherit originals, consume one completed native bar once, and duplicate no strategy formula. Their complete resolved kwargs, including non-tunable defaults, are the canonical component nodes in `.omx/plans/alpha-max-current-trial-nodes-v1.json`; the runtime config embeds them identically. Each node's `symbols` field remains the ten-symbol candidate identity and is exposed as `candidate_symbols`; it is not the post-admission active list.
- All resolve `research_only`; originals retain baseline tier/behavior.
- Adapter decisions match originals on identical completed native-bar sequences.
- `ArtifactPortfolioModeStrategy(..., decision_cadence_seconds=60)` preserves the legacy default. Alpha-max passes exactly `strategy_params={"portfolio_mode":f"manifest:{manifest_path}","decision_cadence_seconds":1}` and asserts config/argument/constructed attribute equality before engine start. The fourteen full/LOO registry `params` are research-control fields, not direct constructor kwargs; component params are child-only. Any undeclared parameter, direct leaf execution, default/ambient manifest, in-memory definition injection, clock conversion, class fallback, post-construction cadence mutation, or post-freeze mutation fails.
- Exact contract universe is lexicographic `ADAUSDT, AVAXUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, SOLUSDT, TONUSDT, TRXUSDT, XRPUSDT`, Binance linear USDT perpetual only; no discovery/substitution.
- The required canonical contract manifest has schema `alpha_max_contract_manifest.v1`, exchange `binance`, and exactly the frozen symbols with perpetual/linear/non-inverse/USDT quote-margin-settlement/base-volume/multiplier-1 metadata; it is the sole metadata assertion source and is content-hashed into prelock.
- Train-only common admission uses exactly 517 UTC days. For each day, `Q=math.fsum(float64(close_i)*float64(volume_i))` over sorted raw observations; close is finite positive, volume finite nonnegative, zero-volume rows remain zero, and missing any of six completed 4h buckets is failure. Median and 10th percentile are explicit Hyndman-Fan type-7 values over all 517 `Q`s, requiring `>=20m` and `>=2m`. Aggregated-daily proxies, quote-volume substitution, day deletion, interpolation, winsorization, or expected-1s row checks are forbidden.
- Admission also requires readable monotone unique finite partitions, 366 consecutive pre-train daily bars, complete train daily/4h keys, every causal `<=8h` funding lookup, no unresolved daily cross-section, and at least five admitted symbols. Missing later evidence rejects rather than shrinking membership. Exactly one immutable lexicographic `admitted_symbols` tuple and admission artifact are sealed from train. Every manifest child active `symbols`, Backtest/handler/Portfolio/adapter symbol list, funding domain, and near-high barrier equals that same subset before engine start; the ten candidates remain separately visible for trial identity. Rejected candidates cannot emit market events, signals, funding, positions, orders, fills, or trades.

### FR2 — Raw-first native-clock evidence

- Raw 1-second/windowed replay drives intrabar execution, stops, margin, liquidation.
- Trend/near-high decide only on completed daily bars; carry only on completed 4h bars. Forming bars excluded.
- Near-high collects one bar per admitted symbol per daily key, rejects conflicting duplicates/incomplete cross-sections, sorts symbols, and invokes the inherited batch exactly once; symbol arrival order cannot change `_tick`, decisions, or state.
- Common UTC 4h calendar is reporting/comparison only; retain full-event and reporting equity/MDD; gate MDD is their maximum.
- Factory dispatch absent from selectable runs. Every cost cell is an independent engine run.

### FR3 — Explicit ordered funding sidecars

- Canonical `FeaturePointLookup.get_latest_point` adds source-timestamp provenance while existing `get_latest` remains behavior-identical. `HistoricParquetWindowedDataHandler.get_latest_raw_point(...,timestamp_ms=...)` adds an observational `(value,row_timestamp_ms,close_timestamp_ms)` accessor over emitted canonical 1-second rows, where `close_timestamp_ms=row_timestamp_ms+1000`; alpha chooses the greatest completed close `<=query`, permits only finite positive `close`, and never uses the raw row starting at the query boundary or future/feature/default data. The handler also accepts keyword-only `feature_db_path=None`, `feature_exchange=None`, and mutually exclusive `feature_lookup=None`; all defaults preserve baseline, while injection suppresses ambient lookup.
- `AlphaMaxOrderedFundingLookup` accepts immutable root id/path/exchange/start/end/inventory-hash/content-hash specs and only `funding_rate`. It rejects invalid hashes/bounds, out-of-range content, overlap/gaps/duplicate timestamps, unrequested roots, and later-root presence before replay.
- Each phase passes exactly `strict_data_handler_construction=true` and `data_handler_kwargs={"backtest_poll_seconds":1,"backtest_window_seconds":1,"market_window_parity_v2_enabled":true,"feature_db_path":None,"feature_exchange":"binance","feature_lookup":ordered_lookup}`. The default-false opt-in preserves the baseline try/TypeError fallback for every legacy caller, including existing nonempty mappings; alpha's true value requires nonempty kwargs and makes one strict constructor call with no fallback. The windowed handler internally passes exactly empty path/`binance` to its parent before installing the injected object, so no default runtime path is read. Before engine start, lookup object identity and exact phase/root order/bounds/exchange/inventory/content/query bounds must match or reject.
- Each scoring phase opens exactly its immediately-previous/current feature pair: warmup alone; warmup+train; train+purge; purge+validation; validation+embargo; and, only in the separate historical process, sealed embargo+historical. The greatest source timestamp satisfying root ownership, `<=query`, and `<=8h` staleness wins; equal-timestamp conflicts reject. Historical re-verifies the prelock embargo hashes.
- Carry decisions query only at completed 4h bar close and never after raw watermark. Missing/stale is invalid/missing; genuine zero remains zero; no future, older-root fallback beyond eight hours, cross-gap, ambient, or zero-fill fallback.
- Alpha-only `AlphaMaxFundingBoundaryResolver(ordered_lookup, admitted_symbols)` is passed through `Portfolio(...,funding_boundary_resolver=...)`; it stores/exposes the exact immutable tuple by identity, never infers it from ten-candidate roots or positions, and rejects any outside symbol before feature/raw lookup or mutation. Its allowed symbol/coverage domain is exactly the sealed `admitted_symbols` subset. `None` preserves the legacy funding path exactly. It enumerates each crossed 28800-second epoch boundary in order, queries rate independently at that exact boundary (`<=8h`), queries the already-emitted raw 1-second row with greatest `row_start+1000<=boundary` close (`<=1000ms` stale), uses the signed quantity held immediately before it, calls canonical funding payment with `periods=1`, advances the anchor to the charged boundary, and emits a reconciled ledger. The full all-held-symbol crossed-boundary batch validates before any cash/ledger/anchor mutation; any missing/stale/future/mismatched point fails `funding_boundary_coverage` atomically, with no current-snapshot reuse/static/default fallback.
- Alpha fill anchoring sets the funding anchor to fill time only for zero-to-nonzero entry or sign flip, retains it for same-sign add/reduce, and clears it when flat. Event ordering charges entry one second before a boundary but never retrocharges entry exactly at/after it.

### FR4 — Indicator-only warmup and split boundaries

- Scoring starts UTC 00:00 with at least 366 completed daily and 64 completed 4h warmup bars.
- Warmup-only engine emits no equity/orders/fills; allowlisted hashed capsule transfers only indicator/cadence history.
- Fresh scorer restores capsule with flat cash/positions/orders/margin and empty aggregator.
- Train warmup sees only the warmup pair; validation warmup sees sealed train plus purge pairs; the historical-evaluation capsule sees sealed train/purge/validation/embargo prefixes and is built before any historical-evaluation input.
- Historical evaluation roots with pre-boundary raw/features reject. Cost cells reuse capsule hash in independent fresh engines under identical seed schedule.
- Before every capsule/boundary handoff, `finalize_completed_native_buckets(boundary_watermark)` consumes each working bucket with `start + timeframe <= watermark`, atomically closes near-high, performs funding as-of at 4h close, rejects genuinely partial buckets, and matches natural next-row promotion without transferring generic aggregator state.
- Frozen half-open UTC chronology is warmup `[2022-12-31,2024-01-01)`, train `[2024-01-01,2025-06-01)`, purge `[2025-06-01,2025-06-08)`, validation `[2025-06-08,2025-08-31)` as twelve 7-day folds, embargo `[2025-08-31,2025-09-07)`, historical-evaluation report `[2025-09-07,2026-07-01)`. Validation regimes are three consecutive 28-day blocks; historical-evaluation report folds are the initial partial September block plus full months October 2025–June 2026.
- Root ownership is one-to-one in that same order: `warmup`, `train`, `purge`, `validation`, `embargo`, then historical command `historical-evaluation`. Each raw/feature timestamp belongs to exactly one pair; extra/out-of-range/overlapping records reject. Only the exact immediately-previous/current composite in FR3 may answer a causal `<=8h` boundary lookup; later or nonadjacent pairs are never opened. Purge/embargo warm only and never score/fit/select.

### FR5 — Exact cost trace and realistic cells

- `ExecutionModel.compute_fill(..., attribution_sink=None)` optionally emits exact attribution only for `executed_qty > 0`, without changing `FillResult`, RNG, fills, orders, equity, or event schema.
- Trace activation is exact runner-owned `SimulatedExecutionHandler(..., *, record_cost_attribution=False)` and forwarded only through positive-execution fill metadata/evidence when requested; no config/env/global fallback.
- When `compute_fill` is called but returns zero execution, the handler emits exactly one immutable `no_fill_attempt` with requested/unfilled quantity, raw price, bar volume, cap ratio, lineage, maker/RNG flags, and one of `liquidity_cap_zero_market|liquidity_cap_zero_limit|liquidity_cap_zero_conditional`. It emits no pricing trace, `FillEvent`, portfolio-application record, trade, fee, or equity mutation. A non-crossed limit that never calls `compute_fill` is not a no-fill pricing attempt.
- Base slip, volatility multiplier, half-spread, sqrt impact, pre/post clamp, clamp adjustment, participation, maker/taker, partial qty, fee, order kind, funding, financing, liquidation reconcile.
- Exact `Portfolio(..., sampling_timeframe=None, *, fill_application_attribution_sink=None, funding_boundary_resolver=None)` emits one immutable linked record for every positive-execution pricing trace after reduce-only clamping, with model/applied quantity, fill cost, commission, scale, status, and zero-applied reason. Flat/wrong-side discarded positive fills remain in reconciliation evidence; handler no-fill attempts are outside this bijection and have zero application records. OFF/ON prices, fills, equity, RNG, and later fills match.
- `Backtest` adds only default-empty `portfolio_kwargs` and `execution_handler_kwargs`. Empty values preserve the existing constructor/fallback path exactly; nonempty values make one strict no-fallback constructor call. Alpha-max passes exactly `portfolio_kwargs={"fill_application_attribution_sink":collector.record_application,"funding_boundary_resolver":funding_boundary_resolver}` and `execution_handler_kwargs={"record_cost_attribution":True}`, asserts sink/resolver/handler/Portfolio-bars/ordered-lookup identities, resolver admitted-tuple identity/ordered equality to every active engine symbol list, plus exact bound raw-accessor owner/function and execution flag before replay, and never mutates instances after construction. Ordinary callers remain OFF.
- Nominal taker-reference cells 10/15/20/30 use taker fee 4, maker fee 2, half-spread 1, base slips 5/10/15/25 bps; actual all-in cost is separate.

### FR6 — Evidence tiers and completeness

- Identity/diagnostic evidence cannot set `selection_valid:true`.
- Complete evidence requires engine, raw/report equity, cost, exposure, funding, coverage, capsule, ruin, matrix, provenance hashes.
- Missing fields produce stable invalid reasons. Stable serialization deterministic; runtime metadata isolated.
- The runner reads no config/profile/environment/merge source beyond the versioned frozen experiment JSON and explicit CLI data/manifests/output paths. Every `LQ_` environment key fails `ambient_lq_environment`; non-null explicit handler/backtest kwargs prevent every `get_default_runtime_config()` fallback. Every engine/portfolio/handler runtime-field read and every direct environment/default-config callsite is source-audited; any other read fails `unfrozen_runtime_field`. Every run seals the complete runtime-contract hash.
- Frozen runtime values are exactly those in §4.7.1 of the architecture plan: 1-second base/window/poll/decision clocks, explicit parity v2, native 4h/1d clocks, one-event chunks, no skip-ahead, 1/1-bar latency, explicit MKT and inherited-limit policy, CPU, 10,000 USDT initial capital, notional-fraction sizing, 3x isolated leverage, normalized lot/tick contract, exact risk/order ceilings, exact execution/cost/margin/funding fields, annual periods 2190, zero risk-free/Sortino target, and no risk-free series.
- Common-random-number seeds are derived only from `alpha_max_20260710`, split/fold id, and nominal cost bps by the exact SHA-256 rule; row id is excluded.
- Core metrics use the complete finite positive UTC 4h arithmetic net-return stream and `portfolio.optimizer_core.metrics(..., periods_per_year=2190)` only. Full-event MDD remains separate; DSR uses exact `num_trials=1487` from the frozen G004 blob plus the exact canonical 21-node registry (current-key-set SHA-256 `3a4791cf353abcb82f9717ce89ee16b9d73d84f431d5b058135046c2ba8e332b`) and the pre-gate finite 30-bps Sharpe variance, SPA uses 2000 rounds/frozen block rule/seed 12345, and CSCV PBO uses eight splits. The exact node schema/dedup/key/hash, drawdown-duration, VaR/ES, turnover, RPT, and capacity formulas are those in §§4.7.2/4.11; no alternate inventory or metric implementation is eligible.

### FR7 — Fit, policies, scaling, and fixed refit

- Train-only admission and initial equal-risk/HRP fit use nominal-20-bps actual-engine component equity converted to finite, exact-calendar UTC daily arithmetic net returns; sorted ids, no imputation/trailing truncation, and at least 252 complete observations.
- Equal-risk is exactly `ERCPortfolio(max_iter=10000,tol=1e-10,cov_window=None)` using canonical Ledoit-Wolf covariance, capped at `.50` full/`.70` LOO. Shrunk-HRP is exactly analytic correlation shrinkage `True`, threshold `.60`, sorted-id greedy first-cluster linkage, inverse-variance splits, then canonical cap projection. MLE std must exceed `1e-12`; invalid allocation rejects without fallback. Apply existing 10-decimal rounding to sorted-id weights, then `cash_residual=1-math.fsum(rounded_weights)`: preserve only `0 <= residual <1e-9`; negative or `>=1e-9` rejects `allocator_rounding_invalid` rather than reallocating.
- Full equal-weight requires all three admitted family buckets; missing family is insufficient and weights are not redistributed. LOO equal-weight splits only its two declared families.
- Validation replays train-fitted 1x rows and fixes scaled gross `clip(.25,2.25,.27/max(mdd,1e-12))`.
- Scaled manifests explicitly scale and replay; no analytical return scaling; passing positive exposure-normalized 1x sibling required.
- `final_weight_refit=true` recomputes only equal-risk/HRP full+LOO static weights on combined train+validation among frozen members; no validation rescore.
- Target/realized gross and clips recorded.

### FR8 — Matrix, trials, and statistical gates

- Attempt all 21 frozen rows on identical roots, universe, calendar, and four costs; validation never prunes matrix.
- Unreplayable/missing rows remain explicit; diagnostic rows remain nonselecting. The exact embedded `.omx/plans/alpha-max-incumbent-resolution-v1.json` value freezes all three named incumbents `incumbent_replay_unavailable` from finite baseline source/report paths, Git blobs, and SHA-256s; runtime scan/re-resolution/proxying is forbidden.
- Each distinct component/policy/gross/LOO/incumbent definition is a hypothesis trial; cost cells are stress observations of that trial, though every run id is recorded.
- Build the binding ledger only from the exact G004 baseline blob plus all 21 current matrix definitions under the frozen node/key rules. Failed, unavailable, and diagnostic current rows remain counted; cost cells do not add trials; ambient/prior globs and status-based deletion are forbidden.
- Apply purged/embargoed folds, DSR `>=.90`, SPA `<=.05`, PBO `<=.50`, positive nominal-30-bps return/CAGR/Calmar/net Sharpe, coverage, hashes, and zero ruin before MDD. Turnover, RPT, and capacity remain report-only.

### FR9 — Exact MDD, prelock selection, and historical-evaluation report ranking

- Use nominal-30-bps gate MDD `max(full_event_mdd, reporting_4h_mdd)` independently in validation selection and historical-evaluation reporting domains.
- `<=.30` normal; `>.35` reject. `(0.30,0.35]` requires nonempty matched normal set and strictly greater CAGR and Calmar than its deterministic return-first best normal row.
- Rank cumulative return, CAGR, Calmar, net Sharpe, lower MDD, lexicographic id.
- Validation first row becomes the only `prelock_champion`/`selected_candidate_id`. Historical-evaluation replay may expose only `historical_evaluation_leader`; it cannot change selection/manifests/weights/gross/caps/thresholds/trials/provenance/deployable fields.
- `terminal_outcome` uses first-match precedence: absent prelock champion -> `no_demonstrated_alpha`; champion historical evidence missing/incomplete -> `historical_evaluation_incomplete`; complete champion failing any fixed historical gate -> `prelock_champion_historical_robustness_failed`; complete champion passing all fixed historical gates -> `prelock_champion_historical_robustness_passed`.
- `leader_differs_from_prelock_champion` and nullable `historical_evaluation_leader` are orthogonal. `incumbent_comparison_status` is `matched_outperformed|matched_not_outperformed|unavailable|not_applicable`. They never replace terminal outcome. `historical_exposure_status="committed_period_outcomes_observed"`, `requires_fresh_confirmation=true`, and `confirmation_status="not_run"` are invariant.

### FR10 — Physical historical evaluation separation

- Prelock exposes/opens no historical evaluation resource and refuses overwrite.
- Historical evaluation verifies all hashes, reads prelock only, writes separate append-only output, refuses duplicate id.
- Historical evaluation raw/feature inventory/content/metadata cannot affect prelock bytes.
- Overlap, purge/embargo violation, pre-boundary records, or config/runtime/source/capsule/membership/manifest mismatch reject before replay.
- Historical evaluation metrics cannot mutate choices/prelock. Arbitrary result poison must leave every stable prelock byte and all selection/deployable fields identical.
- The exposed interval is never described as untouched, locked, prospective, confirmatory, deployable, or independent; even a robustness pass requires a future/withheld uninspected interval.

### FR11 — Manifest consumption and fail-closed safety

- Every resolvable current component/full/LOO/scaled row uses the sole canonical `alpha_max_engine_portfolio_manifest.v1` materializer; component rows are one-child portfolios and full/LOO rows resolve only their frozen member nodes. The manifest top level seals exact `candidate_symbols`, `admitted_symbols`, and `admission_manifest_sha256`; every child retains ten `candidate_symbols` from its registry node but sets active `symbols` to the common admitted subset. The three clean-incumbent attempts consume only their exact frozen unavailable audit records and materialize no manifest or nearby proxy.
- Validation manifests are immutable `validation_train_fit` files using train-fitted weights; final immutable `prelock_final_refit` files use the one allowed train+validation refit. Fixed component/equal-weight payloads and hashes remain byte-identical at their distinct phase paths; scaled rows multiply child weight/cap by the already fixed validation gross and replay in-engine.
- The sole source artifact is the explicit versioned config at its sealed absolute path/hash with readiness true and `max_age_hours=876000`. Before consumer resolution, the runner normalizes/contains the manifest path, rejects symlink/non-regular/multi-link/escaped targets or ancestors, and independently seals both manifest and config with exact bytes/SHA plus one-descriptor pre/post-`fstat` receipts. The real consumer parses manifest JSON only from the exact bytes read once from its opened descriptor and hashes/validates the config from its own exact descriptor bytes; it exposes the manifest receipt followed by one `source:<actual_artifact_id>` receipt per validated source, sorted by unique id, through a default-empty `PortfolioModeDefinition.artifact_read_receipts` tuple; every definition copy preserves the tuple. Only the alpha runner requires the exact ordered pair `artifact_portfolio_manifest`, `source:alpha_max_config`. Immediately after construction and before the first replay event, both alpha consumer receipts must equal the runner receipts exactly, and the runner also rechecks path/ancestor/target/bytes/SHA plus the actual strategy's `portfolio_mode`, source path, and only consumer-retained definition fields (ids/classes/active symbols/params/weights/cash/source path/strategy-derived native timeframes). Exact candidate/cap/source metadata are bound by consumed-byte receipt equality plus no fail-closed reason, not attributed to resolved fields. Persistent or transient in-place/same-byte rewrite, hard-link/replacement/atomic/ancestor/symlink swap, swap-and-restore, receipt omission/reorder, or definition/source mismatch rejects with zero replay events. `Backtest.symbol_list` and handler/Portfolio identity plus resolved component/adapter equality, near-high membership, resolver domain, and funding coverage all match the one admitted tuple.
- Valid manifests resolve children/caps; invalid source/hash/freshness/OOS/real/capsule/child/cap/constructor states fail closed or reject `portfolio_manifest_activation_mismatch`. Gross cap is 2.25; zero real allocation and no promotion flag.

### FR12 — Delivery and observability

- Note documents primary and counterevidence, mechanism limits, params, costs/capacity, liquidity-bucket trend attribution, scaled-vs-1x attribution labeled `risk_transform_not_alpha`, the missing passive-scaled counterfactual, distinct DSR/PBO/SPA meanings, falsifiers, collisions, tiers, and no-claim boundary.
- Handoff supplies exact commands/roots/runtime-contract/schema hashes/outputs/failures/one-touch warning and states that no profile/runtime override is accepted.
- Runs record baseline/worktree, commands, RSS/duration, all hashes, native/report calendars, membership/trials/refit, metrics/costs/exposure/ruin, rejection reasons, paths.
- Full CI, cleaner/reverification, independent reviews, Lore commit, isolated push, hosted CI green mandatory.

## Non-functional requirements

- Pure deterministic evidence/gate core; runner/CLIs own I/O/engine calls.
- No duplicated strategy, metric, allocation, or pricing formulas where canonical code exists; exact reused private allocator helper is pinned by tests and intentional. Report-only RPT/capacity formulas are isolated evidence helpers, not eligibility logic.
- Shared-path changes default off/`None`; non-manifest receipt tuples default empty; generic legacy source IDs/counts and definition-copy receipt preservation retain existing definitions/events/equity byte/numeric golden parity.
- Versioned schemas, stable sorted output, explicit invalid reasons, bounded memory, peak RSS.
- No unsupported “best,” superiority, or performance claim before data-PC evidence; afterward, claims remain explicitly exposed-historical diagnostics and the historical-evaluation leader remains report-only.

## Success criteria

- Every test in `.omx/plans/test-spec-alpha-max-independent-20260710.md` passes.
- All applicable local CI-equivalent and hosted CI jobs pass.
- Independent code-reviewer `APPROVE`, architect `CLEAR`, every invariant proved.
- Lore-compliant commit exists only on isolated pushed branch.
- Data-PC run needs no edit/discretion/ambient feature path/profile/runtime override.
- Local delivery is explicitly not an alpha result.

## Product risks

- Hypotheses may fail costs/statistics; acceptable and cannot trigger tuning against the same exposed historical evaluation.
- Sidecar coverage may reject carry/portfolios; missing remains missing.
- Native adapter/capsule bugs could alter decisions; original-parity and flat-start tests gate release.
- Exact replays are compute-heavy; streaming and RSS observability mitigate.
- Refit changes final weights; fixed scope/no-validation-rescore prevents post-hoc selection.
- Incumbents may not map faithfully; unavailable is reported rather than approximated.
- Shared seams may regress baseline behavior; default-OFF/default-empty plus generic-legacy/copy receipt parity and full CI gate release.
- Fixed 2022–2026 roots may be incomplete on the data PC; the run fails `insufficient_history/coverage` rather than shortening dates or substituting symbols.
- A non-champion may lead historical-evaluation results; it remains research evidence and requires a new untouched holdout rather than post-historical evaluation reselection.

## Source of architecture truth

`.omx/plans/ralplan-alpha-max-independent-20260710.md` is the architecture and ADR. This PRD authorizes no execution until durable Architect-to-Critic consensus completes.
