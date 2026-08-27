# Ralplan — Independent Return-First Alpha Portfolio (Deliberate, Revision 5.14)

## 0. Planning state and immutable inputs

- **Workflow:** `$deep-interview -> $ralplan --deliberate -> $ultragoal + $team`.
- **Current boundary:** planning only. This file does not authorize source edits, branch mutation, Git staging, or execution-worktree creation.
- **Requirements:** `.omx/specs/deep-interview-alpha-max-independent-20260710.md`.
- **Context:** `.omx/context/alpha-max-independent-20260710-20260709T231936Z.md`.
- **Review state:** Architect iterations 1–4 returned `REVISE`; Architect iteration 5 approved Revision 5; later Architect/Critic passes drove runtime, metrics, gate, zero-fill, cadence, sidecar, node-identity, sink-injection, and actual-manifest repairs through Revision 5.4. Architect approved Revision 5.4, but the later independent Critic found three exact defects: the prior-key-set golden hashed literal `\n` instead of byte `0x0a`; legacy funding cash charges reused one latest rate/price across crossed UTC boundaries; and named incumbents lacked a finite frozen replay audit. Revision 5.5 fixed the byte golden, added alpha-only per-boundary rate/price/position settlement with legacy OFF parity, and froze all three incumbent resolutions by exact baseline paths/blob ids/content hashes. A fresh Architect approved those repairs; the later independent Critic then found one remaining test-only contradiction: the hostile E2E list demanded an `incumbent winner` even though the frozen audit makes every incumbent unavailable. Revision 5.6 removed the headline impossible frozen-runtime fixture and confined generic matched-incumbent comparator behavior to a pure synthetic matched-domain unit test outside the frozen experiment. Its fresh Architect review found a second stale E2E bullet that still said `faithful incumbent`. Revision 5.7 made the synthetic matched-incumbent comparator fixture mandatory and exclusively pure-unit, while frozen-runtime E2E covers only the three audit-frozen unavailable statuses and Track-B diagnostic. Its later Critic confirmed every prior blocker closed but found a separate candidate/admitted membership contradiction: the frozen registry carries ten identity symbols while the real manifest consumer would execute exactly the child `symbols` list, yet admission permits only five to nine. Revision 5.8 explicitly preserved the ten registry symbols as `candidate_symbols` for trial identity and sealed one train-only `admitted_symbols` subset for every engine/runtime surface. Its fresh Architect review confirmed that design executable but found that the funding resolver constructor still received only the ordered lookup, so it could not independently enforce or expose the admitted domain. Revision 5.9 injected the same immutable admitted tuple into the alpha-only resolver, verified its identity/value before engine start, and rejected outside-domain queries before lookup or mutation. Its fresh Architect review confirmed that blocker closed but found that the baseline consumer does not retain the manifest-only `candidate_symbols` field in its resolved dataclass. Revision 5.10 validates candidate provenance from the sealed raw manifest bytes/hash before resolution and validates only consumer-represented active fields after resolution; it adds no shared consumer provenance seam. Its fresh Architect review found that the consumer independently reopens the path, so the pre-read was not bound against an inter-stage rewrite or path swap. Revision 5.11 captures the canonical path, ancestor/file identities, bytes, and hash before resolution and revalidates the same identities, bytes, hash, constructed definition, and source path immediately after construction and before the first replay event. Its fresh Architect review confirmed that binding but found one impossible post-construction assertion: the baseline resolved dataclasses validate and then discard cap values. Revision 5.12 keeps exact caps bound to the sealed raw canonical bytes/hash, requires real-consumer fail-closed acceptance, and post-validates only retained fields. Its Architect approved, but the later independent Critic demonstrated a transient swap-and-restore attack: consumer B can be accepted only during its open while runner pre/post snapshots both see A. Revision 5.13 makes the real consumer parse one descriptor-read byte string, exposes immutable manifest and config-source read receipts with actual SHA and before/after `fstat` identity, and requires those receipts to equal the runner's independently sealed receipts before replay. Its fresh Architect review found that the shared receipt cardinality was incorrectly alpha-specific, that definition override copies could discard receipts, and that one PRD exact signature omitted the funding resolver. Revision 5.14 makes the shared receipt list generic for arbitrary validated source IDs, scopes the exact alpha pair to the alpha runner, preserves receipts through every definition copy, and aligns the signature. It now requires a fresh sequential Architect approval followed by a later Critic approval.
- **Execution baseline:** exact commit `252910e54e280cc593365484cbc99d6ca87893f9`.
- **Future isolate:** branch `feat/alpha-max-20260710`, worktree `/home/hoky/Quants-agent-alpha-max-20260710`.
- **Isolation invariant:** the changing shared checkout and all `.omc/` state are reference-only. Never stage, reset, clean, copy, overwrite, or base implementation on that session's uncommitted work.
- **Measurement boundary:** this machine proves semantics, causality, determinism, integration, and CI only. Realized alpha is determined later by the frozen data-PC run.

## 1. Outcome contract

Deliver a research-only **Tri-Core Alpha Portfolio experiment** that gives the data PC the strongest evidence-bounded chance of improving cumulative return and CAGR without inventing implausible rules or hiding risk.

The terminal experiment must:

1. preserve the exact native hypotheses: daily trend persistence, daily 52-week-high anchoring, and 4-hour funding carry;
2. run every selection-eligible component, portfolio, control, ablation, and faithfully replayable clean incumbent through the same raw-first event engine;
3. select exactly one `prelock_champion` from validation, then report a fixed-gate **historical-exposed** return-first leaderboard without allowing it to mutate selection or masquerade as untouched confirmation;
4. report Sharpe, DSR, Sortino, Calmar, CAGR, cumulative return, MDD, drawdown duration, tail loss, turnover/RPT, capacity proxy, all-in costs, exposure, funding, coverage, folds/regimes, liquidation, and wipeout under the exact metric contract below; RPT and capacity are report-only diagnostics and never gates or rank keys;
5. target MDD `<=30%`; allow `(30%,35%]` only under the exact strict-dominance rule in §4.12; reject `>35%`;
6. require zero liquidations and zero bankruptcy/wipeout in the actual-engine nominal 30-bps cell;
7. keep the historical-evaluation raw roots physically unopened by the prelock process, while explicitly recording that the interval itself is already exposed by committed research; use immutable content hashes, report-only evaluation behavior, and the closed 1487-node trial ledger;
8. emit fail-closed research artifacts with zero paper/live/real allocation; and
9. pass local CI, cleanup, independent review, hosted CI, Lore commit, and isolated branch push.

The correct terminal result may be **`no_demonstrated_alpha`**. No gate, symbol rule, parameter, lookback, cost profile, comparator, or tie-break may be relaxed after validation or historical-evaluation observation. No outcome from `[2025-09-07,2026-07-01)` may be called untouched, locked, confirmatory, deployable, or prospective; a genuinely fresh future/withheld interval is required for confirmation.

## 2. Evidence synthesis

### 2.1 Repository facts

- The repository has no currently demonstrated, promotable, cost-robust alpha. Historical/router headlines are diagnostic rather than clean selection evidence.
- The strongest clean Track-A row is frozen by exact id:
  `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled`.
- The historical Track-B diagnostic is frozen by exact id:
  `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`.
- The selected native implementations are already present:
  - `LowTurnoverTrendPersistenceStrategy`, honest only on daily bars;
  - `CrossSectionalNearHighAnchoringStrategy`, committed daily row `wk52`;
  - `FundingHarvestCarryStrategy`, committed 4-hour row `swing_ls`.
- `TimeframeAggregator.get_bars` includes the forming bar; a research adapter must exclude it.
- `ArtifactPortfolioModeStrategy` can forward a shared aggregator and child timeframes but does not currently expose an indicator-only warmup-state protocol.
- The existing generic engine warmup suppresses downstream orders but still lets strategies mutate virtual position state, which is unsafe for these leaves.
- `HistoricParquetWindowedDataHandler` does not currently pass feature-sidecar arguments to its parent, so raw-first carry cannot obtain funding without a bounded repair.
- `ExecutionModel.compute_fill` owns the exact randomized slippage/spread/sqrt-impact/clamp calculation, while `FillResult` exposes only aggregate fill outputs. Exact attribution therefore requires an observational seam inside the canonical calculation.
- DSR, SPA, CSCV/PBO, allocator utilities, event-engine execution, liquidation, and fail-closed artifact consumption already exist and must be reused.

### 2.2 External evidence, counterevidence, and bounded inference

Primary research supports conditional mechanisms, not this experiment's performance:

- **Time-series trend:** Moskowitz, Ooi, and Pedersen document 1–12-month continuation across 58 traditional futures, supporting a medium-horizon trend prior but not daily crypto/perpetual parameters or an MDD ceiling: <https://doi.org/10.1016/j.jfineco.2011.11.003>. Zaremba et al. report broad daily crypto reversal with momentum concentrated in the largest/liquid subset, an explicit falsifier for universal daily crypto trend: <https://doi.org/10.1016/j.irfa.2021.101908>.
- **Near-52-week-high anchoring:** George and Hwang support a U.S.-equity cross-sectional price-to-52-week-high predictor and an anchoring interpretation; they do not validate a crypto-perpetual time-series rule, daily execution, or causal psychology: <https://doi.org/10.1111/j.1540-6261.2004.00695.x>.
- **Perpetual funding/carry:** Ackerer, Hugonnier, and Jermann establish how periodic funding links spot and perpetual-future prices, not that funding predicts positive net returns: <https://doi.org/10.1111/mafi.70018>. BIS Working Paper 1087 documents large, time-varying crypto futures carry, arbitrage-capital constraints, and crash risk, but its main basis evidence is not a direct perpetual-funding backtest: <https://www.bis.org/publ/work1087.htm>.
- **Risk scaling and allocation:** Moreira and Muir show volatility-managed exposure can improve Sharpe in some traditional factor/carry samples, not guarantee crypto CAGR or MDD: <https://doi.org/10.1111/jofi.12513>. Kim, Tse, and Wald caution that much reported TSMOM alpha may come from volatility scaling rather than the signal itself: <https://doi.org/10.1016/j.finmar.2016.05.003>. DeMiguel, Garlappi, and Uppal show complex optimized portfolios often fail to consistently beat 1/N out of sample, supporting the frozen equal-weight control, shrinkage, and caps: <https://doi.org/10.1093/rfs/hhm075>.
- **Multiple testing:** DSR adjusts Sharpe for selection bias, trial dispersion, sample length, skew, and kurtosis: <https://doi.org/10.3905/jpm.2014.40.5.094>. CSCV/PBO diagnoses how the in-sample winner's rank degrades out of sample and requires the synchronized complete trial family: <https://doi.org/10.21314/jcf.2016.322>. Hansen's SPA tests alternatives relative to a fixed benchmark under a studentized bootstrap: <https://doi.org/10.1198/073500105000000063>. None detects bad execution costs/lookahead, proves structural stability, optimizes a strategy, or guarantees MDD; the three statistics are not interchangeable “overfit passes.”

Bounded inference: combining three economically distinct sleeves may improve return continuity and drawdown efficiency, while simple controls/shrinkage may limit estimation error. This remains falsifiable and may fail after costs, exact funding cashflows, scaling attribution, trial correction, or historical-exposed evaluation. The data-PC report and note must expose three falsifiers without changing selection after observation:

1. **Liquidity/horizon sign:** freeze train-liquidity buckets and report trend P&L/contribution by bucket; if positive trend is confined to the weakest-liquidity bucket or disappears after costs in the liquid bucket, label `trend_mechanism_not_supported` rather than generalizing momentum.
2. **Signal versus scaling:** compare every scaled row with its frozen 1x sibling under identical engine/cost inputs, always label incremental scaling return `risk_transform_not_alpha`, and reject scaled selection when the 1x sibling lacks positive exposure-normalized evidence. The experiment does not contain the full passive-scaled counterfactual, so it may not claim that scaling uplift is signal alpha.
3. **Complete-trial/statistic roles:** retain all 1487 trials and synchronized pre-gate returns; record distinct DSR/PBO/SPA inputs/meaning. These statistics cannot cure hidden trials, lookahead, wrong costs, or future regime breaks.

The period `[2025-09-07,2026-07-01)` is **not an untouched holdout**: committed baseline artifacts already expose monthly outcomes for that interval (including `var/reports/best_historical_strategy_improvement/best_historical_strategy_improvement_latest.json`). Revision 5.14 therefore treats it only as `historical_exposed_evaluation`. Physical process separation prevents it from changing prelock state, but cannot restore statistical independence. Every produced report sets `requires_fresh_confirmation=true` and `confirmation_status="not_run"`; only a demonstrably uninspected future/withheld interval can supply confirmation in a later experiment.

## 3. RALPLAN-DR summary

### Principles

1. **Return-first only after eligibility:** cumulative return and CAGR never override causality, cost, ruin, statistical, coverage, or MDD gates.
2. **Preserve the hypothesis clock:** adapters may bridge infrastructure but may not reinterpret daily hypotheses on a 4-hour signal clock.
3. **Actual engine over analytical proxy:** every selectable row is measured through the same raw-first execution, funding, margin, and liquidation path.
4. **One-way information flow:** train fits, validation fixes choices, predeclared refit is bounded, historical evaluation reports only.
5. **Fail closed and count everything:** missing evidence, unmatched incumbents, duplicates, ablations, and prior lineage remain visible and conservatively counted.

### Decision drivers

1. Maximize honest historical-exposed cumulative return/CAGR while retaining a conditional soft MDD ceiling of 35%.
2. Avoid lookahead, forming-bar use, synthetic funding, warmup ghost positions, and analytical leverage/cost fabrication.
3. Produce a bounded, CI-verifiable branch that the data PC can run without code edits or discretion.

### Viable options

- **Option A — allocator-only over existing clean artifacts:** smallest diff and lowest operational risk, but existing artifacts are weak and not all share faithful event-engine evidence.
- **Option B — selected: native-clock Tri-Core actual-engine experiment:** strongest combination of economic rationale, portfolio diversification, and measurement integrity; requires bounded adapters and observational engine seams.
- **Option C — broad factor/router search:** potentially higher in-sample headline, but rejected because effective trials, overfit risk, implementation scope, and historical evaluation consumption become unacceptable.
- **Option D — common 4-hour signal clock:** rejected because it changes daily trend and near-high economics rather than adapting them.

### Deliberate pre-mortem

1. **Failure: apparent edge is timing leakage.** Cause: forming bars, mixed stale/current near-high cross-sections, forward funding, a lost final warmup bucket, or historical evaluation-dependent preprocessing. Mitigation: atomic completed-key barrier, explicit boundary finalization, close-time as-of lookup, physical CLI separation, hostile poison tests.
2. **Failure: high return is a leverage/cost illusion.** Cause: arithmetic scaling, passive reconstructed costs, reduce-only scaling after pricing, or ignored liquidation. Mitigation: actual manifest replay, immutable pricing trace plus portfolio-application evidence, 30-bps ruin gate, full-event MDD, and one-to-one fill reconciliation.
3. **Failure: selection survives by excluding controls.** Cause: incomplete comparison/MDD set or understated trials. Mitigation: frozen matrix in §4.10, closed trial ledger, unmatched rows retained with reason codes, and soft-MDD comparator over every matched prelock-eligible normal row.
4. **Failure: historical evaluation becomes a hidden selector or false confirmation.** Cause: a non-champion historical-evaluation leader overwrites manifests/labels, or an already exposed period is described as untouched support. Mitigation: `prelock_champion` is the only selection state; historical-evaluation outputs are append-only, the exposure flag is immutable, terminal claims are diagnostic, and fresh confirmation is always required.
5. **Failure: implementers choose favorable allocator/data variants.** Cause: unspecified universe, roots, quote-notional formula, percentile method, contract metadata, ERC/HRP numerics, rounding, caps, order, or epsilon. Mitigation: §4.9 freezes every performance-sensitive degree and golden vectors; invalid input rejects without fallback.

## 4. Chosen architecture

### 4.1 Three evidence tiers

1. `identity_coverage`: class, params, provenance, availability only.
2. `approximate_diagnostic`: factory/historical/router evidence; never selection-valid.
3. `actual_engine_complete`: raw-first engine bundle with full equity, costs, funding, coverage, exposures, ruin, hashes, and audit fields; the only selectable tier.

The schema rejects tiers 1–2 with `selection_valid:true`. Missing actual-engine fields yields `incomplete_engine_evidence`; nothing is imputed.

### 4.2 Frozen native hypotheses

The config contains exactly one row for each new family. No grid or hidden fallback exists.

**Daily trend persistence — exact committed row**

- class `ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy`;
- TSMOM horizons `28/56/84` daily bars;
- efficiency `20 / 0.30`, ADX `14 / 20`, volatility persistence `16 / 64 / 1.5`;
- weekly decision cadence, minimum hold `36` weekly decisions, cooldown `4` weekly decisions, max hold `2000`;
- volatility window `56`, target volatility `0.20`, shorts allowed.

**Daily near-high anchoring — exact `wk52` row**

- class `ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy`;
- high lookback `364`, minimum history `60`, volatility window `20`;
- quantile `0.25`, rebalance `7` daily bars, minimum hold `7` daily bars, minimum symbols `5`;
- shorts allowed, target gross `1.0`, target volatility `0.20`, stop loss `0.10`.

**4-hour funding carry — exact `swing_ls` row**

- class `ResearchOnlyFourHourFundingHarvestCarryStrategy`;
- funding window `6`, entry `0.00005`, exit `0`, scale `0.0003`;
- no-fight ROC `4 / 0.06`, trailing ATR `4`, ATR period `14`;
- max adds `2`, add step `1 ATR`, volatility window `36`, target volatility `0.03`;
- max hold `180` 4-hour bars, shorts allowed, add allocation fraction `0.5`.

The prose above is only a readable summary. The sole complete current-node parameter dictionaries—including non-tunable `target_allocation`, `max_order_value`, `min_price`, near-high `max_hold_bars`/`base_allocation`/symbol cap, and every portfolio/incumbent field—are the 21 canonical objects in `.omx/plans/alpha-max-current-trial-nodes-v1.json`. The implementation config must embed those objects byte-for-byte at the JSON-value level; defaults may not be omitted or re-resolved. The registry field named `symbols` remains part of the immutable trial key and always means the ten-symbol **candidate universe**; it is copied to the explicit runtime/manifest field `candidate_symbols` and is never treated as the post-admission execution list.

### 4.3 Native-timeframe research adapters

Create `src/lumina_quant/strategies/alpha_max_research_sleeves.py` with exactly three explicit `research_only` adapters:

- `ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy`;
- `ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy`;
- `ResearchOnlyFourHourFundingHarvestCarryStrategy`.

Each inherits the original class and exact parameter schema, declares aggregator use and only its native required timeframe, excludes the forming bucket returned by `TimeframeAggregator`, tracks the last completed native key, and invokes inherited processing only once per completed bar. It must not duplicate indicator, score, position, or risk formulas. Original classes and default tiers/behavior remain untouched.

The near-high adapter additionally owns a strict **atomic completed-cross-section barrier** keyed by UTC daily bucket start. For key `k`, it accepts at most one completed bar per frozen train-admitted symbol, rejects a conflicting duplicate, and never evaluates symbol-locally. Completion means `k + 1d <= raw_watermark`. When all admitted symbols are present it sorts symbols lexicographically and invokes the inherited `calculate_signals_window` exactly once for the single batch; `_tick` and `_last_eval_time_key` advance exactly once. If the watermark passes completion with any admitted symbol absent, the row/split fails `incomplete_near_high_cross_section` with the missing symbols; there is no stale carry-forward, universe shrink, or later repair.

Decision-trace parity is mandatory: an original class fed the same completed native bars must match adapter signals and indicator trajectory apart from strategy id/provenance and the explicit warmup reset boundary.

The engine base remains raw-first 1-second/windowed replay for intrabar fills, stops, margin, and liquidation. Trend/near-high decisions occur only on completed UTC daily bars; carry decisions occur only on completed UTC 4-hour bars. A common UTC 4-hour calendar is reporting/comparison only and never a signal clock.

### 4.4 Funding sidecar, UTC cash settlement, and ordered cross-root causality

Make four narrow default-neutral seams:

1. `FeaturePointLookup.get_latest_point(symbol, field, *, timestamp_ms)` returns an immutable `(value, source_timestamp_ms)` record under the existing `<=timestamp`, finite-value, and eight-hour-staleness rules; existing `get_latest` delegates and returns only `.value`, preserving its public behavior.
2. `HistoricParquetWindowedDataHandler.get_latest_raw_point(symbol, field, *, timestamp_ms)` returns immutable `(value, row_timestamp_ms, close_timestamp_ms)` from **already emitted** raw OHLCV only. Canonical raw 1-second rows are timestamped by bucket start, so `close_timestamp_ms=row_timestamp_ms+1000`; alpha uses only finite positive `close` and chooses the greatest row whose `close_timestamp_ms<=query`. A row starting exactly at the boundary closes one second later and is ineligible. The accessor never opens/replays a future row and never consults a feature/default/ambient source. The added method is observational and no existing handler path calls it.
3. `HistoricParquetWindowedDataHandler.__init__` adds keyword-only `feature_db_path=None`, `feature_exchange=None`, and `feature_lookup=None`. The first pair passes through to the parent. A non-null injected lookup rejects any `feature_db_path` other than `None`/empty, causes the parent to receive exactly `feature_db_path=""` and `feature_exchange="binance"` so `_build_feature_lookup` cannot read ambient defaults, and then becomes the handler's `_feature_lookup` by identity. All-three-`None` remains byte/numerically identical to baseline.
4. `alpha_max_evidence.AlphaMaxOrderedFundingLookup` is the sole injected feature implementation. It accepts exact immutable `FeatureRootSpec(root_id,path,exchange,start_utc,end_utc,inventory_sha256,content_sha256)` records, constructs one bounded canonical lookup per record, and exposes only `funding_rate` through `get_latest_point`/`get_latest`.

Every data-PC command requires explicit raw root, feature root, and exchange for each split. Ambient feature discovery is forbidden. Hash canonical raw partitions, feature content, exchange, half-open bounds, and coverage inventory before any engine is constructed. Duplicate feature timestamps across adjacent roots, content outside the owned interval, a hash mismatch, non-UTC bounds, gaps/overlap, or any later root in the constructor fails closed.

For a scoring/finalization phase, the lookup receives exactly the ordered immediately-previous/current pair: `warmup` receives only itself; `train=[warmup,train]`; `purge=[train,purge]`; `validation=[purge,validation]`; `embargo=[validation,embargo]`; the separate historical process receives exactly `[embargo,historical_exposed_evaluation]`. No other prefix/root is opened. Query timestamps must lie in the current phase's `[start,end]`, where the inclusive end is allowed only for finalizing a bar whose source point remains inside the current half-open root. Each bounded child may return only a point with `root.start <= source_timestamp < root.end`, `source_timestamp <= query`, and `query-source_timestamp <=8h`. Select the greatest source timestamp across the two children; an equal-timestamp conflict rejects rather than choosing by path/order. Thus an eligible earlier-root point may answer immediately after a split boundary, but no later root can be observed. The historical process re-verifies the sealed embargo-root hashes from prelock before opening the historical root.

The carry adapter consumes `calculate_signals_context(context)` to access this exact composite lookup. A completed 4-hour tuple is timestamped by bucket start, so derive `bar_close = bucket_start + 4h` and require `bar_close <= current raw watermark`. During inherited processing temporarily bind the composite as-of lookup. Missing funding is `None` and fails coverage; genuine numeric zero remains zero. No nearest-future, ambient, wall-clock, cross-gap, older-than-eight-hour, or zero-fill fallback exists.

Funding **cash settlement** is separately frozen. `Portfolio.__init__(..., *, fill_application_attribution_sink=None, funding_boundary_resolver=None)` adds an optional resolver. `None` executes the baseline `_apply_funding` path byte/numerically unchanged, including its legacy lazy anchor, current-bar lookup, defaults, and multi-period calculation. Alpha requires `FUNDING_ON_UTC_BOUNDARY=true` and one `AlphaMaxFundingBoundaryResolver(ordered_lookup, admitted_symbols)` whose feature object is the exact handler-injected lookup and whose immutable ordered `admitted_symbols` tuple is the same object the runner passes to `Backtest`. On every resolve call, `Portfolio` supplies `raw_point_accessor=self.bars.get_latest_raw_point`; the resolver first rejects any requested/held symbol outside its immutable admitted tuple, then verifies the bound method `.__self__ is portfolio.bars` and its `.__func__` is the exact windowed-handler accessor before using it. The resolver never infers membership from ten-candidate roots, handler contents, or current positions, exposes its ordered allowed domain for preflight identity/value assertion, and permits no post-construction replacement. This is constructible because Backtest creates the handler before Portfolio while the runner can create the lookup-bound resolver before Backtest; no factory or post-construction mutation is needed.

For alpha, let `I=FUNDING_INTERVAL_HOURS*3600=28800` seconds, `last` be the position funding anchor, and `now` the current emitted raw-event time. Enumerate exact boundary indices `range(floor(last/I)+1, floor(now/I)+1)` in ascending order. For every `boundary_seconds=index*I` (exactly 00/08/16 UTC), independently resolve:

- `rate_point = ordered_lookup.get_latest_point(symbol,"funding_rate",timestamp_ms=boundary_ms)` with `source_timestamp_ms <= boundary_ms` and age `<=28800000` ms;
- `price_point = handler.get_latest_raw_point(symbol,"close",timestamp_ms=boundary_ms)` with `row_timestamp_ms+1000=close_timestamp_ms <= boundary_ms`, finite positive close, and `0 <= boundary_ms-close_timestamp_ms <=1000` ms;
- the signed quantity held immediately before that boundary.

Resolve and validate the **entire crossed-boundary batch for all held symbols before mutating cash, funding totals, ledger, or anchor**. Any missing, stale, future, nonfinite, identity-mismatched, or duplicate-boundary input fails `funding_boundary_coverage` atomically; alpha never falls back to current/latest bar price, a later sidecar point, a static configured rate, or zero. After the batch is complete, call canonical `ExecutionModel.compute_funding_payment(signed_qty=boundary_qty, price=boundary_close, periods=1, rate=boundary_rate)` once per boundary, apply that one payment to cash/total/funding, persist an immutable ledger row containing boundary/rate-source/price-row-start/price-close/qty/rate/price/payment, and advance `_last_funding_ts[symbol]` to that charged boundary—not `now`. Multiple crossed boundaries therefore use distinct as-of rate and close queries and can never reuse one current snapshot.

Alpha-only fill anchoring makes position ownership at a boundary explicit. After an applied fill, a zero-to-nonzero entry or sign flip sets `_last_funding_ts[symbol]` to the fill event timestamp; an addition/reduction that preserves sign retains the prior anchor; flat clears it. Since `update_timeindex` settles the market timestamp before same-timestamp fills, a position entered one second before a boundary pays, while one entered exactly at or after the boundary does not receive a retroactive charge. The `None` resolver path retains baseline fill-anchor behavior exactly.

The runner seals and reconciles every funding-ledger row: each charged boundary is unique and ascending per symbol, every rate/price source is causal and within its bound, `sum(payment)` equals portfolio funding/cash/equity deltas, and no held crossed boundary is absent. A feature point at `boundary+1ms`, or a raw 1-second row whose bucket starts at `boundary` and closes at `boundary+1000ms`, is hostile future poison and cannot affect the charge at `boundary`.

### 4.5 Two-engine indicator-only warmup protocol

Generic engine warmup is unsafe because the leaves mutate virtual position modes. Every split, row, and cost cell follows:

1. **Warmup engine:** raw replay ends exactly at scoring boundary with downstream equity/orders/fills disabled and sees only the allowed causal prefix.
2. **Indicator capsule:** extract adapter-specific `get_research_indicator_state()`, canonicalize and hash it.
3. **Fresh scoring engine:** instantiate new strategy/portfolio/execution objects with flat cash, positions, orders, and fixed seed; restore only `set_research_indicator_state(capsule)`. The aggregator starts empty at the aligned boundary.

Scoring boundaries are UTC `00:00`, thus daily and 4-hour aligned; partial first native bars reject. Warmup requires at least `366` completed daily bars and `64` completed 4-hour bars. Insufficient history rejects rather than shortens a lookback.

Capsules preserve indicator/cadence state but never economic position state:

- trend: OHLC histories and completed-bar/week keys; reset mode `OUT`, entry/score/bars-held; cooldown satisfied;
- near-high: close/high histories, `_tick`, completed evaluation/time keys; reset modes, entry, bars-held, transient scores;
- carry: OHLC/funding histories, completed keys and prior ROC; reset mode, entry, stop, watermarks, adds, bars-held.

`ArtifactPortfolioModeStrategy` receives narrow behavior-neutral forwarding methods to collect/restore child research capsules. Generic position serialization is not reused.

Its constructor also adds keyword-only `decision_cadence_seconds: int = 60`; the default retains every legacy caller's hardcoded 60-second behavior, while alpha-max passes exactly `1` through `strategy_params`. `AlphaMaxBacktestConfig.DECISION_CADENCE_SECONDS`, the constructor argument, and the constructed strategy attribute must all equal `1` before `TradingEngine` starts or the run fails `effective_decision_cadence_mismatch`. No post-construction mutation, subclass, factory override, or reliance on `Backtest`'s nonpositive-only cadence fallback is allowed. The wrapper therefore observes the first eligible one-second market-window event after a sparse/raw or native boundary, while children still decide only through their completed daily/4-hour adapters.

Boundary rules:

- train warmup uses exactly the assigned warmup raw/feature roots before train scoring;
- validation warmup consumes the already sealed train prefix plus the assigned purge roots, never validation or later records;
- prelock builds the historical-evaluation-boundary capsule from the sealed train, purge, validation, and embargo prefixes before any historical-evaluation root argument exists;
- historical evaluation restores that capsule and rejects every pre-boundary raw/feature record in the historical-evaluation root;
- every scoring split starts flat and discards warmup signals;
- every cost cell receives the same capsule hash and an independent fresh engine under the same deterministic seed schedule.

Assert zero warmup equity/orders/fills and zero ghost position/order/margin state at score start.

Before every capsule extraction or scoring-boundary handoff, the runner calls one narrow adapter lifecycle operation, `finalize_completed_native_buckets(boundary_watermark)`. A working bucket is finalizable iff `bucket_start + native_timeframe <= boundary_watermark`; later buckets are genuinely partial and reject if requested. Finalization atomically feeds deterministic symbol-sorted daily/4h batches, closes the near-high barrier, and performs carry funding lookup exactly at the completed 4h close. It must be decision/state equivalent to natural promotion caused by the next bucket's first raw row, and it may not transfer generic `TimeframeAggregator._working` state into the fresh scorer. Every finalizable key is consumed once or the boundary fails; no drop, duplicate, partial carry, or next-row consumption is allowed.

### 4.6 Exact execution-cost attribution seam

Do not reconstruct costs outside the pricing formula. Add optional keyword-only `attribution_sink=None` to `ExecutionModel.compute_fill`. When non-null, the canonical calculation emits an immutable **pricing trace only when `executed_qty > 0`** after computing the same fill; zero-execution results do not emit a pricing trace. When null, behavior is unchanged. `FillResult` remains unchanged.

`SimulatedExecutionHandler.__init__(events, bars, config, *, record_cost_attribution: bool = False)` is the exact activation seam; only the alpha-max runner passes true. When enabled it passes a local sink for each canonical calculation and attaches a positive-execution immutable pricing trace to `FillEvent.metadata["cost_attribution"]`. If `compute_fill` returns `executed_qty <= 0`, the handler emits exactly one separate immutable `no_fill_attempt` record before its existing continue/pending/remainder behavior, with requested quantity, zero executed quantity, unfilled quantity, raw price, bar volume, cap ratio, order id/kind/lineage, maker flag, RNG-consumed flag, and one reason from `liquidity_cap_zero_market|liquidity_cap_zero_limit|liquidity_cap_zero_conditional`; it never creates a `FillEvent`, portfolio application record, trade, fee, or equity mutation. A non-crossed limit that never calls `compute_fill` is not a no-fill pricing attempt. No config-file fallback, environment variable, class-global, or ambient `RECORD_COST_ATTRIBUTION` lookup is allowed.

`Portfolio.__init__(..., sampling_timeframe=None, *, fill_application_attribution_sink=None, funding_boundary_resolver=None)` is the exact second activation seam. For every **positive-execution pricing trace/FillEvent**, after `_clamp_reduce_only_fill` and even when the positive model fill is discarded, it emits exactly one second-layer application record linked by pricing-trace hash: model quantity/fill cost/commission, applied quantity/fill cost/commission, absolute reduce-only scale, `applied_unchanged|applied_scaled|rejected`, and a stable zero-applied reason (`reduce_only_flat`, `reduce_only_wrong_side`, or `zero_quantity`). Conditional/remainder lineage and order id remain linked. The pricing trace is never mutated. Applied trades copy both layers into research evidence; reduce-only zero-applied fills remain in the reconciliation ledger although no trade is booked. Handler-level `no_fill_attempt` records are explicitly outside this pricing-to-application bijection and must have zero application records. OFF means the handler flag is false and the portfolio sink is `None`, so baseline events/equity/RNG remain byte/numerically unchanged.

The sole construction path is a default-neutral `Backtest.__init__(..., strict_data_handler_construction: bool = False, portfolio_kwargs: Mapping[str,Any] | None = None, execution_handler_kwargs: Mapping[str,Any] | None = None)`. Default-false data-handler construction preserves the existing try/`TypeError` fallback byte-for-byte even when legacy callers already supply nonempty `data_handler_kwargs`. Only alpha passes `strict_data_handler_construction=True`, which requires nonempty handler kwargs and makes exactly one constructor call with no fallback. Empty/`None` portfolio/execution kwargs execute the existing Portfolio `TypeError` compatibility chain and three-positional execution constructor byte-for-byte; nonempty portfolio/execution kwargs use one strict call with no fallback and fail loud on rejection. Alpha-max passes exactly `portfolio_kwargs={"fill_application_attribution_sink": collector.record_application,"funding_boundary_resolver": funding_boundary_resolver}` and `execution_handler_kwargs={"record_cost_attribution": True}`; it never mutates constructed objects. Before replay the runner asserts callable identity on the portfolio sink, resolver identity on the portfolio, `portfolio.bars is data_handler`, resolver ordered-lookup identity equal to `data_handler._feature_lookup`, `funding_boundary_resolver.admitted_symbols is admitted_symbols` plus exact ordered value equality to handler/Portfolio symbols, exact raw-accessor bound-method owner/function validation, `execution_handler.record_cost_attribution is True`, and that the handler's pricing sink is the local metadata-attaching sink; mismatch is `cost_attribution_activation_mismatch` or `funding_boundary_activation_mismatch`. Every selectable alpha engine uses this path, while ordinary `Backtest` and every non-alpha caller pass no kwargs and remain OFF.

Trace fields: raw/fill price, requested/executed/unfilled quantity, maker/taker, fee rate/commission, sampled base slip, volatility multiplier/applied slip, half-spread, sqrt impact, participation/denominator, penalty before/after clamp, clamp adjustment, liquidity cap, and order kind. Reconciliation is:

`base slippage + half-spread + sqrt impact + clamp adjustment = realized adverse price penalty`, with commission, funding, financing, and liquidation separate.

Maker, taker, partial, conditional, 99%-clamp, funding, liquidation, and zero-volume market/crossed-limit/triggered-conditional cases are mandatory. The sinks are observational and must not consume RNG or change control flow. Reconciliation invariants are: every positive pricing trace has exactly one application record; every application record references exactly one positive pricing trace; every `no_fill_attempt` has neither.

### 4.7 Actual-engine evidence and cost cells

Every selection-eligible row uses identical raw roots, feature roots, universe, seed schedule, split boundaries, calendars, portfolio/execution semantics, and cost grid. Record full-event/raw equity and common 4-hour reporting equity. Gate MDD is `max(full_event_mdd, reporting_4h_mdd)`; report both.

| Nominal one-way cell | Taker fee | Maker fee | Half-spread | Expected base slip | Additional modeled costs |
|---|---:|---:|---:|---:|---|
| 10 bps | 4 | 2 | 1 | 5 | sqrt impact, funding, financing, liquidation |
| 15 bps | 4 | 2 | 1 | 10 | same |
| 20 bps | 4 | 2 | 1 | 15 | same |
| 30 bps | 4 | 2 | 1 | 25 | same |

The nominal label is a taker reference, not an all-in cap. Actual all-in cost may exceed it. Every cell is an independent engine replay; nominal 30 bps is the cost/ruin/MDD eligibility reference.

#### 4.7.1 Immutable alpha-max runtime contract

The alpha runner does **not** load or merge `config.yaml`, `configs/profiles/*.yaml`, `LQ_CONFIG_PATH`, environment settings, or CLI profiles. The frozen experiment JSON contains a versioned `runtime_contract`, and the runner constructs one strict, final `AlphaMaxBacktestConfig` from that object **without `._rt`**, so execution uses only explicit uppercase attributes and never a `RuntimeConfig` fallback. Any engine/portfolio/handler read of an attribute outside the allowlisted contract fails `unfrozen_runtime_field`; a test spy records every read. The CLI exposes no profile argument or runtime override. The stable hash of the entire runtime contract is sealed into every run.

Exact common values:

| Domain | Frozen values |
|---|---|
| Clock/data | base/window/decision/poll cadence `1s`; native `4h` and `1d`; `chunk_days=1`; `chunk_warmup_bars=0`; `skip_ahead_enabled=false`; `market_window_parity_v2_enabled=true`; CPU backend only; no GPU/nondeterministic backend |
| Capital/sizing | initial capital `10,000 USDT`; `target_allocation=0.10`; `target_allocation_mode="notional_fraction"`; leverage `3`; isolated margin; effective fraction `1`; min qty/step `.001`, min notional `5 USDT`, price tick `1e-8` for every frozen symbol; data-handler market-spec overrides forbidden |
| Risk | `RISK_PER_TRADE=.005`; `MAX_DAILY_LOSS_PCT=.03`; `MAX_INTRADAY_DRAWDOWN_PCT=.03`; `MAX_ROLLING_LOSS_PCT_1H=.05`; `MAX_TOTAL_MARGIN_PCT=.75`; `MAX_SYMBOL_EXPOSURE_PCT=.50`; `MAX_POSITION_SIZE_PCT=.50`; `MAX_ORDER_VALUE=5000`; `MAX_ORDER_NOTIONAL_PCT=.50`; `MAX_TOTAL_NOTIONAL_PCT=2.25`; `DEFAULT_STOP_LOSS_PCT=.01`; `FREEZE_NEW_ENTRIES_ON_BREACH=true`; `AUTO_FLATTEN_ON_BREACH=false`; `HARD_DRAWDOWN_FLATTEN_PCT=0`; `CONSECUTIVE_LOSS_HALT_COUNT=0`; `ALLOW_METADATA_RISK_OVERRIDE=false`; `MAX_LEVERAGE=3`; `ENFORCE_ORDER_RISK_GATE_IN_BACKTEST=false`; `ATTACH_DEFAULT_PROTECTIVE_STOP=false`; `STRATEGY_QUALITY_ENABLED=false` |
| Order/latency policy | `DEFAULT_ORDER_TYPE="MKT"`; `ALLOW_MARKET_ORDERS=true`; inherited explicit `LMT` uses `LIMIT_PRICE_MODE="one_tick_worse"`, `LIMIT_PRICE_OFFSET_TICKS=1`, `LIMIT_TIME_IN_FORCE="GTC"`; `SIM_LATENCY_MIN_BARS=1`, `SIM_LATENCY_MAX_BARS=1`; reduce-only enforcement true; conditional liquidity cap true; unfilled/remainder behavior remains canonical and traced |
| Execution | `MAKER_FEE_RATE=.0002`; `TAKER_FEE_RATE=COMMISSION_RATE=.0004`; `SPREAD_RATE=.0002` (half-spread `.0001`); cell `SLIPPAGE_RATE=.0005/.0010/.0015/.0025`; `SLIPPAGE_IMPACT_MODEL="sqrt_impact"`; `SLIPPAGE_IMPACT_COEFFICIENT=.10`; `SLIPPAGE_ADV_QUOTE=0`; `SIM_MAX_BAR_VOLUME_RATIO=.10`; `MAINTENANCE_MARGIN_RATE=.005`; `LIQUIDATION_BUFFER_RATE=.0005`; `FUNDING_INTERVAL_HOURS=8`; `FUNDING_RATE_PER_8H=0` but explicit sidecar coverage required; `REQUIRE_FUNDING_COVERAGE=true`; `FUNDING_ON_UTC_BOUNDARY=true`; `ENFORCE_REDUCE_ONLY=true`; `APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS=true` |
| Reporting | annual periods `2190`; risk-free mode/annual/per-period `zero/0/0`; Sortino target `zero`; no risk-free series; persistence false |

The normalized lot/tick contract is a research-engine discretization, not a claim about current exchange filters; the report labels it and the capacity proxy remains non-deployable. Per-row portfolio weights/gross/caps may only lower the common runtime ceilings. The nominal cell changes only `slippage_rate`; all other runtime bytes are identical.

The runner preflight fails `ambient_lq_environment` if **any** environment key starts with `LQ_` (including `LQ__`); it never clears, rewrites, or temporarily masks the process environment. This closes direct `os.getenv` overrides and even logging-only LQ toggles conservatively. It passes non-null explicit `config=AlphaMaxBacktestConfig`, `strategy_timeframe="1s"`, `warmup_bars=0`, `record_history=true`, `track_metrics=true`, and `record_trades=true` to the backtest. For each phase it first constructs the exact phase-owned `AlphaMaxOrderedFundingLookup`, then constructs exactly `AlphaMaxFundingBoundaryResolver(ordered_lookup, admitted_symbols)` from the already sealed admission tuple, and then passes exactly `data_handler_kwargs={"backtest_poll_seconds":1,"backtest_window_seconds":1,"market_window_parity_v2_enabled":true,"feature_db_path":None,"feature_exchange":"binance","feature_lookup":ordered_lookup}` to `HistoricParquetWindowedDataHandler`. Alpha also passes `strict_data_handler_construction=True`; this opt-in plus the nonempty handler mapping makes one strict `Backtest` data-handler constructor call with no TypeError fallback. The default false path retains baseline behavior for every legacy caller, including legacy nonempty mappings. Before engine start the runner asserts `data_handler._feature_lookup is ordered_lookup`, `funding_boundary_resolver.ordered_lookup is ordered_lookup`, `funding_boundary_resolver.admitted_symbols is admitted_symbols` and ordered-equals every active engine symbol list, `portfolio.bars is data_handler`, each resolver call requires the raw accessor bound to that handler with the exact method function, plus exact current-phase id, ordered root ids/bounds/exchange/inventory/content hashes, and feature/raw query bounds. `get_default_runtime_config()` is forbidden and a monkeypatch-to-raise test proves it is never called. Handler market-spec lookup must return no sizing overrides; the sealed normalized `SYMBOL_LIMITS` are the sole lot/tick source. A source/AST audit at baseline paths enumerates direct `os.getenv`, `get_default_runtime_config`, dynamic audit flags, and config reads in the instantiated backtest/data/execution/portfolio path; any new read not represented by the versioned allowlist fails CI.

Common-random-number seed for each fresh engine is exact: `payload = b"alpha_max_20260710\0" + split_or_fold_id.encode("utf-8") + b"\0" + str(nominal_cost_bps).encode("ascii")`; `seed = int.from_bytes(sha256(payload).digest()[:8], "big") % 2147483647`, replacing `0` with `1`. Row id is deliberately excluded so matched rows in the same split/cost cell start with the same RNG state. Every engine is fresh; later draw counts may diverge only through actual order paths.

#### 4.7.2 Exact metric and statistical contract

- The primary metric stream is the complete UTC 4-hour close-to-close **arithmetic net** equity return stream. At the first 4h endpoint, prior equity is the split's flat initial capital; thereafter it is the preceding 4h endpoint. Every endpoint is required, finite, and strictly positive; no zero fill, interpolation, daily substitution, or trailing truncation. Crypto annualization is exactly `6 * 365 = 2190`.
- `portfolio.optimizer_core.metrics(returns, periods_per_year=2190)` is the sole total-return/CAGR/net-Sharpe/Sortino/Calmar/volatility primitive. Thus risk-free and Sortino target are zero, standard deviations are sample `ddof=1`, downside is the negative-return subset, CAGR uses `n/2190`, and Calmar uses canonical return-series MDD. Full-event MDD is separately computed from every event-equity point with the same canonical peak rule; gate MDD remains the maximum of the two.
- Drawdown duration is the maximum count of consecutive primary 4h endpoints strictly below the running equity peak, reported as count and `count*4` hours. Tail loss is report-only: 5% VaR uses the already frozen type-7 quantile of primary returns and expected shortfall is the arithmetic mean of the worst `max(1,ceil(.05*n))` stably sorted returns.
- Gate DSR calls canonical `deflated_sharpe_ratio` on each nominal-30-bps primary stream with `hac_inference=true`, the exact binding `num_trials=1487` derived in §4.11, and `variance_across_trials` equal to the sample variance of finite nonannualized mean/sample-std Sharpes from every executed matched selection-eligible 30-bps row before gates. There is no correlation discount; correlation/effective-trial graphs are audit fields only. Fewer than two finite Sharpes gives variance `0` through the canonical helper and cannot be patched.
- Gate SPA calls `spa_like_pvalue(stream, bootstrap_rounds=2000, block_size=max(1,round(n**(1/3))), seed=12345)`. Gate PBO calls `cscv_pbo(matrix, n_splits=8)` on the exact-calendar nominal-30-bps primary-return matrix of every executed matched selection-eligible row before gates, excluding diagnostic/unavailable rows; fewer than two usable candidates returns canonical `1.0`. Thresholds remain DSR `>=.90`, SPA `<=.05`, PBO `<=.50`; degenerate/nonfinite output fails.
- Turnover, RPT, and capacity are **report-only and absent from eligibility/ranking**. `turnover_notional = math.fsum(abs(applied_qty*fill_price))` over positive applied records; `turnover_multiple = turnover_notional/initial_capital`; `rpt_bps = 1e4*(ending_equity-initial_capital)/turnover_notional`, or null with `undefined_zero_turnover`. For every positive requested model order, `capacity_i = .10*(bar_volume*raw_price)*equity_before/abs(requested_qty*raw_price)`; report finite-positive minimum, type-7 10th percentile, and median as `capacity_proxy_equity_usdt`, or null with `undefined_no_positive_order`. Funding/fees remain inside ending equity. These proxies authorize no gate, rank, or deployment claim.

### 4.8 Physical prelock and historical evaluation separation

`run_alpha_max_prelock.py` accepts the experiment config, the five exact prelock raw/feature root pairs from the §4.9 ownership table, one explicit contract manifest, exchange, and a new output root. It accepts no profile/runtime override. Its parser/process have no historical-evaluation path, inventory, metadata, or hash surface.

`run_alpha_max_historical_evaluation.py` accepts a read-only sealed prelock directory, explicit `--embargo-feature-root` as the sole previous-boundary lookup root, the separate historical-evaluation raw/feature roots, exchange, and a new append-only output root. It re-hashes the supplied embargo root against the sealed prelock root spec before constructing `[embargo,historical_exposed_evaluation]`, then reuses and verifies the sealed contract-manifest plus config/runtime-contract/source/data/capsule/membership/policy/gross/manifest hashes. It cannot open embargo raw data, any nonadjacent feature root, or mutate prelock, and refuses duplicate completed ids.

Adding/removing/renaming/touching/chmod-changing/poisoning historical-evaluation files must leave all stable prelock bytes/hashes unchanged. The command and every output declare `historical_exposure_status="committed_period_outcomes_observed"`, `requires_fresh_confirmation=true`, and `confirmation_status="not_run"`. Historical-evaluation roots contain only assigned post-boundary records; overlap or pre-boundary raw/sidecar points reject.

### 4.9 Frozen universe, dates, allocation, scaling, and refit

The config is the sole source of the following constants; the CLIs expose no override.

**Contract universe and train-only admission**

- exchange/contract: `binance`, linear USDT-margined perpetual only;
- exact candidate symbols, canonicalized lexicographically: `ADAUSDT, AVAXUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, SOLUSDT, TONUSDT, TRXUSDT, XRPUSDT`;
- no substitute or newly discovered symbol is permitted;
- `--contract-manifest` is a required canonical UTF-8 JSON object with `schema_version="alpha_max_contract_manifest.v1"`, `exchange="binance"`, and exactly ten lexicographically sorted records. Each record must equal the frozen config for `symbol`, `market_type="perpetual"`, `linear=true`, `inverse=false`, `quote_asset="USDT"`, `margin_asset="USDT"`, `settle_asset="USDT"`, `volume_unit="base_asset"`, and numeric `contract_multiplier=1.0`; missing/extra fields, duplicate symbols, alternative venue metadata, or hash mismatch reject. The sealed manifest is the sole contract-assertion source; filenames, ambient exchange APIs, and inferred symbols are not metadata.
- admission reads only the assigned warmup+train roots. Raw observations are sorted by unique UTC timestamp. For symbol `s` and each of the exactly `517` train UTC days `d`, compute `Q[s,d] = math.fsum(float64(close_i) * float64(volume_i))` over every raw observation with `timestamp in d`, requiring finite `close_i > 0` and finite `volume_i >= 0`; zero-volume observations contribute zero, while a day missing any of its six completed 4h native buckets is coverage failure rather than an injected zero. Aggregated-daily `close * volume`, quote-volume columns, VWAP, interpolation, and expected-one-second row counts are forbidden substitutes.
- percentile calculation is frozen to Hyndman-Fan type 7: for sorted `x[0:n]`, `h=(n-1)*p`, `j=floor(h)`, `g=h-j`, `quantile_p=(1-g)*x[j]+g*x[min(j+1,n-1)]`. Apply it to all 517 daily `Q` values including genuine zeros, with `p=.50` and `p=.10`; no day deletion, winsorization, log transform, or library-default substitution is allowed. Admission requires median `>=20,000,000 USDT` and 10th percentile `>=2,000,000 USDT`.
- admission additionally requires all assigned partitions readable with strictly increasing unique finite timestamps; `366` consecutive completed daily bars before train scoring; complete daily and 4h native keys throughout train; causal funding at every 4h decision with age `<=8h`; and no unresolved cross-sectional daily key;
- absent raw seconds are event sparsity, not synthetic zero rows; coverage is measured on required partitions and completed native keys, never expected 1-second row count;
- common admitted membership is sorted and frozen for all new components/policies. Fewer than `5` symbols makes every new Tri-Core row `insufficient_train_universe`. Validation/historical-evaluation missingness never changes membership and instead invalidates affected evidence.
- The runner creates exactly one immutable lexicographic tuple `admitted_symbols` from the train admission artifact; it must contain 5–10 unique members and be a subset of the exact ten `candidate_symbols`. The admission artifact seals candidate list/hash, admitted list/hash, per-candidate reasons/statistics, and input-root hashes before any validation replay. The config retains the ten candidates and embeds every registry node unchanged; no admitted list is written back into a trial node or its canonical key.
- For every resolvable row and phase, the same `admitted_symbols` value is the sole active execution universe. Manifest child `symbols`, `Backtest.symbol_list`, `HistoricParquetWindowedDataHandler.symbol_list`, `Portfolio.symbol_list`, every adapter/proxy `symbol_list`, funding coverage/ledger symbol domain, and near-high barrier membership must equal it in the same order before `TradingEngine` starts. The runner passes the identical tuple object to `Backtest`; baseline Backtest/handler/Portfolio preserve that object identity, while serialized manifest lists and `_BarsSubsetProxy`/adapter lists must be exact ordered value copies. Any extra/missing/reordered/replaced symbol, any manifest `candidate_symbols` mismatch, or any funding/order/fill/trade for `candidate_symbols - admitted_symbols` fails `admitted_universe_activation_mismatch` before scoring.

**Half-open UTC chronology and fixed folds**

- indicator warmup: `[2022-12-31T00:00:00Z, 2024-01-01T00:00:00Z)`;
- train scoring/fit: `[2024-01-01T00:00:00Z, 2025-06-01T00:00:00Z)`;
- purge: `[2025-06-01T00:00:00Z, 2025-06-08T00:00:00Z)` = exactly 7 days;
- validation: `[2025-06-08T00:00:00Z, 2025-08-31T00:00:00Z)` = twelve fixed consecutive 7-day folds `validation_w01`…`validation_w12`;
- embargo: `[2025-08-31T00:00:00Z, 2025-09-07T00:00:00Z)` = exactly 7 days;
- historical-exposed evaluation: `[2025-09-07T00:00:00Z, 2026-07-01T00:00:00Z)` with report folds `[2025-09-07,2025-10-01)` then each full UTC calendar month `2025-10`…`2026-06`;
- fixed validation regimes are three 28-day blocks: `[2025-06-08,2025-07-06)`, `[2025-07-06,2025-08-03)`, `[2025-08-03,2025-08-31)`. Historical-evaluation regimes are its ten report folds. No price-derived regime label is used for fitting or selection.

**Exact interval-to-root ownership**

| CLI root pair | Sole owned timestamp interval | Use |
|---|---|---|
| `--warmup-raw-root`, `--warmup-feature-root` | `[2022-12-31,2024-01-01)` | indicator/admission prefix only |
| `--train-raw-root`, `--train-feature-root` | `[2024-01-01,2025-06-01)` | train scoring, admission, initial allocation fit |
| `--purge-raw-root`, `--purge-feature-root` | `[2025-06-01,2025-06-08)` | causal indicator/funding warm only |
| `--validation-raw-root`, `--validation-feature-root` | `[2025-06-08,2025-08-31)` | fixed validation folds/regimes |
| `--embargo-raw-root`, `--embargo-feature-root` | `[2025-08-31,2025-09-07)` | causal boundary warm only |
| historical command `--historical-evaluation-raw-root`, `--historical-evaluation-feature-root` | `[2025-09-07,2026-07-01)` | exposed diagnostic report only |

Each raw observation and feature point is owned by exactly one row according to its timestamp. Root inventory validation uses the exact ten candidate partition names: warmup/train must contain all ten to compute admission, and later owned roots remain sealed against the same no-substitution candidate inventory; the actual engine/handler opens only the admitted subset and rejected-candidate files are inert. Prelock opens only the first five pairs. Ordered causal feature lookup may read already opened earlier pairs so an `<=8h` funding point can cross a boundary, but it may never copy a feature into another interval or read a later pair. Duplicate/overlap, out-of-bound point, missing interval endpoint/native key, non-UTC boundary, or extra partition rejects. Purge/embargo may warm indicators but contribute no scored return, allocation fit, selection metric, or trial statistic. Shortening dates or placing purge/embargo inside a guessed train/validation root is forbidden.

**Static allocation input and exact numerics**

- Fit input is the nominal-20-bps actual-engine component equity, converted to UTC daily close-to-close arithmetic net returns. Component ids are sorted lexicographically, calendars are exact inner-equality (not trailing truncation), values must be finite, and no imputation/zero-fill is allowed. Train fit requires `>=252` complete daily observations; final refit uses the same rule on train+validation scored observations only.
- `equal_risk` is exactly `ERCPortfolio(max_iter=10000, tol=1e-10, cov_window=None).allocate(...)`, which uses `optimizer_core.ledoit_wolf_shrunk_covariance` and `erc_weights`; no quality survivor gate, expected-return term, turnover tilt, family momentum, or alternative ERC is allowed.
- `shrunk_hrp` is exactly `quality_gated_allocation._hrp_weights_with_correlation_shrinkage(sorted_ids, matrix, correlation_shrinkage=True, corr_threshold=0.60)` followed by `project_simplex_with_upper_bounds`; the analytic correlation-shrinkage intensity, absolute-correlation `>=.60` greedy first-cluster rule, sorted-id linkage/ties, inverse-variance within/across clusters, and MLE divisor `T` are frozen. No scipy/sklearn linkage or covariance-HRP substitution is allowed.
- Every input column must have MLE standard deviation `>1e-12`; allocator empty/nonfinite/mismatch/degeneracy is `allocator_fit_invalid`, never an equal-weight fallback. Full per-component cap is `0.50`; LOO cap is `0.70`; target sum is `1.0`. After cap projection, stable artifacts apply existing `_round(..., ndigits=10)` to sorted-id weights, then compute `cash_residual = 1.0 - math.fsum(rounded_weights)`. Exactly `0 <= cash_residual < 1e-9` is preserved as cash without reallocation; any negative residual or residual `>=1e-9` rejects `allocator_rounding_invalid`. Raw projected sum/cap violation also rejects before rounding.
- Golden vectors freeze: equal-variance orthogonal two-column ERC=`[.5,.5]`; a higher-variance second column receives lower ERC weight; equal-variance uncorrelated HRP is equal within tolerance; permuting input mappings after sorted-id canonicalization yields byte-identical weights; cap fixtures hit `.50`/`.70` without violating target/cash reconciliation.

The config freezes `final_weight_refit=true`. Train fixes admission and initial weights. Validation evaluates train-fitted 1x rows and selects the sole `prelock_champion`; scaled gross is `clip(0.25, 2.25, 0.27 / max(validation_1x_mdd, 1e-12))`. Scaled manifests replay in-engine. After validation seals, equal-risk/HRP full+LOO weights refit once on train+validation among unchanged members; equal-weight, components, policy ids, gross, params, caps, membership, and `prelock_champion` cannot change, and no post-refit validation rescore occurs. All manifests and the finalized historical evaluation-boundary capsule hash before any historical evaluation argument exists. A scaled row requires a passing positive exposure-normalized 1x sibling.

### 4.10 Frozen actual-engine comparison matrix

The registry is frozen at exactly 21 rows: 3 components + 5 full rows + 9 LOO rows + 3 clean-incumbent attempts + 1 Track-B diagnostic. Every row is attempted on every mandatory cost cell. Missing/unreplayable rows remain explicit and counted.

**Selection-eligible when complete and matched**

| Group | Frozen rows |
|---|---|
| Components | `component_trend_1x`, `component_near_high_1x`, `component_carry_1x` |
| Full controls/policies | `full_equal_weight_1x`, `full_equal_risk_1x`, `full_equal_risk_scaled`, `full_shrunk_hrp_1x`, `full_shrunk_hrp_scaled` |
| 1x LOO ablations | exactly `loo_equal_weight_omit_{trend|near_high|carry}_1x`, `loo_equal_risk_omit_{trend|near_high|carry}_1x`, and `loo_shrunk_hrp_omit_{trend|near_high|carry}_1x` = 9 rows |
| Clean incumbents | exact Track-A id; `cross_asset_lead_lag_momentum`; `cross_candidate_hybrid:hybrid_v3_5` |

Equal-weight assigns one-third to each of the three required admitted family buckets and deterministic equal weight within a bucket. If any required full-row family is unavailable, the full row is `insufficient_family_coverage`; weights are never redistributed. LOO splits across its two declared remaining buckets. Components and LOO rows are portfolio manifests, so any winner uses one manifest contract.

The finite baseline audit `.omx/plans/alpha-max-incumbent-resolution-v1.json` (`schema="alpha_max_incumbent_resolution.v1"`, file SHA-256 `5133bc40116399fe7af32e75a1ecc52a4f385dc8a0b5d3a4a9585e2437615ed8`) is normative and is embedded JSON-value-identically in runtime config. It freezes exact ordered source/report paths, Git blob ids, content SHA-256s, representation evidence, and resolution for all three named rows. Track-A is a fold-local offline selector/scaled return series; `cross_asset_lead_lag_momentum` is a non-unique family over 2027 report occurrences and many parameter rows; `cross_candidate_hybrid:hybrid_v3_5` is a fold-local Optuna return blend. None has a complete actual-engine strategy/member/weight/gross/native-timeframe manifest at the baseline. All three statuses are therefore predeclared `incumbent_replay_unavailable`, attempted explicitly on every cost cell, counted in the 21 current trials, and excluded from selection, superiority, and MDD comparison. Runtime globbing, re-resolution, nearby-row selection, proxy construction, or inferred translation is forbidden; a later faithful rebuild is a new trial and cannot mutate this audit.

The normative 21-row identities are `.omx/plans/alpha-max-current-trial-nodes-v1.json` (`schema="alpha_max_current_trial_registry.v1"`, file SHA-256 `cfe3a04620c52cc235d6f1cda1cac617ba30cd7327c753fc2f620d8250d51a4e`). It contains every exact row id, implementation, source id, native timeframe, ten-symbol candidate universe, complete component/portfolio/incumbent params, sorted members, allocation method/numerics/caps/fixed-weight policy, gross rule/value, and omission. The runtime config embeds its `nodes` JSON-value identically and does not read `.omx`; any prose/table abbreviation defers to this artifact.

### 4.10.1 Exact actual-engine manifest materialization and constructor binding

The current registry is the hypothesis/trial identity, not a bag of constructor kwargs. Runtime materialization is one global, non-searched contract captured by the config/runtime hash:

- Every resolvable component, full, LOO, and scaled current row runs through `ArtifactPortfolioModeStrategy`; a component row materializes as a one-child portfolio whose child class/params and ten `candidate_symbols` come byte-for-byte from that component node, but whose active child `symbols` equal the single sealed `admitted_symbols` subset. A full/LOO row materializes the lexicographically sorted `members`; each child class/params/`candidate_symbols` come byte-for-byte from its referenced component node and every active child `symbols` field equals that same admitted tuple by ordered value. The fourteen full/LOO registry `params` (`decision_cadence_seconds`, `final_weight_refit`, `score_from_flat`) are research-control fields and are never splatted into the strategy constructor. Component `params` are splatted only into the child constructor. The three clean-incumbent attempts consume only their frozen unavailable resolutions from `alpha-max-incumbent-resolution-v1.json` and materialize no engine manifest; Track-B remains diagnostic and has no engine constructor.
- `materialize_alpha_max_manifest(row, resolved_weights, resolved_gross, phase, config_path, output_root, candidate_symbols, admitted_symbols, admission_manifest_sha256)` is the sole current-row builder. It requires `candidate_symbols` to be the exact ten config symbols and equal every referenced component-node `symbols` value; `admitted_symbols` must be the exact sealed train-admission tuple; the admission-manifest hash must match prelock. No caller may derive or filter membership inside the builder. `phase` is exactly `validation_train_fit` or `prelock_final_refit`; paths are immutable `<output_root>/manifests/<phase>/<row_id>.json`. The runner atomically creates the new output root and both run-owned phase directories once; the materializer accepts only those resolved directories and rejects an existing row file, symlink, escape, or non-run-owned parent. JSON bytes are UTF-8 `json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()+b"\n"`; the SHA-256 is sealed before construction.
- The payload top-level key set is exactly `artifact_kind`, `candidate_symbols`, `admitted_symbols`, `admission_manifest_sha256`, `real_money_execution`, `allow_real_money`, `ready_for_real`, the nine baseline forbidden keys (`uses_current_fold_oos`, `uses_locked_oos_for_selection`, `uses_locked_oos_for_objective`, `uses_locked_oos_for_pruning`, `uses_locked_oos_for_parameter_fitting`, `uses_locked_oos_for_threshold`, `uses_locked_oos_for_tie_break`, `uses_locked_oos_for_correlation`, `uses_locked_oos_for_sizing`), `gross_cap`, `cash_weight`, `allocation_method`, `optimizer_provenance`, `correlation_input_provenance`, `source_artifacts`, and `children`. `artifact_kind="alpha_max_engine_portfolio_manifest.v1"`; `candidate_symbols` is the exact ten-symbol registry identity list, `admitted_symbols` is the one sealed 5–10-symbol train subset, and `admission_manifest_sha256` is its immutable artifact hash; every real-money and enumerated forbidden boolean is false; `gross_cap=resolved_gross`; `cash_weight=max(0.0,1.0-resolved_gross*math.fsum(resolved_weights.values()))` using the already validated rounded allocation and its preserved sub-`1e-9` cash residual; `allocation_method=row.allocation.method`; `source_artifacts` is the single row `{id:"alpha_max_config",path:<absolute resolved explicit --config path>,sha256:<sealed config hash>,max_age_hours:876000,ready:true,portfolio_ready:true}`. No extra top-level field or ambient/default manifest path is allowed.
- Top-level optimizer provenance is exactly `{selection_inputs:["train"]}` for every `validation_train_fit` manifest and for fixed component/equal-weight `prelock_final_refit` manifests. Only equal-risk/HRP `prelock_final_refit` rows, including their scaled siblings, use `{selection_inputs:["train","validation"]}`. Correlation provenance adds `ready:true` and `source="alpha_max_train_daily_net_returns"` or `"alpha_max_train_validation_daily_net_returns"` according to that same rule. Each child repeats those exact provenance objects, sets `ready=true`, `portfolio_ready=true`, `no_current_fold_oos_provenance=true`, `train_validation_optimizer_provenance=true`, `lagged_completed_shadow_optimizer_provenance=false`, every real-money and only the enumerated forbidden current/locked-OOS booleans false, and `source_artifact_id="alpha_max_config"`.
- Children are sorted by `candidate_id`. Each child key set is exactly `candidate_id`, `name`, `strategy_class`, `candidate_symbols`, `symbols`, `params`, `weight`, `leaf_gross`, `leaf_gross_cap`, `netting_group`, `netting_group_gross_cap`, `source_artifact_id`, `ready`, `portfolio_ready`, the three real-money booleans, `no_current_fold_oos_provenance`, `train_validation_optimizer_provenance`, `lagged_completed_shadow_optimizer_provenance`, the same nine forbidden OOS booleans, `optimizer_provenance`, and `correlation_input_provenance`. It has `candidate_id=name=<component row id>`, exact adapter `strategy_class`, `candidate_symbols` equal the referenced node's frozen ten-symbol `symbols`, active `symbols` equal the sealed `admitted_symbols`, exact component-node params, and `weight=leaf_gross=resolved_weights[id]*resolved_gross`. `leaf_gross_cap=netting_group_gross_cap=row.allocation.per_component_cap*resolved_gross`; `netting_group=<component row id>`. The materializer requires finite positive gross `<=2.25`, finite nonnegative weights, exact member coverage, `0 <= allocation_cash_residual=1.0-math.fsum(resolved_weights.values()) < 1e-9` under the already frozen rounding/cash rule, and every leaf/cap/gross identity; otherwise it fails before engine construction.
- For fixed component/equal-weight rows, both phase payloads and hashes are byte-identical while their immutable paths remain distinct; equal-risk/HRP validation uses train-fitted weights, while `prelock_final_refit` uses the one frozen train+validation refit. Scaled rows multiply every child weight/cap by the already frozen validation gross and replay nonlinearly. No manifest is overwritten or analytically rescaled.
- Actual constructor kwargs are exactly `strategy_params={"portfolio_mode":f"manifest:{manifest_path}","decision_cadence_seconds":1}`. `Backtest` receives that mapping and the strict sink kwargs already frozen in §4.6. Validation is deliberately two-stage and binds the runner's seal to what the real consumer actually read. A frozen `ArtifactReadReceipt` contains exactly `artifact_id`, normalized absolute requested path, strict canonical path, SHA-256, byte count, and pre/post descriptor identity `(st_dev,st_ino,stat.S_IFMT(st_mode),st_nlink,st_size,st_mtime_ns,st_ctime_ns)`. The public standard-library-only helper in `src/lumina_quant/utils/artifact_read_receipt.py` rejects symlink/non-regular/multi-link targets, opens exactly once with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`, captures `fstat` before and after reading all bytes from that descriptor, and rejects any identity change; JSON is parsed only from that exact returned byte string. The helper never reopens a path to hash or parse it.
- **Before consumer resolution**, the runner forms the normalized absolute lexical manifest target with `os.path.abspath`, requires it under the run-owned phase directory, `lstat`s the run-owned output root, `manifests` directory, phase directory, and target without following links, rejects any non-directory ancestor, symlink, escape, non-regular target, or target `st_nlink != 1`, and requires `Path.resolve(strict=True)` to equal the lexical target. It captures the ordered ancestor identity vector and an independent runner `ArtifactReadReceipt` plus exact `pre_manifest_bytes`. Only those receipt bytes are parsed/canonicalized; top-level/child candidate/admission fields, ids/classes/params/weights/caps/cash/gross/provenance, canonical bytes, path, and config source are asserted against the frozen row/admission artifacts. The runner independently seals the sole canonical config source through the same one-descriptor receipt contract and matches the manifest-declared path/SHA/readiness/freshness. Any mismatch fails before strategy construction.
- The real manifest consumer uses that receipt helper instead of `path.read_text()`, parses the manifest only from its one descriptor-read byte string, and validates every source artifact by hashing the exact bytes returned with its own one-descriptor receipt rather than `_file_sha256` plus a later path `stat`. `PortfolioModeDefinition` gains only default-empty `artifact_read_receipts: tuple[ArtifactReadReceipt,...]=()`. The shared consumer stays generic: every successful manifest definition exposes the manifest receipt first, followed by exactly one `source:<actual_artifact_id>` receipt for every validated source artifact sorted lexicographically by unique artifact id; zero, duplicate, omitted, or mismatched sources still fail closed under the existing contract. Every `PortfolioModeDefinition` reconstruction/copy path forwards the tuple unchanged, with `_apply_component_param_overrides` specifically assigning `artifact_read_receipts=definition.artifact_read_receipts` by identity. All non-manifest definitions retain the empty default; fail-closed manifest definitions cannot replay. No candidate or cap field is added to the resolved schema.
- **Immediately after the real `ArtifactPortfolioModeStrategy` is constructed and immediately before its first replay event, with no unrelated I/O or engine event between the check and replay**, the alpha runner first requires `artifact_kind="alpha_max_engine_portfolio_manifest.v1"` and exactly the two ordered receipt ids `("artifact_portfolio_manifest","source:alpha_max_config")`, then requires exact value equality between those consumer receipts and the two independently sealed runner receipts. This exact cardinality/id rule exists only at the alpha runner boundary; legacy manifests retain arbitrary validated source ids/counts. It also asserts `constructed_strategy.portfolio_mode == f"manifest:{manifest_path}"`, `constructed_strategy.definition.source_artifacts["artifact_portfolio_manifest_path"] == manifest_path`, and retained child ids/classes/active admitted symbols/params/weights/cash plus strategy-derived native timeframes against the prevalidated execution fields. Exact candidates/caps/source metadata remain proven by the consumer manifest receipt's byte/SHA equality; the absence of a consumer fail-closed reason proves those same consumed bytes passed cap/source validation. The runner then repeats its path/ancestor/target/byte/SHA checks as defense in depth. Persistent or transient in-place/same-byte rewrites, atomic/hard-link replacements, ancestor/target symlink swaps, swap-and-restore attacks, source-path changes, receipt omissions/reordering, or any receipt/byte/hash/identity/definition mismatch fail `portfolio_manifest_activation_mismatch` before replay. Raw candidate/cap fields are never claimed to survive as resolved fields.
- After that bound post-construction check, the runner asserts exact constructed cadence `1` and the sole runtime admission tuple identity/value contract: `backtest.symbol_list is admitted_symbols`, `data_handler.symbol_list is admitted_symbols`, `portfolio.symbol_list is data_handler.symbol_list`, every resolved component and adapter/proxy symbol list equals `list(admitted_symbols)` in order, the near-high barrier membership and funding resolver coverage equal that value, and no rejected candidate can reach a market event, position, funding ledger, order, fill, or trade. Missing/extra raw fields, passing research-control fields, direct leaf execution, default `artifact_manifest_mode`, an in-memory definition seam, post-resolution filter/mutation, manifest fallback, or consumer mismatch fails `portfolio_manifest_activation_mismatch`.
- The raw manifest/config files and hashes, runner and consumer descriptor receipts, pre/post path and filesystem identities, validated candidate/admission provenance, separately resolved definition hash over consumer-retained execution fields, constructed source path, exact constructor mapping, config source-artifact identity, and phase are part of every evidence bundle and prelock. Historical evaluation may use only `prelock_final_refit` hashes and config receipt sealed before its command exists.

This builder is implemented as pure schema/materialization logic in `alpha_max_evidence.py`; no `ArtifactPortfolioModeStrategy` definition-injection seam is added. The shared strategy module gains only the default-empty immutable read-receipt field plus the already specified default-neutral cadence/capsule API, and both consumer and runner reuse the single utility receipt helper; existing non-manifest definitions, generic legacy manifests with arbitrary source IDs/counts, and all strategy economic behavior remain unchanged.

**Diagnostic-only**

- the exact Track-B id above;
- historical +159.833% router headline and any missing H35 artifact;
- factory/proxy identity reports.

Diagnostic rows never select, enter the MDD comparator, or support superiority.

### 4.11 Closed conservative trial ledger

The only prior-lineage source is the immutable baseline-commit blob `252910e54e280cc593365484cbc99d6ca87893f9:var/reports/ultragoal_full_pool_strategy/g004_frozen_candidate_manifest.json`, Git blob `1bb06b6e9d4ca5a82af4686001b880db9709d9b8`, with `artifact_kind="g004_frozen_candidate_manifest"`, `candidate_count=1466`, `candidate_manifest_sha256="1292498b3b729038c74932175a12d910fc4351b2feb3bbfc95f827517e423efe"`, and `candidate_set_sha256="01ca7a5c04b490b5472a62b49d0fcc7d432f0e2045c0e6fae9b1bfcb079a0564"`. No glob, newer file, worktree artifact, `.omc`, output directory, or ambient registry is scanned. Missing or mismatched bytes fail `prior_trial_inventory_mismatch` before replay.

Every prior candidate becomes one `alpha_max_trial_node.v1` object with keys exactly: `schema`, `kind="prior_strategy_leaf"`, `implementation=strategy_class||strategy`, `timeframe=strategy_timeframe||timeframe`, lexicographically sorted symbols after uppercase and slash removal, complete `params`, complete `behavior_metadata=metadata`, and empty `members`, empty `allocation`, null `gross`, null `omission`. Every one of the 21 frozen matrix rows is exactly one object from `.omx/plans/alpha-max-current-trial-nodes-v1.json`, with keys exactly `schema`, `kind="current_matrix_row"`, `row_id`, `implementation`, `source_id`, `timeframe`, `symbols`, `params`, `members`, `allocation`, `gross`, and `omission`. Status, availability, pass/fail, costs, returns, rank, name, notes, tags, timestamps, paths, and hashes are cosmetic and excluded. Thus renamed cosmetic duplicates collapse within a source kind, while any behavioral parameter/metadata/member/allocation/gross/omission difference remains distinct; prior and current nodes never collapse across kinds. Admission is data-determined and stored as evidence inside each current row rather than counted as another searched hypothesis.

Canonical bytes are `json.dumps(node, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`; `trial_key=sha256(bytes).hexdigest()`. Sort and exact-deduplicate full keys. The frozen prior inventory must produce exactly 1466 unique keys and prior-key-set hash `sha256(("\n".join(sorted(prior_keys))+"\n").encode())="3b078011040f89e8d788b2cef9214c58f687221104381e26a688a7f8cdbddd78"`; the serialized separator/trailer bytes are actual LF `0x0a`, and literal bytes `0x5c 0x6e` are forbidden by a byte-level golden. The frozen current registry must produce exactly 21 unique keys and `sha256(("\n".join(sorted(current_keys))+"\n").encode())="3a4791cf353abcb82f9717ce89ee16b9d73d84f431d5b058135046c2ba8e332b"`; its wrapper file hash must equal `cfe3a04620c52cc235d6f1cda1cac617ba30cd7327c753fc2f620d8250d51a4e`. Because `kind` differs, the binding union is exactly `num_trials=1466+21=1487`, regardless of row execution/availability/failure. Four cost cells are stress observations, not trials. Any count/hash/key collision/missing row fails before DSR. Correlation graphs and participation-ratio `N_eff` are diagnostics only and never reduce 1487.

Apply purged/embargoed folds and fixed regimes. Gates include DSR `>=0.90`, SPA `<=0.05`, PBO `<=0.50`, positive nominal-30-bps cumulative return/CAGR/Calmar/net Sharpe, complete native/data/funding coverage, valid hashes, and zero ruin. These run before MDD. RPT and capacity are report-only and cannot reject or rank a row.

### 4.12 Prelock selection and historical-evaluation report ranking

The same nominal-30-bps gate MDD `max(full_event_mdd, reporting_4h_mdd)` and non-MDD gates are applied first to validation to create `prelock_matched_set`. Its `normal_set` contains every matched component/control/ablation/clean incumbent passing earlier gates with MDD `<=.30`; `>.35` rejects; `(0.30,.35]` requires nonempty normal set and strictly greater CAGR and Calmar than the deterministic return-first best normal row. Final validation ranking is cumulative return, CAGR, Calmar, net Sharpe, lower MDD, lexicographic id. The first row is the sole immutable `prelock_champion` selection state.

Historical-evaluation replay creates a separate `historical_evaluation_report`. It may calculate the identical fixed gates and soft-MDD comparator across the complete frozen matched matrix, then expose a nullable `historical_evaluation_leader` using the same return-first ordering, but this is an exposed diagnostic leaderboard—not confirmation and not a selection set. Historical-evaluation values cannot alter any manifest, member, weight, gross, cap, threshold, trial lineage, provenance, row status frozen at prelock, `selected_candidate_id`, or deployable field. The selected id remains `prelock_champion` even if it fails historical gates or another row leads. `requires_fresh_confirmation` is always true; a different leader additionally sets `leader_differs_from_prelock_champion=true`, but cannot become selected without a new experiment and genuinely uninspected holdout.

Different calendar hashes, missing evidence, and diagnostic tiers cannot compare. No historical-evaluation report survivor emits `no_demonstrated_alpha`; no gate is relaxed.

### 4.13 Total terminal-state machine and claims

- **`prelock_champion`:** the only selected id, fixed from validation-only pre-refit evidence and preserved byte-identically after sealing.
- **`historical_evaluation_leader`:** nullable report-only best exposed-historical row under frozen gates; never copied into a selection/deployable field.
- **Incumbent comparison:** diagnostic only. `incumbent_comparison_status` is exactly `matched_outperformed`, `matched_not_outperformed`, `unavailable`, or `not_applicable`; it never changes terminal outcome or selection.

`terminal_outcome` is singular and assigned by the following ordered, exhaustive precedence (first match wins):

1. no `prelock_champion` -> `no_demonstrated_alpha`;
2. champion exists but its historical replay/evidence is missing or incomplete -> `historical_evaluation_incomplete`;
3. champion evidence is complete but it fails any fixed historical-evaluation gate -> `prelock_champion_historical_robustness_failed`;
4. champion evidence is complete and passes every fixed historical-evaluation gate -> `prelock_champion_historical_robustness_passed`.

Leader disagreement and incumbent availability are orthogonal fields and cannot collide with or replace `terminal_outcome`: `leader_differs_from_prelock_champion` is a boolean, `historical_evaluation_leader` is nullable, `incumbent_comparison_status` uses the enum above, `historical_exposure_status="committed_period_outcomes_observed"`, `requires_fresh_confirmation=true`, and `confirmation_status="not_run"`. Even `historical_robustness_passed` is diagnostic evidence, not support from an untouched holdout. No field or label authorizes paper/live/real deployment, and local CI completion is never alpha performance.

## 5. Planned implementation surface

Implementation begins only after approving Architect and later Critic reviews plus durable consensus.

1. `configs/research/alpha_max_portfolio_20260710.json` — versioned immutable runtime contract embedding the exact 21 canonical planning nodes, native rows, sidecars, splits/calendars, warmup, fixed refit, matrix, costs, gates, trials, ranking, safety.
2. `src/lumina_quant/strategies/alpha_max_research_sleeves.py` — three native-timeframe adapters, atomic near-high barrier, boundary finalization, and indicator capsules.
3. `src/lumina_quant/strategies/registry.py` — explicit research-only registrations; originals unchanged.
4. `src/lumina_quant/utils/artifact_read_receipt.py` and `src/lumina_quant/strategies/artifact_portfolio_mode.py` — one public standard-library-only frozen receipt/helper reused by runner and consumer; generic manifest/source receipt enumeration and copy preservation; default-neutral child capsule forwarding, keyword cadence default `60`/alpha value `1`, and default-empty non-manifest receipt parity.
5. `src/lumina_quant/data/feature_points.py` — default-neutral latest-point provenance accessor; existing value-only lookup unchanged.
6. `src/lumina_quant/backtesting/data_windowed_parquet.py` — optional path/exchange/injected-lookup passthrough plus observational emitted-raw as-of accessor, all with OFF parity.
7. `src/lumina_quant/backtesting/execution_model.py` — optional attribution sink inside canonical fill calculation.
8. `src/lumina_quant/backtesting/execution_sim.py` — explicit runner-owned default-OFF positive-pricing-trace activation, metadata forwarding, and separate zero-execution no-fill-attempt evidence.
9. `src/lumina_quant/backtesting/portfolio_backtest.py` — default-`None` application sink and funding-boundary resolver, post-reduce-only evidence including zero-applied fills, exact alpha fill anchoring/per-boundary settlement, and OFF parity. `src/lumina_quant/core/events.py` is explicitly not changed.
10. `src/lumina_quant/backtesting/backtest.py` — default-false alpha-only `strict_data_handler_construction` opt-in plus default-empty strict `portfolio_kwargs`/`execution_handler_kwargs`; legacy data-handler fallback remains unchanged even for nonempty legacy mappings, and portfolio/execution fallbacks remain when their new kwargs are empty.
11. `src/lumina_quant/research/alpha_max_evidence.py` — schemas, ordered bounded funding lookup and boundary resolver/ledger, hashing/calendar, frozen incumbent audit, universe/admission, exact ERC/HRP wrappers, canonical trial ledger, the sole canonical engine-manifest materializer, prelock selection, historical-evaluation report ranking, serialization.
12. `src/lumina_quant/research/alpha_max_engine_runner.py` — exact sealed-manifest constructor binding plus cadence/sink assertions, two-engine warmup/finalization, atomic native-clock raw-first orchestration, feature coverage, matrix, dual-layer reconciliation, evidence.
13. `scripts/research/run_alpha_max_prelock.py` and `scripts/research/run_alpha_max_historical_evaluation.py` — physically separate executables.
14. Focused unit/integration/process tests named in the test spec; existing golden/registry expectations update only where inventory changes.
15. `docs/research_note/alpha_max_portfolio_20260710.md` and `docs/research_note/alpha_max_portfolio_data_pc_handoff_20260710.md`.

No dependency, factory rewrite, event-schema change, legacy behavior change, deployment path, or broad engine redesign is allowed. Wider scope needs a leader-recorded amendment and fresh Architect/Critic review before edit.

## 6. Execution sequence

1. Complete durable Ralplan consensus and terminal planning handoff.
2. Create the isolated worktree at exact baseline; prove ancestry/contamination absence.
3. Initialize leader-owned Ultragoal artifacts and launch explicit `omx team` from the isolate.
4. Lock behavior with regression tests for adapter parity/forming-bar exclusion, atomic cross-sections, boundary finalization, sidecar OFF path, capsule resets, dual-layer trace OFF/ON parity, one-descriptor receipt default/generic-legacy/copy parity, and artifact consumption including transient swap-and-restore.
5. Implement adapters, feature passthrough, state forwarding, and trace seam in bounded ownership lanes.
6. Implement pure evidence/config/matrix logic, then two-engine runner and separate CLIs.
7. Add synthetic raw+feature integration, hostile historical evaluation tests, deterministic E2E; collect no market data.
8. Write research note and data-PC handoff.
9. Run targeted tests and every applicable local CI-equivalent job, including architecture, golden, coverage, benchmark, native, dashboard, and 8-GiB gates.
10. Run `ai-slop-cleaner`, rerun affected/full verification, then obtain distinct `code-reviewer` APPROVE and `architect` CLEAR plus invariant audit.
11. Lore commit, push only `feat/alpha-max-20260710`, monitor hosted CI, repair only in isolate, complete Ultragoal after green CI.

## 7. Expanded verification plan

### Unit

- evidence-tier non-escalation, deterministic schema/hash/calendar, missing-not-zero;
- exact complete native params, effective one-second outer cadence, completed-bar selection, cross-sectional atomicity/permutation invariance, dedup;
- latest-point provenance, emitted-raw as-of close, exact previous/current bounded-root order, cross-boundary newest-point selection, actual-LF prior-set bytes, frozen incumbent audit, and genuine-zero distinction;
- capsule allowlist/reset/hash and insufficient warmup rejection;
- exact universe/dates/folds/regimes, train-only admission, ERC/HRP golden vectors/caps/order, fixed refit and epsilon;
- manifest scaling/caps/cash, matrix completeness;
- exact canonical 21-node file/current-set hash plus prior trials/lineage/dedup, DSR/SPA/PBO, every MDD boundary;
- immutable positive pricing trace plus partial/flat/wrong-side/unchanged/conditional reduce-only application reconciliation, separate market/limit/conditional zero-execution attempts, and strict Backtest constructor-kwargs activation/OFF parity;
- strict runtime-field allowlist/read audit/hash, exact common seed, complete 4h metric primitive, DSR/SPA/PBO inputs, and report-only RPT/capacity formulas.

### Integration

- each adapter versus original on identical completed bars, including one atomic near-high batch;
- mixed daily/4h portfolio over raw 1-second fixtures with common 4h reporting;
- feature-aware handler with OFF parity and causal funding signals plus exact per-00/08/16-UTC cash settlement using independent boundary rate/close/quantity inputs;
- exact-boundary finalization equivalent to natural next-row promotion, then capsule to flat scorer with zero ghost state;
- positive-pricing/application/no-fill trace OFF/ON orders/fills/equity/RNG parity, one application record per positive pricing trace, and zero applications per no-fill attempt;
- all matrix rows/cost cells use identical roots/calendar and actual engine;
- valid manifests run; invalid hashes/caps/OOS/real/capsule fail closed.

### End-to-end/process

- prelock exposes/opens no historical evaluation resource;
- historical evaluation poison leaves prelock byte-identical;
- historical evaluation rejects pre-boundary data, overlap, hash/config/runtime-contract/capsule mismatch, mutation;
- all complete prelock rows and frozen historical-evaluation report rows replay once;
- arbitrary historical-evaluation metrics cannot change any prelock byte or selection/deployable field; leader disagreement requires a new untouched holdout;
- no-survivor emits stable `no_demonstrated_alpha`;
- two synthetic runs are byte-identical except isolated runtime metadata.

### Observability

Record baseline/worktree, command, seed, duration/RSS, source/config/runtime-contract/raw/feature/capsule/manifest/evidence hashes, exact split/native/report calendars and folds, frozen candidate/admitted universe, allocator method/parameters/input hash/weights, trials, class/params, refit, positive-pricing/application/no-fill traces including reduce-only zero-applied fills, exact metric/statistical inputs, report-only turnover/RPT/capacity, exact costs, coverage, exposure/clipping, orders/fills/trades, `prelock_champion`, report-only `historical_evaluation_leader`, rejection codes, ruin, immutable output paths.

## 8. Architecture invariants

1. Exact-baseline isolate; no shared-session or `.omc` contamination.
2. Factory/proxy/diagnostic evidence never selection-valid.
3. The outer portfolio receives every eligible one-second decision window; daily hypotheses remain daily, near-high evaluates one complete lexicographic admitted-universe batch per key, carry remains completed 4h, and 4h reporting is not a signal clock.
4. Carry decisions use only the exact immediately-previous/current bounded hashed sidecar pair, greatest eligible source timestamp at bar-close, never a later root; missing is not zero.
5. Alpha funding cash settlement enumerates every crossed 00/08/16 UTC boundary separately, uses only causal bounded rate and emitted raw close at that exact boundary plus the pre-boundary held quantity, records/reconciles one payment per boundary, and never reuses a current snapshot; resolver `None` preserves baseline.
6. Every finalizable pre-boundary working bucket is consumed once before indicator-only capsule extraction; fresh scorers have no ghost economic state or transferred generic aggregator state.
7. Immutable positive-execution pricing and post-reduce-only application traces reconcile one-to-one, including reduce-only zero-applied fills; zero-execution handler attempts have neither. Only strict nonempty Backtest constructor kwargs activate alpha sinks/resolver; default callers remain OFF with unchanged fill/funding/RNG/equity behavior.
8. Every selectable matrix row/cost cell is complete matched raw-first evidence through the sole actual-engine manifest materializer; the three frozen unavailable incumbents never acquire proxy evidence.
9. Prelock has no historical-evaluation dependency.
10. The ten-symbol candidate trial identity is distinct from the single sealed 5–10-symbol admitted execution tuple; manifest children, Backtest, handler, Portfolio, adapters, funding, and near-high share that admitted universe exactly. Admission predicate, UTC dates/folds/regimes, ERC/HRP numerics/caps/order, and epsilon are fixed; train fits/admission, validation selects the sole champion/fixes gross, fixed refit uses only train+validation, historical evaluation reports only.
11. One reporting calendar per split; full-event MDD retained; no missing zero-fill.
12. Trial accounting is the exact baseline G004 1466-node set serialized with real LF `0x0a` plus the canonical 21-node registry/file/current-key-set hashes, binding DSR to 1487 with no ambient source, alternate row identity, correlation discount, or status-based deletion.
13. Soft-MDD comparison includes every matched normal row in its prelock or historical-evaluation report domain; historical-evaluation results never enter selection/deployable state.
14. Scaled performance comes from actual replay, never arithmetic scaling.
15. The exact baseline incumbent-resolution artifact freezes all three named clean-incumbent attempts unavailable by source/report blob and content hash; runtime resolution, proxying, and inferred translation are forbidden.
16. Research-only adapters, legacy cadence default 60, legacy funding resolver `None`, default-empty non-manifest receipt tuples, generic legacy receipt IDs/counts and override-copy preservation, and all shared sink/feature/raw-accessor OFF paths remain unchanged; zero paper/live/real allocation.
17. The exhaustive no-`._rt` runtime contract, `LQ_` rejection, exact manifest construction, strict data-handler/lookup/resolver/sink identities, cadence, latency/order policy, seed, metrics, and report-only diagnostics admit no ambient alternative.
18. No local/documentation performance claim before data-PC evidence and no untouched-confirmation claim afterward without a genuinely fresh interval.

Independent final reviews must cite implementation/test evidence for all eighteen invariants.

## 9. ADR

### Decision

Build a compact native-clock Tri-Core experiment and treat only complete, matched raw-first actual-engine bundles as prelock-selection-valid. Use bounded adapters, atomic cross-sectional/boundary lifecycle rules, explicit funding sidecars, indicator-only split capsules, and default-OFF dual-layer cost evidence. Freeze universe, dates, allocator numerics, and the full matrix before separate report-only historical evaluation replay.

### Drivers

- Return priority with rational rules and bounded drawdown.
- Three mechanisms have independent economic bases and plausible diversification.
- Factory/proxy and common-clock conversions cannot faithfully measure them.
- Existing engine, metrics, allocators, and artifact consumer support bounded reuse.

### Alternatives considered

- Allocator-only: too constrained by weak evidence.
- Factory proxy: measures substitutes/incomplete nonlinear costs.
- Common 4h conversion: changes daily semantics.
- Broad factor zoo/router: excessive trials and scope.
- Passive cost reconstruction: cannot represent canonical clamp/randomization exactly.
- Carry without funding: economically invalid.

### Why chosen

This is the smallest architecture that preserves hypotheses, offers portfolio upside, measures nonlinear execution honestly, prevents historical evaluation influence, and permits an unfavorable outcome without tuning.

### Consequences

- Higher data-PC runtime/artifact size for full matrix × four cost cells.
- Seven shared files receive narrow default-neutral seams with parity gates.
- Validation selects exactly one prelock champion or none; historical-evaluation replay may report a different leader but cannot select it.
- Future tuning requires a new experiment and a genuinely uninspected future/withheld confirmation interval.

### Follow-ups

- Run frozen data-PC experiment once.
- Archive falsifiers/trial ledger regardless of result.
- Promotion/deployment is a separate future reviewed workflow.

## 10. Execution staffing and handoff

### Available agent types

`explore`, `researcher`, `dependency-expert`, `planner`, `architect`, `critic`, `executor`, `team-executor`, `test-engineer`, `debugger`, `verifier`, `code-reviewer`, `code-simplifier`, `writer`, `git-master`.

### Recommended Team staffing

- **Leader / Ultragoal owner (high):** isolate, ledger, shared-file integration, full verification, reviews, Git/CI, checkpoints.
- **Worker 1 — `team-executor` (high):** adapters, capsules, registry/artifact forwarding, parity tests.
- **Worker 2 — `team-executor` (high):** feature handler, cost trace seam, OFF/ON and reconciliation tests.
- **Worker 3 — `team-executor` or `test-engineer` (high):** evidence core, runner/CLIs, hostile process tests, config/docs; leader resolves overlap before dispatch.
- **Final native reviewers:** distinct `code-reviewer` and `architect`, high effort, after cleaner and post-cleaner verification.

Launch hint from isolate after Ultragoal initialization:

```bash
omx team 3:team-executor "Execute approved Revision-5.14 alpha-max plan. Preserve file ownership, exhaustive runtime/environment/metric/statistical/trial contract, frozen universe/root ownership/admission/dates/allocators, atomic native barriers and boundary finalization, sealed and pre/post filesystem-bound real-consumer manifest construction, alpha-only strict injected sidecar causality, indicator-only warmup, positive-fill/application/no-fill attribution parity, exposed-diagnostic historical-evaluation separation, total terminal-state precedence, full comparison matrix, and return terminal evidence to leader-owned Ultragoal."
```

Leader verifies panes/state, monitors `omx team status/await`, waits for `pending=0,in_progress=0,failed=0`, integrates/tests, then shuts down. Workers never own/checkpoint Ultragoal.

### Team verification path

1. worker targeted evidence;
2. leader integration/diff and combined tests;
3. full local CI-equivalent suite;
4. changed-file cleaner and post-cleaner rerun;
5. invariant audit and independent code-reviewer/architect;
6. Lore commit, isolated push, hosted CI green;
7. final Codex goal completion and Ultragoal quality checkpoint.

`$ultragoal + $team` is pre-authorized. `$autoresearch-goal` is insufficient because implementation/CI are required. `$performance-goal` is software-runtime focused. `$ralph` remains an explicit later repair fallback.

## 11. Stop condition

Ralplan completes only when a role-specific Architect approves Revision 5.14, a later role-specific Critic approves it, and `.omx/plans/ralplan-consensus-alpha-max-independent-20260710.json` records both in order with `ralplan_consensus_gate.complete:true`. Until terminal handoff is persisted, no execution worktree or source implementation is authorized.
