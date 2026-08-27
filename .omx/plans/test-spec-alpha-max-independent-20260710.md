# Test Spec — Independent Return-First Native-Clock Tri-Core Portfolio (Revision 5.14)

## 1. Verification objective

Prove the implementation preserves all three native hypotheses, effective one-second outer cadence, atomic cross-sectional/boundary lifecycle, exact ordered bounded cross-root causal funding and indicator-only warmup, constructor-only immutable positive-fill pricing plus post-clamp application evidence and separate zero-execution attempts without behavior change, the exact immutable runtime/metric/statistical/canonical-current-node contract, exact universe/admission/root/date/allocator contracts, complete matched raw-first matrix replay, validation-only selection, and a physically separated but explicitly exposed-historical diagnostic evaluation. Local tests prove integrity only, not alpha magnitude or untouched confirmation.

## 2. Unit tests — evidence, gates, matrix

| ID | Behavior | Required assertion |
|---|---|---|
| U01 | Evidence tiers | identity/diagnostic/factory payload cannot serialize/deserialize as selection-valid. |
| U02 | Completeness | removing raw/report equity, exposure, cost, funding, coverage, capsule, ruin, matrix, or hash yields `incomplete_engine_evidence`. |
| U03 | Hashing | order/path normalization deterministic; raw/feature/config/runtime-contract/capsule/manifest mutation changes correct hash. |
| U04 | Calendars | exact daily/4h/full-event/report boundaries; duplicate, partial first bar, timezone, invalid/non-finite reject. |
| U05 | Missing not zero | absent raw/funding point is missing; genuine funding zero stays numeric zero. |
| U06 | Comparison domain | ranking/MDD rejects unequal raw root, feature root, universe, cost, seed schedule, or calendar hash. |
| U07 | Matrix completeness | exact 3 components + 5 full + 9 LOO + 3 clean-incumbent attempts + 1 Track-B diagnostic = 21 frozen rows. |
| U08 | Selection status | complete component/full/LOO rows may enter prelock selection; the three clean-incumbent attempts are frozen unavailable by the exact audit and never select; diagnostic rows never select; historical-evaluation rows only report. |
| U09 | Equal-weight | full row uses 1/3 family buckets and rejects missing required family without redistribution; LOO splits its two declared families; deterministic/permutation invariant. |
| U10 | Equal-risk | nominal-20-bps exact-calendar daily matrix; sorted ids; keyword-only `ERCPortfolio(max_iter=10000,tol=1e-10,cov_window=None)`; LW covariance; `>=252` rows; `.50/.70` caps; golden risk contribution; no quality/return/tilt/filter/fallback or validation/historical evaluation read. |
| U11 | Shrunk-HRP | exact private helper, analytic correlation shrinkage `True`, MLE divisor T, abs-corr `.60` greedy first-cluster linkage, sorted ties, inverse-variance splits, `.50/.70` caps, golden vectors; no scipy/sklearn/expected-return/fallback. |
| U12 | Gross | zero/tiny/.12/.27/.30/>1 boundaries use exact epsilon `1e-12` and clip `[.25,2.25]`; 27% labeled hypothesis. |
| U13 | Manifest scaling | weights/leaf gross/caps/portfolio cap/cash reconcile; provenance unchanged. |
| U14 | No analytical scaling | scaled row without independent actual-engine id/hash rejects. |
| U15 | Alpha before leverage | scaled rejects if 1x sibling fails or exposure-normalized 1x evidence non-positive. |
| U16 | Fixed refit | `final_weight_refit=true`, hashed, no CLI override; only equal-risk/HRP full+LOO weights change from exact train+validation daily inputs; policy/gross/params/caps/membership/equal-weight/prelock champion unchanged; no validation rescore. |
| U17 | Trials | exact immutable G004 prior blob yields 1466 nodes and actual-LF prior-key-set SHA-256 `3b078011040f89e8d788b2cef9214c58f687221104381e26a688a7f8cdbddd78`; the canonical current registry file hash is `cfe3a04620c52cc235d6f1cda1cac617ba30cd7327c753fc2f620d8250d51a4e`, its exact 21 nodes yield current-key-set hash `3a4791cf353abcb82f9717ce89ee16b9d73d84f431d5b058135046c2ba8e332b`; four costs are stress observations; binding union is exactly 1487 regardless of execution/status. |
| U18 | Conservative lineage | prior normalization with actual separator/trailer byte `0x0a` (never literal bytes `0x5c 0x6e`) plus every exact current key (`row_id`, implementation, source id, timeframe, full kwargs, members, allocation numerics/caps/fixed-weight policy, gross, omission), cosmetic exclusions, canonical JSON bytes, source-kind separation, unique counts, prior/current set hashes match §4.11 goldens; config embeds current `nodes` JSON-value identically. |
| U19 | Gate order | DSR/SPA/PBO/positive metrics/coverage/hash/funding/manifest/ruin precede MDD with stable reasons; turnover/RPT/capacity are absent from rejection and ranking. |
| U20 | Statistics | DSR `.90`, SPA `.05`, PBO `.50`; exact twelve validation weeks/three 28-day regimes and ten historical-evaluation report folds; positive-metric and degenerate cases exact. |
| U21 | Gate MDD | equals `max(full_event_mdd,reporting_4h_mdd)`. |
| U22 | Normal MDD | `<=.30` passes only after earlier gates. |
| U23 | Soft MDD | `(0.30,0.35]` passes only with nonempty matched normal set and strictly higher CAGR and Calmar than return-first best normal row. |
| U24 | Comparator universe | every matched eligible component/full/control/LOO/clean-incumbent normal row included; diagnostic/unavailable excluded. A mandatory pure selector unit fixture uses a synthetic faithfully replayable matched-domain incumbent exclusively outside Revision 5.7 frozen runtime and proves its comparator inclusion; the three exact frozen audit rows remain unavailable and excluded in every integration/E2E fixture. |
| U25 | Soft failures | empty comparator, equality, one-metric dominance, mismatched domain, or omitted stronger normal row rejects. |
| U26 | Hard MDD | `>.35` always rejects; `.35` follows soft rule. |
| U27 | Ranking | validation ranking fixes sole `prelock_champion`; historical-evaluation ranking only exposes `historical_evaluation_leader`; both use cumulative return, CAGR, Calmar, net Sharpe, lower MDD, lexicographic id. |
| U28 | Ruin | any nominal-30-bps liquidation or bankruptcy/wipeout rejects. |
| U29 | Terminal precedence | first-match outcome is exactly no champion -> `no_demonstrated_alpha`; champion historical evidence incomplete -> `historical_evaluation_incomplete`; complete champion gate fail -> `prelock_champion_historical_robustness_failed`; complete champion all-gates pass -> `prelock_champion_historical_robustness_passed`. |
| U30 | Serialization | stable output byte-identical across repeats/permutations; runtime metadata isolated. |
| U31 | Research-only | paper/live/real/promotion flags false; forbidden OOS provenance false. |
| U32 | Frozen universe/dates | exact ten-symbol lexicographic `candidate_symbols`, Binance linear USDT-perp assertion, admission thresholds, one sealed lexicographic 5–10-symbol `admitted_symbols` subset, half-open warmup/train/purge/validation/embargo/historical-evaluation dates; any override/substitute/shortening rejects. |
| U33 | Historical non-selection | arbitrary historical-evaluation metrics/leader poison leaves prelock bytes, selected id, manifests, weights, gross, caps, thresholds, trials, provenance, and deployable flags byte-identical. |
| U34 | Contract manifest | only canonical `alpha_max_contract_manifest.v1` with exact ten sorted candidate Binance perpetual/linear/non-inverse/USDT/base-volume/multiplier-1 records passes; admission separately freezes the 5–10 active subset without mutating candidate identity; missing/extra/duplicate/inferred/ambient/mismatched metadata or hash rejects. |
| U35 | Quote notional | for exactly 517 train days use sorted unique raw rows and `math.fsum(float64(close)*float64(volume))`; finite positive close, finite nonnegative volume, genuine zero included; aggregated-daily proxy, quote-volume substitution, missing 4h bucket, row-count synthesis, deletion, interpolation, or winsorization rejects. |
| U36 | Percentiles | explicit Hyndman-Fan type-7 `p=.50/.10` over all 517 values matches golden odd/even/interpolated/zero/boundary vectors and thresholds `20m/2m`; library default cannot silently change result. |
| U37 | Root ownership | phase roots own exactly their intervals; warmup alone then exact adjacent `[previous,current]` pairs are the only composite order; extra/nonadjacent/out-of-range/overlap/later-root read rejects, while the newest eligible `<=8h` prior-root point works across a boundary. |
| U38 | Allocation rounding | sorted weights use `_round(...,ndigits=10)` and `cash=1-math.fsum`; `0 <= cash <1e-9` persists, negative or `>=1e-9` rejects `allocator_rounding_invalid`; no reallocation. |
| U39 | Exposure/orthogonal fields | every historical report fixes `historical_exposure_status=committed_period_outcomes_observed`, `requires_fresh_confirmation=true`, `confirmation_status=not_run`; nullable leader, disagreement boolean, and exact incumbent enum never replace terminal outcome. |
| U40 | Runtime source isolation | CLI exposes no profile/runtime override; any `LQ_` key fails `ambient_lq_environment`; config/profile/`LQ_CONFIG_PATH`/merge poison cannot change the final no-`._rt` `AlphaMaxBacktestConfig`; undeclared runtime-field read fails `unfrozen_runtime_field`. |
| U41 | Runtime allowlist/hash | a read/source spy observes exactly the frozen §4.7.1 attribute set plus enumerated direct env/default-config sites; each value matches the contract and any byte change changes the sealed runtime hash. |
| U42 | Common RNG | exact SHA-256 payload/first-eight-byte/modulo rule matches golden seeds; row id changes nothing while split/fold or cost changes the seed; zero maps to one. |
| U43 | Primary metric stream | complete positive UTC 4h endpoints produce arithmetic returns from flat initial capital, annual periods 2190; missing/nonfinite/nonpositive/duplicate/interpolated/truncated/daily-substituted input rejects. |
| U44 | Canonical metrics | total return/CAGR/net Sharpe/Sortino/Calmar/volatility equal `portfolio.optimizer_core.metrics(...,2190)` with zero RF/target, `ddof=1`, negative subset, and no alternate implementation; full-event MDD remains separate. |
| U45 | Statistical inputs | DSR gets exact `num_trials=1487`, prior-set hash, pre-gate finite 30-bps nonannualized-Sharpe sample variance, and HAC true; missing/mutated prior blob or count/key collision fails before replay and no correlation discount is allowed. SPA is exactly 2000 rounds, frozen block rule, seed 12345; PBO is exact nominal-30-bps matched matrix with eight splits. |
| U46 | Drawdown/tail diagnostics | duration counts consecutive below-peak 4h endpoints and hours=`4*count`; type-7 5% VaR and worst-`max(1,ceil(.05*n))` ES match goldens and never gate/rank unless already named elsewhere. |
| U47 | Turnover/RPT diagnostics | turnover uses `math.fsum(abs(applied_qty*fill_price))`, RPT uses total ending-equity P&L; zero turnover yields null plus `undefined_zero_turnover`; mutations never alter eligibility/ranking. |
| U48 | Capacity diagnostic | per-positive-request formula, finite-positive min/type-7 p10/median, and `undefined_no_positive_order` match goldens; it never alters eligibility/ranking or authorizes deployment. |
| U49 | Latency/order policy | 1/1-bar latency, MKT default, allowed market, one-tick-worse explicit limits, one tick, and GTC match goldens; every boundary mutation changes runtime hash and the expected order/fill fixture. |
| U50 | Window/default bypass | explicit poll/window/decision=1, parity-v2 true, skip false, explicit injected feature lookup/exchange; monkeypatched `get_default_runtime_config` raises if called yet the run succeeds; omitted kwargs fail contract construction. |
| U51 | Effective cadence | legacy `ArtifactPortfolioModeStrategy` omitted-kwarg value stays 60; alpha passes kwarg 1 and config/argument/constructed attr all equal 1 before engine start; 60, zero, missing, post-mutation, subclass, or fallback mismatch rejects. |
| U52 | Funding composite | `get_latest_point` provenance preserves value-only parity; exact root spec hashes/bounds/order, newest-timestamp selection, inclusive-finalization query/end-exclusive source, equal-timestamp conflict, >8h, gap, nonadjacent, later-root, and historical embargo-hash cases match goldens. |
| U53 | Current trial registry | the planning file contains exactly the 21 named rows including all nine exact LOO ids and full component defaults; canonical keys/current-set/file hashes equal the frozen constants; any node field/default/member/allocation/gross/omission mutation fails before replay. |
| U54 | Strict sink/resolver factory | empty Backtest kwargs execute the legacy constructor/fallback path exactly; nonempty kwargs make one strict call and never fall back; alpha exact dicts install verified application sink, `AlphaMaxFundingBoundaryResolver(ordered_lookup,admitted_symbols)`, Portfolio-bars/lookup/admitted-domain identities, and execution sink flag, while missing/extra/post-construction mutation rejects. |
| U55 | Canonical engine manifest | golden 5-of-10 and 10-of-10 component/full/LOO/scaled payload bytes/path/hash, source row, candidate/admitted/admission fields, child class/candidates/active symbols/params/weights/caps, cash/gross, and exact `strategy_params` match §4.10.1. Frozen `ArtifactReadReceipt` goldens prove one open descriptor, JSON/hash from its exact byte string, requested/strict path, byte count, SHA, and unchanged pre/post `fstat` identity for manifest and config; non-manifest definitions default to an empty receipt tuple. Legacy `artifact_portfolio_manifest` fixtures with `survivors` and multiple lexicographically unsorted source IDs retain unchanged components/cash/economic behavior while exposing manifest-first then `source:<actual_id>` receipts sorted by unique id; nonempty component overrides preserve the identical receipt tuple object through every definition copy. Only the alpha runner requires exactly `artifact_portfolio_manifest`, `source:alpha_max_config`, and its successful consumer tuple must match the runner tuple exactly. In-place/same-byte rewrite, hard-link/atomic replacement, ancestor/target symlink swap, and a transient B swap that changes only discarded candidate/cap/source fields then restores A before construction returns each reject `portfolio_manifest_activation_mismatch` with zero replay events. Receipt omission/reorder, candidate/admitted mismatch, rejected-symbol activation, research-control kwarg, direct leaf, post-resolution filter, default/ambient path, in-memory seam, overwrite, or fallback also rejects. |
| U56 | Strict data-handler factory | default `strict_data_handler_construction=false` preserves baseline empty and nonempty-mapping TypeError fallback exactly; true requires nonempty kwargs, makes one call, and never falls back. Alpha's exact true flag/six-key mapping installs the ordered lookup by identity; missing/extra/path-nonempty/ignored/replaced/post-mutation rejects. |
| U57 | Incumbent resolution freeze | exact audit file SHA-256 `5133bc40116399fe7af32e75a1ecc52a4f385dc8a0b5d3a4a9585e2437615ed8`, baseline commit, ordered paths, Git blob ids, content SHA-256s, occurrence evidence, and three unavailable reasons match; config embeds its JSON value identically. |
| U58 | Incumbent fail closed | Track-A offline selector, non-unique lead-lag family, and fold-local Optuna hybrid cannot become resolvable via glob/latest/proxy/inference/runtime translation; each remains counted but excluded from selection/MDD. |
| U59 | Boundary enumeration | exact floor-index rule emits every crossed 28800-second 00/08/16 UTC boundary once in ascending order, advances anchor to each boundary, and never advances to current snapshot. |
| U60 | Boundary point rules | every boundary independently uses causal funding point age `<=28800000ms`, emitted raw positive close with `close_ts=row_start+1000<=boundary` and age `<=1000ms`, and pre-boundary signed quantity; missing/stale/future/duplicate/identity mismatch fails. |
| U61 | Funding payment ledger | canonical `compute_funding_payment(...,periods=1)` per boundary, immutable source/quantity/rate/price/payment rows, uniqueness, full all-held-symbol batch validation-before-mutation, and `math.fsum(payment)` cash/total/funding reconciliation match goldens. |

## 3. Native adapter and funding tests

| ID | Behavior | Required assertion |
|---|---|---|
| S01 | Exact rows | exact complete canonical component kwargs (including non-tunable allocation/order/min-price defaults), daily trend `28/56/84`, daily `wk52`, 4h `swing_ls`; no omitted default, conversion, sweep, undeclared field, or fallback. |
| S02 | Tier safety | `ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy`, `ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy`, and `ResearchOnlyFourHourFundingHarvestCarryStrategy` are explicit `research_only`; originals retain baseline tier/default behavior. |
| S03 | Forming exclusion | aggregator includes forming bucket; poison mutations to forming OHLC do not enter history/capsule/signal. |
| S04 | Completed dedup | repeated raw events in native bucket do not reprocess completed key. |
| S05 | Trend parity | adapter/original on identical completed daily bars have identical indicator/decision traces aside from id/provenance. |
| S06 | Near-high parity | same daily parity including `_tick`, rebalance, hold cadence, cross-sectional scores. |
| S07 | Carry parity | same completed-4h parity with identical causal funding. |
| S08 | Mixed clocks | daily children do not advance on 4h report ticks; carry only completed 4h; report calendar changes no decisions. |
| S09 | Timeframe union | mixed portfolio exposes required 1d and 4h aggregation without converting either child. |
| S10 | Bar-close as-of | bucket start maps to +4h, not after watermark, latest eligible feature chosen. |
| S11 | Future/stale poison | post-close future funding cannot change decision; stale beyond coverage fails rather than backfills indefinitely. |
| S12 | Missing versus zero | missing gives `None`/coverage failure; exact `0.0` remains valid observation. |
| S13 | No ambient feature | absent explicit root/exchange fails; environment/ambient files cannot change lookup. |
| S14 | Lineage/no duplication | adapter hashes link original; no duplicated indicator/position formula body. |
| S15 | Near-high atomic batch | one completed daily bar per frozen admitted symbol is sorted and passed to inherited `calculate_signals_window` exactly once; `_tick`/eval key advance once. |
| S16 | Arrival permutation | all symbol/window arrival permutations and identical duplicates produce byte-identical batch, signals, scores, state, and decision trace. |
| S17 | Staggered barrier | no evaluation occurs until the last admitted symbol arrives; prior symbols never compare with stale peer state. |
| S18 | Incomplete cross-section | watermark past close with one/many missing admitted symbols fails `incomplete_near_high_cross_section`; no stale carry, universe shrink, or later repair. |
| S19 | Conflicting duplicate | identical duplicate is idempotent; conflicting OHLCV for same symbol/key fails. |
| S20 | Forming poison at barrier | forming or post-close poison cannot enter the completed atomic batch or change the inherited decision. |
| S21 | Outer cadence | sparse 1-second windows and the first window after each 4h/daily boundary reach the portfolio wrapper immediately at the next eligible second; native children still advance only on completed native bars. |
| S22 | Wrapper parameter separation | component node params reach only its child; full/LOO research-control params are not passed; the wrapper receives exactly manifest path plus cadence 1 and no other strategy kwarg. |

## 4. Warmup and state tests

| ID | Behavior | Required assertion |
|---|---|---|
| W01 | Minimum history | fewer than 366 completed 1d or 64 completed 4h bars rejects. |
| W02 | Alignment | score starts UTC 00:00; non-day/non-4h or partial first native bar rejects. |
| W03 | Warmup silence | zero scoring equity, orders, fills, trades, fees, funding, liquidation. |
| W04 | Capsule allowlist | `get_research_indicator_state`/`set_research_indicator_state` transfer only declared histories/dedup/cadence; economic mode/entry/stop/adds/held/positions/orders absent/reset. |
| W05 | Capsule determinism | same causal prefix gives identical capsule/hash; future/historical evaluation raw or feature poison cannot change it. |
| W06 | Flat scorer | restored scorer starts flat cash/positions/orders/margin, empty aggregator, no ghost mode. |
| W07 | Trend state | OHLC/week/dedup keys preserved; mode OUT, entry/score/held reset, cooldown satisfied. |
| W08 | Near-high state | close/high, `_tick`, eval/time keys preserved; modes/entry/held/scores reset. |
| W09 | Carry state | OHLC/funding/native keys/`prev_roc` preserved; mode/entry/stop/watermarks/adds/held reset. |
| W10 | Portfolio forwarding | child capsules keyed by stable child id; missing/extra/wrong hash rejects. |
| W11 | Shared OFF parity | existing artifact portfolio unchanged when research capsule APIs unused. |
| W12 | Split causality | validation uses allowed prior data; historical evaluation capsule exists before historical evaluation root. |
| W13 | Cost independence | each cost cell restores same capsule hash in fresh engine under same seed schedule; no state/RNG carry. |
| W14 | Exact boundary flush | a temporally complete final working 1d/4h bucket with no next-bucket row is consumed before capsule extraction. |
| W15 | Natural-promotion parity | explicit boundary finalization and natural next-row promotion yield identical adapter state/capsule/first scorer decision while finalization consumes no next-row data. |
| W16 | No drop/dup | each finalizable key is consumed exactly once across ordinary and boundary paths; near-high finalizes one atomic batch. |
| W17 | Partial rejection | `bucket_start + timeframe > watermark` is never finalized; requesting capsule/scoring across it fails without generic aggregator transfer. |
| W18 | Historical evaluation boundary | prelock finalizes all allowed train+validation/embargo buckets and funding at 4h close before historical evaluation root exists; poison historical evaluation cannot change capsule. |
| W19 | Cross-root boundary funding | previous-root point at boundary-minus-one-second is selected when current has none; a newer current point wins; later/nonadjacent roots are unopened; historical opens only sealed embargo plus historical. |

## 5. Shared-path regression and exact cost tests

Use deterministic maker/taker, partial, conditional, clamp, funding, liquidation, liquidity fixtures.

| ID | Flow | Required assertion |
|---|---|---|
| C01 | Feature OFF | `feature_db_path=None`, `feature_exchange=None`, and `feature_lookup=None` are byte/numerically identical to baseline windowed handler and existing value-only `FeaturePointLookup`. |
| C02 | Feature ON | exact true strict flag plus six-key `data_handler_kwargs` construct the windowed handler once, whose parent receives empty path/`binance` before the injected ordered lookup is installed by identity; bounded children/root id/inventory/content/exchange/bounds/query coverage match and OHLCV/default-runtime behavior is unchanged. |
| C03 | Trace OFF | default `ExecutionModel.compute_fill(..., attribution_sink=None)`, `FillResult`, orders, fills, equity, next RNG state, event schema match baseline golden. |
| C04 | Trace ON | only metadata/evidence differs; price/qty/commission/events/equity/next RNG/later fills equal OFF. |
| C05 | Taker | applied base slip, half-spread, impact, clamp adjustment, commission reconcile. |
| C06 | Maker | maker fee/status and components reconcile without taker substitution. |
| C07 | Partial | requested/executed/unfilled qty, participation denominator, impact, liquidity cap reconcile for maker/taker. |
| C08 | Conditional | STOP/TAKE_PROFIT/TRAIL_STOP order kind/trigger recorded; costs reconcile. |
| C09 | 99% clamp | pre/post penalty and clamp adjustment exactly explain realized price without double count. |
| C10 | Funding/financing | separate canonical charges reconcile to equity, not folded into price penalty. |
| C11 | Liquidation | execution/fee/terminal state reconcile separately and reach ruin gate. |
| C12 | Metadata absent | trace metadata absent when OFF; no `FillResult` or `core/events.py` schema change. |
| C13 | Sink failure | sink exception fails loud before evidence acceptance, never yields silently incomplete selection evidence. |
| C14 | Explicit activation | only exact `resolver=AlphaMaxFundingBoundaryResolver(ordered_lookup,admitted_symbols)` plus nonempty `Backtest(portfolio_kwargs={"fill_application_attribution_sink":collector.record_application,"funding_boundary_resolver":resolver},execution_handler_kwargs={"record_cost_attribution":True})` strict construction installs verified sink/resolver/allowed-domain identities and execution flag; no fallback, config/env/global/class/post-construction mutation can activate them. |
| C15 | Application unchanged | already-within-position/non-clamped fill has model=applied qty/cost/commission, scale `1`, status `applied_unchanged`, one immutable pricing-hash link. |
| C16 | Partial reduce-only clamp | applied quantity/cost/commission and scale exactly match the clamped event while the immutable pricing trace retains model values. |
| C17 | Flat rejection | reduce-only fill while flat books no trade but emits one zero-applied record with zeros and `reduce_only_flat`. |
| C18 | Wrong-side rejection | reduce-only fill increasing/reversing exposure books no trade but emits one zero-applied record with `reduce_only_wrong_side`. |
| C19 | Conditional/remainder linkage | STOP/TAKE_PROFIT/TRAIL_STOP and remainder/partial lineage survives pricing and application records through clamp/reject. |
| C20 | One-to-one reconciliation | every positive-execution pricing trace has exactly one application record, including reduce-only zero-applied records; every application points to one pricing trace; handler `no_fill_attempt` records have neither. |
| C21 | Dual-layer OFF/ON parity | enabling both sinks changes only metadata/evidence; event sequence, prices, quantities before portfolio policy, applied portfolio state, equity, next RNG, and later fills equal OFF. |
| C22 | Zero-volume market | a called market calculation with zero liquidity emits one `liquidity_cap_zero_market` no-fill attempt and no pricing trace, `FillEvent`, application, trade, fee, or equity mutation; RNG/remainder behavior matches OFF. |
| C23 | Zero-volume crossed limit | a crossed limit with zero liquidity emits one `liquidity_cap_zero_limit` no-fill attempt; a non-crossed limit that never calls `compute_fill` emits none; pending/remainder behavior matches OFF. |
| C24 | Zero-volume conditional | each triggered STOP/TAKE_PROFIT/TRAIL_STOP zero-liquidity calculation emits one `liquidity_cap_zero_conditional` with preserved lineage and no positive-fill artifacts; untriggered orders emit none. |
| C25 | Zero-fill determinism | ON/OFF orders, pending/remainder state, next RNG, later fills, and equity are identical across zero-fill market/limit/conditional fixtures; no-fill sink failure is loud. |
| C26 | Funding resolver OFF | `funding_boundary_resolver=None` preserves baseline `_apply_funding`, lazy fill anchor, current-bar/default behavior, multi-period call, cash/equity, and serialized state byte/numerically. |
| C27 | Raw as-of accessor | canonical 1s row start maps to `close_ts=start+1000`; greatest already-emitted completed close `<=query` returns value/row-start/close timestamps. A row starting at query is excluded; future/unemitted, feature/default source, nonfinite/nonpositive close, and unsupported field cannot satisfy alpha coverage. |
| C28 | Boundary future poison | funding at `boundary+1ms` and a raw row starting at boundary (close `boundary+1000ms`) cannot alter the boundary charge; the completed row starting at `boundary-1000ms` can. |
| C29 | Multiple boundaries | sparse jump across two or more funding epochs makes distinct rate/close queries and one `periods=1` payment per boundary; current/latest rate or price is never reused. |
| C30 | Fill ownership | entry one second before a boundary pays; zero-to-nonzero or flip exactly at/after boundary anchors there and is not retrocharged; same-sign add/reduce retains anchor; flat clears it. |
| C31 | Funding failure atomicity | missing/stale/future/mismatched rate or close fails `funding_boundary_coverage` before any cash/ledger/anchor mutation; no partial batch or silent advance is allowed. |

## 6. Actual-engine integration tests

Use deterministic raw-first 1-second/windowed partitions plus explicit feature sidecars, liquidity/ADV, intrabar extremes. Run real data handler, engine, simulated execution, portfolio, strategies.

| ID | Flow | Required assertion |
|---|---|---|
| E01 | True components | each yields complete evidence with true class/source/params/capsule hashes; factory never imported/called. |
| E02 | Mixed portfolio | daily/4h children over raw events match standalone native adapter decisions; near-high remains atomic under shuffled symbol partitions. |
| E03 | Full history | full-event plus common-4h equity/returns/holdings/exposure/turnover/orders/fills/trades complete/reconcile. |
| E04 | Funding | exact adjacent-root composite rates are causal/newest at carry bar-close and each 00/08/16 cash boundary; each boundary uses its own emitted close/pre-boundary quantity/payment ledger over exactly admitted symbols; the resolver exposes the identical admitted tuple and a direct or held rejected-candidate request fails before lookup/mutation, while missing, stale, future, conflicting, later-root, wrong-hash, or unreconciled funding rejects. |
| E05 | Warmup handoff | exact-boundary finalization, correct first decisions, and zero ghost state for component/multi-child portfolio. |
| E06 | Nonlinear gross | 1x/scaled differ through sizes/costs/caps/equity; not arithmetic multiples. |
| E07 | Clipping | target/realized gross and every cap clip recorded; unreported clip invalid. |
| E08 | Liquidation | intrabar extremes trigger modeled liquidation/wipeout with complete terminal fields. |
| E09 | Cost grid | 10/15/20/30 independent runs, base slips 5/10/15/25, distinct hashes, sqrt impact/funding intact. |
| E10 | Matrix | each complete eligible row shares roots/universe/calendar/seed semantics and four cells; missing row stable status. |
| E11 | Incumbent mapping | the exact frozen baseline audit yields `incumbent_replay_unavailable` for Track-A, `cross_asset_lead_lag_momentum`, and `cross_candidate_hybrid:hybrid_v3_5`; all remain explicit/countable on four cells, none materializes or selects, and no nearby proxy/runtime resolution is attempted. |
| E12 | Validation no prune | every frozen row/status reaches prelock regardless of validation champion. |
| E13 | Determinism | fixed inputs/config/seed/capsule gives identical stable evidence. |
| E14 | Frozen allocator inputs | 20-bps component daily-return hash, sorted ids, observation count, covariance/shrinkage parameters, caps, raw/rounded weights, and cash residual are complete/reproducible. |
| E15 | Admission freeze | 5-of-10 and 10-of-10 train fixtures seal one candidate/admitted mapping; validation/historical-evaluation liquidity, funding, missingness, or new symbol files cannot change it. Manifest, Backtest, handler, Portfolio, adapters, resolver allowed domain, funding, and near-high all use the exact admitted order/identity where specified; affected admitted rows fail rather than substitute, while rejected candidates cannot emit signals, funding, positions, orders, fills, or trades. |
| E16 | Admission computation | actual windowed raw partitions produce the same 517 daily quote-notional vector/type-7 statistics/content hash as pure golden computation; manifest hash and per-symbol reasons are recorded. |
| E17 | Owned-root replay | engine opens only the current raw root and exact previous/current feature lookup pair in chronological order, never historical/nonadjacent roots during prelock, and records root-id/inventory/content/interval hashes. |
| E18 | Real manifest construction | every resolvable current component/full/LOO/scaled row constructs the real `ArtifactPortfolioModeStrategy` before engine start. In 5-of-10 and 10-of-10 fixtures, the actual consumer parses the manifest from one descriptor byte string, validates config SHA from one config descriptor byte string, and the alpha runner rejects unless `artifact_kind` is exact and the consumer exposes exactly the ordered manifest/`source:alpha_max_config` pair matching the runner-sealed receipts. Generic legacy manifest receipt cardinality is not constrained by this alpha assertion. Immediately before the first replay event, constructed `portfolio_mode`, manifest/config source paths, retained child ids/classes/params/weights/cash and strategy-derived native timeframes match prevalidated execution fields; exact candidate/caps/source metadata are proven by consumer-receipt byte/SHA equality plus no fail-closed reason. A real transient swap-and-restore hook presents B only during consumer open, changes discarded candidate/cap/source fields, restores A before construction returns, and must reject with zero replay events. Active child/proxy/adapter symbols equal the sealed admitted subset and Backtest/handler/Portfolio identities pass; no rejected symbol, direct leaf, filter, or default path exists. |
| E19 | Funding engine chronology | real handler/Portfolio event ordering settles pre-existing quantity at each exact boundary before same-timestamp fills, uses exact resolver/Portfolio-bars/lookup/admitted-domain identities plus bound raw-accessor owner/function, rejects an outside-domain symbol before any resolver lookup/mutation, and reconciles ledger to full-event equity under sparse multi-boundary fixtures. |
| E20 | Trend liquidity falsifier | train-frozen liquidity buckets produce deterministic per-bucket nominal-30-bps trend contribution; liquid-bucket failure or edge confined to weakest liquidity emits `trend_mechanism_not_supported` and cannot be described as broad momentum. |
| E21 | Scaling attribution | each scaled row reports exact matched delta versus its 1x sibling, labels it `risk_transform_not_alpha`, discloses absent passive-scaled counterfactual, and remains ineligible when 1x exposure-normalized evidence is non-positive. |
| E22 | Statistical role audit | DSR selection-bias inputs, synchronized CSCV/PBO rank-degradation matrix, and fixed-benchmark SPA loss/bootstrap inputs are separately serialized and never collapsed into one overfit-pass claim. |

## 7. Manifest consumer integration

| ID | Flow | Required assertion |
|---|---|---|
| M01 | Valid component/full/LOO/scaled | intended children, positive components, gross/cash/caps, native timeframes, capsule ids resolve. |
| M02 | Source/hash/freshness | mismatch fails to cash or runner rejects stably. |
| M03 | OOS/real mutation | fails closed. |
| M04 | Leaf/netting/portfolio caps | breach fails closed. |
| M05 | Invalid/empty child/capsule | fails closed. |
| M06 | Gross >2.25 | rejects. |
| M07 | Existing artifacts | new unused APIs preserve golden decisions byte/numerically. |
| M08 | Baseline child provenance gate | real baseline consumer accepts only the exact child `no_current_fold_oos_provenance=true`, `train_validation_optimizer_provenance=true`, `lagged_completed_shadow_optimizer_provenance=false` plus allowed provenance objects; deleting/flipping any required field or setting any forbidden OOS/real flag fail-closes with the exact reason. |

## 8. Physical prelock/historical evaluation process tests

Subprocess/filesystem-audit tests, not pure substitutes.

| ID | Flow | Required assertion |
|---|---|---|
| P01 | Prelock surface | help/parser/schema expose no historical evaluation raw/feature path, inventory, hash, metadata. |
| P02 | Explicit inputs | prelock requires only config, canonical contract manifest, exchange/output, and warmup/train/purge/validation/embargo raw+feature pairs; profile/runtime overrides, historical arguments, and ambient paths are absent/rejected. |
| P03 | Historical evaluation poison | add/remove/rename/chmod/touch/content poison of raw/feature trees leaves all prelock stable bytes/hashes identical. |
| P04 | Immutable prelock | rerun refuses overwrite; historical evaluation reads prelock only, cannot mutate. |
| P05 | Hash match | config/runtime-contract/source/raw/feature/capsule/membership/policy/gross/manifest mismatch rejects before replay. |
| P06 | Boundary capsule | exists/hashes before historical evaluation command and is inventory-independent. |
| P07 | Post-boundary only | pre-boundary raw/feature or overlap/purge/embargo violation rejects. |
| P08 | Matrix freeze | all 21 manifest-or-unavailable/diagnostic statuses, including the exact three incumbent-audit unavailable records, are sealed before historical evaluation. |
| P09 | One-touch append | historical evaluation writes new tree; duplicate completed id refuses rerun. |
| P10 | Historical poison scope | historical-evaluation values change only the historical-evaluation report/orthogonal fields; every prelock byte and selection/deployable field remains identical. |
| P11 | Replay once | every resolvable frozen prelock/report row replays exactly once; three frozen unavailable incumbents and diagnostic Track-B never invoke an engine constructor. |
| P12 | No survivor | all reject terminates `no_demonstrated_alpha`, zero promotion, complete ledger. |
| P13 | Leader disagreement | non-champion historical-evaluation leader leaves `selected_candidate_id=prelock_champion`, sets disagreement true, retains always-true fresh-confirmation requirement, and cannot emit a deployable-selection change. |
| P14 | Frozen chronology | exact dates/folds/regimes/purge/embargo and endpoint required; overlap, shortening, shifted boundary, partial final endpoint, or CLI override rejects before replay. |
| P15 | Exposed provenance | help/schema/output/note never call the interval untouched/locked/prospective/confirmatory; committed-artifact exposure status is mandatory and immutable. |
| P16 | Collision precedence | champion failure plus different leader plus unavailable incumbent, no survivor plus unavailable incumbent, and pass plus different leader each preserve the ordered singular outcome while emitting only orthogonal disagreement/incumbent fields. |
| P17 | Root isolation | files outside each owned interval, purge/embargo hidden in adjacent roots, gap/overlap/duplicate timestamps, nonadjacent/later root injection, embargo-hash mutation, or contract-manifest mutation reject before scoring; historical-root poison cannot affect prelock. |
| P18 | Environment poison | each baseline-discovered `LQ_` override plus arbitrary unknown `LQ_FOO` fails before handler/engine construction; a clean environment produces identical stable bytes across config/profile/default-runtime poison outside the process. |
| P19 | Trial inventory isolation | worktree/newer/output/`.omc` candidate artifacts cannot change 1487; only the exact baseline G004 blob plus embedded canonical current nodes are used, and prior/current file/key-set identities are sealed. |
| P20 | Constructor activation isolation | ordinary Backtest and every non-alpha runner create no cost metadata; every selectable alpha replay constructs both sinks through exact kwargs before start and fails if a test double rejects or ignores either kwarg. |
| P21 | Current registry isolation | runtime never reads `.omx`; config-embedded nodes equal the frozen planning JSON values/hash and each node `symbols` remains the ten-symbol candidate trial identity. The separate admission artifact alone supplies active `admitted_symbols`; rename/reorder is deterministic where cosmetic, but any behavioral node, candidate/admitted mapping, or alternate LOO/source id mutation fails before engine construction. |
| P22 | Historical composite surface | historical CLI requires sealed prelock, explicit embargo feature root, historical raw/feature roots, exchange/output only; embargo re-hash must match prelock and no embargo raw/nonadjacent feature/profile/runtime argument is accepted. |
| P23 | Manifest activation isolation | prelock seals both phase manifest sets, config, runner receipts, exact bytes/SHA, source paths, and constructor mappings; historical can open only sealed `prelock_final_refit` manifests. Hostile hooks rewrite in place/identically, atomically replace/hard-link, swap an ancestor/target symlink, alter consumer source/definition, or present manifest/config B only during the actual consumer descriptor open and restore A before construction returns. Consumer-receipt mismatch catches the transient case even when B preserves retained execution fields and changes only discarded candidate/cap/source metadata. Every case rejects `portfolio_manifest_activation_mismatch` before the first replay event and emits no market/funding/order/fill/trade event. Alternate paths, alpha receipt cardinality/id/order errors, missing/reordered receipts, source mismatch/staleness, validation manifest reuse, definition-copy receipt loss, or consumer-definition mismatch likewise rejects; generic legacy source cardinality remains valid. |
| P24 | Funding lookup activation isolation | every phase seals/asserts the exact true strict flag, `AlphaMaxOrderedFundingLookup` identity, and ordered root-pair contract before engine start. A rejecting handler, alpha Backtest fallback, ambient path, copied/replaced lookup, missing root, or later/nonadjacent root fails before scoring; default-false legacy nonempty fallback remains green. |
| P25 | Funding resolver activation isolation | every alpha engine seals/asserts exact Portfolio resolver constructed with the ordered lookup and identical admitted tuple, `portfolio.bars is data_handler`, ordered feature-lookup and resolver admitted-domain identities, and bound raw-accessor owner/function before start. Ten-candidate-root inference and outside-domain requests reject before lookup/mutation. Missing/extra/replaced/copied/post-mutated resolver or accessor fails; ordinary callers with resolver `None` remain baseline-green. |
| P26 | Incumbent audit isolation | runtime reads no `.omx`, glob, report-latest alias, or worktree artifact; config-embedded audit equals the frozen planning value/hash and any path/blob/content/status/reason mutation rejects before row construction. |

## 9. Fitting/refit/selection workflow tests

1. Train alone determines admission and initial policy weights.
2. Validation replays train-fitted 1x and fixes scaled gross plus the sole `prelock_champion`.
3. Scaled validation rows are independent actual-engine runs.
4. Fixed refit recomputes only equal-risk/HRP full+LOO weights from combined train+validation among frozen members.
5. Equal-weight, components, policy, gross, params, caps, membership, validation metrics, and `prelock_champion` unchanged; no post-refit validation rescore.
6. All final rows/statuses and historical evaluation capsule freeze before historical evaluation.
7. Historical evaluation replays frozen matched matrix once and applies trials/statistics/ruin/MDD only to a separate report leaderboard; selection remains the prelock champion.

Mutations:

- validation may affect validation metrics/gross/refit weights, never train admission/initial fit;
- historical evaluation changes only historical-evaluation metrics/report labels and never selected/deployable state;
- missing historical evaluation member/feature rejects and cannot substitute/shrink universe;
- no row compares on unavailable calendar/root/cost domain.

## 10. End-to-end hostile fixtures

Run twice in clean temp roots; stable outputs byte-identical:

- three families valid with different native decision times;
- forming daily/4h bars contain extreme poison;
- near-high symbols arrive staggered/permuted/duplicated with one incomplete and one conflicting cross-section;
- final warmup working bucket has no next-row flush, plus a genuinely partial boundary bucket;
- future/stale/genuine-zero/missing funding;
- warmup would be virtually positioned without capsule reset;
- one valid family, missing required full family, LOO winner, control winner, all three incumbents unavailable and therefore unable to win, no valid family;
- all rows fail costs/statistics;
- scaled highest raw return with failing 1x;
- full-event MDD exceeds 4h and vice versa;
- 36% row; 34% row dominates CAGR only, Calmar only, neither, both;
- stronger normal component/control changes soft admission; each exact frozen incumbent remains unavailable and cannot enter the comparator;
- soft rows with empty normal set;
- nominal-30-bps liquidation/wipeout;
- maker/taker/partial/conditional/99%-clamp positive pricing plus unchanged/scaled/flat/wrong-side zero-applied reconciliation and separate zero-liquidity market/crossed-limit/triggered-conditional no-fill attempts;
- realized gross clipped;
- prelock config/runtime-contract/source/raw/feature/capsule mutation plus profile/environment/merge poison;
- runtime read outside the strict allowlist, any `LQ_` environment key, default-runtime fallback, latency/order/window mutation, and every frozen seed/metric/statistical boundary;
- exact G004 blob/node normalization/prior-set hash/1466+21=1487 trial boundaries plus ambient/newer-artifact poison;
- zero-turnover RPT and no-positive-order capacity diagnostics, including proof that diagnostic mutations cannot affect gates/rank;
- historical evaluation pre-boundary point;
- historical-evaluation metrics are arbitrarily poisoned so a non-champion leads while every prelock/selected byte remains fixed;
- allocator id order, exact-return calendar, zero-variance column, insufficient observations, and cap fixtures;
- 517-day raw notional vectors with zero-volume rows, type-7 interpolation threshold edges, missing 4h bucket, forbidden daily-close proxy, and both 5-of-10/10-of-10 candidate-to-admitted activation with rejected-candidate event/funding/order poison;
- contract manifest extra/missing/duplicate/mismatched records and each root-ownership violation;
- rounded allocation residual negative, zero, just below `1e-9`, exactly `1e-9`, and above;
- all frozen-runtime terminal-state collisions across leader disagreement and the incumbent-unavailable status;
- all three exact audit-frozen incumbent-unavailable statuses and the Track-B diagnostic;
- complete no-survivor ledger.

## 11. Observability/audit assertions

Every run includes:

- baseline, branch/worktree, command, seed, duration, peak RSS;
- experiment/source/runtime-contract/raw/feature/exchange/contract-manifest/capsule/manifest/evidence hashes;
- exact split/fold/regime dates, purge/embargo, native completed-bar/barrier/finalization keys and coverage, full-event and 4h calendar hashes;
- contract universe/metadata records, per-root ownership interval/inventory, 517-day quote-notional vectors/type-7 statistics, train admission inputs/reasons/frozen membership, class/params, matrix row/status/trial/lineage/dedup ids;
- warmup interval/capsule hash/flat-start proof, fixed-refit flag;
- allocator input cost/frequency/ids/observation hash/method/numerics/caps/raw+rounded weights/cash; per-cost nominal/actual costs, positive-pricing/application/no-fill trace schemas and counts, zero-applied reconciliation, per-boundary funding rate/price-row-start/price-close/quantity/payment ledger and cash/equity reconciliation, incumbent audit hash/status, metric/statistical inputs, report-only turnover/RPT/capacity, coverage, exposure/clipping, trades;
- full-event and 4h MDD plus selected max;
- `prelock_champion`, immutable selected id, report-only nullable `historical_evaluation_leader`, singular terminal outcome, disagreement/incumbent fields, mandatory exposed-provenance/fresh-confirmation fields, frozen statuses, OOS non-use flags;
- liquidation/wipeout, stable terminal reason/status/path.

Missing audit fields fail. Stable reason/status changes require intentional golden updates.

## 12. Architecture invariants to prove

1. Ancestry exactly `252910e`; shared-session/`.omc` absent from diff/history.
2. Factory/proxy/diagnostic evidence never selection-valid.
3. Outer portfolio handles each eligible one-second window; daily trend and atomic complete-cross-section near-high plus completed-4h carry match originals; reporting is not the signal clock.
4. Exact strict-injected adjacent bounded hashed sidecar composite identity, newest eligible source timestamp at bar-close and at each funding boundary, exact emitted-raw as-of close/resolver identities, no Backtest fallback or ambient path, no later/nonadjacent/future root; missing not zero.
5. Every finalizable boundary bucket is consumed once before indicator-only capsule into fresh flat scorer; no generic working state or warmup economic state.
6. Exact strict Backtest constructor kwargs activate immutable positive-execution pricing/application traces and alpha funding resolver. Each crossed UTC boundary settles independently from causal rate/close/pre-boundary quantity; empty kwargs/resolver `None` preserve OFF fill/funding/RNG/equity parity.
7. Every selectable matrix row/cost cell has complete matched raw-first evidence through the sole immutable real-consumer manifest materializer and exact constructor mapping; the three audit-frozen unavailable incumbents materialize no proxy.
8. Prelock cannot observe historical evaluation.
9. Exact ten candidate identities and the single sealed admitted execution subset propagate to manifest/Backtest/handler/Portfolio/adapters/funding/near-high, with rejected candidates inert; contract manifest/notional/type-7 admission/root ownership/dates/folds/regimes/allocator/rounding/caps/epsilon are frozen; train fits/admission, validation sole champion/gross, fixed refit bounded, historical evaluation exposed and report-only.
10. Same reporting domain, full-event MDD retained, no zero-fill/OOS membership change.
11. The closed trial ledger includes prior lineage/failures.
12. Soft-MDD comparator uses the complete matched normal universe separately in validation selection and historical-evaluation reporting; historical-evaluation values never select.
13. Scaled outputs actual replay with 1x prerequisites.
14. Research-only adapters; original strategies, legacy portfolio cadence 60, feature value-only API, default-empty non-manifest receipt tuples, generic legacy receipt IDs/counts and override-copy preservation, and shared sink/feature OFF paths unchanged.
15. The exhaustive no-`._rt` runtime contract, `LQ_` rejection, exact component/full/LOO/scaled manifest construction and child provenance, strict data-handler/lookup and sink kwargs/identities, effective constructed cadence 1, explicit window/parity/latency/order policy, seed schedule, 4h metric primitive, statistical inputs, and report-only RPT/capacity formulas are frozen; no profile/env/merge/default-runtime/alternate metric, manifest, or feature path exists.
16. The exact G004 blob/prior-set hash from actual LF `0x0a`, canonical current registry file/current-set hashes, and incumbent-resolution audit bind 1466+21 DSR trials to 1487; literal `\n`, proxy resolution, alternate node identity, ambient artifacts/status/correlation cannot change it.
17. Turnover/RPT/capacity cannot reject or rank a row and authorize no deployment claim.
18. No performance fabrication before data-PC output and no untouched-confirmation claim afterward without a genuinely fresh interval.

Final code-reviewer and architect must cite implementation/test evidence for all eighteen.

## 13. Verification commands and gates

```bash
uv run ruff check <changed-python-files>
uv run pytest -q \
  tests/research/test_alpha_max_evidence.py \
  tests/research/test_alpha_max_native_adapters.py \
  tests/research/test_alpha_max_engine_runner.py \
  tests/research/test_alpha_max_cost_attribution.py \
  tests/research/test_alpha_max_warmup_state.py \
  tests/research/test_alpha_max_cli_boundary.py \
  tests/test_strategy_tier_guard.py
uv run python -m compileall -q src scripts
```

Then run every applicable `.github/workflows/ci.yml` job: architecture/purity/hardcoded-parameter audits, full pytest, golden parity, coverage/dashboard, benchmark, native build, 8-GiB gates. Run `ai-slop-cleaner`, rerun affected/full gates, obtain independent code-reviewer `APPROVE` and architect `CLEAR`, then require hosted CI green.

## 14. Pass/fail boundary

**Local delivery passes** when semantics, isolation, tests, cleanup, reviews, invariant proof, Git history, and hosted CI are clean and data-PC commands executable. This is not an alpha pass.

**Historical robustness evidence is produced later** by the one-touch data-PC report under fixed gates. Because that interval is already exposed, even `prelock_champion_historical_robustness_passed` is diagnostic only. Genuine alpha confirmation requires a new future/withheld uninspected interval; a different historical-evaluation leader remains report-only and cannot be selected from this experiment.
