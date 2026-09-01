# Independent Verifier Continuation Blueprint — Alpha Max Revision 5.14

## Scope and checkpoint

- Read-only independent verifier; no file edits were made by the verifier.
- Reviewed checkpoint: `a47489b97e51` plus the uncommitted G003-A/G003-B1 working tree.
- `alpha_max_engine_runner.py` intentionally stops before constructing or running `Backtest`.
- Both physical CLIs and the planned `tests/research/test_alpha_max_*` integration suites are absent.
- File line numbers below refer to the reviewed pre-checkpoint tree; prefer the named symbol if formatting shifts a line.

## Hard blockers

### B1 — Carry context mismatch

- The engine supplies `StrategyInputContext.feature_lookup`: `src/lumina_quant/core/engine.py:408-423`, schema `src/lumina_quant/core/strategy_input.py:9-19`.
- Carry currently reads a nonexistent `funding_lookup`: `src/lumina_quant/strategies/alpha_max_research_sleeves.py:822-827`.
- Repair carry to consume canonical `feature_lookup`; test using a real `StrategyInputContext`, not a dict fixture.

### B2 — Ordered lookup fails the engine capability gate

- Engine requires truthy `feature_lookup.db_path`: `src/lumina_quant/core/engine.py:135-147`.
- `AlphaMaxOrderedFundingLookup` has no `db_path`: `src/lumina_quant/research/alpha_max_evidence.py:367-417`.
- Add an immutable deterministic capability property representing the current exact root and test through `_assert_strategy_requirements`.

### B3 — No safe indicator-only warmup boundary

- Normal `Backtest` constructs Strategy, Portfolio, and Execution immediately: `src/lumina_quant/backtesting/backtest.py:133-239`.
- `_run_backtest()` processes all economic events: `src/lumina_quant/backtesting/backtest.py:460-480`.
- Frozen runner forces `warmup_bars=0`: `src/lumina_quant/research/alpha_max_engine_runner.py:990-1015`.
- Implement runner-owned indicator replay using the real windowed handler, real `ArtifactPortfolioModeStrategy`, `TimeframeAggregator.update_from_1s_batch()` (`src/lumina_quant/data/timeframe_aggregator.py:226-280`), and a discarded signal queue, without Portfolio/Execution economics.

### B4 — Full-event evidence conflicts with the 8-GiB bound

- Alpha samples at 1s: `src/lumina_quant/backtesting/backtest.py:310-315`.
- `all_holdings` is unbounded: `src/lumina_quant/backtesting/portfolio_backtest.py:277-281,933-989`.
- `_equity_points` retains only 20,000 points: `src/lumina_quant/backtesting/portfolio_backtest.py:362-364`.
- Neither source can currently prove complete multi-year full-event equity while respecting the memory gate. Approve and implement a bounded streaming/full-event evidence seam before satisfying E03 and the 8-GiB CI gate.

### B5 — Pure evidence layer is incomplete

Existing:

- manifest materializer: `alpha_max_evidence.py:1170-1568`
- metrics/statistics/trial ledger: `alpha_max_evidence.py:1620-2588`

Missing:

- raw/feature inventory and content hashing;
- actual train admission computation;
- row/cell evidence bundle;
- reconciliation serializer;
- validation selection and soft-MDD comparator;
- historical report ranking;
- terminal-state machine.

### B6 — Final-refit capsule cannot reuse a validation capsule

- Capsule includes exact `portfolio_mode`: `artifact_portfolio_mode.py:2540-2547`.
- Restore rejects a different manifest path: `artifact_portfolio_mode.py:2549-2563`.
- Every `prelock_final_refit` row must replay its complete permitted indicator prefix through embargo. A `validation_train_fit` capsule cannot be relabeled or transformed.

## Dependency-ordered implementation slices

### S0 — Shared invariant repairs

Files:

- `src/lumina_quant/strategies/alpha_max_research_sleeves.py`
- `src/lumina_quant/research/alpha_max_evidence.py`
- `src/lumina_quant/strategies/artifact_portfolio_mode.py`
- a narrowly approved full-event streaming seam if needed

Tasks:

- close B1/B2;
- expose exact child readiness, e.g. `validate_research_warmup_ready()`, without unchecked private access;
- require `finalize_completed_native_buckets()` result keys to equal all expected child ids; it currently silently skips incapable children at `artifact_portfolio_mode.py:2609-2615`;
- resolve full-event evidence storage;
- consolidate duplicated seed implementations in runner `:811-829` and evidence `:1593-1618`.

### S1 — Complete pure evidence contracts

Extend `alpha_max_evidence.py` after the current statistical primitives with:

- canonical raw/feature tree inventory and content seals;
- exact contract-manifest validation and 517-day admission computation;
- `AlphaMaxRowEvidence`, `AlphaMaxCostCellEvidence`, root/capsule/manifest receipts;
- pricing-trace to application bijection, no-fill exclusion, and fee/funding/liquidation reconciliation;
- canonical row/cell serialization;
- validation gate order, soft-MDD selection, historical report ranking, terminal outcome;
- immutable prelock inventory/seal payload.

No CLI or Backtest orchestration may invent these rules.

### S2 — Manifest-bound actual-engine activation

Extend `alpha_max_engine_runner.py` after its constructor-plan foundation.

Recommended immutable records:

- `AlphaMaxAncestorIdentity`
- `AlphaMaxArtifactSeal`
- `AlphaMaxExpectedDefinition`
- `AlphaMaxEngineActivation`
- `AlphaMaxReplayEvidence`

Before construction:

1. Validate a lexical absolute target under the owned phase directory.
2. `lstat` output root, manifests, phase, and target; reject symlink, escape, wrong type, or `nlink != 1`.
3. Seal manifest and config through `read_artifact_bytes()` only: `artifact_read_receipt.py:43-98`.
4. Parse only descriptor-returned bytes.
5. Validate the complete candidate/admission/child/cap/cash/gross/source/provenance contract.

Construct exact `Backtest` with:

- `HistoricParquetWindowedDataHandler`
- `ArtifactPortfolioModeStrategy`
- `Portfolio`
- `SimulatedExecutionHandler`
- `strategy_params={"portfolio_mode": f"manifest:{path}", "decision_cadence_seconds": 1}`
- exact constructor plan from `alpha_max_engine_runner.py:968-1015`.

Immediately after construction and again as the final operation before event 1, assert:

- consumer receipt ids exactly `("artifact_portfolio_manifest", "source:alpha_max_config")`;
- consumer receipts value-equal runner receipts;
- no `manifest_fail_closed_reason`;
- exact portfolio mode, manifest/config paths, components, classes, params, weights, cash, and native timeframes;
- candidate/caps/source metadata via receipt byte/SHA equality because `PortfolioModeDefinition` does not retain all of it (`artifact_portfolio_mode.py:194-214`);
- repeated ancestor/target/bytes/SHA seal.

Any mismatch must raise `portfolio_manifest_activation_mismatch` with zero market, funding, order, fill, or trade events.

### S3 — Identity, cadence, sink, and resolver activation

Before replay assert identity, not merely equality:

```text
backtest.symbol_list is admitted_symbols
data_handler.symbol_list is admitted_symbols
portfolio.symbol_list is data_handler.symbol_list
portfolio.bars is data_handler
data_handler._feature_lookup is ordered_lookup
resolver.ordered_lookup is ordered_lookup
resolver.admitted_symbols is admitted_symbols
strategy.decision_cadence_seconds == 1
config.DECISION_CADENCE_SECONDS == 1
```

Also prove ordered-value equality for definition components, child adapters/proxies, near-high barrier, funding domain, and every active symbol list.

For bound callbacks compare `.__self__` and `.__func__`, not a fresh bound method with `is`:

- portfolio application sink to collector;
- execution pricing sink to `SimulatedExecutionHandler._capture_pricing_trace`;
- resolver raw accessor to the exact handler `get_latest_raw_point`.

Relevant APIs:

- resolver identities: `alpha_max_evidence.py:685-938`
- strict handler injection: `data_windowed_parquet.py:78-110`
- execution activation: `execution_sim.py:193-259`
- portfolio seams: `portfolio_backtest.py:225-255,372-378`

### S4 — Indicator capsule and raw-first replay

Build one indicator capsule per row/manifest/prefix, then restore it into four independent fresh cost-cell engines.

Frozen phase root sequences in `alpha_max_evidence.py:279-313`:

```text
warmup     = [warmup]
train      = [warmup, train]
purge      = [train, purge]
validation = [purge, validation]
embargo    = [validation, embargo]
historical = [embargo, historical_exposed_evaluation]
```

Warmup flow:

1. Real handler groups rows at one timestamp: `data_windowed_parquet.py:196-236`.
2. Emit raw one-second window: `data_windowed_parquet.py:264-323`.
3. Update aggregator before strategy: `core/engine.py:375-438`.
4. Drain and discard every warmup signal.
5. At the boundary call `finalize_completed_native_buckets`.
6. Require exact child readiness and zero economic artifacts.
7. Extract the canonical capsule.
8. Restore only research indicator state into a fresh scorer with 10,000 cash, empty positions/orders/trades, and a new empty aggregator.

Raw-first economic chronology:

- strategy decisions queue first;
- Portfolio settles funding, liquidation, and equity in `update_timeindex`: `portfolio_backtest.py:901-912`;
- the enclosing event queue processes signals/orders/fills afterward;
- pre-existing quantity therefore pays boundary funding before a same-timestamp fill. E19 must prove this explicitly.

After final refit, replay the allowed full indicator prefix through embargo under each `prelock_final_refit` manifest before sealing historical capsules.

### S5 — Matrix, allocation, evidence, and selection

Frozen matrix:

- 3 components;
- 5 full policies;
- 9 LOO rows;
- 3 explicitly unavailable incumbents;
- 1 Track-B diagnostic;
- total 21 rows x 4 cells = 84 statuses;
- 17 resolvable rows x 4 = 68 actual-engine cells per scored domain.

Rules:

- incumbents and Track-B never construct an engine;
- every actual-engine cell is an independent raw replay;
- a row uses the same capsule hash across 10/15/20/30 bps;
- nominal 20 bps components fit ERC/HRP;
- nominal 30 bps is the eligibility/ruin/MDD reference;
- all-in cost may exceed nominal, so pricing, application, fees, funding, and liquidation reconcile separately.

Selection:

1. Apply DSR/SPA/PBO, positive metrics, coverage, hashes, funding, manifest, reconciliation, and ruin before MDD.
2. Gate MDD is `max(full_event_mdd, reporting_4h_mdd)`.
3. `<= 0.30` enters the normal set.
4. `> 0.35` rejects.
5. `(0.30, 0.35]` survives only if a normal row exists and both CAGR and Calmar strictly beat the deterministic best normal row.
6. Rank cumulative return, CAGR, Calmar, net Sharpe, lower MDD, lexicographic id.
7. Freeze one `prelock_champion`.
8. Refit only equal-risk/HRP weights from train+validation; never rescore validation or change the champion.

Terminal precedence:

1. no champion: `no_demonstrated_alpha`
2. missing/incomplete champion historical evidence: `historical_evaluation_incomplete`
3. complete but failed: `prelock_champion_historical_robustness_failed`
4. complete and passed: `prelock_champion_historical_robustness_passed`

Historical leader remains report-only.

### S6 — Physically separate CLIs

#### Prelock CLI

Create `scripts/research/run_alpha_max_prelock.py` with only:

```text
--config
--contract-manifest
--exchange
--output-root
--warmup-raw-root / --warmup-feature-root
--train-raw-root / --train-feature-root
--purge-raw-root / --purge-feature-root
--validation-raw-root / --validation-feature-root
--embargo-raw-root / --embargo-feature-root
```

It must expose no historical path/hash/inventory, profile, or runtime override. It seals config/runtime/contract/admission, roots, both manifest phases, all 84 statuses, champion/fixed gross/refit, historical-boundary capsules, and immutable `SEALED.json`.

#### Historical CLI

Create `scripts/research/run_alpha_max_historical_evaluation.py` with only:

```text
--sealed-prelock-directory
--embargo-feature-root
--historical-evaluation-raw-root
--historical-evaluation-feature-root
--exchange
--output-root
```

No config argument, embargo raw root, prelock raw roots, profile, runtime override, or alternate feature root.

It must:

- read config/contract/manifests only through sealed prelock receipts;
- rehash the embargo feature root;
- open only `[embargo, historical_exposed_evaluation]`;
- use only `prelock_final_refit` manifests/capsules;
- snapshot prelock before/after and never modify it;
- write a new append-only output root;
- reject duplicate completion ids.

Both CLIs expose `build_parser()` and `main(argv=None) -> int`, reject any `LQ_*` before all other I/O, create output roots exclusively/atomically, fsync files/directories, and keep duration/RSS observations outside stable hashed bytes.

### S7 — Final verification

- Add missing Revision 5.14 integration suites from `test-spec-alpha-max-independent-20260710.md`, especially hostile activation, warmup, real-engine, and subprocess filesystem tests.
- Run all applicable CI jobs and the 8-GiB gate.
- Run independent code-reviewer and architect reviews plus the architecture-invariant audit.
- Make a final Lore commit only after the gate is clean.

## Stop/no-claim rule

Local CI proves implementation and isolation only. It cannot claim superior alpha or historical robustness. The exposed historical interval is diagnostic and still requires a genuinely fresh, uninspected confirmation interval.
