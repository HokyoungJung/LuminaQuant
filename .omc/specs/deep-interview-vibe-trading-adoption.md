# Deep Interview Spec: Vibe-Trading → LuminaQuant adoption (crypto-perp, framework-faithful)

## Metadata
- Interview ID: di-vibe-adoption-20260701
- Rounds: 2 (topology + 2 scoring rounds)
- Final Ambiguity Score: 11.5%
- Type: brownfield (`/home/hoky/Quants-agent`, package `lumina_quant`)
- Generated: 2026-07-01
- Threshold: 0.2
- Threshold Source: default
- Initial Context Summarized: no (deep brownfield context carried from prior adoption audit)
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.92 | 0.35 | 0.322 |
| Constraint Clarity | 0.88 | 0.25 | 0.220 |
| Success Criteria | 0.82 | 0.25 | 0.205 |
| Context Clarity | 0.92 | 0.15 | 0.138 |
| **Total Clarity** | | | **0.885** |
| **Ambiguity** | | | **0.115** |

## Source of truth
This spec operationalizes the prior **Vibe-Trading adoption audit** (23-agent workflow). Adopt the *ideas*, reimplement inside LuminaQuant's framework (`@register` plugins, formulaic-alpha IR, Polars-lazy, hotspot-only Rust `native/lumina_compute`, config-gated numerics, golden regression rtol=1e-8). **Direct code copy is forbidden.** Both repos are MIT — keep attribution in adapted files.

## Topology
| # | Component | Status | Description | Coverage / Note |
|---|-----------|--------|-------------|-----------------|
| 1 | Factor/alpha library expansion | active | Qlib158-style families reimplemented in formulaic IR, adapted to crypto-perp | AC-1x |
| 2 | Alpha discovery/search engine | active | generate → deterministic eval → survivorship gate → promotion (user's priority) | AC-2x |
| 3 | Lookahead & purity guards | active | shift-invariance factor test + AST blocklist (audit's only `adopt_now`) | AC-3x |
| 4 | Portfolio optimizer expansion | active | pure-numpy ERC / max-div / HRP via `@register("portfolio")` | AC-4x |
| 5 | Product surfaces | active | alpha-research CLI/report, execution-attribution, MCP bridge, dashboard views | AC-5x |
| 6 | Realism & correctness / delivery | active (cross-cutting) | golden byte-identical, gated-OFF numerics, crypto-perp IC re-validation, cost-realism, feature-branch+PR | AC-6x |

No deferred components. Multi-asset (options/forex/TradFi) items land **only** as gated seams/reference under component 5/6, never as live multi-asset engines.

## Goal
Extend LuminaQuant into a stronger **alpha research + production crypto-perp trading** engine by adopting the deterministic domain assets identified in the Vibe-Trading audit — reimplemented natively — with **the alpha find/create/improve loop as the centerpiece**, plus product surfaces, while guaranteeing that (a) the default backtest/live path stays byte-identical to today (golden rtol=1e-8), (b) every new numeric behavior is config-gated OFF by default, (c) everything that can be a hot numeric path is Rust-accelerated only where profiling justifies it (hotspot-only), and (d) backtest/live remain realistic and accurate (cost-realism + risk-gate parity, crypto-perp IC re-validation before any alpha is trusted).

## Constraints
- **Asset scope = Binance USDⓈ-M crypto-perp.** No live options/forex/equity engines. equity-derived factor formulas are adapted to per-symbol crypto OHLCV and IC-revalidated on crypto.
- **All 8 audit items land**, but asset-class-specific ones enter as config-gated-OFF **seams/reference/research-only**:
  - Options BS/greeks → pure `erf`-based math library (deterministic, for position/greeks *analytics* and validation), no options execution path.
  - Forex → parameter seam only.
  - TradFi fetchers (yahoo/stooq/sec_edgar) → research-only, versioned snapshot→replay, env `LUMINA_ENABLE_TRADFI_EXTERNAL_FETCH` default False, must pass `assert_source_registry_is_free_unauthenticated`.
  - Sharpe CI → advisory report sub-block, default emit OFF, never fed into hurdle/robust scores.
  - MCP → read-only, `[project.optional-dependencies] mcp` extra, lazy import, **no order tools exposed**.
- **Golden discipline (absolute):** `tests/test_golden_roundtrip_pnl.py` + `integration/test_walk_forward_golden.py` unchanged. New fields must NOT leak into existing metric dicts/JSON — additive sub-blocks with default-OFF emitters only.
- **Determinism:** all sampling/bootstrap uses `np.random.default_rng(fixed_seed)`. **No scipy, no sklearn** (repo non-deps; BLAS/version non-reproducibility). SLSQP → pure-numpy projected-gradient; norm CDF → in-repo `erf`/`_norm_ppf`; block-bootstrap → reuse `strategy_factory/research_metrics.spa_like_pvalue`. Never copy Vibe's i.i.d. `rng.choice`.
- **Rust policy (hotspot-only, `docs/RUST_NATIVE_ACCELERATION.md`):** default to Polars-vectorized formulaic IR; escalate an operation to a `native/lumina_compute` kernel only when a benchmark shows it's a hotspot (prime candidate: batch factor-IC over the alpha-search candidate set). New numeric behavior is added Rust-first *only if* it is a genuine hotspot AND stays byte-identical / gated; otherwise Python/Polars is acceptable.
- **Plugin pattern:** new strategies/indicators/portfolio optimizers via `@register(...)` + registry import + `tuning/param_registry.py` schema. Additive modules must not be imported on the backtesting/optimization hot path unless gated ON.
- **Delivery:** feature branch `feat/vibe-adoption` (or per-item sub-branches), atomic commits per item, golden + full suite green before push, `gh pr create`. **No merge to main, no live-real switch** — user decides those.
- **Do-not-port (conflict with determinism/realism):** `FALLBACK_CHAINS` auto-resolve, `_sanitize_data_map` silent row-drop, per-bar `on_bar` ABC dispatch, runtime LLM transpile to Pine/TDX/MQL, sklearn KMeans rule-mining. Reference logic only.

## Non-Goals
- No live options/forex/equity trading; no per-asset live-readiness beyond crypto-perp.
- No strategy export to Pine/TDX/MQL5 (audit: net-new but low-value + non-deterministic mechanism).
- No new OHLC-integrity layer — `compute/ohlcv_validation.py` already exceeds Vibe's coverage (skip).
- No new trade-analytics metrics beyond small additive helpers — `report_generator._analyze_trades` already covers profit_factor/payoff/loss-streak (skip; optional tiny by-exit-reason/spearman helper only if free).
- No LLM swarm / skills / channels / broker connectors from Vibe (out of scope; different product). LLM alpha proposer is **optional and gated**, deterministic-registration-only.
- No merge to main / no `LUMINA_ENABLE_LIVE_REAL` flip.

## Acceptance Criteria

### Component 1 — Factor library (Qlib158 → crypto-perp)
- [ ] AC-11 `ts_mean` (+ `ts_mean_series`) and any other missing operators added to `indicators/formulaic_operators.py`, unit-tested vs a hand-computed fixture.
- [ ] AC-12 Qlib158-family factor definitions expressed in the existing formulaic IR / `formulaic_definitions.py` (per-symbol OHLCV, no cross-sectional rank), duplicates already in `alpha_zoo/crypto_fx_factors.py` (kmid/klen/kup/klow ~4) excluded and cited.
- [ ] AC-13 Registered behind `strategies.qlib158_formula.enabled=false` (or candidate-library opt-in); with flag OFF the golden roundtrip + walk-forward golden are byte-identical.
- [ ] AC-14 Each new factor has a lookahead/purity test (see AC-3x) and a crypto-perp IC/IR readout artifact (advisory).

### Component 2 — Alpha discovery/search engine (centerpiece)
- [ ] AC-21 Extend `workflows/alpha_research_pipeline.py` + `strategy_factory/candidate_library.py` into a generate→evaluate→select loop: operator-tree / formula generation over the formulaic operator set (seeded, deterministic enumeration or seeded sampling).
- [ ] AC-22 Deterministic evaluation: rank-IC, IC-IR, turnover, decay per candidate over a fixed panel. **Batch factor-IC is the Rust-kernel candidate** (`native/lumina_compute`) — add only if benchmark shows hotspot; else Polars-vectorized. Byte-identical result vs a pure-Python reference within rtol=1e-8 golden fixture.
- [ ] AC-23 Survivorship gate reuses `research_metrics.deflated_sharpe_ratio` + `spa_like_pvalue` (block-bootstrap) + `approx_pbo`; only candidates passing the multiple-testing-aware gate are promoted.
- [ ] AC-24 Promotion writes to a candidate ledger (extend `research/candidate_outcome_ledger.py`) with full provenance (formula, seed, IC, DSR, SPA p, PBO, cost-realism A/B).
- [ ] AC-25 Optional LLM proposer behind a gate (default OFF): drafts candidate formulas, but only formulas that pass lookahead+purity+DSR/SPA gate are registered; registration path itself is deterministic.
- [ ] AC-26 Whole loop is config-gated; running it does not alter any default backtest/live numeric output.

### Component 3 — Lookahead & purity guards (adopt_now)
- [ ] AC-31 Property test: parametrize `indicators/alpha101/registry.list_alpha_ids` (101) + `ALPHA_FUNCTION_SPECS`; recompute each factor on a shifted synthetic panel via `alpha101/compiler.evaluate_compiled_formula` (full-series path, NOT the `_last_finite_value` scalar callable) and assert shift-invariance / no future leakage. Seed `default_rng(42)`; `>95% NaN` degenerate windows skipped; genuine leaks surfaced (xfail/allowlist to keep CI green initially).
- [ ] AC-32 Test-only, read-only, no numeric-path touch → golden unaffected, no config-gate needed, lands CI-green.
- [ ] AC-33 (optional, separate) `scripts/check_architecture.py` gains an os/sys/eval/subprocess/socket/urllib **blocklist** for factor modules (blocklist, not Vibe's whitelist — polars/numpy/lumina internals pass).

### Component 4 — Portfolio optimizers
- [ ] AC-41 Pure-numpy **ERC** (equal-risk-contribution, full covariance) registered `@register("portfolio","ERC")`; reuses existing `optimizer_core` LW-shrinkage cov / `project_simplex_with_upper_bounds` / `apply_caps`.
- [ ] AC-42 max-diversification + mean-variance as pure-numpy projected-gradient (no scipy SLSQP); HRP reuses the currently-unused `optimizer_core.cluster_by_correlation`.
- [ ] AC-43 `portfolio.allocation_method` defaults to `equal_weight`; with default, existing inverse-vol RP sleeve path (`cross_asset_trend_alpha_sleeves.py`) byte-identical (golden green).
- [ ] AC-44 Deterministic (seeded, fixed iteration count/tolerance) + unit tests vs known-answer fixtures (e.g. ERC of 2 uncorrelated equal-vol assets = 50/50).

### Component 5 — Product surfaces
- [ ] AC-51 `lq alpha` CLI (`cli/alpha.py` + register in `cli/main.py`): `rank` (IC/IR/turnover/decay), `card` (factor card md+json), `promote` (candidate ledger workflow).
- [ ] AC-52 Execution-attribution: additive `research/execution_attribution.py` — FIFO round-trip pairing, pure-math bias severity, delta-PnL attribution (noise/early/late/overtrading/missed=residual). `research.execution_attribution_enabled: false` default; never imported on hot path. Reconciled with LuminaQuant fee/funding model. **Reject** Vibe's reporter/codegen/j2/CJK parser/sklearn rule-mining.
- [ ] AC-53 Options BS/greeks pure-`erf` math library (analytics/validation only) + forex param seam; no execution path; own golden fixtures.
- [ ] AC-54 Sharpe-CI advisory sub-block (block-bootstrap via `spa_like_pvalue`, seed fixed), `emit_bootstrap_sharpe_ci=0.0` default; never in hurdle/robust score; golden unaffected.
- [ ] AC-55 TradFi research fetchers (yahoo/stooq/sec_edgar) research-only, gated OFF, snapshot→replay, free-unauthenticated assertion; source-pinned + fail-loud (no auto fallback chain).
- [ ] AC-56 MCP read-only stdio bridge (`lq mcp` or console script) as optional-dep, lazy import, only read-only tools (backtest/rank/report), order/place/cancel never exposed; read-only smoke test.
- [ ] AC-57 Dashboard: Next.js factor IC-heatmap + candidate-queue views wired to the existing `DashboardBridgeContractV2` read-only surface (additive routes/services).

### Component 6 — Realism, correctness, delivery (cross-cutting)
- [ ] AC-61 Full suite green (`uv run pytest`), golden byte-identical, `lq config validate` passes.
- [ ] AC-62 Every promoted alpha carries a cost-realism A/B (flat vs `sqrt_impact` slippage + funding coverage) and passes the DSR/SPA/PBO gate before being marked trusted — no unrealistic edge.
- [ ] AC-63 New live-eligible factors go through the same `RiskManager` order-risk gate + go-live stages as existing strategies (no divergence backtest↔live).
- [ ] AC-64 Native Rust builds (`scripts/build_native_backends.py`) green; any new kernel has a pure-Python parity oracle at rtol=1e-8.
- [ ] AC-65 Atomic commits per item on `feat/vibe-adoption`, golden+suite green, `gh pr create` (no main merge, no live-real).

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "All 8 audit items" ⇒ build options/forex/equity engines | But user also chose "stay crypto-perp, no options/fx/equity engines" | All 8 land, multi-asset ones as config-gated-OFF seams/reference/research-only; crypto-perp live path byte-identical |
| Adopt = copy Vibe code | Repo is deterministic/Rust/plugin-based; MIT both | Reimplement natively behind `@register`/formulaic-IR; attribution kept; do-not-port list enforced |
| "Rust more optimized" ⇒ rewrite everything in Rust | Repo is hotspot-only Rust | Rust only where benchmark shows hotspot (batch factor-IC prime candidate); Polars/Python elsewhere; always byte-identical/gated |
| New stats/reports can extend existing metric dicts | Would break every golden JSON | Additive sub-blocks, default-OFF emitters, advisory-only |
| Alpha "expansion" = bigger static library | User emphasized find/create/improve | Centerpiece = discovery/search engine (gen→eval→survivorship gate→promote), library is a supporting input |
| Commit ⇒ push to main | Memory: user owns merge/live decisions | feature branch + PR only |

## Technical Context (brownfield, verified)
- Operators: `indicators/formulaic_operators.py` has `ts_sum`/`ts_rank` (+`_series`); **`ts_mean` absent** (AC-11).
- Alpha101 IR: `indicators/alpha101/{registry,compiler,formula_sources,formula_ir}.py`; `compiler.evaluate_compiled_formula` returns full `pd.Series` (required for AC-31); `AlphaFunctionSpec.callable` returns only last scalar (unsuitable).
- Portfolio registry: only `@register("portfolio","EqualWeight")` today → ERC/max-div/HRP net-new (AC-4x).
- Research stats: `strategy_factory/research_metrics.py` already has `_norm_ppf` (Acklam), `deflated_sharpe_ratio`, `approx_pbo`, `spa_like_pvalue` (circular block-bootstrap, Politis–Romano) → reuse (AC-23, AC-54).
- Research scaffolds: `workflows/alpha_research_pipeline.py` (`DEFAULT_FAMILIES`), `strategy_factory/candidate_library.py` (`build_candidate_manifest`), `workflows/autonomous_portfolio_research_loop.py`, `research/candidate_outcome_ledger.py` → extend for AC-2x.
- CLI: per-command modules in `cli/*.py` mapped in `cli/main.py` (`autonomous_research.py` precedent) → add `cli/alpha.py` (AC-51).
- Rust: `native/lumina_compute/src/lib.rs` (7 kernels, maturin) → add batch factor-IC kernel iff hotspot (AC-22/64).
- Dashboard: `dashboard/bridge.py` `DashboardBridgeContractV2` read-only surface → additive routes (AC-57).
- Existing dedup to respect: `alpha_zoo/crypto_fx_factors.py` k-bar shapes; `compute/ohlcv_validation.py`; `report_generator._analyze_trades`; `data_sync._fetch_trades_with_retry` / `execution_live._call_with_retry`; `timeframe_aggregator.drop_incomplete_last`; `external_source_registry` fail-closed policy.

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Factor | core domain | id, formula/IR, family, operator-tree | evaluated over Panel; produces IC/IR |
| AlphaCandidate | core domain | formula, seed, IC, IR, turnover, DSR, SPA_p, PBO, cost_ab | promoted → Strategy sleeve; logged in ResearchLedger |
| Operator | supporting | name, window, series-variant | composes Factor |
| PortfolioOptimizer | core domain | method (ERC/maxdiv/HRP/MV), cov-estimator, caps | plugin via `@register("portfolio")` |
| ExecutionAttribution | supporting | fifo_pairs, bias_severity, delta_pnl_buckets | compares realized vs backtest |
| ResearchLedger | supporting | candidate provenance, gate results | records AlphaCandidate promotions |
| CostRealismProfile | supporting | slippage_model, funding_coverage | gates AlphaCandidate trust |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 0 (topology) | 6 | 6 | - | - | N/A |
| 1 | 6 | 1 (CostRealismProfile) | 1 (Alpha→AlphaCandidate) | 5 | 83% |
| 2 | 7 | 0 | 0 | 7 | 100% (converged) |

## Interview Transcript
<details>
<summary>Full Q&A (Round 0 + 2 rounds)</summary>

**Round 0 — Topology:** 6 components confirmed (library, discovery engine, guards, optimizers, products, realism/delivery). No deferrals.

**Round 1:**
- Scope: **all 8 audit items** (adopt_now + 7 consider)
- Asset class: **stay crypto-perp** (no options/fx/equity engines)
- Alpha depth: **search engine** (gen→eval→survivorship gate→promote; LLM optional/gated)
- Products: **all four** (alpha CLI/report, execution-attribution, MCP read-only, dashboard)
- Ambiguity: 23.2% (weakest: Constraints — Q1/Q2 tension)

**Round 2:**
- Multi-asset reconciliation: **gated seams/reference only** — all 8 land, options/fx/TradFi as config-gated-OFF seams/reference/research-only, crypto-perp live path byte-identical
- Git delivery: **feature branch + PR** (no main merge, user owns merge/live)
- Ambiguity: 11.5% ✅ (threshold 20%)
</details>
