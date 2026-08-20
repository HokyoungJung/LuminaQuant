# Alpha-Sleeve Batch + Engine Levers — Data-PC Handoff (2026-08-20)

Build-not-measure batch authored on the code PC. Everything is `research_only`
and/or config-gated OFF (default paths byte-identical; golden pins green; full
suite 4914 passed). **No performance claims.** Standing verdict unchanged:
do_not_promote / research_only_no_execution / 0% real allocation.

Design provenance: 8-lens repo mapping -> 4-lens proposal panel -> 3
adversarial judges (graveyard-keeper / cost-and-data realist / novelty
checker). Every shipped lane names its nearest graveyard relative, its
EXPECTED NULL, and one falsifying measurement in the module docstring
(pre-registration). Killed/deferred by the judges: same-bar MKT netting
(measure incidence first, no engine change), liquidation-imbalance XS sleeve
(coverage audit first), leader-RV sizing overlay (blocked behind V-DIAG
admission; frozen design block lives in `research/vol_spillover_diagnostic.py`).

## 1. What shipped

### New research_only strategy classes (6 modules, 7 classes, 21 default-manifest rows)

| Class | Module | Mechanism (one line) | Cells |
|---|---|---|---|
| CrossSectionalResidualTakerFlowStrategy | strategies/xs_residual_taker_flow_alpha_sleeves.py | ~5d signed taker imbalance / turnover, residualized on return z; weekly L/S quintiles, 1-week min-hold | 1h+4h x {baseline_ls, guarded_ls} |
| BasisFundingGapConvergenceStrategy | strategies/basis_funding_gap_alpha_sleeves.py | XS z of basis_bps(mark,index) - funding-implied basis at 8h boundaries; \|z\|>2 fade, 2-interval min-hold, 72h max | 4h x {boundary_fade, patient_fade} + 1h boundary_fade |
| OffSessionBasisDislocationStrategy | strategies/offsession_basis_dislocation_alpha_sleeves.py | TradFi-perp off-session mark-index dislocation, faded at cash reopen; \|z\|>2 AND >=10bps floor, 36h/72h holds | 1h x {q20, q25} (TradFi universe) |
| SalienceTheoryValueStrategy | strategies/behavioral_value_xs_alpha_sleeves.py | BGS salience-distortion fade (theta=0.1, delta=0.7, 21d), weekly ISO, q20/q35 hysteresis | 1h/4h/1d x {minhold5, minhold10} |
| ProspectTheoryValueStrategy | strategies/behavioral_value_xs_alpha_sleeves.py | TK-1992 value of trailing 90d histogram, fade high-TK; tournament-paired with ST (pre-registered) | 1h/4h/1d x {minhold5, minhold10} |
| OpenInterestGrowthPressureStrategy | strategies/oi_growth_pressure_alpha_sleeves.py | Hong-Yogo 7d OI-notional growth / dollar volume, momentum-residualized continuation, weekly | 1h x {7d, 14d_fallback} |

All are cross-sectional baskets: admission ONLY via
`select_diversified_shortlist(..., allow_multi_asset=True)`. Entry signals
stamp `intended_hold_seconds` so the L-D guard composes. Redundancy
tournament ST-vs-PT: if |XS rank corr| > 0.6, only the higher orthogonal-IC
lane admits (both docstrings).

### New indicators (re-exported from `lumina_quant.indicators`)

- `har_rv.py`: daily_realized_variance / log_rv_transform / har_design /
  har_fit / har_rv_forecast / har_annualized_vol_forecast (Corsi HAR-RV).
- `variance_ratio.py`: Lo-MacKinlay VR extracted from cusum_varratio sleeve
  (BIASED overlapping estimator stays the parity-locked default; unbiased +
  robust z-stat opt-in).
- `funding_structure.py`: funding_momentum (extracted, parity-locked),
  funding_term_structure_spread(3, 21 prints), funding_implied_basis_bps,
  basis_funding_gap_bps. ONE funding module (judge-merged).
- `behavioral_value.py`: salience_theory_value, prospect_theory_value
  (canonical constants frozen).
- `rolling_stats.py` additions: rolling_skewness, rolling_excess_kurtosis
  (dedups two strategy-private copies; fsum recipe, parity goldens pinned).
- `CrossSectionalFundingMomentumCarryStrategy` gains config-gated
  `require_term_structure_agreement` (default False, byte-identical; entry-
  SKIP-only).

### V-DIAG (volatility-program admission diagnostic; non-trading)

`research/vol_spillover_diagnostic.py` + `scripts/research/run_vol_spillover_diagnostic.py`.
HAR-RV baseline vs +leader-lagged-RV-block candidate; QLIKE (Patton) +
log-RV MSE on 8 anchored expanding folds; paired circular block bootstrap
(seed 20260820, sharpe_ci.py conventions); BH-FDR across pairs. Admission per
pair: median QLIKE improvement >=5% AND >=60% folds AND BH p<=0.05. Pairs:
BTCUSDT->9 alts + XAUUSDT->XAG/XPT/XPD. Failure KILLS all direct
vol-spillover strategies and new multivariate-vol code (master plan §6.3).
The deferred LeaderRvConditionedSizingOverlay design is frozen verbatim in
the module docstring; its build gate is `sizing_overlay_build_gate_open`
(>=1 BTC->alt admission).

### Engine levers (commit 470b9be; all default OFF, byte-identical)

- **L-C**: `strategy_quality.min_hold_bars` + `strategy_quality.no_trade_band_bps`
  (real-engine seam: StrategyQualityOverlay + Portfolio.generate_order_from_signal).
  Min-hold blocks only BARE strategy EXITs and reversals inside the hold
  window; `risk_exit`/`exit_reason`/`overlay_reason` metadata and engine-level
  protective stop/TP/liquidation fills always pass.
- **L-D**: `execution.funding_entry_guard` — declared sub-8h holds that would
  straddle a 00/08/16 UTC settlement are never opened; undeclared signals
  never blocked; sign-blind, no tunables.
- **Warmup-end hook**: engine fires optional `strategy.on_warmup_end()` once
  at the warmup->live transition (ghost-position reset), persisted across
  chunk boundaries.
- **H1**: unknown-strategy tier fallback now fails safe to research_only
  (68-name legacy snapshot mirrored into the registry; zero behavior change
  for every currently-registered class).

### Coverage audit tool

`scripts/research/audit_liquidation_feature_coverage.py`: per-group share of
symbol-days with non-None open_interest AND liquidation notionals; gates
open_interest>=0.90, liquidation>=0.80 (CLI-overridable); zero-data run
closes as insufficient_data.

## 2. Exact data-PC order (after absorbing the v3/v5 protocol)

1. **Prerequisite backfills/audits (before any sleeve measurement):**
   - Taker flow: run `scripts/research/backfill_raw_taker_flow_feature_points.py`
     over the archive so taker columns are dense; the taker sleeve treats a
     None taker bar as a dropped bar (all-or-nothing capture).
   - `uv run python scripts/research/audit_liquidation_feature_coverage.py
     --data-root <parquet> --symbols <universe> --start 2025-01-01 --end <now>`
     -> OI group coverage >=90% unlocks OpenInterestGrowthPressure cells;
     liquidation coverage >=80% unlocks BUILDING the deferred liquidation XS
     sleeve (do not build it before this gate passes).
2. **Standard full-universe walkforward** (same invocation as
   `alpha_pool_evaluability_v3_handoff.md`, research.yaml profile): the 21 new
   rows ride the default manifest. Audit `evaluation_mode` — new lanes must be
   `registry_simulator`, never `generic_fallback_proxy`. Two-tier gate as
   always (net-20bps edge>0, DSR>0 after N_eff, incremental orthogonal
   factor_ic>0 vs the named incumbents in each docstring, RPT>=10bps/split).
3. **V-DIAG**: `uv run python scripts/research/run_vol_spillover_diagnostic.py
   --series-path <parquet-root>` (or --rv-csv). Persist the JSON verdict
   artifact next to the run. No admission => the vol program and the deferred
   overlay stay dead; do not salvage.
4. **L-C/L-D A/B (pre-registered, no grids):** identical sleeve set, flags OFF
   vs ON, cost grid 10/15/20/30bps. ON arm (fixed ex-ante):
   `strategy_quality.min_hold_bars` = 24 (1h) / 6 (4h) / 2 (1d) [= ~1 day],
   `strategy_quality.no_trade_band_bps` = 8.0 (the dead constructor's
   original default), `execution.funding_entry_guard` = true. Lever dies if
   median per-sleeve RPT does not improve at 20bps or median net-Sharpe delta
   <= 0 (L-C), or net PnL delta <= 0 / blocked-entry count ~0 (L-D). Report
   per-sleeve, no cherry-picking.
5. **Same-bar netting incidence (no engine change):** from any multi-sleeve
   portfolio run's order/fill logs, measure same-bar opposite-direction MKT
   notional per symbol as a share of gross turnover. Building the crossing
   engine requires >=1% share AND a live-executor parity plan (judge
   condition).

## 3. Deferred (documented, do not build without the stated gate)

| Item | Gate |
|---|---|
| LeaderRvConditionedSizingOverlay | V-DIAG admits >=1 BTC->alt pair |
| LiquidationImbalanceFireSaleXS sleeve | liquidation coverage audit >=80% |
| Same-bar MKT order netting | measured crossed share >=1% of gross turnover + live parity plan |
