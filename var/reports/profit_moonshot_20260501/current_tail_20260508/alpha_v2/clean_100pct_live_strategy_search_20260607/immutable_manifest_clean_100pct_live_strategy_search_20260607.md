# Immutable manifest — clean 100%+ live strategy search

- Generated UTC: `2026-06-07T07:11:42Z`
- Manifest id: `clean-100pct-live-strategy-search-20260607-v1`
- SHA256: `2ace861549f3c72182e5ac18ec87fd44f1d53b72f102eef1c42725f4bbece9ea`
- Ultragoal: `.omx/ultragoal G003-immutable-manifest`

## Scope

This manifest freezes the audit/search contract. No new candidate is promoted from historical locked-OOS artifacts. Any future fresh candidate must be run after this manifest or a successor manifest is emitted.

## Hard gates

- `no_nested_oos_mining`: existing OOS artifacts are contamination maps/report-only controls only.
- `execution_cost_gate`: 10/15/20bps, turnover/RPT, capacity/liquidity and paper fill telemetry required before live.
- `theory_plausibility_gate`: only lagged trend/momentum, lagged volatility/risk scaling, and cost-aware implementation are admissible.

## 100% threshold policy

The 100% annualized threshold is a post-evaluation report label only. It is forbidden as selector objective, Optuna objective, promotion gate, pruning input, or tie-break.

## Asset policy

Current data-bound universe is the existing Binance crypto universe in the clean artifacts. TradFi instruments (`SPY`, `QQQ`, `IWM`, `TLT`, `IEF`, `GLD`, `USO`, DXY proxy, `VIX`, `US10Y`) are monitoring-only until a separate data/cost/session manifest exists.

## Current result implication

Because this manifest was emitted for the fail-closed audit and not before a new heavy search, prior results remain `diagnostic_only`/`paper_control`/`shadow_freeze_only`/`rejected`, not real-money evidence.
