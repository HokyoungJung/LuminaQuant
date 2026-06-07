# Worker-4 Cost / Turnover / Capacity Verifier Plan

Generated: `2026-06-07T07:03:58.522720Z`
Team: `clean-100pct-live-tar-4a549153`
Task: `6` — Lane 4 execution cost turnover capacity verifier
Ultragoal: `G002-contamination-and-eligibility-audit` (leader-owned checkpoint; this lane only contributes evidence)

## Verdict

**No candidate is approved by this artifact.** This is a verifier contract and implementation plan. It keeps `real_money_execution_allowed=false`, `paper_execution_allowed_by_this_artifact=false`, and `locked_oos=diagnostic_report_only_after_freeze`.

Current repo evidence supports a conservative **10bps** primary cost gate and paper/testnet fail-closed telemetry behavior, but a clean 100pct-live eligibility claim remains blocked until the verifier emits explicit **10/15/20bps** rows, tail-loss fields, and capacity/liquidity outputs for the frozen candidate set.

## Required Cost Grid

| Scenario | Required status | Minimum verifier output | Eligibility consequence |
| --- | --- | --- | --- |
| `10bps` | Primary hard gate | split return, turnover, RPT, MDD/tail, trade count, cost breakdown | RPT/return failure rejects; missing row blocks |
| `15bps` | Robustness stress | same row fields plus stress deltas vs 10bps | failure downgrades to shadow/diagnostic until predeclared otherwise |
| `20bps` | Severe stress | same row fields plus tail/capacity deltas | missing/failing row blocks unqualified clean-live label |

The verifier must report rows for `train`, `validation`, and `locked_oos`, but **only `train` and `validation` may feed fitting/objective/pruning/selection**. Locked-OOS is report-only after freeze.

## Gate Matrix

1. **Fixed cost grid** — every candidate/model/split has `cost_bps in [10, 15, 20]`, `gross_return`, `net_return_after_cost`, `turnover`, `return_per_turnover_proxy_bps`, `trade_count`, `max_drawdown`, and `tail_loss_metrics`.
2. **RPT / turnover** — backtest proxy formula: `split_total_return * 10000 / max(split_turnover, epsilon)`; paper formula: `realized_pnl_quote * 10000 / sum(abs(notional_quote))`. RPT must exceed the scenario all-in cost for train/validation; a 10bps RPT failure is an immediate reject.
3. **Turnover/sample quality** — turnover denominator, trade count, active/inactive fold metrics, and low-sample warnings must be explicit. Missing or zero-denominator turnover cannot support an RPT claim.
4. **Capacity/liquidity proxy** — report ADV/ADTV/sigma, order notional, fill rate, unfilled notional, participation p50/p90/p99, and AUM-scale decay at `0.5x/1.0x/1.5x/2.0x`.
5. **Slippage assumptions** — separate modeled `spread_bps`, `impact_bps`, `fees_bps`, `funding_bps`, `total_bps`, and `realized_slippage_bps` from actual paper/testnet measurements. Missing BBO or guard breach must fail closed with no market fallback.
6. **MDD/tail** — report max drawdown, worst-period return, p05/p01 return, CVaR 5%, liquidation count, account-wipeout count, and minimum margin buffer. Missing tail quantiles/CVaR block clean-live eligibility.
7. **Paper fill telemetry** — require fill count, notional, realized PnL, BBO spread, mean/p95 all-in cost, timeout/cancel/partial-fill rates, submit-to-fill latency, and reconciliation drift before any real-money review.

## Current Repo Evidence

- `scripts/research/run_alpha_zoo_10bps_full_retune.py:2-7,50,180-195,198-215` — conservative 10bps artifact, locked-OOS report/gate-only, train+validation score path.
- `scripts/research/assert_alpha_zoo_10bps_full_retune_artifact.py:14-29,50-66,128-132,194-208,229-232` — exact 10bps and locked-OOS non-use assertions.
- `scripts/research/run_alpha_zoo_top_seed_hybrid_v35_v36_cost_validation.py:45-48,840-875,899-974` — current cost validation reports 5/10bps and stream costs 0/5/10bps; no explicit 15/20bps grid yet.
- `scripts/run_cost_aware_framework.py:286-378,432-463` — records fill costs, realized slippage, turnover, participation quantiles, capacity decay, and post-cost MDD metrics.
- `src/lumina_quant/backtesting/cost_models.py:8-23,56-92,100-170` — spread/impact/fees, participation, max-participation fills, unfilled carry/drop, no-close-fill execution.
- `src/lumina_quant/backtesting/liquidity_metrics.py:8-43` — rolling ADV, ADTV, and sigma liquidity proxies.
- `src/lumina_quant/eval/cost_aware_reports.py:14-56` — current perf metrics include MDD but not tail quantile/CVaR fields.
- `scripts/research/run_alpha_zoo_paper_fill_efficiency_gate.py:42-49,253-369,619-648` — paper gate computes realized RPT, BBO spread, mean/p95 all-in cost, timeout/cancel/partial, liquidation/wipeout, and stays paper-only/fail-closed.
- `scripts/ops/write_alpha_zoo_optuna_hybrid_live_decision.py:224-267` — microstructure telemetry contract and minimum two-week paper/testnet observation requirement.
- `tests/test_alpha_zoo_69_asset_efficiency_live_adapter.py:146-181` — no market fallback, BBO requirement, paper-only flags, and blocked clean-OOS RPT failure handling.
- `tests/test_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna.py:20-66` — train/validation RPT and 15/20bps stress proxy fields; 10bps RPT gate is not relaxed.

## Gaps to Close

- Generalize primary-cost constants into explicit `10/15/20bps` verifier rows and assertions.
- Add/surface tail-loss quantiles and CVaR; MDD alone is insufficient for Task 6.
- Wire capacity/liquidity proxy outputs into the Alpha Zoo verifier artifact, not only the generic cost-aware framework.
- Summarize paper/testnet all-in cost and slippage telemetry by symbol/timeframe with missing-field counts.
- Preserve locked-OOS report-only status; no objective, pruning, fitting, parameter grid, selector, or promotion claim may consume it.

## Minimal Implementation Plan

1. Add `COST_VERIFIER_GRID_BPS = (10.0, 15.0, 20.0)` to the relevant verifier/cost-validation runner.
2. Emit required metrics rows for every candidate/model/split/cost scenario.
3. Extend assertion coverage to fail on missing cost rows, missing tail/capacity fields, or locked-OOS leakage.
4. Normalize paper fill telemetry fields and keep paper/real execution fail-closed until a separate review.

## Validation Commands

- `python3 -m json.tool team_lanes/worker4_cost_verifier.json >/tmp/worker4_cost_verifier.json.pretty`
- `uv run python -m compileall -q src scripts tests`
- `uv run pytest -q tests/test_cost_aware_framework_e2e.py tests/test_cost_calibration.py tests/test_alpha_zoo_69_asset_efficiency_live_adapter.py tests/test_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna.py`
- `uv run ruff check scripts/research/run_alpha_zoo_paper_fill_efficiency_gate.py scripts/run_cost_aware_framework.py src/lumina_quant/backtesting/cost_models.py src/lumina_quant/backtesting/liquidity_metrics.py tests/test_alpha_zoo_69_asset_efficiency_live_adapter.py tests/test_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna.py`

## Subagent Evidence Integrated

Subagents spawned: `2` (`Copernicus` `019ea0de-19e4-7053-89ae-2c949cb884ea`; `Newton` `019ea0de-2b39-7480-b329-9844654ed540`). Requested model per task: `gpt-5.4-mini`; the exposed spawn tool did not provide model/agent_type fields, so executor/model intent was included in prompts. Serial repo searches before spawn: `0`.

Integrated findings:

- 10bps primary coverage is present; explicit 15/20bps verifier coverage is incomplete.
- Capacity/liquidity proxy exists through ADTV/ADV/sigma/participation but needs direct Alpha Zoo verifier linkage.
- Paper telemetry exists and remains fail-closed; all-in cost/slippage summaries and missing telemetry counts must be present before real-money review.
- MDD/liquidation controls exist, but CVaR/tail-quantile reporting was not obvious and must be added or surfaced.
