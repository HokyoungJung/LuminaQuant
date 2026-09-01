# Profit moonshot liquidation-aware 5x result — 2026-05-10

## Decision

Forced integer `5x` on the current-base sleeve tuple is **not deployable** under the liquidation-aware gate.

Blocking evidence:
- Forced `5x` has `validation` liquidation count `1`.
- Liquidation event: `2026-02-05T20:00:00Z` `BTC/USDT` `LONG` in `fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all`; entry `76515.97509`, liquidation threshold `62674.23520`, intrabar adverse trigger `62233.30000`.
- OOS return improves versus the liquidation-aware current-base replay by `+0.073696%`, but OOS return/MDD declines by `-0.015007`.
- OOS Sharpe/Sortino/smart-Sortino fail the promotion quality floors in the liquidation-aware replay.

Retain the current-base reference; do not promote `5x`.

## Baseline preservation

- Pushed green handoff head preserved: `77f10d54174628c24f1a6bbba34a74505a2a40b5`.
- Performance baseline preserved: `02f4520cf906f48089b8852c2651a0f1e4bd0c1c`.
- Previous integer audit remains evidence only; this run adds liquidation/margin replay instead of overwriting the baseline.

## Selection boundary

- Integer grid evaluated: `1x..6x`, plus current-base leverage `2.342733429770x`.
- Selection policy: train/validation only.
- Locked-OOS: report-only/gate-only; `uses_locked_oos_for_selection=false`.
- Train/validation-safe selected integer: `3x`, but deployable success `false` because OOS return/MDD and OOS risk metrics do not beat the current-base replay quality gates.
- Highest zero-liquidation integer across train/validation/OOS: `4x`, also not deployable for the same performance-quality reasons.

## Forced 5x split metrics

| Split | Return | MDD | Liquidations | Min margin buffer | Min margin ratio | Sharpe | Sortino | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | -3.176187% | 3.377761% | 0 | 9668.563283 | 822.786932 | -2.253981 | -1.923932 | -0.940429 |
| validation | +0.587702% | 0.314859% | 1 | 9967.299062 | 832.058470 | 2.526233 | 3.368284 | 11.733209 |
| oos | +0.143155% | 0.511099% | 0 | 9974.584244 | 846.858584 | 0.585941 | 0.544070 | 1.546184 |

## Current-base replay split metrics

| Split | Return | MDD | Liquidations | Min margin buffer | Min margin ratio | Sharpe | Sortino | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | -1.996967% | 2.091894% | 0 | 9793.740396 | 1770.807439 | -2.966225 | -2.539262 | -0.954729 |
| validation | +0.252155% | 0.146477% | 0 | 9985.045717 | 1783.441482 | 2.500234 | 3.346716 | 10.727296 |
| oos | +0.069459% | 0.235375% | 0 | 9988.083889 | 1805.019112 | 0.596448 | 0.556492 | 1.626327 |

## Margin model

Conservative Binance USDⓈ-M perpetual-style scalar model:
- margin mode: `cross`
- maintenance margin rate: `1.0000%`
- taker fee: `0.1000%`
- slippage: `0.0500%`
- funding reserve per 8h: `0.0100%`
- stress buffer: `0.2500%`
- liquidation fee reserve: `0.5000%`
- total liquidation reserve: `1.9100%`
- Official references recorded in the JSON artifact under `source_references`.

## Artifacts

- Latest JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_5x_20260510/liquidation_aware_current_base_latest.json`
- Latest Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_5x_20260510/liquidation_aware_current_base_latest.md`
- Timestamped JSON/Markdown: see `liquidation_aware_current_base_20260510T035651Z.*`

## Verification evidence so far

- Targeted tests: `uv run --extra dev pytest -q tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_profit_moonshot_pass_under_8gb_validator.py` → `32 passed in 0.09s`.
- Liquidation-aware replay: `/usr/bin/time -v uv run --extra dev python scripts/research/run_profit_moonshot_liquidation_aware_validation.py --output-dir .../liquidation_aware_5x_20260510` → exit `0`, max RSS `257224 KiB`; artifact memory peak `251.094 MiB`, under 8 GiB `true`.
- Full pytest: `/usr/bin/time -v uv run --extra dev pytest -q` → `1226 passed in 228.81s (0:03:48)`, max RSS `2784220 KiB`.
- Additional local gates: `uv run --extra dev ruff check .`, `python3 -m compileall -q src scripts tests`, and `git diff --check` all passed. Remaining after this handoff: Lore commit/push and GitHub Actions `ci`/`private-ci` confirmation.
