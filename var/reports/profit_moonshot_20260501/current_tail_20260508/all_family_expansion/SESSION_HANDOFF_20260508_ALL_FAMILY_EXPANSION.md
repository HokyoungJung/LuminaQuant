# Profit moonshot all-family expansion handoff — 2026-05-08

## Status
- Latest-tail replay/tuning/Optuna completed with locked-OOS report-only policy preserved.
- New strategy families were included in the replay grid: `residual_momentum, cross_sectional_sharpe_reversal, funding_carry_momentum, adaptive_trend_fade, compression_breakout_fade`.
- None of the new non-calendar families passed train/val/OOS gates; they are **not promoted**.
- Promoted pass remains `calendar_rotation`, but return improved through expanded TRX take-profit/calendar grid and family-balanced portfolio tuning.

## Best passing portfolio by OOS return
- name: `fresh_portfolio_equal_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0`
- mode: `equal_weight`; sleeves: 4
- train return: 3.5993%
- validation return: 2.6755%
- locked-OOS return: 1.2181%
- locked-OOS max drawdown: 0.1662%
- locked-OOS Sharpe: 6.7264
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600,fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600,fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0,fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0`

## Comparison to prior continuation pass
- prior promoted portfolio OOS return: 0.8789%
- current best passing portfolio OOS return: 1.2181%
- improvement: 0.3392%

## Search scale / pass counts
- replay specs: 6805; replay successes: 300; survivors: 300
- portfolio specs: 58224; portfolio successes: 6129; candidate sleeves: 72
- Optuna calendar trials: 128; Optuna successes: 8
- success families: `{'calendar_rotation': 300}`

## New-family result
All new families were evaluated but failed promotion. Best OOS/validation rows are in `var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion/passing_candidate_latest.json` under `new_family_assessment`.

## Risk policy
- Locked OOS remains report-only/gated; no OOS-selected diagnostic is promoted.
- Diagnostic selected-by-validation portfolio failed `oos_mdd_beats_shadow` and is recorded as not promoted.
- Memory guard stayed far below 8 GiB: replay 289.46 MiB, portfolio 652.68 MiB, Optuna 242.76 MiB.

## Verification
- 1193 passed in 269.27s (0:04:29) via uv run --extra dev pytest -q
- All checks passed via uv run --extra dev ruff check .
- python3 -m compileall -q src scripts tests passed
- git diff --check passed
- mission validator status: passed

## Next handoff
1. Do not promote non-calendar families from this run; use them as negative evidence unless their train/val signs improve in a future hypothesis.
2. If chasing higher return, explore calendar/TRX parameter robustness and MDD-controlled sleeve construction before accepting high-return additive diagnostics.
3. Keep `--family-quota` available for exploration, but final promotion must stay gate-based, not family-presence-based.
