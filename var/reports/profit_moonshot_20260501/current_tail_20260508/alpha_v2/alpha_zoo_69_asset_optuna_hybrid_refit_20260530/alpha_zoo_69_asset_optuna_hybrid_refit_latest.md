# Alpha Zoo 69-asset Optuna hybrid refit

- generated_at: `2026-05-30T07:36:59.155358Z`
- universe: `69` symbols
- timeframes: `30m, 1h, 2h, 4h`
- execution: paper/testnet research only; real-money remains blocked.
- data source: direct 1m OHLCV resampled to >=30m bars.

## Split policy

Latest 8 complete weeks are validation. Locked OOS/test set is disabled for this live final-refit mode; post-freeze paper/testnet forward telemetry is required before any real review.

## Candidate summary

- candidate_count: `45120`
- decision_counts: `{'no_promotion_shadow_or_reject': 45103, 'sample_pass_shadow_until_execution_efficiency': 17}`

## Selected hybrid

- status: `optimized`
- backtest gate: `{'strict_backtest_gate_pass': True, 'relaxed_backtest_gate_pass': True, 'ready_for_paper': True, 'ready_for_real': False, 'real_money_execution': False, 'real_execution_allowed': False, 'rejection_reasons': []}`
- train return: `9.0960%`
- validation return: `8.5640%`
- validation MDD: `0.6934%`
- train RPT proxy: `10.960323416727334` bps
- validation RPT proxy: `31.837053294540155` bps
- top symbol share: `16.27%` (`TONUSDT`)
- top rule-side share: `100.00%` (`long_short`)
- validation long/short exposure share: `0.5206173221349235` / `0.4793826778650765`
- effective symbol count: `11.59`
- concentration flags: `[]`

## Real-money status

`ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` remain hard-false. The path to real requires paper/testnet fill/BBO/slippage/protective-order/reconciliation telemetry, not only this backtest/refit artifact.
