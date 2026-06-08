# Alpha Zoo 69-asset Optuna hybrid refit

- generated_at: `2026-06-08T13:24:06.869913Z`
- universe: `10` symbols
- timeframes: `1h, 4h`
- execution: paper/testnet research only; real-money remains blocked.
- data source: direct 1m OHLCV resampled to >=30m bars.

## Split policy

Latest 8 complete weeks are validation. Locked OOS/test set is disabled for this live final-refit mode; post-freeze paper/testnet forward telemetry is required before any real review.

## Candidate summary

- candidate_count: `3840`
- decision_counts: `{'no_promotion_shadow_or_reject': 3803, 'sample_pass_shadow_until_execution_efficiency': 26, 'strict_paper_testnet_candidate_pending_forward_fill_telemetry': 9, 'relaxed_paper_testnet_candidate_pending_forward_fill_telemetry': 2}`

## Selected hybrid

- status: `optimized`
- backtest gate: `{'strict_backtest_gate_pass': True, 'relaxed_backtest_gate_pass': True, 'ready_for_paper': True, 'ready_for_real': False, 'real_money_execution': False, 'real_execution_allowed': False, 'rejection_reasons': []}`
- train return: `19.4962%`
- validation return: `18.5503%`
- validation MDD: `1.5113%`
- train RPT proxy: `30.47952168166796` bps
- validation RPT proxy: `237.95139403615903` bps
- top symbol share: `33.18%` (`TONUSDT`)
- top rule-side share: `100.00%` (`long_short`)
- validation long/short exposure share: `0.38775553391611917` / `0.612244466083881`
- effective symbol count: `4.99`
- concentration flags: `['top_asset_group_share_above_70pct']`

## Real-money status

`ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` remain hard-false. The path to real requires paper/testnet fill/BBO/slippage/protective-order/reconciliation telemetry, not only this backtest/refit artifact.
