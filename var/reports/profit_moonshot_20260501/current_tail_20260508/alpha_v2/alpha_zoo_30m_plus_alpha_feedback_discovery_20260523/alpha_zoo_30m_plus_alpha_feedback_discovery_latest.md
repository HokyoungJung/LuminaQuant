# Alpha Zoo 30m+ feedback discovery

Generated: `2026-05-23T09:01:25.099166Z`

New >=30m Alpha Zoo discovery pass using native 30m bar construction and train+validation ranking.
Locked-OOS is gate/report-only after candidate freeze. Real-money remains blocked.

## Summary

- Candidates evaluated: `18450`
- Train-dominant sample gate pass: `23`
- Execution-efficiency proxy gate pass: `73`
- Full paper candidate gate pass: `4`
- Decision: `paper_testnet_candidate_after_fill_preflight`
- Runner peak RSS MiB: `1864.070`
- `ready_for_real=false`, `real_money_execution=false`

## Top train+validation-ranked rows

| Rank | Symbol | TF | Family | Train | Val | OOS | RPT train/val/OOS | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | SOLUSDT | 2h | donchian_atr_volatility_breakout | 37.1592% | 34.3856% | -5.2063% | 32.77/146.95/-28.92 | no_promotion_shadow_or_reject |
| 2 | SOLUSDT | 2h | donchian_atr_volatility_breakout | 30.6194% | 31.6860% | -4.8084% | 28.12/143.70/-29.68 | no_promotion_shadow_or_reject |
| 3 | ETHUSDT | 4h | volatility_adjusted_trend_persistence | 33.3487% | 28.5070% | -7.6467% | 31.54/105.58/-53.10 | no_promotion_shadow_or_reject |
| 4 | SOLUSDT | 2h | donchian_atr_volatility_breakout | 24.2761% | 21.9408% | -3.4728% | 32.11/140.65/-28.94 | no_promotion_shadow_or_reject |
| 5 | SOLUSDT | 2h | donchian_atr_volatility_breakout | 24.2761% | 21.9408% | -3.4728% | 32.11/140.65/-28.94 | no_promotion_shadow_or_reject |
| 6 | SOLUSDT | 1h | donchian_atr_volatility_breakout | 24.2576% | 22.8835% | -5.3534% | 17.97/77.05/-24.78 | no_promotion_shadow_or_reject |
| 7 | SOLUSDT | 1h | donchian_atr_volatility_breakout | 24.3883% | 29.7478% | -5.9835% | 18.07/103.29/-28.91 | no_promotion_shadow_or_reject |
| 8 | SOLUSDT | 2h | donchian_atr_volatility_breakout | 48.0327% | 21.9239% | -3.5875% | 47.23/99.43/-22.15 | no_promotion_shadow_or_reject |
| 9 | SOLUSDT | 2h | donchian_atr_volatility_breakout | 20.3581% | 20.3002% | -3.2014% | 28.04/138.10/-29.64 | no_promotion_shadow_or_reject |
| 10 | SOLUSDT | 2h | donchian_atr_volatility_breakout | 20.3581% | 20.3002% | -3.2014% | 28.04/138.10/-29.64 | no_promotion_shadow_or_reject |

## Paper/testnet-only candidates

| Rank | Symbol | TF | Family | Train | Val | OOS | Trades | RPT train/val/OOS |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 244 | SOLUSDT | 6h | volatility_adjusted_trend_persistence | 14.4540% | 5.5469% | 1.8230% | 164/42/23 | 29.38/44.02/26.42 |
| 245 | SOLUSDT | 6h | volatility_adjusted_trend_persistence | 14.4540% | 5.5469% | 1.8230% | 164/42/23 | 29.38/44.02/26.42 |
| 458 | TRXUSDT | 4h | volatility_adjusted_trend_persistence | 13.4914% | 2.9576% | 1.1825% | 199/52/25 | 15.07/12.64/10.51 |
| 459 | TRXUSDT | 4h | volatility_adjusted_trend_persistence | 13.2777% | 2.9576% | 1.1825% | 199/52/25 | 14.83/12.64/10.51 |

## Feedback loop and memory evidence

- Team feedback found an initial memory regression above the user cap; runner loading was repaired to per-file native 1s→30m aggregation before combining shards.
- Final leader real-data run: max RSS `1,908,808 KB` (`1864.07 MiB`), cap pass `true`, elapsed `3:11.64`, exit `0`.
- Strict locked-OOS policy stayed report/gate-only after train+validation ranking freeze; no OOS metrics were used for objective/pruning/parameter fitting.
