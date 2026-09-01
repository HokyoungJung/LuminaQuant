# Pair-spread robustness continuation summary — 2026-04-12

## What was run

- All runs respected the low-memory sequential envelope: `POLARS_MAX_THREADS=1`, `LQ_BACKTEST_LOW_MEMORY=1`, `LQ_AUTO_COLLECT_DB=0`, no concurrent heavy jobs.
- No article batch (`batch_01~44`) was rerun.

### 1) Bridge follow-up (6 candidates)
- Path: `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/pair_spread_robustness_bridge_followup_current/research_run`
- Envelope: 38.64s wall, max RSS 1,187,684 KiB.
- Design: move only moderately lower on threshold (2.45 / 2.35), but shorten windows/cooldown/re-entry/hold to reduce zero-trade OOS folds.
- Result: the 2.35 variants did increase OOS trade_count to 10 and kept `oos_pbo=0.5`, but train collapsed hard (`train return ≈ -14%`, `train sharpe ≈ -4.2`). The 2.45 variants went negative across splits.

### 2) Untested slice from the original robustness manifest (5 candidates)
- Path: `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/pair_spread_robustness_followup_untested_slice_current/research_run`
- Envelope: 34.41s wall, max RSS 1,164,580 KiB.
- Purpose: the original 8-candidate robustness follow-up had been run through the batch wrapper with `stage1_keep_ratio=0.35`, so 5 candidates never actually reached stage2. This run evaluated them directly with `--stage1-keep-ratio 1.0`.
- Result: no hidden winner. Best was `pair_spread_1h_robust_hybrid_bnbusdt_trxusdt_2.4_0.60` with train `+3.05%`, val `+0.36%`, OOS `+0.63%`, but `oos_sharpe=0.211` and `oos_pbo=0.75` still hard-rejected. The strict 2.6 variants became flat/negative and even sparser.

### 3) Mid-bridge 2.5 follow-up (2 candidates)
- Path: `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/pair_spread_robustness_midbridge25_followup_current/research_run`
- Envelope: 25.57s wall, max RSS 1,105,984 KiB.
- Purpose: test the most plausible remaining narrow lane — 2.5/0.65 for the two 2.6 families that already had positive train Sharpe (`exec_tightstop_tp`, `state_atr`).
- Result: both candidates preserved all-splits-positive performance and positive train Sharpe, but `oos_trade_count` stayed at `6` and `oos_pbo` stayed at `0.625`. This confirmed that even the best mid-bridge tweak does not break the sparsity/PBO wall.

## Best new results

- `pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.5_0.65`
  - train `+3.2205%`, sharpe `0.2629`, trades `52`, pbo `0.500`
  - val `+7.4445%`, sharpe `2.5549`, trades `18`, pbo `0.625`
  - oos `+3.3809%`, sharpe `4.4219`, trades `6`, pbo `0.625`
  - verdict: quality preserved, but still too sparse / high-PBO.
- `pair_spread_1h_state_atr_bnbusdt_trxusdt_2.5_0.65`
  - train `+2.7564%`, sharpe `0.1712`, trades `52`, pbo `0.500`
  - val `+5.5401%`, sharpe `1.7986`, trades `16`, pbo `0.625`
  - oos `+3.0036%`, sharpe `3.9495`, trades `6`, pbo `0.625`
  - verdict: same structural problem as above.

## Conclusion

- The BNB/TRX 1h pair-spread lane still has a frontier, but it is structurally sparse:
  - profitable settings around `2.5~2.6` keep OOS positive, but stay at only `6` OOS trades and `oos_pbo=0.5~0.625`
  - attempts to force more fold coverage raise OOS trade_count to `10~14`, but destroy train quality or OOS Sharpe
- This now looks like a **structural robustness limit**, not an undiscovered local knob.
- Recommended next move: **broader redesign** (new family / new pair / different timeframe / stronger pre-search robustness prior), not more threshold micro-sweeps on this exact BNB/TRX 1h lane.
