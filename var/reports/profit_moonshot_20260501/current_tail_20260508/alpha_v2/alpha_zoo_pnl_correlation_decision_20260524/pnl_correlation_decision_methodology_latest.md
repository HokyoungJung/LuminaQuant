# PnL Correlation Decision Methodology

This record defines how the paper/testnet Alpha Zoo portfolio is reduced after individual candidate gates pass.

## Inputs

- Candidate universe: `paper_testnet_monitor` rows from the multi-asset monitoring slate.
- PnL stream: per-bar fractional strategy return after the 10bps round-trip cost assumption and native notional fraction.
- Ranking: train+validation monitoring score only.
- locked-OOS: report-only after selection freeze; never used for discovery, fitting, pruning, objective, or selection.

## Correlation construction

- Method: `pearson_per_bar_pnl_returns_aligned_by_datetime_missing_bars_filled_zero`.
- Each strategy is replayed and indexed by bar datetime.
- Different timeframes are aligned on the union of timestamps.
- A missing timestamp for a strategy means the strategy has no bar/position update there, so PnL is filled with zero before correlation.
- Primary matrix: combined train+validation PnL streams.
- Validation-only matrix is a guardrail to avoid selecting candidates that diversify only because of train behavior.
- locked-OOS correlation matrix is saved for report-only monitoring diagnostics.

## Selection rule

- Sort by: monitoring_score_train_validation_only, train_return, validation_return, validation_mdd, train_return_per_turnover_proxy_bps, validation_return_per_turnover_proxy_bps.
- Accept greedily if max abs train+validation corr to selected <= 0.70.
- Also require max abs validation corr to selected <= 0.75.
- Reject otherwise as a high-PnL-correlation duplicate, not as a bad standalone alpha.
- Any accepted strategy remains paper/testnet-only and inherits original risk/efficiency gates.

## Interpretation

- Correlation is a de-duplication and diversification diagnostic; it is not a real-money approval.
- High-correlation clusters should be monitored as one alpha sleeve, not many independent strategies.
- Unscaled adoption of all candidates is rejected when gross notional and high-correlation clusters are large.

## Guardrails

- ready_for_real=false.
- real_money_execution=false.
- uses_locked_oos_for_selection=false.
