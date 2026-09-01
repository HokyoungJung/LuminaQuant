# Profit Moonshot continuation — precious metals aggressive alpha follow-up

Generated: `2026-05-05T13:23:07.152207+00:00`

## Current state

- Implemented new `TimeframePairZScoreReversionStrategy` and `profit_moonshot_precious_metal_pair_aggressive_mode` covering XAU/XAG and XPT/XPD.
- The new metals mode is **not deployment-ready**: standard raw-first train/val gate is blocked, and legacy-windowed OOS is negative.
- OOS-return best remains `profit_moonshot_hourly_shock_reversion_eth_12h_mode`: OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`.
- Risk-adjusted shadow remains `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`: OOS `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`.
- Conservative legacy candidate remains `profit_moonshot_momentum_hybrid_safe_mode`; do not mark it deployment-ready without raw-first/OOS improvement.

## Precious metals results

- Tail refresh attempted for XAU/XAG/XPT/XPD (`2026-03-28` to `2026-05-05`): XAU raw archive reached `date=2026-05-04`, but XAG/XPT/XPD did not complete after a CloudFront/no-progress stall; peak RSS `1537.184 MiB`.
- Standard live-equivalent gate for `profit_moonshot_precious_metal_pair_aggressive_mode`: `blocked_missing_raw_first_market_data` for XAU/XAG/XPT/XPD train+val.
- Legacy-windowed all-four-metal split evidence:
  - train: `-0.0570%`, MDD `0.2624%`, Sharpe `-0.024200`, trades `70`, liquidations `0`
  - val: `+0.1914%`, MDD `0.4715%`, Sharpe `0.037224`, trades `70`, liquidations `0`
  - OOS: `-0.0478%`, MDD `0.0528%`, Sharpe `-0.164066`, trades `13`, liquidations `0`
- Decision: reject/not promoted. The code is useful as a tested alpha-family scaffold plus negative evidence; it is not a better candidate.

## Key artifacts

- `var/reports/profit_moonshot_20260501/current_tail_20260505/precious_metal_pair_aggressive/precious_metal_pair_aggressive_report_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/precious_metal_pair_aggressive/precious_pair_screen_top.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/precious_metal_pair_aggressive/precious_pair_available_window_split_backtest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_precious_metal_pair_aggressive_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/data_refresh/precious_metals_tail_refresh_attempt.json`
- `var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.md`

## Next priority

1. Backfill/materialize raw-first train/val/OOS for XAU/XAG/XPT/XPD before re-running metals promotion gates.
2. Add fill-aware entry scheduling/order-cancel logic for thin metals; current partial-fill realism is the main reason screen edge failed engine validation.
3. If continuing metals, start with XAU/XAG only as a diagnostic, then re-introduce XPT/XPD only if train-stable.
4. Do not claim success unless a candidate beats `profit_moonshot_hourly_shock_reversion_eth_12h_mode` on OOS return and passes live-equivalent/raw-first gates; Sharpe target remains above `1.0`.

## Verification to rerun after changes

- `uv run pytest tests/unit/test_timeframe_pair_zscore_reversion_strategy.py tests/test_pair_trading_zscore.py tests/unit/test_artifact_portfolio_mode.py::test_profit_moonshot_precious_metal_pair_mode_includes_four_metals_with_caps tests/test_live_selection_infer.py::test_infer_strategy_class_name_precious_metal_pair_mode -q`
- `uv run ruff check src/lumina_quant/strategies/timeframe_pair_zscore_reversion.py src/lumina_quant/strategies/pair_trading_zscore.py src/lumina_quant/strategies/artifact_portfolio_mode.py src/lumina_quant/live_selection.py tests/unit/test_timeframe_pair_zscore_reversion_strategy.py tests/test_pair_trading_zscore.py tests/unit/test_artifact_portfolio_mode.py tests/test_live_selection_infer.py`
- `uv run python -m compileall -q src tests`
