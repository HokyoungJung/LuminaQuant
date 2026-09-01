# Profit reboot live-equivalent validation — 2026-05-01

Mode: `profit_reboot_adaptive_momentum_mode`  
Strategy: `AdaptiveRegimeMomentumStrategy`  
Generated: `2026-05-01T07:35:08.738301Z`

## Selection result

- Alpha gate: `PASS`
- Blocking reasons: `none`
- Selection score: `10.611527`

## Train/validation engine backtest evidence

| split | total return | max drawdown | Sharpe | Sortino | trades | liquidations |
|---|---:|---:|---:|---:|---:|---:|
| train 2025-01-01..2025-12-31 | -1.8485% | 12.0609% | 0.0044 | 0.0046 | 353 | 0 |
| val 2026-01-01..2026-02-28 | +0.2551% | 0.7602% | 0.0120 | 0.0116 | 51 | 0 |

## Design notes

- Uses one bar per `MARKET_WINDOW` decision tick and does not require `TimeframeAggregator`.
- Uses low-turnover 6-hour regime momentum (`lookback_bars=360`, `rebalance_bars=72`) with small live-equivalent exposure caps.
- Cash/fallback and zero-trade candidates are excluded from alpha ranking by the updated profit gate.

## Artifacts

- JSON: `var/reports/profit_reboot_20260501/profit_reboot_live_equivalent_report_latest.json`
- Train checkpoint: `var/reports/profit_reboot_20260501/balanced_train_checkpoint.json`
- Val checkpoint: `var/reports/profit_reboot_20260501/balanced_val_checkpoint.json`
