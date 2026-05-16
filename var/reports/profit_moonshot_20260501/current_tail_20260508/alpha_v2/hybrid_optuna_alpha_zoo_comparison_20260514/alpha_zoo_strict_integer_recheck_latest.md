# Crypto/FX Alpha Zoo state real-data replay

Generated: `2026-05-16T13:25:50.357979Z`
Rows: `58845`
Signals: `5684`
Trades: `2843`
Deployable success: `True`
Reason: strict zero-liquidation lane passed OOS return, MDD, Sharpe/Sortino/smart Sortino/Calmar gates; return/MDD is diagnostic-only

## Provenance

- selection inputs: `train, validation`
- uses_locked_oos_for_selection: `False`
- locked-OOS role: `gate/report only after candidate freeze`
- current-base/calendar tuple: `hypothesis_reference_only`, not selection/promotion target
- promotion policy: OOS return must beat the current-base reference; Sharpe/Sortino/smart Sortino/Calmar must be positive; return/MDD is diagnostic report-only

## Locked-OOS report-only comparison

- candidate OOS return: `6.1108%`
- candidate OOS return/MDD: `2.553222`
- current-base reference OOS return: `6.4281%`
- current-base reference return/MDD: `6.916878`

## Strict zero-liquidation lane

- strict candidate count: `6`
- deployable candidate count: `5`

## Diagnostic nonfatal 5x/6x lane

- `5.0x`: total_liquidations `0`, min_margin_buffer `9207.6050`, promotion_allowed `False`
- `6.0x`: total_liquidations `0`, min_margin_buffer `9049.1260`, promotion_allowed `False`

## Paper-forward diagnostics

- diagnostic_only: `True`; promotion_allowed: `False`
- diagnostic leverage/allocation: `6.0x` / `10.00%`
- locked-OOS by regime: neutral: 41.10% (540)
- locked-OOS by symbol: SOL/USDT: 19.27% (126), BNB/USDT: 10.32% (128), TRX/USDT: 4.96% (139), ETH/USDT: 1.48% (132), BTC/USDT: 0.68% (15)
- locked-OOS by side: SHORT: 26.30% (259), LONG: 11.71% (281)
- locked-OOS by factor family: crypto_residual_momentum: 30.18% (184), crypto_residual_reversal: 8.42% (237), volume_vwap_pressure: -0.04% (119)
- locked-OOS by exit reason: score_exit: 41.09% (526), take_profit: 16.20% (4), end_of_sample: -0.08% (2), stop_loss: -13.87% (8)
- locked-OOS slippage_sensitivity: 0bps: 41.10%, 2.5bps: 30.12%, 5bps: 20.00%, 10bps: 2.06%, 20bps: -26.19%
- locked-OOS funding_cost_sensitivity: 0bps: 41.10%, 1bps: 40.42%, 2bps: 39.75%, 5bps: 37.75%, 10bps: 34.48%

## Memory

- peak_rss_mib: `291.738`
