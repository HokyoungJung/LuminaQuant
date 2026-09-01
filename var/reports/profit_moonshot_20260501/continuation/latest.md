# Profit Moonshot — useful-alpha execution status

Generated: `2026-05-07T10:01:28.645017Z`

## Current decision

No new successful alpha was promoted. The strict gate remains unmet: OOS return must exceed `+0.8284%`, OOS MDD must improve on `0.1778%`, and OOS Sharpe must be `>1.0` with separated train/val/OOS raw-first evidence.

## Latest multiasset exchange expansion follow-up

- Report: `var/reports/profit_moonshot_20260501/current_tail_20260506/multiasset_exchange_expansion/multiasset_exchange_alpha_execution_report_20260506.md`.
- Coverage: BTC/ETH/SOL raw-first train/val complete; OOS safe end `2026-05-04` because `2026-05-05` is missing. Metals remain raw-first blocked.
- Hyperliquid read-only: `35211` public `/info` feature rows upserted; funding history replay-eligible, current OI/mark snapshot not historical.
- Tickmill/MT5 read-only: blocked (`LQ_MT5_BRIDGE_PYTHON / LQ__LIVE__MT5_BRIDGE_PYTHON is not configured.`).
- Stateful replay: `38` specs, `0` replay survivors, max RSS `6710.816 MiB`; no live-equivalent backtest run.

## Ground truth remains

- OOS-return best: `profit_moonshot_hourly_shock_reversion_eth_12h_mode` — OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`.
- Sharpe/MDD shadow: `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` — OOS `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`.
- Hyperliquid/Tickmill direct trading remains blocked until cost/session/fill/lot-size and raw-first evidence exist.
