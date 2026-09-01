# Profit Moonshot — useful-alpha execution status

Generated: `2026-05-06T10:50:00Z`

## Current decision

No new successful alpha was promoted. The strict gate remains unmet: OOS return must exceed `+0.8284%`, OOS MDD must improve on `0.1778%`, and OOS Sharpe must be `>1.0` with separated train/val/OOS raw-first evidence.

## Evidence report

Read first next session:

- `var/reports/profit_moonshot_20260501/current_tail_20260506/useful_alpha_execution_report_20260506.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260506/eth_shock_filter_replay/eth_shock_filter_replay_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260506/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode_context_fix/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260506/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode/live_equivalent_revalidation_latest.json`

## Tested and rejected

| mode | OOS return | OOS MDD | OOS Sharpe | decision |
|---|---:|---:|---:|---|
| `profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode` | `+0.5871%` | `0.3203%` | `0.070688` | reject: below return/MDD/Sharpe gates |
| `profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode` | `+0.3221%` | `0.9275%` | `0.014160` | reject: below return/MDD/Sharpe gates |

## Ground truth remains

- OOS-return best: `profit_moonshot_hourly_shock_reversion_eth_12h_mode` — OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`.
- Sharpe/MDD shadow: `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` — OOS `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`.
- Metals direct alpha remains rejected due raw-first coverage and OOS failure.

## Next-session warning

Do not spend another full backtest on the remaining replay survivors unless a new replay edge beats the two full-tested survivors or new liquidation data appears. Current BTC/ETH/SOL liquidation rows are `0`, and taker-flow tails end at `2026-05-03T00:00:00+00:00`.
