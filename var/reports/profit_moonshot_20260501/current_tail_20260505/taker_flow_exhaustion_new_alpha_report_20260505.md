# Taker-flow exhaustion new alpha follow-up — 2026-05-05

Generated: `2026-05-05T12:10:46.009037Z`

## Decision

- Implemented a new alpha family: `TakerFlowExhaustionReversalStrategy`.
- **Do not promote it.** None of the four live-equivalent variants passed train/val/OOS with positive OOS and acceptable Sharpe.
- Current OOS-return best remains `profit_moonshot_hourly_shock_reversion_eth_12h_mode`; Sharpe/MDD shadow remains `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`.
- No gross exposure increase: all variants kept `target_allocation=0.008` and `max_order_value=175.0`.

## Raw-first screen

- Screen artifact: `var/reports/profit_moonshot_20260501/current_tail_20260505/taker_flow_exhaustion_screen/taker_flow_exhaustion_screen_20260505.json`
- Checked `1296000` parameter combinations; saved `506` survivors; peak RSS `2577.43 MiB`.
- Top raw row: `ETH/USDT`, params `{'cadence': 180, 'flow_lb': 90, 'mom_lb': 180, 'horizon': '12h', 'flow_abs': 0.14, 'shock_abs': 0.006, 'funding_cap': 0.00015, 'vol_lb': 180, 'vol_cap': 0.008, 'hour_profile': 'us_eu_overlap'}`; train edge `+0.2244%`, val edge `+0.6467%`, OOS edge `+1.0923%`, OOS trade Sharpe `1.923`.

## Full live-equivalent train/val/OOS results

| mode | train ret / MDD / trades | val ret / MDD / trades | OOS ret / MDD / Sharpe / trades | decision |
|---|---:|---:|---:|---|
| `profit_moonshot_taker_flow_exhaustion_eth_mode` | +0.0089% / 0.0048% / 4 | +0.0000% / 0.0000% / 0 | +0.0000% / 0.0000% / 0.000000 / 0 | Rejected: 7-day live-equivalent chunk phase left val/OOS with zero trades; raw cadence screen was not live-equivalent. |
| `profit_moonshot_taker_flow_exhaustion_eth_reactive_mode` | -0.0019% / 0.0753% / 147 | +0.0210% / 0.0134% / 18 | -0.0291% / 0.0364% / -0.085988 / 32 | Rejected: validation was barely positive, but train and OOS were negative after fees/one-position realism. |
| `profit_moonshot_taker_flow_exhaustion_eth_hold_mode` | -1.3901% / 1.6426% / 119 | -0.0041% / 0.0408% / 16 | -0.0875% / 0.1860% / -0.031058 / 23 | Rejected: wider exits increased train loss and kept val/OOS negative. |
| `profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode` | -0.2422% / 0.7383% / 103 | -0.3210% / 0.3237% / 18 | -0.0507% / 0.1051% / -0.036304 / 16 | Rejected: cooldown reduced churn but train/val/OOS all remained negative. |

## Risk controls implemented/tested

- true taker-flow feature required; no OHLCV proxy fallback for promotion claims.
- funding_abs_cap gate.
- realized-volatility cap.
- UTC session filter.
- volatility-scaled target allocation under the same 0.8% / $175 cap.
- wide 12h hold replay and 12h cooldown variant tested to reduce churn/fee risk.

## External material

- [Binance USDⓈ-M Futures: Taker Buy/Sell Volume](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume)
- [Binance USDⓈ-M Futures: Funding Rate History](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
- [Binance USDⓈ-M Futures: Open Interest Statistics](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- [arXiv 2212.06888: Crypto perpetual futures / funding context](https://arxiv.org/abs/2212.06888)

## Resource guard

| operation | elapsed | max RSS MiB | exit |
|---|---:|---:|---:|
| taker-flow screen | 2:20.04 | 2577.43 | 0 |
| `profit_moonshot_taker_flow_exhaustion_eth_mode` | 3:45.82 | 1278.12 | 0 |
| `profit_moonshot_taker_flow_exhaustion_eth_reactive_mode` | 5:06.60 | 1284.29 | 0 |
| `profit_moonshot_taker_flow_exhaustion_eth_hold_mode` | 5:47.85 | 1286.59 | 0 |
| `profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode` | 6:57.31 | 1283.66 | 0 |

## Next action

- Stop replaying this family as-is. The gap is not exposure; it is live-engine path/cost mismatch versus raw overlapping-event screens.
- Next alpha should either use stateful, non-overlapping screening up front or improve the existing ETH shock family with a true regime model before spending another full backtest.
## Verification

- `uv run ruff check` — passed full repo.
- `uv run python -m compileall -q src tests` — passed.
- Targeted pytest — `51 passed in 0.22s`.
- Continuation validator — passed with operator OOS override: `profit_moonshot_hourly_shock_reversion_eth_12h_mode` OOS return `0.008284`, Sharpe `0.100651`.
- Backtests ran sequentially; taker-flow full-engine max RSS stayed below 1.29 GiB and below the 8 GiB guard.
