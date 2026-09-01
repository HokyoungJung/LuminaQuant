# Profit Moonshot Derivatives Taker-flow Session Report — 2026-05-03

Generated: `2026-05-03T07:36:41.813258Z`

## Decision

- Decision: **no deployment-ready candidate** under the stricter raw-first live-equivalent/OOS gate.
- Best new candidate: `profit_moonshot_derivatives_taker_flow_sparse_mode` as **shadow/review only**, not deployment.
- Conservative research candidate retained: `profit_moonshot_momentum_hybrid_safe_mode`.
- No simple gross-exposure increase was used; the sparse candidate reduced cadence/trade count and uses `target_allocation=0.008`, `max_order_value=175`.

## Data refresh

- Latest tail refresh: `completed` at cutoff `2026-05-03T04:10:03Z` from `binance_raw_aggtrades`.
- OHLCV derivation: `derived_from_raw_aggtrades`; feature symbols `5`, OHLCV symbols `5`.
- Refresh peak RSS: `2584.53 MiB`; artifact `var/reports/profit_moonshot_20260501/latest/data_refresh_20260503_current.json`.

## Feature replay status

- Raw aggTrade taker-flow backfill: `var/reports/profit_moonshot_20260501/feature_replay/raw_taker_flow_backfill_top3_20260503.json`
- Symbols: `BTC/USDT, ETH/USDT, SOL/USDT`; window `2025-01-01`–`2026-05-02`; cadence `20s`.
- Raw rows `1,986,107,402`; feature rows `6,303,687`; missing raw days `0`; peak RSS `4192.59 MiB`.

| symbol | taker-flow rows | taker first | taker last | OI first | liquidation rows |
|---|---:|---|---|---|---:|
| `BTCUSDT` | 2,101,668 | `2025-01-01T00:00:20+00:00` | `2026-05-03T00:00:00+00:00` | `2026-03-08T00:00:00+00:00` | 0 |
| `ETHUSDT` | 2,102,583 | `2025-01-01T00:00:20+00:00` | `2026-05-03T00:00:00+00:00` | `2026-03-07T00:00:00+00:00` | 0 |
| `SOLUSDT` | 2,099,436 | `2025-01-01T00:58:40+00:00` | `2026-05-03T00:00:00+00:00` | `2026-03-07T00:00:00+00:00` | 0 |

Key caveat: taker-flow replay is now real for BTC/ETH/SOL, but liquidation replay remains `0` rows and OI starts only in March 2026, so train/val OI still uses the documented volume proxy.

## Raw-first coverage

- Preflight output: `var/reports/profit_moonshot_20260501/derivatives_oos/profit_moonshot_derivatives_taker_flow_mode_preflight/live_equivalent_revalidation_latest.json`
- BTC/ETH/SOL all passed split coverage: train `365/365`, val `59/59`, OOS `63/63`, missing days `0`.

## Train / Val / OOS evidence

| mode | train ret | train MDD | val ret | val MDD | OOS ret | OOS MDD | OOS Sharpe | OOS Sortino | OOS trades | max RSS | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | +12.3695% | +0.2837% | +1.0438% | -0.3832% | +3.3917% | -0.003750 | -0.003777 | 136 | - | 보수 연구 후보 유지 / OOS 실패 |
| `profit_moonshot_derivatives_taker_flow_mode` | -1.4181% | +1.5123% | -0.1302% | +0.3348% | -0.0059% | +1.1541% | 0.000136 | 0.000085 | 488 | 2,467,008 KB | 실패: val/OOS 약함 |
| `profit_moonshot_derivatives_taker_flow_sparse_mode` | -0.3765% | +1.0365% | +0.0799% | +0.0526% | +0.0247% | +0.8590% | 0.001444 | 0.000598 | 140 | 2,469,756 KB | 최고 신규 shadow 후보 / deployment 불가 |

## Why failures are failures

- `profit_moonshot_derivatives_taker_flow_mode`: scoped rerun stayed under RSS but failed validation (`val -0.1302%`) and OOS (`-0.0059%`), so it is rejected.
- Unscoped first run of the same mode was killed: max RSS `9,064,676 KB`, exit `143`; this caused the scoped feature lookup patch and is recorded as failed evidence, not success.
- `profit_moonshot_derivatives_taker_flow_sparse_mode`: passes the repo alpha selection row (`selection_eligible=True`) and is OOS-positive, but the OOS return is only `+0.0247%` with Sharpe `0.00144`; liquidation/OI replay is incomplete, so strict deployment is blocked.
- `profit_moonshot_momentum_hybrid_safe_mode`: still the conservative research baseline, but repaired OOS is `-0.3832%`, so it is not deployment-ready.

## Verification

- Duplicate/leftover process check: no active `revalidate_live_equivalent_candidates`, backfill, materialize, or backtest process.
- `uv run ruff check` → `All checks passed!`
- Targeted pytest suite → `26 passed in 0.47s`.
- Backtest RSS: scoped dense mode `2,467,008 KB`, sparse mode `2,469,756 KB`; both below 8GB.

## Next work

- Do not promote or live-select any moonshot candidate from this round.
- Next priority is true liquidation replay/provider repair and longer OI history; until then liquidation/OI alpha claims are under-supported.
- If no liquidation/OI history can be sourced, design a funding+taker-only family with an explicit no-OI/no-liquidation contract and require materially positive OOS after costs.
