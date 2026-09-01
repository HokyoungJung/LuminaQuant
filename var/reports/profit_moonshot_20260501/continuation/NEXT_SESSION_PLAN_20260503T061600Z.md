# Profit Moonshot continuation — after OOS raw-first repair

Generated: `2026-05-03T06:16:00Z`

## Current state

- `private-main` was validated from `private/main` latest at session start.
- Latest data tail refreshed to `2026-05-03T04:10:03Z` with peak RSS `2584.5 MiB`.
- OOS raw-first materialized coverage was repaired for `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT` through latest complete UTC day `2026-05-02`.
- OOS preflight now reports all five symbols `63/63` committed days for `2026-03-01`–`2026-05-02`.

## Candidate decision

- No deployment-ready candidate was found after repaired raw-first OOS validation.
- Keep `profit_moonshot_momentum_hybrid_safe_mode` only as the conservative research candidate, not a live promotion.
- `profit_moonshot_adaptive_momentum_boost_mode` remains raw validation leader but fails train robustness and OOS.

## Fresh raw-first results

| mode | train ret | train MDD | val ret | OOS ret | OOS MDD | OOS Sharpe | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | 12.3695% | +0.2837% | -0.3832% | 3.3917% | -0.003750 | FAIL |
| `profit_moonshot_adaptive_momentum_vol_target_132_mode` | -2.1161% | 14.0900% | +0.4176% | -0.4720% | 4.3259% | -0.003135 | FAIL |
| `profit_moonshot_adaptive_momentum_boost_mode` | -2.9948% | 18.0211% | +0.5091% | -0.5279% | 5.4134% | -0.002253 | FAIL |

## Key artifacts

- `var/reports/profit_moonshot_20260501/oos_full/session_oos_repair_report_20260503.md`
- `var/reports/profit_moonshot_20260501/oos_full/session_oos_repair_report_20260503.json`
- `var/reports/profit_moonshot_20260501/latest/data_refresh_20260503_current.md`
- `var/reports/profit_moonshot_20260501/oos_probe/raw_first_oos_materialize_btc_sol_20260319_20.json`

## Next priority

Implement/verify a genuinely new alpha family only after funding/OI/taker-flow/liquidation feature-point replay is available inside the live-equivalent path. Do not count train/val-only or non-OOS evidence as success.