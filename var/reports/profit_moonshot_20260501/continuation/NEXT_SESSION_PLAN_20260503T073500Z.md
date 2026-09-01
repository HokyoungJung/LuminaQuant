# Profit Moonshot continuation — derivatives taker-flow replay

Generated: `2026-05-03T07:36:41.813258Z`

## Current state

- `private-main` started from `private/main` commit `755c04708794ae960f03f81cf85a76ce3aa3f762`.
- No duplicate backtest processes were left running; all counted runs were one-mode-at-a-time.
- Latest data tail refresh completed at `2026-05-03T04:10:03Z` with peak RSS `2584.53 MiB`; current partial UTC day was not counted as complete-day OOS.
- Raw aggTrade taker-flow replay was backfilled for BTC/ETH/SOL from `2025-01-01` through `2026-05-02` with `6,303,687` feature rows and zero missing raw days.
- Feature lookup is now scoped to the split/chunk window to keep RSS below the 8GB guard.

## Candidate decision

- No deployment-ready candidate.
- Best new candidate: `profit_moonshot_derivatives_taker_flow_sparse_mode`, **shadow/review only**.
- Retain `profit_moonshot_momentum_hybrid_safe_mode` only as conservative research baseline; repaired OOS remains negative.

## Fresh raw-first results

| mode | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | status |
|---|---:|---:|---:|---:|---:|---|
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | +0.2837% | -0.3832% | +3.3917% | -0.003750 | 보수 연구 후보 유지 / OOS 실패 |
| `profit_moonshot_derivatives_taker_flow_mode` | -1.4181% | -0.1302% | -0.0059% | +1.1541% | 0.000136 | 실패: val/OOS 약함 |
| `profit_moonshot_derivatives_taker_flow_sparse_mode` | -0.3765% | +0.0799% | +0.0247% | +0.8590% | 0.001444 | 최고 신규 shadow 후보 / deployment 불가 |

## Key artifacts

- `var/reports/profit_moonshot_20260501/derivatives_oos/session_derivatives_taker_flow_report_20260503.md`
- `var/reports/profit_moonshot_20260501/derivatives_oos/session_derivatives_taker_flow_report_20260503.json`
- `var/reports/profit_moonshot_20260501/feature_replay/raw_taker_flow_backfill_top3_20260503.json`
- `var/reports/profit_moonshot_20260501/feature_replay/support_inventory_after_taker_flow_20260503.json`
- `var/reports/profit_moonshot_20260501/derivatives_oos/profit_moonshot_derivatives_taker_flow_sparse_mode/live_equivalent_revalidation_latest.json`

## Next priority

1. Repair/source liquidation event replay; current inventory has `0` liquidation rows for all tracked symbols.
2. Source longer OI history or remove OI dependency with a clearly named funding+taker-only alpha contract.
3. Reduce partial-fill churn/order-size mismatch before another live-equivalent run; sparse mode still shows partial-fill noise despite lower cadence.
4. Require materially positive OOS, not just `selection_eligible=True`, before deployment claims.
