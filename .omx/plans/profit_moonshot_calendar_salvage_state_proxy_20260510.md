# Plan/result: calendar salvage through causal state proxies — 2026-05-10

## Objective
Replace rejected calendar-primary TRX/ETH sleeves with live-causal state rules and determine whether similar performance survives.

## Acceptance checks
- No month/day calendar selection in new proxy families.
- Train/validation/OOS replay is generated under memory < 8 GiB.
- Keep locked OOS report-only; no promotion if current-base gates fail.

## Result
- Implemented families: `state_momentum_proxy`, `state_relative_strength_spread`, `fresh_state_trx_longonly_*`, `fresh_state_trx_dual_mom_*`.
- Combined replay: `var/reports/profit_moonshot_20260501/calendar_salvage_state_proxy_20260510/combined_replay/fresh_start_overhaul_replay_latest.json` with `7092` specs, RSS `321.875 MiB`.
- Best salvage seed: `fresh_state_trx_longonly_lb72_z050_ret60_h168_ls800_tp450`; train `+0.50%`, val `+0.20%`, OOS `+1.09%`, OOS MDD `0.22%`, OOS Sharpe `4.94`.
- Decision: research-only keep; live promotion blocked until current-base/liquidation-aware gates pass.

Detailed report: `var/reports/profit_moonshot_20260501/calendar_salvage_state_proxy_20260510/calendar_salvage_state_proxy_latest.md`