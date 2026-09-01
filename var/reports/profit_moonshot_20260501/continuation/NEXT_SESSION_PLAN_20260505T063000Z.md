# Profit Moonshot continuation — better OOS candidate found

Generated: `2026-05-05T10:05:00Z`

## Current state

- Latest `private/main` continuation base before this work: `933442026ecc006f833087dd6164f51e62923e2a`.
- Latest data tail remains refreshed through `2026-05-05T04:14:33Z`; OOS materialized coverage is complete through `2026-05-04`.
- The prior BTC→ETH lead-lag result is downgraded to weak shadow only: OOS `+0.2910%`, Sharpe `0.004059`, MDD `7.0817%`.
- New best current-tail candidate: `profit_moonshot_hourly_shock_reversion_eth_12h_mode`.
  - train `+2.4523%`, train MDD `2.1092%`, trades `264`, liq `0`
  - val `+0.6323%`, val MDD `0.4377%`, trades `47`, liq `0`
  - OOS `+0.8284%`, OOS MDD `0.2819%`, Sharpe `0.100651`, Sortino `0.128321`, trades `30`, liq `0`
- Sizing constraint respected: `target_allocation=0.008`, `max_order_value=175.0`; no gross exposure-only increase.
- Full backtests were sequential, one mode at a time; max RSS for the best run was `1300.17 MiB`.

## Candidate decision

- **Deployment-review candidate / current best:** `profit_moonshot_hourly_shock_reversion_eth_12h_mode`.
- **Weak shadow baseline:** `profit_moonshot_leadlag_slow_diffusion_mode`.
- **Conservative fallback only:** `profit_moonshot_momentum_hybrid_safe_mode`; still OOS negative.
- **Rejected follow-ups:** `profit_moonshot_hourly_shock_reversion_eth_mode` (OOS `+0.2716%`, weaker than the old `+0.2910%` bar) and `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode` (OOS negative).
- Funding/taker-flow remains blocked until corrected replay produces real nonzero-flow survivors.

## Fresh raw-first results

| mode | train ret | train MDD | val ret | OOS ret | OOS MDD | OOS Sharpe | user decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `profit_moonshot_hourly_shock_reversion_eth_12h_mode` | +2.4523% | 2.1092% | +0.6323% | +0.8284% | 0.2819% | 0.100651 | new best |
| `profit_moonshot_leadlag_slow_diffusion_mode` | +3.1274% | 9.1302% | +0.6833% | +0.2910% | 7.0817% | 0.004059 | weak shadow only |
| `profit_moonshot_hourly_shock_reversion_eth_mode` | +0.7538% | 1.2379% | +1.0422% | +0.2716% | 0.5648% | 0.020248 | reject: weaker than bar |
| `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode` | +8.8076% | 19.6453% | +0.4457% | -0.3629% | 0.9596% | -0.024122 | reject |
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | 12.3695% | +0.2837% | -0.3342% | 3.4942% | -0.001411 | reject deployment |

## Key artifacts

- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/simple_hourly_alpha_screen/stateful_reversion_screen_20260505.json`

## Next priority

1. Do not dilute the result with exposure increases; next improvement must beat `+0.8284%` OOS at the same risk cap or explain any risk change explicitly.
2. Before production promotion, run one final live-selection/readiness pass that marks `profit_moonshot_hourly_shock_reversion_eth_12h_mode` as the candidate and confirms live config equivalence.
3. If continuing research, prioritize non-price feature replay only after funding/OI/taker/liquidation coverage is truly nonzero and raw-first across train/val/OOS.
