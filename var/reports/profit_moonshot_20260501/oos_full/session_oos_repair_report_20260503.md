# Profit Moonshot OOS Repair / Raw-first Gate Report — 2026-05-03

Generated: `2026-05-03T06:16:42.518917+00:00`

## Decision

- Decision: **no deployment-ready candidate** under the stricter train/val/OOS gate.
- Conservative research candidate retained: `profit_moonshot_momentum_hybrid_safe_mode`.
- Do **not** promote `boost` or `vol_target_132`: both lose on repaired raw-first OOS despite positive validation.
- No simple gross exposure increase was made.

## Data refresh and OOS coverage

- Latest tail refresh status: `completed` at cutoff `2026-05-03T04:10:03Z`.
- Refresh peak RSS: `2584.5 MiB`; workers: `1`.
- OOS window used for backtests: `2026-03-01` through `2026-05-02` inclusive. `2026-05-03` was refreshed as current tail but excluded from day-level OOS because it was not a complete UTC day.
- OOS raw-first materialized coverage after repair:

| symbol | committed / required OOS days | missing | complete |
|---|---:|---:|---|
| `BNB/USDT` | 63 / 63 | 0 | `True` |
| `BTC/USDT` | 63 / 63 | 0 | `True` |
| `ETH/USDT` | 63 / 63 | 0 | `True` |
| `SOL/USDT` | 63 / 63 | 0 | `True` |
| `TRX/USDT` | 63 / 63 | 0 | `True` |

## Raw-first train/val/OOS results

| mode | train ret | train MDD | val ret | val MDD | OOS ret | OOS MDD | OOS Sharpe | OOS Sortino | OOS trades | max RSS | deployment gate | failure reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | 12.3695% | +0.2837% | 1.0438% | -0.3832% | 3.3917% | -0.003750 | -0.003777 | 136 | 5,123,720 KB | FAIL | oos_total_return_not_positive, oos_sharpe_not_positive, oos_sortino_not_positive |
| `profit_moonshot_adaptive_momentum_vol_target_132_mode` | -2.1161% | 14.0900% | +0.4176% | 1.1911% | -0.4720% | 4.3259% | -0.003135 | -0.003163 | 37 | 6,325,604 KB | FAIL | oos_total_return_not_positive, oos_sharpe_not_positive, oos_sortino_not_positive |
| `profit_moonshot_adaptive_momentum_boost_mode` | -2.9948% | 18.0211% | +0.5091% | 1.3583% | -0.5279% | 5.4134% | -0.002253 | -0.002287 | 37 | 6,269,180 KB | FAIL | train_total_return_below_-2.5%, train_mdd_above_15%, oos_total_return_not_positive, oos_sharpe_not_positive, oos_sortino_not_positive |

## Failure diagnosis

- `profit_moonshot_momentum_hybrid_safe_mode`: train/val remains the best conservative compromise, but repaired OOS return is negative and OOS Sharpe/Sortino are negative; keep only as research candidate.
- `profit_moonshot_adaptive_momentum_vol_target_132_mode`: validation is stronger than `hybrid_safe`, but OOS return is more negative and drawdown is larger; not deployment-ready.
- `profit_moonshot_adaptive_momentum_boost_mode`: still the raw validation leader, but train return/MDD fail robustness and repaired OOS is also negative; do not promote.

## Code/data path fixes made

- Revalidation now defaults OOS end to the latest complete UTC day, avoiding an impossible current-day materialized coverage requirement.
- Raw materialization ignores stale symbol checkpoints when an explicit historical range starts after the checkpoint, allowing repaired OOS days to commit.
- Raw parquet loading now ignores `part-*.corrupt-*.parquet` quarantine files instead of sending them through the fast scan path.
- Backfill materializer run records now expose materializer success/status in JSON logs for clearer failure triage.

## Next work

- Do not keep optimizing adaptive-momentum blend weights unless OOS behavior changes materially.
- Next alpha family should be funding/OI/taker-flow/liquidation replay through live-equivalent `MARKET_WINDOW`; no success claim until train/val/OOS all pass.
