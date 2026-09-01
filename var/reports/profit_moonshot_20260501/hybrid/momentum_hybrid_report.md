# Momentum Hybrid Live-Equivalent Report

- Generated: `2026-05-02T14:27:04.275028+00:00`
- Goal: combine only the candidates with some live-equivalent promise, without simple gross-exposure escalation.
- Baseline gate: `+0.2649%` (`profit_moonshot_adaptive_momentum_mode`).
- Boost raw target: `+0.5091%` (`profit_moonshot_adaptive_momentum_boost_mode`) but train return/MDD remain fragile.
- Data note: OOS split is still `skipped_oos_data_incomplete`; this is train/val live-equivalent evidence, not full deployment readiness.

## Result

- Best conservative hybrid: `profit_moonshot_momentum_hybrid_safe_mode`.
- Best validation-return hybrid: `profit_moonshot_momentum_hybrid_safe_mode`.
- Best validation-return overall remains: `profit_moonshot_adaptive_momentum_boost_mode`.
- Deployment status: **research/live-equivalent candidate**, not full deployment-ready promotion until raw-first OOS coverage is available.

## Comparison

| Mode | Role | Weights | Train ret | Train MDD | Train trades | Val ret | Val MDD | Val Sharpe | Val Sortino | Val trades | Score | User gate | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `profit_moonshot_adaptive_momentum_mode` | baseline | - | -1.8628% | +12.0686% | 356 | +0.2649% | +0.7544% | 0.012417 | 0.012036 | 52 | n/a | FAIL | incumbent baseline |
| `profit_moonshot_adaptive_momentum_boost_mode` | raw validation leader / fragility target | - | -2.9948% | +18.0211% | 361 | +0.5091% | +1.3583% | 0.014751 | 0.014527 | 56 | n/a | FAIL | research-candidate only: high val, fragile train |
| `profit_moonshot_adaptive_momentum_vol_target_132_mode` | safer high-val sleeve | - | -2.1161% | +14.0900% | n/a | +0.4176% | n/a | n/a | n/a | n/a | n/a | FAIL | source sleeve: preferred return threshold, MDD below 15% |
| `profit_moonshot_adaptive_momentum_governed_mode` | train-return stabilizer | - | +1.9280% | +14.8471% | n/a | +0.1695% | n/a | n/a | n/a | n/a | n/a | FAIL | source sleeve: positive train, lower val |
| `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | drawdown damper | - | -0.4211% | +1.0642% | n/a | +0.0000% | n/a | n/a | n/a | n/a | n/a | FAIL | source sleeve: near-flat but suppresses risk |
| `profit_moonshot_momentum_hybrid_return_mode` | tested hybrid | boost_mode: 60%, vol_target_132_mode: 25%, governed_mode: 15% | -1.7990% | +16.0694% | 909 | +0.2687% | +1.0144% | 0.009768 | 0.009451 | 134 | 10.241826 | FAIL | - |
| `profit_moonshot_momentum_hybrid_safe_mode` | tested hybrid | boost_mode: 35%, vol_target_132_mode: 35%, governed_mode: 20%, asym_dynamic_mode: 10% | -1.3551% | +12.3695% | 1185 | +0.2837% | +1.0438% | 0.010168 | 0.009833 | 183 | 10.654230 | PASS | - |
| `profit_moonshot_momentum_hybrid_core_mode` | tested hybrid | boost_mode: 40%, vol_target_132_mode: 40%, governed_mode: 15%, asym_dynamic_mode: 5% | -1.4765% | +14.1011% | 932 | +0.2550% | +1.0143% | 0.009413 | 0.009093 | 138 | 10.452102 | FAIL | - |

## What changed

- Added recursive portfolio aliases for `profit_moonshot_momentum_hybrid_return_mode`, `profit_moonshot_momentum_hybrid_safe_mode`, and `profit_moonshot_momentum_hybrid_core_mode`.
- The hybrids expand existing candidate rows and scale child signals; they are not standalone gross-exposure bumps.
- `hybrid_safe` is the useful one: it beats baseline validation while improving boost train return from `-2.9948%` to `-1.3551%` and train MDD from `18.0211%` to `12.3695%`.
- `hybrid_return` failed the strict train MDD gate; `hybrid_core` failed the baseline validation-return gate.

## Runtime / resource evidence

| Mode | Wall clock | Max RSS | OOS status |
| --- | ---: | ---: | --- |
| `profit_moonshot_momentum_hybrid_return_mode` | 24:10.45 | 4,803,196 KB | `skipped_oos_data_incomplete` |
| `profit_moonshot_momentum_hybrid_safe_mode` | 24:39.81 | 4,804,884 KB | `skipped_oos_data_incomplete` |
| `profit_moonshot_momentum_hybrid_core_mode` | 23:06.02 | 4,807,492 KB | `skipped_oos_data_incomplete` |

## Next action

- If the mandate is higher than `+0.2837%` validation return, more adaptive-momentum recombination is unlikely to be enough; the next real uplift needs a new alpha family with complete historical feature coverage.
- Do not promote `boost` despite its `+0.5091%` validation result; its train loss and MDD are the exact fragility the hybrid was built to reduce.
