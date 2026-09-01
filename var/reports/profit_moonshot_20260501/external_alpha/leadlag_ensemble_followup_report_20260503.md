# Leadlag ensemble follow-up — 2026-05-03

## Decision

- **Reject:** `profit_moonshot_leadlag_slow_diffusion_ensemble_mode`
- **Keep promoted external alpha:** `profit_moonshot_leadlag_slow_diffusion_mode`
- **Keep conservative fallback only:** `profit_moonshot_momentum_hybrid_safe_mode`

This follow-up did **not** randomly sweep modes. `hybrid_safe` blending was rejected before a full backtest because its repaired OOS is negative. The only full run was the second raw-first leadlag survivor, added as a same-risk ensemble sleeve.

## Why the hybrid blend was not run

`hybrid_safe` repaired OOS remains -0.3832%, with OOS Sharpe -0.003750 and Sortino -0.003777. Blending that into the positive-OOS leadlag candidate would be dilution, not useful alpha discovery.

## Ensemble candidate design

- BTC/USDT -> ETH/USDT, 2h lag, 8h hold, 60% sleeve.
- SOL/USDT -> ETH/USDT, 1h lag, 8h hold, 40% sleeve.
- Both sleeves keep `target_allocation=0.008`; component weights split the same total ETH target allocation, so this is not a gross-exposure increase.
- Preflight: `ready_for_live_equivalent_backtest`, raw-first coverage complete for BTC/ETH/SOL train/val/OOS, max RSS 344,760 KB.

## Live-equivalent comparison

| mode | train ret | train MDD | val ret | val MDD | OOS ret | OOS MDD | OOS Sharpe | OOS Sortino | trades train/val/OOS | OOS liq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_safe` | -1.3551% | +12.3695% | +0.2837% | +1.0438% | -0.3832% | +3.3917% | -0.003750 | -0.003777 | 1185/183/136 | 0 |
| `leadlag_single` | +3.1274% | +9.1302% | +0.6833% | +1.2601% | +0.2209% | +1.0096% | 0.011127 | 0.014112 | 176/40/38 | 0 |
| `leadlag_ensemble` | +4.9422% | +11.3094% | +0.1924% | +2.0263% | -0.3059% | +0.7261% | -0.037976 | -0.044528 | 530/96/72 | 0 |

## Rejection reason

The ensemble passed train/val engine validation, but it fails the user-mandated OOS gate: OOS return -0.3059%, OOS Sharpe -0.037976, OOS Sortino -0.044528. It also underperforms the single BTC->ETH leadlag mode on validation return and validation Sharpe.

Resource evidence: elapsed 20:12.32, max RSS 2,501,688 KB (<8GB), one full mode backtested.
