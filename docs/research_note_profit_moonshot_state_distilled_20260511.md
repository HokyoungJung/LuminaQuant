# Research Note — Profit Moonshot State-Distilled Non-Calendar Strategy

Date: 2026-05-11 KST
Repo: `/home/hoky/Quants-agent/LuminaQuant`
Pushed green head: `private/main 7e451311757a1ce0e43bebaec0a24b3746dbcb65`
Baseline to preserve: `02f4520cf906f48089b8852c2651a0f1e4bd0c1c`

## Summary

The old current-base result is not acceptable as a live strategy because it is calendar-primary. A new non-calendar family, `state_distilled_leadership_unwind`, was implemented to turn the rejected calendar result into a market-state hypothesis rather than a date rule.

The new family uses observable state variables only: broad-market anchor, cross-sectional leadership/laggard ranking, residual z-score, fast momentum, flow, open interest, funding, and regime context. It does not use month/day/hour calendar entry rules.

Current conclusion: strategy validity improved, but deployable improvement is not achieved yet. The best strict zero-liquidation valid candidate has clean train/validation/OOS liquidation behavior, but OOS return and return/MDD do not beat the current-base reference.

## Latest Artifacts

- Replay: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_leadership_unwind_20260511/`
- Liquidation-aware validation: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_state_distilled_20260511/`
- Plan: `.omx/plans/profit_moonshot_state_distilled_next_plan_20260511.md`
- Session handoff: `docs/session_handoff_20260511_profit_moonshot_state_distilled_leadership_unwind.md`

## Replay Results

- Spec count: 648
- Replay survivor count: 0
- Success candidate count: 0
- Peak RSS: about 254 MiB

Best train/validation retune candidate:

`fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600`

## Strict Zero-Liquidation 4x Result

| Split | Return | MDD | Sharpe | Sortino | Calmar | Liquidations | Min margin buffer | Min margin ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 32.9431% | 9.4768% | 1.9463 | 2.0182 | 3.4766 | 0 | 9740.5206 | 177.5034 |
| validation | 11.6925% | 3.1028% | 4.9606 | 5.9849 | 31.6786 | 0 | 9959.0876 | 204.1618 |
| locked-OOS | 2.4722% | 2.5328% | 1.5131 | 1.8815 | 5.6787 | 0 | 9875.3540 | 208.3866 |

Comparison to current-base reference:

- Candidate OOS return: 2.4722%
- Current-base reference OOS return: 6.4281%
- Candidate OOS return/MDD: 0.9761
- Current-base reference OOS return/MDD: 6.9169
- Deployable success: false

## 5x/6x Diagnostic Results

These are diagnostic only. They do not pass strict zero-liquidation promotion because train liquidations occur. No account wipeout was observed.

| Leverage | Train return | Train MDD | Train liquidation | Validation return | Validation liquidation | OOS return | OOS MDD | OOS liquidation | Wipeout |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5x | 34.3497% | 15.5461% | 2 | 14.7888% | 0 | 3.0887% | 3.1589% | 0 | 0 |
| 6x | 34.8783% | 22.3058% | 4 | 17.9560% | 0 | 3.7036% | 3.7832% | 0 | 0 |

## Main Risks

1. Hidden calendar proxy: state signals could accidentally encode the same historical date episode.
2. OOS economics too weak: valid candidate does not beat current-base reference.
3. False discovery risk: 648 specs, zero all-gate survivors.
4. Liquidation policy mismatch: user may tolerate nonfatal liquidation, but strict live promotion blocks any liquidation.
5. Margin model approximation: Binance USDT perpetual model is conservative but not exact.
6. Short locked-OOS window and regime brittleness.
7. Universe concentration and symbol/date-block dependence.
8. OOS leakage risk from repeated diagnostics.

## Next Research Direction

Use the invalid calendar tuple only as a hypothesis generator, not as a selection target. Build market-observable mechanisms that could naturally produce a similar result shape:

1. Crowded leadership unwind v2.
2. Funding/OI exhaustion carry reversal.
3. Beta-hedged residual reversion.
4. Dispersion compression breakout/unwind.
5. Volatility/regime/margin-buffer exposure scaler.

Selection must remain train/validation-only. Locked-OOS is gate/report-only after candidate freeze.
