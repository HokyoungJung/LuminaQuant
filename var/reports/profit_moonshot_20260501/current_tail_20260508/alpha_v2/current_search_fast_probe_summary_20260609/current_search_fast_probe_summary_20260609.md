# Current search fast probe summary — 2026-06-09

## Verdict

- **NO PROMOTION.** Fast probes remain weak on latest folds.
- Cost assumption: default runner `10bps` round-trip.
- Allocation remains `0%`.
- Speed path added: `--families`, `--leverages`, `--fold-workers`, `--max-candidate-rows-output`, eligible-first heap cap.

## Probe results

| Probe | Comp | Ann | Pos | MDD | PF | Rows | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `combined_4symbol_3fold_newalpha` | -3.24% | -12.34% | 0/3 | 3.24% | 0.00 | 1500 | `1:05.69` |
| `family_kalman_residual` | 0.66% | 2.66% | 1/3 | 0.00% | 0.00 | 1500 | `0:38.94` |
| `family_btc_beta_residual` | 0.00% | 0.00% | 0/0 | 0.00% | 0.00 | 0 | `0:14.56` |
| `family_kalman_vol_trend` | -1.86% | -7.23% | 1/3 | 11.28% | 0.93 | 1500 | `0:28.86` |
| `family_standardized_ridge` | -0.22% | -1.29% | 1/2 | 1.59% | 0.88 | 384 | `0:19.99` |
| `family_lead_lag` | 2.60% | 10.82% | 1/3 | 2.38% | 2.13 | 864 | `0:24.33` |
| `family_xs_vol_adjusted_momentum` | -3.67% | -13.90% | 1/3 | 3.74% | 0.02 | 1500 | `0:37.02` |
| `combined_4symbol_candidate_dump` | -3.24% | -12.34% | 0/3 | 3.24% | 0.00 | 2400 | `1:36.10` |
| `core10_momentum_probe` | -5.41% | -19.93% | 1/3 | 9.34% | 0.49 | 3000 | `3:56.46` |

## Best/least bad observations

- Best latest 3-fold smoke was `family_lead_lag`: `+2.60%` comp but only `1/3` positive; not enough.
- `indicator_kalman_residual_reversion`: `+0.66%` comp and `1/3` positive; too weak.
- Combined/new alpha probes selected train/validation winners that decayed sharply in locked OOS.
- `cross_sectional_vol_adjusted_momentum` is theory-plausible but failed this latest 3-fold smoke (`-3.67%`).

## Next

- Keep using 3-fold family smoke before any expensive 10-fold run.
- Add/try new alpha families, but promote only after current-search fresh-forward + 10bps cost/fill telemetry.
- Do not tune directly on this locked OOS; treat these as rejection/diagnostic records.
