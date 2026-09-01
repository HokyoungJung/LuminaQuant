# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-03T13:21:31.986197Z`
- latest available data: `2026-06-01T06:30:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `12` / `72` / `192`
- selection/refit input: train + 2M validation only; OOS month is evaluated after frozen fold params.

## Fold schedule

| Fold | Refit | Train | Validation | OOS |
| --- | --- | --- | --- | --- |
| `2025-09` | `2025-09-01T00:00:00` | `2025-01-01T00:00:00 → 2025-06-30T23:30:00` | `2025-07-01T00:00:00 → 2025-08-31T23:30:00` | `2025-09-01T00:00:00 → 2025-09-30T23:30:00` |
| `2025-10` | `2025-10-01T00:00:00` | `2025-01-01T00:00:00 → 2025-07-31T23:30:00` | `2025-08-01T00:00:00 → 2025-09-30T23:30:00` | `2025-10-01T00:00:00 → 2025-10-31T23:30:00` |
| `2025-11` | `2025-11-01T00:00:00` | `2025-01-01T00:00:00 → 2025-08-31T23:30:00` | `2025-09-01T00:00:00 → 2025-10-31T23:30:00` | `2025-11-01T00:00:00 → 2025-11-30T23:30:00` |
| `2025-12` | `2025-12-01T00:00:00` | `2025-01-01T00:00:00 → 2025-09-30T23:30:00` | `2025-10-01T00:00:00 → 2025-11-30T23:30:00` | `2025-12-01T00:00:00 → 2025-12-31T23:30:00` |
| `2026-01` | `2026-01-01T00:00:00` | `2025-01-01T00:00:00 → 2025-10-31T23:30:00` | `2025-11-01T00:00:00 → 2025-12-31T23:30:00` | `2026-01-01T00:00:00 → 2026-01-31T23:30:00` |
| `2026-02` | `2026-02-01T00:00:00` | `2025-01-01T00:00:00 → 2025-11-30T23:30:00` | `2025-12-01T00:00:00 → 2026-01-31T23:30:00` | `2026-02-01T00:00:00 → 2026-02-28T23:30:00` |
| `2026-03` | `2026-03-01T00:00:00` | `2025-01-01T00:00:00 → 2025-12-31T23:30:00` | `2026-01-01T00:00:00 → 2026-02-28T23:30:00` | `2026-03-01T00:00:00 → 2026-03-31T23:30:00` |
| `2026-04` | `2026-04-01T00:00:00` | `2025-01-01T00:00:00 → 2026-01-31T23:30:00` | `2026-02-01T00:00:00 → 2026-03-31T23:30:00` | `2026-04-01T00:00:00 → 2026-04-30T23:30:00` |
| `2026-05` | `2026-05-01T00:00:00` | `2025-01-01T00:00:00 → 2026-02-28T23:30:00` | `2026-03-01T00:00:00 → 2026-04-30T23:30:00` | `2026-05-01T00:00:00 → 2026-05-31T23:30:00` |
| `2026-06` | `2026-06-01T00:00:00` | `2025-01-01T00:00:00 → 2026-03-31T23:30:00` | `2026-04-01T00:00:00 → 2026-05-31T23:30:00` | `2026-06-01T00:00:00 → 2026-06-01T06:30:00` |

## Aggregate ranking

| Rank | Candidate | Family | Clean | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | `True` | `False` | 61.61% | 7/10 | -9.48% | -0.57% | 1.55 | 4.28 | 27.57% |
| 2 | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | `True` | `False` | 52.06% | 6/10 | -10.69% | -0.41% | 1.38 | 3.95 | 25.76% |
| 3 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `True` | `False` | 48.42% | 6/10 | -11.38% | -0.47% | 1.32 | 3.46 | 26.05% |
| 4 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | `True` | `False` | 54.21% | 5/10 | -9.97% | -0.47% | 1.40 | 4.01 | 28.28% |
| 5 | `validation_selector:validation_calmar_mdd12` | `validation_selector` | `False` | `False` | 39.29% | 4/10 | -11.38% | -0.32% | 1.09 | 3.16 | 28.74% |
| 6 | `asset_timeframe_leverage:hybrid_v3_6` | `asset_timeframe_leverage` | `True` | `False` | 21.78% | 4/10 | -15.87% | -0.57% | 0.72 | 1.67 | 21.97% |
| 7 | `asset_timeframe_leverage:selected_optuna` | `asset_timeframe_leverage` | `True` | `False` | 18.23% | 4/10 | -22.90% | -0.31% | 0.60 | 1.18 | 29.60% |
| 8 | `asset_timeframe_leverage:hybrid_v3_5` | `asset_timeframe_leverage` | `True` | `False` | 17.82% | 4/10 | -22.90% | -0.31% | 0.60 | 1.11 | 29.60% |
| 9 | `profile_optuna:selected_optuna` | `profile_optuna` | `True` | `False` | 16.35% | 4/10 | -16.14% | -0.32% | 0.61 | 1.21 | 28.74% |
| 10 | `validation_selector:validation_sharpe_mdd10` | `validation_selector` | `False` | `False` | 13.33% | 4/10 | -11.38% | -0.35% | 0.52 | 1.58 | 28.74% |
| 11 | `profile_optuna:selected_train_validation_legal` | `profile_optuna` | `True` | `False` | 8.92% | 4/10 | -16.14% | -0.32% | 0.42 | 0.88 | 38.03% |
| 12 | `meta_portfolio:validation_inverse_mdd_top10_capped` | `meta_portfolio` | `True` | `False` | 0.82% | 4/10 | -10.43% | -0.35% | 0.19 | 0.62 | 27.25% |

## Best candidate monthly OOS detail: `cross_candidate_hybrid:hybrid_v3_6`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 63.37% | 0.57% | 7.78% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.05702627354935056, "meta_portfolio:validation_stability_top8_equal": 0.057194690691175...` |
| `2025-10` | 55.28% | 2.01% | 9.97% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.006425959543861126, "meta_portfolio:validation_stability_top8_equal": 0.00642544562335...` |
| `2025-11` | 105.69% | 17.79% | 16.56% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.022180866744168394, "meta_portfolio:validation_stability_top8_equal": 0.02211761393829...` |
| `2025-12` | 104.08% | 0.88% | 7.65% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.10780739837338203, "profile_optuna:hybrid_v3_5": 0.11025519388735118, "profile_optuna:...` |
| `2026-01` | 171.11% | 19.47% | 4.22% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.10052500102857533, "meta_portfolio:validation_inverse_mdd_top10_capped": 0.10203354654...` |
| `2026-02` | 121.03% | 29.16% | 14.68% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.11042942490503972, "profile_optuna:hybrid_v3_5": 0.10645902226119794, "profile_optuna:...` |
| `2026-03` | 137.82% | -5.91% | 10.76% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.13331933639298107, "meta_portfolio:validation_inverse_mdd_top10_capped": 0.13500497292...` |
| `2026-04` | 110.31% | -9.48% | 10.94% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.0186786231186501, "meta_portfolio:validation_inverse_mdd_top10_capped": 0.018641490965...` |
| `2026-05` | 141.38% | 1.46% | 27.57% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.0938476891220896, "meta_portfolio:validation_stability_top8_equal": 0.0935788350301899...` |
| `2026-06` | 140.94% | -0.57% | 0.55% | `hybrid_v3_6_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.0858827119088761, "meta_portfolio:validation_inverse_mdd_top10_capped": 0.084516061346...` |

### Best candidate extended metrics

- OOS comp: `61.61%`
- hit rate: `7/10`
- monthly Sharpe / Sortino approx: `1.55` / `4.28`
- 5% monthly VaR / 25% CVaR: `-7.88%` / `-5.32%`
- avg gain / avg loss: `10.19%` / `-5.32%`
- gain/loss ratio: `1.91`
- max loss streak: `2`
- mean/min validation: `115.10%` / `55.28%`

## Timeframe coverage

| Timeframe | Symbols with rows | Symbols skipped | Median rows | Latest |
| --- | ---: | ---: | ---: | --- |
| `30m` | 69 | 0 | 2674.0 | `2026-06-01T06:30:00` |
| `1h` | 69 | 0 | 1337.0 | `2026-06-01T06:00:00` |
| `2h` | 69 | 0 | 668.0 | `2026-06-01T04:00:00` |
| `4h` | 69 | 0 | 333.0 | `2026-06-01T00:00:00` |
| `6h` | 69 | 0 | 222.0 | `2026-06-01T00:00:00` |
| `8h` | 69 | 0 | 166.0 | `2026-05-31T16:00:00` |
| `12h` | 69 | 0 | 110.0 | `2026-05-31T12:00:00` |
| `1d` | 69 | 0 | 55.0 | `2026-05-31T00:00:00` |

## Interpretation guardrails

- This is still research/paper-testnet evidence, not real-money approval.
- The latest OOS month can be partial when the data feed ends before month-end.
- If a candidate has a negative validation fold or low OOS consistency, prefer shadow monitoring over allocation.
