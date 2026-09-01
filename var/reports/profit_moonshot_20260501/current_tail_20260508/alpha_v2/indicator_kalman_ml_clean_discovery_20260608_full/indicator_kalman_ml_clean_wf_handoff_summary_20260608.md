# Indicator/Kalman/standardized-ML clean WF handoff — 2026-06-08

## Verdict

- **FAIL promotion; research-only save.**
- Search hash: `ee6ecd539a6b5a8c078bd0e39f22ef0bb483d10e1ecbf97236c51b9a6fb087e8`
- Selection input: `train + validation only`; locked OOS remained report/gate only.
- Real/shadow allocation: `0%` for this line until a fresh pre-registered optimization/shadow pass exists.

## Full 10-fold clean OOS result

- OOS compounded: `-8.77%`
- Annualized approx: `-10.43%`
- Positive folds: `4/10`
- Monthly equity MDD: `28.82%`
- Max fold OOS MDD: `16.70%`
- Profit factor: `0.88`

## Selected folds

| Fold | Family | Symbol | TF | Train | Val | Locked OOS | OOS MDD |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `standardized_indicator_ridge_directional` | `SOLUSDT` | `4h` | 43.97% | 24.86% | 9.28% | 3.04% |
| `2025-10` | `standardized_indicator_ridge_directional` | `SOLUSDT` | `4h` | 34.17% | 28.29% | 14.45% | 6.15% |
| `2025-11` | `standardized_indicator_ridge_directional` | `SOLUSDT` | `4h` | 75.85% | 27.12% | -0.39% | 10.49% |
| `2025-12` | `indicator_kalman_volatility_trend` | `ADAUSDT` | `4h` | 47.40% | 18.69% | -6.97% | 11.02% |
| `2026-01` | `cross_asset_lead_lag_momentum` | `AVAXUSDT` | `4h` | 43.53% | 23.21% | 9.71% | 1.97% |
| `2026-02` | `cross_asset_lead_lag_momentum` | `AVAXUSDT` | `4h` | 63.11% | 18.95% | 0.79% | 8.28% |
| `2026-03` | `standardized_indicator_ridge_directional` | `DOGEUSDT` | `4h` | 94.31% | 21.54% | -16.11% | 16.70% |
| `2026-04` | `standardized_indicator_ridge_directional` | `AVAXUSDT` | `4h` | 127.77% | 23.83% | -13.45% | 14.95% |
| `2026-05` | `indicator_vwap_atr_bollinger_reversion` | `AVAXUSDT` | `4h` | 12.62% | 11.84% | -1.60% | 3.59% |
| `2026-06` | `deep_research_vol_managed_momentum_crash_gate` | `TONUSDT` | `4h` | 18.48% | 42.80% | -0.37% | 4.35% |

## New indicator/ML family stored-candidate diagnostics

| Family | Rows | Median OOS | Best OOS | Worst OOS | Positive rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `indicator_vwap_atr_bollinger_reversion` | 261 | -1.06% | 10.70% | -16.84% | 94 |
| `indicator_kalman_volatility_trend` | 1282 | -0.25% | 13.55% | -21.36% | 575 |
| `standardized_indicator_ridge_directional` | 967 | -0.67% | 20.03% | -32.46% | 367 |

## Resume instructions

- Do not tune on this locked OOS result; use it only as a rejection/gate record.
- If optimizing indicator/ML leaves, freeze a new train/validation-only Optuna/search space before the next locked-OOS or fresh-forward window.
- Consider robustness penalties for train/validation gaps, leverage caps, turnover/RPT, and cost sensitivity; ML must remain train-only standardized.
- Re-run clean walk-forward and add 10/15/20bps cost stress before any shadow candidate discussion.
- Keep pre-registered lagged leaf-router shadow candidate separate from this failed indicator/ML alpha line.

Raw local JSON was left uncommitted to avoid bloating git; concise summary JSON is committed for resume.
