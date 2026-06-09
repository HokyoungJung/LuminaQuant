# Indicator/Kalman/ML robust selector generated-candidate summary — 2026-06-09

## Verdict

- **Improved research result, but no promotion.**
- Selector status: `post_failure_research_variant_requires_fresh_forward`.
- Real/shadow allocation remains `0%`.
- Raw 386MB JSON is kept local only; this concise summary is commit-safe.

## Capped generated-candidate result

- Candidate rows retained: `100,000`
- Candidate cap note: `100,000 candidate rows retained after the runner cap; observed cap is 10,000/fold over 10 folds. Treat as a capped generated-candidate pass, not an exhaustive proof, unless rerun uncapped.`
- Selection policy: `robust_train_validation_v1`
- OOS compounded: `22.14%`
- Annualized approx: `27.12%`
- Positive folds: `6/10`
- Monthly equity MDD: `3.10%`
- Max fold OOS MDD: `9.89%`
- Profit factor: `4.95`

Prior default selector on same alpha line: `-8.77%` comp / `-10.43%` ann / `4/10` positive / PF `0.88`.

## Search-space caveat

- Source run hash: `ee6ecd539a6b5a8c078bd0e39f22ef0bb483d10e1ecbf97236c51b9a6fb087e8`
- Current code hash after `btc_beta_residual_momentum`: `57121f6a8ade6faeaf1a83b06276728a8f3590d320d5af501ce3115e9b260a82`
- This report is retained robust-selector evidence, not a current full-search proof; current code also adds beta-residual momentum and active-policy-aware cap sorting, so rerun current search space before promotion or exhaustive comparison.

## Live realism diagnostics

- live plausibility: `not_supported`
- mean validation return: `20.10%`
- mean locked-OOS return: `2.10%`
- positive locked-OOS fold share: `0.60`
- min validation trade events: `13`
- max validation Sharpe: `7.46`
- blockers: `continuous_position_state_across_split_boundaries`, `continuous_position_state_split_simulation_not_live_equivalent`, `fresh_forward_required_before_promotion`, `robust_selector_is_post_failure_diagnostic_requires_fresh_forward`, `selected_rows_not_ready_for_real_money`, `some_validation_samples_below_30_trade_events`, `validation_sharpe_too_high_for_live_assumption_without_forward_fill_telemetry`, `validation_to_locked_oos_decay_large`

## Fold selections

| Fold | Family | Symbol | TF | Train | Val | Ratio | Locked OOS | OOS MDD |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `2025-09` | `cross_asset_lead_lag_momentum` | `ADAUSDT` | `1h` | 6.37% | 15.72% | 0.41 | 0.56% | 0.83% |
| `2025-10` | `standardized_indicator_ridge_directional` | `XRPUSDT` | `4h` | 34.53% | 23.44% | 1.47 | 8.44% | 4.72% |
| `2025-11` | `standardized_indicator_ridge_directional` | `TONUSDT` | `1h` | 3.55% | 13.45% | 0.26 | 2.09% | 3.98% |
| `2025-12` | `indicator_kalman_volatility_trend` | `XRPUSDT` | `4h` | 14.98% | 24.38% | 0.61 | -3.10% | 4.29% |
| `2026-01` | `cross_asset_lead_lag_momentum` | `AVAXUSDT` | `4h` | 43.53% | 23.21% | 1.88 | 9.71% | 1.97% |
| `2026-02` | `cross_asset_lead_lag_momentum` | `AVAXUSDT` | `4h` | 50.30% | 18.59% | 2.71 | -0.98% | 9.89% |
| `2026-03` | `volatility_squeeze_breakout` | `ETHUSDT` | `30m` | 4.53% | 13.88% | 0.33 | 0.34% | 2.75% |
| `2026-04` | `feature_taker_flow_exhaustion_reversal` | `ETHUSDT` | `4h` | 6.74% | 13.13% | 0.51 | 5.16% | 3.83% |
| `2026-05` | `indicator_vwap_atr_bollinger_reversion` | `ADAUSDT` | `4h` | 8.65% | 12.42% | 0.70 | -0.85% | 3.85% |
| `2026-06` | `deep_research_vol_managed_momentum_crash_gate` | `TONUSDT` | `4h` | 18.48% | 42.80% | 0.43 | -0.37% | 4.35% |

## Governance / next work

- freeze robust selector before a new unseen/fresh-forward slice
- rerun candidate generation on that fresh-forward slice; if claiming exhaustive generation, rerun uncapped or record the cap explicitly
- add exact 10/15/20bps cost stress or paper fill telemetry
- check turnover/RPT, spread/slippage, partial/reject/cancel, and reconciliation telemetry
- Do not call this clean/live-ready; it is a post-failure design input.
