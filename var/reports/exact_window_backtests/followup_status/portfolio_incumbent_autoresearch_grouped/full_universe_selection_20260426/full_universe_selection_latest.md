# Full-universe quant selection report — 2026-04-26

Generated: `2026-04-26T06:59:47.631175+00:00`

## Selection policy

- Universe is repo artifact-driven and includes JSON/CSV candidates from saved strategy, portfolio, leverage, pair, source-sleeve, static-blend, wave2/post-2026-04-20, and HYBRID report roots.
- Selection is **validation-primary**: `selection_score = val_scaled_score + 0.18 * train_scaled_score`.
- OOS is **report-only** and is not used for tuning, ranking, health priors, or HYBRID parameter choice.
- Cash efficiency is not directly scored.
- Train/validation MDD up to 25% is eligible; MDD is scored only through bounded headroom.
- Scaled score uses return, Sharpe, Sortino, Calmar, and MDD headroom with bounded `tanh` ratio scaling.

```text
100*(0.30*tanh(return/0.18)+0.30*tanh(Sharpe/4)+0.15*tanh(Sortino/12)+0.15*tanh(Calmar/80)+0.10*MDD_headroom)
MDD_headroom = 1 - min(max(MDD, 0), 0.25)/0.25
```

## Universe discovery

- Scan roots: `var/reports/exact_window_backtests/followup_status, var/reports/portfolio_superiority_dense_pairs, var/reports/portfolio_superiority_wave2, var/reports/portfolio_superiority_overlay_followup, reports`
- JSON files seen: `1771`
- CSV files seen: `141`
- Raw candidate rows extracted: `19215`
- Deduped candidate rows before generated HYBRIDs: `12541`
- Final ranked candidates including generated HYBRIDs: `12544`
- Stream-combinable rows before generated HYBRIDs: `596`

## Final recommendations

- **Best full-universe candidate:** `legacy_metric_no_highvol_baseline_raw_score` (score 95.41, category `true_hybrid`).
- **Best deployable true-HYBRID candidate:** `retuned_full_universe_hybrid_online` (score 85.83, category `true_hybrid`).
- **Conservative fallback/shadow candidate:** `autoresearch_candidate_portfolio_opt`. This uses OOS only as an audit label, not as the selected objective.

## Generated full-universe HYBRID sleeve test

- Sleeve count: `14`
- Retuned grid evaluations: `3888`
- Default sleeve by train score: `grouped_allocator_leverage_tuning`
- Top stream-available non-HYBRID candidates were added as online allocator sleeves; OOS was not used in parameter selection.

### Generated HYBRID candidates

| Rank | Candidate | Category | Score | Train ret/Sharpe/Sortino/Calmar/MDD | Val ret/Sharpe/Sortino/Calmar/MDD | OOS ret/Sharpe/Sortino/Calmar/MDD | Hybrid | Stream | Eligible |
| ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `retuned_full_universe_hybrid_online` | true_hybrid | 85.83 | +63.33% / 1.86 / 3.83 / 5.87 / +10.79% | +23.82% / 3.72 / 8.99 / 66.67 / +4.13% | +0.07% / 0.18 / 0.21 / 0.76 / +0.73% | True | True | True |
| 2 | `full_universe_v35_train_learned_high_vol_hybrid` | true_hybrid_v35 | 73.77 | +16.77% / 1.09 / 1.86 / 1.14 / +14.77% | +16.50% / 3.63 / 7.13 / 42.75 / +3.68% | +0.37% / 1.01 / 1.05 / 4.42 / +0.71% | True | True | True |
| 3 | `full_universe_v36_rolling_dynamic_high_vol_hybrid` | true_hybrid_v36 | 70.43 | +9.52% / 0.65 / 1.03 / 0.56 / +16.94% | +16.46% / 3.58 / 6.68 / 40.22 / +3.90% | +0.32% / 0.89 / 0.90 / 3.77 / +0.71% | True | True | True |

## Full-universe ranking (clean eligible first)

| Rank | Candidate | Category | Score | Train ret/Sharpe/Sortino/Calmar/MDD | Val ret/Sharpe/Sortino/Calmar/MDD | OOS ret/Sharpe/Sortino/Calmar/MDD | Hybrid | Stream | Eligible |
| ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `legacy_metric_no_highvol_baseline_raw_score` | true_hybrid | 95.41 | +27.83% / 1.24 / 1.95 / 2.27 / +12.25% | +23.87% / 4.27 / 24.82 / 134.96 / +2.04% | +0.10% / 0.28 / 0.17 / 1.05 / +0.70% | True | False | True |
| 2 | `legacy_metric_highvol_active_raw_score` | true_hybrid | 92.52 | +31.51% / 1.50 / 2.07 / 3.35 / +9.41% | +20.97% / 3.81 / 22.48 / 114.29 / +1.97% | +0.05% / 0.15 / 0.09 / 0.52 / +0.73% | True | False | True |
| 3 | `final_scaled_v35_train_high_vol` | true_hybrid_v35 | 90.38 | +39.65% / 1.59 / 2.39 / 4.56 / +8.69% | +25.35% / 3.91 / 11.70 / 89.79 / +3.39% | +0.12% / 0.17 / 0.14 / 0.52 / +1.62% | True | False | True |
| 4 | `full_hybrid_val_only_no_cash_penalty` | true_hybrid | 90.11 | +26.54% / 1.30 / 1.77 / 2.82 / +9.41% | +23.28% / 4.03 / 18.66 / 73.88 / +3.59% | +0.29% / 0.55 / 0.31 / 2.44 / +0.86% | True | False | True |
| 5 | `legacy_full_hybrid_wave2_val_only` | true_hybrid | 90.11 | +26.54% / 1.30 / 1.77 / 2.82 / +9.41% | +23.28% / 4.03 / 18.66 / 73.88 / +3.59% | +0.29% / 0.55 / 0.31 / 2.44 / +0.86% | True | False | True |
| 6 | `legacy_metric_highvol_optional_raw_score` | true_hybrid | 89.54 | +28.76% / 1.29 / 1.97 / 2.35 / +12.25% | +22.30% / 4.25 / 16.83 / 70.56 / +3.51% | -1.15% / -3.38 / -1.40 / -6.46 / +1.23% | True | False | True |
| 7 | `source_three_way_regime` | metric_only | 89.07 | +14.84% / 0.63 / 1.15 / 0.73 / +20.31% | +30.68% / 4.17 / 12.14 / 99.81 / +4.24% | +0.56% / 1.31 / 3.62 / 15.54 / +2.18% | False | False | True |
| 8 | `final_scaled_v36_dynamic_high_vol` | true_hybrid_v36 | 88.50 | +24.88% / 1.06 / 1.67 / 2.05 / +12.12% | +24.99% / 3.85 / 12.03 / 88.16 / +3.37% | -0.13% / -0.15 / -0.11 / -0.59 / +1.52% | True | False | True |
| 9 | `static_challenger_static_anchor_val_return` | static_blend | 88.39 | +11.15% / 0.54 / 0.96 / 0.58 / +19.38% | +29.18% / 4.22 / 12.17 / 101.89 / +3.80% | +0.81% / 0.74 / 0.77 / 3.04 / +1.95% | False | False | True |
| 10 | `static_challenger_static_dominant_val_return` | static_blend | 88.03 | +9.59% / 0.49 / 0.87 / 0.50 / +19.11% | +28.54% / 4.23 / 12.31 / 102.23 / +3.64% | +0.92% / 0.84 / 0.91 / 3.63 / +1.85% | False | False | True |
| 11 | `final_scaled_baseline_no_hv` | true_hybrid | 87.75 | +36.18% / 1.55 / 2.24 / 4.16 / +8.69% | +22.39% / 3.84 / 11.32 / 77.61 / +3.21% | -0.98% / -2.47 / -1.09 / -5.03 / +1.36% | True | False | True |
| 12 | `source_soft_three_way_regime` | metric_only | 87.35 | +10.98% / 0.52 / 0.94 / 0.53 / +20.53% | +29.09% / 4.11 / 11.95 / 95.20 / +4.05% | +0.25% / 0.74 / 2.04 / 7.96 / +1.72% | False | False | True |
| 13 | `legacy_robust_val_low_cash_no_oos_health` | true_hybrid | 86.63 | +28.87% / 1.28 / 1.99 / 2.36 / +12.25% | +22.94% / 3.86 / 10.80 / 75.95 / +3.41% | -0.91% / -2.04 / -0.95 / -4.41 / +1.44% | True | False | True |
| 14 | `source_static_blend_76_24` | metric_only | 86.42 | +4.44% / 0.31 / 0.52 / 0.24 / +18.20% | +26.40% / 4.29 / 12.84 / 102.38 / +3.18% | +1.27% / 3.04 / 10.42 / 60.90 / +1.53% | False | False | True |
| 15 | `retuned_full_universe_hybrid_online` | true_hybrid | 85.83 | +63.33% / 1.86 / 3.83 / 5.87 / +10.79% | +23.82% / 3.72 / 8.99 / 66.67 / +4.13% | +0.07% / 0.18 / 0.21 / 0.76 / +0.73% | True | True | True |
| 16 | `full_hybrid_val_low_cash` | true_hybrid | 85.26 | +27.60% / 1.34 / 1.85 / 2.93 / +9.41% | +22.55% / 3.62 / 10.36 / 73.04 / +3.45% | +1.42% / 2.95 / 3.31 / 24.90 / +0.43% | True | False | True |
| 17 | `legacy_full_hybrid_val_low_cash` | true_hybrid | 85.26 | +27.60% / 1.34 / 1.85 / 2.93 / +9.41% | +22.55% / 3.62 / 10.36 / 73.04 / +3.45% | +1.42% / 2.95 / 3.31 / 24.90 / +0.43% | True | False | True |
| 18 | `legacy_robust_val_aggressive_cash_no_oos_health` | true_hybrid | 85.24 | +27.53% / 1.34 / 1.85 / 2.93 / +9.41% | +22.68% / 3.82 / 9.95 / 68.84 / +3.69% | -0.63% / -1.24 / -0.68 / -3.11 / +1.41% | True | False | True |
| 19 | `composite_trend_30m_0.45_0.60_0.20_0.80` | metric_only | 83.69 | +1.73% / 15.97 / 23.32 / 107.87 / +0.60% | +0.93% / 18.58 / 28.28 / 285.96 / +0.43% | +1.11% / 18.59 / 20.85 / 571.93 / +0.29% | False | False | True |
| 20 | `composite_trend_30m_0.60_0.75_0.20_0.80` | metric_only | 83.66 | +1.48% / 20.35 / 26.31 / 230.83 / +0.23% | +0.61% / 21.45 / 29.01 / 428.09 / +0.16% | +0.87% / 22.36 / 34.63 / 410.60 / +0.28% | False | False | True |
| 21 | `composite_trend_30m_0.60_0.75_0.25_0.80` | metric_only | 83.66 | +1.48% / 20.35 / 26.31 / 230.83 / +0.23% | +0.61% / 21.45 / 29.01 / 428.09 / +0.16% | +0.85% / 21.68 / 33.97 / 393.99 / +0.28% | False | False | True |
| 22 | `composite_trend_30m_0.45_0.75_0.20_0.80` | metric_only | 83.63 | +1.62% / 16.62 / 23.11 / 193.61 / +0.31% | +0.78% / 19.85 / 27.09 / 235.13 / +0.42% | +1.00% / 18.01 / 19.36 / 490.23 / +0.28% | False | False | True |
| 23 | `composite_trend_30m_0.45_0.60_0.25_0.80` | metric_only | 83.59 | +1.73% / 16.13 / 23.35 / 108.38 / +0.60% | +0.90% / 17.97 / 27.33 / 272.51 / +0.43% | +1.07% / 17.91 / 20.13 / 540.44 / +0.29% | False | False | True |
| 24 | `composite_trend_30m_0.45_0.75_0.25_0.80` | metric_only | 83.58 | +1.59% / 16.54 / 22.83 / 212.05 / +0.28% | +0.77% / 19.60 / 26.63 / 229.57 / +0.42% | +0.97% / 17.44 / 18.78 / 467.45 / +0.28% | False | False | True |
| 25 | `composite_trend_30m_0.60_0.60_0.25_0.80` | metric_only | 83.56 | +1.61% / 18.04 / 24.86 / 116.17 / +0.51% | +0.74% / 17.88 / 28.48 / 463.83 / +0.19% | +0.95% / 20.99 / 33.41 / 503.00 / +0.26% | False | False | True |
| 26 | `composite_trend_30m_0.60_0.60_0.20_0.80` | metric_only | 83.55 | +1.58% / 17.69 / 24.42 / 107.68 / +0.54% | +0.75% / 18.36 / 29.50 / 478.24 / +0.19% | +0.98% / 21.77 / 34.27 / 529.41 / +0.26% | False | False | True |
| 27 | `composite_trend_30m_0.75_0.75_0.20_0.80` | metric_only | 83.48 | +1.10% / 16.48 / 21.15 / 199.91 / +0.19% | +0.53% / 23.73 / 35.32 / 470.05 / +0.13% | +0.79% / 28.96 / 33.90 / 837.83 / +0.12% | False | False | True |
| 28 | `composite_trend_30m_0.75_0.75_0.25_0.80` | metric_only | 83.48 | +1.10% / 16.48 / 21.15 / 199.91 / +0.19% | +0.53% / 23.73 / 35.32 / 470.05 / +0.13% | +0.76% / 27.91 / 33.12 / 801.46 / +0.12% | False | False | True |
| 29 | `composite_trend_30m_0.75_0.60_0.25_0.80` | metric_only | 82.89 | +1.23% / 14.67 / 19.59 / 76.96 / +0.56% | +0.66% / 17.70 / 31.50 / 427.60 / +0.18% | +0.86% / 24.28 / 31.74 / 702.14 / +0.16% | False | False | True |
| 30 | `composite_trend_30m_0.75_0.60_0.20_0.80` | metric_only | 82.85 | +1.20% / 14.31 / 19.14 / 71.25 / +0.59% | +0.68% / 18.26 / 33.28 / 442.02 / +0.18% | +0.90% / 25.31 / 32.73 / 741.37 / +0.16% | False | False | True |
| 31 | `legacy_static_anchor_true_hybrid_val` | true_hybrid | 82.72 | +26.80% / 1.47 / 1.99 / 3.27 / +8.19% | +21.20% / 3.39 / 11.97 / 55.02 / +4.16% | -0.39% / -1.26 / -0.82 / -3.83 / +0.72% | True | False | True |
| 32 | `legacy_static_anchor_true_hybrid_low_cash` | true_hybrid | 82.69 | +26.89% / 1.47 / 2.00 / 3.29 / +8.19% | +21.26% / 3.39 / 11.78 / 55.20 / +4.16% | -0.51% / -1.73 / -1.00 / -4.83 / +0.74% | True | False | True |
| 33 | `composite_trend_30m_0.60_0.60_0.25_0.95` | metric_only | 81.51 | +0.81% / 10.75 / 12.80 / 70.21 / +0.38% | +0.50% / 14.24 / 23.27 / 372.38 / +0.15% | +0.17% / 5.76 / 6.63 / 58.30 / +0.27% | False | False | True |
| 34 | `composite_trend_30m_0.60_0.60_0.20_0.95` | metric_only | 81.48 | +0.78% / 10.36 / 12.40 / 62.92 / +0.41% | +0.51% / 14.80 / 24.61 / 387.96 / +0.15% | +0.17% / 5.76 / 6.63 / 58.30 / +0.27% | False | False | True |
| 35 | `full_hybrid_val_very_low_cash` | true_hybrid | 81.38 | +28.23% / 1.28 / 1.95 / 2.30 / +12.25% | +21.18% / 3.35 / 10.84 / 54.58 / +4.18% | -0.12% / -0.16 / -0.13 / -0.69 / +1.25% | True | False | True |
| 36 | `legacy_full_hybrid_val_very_low_cash` | true_hybrid | 81.38 | +28.23% / 1.28 / 1.95 / 2.30 / +12.25% | +21.18% / 3.35 / 10.84 / 54.58 / +4.18% | -0.12% / -0.16 / -0.13 / -0.69 / +1.25% | True | False | True |
| 37 | `composite_trend_30m_0.45_0.60_0.20_0.95` | metric_only | 80.83 | +0.78% / 9.01 / 11.47 / 44.65 / +0.57% | +0.61% / 14.60 / 23.88 / 221.77 / +0.32% | -0.07% / -1.56 / -1.34 / -13.28 / +0.44% | False | False | True |
| 38 | `composite_trend_30m_0.45_0.60_0.25_0.95` | metric_only | 80.83 | +0.80% / 9.31 / 11.73 / 47.99 / +0.55% | +0.60% / 14.15 / 23.12 / 214.00 / +0.32% | -0.08% / -1.70 / -1.46 / -14.31 / +0.45% | False | False | True |
| 39 | `composite_trend_30m_0.75_0.75_0.20_0.95` | metric_only | 80.77 | +0.48% / 9.24 / 9.60 / 79.62 / +0.19% | +0.28% / 15.83 / 22.77 / 265.02 / +0.10% | +0.19% / 9.55 / 9.77 / 140.15 / +0.13% | False | False | True |
| 40 | `composite_trend_30m_0.75_0.75_0.25_0.95` | metric_only | 80.77 | +0.48% / 9.24 / 9.60 / 79.62 / +0.19% | +0.28% / 15.83 / 22.77 / 265.02 / +0.10% | +0.19% / 9.55 / 9.77 / 140.15 / +0.13% | False | False | True |

## True-HYBRID ranking

| Rank | Candidate | Category | Score | Train ret/Sharpe/Sortino/Calmar/MDD | Val ret/Sharpe/Sortino/Calmar/MDD | OOS ret/Sharpe/Sortino/Calmar/MDD | Hybrid | Stream | Eligible |
| ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `legacy_metric_no_highvol_baseline_raw_score` | true_hybrid | 95.41 | +27.83% / 1.24 / 1.95 / 2.27 / +12.25% | +23.87% / 4.27 / 24.82 / 134.96 / +2.04% | +0.10% / 0.28 / 0.17 / 1.05 / +0.70% | True | False | True |
| 2 | `legacy_metric_highvol_active_raw_score` | true_hybrid | 92.52 | +31.51% / 1.50 / 2.07 / 3.35 / +9.41% | +20.97% / 3.81 / 22.48 / 114.29 / +1.97% | +0.05% / 0.15 / 0.09 / 0.52 / +0.73% | True | False | True |
| 3 | `final_scaled_v35_train_high_vol` | true_hybrid_v35 | 90.38 | +39.65% / 1.59 / 2.39 / 4.56 / +8.69% | +25.35% / 3.91 / 11.70 / 89.79 / +3.39% | +0.12% / 0.17 / 0.14 / 0.52 / +1.62% | True | False | True |
| 4 | `full_hybrid_val_only_no_cash_penalty` | true_hybrid | 90.11 | +26.54% / 1.30 / 1.77 / 2.82 / +9.41% | +23.28% / 4.03 / 18.66 / 73.88 / +3.59% | +0.29% / 0.55 / 0.31 / 2.44 / +0.86% | True | False | True |
| 5 | `legacy_full_hybrid_wave2_val_only` | true_hybrid | 90.11 | +26.54% / 1.30 / 1.77 / 2.82 / +9.41% | +23.28% / 4.03 / 18.66 / 73.88 / +3.59% | +0.29% / 0.55 / 0.31 / 2.44 / +0.86% | True | False | True |
| 6 | `legacy_metric_highvol_optional_raw_score` | true_hybrid | 89.54 | +28.76% / 1.29 / 1.97 / 2.35 / +12.25% | +22.30% / 4.25 / 16.83 / 70.56 / +3.51% | -1.15% / -3.38 / -1.40 / -6.46 / +1.23% | True | False | True |
| 7 | `final_scaled_v36_dynamic_high_vol` | true_hybrid_v36 | 88.50 | +24.88% / 1.06 / 1.67 / 2.05 / +12.12% | +24.99% / 3.85 / 12.03 / 88.16 / +3.37% | -0.13% / -0.15 / -0.11 / -0.59 / +1.52% | True | False | True |
| 8 | `final_scaled_baseline_no_hv` | true_hybrid | 87.75 | +36.18% / 1.55 / 2.24 / 4.16 / +8.69% | +22.39% / 3.84 / 11.32 / 77.61 / +3.21% | -0.98% / -2.47 / -1.09 / -5.03 / +1.36% | True | False | True |
| 9 | `legacy_robust_val_low_cash_no_oos_health` | true_hybrid | 86.63 | +28.87% / 1.28 / 1.99 / 2.36 / +12.25% | +22.94% / 3.86 / 10.80 / 75.95 / +3.41% | -0.91% / -2.04 / -0.95 / -4.41 / +1.44% | True | False | True |
| 10 | `retuned_full_universe_hybrid_online` | true_hybrid | 85.83 | +63.33% / 1.86 / 3.83 / 5.87 / +10.79% | +23.82% / 3.72 / 8.99 / 66.67 / +4.13% | +0.07% / 0.18 / 0.21 / 0.76 / +0.73% | True | True | True |
| 11 | `full_hybrid_val_low_cash` | true_hybrid | 85.26 | +27.60% / 1.34 / 1.85 / 2.93 / +9.41% | +22.55% / 3.62 / 10.36 / 73.04 / +3.45% | +1.42% / 2.95 / 3.31 / 24.90 / +0.43% | True | False | True |
| 12 | `legacy_full_hybrid_val_low_cash` | true_hybrid | 85.26 | +27.60% / 1.34 / 1.85 / 2.93 / +9.41% | +22.55% / 3.62 / 10.36 / 73.04 / +3.45% | +1.42% / 2.95 / 3.31 / 24.90 / +0.43% | True | False | True |
| 13 | `legacy_robust_val_aggressive_cash_no_oos_health` | true_hybrid | 85.24 | +27.53% / 1.34 / 1.85 / 2.93 / +9.41% | +22.68% / 3.82 / 9.95 / 68.84 / +3.69% | -0.63% / -1.24 / -0.68 / -3.11 / +1.41% | True | False | True |
| 14 | `legacy_static_anchor_true_hybrid_val` | true_hybrid | 82.72 | +26.80% / 1.47 / 1.99 / 3.27 / +8.19% | +21.20% / 3.39 / 11.97 / 55.02 / +4.16% | -0.39% / -1.26 / -0.82 / -3.83 / +0.72% | True | False | True |
| 15 | `legacy_static_anchor_true_hybrid_low_cash` | true_hybrid | 82.69 | +26.89% / 1.47 / 2.00 / 3.29 / +8.19% | +21.26% / 3.39 / 11.78 / 55.20 / +4.16% | -0.51% / -1.73 / -1.00 / -4.83 / +0.74% | True | False | True |
| 16 | `full_hybrid_val_very_low_cash` | true_hybrid | 81.38 | +28.23% / 1.28 / 1.95 / 2.30 / +12.25% | +21.18% / 3.35 / 10.84 / 54.58 / +4.18% | -0.12% / -0.16 / -0.13 / -0.69 / +1.25% | True | False | True |
| 17 | `legacy_full_hybrid_val_very_low_cash` | true_hybrid | 81.38 | +28.23% / 1.28 / 1.95 / 2.30 / +12.25% | +21.18% / 3.35 / 10.84 / 54.58 / +4.18% | -0.12% / -0.16 / -0.13 / -0.69 / +1.25% | True | False | True |
| 18 | `hybrid_online_optuna_20260415T120755Z:top_trials:6:scenarios:historical_saved_baseline` | true_hybrid | 75.49 | +6.89% / 0.96 / 1.54 / 1.45 / +4.74% | +7.78% / 3.91 / 20.99 / 84.81 / +0.70% | +0.89% / 2.73 / 5.74 / 27.22 / +0.37% | True | False | True |
| 19 | `hybrid_online_optuna_20260416T120914Z:top_trials:6:scenarios:historical_saved_baseline` | true_hybrid | 75.49 | +6.89% / 0.96 / 1.54 / 1.45 / +4.74% | +7.78% / 3.91 / 20.99 / 84.81 / +0.70% | +0.89% / 2.73 / 5.74 / 27.22 / +0.37% | True | False | True |
| 20 | `hybrid_online_optuna_latest:top_trials:6:scenarios:historical_saved_baseline` | true_hybrid | 75.49 | +6.89% / 0.96 / 1.54 / 1.45 / +4.74% | +7.78% / 3.91 / 20.99 / 84.81 / +0.70% | +0.89% / 2.73 / 5.74 / 27.22 / +0.37% | True | False | True |
| 21 | `hybrid_online_optuna_20260416T120927Z:top_trials:6:scenarios:historical_saved_baseline` | true_hybrid | 75.49 | +6.89% / 0.96 / 1.54 / 1.45 / +4.74% | +7.78% / 3.91 / 20.99 / 84.81 / +0.70% | +0.89% / 2.73 / 5.74 / 27.22 / +0.37% | True | False | True |
| 22 | `hybrid_online_tuning_20260415T113057Z:leaderboard:3:scenarios:historical_saved_baseline` | true_hybrid | 74.79 | +6.73% / 0.94 / 1.51 / 1.47 / +4.58% | +7.79% / 3.91 / 20.22 / 78.19 / +0.76% | +1.38% / 3.96 / 8.01 / 40.75 / +0.39% | True | False | True |
| 23 | `hybrid_online_tuning_20260415T120743Z:leaderboard:3:scenarios:historical_saved_baseline` | true_hybrid | 74.79 | +6.73% / 0.94 / 1.51 / 1.47 / +4.58% | +7.79% / 3.91 / 20.22 / 78.19 / +0.76% | +1.38% / 3.96 / 8.01 / 40.75 / +0.39% | True | False | True |
| 24 | `hybrid_online_tuning_20260415T122932Z:leaderboard:2:scenarios:historical_saved_baseline` | true_hybrid | 74.79 | +6.73% / 0.94 / 1.51 / 1.47 / +4.58% | +7.79% / 3.91 / 20.22 / 78.19 / +0.76% | +1.38% / 3.96 / 8.01 / 40.75 / +0.39% | True | False | True |
| 25 | `hybrid_online_tuning_20260416T120904Z:leaderboard:3:scenarios:historical_saved_baseline` | true_hybrid | 74.79 | +6.73% / 0.94 / 1.51 / 1.47 / +4.58% | +7.79% / 3.91 / 20.22 / 78.19 / +0.76% | +1.38% / 3.96 / 8.01 / 40.75 / +0.39% | True | False | True |
| 26 | `hybrid_online_tuning_latest:leaderboard:3:scenarios:historical_saved_baseline` | true_hybrid | 74.79 | +6.73% / 0.94 / 1.51 / 1.47 / +4.58% | +7.79% / 3.91 / 20.22 / 78.19 / +0.76% | +1.38% / 3.96 / 8.01 / 40.75 / +0.39% | True | False | True |
| 27 | `hybrid_online_optuna_20260415T113134Z:top_trials:1:scenarios:historical_saved_baseline` | true_hybrid | 74.75 | +6.71% / 0.94 / 1.51 / 1.47 / +4.57% | +7.65% / 3.91 / 20.83 / 78.93 / +0.73% | +1.43% / 4.06 / 8.62 / 43.80 / +0.38% | True | False | True |
| 28 | `hybrid_online_optuna_20260415T120755Z:top_trials:7:scenarios:historical_saved_baseline` | true_hybrid | 74.69 | +6.66% / 0.93 / 1.49 / 1.41 / +4.71% | +7.53% / 3.98 / 21.68 / 76.68 / +0.74% | +0.72% / 2.39 / 4.34 / 20.37 / +0.39% | True | False | True |
| 29 | `hybrid_online_optuna_20260416T120914Z:top_trials:7:scenarios:historical_saved_baseline` | true_hybrid | 74.69 | +6.66% / 0.93 / 1.49 / 1.41 / +4.71% | +7.53% / 3.98 / 21.68 / 76.68 / +0.74% | +0.72% / 2.39 / 4.34 / 20.37 / +0.39% | True | False | True |
| 30 | `hybrid_online_optuna_latest:top_trials:7:scenarios:historical_saved_baseline` | true_hybrid | 74.69 | +6.66% / 0.93 / 1.49 / 1.41 / +4.71% | +7.53% / 3.98 / 21.68 / 76.68 / +0.74% | +0.72% / 2.39 / 4.34 / 20.37 / +0.39% | True | False | True |

## Non-HYBRID/static/source ranking

| Rank | Candidate | Category | Score | Train ret/Sharpe/Sortino/Calmar/MDD | Val ret/Sharpe/Sortino/Calmar/MDD | OOS ret/Sharpe/Sortino/Calmar/MDD | Hybrid | Stream | Eligible |
| ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `source_three_way_regime` | metric_only | 89.07 | +14.84% / 0.63 / 1.15 / 0.73 / +20.31% | +30.68% / 4.17 / 12.14 / 99.81 / +4.24% | +0.56% / 1.31 / 3.62 / 15.54 / +2.18% | False | False | True |
| 2 | `static_challenger_static_anchor_val_return` | static_blend | 88.39 | +11.15% / 0.54 / 0.96 / 0.58 / +19.38% | +29.18% / 4.22 / 12.17 / 101.89 / +3.80% | +0.81% / 0.74 / 0.77 / 3.04 / +1.95% | False | False | True |
| 3 | `static_challenger_static_dominant_val_return` | static_blend | 88.03 | +9.59% / 0.49 / 0.87 / 0.50 / +19.11% | +28.54% / 4.23 / 12.31 / 102.23 / +3.64% | +0.92% / 0.84 / 0.91 / 3.63 / +1.85% | False | False | True |
| 4 | `source_soft_three_way_regime` | metric_only | 87.35 | +10.98% / 0.52 / 0.94 / 0.53 / +20.53% | +29.09% / 4.11 / 11.95 / 95.20 / +4.05% | +0.25% / 0.74 / 2.04 / 7.96 / +1.72% | False | False | True |
| 5 | `source_static_blend_76_24` | metric_only | 86.42 | +4.44% / 0.31 / 0.52 / 0.24 / +18.20% | +26.40% / 4.29 / 12.84 / 102.38 / +3.18% | +1.27% / 3.04 / 10.42 / 60.90 / +1.53% | False | False | True |
| 6 | `composite_trend_30m_0.45_0.60_0.20_0.80` | metric_only | 83.69 | +1.73% / 15.97 / 23.32 / 107.87 / +0.60% | +0.93% / 18.58 / 28.28 / 285.96 / +0.43% | +1.11% / 18.59 / 20.85 / 571.93 / +0.29% | False | False | True |
| 7 | `composite_trend_30m_0.60_0.75_0.20_0.80` | metric_only | 83.66 | +1.48% / 20.35 / 26.31 / 230.83 / +0.23% | +0.61% / 21.45 / 29.01 / 428.09 / +0.16% | +0.87% / 22.36 / 34.63 / 410.60 / +0.28% | False | False | True |
| 8 | `composite_trend_30m_0.60_0.75_0.25_0.80` | metric_only | 83.66 | +1.48% / 20.35 / 26.31 / 230.83 / +0.23% | +0.61% / 21.45 / 29.01 / 428.09 / +0.16% | +0.85% / 21.68 / 33.97 / 393.99 / +0.28% | False | False | True |
| 9 | `composite_trend_30m_0.45_0.75_0.20_0.80` | metric_only | 83.63 | +1.62% / 16.62 / 23.11 / 193.61 / +0.31% | +0.78% / 19.85 / 27.09 / 235.13 / +0.42% | +1.00% / 18.01 / 19.36 / 490.23 / +0.28% | False | False | True |
| 10 | `composite_trend_30m_0.45_0.60_0.25_0.80` | metric_only | 83.59 | +1.73% / 16.13 / 23.35 / 108.38 / +0.60% | +0.90% / 17.97 / 27.33 / 272.51 / +0.43% | +1.07% / 17.91 / 20.13 / 540.44 / +0.29% | False | False | True |
| 11 | `composite_trend_30m_0.45_0.75_0.25_0.80` | metric_only | 83.58 | +1.59% / 16.54 / 22.83 / 212.05 / +0.28% | +0.77% / 19.60 / 26.63 / 229.57 / +0.42% | +0.97% / 17.44 / 18.78 / 467.45 / +0.28% | False | False | True |
| 12 | `composite_trend_30m_0.60_0.60_0.25_0.80` | metric_only | 83.56 | +1.61% / 18.04 / 24.86 / 116.17 / +0.51% | +0.74% / 17.88 / 28.48 / 463.83 / +0.19% | +0.95% / 20.99 / 33.41 / 503.00 / +0.26% | False | False | True |
| 13 | `composite_trend_30m_0.60_0.60_0.20_0.80` | metric_only | 83.55 | +1.58% / 17.69 / 24.42 / 107.68 / +0.54% | +0.75% / 18.36 / 29.50 / 478.24 / +0.19% | +0.98% / 21.77 / 34.27 / 529.41 / +0.26% | False | False | True |
| 14 | `composite_trend_30m_0.75_0.75_0.20_0.80` | metric_only | 83.48 | +1.10% / 16.48 / 21.15 / 199.91 / +0.19% | +0.53% / 23.73 / 35.32 / 470.05 / +0.13% | +0.79% / 28.96 / 33.90 / 837.83 / +0.12% | False | False | True |
| 15 | `composite_trend_30m_0.75_0.75_0.25_0.80` | metric_only | 83.48 | +1.10% / 16.48 / 21.15 / 199.91 / +0.19% | +0.53% / 23.73 / 35.32 / 470.05 / +0.13% | +0.76% / 27.91 / 33.12 / 801.46 / +0.12% | False | False | True |
| 16 | `composite_trend_30m_0.75_0.60_0.25_0.80` | metric_only | 82.89 | +1.23% / 14.67 / 19.59 / 76.96 / +0.56% | +0.66% / 17.70 / 31.50 / 427.60 / +0.18% | +0.86% / 24.28 / 31.74 / 702.14 / +0.16% | False | False | True |
| 17 | `composite_trend_30m_0.75_0.60_0.20_0.80` | metric_only | 82.85 | +1.20% / 14.31 / 19.14 / 71.25 / +0.59% | +0.68% / 18.26 / 33.28 / 442.02 / +0.18% | +0.90% / 25.31 / 32.73 / 741.37 / +0.16% | False | False | True |
| 18 | `composite_trend_30m_0.60_0.60_0.25_0.95` | metric_only | 81.51 | +0.81% / 10.75 / 12.80 / 70.21 / +0.38% | +0.50% / 14.24 / 23.27 / 372.38 / +0.15% | +0.17% / 5.76 / 6.63 / 58.30 / +0.27% | False | False | True |
| 19 | `composite_trend_30m_0.60_0.60_0.20_0.95` | metric_only | 81.48 | +0.78% / 10.36 / 12.40 / 62.92 / +0.41% | +0.51% / 14.80 / 24.61 / 387.96 / +0.15% | +0.17% / 5.76 / 6.63 / 58.30 / +0.27% | False | False | True |
| 20 | `composite_trend_30m_0.45_0.60_0.20_0.95` | metric_only | 80.83 | +0.78% / 9.01 / 11.47 / 44.65 / +0.57% | +0.61% / 14.60 / 23.88 / 221.77 / +0.32% | -0.07% / -1.56 / -1.34 / -13.28 / +0.44% | False | False | True |
| 21 | `composite_trend_30m_0.45_0.60_0.25_0.95` | metric_only | 80.83 | +0.80% / 9.31 / 11.73 / 47.99 / +0.55% | +0.60% / 14.15 / 23.12 / 214.00 / +0.32% | -0.08% / -1.70 / -1.46 / -14.31 / +0.45% | False | False | True |
| 22 | `composite_trend_30m_0.75_0.75_0.20_0.95` | metric_only | 80.77 | +0.48% / 9.24 / 9.60 / 79.62 / +0.19% | +0.28% / 15.83 / 22.77 / 265.02 / +0.10% | +0.19% / 9.55 / 9.77 / 140.15 / +0.13% | False | False | True |
| 23 | `composite_trend_30m_0.75_0.75_0.25_0.95` | metric_only | 80.77 | +0.48% / 9.24 / 9.60 / 79.62 / +0.19% | +0.28% / 15.83 / 22.77 / 265.02 / +0.10% | +0.19% / 9.55 / 9.77 / 140.15 / +0.13% | False | False | True |
| 24 | `composite_trend_30m_0.60_0.75_0.20_0.95` | metric_only | 80.60 | +0.72% / 13.04 / 14.90 / 141.61 / +0.17% | +0.29% / 13.11 / 16.07 / 261.90 / +0.11% | +0.12% / 5.20 / 5.34 / 43.09 / +0.26% | False | False | True |
| 25 | `composite_trend_30m_0.60_0.75_0.25_0.95` | metric_only | 80.60 | +0.72% / 13.04 / 14.90 / 141.61 / +0.17% | +0.29% / 13.11 / 16.07 / 261.90 / +0.11% | +0.12% / 5.20 / 5.34 / 43.09 / +0.26% | False | False | True |
| 26 | `composite_trend_30m_0.75_0.60_0.25_0.95` | metric_only | 80.40 | +0.57% / 7.83 / 8.69 / 42.28 / +0.43% | +0.49% / 14.86 / 25.20 / 443.45 / +0.12% | +0.24% / 8.98 / 10.32 / 148.79 / +0.16% | False | False | True |
| 27 | `composite_trend_30m_0.75_0.60_0.20_0.95` | metric_only | 80.33 | +0.54% / 7.44 / 8.31 / 37.60 / +0.45% | +0.51% / 15.49 / 27.31 / 462.27 / +0.12% | +0.24% / 8.98 / 10.32 / 148.79 / +0.16% | False | False | True |
| 28 | `composite_trend_1h_0.45_0.75_0.20_0.80` | metric_only | 80.13 | +0.84% / 7.08 / 8.39 / 37.02 / +0.35% | +0.81% / 15.27 / 23.10 / 173.55 / +0.24% | -0.35% / -4.38 / -4.69 / -26.80 / +0.53% | False | False | True |
| 29 | `composite_trend_1h_0.45_0.75_0.25_0.80` | metric_only | 80.10 | +0.77% / 6.49 / 7.67 / 33.63 / +0.35% | +0.82% / 16.24 / 23.65 / 203.30 / +0.21% | -0.25% / -3.11 / -3.32 / -19.90 / +0.52% | False | False | True |
| 30 | `composite_trend_30m_0.45_0.75_0.20_0.95` | metric_only | 79.16 | +0.72% / 9.84 / 12.49 / 85.32 / +0.27% | +0.39% / 12.36 / 16.69 / 133.20 / +0.30% | -0.12% / -2.90 / -2.27 / -25.84 / +0.38% | False | False | True |
| 31 | `composite_trend_30m_0.45_0.75_0.25_0.95` | metric_only | 79.13 | +0.71% / 9.86 / 12.42 / 83.79 / +0.27% | +0.39% / 12.36 / 16.69 / 133.20 / +0.30% | -0.12% / -3.07 / -2.38 / -27.19 / +0.38% | False | False | True |
| 32 | `grouped_allocator_leverage_tuning` | leverage_sweep | 79.11 | +51.83% / 1.29 / 2.32 / 2.43 / +21.29% | +32.49% / 2.96 / 8.24 / 52.16 / +9.01% | +0.42% / 1.19 / 1.33 / 13.70 / +1.77% | False | True | True |
| 33 | `soft_three_way_3x_leverage_sweep` | metric_only | 79.07 | +11.82% / 0.42 / 0.79 / 0.61 / +19.45% | +10.79% / 5.23 / 9.79 / 117.52 / +1.99% | +18.78% / 3.74 / 13.01 / 121.57 / +4.13% | False | False | True |
| 34 | `soft_three_way_market_regime_allocator_latest_leverage_sweep` | leverage_sweep | 79.07 | +11.82% / 0.42 / 0.79 / 0.61 / +19.45% | +10.79% / 5.23 / 9.79 / 117.52 / +1.99% | +18.78% / 3.74 / 13.01 / 121.57 / +4.13% | False | True | True |
| 35 | `soft_three_way_market_regime_allocator_latest_leverage_candidate_latest:portfolio_metrics` | leverage_sweep | 79.07 | +11.82% / 0.42 / 0.79 / 0.61 / +19.45% | +10.79% / 5.23 / 9.79 / 117.52 / +1.99% | +18.78% / 3.74 / 13.01 / 121.57 / +4.13% | False | False | True |
| 36 | `soft_three_way_market_regime_allocator_latest_leverage_sweep_latest:best_result` | leverage_sweep | 79.07 | +11.82% / 0.42 / 0.79 / 0.61 / +19.45% | +10.79% / 5.23 / 9.79 / 117.52 / +1.99% | +18.78% / 3.74 / 13.01 / 121.57 / +4.13% | False | False | True |
| 37 | `soft_three_way_market_regime_allocator_latest_leverage_sweep_latest:best_result:portfolio_metrics` | leverage_sweep | 79.07 | +11.82% / 0.42 / 0.79 / 0.61 / +19.45% | +10.79% / 5.23 / 9.79 / 117.52 / +1.99% | +18.78% / 3.74 / 13.01 / 121.57 / +4.13% | False | False | True |
| 38 | `soft_three_way_market_regime_allocator_latest_leverage_sweep_latest:results:2` | leverage_sweep | 79.07 | +11.82% / 0.42 / 0.79 / 0.61 / +19.45% | +10.79% / 5.23 / 9.79 / 117.52 / +1.99% | +18.78% / 3.74 / 13.01 / 121.57 / +4.13% | False | False | True |
| 39 | `soft_three_way_market_regime_allocator_latest_leverage_sweep_latest:results:2:portfolio_metrics` | leverage_sweep | 79.07 | +11.82% / 0.42 / 0.79 / 0.61 / +19.45% | +10.79% / 5.23 / 9.79 / 117.52 / +1.99% | +18.78% / 3.74 / 13.01 / 121.57 / +4.13% | False | False | True |
| 40 | `composite_trend_30m_0.45_0.45_0.20_0.80` | metric_only | 78.87 | +1.38% / 10.60 / 16.01 / 45.14 / +1.09% | +0.76% / 12.18 / 14.10 / 154.63 / +0.61% | +1.23% / 20.01 / 25.64 / 452.33 / +0.43% | False | False | True |

## Explicit caveats

- This is still a repeated validation-mining exercise; strong validation scores can meta-overfit and require fresh forward/paper evidence after 2026-04-20.
- OOS figures are shown to expose fragility only; they did not enter scoring, tuning, health priors, or candidate selection.
- Some saved artifacts are metric-only and non-combinable; they can win the external ranking but cannot be inserted into the true HYBRID allocator without daily streams.
- Stream union uses zero return on missing candidate days, matching the saved-sleeve panel convention but still a caveat for sparse pair strategies.
- The generated v3.5/v3.6 variants are portfolio-governor analogues of the referenced ensemble strategies, not prediction-level ensemble model ports.
