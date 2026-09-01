# MDD30 High-Vol Sleeve/Gate Final Metrics

- Source: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_mdd30_high_vol_20260602/mdd30_high_vol_walkforward_latest.json`
- OOS period: `2025-09-01T00:00:00` → `2026-06-01T06:30:00`
- Full months: 2025-09 through 2026-05; 2026-06 is partial only.
- Refit: monthly day-1, expanding train, previous 2 calendar months validation, next 1 calendar month locked OOS.
- Timeframes: 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d; slippage/cost: 10.0 bps.

## Decision

- Highest OOS comp under MDD<=30 is `mdd30_risk_scaled:dyn085_x1_50` / `mdd30_risk_scaled:dyn085_val_mdd30_cap1_50`: 84.64% comp, 27.17% max OOS MDD, Sharpe 2.07, Sortino 15.16, PF 8.30.
- This is **research / fresh-forward shadow only**, not clean real-money promotion, because the MDD30 family was introduced after prior OOS review. Within this run it does not use current-fold locked OOS for selection.
- Highest clean deployable candidate remains `dynamic_conviction_switch:t0.85_risk_capped_fallback`: 53.38% comp, 18.80% max OOS MDD, Sharpe 2.07, Sortino 15.31, PF 8.40.

## Top candidates: MDD<=30, fold_count>=9

| Rank | Candidate | Family | Comp | Ann. approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF | Hit | Clean | Latest OOS |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mdd30_risk_scaled:dyn085_val_mdd30_cap1_50` | mdd30_risk_scaled | 84.64% | 108.73% | 27.17% | 6.90% | 2.07 | 15.16 | 8.30 | 5/10 | False | -1.07% |
| 2 | `mdd30_risk_scaled:dyn085_x1_50` | mdd30_risk_scaled | 84.64% | 108.73% | 27.17% | 6.90% | 2.07 | 15.16 | 8.30 | 5/10 | False | -1.07% |
| 3 | `mdd30_risk_scaled:dyn085_x1_25` | mdd30_risk_scaled | 68.68% | 87.27% | 23.07% | 5.76% | 2.07 | 15.24 | 8.35 | 5/10 | False | -0.89% |
| 4 | `mdd30_risk_scaled:dyn100_x1_50` | mdd30_risk_scaled | 61.04% | 77.15% | 27.17% | 6.90% | 1.69 | 10.72 | 7.45 | 5/10 | False | 0.00% |
| 5 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | dynamic_conviction_switch | 53.38% | 67.08% | 18.80% | 4.62% | 2.07 | 15.31 | 8.40 | 5/10 | True | -0.71% |
| 6 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | dynamic_conviction_switch | 53.38% | 67.08% | 18.80% | 4.62% | 2.07 | 15.31 | 8.40 | 5/10 | True | -0.71% |
| 7 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | dynamic_conviction_switch | 53.38% | 67.08% | 18.80% | 4.62% | 2.07 | 15.31 | 8.40 | 5/10 | True | -0.71% |
| 8 | `risk_enhanced_blend:dyn085_70_aware_v36tv_30` | risk_enhanced_blend | 47.60% | 59.55% | 14.67% | 4.54% | 1.98 | 16.23 | 7.23 | 5/10 | False | -0.73% |
| 9 | `risk_enhanced_blend:dyn085_60_aware_v36tv_40` | risk_enhanced_blend | 45.60% | 56.96% | 14.14% | 4.53% | 1.93 | 16.93 | 6.63 | 5/10 | False | -0.74% |
| 10 | `dynamic_conviction_switch:t0.85_strict_fallback` | dynamic_conviction_switch | 43.73% | 54.55% | 18.80% | 2.11% | 1.62 | 4.24 | 4.27 | 4/10 | True | -0.71% |
| 11 | `dynamic_conviction_switch:t0.90_strict_fallback` | dynamic_conviction_switch | 43.73% | 54.55% | 18.80% | 2.11% | 1.62 | 4.24 | 4.27 | 4/10 | True | -0.71% |
| 12 | `dynamic_conviction_switch:t0.95_strict_fallback` | dynamic_conviction_switch | 43.73% | 54.55% | 18.80% | 2.11% | 1.62 | 4.24 | 4.27 | 4/10 | True | -0.71% |

## Clean candidates only: MDD<=30, fold_count>=9

| Rank | Candidate | Comp | Max OOS MDD | Sharpe | Sortino | PF | Hit | Hard stop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | 53.38% | 18.80% | 2.07 | 15.31 | 8.40 | 5/10 | False |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | 53.38% | 18.80% | 2.07 | 15.31 | 8.40 | 5/10 | False |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | 53.38% | 18.80% | 2.07 | 15.31 | 8.40 | 5/10 | False |
| 4 | `dynamic_conviction_switch:t0.85_strict_fallback` | 43.73% | 18.80% | 1.62 | 4.24 | 4.27 | 4/10 | False |
| 5 | `dynamic_conviction_switch:t0.90_strict_fallback` | 43.73% | 18.80% | 1.62 | 4.24 | 4.27 | 4/10 | False |
| 6 | `dynamic_conviction_switch:t0.95_strict_fallback` | 43.73% | 18.80% | 1.62 | 4.24 | 4.27 | 4/10 | False |
| 7 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | 39.53% | 18.80% | 1.69 | 10.82 | 7.54 | 5/10 | False |
| 8 | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | 32.94% | 11.22% | 1.45 | 10.16 | 4.10 | 5/10 | True |

## New MDD30 family candidates

| Candidate | Fold count | Comp | Max OOS MDD | Sharpe | Sortino | PF | Hit | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mdd30_risk_scaled:dyn085_val_mdd30_cap1_50` | 10 | 84.64% | 27.17% | 2.07 | 15.16 | 8.30 | 5/10 | primary shadow |
| `mdd30_risk_scaled:dyn085_x1_50` | 10 | 84.64% | 27.17% | 2.07 | 15.16 | 8.30 | 5/10 | primary shadow |
| `mdd30_risk_scaled:dyn085_x1_25` | 10 | 68.68% | 23.07% | 2.07 | 15.24 | 8.35 | 5/10 | shadow |
| `mdd30_risk_scaled:dyn100_x1_50` | 10 | 61.04% | 27.17% | 1.69 | 10.72 | 7.45 | 5/10 | shadow |
| `mdd30_risk_scaled:cross_v35_x1_50` | 4 | 44.46% | 20.07% | 2.40 | 20.65 | 16.44 | 2/4 | partial-fold only |
| `mdd30_barbell_blend:dyn085_75_aware_v35_25_x1_50` | 4 | 30.38% | 22.51% | 2.73 | 46.34 | 21.55 | 2/4 | partial-fold only |
| `mdd30_barbell_blend:dyn085_70_strict_growth_30_x1_50` | 6 | 29.83% | 19.72% | 1.56 | 11.37 | 4.74 | 2/6 | partial-fold only |
| `mdd30_high_vol_gate:validation_breakout_or_defensive_scaled` | 9 | 22.48% | 27.17% | 0.80 | 2.57 | 1.92 | 4/9 | shadow |
| `mdd30_high_vol_gate:breakout_barbell_blend` | 9 | 20.20% | 22.12% | 0.76 | 3.00 | 1.85 | 4/9 | shadow |

## OOS-mining controls

- Metric reconciliation: `{'candidate_count': 73, 'metrics_reconciled': True, 'mismatches': []}`
- Dynamic self-feed audit: `{'no_same_month_dynamic_self_feeding': True, 'rule': 'same_fold_dynamic_switch_label_oos_utility_or_oracle_rank_not_used', 'violations': []}`
- Online weight audit: `{'fully_lagged_online_weights': True, 'rule': 'month_m_weights_use_only_completed_months_before_m', 'violating_months': []}`
- Bridge protocol audit: `{'current_fold_oos_used_for_bridge_weighting': False, 'manifest_frozen_before_bridge_evaluation': True, 'post_oos_expansion_for_same_protocol': False, 'same_month_dynamic_self_feeding': False}`
- Promotion hard stop: `{'if_false_recommendation': 'fresh_forward_shadow_required_before_promotion', 'promotable': False, 'promotion_hard_stop_pass': False, 'promotion_hard_stop_reasons': ['blocked_non_clean_research_variant']}`

## Fold-level OOS returns: primary shadow vs clean baseline

### `mdd30_risk_scaled:dyn085_x1_50`

| Fold | Validation ret | Validation MDD | OOS ret | OOS MDD | OOS-used-for-selection | Post-OOS research |
|---|---:|---:|---:|---:|---:|---:|
| 2025-09 | 0.02% | 0.38% | 0.20% | 0.11% | False | True |
| 2025-10 | 2.58% | 3.80% | -1.09% | 2.07% | False | True |
| 2025-11 | 41.12% | 11.25% | 18.00% | 14.79% | False | True |
| 2025-12 | 2.85% | 3.74% | -0.21% | 2.44% | False | True |
| 2026-01 | 139.87% | 7.47% | 14.39% | 5.85% | False | True |
| 2026-02 | 192.30% | 16.63% | 28.69% | 27.17% | False | True |
| 2026-03 | 2.64% | 5.50% | -3.97% | 4.19% | False | True |
| 2026-04 | 8.64% | 5.59% | -3.05% | 6.81% | False | True |
| 2026-05 | 70.24% | 9.64% | 16.69% | 23.30% | False | True |
| 2026-06 | 151.59% | 13.30% | -1.07% | 1.56% | False | True |

### `mdd30_risk_scaled:dyn085_x1_25`

| Fold | Validation ret | Validation MDD | OOS ret | OOS MDD | OOS-used-for-selection | Post-OOS research |
|---|---:|---:|---:|---:|---:|---:|
| 2025-09 | 0.02% | 0.32% | 0.17% | 0.09% | False | True |
| 2025-10 | 2.15% | 3.18% | -0.91% | 1.73% | False | True |
| 2025-11 | 33.90% | 9.43% | 15.13% | 12.46% | False | True |
| 2025-12 | 2.39% | 3.12% | -0.17% | 2.04% | False | True |
| 2026-01 | 107.93% | 6.26% | 12.01% | 4.89% | False | True |
| 2026-02 | 147.26% | 13.98% | 24.00% | 23.07% | False | True |
| 2026-03 | 2.24% | 4.60% | -3.31% | 3.50% | False | True |
| 2026-04 | 7.22% | 4.67% | -2.53% | 5.69% | False | True |
| 2026-05 | 56.56% | 8.07% | 13.98% | 19.75% | False | True |
| 2026-06 | 116.91% | 11.09% | -0.89% | 1.30% | False | True |

### `dynamic_conviction_switch:t0.85_risk_capped_fallback`

| Fold | Validation ret | Validation MDD | OOS ret | OOS MDD | OOS-used-for-selection | Post-OOS research |
|---|---:|---:|---:|---:|---:|---:|
| 2025-09 | 0.01% | 0.25% | 0.14% | 0.07% | False | False |
| 2025-10 | 1.73% | 2.55% | -0.72% | 1.39% | False | False |
| 2025-11 | 26.81% | 7.58% | 12.20% | 10.08% | False | False |
| 2025-12 | 1.92% | 2.50% | -0.13% | 1.63% | False | False |
| 2026-01 | 80.04% | 5.03% | 9.62% | 3.92% | False | False |
| 2026-02 | 108.23% | 11.29% | 19.24% | 18.80% | False | False |
| 2026-03 | 1.82% | 3.70% | -2.65% | 2.80% | False | False |
| 2026-04 | 5.79% | 3.74% | -2.02% | 4.57% | False | False |
| 2026-05 | 43.70% | 6.48% | 11.24% | 16.08% | False | False |
| 2026-06 | 86.60% | 8.87% | -0.71% | 1.04% | False | False |
