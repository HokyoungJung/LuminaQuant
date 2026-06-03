# OOS-oracle hybrid assimilation final report

- source: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_oos_oracle_bridge_20260602/oos_oracle_bridge_walkforward_latest.json`
- manifest sha256: `14e527f083f20c805d4044bf8404efb62c3c48d922d4fada1e769e826fc09e9c`
- metric reconciliation: `True`
- no same-month dynamic self-feeding: `True`
- bridge OOS weighting used current fold OOS: `False`

## Clean comparison

| Role | Candidate | OOS comp | Oracle ratio | Hit | Min OOS | Latest OOS | Max MDD | Sharpe | Sortino | Mean train | Mean val | Min val | Hard-stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Top dynamic clean challenger | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | 53.33% | 56.06% | 5/10 | -2.65% | 0.00% | 19.20% | 2.07 | 13.82 | 113.78% | 27.00% | 0.01% | `False` |
| Dynamic previous reference threshold | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | 52.20% | 54.88% | 5/10 | -2.65% | -0.73% | 19.20% | 2.02 | 15.08 | 125.43% | 35.35% | 0.01% | `False` |
| Robust v3.5 baseline | `cross_candidate_hybrid:hybrid_v3_5` | 18.95% | 19.92% | 4/10 | -5.68% | -0.76% | 15.40% | 0.84 | 3.19 | 174.17% | 53.08% | 22.76% | `False` |
| Robust v3.6 baseline | `cross_candidate_hybrid:hybrid_v3_6` | 18.89% | 19.86% | 4/10 | -5.73% | -0.58% | 13.75% | 0.81 | 3.86 | 157.40% | 57.94% | 32.10% | `False` |
| Robust v3.5 train+val fit | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | 22.18% | 23.32% | 4/10 | -5.29% | -0.85% | 15.43% | 0.97 | 4.54 | 170.25% | 53.33% | 24.43% | `False` |
| Robust v3.6 train+val fit | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | 30.85% | 32.43% | 5/10 | -5.01% | -0.58% | 14.25% | 1.27 | 5.91 | 153.01% | 58.17% | 33.03% | `True` |
| Bridge fixed dynamic assimilation | `hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1` | 11.94% | 12.55% | 4/10 | -8.89% | -0.73% | 19.20% | 0.57 | 1.42 | 158.85% | 55.06% | 31.02% | `False` |
| Bridge risk-capped dynamic assimilation | `hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1_riskcap` | 18.39% | 19.34% | 5/10 | -7.62% | -0.73% | 19.20% | 0.81 | 2.50 | 150.74% | 50.99% | 25.40% | `False` |
| Bridge fully-lagged hedge assimilation | `hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1_hedge` | 9.46% | 9.94% | 4/10 | -9.78% | -0.73% | 19.20% | 0.48 | 1.28 | 161.21% | 57.29% | 30.09% | `False` |

## OOS oracle diagnostic only

- OOS oracle comp: `95.12%`
- hit: `7/10`
- min OOS: `-0.40%`
- max OOS MDD: `13.21%`
- promotion allowed: `False` (current OOS winner selection, diagnostic upper bound only)

## Conclusion

- Direct OOS-oracle fitting is rejected for deployable/live use.
- Frozen bridge assimilation did not improve performance: best bridge riskcap OOS comp is below both dynamic challenger and robust v3.6 train+val fit.
- Best high-comp path remains dynamic switch as paper/shadow challenger, not real-money.
- Best robust paper default from this clean rerun is `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` because it beats the robust-default threshold with max OOS MDD <= 15%.
- Further uplift after this run should be a new pre-registered protocol or forward-shadow evidence, not in-run tuning against this OOS artifact.
