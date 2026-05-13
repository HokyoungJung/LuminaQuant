# StateDistilledRegimeBoostPortfolio — Real current-tail report

Generated: `2026-05-13T14:04:49Z`

## Strategy / factor / calibration used

- Core A: `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125` external-risk state-distilled seed.
- Core B: `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600` pure leadership/unwind seed.
- Overlay: tunable regime classifier, side-bias multipliers, volatility-targeted leverage, conditional booster up to 25x, and frozen neutral-pair overlay.
- Calibration/selection: bounded grid, train+validation score only; locked-OOS opened after freeze as gate/report only.
- Calendar/current-base teacher: hypothesis_reference_only, not selection or promotion target.

## Train/validation selection provenance

- uses_locked_oos_for_selection: `False`
- locked_oos_metrics_visible_during_selection: `False`
- configured/evaluated/product grid: `64` / `64` / `5832`
- freeze hash: `68db1c473bf43778ccdaba7c2e78ab4a754f71dde2557643fa4267b73d8b3535`

## Strict zero-liquidation lane

| Split | Return | MDD | Sharpe | Sortino | Calmar | Liq | Min buffer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | +3.4706% | +3.7181% | 3.4408 | 4.3918 | 0.9334 | 0 | 9837.0960 |
| validation | -1.7179% | +1.7999% | -20.0965 | -10.6949 | -0.9544 | 0 | 9783.9340 |
| locked_oos | -0.3208% | +0.6141% | -7.9524 | -4.6065 | -0.5224 | 0 | 9936.6557 |

Strict promoted success: `False`
Strict rejection reasons: `['validation_return_non_positive', 'locked_oos_return_non_positive', 'locked_oos_sharpe_non_positive', 'locked_oos_sortino_non_positive', 'locked_oos_smart_sortino_non_positive', 'locked_oos_calmar_non_positive']`
Max effective leverage: `4.5000`

## Diagnostic high-leverage nonfatal lane

| Booster cap | Max effective lev | OOS return | OOS MDD | Total liq | Min buffer | Promotion |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5.0 | 4.5000 | -0.3208% | +0.6141% | 0 | 9783.9340 | diagnostic-only |
| 6.0 | 4.5000 | -0.3208% | +0.6141% | 0 | 9783.9340 | diagnostic-only |
| 10.0 | 4.5000 | -0.3208% | +0.6141% | 0 | 9783.9340 | diagnostic-only |
| 15.0 | 4.5000 | -0.3208% | +0.6141% | 0 | 9783.9340 | diagnostic-only |
| 25.0 | 4.5000 | -0.3208% | +0.6141% | 0 | 9783.9340 | diagnostic-only |

## Memory / artifacts

- Peak RSS: `307077120` bytes (`292.85` MiB).
- Summary JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/state_distilled_regime_boost_summary_latest.json`
- Frozen config: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/frozen_config.json`
- Locked-OOS gate: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/locked_oos_gate.json`
- Selection ledger: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/selection_ledger.jsonl`

Research history/source inventory: no global source ledger change; this uses the existing current-tail crypto panel and existing lagged FRED external-state source.
