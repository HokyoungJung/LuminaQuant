# Clean strategy logic / sizing audit

## Clean boundary

- Clean means: current fold locked OOS is not used for selection, same-month OOS/oracle labels are not used, and no post-OOS research variant is allowed to feed a downstream clean selector/optimizer.
- After audit, post-OOS research rows such as `mdd30_*` and `risk_enhanced_blend:*` are blocked from clean downstream pools and marked fresh-forward shadow only.
- Fast recompute path: existing fold rows can now be re-aggregated without rerunning Optuna via `--recompute-from-json`.

## Best clean candidate

`dynamic_conviction_switch:t0.85_risk_capped_fallback`

- OOS period: 2025-09-01T00:00:00 → 2026-06-01T06:30:00
- OOS comp: 53.38%
- Annualized approx: 67.08%
- Max OOS MDD: 18.80%
- Sharpe / Sortino: 2.07 / 15.31
- Profit factor: 8.40
- Hit: 5/10
- Clean: true

## Best research / shadow candidate

`mdd30_risk_scaled:dyn085_x1_50` / `mdd30_risk_scaled:dyn085_val_mdd30_cap1_50`

- OOS comp: 84.64%
- Annualized approx: 108.73%
- Max OOS MDD: 27.17%
- Sharpe / Sortino: 2.07 / 15.16
- Profit factor: 8.30
- Hit: 5/10
- Clean: false; post-OOS research variant, fresh-forward shadow required.

## Position sizing status

- Base per-asset/profile sleeves: tuned by train/validation Optuna/profile selection.
- Individual robust portfolios: position multipliers are Optuna-tuned on train/validation, capped by gross notional and concentration constraints.
- Cross-candidate and dynamic-aware hybrids: v3.5/v3.6 weights are Optuna-tuned on train-only or train+validation according to each candidate label; locked OOS is report-only.
- Dynamic conviction switch: not a continuous sizing optimizer; it chooses one validated expert stream per fold with weight 1.0.
- MDD30 research sleeves: not fully Optuna-sized; they are fixed or validation-MDD-capped scale overlays, e.g. 1.25x/1.50x on the clean dynamic stream. Therefore they are risk-budget experiments, not clean optimized live sizing yet.

## Why full reruns were slow

Full walk-forward reruns rebuild each monthly fold from source candidates and repeatedly run profile/individual/strict/relaxed/hybrid Optuna. With 10 folds, 69 assets, 8 timeframes, and hybrid trials at 96, this is expected to be long. Clean label/report fixes should not require that, so the new `--recompute-from-json` path reduces those recalculations to sub-second report regeneration.
