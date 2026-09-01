# Profit moonshot integer-leverage tuning/audit — 2026-05-09

## Objective
Continue from the green monthly-budget base and verify whether the apparent performance lift is real when leverage is constrained to integer values and raw/unlevered train quality is audited.

## Baseline preserved
- Source baseline before this lane: `0cf9bcf5d1dbef6ffec34f248c6331ebdf9a7b5f` on `private-main` / `private/main`.
- Previous retained base artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/continued_optimization_20260509/passing_candidate_latest.json`.
- Previous base metrics: train monthly `+2.0000%`, validation monthly `+9.8490%`, locked-OOS return `+6.8582%`, locked-OOS MDD `+0.8198%`, OOS return/risk `8.365933`.

## Code policy changes
- `train_val_monthly_return_budget` now evaluates an integer leverage grid; the continuous leverage required to hit a monthly floor is recorded only as a diagnostic.
- Promotion gates now include:
  - integer leverage,
  - post-leverage train monthly buffer `>= +2.25%`,
  - raw/unlevered train monthly `>= +1.0%`,
  - raw/unlevered validation monthly `>= +2.0%`.
- Current-base sleeve names are anchored into the candidate pool when present, and the exact current-base sleeve tuple is forced into combination evaluation so the base is not accidentally dropped by top-N/cap ordering.
- Locked-OOS remains report-only/gate-only; no ranking, weights, or leverage fitting use OOS.

## Replays
### Default all-family candidate universe
- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_20260509/fresh_portfolio_tuning_latest.json`.
- Specs `73,465`; sleeves `30`; success candidates `0`; peak RSS `1308.7266 MiB`.
- Selected-by-validation: leverage `3`, train monthly `+2.7510%`, validation monthly `+11.8767%`, raw train `+1.0111%`, OOS return `+6.7836%`; rejected because OOS return/risk did not beat current base.

### Alpha-v2 merged candidate universe, top30
- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_alpha_v2_20260509/fresh_portfolio_tuning_latest.json`.
- Specs `73,465`; sleeves `30`; success candidates `0`; peak RSS `1319.5859 MiB`.
- Selected-by-validation: leverage `6`, train monthly `+3.1264%`, validation monthly `+18.7177%`, raw train `+0.6004%`, OOS return `+7.9515%`; rejected on raw-train and current-base return/risk gates.

### Alpha-v2 merged candidate universe, top40 wider search
- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_alpha_v2_top40_20260509/fresh_portfolio_tuning_latest.json`.
- Audit summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_audit_20260509.json` / `.md`.
- Specs `158,620`; sleeves `40`; success candidates `0`; peak RSS `2523.5234 MiB`; `/usr/bin/time` max RSS `2,584,088 KiB`; wall `22:34.38`.
- Selected-by-validation: leverage `3`, train monthly `+2.7510%`, validation monthly `+11.8767%`, raw train `+1.0111%`, OOS return `+6.7836%`; rejected because current-base OOS return/risk was not beaten.
- Diagnostic-best-OOS: leverage `6`, OOS return `+18.4446%`, OOS MDD `+2.2305%`, OOS return/risk `8.2694`; rejected because raw train `+0.8615%`, train Sortino, and current-base return/risk failed.

## Current-base integer stress result
The exact current-base sleeve tuple was forced through the integer grid. Its integer `5x` row looked attractive on OOS but failed the stricter train-quality audit:
- Train monthly `+3.8443%`; validation monthly `+20.1014%`; OOS monthly `+6.4641%`.
- OOS return `+14.6371%`; MDD `+1.6919%`; return/risk `8.6514`; Sharpe `5.7215`; Sortino `7.4828`; smart Sortino `6.9764`; Calmar `66.2284`.
- Failed gates: `raw_train_monthly_return_gte_1pct`, `train_sortino_high`.

## Decision
- Final research outcome for this lane: `no_improvement_current_base_retained`.
- Promoted success candidates: `0`.
- The leverage suspicion is confirmed: larger integer leverage can make OOS return appear dramatically better, but the raw train and train-quality gates prevent automatic promotion.
- Current base remains the reference until a candidate passes the raw/integer train-quality audit and beats current-base OOS return/risk with train/validation-only selection.

## Verification status
- Targeted tests after the integer/raw-gate patch: `26 passed in 0.11s`.
- Focused ruff on touched files: passed.
- Full pytest: `1220 passed in 338.07s (0:05:38)`; `/usr/bin/time` max RSS `2,673,972 KiB`.
- Repo ruff: `All checks passed`.
- Compileall: `python3 -m compileall -q src scripts tests` -> pass.
- Whitespace: `git diff --check` -> pass.
- Mission result updated with integer-leverage audit evidence; validator and GitHub Actions finalization pending until source commit is pushed.

## Source push / CI finalization
- Source/evidence Lore commit: `6055d691b07fac1d54362c51157935d628f3f129`, pushed to `private/main`.
- GitHub Actions for `6055d69`: `private-ci` run `25602373489` success; `ci` run `25602373501` success.
- Mission validator: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/validation_latest.json` -> `status=passed`, `passed=true`.
