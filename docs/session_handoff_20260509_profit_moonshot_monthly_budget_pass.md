# Session handoff — profit moonshot monthly-budget pass — 2026-05-09

## Status
A return-quality portfolio candidate now passes the clarified target: stable train/validation/OOS monthlyized return, relaxed MDD budget, high risk-adjusted OOS quality, train/validation-only selection, locked-OOS report/gate-only.

## Candidate
- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`
- Source replay: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_monthly_budget_v1/fresh_portfolio_tuning_latest.json`
- Mode: `train_val_monthly_return_budget`
- Leverage: `2.3427334297703024`
- Selection basis: train/validation-only monthly return budget; locked-OOS was not used for selection.
- Sleeve note: this is a portfolio success, not a standalone non-calendar promotion. The residual-pair sleeve is only admitted inside the train/validation-ranked portfolio and the complete portfolio passes locked-OOS gates.

## Metrics
- Train: return `+26.8207%`, monthlyized `+2.0000%`, MDD `+6.9060%`, Sharpe `1.7213`, Sortino `1.5151`, Calmar `3.8842`, trips `120`.
- Validation: return `+19.9713%`, monthlyized `+9.8490%`, MDD `+6.4935%`, Sharpe `4.0964`, Sortino `4.8859`, Calmar `32.1417`, trips `41`.
- Locked-OOS: return `+6.8582%`, monthlyized `+3.0883%`, MDD `+0.8198%`, Sharpe `5.6537`, Sortino `7.3961`, smart Sortino `7.1536`, Calmar `53.7350`, trips `33`.
- Current champion reference: locked-OOS return `+1.2181%`; candidate beats return and return/risk gates.

## Replay evidence
- Portfolio specs evaluated: `73,465`.
- Success candidates under stricter train/val/OOS quality gates: `8`.
- Peak RSS payload: `920.51171875 MiB`; `/usr/bin/time` max RSS `942604 KB`; below 8 GiB.
- Large generated CSV was intentionally omitted from git; JSON/MD/memory summaries are retained.

## Verification state
- Targeted tests after allocator/gate patch: `14 passed in 0.10s`.
- Mission validator now passes `candidate_return_quality_contract` and RSS checks, but final mission status remains pending until full local tests, source push, and GitHub Actions `ci`/`private-ci` evidence are recorded.

## Next required steps
1. Run full `pytest`, full `ruff`, `compileall`, and `git diff --check`.
2. Commit source + artifacts with Lore message and push to `private/main`.
3. Confirm GitHub Actions `ci` and `private-ci` green for the source commit.
4. Record final CI/push evidence in `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json` and rerun validator to passed.

## Final source/CI evidence
- Source Lore commit: `a1f4e0718975202f416b3e76e358c46a6a3d7ae8`, pushed to `private/main`.
- GitHub Actions: `ci` run `25594783480` success; `private-ci` run `25594783476` success.
- Final mission validator: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/validation_latest.json` -> `passed=true`, `status=passed`.
- OMX performance-goal and autoresearch-goal ledgers recorded pass verdicts for `profit-moonshot-return-quality` / `profit-moonshot-return-quality-alpha`.
