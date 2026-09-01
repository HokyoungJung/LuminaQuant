# Session Handoff — Nested Hybrid Cleanup + Monthly Refit Eval Performance (2026-06-04 KST)

## What changed

- Enforced no-nested-hybrid policy in `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`.
- Hybrid/blend/selector/gate/bridge rows are no longer valid ingredients for another hybrid/portfolio layer.
- Deprecated nested families now no-op: `dynamic_aware_hybrid`, `risk_enhanced_blend`, `fixed_relaxed_dynamic_blend`.
- Dynamic switch, validation selector, bridge eligible pool, and MDD30 research families now use leaf strategy material only.
- Existing payload recompute marks rows with non-leaf references as `nested_hybrid_dependency=true` and non-clean.

## Performance optimization

- Added cached `_period_metrics` hotpath with prepared return arrays and window metric cache.
- Window slicing now uses int64 timestamp `searchsorted` instead of repeated Pandas sort/mask work.
- Added benchmark: `scripts/research/benchmark_monthly_refit_eval_hotpath.py`.
- Added `--checkpoint-markdown-interval` so full reruns do not re-render a growing Markdown report every fold by default; JSON checkpoint cadence remains configurable with `--checkpoint-interval`.

## No-nested clean recompute

- Historical external report `C:\Users\hoky1\Desktop\deep-research-report.md` used the old 2026-06-03 nested candidate surface. Its governance/execution warnings remain useful, but the strategy ranking is superseded by the no-nested recompute.
- New recompute artifact:
  `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.json`
  and `.md`.
- Source artifact sha256:
  `563aff7f59174a7ebb6b53f9164eb1feb0cf67881e7f203aecb06987024fa58f`.
- This is a governance/ranking repair from existing rows (`recomputed_from_existing_rows=true`), not a fresh no-nested Optuna rerun.
- Clean-only top candidate is now `relaxed_efficiency:hybrid_v3_5` with OOS comp `156.03%`, max OOS MDD `19.75%`, monthly Sharpe `1.69`, Sortino `10.48`, hit `5/10`.
- Raw old winners `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` and `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` are demoted for `nested_hybrid_dependency`, `post_oos_research_variant`, and `requires_fresh_forward_shadow`.

## Verification

```text
uv run python scripts/research/benchmark_monthly_refit_eval_hotpath.py --min-speedup 1.5
# latest rerun speedup=6.375x, checksum identical, PASS

uv run python -m pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# 27 passed in 0.25s

uv run python -m ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/benchmark_monthly_refit_eval_hotpath.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# All checks passed
```

## Important interpretation

The prior 2026-06-03 exact-blend final shadow candidates such as `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` are now historical/deprecated under the new policy. They should not be promoted or used as final selection evidence unless rebuilt from leaf sleeves without nested hybrid inputs and then fresh-forward shadowed.
