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

## Verification

```text
uv run python scripts/research/benchmark_monthly_refit_eval_hotpath.py --min-speedup 1.5
# speedup=7.519x, PASS

uv run python -m pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# 23 passed in 0.16s

uv run python -m ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/benchmark_monthly_refit_eval_hotpath.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# All checks passed
```

## Important interpretation

The prior 2026-06-03 exact-blend final shadow candidates such as `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` are now historical/deprecated under the new policy. They should not be promoted or used as final selection evidence unless rebuilt from leaf sleeves without nested hybrid inputs and then fresh-forward shadowed.
