# Divergence: Walk-forward -999 sentinel replacement

**Date:** 2026-06-10  
**Phase:** 4.3  
**Classification:** Precision improvement (not a regression)  
**Artifact:** `baseline/golden/walk_forward_results.json` (Variant A) folds 1 and 3

---

## What changed

Variant A (`walk_forward_results.json`) folds 1 and 3 contained:

```json
"val_metrics": {"sharpe": -999.0, "cagr": 0.0, "max_dd": 0.0}
```

These were degenerate sentinel values, not real metrics. They arose because the
validation window (3 months ≈ 90 bars) was shorter than the long MA window
(120 bars), so no trades were generated and the equity curve was flat
(`std_r == 0` → `sharpe = -999.0`).

Phase 4.3 replaces this with:

- `InsufficientWarmupError` raised when warmup context is absent and the window
  is too short for the indicator
- Walk-forward evaluation **always** provides warmup context (120 bars prepended),
  so all folds produce real metrics (matching Variant B)

## Cause

The `-999.0` sentinel in `fast_eval.py` was intentionally emitted for the
zero-variance-returns case. It was never intended to appear in golden output —
it signalled a degenerate condition that should have been prevented by proper
warmup-context supply.

## Why this is an improvement

| Fold | Variant A `val_sharpe` | Variant B `val_sharpe` |
|------|------------------------|------------------------|
| 1    | −999.0 (sentinel)      | −0.2350 (real metric)  |
| 2    | 0.7141                 | −1.2458                |
| 3    | −999.0 (sentinel)      | 0.6152 (real metric)   |

The Variant B values are real strategy performance metrics over the validation
window. The Variant A −999 values are meaningless: they indicate "the strategy
made no trades" due to missing warmup context, not genuine strategy performance.

## Phase 4 oracle

**Variant B** (`baseline/golden/walk_forward_results_warmup.json`) is the Phase 4
correctness oracle. The tolerance comparator (`GoldenComparator`) skips
Variant A −999 sentinels via `skip_sentinel=-999.0`.

Any new divergence from Variant B at `rtol > 1e-8` is a hard correctness
regression and must block the gate.

## References

- `src/lumina_quant/optimization/walkers.py`: `InsufficientWarmupError`,
  `check_warmup_sufficient`, `ma_cross_equity`
- `src/lumina_quant/backtesting/golden_comparator.py`: `compare_to_golden`
- `tests/integration/test_walk_forward_golden.py`: integration test
- `scripts/capture_golden_baseline.py`: golden generation script
