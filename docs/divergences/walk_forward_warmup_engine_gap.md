# Divergence: Variant B warmup context is a reference oracle, NOT an engine guarantee

**Date:** 2026-06-10
**Branch:** `refactor/overhaul`
**Classification:** Documented scope clarification (no behaviour change) — closes a silent gap
**Artifacts:** `baseline/golden/walk_forward_results_warmup.json` (Variant B)

---

## The gap

`baseline/golden/walk_forward_results_warmup.json` (Variant B) is produced by the
**numpy reference oracle** `ma_cross_equity(..., context=...)` in
`src/lumina_quant/optimization/walkers.py`, which prepends `max(long_window)`
warmup bars so both MAs are fully warm at bar 0 of each eval window. The golden
integration test (`tests/integration/test_walk_forward_golden.py`) reproduces
exactly that numpy computation.

The **event-driven engine** (the windowed/chunked optimizer path) does **not**
reproduce Variant B. Concretely:

1. The chunk loader requests warmup rows
   (`cli/optimize.py` → `warmup_bars=BT_CHUNK_WARMUP_BARS`,
   `storage/parquet/ohlcv_repo.py` extends the query window backwards).
2. But `run_backtest_chunked` (`backtesting/chunked_runner.py`) builds each
   `Backtest` with `start_date=chunk_start`, and
   `OHLCVFrameLoader.select` (`compute/ohlcv_loader.py:85,97`) then filters
   `datetime >= start_date` — **dropping the warmup rows before the strategy
   sees them.** First-chunk / first-fold indicators therefore cold-start inside
   the eval window.
3. `BT_CHUNK_WARMUP_BARS` defaults to `0` (`cli/optimize.py:209`), so by default
   no warmup is even requested.

So `InsufficientWarmupError`, `check_warmup_sufficient`, and the "no −999
sentinel" contract (`optimization/walkers.py`) are **reference-oracle / test
constructs** — they are referenced only by `scripts/capture_golden_baseline.py`
and `tests/integration/test_walk_forward_golden.py`. The production optimizer
never calls them and still emits `-999.0` on the no-data path and swallows
backtest exceptions as `-999.0` (`cli/optimize.py`).

## What is actually computed (corrected statement)

- **Variant B golden** constrains the **numpy reference implementation only**
  (warmup-context MA-cross). It is NOT a statement about engine output.
- The **event-driven engine** evaluates each window/chunk **without** warmup
  context — indicators warm up inside the eval window for the first chunk/fold.

This divergence doc makes that explicit so the gap is no longer silent: a reader
of the Variant B golden must not assume the live optimizer reproduces it.

## Remediation (future work — cross-cutting, multiple owners)

> **Tracked:** full implementation plan + acceptance criteria live in
> [`docs/TODO.md`](../TODO.md) item 1 (priority HIGH before relying on
> walk-forward parameter selection for real money).

To make warmup context real in the engine (so Variant B becomes an engine
guarantee), all of the following are required and span files owned by different
workers:

- `compute/ohlcv_loader.py` + `backtesting/chunked_runner.py`: load
  `start_date = chunk_start − warmup` and tag warmup rows.
- engine/strategy: **suppress order generation** until `chunk_start` so warmup
  rows only prime indicators (no trades booked in the warmup region).
- `cli/optimize.py` (worker-3): wire `check_warmup_sufficient` into the fold
  evaluator and stop swallowing backtest exceptions / no-data as `-999.0`
  sentinels; either honour `BT_CHUNK_WARMUP_BARS` end-to-end or remove the dead
  plumbing.

Until then, Variant B stands as a documented reference oracle and the engine's
cold-start behaviour is the known, documented reality.

## References

- `src/lumina_quant/optimization/walkers.py`: `ma_cross_equity`,
  `check_warmup_sufficient`, `InsufficientWarmupError`
- `src/lumina_quant/compute/ohlcv_loader.py`: `OHLCVFrameLoader.select`
  (`datetime >= start_date` filter)
- `src/lumina_quant/backtesting/chunked_runner.py`: `run_backtest_chunked`
- `src/lumina_quant/cli/optimize.py`: `BT_CHUNK_WARMUP_BARS`, `-999.0` sentinels
- `docs/divergences/walk_forward_no_sentinel.md`: companion sentinel divergence
- `docs/divergences/walk_forward_oracle_lookahead.md`: companion oracle correction
