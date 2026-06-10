# Phase 4 Performance Results

Measured 2026-06-10 on the Phase 4 refactor tree (`refactor/overhaul`).
Workload is SHA-pinned (PROVENANCE.json): BTCUSDT + ETHUSDT synthetic 1 000-day
fixtures, seed 42, identical hardware to Phase 0 baseline.

---

## Before / After

| Axis | Baseline (Phase 0) | Phase 4 | Speedup |
|------|--------------------|---------|---------|
| **bars/sec** (median, 3 iters, 1 268 bars, 14 symbols, RsiStrategy, seed 42) | 22.44 | 6 632 | **295×** |
| **Single-backtest median wall-clock** (same workload) | 56.50 s | 0.191 s | 295× |
| **Walk-forward E2E** (27 runs: 3 folds × 9 MA-cross combos, SHA-pinned fixtures) | 170.71 s | 1.768 s | **97×** |
| **Walk-forward runs/sec** | 0.158 | 15.27 | 97× |

Independent verification (team-lead, same workload): 6 904 bars/sec (308×),
1.800 s E2E (95×).  Difference is run-to-run noise (JIT warm-up, OS scheduler).

Baseline source: `baseline/perf-baseline.json`
Phase 4 snapshots: `reports/benchmarks/baseline_snapshot.json`,
`baseline/bench_walkforward_e2e.json`

---

## Attribution

All speedup is correctness-gated: the golden integration test
(`tests/integration/test_walk_forward_golden.py`) asserts Variant-B fold metrics
agree with the committed baseline at **rtol 1e-8**.

### Primary driver — Phase 2 pyo3 kernels

The walk-forward optimizer loop calls `fast_eval.py` → `walkers.py` →
`lumina_quant._compute.evaluate_metrics` (Rust) and
`lumina_quant._compute.simulate_symbol_fold` (Rust).  These replace the
pure-Python per-bar event loop for the inner fold evaluation.  A single
`simulate_symbol_fold` call vectorises the entire fold in Rust; Python overhead
is one FFI call per fold, not one Python function call per bar.

### Secondary driver — Phase 4.1 unified ExecutionModel

`ExecutionModelConfig.from_runtime()` is called once at construction and the
hot-path `compute_fill()` method is a plain Python function with no attribute
lookup overhead beyond `self._rng`.  The previous per-fill `getattr` chain
(FillModel + LiquidityModel + LatencyModel all reading from BacktestConfigView
at fill time) is replaced by a single dataclass field access.

### Why the single-backtest event loop is untouched

The `Backtest` event loop in `backtest.py` is intentionally left as an
event-driven pure-Python loop in Phase 4.  The loop processes one bar per
iteration (`market → signals → orders → fills`); this structure is needed for
live parity (Phase 5) and for correct LMT order lifecycle management.

The speedup on the single-backtest axis comes entirely from the fact that
`benchmark_backtest.py` exercises `RsiStrategy`, which routes signal computation
through `lumina_quant._compute.debounced_state_signal` / `trailing_state_signal`
(Phase 2 kernels).  The per-bar overhead of the Python event dispatch
(`Queue.put` / `Queue.get`) is now amortised by the fast kernel; on the 1 268-bar
benchmark the combined event-loop overhead is ~0.19 s.

**Path to further improvement**: a vectorised bar engine (processing all bars in
a single NumPy/Rust pass before feeding the event queue) would further reduce the
event-loop overhead for the single-backtest axis.  This is tracked as a Phase 7+
follow-up.  The walk-forward optimizer is the primary user bottleneck (spec R4)
and is already ~97× faster; the event-loop is not on the critical path for
optimization workloads.

---

## Measurement commands

```bash
uv run python scripts/benchmark_backtest.py --iters 3 --seed 42
uv run python scripts/measure_walkforward_e2e.py --output docs/perf/data/bench_walkforward_e2e.json
```

> **Frozen-baseline rule**: `baseline/` is a frozen Phase 0 artifact set — the numbers
> in `baseline/perf-baseline.json` are the permanent denominators.  Re-measurements
> MUST write outside `baseline/` (e.g. `docs/perf/data/` or `/tmp/`).  Never pass
> `--output baseline/...` or allow a script's default to target that directory.
> `benchmark_backtest.py` already defaults to `reports/benchmarks/`; the
> `measure_walkforward_e2e.py` default was corrected from `baseline/` to
> `docs/perf/data/` in the same commit as this doc.
