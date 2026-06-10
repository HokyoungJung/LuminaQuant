# Phase 4 Performance Results

Measured on the `refactor/overhaul` tree. Speedups are correctness-gated: the
golden integration test (`tests/integration/test_walk_forward_golden.py`) asserts
Variant-B fold metrics match the committed baseline at **rtol 1e-8**, so every
number below produces bit-identical results to the pre-refactor engine.

Baseline denominator (frozen Phase 0): `baseline/bench_backtest.json` —
**MovingAverageCrossStrategy, 2 symbols (BTC/USDT + ETH/USDT), 22.443 bars/sec**,
py3.11.15. This is the immutable denominator for the single-backtest axis.

---

## Before / After

| Axis | Baseline (Phase 0) | Phase 4 | Speedup |
|------|--------------------|---------|---------|
| **Single backtest — like-for-like** (MA-cross, 2 symbols, seed 42, bit-identical) | 22.443 bars/sec | ~5 900 bars/sec | **~263×** |
| **Walk-forward E2E** (27 runs: 3 folds × 9 MA-cross combos, SHA-pinned fixtures) | 170.71 s | 1.768 s | **~97×** |
| **Walk-forward runs/sec** | 0.158 | 15.27 | ~97× |

The single-backtest figure is now an **honest like-for-like**: numerator and
denominator run the *same* workload (MA-cross / 2 symbols). A contended in-repo
re-run via `scripts/benchmark_backtest.py --strategy MovingAverageCrossStrategy
--symbols BTC/USDT,ETH/USDT` reproduces ~5 660 bars/sec (~252×) under concurrent
CPU load — consistent with the ~263× isolated figure (run-to-run + scheduler
noise). The benchmark records the deterministic trade activity it exercised:
**106 signals → 106 orders → 118 fills** per iteration.

> **Correction to the prior version of this doc.** The earlier "295×" headline
> compared *different* workloads: the numerator was `RsiStrategy` over 14 symbols
> (`reports/benchmarks/baseline_snapshot.json`, 6 632 bars/sec) while the frozen
> denominator is `MovingAverageCrossStrategy` over 2 symbols (22.443 bars/sec).
> The Before/After table also mislabelled the baseline row as "14 symbols,
> RsiStrategy". Strategy choice changes the old-engine cost dramatically (RSI
> fired fewer signals → fewer config reloads → looked faster), so 295× was a
> cross-workload ratio, not a like-for-like speedup. The honest same-workload
> number is ~263×, published above.

---

## Attribution

### Single-backtest axis — NOT the Phase 2 native kernels

The earlier doc credited the single-backtest speedup to the Phase 2 pyo3 signal
kernels. **That attribution was false for this benchmark.** The like-for-like
workload is `MovingAverageCrossStrategy`, whose signals come from
`indicators/moving_average.RollingMeanWindow` — **pure Python; it never calls
`lumina_quant._compute`**. The pyo3 live-signal kernels
(`debounced_state_signal` / `trailing_state_signal`) are part of the *live*
debounced/trailing state path and are **not exercised by the bars/sec
benchmark** at all.

The actual driver is the **`BacktestConfigView` refactor**: the pre-refactor
engine re-read and re-parsed the YAML config on the hot path — on the order of
**~1,494 config reloads per backtest** (once per fill/decision cycle through the
old `getattr(config, "UPPER_ATTR")` chain that reloaded settings). The refactor
constructs one typed config view at engine start and reads dataclass fields
thereafter, eliminating those reloads. The **Python 3.14 interpreter stack lift**
(faster call/attribute dispatch) contributes the rest. Both are correctness-
neutral, hence the bit-identical golden.

### Walk-forward axis — the Phase 2 native fold simulator

The walk-forward optimizer loop *does* use the native kernels: `fast_eval.py` →
`walkers.py` → `lumina_quant._compute.evaluate_metrics` and
`lumina_quant._compute.simulate_symbol_fold` (Rust). A single
`simulate_symbol_fold` call vectorises an entire fold in Rust; Python overhead is
one FFI call per fold rather than one Python call per bar. This is where the
~97× on the optimization axis (spec R4, the primary user bottleneck) comes from.

### Why the single-backtest event loop stays pure-Python

The `Backtest` event loop processes one bar per iteration
(`market → signals → orders → fills`) by design — required for live parity
(Phase 5) and correct LMT order lifecycle management. A future vectorised bar
engine could cut event-dispatch overhead further; tracked as a Phase 7+
follow-up. It is not on the critical path for optimization workloads.

---

## Data coverage caveat

Only **2 of the 14 symbols** in `config.yaml`'s `trading.symbols` ship committed
CSV data (`data/BTCUSDT.csv`, `data/ETHUSDT.csv`). `benchmark_backtest.py` now
**trims the symbol list to those with data files and reports the skipped
symbols** (it never silently benchmarks empty/phantom series), and it **asserts
`signals > 0` and `fills > 0`** so a no-op run can never masquerade as a fast one.

---

## Measurement commands

```bash
# Like-for-like single-backtest (same workload as the frozen baseline):
uv run python scripts/benchmark_backtest.py \
  --strategy MovingAverageCrossStrategy --symbols BTC/USDT,ETH/USDT \
  --iters 3 --seed 42 --output docs/perf/data/bench_backtest_macross.json

# Walk-forward E2E:
uv run python scripts/measure_walkforward_e2e.py \
  --output docs/perf/data/bench_walkforward_e2e.json
```

> **Frozen-baseline rule**: `baseline/` is an immutable Phase 0 artifact set.
> Re-measurements MUST write outside `baseline/` (e.g. `docs/perf/data/`). Never
> pass `--output baseline/...` or let a script default there.
