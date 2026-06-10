# Post-Refactor Live Tick Latency

Records the tick→signal→order latency proxies after the Phase 2 pyo3 migration,
closing AC6 (the previous measurement script crashed on the deleted ctypes
loader `native_live_signal_backend._NATIVE_DLL`).

- Measured: 2026-06-10 on `refactor/overhaul`, 5 000 iterations.
- Tool: `scripts/measure_tick_latency.py` (now uses the pyo3 backend).
- Data: `docs/perf/data/bench_tick_latency.json`.
- Backend resolved: `rust:pyo3` (`native_available: true`).

## Live signal per-tick (debounced + trailing state kernels)

| Metric | Baseline (Phase 0) | Post-refactor (pyo3) | Change |
|--------|--------------------|----------------------|--------|
| Combined debounced+trailing **median** | 0.0305 ms | **0.0090 ms** | ~3.4× faster |
| Combined **p95** | — | 0.0161 ms | — |
| Debounced median | — | 0.0041 ms | — |
| Trailing median | — | 0.0049 ms | — |

Baseline denominator: `baseline/perf-baseline.json` →
`live_signal_kernel_latency_median_ms = 0.030498` (frozen Phase 0 artifact).

**Justification (report-and-justify, plan §8 #6):** the post-refactor median is
*lower* than the frozen baseline — the pyo3 kernel call is cheaper per tick than
the former ctypes `CDLL` loader path it replaced. This is the one benchmark in
this repo that genuinely exercises the Phase 2 native kernels (the backtest
bars/sec benchmark does not — see `docs/perf/phase4-results.md`). No regression;
the live signal hot path got faster.

## Adjacent hot-path proxies (for context; not signal latency)

| Proxy | Post-refactor median |
|-------|----------------------|
| Paper order-state transition chain (SUBMITTED→ACKED→FILLED) | 0.00091 ms |
| Replay parity-checker overhead (single synthetic event) | 0.00198 ms |

These are pure-Python dict-lookup / comparison costs with no I/O; full
tick→order latency including network/exchange I/O is measured against testnet in
the live phase, not here.

## Reproduce

```bash
uv run python scripts/measure_tick_latency.py --iters 5000 \
  --output docs/perf/data/bench_tick_latency.json
```

> The script default output is `docs/perf/data/` — never `baseline/` (frozen).
