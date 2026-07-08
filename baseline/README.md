# Phase 0b — Golden Outputs + PROVENANCE

**Branch:** `refactor/overhaul`
**Captured:** 2026-06-10
**Plan section:** §3 Phase 0.3/0.3a/0.3b (`quants-agent-overhaul-consensus.md`)

These are the **correctness goldens** — the frozen oracle for all Phase 1–6 hard-blocking
tolerance gates (rtol `1e-8`). Every refactored kernel must produce output within `1e-8`
of these goldens, or requires a `docs/divergences/<artifact>.md` root-cause doc + new
golden approval.

---

## Regeneration Procedure

**NEVER re-fetch the aggTrades fixture during normal regeneration** — re-fetching produces
a different window on a fresh machine and breaks acceptance criterion #2 (bit-exact
reproduction across machines). The fixture is frozen and committed to git.

```bash
# Step 1 (ONE-TIME ONLY — fixture already committed):
# uv run python scripts/fetch_aggtrades_fixture.py

# Step 2 (deterministic, safe to re-run any time):
uv run python scripts/capture_golden_baseline.py
```

Re-running step 2 always produces bit-identical parquet and JSON golden files
(verified across 3 consecutive runs: SHA-256 of all data files identical).

If you need a FRESH baseline (e.g. after Phase 0.5 stack lift), re-fetch AND re-run,
then update `baseline/golden/PROVENANCE.json` and obtain new golden approval before
any Phase gate comparison proceeds.

---

## aggTrades Fixture

| Field | Value |
|---|---|
| File | `tests/fixtures/aggtrades/BTCUSDT_1h_1781046000000_1781049600000.parquet` |
| Symbol | BTCUSDT (Binance Futures) |
| Window | 1781046000000 → 1781049600000 ms (~1 hour) |
| Records | 48,730 aggTrades |
| SHA-256 | `51426ca3fe8493488d637b1bd66f4ba2253c863c9abb32195a8ac8ecef59e0d7` |
| Source | Public REST: `GET https://fapi.binance.com/fapi/v1/aggTrades` (no API key) |

`capture_golden_baseline.py` verifies this SHA-256 on every run and exits non-zero
on mismatch — ensuring the fixture is always the same frozen input.

---

## Synthetic OHLCV Fixtures

| Field | Value |
|---|---|
| Generator | `scripts/capture_golden_baseline.py` (logic from `generate_data.py`) |
| Seed | `numpy.random.seed(42)` set before all random calls |
| Symbols | BTCUSDT, ETHUSDT |
| Days | 1000 (2022-01-01 → 2024-09-26) |
| BTCUSDT path | `tests/fixtures/ohlcv/BTCUSDT_seed42_1000d.parquet` |
| ETHUSDT path | `tests/fixtures/ohlcv/ETHUSDT_seed42_1000d.parquet` |

---

## Golden Artifacts

All files under `baseline/golden/`. Full metadata in `baseline/golden/PROVENANCE.json`.

| Artifact | File | Description |
|---|---|---|
| MA Cross equity curve | `ma_cross_equity_curve.parquet` | 1001-row equity history, MA(10,30) on BTCUSDT+ETHUSDT |
| MA Cross stats | `ma_cross_stats.json` | Sharpe −0.9177, MaxDD 7.65%, Total Return −3.61% (2026-07-08 data-dict isolation recapture) |
| MA Cross trades | `ma_cross_trades.json` | Full trade log (partial fills exercised) |
| MA Cross positions | `ma_cross_positions_sample.json` | First 100 position snapshots |
| BuyHold equity curve | `buyholdstrategy_equity_curve.parquet` | Sanity golden: single LONG entry |
| BuyHold stats | `buyholdstrategy_stats.json` | Sharpe −1.6746, MaxDD 5.63% (2026-07-08 data-dict isolation recapture) |
| Native backends | `native_backends.json` | All 6 backends: sharpe/cagr/mdd + array outputs |
| Walk-forward results | `walk_forward_results.json` | 3 folds × 4 param combos, best params + metrics |
| Frozen configs | `configs/config_frozen.yaml`, `configs/research_frozen.yaml` | SHA-256 locked |
| Provenance | `PROVENANCE.json` | Seed, all SHA-256s, env fingerprint, tolerance contract |

---

## Native Backend Coverage (all 6 required by plan §0.3a)

| Crate | Library | Functions exercised | Golden result |
|---|---|---|---|
| `rust_metrics` | `target/release/liblumina_metrics.so` | `evaluate_metrics` | sharpe=0.2994, cagr=0.0449, max_dd=0.5086 |
| `c_metrics` | `build/liblumina_metrics.so` | `evaluate_metrics` | sharpe=0.2994, cagr=0.0449, max_dd=0.5086 (identical — confirms Phase 2 dedup target) |
| `rust_alpha_fold` | `target/release/liblumina_alpha_fold.so` | `simulate_symbol_fold_native` | n=1000, total_return=−1.335 |
| `rust_hybrid_optuna` | `target/release/liblumina_hybrid_optuna.so` | `evaluate_hybrid_optuna_portfolio` | portfolio_final=9824.91 |
| `rust_live_signals` | `target/release/liblumina_live_signals.so` | `debounced_state_signal_native`, `trailing_state_signal_native` | n=1000 each |
| `rust_rawfirst` | `target/release/liblumina_rawfirst.so` | `aggregate_raw_aggtrades_to_1s` | 3598 1s-OHLCV bars from 48730 aggTrades |

Note: `rust_metrics` and `c_metrics` produce **identical results** — this confirms the
duplication described in plan §2.2 (one is deleted in Phase 2, Rust implementation kept).

---

## 2026-07-08 Engine Golden Data-Dict Isolation Recapture

`docs/divergences/engine-golden-funding-rebaseline-20260707.md` now records that the 2026-07-07 recapture was superseded: preloaded `data_dict` fixtures must not read ambient local sidecar feature stores unless a caller passes `feature_db_path`/`feature_exchange` explicitly. The current engine stats/trades goldens match the CI/no-sidecar values and remain synthetic-fixture evidence only, not a strategy-performance promotion.

## Tolerance Contract

- **rtol:** `1e-8` (default; config-tunable via `validation.golden_rtol`)
- **Determinism:** `same backend + same seed → bit-exact reproduction across machines`
- **Divergence procedure:** comparator fails → author writes `docs/divergences/<artifact>.md`
  (cause: precision-improvement vs bug) → improvement-only divergences get new approved golden;
  bugs block.
- **Cross-reference:** `baseline/env.lock` (sha256 in PROVENANCE.json) pinned by worker-1.

---

# Phase 0c — Performance Baselines

**Branch:** `refactor/overhaul`
**Captured:** 2026-06-10
**Plan section:** §3 Phase 0.4–0.5 (`quants-agent-overhaul-consensus.md`)

These baselines are the authoritative denominators for all Phase 1–6 **report-and-justify**
performance gates (plan §5 gate taxonomy: correctness gates hard-block; perf gates must record
a number and document any regression in `docs/perf/`).

---

## Hardware Fingerprint

| Field | Value |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i5-12400F |
| RAM | 15,871 MB |
| GPU present | yes |
| GPU model | NVIDIA GeForce RTX 3050 OEM |
| GPU VRAM | 8,192 MiB |
| NVIDIA-SMI | 610.43.02 |
| CUDA UMD | 13.3 |
| GPU compute cap | 8.6 (Ampere) |
| Python (uv env) | 3.11.15 (Clang 22.1.3) |
| Platform / kernel | Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.39 |

---

## Canonical Data for Benchmarks

Synthetic benchmark CSV data (Axes 1-E2E, 2) is generated by `generate_data.py` with
`numpy.random.seed(42)`, starting 2024-01-01, 1000 daily bars, written to:

- `data/BTCUSDT.csv`
- `data/ETHUSDT.csv`

**These files are benchmark-only synthetic data** — NOT the canonical golden fixtures.
The canonical fixtures for numerical correctness testing (seed-42, BTCUSDT + ETHUSDT,
1000 daily bars from 2022-01-01, `MovingAverageCrossStrategy`) are authored by worker-2
in task #2 and stored at `baseline/golden/` with SHA-256 pinned in
`baseline/golden/PROVENANCE.json`.

---

## Axis 1 — Walk-Forward / Optimizer Wall-Clock

### 1a. Kernel Micro-Benchmark

**Harness:** `scripts/benchmark_optimization_kernel.py`

**Command:**
```bash
uv run python scripts/benchmark_optimization_kernel.py --bars 50000 --evals 5000 \
  2>&1 | tee baseline/bench_optimizer_kernel.txt
```

**What it measures:** Wall-clock for 5,000 evaluations of `evaluate_metrics_numba`
(the metric-kernel hot inner loop of the walk-forward optimizer) on a 50,000-bar
synthetic return series seeded with numpy default_rng(42).  Includes one warmup call
before the timed loop (JIT compilation amortised).

**Result:** See `baseline/bench_optimizer_kernel.txt`

| Metric | Value |
|---|---|
| `walk_forward_kernel_elapsed_s` | 178.08 |
| `walk_forward_kernel_evals_per_sec` | 28.08 |

### 1b. End-to-End Walk-Forward Run

**Harness:** Canonical 9-combo × 3-fold `MovingAverageCrossStrategy` walk-forward.

**Canonical spec (shared with task #2 golden run):**
- Strategy: `MovingAverageCrossStrategy`, `allow_short=True`
- Symbols: BTCUSDT + ETHUSDT (1000 daily bars from 2022-01-01, seed=42)
- Fixtures: `tests/fixtures/ohlcv/BTCUSDT_seed42_1000d.parquet` + `ETHUSDT_seed42_1000d.parquet`
  — SHA-256 verified against `baseline/golden/PROVENANCE.json` (keyed by symbol name) before use
- Grid: `short_window=[10,20,30]` × `long_window=[40,80,120]` → 9 combinations
- Folds: 3; train=6mo, val=3mo, test=3mo, step=3mo, start=2022-01-01
- Seed: 42

**Command:**
```bash
uv run python scripts/measure_walkforward_e2e.py \
  --output baseline/bench_walkforward_e2e.json
```

**Result:** See `baseline/bench_walkforward_e2e.json`

| Metric | Value |
|---|---|
| `walk_forward_e2e_elapsed_s` | 170.71 |
| runs/sec | 0.158 |
| ms/run | 6322.6 |
| total backtest runs | 27 (9 combos × 3 folds) |

> **Timing denominator note — Variant A workload:** The 170.71s wall-clock is measured
> against the **variant A** golden (fold geometry: train=6mo/val=3mo/test=3mo/step=3mo).
> With `long_window=120` and only ~90 val/test bars, the optimizer kernel emits `-999`
> sentinels for windows shorter than the MA warmup — this is correct sentinel behaviour,
> not a data defect, and the event-driven backtest machinery still runs the full bar loop.
> Worker-2 is separately producing a **variant B** golden (warmup-context fed to val/test
> so all windows yield real metrics) as a richer Phase 4 oracle.  Variant B does **not**
> replace this timing denominator; the Phase 1–6 perf gates reference the variant A
> wall-clock recorded here.

---

## Axis 2 — Single-Backtest Throughput (bars/sec)

**Harness:** `scripts/benchmark_backtest.py`

**Command:**
```bash
uv run python scripts/benchmark_backtest.py \
  --config config.yaml \
  --strategy MovingAverageCrossStrategy \
  --symbols BTC/USDT,ETH/USDT \
  --seed 42 --iters 5 --warmup 1 \
  --output baseline/bench_backtest.json
```

**What it measures:** Median wall-clock seconds and bars/sec for
`MovingAverageCrossStrategy.simulate_trading()` on 1,000-bar BTC/USDT + ETH/USDT
synthetic CSV data (2024-01-01 start, seed=42).  5 measured iterations + 1 warmup
(warms numba JIT and caches).

**Note on psutil:** `psutil` is not in the uv.lock — `max_peak_rss_mb` in the
benchmark JSON is `null`.  Peak RSS is measured separately via
`/usr/bin/time -v` (see Axis 4).

**Result:** See `baseline/bench_backtest.json`

| Metric | Value |
|---|---|
| `backtest_median_bars_per_sec` | 22.44 |
| `backtest_median_s` | 56.50 |
| bars_processed | 1,268 |
| iterations | 5 (1 warmup) |

---

## Axis 3 — Live Tick → Signal → Order Latency (proxies, no live exchange)

Three measurements from `scripts/benchmark_live_signal_backend.py` (existing harness)
and `scripts/measure_tick_latency.py` (new script).  None require a live exchange.
Real end-to-end tick→order latency including network I/O is a Phase 5 deliverable.

### 3a. Live Signal Kernel Latency (Rust ctypes, per-50k-row batch)

**Harness:** `scripts/benchmark_live_signal_backend.py`

**Command:**
```bash
uv run python scripts/benchmark_live_signal_backend.py \
  --rows 50000 --evals 100 --backend both \
  --output baseline/bench_live_signal.json
```

**What it measures:** Per-evaluation wall-clock for `debounced_state_signal_native` and
`trailing_state_signal_native` via the existing `ctypes.CDLL` loader in
`alpha_zoo/native_live_signal_backend.py`.  Measures both Python and Rust backends;
reports speedup and exact parity.

**Result:** See `baseline/bench_live_signal.json`

| Metric | Value |
|---|---|
| Backend resolved | rust |
| Rust debounced (s/eval, 50k rows) | 0.000502 |
| Rust trailing (s/eval, 50k rows) | 0.000773 |
| Rust total (s/eval) | 0.001364 |
| Python total (s/eval) | 0.2626 |
| Speedup (Rust vs Python) | 192.6× |
| Parity | exact (max_abs_diff=0.0) |

### 3b. Per-Single-Tick Signal Latency (Rust ctypes, 10k iters)

**Harness:** `scripts/measure_tick_latency.py`

**Command:**
```bash
uv run python scripts/measure_tick_latency.py --iters 10000 \
  --output baseline/bench_tick_latency.json
```

**What it measures:** Per-tick (single-bar array of length 1) call latency for
`evaluate_debounced_state_native` + `evaluate_trailing_state_native`, 10,000 iterations.
Distribution (median/p95/p99) reflects kernel + ctypes FFI overhead per tick.

| Metric | Value |
|---|---|
| Backend | rust |
| Debounced median | 0.0132 ms |
| Debounced p95 | 0.0226 ms |
| Trailing median | 0.0167 ms |
| Trailing p95 | 0.0388 ms |
| Combined median | 0.0305 ms |
| Combined p95 | 0.0731 ms |
| Combined p99 | 0.1268 ms |

### 3c. Paper Order Path Latency (OrderStateMachine, no I/O)

**What it measures:** `OrderStateMachine.transition()` chain (SUBMITTED→ACKED→FILLED),
the innermost hot path of the paper/testnet order lifecycle.  Pure Python dict lookup,
zero I/O.  10,000 iterations.

| Metric | Value |
|---|---|
| Full chain median | 0.000854 ms |
| Full chain p95 | 0.00150 ms |
| Full chain p99 | 0.00188 ms |

**Note:** Does NOT include network, exchange serialisation, or WebSocket overhead.
Full tick→live-order latency is measured against testnet in Phase 5.

### 3d. Replay Path Proxy (ShadowLiveRunner, parity checker)

**What it measures:** `ShadowLiveRunner.run()` with a single synthetic event —
`stable_event_sort` + timestamp/sequence comparison overhead only.
**NOT signal or order latency** — this is the parity-checker inner-loop cost.

| Metric | Value |
|---|---|
| Median | 0.00128 ms |
| p95 | 0.00239 ms |
| p99 | 0.00273 ms |

---

## Axis 4 — Peak RSS Memory

**Harness:** `scripts/verify_8gb_baseline.py`

**Command:**
```bash
# Step 1: capture RSS via /usr/bin/time -v
/usr/bin/time -v uv run python scripts/benchmark_backtest.py \
  --config config.yaml --strategy MovingAverageCrossStrategy \
  --symbols BTC/USDT,ETH/USDT --seed 42 --iters 1 --warmup 0 \
  --output /tmp/rss_bench_single.json 2>&1 | tee baseline/bench_rss_time.log

# Step 2: gate check against 8 GiB cap
uv run python scripts/verify_8gb_baseline.py \
  --benchmark-json /tmp/rss_bench_single.json \
  --rss-log baseline/bench_rss_time.log \
  --rss-limit-gib 8.0 \
  --allow-missing-oom-sources --skip-dmesg \
  --output baseline/8gb_gate.json
```

**Note on psutil:** Not in uv.lock; `max_peak_rss_mb` in benchmark JSON is `null`.
`verify_8gb_baseline.py` falls back to the `/usr/bin/time -v` "Maximum resident set size"
log, which is fully supported by `_peak_rss_from_time_log()`.

**Result:** See `baseline/8gb_gate.json` and `baseline/bench_rss_time.log`

| Metric | Value |
|---|---|
| `peak_rss_mb` | 129.59 |
| RSS source | `/usr/bin/time -v` log |
| Gate (≤ 8 GiB) | **PASS** |
| OOM signatures | none |

---

## How to Regenerate

```bash
# 0. Ensure on branch refactor/overhaul with uv env active
git checkout refactor/overhaul

# 1. Generate benchmark CSV data (seed=42, 2024-01-01 start)
python -c "
import numpy; numpy.random.seed(42)
exec(open('generate_data.py').read().replace(
    'datetime(2022, 1, 1)', 'datetime(2024, 1, 1)'))
"

# 2. Axis 1 MICRO
uv run python scripts/benchmark_optimization_kernel.py --bars 50000 --evals 5000 \
  2>&1 | tee baseline/bench_optimizer_kernel.txt

# 3. Axis 2
uv run python scripts/benchmark_backtest.py --config config.yaml \
  --strategy MovingAverageCrossStrategy --symbols BTC/USDT,ETH/USDT \
  --seed 42 --iters 5 --warmup 1 --output baseline/bench_backtest.json

# 4. Axis 3 (live signal batch)
uv run python scripts/benchmark_live_signal_backend.py \
  --rows 50000 --evals 100 --backend both --output baseline/bench_live_signal.json

# 5. Axis 3 (per-tick + order path + replay proxy)
uv run python scripts/measure_tick_latency.py --iters 10000 \
  --output baseline/bench_tick_latency.json

# 6. Axis 4 (RSS)
/usr/bin/time -v uv run python scripts/benchmark_backtest.py \
  --config config.yaml --strategy MovingAverageCrossStrategy \
  --symbols BTC/USDT,ETH/USDT --seed 42 --iters 1 --warmup 0 \
  --output /tmp/rss_bench_single.json 2>&1 | tee baseline/bench_rss_time.log
uv run python scripts/verify_8gb_baseline.py \
  --benchmark-json /tmp/rss_bench_single.json \
  --rss-log baseline/bench_rss_time.log --rss-limit-gib 8.0 \
  --allow-missing-oom-sources --skip-dmesg --output baseline/8gb_gate.json

# 7. Axis 1 E2E (requires worker-2 fixtures at baseline/golden/)
#    Verify SHA-256 against PROVENANCE.json first, then run the canonical
#    9-combo x 3-fold walk-forward timing script (see bench_walkforward_e2e.json).

# 8. Assemble perf-baseline.json
uv run python scripts/assemble_perf_baseline.py
```

---

## How Phase 4 Tolerance Comparison Consumes These Files

`baseline/perf-baseline.json` is the sole denominator for Phase 4's report-and-justify gates:

- `walk_forward_e2e_elapsed_s` → walk-forward wall-clock comparison (report-and-justify)
- `backtest_median_bars_per_sec` → bars/sec comparison (report-and-justify)
- `live_signal_kernel_latency_median_ms` → tick latency comparison (report-and-justify)
- `peak_rss_mb` → 8 GiB hard cap in CI via `verify_8gb_baseline.py`

Per plan §5 gate taxonomy: if a post-refactor number does not improve vs the baseline,
a documented reason in `docs/perf/<axis>.md` is required — but the build is NOT blocked.
Only correctness gates (rtol `1e-8` golden comparison) are hard-blocking.
