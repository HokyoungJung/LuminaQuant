[Korean Version (한국어 가이드)](README_KR.md)

# LuminaQuant

**LuminaQuant** is a high-performance, config-driven quantitative trading engine for professional backtesting, walk-forward optimization, and live trading. The post-overhaul architecture (Phase 4–7) replaces the legacy event-loop with Rust-accelerated kernels, a unified execution model, and a Next.js 15 dashboard — while keeping a single `config.yaml` as the user-facing control surface.

## Repository Role (Source of Truth)

- **Private source-of-truth repo** (maintainers/internal): `https://github.com/hoky1227/Quants-agent.git`
- **Public distribution repo** (external/read-only subset): `https://github.com/HokyoungJung/LuminaQuant.git`
- Python package/import namespace: `lumina_quant` (distribution name: `lumina-quant`)

---

## Quick Start

### Prerequisites

| Requirement | Notes |
| :--- | :--- |
| Python >=3.14 | Managed via `uv` |
| [uv](https://docs.astral.sh/uv/) | Dependency and runtime management |
| ta-lib system library | `apt install libta-lib-dev` or equivalent |
| Rust + maturin | Required to build `native/lumina_compute` pyo3 extension |
| Node 20+ | Dashboard frontend only |

### Install

```bash
# Private repository (this repo)
git clone https://github.com/hoky1227/Quants-agent.git
cd Quants-agent

# Public mirror (LuminaQuant)
git clone https://github.com/HokyoungJung/LuminaQuant.git
cd LuminaQuant

# Core + most extras
uv sync --extra optimize --extra live-binance --extra dashboard --extra dev

# Build the Rust pyo3 extension (required for backtest/optimize/live)
python scripts/build_native_backends.py

# Dashboard frontend (once)
cd apps/dashboard_web && npm install && cd ../..

# Optional: GPU runtime (Linux x86_64 + CUDA 12 only)
# Key pins: polars>=1.35.2, GPU engine cudf-polars-cu12>=26.6
uv sync --extra gpu
```

**Available extras:** `backtest` · `optimize` · `gpu` · `live-binance` · `live-mt5` · `live-polymarket` · `dashboard` · `dev`

### Smoke test (no DB, no API keys)

```bash
uv run python scripts/minimum_viable_run.py
```

### Run a backtest

```bash
uv run lq backtest
```

### Walk-forward optimization

```bash
uv run lq optimize
```

### Launch the dashboard

```bash
uv run lq dashboard --run
```

---

## CLI Reference

`uv run lq <command>` is the single supported entry point. Root compatibility shims have been removed.

| Command | Purpose |
| :--- | :--- |
| `lq backtest` | Run a backtest for the configured strategy |
| `lq optimize` | Walk-forward optimization with Optuna |
| `lq live` | Start live trading (paper/testnet by default) |
| `lq data` | Data collection and materialization helpers |
| `lq exact-window` | Exact-window evaluation against a tick-replay window |
| `lq autonomous-research` | Autonomous strategy research pipeline |
| `lq dashboard` | Manage the Next.js dashboard (`--run`, `--print-contract`) |
| `lq config show` | Print the resolved `RuntimeConfig` as JSON |
| `lq config validate` | Validate a config YAML through the full normalisation pipeline |
| `lq registry list` | Enumerate registered strategies, indicators, and portfolio optimizers |

---

## Configuration

All user-facing knobs live in **`config.yaml`** (root) and the active profile from **`configs/profiles/{paper,real,research}.yaml`**. You should rarely need to touch source code.

Key config sections and knobs:

```yaml
# Exchange / driver selection
live:
  exchange:
    driver: "binance_futures"   # binance_futures | mt5 | polymarket

# Symbols and data kinds
trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
data:
  kinds: [ohlcv, funding, feature_points]  # ohlcv | funding | feature_points | aggtrades_tick

# Strategy selection
optimization:
  strategy: "RsiStrategy"       # class name from plugin registry

# Memory budget and golden-regression tolerance
memory:
  cap_gb: 8.0
validation:
  golden_rtol: 1.0e-8

# Live trading safety pipeline
live:
  go_live_stage: "testnet"      # testnet | shadow | canary | full
  kill_switch_enabled: true     # ALWAYS-ON — setting false is rejected at load time
  canary_position_fraction: 0.10
```

See [`AGENTS.md`](AGENTS.md) and [`docs/CONFIG_SPEC.md`](docs/CONFIG_SPEC.md) for the full schema.

---

## Architecture

### Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python >=3.14 |
| Package / runtime | uv |
| Native acceleration | Rust pyo3 extension `native/lumina_compute` (maturin) |
| Compute | Polars Lazy + optional GPU (cudf-polars) |
| Storage | Parquet (ZSTD, exchange/symbol/date partitions) + PostgreSQL audit |
| Dashboard | Next.js 15 (`apps/dashboard_web`) |

### Rust native extension — `native/lumina_compute`

A single pyo3 cdylib (`lumina_quant._compute`) built via `maturin`. It exposes 7 kernels that replace five former ctypes loaders:

| Kernel | Purpose |
| :--- | :--- |
| `evaluate_metrics` | Sharpe, Sortino, MDD, etc. |
| `simulate_symbol_fold` | Full fold vectorised backtest (inner loop) |
| `debounced_state_signal` | Live signal state machine |
| `trailing_state_signal` | Live trailing-stop state machine |
| `evaluate_hybrid_optuna_portfolio` | Optuna hybrid portfolio evaluator |
| `aggregate_raw_aggtrades_to_1s` | Raw aggTrades → 1s OHLCV |
| `append_ohlcv_1s_wal` | WAL append for 1s market data |

Build: `python scripts/build_native_backends.py` (runs `maturin develop --release`).

### Repo layout

```
LuminaQuant/
├── config.yaml                  ← single user config
├── configs/profiles/            ← paper.yaml / real.yaml / research.yaml
├── pyproject.toml
├── src/lumina_quant/
│   ├── cli/                     ← lq entry point
│   ├── configuration/           ← RuntimeConfig schema + validation
│   ├── core/                    ← engine, events, plugin_registry
│   ├── compute/                 ← Python wrappers for _compute kernels
│   ├── data/                    ← DataCollector, loaders
│   ├── storage/                 ← Parquet, PostgreSQL, WAL
│   ├── exchanges/               ← Binance futures, MT5, Polymarket adapters
│   ├── backtesting/             ← ExecutionModel, backtest engine
│   ├── optimization/            ← walk-forward, Optuna, search_policy
│   ├── live/                    ← LiveTrader, readiness checks, paper exchange
│   ├── strategies/              ← strategy registry + built-in strategies
│   ├── indicators/              ← indicator registry (alpha101 etc.)
│   ├── portfolio/               ← portfolio optimizers
│   ├── dashboard/               ← bridge contract, backend services
│   └── workflows/               ← research / autonomous pipelines
├── native/lumina_compute/       ← Rust pyo3 cdylib (maturin)
├── apps/dashboard_web/          ← Next.js 15 dashboard
├── baseline/                    ← frozen perf baseline artifacts
├── docs/perf/                   ← phase benchmark results
├── docs/divergences/            ← design decision records
└── scripts/                     ← ci/, ops/, dev/, research/
```

### Backtest rigor

The backtest engine uses a **unified `ExecutionModel`** (`backtesting/execution_model.py`) shared by both simulated backtest and live execution:

- Fees (maker/taker), funding-rate payments, leverage, liquidation threshold
- Slippage model, partial fills with per-bar volume cap
- LMT strict-cross fill rule (BUY fills IFF `bar_low < limit_price`; SELL IFF `bar_high > limit_price`)
- Golden regression at `rtol=1e-8` (`validation.golden_rtol`) enforced by CI
- Tick-replay validator (`TickReplayValidator`) for fill-price parity
- Walk-forward with configurable fold count and warmup period

### Cost realism & edge re-measurement

Headline backtest numbers are produced under **optimistic defaults**: flat, size-blind
slippage and zero funding. Several realism controls ship **config-gated OFF** (so the golden
regression stays byte-identical) and should be enabled on the backtest machine to measure how
much edge survives realistic execution:

```yaml
execution:
  slippage_impact_model: "sqrt_impact"   # size/impact-aware slippage (default "flat")
  slippage_impact_coefficient: 0.10      # impact strength (calibrate)
  require_funding_coverage: true         # fail loudly if leveraged + funding data missing
risk:
  allow_metadata_risk_override: false    # clamp metadata to config caps (already default)
  attach_default_protective_stop: true   # no naked positions
  enforce_order_risk_gate_in_backtest: true  # same RiskManager gate as live
```

Then re-run `lq backtest` / `lq optimize` and A/B against the flat baseline (plus a
10/15/20 bps cost-stress grid). Full protocol: **[`docs/COST_REALISM_REMEASUREMENT.md`](docs/COST_REALISM_REMEASUREMENT.md)**.

### Performance

Measured on the Phase 4 refactor tree vs the Phase 0 pure-Python baseline (source: [`docs/perf/phase4-results.md`](docs/perf/phase4-results.md)):

| Axis | Baseline | Phase 4 | Speedup |
| :--- | :--- | :--- | :--- |
| Backtest bars/sec (RsiStrategy, 1 268 bars, 14 symbols) | 22.44 | 6 632 | **295×** |
| Walk-forward E2E (27 runs: 3 folds × 9 combos) | 170.71 s | 1.768 s | **97×** |

Primary driver: `simulate_symbol_fold` Rust kernel replaces the pure-Python per-bar event loop.

---

## Dashboard

`uv run lq dashboard --run` starts the Next.js 15 frontend. The Python backend exposes a `DashboardBridgeContractV2` JSON contract consumed by 10 Next.js routes:

`/` (home) · `/performance-price` · `/risk-health` · `/optimization-insights` · `/market-data` · `/execution-analytics` · `/exact-window` · `/raw-data` · `/report-export` · `/workflows`

The workflows route provides no-code controls for backtest, optimize, and live sessions with asynchronous job management and log streaming.

---

## Live Trading Safety Model

Live trading follows a four-stage promotion pipeline gated by `live.go_live_stage`:

1. **testnet** — exchange testnet; no real funds
2. **shadow** — live market data, simulated orders
3. **canary** — small real position fraction (`canary_position_fraction`, default 10%)
4. **full** — full position sizing

**Kill switch is always on.** Setting `kill_switch_enabled: false` in config is structurally rejected at load time.

Real-money mode is gated by `enforce_live_readiness_from_files` (`live/readiness_policy.py`), which fail-closes unless **all** of the following hold (it does not itself measure fill/slippage/BBO parity — those must be attested in a referenced artifact):

- the `LUMINA_ENABLE_LIVE_REAL` environment variable is set (with `live.require_real_enable_flag: true`);
- a **completed, non-stale** portfolio-validation refresh artifact (freshness is judged by its `collection_cutoff_utc` age against `readiness_preflight_stale_minutes`, default 30);
- a live-readiness **decision** artifact (`keep_incumbent` or a `promote_candidate`/`selected_live_mode` whose strategy is runtime-compatible with the live registry map);
- a **referenced** readiness artifact (not the decision JSON's own hand-typed flags) that positively asserts `ready_for_real`/`real_execution_allowed`/`real_money_execution` and carries no paper-only/governance blocker — a decision cannot self-attest into real money;
- a Postgres DSN is configured.

The `full` (100% sizing) stage additionally requires **recorded canary evidence** (`canary_execution_recorded`) or an explicit `LUMINA_ALLOW_FULL_WITHOUT_CANARY` override, so full sizing can never be the first reachable real stage. Use `scripts/ops/write_real_money_attestation.py` to emit the referenced attestation artifact — it refuses to set any positive flag without embedded, on-disk, verified evidence references.

```bash
# Paper/testnet (default)
uv run lq live

# Real mode (requires env var + artifacts)
LUMINA_ENABLE_LIVE_REAL=true uv run lq live --enable-live-real
```

See [`docs/live-readiness/04-paper-trading-runbook.md`](docs/live-readiness/04-paper-trading-runbook.md) for the operator checklist.

---

## Adding Strategies, Indicators, and Portfolio Optimizers

The plugin system uses `@register` decorators from `lumina_quant.core.plugin_registry`. Full step-by-step instructions are in [`AGENTS.md`](AGENTS.md). Summary:

1. Create `src/lumina_quant/strategies/<name>.py` with a class implementing the strategy interface.
2. Decorate with `@register("strategy", "ClassName", interface="event_driven"|"polars_batch")`.
3. Add an import entry in `src/lumina_quant/strategies/registry.py`.
4. Add a param schema entry in `src/lumina_quant/tuning/param_registry.py`.
5. Set `optimization.strategy: "ClassName"` in `config.yaml` to activate.

The same `@register` pattern applies for indicators (`"indicator"`) and portfolio optimizers (`"portfolio"`). See [`AGENTS.md`](AGENTS.md) for details.

---

## Documentation Index

| Document | Description |
| :--- | :--- |
| [`AGENTS.md`](AGENTS.md) | Architecture notes, ownership map, how-to guides |
| [`docs/CONFIG_SPEC.md`](docs/CONFIG_SPEC.md) | Full RuntimeConfig schema reference |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Deployment notes and operational checklist |
| [`docs/live-readiness/04-paper-trading-runbook.md`](docs/live-readiness/04-paper-trading-runbook.md) | Paper/testnet live handoff runbook |
| [`docs/perf/phase4-results.md`](docs/perf/phase4-results.md) | Phase 4 benchmark results |
| [`docs/EXCHANGES.md`](docs/EXCHANGES.md) | Binance USDⓈ-M Futures, MetaTrader 5, Polymarket setup |
| [`docs/EXTERNAL_DATA.md`](docs/EXTERNAL_DATA.md) | Canonical contracts for user-managed data |
| [`docs/METRICS.md`](docs/METRICS.md) | Sharpe, Sortino, Alpha, Beta, Calmar definitions |
| [`docs/COST_REALISM_REMEASUREMENT.md`](docs/COST_REALISM_REMEASUREMENT.md) | Enable realistic slippage/funding/risk flags and re-measure edge survival |
| [`docs/RUST_NATIVE_ACCELERATION.md`](docs/RUST_NATIVE_ACCELERATION.md) | Hotspot-only Rust policy, build commands, benchmarks |
| [`docs/QUICKSTART_8GB_BASELINE.md`](docs/QUICKSTART_8GB_BASELINE.md) | 8 GB RAM minimal install and smoke flow |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Local checks, CI parity commands, PR expectations |
| [`SECURITY.md`](SECURITY.md) | Vulnerability reporting and credential policy |

---

## Environment Variables

Never commit API keys. Create a `.env` file at the repository root (see `.env.example`):

```ini
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
LQ_POSTGRES_DSN=postgresql://localhost:5432/luminaquant
```

PostgreSQL is optional for backtesting — if `LQ_POSTGRES_DSN` is unset, audit persistence is skipped.

---

## License & Disclaimer

This software is for research and educational purposes. Past backtest performance does not guarantee future results. Real-money trading carries substantial risk of loss. The kill-switch and go_live_stage pipeline exist to slow down promotion, not to eliminate risk. Maintainers provide no warranty of any kind.
