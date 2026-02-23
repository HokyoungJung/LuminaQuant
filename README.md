[Korean Version (한국어 가이드)](README_KR.md)

# LuminaQuant Documentation

**LuminaQuant** is an advanced, event-driven quantitative trading system designed for professional-grade backtesting and live trading. It features a modular architecture that supports multiple exchanges, robust state management, and sophisticated strategy optimization.

---

## 📚 Documentation Index

| Section | Description |
| :--- | :--- |
| **[Installation & Setup](#installation)** | Getting started with LuminaQuant. |
| **[Deployment Guide](docs/DEPLOYMENT.md)** | **New**: Docker & VPS Setup for 24/7 Trading. |
| **[Validation Report](docs/VALIDATION_REPORT.md)** | Verification + optimization report for core workflows. |
| **[Workflow Guide](docs/WORKFLOW.md)** | Private/Public branch operation and publish checklist. |
| **[Dashboard Realtime Report](docs/DASHBOARD_REALTIME_ANALYSIS_REPORT.md)** | Analysis + implementation report for live-refresh dashboard behavior. |
| **[Exchange Guide](docs/EXCHANGES.md)** | Detailed setup for **Binance** (CCXT) and **MetaTrader 5**. |
| **[Trading Manual](docs/TRADING_MANUAL.md)** | **How-To**: Buy/Sell, Leverage, TP/SL, Trailing Stops. |
| **[Performance Metrics](docs/METRICS.md)** | Explanation of Sharpe, Sortino, Alpha, Beta, etc. |
| **[Developer API](docs/API.md)** | How to write Strategies and extend the system. |
| **[Configuration](#configuration)** | Quick reference for `config.yaml`. |

## 🏗 Architecture

LuminaQuant follows a modular **Event-Driven Architecture**:

```mermaid
graph TD
    Data[Data Handler] -->|MarketEvent| Engine[Trading Engine]
    Engine -->|MarketEvent| Strategy[Strategy]
    Strategy -->|SignalEvent| Portfolio[Portfolio]
    Portfolio -->|OrderEvent| Execution[Execution Handler]
    Execution -->|FillEvent| Portfolio
```

- **DataHandler**: Manages historical (CSV) or live (WebSocket) data feeds.
- **Strategy**: Generates `SignalEvent` based on market data (e.g., RSI < 30).
- **Portfolio**: Manages state, positions, and risk. Converts Signals to `OrderEvent`.
- **ExecutionHandler**: Simulates fills (Backtest) or executes usage API (Live).

---

## ⚙️ Setup & Configuration

### Prerequisites
- Python 3.11 to 3.13
- [uv](https://docs.astral.sh/uv/) for dependency/runtime management
- [Polars](https://pola.rs/) (for high-performance data)
- [Talib](https://github.com/TA-Lib/ta-lib-python) (for technical indicators)

### Environment Variables
For security, **never commit API keys**. Create a `.env` file in the root directory:

```ini
# .env file
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
LOG_LEVEL=INFO
```

*See `.env.example` for a template.*

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/HokyoungJung/LuminaQuant.git
cd lumina-quant

# Ensure compatible Python (project requires < 3.14)
uv python pin 3.13

# Install dependencies
uv sync --all-extras  # or pip install ".[live,optimize,dashboard]"

# Verify install and tests
uv run python scripts/verify_install.py

# (Optional) For MT5 Support
uv sync --extra mt5
```

### 2. Configuration

LuminaQuant uses `config.yaml` for all settings.

**Generic Setup:**
```yaml
trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
  timeframe: "1h"
  initial_capital: 10000.0
```

**Choose Your Exchange:**

*   **Binance (Crypto)**: Set `driver: "ccxt"`
*   **MetaTrader 5 (Forex/Stocks)**: Set `driver: "mt5"`

*👉 See [Exchange Guide](docs/EXCHANGES.md) for detailed credentials setup.*

### Public vs Private Repository Scope

- This public repository intentionally excludes proprietary research IP:
  - `lumina_quant/indicators/`
  - `strategies/`
  - private strategy/indicator test files
- This public repository also excludes DB construction/sync code:
  - `lumina_quant/data_sync.py`
  - `lumina_quant/data_collector.py`
  - `scripts/sync_binance_ohlcv.py`
  - `scripts/collect_market_data.py`
  - `scripts/collect_universe_1s.py`
  - `tests/test_data_sync.py`
- Full strategy/indicator implementation and AGENTS guidance are maintained in the private repository.
- Database/runtime artifacts are never published (`*.db`, `*.sqlite*`, `data/`, `logs/`, `.omx/`, `.sisyphus/`).

### 3. Running the System

**(Private repo only) Sync Binance OHLCV into SQLite (and CSV mirror):**
```bash
uv run python scripts/sync_binance_ohlcv.py \
  --symbols BTC/USDT ETH/USDT \
  --timeframe 1m \
  --db-path data/lumina_quant.db \
  --force-full
```

In the public repository, DB sync/build helpers are intentionally removed. Use prebuilt market DB files or CSV data.

**Backtest a Strategy:**
```bash
uv run python run_backtest.py

# Force DB-only data source
uv run python run_backtest.py --data-source db --market-db-path data/lumina_quant.db
```

**Walk-Forward Optimization (multi-fold):**
```bash
uv run python optimize.py

# Prefer DB data, fallback to CSV in auto mode
uv run python optimize.py --data-source auto --market-db-path data/lumina_quant.db
```

**Architecture/Lint Gate:**
```bash
uv run python scripts/check_architecture.py
uv run ruff format . --check
uv run ruff check .
```

**Visualize Results:**
```bash
uv run streamlit run dashboard.py
```

Dashboard now includes no-code workflow controls for backtest, optimization, and live launch/stop with:
- asynchronous managed jobs and log tail viewer
- explicit real-mode arming phrase (`ENABLE REAL`)
- graceful stop via control-file signal and emergency force-kill fallback
- optimization results panel from SQLite (`optimization_results`)
- ghost cleanup controls (dry-run/apply) for stale `RUNNING` rows
- strategy-scoped run filtering (`Filter Run IDs By Strategy`) and automatic run reselection on strategy change
- separate `Market Data SQLite Path` so market OHLCV source can differ from audit DB path
- explicit CSV fallback warning when equity is rendered from CSV samples instead of SQLite run rows

**Ghost Cleanup CLI (stale RUNNING rows):**
```bash
# Dry-run (recommended first)
uv run python scripts/cleanup_ghost_runs.py --db data/lumina_quant.db --stale-sec 300 --startup-grace-sec 90

# Apply cleanup
uv run python scripts/cleanup_ghost_runs.py --db data/lumina_quant.db --stale-sec 300 --startup-grace-sec 90 --apply
```

**Realtime Dashboard Smoke Check (equity row growth):**
```bash
# Run while live trader is writing to data/lumina_quant.db
uv run python scripts/smoke_dashboard_realtime.py \
  --db-path data/lumina_quant.db \
  --require-running \
  --timeout-sec 90 \
  --poll-sec 3
```

**Start Live Trading:**
```bash
uv run python run_live.py
# Real mode requires explicit safety flag:
# LUMINA_ENABLE_LIVE_REAL=true uv run python run_live.py --enable-live-real
```

**Generate 14-day Soak Report (Promotion Gate):**
```bash
uv run python scripts/generate_soak_report.py --db data/lumina_quant.db --days 14
```

**Generate Promotion Gate Report (Soak + Runtime Reliability):**
```bash
# Uses defaults from promotion_gate in config.yaml
uv run python scripts/generate_promotion_gate_report.py \
  --db data/lumina_quant.db \
  --config config.yaml

# Strategy-specific profile from promotion_gate.strategy_profiles
uv run python scripts/generate_promotion_gate_report.py \
  --db data/lumina_quant.db \
  --config config.yaml \
  --strategy RsiStrategy

# Override specific threshold from CLI
uv run python scripts/generate_promotion_gate_report.py \
  --db data/lumina_quant.db \
  --strategy RsiStrategy \
  --max-order-rejects 0

# Generate Alpha Card scaffold from runtime config
uv run python scripts/generate_alpha_card_template.py \
  --config config.yaml \
  --strategy RsiStrategy \
  --output reports/alpha_card_rsi_strategy.md

# Optional Alpha Card requirement
uv run python scripts/generate_promotion_gate_report.py \
  --db data/lumina_quant.db \
  --strategy RsiStrategy \
  --alpha-card reports/alpha_card_rsi_strategy.md \
  --require-alpha-card
```

**Backtest Benchmark Baseline/Regression:**
```bash
uv run python scripts/benchmark_backtest.py --output reports/benchmarks/baseline_snapshot.json

# Compare current run vs previous snapshot
uv run python scripts/benchmark_backtest.py \
  --output reports/benchmarks/current_snapshot.json \
  --compare-to reports/benchmarks/baseline_snapshot.json
```

**Phase-1 Binance USDT-M Research Starter (sync + OOS sweep):**
```bash
# Dry run to inspect the generated sweep command
uv run python scripts/run_phase1_research.py --skip-sync --dry-run

# Full phase-1 kickoff with default liquid USDT universe
uv run python scripts/run_phase1_research.py

# Faster iteration profile
uv run python scripts/run_phase1_research.py \
  --topcap-iters 120 \
  --pair-iters 90 \
  --ensemble-iters 1200 \
  --timeframes 15m 1h
```

**Two-Book Research Starter (market-neutral alpha + trend overlay):**
```bash
# Print command only (no sweep run)
uv run python scripts/run_two_book_research.py --dry-run

# Run sweep and emit two-book selection artifact
uv run python scripts/run_two_book_research.py \
  --timeframes 15m 1h 4h \
  --alpha-risk-budget 0.8 \
  --trend-risk-budget 0.2

# Rebuild two-book selection from an existing sweep report
uv run python scripts/run_two_book_research.py \
  --dry-run \
  --sweep-report reports/timeframe_sweep_oos_YYYYMMDDTHHMMSSZ.json
```

**Strategy-Team Research Factory (many sleeves, many seeds/timeframes):**
```bash
# Preview run matrix only
uv run python scripts/run_strategy_team_research.py --dry-run

# Build a broad candidate pool and select a diversified strategy team
uv run python scripts/run_strategy_team_research.py \
  --timeframes 1s 1m 5m 15m 30m 1h 4h 1d \
  --seeds 20260220 20260221 20260222 \
  --search-engine random \
  --max-selected 32
```

**Unified Futures Data Bundle (1s OHLCV + derivatives feature points):**
```bash
# Canonical 1s base stream + required futures features
uv run python scripts/collect_futures_bundle.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT XAU/USDT XAG/USDT \
  --db-path data/lumina_quant.db \
  --since 2024-01-01T00:00:00+00:00

# Feature-only refresh (skip OHLCV)
uv run python scripts/collect_futures_bundle.py \
  --skip-ohlcv \
  --mark-index-interval 1m \
  --open-interest-period 5m
```

Collected feature points are stored in SQLite table `futures_feature_points` with:
- `funding_rate`, `funding_mark_price`
- `mark_price`, `index_price`
- `open_interest`
- `liquidation_long_qty`, `liquidation_short_qty`
- `liquidation_long_notional`, `liquidation_short_notional`

**Fast Local Scan Profiles (avoid full 20min+ runs while iterating):**
```bash
# Very fast smoke scan (default)
uv run python scripts/quick_scan.py --profile quick

# Broader targeted scan
uv run python scripts/quick_scan.py --profile standard

# Full suite + build when needed
uv run python scripts/quick_scan.py --profile full --with-build

# Preview commands only
uv run python scripts/quick_scan.py --profile quick --dry-run
```

---

## 🌟 Key Features

- **Event-Driven Core**: Simulates realistic execution by processing events (`Market`, `Signal`, `Order`, `Fill`) sequentially.
- **Multi-Asset & Multi-Exchange**:
    - Trade **Crypto** on Binance, Bybit, Upbit (via CCXT).
    - Trade **Forex, CFTs, Stocks** on MetaTrader 5.
- **Advanced Backtesting**: Includes slippage, commission models, and trailing stop logic.
- **Optimization**: Built-in Bayesian Optimization using **Optuna** to find the best strategy parameters.
- **Live Resilience**:
    - **State Recovery**: Syncs positions on restart.
    - **Circuit Breakers**: Halts trading if daily loss exceeds limits.

---

## 📊 Dashboard Preview

The included Streamlit dashboard provides professional-grade analytics:

- **Equity Curve & Drawdowns**: Visualize your portfolio growth and risk.
- **Trade Analysis**: See exactly where buys and sells occurred on the chart.
- **Comprehensive Metrics**: Sharpe Ratio, Sortino, Calmar, Alpha, Beta, etc.

*👉 See [Performance Metrics](docs/METRICS.md) for a full definition of all stats.*
