# Minimal Install Profiles

LuminaQuant now supports persona-oriented extras so users can install only what they need.

## Recommended profiles

### Backtest-only
```bash
uv sync --group dev --extra backtest
```
Use when you only need historical backtests with canonical CSV/parquet OHLCV data.

### Optimize
```bash
uv sync --group dev --extra backtest --extra optimize
```
Use when you need walk-forward/Optuna optimization on top of backtest support.

### Live Binance
```bash
uv sync --group dev --extra live-binance
```
Includes `websockets` for Binance Futures live streams; REST execution/data sync use the native stdlib client.

### Live MT5
```bash
uv sync --group dev --extra live-mt5
```
Use on Windows or with MT5 bridge mode on WSL/Linux.

### Live Polymarket (Phase 1)
```bash
uv sync --group dev --extra live-polymarket
```
This lane supports market data, paper/shadow workflows, and experimental real execution paths when Polymarket credentials/private key are configured and `allow_real_execution` is explicitly enabled.

### Dashboard
```bash
uv sync --group dev --extra dashboard
cd apps/dashboard_web && npm install
```
Use when you need the Python dashboard helpers plus the primary Next.js dashboard runtime (Node 20+).

### Full local maintainer setup
```bash
uv sync --group dev --extra backtest --extra optimize --extra live-binance --extra live-mt5 --extra live-polymarket --extra dashboard
cd apps/dashboard_web && npm install
```

## Compatibility aliases
- `live` remains a compatibility alias for the legacy live install path.
- `all` remains a convenience alias for broad local development installs.

## Verification suggestions
- Backtest-only: `uv run lq backtest --help`
- Optimize: `uv run lq optimize --help`
- Live Binance: `uv run lq live --help`
- Live MT5: `uv run python -m pytest tests/test_mt5_exchange.py`
- Live Polymarket: `uv run python -m pytest tests/test_polymarket_exchange.py`
