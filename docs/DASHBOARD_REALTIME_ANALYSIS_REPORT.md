# Dashboard Realtime Analysis Report

## Scope

- Target: `dashboard.py` end-to-end observability and realtime visibility.
- Goal: Let users understand strategy, price, execution, risk, health, and report outputs directly from the dashboard.
- Constraints: preserve existing SQLite/CSV compatibility and keep dashboard responsive under frequent refresh.

## Findings (Before)

### 1) Realtime trigger and freshness visibility were insufficient

- Dashboard refresh path was not explicit enough for continuous live monitoring.
- Result: users could misread stale views as "not updating" despite live engine writes.

### 2) Observability breadth was too narrow

- Previous view emphasized a subset of performance charts only.
- Missing from main workflows: order lifecycle, risk events, heartbeat health, order-state transitions, and dedicated market OHLCV section.

### 3) Trade marker context lacked actionable details

- Buy/Sell markers showed timing/price, but not position aftermath or trade-level realized outcome.
- Users could not inspect position size progression, per-trade realized PnL, or trade return directly on marker hover.

### 4) Performance risk for heavy windows

- Frequent refresh with large series can degrade chart responsiveness if all points are re-rendered each tick.

## Changes Implemented

### A) Realtime refresh and source-selection robustness

- Added `Auto Refresh` toggle + interval control.
- Added refresh mechanism with fallback:
  - Primary: `streamlit_autorefresh`
  - Fallback: meta-refresh based rerun
- Added run auto-selection logic favoring RUNNING/equity-present runs and CSV fallback when SQLite run has no equity rows yet.

### B) Full observability data model in dashboard

- Added/expanded loaders for:
  - `runs`
  - `equity`
  - `fills`
  - `orders`
  - `risk_events`
  - `heartbeats`
  - `order_state_events`
  - `market_ohlcv`
- Preserved CSV fallback for `equity.csv` and `trades.csv`.

### C) Trade analytics and marker-level context

- Added trade enrichment logic:
  - position after fill
  - average cost after fill
  - realized PnL (closing portion)
  - realized return (%)
  - cumulative realized PnL
- Buy/Sell markers now include these fields in hover payload for direct investigation.

### D) Expanded visual architecture (all critical information)

- Dashboard now provides dedicated tabs:
  1. Performance & Price
  2. Execution Analytics
  3. Risk & Health
  4. Market Data
  5. Report Export
  6. Raw Data
- Included execution, risk, health, and market panels so users can inspect the full lifecycle in one place.

### E) Large-window responsiveness

- Added `Auto Downsample Plots` + `Downsample Target Points`.
- Charts can render reduced points while metrics/raw tables keep full loaded window.

### F) Built-in report export

- Added snapshot report payload generation from dashboard state.
- Export outputs:
  - JSON report
  - Markdown report
- Includes period, fill cadence, PnL, return, risk/heartbeat row counts.

## Risk Assessment

### Low-risk characteristics

- No schema changes.
- No changes to live engine write path or SQLite schema.
- No removal of old CSV/SQLite modes.
- Additive UI controls with safe defaults.

### Known caveats

- If no run has equity rows yet, dashboard can fallback to CSV in Auto mode.
- Realized PnL logic is inventory-based approximation from fill stream; exchange-grade accounting may differ on advanced position modes.

## Validation Summary

- `uv run ruff check .` passed.
- `uv run pytest -q` passed.
- `uv run python -m streamlit run dashboard.py --server.headless true` startup confirmed.
- `uv run python scripts/smoke_dashboard_realtime.py --db-path logs/lumina_quant.db --dry-run` passed.

## Completed Follow-ups

1. Added `streamlit-autorefresh` to dashboard optional dependency set (`pyproject.toml`).
2. Added dashboard realtime smoke script (`scripts/smoke_dashboard_realtime.py`).
3. Added full execution/risk/health/market/report tabs.
4. Added marker hover details: position size, realized PnL, trade return.
5. Added report snapshot export (JSON/Markdown).
6. Added no-code workflow control jobs table (`workflow_jobs`) for async backtest/optimize/live launches.
7. Added live control arming UX for real mode (`ENABLE REAL` phrase + dual acknowledgments).
8. Added graceful stop control using stop-file signaling and emergency kill fallback.
9. Added optimization insights panel reading SQLite `optimization_results`.
10. Added ghost cleanup utility (`scripts/cleanup_ghost_runs.py`) and dashboard dry-run/apply controls.
