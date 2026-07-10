# Dashboard Web

The primary React/Next.js dashboard runtime for LuminaQuant. Launch it with
`uv run lq dashboard --run` from the repository root (or `npm run dev` here for
frontend-only work).

## What it serves

Thirteen operator pages, each backed by a Python payload service through the
`/api/python/dashboard/*` bridge routes:

- `/` — overview: full-run headline metrics, equity/drawdown charts, recent runs and jobs
- `/performance-price` — equity vs. benchmark, drawdown, funding, and trade markers per run
- `/market-data` — OHLCV bars, indicator readings, and market context per run and symbol
- `/optimization-insights` — optimization candidate quality, stage medians, best parameters
- `/exact-window` — latest exact-window research bundle summary and portfolio snapshot
- `/factor-insights` — factor IC decay heatmap and the candidate review queue
- `/alpha-evidence` — alpha classification evidence, run cards, live-readiness verdict
- `/execution-analytics` — fill quality, closed-trade outcomes, streaks, order status
- `/risk-health` — risk events, heartbeats, and order-state changes
- `/workflows` — managed job queue with polling plus two-step Stop/Kill controls
- `/raw-data` — row counts and capped previews of the underlying tables
- `/report-export` — JSON/Markdown snapshot exports of the current run state
- `/system` — runtime reference: data-source routes, launcher, memory budget

## UI conventions

- Pages share `PageContextBar` (as-of/run/status/refresh), `RunSelector`
  (and a symbol selector on market data), `SurfaceState` (actionable
  empty/error guidance), and the SVG `TimeSeriesChart` with legend, ticks,
  and hover tooltip.
- Percent-like payload values travel as raw fractions (`0.05` = 5%); the
  frontend multiplies by 100 at render time via `lib/format.ts`.
- Performance/summary metrics are computed backend-side from the FULL equity
  series of the selected run; payload curves are downsampled and the
  `equity_window` field disclosed in the UI reports the true metric window.
- Destructive job actions (Stop/Kill) use a two-step `ConfirmButton` and the
  token-gated control route (`LQ_DASHBOARD_CONTROL_TOKEN`).

## Commands

```bash
npm install
npm run lint
npm run test
npm run typecheck
npm run build
```

## Compatibility notes

- The retired legacy entry stub remains `src/lumina_quant/dashboard/retired_stub.py` only to direct operators to the Next launcher
- Exact-window research still runs on the Python side; the Next route reads the latest exported artifact bundle without re-running heavy jobs
- The runtime stays Python-contract-backed and bounded (row caps, curve downsampling) to stay safe on the 8GB baseline
- Use Node 20+ locally and in CI so the dashboard runtime matches the supported Next.js toolchain
