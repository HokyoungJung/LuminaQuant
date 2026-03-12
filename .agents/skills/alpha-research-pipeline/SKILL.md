---
name: alpha-research-pipeline
description: Build and refresh the article-inspired LuminaQuant alpha research pipeline manifest, strategy-family thesis map, and memory-safe backtest operating rules.
---

# Alpha Research Pipeline

Use this skill when the user wants to:
- turn external research (articles, screenshots, notes) into new LuminaQuant strategy hypotheses,
- refresh the reusable research pipeline / thesis manifest,
- keep exact-window backtests memory-safe and non-duplicative,
- or summarize which strategy families and metrics should be expanded next.

## Workflow

1. Read the latest article/screenshot notes if provided.
2. Generate or refresh the pipeline manifest:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv run python scripts/research/run_llm_alpha_pipeline.py
```

3. Use the generated artifacts as the source of truth:
   - `var/reports/exact_window_backtests/pipeline/alpha_research_pipeline_latest.json`
   - `var/reports/exact_window_backtests/pipeline/alpha_research_pipeline_latest.md`
4. Treat the manifest as a planning / orchestration layer only:
   - hypothesis discovery can be LLM-assisted,
   - execution and backtests remain rule-based and reproducible,
   - heavy jobs stay serialized under the total memory cap.
5. Before launching new heavy runs, check the canonical signature registry first:
   - `var/reports/exact_window_backtests/exact_window_run_registry.jsonl`
   - `var/reports/exact_window_backtests/exact_window_backtest_registry_latest.json`
   - if runtime artifacts were cleaned, also review the recovered archive:
     `var/reports/exact_window_backtests/followup_status/backtest_log_archive_latest.json`

## Operating Rules

- Total memory cap is **global across all active sessions/services/workers**, not per process.
- Keep **one heavy backtest run at a time**.
- Treat recovered log archives as advisory history, not as the canonical duplicate-signature registry.
- Always attach rich metrics: return, CAGR, Sharpe, Sortino, Calmar, max DD, volatility, trades, turnover, win rate, avg trade, exposure, deflated Sharpe, PBO, SPA p-value, benchmark corr, rolling Sharpe floor, stability, and peak RSS.
- Prefer saved artifacts over recomputation for dashboard/deployment panels.
