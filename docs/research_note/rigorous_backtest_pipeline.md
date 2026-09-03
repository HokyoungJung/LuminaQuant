# Rigorous backtest pipeline

## One supported flow

`run_rigorous_backtest_pipeline.py` is the strategy-agnostic orchestration entry point.
It owns stage ordering, path safety, memory limits, logs, content-addressed receipts,
and restart behavior. It never copies market data and it never enables order routing.

1. `coarse_screen` — registry-wide 1-minute filter via `run_strategy_screen.py`.
2. `event_driven_walk_forward` — the frozen screen shortlist is evaluated in
   validation and locked-OOS cells by `run_event_driven_walkforward.py`.
   `run_monthly_refit_walkforward.py` remains the parameter-refit adapter, while
   manifest candidates use `run_event_driven_candidate_evaluation.py`.
3. `validation_selection` — `build_walkforward_evidence.py select` freezes finalists
   using validation cells only.
4. `portfolio_construction` — `build_walkforward_portfolios.py` compares equal
   weight, ERC, threshold HRP, dendrogram HRP, HERC and NCO on aligned validation
   returns; insufficient survivors or observations are recorded as an explicit skip.
5. `execution_model_tick_validation` — finalist execution-model feasibility against
   canonical raw aggTrades via `validate_execution_model_ticks.py`; Rust is required.
6. `report_only_evaluation` — `build_walkforward_evidence.py report` attaches the
   already-computed locked-OOS cells to the frozen selection. Its result cannot feed
   back into candidate or parameter selection.

Specialized research scripts are adapters, not alternate pipelines. The retired
Alpha-Max orchestrator was removed; its exact prelock/historical programs can still
be invoked as adapters when a frozen Alpha-Max manifest is deliberately evaluated.

## Canonical directories

- `data/market_parquet/` — the only market-data root. Read-only during backtests.
- `src/lumina_quant/strategies/registry.py` — the only strategy discovery surface.
- `src/lumina_quant/backtesting/` — reusable execution, tick validation and pipeline
  runtime code.
- `scripts/research/` — thin CLIs and research adapters.
- `var/backtests/<run-id>/` — logs, stage receipts, checkpoints and generated results.
- `docs/research_note/` — durable, reviewed evidence only.

Do not create version-suffixed checkpoint trees or private copies of canonical data.
A retry uses `--resume`. A completed stage is reused only when its command, script,
git commit, explicit inputs, canonical data receipt and output hashes all match.

Official funding refreshes live in
`data/market_parquet/funding_settlements/` as immutable sparse overlays. The loader
merges settlement-only fields over the base feature row at the same nominal
timestamp while preserving the source timestamp; OHLCV and unrelated feature
columns are not copied or replaced. `refresh_funding_settlements.py` records API
receipts and rejects conflicting outputs.

## Resource contract

- Use `uv sync --frozen --extra native`; the CPU Rust/PyO3 backend is built
  reproducibly into the uv environment.
- The plan records one `memory_max_bytes` value, capped at 7 GiB. Production
  execution runs in a systemd cgroup with matching `MemoryMax`; Python
  `RLIMIT_AS` is deliberately not used because mapped Rust/Python libraries
  consume virtual address space without consuming equivalent resident memory.
- Every stage runs in a fresh child process, so Python/Polars allocations are returned
  between stages.
- Stage inputs are hashed with streaming 1 MiB reads. The canonical database is bound
  through its audit receipt instead of rescanning or copying the payload per stage.
- `LQ_RAW_FIRST_BACKEND=rust` is the supported raw aggTrades/1-second path.
- CUDA is optional and not required by this pipeline.

## Plan contract

Plans use schema `lumina_quant.rigorous_backtest_pipeline.v1`. Paths may be repository
relative. `${REPOSITORY}`, `${DATA_ROOT}`, and `${RUN_ROOT}` placeholders are expanded
without strategy-specific constants. Each stage declares its script, arguments,
immutable input files, expected output files and only the permitted `LQ_CONFIG_PATH`
or `LQ_RAW_FIRST_BACKEND` environment values.

Safety validation rejects symlinked data/scripts, scripts outside `scripts/`, output
outside `var/backtests/<run-id>`, output inside the canonical DB, routing-enabled
plans, wrong stage order, and memory limits above 7 GiB.

```bash
uv sync --frozen --extra native
uv run python scripts/research/run_rigorous_backtest_pipeline.py \
  --plan configs/research/rigorous_backtest_plan.json

# After an interruption, with the same plan and immutable inputs:
uv run python scripts/research/run_rigorous_backtest_pipeline.py \
  --plan configs/research/rigorous_backtest_plan.json --resume
```

A tick verdict validates the execution model against independent tape prints. It does
not by itself prove strategy profitability. Performance authority comes from the
walk-forward and frozen report-only stages; execution feasibility comes from the tick
stage. Both are required for a rigorous result.
