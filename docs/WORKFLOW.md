# Developer Workflow

This repository uses a dual-branch strategy to maintain a **private codebase** (full research + operations) and a **public API** (core engine only).

## Repository Identity (Source of Truth)

- Public repo: `https://github.com/HokyoungJung/LuminaQuant.git` (typical folder: `LuminaQuant`)
- Private repo: `https://github.com/hoky1227/Quants-agent.git` (typical folder: `Quants-agent`)
- Package/import namespace: `lumina_quant`

## 1. Branch Structure

- **`private-main`**: The primary development branch. Contains ALL files including sensitive strategies, data, and logs.
- **`main`**: The public branch. Strict filtering removes private IP and DB build/sync helpers.

## 2. Public Removal Rules

When publishing to `main`, remove (or keep removed) DB construction/sync code:

- `lumina_quant/data_sync.py`
- `lumina_quant/data_collector.py`
- `scripts/sync_binance_ohlcv.py`
- `scripts/collect_market_data.py`
- `tests/test_data_sync.py`

Public branch policy:

- Keep DB **read-only** workflows (consume existing Postgres/Parquet/CSV artifacts).
- Do not include exchange OHLCV bootstrap/sync pipelines.
- Do not include tuned strategy-factory research metadata (candidate libraries, research runners, article-pipeline/deployment generators, or strategy-specific metadata tests).
- Keep runtime DB/data artifacts out of git.

## 3. Automation Scripts

We provide automation scripts for both Windows (PowerShell) and Mac/Linux (Bash).

### A. Syncing to Private Repository
**Goal**: Backup all your work (including secrets) to your private GitHub repo.

*   **Usage**: Switch to `private-main` first (`git checkout private-main`).
*   **Windows**:
    ```powershell
    .\sync_private.ps1
    ```
*   **Mac/Linux**:
    ```bash
    ./sync_private.sh
    ```

### B. Publishing to Public API
**Goal**: Create a conflict-free public sync PR from private source while excluding private lumina_quant/strategies/data.

*   **Windows**:
    ```powershell
    .\publish_api.ps1
    ```
*   **Mac/Linux**:
    ```bash
    chmod +x publish_api.sh  # Run once
    ./publish_api.sh
    ```

Default behavior:
- Builds a fresh branch from `origin/main` (ex: `public-sync-YYYYMMDD-HHMMSS`)
- Merges from `private/main` in staging mode
- Removes protected paths from the staged set
- Validates no sensitive paths remain
- Pushes the branch and opens a PR to `main`

Useful options:
```bash
# only push sanitized branch (skip PR creation)
./publish_api.sh --no-pr

# use local private-main instead of private/main
./publish_api.sh --source-ref private-main

# publish the currently checked-out feature branch instead of private/main
./publish_api.sh --source-ref HEAD

# create PR and enable auto-merge after CI is green
./publish_api.sh --auto-merge
```

## 4. Manual Process (If scripts fail)

If you need to do this manually:

**Sync Private:**
```bash
git checkout private-main
git merge main
git add .
git commit -m "sync"
git push private private-main:main
```

**Publish Public (PR-based):**
```bash
git checkout private-main
uv run python scripts/publish_public_pr.py --source-ref private/main

# or from a feature branch that is not merged to private/main yet
uv run python scripts/publish_public_pr.py --source-ref HEAD
```

## 5. Authentication Setup (Multiple Accounts)

If you use different GitHub accounts for public (`HokyoungJung`) and private (`hoky1227`) repos, use **Personal Access Tokens (PAT)** in the remote URL to avoid conflicts.

1.  Generate a **Classic PAT** (checking `repo` scope) for each account on GitHub.
2.  Run the following commands in your terminal (replace `<YOUR_TOKEN>`):

**Public Repo (origin):**
```powershell
git remote set-url origin https://HokyoungJung:<YOUR_TOKEN>@github.com/HokyoungJung/LuminaQuant.git
```

**Private Repo (private):**
```powershell
git remote set-url private https://hoky1227:<YOUR_TOKEN>@github.com/hoky1227/Quants-agent.git
```

*Note: This saves your token in plain text in `.git/config`. Ensure your local machine is secure.*
```

## 6. Live Promotion Gate Workflow

Recommended flow before enabling real mode:

1. Produce an Alpha Card (strategy assumptions, market, risk limits, and deployment parameters).

```bash
uv run python scripts/generate_alpha_card_template.py \
  --config config.yaml \
  --strategy RsiStrategy \
  --output reports/alpha_card_rsi_strategy.md
```

2. Run promotion gate report (soak + runtime reliability checks):

```bash
# uses promotion_gate defaults from config.yaml
uv run python scripts/generate_promotion_gate_report.py \
  --config config.yaml

# strategy-specific profile from promotion_gate.strategy_profiles
uv run python scripts/generate_promotion_gate_report.py \
  --config config.yaml \
  --strategy RsiStrategy
```

3. Persist the resolved promotion gate profile for review:

```bash
uv run python scripts/generate_promotion_gate_report.py \
  --config config.yaml \
  --strategy RsiStrategy \
  > reports/promotion_gate_rsi_strategy.json
```

## 7. Raw-First Data Pipeline Workflow

For live/backtest contract alignment, run market data in this order:

```bash
uv run python scripts/collect_binance_aggtrades_raw.py --symbols BTC/USDT,ETH/USDT --periodic --poll-seconds 2
uv run python scripts/materialize_market_windows.py --symbols BTC/USDT,ETH/USDT --timeframes 1s,1m,5m,15m,30m,1h,4h,1d --periodic --poll-seconds 5
uv run lq backtest --data-mode raw-first --data-source db --backtest-mode windowed
```

Before merging changes that touch live data files, run:

```bash
bash scripts/ci/architecture_gate_live_data.sh
bash scripts/ci/architecture_gate_market_window_contract.sh
uv run pytest -q \
  tests/test_periodic_loop_config_contract.py \
  tests/test_collector_periodic_loop_runtime.py \
  tests/test_materializer_periodic_loop_timeframe_resolution.py \
  tests/test_live_fail_fast_missing_committed_data.py \
  tests/test_live_no_empty_window_degradation.py \
  tests/test_market_window_schema_parity.py \
  tests/test_market_window_emission_parity_live_vs_backtest.py \
  tests/test_market_window_payload_flag.py \
  tests/test_market_window_payload_rollout_gates.py \
  tests/test_market_window_parity_flag_validation.py
```

Rollout gate decision workflow (parity_v2):

```bash
uv run python scripts/ci/export_market_window_gate_metrics.py --input logs/live/market_window_metrics.ndjson --output reports/live_rollout/baseline_gate_metrics.json --window-hours 24 --require-flag false
uv run python scripts/ci/export_market_window_gate_metrics.py --input logs/live/market_window_metrics.ndjson --output reports/live_rollout/canary_gate_metrics.json --window-hours 24 --require-flag true
uv run python scripts/ci/check_market_window_rollout_gates.py --baseline reports/live_rollout/baseline_gate_metrics.json --canary reports/live_rollout/canary_gate_metrics.json --max-p95-payload-bytes 131072 --max-queue-lag-increase-pct 5 --max-fail-fast-incidents 0
```
