# Developer Workflow

This repository uses a dual-branch strategy to maintain a **private codebase** (full research + operations) and a **public API** (core engine only).

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
**Goal**: Create a conflict-free public sync PR from private source while excluding private strategies/data.

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
