# WSL Clone + Publish Commands

Use these commands from WSL to clone, sync private, and publish public safely.

## 1) Clone public repo (read-only/open)

```bash
git clone https://github.com/HokyoungJung/LuminaQuant.git
cd LuminaQuant
```

## 2) Clone private repo (requires access token)

```bash
git clone https://github.com/hoky1227/Quants-agent.git
cd Quants-agent
```

If HTTPS auth prompts fail in WSL, set credential helper once:

```bash
git config --global credential.helper manager-core
```

## 3) Configure dual remotes in private working copy

```bash
# inside private repo working tree
git remote -v

# expected mapping
# - private -> your private GitHub repo
# - origin  -> your public GitHub repo
```

If needed:

```bash
git remote set-url private https://github.com/hoky1227/Quants-agent.git
git remote set-url origin https://github.com/HokyoungJung/LuminaQuant.git
```

## 4) Private sync (full code, sensitive files included)

```bash
git checkout private-main
git pull --ff-only private main
git push private private-main:main
```

## 5) Public publish (sensitive paths removed)

The publish script is PR-first and enforces filtering for strategies/indicators/data/logs/secrets and DB build/sync helpers.

```bash
git checkout private-main
./publish_api.sh
```

This creates a sanitized `public-sync-*` branch from `origin/main`, pushes it, then opens a PR.

Optional:

```bash
./publish_api.sh --auto-merge
```

## 6) Verify public branch does not contain sensitive files

```bash
git fetch origin
git ls-tree -r --name-only origin/main | rg "^strategies/|^lumina_quant/indicators/|^data/|^logs/|^reports/|^best_optimized_parameters/|^\.omx/|^\.sisyphus/|^AGENTS\.md$|^\.env$|^lumina_quant/data_sync\.py$|^lumina_quant/data_collector\.py$|^scripts/sync_binance_ohlcv\.py$|^scripts/collect_market_data\.py$|^scripts/collect_universe_1s\.py$|^tests/test_data_sync\.py$"
```

No output from the command above means filtering worked as intended.
