#!/bin/bash

set -euo pipefail

cleanup_main_branch() {
  git reset --hard HEAD >/dev/null 2>&1 || true
  git clean -fd >/dev/null 2>&1 || true
  git checkout private-main >/dev/null 2>&1 || true
}

PROTECTED_PATHS=(
  "AGENTS.md"
  ".env"
  ".omx"
  ".sisyphus"
  "data"
  "logs"
  "reports"
  "best_optimized_parameters"
  "equity.csv"
  "trades.csv"
  "live_equity.csv"
  "live_trades.csv"
  "strategies"
  "lumina_quant/indicators"
  "lumina_quant/data_sync.py"
  "lumina_quant/data_collector.py"
  "scripts/sync_binance_ohlcv.py"
  "scripts/collect_market_data.py"
  "scripts/collect_universe_1s.py"
  "tests/test_data_sync.py"
)

script_path=$(realpath "$0")
if [[ "$script_path" == *"LuminaQuant"* ]]; then
  temp_path="/tmp/publish_api.sh"
  echo "Copying script to temp: $temp_path"
  cp "$script_path" "$temp_path"
  chmod +x "$temp_path"
  exec "$temp_path"
fi

current_branch=$(git branch --show-current)
if [ "$current_branch" != "private-main" ]; then
  echo -e "\033[0;31mPlease run this script from the 'private-main' branch.\033[0m"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo -e "\033[0;31mYou have uncommitted changes. Please commit or stash them first.\033[0m"
  exit 1
fi

echo -e "\033[0;36mSwitching to main...\033[0m"
git checkout main

echo -e "\033[0;36mMerging changes from private-main (without committing)...\033[0m"
if ! git merge private-main --no-commit --no-ff; then
  echo -e "\033[0;33mMerge had conflicts. Preferring private-main content before filtering...\033[0m"
  git checkout --theirs -- . >/dev/null 2>&1 || true
  git add -A >/dev/null 2>&1 || true
fi

echo -e "\033[0;36mEnforcing public .gitignore...\033[0m"
git checkout HEAD -- .gitignore

echo -e "\033[0;36mPreparing staged public set...\033[0m"
git reset
git add .

echo -e "\033[0;36mRemoving protected/sensitive paths from staging...\033[0m"
for protected in "${PROTECTED_PATHS[@]}"; do
  git rm -r --cached --ignore-unmatch -- "$protected" >/dev/null 2>&1 || true
done

echo -e "\033[0;36mValidating sensitive paths are absent from staged tree...\033[0m"
if git diff --cached --name-only --diff-filter=ACMRT | rg "^strategies/|^lumina_quant/indicators/|^data/|^logs/|^reports/|^best_optimized_parameters/|^\.omx/|^\.sisyphus/|^AGENTS\.md$|^\.env$|^lumina_quant/data_sync\.py$|^lumina_quant/data_collector\.py$|^scripts/sync_binance_ohlcv\.py$|^scripts/collect_market_data\.py$|^scripts/collect_universe_1s\.py$|^tests/test_data_sync\.py$|(^|/)live_?equity\.csv$|(^|/)live_?trades\.csv$|(^|/)equity\.csv$|(^|/)trades\.csv$" >/dev/null 2>&1; then
  echo -e "\033[0;31mSensitive files are still staged. Aborting publish.\033[0m"
  cleanup_main_branch
  exit 1
fi

if git diff --cached --quiet; then
  echo -e "\033[0;33mNo public changes to publish.\033[0m"
  cleanup_main_branch
  exit 0
fi

echo -e "\033[0;36mCommitting public changes...\033[0m"
git commit -m "chore: publish updates from private repository"

echo -e "\033[0;36mPushing to origin main...\033[0m"
git push origin main

echo -e "\033[0;36mSwitching back to private-main...\033[0m"
git checkout private-main

echo -e "\033[0;32mDone! Public API published to 'main'.\033[0m"
