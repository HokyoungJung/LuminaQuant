#!/usr/bin/env bash
set -euo pipefail

targets=(
  "src/lumina_quant/live/data_materialized.py"
  "src/lumina_quant/live/market_window_rolling.py"
  "src/lumina_quant/backtesting/data_windowed_parquet.py"
)

hits=0
for target in "${targets[@]}"; do
  if [[ ! -f "${target}" ]]; then
    echo "missing-target:${target}"
    hits=$((hits + 1))
    continue
  fi

  while IFS=: read -r line_no _line_text; do
    [[ -z "${line_no}" ]] && continue
    echo "${target}:${line_no}:MarketWindowEvent constructor forbidden; use build_market_window_event"
    hits=$((hits + 1))
  done < <(grep -nE "MarketWindowEvent\\(" "${target}" || true)

  if ! grep -q "build_market_window_event(" "${target}"; then
    echo "${target}:1:missing required build_market_window_event usage"
    hits=$((hits + 1))
  fi
done

if (( hits > 0 )); then
  echo "Architecture gate (market-window parity contract) failed with ${hits} violation(s)." >&2
  exit 1
fi

echo "Architecture gate (market-window parity contract) passed."
