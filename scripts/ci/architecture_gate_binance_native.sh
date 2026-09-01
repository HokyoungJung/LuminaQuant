#!/usr/bin/env bash
set -euo pipefail

critical_targets=(
  "src/lumina_quant/configuration/loader.py"
  "src/lumina_quant/configuration/validate.py"
  "src/lumina_quant/data_collector.py"
  "src/lumina_quant/data_sync.py"
  "src/lumina_quant/exchanges/__init__.py"
  "src/lumina_quant/exchanges/binance_futures_exchange.py"
  "src/lumina_quant/live/binance_market_stream.py"
  "src/lumina_quant/live/binance_user_stream.py"
  "src/lumina_quant/live/data_binance_live.py"
  "src/lumina_quant/live/data_poll.py"
  "src/lumina_quant/live/data_ws.py"
  "src/lumina_quant/live/trader.py"
  "scripts/research/refresh_final_portfolio_validation_data.py"
  "scripts/research/validate_saved_incumbent_portfolio.py"
  "scripts/research/validate_saved_incumbent_portfolio_continuity.py"
)

banned_patterns=(
  "\\bccxt\\b"
  "CCXT"
  "ccxt_exchange"
  "create_binance_exchange\\("
)

hits=0
for target in "${critical_targets[@]}"; do
  if [[ ! -f "${target}" ]]; then
    echo "missing-target:${target}"
    hits=$((hits + 1))
    continue
  fi
  for pattern in "${banned_patterns[@]}"; do
    while IFS=: read -r line_no _line_text; do
      [[ -z "${line_no}" ]] && continue
      echo "${target}:${line_no}:${pattern}"
      hits=$((hits + 1))
    done < <(grep -nE "${pattern}" "${target}" || true)
  done
done

# Required native Binance Futures integration surfaces.
required_files=(
  "src/lumina_quant/exchanges/binance_futures_exchange.py"
  "src/lumina_quant/exchanges/binance_futures_client.py"
  "src/lumina_quant/eval/final_validation.py"
  "src/lumina_quant/utils/risk_free.py"
  "src/lumina_quant/data/raw_first_lineage.py"
)

for target in "${required_files[@]}"; do
  if [[ ! -f "${target}" ]]; then
    echo "missing-required-native-file:${target}"
    hits=$((hits + 1))
  fi
done

if (( hits > 0 )); then
  echo "Architecture gate (native Binance Futures / no-CCXT critical path) failed with ${hits} violation(s)." >&2
  exit 1
fi

echo "Architecture gate (native Binance Futures / no-CCXT critical path) passed."
