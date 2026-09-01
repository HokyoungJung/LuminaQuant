#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="src/lumina_quant"

legacy_targets=(
  "${PACKAGE_ROOT}/cli/live.py"
  "${PACKAGE_ROOT}/system_assembly.py"
  "${PACKAGE_ROOT}/live/data_materialized.py"
  "${PACKAGE_ROOT}/live/data_poll.py"
  "${PACKAGE_ROOT}/live/data_ws.py"
  "${PACKAGE_ROOT}/live/trader.py"
)

stream_targets=(
  "${PACKAGE_ROOT}/live/data_binance_live.py"
  "${PACKAGE_ROOT}/live/binance_market_stream.py"
  "${PACKAGE_ROOT}/live/binance_user_stream.py"
  "${PACKAGE_ROOT}/live/order_gateway.py"
  "${PACKAGE_ROOT}/live/order_state_machine.py"
  "${PACKAGE_ROOT}/live/order_state_projector.py"
  "${PACKAGE_ROOT}/live/recovery_reconciliation.py"
  "${PACKAGE_ROOT}/live/shadow_live_runner.py"
)

banned_common_patterns=(
  "fetch_ohlcv\\("
  "auto_collect_market_data\\("
  "from lumina_quant\\.data_sync import"
  "import lumina_quant\\.data_sync"
)
banned_legacy_extra_patterns=("fetch_trades\\(")

allow_tag="# architecture-gate: allow-live-data"
hits=0

check_patterns() {
  local target="$1"
  shift
  local patterns=("$@")
  for pattern in "${patterns[@]}"; do
    while IFS=: read -r line_no line_text; do
      [[ -z "${line_no}" ]] && continue
      if [[ "${line_text}" == *"${allow_tag}"* ]]; then
        continue
      fi
      echo "${target}:${line_no}:${pattern}"
      hits=$((hits + 1))
    done < <(grep -nE "${pattern}" "${target}" || true)
  done
}

for target in "${legacy_targets[@]}"; do
  if [[ ! -f "${target}" ]]; then
    echo "missing-target:${target}"
    hits=$((hits + 1))
    continue
  fi
  check_patterns "${target}" "${banned_common_patterns[@]}" "${banned_legacy_extra_patterns[@]}"
done

for target in "${stream_targets[@]}"; do
  [[ -f "${target}" ]] || continue
  check_patterns "${target}" "${banned_common_patterns[@]}"
done

# Critical fail-fast invariant: no synthetic empty DataFrame fallback in committed reader.
if [[ -f "${PACKAGE_ROOT}/live/data_materialized.py" ]]; then
  while IFS=: read -r line_no line_text; do
    [[ -z "${line_no}" ]] && continue
    if [[ "${line_text}" == *"${allow_tag}"* ]]; then
      continue
    fi
    echo "${PACKAGE_ROOT}/live/data_materialized.py:${line_no}:frame = pl.DataFrame\\("
    hits=$((hits + 1))
  done < <(grep -nE "frame\\s*=\\s*pl\\.DataFrame\\(" "${PACKAGE_ROOT}/live/data_materialized.py" || true)
fi

if (( hits > 0 )); then
  echo "Architecture gate failed with ${hits} violation(s)." >&2
  exit 1
fi

echo "Architecture gate passed."
