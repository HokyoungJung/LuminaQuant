#!/bin/bash

set -euo pipefail

echo "---------------------------------------------------"
echo "[PRODUCTION] Starting LuminaQuant in Resilient Mode"
echo "---------------------------------------------------"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p logs .omx/tmp

# The RotatingFileHandler already captures structured logs to logs/lumina_quant.log;
# tell the app to skip the stderr console handler so it is not duplicated into the
# redirected crash.log below (audit O5).  crash.log then holds only true crash
# output (uncaught tracebacks / third-party stderr).
export LQ_DISABLE_CONSOLE_LOG=1

# Bound crash.log so it cannot grow without limit across restarts.  Keeps up to 5
# ~10 MiB backups, mirroring the RotatingFileHandler policy.
CRASH_LOG="logs/crash.log"
CRASH_LOG_MAX_BYTES=$((10 * 1024 * 1024))
CRASH_LOG_BACKUPS=5

rotate_crash_log() {
    [[ -f "$CRASH_LOG" ]] || return 0
    local size
    size=$(wc -c < "$CRASH_LOG" 2>/dev/null || echo 0)
    (( size >= CRASH_LOG_MAX_BYTES )) || return 0
    local i
    for (( i = CRASH_LOG_BACKUPS - 1; i >= 1; i-- )); do
        [[ -f "${CRASH_LOG}.${i}" ]] && mv -f "${CRASH_LOG}.${i}" "${CRASH_LOG}.$((i + 1))"
    done
    mv -f "$CRASH_LOG" "${CRASH_LOG}.1"
}

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is required but not found in PATH."
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash run_bot.sh [start_live_session.sh options]

Resilient wrapper around `uv run lq live` via scripts/ops/start_live_session.sh.
- First launch runs the full controlled startup flow
- Crash restarts skip heavy prep steps
- Graceful stop / clean exit does not restart

Common examples:
  bash run_bot.sh --dsn 'postgresql:///luminaquant'
  bash run_bot.sh --real --allow-real --dsn 'postgresql:///luminaquant'
  bash run_bot.sh --transport ws --dsn 'postgresql:///luminaquant'
EOF
    echo
    bash scripts/ops/start_live_session.sh --help
    exit 0
fi

FIRST_ATTEMPT=1
TARGET_MODE="paper"

for arg in "$@"; do
    case "$arg" in
        --real)
            TARGET_MODE="real"
            ;;
        --paper)
            TARGET_MODE="paper"
            ;;
    esac
done

while true; do
    rotate_crash_log

    LAUNCH_MARKER=".omx/tmp/run_bot_live_started.marker"
    rm -f "$LAUNCH_MARKER"

    CMD=(bash scripts/ops/start_live_session.sh --launch-marker "$LAUNCH_MARKER")
    if [[ "$FIRST_ATTEMPT" != "1" ]]; then
        CMD+=(--skip-init-schema --skip-refresh --skip-validate)
        if [[ "$TARGET_MODE" != "real" ]]; then
            CMD+=(--skip-preflight)
        fi
    fi
    if [[ "$#" -gt 0 ]]; then
        CMD+=("$@")
    fi

    echo "[INFO] Launching controlled live session at $(date)..."
    printf '[INFO] Command: '
    printf '%q ' "${CMD[@]}"
    printf '\n'

    set +e
    "${CMD[@]}" >> logs/crash.log 2>&1
    EXIT_CODE=$?
    set -e

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        echo "[INFO] Live session exited cleanly. Not restarting."
        exit 0
    fi

    if [[ "$EXIT_CODE" -eq 130 || "$EXIT_CODE" -eq 143 ]]; then
        echo "[INFO] Live session interrupted. Not restarting."
        exit "$EXIT_CODE"
    fi

    if [[ ! -f "$LAUNCH_MARKER" ]]; then
        echo "[ERROR] Live session failed before launch/preflight completed. Check logs/crash.log."
        exit "$EXIT_CODE"
    fi

    FIRST_ATTEMPT=0
    echo "[WARNING] Live session crashed after launch. Exit code: $EXIT_CODE"
    echo "[INFO] Restarting in 5 seconds... (Use stop-file or Ctrl+C to stop)"
    sleep 5
done
