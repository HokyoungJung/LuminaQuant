#!/bin/bash

# Production Startup Script for macOS/Linux
# Uses uv-only runtime and auto-restarts on crash.

echo "---------------------------------------------------"
echo "[PRODUCTION] Starting LuminaQuant in Resilient Mode"
echo "---------------------------------------------------"

# Ensure log directory exists
mkdir -p logs

while true; do
    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] uv is required but not found in PATH."
        exit 1
    fi

    echo "[INFO] Launching Live Trader at $(date)..."
    uv run python run_live.py >> logs/crash.log 2>&1
    EXIT_CODE=$?

    echo "[WARNING] Bot crashed or stopped! Exit Code: $EXIT_CODE"
    echo "[INFO] Restarting in 5 seconds... (Press Ctrl+C to stop)"
    sleep 5
done
