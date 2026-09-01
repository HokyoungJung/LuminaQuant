#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/ops/stop_live_session.sh [options]

Request graceful shutdown for a live session by touching the configured stop-file.

Options:
  --paper              Stop the default paper session (default)
  --real               Stop the default real session
  --stop-file PATH     Override stop-file path
  -h, --help           Show this help

Examples:
  scripts/ops/stop_live_session.sh
  scripts/ops/stop_live_session.sh --real
  scripts/ops/stop_live_session.sh --stop-file /tmp/custom.stop
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="paper"
STOP_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper)
      MODE="paper"
      shift
      ;;
    --real)
      MODE="real"
      shift
      ;;
    --stop-file)
      STOP_FILE="${2:?missing value for --stop-file}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$STOP_FILE" ]]; then
  STOP_FILE="/tmp/lq-${MODE}.stop"
fi

exec uv run python scripts/ops/request_live_stop.py --stop-file "$STOP_FILE"
