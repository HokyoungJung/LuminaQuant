#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/ops/start_live_session.sh [options] [-- <extra lq live args>]

Start one controlled LuminaQuant live session with optional preparation steps.

This is intentionally different from ./run_bot.sh:
  - run_bot.sh: simple infinite restart loop around `uv run lq live`
  - this script: one safe launch with env loading, optional schema init,
    optional refresh/validation, and fail-closed live readiness gating

Options:
  --paper                    Launch in paper mode (default)
  --real                     Launch in real mode
  --allow-real               Required with --real; also passes --enable-live-real
  --transport poll|ws        Live transport (default: poll)
  --env-file PATH            Env file to source before launch (default: .env)
  --no-env-file              Do not source an env file
  --dsn DSN                  Override LQ_POSTGRES_DSN for this run
  --run-id ID                Explicit run id (default: <mode>-<utc timestamp>)
  --stop-file PATH           Stop-file path (default: /tmp/lq-<mode>.stop)
  --selection-file PATH      Pass an explicit live selection file
  --no-selection             Disable selection artifact loading
  --strategy NAME            Override strategy class
  --skip-init-schema         Skip Postgres schema initialization
  --skip-refresh             Skip refresh_final_portfolio_validation_data.py
  --skip-validate            Skip validate_saved_incumbent_portfolio.py
  --skip-live-decision       Skip write_portfolio_live_readiness_decision.py
  --skip-preflight           Skip live_readiness_preflight.py
  --preflight-stale-minutes N  Freshness threshold for preflight (default: 30)
  --dry-run                  Print actions without executing them
  -h, --help                 Show this help

Examples:
  scripts/ops/start_live_session.sh
  scripts/ops/start_live_session.sh --transport ws --skip-refresh
  scripts/ops/start_live_session.sh --real --allow-real
  scripts/ops/start_live_session.sh --selection-file best_optimized_parameters/live/live_selection_20260217T150255Z.json
EOF
}

print_cmd() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="paper"
ALLOW_REAL=0
TRANSPORT="poll"
ENV_FILE=".env"
USE_ENV_FILE=1
INIT_SCHEMA=1
RUN_REFRESH=1
RUN_VALIDATE=1
RUN_LIVE_DECISION=1
RUN_PREFLIGHT=1
PREFLIGHT_STALE_MINUTES=30
SELECTION_FILE=""
USE_SELECTION=1
STRATEGY_NAME=""
RUN_ID=""
STOP_FILE=""
DSN_OVERRIDE=""
LAUNCH_MARKER=""
DRY_RUN=0
LIVE_EXTRA_ARGS=()

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
    --allow-real)
      ALLOW_REAL=1
      shift
      ;;
    --transport)
      TRANSPORT="${2:?missing value for --transport}"
      shift 2
      ;;
    --env-file)
      ENV_FILE="${2:?missing value for --env-file}"
      USE_ENV_FILE=1
      shift 2
      ;;
    --no-env-file)
      USE_ENV_FILE=0
      shift
      ;;
    --dsn)
      DSN_OVERRIDE="${2:?missing value for --dsn}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --stop-file)
      STOP_FILE="${2:?missing value for --stop-file}"
      shift 2
      ;;
    --launch-marker)
      LAUNCH_MARKER="${2:?missing value for --launch-marker}"
      shift 2
      ;;
    --selection-file)
      SELECTION_FILE="${2:?missing value for --selection-file}"
      USE_SELECTION=1
      shift 2
      ;;
    --no-selection)
      USE_SELECTION=0
      shift
      ;;
    --strategy)
      STRATEGY_NAME="${2:?missing value for --strategy}"
      shift 2
      ;;
    --skip-init-schema)
      INIT_SCHEMA=0
      shift
      ;;
    --skip-refresh)
      RUN_REFRESH=0
      shift
      ;;
    --skip-validate)
      RUN_VALIDATE=0
      shift
      ;;
    --skip-live-decision)
      RUN_LIVE_DECISION=0
      shift
      ;;
    --skip-preflight)
      RUN_PREFLIGHT=0
      shift
      ;;
    --preflight-stale-minutes)
      PREFLIGHT_STALE_MINUTES="${2:?missing value for --preflight-stale-minutes}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      LIVE_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$TRANSPORT" != "poll" && "$TRANSPORT" != "ws" ]]; then
  echo "Unsupported transport: $TRANSPORT" >&2
  exit 2
fi

if [[ "$MODE" == "real" && "$ALLOW_REAL" != "1" ]]; then
  echo "--real requires --allow-real." >&2
  exit 2
fi

if [[ "$MODE" == "real" && "$RUN_PREFLIGHT" != "1" ]]; then
  echo "Real mode requires live_readiness_preflight.py and cannot use --skip-preflight." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if [[ "$USE_ENV_FILE" == "1" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  else
    echo "Env file not found: $ENV_FILE" >&2
    exit 1
  fi
fi

if [[ -n "$DSN_OVERRIDE" ]]; then
  export LQ_POSTGRES_DSN="$DSN_OVERRIDE"
fi

export LQ__LIVE__MODE="$MODE"

ENABLE_REAL_ARGS=()
if [[ "$MODE" == "real" ]]; then
  export LUMINA_ENABLE_LIVE_REAL="true"
  ENABLE_REAL_ARGS=(--enable-live-real)
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="${MODE}-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "$STOP_FILE" ]]; then
  STOP_FILE="/tmp/lq-${MODE}.stop"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF
Repo root: $REPO_ROOT
Mode: $MODE
Transport: $TRANSPORT
Run ID: $RUN_ID
Stop file: $STOP_FILE
Selection mode: $([[ "$USE_SELECTION" == "1" ]] && echo "enabled" || echo "disabled")
Refresh: $RUN_REFRESH
Validate: $RUN_VALIDATE
Live decision: $RUN_LIVE_DECISION
Preflight: $RUN_PREFLIGHT
Init schema: $INIT_SCHEMA
EOF
fi

if [[ "$INIT_SCHEMA" == "1" ]]; then
  run_cmd uv run python scripts/init_postgres_schema.py
fi

if [[ "$RUN_REFRESH" == "1" ]]; then
  run_cmd uv run python scripts/research/refresh_final_portfolio_validation_data.py
fi

if [[ "$RUN_VALIDATE" == "1" ]]; then
  run_cmd uv run python scripts/research/validate_saved_incumbent_portfolio.py
fi

if [[ "$RUN_LIVE_DECISION" == "1" ]]; then
  run_cmd uv run python scripts/research/write_portfolio_live_readiness_decision.py
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  PREFLIGHT_TMP="$(mktemp)"
  cleanup_preflight_tmp() {
    rm -f "$PREFLIGHT_TMP"
  }
  trap cleanup_preflight_tmp EXIT

  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd uv run python scripts/ops/live_readiness_preflight.py --stale-minutes "$PREFLIGHT_STALE_MINUTES"
  else
    uv run python scripts/ops/live_readiness_preflight.py \
      --stale-minutes "$PREFLIGHT_STALE_MINUTES" | tee "$PREFLIGHT_TMP"
  fi

  if [[ "$MODE" == "paper" && "$DRY_RUN" != "1" ]]; then
    uv run python - "$PREFLIGHT_TMP" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

status = dict(payload.get("status") or {})
if not bool(status.get("ready_for_paper")):
    recommended = payload.get("recommended_action")
    raise SystemExit(
        "Paper preflight blocked. "
        f"recommended_action={recommended!r}"
    )
PY
  elif [[ "$MODE" == "real" && "$DRY_RUN" != "1" ]]; then
    uv run python - "$PREFLIGHT_TMP" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

status = dict(payload.get("status") or {})
if not bool(status.get("ready_for_real")):
    recommended = payload.get("recommended_action")
    raise SystemExit(
        "Real-mode preflight blocked. "
        f"recommended_action={recommended!r}"
    )
PY
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  rm -f "$STOP_FILE"
fi

LIVE_CMD=(uv run lq live --transport "$TRANSPORT" --run-id "$RUN_ID" --stop-file "$STOP_FILE")
if [[ "${#ENABLE_REAL_ARGS[@]}" -gt 0 ]]; then
  LIVE_CMD+=("${ENABLE_REAL_ARGS[@]}")
fi
if [[ "$USE_SELECTION" == "0" ]]; then
  LIVE_CMD+=(--no-selection)
elif [[ -n "$SELECTION_FILE" ]]; then
  LIVE_CMD+=(--selection-file "$SELECTION_FILE")
fi
if [[ -n "$STRATEGY_NAME" ]]; then
  LIVE_CMD+=(--strategy "$STRATEGY_NAME")
fi
if [[ "${#LIVE_EXTRA_ARGS[@]}" -gt 0 ]]; then
  LIVE_CMD+=("${LIVE_EXTRA_ARGS[@]}")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  print_cmd "${LIVE_CMD[@]}"
  exit 0
fi

if [[ -n "$LAUNCH_MARKER" ]]; then
  mkdir -p "$(dirname "$LAUNCH_MARKER")"
  : > "$LAUNCH_MARKER"
fi

echo "Launching LuminaQuant live session..."
print_cmd "${LIVE_CMD[@]}"
exec "${LIVE_CMD[@]}"
