#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/ops/install_shell_aliases.sh [options]

Install simple LuminaQuant shell helpers into ~/.bashrc (or another shell rc file).

Helpers added:
  lq-paper-on   -> resilient paper runner
  lq-paper-off  -> graceful paper stop
  lq-real-on    -> resilient real runner
  lq-real-off   -> graceful real stop

Options:
  --file PATH   Target shell rc file (default: ~/.bashrc)
  --print       Print the shell block instead of writing it
  -h, --help    Show this help
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_FILE="${HOME}/.bashrc"
PRINT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      TARGET_FILE="${2:?missing value for --file}"
      shift 2
      ;;
    --print)
      PRINT_ONLY=1
      shift
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

read -r -d '' BLOCK <<EOF || true
# >>> LuminaQuant live shortcuts >>>
lq-paper-on() {
  (
    cd "$REPO_ROOT" && \
    bash run_bot.sh --dsn "\${LQ_RUNTIME_POSTGRES_DSN:-postgresql:///luminaquant}" "\$@"
  )
}

lq-paper-off() {
  (
    cd "$REPO_ROOT" && \
    bash scripts/ops/stop_live_session.sh "\$@"
  )
}

lq-real-on() {
  (
    cd "$REPO_ROOT" && \
    bash run_bot.sh --real --allow-real --dsn "\${LQ_RUNTIME_POSTGRES_DSN:-postgresql:///luminaquant}" "\$@"
  )
}

lq-real-off() {
  (
    cd "$REPO_ROOT" && \
    bash scripts/ops/stop_live_session.sh --real "\$@"
  )
}
# <<< LuminaQuant live shortcuts <<<
EOF

if [[ "$PRINT_ONLY" == "1" ]]; then
  printf '%s\n' "$BLOCK"
  exit 0
fi

mkdir -p "$(dirname "$TARGET_FILE")"
touch "$TARGET_FILE"

python3 - "$TARGET_FILE" "$BLOCK" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
block = sys.argv[2]
text = path.read_text(encoding="utf-8") if path.exists() else ""
start = "# >>> LuminaQuant live shortcuts >>>"
end = "# <<< LuminaQuant live shortcuts <<<"

if start in text and end in text:
    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    new_text = prefix.rstrip() + "\n\n" + block + "\n" + suffix.lstrip("\n")
else:
    new_text = text.rstrip() + ("\n\n" if text.strip() else "") + block + "\n"

path.write_text(new_text, encoding="utf-8")
PY

echo "Installed LuminaQuant shortcuts into $TARGET_FILE"
echo "Reload with: source $TARGET_FILE"
