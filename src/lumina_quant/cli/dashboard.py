from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_RETIRED_STUB_PATH = REPO_ROOT / "src" / "lumina_quant" / "dashboard" / "retired_stub.py"


def build_dashboard_command() -> list[str]:
    return [sys.executable, str(DASHBOARD_RETIRED_STUB_PATH)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or print dashboard launch command.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the retired dashboard stub for guidance.",
    )
    args = parser.parse_args(argv)

    command = build_dashboard_command()
    if args.run:
        return int(subprocess.call(command, cwd=str(REPO_ROOT)))
    print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
