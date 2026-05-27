from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from lumina_quant.dashboard.bridge import (
    DEFAULT_DASHBOARD_COMPAT_PATH,
    DashboardBridgeContract,
    resolve_dashboard_bridge_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_RETIRED_STUB_PATH = REPO_ROOT / "src" / "lumina_quant" / "dashboard" / "retired_stub.py"
DASHBOARD_NEXT_APP_DIR = REPO_ROOT / "apps" / "dashboard_web"


def _dashboard_pythonpath() -> str:
    entries = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    existing = str(os.environ.get("PYTHONPATH", "") or "").strip()
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _dashboard_env(contract: DashboardBridgeContract | None = None) -> dict[str, str]:
    resolved_contract = contract or resolve_dashboard_bridge_contract(
        launch_mode="next",
        retired_stub_path=DASHBOARD_RETIRED_STUB_PATH,
        next_app_dir=DASHBOARD_NEXT_APP_DIR,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _dashboard_pythonpath()
    env["LQ_DASHBOARD_LAUNCH_MODE"] = resolved_contract.launch_mode
    env["LQ_DASHBOARD_COMPAT_PATH"] = resolved_contract.compatibility_path
    env["LQ_DASHBOARD_FRONTEND_TARGET"] = resolved_contract.frontend_target
    return env


def build_dashboard_contract(
    *,
    compat_path: str | None = None,
) -> DashboardBridgeContract:
    return resolve_dashboard_bridge_contract(
        launch_mode="next",
        retired_stub_path=DASHBOARD_RETIRED_STUB_PATH.resolve(),
        next_app_dir=DASHBOARD_NEXT_APP_DIR,
        compatibility_path=compat_path,
    )


def build_dashboard_command(
    *,
    compat_path: str | None = None,
) -> list[str]:
    build_dashboard_contract(compat_path=compat_path)
    return ["npm", "run", "dev"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or print dashboard launch command.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch the dashboard Python bridge instead of printing the command.",
    )
    parser.add_argument(
        "--compat-path",
        default=DEFAULT_DASHBOARD_COMPAT_PATH,
        help="Stable compatibility JSON path for the first migrated dashboard slice.",
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="Print the resolved dashboard migration compatibility contract as JSON.",
    )
    args = parser.parse_args(argv)

    contract = build_dashboard_contract(
        compat_path=args.compat_path,
    )
    command = build_dashboard_command(
        compat_path=args.compat_path,
    )

    if args.print_contract:
        print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.run:
        return int(
            subprocess.call(command, env=_dashboard_env(contract), cwd=str(DASHBOARD_NEXT_APP_DIR))
        )
    print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
