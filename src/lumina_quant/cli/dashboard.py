"""lq dashboard — launch the Next.js dashboard against the active RuntimeConfig.

All parameters are config-driven; no flags needed for normal operation.
``LQ_POSTGRES_DSN`` is set in the subprocess environment from
``RuntimeConfig.storage.postgres_dsn`` so the dashboard backend never reads
config independently.

Flags
-----
--run               Launch ``npm run dev`` in ``apps/dashboard_web`` (default action).
--print-contract    Print DashboardBridgeContractV2 as JSON and exit (diagnostic).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from lumina_quant.dashboard.bridge import (
    DashboardBridgeContractV2,
    build_dashboard_bridge_contract_v2,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_NEXT_APP_DIR = REPO_ROOT / "apps" / "dashboard_web"


def _dashboard_env(contract: DashboardBridgeContractV2) -> dict[str, str]:
    env = dict(os.environ)
    entries = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    existing = str(env.get("PYTHONPATH", "") or "").strip()
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env["LQ_DASHBOARD_LAUNCH_MODE"] = contract.launch_mode
    env["LQ_DASHBOARD_FRONTEND_TARGET"] = contract.frontend_target
    # Propagate DSN from RuntimeConfig so the dashboard services never call
    # get_default_runtime_config() independently.
    if contract.postgres_dsn:
        env["LQ_POSTGRES_DSN"] = contract.postgres_dsn
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lq dashboard",
        description="Launch the Next.js dashboard (config-driven via RuntimeConfig).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch dashboard dev server (npm run dev in apps/dashboard_web).",
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        dest="print_contract",
        help="Print DashboardBridgeContractV2 as JSON and exit.",
    )
    args = parser.parse_args(argv)

    contract = build_dashboard_bridge_contract_v2()

    if args.print_contract:
        print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.run:
        return int(
            subprocess.call(
                ["npm", "run", "dev"],
                env=_dashboard_env(contract),
                cwd=str(DASHBOARD_NEXT_APP_DIR),
            )
        )

    # Default: print the npm command (unchanged from v1 behaviour when no flag given)
    print("npm run dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
