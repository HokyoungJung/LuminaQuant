"""Shared dashboard migration compatibility contract helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lumina_quant.dashboard.overview_service import (
    build_overview_payload_from_frames,
    load_overview_payload as _load_overview_payload_impl,
    resolve_dashboard_postgres_dsn,
)

DEFAULT_DASHBOARD_COMPAT_PATH = "/api/python/dashboard/overview"


class DashboardCompatibilityError(RuntimeError):
    """Raised when dashboard migration compatibility options are invalid."""


@dataclass(slots=True, frozen=True)
class DashboardSliceContract:
    """JSON contract for the first dashboard slice moved behind the compatibility bridge."""

    slice_id: str
    title: str
    transport: str
    producer: str
    path: str
    payload_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DashboardBridgeContract:
    """Normalized launch + compatibility bridge contract for dashboard migration."""

    launch_mode: str
    python_backend: str
    frontend_target: str
    retired_stub_path: str
    compatibility_path: str
    slice_contract: DashboardSliceContract

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_version"] = 1
        return payload


def normalize_dashboard_launch_mode(value: str | None, default: str = "auto") -> str:
    token = str(value or default).strip().lower()
    if token in {"auto", "next"}:
        return "next"
    raise DashboardCompatibilityError(
        f"Unsupported dashboard launch mode '{value}'. Expected one of: auto, next."
    )


def _normalize_compatibility_path(value: str | None) -> str:
    token = str(value or DEFAULT_DASHBOARD_COMPAT_PATH).strip() or DEFAULT_DASHBOARD_COMPAT_PATH
    normalized = token if token.startswith("/") else f"/{token}"
    if not normalized.startswith("/api/"):
        raise DashboardCompatibilityError(
            "Unsupported dashboard compatibility path "
            f"'{value}'. Expected a stable '/api/...' path."
        )
    return normalized


def resolve_dashboard_bridge_contract(
    *,
    launch_mode: str | None,
    retired_stub_path: str | Path,
    next_app_dir: str | Path,
    compatibility_path: str | None = None,
) -> DashboardBridgeContract:
    """Resolve the launch mode and minimal bridge contract for the first migrated slice."""
    resolved_mode = normalize_dashboard_launch_mode(launch_mode, default="auto")
    compat_path = _normalize_compatibility_path(compatibility_path)
    retired_stub_target = str(Path(retired_stub_path).resolve())
    next_target = str(Path(next_app_dir).resolve())

    frontend_target = next_target
    python_backend = "python"
    slice_contract = DashboardSliceContract(
        slice_id="overview",
        title="Dashboard overview compatibility slice",
        transport="json",
        producer="python",
        path=compat_path,
        payload_schema={
            "as_of": "iso8601-datetime",
            "summary_metrics": [
                {"key": "string", "label": "string", "value": "number|string|null"}
            ],
            "recent_runs": [
                {
                    "run_id": "string",
                    "mode": "string",
                    "status": "string",
                    "strategy": "string",
                    "started_at": "iso8601-datetime|null",
                }
            ],
            "equity_curve": [{"timestamp": "iso8601-datetime", "equity": "number"}],
            "drawdown_curve": [{"timestamp": "iso8601-datetime", "drawdown": "number"}],
            "source": {"mode": resolved_mode, "backend": python_backend},
        },
    )
    return DashboardBridgeContract(
        launch_mode=resolved_mode,
        python_backend=python_backend,
        frontend_target=frontend_target,
        retired_stub_path=retired_stub_target,
        compatibility_path=compat_path,
        slice_contract=slice_contract,
    )


def load_overview_payload(
    *,
    launch_mode: str | None = "next",
    dsn: str | None = None,
    limit: int = 120,
    run_limit: int = 10,
    compatibility_path: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    contract = resolve_dashboard_bridge_contract(
        launch_mode=launch_mode,
        retired_stub_path=repo_root / "src" / "lumina_quant" / "dashboard" / "retired_stub.py",
        next_app_dir=repo_root / "apps" / "dashboard_web",
        compatibility_path=compatibility_path,
    )
    return _load_overview_payload_impl(
        contract=contract,
        dsn=dsn,
        limit=limit,
        run_limit=run_limit,
    )


__all__ = [
    "DEFAULT_DASHBOARD_COMPAT_PATH",
    "DashboardBridgeContract",
    "DashboardCompatibilityError",
    "DashboardSliceContract",
    "build_overview_payload_from_frames",
    "load_overview_payload",
    "normalize_dashboard_launch_mode",
    "resolve_dashboard_bridge_contract",
    "resolve_dashboard_postgres_dsn",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the dashboard migration bridge contract.")
    parser.add_argument("--json", action="store_true", help="Print the contract as JSON.")
    parser.add_argument(
        "--next-app-dir",
        default=str(Path(__file__).resolve().parents[3] / "apps" / "dashboard_web"),
    )
    parser.add_argument(
        "--retired-stub-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "src"
            / "lumina_quant"
            / "dashboard"
            / "retired_stub.py"
        ),
    )
    parser.add_argument("--compat-path", default=DEFAULT_DASHBOARD_COMPAT_PATH)
    parser.add_argument("--overview-json", action="store_true", help="Print overview payload JSON.")
    args = parser.parse_args(argv)

    if args.overview_json:
        print(
            json.dumps(
                load_overview_payload(
                    launch_mode="next",
                    compatibility_path=args.compat_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    contract = resolve_dashboard_bridge_contract(
        launch_mode="next",
        retired_stub_path=args.retired_stub_path,
        next_app_dir=args.next_app_dir,
        compatibility_path=args.compat_path,
    )
    if args.json:
        print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    else:
        print(contract.compatibility_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
