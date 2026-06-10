"""Shared dashboard migration compatibility contract helpers.

v1 — DashboardBridgeContract / DashboardSliceContract (compat, kept for legacy callers)
v2 — DashboardBridgeContractV2 / DashboardRouteDescriptor (Phase 6, canonical)
     All 11 routes use module-mode invocation (runUvPythonModuleJson).
     Zero snippet-bridge invocations.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

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


# ---------------------------------------------------------------------------
# v2 contract — canonical for Phase 6 and beyond
# ---------------------------------------------------------------------------

_V2_ROUTES: tuple[tuple[str, str, str], ...] = (
    # (api_path, python_module, status)
    ("/api/python/dashboard/overview",              "lumina_quant.dashboard.bridge",                   "active"),
    ("/api/python/dashboard/risk-health",           "lumina_quant.dashboard.risk_health_service",      "active"),
    ("/api/python/dashboard/workflow-jobs",         "lumina_quant.dashboard.workflow_jobs_service",    "active"),
    ("/api/python/dashboard/workflow-jobs/control", "lumina_quant.dashboard.workflow_jobs_service",    "active"),
    ("/api/python/dashboard/exact-window",          "lumina_quant.dashboard.exact_window_service",     "active"),
    ("/api/python/dashboard/performance-price",     "lumina_quant.dashboard.cutover_surfaces_service", "active"),
    ("/api/python/dashboard/execution-analytics",   "lumina_quant.dashboard.cutover_surfaces_service", "active"),
    ("/api/python/dashboard/market-data",           "lumina_quant.dashboard.cutover_surfaces_service", "active"),
    ("/api/python/dashboard/optimization-insights", "lumina_quant.dashboard.cutover_surfaces_service", "active"),
    ("/api/python/dashboard/raw-data",              "lumina_quant.dashboard.cutover_surfaces_service", "active"),
    ("/api/python/dashboard/report-export",         "lumina_quant.dashboard.cutover_surfaces_service", "active"),
)


@dataclass(slots=True, frozen=True)
class DashboardRouteDescriptor:
    """Descriptor for a single dashboard API route — drives module-mode invocation."""

    route: str       # e.g. "/api/python/dashboard/overview"
    python_fn: str   # module name for `uv run python -m <python_fn> --json`
    status: str      # "active" | "guarded" | "deprecated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DashboardBridgeContractV2:
    """v2 contract — all routes use module-mode invocation (runUvPythonModuleJson).

    The ``dsn_config_key`` is a dot-path into RuntimeConfig that resolves to the
    Postgres DSN (e.g. ``storage.postgres_dsn``).  Frontend and bridge server
    read this via ``RuntimeConfig.storage.postgres_dsn`` — they never call
    ``get_default_runtime_config()`` independently.
    """

    CONTRACT_VERSION: ClassVar[int] = 2

    launch_mode: str = "next"
    frontend_target: str = ""
    dsn_config_key: str = "storage.postgres_dsn"
    postgres_dsn: str = ""
    routes: tuple[DashboardRouteDescriptor, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract_version": self.CONTRACT_VERSION,
            "launch_mode": self.launch_mode,
            "frontend_target": self.frontend_target,
            "dsn_config_key": self.dsn_config_key,
            "routes": [r.to_dict() for r in self.routes],
        }
        return payload


def build_dashboard_bridge_contract_v2(
    *,
    dsn: str | None = None,
    frontend_target: str | None = None,
) -> DashboardBridgeContractV2:
    """Build the canonical v2 contract from RuntimeConfig.

    DSN resolution order: explicit ``dsn`` arg → ``LQ_POSTGRES_DSN`` env →
    ``RuntimeConfig.storage.postgres_dsn``.
    """
    from lumina_quant.configuration import get_default_runtime_config

    rt = get_default_runtime_config()
    resolved_dsn = str(
        dsn or os.getenv("LQ_POSTGRES_DSN") or rt.storage.postgres_dsn or ""
    ).strip()
    repo_root = Path(__file__).resolve().parents[3]
    resolved_target = str(
        frontend_target or (repo_root / "apps" / "dashboard_web")
    )
    routes = tuple(
        DashboardRouteDescriptor(route=r, python_fn=m, status=s)
        for r, m, s in _V2_ROUTES
    )
    return DashboardBridgeContractV2(
        launch_mode="next",
        frontend_target=resolved_target,
        dsn_config_key="storage.postgres_dsn",
        postgres_dsn=resolved_dsn,
        routes=routes,
    )


__all__ = [
    "DEFAULT_DASHBOARD_COMPAT_PATH",
    # v1 (kept for compat)
    "DashboardBridgeContract",
    "DashboardCompatibilityError",
    "DashboardSliceContract",
    "normalize_dashboard_launch_mode",
    "resolve_dashboard_bridge_contract",
    # v2 (canonical)
    "DashboardBridgeContractV2",
    "DashboardRouteDescriptor",
    "build_dashboard_bridge_contract_v2",
    # shared helpers
    "build_overview_payload_from_frames",
    "load_overview_payload",
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
