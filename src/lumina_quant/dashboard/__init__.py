"""Dashboard migration compatibility helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

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


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.bridge")
    value = getattr(module, name)
    globals()[name] = value
    return value
