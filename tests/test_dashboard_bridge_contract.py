from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lumina_quant.dashboard.bridge import (
    DEFAULT_DASHBOARD_COMPAT_PATH,
    DashboardCompatibilityError,
    build_overview_payload_from_frames,
    load_overview_payload,
    normalize_dashboard_launch_mode,
    resolve_dashboard_bridge_contract,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "next"), ("auto", "next"), ("next", "next")],
)
def test_normalize_dashboard_launch_mode_accepts_supported_values(
    value: str | None, expected: str
) -> None:
    assert normalize_dashboard_launch_mode(value) == expected


def test_normalize_dashboard_launch_mode_rejects_unsupported_value() -> None:
    with pytest.raises(DashboardCompatibilityError):
        normalize_dashboard_launch_mode("legacy")


def test_resolve_dashboard_bridge_contract_defaults_to_next() -> None:
    contract = resolve_dashboard_bridge_contract(
        launch_mode="auto",
        retired_stub_path=ROOT / "src" / "lumina_quant" / "dashboard" / "retired_stub.py",
        next_app_dir=ROOT / "apps" / "dashboard_web",
    )

    assert contract.launch_mode == "next"
    assert contract.python_backend == "python"
    assert contract.frontend_target.endswith("apps/dashboard_web")
    assert contract.retired_stub_path.endswith("src/lumina_quant/dashboard/retired_stub.py")
    assert contract.compatibility_path == DEFAULT_DASHBOARD_COMPAT_PATH
    assert contract.slice_contract.path == DEFAULT_DASHBOARD_COMPAT_PATH


def test_resolve_dashboard_bridge_contract_supports_next_mode_with_api_path() -> None:
    contract = resolve_dashboard_bridge_contract(
        launch_mode="next",
        retired_stub_path=ROOT / "src" / "lumina_quant" / "dashboard" / "retired_stub.py",
        next_app_dir=ROOT / "apps" / "dashboard_web",
        compatibility_path="api/python/dashboard/overview",
    )
    payload = contract.to_dict()

    assert contract.launch_mode == "next"
    assert contract.frontend_target.endswith("apps/dashboard_web")
    assert contract.compatibility_path == DEFAULT_DASHBOARD_COMPAT_PATH
    assert payload["slice_contract"]["payload_schema"]["source"] == {
        "mode": "next",
        "backend": "python",
    }


def test_resolve_dashboard_bridge_contract_rejects_non_api_compatibility_path() -> None:
    with pytest.raises(DashboardCompatibilityError):
        resolve_dashboard_bridge_contract(
            launch_mode="next",
            retired_stub_path=ROOT / "src" / "lumina_quant" / "dashboard" / "retired_stub.py",
            next_app_dir=ROOT / "apps" / "dashboard_web",
            compatibility_path="/dashboard/overview",
        )


def test_build_overview_payload_from_frames_uses_real_run_and_equity_data() -> None:
    contract = resolve_dashboard_bridge_contract(
        launch_mode="next",
        retired_stub_path=ROOT / "src" / "lumina_quant" / "dashboard" / "retired_stub.py",
        next_app_dir=ROOT / "apps" / "dashboard_web",
    )
    runs = pd.DataFrame(
        [
            {
                "run_id": "run-123",
                "mode": "backtest",
                "status": "COMPLETED",
                "metadata": {"strategy": "RsiStrategy"},
                "strategy": "RsiStrategy",
                "started_at": "2026-03-01T00:00:00Z",
            },
            {
                "run_id": "run-122",
                "mode": "live",
                "status": "RUNNING",
                "metadata": {"strategy": "MomentumStrategy"},
                "strategy": "MomentumStrategy",
                "started_at": "2026-02-28T00:00:00Z",
            },
        ]
    )
    equity = pd.DataFrame(
        [
            {"datetime": "2026-03-01T00:00:00Z", "total": 1000.0},
            {"datetime": "2026-03-02T00:00:00Z", "total": 1050.0},
        ]
    )

    payload = build_overview_payload_from_frames(
        contract=contract,
        runs_frame=runs,
        equity_frame=equity,
    )

    assert payload["source"]["status"] == "ok"
    assert payload["source"]["run_id"] == "run-123"
    assert payload["summary_metrics"][0]["value"] == "run-123"
    assert payload["recent_runs"][0]["run_id"] == "run-123"
    assert payload["recent_runs"][1]["status"] == "RUNNING"
    assert payload["equity_curve"][-1]["equity"] == 1050.0
    assert payload["drawdown_curve"][-1]["drawdown"] == 0.0


def test_load_overview_payload_without_dsn_returns_safe_empty_payload() -> None:
    payload = load_overview_payload(
        launch_mode="next",
        dsn="",
    )

    assert payload["source"]["status"] == "missing_dsn"
    assert payload["summary_metrics"] == []
    assert payload["recent_runs"] == []
