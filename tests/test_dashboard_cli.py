"""Tests for the lq dashboard CLI (v2 — DashboardBridgeContractV2)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from lumina_quant.cli import dashboard


def test_dashboard_env_includes_repo_root_and_src() -> None:
    from lumina_quant.dashboard.bridge import build_dashboard_bridge_contract_v2

    contract = build_dashboard_bridge_contract_v2()
    env = dashboard._dashboard_env(contract)
    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)

    assert str(dashboard.REPO_ROOT) in pythonpath_entries
    assert str(dashboard.REPO_ROOT / "src") in pythonpath_entries


def test_dashboard_env_exposes_launch_mode_and_frontend_target() -> None:
    from lumina_quant.dashboard.bridge import build_dashboard_bridge_contract_v2

    contract = build_dashboard_bridge_contract_v2()
    env = dashboard._dashboard_env(contract)

    assert env["LQ_DASHBOARD_LAUNCH_MODE"] == "next"
    assert env["LQ_DASHBOARD_FRONTEND_TARGET"].endswith("apps/dashboard_web")


def test_dashboard_main_run_uses_dashboard_web_cwd_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_call(command, *, env, cwd):
        captured["command"] = list(command)
        captured["env"] = dict(env)
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(dashboard.subprocess, "call", _fake_call)

    exit_code = dashboard.main(["--run"])

    assert exit_code == 0
    assert captured["command"] == ["npm", "run", "dev"]
    assert captured["cwd"] == str(dashboard.DASHBOARD_NEXT_APP_DIR)


def test_dashboard_main_print_contract_emits_v2_json(capsys) -> None:
    exit_code = dashboard.main(["--print-contract"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["contract_version"] == 2
    assert payload["launch_mode"] == "next"
    assert payload["frontend_target"].endswith("apps/dashboard_web")
    assert len(payload["routes"]) == 12
    assert any(
        route["route"] == "/api/python/dashboard/alpha-evidence" for route in payload["routes"]
    )
    repo_root = Path(__file__).resolve().parents[1]
    assert (
        repo_root / "apps/dashboard_web/app/api/python/dashboard/alpha-evidence/route.ts"
    ).is_file()
    assert (repo_root / "apps/dashboard_web/lib/alpha-evidence-server.ts").is_file()


def test_dashboard_main_print_contract_routes_all_module_mode(capsys) -> None:
    dashboard.main(["--print-contract"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    for route in payload["routes"]:
        assert route["python_fn"].startswith("lumina_quant."), (
            f"route {route['route']} has non-module python_fn: {route['python_fn']}"
        )
        assert route["status"] in {"active", "guarded", "deprecated"}


def test_dashboard_repo_root_and_next_app_dir_defined() -> None:
    assert dashboard.REPO_ROOT.is_dir()
    assert str(dashboard.DASHBOARD_NEXT_APP_DIR).endswith("apps/dashboard_web")
