from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "minimum_viable_run.py"
_SPEC = importlib.util.spec_from_file_location("minimum_viable_run_module", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load script module from {_SCRIPT_PATH}")
mvr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mvr)


def test_build_demo_env_contains_no_infra_overrides():
    env = mvr.build_demo_env({})
    assert env["LQ__TRADING__SYMBOLS"] == '["BTC/USDT","ETH/USDT"]'
    assert env["LQ__STORAGE__BACKEND"] == "local"
    assert env["LQ_AUTO_COLLECT_DB"] == "0"
    assert env["LQ_BACKTEST_LOW_MEMORY"] == "1"
    assert env["LQ_BACKTEST_PERSIST_OUTPUT"] == "0"


def test_run_minimum_viable_backtest_invokes_csv_backtest(monkeypatch):
    captured: dict[str, object] = {}

    def _stub_ensure_sample_data(*, data_dir, days):
        captured["data_dir"] = str(data_dir)
        captured["days"] = int(days)

    def _stub_subprocess_run(cmd, cwd, env, check):
        captured["cmd"] = list(cmd)
        captured["cwd"] = str(cwd)
        captured["env"] = dict(env)
        captured["check"] = bool(check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mvr, "ensure_sample_data", _stub_ensure_sample_data)
    monkeypatch.setattr(mvr.subprocess, "run", _stub_subprocess_run)

    rc = mvr.run_minimum_viable_backtest(days=45)
    assert rc == 0
    assert captured["days"] == 45
    assert captured["cmd"][:6] == [
        mvr.sys.executable,
        "-m",
        "lumina_quant.cli.main",
        "backtest",
        "--data-source",
        "csv",
    ]
    assert "--no-auto-collect-db" in captured["cmd"]
    assert captured["env"]["LQ__TRADING__SYMBOLS"] == '["BTC/USDT","ETH/USDT"]'
