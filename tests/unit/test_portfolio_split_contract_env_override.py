from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "lumina_quant" / "portfolio_split_contract.py"
)


def test_split_contract_respects_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LQ_PORTFOLIO_TRAIN_START", "2025-01-01")
    monkeypatch.setenv("LQ_PORTFOLIO_TRAIN_END", "2025-12-31")
    monkeypatch.setenv("LQ_PORTFOLIO_VAL_START", "2026-01-01")
    monkeypatch.setenv("LQ_PORTFOLIO_VAL_END", "2026-02-28")
    monkeypatch.setenv("LQ_PORTFOLIO_OOS_START", "2026-03-01")

    spec = importlib.util.spec_from_file_location("portfolio_split_contract_envtest", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.split_for_day_key("2026-02-15") == "val"
    assert module.split_for_day_key("2026-03-01") == "oos"
    assert module.split_windows()["oos_start"] == "2026-03-01T00:00:00Z"
