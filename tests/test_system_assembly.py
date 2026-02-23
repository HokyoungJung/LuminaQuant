from __future__ import annotations

from lumina_quant.system_assembly import build_system


def test_build_system_backtest_contains_required_components():
    components = build_system("backtest")
    assert components["mode"] == "backtest"
    assert "engine_cls" in components
    assert "data_handler_cls" in components
    assert "execution_handler_cls" in components
    assert "portfolio_cls" in components


def test_build_system_sandbox_contains_required_components():
    components = build_system("sandbox")
    assert components["mode"] == "sandbox"
    assert "engine_cls" in components
    assert "data_handler_cls" in components
    assert "execution_handler_cls" in components
    assert "portfolio_cls" in components
