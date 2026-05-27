from __future__ import annotations

from lumina_quant.live.portfolio import get_live_portfolio_cls
from lumina_quant.system_assembly import RuntimeContract, build_live_runtime_contract, build_system


def _assert_runtime_portfolio_cls(portfolio_cls):
    resolved = get_live_portfolio_cls()
    assert portfolio_cls is resolved
    assert portfolio_cls.__name__ == "Portfolio"
    assert portfolio_cls.__module__ == "lumina_quant.backtesting.portfolio_backtest"


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
    _assert_runtime_portfolio_cls(components["portfolio_cls"])


def test_build_system_live_uses_canonical_live_runtime_contract():
    runtime_contract = build_live_runtime_contract()
    components = build_system("live")
    assert components["mode"] == "live"
    assert components["engine_cls"] is runtime_contract.engine_cls
    assert components["data_handler_cls"] is runtime_contract.data_handler_cls
    assert components["execution_handler_cls"] is runtime_contract.execution_handler_cls
    _assert_runtime_portfolio_cls(components["portfolio_cls"])


def test_build_live_runtime_contract_supports_ws_transport():
    poll_contract = build_live_runtime_contract()
    ws_contract = build_live_runtime_contract(transport="ws")

    assert isinstance(poll_contract, RuntimeContract)
    assert poll_contract.transport == "poll"
    assert ws_contract.transport == "ws"
    assert poll_contract.engine_cls is ws_contract.engine_cls
    assert poll_contract.execution_handler_cls is ws_contract.execution_handler_cls
    _assert_runtime_portfolio_cls(poll_contract.portfolio_cls)
    _assert_runtime_portfolio_cls(ws_contract.portfolio_cls)
    assert poll_contract.data_handler_cls is not ws_contract.data_handler_cls


def test_build_system_sandbox_supports_explicit_ws_transport():
    components = build_system("sandbox", transport="ws")

    assert components["mode"] == "sandbox"
    assert components["transport"] == "ws"
    _assert_runtime_portfolio_cls(components["portfolio_cls"])
