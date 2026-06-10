"""Phase 5 structural gate: live/ must share Phase 4's ExecutionModel.

CI gate: live/execution_live.py MUST import ExecutionModel from
lumina_quant.backtesting.execution_model so the live order path
uses the same unified cost model as the backtest path.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


_EXECUTION_LIVE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "lumina_quant" / "live" / "execution_live.py"
)


def test_execution_live_imports_execution_model():
    """execution_live.py must import ExecutionModel from the shared backtesting module."""
    source = _EXECUTION_LIVE_PATH.read_text(encoding="utf-8")
    assert "ExecutionModel" in source, (
        "live/execution_live.py must import ExecutionModel from "
        "lumina_quant.backtesting.execution_model (Phase 5 structural gate)"
    )
    assert "execution_model" in source, (
        "live/execution_live.py must reference lumina_quant.backtesting.execution_model"
    )


def test_execution_live_imports_execution_model_config():
    """execution_live.py must also import ExecutionModelConfig for live config construction."""
    source = _EXECUTION_LIVE_PATH.read_text(encoding="utf-8")
    assert "ExecutionModelConfig" in source, (
        "live/execution_live.py must import ExecutionModelConfig alongside ExecutionModel"
    )


def test_execution_live_import_is_from_backtesting():
    """The ExecutionModel import must come from lumina_quant.backtesting.execution_model."""
    tree = ast.parse(_EXECUTION_LIVE_PATH.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            if "execution_model" in module and "ExecutionModel" in names:
                found = True
                break
    assert found, (
        "live/execution_live.py must have an explicit "
        "'from lumina_quant.backtesting.execution_model import ExecutionModel' statement"
    )


def test_execution_model_importable_from_backtesting():
    """The shared ExecutionModel must be importable (build gate: pyo3 present)."""
    from lumina_quant.backtesting.execution_model import ExecutionModel, ExecutionModelConfig


def test_execution_model_config_from_runtime_live_mode():
    """ExecutionModelConfig.from_runtime(rt, mode='live') produces a valid config."""
    from lumina_quant.backtesting.execution_model import ExecutionModelConfig
    from lumina_quant.configuration.loader import build_runtime_config

    raw = {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "live": {
            "mode": "paper",
            "market_data_source": "committed",
            "order_state_source": "polling",
            "exchange": {
                "driver": "binance_futures",
                "name": "binance",
                "market_type": "future",
                "position_mode": "HEDGE",
                "margin_mode": "isolated",
                "leverage": 3,
            },
        },
    }
    rt = build_runtime_config(raw, env={})
    cfg = ExecutionModelConfig.from_runtime(rt, mode="live")
    assert cfg.leverage == 3
    assert cfg.margin_mode == "isolated"
    assert cfg.taker_fee_rate > 0


def test_execution_live_instantiates_execution_model_ast():
    """execution_live.py must INSTANTIATE ExecutionModel (not just import it).

    AST check: at least one ``ExecutionModel(`` call-expression must appear
    in the source so the gate can never be satisfied by a bare import.
    """
    import ast

    source = _EXECUTION_LIVE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Direct call: ExecutionModel(...)
            if isinstance(func, ast.Name) and func.id == "ExecutionModel":
                found = True
                break
            # Attribute call: something.ExecutionModel(...)
            if isinstance(func, ast.Attribute) and func.attr == "ExecutionModel":
                found = True
                break
    assert found, (
        "live/execution_live.py must instantiate ExecutionModel(...) — "
        "bare import is not sufficient (Phase 5 structural gate)"
    )


def test_live_execution_handler_has_execution_model_attribute():
    """LiveExecutionHandler.__init__ must set self._execution_model to an ExecutionModel."""
    import queue
    from types import SimpleNamespace

    from lumina_quant.backtesting.execution_model import ExecutionModel
    from lumina_quant.live.execution_live import LiveExecutionHandler

    events = queue.Queue()
    bars = SimpleNamespace(
        get_latest_bar_value=lambda *a, **kw: 0.0,
        get_latest_bar_datetime=lambda *a, **kw: None,
    )
    config = SimpleNamespace(
        MODE="paper",
        ORDER_TIMEOUT=10,
        ORDER_STATE_SOURCE="polling",
        RECONCILIATION_POLL_FALLBACK_ENABLED=True,
        TAKER_FEE_RATE=0.0004,
        MAKER_FEE_RATE=0.0002,
        SLIPPAGE_RATE=0.0005,
        SPREAD_RATE=0.0002,
        LEVERAGE=3,
        MARGIN_MODE="isolated",
        MAINTENANCE_MARGIN_RATE=0.005,
        LIQUIDATION_BUFFER_RATE=0.0005,
        FUNDING_RATE_PER_8H=0.0,
        FUNDING_INTERVAL_HOURS=8,
        MARKET_TYPE="future",
        PAPER_EXCHANGE_PROTECTIVE_ORDERS=False,
        ALLOW_MARKET_ORDERS=False,
    )

    class _FakeGateway:
        def __init__(self, *a, **kw):
            pass

    class _FakeExchange:
        pass

    handler = LiveExecutionHandler(events, bars, config, _FakeExchange())
    assert hasattr(handler, "_execution_model"), (
        "LiveExecutionHandler must set self._execution_model in __init__"
    )
    assert isinstance(handler._execution_model, ExecutionModel), (
        "self._execution_model must be an ExecutionModel instance"
    )
