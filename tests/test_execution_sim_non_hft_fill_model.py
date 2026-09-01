from __future__ import annotations

import queue

from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.core.events import MarketEvent, OrderEvent


class _Bars:
    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        if val_type == "high":
            return 101.0
        if val_type == "low":
            return 99.0
        if val_type == "open":
            return 100.0
        return 100.0


class _Config:
    RANDOM_SEED = 7
    SLIPPAGE_RATE = 0.0005
    SPREAD_RATE = 0.0002
    TAKER_FEE_RATE = 0.0004
    SIM_MAX_BAR_VOLUME_RATIO = 1.0
    SIM_LATENCY_MIN_BARS = 2
    SIM_LATENCY_MAX_BARS = 2


def test_simulated_execution_latency_model_releases_after_configured_bars():
    events = queue.Queue()
    handler = SimulatedExecutionHandler(events, _Bars(), _Config)

    handler.execute_order(OrderEvent("BTC/USDT", "MKT", 1.0, "BUY"))

    bar_1 = MarketEvent(
        time=1,
        symbol="BTC/USDT",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
    )
    bar_2 = MarketEvent(
        time=2,
        symbol="BTC/USDT",
        open=100.5,
        high=101.5,
        low=100.0,
        close=101.0,
        volume=10.0,
    )

    handler.check_open_orders(bar_1)
    assert events.qsize() == 0
    assert len(handler.active_orders) == 1

    handler.check_open_orders(bar_2)
    assert events.qsize() == 1
    fill = events.get_nowait()
    assert fill.type == "FILL"
    assert float(fill.quantity) == 1.0
