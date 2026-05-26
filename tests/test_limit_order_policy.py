from __future__ import annotations

import queue
from datetime import datetime

import pytest

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import SignalEvent
from lumina_quant.core.order_policy import limit_price_for_direction


class _Bars:
    symbol_list = ["ETH/USDT"]

    def get_latest_bar_value(self, symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    def get_latest_bar_datetime(self, symbol):
        _ = symbol
        return datetime(2026, 1, 1)

    def get_market_spec(self, symbol):
        _ = symbol
        return {
            "min_qty": 0.001,
            "qty_step": 0.001,
            "min_notional": 5.0,
            "price_tick_size": 0.1,
        }


class _LimitConfig:
    INITIAL_CAPITAL = 10_000.0
    MIN_TRADE_QTY = 0.001
    TARGET_ALLOCATION = 0.1
    MAX_DAILY_LOSS_PCT = 0.03
    RISK_PER_TRADE = 0.005
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_ORDER_VALUE = 5_000.0
    DEFAULT_STOP_LOSS_PCT = 0.01
    DEFAULT_ORDER_TYPE = "LMT"
    ALLOW_MARKET_ORDERS = False
    LIMIT_PRICE_MODE = "one_tick_worse"
    LIMIT_PRICE_OFFSET_TICKS = 1
    LIMIT_PRICE_TICK_FALLBACK = 0.0
    LIMIT_TIME_IN_FORCE = "GTC"


class _MarketConfig(_LimitConfig):
    DEFAULT_ORDER_TYPE = "MKT"
    ALLOW_MARKET_ORDERS = True


def _portfolio(config=_LimitConfig):
    return Portfolio(_Bars(), queue.Queue(), datetime(2026, 1, 1), config)


def _signal(signal_type: str, *, metadata=None):
    return SignalEvent(
        strategy_id="test",
        symbol="ETH/USDT",
        datetime=datetime(2026, 1, 1),
        signal_type=signal_type,
        stop_loss=98.0 if signal_type == "LONG" else 102.0,
        metadata=metadata,
    )


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("BUY", 100.1),
        ("SELL", 99.9),
    ],
)
def test_one_tick_worse_limit_price_is_side_safe(direction: str, expected: float) -> None:
    assert (
        limit_price_for_direction(
            reference_price=100.0,
            direction=direction,
            tick_size=0.1,
            mode="one_tick_worse",
            offset_ticks=1,
        )
        == pytest.approx(expected)
    )


def test_portfolio_generates_limit_buy_with_one_tick_worse_price() -> None:
    order = _portfolio().generate_order_from_signal(_signal("LONG"))

    assert order is not None
    assert order.order_type == "LMT"
    assert order.direction == "BUY"
    assert order.price == pytest.approx(100.1)
    assert order.time_in_force == "GTC"
    assert order.metadata["order_policy"]["limit_price_mode"] == "one_tick_worse"
    assert order.metadata["order_policy"]["price_tick_size"] == pytest.approx(0.1)


def test_portfolio_generates_limit_sell_for_short_and_exit() -> None:
    portfolio = _portfolio()
    short_order = portfolio.generate_order_from_signal(_signal("SHORT"))
    assert short_order is not None
    assert short_order.order_type == "LMT"
    assert short_order.direction == "SELL"
    assert short_order.price == pytest.approx(99.9)

    portfolio.current_positions["ETH/USDT"] = 2.0
    exit_order = portfolio.generate_order_from_signal(_signal("EXIT"))
    assert exit_order is not None
    assert exit_order.reduce_only is True
    assert exit_order.direction == "SELL"
    assert exit_order.order_type == "LMT"
    assert exit_order.price == pytest.approx(99.9)


def test_market_order_is_available_only_when_config_explicitly_allows_it() -> None:
    order = _portfolio(_MarketConfig).generate_order_from_signal(_signal("LONG"))

    assert order is not None
    assert order.order_type == "MKT"
    assert order.price is None
    assert order.time_in_force is None


def test_signal_market_request_is_coerced_to_limit_when_market_disabled() -> None:
    order = _portfolio().generate_order_from_signal(_signal("LONG", metadata={"order_type": "MKT"}))

    assert order is not None
    assert order.order_type == "LMT"
    assert order.price == pytest.approx(100.1)
