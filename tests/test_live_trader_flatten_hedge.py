from __future__ import annotations

import queue
from types import SimpleNamespace
from typing import Any, cast

from lumina_quant.live_trader import LiveTrader


class _Audit:
    def __init__(self):
        self.events = []

    def log_risk_event(self, run_id, reason, details=None):
        self.events.append((run_id, reason, details or {}))


class _Notifier:
    @staticmethod
    def send_message(_msg):
        return None


class _ExchangeWithLegs:
    @staticmethod
    def get_all_position_legs():
        return {"BTC/USDT": {"LONG": 1.2, "SHORT": 0.4}}


class _ExchangeWithoutLegs:
    @staticmethod
    def get_all_position_legs():
        return {}


def _make_trader(exchange, current_positions):
    trader = LiveTrader.__new__(LiveTrader)
    trader._audit_closed = True
    trader.exchange = exchange
    trader.symbol_list = ["BTC/USDT"]
    trader.portfolio = SimpleNamespace(current_positions=current_positions)
    trader.events = queue.Queue()
    trader.audit_store = cast(Any, _Audit())
    trader.run_id = "run-test"
    trader.notifier = cast(Any, _Notifier())
    return trader


def test_flatten_all_positions_prefers_side_aware_legs():
    trader = _make_trader(_ExchangeWithLegs(), {"BTC/USDT": 0.8})

    orders_sent = trader._flatten_all_positions(reason="unit-test")

    assert orders_sent == 2
    first = trader.events.get_nowait()
    second = trader.events.get_nowait()
    sides = {(first.direction, first.position_side), (second.direction, second.position_side)}
    assert ("SELL", "LONG") in sides
    assert ("BUY", "SHORT") in sides


def test_flatten_all_positions_fallbacks_to_net_position_when_legs_missing():
    trader = _make_trader(_ExchangeWithoutLegs(), {"BTC/USDT": -0.7})

    orders_sent = trader._flatten_all_positions(reason="unit-test")

    assert orders_sent == 1
    event = trader.events.get_nowait()
    assert event.direction == "BUY"
    assert event.position_side == "SHORT"
    assert float(event.quantity) == 0.7
