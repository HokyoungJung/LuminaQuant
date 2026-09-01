from __future__ import annotations

import queue
from types import SimpleNamespace
from typing import Any, cast

from lumina_quant.live.trader import LiveTrader
from lumina_quant.runtime_cache import RuntimeCache


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

    @staticmethod
    def get_all_positions():
        return {"BTC/USDT": 0.8}


class _ExchangeWithoutLegs:
    @staticmethod
    def get_all_position_legs():
        return {}

    @staticmethod
    def get_all_positions():
        return {"BTC/USDT": -0.7}


def _make_trader(exchange, current_positions):
    trader = LiveTrader.__new__(LiveTrader)
    trader._audit_closed = True
    trader.exchange = exchange
    trader.symbol_list = ["BTC/USDT"]
    trader.config = SimpleNamespace(POSITION_MODE="ONE_WAY")
    trader.portfolio = SimpleNamespace(current_positions=current_positions)
    trader.events = queue.Queue()
    trader.runtime_cache = RuntimeCache()
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


def test_flatten_all_positions_blocks_in_hedge_mode_when_legs_missing():
    trader = _make_trader(_ExchangeWithoutLegs(), {"BTC/USDT": 0.0})
    trader.config = SimpleNamespace(POSITION_MODE="HEDGE")

    orders_sent = trader._flatten_all_positions(reason="unit-test")

    assert orders_sent == 0
    assert trader.events.empty()
    assert trader.audit_store.events[-1][1] == "FLATTEN_ALL_BLOCKED_MISSING_LEGS"


def test_flatten_all_positions_uses_cached_legs_in_hedge_mode_when_fetch_is_empty():
    trader = _make_trader(_ExchangeWithoutLegs(), {"BTC/USDT": 0.0})
    trader.config = SimpleNamespace(POSITION_MODE="HEDGE")
    trader.runtime_cache = RuntimeCache()
    trader.runtime_cache.update_position_legs({"BTC/USDT": {"LONG": 1.2, "SHORT": 0.4}})
    trader.portfolio.current_position_legs = {"BTC/USDT": {"LONG": 1.2, "SHORT": 0.4}}

    orders_sent = trader._flatten_all_positions(reason="unit-test")

    assert orders_sent == 2
    first = trader.events.get_nowait()
    second = trader.events.get_nowait()
    sides = {(first.direction, first.position_side), (second.direction, second.position_side)}
    assert ("SELL", "LONG") in sides
    assert ("BUY", "SHORT") in sides


def test_reconcile_positions_syncs_portfolio_position_legs():
    trader = LiveTrader.__new__(LiveTrader)
    trader._audit_closed = True
    trader.logger = cast(Any, SimpleNamespace(error=lambda *args, **kwargs: None))
    trader.exchange = _ExchangeWithLegs()
    trader.symbol_list = ["BTC/USDT"]
    trader.portfolio = SimpleNamespace(current_positions={"BTC/USDT": 0.8})
    trader.runtime_cache = RuntimeCache()
    trader.config = SimpleNamespace(POSITION_MODE="HEDGE")
    trader.audit_store = cast(Any, _Audit())
    trader.notifier = cast(Any, _Notifier())
    trader.run_id = "run-test"
    trader.reconciliation_interval_sec = 1
    trader._last_reconciliation_monotonic = 0.0
    trader._last_dual_leg_signature = ()
    trader._last_drift_signature = ()
    trader._reconciliation_drift_events = 0

    trader._reconcile_positions(force=True)

    assert trader.runtime_cache.position_legs["BTC/USDT"]["LONG"] == 1.2
    assert trader.runtime_cache.position_legs["BTC/USDT"]["SHORT"] == 0.4
    assert trader.portfolio.current_position_legs["BTC/USDT"]["LONG"] == 1.2
    assert trader.portfolio.current_position_legs["BTC/USDT"]["SHORT"] == 0.4


def test_account_update_syncs_portfolio_position_legs_for_hedge_risk_checks():
    trader = LiveTrader.__new__(LiveTrader)
    trader._audit_closed = True
    trader.runtime_cache = RuntimeCache()
    trader.portfolio = SimpleNamespace(
        current_positions={"BTC/USDT": 0.0},
        current_position_legs={},
    )

    trader._apply_account_update(
        {
            "positions": [
                {"symbol": "BTCUSDT", "position_amount": 1.2, "position_side": "LONG"},
                {"symbol": "BTCUSDT", "position_amount": -0.4, "position_side": "SHORT"},
            ],
            "balances": [],
            "reason": "ACCOUNT_UPDATE",
            "exchange_ts_ms": 1_700_000_000_000,
        }
    )

    assert trader.runtime_cache.position_legs["BTC/USDT"]["LONG"] == 1.2
    assert trader.runtime_cache.position_legs["BTC/USDT"]["SHORT"] == 0.4
    assert trader.portfolio.current_position_legs["BTC/USDT"]["LONG"] == 1.2
    assert trader.portfolio.current_position_legs["BTC/USDT"]["SHORT"] == 0.4


def test_reconcile_positions_preserves_cached_legs_when_hedge_fetch_is_empty():
    trader = LiveTrader.__new__(LiveTrader)
    trader._audit_closed = True
    trader.logger = cast(Any, SimpleNamespace(error=lambda *args, **kwargs: None))
    trader.exchange = _ExchangeWithoutLegs()
    trader.symbol_list = ["BTC/USDT"]
    trader.portfolio = SimpleNamespace(
        current_positions={"BTC/USDT": 0.0},
        current_position_legs={"BTC/USDT": {"LONG": 1.2, "SHORT": 0.4}},
    )
    trader.runtime_cache = RuntimeCache()
    trader.runtime_cache.update_position_legs({"BTC/USDT": {"LONG": 1.2, "SHORT": 0.4}})
    trader.config = SimpleNamespace(POSITION_MODE="HEDGE")
    trader.audit_store = cast(Any, _Audit())
    trader.notifier = cast(Any, _Notifier())
    trader.run_id = "run-test"
    trader.reconciliation_interval_sec = 1
    trader._last_reconciliation_monotonic = 0.0
    trader._last_dual_leg_signature = ()
    trader._last_drift_signature = ()
    trader._reconciliation_drift_events = 0

    trader._reconcile_positions(force=True)

    assert trader.runtime_cache.position_legs["BTC/USDT"]["LONG"] == 1.2
    assert trader.runtime_cache.position_legs["BTC/USDT"]["SHORT"] == 0.4
    assert trader.portfolio.current_position_legs["BTC/USDT"]["LONG"] == 1.2
    assert trader.portfolio.current_position_legs["BTC/USDT"]["SHORT"] == 0.4
