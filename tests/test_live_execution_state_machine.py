"""Live execution state-machine tests."""

import queue
import time
import unittest
from datetime import datetime

from lumina_quant.core.events import OrderEvent
from lumina_quant.live.execution_live import (
    STATE_OPEN,
    STATE_PARTIAL,
    STATE_TIMEOUT,
    LiveExecutionHandler,
)


class MockBars:
    """Minimal bars adapter for live execution tests."""

    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    @staticmethod
    def get_latest_bar_datetime(symbol):
        _ = symbol
        return datetime(2026, 1, 1)

    @staticmethod
    def get_market_spec(symbol):
        _ = symbol
        return {"price_tick_size": 0.1}


class MockConfig:
    """Minimal config for live execution tests."""

    EXCHANGE_ID = "BINANCE"
    MARKET_TYPE = "future"
    TAKER_FEE_RATE = 0.0004
    ORDER_TIMEOUT = 2
    MODE = "paper"


class MockExchange:
    """Mock exchange implementing submit + fetch order lifecycle."""

    def __init__(self):
        self.fetch_calls = 0
        self.orders = []

    def execute_order(self, **kwargs):
        self.orders.append(dict(kwargs))
        return {
            "id": "order-1",
            "status": "open",
            "filled": 0.0,
            "amount": 2.0,
            "price": 100.0,
        }

    def fetch_order(self, order_id, symbol):
        _ = (order_id, symbol)
        self.fetch_calls += 1
        if self.fetch_calls == 1:
            return {
                "id": "order-1",
                "status": "open",
                "filled": 1.0,
                "amount": 2.0,
                "price": 100.0,
                "average": 100.0,
            }
        return {
            "id": "order-1",
            "status": "closed",
            "filled": 2.0,
            "amount": 2.0,
            "price": 100.0,
            "average": 100.0,
        }

    @staticmethod
    def get_balance(currency):
        _ = currency
        return 1000.0

    @staticmethod
    def get_all_positions():
        return {}


class MockProtectiveExchange(MockExchange):
    def __init__(self):
        super().__init__()
        self.last_params = None
        self.algo_orders = []

    def execute_order(self, **kwargs):
        self.last_params = dict(kwargs.get("params") or {})
        return super().execute_order(**kwargs)

    def execute_algo_order(self, **kwargs):
        self.algo_orders.append(dict(kwargs))
        params = dict(kwargs.get("params") or {})
        return {
            "id": f"algo-{len(self.algo_orders)}",
            "status": "open",
            "amount": kwargs.get("quantity", 0.0),
            "price": params.get("triggerPrice", 0.0),
            "clientOrderId": params.get("clientAlgoId"),
            "type": kwargs.get("type"),
            "side": kwargs.get("side"),
            "positionSide": params.get("positionSide"),
        }


class MockFilledProtectiveExchange(MockProtectiveExchange):
    def execute_order(self, **kwargs):
        self.last_params = dict(kwargs.get("params") or {})
        return {
            "id": "entry-filled",
            "status": "closed",
            "filled": kwargs.get("quantity", 0.0),
            "amount": kwargs.get("quantity", 0.0),
            "price": 100.0,
            "average": 100.0,
        }


class MockTimeoutExchange:
    """Mock exchange for timeout/cancel flow."""

    def __init__(self):
        self.cancel_calls = 0

    @staticmethod
    def execute_order(**kwargs):
        _ = kwargs
        return {
            "id": "timeout-order-1",
            "status": "open",
            "filled": 0.0,
            "amount": 1.0,
            "price": 100.0,
        }

    @staticmethod
    def fetch_order(order_id, symbol):
        _ = (order_id, symbol)
        return {
            "id": "timeout-order-1",
            "status": "open",
            "filled": 0.0,
            "amount": 1.0,
            "price": 100.0,
            "average": 100.0,
        }

    def cancel_order(self, order_id, symbol):
        _ = (order_id, symbol)
        self.cancel_calls += 1
        return True

    @staticmethod
    def get_balance(currency):
        _ = currency
        return 1000.0

    @staticmethod
    def get_all_positions():
        return {}


class TestLiveExecutionStateMachine(unittest.TestCase):
    """Validate order lifecycle transitions and fill emission."""

    def test_open_partial_filled_transitions(self):
        events = queue.Queue()
        exchange = MockExchange()
        handler = LiveExecutionHandler(events, MockBars(), MockConfig, exchange)

        order = OrderEvent("BTC/USDT", "MKT", 2.0, "BUY")
        handler.execute_order(order)
        self.assertEqual(len(handler.tracked_orders), 1)
        tracked = next(iter(handler.tracked_orders.values()))
        self.assertEqual(tracked["state"], STATE_OPEN)

        handler.check_open_orders()
        tracked = next(iter(handler.tracked_orders.values()))
        self.assertEqual(tracked["state"], STATE_PARTIAL)
        self.assertFalse(events.empty())

        handler.check_open_orders()
        self.assertEqual(len(handler.tracked_orders), 0)
        fill_events = []
        while not events.empty():
            fill_events.append(events.get())
        self.assertEqual(len(fill_events), 2)
        self.assertAlmostEqual(sum(event.quantity for event in fill_events), 2.0)

    def test_timeout_transition(self):
        events = queue.Queue()
        exchange = MockTimeoutExchange()
        handler = LiveExecutionHandler(events, MockBars(), MockConfig, exchange)
        state_events = []
        handler.set_order_state_callback(state_events.append)

        order = OrderEvent("BTC/USDT", "MKT", 1.0, "BUY")
        handler.execute_order(order)
        self.assertEqual(len(handler.tracked_orders), 1)
        _order_id, tracked = next(iter(handler.tracked_orders.items()))
        tracked["created_at"] = time.time() - 10.0

        handler.check_open_orders()
        self.assertEqual(exchange.cancel_calls, 1)
        self.assertEqual(len(handler.tracked_orders), 0)
        self.assertTrue(any(payload["state"] == STATE_TIMEOUT for payload in state_events))

    def test_limit_order_submits_limit_price_and_time_in_force(self):
        events = queue.Queue()
        exchange = MockExchange()
        handler = LiveExecutionHandler(events, MockBars(), MockConfig, exchange)

        order = OrderEvent(
            "BTC/USDT",
            "LMT",
            1.0,
            "SELL",
            price=99.9,
            position_side="LONG",
            reduce_only=True,
            time_in_force="GTC",
        )
        handler.execute_order(order)

        submitted = exchange.orders[-1]
        self.assertEqual(submitted["type"], "limit")
        self.assertEqual(submitted["side"], "sell")
        self.assertEqual(submitted["price"], 99.9)
        self.assertEqual(submitted["params"]["timeInForce"], "GTC")
        self.assertTrue(submitted["params"]["reduceOnly"])

    def test_market_order_requires_explicit_live_opt_in_when_guard_present(self):
        events = queue.Queue()
        exchange = MockExchange()

        class _StrictConfig(MockConfig):
            ALLOW_MARKET_ORDERS = False

        handler = LiveExecutionHandler(events, MockBars(), _StrictConfig, exchange)

        with self.assertRaisesRegex(RuntimeError, "Market orders are disabled"):
            handler.execute_order(OrderEvent("BTC/USDT", "MKT", 1.0, "BUY"))

    def test_live_protective_orders_fail_fast_without_exchange_params(self):
        events = queue.Queue()
        exchange = MockProtectiveExchange()

        class _RealConfig(MockConfig):
            MODE = "real"

        handler = LiveExecutionHandler(events, MockBars(), _RealConfig, exchange)

        order = OrderEvent(
            "BTC/USDT",
            "MKT",
            1.0,
            "BUY",
            stop_loss=99.0,
            take_profit=101.0,
        )
        with self.assertRaisesRegex(RuntimeError, "Real-mode live protective orders require"):
            handler.execute_order(order)

    def test_live_protective_orders_are_not_embedded_in_parent_market_order(self):
        events = queue.Queue()
        exchange = MockProtectiveExchange()
        handler = LiveExecutionHandler(events, MockBars(), MockConfig, exchange)

        order = OrderEvent(
            "BTC/USDT",
            "MKT",
            1.0,
            "BUY",
            stop_loss=99.0,
            take_profit=101.0,
        )

        handler.execute_order(order)
        assert exchange.last_params is not None
        self.assertNotIn("stopLossPrice", exchange.last_params)
        self.assertNotIn("takeProfitPrice", exchange.last_params)

    def test_paper_mode_submits_exchange_algo_protection_after_entry_fill(self):
        events = queue.Queue()
        exchange = MockFilledProtectiveExchange()
        handler = LiveExecutionHandler(events, MockBars(), MockConfig, exchange)
        states = []
        handler.set_order_state_callback(states.append)

        order = OrderEvent(
            "ETH/USDT",
            "MKT",
            1.5,
            "BUY",
            position_side="LONG",
            stop_loss=95.0,
            take_profit=110.0,
            metadata={"component_id": "eth-component"},
        )

        handler.execute_order(order)

        self.assertEqual([item["type"] for item in exchange.algo_orders], ["STOP", "TAKE_PROFIT"])
        self.assertEqual({item["side"] for item in exchange.algo_orders}, {"sell"})
        self.assertTrue(all(item["symbol"] == "ETH/USDT" for item in exchange.algo_orders))
        self.assertEqual(
            [item["params"]["triggerPrice"] for item in exchange.algo_orders], [95.0, 110.0]
        )
        self.assertEqual([item["params"]["price"] for item in exchange.algo_orders], [94.9, 109.9])
        self.assertTrue(
            all(item["params"]["timeInForce"] == "GTC" for item in exchange.algo_orders)
        )
        self.assertTrue(
            all(item["params"]["positionSide"] == "LONG" for item in exchange.algo_orders)
        )
        self.assertTrue(
            any(
                payload["message"] == "paper_exchange_protective_orders_submitted"
                for payload in states
            )
        )
        self.assertEqual(len(handler.protective_orders), 2)

    def test_market_style_protective_orders_require_explicit_opt_in(self):
        events = queue.Queue()
        exchange = MockFilledProtectiveExchange()

        class _MarketProtectiveConfig(MockConfig):
            ALLOW_MARKET_ORDERS = True
            PROTECTIVE_ORDER_STYLE = "market"

        handler = LiveExecutionHandler(events, MockBars(), _MarketProtectiveConfig, exchange)
        order = OrderEvent(
            "ETH/USDT",
            "MKT",
            1.5,
            "BUY",
            position_side="LONG",
            stop_loss=95.0,
            take_profit=110.0,
        )

        handler.execute_order(order)

        self.assertEqual(
            [item["type"] for item in exchange.algo_orders],
            ["STOP_MARKET", "TAKE_PROFIT_MARKET"],
        )
        self.assertNotIn("price", exchange.algo_orders[0]["params"])

    def test_paper_exchange_algo_protection_is_generic_across_assets_and_sides(self):
        events = queue.Queue()
        exchange = MockFilledProtectiveExchange()
        handler = LiveExecutionHandler(events, MockBars(), MockConfig, exchange)

        for symbol, direction, position_side, stop in (
            ("SOL/USDT", "SELL", "SHORT", 105.0),
            ("TRX/USDT", "BUY", "LONG", 0.09),
        ):
            order = OrderEvent(
                symbol,
                "MKT",
                2.0,
                direction,
                position_side=position_side,
                stop_loss=stop,
            )
            handler.execute_order(order)

        self.assertEqual(len(exchange.algo_orders), 2)
        sol_order, trx_order = exchange.algo_orders
        self.assertEqual(sol_order["symbol"], "SOL/USDT")
        self.assertEqual(sol_order["side"], "buy")
        self.assertEqual(sol_order["params"]["positionSide"], "SHORT")
        self.assertEqual(trx_order["symbol"], "TRX/USDT")
        self.assertEqual(trx_order["side"], "sell")
        self.assertEqual(trx_order["params"]["positionSide"], "LONG")

    def test_live_protective_orders_allow_explicit_exchange_params_mapping(self):
        events = queue.Queue()
        exchange = MockProtectiveExchange()
        handler = LiveExecutionHandler(events, MockBars(), MockConfig, exchange)

        order = OrderEvent(
            "BTC/USDT",
            "MKT",
            1.0,
            "BUY",
            stop_loss=99.0,
            take_profit=101.0,
            metadata={"exchange_params": {"stopLossPrice": 99.0, "takeProfitPrice": 101.0}},
        )
        handler.execute_order(order)
        self.assertEqual(exchange.last_params["stopLossPrice"], 99.0)
        self.assertEqual(exchange.last_params["takeProfitPrice"], 101.0)


if __name__ == "__main__":
    unittest.main()
