from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest
from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.live.data_poll import LiveDataHandler
from lumina_quant.live.trader import LiveTrader
from lumina_quant.strategy import Strategy


class _WindowFallbackStrategy(Strategy):
    def __init__(self):
        self.calls = []

    def calculate_signals(self, event):
        self.calls.append((str(getattr(event, "type", "")), event.symbol, float(event.close)))


def test_strategy_default_window_fallback_emits_legacy_market_events():
    strategy = _WindowFallbackStrategy()
    event = MarketWindowEvent(
        time=1_700_000_000_000,
        window_seconds=5,
        bars_1s={
            "BTC/USDT": ((1_700_000_000_000, 10.0, 11.0, 9.0, 10.5, 100.0),),
            "ETH/USDT": ((1_700_000_000_000, 20.0, 21.0, 19.0, 20.5, 200.0),),
        },
    )

    strategy.calculate_signals_window(event, aggregator=None)

    assert strategy.calls == [
        ("MARKET", "BTC/USDT", 10.5),
        ("MARKET", "ETH/USDT", 20.5),
    ]


def test_live_data_handler_emits_market_window_each_poll_with_configured_window(monkeypatch):
    class _Config:
        TIMEFRAME = "1m"
        LIVE_POLL_SECONDS = 1
        INGEST_WINDOW_SECONDS = 7
        LIVE_INGEST_INTERVAL_SEC = 999
        POLL_ERROR_BACKOFF_SECONDS = 1

    class _Exchange:
        def __init__(self):
            self.calls = []

        def fetch_ohlcv(self, symbol, timeframe, limit=0):
            self.calls.append((symbol, timeframe, int(limit)))
            if timeframe == "1m":
                return [[1_699_999_940_000, 1, 1, 1, 1, 1]]
            base = 1_700_000_000_000
            return [[base + (i * 1000), 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(limit)]

    class _ThreadStub:
        def __init__(self, target):
            self.target = target
            self.daemon = True

        def start(self):
            return None

    events = queue.Queue()
    exchange = _Exchange()

    monkeypatch.setattr("lumina_quant.live.data_poll.threading.Thread", _ThreadStub)
    handler = LiveDataHandler(events, ["BTC/USDT"], _Config, exchange)

    sleeps = {"count": 0}

    def _sleep(_seconds):
        sleeps["count"] += 1
        if sleeps["count"] >= 2:
            handler.continue_backtest = False

    monkeypatch.setattr("lumina_quant.live.data_poll.time.sleep", _sleep)
    handler.continue_backtest = True
    handler._poll_market_data()

    emitted = [events.get_nowait() for _ in range(events.qsize())]
    market_windows = [evt for evt in emitted if getattr(evt, "type", "") == "MARKET_WINDOW"]

    assert len(market_windows) == 2
    assert all(evt.window_seconds == 7 for evt in market_windows)
    assert all("BTC/USDT" in evt.bars_1s for evt in market_windows)
    assert all(len(evt.bars_1s["BTC/USDT"]) == 7 for evt in market_windows)
    assert all(
        timeframe != "1s" or limit == 7 for _, timeframe, limit in exchange.calls if timeframe == "1s"
    )


@pytest.mark.parametrize("strategy_cadence, expected", [(None, 13), (0, 13), (5, 5)])
def test_live_trader_strategy_decision_cadence_defaults_from_config(
    monkeypatch, strategy_cadence, expected
):
    class _Logger:
        @staticmethod
        def info(*_args, **_kwargs):
            return None

        warning = error = debug = info

    class _StateManager:
        @staticmethod
        def load_state():
            return {}

        @staticmethod
        def save_state(_state):
            return None

    class _AuditStore:
        def __init__(self, _dsn):
            pass

        @staticmethod
        def start_run(**_kwargs):
            return "run-1"

        @staticmethod
        def close():
            return None

    class _Notifier:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def send_message(_msg):
            return None

    class _DataHandler:
        def __init__(self, events, symbol_list, config, exchange):
            self.events = events
            self.symbol_list = symbol_list
            self.config = config
            self.exchange = exchange
            self.continue_backtest = True

        @staticmethod
        def get_latest_bar_value(_symbol, _val_type):
            return 0.0

    class _ExecutionHandler:
        def __init__(self, events, data_handler, config, exchange):
            self.events = events
            self.data_handler = data_handler
            self.config = config
            self.exchange = exchange

        @staticmethod
        def set_order_state_callback(_cb):
            return None

    class _Portfolio:
        def __init__(self, *_args, **_kwargs):
            self.current_positions = {"BTC/USDT": 0.0}
            self.current_holdings = {"total": 0.0, "cash": 0.0}

        @staticmethod
        def get_state():
            return {}

        @staticmethod
        def set_state(_state):
            return None

    class _Strategy:
        decision_cadence_seconds = strategy_cadence

        def __init__(self, bars, events, **_kwargs):
            self.bars = bars
            self.events = events

        @staticmethod
        def calculate_signals(_event):
            return None

        @staticmethod
        def get_state():
            return {}

        @staticmethod
        def set_state(_state):
            return None

    class _Config:
        DECISION_CADENCE_SECONDS = 13
        HEARTBEAT_INTERVAL_SEC = 1
        RECONCILIATION_INTERVAL_SEC = 5
        POSTGRES_DSN = ""
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
        EXCHANGE = {"driver": "ccxt", "name": "binance", "market_type": "future"}
        MODE = "paper"
        STORAGE_EXPORT_CSV = False

        @classmethod
        def validate(cls):
            return None

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _name: _Logger())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", _Config)
    monkeypatch.setattr("lumina_quant.live.trader.StateManager", _StateManager)
    monkeypatch.setattr("lumina_quant.live.trader.RiskManager", lambda _cfg: SimpleNamespace())
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: SimpleNamespace())
    monkeypatch.setattr("lumina_quant.live.trader.NotificationManager", _Notifier)
    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _AuditStore)
    monkeypatch.setattr("lumina_quant.live.trader.get_exchange", lambda _cfg: SimpleNamespace())

    trader = LiveTrader(
        symbol_list=["BTC/USDT"],
        data_handler_cls=_DataHandler,
        execution_handler_cls=_ExecutionHandler,
        portfolio_cls=_Portfolio,
        strategy_cls=_Strategy,
    )

    assert int(trader.strategy.decision_cadence_seconds) == expected
    trader._audit_closed = True
