from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.live.trader import LiveTrader


def _build_trader(monkeypatch):
    class _Logger:
        @staticmethod
        def info(*_args, **_kwargs):
            return None

        debug = warning = error = info

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

        @staticmethod
        def log_risk_event(*_args, **_kwargs):
            return None

    class _Notifier:
        def __init__(self, *_args, **_kwargs):
            self.messages = []

        def send_message(self, msg):
            self.messages.append(str(msg))

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
            _ = events, data_handler, config, exchange

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
        decision_cadence_seconds = 1

        def __init__(self, *_args, **_kwargs):
            return

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
        DECISION_CADENCE_SECONDS = 1
        HEARTBEAT_INTERVAL_SEC = 1
        RECONCILIATION_INTERVAL_SEC = 1
        POSTGRES_DSN = ""
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        MODE = "paper"
        STORAGE_EXPORT_CSV = False
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 1
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 1

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
    trader._audit_closed = True
    return trader


def test_live_staleness_gate_blocks_until_two_fresh_windows(monkeypatch):
    trader = _build_trader(monkeypatch)

    stale = MarketWindowEvent(
        time=1,
        window_seconds=5,
        bars_1s={"BTC/USDT": ((1, 1.0, 1.0, 1.0, 1.0, 1.0),)},
        lag_ms=2_000,
        is_stale=True,
        commit_id="c1",
    )
    fresh = MarketWindowEvent(
        time=2,
        window_seconds=5,
        bars_1s={"BTC/USDT": ((2, 1.0, 1.0, 1.0, 1.0, 1.0),)},
        lag_ms=100,
        is_stale=False,
        commit_id="c2",
    )

    assert trader._handle_market_window_staleness(stale) is True
    assert trader._materialized_stale_block_active is True

    assert trader._handle_market_window_staleness(fresh) is True
    assert trader._materialized_stale_block_active is True

    assert trader._handle_market_window_staleness(fresh) is False
    assert trader._materialized_stale_block_active is False
