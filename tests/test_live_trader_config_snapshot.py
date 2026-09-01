from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.live.trader import LiveTrader


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
        self.started = []

    def start_run(self, **kwargs):
        self.started.append(dict(kwargs))
        return "run-1"

    @staticmethod
    def close():
        return None


class _Notifier:
    def __init__(self, *_args, **_kwargs):
        self.messages = []

    def send_message(self, msg):
        self.messages.append(str(msg))


class _DataHandler:
    def __init__(self, events, symbol_list, config, exchange):
        self.events = events
        self.symbol_list = list(symbol_list)
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
    decision_cadence_seconds = 1

    def __init__(self, *_args, **_kwargs):
        return None

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
    SYMBOLS = ["BTC/USDT", "ETH/USDT"]
    DECISION_CADENCE_SECONDS = 1
    HEARTBEAT_INTERVAL_SEC = 1
    RECONCILIATION_INTERVAL_SEC = 5
    POSTGRES_DSN = ""
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
    MODE = "paper"
    STORAGE_EXPORT_CSV = False

    @classmethod
    def validate(cls):
        return None


class _Exchange:
    @staticmethod
    def load_markets():
        return {"BTC/USDT": {"symbol": "BTC/USDT"}}


def test_live_trader_uses_config_snapshot_without_mutating_global_live_config(monkeypatch):
    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _name: _Logger())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", _Config)
    monkeypatch.setattr("lumina_quant.live.trader.StateManager", _StateManager)
    monkeypatch.setattr(
        "lumina_quant.live.trader.RiskManager", lambda cfg: SimpleNamespace(config=cfg)
    )
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: SimpleNamespace())
    monkeypatch.setattr("lumina_quant.live.trader.NotificationManager", _Notifier)
    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _AuditStore)
    monkeypatch.setattr("lumina_quant.live.trader.get_exchange", lambda _cfg: _Exchange())

    trader = LiveTrader(
        symbol_list=["BTC/USDT", "ETH/USDT"],
        data_handler_cls=_DataHandler,
        execution_handler_cls=_ExecutionHandler,
        portfolio_cls=_Portfolio,
        strategy_cls=_Strategy,
    )

    assert _Config.SYMBOLS == ["BTC/USDT", "ETH/USDT"]
    assert trader.symbol_list == ["BTC/USDT"]
    assert trader.config.SYMBOLS == ["BTC/USDT"]
    assert trader.data_handler.symbol_list == ["BTC/USDT"]
    assert trader.data_handler.config.SYMBOLS == ["BTC/USDT"]
    assert trader.execution_handler.config.SYMBOLS == ["BTC/USDT"]
    trader._audit_closed = True
