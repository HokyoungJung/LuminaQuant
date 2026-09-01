from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    instances: list[_AuditStore] = []

    def __init__(self, _dsn):
        self.started: list[dict] = []
        self.ended: list[dict] = []
        self.events: list[tuple[str, dict]] = []
        self.__class__.instances.append(self)

    def start_run(self, **kwargs):
        self.started.append(dict(kwargs))
        return "run-1"

    def end_run(self, run_id, status="COMPLETED", metadata=None):
        self.ended.append(
            {
                "run_id": str(run_id),
                "status": str(status),
                "metadata": dict(metadata or {}),
            }
        )

    def log_risk_event(self, run_id, reason, details=None):
        _ = run_id
        self.events.append((str(reason), dict(details or {})))

    @staticmethod
    def close():
        return None


class _Notifier:
    instances: list[_Notifier] = []

    def __init__(self, *_args, **_kwargs):
        self.messages: list[str] = []
        self.__class__.instances.append(self)

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

    @staticmethod
    def shutdown(join_timeout=5.0):
        _ = join_timeout
        return None


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
        self.trading_frozen = False

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
    SYMBOLS = ["BTC/USDT"]
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


class _KeyboardInterruptEvents:
    @staticmethod
    def get(*_args, **_kwargs):
        raise KeyboardInterrupt


class _RepeatingEvents:
    def __init__(self, count: int) -> None:
        self._remaining = int(count)

    def get(self, *_args, **_kwargs):
        if self._remaining <= 0:
            raise KeyboardInterrupt
        self._remaining -= 1
        return SimpleNamespace(type="MARKET", symbol="BTC/USDT")

    @staticmethod
    def qsize() -> int:
        return 0


class _SingleEventThenInterrupt:
    def __init__(self) -> None:
        self._served = False

    def get(self, *_args, **_kwargs):
        if self._served:
            raise KeyboardInterrupt
        self._served = True
        return SimpleNamespace(type="MARKET", symbol="BTC/USDT")

    @staticmethod
    def qsize() -> int:
        return 0


def _build_trader(monkeypatch, config_cls=_Config) -> LiveTrader:
    _AuditStore.instances.clear()
    _Notifier.instances.clear()
    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _name: _Logger())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", config_cls)
    monkeypatch.setattr("lumina_quant.live.trader.StateManager", _StateManager)
    monkeypatch.setattr(
        "lumina_quant.live.trader.RiskManager", lambda cfg: SimpleNamespace(config=cfg)
    )
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: SimpleNamespace())
    monkeypatch.setattr("lumina_quant.live.trader.NotificationManager", _Notifier)
    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _AuditStore)
    monkeypatch.setattr("lumina_quant.live.trader.get_exchange", lambda _cfg: _Exchange())

    trader = LiveTrader(
        symbol_list=["BTC/USDT"],
        data_handler_cls=_DataHandler,
        execution_handler_cls=_ExecutionHandler,
        portfolio_cls=_Portfolio,
        strategy_cls=_Strategy,
    )
    trader.events = _KeyboardInterruptEvents()
    trader._sync_portfolio = lambda: None
    trader._emit_heartbeat = lambda force=False: None
    trader._evaluate_risk_guards = lambda: None
    trader._consume_data_fatal = lambda: None
    return trader


def test_ready_startup_notification_is_deferred_until_safe_init(monkeypatch):
    trader = _build_trader(monkeypatch)

    assert trader.notifier.messages == []

    trader.run()

    assert trader._startup_state == "ready"
    assert trader.notifier.messages == [
        "🚀 **LuminaQuant Started**\nSymbols: ['BTC/USDT']\nStrategy: _Strategy"
    ]
    assert trader.audit_store.ended[-1]["status"] == "STOPPED"
    assert trader.audit_store.ended[-1]["metadata"]["startup_state"] == "ready"


def test_failed_init_sends_failure_notification_and_closes_audit_run(monkeypatch):
    trader = _build_trader(monkeypatch)
    trader._sync_portfolio = lambda: (_ for _ in ()).throw(RuntimeError("sync exploded"))

    with pytest.raises(RuntimeError, match="sync exploded"):
        trader.run()

    assert trader._startup_state == "failed_init"
    assert trader.notifier.messages == [
        "🛑 **LuminaQuant Failed During Startup**\nReason: sync exploded"
    ]
    assert trader.audit_store.ended[-1]["status"] == "FAILED"
    assert trader.audit_store.ended[-1]["metadata"]["startup_state"] == "failed_init"


def test_constructor_failure_still_emits_failure_notification(monkeypatch):
    sent_messages: list[str] = []

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _name: _Logger())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", _Config)
    monkeypatch.setattr("lumina_quant.live.trader.StateManager", _StateManager)
    monkeypatch.setattr(
        "lumina_quant.live.trader.RiskManager", lambda cfg: SimpleNamespace(config=cfg)
    )
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: SimpleNamespace())

    class _CapturingNotifier:
        def __init__(self, *_args, **_kwargs):
            return None

        @staticmethod
        def send_message(msg):
            sent_messages.append(str(msg))

    monkeypatch.setattr("lumina_quant.live.trader.NotificationManager", _CapturingNotifier)
    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _AuditStore)
    monkeypatch.setattr(
        "lumina_quant.live.trader.get_exchange",
        lambda _cfg: (_ for _ in ()).throw(RuntimeError("exchange exploded")),
    )

    with pytest.raises(RuntimeError, match="exchange exploded"):
        LiveTrader(
            symbol_list=["BTC/USDT"],
            data_handler_cls=_DataHandler,
            execution_handler_cls=_ExecutionHandler,
            portfolio_cls=_Portfolio,
            strategy_cls=_Strategy,
        )

    assert sent_messages == ["🛑 **LuminaQuant Failed During Startup**\nReason: exchange exploded"]


def test_degraded_startup_sends_degraded_notification_after_gate(monkeypatch):
    trader = _build_trader(monkeypatch)

    def _degraded_gate():
        trader._startup_reconciliation_complete = True
        trader._set_startup_state("degraded", reason="startup_reconciliation_timeout")

    trader._run_startup_reconciliation_gate = _degraded_gate

    trader.run()

    assert trader._startup_state == "degraded"
    assert trader.notifier.messages == [
        "⚠️ **LuminaQuant Started in Degraded Mode**\nReason: startup_reconciliation_timeout\nSymbols: ['BTC/USDT']\nStrategy: _Strategy"
    ]
    assert trader.audit_store.ended[-1]["status"] == "STOPPED"
    assert trader.audit_store.ended[-1]["metadata"]["startup_state"] == "degraded"


def test_constructor_failure_after_audit_start_sends_failed_init_and_closes_run(monkeypatch):
    _AuditStore.instances.clear()
    _Notifier.instances.clear()
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
    monkeypatch.setattr(
        "lumina_quant.live.trader.RecoveryReconciliationService",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("recovery init exploded")),
    )

    with pytest.raises(RuntimeError, match="recovery init exploded"):
        LiveTrader(
            symbol_list=["BTC/USDT"],
            data_handler_cls=_DataHandler,
            execution_handler_cls=_ExecutionHandler,
            portfolio_cls=_Portfolio,
            strategy_cls=_Strategy,
        )

    assert _AuditStore.instances
    audit_store = _AuditStore.instances[-1]
    assert audit_store.started
    assert audit_store.ended[-1]["status"] == "FAILED"
    assert audit_store.ended[-1]["metadata"]["startup_state"] == "failed_init"
    assert _Notifier.instances
    assert _Notifier.instances[-1].messages == [
        "🛑 **LuminaQuant Failed During Startup**\nReason: recovery init exploded"
    ]


def test_repeated_main_loop_errors_trigger_fail_fast_shutdown(monkeypatch):
    trader = _build_trader(monkeypatch)
    trader.events = _RepeatingEvents(2)
    trader.main_loop_error_retry_limit = 2
    trader.main_loop_error_window_seconds = 60.0
    trader._ordered_shutdown = lambda: None
    monkeypatch.setattr("lumina_quant.live.trader.time.sleep", lambda *_args, **_kwargs: None)
    trader.process_event = lambda _event: (_ for _ in ()).throw(RuntimeError("loop exploded"))

    with pytest.raises(Exception, match="loop exploded"):
        trader.run()

    assert trader._hard_halt_active is True
    assert trader._hard_halt_reason == "main_loop_error_threshold"
    assert trader.audit_store.ended[-1]["status"] == "FAILED"


def test_single_main_loop_error_below_threshold_does_not_fail_stop(monkeypatch):
    trader = _build_trader(monkeypatch)
    trader.events = _SingleEventThenInterrupt()
    trader.main_loop_error_retry_limit = 2
    trader.main_loop_error_window_seconds = 60.0
    monkeypatch.setattr("lumina_quant.live.trader.time.sleep", lambda *_args, **_kwargs: None)
    trader.process_event = lambda _event: (_ for _ in ()).throw(RuntimeError("one-off loop error"))

    trader.run()

    assert trader._hard_halt_active is False
    assert trader.audit_store.ended[-1]["status"] == "STOPPED"


def test_real_mode_forces_startup_reconciliation_hard_fail(monkeypatch):
    class _RealConfig(_Config):
        MODE = "real"
        STARTUP_RECONCILIATION_HARD_FAIL = False

    trader = _build_trader(monkeypatch, config_cls=_RealConfig)

    assert trader.live_mode == "real"
    assert trader.startup_reconciliation_hard_fail is True


def test_real_mode_run_blocks_when_readiness_gate_fails(monkeypatch):
    class _RealConfig(_Config):
        MODE = "real"
        STARTUP_RECONCILIATION_HARD_FAIL = False

    trader = _build_trader(monkeypatch, config_cls=_RealConfig)
    monkeypatch.setattr(
        "lumina_quant.live.trader.enforce_live_readiness_from_files",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("real gate blocked")),
    )

    with pytest.raises(RuntimeError, match="real gate blocked"):
        trader.run()

    assert trader._startup_state == "failed_init"
    assert trader.audit_store.ended[-1]["status"] == "FAILED"
