"""Tests that verify the 7+1 live-safety fixes from the adversarial audit.

Covers:
- Prod-routing gate: canary/full with mode=paper is rejected at config validation time
- Kill-switch envelope: zero/negative caps for max_order_value / max_symbol_exposure_pct /
  max_total_margin_pct are rejected
- Canary sizing: submitted order qty == base_qty * canary_position_fraction
- STORAGE_EXPORT_CSV present in _build_live_config_namespace output
- Circuit breaker: reduce-only orders are exempted even when breaker is tripped
- record_loss wired in handle_fill_event (consecutive-loss counter advances on closing fill)
- Stage gate: go_live_stage passed to enforce_live_readiness_from_files
- End-to-end contract: LiveTrader via typed RuntimeConfig processes one synthetic bar
  in paper/testnet mode without crash (STORAGE_EXPORT_CSV crash regression)
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.configuration.validate import validate_runtime_config
from lumina_quant.core.events import MarketEvent
from lumina_quant.live.trader import LiveTrader, _build_live_config_namespace
from lumina_quant.risk_manager import RiskManager


# ── helpers ───────────────────────────────────────────────────────────────────


def _base_raw(*, stage: str = "testnet", mode: str = "paper") -> dict:
    return {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "live": {
            "mode": mode,
            "go_live_stage": stage,
            "market_data_source": "committed",
            "order_state_source": "polling",
            "exchange": {
                "driver": "binance_futures",
                "name": "binance",
                "market_type": "future",
                "position_mode": "HEDGE",
                "margin_mode": "isolated",
                "leverage": 2,
            },
        },
    }


def _rt(*, stage: str = "testnet", mode: str = "paper"):
    return build_runtime_config(_base_raw(stage=stage, mode=mode), env={})


def _view(*, stage: str = "testnet", export_csv: bool = False, canary_fraction: float = 0.10):
    raw = _base_raw(stage=stage)
    raw["live"]["canary_position_fraction"] = canary_fraction
    raw["storage"]["export_csv"] = export_csv
    return _build_live_config_namespace(build_runtime_config(raw, env={}), symbols=["BTC/USDT"])


def _rm_cfg(
    *,
    max_order_value: float = 5000.0,
    max_sym_exp: float = 0.25,
    max_margin: float = 0.50,
    halt: int = 3,
):
    """Fake config that matches the attrs RiskManager reads."""
    return SimpleNamespace(
        MAX_ORDER_VALUE=max_order_value,
        MAX_ORDER_NOTIONAL_PCT=0.0,
        MAX_DAILY_LOSS_PCT=0.05,
        MAX_INTRADAY_DRAWDOWN_PCT=0.05,
        MAX_ROLLING_LOSS_PCT_1H=0.99,
        MAX_SYMBOL_EXPOSURE_PCT=max_sym_exp,
        MAX_TOTAL_MARGIN_PCT=max_margin,
        MAX_TOTAL_NOTIONAL_PCT=0.0,
        FREEZE_NEW_ENTRIES_ON_BREACH=True,
        AUTO_FLATTEN_ON_BREACH=False,
        TARGET_ALLOCATION_MODE="legacy_notional_cap",
        LEVERAGE=1,
        CONSECUTIVE_LOSS_HALT_COUNT=halt,
    )


def _portfolio(*, equity: float = 10_000.0, frozen: bool = False, breaker: bool = False):
    p = SimpleNamespace(
        symbol_list=["BTC/USDT"],
        current_holdings={"total": equity, "BTC/USDT": 0.0},
        current_positions={"BTC/USDT": 0.0},
        trading_frozen=frozen,
        circuit_breaker_tripped=breaker,
        day_start_equity=equity,
    )
    p.get_rolling_loss_pct = lambda _w=3600: 0.0
    return p


# ── Fix 1: prod-routing gate ──────────────────────────────────────────────────


def test_canary_stage_paper_mode_rejected_at_validation():
    """canary + mode=paper must be rejected — cross-field invariant."""
    rt = _rt(stage="canary", mode="paper")
    with pytest.raises(ValueError, match=r"canary.*real|real.*canary"):
        validate_runtime_config(rt)


def test_full_stage_paper_mode_rejected_at_validation():
    """full + mode=paper must be rejected — cross-field invariant."""
    rt = _rt(stage="full", mode="paper")
    with pytest.raises(ValueError, match=r"full.*real|real.*full"):
        validate_runtime_config(rt)


def test_testnet_stage_paper_mode_accepted():
    """testnet + mode=paper is the default dev config and must pass."""
    rt = _rt(stage="testnet", mode="paper")
    validate_runtime_config(rt)  # must not raise


def test_shadow_stage_paper_mode_accepted():
    """shadow + mode=paper is a valid testnet-routed shadow config."""
    rt = _rt(stage="shadow", mode="paper")
    validate_runtime_config(rt)  # must not raise


# ── Fix 5: kill-switch envelope — zero caps rejected ─────────────────────────


def test_max_order_value_zero_rejected():
    rt = _rt()
    rt.risk.max_order_value = 0.0
    with pytest.raises(ValueError, match="max_order_value"):
        validate_runtime_config(rt)


def test_max_order_value_negative_rejected():
    rt = _rt()
    rt.risk.max_order_value = -1.0
    with pytest.raises(ValueError, match="max_order_value"):
        validate_runtime_config(rt)


def test_max_symbol_exposure_zero_rejected():
    rt = _rt()
    rt.risk.max_symbol_exposure_pct = 0.0
    with pytest.raises(ValueError, match="max_symbol_exposure_pct"):
        validate_runtime_config(rt)


def test_max_total_margin_zero_rejected():
    rt = _rt()
    rt.risk.max_total_margin_pct = 0.0
    with pytest.raises(ValueError, match="max_total_margin_pct"):
        validate_runtime_config(rt)


def test_max_total_margin_excessive_rejected():
    rt = _rt()
    rt.risk.max_total_margin_pct = 6.0
    with pytest.raises(ValueError, match="max_total_margin_pct"):
        validate_runtime_config(rt)


# ── Fix 2: canary sizing — EFFECTIVE_POSITION_FRACTION consumed in order qty ─


def test_canary_qty_is_reduced_by_fraction():
    """Order qty with stage=canary must equal base_qty * canary_position_fraction."""
    from lumina_quant.backtesting.portfolio_backtest import Portfolio
    from lumina_quant.core.events import SignalEvent

    canary_frac = 0.15
    view_canary = _view(stage="canary", canary_fraction=canary_frac)
    view_full = _view(stage="full")  # EFFECTIVE_POSITION_FRACTION == 1.0

    # Minimal bars mock so _risk_based_quantity can call bars.get_latest_bar_value
    bars_mock = SimpleNamespace(
        get_latest_bar_value=lambda sym, field: 1000.0,
        get_latest_bar_datetime=lambda sym: "2024-01-01",
        symbol_list=["BTC/USDT"],
    )

    class _MkBars:
        def __new__(cls, *_, **__):
            return bars_mock

    def _make_portfolio(cfg):
        p = Portfolio.__new__(Portfolio)
        # Inject minimal attrs so _risk_based_quantity works
        p.config = cfg
        p.bars = bars_mock
        p.current_holdings = {"total": 10_000.0, "BTC/USDT": 0.0}
        p.current_positions = {"BTC/USDT": 0.0}
        p.entry_prices = {}
        p.current_position_legs = {}
        p.risk_per_trade = float(getattr(cfg, "RISK_PER_TRADE", 0.01))
        p.default_stop_loss_pct = float(getattr(cfg, "DEFAULT_STOP_LOSS_PCT", 0.02))
        p.max_symbol_exposure_pct = float(getattr(cfg, "MAX_SYMBOL_EXPOSURE_PCT", 0.25))
        p.max_order_value = float(getattr(cfg, "MAX_ORDER_VALUE", 5000.0))
        p.max_order_notional_pct = float(getattr(cfg, "MAX_ORDER_NOTIONAL_PCT", 0.0))
        p.target_allocation_mode = "legacy_notional_cap"
        p.leverage = 1.0
        return p

    signal = SimpleNamespace(
        symbol="BTC/USDT",
        signal_type="LONG",
        stop_loss=None,
        position_side="LONG",
        metadata={},
    )

    p_canary = _make_portfolio(view_canary)
    p_full = _make_portfolio(view_full)

    qty_canary = p_canary._risk_based_quantity(signal, 1000.0)
    qty_full = p_full._risk_based_quantity(signal, 1000.0)

    assert qty_full > 0, "base qty must be positive"
    assert pytest.approx(qty_canary, rel=1e-6) == qty_full * canary_frac


def test_full_stage_qty_not_reduced():
    """stage=full → EFFECTIVE_POSITION_FRACTION==1.0 → qty unchanged."""
    from lumina_quant.backtesting.portfolio_backtest import Portfolio

    view = _view(stage="full")
    p = Portfolio.__new__(Portfolio)
    p.config = view
    p.bars = SimpleNamespace(get_latest_bar_value=lambda s, f: 500.0)
    p.current_holdings = {"total": 5_000.0}
    p.risk_per_trade = 0.01
    p.default_stop_loss_pct = 0.02
    p.max_symbol_exposure_pct = 0.25
    p.max_order_value = 5000.0
    p.max_order_notional_pct = 0.0
    p.target_allocation_mode = "legacy_notional_cap"
    p.leverage = 1.0

    signal = SimpleNamespace(
        symbol="BTC/USDT",
        signal_type="LONG",
        stop_loss=None,
        position_side="LONG",
        metadata={},
    )
    qty = p._risk_based_quantity(signal, 500.0)
    # EFFECTIVE_POSITION_FRACTION==1.0 → no reduction
    assert qty > 0
    assert pytest.approx(view.EFFECTIVE_POSITION_FRACTION) == 1.0


# ── Fix 3: STORAGE_EXPORT_CSV in namespace ────────────────────────────────────


def test_storage_export_csv_present_in_namespace_false():
    view = _view(export_csv=False)
    assert hasattr(view, "STORAGE_EXPORT_CSV")
    assert view.STORAGE_EXPORT_CSV is False


def test_storage_export_csv_present_in_namespace_true():
    view = _view(export_csv=True)
    assert hasattr(view, "STORAGE_EXPORT_CSV")
    assert view.STORAGE_EXPORT_CSV is True


# ── Fix 7 (circuit breaker): reduce-only exempted even when breaker tripped ───


def test_circuit_breaker_blocks_normal_order():
    manager = RiskManager(_rm_cfg())
    port = _portfolio(breaker=True)
    order = SimpleNamespace(symbol="BTC/USDT", quantity=0.1, direction="BUY", reduce_only=False)
    passed, reason = manager.check_order(order, current_price=100.0, portfolio=port)
    assert passed is False
    assert "circuit breaker" in reason.lower()


def test_circuit_breaker_exempts_reduce_only():
    """After daily-loss trip, reduce-only orders must pass through to de-risk positions."""
    manager = RiskManager(_rm_cfg())
    port = _portfolio(breaker=True)
    order = SimpleNamespace(symbol="BTC/USDT", quantity=0.1, direction="SELL", reduce_only=True)
    passed, reason = manager.check_order(order, current_price=100.0, portfolio=port)
    assert passed is True, f"reduce-only must bypass circuit breaker; got: {reason}"


# ── Fix 4: record_loss wired in handle_fill_event ─────────────────────────────


def test_record_loss_wired_closing_long_fill(monkeypatch):
    """Closing a long position at a loss must advance the consecutive-loss counter."""
    from lumina_quant.core.events import FillEvent

    # Build a LiveTrader via the test-injection (LiveConfig) path
    class _FakeConfig:
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
        MAIN_LOOP_ERROR_RETRY_LIMIT = 3
        MAIN_LOOP_ERROR_WINDOW_SECONDS = 60
        ORDER_STATE_SOURCE = "polling"
        MARKET_DATA_SOURCE = "committed"
        RECONCILIATION_POLL_FALLBACK_ENABLED = True
        STARTUP_RECONCILIATION_HARD_FAIL = False
        GO_LIVE_STAGE = "testnet"

    class _FakeExchange:
        @staticmethod
        def load_markets():
            return {"BTC/USDT": {"symbol": "BTC/USDT"}}

    class _FakeDataHandler:
        def __init__(self, events, symbol_list, config, exchange):
            self.events = events
            self.symbol_list = list(symbol_list)
            self.config = config
            self.exchange = exchange
            self.continue_backtest = True

        @staticmethod
        def get_latest_bar_value(_s, _v):
            return 0.0

    class _FakeExecutionHandler:
        def __init__(self, events, data_handler, config, exchange):
            pass

        @staticmethod
        def set_order_state_callback(_cb):
            pass

    class _FakePortfolio:
        def __init__(self, *_, **__):
            self.current_positions = {"BTC/USDT": 1.0}  # long position
            self.entry_prices = {"BTC/USDT": 200.0}  # entry at 200
            self.current_holdings = {"total": 10_000.0, "cash": 10_000.0}
            self.current_position_legs = {}
            self.trading_frozen = False
            self.circuit_breaker_tripped = False

        def update_timeindex(self, _e):
            pass

        def update_fill(self, _e):
            # Simulate position closed after fill
            self.current_positions["BTC/USDT"] = 0.0
            self.entry_prices["BTC/USDT"] = None

        def create_equity_curve_dataframe(self):
            pass

        def output_trade_log(self, _path):
            pass

        @staticmethod
        def get_state():
            return {}

        @staticmethod
        def set_state(_s):
            pass

    class _FakeAuditStore:
        def __init__(self, _dsn):
            pass

        def start_run(self, **_kwargs):
            return "run-1"

        def log_fill(self, *_, **__):
            pass

        def close(self, *_, **__):
            pass

    class _FakeStateManager:
        @staticmethod
        def load_state():
            return {}

        @staticmethod
        def save_state(_s):
            pass

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _n: MagicMock())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", _FakeConfig)
    monkeypatch.setattr("lumina_quant.live.trader.StateManager", _FakeStateManager)
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: MagicMock())
    monkeypatch.setattr(
        "lumina_quant.live.trader.NotificationManager", lambda *_a, **_k: MagicMock()
    )
    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _FakeAuditStore)
    monkeypatch.setattr("lumina_quant.live.trader.get_exchange", lambda _c: _FakeExchange())

    trader = LiveTrader(
        symbol_list=["BTC/USDT"],
        data_handler_cls=_FakeDataHandler,
        execution_handler_cls=_FakeExecutionHandler,
        portfolio_cls=_FakePortfolio,
        strategy_cls=MagicMock(return_value=MagicMock(decision_cadence_seconds=1)),
    )
    trader._audit_closed = True

    initial_count = trader.risk_manager._consecutive_loss_count

    # Closing fill: sell 1.0 BTC at 150 (entry was 200 → loss)
    fill = FillEvent(
        timeindex="2024-01-01",
        symbol="BTC/USDT",
        exchange="BINANCE",
        quantity=1.0,
        direction="SELL",
        fill_cost=150.0,  # fill_cost = price * qty = 150 * 1.0
        commission=0.1,
        status="FILLED",
    )
    trader.handle_fill_event(fill)

    assert trader.risk_manager._consecutive_loss_count == initial_count + 1, (
        "record_loss must be called on a closing losing fill"
    )


def test_record_loss_not_called_on_opening_fill(monkeypatch):
    """Opening a new position has no realized PnL — counter must not change."""
    from lumina_quant.core.events import FillEvent

    class _FakeConfig:
        SYMBOLS = ["BTC/USDT"]
        DECISION_CADENCE_SECONDS = 1
        HEARTBEAT_INTERVAL_SEC = 1
        RECONCILIATION_INTERVAL_SEC = 5
        POSTGRES_DSN = ""
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
        EXCHANGE = {}
        MODE = "paper"
        STORAGE_EXPORT_CSV = False
        MAIN_LOOP_ERROR_RETRY_LIMIT = 3
        MAIN_LOOP_ERROR_WINDOW_SECONDS = 60
        ORDER_STATE_SOURCE = "polling"
        MARKET_DATA_SOURCE = "committed"
        RECONCILIATION_POLL_FALLBACK_ENABLED = True
        STARTUP_RECONCILIATION_HARD_FAIL = False
        GO_LIVE_STAGE = "testnet"

    class _FakePortfolioFlat:
        def __init__(self, *_, **__):
            self.current_positions = {"BTC/USDT": 0.0}  # no existing position
            self.entry_prices = {}  # no entry price → no realized PnL
            self.current_holdings = {"total": 10_000.0, "cash": 10_000.0}
            self.current_position_legs = {}
            self.trading_frozen = False
            self.circuit_breaker_tripped = False

        def update_timeindex(self, _e):
            pass

        def update_fill(self, _e):
            self.current_positions["BTC/USDT"] = 1.0

        def create_equity_curve_dataframe(self):
            pass

        def output_trade_log(self, _path):
            pass

        @staticmethod
        def get_state():
            return {}

        @staticmethod
        def set_state(_s):
            pass

    class _FakeAuditStore2:
        def __init__(self, _dsn):
            pass

        def start_run(self, **_kwargs):
            return "run-1"

        def log_fill(self, *_, **__):
            pass

        def close(self, *_, **__):
            pass

    class _FakeStateManager2:
        @staticmethod
        def load_state():
            return {}

        @staticmethod
        def save_state(_s):
            pass

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _n: MagicMock())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", _FakeConfig)
    monkeypatch.setattr("lumina_quant.live.trader.StateManager", _FakeStateManager2)
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: MagicMock())
    monkeypatch.setattr(
        "lumina_quant.live.trader.NotificationManager", lambda *_a, **_k: MagicMock()
    )
    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _FakeAuditStore2)
    monkeypatch.setattr(
        "lumina_quant.live.trader.get_exchange",
        lambda _c: SimpleNamespace(load_markets=lambda: {"BTC/USDT": {"symbol": "BTC/USDT"}}),
    )

    trader = LiveTrader(
        symbol_list=["BTC/USDT"],
        data_handler_cls=lambda *_, **__: SimpleNamespace(
            symbol_list=["BTC/USDT"],
            continue_backtest=True,
            get_latest_bar_value=lambda s, v: 0.0,
        ),
        execution_handler_cls=lambda *_, **__: SimpleNamespace(
            set_order_state_callback=lambda _: None,
        ),
        portfolio_cls=_FakePortfolioFlat,
        strategy_cls=MagicMock(return_value=MagicMock(decision_cadence_seconds=1)),
    )
    trader._audit_closed = True

    initial_count = trader.risk_manager._consecutive_loss_count

    # Opening buy fill: no prior position → no realized PnL
    fill = FillEvent(
        timeindex="2024-01-01",
        symbol="BTC/USDT",
        exchange="BINANCE",
        quantity=1.0,
        direction="BUY",
        fill_cost=200.0,
        commission=0.1,
        status="FILLED",
    )
    trader.handle_fill_event(fill)

    assert trader.risk_manager._consecutive_loss_count == initial_count, (
        "Opening fill has no entry_price → consecutive-loss counter must not change"
    )


# ── Fix 6: stage gate — go_live_stage passed through ─────────────────────────


def test_run_live_readiness_gate_passes_stage_to_enforcer(monkeypatch):
    """_run_live_readiness_gate must call enforce_live_readiness_from_files with
    the correct go_live_stage (not None, which would fall back to mode-only dispatch)."""
    captured_kwargs: dict = {}

    def _fake_enforce(*, mode, stale_minutes, env, go_live_stage=None, shadow_parity_ratio=None):
        captured_kwargs.update(
            mode=mode,
            go_live_stage=go_live_stage,
            shadow_parity_ratio=shadow_parity_ratio,
        )
        # Simulate a blocked gate to avoid side-effects
        from lumina_quant.live.readiness_policy import LiveReadinessBlockedError

        raise LiveReadinessBlockedError(mode=mode, recommended_action="test_block", payload={})

    monkeypatch.setattr(
        "lumina_quant.live.trader.enforce_live_readiness_from_files",
        _fake_enforce,
    )

    class _Cfg:
        GO_LIVE_STAGE = "shadow"
        MODE = "paper"
        SYMBOLS = ["BTC/USDT"]
        DECISION_CADENCE_SECONDS = 1
        HEARTBEAT_INTERVAL_SEC = 1
        RECONCILIATION_INTERVAL_SEC = 5
        POSTGRES_DSN = ""
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
        EXCHANGE = {}
        STORAGE_EXPORT_CSV = False
        MAIN_LOOP_ERROR_RETRY_LIMIT = 3
        MAIN_LOOP_ERROR_WINDOW_SECONDS = 60
        ORDER_STATE_SOURCE = "polling"
        MARKET_DATA_SOURCE = "committed"
        RECONCILIATION_POLL_FALLBACK_ENABLED = True
        STARTUP_RECONCILIATION_HARD_FAIL = False

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _n: MagicMock())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", _Cfg)
    monkeypatch.setattr(
        "lumina_quant.live.trader.StateManager",
        lambda: SimpleNamespace(load_state=lambda: {}, save_state=lambda _: None),
    )
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: MagicMock())
    monkeypatch.setattr(
        "lumina_quant.live.trader.NotificationManager", lambda *_a, **_k: MagicMock()
    )

    class _FakeAuditStore3:
        def __init__(self, _dsn):
            pass

        def start_run(self, **_kwargs):
            return "run-1"

        def log_risk_event(self, *_, **__):
            pass

        def close(self, *_, **__):
            pass

    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _FakeAuditStore3)
    monkeypatch.setattr(
        "lumina_quant.live.trader.get_exchange",
        lambda _c: SimpleNamespace(load_markets=lambda: {"BTC/USDT": {"symbol": "BTC/USDT"}}),
    )

    trader = LiveTrader(
        symbol_list=["BTC/USDT"],
        data_handler_cls=lambda *_, **__: SimpleNamespace(
            symbol_list=["BTC/USDT"],
            continue_backtest=True,
            get_latest_bar_value=lambda s, v: 0.0,
        ),
        execution_handler_cls=lambda *_, **__: SimpleNamespace(
            set_order_state_callback=lambda _: None,
        ),
        portfolio_cls=lambda *_, **__: SimpleNamespace(
            current_positions={},
            current_holdings={"total": 0.0},
            get_state=lambda: {},
            set_state=lambda _: None,
            trading_frozen=False,
        ),
        strategy_cls=MagicMock(return_value=MagicMock(decision_cadence_seconds=1)),
    )
    trader._audit_closed = True

    from lumina_quant.live.readiness_policy import LiveReadinessBlockedError

    with pytest.raises(LiveReadinessBlockedError):
        trader._run_live_readiness_gate()

    assert captured_kwargs.get("go_live_stage") == "shadow", (
        "go_live_stage must be forwarded to enforce_live_readiness_from_files"
    )


def test_testnet_stage_readiness_gate_skipped(monkeypatch):
    """testnet stage must skip the readiness gate entirely (no artifact constraints)."""
    enforcer_called = []

    monkeypatch.setattr(
        "lumina_quant.live.trader.enforce_live_readiness_from_files",
        lambda **_kwargs: enforcer_called.append(_kwargs) or {},
    )

    class _Cfg:
        GO_LIVE_STAGE = "testnet"
        MODE = "paper"
        SYMBOLS = ["BTC/USDT"]
        DECISION_CADENCE_SECONDS = 1
        HEARTBEAT_INTERVAL_SEC = 1
        RECONCILIATION_INTERVAL_SEC = 5
        POSTGRES_DSN = ""
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
        EXCHANGE = {}
        STORAGE_EXPORT_CSV = False
        MAIN_LOOP_ERROR_RETRY_LIMIT = 3
        MAIN_LOOP_ERROR_WINDOW_SECONDS = 60
        ORDER_STATE_SOURCE = "polling"
        MARKET_DATA_SOURCE = "committed"
        RECONCILIATION_POLL_FALLBACK_ENABLED = True
        STARTUP_RECONCILIATION_HARD_FAIL = False

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _n: MagicMock())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", _Cfg)
    monkeypatch.setattr(
        "lumina_quant.live.trader.StateManager",
        lambda: SimpleNamespace(load_state=lambda: {}, save_state=lambda _: None),
    )
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: MagicMock())
    monkeypatch.setattr(
        "lumina_quant.live.trader.NotificationManager", lambda *_a, **_k: MagicMock()
    )

    class _FakeAuditStore4:
        def __init__(self, _dsn):
            pass

        def start_run(self, **_kwargs):
            return "run-1"

        def close(self, *_, **__):
            pass

    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _FakeAuditStore4)
    monkeypatch.setattr(
        "lumina_quant.live.trader.get_exchange",
        lambda _c: SimpleNamespace(load_markets=lambda: {"BTC/USDT": {"symbol": "BTC/USDT"}}),
    )

    trader = LiveTrader(
        symbol_list=["BTC/USDT"],
        data_handler_cls=lambda *_, **__: SimpleNamespace(
            symbol_list=["BTC/USDT"],
            continue_backtest=True,
            get_latest_bar_value=lambda s, v: 0.0,
        ),
        execution_handler_cls=lambda *_, **__: SimpleNamespace(
            set_order_state_callback=lambda _: None,
        ),
        portfolio_cls=lambda *_, **__: SimpleNamespace(
            current_positions={},
            current_holdings={"total": 0.0},
            get_state=lambda: {},
            set_state=lambda _: None,
            trading_frozen=False,
        ),
        strategy_cls=MagicMock(return_value=MagicMock(decision_cadence_seconds=1)),
    )
    trader._audit_closed = True

    trader._run_live_readiness_gate()  # must not raise

    assert not enforcer_called, "enforce_live_readiness_from_files must not be called for testnet"


# ── Fix 3 (end-to-end contract): LiveTrader via typed RuntimeConfig + market bar


def test_live_trader_typed_config_handles_market_bar(monkeypatch):
    """LiveTrader built via the typed RuntimeConfig path (not LiveConfig injection)
    must process one synthetic market bar in testnet/paper mode without crash.

    Specifically exercises the STORAGE_EXPORT_CSV crash regression: before the fix,
    _build_live_config_namespace omitted STORAGE_EXPORT_CSV causing AttributeError
    on the first market bar.
    """
    from lumina_quant.configuration.loader import build_runtime_config

    rt = build_runtime_config(
        {
            "trading": {
                "symbols": ["ETH/USDT"],
                "timeframe": "1m",
                "timeframes": ["1s", "1m"],
            },
            "storage": {
                "materializer_required_timeframes": ["1s", "1m"],
                "materializer_base_timeframe": "1s",
                "export_csv": False,
            },
            "live": {
                "mode": "paper",
                "go_live_stage": "testnet",
                "market_data_source": "committed",
                "order_state_source": "polling",
                "exchange": {
                    "driver": "binance_futures",
                    "name": "binance",
                    "market_type": "future",
                    "position_mode": "HEDGE",
                    "margin_mode": "isolated",
                    "leverage": 1,
                },
            },
        },
        # Dummy env keys satisfy the for_live=True API-key check; not used in paper mode.
        env={"BINANCE_API_KEY": "test_key", "BINANCE_SECRET_KEY": "test_secret"},
    )

    class _FakeExchangeT:
        @staticmethod
        def load_markets():
            return {"ETH/USDT": {"symbol": "ETH/USDT"}}

    class _FakeDataHandlerT:
        def __init__(self, events, symbol_list, config, exchange):
            self.events = events
            self.symbol_list = list(symbol_list)
            self.config = config
            self.exchange = exchange
            self.continue_backtest = True

        @staticmethod
        def get_latest_bar_value(_s, _v):
            return 2000.0

        def update_bars(self, _event):
            pass

    class _FakeExecHandlerT:
        def __init__(self, *_, **__):
            pass

        @staticmethod
        def set_order_state_callback(_cb):
            pass

    class _FakePortfolioT:
        def __init__(self, *_, **__):
            self.current_positions = {"ETH/USDT": 0.0}
            self.current_holdings = {"total": 10_000.0, "cash": 10_000.0, "ETH/USDT": 0.0}
            self.current_position_legs = {}
            self.entry_prices = {}
            self.trading_frozen = False
            self.circuit_breaker_tripped = False
            self.symbol_list = ["ETH/USDT"]

        def update_timeindex(self, _e):
            pass

        def update_fill(self, _e):
            pass

        def create_equity_curve_dataframe(self):
            pass

        def output_trade_log(self, _path):
            pass

        @staticmethod
        def get_state():
            return {}

        @staticmethod
        def set_state(_s):
            pass

    class _FakeStrategyT:
        decision_cadence_seconds = 1

        def __init__(self, *_, **__):
            pass

        @staticmethod
        def calculate_signals(_e):
            return []

        @staticmethod
        def get_state():
            return {}

        @staticmethod
        def set_state(_s):
            pass

    class _FakeAuditT:
        def __init__(self, _dsn):
            pass

        def start_run(self, **_kwargs):
            return "run-contract"

        def log_equity(self, *_, **__):
            pass

        def log_fill(self, *_, **__):
            pass

        def log_risk_event(self, *_, **__):
            pass

        def close(self, *_, **__):
            pass

    class _FakeStateManagerT:
        @staticmethod
        def load_state():
            return {}

        @staticmethod
        def save_state(_s):
            pass

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _n: MagicMock())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", None)  # force typed path
    monkeypatch.setattr("lumina_quant.live.trader.StateManager", _FakeStateManagerT)
    monkeypatch.setattr(
        "lumina_quant.live.trader.RuntimeCache",
        lambda: SimpleNamespace(
            update_market=lambda *_, **__: None,
            open_orders={},
            position_legs={},
        ),
    )
    monkeypatch.setattr(
        "lumina_quant.live.trader.NotificationManager", lambda *_a, **_k: MagicMock()
    )
    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _FakeAuditT)
    monkeypatch.setattr("lumina_quant.live.trader.get_exchange", lambda _c: _FakeExchangeT())

    trader = LiveTrader(
        symbol_list=["ETH/USDT"],
        data_handler_cls=_FakeDataHandlerT,
        execution_handler_cls=_FakeExecHandlerT,
        portfolio_cls=_FakePortfolioT,
        strategy_cls=_FakeStrategyT,
        runtime_config=rt,
    )
    trader._audit_closed = True

    # Assert STORAGE_EXPORT_CSV is now present in config
    assert hasattr(trader.config, "STORAGE_EXPORT_CSV"), (
        "_build_live_config_namespace must include STORAGE_EXPORT_CSV"
    )

    # Drive one synthetic market bar — must not raise AttributeError or any other error
    bar = MarketEvent(
        time="2024-01-01T00:00:00",
        symbol="ETH/USDT",
        open=2000.0,
        high=2010.0,
        low=1990.0,
        close=2005.0,
        volume=100.0,
    )
    trader.handle_market_event(bar)  # must not raise


# ── Fix 6 (measured shadow parity): injected ratio forwarded to the gate ──────


def test_measured_shadow_parity_ratio_forwarded_to_gate(monkeypatch):
    """An operator-injected measured shadow parity ratio must flow into
    enforce_live_readiness_from_files (so the shadow gate is genuinely wired, not
    permanently fail-closed)."""
    captured: dict = {}

    def _fake_enforce(*, mode, stale_minutes, env, go_live_stage=None, shadow_parity_ratio=None):
        captured.update(go_live_stage=go_live_stage, shadow_parity_ratio=shadow_parity_ratio)
        return {"recommended_action": "shadow_run_allowed", "status": {}}

    monkeypatch.setattr("lumina_quant.live.trader.enforce_live_readiness_from_files", _fake_enforce)

    class _Cfg:
        GO_LIVE_STAGE = "shadow"
        MODE = "paper"
        SYMBOLS = ["BTC/USDT"]
        DECISION_CADENCE_SECONDS = 1
        HEARTBEAT_INTERVAL_SEC = 1
        RECONCILIATION_INTERVAL_SEC = 5
        POSTGRES_DSN = ""
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
        EXCHANGE = {}
        STORAGE_EXPORT_CSV = False
        MAIN_LOOP_ERROR_RETRY_LIMIT = 3
        MAIN_LOOP_ERROR_WINDOW_SECONDS = 60
        ORDER_STATE_SOURCE = "polling"
        MARKET_DATA_SOURCE = "committed"
        RECONCILIATION_POLL_FALLBACK_ENABLED = True
        STARTUP_RECONCILIATION_HARD_FAIL = False

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _n: MagicMock())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", _Cfg)
    monkeypatch.setattr(
        "lumina_quant.live.trader.StateManager",
        lambda: SimpleNamespace(load_state=lambda: {}, save_state=lambda _: None),
    )
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: MagicMock())
    monkeypatch.setattr(
        "lumina_quant.live.trader.NotificationManager", lambda *_a, **_k: MagicMock()
    )

    class _FakeAuditStore5:
        def __init__(self, _dsn):
            pass

        def start_run(self, **_kwargs):
            return "run-1"

        def log_risk_event(self, *_, **__):
            pass

        def close(self, *_, **__):
            pass

    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _FakeAuditStore5)
    monkeypatch.setattr(
        "lumina_quant.live.trader.get_exchange",
        lambda _c: SimpleNamespace(load_markets=lambda: {"BTC/USDT": {"symbol": "BTC/USDT"}}),
    )

    trader = LiveTrader(
        symbol_list=["BTC/USDT"],
        data_handler_cls=lambda *_, **__: SimpleNamespace(
            symbol_list=["BTC/USDT"],
            continue_backtest=True,
            get_latest_bar_value=lambda s, v: 0.0,
        ),
        execution_handler_cls=lambda *_, **__: SimpleNamespace(
            set_order_state_callback=lambda _: None,
        ),
        portfolio_cls=lambda *_, **__: SimpleNamespace(
            current_positions={},
            current_holdings={"total": 0.0},
            get_state=lambda: {},
            set_state=lambda _: None,
            trading_frozen=False,
        ),
        strategy_cls=MagicMock(return_value=MagicMock(decision_cadence_seconds=1)),
    )
    trader._audit_closed = True

    # Fail-closed default: no measurement injected yet.
    assert trader._measured_shadow_parity_ratio is None

    trader.set_measured_shadow_parity_ratio(0.97)
    trader._run_live_readiness_gate()

    assert captured.get("go_live_stage") == "shadow"
    assert captured.get("shadow_parity_ratio") == pytest.approx(0.97)
