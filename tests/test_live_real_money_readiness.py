"""Real-money-readiness live-safety fixes (audit 2026-07-06).

Covers the TRADER-workstream findings:
- banner/routing: one shared endpoint resolver (resolve_live_endpoint)
- C1: FLATTEN_FAILED on exhausted retries, reference-less MARKET escape, never-latch
- C4: marginBalance equity fetch + periodic reconciliation fold + divergence freeze
- C6: market-data-silence watchdog block + escalation + recovery
- M1: state fingerprint accept/reject + RiskManager state + hard-halt restore
- M2: freeze provenance (risk_recovered clears only the risk hold)
- M6: active_orders alias so the engine per-row sweep early-outs
- O5: save_state failure alert + freeze after N
- config namespace exposes the new gated safety keys, all default-disabled
"""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.live.trader import (
    LiveTrader,
    _build_live_config_namespace,
    resolve_live_endpoint,
)


def _noop(*_args, **_kwargs):
    return None


class _Audit:
    def __init__(self):
        self.events = []

    def log_risk_event(self, run_id, reason, details=None):
        self.events.append((str(reason), dict(details or {})))

    def reasons(self):
        return [reason for reason, _ in self.events]


class _Notifier:
    def __init__(self):
        self.messages = []

    def send_message(self, msg):
        self.messages.append(str(msg))


def _mk_trader() -> LiveTrader:
    """Minimal ``__new__``-built trader with only the shared attrs the pure methods read."""
    t = LiveTrader.__new__(LiveTrader)
    t._audit_closed = True
    t.logger = SimpleNamespace(info=_noop, error=_noop, warning=_noop, debug=_noop, critical=_noop)
    t.strategy_name = "StratA"
    t.symbol_list = ["BTC/USDT"]
    t.live_mode = "paper"
    t.config = SimpleNamespace(
        EXCHANGE_ID="binance",
        MARKET_TYPE="future",
        EXCHANGE={"name": "binance", "market_type": "future"},
    )
    t.require_state_fingerprint = False
    t.state_fingerprint_max_age_sec = 86400.0
    t.run_id = "run-1"
    t.audit_store = _Audit()
    t.notifier = _Notifier()
    t._freeze_reasons = set()
    t.portfolio = SimpleNamespace(
        trading_frozen=False,
        current_positions={"BTC/USDT": 0.0},
        current_holdings={"total": 1000.0, "cash": 1000.0},
    )
    t._hard_halt_active = False
    t._hard_halt_reason = ""
    t._save_state_consecutive_failures = 0
    t.save_state_failure_halt_threshold = 3
    return t


# ── banner / routing: one shared endpoint resolver ───────────────────────────


def test_resolve_live_endpoint_matrix():
    assert resolve_live_endpoint(go_live_stage="testnet", mode="paper") == "testnet"
    assert resolve_live_endpoint(go_live_stage="shadow", mode="paper") == "testnet"
    assert resolve_live_endpoint(go_live_stage="canary", mode="real") == "production"
    assert resolve_live_endpoint(go_live_stage="full", mode="real") == "production"
    # The previously-inconsistent case: testnet stage + testnet flag False + real mode
    # now routes to PRODUCTION (matches the banner) instead of silently to testnet.
    assert (
        resolve_live_endpoint(go_live_stage="testnet", mode="real", testnet=False) == "production"
    )
    assert resolve_live_endpoint(go_live_stage="testnet", mode="paper", testnet=False) == "testnet"


def test_cli_banner_matches_endpoint_resolver():
    from lumina_quant.cli.live import _resolve_live_banner

    cfg = SimpleNamespace(go_live_stage="testnet", mode="real", testnet=False)
    banner = _resolve_live_banner(cfg)
    assert banner["endpoint_is_prod"] is True
    assert "PRODUCTION" in str(banner["endpoint"])
    # And the exchange routing (IS_TESTNET) agrees for the same composition.
    raw = {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "live": {"mode": "paper", "go_live_stage": "shadow"},
    }
    view = _build_live_config_namespace(build_runtime_config(raw, env={}), symbols=["BTC/USDT"])
    assert view.IS_TESTNET is True  # shadow → testnet endpoint (unchanged default)


# ── config namespace: new gated safety keys, all default-disabled ────────────


def test_namespace_exposes_new_safety_keys_defaulted_off():
    raw = {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "live": {"mode": "paper", "go_live_stage": "testnet"},
    }
    view = _build_live_config_namespace(build_runtime_config(raw, env={}), symbols=["BTC/USDT"])
    assert view.MAX_ORDERS_PER_MINUTE == 0
    assert view.MAX_DAILY_NOTIONAL_TURNOVER_PCT == 0.0
    assert view.MAX_POSITION_AGE_HOURS == 0.0
    assert view.ENFORCE_GROSS_EXPOSURE_IN_HEDGE is False
    assert view.MAX_BBO_SPREAD_BPS_AT_SUBMIT == 0.0
    assert view.MAX_ESTIMATED_ONE_WAY_SLIPPAGE_BPS == 0.0
    assert view.REQUIRE_BBO_FOR_LIMIT_ORDERS is False
    assert view.MAX_LIMIT_PRICE_BAND_PCT == 0.0
    assert view.EQUITY_RECONCILIATION_INTERVAL_SEC == 0.0
    assert view.MARKET_DATA_SILENCE_TIMEOUT_SEC == 0.0
    assert view.REQUIRE_STATE_FINGERPRINT is False
    assert view.REAL_MODE_MANAGED_PROTECTIVE_STOPS is False


# ── M2: freeze provenance ────────────────────────────────────────────────────


def test_risk_recovered_does_not_clear_drift_freeze():
    t = _mk_trader()
    # A reconciliation-drift freeze is active.
    t._set_trade_freeze(enabled=True, reason="reconciliation_drift", provenance="drift")
    assert t.portfolio.trading_frozen is True
    # Risk also breaches (freeze already up): risk provenance is recorded silently.
    t._add_freeze_provenance("risk", reason="drawdown")
    assert t._freeze_reasons == {"drift", "risk"}
    # Risk recovers → clear ONLY the risk hold; the drift hold must persist.
    t._clear_freeze_provenance("risk", reason="risk_recovered")
    assert t.portfolio.trading_frozen is True
    assert t._freeze_reasons == {"drift"}
    # Drift finally clears → freeze lifts.
    t._clear_freeze_provenance("drift", reason="drift_cleared")
    assert t.portfolio.trading_frozen is False


def test_default_risk_only_freeze_lifts_normally():
    """Byte-identical default path: a lone risk freeze lifts on risk_recovered."""
    t = _mk_trader()
    t._add_freeze_provenance("risk", reason="drawdown")
    assert t.portfolio.trading_frozen is True
    t._clear_freeze_provenance("risk", reason="risk_recovered")
    assert t.portfolio.trading_frozen is False
    assert t._freeze_reasons == set()


# ── C1: FLATTEN failure + never-latch ────────────────────────────────────────


def _flatten_trader(*, positions, inflight, attempts, started_ago):
    t = _mk_trader()
    t.events = queue.Queue()
    t.config = SimpleNamespace(
        FLATTEN_MAX_RETRIES=3,
        FLATTEN_RETRY_SECONDS=30.0,
        FLATTEN_ESCALATE_TO_MARKET=False,
        POSITION_MODE="ONE_WAY",
    )
    t.portfolio.current_positions = dict(positions)
    t._flatten_inflight = inflight
    t._flatten_attempts = attempts
    t._flatten_started_at = (time.time() - started_ago) if started_ago is not None else None
    t._flatten_failed_emitted = False
    t.risk_manager = SimpleNamespace(
        evaluate_portfolio_risk=lambda _p: (False, "hard_drawdown", "FLATTEN", {})
    )
    return t


def test_flatten_failed_emitted_when_retries_exhausted_with_positions():
    t = _flatten_trader(positions={"BTC/USDT": 1.0}, inflight=True, attempts=3, started_ago=1000.0)
    t._evaluate_risk_guards()
    assert "FLATTEN_FAILED" in t.audit_store.reasons()
    assert t._flatten_failed_emitted is True
    # Idempotent: a second cycle does not re-page.
    before = t.audit_store.reasons().count("FLATTEN_FAILED")
    t._evaluate_risk_guards()
    assert t.audit_store.reasons().count("FLATTEN_FAILED") == before


def test_flatten_inflight_cleared_when_flat():
    t = _flatten_trader(positions={"BTC/USDT": 0.0}, inflight=True, attempts=2, started_ago=5.0)
    t._evaluate_risk_guards()
    assert t._flatten_inflight is False
    assert t._flatten_attempts == 0
    assert "FLATTEN_FAILED" not in t.audit_store.reasons()


def test_referenceless_leg_escalates_to_market_when_fallback_enabled():
    t = _mk_trader()
    t.events = queue.Queue()
    t.config = SimpleNamespace(
        DEFAULT_ORDER_TYPE="LMT",
        ALLOW_MARKET_ORDERS=True,
        LIMIT_PRICE_MODE="one_tick_worse",
        LIMIT_PRICE_OFFSET_TICKS=1,
        LIMIT_TIME_IN_FORCE="GTC",
    )
    # No reference price available → LMT policy cannot price the leg.
    t.data_handler = SimpleNamespace(get_latest_bar_value=lambda _s, _f: 0.0)
    t.exchange = SimpleNamespace()

    # Without the fallback the reference-less leg is silently skipped (legacy).
    assert (
        t._queue_reduce_only_order(
            symbol="BTC/USDT",
            quantity=1.0,
            direction="SELL",
            position_side="LONG",
            market_fallback_referenceless=False,
        )
        is False
    )
    assert t.events.empty()

    # With the fallback it is covered by a reduce-only MARKET escape.
    assert (
        t._queue_reduce_only_order(
            symbol="BTC/USDT",
            quantity=1.0,
            direction="SELL",
            position_side="LONG",
            market_fallback_referenceless=True,
        )
        is True
    )
    event = t.events.get_nowait()
    assert event.reduce_only is True
    assert str(event.order_type).upper() in {"MKT", "MARKET"}
    assert event.metadata.get("risk_flatten_escalation") is True
    assert event.metadata.get("reference_price_unavailable") is True


# ── C4: equity fetch + periodic reconciliation ───────────────────────────────


def test_fetch_exchange_equity_prefers_margin_balance():
    t = _mk_trader()
    t.exchange = SimpleNamespace(get_margin_balance=lambda: 12_345.0)
    snap = t._fetch_exchange_equity()
    assert snap["ok"] is True
    assert snap["equity"] == 12_345.0
    assert snap["source"] == "get_margin_balance"


def test_fetch_exchange_equity_unavailable_is_not_ok():
    t = _mk_trader()
    t.exchange = SimpleNamespace()  # no marginBalance-capable accessor
    snap = t._fetch_exchange_equity()
    assert snap["ok"] is False
    assert snap["reason"] == "no_margin_balance_source"


def test_reconcile_equity_folds_delta_and_freezes_on_divergence():
    t = _mk_trader()
    t.equity_reconciliation_interval_sec = 10.0
    t.equity_reconciliation_divergence_pct = 0.10
    t._last_equity_reconciliation_monotonic = 0.0
    t._equity_reconcile_unavailable_alerted = False
    t.runtime_cache = SimpleNamespace(update_account=_noop)
    t.portfolio.current_holdings = {"total": 1000.0, "cash": 1000.0}
    t.exchange = SimpleNamespace(get_margin_balance=lambda: 800.0)  # 20% drop in one interval

    t._reconcile_equity(force=True)

    assert t.portfolio.current_holdings["total"] == 800.0
    assert t.portfolio.current_holdings["cash"] == pytest.approx(800.0)  # 1000 + (800 - 1000)
    assert t.portfolio.trading_frozen is True
    assert "equity_divergence" in t._freeze_reasons
    assert "EQUITY_RECONCILED" in t.audit_store.reasons()


def test_reconcile_equity_within_tolerance_clears_divergence_hold():
    t = _mk_trader()
    t.equity_reconciliation_interval_sec = 10.0
    t.equity_reconciliation_divergence_pct = 0.10
    t._last_equity_reconciliation_monotonic = 0.0
    t._equity_reconcile_unavailable_alerted = False
    t.runtime_cache = SimpleNamespace(update_account=_noop)
    t.portfolio.current_holdings = {"total": 1000.0, "cash": 1000.0}
    t.portfolio.trading_frozen = True
    t._freeze_reasons = {"equity_divergence"}
    t.exchange = SimpleNamespace(get_margin_balance=lambda: 1001.0)  # tiny drift

    t._reconcile_equity(force=True)

    assert t.portfolio.trading_frozen is False
    assert "equity_divergence" not in t._freeze_reasons


def test_reconcile_equity_disabled_is_noop():
    t = _mk_trader()
    t.equity_reconciliation_interval_sec = 0.0  # disabled (default)
    t.portfolio.current_holdings = {"total": 1000.0, "cash": 1000.0}
    called = []
    t.exchange = SimpleNamespace(get_margin_balance=lambda: called.append(1) or 500.0)

    t._reconcile_equity(force=True)

    assert called == []  # never even queries the exchange
    assert t.portfolio.current_holdings["total"] == 1000.0


# ── C6: market-data-silence watchdog ─────────────────────────────────────────


def _silence_trader():
    t = _mk_trader()
    t.market_data_silence_timeout_sec = 10.0
    t.market_data_silence_freeze_intervals = 3
    t.materialized_staleness_alert_cooldown_seconds = 0
    t._data_silence_block_active = False
    t._data_silence_last_alert_monotonic = 0.0
    return t


def test_market_data_silence_blocks_and_escalates():
    t = _silence_trader()
    # Last window 40s ago: age 40 > timeout*intervals (10*3=30) → block + freeze.
    t._last_market_window_monotonic = time.monotonic() - 40.0
    t._check_market_data_silence()
    assert t._data_silence_block_active is True
    assert t.portfolio.trading_frozen is True
    assert "data_silence" in t._freeze_reasons
    assert "MARKET_DATA_SILENCE" in t.audit_store.reasons()


def test_market_data_silence_recovers_and_clears_freeze():
    t = _silence_trader()
    t._data_silence_block_active = True
    t.portfolio.trading_frozen = True
    t._freeze_reasons = {"data_silence"}
    t._last_market_window_monotonic = time.monotonic()  # fresh
    t._check_market_data_silence()
    assert t._data_silence_block_active is False
    assert t.portfolio.trading_frozen is False
    assert "data_silence" not in t._freeze_reasons


def test_market_data_silence_disabled_is_noop():
    t = _silence_trader()
    t.market_data_silence_timeout_sec = 0.0  # disabled (default)
    t._last_market_window_monotonic = time.monotonic() - 100_000.0
    t._check_market_data_silence()
    assert t._data_silence_block_active is False
    assert t.portfolio.trading_frozen is False


# ── O5: save_state failure alert + freeze after N ────────────────────────────


def test_save_state_failure_freezes_after_threshold():
    t = _mk_trader()
    t.save_state_failure_halt_threshold = 2
    t._save_state_consecutive_failures = 0
    t.state_manager = SimpleNamespace(save_state=lambda _s: False)  # genuine write failure
    t.risk_manager = SimpleNamespace(get_state=lambda: {})
    t.strategy = SimpleNamespace(get_state=lambda: {})
    t.runtime_cache = SimpleNamespace(snapshot=lambda: {})
    t.outbox_events = []
    t.portfolio.get_state = lambda: {}

    t._save_state()  # failure 1 — alert, no freeze yet
    assert t.portfolio.trading_frozen is False
    assert t._save_state_consecutive_failures == 1

    t._save_state()  # failure 2 — threshold reached → freeze
    assert t.portfolio.trading_frozen is True
    assert "save_state" in t._freeze_reasons
    assert "SAVE_STATE_FAILED" in t.audit_store.reasons()


def test_save_state_success_resets_failure_counter():
    t = _mk_trader()
    t._save_state_consecutive_failures = 5
    t.state_manager = SimpleNamespace(save_state=lambda _s: True)
    t.risk_manager = SimpleNamespace(get_state=lambda: {})
    t.strategy = SimpleNamespace(get_state=lambda: {})
    t.runtime_cache = SimpleNamespace(snapshot=lambda: {})
    t.outbox_events = []
    t.portfolio.get_state = lambda: {}

    t._save_state()
    assert t._save_state_consecutive_failures == 0


def test_save_state_none_return_is_treated_as_success():
    """Test doubles return None from save_state — must NOT count as a failure."""
    t = _mk_trader()
    t._save_state_consecutive_failures = 0
    t.state_manager = SimpleNamespace(save_state=lambda _s: None)
    t.risk_manager = SimpleNamespace(get_state=lambda: {})
    t.strategy = SimpleNamespace(get_state=lambda: {})
    t.runtime_cache = SimpleNamespace(snapshot=lambda: {})
    t.outbox_events = []
    t.portfolio.get_state = lambda: {}

    t._save_state()
    assert t._save_state_consecutive_failures == 0
    assert t.portfolio.trading_frozen is False


# ── M1: state fingerprint + risk/hard-halt persistence ───────────────────────


def test_paper_mode_rejects_incomplete_legacy_state():
    """Partial legacy state cannot bypass transactional restore in paper mode."""
    t = _mk_trader()  # live_mode=paper, require_state_fingerprint=False
    applied = []
    t.state_manager = SimpleNamespace(load_state=lambda: {"portfolio": {"x": 1}})
    t.portfolio = SimpleNamespace(set_state=lambda s: applied.append(s), trading_frozen=False)
    t.strategy = SimpleNamespace(set_state=_noop)
    t.runtime_cache = SimpleNamespace(restore=_noop, open_orders={})
    t.execution_handler = SimpleNamespace()
    t.risk_manager = SimpleNamespace()
    t.outbox_events = []

    with pytest.raises(RuntimeError, match="incomplete"):
        t._load_state()
    assert applied == []


def test_enforced_mode_refuses_foreign_state():
    t = _mk_trader()
    t.live_mode = "real"
    t.require_state_fingerprint = True
    foreign = {
        "portfolio": {"x": 1},
        "fingerprint": {
            "strategy_name": "OTHER",
            "symbols": ["ETH/USDT"],
            "exchange": "binance",
            "market_type": "future",
            "mode": "real",
            "saved_at": time.time(),
        },
    }
    applied = []
    t.state_manager = SimpleNamespace(load_state=lambda: foreign)
    t.portfolio = SimpleNamespace(set_state=lambda s: applied.append(s), trading_frozen=False)
    t.strategy = SimpleNamespace(set_state=_noop)
    t.runtime_cache = SimpleNamespace(restore=_noop, open_orders={})
    t.execution_handler = SimpleNamespace()
    t.risk_manager = SimpleNamespace(set_state=_noop)
    t.outbox_events = []

    with pytest.raises(RuntimeError, match="fingerprint rejected"):
        t._load_state()
    assert applied == []  # foreign state refused
    assert "STATE_FINGERPRINT_REJECTED" in t.audit_store.reasons()


def test_state_roundtrip_restores_risk_and_hard_halt_in_enforced_mode():
    saved: dict = {}
    restored_risk: dict = {}

    src = _mk_trader()
    src.live_mode = "real"
    src.require_state_fingerprint = True
    src.state_manager = SimpleNamespace(
        save_state=lambda s: (saved.clear(), saved.update(s), True)[-1],
    )
    src.risk_manager = SimpleNamespace(
        get_state=lambda: {"consecutive_loss_count": 4, "hard_halt": True}
    )
    src.strategy = SimpleNamespace(get_state=lambda: {})
    src.runtime_cache = SimpleNamespace(snapshot=lambda: {})
    src.outbox_events = []
    src.portfolio.get_state = lambda: {}
    src._hard_halt_active = True
    src._hard_halt_reason = "consecutive_loss"

    src._save_state()
    assert saved["hard_halt"]["active"] is True
    assert saved["risk"] == {"consecutive_loss_count": 4, "hard_halt": True}
    assert saved["fingerprint"]["strategy_name"] == "StratA"

    # A fresh identical trader adopts the saved state (matching fingerprint, fresh age).
    dst = _mk_trader()
    dst.live_mode = "real"
    dst.require_state_fingerprint = True
    dst.state_manager = SimpleNamespace(load_state=lambda: dict(saved))
    dst.risk_manager = SimpleNamespace(
        set_state=lambda s: (restored_risk.clear(), restored_risk.update(s)),
        get_state=lambda: dict(restored_risk),
    )
    strategy_state: dict = {}
    dst.strategy = SimpleNamespace(
        set_state=lambda s: (strategy_state.clear(), strategy_state.update(s)),
        get_state=lambda: dict(strategy_state),
    )
    runtime_state: dict = {}
    dst.runtime_cache = SimpleNamespace(
        restore=lambda s: (runtime_state.clear(), runtime_state.update(s)),
        snapshot=lambda: dict(runtime_state),
        open_orders={},
    )
    dst.execution_handler = SimpleNamespace()
    portfolio_state: dict = {}
    dst.portfolio = SimpleNamespace(
        set_state=lambda s: (portfolio_state.clear(), portfolio_state.update(s)),
        get_state=lambda: dict(portfolio_state),
        trading_frozen=False,
    )
    dst.outbox_events = []
    dst._hard_halt_active = False
    dst._hard_halt_reason = ""

    dst._load_state()
    assert restored_risk.get("consecutive_loss_count") == 4
    assert dst._hard_halt_active is True
    assert dst.portfolio.trading_frozen is True


def test_transactional_state_restore_rolls_back_before_order_rehydration():
    t = _mk_trader()
    t.live_mode = "real"
    t.require_state_fingerprint = True
    portfolio_state = {"old": 1}
    strategy_state = {"old": 2}
    risk_state = {"old": 3}
    runtime_state = {"old": 4}
    rehydrated: list[object] = []
    payload = {
        "portfolio": {"new": 1},
        "strategy": {"new": 2},
        "runtime_cache": {"new": 4},
        "outbox_events": [{"new": 5}],
        "risk": {"new": 3},
        "hard_halt": {"active": False, "reason": ""},
        "fingerprint": t._state_fingerprint(),
    }
    t.state_manager = SimpleNamespace(load_state=lambda: payload)
    t.portfolio = SimpleNamespace(
        set_state=lambda s: (portfolio_state.clear(), portfolio_state.update(s)),
        get_state=lambda: dict(portfolio_state),
        trading_frozen=False,
    )

    def reject_strategy(state):
        _ = state
        raise ValueError("reject")

    t.strategy = SimpleNamespace(
        set_state=reject_strategy,
        get_state=lambda: dict(strategy_state),
    )
    t.risk_manager = SimpleNamespace(
        set_state=lambda s: (risk_state.clear(), risk_state.update(s)),
        get_state=lambda: dict(risk_state),
    )
    t.runtime_cache = SimpleNamespace(
        restore=lambda s: (runtime_state.clear(), runtime_state.update(s)),
        snapshot=lambda: dict(runtime_state),
        open_orders={},
    )
    t.execution_handler = SimpleNamespace(rehydrate_orders=lambda rows: rehydrated.append(rows))
    t.outbox_events = [{"old": 5}]

    with pytest.raises(RuntimeError, match="startup aborted"):
        t._load_state()

    assert portfolio_state == {"old": 1}
    assert strategy_state == {"old": 2}
    assert risk_state == {"old": 3}
    assert runtime_state == {"old": 4}
    assert t.outbox_events == [{"old": 5}]
    assert rehydrated == []


def test_state_fingerprint_ok_matrix():
    t = _mk_trader()
    t.live_mode = "real"  # enforced
    good = {"fingerprint": t._state_fingerprint()}
    assert t._state_fingerprint_ok(good) is True

    stale = {"fingerprint": {**t._state_fingerprint(), "saved_at": time.time() - 999_999.0}}
    assert t._state_fingerprint_ok(stale) is False

    mismatched = {"fingerprint": {**t._state_fingerprint(), "strategy_name": "Other"}}
    assert t._state_fingerprint_ok(mismatched) is False

    assert t._state_fingerprint_ok({}) is False  # required but absent

    # Not enforced (paper) accepts anything.
    t.live_mode = "paper"
    t.require_state_fingerprint = False
    assert t._state_fingerprint_ok({}) is True


# ── M6: active_orders alias enables the engine's empty-book early-out ─────────


def test_active_orders_aliased_to_tracked_orders(monkeypatch):
    rt = build_runtime_config(
        {
            "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
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
            },
        },
        env={"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"},
    )

    class _ExecWithTracked:
        def __init__(self, *_a, **_k):
            self.tracked_orders = {}

        def set_order_state_callback(self, _cb):
            return None

    class _Portfolio:
        def __init__(self, *_a, **_k):
            self.current_positions = {"BTC/USDT": 0.0}
            self.current_holdings = {"total": 0.0, "cash": 0.0}

        @staticmethod
        def get_state():
            return {}

        @staticmethod
        def set_state(_s):
            return None

    monkeypatch.setattr("lumina_quant.live.trader.setup_logging", lambda _n: MagicMock())
    monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", None)
    monkeypatch.setattr(
        "lumina_quant.live.trader.StateManager",
        lambda: SimpleNamespace(load_state=lambda: {}, save_state=lambda _s: True),
    )
    monkeypatch.setattr("lumina_quant.live.trader.RuntimeCache", lambda: MagicMock())
    monkeypatch.setattr(
        "lumina_quant.live.trader.NotificationManager", lambda *_a, **_k: MagicMock()
    )

    class _Audit2:
        def __init__(self, _dsn):
            pass

        def start_run(self, **_k):
            return "run-1"

        def close(self, *_a, **_k):
            return None

    monkeypatch.setattr("lumina_quant.live.trader.AuditStore", _Audit2)
    monkeypatch.setattr(
        "lumina_quant.live.trader.get_exchange",
        lambda _c: SimpleNamespace(load_markets=lambda: {"BTC/USDT": {"symbol": "BTC/USDT"}}),
    )

    trader = LiveTrader(
        symbol_list=["BTC/USDT"],
        data_handler_cls=lambda *_a, **_k: SimpleNamespace(
            symbol_list=["BTC/USDT"], continue_backtest=True, get_latest_bar_value=lambda s, v: 0.0
        ),
        execution_handler_cls=_ExecWithTracked,
        portfolio_cls=_Portfolio,
        strategy_cls=MagicMock(return_value=MagicMock(decision_cadence_seconds=1)),
        runtime_config=rt,
    )
    trader._audit_closed = True

    # The alias must be the SAME dict object so it tracks working orders live.
    assert trader.execution_handler.active_orders is trader.execution_handler.tracked_orders
    assert not trader.execution_handler.active_orders  # empty book → engine early-out fires
