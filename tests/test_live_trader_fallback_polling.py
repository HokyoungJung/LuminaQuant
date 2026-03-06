from __future__ import annotations

import time
from types import SimpleNamespace

from lumina_quant.live.recovery_reconciliation import ReconciliationOutcome
from lumina_quant.live.trader import LiveTrader


class _AuditStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_risk_event(self, run_id, reason, details):
        _ = run_id
        self.events.append((str(reason), dict(details or {})))


def _build_trader_for_fallback_tests() -> LiveTrader:
    trader = LiveTrader.__new__(LiveTrader)
    trader.order_state_source = "user_stream"
    trader.reconciliation_poll_fallback_enabled = True
    trader.reconciliation_fallback_window_seconds = 60.0
    trader.user_stream_stale_timeout_seconds = 30.0
    trader._fallback_poll_until_monotonic = 0.0
    trader._last_fallback_reason = ""
    trader._user_stream_last_event_monotonic = time.monotonic()
    trader.audit_store = _AuditStore()
    trader.run_id = "test"
    trader._audit_closed = True
    return trader


def test_fallback_polling_is_disabled_when_user_stream_is_healthy():
    trader = _build_trader_for_fallback_tests()
    trader._user_stream_last_event_monotonic = time.monotonic()
    trader._fallback_poll_until_monotonic = time.monotonic() - 1.0
    assert trader._is_fallback_polling_required() is False


def test_fallback_polling_activates_when_user_stream_is_stale():
    trader = _build_trader_for_fallback_tests()
    trader._user_stream_last_event_monotonic = time.monotonic() - 120.0
    assert trader._is_fallback_polling_required() is True
    assert trader._fallback_poll_until_monotonic > time.monotonic()
    assert any(reason == "POLL_FALLBACK_ACTIVATED" for reason, _ in trader.audit_store.events)


def test_startup_reconciliation_timeout_enters_degraded_mode_when_not_hard_fail():
    trader = _build_trader_for_fallback_tests()
    trader.startup_reconciliation_hard_fail = False
    trader.startup_reconciliation_timeout_seconds = 1.0
    trader._startup_reconciliation_complete = False
    trader.portfolio = SimpleNamespace(trading_frozen=True)

    class _RecoveryService:
        @staticmethod
        def startup_converge(timeout_seconds: float):
            _ = timeout_seconds
            return ReconciliationOutcome(
                converged=False,
                elapsed_seconds=1.0,
                tracked_orders=2,
                cache_open_orders=1,
                exchange_open_orders=2,
                exchange_snapshot_ready=False,
            )

    trader._recovery_service = _RecoveryService()
    trader._run_startup_reconciliation_gate()

    assert trader._startup_reconciliation_complete is True
    assert trader.portfolio.trading_frozen is False
    assert any(
        reason == "STARTUP_RECONCILIATION_DEGRADED" for reason, _ in trader.audit_store.events
    )
