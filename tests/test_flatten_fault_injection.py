"""V4 — fault-injection on the FLATTEN chain under a failing exchange.

Drives the REAL kill-switch (``LiveTrader._evaluate_risk_guards`` ->
``_flatten_all_positions`` -> ``LiveExecutionHandler.execute_order``) against
the scripted fake exchange from :mod:`tests.integration.live_harness`, injecting
the failure modes the audit calls out:

* reduce-only submit returns 502 then succeeds  -> must converge to flat,
* reduce-only order rests unfilled past ORDER_TIMEOUT on every retry
  -> retries must exhaust and raise a loud FLATTEN_FAILED alert,
* reduce-only rejected -2022 for the whole flatten
  -> retries must exhaust and raise a loud FLATTEN_FAILED alert.

The safety contract asserted throughout: the automated de-risk either converges
to flat, or it pages the operator (FLATTEN_FAILED) — it never silently latches
"in progress" forever while positions stay open.
"""

from __future__ import annotations

import pytest

from tests.integration.live_harness import (
    RejectExchangeError,
    RetryableExchangeError,
    ScriptedFakeExchange,
    build_live_trader,
    drive_events,
    open_long_position,
)


def _arm_flatten_breach(trader, exchange, qty: float = 1.0) -> None:
    """Seed an open LONG and force the hard-drawdown FLATTEN tier to fire."""
    open_long_position(trader, "BTC/USDT", qty=qty, price=100.0)
    # Default config is HEDGE -> flatten reads side-aware legs from the exchange.
    exchange.position_legs = {"BTC/USDT": {"LONG": float(qty), "SHORT": 0.0}}
    trader.portfolio.day_start_equity = 10_000.0
    trader.portfolio.current_holdings["total"] = 8_000.0
    trader.risk_manager.hard_drawdown_flatten_pct = 0.10


def _age_flatten_clock(trader, seconds: float = 5.0) -> None:
    """Make the in-flight flatten's retry timer look overdue."""
    if trader._flatten_started_at is not None:
        trader._flatten_started_at -= float(seconds)


# --------------------------------------------------------------------------- #
# 1. reduce-only submit 502-then-success -> converges to flat                 #
# --------------------------------------------------------------------------- #
def test_flatten_reduce_only_502_then_success_converges(monkeypatch):
    monkeypatch.setattr("lumina_quant.live.execution_live.time.sleep", lambda *_a, **_k: None)

    exchange = ScriptedFakeExchange()
    trader, exchange, audit, _notifier = build_live_trader(exchange=exchange)
    _arm_flatten_breach(trader, exchange)

    # First reduce-only submit gets a 502; the idempotent retry re-sends and fills.
    exchange.submit_faults.append(RetryableExchangeError(status_code=502))

    trader._evaluate_risk_guards()
    assert trader._flatten_inflight is True
    result = drive_events(trader)
    assert result["errors"] == []

    # Converged to flat, latch cleared on the next tick, no operator page.
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(0.0)
    trader._evaluate_risk_guards()
    assert trader._flatten_inflight is False
    assert "FLATTEN_FAILED" not in audit.reasons()
    # exactly one order landed despite the retry (no double reduce-only send)
    assert len(exchange.orders) == 1
    assert len(exchange.submit_calls) == 2


# --------------------------------------------------------------------------- #
# 2. reduce-only rests unfilled past ORDER_TIMEOUT on every retry -> alert     #
# --------------------------------------------------------------------------- #
def test_flatten_unfilled_rest_exhausts_retries_and_pages(monkeypatch):
    monkeypatch.setattr("lumina_quant.live.execution_live.time.sleep", lambda *_a, **_k: None)

    exchange = ScriptedFakeExchange(fill_mode="rest")
    trader, exchange, audit, notifier = build_live_trader(
        exchange=exchange,
        config_overrides={"FLATTEN_MAX_RETRIES": 2, "FLATTEN_RETRY_SECONDS": 1.0},
    )
    _arm_flatten_breach(trader, exchange)

    def _flatten_tick():
        trader._evaluate_risk_guards()
        drive_events(trader)  # reduce-only order rests (never fills)
        # age the resting tracked order past ORDER_TIMEOUT and poll -> cancel path
        for entry in list(trader.execution_handler.tracked_orders.values()):
            entry["created_at"] -= 100.0
        trader.execution_handler.check_open_orders(None)

    _flatten_tick()  # attempt 1
    assert trader._flatten_inflight is True
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(1.0)

    _age_flatten_clock(trader)
    _flatten_tick()  # attempt 2 (retry)
    assert trader._flatten_attempts == 2

    _age_flatten_clock(trader)
    trader._evaluate_risk_guards()  # attempts exhausted -> FLATTEN_FAILED

    assert "FLATTEN_FAILED" in audit.reasons()
    assert any("FLATTEN FAILED" in m for m in notifier.messages)
    # positions never converged and the timed-out orders were actually canceled
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(1.0)
    assert len(exchange.cancel_calls) >= 1


# --------------------------------------------------------------------------- #
# 3. reduce-only rejected -2022 for the whole flatten -> loud alert            #
# --------------------------------------------------------------------------- #
def test_flatten_reduce_only_2022_reject_pages_operator(monkeypatch):
    monkeypatch.setattr("lumina_quant.live.execution_live.time.sleep", lambda *_a, **_k: None)

    exchange = ScriptedFakeExchange()
    trader, exchange, audit, notifier = build_live_trader(
        exchange=exchange,
        config_overrides={"FLATTEN_MAX_RETRIES": 2, "FLATTEN_RETRY_SECONDS": 1.0},
    )
    _arm_flatten_breach(trader, exchange)
    # Every reduce-only submit is rejected -2022 (non-retryable) for the whole flatten.
    exchange.persistent_submit_error = RejectExchangeError(error_code=-2022)

    # attempt 1
    trader._evaluate_risk_guards()
    result = drive_events(trader)
    assert result["errors"] and all(isinstance(e, RejectExchangeError) for e in result["errors"])
    assert trader._flatten_inflight is True

    # attempt 2 (retry)
    _age_flatten_clock(trader)
    trader._evaluate_risk_guards()
    drive_events(trader)
    assert trader._flatten_attempts == 2

    # retries exhausted -> operator paged
    _age_flatten_clock(trader)
    trader._evaluate_risk_guards()

    assert "FLATTEN_FAILED" in audit.reasons()
    assert any("FLATTEN FAILED" in m for m in notifier.messages)
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(1.0)
    # never actually placed an order (all submits rejected)
    assert len(exchange.orders) == 0


# --------------------------------------------------------------------------- #
# 4. a transient reject that later clears still converges (no false page)      #
# --------------------------------------------------------------------------- #
def test_flatten_recovers_after_transient_reject(monkeypatch):
    monkeypatch.setattr("lumina_quant.live.execution_live.time.sleep", lambda *_a, **_k: None)

    exchange = ScriptedFakeExchange()
    trader, exchange, audit, _notifier = build_live_trader(
        exchange=exchange,
        config_overrides={"FLATTEN_MAX_RETRIES": 3, "FLATTEN_RETRY_SECONDS": 1.0},
    )
    _arm_flatten_breach(trader, exchange)
    # First flatten attempt is rejected; the retry succeeds and fills.
    exchange.submit_faults.append(RejectExchangeError(error_code=-2022))

    trader._evaluate_risk_guards()
    drive_events(trader)  # attempt 1 rejected
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(1.0)

    _age_flatten_clock(trader)
    trader._evaluate_risk_guards()  # attempt 2 re-queues
    drive_events(trader)  # fills now
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(0.0)

    trader._evaluate_risk_guards()  # latch clears
    assert trader._flatten_inflight is False
    assert "FLATTEN_FAILED" not in audit.reasons()
