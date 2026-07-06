"""Real-money desk-control tests for RiskManager (audit 2026-07-06).

Covers the RISK workstream findings:
  * C3 - NaN/non-finite price reject; limit-price band guard; conservative BUY
    notional priced at max(limit, close).
  * M3 - GROSS per-leg exposure + net-direction metric in HEDGE mode.
  * M4 - order-rate, daily-turnover, and position-age budgets (fail-closed).
  * M1 - get_state/set_state persistence of the consecutive-loss counter and the
    latched hard-halt flag.

Every new control defaults OFF, so the byte-identical guarantees are asserted
alongside the enabled-behaviour tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lumina_quant.risk_manager import RiskManager


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _config(**overrides):
    base = dict(
        MAX_ORDER_VALUE=5000.0,
        MAX_ORDER_NOTIONAL_PCT=0.0,
        MAX_DAILY_LOSS_PCT=0.05,
        MAX_INTRADAY_DRAWDOWN_PCT=0.10,
        MAX_ROLLING_LOSS_PCT_1H=0.20,
        MAX_SYMBOL_EXPOSURE_PCT=0.25,
        MAX_TOTAL_MARGIN_PCT=0.5,
        MAX_TOTAL_NOTIONAL_PCT=0.0,
        FREEZE_NEW_ENTRIES_ON_BREACH=True,
        AUTO_FLATTEN_ON_BREACH=False,
        TARGET_ALLOCATION_MODE="legacy_notional_cap",
        LEVERAGE=1,
        CONSECUTIVE_LOSS_HALT_COUNT=5,
        # New desk controls default OFF unless overridden.
        ENFORCE_GROSS_EXPOSURE_IN_HEDGE=False,
        POSITION_MODE="HEDGE",
        MAX_ORDERS_PER_MINUTE=0,
        MAX_DAILY_NOTIONAL_TURNOVER_PCT=0.0,
        MAX_POSITION_AGE_HOURS=0.0,
        MAX_LIMIT_PRICE_BAND_PCT=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Bars:
    def __init__(self, prices):
        self._prices = dict(prices)

    def get_latest_bar_value(self, symbol, val_type):
        _ = val_type
        return self._prices[symbol]


def _portfolio(
    *,
    equity=10_000.0,
    symbols=("BTC/USDT",),
    holdings=None,
    positions=None,
    legs=None,
    prices=None,
    frozen=False,
    breaker=False,
    rolling_loss=0.0,
    max_position_age_hours=None,
):
    holdings = dict(holdings or {})
    holdings.setdefault("total", equity)
    positions = dict(positions) if positions else dict.fromkeys(symbols, 0.0)

    class _P:
        symbol_list = list(symbols)

        def __init__(self):
            self.current_holdings = holdings
            self.current_positions = positions
            self.current_position_legs = dict(legs or {})
            self.day_start_equity = equity
            self.circuit_breaker_tripped = breaker
            self.trading_frozen = frozen
            if prices is not None:
                self.bars = _Bars(prices)

        @staticmethod
        def get_rolling_loss_pct(window_seconds=3600):
            _ = window_seconds
            return rolling_loss

    p = _P()
    if max_position_age_hours is not None:
        p.get_max_position_age_hours = lambda: max_position_age_hours
    return p


def _order(**kw):
    kw.setdefault("reduce_only", False)
    kw.setdefault("direction", "BUY")
    return SimpleNamespace(**kw)


# --------------------------------------------------------------------------- #
# C3 - NaN backstop hole                                                       #
# --------------------------------------------------------------------------- #
def test_nan_price_is_rejected():
    manager = RiskManager(_config())
    order = _order(symbol="BTC/USDT", quantity=0.04, direction="BUY")
    passed, reason = manager.check_order(order, current_price=float("nan"))
    assert passed is False
    assert "market price" in reason.lower()


def test_inf_price_is_rejected():
    manager = RiskManager(_config())
    order = _order(symbol="BTC/USDT", quantity=0.04, direction="BUY")
    passed, reason = manager.check_order(order, current_price=float("inf"))
    assert passed is False
    assert "market price" in reason.lower()


def test_valid_price_still_passes_without_portfolio():
    manager = RiskManager(_config())
    order = _order(symbol="BTC/USDT", quantity=0.04, direction="BUY")
    passed, reason = manager.check_order(order, current_price=100.0)
    assert passed is True
    assert reason == "Passed"


# --------------------------------------------------------------------------- #
# C3 - fat-finger limit-price band                                            #
# --------------------------------------------------------------------------- #
def test_100x_limit_rejected_when_band_set():
    # Micro-repro: BUY LMT at 100x the reference must reject once the band is on.
    manager = RiskManager(_config(MAX_LIMIT_PRICE_BAND_PCT=0.1))
    order = _order(symbol="BTC/USDT", quantity=0.001, direction="BUY", price=5_000_000.0)
    passed, reason = manager.check_order(order, current_price=50_000.0)
    assert passed is False
    assert "band" in reason.lower()


def test_limit_within_band_passes():
    manager = RiskManager(_config(MAX_LIMIT_PRICE_BAND_PCT=0.1))
    order = _order(symbol="BTC/USDT", quantity=0.001, direction="BUY", price=52_000.0)
    passed, reason = manager.check_order(order, current_price=50_000.0)
    assert passed is True
    assert reason == "Passed"


def test_band_disabled_is_byte_identical_no_op():
    # Band OFF (default 0.0): the 100x limit is NOT band-rejected. With a tiny qty
    # the conservative notional still clears MAX_ORDER_VALUE, so the order passes.
    manager = RiskManager(_config(MAX_LIMIT_PRICE_BAND_PCT=0.0))
    order = _order(symbol="BTC/USDT", quantity=0.0001, direction="BUY", price=10_000.0)
    passed, reason = manager.check_order(order, current_price=100.0)
    assert passed is True
    assert reason == "Passed"


def test_reduce_only_exit_exempt_from_band_guard():
    # A reduce-only exit with an off-band limit must NOT be blocked by the band
    # guard (de-risk priority; reduce-only cannot increase exposure).
    manager = RiskManager(_config(MAX_LIMIT_PRICE_BAND_PCT=0.1))
    portfolio = _portfolio(equity=1_000_000.0, positions={"BTC/USDT": 1.0})
    order = _order(
        symbol="BTC/USDT",
        quantity=0.001,
        direction="SELL",
        reduce_only=True,
        price=5_000_000.0,
    )
    passed, reason = manager.check_order(order, current_price=50_000.0, portfolio=portfolio)
    assert passed is True
    assert "reduce-only" in reason.lower()


# --------------------------------------------------------------------------- #
# C3 - conservative BUY notional priced at max(limit, close)                   #
# --------------------------------------------------------------------------- #
def test_buy_notional_priced_at_limit_when_above_close():
    manager = RiskManager(_config(MAX_ORDER_VALUE=5000.0))
    # 30 * 100 = 3000 (< cap) at close, but 30 * 200 = 6000 (> cap) at the limit.
    order = _order(symbol="BTC/USDT", quantity=30.0, direction="BUY", price=200.0)
    passed, reason = manager.check_order(order, current_price=100.0)
    assert passed is False
    assert "exceeds limit" in reason


def test_sell_notional_not_repriced_to_limit():
    manager = RiskManager(_config(MAX_ORDER_VALUE=5000.0))
    # A SELL limit above close keeps the (lower) reference notional -> passes.
    order = _order(symbol="BTC/USDT", quantity=30.0, direction="SELL", price=200.0)
    passed, reason = manager.check_order(order, current_price=100.0)
    assert passed is True
    assert reason == "Passed"


def test_market_buy_notional_unchanged_when_no_limit():
    manager = RiskManager(_config(MAX_ORDER_VALUE=5000.0))
    order = _order(symbol="BTC/USDT", quantity=30.0, direction="BUY", price=None)
    passed, reason = manager.check_order(order, current_price=100.0)
    assert passed is True
    assert reason == "Passed"


# --------------------------------------------------------------------------- #
# M3 - gross exposure & net-direction in HEDGE mode                           #
# --------------------------------------------------------------------------- #
def test_fully_hedged_legs_count_gross_when_enabled():
    # BTC LONG 0.04 + SHORT 0.04 @ 50k -> gross 4000, net 0.
    cfg = _config(
        ENFORCE_GROSS_EXPOSURE_IN_HEDGE=True,
        POSITION_MODE="HEDGE",
        MAX_TOTAL_MARGIN_PCT=0.5,
    )
    manager = RiskManager(cfg)
    portfolio = _portfolio(
        equity=5000.0,
        holdings={"total": 5000.0, "BTC/USDT": 0.0},
        positions={"BTC/USDT": 0.0},
        legs={"BTC/USDT": {"LONG": 0.04, "SHORT": 0.04}},
        prices={"BTC/USDT": 50_000.0},
    )
    passed, reason, _action, details = manager.evaluate_portfolio_risk(portfolio)
    # gross 4000 / equity 5000 = 0.8 utilization > 0.5 cap.
    assert passed is False
    assert reason == "Margin utilization breach"
    assert details["margin_utilization"] == 0.8
    assert details["net_direction_notional"] == 0.0


def test_fully_hedged_legs_are_zero_when_disabled_byte_identical():
    cfg = _config(
        ENFORCE_GROSS_EXPOSURE_IN_HEDGE=False,  # legacy net accounting
        POSITION_MODE="HEDGE",
        MAX_TOTAL_MARGIN_PCT=0.5,
    )
    manager = RiskManager(cfg)
    portfolio = _portfolio(
        equity=5000.0,
        holdings={"total": 5000.0, "BTC/USDT": 0.0},
        positions={"BTC/USDT": 0.0},
        legs={"BTC/USDT": {"LONG": 0.04, "SHORT": 0.04}},
        prices={"BTC/USDT": 50_000.0},
    )
    passed, reason, _action, details = manager.evaluate_portfolio_risk(portfolio)
    assert passed is True
    assert reason == "Passed"
    assert details["margin_utilization"] == 0.0  # net legs -> zero (legacy)
    assert details["net_direction_notional"] == 0.0


def test_net_direction_metric_signed_from_legs():
    cfg = _config(ENFORCE_GROSS_EXPOSURE_IN_HEDGE=True, POSITION_MODE="HEDGE")
    manager = RiskManager(cfg)
    # LONG 0.06 - SHORT 0.02 @ 50k -> net_direction (0.04 * 50k) = 2000; gross 4000.
    portfolio = _portfolio(
        equity=100_000.0,
        holdings={"total": 100_000.0, "BTC/USDT": 2000.0},
        positions={"BTC/USDT": 0.04},
        legs={"BTC/USDT": {"LONG": 0.06, "SHORT": 0.02}},
        prices={"BTC/USDT": 50_000.0},
    )
    passed, _reason, _action, details = manager.evaluate_portfolio_risk(portfolio)
    assert passed is True
    assert details["net_direction_notional"] == pytest.approx(2000.0)


def test_gross_not_applied_when_position_mode_not_hedge():
    cfg = _config(
        ENFORCE_GROSS_EXPOSURE_IN_HEDGE=True,
        POSITION_MODE="ONEWAY",  # not HEDGE -> gross gate stays off
        MAX_TOTAL_MARGIN_PCT=0.5,
    )
    manager = RiskManager(cfg)
    portfolio = _portfolio(
        equity=5000.0,
        holdings={"total": 5000.0, "BTC/USDT": 0.0},
        positions={"BTC/USDT": 0.0},
        legs={"BTC/USDT": {"LONG": 0.04, "SHORT": 0.04}},
        prices={"BTC/USDT": 50_000.0},
    )
    passed, _reason, _action, details = manager.evaluate_portfolio_risk(portfolio)
    assert passed is True
    assert details["margin_utilization"] == 0.0


def test_check_order_total_notional_uses_gross_in_hedge():
    # Two symbols: BTC hedged legs (gross 4000) + a small BUY on ETH. The
    # portfolio-wide total-notional cap must see the BTC gross when enabled.
    cfg = _config(
        ENFORCE_GROSS_EXPOSURE_IN_HEDGE=True,
        POSITION_MODE="HEDGE",
        MAX_TOTAL_NOTIONAL_PCT=0.40,  # cap = 4000 at 10k equity
        MAX_SYMBOL_EXPOSURE_PCT=1.0,
        MAX_ORDER_VALUE=0.0,
    )
    manager = RiskManager(cfg)
    portfolio = _portfolio(
        equity=10_000.0,
        symbols=("BTC/USDT", "ETH/USDT"),
        holdings={"total": 10_000.0, "BTC/USDT": 0.0, "ETH/USDT": 0.0},
        positions={"BTC/USDT": 0.0, "ETH/USDT": 0.0},
        legs={"BTC/USDT": {"LONG": 0.04, "SHORT": 0.04}},
        prices={"BTC/USDT": 50_000.0, "ETH/USDT": 100.0},
    )
    order = _order(symbol="ETH/USDT", quantity=1.0, direction="BUY")
    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)
    # gross 4000 + order 100 = 4100 > cap 4000 -> reject.
    assert passed is False
    assert "Total exposure" in reason


def test_check_order_total_notional_net_when_gross_disabled():
    cfg = _config(
        ENFORCE_GROSS_EXPOSURE_IN_HEDGE=False,
        POSITION_MODE="HEDGE",
        MAX_TOTAL_NOTIONAL_PCT=0.40,
        MAX_SYMBOL_EXPOSURE_PCT=1.0,
        MAX_ORDER_VALUE=0.0,
    )
    manager = RiskManager(cfg)
    portfolio = _portfolio(
        equity=10_000.0,
        symbols=("BTC/USDT", "ETH/USDT"),
        holdings={"total": 10_000.0, "BTC/USDT": 0.0, "ETH/USDT": 0.0},
        positions={"BTC/USDT": 0.0, "ETH/USDT": 0.0},
        legs={"BTC/USDT": {"LONG": 0.04, "SHORT": 0.04}},
        prices={"BTC/USDT": 50_000.0, "ETH/USDT": 100.0},
    )
    order = _order(symbol="ETH/USDT", quantity=1.0, direction="BUY")
    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)
    # legacy net: BTC 0 + order 100 = 100 << cap -> pass.
    assert passed is True
    assert reason == "Passed"


# --------------------------------------------------------------------------- #
# M4 - order-rate budget                                                       #
# --------------------------------------------------------------------------- #
_NS = 1_000_000_000


def test_order_rate_limit_blocks_third_order_in_window():
    manager = RiskManager(_config(MAX_ORDERS_PER_MINUTE=2))
    portfolio = _portfolio(equity=1_000_000.0)
    base = 1_000_000 * _NS

    def _o(ts):
        return _order(symbol="BTC/USDT", quantity=0.001, direction="BUY", timestamp_ns=ts)

    p1, _ = manager.check_order(_o(base), current_price=100.0, portfolio=portfolio)
    p2, _ = manager.check_order(_o(base + 10 * _NS), current_price=100.0, portfolio=portfolio)
    p3, reason3 = manager.check_order(_o(base + 20 * _NS), current_price=100.0, portfolio=portfolio)
    assert p1 is True and p2 is True
    assert p3 is False
    assert "order-rate" in reason3.lower()


def test_order_rate_window_rolls_off_after_60s():
    manager = RiskManager(_config(MAX_ORDERS_PER_MINUTE=1))
    portfolio = _portfolio(equity=1_000_000.0)
    base = 1_000_000 * _NS

    def _o(ts):
        return _order(symbol="BTC/USDT", quantity=0.001, direction="BUY", timestamp_ns=ts)

    p1, _ = manager.check_order(_o(base), current_price=100.0, portfolio=portfolio)
    p2, _ = manager.check_order(_o(base + 30 * _NS), current_price=100.0, portfolio=portfolio)
    # 61s later the first order has aged out of the trailing minute -> allowed again.
    p3, _ = manager.check_order(_o(base + 61 * _NS), current_price=100.0, portfolio=portfolio)
    assert p1 is True
    assert p2 is False
    assert p3 is True


def test_order_rate_reduce_only_exempt():
    manager = RiskManager(_config(MAX_ORDERS_PER_MINUTE=1))
    portfolio = _portfolio(equity=1_000_000.0)
    base = 1_000_000 * _NS
    entry = _order(symbol="BTC/USDT", quantity=0.001, direction="BUY", timestamp_ns=base)
    exit_ord = _order(
        symbol="BTC/USDT",
        quantity=0.001,
        direction="SELL",
        reduce_only=True,
        timestamp_ns=base + 1 * _NS,
    )
    assert manager.check_order(entry, current_price=100.0, portfolio=portfolio)[0] is True
    # Reduce-only bypasses the rate budget even though the minute is "full".
    passed, reason = manager.check_order(exit_ord, current_price=100.0, portfolio=portfolio)
    assert passed is True
    assert "reduce-only" in reason.lower()


def test_order_rate_disabled_is_no_op():
    manager = RiskManager(_config(MAX_ORDERS_PER_MINUTE=0))
    portfolio = _portfolio(equity=1_000_000.0)
    for _ in range(20):
        order = _order(symbol="BTC/USDT", quantity=0.001, direction="BUY")
        assert manager.check_order(order, current_price=100.0, portfolio=portfolio)[0] is True


# --------------------------------------------------------------------------- #
# M4 - daily turnover budget                                                   #
# --------------------------------------------------------------------------- #
def test_daily_turnover_budget_blocks_over_cap():
    manager = RiskManager(_config(MAX_DAILY_NOTIONAL_TURNOVER_PCT=0.1, MAX_ORDER_VALUE=0.0))
    portfolio = _portfolio(equity=10_000.0)  # budget = 1000
    base = 1_000_000 * _NS

    def _o(ts):
        return _order(symbol="BTC/USDT", quantity=4.0, direction="BUY", timestamp_ns=ts)

    p1, _ = manager.check_order(_o(base), current_price=100.0, portfolio=portfolio)  # 400
    p2, _ = manager.check_order(_o(base + _NS), current_price=100.0, portfolio=portfolio)  # 800
    p3, reason3 = manager.check_order(_o(base + 2 * _NS), current_price=100.0, portfolio=portfolio)
    assert p1 is True and p2 is True
    assert p3 is False  # projected 1200 > 1000
    assert "turnover" in reason3.lower()


def test_daily_turnover_resets_next_day():
    manager = RiskManager(_config(MAX_DAILY_NOTIONAL_TURNOVER_PCT=0.1, MAX_ORDER_VALUE=0.0))
    portfolio = _portfolio(equity=10_000.0)
    base = 1_000_000 * _NS

    def _o(ts):
        return _order(symbol="BTC/USDT", quantity=4.0, direction="BUY", timestamp_ns=ts)

    manager.check_order(_o(base), current_price=100.0, portfolio=portfolio)  # 400
    manager.check_order(_o(base + _NS), current_price=100.0, portfolio=portfolio)  # 800
    blocked, _ = manager.check_order(_o(base + 2 * _NS), current_price=100.0, portfolio=portfolio)
    assert blocked is False
    # +1 day: accumulator resets, order admitted again.
    next_day = base + 86_400 * _NS
    passed, reason = manager.check_order(_o(next_day), current_price=100.0, portfolio=portfolio)
    assert passed is True
    assert reason == "Passed"


def test_daily_turnover_disabled_is_no_op():
    manager = RiskManager(_config(MAX_DAILY_NOTIONAL_TURNOVER_PCT=0.0, MAX_ORDER_VALUE=0.0))
    portfolio = _portfolio(equity=10_000.0)
    for _ in range(50):
        order = _order(symbol="BTC/USDT", quantity=4.0, direction="BUY")
        assert manager.check_order(order, current_price=100.0, portfolio=portfolio)[0] is True


# --------------------------------------------------------------------------- #
# M4 - max position age                                                        #
# --------------------------------------------------------------------------- #
def test_position_age_breach_freezes():
    manager = RiskManager(_config(MAX_POSITION_AGE_HOURS=24.0))
    portfolio = _portfolio(equity=10_000.0, max_position_age_hours=30.0)
    passed, reason, action, details = manager.evaluate_portfolio_risk(portfolio)
    assert passed is False
    assert reason == "Position age breach"
    assert action == "FREEZE"
    assert details["max_position_age_hours"] == 30.0


def test_position_age_breach_flattens_with_auto_flatten():
    manager = RiskManager(_config(MAX_POSITION_AGE_HOURS=24.0, AUTO_FLATTEN_ON_BREACH=True))
    portfolio = _portfolio(equity=10_000.0, max_position_age_hours=30.0)
    passed, reason, action, _ = manager.evaluate_portfolio_risk(portfolio)
    assert passed is False
    assert reason == "Position age breach"
    assert action == "FLATTEN"


def test_position_age_within_limit_passes():
    manager = RiskManager(_config(MAX_POSITION_AGE_HOURS=24.0))
    portfolio = _portfolio(equity=10_000.0, max_position_age_hours=10.0)
    passed, reason, _action, _ = manager.evaluate_portfolio_risk(portfolio)
    assert passed is True
    assert reason == "Passed"


def test_position_age_inert_without_portfolio_method():
    # No get_max_position_age_hours() -> tier is inert (safe until wired).
    manager = RiskManager(_config(MAX_POSITION_AGE_HOURS=24.0))
    portfolio = _portfolio(equity=10_000.0)  # no age method
    passed, reason, _action, _ = manager.evaluate_portfolio_risk(portfolio)
    assert passed is True
    assert reason == "Passed"


def test_position_age_disabled_is_no_op():
    manager = RiskManager(_config(MAX_POSITION_AGE_HOURS=0.0))
    portfolio = _portfolio(equity=10_000.0, max_position_age_hours=1_000.0)
    passed, reason, _action, _ = manager.evaluate_portfolio_risk(portfolio)
    assert passed is True
    assert reason == "Passed"


# --------------------------------------------------------------------------- #
# M1 - kill-switch state persistence                                          #
# --------------------------------------------------------------------------- #
def test_get_set_state_round_trips_consecutive_losses():
    manager = RiskManager(_config())
    manager.record_loss(realized_pnl=-1.0)
    manager.record_loss(realized_pnl=-1.0)
    manager.record_loss(realized_pnl=-1.0)
    state = manager.get_state()
    assert state["consecutive_loss_count"] == 3
    assert state["hard_halt"] is False

    restored = RiskManager(_config())
    restored.set_state(state)
    assert restored._consecutive_loss_count == 3


def test_persisted_loss_count_survives_restart_and_reaches_halt():
    # Restarting must NOT re-arm the counter to 0 (M1). With 3 losses persisted and
    # a halt at 5, two further losses on the restored manager trip the halt.
    original = RiskManager(_config(CONSECUTIVE_LOSS_HALT_COUNT=5))
    for _ in range(3):
        original.record_loss(realized_pnl=-1.0)

    restored = RiskManager(_config(CONSECUTIVE_LOSS_HALT_COUNT=5))
    restored.set_state(original.get_state())
    restored.record_loss(realized_pnl=-1.0)
    restored.record_loss(realized_pnl=-1.0)  # now 5 -> halt

    portfolio = _portfolio(equity=10_000.0)
    order = _order(symbol="BTC/USDT", quantity=0.001, direction="BUY")
    passed, reason = restored.check_order(order, current_price=100.0, portfolio=portfolio)
    assert passed is False
    assert "consecutive-loss halt" in reason.lower()


def test_hard_halt_persists_and_blocks_new_entries():
    manager = RiskManager(_config())
    manager.trip_hard_halt()
    state = manager.get_state()
    assert state["hard_halt"] is True

    restored = RiskManager(_config())
    restored.set_state(state)
    portfolio = _portfolio(equity=10_000.0)

    entry = _order(symbol="BTC/USDT", quantity=0.001, direction="BUY")
    passed, reason = restored.check_order(entry, current_price=100.0, portfolio=portfolio)
    assert passed is False
    assert "hard halt" in reason.lower()

    # Reduce-only exits stay allowed under the hard halt.
    exit_ord = _order(symbol="BTC/USDT", quantity=0.001, direction="SELL", reduce_only=True)
    passed_exit, _ = restored.check_order(exit_ord, current_price=100.0, portfolio=portfolio)
    assert passed_exit is True


def test_hard_halt_default_off_is_byte_identical():
    manager = RiskManager(_config())
    assert manager._hard_halt is False
    portfolio = _portfolio(equity=10_000.0)
    order = _order(symbol="BTC/USDT", quantity=0.001, direction="BUY")
    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)
    assert passed is True
    assert reason == "Passed"


def test_set_state_tolerant_of_junk():
    manager = RiskManager(_config())
    manager.record_loss(realized_pnl=-1.0)
    manager.record_loss(realized_pnl=-1.0)
    before = manager._consecutive_loss_count
    manager.set_state(None)  # non-dict ignored
    manager.set_state({})  # missing keys ignored
    manager.set_state({"consecutive_loss_count": -5})  # negative ignored
    manager.set_state({"consecutive_loss_count": True})  # bool ignored
    manager.set_state({"hard_halt": "yes"})  # non-bool ignored
    assert manager._consecutive_loss_count == before
    assert manager._hard_halt is False
