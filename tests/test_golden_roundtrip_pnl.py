"""Hand-computed ground-truth PnL for the smallest deterministic round trip.

This is NOT a random-seed parity check. It pins the exact final equity of a
single-unit buy-then-sell round trip against an arithmetic value computed by
hand from the real fill + portfolio cash/mark-to-market path.

Determinism: we use LMT (maker) fills, so ``ExecutionModel.compute_fill`` returns
the limit price exactly with no RNG slippage (see execution_model.py lines 230-233
and the LMT fill assumptions docstring). The only costs are maker commissions.

Round trip (single unit BTC, maker fee = 0.0002, initial capital = 10_000):

    BUY  1 @ 100  -> commission_buy  = 100 * 1 * 0.0002 = 0.02
                     cost = +100
                     cash = 10000 - 100 - 0.02       = 9899.98
                     position = +1

    SELL 1 @ 110  -> commission_sell = 110 * 1 * 0.0002 = 0.022
                     cost = -110
                     cash = 9899.98 - (-110 + 0.022)  = 10009.958
                     position = 0  (flat)

    mark-to-market (update_timeindex): position flat -> market_value = 0
                     final equity = cash = 10009.958

    Net PnL = +10 (price gain 110-100) - 0.042 (total fees) = +9.958
            = 10009.958 - 10000 = 9.958  (verified below)

The expected value is the hand-derived number, locked as ground truth.
"""

from __future__ import annotations

import queue
from datetime import datetime

import pytest

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import FillEvent


class _MockBars:
    """Minimal single-symbol bars stub (mirrors tests/test_funding_liquidation.py)."""

    symbol_list = ["BTC/USDT"]

    def __init__(self, start_dt: datetime, price: float) -> None:
        self.current_dt = start_dt
        self.close = price

    def get_latest_bar_datetime(self, symbol):  # noqa: D401 - stub
        _ = symbol
        return self.current_dt

    def get_latest_bar_value(self, symbol, val_type):  # noqa: D401 - stub
        _ = (symbol, val_type)
        return self.close


class _RoundTripConfig:
    """Plain-class config -> routed through execution_model._config_from_attrs.

    Funding and liquidation are disabled so the ONLY cash effect besides the
    trade itself is the maker commission, keeping the hand computation exact.
    """

    INITIAL_CAPITAL = 10_000.0
    MAKER_FEE_RATE = 0.0002
    TAKER_FEE_RATE = 0.0004
    COMMISSION_RATE = 0.0004
    SLIPPAGE_RATE = 0.0  # irrelevant for maker fills, set to 0 for clarity
    SPREAD_RATE = 0.0
    LEVERAGE = 1  # no liquidation model (leverage <= 1 returns None)
    MARGIN_MODE = "isolated"
    MAINTENANCE_MARGIN_RATE = 0.005
    LIQUIDATION_BUFFER_RATE = 0.0
    FUNDING_RATE_PER_8H = 0.0  # no funding payments
    FUNDING_INTERVAL_HOURS = 8
    MAX_DAILY_LOSS_PCT = 0.99
    RANDOM_SEED = 42


def _maker_fill(portfolio: Portfolio, *, raw_price: float, qty: float, direction: str):
    """Compute a maker (LMT) fill through the portfolio's real ExecutionModel.

    LMT fills are deterministic: fill_price == raw_price exactly, maker fee,
    no RNG draw. ``apply_liquidity_cap=False`` so the full unit fills.
    """
    return portfolio.execution_model.compute_fill(
        raw_price=raw_price,
        qty=qty,
        direction=direction,
        bar_volume=1_000.0,
        is_maker=True,
        apply_liquidity_cap=False,
    )


def test_single_unit_roundtrip_final_equity_is_hand_computed():
    events: queue.Queue = queue.Queue()
    start_dt = datetime(2026, 1, 1, 0, 0)
    bars = _MockBars(start_dt, price=100.0)
    p = Portfolio(bars, events, start_dt, _RoundTripConfig)

    # Sanity: starting equity equals initial capital exactly.
    assert p.current_holdings["total"] == pytest.approx(10_000.0, rel=1e-12)

    # --- BUY 1 @ 100 (maker) -------------------------------------------------
    buy = _maker_fill(p, raw_price=100.0, qty=1.0, direction="BUY")
    # The real model must return the limit price exactly (no slippage on maker).
    assert buy.fill_price == pytest.approx(100.0, rel=1e-12)
    assert buy.executed_qty == pytest.approx(1.0, rel=1e-12)
    # Hand-computed maker commission: 100 * 1 * 0.0002.
    assert buy.commission == pytest.approx(0.02, rel=1e-9)

    p.update_fill(
        FillEvent(
            timeindex=start_dt,
            symbol="BTC/USDT",
            exchange="TEST",
            quantity=buy.executed_qty,
            direction="BUY",
            fill_cost=buy.fill_price * buy.executed_qty,
            commission=buy.commission,
        )
    )

    assert p.current_positions["BTC/USDT"] == pytest.approx(1.0, rel=1e-12)
    # cash = 10000 - 100 - 0.02
    assert p.current_holdings["cash"] == pytest.approx(9_899.98, rel=1e-12)

    # --- SELL 1 @ 110 (maker) ------------------------------------------------
    sell = _maker_fill(p, raw_price=110.0, qty=1.0, direction="SELL")
    assert sell.fill_price == pytest.approx(110.0, rel=1e-12)
    # Hand-computed maker commission: 110 * 1 * 0.0002.
    assert sell.commission == pytest.approx(0.022, rel=1e-9)

    p.update_fill(
        FillEvent(
            timeindex=start_dt,
            symbol="BTC/USDT",
            exchange="TEST",
            quantity=sell.executed_qty,
            direction="SELL",
            fill_cost=sell.fill_price * sell.executed_qty,
            commission=sell.commission,
        )
    )

    # Position is now flat.
    assert p.current_positions["BTC/USDT"] == pytest.approx(0.0, abs=1e-12)
    # cash = 9899.98 + 110 - 0.022
    assert p.current_holdings["cash"] == pytest.approx(10_009.958, rel=1e-12)

    # --- mark-to-market: flat position => equity == cash ---------------------
    # bars.close stays at 110, but qty == 0 so market_value == 0.
    bars.close = 110.0
    p.update_timeindex(None)

    # GROUND TRUTH: final equity = 10009.958 (= 10000 + 10 price gain - 0.042 fees).
    expected_final_equity = 10_009.958
    assert p.current_holdings["total"] == pytest.approx(expected_final_equity, rel=1e-9)

    # Net PnL decomposition also pins the arithmetic explicitly.
    net_pnl = p.current_holdings["total"] - _RoundTripConfig.INITIAL_CAPITAL
    assert net_pnl == pytest.approx(10.0 - (0.02 + 0.022), rel=1e-9)
    # Total commission booked equals the sum of the two maker fees.
    assert p.current_holdings["commission"] == pytest.approx(0.042, rel=1e-9)
