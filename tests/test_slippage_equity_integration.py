"""Regression guard: sqrt_impact slippage must REDUCE realised portfolio equity.

The existing slippage-impact suite (``test_execution_model_slippage_impact.py``)
only pins ``fill_price`` from ``ExecutionModel.compute_fill``.  Nothing asserts the
downstream consequence that matters for a backtest: a deterministic round-trip
(BUY then SELL) routed through the real fill path (``SimulatedExecutionHandler``,
which wires ``order_notional`` into the model) ends with **strictly lower** total
return / final equity when ``sqrt_impact`` market impact is enabled, compared to
the flat-slippage baseline holding everything else (seed, prices, latency) fixed.
"""

from __future__ import annotations

import queue

from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.core.events import MarketEvent, OrderEvent


# ── deterministic synthetic bars ─────────────────────────────────────────────
#
# Fixed, monotone-up price path so the gross PnL of a BUY→SELL round trip is
# positive in the absence of costs; any slippage worsens the entry/exit prices.

_OPEN = 100.0
_BUY_BAR_OPEN = _OPEN
_SELL_BAR_OPEN = 110.0  # price rises 10% before we exit
_QTY = 1.0
_BAR_VOLUME = 1_000_000.0  # huge bar volume → no liquidity-cap partial fills


class _Bars:
    """Minimal bars stub matching the interface SimulatedExecutionHandler uses."""

    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        return _OPEN


def _flat_config():
    class _Config:
        RANDOM_SEED = 1234
        SLIPPAGE_RATE = 0.0005
        SPREAD_RATE = 0.0002
        TAKER_FEE_RATE = 0.0004
        SIM_MAX_BAR_VOLUME_RATIO = 1.0  # allow full fills
        SIM_LATENCY_MIN_BARS = 0  # release immediately on the next check
        SIM_LATENCY_MAX_BARS = 0
        SLIPPAGE_IMPACT_MODEL = "flat"
        SLIPPAGE_IMPACT_COEFFICIENT = 0.0
        SLIPPAGE_ADV_QUOTE = 0.0

    return _Config


def _sqrt_config():
    class _Config:
        RANDOM_SEED = 1234  # identical seed → identical flat-slippage RNG draws
        SLIPPAGE_RATE = 0.0005
        SPREAD_RATE = 0.0002
        TAKER_FEE_RATE = 0.0004
        SIM_MAX_BAR_VOLUME_RATIO = 1.0
        SIM_LATENCY_MIN_BARS = 0
        SIM_LATENCY_MAX_BARS = 0
        SLIPPAGE_IMPACT_MODEL = "sqrt_impact"
        SLIPPAGE_IMPACT_COEFFICIENT = 0.05  # positive → strictly more cost
        # Small ADV relative to the ~100 quote notional of each leg → material
        # participation → material sqrt impact.
        SLIPPAGE_ADV_QUOTE = 1_000.0

    return _Config


def _bar(time: int, open_: float) -> MarketEvent:
    return MarketEvent(
        time=time,
        symbol="BTC/USDT",
        open=open_,
        high=open_ * 1.001,
        low=open_ * 0.999,
        close=open_,
        volume=_BAR_VOLUME,
    )


def _round_trip_final_equity(config) -> tuple[float, float, float]:
    """Drive a BUY→SELL round trip through the real fill path.

    Returns (final_equity, buy_fill_price, sell_fill_price).

    Cash accounting (long, no leverage):
        start_cash -> buy: cash -= buy_cost + buy_commission
                   -> sell: cash += sell_proceeds - sell_commission
    Final equity == cash after the position is flat.
    """
    events: queue.Queue = queue.Queue()
    handler = SimulatedExecutionHandler(events, _Bars(), config)
    start_cash = 100_000.0

    # ── BUY leg ──────────────────────────────────────────────────────────────
    handler.execute_order(OrderEvent("BTC/USDT", "MKT", _QTY, "BUY"))
    handler.check_open_orders(_bar(1, _BUY_BAR_OPEN))
    buy_fill = events.get_nowait()
    assert buy_fill.type == "FILL"
    assert float(buy_fill.quantity) == _QTY  # full fill, no partials
    buy_fill_price = buy_fill.fill_cost / buy_fill.quantity
    cash = start_cash - buy_fill.fill_cost - buy_fill.commission

    # ── SELL leg ─────────────────────────────────────────────────────────────
    handler.execute_order(OrderEvent("BTC/USDT", "MKT", _QTY, "SELL"))
    handler.check_open_orders(_bar(2, _SELL_BAR_OPEN))
    sell_fill = events.get_nowait()
    assert sell_fill.type == "FILL"
    assert float(sell_fill.quantity) == _QTY
    sell_fill_price = sell_fill.fill_cost / sell_fill.quantity
    cash = cash + sell_fill.fill_cost - sell_fill.commission

    return cash, buy_fill_price, sell_fill_price


def test_sqrt_impact_strictly_reduces_round_trip_equity():
    """Same deterministic run: sqrt_impact final equity < flat baseline equity."""
    flat_equity, flat_buy, flat_sell = _round_trip_final_equity(_flat_config())
    sqrt_equity, sqrt_buy, sqrt_sell = _round_trip_final_equity(_sqrt_config())

    start_cash = 100_000.0
    flat_total_return = (flat_equity - start_cash) / start_cash
    sqrt_total_return = (sqrt_equity - start_cash) / start_cash

    # Impact must move each leg adversarially: pay more on the BUY, get less on
    # the SELL — confirming order_notional is actually wired into the fill path.
    assert sqrt_buy > flat_buy, f"sqrt BUY fill {sqrt_buy} should exceed flat BUY fill {flat_buy}"
    assert sqrt_sell < flat_sell, (
        f"sqrt SELL fill {sqrt_sell} should be below flat SELL fill {flat_sell}"
    )

    # The headline guard: realised equity / total return is strictly lower.
    assert sqrt_equity < flat_equity, (
        f"sqrt_impact equity {sqrt_equity} should be strictly below flat {flat_equity}"
    )
    assert sqrt_total_return < flat_total_return


def test_flat_baseline_round_trip_is_profitable_pre_impact():
    """Sanity: the flat round trip is profitable, so impact has room to bite.

    Without this, a strictly-lower assertion could be satisfied trivially by two
    equally-broken runs; this pins that the baseline actually makes money on the
    +10% price move (net of flat costs).
    """
    flat_equity, _, _ = _round_trip_final_equity(_flat_config())
    assert flat_equity > 100_000.0
