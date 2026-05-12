"""Triple-barrier forward outcome labeling for candidate signals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class BarrierType(StrEnum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIME_EXIT = "time_exit"
    NO_PATH = "no_path"


@dataclass(frozen=True, slots=True)
class TripleBarrierOutcome:
    entry_index: int
    exit_index: int
    side: str
    barrier_type: BarrierType
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    net_pnl_bps: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    slippage_cost: float
    fee_cost: float
    funding_cost: float
    bars_held: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["barrier_type"] = self.barrier_type.value
        return payload


def _side_multiplier(side: str | int | float) -> int:
    if isinstance(side, str):
        token = side.upper()
        if token in {"LONG", "BUY", "1"}:
            return 1
        if token in {"SHORT", "SELL", "-1"}:
            return -1
    return 1 if float(side) >= 0.0 else -1


def _price_at(values: Sequence[float] | None, index: int, fallback: float) -> float:
    if values is None or index >= len(values):
        return fallback
    try:
        value = float(values[index])
    except Exception:
        return fallback
    return value if math.isfinite(value) and value > 0.0 else fallback


def label_triple_barrier(
    closes: Sequence[float],
    *,
    entry_index: int,
    side: str | int | float,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_bars: int,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    funding_bps_per_bar: float = 0.0,
) -> TripleBarrierOutcome:
    """Label one candidate entry with stop/take/time barriers.

    If both intrabar stop and take-profit are possible, the labeler uses a
    conservative stop-first convention.  Costs are subtracted from the realized
    side-adjusted gross PnL.
    """
    prices = [float(price) for price in closes]
    if not prices or entry_index < 0 or entry_index >= len(prices):
        raise ValueError("entry_index must reference a valid close")
    entry_price = prices[entry_index]
    if not math.isfinite(entry_price) or entry_price <= 0.0:
        raise ValueError("entry close must be finite and positive")

    side_mult = _side_multiplier(side)
    horizon = max(1, int(max_hold_bars))
    stop = max(0.0, float(stop_loss_pct))
    take = max(0.0, float(take_profit_pct))
    last_index = min(len(prices) - 1, entry_index + horizon)
    if last_index <= entry_index:
        cost = (2.0 * (float(fee_bps) + float(slippage_bps))) / 10_000.0
        return TripleBarrierOutcome(
            entry_index=entry_index,
            exit_index=entry_index,
            side="LONG" if side_mult > 0 else "SHORT",
            barrier_type=BarrierType.NO_PATH,
            entry_price=entry_price,
            exit_price=entry_price,
            gross_pnl=0.0,
            net_pnl=-cost,
            net_pnl_bps=-cost * 10_000.0,
            max_adverse_excursion=0.0,
            max_favorable_excursion=0.0,
            slippage_cost=(2.0 * float(slippage_bps)) / 10_000.0,
            fee_cost=(2.0 * float(fee_bps)) / 10_000.0,
            funding_cost=0.0,
            bars_held=0,
        )

    mae = 0.0
    mfe = 0.0
    exit_index = last_index
    barrier = BarrierType.TIME_EXIT
    for index in range(entry_index + 1, last_index + 1):
        close_price = prices[index]
        high_price = _price_at(highs, index, close_price)
        low_price = _price_at(lows, index, close_price)
        favorable_price = high_price if side_mult > 0 else low_price
        adverse_price = low_price if side_mult > 0 else high_price
        favorable = side_mult * ((favorable_price / entry_price) - 1.0)
        adverse = side_mult * ((adverse_price / entry_price) - 1.0)
        close_pnl = side_mult * ((close_price / entry_price) - 1.0)
        mfe = max(mfe, favorable, close_pnl)
        mae = min(mae, adverse, close_pnl)
        if stop > 0.0 and adverse <= -stop:
            exit_index = index
            barrier = BarrierType.STOP_LOSS
            break
        if take > 0.0 and favorable >= take:
            exit_index = index
            barrier = BarrierType.TAKE_PROFIT
            break

    exit_price = prices[exit_index]
    gross = side_mult * ((exit_price / entry_price) - 1.0)
    bars_held = max(0, exit_index - entry_index)
    slippage_cost = (2.0 * float(slippage_bps)) / 10_000.0
    fee_cost = (2.0 * float(fee_bps)) / 10_000.0
    funding_cost = (float(funding_bps_per_bar) * bars_held) / 10_000.0
    net = gross - slippage_cost - fee_cost - funding_cost
    return TripleBarrierOutcome(
        entry_index=entry_index,
        exit_index=exit_index,
        side="LONG" if side_mult > 0 else "SHORT",
        barrier_type=barrier,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_pnl=gross,
        net_pnl=net,
        net_pnl_bps=net * 10_000.0,
        max_adverse_excursion=mae,
        max_favorable_excursion=mfe,
        slippage_cost=slippage_cost,
        fee_cost=fee_cost,
        funding_cost=funding_cost,
        bars_held=bars_held,
    )
