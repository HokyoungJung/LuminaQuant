"""Offline execution-attribution kernel (research-only, additive).

Headless arithmetic core that turns a hand-built or exported fill sequence into

    * FIFO round-trip pairings (netting long/short inventory),
    * four pure-math execution-bias severities in ``[0, 1]``, and
    * a delta-PnL attribution split into
      ``noise / early / late / overtrading / missed`` buckets whose sum is
      conserved (``missed`` is the exact residual).

Everything here is offline and side-effect free.  It never imports the live /
backtest hot path and never mutates an existing metric mapping — every result
is returned as a brand-new frozen dataclass so it cannot contaminate the
existing candidate-outcome / metric structures.

Cost conventions are aligned with Lumina's own execution model
(``lumina_quant.backtesting.execution_model``):

    * fee cash  = ``fill_price * qty * fee_rate`` (per fill, apportioned by the
      matched quantity when a fill is split across several round trips),
    * funding cash (a *cost* when positive) =
      ``sign * notional * (rate * interval_hours / 8) * periods`` with
      ``sign = +1`` for longs (longs pay a positive funding rate), matching
      ``ExecutionModel.compute_funding_payment``.

The module is gated by ``research.execution_attribution_enabled`` (default
``False``); nothing in the default runtime imports it.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from lumina_quant.market_units import BPS_PER_UNIT

__all__ = [
    "AttributionCostModel",
    "ExecutionAttribution",
    "ExecutionAttributionReport",
    "ExecutionBiasSeverity",
    "FillEvent",
    "RoundTrip",
    "attribute_execution_delta",
    "early_exit_bias_severity",
    "late_exit_bias_severity",
    "noise_bias_severity",
    "overtrading_bias_severity",
    "pair_round_trips_fifo",
    "run_execution_attribution",
]


_EPS = 1e-12


def _clip01(value: float) -> float:
    """Clamp ``value`` into the closed unit interval deterministically."""
    if not math.isfinite(value):
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


# ── Inputs ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FillEvent:
    """A single execution fill.

    ``qty`` is always strictly positive; direction is carried by ``side``
    (``"BUY"`` / ``"SELL"``).  ``timestamp`` is any sortable numeric clock — the
    caller decides the unit, but funding uses ``timestamp`` deltas measured in
    the same unit as ``AttributionCostModel.funding_interval_hours`` (hours by
    default).  ``fee`` is the quote-ccy cash fee already charged for this fill;
    when ``fee`` is ``None`` the fee is derived from the cost model as
    ``price * qty * fee_rate`` (taker unless ``is_maker``).
    """

    symbol: str
    side: str
    qty: float
    price: float
    timestamp: float
    fee: float | None = None
    is_maker: bool = False

    def direction(self) -> str:
        return "long" if str(self.side).strip().upper() == "BUY" else "short"


@dataclass(frozen=True, slots=True)
class AttributionCostModel:
    """Fee/funding parameters, mirroring ``ExecutionConfig`` semantics."""

    taker_fee_rate: float = 0.0004
    maker_fee_rate: float = 0.0002
    funding_rate_per_8h: float = 0.0
    funding_interval_hours: float = 8.0
    # Trades whose absolute net edge is below this band are treated as noise.
    noise_threshold_bps: float = 2.0

    def fee_for(self, *, price: float, qty: float, is_maker: bool) -> float:
        rate = self.maker_fee_rate if bool(is_maker) else self.taker_fee_rate
        return abs(float(price)) * abs(float(qty)) * float(rate)

    def funding_for(
        self, *, direction: str, notional: float, entry_time: float, exit_time: float
    ) -> float:
        """Funding cash *cost* (positive => paid) for one matched round trip.

        Mirrors ``ExecutionModel.compute_funding_payment``: a positive funding
        rate means longs pay and shorts receive.  Only whole elapsed intervals
        accrue, matching the backtest's discrete funding cadence.
        """
        rate = float(self.funding_rate_per_8h)
        interval_hours = float(self.funding_interval_hours)
        held = float(exit_time) - float(entry_time)
        if abs(rate) <= _EPS or interval_hours <= _EPS or held <= 0.0:
            return 0.0
        periods = int(held // interval_hours)
        if periods <= 0:
            return 0.0
        interval_rate = rate * (interval_hours / 8.0)
        sign = 1.0 if direction == "long" else -1.0
        return sign * abs(float(notional)) * interval_rate * float(periods)


# ── Round trips ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """One FIFO-matched open→close round trip.

    ``mfe_bps`` / ``mae_bps`` are optional favorable / adverse excursion
    magnitudes (both non-negative bps of entry notional) that the caller may
    attach for delta attribution; they default to ``0.0`` when unknown.
    """

    symbol: str
    direction: str
    qty: float
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    entry_fee: float
    exit_fee: float
    funding_paid: float
    gross_pnl: float
    net_pnl: float
    net_pnl_bps: float
    holding_time: float
    mfe_bps: float = 0.0
    mae_bps: float = 0.0

    @property
    def entry_notional(self) -> float:
        return abs(float(self.entry_price) * float(self.qty))

    @property
    def friction(self) -> float:
        return float(self.entry_fee) + float(self.exit_fee) + float(self.funding_paid)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _OpenLot:
    direction: str
    qty: float
    price: float
    time: float
    fee_per_unit: float


def _resolve_fill_fee(fill: FillEvent, cost_model: AttributionCostModel | None) -> float:
    if fill.fee is not None:
        return abs(float(fill.fee))
    if cost_model is None:
        return 0.0
    return cost_model.fee_for(price=fill.price, qty=fill.qty, is_maker=fill.is_maker)


def pair_round_trips_fifo(
    fills: Iterable[FillEvent],
    *,
    cost_model: AttributionCostModel | None = None,
    excursions: Sequence[tuple[float, float]] | None = None,
) -> list[RoundTrip]:
    """Pair fills into FIFO round trips with per-symbol inventory netting.

    Fills are grouped by symbol and processed in a stable chronological order
    (sorted by ``timestamp``; ties keep input order).  Opposite-direction fills
    close the oldest open lot first (FIFO); any remainder opens a new lot.  Fees
    are apportioned by matched quantity; funding is charged per matched lot via
    ``cost_model``.

    ``excursions`` optionally supplies ``(mfe_bps, mae_bps)`` per emitted round
    trip, in emission order, so the delta attribution can be driven by favorable
    / adverse excursion magnitudes.
    """
    materialized = list(fills)
    # Group by symbol, remembering original index for a fully stable ordering.
    by_symbol: dict[str, list[tuple[int, FillEvent]]] = {}
    for index, fill in enumerate(materialized):
        by_symbol.setdefault(str(fill.symbol), []).append((index, fill))

    round_trips: list[RoundTrip] = []
    for symbol in sorted(by_symbol):
        ordered = sorted(by_symbol[symbol], key=lambda item: (float(item[1].timestamp), item[0]))
        open_lots: deque[_OpenLot] = deque()
        for _index, fill in ordered:
            qty = abs(float(fill.qty))
            if qty <= _EPS:
                continue
            incoming_dir = fill.direction()
            fill_fee_total = _resolve_fill_fee(fill, cost_model)
            fee_per_unit = fill_fee_total / qty if qty > _EPS else 0.0
            remaining = qty

            while remaining > _EPS and open_lots and open_lots[0].direction != incoming_dir:
                lot = open_lots[0]
                matched = min(remaining, lot.qty)
                direction = lot.direction
                entry_price = lot.price
                exit_price = float(fill.price)
                entry_fee = matched * lot.fee_per_unit
                exit_fee = matched * fee_per_unit
                if direction == "long":
                    gross = (exit_price - entry_price) * matched
                else:
                    gross = (entry_price - exit_price) * matched
                notional = abs(entry_price * matched)
                funding_paid = (
                    cost_model.funding_for(
                        direction=direction,
                        notional=notional,
                        entry_time=lot.time,
                        exit_time=float(fill.timestamp),
                    )
                    if cost_model is not None
                    else 0.0
                )
                net = gross - entry_fee - exit_fee - funding_paid
                net_bps = (net / notional) * BPS_PER_UNIT if notional > _EPS else 0.0
                round_trips.append(
                    RoundTrip(
                        symbol=symbol,
                        direction=direction,
                        qty=matched,
                        entry_time=lot.time,
                        exit_time=float(fill.timestamp),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_fee=entry_fee,
                        exit_fee=exit_fee,
                        funding_paid=funding_paid,
                        gross_pnl=gross,
                        net_pnl=net,
                        net_pnl_bps=net_bps,
                        holding_time=float(fill.timestamp) - lot.time,
                    )
                )
                lot.qty -= matched
                remaining -= matched
                if lot.qty <= _EPS:
                    open_lots.popleft()

            if remaining > _EPS:
                open_lots.append(
                    _OpenLot(
                        direction=incoming_dir,
                        qty=remaining,
                        price=float(fill.price),
                        time=float(fill.timestamp),
                        fee_per_unit=fee_per_unit,
                    )
                )

    if excursions is not None:
        attached: list[RoundTrip] = []
        for position, trip in enumerate(round_trips):
            if position < len(excursions):
                mfe_bps, mae_bps = excursions[position]
                attached.append(
                    RoundTrip(
                        **{
                            **trip.to_dict(),
                            "mfe_bps": max(0.0, float(mfe_bps)),
                            "mae_bps": max(0.0, float(mae_bps)),
                        }
                    )
                )
            else:
                attached.append(trip)
        round_trips = attached

    return round_trips


# ── Bias severities (four pure-math functions) ───────────────────────────────


def noise_bias_severity(
    net_pnls_bps: Sequence[float], *, noise_threshold_bps: float = 2.0
) -> float:
    """Fraction of round trips whose |net edge| falls inside the noise band."""
    values = [float(x) for x in net_pnls_bps]
    if not values:
        return 0.0
    band = abs(float(noise_threshold_bps))
    inside = sum(1 for value in values if abs(value) <= band)
    return _clip01(inside / len(values))


def early_exit_bias_severity(
    realized_bps: Sequence[float], mfe_bps: Sequence[float]
) -> float:
    """Mean fraction of favorable excursion left on the table (winners only).

    For every trade with a positive favorable excursion, the shortfall ratio is
    ``clip((mfe - realized) / mfe, 0, 1)``; the severity is the average over
    those trades.  Trades with no favorable excursion do not contribute.
    """
    realized = [float(x) for x in realized_bps]
    mfe = [float(x) for x in mfe_bps]
    ratios: list[float] = []
    for r, m in zip(realized, mfe, strict=False):
        if m <= _EPS:
            continue
        ratios.append(_clip01((m - r) / m))
    if not ratios:
        return 0.0
    return _clip01(sum(ratios) / len(ratios))


def late_exit_bias_severity(
    realized_bps: Sequence[float],
    mfe_bps: Sequence[float],
    mae_bps: Sequence[float],
) -> float:
    """Mean giveback attributable to round-tripping through an adverse move.

    For each trade the giveback fraction is
    ``clip(min(mfe - realized, mae) / mfe, 0, 1)`` — the missed upside that an
    adverse excursion (``mae``) can account for, normalized by the favorable
    excursion.  Averaged over trades with a favorable excursion.
    """
    realized = [float(x) for x in realized_bps]
    mfe = [float(x) for x in mfe_bps]
    mae = [float(x) for x in mae_bps]
    ratios: list[float] = []
    for idx, m in enumerate(mfe):
        if m <= _EPS:
            continue
        r = realized[idx] if idx < len(realized) else 0.0
        a = mae[idx] if idx < len(mae) else 0.0
        unrealized = max(0.0, m - r)
        giveback = min(unrealized, max(0.0, a))
        ratios.append(_clip01(giveback / m))
    if not ratios:
        return 0.0
    return _clip01(sum(ratios) / len(ratios))


def overtrading_bias_severity(total_friction: float, total_gross_abs: float) -> float:
    """Cost-drag ratio ``friction / (friction + |gross|)`` in ``[0, 1]``."""
    friction = abs(float(total_friction))
    gross = abs(float(total_gross_abs))
    denom = friction + gross
    if denom <= _EPS:
        return 0.0
    return _clip01(friction / denom)


@dataclass(frozen=True, slots=True)
class ExecutionBiasSeverity:
    """Container for the four bias severities plus round-trip count."""

    artifact_kind: str = "execution_bias_severity"
    round_trip_count: int = 0
    noise: float = 0.0
    early_exit: float = 0.0
    late_exit: float = 0.0
    overtrading: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _severities_from_round_trips(
    round_trips: Sequence[RoundTrip], *, noise_threshold_bps: float
) -> ExecutionBiasSeverity:
    realized_bps = [trip.net_pnl_bps for trip in round_trips]
    mfe_bps = [trip.mfe_bps for trip in round_trips]
    mae_bps = [trip.mae_bps for trip in round_trips]
    total_friction = sum(trip.friction for trip in round_trips)
    total_gross_abs = sum(abs(trip.gross_pnl) for trip in round_trips)
    return ExecutionBiasSeverity(
        round_trip_count=len(round_trips),
        noise=noise_bias_severity(realized_bps, noise_threshold_bps=noise_threshold_bps),
        early_exit=early_exit_bias_severity(realized_bps, mfe_bps),
        late_exit=late_exit_bias_severity(realized_bps, mfe_bps, mae_bps),
        overtrading=overtrading_bias_severity(total_friction, total_gross_abs),
    )


# ── Delta-PnL attribution ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExecutionAttribution:
    """Delta-PnL attribution buckets (cash, quote ccy).

    ``total_delta = benchmark_pnl - realized_pnl`` is split into
    ``noise / early / late / overtrading`` explained components, with ``missed``
    holding the exact residual so that

        ``noise + early + late + overtrading + missed == total_delta``

    holds by construction (up to float round-off carried entirely in
    ``missed``).
    """

    artifact_kind: str = "execution_delta_attribution"
    round_trip_count: int = 0
    benchmark_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_delta: float = 0.0
    noise: float = 0.0
    early: float = 0.0
    late: float = 0.0
    overtrading: float = 0.0
    missed: float = 0.0

    def bucket_sum(self) -> float:
        return self.noise + self.early + self.late + self.overtrading + self.missed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def attribute_execution_delta(
    round_trips: Sequence[RoundTrip], *, noise_threshold_bps: float = 2.0
) -> ExecutionAttribution:
    """Attribute the benchmark-vs-realized PnL gap into conserved buckets.

    Benchmark PnL per trade is the frictionless favorable-excursion capture
    ``mfe_cash = (mfe_bps / 1e4) * entry_notional``.  The per-trade gap
    ``g = mfe_cash - realized_net`` is allocated as:

        * noise trades (``|net_bps| <= band``): the whole ``g`` is noise;
        * otherwise ``g = (mfe_cash - gross) + friction`` splits into an
          unrealized-upside part ``u = max(0, mfe_cash - gross)`` and the
          explicit ``friction`` (fees + funding, booked to overtrading).  ``u``
          is further split into ``late = min(u, mae_cash)`` (round-tripped
          through an adverse move) and ``early = u - late`` (exited before the
          move materialized).

    ``missed`` absorbs the residual so the buckets always sum to ``total_delta``.
    """
    band = abs(float(noise_threshold_bps))
    benchmark = 0.0
    realized = 0.0
    noise = 0.0
    early = 0.0
    late = 0.0
    overtrading = 0.0

    for trip in round_trips:
        entry_notional = trip.entry_notional
        mfe_cash = (float(trip.mfe_bps) / BPS_PER_UNIT) * entry_notional
        mae_cash = (float(trip.mae_bps) / BPS_PER_UNIT) * entry_notional
        realized_net = float(trip.net_pnl)
        gap = mfe_cash - realized_net
        benchmark += mfe_cash
        realized += realized_net

        if abs(float(trip.net_pnl_bps)) <= band:
            noise += gap
            continue

        friction = trip.friction
        unrealized = max(0.0, mfe_cash - float(trip.gross_pnl))
        late_part = min(unrealized, max(0.0, mae_cash))
        early_part = unrealized - late_part
        overtrading += friction
        early += early_part
        late += late_part

    total_delta = benchmark - realized
    missed = total_delta - (noise + early + late + overtrading)
    return ExecutionAttribution(
        round_trip_count=len(round_trips),
        benchmark_pnl=benchmark,
        realized_pnl=realized,
        total_delta=total_delta,
        noise=noise,
        early=early,
        late=late,
        overtrading=overtrading,
        missed=missed,
    )


# ── Top-level report ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExecutionAttributionReport:
    """Combined round trips + severities + attribution (all new objects)."""

    artifact_kind: str = "execution_attribution_report"
    round_trips: tuple[RoundTrip, ...] = field(default_factory=tuple)
    severities: ExecutionBiasSeverity = field(default_factory=ExecutionBiasSeverity)
    attribution: ExecutionAttribution = field(default_factory=ExecutionAttribution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "round_trips": [trip.to_dict() for trip in self.round_trips],
            "severities": self.severities.to_dict(),
            "attribution": self.attribution.to_dict(),
        }


def run_execution_attribution(
    fills: Iterable[FillEvent],
    *,
    cost_model: AttributionCostModel | None = None,
    excursions: Sequence[tuple[float, float]] | None = None,
) -> ExecutionAttributionReport:
    """Full offline pipeline: FIFO pairing → severities → delta attribution."""
    model = cost_model if cost_model is not None else AttributionCostModel()
    round_trips = pair_round_trips_fifo(fills, cost_model=model, excursions=excursions)
    severities = _severities_from_round_trips(
        round_trips, noise_threshold_bps=model.noise_threshold_bps
    )
    attribution = attribute_execution_delta(
        round_trips, noise_threshold_bps=model.noise_threshold_bps
    )
    return ExecutionAttributionReport(
        round_trips=tuple(round_trips),
        severities=severities,
        attribution=attribution,
    )
