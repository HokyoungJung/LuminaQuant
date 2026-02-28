"""Cost model and execution primitives for the cost-aware framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class CostModelParams:
    spread_bps: float = 4.0
    impact_k: float = 1.0
    fees_bps: float = 1.0
    tax_bps: float = 0.0
    participation_lambda: float = 0.0
    participation_power: float = 1.0


@dataclass(slots=True)
class ExecutionPolicy:
    fill_basis: str = "next_open"  # next_open | mid
    max_participation: float = 0.1
    unfilled_policy: str = "carry"  # carry | drop
    carry_decay: float = 1.0


@dataclass(slots=True)
class CostBreakdown:
    spread_bps: float
    impact_bps: float
    fees_bps: float
    total_bps: float
    participation: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionFill:
    requested_notional: float
    executed_notional: float
    unfilled_notional: float
    next_pending_notional: float
    basis_price: float
    fill_price: float
    participation: float
    spread_bps: float
    impact_bps: float
    fees_bps: float
    total_bps: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def estimate_cost_bps(
    *,
    order_notional: float,
    sigma: float,
    adtv: float,
    bar_volume: float,
    price: float,
    params: CostModelParams,
) -> CostBreakdown:
    notional = abs(float(order_notional))
    safe_price = max(1e-9, float(price))
    safe_adtv = max(1e-9, float(adtv))
    volume_qty = max(1e-9, float(bar_volume))
    participation = (notional / safe_price) / volume_qty

    spread_bps = max(0.0, float(params.spread_bps))
    impact_bps = max(0.0, float(params.impact_k) * max(0.0, float(sigma)) * ((notional / safe_adtv) ** 0.5) * 10_000.0)
    fees_bps = max(0.0, float(params.fees_bps) + float(params.tax_bps))
    participation_penalty = max(
        0.0,
        float(params.participation_lambda) * (max(0.0, participation) ** max(0.1, float(params.participation_power))),
    )
    total_bps = spread_bps + impact_bps + fees_bps + participation_penalty
    return CostBreakdown(
        spread_bps=spread_bps,
        impact_bps=impact_bps,
        fees_bps=fees_bps,
        total_bps=total_bps,
        participation=max(0.0, participation),
    )


def apply_bps_to_price(*, basis_price: float, total_bps: float, side: str) -> float:
    sign = 1.0 if str(side).upper() == "BUY" else -1.0
    return float(basis_price) * (1.0 + sign * (float(total_bps) / 10_000.0))


def simulate_market_fill(
    *,
    order_notional: float,
    pending_notional: float,
    next_open: float,
    next_mid: float | None,
    close_price: float | None,
    adtv: float,
    sigma: float,
    bar_volume: float,
    params: CostModelParams,
    policy: ExecutionPolicy,
) -> ExecutionFill:
    """Simulate one aggressive execution using next_open or mid basis (never close)."""
    requested = float(order_notional) + float(pending_notional)
    if abs(requested) < 1e-12:
        base = float(next_mid if policy.fill_basis == "mid" and next_mid is not None else next_open)
        return ExecutionFill(0.0, 0.0, 0.0, 0.0, base, base, 0.0, 0.0, 0.0, 0.0, 0.0)

    basis = str(policy.fill_basis).strip().lower()
    if basis not in {"next_open", "mid"}:
        raise ValueError(f"Unsupported fill basis: {policy.fill_basis}")

    if basis == "mid":
        if next_mid is None:
            raise ValueError("fill_basis='mid' requires next_mid price")
        basis_price = float(next_mid)
    else:
        basis_price = float(next_open)

    _ = close_price  # explicitly ignored to enforce no-close-fill rule

    max_participation = max(0.0, float(policy.max_participation))
    cap_notional = max_participation * max(0.0, float(bar_volume)) * max(1e-9, basis_price)
    if cap_notional <= 0.0:
        executed = 0.0
    elif requested > 0.0:
        executed = min(requested, cap_notional)
    else:
        executed = max(requested, -cap_notional)

    unfilled = requested - executed
    if str(policy.unfilled_policy).strip().lower() == "carry":
        next_pending = unfilled * max(0.0, float(policy.carry_decay))
    else:
        next_pending = 0.0

    side = "BUY" if executed >= 0 else "SELL"
    cost = estimate_cost_bps(
        order_notional=executed,
        sigma=sigma,
        adtv=adtv,
        bar_volume=bar_volume,
        price=basis_price,
        params=params,
    )
    fill_price = apply_bps_to_price(basis_price=basis_price, total_bps=cost.total_bps, side=side)

    return ExecutionFill(
        requested_notional=requested,
        executed_notional=executed,
        unfilled_notional=unfilled,
        next_pending_notional=next_pending,
        basis_price=basis_price,
        fill_price=fill_price,
        participation=cost.participation,
        spread_bps=cost.spread_bps,
        impact_bps=cost.impact_bps,
        fees_bps=cost.fees_bps,
        total_bps=cost.total_bps,
    )


__all__ = [
    "CostBreakdown",
    "CostModelParams",
    "ExecutionFill",
    "ExecutionPolicy",
    "apply_bps_to_price",
    "estimate_cost_bps",
    "simulate_market_fill",
]
