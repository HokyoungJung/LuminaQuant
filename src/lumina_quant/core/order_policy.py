"""Shared order policy helpers.

The live stack is limit-first: market orders remain available only behind an
explicit live configuration flag.  These helpers are intentionally dependency
light so both portfolio order generation and live risk-flatten paths can share
the same price/tick semantics.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from typing import Any

MARKET_ORDER_ALIASES = {"MKT", "MARKET"}
LIMIT_ORDER_ALIASES = {"LMT", "LIMIT"}
SUPPORTED_ORDER_TYPES = MARKET_ORDER_ALIASES | LIMIT_ORDER_ALIASES

SAME_PRICE_MODE = "same_price"
ONE_TICK_WORSE_MODE = "one_tick_worse"
ONE_TICK_BETTER_MODE = "one_tick_better"
SUPPORTED_LIMIT_PRICE_MODES = {
    SAME_PRICE_MODE,
    ONE_TICK_WORSE_MODE,
    ONE_TICK_BETTER_MODE,
}


def canonical_order_type(value: Any, *, default: str = "LMT") -> str:
    """Return the internal canonical order type (``LMT`` or ``MKT``)."""
    token = str(value or default or "LMT").strip().upper()
    if token in LIMIT_ORDER_ALIASES:
        return "LMT"
    if token in MARKET_ORDER_ALIASES:
        return "MKT"
    return canonical_order_type(default if token else "LMT", default="LMT")


def policy_order_type(
    value: Any,
    *,
    default: str = "LMT",
    allow_market_orders: bool = False,
) -> str:
    """Resolve the policy order type, coercing market to limit unless allowed."""
    resolved = canonical_order_type(value, default=default)
    if resolved == "MKT" and not bool(allow_market_orders):
        return "LMT"
    return resolved


def normalize_limit_price_mode(value: Any, *, default: str = ONE_TICK_WORSE_MODE) -> str:
    """Normalize the configured limit-price mode."""
    token = str(value or default or ONE_TICK_WORSE_MODE).strip().lower().replace("-", "_")
    aliases = {
        "same": SAME_PRICE_MODE,
        "current": SAME_PRICE_MODE,
        "current_price": SAME_PRICE_MODE,
        "at_price": SAME_PRICE_MODE,
        "at_signal_price": SAME_PRICE_MODE,
        "same_price": SAME_PRICE_MODE,
        "one_tick": ONE_TICK_WORSE_MODE,
        "one_tick_cross": ONE_TICK_WORSE_MODE,
        "one_tick_aggressive": ONE_TICK_WORSE_MODE,
        "one_tick_worse": ONE_TICK_WORSE_MODE,
        "aggressive": ONE_TICK_WORSE_MODE,
        "immediate": ONE_TICK_WORSE_MODE,
        "one_tick_better": ONE_TICK_BETTER_MODE,
        "passive": ONE_TICK_BETTER_MODE,
        "maker": ONE_TICK_BETTER_MODE,
    }
    normalized = aliases.get(token, token)
    if normalized in SUPPORTED_LIMIT_PRICE_MODES:
        return normalized
    return normalize_limit_price_mode(default, default=ONE_TICK_WORSE_MODE)


def _positive_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except TypeError, ValueError:
        return float(default)
    return out if out > 0.0 else float(default)


def price_tick_size_from_sources(
    symbol: str,
    *,
    market_spec: dict[str, Any] | None = None,
    config: Any | None = None,
) -> float:
    """Resolve a tradable price tick from exchange spec, symbol limits, or fallback."""
    spec = dict(market_spec or {})
    for key in (
        "price_tick_size",
        "tick_size",
        "price_step",
        "price_tick",
        "min_price_increment",
    ):
        tick = _positive_float(spec.get(key), 0.0)
        if tick > 0.0:
            return tick

    symbol_limits = dict(getattr(config, "SYMBOL_LIMITS", {}) or {}) if config is not None else {}
    for candidate in (symbol, str(symbol).replace("/", ""), str(symbol).replace("/", "").upper()):
        limits = dict(symbol_limits.get(candidate, {}) or {})
        for key in (
            "price_tick_size",
            "tick_size",
            "price_step",
            "price_tick",
            "min_price_increment",
        ):
            tick = _positive_float(limits.get(key), 0.0)
            if tick > 0.0:
                return tick

    return _positive_float(getattr(config, "LIMIT_PRICE_TICK_FALLBACK", 0.0), 0.0)


def _decimal(value: float | int | str) -> Decimal:
    return Decimal(str(value))


def limit_price_for_direction(
    *,
    reference_price: float,
    direction: str,
    tick_size: float,
    mode: str = ONE_TICK_WORSE_MODE,
    offset_ticks: int = 1,
) -> float:
    """Return a side-safe limit price.

    ``one_tick_worse`` is the live default: BUY prices one tick above the
    rounded reference and SELL prices one tick below it so the order is a
    marketable limit rather than an unlimited market order.
    """
    ref = _positive_float(reference_price, 0.0)
    if ref <= 0.0:
        raise ValueError("reference_price must be positive for limit order pricing.")

    normalized_mode = normalize_limit_price_mode(mode)
    direction_token = str(direction or "").strip().upper()
    if direction_token not in {"BUY", "SELL"}:
        raise ValueError("direction must be BUY or SELL for limit order pricing.")

    tick = _positive_float(tick_size, 0.0)
    if tick <= 0.0:
        return ref

    try:
        ref_dec = _decimal(ref)
        tick_dec = _decimal(tick)
        base_ceil = (ref_dec / tick_dec).to_integral_value(rounding=ROUND_CEILING)
        base_floor = (ref_dec / tick_dec).to_integral_value(rounding=ROUND_FLOOR)
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise ValueError("invalid tick_size for limit order pricing.") from exc

    offset = max(0, int(offset_ticks or 0))
    if normalized_mode == SAME_PRICE_MODE:
        steps = base_ceil if direction_token == "BUY" else base_floor
    elif normalized_mode == ONE_TICK_WORSE_MODE:
        steps = (base_ceil + offset) if direction_token == "BUY" else (base_floor - offset)
    elif normalized_mode == ONE_TICK_BETTER_MODE:
        steps = (base_ceil - offset) if direction_token == "BUY" else (base_floor + offset)
    else:  # pragma: no cover - normalize_limit_price_mode prevents this.
        steps = base_ceil if direction_token == "BUY" else base_floor

    if steps < 1:
        steps = Decimal(1)
    return float(steps * tick_dec)


def merge_order_policy_metadata(
    metadata: dict[str, Any] | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Attach order policy details without discarding existing metadata."""
    merged = dict(metadata or {})
    existing = merged.get("order_policy")
    order_policy = dict(existing) if isinstance(existing, dict) else {}
    order_policy.update(policy)
    merged["order_policy"] = order_policy
    return merged


__all__ = [
    "LIMIT_ORDER_ALIASES",
    "MARKET_ORDER_ALIASES",
    "ONE_TICK_BETTER_MODE",
    "ONE_TICK_WORSE_MODE",
    "SAME_PRICE_MODE",
    "SUPPORTED_LIMIT_PRICE_MODES",
    "SUPPORTED_ORDER_TYPES",
    "canonical_order_type",
    "limit_price_for_direction",
    "merge_order_policy_metadata",
    "normalize_limit_price_mode",
    "policy_order_type",
    "price_tick_size_from_sources",
]
