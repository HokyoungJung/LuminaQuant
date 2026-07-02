"""Forex instrument parameter seam (constants only; NO execution logic).

This module declares a frozen, data-only description of forex pip / contract /
swap conventions so downstream research analytics have a single pinned source of
those constants.  It deliberately contains no order-sizing, no P&L, and no
routing logic — it is a parameter structure the crypto/perp execution path never
imports.  Populating the registry does not enable any forex trading; it only
records conventions for offline analytics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ForexPairSpec:
    """Static pip / contract / swap conventions for a single forex pair.

    Fields are pure constants:

    * ``pip_decimal_place`` — the decimal position of one pip (4 for most pairs,
      2 for JPY-quoted pairs).
    * ``pip_size`` — the price increment of one pip (``10 ** -pip_decimal_place``).
    * ``contract_size`` — units of base currency in one standard lot.
    * ``swap_long_points`` / ``swap_short_points`` — indicative overnight swap in
      points (broker-specific; recorded as provided constants, not computed).
    """

    symbol: str
    base_currency: str
    quote_currency: str
    pip_decimal_place: int
    pip_size: float
    contract_size: float
    swap_long_points: float = 0.0
    swap_short_points: float = 0.0


def _spec(
    symbol: str,
    base: str,
    quote: str,
    pip_decimal_place: int,
    *,
    contract_size: float = 100_000.0,
    swap_long_points: float = 0.0,
    swap_short_points: float = 0.0,
) -> ForexPairSpec:
    return ForexPairSpec(
        symbol=symbol,
        base_currency=base,
        quote_currency=quote,
        pip_decimal_place=pip_decimal_place,
        pip_size=10.0 ** (-pip_decimal_place),
        contract_size=contract_size,
        swap_long_points=swap_long_points,
        swap_short_points=swap_short_points,
    )


# Data-only registry of common majors.  Swap points default to 0.0 (unset);
# callers may substitute broker-specific values without any behavior change here.
DEFAULT_FOREX_PAIR_SPECS: tuple[ForexPairSpec, ...] = (
    _spec("EUR/USD", "EUR", "USD", 4),
    _spec("GBP/USD", "GBP", "USD", 4),
    _spec("AUD/USD", "AUD", "USD", 4),
    _spec("USD/CHF", "USD", "CHF", 4),
    _spec("USD/CAD", "USD", "CAD", 4),
    _spec("USD/JPY", "USD", "JPY", 2),
    _spec("EUR/JPY", "EUR", "JPY", 2),
    _spec("GBP/JPY", "GBP", "JPY", 2),
)


def forex_pair_registry() -> dict[str, ForexPairSpec]:
    """Return a symbol-keyed copy of the default forex pair spec registry."""
    return {spec.symbol: spec for spec in DEFAULT_FOREX_PAIR_SPECS}


def get_forex_pair_spec(symbol: str) -> ForexPairSpec:
    """Look up a pinned :class:`ForexPairSpec` by symbol (fail-loud on miss)."""
    key = str(symbol or "").strip().upper()
    registry = {spec.symbol.upper(): spec for spec in DEFAULT_FOREX_PAIR_SPECS}
    try:
        return registry[key]
    except KeyError as exc:
        raise KeyError(f"unknown forex pair spec: {symbol!r}") from exc


__all__ = [
    "DEFAULT_FOREX_PAIR_SPECS",
    "ForexPairSpec",
    "forex_pair_registry",
    "get_forex_pair_spec",
]
