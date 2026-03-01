"""Helpers for symbol canonicalization and exchange availability matching."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lumina_quant.symbols import canonical_symbol

RESEARCH_CANONICAL_QUOTE = "USDT"
REQUIRED_RESEARCH_TIMEFRAMES: tuple[str, ...] = (
    "1s",
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
)


def canonicalize_research_symbol(symbol: str) -> str:
    """Canonicalize symbols to BASE/USDT while accepting legacy aliases.

    Examples:
      BTCUSDT -> BTC/USDT
      BTC-USDT -> BTC/USDT
      XAU/USDT:USDT -> XAU/USDT
    """
    canonical = canonical_symbol(symbol)
    if not canonical:
        return ""
    base, _, _quote = canonical.partition("/")
    if not base:
        return ""
    return f"{base}/{RESEARCH_CANONICAL_QUOTE}"


def canonicalize_research_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    """Canonicalize and de-duplicate research symbols preserving order."""
    out: list[str] = []
    for raw in symbols:
        symbol = canonicalize_research_symbol(str(raw))
        if symbol and symbol not in out:
            out.append(symbol)
    return tuple(out)


def normalize_research_timeframes(
    timeframes: Iterable[str],
    *,
    strict_required_set: bool = False,
) -> tuple[str, ...]:
    """Normalize timeframe tokens and optionally enforce canonical required set."""
    out: list[str] = []
    for raw in timeframes:
        token = str(raw or "").strip().lower()
        if token and token not in out:
            out.append(token)

    if strict_required_set:
        out_set = set(out)
        required_set = set(REQUIRED_RESEARCH_TIMEFRAMES)
        if out_set != required_set:
            required = ", ".join(REQUIRED_RESEARCH_TIMEFRAMES)
            raise ValueError(
                "Strategy timeframes must match canonical set exactly: "
                f"{required}."
            )
        return tuple(token for token in REQUIRED_RESEARCH_TIMEFRAMES if token in out_set)

    if not out:
        return REQUIRED_RESEARCH_TIMEFRAMES
    return tuple(out)


def _symbol_aliases(symbol: str) -> set[str]:
    canonical = canonical_symbol(symbol)
    if not canonical:
        return set()

    aliases = {
        canonical,
        canonicalize_research_symbol(canonical),
        canonical.replace("/", ""),
        canonical.replace("/", "-"),
        canonical.replace("/", "_"),
    }

    base, _, quote = canonical.partition("/")
    if base and quote:
        aliases.add(f"{base}/{quote}:{quote}")
        aliases.add(f"{base}/{RESEARCH_CANONICAL_QUOTE}:{RESEARCH_CANONICAL_QUOTE}")
    return {item for item in aliases if item}


def _build_available_aliases(markets: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for key, payload in dict(markets or {}).items():
        aliases.update(_symbol_aliases(str(key)))
        if isinstance(payload, dict):
            symbol_token = payload.get("symbol")
            if symbol_token:
                aliases.update(_symbol_aliases(str(symbol_token)))
            market_id = payload.get("id")
            if market_id:
                aliases.update(_symbol_aliases(str(market_id)))
            base = payload.get("base")
            quote = payload.get("quote")
            if base and quote:
                aliases.update(_symbol_aliases(f"{base}/{quote}"))
    return aliases


def resolve_available_symbols(
    symbols: Iterable[str],
    markets: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Return (kept, dropped) symbols after exchange-market availability matching."""
    requested: list[str] = []
    for raw in symbols:
        symbol = canonicalize_research_symbol(str(raw))
        if symbol and symbol not in requested:
            requested.append(symbol)

    market_aliases = _build_available_aliases(dict(markets or {}))
    if not market_aliases:
        return requested, []

    kept: list[str] = []
    dropped: list[str] = []
    for symbol in requested:
        aliases = _symbol_aliases(symbol)
        if aliases.intersection(market_aliases):
            kept.append(symbol)
        else:
            dropped.append(symbol)
    return kept, dropped


__all__ = [
    "REQUIRED_RESEARCH_TIMEFRAMES",
    "RESEARCH_CANONICAL_QUOTE",
    "canonicalize_research_symbol",
    "canonicalize_research_symbols",
    "normalize_research_timeframes",
    "resolve_available_symbols",
]
