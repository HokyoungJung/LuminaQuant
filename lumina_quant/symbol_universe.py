"""Helpers for validating requested symbols against exchange market metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_KNOWN_QUOTES = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")


def _normalize_symbol(symbol: str) -> str:
    token = str(symbol or "").strip().upper().replace("_", "/").replace("-", "/")
    while "//" in token:
        token = token.replace("//", "/")
    if "/" not in token:
        for quote in _KNOWN_QUOTES:
            if token.endswith(quote) and len(token) > len(quote):
                token = f"{token[: -len(quote)]}/{quote}"
                break
    return token


def _symbol_aliases(symbol: str) -> set[str]:
    base = _normalize_symbol(symbol)
    if not base:
        return set()

    aliases = {base}
    aliases.add(base.split(":", 1)[0])
    for token in tuple(aliases):
        for suffix in ("/PERP", "_PERP", "-PERP"):
            if token.endswith(suffix):
                aliases.add(token[: -len(suffix)])
        aliases.add(token.replace("/", ""))
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
        symbol = _normalize_symbol(str(raw))
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


__all__ = ["resolve_available_symbols"]
