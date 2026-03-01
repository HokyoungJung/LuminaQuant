"""Symbol and timeframe canonicalization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

KNOWN_QUOTES: tuple[str, ...] = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
CANONICAL_STRATEGY_TIMEFRAMES: tuple[str, ...] = (
    "1s",
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
)


def _normalize_raw_symbol(symbol: str) -> str:
    token = str(symbol or "").strip().upper().replace("_", "/").replace("-", "/")
    while "//" in token:
        token = token.replace("//", "/")
    return token


def canonical_symbol(symbol: str) -> str:
    """Canonicalize symbol to ``BASE/QUOTE`` while accepting legacy futures suffixes.

    Examples:
    - ``BTCUSDT`` -> ``BTC/USDT``
    - ``XAU/USDT:USDT`` -> ``XAU/USDT``
    - ``eth-usdt`` -> ``ETH/USDT``
    """
    token = _normalize_raw_symbol(symbol)
    if not token:
        return ""

    if ":" in token:
        token = token.split(":", 1)[0]

    for suffix in ("/PERP", "_PERP", "-PERP", "PERP"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break

    if "/" not in token:
        for quote in KNOWN_QUOTES:
            if token.endswith(quote) and len(token) > len(quote):
                token = f"{token[: -len(quote)]}/{quote}"
                break

    if "/" not in token:
        return token

    base, quote = token.split("/", 1)
    quote = quote.split(":", 1)[0]
    if quote.startswith("USDT"):
        quote = "USDT"
    return f"{base}/{quote}"


def normalize_symbol(symbol: str) -> str:
    """Backwards-compatible alias for canonical symbol normalization."""
    return canonical_symbol(symbol)


def symbol_aliases(symbol: str) -> set[str]:
    """Return canonical + compatibility aliases for market lookup."""
    canonical = canonical_symbol(symbol)
    if not canonical:
        return set()

    aliases = {
        canonical,
        canonical.replace("/", ""),
        canonical.replace("/", "-"),
        canonical.replace("/", "_"),
    }
    base, _, quote = canonical.partition("/")
    if base and quote:
        aliases.add(f"{base}/{quote}:{quote}")
    return {item for item in aliases if item}


def canonicalize_symbol_list(symbols: Iterable[str]) -> list[str]:
    """Normalize, de-duplicate and preserve order for symbol lists."""
    out: list[str] = []
    for raw in symbols:
        symbol = canonical_symbol(raw)
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def normalize_strategy_timeframes(
    timeframes: Sequence[str] | None,
    *,
    required: Sequence[str] = CANONICAL_STRATEGY_TIMEFRAMES,
    strict_subset: bool = True,
) -> list[str]:
    """Normalize timeframe tokens and optionally enforce canonical subset."""
    allowed = {str(item).strip().lower() for item in required if str(item).strip()}
    out: list[str] = []
    for raw in list(timeframes or []):
        token = str(raw).strip().lower()
        if not token:
            continue
        if strict_subset and token not in allowed:
            continue
        if token not in out:
            out.append(token)
    if out:
        return out
    return [str(item) for item in required]
