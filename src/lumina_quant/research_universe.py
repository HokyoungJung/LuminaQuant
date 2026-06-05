"""Canonical static research universes for data coverage and alpha monitoring.

The lists in this module are side-effect-free snapshots. They do not collect data,
place orders, or imply live/real-money eligibility. Strategy runners may use these
symbols for shadow research only until the standard train/validation/refit and
paper/testnet gates pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BINANCE_RESEARCH_UNIVERSE_SNAPSHOT_UTC = "2026-06-05T12:28:39Z"
BINANCE_RESEARCH_UNIVERSE_SOURCE = "Binance USD-M Futures /fapi/v1/exchangeInfo"

BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "TRXUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "TONUSDT",
)

BINANCE_TRADFI_PRECIOUS_METAL_SYMBOLS: tuple[str, ...] = (
    "XAUUSDT",
    "XAGUSDT",
    "XPTUSDT",
    "XPDUSDT",
)

BINANCE_TRADFI_ENERGY_INDUSTRIAL_COMMODITY_SYMBOLS: tuple[str, ...] = (
    "COPPERUSDT",
    "CLUSDT",
    "BZUSDT",
    "NATGASUSDT",
)

BINANCE_TRADFI_COMMODITY_SYMBOLS: tuple[str, ...] = (
    *BINANCE_TRADFI_PRECIOUS_METAL_SYMBOLS,
    *BINANCE_TRADFI_ENERGY_INDUSTRIAL_COMMODITY_SYMBOLS,
)

BINANCE_TRADFI_ETF_INDEX_SYMBOLS: tuple[str, ...] = (
    "QQQUSDT",
    "SPYUSDT",
    "EWYUSDT",
    "EWJUSDT",
    "SOXLUSDT",
    "EWTUSDT",
)

BINANCE_TRADFI_EQUITY_SYMBOLS: tuple[str, ...] = (
    "TSLAUSDT",
    "INTCUSDT",
    "HOODUSDT",
    "MSTRUSDT",
    "AMZNUSDT",
    "CRCLUSDT",
    "COINUSDT",
    "PLTRUSDT",
    "PAYPUSDT",
    "METAUSDT",
    "NVDAUSDT",
    "GOOGLUSDT",
    "AAPLUSDT",
    "TSMUSDT",
    "MUUSDT",
    "SNDKUSDT",
    "MSFTUSDT",
    "AVGOUSDT",
    "BABAUSDT",
    "AMDUSDT",
    "QCOMUSDT",
    "USARUSDT",
    "LITEUSDT",
    "ORCLUSDT",
    "DISUSDT",
    "UBERUSDT",
    "CSCOUSDT",
    "HDUSDT",
    "MRVLUSDT",
    "CRWVUSDT",
    "WMTUSDT",
    "JPMUSDT",
    "VUSDT",
    "BRKBUSDT",
    "FLNCUSDT",
    "DRAMUSDT",
    "RKLBUSDT",
    "CBRSUSDT",
    "NBISUSDT",
    "WDCUSDT",
    "ARMUSDT",
    "BEUSDT",
    "COHRUSDT",
    "ASTSUSDT",
    "BBXUSDT",
    "CRMUSDT",
    "DELLUSDT",
    "HYUNDAIUSDT",
    "IBMUSDT",
    "IRENUSDT",
    "LLYUSDT",
    "NOKUSDT",
    "NOWUSDT",
    "NVOUSDT",
    "ONDSUSDT",
    "SAMSUNGUSDT",
    "SKHYNIXUSDT",
)

BINANCE_TRADFI_PREMARKET_SYMBOLS: tuple[str, ...] = (
    "SPCXUSDT",
    "OPENAIUSDT",
    "QNTXUSDT",
    "ANTHROPICUSDT",
)

BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS: tuple[str, ...] = (
    *BINANCE_TRADFI_COMMODITY_SYMBOLS,
    *BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    *BINANCE_TRADFI_EQUITY_SYMBOLS,
    *BINANCE_TRADFI_PREMARKET_SYMBOLS,
)

BINANCE_EXTENDED_RESEARCH_SYMBOLS: tuple[str, ...] = (
    *BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS,
    *BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS,
)


def _compact_usdt_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper().replace("/", "").replace("-", "")


def binance_tradfi_perp_symbols_from_exchange_info(
    exchange_info: Mapping[str, Mapping[str, Any]] | Mapping[str, Any],
    *,
    trading_only: bool = True,
) -> tuple[str, ...]:
    """Return currently listed Binance USD-M TRADIFI_PERPETUAL USDT symbols.

    The canonical constants above are intentionally static snapshots for
    reproducible research.  Collectors can call this helper with a freshly loaded
    ``/fapi/v1/exchangeInfo`` payload to keep monitoring newly supported TradFi
    perps without making this module perform network I/O at import time.
    """
    rows: list[Mapping[str, Any]]
    if isinstance(
        exchange_info.get("symbols") if isinstance(exchange_info, Mapping) else None, list
    ):
        rows = [row for row in exchange_info.get("symbols", []) if isinstance(row, Mapping)]
    else:
        rows = [row for row in exchange_info.values() if isinstance(row, Mapping)]

    symbols: set[str] = set()
    for row in rows:
        if str(row.get("contractType") or "").upper() != "TRADIFI_PERPETUAL":
            continue
        if str(row.get("quoteAsset") or "").upper() != "USDT":
            continue
        if trading_only and str(row.get("status") or "").upper() != "TRADING":
            continue
        symbol = _compact_usdt_symbol(row.get("symbol"))
        if symbol.endswith("USDT") and len(symbol) > len("USDT"):
            symbols.add(symbol)
    return tuple(sorted(symbols))


def binance_extended_research_symbols_from_exchange_info(
    exchange_info: Mapping[str, Mapping[str, Any]] | Mapping[str, Any],
    *,
    include_static_tradfi_snapshot: bool = True,
    trading_only: bool = True,
) -> tuple[str, ...]:
    """Return core crypto plus static and currently discovered TradFi perps.

    This is the long-term expansion path: keep all existing 69 snapshot symbols
    reproducible, then append any newly supported Binance TRADIFI_PERPETUAL USDT
    contracts for monitoring/backfill as soon as exchangeInfo exposes them.
    """
    ordered: list[str] = []
    for symbol in BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS:
        token = _compact_usdt_symbol(symbol)
        if token not in ordered:
            ordered.append(token)
    if include_static_tradfi_snapshot:
        for symbol in BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS:
            token = _compact_usdt_symbol(symbol)
            if token not in ordered:
                ordered.append(token)
    for symbol in binance_tradfi_perp_symbols_from_exchange_info(
        exchange_info,
        trading_only=trading_only,
    ):
        if symbol not in ordered:
            ordered.append(symbol)
    return tuple(ordered)


def compact_to_slashed_usdt(symbol: str) -> str:
    """Convert compact Binance USDT symbols into ``BASE/USDT`` notation."""
    token = str(symbol).strip().upper().replace("/", "").replace("-", "")
    if not token.endswith("USDT") or len(token) <= len("USDT"):
        raise ValueError(f"expected compact USDT symbol, got {symbol!r}")
    return f"{token[:-4]}/USDT"


BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED: tuple[str, ...] = tuple(
    compact_to_slashed_usdt(symbol) for symbol in BINANCE_EXTENDED_RESEARCH_SYMBOLS
)
BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS_SLASHED: tuple[str, ...] = tuple(
    compact_to_slashed_usdt(symbol) for symbol in BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS
)

__all__ = [
    "BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS",
    "BINANCE_EXTENDED_RESEARCH_SYMBOLS",
    "BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED",
    "BINANCE_RESEARCH_UNIVERSE_SNAPSHOT_UTC",
    "BINANCE_RESEARCH_UNIVERSE_SOURCE",
    "BINANCE_TRADFI_COMMODITY_SYMBOLS",
    "BINANCE_TRADFI_ENERGY_INDUSTRIAL_COMMODITY_SYMBOLS",
    "BINANCE_TRADFI_EQUITY_SYMBOLS",
    "BINANCE_TRADFI_ETF_INDEX_SYMBOLS",
    "BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS",
    "BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS_SLASHED",
    "BINANCE_TRADFI_PRECIOUS_METAL_SYMBOLS",
    "BINANCE_TRADFI_PREMARKET_SYMBOLS",
    "binance_extended_research_symbols_from_exchange_info",
    "binance_tradfi_perp_symbols_from_exchange_info",
    "compact_to_slashed_usdt",
]
