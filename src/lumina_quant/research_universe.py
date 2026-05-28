"""Canonical static research universes for data coverage and alpha monitoring.

The lists in this module are side-effect-free snapshots. They do not collect data,
place orders, or imply live/real-money eligibility. Strategy runners may use these
symbols for shadow research only until the standard train/validation/refit and
paper/testnet gates pass.
"""

from __future__ import annotations

BINANCE_RESEARCH_UNIVERSE_SNAPSHOT_UTC = "2026-05-28T13:40:47Z"
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
)

BINANCE_TRADFI_PREMARKET_SYMBOLS: tuple[str, ...] = (
    "SPCXUSDT",
    "OPENAIUSDT",
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
    "compact_to_slashed_usdt",
]
