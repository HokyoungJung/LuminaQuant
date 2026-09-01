"""Shared live market-data source routing helpers."""

from __future__ import annotations


def resolve_market_data_source(config) -> str:
    """Normalize configured live market-data source into a stable token."""
    token = str(getattr(config, "MARKET_DATA_SOURCE", "committed") or "committed")
    token = token.strip().lower().replace("-", "_")
    if token in {"binance_futures", "binance_futures_live", "binance_live", "binance", "live"}:
        return "binance_futures"
    if token in {"external", "custom"}:
        return "external"
    if token in {"polymarket_live", "polymarket"}:
        return "polymarket_live"
    return "committed"


def binance_futures_handler_cls():
    from lumina_quant.live.data_binance_live import BinanceLiveDataHandler

    return BinanceLiveDataHandler


def external_handler_cls():
    from lumina_quant.live.data_external import ExternalWindowDataHandler

    return ExternalWindowDataHandler


def polymarket_live_handler_cls():
    from lumina_quant.live.data_polymarket_live import PolymarketLiveDataHandler

    return PolymarketLiveDataHandler


def committed_handler_cls():
    from lumina_quant.live.data_materialized import CommittedWindowDataHandler

    return CommittedWindowDataHandler
