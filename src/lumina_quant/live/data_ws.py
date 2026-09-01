"""Live WS handler with feature-flagged live market-data routing."""

from __future__ import annotations

from lumina_quant.live import _source_routing


def _resolve_market_data_source(config) -> str:
    return _source_routing.resolve_market_data_source(config)


def _binance_futures_handler_cls():
    return _source_routing.binance_futures_handler_cls()


def _binance_live_handler_cls():
    return _binance_futures_handler_cls()


def _external_handler_cls():
    return _source_routing.external_handler_cls()


def _polymarket_live_handler_cls():
    return _source_routing.polymarket_live_handler_cls()


def _committed_handler_cls():
    return _source_routing.committed_handler_cls()


class BinanceWebSocketDataHandler:
    """Factory wrapper preserving WS entrypoint compatibility."""

    def __new__(cls, events, symbol_list, config, exchange=None):
        source = _resolve_market_data_source(config)
        if source == "binance_futures":
            return _binance_futures_handler_cls()(
                events,
                symbol_list,
                config,
                exchange,
                transport="ws",
            )
        if source == "external":
            return _external_handler_cls()(events, symbol_list, config, exchange)
        if source == "polymarket_live":
            return _polymarket_live_handler_cls()(
                events,
                symbol_list,
                config,
                exchange,
                transport="ws",
            )
        return _committed_handler_cls()(events, symbol_list, config, exchange)


__all__ = ["BinanceWebSocketDataHandler"]
