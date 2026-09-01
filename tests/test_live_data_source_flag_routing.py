from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.live import data_poll, data_ws


def test_live_data_poll_routes_to_committed_by_default(monkeypatch):
    captured = {}

    class _Committed:
        def __init__(self, events, symbol_list, config, exchange):
            captured["handler"] = "committed"
            captured["symbols"] = list(symbol_list)
            _ = (events, config, exchange)

    class _BinanceLive:
        def __init__(self, *args, **kwargs):
            captured["handler"] = "binance_futures"
            _ = (args, kwargs)

    monkeypatch.setattr(data_poll, "_committed_handler_cls", lambda: _Committed)
    monkeypatch.setattr(data_poll, "_binance_futures_handler_cls", lambda: _BinanceLive)

    cfg = SimpleNamespace(MARKET_DATA_SOURCE="committed")
    _ = data_poll.LiveDataHandler(SimpleNamespace(), ["BTC/USDT"], cfg, SimpleNamespace())
    assert captured["handler"] == "committed"
    assert captured["symbols"] == ["BTC/USDT"]


def test_live_data_poll_routes_to_binance_futures_when_enabled(monkeypatch):
    captured = {}

    class _Committed:
        def __init__(self, *args, **kwargs):
            captured["handler"] = "committed"
            _ = (args, kwargs)

    class _BinanceLive:
        def __init__(self, events, symbol_list, config, exchange, *, transport):
            captured["handler"] = "binance_futures"
            captured["transport"] = transport
            _ = (events, symbol_list, config, exchange)

    monkeypatch.setattr(data_poll, "_committed_handler_cls", lambda: _Committed)
    monkeypatch.setattr(data_poll, "_binance_futures_handler_cls", lambda: _BinanceLive)

    cfg = SimpleNamespace(MARKET_DATA_SOURCE="binance_futures")
    _ = data_poll.LiveDataHandler(SimpleNamespace(), ["BTC/USDT"], cfg, SimpleNamespace())
    assert captured["handler"] == "binance_futures"
    assert captured["transport"] == "poll"


def test_live_data_ws_routes_to_binance_futures_when_enabled(monkeypatch):
    captured = {}

    class _Committed:
        def __init__(self, *args, **kwargs):
            captured["handler"] = "committed"
            _ = (args, kwargs)

    class _BinanceLive:
        def __init__(self, events, symbol_list, config, exchange, *, transport):
            captured["handler"] = "binance_futures"
            captured["transport"] = transport
            _ = (events, symbol_list, config, exchange)

    monkeypatch.setattr(data_ws, "_committed_handler_cls", lambda: _Committed)
    monkeypatch.setattr(data_ws, "_binance_futures_handler_cls", lambda: _BinanceLive)

    cfg = SimpleNamespace(MARKET_DATA_SOURCE="binance_futures")
    _ = data_ws.BinanceWebSocketDataHandler(SimpleNamespace(), ["BTC/USDT"], cfg, SimpleNamespace())
    assert captured["handler"] == "binance_futures"
    assert captured["transport"] == "ws"


def test_live_data_poll_routes_to_external_when_enabled(monkeypatch):
    captured = {}

    class _External:
        def __init__(self, events, symbol_list, config, exchange):
            captured["handler"] = "external"
            _ = (events, symbol_list, config, exchange)

    monkeypatch.setattr(data_poll, "_external_handler_cls", lambda: _External)
    cfg = SimpleNamespace(MARKET_DATA_SOURCE="external")
    _ = data_poll.LiveDataHandler(SimpleNamespace(), ["BTC/USDT"], cfg, SimpleNamespace())
    assert captured["handler"] == "external"


def test_live_data_ws_routes_to_polymarket_when_enabled(monkeypatch):
    captured = {}

    class _Polymarket:
        def __init__(self, events, symbol_list, config, exchange, *, transport):
            captured["handler"] = "polymarket_live"
            captured["transport"] = transport
            _ = (events, symbol_list, config, exchange)

    monkeypatch.setattr(data_ws, "_polymarket_live_handler_cls", lambda: _Polymarket)
    cfg = SimpleNamespace(MARKET_DATA_SOURCE="polymarket_live")
    _ = data_ws.BinanceWebSocketDataHandler(SimpleNamespace(), ["asset-1"], cfg, SimpleNamespace())
    assert captured["handler"] == "polymarket_live"
    assert captured["transport"] == "ws"
