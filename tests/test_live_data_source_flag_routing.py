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
            captured["handler"] = "binance_live"
            _ = (args, kwargs)

    monkeypatch.setattr(data_poll, "CommittedWindowDataHandler", _Committed)
    monkeypatch.setattr(data_poll, "BinanceLiveDataHandler", _BinanceLive)

    cfg = SimpleNamespace(MARKET_DATA_SOURCE="committed")
    _ = data_poll.LiveDataHandler(SimpleNamespace(), ["BTC/USDT"], cfg, SimpleNamespace())
    assert captured["handler"] == "committed"
    assert captured["symbols"] == ["BTC/USDT"]


def test_live_data_poll_routes_to_binance_live_when_enabled(monkeypatch):
    captured = {}

    class _Committed:
        def __init__(self, *args, **kwargs):
            captured["handler"] = "committed"
            _ = (args, kwargs)

    class _BinanceLive:
        def __init__(self, events, symbol_list, config, exchange, *, transport):
            captured["handler"] = "binance_live"
            captured["transport"] = transport
            _ = (events, symbol_list, config, exchange)

    monkeypatch.setattr(data_poll, "CommittedWindowDataHandler", _Committed)
    monkeypatch.setattr(data_poll, "BinanceLiveDataHandler", _BinanceLive)

    cfg = SimpleNamespace(MARKET_DATA_SOURCE="binance_live")
    _ = data_poll.LiveDataHandler(SimpleNamespace(), ["BTC/USDT"], cfg, SimpleNamespace())
    assert captured["handler"] == "binance_live"
    assert captured["transport"] == "poll"


def test_live_data_ws_routes_to_binance_live_when_enabled(monkeypatch):
    captured = {}

    class _Committed:
        def __init__(self, *args, **kwargs):
            captured["handler"] = "committed"
            _ = (args, kwargs)

    class _BinanceLive:
        def __init__(self, events, symbol_list, config, exchange, *, transport):
            captured["handler"] = "binance_live"
            captured["transport"] = transport
            _ = (events, symbol_list, config, exchange)

    monkeypatch.setattr(data_ws, "CommittedWindowDataHandler", _Committed)
    monkeypatch.setattr(data_ws, "BinanceLiveDataHandler", _BinanceLive)

    cfg = SimpleNamespace(MARKET_DATA_SOURCE="binance_live")
    _ = data_ws.BinanceWebSocketDataHandler(SimpleNamespace(), ["BTC/USDT"], cfg, SimpleNamespace())
    assert captured["handler"] == "binance_live"
    assert captured["transport"] == "ws"
