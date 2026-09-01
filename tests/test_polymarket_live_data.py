from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.live.data_polymarket_live import PolymarketLiveDataHandler


class _ThreadStub:
    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon

    def start(self):
        return None

    def is_alive(self):
        return False

    def join(self, timeout=None):
        _ = timeout


def _handler(monkeypatch):
    monkeypatch.setattr("lumina_quant.live.data_polymarket_live.threading.Thread", _ThreadStub)
    cfg = SimpleNamespace(
        POLYMARKET_ASSET_IDS=["asset-1"],
        LIVE_POLL_SECONDS=1,
        INGEST_WINDOW_SECONDS=5,
        POLYMARKET_MARKET_WS_URL="wss://example.invalid/ws/market",
    )
    return PolymarketLiveDataHandler([], ["asset-1"], cfg, exchange=None)


def test_polymarket_handler_uses_explicit_market_subscription_contract(monkeypatch):
    handler = _handler(monkeypatch)
    messages = handler._subscribe_messages()
    assert messages == [{"type": "market", "assets_ids": ["asset-1"]}]


def test_polymarket_handler_normalizes_last_trade_price_only(monkeypatch):
    handler = _handler(monkeypatch)
    tick = handler._normalize_message(
        {
            "event_type": "last_trade_price",
            "asset_id": "asset-1",
            "price": "0.62",
            "size": "12",
            "timestamp": 1_700_000_000_000,
        }
    )
    assert tick is not None
    assert tick.symbol == "asset-1"
    assert tick.price == 0.62
    assert tick.quantity == 12.0

    ignored = handler._normalize_message(
        {
            "event_type": "book",
            "asset_id": "asset-1",
            "price": "0.62",
            "size": "12",
            "timestamp": 1_700_000_000_000,
        }
    )
    assert ignored is None
