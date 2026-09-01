from __future__ import annotations

from types import SimpleNamespace

import pytest

from lumina_quant.exchanges import get_exchange
from lumina_quant.exchanges.polymarket_exchange import PolymarketExchange


class _FakeOrderType:
    GTC = "GTC"
    FOK = "FOK"


class _FakeOpenOrderParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeApiCreds:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeOrderArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeMarketOrderArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.creds = None
        self.last_post = None
        self.last_cancel = None
        self.last_get_orders = None
        self.last_get_order = None

    def create_or_derive_api_creds(self):
        return {"api_key": "k", "api_secret": "s", "api_passphrase": "p"}

    def set_api_creds(self, creds):
        self.creds = creds

    def create_order(self, args):
        return {"kind": "limit", "args": args.kwargs}

    def create_market_order(self, args):
        return {"kind": "market", "args": args.kwargs}

    def post_order(self, order, order_type):
        self.last_post = (order, order_type)
        return {
            "orderID": "ord-1",
            "status": "open",
            "size": order["args"].get("size", order["args"].get("amount")),
            "price": order["args"].get("price", 0.55),
            "side": order["args"]["side"],
        }

    def get_orders(self, params):
        self.last_get_orders = params.kwargs
        return [{"id": "open-1", "status": "open", "size": 5, "price": 0.5}]

    def get_order(self, order_id):
        self.last_get_order = order_id
        return {"id": order_id, "status": "filled", "size": 5, "price": 0.6, "filled": 5}

    def cancel(self, order_id):
        self.last_cancel = order_id
        return {"success": True}


def _sdk():
    return {
        "ClobClient": _FakeClient,
        "ApiCreds": _FakeApiCreds,
        "OpenOrderParams": _FakeOpenOrderParams,
        "BUY": "BUY",
        "SELL": "SELL",
        "OrderArgs": _FakeOrderArgs,
        "MarketOrderArgs": _FakeMarketOrderArgs,
        "OrderType": _FakeOrderType,
    }


def _config(mode="paper", allow_real=False):
    return SimpleNamespace(
        EXCHANGE={"driver": "polymarket", "name": "polymarket"},
        MODE=mode,
        POLYMARKET_ALLOW_REAL_EXECUTION=allow_real,
        POLYMARKET_PRIVATE_KEY_ENV="POLYMARKET_PRIVATE_KEY",
        POLYMARKET_API_KEY_ENV="POLYMARKET_API_KEY",
        POLYMARKET_API_SECRET_ENV="POLYMARKET_API_SECRET",
        POLYMARKET_API_PASSPHRASE_ENV="POLYMARKET_API_PASSPHRASE",
        POLYMARKET_HOST="https://clob.polymarket.com",
        POLYMARKET_DATA_HOST="https://data-api.polymarket.com",
        POLYMARKET_CHAIN_ID=137,
        POLYMARKET_FUNDER="0xabc",
        POLYMARKET_SIGNATURE_TYPE=0,
    )


def test_get_exchange_supports_polymarket_driver(monkeypatch):
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    exchange = get_exchange(_config())
    assert isinstance(exchange, PolymarketExchange)


def test_polymarket_exchange_paper_execute_order_returns_stub(monkeypatch):
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    exchange = PolymarketExchange(_config())
    result = exchange.execute_order("asset-1", "limit", "buy", 5.0, price=0.55)
    assert result["status"] == "paper"


def test_polymarket_exchange_real_limit_order_is_fenced(monkeypatch):
    # Real-money execution is fenced: adapter raises even with credentials present.
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "priv")
    exchange = PolymarketExchange(_config(mode="real", allow_real=True))
    with pytest.raises(RuntimeError, match="not approved for real-money execution"):
        exchange.execute_order("asset-1", "limit", "buy", 5.0, price=0.55)


def test_polymarket_exchange_real_market_order_is_fenced(monkeypatch):
    # Real-money execution is fenced: adapter raises even with credentials present.
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "priv")
    exchange = PolymarketExchange(_config(mode="real", allow_real=True))
    with pytest.raises(RuntimeError, match="not approved for real-money execution"):
        exchange.execute_order("asset-1", "market", "sell", 3.0, price=None)


def test_polymarket_exchange_fetch_open_orders_and_cancel(monkeypatch):
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    exchange = PolymarketExchange(_config())
    rows = exchange.fetch_open_orders("asset-1")
    assert rows[0]["id"] == "open-1"
    assert exchange.client.last_get_orders == {"asset_id": "asset-1"}
    assert exchange.cancel_order("open-1") is True
    assert exchange.client.last_cancel == "open-1"


def test_polymarket_exchange_fetch_order(monkeypatch):
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    exchange = PolymarketExchange(_config())
    row = exchange.fetch_order("ord-9", "asset-1")
    assert row["id"] == "ord-9"
    assert row["status"] == "filled"


def test_polymarket_exchange_real_execution_requires_private_key(monkeypatch):
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    exchange = PolymarketExchange(_config(mode="real", allow_real=True))
    with pytest.raises(RuntimeError):
        exchange.execute_order("asset-1", "limit", "buy", 5.0, price=0.55)
