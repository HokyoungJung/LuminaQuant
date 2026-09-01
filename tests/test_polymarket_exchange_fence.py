"""Tests for the PolymarketExchange real-money execution fence (supports_real_money=False)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lumina_quant.exchanges.polymarket_exchange import PolymarketExchange


# ---------------------------------------------------------------------------
# Minimal fake SDK (copied from test_polymarket_exchange.py pattern)
# ---------------------------------------------------------------------------


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

    def create_or_derive_api_creds(self):
        return {"api_key": "k", "api_secret": "s", "api_passphrase": "p"}

    def set_api_creds(self, creds):
        self.creds = creds

    def create_order(self, args):
        return {"kind": "limit", "args": args.kwargs}

    def create_market_order(self, args):
        return {"kind": "market", "args": args.kwargs}

    def post_order(self, order, order_type):
        return {
            "orderID": "ord-1",
            "status": "open",
            "size": order["args"].get("size", order["args"].get("amount")),
            "price": order["args"].get("price", 0.55),
            "side": order["args"]["side"],
        }


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


# ---------------------------------------------------------------------------
# Class-level capability flag
# ---------------------------------------------------------------------------


def test_polymarket_supports_real_money_is_false():
    assert PolymarketExchange.supports_real_money is False


# ---------------------------------------------------------------------------
# Fence: real-money path raises regardless of private key presence
# ---------------------------------------------------------------------------


def test_polymarket_fence_raises_with_private_key(monkeypatch):
    """Real-money execution must raise even when private key is configured."""
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "priv-key-value")
    exchange = PolymarketExchange(_config(mode="real", allow_real=True))
    with pytest.raises(RuntimeError, match="not approved for real-money execution"):
        exchange.execute_order("asset-1", "limit", "buy", 5.0, price=0.55)


def test_polymarket_fence_raises_for_market_order(monkeypatch):
    """Market orders are also blocked by the fence."""
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "priv-key-value")
    exchange = PolymarketExchange(_config(mode="real", allow_real=True))
    with pytest.raises(RuntimeError, match="not approved for real-money execution"):
        exchange.execute_order("asset-1", "market", "sell", 3.0, price=None)


# ---------------------------------------------------------------------------
# Paper path: must be completely unaffected by the fence
# ---------------------------------------------------------------------------


def test_polymarket_paper_execute_order_unaffected(monkeypatch):
    """Paper mode must return a stub without raising."""
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    exchange = PolymarketExchange(_config(mode="paper", allow_real=False))
    result = exchange.execute_order("asset-1", "limit", "buy", 5.0, price=0.55)
    assert result["status"] == "paper"
    assert result["symbol"] == "asset-1"


def test_polymarket_paper_with_allow_real_false_unaffected(monkeypatch):
    """allow_real=False must always fall through to paper regardless of MODE."""
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    exchange = PolymarketExchange(_config(mode="real", allow_real=False))
    result = exchange.execute_order("asset-1", "market", "buy", 2.0)
    assert result["status"] == "paper"


def test_polymarket_paper_default_config_unaffected(monkeypatch):
    """Default config (no MODE, no allow_real) must not raise."""
    monkeypatch.setattr("lumina_quant.exchanges.polymarket_exchange._load_sdk_symbols", _sdk)
    exchange = PolymarketExchange(_config())
    result = exchange.execute_order("asset-99", "limit", "sell", 1.0, price=0.4)
    assert result["status"] == "paper"
