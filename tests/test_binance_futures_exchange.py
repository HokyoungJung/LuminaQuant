from __future__ import annotations

from lumina_quant.exchanges.binance_futures_client import (
    BinanceFuturesAPIError,
    BinanceFuturesClientConfig,
    BinanceFuturesRESTClient,
)
from lumina_quant.exchanges.binance_futures_exchange import BinanceFuturesExchange


class MockConfig:
    EXCHANGE = {
        "driver": "binance_futures",
        "name": "binance",
        "market_type": "future",
    }
    BINANCE_API_KEY = "test_key"
    BINANCE_SECRET_KEY = "test_secret"
    IS_TESTNET = False
    POSITION_MODE = "HEDGE"
    MARGIN_MODE = "isolated"
    LEVERAGE = 3
    SYMBOLS = ["BTC/USDT"]


def _stub_exchange_bootstrap(monkeypatch) -> None:
    monkeypatch.setattr(
        BinanceFuturesRESTClient,
        "exchange_info",
        lambda self: {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ],
                }
            ]
        },
    )
    monkeypatch.setattr(BinanceFuturesRESTClient, "change_position_mode", lambda self, **kwargs: {}, raising=True)
    monkeypatch.setattr(BinanceFuturesRESTClient, "change_margin_type", lambda self, **kwargs: {}, raising=True)
    monkeypatch.setattr(BinanceFuturesRESTClient, "change_initial_leverage", lambda self, **kwargs: {}, raising=True)


def test_exchange_uses_native_rest_client(monkeypatch) -> None:
    _stub_exchange_bootstrap(monkeypatch)
    exchange = BinanceFuturesExchange(MockConfig())
    assert isinstance(exchange.exchange, BinanceFuturesRESTClient)
    assert isinstance(exchange.exchange.config, BinanceFuturesClientConfig)
    assert exchange.exchange.config.api_key == "test_key"
    assert exchange.exchange.config.secret_key == "test_secret"


def test_exchange_normalizes_aggtrade_rows(monkeypatch) -> None:
    _stub_exchange_bootstrap(monkeypatch)
    exchange = BinanceFuturesExchange(MockConfig())
    monkeypatch.setattr(
        exchange._client(),
        "agg_trades",
        lambda **_kwargs: [
            {"a": 11, "T": 1_700_000_000_000, "p": "100.5", "q": "0.25", "m": False}
        ],
    )
    rows = exchange.fetch_trades("BTC/USDT", since=1_700_000_000_000, limit=1000)
    assert rows == [
        {
            "id": 11,
            "symbol": "BTC/USDT",
            "timestamp": 1_700_000_000_000,
            "price": 100.5,
            "amount": 0.25,
            "side": "buy",
            "isBuyerMaker": False,
            "info": {"a": 11, "T": 1_700_000_000_000, "p": "100.5", "q": "0.25", "m": False},
        }
    ]


def test_exchange_normalizes_open_orders(monkeypatch) -> None:
    _stub_exchange_bootstrap(monkeypatch)
    exchange = BinanceFuturesExchange(MockConfig())
    monkeypatch.setattr(
        exchange._client(),
        "query_open_orders",
        lambda **_kwargs: [
            {
                "orderId": 123,
                "status": "NEW",
                "executedQty": "0.1",
                "origQty": "1.0",
                "avgPrice": "0",
                "price": "100",
                "time": 1_700_000_000_000,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "clientOrderId": "LQ-1",
                "positionSide": "LONG",
                "reduceOnly": "false",
                "timeInForce": "GTC",
            }
        ],
    )
    rows = exchange.fetch_open_orders("BTC/USDT")
    assert rows[0]["id"] == "123"
    assert rows[0]["symbol"] == "BTC/USDT"
    assert rows[0]["type"] == "limit"
    assert rows[0]["side"] == "buy"
    assert rows[0]["amount"] == 1.0
    assert rows[0]["filled"] == 0.1
    assert rows[0]["reduceOnly"] is False


def test_exchange_submits_conditional_algo_order(monkeypatch) -> None:
    _stub_exchange_bootstrap(monkeypatch)
    exchange = BinanceFuturesExchange(MockConfig())
    captured = {}

    def _new_algo_order(**kwargs):
        captured.update(kwargs)
        return {
            "algoId": 456,
            "algoStatus": "NEW",
            "quantity": "1.5",
            "price": "0",
            "symbol": "ETHUSDT",
            "side": "SELL",
            "orderType": "STOP_MARKET",
            "clientAlgoId": "LQ-P-SL-test",
            "positionSide": "LONG",
            "reduceOnly": "false",
        }

    monkeypatch.setattr(exchange._client(), "new_algo_order", _new_algo_order)

    order = exchange.execute_algo_order(
        symbol="ETH/USDT",
        type="STOP_MARKET",
        side="sell",
        quantity=1.5,
        params={
            "clientAlgoId": "LQ-P-SL-test",
            "positionSide": "LONG",
            "triggerPrice": 95.0,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": "true",
            "parentOrderId": "entry-filled",
            "parentClientOrderId": "LQ-entry",
            "protectionRole": "stop_loss",
        },
    )

    assert captured["algoType"] == "CONDITIONAL"
    assert captured["symbol"] == "ETHUSDT"
    assert captured["type"] == "STOP_MARKET"
    assert captured["triggerPrice"] == 95.0
    assert "parentOrderId" not in captured
    assert "parentClientOrderId" not in captured
    assert "protectionRole" not in captured
    assert order["id"] == "456"
    assert order["status"] == "open"
    assert order["symbol"] == "ETH/USDT"
    assert order["type"] == "stop_market"


def test_exchange_exposes_side_aware_position_legs(monkeypatch) -> None:
    _stub_exchange_bootstrap(monkeypatch)
    exchange = BinanceFuturesExchange(MockConfig())
    monkeypatch.setattr(
        exchange,
        "fetch_positions",
        lambda symbol=None: [
            {
                "symbol": "BTC/USDT",
                "positionAmt": 1.2,
                "positionSide": "LONG",
            },
            {
                "symbol": "BTC/USDT",
                "positionAmt": 0.4,
                "positionSide": "SHORT",
            },
            {
                "symbol": "ETH/USDT",
                "positionAmt": 0.0,
                "positionSide": "LONG",
            },
        ],
    )

    legs = exchange.get_all_position_legs()

    assert legs == {"BTC/USDT": {"LONG": 1.2, "SHORT": 0.4}}


def test_exchange_bootstrap_does_not_silently_swallow_setup_failures(monkeypatch) -> None:
    _stub_exchange_bootstrap(monkeypatch)

    def _raise_for_leverage(self, **_kwargs):
        raise BinanceFuturesAPIError("leverage mismatch", error_code=-4019)

    monkeypatch.setattr(
        BinanceFuturesRESTClient,
        "change_initial_leverage",
        _raise_for_leverage,
        raising=True,
    )

    try:
        BinanceFuturesExchange(MockConfig())
    except BinanceFuturesAPIError as exc:
        assert exc.error_code == -4019
        assert "leverage mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected BinanceFuturesAPIError to propagate during bootstrap")


def test_exchange_bootstrap_surfaces_credential_guidance_for_invalid_testnet_keys(monkeypatch) -> None:
    _stub_exchange_bootstrap(monkeypatch)

    def _raise_invalid_key(self, **_kwargs):
        raise BinanceFuturesAPIError("Invalid API-key, IP, or permissions for action", error_code=-2015)

    monkeypatch.setattr(
        BinanceFuturesRESTClient,
        "change_position_mode",
        _raise_invalid_key,
        raising=True,
    )

    class _TestnetConfig(MockConfig):
        IS_TESTNET = True

    try:
        BinanceFuturesExchange(_TestnetConfig())
    except BinanceFuturesAPIError as exc:
        message = str(exc)
        assert exc.error_code == -2015
        assert "paper/testnet credentials were rejected" in message
        assert "BINANCE_API_KEY/BINANCE_SECRET_KEY" in message
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected BinanceFuturesAPIError to propagate during bootstrap")
