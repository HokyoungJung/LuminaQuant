from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.exchanges import get_exchange
from lumina_quant.exchanges.binance_futures_exchange import BinanceFuturesExchange


class _Config(SimpleNamespace):
    BINANCE_API_KEY = ""
    BINANCE_SECRET_KEY = ""
    IS_TESTNET = True
    SYMBOLS = []
    POSITION_MODE = "HEDGE"
    MARGIN_MODE = "isolated"
    LEVERAGE = 1


def test_get_exchange_routes_binance_futures_driver_to_native_exchange(monkeypatch) -> None:
    monkeypatch.setattr(BinanceFuturesExchange, "load_markets", lambda self: {})
    monkeypatch.setattr(BinanceFuturesExchange, "set_position_mode", lambda self, _mode: True)
    monkeypatch.setattr(
        BinanceFuturesExchange, "set_margin_mode", lambda self, _symbol, _mode: True
    )
    monkeypatch.setattr(BinanceFuturesExchange, "set_leverage", lambda self, _symbol, _lev: True)

    cfg = _Config(
        EXCHANGE={"driver": "binance_futures", "name": "binance", "market_type": "future"},
    )

    exchange = get_exchange(cfg)

    assert isinstance(exchange, BinanceFuturesExchange)
