from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.live.trader import LiveTrader
from lumina_quant.symbol_universe import resolve_available_symbols


def test_symbol_universe_validation():
    requested = [
        "BTC/USDT",
        "ETH/USDT",
        "XAU/USDT",
        "UNKNOWN/USDT",
        "btcusdt",
    ]
    markets = {
        "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "id": "BTCUSDT"},
        "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT", "id": "ETHUSDT"},
        "XAU/USDT:USDT": {"symbol": "XAU/USDT:USDT", "id": "XAUUSDT"},
    }

    kept, dropped = resolve_available_symbols(requested, markets)

    assert kept == ["BTC/USDT", "ETH/USDT", "XAU/USDT"]
    assert dropped == ["UNKNOWN/USDT"]


def test_live_trader_drops_unknown_symbols_gracefully():
    warnings: list[str] = []
    notifications: list[str] = []

    class _Exchange:
        def load_markets(self):
            return {"BTC/USDT:USDT": {"id": "BTCUSDT"}}

    trader = LiveTrader.__new__(LiveTrader)
    trader._audit_closed = True
    trader.config = SimpleNamespace(EXCHANGE={"driver": "binance_futures"})
    trader.exchange = _Exchange()
    trader.logger = SimpleNamespace(warning=lambda message, *args: warnings.append(str(message)))
    trader.notifier = SimpleNamespace(
        send_message=lambda message: notifications.append(str(message))
    )

    resolved = trader._filter_unavailable_symbols(["BTC/USDT", "UNKNOWN/USDT"])

    assert resolved == ["BTC/USDT"]
    assert any("Dropping unavailable symbols" in item for item in warnings)
    assert any("Dropping unavailable symbols" in item for item in notifications)
