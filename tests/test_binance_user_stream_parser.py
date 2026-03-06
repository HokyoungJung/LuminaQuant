from __future__ import annotations

from lumina_quant.live.binance_user_stream import BinanceUserStreamClient


def test_parse_order_trade_update_normalizes_to_execution_report():
    payload = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1_700_000_000_500,
        "o": {
            "s": "BTCUSDT",
            "i": 12345,
            "c": "LQ-abc",
            "x": "TRADE",
            "X": "PARTIALLY_FILLED",
            "l": "0.01",
            "z": "0.03",
            "L": "65000.5",
            "t": 99,
            "S": "BUY",
            "ps": "LONG",
            "R": False,
        },
    }

    parsed = BinanceUserStreamClient.parse_message(payload)
    assert parsed is not None
    assert parsed["event_type"] == "executionReport"
    assert parsed["symbol"] == "BTCUSDT"
    assert parsed["order_id"] == "12345"
    assert parsed["cum_fill_qty"] == 0.03
    assert parsed["last_fill_price"] == 65000.5
    assert parsed["position_side"] == "LONG"


def test_parse_account_update_extracts_balances_and_positions():
    payload = {
        "e": "ACCOUNT_UPDATE",
        "E": 1_700_000_000_700,
        "a": {
            "m": "ORDER",
            "B": [{"a": "USDT", "wb": "1000.0", "cw": "990.0"}],
            "P": [{"s": "BTCUSDT", "pa": "0.01", "ps": "LONG"}],
        },
    }

    parsed = BinanceUserStreamClient.parse_message(payload)
    assert parsed is not None
    assert parsed["event_type"] == "accountUpdate"
    assert parsed["reason"] == "ORDER"
    assert len(parsed["balances"]) == 1
    assert len(parsed["positions"]) == 1
    assert parsed["positions"][0]["s"] == "BTCUSDT"
