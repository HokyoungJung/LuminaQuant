from __future__ import annotations

import pytest
from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.configuration.validate import validate_runtime_config


def _base_raw() -> dict:
    return {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "live": {
            "mode": "paper",
            "market_data_source": "committed",
            "order_state_source": "polling",
            "exchange": {
                "driver": "ccxt",
                "name": "binance",
                "market_type": "future",
                "position_mode": "HEDGE",
                "margin_mode": "isolated",
                "leverage": 2,
            },
        },
    }


def test_validate_rejects_user_stream_without_binance_live_source():
    raw = _base_raw()
    raw["live"]["order_state_source"] = "user_stream"
    raw["live"]["market_data_source"] = "committed"
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError):
        validate_runtime_config(runtime)


def test_validate_rejects_binance_live_on_non_binance_exchange():
    raw = _base_raw()
    raw["live"]["market_data_source"] = "binance_live"
    raw["live"]["exchange"]["name"] = "kraken"
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError):
        validate_runtime_config(runtime)


def test_validate_accepts_binance_live_with_user_stream_on_ccxt_binance():
    raw = _base_raw()
    raw["live"]["market_data_source"] = "binance_live"
    raw["live"]["order_state_source"] = "user_stream"
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)
