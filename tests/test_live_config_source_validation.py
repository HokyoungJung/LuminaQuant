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
                "driver": "binance_futures",
                "name": "binance",
                "market_type": "future",
                "position_mode": "HEDGE",
                "margin_mode": "isolated",
                "leverage": 2,
            },
        },
    }


def test_validate_rejects_user_stream_without_binance_futures_source():
    raw = _base_raw()
    raw["live"]["order_state_source"] = "user_stream"
    raw["live"]["market_data_source"] = "committed"
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError):
        validate_runtime_config(runtime)


def test_validate_rejects_binance_futures_on_non_binance_exchange():
    raw = _base_raw()
    raw["live"]["market_data_source"] = "binance_futures"
    raw["live"]["exchange"]["name"] = "kraken"
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError):
        validate_runtime_config(runtime)


def test_blank_driver_stays_invalid_on_non_binance_exchange():
    raw = _base_raw()
    raw["live"]["exchange"]["driver"] = ""
    raw["live"]["exchange"]["name"] = "kraken"
    runtime = build_runtime_config(raw, env={})
    assert runtime.live.exchange.driver == ""
    with pytest.raises(ValueError):
        validate_runtime_config(runtime)


def test_blank_driver_normalizes_to_binance_futures_for_binance_exchange():
    raw = _base_raw()
    raw["live"]["exchange"]["driver"] = ""
    runtime = build_runtime_config(raw, env={})
    assert runtime.live.exchange.driver == "binance_futures"
    validate_runtime_config(runtime)


def test_validate_accepts_binance_futures_with_user_stream_on_binance_futures():
    raw = _base_raw()
    raw["live"]["market_data_source"] = "binance_futures"
    raw["live"]["order_state_source"] = "user_stream"
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


def test_validate_accepts_research_promoted_7x_live_leverage():
    raw = _base_raw()
    raw["live"]["exchange"]["leverage"] = 7
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


def test_validate_rejects_live_leverage_above_research_promoted_cap():
    raw = _base_raw()
    raw["live"]["exchange"]["leverage"] = 21
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError, match=r"\[1, 20\]"):
        validate_runtime_config(runtime)


def test_validate_accepts_external_source_with_path():
    raw = _base_raw()
    raw["live"]["market_data_source"] = "external"
    raw["live"]["external"] = {
        "source_kind": "jsonl",
        "path": "var/data/external/live_windows.jsonl",
        "schema": "market_window_v1",
    }
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


def test_validate_rejects_external_source_without_path():
    raw = _base_raw()
    raw["live"]["market_data_source"] = "external"
    raw["live"]["external"] = {"source_kind": "jsonl", "path": "", "schema": "market_window_v1"}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError):
        validate_runtime_config(runtime)


def test_validate_accepts_polymarket_live_with_asset_ids():
    raw = _base_raw()
    raw["live"]["market_data_source"] = "polymarket_live"
    raw["live"]["exchange"]["driver"] = "polymarket"
    raw["live"]["exchange"]["name"] = "polymarket"
    raw["live"]["polymarket"] = {"asset_ids": ["asset-1"]}
    runtime = build_runtime_config(raw, env={})
    validate_runtime_config(runtime)


def test_validate_rejects_polymarket_real_mode_in_phase1():
    raw = _base_raw()
    raw["live"]["mode"] = "real"
    raw["live"]["market_data_source"] = "polymarket_live"
    raw["live"]["exchange"]["driver"] = "polymarket"
    raw["live"]["exchange"]["name"] = "polymarket"
    raw["live"]["polymarket"] = {"asset_ids": ["asset-1"], "allow_real_execution": False}
    runtime = build_runtime_config(raw, env={})
    with pytest.raises(ValueError):
        validate_runtime_config(runtime)
