"""Tests for driver-gated credential validation in validate_runtime_config(for_live=True)."""

from __future__ import annotations

import os

import pytest

from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.configuration.validate import validate_runtime_config


def _binance_raw() -> dict:
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


def _polymarket_raw() -> dict:
    return {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "live": {
            "mode": "paper",
            "market_data_source": "polymarket_live",
            "order_state_source": "polling",
            "exchange": {
                "driver": "polymarket",
                "name": "polymarket",
                "market_type": "spot",
                "position_mode": "ONEWAY",
                "margin_mode": "isolated",
                "leverage": 1,
            },
            "polymarket": {
                "asset_ids": ["asset-1"],
            },
        },
    }


def _mt5_raw() -> dict:
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
            "mt5_bridge_python": "/usr/bin/python3",
            "exchange": {
                "driver": "mt5",
                "name": "mt5",
                "market_type": "spot",
                "position_mode": "ONEWAY",
                "margin_mode": "isolated",
                "leverage": 1,
            },
        },
    }


# ---------------------------------------------------------------------------
# Binance driver: existing behaviour must not regress
# ---------------------------------------------------------------------------


def test_binance_for_live_without_keys_raises():
    """Binance driver for_live=True without API keys must still raise."""
    runtime = build_runtime_config(_binance_raw(), env={})
    with pytest.raises(ValueError, match="BINANCE_API_KEY"):
        validate_runtime_config(runtime, for_live=True)


def test_binance_for_live_with_keys_passes():
    """Binance driver for_live=True with both keys passes the credential gate."""
    runtime = build_runtime_config(
        _binance_raw(),
        env={"BINANCE_API_KEY": "key", "BINANCE_SECRET_KEY": "secret"},
    )
    validate_runtime_config(runtime, for_live=True)


# ---------------------------------------------------------------------------
# Polymarket driver: must NOT require Binance keys; must require own cred
# ---------------------------------------------------------------------------


def test_polymarket_for_live_without_binance_keys_but_with_polymarket_key_passes():
    """Polymarket driver must not require Binance keys; POLYMARKET_PRIVATE_KEY is sufficient."""
    runtime = build_runtime_config(_polymarket_raw(), env={})
    old = os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
    try:
        os.environ["POLYMARKET_PRIVATE_KEY"] = "0xdeadbeef"
        validate_runtime_config(runtime, for_live=True)
    finally:
        if old is None:
            os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
        else:
            os.environ["POLYMARKET_PRIVATE_KEY"] = old


def test_polymarket_for_live_without_polymarket_key_raises_polymarket_message():
    """Polymarket driver for_live=True without POLYMARKET_PRIVATE_KEY raises a Polymarket-specific message."""
    runtime = build_runtime_config(_polymarket_raw(), env={})
    old = os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
    try:
        with pytest.raises(ValueError, match="POLYMARKET_PRIVATE_KEY"):
            validate_runtime_config(runtime, for_live=True)
    finally:
        if old is None:
            os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
        else:
            os.environ["POLYMARKET_PRIVATE_KEY"] = old


def test_polymarket_for_live_without_polymarket_key_does_not_mention_binance():
    """Error message for missing Polymarket cred must not say 'BINANCE'."""
    runtime = build_runtime_config(_polymarket_raw(), env={})
    old = os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
    try:
        with pytest.raises(ValueError) as exc_info:
            validate_runtime_config(runtime, for_live=True)
        assert "BINANCE" not in str(exc_info.value)
    finally:
        if old is None:
            os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
        else:
            os.environ["POLYMARKET_PRIVATE_KEY"] = old


# ---------------------------------------------------------------------------
# MT5 driver: must NOT require Binance keys
# ---------------------------------------------------------------------------


def test_mt5_for_live_without_binance_keys_passes():
    """MT5 driver for_live=True must not require Binance API keys."""
    runtime = build_runtime_config(_mt5_raw(), env={})
    # MT5 does not require Binance keys; no platform check on Linux with bridge_python set.
    validate_runtime_config(runtime, for_live=True)
