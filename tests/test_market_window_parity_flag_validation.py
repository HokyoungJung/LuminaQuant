from __future__ import annotations

import os
import tempfile

import pytest
import yaml
from lumina_quant.configuration.loader import load_runtime_config
from lumina_quant.configuration.validate import validate_runtime_config


def _base_payload() -> dict[str, object]:
    return {
        "trading": {
            "symbols": ["BTC/USDT"],
            "timeframe": "1m",
            "timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        },
        "live": {
            "mode": "paper",
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


def _load_runtime(payload: dict[str, object]):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LQ__")}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        path = handle.name
    try:
        return load_runtime_config(config_path=path, env=env)
    finally:
        os.remove(path)


def test_market_window_parity_deprecated_flags_must_match():
    payload = _base_payload()
    payload["live"]["market_window_parity_v2_enabled"] = True
    payload["backtest"] = {"market_window_parity_v2_enabled": False}

    runtime = _load_runtime(payload)
    with pytest.raises(ValueError) as exc:
        validate_runtime_config(runtime)
    assert (
        str(exc.value)
        == "live.market_window_parity_v2_enabled and backtest.market_window_parity_v2_enabled must match; use market_window.parity_v2_enabled."
    )


def test_market_window_shared_flag_inherits_matching_deprecated_flags():
    payload = _base_payload()
    payload["live"]["market_window_parity_v2_enabled"] = True
    payload["backtest"] = {"market_window_parity_v2_enabled": True}

    runtime = _load_runtime(payload)
    validate_runtime_config(runtime)
    assert runtime.market_window.parity_v2_enabled is True
