from __future__ import annotations

import os
import tempfile

import pytest
import yaml
from lumina_quant.backtesting._config_view import BacktestConfigView
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


def _load_runtime(payload: dict[str, object], env: dict[str, str] | None = None):
    effective_env = dict(env or os.environ)
    for key in list(effective_env):
        if key.startswith("LQ__"):
            effective_env.pop(key, None)
    if env is not None:
        effective_env.update(env)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        path = handle.name
    try:
        return load_runtime_config(config_path=path, env=effective_env)
    finally:
        os.remove(path)


def test_periodic_loop_config_defaults_and_env_override():
    env = {k: v for k, v in os.environ.items() if not k.startswith("LQ__")}
    env["LQ__STORAGE__COLLECTOR_POLL_SECONDS"] = "7"
    env["LQ__STORAGE__MATERIALIZER_REQUIRED_TIMEFRAMES"] = '["5M","1S","1h","5m"]'

    runtime = _load_runtime(_base_payload(), env=env)
    validate_runtime_config(runtime)

    assert runtime.storage.collector_periodic_enabled is True
    assert runtime.storage.collector_poll_seconds == 7
    assert runtime.storage.materializer_periodic_enabled is True
    assert runtime.storage.materializer_poll_seconds == 5
    assert runtime.storage.materializer_base_timeframe == "1s"
    assert runtime.storage.materializer_required_timeframes == ["1s", "5m", "1h"]


def test_backtest_config_view_exposes_storage_loop_controls():
    payload = _base_payload()
    payload["storage"] = {
        "wal_max_bytes": 123_456,
        "wal_compact_on_threshold": False,
        "wal_compaction_interval_seconds": 17,
        "collector_periodic_enabled": False,
        "collector_poll_seconds": 11,
        "collector_bootstrap_lookback_hours": 48,
        "materializer_periodic_enabled": False,
        "materializer_poll_seconds": 13,
        "materializer_base_timeframe": "1s",
        "materializer_required_timeframes": ["1s", "5m", "1h"],
    }

    view = BacktestConfigView(_load_runtime(payload))

    assert view.WAL_MAX_BYTES == 123_456
    assert view.WAL_COMPACT_ON_THRESHOLD is False
    assert view.WAL_COMPACTION_INTERVAL_SECONDS == 17
    assert view.COLLECTOR_PERIODIC_ENABLED is False
    assert view.COLLECTOR_POLL_SECONDS == 11
    assert view.COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS == 48
    assert view.MATERIALIZER_PERIODIC_ENABLED is False
    assert view.MATERIALIZER_POLL_SECONDS == 13
    assert view.MATERIALIZER_BASE_TIMEFRAME == "1s"
    assert view.MATERIALIZER_REQUIRED_TIMEFRAMES == ["1s", "5m", "1h"]


@pytest.mark.parametrize(
    ("storage_patch", "expected_message"),
    [
        ({"collector_poll_seconds": 0}, "storage.collector_poll_seconds must be >= 1."),
        ({"materializer_poll_seconds": 0}, "storage.materializer_poll_seconds must be >= 1."),
        (
            {"materializer_base_timeframe": "1m"},
            "storage.materializer_base_timeframe must be '1s'.",
        ),
        (
            {"materializer_required_timeframes": []},
            "storage.materializer_required_timeframes must be a non-empty list.",
        ),
        (
            {"materializer_required_timeframes": ["1m"]},
            "storage.materializer_required_timeframes must include '1s'.",
        ),
        (
            {"materializer_required_timeframes": ["1s", "abc"]},
            "storage.materializer_required_timeframes contains invalid timeframe 'abc'. "
            "Expected format like 1s, 5m, 1h, 1d.",
        ),
        (
            {"materializer_required_timeframes": ["1s", "20m"]},
            "storage.materializer_required_timeframes contains timeframe '20m' not present in trading.timeframes.",
        ),
    ],
)
def test_periodic_loop_config_validation_errors(storage_patch, expected_message):
    payload = _base_payload()
    payload["storage"] = storage_patch
    runtime = _load_runtime(payload)

    with pytest.raises(ValueError) as exc:
        validate_runtime_config(runtime)
    assert str(exc.value) == expected_message
