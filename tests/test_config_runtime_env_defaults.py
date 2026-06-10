"""Tests for the new typed configuration loading API.

Replaces the old test suite that validated config.py / runtime_access.py behavior.
The new invariants:
  - load_runtime_config(path) → RuntimeConfig with correct typed fields
  - get_default_runtime_config() respects LQ_CONFIG_PATH env var
  - current_market_data_runtime_settings() reads fresh config on every call
  - No os.environ hidden-bus: env vars are NOT silently seeded from config values
"""

from __future__ import annotations

import textwrap

from lumina_quant.configuration import (
    current_market_data_runtime_settings,
    get_default_runtime_config,
    load_runtime_config,
)


def _write_config(tmp_path) -> str:
    cfg = textwrap.dedent(
        """
        trading:
          symbols: ["BTC/USDT"]
          timeframe: "5m"
        storage:
          backend: "local"
          market_data_parquet_path: "var/data/custom_parquet"
          collector_periodic_enabled: false
          materializer_required_timeframes: ["1s", "5m"]
        execution:
          gpu_mode: "auto"
          gpu_vram_gb: 4.5
        backtest:
          chunk_days: 9
        market_window:
          parity_v2_enabled: true
        live:
          mode: "paper"
          market_data_source: "external"
          exchange:
            driver: "binance_futures"
            name: "binance"
            market_type: "future"
            position_mode: "HEDGE"
            margin_mode: "isolated"
            leverage: 2
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return str(cfg_path)


def _write_market_data_config(tmp_path, *, symbols, root_path: str, exchange: str) -> str:
    import json

    cfg = textwrap.dedent(
        f"""
        trading:
          symbols: {json.dumps(list(symbols))}
          timeframe: "15m"
        storage:
          backend: "local"
          market_data_parquet_path: "{root_path}"
          market_data_exchange: "{exchange}"
        live:
          mode: "paper"
          exchange:
            driver: "binance_futures"
            name: "binance"
            market_type: "future"
            position_mode: "HEDGE"
            margin_mode: "isolated"
            leverage: 2
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return str(cfg_path)


def test_load_runtime_config_parses_all_sections(tmp_path):
    cfg_path = _write_config(tmp_path)
    rt = load_runtime_config(cfg_path)

    assert rt.trading.timeframe == "5m"
    assert rt.trading.symbols == ["BTC/USDT"]
    assert rt.storage.backend == "local"
    assert rt.storage.market_data_parquet_path == "var/data/custom_parquet"
    assert rt.storage.collector_periodic_enabled is False
    assert rt.backtest.chunk_days == 9
    assert rt.market_window.parity_v2_enabled is True
    assert rt.live.mode == "paper"
    assert rt.live.market_data_source == "external"
    assert rt.live.exchange.name == "binance"
    assert rt.live.exchange.leverage == 2


def test_get_default_runtime_config_uses_LQ_CONFIG_PATH(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    rt = get_default_runtime_config()

    assert rt.backtest.chunk_days == 9
    assert rt.live.market_data_source == "external"
    assert rt.market_window.parity_v2_enabled is True
    assert rt.storage.collector_periodic_enabled is False
    assert rt.trading.timeframe == "5m"
    assert rt.live.exchange.leverage == 2


def test_current_market_data_runtime_settings_reads_fresh_config_per_call(tmp_path, monkeypatch):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_cfg = _write_market_data_config(
        first_dir,
        symbols=["BTC/USDT"],
        root_path="var/data/first_parquet",
        exchange="binance",
    )
    second_cfg = _write_market_data_config(
        second_dir,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/second_parquet",
        exchange="kraken",
    )

    monkeypatch.setenv("LQ_CONFIG_PATH", first_cfg)
    first_settings = current_market_data_runtime_settings()
    assert first_settings == {
        "symbols": ["BTC/USDT"],
        "market_data_parquet_path": "var/data/first_parquet",
        "market_data_exchange": "binance",
    }

    # Switching LQ_CONFIG_PATH yields fresh values with no stale state.
    monkeypatch.setenv("LQ_CONFIG_PATH", second_cfg)
    second_settings = current_market_data_runtime_settings()
    assert second_settings == {
        "symbols": ["ETH/USDT", "SOL/USDT"],
        "market_data_parquet_path": "var/data/second_parquet",
        "market_data_exchange": "kraken",
    }


def test_load_runtime_config_returns_RuntimeConfig_with_new_knobs(tmp_path):
    cfg_path = _write_config(tmp_path)
    rt = load_runtime_config(cfg_path)

    # Phase 1 knobs
    assert hasattr(rt, "memory")
    assert hasattr(rt.memory, "cap_gb")
    assert rt.memory.cap_gb == 8.0  # default

    assert hasattr(rt, "validation")
    assert hasattr(rt.validation, "golden_rtol")
    assert rt.validation.golden_rtol == 1e-8  # default


def test_get_default_runtime_config_missing_file_returns_default(tmp_path, monkeypatch):
    """get_default_runtime_config() must not raise on a missing file — returns defaults."""
    from lumina_quant.configuration import get_default_runtime_config
    from lumina_quant.configuration.schema import RuntimeConfig

    monkeypatch.setenv("LQ_CONFIG_PATH", str(tmp_path / "nonexistent.yaml"))
    rt = get_default_runtime_config()
    # Must be a valid RuntimeConfig with defaults
    assert isinstance(rt, RuntimeConfig)
    assert rt.trading.timeframe == "1m"  # schema default
    assert rt.backtest.chunk_days == 2  # schema default
