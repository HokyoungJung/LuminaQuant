from __future__ import annotations

import importlib
import json
import sys
import textwrap


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


def test_config_module_exposes_explicit_runtime_env_seeding(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)
    for key in (
        "LQ__STORAGE__COLLECTOR_PERIODIC_ENABLED",
        "LQ__STORAGE__MATERIALIZER_REQUIRED_TIMEFRAMES",
        "LQ__LIVE__MARKET_DATA_SOURCE",
        "LQ__BACKTEST__CHUNK_DAYS",
        "LQ__MARKET_WINDOW__PARITY_V2_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    import lumina_quant.config as config_module

    config_module = importlib.reload(config_module)

    assert "TIMEFRAME" not in config_module.BaseConfig.__dict__
    assert config_module.BaseConfig.COLLECTOR_PERIODIC_ENABLED is False
    assert config_module.LiveConfig.MARKET_DATA_SOURCE == "external"
    assert config_module.BacktestConfig.CHUNK_DAYS == 9
    assert config_module.BaseConfig.MARKET_WINDOW_PARITY_V2_ENABLED is True
    assert "LQ__STORAGE__COLLECTOR_PERIODIC_ENABLED" not in config_module.os.environ

    seeded = config_module.seed_runtime_env_defaults()

    assert seeded["LQ__STORAGE__COLLECTOR_PERIODIC_ENABLED"] == "0"
    assert config_module.os.environ["LQ__STORAGE__COLLECTOR_PERIODIC_ENABLED"] == "0"
    assert json.loads(
        config_module.os.environ["LQ__STORAGE__MATERIALIZER_REQUIRED_TIMEFRAMES"]
    ) == [
        "1s",
        "5m",
    ]
    assert config_module.os.environ["LQ__LIVE__MARKET_DATA_SOURCE"] == "external"
    assert config_module.os.environ["LQ__BACKTEST__CHUNK_DAYS"] == "9"
    assert config_module.os.environ["LQ__MARKET_WINDOW__PARITY_V2_ENABLED"] == "1"


def test_config_module_keeps_existing_env_overrides(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)
    monkeypatch.setenv("LQ__LIVE__MARKET_DATA_SOURCE", "committed")
    monkeypatch.setenv("LQ__BACKTEST__CHUNK_DAYS", "13")
    import lumina_quant.config as config_module

    config_module = importlib.reload(config_module)

    assert config_module.os.environ["LQ__LIVE__MARKET_DATA_SOURCE"] == "committed"
    assert config_module.os.environ["LQ__BACKTEST__CHUNK_DAYS"] == "13"
    assert config_module.LiveConfig.MARKET_DATA_SOURCE == "committed"
    assert config_module.BacktestConfig.CHUNK_DAYS == 13


def test_config_module_keeps_runtime_alias_fields_in_sync(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)
    monkeypatch.delenv("LQ_GPU_MODE", raising=False)
    monkeypatch.delenv("LQ__EXECUTION__GPU_MODE", raising=False)
    import lumina_quant.config as config_module

    config_module = importlib.reload(config_module)

    assert config_module.BaseConfig.TIMEFRAME == "5m"
    assert config_module.BaseConfig.TIMEFRAME in config_module.BaseConfig.TIMEFRAMES
    assert config_module.BaseConfig.GPU_MODE == "auto"
    assert config_module.BaseConfig.COMPUTE_BACKEND == "auto"
    assert (
        config_module.BacktestConfig.BACKTEST_POLL_SECONDS
        == config_module.BacktestConfig.POLL_SECONDS
    )
    assert config_module.BacktestConfig.MARGIN_MODE == "isolated"
    assert config_module.LiveConfig.EXCHANGE_ID == "binance"
    assert config_module.LiveConfig.LEVERAGE == 2


def test_current_runtime_settings_ignore_auto_seeded_defaults_from_previous_config(
    tmp_path, monkeypatch
):
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
    import lumina_quant.config as config_module

    config_module = importlib.reload(config_module)
    assert config_module.current_market_data_runtime_settings() == {
        "symbols": ["BTC/USDT"],
        "market_data_parquet_path": "var/data/first_parquet",
        "market_data_exchange": "binance",
    }

    monkeypatch.setenv("LQ_CONFIG_PATH", second_cfg)

    assert config_module.current_market_data_runtime_settings() == {
        "symbols": ["ETH/USDT", "SOL/USDT"],
        "market_data_parquet_path": "var/data/second_parquet",
        "market_data_exchange": "kraken",
    }


def test_config_module_reexports_runtime_access_module(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)
    sys.modules.pop("lumina_quant.configuration.runtime_access", None)
    import lumina_quant.config as config_module

    config_module = importlib.reload(config_module)

    assert config_module.load_config(cfg_path)["live"]["mode"] == "paper"
    assert "lumina_quant.configuration.runtime_access" in sys.modules
    assert config_module.BaseConfig.__module__ == "lumina_quant.configuration.runtime_access"
    assert config_module.BaseConfig.TIMEFRAME == "5m"


def test_config_module_import_clears_views_without_resetting_runtime_cache(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    runtime_access = importlib.import_module("lumina_quant.configuration.runtime_access")
    reset_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    clear_calls: list[str] = []

    original_clear = runtime_access.clear_runtime_config_views

    monkeypatch.setattr(
        runtime_access,
        "reset_runtime_config_cache",
        lambda *args, **kwargs: reset_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        runtime_access,
        "clear_runtime_config_views",
        lambda: clear_calls.append("clear") or original_clear(),
    )

    sys.modules.pop("lumina_quant.config", None)
    config_module = importlib.import_module("lumina_quant.config")

    assert clear_calls == ["clear"]
    assert reset_calls == []
    assert "TIMEFRAME" not in config_module.BaseConfig.__dict__
