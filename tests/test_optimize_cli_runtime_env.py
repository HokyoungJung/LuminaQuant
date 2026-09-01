"""Tests for cli.optimize runtime settings loading via typed config."""

from __future__ import annotations

import textwrap

from lumina_quant.cli import optimize


def test_optimize_runtime_settings_reads_env_and_config_at_call_time(tmp_path, monkeypatch):
    """_current_optimize_runtime_settings() reads env vars and typed config at call time.

    Replaces the old BaseConfig/OptimizationConfig monkeypatch approach with
    LQ_CONFIG_PATH + YAML, which is the Phase 1 config contract.
    """
    cfg = textwrap.dedent(
        """
        trading:
          symbols: ["BTC/USDT", "ETH/USDT"]
          timeframe: "15m"
        storage:
          market_data_parquet_path: var/data/runtime_parquet
          market_data_exchange: kraken
          backend: parquet
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")

    monkeypatch.setenv("LQ_CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("LQ_DATA_MODE", "legacy")
    monkeypatch.setenv("LQ_BASE_TIMEFRAME", "5m")
    monkeypatch.setenv("LQ_AUTO_COLLECT_DB", "1")
    monkeypatch.setenv("LQ_BACKTEST_MODE", "legacy_batch")

    settings = optimize._current_optimize_runtime_settings()

    assert settings.data_mode == "legacy"
    assert settings.base_timeframe == "5m"
    assert settings.auto_collect_db is True
    assert settings.backtest_mode == "legacy_batch"
    assert settings.market_db_path == "var/data/runtime_parquet"
    assert settings.market_db_exchange == "kraken"
    assert settings.market_db_backend == "parquet"
    assert settings.symbol_list == ["BTC/USDT", "ETH/USDT"]
    assert settings.strategy_timeframe == "15m"
    # max_workers is capped to min(2, max(1, opt.max_workers)); default opt.max_workers=1
    assert settings.max_workers == 1
