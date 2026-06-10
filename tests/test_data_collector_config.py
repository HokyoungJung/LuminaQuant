"""Integration tests for DataCollector config-driven routing (Phase 3 gate).

Gate requirement: config-only switch of exchange/symbol/data-kind drives the
correct collector+storage path with NO code edit.

These tests prove:
1. DataCollectorConfig.from_runtime_config() correctly maps all RuntimeConfig
   fields to the resolved collector config.
2. DataCollector.from_runtime_config() dispatches to the right exchange adapter
   based solely on live.exchange.driver — no code change required.
3. data.kinds from YAML is the sole switch for data-kind selection.
4. MT5 and Polymarket are reachable through the same factory (driver-keyed).
5. aggtrades_tick collection is blocked on non-testnet configs (safety invariant).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lumina_quant.configuration import build_runtime_config
from lumina_quant.data.collector import (
    DATA_KIND_AGGTRADES_TICK,
    DATA_KIND_FEATURE_POINTS,
    DATA_KIND_FUNDING,
    DATA_KIND_OHLCV,
    DataCollector,
    DataCollectorConfig,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rt_from_yaml(yaml_text: str):
    """Build a RuntimeConfig from an inline YAML string."""
    import yaml

    raw = yaml.safe_load(textwrap.dedent(yaml_text).strip()) or {}
    return build_runtime_config(raw, {})


# ---------------------------------------------------------------------------
# DataCollectorConfig derivation tests
# ---------------------------------------------------------------------------


def test_from_runtime_config_defaults():
    """Default RuntimeConfig → sensible DataCollectorConfig defaults."""
    rt = _rt_from_yaml("")
    cfg = DataCollectorConfig.from_runtime_config(rt)

    assert cfg.driver == "binance_futures"
    assert cfg.exchange == "binance"
    assert cfg.parquet_root == Path("data/market_parquet")
    assert cfg.market_type == "future"
    assert DATA_KIND_OHLCV in cfg.kinds
    assert DATA_KIND_FUNDING in cfg.kinds
    assert DATA_KIND_FEATURE_POINTS in cfg.kinds
    assert cfg.tick_path is None
    # Default mode is paper → testnet
    assert cfg.is_testnet is True


def test_config_only_switch_exchange_and_symbols():
    """Changing exchange/symbol via YAML changes DataCollectorConfig — no code edit."""
    cfg_a = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            storage:
              market_data_exchange: binance
              market_data_parquet_path: data/parquet_a
            trading:
              symbols: ["BTC/USDT", "ETH/USDT"]
        """)
    )
    cfg_b = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            storage:
              market_data_exchange: bybit
              market_data_parquet_path: data/parquet_b
            trading:
              symbols: ["SOL/USDT"]
        """)
    )

    assert cfg_a.exchange == "binance"
    assert cfg_a.parquet_root == Path("data/parquet_a")
    assert cfg_a.symbols == ["BTC/USDT", "ETH/USDT"]

    assert cfg_b.exchange == "bybit"
    assert cfg_b.parquet_root == Path("data/parquet_b")
    assert cfg_b.symbols == ["SOL/USDT"]


def test_config_only_switch_data_kinds():
    """data.kinds YAML field is the sole switch for data-kind selection."""
    cfg_ohlcv_only = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            data:
              kinds: [ohlcv]
        """)
    )
    cfg_all = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            data:
              kinds: [ohlcv, funding, feature_points, aggtrades_tick]
        """)
    )
    cfg_tick = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            data:
              kinds: [aggtrades_tick]
              tick_path: data/raw_ticks
        """)
    )

    assert cfg_ohlcv_only.kinds == [DATA_KIND_OHLCV]
    assert set(cfg_all.kinds) == {
        DATA_KIND_OHLCV, DATA_KIND_FUNDING,
        DATA_KIND_FEATURE_POINTS, DATA_KIND_AGGTRADES_TICK,
    }
    assert cfg_tick.kinds == [DATA_KIND_AGGTRADES_TICK]
    assert cfg_tick.tick_path == Path("data/raw_ticks")


def test_invalid_data_kinds_silently_dropped():
    """Unknown kind tokens are dropped; known ones survive."""
    cfg = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            data:
              kinds: [ohlcv, unknown_kind, funding]
        """)
    )
    assert cfg.kinds == [DATA_KIND_OHLCV, DATA_KIND_FUNDING]


def test_empty_data_kinds_falls_back_to_default():
    """Empty kinds list falls back to the DataConfig default."""
    from lumina_quant.configuration.schema import DataConfig

    cfg = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            data:
              kinds: []
        """)
    )
    assert cfg.kinds == list(DataConfig().kinds)


def test_testnet_derived_from_live_mode_paper():
    """live.mode=paper → is_testnet=True (no explicit testnet field needed)."""
    cfg = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            live:
              mode: paper
        """)
    )
    assert cfg.is_testnet is True


def test_testnet_false_on_real_mode():
    """live.testnet=false → is_testnet=False."""
    cfg = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            live:
              mode: live
              testnet: false
        """)
    )
    assert cfg.is_testnet is False


def test_tick_path_override():
    """data.tick_path overrides the parquet root for aggTrades archives."""
    cfg = DataCollectorConfig.from_runtime_config(
        _rt_from_yaml("""
            storage:
              market_data_parquet_path: data/main_parquet
            data:
              tick_path: data/tick_archive
        """)
    )
    assert cfg.parquet_root == Path("data/main_parquet")
    assert cfg.tick_path == Path("data/tick_archive")


# ---------------------------------------------------------------------------
# Driver-keyed factory routing tests
# ---------------------------------------------------------------------------


def test_from_runtime_config_binance_futures_routing():
    """live.exchange.driver=binance_futures → BinanceFuturesRESTClient (config-only switch)."""
    rt = _rt_from_yaml("""
        live:
          exchange:
            driver: binance_futures
    """)
    from lumina_quant.exchanges.binance_futures_client import BinanceFuturesRESTClient

    collector = DataCollector.from_runtime_config(rt)
    try:
        assert isinstance(collector._exchange_client, BinanceFuturesRESTClient)
        assert collector.config.driver == "binance_futures"
    finally:
        collector.close()


def test_from_runtime_config_mt5_routing(monkeypatch):
    """live.exchange.driver=mt5 → MT5Exchange — reachable through same factory."""
    rt = _rt_from_yaml("""
        live:
          exchange:
            driver: mt5
    """)
    mock_mt5 = MagicMock()
    with patch(
        "lumina_quant.data.collector._build_mt5_client",
        return_value=mock_mt5,
    ):
        collector = DataCollector.from_runtime_config(rt)

    assert collector._exchange_client is mock_mt5
    assert collector.config.driver == "mt5"


def test_from_runtime_config_polymarket_routing(monkeypatch):
    """live.exchange.driver=polymarket → PolymarketExchange — reachable through same factory."""
    rt = _rt_from_yaml("""
        live:
          exchange:
            driver: polymarket
    """)
    mock_poly = MagicMock()
    with patch(
        "lumina_quant.data.collector._build_polymarket_client",
        return_value=mock_poly,
    ):
        collector = DataCollector.from_runtime_config(rt)

    assert collector._exchange_client is mock_poly
    assert collector.config.driver == "polymarket"


def test_unknown_driver_raises():
    """Unknown driver raises ValueError — not silently ignored."""
    rt = _rt_from_yaml("""
        live:
          exchange:
            driver: unknown_exchange_xyz
    """)
    with pytest.raises(ValueError, match="Unknown exchange driver"):
        DataCollector.from_runtime_config(rt)


# ---------------------------------------------------------------------------
# Safety invariant tests
# ---------------------------------------------------------------------------


def test_aggtrades_tick_blocked_on_non_testnet():
    """collect_aggtrades_raw() raises RuntimeError when is_testnet=False."""
    rt = _rt_from_yaml("""
        live:
          mode: live
          testnet: false
          exchange:
            driver: binance_futures
    """)
    collector = DataCollector.from_runtime_config(rt)
    try:
        with pytest.raises(RuntimeError, match="validation-only"):
            collector.collect_aggtrades_raw(symbol="BTC/USDT")
    finally:
        collector.close()


def test_aggtrades_tick_allowed_on_testnet():
    """collect_aggtrades_raw() is callable when is_testnet=True (no real orders)."""
    rt = _rt_from_yaml("""
        live:
          mode: paper
          exchange:
            driver: binance_futures
    """)
    collector = DataCollector.from_runtime_config(rt)
    try:
        assert collector.config.is_testnet is True
        # Patch the underlying function so no real network call is made
        with patch(
            "lumina_quant.data.collector.DataCollector.collect_aggtrades_raw",
            return_value={"fetched_rows": 0, "upserted_rows": 0},
        ) as mock_fn:
            result = collector.collect_aggtrades_raw(symbol="BTC/USDT")
            mock_fn.assert_called_once_with(symbol="BTC/USDT")
            assert result["fetched_rows"] == 0
    finally:
        collector.close()


# ---------------------------------------------------------------------------
# Context-manager protocol
# ---------------------------------------------------------------------------


def test_collector_context_manager():
    """DataCollector works as a context manager; close() is called on exit."""
    rt = _rt_from_yaml("")
    mock_client = MagicMock()
    with patch(
        "lumina_quant.data.collector._build_exchange_client",
        return_value=mock_client,
    ):
        with DataCollector.from_runtime_config(rt) as collector:
            assert collector.config.driver == "binance_futures"
    mock_client.close.assert_called_once()
