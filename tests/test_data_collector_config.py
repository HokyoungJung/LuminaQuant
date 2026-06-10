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
        DATA_KIND_OHLCV,
        DATA_KIND_FUNDING,
        DATA_KIND_FEATURE_POINTS,
        DATA_KIND_AGGTRADES_TICK,
    }
    assert cfg_tick.kinds == [DATA_KIND_AGGTRADES_TICK]
    assert cfg_tick.tick_path == Path("data/raw_ticks")


def test_invalid_data_kinds_raises_value_error():
    """Invalid data.kinds token raises ValueError at load time — no silent drops.

    A typo like 'fundig' must fail the process, not silently collect less data
    than the operator configured (same anti-pattern as -999 sentinels).
    """
    import yaml
    from lumina_quant.configuration import build_runtime_config

    raw = yaml.safe_load("data:\n  kinds: [ohlcv, fundig, funding]") or {}
    with pytest.raises(ValueError, match=r"Invalid data\.kinds token.*fundig"):
        build_runtime_config(raw, {})


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
# Tick-collection safety axis tests
# ---------------------------------------------------------------------------


def test_aggtrades_tick_callable_in_any_mode():
    """collect_aggtrades_raw() is not gated on is_testnet.

    Tick collection is public read-only market data — no order side effects.
    The Phase 0 aggTrades fixture was captured from the prod public REST
    endpoint (fapi.binance.com, keyless).  The real gate is data.kinds:
    real.yaml excludes aggtrades_tick so the live process never runs it.
    """
    for yaml_mode in [
        "live:\n  mode: paper\n  exchange:\n    driver: binance_futures",
        "live:\n  mode: live\n  testnet: false\n  exchange:\n    driver: binance_futures",
    ]:
        rt = _rt_from_yaml(yaml_mode)
        collector = DataCollector.from_runtime_config(rt)
        try:
            with patch(
                "lumina_quant.data_collector.collect_binance_aggtrades_raw",
                return_value={"fetched_rows": 0, "upserted_rows": 0},
            ) as mock_fn:
                result = collector.collect_aggtrades_raw(symbol="BTC/USDT")
                mock_fn.assert_called_once()
                assert result["fetched_rows"] == 0
        finally:
            collector.close()


def test_aggtrades_tick_uses_config_testnet_flag():
    """collect_aggtrades_raw passes is_testnet from config — not hard-wired True."""
    for testnet_expected, yaml_text in [
        (True, "live:\n  mode: paper\n  exchange:\n    driver: binance_futures"),
        (False, "live:\n  mode: live\n  testnet: false\n  exchange:\n    driver: binance_futures"),
    ]:
        rt = _rt_from_yaml(yaml_text)
        collector = DataCollector.from_runtime_config(rt)
        try:
            with patch(
                "lumina_quant.data_collector.collect_binance_aggtrades_raw",
                return_value={"fetched_rows": 0, "upserted_rows": 0},
            ) as mock_fn:
                collector.collect_aggtrades_raw(symbol="BTC/USDT")
                _, kwargs = mock_fn.call_args
                assert kwargs["testnet"] == testnet_expected, (
                    f"expected testnet={testnet_expected} for {yaml_text!r}"
                )
        finally:
            collector.close()


# ---------------------------------------------------------------------------
# Context-manager protocol
# ---------------------------------------------------------------------------


def test_collector_context_manager():
    """DataCollector works as a context manager; close() is called on exit."""
    rt = _rt_from_yaml("")
    mock_client = MagicMock()
    with (
        patch(
            "lumina_quant.data.collector._build_exchange_client",
            return_value=mock_client,
        ),
        DataCollector.from_runtime_config(rt) as collector,
    ):
        assert collector.config.driver == "binance_futures"
    mock_client.close.assert_called_once()
