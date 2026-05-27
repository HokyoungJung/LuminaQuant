from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lumina_quant.strategy_factory import research_runner


def _write_runtime_config(
    tmp_path: Path,
    *,
    symbols: list[str],
    root_path: str,
    exchange: str,
) -> str:
    cfg = textwrap.dedent(
        f"""
        trading:
          symbols: {symbols!r}
          timeframe: \"15m\"
        storage:
          backend: \"local\"
          market_data_parquet_path: \"{root_path}\"
          market_data_exchange: \"{exchange}\"
        live:
          mode: \"paper\"
          exchange:
            driver: \"binance_futures\"
            name: \"binance\"
            market_type: \"future\"
            position_mode: \"HEDGE\"
            margin_mode: \"isolated\"
            leverage: 2
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return str(cfg_path)


def test_run_candidate_research_uses_runtime_symbols_when_symbol_universe_is_omitted(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    report = research_runner.run_candidate_research(
        candidates=[],
        strategy_timeframes=["1m"],
        symbol_universe=None,
    )

    assert report["symbol_universe"] == ["ETH/USDT", "SOL/USDT"]


def test_build_default_candidate_rows_uses_runtime_symbols_when_symbols_are_omitted(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    captured: dict[str, object] = {}

    def _stub_build_candidates(*, symbols, timeframes):
        captured.update({"symbols": list(symbols), "timeframes": list(timeframes)})
        return []

    monkeypatch.setattr(
        "lumina_quant.strategy_factory.candidate_library.build_binance_futures_candidates",
        _stub_build_candidates,
    )

    rows = research_runner.build_default_candidate_rows(symbols=None, timeframes=["1m"])

    assert rows == []
    assert captured["symbols"] == ["ETH/USDT", "SOL/USDT"]
    assert captured["timeframes"] == ["1m"]


def test_load_bundle_cache_uses_runtime_market_data_settings(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    captured: dict[str, object] = {}

    def _stub_load_data_dict(
        root_path,
        *,
        exchange,
        symbol_list,
        timeframe,
        start_date=None,
        end_date=None,
        data_mode="legacy",
    ):
        captured.update(
            {
                "root_path": root_path,
                "exchange": exchange,
                "symbol_list": list(symbol_list),
                "timeframe": timeframe,
                "data_mode": data_mode,
            }
        )
        return {}

    monkeypatch.setattr(research_runner, "load_data_dict_from_parquet", _stub_load_data_dict)

    with pytest.raises(research_runner.RawFirstDataMissingError):
        research_runner._load_bundle_cache(
            symbols=["ETH/USDT"],
            timeframes=["1m"],
            allow_csv_fallback=False,
            allow_synthetic_fallback=False,
        )

    assert captured["root_path"] == "var/data/research_runtime"
    assert captured["exchange"] == "kraken"


def test_load_bundle_cache_honors_explicit_market_data_settings(
    monkeypatch,
):
    captured: dict[str, object] = {}

    def _stub_load_data_dict(
        root_path,
        *,
        exchange,
        symbol_list,
        timeframe,
        start_date=None,
        end_date=None,
        data_mode="legacy",
    ):
        captured.update(
            {
                "root_path": root_path,
                "exchange": exchange,
                "symbol_list": list(symbol_list),
                "timeframe": timeframe,
                "data_mode": data_mode,
            }
        )
        return {}

    monkeypatch.setattr(research_runner, "load_data_dict_from_parquet", _stub_load_data_dict)

    with pytest.raises(research_runner.RawFirstDataMissingError):
        research_runner._load_bundle_cache(
            symbols=["ETH/USDT"],
            timeframes=["1m"],
            allow_csv_fallback=False,
            allow_synthetic_fallback=False,
            market_data_settings={
                "symbols": ["BTC/USDT"],
                "market_data_parquet_path": "explicit/runtime/root",
                "market_data_exchange": "bybit",
            },
        )

    assert captured["root_path"] == "explicit/runtime/root"
    assert captured["exchange"] == "bybit"
    assert captured["symbol_list"] == ["ETH/USDT"]


def test_load_bundle_cache_uses_csv_fallback_with_date_bounds(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "BTCUSDT.csv").write_text(
        "\n".join(
            [
                "datetime,open,high,low,close,volume",
                "2024-01-01T00:00:00,100,101,99,100.5,10",
                "2024-01-02T00:00:00,101,102,100,101.5,12",
                "2024-01-03T00:00:00,102,103,101,102.5,11",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(research_runner, "load_data_dict_from_parquet", lambda *args, **kwargs: {})

    cache, source_map = research_runner._load_bundle_cache(
        symbols=["BTC/USDT"],
        timeframes=["1m"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        allow_synthetic_fallback=False,
        min_bars=1,
        market_data_settings={
            "symbols": ["BTC/USDT"],
            "market_data_parquet_path": "unused",
            "market_data_exchange": "binance",
        },
    )

    bundle = cache[("BTC/USDT", "1m")]
    assert source_map["csv"] == ["BTC/USDT@1m"]
    assert bundle.close.tolist() == [101.5]


def test_load_bundle_cache_uses_synthetic_fallback_when_other_sources_are_missing(monkeypatch):
    bundle = research_runner.SeriesBundle(
        symbol="BTC/USDT",
        timeframe="1m",
        datetime=np.asarray(["2024-01-01T00:00:00"], dtype="datetime64[ms]"),
        open=np.asarray([100.0], dtype=float),
        high=np.asarray([101.0], dtype=float),
        low=np.asarray([99.0], dtype=float),
        close=np.asarray([100.5], dtype=float),
        volume=np.asarray([10.0], dtype=float),
    )

    monkeypatch.setattr(research_runner, "load_data_dict_from_parquet", lambda *args, **kwargs: {})
    monkeypatch.setattr(research_runner, "_load_csv_bundle", lambda **kwargs: None)
    monkeypatch.setattr(research_runner, "_synthetic_bundle", lambda *args, **kwargs: bundle)

    cache, source_map = research_runner._load_bundle_cache(
        symbols=["BTC/USDT"],
        timeframes=["1m"],
        allow_csv_fallback=True,
        allow_synthetic_fallback=True,
        min_bars=1,
        market_data_settings={
            "symbols": ["BTC/USDT"],
            "market_data_parquet_path": "unused",
            "market_data_exchange": "binance",
        },
    )

    assert cache[("BTC/USDT", "1m")] is bundle
    assert source_map["synthetic"] == ["BTC/USDT@1m"]


def test_load_bundle_cache_prefers_parquet_frames_and_records_their_source(monkeypatch):
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 1, 0, 0, 0)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        }
    )

    monkeypatch.setattr(
        research_runner, "load_data_dict_from_parquet", lambda *args, **kwargs: {"BTC/USDT": frame}
    )
    monkeypatch.setattr(
        research_runner,
        "_load_csv_bundle",
        lambda **kwargs: pytest.fail("csv fallback should not run"),
    )
    monkeypatch.setattr(
        research_runner,
        "_synthetic_bundle",
        lambda *args, **kwargs: pytest.fail("synthetic fallback should not run"),
    )

    cache, source_map = research_runner._load_bundle_cache(
        symbols=["BTC/USDT"],
        timeframes=["1m"],
        allow_csv_fallback=True,
        allow_synthetic_fallback=True,
        min_bars=1,
        market_data_settings={
            "symbols": ["BTC/USDT"],
            "market_data_parquet_path": "unused",
            "market_data_exchange": "binance",
        },
    )

    bundle = cache[("BTC/USDT", "1m")]
    assert bundle.close.tolist() == [100.5]
    assert source_map["parquet"] == ["BTC/USDT@1m"]


def test_load_bundle_cache_emits_symbol_timeframe_progress(monkeypatch):
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 1, 0, 0, 0)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        }
    )
    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        research_runner, "load_data_dict_from_parquet", lambda *args, **kwargs: {"BTC/USDT": frame}
    )
    counter = iter([10.0, 10.125, 10.5, 10.625])
    monkeypatch.setattr(research_runner, "perf_counter", lambda: next(counter))

    cache, source_map = research_runner._load_bundle_cache(
        symbols=["BTC/USDT"],
        timeframes=["1m"],
        allow_csv_fallback=True,
        allow_synthetic_fallback=True,
        min_bars=1,
        market_data_settings={
            "symbols": ["BTC/USDT"],
            "market_data_parquet_path": "unused",
            "market_data_exchange": "binance",
        },
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    assert cache[("BTC/USDT", "1m")].close.tolist() == [100.5]
    assert source_map["parquet"] == ["BTC/USDT@1m"]
    assert events == [
        (
            "resource_bundle_timeframe_started",
            {
                "timeframe": "1m",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "symbol_count": 1,
                "loaded_count": 0,
                "total_count": 1,
            },
        ),
        (
            "resource_bundle_timeframe_completed",
            {
                "timeframe": "1m",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "symbol_count": 1,
                "parquet_symbol_count": 1,
                "missing_symbol_count": 0,
                "loaded_count": 0,
                "total_count": 1,
                "elapsed_seconds": 0.125,
            },
        ),
        (
            "resource_bundle_item_loaded",
            {
                "symbol": "BTC/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "timeframe": "1m",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "loaded_count": 1,
                "total_count": 1,
                "source": "parquet",
                "bar_count": 1,
                "elapsed_seconds": 0.125,
            },
        ),
    ]


def test_load_feature_cache_uses_runtime_market_data_settings(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    captured: dict[str, object] = {}

    def _stub_load_feature_points(
        db_path,
        *,
        exchange,
        symbol,
        start_date=None,
        end_date=None,
    ):
        captured.update({"db_path": db_path, "exchange": exchange, "symbol": symbol})
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(
        research_runner, "load_futures_feature_points_from_db", _stub_load_feature_points
    )

    cache = research_runner._load_feature_cache(symbols=["ETH/USDT"])
    assert "ETH/USDT" in cache
    assert cache["ETH/USDT"].is_empty()
    assert captured["db_path"] == "var/data/research_runtime"
    assert captured["exchange"] == "kraken"
    assert captured["symbol"] == "ETH/USDT"


def test_load_feature_cache_normalizes_feature_frames(monkeypatch):
    lead_field = research_runner._FEATURE_POINT_COLUMNS[0]
    trailing_field = research_runner._FEATURE_POINT_COLUMNS[-1]

    monkeypatch.setattr(
        research_runner,
        "load_futures_feature_points_from_db",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "timestamp_ms": [1_000, 1_000, 2_000],
                lead_field: [0.1, 0.2, None],
            }
        ),
    )

    cache = research_runner._load_feature_cache(
        symbols=["ETH/USDT"],
        market_data_settings={
            "symbols": ["ETH/USDT"],
            "market_data_parquet_path": "unused",
            "market_data_exchange": "binance",
        },
    )

    frame = cache["ETH/USDT"]
    assert frame["timestamp_ms"].to_list() == [1_000, 2_000]
    assert frame["datetime"].to_list() == [
        datetime(1970, 1, 1, 0, 0, 1),
        datetime(1970, 1, 1, 0, 0, 2),
    ]
    assert frame[lead_field].to_list() == [0.2, 0.2]
    assert trailing_field in frame.columns


def test_load_feature_cache_emits_symbol_progress(monkeypatch):
    lead_field = research_runner._FEATURE_POINT_COLUMNS[0]
    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        research_runner,
        "load_futures_feature_points_from_db",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "timestamp_ms": [1_000, 2_000],
                lead_field: [0.1, 0.2],
            }
        ),
    )
    counter = iter([20.0, 20.25])
    monkeypatch.setattr(research_runner, "perf_counter", lambda: next(counter))

    cache = research_runner._load_feature_cache(
        symbols=["ETH/USDT"],
        market_data_settings={
            "symbols": ["ETH/USDT"],
            "market_data_parquet_path": "unused",
            "market_data_exchange": "binance",
        },
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    assert cache["ETH/USDT"].height == 2
    assert events == [
        (
            "resource_feature_symbol_started",
            {
                "symbol": "ETH/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "loaded_count": 0,
            },
        ),
        (
            "resource_feature_symbol_loaded",
            {
                "symbol": "ETH/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "loaded_count": 1,
                "row_count": 2,
                "elapsed_seconds": 0.25,
            },
        ),
    ]


def test_benchmark_cache_emits_timeframe_progress():
    bundle = research_runner.SeriesBundle(
        symbol="BTC/USDT",
        timeframe="1m",
        datetime=np.asarray(["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[ms]"),
        open=np.asarray([100.0, 101.0], dtype=float),
        high=np.asarray([101.0, 102.0], dtype=float),
        low=np.asarray([99.0, 100.0], dtype=float),
        close=np.asarray([100.0, 101.0], dtype=float),
        volume=np.asarray([10.0, 11.0], dtype=float),
    )
    events: list[tuple[str, dict[str, object]]] = []
    original_perf_counter = research_runner.perf_counter
    counter = iter([30.0, 30.5])
    research_runner.perf_counter = lambda: next(counter)

    try:
        benchmark = research_runner._benchmark_cache(
            {("BTC/USDT", "1m"): bundle},
            ["1m"],
            progress_callback=lambda event, payload: events.append((event, dict(payload))),
        )
    finally:
        research_runner.perf_counter = original_perf_counter

    assert benchmark["1m"]["returns"].tolist() == [0.0, 0.01]
    assert events == [
        (
            "resource_benchmark_timeframe_started",
            {
                "timeframe": "1m",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "built_count": 0,
            },
        ),
        (
            "resource_benchmark_timeframe_built",
            {
                "timeframe": "1m",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "built_count": 1,
                "return_count": 2,
                "elapsed_seconds": 0.5,
            },
        ),
    ]


def test_synthetic_bundle_is_deterministic_for_symbol_and_timeframe():
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 31, 0)
    first = research_runner._synthetic_bundle(
        "BTC/USDT", "1m", bars=32, start_date=start, end_date=end
    )
    second = research_runner._synthetic_bundle(
        "BTC/USDT", "1m", bars=32, start_date=start, end_date=end
    )

    assert np.array_equal(first.datetime, second.datetime)
    assert np.array_equal(first.open, second.open)
    assert np.array_equal(first.high, second.high)
    assert np.array_equal(first.low, second.low)
    assert np.array_equal(first.close, second.close)
    assert np.array_equal(first.volume, second.volume)


def test_synthetic_bundle_expands_to_cover_requested_date_window():
    bundle = research_runner._synthetic_bundle(
        "BTC/USDT",
        "1m",
        bars=32,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert bundle.datetime.size == 2880
    assert bundle.datetime[0] == np.datetime64(datetime(2024, 1, 1, 0, 0, 0), "ms")
    assert bundle.datetime[-1] == np.datetime64(datetime(2024, 1, 2, 23, 59, 0), "ms")


def test_current_research_market_data_settings_accepts_explicit_runtime_mapping():
    settings = research_runner._current_research_market_data_settings(
        {
            "symbols": ["eth/usdt", "sol/usdt"],
            "market_data_parquet_path": "explicit/runtime/root",
            "market_data_exchange": "kraken",
        }
    )

    assert settings["symbols"] == ["ETH/USDT", "SOL/USDT"]
    assert settings["parquet_root"] == "explicit/runtime/root"
    assert settings["exchange"] == "kraken"


def test_resolve_feature_points_path_uses_runtime_market_data_path(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    feature_points_dir = tmp_path / "var" / "data" / "research_runtime" / "feature_points"
    feature_points_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    assert research_runner._resolve_feature_points_path() == feature_points_dir.resolve()
