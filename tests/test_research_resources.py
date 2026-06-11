from __future__ import annotations

import numpy as np
import polars as pl

from lumina_quant.strategy_factory.research_resources import ResearchResourceLoader


def test_research_resource_loader_loads_feature_symbols_for_carry_trend_rotation():
    captured: dict[str, object] = {}

    def _load_bundle_cache(**kwargs):
        _ = kwargs
        return {}, {}

    def _load_feature_cache(**kwargs):
        captured.update(kwargs)
        return {}

    loader = ResearchResourceLoader(
        split_window_bounds=lambda split: (None, None),
        datetime_to_iso_z=lambda value: None,
        load_bundle_cache=_load_bundle_cache,
        load_feature_cache=_load_feature_cache,
        benchmark_cache=lambda cache, normalized_timeframes: {},
        canonicalize_symbol_list=lambda values: list(dict.fromkeys(values)),
    )

    loader.load(
        adapted=[
            {
                "strategy_class": "CarryTrendFactorRotationStrategy",
                "symbols": ["BTC/USDT", "ETH/USDT"],
            }
        ],
        normalized_timeframes=["1h"],
        universe=["BTC/USDT", "ETH/USDT"],
        resolved_split={},
        data_mode="legacy",
        allow_csv_fallback=True,
        allow_synthetic_fallback=True,
        min_bundle_bars=360,
    )

    assert captured.get("symbols") == ["BTC/USDT", "ETH/USDT"]


def test_research_resource_loader_passes_exact_candidate_bundle_pairs():
    captured: dict[str, object] = {}

    def _load_bundle_cache(**kwargs):
        captured.update(kwargs)
        return {}, {}

    def _load_feature_cache(**kwargs):
        _ = kwargs
        return {}

    loader = ResearchResourceLoader(
        split_window_bounds=lambda split: (None, None),
        datetime_to_iso_z=lambda value: None,
        load_bundle_cache=_load_bundle_cache,
        load_feature_cache=_load_feature_cache,
        benchmark_cache=lambda cache, normalized_timeframes: {},
        canonicalize_symbol_list=lambda values: list(dict.fromkeys(values)),
    )

    loader.load(
        adapted=[
            {
                "strategy_class": "CompositeTrendStrategy",
                "strategy_timeframe": "1h",
                "symbols": ["BTC/USDT"],
            },
            {
                "strategy_class": "AbnormalReturnContinuationStrategy",
                "strategy_timeframe": "1d",
                "symbols": ["ORCL/USDT"],
            },
        ],
        normalized_timeframes=["1h", "1d"],
        universe=["BTC/USDT", "ORCL/USDT"],
        resolved_split={},
        data_mode="legacy",
        allow_csv_fallback=True,
        allow_synthetic_fallback=False,
        min_bundle_bars=360,
    )

    assert captured["required_pairs"] == [
        ("BTC/USDT", "1h"),
        ("ORCL/USDT", "1d"),
    ]


def test_research_resource_loader_emits_finer_grained_resource_progress(monkeypatch):
    events: list[tuple[str, dict[str, object]]] = []
    counter = iter([1.0, 1.4, 2.0, 2.3, 3.0, 3.6])
    monkeypatch.setattr(
        "lumina_quant.strategy_factory.research_resources.perf_counter",
        lambda: next(counter),
    )

    def _progress_callback(event: str, payload: dict[str, object]) -> None:
        events.append((event, dict(payload)))

    def _load_bundle_cache(**kwargs):
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback(
                "resource_bundle_timeframe_started",
                {
                    "timeframe": "1h",
                    "timeframe_index": 1,
                    "timeframe_count": 1,
                    "symbol_count": 1,
                    "loaded_count": 0,
                    "total_count": 1,
                },
            )
            callback(
                "resource_bundle_timeframe_completed",
                {
                    "timeframe": "1h",
                    "timeframe_index": 1,
                    "timeframe_count": 1,
                    "symbol_count": 1,
                    "parquet_symbol_count": 1,
                    "missing_symbol_count": 0,
                    "loaded_count": 0,
                    "total_count": 1,
                    "elapsed_seconds": 0.4,
                },
            )
            callback(
                "resource_bundle_symbol_fetch_started",
                {
                    "symbol": "BTC/USDT",
                    "symbol_index": 1,
                    "symbol_count": 1,
                    "timeframe": "1h",
                    "data_mode": "legacy",
                },
            )
            callback(
                "resource_bundle_symbol_window_loaded",
                {
                    "symbol": "BTC/USDT",
                    "symbol_index": 1,
                    "symbol_count": 1,
                    "timeframe": "1h",
                    "unit_kind": "chunk",
                    "unit_index": 1,
                    "unit_count": 1,
                    "row_count": 128,
                    "elapsed_seconds": 0.1,
                },
            )
            callback(
                "resource_bundle_symbol_fetch_completed",
                {
                    "symbol": "BTC/USDT",
                    "symbol_index": 1,
                    "symbol_count": 1,
                    "timeframe": "1h",
                    "data_mode": "legacy",
                    "row_count": 128,
                    "was_missing": False,
                    "elapsed_seconds": 0.2,
                },
            )
            callback(
                "resource_bundle_item_loaded",
                {
                    "symbol": "BTC/USDT",
                    "symbol_index": 1,
                    "symbol_count": 1,
                    "timeframe": "1h",
                    "timeframe_index": 1,
                    "timeframe_count": 1,
                    "loaded_count": 1,
                    "total_count": 1,
                    "source": "parquet",
                    "bar_count": 128,
                },
            )
        return {("BTC/USDT", "1h"): object()}, {"parquet": ["BTC/USDT@1h"]}

    def _load_feature_cache(**kwargs):
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback(
                "resource_feature_symbol_started",
                {
                    "symbol": "BTC/USDT",
                    "symbol_index": 1,
                    "symbol_count": 1,
                    "loaded_count": 0,
                },
            )
            callback(
                "resource_feature_partition_scan_completed",
                {
                    "symbol": "BTC/USDT",
                    "partition_count": 1,
                    "parquet_file_count": 1,
                    "elapsed_seconds": 0.1,
                },
            )
            callback(
                "resource_feature_collect_started",
                {
                    "symbol": "BTC/USDT",
                    "partition_count": 1,
                    "parquet_file_count": 1,
                },
            )
            callback(
                "resource_feature_collect_completed",
                {
                    "symbol": "BTC/USDT",
                    "partition_count": 1,
                    "parquet_file_count": 1,
                    "row_count": 64,
                    "elapsed_seconds": 0.2,
                },
            )
            callback(
                "resource_feature_symbol_loaded",
                {
                    "symbol": "BTC/USDT",
                    "symbol_index": 1,
                    "symbol_count": 1,
                    "loaded_count": 1,
                    "row_count": 64,
                },
            )
        return {"BTC/USDT": pl.DataFrame({"timestamp_ms": [1]})}

    def _benchmark_cache(cache, timeframes, progress_callback=None):
        _ = cache
        if callable(progress_callback):
            progress_callback(
                "resource_benchmark_timeframe_started",
                {
                    "timeframe": "1h",
                    "timeframe_index": 1,
                    "timeframe_count": 1,
                    "built_count": 0,
                },
            )
            progress_callback(
                "resource_benchmark_timeframe_built",
                {
                    "timeframe": "1h",
                    "timeframe_index": 1,
                    "timeframe_count": 1,
                    "built_count": 1,
                    "return_count": 63,
                },
            )
        return {"1h": {"returns": np.asarray([0.1, 0.2])}}

    loader = ResearchResourceLoader(
        split_window_bounds=lambda split: (None, None),
        datetime_to_iso_z=lambda value: None,
        load_bundle_cache=_load_bundle_cache,
        load_feature_cache=_load_feature_cache,
        benchmark_cache=_benchmark_cache,
        canonicalize_symbol_list=lambda values: list(dict.fromkeys(values)),
    )

    loader.load(
        adapted=[
            {
                "strategy_class": "CarryTrendFactorRotationStrategy",
                "symbols": ["BTC/USDT"],
            }
        ],
        normalized_timeframes=["1h"],
        universe=["BTC/USDT"],
        resolved_split={},
        data_mode="legacy",
        allow_csv_fallback=True,
        allow_synthetic_fallback=True,
        min_bundle_bars=360,
        progress_callback=_progress_callback,
    )

    event_names = [name for name, _ in events]
    assert event_names == [
        "resource_bundle_load_started",
        "resource_bundle_timeframe_started",
        "resource_bundle_timeframe_completed",
        "resource_bundle_symbol_fetch_started",
        "resource_bundle_symbol_window_loaded",
        "resource_bundle_symbol_fetch_completed",
        "resource_bundle_item_loaded",
        "resource_bundle_load_completed",
        "resource_feature_load_started",
        "resource_feature_symbol_started",
        "resource_feature_partition_scan_completed",
        "resource_feature_collect_started",
        "resource_feature_collect_completed",
        "resource_feature_symbol_loaded",
        "resource_feature_load_completed",
        "resource_benchmark_build_started",
        "resource_benchmark_timeframe_started",
        "resource_benchmark_timeframe_built",
        "resource_benchmark_build_completed",
    ]
    assert events[2][1]["parquet_symbol_count"] == 1
    assert events[2][1]["elapsed_seconds"] == 0.4
    assert events[4][1]["unit_kind"] == "chunk"
    assert events[5][1]["was_missing"] is False
    assert events[7][1]["source_counts"] == {"parquet": 1}
    assert events[7][1]["elapsed_seconds"] == 0.4
    assert events[10][1]["partition_count"] == 1
    assert events[12][1]["row_count"] == 64
    assert events[14][1]["total_rows"] == 1
    assert events[14][1]["elapsed_seconds"] == 0.3
    assert events[18][1]["nonempty_timeframe_count"] == 1
    assert events[18][1]["elapsed_seconds"] == 0.6
