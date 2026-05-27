import importlib.util
import sys
from pathlib import Path

import polars as pl

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_grouped_allocator_strict_leverage_batch_retune.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_grouped_allocator_strict_leverage_batch_retune",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prepare_shared_resources_uses_disk_cache(tmp_path: Path, monkeypatch) -> None:
    call_counter = {"count": 0}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        MODULE,
        "current_research_market_data_settings",
        lambda: {"parquet_root": tmp_path / "parquet", "exchange": "binance"},
    )
    monkeypatch.setattr(MODULE._VALIDATION_CORE, "STRICT_VALIDATION_DATA_MODE", "exact")
    monkeypatch.setattr(
        MODULE._runner,
        "_adapt_candidate_inputs",
        lambda candidates, max_candidates: list(candidates),
    )
    monkeypatch.setattr(
        MODULE._runner,
        "_resolve_research_run_timeframes_and_universe",
        lambda adapted, strategy_timeframes, symbol_universe: (
            captured.setdefault("strategy_timeframes", list(strategy_timeframes or [])) and ["1h"],
            captured.setdefault("symbol_universe", list(symbol_universe or [])) and ["BTC/USDT"],
        ),
    )
    monkeypatch.setattr(
        MODULE._runner,
        "_resolve_split_config",
        lambda split, strategy_timeframe: {
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "val_start": "2024-07-01",
            "val_end": "2024-09-30",
            "oos_start": "2024-10-01",
            "oos_end": "2024-12-31",
        },
    )

    def _fake_loader(**kwargs):
        call_counter["count"] += 1
        return (
            {
                ("BTC/USDT", "1h"): MODULE._runner.SeriesBundle(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    datetime=[],
                    open=[],
                    high=[],
                    low=[],
                    close=[],
                    volume=[],
                )
            },
            {"parquet": ["BTC/USDT@1h"], "csv": [], "synthetic": []},
            {"BTC/USDT": pl.DataFrame({"timestamp_ms": [1], "funding_rate": [0.1]})},
            {"1h": {"returns": []}},
        )

    monkeypatch.setattr(MODULE._runner, "_load_research_run_resources", _fake_loader)

    candidates = [
        {
            "candidate_id": "cand-1",
            "name": "composite",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT"],
        }
    ]
    split = {"oos_end": "2024-12-31"}

    first, first_source, first_key = MODULE._prepare_shared_resources(
        candidates=candidates,
        split=split,
        cache_dir=tmp_path,
        refresh_cache=False,
    )
    second, second_source, second_key = MODULE._prepare_shared_resources(
        candidates=candidates,
        split=split,
        cache_dir=tmp_path,
        refresh_cache=False,
    )

    assert call_counter["count"] == 1
    assert first_source == "fresh_load"
    assert second_source == "disk_cache"
    assert first_key == second_key
    assert first.data_sources == second.data_sources
    assert captured["strategy_timeframes"] == ["1h"]
    assert captured["symbol_universe"] == ["BTC/USDT"]
    assert sorted(tmp_path.glob("*.pkl"))
    assert sorted(tmp_path.glob("*.meta.json"))


def test_shared_resource_cache_key_changes_with_feature_symbols(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        MODULE,
        "current_research_market_data_settings",
        lambda: {"parquet_root": tmp_path / "parquet", "exchange": "binance"},
    )

    base_payload = MODULE._shared_resource_cache_key_payload(
        normalized_timeframes=["1h"],
        universe=["BTC/USDT"],
        resolved_split={"oos_end": "2024-12-31"},
        candidates=[
            {
                "strategy_class": "CompositeTrendStrategy",
                "symbols": ["BTC/USDT"],
            }
        ],
        data_mode="exact",
        allow_csv_fallback=False,
        allow_synthetic_fallback=False,
        min_bundle_bars=1,
        market_data_settings=MODULE.current_research_market_data_settings(),
    )
    changed_payload = MODULE._shared_resource_cache_key_payload(
        normalized_timeframes=["1h"],
        universe=["BTC/USDT"],
        resolved_split={"oos_end": "2024-12-31"},
        candidates=[
            {
                "strategy_class": "CompositeTrendStrategy",
                "symbols": ["ETH/USDT"],
            }
        ],
        data_mode="exact",
        allow_csv_fallback=False,
        allow_synthetic_fallback=False,
        min_bundle_bars=1,
        market_data_settings=MODULE.current_research_market_data_settings(),
    )

    assert MODULE._shared_resource_cache_key(base_payload) != MODULE._shared_resource_cache_key(
        changed_payload
    )


def test_build_parser_exposes_shared_cache_controls() -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args([])

    assert args.shared_cache_dir == MODULE.DEFAULT_SHARED_RESOURCE_CACHE_DIR
    assert args.refresh_shared_cache is False


def test_candidate_universe_and_timeframes_are_minimal() -> None:
    candidates = [
        {
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1H",
            "symbols": ["BTC/USDT", "ETH/USDT"],
        },
        {
            "strategy_class": "PairSpreadZScoreStrategy",
            "timeframe": "30m",
            "symbols": ["ETH/USDT", "SOL/USDT"],
        },
    ]

    assert MODULE._candidate_timeframes(candidates) == ["1h", "30m"]
    assert MODULE._candidate_universe(candidates) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_load_legacy_shared_resources_fast_resamples_once_per_symbol(
    tmp_path: Path, monkeypatch
) -> None:
    one_second_frame = pl.DataFrame(
        {
            "datetime": [1, 2, 3],
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2],
            "volume": [10.0, 10.0, 10.0],
        }
    )
    load_calls: list[tuple[tuple[str, ...], str]] = []
    resample_calls: list[str] = []

    monkeypatch.setattr(
        MODULE._runner,
        "_split_window_bounds",
        lambda resolved_split: ("2024-01-01", "2024-12-31"),
    )
    monkeypatch.setattr(MODULE._runner, "_datetime_to_iso_z", lambda value: value)
    monkeypatch.setattr(
        MODULE,
        "load_data_dict_from_parquet",
        lambda root_path, *, exchange, symbol_list, timeframe, start_date, end_date, data_mode: (
            load_calls.append((tuple(symbol_list), timeframe))
            or dict.fromkeys(symbol_list, one_second_frame)
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "resample_1s_frame",
        lambda frame, *, timeframe: resample_calls.append(timeframe) or frame,
    )
    monkeypatch.setattr(
        MODULE._runner,
        "_frame_to_bundle",
        lambda symbol, timeframe, frame: {
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": frame.height,
        },
    )
    monkeypatch.setattr(
        MODULE._runner,
        "_load_feature_cache",
        lambda symbols, start_date, end_date, market_data_settings: {"BTC/USDT": pl.DataFrame()},
    )
    monkeypatch.setattr(
        MODULE._runner,
        "_benchmark_cache",
        lambda cache, normalized_timeframes: {tf: {"returns": []} for tf in normalized_timeframes},
    )

    resources = MODULE._load_legacy_shared_resources_fast(
        normalized_timeframes=["30m", "1h"],
        universe=["BTC/USDT", "ETH/USDT"],
        resolved_split={"oos_end": "2024-12-31"},
        candidates=[
            {"strategy_class": "CompositeTrendStrategy", "symbols": ["BTC/USDT"]},
        ],
        market_data_settings={"parquet_root": tmp_path / "parquet", "exchange": "binance"},
    )

    assert load_calls == [(("BTC/USDT", "ETH/USDT"), "1s")]
    assert sorted(resample_calls) == ["1h", "1h", "30m", "30m"]
    assert sorted(resources.cache) == [
        ("BTC/USDT", "1h"),
        ("BTC/USDT", "30m"),
        ("ETH/USDT", "1h"),
        ("ETH/USDT", "30m"),
    ]
