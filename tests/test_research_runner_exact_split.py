from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from lumina_quant.strategy_factory import research_runner


def _bundle(symbol: str, timestamps: list[datetime]) -> research_runner.SeriesBundle:
    closes = np.linspace(100.0, 110.0, len(timestamps), dtype=float)
    return research_runner.SeriesBundle(
        symbol=symbol,
        timeframe="1d",
        datetime=np.asarray([np.datetime64(ts.replace(tzinfo=None), "ms") for ts in timestamps]),
        open=closes - 0.1,
        high=closes + 0.2,
        low=closes - 0.2,
        close=closes,
        volume=np.full(len(timestamps), 1_000.0, dtype=float),
    )


def test_align_bundles_uses_exact_timestamp_intersection():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bundle_a = _bundle(
        "BTC/USDT",
        [start + timedelta(days=offset) for offset in range(420)],
    )
    bundle_b = _bundle(
        "ETH/USDT",
        [start + timedelta(days=offset) for offset in range(5, 425)],
    )

    aligned = research_runner._align_bundles([bundle_a, bundle_b])

    assert aligned is not None
    datetimes = np.asarray(aligned["datetime"], dtype="datetime64[ms]")
    assert datetimes.size == 415
    assert datetimes[0] == np.datetime64(datetime(2026, 1, 6), "ms")
    assert datetimes[-1] == np.datetime64(datetime(2027, 2, 24), "ms")


def test_run_candidate_research_exact_split_emits_timestamped_streams(monkeypatch):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(days=offset) for offset in range(420)]
    bundle = _bundle("BTC/USDT", timestamps)

    def _mock_load_bundle_cache(*, symbols, timeframes, start_date=None, end_date=None):
        _ = symbols, timeframes, start_date, end_date
        return {("BTC/USDT", "1d"): bundle}, {"parquet": ["BTC/USDT@1d"], "csv": [], "synthetic": []}

    def _mock_load_feature_cache(*, symbols, start_date=None, end_date=None):
        _ = symbols, start_date, end_date
        return {}

    def _mock_strategy_signal(candidate, *, aligned, symbols):
        _ = candidate, symbols
        length = len(aligned["datetime"])
        returns = np.linspace(0.001, 0.003, length, dtype=float)
        turnover = np.zeros(length, dtype=float)
        exposure = np.ones(length, dtype=float)
        return returns, turnover, exposure, {}

    monkeypatch.setattr(research_runner, "_load_bundle_cache", _mock_load_bundle_cache)
    monkeypatch.setattr(research_runner, "_load_feature_cache", _mock_load_feature_cache)
    monkeypatch.setattr(research_runner, "_strategy_signal", _mock_strategy_signal)

    report = research_runner.run_candidate_research(
        candidates=[
            {
                "candidate_id": "exact-1",
                "name": "exact-1",
                "strategy_class": "CompositeTrendStrategy",
                "strategy_timeframe": "1d",
                "symbols": ["BTC/USDT"],
                "params": {},
            }
        ],
        strategy_timeframes=["1d"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=8,
        split={
            "train_start": "2026-01-01",
            "train_end": "2026-01-31",
            "val_start": "2026-02-01",
            "val_end": "2026-02-28",
            "oos_start": "2026-03-01",
            "oos_end": "2026-03-08",
            "strategy_timeframe": "1d",
            "mode": "exact_dates",
        },
    )

    row = next(iter(report.get("candidates") or []))
    train_stream = list((row.get("return_streams") or {}).get("train") or [])
    val_stream = list((row.get("return_streams") or {}).get("val") or [])
    oos_stream = list((row.get("return_streams") or {}).get("oos") or [])

    assert len(train_stream) == 31
    assert len(val_stream) == 28
    assert len(oos_stream) == 8
    assert train_stream[0]["t"] == 1_767_225_600_000
    assert val_stream[0]["t"] == 1_769_904_000_000
    assert oos_stream[-1]["t"] == 1_772_928_000_000
    assert report["split"]["mode"] == "exact_dates"
