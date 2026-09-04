from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from lumina_quant.research_universe import (
    FEATURE_RESEARCH_SYMBOLS,
    research_symbols_for_strategy,
)
from scripts.research import run_strategy_screen as subject


def _frame(times: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": times,
            "open": [100.0 + idx for idx, _ in enumerate(times)],
            "high": [101.0 + idx for idx, _ in enumerate(times)],
            "low": [99.0 + idx for idx, _ in enumerate(times)],
            "close": [100.5 + idx for idx, _ in enumerate(times)],
            "volume": [10.0 for _ in times],
        }
    )


def _audit(
    frame: pl.DataFrame,
    *,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    return subject.audit_ohlcv_frame(
        "BTC/USDT",
        frame,
        start=start,
        end=end,
        max_gap_ratio=0.0,
    )


def test_strategy_research_universe_is_centralized():
    assert FEATURE_RESEARCH_SYMBOLS == ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    assert research_symbols_for_strategy("FundingDislocationTrendCarryStrategy") == (
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
    )
    assert len(subject._strategy_specs(scope="all")) >= len(subject._strategy_specs(scope="live"))


def test_strategy_specs_apply_explicit_delisted_symbol_exclusion() -> None:
    spec = subject._strategy_specs(
        ["PriceVolumeCorrContinuationStrategy"],
        excluded_symbols=("TON/USDT",),
    )[0]

    assert "TON/USDT" not in spec.symbols
    assert "BTC/USDT" in spec.symbols


def test_canonical_output_mode_writes_no_timestamp_or_markdown_copies(
    tmp_path: Path,
) -> None:
    payload = {"generated_at": "2026-09-03T00:00:00Z", "issues": []}

    outputs = subject._write_outputs(payload, tmp_path, canonical_only=True)

    assert outputs == {"latest_json": str(tmp_path / "strategy_screen_latest.json")}
    assert [path.name for path in tmp_path.iterdir()] == ["strategy_screen_latest.json"]


def test_strategy_specs_can_expand_live_scope_without_research_only_names():
    specs = subject._strategy_specs(scope="live")
    names = {spec.strategy for spec in specs}

    assert "RsiStrategy" in names
    assert "FundingDislocationTrendCarryStrategy" in names
    assert "Alpha101FormulaStrategy" not in names


def test_zero_trade_reason_distinguishes_no_signal_from_runtime_failure():
    reason = subject._zero_trade_reason(
        {
            "status": "pass",
            "trade_count": 0,
            "market_events": 120,
            "signals": 0,
            "orders": 0,
            "fills": 0,
            "feature_audit_status": "pass",
        }
    )

    assert reason == "no_signal_generated_under_default_params_window"


def test_required_features_supports_instance_property_shape():
    class PropertyFeatureStrategy:
        def __init__(self, bars, events) -> None:
            self.bars = bars
            self.events = events

        @property
        def required_features(self):
            return ("funding_rate", "open_interest")

    assert subject._required_features_for_strategy(PropertyFeatureStrategy) == (
        "funding_rate",
        "open_interest",
    )


def test_main_returns_success_for_unavailable_warnings(monkeypatch):
    monkeypatch.setattr(
        subject,
        "run_strategy_screen",
        lambda _args: {"issues": [{"severity": "warn", "scope": "S", "message": "excluded"}]},
    )

    assert subject.main([]) == 0


def test_audit_ohlcv_frame_passes_contiguous_real_bars():
    start = datetime(2026, 6, 1, 0, 0)
    end = datetime(2026, 6, 1, 0, 3)
    frame = _frame(
        [
            start,
            datetime(2026, 6, 1, 0, 1),
            datetime(2026, 6, 1, 0, 2),
        ]
    )

    audit = _audit(frame, start=start, end=end)

    assert audit["status"] == "pass"
    assert audit["window_contract"] == "[start,end)"
    assert audit["expected_1m_bars"] == 3
    assert audit["missing_1m_bars"] == 0
    assert audit["errors"] == []


def test_audit_ohlcv_frame_fails_missing_bar_without_filling():
    start = datetime(2026, 6, 1, 0, 0)
    end = datetime(2026, 6, 1, 0, 3)
    frame = _frame(
        [
            start,
            datetime(2026, 6, 1, 0, 2),
        ]
    )

    audit = _audit(frame, start=start, end=end)

    assert audit["status"] == "fail"
    assert audit["missing_1m_bars"] == 1
    assert any(str(item).startswith("missing_1m_bars:1/3") for item in audit["errors"])


def test_audit_ohlcv_frame_warns_but_does_not_impute_zero_volume():
    start = datetime(2026, 6, 1, 0, 0)
    end = datetime(2026, 6, 1, 0, 2)
    frame = _frame([start, datetime(2026, 6, 1, 0, 1)])
    frame = frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 1).then(0.0).otherwise(10.0).alias("volume")
    )

    audit = _audit(frame, start=start, end=end)

    assert audit["status"] == "pass"
    assert audit["errors"] == []
    assert "volume_zero:1" in audit["warnings"]


def test_audit_ohlcv_frame_fails_when_requested_first_bar_is_missing():
    start = datetime(2026, 6, 1, 0, 0)
    end = datetime(2026, 6, 1, 0, 3)
    frame = _frame(
        [
            datetime(2026, 6, 1, 0, 1),
            datetime(2026, 6, 1, 0, 2),
        ]
    )

    audit = _audit(frame, start=start, end=end)

    assert audit["status"] == "fail"
    assert any(str(item).startswith("first_timestamp_mismatch:") for item in audit["errors"])
    assert "missing_1m_bars:1/3" in audit["errors"]


def test_audit_ohlcv_frame_fails_when_requested_last_bar_is_missing():
    start = datetime(2026, 6, 1, 0, 0)
    end = datetime(2026, 6, 1, 0, 3)
    frame = _frame([start, datetime(2026, 6, 1, 0, 1)])

    audit = _audit(frame, start=start, end=end)

    assert audit["status"] == "fail"
    assert any(str(item).startswith("last_timestamp_mismatch:") for item in audit["errors"])
    assert "missing_1m_bars:1/3" in audit["errors"]


def test_audit_ohlcv_frame_fails_off_requested_minute_grid():
    start = datetime(2026, 6, 1, 0, 0)
    end = datetime(2026, 6, 1, 0, 3)
    frame = _frame(
        [
            start,
            datetime(2026, 6, 1, 0, 1, 30),
            datetime(2026, 6, 1, 0, 2),
        ]
    )

    audit = _audit(frame, start=start, end=end)

    assert audit["status"] == "fail"
    assert "off_requested_minute_grid:1" in audit["errors"]
    assert "missing_1m_bars:1/3" in audit["errors"]


def test_load_and_audit_uses_last_included_bar_and_attaches_source_lineage():
    start = datetime(2026, 6, 1, 0, 0)
    end = datetime(2026, 6, 1, 0, 2)
    frame = _frame([start, datetime(2026, 6, 1, 0, 1)])
    captured: dict[str, object] = {}

    class FakeRepo:
        @staticmethod
        def load_ohlcv_with_source_audit(**kwargs):
            captured.update(kwargs)
            return frame, {"precedence": "direct_1m_over_resampled_1s_derived"}

    data, audits = subject._load_and_audit_data(
        cast(Any, FakeRepo()),
        exchange="binance",
        symbols=("BTC/USDT",),
        start=start,
        end=end,
        max_gap_ratio=0.0,
    )

    assert captured["start_date"] == start
    assert captured["end_date"] == datetime(2026, 6, 1, 0, 1)
    assert data["BTC/USDT"].height == 2
    assert audits[0]["source_lineage"] == {"precedence": "direct_1m_over_resampled_1s_derived"}


def test_feature_audit_fails_when_required_features_are_only_null(monkeypatch):
    import lumina_quant.market_data as market_data

    def fake_load_features(*args, **kwargs):
        return pl.DataFrame(
            {
                "datetime": [datetime(2026, 5, 1, 0, 0)],
                "funding_rate": [None],
                "open_interest": [None],
            }
        )

    monkeypatch.setattr(market_data, "load_futures_feature_points_from_db", fake_load_features)

    audit = subject._feature_audit(
        db_path="data/market_parquet",
        exchange="binance",
        symbols=("BTC/USDT",),
        start=datetime(2026, 5, 1, 0, 0),
        end=datetime(2026, 5, 1, 0, 1),
        required_features=("funding_rate", "open_interest"),
    )

    assert audit["status"] == "fail"
    assert audit["complete_feature_symbols"] == []
    assert audit["missing_features_by_symbol"] == {"BTC/USDT": ["funding_rate", "open_interest"]}


def test_feature_audit_requires_complete_bar_window_coverage(monkeypatch):
    import lumina_quant.market_data as market_data

    start = datetime(2026, 5, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)

    def fake_load_features(*args, **kwargs):
        return pl.DataFrame(
            {
                "timestamp_ms": [start_ms, start_ms + 24 * 60 * 60 * 1000],
                "open_interest": [1_000_000.0, 1_100_000.0],
            }
        )

    monkeypatch.setattr(market_data, "load_futures_feature_points_from_db", fake_load_features)

    audit = subject._feature_audit(
        db_path="data/market_parquet",
        exchange="binance",
        symbols=("BTC/USDT",),
        start=start,
        end=datetime(2026, 5, 2, tzinfo=UTC),
        required_features=("open_interest",),
    )

    assert audit["status"] == "fail"
    coverage = audit["coverage_failures_by_symbol"]["BTC/USDT"]["open_interest"]
    assert coverage["available_bars"] < coverage["expected_bars"]
    assert coverage["missing_bars"] > 0


def test_feature_audit_accepts_exact_bounded_forward_fill(monkeypatch):
    import lumina_quant.market_data as market_data

    start = datetime(2026, 5, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    interval_ms = 8 * 60 * 60 * 1000

    def fake_load_features(*args, **kwargs):
        return pl.DataFrame(
            {
                "timestamp_ms": [
                    start_ms,
                    start_ms + interval_ms,
                    start_ms + 2 * interval_ms,
                ],
                "funding_rate": [0.0001, -0.0001, 0.0002],
            }
        )

    monkeypatch.setattr(market_data, "load_futures_feature_points_from_db", fake_load_features)

    audit = subject._feature_audit(
        db_path="data/market_parquet",
        exchange="binance",
        symbols=("BTC/USDT",),
        start=start,
        end=datetime(2026, 5, 1, 16, tzinfo=UTC),
        required_features=("funding_rate",),
    )

    assert audit["status"] == "pass"
    assert audit["coverage_failures_by_symbol"] == {}
    coverage = audit["symbols"]["BTC/USDT"]["coverage"]["funding_rate"]
    assert coverage["available_bars"] == coverage["expected_bars"]
    assert coverage["expected_bars"] == 16 * 60


def test_run_backtest_wires_feature_database_for_preloaded_frames(monkeypatch):
    captured: dict[str, object] = {}

    class FakePortfolio:
        _metric_totals = [10_000.0]
        current_holdings = {"total": 10_000.0}
        trade_count = 0

        @staticmethod
        def output_summary_stats_fast():
            return {}

    class FakeBacktest:
        market_events = 0
        signals = 0
        orders = 0
        fills = 0

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.portfolio = FakePortfolio()

        @staticmethod
        def simulate_trading(*, output):
            assert output is False

    monkeypatch.setattr(subject, "Backtest", FakeBacktest)
    monkeypatch.setattr(subject, "resolve_strategy_class", lambda _name: object)

    subject._run_backtest(
        strategy="FeatureStrategy",
        symbols=("BTC/USDT",),
        data={},
        data_root="data/market_parquet",
        exchange="binance",
        start=datetime(2026, 5, 1),
        end=datetime(2026, 5, 2),
        annual_periods=365 * 24 * 60,
    )

    assert captured["data_handler_kwargs"] == {
        "feature_db_path": "data/market_parquet",
        "feature_exchange": "binance",
    }
