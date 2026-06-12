from __future__ import annotations

from datetime import datetime

import polars as pl

from scripts.research import backtest_latest_alpha_sleeves as subject


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


def test_latest_alpha_sleeve_strategy_catalog_is_complete():
    assert len(subject.NEW_ALPHA_SLEEVE_STRATEGIES) == 19
    assert "FundingDislocationTrendCarryStrategy" in subject.NEW_ALPHA_SLEEVE_STRATEGIES
    assert "MetalEquityDivergenceReversalStrategy" in subject.NEW_ALPHA_SLEEVE_STRATEGIES
    assert subject.FEATURE_SYMBOLS == ("BTC/USDT", "ETH/USDT", "SOL/USDT")


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
        "run_latest_alpha_sleeve_backtests",
        lambda _args: {"issues": [{"severity": "warn", "scope": "S", "message": "excluded"}]},
    )

    assert subject.main([]) == 0


def test_audit_ohlcv_frame_passes_contiguous_real_bars():
    frame = _frame(
        [
            datetime(2026, 6, 1, 0, 0),
            datetime(2026, 6, 1, 0, 1),
            datetime(2026, 6, 1, 0, 2),
        ]
    )

    audit = subject.audit_ohlcv_frame("BTC/USDT", frame, max_gap_ratio=0.0)

    assert audit["status"] == "pass"
    assert audit["missing_1m_bars"] == 0
    assert audit["errors"] == []


def test_audit_ohlcv_frame_fails_missing_bar_without_filling():
    frame = _frame(
        [
            datetime(2026, 6, 1, 0, 0),
            datetime(2026, 6, 1, 0, 2),
        ]
    )

    audit = subject.audit_ohlcv_frame("BTC/USDT", frame, max_gap_ratio=0.0)

    assert audit["status"] == "fail"
    assert audit["missing_1m_bars"] == 1
    assert any(str(item).startswith("missing_1m_bars:1/3") for item in audit["errors"])


def test_audit_ohlcv_frame_warns_but_does_not_impute_zero_volume():
    frame = _frame([datetime(2026, 6, 1, 0, 0), datetime(2026, 6, 1, 0, 1)])
    frame = frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 1).then(0.0).otherwise(10.0).alias("volume")
    )

    audit = subject.audit_ohlcv_frame("XAU/USDT", frame, max_gap_ratio=0.0)

    assert audit["status"] == "pass"
    assert audit["errors"] == []
    assert "volume_zero:1" in audit["warnings"]


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
