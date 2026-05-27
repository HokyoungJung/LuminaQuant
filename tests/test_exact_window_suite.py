from __future__ import annotations

from datetime import UTC, datetime
import textwrap

import numpy as np
import pytest

from lumina_quant.eval.exact_window_suite import (
    _build_candidate_result_row,
    _monthly_hurdle_rows,
    _min_bars_for_timeframe,
    _portfolio_weights,
    _recent_three_month_two_pct_pass,
    aggregate_stream_by_period,
    half_open_slice_indices,
    run_exact_window_suite,
    resolve_coverage_adaptive_windows,
)


def _ts(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def test_half_open_slice_indices_excludes_boundary_endpoints():
    timestamps = np.asarray(
        [
            np.datetime64("2025-12-31T23:59:59.000", "ms"),
            np.datetime64("2026-01-01T00:00:00.000", "ms"),
            np.datetime64("2026-01-31T23:59:59.000", "ms"),
            np.datetime64("2026-02-01T00:00:00.000", "ms"),
        ]
    )
    validation = half_open_slice_indices(
        timestamps,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert validation.start == 1
    assert validation.stop == 3


def test_aggregate_stream_by_month_compounds_returns():
    stream = [
        {"t": _ts("2026-02-01T00:00:00Z"), "v": 0.10},
        {"t": _ts("2026-02-10T00:00:00Z"), "v": -0.05},
        {"t": _ts("2026-03-01T00:00:00Z"), "v": 0.02},
    ]
    rows = aggregate_stream_by_period(stream, period="month")
    assert rows[0]["period"] == "2026-02"
    assert abs(rows[0]["return"] - ((1.10 * 0.95) - 1.0)) < 1e-12
    assert rows[1]["period"] == "2026-03"
    assert abs(rows[1]["return"] - 0.02) < 1e-12


def test_aggregate_stream_by_day_keeps_calendar_days_separate():
    stream = [
        {"t": _ts("2026-03-07T00:00:00Z"), "v": 0.01},
        {"t": _ts("2026-03-07T12:00:00Z"), "v": 0.02},
        {"t": _ts("2026-03-08T00:00:00Z"), "v": -0.01},
    ]
    rows = aggregate_stream_by_period(stream, period="day")
    assert [row["period"] for row in rows] == ["2026-03-07", "2026-03-08"]
    assert abs(rows[0]["return"] - ((1.01 * 1.02) - 1.0)) < 1e-12
    assert abs(rows[1]["return"] + 0.01) < 1e-12


def test_monthly_hurdle_rows_track_btc_pass_separately_from_strict_floor():
    stream = [{"t": _ts("2026-01-15T00:00:00Z"), "v": 0.015}]
    rows = _monthly_hurdle_rows(
        stream,
        {"2026-01": {"btc_buy_hold_return": 0.01, "threshold": 0.02}},
    )
    assert rows[0]["strict_pass"] is False
    assert rows[0]["btc_pass"] is True
    assert rows[0]["pass"] is False


def test_portfolio_weights_fallback_prefers_btc_beating_candidates_before_full_fallback():
    rows = [
        {
            "candidate_id": "strict-miss-btc-win",
            "name": "btc-win",
            "strategy_class": "CompositeTrendStrategy",
            "family": "trend",
            "strategy_timeframe": "30m",
            "btc_beating_candidate": True,
            "candidate_pool_eligible": True,
            "promoted": False,
            "val": {"deflated_sharpe": 0.6},
            "return_streams": {"val": [{"t": _ts("2026-01-15T00:00:00Z"), "v": 0.01}]},
        },
        {
            "candidate_id": "non-candidate",
            "name": "fallback-only",
            "strategy_class": "PairSpreadZScoreStrategy",
            "family": "market_neutral",
            "strategy_timeframe": "15m",
            "btc_beating_candidate": False,
            "candidate_pool_eligible": False,
            "promoted": False,
            "val": {"deflated_sharpe": 0.8},
            "return_streams": {"val": [{"t": _ts("2026-01-15T00:00:00Z"), "v": 0.03}]},
        },
    ]
    weights = _portfolio_weights(rows)
    assert len(weights) == 1
    assert weights[0]["candidate_id"] == "strict-miss-btc-win"
    assert weights[0]["basis"] == "candidate_pool_candidates"


def test_recent_three_month_two_pct_pass_requires_three_recent_months():
    recent_rows = [
        {"month": "2026-01", "strategy_return": 0.021},
        {"month": "2026-02", "strategy_return": 0.024},
        {"month": "2026-03", "strategy_return": 0.0201},
    ]
    assert _recent_three_month_two_pct_pass(recent_rows) is True
    assert _recent_three_month_two_pct_pass(recent_rows[:2]) is False
    assert (
        _recent_three_month_two_pct_pass(
            [
                {"month": "2026-01", "strategy_return": 0.021},
                {"month": "2026-02", "strategy_return": 0.019},
                {"month": "2026-03", "strategy_return": 0.025},
            ]
        )
        is False
    )


def test_min_bars_for_timeframe_relaxes_short_daily_adaptive_windows():
    short_start = datetime(2026, 2, 1, tzinfo=UTC)
    short_end = datetime(2026, 3, 8, tzinfo=UTC)
    long_start = datetime(2026, 1, 1, tzinfo=UTC)
    long_end = datetime(2026, 3, 15, tzinfo=UTC)

    assert (
        _min_bars_for_timeframe("1d", window_start=short_start, window_end_exclusive=short_end)
        == 28
    )
    assert (
        _min_bars_for_timeframe("1d", window_start=long_start, window_end_exclusive=long_end) == 45
    )
    assert (
        _min_bars_for_timeframe("4h", window_start=short_start, window_end_exclusive=short_end)
        == 180
    )


def test_min_bars_for_high_timeframes_is_relaxed_for_adaptive_windows():
    assert _min_bars_for_timeframe("1m") >= 360
    assert _min_bars_for_timeframe("4h") == 180
    assert _min_bars_for_timeframe("1d") == 45


def test_resolve_coverage_adaptive_windows_allocates_from_common_coverage(monkeypatch):
    def _stub_discover_symbol_coverage(**kwargs):
        _ = kwargs
        return (
            [
                {
                    "symbol": "BTC/USDT",
                    "coverage_start": "2025-12-01T00:00:00+00:00",
                    "coverage_end": "2026-03-08T23:59:00+00:00",
                },
                {
                    "symbol": "XAU/USDT",
                    "coverage_start": "2026-01-01T00:00:00+00:00",
                    "coverage_end": "2026-03-08T23:59:00+00:00",
                },
            ],
            ["BTC/USDT", "XAU/USDT"],
            datetime(2026, 3, 8, 23, 59, tzinfo=UTC),
        )

    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.discover_symbol_coverage",
        _stub_discover_symbol_coverage,
    )

    resolved = resolve_coverage_adaptive_windows(
        symbols=["BTC/USDT", "XAU/USDT"],
        root_path="data/market_parquet",
        exchange="binance",
        requested_oos_end_exclusive="2026-03-09",
        profile="metals",
    )

    assert resolved["common_start"] == datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        resolved["train_start"]
        < resolved["val_start"]
        < resolved["oos_start"]
        < resolved["requested_oos_end_exclusive"]
    )
    assert resolved["allocation_days"]["train"] >= 24
    assert resolved["allocation_days"]["val"] >= 14
    assert resolved["allocation_days"]["oos"] >= 14


def test_resolve_coverage_adaptive_windows_supports_shorter_metals_4h_history(monkeypatch):
    def _stub_discover_symbol_coverage(**kwargs):
        _ = kwargs
        return (
            [
                {
                    "symbol": "XPT/USDT",
                    "coverage_start": "2026-01-30T00:00:00+00:00",
                    "coverage_end": "2026-03-08T23:59:00+00:00",
                },
                {
                    "symbol": "XPD/USDT",
                    "coverage_start": "2026-01-30T00:00:00+00:00",
                    "coverage_end": "2026-03-08T23:59:00+00:00",
                },
            ],
            [],
            None,
        )

    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.discover_symbol_coverage",
        _stub_discover_symbol_coverage,
    )

    resolved = resolve_coverage_adaptive_windows(
        symbols=["XPT/USDT", "XPD/USDT"],
        root_path="data/market_parquet",
        exchange="binance",
        requested_oos_end_exclusive="2026-03-09",
        profile="metals_4h",
    )

    assert resolved["common_start"] == datetime(2026, 1, 30, tzinfo=UTC)
    assert (
        resolved["train_start"]
        < resolved["val_start"]
        < resolved["oos_start"]
        < resolved["requested_oos_end_exclusive"]
    )
    assert resolved["allocation_days"]["train"] >= 16
    assert resolved["allocation_days"]["val"] >= 8
    assert resolved["allocation_days"]["oos"] >= 10


def test_resolve_coverage_adaptive_windows_accepts_35_day_full_metals_overlap(monkeypatch):
    def _stub_discover_symbol_coverage(**kwargs):
        _ = kwargs
        return (
            [
                {
                    "symbol": "BTC/USDT",
                    "coverage_start": "2026-02-01T00:00:00+00:00",
                    "coverage_end": "2026-03-08T23:59:00+00:00",
                },
                {
                    "symbol": "XPT/USDT",
                    "coverage_start": "2026-02-01T00:00:00+00:00",
                    "coverage_end": "2026-03-08T23:59:00+00:00",
                },
                {
                    "symbol": "XPD/USDT",
                    "coverage_start": "2026-02-01T00:00:00+00:00",
                    "coverage_end": "2026-03-08T23:59:00+00:00",
                },
            ],
            [],
            None,
        )

    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.discover_symbol_coverage",
        _stub_discover_symbol_coverage,
    )

    resolved = resolve_coverage_adaptive_windows(
        symbols=["BTC/USDT", "XPT/USDT", "XPD/USDT"],
        root_path="data/market_parquet",
        exchange="binance",
        requested_oos_end_exclusive="2026-03-09",
        profile="metals",
    )
    assert resolved["total_days"] == 35
    assert resolved["allocation_days"]["train"] >= 16
    assert resolved["allocation_days"]["val"] >= 8
    assert resolved["allocation_days"]["oos"] >= 10


def test_run_exact_window_suite_uses_runtime_config_defaults_when_symbols_are_omitted(
    tmp_path,
    monkeypatch,
):
    cfg = textwrap.dedent(
        """
        trading:
          symbols: ["SOL/USDT", "ADA/USDT"]
          timeframe: "15m"
        storage:
          market_data_parquet_path: "var/data/suite_runtime_exact_window"
          market_data_exchange: "coinbase"
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")

    captured: dict[str, object] = {}

    def _stub_discover_symbol_coverage(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-capture")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.discover_symbol_coverage",
        _stub_discover_symbol_coverage,
    )

    with pytest.raises(RuntimeError, match="stop-after-capture"):
        run_exact_window_suite(
            output_dir=str(tmp_path / "out"),
            score_config={"minimum_trade_count": 0},
            timeframes=["4h"],
            symbols=None,
            chunk_days=7,
        )

    assert captured["symbols"] == ["SOL/USDT", "ADA/USDT"]
    assert captured["root_path"] == "var/data/suite_runtime_exact_window"
    assert captured["exchange"] == "coinbase"


def test_run_exact_window_suite_emits_progress_and_summary_counts(tmp_path, monkeypatch):
    class _Candidate:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def to_dict(self) -> dict[str, object]:
            return dict(self._payload)

    class _FrameStub:
        def is_empty(self) -> bool:
            return False

    timestamps = np.asarray(
        [
            np.datetime64("2026-01-01T00:00:00.000", "ms"),
            np.datetime64("2026-01-02T00:00:00.000", "ms"),
            np.datetime64("2026-01-03T00:00:00.000", "ms"),
            np.datetime64("2026-01-04T00:00:00.000", "ms"),
            np.datetime64("2026-01-05T00:00:00.000", "ms"),
            np.datetime64("2026-01-06T00:00:00.000", "ms"),
        ]
    )
    aligned = {
        "datetime": timestamps,
        "BTC/USDT:close": np.asarray([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], dtype=float),
        "benchmark_close": np.asarray([100.0, 100.5, 101.0, 101.5, 102.0, 102.5], dtype=float),
    }
    candidate = _Candidate(
        {
            "candidate_id": "cand-1",
            "name": "rolling_breakout_4h_stub",
            "strategy_class": "RollingBreakoutStrategy",
            "family": "trend",
            "timeframe": "4h",
            "strategy_timeframe": "4h",
            "symbols": ["BTC/USDT"],
            "params": {"lookback": 24},
        }
    )
    progress_events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.current_market_data_runtime_settings",
        lambda: {
            "market_data_parquet_path": "var/data/test_exact_window_suite",
            "market_data_exchange": "binance",
            "symbols": ["BTC/USDT"],
        },
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.discover_symbol_coverage",
        lambda **kwargs: (
            [
                {
                    "symbol": "BTC/USDT",
                    "full_start_coverage": True,
                    "coverage_end": "2026-01-06T00:00:00+00:00",
                }
            ],
            ["BTC/USDT"],
            datetime(2026, 1, 6, tzinfo=UTC),
        ),
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.build_binance_futures_candidates",
        lambda **kwargs: [candidate],
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite._strict_load_frame",
        lambda **kwargs: _FrameStub(),
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.strict_align_bundles",
        lambda **kwargs: aligned,
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite._monthly_btc_thresholds",
        lambda **kwargs: {"2026-01": {"btc_buy_hold_return": 0.0, "threshold": 0.0}},
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite._validation_score",
        lambda row, *, scoring_config: 1.0,
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.rr._resolve_score_config",
        lambda scope: {"minimum_trade_count": 0, "scope": scope},
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.rr._candidate_cost_rate",
        lambda candidate_row: 0.0,
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.rr._strategy_signal",
        lambda candidate_row, *, aligned, symbols: (
            np.asarray([0.0, 0.01, 0.015, 0.01, 0.02, 0.01], dtype=float),
            np.zeros(6, dtype=float),
            np.asarray([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
            {"stubbed": True},
        ),
    )

    def _stub_compute_metrics(returns, **kwargs):
        _ = kwargs
        return {
            "return": float(np.sum(np.asarray(returns, dtype=float))),
            "sharpe": 1.25,
            "trade_count": 6.0,
            "pbo": 0.1,
            "mdd": 0.05,
            "deflated_sharpe": 0.7,
        }

    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.rr._compute_metrics",
        _stub_compute_metrics,
    )
    monkeypatch.setattr(
        "lumina_quant.eval.exact_window_suite.rr._hurdle_fields",
        lambda train, val, oos, *, scoring_config: (
            {
                "train": {"pass": True},
                "val": {"pass": True},
                "oos": {"pass": True},
            },
            None,
            {},
        ),
    )

    summary = run_exact_window_suite(
        output_dir=str(tmp_path / "out"),
        score_config={"minimum_trade_count": 0},
        timeframes=["4h"],
        symbols=["BTC/USDT"],
        train_start="2026-01-01",
        val_start="2026-01-03",
        oos_start="2026-01-05",
        requested_oos_end_exclusive="2026-01-07",
        progress_callback=lambda event, payload: progress_events.append((event, dict(payload))),
    )

    assert [event for event, _ in progress_events] == [
        "suite_start",
        "timeframe_start",
        "candidate_evaluated",
        "timeframe_complete",
        "suite_complete",
    ]
    assert summary["candidate_count"] == 1
    assert summary["evaluated_count"] == 1
    assert summary["promoted_count"] == 1
    assert summary["candidate_pool_count"] == 1
    assert summary["eligible_symbols"] == ["BTC/USDT"]
    assert summary["best_per_strategy"][0]["candidate_id"] == "cand-1"
    assert summary["best_per_strategy"][0]["metadata"]["stubbed"] is True
    assert (tmp_path / "out" / "exact_window_suite_summary_latest.json").exists()
    assert progress_events[-1][1]["promoted_count"] == 1


def test_build_candidate_result_row_preserves_candidate_provenance_fields():
    row = _build_candidate_result_row(
        candidate={
            "candidate_id": "cand-1",
            "name": "pair_spread_4h_participation_btcusdt_xauusdt_1.6_0.35",
            "strategy_class": "PairSpreadZScoreStrategy",
            "family": "market_neutral",
            "params": {"entry_z": 1.6},
            "notes": "Mixed-asset residual pair.",
            "tags": ["market_neutral", "article_family:crypto-metal-residual-pairs"],
            "metadata": {
                "article_pipeline_family_ids": ["crypto-metal-residual-pairs"],
                "hypothesis_origin": "article_research_pipeline",
            },
        },
        timeframe="4h",
        symbols_for_candidate=["BTC/USDT", "XAU/USDT"],
        metrics={
            "train": {"return": 0.01},
            "val": {"return": 0.02},
            "oos": {"return": 0.03},
        },
        hurdles={"oos": {"pass": True}},
        hard_reject={},
        streams={"oos": [{"t": _ts("2026-03-01T00:00:00Z"), "v": 0.03}]},
        cost_rate=0.0005,
        runtime_metadata={"rss_guard_triggered": False},
    )

    assert row["notes"] == "Mixed-asset residual pair."
    assert row["tags"] == ["market_neutral", "article_family:crypto-metal-residual-pairs"]
    assert row["metadata"]["article_pipeline_family_ids"] == ["crypto-metal-residual-pairs"]
    assert row["metadata"]["hypothesis_origin"] == "article_research_pipeline"
    assert row["metadata"]["rss_guard_triggered"] is False
    assert row["metadata"]["cost_rate"] == 0.0005


def test_build_candidate_result_row_includes_committee_block():
    row = _build_candidate_result_row(
        candidate={
            "candidate_id": "cand-committee",
            "name": "pair_spread_4h_participation_btcusdt_xauusdt_1.6_0.35",
            "strategy_class": "PairSpreadZScoreStrategy",
            "family": "market_neutral",
            "params": {"entry_z": 1.6},
            "notes": "Mixed-asset residual pair.",
            "tags": ["market_neutral"],
            "metadata": {},
        },
        timeframe="4h",
        symbols_for_candidate=["BTC/USDT", "XAU/USDT"],
        metrics={
            "train": {"return": 0.01, "sharpe": 1.2},
            "val": {"return": 0.02, "sharpe": 1.8, "trade_count": 12, "pbo": 0.21, "mdd": 0.06},
            "oos": {"return": 0.03, "sharpe": 1.6, "trade_count": 11, "pbo": 0.18, "mdd": 0.10},
        },
        hurdles={"oos": {"pass": True}},
        hard_reject={},
        streams={
            "val": [{"t": _ts("2026-03-01T00:00:00Z"), "v": 0.02}],
            "oos": [{"t": _ts("2026-03-02T00:00:00Z"), "v": 0.03}],
        },
        cost_rate=0.0005,
        runtime_metadata={"rss_guard_triggered": False},
    )

    committee = dict(row.get("committee") or {})
    assert set(committee.keys()) == {
        "technical_score",
        "support_score",
        "regime_score",
        "robustness_score",
        "risk_veto",
        "risk_flags",
        "final_decision",
    }
    assert isinstance(committee["technical_score"], float)
    assert 0.0 <= float(committee["technical_score"]) <= 1.0
    assert 0.0 <= float(committee["support_score"]) <= 1.0
    assert 0.0 <= float(committee["regime_score"]) <= 1.0
    assert 0.0 <= float(committee["robustness_score"]) <= 1.0
    assert isinstance(committee["risk_flags"], list)
