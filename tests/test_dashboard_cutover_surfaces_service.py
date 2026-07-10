from __future__ import annotations

import pandas as pd
import pytest

from lumina_quant.dashboard import cutover_surfaces_service
from lumina_quant.dashboard.cutover_surfaces_service import (
    RECENT_BAR_LIMIT,
    build_execution_analytics_payload,
    build_market_data_payload,
    build_optimization_insights_payload,
    build_performance_price_payload,
    build_raw_data_payload,
    build_report_export_payload,
    compute_trade_analytics,
    load_execution_analytics_payload,
    load_market_data_payload,
    load_optimization_insights_payload,
    load_performance_price_payload,
    load_raw_data_payload,
    load_report_export_payload,
)


def _overview_payload() -> dict[str, object]:
    return {
        "source": {
            "status": "ok",
            "run_id": "run-123",
            "mode": "next",
            "backend": "python",
        },
        "summary_metrics": [
            {"key": "latest_equity", "label": "Latest Equity", "value": 1105.0},
            {"key": "total_return", "label": "Total Return", "value": 0.105},
        ],
        "performance_metrics": {
            "cagr": 0.22,
            "max_drawdown": -0.08,
        },
        "equity_curve": [
            {"timestamp": "2026-03-21T00:00:00+00:00", "equity": 1000.0},
            {"timestamp": "2026-03-22T00:00:00+00:00", "equity": 1105.0},
        ],
        "drawdown_curve": [
            {"timestamp": "2026-03-21T00:00:00+00:00", "drawdown": 0.0},
            {"timestamp": "2026-03-22T00:00:00+00:00", "drawdown": -0.08},
        ],
    }


def test_compute_trade_analytics_tracks_realized_pnl_and_position() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 2.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": "2026-03-21T00:05:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 0.5,
            },
        ]
    )

    analytics = compute_trade_analytics(fills)

    assert analytics["position_after"].tolist() == [2.0, 1.0]
    assert analytics["realized_pnl"].tolist() == [0.0, 9.5]
    close_side = analytics["close_side"].tolist()
    assert pd.isna(close_side[0])
    assert close_side[1] == "LONG"
    # realized_return_pct travels as a raw fraction (9.5 pnl / 110 basis).
    assert analytics["realized_return_pct"].iloc[1] == pytest.approx(9.5 / 110.0)


def test_build_performance_price_payload_exposes_benchmark_and_trade_markers() -> None:
    metrics = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "total": 1000.0,
                "benchmark_price": 100.0,
                "funding": 0.0,
            },
            {
                "datetime": "2026-03-22T00:00:00Z",
                "total": 1105.0,
                "benchmark_price": 108.0,
                "funding": 1.5,
            },
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 2.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": "2026-03-21T00:05:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 0.5,
            },
        ]
    )

    payload = build_performance_price_payload(
        overview_payload=_overview_payload(),
        metrics_frame=metrics,
        fills_frame=fills,
    )

    assert payload["status"] == "ok"
    assert payload["benchmark_curve"][-1]["price"] == 108.0
    assert payload["funding_curve"][-1]["funding"] == 1.5
    assert payload["trade_markers"][-1]["realized_pnl"] == 9.5


def test_build_execution_analytics_payload_summarizes_fills_orders_and_streaks() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 2.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": "2026-03-21T00:05:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 0.5,
            },
            {
                "datetime": "2026-03-21T00:10:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 95.0,
                "commission": 0.5,
            },
        ]
    )
    orders = pd.DataFrame(
        [
            {"status": "FILLED"},
            {"status": "CANCELLED"},
            {"status": "FILLED"},
        ]
    )

    payload = build_execution_analytics_payload(
        run_id="run-123",
        fills_frame=fills,
        orders_frame=orders,
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["buy_fills"] == 1
    assert payload["summary"]["sell_fills"] == 2
    assert payload["summary"]["closed_trade_count"] == 2
    assert payload["summary"]["win_streak_max"] == 1
    assert payload["summary"]["loss_streak_max"] == 1
    assert payload["direction_breakdown"][0] == {
        "direction": "Long",
        "closed_trades": 2,
        "win_rate": 0.5,
    }
    assert payload["order_status"][0]["count"] == 2


def test_build_report_export_payload_generates_json_and_markdown() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 2.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": "2026-03-21T00:05:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 0.5,
            },
        ]
    )

    payload = build_report_export_payload(
        run_row={
            "run_id": "run-123",
            "strategy": "RsiStrategy",
            "mode": "backtest",
            "status": "COMPLETED",
            "started_at": "2026-03-21T00:00:00Z",
        },
        overview_payload=_overview_payload(),
        fills_frame=fills,
        risk_frame=pd.DataFrame([{"event_time": "2026-03-22T00:00:00Z"}]),
        heartbeats_frame=pd.DataFrame([{"heartbeat_time": "2026-03-22T00:00:00Z"}]),
    )

    assert payload["status"] == "ok"
    assert payload["json_report"]["closed_trade_count"] == 1
    assert payload["json_report"]["risk_event_count"] == 1
    assert "Default Launcher: next" in payload["markdown_report"]
    assert payload["filenames"]["json"].endswith("-dashboard-report.json")
    assert "Market Data route is available in Next.js." in payload["cutover_gate"]["evidence"]


def test_build_market_data_payload_exposes_context_and_recent_bars() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 1.0,
                "price": 100.0,
                "commission": 0.0,
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "open": 99.0,
                "high": 101.0,
                "low": 98.5,
                "close": 100.0,
                "volume": 12.0,
            },
            {
                "datetime": "2026-03-21T00:01:00Z",
                "open": 100.0,
                "high": 102.0,
                "low": 99.5,
                "close": 101.5,
                "volume": 18.0,
            },
        ]
    )

    payload = build_market_data_payload(
        run_row={"run_id": "run-123", "strategy": "RsiStrategy"},
        fills_frame=fills,
        market_frame=market,
    )

    assert payload["status"] == "ok"
    assert payload["market_context"]["symbol"] == "BTC/USDT"
    assert payload["symbols"] == ["BTC/USDT"]
    assert payload["summary_metrics"][0]["value"] == 2
    assert payload["recent_bars"][-1]["close"] == 101.5
    assert payload["bar_window"] == {
        "start": "2026-03-21T00:00:00+00:00",
        "end": "2026-03-21T00:01:00+00:00",
        "bar_count": 2,
    }
    # Price Change % is a raw fraction: (101.5 - 100.0) / 100.0.
    price_change = next(
        metric for metric in payload["summary_metrics"] if metric["key"] == "price_change_pct"
    )
    assert price_change["label"] == "Price Change %"
    assert price_change["value"] == pytest.approx(0.015)


def _synthetic_market_frame(bars: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-03-21", periods=bars, freq="1h", tz="UTC")
    closes = [100.0 + 0.5 * index for index in range(bars)]
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": [close - 0.2 for close in closes],
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
            "close": closes,
            "volume": [10.0] * bars,
        }
    )


def test_build_market_data_payload_computes_real_indicators() -> None:
    market = _synthetic_market_frame(60)

    payload = build_market_data_payload(
        run_row={"run_id": "run-123", "strategy": "RsiStrategy"},
        fills_frame=pd.DataFrame(),
        market_frame=market,
    )

    indicators = {metric["key"]: metric["value"] for metric in payload["indicator_summary"]}
    # Monotonically rising closes pin Wilder RSI at 100.
    assert indicators["rsi_14"] == pytest.approx(100.0)
    # TR is constant at 2.0 (high-low dominates a 0.5 close step).
    assert indicators["atr_14"] == pytest.approx(2.0, rel=1e-2)
    assert 0.0 < indicators["atr_pct"] < 1.0
    assert indicators["atr_pct"] == pytest.approx(indicators["atr_14"] / 129.5, rel=1e-2)
    assert indicators["realized_vol_30"] is not None
    assert indicators["realized_vol_30"] > 0.0
    assert indicators["last_close"] == pytest.approx(129.5)
    # Last close sits 1.0 below the 20-bar high and above the 20-bar low.
    assert indicators["dist_from_20bar_high"] == pytest.approx(129.5 / 130.5 - 1.0, abs=1e-6)
    assert indicators["dist_from_20bar_high"] < 0.0
    assert indicators["dist_from_20bar_low"] > 0.0
    assert indicators["window_volume"] == pytest.approx(600.0)
    assert "strategy" in indicators
    assert "price_range" in indicators
    assert "timeframe_clamped" in indicators
    assert "indicator_mode" not in indicators


def test_build_market_data_payload_recent_bars_cover_metric_window() -> None:
    market = _synthetic_market_frame(RECENT_BAR_LIMIT + 10)

    payload = build_market_data_payload(
        run_row={"run_id": "run-123", "strategy": "RsiStrategy"},
        fills_frame=pd.DataFrame(),
        market_frame=market,
    )

    assert len(payload["recent_bars"]) == RECENT_BAR_LIMIT
    assert payload["bar_window"]["bar_count"] == RECENT_BAR_LIMIT + 10
    assert payload["bar_window"]["start"] == market["datetime"].iloc[0].isoformat()
    assert payload["bar_window"]["end"] == market["datetime"].iloc[-1].isoformat()
    assert payload["recent_bars"][-1]["timestamp"] == market["datetime"].iloc[-1].isoformat()


def test_build_market_data_payload_honors_requested_symbol() -> None:
    fills = pd.DataFrame(
        [
            {"datetime": "2026-03-21T00:00:00Z", "symbol": "ETH/USDT"},
            {"datetime": "2026-03-21T00:01:00Z", "symbol": "BTC/USDT"},
        ]
    )

    payload = build_market_data_payload(
        run_row={"run_id": "run-123", "strategy": "RsiStrategy"},
        fills_frame=fills,
        market_frame=_synthetic_market_frame(4),
        symbol="ETH/USDT",
    )

    assert payload["market_context"]["symbol"] == "ETH/USDT"
    assert payload["symbols"] == ["BTC/USDT", "ETH/USDT"]


def test_select_run_row_prefers_requested_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit run_id lookup must be an exact SQL match, not a capped scan.

    A widened recency-limited frame used to cap the lookup at the newest 300
    runs, returning false run_not_found for older bookmarked runs. The
    fallback now issues a parameterized ``WHERE run_id = %s`` query, so any
    run in the table resolves and only genuinely missing ids return None.
    """
    runs = pd.DataFrame(
        [
            {"run_id": "run-2", "mode": "backtest", "status": "COMPLETED"},
            {"run_id": "run-1", "mode": "live", "status": "RUNNING"},
        ]
    )
    columns = [
        ("run_id",),
        ("mode",),
        ("started_at",),
        ("ended_at",),
        ("status",),
        ("metadata",),
        ("strategy",),
    ]
    store = {
        "run-0": (
            "run-0",
            "backtest",
            "2025-01-01T00:00:00Z",
            None,
            "COMPLETED",
            "{}",
            "RsiStrategy",
        )
    }
    executed: list[tuple[str, tuple]] = []

    class _Cursor:
        def __init__(self) -> None:
            self._rows: list[tuple] = []
            self.description = columns

        def __enter__(self) -> _Cursor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def execute(self, query: str, params: tuple) -> None:
            executed.append((query, params))
            row = store.get(str(params[0]))
            self._rows = [row] if row is not None else []

        def fetchall(self) -> list[tuple]:
            return self._rows

    class _Conn:
        def cursor(self) -> _Cursor:
            return _Cursor()

        def close(self) -> None:
            return None

    monkeypatch.setattr(cutover_surfaces_service, "_connect_postgres", lambda dsn: _Conn())

    assert cutover_surfaces_service._select_run_row("dsn", runs, None)["run_id"] == "run-2"
    assert cutover_surfaces_service._select_run_row("dsn", runs, "run-1")["run_id"] == "run-1"
    # In-frame hits never touch the store.
    assert executed == []
    # A run far older than any selector/lookup window resolves via exact SQL.
    resolved = cutover_surfaces_service._select_run_row("dsn", runs, "run-0")
    assert resolved["run_id"] == "run-0"
    assert resolved["strategy"] == "RsiStrategy"
    assert "WHERE run_id = %s" in executed[-1][0]
    assert executed[-1][1] == ("run-0",)
    # Genuinely missing ids still surface run_not_found.
    assert cutover_surfaces_service._select_run_row("dsn", runs, "missing") is None
    assert executed[-1][1] == ("missing",)


def test_build_optimization_insights_payload_exposes_stage_breakdown_and_best_candidate() -> None:
    # cagr/mdd land from Postgres as percent-magnitude TEXT ('12.3456') or
    # literal 'N/A' (cli/optimize.py); the payload must expose number|null
    # RAW FRACTIONS per the cross-side contract.
    optimization = pd.DataFrame(
        [
            {
                "created_at": "2026-03-21T00:00:00Z",
                "run_id": "run-123",
                "stage": "explore",
                "sharpe": 1.1,
                "train_sharpe": 1.3,
                "robustness_score": 0.7,
                "cagr": "N/A",
                "mdd": "N/A",
                "params": {"window": 10},
            },
            {
                "created_at": "2026-03-21T00:05:00Z",
                "run_id": "run-123",
                "stage": "promote",
                "sharpe": 1.4,
                "train_sharpe": 1.5,
                "robustness_score": 0.8,
                "cagr": "24.0000",
                "mdd": "-8.1234",
                "params": {"window": 20},
            },
        ]
    )

    payload = build_optimization_insights_payload(
        run_id="run-123",
        optimization_frame=optimization,
    )

    assert payload["status"] == "ok"
    assert payload["summary_metrics"][0]["value"] == 2
    assert payload["stage_breakdown"][0]["stage"] == "explore"
    assert payload["best_candidate"]["stage"] == "promote"
    assert payload["top_candidates"][0]["params"] == {"window": 20}
    # Percent-magnitude text coerced to numeric raw fractions.
    assert payload["best_candidate"]["cagr"] == pytest.approx(0.24)
    assert payload["best_candidate"]["mdd"] == pytest.approx(-0.081234)
    assert payload["top_candidates"][0]["cagr"] == pytest.approx(0.24)
    assert payload["top_candidates"][0]["mdd"] == pytest.approx(-0.081234)
    # Literal 'N/A' maps to None, not a passthrough string.
    assert payload["top_candidates"][1]["cagr"] is None
    assert payload["top_candidates"][1]["mdd"] is None


def test_build_raw_data_payload_exposes_frame_summaries_and_previews() -> None:
    payload = build_raw_data_payload(
        run_id="run-123",
        context={"source": "postgres", "market": "BTC/USDT 1m (binance)"},
        frames=[
            ("Runs", pd.DataFrame([{"run_id": "run-123", "status": "COMPLETED"}])),
            ("Orders", pd.DataFrame([{"status": "FILLED", "quantity": 1.0}])),
        ],
    )

    assert payload["status"] == "ok"
    assert payload["frame_summaries"] == [
        {"label": "Runs", "rows": 1, "columns": 2},
        {"label": "Orders", "rows": 1, "columns": 2},
    ]
    assert payload["previews"][0]["rows"][0]["run_id"] == "run-123"


def test_load_performance_price_payload_computes_metrics_from_full_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Performance metrics must come from the FULL run, not the curve window.

    Mirrors the overview P0 fix: the loader asks the store for up to
    FULL_EQUITY_ROW_CAP rows, computes metrics/summary from all of them, and
    only downsamples the payload curves to point_limit.
    """
    from lumina_quant.dashboard.overview_service import FULL_EQUITY_ROW_CAP

    total_rows = 1_000
    point_limit = 240
    timestamps = pd.date_range("2026-01-01", periods=total_rows, freq="1h", tz="UTC")
    metrics = pd.DataFrame(
        {
            "datetime": timestamps,
            "total": [1000.0 + index for index in range(total_rows)],
            "benchmark_price": [100.0 + index for index in range(total_rows)],
            "funding": [0.0] * total_rows,
        }
    )
    runs = pd.DataFrame(
        [{"run_id": "run-123", "mode": "backtest", "status": "COMPLETED", "metadata": "{}"}]
    )
    requested_point_limits: list[int] = []

    class _Contract:
        launch_mode = "next"
        python_backend = "python"

    monkeypatch.setattr(
        cutover_surfaces_service, "resolve_dashboard_postgres_dsn", lambda dsn=None: "postgres://x"
    )
    monkeypatch.setattr(cutover_surfaces_service, "_dashboard_contract", lambda: _Contract())
    monkeypatch.setattr(cutover_surfaces_service, "_load_runs", lambda dsn, *, run_limit: runs)

    def _fake_load_metrics(dsn: str, run_id: str, *, point_limit: int) -> pd.DataFrame:
        requested_point_limits.append(point_limit)
        return metrics.copy()

    monkeypatch.setattr(cutover_surfaces_service, "_load_metrics", _fake_load_metrics)
    monkeypatch.setattr(
        cutover_surfaces_service,
        "_load_fills",
        lambda dsn, run_id, *, fill_limit: pd.DataFrame(),
    )

    payload = load_performance_price_payload(point_limit=point_limit)

    assert payload["status"] == "ok"
    # The store was asked for the full series, not the curve window.
    assert requested_point_limits == [FULL_EQUITY_ROW_CAP]
    # Metrics describe the full run: total return uses first/last of all rows.
    summary = {metric["key"]: metric["value"] for metric in payload["summary_metrics"]}
    assert summary["equity_points"] == total_rows
    assert summary["total_return"] == pytest.approx((1999.0 - 1000.0) / 1000.0)
    assert payload["performance_metrics"]["cagr"] != 0.0
    # Curves stay bounded by point_limit while spanning the whole run.
    assert len(payload["equity_curve"]) <= point_limit
    assert payload["equity_curve"][0]["equity"] == 1000.0
    assert payload["equity_curve"][-1]["equity"] == 1999.0
    assert len(payload["benchmark_curve"]) <= point_limit
    assert payload["benchmark_curve"][0]["price"] == 100.0
    assert payload["benchmark_curve"][-1]["price"] == 100.0 + (total_rows - 1)
    # equity_window discloses the true metric window.
    window = payload["equity_window"]
    assert window["points_total"] == total_rows
    assert window["points_returned"] == len(payload["equity_curve"])
    assert window["truncated"] is False
    assert window["start"].startswith("2026-01-01T00:00:00")


def test_cutover_surface_loaders_short_circuit_on_blank_dsn() -> None:
    assert load_performance_price_payload(dsn="")["status"] == "missing_dsn"
    assert load_execution_analytics_payload(dsn="")["status"] == "missing_dsn"
    assert load_market_data_payload(dsn="")["status"] == "missing_dsn"
    assert load_optimization_insights_payload(dsn="")["status"] == "missing_dsn"
    assert load_raw_data_payload(dsn="")["status"] == "missing_dsn"
    assert load_report_export_payload(dsn="")["status"] == "missing_dsn"


def test_run_selector_loaders_expose_runs_field_even_when_empty() -> None:
    for loader in (
        load_performance_price_payload,
        load_execution_analytics_payload,
        load_market_data_payload,
        load_raw_data_payload,
        load_report_export_payload,
    ):
        payload = loader(dsn="")
        assert payload["runs"] == []


def test_run_selector_loaders_accept_run_id_kwarg() -> None:
    # run_id short-circuits with the rest of the payload on a blank DSN.
    assert load_performance_price_payload(dsn="", run_id="run-1")["status"] == "missing_dsn"
    assert load_execution_analytics_payload(dsn="", run_id="run-1")["status"] == "missing_dsn"
    assert load_market_data_payload(dsn="", run_id="run-1", symbol="BTC/USDT")["status"] == (
        "missing_dsn"
    )
    assert load_raw_data_payload(dsn="", run_id="run-1")["status"] == "missing_dsn"
    assert load_report_export_payload(dsn="", run_id="run-1")["status"] == "missing_dsn"
