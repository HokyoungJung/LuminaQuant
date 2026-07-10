from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from lumina_quant.dashboard import overview_service


def _contract() -> SimpleNamespace:
    return SimpleNamespace(launch_mode="next", python_backend="python")


def test_empty_overview_payload_tracks_reason() -> None:
    payload = overview_service.empty_overview_payload(contract=_contract(), reason="missing_dsn")

    assert payload["source"]["status"] == "missing_dsn"
    assert payload["recent_runs"] == []
    assert payload["workflow_jobs"] == []
    assert payload["equity_curve"] == []
    assert payload["equity_window"] == {
        "start": None,
        "end": None,
        "points_total": 0,
        "points_returned": 0,
        "truncated": False,
    }


def test_build_overview_payload_from_frames_exposes_recent_runs_and_curves() -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "run-2",
                "mode": "backtest",
                "status": "COMPLETED",
                "metadata": {"strategy": "RsiStrategy"},
                "strategy": "RsiStrategy",
                "started_at": "2026-03-02T00:00:00Z",
            },
            {
                "run_id": "run-1",
                "mode": "live",
                "status": "RUNNING",
                "metadata": {"strategy": "Momentum"},
                "strategy": "Momentum",
                "started_at": "2026-03-01T00:00:00Z",
            },
        ]
    )
    equity = pd.DataFrame(
        [
            {"datetime": "2026-03-02T00:00:00Z", "total": 1000.0},
            {"datetime": "2026-03-03T00:00:00Z", "total": 1100.0},
            {"datetime": "2026-03-04T00:00:00Z", "total": 1050.0},
        ]
    )

    payload = overview_service.build_overview_payload_from_frames(
        contract=_contract(),
        runs_frame=runs,
        equity_frame=equity,
    )

    assert payload["source"]["run_id"] == "run-2"
    # workflow_jobs is contract-required on the success shape too.
    assert payload["workflow_jobs"] == []
    assert payload["recent_runs"][0]["run_id"] == "run-2"
    assert payload["recent_runs"][1]["status"] == "RUNNING"
    assert payload["summary_metrics"][4]["value"] == 1000.0
    assert payload["summary_metrics"][5]["value"] == 1050.0
    assert payload["performance_metrics"]["sharpe_ratio"] != 0.0
    assert payload["performance_metrics"]["max_drawdown"] > 0.0
    assert payload["equity_curve"][-1]["equity"] == 1050.0
    assert payload["drawdown_curve"][-1]["drawdown"] < 0.0
    assert payload["equity_window"] == {
        "start": "2026-03-02T00:00:00+00:00",
        "end": "2026-03-04T00:00:00+00:00",
        "points_total": 3,
        "points_returned": 3,
        "truncated": False,
    }


def test_build_overview_payload_selected_run_row_overrides_latest() -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "run-2",
                "mode": "backtest",
                "status": "COMPLETED",
                "metadata": {"strategy": "RsiStrategy"},
                "strategy": "RsiStrategy",
                "started_at": "2026-03-02T00:00:00Z",
            },
            {
                "run_id": "run-1",
                "mode": "live",
                "status": "RUNNING",
                "metadata": {"strategy": "Momentum"},
                "strategy": "Momentum",
                "started_at": "2026-03-01T00:00:00Z",
            },
        ]
    )
    equity = pd.DataFrame(
        [
            {"datetime": "2026-03-01T00:00:00Z", "total": 1000.0},
            {"datetime": "2026-03-01T01:00:00Z", "total": 1010.0},
        ]
    )

    payload = overview_service.build_overview_payload_from_frames(
        contract=_contract(),
        runs_frame=runs,
        equity_frame=equity,
        selected_run_row=runs.iloc[1],
    )

    assert payload["source"]["run_id"] == "run-1"
    assert payload["summary_metrics"][0]["value"] == "run-1"
    assert payload["summary_metrics"][3]["value"] == "Momentum"
    # recent_runs stays the latest-N list regardless of the selected run.
    assert [run["run_id"] for run in payload["recent_runs"]] == ["run-2", "run-1"]


def test_downsample_curve_indices_keeps_first_and_last_within_limit() -> None:
    indices = overview_service.downsample_curve_indices([float(i) for i in range(1000)], 10)

    assert len(indices) <= 10
    assert indices[0] == 0
    assert indices[-1] == 999
    assert indices == sorted(set(indices))
    assert overview_service.downsample_curve_indices([1.0] * 5, 10) == [0, 1, 2, 3, 4]
    assert overview_service.downsample_curve_indices([], 10) == []
    # A cap that only fits the endpoints returns exactly the endpoints.
    assert overview_service.downsample_curve_indices([float(i) for i in range(50)], 2) == [0, 49]


def test_downsample_curve_indices_preserves_extreme_spikes() -> None:
    """Min-max bucket downsampling: a single extreme inside a bucket survives.

    Uniform linspace sampling used to drop mid-bucket spikes, letting the
    rendered drawdown curve contradict the full-series Max Drawdown tile.
    """
    values = [1000.0 + index for index in range(10_000)]
    values[4321] = 1.0  # deep trough far from every uniform stride
    values[7777] = 1_000_000.0  # extreme peak inside another bucket

    indices = overview_service.downsample_curve_indices(values, 60)

    assert len(indices) <= 60
    assert indices[0] == 0
    assert indices[-1] == 9_999
    assert 4321 in indices
    assert 7777 in indices
    sampled = [values[index] for index in indices]
    assert min(sampled) == 1.0
    assert max(sampled) == 1_000_000.0


def test_curves_are_downsampled_but_metrics_use_full_series() -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "run-2",
                "mode": "backtest",
                "status": "COMPLETED",
                "metadata": {"strategy": "RsiStrategy"},
                "strategy": "RsiStrategy",
                "started_at": "2026-03-02T00:00:00Z",
            }
        ]
    )
    timestamps = pd.date_range("2026-03-01", periods=500, freq="1h", tz="UTC")
    totals = [1000.0 + index for index in range(500)]
    totals[100] = 900.0  # drawdown trough far outside any 50-point tail
    equity = pd.DataFrame({"datetime": timestamps, "total": totals})

    payload = overview_service.build_overview_payload_from_frames(
        contract=_contract(),
        runs_frame=runs,
        equity_frame=equity,
        curve_point_limit=50,
    )

    assert len(payload["equity_curve"]) <= 50
    assert payload["equity_curve"][0]["timestamp"] == timestamps[0].isoformat()
    assert payload["equity_curve"][-1]["timestamp"] == timestamps[-1].isoformat()
    assert payload["equity_curve"][0]["equity"] == 1000.0
    assert payload["equity_curve"][-1]["equity"] == 1499.0
    assert payload["equity_window"]["points_total"] == 500
    assert payload["equity_window"]["points_returned"] == len(payload["equity_curve"])
    assert payload["equity_window"]["truncated"] is False
    # Max drawdown reflects the full-run trough (1099 -> 900), not a tail.
    assert payload["performance_metrics"]["max_drawdown"] > 0.15
    # Min-max downsampling keeps the trough in the rendered curves, so the
    # chart cannot contradict the Max Drawdown tile.
    assert min(point["equity"] for point in payload["equity_curve"]) == 900.0
    assert min(point["drawdown"] for point in payload["drawdown_curve"]) < -0.15


def test_infer_periods_per_year_from_hourly_cadence() -> None:
    hourly = pd.Series(pd.date_range("2026-03-01", periods=48, freq="1h", tz="UTC"))

    periods = overview_service.infer_periods_per_year(hourly)

    assert abs(periods - 365.25 * 24.0) < 1.0


def test_infer_periods_per_year_falls_back_and_clamps() -> None:
    unusable = pd.Series(["not-a-date", None])
    assert (
        overview_service.infer_periods_per_year(unusable)
        == overview_service.FALLBACK_PERIODS_PER_YEAR
    )

    sub_second = pd.Series(pd.date_range("2026-03-01", periods=10, freq="100ms", tz="UTC"))
    assert (
        overview_service.infer_periods_per_year(sub_second) == overview_service.MAX_PERIODS_PER_YEAR
    )


def test_annualization_uses_equity_cadence_not_252() -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "run-2",
                "mode": "backtest",
                "status": "COMPLETED",
                "metadata": {"strategy": "RsiStrategy"},
                "strategy": "RsiStrategy",
                "started_at": "2026-03-02T00:00:00Z",
            }
        ]
    )
    timestamps = pd.date_range("2026-03-01", periods=100, freq="1h", tz="UTC")
    totals = [1000.0 * (1.001 ** ((index % 2) * 2 - 1)) ** index for index in range(100)]
    hourly = pd.DataFrame({"datetime": timestamps, "total": totals})
    daily = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-03-01", periods=100, freq="1D", tz="UTC"),
            "total": totals,
        }
    )

    hourly_payload = overview_service.build_overview_payload_from_frames(
        contract=_contract(), runs_frame=runs, equity_frame=hourly
    )
    daily_payload = overview_service.build_overview_payload_from_frames(
        contract=_contract(), runs_frame=runs, equity_frame=daily
    )

    hourly_vol = hourly_payload["performance_metrics"]["annualized_volatility"]
    daily_vol = daily_payload["performance_metrics"]["annualized_volatility"]
    # Same per-bar returns annualized at hourly cadence must scale ~sqrt(24).
    assert hourly_vol / daily_vol == pytest.approx(24.0**0.5, rel=1e-3)


def test_load_overview_payload_short_circuits_for_blank_dsn() -> None:
    payload = overview_service.load_overview_payload(contract=_contract(), dsn="")

    assert payload["source"]["status"] == "missing_dsn"
    assert payload["performance_metrics"] == {}


class _FakeCursor:
    """Replays (rows, description) pairs in query order."""

    def __init__(self, results: list[tuple[list[tuple], list[tuple]]]):
        self._results = results
        self._rows: list[tuple] = []
        self.description: list[tuple] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple = ()) -> None:
        self._rows, self.description = self._results.pop(0)

    def fetchall(self) -> list[tuple]:
        return self._rows


class _FakeConnection:
    def __init__(self, results: list[tuple[list[tuple], list[tuple]]]):
        self._results = results
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._results)

    def close(self) -> None:
        self.closed = True


def test_load_overview_payload_success_emits_empty_workflow_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs present + zero workflow jobs must still emit workflow_jobs == [].

    The Next runtime reads ``payload.workflow_jobs`` unguarded on every 200
    response, so the key is contract-required even when the jobs table is
    empty (P0: landing page crashed on backtest-only databases).
    """
    run_columns = [
        ("run_id",),
        ("mode",),
        ("started_at",),
        ("ended_at",),
        ("status",),
        ("metadata",),
        ("strategy",),
    ]
    run_rows = [
        (
            "run-123",
            "backtest",
            "2026-03-21T00:00:00Z",
            None,
            "COMPLETED",
            '{"strategy": "RsiStrategy"}',
            "RsiStrategy",
        )
    ]
    equity_columns = [("datetime",), ("total",)]
    equity_rows = [
        ("2026-03-21T00:00:00Z", 1000.0),
        ("2026-03-22T00:00:00Z", 1105.0),
    ]
    conn = _FakeConnection([(run_rows, run_columns), (equity_rows, equity_columns)])
    monkeypatch.setattr(overview_service, "_connect_postgres", lambda dsn: conn)
    monkeypatch.setattr(overview_service, "load_recent_workflow_jobs", lambda conn, limit: [])

    payload = overview_service.load_overview_payload(contract=_contract(), dsn="postgres://x")

    assert payload["source"]["status"] == "ok"
    assert "workflow_jobs" in payload
    assert payload["workflow_jobs"] == []
    assert conn.closed is True
