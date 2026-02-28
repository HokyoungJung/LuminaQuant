from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_RUN_BACKTEST_PATH = Path(__file__).resolve().parents[1] / "run_backtest.py"
_RUN_BACKTEST_SPEC = importlib.util.spec_from_file_location("run_backtest_module", _RUN_BACKTEST_PATH)
if _RUN_BACKTEST_SPEC is None or _RUN_BACKTEST_SPEC.loader is None:
    raise RuntimeError(f"Failed to load run_backtest module from {_RUN_BACKTEST_PATH}")
run_backtest = importlib.util.module_from_spec(_RUN_BACKTEST_SPEC)
_RUN_BACKTEST_SPEC.loader.exec_module(run_backtest)


class _StubAuditStore:
    def __init__(self, _dsn):
        self.started = False
        self.ended = False

    def start_run(self, **_kwargs):
        self.started = True

    def end_run(self, *_args, **_kwargs):
        self.ended = True

    def close(self):
        return None


def test_create_audit_store_falls_back_when_postgres_dsn_missing(monkeypatch):
    class _MissingDsnAuditStore:
        def __init__(self, _dsn):
            raise ValueError("Postgres DSN is required. Set LQ_POSTGRES_DSN (or storage.postgres_dsn).")

    monkeypatch.setattr(run_backtest, "AuditStore", _MissingDsnAuditStore)
    store, enabled = run_backtest._create_audit_store()

    assert store is None
    assert enabled is False


def test_run_low_memory_uses_lightweight_backtest_flow(monkeypatch):
    captured: dict[str, object] = {}

    class _StubBacktest:
        def __init__(self, *args, **kwargs):
            _ = args
            captured["record_history"] = kwargs.get("record_history")
            captured["track_metrics"] = kwargs.get("track_metrics")
            captured["record_trades"] = kwargs.get("record_trades")
            self.config = SimpleNamespace(PERSIST_OUTPUT=True)
            self.portfolio = SimpleNamespace(
                current_holdings={"total": 1234.5},
                trade_count=0,
                output_summary_stats_fast=lambda: {
                    "status": "ok",
                    "sharpe": 1.0,
                    "cagr": 0.1,
                    "max_drawdown": 0.05,
                },
                create_equity_curve_dataframe=lambda: None,
                output_trade_log=lambda _path: None,
                save_equity_curve=lambda _path: None,
            )

        def simulate_trading(self, output=True, persist_output=None, verbose=True):
            captured["simulate_output"] = output
            captured["simulate_persist_output"] = persist_output
            captured["simulate_verbose"] = verbose
            return None

    monkeypatch.setattr(run_backtest, "AuditStore", _StubAuditStore)
    monkeypatch.setattr(run_backtest, "Backtest", _StubBacktest)
    monkeypatch.setattr(run_backtest, "_persist_backtest_audit_rows", lambda *_args, **_kwargs: {})

    run_backtest.run(
        data_source="csv",
        low_memory=True,
        persist_output=None,
        auto_collect_db=False,
        run_id="test-low-memory",
    )

    assert captured["record_history"] is False
    assert captured["track_metrics"] is True
    assert captured["record_trades"] is False
    assert captured["simulate_output"] is False
    assert captured["simulate_persist_output"] is None


def test_year_scale_low_memory_profile_default(monkeypatch):
    monkeypatch.delenv("LQ_BACKTEST_LOW_MEMORY", raising=False)
    monkeypatch.delenv("LQ_BACKTEST_PERSIST_OUTPUT", raising=False)

    profile = run_backtest._resolve_execution_profile(
        low_memory=None,
        persist_output=None,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 2, 15),
    )

    assert profile["low_memory"] is True
    assert profile["record_history"] is False
    assert profile["record_trades"] is False


def test_streaming_equity_logging_no_equity_curve_materialization():
    logged: list[dict[str, object]] = []

    class _Audit:
        def log_equity(self, run_id, **kwargs):
            logged.append({"run_id": run_id, **kwargs})

        def log_fill(self, *_args, **_kwargs):
            return None

    class _Portfolio:
        def __init__(self):
            self._equity_points = [(1735689600.0, 1000.0), (1735689660.0, 1005.0)]
            self.trades = []

        def create_equity_curve_dataframe(self):
            raise AssertionError("equity curve materialization should not be called")

    backtest = SimpleNamespace(portfolio=_Portfolio())
    counts = run_backtest._persist_backtest_audit_rows(
        _Audit(),
        "run-low-memory",
        backtest,
        low_memory=True,
    )

    assert counts["equity_rows"] == 2
    assert counts["fill_rows"] == 0
    assert len(logged) == 2
