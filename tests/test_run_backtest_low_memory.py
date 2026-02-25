from __future__ import annotations

import importlib.util
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
