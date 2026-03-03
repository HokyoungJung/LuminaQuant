from __future__ import annotations

from datetime import datetime

from lumina_quant.cli import backtest as run_backtest


def test_year_scale_low_memory_profile_default(monkeypatch):
    monkeypatch.delenv("LQ_BACKTEST_LOW_MEMORY", raising=False)
    monkeypatch.delenv("LQ_BACKTEST_PERSIST_OUTPUT", raising=False)

    profile = run_backtest._resolve_execution_profile(
        low_memory=None,
        persist_output=None,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 31),
    )

    assert profile["low_memory"] is True
    assert profile["record_history"] is False
    assert profile["record_trades"] is False
    assert profile["persist_output"] is False
    assert profile["track_metrics"] is True
