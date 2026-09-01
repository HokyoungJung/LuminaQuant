from __future__ import annotations

from lumina_quant.cli import backtest as backtest_cli


def test_backtest_main_reads_env_backed_defaults_at_call_time(monkeypatch):
    captured: dict[str, object] = {}

    def _stub_run(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setenv("LQ_DATA_MODE", "legacy")
    monkeypatch.setenv("LQ_BACKTEST_MODE", "legacy_batch")
    monkeypatch.setenv("LQ_BASE_TIMEFRAME", "5m")
    monkeypatch.setenv("LQ_AUTO_COLLECT_DB", "1")
    monkeypatch.setattr(backtest_cli, "run", _stub_run)

    assert backtest_cli.main([]) == 0
    assert captured["data_mode"] == "legacy"
    assert captured["backtest_mode"] == "legacy_batch"
    assert captured["base_timeframe"] == "5m"
    assert captured["auto_collect_db"] is True
