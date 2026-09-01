from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_binance_aggtrades_raw.py"
_SPEC = importlib.util.spec_from_file_location("collect_script_module", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load collector script module from {_SCRIPT_PATH}")
collector_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collector_script)


def test_collector_periodic_loop_uses_configured_poll_interval(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    def _collect(**kwargs):
        calls.append(dict(kwargs))
        idx = len(calls)
        return {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange_id"],
            "start_cursor_ms": 0,
            "end_cursor_ms": 0,
            "fetched_rows": 1,
            "upserted_rows": 1,
            "last_timestamp_ms": 1_700_000_000_000 + idx,
            "last_trade_id": idx,
            "last_agg_trade_id": idx,
        }

    monkeypatch.setattr(collector_script, "collect_binance_aggtrades_raw", _collect)

    cycles = collector_script.run_collector_periodic_loop(
        db_path=str(tmp_path / "market_parquet"),
        exchange_id="binance",
        symbols=["BTC/USDT"],
        since_ms=None,
        until_ms=None,
        bootstrap_lookback_hours=12,
        limit=1000,
        max_batches=10,
        retries=0,
        base_wait_sec=0.1,
        periodic_enabled=True,
        poll_seconds=3,
        max_cycles=2,
        sleep_fn=lambda seconds: sleeps.append(float(seconds)),
    )

    assert len(calls) == 2
    assert len(cycles) == 2
    assert all(call["symbol"] == "BTC/USDT" for call in calls)
    assert all(int(call["bootstrap_lookback_hours"]) == 12 for call in calls)
    assert all(int(cycle["bootstrap_lookback_hours"]) == 12 for cycle in cycles)
    assert sleeps == [3.0]
