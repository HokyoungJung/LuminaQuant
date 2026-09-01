from __future__ import annotations

import queue

from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.core.market_window_contract import (
    build_market_window_event,
    market_window_event_payload,
)


def test_market_window_emission_parity_live_vs_backtest():
    bars = {
        "BTC/USDT": (
            (1_700_000_001_000, 1.1, 2.1, 0.6, 1.6, 110.0),
            (1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 100.0),
        ),
    }

    live_event = build_market_window_event(
        time=1_700_000_001_000,
        window_seconds=20,
        bars_1s=bars,
        event_time_watermark_ms=1_700_000_001_000,
        commit_id="commit-1",
        lag_ms=0,
        is_stale=False,
        parity_v2_enabled=True,
    )

    handler = object.__new__(HistoricParquetWindowedDataHandler)
    handler.events = queue.Queue()
    handler.backtest_window_seconds = 20
    handler._parity_v2_enabled = True
    handler._metrics_log_path = "logs/live/market_window_metrics.ndjson"
    handler.last_emitted_timestamp_ms = 1_700_000_001_000
    handler._window_snapshot = lambda: bars
    handler._last_window_event_ms = None

    HistoricParquetWindowedDataHandler._emit_window_event(handler, event_time=1_700_000_001_000)
    backtest_event = handler.events.get_nowait()

    live_payload = market_window_event_payload(live_event)
    backtest_payload = market_window_event_payload(backtest_event)

    assert set(live_payload.keys()) == set(backtest_payload.keys())
    assert live_payload["type"] == backtest_payload["type"] == "MARKET_WINDOW"
    assert list(live_payload["bars_1s"].keys()) == list(backtest_payload["bars_1s"].keys())
    assert live_payload["bars_1s"]["BTC/USDT"] == backtest_payload["bars_1s"]["BTC/USDT"]
