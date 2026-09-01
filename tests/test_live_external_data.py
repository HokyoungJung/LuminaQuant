from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from lumina_quant.live.data_external import ExternalWindowDataHandler


class _ThreadStub:
    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon

    def start(self):
        return None

    def is_alive(self):
        return False

    def join(self, timeout=None):
        _ = timeout


def test_external_window_data_handler_emits_market_window_from_payload(monkeypatch):
    payload = {
        "time": 1_700_000_000_000,
        "window_seconds": 5,
        "bars_1s": {
            "BTC/USDT": [[1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 10.0]],
        },
        "event_time_watermark_ms": 1_700_000_000_000,
        "lag_ms": 0,
        "is_stale": False,
    }

    monkeypatch.setattr("lumina_quant.live.data_external.threading.Thread", _ThreadStub)
    events = queue.Queue()
    cfg = SimpleNamespace(
        EXTERNAL_DATA_SOURCE_KIND="jsonl",
        EXTERNAL_DATA_PATH="unused.jsonl",
        EXTERNAL_DATA_SCHEMA="market_window_v1",
        EXTERNAL_DATA_POLL_SECONDS=1,
        EXTERNAL_DATA_ALLOW_STALE_SECONDS=45,
        INGEST_WINDOW_SECONDS=5,
        EXTERNAL_DATA_SYMBOL_MAP={},
    )
    handler = ExternalWindowDataHandler(events, ["BTC/USDT"], cfg, exchange=None)
    handler._emit_payload(payload)

    event = events.get_nowait()
    assert event.type == "MARKET_WINDOW"
    assert "BTC/USDT" in event.bars_1s


def test_external_live_parquet_single_file_rejects_multi_symbol(monkeypatch, tmp_path):
    path = tmp_path / "single.parquet"
    path.write_bytes(b"stub")
    monkeypatch.setattr("lumina_quant.live.data_external.threading.Thread", _ThreadStub)
    cfg = SimpleNamespace(
        EXTERNAL_DATA_SOURCE_KIND="parquet",
        EXTERNAL_DATA_PATH=str(path),
        EXTERNAL_DATA_SCHEMA="ohlcv_1s_v1",
        EXTERNAL_DATA_POLL_SECONDS=1,
        EXTERNAL_DATA_ALLOW_STALE_SECONDS=45,
        INGEST_WINDOW_SECONDS=5,
        EXTERNAL_DATA_SYMBOL_MAP={},
    )
    handler = ExternalWindowDataHandler(None, ["BTC/USDT", "ETH/USDT"], cfg, exchange=None)
    with pytest.raises(RuntimeError):
        handler._resolve_parquet_path("BTC/USDT")
