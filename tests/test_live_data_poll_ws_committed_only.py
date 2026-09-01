from __future__ import annotations

import queue
from types import SimpleNamespace

from lumina_quant.live.data_poll import LiveDataHandler
from lumina_quant.live.data_ws import BinanceWebSocketDataHandler


class _Config:
    MARKET_DATA_PARQUET_PATH = "data/market_parquet"
    MARKET_DATA_EXCHANGE = "binance"
    LIVE_POLL_SECONDS = 1
    INGEST_WINDOW_SECONDS = 5
    MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45


class _ReaderStub:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs

    @staticmethod
    def read_snapshot():
        return SimpleNamespace(
            event_time_ms=1_700_000_000_000,
            event_time_watermark_ms=1_700_000_000_000,
            bars_1s={"BTC/USDT": ((1_700_000_000_000, 1.0, 1.0, 1.0, 1.0, 1.0),)},
            commit_id="c1",
            lag_ms=0,
            is_stale=False,
        )


class _ThreadStub:
    def __init__(self, target, daemon=True):
        self.target = target
        self.daemon = daemon

    def start(self):
        return None


class _ExchangeGuard:
    def fetch_ohlcv(self, *_args, **_kwargs):
        raise AssertionError("fetch_ohlcv must not be called in committed-only handler")

    def fetch_trades(self, *_args, **_kwargs):
        raise AssertionError("fetch_trades must not be called in committed-only handler")


def _assert_committed_only(handler_cls, monkeypatch):
    events = queue.Queue()
    exchange = _ExchangeGuard()
    sleeps = {"count": 0}

    def _sleep(_seconds):
        sleeps["count"] += 1
        if sleeps["count"] >= 1:
            handler.continue_backtest = False

    monkeypatch.setattr("lumina_quant.live.data_materialized.MaterializedWindowReader", _ReaderStub)
    monkeypatch.setattr("lumina_quant.live.data_materialized.threading.Thread", _ThreadStub)
    monkeypatch.setattr("lumina_quant.live.data_materialized.time.sleep", _sleep)
    handler = handler_cls(events, ["BTC/USDT"], _Config, exchange)
    handler.continue_backtest = True
    handler._poll_market_data()

    emitted = [events.get_nowait() for _ in range(events.qsize())]
    assert emitted
    assert getattr(emitted[0], "type", "") == "MARKET_WINDOW"


def test_live_data_poll_handler_committed_only(monkeypatch):
    _assert_committed_only(LiveDataHandler, monkeypatch)


def test_live_data_ws_handler_committed_only(monkeypatch):
    _assert_committed_only(BinanceWebSocketDataHandler, monkeypatch)
