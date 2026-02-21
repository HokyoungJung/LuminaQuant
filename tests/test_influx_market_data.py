from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from lumina_quant.influx_market_data import InfluxMarketDataRepository


class _FakeInfluxRepo(InfluxMarketDataRepository):
    def __init__(self, frame: pl.DataFrame):
        self.url = "http://localhost:8086"
        self.org = "test-org"
        self.bucket = "test-bucket"
        self.token = "test-token"
        self._frame = frame

    def _query_ohlcv_1s_frame(
        self, *, exchange: str, symbol: str, start_date, end_date
    ) -> pl.DataFrame:
        _ = (exchange, symbol, start_date, end_date)
        return self._frame


def _make_1s_frame() -> pl.DataFrame:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = []
    for i in range(4):
        dt = start + timedelta(seconds=i)
        rows.append(
            {
                "datetime": dt,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 10.0 + i,
            }
        )
    return pl.DataFrame(rows)


def test_influx_repo_aggregates_1s_to_minute():
    repo = _FakeInfluxRepo(_make_1s_frame())
    frame = repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1m")
    assert frame.height == 1
    assert float(frame["open"][0]) == 100.0
    assert float(frame["high"][0]) == 104.0
    assert float(frame["low"][0]) == 99.0
    assert float(frame["close"][0]) == 103.5
    assert float(frame["volume"][0]) == 46.0


def test_influx_repo_market_data_exists_uses_timeframe_query():
    repo = _FakeInfluxRepo(_make_1s_frame())
    assert repo.market_data_exists(exchange="binance", symbol="BTC/USDT", timeframe="1m") is True
