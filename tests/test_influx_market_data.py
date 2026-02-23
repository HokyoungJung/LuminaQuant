from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from lumina_quant.influx_market_data import InfluxMarketDataRepository


class _FakeInfluxRepo(InfluxMarketDataRepository):
    def __init__(
        self,
        frame_1s: pl.DataFrame,
        frame_timeframe: pl.DataFrame | None = None,
        frame_aggregated: pl.DataFrame | None = None,
    ):
        self.url = "http://localhost:8086"
        self.org = "test-org"
        self.bucket = "test-bucket"
        self.token = "test-token"
        self._frame_1s = frame_1s
        self._frame_timeframe = frame_timeframe if frame_timeframe is not None else pl.DataFrame()
        self._frame_aggregated = frame_aggregated if frame_aggregated is not None else pl.DataFrame()

    def _query_ohlcv_1s_frame(
        self, *, exchange: str, symbol: str, start_date, end_date
    ) -> pl.DataFrame:
        _ = (exchange, symbol, start_date, end_date)
        return self._frame_1s

    def _query_ohlcv_timeframe_frame(
        self, *, exchange: str, symbol: str, timeframe: str, start_date, end_date
    ) -> pl.DataFrame:
        _ = (exchange, symbol, timeframe, start_date, end_date)
        return self._frame_timeframe

    def _query_ohlcv_aggregated_from_1s_frame(
        self, *, exchange: str, symbol: str, timeframe: str, start_date, end_date
    ) -> pl.DataFrame:
        _ = (exchange, symbol, timeframe, start_date, end_date)
        return self._frame_aggregated


class _CaptureWriteRepo(InfluxMarketDataRepository):
    def __init__(self):
        self.url = "http://localhost:8086"
        self.org = "test-org"
        self.bucket = "test-bucket"
        self.token = "test-token"
        self.last_body = b""
        self.last_path = ""

    def _post(self, *, path: str, body: bytes, content_type: str) -> bytes:
        _ = content_type
        self.last_path = path
        self.last_body = body
        return b""


class _FeatureQueryRepo(InfluxMarketDataRepository):
    def __init__(self):
        self.url = "http://localhost:8086"
        self.org = "test-org"
        self.bucket = "test-bucket"
        self.token = "test-token"

    def _query_csv(self, flux_query: str) -> list[dict[str, str]]:
        _ = flux_query
        return [
            {
                "_time": "2026-01-01T00:00:00Z",
                "funding_rate": "0.0001",
                "funding_mark_price": "42000.0",
                "mark_price": "42100.0",
                "index_price": "42050.0",
                "open_interest": "1500000.0",
                "liquidation_long_qty": "10.0",
                "liquidation_short_qty": "8.0",
                "liquidation_long_notional": "420000.0",
                "liquidation_short_notional": "336000.0",
            }
        ]


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


def test_influx_repo_prefers_direct_timeframe_measurement_when_present():
    direct = pl.DataFrame(
        {
            "datetime": [datetime(2026, 1, 1, 0, 0, 0)],
            "open": [111.0],
            "high": [112.0],
            "low": [110.0],
            "close": [111.5],
            "volume": [42.0],
        }
    )
    repo = _FakeInfluxRepo(_make_1s_frame(), frame_timeframe=direct)
    frame = repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1m")
    assert frame.height == 1
    assert float(frame["open"][0]) == 111.0
    assert float(frame["close"][0]) == 111.5


def test_influx_repo_prefers_aggregated_1s_query_when_direct_missing():
    aggregated = pl.DataFrame(
        {
            "datetime": [datetime(2026, 1, 1, 0, 0, 0)],
            "open": [210.0],
            "high": [215.0],
            "low": [205.0],
            "close": [212.0],
            "volume": [84.0],
        }
    )
    repo = _FakeInfluxRepo(
        _make_1s_frame(),
        frame_timeframe=pl.DataFrame(),
        frame_aggregated=aggregated,
    )
    frame = repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1m")
    assert frame.height == 1
    assert float(frame["open"][0]) == 210.0
    assert float(frame["close"][0]) == 212.0


def test_influx_repo_writes_futures_feature_points_line_protocol():
    repo = _CaptureWriteRepo()
    written = repo.write_futures_feature_points(
        exchange="binance",
        symbol="BTC/USDT",
        source="binance_futures_api",
        rows=[
            {
                "timestamp_ms": 1_700_000_000_000,
                "funding_rate": 0.0001,
                "open_interest": 1_000_000.0,
                "liquidation_long_notional": 25_000.0,
            }
        ],
    )
    assert written == 1
    payload = repo.last_body.decode("utf-8")
    assert "futures_feature_points,exchange=binance,symbol=BTC/USDT,source=binance_futures_api" in payload
    assert "funding_rate=0.0001" in payload
    assert "open_interest=1000000.0" in payload
    assert "liquidation_long_notional=25000.0" in payload
    assert "&precision=ms" in repo.last_path


def test_influx_repo_loads_futures_feature_points_frame():
    repo = _FeatureQueryRepo()
    frame = repo.load_futures_feature_points(exchange="binance", symbol="BTC/USDT")
    assert frame.height == 1
    assert float(frame["funding_rate"][0]) == 0.0001
    assert float(frame["mark_price"][0]) == 42100.0
    assert float(frame["open_interest"][0]) == 1_500_000.0
