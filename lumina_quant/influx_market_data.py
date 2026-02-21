"""InfluxDB-backed market data helpers.

This module intentionally avoids mandatory runtime dependencies by using
stdlib HTTP requests against InfluxDB v2 endpoints.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from io import StringIO
from typing import Any

import polars as pl


def _empty_ohlcv_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        },
        schema={
            "datetime": pl.Datetime(time_unit="ms"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


def _escape_tag(value: str) -> str:
    return (
        str(value).replace("\\", "\\\\").replace(",", "\\,").replace(" ", "\\ ").replace("=", "\\=")
    )


def _escape_flux_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _normalize_timeframe_token(timeframe: str) -> str:
    raw = str(timeframe or "").strip()
    if not raw or len(raw) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    value = raw[:-1]
    unit = raw[-1]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"Invalid timeframe value: {timeframe}")
    normalized_unit = "M" if unit == "M" else unit.lower()
    if normalized_unit not in {"s", "m", "h", "d", "w", "M"}:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return f"{int(value)}{normalized_unit}"


def _timeframe_to_milliseconds(timeframe: str) -> int:
    token = _normalize_timeframe_token(timeframe)
    unit_ms = {
        "s": 1_000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
        "M": 2_592_000_000,
    }
    return int(token[:-1]) * int(unit_ms[token[-1]])


def _to_flux_time_expr(value: Any, *, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return f'time(v: "{dt.isoformat()}")'
    if isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            numeric *= 1000
        dt = datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
        return f'time(v: "{dt.isoformat()}")'
    text = str(value).strip()
    if not text:
        return fallback
    return f'time(v: "{text.replace("Z", "+00:00")}")'


class InfluxMarketDataRepository:
    """Thin InfluxDB v2 HTTP client for OHLCV points."""

    def __init__(self, *, url: str, org: str, bucket: str, token: str):
        self.url = str(url).rstrip("/")
        self.org = str(org)
        self.bucket = str(bucket)
        self.token = str(token)
        if not self.url or not self.org or not self.bucket or not self.token:
            raise ValueError(
                "InfluxDB configuration incomplete. Require url/org/bucket/token for backend=influxdb."
            )

    def _post(self, *, path: str, body: bytes, content_type: str) -> bytes:
        req = urllib.request.Request(
            url=f"{self.url}{path}",
            data=body,
            headers={
                "Authorization": f"Token {self.token}",
                "Content-Type": content_type,
                "Accept": "application/csv",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            return bytes(resp.read())

    def _query_csv(self, flux_query: str) -> list[dict[str, str]]:
        payload = {
            "query": flux_query,
            "dialect": {
                "annotations": [],
                "header": True,
            },
        }
        raw = self._post(
            path=f"/api/v2/query?org={urllib.parse.quote(self.org)}",
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
        text = raw.decode("utf-8")
        reader = csv.DictReader(StringIO(text))
        rows: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            rows.append({k: str(v or "") for k, v in row.items()})
        return rows

    @staticmethod
    def _rows_to_frame(rows: list[dict[str, str]]) -> pl.DataFrame:
        if not rows:
            return _empty_ohlcv_frame()
        normalized: list[dict[str, Any]] = []
        for row in rows:
            ts = row.get("_time", "")
            if not ts:
                continue
            normalized.append(
                {
                    "datetime": datetime.fromisoformat(ts.replace("Z", "+00:00")),
                    "open": float(row.get("open", "0") or 0.0),
                    "high": float(row.get("high", "0") or 0.0),
                    "low": float(row.get("low", "0") or 0.0),
                    "close": float(row.get("close", "0") or 0.0),
                    "volume": float(row.get("volume", "0") or 0.0),
                }
            )
        if not normalized:
            return _empty_ohlcv_frame()
        return pl.DataFrame(normalized).select(
            ["datetime", "open", "high", "low", "close", "volume"]
        )

    def _query_ohlcv_1s_frame(
        self, *, exchange: str, symbol: str, start_date: Any, end_date: Any
    ) -> pl.DataFrame:
        start_flux = _to_flux_time_expr(
            start_date,
            fallback='time(v: "1970-01-01T00:00:00+00:00")',
        )
        stop_flux = _to_flux_time_expr(end_date, fallback="now()")
        flux = f"""
from(bucket: \"{self.bucket}\")
  |> range(start: {start_flux}, stop: {stop_flux})
  |> filter(fn: (r) => r._measurement == \"market_ohlcv_1s\")
  |> filter(fn: (r) => r.exchange == \"{_escape_flux_string(exchange)}\" and r.symbol == \"{_escape_flux_string(symbol)}\")
  |> filter(fn: (r) => r._field == \"open\" or r._field == \"high\" or r._field == \"low\" or r._field == \"close\" or r._field == \"volume\")
  |> pivot(rowKey: [\"_time\"], columnKey: [\"_field\"], valueColumn: \"_value\")
  |> keep(columns: [\"_time\", \"open\", \"high\", \"low\", \"close\", \"volume\"])
  |> sort(columns: [\"_time\"])
""".strip()
        rows = self._query_csv(flux)
        return self._rows_to_frame(rows)

    def load_ohlcv_1s(
        self, *, exchange: str, symbol: str, start_date: Any = None, end_date: Any = None
    ) -> pl.DataFrame:
        return self._query_ohlcv_1s_frame(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    def load_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        timeframe_ms = int(_timeframe_to_milliseconds(timeframe))
        frame_1s = self._query_ohlcv_1s_frame(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if frame_1s.is_empty() or timeframe_ms <= 1000:
            return frame_1s

        return (
            frame_1s.sort("datetime")
            .with_columns(
                pl.col("datetime").dt.truncate(f"{int(timeframe_ms / 1000)}s").alias("bucket")
            )
            .group_by("bucket")
            .agg(
                [
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                ]
            )
            .rename({"bucket": "datetime"})
            .sort("datetime")
            .select(["datetime", "open", "high", "low", "close", "volume"])
        )

    def load_data_dict(
        self,
        *,
        exchange: str,
        symbol_list: list[str],
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> dict[str, pl.DataFrame]:
        out: dict[str, pl.DataFrame] = {}
        for symbol in symbol_list:
            frame = self.load_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            if not frame.is_empty():
                out[str(symbol)] = frame
        return out

    def export_ohlcv_to_csv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        csv_path: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> int:
        frame = self.load_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        frame.write_csv(csv_path)
        return int(frame.height)

    def market_data_exists(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> bool:
        frame = self.load_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        return not frame.is_empty()

    def get_last_timestamp_ms(self, *, exchange: str, symbol: str) -> int | None:
        frame = self._query_ohlcv_1s_frame(
            exchange=exchange,
            symbol=symbol,
            start_date="1970-01-01T00:00:00+00:00",
            end_date=None,
        )
        if frame.is_empty():
            return None
        last_dt = frame["datetime"].max()
        if last_dt is None:
            return None
        return int(last_dt.timestamp() * 1000)

    def write_ohlcv_1s(self, *, exchange: str, symbol: str, rows: list[tuple[Any, ...]]) -> int:
        if not rows:
            return 0
        lines: list[str] = []
        escaped_exchange = _escape_tag(exchange)
        escaped_symbol = _escape_tag(symbol)
        for row in rows:
            ts = int(row[0])
            open_v = float(row[1])
            high_v = float(row[2])
            low_v = float(row[3])
            close_v = float(row[4])
            volume_v = float(row[5])
            lines.append(
                "market_ohlcv_1s,"
                f"exchange={escaped_exchange},symbol={escaped_symbol} "
                f"open={open_v},high={high_v},low={low_v},close={close_v},volume={volume_v} {ts}"
            )
        body = "\n".join(lines).encode("utf-8")
        self._post(
            path=(
                "/api/v2/write?"
                f"org={urllib.parse.quote(self.org)}"
                f"&bucket={urllib.parse.quote(self.bucket)}"
                "&precision=ms"
            ),
            body=body,
            content_type="text/plain; charset=utf-8",
        )
        return len(rows)
