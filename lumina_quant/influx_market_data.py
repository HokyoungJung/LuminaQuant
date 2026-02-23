"""InfluxDB-backed market data helpers.

This module intentionally avoids mandatory runtime dependencies by using
stdlib HTTP requests against InfluxDB v2 endpoints.
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from io import StringIO
from typing import Any

import polars as pl

MARKET_OHLCV_1S_MEASUREMENT = "market_ohlcv_1s"
MARKET_OHLCV_MEASUREMENT = "market_ohlcv"
FUTURES_FEATURE_POINTS_MEASUREMENT = "futures_feature_points"


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


def _coerce_datetime_utc(value: Any, *, default: datetime) -> datetime:
    if value is None:
        return default
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    if isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            numeric *= 1000
        return datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
    text = str(value).strip()
    if not text:
        return default
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class InfluxMarketDataRepository:
    """Thin InfluxDB v2 HTTP client for OHLCV points."""

    def __init__(self, *, url: str, org: str, bucket: str, token: str):
        self.url = str(url).rstrip("/")
        self.org = str(org)
        self.bucket = str(bucket)
        self.token = str(token)
        timeout_raw = (
            os.getenv("LQ__STORAGE__INFLUX_TIMEOUT_SEC")
            or os.getenv("LQ_STORAGE_INFLUX_TIMEOUT_SEC")
            or os.getenv("INFLUX_TIMEOUT_SEC")
            or "120"
        )
        try:
            self.timeout_sec = max(5.0, float(timeout_raw))
        except Exception:
            self.timeout_sec = 120.0
        if not self.url or not self.org or not self.bucket or not self.token:
            raise ValueError(
                "InfluxDB configuration incomplete. Require url/org/bucket/token for backend=influxdb."
            )

    def _post(self, *, path: str, body: bytes, content_type: str) -> bytes:
        retry_raw = (
            os.getenv("LQ__STORAGE__INFLUX_RETRY_MAX")
            or os.getenv("LQ_STORAGE_INFLUX_RETRY_MAX")
            or "2"
        )
        try:
            retry_max = max(0, int(retry_raw))
        except Exception:
            retry_max = 2

        last_error: Exception | None = None
        for attempt in range(retry_max + 1):
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
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:  # nosec B310
                    return bytes(resp.read())
            except Exception as exc:  # pragma: no cover - runtime network path
                last_error = exc
                if attempt >= retry_max:
                    break
                backoff = min(4.0, 0.35 * (2**attempt))
                time.sleep(backoff)

        if last_error is not None:
            raise last_error
        return b""

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
                    "datetime": datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
                        UTC
                    ).replace(tzinfo=None),
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

    @staticmethod
    def _rows_to_feature_frame(rows: list[dict[str, str]]) -> pl.DataFrame:
        schema = {
            "datetime": pl.Datetime(time_unit="ms"),
            "funding_rate": pl.Float64,
            "funding_mark_price": pl.Float64,
            "mark_price": pl.Float64,
            "index_price": pl.Float64,
            "open_interest": pl.Float64,
            "liquidation_long_qty": pl.Float64,
            "liquidation_short_qty": pl.Float64,
            "liquidation_long_notional": pl.Float64,
            "liquidation_short_notional": pl.Float64,
        }
        if not rows:
            return pl.DataFrame(
                {
                    "datetime": [],
                    "funding_rate": [],
                    "funding_mark_price": [],
                    "mark_price": [],
                    "index_price": [],
                    "open_interest": [],
                    "liquidation_long_qty": [],
                    "liquidation_short_qty": [],
                    "liquidation_long_notional": [],
                    "liquidation_short_notional": [],
                },
                schema=schema,
            )

        normalized: list[dict[str, Any]] = []
        for row in rows:
            ts = row.get("_time", "")
            if not ts:
                continue

            def _to_float(value: str) -> float | None:
                text = str(value or "").strip()
                if not text:
                    return None
                try:
                    return float(text)
                except Exception:
                    return None

            normalized.append(
                {
                    "datetime": datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
                        UTC
                    ).replace(tzinfo=None),
                    "funding_rate": _to_float(row.get("funding_rate", "")),
                    "funding_mark_price": _to_float(row.get("funding_mark_price", "")),
                    "mark_price": _to_float(row.get("mark_price", "")),
                    "index_price": _to_float(row.get("index_price", "")),
                    "open_interest": _to_float(row.get("open_interest", "")),
                    "liquidation_long_qty": _to_float(row.get("liquidation_long_qty", "")),
                    "liquidation_short_qty": _to_float(row.get("liquidation_short_qty", "")),
                    "liquidation_long_notional": _to_float(
                        row.get("liquidation_long_notional", "")
                    ),
                    "liquidation_short_notional": _to_float(
                        row.get("liquidation_short_notional", "")
                    ),
                }
            )
        if not normalized:
            return pl.DataFrame(
                {
                    "datetime": [],
                    "funding_rate": [],
                    "funding_mark_price": [],
                    "mark_price": [],
                    "index_price": [],
                    "open_interest": [],
                    "liquidation_long_qty": [],
                    "liquidation_short_qty": [],
                    "liquidation_long_notional": [],
                    "liquidation_short_notional": [],
                },
                schema=schema,
            )
        return pl.DataFrame(normalized).select(
            [
                "datetime",
                "funding_rate",
                "funding_mark_price",
                "mark_price",
                "index_price",
                "open_interest",
                "liquidation_long_qty",
                "liquidation_short_qty",
                "liquidation_long_notional",
                "liquidation_short_notional",
            ]
        )

    def _query_ohlcv_1s_frame(
        self, *, exchange: str, symbol: str, start_date: Any, end_date: Any
    ) -> pl.DataFrame:
        start_dt = _coerce_datetime_utc(
            start_date,
            default=datetime(1970, 1, 1, tzinfo=UTC),
        )
        end_dt = _coerce_datetime_utc(
            end_date,
            default=datetime.now(UTC),
        )
        if end_dt <= start_dt:
            return _empty_ohlcv_frame()

        chunk_hours_raw = os.getenv("LQ__STORAGE__INFLUX_1S_QUERY_CHUNK_HOURS", "0")
        chunk_days_raw = os.getenv("LQ__STORAGE__INFLUX_1S_QUERY_CHUNK_DAYS", "3")
        try:
            chunk_hours = max(0, int(chunk_hours_raw))
        except Exception:
            chunk_hours = 0
        try:
            chunk_days = max(1, int(chunk_days_raw))
        except Exception:
            chunk_days = 3

        exchange_escaped = _escape_flux_string(exchange)
        symbol_escaped = _escape_flux_string(symbol)
        step = timedelta(hours=chunk_hours) if chunk_hours > 0 else timedelta(days=chunk_days)
        frames: list[pl.DataFrame] = []
        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + step, end_dt)
            start_flux = f'time(v: "{cursor.isoformat()}")'
            stop_flux = f'time(v: "{chunk_end.isoformat()}")'
            flux = f"""
from(bucket: "{self.bucket}")
  |> range(start: {start_flux}, stop: {stop_flux})
  |> filter(fn: (r) => r._measurement == "{MARKET_OHLCV_1S_MEASUREMENT}")
  |> filter(fn: (r) => r.exchange == "{exchange_escaped}" and r.symbol == "{symbol_escaped}")
  |> filter(fn: (r) => r._field == "open" or r._field == "high" or r._field == "low" or r._field == "close" or r._field == "volume")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])
  |> sort(columns: ["_time"])
""".strip()
            rows = self._query_csv(flux)
            frame = self._rows_to_frame(rows)
            if not frame.is_empty():
                frames.append(frame)
            cursor = chunk_end

        if not frames:
            return _empty_ohlcv_frame()
        return (
            pl.concat(frames, how="vertical_relaxed")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
            .select(["datetime", "open", "high", "low", "close", "volume"])
        )

    def _query_ohlcv_timeframe_frame(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any,
        end_date: Any,
    ) -> pl.DataFrame:
        start_flux = _to_flux_time_expr(
            start_date,
            fallback='time(v: "1970-01-01T00:00:00+00:00")',
        )
        stop_flux = _to_flux_time_expr(end_date, fallback="now()")
        tf = _normalize_timeframe_token(timeframe)
        flux = f"""
from(bucket: \"{self.bucket}\")
  |> range(start: {start_flux}, stop: {stop_flux})
  |> filter(fn: (r) => r._measurement == \"{MARKET_OHLCV_MEASUREMENT}\")
  |> filter(fn: (r) => r.exchange == \"{_escape_flux_string(exchange)}\" and r.symbol == \"{_escape_flux_string(symbol)}\" and r.timeframe == \"{_escape_flux_string(tf)}\")
  |> filter(fn: (r) => r._field == \"open\" or r._field == \"high\" or r._field == \"low\" or r._field == \"close\" or r._field == \"volume\")
  |> pivot(rowKey: [\"_time\"], columnKey: [\"_field\"], valueColumn: \"_value\")
  |> keep(columns: [\"_time\", \"open\", \"high\", \"low\", \"close\", \"volume\"])
  |> sort(columns: [\"_time\"])
""".strip()
        rows = self._query_csv(flux)
        return self._rows_to_frame(rows)

    def _query_ohlcv_aggregated_from_1s_frame(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any,
        end_date: Any,
    ) -> pl.DataFrame:
        tf = _normalize_timeframe_token(timeframe)
        if tf == "1s":
            return self._query_ohlcv_1s_frame(
                exchange=exchange,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
        start_dt = _coerce_datetime_utc(
            start_date,
            default=datetime(1970, 1, 1, tzinfo=UTC),
        )
        end_dt = _coerce_datetime_utc(
            end_date,
            default=datetime.now(UTC),
        )
        if end_dt <= start_dt:
            return _empty_ohlcv_frame()

        chunk_hours_raw = os.getenv("LQ__STORAGE__INFLUX_1S_AGG_CHUNK_HOURS", "0")
        chunk_days_raw = os.getenv("LQ__STORAGE__INFLUX_1S_AGG_CHUNK_DAYS", "14")
        try:
            chunk_hours = max(0, int(chunk_hours_raw))
        except Exception:
            chunk_hours = 0
        try:
            chunk_days = max(1, int(chunk_days_raw))
        except Exception:
            chunk_days = 14
        exchange_escaped = _escape_flux_string(exchange)
        symbol_escaped = _escape_flux_string(symbol)
        step = timedelta(hours=chunk_hours) if chunk_hours > 0 else timedelta(days=chunk_days)
        frames: list[pl.DataFrame] = []
        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + step, end_dt)
            start_flux = f'time(v: "{cursor.isoformat()}")'
            stop_flux = f'time(v: "{chunk_end.isoformat()}")'
            flux = f"""
base = from(bucket: "{self.bucket}")
  |> range(start: {start_flux}, stop: {stop_flux})
  |> filter(fn: (r) => r._measurement == "{MARKET_OHLCV_1S_MEASUREMENT}")
  |> filter(fn: (r) => r.exchange == "{exchange_escaped}" and r.symbol == "{symbol_escaped}")

open_t = base |> filter(fn: (r) => r._field == "open") |> aggregateWindow(every: {tf}, fn: first, createEmpty: false)
high_t = base |> filter(fn: (r) => r._field == "high") |> aggregateWindow(every: {tf}, fn: max, createEmpty: false)
low_t = base |> filter(fn: (r) => r._field == "low") |> aggregateWindow(every: {tf}, fn: min, createEmpty: false)
close_t = base |> filter(fn: (r) => r._field == "close") |> aggregateWindow(every: {tf}, fn: last, createEmpty: false)
volume_t = base |> filter(fn: (r) => r._field == "volume") |> aggregateWindow(every: {tf}, fn: sum, createEmpty: false)

union(tables: [open_t, high_t, low_t, close_t, volume_t])
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])
  |> sort(columns: ["_time"])
""".strip()
            rows = self._query_csv(flux)
            frame = self._rows_to_frame(rows)
            if not frame.is_empty():
                frames.append(frame)
            cursor = chunk_end

        if not frames:
            return _empty_ohlcv_frame()
        return (
            pl.concat(frames, how="vertical_relaxed")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
            .select(["datetime", "open", "high", "low", "close", "volume"])
        )

    def _query_futures_feature_frame(
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
  |> filter(fn: (r) => r._measurement == \"{FUTURES_FEATURE_POINTS_MEASUREMENT}\")
  |> filter(fn: (r) => r.exchange == \"{_escape_flux_string(exchange)}\" and r.symbol == \"{_escape_flux_string(symbol)}\")
  |> filter(fn: (r) => r._field == \"funding_rate\" or r._field == \"funding_mark_price\" or r._field == \"mark_price\" or r._field == \"index_price\" or r._field == \"open_interest\" or r._field == \"liquidation_long_qty\" or r._field == \"liquidation_short_qty\" or r._field == \"liquidation_long_notional\" or r._field == \"liquidation_short_notional\")
  |> pivot(rowKey: [\"_time\"], columnKey: [\"_field\"], valueColumn: \"_value\")
  |> keep(columns: [\"_time\", \"funding_rate\", \"funding_mark_price\", \"mark_price\", \"index_price\", \"open_interest\", \"liquidation_long_qty\", \"liquidation_short_qty\", \"liquidation_long_notional\", \"liquidation_short_notional\"])
  |> sort(columns: [\"_time\"])
""".strip()
        rows = self._query_csv(flux)
        return self._rows_to_feature_frame(rows)

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
        timeframe_token = _normalize_timeframe_token(timeframe)
        timeframe_ms = int(_timeframe_to_milliseconds(timeframe_token))
        if timeframe_token != "1s":
            direct = self._query_ohlcv_timeframe_frame(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe_token,
                start_date=start_date,
                end_date=end_date,
            )
            if not direct.is_empty():
                return direct
            aggregated_from_1s = self._query_ohlcv_aggregated_from_1s_frame(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe_token,
                start_date=start_date,
                end_date=end_date,
            )
            if not aggregated_from_1s.is_empty():
                return aggregated_from_1s

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

    @staticmethod
    def _timestamp_ms_from_flux_rows(rows: list[dict[str, str]]) -> int | None:
        if not rows:
            return None
        ts = str(rows[-1].get("_time", "")).strip()
        if not ts:
            return None
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)

    def get_last_timestamp_ms(
        self, *, exchange: str, symbol: str, timeframe: str = "1s"
    ) -> int | None:
        tf = _normalize_timeframe_token(timeframe)
        measurement = (
            MARKET_OHLCV_1S_MEASUREMENT if tf == "1s" else MARKET_OHLCV_MEASUREMENT
        )
        filter_timeframe = (
            ""
            if tf == "1s"
            else f'|> filter(fn: (r) => r.timeframe == "{_escape_flux_string(tf)}")\n  '
        )
        flux = f"""
from(bucket: \"{self.bucket}\")
  |> range(start: time(v: \"1970-01-01T00:00:00+00:00\"), stop: now())
  |> filter(fn: (r) => r._measurement == \"{measurement}\")
  |> filter(fn: (r) => r.exchange == \"{_escape_flux_string(exchange)}\" and r.symbol == \"{_escape_flux_string(symbol)}\")
  {filter_timeframe}|> filter(fn: (r) => r._field == \"close\")
  |> last()
  |> keep(columns: [\"_time\"])
""".strip()
        rows = self._query_csv(flux)
        return self._timestamp_ms_from_flux_rows(rows)

    def get_ohlcv_coverage(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> tuple[int | None, int | None, int]:
        return self.get_ohlcv_coverage_between(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=None,
            end_date=None,
        )

    def get_ohlcv_coverage_between(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> tuple[int | None, int | None, int]:
        tf = _normalize_timeframe_token(timeframe)
        measurement = (
            MARKET_OHLCV_1S_MEASUREMENT if tf == "1s" else MARKET_OHLCV_MEASUREMENT
        )
        filter_timeframe = (
            ""
            if tf == "1s"
            else f'|> filter(fn: (r) => r.timeframe == "{_escape_flux_string(tf)}")\n  '
        )
        start_expr = _to_flux_time_expr(start_date, fallback='time(v: "1970-01-01T00:00:00+00:00")')
        stop_expr = _to_flux_time_expr(end_date, fallback="now()")
        common = (
            f'from(bucket: "{self.bucket}")\n'
            f"  |> range(start: {start_expr}, stop: {stop_expr})\n"
            f'  |> filter(fn: (r) => r._measurement == "{measurement}")\n'
            f'  |> filter(fn: (r) => r.exchange == "{_escape_flux_string(exchange)}" and r.symbol == "{_escape_flux_string(symbol)}")\n'
            f"  {filter_timeframe}|> filter(fn: (r) => r._field == \"close\")\n"
        )

        first_rows = self._query_csv(
            (
                common
                + '  |> first()\n'
                + '  |> keep(columns: ["_time"])'
            ).strip()
        )
        last_rows = self._query_csv(
            (
                common
                + '  |> last()\n'
                + '  |> keep(columns: ["_time"])'
            ).strip()
        )
        count_rows = self._query_csv(
            (
                common
                + '  |> count(column: "_value")\n'
                + '  |> keep(columns: ["_value"])'
            ).strip()
        )
        first_ts = self._timestamp_ms_from_flux_rows(first_rows)
        last_ts = self._timestamp_ms_from_flux_rows(last_rows)
        row_count = 0
        if count_rows:
            raw = str(count_rows[-1].get("_value", "")).strip()
            try:
                row_count = int(raw)
            except Exception:
                row_count = 0
        if first_ts is None or last_ts is None or row_count <= 0:
            return None, None, 0
        return first_ts, last_ts, row_count

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
                f"{MARKET_OHLCV_1S_MEASUREMENT},"
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

    def write_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        rows: list[tuple[Any, ...]],
    ) -> int:
        if not rows:
            return 0
        tf = _normalize_timeframe_token(timeframe)
        lines: list[str] = []
        escaped_exchange = _escape_tag(exchange)
        escaped_symbol = _escape_tag(symbol)
        escaped_timeframe = _escape_tag(tf)
        for row in rows:
            ts = int(row[0])
            open_v = float(row[1])
            high_v = float(row[2])
            low_v = float(row[3])
            close_v = float(row[4])
            volume_v = float(row[5])
            lines.append(
                f"{MARKET_OHLCV_MEASUREMENT},"
                f"exchange={escaped_exchange},symbol={escaped_symbol},timeframe={escaped_timeframe} "
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

    def write_futures_feature_points(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: list[dict[str, Any]],
        source: str = "binance_futures_api",
    ) -> int:
        if not rows:
            return 0
        lines: list[str] = []
        escaped_exchange = _escape_tag(str(exchange).strip().lower())
        escaped_symbol = _escape_tag(str(symbol).strip().upper().replace("-", "/").replace("_", "/"))
        escaped_source = _escape_tag(str(source).strip().lower())
        for row in rows:
            ts_raw = row.get("timestamp_ms")
            if ts_raw is None:
                continue
            ts = int(ts_raw)
            fields: list[str] = []
            for key in (
                "funding_rate",
                "funding_mark_price",
                "mark_price",
                "index_price",
                "open_interest",
                "liquidation_long_qty",
                "liquidation_short_qty",
                "liquidation_long_notional",
                "liquidation_short_notional",
            ):
                value = row.get(key)
                if value is None:
                    continue
                fields.append(f"{key}={float(value)}")
            if not fields:
                continue
            lines.append(
                f"{FUTURES_FEATURE_POINTS_MEASUREMENT},"
                f"exchange={escaped_exchange},symbol={escaped_symbol},source={escaped_source} "
                f"{','.join(fields)} {ts}"
            )
        if not lines:
            return 0
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
        return len(lines)

    def load_futures_feature_points(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        return self._query_futures_feature_frame(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
