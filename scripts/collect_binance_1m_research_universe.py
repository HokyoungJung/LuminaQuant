#!/usr/bin/env python3
"""Collect Binance USD-M Futures 1m klines for the extended research universe.

This is a direct 1m OHLCV collector for broad research/shadow scans. It does not
collect raw aggTrades, does not derive 1s bars, and does not place orders.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.market_data import MarketDataRepository, normalize_timeframe_token  # noqa: E402
from lumina_quant.research_universe import (  # noqa: E402
    BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS,
    BINANCE_EXTENDED_RESEARCH_SYMBOLS,
)
from lumina_quant.symbols import canonical_symbol  # noqa: E402

FAPI_BASE_URL = "https://fapi.binance.com"
DEFAULT_DB_PATH = REPO_ROOT / "data/market_parquet"
DEFAULT_REPORT_DIR = REPO_ROOT / "var/reports/data_collection/binance_1m_research_universe"
DEFAULT_SINCE_UTC = "2025-01-01T00:00:00Z"
KLINE_INTERVAL_MS = 60_000
PRINT_LOCK = threading.Lock()
BINANCE_BAN_UNTIL_RE = re.compile(r"banned until (?P<until_ms>\d+)")


@dataclass(frozen=True, slots=True)
class SymbolPlan:
    symbol: str
    underlying_type: str
    start_ms: int
    end_ms: int
    onboard_ms: int | None


@dataclass(frozen=True, slots=True)
class SymbolResult:
    symbol: str
    status: str
    request_count: int
    fetched_rows: int
    upserted_rows: int
    started_at_utc: str
    completed_at_utc: str
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    source_files: int = 0
    missing_files: int = 0
    error: str = ""


class GlobalRequestThrottle:
    """Shared request throttle for Binance endpoints across worker threads."""

    def __init__(self, *, min_interval_sec: float) -> None:
        self._min_interval_sec = max(0.0, float(min_interval_sec))
        self._next_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                wait_sec = self._next_at - now
                if wait_sec <= 0:
                    self._next_at = now + self._min_interval_sec
                    return
            time.sleep(min(wait_sec, 5.0))

    def pause(self, wait_sec: float) -> None:
        if wait_sec <= 0:
            return
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + float(wait_sec))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_utc_ms(raw: str | None) -> int:
    if not raw:
        raise ValueError("timestamp is required")
    token = str(raw).strip().replace("Z", "+00:00")
    return int(datetime.fromisoformat(token).astimezone(UTC).timestamp() * 1000)


def ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")


def compact_symbol(symbol: str) -> str:
    return canonical_symbol(str(symbol or "")).replace("/", "").upper()


def direct_series_path(db_path: Path, *, exchange: str, symbol: str, timeframe: str) -> Path:
    return (
        db_path
        / f"exchange={str(exchange).strip().lower()}"
        / f"symbol={compact_symbol(symbol)}"
        / f"timeframe={normalize_timeframe_token(timeframe)}"
    )


def direct_last_timestamp_ms(
    db_path: Path,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> int | None:
    base = direct_series_path(db_path, exchange=exchange, symbol=symbol, timeframe=timeframe)
    paths = sorted(str(path) for path in base.glob("date=*/*.parquet"))
    if not paths:
        return None
    try:
        value = (
            pl.scan_parquet(paths).select(pl.col("datetime").max().alias("latest")).collect().item()
        )
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.replace(tzinfo=UTC).timestamp() * 1000)
    return int(value)


def load_exchange_info() -> dict[str, dict[str, Any]]:
    with urllib.request.urlopen(f"{FAPI_BASE_URL}/fapi/v1/exchangeInfo", timeout=30) as response:
        payload = json.load(response)
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("symbols") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if symbol:
            out[symbol] = row
    return out


def default_symbols_from_exchange_info(exchange_info: dict[str, dict[str, Any]]) -> list[str]:
    tradfi = [
        symbol
        for symbol, row in exchange_info.items()
        if row.get("contractType") == "TRADIFI_PERPETUAL"
        and row.get("quoteAsset") == "USDT"
        and row.get("status") == "TRADING"
    ]
    ordered: list[str] = []
    for symbol in (*BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS, *sorted(tradfi)):
        token = compact_symbol(symbol)
        if token and token not in ordered:
            ordered.append(token)
    return ordered


def make_symbol_plans(
    symbols: Iterable[str],
    *,
    exchange_info: dict[str, dict[str, Any]],
    since_ms: int,
    until_ms: int,
    db_path: Path,
    exchange: str,
    resume: bool,
) -> list[SymbolPlan]:
    plans: list[SymbolPlan] = []
    for raw_symbol in symbols:
        symbol = compact_symbol(raw_symbol)
        row = exchange_info.get(symbol, {})
        onboard_raw = row.get("onboardDate")
        onboard_ms = int(onboard_raw) if onboard_raw is not None else None
        start_ms = max(int(since_ms), int(onboard_ms or since_ms))
        if resume:
            last_ms = direct_last_timestamp_ms(
                db_path,
                exchange=exchange,
                symbol=symbol,
                timeframe="1m",
            )
            if last_ms is not None:
                start_ms = max(start_ms, int(last_ms) + KLINE_INTERVAL_MS)
        start_ms = (start_ms // KLINE_INTERVAL_MS) * KLINE_INTERVAL_MS
        end_ms = ((int(until_ms) + 1) // KLINE_INTERVAL_MS) * KLINE_INTERVAL_MS - 1
        if end_ms < start_ms:
            end_ms = start_ms - 1
        plans.append(
            SymbolPlan(
                symbol=symbol,
                underlying_type=str(row.get("underlyingType") or row.get("contractType") or ""),
                start_ms=start_ms,
                end_ms=end_ms,
                onboard_ms=onboard_ms,
            )
        )
    return plans


def request_klines(
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    retries: int,
    base_wait_sec: float,
    throttle: GlobalRequestThrottle,
) -> list[list[Any]]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": "1m",
            "startTime": int(start_ms),
            "endTime": int(end_ms),
            "limit": int(limit),
        }
    )
    url = f"{FAPI_BASE_URL}/fapi/v1/klines?{params}"
    attempt = 0
    while True:
        try:
            throttle.wait()
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
            return list(payload or []) if isinstance(payload, list) else []
        except urllib.error.HTTPError as exc:
            attempt += 1
            body = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else ""
            retry_after = float(exc.headers.get("Retry-After") or 0.0)
            if exc.code == 418:
                ban_wait_sec = 0.0
                match = BINANCE_BAN_UNTIL_RE.search(body)
                if match:
                    ban_wait_sec = max(
                        0.0,
                        (int(match.group("until_ms")) - int(datetime.now(UTC).timestamp() * 1000))
                        / 1000.0,
                    )
                wait_sec = max(retry_after, ban_wait_sec + 5.0, 60.0 * min(attempt, 5))
                throttle.pause(wait_sec)
            elif exc.code == 429:
                wait_sec = max(retry_after, 5.0, float(base_wait_sec) * (2 ** (attempt - 1)))
                throttle.pause(wait_sec)
            else:
                wait_sec = float(base_wait_sec) * (2 ** (attempt - 1))
            if attempt > retries:
                raise RuntimeError(
                    f"{symbol} kline request failed after {retries} retries: "
                    f"HTTP {exc.code}: {exc.reason}; {body[:200]}"
                ) from exc
            time.sleep(wait_sec)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"{symbol} kline request failed after {retries} retries: {exc}")
            time.sleep(float(base_wait_sec) * (2 ** (attempt - 1)))


def rows_to_frame(rows: list[list[Any]]) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        records.append(
            {
                "timestamp_ms": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    if not records:
        return pl.DataFrame(
            schema={
                "timestamp_ms": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    return pl.DataFrame(records)


def floor_utc_day_ms(ms: int) -> int:
    day = datetime.fromtimestamp(ms / 1000.0, tz=UTC).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return int(day.timestamp() * 1000)


def next_utc_day_ms(ms: int) -> int:
    return floor_utc_day_ms(ms) + 86_400_000


def month_start_ms(ms: int) -> int:
    month = datetime.fromtimestamp(ms / 1000.0, tz=UTC).replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return int(month.timestamp() * 1000)


def next_month_ms(ms: int) -> int:
    current = datetime.fromtimestamp(month_start_ms(ms) / 1000.0, tz=UTC)
    if current.month == 12:
        nxt = current.replace(year=current.year + 1, month=1)
    else:
        nxt = current.replace(month=current.month + 1)
    return int(nxt.timestamp() * 1000)


def latest_data_vision_until_ms() -> int:
    """Return the latest conservative full UTC day likely available on data.binance.vision."""
    return floor_utc_day_ms(int(datetime.now(UTC).timestamp() * 1000)) - 86_400_000 - 1


def iter_data_vision_specs(since_ms: int, until_ms: int) -> Iterable[tuple[str, str]]:
    """Yield (period, token) specs using monthly files where possible, daily otherwise."""
    cursor = floor_utc_day_ms(since_ms)
    while cursor <= until_ms:
        month_start = month_start_ms(cursor)
        month_end = next_month_ms(cursor) - 1
        if cursor == month_start and since_ms <= month_start and month_end <= until_ms:
            token = datetime.fromtimestamp(cursor / 1000.0, tz=UTC).strftime("%Y-%m")
            yield ("monthly", token)
            cursor = month_end + 1
            continue
        token = datetime.fromtimestamp(cursor / 1000.0, tz=UTC).strftime("%Y-%m-%d")
        yield ("daily", token)
        cursor = next_utc_day_ms(cursor)


def data_vision_url(symbol: str, *, period: str, token: str) -> str:
    file_token = f"{symbol}-1m-{token}.zip"
    return f"https://data.binance.vision/data/futures/um/{period}/klines/{symbol}/1m/{file_token}"


def download_data_vision_zip(
    url: str,
    *,
    retries: int,
    base_wait_sec: float,
    throttle: GlobalRequestThrottle,
) -> bytes | None:
    attempt = 0
    while True:
        try:
            throttle.wait()
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            attempt += 1
            retry_after = float(exc.headers.get("Retry-After") or 0.0)
            wait_sec = max(retry_after, float(base_wait_sec) * (2 ** (attempt - 1)))
            if exc.code in {429, 503}:
                wait_sec = max(wait_sec, 5.0)
                throttle.pause(wait_sec)
            if attempt > retries:
                raise RuntimeError(
                    f"data.vision download failed after {retries} retries: "
                    f"HTTP {exc.code}: {exc.reason}"
                ) from exc
            time.sleep(wait_sec)
        except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile) as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(
                    f"data.vision download failed after {retries} retries: {exc}"
                ) from exc
            time.sleep(float(base_wait_sec) * (2 ** (attempt - 1)))


def data_vision_zip_to_frame(blob: bytes, *, since_ms: int, until_ms: int) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not names:
            return rows_to_frame([])
        csv_bytes = archive.read(names[0])
    frame = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=True,
        columns=["open_time", "open", "high", "low", "close", "volume"],
        schema_overrides={
            "open_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    ).rename({"open_time": "timestamp_ms"})
    return frame.filter(
        (pl.col("timestamp_ms") >= int(since_ms)) & (pl.col("timestamp_ms") <= int(until_ms))
    )


def collect_symbol_data_vision(
    plan: SymbolPlan,
    *,
    db_path: Path,
    exchange: str,
    retries: int,
    base_wait_sec: float,
    throttle: GlobalRequestThrottle,
    dry_run: bool,
) -> SymbolResult:
    started = utc_now_iso()
    if plan.end_ms < plan.start_ms:
        return SymbolResult(
            symbol=plan.symbol,
            status="up_to_date",
            request_count=0,
            fetched_rows=0,
            upserted_rows=0,
            started_at_utc=started,
            completed_at_utc=utc_now_iso(),
            first_timestamp_ms=None,
            last_timestamp_ms=None,
        )
    repo = MarketDataRepository(str(db_path))
    requests = 0
    source_files = 0
    missing_files = 0
    fetched = 0
    upserted = 0
    first_ts: int | None = None
    last_ts: int | None = None
    try:
        for period, token in iter_data_vision_specs(plan.start_ms, plan.end_ms):
            requests += 1
            blob = download_data_vision_zip(
                data_vision_url(plan.symbol, period=period, token=token),
                retries=retries,
                base_wait_sec=base_wait_sec,
                throttle=throttle,
            )
            if blob is None:
                missing_files += 1
                continue
            source_files += 1
            frame = data_vision_zip_to_frame(blob, since_ms=plan.start_ms, until_ms=plan.end_ms)
            if frame.is_empty():
                continue
            fetched += int(frame.height)
            batch_first = int(frame["timestamp_ms"].min())
            batch_last = int(frame["timestamp_ms"].max())
            first_ts = batch_first if first_ts is None else min(first_ts, batch_first)
            last_ts = batch_last if last_ts is None else max(last_ts, batch_last)
            if not dry_run:
                upserted += int(
                    repo.upsert_ohlcv(
                        exchange=exchange,
                        symbol=plan.symbol,
                        timeframe="1m",
                        rows=frame,
                    )
                )
        status = "ok" if fetched else "empty"
        return SymbolResult(
            symbol=plan.symbol,
            status=status,
            request_count=requests,
            fetched_rows=fetched,
            upserted_rows=upserted,
            started_at_utc=started,
            completed_at_utc=utc_now_iso(),
            first_timestamp_ms=first_ts,
            last_timestamp_ms=last_ts,
            source_files=source_files,
            missing_files=missing_files,
        )
    except Exception as exc:
        return SymbolResult(
            symbol=plan.symbol,
            status="error",
            request_count=requests,
            fetched_rows=fetched,
            upserted_rows=upserted,
            started_at_utc=started,
            completed_at_utc=utc_now_iso(),
            first_timestamp_ms=first_ts,
            last_timestamp_ms=last_ts,
            source_files=source_files,
            missing_files=missing_files,
            error=str(exc),
        )


def collect_symbol(
    plan: SymbolPlan,
    *,
    db_path: Path,
    exchange: str,
    limit: int,
    retries: int,
    base_wait_sec: float,
    request_sleep_sec: float,
    throttle: GlobalRequestThrottle,
    dry_run: bool,
) -> SymbolResult:
    started = utc_now_iso()
    if plan.end_ms < plan.start_ms:
        return SymbolResult(
            symbol=plan.symbol,
            status="up_to_date",
            request_count=0,
            fetched_rows=0,
            upserted_rows=0,
            started_at_utc=started,
            completed_at_utc=utc_now_iso(),
            first_timestamp_ms=None,
            last_timestamp_ms=None,
        )
    repo = MarketDataRepository(str(db_path))
    cursor = int(plan.start_ms)
    requests = 0
    fetched = 0
    upserted = 0
    first_ts: int | None = None
    last_ts: int | None = None
    status = "ok"
    try:
        while cursor <= plan.end_ms:
            batch_end = min(plan.end_ms, cursor + int(limit) * KLINE_INTERVAL_MS - 1)
            requests += 1
            rows = request_klines(
                symbol=plan.symbol,
                start_ms=cursor,
                end_ms=batch_end,
                limit=limit,
                retries=retries,
                base_wait_sec=base_wait_sec,
                throttle=throttle,
            )
            if not rows:
                cursor = batch_end + 1
                if request_sleep_sec > 0:
                    time.sleep(request_sleep_sec)
                continue
            frame = rows_to_frame(rows)
            if not frame.is_empty():
                fetched += int(frame.height)
                batch_first = int(frame["timestamp_ms"].min())
                batch_last = int(frame["timestamp_ms"].max())
                first_ts = batch_first if first_ts is None else min(first_ts, batch_first)
                last_ts = batch_last if last_ts is None else max(last_ts, batch_last)
                if not dry_run:
                    upserted += int(
                        repo.upsert_ohlcv(
                            exchange=exchange,
                            symbol=plan.symbol,
                            timeframe="1m",
                            rows=frame,
                        )
                    )
                cursor = max(batch_last + KLINE_INTERVAL_MS, batch_end + 1)
            else:
                cursor = batch_end + 1
            if request_sleep_sec > 0:
                time.sleep(request_sleep_sec)
        if fetched == 0:
            status = "empty"
        return SymbolResult(
            symbol=plan.symbol,
            status=status,
            request_count=requests,
            fetched_rows=fetched,
            upserted_rows=upserted,
            started_at_utc=started,
            completed_at_utc=utc_now_iso(),
            first_timestamp_ms=first_ts,
            last_timestamp_ms=last_ts,
        )
    except Exception as exc:
        return SymbolResult(
            symbol=plan.symbol,
            status="error",
            request_count=requests,
            fetched_rows=fetched,
            upserted_rows=upserted,
            started_at_utc=started,
            completed_at_utc=utc_now_iso(),
            first_timestamp_ms=first_ts,
            last_timestamp_ms=last_ts,
            error=str(exc),
        )


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument(
        "--source",
        choices=("data-vision", "fapi"),
        default="data-vision",
        help="Historical source. data-vision avoids Binance REST weight limits; fapi uses /fapi/v1/klines.",
    )
    out.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    out.add_argument("--exchange", default="binance")
    out.add_argument("--since", default=DEFAULT_SINCE_UTC)
    out.add_argument(
        "--until",
        default=None,
        help=(
            "UTC ISO; default is latest full UTC day for data-vision, latest closed minute for fapi"
        ),
    )
    out.add_argument("--symbols", nargs="*", default=None)
    out.add_argument("--workers", type=int, default=4)
    out.add_argument("--limit", type=int, default=1500)
    out.add_argument("--retries", type=int, default=4)
    out.add_argument("--base-wait-sec", type=float, default=0.5)
    out.add_argument("--request-sleep-sec", type=float, default=0.02)
    out.add_argument(
        "--global-request-interval-sec",
        type=float,
        default=1.0,
        help="Minimum interval between Binance kline requests across all workers.",
    )
    out.add_argument("--no-resume", action="store_true")
    out.add_argument("--dry-run", action="store_true")
    out.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    db_path = Path(args.db_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    since_ms = parse_utc_ms(args.since)
    if args.until:
        until_raw_ms = parse_utc_ms(args.until)
        until_ms = (until_raw_ms // KLINE_INTERVAL_MS) * KLINE_INTERVAL_MS + KLINE_INTERVAL_MS - 1
    elif args.source == "data-vision":
        until_ms = latest_data_vision_until_ms()
    else:
        until_ms = (
            int(datetime.now(UTC).timestamp() * 1000) // KLINE_INTERVAL_MS
        ) * KLINE_INTERVAL_MS - 1
    if args.source == "fapi":
        exchange_info = {} if args.symbols else load_exchange_info()
        symbols = [
            compact_symbol(item)
            for item in (args.symbols or default_symbols_from_exchange_info(exchange_info))
        ]
        plans = make_symbol_plans(
            symbols,
            exchange_info=exchange_info,
            since_ms=since_ms,
            until_ms=until_ms,
            db_path=db_path,
            exchange=str(args.exchange),
            resume=not bool(args.no_resume),
        )
    else:
        symbols = [
            compact_symbol(item) for item in (args.symbols or BINANCE_EXTENDED_RESEARCH_SYMBOLS)
        ]
        plans = make_symbol_plans(
            symbols,
            exchange_info={},
            since_ms=since_ms,
            until_ms=until_ms,
            db_path=db_path,
            exchange=str(args.exchange),
            resume=not bool(args.no_resume),
        )

    run_started = utc_now_iso()
    throttle = GlobalRequestThrottle(min_interval_sec=float(args.global_request_interval_sec))
    print(
        json.dumps(
            {
                "event": "start",
                "source": str(args.source),
                "symbol_count": len(plans),
                "since_utc": ms_to_iso(since_ms),
                "until_utc": ms_to_iso(until_ms),
                "workers": int(args.workers),
                "global_request_interval_sec": float(args.global_request_interval_sec),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    results: list[SymbolResult] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        if args.source == "fapi":
            futures = {
                pool.submit(
                    collect_symbol,
                    plan,
                    db_path=db_path,
                    exchange=str(args.exchange),
                    limit=max(1, min(int(args.limit), 1500)),
                    retries=int(args.retries),
                    base_wait_sec=float(args.base_wait_sec),
                    request_sleep_sec=float(args.request_sleep_sec),
                    throttle=throttle,
                    dry_run=bool(args.dry_run),
                ): plan
                for plan in plans
            }
        else:
            futures = {
                pool.submit(
                    collect_symbol_data_vision,
                    plan,
                    db_path=db_path,
                    exchange=str(args.exchange),
                    retries=int(args.retries),
                    base_wait_sec=float(args.base_wait_sec),
                    throttle=throttle,
                    dry_run=bool(args.dry_run),
                ): plan
                for plan in plans
            }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with PRINT_LOCK:
                print(
                    json.dumps({"event": "symbol_done", **asdict(result)}, sort_keys=True),
                    flush=True,
                )

    results_sorted = sorted(results, key=lambda item: item.symbol)
    payload = {
        "artifact_kind": "binance_1m_research_universe_collection",
        "generated_at_utc": utc_now_iso(),
        "started_at_utc": run_started,
        "completed_at_utc": utc_now_iso(),
        "db_path": str(db_path),
        "exchange": str(args.exchange),
        "timeframe": "1m",
        "source": (
            "Binance public data.vision USD-M Futures klines"
            if args.source == "data-vision"
            else "Binance USD-M Futures /fapi/v1/klines"
        ),
        "source_mode": str(args.source),
        "since_utc": ms_to_iso(since_ms),
        "until_utc": ms_to_iso(until_ms),
        "symbol_count": len(plans),
        "dry_run": bool(args.dry_run),
        "workers": int(args.workers),
        "global_request_interval_sec": float(args.global_request_interval_sec),
        "request_limit": int(args.limit),
        "summary": {
            "ok_count": sum(1 for item in results_sorted if item.status == "ok"),
            "empty_count": sum(1 for item in results_sorted if item.status == "empty"),
            "up_to_date_count": sum(1 for item in results_sorted if item.status == "up_to_date"),
            "error_count": sum(1 for item in results_sorted if item.status == "error"),
            "request_count": sum(item.request_count for item in results_sorted),
            "source_files": sum(item.source_files for item in results_sorted),
            "missing_files": sum(item.missing_files for item in results_sorted),
            "fetched_rows": sum(item.fetched_rows for item in results_sorted),
            "upserted_rows": sum(item.upserted_rows for item in results_sorted),
        },
        "symbols": [asdict(item) for item in results_sorted],
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"binance_1m_research_universe_collection_{stamp}.json"
    latest_path = report_dir / "binance_1m_research_universe_collection_latest.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"event": "report", "path": str(json_path), "latest": str(latest_path)}),
        flush=True,
    )
    return 1 if payload["summary"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
