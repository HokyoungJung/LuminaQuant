#!/usr/bin/env python3
"""Memory-bounded raw-first data backfill for live-equivalent validation.

The live-equivalent candidate validator requires committed raw-first 1s
materialized manifests for train/validation windows.  The normal materializer
can load a very large range at once, so this helper backfills raw aggTrades and
materializes in bounded chunks with a child-process RSS guard.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any

from lumina_quant.configuration import BacktestConfigView, get_default_runtime_config
from lumina_quant.data_sync import create_binance_futures_client, sync_symbol_aggtrades_raw

BaseConfig = BacktestConfigView(get_default_runtime_config())

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped"
    / "live_equivalent_data_backfill_20260426"
    / "raw_first_backfill_train_val_latest.json"
)


@dataclass(frozen=True, slots=True)
class DateRun:
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def as_payload(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "days": self.days}


def _symbol_token(symbol: str) -> str:
    return str(symbol).replace("/", "").strip().upper()


def _parse_symbols(value: str) -> list[str]:
    out = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return out or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value).strip()[:10])


def _day_start(day: date) -> datetime:
    return datetime.combine(day, datetime_time.min, tzinfo=UTC)


def _day_end(day: date) -> datetime:
    return datetime.combine(day, datetime_time.max, tzinfo=UTC)


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _iter_days(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _group_days(days: list[date], *, max_days: int) -> list[DateRun]:
    ordered = sorted(set(days))
    if not ordered:
        return []
    runs: list[DateRun] = []
    start = prev = ordered[0]
    for day in ordered[1:]:
        contiguous = day == prev + timedelta(days=1)
        within_limit = (day - start).days + 1 <= max(1, int(max_days))
        if contiguous and within_limit:
            prev = day
            continue
        runs.append(DateRun(start, prev))
        start = prev = day
    runs.append(DateRun(start, prev))
    return runs


def _raw_days(root: Path, *, exchange: str, symbol: str) -> set[date]:
    symbol_root = root / "market_data_raw_aggtrades" / str(exchange).lower() / _symbol_token(symbol)
    out: set[date] = set()
    for path in symbol_root.glob("date=*/part-*.parquet"):
        for part in path.parts:
            if part.startswith("date="):
                try:
                    out.add(date.fromisoformat(part.split("=", 1)[1]))
                except ValueError:
                    pass
    return out


def _manifest_days(root: Path, *, exchange: str, symbol: str, timeframe: str) -> set[date]:
    symbol_root = (
        root
        / "market_data_materialized"
        / str(exchange).lower()
        / _symbol_token(symbol)
        / f"timeframe={timeframe}"
    )
    out: set[date] = set()
    for path in symbol_root.glob("date=*/manifest.json"):
        for part in path.parts:
            if part.startswith("date="):
                try:
                    out.add(date.fromisoformat(part.split("=", 1)[1]))
                except ValueError:
                    pass
    return out


def _rss_kb(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return 0
    for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return 0
    return 0


def _last_json_payload(text: str) -> dict[str, Any] | None:
    for line in reversed(str(text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _materializer_payload_committed(
    payload: dict[str, Any] | None, *, symbol: str, run: DateRun
) -> bool:
    """Return true when materializer stdout proves the requested day(s) committed.

    Some parquet/arrow shutdown paths can print a complete successful JSON
    payload and then abort during interpreter cleanup (for example
    ``free(): invalid pointer``).  Treat that as success only when the payload
    names the requested symbol, all requested 1s partitions, and committed
    manifest files that exist on disk.
    """
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return False
    required_partitions = {day.isoformat() for day in _iter_days(run.start, run.end)}
    for item in payload.get("symbols") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("symbol") or "") != str(symbol):
            continue
        if str(item.get("status") or "").strip().lower() != "committed":
            continue
        commits = (item.get("timeframes") or {}).get("1s") or []
        committed_partitions: set[str] = set()
        for commit in commits:
            if not isinstance(commit, dict):
                continue
            partition = str(commit.get("partition") or "")
            manifest = str(commit.get("manifest_path") or "")
            if not partition or not manifest:
                continue
            manifest_path = Path(manifest)
            if not manifest_path.is_absolute():
                manifest_path = REPO_ROOT / manifest_path
            if manifest_path.exists():
                committed_partitions.add(partition)
        return required_partitions.issubset(committed_partitions)
    return False


def _run_raw_child(
    *,
    symbol: str,
    db_path: Path,
    exchange: str,
    run: DateRun,
    retries: int,
    base_wait_sec: float,
) -> dict[str, Any]:
    client = create_binance_futures_client(
        api_key="",
        secret_key="",
        market_type="future",
        testnet=False,
    )
    try:
        started = time.monotonic()
        stats = sync_symbol_aggtrades_raw(
            exchange=client,
            db_path=str(db_path),
            exchange_id=exchange,
            symbol=symbol,
            start_ms=_to_ms(_day_start(run.start)),
            end_ms=_to_ms(_day_end(run.end)),
            limit=1000,
            max_batches=max(10, int(run.days) + 5),
            retries=max(0, int(retries)),
            base_wait_sec=max(0.05, float(base_wait_sec)),
            resume_from_checkpoint=False,
        )
        return {
            **run.as_payload(),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "fetched_rows": int(stats.fetched_rows),
            "upserted_rows": int(stats.upserted_rows),
            "first_timestamp_ms": stats.first_timestamp_ms,
            "last_timestamp_ms": stats.last_timestamp_ms,
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _run_raw_guarded(
    *,
    symbol: str,
    db_path: Path,
    exchange: str,
    run: DateRun,
    retries: int,
    base_wait_sec: float,
    max_rss_mb: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--raw-child",
        "--symbols",
        symbol,
        "--db-path",
        str(db_path),
        "--exchange-id",
        str(exchange),
        "--start",
        run.start.isoformat(),
        "--end",
        run.end.isoformat(),
        "--retries",
        str(max(0, int(retries))),
        "--base-wait-sec",
        str(max(0.05, float(base_wait_sec))),
    ]
    started = time.monotonic()
    env = os.environ.copy()
    env.setdefault("LQ_RAW_ARCHIVE_CHUNK_ROWS", "250000")
    # Keep historical archive backfills append-only and memory-bounded.  Raw
    # partition compaction can load a full high-volume day after several chunk
    # appends, which defeats the session-wide 8 GB memory ceiling.
    env.setdefault("LQ_RAW_PARTITION_MAX_PARTS", "512")
    env.setdefault("LQ_RAW_COMPACT_ON_THRESHOLD", "false")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_rss_kb = 0
    killed_for_memory = False
    threshold_kb = max(256, int(max_rss_mb)) * 1024
    while proc.poll() is None:
        rss = _rss_kb(proc.pid)
        peak_rss_kb = max(peak_rss_kb, rss)
        if rss > threshold_kb:
            killed_for_memory = True
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "phase": "raw_backfill",
                    "symbol": symbol,
                    "run": run.as_payload(),
                    "returncode": proc.returncode,
                    "peak_rss_mb": round(peak_rss_kb / 1024, 3),
                    "killed_for_memory": killed_for_memory,
                    "stdout_tail": stdout[-2000:],
                    "stderr_tail": stderr[-2000:],
                },
                ensure_ascii=False,
            )
        )
    payload = _last_json_payload(stdout)
    if payload is None:
        payload = {**run.as_payload(), "fetched_rows": 0, "upserted_rows": 0}
    payload.update({"elapsed_sec": round(elapsed, 3), "peak_rss_mb": round(peak_rss_kb / 1024, 3)})
    return payload


def _run_materializer_guarded(
    *,
    symbol: str,
    db_path: Path,
    exchange: str,
    run: DateRun,
    producer: str,
    max_rss_mb: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/materialize_market_windows.py"),
        "--once",
        "--symbols",
        symbol,
        "--required-timeframes",
        "1s",
        "--db-path",
        str(db_path),
        "--exchange",
        str(exchange),
        "--start-date",
        _day_start(run.start).isoformat(),
        "--end-date",
        _day_end(run.end).isoformat(),
        "--producer",
        producer,
    ]
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_rss_kb = 0
    killed_for_memory = False
    threshold_kb = max(256, int(max_rss_mb)) * 1024
    while proc.poll() is None:
        rss = _rss_kb(proc.pid)
        peak_rss_kb = max(peak_rss_kb, rss)
        if rss > threshold_kb:
            killed_for_memory = True
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    elapsed = time.monotonic() - started
    payload = _last_json_payload(stdout)
    if proc.returncode != 0:
        if not killed_for_memory and _materializer_payload_committed(
            payload, symbol=symbol, run=run
        ):
            return {
                **run.as_payload(),
                "elapsed_sec": round(elapsed, 3),
                "peak_rss_mb": round(peak_rss_kb / 1024, 3),
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
                "returncode": proc.returncode,
                "nonzero_exit_after_success": True,
            }
        raise RuntimeError(
            json.dumps(
                {
                    "symbol": symbol,
                    "run": run.as_payload(),
                    "returncode": proc.returncode,
                    "peak_rss_mb": round(peak_rss_kb / 1024, 3),
                    "killed_for_memory": killed_for_memory,
                    "stdout_tail": stdout[-2000:],
                    "stderr_tail": stderr[-2000:],
                },
                ensure_ascii=False,
            )
        )
    materializer_success = bool(payload.get("success")) if isinstance(payload, dict) else None
    materializer_statuses: list[str] = []
    if isinstance(payload, dict):
        for item in payload.get("symbols") or []:
            if isinstance(item, dict):
                materializer_statuses.append(str(item.get("status") or ""))
    return {
        **run.as_payload(),
        "elapsed_sec": round(elapsed, 3),
        "peak_rss_mb": round(peak_rss_kb / 1024, 3),
        "materializer_success": materializer_success,
        "materializer_statuses": materializer_statuses,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def build_backfill(
    *,
    db_path: Path,
    exchange_id: str,
    symbols: list[str],
    start: date,
    end: date,
    chunk_days: int,
    raw_chunk_days: int,
    max_rss_mb: int,
    retries: int,
    base_wait_sec: float,
    collect_raw: bool,
    materialize: bool,
    output: Path,
) -> dict[str, Any]:
    required_days = set(_iter_days(start, end))
    symbol_payloads: list[dict[str, Any]] = []
    for symbol in symbols:
        before_raw = _raw_days(db_path, exchange=exchange_id, symbol=symbol)
        missing_raw = sorted(required_days - before_raw)
        raw_runs_payload: list[dict[str, Any]] = []
        if collect_raw:
            for run in _group_days(missing_raw, max_days=raw_chunk_days):
                row = _run_raw_guarded(
                    symbol=symbol,
                    db_path=db_path,
                    exchange=exchange_id,
                    run=run,
                    retries=retries,
                    base_wait_sec=base_wait_sec,
                    max_rss_mb=max_rss_mb,
                )
                raw_runs_payload.append(row)
                print(
                    json.dumps(
                        {"event": "raw_backfill_run", "symbol": symbol, **row}, ensure_ascii=False
                    ),
                    flush=True,
                )

        after_raw = _raw_days(db_path, exchange=exchange_id, symbol=symbol)
        before_mat = _manifest_days(db_path, exchange=exchange_id, symbol=symbol, timeframe="1s")
        missing_mat = sorted(required_days - before_mat)
        materialize_runs_payload: list[dict[str, Any]] = []
        if materialize:
            # Recompute after raw collection, but only materialize days whose
            # manifests are absent.  Chunked child processes keep RSS bounded.
            for run in _group_days(missing_mat, max_days=chunk_days):
                row = _run_materializer_guarded(
                    symbol=symbol,
                    db_path=db_path,
                    exchange=exchange_id,
                    run=run,
                    producer="live_equivalent_data_backfill_20260426",
                    max_rss_mb=max_rss_mb,
                )
                materialize_runs_payload.append(row)
                print(
                    json.dumps(
                        {
                            "event": "materialize_run",
                            "symbol": symbol,
                            **{
                                k: v
                                for k, v in row.items()
                                if k not in {"stdout_tail", "stderr_tail"}
                            },
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        after_mat = _manifest_days(db_path, exchange=exchange_id, symbol=symbol, timeframe="1s")
        payload = {
            "symbol": symbol,
            "required_days": len(required_days),
            "raw_present_before": len(required_days & before_raw),
            "raw_missing_before": len(missing_raw),
            "raw_present_after": len(required_days & after_raw),
            "raw_missing_after": len(required_days - after_raw),
            "materialized_present_before": len(required_days & before_mat),
            "materialized_missing_before": len(missing_mat),
            "materialized_present_after": len(required_days & after_mat),
            "materialized_missing_after": len(required_days - after_mat),
            "first_raw_missing_after": [
                d.isoformat() for d in sorted(required_days - after_raw)[:10]
            ],
            "first_materialized_missing_after": [
                d.isoformat() for d in sorted(required_days - after_mat)[:10]
            ],
            "raw_runs": raw_runs_payload,
            "materialize_runs": materialize_runs_payload,
        }
        symbol_payloads.append(payload)
        print(
            json.dumps(
                {
                    "event": "symbol_complete",
                    **{
                        k: v
                        for k, v in payload.items()
                        if k not in {"raw_runs", "materialize_runs"}
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = {
        "artifact_kind": "live_equivalent_raw_first_backfill",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "db_path": str(db_path),
        "exchange_id": str(exchange_id),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "chunk_days": int(chunk_days),
        "raw_chunk_days": int(raw_chunk_days),
        "max_rss_mb": int(max_rss_mb),
        "symbols": symbol_payloads,
        "success": all(
            item["raw_missing_after"] == 0 and item["materialized_missing_after"] == 0
            for item in symbol_payloads
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange-id", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-02-28")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--raw-chunk-days", type=int, default=7)
    parser.add_argument("--max-rss-mb", type=int, default=7600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--base-wait-sec", type=float, default=0.25)
    parser.add_argument("--no-collect-raw", action="store_true")
    parser.add_argument("--raw-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    if bool(args.raw_child):
        symbols = _parse_symbols(args.symbols)
        if len(symbols) != 1:
            raise SystemExit("--raw-child requires exactly one symbol")
        row = _run_raw_child(
            symbol=symbols[0],
            db_path=Path(args.db_path),
            exchange=str(args.exchange_id).strip().lower(),
            run=DateRun(_parse_date(args.start), _parse_date(args.end)),
            retries=max(0, int(args.retries)),
            base_wait_sec=max(0.05, float(args.base_wait_sec)),
        )
        print(json.dumps(row, ensure_ascii=False))
        return 0

    result = build_backfill(
        db_path=Path(args.db_path),
        exchange_id=str(args.exchange_id).strip().lower(),
        symbols=_parse_symbols(args.symbols),
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        chunk_days=max(1, int(args.chunk_days)),
        raw_chunk_days=max(1, int(args.raw_chunk_days)),
        max_rss_mb=max(256, int(args.max_rss_mb)),
        retries=max(0, int(args.retries)),
        base_wait_sec=max(0.05, float(args.base_wait_sec)),
        collect_raw=not bool(args.no_collect_raw),
        materialize=not bool(args.no_materialize),
        output=Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
