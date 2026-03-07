#!/usr/bin/env python3
"""Materialize committed OHLCV windows from raw aggTrades parquet source."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.config import BaseConfig
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.services.materialize_from_raw import materialize_raw_aggtrades_bundle
from lumina_quant.storage.parquet import ParquetMarketDataRepository


def _parse_symbols(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return list(BaseConfig.SYMBOLS)
    out: list[str] = []
    for item in raw.split(","):
        token = str(item).strip()
        if token:
            out.append(token)
    return out or list(BaseConfig.SYMBOLS)


def _parse_timeframes(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return list(BaseConfig.MATERIALIZER_REQUIRED_TIMEFRAMES)
    out: list[str] = []
    for item in raw.split(","):
        token = str(item).strip().lower()
        if token:
            out.append(token)
    deduped = list(dict.fromkeys(out))
    if "1s" in deduped and deduped[0] != "1s":
        deduped = ["1s", *[token for token in deduped if token != "1s"]]
    return deduped or list(BaseConfig.MATERIALIZER_REQUIRED_TIMEFRAMES)


def _normalize_timeframes(values: list[str]) -> list[str]:
    normalized = [normalize_timeframe_token(str(value)) for value in list(values or []) if str(value).strip()]
    deduped = list(dict.fromkeys(normalized))
    if "1s" in deduped and deduped[0] != "1s":
        deduped = ["1s", *[token for token in deduped if token != "1s"]]
    return deduped or list(BaseConfig.MATERIALIZER_REQUIRED_TIMEFRAMES)


def _coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value or 0)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _isoformat_utc_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")


def _resolve_symbol_materialization_window(
    *,
    db_path: str,
    exchange: str,
    symbol: str,
    required_timeframes: list[str],
    start_date: str | None,
    end_date: str | None,
    incremental_window: bool,
) -> tuple[str | None, str | None]:
    if not incremental_window or start_date is not None or end_date is not None:
        return start_date, end_date

    repo = ParquetMarketDataRepository(str(db_path))
    latest_manifest = repo.read_latest_materialized_manifest(
        exchange=str(exchange),
        symbol=str(symbol),
        timeframe="1s",
    )
    if not latest_manifest:
        return None, None

    anchor_ms = _coerce_positive_int(latest_manifest.get("source_checkpoint_end"))
    if anchor_ms is None:
        anchor_ms = _coerce_positive_int(latest_manifest.get("event_time_watermark_ms"))
    if anchor_ms is None:
        return None, None

    earliest_bucket_ms = int(anchor_ms)
    for timeframe in _normalize_timeframes(required_timeframes):
        tf_ms = max(1, int(timeframe_to_milliseconds(timeframe)))
        earliest_bucket_ms = min(earliest_bucket_ms, (int(anchor_ms) // tf_ms) * tf_ms)

    day_ms = 24 * 60 * 60 * 1000
    earliest_partition_start_ms = (int(earliest_bucket_ms) // day_ms) * day_ms
    return _isoformat_utc_from_ms(earliest_partition_start_ms), None


def run_materializer_cycle(
    *,
    db_path: str,
    exchange: str,
    symbols: list[str],
    required_timeframes: list[str],
    base_timeframe: str = "1s",
    start_date: str | None,
    end_date: str | None,
    producer: str,
    incremental_window: bool = True,
) -> dict[str, object]:
    if str(base_timeframe or "").strip().lower() != "1s":
        raise ValueError("storage.materializer_base_timeframe must be '1s'.")

    symbol_results: list[dict[str, object]] = []
    success = True

    for symbol in symbols:
        try:
            resolved_start_date, resolved_end_date = _resolve_symbol_materialization_window(
                db_path=str(db_path),
                exchange=str(exchange),
                symbol=str(symbol),
                required_timeframes=list(required_timeframes),
                start_date=start_date,
                end_date=end_date,
                incremental_window=bool(incremental_window),
            )
            bundle_id = f"bundle-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}-{symbol.replace('/', '-')}"
            result = materialize_raw_aggtrades_bundle(
                root_path=str(db_path),
                exchange=str(exchange),
                symbol=str(symbol),
                timeframes=list(required_timeframes),
                start_date=resolved_start_date,
                end_date=resolved_end_date,
                producer=str(producer),
                require_complete=True,
            )
            commits_by_timeframe = (
                dict(result.commits_by_timeframe)
                if hasattr(result, "commits_by_timeframe")
                else dict(result)
            )
            symbol_results.append(
                {
                    "symbol": str(symbol),
                    "status": "committed",
                    "bundle_id": bundle_id,
                    "timeframes": {
                        timeframe: [
                            {
                                "partition": commit.partition,
                                "commit_id": commit.commit_id,
                                "row_count": commit.row_count,
                                "canonical_row_checksum": commit.canonical_row_checksum,
                                "manifest_path": commit.manifest_path,
                            }
                            for commit in commits
                        ]
                        for timeframe, commits in commits_by_timeframe.items()
                    },
                }
            )
        except RawFirstDataMissingError as exc:
            success = False
            symbol_results.append(
                {
                    "symbol": str(symbol),
                    "status": "skipped_incomplete_required_timeframes",
                    "error": str(exc),
                }
            )

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "db_path": str(db_path),
        "exchange": str(exchange),
        "required_timeframes": list(required_timeframes),
        "success": bool(success),
        "symbols": symbol_results,
    }


def run_materializer_periodic_loop(
    *,
    db_path: str,
    exchange: str,
    symbols: list[str],
    required_timeframes: list[str],
    base_timeframe: str = "1s",
    start_date: str | None,
    end_date: str | None,
    producer: str,
    periodic_enabled: bool,
    poll_seconds: int,
    max_cycles: int | None = None,
    incremental_window: bool = True,
    sleep_fn=time.sleep,
) -> list[dict[str, object]]:
    cycles: list[dict[str, object]] = []
    cycle = 0

    while True:
        cycle += 1
        payload = run_materializer_cycle(
            db_path=db_path,
            exchange=exchange,
            symbols=symbols,
            required_timeframes=required_timeframes,
            base_timeframe=base_timeframe,
            start_date=start_date,
            end_date=end_date,
            producer=producer,
            incremental_window=incremental_window,
        )
        payload["cycle"] = int(cycle)
        cycles.append(payload)
        print(json.dumps(payload, ensure_ascii=False))

        if not periodic_enabled:
            break
        if max_cycles is not None and int(max_cycles) > 0 and cycle >= int(max_cycles):
            break
        sleep_fn(max(1, int(poll_seconds)))

    return cycles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize raw aggTrades into committed OHLCV manifests."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(BaseConfig.SYMBOLS),
        help="Comma-separated symbol list (e.g. BTC/USDT,ETH/USDT).",
    )
    parser.add_argument(
        "--exchange",
        default=str(BaseConfig.MARKET_DATA_EXCHANGE),
        help="Exchange partition token.",
    )
    parser.add_argument(
        "--required-timeframes",
        default=",".join(BaseConfig.MATERIALIZER_REQUIRED_TIMEFRAMES),
        help="Comma-separated materializer required timeframes.",
    )
    parser.add_argument(
        "--timeframes",
        default="",
        help="Legacy alias for --required-timeframes.",
    )
    parser.add_argument(
        "--base-timeframe",
        default=str(getattr(BaseConfig, "MATERIALIZER_BASE_TIMEFRAME", "1s")),
        help="Materializer base timeframe (must stay 1s).",
    )
    parser.add_argument(
        "--db-path",
        default=str(BaseConfig.MARKET_DATA_PARQUET_PATH),
        help="Parquet market-data root path.",
    )
    parser.add_argument("--start-date", default=None, help="Optional start date/time.")
    parser.add_argument("--end-date", default=None, help="Optional end date/time.")
    parser.add_argument(
        "--producer",
        default="materialize_market_windows.py",
        help="Manifest producer identifier.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=int(BaseConfig.MATERIALIZER_POLL_SECONDS),
        help="Periodic materializer cadence in seconds.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Optional max loop count when periodic mode is enabled (0 = run forever).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Alias for --max-cycles.",
    )
    parser.add_argument(
        "--periodic",
        dest="periodic",
        action="store_true",
        default=None,
        help="Force periodic materializer mode.",
    )
    parser.add_argument(
        "--no-periodic",
        dest="periodic",
        action="store_false",
        help="Force one-shot mode.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one materializer cycle and exit.",
    )
    parser.add_argument(
        "--periodic-enabled",
        default="",
        help="Override periodic mode (true/false). Empty = config default.",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help=(
            "Disable incremental partition-safe windowing when --start-date/--end-date are empty. "
            "Default behavior re-reads only the UTC date partitions that can still change based "
            "on the latest committed 1s manifest. Use this for historical backfills or raw "
            "repairs older than the latest committed anchor."
        ),
    )
    return parser.parse_args()


def _resolve_periodic_enabled(raw: str | bool | None, *, once: bool) -> bool:
    if isinstance(raw, bool):
        return bool(raw)
    if once:
        return False
    token = str(raw or "").strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(BaseConfig.MATERIALIZER_PERIODIC_ENABLED)


def main() -> int:
    args = _parse_args()
    periodic_raw = args.periodic if args.periodic is not None else args.periodic_enabled
    periodic_enabled = _resolve_periodic_enabled(periodic_raw, once=bool(args.once))
    max_cycles_raw = args.cycles if args.cycles is not None else args.max_cycles
    max_cycles = int(max_cycles_raw) if int(max_cycles_raw) > 0 else None
    timeframes_raw = str(args.required_timeframes or "").strip()
    if str(args.timeframes or "").strip():
        timeframes_raw = str(args.timeframes)

    run_materializer_periodic_loop(
        db_path=str(args.db_path),
        exchange=str(args.exchange),
        symbols=_parse_symbols(args.symbols),
        required_timeframes=_parse_timeframes(timeframes_raw),
        base_timeframe=str(args.base_timeframe),
        start_date=args.start_date,
        end_date=args.end_date,
        producer=str(args.producer),
        periodic_enabled=periodic_enabled,
        poll_seconds=max(1, int(args.poll_seconds)),
        max_cycles=max_cycles,
        incremental_window=not bool(args.full_rebuild),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
