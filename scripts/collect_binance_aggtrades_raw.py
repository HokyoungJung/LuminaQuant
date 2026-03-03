"""Collect Binance aggTrades into raw parquet partitions."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.config import BaseConfig, LiveConfig
from lumina_quant.data_collector import collect_binance_aggtrades_raw
from lumina_quant.data_sync import parse_timestamp_input
from lumina_quant.storage.parquet import ParquetMarketDataRepository


def _parse_symbols(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return list(BaseConfig.SYMBOLS)
    out = []
    for item in raw.split(","):
        token = str(item).strip()
        if token:
            out.append(token)
    return out or list(BaseConfig.SYMBOLS)


def _resolve_periodic(value: bool | None) -> bool:
    if value is None:
        return bool(getattr(BaseConfig, "COLLECTOR_PERIODIC_ENABLED", True))
    return bool(value)


def _write_cycle_checkpoint_snapshot(
    *,
    db_path: str,
    exchange_id: str,
    symbols: list[str],
    cycle_index: int,
) -> str:
    repo = ParquetMarketDataRepository(str(db_path))
    checkpoint_payload = {
        "cycle_index": int(cycle_index),
        "exchange": str(exchange_id),
        "updated_at_utc": datetime.now(tz=UTC).isoformat(),
        "symbols": {
            str(symbol): repo.read_raw_checkpoint(exchange=str(exchange_id), symbol=str(symbol))
            for symbol in symbols
        },
    }
    output_path = Path(db_path) / "raw_collector_cycle_checkpoint.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checkpoint_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def run_collector_periodic_loop(
    *,
    db_path: str,
    exchange_id: str,
    symbols: list[str],
    since_ms: int | None,
    until_ms: int | None,
    limit: int,
    max_batches: int,
    retries: int,
    base_wait_sec: float,
    periodic_enabled: bool,
    poll_seconds: int,
    max_cycles: int,
    sleep_fn=time.sleep,
) -> list[dict[str, object]]:
    cycles: list[dict[str, object]] = []
    cycle_index = 0
    while True:
        cycle_index += 1
        cycle_since_ms = since_ms if cycle_index == 1 else None
        results = []
        for symbol in symbols:
            result = collect_binance_aggtrades_raw(
                db_path=str(db_path),
                exchange_id=str(exchange_id),
                symbol=str(symbol),
                market_type=str(LiveConfig.MARKET_TYPE),
                api_key=str(LiveConfig.BINANCE_API_KEY or ""),
                secret_key=str(LiveConfig.BINANCE_SECRET_KEY or ""),
                testnet=bool(LiveConfig.IS_TESTNET),
                since_ms=cycle_since_ms,
                until_ms=until_ms,
                limit=max(1, int(limit)),
                max_batches=max(1, int(max_batches)),
                retries=max(0, int(retries)),
                base_wait_sec=max(0.05, float(base_wait_sec)),
            )
            results.append(result)
            print(
                f"[RAW][cycle={cycle_index}] {symbol}: fetched={result['fetched_rows']} "
                f"upserted={result['upserted_rows']} last_ts={result['last_timestamp_ms']}"
            )

        checkpoint_snapshot_path = _write_cycle_checkpoint_snapshot(
            db_path=str(db_path),
            exchange_id=str(exchange_id),
            symbols=symbols,
            cycle_index=cycle_index,
        )
        payload = {
            "cycle_index": cycle_index,
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "db_path": str(db_path),
            "exchange_id": str(exchange_id),
            "symbols": symbols,
            "results": results,
            "checkpoint_snapshot": checkpoint_snapshot_path,
        }
        cycles.append(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))

        if (not periodic_enabled) or cycle_index >= max(1, int(max_cycles)):
            break
        sleep_fn(max(1, int(poll_seconds)))
    return cycles


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw Binance aggTrades.")
    parser.add_argument(
        "--db-path",
        default=BaseConfig.MARKET_DATA_PARQUET_PATH,
        help="Parquet market-data root path.",
    )
    parser.add_argument(
        "--exchange-id",
        default=BaseConfig.MARKET_DATA_EXCHANGE,
        help="Exchange token used in storage partitions.",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(BaseConfig.SYMBOLS),
        help="Comma-separated symbol list (e.g. BTC/USDT,ETH/USDT).",
    )
    parser.add_argument(
        "--since",
        default="",
        help="Optional start timestamp (ms/sec/ISO8601). Empty = checkpoint resume.",
    )
    parser.add_argument(
        "--until",
        default="",
        help="Optional end timestamp (ms/sec/ISO8601). Empty = now.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Exchange fetch_trades batch size.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100000,
        help="Maximum pagination batches per symbol.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Per-batch retry count.")
    parser.add_argument(
        "--base-wait-sec",
        type=float,
        default=0.5,
        help="Initial retry backoff seconds.",
    )
    parser.add_argument(
        "--periodic",
        dest="periodic",
        action="store_true",
        default=None,
        help="Run collector in periodic loop mode.",
    )
    parser.add_argument(
        "--no-periodic",
        dest="periodic",
        action="store_false",
        help="Run one collection cycle and exit.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=int(getattr(BaseConfig, "COLLECTOR_POLL_SECONDS", 2)),
        help="Periodic poll interval in seconds.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=1,
        help="Maximum cycles to run (only used for periodic mode).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Legacy alias for --max-cycles.",
    )
    args = parser.parse_args()

    since_ms = parse_timestamp_input(args.since) if str(args.since or "").strip() else None
    until_ms = parse_timestamp_input(args.until) if str(args.until or "").strip() else None
    symbols = _parse_symbols(args.symbols)
    periodic = _resolve_periodic(args.periodic)
    poll_seconds = max(1, int(args.poll_seconds))
    max_cycles_value = args.cycles if args.cycles is not None else args.max_cycles
    max_cycles = max(1, int(max_cycles_value))

    run_collector_periodic_loop(
        db_path=str(args.db_path),
        exchange_id=str(args.exchange_id),
        symbols=symbols,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=max(1, int(args.limit)),
        max_batches=max(1, int(args.max_batches)),
        retries=max(0, int(args.retries)),
        base_wait_sec=max(0.05, float(args.base_wait_sec)),
        periodic_enabled=periodic,
        poll_seconds=poll_seconds,
        max_cycles=max_cycles,
    )


if __name__ == "__main__":
    main()
