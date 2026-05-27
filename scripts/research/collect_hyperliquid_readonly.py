#!/usr/bin/env python3
"""Collect Hyperliquid public read-only feature context for profit moonshot."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from lumina_quant.config import BaseConfig
from lumina_quant.data.hyperliquid_readonly import (
    DEFAULT_INFO_URL,
    HyperliquidInfoClient,
    parse_candle_snapshot,
    parse_funding_history_page,
    parse_meta_asset_context_rows,
)
from lumina_quant.market_data import upsert_futures_feature_points_rows

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "var/reports/profit_moonshot_20260501/current_tail_20260506/multiasset_exchange_expansion"
)
DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
SPLITS = {
    "train": ("2025-01-01", "2025-12-31"),
    "val": ("2026-01-01", "2026-02-28"),
    "oos": ("2026-03-01", "2026-05-04"),
}


def _parse_day(raw: str) -> date:
    return datetime.fromisoformat(str(raw)).date()


def _start_ms(day: date) -> int:
    return int(datetime.combine(day, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)


def _end_ms(day: date) -> int:
    return int(
        (
            datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
            - timedelta(milliseconds=1)
        ).timestamp()
        * 1000
    )


def _coin(symbol: str) -> str:
    return str(symbol).split("/", 1)[0].upper()


def _iso(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).isoformat()


def _split_count(rows: list[dict[str, Any]], *, start: date, end: date) -> int:
    start_ms = _start_ms(start)
    end_ms = _end_ms(end)
    return sum(start_ms <= int(row["timestamp_ms"]) <= end_ms for row in rows)


def _collect_funding_pages(
    *,
    client: HyperliquidInfoClient,
    coin: str,
    start_time_ms: int,
    end_time_ms: int,
    throttle_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cursor = int(start_time_ms)
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    while cursor <= end_time_ms:
        payload = client.funding_history(coin=coin, start_time_ms=cursor, end_time_ms=end_time_ms)
        page = parse_funding_history_page(coin, payload)
        if not page.rows:
            pages.append({"cursor_ms": cursor, "row_count": 0, "stopped": True})
            break
        rows.extend(page.rows)
        pages.append(
            {
                "cursor_ms": cursor,
                "row_count": len(page.rows),
                "first_timestamp_ms": page.first_timestamp_ms,
                "last_timestamp_ms": page.last_timestamp_ms,
                "first_timestamp_utc": _iso(page.first_timestamp_ms),
                "last_timestamp_utc": _iso(page.last_timestamp_ms),
            }
        )
        last = int(page.last_timestamp_ms or cursor)
        if last < cursor:
            break
        cursor = last + 1
        if len(page.rows) < 500:
            break
        if throttle_seconds > 0.0:
            time.sleep(float(throttle_seconds))
    deduped = {int(row["timestamp_ms"]): row for row in rows}
    return [deduped[key] for key in sorted(deduped)], pages


def _candle_split_summary(
    *,
    client: HyperliquidInfoClient,
    coin: str,
    interval: str,
    throttle_seconds: float,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split_name, (start_raw, end_raw) in SPLITS.items():
        start = _parse_day(start_raw)
        end = _parse_day(end_raw)
        rows = parse_candle_snapshot(
            client.candle_snapshot(
                coin=coin,
                interval=interval,
                start_time_ms=_start_ms(start),
                end_time_ms=_end_ms(end),
            )
        )
        expected_hours = (end - start).days * 24 + 24
        summaries[split_name] = {
            "start": start.isoformat(),
            "end_inclusive": end.isoformat(),
            "rows": len(rows),
            "expected_hourly_rows": expected_hours,
            "complete_hourly_snapshot": len(rows) >= expected_hours,
            "first_timestamp_ms": int(rows[0]["timestamp_ms"]) if rows else None,
            "last_timestamp_ms": int(rows[-1]["timestamp_ms"]) if rows else None,
            "first_timestamp_utc": _iso(int(rows[0]["timestamp_ms"])) if rows else None,
            "last_timestamp_utc": _iso(int(rows[-1]["timestamp_ms"])) if rows else None,
            "note": "Hyperliquid official candleSnapshot is exchange-aggregated context, not repo raw-first trade data.",
        }
        if throttle_seconds > 0.0:
            time.sleep(float(throttle_seconds))
    return summaries


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Hyperliquid read-only collection",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Endpoint: `{payload['endpoint']}`",
        "",
        "## External-doc anchors verified at runtime",
        "",
        "- Hyperliquid public `/info` supports `metaAndAssetCtxs`, `fundingHistory`, `predictedFundings`, and related perp context requests.",
        "- Hyperliquid public `/info` also documents `candleSnapshot`, with only the most recent 5000 candles available.",
        "- Current official base perp fee tier lists taker 0.045% and maker 0.015%; this report does **not** authorize direct Hyperliquid trading.",
        "",
        "## Collection summary",
        "",
        "| symbol | funding rows | first funding | last funding | current mark | current OI | candle train/val/oos rows |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in payload["symbols"]:
        current = row.get("current_context") or {}
        candles = row.get("candle_snapshot_coverage") or {}
        lines.append(
            f"| `{row['symbol']}` | {row['funding_rows']} | `{row.get('first_funding_utc')}` | "
            f"`{row.get('last_funding_utc')}` | {current.get('mark_price')} | {current.get('open_interest')} | "
            f"{candles.get('train', {}).get('rows')}/"
            f"{candles.get('val', {}).get('rows')}/"
            f"{candles.get('oos', {}).get('rows')} |"
        )
    lines.extend(
        [
            "",
            "## Replay eligibility",
            "",
            "- Funding history is usable as a read-only confirmation feature when split coverage is present.",
            "- Historical OI is **not** replay-eligible from `metaAndAssetCtxs`; that endpoint provides current context only.",
            "- Historical mark/candle context is not raw-first and train coverage is partial because the official candle endpoint is capped at recent candles; it is report/context only unless a raw-first/listing-aware source is added.",
            "- Direct trading remains blocked: no Hyperliquid fill/funding/liquidation/orderbook parity model was promoted.",
        ]
    )
    return "\n".join(lines) + "\n"


def collect_hyperliquid(
    *,
    market_root: Path,
    output_dir: Path,
    symbols: list[str],
    start_date: date,
    end_date: date,
    interval: str,
    throttle_seconds: float,
    write_feature_points: bool,
) -> dict[str, Any]:
    client = HyperliquidInfoClient()
    generated_ms = int(datetime.now(UTC).timestamp() * 1000)
    context_payload = client.meta_and_asset_contexts()
    context_rows = parse_meta_asset_context_rows(
        context_payload, symbols=symbols, timestamp_ms=generated_ms
    )
    context_by_symbol = {str(row["symbol"]): row for row in context_rows}
    symbol_payloads: list[dict[str, Any]] = []
    upserted_total = 0
    for symbol in symbols:
        coin = _coin(symbol)
        funding_rows, pages = _collect_funding_pages(
            client=client,
            coin=coin,
            start_time_ms=_start_ms(start_date),
            end_time_ms=_end_ms(end_date),
            throttle_seconds=throttle_seconds,
        )
        current_context = context_by_symbol.get(symbol)
        feature_rows = [
            {key: value for key, value in row.items() if key in {"timestamp_ms", "funding_rate"}}
            for row in funding_rows
        ]
        if current_context:
            feature_rows.append(
                {
                    "timestamp_ms": int(current_context["timestamp_ms"]),
                    "funding_rate": current_context.get("funding_rate"),
                    "funding_mark_price": current_context.get("funding_mark_price"),
                    "mark_price": current_context.get("mark_price"),
                    "index_price": current_context.get("index_price"),
                    "open_interest": current_context.get("open_interest"),
                }
            )
        if write_feature_points and feature_rows:
            upserted_total += upsert_futures_feature_points_rows(
                str(market_root),
                exchange="hyperliquid",
                symbol=symbol,
                rows=feature_rows,
                source="hyperliquid_info_readonly",
            )
        split_counts = {
            split_name: _split_count(
                funding_rows, start=_parse_day(start_raw), end=_parse_day(end_raw)
            )
            for split_name, (start_raw, end_raw) in SPLITS.items()
        }
        candle_summary = _candle_split_summary(
            client=client,
            coin=coin,
            interval=interval,
            throttle_seconds=throttle_seconds,
        )
        symbol_payloads.append(
            {
                "symbol": symbol,
                "coin": coin,
                "funding_rows": len(funding_rows),
                "funding_pages": pages,
                "first_funding_timestamp_ms": int(funding_rows[0]["timestamp_ms"])
                if funding_rows
                else None,
                "last_funding_timestamp_ms": int(funding_rows[-1]["timestamp_ms"])
                if funding_rows
                else None,
                "first_funding_utc": _iso(int(funding_rows[0]["timestamp_ms"]))
                if funding_rows
                else None,
                "last_funding_utc": _iso(int(funding_rows[-1]["timestamp_ms"]))
                if funding_rows
                else None,
                "funding_split_counts": split_counts,
                "current_context": {
                    key: current_context.get(key)
                    for key in (
                        "timestamp_ms",
                        "funding_rate",
                        "mark_price",
                        "index_price",
                        "open_interest",
                    )
                }
                if current_context
                else None,
                "candle_snapshot_coverage": candle_summary,
            }
        )
    payload: dict[str, Any] = {
        "artifact_kind": "hyperliquid_readonly_collection",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "endpoint": DEFAULT_INFO_URL,
        "market_root": str(market_root),
        "exchange_id": "hyperliquid",
        "symbols": symbol_payloads,
        "upserted_feature_point_rows": int(upserted_total),
        "read_only": True,
        "direct_trading_blocked": True,
        "direct_trading_blocked_reason": "No direct trading until cost/funding/fill/session/liquidation/lot-size models and raw-first evidence exist.",
        "official_docs": {
            "perps_info": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals",
            "info_endpoint": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint",
            "fees": "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hyperliquid_readonly_collection_latest.json"
    md_path = output_dir / "hyperliquid_readonly_collection_latest.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"payload": payload, "paths": {"json": str(json_path), "markdown": str(md_path)}}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-05-04")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--throttle-seconds", type=float, default=0.05)
    parser.add_argument("--no-write-feature-points", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = collect_hyperliquid(
        market_root=Path(args.market_root),
        output_dir=Path(args.output_dir),
        symbols=[str(symbol) for symbol in args.symbols],
        start_date=_parse_day(args.start_date),
        end_date=_parse_day(args.end_date),
        interval=str(args.interval),
        throttle_seconds=float(args.throttle_seconds),
        write_feature_points=not bool(args.no_write_feature_points),
    )
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
