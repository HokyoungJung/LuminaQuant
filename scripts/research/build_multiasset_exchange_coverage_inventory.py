#!/usr/bin/env python3
"""Build raw-first and feature coverage inventory for multiasset exchange expansion."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from lumina_quant.config import BaseConfig
from lumina_quant.data.support_inventory import build_strategy_support_inventory

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "var/reports/profit_moonshot_20260501/current_tail_20260506/multiasset_exchange_expansion"
)
DEFAULT_SYMBOLS = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
)
SPLITS = {
    "train": ("2025-01-01", "2025-12-31"),
    "val": ("2026-01-01", "2026-02-28"),
    "oos": ("2026-03-01", "2026-05-06"),
}


def _parse_day(raw: str) -> date:
    return datetime.fromisoformat(str(raw)).date()


def _date_iter(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _compact(symbol: str) -> str:
    return str(symbol).replace("/", "").upper()


def _committed_day(market_root: Path, *, exchange: str, symbol: str, day: date) -> bool:
    root = (
        market_root
        / "market_data_materialized"
        / str(exchange).lower()
        / _compact(symbol)
        / "timeframe=1s"
        / f"date={day.isoformat()}"
    )
    manifest = root / "manifest.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(payload.get("status") or "").lower() != "committed":
        return False
    return any((root / str(item)).exists() for item in list(payload.get("data_files") or []))


def _legacy_monthly_files(
    market_root: Path, *, exchange: str, symbol: str, start: date, end: date
) -> list[str]:
    months: list[str] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        path = (
            market_root
            / "market_ohlcv_1s"
            / str(exchange).lower()
            / _compact(symbol)
            / f"{cursor.year:04d}-{cursor.month:02d}.parquet"
        )
        if path.exists():
            months.append(str(path))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def _raw_symbol_coverage(market_root: Path, *, exchange: str, symbols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for symbol in symbols:
        split_payload: dict[str, Any] = {}
        for split_name, (start_raw, end_raw) in SPLITS.items():
            start = _parse_day(start_raw)
            end = _parse_day(end_raw)
            days = _date_iter(start, end)
            missing = [
                day.isoformat()
                for day in days
                if not _committed_day(market_root, exchange=exchange, symbol=symbol, day=day)
            ]
            split_payload[split_name] = {
                "start": start.isoformat(),
                "end_inclusive": end.isoformat(),
                "required_day_count": len(days),
                "committed_day_count": len(days) - len(missing),
                "missing_day_count": len(missing),
                "first_missing_days": missing[:10],
                "complete_raw_first": not missing,
                "legacy_monthly_files": _legacy_monthly_files(
                    market_root, exchange=exchange, symbol=symbol, start=start, end=end
                )[:12],
            }
        out[symbol] = split_payload
    return out


def _safe_common_oos_end(raw_coverage: dict[str, Any], *, symbols: list[str]) -> str | None:
    candidates: list[date] = []
    for symbol in symbols:
        first_missing = list(
            raw_coverage.get(symbol, {}).get("oos", {}).get("first_missing_days") or []
        )
        if first_missing:
            candidates.append(_parse_day(first_missing[0]) - timedelta(days=1))
        else:
            candidates.append(_parse_day(SPLITS["oos"][1]))
    return min(candidates).isoformat() if candidates else None


def _feature_inventory(market_root: Path, *, exchange: str, symbols: list[str]) -> dict[str, Any]:
    root = market_root / "feature_points" / f"exchange={str(exchange).lower()}"
    if not root.exists():
        return {
            "exchange_id": str(exchange).lower(),
            "db_path": str(market_root),
            "symbol_count": len(symbols),
            "symbols": [
                {
                    "symbol": _compact(symbol),
                    "rows": 0,
                    "blocked": True,
                    "blocked_reason": f"feature_points exchange={str(exchange).lower()} not present",
                }
                for symbol in symbols
            ],
        }
    return build_strategy_support_inventory(
        db_path=str(market_root), exchange=exchange, symbols=symbols
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Multiasset exchange expansion coverage inventory",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Market root: `{payload['market_root']}`",
        f"Safe OOS end for BTC/ETH/SOL raw-first replay/backtest: `{payload['safe_crypto_oos_end_date']}`",
        "",
        "## Raw-first OHLCV coverage",
        "",
        "| symbol | train | val | oos | first OOS missing |",
        "|---|---:|---:|---:|---|",
    ]
    for symbol, splits in payload["raw_first_coverage"].items():
        train = splits["train"]
        val = splits["val"]
        oos = splits["oos"]
        lines.append(
            f"| `{symbol}` | {train['committed_day_count']}/{train['required_day_count']} | "
            f"{val['committed_day_count']}/{val['required_day_count']} | "
            f"{oos['committed_day_count']}/{oos['required_day_count']} | "
            f"`{','.join(oos['first_missing_days'][:3])}` |"
        )
    lines.extend(["", "## Feature inventory", ""])
    for exchange, inventory in payload["feature_inventories"].items():
        lines.extend(
            [
                f"### `{exchange}`",
                "",
                "| symbol | rows | funding | mark | index | OI | taker-flow | liquidation | last timestamp |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in inventory.get("symbols", []):
            lines.append(
                f"| `{row.get('symbol')}` | {int(row.get('rows') or 0)} | "
                f"{int(row.get('funding_rows') or 0)} | {int(row.get('mark_rows') or 0)} | "
                f"{int(row.get('index_rows') or 0)} | {int(row.get('open_interest_rows') or 0)} | "
                f"{int(row.get('taker_flow_rows') or 0)} | {int(row.get('liquidation_rows') or 0)} | "
                f"`{row.get('last_timestamp_utc')}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decisions",
            "",
            "- BTC/ETH/SOL Binance raw-first train/val are usable; current-tail OOS must stop at the safe complete date if the latest configured OOS day is missing.",
            "- Hyperliquid and Tickmill are feature/regime sources only. Direct trading is blocked until spread/swap/funding/fill/session/lot-size models and raw-first evidence exist.",
            "- Tickmill/MT5 coverage remains blocked when the MT5 bridge is not configured; the Tickmill collector report records this separately.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_inventory(
    *, market_root: Path, exchange: str, output_dir: Path, symbols: list[str]
) -> dict[str, Any]:
    raw_coverage = _raw_symbol_coverage(market_root, exchange=exchange, symbols=symbols)
    crypto_symbols = [
        symbol for symbol in symbols if symbol in {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
    ]
    payload: dict[str, Any] = {
        "artifact_kind": "multiasset_exchange_coverage_inventory",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "market_root": str(market_root),
        "raw_exchange": str(exchange).lower(),
        "split_windows": {
            key: {"start": value[0], "end_inclusive": value[1]} for key, value in SPLITS.items()
        },
        "safe_crypto_oos_end_date": _safe_common_oos_end(raw_coverage, symbols=crypto_symbols),
        "raw_first_coverage": raw_coverage,
        "feature_inventories": {
            "binance": _feature_inventory(market_root, exchange="binance", symbols=symbols),
            "hyperliquid": _feature_inventory(
                market_root, exchange="hyperliquid", symbols=crypto_symbols
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "coverage_inventory_latest.json"
    md_path = output_dir / "coverage_inventory_latest.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"payload": payload, "paths": {"json": str(json_path), "markdown": str(md_path)}}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_inventory(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        output_dir=Path(args.output_dir),
        symbols=[str(symbol) for symbol in args.symbols],
    )
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
