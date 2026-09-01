#!/usr/bin/env python3
"""Step-0 feature-coverage audit for the OI and liquidation XS lanes.

Both ``oi-growth-pressure-xs`` (built) and ``liquidation-imbalance-firesale-xs``
(DEFERRED, measurement-first) pre-register the SAME cheapest falsifier: a
pre-backtest coverage audit that runs BEFORE any backtest spend (graveyard
killer-iii, feature-coverage starvation).  This CLI is that audit, dual-purpose:

* OI lane step-0 kill gate: fraction of symbol-days with non-None
  ``open_interest`` points, per symbol group (10 core crypto / 100 TradFi
  perps); ``< 90%`` coverage on a group kills that group's lane immediately.
* Liquidation lane step-0 kill gate: share of symbol-days with non-None
  ``liquidation_long_notional`` AND ``liquidation_short_notional``; ``< 80%``
  kills the deferred liquidation lane before any build/backtest spend.

Given a feature-points parquet root (``--data-root``) plus ``--symbols`` /
``--start`` / ``--end``, the script loads each symbol's feature points through
the repository's own read-only loader (the same
``load_futures_feature_points_from_db`` machinery
``lumina_quant.data.feature_points.FeaturePointLookup`` uses, mirroring how
``scripts/research/report_data_coverage.py`` probes the OHLCV store), counts
the UTC symbol-days carrying at least one non-null finite point per audited
column, aggregates per symbol group, and writes a sorted-key JSON artifact.

The report is DETERMINISTIC for fixed inputs (no wall-clock stamp).  A run
against a root with zero feature data FAILS CLOSED: it emits a clean
``"status": "insufficient_data"`` JSON with every gate ``passed: false`` and
never raises; a non-zero exit is reserved for usage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.data.feature_points import FEATURE_COLUMNS
from lumina_quant.market_data import load_futures_feature_points_from_db
from lumina_quant.research_universe import (
    BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS,
    BINANCE_EXTENDED_RESEARCH_SYMBOLS,
    BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS,
)
from lumina_quant.symbols import canonical_symbol

ARTIFACT_KIND = "liquidation_feature_coverage_audit"

# Audited feature columns (all members of the canonical FEATURE_COLUMNS tuple).
OI_COLUMN = "open_interest"
LIQ_LONG_COLUMN = "liquidation_long_notional"
LIQ_SHORT_COLUMN = "liquidation_short_notional"
AUDIT_COLUMNS: tuple[str, ...] = (OI_COLUMN, LIQ_LONG_COLUMN, LIQ_SHORT_COLUMN)

# Pre-registered kill-gate floors (proposal constants, NOT tunables).
DEFAULT_OI_COVERAGE_FLOOR = 0.90
DEFAULT_LIQUIDATION_COVERAGE_FLOOR = 0.80

_MS_PER_DAY = 86_400_000

_CORE_CRYPTO = frozenset(BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS)
_TRADFI_PERP = frozenset(BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS)


def _compact(symbol: str) -> str:
    """Compact on-disk token (``BTC/USDT`` -> ``BTCUSDT``); never raises."""
    canonical = canonical_symbol(str(symbol)) or str(symbol)
    return canonical.replace("/", "").upper()


def _symbol_group(symbol: str) -> str:
    token = _compact(symbol)
    if token in _CORE_CRYPTO:
        return "core_crypto"
    if token in _TRADFI_PERP:
        return "tradfi_perp"
    return "other"


def _parse_utc_date(token: str, *, label: str) -> date:
    try:
        return date.fromisoformat(str(token).strip())
    except ValueError as exc:
        raise SystemExit(f"[audit_liquidation_feature_coverage] bad {label}: {exc}") from exc


def _share(covered: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(covered / float(total), 6)


def _covered_days(frame: pl.DataFrame | None, column: str) -> set[int]:
    """UTC day ordinals (epoch-days) with >=1 non-null finite value in ``column``."""
    if frame is None or frame.is_empty() or column not in frame.columns:
        return set()
    try:
        filtered = frame.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
        if filtered.is_empty():
            return set()
        days = (filtered.get_column("timestamp_ms") // _MS_PER_DAY).unique().to_list()
    except Exception:
        return set()
    return {int(value) for value in days if value is not None}


def _load_symbol_frame(
    data_root: str,
    *,
    exchange: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pl.DataFrame | None:
    """Read-only feature-point load; any failure degrades to ``None`` (no raise)."""
    try:
        frame = load_futures_feature_points_from_db(
            data_root,
            exchange=exchange,
            symbol=symbol,
            start_date=start_ms,
            end_date=end_ms,
        )
    except Exception:
        return None
    if frame is None or frame.is_empty() or "timestamp_ms" not in frame.columns:
        return None
    try:
        return frame.filter(
            (pl.col("timestamp_ms") >= start_ms) & (pl.col("timestamp_ms") <= end_ms)
        )
    except Exception:
        return None


def build_report(
    *,
    data_root: str,
    exchange: str,
    symbols: Sequence[str],
    start: date,
    end: date,
    oi_coverage_floor: float,
    liquidation_coverage_floor: float,
) -> dict[str, Any]:
    n_days = (end - start).days + 1
    start_ms = (start - date(1970, 1, 1)).days * _MS_PER_DAY
    end_ms = start_ms + n_days * _MS_PER_DAY - 1

    # Deterministic symbol order; duplicates collapse on the compact token.
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        token = _compact(raw)
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    ordered.sort()

    symbol_rows: list[dict[str, Any]] = []
    group_acc: dict[str, dict[str, int]] = {}
    total_rows = 0
    for token in ordered:
        frame = _load_symbol_frame(
            data_root, exchange=exchange, symbol=token, start_ms=start_ms, end_ms=end_ms
        )
        row_count = int(frame.height) if frame is not None else 0
        total_rows += row_count
        oi_days = _covered_days(frame, OI_COLUMN)
        liq_long_days = _covered_days(frame, LIQ_LONG_COLUMN)
        liq_short_days = _covered_days(frame, LIQ_SHORT_COLUMN)
        liq_both_days = liq_long_days & liq_short_days
        group = _symbol_group(token)
        symbol_rows.append(
            {
                "symbol": token,
                "group": group,
                "feature_rows": row_count,
                "days_total": n_days,
                "days_with_open_interest": len(oi_days),
                "days_with_liquidation_long": len(liq_long_days),
                "days_with_liquidation_short": len(liq_short_days),
                "days_with_liquidation_both": len(liq_both_days),
                "open_interest_share": _share(len(oi_days), n_days),
                "liquidation_long_share": _share(len(liq_long_days), n_days),
                "liquidation_short_share": _share(len(liq_short_days), n_days),
                "liquidation_both_share": _share(len(liq_both_days), n_days),
            }
        )
        acc = group_acc.setdefault(
            group,
            {
                "symbols": 0,
                "symbol_days": 0,
                "open_interest_days": 0,
                "liquidation_long_days": 0,
                "liquidation_short_days": 0,
                "liquidation_both_days": 0,
            },
        )
        acc["symbols"] += 1
        acc["symbol_days"] += n_days
        acc["open_interest_days"] += len(oi_days)
        acc["liquidation_long_days"] += len(liq_long_days)
        acc["liquidation_short_days"] += len(liq_short_days)
        acc["liquidation_both_days"] += len(liq_both_days)

    status = "ok" if total_rows > 0 else "insufficient_data"

    groups: dict[str, Any] = {}
    for group in sorted(group_acc):
        acc = group_acc[group]
        oi_share = _share(acc["open_interest_days"], acc["symbol_days"])
        liq_share = _share(acc["liquidation_both_days"], acc["symbol_days"])
        groups[group] = {
            "symbols": acc["symbols"],
            "symbol_days": acc["symbol_days"],
            "open_interest_share": oi_share,
            "liquidation_long_share": _share(acc["liquidation_long_days"], acc["symbol_days"]),
            "liquidation_short_share": _share(acc["liquidation_short_days"], acc["symbol_days"]),
            "liquidation_both_share": liq_share,
            # Fail-closed gates: a zero-data run passes NOTHING.
            "open_interest_gate": {
                "floor": oi_coverage_floor,
                "share": oi_share,
                "passed": bool(status == "ok" and oi_share >= oi_coverage_floor),
            },
            "liquidation_gate": {
                "floor": liquidation_coverage_floor,
                "share": liq_share,
                "passed": bool(status == "ok" and liq_share >= liquidation_coverage_floor),
            },
        }

    return {
        "artifact_kind": ARTIFACT_KIND,
        "status": status,
        "data_root": str(data_root),
        "exchange": str(exchange),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days_in_range": n_days,
        "audited_columns": list(AUDIT_COLUMNS),
        "feature_columns_known": len(FEATURE_COLUMNS),
        "oi_coverage_floor": oi_coverage_floor,
        "liquidation_coverage_floor": liquidation_coverage_floor,
        "symbol_count": len(ordered),
        "symbols": symbol_rows,
        "groups": groups,
        "summary": {
            "total_feature_rows": total_rows,
            "groups_passing_oi_gate": sum(
                1 for payload in groups.values() if payload["open_interest_gate"]["passed"]
            ),
            "groups_passing_liquidation_gate": sum(
                1 for payload in groups.values() if payload["liquidation_gate"]["passed"]
            ),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="Feature-points parquet store root (the market-data repository root).",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to audit (default: the full 110-name extended research universe).",
    )
    parser.add_argument("--start", required=True, help="Audit window start, UTC date YYYY-MM-DD.")
    parser.add_argument(
        "--end", required=True, help="Audit window end (inclusive), UTC date YYYY-MM-DD."
    )
    parser.add_argument("--exchange", default="binance", help="Exchange partition to inspect.")
    parser.add_argument(
        "--oi-coverage-floor",
        type=float,
        default=DEFAULT_OI_COVERAGE_FLOOR,
        help="Per-group OI symbol-day coverage kill floor (default: %(default)s).",
    )
    parser.add_argument(
        "--liquidation-coverage-floor",
        type=float,
        default=DEFAULT_LIQUIDATION_COVERAGE_FLOOR,
        help="Per-group liquidation symbol-day coverage kill floor (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Write the sorted-key JSON artifact to PATH (stdout always gets it).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    start = _parse_utc_date(args.start, label="--start")
    end = _parse_utc_date(args.end, label="--end")
    if end < start:
        parser.error(f"--end {end.isoformat()} precedes --start {start.isoformat()}")

    symbols = list(args.symbols) if args.symbols else list(BINANCE_EXTENDED_RESEARCH_SYMBOLS)
    root = Path(args.data_root)
    if not root.exists():
        print(
            f"[audit_liquidation_feature_coverage] data root does not exist: {root}",
            file=sys.stderr,
        )

    report = build_report(
        data_root=str(root),
        exchange=str(args.exchange),
        symbols=symbols,
        start=start,
        end=end,
        oi_coverage_floor=float(args.oi_coverage_floor),
        liquidation_coverage_floor=float(args.liquidation_coverage_floor),
    )

    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(payload, encoding="utf-8")
        print(
            f"[audit_liquidation_feature_coverage] wrote JSON -> {json_path}",
            file=sys.stderr,
        )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
