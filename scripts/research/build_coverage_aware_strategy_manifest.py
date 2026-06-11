"""Build a strategy-factory manifest with per-candidate data-window splits.

The default candidate library assumes one common research window.  This builder
keeps late-starting TradFi/ETF/proxy symbols usable by attaching an
``effective_split`` to each candidate based on the overlapping market-data
window of that candidate's symbols.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from lumina_quant.strategy_factory.candidate_library import build_candidate_manifest
from lumina_quant.symbols import canonicalize_symbol_list, normalize_strategy_timeframes

_DEFAULT_TIMEFRAMES = ("30m", "1h", "4h", "1d")
_BARS_PER_DAY = {"30m": 48, "1h": 24, "4h": 6, "1d": 1}
_MIN_BARS_BY_TIMEFRAME = {"30m": 360, "1h": 240, "4h": 90, "1d": 20}
_COHORT_MIN_DAYS = (120, 60, 20)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a coverage-aware 30m+ strategy-factory manifest."
    )
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument("--balanced-output", default="", help="Optional balanced subset path.")
    parser.add_argument("--balanced-limit", type=int, default=1200)
    parser.add_argument("--parquet-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--timeframes", nargs="+", default=list(_DEFAULT_TIMEFRAMES))
    parser.add_argument(
        "--include-all-discovered",
        action="store_true",
        help="Also build one all-symbol cohort for single-symbol families.",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=20,
        help="Minimum 1m date partitions required before a symbol can join any cohort.",
    )
    return parser


def _iso_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_date_token(token: str) -> date | None:
    try:
        return date.fromisoformat(str(token))
    except ValueError:
        return None


def _discover_symbol_windows(
    *,
    parquet_root: Path,
    exchange: str,
) -> dict[str, dict[str, Any]]:
    exchange_root = parquet_root / f"exchange={exchange}"
    windows: dict[str, dict[str, Any]] = {}
    for symbol_dir in sorted(exchange_root.glob("symbol=*")):
        compact = symbol_dir.name.split("=", 1)[1]
        timeframe_dir = symbol_dir / "timeframe=1m"
        if not timeframe_dir.exists():
            continue
        dates = sorted(
            parsed
            for partition in timeframe_dir.glob("date=*")
            if partition.is_dir()
            for parsed in [_parse_date_token(partition.name.split("=", 1)[1])]
            if parsed is not None
        )
        if not dates:
            continue
        symbol = canonicalize_symbol_list([compact])[0]
        windows[symbol] = {
            "symbol": symbol,
            "raw_symbol": compact,
            "first_date": dates[0].isoformat(),
            "last_date": dates[-1].isoformat(),
            "partition_count": len(dates),
        }
    return windows


def _symbol_count_for_min_days(
    windows: Mapping[str, Mapping[str, Any]],
    *,
    min_days: int,
) -> list[str]:
    return sorted(
        symbol
        for symbol, payload in windows.items()
        if int(payload.get("partition_count", 0) or 0) >= int(min_days)
    )


def _cohort_specs(
    windows: Mapping[str, Mapping[str, Any]],
    *,
    min_days: int,
    include_all_discovered: bool,
) -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = []
    for days in _COHORT_MIN_DAYS:
        if days < min_days:
            continue
        symbols = _symbol_count_for_min_days(windows, min_days=days)
        if len(symbols) >= 2:
            specs.append((f"coverage_{days}d_plus", symbols))

    min_symbols = _symbol_count_for_min_days(windows, min_days=min_days)
    if len(min_symbols) >= 2:
        specs.append((f"coverage_{min_days}d_plus", min_symbols))

    if include_all_discovered:
        all_symbols = sorted(windows)
        if len(all_symbols) >= 2:
            specs.append(("all_discovered_symbols", all_symbols))

    deduped: list[tuple[str, list[str]]] = []
    seen_symbol_sets: set[tuple[str, ...]] = set()
    for name, symbols in specs:
        key = tuple(symbols)
        if key in seen_symbol_sets:
            continue
        seen_symbol_sets.add(key)
        deduped.append((name, symbols))
    return deduped


def _candidate_symbols(row: Mapping[str, Any]) -> list[str]:
    return canonicalize_symbol_list(list(row.get("symbols") or []))


def _candidate_timeframe(row: Mapping[str, Any]) -> str:
    return str(row.get("strategy_timeframe") or row.get("timeframe") or "1h").strip().lower()


def _window_dates_for_candidate(
    row: Mapping[str, Any],
    windows: Mapping[str, Mapping[str, Any]],
) -> tuple[date, date] | None:
    symbols = _candidate_symbols(row)
    if not symbols:
        return None
    payloads = [windows.get(symbol) for symbol in symbols]
    if any(payload is None for payload in payloads):
        return None
    starts = [
        _parse_date_token(str(payload.get("first_date") or ""))
        for payload in payloads
        if payload is not None
    ]
    ends = [
        _parse_date_token(str(payload.get("last_date") or ""))
        for payload in payloads
        if payload is not None
    ]
    if any(item is None for item in starts) or any(item is None for item in ends):
        return None
    return max(starts), min(ends)  # type: ignore[arg-type, return-value]


def _approx_bar_count(start: date, end: date, timeframe: str) -> int:
    days = max(0, (end - start).days + 1)
    return int(days * int(_BARS_PER_DAY.get(timeframe, 24)))


def _candidate_split(
    *,
    start: date,
    end: date,
    timeframe: str,
) -> dict[str, Any] | None:
    start_dt = datetime.combine(start, time.min, tzinfo=UTC)
    end_dt = datetime.combine(end, time.max, tzinfo=UTC).replace(microsecond=999000)
    if end_dt <= start_dt:
        return None
    span = end_dt - start_dt
    train_end = start_dt + timedelta(seconds=span.total_seconds() * 0.60)
    val_end = start_dt + timedelta(seconds=span.total_seconds() * 0.80)
    if train_end <= start_dt or val_end <= train_end or end_dt <= val_end:
        return None
    return {
        "train_start": _iso_z(start_dt),
        "train_end": _iso_z(train_end),
        "val_start": _iso_z(train_end),
        "val_end": _iso_z(val_end),
        "oos_start": _iso_z(val_end),
        "oos_end": _iso_z(end_dt),
        "strategy_timeframe": timeframe,
        "mode": "candidate_data_window",
    }


def _enrich_and_filter_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    windows: Mapping[str, Mapping[str, Any]],
    cohort_name: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        timeframe = _candidate_timeframe(row)
        window = _window_dates_for_candidate(row, windows)
        if window is None:
            continue
        start, end = window
        approx_bars = _approx_bar_count(start, end, timeframe)
        min_bars = int(_MIN_BARS_BY_TIMEFRAME.get(timeframe, 240))
        if approx_bars < min_bars:
            continue
        split = _candidate_split(start=start, end=end, timeframe=timeframe)
        if split is None:
            continue
        metadata = dict(row.get("metadata") or {})
        metadata["coverage_cohort"] = cohort_name
        metadata["effective_split"] = dict(split)
        metadata["data_window"] = {
            "mode": "candidate_data_window",
            "first_date": start.isoformat(),
            "last_date": end.isoformat(),
            "approx_bars": int(approx_bars),
            "min_bars": int(min_bars),
            "symbols": {
                symbol: dict(windows[symbol])
                for symbol in _candidate_symbols(row)
                if symbol in windows
            },
        }
        row["metadata"] = metadata
        row["effective_split"] = dict(split)
        row["candidate_id"] = f"{row.get('candidate_id', '')}_{cohort_name}"[:96]
        row["name"] = f"{row.get('name', row['candidate_id'])}_{cohort_name}"
        enriched.append(row)
    return enriched


def _dedupe_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("candidate_id") or row.get("name") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _balanced_subset(rows: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        key = (
            str(row.get("family") or ""),
            _candidate_timeframe(row),
            str(metadata.get("coverage_cohort") or ""),
        )
        buckets.setdefault(key, []).append(dict(row))
    for bucket in buckets.values():
        bucket.sort(key=lambda item: str(item.get("name") or ""))

    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < limit and any(buckets.values()):
        for key in keys:
            bucket = buckets.get(key) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= limit:
                break
    return selected


def _manifest_payload(
    *,
    rows: Sequence[dict[str, Any]],
    windows: Mapping[str, Mapping[str, Any]],
    timeframes: Sequence[str],
    cohort_details: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family") or "") for row in rows)
    timeframe_counts = Counter(_candidate_timeframe(row) for row in rows)
    strategy_counts = Counter(str(row.get("strategy_class") or "") for row in rows)
    cohort_counts = Counter(
        str(dict(row.get("metadata") or {}).get("coverage_cohort") or "") for row in rows
    )
    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "schema_version": "coverage_aware_candidate_manifest_v1",
        "timeframes": list(timeframes),
        "candidate_count": len(rows),
        "symbol_count": len(windows),
        "symbol_windows": dict(windows),
        "cohorts": [dict(item) for item in cohort_details],
        "family_counts": dict(family_counts),
        "strategy_counts": dict(strategy_counts),
        "timeframe_counts": dict(timeframe_counts),
        "coverage_cohort_counts": dict(cohort_counts),
        "min_bars_by_timeframe": dict(_MIN_BARS_BY_TIMEFRAME),
        "candidates": list(rows),
    }


def main() -> int:
    args = _build_parser().parse_args()
    output = Path(str(args.output)).resolve()
    balanced_output = (
        Path(str(args.balanced_output)).resolve() if str(args.balanced_output).strip() else None
    )
    parquet_root = Path(str(args.parquet_root)).resolve()
    timeframes = normalize_strategy_timeframes(
        list(args.timeframes),
        required=_DEFAULT_TIMEFRAMES,
        strict_subset=True,
    )

    windows = _discover_symbol_windows(parquet_root=parquet_root, exchange=str(args.exchange))
    cohorts = _cohort_specs(
        windows,
        min_days=max(1, int(args.min_days)),
        include_all_discovered=bool(args.include_all_discovered),
    )

    all_rows: list[dict[str, Any]] = []
    cohort_details: list[dict[str, Any]] = []
    for cohort_name, symbols in cohorts:
        manifest = build_candidate_manifest(timeframes=timeframes, symbols=symbols)
        rows = _enrich_and_filter_candidates(
            list(manifest.get("candidates") or []),
            windows=windows,
            cohort_name=cohort_name,
        )
        cohort_details.append(
            {
                "name": cohort_name,
                "symbol_count": len(symbols),
                "candidate_count_before_filter": int(manifest.get("candidate_count", 0) or 0),
                "candidate_count_after_filter": len(rows),
                "symbols": list(symbols),
            }
        )
        all_rows.extend(rows)

    rows = _dedupe_candidates(all_rows)
    rows.sort(
        key=lambda row: (
            str(dict(row.get("metadata") or {}).get("coverage_cohort") or ""),
            str(row.get("family") or ""),
            _candidate_timeframe(row),
            str(row.get("name") or ""),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            _manifest_payload(
                rows=rows,
                windows=windows,
                timeframes=timeframes,
                cohort_details=cohort_details,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    if balanced_output is not None:
        balanced = _balanced_subset(rows, limit=max(1, int(args.balanced_limit)))
        balanced_output.parent.mkdir(parents=True, exist_ok=True)
        balanced_output.write_text(
            json.dumps(
                _manifest_payload(
                    rows=balanced,
                    windows=windows,
                    timeframes=timeframes,
                    cohort_details=cohort_details,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[MANIFEST] balanced_candidate_count={len(balanced)}")
        print(f"[MANIFEST] saved_balanced={balanced_output}")

    print(f"[MANIFEST] symbol_count={len(windows)}")
    print(f"[MANIFEST] cohort_count={len(cohorts)}")
    print(f"[MANIFEST] candidate_count={len(rows)}")
    print(f"[MANIFEST] saved={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
