#!/usr/bin/env python3
"""Report local market-data coverage for research evaluability.

The data-PC evaluation closes candidate rows as ``insufficient_data`` when a
``(symbol, timeframe)`` series is absent (or too short) in its local parquet
store. Nobody could previously see *what* coverage actually exists there. This
script makes that visible.

Two discovery seams (deliberately explicit, never a silent fallback):

* **registry** (default) -- reuse the repository's own loading path
  (:class:`lumina_quant.storage.parquet.ParquetMarketDataRepository`). For each
  discovered symbol and each requested timeframe it calls ``load_ohlcv``, i.e.
  the *same* monthly-parquet + WAL merge and bucket resampling the research
  runner uses. This is the honest "would ``symbol@tf`` yield bars" probe.
* **scan** (``--scan-dir PATH``) -- a documented directory scan of the standard
  parquet layout that reports the *physical* footprint (row counts + spans from
  parquet metadata) without loading or resampling. Cheap; explicit opt-in.

Both modes enumerate symbols from the standard on-disk layouts:

* ``<root>/market_ohlcv_1s/<exchange>/<symbol>/<YYYY-MM>.parquet`` (1s base)
* ``<root>/market_data_materialized/<exchange>/<symbol>/timeframe=<tf>/date=*/``
* ``<root>/exchange=<exchange>/symbol=<symbol>/timeframe=<tf>/date=*/*.parquet``

``--check-manifest MANIFEST.json`` performs a dry-run of the ``insufficient_data``
outcome: for every candidate row it reports which of the row's symbols are
missing at that row's timeframe -- before the grid ever runs.

The tool never raises on a missing/absent store (it prints a clean message and
emits an empty report); a non-zero exit is reserved for usage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.data.timeframe import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.storage.parquet import (
    ParquetMarketDataRepository,
    is_parquet_market_data_store,
)
from lumina_quant.strategy_factory.runtime_settings import current_research_market_data_settings
from lumina_quant.symbols import canonical_symbol

ARTIFACT_KIND = "data_coverage_report"
# Mirrors ``research_runner._MIN_BARS`` -- the bar floor below which a bundle is
# dropped and the candidate row falls through to ``insufficient_data``.
DEFAULT_MIN_BARS = 360
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")

_MONTHLY_GLOB = "????-??.parquet"


# --------------------------------------------------------------------------- #
# small deterministic helpers
# --------------------------------------------------------------------------- #
def _iso_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    naive = value.replace(tzinfo=None)
    return naive.strftime("%Y-%m-%dT%H:%M:%SZ")


def _dt_to_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    naive = value.replace(tzinfo=None)
    return int(naive.replace(tzinfo=UTC).timestamp() * 1000)


def _safe_tf_ms(timeframe: str) -> int | None:
    try:
        return int(timeframe_to_milliseconds(normalize_timeframe_token(timeframe)))
    except ValueError, TypeError:
        return None


def _tf_sort_key(timeframe: str) -> tuple[int, str]:
    tf_ms = _safe_tf_ms(timeframe)
    return (tf_ms if tf_ms is not None else 1 << 62, str(timeframe))


def _norm_tf(timeframe: str) -> str:
    try:
        return normalize_timeframe_token(timeframe)
    except ValueError, TypeError:
        return str(timeframe).strip().lower()


def _gap_ratio(
    *, bar_count: int, first_ms: int | None, last_ms: int | None, tf_ms: int | None
) -> float | None:
    if bar_count <= 1 or tf_ms is None or tf_ms <= 0 or first_ms is None or last_ms is None:
        return None
    span = last_ms - first_ms
    if span <= 0:
        return None
    expected = span // tf_ms + 1
    if expected <= 0:
        return None
    return max(0.0, round(1.0 - bar_count / expected, 6))


def _span_from_parquet(paths: Sequence[Path]) -> tuple[int, datetime | None, datetime | None]:
    """Row count + first/last datetime from parquet metadata (never raises)."""
    real = [str(path) for path in paths if path.exists()]
    if not real:
        return 0, None, None
    try:
        agg = (
            pl.scan_parquet(real)
            .select(
                pl.len().alias("n"),
                pl.col("datetime").min().alias("mn"),
                pl.col("datetime").max().alias("mx"),
            )
            .collect()
        )
    except OSError, RuntimeError, ValueError, pl.exceptions.PolarsError:
        return 0, None, None
    if agg.is_empty():
        return 0, None, None
    return int(agg["n"][0] or 0), agg["mn"][0], agg["mx"][0]


def _span_from_frame(frame: pl.DataFrame | None) -> tuple[int, datetime | None, datetime | None]:
    if frame is None or frame.is_empty() or "datetime" not in frame.columns:
        return 0, None, None
    column = frame.get_column("datetime")
    return int(frame.height), column.min(), column.max()


# --------------------------------------------------------------------------- #
# store enumeration (shared by both modes)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SymbolInventory:
    symbol: str  # canonical BASE/QUOTE
    token: str  # compact on-disk token (e.g. BTCUSDT)
    monthly_1s_dir: Path | None = None
    materialized_tfs: dict[str, Path] = field(default_factory=dict)
    partitioned_tfs: dict[str, Path] = field(default_factory=dict)

    def has_base(self) -> bool:
        return self.monthly_1s_dir is not None or "1m" in self.partitioned_tfs


def enumerate_symbols(root: Path, exchange: str) -> dict[str, SymbolInventory]:
    """Discover ``(symbol -> layouts)`` from the standard on-disk store layouts."""
    exchange_lc = str(exchange).strip().lower()
    out: dict[str, SymbolInventory] = {}

    def _inv(token: str) -> SymbolInventory:
        symbol = canonical_symbol(token) or token
        entry = out.get(symbol)
        if entry is None:
            entry = SymbolInventory(symbol=symbol, token=token)
            out[symbol] = entry
        return entry

    # 1) monthly 1s: <root>/market_ohlcv_1s/<exchange>/<token>/<YYYY-MM>.parquet
    monthly_root = root / "market_ohlcv_1s" / exchange_lc
    if monthly_root.is_dir():
        for symbol_dir in sorted(p for p in monthly_root.iterdir() if p.is_dir()):
            if any(symbol_dir.glob(_MONTHLY_GLOB)):
                _inv(symbol_dir.name).monthly_1s_dir = symbol_dir

    # 2) materialized: <root>/market_data_materialized/<exchange>/<token>/timeframe=<tf>/
    materialized_root = root / "market_data_materialized" / exchange_lc
    if materialized_root.is_dir():
        for symbol_dir in sorted(p for p in materialized_root.iterdir() if p.is_dir()):
            entry = _inv(symbol_dir.name)
            for tf_dir in sorted(symbol_dir.glob("timeframe=*")):
                if not tf_dir.is_dir():
                    continue
                token = tf_dir.name.split("=", 1)[1]
                entry.materialized_tfs[_norm_tf(token)] = tf_dir

    # 3) partitioned: <root>/exchange=<exchange>/symbol=<token>/timeframe=<tf>/
    for exchange_dir in sorted(root.glob("exchange=*")):
        if not exchange_dir.is_dir():
            continue
        if exchange_dir.name.split("=", 1)[1].strip().lower() != exchange_lc:
            continue
        for symbol_dir in sorted(exchange_dir.glob("symbol=*")):
            if not symbol_dir.is_dir():
                continue
            entry = _inv(symbol_dir.name.split("=", 1)[1])
            for tf_dir in sorted(symbol_dir.glob("timeframe=*")):
                if not tf_dir.is_dir():
                    continue
                token = tf_dir.name.split("=", 1)[1]
                entry.partitioned_tfs[_norm_tf(token)] = tf_dir

    return out


# --------------------------------------------------------------------------- #
# coverage rows
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CoverageRow:
    exchange: str
    symbol: str
    timeframe: str
    bar_count: int | None
    first_ts: str | None
    last_ts: str | None
    gap_ratio: float | None
    source: str
    sufficient: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_count": self.bar_count,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "gap_ratio": self.gap_ratio,
            "source": self.source,
            "sufficient": self.sufficient,
        }

    def sort_key(self) -> tuple[Any, ...]:
        return (self.exchange, self.symbol, *_tf_sort_key(self.timeframe), self.source)


def _coverage_row(
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    bar_count: int,
    first_dt: datetime | None,
    last_dt: datetime | None,
    source: str,
    min_bars: int,
) -> CoverageRow:
    tf_ms = _safe_tf_ms(timeframe)
    first_ms = _dt_to_ms(first_dt)
    last_ms = _dt_to_ms(last_dt)
    return CoverageRow(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        bar_count=bar_count,
        first_ts=_iso_z(first_dt),
        last_ts=_iso_z(last_dt),
        gap_ratio=_gap_ratio(bar_count=bar_count, first_ms=first_ms, last_ms=last_ms, tf_ms=tf_ms),
        source=source,
        sufficient=bar_count >= max(1, int(min_bars)),
    )


def _scan_pair_paths(inv: SymbolInventory, timeframe: str) -> tuple[list[Path], str] | None:
    """Physical parquet paths + source label for a ``(symbol, tf)``, if present."""
    tf = _norm_tf(timeframe)
    if tf in inv.partitioned_tfs:
        return sorted(inv.partitioned_tfs[tf].glob("date=*/*.parquet")), "partitioned"
    if tf in inv.materialized_tfs:
        return sorted(inv.materialized_tfs[tf].glob("date=*/*.parquet")), "materialized"
    if tf == "1s" and inv.monthly_1s_dir is not None:
        return sorted(inv.monthly_1s_dir.glob(_MONTHLY_GLOB)), "monthly_1s"
    return None


def scan_coverage_rows(
    inventories: Mapping[str, SymbolInventory],
    *,
    exchange: str,
    timeframes: Sequence[str] | None,
    min_bars: int,
) -> list[CoverageRow]:
    """Physical-footprint coverage rows (no loading/resampling)."""
    tf_filter = {_norm_tf(tf) for tf in timeframes} if timeframes else None
    rows: list[CoverageRow] = []
    for symbol in sorted(inventories):
        inv = inventories[symbol]
        physical: dict[str, tuple[list[Path], str]] = {}
        if inv.monthly_1s_dir is not None:
            physical["1s"] = (sorted(inv.monthly_1s_dir.glob(_MONTHLY_GLOB)), "monthly_1s")
        for tf, tf_dir in inv.partitioned_tfs.items():
            physical[tf] = (sorted(tf_dir.glob("date=*/*.parquet")), "partitioned")
        for tf, tf_dir in inv.materialized_tfs.items():
            physical.setdefault(tf, (sorted(tf_dir.glob("date=*/*.parquet")), "materialized"))
        for tf, (paths, source) in physical.items():
            if tf_filter is not None and tf not in tf_filter:
                continue
            bar_count, first_dt, last_dt = _span_from_parquet(paths)
            rows.append(
                _coverage_row(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=tf,
                    bar_count=bar_count,
                    first_dt=first_dt,
                    last_dt=last_dt,
                    source=source,
                    min_bars=min_bars,
                )
            )
    rows.sort(key=CoverageRow.sort_key)
    return rows


def _probe_pair_registry(
    repo: ParquetMarketDataRepository,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    min_bars: int,
) -> CoverageRow:
    """Reuse the repository's own resampling loader to probe one ``(symbol, tf)``."""
    try:
        frame = repo.load_ohlcv(exchange=exchange, symbol=symbol, timeframe=timeframe)
    except OSError, RuntimeError, ValueError, pl.exceptions.PolarsError:
        frame = None
    except Exception:
        # A diagnostic tool must never crash on a single malformed pair.
        frame = None
    bar_count, first_dt, last_dt = _span_from_frame(frame)
    return _coverage_row(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        bar_count=bar_count,
        first_dt=first_dt,
        last_dt=last_dt,
        source="loader",
        min_bars=min_bars,
    )


def registry_coverage_rows(
    root: Path,
    inventories: Mapping[str, SymbolInventory],
    *,
    exchange: str,
    timeframes: Sequence[str],
    min_bars: int,
) -> list[CoverageRow]:
    repo = ParquetMarketDataRepository(root)
    rows: list[CoverageRow] = []
    for symbol in sorted(inventories):
        for timeframe in timeframes:
            rows.append(
                _probe_pair_registry(
                    repo,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    min_bars=min_bars,
                )
            )
    rows.sort(key=CoverageRow.sort_key)
    return rows


# --------------------------------------------------------------------------- #
# manifest dry-run (insufficient_data preview)
# --------------------------------------------------------------------------- #
def load_manifest_candidates(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("candidates", "rows", "manifest", "grid"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
    return []


def candidate_symbols_and_timeframe(candidate: Mapping[str, Any]) -> tuple[list[str], str]:
    """Mirror ``research_runner._candidate_symbols_and_timeframe`` (canonical + tf)."""
    symbols: list[str] = []
    for raw in candidate.get("symbols") or []:
        token = canonical_symbol(str(raw))
        if token and token not in symbols:
            symbols.append(token)
    timeframe = str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or "1m")
    return symbols, _norm_tf(timeframe)


def _pair_covered(row: CoverageRow | None, *, min_bars: int) -> bool:
    if row is None:
        return False
    if row.bar_count is None:  # scan-mode "derivable" -- present but not counted
        return True
    return row.bar_count >= max(1, int(min_bars))


def _coverage_for_pair(
    symbol: str,
    timeframe: str,
    *,
    mode: str,
    repo: ParquetMarketDataRepository | None,
    inventories: Mapping[str, SymbolInventory],
    exchange: str,
    min_bars: int,
    cache: dict[tuple[str, str], CoverageRow | None],
) -> CoverageRow | None:
    key = (symbol, timeframe)
    if key in cache:
        return cache[key]
    row: CoverageRow | None
    if mode == "registry":
        assert repo is not None
        row = _probe_pair_registry(
            repo, exchange=exchange, symbol=symbol, timeframe=timeframe, min_bars=min_bars
        )
    else:
        inv = inventories.get(symbol)
        row = None
        if inv is not None:
            physical = _scan_pair_paths(inv, timeframe)
            if physical is not None:
                bar_count, first_dt, last_dt = _span_from_parquet(physical[0])
                row = _coverage_row(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    bar_count=bar_count,
                    first_dt=first_dt,
                    last_dt=last_dt,
                    source=physical[1],
                    min_bars=min_bars,
                )
            elif inv.has_base():
                # A 1s/1m base exists; the requested tf is derivable but its bar
                # count is unknown without resampling. Presence-only (necessary
                # but not sufficient) -- use registry mode for the exact count.
                row = CoverageRow(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    bar_count=None,
                    first_ts=None,
                    last_ts=None,
                    gap_ratio=None,
                    source="derivable_from_base",
                    sufficient=True,
                )
    cache[key] = row
    return row


def check_manifest(
    manifest_path: Path,
    *,
    mode: str,
    root: Path,
    exchange: str,
    inventories: Mapping[str, SymbolInventory],
    min_bars: int,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = load_manifest_candidates(payload)
    repo = ParquetMarketDataRepository(root) if mode == "registry" else None
    cache: dict[tuple[str, str], CoverageRow | None] = {}

    rows: list[dict[str, Any]] = []
    evaluable_count = 0
    for index, candidate in enumerate(candidates):
        symbols, timeframe = candidate_symbols_and_timeframe(candidate)
        missing: list[str] = []
        for symbol in symbols:
            row = _coverage_for_pair(
                symbol,
                timeframe,
                mode=mode,
                repo=repo,
                inventories=inventories,
                exchange=exchange,
                min_bars=min_bars,
                cache=cache,
            )
            if not _pair_covered(row, min_bars=min_bars):
                missing.append(symbol)
        evaluable = not missing and bool(symbols)
        if evaluable:
            evaluable_count += 1
        rows.append(
            {
                "index": index,
                "name": str(candidate.get("name") or candidate.get("candidate_id") or ""),
                "timeframe": timeframe,
                "symbols": symbols,
                "missing_symbols": missing,
                "evaluable": evaluable,
            }
        )

    return {
        "manifest_path": str(manifest_path),
        "mode": mode,
        "candidate_count": len(candidates),
        "evaluable_count": evaluable_count,
        "insufficient_count": len(candidates) - evaluable_count,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# report assembly + rendering
# --------------------------------------------------------------------------- #
def _resolve_timeframes(requested: Sequence[str] | None) -> list[str]:
    source = list(requested) if requested else list(DEFAULT_TIMEFRAMES)
    seen: list[str] = []
    for tf in source:
        token = _norm_tf(tf)
        if token not in seen:
            seen.append(token)
    return seen


def _filter_inventories(
    inventories: Mapping[str, SymbolInventory], symbols: Sequence[str] | None
) -> dict[str, SymbolInventory]:
    if not symbols:
        return dict(inventories)
    wanted = {canonical_symbol(str(item)) or str(item) for item in symbols}
    return {symbol: inv for symbol, inv in inventories.items() if symbol in wanted}


def build_report(
    *,
    root: Path,
    exchange: str,
    mode: str,
    symbols: Sequence[str] | None,
    timeframes: Sequence[str] | None,
    min_bars: int,
    manifest_path: Path | None,
) -> dict[str, Any]:
    store_detected = root.exists() and is_parquet_market_data_store(str(root))
    inventories = enumerate_symbols(root, exchange) if root.exists() else {}
    inventories = _filter_inventories(inventories, symbols)

    if mode == "registry":
        resolved_tfs = _resolve_timeframes(timeframes)
        coverage_rows = registry_coverage_rows(
            root, inventories, exchange=exchange, timeframes=resolved_tfs, min_bars=min_bars
        )
    else:
        resolved_tfs = _resolve_timeframes(timeframes) if timeframes else []
        coverage_rows = scan_coverage_rows(
            inventories, exchange=exchange, timeframes=timeframes, min_bars=min_bars
        )

    report: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(root),
        "exchange": str(exchange),
        "mode": mode,
        "min_bars": int(min_bars),
        "store_detected": bool(store_detected),
        "symbol_filter": sorted({canonical_symbol(str(s)) or str(s) for s in symbols})
        if symbols
        else None,
        "timeframes": resolved_tfs,
        "symbol_count": len(inventories),
        "coverage": [row.to_dict() for row in coverage_rows],
        "summary": {
            "symbols": len(inventories),
            "pairs": len(coverage_rows),
            "sufficient_pairs": sum(1 for row in coverage_rows if row.sufficient),
        },
    }
    if manifest_path is not None:
        report["manifest_check"] = check_manifest(
            manifest_path,
            mode=mode,
            root=root,
            exchange=exchange,
            inventories=inventories,
            min_bars=min_bars,
        )
    return report


def _fmt_gap(gap_ratio: float | None) -> str:
    if gap_ratio is None:
        return "-"
    return f"{gap_ratio * 100:.2f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Data coverage report ({report['mode']})",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- root: `{report['root']}`",
        f"- exchange: `{report['exchange']}`",
        f"- store_detected: `{report['store_detected']}`",
        f"- min_bars: `{report['min_bars']}`",
        f"- symbols: `{report['summary']['symbols']}` | pairs: "
        f"`{report['summary']['pairs']}` | sufficient: "
        f"`{report['summary']['sufficient_pairs']}`",
        "",
    ]
    if not report["store_detected"] and not report["coverage"]:
        lines.append("_No parquet market-data store detected at root._")
        lines.append("")

    lines.extend(
        [
            "## Coverage",
            "",
            "| symbol | tf | bars | first_ts | last_ts | gap | source | ok |",
            "|---|---|---:|---|---|---:|---|:--:|",
        ]
    )
    for row in report["coverage"]:
        bars = "-" if row["bar_count"] is None else str(row["bar_count"])
        lines.append(
            f"| `{row['symbol']}` | {row['timeframe']} | {bars} | "
            f"`{row['first_ts'] or '-'}` | `{row['last_ts'] or '-'}` | "
            f"{_fmt_gap(row['gap_ratio'])} | {row['source']} | "
            f"{'Y' if row['sufficient'] else 'N'} |"
        )
    lines.append("")

    manifest_check = report.get("manifest_check")
    if isinstance(manifest_check, Mapping):
        lines.extend(
            [
                "## Manifest evaluability dry-run",
                "",
                f"- manifest: `{manifest_check['manifest_path']}`",
                f"- candidates: `{manifest_check['candidate_count']}` | evaluable: "
                f"`{manifest_check['evaluable_count']}` | insufficient: "
                f"`{manifest_check['insufficient_count']}`",
                "",
                "| # | name | tf | symbols | missing | evaluable |",
                "|---:|---|---|---:|---|:--:|",
            ]
        )
        for row in manifest_check["rows"]:
            missing = ", ".join(row["missing_symbols"]) or "-"
            lines.append(
                f"| {row['index']} | `{row['name']}` | {row['timeframe']} | "
                f"{len(row['symbols'])} | `{missing}` | "
                f"{'Y' if row['evaluable'] else 'N'} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    defaults = current_research_market_data_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(defaults["parquet_root"]),
        help="Parquet store root for registry mode (default from research settings).",
    )
    parser.add_argument(
        "--scan-dir",
        default=None,
        help="Explicit directory-scan mode rooted at this path (physical footprint only).",
    )
    parser.add_argument(
        "--exchange",
        default=str(defaults["exchange"]),
        help="Exchange partition to inspect (default from research settings).",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Restrict to these symbols (canonicalized).",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Timeframes to probe (registry mode default: %s)." % ",".join(DEFAULT_TIMEFRAMES),
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=DEFAULT_MIN_BARS,
        help="Bar floor for the sufficiency flag / dry-run (default: %(default)s).",
    )
    parser.add_argument(
        "--check-manifest",
        default=None,
        metavar="MANIFEST_JSON",
        help="Candidate-manifest JSON: dry-run the insufficient_data outcome per row.",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Write the full JSON report to PATH.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.scan_dir is not None:
        mode = "scan"
        root = Path(args.scan_dir)
    else:
        mode = "registry"
        root = Path(args.root)

    manifest_path: Path | None = None
    if args.check_manifest is not None:
        manifest_path = Path(args.check_manifest)
        if not manifest_path.exists():
            parser.error(f"--check-manifest file not found: {manifest_path}")
        try:
            json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"--check-manifest is not readable JSON: {exc}")

    if not root.exists():
        print(f"[report_data_coverage] store root does not exist: {root}", file=sys.stderr)

    report = build_report(
        root=root,
        exchange=str(args.exchange),
        mode=mode,
        symbols=args.symbols,
        timeframes=args.timeframes,
        min_bars=int(args.min_bars),
        manifest_path=manifest_path,
    )

    if args.json is not None:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[report_data_coverage] wrote JSON -> {json_path}", file=sys.stderr)

    sys.stdout.write(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
