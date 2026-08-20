"""Run the V-DIAG HAR-RV leader-spillover admission diagnostic (non-trading).

Thin CLI around :mod:`lumina_quant.research.vol_spillover_diagnostic`: loads
daily realized-variance series (or computes them from intraday closes), runs
the pre-registered whole-search diagnostic, and writes the deterministic JSON
verdict artifact (default ``var/research/vol_spillover_diagnostic.json``).
Reruns on identical inputs are byte-identical (no timestamps in the artifact).

Inputs (choose one):
- ``--rv-csv PATH``      CSV ``symbol,day,rv`` with precomputed daily RV
                         (``day`` = integer epoch-day id or ``YYYY-MM-DD``).
                         This is the data-free canonical path.
- ``--closes-csv PATH``  CSV ``symbol,epoch_seconds,close`` intraday closes;
                         daily RV is computed via
                         ``indicators.har_rv.daily_realized_variance``.
- ``--series-path DIR``  Data-PC path: per-symbol ``<SYMBOL>.parquet`` (read
                         via polars when installed) or ``<SYMBOL>.csv``
                         (``epoch_seconds,close``) files.

The pre-registered SPEC (pairs, folds, thresholds, seed, block convention) is
defined in the library module and echoed verbatim into the artifact;
``--pairs "L:F,L:F"`` restricts/overrides the pair list for smoke runs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumina_quant.indicators.har_rv import daily_realized_variance  # noqa: E402
from lumina_quant.research.vol_spillover_diagnostic import run_diagnostic  # noqa: E402

_DEFAULT_OUT = REPO_ROOT / "var" / "research" / "vol_spillover_diagnostic.json"


def _parse_day(raw: str) -> int | None:
    """Parse an epoch-day id (int) or ``YYYY-MM-DD`` date into a day id."""
    token = raw.strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        parsed = datetime.strptime(token, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None
    return int(parsed.timestamp() // 86_400)


def _load_rv_csv(path: Path) -> dict[str, dict[int, float]]:
    """Load ``symbol,day,rv`` rows into per-symbol day-keyed RV maps."""
    series: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = (row.get("symbol") or "").strip()
            day = _parse_day(row.get("day") or "")
            try:
                rv = float(row.get("rv") or "")
            except ValueError:
                continue
            if not symbol or day is None:
                continue
            series.setdefault(symbol, {})[day] = rv
    return series


def _load_closes_csv(path: Path) -> dict[str, dict[int, float]]:
    """Load ``symbol,epoch_seconds,close`` rows and aggregate daily RV."""
    closes: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = (row.get("symbol") or "").strip()
            try:
                epoch = float(row.get("epoch_seconds") or "")
                close = float(row.get("close") or "")
            except ValueError:
                continue
            if symbol:
                closes.setdefault(symbol, []).append((epoch, close))
    series: dict[str, dict[int, float]] = {}
    for symbol, rows in closes.items():
        rows.sort(key=lambda item: item[0])
        days, rv_values = daily_realized_variance(
            [close for _, close in rows], [epoch for epoch, _ in rows]
        )
        series[symbol] = dict(zip(days, rv_values, strict=False))
    return series


def _load_series_dir(path: Path) -> dict[str, dict[int, float]]:
    """Load per-symbol parquet/CSV close files from a directory (data-PC)."""
    series: dict[str, dict[int, float]] = {}
    for child in sorted(path.iterdir()):
        symbol = child.stem.upper()
        pairs: list[tuple[float, float]] = []
        if child.suffix.lower() == ".parquet":
            try:
                import polars as pl  # optional data-PC dependency
            except ImportError:
                print(f"skip {child.name}: polars unavailable for parquet", file=sys.stderr)
                continue
            frame = pl.read_parquet(child)
            time_col = next(
                (
                    name
                    for name in ("epoch_seconds", "timestamp", "open_time", "ts", "time")
                    if name in frame.columns
                ),
                None,
            )
            if time_col is None or "close" not in frame.columns:
                print(f"skip {child.name}: missing time/close columns", file=sys.stderr)
                continue
            raw_times = frame[time_col].to_list()
            raw_closes = frame["close"].to_list()
            for raw_time, raw_close in zip(raw_times, raw_closes, strict=False):
                try:
                    epoch = float(raw_time)
                    close = float(raw_close)
                except TypeError, ValueError:
                    continue
                if epoch > 1e12:  # milliseconds -> seconds
                    epoch /= 1000.0
                pairs.append((epoch, close))
        elif child.suffix.lower() == ".csv":
            with child.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        epoch = float(row.get("epoch_seconds") or "")
                        close = float(row.get("close") or "")
                    except ValueError:
                        continue
                    pairs.append((epoch, close))
        else:
            continue
        pairs.sort(key=lambda item: item[0])
        days, rv_values = daily_realized_variance(
            [close for _, close in pairs], [epoch for epoch, _ in pairs]
        )
        series[symbol] = dict(zip(days, rv_values, strict=False))
    return series


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    """Parse ``LEADER:FOLLOWER,LEADER:FOLLOWER`` into pair tuples."""
    pairs: list[tuple[str, str]] = []
    for token in raw.split(","):
        piece = token.strip()
        if not piece or ":" not in piece:
            continue
        leader, _, follower = piece.partition(":")
        if leader.strip() and follower.strip():
            pairs.append((leader.strip(), follower.strip()))
    return pairs


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rv-csv", type=Path, help="CSV symbol,day,rv of precomputed daily RV")
    source.add_argument(
        "--closes-csv", type=Path, help="CSV symbol,epoch_seconds,close intraday closes"
    )
    source.add_argument(
        "--series-path", type=Path, help="directory of per-symbol parquet/CSV close files"
    )
    parser.add_argument("--pairs", type=str, default="", help="override pairs 'L:F,L:F'")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="JSON artifact path")
    args = parser.parse_args(argv)

    if args.rv_csv is not None:
        source_path = args.rv_csv
        loader = _load_rv_csv
    elif args.closes_csv is not None:
        source_path = args.closes_csv
        loader = _load_closes_csv
    else:
        source_path = args.series_path
        loader = _load_series_dir
    if source_path is None or not source_path.exists():
        print(f"input not found: {source_path}", file=sys.stderr)
        return 2
    rv_series = loader(source_path)

    pairs = _parse_pairs(args.pairs) or None
    report = run_diagnostic(rv_series, pairs=pairs)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json(), encoding="utf-8")

    print(f"artifact: {out_path}")
    print(f"program_verdict: {report.program_verdict}")
    print(f"evaluated_pairs: {report.evaluated_pairs}")
    print(f"admitted_pairs: {list(report.admitted_pairs)}")
    print(f"insufficient_pairs: {list(report.insufficient_pairs)}")
    print(f"sizing_overlay_build_gate_open: {report.sizing_overlay_build_gate_open}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
