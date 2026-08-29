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
import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumina_quant.indicators.har_rv import daily_realized_variance  # noqa: E402
from lumina_quant.research.vol_spillover_diagnostic import (  # noqa: E402
    _mint_cli_authority,
    run_diagnostic,
)

_DEFAULT_OUT_DIR = REPO_ROOT / "var" / "research"
_LOADER_VERSION = "vdiag-cli-v3"


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
    """Load explicit daily RV rows; reject every ambiguous or degraded row."""
    series: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = (row.get("symbol") or "").strip()
            day = _parse_day(row.get("day") or "")
            try:
                rv = float(row.get("rv") or "")
            except ValueError:
                raise ValueError("non-numeric RV")
            if not symbol or day is None or not math.isfinite(rv) or rv < 0.0:
                raise ValueError("invalid symbol, day, or RV")
            target = series.setdefault(symbol, {})
            if day in target:
                raise ValueError(f"duplicate day for {symbol}")
            target[day] = rv
    return series


def _strict_epoch(raw: object) -> int:
    """Accept only unambiguous epoch-second integer tokens."""
    if not isinstance(raw, str) or not raw.strip().isdigit():
        raise ValueError("timestamp must be an integer epoch-second token")
    epoch = int(raw.strip())
    if not 946684800 <= epoch <= 4102444800:
        raise ValueError("timestamp unit is not unambiguous epoch seconds")
    return epoch


def _registered_cadence(rows: list[tuple[int, float]]) -> int:
    """Return the single complete UTC-day cadence registered by close rows."""
    if not rows:
        raise ValueError("empty intraday series")
    if len({epoch for epoch, _ in rows}) != len(rows):
        raise ValueError("duplicate intraday timestamp")
    rows.sort()
    by_day: dict[int, list[int]] = {}
    for epoch, _ in rows:
        by_day.setdefault(epoch // 86_400, []).append(epoch)
    cadences: set[int] = set()
    for day, times in by_day.items():
        if len(times) < 13:
            raise ValueError("unregistered intraday coverage")
        deltas = [later - earlier for earlier, later in pairwise(times)]
        if not deltas or len(set(deltas)) != 1 or deltas[0] <= 0:
            raise ValueError("irregular intraday coverage")
        cadence = deltas[0]
        if 86_400 % cadence:
            raise ValueError("cadence does not divide a UTC day")
        if times != list(range(day * 86_400, (day + 1) * 86_400, cadence)):
            raise ValueError("partial UTC-day bucket grid")
        cadences.add(cadence)
    if len(cadences) != 1:
        raise ValueError("inconsistent intraday cadence")
    return cadences.pop()


def _aggregate_strict(rows: list[tuple[int, float]], cadence_seconds: int) -> dict[int, float]:
    """Aggregate one symbol against the registered complete UTC-day grid."""
    if _registered_cadence(rows) != cadence_seconds:
        raise ValueError("inconsistent intraday cadence")
    days, rv_values = daily_realized_variance(
        [close for _, close in rows], [epoch for epoch, _ in rows]
    )
    expected_days = {epoch // 86_400 for epoch, _ in rows}
    if len(days) != len(expected_days) or any(not math.isfinite(rv) or rv < 0 for rv in rv_values):
        raise ValueError("intraday aggregation dropped a day")
    return dict(zip(days, rv_values, strict=True))


def _load_closes_csv(path: Path) -> dict[str, dict[int, float]]:
    """Load ``symbol,epoch_seconds,close`` rows and aggregate daily RV."""
    closes: dict[str, list[tuple[int, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = (row.get("symbol") or "").strip()
            try:
                epoch = _strict_epoch(row.get("epoch_seconds") or "")
                close = float(row.get("close") or "")
            except ValueError as exc:
                raise ValueError("invalid closes CSV row") from exc
            if not symbol or not math.isfinite(close) or close <= 0.0:
                raise ValueError("invalid symbol or close")
            closes.setdefault(symbol, []).append((epoch, close))
    series: dict[str, dict[int, float]] = {}
    cadence_seconds: int | None = None
    for symbol, rows in closes.items():
        candidate = _registered_cadence(rows)
        if cadence_seconds is None:
            cadence_seconds = candidate
        elif candidate != cadence_seconds:
            raise ValueError("inconsistent intraday cadence across symbols")
        series[symbol] = _aggregate_strict(rows, cadence_seconds)
    return series


def _load_series_dir(path: Path) -> dict[str, dict[int, float]]:
    """Load per-symbol parquet/CSV close files from a directory (data-PC)."""
    series: dict[str, dict[int, float]] = {}
    cadence_seconds: int | None = None
    for child in sorted(path.iterdir()):
        symbol = child.stem.upper()
        if symbol in series:
            raise ValueError(f"duplicate symbol file for {symbol}")
        pairs: list[tuple[int, float]] = []
        if child.suffix.lower() == ".parquet":
            try:
                import polars as pl  # optional data-PC dependency
            except ImportError:
                raise ValueError(f"{child.name}: polars is required for parquet input")
            frame = pl.read_parquet(child)
            if "epoch_seconds" not in frame.columns or "close" not in frame.columns:
                raise ValueError(f"{child.name}: requires epoch_seconds and close columns")
            raw_times = frame["epoch_seconds"].to_list()
            raw_closes = frame["close"].to_list()
            for raw_time, raw_close in zip(raw_times, raw_closes, strict=False):
                try:
                    if type(raw_time) is not int:
                        raise ValueError("non-integer timestamp")
                    epoch = _strict_epoch(str(raw_time))
                    close = float(raw_close)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{child.name}: invalid parquet row") from exc
                if not math.isfinite(close) or close <= 0:
                    raise ValueError(f"{child.name}: invalid close")
                pairs.append((epoch, close))
        elif child.suffix.lower() == ".csv":
            with child.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        epoch = _strict_epoch(row.get("epoch_seconds") or "")
                        close = float(row.get("close") or "")
                    except ValueError as exc:
                        raise ValueError(f"{child.name}: invalid CSV row") from exc
                    if not math.isfinite(close) or close <= 0:
                        raise ValueError(f"{child.name}: invalid close")
                    pairs.append((epoch, close))
        else:
            continue
        candidate = _registered_cadence(pairs)
        if cadence_seconds is None:
            cadence_seconds = candidate
        elif candidate != cadence_seconds:
            raise ValueError("inconsistent intraday cadence across symbols")
        series[symbol] = _aggregate_strict(pairs, cadence_seconds)
    return series


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    """Parse ``LEADER:FOLLOWER,LEADER:FOLLOWER`` into pair tuples."""
    if not raw or not raw.strip():
        raise ValueError("pairs override must not be empty")
    pairs: list[tuple[str, str]] = []
    for token in raw.split(","):
        piece = token.strip()
        if not piece or piece.count(":") != 1:
            raise ValueError(f"malformed pair token: {token!r}")
        leader, follower = (part.strip() for part in piece.split(":"))
        if not leader or not follower:
            raise ValueError(f"malformed pair token: {token!r}")
        pairs.append((leader, follower))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate pair override")
    return pairs


def _source_content_sha256(path: Path) -> str:
    """Hash the exact file or directory payload consumed by the CLI loader."""
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError("source is neither a file nor a directory")
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _publish_json_noreplace(path: Path, payload: object) -> None:
    """Atomically publish exactly once; diagnostic receipts are immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace publication is unsupported")
        result = renameat2(-100, os.fsencode(temporary), -100, os.fsencode(path), 1)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(path)
        raise OSError(error_number, os.strerror(error_number), str(path))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


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
    parser.add_argument("--pairs", type=str, default=None, help="override pairs 'L:F,L:F'")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON artifact path; omitted paths are unique per invocation",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = _DEFAULT_OUT_DIR / (
            f"vol_spillover_diagnostic-{time.time_ns()}-{os.getpid()}.json"
        )

    try:
        pairs = None if args.pairs is None else _parse_pairs(args.pairs)
    except ValueError as exc:
        manifest = {
            "input_manifest": {
                "loader_aggregation_version": _LOADER_VERSION,
                "quality": "rejected",
                "reason": str(exc),
            },
            "program_verdict": "insufficient_data",
            "sizing_overlay_build_gate_open": False,
        }
        _publish_json_noreplace(args.out, manifest)
        print(f"input rejected: {exc}", file=sys.stderr)
        return 2

    if args.rv_csv is not None:
        source_path = args.rv_csv
        loader = _load_rv_csv
        source_authority = "explicit_epoch_day"
    elif args.closes_csv is not None:
        source_path = args.closes_csv
        loader = _load_closes_csv
        source_authority = "canonical_intraday"
    else:
        source_path = args.series_path
        loader = _load_series_dir
        source_authority = "canonical_intraday_series_dir"
    if source_path is None or not source_path.exists():
        _publish_json_noreplace(
            args.out,
            {
                "input_manifest": {
                    "loader_aggregation_version": _LOADER_VERSION,
                    "source": str(source_path),
                    "quality": "rejected",
                    "reason": "input not found",
                },
                "program_verdict": "insufficient_data",
                "sizing_overlay_build_gate_open": False,
            },
        )
        print(f"input not found: {source_path}", file=sys.stderr)
        return 2
    try:
        source_digest = _source_content_sha256(source_path)
        rv_series = loader(source_path)
        if source_digest != _source_content_sha256(source_path):
            raise ValueError("source changed while loading")
        authority = _mint_cli_authority(
            rv_series,
            source_authority=source_authority,
            loader=loader.__name__,
            loader_version=_LOADER_VERSION,
            source_identity=str(source_path.resolve()),
            source_content_sha256=source_digest,
        )
    except (OSError, ValueError) as exc:
        manifest = {
            "input_manifest": {
                "loader_aggregation_version": _LOADER_VERSION,
                "source": str(source_path),
                "quality": "rejected",
                "reason": str(exc),
            },
            "program_verdict": "insufficient_data",
            "sizing_overlay_build_gate_open": False,
        }
        _publish_json_noreplace(args.out, manifest)
        print(f"input rejected: {exc}", file=sys.stderr)
        return 2

    report = run_diagnostic(
        rv_series,
        pairs=pairs,
        authority=authority,
    )

    out_path = args.out
    _publish_json_noreplace(out_path, json.loads(report.to_json()))

    print(f"artifact: {out_path}")
    print(f"program_verdict: {report.program_verdict}")
    print(f"evaluated_pairs: {report.evaluated_pairs}")
    print(f"admitted_pairs: {list(report.admitted_pairs)}")
    print(f"insufficient_pairs: {list(report.insufficient_pairs)}")
    print(f"sizing_overlay_build_gate_open: {report.sizing_overlay_build_gate_open}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
