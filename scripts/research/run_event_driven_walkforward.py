#!/usr/bin/env python3
"""Evaluate a frozen strategy shortlist across validation and locked-OOS folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _shortlist(path: Path, *, limit: int) -> tuple[str, ...]:
    payload = json.loads(path.read_bytes())
    rows = payload.get("selected")
    if type(rows) is not list:
        candidates = payload.get("strategy_results")
        if type(candidates) is not list:
            raise ValueError("shortlist must contain selected or strategy_results")
        rows = sorted(
            (
                row
                for row in candidates
                if type(row) is dict
                and row.get("status") == "pass"
                and int(row.get("trade_count") or 0) > 0
                and float(row.get("total_return") or 0.0) > 0.0
            ),
            key=lambda row: (
                float((row.get("fast_stats") or {}).get("sharpe") or 0.0),
                float(row.get("total_return") or 0.0),
            ),
            reverse=True,
        )
    names = tuple(
        dict.fromkeys(
            str(row["strategy"])
            for row in rows[:limit]
            if type(row) is dict and type(row.get("strategy")) is str
        )
    )
    if not names:
        raise ValueError("shortlist contains no strategies")
    return names


def _folds(path: Path) -> tuple[dict[str, str], ...]:
    payload = json.loads(path.read_bytes())
    rows = payload.get("folds")
    required = {
        "fold_id",
        "validation_start",
        "validation_end",
        "locked_oos_start",
        "locked_oos_end",
    }
    if type(rows) is not list or not rows:
        raise ValueError("fold plan must contain folds")
    result: list[dict[str, str]] = []
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != required
            or not all(type(row[key]) is str for key in required)
        ):
            raise ValueError("fold plan row is invalid")
        validation_start = datetime.fromisoformat(row["validation_start"].replace("Z", "+00:00"))
        validation_end = datetime.fromisoformat(row["validation_end"].replace("Z", "+00:00"))
        oos_start = datetime.fromisoformat(row["locked_oos_start"].replace("Z", "+00:00"))
        oos_end = datetime.fromisoformat(row["locked_oos_end"].replace("Z", "+00:00"))
        if not validation_start < validation_end <= oos_start < oos_end:
            raise ValueError("fold windows must be ordered and non-overlapping")
        result.append(dict(row))
    if len({row["fold_id"] for row in result}) != len(result):
        raise ValueError("fold ids must be unique")
    return tuple(result)


def _valid_cell(path: Path, *, strategy: str, start: str, end: str) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        payload = json.loads(path.read_bytes())
        rows = payload["strategy_results"]
        return (
            payload["start"] == start
            and payload["end"] == end
            and len(rows) == 1
            and rows[0]["strategy"] == strategy
        )
    except OSError, KeyError, TypeError, ValueError, json.JSONDecodeError:
        return False


def run_walkforward(
    *,
    repository: Path,
    data_root: Path,
    config: Path,
    shortlist: Path,
    fold_plan: Path,
    output_root: Path,
    exchange: str,
    limit: int,
) -> dict[str, Any]:
    strategies = _shortlist(shortlist, limit=limit)
    folds = _folds(fold_plan)
    runner = repository / "scripts" / "research" / "run_strategy_screen.py"
    output_root.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    for fold in folds:
        for phase in ("validation", "locked_oos"):
            start = fold[f"{phase}_start"]
            end = fold[f"{phase}_end"]
            for strategy in strategies:
                cell_root = output_root / "folds" / fold["fold_id"] / phase / strategy
                result_path = cell_root / "strategy_screen_latest.json"
                mode = (
                    "reused"
                    if _valid_cell(result_path, strategy=strategy, start=start, end=end)
                    else "ran"
                )
                return_code = 0
                if mode == "ran":
                    cell_root.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable,
                        str(runner),
                        "--data-root",
                        str(data_root),
                        "--exchange",
                        exchange,
                        "--start",
                        start,
                        "--end",
                        end,
                        "--strategy",
                        strategy,
                        "--output-dir",
                        str(cell_root),
                        "--config",
                        str(config),
                        "--no-exchange-audit",
                        "--allow-unavailable",
                    ]
                    with (
                        (cell_root / "stdout.log").open("wb") as stdout,
                        (cell_root / "stderr.log").open("wb") as stderr,
                    ):
                        completed = subprocess.run(
                            command,
                            cwd=repository,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout,
                            stderr=stderr,
                            check=False,
                        )
                    return_code = completed.returncode
                if not _valid_cell(result_path, strategy=strategy, start=start, end=end):
                    raise RuntimeError(
                        f"walkforward cell did not publish a valid result:{fold['fold_id']}:{phase}:{strategy}"
                    )
                payload = json.loads(result_path.read_bytes())
                row = payload["strategy_results"][0]
                cells.append(
                    {
                        "fold_id": fold["fold_id"],
                        "phase": phase,
                        "strategy": strategy,
                        "status": row["status"],
                        "total_return": row.get("total_return"),
                        "trade_count": row.get("trade_count"),
                        "fast_stats": row.get("fast_stats") or {},
                        "mode": mode,
                        "return_code": return_code,
                        "result_path": str(result_path),
                        "result_sha256": _sha256(result_path),
                    }
                )
    result = {
        "artifact_kind": "lumina_quant.event_driven_walkforward.v1",
        "status": "complete",
        "selection_uses_locked_oos": False,
        "shortlist": {
            "path": str(shortlist.resolve()),
            "sha256": _sha256(shortlist),
            "strategies": list(strategies),
        },
        "fold_plan": {"path": str(fold_plan.resolve()), "sha256": _sha256(fold_plan)},
        "data_root": str(data_root.resolve()),
        "config": {"path": str(config.resolve()), "sha256": _sha256(config)},
        "order_routing_enabled": False,
        "status_counts": dict(sorted(Counter(cell["status"] for cell in cells).items())),
        "cells": cells,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write(output_root / "walkforward_result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shortlist", required=True, type=Path)
    parser.add_argument("--fold-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    result = run_walkforward(
        repository=repository,
        data_root=args.data_root.resolve(),
        config=args.config.resolve(),
        shortlist=args.shortlist.resolve(),
        fold_plan=args.fold_plan.resolve(),
        output_root=args.output_root.resolve(),
        exchange=args.exchange,
        limit=max(1, args.limit),
    )
    print(json.dumps({"status": result["status"], "cells": len(result["cells"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
