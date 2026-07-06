"""Offline enablement runner: build + PERSIST the alpha scoreboard leaderboard.

Where ``build_alpha_scoreboard.py`` is the low-level "input -> two explicit
output paths" CLI, this runner is the one-command operation that the
data-bearing PC uses to *persist an audited ranking*: it loads candidate /
backtest result rows (from a passed manifest file OR a ``var/reports`` directory
of exported row files), builds the scoreboard, and writes a VERSIONED artifact
plus a ``_latest`` pointer for both ``.json`` and ``.md`` -- mirroring the
repo's ``<name>_<stamp>.{json,md}`` + ``<name>_latest.{json,md}`` convention.

This closes the audit gap noted in
``docs/audits/strategy_viability_assessment_20260706.md`` ("``alpha_scoreboard``
has never been run to persist a single audited ranking"). It does NOT run the
real large-scale scoreboard (that needs the data PC's result rows); it makes
running + persisting it a single command over whatever rows are supplied.

Input shapes accepted (file or every ``*.json`` in a directory):
  - a JSON list of result rows, or
  - a JSON object with a ``rows`` / ``candidates`` / ``results`` / ``records``
    list, or
  - a single result-row object (has ``id`` / ``candidate_id`` / ``name``).

Each row uses the canonical flat schema consumed by ``alpha_scoreboard``
(``id``, ``metrics``, ``returns``, ``turnover``, ``trade_count``,
``liquidation_count``, ``bars``, ...). Output is deterministic for a given
input + pinned ``--version`` (sorted keys, stable tie-breaks, no wall clock in
the body).

Usage:
    .venv/bin/python scripts/research/run_alpha_scoreboard.py \
        --input rows.json \
        [--output-dir var/reports/alpha_scoreboard] \
        [--name alpha_scoreboard] [--top-n 10] \
        [--gates '{"max_mdd": 0.3, "min_trades": 5}'] \
        [--weights '{"sharpe": 2.0, "mdd": 1.0, "return": 1.0}'] \
        [--periods-per-year 365] [--version 20260706T000000Z]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.research.alpha_scoreboard import (
    DEFAULT_PERIODS_PER_YEAR,
    build_scoreboard,
    render_scoreboard_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "var" / "reports" / "alpha_scoreboard"
DEFAULT_NAME = "alpha_scoreboard"
_ROW_LIST_KEYS = ("rows", "candidates", "results", "records")
_ROW_ID_KEYS = ("id", "candidate_id", "name")


def _rows_from_obj(obj: Any) -> list[dict[str, Any]]:
    """Extract result-row dicts from one decoded JSON object."""
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in _ROW_LIST_KEYS:
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(obj.get(key) for key in _ROW_ID_KEYS):
            return [obj]
    return []


def load_rows(input_path: str | Path) -> list[dict[str, Any]]:
    """Load result rows from a manifest file or a directory of row files.

    A directory is scanned for every ``*.json`` (sorted for determinism); each
    file contributes its rows. Files that decode to something without rows
    (config blobs, prior scoreboard outputs, ...) contribute nothing.
    """
    path = Path(input_path)
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for file_path in sorted(path.glob("*.json")):
            try:
                obj = json.loads(file_path.read_text(encoding="utf-8"))
            except OSError, json.JSONDecodeError:
                continue
            rows.extend(_rows_from_obj(obj))
        return rows
    obj = json.loads(path.read_text(encoding="utf-8"))
    return _rows_from_obj(obj)


def run_scoreboard(
    raw_rows: list[dict[str, Any]],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = DEFAULT_NAME,
    title: str = "Alpha Scoreboard",
    gates: dict[str, float | int | None] | None = None,
    weights: dict[str, float] | None = None,
    top_n: int = 10,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    version: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Build the scoreboard and persist versioned + ``_latest`` json/md.

    ``version`` pins the artifact stamp (default: current UTC
    ``%Y%m%dT%H%M%SZ``); pass it for reproducible / testable runs. The persisted
    JSON is the :func:`build_scoreboard` payload with ``version`` + ``source``
    provenance keys added at top level. Returns a dict with the resolved
    ``version``, the persisted ``payload``, and the four written ``paths``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    payload = build_scoreboard(
        raw_rows,
        gates=gates,
        weights=weights,
        top_n=top_n,
        periods_per_year=periods_per_year,
    )
    leaderboard: dict[str, Any] = {"version": stamp, "source": source, **payload}
    json_text = json.dumps(leaderboard, sort_keys=True, indent=2) + "\n"

    md_text = render_scoreboard_markdown(payload, title=title)
    if not md_text.endswith("\n"):
        md_text += "\n"
    md_text += f"\n_version: {stamp}_\n"

    versioned_json = out_dir / f"{name}_{stamp}.json"
    versioned_md = out_dir / f"{name}_{stamp}.md"
    latest_json = out_dir / f"{name}_latest.json"
    latest_md = out_dir / f"{name}_latest.md"
    for path in (versioned_json, latest_json):
        path.write_text(json_text, encoding="utf-8")
    for path in (versioned_md, latest_md):
        path.write_text(md_text, encoding="utf-8")

    return {
        "version": stamp,
        "payload": leaderboard,
        "paths": {
            "versioned_json": versioned_json,
            "versioned_md": versioned_md,
            "latest_json": latest_json,
            "latest_md": latest_md,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Result-row JSON file OR a directory of row files"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=DEFAULT_NAME, help="Artifact basename")
    parser.add_argument("--title", default="Alpha Scoreboard")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--gates", default="", help="JSON dict overriding default gates")
    parser.add_argument("--weights", default="", help="JSON dict of composite metric weights")
    parser.add_argument("--periods-per-year", type=int, default=DEFAULT_PERIODS_PER_YEAR)
    parser.add_argument("--version", default="", help="Pin the artifact stamp (reproducible runs)")
    args = parser.parse_args(argv)

    raw_rows = load_rows(args.input)
    if not raw_rows:
        raise SystemExit(f"no result rows found at {args.input}")
    gates = json.loads(args.gates) if args.gates else None
    weights = json.loads(args.weights) if args.weights else None

    result = run_scoreboard(
        raw_rows,
        output_dir=args.output_dir,
        name=args.name,
        title=args.title,
        gates=gates,
        weights=weights,
        top_n=args.top_n,
        periods_per_year=args.periods_per_year,
        version=(args.version or None),
        source=str(args.input),
    )
    payload = result["payload"]
    top = ", ".join(item["id"] for item in payload["composite"][:3])
    print(
        f"alpha_scoreboard {result['version']}: "
        f"{payload['eligible_count']}/{payload['input_count']} eligible; "
        f"latest -> {result['paths']['latest_json']}; top: {top or '(none)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
