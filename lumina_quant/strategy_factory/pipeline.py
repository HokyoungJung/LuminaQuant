"""Pipeline helpers for strategy-factory orchestration."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strategies.factory_candidate_set import (
    DEFAULT_TIMEFRAMES,
    build_candidate_set,
    summarize_candidate_set,
)
from strategies.factory_candidate_set import (
    DEFAULT_TOP10_PLUS_METALS as DEFAULT_BINANCE_TOP10_PLUS_METALS,
)

from .selection import select_diversified_shortlist, summarize_shortlist


def extract_saved_report_path(output: str) -> Path | None:
    markers = ("Saved:", "Saved report:")
    for line in output.splitlines()[::-1]:
        for marker in markers:
            if marker in line:
                text = line.split(marker, 1)[1].strip()
                if text:
                    return Path(text)
    return None


def build_research_command(
    *,
    db_path: str,
    backend: str = "parquet-postgres",
    exchange: str,
    market_type: str,
    mode: str,
    strategy_set: str,
    base_timeframe: str,
    base_timeframes: Sequence[str],
    timeframes: Sequence[str],
    seeds: Sequence[int],
    topcap_symbols: Sequence[str],
    max_selected: int,
    max_per_family: int,
    max_per_timeframe: int,
    max_runs: int,
    candidate_manifest: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_strategy_team_research.py",
        "--db-path",
        str(db_path),
        "--exchange",
        str(exchange),
        "--market-type",
        str(market_type),
        "--mode",
        str(mode),
        "--strategy-set",
        str(strategy_set),
        "--base-timeframe",
        str(base_timeframe),
        "--max-selected",
        str(int(max_selected)),
        "--max-per-family",
        str(int(max_per_family)),
        "--max-per-timeframe",
        str(int(max_per_timeframe)),
        "--max-runs",
        str(int(max_runs)),
    ]
    if str(backend).strip():
        cmd.extend(["--backend", str(backend).strip()])

    if base_timeframes:
        cmd.append("--base-timeframes")
        cmd.extend(str(token) for token in base_timeframes)
    if timeframes:
        cmd.append("--timeframes")
        cmd.extend(str(token) for token in timeframes)
    if seeds:
        cmd.append("--seeds")
        cmd.extend(str(int(seed)) for seed in seeds)
    if topcap_symbols:
        cmd.append("--topcap-symbols")
        cmd.extend(str(symbol) for symbol in topcap_symbols)
    if candidate_manifest:
        cmd.extend(["--candidate-manifest", str(candidate_manifest)])

    return cmd


def write_candidate_manifest(
    *,
    output_dir: Path,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> tuple[Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidate_set(timeframes=timeframes, symbols=symbols)
    summary = summarize_candidate_set(candidates)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(candidates),
        "summary": summary,
        "timeframes": list(timeframes),
        "symbol_universe": list(symbols),
        "candidates": candidates,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"strategy_factory_candidates_{stamp}.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    return path, manifest


def build_shortlist_payload(
    *,
    report: dict[str, Any],
    mode: str,
    shortlist_max_total: int,
    shortlist_max_per_family: int,
    shortlist_max_per_timeframe: int,
    manifest_path: Path,
    research_report_path: Path,
) -> dict[str, Any]:
    selected_team = list(report.get("selected_team") or [])
    shortlist = select_diversified_shortlist(
        selected_team,
        mode=mode,
        max_total=shortlist_max_total,
        max_per_family=shortlist_max_per_family,
        max_per_timeframe=shortlist_max_per_timeframe,
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": str(mode),
        "manifest_path": str(manifest_path),
        "research_report_path": str(research_report_path),
        "selected_team_input_count": len(selected_team),
        "shortlist_count": len(shortlist),
        "summary": summarize_shortlist(shortlist),
        "shortlist": shortlist,
    }


def render_shortlist_markdown(shortlist_payload: dict[str, Any]) -> str:
    rows = list(shortlist_payload.get("shortlist") or [])
    lines = [
        "# Strategy Factory Portfolio Shortlist",
        "",
        f"- Generated: {shortlist_payload.get('generated_at', '')}",
        f"- Mode: {shortlist_payload.get('mode', '')}",
        f"- Manifest: `{shortlist_payload.get('manifest_path', '')}`",
        f"- Research report: `{shortlist_payload.get('research_report_path', '')}`",
        f"- Candidate count: {shortlist_payload.get('shortlist_count', 0)}",
        "",
        "## Family / timeframe mix",
        "",
    ]

    summary = shortlist_payload.get("summary") if isinstance(shortlist_payload.get("summary"), dict) else {}
    family_summary = summary.get("family") if isinstance(summary, dict) else {}
    timeframe_summary = summary.get("timeframe") if isinstance(summary, dict) else {}

    if family_summary:
        lines.append("- Family:")
        for key in sorted(family_summary):
            lines.append(f"  - {key}: {family_summary[key]}")
    if timeframe_summary:
        lines.append("- Timeframe:")
        for key in sorted(timeframe_summary):
            lines.append(f"  - {key}: {timeframe_summary[key]}")

    lines.extend(
        [
            "",
            "## Top candidates",
            "",
            "| # | Name | Timeframe | Family | Score | Symbols |",
            "|---:|---|---|---|---:|---:|",
        ]
    )

    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{idx} | "
            f"{row.get('name', '')} | "
            f"{row.get('strategy_timeframe') or row.get('timeframe') or ''} | "
            f"{row.get('family', '')} | "
            f"{float(row.get('shortlist_score', 0.0)):.4f} | "
            f"{len(row.get('symbols') or [])} |"
        )

    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python scripts/run_strategy_factory_pipeline.py --backend parquet-postgres"
    )
    lines.append("```")
    lines.append("")

    return "\n".join(lines)
