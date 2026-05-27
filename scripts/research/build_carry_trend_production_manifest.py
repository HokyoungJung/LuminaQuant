"""Build a focused carry/trend production-retune manifest.

This stays lightweight: it only emits candidate definitions, so downstream runs
can choose whether to execute heavy research later. The manifest is intentionally
narrow and production-biased, centered on CarryTrendFactorRotationStrategy rows
that are marked production_ready in the candidate library.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.strategy_factory.candidate_library import (
    DEFAULT_BINANCE_TOP10_PLUS_METALS,
    StrategyCandidate,
    build_binance_futures_candidates,
)

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_ROOT = ROOT / "var" / "reports" / "exact_window_backtests" / "followup_status"
GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "carry_trend_production_retune_current"
ARTIFACT_KIND = "carry_trend_production_manifest"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _candidate_rows(*, timeframes: list[str], symbols: list[str]) -> list[StrategyCandidate]:
    rows = build_binance_futures_candidates(timeframes=timeframes, symbols=symbols)
    carry_rows = [
        row
        for row in rows
        if row.strategy_class == "CarryTrendFactorRotationStrategy"
        and bool(row.metadata.get("production_ready"))
    ]
    carry_rows.sort(key=lambda row: (row.timeframe, row.name))
    return carry_rows


def build_manifest(*, timeframes: list[str], symbols: list[str]) -> dict[str, Any]:
    rows = _candidate_rows(timeframes=timeframes, symbols=symbols)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": _utc_now_iso(),
        "candidate_count": len(rows),
        "timeframes": list(timeframes),
        "symbol_universe": list(symbols),
        "notes": [
            "Focused manifest for production-safe carry/trend factor-rotation retunes.",
            "All candidates are long-biased / low-turnover variants marked production_ready in the candidate library.",
            "Use this manifest with low-memory sequential research runs when testing new alpha sleeves.",
        ],
        "candidates": [row.to_dict() for row in rows],
    }


def write_manifest(
    *, output_dir: Path, timeframes: list[str], symbols: list[str]
) -> dict[str, str]:
    payload = build_manifest(timeframes=timeframes, symbols=symbols)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "carry_trend_production_manifest_latest.json"
    md_path = output_dir / "carry_trend_production_manifest_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# carry trend production manifest",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- candidate_count: `{payload['candidate_count']}`",
        f"- timeframes: `{payload['timeframes']}`",
        f"- symbols: `{payload['symbol_universe']}`",
        "",
        "## candidates",
    ]
    for row in payload["candidates"]:
        params = dict(row.get("params") or {})
        lines.append(
            f"- `{row['name']}` | tf={row.get('strategy_timeframe')} | longs={params.get('max_longs')} | shorts={params.get('max_shorts')} | lookback={params.get('lookback_bars')} | rebalance={params.get('rebalance_bars')} | threshold={params.get('signal_threshold')}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json_path": str(json_path.resolve()), "md_path": str(md_path.resolve())}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframes", nargs="+", default=["1h", "4h"])
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_BINANCE_TOP10_PLUS_METALS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = write_manifest(
        output_dir=Path(args.output_dir).resolve(),
        timeframes=[str(item) for item in list(args.timeframes)],
        symbols=[str(item) for item in list(args.symbols)],
    )
    print(result["json_path"])
    print(result["md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
