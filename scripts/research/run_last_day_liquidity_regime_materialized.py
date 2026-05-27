"""Run last-day liquidity-regime candidate research from materialized OHLCV.

This avoids the generic candidate runner's heavy raw-fallback path by reading the
already materialized higher-timeframe parquet partitions directly.
"""

from __future__ import annotations

import argparse
import json
import resource
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.strategy_factory import research_run_support, research_runner
from lumina_quant.strategy_factory.runtime_settings import current_research_market_data_settings


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _compact_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "").upper()


def _parse_date_token(value: str) -> datetime:
    normalized = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _materialized_file_paths(
    *,
    parquet_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> list[str]:
    start_dt = _parse_date_token(start_date).date()
    end_dt = _parse_date_token(end_date).date()
    base = (
        parquet_root
        / "market_data_materialized"
        / str(exchange).lower()
        / _compact_symbol(symbol)
        / f"timeframe={timeframe}"
    )
    if not base.exists():
        return []

    files: list[str] = []
    for date_dir in sorted(base.glob("date=*")):
        token = str(date_dir.name).split("=", 1)[-1]
        try:
            current_date = datetime.fromisoformat(f"{token}T00:00:00+00:00").date()
        except Exception:
            continue
        if current_date < start_dt or current_date > end_dt:
            continue
        commits = sorted(date_dir.glob("commit=*/part-*.parquet"))
        if commits:
            files.extend(str(path.resolve()) for path in commits)
    return files


def _load_materialized_bundle(
    *,
    parquet_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> research_runner.SeriesBundle:
    files = _materialized_file_paths(
        parquet_root=parquet_root,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    if not files:
        raise FileNotFoundError(f"no materialized parquet files for {symbol}@{timeframe}")

    frame = (
        pl.scan_parquet(files)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .sort("datetime")
        .collect(streaming=True)
    )
    if frame.is_empty():
        raise ValueError(f"empty materialized frame for {symbol}@{timeframe}")
    return research_runner.SeriesBundle(
        symbol=symbol,
        timeframe=timeframe,
        datetime=frame["datetime"].to_numpy(),
        open=frame["open"].to_numpy(),
        high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(),
        close=frame["close"].to_numpy(),
        volume=frame["volume"].to_numpy(),
    )


def _load_manifest_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        research_run_support.adapt_legacy_candidate(dict(row))
        for row in list(payload.get("candidates") or [])
        if isinstance(row, dict)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--oos-start", required=True)
    parser.add_argument("--oos-end", required=True)
    return parser


def _resolved_split(*, timeframe: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "train_start": str(args.train_start),
        "train_end": str(args.train_end),
        "val_start": str(args.validation_start),
        "val_end": str(args.validation_end),
        "oos_start": str(args.oos_start),
        "oos_end": str(args.oos_end),
        "strategy_timeframe": str(timeframe),
        "mode": "materialized_exact_window",
    }


def main() -> None:
    args = _build_parser().parse_args()
    candidates = _load_manifest_candidates(Path(args.manifest).resolve())
    if not candidates:
        raise SystemExit("no candidates in manifest")

    timeframe = str(
        candidates[0].get("strategy_timeframe") or candidates[0].get("timeframe") or "1d"
    )
    _, universe = research_run_support._resolve_research_run_timeframes_and_universe(
        adapted=candidates,
        strategy_timeframes=[timeframe],
        symbol_universe=None,
    )

    settings = current_research_market_data_settings()
    parquet_root = Path(str(settings["parquet_root"])).resolve()
    exchange = str(settings["exchange"])
    resolved_split = _resolved_split(timeframe=timeframe, args=args)
    cache: dict[tuple[str, str], research_runner.SeriesBundle] = {}
    for symbol in universe:
        cache[(symbol, timeframe)] = _load_materialized_bundle(
            parquet_root=parquet_root,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=str(args.train_start),
            end_date=str(args.oos_end),
        )

    benchmark_cache = research_runner._benchmark_cache(cache, [timeframe])
    scoring = research_run_support._resolve_research_run_scoring_config(
        score_config=None, stage1_keep_ratio=1.0
    )
    stage2_results = [
        research_runner._evaluate_candidate(
            candidate,
            cache=cache,
            feature_cache={},
            benchmark_cache=benchmark_cache,
            candidate_count=len(candidates),
            scoring_config=scoring.resolved_scoring_config,
            split=resolved_split,
        )
        for candidate in candidates
    ]
    report_candidates = research_runner._report_candidates_from_stage2_results(
        stage2_results=stage2_results,
        candidate_count=len(candidates),
        resolved_split=resolved_split,
        scoring=scoring,
    )
    report_candidates = research_runner._sorted_report_candidates(
        report_candidates, scoring=scoring
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v2",
        "generated_at": _utc_now_iso(),
        "artifact_kind": "candidate_research_materialized",
        "base_timeframe": timeframe,
        "strategy_timeframes": [timeframe],
        "symbol_universe": universe,
        "split": resolved_split,
        "candidates": report_candidates,
        "stage1": {
            "input_count": len(candidates),
            "selected_count": len(stage2_results),
            "keep_ratio": 1.0,
            "keep_ratio_applied": 1.0,
        },
        "data_sources": {"materialized": [f"{symbol}@{timeframe}" for symbol in universe]},
        "memory": {
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    json_path = output_dir / "candidate_research_latest.json"
    md_path = output_dir / "candidate_research_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# candidate research (materialized)",
                "",
                f"- generated_at: `{payload['generated_at']}`",
                f"- timeframe: `{timeframe}`",
                f"- symbol_universe: `{', '.join(universe)}`",
                f"- peak_rss_kib: `{payload['memory']['peak_rss_kib']}`",
                "",
                "## candidates",
                *[
                    f"- `{row['name']}` | train_return `{float((row.get('train') or {}).get('total_return', (row.get('train') or {}).get('return', 0.0))):.4%}` | val_return `{float((row.get('val') or {}).get('total_return', (row.get('val') or {}).get('return', 0.0))):.4%}` | oos_return `{float((row.get('oos') or {}).get('total_return', (row.get('oos') or {}).get('return', 0.0))):.4%}` | oos_sharpe `{float((row.get('oos') or {}).get('sharpe', 0.0)):.4f}`"
                    for row in report_candidates
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
