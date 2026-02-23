"""Select a diversified strategy shortlist from multi-timeframe report files."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _candidate_identity(row: dict) -> str:
    payload = {
        "name": str(row.get("name", "")),
        "timeframe": str(row.get("strategy_timeframe", "")),
        "symbols": list(row.get("symbols") or []),
        "params": row.get("params") if isinstance(row.get("params"), dict) else {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_score(row: dict, *, mode: str, require_pass: bool) -> float:
    hurdle = (row.get("hurdle_fields") or {}).get(mode) or {}
    pass_flag = bool(hurdle.get("pass", False))
    base_score = _safe_float(hurdle.get("score"), _safe_float(row.get("selection_score"), -1e9))

    metrics = row.get(mode) if isinstance(row.get(mode), dict) else {}
    ret = _safe_float(metrics.get("return"), 0.0)
    sharpe = _safe_float(metrics.get("sharpe"), 0.0)
    drawdown = _safe_float(metrics.get("mdd"), 0.0)
    trades = _safe_float(metrics.get("trades"), 0.0)

    composite = base_score + (0.35 * sharpe) + (45.0 * ret) - (20.0 * drawdown)
    if trades > 0.0:
        composite += min(0.4, 0.03 * trades)

    if require_pass and not pass_flag:
        composite -= 100.0
    return composite


def _flatten_reports(paths: list[str], *, mode: str) -> list[dict]:
    rows: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        split = data.get("split") if isinstance(data.get("split"), dict) else {}
        timeframe = str(split.get("strategy_timeframe", "")).strip().lower()

        for candidate in list(data.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            row = dict(candidate)
            row["strategy_timeframe"] = timeframe
            row["source_report"] = str(path)
            row["identity"] = _candidate_identity(row)
            rows.append(row)
    return rows


def _select_diversified(
    rows: list[dict],
    *,
    mode: str,
    max_selected: int,
    max_per_strategy: int,
    max_per_timeframe: int,
    max_per_symbol: int,
    require_pass: bool,
    min_trades: int,
) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda row: _candidate_score(row, mode=mode, require_pass=require_pass),
        reverse=True,
    )

    selected: list[dict] = []
    seen_identities: set[str] = set()
    strategy_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}

    for row in ranked:
        if len(selected) >= int(max_selected):
            break

        identity = str(row.get("identity", "")).strip()
        if not identity or identity in seen_identities:
            continue

        timeframe = str(row.get("strategy_timeframe", "")).strip().lower()
        strategy_name = str(row.get("name", "")).strip().lower()
        symbols = [str(symbol).strip().upper() for symbol in list(row.get("symbols") or [])]

        metrics = row.get(mode) if isinstance(row.get(mode), dict) else {}
        trades = int(max(0.0, _safe_float(metrics.get("trades"), 0.0)))
        if trades < int(min_trades):
            continue

        if strategy_counts.get(strategy_name, 0) >= int(max_per_strategy):
            continue
        if timeframe_counts.get(timeframe, 0) >= int(max_per_timeframe):
            continue
        if any(symbol_counts.get(symbol, 0) >= int(max_per_symbol) for symbol in symbols):
            continue

        row = dict(row)
        row["selection_score"] = _candidate_score(row, mode=mode, require_pass=require_pass)
        selected.append(row)
        seen_identities.add(identity)
        strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    return selected


def _summarize(rows: list[dict]) -> dict[str, object]:
    strategy_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}

    for row in rows:
        strategy = str(row.get("name", "")).strip().lower()
        timeframe = str(row.get("strategy_timeframe", "")).strip().lower()
        symbols = [str(symbol).strip().upper() for symbol in list(row.get("symbols") or [])]

        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    return {
        "count": len(rows),
        "strategies": dict(sorted(strategy_counts.items(), key=lambda item: item[0])),
        "timeframes": dict(sorted(timeframe_counts.items(), key=lambda item: item[0])),
        "symbols": dict(sorted(symbol_counts.items(), key=lambda item: item[0])),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select diversified shortlist from oos_guarded_multistrategy reports."
    )
    parser.add_argument(
        "--report-glob",
        default="reports/oos_guarded_multistrategy_oos_*.json",
        help="Glob pattern for input report files.",
    )
    parser.add_argument("--mode", choices=["train", "val", "oos"], default="oos")
    parser.add_argument("--max-selected", type=int, default=32)
    parser.add_argument("--max-per-strategy", type=int, default=8)
    parser.add_argument("--max-per-timeframe", type=int, default=6)
    parser.add_argument("--max-per-symbol", type=int, default=4)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = sorted(glob.glob(str(args.report_glob)))
    if not paths:
        raise RuntimeError(f"No report files matched: {args.report_glob}")

    rows = _flatten_reports(paths, mode=str(args.mode))
    selected = _select_diversified(
        rows,
        mode=str(args.mode),
        max_selected=max(1, int(args.max_selected)),
        max_per_strategy=max(1, int(args.max_per_strategy)),
        max_per_timeframe=max(1, int(args.max_per_timeframe)),
        max_per_symbol=max(1, int(args.max_per_symbol)),
        require_pass=bool(args.require_pass),
        min_trades=max(0, int(args.min_trades)),
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output)
        if str(args.output).strip()
        else Path("reports") / f"strategy_factory_shortlist_{args.mode}_{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "input_report_count": len(paths),
        "input_candidate_count": len(rows),
        "selected_count": len(selected),
        "selected_summary": _summarize(selected),
        "selected": selected,
    }
    indent = 2 if bool(args.pretty) else None
    output.write_text(json.dumps(payload, indent=indent), encoding="utf-8")

    print(f"Saved shortlist: {output}")
    print(f"Input candidates: {len(rows)}")
    print(f"Selected: {len(selected)}")


if __name__ == "__main__":
    main()
