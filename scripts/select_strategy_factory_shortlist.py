"""Select a diversified strategy shortlist from multi-timeframe report files."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
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
    single_min_score: float | None = 0.0,
    single_min_return: float | None = None,
    single_min_sharpe: float | None = None,
    single_min_trades: int | None = 20,
    drop_single_without_metrics: bool = False,
    allow_multi_asset: bool = False,
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
        is_single = len(symbols) <= 1
        is_multi_asset = len(symbols) >= 3
        if not bool(allow_multi_asset) and is_multi_asset:
            continue

        metrics = row.get(mode) if isinstance(row.get(mode), dict) else {}
        trades = int(max(0.0, _safe_float(metrics.get("trades"), 0.0)))
        if trades < int(min_trades):
            continue

        selection_score = _candidate_score(row, mode=mode, require_pass=require_pass)
        if is_single:
            if bool(drop_single_without_metrics) and not metrics:
                continue
            if single_min_score is not None and selection_score < float(single_min_score):
                continue
            if single_min_return is not None:
                metric_return = _safe_float(metrics.get("return"), float("-inf"))
                if metric_return < float(single_min_return):
                    continue
            if single_min_sharpe is not None:
                metric_sharpe = _safe_float(metrics.get("sharpe"), float("-inf"))
                if metric_sharpe < float(single_min_sharpe):
                    continue
            if single_min_trades is not None and trades < int(single_min_trades):
                continue

        if strategy_counts.get(strategy_name, 0) >= int(max_per_strategy):
            continue
        if timeframe_counts.get(timeframe, 0) >= int(max_per_timeframe):
            continue
        if any(symbol_counts.get(symbol, 0) >= int(max_per_symbol) for symbol in symbols):
            continue

        row = dict(row)
        row["selection_score"] = selection_score
        selected.append(row)
        seen_identities.add(identity)
        strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    return selected


def _apply_portfolio_weights(rows: list[dict], *, max_weight: float = 0.35, temperature: float = 0.35) -> list[dict]:
    if not rows:
        return rows

    cap = min(1.0, max(0.05, float(max_weight)))
    temp = max(0.05, float(temperature))
    best = max(_safe_float(row.get("selection_score"), -1e9) for row in rows)
    raw: list[float] = []
    for row in rows:
        score = _safe_float(row.get("selection_score"), -1e9)
        scaled = (score - best) / temp
        base = math.exp(max(-60.0, min(0.0, scaled)))
        mdd = abs(_safe_float(((row.get("oos") or {}) if isinstance(row.get("oos"), dict) else {}).get("mdd"), 0.0))
        risk_penalty = 1.0 / (1.0 + 2.5 * mdd)
        raw.append(base * risk_penalty)

    total = float(sum(raw))
    if total <= 0.0:
        equal = 1.0 / float(len(rows))
        for row in rows:
            row["portfolio_weight"] = equal
        return rows

    weights = [value / total for value in raw]
    weights = [min(cap, value) for value in weights]
    capped_total = float(sum(weights))
    if capped_total <= 0.0:
        equal = 1.0 / float(len(rows))
        for row in rows:
            row["portfolio_weight"] = equal
        return rows

    weights = [value / capped_total for value in weights]
    for row, weight in zip(rows, weights, strict=True):
        row["portfolio_weight"] = float(weight)
    rows.sort(key=lambda item: float(item.get("portfolio_weight", 0.0)), reverse=True)
    return rows


def _build_single_asset_sets(rows: list[dict], *, max_per_asset: int = 2, max_sets: int = 16) -> list[dict]:
    by_symbol: dict[str, list[dict]] = {}
    for row in rows:
        symbols = [str(symbol).strip().upper() for symbol in list(row.get("symbols") or [])]
        if len(symbols) != 1:
            continue
        by_symbol.setdefault(symbols[0], []).append(dict(row))

    if not by_symbol:
        return []

    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda item: _safe_float(item.get("selection_score"), -1e9), reverse=True)
        by_symbol[symbol] = by_symbol[symbol][: max(1, int(max_per_asset))]

    symbols_sorted = sorted(by_symbol)
    base_members = [by_symbol[symbol][0] for symbol in symbols_sorted if by_symbol[symbol]]
    if not base_members:
        return []

    def _weight(items: list[dict]) -> list[dict]:
        scores = [_safe_float(item.get("selection_score"), -1e9) for item in items]
        best = max(scores)
        raw = [math.exp(max(-60.0, min(0.0, score - best))) for score in scores]
        total = float(sum(raw))
        if total <= 0.0:
            eq = 1.0 / float(len(items))
            return [{**item, "portfolio_weight": eq} for item in items]
        return [{**item, "portfolio_weight": float(value / total)} for item, value in zip(items, raw, strict=True)]

    out = [
        {
            "set_id": "single_asset_top_set",
            "member_count": len(base_members),
            "members": _weight(base_members),
        }
    ]

    for symbol in symbols_sorted:
        if len(out) >= max(1, int(max_sets)):
            break
        choices = by_symbol[symbol]
        if len(choices) < 2:
            continue
        members = []
        for current_symbol in symbols_sorted:
            if current_symbol == symbol and len(by_symbol[current_symbol]) >= 2:
                members.append(by_symbol[current_symbol][1])
            else:
                members.append(by_symbol[current_symbol][0])
        out.append(
            {
                "set_id": f"single_asset_variant_{symbol.replace('/', '')}",
                "member_count": len(members),
                "members": _weight(members),
            }
        )
    return out


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
    parser.add_argument("--single-min-score", type=float, default=0.0)
    parser.add_argument("--single-min-return", type=float, default=0.0)
    parser.add_argument("--single-min-sharpe", type=float, default=0.7)
    parser.add_argument("--single-min-trades", type=int, default=20)
    parser.add_argument("--drop-single-without-metrics", action="store_true")
    parser.add_argument("--allow-multi-asset", action="store_true")
    parser.add_argument("--disable-weights", action="store_true")
    parser.add_argument("--weight-temperature", type=float, default=0.35)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--set-max-per-asset", type=int, default=2)
    parser.add_argument("--set-max-sets", type=int, default=16)
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
        single_min_score=float(args.single_min_score),
        single_min_return=float(args.single_min_return),
        single_min_sharpe=float(args.single_min_sharpe),
        single_min_trades=int(args.single_min_trades),
        drop_single_without_metrics=bool(args.drop_single_without_metrics),
        allow_multi_asset=bool(args.allow_multi_asset),
    )
    if not bool(args.disable_weights):
        selected = _apply_portfolio_weights(
            selected,
            max_weight=float(args.max_weight),
            temperature=float(args.weight_temperature),
        )
    portfolio_sets = _build_single_asset_sets(
        selected,
        max_per_asset=max(1, int(args.set_max_per_asset)),
        max_sets=max(1, int(args.set_max_sets)),
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
        "single_min_score": float(args.single_min_score),
        "single_min_return": float(args.single_min_return),
        "single_min_sharpe": float(args.single_min_sharpe),
        "single_min_trades": int(args.single_min_trades),
        "drop_single_without_metrics": bool(args.drop_single_without_metrics),
        "allow_multi_asset": bool(args.allow_multi_asset),
        "weights_enabled": not bool(args.disable_weights),
        "weight_temperature": float(args.weight_temperature),
        "max_weight": float(args.max_weight),
        "portfolio_set_count": len(portfolio_sets),
        "portfolio_sets": portfolio_sets,
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
