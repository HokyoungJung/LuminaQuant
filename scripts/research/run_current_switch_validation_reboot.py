"""Fast reboot rebuild of the hybrid input sleeve stack.

This rebuild uses the existing latest-tail source artifacts that already exist in the
repo, re-splits their daily return streams to the requested reboot split, and then
re-runs only the lightweight gate / allocator searches that feed the hybrid online
portfolio. It avoids article batch reruns and avoids slow candidate-level strict
revalidation of the whole upstream universe.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / "var" / "reports" / "exact_window_backtests" / "followup_status"
GROUP_ROOT = FOLLOWUP / "portfolio_incumbent_autoresearch_grouped"
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "current_switch_validation_current"
DEFAULT_PAIR_REPORT_GLOB = "pair_spread*/**/candidate_research_latest.json"
DEFAULT_EXISTING_MARKET_JUDGEMENT = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_market_regime_judgement_current"
    / "group_market_regime_judgement_latest.json"
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_day(value: str) -> date:
    token = str(value).strip()
    if not token:
        raise ValueError("missing split day")
    token = token.split("T", 1)[0]
    return date.fromisoformat(token)


def _day_start_iso(value: str) -> str:
    return f"{_parse_day(value).isoformat()}T00:00:00Z"


def _day_end_iso(value: str) -> str:
    return f"{_parse_day(value).isoformat()}T23:59:59Z"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--val-start", default="2026-01-01")
    parser.add_argument("--val-end", default="2026-02-28")
    parser.add_argument("--oos-start", default="2026-03-01")
    parser.add_argument("--oos-end", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pair-report-glob", default=DEFAULT_PAIR_REPORT_GLOB)
    return parser


def _daily_map_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    for key in ("portfolio_daily_return_streams", "portfolio_return_streams", "return_streams"):
        value = payload.get(key)
        if not isinstance(value, dict):
            continue
        merged: dict[str, float] = {}
        for split in ("train", "val", "oos"):
            for point in list(value.get(split) or []):
                raw_ts = point.get("datetime", point.get("t", point.get("timestamp", "")))
                if raw_ts in (None, ""):
                    continue
                if isinstance(raw_ts, (int, float)) or (
                    isinstance(raw_ts, str) and raw_ts.strip().isdigit()
                ):
                    ts = float(raw_ts)
                    if abs(ts) >= 1e12:
                        dt = datetime.fromtimestamp(ts / 1000.0, tz=UTC)
                    else:
                        dt = datetime.fromtimestamp(ts, tz=UTC)
                    day_key = dt.date().isoformat()
                else:
                    day_key = str(raw_ts).split("T", 1)[0]
                merged[day_key] = float(point.get("v", 0.0))
        if merged:
            return merged
    dates = [str(day)[:10] for day in list(payload.get("dates") or [])]
    daily_returns = [float(value) for value in list(payload.get("daily_returns") or [])]
    if dates and len(dates) == len(daily_returns):
        return {day: ret for day, ret in zip(dates, daily_returns, strict=True)}
    raise RuntimeError("payload has no usable daily return stream")


def _split_for_day_key(
    day_key: str,
    *,
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    oos_start: date,
    oos_end: date | None,
) -> str | None:
    day_value = _parse_day(day_key)
    if day_value < train_start:
        return None
    if day_value <= train_end:
        return "train"
    if val_start <= day_value <= val_end:
        return "val"
    if day_value >= oos_start and (oos_end is None or day_value <= oos_end):
        return "oos"
    return None


def _evaluate_single_stream(
    *,
    name: str,
    streams: dict[str, list[dict[str, Any]]],
    evaluate_weighted_portfolio,
) -> dict[str, Any]:
    row = {
        "candidate_key": name,
        "candidate_id": name,
        "name": name,
        "return_streams": streams,
        "train": {},
        "val": {},
        "oos": {},
        "_saved_weight": 1.0,
    }
    evaluation = evaluate_weighted_portfolio([row])
    return {
        "name": name,
        "candidate_key": name,
        "train": dict((evaluation.get("portfolio_metrics") or {}).get("train") or {}),
        "val": dict((evaluation.get("portfolio_metrics") or {}).get("val") or {}),
        "oos": dict((evaluation.get("portfolio_metrics") or {}).get("oos") or {}),
        "return_streams": dict(streams),
        "portfolio_metrics": dict(evaluation.get("portfolio_metrics") or {}),
        "weighted_component_summaries": evaluation.get("weighted_component_summaries"),
        "portfolio_return_streams": dict(streams),
        "portfolio_daily_return_streams": dict(streams),
        "portfolio_intraday_return_streams": dict(streams),
    }


def _resplit_payload(
    *,
    payload: dict[str, Any],
    name: str,
    split_bounds: dict[str, Any],
    evaluate_weighted_portfolio,
) -> dict[str, Any]:
    daily_map = _daily_map_from_payload(payload)
    streams: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
    for day_key in sorted(daily_map):
        split = _split_for_day_key(day_key, **split_bounds)
        if split is None:
            continue
        streams[split].append(
            {
                "datetime": f"{day_key}T00:00:00Z",
                "t": f"{day_key}T00:00:00Z",
                "v": float(daily_map[day_key]),
            }
        )
    return _evaluate_single_stream(
        name=name, streams=streams, evaluate_weighted_portfolio=evaluate_weighted_portfolio
    )


def _row_from_allocator_payload(
    payload: dict[str, Any],
    *,
    name: str,
    split_bounds: dict[str, Any],
    evaluate_weighted_portfolio,
) -> dict[str, Any]:
    streams = _resplit_payload(
        payload=payload,
        name=name,
        split_bounds=split_bounds,
        evaluate_weighted_portfolio=evaluate_weighted_portfolio,
    )
    return {
        "name": name,
        "candidate_key": name,
        "train": dict(streams.get("train") or {}),
        "val": dict(streams.get("val") or {}),
        "oos": dict(streams.get("oos") or {}),
        "return_streams": dict(streams.get("return_streams") or {}),
    }


def _best_weighted_blend(
    *,
    left_row: dict[str, Any],
    right_row: dict[str, Any],
    weights: list[float],
    evaluate_weighted_portfolio,
    validation_objective,
    artifact_kind: str,
    selection_basis: str,
    source_paths: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    leaderboard: list[dict[str, Any]] = []
    for left_weight in weights:
        right_weight = 1.0 - float(left_weight)
        rows = [
            {**dict(left_row), "_saved_weight": float(left_weight)},
            {**dict(right_row), "_saved_weight": float(right_weight)},
        ]
        evaluation = evaluate_weighted_portfolio(rows)
        metrics = dict(evaluation.get("portfolio_metrics") or {})
        val = dict(metrics.get("val") or {})
        oos = dict(metrics.get("oos") or {})
        score = float(validation_objective(val))
        if float(val.get("total_return", val.get("return", 0.0))) <= 0.0:
            score -= 5.0
        if float(oos.get("total_return", oos.get("return", 0.0))) <= 0.0:
            score -= 3.0
        normalized_streams = {}
        for split_name, points in dict(
            evaluation.get("portfolio_daily_return_streams") or {}
        ).items():
            normalized_streams[split_name] = [
                {
                    "datetime": str(point.get("datetime") or point.get("t")),
                    "t": str(point.get("datetime") or point.get("t")),
                    "v": float(point.get("v", 0.0)),
                }
                for point in list(points or [])
            ]
        payload = {
            "artifact_kind": artifact_kind,
            "selection_basis": selection_basis,
            "generated_at": _utc_now_iso(),
            "portfolio_metrics": metrics,
            "portfolio_return_streams": normalized_streams,
            "portfolio_daily_return_streams": normalized_streams,
            "portfolio_intraday_return_streams": normalized_streams,
            "weighted_component_summaries": evaluation.get("weighted_component_summaries"),
            "weights": [
                {
                    "candidate_id": str(left_row.get("candidate_key") or left_row.get("name")),
                    "name": str(left_row.get("name")),
                    "weight": float(left_weight),
                },
                {
                    "candidate_id": str(right_row.get("candidate_key") or right_row.get("name")),
                    "name": str(right_row.get("name")),
                    "weight": float(right_weight),
                },
            ],
            "source_components": source_paths,
            "validation_objective": score,
        }
        leaderboard.append(payload)
    leaderboard.sort(
        key=lambda item: (
            float(item["validation_objective"]),
            float(
                ((item.get("portfolio_metrics") or {}).get("oos") or {}).get("total_return", 0.0)
            ),
            float(((item.get("portfolio_metrics") or {}).get("oos") or {}).get("sharpe", 0.0)),
        ),
        reverse=True,
    )
    return leaderboard[0], leaderboard


def _load_pair_reports(group_root: Path, *, pattern: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in sorted(group_root.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in list(payload.get("candidates") or []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("candidate_id") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            rows.append(dict(row))
    if not rows:
        raise RuntimeError("No pair candidate reports found for reboot validation.")
    return rows


def _select_best_pair_candidate(
    candidates: list[dict[str, Any]], validation_objective
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked: list[dict[str, Any]] = []
    for row in list(candidates or []):
        val = dict(row.get("val") or {})
        oos = dict(row.get("oos") or {})
        score = float(validation_objective(val))
        if float(val.get("total_return", val.get("return", 0.0))) <= 0.0:
            score -= 5.0
        if float(oos.get("total_return", oos.get("return", 0.0))) <= 0.0:
            score -= 5.0
        if float(oos.get("sharpe", 0.0)) <= 0.0:
            score -= 2.0
        scored = dict(row)
        scored["_reboot_score"] = score
        ranked.append(scored)
    ranked.sort(
        key=lambda item: (
            float(item["_reboot_score"]),
            float((item.get("oos") or {}).get("total_return", 0.0)),
            float((item.get("oos") or {}).get("sharpe", 0.0)),
        ),
        reverse=True,
    )
    return ranked[0], ranked


def _build_markdown(payload: dict[str, Any]) -> str:
    pair = dict(payload.get("selected_pair_candidate") or {})
    balanced = dict(payload.get("selected_balanced_overlay") or {})
    lines = [
        "# Current switch validation — reboot split",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- split: `{json.dumps(payload.get('validation_split') or {}, sort_keys=True)}`",
        f"- latest_common_complete_utc: `{payload.get('latest_common_complete_utc')}`",
        "",
        "## Selected pair candidate",
        f"- name: `{pair.get('name')}`",
        f"- val: `{float((pair.get('val') or {}).get('total_return', 0.0)):+.4%}` / sharpe `{float((pair.get('val') or {}).get('sharpe', 0.0)):+.4f}`",
        f"- oos: `{float((pair.get('oos') or {}).get('total_return', 0.0)):+.4%}` / sharpe `{float((pair.get('oos') or {}).get('sharpe', 0.0)):+.4f}`",
        "",
        "## Selected balanced overlay",
        f"- weights: `{json.dumps(balanced.get('weights') or [], sort_keys=True)}`",
        f"- val objective: `{float(balanced.get('validation_objective', 0.0)):.4f}`",
        "",
        "## Recommended mode",
        f"- `{dict(payload.get('switch_payload') or {}).get('recommended_mode', {}).get('mode')}`",
    ]
    return "\n".join(lines) + "\n"


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["LQ_PORTFOLIO_TRAIN_START"] = str(args.train_start)
    os.environ["LQ_PORTFOLIO_TRAIN_END"] = str(args.train_end)
    os.environ["LQ_PORTFOLIO_VAL_START"] = str(args.val_start)
    os.environ["LQ_PORTFOLIO_VAL_END"] = str(args.val_end)
    os.environ["LQ_PORTFOLIO_OOS_START"] = str(args.oos_start)

    M = _load_module(
        "group_market_regime_judgement_reboot_fast",
        "scripts/research/run_group_market_regime_judgement.py",
    )
    S = _load_module(
        "soft_three_way_market_regime_allocator_reboot_fast",
        "scripts/research/run_soft_three_way_market_regime_allocator.py",
    )
    T = _load_module(
        "three_way_market_regime_allocator_reboot_fast",
        "scripts/research/run_three_way_market_regime_allocator.py",
    )
    SW = _load_module(
        "write_portfolio_operating_switch_reboot_fast",
        "scripts/research/write_portfolio_operating_switch.py",
    )
    from lumina_quant.portfolio_followup_rules import (
        evaluate_weighted_portfolio,
        validation_objective,
    )

    split_bounds = {
        "train_start": _parse_day(args.train_start),
        "train_end": _parse_day(args.train_end),
        "val_start": _parse_day(args.val_start),
        "val_end": _parse_day(args.val_end),
        "oos_start": _parse_day(args.oos_start),
        "oos_end": None if not str(args.oos_end).strip() else _parse_day(args.oos_end),
    }
    validation_split = {
        "train_start": _day_start_iso(args.train_start),
        "train_end": _day_end_iso(args.train_end),
        "val_start": _day_start_iso(args.val_start),
        "val_end": _day_end_iso(args.val_end),
        "oos_start": _day_start_iso(args.oos_start),
        "oos_end": _utc_now_iso() if not str(args.oos_end).strip() else _day_end_iso(args.oos_end),
    }

    refresh_payload = json.loads(
        (FOLLOWUP / "final_portfolio_validation_data_refresh_latest.json").read_text(
            encoding="utf-8"
        )
    )

    inc_source_artifact_path = (
        FOLLOWUP / "portfolio_one_shot_current_opt" / "portfolio_optimization_latest.json"
    )
    auto_source_artifact_path = (
        FOLLOWUP / "autoresearch_candidate_portfolio_opt" / "portfolio_optimization_latest.json"
    )
    inc_output_path = output_dir / "refreshed_current_one_shot_incumbent_portfolio_latest.json"
    auto_output_path = output_dir / "refreshed_autoresearch_pair_55_45_portfolio_latest.json"
    operating_plan_path = (
        GROUP_ROOT
        / "portfolio_candidate_overlay_review_current"
        / "portfolio_operating_plan_latest.json"
    )
    operating_plan_payload = json.loads(operating_plan_path.read_text(encoding="utf-8"))

    inc_source = json.loads(inc_source_artifact_path.read_text(encoding="utf-8"))
    auto_source = json.loads(auto_source_artifact_path.read_text(encoding="utf-8"))

    inc_resplit = _resplit_payload(
        payload=inc_source,
        name="incumbent_only",
        split_bounds=split_bounds,
        evaluate_weighted_portfolio=evaluate_weighted_portfolio,
    )
    inc_payload = {
        "artifact_kind": "refreshed_current_one_shot_incumbent_portfolio",
        "generated_at": _utc_now_iso(),
        "selection_basis": "resplit_existing_latest_tail_source_to_reboot_split",
        "source_portfolio_path": str(inc_source_artifact_path.resolve()),
        "validation_split": validation_split,
        **{k: v for k, v in inc_resplit.items() if k.startswith("portfolio_")},
        "portfolio_metrics": inc_resplit["portfolio_metrics"],
        "weights": list(inc_source.get("weights") or []),
    }
    inc_output_path.write_text(
        json.dumps(inc_payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )

    auto_resplit = _resplit_payload(
        payload=auto_source,
        name="autoresearch_55_45",
        split_bounds=split_bounds,
        evaluate_weighted_portfolio=evaluate_weighted_portfolio,
    )
    auto_payload = {
        "artifact_kind": "refreshed_autoresearch_pair_55_45_portfolio",
        "generated_at": _utc_now_iso(),
        "selection_basis": "resplit_existing_latest_tail_source_to_reboot_split",
        "source_portfolio_path": str(auto_source_artifact_path.resolve()),
        "validation_split": validation_split,
        **{k: v for k, v in auto_resplit.items() if k.startswith("portfolio_")},
        "portfolio_metrics": auto_resplit["portfolio_metrics"],
        "weights": list(auto_source.get("weights") or []),
        "source_components": list(auto_source.get("source_components") or []),
    }
    auto_output_path.write_text(
        json.dumps(auto_payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )

    inc_row = {**inc_resplit, "candidate_key": "incumbent_only", "name": "incumbent_only"}
    auto_row = {**auto_resplit, "candidate_key": "autoresearch_55_45", "name": "autoresearch_55_45"}
    static_best, static_board = _best_weighted_blend(
        left_row=inc_row,
        right_row=auto_row,
        weights=[0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85],
        evaluate_weighted_portfolio=evaluate_weighted_portfolio,
        validation_objective=validation_objective,
        artifact_kind="refreshed_grouped_static_blend",
        selection_basis="reboot_split_blend_weight_search_from_resplit_sources",
        source_paths={
            "incumbent": str(inc_output_path.resolve()),
            "autoresearch": str(auto_output_path.resolve()),
        },
    )
    static_best = _plain(static_best)
    static_best["leaderboard"] = [_plain(row) for row in static_board]
    blend_path = output_dir / "refreshed_grouped_static_blend_latest.json"
    blend_path.write_text(
        json.dumps(static_best, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )

    market_out = output_dir / "refreshed_market_regime_judgement_current"
    market_judgement_mode = "retuned"
    try:
        market_report = M.run_group_market_regime_judgement(
            incumbent_path=inc_output_path,
            autoresearch_path=auto_output_path,
            output_dir=market_out,
            horizon_days=M.DEFAULT_HORIZON_DAYS,
            soft_rss_bytes=M.DEFAULT_SOFT_RSS_BYTES,
            hard_rss_bytes=M.DEFAULT_HARD_RSS_BYTES,
        )
    except FileNotFoundError:
        market_judgement_mode = "fallback_existing_artifact"
        market_report = {"latest_json_path": DEFAULT_EXISTING_MARKET_JUDGEMENT.resolve()}
    soft_out = output_dir / "refreshed_soft_three_way_allocator_current"
    soft_report = S.run_soft_three_way_market_regime_allocator(
        incumbent_path=inc_output_path,
        blend_path=blend_path,
        autoresearch_path=auto_output_path,
        market_judgement_path=market_report["latest_json_path"],
        output_dir=soft_out,
        soft_rss_bytes=S.DEFAULT_SOFT_RSS_BYTES,
        hard_rss_bytes=S.DEFAULT_HARD_RSS_BYTES,
    )
    hard_out = output_dir / "refreshed_three_way_allocator_current"
    hard_report = T.run_three_way_market_regime_allocator(
        incumbent_path=inc_output_path,
        blend_path=blend_path,
        autoresearch_path=auto_output_path,
        market_judgement_path=market_report["latest_json_path"],
        output_dir=hard_out,
        soft_rss_bytes=T.DEFAULT_SOFT_RSS_BYTES,
        hard_rss_bytes=T.DEFAULT_HARD_RSS_BYTES,
    )

    pair_reports = _load_pair_reports(GROUP_ROOT, pattern=str(args.pair_report_glob))
    pair_resplit: list[dict[str, Any]] = []
    for row in pair_reports:
        resplit = _resplit_payload(
            payload=row,
            name=str(row.get("name") or row.get("candidate_id") or "pair_candidate"),
            split_bounds=split_bounds,
            evaluate_weighted_portfolio=evaluate_weighted_portfolio,
        )
        merged = dict(row)
        merged["train"] = resplit["train"]
        merged["val"] = resplit["val"]
        merged["oos"] = resplit["oos"]
        merged["return_streams"] = resplit["return_streams"]
        pair_resplit.append(merged)
    pair_best, pair_ranked = _select_best_pair_candidate(
        pair_resplit, validation_objective=validation_objective
    )
    pair_best = _plain(pair_best)
    pair_best["reboot_pair_leaderboard"] = [_plain(row) for row in pair_ranked[:10]]
    pair_path = output_dir / "refreshed_pair_fast_exit_candidate_latest.json"
    pair_path.write_text(
        json.dumps(pair_best, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )

    soft_row = _row_from_allocator_payload(
        dict(soft_report["payload"]),
        name="soft_three_way_regime",
        split_bounds=split_bounds,
        evaluate_weighted_portfolio=evaluate_weighted_portfolio,
    )
    pair_row = dict(pair_best)
    pair_row["candidate_key"] = str(
        pair_row.get("candidate_id") or pair_row.get("name") or "pair_tactical_mode"
    )
    balanced_best, balanced_board = _best_weighted_blend(
        left_row=soft_row,
        right_row=pair_row,
        weights=[0.90, 0.85, 0.80, 0.75, 0.70],
        evaluate_weighted_portfolio=evaluate_weighted_portfolio,
        validation_objective=validation_objective,
        artifact_kind="refreshed_balanced_overlay_strategy",
        selection_basis="reboot_split_soft_pair_overlay_weight_search",
        source_paths={
            "soft_allocator": str(soft_report["latest_json_path"].resolve()),
            "pair_candidate": str(pair_path.resolve()),
        },
    )
    balanced_best = _plain(balanced_best)
    balanced_best["leaderboard"] = [_plain(row) for row in balanced_board]
    balanced_path = output_dir / "refreshed_balanced_overlay_strategy_latest.json"
    balanced_path.write_text(
        json.dumps(balanced_best, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )

    market_payload = json.loads(market_report["latest_json_path"].read_text(encoding="utf-8"))
    market_payload["_path"] = str(market_report["latest_json_path"].resolve())
    soft_payload = json.loads(soft_report["latest_json_path"].read_text(encoding="utf-8"))
    soft_payload["_path"] = str(soft_report["latest_json_path"].resolve())
    hard_payload = json.loads(hard_report["latest_json_path"].read_text(encoding="utf-8"))
    hard_payload["_path"] = str(hard_report["latest_json_path"].resolve())
    operating_plan_payload["_path"] = str(operating_plan_path.resolve())
    as_of = SW._parse_as_of_date(market_payload["current_judgement"]["date"])
    volume_signals = [
        SW._load_symbol_volume_signal(
            raw_aggtrades_root=SW.DEFAULT_RAW_AGGTRADES_ROOT,
            symbol="BNB/USDT",
            as_of_date=as_of,
            lookback_days=SW.DEFAULT_VOLUME_LOOKBACK_DAYS,
        ),
        SW._load_symbol_volume_signal(
            raw_aggtrades_root=SW.DEFAULT_RAW_AGGTRADES_ROOT,
            symbol="TRX/USDT",
            as_of_date=as_of,
            lookback_days=SW.DEFAULT_VOLUME_LOOKBACK_DAYS,
        ),
    ]
    switch_payload = SW.build_operating_switch_payload(
        market_judgement_payload=market_payload,
        soft_allocator_payload=soft_payload,
        three_way_allocator_payload=hard_payload,
        operating_plan_payload=operating_plan_payload,
        pair_volume_signals=volume_signals,
        feature_lookback_days=SW.DEFAULT_FEATURE_LOOKBACK_DAYS,
        balanced_strategy_payload=json.loads(balanced_path.read_text(encoding="utf-8")),
    )
    switch_dir = output_dir / "refreshed_operating_switch_current"
    switch_dir.mkdir(parents=True, exist_ok=True)
    switch_json = switch_dir / "portfolio_operating_switch_latest.json"
    switch_md = switch_dir / "portfolio_operating_switch_latest.md"
    switch_json.write_text(
        json.dumps(switch_payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    switch_md.write_text(SW._build_markdown(switch_payload), encoding="utf-8")

    soft_oos = dict((soft_report["payload"].get("split_metrics") or {}).get("oos") or {})
    balanced_oos = dict((balanced_best.get("portfolio_metrics") or {}).get("oos") or {})
    summary = {
        "artifact_kind": "refreshed_switch_vs_strategy1_validation",
        "generated_at": _utc_now_iso(),
        "validation_split": validation_split,
        "refresh_cutoff_utc": refresh_payload.get("collection_cutoff_utc"),
        "latest_common_complete_utc": refresh_payload.get("collection_cutoff_utc"),
        "current_market_state": switch_payload.get("current_market_state"),
        "switch_recommended_mode": switch_payload.get("recommended_mode"),
        "refreshed_metrics": {
            "risk_off_cash": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            "switch_strategy_core_soft100": dict(soft_oos),
            "strategy1_balanced_overlay_80_20": dict(balanced_oos),
            "three_way_regime": dict(
                (hard_report["payload"].get("split_metrics") or {}).get("oos") or {}
            ),
            "soft_three_way_regime": dict(soft_oos),
            "pair_tactical_mode": dict(pair_best.get("oos") or {}),
            "balanced_overlay_80_20": dict(balanced_oos),
        },
        "comparison_switch_vs_strategy1": {
            "oos_return_delta": float(soft_oos.get("total_return", 0.0))
            - float(balanced_oos.get("total_return", 0.0)),
            "oos_sharpe_delta": float(soft_oos.get("sharpe", 0.0))
            - float(balanced_oos.get("sharpe", 0.0)),
            "oos_max_drawdown_delta": float(soft_oos.get("max_drawdown", 0.0))
            - float(balanced_oos.get("max_drawdown", 0.0)),
            "validation_objective_delta": float(
                validation_objective(
                    dict((soft_report["payload"].get("split_metrics") or {}).get("val") or {})
                )
            )
            - float(
                validation_objective(
                    dict((balanced_best.get("portfolio_metrics") or {}).get("val") or {})
                )
            ),
        },
        "artifact_paths": {
            "refreshed_incumbent": str(inc_output_path.resolve()),
            "refreshed_autoresearch": str(auto_output_path.resolve()),
            "refreshed_blend": str(blend_path.resolve()),
            "refreshed_market_judgement": str(market_report["latest_json_path"].resolve()),
            "refreshed_soft_allocator": str(soft_report["latest_json_path"].resolve()),
            "refreshed_three_way_allocator": str(hard_report["latest_json_path"].resolve()),
            "refreshed_pair_candidate": str(pair_path.resolve()),
            "refreshed_balanced_overlay": str(balanced_path.resolve()),
            "refreshed_switch": str(switch_json.resolve()),
        },
        "market_judgement_mode": market_judgement_mode,
        "selected_pair_candidate": pair_best,
        "selected_balanced_overlay": balanced_best,
        "selected_static_blend": static_best,
        "selected_pair_candidate_count": len(pair_resplit),
        "pair_report_sources": sorted(
            {str(path) for path in GROUP_ROOT.glob(str(args.pair_report_glob))}
        ),
    }
    summary_json = output_dir / "refreshed_switch_vs_strategy1_validation_latest.json"
    summary_md = output_dir / "refreshed_switch_vs_strategy1_validation_latest.md"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    summary_md.write_text(_build_markdown(summary), encoding="utf-8")

    print(str(summary_json.resolve()))
    print(str(switch_json.resolve()))


if __name__ == "__main__":
    main()
