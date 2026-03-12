"""Canonical decision artifacts for exact-window multi-slice sweeps."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.eval.exact_window_reporting import (
    DETAILS_LATEST,
    FAIL_ANALYSIS_LATEST,
    MEMORY_EVIDENCE_LATEST,
    SUMMARY_LATEST,
    _rejection_reasons,
)
from lumina_quant.symbols import CANONICAL_STRATEGY_TIMEFRAMES

DECISION_LATEST = "exact_window_decision_latest.json"
DECISION_MD_LATEST = "exact_window_decision_latest.md"
STRICT_VALID_LATEST = "strict_valid_strategy_latest.json"
STRICT_VALID_MD_LATEST = "strict_valid_strategy_latest.md"
STRICT_PASS_DIRNAME = "strict_pass"
STRICT_PASS_JSON_LATEST = "strict_pass_latest.json"
STRICT_PASS_MD_LATEST = "strict_pass_latest.md"
VALIDITY_RULE = "exact-window promoted_count > 0 under current promotion gates"

_TIMEFRAME_ORDER = {tf: idx for idx, tf in enumerate(CANONICAL_STRATEGY_TIMEFRAMES)}


@dataclass(frozen=True)
class _SummaryCandidate:
    summary_path: Path
    details_path: Path | None
    fail_analysis_path: Path | None
    memory_evidence_path: Path | None
    latest_pointer_path: Path | None
    generated_at: datetime
    summary: dict[str, Any]
    details: list[dict[str, Any]]
    fail_analysis: dict[str, Any] | None
    memory_evidence: dict[str, Any] | None
    selected_timeframes: tuple[str, ...]


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _text_dump(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(payload), encoding="utf-8")


def _parse_datetime(value: Any, *, fallback: datetime | None = None) -> datetime:
    token = str(value or "").strip()
    if token:
        normalized = token.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            pass
    return fallback or datetime.now(UTC)


def _timeframe_sort_key(timeframe: str) -> tuple[int, str]:
    return (_TIMEFRAME_ORDER.get(timeframe, len(_TIMEFRAME_ORDER)), timeframe)


def _row_timeframe(row: dict[str, Any]) -> str:
    return str(row.get("strategy_timeframe") or row.get("timeframe") or "unknown")


def _extract_timeframes(summary: dict[str, Any], details: list[dict[str, Any]]) -> list[str]:
    requested = [str(tf) for tf in list((summary.get("execution_profile") or {}).get("requested_timeframes") or []) if str(tf)]
    if requested:
        return sorted(dict.fromkeys(requested), key=_timeframe_sort_key)
    inferred = [_row_timeframe(row) for row in details if _row_timeframe(row) != "unknown"]
    if inferred:
        return sorted(dict.fromkeys(inferred), key=_timeframe_sort_key)
    inferred = [
        str(row.get("strategy_timeframe") or row.get("timeframe") or "")
        for row in list(summary.get("best_per_strategy") or [])
        if str(row.get("strategy_timeframe") or row.get("timeframe") or "")
    ]
    return sorted(dict.fromkeys(inferred), key=_timeframe_sort_key)


def _find_latest_pointer(root: Path, summary_dir: Path) -> Path | None:
    for parent in summary_dir.parents:
        if parent == root:
            break
        candidate = parent / "latest.json"
        if candidate.exists():
            return candidate.resolve()
    return None


_DEFAULT_RANK_WEIGHTS = {
    "sharpe_weight": 2.8,
    "deflated_sharpe_weight": 1.4,
    "pbo_penalty": 2.0,
    "return_weight": 35.0,
    "turnover_penalty": 2.5,
    "turnover_threshold": 2.5,
    "drawdown_penalty": 3.0,
}


def _resolve_rank_weights(scoring_config: dict[str, Any] | None) -> dict[str, float]:
    payload = dict(scoring_config or {})
    if not payload:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "score_config.example.json"
        if config_path.exists():
            try:
                payload = _json_load(config_path)
            except Exception:
                payload = {}
    if "candidate_rank_score_weights" in payload:
        section = payload
    else:
        section = dict(payload.get("candidate_research") or {})
    weights = dict(section.get("candidate_rank_score_weights") or {})
    resolved = dict(_DEFAULT_RANK_WEIGHTS)
    for key, value in weights.items():
        if key in resolved:
            resolved[key] = float(value)
    return resolved


def _validation_score(row: dict[str, Any], *, scoring_config: dict[str, Any] | None = None) -> float:
    weights = _resolve_rank_weights(scoring_config)
    metrics = dict(row.get("val") or {})
    return float(
        (float(weights["sharpe_weight"]) * float(metrics.get("sharpe", 0.0)))
        + (float(weights["deflated_sharpe_weight"]) * float(metrics.get("deflated_sharpe", 0.0)))
        - (float(weights["pbo_penalty"]) * float(metrics.get("pbo", 1.0)))
        + (float(weights["return_weight"]) * float(metrics.get("return", 0.0)))
        - (
            float(weights["turnover_penalty"])
            * max(0.0, float(metrics.get("turnover", 0.0)) - float(weights["turnover_threshold"]))
        )
        - (float(weights["drawdown_penalty"]) * float(metrics.get("mdd", 0.0)))
    )


def _timeframe_selection_score(
    row: dict[str, Any],
    *,
    scoring_config: dict[str, Any] | None = None,
) -> float:
    score = _validation_score(row, scoring_config=scoring_config)
    oos = dict(row.get("oos") or {})
    hard_reject_reasons = dict(row.get("hard_reject_reasons") or {})
    trade_count = float(oos.get("trade_count", 0.0))
    oos_sharpe = float(oos.get("sharpe", 0.0))
    oos_return = float(oos.get("return", 0.0))
    score += min(trade_count, 20.0) * 0.15
    score += max(-2.0, min(3.0, oos_sharpe)) * 0.8
    score += oos_return * 50.0
    if trade_count <= 0.0:
        score -= 8.0
    elif "trade_count" in hard_reject_reasons:
        score -= 1.0
    if "oos_sharpe" in hard_reject_reasons and oos_sharpe <= 0.0:
        score -= 1.0
    if oos_return <= 0.0:
        score -= 0.5
    return float(score)


def _compound_returns(values: list[float]) -> float:
    if not values:
        return 0.0
    equity = 1.0
    for value in values:
        equity *= 1.0 + float(value)
    return float(equity - 1.0)


def _aggregate_stream_by_month(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        epoch_ms = int(point.get("t", 0))
        dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
        buckets[f"{dt.year:04d}-{dt.month:02d}"].append(float(point.get("v", 0.0)))
    return [
        {"period": key, "return": _compound_returns(values)}
        for key, values in sorted(buckets.items())
    ]


def _monthly_hurdle_rows(stream: list[dict[str, Any]], thresholds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    realized = {str(row["period"]): float(row["return"]) for row in _aggregate_stream_by_month(stream)}
    rows: list[dict[str, Any]] = []
    for month in sorted(realized):
        benchmark = dict(thresholds.get(month) or {})
        actual = float(realized.get(month, 0.0))
        threshold = float(benchmark.get("threshold", 0.02))
        btc_ret = float(benchmark.get("btc_buy_hold_return", 0.0))
        strict_pass = bool(actual >= threshold)
        btc_pass = bool(actual >= btc_ret)
        rows.append(
            {
                "month": month,
                "strategy_return": actual,
                "btc_buy_hold_return": btc_ret,
                "threshold": threshold,
                "excess_vs_threshold": actual - threshold,
                "strict_pass": strict_pass,
                "btc_pass": btc_pass,
                "pass": strict_pass,
            }
        )
    return rows


def _latest_three_month_rows(*row_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in row_groups:
        for row in list(group or []):
            month = str(row.get("month") or "").strip()
            if month:
                merged[month] = dict(row)
    return [merged[month] for month in sorted(merged)[-3:]]


def _recent_three_month_two_pct_pass(*row_groups: list[dict[str, Any]]) -> bool:
    latest_rows = _latest_three_month_rows(*row_groups)
    if len(latest_rows) < 3:
        return False
    return all(float(row.get("strategy_return", 0.0)) >= 0.02 for row in latest_rows)


def _summary_candidates(output_dir: str | Path = "var/reports/exact_window_backtests") -> list[_SummaryCandidate]:
    root = Path(output_dir).resolve()
    candidates: list[_SummaryCandidate] = []
    for summary_path in sorted(root.rglob(SUMMARY_LATEST)):
        try:
            summary = _json_load(summary_path)
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        details_path = summary_path.parent / DETAILS_LATEST
        try:
            details_payload = _json_load(details_path) if details_path.exists() else []
        except Exception:
            details_payload = []
        details = list(details_payload) if isinstance(details_payload, list) else []
        timeframes = _extract_timeframes(summary, details)
        if not timeframes:
            continue
        fail_analysis_path = summary_path.parent / FAIL_ANALYSIS_LATEST
        memory_evidence_path = summary_path.parent / MEMORY_EVIDENCE_LATEST
        try:
            fail_analysis = _json_load(fail_analysis_path) if fail_analysis_path.exists() else None
        except Exception:
            fail_analysis = None
        try:
            memory_evidence = _json_load(memory_evidence_path) if memory_evidence_path.exists() else None
        except Exception:
            memory_evidence = None
        generated_at = _parse_datetime(
            summary.get("generated_at"),
            fallback=datetime.fromtimestamp(summary_path.stat().st_mtime, tz=UTC),
        )
        candidates.append(
            _SummaryCandidate(
                summary_path=summary_path.resolve(),
                details_path=details_path.resolve() if details_path.exists() else None,
                fail_analysis_path=fail_analysis_path.resolve() if fail_analysis_path.exists() else None,
                memory_evidence_path=memory_evidence_path.resolve() if memory_evidence_path.exists() else None,
                latest_pointer_path=_find_latest_pointer(root, summary_path.parent),
                generated_at=generated_at,
                summary=summary,
                details=details,
                fail_analysis=fail_analysis if isinstance(fail_analysis, dict) else None,
                memory_evidence=memory_evidence if isinstance(memory_evidence, dict) else None,
                selected_timeframes=tuple(timeframes),
            )
        )
    return candidates


def _select_latest_by_timeframe(candidates: list[_SummaryCandidate]) -> dict[str, _SummaryCandidate]:
    selected: dict[str, _SummaryCandidate] = {}
    for candidate in candidates:
        for timeframe in candidate.selected_timeframes:
            previous = selected.get(timeframe)
            if previous is None:
                selected[timeframe] = candidate
                continue
            if candidate.generated_at > previous.generated_at:
                selected[timeframe] = candidate
                continue
            if candidate.generated_at == previous.generated_at and str(candidate.summary_path) > str(previous.summary_path):
                selected[timeframe] = candidate
    return selected


def _promoted_ids(summary: dict[str, Any], *, timeframe: str | None = None) -> set[str]:
    promoted: set[str] = set()
    for row in list(summary.get("best_per_strategy") or []):
        if timeframe is not None and _row_timeframe(dict(row)) != timeframe:
            continue
        if bool(row.get("promoted")):
            promoted.add(str(row.get("candidate_id") or ""))
    return promoted


def _timeframe_rows(candidate: _SummaryCandidate, timeframe: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in candidate.details if _row_timeframe(dict(row)) == timeframe]
    if rows:
        return rows
    return [dict(row) for row in list(candidate.summary.get("best_per_strategy") or []) if _row_timeframe(dict(row)) == timeframe]


def _timeframe_reason_counts(candidate: _SummaryCandidate, timeframe: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    promoted_ids = _promoted_ids(candidate.summary, timeframe=timeframe)
    for row in _timeframe_rows(candidate, timeframe):
        counts.update(_rejection_reasons(row, promoted_ids=promoted_ids))
    return counts


def _best_row_for_timeframe(
    candidate: _SummaryCandidate,
    timeframe: str,
    *,
    scoring_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    rows = _timeframe_rows(candidate, timeframe)
    if not rows:
        return None
    promoted_ids = _promoted_ids(candidate.summary, timeframe=timeframe)
    ranked = sorted(
        rows,
        key=lambda row: (
            _timeframe_selection_score(row, scoring_config=scoring_config),
            _validation_score(row, scoring_config=scoring_config),
            float((row.get("val") or {}).get("sharpe", 0.0)),
            float((row.get("oos") or {}).get("sharpe", 0.0)),
            float((row.get("oos") or {}).get("return", 0.0)),
            str(row.get("candidate_id") or ""),
        ),
        reverse=True,
    )
    top = dict(ranked[0])
    thresholds = dict(candidate.summary.get("monthly_thresholds") or {})
    val_months = _monthly_hurdle_rows(list((top.get("return_streams") or {}).get("val") or []), thresholds)
    oos_months = _monthly_hurdle_rows(list((top.get("return_streams") or {}).get("oos") or []), thresholds)
    val_hurdle_pass = all(bool(row.get("strict_pass")) for row in val_months if str(row.get("month", "")).startswith("2026-01"))
    val_btc_hurdle_pass = all(bool(row.get("btc_pass")) for row in val_months if str(row.get("month", "")).startswith("2026-01"))
    oos_btc_hurdle_pass = all(bool(row.get("btc_pass")) for row in oos_months)
    recent_three_months = _latest_three_month_rows(val_months, oos_months)
    recent_three_month_two_pct_pass = _recent_three_month_two_pct_pass(val_months, oos_months)
    train_pass = bool((top.get("hurdle_fields") or {}).get("train", {}).get("pass"))
    val_pass = bool((top.get("hurdle_fields") or {}).get("val", {}).get("pass"))
    top["validation_score"] = _validation_score(top, scoring_config=scoring_config)
    top["timeframe_selection_score"] = _timeframe_selection_score(top, scoring_config=scoring_config)
    top["validation_monthly_hurdle"] = val_months
    top["oos_monthly_hurdle"] = oos_months
    top["recent_three_months"] = recent_three_months
    top["validation_hurdle_pass"] = bool(val_hurdle_pass)
    top["validation_btc_hurdle_pass"] = bool(val_btc_hurdle_pass)
    top["oos_btc_hurdle_pass"] = bool(oos_btc_hurdle_pass)
    top["recent_three_month_two_pct_pass"] = bool(recent_three_month_two_pct_pass)
    top["promoted"] = str(top.get("candidate_id") or "") in promoted_ids
    top["btc_beating_candidate"] = train_pass and val_pass and bool(val_btc_hurdle_pass)
    top["three_month_two_pct_candidate"] = train_pass and val_pass and bool(recent_three_month_two_pct_pass)
    top["candidate_pool_eligible"] = bool(
        top.get("promoted")
        or top.get("btc_beating_candidate")
        or top.get("three_month_two_pct_candidate")
    )
    top["rejection_reasons"] = _rejection_reasons(top, promoted_ids=promoted_ids)
    top["source_summary_path"] = str(candidate.summary_path)
    top["source_details_path"] = str(candidate.details_path) if candidate.details_path else None
    return top


def _memory_summary(candidate: _SummaryCandidate) -> dict[str, Any] | None:
    payload = candidate.memory_evidence
    if not isinstance(payload, dict):
        return None
    return {
        "status": str(payload.get("status") or ""),
        "peak_rss_mib": float(payload.get("peak_rss_mib") or 0.0),
        "budget_mib": float(payload.get("budget_mib") or 0.0),
        "soft_limit_mib": float(payload.get("soft_limit_mib") or 0.0),
        "hard_limit_mib": float(payload.get("hard_limit_mib") or 0.0),
        "rss_log_path": str(payload.get("rss_log_path") or ""),
        "memory_evidence_path": str(candidate.memory_evidence_path) if candidate.memory_evidence_path else None,
    }


def resolve_exact_window_decision_paths(
    output_dir: str | Path = "var/reports/exact_window_backtests",
) -> dict[str, Path]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "json_latest": root / DECISION_LATEST,
        "md_latest": root / DECISION_MD_LATEST,
    }


def build_exact_window_decision(
    output_dir: str | Path = "var/reports/exact_window_backtests",
    *,
    scoring_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    selected_by_timeframe = _select_latest_by_timeframe(_summary_candidates(root))
    if not selected_by_timeframe:
        raise FileNotFoundError(f"No exact-window summary artifacts found under {root}")

    timeframes = sorted(selected_by_timeframe, key=_timeframe_sort_key)
    totals: Counter[str] = Counter()
    best_row_reasons: Counter[str] = Counter()
    source_timeframes: dict[Path, set[str]] = defaultdict(set)
    timeframe_rows: list[dict[str, Any]] = []

    for timeframe in timeframes:
        candidate = selected_by_timeframe[timeframe]
        source_timeframes[candidate.summary_path].add(timeframe)
        reason_counts = _timeframe_reason_counts(candidate, timeframe)
        totals.update(reason_counts)
        best_row = _best_row_for_timeframe(candidate, timeframe, scoring_config=scoring_config)
        if best_row is not None:
            best_row_reasons.update(list(best_row.get("rejection_reasons") or []))
        thresholds = dict(candidate.summary.get("monthly_thresholds") or {})
        summary_rows = [
            dict(row)
            for row in list(candidate.summary.get("best_per_strategy") or [])
            if _row_timeframe(dict(row)) == timeframe
        ]
        annotated_summary_rows = []
        for row in summary_rows:
            val_months = _monthly_hurdle_rows(list((row.get("return_streams") or {}).get("val") or []), thresholds)
            oos_months = _monthly_hurdle_rows(list((row.get("return_streams") or {}).get("oos") or []), thresholds)
            train_pass = bool((row.get("hurdle_fields") or {}).get("train", {}).get("pass"))
            val_pass = bool((row.get("hurdle_fields") or {}).get("val", {}).get("pass"))
            btc_pass = all(bool(item.get("btc_pass")) for item in val_months if str(item.get("month", "")).startswith("2026-01"))
            three_month_two_pct_pass = _recent_three_month_two_pct_pass(val_months, oos_months)
            copied = dict(row)
            copied["btc_beating_candidate"] = train_pass and val_pass and bool(btc_pass)
            copied["three_month_two_pct_candidate"] = train_pass and val_pass and bool(three_month_two_pct_pass)
            copied["candidate_pool_eligible"] = bool(
                copied.get("promoted")
                or copied.get("btc_beating_candidate")
                or copied.get("three_month_two_pct_candidate")
            )
            annotated_summary_rows.append(copied)
        promoted_strategy_count = sum(1 for row in annotated_summary_rows if bool(row.get("promoted")))
        btc_beating_strategy_count = sum(1 for row in annotated_summary_rows if bool(row.get("btc_beating_candidate")))
        three_month_two_pct_strategy_count = sum(
            1 for row in annotated_summary_rows if bool(row.get("three_month_two_pct_candidate"))
        )
        provisional_strategy_count = sum(
            1 for row in annotated_summary_rows if bool(row.get("candidate_pool_eligible")) and not bool(row.get("promoted"))
        )
        candidate_pool_strategy_count = sum(1 for row in annotated_summary_rows if bool(row.get("candidate_pool_eligible")))
        timeframe_rows.append(
            {
                "timeframe": timeframe,
                "generated_at": candidate.generated_at.isoformat(),
                "evaluated_count": len(_timeframe_rows(candidate, timeframe)),
                "promoted_strategy_count": int(promoted_strategy_count),
                "btc_beating_strategy_count": int(btc_beating_strategy_count),
                "three_month_two_pct_strategy_count": int(three_month_two_pct_strategy_count),
                "provisional_strategy_count": int(provisional_strategy_count),
                "candidate_pool_strategy_count": int(candidate_pool_strategy_count),
                "reject_reason_counts": [
                    {"rejection_reason": reason, "count": int(count)}
                    for reason, count in reason_counts.most_common()
                ],
                "best_row": best_row,
                "monthly_hurdle_outcomes": {
                    "validation": list((best_row or {}).get("validation_monthly_hurdle") or []),
                    "oos": list((best_row or {}).get("oos_monthly_hurdle") or []),
                    "recent_three_months": list((best_row or {}).get("recent_three_months") or []),
                    "validation_pass": bool((best_row or {}).get("validation_hurdle_pass")),
                    "validation_btc_pass": bool((best_row or {}).get("validation_btc_hurdle_pass")),
                    "recent_three_month_two_pct_pass": bool((best_row or {}).get("recent_three_month_two_pct_pass")),
                    "oos_pass": all(
                        bool(row.get("pass"))
                        for row in list((best_row or {}).get("oos_monthly_hurdle") or [])
                    )
                    if best_row is not None
                    else False,
                    "oos_btc_pass": bool((best_row or {}).get("oos_btc_hurdle_pass")),
                },
                "memory_evidence": _memory_summary(candidate),
                "windows": dict(candidate.summary.get("windows") or {}),
                "summary_path": str(candidate.summary_path),
                "details_path": str(candidate.details_path) if candidate.details_path else None,
                "fail_analysis_path": str(candidate.fail_analysis_path) if candidate.fail_analysis_path else None,
            }
        )

    source_batches: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for timeframe in timeframes:
        candidate = selected_by_timeframe[timeframe]
        if candidate.summary_path in seen_paths:
            continue
        seen_paths.add(candidate.summary_path)
        selected_timeframes = sorted(source_timeframes[candidate.summary_path], key=_timeframe_sort_key)
        source_batches.append(
            {
                "summary_path": str(candidate.summary_path),
                "details_path": str(candidate.details_path) if candidate.details_path else None,
                "fail_analysis_path": str(candidate.fail_analysis_path) if candidate.fail_analysis_path else None,
                "memory_evidence_path": str(candidate.memory_evidence_path) if candidate.memory_evidence_path else None,
                "latest_pointer_path": str(candidate.latest_pointer_path) if candidate.latest_pointer_path else None,
                "generated_at": candidate.generated_at.isoformat(),
                "selected_timeframes": selected_timeframes,
                "actual_max_timestamp": str((candidate.summary.get("windows") or {}).get("actual_max_timestamp") or ""),
                "peak_rss_mib": float((candidate.memory_evidence or {}).get("peak_rss_mib") or 0.0),
            }
        )

    total_evaluated = sum(int(row.get("evaluated_count") or 0) for row in timeframe_rows)
    promoted_total = sum(int(row.get("promoted_strategy_count") or 0) for row in timeframe_rows)
    btc_beating_candidate_total = sum(int(row.get("btc_beating_strategy_count") or 0) for row in timeframe_rows)
    three_month_two_pct_candidate_total = sum(
        int(row.get("three_month_two_pct_strategy_count") or 0) for row in timeframe_rows
    )
    provisional_candidate_total = sum(int(row.get("provisional_strategy_count") or 0) for row in timeframe_rows)
    candidate_pool_total = sum(int(row.get("candidate_pool_strategy_count") or 0) for row in timeframe_rows)
    common_clamp_values = sorted(
        {
            str((row.get("windows") or {}).get("actual_max_timestamp") or "")
            for row in timeframe_rows
            if str((row.get("windows") or {}).get("actual_max_timestamp") or "")
        }
    )
    max_peak_rss_mib = max(
        [float((row.get("memory_evidence") or {}).get("peak_rss_mib") or 0.0) for row in timeframe_rows] or [0.0]
    )

    return {
        "schema_version": "1.0",
        "artifact_kind": "exact_window_decision",
        "generated_at": datetime.now(UTC).isoformat(),
        "validity_rule": VALIDITY_RULE,
        "valid_strategy_found": bool(promoted_total > 0),
        "next_action": (
            "done"
            if promoted_total > 0
            else "review_candidate_pool_candidates"
            if candidate_pool_total > 0
            else "ralplan_team_ralph_required"
        ),
        "timeframes": timeframes,
        "total_evaluated": int(total_evaluated),
        "promoted_total": int(promoted_total),
        "btc_beating_candidate_total": int(btc_beating_candidate_total),
        "three_month_two_pct_candidate_total": int(three_month_two_pct_candidate_total),
        "provisional_candidate_total": int(provisional_candidate_total),
        "candidate_pool_total": int(candidate_pool_total),
        "common_actual_max_timestamp": common_clamp_values[0] if len(common_clamp_values) == 1 else None,
        "actual_max_timestamps": common_clamp_values,
        "reject_counts_all_rows": [
            {"rejection_reason": reason, "count": int(count)}
            for reason, count in totals.most_common()
        ],
        "reject_counts_best_rows": [
            {"rejection_reason": reason, "count": int(count)}
            for reason, count in best_row_reasons.most_common()
        ],
        "max_peak_rss_mib": float(max_peak_rss_mib),
        "source_batches": source_batches,
        "timeframe_rows": timeframe_rows,
    }


def _render_exact_window_decision_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exact-Window Decision Artifact",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Total evaluated: {int(payload.get('total_evaluated', 0))}",
        f"- Promoted total: {int(payload.get('promoted_total', 0))}",
        f"- BTC-beating candidate total: {int(payload.get('btc_beating_candidate_total', 0))}",
        f"- Recent-3M 2% candidate total: {int(payload.get('three_month_two_pct_candidate_total', 0))}",
        f"- Provisional candidate total: {int(payload.get('provisional_candidate_total', 0))}",
        f"- Candidate pool total: {int(payload.get('candidate_pool_total', 0))}",
        f"- Valid strategy found: `{bool(payload.get('valid_strategy_found'))}`",
        f"- Next action: `{payload.get('next_action', '')}`",
        f"- Clamp timestamp: `{payload.get('common_actual_max_timestamp') or ''}`",
        "",
        "## Timeframe Best Rows",
        "",
        "| TF | Strategy | Name | Val Score | Promoted | BTC-beat | 3M>=2% | OOS Return | OOS Sharpe | Rejects | Peak RSS MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in list(payload.get("timeframe_rows") or []):
        best = dict(row.get("best_row") or {})
        memory = dict(row.get("memory_evidence") or {})
        lines.append(
            f"| {row.get('timeframe','')} | {best.get('strategy_class','')} | {best.get('name','')} | "
            f"{float(best.get('validation_score', 0.0)):.3f} | {int(best.get('promoted', False))} | {int(best.get('btc_beating_candidate', False))} | {int(best.get('three_month_two_pct_candidate', False))} | "
            f"{float((best.get('oos') or {}).get('return', 0.0)):.2%} | {float((best.get('oos') or {}).get('sharpe', 0.0)):.3f} | "
            f"{', '.join(best.get('rejection_reasons') or [])} | {float(memory.get('peak_rss_mib', 0.0)):.2f} |"
        )
    lines.extend(["", "## Reject Counts (All Selected Rows)", "", "| Reason | Count |", "|---|---:|"])
    for row in list(payload.get("reject_counts_all_rows") or []):
        lines.append(f"| {row.get('rejection_reason','')} | {int(row.get('count', 0))} |")
    return "\n".join(lines) + "\n"


def build_strict_valid_strategy_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    strategies: list[dict[str, Any]] = []
    for timeframe_row in list(payload.get("timeframe_rows") or []):
        best = dict(timeframe_row.get("best_row") or {})
        if not bool(best.get("promoted")):
            continue
        strategies.append(
            {
                "qualification": "strict_pass",
                "timeframe": timeframe_row.get("timeframe"),
                "candidate_id": best.get("candidate_id"),
                "name": best.get("name"),
                "strategy_class": best.get("strategy_class"),
                "family": best.get("family"),
                "params": dict(best.get("params") or {}),
                "train": dict(best.get("train") or {}),
                "val": dict(best.get("val") or {}),
                "oos": dict(best.get("oos") or {}),
                "validation_monthly_hurdle": list(best.get("validation_monthly_hurdle") or []),
                "oos_monthly_hurdle": list(best.get("oos_monthly_hurdle") or []),
                "source_summary_path": best.get("source_summary_path") or timeframe_row.get("summary_path"),
                "source_details_path": best.get("source_details_path") or timeframe_row.get("details_path"),
            }
        )
    strategies.sort(
        key=lambda item: (
            float((item.get("oos") or {}).get("sharpe", 0.0)),
            float((item.get("oos") or {}).get("return", 0.0)),
            str(item.get("timeframe") or ""),
        ),
        reverse=True,
    )
    return {
        "schema_version": "1.0",
        "artifact_kind": "strict_valid_strategy",
        "generated_at": payload.get("generated_at"),
        "selection_basis": "strict_promoted_only",
        "count": len(strategies),
        "strategies": strategies,
    }


def _render_strict_valid_strategy_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strict Pass Strategy",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Selection basis: `{payload.get('selection_basis', '')}`",
        f"- Strict-pass count: {int(payload.get('count', 0))}",
        "",
    ]
    for item in list(payload.get("strategies") or []):
        oos = dict(item.get("oos") or {})
        lines.extend(
            [
                f"## {item.get('timeframe', '')} — {item.get('name', '')}",
                f"- qualification: `{item.get('qualification', '')}`",
                f"- strategy: `{item.get('strategy_class', '')}`",
                f"- candidate_id: `{item.get('candidate_id', '')}`",
                f"- OOS return: {float(oos.get('return', 0.0)):.2%}",
                f"- OOS Sharpe: {float(oos.get('sharpe', 0.0)):.3f}",
                f"- OOS max drawdown: {float(oos.get('mdd', 0.0)):.2%}",
                f"- trade_count: {int(float(oos.get('trade_count', 0.0)))}",
                f"- source summary: `{item.get('source_summary_path', '')}`",
                "",
                "```json",
                json.dumps(item.get("params") or {}, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    if not list(payload.get("strategies") or []):
        lines.extend(["No strict-pass strategy is currently saved.", ""])
    return "\n".join(lines)


def write_exact_window_decision_bundle(
    output_dir: str | Path = "var/reports/exact_window_backtests",
    *,
    scoring_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = resolve_exact_window_decision_paths(output_dir)
    payload = build_exact_window_decision(paths["root"], scoring_config=scoring_config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = paths["root"] / f"exact_window_decision_{stamp}.json"
    md_path = paths["root"] / f"exact_window_decision_{stamp}.md"
    strict_json_path = paths["root"] / f"strict_valid_strategy_{stamp}.json"
    strict_md_path = paths["root"] / f"strict_valid_strategy_{stamp}.md"
    _json_dump(json_path, payload)
    _json_dump(paths["json_latest"], payload)
    markdown = _render_exact_window_decision_markdown(payload)
    _text_dump(md_path, markdown)
    _text_dump(paths["md_latest"], markdown)
    strict_payload = build_strict_valid_strategy_artifact(payload)
    strict_markdown = _render_strict_valid_strategy_markdown(strict_payload)
    strict_dir = paths["root"] / STRICT_PASS_DIRNAME
    strict_json_latest = paths["root"] / STRICT_VALID_LATEST
    strict_md_latest = paths["root"] / STRICT_VALID_MD_LATEST
    strict_pass_json_path = strict_dir / f"strict_pass_{stamp}.json"
    strict_pass_md_path = strict_dir / f"strict_pass_{stamp}.md"
    strict_pass_json_latest = strict_dir / STRICT_PASS_JSON_LATEST
    strict_pass_md_latest = strict_dir / STRICT_PASS_MD_LATEST
    _json_dump(strict_json_path, strict_payload)
    _json_dump(strict_json_latest, strict_payload)
    _text_dump(strict_md_path, strict_markdown)
    _text_dump(strict_md_latest, strict_markdown)
    _json_dump(strict_pass_json_path, strict_payload)
    _json_dump(strict_pass_json_latest, strict_payload)
    _text_dump(strict_pass_md_path, strict_markdown)
    _text_dump(strict_pass_md_latest, strict_markdown)
    return {
        "payload": payload,
        "json_path": json_path,
        "json_latest": paths["json_latest"],
        "md_path": md_path,
        "md_latest": paths["md_latest"],
        "strict_payload": strict_payload,
        "strict_json_path": strict_json_path,
        "strict_json_latest": strict_json_latest,
        "strict_md_path": strict_md_path,
        "strict_md_latest": strict_md_latest,
        "strict_pass_json_path": strict_pass_json_path,
        "strict_pass_json_latest": strict_pass_json_latest,
        "strict_pass_md_path": strict_pass_md_path,
        "strict_pass_md_latest": strict_pass_md_latest,
    }


def load_exact_window_decision_artifact(
    output_dir: str | Path = "var/reports/exact_window_backtests",
) -> dict[str, Any] | None:
    path = resolve_exact_window_decision_paths(output_dir)["json_latest"]
    if not path.exists():
        return None
    payload = _json_load(path)
    return payload if isinstance(payload, dict) else None


__all__ = [
    "DECISION_LATEST",
    "DECISION_MD_LATEST",
    "STRICT_PASS_DIRNAME",
    "STRICT_PASS_JSON_LATEST",
    "STRICT_PASS_MD_LATEST",
    "STRICT_VALID_LATEST",
    "STRICT_VALID_MD_LATEST",
    "VALIDITY_RULE",
    "build_exact_window_decision",
    "build_strict_valid_strategy_artifact",
    "load_exact_window_decision_artifact",
    "resolve_exact_window_decision_paths",
    "write_exact_window_decision_bundle",
]
