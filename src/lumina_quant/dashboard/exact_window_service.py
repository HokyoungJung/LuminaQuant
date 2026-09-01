"""Exact-window payload helpers for the Next dashboard migration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from lumina_quant.dashboard.exact_window_bundle import load_exact_window_bundle


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except TypeError, ValueError:
        return None


def _safe_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(value)
    except TypeError, ValueError:
        return 0


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if str(item or "").strip()]


def _normalize_reject_reasons(row: dict[str, Any]) -> list[str]:
    counts = row.get("reject_reason_counts")
    if isinstance(counts, list):
        return [
            str(item.get("rejection_reason") or "")
            for item in counts
            if isinstance(item, dict) and str(item.get("rejection_reason") or "").strip()
        ]
    hard_rejects = row.get("hard_reject_reasons")
    if isinstance(hard_rejects, dict):
        return [str(key) for key in hard_rejects if str(key or "").strip()]
    return []


def _normalize_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    oos = row.get("oos") if isinstance(row.get("oos"), dict) else {}
    return {
        "timeframe": str(row.get("timeframe") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "name": str(row.get("name") or ""),
        "family": str(row.get("family") or ""),
        "promoted": bool(row.get("promoted", False)),
        "oos_return": _safe_float(oos.get("return")),
        "oos_sharpe": _safe_float(oos.get("sharpe")),
        "oos_max_drawdown": _safe_float(oos.get("max_drawdown")),
        "trade_count": _safe_float(oos.get("trade_count") or oos.get("trades")),
        "reject_reasons": _normalize_reject_reasons(row),
    }


def _normalize_portfolio_weight(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(row.get("name") or ""),
        "timeframe": str(row.get("timeframe") or ""),
        "family": str(row.get("family") or ""),
        "weight": _safe_float(row.get("weight")),
        "oos_return": _safe_float(row.get("oos_return")),
        "oos_sharpe": _safe_float(row.get("oos_sharpe")),
    }


def empty_exact_window_payload(
    *,
    reason: str,
    root: str = "",
    run_root: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "generated_at": None,
        "status": reason,
        "error": error,
        "root": root,
        "run_root": run_root,
        "summary": {
            "candidate_count": 0,
            "evaluated_count": 0,
            "promoted_count": 0,
            "btc_beating_candidate_count": 0,
            "provisional_candidate_count": 0,
            "candidate_pool_count": 0,
            "requested_timeframes": [],
            "requested_symbols": [],
            "low_ram_profile": False,
        },
        "decision": {
            "next_action": "",
            "promoted_total": 0,
            "total_evaluated": 0,
            "max_peak_rss_mib": None,
            "valid_strategy_found": False,
        },
        "memory": {
            "status": "",
            "peak_rss_mib": None,
            "soft_limit_mib": None,
            "hard_limit_mib": None,
        },
        "portfolio": {
            "construction_basis": "",
            "oos_return": None,
            "oos_sharpe": None,
            "oos_max_drawdown": None,
        },
        "time_window": {},
        "timeframes": [],
        "top_candidates": [],
        "portfolio_weights": [],
        "notes": [],
        "warnings": [],
    }


def load_exact_window_summary_payload(*, root: str | None = None) -> dict[str, Any]:
    try:
        bundle = load_exact_window_bundle(root)
    except FileNotFoundError as exc:
        return empty_exact_window_payload(
            reason="missing_bundle", root=str(root or ""), error=str(exc)
        )
    except Exception as exc:  # pragma: no cover - defensive runtime guard for artifact drift
        return empty_exact_window_payload(
            reason="load_failed", root=str(root or ""), error=str(exc)
        )

    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else None
    if summary is None:
        return empty_exact_window_payload(
            reason="missing_summary",
            root=str(bundle.get("root") or root or ""),
            run_root=str(bundle.get("run_root") or ""),
        )

    decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
    memory = (
        bundle.get("memory_evidence") if isinstance(bundle.get("memory_evidence"), dict) else {}
    )
    execution_profile = (
        summary.get("execution_profile")
        if isinstance(summary.get("execution_profile"), dict)
        else {}
    )
    portfolio = summary.get("portfolio") if isinstance(summary.get("portfolio"), dict) else {}
    portfolio_oos = portfolio.get("oos") if isinstance(portfolio.get("oos"), dict) else {}
    notes = summary.get("notes") if isinstance(summary.get("notes"), dict) else {}
    timeframe_rows = (
        decision.get("timeframe_rows") if isinstance(decision.get("timeframe_rows"), list) else []
    )
    top_candidates = (
        summary.get("best_per_strategy")
        if isinstance(summary.get("best_per_strategy"), list)
        else []
    )
    weights = portfolio.get("weights") if isinstance(portfolio.get("weights"), list) else []

    return {
        "as_of": datetime.now(UTC).isoformat(),
        "generated_at": summary.get("generated_at")
        or decision.get("generated_at")
        or memory.get("generated_at"),
        "status": "ok",
        "error": None,
        "root": str(bundle.get("root") or root or ""),
        "run_root": str(bundle.get("run_root") or ""),
        "summary": {
            "candidate_count": _safe_int(summary.get("candidate_count")),
            "evaluated_count": _safe_int(summary.get("evaluated_count")),
            "promoted_count": _safe_int(summary.get("promoted_count")),
            "btc_beating_candidate_count": _safe_int(summary.get("btc_beating_candidate_count")),
            "provisional_candidate_count": _safe_int(summary.get("provisional_candidate_count")),
            "candidate_pool_count": _safe_int(summary.get("candidate_pool_count")),
            "requested_timeframes": _string_list(execution_profile.get("requested_timeframes")),
            "requested_symbols": _string_list(execution_profile.get("requested_symbols")),
            "low_ram_profile": bool(execution_profile.get("low_ram_profile", False)),
        },
        "decision": {
            "next_action": str(decision.get("next_action") or ""),
            "promoted_total": _safe_int(decision.get("promoted_total")),
            "total_evaluated": _safe_int(decision.get("total_evaluated")),
            "max_peak_rss_mib": _safe_float(decision.get("max_peak_rss_mib")),
            "valid_strategy_found": bool(decision.get("valid_strategy_found", False)),
        },
        "memory": {
            "status": str(memory.get("status") or ""),
            "peak_rss_mib": _safe_float(memory.get("peak_rss_mib")),
            "soft_limit_mib": _safe_float(memory.get("soft_limit_mib")),
            "hard_limit_mib": _safe_float(memory.get("hard_limit_mib")),
        },
        "portfolio": {
            "construction_basis": str(portfolio.get("construction_basis") or ""),
            "oos_return": _safe_float(portfolio_oos.get("total_return")),
            "oos_sharpe": _safe_float(portfolio_oos.get("sharpe")),
            "oos_max_drawdown": _safe_float(portfolio_oos.get("max_drawdown")),
        },
        "time_window": dict(summary.get("windows") or {}),
        "timeframes": [_normalize_metric_row(row) for row in timeframe_rows],
        "top_candidates": [_normalize_metric_row(row) for row in top_candidates[:8]],
        "portfolio_weights": [_normalize_portfolio_weight(row) for row in weights[:8]],
        "notes": [
            {"label": str(key), "value": str(value)}
            for key, value in notes.items()
            if str(value or "").strip()
        ],
        "warnings": [
            str(item) for item in list(bundle.get("warnings") or []) if str(item or "").strip()
        ],
    }


__all__ = ["empty_exact_window_payload", "load_exact_window_summary_payload"]


def main(argv: list[str] | None = None) -> int:
    """Module-mode entry: uv run python -m lumina_quant.dashboard.exact_window_service."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="lumina_quant.dashboard.exact_window_service",
        description="Emit exact-window summary payload as JSON.",
    )
    parser.add_argument("--root", default=None, help="Override exact-window bundle root path.")
    args = parser.parse_args(argv)
    payload = load_exact_window_summary_payload(root=args.root)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
