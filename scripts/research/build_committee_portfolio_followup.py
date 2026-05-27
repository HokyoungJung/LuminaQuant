"""Build the survivor committee portfolio follow-up artifact from saved sleeves."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.eval.exact_window_suite import _metrics_daily

REPORT_ROOT = Path(__file__).resolve().parents[2] / "var" / "reports" / "exact_window_backtests"
FOLLOWUP_ROOT = REPORT_ROOT / "followup_status"
COMPONENT_SOURCES = (
    (
        "composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80",
        REPORT_ROOT
        / "expansion_crypto_15m_30m_1h_20260310T115853Z"
        / "15m-30m-1h"
        / "exact_window_candidate_details_latest.json",
    ),
    (
        "topcap_tsmom_1h_balanced_16_4_0.015",
        REPORT_ROOT
        / "topcap_crypto_1h_focus_20260310T123813Z"
        / "1h"
        / "exact_window_candidate_details_latest.json",
    ),
    (
        "regime_breakout_1h_trend_ls_48_0.70",
        REPORT_ROOT
        / "topcap_crypto_1h_focus_20260310T123813Z"
        / "1h"
        / "exact_window_candidate_details_latest.json",
    ),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_stream(stream: list[dict[str, Any]]) -> list[tuple[pd.Timestamp, float]]:
    normalized: list[tuple[pd.Timestamp, float]] = []
    for point in list(stream or []):
        raw_ts = point.get("datetime", point.get("t"))
        ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        if pd.isna(ts) and isinstance(raw_ts, (int, float)):
            ts = pd.to_datetime(raw_ts, unit="ms", utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        normalized.append((ts, _safe_float(point.get("v"), 0.0)))
    normalized.sort(key=lambda item: item[0])
    return normalized


def _weighted_stream(rows: list[dict[str, Any]], split: str) -> list[dict[str, float]]:
    bucket: dict[pd.Timestamp, float] = defaultdict(float)
    for row in rows:
        weight = _safe_float(row.get("_portfolio_weight"), 0.0)
        for ts, value in _normalize_stream(
            list((row.get("return_streams") or {}).get(split) or [])
        ):
            bucket[ts] += weight * value
    return [{"t": float(ts.timestamp() * 1000.0), "v": float(bucket[ts])} for ts in sorted(bucket)]


def _metrics(
    rows: list[dict[str, Any]], split: str
) -> tuple[dict[str, float], list[dict[str, float]]]:
    stream = _weighted_stream(rows, split)
    returns = np.asarray([point["v"] for point in stream], dtype=float)
    metrics = dict(_metrics_daily(returns))
    metrics["return"] = float(metrics.get("total_return", 0.0))
    metrics["trade_count"] = float(
        sum(
            _safe_float((row.get(split) or {}).get("trade_count"), 0.0)
            * _safe_float(row.get("_portfolio_weight"), 0.0)
            for row in rows
        )
    )
    metrics["turnover"] = float(
        sum(
            _safe_float((row.get(split) or {}).get("turnover"), 0.0)
            * _safe_float(row.get("_portfolio_weight"), 0.0)
            for row in rows
        )
    )
    metrics["win_rate"] = float(
        sum(
            _safe_float((row.get(split) or {}).get("win_rate"), 0.0)
            * _safe_float(row.get("_portfolio_weight"), 0.0)
            for row in rows
        )
    )
    metrics["avg_trade"] = float(
        sum(
            _safe_float((row.get(split) or {}).get("avg_trade"), 0.0)
            * _safe_float(row.get("_portfolio_weight"), 0.0)
            for row in rows
        )
    )
    metrics["exposure"] = float(
        sum(
            _safe_float((row.get(split) or {}).get("exposure"), 0.0)
            * _safe_float(row.get("_portfolio_weight"), 0.0)
            for row in rows
        )
    )
    metrics["deflated_sharpe"] = float(
        sum(
            _safe_float((row.get(split) or {}).get("deflated_sharpe"), 0.0)
            * _safe_float(row.get("_portfolio_weight"), 0.0)
            for row in rows
        )
    )
    metrics["pbo"] = float(max(_safe_float((row.get(split) or {}).get("pbo"), 0.0) for row in rows))
    metrics["spa_pvalue"] = float(
        max(_safe_float((row.get(split) or {}).get("spa_pvalue"), 0.0) for row in rows)
    )
    metrics["benchmark_corr"] = float(
        sum(
            _safe_float((row.get(split) or {}).get("benchmark_corr"), 0.0)
            * _safe_float(row.get("_portfolio_weight"), 0.0)
            for row in rows
        )
    )
    metrics["rolling_sharpe_min"] = float(
        min(_safe_float((row.get(split) or {}).get("rolling_sharpe_min"), 0.0) for row in rows)
    )
    metrics["stability"] = float(
        min(_safe_float((row.get(split) or {}).get("stability"), 0.0) for row in rows)
    )
    metrics["worst_month"] = float(
        min(_safe_float((row.get(split) or {}).get("worst_month"), 0.0) for row in rows)
    )
    metrics["component_count"] = len(rows)
    return metrics, stream


def _load_named_row(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(dict(row) for row in payload if str(row.get("name") or "") == name)


def _load_default_rows(report_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [_load_named_row(path, name) for name, path in COMPONENT_SOURCES]
    rolling_payload = json.loads(
        (report_root / "followup_status" / "rolling_breakout_30m_gate_latest.json").read_text(
            encoding="utf-8"
        )
    )
    if rolling_payload.get("survives"):
        rows.append(dict(rolling_payload["gated_candidate_row"]))
    pair_payload = json.loads(
        (report_root / "followup_status" / "pair_spread_4h_xpt_xpd_retune_latest.json").read_text(
            encoding="utf-8"
        )
    )
    if pair_payload.get("survives") and pair_payload.get("survivor"):
        rows.append(dict(pair_payload["survivor"]))
    return rows, {
        **pair_payload,
        "rolling_survives": bool(rolling_payload.get("survives")),
        "rolling_recommended_action": rolling_payload.get("recommended_action"),
    }


def build_committee_portfolio_followup(
    *,
    report_root: Path | str = REPORT_ROOT,
    component_rows: list[dict[str, Any]] | None = None,
    pair_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = Path(report_root)
    rows, loaded_pair_payload = (
        (component_rows, pair_payload or {})
        if component_rows is not None
        else _load_default_rows(resolved_root)
    )
    resolved_rows = [dict(row) for row in list(rows or [])]
    if not resolved_rows:
        raise ValueError("no component rows available for committee portfolio follow-up")
    if component_rows is not None and pair_payload is None:
        loaded_pair_payload = {}

    equal_weight = 1.0 / float(len(resolved_rows))
    for idx, row in enumerate(resolved_rows):
        row["_portfolio_weight"] = equal_weight
        row["candidate_id"] = str(row.get("candidate_id") or row.get("name") or f"row-{idx}")

    metrics: dict[str, dict[str, float]] = {}
    streams: dict[str, list[dict[str, float]]] = {}
    for split in ("train", "val", "oos"):
        metrics[split], streams[split] = _metrics(resolved_rows, split)

    notes = [
        "Equal-weight survivor portfolio from the March 12 reboot handoff sleeves.",
        "RollingBreakout uses the ex-ante regime gate artifact from rolling_breakout_30m_gate_latest.json.",
    ]
    if loaded_pair_payload.get("rolling_survives"):
        notes.append("RollingBreakout 30m survived the ex-ante regime gate and was included.")
    else:
        notes.append(
            "RollingBreakout 30m was excluded because no regime gate survived the conditional sleeve thresholds."
        )
    if loaded_pair_payload.get("survives"):
        notes.append("PairSpread 4h XPT/XPD survived and was included.")
    else:
        notes.append(
            "PairSpread 4h XPT/XPD was excluded because the retune artifact did not survive."
        )
    if loaded_pair_payload:
        coverage = dict(loaded_pair_payload.get("coverage_guard") or {})
        if coverage and not coverage.get("pass"):
            notes.append(
                f"Coverage guard blocked pair inclusion: observed {coverage.get('observed_total_days')}d < required {coverage.get('min_total_days')}d."
            )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "committee_portfolio_followup",
        "schema_version": "2.0",
        "metric_method": "combined_stream_returns + weighted auxiliary diagnostics",
        "pair_survives": bool(loaded_pair_payload.get("survives")),
        "component_count": len(resolved_rows),
        "selection": [
            {
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "timeframe": row.get("strategy_timeframe"),
                "weight": row.get("_portfolio_weight"),
                "symbols": row.get("symbols"),
                "train": row.get("train"),
                "val": row.get("val"),
                "oos": row.get("oos"),
                "metadata": row.get("metadata"),
            }
            for row in resolved_rows
        ],
        "metrics": metrics,
        "combined_streams": streams,
        "notes": notes,
    }


def write_committee_portfolio_followup(
    *,
    report_root: Path | str = REPORT_ROOT,
    component_rows: list[dict[str, Any]] | None = None,
    pair_payload: dict[str, Any] | None = None,
    run_name: str = "committee_portfolio_followup",
) -> dict[str, Any]:
    payload = build_committee_portfolio_followup(
        report_root=report_root,
        component_rows=component_rows,
        pair_payload=pair_payload,
    )
    followup_root = Path(report_root) / "followup_status"
    followup_root.mkdir(parents=True, exist_ok=True)
    json_path = followup_root / f"{run_name}_latest.json"
    md_path = followup_root / f"{run_name}_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# committee portfolio follow-up",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- pair_survives: `{payload['pair_survives']}`",
        f"- component_count: `{payload['component_count']}`",
        "",
        "## components",
    ]
    for row in list(payload.get("selection") or []):
        oos = dict(row.get("oos") or {})
        lines.append(
            f"- `{row.get('name')}` | tf={row.get('timeframe')} | weight={_safe_float(row.get('weight'), 0.0):.2%} | "
            f"oos_return={_safe_float(oos.get('return'), 0.0):.4%} | "
            f"oos_sharpe={_safe_float(oos.get('sharpe'), 0.0):.3f} | "
            f"oos_pbo={_safe_float(oos.get('pbo'), 0.0):.3f}"
        )
    lines.extend(["", "## portfolio metrics"])
    for split in ("train", "val", "oos"):
        metric = dict((payload.get("metrics") or {}).get(split) or {})
        lines.append(
            f"- {split}: return={_safe_float(metric.get('return'), 0.0):.4%} | "
            f"cagr={_safe_float(metric.get('cagr'), 0.0):.4%} | "
            f"sharpe={_safe_float(metric.get('sharpe'), 0.0):.3f} | "
            f"sortino={_safe_float(metric.get('sortino'), 0.0):.3f} | "
            f"calmar={_safe_float(metric.get('calmar'), 0.0):.3f} | "
            f"max_dd={_safe_float(metric.get('max_drawdown'), 0.0):.4%} | "
            f"vol={_safe_float(metric.get('volatility'), 0.0):.4%} | "
            f"trade_count={_safe_float(metric.get('trade_count'), 0.0):.2f} | "
            f"turnover={_safe_float(metric.get('turnover'), 0.0):.4f} | "
            f"win_rate={_safe_float(metric.get('win_rate'), 0.0):.4f} | "
            f"avg_trade={_safe_float(metric.get('avg_trade'), 0.0):.6f} | "
            f"exposure={_safe_float(metric.get('exposure'), 0.0):.4f} | "
            f"deflated_sharpe={_safe_float(metric.get('deflated_sharpe'), 0.0):.4f} | "
            f"pbo={_safe_float(metric.get('pbo'), 0.0):.3f} | "
            f"spa_pvalue={_safe_float(metric.get('spa_pvalue'), 0.0):.3f} | "
            f"benchmark_corr={_safe_float(metric.get('benchmark_corr'), 0.0):.3f} | "
            f"rolling_sharpe_min={_safe_float(metric.get('rolling_sharpe_min'), 0.0):.3f} | "
            f"stability={_safe_float(metric.get('stability'), 0.0):.3f} | "
            f"worst_month={_safe_float(metric.get('worst_month'), 0.0):.4%}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


if __name__ == "__main__":
    result = write_committee_portfolio_followup()
    print(result["json_path"])
    print(result["md_path"])
