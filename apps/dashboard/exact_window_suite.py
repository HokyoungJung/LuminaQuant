from __future__ import annotations

import html
import importlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_exact_window_bundle = importlib.import_module(
    "apps.dashboard.services.exact_window"
).load_exact_window_bundle
_eval_suite_module = importlib.import_module("lumina_quant.eval.exact_window_suite")
_metrics_daily = _eval_suite_module._metrics_daily

TIMEFRAME_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
TIMEFRAME_ORDER_MAP = {token: idx for idx, token in enumerate(TIMEFRAME_ORDER)}
_SPLIT_ORDER = ("train", "val", "oos")
_METRIC_FIELDS: list[tuple[str, str, str]] = [
    ("return", "Return", "percent"),
    ("cagr", "CAGR", "percent"),
    ("sharpe", "Sharpe", "float3"),
    ("sortino", "Sortino", "float3"),
    ("calmar", "Calmar", "float3"),
    ("max_drawdown", "Max DD", "percent"),
    ("volatility", "Volatility", "percent"),
    ("trade_count", "Trades", "int"),
    ("turnover", "Turnover", "float3"),
    ("win_rate", "Win Rate", "percent"),
    ("avg_trade", "Avg Trade", "percent"),
    ("exposure", "Exposure", "percent"),
    ("deflated_sharpe", "Deflated Sharpe", "float3"),
    ("pbo", "PBO", "float3"),
    ("spa_pvalue", "SPA p-value", "float3"),
    ("benchmark_corr", "Benchmark Corr", "float3"),
    ("rolling_sharpe_min", "Rolling Sharpe Min", "float3"),
    ("stability", "Stability", "float3"),
    ("worst_month", "Worst Month", "percent"),
]

_DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');
.stApp, .stMarkdown, .stDataFrame, .stMetric {
    font-family: "Sora", sans-serif;
}
code, pre, .exact-window-mono {
    font-family: "IBM Plex Mono", monospace;
}
.exact-window-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.8rem;
    margin: 0.25rem 0 1rem 0;
}
.exact-window-card {
    border: 1px solid rgba(120, 145, 190, 0.22);
    background: linear-gradient(180deg, rgba(16,24,39,0.98), rgba(12,20,34,0.94));
    border-radius: 16px;
    padding: 0.95rem 1rem 0.9rem 1rem;
    min-height: 124px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.20);
}
.exact-window-card .label {
    color: #8fb0de;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}
.exact-window-card .value {
    color: #f8fbff;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.05;
    margin-bottom: 0.35rem;
}
.exact-window-card .sub {
    color: #b9c8df;
    font-size: 0.84rem;
    line-height: 1.35;
}
.exact-window-tf-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(255px, 1fr));
    gap: 0.85rem;
    margin: 0.35rem 0 1rem 0;
}
.exact-window-tf-card {
    border: 1px solid rgba(120, 145, 190, 0.20);
    background: linear-gradient(180deg, rgba(9,16,30,0.98), rgba(9,16,30,0.92));
    border-radius: 18px;
    padding: 0.9rem 1rem;
}
.exact-window-tf-card .tf {
    color: #4ad3ff;
    font-size: 1.15rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.exact-window-tf-card .name {
    color: #eff5ff;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.exact-window-tf-card .meta {
    color: #b8c7de;
    font-size: 0.82rem;
    line-height: 1.42;
}
.exact-window-section-caption {
    color: #8eaad0;
    font-size: 0.9rem;
    margin-bottom: 0.55rem;
}
.exact-window-panel {
    border: 1px solid rgba(120, 145, 190, 0.22);
    background: linear-gradient(180deg, rgba(10,18,33,0.98), rgba(7,14,27,0.94));
    border-radius: 18px;
    padding: 1rem 1rem 0.9rem 1rem;
    margin: 0 0 1rem 0;
    box-shadow: 0 12px 30px rgba(0,0,0,0.20);
}
.exact-window-panel .title {
    color: #f3f8ff;
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}
.exact-window-panel .sub {
    color: #9fb4d6;
    font-size: 0.84rem;
    line-height: 1.45;
}
.exact-window-chip-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 0.55rem;
}
.exact-window-chip {
    border: 1px solid rgba(107, 162, 255, 0.28);
    background: rgba(38, 61, 103, 0.35);
    color: #d7e5ff;
    border-radius: 999px;
    padding: 0.28rem 0.62rem;
    font-size: 0.73rem;
    letter-spacing: 0.03em;
}
.exact-window-banner {
    border: 1px solid rgba(98, 196, 255, 0.30);
    background: linear-gradient(90deg, rgba(10,29,48,0.98), rgba(7,17,29,0.96));
    border-radius: 20px;
    padding: 1rem 1.05rem;
    margin-bottom: 1rem;
}
.exact-window-banner .headline {
    color: #f9fbff;
    font-size: 1.15rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.exact-window-banner .body {
    color: #b5c8e9;
    font-size: 0.9rem;
    line-height: 1.45;
}
</style>
"""


def _timeframe_sort_key(timeframe: str) -> tuple[int, str]:
    token = str(timeframe or "")
    return (TIMEFRAME_ORDER_MAP.get(token, len(TIMEFRAME_ORDER_MAP)), token)


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if pd.notna(out) else float(default)


def _format_value(value: Any, mode: str) -> str:
    if value is None or value == "":
        return "—"
    numeric = _safe_float(value)
    if mode == "percent":
        return f"{numeric:.2%}"
    if mode == "float3":
        return f"{numeric:.3f}"
    if mode == "float2":
        return f"{numeric:.2f}"
    if mode == "int":
        return f"{round(numeric):,}"
    return str(value)


def _stream_frame(stream: list[dict[str, Any]], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    equity = 1.0
    peak = 1.0
    for point in list(stream or []):
        ts = _coerce_timestamp(point.get("datetime", point.get("t")))
        if ts is None or pd.isna(ts):
            continue
        ret = float(point.get("v") or 0.0)
        equity *= 1.0 + ret
        peak = max(peak, equity)
        rows.append(
            {
                "timestamp": ts,
                "split": split,
                "return": ret,
                "equity": equity,
                "cumulative_return": equity - 1.0,
                "drawdown": 1.0 - (equity / max(peak, 1e-12)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "split", "return", "equity", "cumulative_return", "drawdown"]
        )
    return pd.DataFrame(rows).sort_values("timestamp")


def _chart_frame(best_row: dict[str, Any], field: str) -> pd.DataFrame:
    stream_map = dict(best_row.get("return_streams") or {})
    frames = [_stream_frame(stream_map.get(split) or [], split) for split in _SPLIT_ORDER]
    merged = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if merged.empty:
        return pd.DataFrame()
    pivot = merged.pivot_table(index="timestamp", columns="split", values=field, aggfunc="last")
    ordered_cols = [split for split in _SPLIT_ORDER if split in pivot.columns]
    return pivot[ordered_cols].sort_index()


def _timeframe_summary_frame(timeframe_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in sorted(timeframe_rows, key=lambda item: _timeframe_sort_key(str(item.get("timeframe") or ""))):
        best = dict(row.get("best_row") or {})
        val = dict(best.get("val") or {})
        oos = dict(best.get("oos") or {})
        memory = dict(row.get("memory_evidence") or {})
        rows.append(
            {
                "timeframe": row.get("timeframe"),
                "strategy": best.get("strategy_class"),
                "best_name": best.get("name"),
                "promoted": bool(best.get("promoted")),
                "candidate_pool": int(row.get("candidate_pool_strategy_count") or 0),
                "val_return": val.get("return"),
                "val_sharpe": val.get("sharpe"),
                "val_pbo": val.get("pbo"),
                "oos_return": oos.get("return"),
                "oos_sharpe": oos.get("sharpe"),
                "oos_pbo": oos.get("pbo"),
                "oos_mdd": oos.get("max_drawdown", oos.get("mdd")),
                "oos_trades": oos.get("trade_count"),
                "oos_turnover": oos.get("turnover"),
                "oos_win_rate": oos.get("win_rate"),
                "peak_rss_mib": memory.get("peak_rss_mib"),
                "rejects": ", ".join(best.get("rejection_reasons") or []),
            }
        )
    return pd.DataFrame(rows)


def _best_row_snapshot(best_row: dict[str, Any]) -> pd.DataFrame:
    if not best_row:
        return pd.DataFrame()
    val = dict(best_row.get("val") or {})
    oos = dict(best_row.get("oos") or {})
    return pd.DataFrame(
        [
            {
                "strategy_class": best_row.get("strategy_class"),
                "name": best_row.get("name"),
                "family": best_row.get("family"),
                "timeframe": best_row.get("strategy_timeframe"),
                "promoted": best_row.get("promoted"),
                "candidate_pool_eligible": best_row.get("candidate_pool_eligible"),
                "validation_score": best_row.get("validation_score"),
                "timeframe_selection_score": best_row.get("timeframe_selection_score"),
                "val_return": val.get("return"),
                "val_sharpe": val.get("sharpe"),
                "val_pbo": val.get("pbo"),
                "oos_return": oos.get("return"),
                "oos_sharpe": oos.get("sharpe"),
                "oos_pbo": oos.get("pbo"),
                "trade_count_oos": oos.get("trade_count"),
                "turnover_oos": oos.get("turnover"),
                "max_drawdown_oos": oos.get("max_drawdown", oos.get("mdd")),
                "win_rate_oos": oos.get("win_rate"),
            }
        ]
    )


def _monthly_hurdle_frame(best_row: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_key, payload in (("validation", best_row.get("validation_monthly_hurdle")), ("oos", best_row.get("oos_monthly_hurdle"))):
        for item in list(payload or []):
            copied = dict(item)
            copied["split"] = split_key
            rows.append(copied)
    return pd.DataFrame(rows)


def _reject_reason_frame(timeframe_row: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(list(timeframe_row.get("reject_reason_counts") or []))
    if frame.empty:
        return frame
    return frame.sort_values(["count", "rejection_reason"], ascending=[False, True])


def _split_metrics_frame(best_row: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field, label, mode in _METRIC_FIELDS:
        row = {"metric": label}
        for split in _SPLIT_ORDER:
            payload = dict(best_row.get(split) or {})
            row[split] = _format_value(payload.get(field), mode)
        rows.append(row)
    return pd.DataFrame(rows)


def _portfolio_metrics_frame(summary: dict[str, Any]) -> pd.DataFrame:
    portfolio = dict(summary.get("portfolio") or {})
    rows: list[dict[str, Any]] = []
    for field, label, mode in _METRIC_FIELDS[:15]:
        row = {"metric": label}
        for split in _SPLIT_ORDER:
            payload = dict(portfolio.get(split) or {})
            row[split] = _format_value(payload.get(field), mode)
        rows.append(row)
    return pd.DataFrame(rows)


def _portfolio_weights_frame(summary: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(list((summary.get("portfolio") or {}).get("weights") or []))
    if frame.empty:
        return frame
    preferred = [
        "timeframe",
        "strategy_class",
        "name",
        "weight",
        "oos_return",
        "oos_sharpe",
        "family",
        "basis",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame[columns].sort_values("weight", ascending=False)


def _portfolio_chart_frame(summary: dict[str, Any], field: str) -> pd.DataFrame:
    portfolio = dict(summary.get("portfolio") or {})
    streams = dict(portfolio.get("return_streams") or {})
    frames = [_stream_frame(streams.get(split) or [], split) for split in _SPLIT_ORDER]
    merged = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if merged.empty:
        return pd.DataFrame()
    pivot = merged.pivot_table(index="timestamp", columns="split", values=field, aggfunc="last")
    ordered_cols = [split for split in _SPLIT_ORDER if split in pivot.columns]
    return pivot[ordered_cols].sort_index()


def _status_chip(label: str, value: Any) -> str:
    return f'<span class="exact-window-chip"><b>{html.escape(str(label))}</b>: {html.escape(str(value))}</span>'


def _panel_html(title: str, sub: str, chips: list[str] | None = None) -> str:
    chip_html = f'<div class="exact-window-chip-row">{"".join(chips or [])}</div>' if chips else ""
    return (
        '<div class="exact-window-panel">'
        f'<div class="title">{html.escape(title)}</div>'
        f'<div class="sub">{html.escape(sub)}</div>'
        f"{chip_html}"
        "</div>"
    )


def _details_rows_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    detail_sources: list[tuple[str, list[dict[str, Any]]]] = [("root_latest", list(bundle.get("details") or []))]
    for stage, stage_rows in sorted((bundle.get("followup_details") or {}).items()):
        if isinstance(stage_rows, list):
            detail_sources.append((stage, stage_rows))
    seen: set[tuple[str, str]] = set()
    for source_stage, details in detail_sources:
        for row in details:
            candidate_key = (str(row.get("candidate_id") or ""), str(source_stage))
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            train = dict(row.get("train") or {})
            val = dict(row.get("val") or {})
            oos = dict(row.get("oos") or {})
            symbols = list(row.get("symbols") or [])
            symbol_set = {str(symbol) for symbol in symbols}
            metal_count = sum(1 for symbol in symbol_set if symbol in {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"})
            asset_mix = (
                "crypto-metal mix"
                if metal_count and metal_count < len(symbol_set)
                else "pure metal"
                if metal_count == len(symbol_set) and metal_count > 0
                else "crypto basket"
                if len(symbol_set) >= 2
                else "single asset"
            )
            reject_counter = Counter((row.get("hard_reject_reasons") or {}).keys())
            rows.append(
                {
                    "source_stage": source_stage,
                    "timeframe": row.get("strategy_timeframe"),
                    "family": row.get("family"),
                    "strategy": row.get("strategy_class"),
                    "name": row.get("name"),
                    "asset_mix": asset_mix,
                    "symbols": ", ".join(symbols),
                    "symbol_count": len(symbols),
                    "train_return": train.get("return"),
                    "train_sharpe": train.get("sharpe"),
                    "val_return": val.get("return"),
                    "val_sharpe": val.get("sharpe"),
                    "val_pbo": val.get("pbo"),
                    "oos_return": oos.get("return"),
                    "oos_sharpe": oos.get("sharpe"),
                    "oos_sortino": oos.get("sortino"),
                    "oos_calmar": oos.get("calmar"),
                    "oos_mdd": oos.get("max_drawdown", oos.get("mdd")),
                    "oos_volatility": oos.get("volatility"),
                    "oos_trades": oos.get("trade_count"),
                    "oos_turnover": oos.get("turnover"),
                    "oos_win_rate": oos.get("win_rate"),
                    "oos_avg_trade": oos.get("avg_trade"),
                    "oos_deflated_sharpe": oos.get("deflated_sharpe"),
                    "oos_pbo": oos.get("pbo"),
                    "reject_count": sum(reject_counter.values()),
                    "rejects": ", ".join(sorted(reject_counter)),
                    "candidate_id": row.get("candidate_id"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["oos_sharpe", "oos_return", "val_sharpe"],
        ascending=[False, False, False],
    )


def _metric_matrix_frame(timeframe_rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in sorted(timeframe_rows, key=lambda item: _timeframe_sort_key(str(item.get("timeframe") or ""))):
        best = dict(row.get("best_row") or {})
        oos = dict(best.get("oos") or {})
        val = dict(best.get("val") or {})
        records.extend(
            [
                {"timeframe": row.get("timeframe"), "metric": "Val Return", "value": val.get("return")},
                {"timeframe": row.get("timeframe"), "metric": "Val Sharpe", "value": val.get("sharpe")},
                {"timeframe": row.get("timeframe"), "metric": "OOS Return", "value": oos.get("return")},
                {"timeframe": row.get("timeframe"), "metric": "OOS Sharpe", "value": oos.get("sharpe")},
                {"timeframe": row.get("timeframe"), "metric": "OOS Sortino", "value": oos.get("sortino")},
                {"timeframe": row.get("timeframe"), "metric": "OOS Calmar", "value": oos.get("calmar")},
                {"timeframe": row.get("timeframe"), "metric": "OOS Max DD", "value": oos.get("max_drawdown", oos.get("mdd"))},
                {"timeframe": row.get("timeframe"), "metric": "OOS Trades", "value": oos.get("trade_count")},
                {"timeframe": row.get("timeframe"), "metric": "OOS Win Rate", "value": oos.get("win_rate")},
                {"timeframe": row.get("timeframe"), "metric": "OOS Turnover", "value": oos.get("turnover")},
                {"timeframe": row.get("timeframe"), "metric": "OOS PBO", "value": oos.get("pbo")},
            ]
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    pivot = frame.pivot_table(index="metric", columns="timeframe", values="value", aggfunc="last")
    ordered = [column for column in TIMEFRAME_ORDER if column in pivot.columns]
    return pivot[ordered]


def _family_mix_frame(details_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if details_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    family = (
        details_frame.groupby(["family", "timeframe"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["candidate_count", "family"], ascending=[False, True])
    )
    mix = (
        details_frame.groupby(["asset_mix", "timeframe"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["candidate_count", "asset_mix"], ascending=[False, True])
    )
    return family, mix


def _top_candidates(details_frame: pd.DataFrame, *, columns: list[str], limit: int = 15) -> pd.DataFrame:
    if details_frame.empty:
        return details_frame
    selected = [column for column in columns if column in details_frame.columns]
    return details_frame.head(limit)[selected]


def _followup_runs_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stem, payload in sorted((bundle.get("followup_status") or {}).items()):
        if not isinstance(payload, dict):
            continue
        best = dict(payload.get("best_row") or {})
        oos = dict(best.get("oos") or {})
        memory = dict(payload.get("memory_evidence") or {})
        rows.append(
            {
                "stage": stem,
                "generated_at": payload.get("generated_at"),
                "run_id": payload.get("run_id"),
                "timeframe": payload.get("timeframe"),
                "status": payload.get("status"),
                "note": payload.get("note") or payload.get("message") or payload.get("reason"),
                "best_name": best.get("name"),
                "strategy": best.get("strategy_class"),
                "oos_return": oos.get("return"),
                "oos_sharpe": oos.get("sharpe"),
                "oos_pbo": oos.get("pbo"),
                "peak_rss_mib": memory.get("peak_rss_mib") or payload.get("peak_rss_mib"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("generated_at", ascending=False)


def _registry_frame(
    bundle: dict[str, Any],
    *,
    key: str = "registry",
    source_label: str = "canonical",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in list(bundle.get(key) or []):
        rows.append(
            {
                "source": source_label,
                "run_id": entry.get("run_id"),
                "status": entry.get("status"),
                "timeframes": ", ".join(list(entry.get("requested_timeframes") or [])),
                "symbols": ", ".join(list(entry.get("requested_symbols") or [])[:6]),
                "allow_metals": entry.get("allow_metals"),
                "chunk_days": entry.get("chunk_days"),
                "run_signature": entry.get("run_signature"),
                "peak_rss_mib": entry.get("peak_rss_mib"),
                "promoted_count": entry.get("promoted_count"),
                "log_or_manifest": entry.get("log_path") or entry.get("manifest_path"),
                "updated_at_utc": entry.get("updated_at_utc"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("updated_at_utc", ascending=False)


def _pipeline_family_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    payload = dict(bundle.get("pipeline_manifest") or {})
    rows: list[dict[str, Any]] = []
    for family in list(payload.get("families") or []):
        rows.append(
            {
                "family_id": family.get("family_id"),
                "execution_style": family.get("execution_style"),
                "target_timeframes": ", ".join(list(family.get("target_timeframes") or [])),
                "target_universe": ", ".join(list(family.get("target_universe") or [])),
                "preferred_metrics": ", ".join(list(family.get("preferred_metrics") or [])),
                "rationale": family.get("rationale"),
            }
        )
    return pd.DataFrame(rows)


def _coverage_status_frame(summary: dict[str, Any]) -> pd.DataFrame:
    frame = _coverage_frame(summary)
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        symbol = str(row.get("symbol") or "")
        category = (
            "metal"
            if symbol in {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}
            else "crypto"
        )
        start = row.get("coverage_start")
        end = row.get("coverage_end")
        rows.append(
            {
                "symbol": symbol,
                "category": category,
                "coverage_start": start,
                "coverage_end": end,
                "full_start_coverage": row.get("full_start_coverage"),
                "requested_oos_end": row.get("requested_oos_end"),
            }
        )
    return pd.DataFrame(rows)


def _summary_banner(decision: dict[str, Any], bundle: dict[str, Any], summary: dict[str, Any]) -> str:
    metals_blocker = bundle.get("followup_status", {}).get("metals_blocker_latest") or {}
    blocked_metals = list(metals_blocker.get("blocked_metals") or [])
    next_action = str((bundle.get("next_iteration") or {}).get("next_action") or decision.get("next_action") or "review")
    headline = "Exact-Window Decision Board"
    body = (
        f"Strict anchor remains {int(decision.get('promoted_total') or 0)} promoted strategy(ies). "
        f"Next action: {next_action}. "
        f"Universe currently evaluates {len(summary.get('eligible_symbols') or [])} eligible symbol(s)"
        f" / {len(summary.get('execution_profile', {}).get('requested_symbols') or [])} requested."
    )
    chips = [
        _status_chip("Decision", decision.get("generated_at") or "n/a"),
        _status_chip("Run root", Path(str(bundle.get("run_root") or '')).name or "n/a"),
        _status_chip("Metals blocked", len(blocked_metals)),
        _status_chip("Candidate pool", int(decision.get("candidate_pool_total") or 0)),
        _status_chip("Peak RSS MiB", f"{float(decision.get('max_peak_rss_mib') or 0.0):.1f}"),
    ]
    return (
        '<div class="exact-window-banner">'
        f'<div class="headline">{html.escape(headline)}</div>'
        f'<div class="body">{html.escape(body)}</div>'
        f'<div class="exact-window-chip-row">{"".join(chips)}</div>'
        '</div>'
    )


def _strict_pass_frame(decision: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timeframe_row in list(decision.get("timeframe_rows") or []):
        best = dict(timeframe_row.get("best_row") or {})
        if not bool(best.get("promoted")):
            continue
        oos = dict(best.get("oos") or {})
        rows.append(
            {
                "timeframe": timeframe_row.get("timeframe"),
                "strategy": best.get("strategy_class"),
                "name": best.get("name"),
                "oos_return": oos.get("return"),
                "oos_sharpe": oos.get("sharpe"),
                "oos_pbo": oos.get("pbo"),
                "oos_trades": oos.get("trade_count"),
            }
        )
    return pd.DataFrame(rows)


def _candidate_pool_frame(decision: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timeframe_row in list(decision.get("timeframe_rows") or []):
        best = dict(timeframe_row.get("best_row") or {})
        if not bool(best.get("candidate_pool_eligible")):
            continue
        oos = dict(best.get("oos") or {})
        val = dict(best.get("val") or {})
        rows.append(
            {
                "timeframe": timeframe_row.get("timeframe"),
                "strategy": best.get("strategy_class"),
                "name": best.get("name"),
                "promoted": bool(best.get("promoted")),
                "validation_score": best.get("validation_score"),
                "timeframe_selection_score": best.get("timeframe_selection_score"),
                "val_return": val.get("return"),
                "oos_return": oos.get("return"),
                "oos_sharpe": oos.get("sharpe"),
                "oos_pbo": oos.get("pbo"),
                "rejects": ", ".join(best.get("rejection_reasons") or []),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["promoted", "timeframe_selection_score"], ascending=[False, False]
    ) if rows else pd.DataFrame()


def _fail_reason_summary(bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fail_analysis = dict(bundle.get("fail_analysis") or {})
    by_reason = pd.DataFrame(list(fail_analysis.get("counts_by_rejection_reason") or []))
    by_timeframe = pd.DataFrame(list(fail_analysis.get("counts_by_timeframe_reason") or []))
    proposals = pd.DataFrame(list(fail_analysis.get("strategy_next_steps") or []))
    return by_reason, by_timeframe, proposals


def _coverage_frame(summary: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(list(summary.get("coverage") or []))
    if frame.empty:
        return frame
    preferred = [
        "symbol",
        "coverage_start",
        "coverage_end",
        "full_start_coverage",
        "requested_oos_end",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame[columns]


def _format_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    formatted = frame.copy()
    for column in formatted.columns:
        lower = str(column).lower()
        if lower.endswith("_return") or lower in {
            "return",
            "cagr",
            "max_drawdown",
            "oos_mdd",
            "oos_return",
            "val_return",
            "win_rate",
            "oos_win_rate",
            "avg_trade",
            "volatility",
            "exposure",
            "worst_month",
            "weight",
        }:
            formatted[column] = formatted[column].map(lambda value: _format_value(value, "percent"))
        elif lower.endswith("_sharpe") or lower in {
            "sharpe",
            "sortino",
            "calmar",
            "pbo",
            "deflated_sharpe",
            "spa_pvalue",
            "benchmark_corr",
            "rolling_sharpe_min",
            "stability",
            "validation_score",
            "timeframe_selection_score",
            "oos_pbo",
            "val_pbo",
        }:
            formatted[column] = formatted[column].map(lambda value: _format_value(value, "float3"))
        elif "trade" in lower or lower.endswith("_count") or lower.endswith("_trades"):
            formatted[column] = formatted[column].map(lambda value: _format_value(value, "int"))
        elif lower.endswith("_mib"):
            formatted[column] = formatted[column].map(lambda value: _format_value(value, "float2"))
    return formatted


def _card_html(label: str, value: str, sub: str) -> str:
    return (
        '<div class="exact-window-card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="sub">{html.escape(sub)}</div>'
        '</div>'
    )


def _timeframe_card_html(timeframe_row: dict[str, Any]) -> str:
    best = dict(timeframe_row.get("best_row") or {})
    oos = dict(best.get("oos") or {})
    lines = [
        f"<div class=\"exact-window-tf-card\"><div class=\"tf\">{html.escape(str(timeframe_row.get('timeframe') or ''))}</div>",
        f"<div class=\"name\">{html.escape(str(best.get('name') or 'No candidate'))}</div>",
        "<div class=\"meta\">",
        f"strategy: <b>{html.escape(str(best.get('strategy_class') or '—'))}</b><br>",
        f"oos return: <b>{_format_value(oos.get('return'), 'percent')}</b> | sharpe: <b>{_format_value(oos.get('sharpe'), 'float3')}</b><br>",
        f"pbo: <b>{_format_value(oos.get('pbo'), 'float3')}</b> | trades: <b>{_format_value(oos.get('trade_count'), 'int')}</b><br>",
        f"mdd: <b>{_format_value(oos.get('max_drawdown', oos.get('mdd')), 'percent')}</b> | promoted: <b>{'yes' if best.get('promoted') else 'no'}</b><br>",
        f"rejects: <b>{html.escape(', '.join(best.get('rejection_reasons') or [])) or '—'}</b>",
        "</div></div>",
    ]
    return "".join(lines)


def _plotly_style(fig: go.Figure, *, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,14,27,0.96)",
        margin=dict(l=18, r=18, t=48, b=18),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def _oos_scatter_figure(summary_frame: pd.DataFrame) -> go.Figure | None:
    if summary_frame.empty:
        return None
    frame = summary_frame.copy()
    frame["label"] = frame["timeframe"].astype(str) + " · " + frame["strategy"].astype(str)
    fig = px.scatter(
        frame,
        x="oos_return",
        y="oos_sharpe",
        size="oos_trades",
        color="oos_pbo",
        text="timeframe",
        hover_name="best_name",
        hover_data={
            "strategy": True,
            "oos_return": ":.2%",
            "oos_sharpe": ":.3f",
            "oos_pbo": ":.3f",
            "oos_mdd": ":.2%",
            "oos_trades": True,
            "rejects": True,
        },
        color_continuous_scale="Turbo",
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="rgba(255,255,255,0.35)")))
    fig.update_layout(
        title="OOS efficiency map — return vs sharpe sized by trades, colored by PBO",
        xaxis_title="OOS Return",
        yaxis_title="OOS Sharpe",
    )
    fig.update_xaxes(tickformat=".1%")
    return _plotly_style(fig, height=360)


def _rss_bar_figure(summary_frame: pd.DataFrame) -> go.Figure | None:
    if summary_frame.empty or "peak_rss_mib" not in summary_frame.columns:
        return None
    frame = summary_frame.copy().dropna(subset=["peak_rss_mib"])
    if frame.empty:
        return None
    frame["status"] = np.where(frame["promoted"], "promoted", np.where(frame["candidate_pool"] > 0, "candidate-pool", "rejected"))
    fig = px.bar(
        frame.sort_values("peak_rss_mib", ascending=False),
        x="timeframe",
        y="peak_rss_mib",
        color="status",
        hover_data={
            "best_name": True,
            "strategy": True,
            "oos_return": ":.2%",
            "oos_sharpe": ":.3f",
        },
        color_discrete_map={"promoted": "#22c55e", "candidate-pool": "#38bdf8", "rejected": "#ef4444"},
    )
    fig.update_layout(title="Peak RSS by timeframe")
    fig.update_yaxes(title="MiB")
    return _plotly_style(fig, height=320)


def _metric_heatmap_figure(metric_matrix: pd.DataFrame) -> go.Figure | None:
    if metric_matrix.empty:
        return None
    matrix = metric_matrix.copy().astype(float)
    invert_metrics = {"OOS Max DD", "OOS Turnover", "OOS PBO"}
    display = matrix.copy()
    score = matrix.copy()
    for metric in score.index:
        values = score.loc[metric].to_numpy(dtype=float)
        if metric in invert_metrics:
            values = values * -1.0
        finite = values[np.isfinite(values)]
        if finite.size > 1 and float(np.std(finite)) > 1e-12:
            values = (values - float(np.mean(finite))) / float(np.std(finite))
        elif finite.size:
            values = values - float(np.mean(finite))
        score.loc[metric] = values
    text = display.copy()
    for metric in text.index:
        for column in text.columns:
            mode = "float3"
            if "Return" in metric or "DD" in metric or "Win Rate" in metric:
                mode = "percent"
            elif "Trades" in metric:
                mode = "int"
            text.loc[metric, column] = _format_value(display.loc[metric, column], mode)
    fig = go.Figure(
        data=
        go.Heatmap(
            z=score.to_numpy(dtype=float),
            x=list(score.columns),
            y=list(score.index),
            text=text.to_numpy(),
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0.0,
            hovertemplate="timeframe=%{x}<br>metric=%{y}<br>value=%{text}<extra></extra>",
        )
    )
    fig.update_layout(title="Relative metric heatmap across timeframes")
    return _plotly_style(fig, height=420)


def _coverage_timeline_figure(coverage_status: pd.DataFrame) -> go.Figure | None:
    if coverage_status.empty:
        return None
    frame = coverage_status.copy()
    frame["coverage_start"] = pd.to_datetime(frame["coverage_start"], utc=True, errors="coerce")
    frame["coverage_end"] = pd.to_datetime(frame["coverage_end"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["coverage_start", "coverage_end"])
    if frame.empty:
        return None
    fig = px.timeline(
        frame,
        x_start="coverage_start",
        x_end="coverage_end",
        y="symbol",
        color="category",
        hover_data=["full_start_coverage", "requested_oos_end"],
        color_discrete_map={"crypto": "#38bdf8", "metal": "#f59e0b"},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Symbol coverage timeline")
    return _plotly_style(fig, height=320)


def _window_timeline_figure(summary: dict[str, Any]) -> go.Figure | None:
    windows = dict(summary.get("windows") or {})
    if not windows:
        return None
    rows = []
    for phase, start_key, end_key, color in (
        ("Train", "train_start", "train_end_exclusive", "#22c55e"),
        ("Validation", "val_start", "val_end_exclusive", "#38bdf8"),
        ("OOS", "val_end_exclusive", "actual_oos_end_exclusive", "#f97316"),
    ):
        start = _coerce_timestamp(windows.get(start_key))
        end = _coerce_timestamp(windows.get(end_key))
        if start is None or end is None or pd.isna(start) or pd.isna(end):
            continue
        rows.append({"phase": phase, "start": start, "end": end, "color": color})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    fig = px.timeline(frame, x_start="start", x_end="end", y="phase", color="phase", color_discrete_map={row["phase"]: row["color"] for row in rows})
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Exact-window phase timeline", showlegend=False)
    return _plotly_style(fig, height=250)


def _split_cockpit_html(best_row: dict[str, Any]) -> str:
    panels: list[str] = []
    for split in _SPLIT_ORDER:
        payload = dict(best_row.get(split) or {})
        panels.append(
            _panel_html(
                f"{split.upper()} cockpit",
                "Dense split snapshot for quick regime comparison.",
                [
                    _status_chip("Return", _format_value(payload.get("return"), "percent")),
                    _status_chip("Sharpe", _format_value(payload.get("sharpe"), "float3")),
                    _status_chip("Sortino", _format_value(payload.get("sortino"), "float3")),
                    _status_chip("Calmar", _format_value(payload.get("calmar"), "float3")),
                    _status_chip("Max DD", _format_value(payload.get("max_drawdown", payload.get("mdd")), "percent")),
                    _status_chip("Trades", _format_value(payload.get("trade_count"), "int")),
                    _status_chip("Win Rate", _format_value(payload.get("win_rate"), "percent")),
                    _status_chip("Avg Trade", _format_value(payload.get("avg_trade"), "percent")),
                    _status_chip("Turnover", _format_value(payload.get("turnover"), "float3")),
                    _status_chip("Exposure", _format_value(payload.get("exposure"), "percent")),
                    _status_chip("Vol", _format_value(payload.get("volatility"), "percent")),
                    _status_chip("PBO", _format_value(payload.get("pbo"), "float3")),
                ],
            )
        )
    return '<div class="exact-window-card-grid">' + "".join(panels) + "</div>"


def _preferred_followup_best(
    bundle: dict[str, Any],
    stems: list[str],
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    followup_status = dict(bundle.get("followup_status") or {})
    for stem in stems:
        payload = dict(followup_status.get(stem) or {})
        best = dict(payload.get("best_row") or {})
        if best:
            return stem, payload, best
    return None, {}, {}


def _deployment_candidate_rows(bundle: dict[str, Any], decision: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timeframe_row in list(decision.get("timeframe_rows") or []):
        best = dict(timeframe_row.get("best_row") or {})
        if not bool(best.get("promoted")):
            continue
        rows.append(
            {
                **best,
                "_deployment_role": "strict anchor",
                "_deployment_label": f"{timeframe_row.get('timeframe')} strict anchor",
                "_deployment_stage": "root_decision",
                "_deployment_run_id": None,
                "_deployment_memory_evidence": dict(timeframe_row.get("memory_evidence") or {}),
                "_deployment_weight": 0.50,
            }
        )
    stage, payload, best = _preferred_followup_best(
        bundle,
        [
            "4h_btc_xag_tuned_latest",
            "4h_metals_adaptive_latest",
            "4h_metals_core_adaptive_latest",
        ],
    )
    if best:
        rows.append(
            {
                **best,
                "_deployment_role": "mixed-asset watchlist",
                "_deployment_label": "4h BTC/XAG watchlist",
                "_deployment_stage": stage,
                "_deployment_run_id": payload.get("run_id"),
                "_deployment_memory_evidence": dict(payload.get("memory_evidence") or {}),
                "_deployment_weight": 0.50,
            }
        )
    return rows


def _weighted_candidate_stream(rows: list[dict[str, Any]], split: str) -> list[dict[str, float]]:
    if not rows:
        return []
    weight_map = {
        str(row.get("candidate_id") or row.get("name") or idx): float(row.get("_deployment_weight", 0.0))
        for idx, row in enumerate(rows)
    }
    total_weight = float(sum(weight_map.values()))
    if total_weight <= 0.0:
        total_weight = float(len(rows))
        for key in list(weight_map):
            weight_map[key] = 1.0
    bucket: dict[pd.Timestamp, float] = {}
    for idx, row in enumerate(rows):
        key = str(row.get("candidate_id") or row.get("name") or idx)
        weight = float(weight_map.get(key, 0.0)) / total_weight
        for point in list((row.get("return_streams") or {}).get(split) or []):
            ts = _coerce_timestamp(point.get("datetime", point.get("t")))
            if ts is None or pd.isna(ts):
                continue
            day = ts.floor("D")
            bucket[day] = float(bucket.get(day, 0.0)) + (weight * float(point.get("v") or 0.0))
    return [
        {"t": int(day.timestamp() * 1000), "v": float(bucket[day])}
        for day in sorted(bucket)
    ]


def _deployment_combo_metrics(rows: list[dict[str, Any]], split: str = "oos") -> dict[str, float]:
    stream = _weighted_candidate_stream(rows, split)
    returns = np.asarray([float(point.get("v", 0.0)) for point in stream], dtype=float)
    metrics = dict(_metrics_daily(returns)) if returns.size else dict(_metrics_daily(np.asarray([], dtype=float)))
    metrics["return"] = float(metrics.get("total_return", 0.0))
    metrics["pbo"] = max((float((row.get(split) or {}).get("pbo", 0.0) or 0.0) for row in rows), default=0.0)
    metrics["trade_count"] = float(sum(float((row.get(split) or {}).get("trade_count", 0.0) or 0.0) for row in rows))
    metrics["component_count"] = float(len(rows))
    return metrics


def _deployment_component_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        oos = dict(row.get("oos") or {})
        val = dict(row.get("val") or {})
        memory = dict(row.get("_deployment_memory_evidence") or {})
        table_rows.append(
            {
                "role": row.get("_deployment_role"),
                "label": row.get("_deployment_label"),
                "timeframe": row.get("strategy_timeframe"),
                "strategy": row.get("strategy_class"),
                "name": row.get("name"),
                "symbols": ", ".join(list(row.get("symbols") or [])),
                "weight": row.get("_deployment_weight"),
                "val_return": val.get("return"),
                "val_sharpe": val.get("sharpe"),
                "oos_return": oos.get("return"),
                "oos_sharpe": oos.get("sharpe"),
                "oos_pbo": oos.get("pbo"),
                "oos_trades": oos.get("trade_count"),
                "peak_rss_mib": memory.get("peak_rss_mib"),
                "stage": row.get("_deployment_stage"),
                "run_id": row.get("_deployment_run_id"),
            }
        )
    return pd.DataFrame(table_rows)


def _deployment_combo_chart_frame(rows: list[dict[str, Any]], split: str = "oos") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in rows:
        label = str(row.get("_deployment_label") or row.get("name") or row.get("strategy_class") or "candidate")
        frame = _stream_frame(list((row.get("return_streams") or {}).get(split) or []), label)
        if not frame.empty:
            frames.append(frame)
    combo_stream = _weighted_candidate_stream(rows, split)
    combo_frame = _stream_frame(combo_stream, f"{split} equal-weight combo")
    if not combo_frame.empty:
        frames.append(combo_frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged.pivot_table(index="timestamp", columns="split", values="cumulative_return", aggfunc="last").sort_index()


def _deployment_component_frame_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    table_rows: list[dict[str, Any]] = []
    for row in list(payload.get("components") or []):
        oos = dict(row.get("oos") or {})
        val = dict(row.get("val") or {})
        memory = dict(row.get("memory_evidence") or {})
        table_rows.append(
            {
                "role": row.get("role"),
                "label": row.get("label"),
                "timeframe": row.get("timeframe"),
                "strategy": row.get("strategy_class"),
                "name": row.get("name"),
                "symbols": ", ".join(list(row.get("symbols") or [])),
                "weight": row.get("weight"),
                "val_return": val.get("return"),
                "val_sharpe": val.get("sharpe"),
                "oos_return": oos.get("return"),
                "oos_sharpe": oos.get("sharpe"),
                "oos_pbo": oos.get("pbo"),
                "oos_trades": oos.get("trade_count"),
                "peak_rss_mib": memory.get("peak_rss_mib"),
                "stage": row.get("stage"),
                "run_id": row.get("run_id"),
            }
        )
    return pd.DataFrame(table_rows)


def _deployment_curve_frame_from_payload(payload: dict[str, Any], split: str = "oos") -> pd.DataFrame:
    stream = list((payload.get("combined_streams") or {}).get(split) or [])
    frame = _stream_frame(stream, str(payload.get("label") or f"{split} combo"))
    if frame.empty:
        return pd.DataFrame()
    return frame.pivot_table(index="timestamp", columns="split", values="cumulative_return", aggfunc="last").sort_index()


def _deployment_scenario_summary_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = list(payload.get("summary") or [])
    if not rows:
        for scenario in list(payload.get("scenarios") or []):
            oos = dict((scenario.get("metrics") or {}).get("oos") or {})
            rows.append(
                {
                    "scenario_id": scenario.get("scenario_id"),
                    "label": scenario.get("label"),
                    "selection_basis": scenario.get("selection_basis"),
                    "component_count": len(list(scenario.get("components") or [])),
                    "oos_return": oos.get("return"),
                    "oos_sharpe": oos.get("sharpe"),
                    "oos_sortino": oos.get("sortino"),
                    "oos_calmar": oos.get("calmar"),
                    "oos_max_drawdown": oos.get("max_drawdown"),
                    "oos_pbo": oos.get("pbo"),
                    "oos_trade_count": oos.get("trade_count"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_cols = [column for column in ["oos_sharpe", "oos_return"] if column in frame.columns]
    if sort_cols:
        return frame.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return frame


def _deployment_scenario_curve_frame(payload: dict[str, Any], split: str = "oos") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario in list(payload.get("scenarios") or []):
        label = str(scenario.get("label") or scenario.get("scenario_id") or "scenario")
        frame = _stream_frame(list((scenario.get("combined_streams") or {}).get(split) or []), label)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged.pivot_table(index="timestamp", columns="split", values="cumulative_return", aggfunc="last").sort_index()


def render_exact_window_dashboard(*, standalone: bool = True) -> None:
    if standalone:
        st.set_page_config(layout="wide", page_title="LuminaQuant Exact-Window Suite")
        st.title("LuminaQuant Exact-Window Validation Dashboard")
    else:
        st.header("Exact-Window Validation Dashboard")
    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)

    bundle = load_exact_window_bundle()
    decision = bundle.get("decision") or {}
    summary = bundle.get("summary") or {}
    memory_evidence = bundle.get("memory_evidence") or {}
    queue_status = bundle.get("queue_status") or {}
    next_iteration = bundle.get("next_iteration") or {}
    details_frame = _details_rows_frame(bundle)
    followup_frame = _followup_runs_frame(bundle)
    registry_frame = _registry_frame(bundle, key="registry", source_label="canonical")
    recovered_registry_frame = _registry_frame(
        bundle,
        key="recovered_registry",
        source_label="recovered_archive",
    )
    pipeline_family_frame = _pipeline_family_frame(bundle)
    coverage_status = _coverage_status_frame(summary)
    warnings = list(bundle.get("warnings") or [])

    if warnings:
        with st.expander("Artifact load warnings", expanded=False):
            for warning in warnings:
                st.warning(str(warning))

    timeframe_rows = list(decision.get("timeframe_rows") or [])
    if not timeframe_rows:
        archive_payload = dict(bundle.get("archive_payload") or bundle.get("followup_status", {}).get("backtest_log_archive_latest") or {})
        deployment_payload = dict(bundle.get("followup_status", {}).get("deployment_combo_latest") or {})
        deployment_scenarios = dict(bundle.get("followup_status", {}).get("deployment_scenarios_latest") or {})
        combined_registry = pd.concat(
            [frame for frame in (registry_frame, recovered_registry_frame) if not frame.empty],
            ignore_index=True,
        ) if (not registry_frame.empty or not recovered_registry_frame.empty) else pd.DataFrame()
        st.warning("Primary exact-window decision artifacts are missing. Showing saved metrics plus recovery/archive cockpit.")
        recovery_cards = [
            ("Canonical Runs", _format_value(len(registry_frame), "int"), "exact-window signature-backed registry snapshot"),
            ("Recovered Runs", _format_value(len(recovered_registry_frame), "int"), "advisory archive rebuilt from logs"),
            ("Pipeline Families", _format_value(len(pipeline_family_frame), "int"), "article-inspired strategy expansion thesis"),
            ("Saved Candidates", _format_value(len(details_frame), "int"), "detail rows still available without decision bundle"),
            ("Follow-up Snapshots", _format_value(len(followup_frame), "int"), "saved JSON/MD follow-up artifacts still available"),
            ("Deployment Scenarios", _format_value(deployment_scenarios.get("scenario_count"), "int"), deployment_payload.get("scenario_id") or "unavailable"),
        ]
        st.markdown(
            '<div class="exact-window-card-grid">' + ''.join(_card_html(*card) for card in recovery_cards) + '</div>',
            unsafe_allow_html=True,
        )
        recovery_tabs = st.tabs(["Saved Metrics", "Candidates", "Registry", "Pipeline Thesis", "Diagnostics"])
        with recovery_tabs[0]:
            top_metrics = st.columns(4)
            top_metrics[0].metric("Evaluated", _format_value(summary.get("evaluated_count"), "int"))
            top_metrics[1].metric("Promoted", _format_value(summary.get("promoted_count"), "int"))
            top_metrics[2].metric("Peak RSS MiB", _format_value(memory_evidence.get("peak_rss_mib"), "float"))
            top_metrics[3].metric("Requested TF", _format_value(len(summary.get("execution_profile", {}).get("requested_timeframes") or []), "int"))
            if summary:
                with st.expander("Saved summary payload", expanded=False):
                    st.json(summary)
            fail_counts = pd.DataFrame((bundle.get("fail_analysis") or {}).get("counts_by_rejection_reason") or [])
            if not fail_counts.empty:
                st.caption("Saved rejection counts")
                st.dataframe(_format_frame(fail_counts), use_container_width=True, hide_index=True)
            if not coverage_status.empty:
                st.caption("Saved symbol coverage")
                st.dataframe(_format_frame(coverage_status), use_container_width=True, hide_index=True)
        with recovery_tabs[1]:
            if details_frame.empty:
                st.info("No candidate detail rows available.")
            else:
                st.caption("Saved candidate leaderboard")
                st.dataframe(
                    _format_frame(
                        _top_candidates(
                            details_frame,
                            columns=[
                                "source_stage",
                                "timeframe",
                                "family",
                                "strategy",
                                "name",
                                "asset_mix",
                                "symbols",
                                "val_return",
                                "val_sharpe",
                                "oos_return",
                                "oos_sharpe",
                                "oos_sortino",
                                "oos_calmar",
                                "oos_mdd",
                                "oos_trades",
                                "oos_pbo",
                                "rejects",
                            ],
                            limit=30,
                        )
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        with recovery_tabs[2]:
            if combined_registry.empty:
                st.info("No canonical or recovered registry entries found.")
            else:
                st.caption("Canonical registry + recovered advisory archive")
                st.dataframe(_format_frame(combined_registry), use_container_width=True, hide_index=True)
            if followup_frame.empty:
                st.info("No follow-up artifacts available.")
            else:
                st.caption("Follow-up / archive artifacts")
                st.dataframe(_format_frame(followup_frame), use_container_width=True, hide_index=True)
        with recovery_tabs[3]:
            if pipeline_family_frame.empty:
                st.info("No pipeline thesis manifest found.")
            else:
                st.caption("Article-inspired strategy family map")
                st.dataframe(_format_frame(pipeline_family_frame), use_container_width=True, hide_index=True)
            with st.expander("Pipeline manifest", expanded=False):
                st.json(bundle.get("pipeline_manifest") or {})
        with recovery_tabs[4]:
            st.caption("Archive notes")
            st.json(archive_payload)
            st.caption("Deployment scenario placeholder")
            st.json(
                {
                    "combo": deployment_payload,
                    "scenarios": deployment_scenarios,
                    "paths": bundle.get("paths"),
                    "followup_status_root": bundle.get("followup_status_root"),
                    "warnings": warnings,
                }
            )
        return

    timeframe_options = [
        str(row.get("timeframe"))
        for row in sorted(timeframe_rows, key=lambda item: _timeframe_sort_key(str(item.get("timeframe") or "")))
    ]
    default_timeframe = "30m" if "30m" in timeframe_options else timeframe_options[0]

    st.sidebar.header("Explorer")
    selected_timeframe = st.sidebar.selectbox(
        "Timeframe",
        timeframe_options,
        index=timeframe_options.index(default_timeframe),
    )
    leaderboard_mode = st.sidebar.selectbox(
        "Leaderboard focus",
        ["overall", "selected timeframe", "mixed assets", "metals"],
        index=0,
    )
    selected_row = next(row for row in timeframe_rows if str(row.get("timeframe")) == selected_timeframe)
    selected_best = dict(selected_row.get("best_row") or {})
    deployment_artifact = dict(bundle.get("followup_status", {}).get("deployment_combo_latest") or {})
    deployment_scenarios_artifact = dict(bundle.get("followup_status", {}).get("deployment_scenarios_latest") or {})
    deployment_rows = _deployment_candidate_rows(bundle, decision)
    deployment_frame_live = _deployment_component_frame(deployment_rows)
    deployment_frame = (
        _deployment_component_frame_from_payload(deployment_artifact)
        if deployment_artifact
        else deployment_frame_live
    )
    deployment_oos_metrics_live = _deployment_combo_metrics(deployment_rows, "oos")
    deployment_oos_metrics = dict((deployment_artifact.get("metrics") or {}).get("oos") or deployment_oos_metrics_live)
    deployment_oos_curve = (
        _deployment_curve_frame_from_payload(deployment_artifact, "oos")
        if deployment_artifact
        else _deployment_combo_chart_frame(deployment_rows, "oos")
    )
    deployment_scenario_frame = _deployment_scenario_summary_frame(deployment_scenarios_artifact)
    deployment_scenario_curve = _deployment_scenario_curve_frame(deployment_scenarios_artifact, "oos")
    deployment_split_metric_rows: list[dict[str, Any]] = []
    for split in _SPLIT_ORDER:
        split_metrics = dict((deployment_artifact.get("metrics") or {}).get(split) or {})
        if not split_metrics and deployment_rows:
            split_metrics = _deployment_combo_metrics(deployment_rows, split)
        if not split_metrics:
            continue
        deployment_split_metric_rows.append(
            {
                "split": split,
                "return": split_metrics.get("return"),
                "sharpe": split_metrics.get("sharpe"),
                "sortino": split_metrics.get("sortino"),
                "calmar": split_metrics.get("calmar"),
                "max_drawdown": split_metrics.get("max_drawdown"),
                "volatility": split_metrics.get("volatility"),
                "trade_count": split_metrics.get("trade_count"),
                "pbo": split_metrics.get("pbo"),
            }
        )
    deployment_split_metrics_frame = pd.DataFrame(deployment_split_metric_rows)
    deployment_scenario_count = int(
        deployment_scenarios_artifact.get("scenario_count")
        or len(list(deployment_scenarios_artifact.get("scenarios") or []))
    )

    st.sidebar.caption(f"Decision generated: {decision.get('generated_at')}")
    st.sidebar.metric("Strict Passes", int(decision.get("promoted_total") or 0))
    st.sidebar.metric("Candidate Pool", int(decision.get("candidate_pool_total") or 0))
    st.sidebar.metric("Total Evaluated", int(decision.get("total_evaluated") or 0))
    st.sidebar.metric("Max Peak RSS (MiB)", f"{float(decision.get('max_peak_rss_mib') or 0.0):.1f}")
    queue_rows = list(queue_status.get("queue") or [])
    if queue_rows:
        st.sidebar.subheader("Queue Snapshot")
        for item in queue_rows:
            st.sidebar.write(f"- {item.get('timeframe')}: {item.get('status')} | run_id={item.get('run_id')}")
    execution_profile = dict(summary.get("execution_profile") or {})
    st.sidebar.caption(f"Requested symbols: {len(execution_profile.get('requested_symbols') or [])}")
    st.sidebar.caption(f"Eligible symbols: {len(summary.get('eligible_symbols') or [])}")
    st.sidebar.caption(f"Allow metals: {'yes' if execution_profile.get('allow_metals') else 'no'}")

    portfolio = dict(summary.get("portfolio") or {})
    selected_oos = dict(selected_best.get("oos") or {})
    candidate_pool_rate = (
        float(decision.get("candidate_pool_total") or 0.0)
        / max(1.0, float(decision.get("total_evaluated") or 0.0))
    )
    strict_rate = (
        float(decision.get("promoted_total") or 0.0)
        / max(1.0, float(decision.get("total_evaluated") or 0.0))
    )
    positive_oos = int((details_frame.get("oos_return", pd.Series(dtype=float)).fillna(0.0) > 0.0).sum()) if not details_frame.empty else 0
    mixed_asset_rows = (
        details_frame[details_frame["asset_mix"] == "crypto-metal mix"]
        if not details_frame.empty and "asset_mix" in details_frame.columns
        else pd.DataFrame()
    )
    pure_metal_rows = (
        details_frame[details_frame["asset_mix"] == "pure metal"]
        if not details_frame.empty and "asset_mix" in details_frame.columns
        else pd.DataFrame()
    )

    st.markdown(_summary_banner(decision, bundle, summary), unsafe_allow_html=True)
    header_cards = [
        ("Strict Passes", _format_value(decision.get("promoted_total"), "int"), "strictly promoted strategies under current gates"),
        ("Strict Pass Rate", _format_value(strict_rate, "percent"), f"{int(decision.get('promoted_total') or 0)} / {int(decision.get('total_evaluated') or 0)}"),
        ("Candidate Pool", _format_value(decision.get("candidate_pool_total"), "int"), str(decision.get("next_action") or "review queue")),
        ("Pool Rate", _format_value(candidate_pool_rate, "percent"), "candidate-pool eligible / total evaluated"),
        ("Total Evaluated", _format_value(decision.get("total_evaluated"), "int"), f"selected timeframe = {selected_timeframe}"),
        ("Peak RSS", f"{float(decision.get('max_peak_rss_mib') or 0.0):.1f} MiB", "highest recorded exact-window memory sample"),
        ("Positive OOS Candidates", _format_value(positive_oos, "int"), "full candidate set with positive OOS return"),
        ("Selected OOS Return", _format_value(selected_oos.get("return"), "percent"), f"{selected_best.get('name') or 'no best row'}"),
        ("Selected OOS Sharpe", _format_value(selected_oos.get("sharpe"), "float3"), f"PBO { _format_value(selected_oos.get('pbo'), 'float3') } | trades { _format_value(selected_oos.get('trade_count'), 'int') }"),
        ("Portfolio OOS Return", _format_value((portfolio.get("oos") or {}).get("return"), "percent"), f"construction = {portfolio.get('construction_basis') or 'n/a'}"),
        ("Portfolio OOS Sharpe", _format_value((portfolio.get("oos") or {}).get("sharpe"), "float3"), f"Portfolio PBO { _format_value((portfolio.get('oos') or {}).get('pbo'), 'float3') }"),
        ("Mixed-Asset Rows", _format_value(len(mixed_asset_rows), "int"), "crypto/metal pairs in saved candidate details"),
        ("Pure Metal Rows", _format_value(len(pure_metal_rows), "int"), "metal-only candidates in saved candidate details"),
    ]
    st.markdown(
        '<div class="exact-window-card-grid">' + ''.join(_card_html(*card) for card in header_cards) + '</div>',
        unsafe_allow_html=True,
    )
    summary_frame = _timeframe_summary_frame(timeframe_rows)
    metric_matrix = _metric_matrix_frame(timeframe_rows)

    st.subheader("Deployment / Portfolio Candidate Panel")
    if deployment_frame.empty:
        st.info("No deployment candidate panel available yet.")
    else:
        deployment_cards = [
            ("Deployment Sleeves", _format_value(len(deployment_rows), "int"), "current composition = strict anchor + mixed-asset watchlist"),
            ("Combo OOS Return", _format_value(deployment_oos_metrics.get("return"), "percent"), "equal-weight experimental blend"),
            ("Combo OOS Sharpe", _format_value(deployment_oos_metrics.get("sharpe"), "float3"), f"PBO { _format_value(deployment_oos_metrics.get('pbo'), 'float3') }"),
            ("Combo OOS Max DD", _format_value(deployment_oos_metrics.get("max_drawdown"), "percent"), f"trades { _format_value(deployment_oos_metrics.get('trade_count'), 'int') }"),
            ("Saved Scenario", str(deployment_artifact.get("scenario_id") or "live"), str(deployment_artifact.get("generated_at") or "computed live")),
            ("Scenario Count", _format_value(deployment_scenario_count, "int"), "saved deployment scenario variants"),
        ]
        st.markdown(
            '<div class="exact-window-card-grid">' + ''.join(_card_html(*card) for card in deployment_cards) + '</div>',
            unsafe_allow_html=True,
        )
        if deployment_artifact:
            st.caption(
                f"Saved deployment artifact: {deployment_artifact.get('label') or deployment_artifact.get('scenario_id')} "
                f"· generated {deployment_artifact.get('generated_at')}"
            )
        dep_left, dep_right = st.columns((1.3, 1.7))
        with dep_left:
            st.caption("Component sleeves")
            st.dataframe(_format_frame(deployment_frame), use_container_width=True, hide_index=True)
        with dep_right:
            st.caption("Experimental OOS overlap / staged blend")
            if deployment_oos_curve.empty:
                st.info("No OOS combo curve available.")
            else:
                st.line_chart(deployment_oos_curve, use_container_width=True)

    command_top_left, command_top_right = st.columns((1.3, 1.1))
    with command_top_left:
        st.subheader("Performance Map")
        scatter_fig = _oos_scatter_figure(summary_frame)
        if scatter_fig is None:
            st.info("No OOS scatter view available.")
        else:
            st.plotly_chart(scatter_fig, use_container_width=True)
    with command_top_right:
        st.subheader("Metric Heatmap")
        heatmap_fig = _metric_heatmap_figure(metric_matrix)
        if heatmap_fig is None:
            st.info("No metric heatmap available.")
        else:
            st.plotly_chart(heatmap_fig, use_container_width=True)

    command_bottom_left, command_bottom_mid, command_bottom_right = st.columns((1.0, 1.0, 1.15))
    with command_bottom_left:
        st.subheader("Memory by Timeframe")
        rss_fig = _rss_bar_figure(summary_frame)
        if rss_fig is None:
            st.info("No memory chart available.")
        else:
            st.plotly_chart(rss_fig, use_container_width=True)
    with command_bottom_mid:
        st.subheader("Window Timeline")
        window_fig = _window_timeline_figure(summary)
        if window_fig is None:
            st.info("No window timeline available.")
        else:
            st.plotly_chart(window_fig, use_container_width=True)
    with command_bottom_right:
        st.subheader("Coverage Timeline")
        coverage_fig = _coverage_timeline_figure(coverage_status)
        if coverage_fig is None:
            st.info("No coverage timeline available.")
        else:
            st.plotly_chart(coverage_fig, use_container_width=True)

    control_left, control_mid, control_right = st.columns((1.2, 1.4, 1.4))
    with control_left:
        st.markdown(
            _panel_html(
                "Execution profile",
                "Windowing, requested symbols/timeframes, and whether this bundle was custom-windowed.",
                [
                    _status_chip("Custom windows", "yes" if execution_profile.get("custom_windows") else "no"),
                    _status_chip("Allow metals", "yes" if execution_profile.get("allow_metals") else "no"),
                    _status_chip("Requested TF", ", ".join(execution_profile.get("requested_timeframes") or [])),
                ],
            ),
            unsafe_allow_html=True,
        )
    with control_mid:
        metals_blocker = bundle.get("followup_status", {}).get("metals_blocker_latest") or {}
        st.markdown(
            _panel_html(
                "Metal / mixed-asset status",
                str(metals_blocker.get("reason") or "Metals allowed when requested; dashboard now surfaces blocker state explicitly."),
                [
                    _status_chip("Requested", len(metals_blocker.get("requested_symbols") or execution_profile.get("requested_symbols") or [])),
                    _status_chip("Eligible", len(metals_blocker.get("eligible_symbols") or summary.get("eligible_symbols") or [])),
                    _status_chip("Blocked metals", len(metals_blocker.get("blocked_metals") or [])),
                ],
            ),
            unsafe_allow_html=True,
        )
    with control_right:
        st.markdown(
            _panel_html(
                "Memory discipline",
                "Heavy runs remain serialized and the page exposes run-level RSS evidence so the 8GB ceiling stays auditable.",
                [
                    _status_chip("Decision peak", f"{float(decision.get('max_peak_rss_mib') or 0.0):.1f} MiB"),
                    _status_chip("Root memory status", (memory_evidence or {}).get("status") or "n/a"),
                    _status_chip("Queue items", len(queue_rows)),
                ],
            ),
            unsafe_allow_html=True,
        )

    st.subheader("All Timeframes At a Glance")
    st.markdown('<div class="exact-window-section-caption">Every timeframe, best candidate, risk/return profile, and rejection reason on one screen.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="exact-window-tf-grid">'
        + ''.join(_timeframe_card_html(row) for row in sorted(timeframe_rows, key=lambda item: _timeframe_sort_key(str(item.get("timeframe") or ""))))
        + '</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(_format_frame(summary_frame), use_container_width=True, hide_index=True)

    matrix_left, matrix_right = st.columns((1.4, 1.6))
    with matrix_left:
        st.subheader("Metric Matrix")
        if metric_matrix.empty:
            st.info("No timeframe metric matrix available.")
        else:
            st.dataframe(_format_frame(metric_matrix.reset_index()), use_container_width=True, hide_index=True)
    with matrix_right:
        st.subheader("Universe Coverage / Metals")
        if coverage_status.empty:
            st.info("No coverage table available.")
        else:
            st.dataframe(coverage_status, use_container_width=True, hide_index=True)

    overview_left, overview_right = st.columns((3, 2))
    with overview_left:
        st.subheader("Selection / Promotion Overview")
        candidate_pool_frame = _candidate_pool_frame(decision)
        if candidate_pool_frame.empty:
            st.info("No candidate-pool rows are currently saved.")
        else:
            st.dataframe(_format_frame(candidate_pool_frame), use_container_width=True, hide_index=True)
    with overview_right:
        st.subheader("Strict Pass Anchor")
        strict_frame = _strict_pass_frame(decision)
        if strict_frame.empty:
            st.info("No strict-pass strategy saved.")
        else:
            st.dataframe(_format_frame(strict_frame), use_container_width=True, hide_index=True)

    st.subheader("Candidate Leaderboards")
    candidate_scope = details_frame
    if not details_frame.empty:
        if leaderboard_mode == "selected timeframe":
            candidate_scope = details_frame[details_frame["timeframe"] == selected_timeframe]
        elif leaderboard_mode == "mixed assets":
            candidate_scope = details_frame[details_frame["asset_mix"] == "crypto-metal mix"]
        elif leaderboard_mode == "metals":
            candidate_scope = details_frame[details_frame["asset_mix"].isin(["pure metal", "crypto-metal mix"])]
    board_left, board_mid, board_right = st.columns(3)
    with board_left:
        st.caption("Top by OOS quality")
        frame = (
            candidate_scope.sort_values(["oos_sharpe", "oos_return"], ascending=[False, False])
            if not candidate_scope.empty
            else pd.DataFrame()
        )
        st.dataframe(
            _format_frame(
                _top_candidates(
                    frame,
                    columns=["timeframe", "asset_mix", "strategy", "name", "symbols", "oos_return", "oos_sharpe", "oos_sortino", "oos_calmar", "oos_mdd", "oos_pbo"],
                )
            ),
            use_container_width=True,
            hide_index=True,
        )
    with board_mid:
        st.caption("Top by validation")
        frame = (
            candidate_scope.sort_values(["val_sharpe", "val_return"], ascending=[False, False])
            if not candidate_scope.empty
            else pd.DataFrame()
        )
        st.dataframe(
            _format_frame(
                _top_candidates(
                    frame,
                    columns=["timeframe", "asset_mix", "strategy", "name", "symbols", "val_return", "val_sharpe", "val_pbo", "oos_return", "oos_sharpe"],
                )
            ),
            use_container_width=True,
            hide_index=True,
        )
    with board_right:
        st.caption("Top by execution realism")
        frame = (
            candidate_scope.sort_values(["oos_trades", "oos_win_rate", "oos_avg_trade"], ascending=[False, False, False])
            if not candidate_scope.empty
            else pd.DataFrame()
        )
        st.dataframe(
            _format_frame(
                _top_candidates(
                    frame,
                    columns=["timeframe", "asset_mix", "strategy", "name", "symbols", "oos_trades", "oos_turnover", "oos_win_rate", "oos_avg_trade", "rejects"],
                )
            ),
            use_container_width=True,
            hide_index=True,
        )

    family_frame, mix_frame = _family_mix_frame(details_frame)
    mix_left, mix_right = st.columns(2)
    with mix_left:
        st.subheader("Family / Timeframe Distribution")
        if family_frame.empty:
            st.info("No family distribution available.")
        else:
            st.dataframe(family_frame, use_container_width=True, hide_index=True)
    with mix_right:
        st.subheader("Asset-Mix Distribution")
        if mix_frame.empty:
            st.info("No asset-mix distribution available.")
        else:
            st.dataframe(mix_frame, use_container_width=True, hide_index=True)

    fail_by_reason, fail_by_timeframe, fail_proposals = _fail_reason_summary(bundle)
    fail_left, fail_mid, fail_right = st.columns((1.2, 1.5, 2.1))
    with fail_left:
        st.subheader("Reject Reasons")
        if fail_by_reason.empty:
            st.info("No fail analysis available.")
        else:
            st.dataframe(fail_by_reason, use_container_width=True, hide_index=True)
    with fail_mid:
        st.subheader("Rejects by Timeframe")
        if fail_by_timeframe.empty:
            st.info("No timeframe breakdown available.")
        else:
            st.dataframe(fail_by_timeframe, use_container_width=True, hide_index=True)
    with fail_right:
        st.subheader("Next-Step Proposals")
        if fail_proposals.empty:
            st.info("No proposals generated.")
        else:
            st.dataframe(fail_proposals, use_container_width=True, hide_index=True)

    if next_iteration:
        with st.expander("Next Iteration Triage", expanded=True):
            st.json(next_iteration)

    st.subheader(f"Selected Timeframe Deep Dive — {selected_timeframe}")
    selected_metrics = [
        ("Evaluated", _format_value(selected_row.get("evaluated_count"), "int")),
        ("Candidate Pool", _format_value(selected_row.get("candidate_pool_strategy_count"), "int")),
        ("BTC Beating", _format_value(selected_row.get("btc_beating_strategy_count"), "int")),
        ("Peak RSS", f"{float((selected_row.get('memory_evidence') or {}).get('peak_rss_mib') or 0.0):.1f} MiB"),
        ("Validation Score", _format_value(selected_best.get("validation_score"), "float3")),
        ("Timeframe Selection", _format_value(selected_best.get("timeframe_selection_score"), "float3")),
        ("OOS Trades", _format_value(selected_oos.get("trade_count"), "int")),
        ("OOS PBO", _format_value(selected_oos.get("pbo"), "float3")),
    ]
    st.markdown(
        '<div class="exact-window-card-grid">'
        + ''.join(_card_html(label, value, selected_best.get('strategy_class') or selected_best.get('name') or '—') for label, value in selected_metrics)
        + '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Train / validation / OOS cockpit")
    st.markdown(_split_cockpit_html(selected_best), unsafe_allow_html=True)

    tabs = st.tabs(
        [
            "Overview",
            "Deployment",
            "Leaderboards",
            "Time-Series",
            "Split Metrics",
            "Portfolio",
            "Monthly Hurdles",
            "Universe & Metals",
            "Follow-up Runs",
            "Run Registry",
            "Reject Reasons",
            "Diagnostics",
        ]
    )

    with tabs[0]:
        best_snapshot = _best_row_snapshot(selected_best)
        if best_snapshot.empty:
            st.info("No best row available for this timeframe.")
        else:
            st.dataframe(_format_frame(best_snapshot), use_container_width=True, hide_index=True)
            left, right = st.columns((3, 2))
            with left:
                st.write("Symbols")
                st.write(selected_best.get("symbols") or [])
                st.write("Rejection reasons")
                st.write(selected_best.get("rejection_reasons") or [])
                st.write("Hard reject reasons")
                st.json(selected_best.get("hard_reject_reasons") or {})
            with right:
                st.write("Parameters")
                st.json(selected_best.get("params") or {})
                st.write("Metadata")
                st.json(selected_best.get("metadata") or {})

    with tabs[1]:
        if deployment_frame.empty:
            st.info("No deployment panel available.")
        else:
            if deployment_artifact:
                st.caption(
                    f"Primary deployment artifact: {deployment_artifact.get('scenario_id')} · "
                    f"{deployment_artifact.get('label')} · generated {deployment_artifact.get('generated_at')}"
                )
            st.dataframe(_format_frame(deployment_frame), use_container_width=True, hide_index=True)
            dep_tab_left, dep_tab_right = st.columns((1.0, 1.6))
            with dep_tab_left:
                st.caption("Blend metrics by split")
                st.dataframe(_format_frame(deployment_split_metrics_frame), use_container_width=True, hide_index=True)
            with dep_tab_right:
                if deployment_oos_curve.empty:
                    st.info("No OOS deployment blend curve available.")
                else:
                    st.caption("Equal-weight OOS blend curve")
                    st.line_chart(deployment_oos_curve, use_container_width=True)
            if not deployment_scenario_frame.empty:
                scenario_left, scenario_right = st.columns((1.2, 1.6))
                with scenario_left:
                    st.caption("Deployment scenario matrix")
                    st.dataframe(_format_frame(deployment_scenario_frame), use_container_width=True, hide_index=True)
                with scenario_right:
                    if deployment_scenario_curve.empty:
                        st.info("No deployment scenario comparison curve available.")
                    else:
                        st.caption("Scenario comparison — OOS cumulative return")
                        st.line_chart(deployment_scenario_curve, use_container_width=True)
            if deployment_artifact or deployment_scenarios_artifact:
                with st.expander("Deployment artifact paths", expanded=False):
                    st.json(
                        {
                            "deployment_combo_json": bundle.get("followup_status_root") and str(Path(str(bundle.get("followup_status_root"))) / "deployment_combo_latest.json"),
                            "deployment_combo_md": bundle.get("followup_status_root") and str(Path(str(bundle.get("followup_status_root"))) / "deployment_combo_latest.md"),
                            "deployment_scenarios_json": bundle.get("followup_status_root") and str(Path(str(bundle.get("followup_status_root"))) / "deployment_scenarios_latest.json"),
                            "deployment_scenarios_md": bundle.get("followup_status_root") and str(Path(str(bundle.get("followup_status_root"))) / "deployment_scenarios_latest.md"),
                        }
                    )

    with tabs[2]:
        if candidate_scope.empty:
            st.info("No leaderboard rows available for this filter.")
        else:
            st.dataframe(_format_frame(candidate_scope), use_container_width=True, hide_index=True)

    with tabs[3]:
        cumulative = _chart_frame(selected_best, "cumulative_return")
        raw_returns = _chart_frame(selected_best, "return")
        drawdown = _chart_frame(selected_best, "drawdown")
        if cumulative.empty:
            st.info("Return streams not available for this timeframe.")
        else:
            top, bottom = st.columns(2)
            with top:
                st.caption("Cumulative return by split")
                st.line_chart(cumulative, use_container_width=True)
            with bottom:
                st.caption("Drawdown by split")
                st.line_chart(drawdown, use_container_width=True)
            st.caption("Raw periodic return by split")
            st.line_chart(raw_returns, use_container_width=True)
            with st.expander("Raw stream preview", expanded=False):
                preview = pd.concat(
                    [_stream_frame((selected_best.get("return_streams") or {}).get(split) or [], split) for split in _SPLIT_ORDER],
                    ignore_index=True,
                )
                st.dataframe(preview.tail(100), use_container_width=True, hide_index=True)

    with tabs[4]:
        split_metrics = _split_metrics_frame(selected_best)
        st.dataframe(split_metrics, use_container_width=True, hide_index=True)

    with tabs[5]:
        portfolio_metrics = _portfolio_metrics_frame(summary)
        portfolio_weights = _portfolio_weights_frame(summary)
        port_left, port_right = st.columns((2, 3))
        with port_left:
            if portfolio_metrics.empty:
                st.info("No portfolio metrics available.")
            else:
                st.dataframe(portfolio_metrics, use_container_width=True, hide_index=True)
        with port_right:
            if portfolio_weights.empty:
                st.info("No portfolio weights available.")
            else:
                st.dataframe(_format_frame(portfolio_weights), use_container_width=True, hide_index=True)
        portfolio_curve = _portfolio_chart_frame(summary, "cumulative_return")
        portfolio_dd = _portfolio_chart_frame(summary, "drawdown")
        if not portfolio_curve.empty:
            port_curve_col, port_dd_col = st.columns(2)
            with port_curve_col:
                st.caption("Portfolio cumulative return by split")
                st.line_chart(portfolio_curve, use_container_width=True)
            with port_dd_col:
                st.caption("Portfolio drawdown by split")
                st.line_chart(portfolio_dd, use_container_width=True)

    with tabs[6]:
        hurdle_frame = _monthly_hurdle_frame(selected_best)
        if hurdle_frame.empty:
            st.info("No monthly hurdle data available.")
        else:
            st.dataframe(_format_frame(hurdle_frame), use_container_width=True, hide_index=True)
            hurdle_chart = hurdle_frame[["month", "split", "strategy_return", "threshold", "btc_buy_hold_return"]].copy()
            hurdle_chart["series"] = hurdle_chart["split"] + " strategy"
            strategy_pivot = hurdle_chart.pivot_table(index="month", columns="series", values="strategy_return", aggfunc="last")
            threshold_pivot = hurdle_chart.pivot_table(index="month", columns="split", values="threshold", aggfunc="last")
            threshold_pivot.columns = [f"{column} threshold" for column in threshold_pivot.columns]
            btc_pivot = hurdle_chart.pivot_table(index="month", columns="split", values="btc_buy_hold_return", aggfunc="last")
            btc_pivot.columns = [f"{column} btc" for column in btc_pivot.columns]
            plot_frame = pd.concat([strategy_pivot, threshold_pivot, btc_pivot], axis=1).sort_index()
            st.caption("Monthly hurdle comparison")
            st.line_chart(plot_frame, use_container_width=True)

    with tabs[7]:
        st.write("Coverage table")
        if coverage_status.empty:
            st.info("No coverage table available.")
        else:
            st.dataframe(coverage_status, use_container_width=True, hide_index=True)
        mixed_scope = details_frame[details_frame["asset_mix"] == "crypto-metal mix"] if not details_frame.empty else pd.DataFrame()
        metal_scope = details_frame[details_frame["asset_mix"].isin(["pure metal", "crypto-metal mix"])] if not details_frame.empty else pd.DataFrame()
        uni_left, uni_right = st.columns(2)
        with uni_left:
            st.caption("Mixed-asset candidates")
            if mixed_scope.empty:
                st.info("No saved crypto-metal candidates in current details bundle.")
            else:
                st.dataframe(
                    _format_frame(
                        _top_candidates(
                            mixed_scope.sort_values(["oos_sharpe", "oos_return"], ascending=[False, False]),
                            columns=["timeframe", "strategy", "name", "symbols", "val_return", "oos_return", "oos_sharpe", "oos_pbo", "oos_trades"],
                        )
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        with uni_right:
            st.caption("Metals blocker / notes")
            st.json(bundle.get("followup_status", {}).get("metals_blocker_latest") or {})
            if not metal_scope.empty:
                st.caption("Metal-linked candidates")
                st.dataframe(
                    _format_frame(
                        _top_candidates(
                            metal_scope.sort_values(["oos_sharpe", "oos_return"], ascending=[False, False]),
                            columns=["timeframe", "asset_mix", "strategy", "name", "symbols", "oos_return", "oos_sharpe", "oos_pbo", "rejects"],
                        )
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[8]:
        if followup_frame.empty:
            st.info("No follow-up runs saved.")
        else:
            st.dataframe(_format_frame(followup_frame), use_container_width=True, hide_index=True)
            status_chart = followup_frame.assign(stage_name=followup_frame["stage"]).set_index("stage_name")[["peak_rss_mib"]].dropna()
            if not status_chart.empty:
                st.caption("Follow-up peak RSS (MiB)")
                st.bar_chart(status_chart, use_container_width=True)

    with tabs[9]:
        archive_payload = bundle.get("followup_status", {}).get("backtest_log_archive_latest") or {}
        if registry_frame.empty:
            st.info("No run registry entries saved.")
        else:
            st.dataframe(_format_frame(registry_frame), use_container_width=True, hide_index=True)
        with st.expander("Archived log ledger", expanded=False):
            st.json(archive_payload)

    with tabs[10]:
        reject_frame = _reject_reason_frame(selected_row)
        if reject_frame.empty:
            st.info("No reject-reason counts available.")
        else:
            st.dataframe(reject_frame, use_container_width=True, hide_index=True)
        with st.expander("Root fail analysis", expanded=False):
            st.json(bundle.get("fail_analysis") or {})

    with tabs[11]:
        path_frame = pd.DataFrame(
            [
                {
                    "summary_path": selected_row.get("summary_path"),
                    "details_path": selected_row.get("details_path"),
                    "fail_analysis_path": selected_row.get("fail_analysis_path"),
                    "source_summary_path": selected_best.get("source_summary_path"),
                    "source_details_path": selected_best.get("source_details_path"),
                }
            ]
        )
        diag_left, diag_right = st.columns((2, 3))
        with diag_left:
            st.write("Artifact Paths")
            st.dataframe(path_frame, use_container_width=True, hide_index=True)
            st.write("Selected timeframe memory evidence")
            st.json(selected_row.get("memory_evidence") or {})
            if memory_evidence:
                with st.expander("Root latest memory evidence", expanded=False):
                    st.json(memory_evidence)
        with diag_right:
            st.write("Coverage")
            coverage_frame = _coverage_frame(summary)
            if coverage_frame.empty:
                st.info("No coverage table available.")
            else:
                st.dataframe(coverage_frame, use_container_width=True, hide_index=True)
            with st.expander("Execution Profile", expanded=False):
                st.json(summary.get("execution_profile") or {})
            with st.expander("Windows", expanded=False):
                st.json(summary.get("windows") or {})
            with st.expander("Bundle Paths", expanded=False):
                st.json(
                    {
                        "summary_generated_at": summary.get("generated_at"),
                        "latest_pointer": bundle.get("latest_pointer"),
                        "run_root": bundle.get("run_root"),
                        "root_paths": bundle.get("paths"),
                        "followup_status_root": bundle.get("followup_status_root"),
                    }
                )

if __name__ == "__main__":
    render_exact_window_dashboard()
