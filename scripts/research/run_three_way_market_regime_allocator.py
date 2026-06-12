"""Three-way market-regime allocator over incumbent, 85/15 blend, and 55/45.

This keeps the existing portfolio groups intact and switches only at the group level:
- current one-shot incumbent
- grouped incumbent/autoresearch static blend (85/15)
- autoresearch pair 55/45

Signals come from the repaired market-regime feature pipeline backed by feature_points,
so 2025+ sparse materialized windows do not leave holes in the regime analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.eval.cost_aware_reports import compute_perf_metrics
from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    MEMORY_GUARD_DIRNAME,
    acquire_portfolio_memory_guard,
    resolve_current_optimization_path,
    split_for_date,
)
from lumina_quant.utils.performance import create_calmar_ratio, create_sortino_ratio

_market_spec = importlib.util.spec_from_file_location(
    "run_group_market_regime_judgement",
    Path(__file__).resolve().parent / "run_group_market_regime_judgement.py",
)
if _market_spec is None or _market_spec.loader is None:
    raise RuntimeError("Failed to load run_group_market_regime_judgement helpers")
_market = importlib.util.module_from_spec(_market_spec)
sys.modules[_market_spec.name] = _market
_market_spec.loader.exec_module(_market)

SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "three_way_market_regime_allocator_current"
)
DEFAULT_BLEND_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_incumbent_autoresearch_static_blend_latest.json"
)
DEFAULT_MARKET_JUDGEMENT_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "market_regime_judgement_current"
    / "group_market_regime_judgement_latest.json"
)
DEFAULT_HORIZON_DAYS = 5
DEFAULT_SOFT_RSS_BYTES = int(7.5 * 1024 * 1024 * 1024)
DEFAULT_HARD_RSS_BYTES = int(8.0 * 1024 * 1024 * 1024)
PERIODS_PER_YEAR = 365


@dataclass(frozen=True, slots=True)
class AllocatorParams:
    min_confidence: float
    min_signal_score: float
    confirmation_days: int
    min_hold_days: int
    enter_autoresearch_confirmation_days: int | None = None
    enter_autoresearch_min_hold_days: int | None = None


PARAM_GRID = [
    AllocatorParams(*combo)
    for combo in (
        (0.0, 0.0, 1, 1),
        (0.0, 0.001, 1, 1),
        (0.2, 0.0, 1, 1),
        (0.2, 0.001, 1, 1),
        (0.2, 0.001, 2, 2),
        (0.35, 0.001, 2, 3),
        (0.35, 0.002, 2, 3),
        (0.5, 0.001, 2, 5),
        (0.5, 0.002, 3, 5),
    )
]
PARAM_GRID.extend(
    [
        AllocatorParams(
            min_confidence=params.min_confidence,
            min_signal_score=params.min_signal_score,
            confirmation_days=params.confirmation_days,
            min_hold_days=params.min_hold_days,
            enter_autoresearch_confirmation_days=1,
            enter_autoresearch_min_hold_days=1,
        )
        for params in list(PARAM_GRID)
        if params.confirmation_days > 1 or params.min_hold_days > 1
    ]
)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve_autoresearch_default_path() -> Path:
    return _market._resolve_autoresearch_default_path()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _streams_from_daily_returns(
    dates: list[str], daily_returns: list[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_date, raw_return in zip(dates, daily_returns, strict=True):
        day = pd.Timestamp(str(raw_date), tz="UTC").floor("D")
        rows.append(
            {
                "date": day,
                "split_group": split_for_date(day.date()),
                "return": float(raw_return),
            }
        )
    return rows


def _load_candidate_frame(*, label: str, path: Path) -> pd.DataFrame:
    payload = _load_json(path)
    portfolio_streams = payload.get("portfolio_return_streams")
    rows: list[dict[str, Any]] = []
    if isinstance(portfolio_streams, dict) and portfolio_streams:
        for split_name, points in portfolio_streams.items():
            for point in list(points or []):
                rows.append(
                    {
                        "date": pd.to_datetime(point["t"], utc=True).floor("D"),
                        "split_group": str(split_name),
                        "return": float(point["v"]),
                    }
                )
    else:
        dates = [str(day) for day in list(payload.get("dates") or [])]
        daily_returns = [
            _safe_float(value, 0.0) for value in list(payload.get("daily_returns") or [])
        ]
        if not dates or len(dates) != len(daily_returns):
            raise ValueError(f"{label}: missing usable return stream in {path}")
        rows = _streams_from_daily_returns(dates, daily_returns)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"{label}: empty return frame in {path}")
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _compute_metrics_from_returns(returns: list[float]) -> dict[str, float]:
    equity = [1.0]
    for ret in returns:
        equity.append(equity[-1] * (1.0 + float(ret)))
    base = compute_perf_metrics(list(returns), equity, PERIODS_PER_YEAR)
    sortino = (
        float(create_sortino_ratio(np.asarray(returns, dtype=float), periods=PERIODS_PER_YEAR))
        if returns
        else 0.0
    )
    calmar = float(
        create_calmar_ratio(float(base.get("cagr", 0.0)), float(base.get("max_drawdown", 0.0)))
    )
    return {
        "total_return": float(base.get("total_return", 0.0)),
        "cagr": float(base.get("cagr", 0.0)),
        "sharpe": float(base.get("sharpe", 0.0)),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(base.get("max_drawdown", 0.0)),
        "volatility": float(base.get("volatility", 0.0)),
    }


def _metrics_by_split(frame: pd.DataFrame, value_column: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val", "oos"):
        returns = (
            frame.loc[frame["split_group"].eq(split_name), value_column].astype(float).tolist()
        )
        out[split_name] = _compute_metrics_from_returns(returns)
    return out


def _objective(metrics: dict[str, float], *, turnover_fraction: float) -> float:
    return float(
        (1.0 * _safe_float(metrics.get("sharpe"), 0.0))
        + (0.35 * _safe_float(metrics.get("sortino"), 0.0))
        + (0.15 * _safe_float(metrics.get("calmar"), 0.0))
        + (12.0 * _safe_float(metrics.get("total_return"), 0.0))
        - (4.0 * _safe_float(metrics.get("max_drawdown"), 0.0))
        - (0.75 * _safe_float(metrics.get("volatility"), 0.0))
        - (1.50 * float(turnover_fraction))
    )


def _build_market_feature_frame(
    *, incumbent_path: Path, autoresearch_path: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    incumbent = _market._load_group_portfolio(label="incumbent", path=incumbent_path)
    autoresearch = _market._load_group_portfolio(label="autoresearch", path=autoresearch_path)
    merged = (
        incumbent.returns.rename(columns={"return": "incumbent"})
        .merge(
            autoresearch.returns[["date", "return"]].rename(columns={"return": "autoresearch"}),
            on="date",
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    start_day = pd.Timestamp(merged["date"].min()).tz_convert("UTC").floor("D")
    end_day = pd.Timestamp(merged["date"].max()).tz_convert("UTC").floor("D")
    symbols = sorted(set(incumbent.symbols) | set(autoresearch.symbols))
    symbol_frames: list[pd.DataFrame] = []
    coverage_summary: list[dict[str, Any]] = []
    for symbol in symbols:
        frame, coverage = _market._load_symbol_close_30m_from_feature_points(
            symbol, start_day=start_day, end_day=end_day
        )
        symbol_frames.append(frame)
        coverage_summary.append(coverage)
    feature_frame = _market._daily_market_feature_frame(symbol_frames)
    feature_frame["split_group"] = feature_frame["date"].map(
        lambda value: split_for_date(pd.Timestamp(value).date())
    )
    return feature_frame.sort_values("date").reset_index(drop=True), coverage_summary


def _signal_from_row(row: pd.Series, *, selected_rules: list[dict[str, Any]]) -> dict[str, Any]:
    judgement = _market._current_judgement(latest_row=row, selected_rules=selected_rules)
    incumbent_score = float(judgement.get("incumbent_score") or 0.0)
    autoresearch_score = float(judgement.get("autoresearch_score") or 0.0)
    return {
        "date": row["date"],
        "split_group": row["split_group"],
        "favored_group": str(judgement.get("favored_group") or "mixed"),
        "confidence": float(judgement.get("confidence") or 0.0),
        "incumbent_score": incumbent_score,
        "autoresearch_score": autoresearch_score,
        "max_signal_score": max(incumbent_score, autoresearch_score),
        "active_rules": list(judgement.get("active_rules") or []),
    }


def _raw_target_state(signal: dict[str, Any], *, params: AllocatorParams) -> str:
    if float(signal["max_signal_score"]) < float(params.min_signal_score):
        return "blend_85_15"
    if float(signal["confidence"]) < float(params.min_confidence):
        return "blend_85_15"
    favored = str(signal["favored_group"])
    if favored == "incumbent":
        return "incumbent"
    if favored == "autoresearch":
        return "autoresearch_55_45"
    return "blend_85_15"


def _transition_requirements(
    *,
    current_state: str,
    raw_state: str,
    params: AllocatorParams,
) -> tuple[int, int]:
    confirmation_days = int(params.confirmation_days)
    min_hold_days = int(params.min_hold_days)
    if current_state != "autoresearch_55_45" and raw_state == "autoresearch_55_45":
        confirmation_days = int(params.enter_autoresearch_confirmation_days or confirmation_days)
        min_hold_days = int(params.enter_autoresearch_min_hold_days or min_hold_days)
    return confirmation_days, min_hold_days


def _state_weights(state: str) -> dict[str, float]:
    return {
        "incumbent": 1.0 if state == "incumbent" else 0.0,
        "blend_85_15": 1.0 if state == "blend_85_15" else 0.0,
        "autoresearch_55_45": 1.0 if state == "autoresearch_55_45" else 0.0,
    }


def _run_allocator(
    *,
    panel: pd.DataFrame,
    params: AllocatorParams,
) -> dict[str, Any]:
    rows = panel.sort_values("date").reset_index(drop=True)
    current_state = "blend_85_15"
    pending_state = current_state
    pending_streak = 0
    days_in_state = 0
    state_rows: list[dict[str, Any]] = []
    for item in rows.itertuples(index=False):
        signal = {
            "date": item.date,
            "split_group": item.split_group,
            "favored_group": item.favored_group,
            "confidence": float(item.confidence),
            "incumbent_score": float(item.incumbent_score),
            "autoresearch_score": float(item.autoresearch_score),
            "max_signal_score": float(item.max_signal_score),
            "active_rules": list(item.active_rules),
        }
        raw_state = _raw_target_state(signal, params=params)

        if raw_state == current_state:
            pending_state = current_state
            pending_streak = 0
        else:
            if raw_state == pending_state:
                pending_streak += 1
            else:
                pending_state = raw_state
                pending_streak = 1
            transition_confirmation_days, transition_min_hold_days = _transition_requirements(
                current_state=current_state,
                raw_state=raw_state,
                params=params,
            )
            if (
                days_in_state >= transition_min_hold_days
                and pending_streak >= transition_confirmation_days
            ):
                current_state = raw_state
                days_in_state = 0
                pending_state = current_state
                pending_streak = 0

        prev_weights = (
            _state_weights(state_rows[-1]["state"]) if state_rows else _state_weights("blend_85_15")
        )
        next_weights = _state_weights(current_state)
        turnover = 0.5 * sum(abs(next_weights[key] - prev_weights[key]) for key in next_weights)
        days_in_state += 1

        chosen_return = float(getattr(item, current_state))
        state_rows.append(
            {
                "date": item.date,
                "split_group": item.split_group,
                "state": current_state,
                "raw_target_state": raw_state,
                "confidence": float(signal["confidence"]),
                "incumbent_score": float(signal["incumbent_score"]),
                "autoresearch_score": float(signal["autoresearch_score"]),
                "max_signal_score": float(signal["max_signal_score"]),
                "active_rule_count": len(signal["active_rules"]),
                "return": chosen_return,
                "turnover": float(turnover),
                "weights": next_weights,
            }
        )

    state_frame = pd.DataFrame(state_rows)
    split_metrics = _metrics_by_split(state_frame, "return")
    val_mask = state_frame["split_group"].isin(["train", "val"])
    val_metrics = _compute_metrics_from_returns(
        state_frame.loc[val_mask, "return"].astype(float).tolist()
    )
    turnover_fraction = (
        float(state_frame.loc[val_mask, "turnover"].mean()) if val_mask.any() else 0.0
    )
    objective = _objective(val_metrics, turnover_fraction=turnover_fraction)
    return {
        "state_frame": state_frame,
        "split_metrics": split_metrics,
        "validation_objective": float(objective),
        "turnover_fraction_train_val": float(turnover_fraction),
    }


def _state_summary(state_frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name in ("train", "val", "oos", "all"):
        sample = (
            state_frame
            if split_name == "all"
            else state_frame.loc[state_frame["split_group"].eq(split_name)]
        )
        counts = sample["state"].value_counts().to_dict()
        summary[split_name] = {
            "days": len(sample),
            "counts": {str(key): int(value) for key, value in counts.items()},
            "avg_turnover": float(sample["turnover"].mean()) if not sample.empty else 0.0,
        }
    return summary


def _compare_benchmarks(panel: pd.DataFrame, allocator_returns: pd.DataFrame) -> dict[str, Any]:
    merged = panel[["date", "split_group", "incumbent", "blend_85_15", "autoresearch_55_45"]].merge(
        allocator_returns[["date", "return"]].rename(columns={"return": "allocator"}),
        on="date",
        how="inner",
    )
    out: dict[str, Any] = {}
    for label in ("incumbent", "blend_85_15", "autoresearch_55_45", "allocator"):
        out[label] = _metrics_by_split(
            merged.rename(columns={label: "metric_return"}), "metric_return"
        )
    return out


def _build_markdown(payload: dict[str, Any]) -> str:
    params = dict(payload.get("selected_params") or {})
    current = dict(payload.get("current_state") or {})
    split_metrics = dict(payload.get("split_metrics") or {})
    state_summary = dict(payload.get("state_summary") or {})
    benchmarks = dict(payload.get("benchmark_metrics") or {})
    memory = dict(payload.get("memory_summary") or {})

    def _metric_line(label: str, metrics: dict[str, Any]) -> str:
        return (
            f"- {label}: return `{float(metrics.get('total_return') or 0.0):.4%}` | "
            f"sharpe `{float(metrics.get('sharpe') or 0.0):.4f}` | "
            f"sortino `{float(metrics.get('sortino') or 0.0):.4f}` | "
            f"calmar `{float(metrics.get('calmar') or 0.0):.4f}` | "
            f"max_dd `{float(metrics.get('max_drawdown') or 0.0):.4%}`"
        )

    lines = [
        "# Three-Way Market Regime Allocator",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- peak_rss_mib: `{float(memory.get('peak_rss_mib') or 0.0):.2f}`",
        f"- memory_log: `{memory.get('rss_log_path')}`",
        "",
        "## Selected Params",
        f"- min_confidence: `{params.get('min_confidence')}`",
        f"- min_signal_score: `{params.get('min_signal_score')}`",
        f"- confirmation_days: `{params.get('confirmation_days')}`",
        f"- min_hold_days: `{params.get('min_hold_days')}`",
        f"- enter_autoresearch_confirmation_days: `{params.get('enter_autoresearch_confirmation_days')}`",
        f"- enter_autoresearch_min_hold_days: `{params.get('enter_autoresearch_min_hold_days')}`",
        f"- validation_objective: `{float(payload.get('validation_objective') or 0.0):.6f}`",
        f"- turnover_fraction_train_val: `{float(payload.get('turnover_fraction_train_val') or 0.0):.6f}`",
        "",
        "## Current State",
        f"- as_of: `{current.get('date')}`",
        f"- selected_state: `{current.get('state')}`",
        f"- raw_target_state_now: `{current.get('raw_target_state')}`",
        f"- confidence: `{float(current.get('confidence') or 0.0):.4f}`",
        f"- incumbent_score: `{float(current.get('incumbent_score') or 0.0):.6f}`",
        f"- autoresearch_score: `{float(current.get('autoresearch_score') or 0.0):.6f}`",
        "",
        "## Allocator Metrics",
    ]
    for split_name in ("train", "val", "oos"):
        lines.append(_metric_line(split_name, dict(split_metrics.get(split_name) or {})))

    lines.extend(["", "## Benchmark OOS Comparison"])
    for label, title in (
        ("incumbent", "incumbent"),
        ("blend_85_15", "blend_85_15"),
        ("autoresearch_55_45", "55/45"),
        ("allocator", "allocator"),
    ):
        lines.append(_metric_line(title, dict((benchmarks.get(label) or {}).get("oos") or {})))

    lines.extend(["", "## State Usage Summary"])
    for split_name in ("train", "val", "oos", "all"):
        item = dict(state_summary.get(split_name) or {})
        lines.append(
            f"- {split_name}: days `{int(item.get('days') or 0)}` | avg_turnover `{float(item.get('avg_turnover') or 0.0):.6f}` | counts `{json.dumps(item.get('counts') or {}, sort_keys=True)}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- incumbent state = 100% current incumbent portfolio",
            "- blend_85_15 state = 100% grouped 85/15 blend portfolio",
            "- autoresearch_55_45 state = 100% 55/45 pair portfolio",
            "- all switching is done at the portfolio-group level; inner sleeve sets stay intact",
            "- market regime features are rebuilt from feature_points so the 2025+ sparse materialized windows stay filled",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def run_three_way_market_regime_allocator(
    *,
    incumbent_path: Path,
    blend_path: Path,
    autoresearch_path: Path,
    market_judgement_path: Path,
    output_dir: Path,
    soft_rss_bytes: int,
    hard_rss_bytes: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name="three_way_market_regime_allocator",
        output_dir=output_dir,
        input_path=str(incumbent_path),
        rss_log_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "three_way_market_regime_allocator_rss_latest.jsonl",
        summary_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "three_way_market_regime_allocator_memory_latest.json",
        budget_bytes=hard_rss_bytes,
        soft_limit_bytes=soft_rss_bytes,
        hard_limit_bytes=hard_rss_bytes,
    )
    status = "ok"
    error: str | None = None
    payload: dict[str, Any] | None = None
    try:
        memory_guard.sample(event="three_way_market_regime_allocator_start", context={})
        incumbent_frame = _load_candidate_frame(label="incumbent", path=incumbent_path).rename(
            columns={"return": "incumbent"}
        )
        blend_frame = _load_candidate_frame(label="blend_85_15", path=blend_path).rename(
            columns={"return": "blend_85_15"}
        )
        autoresearch_frame = _load_candidate_frame(
            label="autoresearch_55_45", path=autoresearch_path
        ).rename(columns={"return": "autoresearch_55_45"})
        panel = (
            incumbent_frame[["date", "split_group", "incumbent"]]
            .merge(blend_frame[["date", "blend_85_15"]], on="date", how="inner")
            .merge(autoresearch_frame[["date", "autoresearch_55_45"]], on="date", how="inner")
            .sort_values("date")
            .reset_index(drop=True)
        )
        memory_guard.checkpoint("three_way_returns_loaded", {"rows": len(panel)})

        market_payload = _load_json(market_judgement_path)
        selected_rules = list(market_payload.get("selected_rules") or [])
        feature_frame, coverage_summary = _build_market_feature_frame(
            incumbent_path=incumbent_path,
            autoresearch_path=autoresearch_path,
        )
        signal_rows = [
            _signal_from_row(row, selected_rules=selected_rules)
            for _, row in feature_frame.iterrows()
        ]
        signal_frame = pd.DataFrame(signal_rows)
        merged = panel.merge(signal_frame, on=["date", "split_group"], how="inner")
        memory_guard.checkpoint(
            "three_way_signals_loaded",
            {"signal_rows": len(merged), "selected_rule_count": len(selected_rules)},
        )

        best: dict[str, Any] | None = None
        for params in PARAM_GRID:
            result = _run_allocator(panel=merged, params=params)
            candidate = {"params": params, **result}
            if best is None or float(candidate["validation_objective"]) > float(
                best["validation_objective"]
            ):
                best = candidate
            memory_guard.checkpoint(
                "three_way_param_evaluated",
                {
                    "params": asdict(params),
                    "validation_objective": float(result["validation_objective"]),
                    "turnover_fraction_train_val": float(result["turnover_fraction_train_val"]),
                },
            )
        if best is None:
            raise RuntimeError("three-way allocator search produced no result")

        state_frame = best["state_frame"]
        benchmark_metrics = _compare_benchmarks(merged, state_frame)
        current_state = state_frame.sort_values("date").iloc[-1].to_dict()
        payload = {
            "artifact_kind": "portfolio_three_way_market_regime_allocator",
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now_iso(),
            "selection_basis": "train_val_three_way_market_regime_switching",
            "groups_move_as_set": True,
            "input_paths": {
                "incumbent": str(incumbent_path.resolve()),
                "blend_85_15": str(blend_path.resolve()),
                "autoresearch_55_45": str(autoresearch_path.resolve()),
                "market_judgement": str(market_judgement_path.resolve()),
            },
            "coverage_summary": coverage_summary,
            "selected_params": asdict(best["params"]),
            "validation_objective": float(best["validation_objective"]),
            "turnover_fraction_train_val": float(best["turnover_fraction_train_val"]),
            "split_metrics": best["split_metrics"],
            "benchmark_metrics": benchmark_metrics,
            "state_summary": _state_summary(state_frame),
            "current_state": current_state,
            "dates": [pd.Timestamp(value).isoformat() for value in state_frame["date"]],
            "daily_returns": [float(value) for value in state_frame["return"]],
            "states": [
                {
                    "date": pd.Timestamp(row["date"]).isoformat(),
                    "split_group": row["split_group"],
                    "state": row["state"],
                    "raw_target_state": row["raw_target_state"],
                    "confidence": float(row["confidence"]),
                    "incumbent_score": float(row["incumbent_score"]),
                    "autoresearch_score": float(row["autoresearch_score"]),
                    "max_signal_score": float(row["max_signal_score"]),
                    "active_rule_count": int(row["active_rule_count"]),
                    "turnover": float(row["turnover"]),
                    "return": float(row["return"]),
                    "weights": {key: float(val) for key, val in dict(row["weights"]).items()},
                }
                for row in state_frame.to_dict(orient="records")
            ],
            "memory_summary": {},
        }
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(
            event="three_way_market_regime_allocator_finish",
            context={"status": status, "error": error},
        )
        memory_summary = memory_guard.finalize(status=status, error=error, context={})
        memory_guard.release()
        if payload is not None:
            payload["memory_summary"] = memory_summary

    if payload is None:
        raise RuntimeError("three-way market regime allocator did not produce payload")

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = output_dir / f"three_way_market_regime_allocator_{timestamp}.json"
    out_md = output_dir / f"three_way_market_regime_allocator_{timestamp}.md"
    latest_json = output_dir / "three_way_market_regime_allocator_latest.json"
    latest_md = output_dir / "three_way_market_regime_allocator_latest.md"
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    markdown = _build_markdown(payload)
    out_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return {"payload": payload, "latest_json_path": latest_json, "latest_md_path": latest_md}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-path", type=Path, default=resolve_current_optimization_path())
    parser.add_argument("--blend-path", type=Path, default=DEFAULT_BLEND_PATH)
    parser.add_argument(
        "--autoresearch-path", type=Path, default=_resolve_autoresearch_default_path()
    )
    parser.add_argument("--market-judgement-path", type=Path, default=DEFAULT_MARKET_JUDGEMENT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--soft-rss-bytes", type=int, default=DEFAULT_SOFT_RSS_BYTES)
    parser.add_argument("--hard-rss-bytes", type=int, default=DEFAULT_HARD_RSS_BYTES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_three_way_market_regime_allocator(
        incumbent_path=Path(args.incumbent_path).resolve(),
        blend_path=Path(args.blend_path).resolve(),
        autoresearch_path=Path(args.autoresearch_path).resolve(),
        market_judgement_path=Path(args.market_judgement_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        soft_rss_bytes=max(1, int(args.soft_rss_bytes)),
        hard_rss_bytes=max(1, int(args.hard_rss_bytes)),
    )
    print(report["latest_json_path"].resolve())
    print(report["latest_md_path"].resolve())


if __name__ == "__main__":
    main()
