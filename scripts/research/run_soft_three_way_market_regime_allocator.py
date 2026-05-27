"""Soft three-way market-regime allocator over incumbent, 85/15 blend, and 55/45.

Unlike the hard-switch allocator, this version keeps all exposure changes continuous:
- target weights are derived from market-regime scores
- exposure can be retained in the 85/15 blend as a stabilizer
- daily movement is smoothed and turnover-capped

Portfolio groups still move as sets; no inner sleeve re-optimization occurs.
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

import pandas as pd

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    MEMORY_GUARD_DIRNAME,
    acquire_portfolio_memory_guard,
    resolve_current_optimization_path,
)

_hard_spec = importlib.util.spec_from_file_location(
    "run_three_way_market_regime_allocator",
    Path(__file__).resolve().parent / "run_three_way_market_regime_allocator.py",
)
if _hard_spec is None or _hard_spec.loader is None:
    raise RuntimeError("Failed to load run_three_way_market_regime_allocator helpers")
_hard = importlib.util.module_from_spec(_hard_spec)
sys.modules[_hard_spec.name] = _hard
_hard_spec.loader.exec_module(_hard)

SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "soft_three_way_market_regime_allocator_current"
)
DEFAULT_BLEND_PATH = _hard.DEFAULT_BLEND_PATH
DEFAULT_MARKET_JUDGEMENT_PATH = _hard.DEFAULT_MARKET_JUDGEMENT_PATH
DEFAULT_SOFT_RSS_BYTES = _hard.DEFAULT_SOFT_RSS_BYTES
DEFAULT_HARD_RSS_BYTES = _hard.DEFAULT_HARD_RSS_BYTES


@dataclass(frozen=True, slots=True)
class SoftAllocatorParams:
    min_confidence: float
    min_signal_score: float
    confidence_scale: float
    score_scale: float
    blend_floor: float
    alpha: float
    max_daily_turnover: float


PARAM_GRID = [
    SoftAllocatorParams(*combo)
    for combo in (
        (0.0, 0.0, 0.35, 0.010, 0.00, 1.00, 1.00),
        (0.0, 0.001, 0.35, 0.010, 0.10, 0.50, 0.40),
        (0.2, 0.001, 0.35, 0.010, 0.10, 0.35, 0.25),
        (0.2, 0.001, 0.50, 0.015, 0.15, 0.35, 0.20),
        (0.2, 0.001, 0.50, 0.020, 0.20, 0.25, 0.15),
        (0.35, 0.001, 0.50, 0.015, 0.20, 0.25, 0.15),
        (0.35, 0.002, 0.50, 0.020, 0.25, 0.25, 0.10),
        (0.35, 0.002, 0.75, 0.020, 0.25, 0.20, 0.10),
        (0.50, 0.002, 0.75, 0.020, 0.30, 0.20, 0.08),
    )
]


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    return _hard._json_default(value)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        return {"incumbent": 0.0, "blend_85_15": 1.0, "autoresearch_55_45": 0.0}
    return {key: max(0.0, float(value)) / total for key, value in weights.items()}


def _base_blend_weights() -> dict[str, float]:
    return {"incumbent": 0.0, "blend_85_15": 1.0, "autoresearch_55_45": 0.0}


def _signal_strength(
    *, confidence: float, max_signal_score: float, params: SoftAllocatorParams
) -> float:
    if max_signal_score < params.min_signal_score or confidence < params.min_confidence:
        return 0.0
    conf_component = (
        1.0 if params.confidence_scale <= 0.0 else min(1.0, confidence / params.confidence_scale)
    )
    score_component = (
        1.0
        if params.score_scale <= 0.0
        else min(
            1.0,
            max(0.0, max_signal_score - params.min_signal_score) / params.score_scale,
        )
    )
    return max(0.0, min(1.0, conf_component * score_component))


def _target_weights(signal: dict[str, Any], *, params: SoftAllocatorParams) -> dict[str, float]:
    incumbent_score = max(0.0, float(signal["incumbent_score"]))
    autoresearch_score = max(0.0, float(signal["autoresearch_score"]))
    total_score = incumbent_score + autoresearch_score
    strength = _signal_strength(
        confidence=float(signal["confidence"]),
        max_signal_score=float(signal["max_signal_score"]),
        params=params,
    )
    if total_score <= 0.0 or strength <= 0.0:
        return _base_blend_weights()

    active_mass = strength * (1.0 - float(params.blend_floor))
    incumbent_share = incumbent_score / total_score
    autoresearch_share = autoresearch_score / total_score
    target = {
        "incumbent": active_mass * incumbent_share,
        "blend_85_15": 1.0 - active_mass,
        "autoresearch_55_45": active_mass * autoresearch_share,
    }
    return _normalize(target)


def _apply_smoothing(
    *,
    previous: dict[str, float],
    target: dict[str, float],
    params: SoftAllocatorParams,
) -> tuple[dict[str, float], float]:
    alpha = max(0.0, min(1.0, float(params.alpha)))
    blended = {
        key: float(previous[key]) + alpha * (float(target[key]) - float(previous[key]))
        for key in previous
    }
    turnover = 0.5 * sum(abs(float(blended[key]) - float(previous[key])) for key in previous)
    max_turnover = max(0.0, float(params.max_daily_turnover))
    if max_turnover > 0.0 and turnover > max_turnover:
        scale = max_turnover / turnover
        blended = {
            key: float(previous[key]) + (float(blended[key]) - float(previous[key])) * scale
            for key in previous
        }
        turnover = max_turnover
    normalized = _normalize(blended)
    actual_turnover = 0.5 * sum(
        abs(float(normalized[key]) - float(previous[key])) for key in previous
    )
    return normalized, float(actual_turnover)


def _dominant_state(weights: dict[str, float]) -> str:
    return max(weights.items(), key=lambda item: (float(item[1]), item[0]))[0]


def _run_soft_allocator(*, panel: pd.DataFrame, params: SoftAllocatorParams) -> dict[str, Any]:
    rows = panel.sort_values("date").reset_index(drop=True)
    previous_weights = _base_blend_weights()
    output_rows: list[dict[str, Any]] = []
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
        target = _target_weights(signal, params=params)
        weights, turnover = _apply_smoothing(
            previous=previous_weights, target=target, params=params
        )
        portfolio_return = (
            float(weights["incumbent"]) * float(item.incumbent)
            + float(weights["blend_85_15"]) * float(item.blend_85_15)
            + float(weights["autoresearch_55_45"]) * float(item.autoresearch_55_45)
        )
        output_rows.append(
            {
                "date": item.date,
                "split_group": item.split_group,
                "state": _dominant_state(weights),
                "raw_target_state": signal["favored_group"],
                "confidence": float(signal["confidence"]),
                "incumbent_score": float(signal["incumbent_score"]),
                "autoresearch_score": float(signal["autoresearch_score"]),
                "max_signal_score": float(signal["max_signal_score"]),
                "active_rule_count": len(signal["active_rules"]),
                "turnover": float(turnover),
                "return": float(portfolio_return),
                "weights": weights,
                "target_weights": target,
                "effective_incumbent_exposure": float(
                    weights["incumbent"] + 0.85 * weights["blend_85_15"]
                ),
                "effective_autoresearch_exposure": float(
                    weights["autoresearch_55_45"] + 0.15 * weights["blend_85_15"]
                ),
            }
        )
        previous_weights = weights

    state_frame = pd.DataFrame(output_rows)
    split_metrics = _hard._metrics_by_split(state_frame, "return")
    train_val_mask = state_frame["split_group"].isin(["train", "val"])
    train_val_metrics = _hard._compute_metrics_from_returns(
        state_frame.loc[train_val_mask, "return"].astype(float).tolist()
    )
    turnover_fraction = (
        float(state_frame.loc[train_val_mask, "turnover"].mean()) if train_val_mask.any() else 0.0
    )
    objective = _hard._objective(train_val_metrics, turnover_fraction=turnover_fraction)
    return {
        "state_frame": state_frame,
        "split_metrics": split_metrics,
        "validation_objective": float(objective),
        "turnover_fraction_train_val": float(turnover_fraction),
    }


def _weight_summary(state_frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name in ("train", "val", "oos", "all"):
        sample = (
            state_frame
            if split_name == "all"
            else state_frame.loc[state_frame["split_group"].eq(split_name)]
        )
        if sample.empty:
            out[split_name] = {
                "days": 0,
                "avg_turnover": 0.0,
                "avg_weights": {"incumbent": 0.0, "blend_85_15": 0.0, "autoresearch_55_45": 0.0},
                "dominant_counts": {},
            }
            continue
        weight_columns = pd.DataFrame(sample["weights"].tolist())
        avg_weights = {key: float(weight_columns[key].mean()) for key in weight_columns.columns}
        dominant_counts = sample["state"].value_counts().to_dict()
        out[split_name] = {
            "days": len(sample),
            "avg_turnover": float(sample["turnover"].mean()),
            "avg_weights": avg_weights,
            "dominant_counts": {str(key): int(value) for key, value in dominant_counts.items()},
        }
    return out


def _build_markdown(payload: dict[str, Any]) -> str:
    params = dict(payload.get("selected_params") or {})
    current = dict(payload.get("current_state") or {})
    split_metrics = dict(payload.get("split_metrics") or {})
    benchmark_metrics = dict(payload.get("benchmark_metrics") or {})
    weight_summary = dict(payload.get("weight_summary") or {})
    memory = dict(payload.get("memory_summary") or {})

    def metric_line(label: str, metrics: dict[str, Any]) -> str:
        return (
            f"- {label}: return `{float(metrics.get('total_return') or 0.0):.4%}` | "
            f"sharpe `{float(metrics.get('sharpe') or 0.0):.4f}` | "
            f"sortino `{float(metrics.get('sortino') or 0.0):.4f}` | "
            f"calmar `{float(metrics.get('calmar') or 0.0):.4f}` | "
            f"max_dd `{float(metrics.get('max_drawdown') or 0.0):.4%}`"
        )

    lines = [
        "# Soft Three-Way Market Regime Allocator",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- peak_rss_mib: `{float(memory.get('peak_rss_mib') or 0.0):.2f}`",
        f"- memory_log: `{memory.get('rss_log_path')}`",
        "",
        "## Selected Params",
        f"- min_confidence: `{params.get('min_confidence')}`",
        f"- min_signal_score: `{params.get('min_signal_score')}`",
        f"- confidence_scale: `{params.get('confidence_scale')}`",
        f"- score_scale: `{params.get('score_scale')}`",
        f"- blend_floor: `{params.get('blend_floor')}`",
        f"- alpha: `{params.get('alpha')}`",
        f"- max_daily_turnover: `{params.get('max_daily_turnover')}`",
        f"- validation_objective: `{float(payload.get('validation_objective') or 0.0):.6f}`",
        f"- turnover_fraction_train_val: `{float(payload.get('turnover_fraction_train_val') or 0.0):.6f}`",
        "",
        "## Current State",
        f"- as_of: `{current.get('date')}`",
        f"- dominant_state: `{current.get('state')}`",
        f"- confidence: `{float(current.get('confidence') or 0.0):.4f}`",
        f"- incumbent_score: `{float(current.get('incumbent_score') or 0.0):.6f}`",
        f"- autoresearch_score: `{float(current.get('autoresearch_score') or 0.0):.6f}`",
        f"- weights: `{json.dumps(current.get('weights') or {}, sort_keys=True)}`",
        f"- effective_incumbent_exposure: `{float(current.get('effective_incumbent_exposure') or 0.0):.6f}`",
        f"- effective_autoresearch_exposure: `{float(current.get('effective_autoresearch_exposure') or 0.0):.6f}`",
        "",
        "## Allocator Metrics",
    ]
    for split_name in ("train", "val", "oos"):
        lines.append(metric_line(split_name, dict(split_metrics.get(split_name) or {})))

    lines.extend(["", "## Benchmark OOS Comparison"])
    for label, title in (
        ("incumbent", "incumbent"),
        ("blend_85_15", "blend_85_15"),
        ("autoresearch_55_45", "55/45"),
        ("allocator", "soft_allocator"),
    ):
        lines.append(
            metric_line(title, dict((benchmark_metrics.get(label) or {}).get("oos") or {}))
        )

    lines.extend(["", "## Weight Summary"])
    for split_name in ("train", "val", "oos", "all"):
        item = dict(weight_summary.get(split_name) or {})
        lines.append(
            f"- {split_name}: days `{int(item.get('days') or 0)}` | avg_turnover `{float(item.get('avg_turnover') or 0.0):.6f}` | avg_weights `{json.dumps(item.get('avg_weights') or {}, sort_keys=True)}` | dominant_counts `{json.dumps(item.get('dominant_counts') or {}, sort_keys=True)}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- target weights are score-driven but continuous rather than hard-switched",
            "- blend_85_15 acts as a stabilizer when confidence or signal strength is weak",
            "- turnover is smoothed with an alpha filter and an explicit max daily turnover cap",
            "- portfolio groups still move as sets; no inner sleeve changes occur",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def run_soft_three_way_market_regime_allocator(
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
        run_name="soft_three_way_market_regime_allocator",
        output_dir=output_dir,
        input_path=str(incumbent_path),
        rss_log_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "soft_three_way_market_regime_allocator_rss_latest.jsonl",
        summary_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "soft_three_way_market_regime_allocator_memory_latest.json",
        budget_bytes=hard_rss_bytes,
        soft_limit_bytes=soft_rss_bytes,
        hard_limit_bytes=hard_rss_bytes,
    )
    payload: dict[str, Any] | None = None
    status = "ok"
    error: str | None = None
    try:
        memory_guard.sample(event="soft_three_way_market_regime_allocator_start", context={})
        incumbent_frame = _hard._load_candidate_frame(
            label="incumbent", path=incumbent_path
        ).rename(columns={"return": "incumbent"})
        blend_frame = _hard._load_candidate_frame(label="blend_85_15", path=blend_path).rename(
            columns={"return": "blend_85_15"}
        )
        autoresearch_frame = _hard._load_candidate_frame(
            label="autoresearch_55_45", path=autoresearch_path
        ).rename(columns={"return": "autoresearch_55_45"})
        panel = (
            incumbent_frame[["date", "split_group", "incumbent"]]
            .merge(blend_frame[["date", "blend_85_15"]], on="date", how="inner")
            .merge(autoresearch_frame[["date", "autoresearch_55_45"]], on="date", how="inner")
            .sort_values("date")
            .reset_index(drop=True)
        )
        memory_guard.checkpoint("soft_three_way_returns_loaded", {"rows": len(panel)})

        market_payload = _hard._load_json(market_judgement_path)
        selected_rules = list(market_payload.get("selected_rules") or [])
        feature_frame, coverage_summary = _hard._build_market_feature_frame(
            incumbent_path=incumbent_path,
            autoresearch_path=autoresearch_path,
        )
        signal_rows = [
            _hard._signal_from_row(row, selected_rules=selected_rules)
            for _, row in feature_frame.iterrows()
        ]
        signal_frame = pd.DataFrame(signal_rows)
        merged = panel.merge(signal_frame, on=["date", "split_group"], how="inner")
        memory_guard.checkpoint(
            "soft_three_way_signals_loaded",
            {"signal_rows": len(merged), "selected_rule_count": len(selected_rules)},
        )

        best: dict[str, Any] | None = None
        for params in PARAM_GRID:
            result = _run_soft_allocator(panel=merged, params=params)
            candidate = {"params": params, **result}
            if best is None or float(candidate["validation_objective"]) > float(
                best["validation_objective"]
            ):
                best = candidate
            memory_guard.checkpoint(
                "soft_three_way_param_evaluated",
                {
                    "params": asdict(params),
                    "validation_objective": float(result["validation_objective"]),
                    "turnover_fraction_train_val": float(result["turnover_fraction_train_val"]),
                },
            )
        if best is None:
            raise RuntimeError("soft three-way allocator search produced no result")

        state_frame = best["state_frame"]
        benchmark_metrics = _hard._compare_benchmarks(merged, state_frame)
        current_state = state_frame.sort_values("date").iloc[-1].to_dict()
        payload = {
            "artifact_kind": "portfolio_soft_three_way_market_regime_allocator",
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now_iso(),
            "selection_basis": "train_val_soft_three_way_market_regime_switching",
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
            "weight_summary": _weight_summary(state_frame),
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
                    "target_weights": {
                        key: float(val) for key, val in dict(row["target_weights"]).items()
                    },
                    "effective_incumbent_exposure": float(row["effective_incumbent_exposure"]),
                    "effective_autoresearch_exposure": float(
                        row["effective_autoresearch_exposure"]
                    ),
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
            event="soft_three_way_market_regime_allocator_finish",
            context={"status": status, "error": error},
        )
        memory_summary = memory_guard.finalize(status=status, error=error, context={})
        memory_guard.release()
        if payload is not None:
            payload["memory_summary"] = memory_summary

    if payload is None:
        raise RuntimeError("soft three-way market regime allocator did not produce payload")

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = output_dir / f"soft_three_way_market_regime_allocator_{timestamp}.json"
    out_md = output_dir / f"soft_three_way_market_regime_allocator_{timestamp}.md"
    latest_json = output_dir / "soft_three_way_market_regime_allocator_latest.json"
    latest_md = output_dir / "soft_three_way_market_regime_allocator_latest.md"
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
        "--autoresearch-path", type=Path, default=_hard._resolve_autoresearch_default_path()
    )
    parser.add_argument("--market-judgement-path", type=Path, default=DEFAULT_MARKET_JUDGEMENT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--soft-rss-bytes", type=int, default=DEFAULT_SOFT_RSS_BYTES)
    parser.add_argument("--hard-rss-bytes", type=int, default=DEFAULT_HARD_RSS_BYTES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_soft_three_way_market_regime_allocator(
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
