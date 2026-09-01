"""Causal overlay allocator on top of the current one-shot optimized backbone."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from lumina_quant.eval.exact_window_runtime import RSSLimitExceeded
from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    PORTFOLIO_CURRENT_OPTIMIZATION,
    PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
    resolve_current_optimization_path,
    resolve_incumbent_bundle_path,
    split_windows,
)

DEFAULT_INPUT = PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE
DEFAULT_BACKBONE = PORTFOLIO_CURRENT_OPTIMIZATION
DEFAULT_OUTPUT_DIR = FOLLOWUP_ROOT / "portfolio_overlay_current"
COMPARISON_INPUT = FOLLOWUP_ROOT / "portfolio_dynamic_comparison_latest.json"

_helper_spec = importlib.util.spec_from_file_location(
    "run_causal_dynamic_portfolio",
    Path(__file__).resolve().parent / "run_causal_dynamic_portfolio.py",
)
if _helper_spec is None or _helper_spec.loader is None:
    raise RuntimeError("Failed to load run_causal_dynamic_portfolio helpers")
_helper = importlib.util.module_from_spec(_helper_spec)
sys.modules[_helper_spec.name] = _helper
_helper_spec.loader.exec_module(_helper)


@dataclass(slots=True)
class OverlayParams:
    lookback_days: int
    rebalance_days: int
    min_trailing_sharpe: float
    min_trailing_return: float
    max_trailing_drawdown: float
    overlay_strength: float
    correlation_penalty: float
    regime_strength: float
    cash_buffer: float


def _load_backbone_weights(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("weights") or [])
    if not rows:
        raise RuntimeError(f"no weights found in {path}")
    return {
        str(row.get("candidate_id") or row.get("name")): float(row.get("weight", 0.0))
        for row in rows
        if row.get("candidate_id") or row.get("name")
    }


def _positive_corr_penalty(
    history: dict[str, np.ndarray], active_ids: list[str], cid: str
) -> float:
    arr_i = np.asarray(history.get(cid, []), dtype=float)
    penalties: list[float] = []
    for other in active_ids:
        if other == cid:
            continue
        arr_j = np.asarray(history.get(other, []), dtype=float)
        n = min(arr_i.size, arr_j.size)
        if n < 3:
            continue
        x = arr_i[-n:]
        y = arr_j[-n:]
        sx = float(np.std(x, ddof=1))
        sy = float(np.std(y, ddof=1))
        if sx <= 1e-12 or sy <= 1e-12:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(corr):
            penalties.append(max(0.0, corr))
    return float(np.mean(penalties)) if penalties else 0.0


def _performance_scale(
    *,
    trailing_sharpe: float,
    trailing_return: float,
    trailing_drawdown: float,
    min_trailing_sharpe: float,
    min_trailing_return: float,
    max_trailing_drawdown: float,
    overlay_strength: float,
) -> float:
    if trailing_drawdown > max_trailing_drawdown:
        return 0.0
    if trailing_sharpe < min_trailing_sharpe:
        return 0.0
    if trailing_return < min_trailing_return:
        return 0.0
    sharpe_term = min(1.0, max(0.0, trailing_sharpe / 3.0))
    return_term = min(1.0, max(0.0, trailing_return / 0.08))
    dd_term = max(0.0, 1.0 - (trailing_drawdown / max(max_trailing_drawdown, 1e-6)))
    blend = (0.45 * sharpe_term) + (0.35 * return_term) + (0.20 * dd_term)
    return 0.5 + (overlay_strength * blend)


def run_causal_overlay_allocator(
    rows: list[dict[str, Any]],
    backbone_weights: dict[str, float],
    params: OverlayParams,
    *,
    regime_features: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ordered_days, matrix, meta = _helper._build_daily_panel(rows)
    regime = regime_features if regime_features is not None else _helper._load_regime_features(rows)
    ids = [cid for cid in matrix if cid in backbone_weights]
    allocations: list[dict[str, Any]] = []
    current_weights: dict[str, float] = {}
    split_returns: dict[str, list[float]] = {"train": [], "val": [], "oos": []}
    all_returns: list[float] = []

    for idx, day_key in enumerate(ordered_days):
        split = _helper._split_index(day_key)
        rebalance_now = idx > 0 and (
            not current_weights or (idx % max(1, params.rebalance_days) == 0)
        )
        if rebalance_now:
            prior_day = ordered_days[idx - 1]
            regime_row = dict(regime.get(prior_day) or {})
            history_window = {
                cid: matrix[cid][max(0, idx - params.lookback_days) : idx] for cid in ids
            }
            raw: dict[str, float] = {}
            diagnostics: dict[str, dict[str, float]] = {}
            active_ids: list[str] = []
            for cid in ids:
                hist = np.asarray(history_window.get(cid, []), dtype=float)
                metric = _helper._metrics(hist)
                trailing_return = float(metric["total_return"])
                trailing_sharpe = float(metric["sharpe"])
                trailing_drawdown = _helper._max_drawdown(hist)
                performance = _performance_scale(
                    trailing_sharpe=trailing_sharpe,
                    trailing_return=trailing_return,
                    trailing_drawdown=trailing_drawdown,
                    min_trailing_sharpe=params.min_trailing_sharpe,
                    min_trailing_return=params.min_trailing_return,
                    max_trailing_drawdown=params.max_trailing_drawdown,
                    overlay_strength=params.overlay_strength,
                )
                regime_mult = _helper._regime_multiplier(
                    meta.get(cid) or {},
                    regime_row,
                    previous_active=bool(current_weights.get(cid, 0.0) > 0.0),
                    strength=params.regime_strength,
                )
                diagnostics[cid] = {
                    "trailing_return": trailing_return,
                    "trailing_sharpe": trailing_sharpe,
                    "trailing_drawdown": trailing_drawdown,
                    "performance_scale": performance,
                    "regime_multiplier": regime_mult,
                }
                if performance <= 0.0 or regime_mult <= 0.0:
                    continue
                active_ids.append(cid)
                raw[cid] = float(backbone_weights[cid] * performance * regime_mult)

            adjusted: dict[str, float] = {}
            for cid in active_ids:
                corr_pen = _positive_corr_penalty(history_window, active_ids, cid)
                diagnostics[cid]["correlation_penalty"] = corr_pen
                adjusted[cid] = float(raw[cid] / (1.0 + (params.correlation_penalty * corr_pen)))

            total = sum(adjusted.values())
            weights: dict[str, float] = {}
            if total > 1e-12:
                for cid, raw_weight in adjusted.items():
                    weights[cid] = min(
                        backbone_weights[cid],
                        raw_weight / total * max(0.0, 1.0 - params.cash_buffer),
                    )
            current_weights = weights
            cash_weight = max(params.cash_buffer, 1.0 - sum(current_weights.values()))
            allocations.append(
                {
                    "date": day_key,
                    "weights": dict(current_weights),
                    "cash_weight": cash_weight,
                    "diagnostics": diagnostics,
                    "regime_row": regime_row,
                }
            )
        elif idx == 0:
            allocations.append(
                {
                    "date": day_key,
                    "weights": {},
                    "cash_weight": 1.0,
                    "diagnostics": {},
                    "regime_row": {},
                }
            )

        day_return = sum(
            float(current_weights.get(cid, 0.0)) * float(matrix[cid][idx]) for cid in ids
        )
        all_returns.append(day_return)
        split_returns[split].append(day_return)

    return {
        "dates": ordered_days,
        "daily_returns": all_returns,
        "allocations": allocations,
        "split_metrics": {
            split: _helper._metrics(np.asarray(values, dtype=float))
            for split, values in split_returns.items()
        },
        "all_metrics": _helper._metrics(np.asarray(all_returns, dtype=float)),
        "meta": meta,
    }


def _portfolio_return_streams_from_daily(
    dates: list[str],
    daily_returns: list[float],
) -> dict[str, list[dict[str, Any]]]:
    streams: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
    for day_key, day_return in zip(dates, daily_returns, strict=True):
        split = _helper._split_index(str(day_key))
        streams[split].append(
            {"t": f"{day_key}T00:00:00Z", "v": _helper._safe_float(day_return, 0.0)}
        )
    return streams


def _objective(metrics: dict[str, float], *, cash_fraction: float) -> float:
    return float(
        (1.0 * _helper._safe_float(metrics.get("sharpe"), 0.0))
        + (0.40 * _helper._safe_float(metrics.get("sortino"), 0.0))
        + (0.10 * _helper._safe_float(metrics.get("calmar"), 0.0))
        + (9.0 * _helper._safe_float(metrics.get("total_return"), 0.0))
        - (4.0 * _helper._safe_float(metrics.get("max_drawdown"), 0.0))
        - (0.50 * _helper._safe_float(metrics.get("volatility"), 0.0))
        - (0.20 * cash_fraction)
    )


def search_overlay_allocator(
    rows: list[dict[str, Any]],
    backbone_weights: dict[str, float],
    *,
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    regime_features = _helper._load_regime_features(rows)
    grid = [
        OverlayParams(*combo)
        for combo in itertools.product(
            [5, 10, 20],  # lookback
            [1, 3],  # rebalance
            [0.0, 0.25],  # min sharpe
            [0.0],  # min return
            [0.10, 0.15],  # max dd
            [0.5, 1.0, 1.5],  # overlay strength
            [0.0, 0.5, 1.0],  # corr penalty
            [0.5, 1.0, 1.5],  # regime strength
            [0.0, 0.1, 0.2],  # cash buffer
        )
    ]
    best: dict[str, Any] | None = None
    total_candidates = len(grid)
    for idx, params in enumerate(grid, start=1):
        result = run_causal_overlay_allocator(
            rows, backbone_weights, params, regime_features=regime_features
        )
        val_metrics = dict((result.get("split_metrics") or {}).get("val") or {})
        cash_fraction = _helper._mean_cash_fraction(
            list(result.get("allocations") or []), split="val"
        )
        objective = _objective(val_metrics, cash_fraction=cash_fraction)
        candidate = {"params": asdict(params), "objective": objective, "result": result}
        if best is None or objective > float(best["objective"]):
            best = candidate
        if progress_callback is not None:
            progress_callback(
                "overlay_candidate_evaluated",
                {
                    "candidate_index": idx,
                    "candidate_count": total_candidates,
                    "objective": objective,
                    "val_total_return": _helper._safe_float(val_metrics.get("total_return"), 0.0),
                    "val_sharpe": _helper._safe_float(val_metrics.get("sharpe"), 0.0),
                    "params": dict(candidate["params"]),
                },
            )
    if best is None:
        raise RuntimeError("overlay search produced no result")
    return best


def _final_allocation_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    latest = list(result.get("allocations") or [])[-1]
    meta = dict(result.get("meta") or {})
    rows: list[dict[str, Any]] = []
    for cid, weight in sorted(
        dict(latest.get("weights") or {}).items(), key=lambda item: item[1], reverse=True
    ):
        item = dict(meta.get(cid) or {})
        rows.append(
            {
                "candidate_id": cid,
                "name": item.get("name"),
                "strategy_class": item.get("strategy_class"),
                "timeframe": item.get("timeframe"),
                "weight": float(weight),
            }
        )
    return rows


def write_overlay_report(
    *,
    input_path: Path = DEFAULT_INPUT,
    backbone_path: Path = DEFAULT_BACKBONE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    resolved_input = resolve_incumbent_bundle_path(input_path)
    resolved_backbone = resolve_current_optimization_path(backbone_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name="causal_overlay_portfolio",
        output_dir=output_dir,
        input_path=resolved_input,
        metadata={"backbone_path": str(resolved_backbone)},
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    status = "completed"
    error: str | None = None
    try:
        memory_guard.sample(
            event="overlay_start",
            context={
                "requested_input_path": str(Path(input_path).resolve()),
                "resolved_input_path": str(resolved_input),
                "backbone_path": str(resolved_backbone),
            },
        )
        rows = _helper._load_candidates(resolved_input)
        backbone_weights = _load_backbone_weights(resolved_backbone)
        memory_guard.checkpoint(
            "overlay_candidates_loaded",
            {"candidate_count": len(rows), "backbone_count": len(backbone_weights)},
        )
        best = search_overlay_allocator(
            rows,
            backbone_weights,
            progress_callback=memory_guard.checkpoint,
        )
        result = dict(best["result"])
    except RSSLimitExceeded as exc:
        status = "aborted_rss_guard"
        error = str(exc)
        raise
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(event="overlay_finish", context={"status": status, "error": error})
        memory_summary = memory_guard.finalize(
            status=status,
            error=error,
            context={
                "resolved_input_path": str(resolved_input),
                "backbone_path": str(resolved_backbone),
            },
        )
        memory_guard.release()
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "causal_overlay_portfolio",
        "schema_version": "1.0",
        "input_path": str(resolved_input),
        "requested_input_path": str(Path(input_path).resolve()),
        "backbone_path": str(resolved_backbone),
        "selection_basis": "validation_only_overlay_search_on_current_one_shot_backbone",
        "objective_profile": "balanced_multi_metric_with_backbone_overlay",
        "split_windows": split_windows(),
        "memory_policy": memory_policy_payload(
            budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        ),
        "memory_summary": memory_summary,
        "best_params": dict(best["params"]),
        "validation_objective": float(best["objective"]),
        "split_metrics": dict(result.get("split_metrics") or {}),
        "all_metrics": dict(result.get("all_metrics") or {}),
        "allocation_count": len(list(result.get("allocations") or [])),
        "final_allocation": _final_allocation_rows(result),
        "dates": list(result.get("dates") or []),
        "daily_returns": list(result.get("daily_returns") or []),
        "portfolio_return_streams": _portfolio_return_streams_from_daily(
            list(result.get("dates") or []),
            list(result.get("daily_returns") or []),
        ),
        "allocations": list(result.get("allocations") or []),
        "universe_scope": "current_one_shot_backbone",
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"causal_overlay_portfolio_{stamp}.json"
    latest_path = output_dir / "causal_overlay_portfolio_latest.json"
    md_path = output_dir / f"causal_overlay_portfolio_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Causal Overlay Portfolio",
        "",
        f"- input_path: `{payload['input_path']}`",
        f"- backbone_path: `{payload['backbone_path']}`",
        f"- selection_basis: `{payload['selection_basis']}`",
        f"- objective_profile: `{payload['objective_profile']}`",
        f"- validation_objective: `{payload['validation_objective']:.6f}`",
        f"- oos_start: `{dict(payload.get('split_windows') or {}).get('oos_start')}`",
        f"- memory_log: `{dict(payload.get('memory_summary') or {}).get('rss_log_path')}`",
        "",
        "## Best params",
        "",
        "```json",
        json.dumps(payload["best_params"], indent=2, sort_keys=True),
        "```",
        "",
        "## Split metrics",
        "",
        f"- train: {json.dumps(payload['split_metrics'].get('train') or {}, sort_keys=True)}",
        f"- val: {json.dumps(payload['split_metrics'].get('val') or {}, sort_keys=True)}",
        f"- oos: {json.dumps(payload['split_metrics'].get('oos') or {}, sort_keys=True)}",
        "",
        "## Final allocation",
        "",
    ]
    for row in payload["final_allocation"]:
        lines.append(
            f"- `{row.get('name')}` | strategy={row.get('strategy_class')} | tf={row.get('timeframe')} | weight={float(row.get('weight', 0.0)):.2%}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "latest_path": str(latest_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def write_overlay_comparison(overlay_payload: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(COMPARISON_INPUT.read_text(encoding="utf-8"))
    scope = list(payload.get("comparison_scope") or [])
    current_entry = _helper._maybe_current_one_shot_comparison_entry()
    if current_entry is not None:
        payload["current_one_shot_optimized"] = current_entry
    if "current_one_shot_optimized" not in scope:
        scope.append("current_one_shot_optimized")
    if "causal_overlay_portfolio" not in scope:
        scope.append("causal_overlay_portfolio")
    payload["comparison_scope"] = scope
    overlay_val = dict((overlay_payload.get("split_metrics") or {}).get("val") or {})
    overlay_oos = dict((overlay_payload.get("split_metrics") or {}).get("oos") or {})
    payload["causal_overlay_portfolio"] = {
        "path": str((DEFAULT_OUTPUT_DIR / "causal_overlay_portfolio_latest.json").resolve()),
        "val": overlay_val,
        "oos": overlay_oos,
        "weights": list(overlay_payload.get("final_allocation") or []),
        "best_params": dict(overlay_payload.get("best_params") or {}),
    }
    payload["deltas"]["overlay_vs_current_one_shot_oos_return"] = _helper._safe_float(
        overlay_oos.get("total_return"), 0.0
    ) - _helper._safe_float(payload["current_one_shot_optimized"]["oos"].get("total_return"), 0.0)
    payload["deltas"]["overlay_vs_current_one_shot_oos_sharpe"] = _helper._safe_float(
        overlay_oos.get("sharpe"), 0.0
    ) - _helper._safe_float(payload["current_one_shot_optimized"]["oos"].get("sharpe"), 0.0)
    out_json = FOLLOWUP_ROOT / "portfolio_overlay_comparison_latest.json"
    out_md = FOLLOWUP_ROOT / "portfolio_overlay_comparison_latest.md"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Overlay Portfolio Comparison",
        "",
        f"- overlay_vs_current_one_shot_oos_return: {payload['deltas']['overlay_vs_current_one_shot_oos_return']:.4%}",
        f"- overlay_vs_current_one_shot_oos_sharpe: {payload['deltas']['overlay_vs_current_one_shot_oos_sharpe']:.3f}",
        "",
        "## Overlay OOS metrics",
        "",
        json.dumps(overlay_oos, sort_keys=True),
        "",
        "## Overlay final allocation",
        "",
    ]
    for row in list(overlay_payload.get("final_allocation") or []):
        lines.append(
            f"- `{row.get('name')}` | strategy={row.get('strategy_class')} | tf={row.get('timeframe')} | weight={float(row.get('weight', 0.0)):.2%}"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json_path": str(out_json.resolve()), "md_path": str(out_md.resolve())}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a causal overlay allocator on the current one-shot optimized backbone."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--backbone", default=str(DEFAULT_BACKBONE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = write_overlay_report(
        input_path=Path(args.input).resolve(),
        backbone_path=Path(args.backbone).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
    comparison = write_overlay_comparison(report["payload"])
    print(report["latest_path"])
    print(report["md_path"])
    print(comparison["json_path"])
    print(comparison["md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
