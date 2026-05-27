"""Retune HYBRID over only live-implementable portfolio modes.

This is the deployment-shaped counterpart to the broader full-universe research
ranking: every sleeve entering this HYBRID is itself a committed live portfolio
mode that `ArtifactPortfolioModeStrategy` can expand.  The dynamic retune is
reported separately from the saved-final-allocation live mode so that live
promotion does not accidentally rely on an unrealized backtest-only allocator.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUP_ROOT = (
    REPO_ROOT
    / "var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped"
)
OUTPUT_DIR = GROUP_ROOT / "live_portfolio_hybrid_retune_20260426"
LIVE_SELECTION_DIR = GROUP_ROOT / "live_implementable_selection_20260426"
MDD_CAP = 0.25
LIVE_MODE_WEIGHTS: dict[str, tuple[dict[str, float], float]] = {
    "core_mode": ({"soft_three_way_regime": 1.0}, 0.0),
    "balanced_overlay_mode": ({"balanced_overlay_80_20": 1.0}, 0.0),
    "defensive_overlay_mode": ({"soft_three_way_regime": 0.7, "pair_tactical_mode": 0.3}, 0.0),
    "aggressive_realized_mode": ({"three_way_regime": 1.0}, 0.0),
    "pair_tactical_mode": ({"pair_tactical_mode": 1.0}, 0.0),
    "production_guarded_state_vwap_pair_mode": (
        {"production_guarded_portfolio": 0.4, "state_vwap_pair": 0.25},
        0.35,
    ),
    "strict_autoresearch_practical_mode": (
        {"production_guarded_portfolio": 0.8, "strict_autoresearch_1x": 0.2},
        0.0,
    ),
}
CSV_FIELDS = (
    "rank",
    "name",
    "kind",
    "live_mode",
    "live_deployable",
    "selection_score",
    "val_scaled_score",
    "train_scaled_score",
    "oos_scaled_score_report_only",
    "train_total_return",
    "train_sharpe",
    "train_max_drawdown",
    "val_total_return",
    "val_sharpe",
    "val_max_drawdown",
    "oos_total_return",
    "oos_sharpe",
    "oos_max_drawdown",
    "cash_weight",
    "caveat",
)

_MAT_SPEC = importlib.util.spec_from_file_location(
    "materialize_legacy_metric_hybrid",
    Path(__file__).resolve().parent / "materialize_legacy_metric_hybrid.py",
)
if _MAT_SPEC is None or _MAT_SPEC.loader is None:
    raise RuntimeError("failed to import materialize_legacy_metric_hybrid helpers")
_MAT = importlib.util.module_from_spec(_MAT_SPEC)
sys.modules[_MAT_SPEC.name] = _MAT
_MAT_SPEC.loader.exec_module(_MAT)
_HYBRID = _MAT._HYBRID


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return _MAT._safe_float(value, default)


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    return _MAT._metric_value(metrics, key)


def _score_from_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    return _MAT._score_from_metrics(metrics)


def _metrics_table(title: str, rows: list[dict[str, Any]], *, limit: int = 20) -> list[str]:
    return _MAT._metrics_table(title, rows, limit=limit)


def _live_mode_row(
    *,
    mode_name: str,
    weights: dict[str, float],
    cash_weight: float,
    rows_by_name: dict[str, dict[str, Any]],
    split_config: Any,
) -> dict[str, Any]:
    replay = _MAT._frozen_final_allocation_replay(
        name=mode_name,
        weights=weights,
        rows_by_name=rows_by_name,
        split_config=split_config,
        cash_weight=cash_weight,
    )
    streams = _HYBRID._portfolio_return_streams_from_daily(
        list(replay.get("dates") or []),
        [_safe_float(value, 0.0) for value in list(replay.get("daily_returns") or [])],
        split_config=split_config,
    )
    split_metrics = _HYBRID._split_metrics_from_streams(streams)
    return {
        "candidate_id": mode_name,
        "name": mode_name,
        "strategy_class": "ArtifactPortfolioModeStrategy",
        "strategy_timeframe": "1d",
        "family": "live_portfolio_mode",
        "symbols": [],
        "params": {"portfolio_mode": mode_name},
        "return_streams": streams,
        "metadata": {
            "source_payload_path": f"derived:{mode_name}",
            "cash_weight": float(cash_weight),
            "component_weights": dict(weights),
        },
        "train": dict(split_metrics.get("train") or {}),
        "val": dict(split_metrics.get("val") or {}),
        "oos": dict(split_metrics.get("oos") or {}),
    }


def _legacy_mode_weights(legacy_payload: dict[str, Any]) -> tuple[dict[str, float], float]:
    final_allocation = dict(
        dict((legacy_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
            "final_allocation"
        )
        or {}
    )
    return (
        {
            str(k): _safe_float(v, 0.0)
            for k, v in dict(final_allocation.get("weights") or {}).items()
        },
        _safe_float(final_allocation.get("cash_weight"), 0.0),
    )


def _live_mode_rows(
    *,
    rows_by_name: dict[str, dict[str, Any]],
    legacy_payload: dict[str, Any],
    split_config: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode_name, (weights, cash_weight) in LIVE_MODE_WEIGHTS.items():
        rows.append(
            _live_mode_row(
                mode_name=mode_name,
                weights=weights,
                cash_weight=cash_weight,
                rows_by_name=rows_by_name,
                split_config=split_config,
            )
        )
    legacy_weights, legacy_cash = _legacy_mode_weights(legacy_payload)
    rows.append(
        _live_mode_row(
            mode_name="legacy_no_highvol_hybrid_mode",
            weights=legacy_weights,
            cash_weight=legacy_cash,
            rows_by_name=rows_by_name,
            split_config=split_config,
        )
    )
    all_day_keys = sorted(
        set().union(*(_HYBRID._merged_daily_map(row["return_streams"]).keys() for row in rows))
    )
    return [_MAT._make_cash_row(all_day_keys, split_config=split_config), *rows]


def _grid_configs() -> list[Any]:
    configs: list[Any] = []
    for (
        variant,
        warmup_days,
        lookback_days,
        default_boost,
        sticky_default_bonus,
        switch_margin,
        score_temperature,
        min_positive_score,
        pair_weight_cap,
        diversified_weight_cap,
        disagreement_cash_scale,
    ) in itertools.product(
        ("fixed_default", "dynamic_default", "disagreement_switching"),
        (30, 60, 120),
        (7, 13, 29),
        (0.0, 0.20),
        (0.20,),
        (0.10,),
        (0.80, 1.30),
        (-0.20, 0.0),
        (0.25, 0.40),
        (0.90,),
        (0.70,),
    ):
        configs.append(
            _HYBRID.HybridOnlineConfig(
                variant=variant,
                warmup_days=warmup_days,
                warmup_ratio=0.0,
                lookback_days=lookback_days,
                default_boost=default_boost,
                sticky_default_bonus=sticky_default_bonus,
                switch_margin=switch_margin,
                score_temperature=score_temperature,
                min_positive_score=min_positive_score,
                pair_score_boost=0.0,
                disagreement_threshold=0.08,
                disagreement_cash_scale=disagreement_cash_scale,
                pair_weight_cap=pair_weight_cap,
                diversified_weight_cap=diversified_weight_cap,
                pair_pbo_penalty_scale=0.0,
                pair_sparsity_penalty_scale=0.0,
                negative_health_floor=1.0,
                mixed_health_floor=1.0,
                use_current_health_priors=False,
            )
        )
    return configs


def _selection_score(result: dict[str, Any]) -> float:
    metrics = dict(result.get("split_metrics") or {})
    score = _score_from_metrics(metrics)["selection_score"]
    train_mdd = _metric_value(dict(metrics.get("train") or {}), "max_drawdown")
    val_mdd = _metric_value(dict(metrics.get("val") or {}), "max_drawdown")
    if train_mdd > MDD_CAP or val_mdd > MDD_CAP:
        score -= 1000.0
    return float(score)


def _retune_hybrid(rows: list[dict[str, Any]], *, split_config: Any) -> tuple[dict[str, Any], Any]:
    best: tuple[float, dict[str, Any], Any] | None = None
    for config in _grid_configs():
        result = _HYBRID.run_hybrid_online_allocator(
            rows,
            config=config,
            refreshed_health_metrics=None,
            split_config=split_config,
        )
        score = _selection_score(result)
        if best is None or score > best[0]:
            best = (score, result, config)
    if best is None:
        raise RuntimeError("live portfolio HYBRID retune produced no candidates")
    _, result, config = best
    result["allocation_summary"] = _MAT._allocation_summary(list(result.get("allocations") or []))
    return result, config


def _static_weight_candidates(names: list[str]) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = [{name: 1.0} for name in names]
    for left, right in itertools.combinations(names, 2):
        for left_tenths in range(1, 10):
            candidates.append({left: left_tenths / 10.0, right: 1.0 - left_tenths / 10.0})
    for first, second, third in itertools.combinations(names, 3):
        for first_tenths in range(1, 10):
            for second_tenths in range(1, 10 - first_tenths):
                third_tenths = 10 - first_tenths - second_tenths
                if third_tenths <= 0:
                    continue
                candidates.append(
                    {
                        first: first_tenths / 10.0,
                        second: second_tenths / 10.0,
                        third: third_tenths / 10.0,
                    }
                )
    return candidates


def _retune_static_final_allocation(
    rows: list[dict[str, Any]],
    *,
    split_config: Any,
) -> dict[str, Any]:
    rows_by_name = {str(row.get("name") or ""): row for row in rows}
    names = [name for name in rows_by_name if name != "risk_off_cash"]
    best: tuple[float, dict[str, Any]] | None = None
    for weights in _static_weight_candidates(names):
        replay = _MAT._frozen_final_allocation_replay(
            name="retuned_live_portfolio_hybrid_mode",
            weights=weights,
            rows_by_name=rows_by_name,
            split_config=split_config,
            cash_weight=0.0,
        )
        score = _score_from_metrics(dict(replay.get("split_metrics") or {}))["selection_score"]
        metrics = dict(replay.get("split_metrics") or {})
        train_mdd = _metric_value(dict(metrics.get("train") or {}), "max_drawdown")
        val_mdd = _metric_value(dict(metrics.get("val") or {}), "max_drawdown")
        if train_mdd > MDD_CAP or val_mdd > MDD_CAP:
            score -= 1000.0
        if best is None or score > best[0]:
            best = (float(score), replay)
    if best is None:
        raise RuntimeError("static live allocation retune produced no candidates")
    _, replay = best
    replay["final_allocation"] = {
        "date": "static_validation_primary_retune",
        "split": "deployable_static",
        "default_sleeve": max(
            dict(replay.get("weights") or {}), key=dict(replay.get("weights") or {}).get
        ),
        "weights": dict(replay.get("weights") or {}),
        "cash_weight": _safe_float(replay.get("cash_weight"), 0.0),
    }
    return replay


def _candidate_row(
    *,
    name: str,
    kind: str,
    metrics: dict[str, dict[str, Any]],
    live_mode: str,
    live_deployable: bool,
    cash_weight: float = 0.0,
    caveat: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "live_mode": live_mode,
        "live_deployable": bool(live_deployable),
        "cash_weight": float(cash_weight),
        "metrics": metrics,
        "caveat": caveat,
        **_score_from_metrics(metrics),
        **(extra or {}),
    }


def _comparison_rows(
    *,
    live_rows: list[dict[str, Any]],
    dynamic_result: dict[str, Any],
    final_replay: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in live_rows:
        if str(row.get("name") or "") == "risk_off_cash":
            mode_name = "risk_off_mode"
            live_mode = "risk_off_mode"
            live_deployable = True
        else:
            mode_name = str(row.get("name") or "")
            live_mode = mode_name
            live_deployable = True
        rows.append(
            _candidate_row(
                name=mode_name,
                kind="live_portfolio_mode_static_replay",
                live_mode=live_mode,
                live_deployable=live_deployable,
                cash_weight=_safe_float(dict(row.get("metadata") or {}).get("cash_weight"), 0.0),
                metrics={
                    "train": dict(row.get("train") or {}),
                    "val": dict(row.get("val") or {}),
                    "oos": dict(row.get("oos") or {}),
                },
            )
        )
    rows.append(
        _candidate_row(
            name="retuned_live_portfolio_hybrid_dynamic_backtest",
            kind="live_mode_sleeve_dynamic_research_path",
            live_mode="",
            live_deployable=False,
            cash_weight=_safe_float(
                dict(dynamic_result.get("final_allocation") or {}).get("cash_weight"),
                0.0,
            ),
            metrics=dict(dynamic_result.get("split_metrics") or {}),
            caveat="dynamic allocation path; live mode below uses the saved final allocation",
            extra={"final_allocation": dict(dynamic_result.get("final_allocation") or {})},
        )
    )
    rows.append(
        _candidate_row(
            name="retuned_live_portfolio_hybrid_mode",
            kind="live_hybrid_static_validation_retune",
            live_mode="retuned_live_portfolio_hybrid_mode",
            live_deployable=True,
            cash_weight=_safe_float(final_replay.get("cash_weight"), 0.0),
            metrics=dict(final_replay.get("split_metrics") or {}),
            caveat="committed live mode using the validation-primary static allocation over live portfolio sleeves",
            extra={"weights": dict(final_replay.get("weights") or {})},
        )
    )
    rows.sort(
        key=lambda row: (
            1 if row.get("live_deployable") else 0,
            _safe_float(row.get("selection_score"), 0.0),
            1 if row.get("name") == "retuned_live_portfolio_hybrid_mode" else 0,
        ),
        reverse=True,
    )
    return rows


def _metric_columns(row: dict[str, Any]) -> dict[str, float]:
    metrics = dict(row.get("metrics") or {})
    out: dict[str, float] = {}
    for split in ("train", "val", "oos"):
        split_metrics = dict(metrics.get(split) or {})
        for key in ("total_return", "sharpe", "max_drawdown"):
            out[f"{split}_{key}"] = _metric_value(split_metrics, key)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "name": row.get("name"),
                    "kind": row.get("kind"),
                    "live_mode": row.get("live_mode"),
                    "live_deployable": row.get("live_deployable"),
                    "selection_score": row.get("selection_score"),
                    "val_scaled_score": row.get("val_scaled_score"),
                    "train_scaled_score": row.get("train_scaled_score"),
                    "oos_scaled_score_report_only": row.get("oos_scaled_score_report_only"),
                    "cash_weight": row.get("cash_weight"),
                    "caveat": row.get("caveat"),
                    **_metric_columns(row),
                }
            )


def _write_reports(
    *,
    payload: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    output_dir: Path,
    live_selection_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    live_selection_dir.mkdir(parents=True, exist_ok=True)
    latest_json = output_dir / "live_portfolio_hybrid_retune_latest.json"
    latest_md = output_dir / "live_portfolio_hybrid_retune_latest.md"
    latest_csv = output_dir / "live_portfolio_hybrid_retune_candidates_20260426.csv"
    live_json = live_selection_dir / "live_implementable_selection_latest.json"
    live_md = live_selection_dir / "live_implementable_selection_latest.md"
    live_csv = live_selection_dir / "live_implementable_selection_candidates_20260426.csv"

    latest_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _write_csv(latest_csv, comparison_rows)

    artifact = payload["scenarios"]["refreshed_latest_tail"]
    final_replay = payload["live_execution_model"]["final_allocation_replay"]
    best_live = next((row for row in comparison_rows if row.get("live_deployable")), {})
    lines = [
        "# Live Portfolio HYBRID Retune",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- tuning universe: committed live portfolio modes only",
        "- selection: validation-primary; OOS report-only; cash efficiency not scored",
        (f"- best dynamic retune score: `{payload['dynamic_scores']['selection_score']:.2f}`"),
        (
            f"- deployable validation-primary static mode: `retuned_live_portfolio_hybrid_mode`, "
            f"score `{final_replay['selection_score']:.2f}`"
        ),
        (
            f"- best live-deployable after retune: `{best_live.get('name')}` "
            f"(score `{_safe_float(best_live.get('selection_score'), 0.0):.2f}`)"
        ),
        "",
        "## Final allocation",
        "",
        f"- date: `{artifact['final_allocation'].get('date')}`",
        f"- cash_weight: `{_safe_float(artifact['final_allocation'].get('cash_weight'), 0.0):.2%}`",
    ]
    for sleeve, weight in sorted(
        dict(artifact["final_allocation"].get("weights") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"- `{sleeve}`: `{_safe_float(weight, 0.0):.2%}`")
    lines.extend([""])
    lines.extend(_metrics_table("Live-implementable ranking", comparison_rows, limit=30))
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in payload["explicit_caveats"])
    latest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    live_payload = {
        "generated_at": payload["generated_at"],
        "selection_policy": payload["selection_policy"],
        "best_live_deployable": best_live,
        "ranked_candidates": comparison_rows,
        "source_retune_report": _display_path(latest_json),
    }
    live_json.write_text(
        json.dumps(live_payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    live_lines = [
        "# Live-Implementable Portfolio Selection",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        (
            f"- best_live_deployable: `{best_live.get('name')}` "
            f"(score `{_safe_float(best_live.get('selection_score'), 0.0):.2f}`)"
        ),
        "- universe: live portfolio modes plus the retuned live HYBRID final-allocation mode",
        "",
    ]
    live_lines.extend(_metrics_table("Ranked candidates", comparison_rows, limit=30))
    live_md.write_text("\n".join(live_lines) + "\n", encoding="utf-8")
    _write_csv(live_csv, comparison_rows)
    return {
        "latest_json_path": str(latest_json.resolve()),
        "latest_md_path": str(latest_md.resolve()),
        "latest_csv_path": str(latest_csv.resolve()),
        "live_json_path": str(live_json.resolve()),
        "live_md_path": str(live_md.resolve()),
        "live_csv_path": str(live_csv.resolve()),
    }


def build_live_portfolio_hybrid_retune(
    *,
    output_dir: Path = OUTPUT_DIR,
    live_selection_dir: Path = LIVE_SELECTION_DIR,
) -> dict[str, Any]:
    split_config = _HYBRID.HybridSplitConfig()
    legacy_report = _MAT.build_materialization_report()
    legacy_payload = dict(legacy_report["payload"])
    base_rows = _MAT._build_sleeve_rows(split_config=split_config)
    rows_by_name = {str(row.get("name") or ""): row for row in base_rows}
    live_rows = _live_mode_rows(
        rows_by_name=rows_by_name,
        legacy_payload=legacy_payload,
        split_config=split_config,
    )
    dynamic_result, config = _retune_hybrid(live_rows, split_config=split_config)
    final_replay = _retune_static_final_allocation(live_rows, split_config=split_config)
    dynamic_scores = _score_from_metrics(dict(dynamic_result.get("split_metrics") or {}))
    comparison_rows = _comparison_rows(
        live_rows=live_rows,
        dynamic_result=dynamic_result,
        final_replay=final_replay,
    )
    gap = {
        "train_total_return_gap": _metric_value(
            dict(dynamic_result["split_metrics"].get("train") or {}),
            "total_return",
        )
        - _metric_value(dict(final_replay["split_metrics"].get("train") or {}), "total_return"),
        "val_total_return_gap": _metric_value(
            dict(dynamic_result["split_metrics"].get("val") or {}),
            "total_return",
        )
        - _metric_value(dict(final_replay["split_metrics"].get("val") or {}), "total_return"),
        "oos_total_return_gap": _metric_value(
            dict(dynamic_result["split_metrics"].get("oos") or {}),
            "total_return",
        )
        - _metric_value(dict(final_replay["split_metrics"].get("oos") or {}), "total_return"),
    }
    payload = {
        "artifact_kind": "live_portfolio_hybrid_retune",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "selection_policy": {
            "selection_score": "val_scaled_score + 0.18 * train_scaled_score",
            "oos_role": "report_only",
            "cash_efficiency_scored": False,
            "train_val_mdd_cap": MDD_CAP,
        },
        "split_windows": split_config.as_payload(),
        "config": asdict(config),
        "dynamic_scores": dynamic_scores,
        "static_final_allocation_scores": _score_from_metrics(
            dict(final_replay.get("split_metrics") or {})
        ),
        "dynamic_research_path": dynamic_result,
        "live_execution_model": {
            "live_mode": "retuned_live_portfolio_hybrid_mode",
            "strategy_class": "ArtifactPortfolioModeStrategy",
            "execution_model": "validation_primary_static_allocation_over_live_portfolio_modes",
            "dynamic_allocator_live_supported": False,
            "final_allocation_replay": final_replay,
            "dynamic_vs_final_allocation_gap": gap,
        },
        "source_live_mode_metrics": {
            str(row.get("name") or ""): {
                "train": dict(row.get("train") or {}),
                "val": dict(row.get("val") or {}),
                "oos": dict(row.get("oos") or {}),
                "metadata": dict(row.get("metadata") or {}),
            }
            for row in live_rows
        },
        "scenarios": {
            "refreshed_latest_tail": {
                "active_sleeves": [str(row.get("name") or "") for row in live_rows],
                "dates": list(final_replay.get("dates") or []),
                "daily_returns": list(final_replay.get("daily_returns") or []),
                "split_metrics": dict(final_replay.get("split_metrics") or {}),
                "final_allocation": dict(final_replay.get("final_allocation") or {}),
                "all_metrics": dict(final_replay.get("all_metrics") or {}),
                "comparison_rows": comparison_rows,
            }
        },
        "live_implementable_ranked_candidates": comparison_rows,
        "explicit_caveats": [
            "The retune only uses committed live portfolio modes as sleeves.",
            "The dynamic allocator path is still research/backtest; the deployable mode uses the validation-primary static allocation.",
            "OOS is report-only and was not used for tuning or health priors.",
            "Paper/canary execution is still required before capital promotion.",
        ],
    }
    paths = _write_reports(
        payload=payload,
        comparison_rows=comparison_rows,
        output_dir=output_dir,
        live_selection_dir=live_selection_dir,
    )
    return {"payload": payload, **paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--live-selection-dir", type=Path, default=LIVE_SELECTION_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_live_portfolio_hybrid_retune(
        output_dir=args.output_dir,
        live_selection_dir=args.live_selection_dir,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "payload"}, indent=2))


if __name__ == "__main__":
    main()
