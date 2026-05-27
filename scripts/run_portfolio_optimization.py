"""Constrained portfolio optimization over shortlisted strategy return streams."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.portfolio.optimizer_core import (
    PortfolioConstraintInfeasibleError,
    apply_caps as _apply_caps,
    StreamCache,
    build_portfolio_returns as _build_portfolio_returns,
    build_portfolio_stream as _build_portfolio_stream,
    canonical_split as _canonical_split,
    cluster_by_correlation as _cluster_by_correlation,
    metrics as _metrics,
    objective_policy_payload as _objective_policy_payload,
    safe_float as _safe_float,
    split_metrics as _split_metrics,
    split_stream as _split_stream,
    stream_to_array as _stream_to_array,
)

from lumina_quant.portfolio_split_contract import (
    acquire_portfolio_memory_guard,
    memory_policy_payload,
    portfolio_followup_default_budget_bytes,
)

DEFAULT_PORTFOLIO_SCORING_CONFIG: dict[str, Any] = {
    "candidate_rank_score_weights": {
        "sharpe_weight": 2.8,
        "deflated_sharpe_weight": 1.5,
        "pbo_penalty": 2.0,
        "return_weight": 25.0,
    },
    "allocation_quality_params": {
        "deflated_sharpe_floor": 0.01,
        "deflated_sharpe_offset": 0.5,
    },
    "vol_targeting": {
        "target_vol_floor": 0.01,
        "vol_scale_cap": 2.0,
        "vol_scale_epsilon": 1e-12,
    },
    "sensitivity": {
        "cost_stress_x2_multiplier": 2.0,
        "cost_stress_x3_multiplier": 3.0,
        "signal_drift_down_multiplier": 0.9,
        "signal_drift_up_multiplier": 1.1,
    },
    "constraints": {
        "max_strategy": 0.15,
        "max_family": 0.40,
        "max_asset": 0.20,
        "max_metals": 0.15,
    },
}


def _load_score_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"score config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid score config JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"score config must be a JSON object: {path}")
    return dict(payload)


def _resolve_portfolio_score_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    resolved = deepcopy(DEFAULT_PORTFOLIO_SCORING_CONFIG)
    if not isinstance(overrides, dict):
        return resolved
    source = overrides
    nested = source.get("portfolio_optimization")
    if isinstance(nested, dict):
        source = nested
    for key, default_value in resolved.items():
        override_value = source.get(key)
        if isinstance(default_value, dict) and isinstance(override_value, dict):
            for sub_key in default_value:
                if sub_key in override_value:
                    default_value[sub_key] = override_value[sub_key]
    return resolved


def _resolved_cli_or_config_float(
    cli_value: float | None, config_value: Any, *, default: float
) -> float:
    if cli_value is not None:
        return max(0.0, _safe_float(cli_value, default))
    return max(0.0, _safe_float(config_value, default))


def _inverse_vol_weight(returns: np.ndarray) -> float:
    sigma = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if sigma <= 1e-9:
        return 1.0
    return 1.0 / sigma


def _load_rows(args) -> tuple[list[dict[str, Any]], str]:
    research_path = Path(args.research_report)
    if research_path.exists():
        payload = json.loads(research_path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)]
        if rows:
            return rows, str(research_path.resolve())

    if args.team_report and Path(args.team_report).exists():
        payload = json.loads(Path(args.team_report).read_text(encoding="utf-8"))
        rows = list(payload.get("selected_team") or [])
        source = str(Path(args.team_report).resolve())
        if rows:
            return [dict(row) for row in rows if isinstance(row, dict)], source

    raise RuntimeError("No candidate rows available in research report or team report.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize shortlisted strategy portfolio.")
    parser.add_argument("--research-report", default="reports/candidate_research_latest.json")
    parser.add_argument("--team-report", default="reports/strategy_factory_report_latest.json")
    parser.add_argument("--score-config", default="", help="Optional scoring config JSON path.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--max-strategies", type=int, default=24)
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--report-split", default="oos")
    parser.add_argument("--target-vol", type=float, default=0.12)
    parser.add_argument("--correlation-threshold", type=float, default=0.60)
    parser.add_argument("--cost-penalty", type=float, default=0.35)
    parser.add_argument("--max-strategy-cap", type=float, default=None)
    parser.add_argument("--max-family-cap", type=float, default=None)
    parser.add_argument("--max-asset-cap", type=float, default=None)
    parser.add_argument("--max-metals-cap", type=float, default=None)
    parser.add_argument(
        "--memory-budget-bytes",
        type=int,
        default=portfolio_followup_default_budget_bytes(),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    score_config_payload: dict[str, Any] | None = None
    score_config_path = None
    memory_budget_bytes = max(1, int(args.memory_budget_bytes))
    memory_guard = acquire_portfolio_memory_guard(
        run_name="portfolio_optimization",
        output_dir=output_dir,
        input_path=args.research_report,
        metadata={
            "team_report": str(Path(args.team_report).resolve())
            if str(args.team_report).strip()
            else None,
            "soft_rss_bytes": memory_budget_bytes,
            "max_strategies": int(args.max_strategies),
        },
        budget_bytes=memory_budget_bytes,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"portfolio_optimization_{stamp}.json"
    json_latest = output_dir / "portfolio_optimization_latest.json"
    md_path = output_dir / f"portfolio_optimization_{stamp}.md"
    md_latest = output_dir / "portfolio_optimization_latest.md"
    markdown = "# Portfolio Optimization Report\n\n- Status: failed\n"
    report: dict[str, Any] = {
        "artifact_kind": "portfolio_optimization",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "error": None,
        "memory_policy": memory_policy_payload(budget_bytes=memory_budget_bytes),
    }

    try:
        if str(args.score_config).strip():
            score_config_path = Path(str(args.score_config)).resolve()
            try:
                score_config_payload = _load_score_config(score_config_path)
            except ValueError as exc:
                raise SystemExit(f"[PORTFOLIO] {exc}")
        optimization_config = _resolve_portfolio_score_config(score_config_payload)
        rank_weights = dict(optimization_config.get("candidate_rank_score_weights") or {})
        allocation_quality_params = dict(optimization_config.get("allocation_quality_params") or {})
        vol_targeting_params = dict(optimization_config.get("vol_targeting") or {})
        sensitivity_params = dict(optimization_config.get("sensitivity") or {})
        constraint_defaults = dict(optimization_config.get("constraints") or {})
        fit_split = _canonical_split(args.fit_split, default="val")
        report_split = _canonical_split(args.report_split, default="oos")

        rows_raw, source_path = _load_rows(args)
        if not rows_raw:
            raise RuntimeError("No candidate rows available for optimization.")
        memory_guard.checkpoint(
            "start",
            {
                "source_report": source_path,
                "fit_split": fit_split,
                "report_split": report_split,
                "candidate_count": len(rows_raw),
            },
        )

        filtered = [
            row
            for row in rows_raw
            if bool(row.get("pass", True))
            and _split_stream(row, fit_split)
            and _split_stream(row, report_split)
        ]
        if not filtered:
            filtered = [
                row
                for row in rows_raw
                if _split_stream(row, fit_split) or _split_stream(row, report_split)
            ] or rows_raw

        def _score(row: dict[str, Any]) -> float:
            fit_metrics = _split_metrics(row, fit_split)
            return float(
                (
                    _safe_float(rank_weights.get("sharpe_weight"), 2.8)
                    * _safe_float(fit_metrics.get("sharpe"), 0.0)
                )
                + (
                    _safe_float(rank_weights.get("deflated_sharpe_weight"), 1.5)
                    * _safe_float(fit_metrics.get("deflated_sharpe"), 0.0)
                )
                - (
                    _safe_float(rank_weights.get("pbo_penalty"), 2.0)
                    * _safe_float(fit_metrics.get("pbo"), 1.0)
                )
                + (
                    _safe_float(rank_weights.get("return_weight"), 25.0)
                    * _safe_float(fit_metrics.get("return"), 0.0)
                )
            )

        filtered.sort(key=_score, reverse=True)
        selected = filtered[: max(1, int(args.max_strategies))]

        rows = {str(row.get("candidate_id") or row.get("name")): row for row in selected}
        ids = list(rows.keys())
        stream_cache = StreamCache()
        fit_streams = {cid: stream_cache.aggregate_for_row(rows[cid], fit_split) for cid in ids}
        fit_map = {cid: stream_cache.array_for_row(rows[cid], fit_split) for cid in ids}
        clusters = _cluster_by_correlation(
            ids, fit_streams, threshold=float(args.correlation_threshold)
        )

        cluster_weight_raw: dict[int, float] = {}
        member_weight_raw: dict[str, float] = {}
        for c_idx, cluster in enumerate(clusters):
            cluster_rets = [fit_map[cid] for cid in cluster if fit_map[cid].size > 0]
            min_len = min((arr.size for arr in cluster_rets), default=0)
            if min_len <= 0:
                cluster_weight_raw[c_idx] = 1.0
                for cid in cluster:
                    member_weight_raw[cid] = 1.0 / max(1, len(cluster))
                continue

            cluster_port = np.zeros(min_len, dtype=float)
            invs = np.asarray(
                [_inverse_vol_weight(arr[-min_len:]) for arr in cluster_rets], dtype=float
            )
            inv_sum = float(np.sum(invs))
            invs = invs / inv_sum if inv_sum > 0 else np.ones_like(invs) / len(invs)

            for arr_weight, arr in zip(invs, cluster_rets, strict=True):
                cluster_port += arr_weight * arr[-min_len:]

            cluster_weight_raw[c_idx] = _inverse_vol_weight(cluster_port)

            for cid in cluster:
                row = rows[cid]
                fit_metrics = _split_metrics(row, fit_split)
                quality = max(
                    _safe_float(allocation_quality_params.get("deflated_sharpe_floor"), 0.01),
                    _safe_float(fit_metrics.get("deflated_sharpe"), 0.0)
                    + _safe_float(allocation_quality_params.get("deflated_sharpe_offset"), 0.5),
                )
                inv_vol = _inverse_vol_weight(fit_map[cid])
                turnover = _safe_float(fit_metrics.get("turnover"), 0.0)
                penalty = 1.0 + (float(args.cost_penalty) * turnover)
                member_weight_raw[cid] = (quality * inv_vol) / max(1e-9, penalty)

        cluster_total = float(sum(cluster_weight_raw.values()))
        cluster_weights = {
            key: (
                value / cluster_total
                if cluster_total > 0
                else 1.0 / max(1, len(cluster_weight_raw))
            )
            for key, value in cluster_weight_raw.items()
        }

        weights: dict[str, float] = {}
        for c_idx, cluster in enumerate(clusters):
            raw = np.asarray([member_weight_raw[cid] for cid in cluster], dtype=float)
            raw_sum = float(np.sum(raw))
            if raw_sum <= 0.0:
                raw = np.ones(len(cluster), dtype=float)
                raw_sum = float(len(cluster))
            normalized = raw / raw_sum
            for cid, w in zip(cluster, normalized, strict=True):
                weights[cid] = float(cluster_weights[c_idx] * w)

        configured_caps = {
            "max_strategy": _resolved_cli_or_config_float(
                args.max_strategy_cap,
                constraint_defaults.get("max_strategy"),
                default=0.15,
            ),
            "max_family": _resolved_cli_or_config_float(
                args.max_family_cap,
                constraint_defaults.get("max_family"),
                default=0.40,
            ),
            "max_asset": _resolved_cli_or_config_float(
                args.max_asset_cap,
                constraint_defaults.get("max_asset"),
                default=0.20,
            ),
            "max_metals": _resolved_cli_or_config_float(
                args.max_metals_cap,
                constraint_defaults.get("max_metals"),
                default=0.15,
            ),
        }
        try:
            weights, effective_caps = _apply_caps(
                weights,
                records=rows,
                max_strategy=configured_caps["max_strategy"],
                max_family=configured_caps["max_family"],
                max_asset=configured_caps["max_asset"],
                max_metals=configured_caps["max_metals"],
            )
        except PortfolioConstraintInfeasibleError as exc:
            report["constraints"] = {
                "status": "infeasible",
                "configured": dict(configured_caps),
                **dict(exc.details),
            }
            raise
        pre_vol_weights = {key: float(value) for key, value in weights.items()}
        active_budget = float(sum(max(0.0, value) for value in pre_vol_weights.values()))
        underdiversified_default_shortlist = (
            len(pre_vol_weights) > 1
            and active_budget <= configured_caps["max_strategy"] * len(pre_vol_weights) + 1e-9
            and active_budget <= 0.300000001
        )
        if active_budget > 0.0 and underdiversified_default_shortlist:
            weight_shares = {
                key: float(max(0.0, value) / active_budget)
                for key, value in pre_vol_weights.items()
            }
        elif active_budget > 0.0:
            weight_shares = dict(pre_vol_weights)
        else:
            weight_shares = dict.fromkeys(pre_vol_weights, 0.0)

        target_vol_floor = max(0.0, _safe_float(vol_targeting_params.get("target_vol_floor"), 0.01))
        vol_scale_cap = max(0.0, _safe_float(vol_targeting_params.get("vol_scale_cap"), 2.0))
        vol_scale_epsilon = max(
            0.0, _safe_float(vol_targeting_params.get("vol_scale_epsilon"), 1e-12)
        )
        portfolio_fit = _build_portfolio_returns(
            pre_vol_weights,
            rows,
            split=fit_split,
            cache=stream_cache,
        )
        fit_vol = _safe_float(np.std(portfolio_fit, ddof=1), 0.0)
        target_vol = max(target_vol_floor, float(args.target_vol))
        vol_scale = (
            1.0
            if fit_vol <= vol_scale_epsilon
            else min(
                vol_scale_cap,
                target_vol / max(vol_scale_epsilon, fit_vol),
            )
        )
        weights = {key: float(pre_vol_weights[key] * vol_scale) for key in pre_vol_weights}
        gross_exposure = float(sum(weights.values()))
        cash_weight = max(0.0, 1.0 - gross_exposure)

        portfolio_train_stream = _build_portfolio_stream(
            weights, rows, split="train", cache=stream_cache
        )
        portfolio_val_stream = _build_portfolio_stream(
            weights, rows, split="val", cache=stream_cache
        )
        portfolio_oos_stream = _build_portfolio_stream(
            weights, rows, split="oos", cache=stream_cache
        )
        portfolio_fit_stream = _build_portfolio_stream(
            weights,
            rows,
            split=fit_split,
            cache=stream_cache,
        )
        portfolio_report_stream = _build_portfolio_stream(
            weights,
            rows,
            split=report_split,
            cache=stream_cache,
        )

        portfolio_train = _stream_to_array(portfolio_train_stream)
        portfolio_val = _stream_to_array(portfolio_val_stream)
        portfolio_oos = _stream_to_array(portfolio_oos_stream)
        portfolio_fit = _stream_to_array(portfolio_fit_stream)
        portfolio_report = _stream_to_array(portfolio_report_stream)

        train_metrics = _metrics(portfolio_train)
        val_metrics = _metrics(portfolio_val)
        oos_metrics = _metrics(portfolio_oos)
        fit_metrics = _metrics(portfolio_fit)
        report_metrics = _metrics(portfolio_report)

        weighted_turnover = 0.0
        weighted_cost = 0.0
        for cid, weight in weights.items():
            row = rows[cid]
            report_row_metrics = _split_metrics(row, report_split)
            cost = _safe_float(((row.get("metadata") or {}).get("cost_rate")), 0.0005)
            weighted_turnover += weight * _safe_float(report_row_metrics.get("turnover"), 0.0)
            weighted_cost += weight * cost

        cost_stress_x2_multiplier = _safe_float(
            sensitivity_params.get("cost_stress_x2_multiplier"), 2.0
        )
        cost_stress_x3_multiplier = _safe_float(
            sensitivity_params.get("cost_stress_x3_multiplier"), 3.0
        )
        signal_drift_down_multiplier = _safe_float(
            sensitivity_params.get("signal_drift_down_multiplier"), 0.9
        )
        signal_drift_up_multiplier = _safe_float(
            sensitivity_params.get("signal_drift_up_multiplier"), 1.1
        )

        report_x2 = portfolio_report - (
            max(0.0, cost_stress_x2_multiplier - 1.0) * weighted_turnover * weighted_cost
        )
        report_x3 = portfolio_report - (
            max(0.0, cost_stress_x3_multiplier - 1.0) * weighted_turnover * weighted_cost
        )

        sensitivity = {
            "cost_stress": {
                "x2": _metrics(report_x2),
                "x3": _metrics(report_x3),
            },
            "param_drift": {
                "minus_10pct_signal": _metrics(portfolio_report * signal_drift_down_multiplier),
                "plus_10pct_signal": _metrics(portfolio_report * signal_drift_up_multiplier),
            },
        }

        sleeve_budget: dict[str, float] = defaultdict(float)
        for cid, weight in weights.items():
            sleeve = str(rows[cid].get("family") or "other")
            sleeve_budget[sleeve] += float(weight)

        ranked_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        allocation_rows = []
        for cid, weight in ranked_weights:
            row = rows[cid]
            fit_row_metrics = _split_metrics(row, fit_split)
            report_row_metrics = _split_metrics(row, report_split)
            allocation_rows.append(
                {
                    "candidate_id": cid,
                    "name": row.get("name"),
                    "strategy_class": row.get("strategy_class"),
                    "family": row.get("family"),
                    "symbols": list(row.get("symbols") or []),
                    "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                    "weight": float(weight),
                    "weight_share": float(weight_shares.get(cid, 0.0)),
                    "fit_split": fit_split,
                    "fit_sharpe": _safe_float(fit_row_metrics.get("sharpe"), 0.0),
                    "fit_return": _safe_float(fit_row_metrics.get("return"), 0.0),
                    "report_split": report_split,
                    "report_sharpe": _safe_float(report_row_metrics.get("sharpe"), 0.0),
                    "report_return": _safe_float(report_row_metrics.get("return"), 0.0),
                    "oos_sharpe": _safe_float((row.get("oos") or {}).get("sharpe"), 0.0),
                    "oos_return": _safe_float((row.get("oos") or {}).get("return"), 0.0),
                }
            )

        report = {
            "artifact_kind": "portfolio_optimization",
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "completed",
            "error": None,
            "source_report": source_path,
            "selection": {
                "fit_split": fit_split,
                "report_split": report_split,
                "selection_basis": "validation_only"
                if fit_split == "val" and report_split == "oos"
                else f"{fit_split}_fit",
            },
            "objective_policy": _objective_policy_payload(
                f"{fit_split}_fit",
                oos_is_objective_input=fit_split == "oos",
            ),
            "cluster_count": len(clusters),
            "clusters": clusters,
            "gross_exposure": gross_exposure,
            "cash_weight": cash_weight,
            "constraints": {
                "max_strategy": float(
                    effective_caps.get("max_strategy", configured_caps["max_strategy"])
                ),
                "max_family": float(
                    effective_caps.get("max_family", configured_caps["max_family"])
                ),
                "max_asset": float(effective_caps.get("max_asset", configured_caps["max_asset"])),
                "max_metals": float(
                    effective_caps.get("max_metals", configured_caps["max_metals"])
                ),
                "target_active_weight": float(
                    effective_caps.get("target_active_weight", active_budget)
                ),
                "active_weight": float(effective_caps.get("active_weight", active_budget)),
                "cash_reserve_weight": float(
                    effective_caps.get("cash_reserve_weight", max(0.0, 1.0 - active_budget))
                ),
                "family_caps": dict(effective_caps.get("family_caps") or {}),
                "configured": configured_caps,
            },
            "scoring": {
                "candidate_rank_score_weights": {
                    "sharpe_weight": _safe_float(rank_weights.get("sharpe_weight"), 2.8),
                    "deflated_sharpe_weight": _safe_float(
                        rank_weights.get("deflated_sharpe_weight"), 1.5
                    ),
                    "pbo_penalty": _safe_float(rank_weights.get("pbo_penalty"), 2.0),
                    "return_weight": _safe_float(rank_weights.get("return_weight"), 25.0),
                },
                "vol_targeting": {
                    "target_vol_floor": float(target_vol_floor),
                    "vol_scale_cap": float(vol_scale_cap),
                    "vol_scale_epsilon": float(vol_scale_epsilon),
                    "target_vol": float(target_vol),
                    "fit_vol": float(fit_vol),
                    "vol_scale": float(vol_scale),
                },
                "sensitivity": {
                    "cost_stress_x2_multiplier": float(cost_stress_x2_multiplier),
                    "cost_stress_x3_multiplier": float(cost_stress_x3_multiplier),
                    "signal_drift_down_multiplier": float(signal_drift_down_multiplier),
                    "signal_drift_up_multiplier": float(signal_drift_up_multiplier),
                },
                "source": str(score_config_path) if score_config_path is not None else "",
            },
            "memory_policy": memory_policy_payload(budget_bytes=memory_budget_bytes),
            "weights": allocation_rows,
            "sleeve_budget": dict(sorted(sleeve_budget.items())),
            "portfolio_return_streams": {
                "train": portfolio_train_stream,
                "val": portfolio_val_stream,
                "oos": portfolio_oos_stream,
                fit_split: portfolio_fit_stream,
                report_split: portfolio_report_stream,
            },
            "portfolio_metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "oos": oos_metrics,
                fit_split: fit_metrics,
                report_split: report_metrics,
            },
            "fit_metrics": fit_metrics,
            "report_metrics": report_metrics,
            "sensitivity": sensitivity,
        }
        memory_guard.checkpoint(
            "completed",
            {
                "selected_candidates": len(allocation_rows),
                "cluster_count": len(clusters),
            },
        )

        lines = [
            "# Portfolio Optimization Report",
            "",
            f"- Source report: `{source_path}`",
            f"- Fit split: `{fit_split}`",
            f"- Report split: `{report_split}`",
            f"- Clusters: {len(clusters)}",
            f"- Gross exposure: `{gross_exposure:.2%}`",
            f"- Cash weight: `{cash_weight:.2%}`",
            "",
            "## Sleeve budgets",
            "",
        ]
        for family, weight in sorted(sleeve_budget.items()):
            lines.append(f"- {family}: {weight:.2%}")

        lines.extend(
            [
                "",
                "## Top strategy weights",
                "",
                "| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |",
                "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for idx, row in enumerate(allocation_rows[:20], start=1):
            lines.append(
                "| "
                f"{idx} | {row['name']} | {row['strategy_class']} | {row['family']} | {row['timeframe']} | "
                f"{row['weight']:.2%} | {row['fit_sharpe']:.3f} | {row['fit_return']:.2%} | "
                f"{row['report_sharpe']:.3f} | {row['report_return']:.2%} |"
            )

        lines.extend(
            [
                "",
                f"JSON: `{json_path}`",
                f"Latest: `{json_latest}`",
                "",
            ]
        )
        markdown = "\n".join(lines) + "\n"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["memory"] = memory_guard.finalize(
            status=str(report.get("status") or "completed"),
            error=str(report.get("error") or "") or None,
        )
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        json_latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        md_latest.write_text(markdown, encoding="utf-8")
        memory_guard.release()

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
