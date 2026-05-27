"""Regime-aware four-sleeve allocator over selected strategy return streams.

The sleeve universe is intentionally small and explicit so the allocator can
run under the shared follow-up memory guard while staying below the 8 GiB
session cap. The default universe combines the incumbent's diversifying
trend/topcap sleeves with the autoresearch pair sleeves:

- pair_spread_1d_balanced_btcusdt_trxusdt_1.8_0.45
- pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.6_0.70
- composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80
- topcap_tsmom_1h_balanced_16_4_0.015

Weights are rebalanced on a weekly cadence, regime-aware, and Kelly-sized at
portfolio level. Outputs include train/val/OOS metrics plus cash/turnover
summaries for each split.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
    resolve_followup_artifact_path,
    split_for_day_key,
)

_THIS_DIR = Path(__file__).resolve().parent


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_helper = _load_script_module(
    "run_causal_dynamic_portfolio_helper", "run_causal_dynamic_portfolio.py"
)
_switch = _load_script_module(
    "run_regime_switching_portfolio_helper", "run_regime_switching_portfolio.py"
)

DEFAULT_OUTPUT_DIR = FOLLOWUP_ROOT / "portfolio_four_sleeve_regime_current"
DEFAULT_CONTINUITY_REPORT = FOLLOWUP_ROOT / "portfolio_continuity_validation_latest.json"

DEFAULT_SLEEVE_SPECS: tuple[dict[str, str], ...] = (
    {
        "alias": "btc_trx_pair_balanced",
        "name": "pair_spread_1d_balanced_btcusdt_trxusdt_1.8_0.45",
        "path": ".omx/team/execute-the-approved-luminaqua/worktrees/worker-2/var/reports/exact_window_backtests/autonomous_daily_1d_20260315T035155Z/1d/exact_window_candidate_details_latest.json",
    },
    {
        "alias": "bnb_trx_pair_tightstop",
        "name": "pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.6_0.70",
        "path": ".omx/team/execute-the-approved-luminaqua/worktrees/worker-2/var/reports/exact_window_backtests/autonomous_pair_exec_takeprofit_bnbtrx_1h_20260316T0958Z/1h/exact_window_candidate_details_latest.json",
    },
    {
        "alias": "composite_trend_core",
        "name": "composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80",
        "path": ".omx/team/execute-the-approved-luminaqua/worktrees/worker-2/var/reports/exact_window_backtests/autonomous_composite_exec_30m_20260315T1057Z/30m/exact_window_candidate_details_latest.json",
    },
    {
        "alias": "topcap_balanced",
        "name": "topcap_tsmom_1h_balanced_16_4_0.015",
        "path": ".omx/team/execute-the-approved-luminaqua/worktrees/worker-2/var/reports/exact_window_backtests/autonomous_topcap_exec_1h_20260315T0945Z/1h/exact_window_candidate_details_latest.json",
    },
)

DEFAULT_INCUMBENT_METRICS = (
    ".omx/team/execute-the-approved-luminaqua/worktrees/worker-2/var/reports/"
    "exact_window_backtests/followup_status/portfolio_one_shot_current_opt/"
    "portfolio_optimization_latest.json"
)
DEFAULT_AUTORESEARCH_METRICS = (
    ".omx/worktrees/autoresearch-omx-specs-autoresearch-binance-top10-metals-regi-20260327t154348z/"
    "var/reports/exact_window_backtests/followup_status/autoresearch_candidate_portfolio_opt/"
    "portfolio_optimization_latest.json"
)


@dataclass(frozen=True)
class SleeveAllocatorParams:
    lookback_days: int
    rebalance_days: int
    min_trailing_sharpe: float
    min_trailing_return: float
    max_trailing_drawdown: float
    max_weight: float
    regime_strength: float
    hysteresis_bonus: float
    turnover_cost_bps: float
    cash_buffer: float = 0.0
    kelly_shrinkage: float = 1.0
    switch_score_hurdle: float = 0.1
    min_regime_score: float = 0.7
    max_weight_turnover: float = 0.75


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_candidate_source(path: str) -> Path:
    return resolve_followup_artifact_path(Path(path)).resolve()


def _load_named_candidate_row(path: Path, *, name: str, alias: str) -> dict[str, Any]:
    payload = _load_json(path)
    rows = payload if isinstance(payload, list) else list(payload.get("candidates") or [])
    for row in rows:
        if str(row.get("name") or "") != name:
            continue
        selected = dict(row)
        selected["candidate_id"] = alias
        selected["alias"] = alias
        selected["source_candidate_id"] = row.get("candidate_id")
        selected["source_path"] = str(path)
        return selected
    raise RuntimeError(f"candidate {name} not found in {path}")


def _load_default_sleeves() -> list[dict[str, Any]]:
    return [
        _load_named_candidate_row(
            _resolve_candidate_source(spec["path"]),
            name=spec["name"],
            alias=spec["alias"],
        )
        for spec in DEFAULT_SLEEVE_SPECS
    ]


def _candidate_runtime_risk(
    continuity_payload: dict[str, Any], *, require_continuity_pass: bool
) -> dict[str, Any]:
    status = str(continuity_payload.get("status") or "missing").lower()
    continuity_failed = status not in {"completed", "passed", "ok", "success"}
    return {
        "continuity_status": status,
        "continuity_blocked": bool(require_continuity_pass and continuity_failed),
    }


def _sleeve_regime_score(
    row: dict[str, Any],
    regime_row: dict[str, Any] | None,
    *,
    previous_active: bool,
    strength: float,
) -> float:
    if not regime_row or strength <= 0.0:
        return 1.0
    if str(row.get("strategy_class") or "") == "PairSpreadZScoreStrategy":
        component = {
            "strategy_class": row.get("strategy_class"),
            "metadata": dict(row.get("metadata") or {}),
            "family": row.get("family"),
            "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
        }
        return float(
            _switch._component_regime_multiplier(
                component,
                regime_row,
                previous_active=previous_active,
                strength=strength,
            )
        )
    normalized = {
        "strategy_class": row.get("strategy_class"),
        "metadata": dict(row.get("metadata") or {}),
    }
    return float(
        _helper._regime_multiplier(
            normalized,
            regime_row,
            previous_active=previous_active,
            strength=strength,
        )
    )


def _weight_turnover(prev_weights: dict[str, float], next_weights: dict[str, float]) -> float:
    ids = set(prev_weights) | set(next_weights)
    return float(
        sum(
            abs(
                _helper._safe_float(prev_weights.get(cid), 0.0)
                - _helper._safe_float(next_weights.get(cid), 0.0)
            )
            for cid in ids
        )
    )


def _cap_transition(
    prev_weights: dict[str, float], next_weights: dict[str, float], *, max_turnover: float
) -> dict[str, float]:
    current_turnover = _weight_turnover(prev_weights, next_weights)
    if current_turnover <= max_turnover or current_turnover <= 1e-12:
        return dict(next_weights)
    mix = max_turnover / current_turnover
    ids = set(prev_weights) | set(next_weights)
    out = {
        cid: (
            _helper._safe_float(prev_weights.get(cid), 0.0)
            + mix
            * (
                _helper._safe_float(next_weights.get(cid), 0.0)
                - _helper._safe_float(prev_weights.get(cid), 0.0)
            )
        )
        for cid in ids
    }
    return {cid: weight for cid, weight in out.items() if weight > 1e-12}


def run_four_sleeve_allocator(
    rows: list[dict[str, Any]],
    params: SleeveAllocatorParams,
    *,
    regime_features: dict[str, dict[str, Any]] | None = None,
    continuity_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_days, matrix, meta = _helper._build_daily_panel(rows)
    regime_by_day = regime_features or _helper._load_regime_features(rows)
    continuity_payload = continuity_payload or {"status": "missing"}
    runtime_risk = _candidate_runtime_risk(continuity_payload, require_continuity_pass=True)

    ids = list(matrix.keys())
    weights: dict[str, float] = {}
    daily_returns: list[float] = []
    allocations: list[dict[str, Any]] = []
    split_returns = {"train": [], "val": [], "oos": []}
    turnover_cost = 0.0

    for idx, day_key in enumerate(ordered_days):
        split = split_for_day_key(day_key)
        regime_row = dict(regime_by_day.get(day_key) or {})

        if idx > 0 and ((idx % params.rebalance_days == 0) or not weights):
            history = {cid: matrix[cid][max(0, idx - params.lookback_days) : idx] for cid in ids}
            raw_scores: dict[str, float] = {}
            diagnostics: dict[str, Any] = {}

            for cid, hist in history.items():
                metric = _helper._metrics(np.asarray(hist, dtype=float))
                trailing_return = float(metric["total_return"])
                trailing_sharpe = float(metric["sharpe"])
                trailing_dd = float(metric["max_drawdown"])
                trailing_vol = float(metric["volatility"])
                regime_score = _sleeve_regime_score(
                    meta[cid],
                    regime_row,
                    previous_active=bool(_helper._safe_float(weights.get(cid), 0.0) > 0.0),
                    strength=params.regime_strength,
                )
                diagnostics[cid] = {
                    "trailing_return": trailing_return,
                    "trailing_sharpe": trailing_sharpe,
                    "trailing_drawdown": trailing_dd,
                    "trailing_volatility": trailing_vol,
                    "regime_score": regime_score,
                    **runtime_risk,
                }
                if runtime_risk["continuity_blocked"]:
                    continue
                if trailing_sharpe < params.min_trailing_sharpe:
                    continue
                if trailing_return < params.min_trailing_return:
                    continue
                if trailing_dd > params.max_trailing_drawdown:
                    continue
                if regime_score < params.min_regime_score:
                    continue

                persistence = 1.0 + (params.hysteresis_bonus if cid in weights else 0.0)
                switch_drag = (
                    1.0 if cid in weights else 1.0 / max(1.0, 1.0 + params.switch_score_hurdle)
                )
                vol_inv = 1.0 / max(trailing_vol, 1e-6)
                dd_penalty = max(
                    0.05, 1.0 - (trailing_dd / max(params.max_trailing_drawdown, 1e-6))
                )
                raw_scores[cid] = float(
                    (1.0 + max(0.0, trailing_sharpe))
                    * (1.0 + (8.0 * max(0.0, trailing_return)))
                    * vol_inv
                    * dd_penalty
                    * regime_score
                    * persistence
                    * switch_drag
                )

            target_total_exposure = max(0.0, 1.0 - params.cash_buffer)
            if raw_scores:
                unconstrained = _switch._blend_scores(
                    history,
                    raw_scores,
                    target_total_exposure=target_total_exposure,
                    max_portfolio_weight=params.max_weight,
                    correlation_penalty=0.5,
                )
                target_total_exposure = _switch._kelly_target_exposure(
                    history,
                    unconstrained,
                    cash_buffer=params.cash_buffer,
                    kelly_shrinkage=params.kelly_shrinkage,
                )

            next_weights = _switch._blend_scores(
                history,
                raw_scores,
                target_total_exposure=target_total_exposure,
                max_portfolio_weight=params.max_weight,
                correlation_penalty=0.5,
            )
            weight_turnover = _weight_turnover(weights, next_weights)
            if weights and weight_turnover > params.max_weight_turnover:
                next_weights = _cap_transition(
                    weights, next_weights, max_turnover=params.max_weight_turnover
                )
                weight_turnover = _weight_turnover(weights, next_weights)

            turnover_cost = (max(0.0, params.turnover_cost_bps) / 10_000.0) * weight_turnover
            weights = next_weights
            allocations.append(
                {
                    "date": day_key,
                    "weights": dict(weights),
                    "cash_weight": max(params.cash_buffer, 1.0 - sum(weights.values())),
                    "diagnostics": diagnostics,
                    "regime_row": regime_row,
                    "weight_turnover": weight_turnover,
                    "sleeve_turnover": weight_turnover,
                    "turnover_cost": turnover_cost,
                    "target_total_exposure": target_total_exposure,
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
                    "weight_turnover": 0.0,
                    "sleeve_turnover": 0.0,
                    "turnover_cost": 0.0,
                    "target_total_exposure": 0.0,
                }
            )

        day_return = sum(float(weights.get(cid, 0.0)) * float(matrix[cid][idx]) for cid in ids)
        day_return -= turnover_cost
        daily_returns.append(day_return)
        split_returns[split].append(day_return)

    return {
        "dates": ordered_days,
        "daily_returns": daily_returns,
        "allocations": allocations,
        "meta": meta,
        "continuity_status": continuity_payload.get("status"),
        "runtime_risk_state": runtime_risk,
        "split_metrics": {
            split: _helper._metrics(np.asarray(values, dtype=float))
            for split, values in split_returns.items()
        },
        "all_metrics": _helper._metrics(np.asarray(daily_returns, dtype=float)),
    }


def _mean_split_fraction(allocations: list[dict[str, Any]], *, split: str, key: str) -> float:
    values = [
        _helper._safe_float(item.get(key), 0.0)
        for item in allocations
        if split_for_day_key(str(item.get("date") or "")) == split
    ]
    return float(np.mean(values)) if values else 0.0


def search_four_sleeve_allocator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regime_features = _helper._load_regime_features(rows)
    continuity_payload = _load_json(resolve_followup_artifact_path(DEFAULT_CONTINUITY_REPORT))
    grid = [
        SleeveAllocatorParams(*combo)
        for combo in itertools.product(
            [10, 14, 21],
            [7],
            [0.0, 0.25],
            [0.0],
            [0.10, 0.15],
            [0.45, 0.55],
            [1.0, 1.5],
            [0.05, 0.20],
            [8.0],
            [0.0],
            [0.75, 1.0],
        )
    ]

    best: dict[str, Any] | None = None
    for params in grid:
        result = run_four_sleeve_allocator(
            rows,
            params,
            regime_features=regime_features,
            continuity_payload=continuity_payload,
        )
        val_metrics = dict((result.get("split_metrics") or {}).get("val") or {})
        allocations = list(result.get("allocations") or [])
        objective = _switch._objective(
            val_metrics,
            cash_fraction=_mean_split_fraction(allocations, split="val", key="cash_weight"),
            turnover_fraction=_mean_split_fraction(allocations, split="val", key="weight_turnover"),
        )
        candidate = {"params": asdict(params), "objective": objective, "result": result}
        if best is None or objective > float(best["objective"]):
            best = candidate

    if best is None:
        raise RuntimeError("four-sleeve search produced no result")
    return best


def _final_allocation_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    allocations = list(result.get("allocations") or [])
    if not allocations:
        return []
    latest = allocations[-1]
    meta = dict(result.get("meta") or {})
    rows = []
    for cid, weight in sorted(
        dict(latest.get("weights") or {}).items(), key=lambda item: float(item[1]), reverse=True
    ):
        row = dict(meta.get(cid) or {})
        row["candidate_id"] = cid
        row["weight"] = float(weight)
        rows.append(row)
    return rows


def _allocation_summary(
    allocations: list[dict[str, Any]], *, candidate_ids: list[str]
) -> dict[str, Any]:
    return _switch._allocation_summary(allocations, candidate_ids=candidate_ids)


def _load_portfolio_metrics(path_str: str) -> dict[str, Any]:
    path = resolve_followup_artifact_path(Path(path_str)).resolve()
    payload = _load_json(path)
    return dict(payload.get("portfolio_metrics") or payload.get("split_metrics") or {})


def write_report(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_default_sleeves()
    memory_guard = acquire_portfolio_memory_guard(
        run_name="portfolio_four_sleeve_regime_allocator",
        output_dir=output_dir,
        input_path="::".join(
            str(_resolve_candidate_source(spec["path"])) for spec in DEFAULT_SLEEVE_SPECS
        ),
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    status = "completed"
    error: str | None = None
    try:
        memory_guard.checkpoint("four_sleeve_regime_start", {"candidate_count": len(rows)})
        best = search_four_sleeve_allocator(rows)
        result = dict(best["result"])
        payload = {
            "artifact_kind": "four_sleeve_regime_allocator",
            "generated_at": datetime.now(UTC).isoformat(),
            "selection_basis": "weekly_regime_kelly_sleeve_allocator",
            "best_params": best["params"],
            "objective": best["objective"],
            "split_metrics": result.get("split_metrics"),
            "all_metrics": result.get("all_metrics"),
            "allocations": result.get("allocations"),
            "dates": result.get("dates"),
            "daily_returns": result.get("daily_returns"),
            "final_allocation": _final_allocation_rows(result),
            "allocation_summary": {
                "all": _allocation_summary(
                    list(result.get("allocations") or []),
                    candidate_ids=[row["candidate_id"] for row in rows],
                ),
                "train": _allocation_summary(
                    [
                        item
                        for item in list(result.get("allocations") or [])
                        if split_for_day_key(str(item.get("date") or "")) == "train"
                    ],
                    candidate_ids=[row["candidate_id"] for row in rows],
                ),
                "val": _allocation_summary(
                    [
                        item
                        for item in list(result.get("allocations") or [])
                        if split_for_day_key(str(item.get("date") or "")) == "val"
                    ],
                    candidate_ids=[row["candidate_id"] for row in rows],
                ),
                "oos": _allocation_summary(
                    [
                        item
                        for item in list(result.get("allocations") or [])
                        if split_for_day_key(str(item.get("date") or "")) == "oos"
                    ],
                    candidate_ids=[row["candidate_id"] for row in rows],
                ),
            },
            "continuity_status": result.get("continuity_status"),
            "memory_policy": memory_policy_payload(
                budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
            ),
            "sleeves": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "name": row.get("name"),
                    "strategy_class": row.get("strategy_class"),
                    "family": row.get("family"),
                    "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                    "symbols": row.get("symbols"),
                    "source_path": row.get("source_path"),
                    "source_candidate_id": row.get("source_candidate_id"),
                }
                for row in rows
            ],
            "benchmarks": {
                "incumbent": _load_portfolio_metrics(DEFAULT_INCUMBENT_METRICS),
                "autoresearch_55_45": _load_portfolio_metrics(DEFAULT_AUTORESEARCH_METRICS),
            },
        }
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(
            event="four_sleeve_regime_finish", context={"status": status, "error": error}
        )
        memory_summary = memory_guard.finalize(
            status=status,
            error=error,
            context={"candidate_count": len(rows)},
        )
        memory_guard.release()

    payload["memory_summary"] = memory_summary
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"four_sleeve_regime_portfolio_{stamp}.json"
    latest_path = output_dir / "four_sleeve_regime_portfolio_latest.json"
    md_path = output_dir / f"four_sleeve_regime_portfolio_{stamp}.md"
    serializable_payload = _switch._json_ready(payload)
    json_path.write_text(json.dumps(serializable_payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(serializable_payload, indent=2), encoding="utf-8")
    lines = [
        "# Four-sleeve regime allocator",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- selection_basis: {payload['selection_basis']}",
        f"- best_params: {json.dumps(payload['best_params'], sort_keys=True)}",
        f"- objective: {payload['objective']}",
        f"- continuity_status: {payload['continuity_status']}",
        "",
        "## Sleeves",
    ]
    for sleeve in payload["sleeves"]:
        lines.append(
            f"- {sleeve['candidate_id']}: {sleeve['name']} "
            f"({sleeve['family']}, {sleeve['timeframe']})"
        )
    lines.extend(
        [
            "",
            "## Split metrics",
            f"- train: {json.dumps(payload['split_metrics'].get('train') or {}, sort_keys=True)}",
            f"- val: {json.dumps(payload['split_metrics'].get('val') or {}, sort_keys=True)}",
            f"- oos: {json.dumps(payload['split_metrics'].get('oos') or {}, sort_keys=True)}",
            "",
            "## Allocation summary",
            f"- all: {json.dumps(payload['allocation_summary'].get('all') or {}, sort_keys=True)}",
            f"- train: {json.dumps(payload['allocation_summary'].get('train') or {}, sort_keys=True)}",
            f"- val: {json.dumps(payload['allocation_summary'].get('val') or {}, sort_keys=True)}",
            f"- oos: {json.dumps(payload['allocation_summary'].get('oos') or {}, sort_keys=True)}",
            "",
            "## Final allocation",
        ]
    )
    for row in payload["final_allocation"]:
        lines.append(f"- {row['candidate_id']}: {row['weight']:.4f} ({row.get('name')})")
    lines.extend(
        [
            "",
            "## Benchmarks",
            f"- incumbent_oos: {json.dumps((payload['benchmarks'].get('incumbent') or {}).get('oos') or {}, sort_keys=True)}",
            f"- autoresearch_55_45_oos: {json.dumps((payload['benchmarks'].get('autoresearch_55_45') or {}).get('oos') or {}, sort_keys=True)}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest_md_path = output_dir / "four_sleeve_regime_portfolio_latest.md"
    latest_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = write_report(output_dir=args.output_dir)
    print((args.output_dir / "four_sleeve_regime_portfolio_latest.json").resolve())
    print(
        json.dumps(
            {
                "oos": (payload.get("split_metrics") or {}).get("oos"),
                "allocation_summary_oos": (payload.get("allocation_summary") or {}).get("oos"),
                "final_allocation": payload.get("final_allocation"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
