from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lumina_quant.dashboard.exact_window_bundle import load_exact_window_bundle  # noqa: E402
from lumina_quant.eval.exact_window_suite import _metrics_daily  # noqa: E402

REPORT_ROOT = REPO_ROOT / "var" / "reports" / "exact_window_backtests"
FOLLOWUP_ROOT = REPORT_ROOT / "followup_status"
METALS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}
EXPERIMENTAL_MAX_PBO = 0.40
EXPERIMENTAL_MIN_OOS_SHARPE = 1.0
EXPERIMENTAL_MIN_OOS_RETURN = 0.0
EXPERIMENTAL_MIN_TRADE_COUNT = 5.0
EXPERIMENTAL_MIN_VAL_SHARPE = 1.0
EXPERIMENTAL_MIN_TRAIN_RETURN = -0.12

PRIMARY_WATCHLIST_STEMS = [
    "4h_btc_xag_tuned_latest",
    "4h_metals_adaptive_latest",
    "4h_metals_core_adaptive_latest",
]
SCENARIO_SPECS: list[dict[str, Any]] = [
    {
        "scenario_id": "strict_anchor_only",
        "label": "Strict anchor only baseline",
        "selection_basis": "strict_promoted_only",
        "stems": [],
        "watchlist_label": None,
        "watchlist_role": None,
        "notes": [
            "Baseline scenario using only strict promoted rows from the root decision artifact.",
        ],
    },
    {
        "scenario_id": "strict_plus_best_4h_watchlist",
        "label": "Strict anchor + best 4h watchlist",
        "selection_basis": "equal_weight_experimental",
        "stems": [
            "4h_pbo_final_latest",
            "4h_metals_mixed_latest",
            "4h_latest",
            "4h_metals_adaptive_latest",
            "4h_metals_core_adaptive_latest",
        ],
        "watchlist_label": "4h best watchlist",
        "watchlist_role": "4h watchlist",
        "notes": [
            "Adds the strongest saved 4h follow-up row, prioritizing the post-PBO retune artifact when present.",
        ],
    },
    {
        "scenario_id": "strict_plus_crypto_metal_watchlist",
        "label": "Strict anchor + 4h crypto-metal watchlist",
        "selection_basis": "equal_weight_experimental",
        "stems": PRIMARY_WATCHLIST_STEMS,
        "watchlist_label": "4h BTC/XAG watchlist",
        "watchlist_role": "mixed-asset watchlist",
        "notes": [
            "Adds the latest crypto-metal overlay, prioritizing the tuned BTC/XAG sleeve.",
            "This is the preferred deployment watchlist for metals diversification review.",
        ],
    },
]


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _preferred_followup_best(
    bundle: dict[str, Any],
    stems: list[str],
    *,
    timeframe: str | None = None,
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    followup_status = dict(bundle.get("followup_status") or {})
    for stem in stems:
        payload = dict(followup_status.get(stem) or {})
        best = dict(payload.get("best_row") or {})
        if not best:
            continue
        if timeframe is not None and str(best.get("strategy_timeframe") or "") != timeframe:
            continue
        return stem, payload, best
    return None, {}, {}


def _decorate_deployment_row(
    row: dict[str, Any],
    *,
    role: str,
    label: str,
    stage: str,
    run_id: Any,
    memory_evidence: dict[str, Any] | None,
    risk_flags: list[str] | None = None,
) -> dict[str, Any]:
    committee = dict(row.get("committee") or {})
    provided_risk_flags = list(risk_flags or [])
    derived_risk_flags: list[str] = list(provided_risk_flags)
    for item in list(committee.get("risk_flags") or []):
        if str(item) not in derived_risk_flags:
            derived_risk_flags.append(str(item))
    decorated = {
        **dict(row),
        "_deployment_role": role,
        "_deployment_label": label,
        "_deployment_stage": stage,
        "_deployment_run_id": run_id,
        "_deployment_memory_evidence": dict(memory_evidence or {}),
        "_deployment_final_decision": str(committee.get("final_decision") or ""),
        "_deployment_weight": 1.0,
    }
    if derived_risk_flags:
        decorated["_deployment_risk_flags"] = derived_risk_flags
    return decorated


def _with_equal_weights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    equal_weight = 1.0 / float(len(rows))
    return [{**row, "_deployment_weight": equal_weight} for row in rows]


def _strict_anchor_rows(decision: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timeframe_row in list(decision.get("timeframe_rows") or []):
        best = dict(timeframe_row.get("best_row") or {})
        if not bool(best.get("promoted")):
            continue
        rows.append(
            _decorate_deployment_row(
                best,
                role="strict anchor",
                label=f"{timeframe_row.get('timeframe')} strict anchor",
                stage="root_decision",
                run_id=None,
                memory_evidence=dict(timeframe_row.get("memory_evidence") or {}),
            )
        )
    return rows


def _deployment_candidate_rows(
    bundle: dict[str, Any],
    decision: dict[str, Any],
    *,
    stems: list[str] | None = None,
    watchlist_label: str | None = None,
    watchlist_role: str | None = None,
) -> list[dict[str, Any]]:
    rows = _strict_anchor_rows(decision)
    if stems:
        stage, payload, best = _preferred_followup_best(bundle, stems, timeframe="4h")
        if best:
            rows.append(
                _decorate_deployment_row(
                    best,
                    role=watchlist_role or "watchlist",
                    label=watchlist_label or "watchlist sleeve",
                    stage=str(stage or ""),
                    run_id=payload.get("run_id"),
                    memory_evidence=dict(payload.get("memory_evidence") or {}),
                )
            )
    return _with_equal_weights(rows)


def _weighted_candidate_stream(rows: list[dict[str, Any]], split: str) -> list[dict[str, float]]:
    if not rows:
        return []
    weight_map = {
        str(row.get("candidate_id") or row.get("name") or idx): float(
            row.get("_deployment_weight", 0.0)
        )
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
    return [{"t": int(day.timestamp() * 1000), "v": float(bucket[day])} for day in sorted(bucket)]


def _combo_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    stream = _weighted_candidate_stream(rows, split)
    returns = np.asarray([float(point.get("v", 0.0)) for point in stream], dtype=float)
    metrics = (
        dict(_metrics_daily(returns))
        if returns.size
        else dict(_metrics_daily(np.asarray([], dtype=float)))
    )
    metrics["return"] = float(metrics.get("total_return", 0.0))
    metrics["pbo"] = max(
        (float((row.get(split) or {}).get("pbo", 0.0) or 0.0) for row in rows), default=0.0
    )
    metrics["trade_count"] = float(
        sum(float((row.get(split) or {}).get("trade_count", 0.0) or 0.0) for row in rows)
    )
    metrics["component_count"] = float(len(rows))
    return metrics


def _component_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload_rows: list[dict[str, Any]] = []
    for row in rows:
        val = dict(row.get("val") or {})
        oos = dict(row.get("oos") or {})
        train = dict(row.get("train") or {})
        memory = dict(row.get("_deployment_memory_evidence") or {})
        symbols = list(row.get("symbols") or [])
        metal_count = sum(1 for symbol in symbols if symbol in METALS)
        asset_mix = (
            "crypto-metal mix"
            if metal_count and metal_count < len(symbols)
            else "pure metal"
            if metal_count == len(symbols) and metal_count > 0
            else "crypto basket"
        )
        payload_rows.append(
            {
                "role": row.get("_deployment_role"),
                "label": row.get("_deployment_label"),
                "stage": row.get("_deployment_stage"),
                "run_id": row.get("_deployment_run_id"),
                "weight": float(row.get("_deployment_weight", 0.0)),
                "strategy_class": row.get("strategy_class"),
                "name": row.get("name"),
                "timeframe": row.get("strategy_timeframe"),
                "symbols": symbols,
                "asset_mix": asset_mix,
                "risk_flags": list(row.get("_deployment_risk_flags") or []),
                "committee": dict(row.get("committee") or {}),
                "final_decision": row.get("_deployment_final_decision"),
                "train": train,
                "val": val,
                "oos": oos,
                "memory_evidence": memory,
            }
        )
    return payload_rows


def _build_payload(
    rows: list[dict[str, Any]],
    *,
    generated_at: str,
    scenario_id: str,
    label: str,
    selection_basis: str,
    notes: list[str],
    source_stems: list[str],
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "schema_version": "1.1",
        "scenario_id": scenario_id,
        "label": label,
        "selection_basis": selection_basis,
        "source_stems": source_stems,
        "components": _component_rows(rows),
        "metrics": {split: _combo_metrics(rows, split) for split in ("train", "val", "oos")},
        "combined_streams": {
            split: _weighted_candidate_stream(rows, split) for split in ("train", "val", "oos")
        },
        "notes": notes,
    }


def _scenario_rows(
    bundle: dict[str, Any], decision: dict[str, Any], spec: dict[str, Any]
) -> list[dict[str, Any]]:
    return _deployment_candidate_rows(
        bundle,
        decision,
        stems=list(spec.get("stems") or []),
        watchlist_label=spec.get("watchlist_label"),
        watchlist_role=spec.get("watchlist_role"),
    )


def _iter_source_detail_rows(decision: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in list(decision.get("source_batches") or []):
        details_path = Path(str(source.get("details_path") or "")).resolve()
        if not details_path.exists():
            continue
        try:
            payload = json.loads(details_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if isinstance(row, dict):
                rows.append(dict(row))
    return rows


def _experimental_watchlist_rows(
    decision: dict[str, Any],
    *,
    include_metals: bool | None = None,
    max_rows: int = 3,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in _iter_source_detail_rows(decision):
        train = dict(row.get("train") or {})
        val = dict(row.get("val") or {})
        oos = dict(row.get("oos") or {})
        symbols = list(row.get("symbols") or [])
        has_metals = any(symbol in METALS for symbol in symbols)
        if include_metals is True and not has_metals:
            continue
        if include_metals is False and has_metals:
            continue
        if float(oos.get("pbo", 1.0) or 1.0) > EXPERIMENTAL_MAX_PBO:
            continue
        if float(oos.get("sharpe", 0.0) or 0.0) <= EXPERIMENTAL_MIN_OOS_SHARPE:
            continue
        if float(oos.get("return", 0.0) or 0.0) <= EXPERIMENTAL_MIN_OOS_RETURN:
            continue
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "").lower()
        min_trade_count = EXPERIMENTAL_MIN_TRADE_COUNT
        if timeframe in {"30m", "1h"}:
            min_trade_count = 20.0
        elif timeframe == "4h":
            min_trade_count = 6.0
        if float(oos.get("trade_count", 0.0) or 0.0) < min_trade_count:
            continue
        if float(val.get("sharpe", 0.0) or 0.0) < EXPERIMENTAL_MIN_VAL_SHARPE:
            continue
        if float(train.get("return", 0.0) or 0.0) < EXPERIMENTAL_MIN_TRAIN_RETURN:
            continue

        copied = _decorate_deployment_row(
            row,
            role="experimental watchlist",
            label="research watchlist sleeve",
            stage="experimental_watchlist",
            run_id=None,
            memory_evidence={},
            risk_flags=[
                "research_only",
                "no_strict_anchor",
            ],
        )
        copied["_deployment_score"] = (
            float(oos.get("sharpe", 0.0) or 0.0)
            + (20.0 * float(oos.get("return", 0.0) or 0.0))
            + (4.0 * float(val.get("sharpe", 0.0) or 0.0))
            - (3.0 * float(oos.get("pbo", 1.0) or 1.0))
        )
        ranked.append(copied)

    ranked.sort(
        key=lambda row: (
            float(row.get("_deployment_score", 0.0)),
            float((row.get("oos") or {}).get("sharpe", 0.0) or 0.0),
            float((row.get("oos") or {}).get("return", 0.0) or 0.0),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_timeframes: set[str] = set()
    seen_names: set[str] = set()
    for row in ranked:
        name = str(row.get("name") or "")
        timeframe = str(row.get("strategy_timeframe") or "")
        if name in seen_names:
            continue
        if timeframe in seen_timeframes and len(selected) >= 1:
            continue
        seen_names.add(name)
        seen_timeframes.add(timeframe)
        selected.append(row)
        if len(selected) >= max_rows:
            break
    if not selected:
        return []
    return _with_equal_weights(selected)


def _build_scenarios(
    bundle: dict[str, Any], decision: dict[str, Any], generated_at: str
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for spec in SCENARIO_SPECS:
        rows = _scenario_rows(bundle, decision, spec)
        if not rows:
            continue
        if spec.get("stems") and len(rows) <= len(_strict_anchor_rows(decision)):
            continue
        scenarios.append(
            _build_payload(
                rows,
                generated_at=generated_at,
                scenario_id=str(spec.get("scenario_id")),
                label=str(spec.get("label")),
                selection_basis=str(spec.get("selection_basis") or "equal_weight_experimental"),
                notes=list(spec.get("notes") or []),
                source_stems=list(spec.get("stems") or []),
            )
        )
    if not scenarios:
        experimental_rows = _experimental_watchlist_rows(decision, include_metals=False, max_rows=3)
        if experimental_rows:
            scenarios.append(
                _build_payload(
                    experimental_rows,
                    generated_at=generated_at,
                    scenario_id="experimental_research_watchlist",
                    label="Experimental research watchlist",
                    selection_basis="research_watchlist_equal_weight",
                    notes=[
                        "No strict anchors are currently available.",
                        "This watchlist keeps only candidates with positive validation/OOS, PBO <= 0.40, and sufficient trade count.",
                    ],
                    source_stems=[],
                )
            )
    return scenarios


def _scenario_summary_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        oos = dict((scenario.get("metrics") or {}).get("oos") or {})
        rows.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "label": scenario.get("label"),
                "selection_basis": scenario.get("selection_basis"),
                "component_count": len(list(scenario.get("components") or [])),
                "oos_return": float(oos.get("return", 0.0) or 0.0),
                "oos_sharpe": float(oos.get("sharpe", 0.0) or 0.0),
                "oos_sortino": float(oos.get("sortino", 0.0) or 0.0),
                "oos_calmar": float(oos.get("calmar", 0.0) or 0.0),
                "oos_max_drawdown": float(oos.get("max_drawdown", 0.0) or 0.0),
                "oos_pbo": float(oos.get("pbo", 0.0) or 0.0),
                "oos_trade_count": float(oos.get("trade_count", 0.0) or 0.0),
            }
        )
    return rows


def write_deployment_artifacts(report_root: Path = REPORT_ROOT) -> dict[str, Any]:
    bundle = load_exact_window_bundle(report_root)
    decision = dict(bundle.get("decision") or {})
    generated_at = datetime.now(UTC).isoformat()
    scenarios = _build_scenarios(bundle, decision, generated_at)
    FOLLOWUP_ROOT.mkdir(parents=True, exist_ok=True)
    combo_json_path = FOLLOWUP_ROOT / "deployment_combo_latest.json"
    combo_md_path = FOLLOWUP_ROOT / "deployment_combo_latest.md"
    scenario_json_path = FOLLOWUP_ROOT / "deployment_scenarios_latest.json"
    scenario_md_path = FOLLOWUP_ROOT / "deployment_scenarios_latest.md"

    if not scenarios:
        primary = {
            "generated_at": generated_at,
            "schema_version": "1.1",
            "scenario_id": "unavailable_missing_decision_artifacts",
            "label": "Deployment combo unavailable",
            "selection_basis": "unavailable",
            "source_stems": [],
            "components": [],
            "metrics": {split: _combo_metrics([], split) for split in ("train", "val", "oos")},
            "combined_streams": {split: [] for split in ("train", "val", "oos")},
            "notes": [
                "Primary decision/follow-up artifacts are currently missing or empty.",
                "Rebuild the registry/log archive and regenerate deployment scenarios after the next successful exact-window run.",
            ],
        }
        scenario_payload = {
            "generated_at": generated_at,
            "schema_version": "1.1",
            "primary_scenario_id": primary.get("scenario_id"),
            "scenario_count": 0,
            "summary": [],
            "scenarios": [],
            "notes": list(primary.get("notes") or []),
        }
    else:
        primary = next(
            (
                scenario
                for scenario in scenarios
                if scenario.get("scenario_id") == "strict_plus_crypto_metal_watchlist"
            ),
            scenarios[0],
        )
        scenario_payload = {
            "generated_at": generated_at,
            "schema_version": "1.1",
            "primary_scenario_id": primary.get("scenario_id"),
            "scenario_count": len(scenarios),
            "summary": _scenario_summary_rows(scenarios),
            "scenarios": scenarios,
            "notes": [
                "Scenario matrix compares strict-anchor-only against saved 4h overlay sleeves.",
                "Equal-weight scenario metrics are for research triage and dashboard visibility, not automatic promotion.",
            ],
        }

    combo_json_path.write_text(json.dumps(primary, indent=2, sort_keys=True), encoding="utf-8")
    primary_oos = dict((primary.get("metrics") or {}).get("oos") or {})
    combo_lines = [
        "# deployment combo",
        "",
        f"- generated_at: `{primary.get('generated_at')}`",
        f"- scenario_id: `{primary.get('scenario_id')}`",
        f"- label: `{primary.get('label')}`",
        f"- selection_basis: `{primary.get('selection_basis')}`",
        f"- oos_return: `{float(primary_oos.get('return', 0.0)):.4%}`",
        f"- oos_sharpe: `{float(primary_oos.get('sharpe', 0.0)):.3f}`",
        f"- oos_max_drawdown: `{float(primary_oos.get('max_drawdown', 0.0)):.4%}`",
        f"- oos_trade_count: `{int(float(primary_oos.get('trade_count', 0.0) or 0.0))}`",
        f"- oos_pbo: `{float(primary_oos.get('pbo', 0.0)):.3f}`",
        "",
        "## components",
    ]
    for row in list(primary.get("components") or []):
        component_oos = dict(row.get("oos") or {})
        combo_lines.append(
            f"- {row.get('label')}: `{row.get('name')}` | tf={row.get('timeframe')} | weight={float(row.get('weight', 0.0)):.2%} | "
            f"oos_return={float(component_oos.get('return', 0.0)):.4%} | "
            f"oos_sharpe={float(component_oos.get('sharpe', 0.0)):.3f} | "
            f"oos_pbo={float(component_oos.get('pbo', 0.0)):.3f}"
        )
    combo_md_path.write_text("\n".join(combo_lines) + "\n", encoding="utf-8")

    scenario_json_path.write_text(
        json.dumps(scenario_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    scenario_lines = [
        "# deployment scenarios",
        "",
        f"- generated_at: `{generated_at}`",
        f"- primary_scenario_id: `{primary.get('scenario_id')}`",
        f"- scenario_count: `{int(scenario_payload.get('scenario_count') or 0)}`",
        "",
        "## scenario summary",
    ]
    for row in list(scenario_payload.get("summary") or []):
        scenario_lines.append(
            f"- {row.get('label')} (`{row.get('scenario_id')}`): return={float(row.get('oos_return', 0.0)):.4%} | "
            f"sharpe={float(row.get('oos_sharpe', 0.0)):.3f} | "
            f"max_dd={float(row.get('oos_max_drawdown', 0.0)):.4%} | "
            f"pbo={float(row.get('oos_pbo', 0.0)):.3f} | "
            f"trades={int(float(row.get('oos_trade_count', 0.0) or 0.0))}"
        )
    scenario_md_path.write_text("\n".join(scenario_lines) + "\n", encoding="utf-8")

    return {
        "generated_at": generated_at,
        "primary": {
            "scenario_id": primary.get("scenario_id"),
            "json_path": str(combo_json_path.resolve()),
            "md_path": str(combo_md_path.resolve()),
            "oos_metrics": primary.get("metrics", {}).get("oos"),
            "component_count": len(list(primary.get("components") or [])),
        },
        "scenarios": {
            "json_path": str(scenario_json_path.resolve()),
            "md_path": str(scenario_md_path.resolve()),
            "scenario_count": int(scenario_payload.get("scenario_count") or 0),
        },
    }


def write_deployment_combo_artifact(report_root: Path = REPORT_ROOT) -> dict[str, Any]:
    result = write_deployment_artifacts(report_root)
    return dict(result.get("primary") or {})


if __name__ == "__main__":
    result = write_deployment_artifacts()
    print(json.dumps(result, indent=2, sort_keys=True))
