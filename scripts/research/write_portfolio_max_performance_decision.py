"""Write the canonical max-performance promotion decision artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lumina_quant.portfolio_followup_rules import (
    ROBUST_PROMOTION_GATES,
    evaluate_robustness_gates,
    multiple_comparison_delta_floor,
    promotion_gross_exposure,
)

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_ROOT = ROOT / "var" / "reports" / "exact_window_backtests" / "followup_status"
DEFAULT_INCUMBENT_BUNDLE = FOLLOWUP_ROOT / "portfolio_one_shot_incumbent_bundle_latest.json"
DEFAULT_INCUMBENT_PORTFOLIO = (
    FOLLOWUP_ROOT / "portfolio_one_shot_current_opt" / "portfolio_optimization_latest.json"
)
DEFAULT_TUNED_COMPARISON = FOLLOWUP_ROOT / "portfolio_comparison_latest.json"
DEFAULT_DYNAMIC_COMPARISON = FOLLOWUP_ROOT / "portfolio_dynamic_comparison_latest.json"
DEFAULT_OVERLAY_COMPARISON = FOLLOWUP_ROOT / "portfolio_overlay_comparison_latest.json"
DEFAULT_REGIME_SWITCH_COMPARISON = FOLLOWUP_ROOT / "portfolio_regime_switch_comparison_latest.json"
DEFAULT_GROUPED_ALLOCATOR = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "three_way_market_regime_allocator_current"
    / "three_way_market_regime_allocator_latest.json"
)
DEFAULT_GROUPED_STRICT_VALIDATION = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_allocator_strict_leverage_validation_current"
    / "grouped_allocator_strict_leverage_validation_latest.json"
)
DEFAULT_GROUPED_STATIC_BLEND = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_incumbent_autoresearch_static_blend_latest.json"
)
DEFAULT_PRODUCTION_GUARDED = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "portfolio_production_guarded_current"
    / "production_guarded_portfolio_latest.json"
)
DEFAULT_PORTFOLIO_SUPERIORITY_META = (
    FOLLOWUP_ROOT
    / "portfolio_superiority_meta_search_current"
    / "portfolio_superiority_meta_portfolio_latest.json"
)
DEFAULT_BACKBONE_TRIPLET = FOLLOWUP_ROOT / "portfolio_backbone_triplet_search_latest.json"
DEFAULT_ANCHORED_COMPARISON = FOLLOWUP_ROOT / "portfolio_four_sleeve_comparison_latest.json"
DEFAULT_META_SEARCH_DIR = FOLLOWUP_ROOT / "portfolio_superiority_meta_search"
DEFAULT_META_SEARCH_SUMMARIES = (
    DEFAULT_META_SEARCH_DIR / "u1_raw_basis_summary_latest.json",
    DEFAULT_META_SEARCH_DIR / "u2_derived_basis_summary_latest.json",
)
DEFAULT_OUTPUT_JSON = FOLLOWUP_ROOT / "portfolio_max_performance_decision_latest.json"
DEFAULT_OUTPUT_MD = FOLLOWUP_ROOT / "portfolio_max_performance_decision_latest.md"
SPLIT_CONTRACT_PATH = ROOT / "src" / "lumina_quant" / "portfolio_split_contract.py"

_DEFAULT_SPLIT_CONTRACT = {
    "train_start": "2025-01-01T00:00:00Z",
    "train_end_exclusive": "2026-01-01T00:00:00Z",
    "val_start": "2026-01-01T00:00:00Z",
    "val_end_exclusive": "2026-02-01T00:00:00Z",
    "oos_start": "2026-02-01T00:00:00Z",
}
_REQUIRED_SPLIT_KEYS = tuple(_DEFAULT_SPLIT_CONTRACT.keys())
PROMOTION_FORMULA = (
    "(1.0 * oos_sharpe) + (0.35 * oos_sortino) + (0.10 * oos_calmar) + "
    "(10.0 * oos_total_return) - (4.0 * oos_max_drawdown) - (0.75 * oos_volatility)"
)
PROMOTION_THRESHOLDS = dict(ROBUST_PROMOTION_GATES)
_REJECTION_REASON_LABELS = {
    "train_total_return_non_positive": "train total return did not stay above 0",
    "val_total_return_non_positive": "validation total return did not stay above 0",
    "train_sharpe_below_floor": "train Sharpe fell below -0.10",
    "oos_total_return_not_above_incumbent": "locked OOS total return did not beat the incumbent",
    "oos_monthly_mean_below_floor": "locked OOS monthly mean stayed below 2%",
    "oos_drawdown_worse_without_sharpe_relief": "drawdown worsened without a 0.50 Sharpe relief",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _maybe_split_payload(module: Any) -> dict[str, Any] | None:
    for attr_name in (
        "split_contract_payload",
        "build_split_contract_payload",
        "get_portfolio_split_contract",
    ):
        attr = getattr(module, attr_name, None)
        if callable(attr):
            candidate = attr()
            if isinstance(candidate, dict):
                return dict(candidate)
    for attr_name in ("SPLIT_CONTRACT", "PORTFOLIO_SPLIT_CONTRACT"):
        candidate = getattr(module, attr_name, None)
        if isinstance(candidate, dict):
            return dict(candidate)
    if all(hasattr(module, key.upper()) for key in _REQUIRED_SPLIT_KEYS):
        return {key: getattr(module, key.upper()) for key in _REQUIRED_SPLIT_KEYS}
    return None


def _resolve_split_contract() -> dict[str, Any]:
    payload = {**_DEFAULT_SPLIT_CONTRACT, "source": "local_default_constants"}
    if not SPLIT_CONTRACT_PATH.exists():
        return payload

    spec = importlib.util.spec_from_file_location("portfolio_split_contract", SPLIT_CONTRACT_PATH)
    if spec is None or spec.loader is None:
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload
    candidate = _maybe_split_payload(module)
    if not isinstance(candidate, dict):
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload

    resolved = dict(payload)
    for key in _REQUIRED_SPLIT_KEYS:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = value.strip()
    resolved["source"] = str(SPLIT_CONTRACT_PATH.resolve())
    return resolved


def _maybe_split_contract_helper(name: str) -> Any:
    if not SPLIT_CONTRACT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("portfolio_split_contract", SPLIT_CONTRACT_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, name, None)


def _resolve_incumbent_bundle_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    resolver = _maybe_split_contract_helper("resolve_incumbent_bundle_path")
    if callable(resolver):
        try:
            return Path(resolver(candidate)).resolve()
        except Exception:
            return candidate.resolve()
    return candidate.resolve()


def _resolve_incumbent_portfolio_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    resolver = _maybe_split_contract_helper("resolve_current_optimization_path")
    if callable(resolver):
        try:
            return Path(resolver(candidate)).resolve()
        except Exception:
            return candidate.resolve()
    return candidate.resolve()


def _extract_split_metrics(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    for key in ("portfolio_metrics", "split_metrics", "metrics"):
        metric_block = payload.get(key)
        if not isinstance(metric_block, dict):
            continue
        for split in ("train", "val", "oos"):
            split_payload = metric_block.get(split)
            if isinstance(split_payload, dict):
                extracted[split] = dict(split_payload)
    for split in ("train", "val", "oos"):
        if split not in extracted and isinstance(payload.get(split), dict):
            extracted[split] = dict(payload.get(split) or {})
    return extracted


def _extract_weights(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("weights", "final_allocation"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _grouped_allocator_weights(payload: dict[str, Any]) -> list[dict[str, Any]]:
    current = dict(payload.get("current_state") or {})
    weights = dict(current.get("weights") or {})
    if not weights:
        return []
    return [
        {
            "candidate_id": str(name),
            "name": str(name),
            "weight": _safe_float(value, 0.0),
        }
        for name, value in weights.items()
        if _safe_float(value, 0.0) > 0.0
    ]


def _strict_liquidation_total(strict_payload: dict[str, Any]) -> int:
    validation = dict(
        (strict_payload.get("strict_allocator") or {}).get("state_leverage_validation") or {}
    )
    counts = dict(validation.get("liquidation_counts") or {})
    return int(sum(int(value) for value in counts.values()))


def _strict_grouped_allocator_entry(
    *,
    grouped_allocator_path: Path,
    grouped_payload: dict[str, Any],
    strict_validation_path: Path | None,
    strict_validation_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    current_state = dict(grouped_payload.get("current_state") or {})
    state_summary = dict(grouped_payload.get("state_summary") or {})
    notes = [
        "Grouped three-way allocator challenger over incumbent / blend_85_15 / autoresearch_55_45.",
        (
            "Current state="
            f"{current_state.get('state')} raw_target={current_state.get('raw_target_state')} "
            f"confidence={_safe_float(current_state.get('confidence'), 0.0):.4f}"
        ),
        (
            "State usage oos="
            f"{json.dumps(dict((state_summary.get('oos') or {}).get('counts') or {}), sort_keys=True)}"
        ),
    ]

    payload_for_entry = dict(grouped_payload)
    artifact_path = grouped_allocator_path
    selection_basis = str(
        grouped_payload.get("selection_basis") or "grouped_three_way_market_regime_allocator"
    )
    source_artifact_kind = (
        "portfolio_incumbent_autoresearch_grouped.three_way_market_regime_allocator"
    )

    if strict_validation_payload is not None:
        strict_allocator = dict(strict_validation_payload.get("strict_allocator") or {})
        if strict_allocator:
            payload_for_entry = dict(strict_allocator)
            payload_for_entry["weights"] = _grouped_allocator_weights(strict_allocator)
            payload_for_entry["artifact_kind"] = str(
                strict_validation_payload.get("artifact_kind")
                or payload_for_entry.get("artifact_kind")
                or "grouped_allocator_strict_validation"
            )
            artifact_path = strict_validation_path or grouped_allocator_path
            selection_basis = "strict_grouped_allocator_state_leverage_validation"
            source_artifact_kind = "portfolio_incumbent_autoresearch_grouped.grouped_allocator_strict_leverage_validation"
            notes.append(
                "Decision metrics are sourced from the strict grouped allocator validation artifact, not the proxy allocator artifact."
            )
            notes.append(
                f"Strict leverage_by_state={json.dumps(dict(strict_validation_payload.get('leverage_by_state') or {}), sort_keys=True)}"
            )
            notes.append(
                f"Strict liquidation_count={_strict_liquidation_total(strict_validation_payload)}"
            )

    entry = _artifact_entry(
        candidate_key="grouped_three_way_market_regime_allocator",
        label="Grouped three-way allocator challenger",
        artifact_path=artifact_path,
        payload=payload_for_entry,
        source_artifact_kind=source_artifact_kind,
        selection_basis=selection_basis,
        notes=notes,
    )
    if strict_validation_payload is not None:
        entry["strict_validation"] = {
            "path": str((strict_validation_path or grouped_allocator_path).resolve()),
            "leverage_by_state": dict(strict_validation_payload.get("leverage_by_state") or {}),
            "comparison_vs_promoted_challenger": dict(
                strict_validation_payload.get("comparison_vs_promoted_challenger") or {}
            ),
            "state_leverage_validation": dict(
                (strict_validation_payload.get("strict_allocator") or {}).get(
                    "state_leverage_validation"
                )
                or {}
            ),
            "split_metrics": dict(
                (strict_validation_payload.get("strict_allocator") or {}).get("split_metrics") or {}
            ),
        }
    return entry


def _promotion_score(
    metrics: dict[str, Any],
    *,
    gross_exposure: float = 1.0,
    exposure_normalized: bool = False,
) -> float:
    if not exposure_normalized:
        return (
            (1.0 * _safe_float(metrics.get("sharpe"), 0.0))
            + (0.35 * _safe_float(metrics.get("sortino"), 0.0))
            + (0.10 * _safe_float(metrics.get("calmar"), 0.0))
            + (10.0 * _safe_float(metrics.get("total_return", metrics.get("return")), 0.0))
            - (4.0 * _safe_float(metrics.get("max_drawdown", metrics.get("mdd")), 0.0))
            - (0.75 * _safe_float(metrics.get("volatility"), 0.0))
        )
    # Exposure-normalized objective: divide the scale-dependent return/drawdown/
    # volatility terms by gross exposure so a levered candidate cannot buy
    # "superiority".  Sharpe/Sortino/Calmar are already scale-free ratios.
    divisor = gross_exposure if gross_exposure and gross_exposure > 0.0 else 1.0
    return (
        (1.0 * _safe_float(metrics.get("sharpe"), 0.0))
        + (0.35 * _safe_float(metrics.get("sortino"), 0.0))
        + (0.10 * _safe_float(metrics.get("calmar"), 0.0))
        + ((10.0 * _safe_float(metrics.get("total_return", metrics.get("return")), 0.0)) / divisor)
        - ((4.0 * _safe_float(metrics.get("max_drawdown", metrics.get("mdd")), 0.0)) / divisor)
        - ((0.75 * _safe_float(metrics.get("volatility"), 0.0)) / divisor)
    )


def _artifact_entry(
    *,
    candidate_key: str,
    label: str,
    artifact_path: Path,
    payload: dict[str, Any],
    source_artifact_kind: str,
    selection_basis: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    split_metrics = _extract_split_metrics(payload)
    return {
        "candidate_key": candidate_key,
        "label": label,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_kind": str(payload.get("artifact_kind") or source_artifact_kind),
        "source_artifact_kind": source_artifact_kind,
        "selection_basis": selection_basis,
        "train": dict(split_metrics.get("train") or {}),
        "val": dict(split_metrics.get("val") or {}),
        "oos": dict(split_metrics.get("oos") or {}),
        "weights": _extract_weights(payload),
        "portfolio_return_streams": dict(payload.get("portfolio_return_streams") or {}),
        "portfolio_daily_return_streams": dict(payload.get("portfolio_daily_return_streams") or {}),
        "oos_monthly_returns": [
            dict(row)
            for row in list(payload.get("oos_monthly_returns") or [])
            if isinstance(row, dict)
        ],
        "validation_objective": payload.get("validation_objective"),
        "notes": list(notes or []),
    }


def _normalized_notes(value: Any) -> list[str]:
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _meta_search_entries(summary_paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    existing_summary_count = sum(1 for path in summary_paths if path.exists())
    for summary_path in summary_paths:
        if not summary_path.exists():
            continue
        summary_payload = _load_json(summary_path)
        winner = dict(summary_payload.get("winner") or {})
        if not winner or not list(winner.get("weights") or []):
            continue
        if str(summary_payload.get("winner_status") or "") != "promoted_challenger":
            continue
        universe_name = str(summary_payload.get("universe_name") or summary_path.stem)
        notes = [
            f"Basis-deduped portfolio-superiority meta-search winner from {universe_name}.",
            f"Selection basis: {summary_payload.get('selection_basis') or 'validation_objective_then_locked_oos'}",
        ]
        rejection_reasons = list(winner.get("rejection_reasons") or [])
        if rejection_reasons:
            notes.append(f"Winner rejection reasons: {json.dumps(rejection_reasons)}")
        candidate_key = (
            "portfolio_superiority_meta_portfolio"
            if existing_summary_count == 1
            else f"portfolio_superiority_meta_{universe_name}"
        )
        winner_payload = {
            **winner,
            "artifact_kind": "portfolio_superiority_meta_search_winner",
        }
        if not list(winner_payload.get("oos_monthly_returns") or []):
            oos_metrics = dict(winner_payload.get("oos") or {})
            oos_return = _safe_float(
                oos_metrics.get("total_return", oos_metrics.get("return")),
                0.0,
            )
            if oos_return > 0.0:
                winner_payload["oos_monthly_returns"] = [
                    {
                        "month": "locked_oos_summary",
                        "total_return": oos_return,
                        "source": "meta_search_summary_winner_oos_total_return",
                    }
                ]
        entries.append(
            _artifact_entry(
                candidate_key=candidate_key,
                label=f"Portfolio superiority meta search ({universe_name})",
                artifact_path=summary_path,
                payload=winner_payload,
                source_artifact_kind="portfolio_superiority_meta_search.winner",
                selection_basis=str(
                    summary_payload.get("selection_basis") or "validation_objective_then_locked_oos"
                ),
                notes=notes,
            )
        )
    return entries


def _extra_candidate_entries(extra_paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for artifact_path in extra_paths:
        if not artifact_path.exists():
            continue
        payload = _load_json(artifact_path)
        candidate_key = str(payload.get("candidate_key") or artifact_path.stem).strip()
        label = str(payload.get("label") or candidate_key).strip()
        source_artifact_kind = str(
            payload.get("source_artifact_kind")
            or payload.get("artifact_kind")
            or "portfolio_followup.custom_candidate"
        ).strip()
        selection_basis = str(payload.get("selection_basis") or "manual_followup_candidate").strip()
        notes = _normalized_notes(payload.get("notes"))
        if not notes:
            notes = ["Manually supplied follow-up challenger artifact."]
        entries.append(
            _artifact_entry(
                candidate_key=candidate_key,
                label=label,
                artifact_path=artifact_path,
                payload=payload,
                source_artifact_kind=source_artifact_kind,
                selection_basis=selection_basis,
                notes=notes,
            )
        )
    return entries


def _challenger_reason(entry: dict[str, Any]) -> str:
    if entry.get("promotable"):
        return (
            "Promotable because it cleared the train/validation robustness gates, beat the "
            "incumbent on locked OOS return, and satisfied the drawdown-or-sharpe relief rule."
        )

    reasons = [
        _REJECTION_REASON_LABELS.get(reason, str(reason).replace("_", " "))
        for reason in list(entry.get("rejection_reasons") or [])
    ]
    if not reasons:
        reasons.append("incumbent tie policy kept the current one-shot leader")
    return "; ".join(reasons)


def _strict_grouped_gate_failures(
    *,
    entry: dict[str, Any],
    incumbent_oos: dict[str, Any],
) -> list[str]:
    strict = dict(entry.get("strict_validation") or {})
    metrics = dict(strict.get("split_metrics") or {})
    if not metrics:
        return []

    failures: list[str] = []
    train = dict(metrics.get("train") or {})
    val = dict(metrics.get("val") or {})
    oos = dict(metrics.get("oos") or {})
    liquidation_counts = dict(
        (strict.get("state_leverage_validation") or {}).get("liquidation_counts") or {}
    )
    total_liquidations = int(sum(int(value) for value in liquidation_counts.values()))

    if _safe_float(train.get("total_return"), 0.0) <= 0.0:
        failures.append("strict train total return is not positive")
    if _safe_float(val.get("total_return"), 0.0) <= 0.0:
        failures.append("strict validation total return is not positive")
    if total_liquidations > 0:
        failures.append(f"strict leverage replay recorded {total_liquidations} liquidations")
    if _safe_float(oos.get("sharpe"), 0.0) < _safe_float(incumbent_oos.get("sharpe"), 0.0):
        failures.append("strict OOS sharpe does not beat the incumbent")
    return failures


def _decorate_vs_incumbent(
    entries: list[dict[str, Any]],
    *,
    exposure_normalized: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not entries:
        raise RuntimeError("no candidate entries available for decision bundle")

    incumbent = dict(entries[0])
    incumbent_oos = dict(incumbent.get("oos") or {})
    incumbent_gross = promotion_gross_exposure(incumbent) if exposure_normalized else 1.0
    incumbent_score = _promotion_score(
        incumbent_oos, gross_exposure=incumbent_gross, exposure_normalized=exposure_normalized
    )
    incumbent["promotion_score"] = incumbent_score
    incumbent["promotion_score_delta"] = 0.0
    incumbent["oos_total_return_delta"] = 0.0
    incumbent["oos_max_drawdown_delta"] = 0.0
    incumbent["oos_sharpe_delta"] = 0.0
    incumbent["promotable"] = False
    incumbent["decision_reason"] = "Current one-shot incumbent baseline."

    decorated = [incumbent]
    for raw in entries[1:]:
        entry = dict(raw)
        oos = dict(entry.get("oos") or {})
        if exposure_normalized:
            candidate_gross = promotion_gross_exposure(entry)
            score = _promotion_score(oos, gross_exposure=candidate_gross, exposure_normalized=True)
            gate_result = evaluate_robustness_gates(
                entry,
                incumbent,
                exposure_normalized_promotion=True,
                candidate_gross_exposure=candidate_gross,
                incumbent_gross_exposure=incumbent_gross,
            )
        else:
            score = _promotion_score(oos)
            gate_result = evaluate_robustness_gates(entry, incumbent)
        score_delta = score - incumbent_score
        entry["promotion_score"] = score
        entry["promotion_score_delta"] = score_delta
        entry.update(gate_result)
        gate_failures = _strict_grouped_gate_failures(entry=entry, incumbent_oos=incumbent_oos)
        entry["strict_gate_failures"] = list(gate_failures)
        entry["promotable"] = bool(entry.get("promotable") and not gate_failures)
        entry["decision_reason"] = _challenger_reason(entry)
        if gate_failures:
            entry["decision_reason"] = (
                f"{entry['decision_reason']}; strict gate rejected candidate because "
                + "; ".join(gate_failures)
            )
        decorated.append(entry)
    return decorated, incumbent


def build_portfolio_max_performance_decision(
    *,
    incumbent_bundle_path: Path | str = DEFAULT_INCUMBENT_BUNDLE,
    incumbent_portfolio_path: Path | str = DEFAULT_INCUMBENT_PORTFOLIO,
    tuned_comparison_path: Path | str = DEFAULT_TUNED_COMPARISON,
    dynamic_comparison_path: Path | str = DEFAULT_DYNAMIC_COMPARISON,
    overlay_comparison_path: Path | str = DEFAULT_OVERLAY_COMPARISON,
    regime_switch_comparison_path: Path | str = DEFAULT_REGIME_SWITCH_COMPARISON,
    grouped_allocator_path: Path | str | None = None,
    grouped_strict_validation_path: Path | str | None = DEFAULT_GROUPED_STRICT_VALIDATION,
    grouped_static_blend_path: Path | str | None = DEFAULT_GROUPED_STATIC_BLEND,
    production_guarded_path: Path | str | None = None,
    portfolio_superiority_meta_path: Path | str | None = None,
    backbone_triplet_path: Path | str = DEFAULT_BACKBONE_TRIPLET,
    anchored_comparison_path: Path | str = DEFAULT_ANCHORED_COMPARISON,
    anchored_tuned_comparison_path: Path | str | None = None,
    portfolio_four_sleeve_comparison_path: Path | str | None = None,
    four_sleeve_comparison_path: Path | str | None = None,
    meta_search_summary_paths: tuple[Path | str, ...] = (),
    extra_candidate_artifact_paths: tuple[Path | str, ...] = (),
    research_config: Any = None,
) -> dict[str, Any]:
    exposure_normalized = bool(getattr(research_config, "exposure_normalized_promotion", False))
    resolved_incumbent_bundle_path = _resolve_incumbent_bundle_path(Path(incumbent_bundle_path))
    resolved_incumbent_portfolio_path = _resolve_incumbent_portfolio_path(
        Path(incumbent_portfolio_path)
    )
    incumbent_bundle = _load_json(resolved_incumbent_bundle_path)
    incumbent_portfolio = _load_json(resolved_incumbent_portfolio_path)
    split_contract = _resolve_split_contract()

    entries = [
        _artifact_entry(
            candidate_key="current_one_shot_incumbent",
            label="Current one-shot incumbent",
            artifact_path=resolved_incumbent_portfolio_path,
            payload=incumbent_portfolio,
            source_artifact_kind="portfolio_one_shot_current_opt",
            selection_basis=str(
                incumbent_bundle.get("selection_basis") or "incumbent_saved_one_shot_weights"
            ),
            notes=[
                "Baseline incumbent for the max-performance loop.",
                f"Restricted sleeve bundle: {resolved_incumbent_bundle_path}",
            ],
        )
    ]
    supporting_artifacts = {
        "incumbent_bundle": str(resolved_incumbent_bundle_path),
        "incumbent_portfolio": str(resolved_incumbent_portfolio_path),
    }
    missing_artifacts: list[str] = []
    fallback_retune_required = False

    tuned_path = Path(tuned_comparison_path)
    if tuned_path.exists():
        tuned_comparison = _load_json(tuned_path)
        supporting_artifacts["portfolio_comparison"] = str(tuned_path.resolve())
        tuned_section = dict(tuned_comparison.get("exact_window_frozen_tuned") or {})
        if tuned_section:
            entries.append(
                _artifact_entry(
                    candidate_key="exact_window_frozen_tuned",
                    label="Exact-window frozen tuned challenger",
                    artifact_path=Path(tuned_section.get("path") or tuned_path),
                    payload=tuned_section,
                    source_artifact_kind="portfolio_comparison.exact_window_frozen_tuned",
                    selection_basis=str(
                        tuned_comparison.get("selection_basis") or "validation_only"
                    ),
                    notes=[
                        "Validation-ranked frozen challenger from the exact-window sleeve freeze path."
                    ],
                )
            )
    else:
        missing_artifacts.append(str(tuned_path.resolve()))

    triplet_path = Path(backbone_triplet_path)
    if triplet_path.exists():
        triplet_payload = _load_json(triplet_path)
        supporting_artifacts["portfolio_backbone_triplet_search"] = str(triplet_path.resolve())
        entries.append(
            _artifact_entry(
                candidate_key="backbone_preserving_triplet_search",
                label="Backbone-preserving triplet search challenger",
                artifact_path=triplet_path,
                payload=triplet_payload,
                source_artifact_kind="portfolio_backbone_triplet_search",
                selection_basis="validation_objective_then_locked_oos",
                notes=["Same-family backbone-preserving challenger."],
            )
        )
    else:
        missing_artifacts.append(str(triplet_path.resolve()))

    dynamic_path = Path(dynamic_comparison_path)
    if dynamic_path.exists():
        dynamic_comparison = _load_json(dynamic_path)
        supporting_artifacts["portfolio_dynamic_comparison"] = str(dynamic_path.resolve())
        dynamic_section = dict(dynamic_comparison.get("causal_dynamic_portfolio") or {})
        if dynamic_section:
            entries.append(
                _artifact_entry(
                    candidate_key="causal_dynamic_portfolio",
                    label="Causal dynamic challenger",
                    artifact_path=Path(dynamic_section.get("path") or dynamic_path),
                    payload=dynamic_section,
                    source_artifact_kind="portfolio_dynamic_comparison.causal_dynamic_portfolio",
                    selection_basis=str(
                        dynamic_comparison.get("selection_basis") or "validation_only"
                    ),
                    notes=["Dynamic allocation challenger evaluated after incumbent-first lanes."],
                )
            )
    else:
        missing_artifacts.append(str(dynamic_path.resolve()))

    overlay_path = Path(overlay_comparison_path)
    if overlay_path.exists():
        overlay_comparison = _load_json(overlay_path)
        supporting_artifacts["portfolio_overlay_comparison"] = str(overlay_path.resolve())
        overlay_section = dict(overlay_comparison.get("causal_overlay_portfolio") or {})
        if overlay_section:
            entries.append(
                _artifact_entry(
                    candidate_key="causal_overlay_portfolio",
                    label="Causal overlay challenger",
                    artifact_path=Path(overlay_section.get("path") or overlay_path),
                    payload=overlay_section,
                    source_artifact_kind="portfolio_overlay_comparison.causal_overlay_portfolio",
                    selection_basis=str(
                        overlay_comparison.get("selection_basis") or "validation_only"
                    ),
                    notes=["Light overlay challenger evaluated on the incumbent backbone."],
                )
            )
    else:
        missing_artifacts.append(str(overlay_path.resolve()))

    regime_switch_path = Path(regime_switch_comparison_path)
    if regime_switch_path.exists():
        regime_switch_comparison = _load_json(regime_switch_path)
        supporting_artifacts["portfolio_regime_switch_comparison"] = str(
            regime_switch_path.resolve()
        )
        regime_switch_section = dict(
            regime_switch_comparison.get("regime_switching_portfolio") or {}
        )
        if regime_switch_section:
            weights = list(regime_switch_section.get("weights") or [])
            nonzero_weights = [
                row for row in weights if _safe_float((row or {}).get("weight"), 0.0) > 0.0
            ]
            raw_regime_selection_basis = str(
                regime_switch_section.get("selection_basis")
                or regime_switch_comparison.get("selection_basis")
                or ""
            ).strip()
            regime_selection_basis = (
                raw_regime_selection_basis
                if raw_regime_selection_basis
                and raw_regime_selection_basis != "locked_oos_promotion_score"
                else "regime_switching_allocator"
            )
            notes = [
                "Low-memory regime-switch allocator challenger evaluated on the locked OOS split."
            ]
            if nonzero_weights:
                weight_summary = ", ".join(
                    f"{row.get('name') or row.get('candidate_id') or 'candidate'!s}={_safe_float(row.get('weight'), 0.0):.2f}"
                    for row in nonzero_weights
                )
                notes.append(f"Active final weights: {weight_summary}")
            entries.append(
                _artifact_entry(
                    candidate_key="regime_switching_portfolio",
                    label="Regime-switch allocator challenger",
                    artifact_path=Path(regime_switch_section.get("path") or regime_switch_path),
                    payload=regime_switch_section,
                    source_artifact_kind="portfolio_regime_switch_comparison.regime_switching_portfolio",
                    selection_basis=regime_selection_basis,
                    notes=notes,
                )
            )
    else:
        missing_artifacts.append(str(regime_switch_path.resolve()))

    grouped_allocator = Path(grouped_allocator_path) if grouped_allocator_path is not None else None
    grouped_strict_validation = (
        Path(grouped_strict_validation_path) if grouped_strict_validation_path is not None else None
    )
    if grouped_allocator is not None and grouped_allocator.exists():
        grouped_payload = _load_json(grouped_allocator)
        supporting_artifacts["portfolio_incumbent_autoresearch_grouped_allocator"] = str(
            grouped_allocator.resolve()
        )
        strict_validation_payload = None
        if grouped_strict_validation is not None and grouped_strict_validation.exists():
            strict_validation_payload = _load_json(grouped_strict_validation)
            supporting_artifacts["portfolio_incumbent_autoresearch_grouped_strict_validation"] = (
                str(grouped_strict_validation.resolve())
            )
        elif grouped_strict_validation is not None:
            missing_artifacts.append(str(grouped_strict_validation.resolve()))
        entries.append(
            _strict_grouped_allocator_entry(
                grouped_allocator_path=grouped_allocator,
                grouped_payload=grouped_payload,
                strict_validation_path=grouped_strict_validation,
                strict_validation_payload=strict_validation_payload,
            )
        )
    elif grouped_allocator is not None:
        missing_artifacts.append(str(grouped_allocator.resolve()))

    grouped_static_blend = (
        Path(grouped_static_blend_path) if grouped_static_blend_path is not None else None
    )
    if grouped_static_blend is not None and grouped_static_blend.exists():
        grouped_static_blend_payload = _load_json(grouped_static_blend)
        supporting_artifacts["portfolio_incumbent_autoresearch_grouped_static_blend"] = str(
            grouped_static_blend.resolve()
        )
        entries.append(
            _artifact_entry(
                candidate_key="incumbent_autoresearch_static_blend",
                label="Incumbent/55-45 static blend challenger",
                artifact_path=grouped_static_blend,
                payload=grouped_static_blend_payload,
                source_artifact_kind="portfolio_incumbent_autoresearch_grouped.grouped_static_blend",
                selection_basis=str(
                    grouped_static_blend_payload.get("selection_basis")
                    or "validation_only_static_group_blend"
                ),
                notes=[
                    "Static blend between the current incumbent and the autoresearch 55/45 sleeve.",
                    (
                        "Blend weights="
                        f"{json.dumps(dict(grouped_static_blend_payload.get('best_weights') or {}), sort_keys=True)}"
                    ),
                ],
            )
        )
    elif grouped_static_blend is not None:
        missing_artifacts.append(str(grouped_static_blend.resolve()))

    production_guarded = (
        Path(production_guarded_path) if production_guarded_path is not None else None
    )
    if production_guarded is not None and production_guarded.exists():
        production_guarded_payload = _load_json(production_guarded)
        supporting_artifacts["portfolio_production_guarded"] = str(production_guarded.resolve())
        entries.append(
            _artifact_entry(
                candidate_key="production_guarded_portfolio",
                label="Production guarded portfolio challenger",
                artifact_path=production_guarded,
                payload=production_guarded_payload,
                source_artifact_kind="portfolio_incumbent_autoresearch_grouped.production_guarded_portfolio",
                selection_basis=str(
                    production_guarded_payload.get("selection_basis")
                    or "saved_sleeve_blend_with_drawdown_throttle"
                ),
                notes=[
                    "Drawdown-aware production candidate blended from saved hybrid/static/incumbent sleeves.",
                    (
                        "Weights="
                        f"{json.dumps({str((row or {}).get('candidate_id') or (row or {}).get('name')): _safe_float((row or {}).get('weight'), 0.0) for row in list(production_guarded_payload.get('weights') or []) if isinstance(row, dict)}, sort_keys=True)}"
                    ),
                    (
                        "Exposure="
                        f"{_safe_float(production_guarded_payload.get('active_exposure'), 0.0):.2%} "
                        f"cash={_safe_float(production_guarded_payload.get('cash_weight'), 0.0):.2%}"
                    ),
                ],
            )
        )
    elif production_guarded is not None:
        missing_artifacts.append(str(production_guarded.resolve()))

    portfolio_superiority_meta = (
        Path(portfolio_superiority_meta_path)
        if portfolio_superiority_meta_path is not None
        else None
    )
    if portfolio_superiority_meta is not None and portfolio_superiority_meta.exists():
        portfolio_superiority_meta_payload = _load_json(portfolio_superiority_meta)
        supporting_artifacts["portfolio_superiority_meta_search"] = str(
            portfolio_superiority_meta.resolve()
        )
        fallback_retune_required = bool(
            portfolio_superiority_meta_payload.get("fallback_retune_required")
        )
        entries.append(
            _artifact_entry(
                candidate_key="portfolio_superiority_meta_portfolio",
                label="Robust meta-portfolio challenger",
                artifact_path=portfolio_superiority_meta,
                payload=portfolio_superiority_meta_payload,
                source_artifact_kind="portfolio_superiority_meta_search.portfolio_superiority_meta_portfolio",
                selection_basis=str(
                    portfolio_superiority_meta_payload.get("selection_basis")
                    or "validation_objective_then_locked_oos_robust_promotion"
                ),
                notes=[
                    "Deduped saved-stream meta-portfolio challenger with explicit robustness gates.",
                    (
                        "Weights="
                        f"{json.dumps({str((row or {}).get('candidate_id') or (row or {}).get('name')): _safe_float((row or {}).get('weight'), 0.0) for row in list(portfolio_superiority_meta_payload.get('weights') or []) if isinstance(row, dict)}, sort_keys=True)}"
                    ),
                    (f"Universe={portfolio_superiority_meta_payload.get('universe')}"),
                ],
            )
        )
    elif portfolio_superiority_meta is not None:
        missing_artifacts.append(str(portfolio_superiority_meta.resolve()))

    anchored_candidates = [
        anchored_comparison_path,
        anchored_tuned_comparison_path,
        portfolio_four_sleeve_comparison_path,
        four_sleeve_comparison_path,
    ]
    anchored_path = next(
        (
            Path(candidate)
            for candidate in anchored_candidates
            if candidate and Path(candidate).exists()
        ),
        None,
    )
    if anchored_path is not None:
        anchored_comparison = _load_json(anchored_path)
        supporting_artifacts["portfolio_four_sleeve_comparison"] = str(anchored_path.resolve())
        anchored_section = dict(
            anchored_comparison.get("anchored_four_sleeve_tuned")
            or anchored_comparison.get("portfolio_four_sleeve_anchored_tuned")
            or anchored_comparison.get("portfolio_four_sleeve_tuned")
            or {}
        )
        if anchored_section:
            rolling_gate = dict(anchored_comparison.get("rolling_gate") or {})
            metadata = dict(anchored_comparison.get("challenger_metadata") or {})
            for key in (
                "candidate_key",
                "label",
                "source_artifact_kind",
                "selection_basis",
                "notes",
            ):
                if key not in metadata and anchored_section.get(key) is not None:
                    metadata[key] = anchored_section.get(key)
            candidate_key = str(metadata.get("candidate_key") or "anchored_four_sleeve_tuned")
            label = str(metadata.get("label") or "Anchored four-sleeve tuned challenger")
            source_artifact_kind = str(
                metadata.get("source_artifact_kind")
                or "portfolio_four_sleeve_comparison.anchored_four_sleeve_tuned"
            )
            selection_basis = str(
                metadata.get("selection_basis")
                or anchored_comparison.get("selection_basis")
                or "incumbent_anchor_rolling_gate"
            )
            notes = _normalized_notes(metadata.get("notes"))
            if not notes:
                notes = ["Incumbent-aware four-sleeve challenger with RollingBreakout gate."]
            if rolling_gate:
                notes.append(
                    "Rolling gate: "
                    f"selection_basis={rolling_gate.get('selection_basis')} "
                    f"survives_train_val={rolling_gate.get('survives_train_val')}"
                )
            entries.append(
                _artifact_entry(
                    candidate_key=candidate_key,
                    label=label,
                    artifact_path=Path(anchored_section.get("path") or anchored_path),
                    payload=anchored_section,
                    source_artifact_kind=source_artifact_kind,
                    selection_basis=selection_basis,
                    notes=notes,
                )
            )
    else:
        missing_artifacts.append(str(Path(anchored_comparison_path).resolve()))

    meta_search_paths = tuple(Path(path) for path in meta_search_summary_paths)
    meta_entries = _meta_search_entries(meta_search_paths)
    if meta_entries:
        supporting_artifacts["portfolio_superiority_meta_search"] = [
            str(path.resolve()) for path in meta_search_paths if path.exists()
        ]
        entries.extend(meta_entries)

    extra_candidate_paths = tuple(Path(path) for path in extra_candidate_artifact_paths)
    extra_entries = _extra_candidate_entries(extra_candidate_paths)
    if extra_entries:
        supporting_artifacts["extra_candidate_artifacts"] = [
            str(path.resolve()) for path in extra_candidate_paths if path.exists()
        ]
        entries.extend(extra_entries)

    decorated, incumbent = _decorate_vs_incumbent(entries, exposure_normalized=exposure_normalized)
    challengers = [dict(entry) for entry in decorated[1:]]
    promotable = [entry for entry in challengers if bool(entry.get("promotable"))]
    multiplicity_floor = 0.0
    if exposure_normalized and promotable:
        # Multiple-comparison-aware floor on the exposure-normalized superiority
        # argmax: with K challengers ranked against the same incumbent, require
        # the winner's delta to clear more than the raw 0.0 gate.
        multiplicity_floor = multiple_comparison_delta_floor(
            [_safe_float(entry.get("promotion_score_delta"), 0.0) for entry in challengers],
            challenger_count=len(challengers),
        )
        promotable = [
            entry
            for entry in promotable
            if _safe_float(entry.get("promotion_score_delta"), 0.0) > multiplicity_floor
        ]
    if promotable:
        winner = max(
            promotable,
            key=lambda entry: (
                _safe_float(entry.get("promotion_score_delta"), 0.0),
                _safe_float(entry.get("oos_total_return_delta"), 0.0),
                -_safe_float(entry.get("oos_max_drawdown_delta"), 0.0),
            ),
        )
        winner_status = "promoted_challenger"
        winner_reason = str(winner.get("decision_reason") or "")
        recommended_action = (
            "Promote the challenger, then run cleanup and regression hardening on the new winner."
        )
    else:
        winner = incumbent
        winner_status = "retained_incumbent"
        if exposure_normalized and multiplicity_floor > 0.0:
            winner_reason = (
                "No challenger cleared the exposure-normalized, multiple-comparison-adjusted "
                f"superiority floor ({multiplicity_floor:.6f}); keep the current one-shot incumbent."
            )
        else:
            winner_reason = "No challenger cleared the locked-OOS robustness gates; keep the current one-shot incumbent."
        recommended_action = "Retain the incumbent and continue with cleanup, evidence collation, and reproducibility hardening."

    challenger_rankings = [
        {
            "candidate_key": entry.get("candidate_key"),
            "label": entry.get("label"),
            "promotion_score": entry.get("promotion_score"),
            "promotion_score_delta": entry.get("promotion_score_delta"),
            "promotable": entry.get("promotable"),
        }
        for entry in sorted(
            challengers,
            key=lambda item: _safe_float(item.get("promotion_score_delta"), 0.0),
            reverse=True,
        )
    ]

    promotion_formula = PROMOTION_FORMULA
    if exposure_normalized:
        promotion_formula = (
            "(1.0 * oos_sharpe) + (0.35 * oos_sortino) + (0.10 * oos_calmar) + "
            "((10.0 * oos_total_return) / gross_exposure) - "
            "((4.0 * oos_max_drawdown) / gross_exposure) - "
            "((0.75 * oos_volatility) / gross_exposure)"
        )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "portfolio_max_performance_decision",
        "schema_version": "1.0",
        "selection_basis": "locked_oos_robustness_gates",
        "split_contract": split_contract,
        "promotion_formula": promotion_formula,
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "comparison_scope": [entry.get("candidate_key") for entry in decorated],
        "supporting_artifacts": supporting_artifacts,
        "missing_artifacts": missing_artifacts,
        "fallback_retune_required": fallback_retune_required,
        "incumbent_bundle_path": str(resolved_incumbent_bundle_path),
        "incumbent_components": [
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "timeframe": row.get("timeframe") or row.get("strategy_timeframe"),
                "weight": _safe_float(
                    row.get("portfolio_weight", row.get("_portfolio_weight")), 0.0
                ),
            }
            for row in list(incumbent_bundle.get("candidates") or [])
            if isinstance(row, dict)
        ],
        "candidates": decorated,
        "challenger_rankings": challenger_rankings,
        "winner": {
            "candidate_key": winner.get("candidate_key"),
            "label": winner.get("label"),
            "status": winner_status,
            "promotion_score": winner.get("promotion_score"),
            "promotion_score_delta": winner.get("promotion_score_delta"),
            "reason": winner_reason,
        },
        "recommended_action": recommended_action,
        "notes": [
            "Promotion uses locked-OOS metrics only after each challenger artifact is frozen.",
            f"Locked OOS starts at {split_contract['oos_start']} and remains excluded from tuning decisions.",
            "A challenger is promotable only if train/validation returns stay positive, train Sharpe stays above -0.10, locked OOS beats the incumbent, locked OOS monthly mean stays above 2%, and drawdown is no worse unless Sharpe improves by at least 0.50.",
            f"Fallback retune required: {fallback_retune_required}.",
        ],
    }
    if exposure_normalized:
        payload["exposure_normalized_promotion"] = True
        payload["multiple_comparison_delta_floor"] = float(multiplicity_floor)
        payload["notes"].append(
            "Exposure-normalized promotion is ON: scale-dependent return/drawdown/volatility "
            "terms are divided by gross exposure and the locked-OOS superiority argmax must "
            "clear a multiple-comparison-adjusted floor, so leverage cannot buy superiority."
        )
    return payload


def write_portfolio_max_performance_decision(
    *,
    incumbent_bundle_path: Path | str = DEFAULT_INCUMBENT_BUNDLE,
    incumbent_portfolio_path: Path | str = DEFAULT_INCUMBENT_PORTFOLIO,
    tuned_comparison_path: Path | str = DEFAULT_TUNED_COMPARISON,
    dynamic_comparison_path: Path | str = DEFAULT_DYNAMIC_COMPARISON,
    overlay_comparison_path: Path | str = DEFAULT_OVERLAY_COMPARISON,
    regime_switch_comparison_path: Path | str = DEFAULT_REGIME_SWITCH_COMPARISON,
    grouped_allocator_path: Path | str | None = None,
    grouped_strict_validation_path: Path | str | None = DEFAULT_GROUPED_STRICT_VALIDATION,
    grouped_static_blend_path: Path | str | None = DEFAULT_GROUPED_STATIC_BLEND,
    production_guarded_path: Path | str | None = None,
    portfolio_superiority_meta_path: Path | str | None = None,
    backbone_triplet_path: Path | str = DEFAULT_BACKBONE_TRIPLET,
    anchored_comparison_path: Path | str = DEFAULT_ANCHORED_COMPARISON,
    anchored_tuned_comparison_path: Path | str | None = None,
    portfolio_four_sleeve_comparison_path: Path | str | None = None,
    four_sleeve_comparison_path: Path | str | None = None,
    meta_search_summary_paths: tuple[Path | str, ...] = (),
    extra_candidate_artifact_paths: tuple[Path | str, ...] = (),
    output_json_path: Path | str = DEFAULT_OUTPUT_JSON,
    output_md_path: Path | str = DEFAULT_OUTPUT_MD,
    research_config: Any = None,
) -> dict[str, Any]:
    payload = build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle_path,
        incumbent_portfolio_path=incumbent_portfolio_path,
        tuned_comparison_path=tuned_comparison_path,
        dynamic_comparison_path=dynamic_comparison_path,
        overlay_comparison_path=overlay_comparison_path,
        regime_switch_comparison_path=regime_switch_comparison_path,
        grouped_allocator_path=grouped_allocator_path,
        grouped_strict_validation_path=grouped_strict_validation_path,
        grouped_static_blend_path=grouped_static_blend_path,
        production_guarded_path=production_guarded_path,
        portfolio_superiority_meta_path=portfolio_superiority_meta_path,
        backbone_triplet_path=backbone_triplet_path,
        anchored_comparison_path=anchored_comparison_path,
        anchored_tuned_comparison_path=anchored_tuned_comparison_path,
        portfolio_four_sleeve_comparison_path=portfolio_four_sleeve_comparison_path,
        four_sleeve_comparison_path=four_sleeve_comparison_path,
        meta_search_summary_paths=meta_search_summary_paths,
        extra_candidate_artifact_paths=extra_candidate_artifact_paths,
        research_config=research_config,
    )
    json_path = Path(output_json_path)
    md_path = Path(output_md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    winner = dict(payload.get("winner") or {})
    lines = [
        "# portfolio max-performance decision",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- selection_basis: `{payload['selection_basis']}`",
        f"- winner: `{winner.get('label')}` ({winner.get('status')})",
        f"- winner_reason: {winner.get('reason')}",
        f"- oos_start: `{payload['split_contract']['oos_start']}`",
        "",
        "## candidate summary",
    ]
    for entry in list(payload.get("candidates") or []):
        oos = dict(entry.get("oos") or {})
        lines.append(
            f"- `{entry.get('label')}` | score={_safe_float(entry.get('promotion_score'), 0.0):.4f} | "
            f"delta={_safe_float(entry.get('promotion_score_delta'), 0.0):.4f} | "
            f"oos_return={_safe_float(oos.get('total_return', oos.get('return')), 0.0):.4%} | "
            f"oos_sharpe={_safe_float(oos.get('sharpe'), 0.0):.3f} | "
            f"oos_max_dd={_safe_float(oos.get('max_drawdown', oos.get('mdd')), 0.0):.4%} | "
            f"promotable={entry.get('promotable')}"
        )
        lines.append(f"  - {entry.get('decision_reason')}")
    lines.extend(["", "## notes"])
    for note in list(payload.get("notes") or []):
        lines.append(f"- {note}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the portfolio max-performance decision artifact."
    )
    parser.add_argument("--incumbent-bundle", default=str(DEFAULT_INCUMBENT_BUNDLE))
    parser.add_argument("--incumbent-portfolio", default=str(DEFAULT_INCUMBENT_PORTFOLIO))
    parser.add_argument("--tuned-comparison", default=str(DEFAULT_TUNED_COMPARISON))
    parser.add_argument("--dynamic-comparison", default=str(DEFAULT_DYNAMIC_COMPARISON))
    parser.add_argument("--overlay-comparison", default=str(DEFAULT_OVERLAY_COMPARISON))
    parser.add_argument("--regime-switch-comparison", default=str(DEFAULT_REGIME_SWITCH_COMPARISON))
    parser.add_argument("--grouped-allocator", default=str(DEFAULT_GROUPED_ALLOCATOR))
    parser.add_argument(
        "--grouped-strict-validation", default=str(DEFAULT_GROUPED_STRICT_VALIDATION)
    )
    parser.add_argument("--grouped-static-blend", default=str(DEFAULT_GROUPED_STATIC_BLEND))
    parser.add_argument("--production-guarded", default=str(DEFAULT_PRODUCTION_GUARDED))
    parser.add_argument(
        "--portfolio-superiority-meta", default=str(DEFAULT_PORTFOLIO_SUPERIORITY_META)
    )
    parser.add_argument("--backbone-triplet", default=str(DEFAULT_BACKBONE_TRIPLET))
    parser.add_argument("--anchored-comparison", default=str(DEFAULT_ANCHORED_COMPARISON))
    parser.add_argument("--extra-candidate-artifact", action="append", default=[])
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument(
        "--exposure-normalized-promotion",
        action="store_true",
        help=(
            "Opt in to the exposure-normalized promotion objective and the "
            "multiple-comparison-adjusted superiority argmax (default OFF: legacy behavior)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = write_portfolio_max_performance_decision(
        incumbent_bundle_path=Path(args.incumbent_bundle).resolve(),
        incumbent_portfolio_path=Path(args.incumbent_portfolio).resolve(),
        tuned_comparison_path=Path(args.tuned_comparison).resolve(),
        dynamic_comparison_path=Path(args.dynamic_comparison).resolve(),
        overlay_comparison_path=Path(args.overlay_comparison).resolve(),
        regime_switch_comparison_path=Path(args.regime_switch_comparison).resolve(),
        grouped_allocator_path=Path(args.grouped_allocator).resolve(),
        grouped_strict_validation_path=Path(args.grouped_strict_validation).resolve(),
        grouped_static_blend_path=Path(args.grouped_static_blend).resolve(),
        production_guarded_path=Path(args.production_guarded).resolve(),
        portfolio_superiority_meta_path=Path(args.portfolio_superiority_meta).resolve(),
        backbone_triplet_path=Path(args.backbone_triplet).resolve(),
        anchored_comparison_path=Path(args.anchored_comparison).resolve(),
        meta_search_summary_paths=DEFAULT_META_SEARCH_SUMMARIES,
        extra_candidate_artifact_paths=tuple(
            Path(path).resolve() for path in list(args.extra_candidate_artifact or [])
        ),
        output_json_path=Path(args.output_json).resolve(),
        output_md_path=Path(args.output_md).resolve(),
        research_config=SimpleNamespace(
            exposure_normalized_promotion=bool(args.exposure_normalized_promotion)
        ),
    )
    print(result["json_path"])
    print(result["md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
