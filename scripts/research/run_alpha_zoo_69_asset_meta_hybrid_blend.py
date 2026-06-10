#!/usr/bin/env python3
"""Evaluate a 69-asset meta-hybrid blend of dynamic alpha plus stability core.

This runner is deliberately explicit about selection surfaces.  The dynamic sleeve
uses train/validation-frozen per-asset/profile tuned rows and only trailing returns
available before each rebalance.  The stability sleeve is supplied as a frozen core
artifact.  If that core was selected with walk-forward/OOS diagnostics, the output
marks the resulting blend as shadow-only even when replay metrics are strong.

The intended current research use is to compare:

* a deployable train/validation-selected diversified core; and
* a diagnostic WFO-positive core used only as a stability sleeve candidate.

All 69 assets remain in the monitor candidate pool through the dynamic source
manifest; only stream-capable/tuned rows can be traded now.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_clean_oos_gate as clean_gate  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_per_asset_dynamic_pool as dynamic  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as profile69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_walkforward_monitor as walkforward  # noqa: E402

DEFAULT_SOURCE_ARTIFACT = (
    profile69.DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json"
)
DEFAULT_OUTPUT_DIR = broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_meta_hybrid_blend_20260531"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_meta_hybrid_blend_latest.json"
DEFAULT_TRAIN_START = clean_gate.DEFAULT_TRAIN_START
DEFAULT_TRAIN_END = clean_gate.DEFAULT_TRAIN_END
DEFAULT_VALIDATION_START = clean_gate.DEFAULT_VALIDATION_START
DEFAULT_VALIDATION_END = clean_gate.DEFAULT_VALIDATION_END
DEFAULT_LOCKED_OOS_START = clean_gate.DEFAULT_LOCKED_OOS_START
DEFAULT_LOCKED_OOS_END = clean_gate.DEFAULT_LOCKED_OOS_END
DEFAULT_MAX_OOS_MDD = clean_gate.DEFAULT_MAX_OOS_MDD
DEFAULT_DYNAMIC_WEIGHT_STEP = 0.01
DEFAULT_WFO_CORE_ARTIFACT = Path(
    "/tmp/lumina_clean_oos_alpha_zoo_69_active_pool/wfo_positive_core10_from_69_pool.json"
)
DEFAULT_TV_CORE_ARTIFACT = Path(
    "/tmp/lumina_clean_oos_alpha_zoo_69_active_pool/diversified_watch_core_10.json"
)


@dataclass(frozen=True)
class MetaHybridParams:
    lookback_days: int = dynamic.SelectorParams.lookback_days
    rebalance_days: int = dynamic.SelectorParams.rebalance_days
    top_n: int = dynamic.SelectorParams.top_n
    target_gross: float = dynamic.SelectorParams.target_gross
    min_trailing_return: float = dynamic.SelectorParams.min_trailing_return
    fit_weight: float = dynamic.SelectorParams.fit_weight
    vol_penalty: float = dynamic.SelectorParams.vol_penalty
    max_symbol_gross: float = dynamic.SelectorParams.max_symbol_gross
    dynamic_weight_step: float = DEFAULT_DYNAMIC_WEIGHT_STEP
    max_oos_mdd: float = DEFAULT_MAX_OOS_MDD

    def selector_params(self) -> dynamic.SelectorParams:
        return dynamic.SelectorParams(
            lookback_days=self.lookback_days,
            rebalance_days=self.rebalance_days,
            top_n=self.top_n,
            target_gross=self.target_gross,
            min_trailing_return=self.min_trailing_return,
            fit_weight=self.fit_weight,
            vol_penalty=self.vol_penalty,
            max_symbol_gross=self.max_symbol_gross,
        )


@dataclass(frozen=True)
class SleeveReturn:
    label: str
    artifact_path: str | None
    profile_id: str
    returns: pd.Series
    gross_notional_fraction: float
    selection_surface: str
    deployable_without_refit: bool
    notes: tuple[str, ...]


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(broad69._json_safe(payload), indent=2, sort_keys=True) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _parse_timestamp(value: Any) -> pd.Timestamp:
    return clean_gate._parse_timestamp(value)


def _coerce_windows(args: argparse.Namespace) -> clean_gate.GateWindows:
    return clean_gate.GateWindows(
        train=(_parse_timestamp(args.train_start), _parse_timestamp(args.train_end)),
        validation=(_parse_timestamp(args.validation_start), _parse_timestamp(args.validation_end)),
        locked_oos=(_parse_timestamp(args.locked_oos_start), _parse_timestamp(args.locked_oos_end)),
    )


def _window_payload(window: tuple[pd.Timestamp, pd.Timestamp], role: str) -> dict[str, Any]:
    return {
        "start": window[0].isoformat() + "Z",
        "end": window[1].isoformat() + "Z",
        "role": role,
        "enabled": True,
    }


def _split_manifest(windows: clean_gate.GateWindows) -> dict[str, Any]:
    return {
        "train": _window_payload(windows.train, "parameter_fitting_and_objective_training"),
        "validation": _window_payload(windows.validation, "holdout_selection_and_report"),
        "locked_oos": _window_payload(
            windows.locked_oos, "gate_report_only_after_train_validation_freeze"
        ),
    }


def _dynamic_return_sleeve(
    context: dynamic.DynamicContext, params: dynamic.SelectorParams
) -> tuple[SleeveReturn, dict[str, Any]]:
    weights, selection_log = dynamic.dynamic_weights(context, params)
    returns = pd.Series(np.sum(context.returns * weights, axis=0), index=context.index).sort_index()
    gross = np.sum(np.abs(weights) * context.notionals[:, None], axis=0)
    ready_count = np.sum(np.abs(weights) > 1e-12, axis=0)
    active_signal_count = np.sum(
        (np.abs(weights) > 1e-12) & (np.abs(context.positions) > 1e-12), axis=0
    )
    activity = {
        "avg_gross_notional_fraction": float(np.mean(gross)) if gross.size else 0.0,
        "max_gross_notional_fraction": float(np.max(gross)) if gross.size else 0.0,
        "avg_ready_row_count": float(np.mean(ready_count)) if ready_count.size else 0.0,
        "max_ready_row_count": int(np.max(ready_count)) if ready_count.size else 0,
        "avg_active_signal_count": float(np.mean(active_signal_count))
        if active_signal_count.size
        else 0.0,
        "max_active_signal_count": int(np.max(active_signal_count))
        if active_signal_count.size
        else 0,
        "selection_log_tail": selection_log[-20:],
    }
    sleeve = SleeveReturn(
        label="dynamic_alpha_sleeve",
        artifact_path=None,
        profile_id="per_asset_dynamic_alpha_selector",
        returns=returns,
        gross_notional_fraction=float(activity["avg_gross_notional_fraction"]),
        selection_surface="train_validation_frozen_rows_plus_online_trailing_rebalance",
        deployable_without_refit=True,
        notes=(
            "selector parameters are fixed before locked-OOS replay",
            "rebalance ranking uses only trailing strategy returns before each rebalance",
        ),
    )
    return sleeve, activity


def _selection_surface_from_artifact(
    label: str, payload: Mapping[str, Any], path: Path
) -> tuple[str, bool, tuple[str, ...]]:
    raw = json.dumps(
        {
            "label": label,
            "path": str(path),
            "evaluation_policy": payload.get("evaluation_policy"),
            "selection_recommendation": payload.get("selection_recommendation"),
            "selected": payload.get("selected_optuna_hybrid_profile"),
        },
        sort_keys=True,
        default=str,
    ).lower()
    explicit_selected = dict(payload.get("selected_optuna_hybrid_profile") or {})
    oos_used = bool(explicit_selected.get("oos_used_for_selection"))
    if "wfo" in raw or "walk-forward" in raw or "walkforward" in raw or "oos diagnostic" in raw:
        oos_used = True
    if oos_used:
        return (
            "walkforward_or_oos_diagnostic_selected_stability_core",
            False,
            (
                "stability core appears selected with WFO/OOS diagnostics",
                "blend is useful for shadow/paper diagnostics but is not clean live-promotable without refit",
            ),
        )
    return (
        "train_validation_selected_stability_core",
        True,
        (
            "stability core metadata does not indicate OOS selection",
            "still requires separate clean locked-OOS gate before any promotion",
        ),
    )


def build_core_sleeve_from_artifact(
    label: str, artifact_path: Path, windows: clean_gate.GateWindows
) -> SleeveReturn:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    context = clean_gate._build_profile_context(payload, windows)
    selected = dict(payload.get("selected_optuna_hybrid_profile") or {})
    profile_id = str(selected.get("profile_id") or payload.get("selected_profile_id") or "")
    profile_returns = dict(context.get("profile_returns") or {})
    if profile_id not in profile_returns:
        if not profile_returns:
            raise ValueError(f"stabilizer artifact has no profile returns: {artifact_path}")
        profile_id = str(next(iter(profile_returns)))
    surface, deployable, notes = _selection_surface_from_artifact(label, payload, artifact_path)
    return SleeveReturn(
        label=label,
        artifact_path=str(artifact_path),
        profile_id=profile_id,
        returns=profile_returns[profile_id].sort_index(),
        gross_notional_fraction=_safe_float(
            dict(context.get("profile_gross") or {}).get(profile_id)
        ),
        selection_surface=surface,
        deployable_without_refit=deployable,
        notes=notes,
    )


def blend_return_series(
    dynamic_returns: pd.Series, core_returns: pd.Series, dynamic_weight: float
) -> pd.Series:
    weight = float(dynamic_weight)
    if weight < -1e-12 or weight > 1.0 + 1e-12:
        raise ValueError("dynamic_weight must be between 0 and 1")
    index = pd.DatetimeIndex(sorted(set(dynamic_returns.index) | set(core_returns.index)))
    dynamic_aligned = dynamic_returns.reindex(index, fill_value=0.0)
    core_aligned = core_returns.reindex(index, fill_value=0.0)
    return (dynamic_aligned * weight + core_aligned * (1.0 - weight)).sort_index()


def evaluate_return_series(
    returns: pd.Series,
    windows: clean_gate.GateWindows,
    folds: Sequence[walkforward.WalkForwardFold] = walkforward.DEFAULT_FOLDS,
) -> dict[str, Any]:
    clean_metrics = {
        "train": walkforward.period_metrics(returns, windows.train),
        "validation": walkforward.period_metrics(returns, windows.validation),
        "locked_oos": walkforward.period_metrics(returns, windows.locked_oos),
    }
    fold_payloads: list[dict[str, Any]] = []
    validation_returns: list[float] = []
    oos_returns: list[float] = []
    validation_mdds: list[float] = []
    oos_mdds: list[float] = []
    for fold in folds:
        validation = walkforward.period_metrics(returns, fold.validation)
        locked_oos = walkforward.period_metrics(returns, fold.locked_oos)
        validation_returns.append(float(validation["total_return"]))
        oos_returns.append(float(locked_oos["total_return"]))
        validation_mdds.append(float(validation["mdd"]))
        oos_mdds.append(float(locked_oos["mdd"]))
        fold_payloads.append(
            {
                "fold_id": fold.fold_id,
                "train": walkforward.period_metrics(returns, fold.train),
                "validation": validation,
                "locked_oos": locked_oos,
            }
        )
    all_validation_positive = bool(validation_returns) and all(
        value > 0.0 for value in validation_returns
    )
    all_oos_positive = bool(oos_returns) and all(value > 0.0 for value in oos_returns)
    return {
        "clean_metrics": clean_metrics,
        "walkforward_folds": fold_payloads,
        "walkforward_summary": {
            "fold_count": len(fold_payloads),
            "min_validation_return": min(validation_returns) if validation_returns else 0.0,
            "min_oos_return": min(oos_returns) if oos_returns else 0.0,
            "max_validation_mdd": max(validation_mdds) if validation_mdds else 0.0,
            "max_oos_mdd": max(oos_mdds) if oos_mdds else 0.0,
            "final_fold_oos_return": oos_returns[-1] if oos_returns else 0.0,
            "final_fold_oos_mdd": oos_mdds[-1] if oos_mdds else 0.0,
            "all_validation_positive": all_validation_positive,
            "all_oos_positive": all_oos_positive,
            "all_validation_and_oos_positive": all_validation_positive and all_oos_positive,
        },
    }


def candidate_score(candidate: Mapping[str, Any]) -> float:
    summary = dict(candidate.get("walkforward_summary") or {})
    clean = dict(candidate.get("clean_metrics") or {})
    locked = dict(clean.get("locked_oos") or {})
    if not bool(summary.get("all_validation_and_oos_positive")):
        return -1e9
    if float(summary.get("max_oos_mdd") or 0.0) > float(candidate.get("max_oos_mdd_gate") or 0.0):
        return -1e9
    dynamic_weight = float(candidate.get("dynamic_weight") or 0.0)
    return float(
        3.0 * float(locked.get("total_return") or 0.0)
        + 2.0 * float(summary.get("min_oos_return") or 0.0)
        + 1.0 * float(summary.get("min_validation_return") or 0.0)
        + 0.25 * dynamic_weight
        - 1.5 * float(summary.get("max_oos_mdd") or 0.0)
    )


def _candidate_payload(
    *,
    dynamic_sleeve: SleeveReturn,
    core_sleeve: SleeveReturn,
    dynamic_weight: float,
    windows: clean_gate.GateWindows,
    max_oos_mdd: float,
) -> dict[str, Any]:
    core_weight = 1.0 - float(dynamic_weight)
    returns = blend_return_series(dynamic_sleeve.returns, core_sleeve.returns, dynamic_weight)
    evaluation = evaluate_return_series(returns, windows)
    summary = dict(evaluation.get("walkforward_summary") or {})
    clean = dict(evaluation.get("clean_metrics") or {})
    clean_oos = dict(clean.get("locked_oos") or {})
    blend_deployable = bool(
        dynamic_sleeve.deployable_without_refit and core_sleeve.deployable_without_refit
    )
    blockers: list[str] = []
    if not blend_deployable:
        blockers.append("one_or_more_sleeves_selected_with_oos_or_diagnostic_surface")
    if not bool(summary.get("all_validation_and_oos_positive")):
        blockers.append("walkforward_validation_or_oos_has_negative_fold")
    if float(summary.get("max_oos_mdd") or 0.0) > max_oos_mdd:
        blockers.append("walkforward_oos_mdd_above_limit")
    if float(clean_oos.get("total_return") or 0.0) <= 0.0:
        blockers.append("clean_locked_oos_return_not_positive")
    return {
        "candidate_id": f"meta_hybrid_{core_sleeve.label}_dyn{dynamic_weight:.2f}_core{core_weight:.2f}",
        "dynamic_weight": float(dynamic_weight),
        "core_weight": core_weight,
        "max_oos_mdd_gate": float(max_oos_mdd),
        "dynamic_sleeve": {
            "label": dynamic_sleeve.label,
            "profile_id": dynamic_sleeve.profile_id,
            "selection_surface": dynamic_sleeve.selection_surface,
            "deployable_without_refit": dynamic_sleeve.deployable_without_refit,
            "gross_notional_fraction": dynamic_sleeve.gross_notional_fraction,
            "notes": list(dynamic_sleeve.notes),
        },
        "stability_sleeve": {
            "label": core_sleeve.label,
            "artifact_path": core_sleeve.artifact_path,
            "profile_id": core_sleeve.profile_id,
            "selection_surface": core_sleeve.selection_surface,
            "deployable_without_refit": core_sleeve.deployable_without_refit,
            "gross_notional_fraction": core_sleeve.gross_notional_fraction,
            "notes": list(core_sleeve.notes),
        },
        "estimated_avg_gross_notional_fraction": (
            float(dynamic_weight) * dynamic_sleeve.gross_notional_fraction
            + core_weight * core_sleeve.gross_notional_fraction
        ),
        "deployable_without_refit": blend_deployable,
        "paper_shadow_only": not blend_deployable,
        "promotion_blockers": blockers,
        **evaluation,
    }


def _dynamic_weight_grid(step: float) -> list[float]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("dynamic weight step must be in (0, 1]")
    count = round(1.0 / step)
    return [round(index / count, 10) for index in range(count + 1)]


def evaluate_meta_hybrid_candidates(
    *,
    dynamic_sleeve: SleeveReturn,
    core_sleeves: Sequence[SleeveReturn],
    windows: clean_gate.GateWindows,
    params: MetaHybridParams,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for core_sleeve in core_sleeves:
        for dynamic_weight in _dynamic_weight_grid(params.dynamic_weight_step):
            candidate = _candidate_payload(
                dynamic_sleeve=dynamic_sleeve,
                core_sleeve=core_sleeve,
                dynamic_weight=dynamic_weight,
                windows=windows,
                max_oos_mdd=params.max_oos_mdd,
            )
            candidate["score"] = candidate_score(candidate)
            candidates.append(candidate)
    candidates.sort(key=lambda item: float(item.get("score") or -1e9), reverse=True)
    deployable = [item for item in candidates if bool(item.get("deployable_without_refit"))]
    deployable.sort(key=lambda item: float(item.get("score") or -1e9), reverse=True)
    all_positive = [
        item
        for item in candidates
        if bool(dict(item.get("walkforward_summary") or {}).get("all_validation_and_oos_positive"))
        and float(dict(item.get("walkforward_summary") or {}).get("max_oos_mdd") or 0.0)
        <= params.max_oos_mdd
    ]
    return {
        "candidate_count": len(candidates),
        "all_positive_candidate_count": len(all_positive),
        "deployable_candidate_count": len(deployable),
        "best_candidate": candidates[0] if candidates else None,
        "best_all_positive_candidate": all_positive[0] if all_positive else None,
        "best_deployable_candidate": deployable[0] if deployable else None,
        "top_candidates": candidates[:20],
        "top_all_positive_candidates": all_positive[:20],
        "top_deployable_candidates": deployable[:20],
    }


def _parse_labeled_artifact(value: str) -> tuple[str, Path]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("empty stabilizer artifact")
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path).expanduser().resolve()
    path = Path(raw).expanduser().resolve()
    return path.stem, path


def _default_stabilizer_artifacts() -> list[tuple[str, Path]]:
    defaults = [
        ("train_validation_diversified_core10", DEFAULT_TV_CORE_ARTIFACT),
        ("diagnostic_wfo_positive_core10", DEFAULT_WFO_CORE_ARTIFACT),
    ]
    return [(label, path) for label, path in defaults if path.exists()]


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    windows = _coerce_windows(args)
    params = MetaHybridParams(
        lookback_days=int(args.lookback_days),
        rebalance_days=int(args.rebalance_days),
        top_n=int(args.top_n),
        target_gross=float(args.target_gross),
        min_trailing_return=float(args.min_trailing_return),
        fit_weight=float(args.fit_weight),
        vol_penalty=float(args.vol_penalty),
        max_symbol_gross=float(args.max_symbol_gross),
        dynamic_weight_step=float(args.dynamic_weight_step),
        max_oos_mdd=float(args.max_oos_mdd),
    )
    source_artifact = Path(args.source_artifact).expanduser().resolve()
    context = dynamic.build_dynamic_context(source_artifact, windows)
    dynamic_sleeve, dynamic_activity = _dynamic_return_sleeve(context, params.selector_params())
    labeled_paths = [_parse_labeled_artifact(value) for value in args.stabilizer_artifact]
    if not labeled_paths:
        labeled_paths = _default_stabilizer_artifacts()
    if not labeled_paths:
        raise ValueError("no stabilizer artifacts supplied/found")
    core_sleeves = [
        build_core_sleeve_from_artifact(label, path, windows) for label, path in labeled_paths
    ]
    search = evaluate_meta_hybrid_candidates(
        dynamic_sleeve=dynamic_sleeve,
        core_sleeves=core_sleeves,
        windows=windows,
        params=params,
    )
    best = dict(search.get("best_candidate") or {})
    best_positive = dict(search.get("best_all_positive_candidate") or {})
    best_deployable = dict(search.get("best_deployable_candidate") or {})
    return {
        "artifact_kind": "alpha_zoo_69_asset_meta_hybrid_blend",
        "generated_at_utc": _utc_now_iso(),
        "source_artifact": str(source_artifact),
        "stabilizer_artifacts": [
            {"label": label, "path": str(path)} for label, path in labeled_paths
        ],
        "evaluation_policy": {
            "candidate_pool": "all_69_symbols_monitor_candidate_pool_from_dynamic_source",
            "locked_oos_used_for_dynamic_parameter_fitting": False,
            "locked_oos_used_for_replay_report": True,
            "dynamic_selector_rebalance_uses_only_prior_trailing_returns": True,
            "blend_weight_search_surface": "walkforward_diagnostic_grid_report",
            "deployable_candidate_requires_each_sleeve_train_validation_selected": True,
            "real_money_execution_allowed": False,
            "paper_testnet_only": True,
        },
        "split_manifest": _split_manifest(windows),
        "candidate_pool_policy": context.candidate_pool_policy,
        "selector_params": asdict(params),
        "dynamic_activity_summary": dynamic_activity,
        "search_summary": {
            "candidate_count": search.get("candidate_count"),
            "all_positive_candidate_count": search.get("all_positive_candidate_count"),
            "deployable_candidate_count": search.get("deployable_candidate_count"),
            "best_candidate_id": best.get("candidate_id"),
            "best_all_positive_candidate_id": best_positive.get("candidate_id"),
            "best_deployable_candidate_id": best_deployable.get("candidate_id"),
        },
        **search,
        "recommended_shadow_candidate": best_positive or best,
        "recommended_deployable_candidate": best_deployable or None,
        "ready_for_paper": bool(best_deployable)
        and not list(best_deployable.get("promotion_blockers") or []),
        "ready_for_real": False,
        "real_execution_allowed": False,
        "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _pct(value: Any) -> str:
    return f"{float(value):+.4%}"


def _candidate_row(candidate: Mapping[str, Any]) -> str:
    summary = dict(candidate.get("walkforward_summary") or {})
    clean = dict(candidate.get("clean_metrics") or {})
    locked = dict(clean.get("locked_oos") or {})
    val = dict(clean.get("validation") or {})
    return (
        f"| `{candidate.get('candidate_id')}` | `{candidate.get('deployable_without_refit')}` | "
        f"{float(candidate.get('dynamic_weight') or 0.0):.2f} | "
        f"{_pct(val.get('total_return') or 0.0)} | {_pct(locked.get('total_return') or 0.0)} | "
        f"{_pct(locked.get('mdd') or 0.0)} | {_pct(summary.get('min_validation_return') or 0.0)} | "
        f"{_pct(summary.get('min_oos_return') or 0.0)} | {_pct(summary.get('max_oos_mdd') or 0.0)} | "
        f"`{summary.get('all_validation_and_oos_positive')}` | `{candidate.get('promotion_blockers')}` |"
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("search_summary") or {})
    shadow = dict(payload.get("recommended_shadow_candidate") or {})
    deployable = dict(payload.get("recommended_deployable_candidate") or {})
    pool = dict(payload.get("candidate_pool_policy") or {})
    lines = [
        "# 69-asset meta-hybrid dynamic/stability blend",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        f"Source: `{payload.get('source_artifact')}`",
        f"Candidate pool / stream-capable: `{pool.get('candidate_pool_symbol_count')}` / `{pool.get('current_stream_capable_symbol_count')}`",
        f"Best all-positive shadow: `{shadow.get('candidate_id')}`",
        f"Best deployable candidate: `{deployable.get('candidate_id')}`",
        f"Ready for paper: `{payload.get('ready_for_paper')}`",
        "",
        "## Search summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Recommended candidates",
            "",
            "| candidate | deployable | dyn wt | clean val | clean OOS | OOS MDD | min WF val | min WF OOS | max WF OOS MDD | all WF val/OOS + | blockers |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if shadow:
        lines.append(_candidate_row(shadow))
    if deployable and deployable.get("candidate_id") != shadow.get("candidate_id"):
        lines.append(_candidate_row(deployable))
    lines.extend(
        [
            "",
            "## Top all-positive candidates",
            "",
            "| candidate | deployable | dyn wt | clean val | clean OOS | OOS MDD | min WF val | min WF OOS | max WF OOS MDD | all WF val/OOS + | blockers |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in payload.get("top_all_positive_candidates") or []:
        lines.append(_candidate_row(dict(candidate)))
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--stabilizer-artifact", action="append", default=[])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--locked-oos-start", default=DEFAULT_LOCKED_OOS_START)
    parser.add_argument("--locked-oos-end", default=DEFAULT_LOCKED_OOS_END)
    parser.add_argument("--max-oos-mdd", type=float, default=DEFAULT_MAX_OOS_MDD)
    parser.add_argument("--lookback-days", type=int, default=MetaHybridParams.lookback_days)
    parser.add_argument("--rebalance-days", type=int, default=MetaHybridParams.rebalance_days)
    parser.add_argument("--top-n", type=int, default=MetaHybridParams.top_n)
    parser.add_argument("--target-gross", type=float, default=MetaHybridParams.target_gross)
    parser.add_argument(
        "--min-trailing-return", type=float, default=MetaHybridParams.min_trailing_return
    )
    parser.add_argument("--fit-weight", type=float, default=MetaHybridParams.fit_weight)
    parser.add_argument("--vol-penalty", type=float, default=MetaHybridParams.vol_penalty)
    parser.add_argument("--max-symbol-gross", type=float, default=MetaHybridParams.max_symbol_gross)
    parser.add_argument("--dynamic-weight-step", type=float, default=DEFAULT_DYNAMIC_WEIGHT_STEP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    output = Path(args.output).expanduser().resolve()
    _write_json(output, payload)
    output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
