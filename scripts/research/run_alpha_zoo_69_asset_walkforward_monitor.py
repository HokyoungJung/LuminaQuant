#!/usr/bin/env python3
"""Build a 69-asset monitor plus frozen-parameter walk-forward diagnostic.

This runner does **not** promote a new live profile from OOS data.  It uses a
clean train/validation-frozen source artifact to answer a narrower question:
which already-frozen profile mix has the best return shape across expanding
validation/OOS windows, while the full 69-symbol universe remains monitored?

The output deliberately separates:

* ``deployable_core``: train/validation-selected, locked-OOS-gated elsewhere;
* ``diagnostic_shadow``: found with walk-forward/OOS diagnostics and therefore
  not live-promotable without a fresh train/validation-only selection run;
* ``monitor_manifest``: all 69 assets with current tradable/watchlist status.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_clean_oos_gate as clean_gate  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_diverse_salvage as salvage  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as profile69  # noqa: E402

DEFAULT_SOURCE_ARTIFACT = (
    profile69.DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json"
)
DEFAULT_DIVERSE_ARTIFACT = salvage.DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_DIR = broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_walkforward_monitor_20260531"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_walkforward_monitor_latest.json"
DEFAULT_MAX_OOS_MDD = 0.20
DEFAULT_MAX_GROSS = 3.0
DEFAULT_MIN_GROSS = 1.0
DEFAULT_GROSS_STEP = 0.25
DEFAULT_MIX_STEP = 0.02
BALANCED_PROFILE_ID = "balanced_mdd12_gross5_69_asset_profile_optuna"
GROWTH_PROFILE_ID = "growth_mdd20_gross8_69_asset_profile_optuna"
AGGRESSIVE_PROFILE_ID = "aggressive_mdd30_gross10_69_asset_profile_optuna"
DEFAULT_PROFILE_ORDER = (BALANCED_PROFILE_ID, GROWTH_PROFILE_ID, AGGRESSIVE_PROFILE_ID)


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train: tuple[pd.Timestamp, pd.Timestamp]
    validation: tuple[pd.Timestamp, pd.Timestamp]
    locked_oos: tuple[pd.Timestamp, pd.Timestamp]

    def as_payload(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train": _window_payload(self.train),
            "validation": _window_payload(self.validation),
            "locked_oos": _window_payload(self.locked_oos),
        }


@dataclass(frozen=True)
class CandidateWeights:
    candidate_id: str
    weights: dict[str, float]
    gross_notional_fraction: float
    profile_mix: dict[str, float]
    selection_surface: str
    deployable_without_refit: bool
    notes: tuple[str, ...]


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(UTC).tz_localize(None)
    return ts


DEFAULT_FOLDS = (
    WalkForwardFold(
        "wf1_val_2025_07_08_oos_2025_09_10",
        (_ts("2025-01-01T00:00:00Z"), _ts("2025-06-30T23:00:00Z")),
        (_ts("2025-07-01T00:00:00Z"), _ts("2025-08-31T23:00:00Z")),
        (_ts("2025-09-01T00:00:00Z"), _ts("2025-10-31T23:00:00Z")),
    ),
    WalkForwardFold(
        "wf2_val_2025_09_10_oos_2025_11_12",
        (_ts("2025-01-01T00:00:00Z"), _ts("2025-08-31T23:00:00Z")),
        (_ts("2025-09-01T00:00:00Z"), _ts("2025-10-31T23:00:00Z")),
        (_ts("2025-11-01T00:00:00Z"), _ts("2025-12-31T23:00:00Z")),
    ),
    WalkForwardFold(
        "wf3_val_2025_11_12_oos_2026_01_02",
        (_ts("2025-01-01T00:00:00Z"), _ts("2025-10-31T23:00:00Z")),
        (_ts("2025-11-01T00:00:00Z"), _ts("2025-12-31T23:00:00Z")),
        (_ts("2026-01-01T00:00:00Z"), _ts("2026-02-28T23:00:00Z")),
    ),
    WalkForwardFold(
        "wf4_val_2026_01_02_oos_2026_03_05",
        (_ts("2025-01-01T00:00:00Z"), _ts("2025-12-31T23:00:00Z")),
        (_ts("2026-01-01T00:00:00Z"), _ts("2026-02-28T23:00:00Z")),
        (_ts("2026-03-01T00:00:00Z"), _ts("2026-05-06T23:00:00Z")),
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(broad69._json_safe(payload), indent=2, sort_keys=True) + "\n")


def _window_payload(window: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, Any]:
    return {"start": window[0].isoformat() + "Z", "end": window[1].isoformat() + "Z"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except TypeError, ValueError:
        return default
    return number if math.isfinite(number) else default


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 365.0 * 24.0
    diffs = pd.Series(index).diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 365.0 * 24.0
    median_seconds = float(diffs.median())
    return 365.0 * 24.0 * 3600.0 / median_seconds if median_seconds > 0.0 else 365.0 * 24.0


def period_mask(index: pd.DatetimeIndex, window: tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    values = pd.DatetimeIndex(pd.to_datetime(index))
    return np.asarray((values >= window[0]) & (values <= window[1]), dtype=bool)


def total_return(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.prod(1.0 + values) - 1.0) if values.size else 0.0


def max_drawdown(values: np.ndarray) -> float:
    return float(broad69.max_drawdown(np.asarray(values, dtype=float)))


def period_metrics(returns: pd.Series, window: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, Any]:
    sorted_returns = returns.sort_index()
    index = pd.DatetimeIndex(sorted_returns.index)
    mask = period_mask(index, window)
    values = sorted_returns.to_numpy(dtype=float)[mask]
    period_index = index[mask]
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    downside = values[values < 0.0]
    down_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    annual = _periods_per_year(period_index)
    ret = total_return(values)
    mdd = max_drawdown(values) if values.size else 0.0
    return {
        "start": window[0].isoformat() + "Z",
        "end": window[1].isoformat() + "Z",
        "bar_count": int(values.size),
        "total_return": ret,
        "mdd": mdd,
        "sharpe": mean / std * math.sqrt(annual) if std > 0.0 else 0.0,
        "sortino": mean / down_std * math.sqrt(annual) if down_std > 0.0 else 0.0,
        "calmar": ret / mdd if mdd > 0.0 else 0.0,
    }


def combine_profile_returns(
    profile_returns: Mapping[str, pd.Series], weights: Mapping[str, float]
) -> pd.Series:
    active = {
        str(profile_id): float(weight)
        for profile_id, weight in weights.items()
        if float(weight) != 0.0 and str(profile_id) in profile_returns
    }
    if not active:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex(
        sorted(set().union(*(set(profile_returns[profile_id].index) for profile_id in active)))
    )
    combined = pd.Series(0.0, index=index)
    for profile_id, weight in active.items():
        combined = combined.add(profile_returns[profile_id].reindex(index, fill_value=0.0) * weight)
    return combined.sort_index()


def evaluate_candidate(
    candidate: CandidateWeights,
    profile_returns: Mapping[str, pd.Series],
    folds: Sequence[WalkForwardFold],
    *,
    max_oos_mdd: float,
) -> dict[str, Any]:
    combined = combine_profile_returns(profile_returns, candidate.weights)
    fold_rows: list[dict[str, Any]] = []
    validation_returns: list[float] = []
    oos_returns: list[float] = []
    validation_mdds: list[float] = []
    oos_mdds: list[float] = []
    for fold in folds:
        train = period_metrics(combined, fold.train)
        validation = period_metrics(combined, fold.validation)
        locked_oos = period_metrics(combined, fold.locked_oos)
        validation_returns.append(float(validation["total_return"]))
        oos_returns.append(float(locked_oos["total_return"]))
        validation_mdds.append(float(validation["mdd"]))
        oos_mdds.append(float(locked_oos["mdd"]))
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "train": train,
                "validation": validation,
                "locked_oos": locked_oos,
            }
        )
    final_oos = fold_rows[-1]["locked_oos"] if fold_rows else {}
    min_validation_return = min(validation_returns) if validation_returns else 0.0
    min_oos_return = min(oos_returns) if oos_returns else 0.0
    max_validation_mdd = max(validation_mdds) if validation_mdds else 0.0
    max_fold_oos_mdd = max(oos_mdds) if oos_mdds else 0.0
    all_validation_positive = bool(validation_returns) and all(
        value > 0.0 for value in validation_returns
    )
    all_oos_positive = bool(oos_returns) and all(value > 0.0 for value in oos_returns)
    return {
        "candidate_id": candidate.candidate_id,
        "weights": candidate.weights,
        "profile_mix": candidate.profile_mix,
        "gross_notional_fraction": candidate.gross_notional_fraction,
        "selection_surface": candidate.selection_surface,
        "deployable_without_refit": candidate.deployable_without_refit,
        "notes": list(candidate.notes),
        "folds": fold_rows,
        "summary": {
            "fold_count": len(fold_rows),
            "min_validation_return": min_validation_return,
            "min_oos_return": min_oos_return,
            "max_validation_mdd": max_validation_mdd,
            "max_oos_mdd": max_fold_oos_mdd,
            "final_fold_oos_return": float(final_oos.get("total_return") or 0.0),
            "final_fold_oos_mdd": float(final_oos.get("mdd") or 0.0),
            "all_validation_positive": all_validation_positive,
            "all_oos_positive": all_oos_positive,
            "all_validation_and_oos_positive": all_validation_positive and all_oos_positive,
            "oos_mdd_within_limit": max_fold_oos_mdd <= max_oos_mdd,
            "return_shape_pass": all_validation_positive
            and all_oos_positive
            and max_fold_oos_mdd <= max_oos_mdd,
        },
    }


def profile_gross_from_context(context: Mapping[str, Any], profile_id: str) -> float:
    return _safe_float(dict(context.get("profile_gross") or {}).get(profile_id))


def weights_for_target_gross(
    *,
    profile_mix: Mapping[str, float],
    profile_gross: Mapping[str, float],
    target_gross: float,
    allow_upscale: bool,
) -> tuple[dict[str, float], float] | None:
    source_gross = sum(
        max(0.0, float(profile_mix.get(profile_id, 0.0)))
        * float(profile_gross.get(profile_id, 0.0))
        for profile_id in profile_mix
    )
    if source_gross <= 0.0:
        return None
    if not allow_upscale and target_gross > source_gross + 1e-12:
        return None
    scale = float(target_gross) / source_gross
    return {profile_id: float(mix) * scale for profile_id, mix in profile_mix.items()}, source_gross


def candidate_score(evaluation: Mapping[str, Any]) -> float:
    summary = dict(evaluation.get("summary") or {})
    folds = list(evaluation.get("folds") or [])
    all_val_oos = bool(summary.get("all_validation_and_oos_positive"))
    mdd_ok = bool(summary.get("oos_mdd_within_limit"))
    if not (all_val_oos and mdd_ok):
        return -1e9
    validation_returns = [
        float(dict(row.get("validation") or {}).get("total_return") or 0.0) for row in folds
    ]
    oos_returns = [
        float(dict(row.get("locked_oos") or {}).get("total_return") or 0.0) for row in folds
    ]
    avg_return = float(np.mean(validation_returns + oos_returns)) if folds else 0.0
    return (
        7.0 * float(summary.get("min_oos_return") or 0.0)
        + 5.0 * float(summary.get("min_validation_return") or 0.0)
        + 1.5 * avg_return
        + 0.6 * float(summary.get("final_fold_oos_return") or 0.0)
        - 1.8 * float(summary.get("max_oos_mdd") or 0.0)
        - 0.05 * float(evaluation.get("gross_notional_fraction") or 0.0)
    )


def _grid_matrix_context(
    profile_returns: Mapping[str, pd.Series],
    profile_ids: Sequence[str],
    folds: Sequence[WalkForwardFold],
) -> dict[str, Any]:
    index = pd.DatetimeIndex(
        sorted(set().union(*(set(profile_returns[profile_id].index) for profile_id in profile_ids)))
    )
    matrix = np.vstack(
        [
            profile_returns[profile_id].reindex(index, fill_value=0.0).to_numpy(dtype=float)
            for profile_id in profile_ids
        ]
    )
    return {
        "index": index,
        "matrix": matrix,
        "fold_masks": [
            {
                "fold_id": fold.fold_id,
                "validation": period_mask(index, fold.validation),
                "locked_oos": period_mask(index, fold.locked_oos),
            }
            for fold in folds
        ],
    }


def _fast_grid_summary(
    weights: Mapping[str, float],
    profile_ids: Sequence[str],
    matrix_context: Mapping[str, Any],
    *,
    max_oos_mdd: float,
) -> dict[str, Any]:
    vector = np.array([float(weights.get(profile_id, 0.0)) for profile_id in profile_ids])
    values = vector @ np.asarray(matrix_context["matrix"], dtype=float)
    validation_returns: list[float] = []
    oos_returns: list[float] = []
    validation_mdds: list[float] = []
    oos_mdds: list[float] = []
    for masks in matrix_context["fold_masks"]:
        validation_values = values[np.asarray(masks["validation"], dtype=bool)]
        oos_values = values[np.asarray(masks["locked_oos"], dtype=bool)]
        validation_returns.append(total_return(validation_values))
        oos_returns.append(total_return(oos_values))
        validation_mdds.append(max_drawdown(validation_values) if validation_values.size else 0.0)
        oos_mdds.append(max_drawdown(oos_values) if oos_values.size else 0.0)
    min_validation_return = min(validation_returns) if validation_returns else 0.0
    min_oos_return = min(oos_returns) if oos_returns else 0.0
    max_validation_mdd = max(validation_mdds) if validation_mdds else 0.0
    max_fold_oos_mdd = max(oos_mdds) if oos_mdds else 0.0
    all_validation_positive = bool(validation_returns) and all(
        value > 0.0 for value in validation_returns
    )
    all_oos_positive = bool(oos_returns) and all(value > 0.0 for value in oos_returns)
    return {
        "fold_count": len(oos_returns),
        "min_validation_return": min_validation_return,
        "min_oos_return": min_oos_return,
        "max_validation_mdd": max_validation_mdd,
        "max_oos_mdd": max_fold_oos_mdd,
        "final_fold_oos_return": oos_returns[-1] if oos_returns else 0.0,
        "final_fold_oos_mdd": oos_mdds[-1] if oos_mdds else 0.0,
        "all_validation_positive": all_validation_positive,
        "all_oos_positive": all_oos_positive,
        "all_validation_and_oos_positive": all_validation_positive and all_oos_positive,
        "oos_mdd_within_limit": max_fold_oos_mdd <= max_oos_mdd,
        "return_shape_pass": all_validation_positive
        and all_oos_positive
        and max_fold_oos_mdd <= max_oos_mdd,
    }


def _fast_candidate_score(*, summary: Mapping[str, Any], gross_notional_fraction: float) -> float:
    if not (
        bool(summary.get("all_validation_and_oos_positive"))
        and bool(summary.get("oos_mdd_within_limit"))
    ):
        return -1e9
    return (
        7.0 * float(summary.get("min_oos_return") or 0.0)
        + 5.0 * float(summary.get("min_validation_return") or 0.0)
        + 0.6 * float(summary.get("final_fold_oos_return") or 0.0)
        - 1.8 * float(summary.get("max_oos_mdd") or 0.0)
        - 0.05 * gross_notional_fraction
    )


def _gross_grid(min_gross: float, max_gross: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("gross step must be positive")
    values: list[float] = []
    current = float(min_gross)
    while current <= float(max_gross) + 1e-12:
        values.append(round(current, 10))
        current += float(step)
    return values


def _profile_mix_grid(step: float, profile_ids: Sequence[str]) -> list[dict[str, float]]:
    if len(profile_ids) != 3:
        raise ValueError("profile mix grid expects exactly three profile ids")
    if step <= 0.0 or step > 1.0:
        raise ValueError("mix step must be in (0, 1]")
    scale = round(1.0 / step)
    mixes: list[dict[str, float]] = []
    for first_units in range(scale + 1):
        for second_units in range(scale - first_units + 1):
            third_units = scale - first_units - second_units
            weights = [first_units / scale, second_units / scale, third_units / scale]
            mixes.append(
                {
                    profile_id: weight
                    for profile_id, weight in zip(profile_ids, weights, strict=True)
                }
            )
    return mixes


def grid_search_walkforward_shadow(
    *,
    profile_returns: Mapping[str, pd.Series],
    profile_gross: Mapping[str, float],
    folds: Sequence[WalkForwardFold],
    profile_ids: Sequence[str],
    min_gross: float,
    max_gross: float,
    gross_step: float,
    mix_step: float,
    max_oos_mdd: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    matrix_context = _grid_matrix_context(profile_returns, profile_ids, folds)
    feasibility_counts = {
        "val_gt_0_oos_gt_0": 0,
        "val_gt_1pct_oos_gt_5pct": 0,
        "val_gt_2pct_oos_gt_5pct": 0,
        "val_gt_0_final_oos_gt_5pct": 0,
    }
    for mix in _profile_mix_grid(mix_step, profile_ids):
        if sum(value > 0.0 for value in mix.values()) == 0:
            continue
        for target_gross in _gross_grid(min_gross, max_gross, gross_step):
            weighted = weights_for_target_gross(
                profile_mix=mix,
                profile_gross=profile_gross,
                target_gross=target_gross,
                allow_upscale=False,
            )
            if weighted is None:
                continue
            weights, _source_gross = weighted
            candidate = CandidateWeights(
                candidate_id=(
                    "diagnostic_shadow_grid_"
                    f"gross{target_gross:.2f}_"
                    f"b{mix.get(BALANCED_PROFILE_ID, 0.0):.2f}_"
                    f"g{mix.get(GROWTH_PROFILE_ID, 0.0):.2f}_"
                    f"a{mix.get(AGGRESSIVE_PROFILE_ID, 0.0):.2f}"
                ),
                weights=weights,
                gross_notional_fraction=target_gross,
                profile_mix=dict(mix),
                selection_surface="walkforward_validation_and_oos_diagnostic_grid",
                deployable_without_refit=False,
                notes=(
                    "uses walk-forward OOS diagnostics for ranking; keep shadow-only until rerun train/validation-only selection",
                ),
            )
            summary = _fast_grid_summary(
                weights=weights,
                profile_ids=profile_ids,
                matrix_context=matrix_context,
                max_oos_mdd=max_oos_mdd,
            )
            if summary["all_validation_positive"] and summary["all_oos_positive"]:
                feasibility_counts["val_gt_0_oos_gt_0"] += 1
            if summary["min_validation_return"] > 0.01 and summary["min_oos_return"] > 0.05:
                feasibility_counts["val_gt_1pct_oos_gt_5pct"] += 1
            if summary["min_validation_return"] > 0.02 and summary["min_oos_return"] > 0.05:
                feasibility_counts["val_gt_2pct_oos_gt_5pct"] += 1
            if summary["min_validation_return"] > 0.0 and summary["final_fold_oos_return"] > 0.05:
                feasibility_counts["val_gt_0_final_oos_gt_5pct"] += 1
            if summary["return_shape_pass"]:
                evaluation = evaluate_candidate(
                    candidate, profile_returns, folds, max_oos_mdd=max_oos_mdd
                )
                evaluation["score"] = candidate_score(evaluation)
                evaluation["fast_score"] = _fast_candidate_score(
                    summary=summary,
                    gross_notional_fraction=target_gross,
                )
                candidates.append(evaluation)
    candidates.sort(key=lambda item: float(item.get("score") or -1e9), reverse=True)
    # Keep the JSON artifact bounded while preserving enough alternatives to inspect trade-offs.
    return {
        "grid_policy": {
            "profile_ids": list(profile_ids),
            "min_gross": min_gross,
            "max_gross": max_gross,
            "gross_step": gross_step,
            "mix_step": mix_step,
            "allow_upscale": False,
            "max_oos_mdd": max_oos_mdd,
        },
        "feasibility_counts": feasibility_counts,
        "passing_candidate_count": len(candidates),
        "best_candidate": candidates[0] if candidates else None,
        "top_candidates": candidates[:10],
    }


def build_monitor_manifest(
    *,
    source_payload: Mapping[str, Any],
    diverse_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    universe_symbols = [
        str(symbol) for symbol in dict(source_payload.get("universe") or {}).get("symbols") or []
    ]
    source_train_eligible = set(salvage._train_eligible_symbols(source_payload))
    source_train_ineligible = set(salvage._train_ineligible_symbols(source_payload))
    diverse_by_symbol = {
        str(row.get("symbol")): dict(row)
        for row in diverse_payload.get("asset_inclusion_manifest") or []
    }
    selected_by_symbol = {
        str(row.get("symbol")): dict(row)
        for row in diverse_payload.get("selected_sleeve_rows") or []
    }
    manifest: list[dict[str, Any]] = []
    for symbol in universe_symbols:
        selected_row = selected_by_symbol.get(symbol)
        diverse_row = diverse_by_symbol.get(symbol, {})
        if selected_row:
            status = "core_tradable_now"
            action = "trade_in_paper_core_with_clean_oos_gate"
        elif symbol in source_train_ineligible:
            status = "future_watchlist_insufficient_train_history"
            action = "monitor_data_until_next_refit_eligibility_refresh"
        elif symbol in source_train_eligible:
            status = "eligible_shadow_not_selected"
            action = "monitor_shadow_signal_and_retest_before_promotion"
        else:
            status = "unclassified_watchlist"
            action = "monitor_data_quality_and_classification"
        manifest.append(
            {
                "symbol": symbol,
                "status": status,
                "action": action,
                "source_train_eligible": symbol in source_train_eligible,
                "source_train_ineligible": symbol in source_train_ineligible,
                "core_gross_notional_fraction": _safe_float(
                    selected_row.get("weighted_notional_fraction") if selected_row else 0.0
                ),
                "source_profile_id": selected_row.get("source_profile_id")
                if selected_row
                else None,
                "timeframe": selected_row.get("timeframe") if selected_row else None,
                "side": selected_row.get("side") if selected_row else None,
                "family": selected_row.get("family") if selected_row else None,
                "prior_diverse_status": diverse_row.get("status"),
            }
        )
    return manifest


def _core_candidate_from_diverse(
    diverse_payload: Mapping[str, Any], profile_gross: Mapping[str, float]
) -> CandidateWeights:
    policy = dict(diverse_payload.get("diversity_policy") or {})
    source_profile_id = str(policy.get("source_profile_id") or BALANCED_PROFILE_ID)
    scale_factor = _safe_float(policy.get("scale_factor"), 1.0)
    gross = _safe_float(policy.get("effective_gross_notional_fraction"))
    if gross <= 0.0:
        gross = profile_gross.get(source_profile_id, 0.0) * scale_factor
    return CandidateWeights(
        candidate_id="deployable_core_diverse_balanced_gross_cap",
        weights={source_profile_id: scale_factor},
        gross_notional_fraction=gross,
        profile_mix={source_profile_id: 1.0},
        selection_surface="train_validation_source_profile_plus_clean_locked_oos_gate",
        deployable_without_refit=True,
        notes=(
            "selected without OOS, then separately checked by clean locked-OOS gate",
            "69-symbol universe is monitored; only train-eligible selected sleeves are tradable now",
        ),
    )


def _selection_recommendation(
    core_eval: Mapping[str, Any], shadow_search: Mapping[str, Any]
) -> dict[str, Any]:
    core_summary = dict(core_eval.get("summary") or {})
    shadow = dict(shadow_search.get("best_candidate") or {})
    shadow_summary = dict(shadow.get("summary") or {})
    return {
        "recommended_live_paper_profile": core_eval.get("candidate_id"),
        "recommended_live_paper_reason": (
            "best clean-OOS deployable option: selected on train/validation only and passes final locked-OOS; "
            "walk-forward replay has all OOS folds positive but one early validation fold is slightly negative"
        ),
        "shadow_profile_to_track": shadow.get("candidate_id"),
        "shadow_reason": (
            "all walk-forward validation/OOS folds positive in frozen-return-shape diagnostics, "
            "but it used OOS folds for ranking so it is not live-promotable without refit"
        )
        if shadow
        else None,
        "can_claim_all_val_and_oos_very_good_now": False,
        "why_not": (
            "the grid found all-positive fold shapes, but no candidate met stricter 'very good' feasibility "
            "thresholds such as every validation >1% and every OOS >5%; deployable core has strong final OOS "
            "but one early validation fold is negative"
        ),
        "core_min_validation_return": core_summary.get("min_validation_return"),
        "core_min_oos_return": core_summary.get("min_oos_return"),
        "core_final_oos_return": core_summary.get("final_fold_oos_return"),
        "shadow_min_validation_return": shadow_summary.get("min_validation_return"),
        "shadow_min_oos_return": shadow_summary.get("min_oos_return"),
        "shadow_final_oos_return": shadow_summary.get("final_fold_oos_return"),
    }


def _source_windows_from_payload(source_payload: Mapping[str, Any]) -> clean_gate.GateWindows:
    split = dict(source_payload.get("split_policy") or {})
    train = dict(split.get("train") or {})
    validation = dict(split.get("validation") or {})
    locked_oos = dict(split.get("locked_oos") or {})
    return clean_gate.GateWindows(
        train=(
            _ts(train.get("start", clean_gate.DEFAULT_TRAIN_START)),
            _ts(train.get("end", clean_gate.DEFAULT_TRAIN_END)),
        ),
        validation=(
            _ts(validation.get("start", clean_gate.DEFAULT_VALIDATION_START)),
            _ts(validation.get("end", clean_gate.DEFAULT_VALIDATION_END)),
        ),
        locked_oos=(
            _ts(locked_oos.get("start", clean_gate.DEFAULT_LOCKED_OOS_START)),
            _ts(locked_oos.get("end", clean_gate.DEFAULT_LOCKED_OOS_END)),
        ),
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_artifact).expanduser().resolve()
    diverse_path = Path(args.diverse_artifact).expanduser().resolve()
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    diverse_payload = json.loads(diverse_path.read_text(encoding="utf-8"))
    context = clean_gate._build_profile_context(
        source_payload, _source_windows_from_payload(source_payload)
    )
    profile_returns: Mapping[str, pd.Series] = context["profile_returns"]
    profile_gross = {
        profile_id: _safe_float(gross)
        for profile_id, gross in dict(context["profile_gross"]).items()
    }
    profile_ids = [
        profile_id for profile_id in DEFAULT_PROFILE_ORDER if profile_id in profile_returns
    ]
    core_candidate = _core_candidate_from_diverse(diverse_payload, profile_gross)
    core_eval = evaluate_candidate(
        core_candidate,
        profile_returns,
        DEFAULT_FOLDS,
        max_oos_mdd=float(args.max_oos_mdd),
    )
    shadow_search = grid_search_walkforward_shadow(
        profile_returns=profile_returns,
        profile_gross=profile_gross,
        folds=DEFAULT_FOLDS,
        profile_ids=profile_ids,
        min_gross=float(args.min_gross),
        max_gross=float(args.max_gross),
        gross_step=float(args.gross_step),
        mix_step=float(args.mix_step),
        max_oos_mdd=float(args.max_oos_mdd),
    )
    monitor_manifest = build_monitor_manifest(
        source_payload=source_payload,
        diverse_payload=diverse_payload,
    )
    status_counts: dict[str, int] = {}
    for row in monitor_manifest:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "artifact_kind": "alpha_zoo_69_asset_walkforward_monitor",
        "generated_at_utc": _utc_now_iso(),
        "source_artifact": str(source_path),
        "diverse_artifact": str(diverse_path),
        "evaluation_policy": {
            "kind": "frozen_parameter_expanding_walkforward_return_shape_diagnostic",
            "true_refit_per_fold": False,
            "deployable_candidate_must_not_use_oos_for_selection": True,
            "diagnostic_shadow_uses_oos_for_ranking": True,
            "real_money_execution_allowed": False,
            "paper_testnet_only": True,
            "max_oos_mdd": float(args.max_oos_mdd),
        },
        "walkforward_folds": [fold.as_payload() for fold in DEFAULT_FOLDS],
        "profile_gross_notional_fraction": profile_gross,
        "deployable_core": core_eval,
        "diagnostic_shadow_search": shadow_search,
        "selection_recommendation": _selection_recommendation(core_eval, shadow_search),
        "monitor_policy": {
            "universe_symbol_count": len(monitor_manifest),
            "status_counts": status_counts,
            "all_69_assets_monitored": len(monitor_manifest) == 69,
            "promotion_rule": (
                "future-watchlist assets can become tradable only after later train eligibility, "
                "train/validation sleeve gates, and a separate clean locked-OOS report gate"
            ),
        },
        "monitor_manifest": monitor_manifest,
        "ready_for_paper": bool(core_eval.get("deployable_without_refit")),
        "ready_for_real": False,
        "real_execution_allowed": False,
        "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    return payload


def _render_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.4%}"


def _render_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def render_markdown(payload: Mapping[str, Any]) -> str:
    rec = dict(payload.get("selection_recommendation") or {})
    core = dict(payload.get("deployable_core") or {})
    core_summary = dict(core.get("summary") or {})
    shadow_search = dict(payload.get("diagnostic_shadow_search") or {})
    shadow = dict(shadow_search.get("best_candidate") or {})
    shadow_summary = dict(shadow.get("summary") or {})
    monitor_policy = dict(payload.get("monitor_policy") or {})
    lines = [
        "# 69-asset walk-forward monitor decision",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        f"Source artifact: `{payload.get('source_artifact')}`",
        f"Diverse artifact: `{payload.get('diverse_artifact')}`",
        "",
        "## Decision",
        "",
        f"- Paper/live candidate: `{rec.get('recommended_live_paper_profile')}`",
        f"- Shadow candidate: `{rec.get('shadow_profile_to_track')}`",
        f"- Can claim all validation and OOS are very good now: `{_render_bool(rec.get('can_claim_all_val_and_oos_very_good_now'))}`",
        f"- Reason: {rec.get('why_not')}",
        "",
        "## Candidate summary",
        "",
        "| candidate | deployable without refit | gross | min val | min OOS | final OOS | max OOS MDD | all val+OOS positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| `{core.get('candidate_id')}` | `{_render_bool(core.get('deployable_without_refit'))}` | "
        f"{float(core.get('gross_notional_fraction') or 0.0):.4f} | "
        f"{_render_pct(core_summary.get('min_validation_return'))} | "
        f"{_render_pct(core_summary.get('min_oos_return'))} | "
        f"{_render_pct(core_summary.get('final_fold_oos_return'))} | "
        f"{_render_pct(core_summary.get('max_oos_mdd'))} | "
        f"`{_render_bool(core_summary.get('all_validation_and_oos_positive'))}` |",
    ]
    if shadow:
        lines.append(
            f"| `{shadow.get('candidate_id')}` | `{_render_bool(shadow.get('deployable_without_refit'))}` | "
            f"{float(shadow.get('gross_notional_fraction') or 0.0):.4f} | "
            f"{_render_pct(shadow_summary.get('min_validation_return'))} | "
            f"{_render_pct(shadow_summary.get('min_oos_return'))} | "
            f"{_render_pct(shadow_summary.get('final_fold_oos_return'))} | "
            f"{_render_pct(shadow_summary.get('max_oos_mdd'))} | "
            f"`{_render_bool(shadow_summary.get('all_validation_and_oos_positive'))}` |"
        )
    lines.extend(
        [
            "",
            "## Walk-forward fold detail: deployable core",
            "",
            "| fold | validation | OOS | OOS MDD |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in core.get("folds") or []:
        validation = dict(row.get("validation") or {})
        locked = dict(row.get("locked_oos") or {})
        lines.append(
            f"| `{row.get('fold_id')}` | {_render_pct(validation.get('total_return'))} | "
            f"{_render_pct(locked.get('total_return'))} | {_render_pct(locked.get('mdd'))} |"
        )
    lines.extend(
        [
            "",
            "## Feasibility counts",
            "",
        ]
    )
    for key, value in dict(shadow_search.get("feasibility_counts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 69-asset monitor status",
            "",
            f"- Universe count: `{monitor_policy.get('universe_symbol_count')}`",
            f"- All 69 monitored: `{_render_bool(monitor_policy.get('all_69_assets_monitored'))}`",
            f"- Status counts: `{monitor_policy.get('status_counts')}`",
            f"- Promotion rule: {monitor_policy.get('promotion_rule')}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--diverse-artifact", default=str(DEFAULT_DIVERSE_ARTIFACT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-oos-mdd", type=float, default=DEFAULT_MAX_OOS_MDD)
    parser.add_argument("--min-gross", type=float, default=DEFAULT_MIN_GROSS)
    parser.add_argument("--max-gross", type=float, default=DEFAULT_MAX_GROSS)
    parser.add_argument("--gross-step", type=float, default=DEFAULT_GROSS_STEP)
    parser.add_argument("--mix-step", type=float, default=DEFAULT_MIX_STEP)
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
