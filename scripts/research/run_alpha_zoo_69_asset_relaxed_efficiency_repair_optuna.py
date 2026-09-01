#!/usr/bin/env python3
"""MDD-guarded relaxed efficiency repair pass for the 69-asset Alpha Zoo hybrid.

This runner is intentionally a sibling of
``run_alpha_zoo_69_asset_efficiency_repair_optuna``.  It keeps the source
per-asset/profile Optuna parameters frozen and retunes only portfolio sleeve and
hybrid allocations, but applies the policy update from 2026-05-31:

* train < validation is not a hard rejection when both train and validation are
  meaningfully positive and MDD remains within the profile guard;
* low train sample is acceptable for TradFi-like sleeves under the same MDD/RPT
  guard, and material-positive low-sample rows are allowed as warnings rather
  than outright rejects;
* gross/concentration pressure is penalized, not rejected, when validation MDD is
  not high;
* the 10bps return-per-turnover execution-efficiency gate stays strict;
* train-ineligible symbols remain excluded from live/paper promotion;
* real-money execution remains disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.optimization.search_policy import (  # noqa: E402
    optimization_search_policy_payload,
    run_optuna_study,
)
from scripts.research import run_alpha_zoo_69_asset_efficiency_repair_optuna as strict  # noqa: E402

try:  # pragma: no cover - exercised by runtime dependency checks.
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore[assignment]

broad69 = strict.broad69
profile69 = strict.profile69
grid_hybrid = strict.grid_hybrid
optuna_hybrid = strict.optuna_hybrid

DEFAULT_SOURCE_ARTIFACT = strict.DEFAULT_SOURCE_ARTIFACT
DEFAULT_OUTPUT_DIR = (
    broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_relaxed_efficiency_repair_optuna_20260531"
)
DEFAULT_PROFILE_TRIALS = strict.DEFAULT_PROFILE_TRIALS
DEFAULT_HYBRID_TRIALS = strict.DEFAULT_HYBRID_TRIALS
PRIMARY_COST_BPS = strict.PRIMARY_COST_BPS
STRESS_COST_BPS = strict.STRESS_COST_BPS
RELAXED_POLICY_ID = "material_positive_tradfi_low_sample_mdd_guard_20260531"
MIN_MATERIAL_RETURN = 0.02
RELAXED_HYBRID_MAX_VALIDATION_MDD = 0.20
RELAXED_HYBRID_MAX_TRAIN_MDD = 0.60
RELAXED_HARD_GROSS_CAP = 12.0

RELAXED_PROFILE_SPECS: tuple[dict[str, Any], ...] = tuple(
    {
        **spec,
        "profile_id": str(spec["profile_id"]).replace(
            "69_asset_efficiency_repair_optuna", "69_asset_relaxed_efficiency_repair_optuna"
        ),
        "relaxed_policy_id": RELAXED_POLICY_ID,
        "min_material_return": MIN_MATERIAL_RETURN,
        "dominance_relaxation": "allow_train_below_validation_when_both_returns_ge_2pct_and_mdd_guard_passes",
        "low_sample_relaxation": "allow_tradfi_or_material_positive_low_sample_when_mdd_guard_and_10bps_rpt_pass",
        "rpt_gate": "strict_gt_10bps_train_and_validation",
    }
    for spec in strict.EFFICIENCY_PROFILE_SPECS
)

CANDIDATE_FIELDS = [
    *strict.CANDIDATE_FIELDS,
    "relaxed_policy_id",
    "relaxations_applied",
]
SLEEVE_FIELDS = [
    *CANDIDATE_FIELDS,
    "sleeve_multiplier",
    "weighted_notional_fraction",
    "profile_weight_rank",
]
PROFILE_FIELDS = [
    *SLEEVE_FIELDS,
    "profile_kind",
    "gross_notional_fraction",
    "selected_sleeve_count",
    "train_return_stress_15bps_proxy",
    "validation_return_stress_15bps_proxy",
    "train_return_stress_20bps_proxy",
    "validation_return_stress_20bps_proxy",
    "low_sample_notional_share",
    "unguarded_low_sample_notional_share",
    "relaxed_low_sample_notional_share",
    "validation_spike_notional_share",
    "relaxed_validation_spike_notional_share",
    "low_efficiency_notional_share",
    "relaxed_notional_share",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    return strict._json_safe(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return strict._safe_float(value, default)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", "utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return _json_safe(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(dict.fromkeys(fields)),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in writer.fieldnames})


def _asset_group(row: Mapping[str, Any]) -> str:
    value = row.get("asset_group")
    if value:
        return str(value)
    symbol = str(row.get("symbol") or "")
    return str(broad69._asset_group(symbol)) if symbol else "unknown"


def _is_tradfi(row: Mapping[str, Any]) -> bool:
    return _asset_group(row).startswith("tradfi")


def _material_positive(row: Mapping[str, Any], *, threshold: float = MIN_MATERIAL_RETURN) -> bool:
    return (
        _safe_float(row.get("train_return")) >= threshold
        and _safe_float(row.get("validation_return")) >= threshold
    )


def _mdd_guard_pass(row: Mapping[str, Any], spec: Mapping[str, Any] | None = None) -> bool:
    max_validation_mdd = (
        _safe_float(spec.get("max_validation_mdd"), RELAXED_HYBRID_MAX_VALIDATION_MDD)
        if spec is not None
        else RELAXED_HYBRID_MAX_VALIDATION_MDD
    )
    max_train_mdd = (
        _safe_float(spec.get("max_train_mdd"), RELAXED_HYBRID_MAX_TRAIN_MDD)
        if spec is not None
        else RELAXED_HYBRID_MAX_TRAIN_MDD
    )
    return (
        _safe_float(row.get("validation_mdd")) <= max_validation_mdd
        and _safe_float(row.get("train_mdd")) <= max_train_mdd
    )


def _dominance_relaxed(row: Mapping[str, Any], spec: Mapping[str, Any] | None = None) -> bool:
    return (
        _safe_float(row.get("train_return")) < _safe_float(row.get("validation_return"))
        and _material_positive(row)
        and _mdd_guard_pass(row, spec)
    )


def _low_sample_relaxed(row: Mapping[str, Any], spec: Mapping[str, Any], *, split: str) -> bool:
    if not _mdd_guard_pass(row, spec):
        return False
    if (
        _safe_float(row.get("train_return")) <= 0.0
        or _safe_float(row.get("validation_return")) <= 0.0
    ):
        return False
    if _is_tradfi(row):
        return True
    return _material_positive(row)


def _candidate_relaxations(row: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    relaxations: list[str] = []
    train_events = int(row.get("train_trade_event_count") or 0)
    val_events = int(row.get("validation_trade_event_count") or 0)
    if _dominance_relaxed(row, spec):
        relaxations.append("train_below_validation_relaxed_material_positive_mdd_guarded")
    if train_events < int(spec["min_candidate_train_events"]) and _low_sample_relaxed(
        row, spec, split="train"
    ):
        if _is_tradfi(row):
            relaxations.append("low_train_sample_relaxed_tradfi_mdd_guarded")
        else:
            relaxations.append("low_train_sample_relaxed_material_positive_mdd_guarded")
    if val_events < int(spec["min_candidate_validation_events"]) and _low_sample_relaxed(
        row, spec, split="validation"
    ):
        if _is_tradfi(row):
            relaxations.append("low_validation_sample_relaxed_tradfi_mdd_guarded")
        else:
            relaxations.append("low_validation_sample_relaxed_material_positive_mdd_guarded")
    return relaxations


def _candidate_repair_reasons(row: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    relaxations = set(_candidate_relaxations(row, spec))
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_events = int(row.get("train_trade_event_count") or 0)
    val_events = int(row.get("validation_trade_event_count") or 0)
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -1e9)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -1e9)
    if train <= 0.0:
        reasons.append("train_return_not_positive")
    if validation < _safe_float(spec.get("min_validation_return"), MIN_MATERIAL_RETURN):
        reasons.append(
            f"validation_return_{validation:.4f}_below_{_safe_float(spec.get('min_validation_return'), MIN_MATERIAL_RETURN):.4f}"
        )
    if (
        train < validation
        and "train_below_validation_relaxed_material_positive_mdd_guarded" not in relaxations
    ):
        reasons.append("train_below_validation_spike_risk")
    if train_events < int(spec["min_candidate_train_events"]) and not any(
        item.startswith("low_train_sample_relaxed") for item in relaxations
    ):
        reasons.append(f"train_events_{train_events}_below_{spec['min_candidate_train_events']}")
    if val_events < int(spec["min_candidate_validation_events"]) and not any(
        item.startswith("low_validation_sample_relaxed") for item in relaxations
    ):
        reasons.append(
            f"validation_events_{val_events}_below_{spec['min_candidate_validation_events']}"
        )
    if train_rpt <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        reasons.append(f"train_rpt_{train_rpt:.3f}_not_above_10bps")
    if val_rpt <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        reasons.append(f"validation_rpt_{val_rpt:.3f}_not_above_10bps")
    if not _mdd_guard_pass(row, spec):
        # MDD is already checked in profile portfolio selection, but candidate-level
        # failure blocks using a row as a relaxed exception source.
        if _safe_float(row.get("validation_mdd")) > _safe_float(spec.get("max_validation_mdd")):
            reasons.append("validation_mdd_above_profile_guard")
        if _safe_float(row.get("train_mdd")) > _safe_float(spec.get("max_train_mdd")):
            reasons.append("train_mdd_above_profile_guard")
    return list(dict.fromkeys(reasons))


def _candidate_live_efficiency_score(row: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_mdd = _safe_float(row.get("train_mdd"))
    val_mdd = _safe_float(row.get("validation_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    train_events = int(row.get("train_trade_event_count") or 0)
    val_events = int(row.get("validation_trade_event_count") or 0)
    anchor_abs = _safe_float(row.get("dominant_anchor_abs_corr"))
    spike = max(0.0, validation - train)
    target_train_rpt = float(spec["target_train_rpt_bps"])
    target_val_rpt = float(spec["target_validation_rpt_bps"])
    relaxations = set(_candidate_relaxations(row, spec))
    train_shortfall = max(0, int(spec["min_candidate_train_events"]) - train_events)
    val_shortfall = max(0, int(spec["min_candidate_validation_events"]) - val_events)
    relaxed_sample = any("sample_relaxed" in item for item in relaxations)
    penalty = 0.0
    if "train_below_validation_relaxed_material_positive_mdd_guarded" not in relaxations:
        penalty += 35.0 * spike
    else:
        penalty += 4.0 * spike
    penalty += max(0.0, -train) * 14.0 + max(0.0, -validation) * 16.0
    penalty += max(0.0, float(spec["min_validation_return"]) - validation) * 18.0
    penalty += max(0.0, val_mdd - float(spec["max_validation_mdd"])) * 30.0
    penalty += max(0.0, train_mdd - float(spec["max_train_mdd"])) * 10.0
    penalty += max(0.0, target_train_rpt - train_rpt) / 7.0
    penalty += max(0.0, target_val_rpt - val_rpt) / 5.0
    sample_scale = 0.20 if relaxed_sample else 1.0
    penalty += sample_scale * (train_shortfall / 8.0 + val_shortfall / 3.0)
    penalty += max(0.0, anchor_abs - 0.70) * 2.5
    if train_events + val_events > 450 and min(train_rpt, val_rpt) < 25.0:
        penalty += (train_events + val_events - 450) / 700.0
    return float(
        9.0 * min(train, validation)
        + 4.8 * validation
        + 1.4 * train
        + min(max(train_rpt, -30.0), 180.0) / 175.0
        + min(max(val_rpt, -30.0), 180.0) / 115.0
        - 2.4 * val_mdd
        - 0.7 * train_mdd
        - penalty
    )


def _allocatable_relaxed_streams(
    candidate_streams: Sequence[broad69.CandidateStream],
) -> list[broad69.CandidateStream]:
    return [
        stream
        for stream in candidate_streams
        if not list(stream.row.get("efficiency_repair_reasons") or [])
    ]


def _build_candidate_streams(
    *,
    source_payload: Mapping[str, Any],
    spec: Mapping[str, Any],
    cache: profile69.FeatureCache,
    windows: broad69.SplitWindows,
    train_eligibility: Mapping[str, Any],
) -> list[broad69.CandidateStream]:
    out: list[broad69.CandidateStream] = []
    source_profile_id = str(spec["source_profile_id"])
    for raw in source_payload.get("asset_tuning_rows", []):
        if str(raw.get("profile_id")) != source_profile_id:
            continue
        if not strict._source_row_train_eligible(raw, train_eligibility):
            continue
        params = raw.get("optuna_params") or {}
        if not isinstance(params, Mapping) or not params:
            continue
        stream = profile69._candidate_from_params(
            symbol=str(raw["symbol"]),
            profile_id=str(spec["profile_id"]),
            params=params,
            cache=cache,
            windows=windows,
            allocation_fraction=_safe_float(raw.get("allocation_fraction"), 0.10),
        )
        row = dict(stream.row)
        row["source_profile_id"] = source_profile_id
        row["source_model_id"] = raw.get("model_id")
        row["optuna_params"] = dict(params)
        row["asset_group"] = _asset_group(row)
        row["relaxed_policy_id"] = RELAXED_POLICY_ID
        row["relaxations_applied"] = _candidate_relaxations(row, spec)
        row["live_efficiency_score"] = _candidate_live_efficiency_score(row, spec)
        row["efficiency_repair_reasons"] = _candidate_repair_reasons(row, spec)
        out.append(
            broad69.CandidateStream(row=row, returns=stream.returns, position=stream.position)
        )
    return out


def _allocation_quality_payload(
    streams: Sequence[broad69.CandidateStream], mult: np.ndarray, spec: Mapping[str, Any]
) -> dict[str, float]:
    total = 0.0
    low_sample = 0.0
    unguarded_low_sample = 0.0
    relaxed_low_sample = 0.0
    validation_spike = 0.0
    relaxed_validation_spike = 0.0
    low_efficiency = 0.0
    relaxed_notional = 0.0
    for idx, stream in enumerate(streams):
        notional = float(mult[idx]) * _safe_float(stream.row.get("notional_fraction"))
        if notional <= 1e-12:
            continue
        total += notional
        row = stream.row
        relaxations = set(row.get("relaxations_applied") or [])
        has_low_sample = int(row.get("train_trade_event_count") or 0) < int(
            spec["min_candidate_train_events"]
        ) or int(row.get("validation_trade_event_count") or 0) < int(
            spec["min_candidate_validation_events"]
        )
        if has_low_sample:
            low_sample += notional
            if any("sample_relaxed" in item for item in relaxations):
                relaxed_low_sample += notional
            else:
                unguarded_low_sample += notional
        if _safe_float(row.get("train_return")) < _safe_float(row.get("validation_return")):
            validation_spike += notional
            if "train_below_validation_relaxed_material_positive_mdd_guarded" in relaxations:
                relaxed_validation_spike += notional
        if (
            _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0) <= 20.0
            or _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0) <= 20.0
        ):
            low_efficiency += notional
        if relaxations:
            relaxed_notional += notional
    if total <= 0.0:
        return {
            "low_sample_notional_share": 0.0,
            "unguarded_low_sample_notional_share": 0.0,
            "relaxed_low_sample_notional_share": 0.0,
            "validation_spike_notional_share": 0.0,
            "relaxed_validation_spike_notional_share": 0.0,
            "low_efficiency_notional_share": 0.0,
            "relaxed_notional_share": 0.0,
        }
    return {
        "low_sample_notional_share": float(low_sample / total),
        "unguarded_low_sample_notional_share": float(unguarded_low_sample / total),
        "relaxed_low_sample_notional_share": float(relaxed_low_sample / total),
        "validation_spike_notional_share": float(validation_spike / total),
        "relaxed_validation_spike_notional_share": float(relaxed_validation_spike / total),
        "low_efficiency_notional_share": float(low_efficiency / total),
        "relaxed_notional_share": float(relaxed_notional / total),
    }


def _selection_reasons(
    row: Mapping[str, Any],
    *,
    max_gross: float = 8.0,
    spec: Mapping[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    validation_mdd = _safe_float(row.get("validation_mdd"))
    train_mdd = _safe_float(row.get("train_mdd"))
    max_validation_mdd = (
        _safe_float(spec.get("max_validation_mdd"), RELAXED_HYBRID_MAX_VALIDATION_MDD)
        if spec is not None
        else RELAXED_HYBRID_MAX_VALIDATION_MDD
    )
    max_train_mdd = (
        _safe_float(spec.get("max_train_mdd"), RELAXED_HYBRID_MAX_TRAIN_MDD)
        if spec is not None
        else RELAXED_HYBRID_MAX_TRAIN_MDD
    )
    if train <= 0.0:
        reasons.append("train_return_not_positive")
    if validation < MIN_MATERIAL_RETURN:
        reasons.append(f"validation_return_{validation:.4f}_below_{MIN_MATERIAL_RETURN:.4f}")
    if train < validation and not _dominance_relaxed(row, spec):
        reasons.append(f"train_return_{train:.4f}_below_validation_return_{validation:.4f}")
    if validation_mdd > max_validation_mdd:
        reasons.append(f"validation_mdd_{validation_mdd:.4f}_above_{max_validation_mdd:.4f}")
    if train_mdd > max_train_mdd:
        reasons.append(f"train_mdd_{train_mdd:.4f}_above_{max_train_mdd:.4f}")
    gross = _safe_float(row.get("gross_notional_fraction"))
    if gross > max_gross and (
        gross > RELAXED_HARD_GROSS_CAP or validation_mdd > max_validation_mdd
    ):
        reasons.append(f"gross_notional_{gross:.4f}_above_{max_gross:.4f}_without_mdd_guard")
    for split in broad69.SPLIT_ORDER:
        if int(row.get(f"{split}_liquidation_count") or 0) != 0:
            reasons.append(f"{split}_liquidation_count_nonzero")
        if int(row.get(f"{split}_account_wipeout_count") or 0) != 0:
            reasons.append(f"{split}_account_wipeout_count_nonzero")
        rpt = row.get(f"{split}_return_per_turnover_proxy_bps")
        if rpt is None or _safe_float(rpt) <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
            rendered = "missing" if rpt is None else f"{_safe_float(rpt):.3f}"
            reasons.append(f"{split}_return_per_turnover_proxy_bps_{rendered}_not_above_10bps")
        for cost_bps in STRESS_COST_BPS:
            value = _safe_float(row.get(f"{split}_return_stress_{int(cost_bps)}bps_proxy"))
            if value <= 0.0:
                reasons.append(f"{split}_return_stress_{int(cost_bps)}bps_proxy_not_positive")
    return list(dict.fromkeys(reasons))


def _diagnostic_warnings(row: Mapping[str, Any]) -> list[str]:
    warnings = strict._diagnostic_warnings(row)
    if _safe_float(row.get("relaxed_notional_share")) > 0.50:
        warnings.append("relaxed_notional_share_above_50pct")
    if _safe_float(row.get("relaxed_low_sample_notional_share")) > 0.35:
        warnings.append("relaxed_low_sample_notional_share_above_35pct")
    if _safe_float(row.get("relaxed_validation_spike_notional_share")) > 0.35:
        warnings.append("relaxed_validation_spike_notional_share_above_35pct")
    return list(dict.fromkeys(warnings))


def objective_value_from_row(row: Mapping[str, Any]) -> float:
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    val_mdd = _safe_float(row.get("validation_mdd"))
    train_mdd = _safe_float(row.get("train_mdd"))
    spike = max(0.0, validation - train)
    spike_penalty = 0.8 * spike if _dominance_relaxed(row) else 5.0 * spike
    return float(
        11.0 * validation
        + 2.5 * min(train, validation)
        + 1.0 * train
        + min(train_rpt, 180.0) / 150.0
        + min(val_rpt, 180.0) / 100.0
        - spike_penalty
        - 3.5 * val_mdd
        - 0.7 * train_mdd
        - 1.2 * _safe_float(row.get("low_efficiency_notional_share"))
        - 0.35 * _safe_float(row.get("relaxed_low_sample_notional_share"))
        - 1.6 * _safe_float(row.get("unguarded_low_sample_notional_share"))
    )


def _relaxed_hybrid_objective_score(row: Mapping[str, Any]) -> float:
    base = objective_value_from_row(row)
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -1e9)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -1e9)
    penalty = 0.0
    if train <= 0.0:
        penalty += 10.0 + abs(train) * 20.0
    if validation < MIN_MATERIAL_RETURN:
        penalty += 8.0 + (MIN_MATERIAL_RETURN - validation) * 20.0
    if _safe_float(row.get("validation_mdd")) > RELAXED_HYBRID_MAX_VALIDATION_MDD:
        penalty += (
            5.0
            + (_safe_float(row.get("validation_mdd")) - RELAXED_HYBRID_MAX_VALIDATION_MDD) * 10.0
        )
    if _safe_float(row.get("train_mdd")) > RELAXED_HYBRID_MAX_TRAIN_MDD:
        penalty += 2.0 + (_safe_float(row.get("train_mdd")) - RELAXED_HYBRID_MAX_TRAIN_MDD) * 5.0
    if train_rpt <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        penalty += 4.0 + max(0.0, broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS - train_rpt) / 5.0
    if val_rpt <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        penalty += 4.0 + max(0.0, broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS - val_rpt) / 5.0
    gross = _safe_float(row.get("gross_notional_fraction"))
    if gross > RELAXED_HARD_GROSS_CAP:
        penalty += (gross - RELAXED_HARD_GROSS_CAP) * 4.0
    return float(base - penalty)


@contextmanager
def _relaxed_optuna_hybrid_objective_context():
    previous = optuna_hybrid._objective_score
    optuna_hybrid._objective_score = _relaxed_hybrid_objective_score
    try:
        yield
    finally:
        optuna_hybrid._objective_score = previous


def tune_relaxed_profile_allocations(
    *,
    spec: Mapping[str, Any],
    candidate_streams: Sequence[broad69.CandidateStream],
    windows: broad69.SplitWindows,
    n_trials: int,
    seed: int,
) -> tuple[grid_hybrid.ProfileStream, dict[str, Any], list[dict[str, Any]]]:
    if optuna is None:
        raise RuntimeError("Optuna is required for relaxed efficiency repair tuning")
    profile_id = str(spec["profile_id"])
    ranked = sorted(
        _allocatable_relaxed_streams(candidate_streams),
        key=lambda stream: _safe_float(stream.row.get("live_efficiency_score"), -1e18),
        reverse=True,
    )[: int(spec["candidate_pool_size"])]
    if not ranked:
        raise ValueError(f"no candidate streams for {profile_id}")
    matrix = profile69._aligned_matrix(ranked)
    values = matrix.to_numpy(dtype=float)
    notionals = np.array([_safe_float(stream.row.get("notional_fraction")) for stream in ranked])

    def normalize(raw: np.ndarray) -> np.ndarray:
        raw = np.where(raw < 0.04, 0.0, raw)
        if float(raw.sum()) <= 0.0:
            raw[
                int(np.argmax([_safe_float(s.row.get("live_efficiency_score")) for s in ranked]))
            ] = 1.0
        gross = float(np.dot(raw, notionals))
        max_gross = float(spec["max_gross_notional"])
        if gross > max_gross and gross > 0.0:
            raw *= max_gross / gross
        return raw

    def row_from_multipliers(
        mult: np.ndarray,
    ) -> tuple[dict[str, Any], pd.Series, dict[str, float], dict[str, int]]:
        returns = pd.Series(values @ mult, index=matrix.index).sort_index()
        turnover, events = strict._weighted_turnover_events(ranked, mult)
        metrics = profile69._profile_metrics_from_returns(
            returns, windows=windows, turnover_by_split=turnover, events_by_split=events
        )
        metrics.update(strict._stress_metrics(metrics, turnover))
        metrics.update(profile69._profile_concentration(ranked, mult))
        metrics.update(_allocation_quality_payload(ranked, mult, spec))
        return metrics, returns, turnover, events

    def objective(trial: Any) -> float:
        raw = normalize(
            np.array([trial.suggest_float(f"m_{idx}", 0.0, 1.0) for idx in range(len(ranked))])
        )
        metrics, _, _, _ = row_from_multipliers(raw)
        train = _safe_float(metrics.get("train_return"))
        validation = _safe_float(metrics.get("validation_return"))
        train_rpt = _safe_float(metrics.get("train_return_per_turnover_proxy_bps"), -100.0)
        val_rpt = _safe_float(metrics.get("validation_return_per_turnover_proxy_bps"), -100.0)
        val_mdd = _safe_float(metrics.get("validation_mdd"))
        train_mdd = _safe_float(metrics.get("train_mdd"))
        spike = max(0.0, validation - train)
        dominance_ok = _material_positive(metrics) and _mdd_guard_pass(metrics, spec)
        mdd_ok = val_mdd <= float(spec["max_validation_mdd"])
        penalty = 0.0
        penalty += (5.0 if dominance_ok else 45.0) * spike
        penalty += max(0.0, float(spec["min_validation_return"]) - validation) * 24.0
        penalty += max(0.0, val_mdd - float(spec["max_validation_mdd"])) * 32.0
        penalty += max(0.0, train_mdd - float(spec["max_train_mdd"])) * 8.0
        penalty += max(0.0, float(spec["target_train_rpt_bps"]) - train_rpt) / 5.0
        penalty += max(0.0, float(spec["target_validation_rpt_bps"]) - val_rpt) / 4.0
        concentration_scale = 0.35 if mdd_ok else 1.0
        penalty += concentration_scale * (
            max(
                0.0,
                _safe_float(metrics.get("top_symbol_share")) - float(spec["top_symbol_share_cap"]),
            )
            * 5.0
        )
        penalty += max(0.0, _safe_float(metrics.get("top_anchor_share")) - 0.42) * 2.5
        penalty += concentration_scale * (
            max(
                0.0,
                _safe_float(metrics.get("top_asset_group_share"))
                - float(spec["top_asset_group_share_cap"]),
            )
            * 3.0
        )
        penalty += (
            max(
                0.0,
                _safe_float(metrics.get("unguarded_low_sample_notional_share"))
                - float(spec["low_sample_weight_cap"]),
            )
            * 3.5
        )
        penalty += _safe_float(metrics.get("relaxed_low_sample_notional_share")) * 0.45
        unrelaxed_spike_share = max(
            0.0,
            _safe_float(metrics.get("validation_spike_notional_share"))
            - _safe_float(metrics.get("relaxed_validation_spike_notional_share")),
        )
        penalty += unrelaxed_spike_share * 2.0
        penalty += _safe_float(metrics.get("relaxed_validation_spike_notional_share")) * 0.35
        penalty += _safe_float(metrics.get("low_efficiency_notional_share")) * 2.5
        for split in broad69.SPLIT_ORDER:
            if _safe_float(metrics.get(f"{split}_return_stress_20bps_proxy")) <= 0.0:
                penalty += 8.0
        trial.set_user_attr("train_return", train)
        trial.set_user_attr("validation_return", validation)
        trial.set_user_attr("train_rpt", train_rpt)
        trial.set_user_attr("validation_rpt", val_rpt)
        trial.set_user_attr("relaxed_share", metrics.get("relaxed_notional_share"))
        return float(
            12.0 * min(train, validation)
            + 4.4 * validation
            + 1.6 * train
            + min(train_rpt, 180.0) / 150.0
            + min(val_rpt, 180.0) / 100.0
            - 3.5 * val_mdd
            - 0.8 * train_mdd
            - penalty
        )

    enqueue: list[dict[str, float]] = []
    equal_value = min(1.0, float(spec["max_gross_notional"]) / max(1.0, len(ranked)))
    enqueue.append({f"m_{idx}": equal_value for idx in range(len(ranked))})
    for idx in range(min(len(ranked), 8)):
        enqueue.append({f"m_{j}": 1.0 if j == idx else 0.0 for j in range(len(ranked))})
    study = run_optuna_study(
        optuna_module=optuna,
        objective=objective,
        n_trials=n_trials,
        direction="maximize",
        seed=seed,
        enqueue_trials=enqueue,
        n_jobs=1,
        show_progress_bar=False,
    )
    best = normalize(
        np.array([float(study.best_params.get(f"m_{idx}", 0.0)) for idx in range(len(ranked))])
    )
    metrics, returns, turnover, events = row_from_multipliers(best)
    selected_rows: list[dict[str, Any]] = []
    asset_gross: dict[str, float] = defaultdict(float)
    leverage_map: dict[str, int] = {}
    model_ids: list[str] = []
    for rank, (mult, stream) in enumerate(
        sorted(
            [(float(m), s) for m, s in zip(best, ranked, strict=True) if float(m) > 1e-6],
            key=lambda item: item[0] * _safe_float(item[1].row.get("notional_fraction")),
            reverse=True,
        ),
        start=1,
    ):
        row = dict(stream.row)
        notional = mult * _safe_float(row.get("notional_fraction"))
        row.update(
            {
                "sleeve_multiplier": mult,
                "weighted_notional_fraction": notional,
                "profile_weight_rank": rank,
            }
        )
        selected_rows.append(row)
        symbol = str(row["symbol"])
        asset_gross[symbol] += notional
        leverage_map[symbol] = max(
            int(leverage_map.get(symbol, 0)), int(row.get("integer_leverage") or 0)
        )
        model_ids.append(str(row["model_id"]))
    stream = grid_hybrid.ProfileStream(
        profile_id=profile_id,
        candidate_tier="per_asset_profile_relaxed_efficiency_repair_source_profile",
        leverage_map=leverage_map,
        gross_notional_fraction=_safe_float(metrics.get("gross_notional_fraction")),
        asset_gross_notional_fraction=dict(sorted(asset_gross.items())),
        selected_model_ids=tuple(model_ids),
        returns=returns,
        turnover_by_split=turnover,
        trade_events_by_split=events,
        liquidation_count_by_split=dict.fromkeys(grid_hybrid.ilp.SPLIT_ORDER, 0),
    )
    row: dict[str, Any] = {
        "profile_id": profile_id,
        "source_profile_id": spec["source_profile_id"],
        "profile_kind": "per_asset_profile_relaxed_efficiency_repair_optuna",
        "candidate_tier": "paper_testnet_relaxed_efficiency_repair_candidate",
        "leverage_map": leverage_map,
        "weights": {profile_id: 1.0},
        "gross_notional_fraction": stream.gross_notional_fraction,
        "selected_sleeve_count": len(selected_rows),
        "optimizer": "optuna_tpe_mdd_guarded_relaxed_efficiency_repair_sleeve_allocation",
        "best_value": float(study.best_value),
        "profile_spec": dict(spec),
        "relaxed_policy_id": RELAXED_POLICY_ID,
        "concentration": {
            key: metrics.get(key)
            for key in (
                "top_symbol",
                "top_symbol_share",
                "top_asset_group",
                "top_asset_group_share",
                "effective_symbol_count",
                "symbol_shares",
                "asset_group_shares",
                "family_shares",
                "anchor_shares",
                "top_anchor",
                "top_anchor_share",
            )
        },
        "report_only_gate_reasons": [],
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "locked_oos_return_report_only": 0.0,
        "locked_oos_mdd_report_only": 0.0,
        "locked_oos_trade_event_count_report_only": 0,
        "locked_oos_return_per_turnover_proxy_bps_report_only": None,
        "locked_oos_liquidation_count_report_only": 0,
        "locked_oos_account_wipeout_count_report_only": 0,
    }
    for split in broad69.SPLIT_ORDER:
        row[f"{split}_return"] = metrics[f"{split}_return"]
        row[f"{split}_mdd"] = metrics[f"{split}_mdd"]
        row[f"{split}_trade_event_count"] = metrics[f"{split}_trade_event_count"]
        row[f"{split}_return_per_turnover_proxy_bps"] = metrics[
            f"{split}_return_per_turnover_proxy_bps"
        ]
        row[f"{split}_liquidation_count"] = 0
        row[f"{split}_account_wipeout_count"] = 0
    for key in (
        "train_return_stress_15bps_proxy",
        "validation_return_stress_15bps_proxy",
        "train_return_stress_20bps_proxy",
        "validation_return_stress_20bps_proxy",
        "low_sample_notional_share",
        "unguarded_low_sample_notional_share",
        "relaxed_low_sample_notional_share",
        "validation_spike_notional_share",
        "relaxed_validation_spike_notional_share",
        "low_efficiency_notional_share",
        "relaxed_notional_share",
    ):
        row[key] = metrics.get(key)
    row["train_validation_score"] = objective_value_from_row(row)
    row["efficiency_repair_score"] = objective_value_from_row(row)
    row["selection_reasons"] = _selection_reasons(
        row, max_gross=float(spec["max_gross_notional"]), spec=spec
    )
    row["diagnostic_warnings"] = _diagnostic_warnings(row)
    row["paper_testnet_candidate"] = not row["selection_reasons"]
    row["ready_for_paper"] = row["paper_testnet_candidate"]
    return stream, row, selected_rows


def optimize_relaxed_blend(
    profile_streams: Sequence[grid_hybrid.ProfileStream], *, n_trials: int, seed: int
) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is required for relaxed efficiency blend tuning")
    labels = [stream.profile_id for stream in profile_streams]
    index = profile_streams[0].returns.index

    def row_for_weights(weights: Mapping[str, float]) -> dict[str, Any]:
        returns = sum(
            stream.returns.reindex(index, fill_value=0.0) * float(weights[stream.profile_id])
            for stream in profile_streams
        )
        turnover = {
            split: float(
                sum(
                    stream.turnover_by_split[split] * float(weights[stream.profile_id])
                    for stream in profile_streams
                )
            )
            for split in grid_hybrid.ilp.SPLIT_ORDER
        }
        events = {
            split: int(
                sum(
                    stream.trade_events_by_split[split]
                    for stream in profile_streams
                    if float(weights[stream.profile_id]) > 1e-6
                )
            )
            for split in grid_hybrid.ilp.SPLIT_ORDER
        }
        gross = float(
            sum(
                stream.gross_notional_fraction * float(weights[stream.profile_id])
                for stream in profile_streams
            )
        )
        row = grid_hybrid._metric_row_from_stream(
            profile_id="hybrid_relaxed_efficiency_repair_live_guarded_three_profile_blend",
            profile_kind="optuna_static_mdd_guarded_relaxed_profile_blend",
            candidate_tier="paper_testnet_relaxed_efficiency_repair_candidate",
            leverage_map={},
            weights=dict(weights),
            gross_notional_fraction=gross,
            returns=returns,
            turnover_by_split=turnover,
            trade_events_by_split=events,
            liquidation_count_by_split=dict.fromkeys(grid_hybrid.ilp.SPLIT_ORDER, 0),
            strict_promotion_profile=False,
            promotion_gate_pass=False,
            paper_testnet_candidate=False,
        )
        row.update(strict._stress_metrics(row, turnover))
        row.update(
            {
                "optimizer": "optuna_tpe_static_mdd_guarded_relaxed_efficiency",
                "hybrid_version": "static_relaxed_efficiency_guarded",
                "ready_for_real": False,
                "real_money_execution": False,
                "real_execution_allowed": False,
                "final_weights": dict(weights),
                "average_weights_train_validation": dict(weights),
                "relaxed_policy_id": RELAXED_POLICY_ID,
                "low_sample_notional_share": 0.0,
                "unguarded_low_sample_notional_share": 0.0,
                "relaxed_low_sample_notional_share": 0.0,
                "validation_spike_notional_share": 0.0,
                "relaxed_validation_spike_notional_share": 0.0,
                "low_efficiency_notional_share": 0.0,
                "relaxed_notional_share": 0.0,
            }
        )
        row["selection_reasons"] = _selection_reasons(row, max_gross=8.0)
        row["diagnostic_warnings"] = _diagnostic_warnings(row)
        row["paper_testnet_candidate"] = not row["selection_reasons"]
        row["ready_for_paper"] = row["paper_testnet_candidate"]
        return row

    def normalized_from_trial(trial: Any) -> dict[str, float]:
        raw = np.array([trial.suggest_float(f"w_{idx}", 0.0, 1.0) for idx in range(len(labels))])
        raw = np.where(raw < 0.03, 0.0, raw)
        if float(raw.sum()) <= 0.0:
            raw[:] = 1.0
        raw = raw / float(raw.sum())
        return {label: float(raw[idx]) for idx, label in enumerate(labels)}

    def objective(trial: Any) -> float:
        row = row_for_weights(normalized_from_trial(trial))
        train = _safe_float(row.get("train_return"))
        validation = _safe_float(row.get("validation_return"))
        train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
        val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
        val_mdd = _safe_float(row.get("validation_mdd"))
        penalty = 0.0
        if not (_material_positive(row) and _mdd_guard_pass(row)):
            penalty += 35.0 * max(0.0, validation - train)
        penalty += (
            max(0.0, _safe_float(row.get("gross_notional_fraction")) - RELAXED_HARD_GROSS_CAP) * 6.0
        )
        penalty += max(0.0, 30.0 - train_rpt) / 5.0
        penalty += max(0.0, 30.0 - val_rpt) / 4.0
        penalty += max(0.0, val_mdd - RELAXED_HYBRID_MAX_VALIDATION_MDD) * 35.0
        for split in broad69.SPLIT_ORDER:
            if _safe_float(row.get(f"{split}_return_stress_20bps_proxy")) <= 0.0:
                penalty += 12.0
        return float(objective_value_from_row(row) - penalty)

    enqueue = []
    for idx in range(len(labels)):
        enqueue.append({f"w_{j}": 1.0 if j == idx else 0.0 for j in range(len(labels))})
    enqueue.append({f"w_{j}": 1.0 for j in range(len(labels))})
    study = run_optuna_study(
        optuna_module=optuna,
        objective=objective,
        n_trials=n_trials,
        direction="maximize",
        seed=seed,
        enqueue_trials=enqueue,
        n_jobs=1,
        show_progress_bar=False,
    )
    best_raw = np.array(
        [float(study.best_params.get(f"w_{idx}", 0.0)) for idx in range(len(labels))]
    )
    best_raw = np.where(best_raw < 0.03, 0.0, best_raw)
    if float(best_raw.sum()) <= 0.0:
        best_raw[:] = 1.0
    best_raw /= float(best_raw.sum())
    row = row_for_weights({label: float(best_raw[idx]) for idx, label in enumerate(labels)})
    row["best_value"] = float(study.best_value)
    row["best_params"] = dict(study.best_params)
    row["top_trials"] = [
        {
            "trial_number": int(trial.number),
            "value": None if trial.value is None else float(trial.value),
            "params": dict(trial.params),
        }
        for trial in sorted(
            study.trials,
            key=lambda trial: float(trial.value) if trial.value is not None else -1e18,
            reverse=True,
        )[:20]
    ]
    return row


def _select_legal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    legal = [dict(row) for row in rows if not row.get("selection_reasons")]
    pool = legal or [dict(row) for row in rows]
    return max(pool, key=objective_value_from_row)


def _strict_reference_payload() -> dict[str, Any]:
    path = strict.DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_efficiency_repair_optuna_latest.json"
    if not path.exists():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text("utf-8"))
    selected = dict(payload.get("selected_train_validation_legal_portfolio") or {})
    selected_optuna = dict(payload.get("selected_optuna_hybrid_profile") or {})
    return {
        "available": True,
        "path": str(path),
        "selected_train_validation_legal_portfolio": {
            key: selected.get(key)
            for key in (
                "profile_id",
                "train_return",
                "validation_return",
                "train_mdd",
                "validation_mdd",
                "train_return_per_turnover_proxy_bps",
                "validation_return_per_turnover_proxy_bps",
                "gross_notional_fraction",
                "selection_reasons",
                "ready_for_paper",
            )
        },
        "selected_optuna_hybrid_profile": {
            key: selected_optuna.get(key)
            for key in (
                "profile_id",
                "hybrid_version",
                "train_return",
                "validation_return",
                "train_mdd",
                "validation_mdd",
                "train_return_per_turnover_proxy_bps",
                "validation_return_per_turnover_proxy_bps",
                "gross_notional_fraction",
                "selection_reasons",
                "ready_for_paper",
            )
        },
        "strict_gate_ok_candidate_rows": sum(
            1
            for row in payload.get("asset_repair_rows", [])
            if not row.get("efficiency_repair_reasons")
        ),
        "strict_selected_symbols": sorted(
            {str(row.get("symbol")) for row in payload.get("selected_sleeve_rows", [])}
        ),
    }


def _render_pct(value: Any) -> str:
    return f"{_safe_float(value):.4%}"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    selected = dict(payload.get("selected_train_validation_legal_portfolio") or {})
    selected_hybrid = dict(payload.get("selected_optuna_hybrid_profile") or {})
    strict_ref = dict(payload.get("strict_reference") or {})
    relaxed_counts = dict(payload.get("relaxed_candidate_summary") or {})
    lines = [
        "# 69-asset relaxed live-efficiency repair Optuna",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "## Policy",
        "",
        f"- policy id: `{payload.get('relaxed_policy_id')}`",
        "- train < validation is allowed only when both train and validation returns are >= 2% and MDD guard passes.",
        "- TradFi/material-positive low-sample rows are allowed as warnings under the same MDD and strict 10bps RPT guards.",
        "- Gross/concentration pressure is optimized/penalized; it is not a hard rejection while MDD stays below the relaxed guard.",
        "- 10bps return-per-turnover gate remains strict on train and validation.",
        "- No test set or locked-OOS is used for selection; real-money execution remains disabled.",
        "",
        "## Candidate expansion versus strict pass",
        "",
        f"- strict gate-ok rows: `{relaxed_counts.get('strict_gate_ok_rows')}`",
        f"- relaxed gate-ok rows: `{relaxed_counts.get('relaxed_gate_ok_rows')}`",
        f"- newly admitted rows: `{relaxed_counts.get('newly_admitted_rows')}`",
        f"- relaxed unique gate-ok symbols: `{relaxed_counts.get('relaxed_gate_ok_symbol_count')}`",
        f"- newly admitted symbols: `{', '.join(relaxed_counts.get('newly_admitted_symbols') or [])}`",
        "",
        "## Selected relaxed portfolio",
        "",
        f"- profile: `{selected.get('profile_id')}`",
        f"- train / validation: `{_render_pct(selected.get('train_return'))}` / `{_render_pct(selected.get('validation_return'))}`",
        f"- train / validation MDD: `{_render_pct(selected.get('train_mdd'))}` / `{_render_pct(selected.get('validation_mdd'))}`",
        f"- RPT bps train / validation: `{_safe_float(selected.get('train_return_per_turnover_proxy_bps')):.2f}` / `{_safe_float(selected.get('validation_return_per_turnover_proxy_bps')):.2f}`",
        f"- 20bps stress train / validation proxy: `{_render_pct(selected.get('train_return_stress_20bps_proxy'))}` / `{_render_pct(selected.get('validation_return_stress_20bps_proxy'))}`",
        f"- gross notional: `{_safe_float(selected.get('gross_notional_fraction')):.4f}x`",
        f"- final weights: `{json.dumps(_json_safe(selected.get('final_weights') or selected.get('weights') or {}), sort_keys=True)}`",
        f"- selection reasons: `{selected.get('selection_reasons')}`",
        "",
        "## Selected relaxed Optuna hybrid",
        "",
        f"- profile: `{selected_hybrid.get('profile_id')}`",
        f"- train / validation: `{_render_pct(selected_hybrid.get('train_return'))}` / `{_render_pct(selected_hybrid.get('validation_return'))}`",
        f"- train / validation MDD: `{_render_pct(selected_hybrid.get('train_mdd'))}` / `{_render_pct(selected_hybrid.get('validation_mdd'))}`",
        f"- RPT bps train / validation: `{_safe_float(selected_hybrid.get('train_return_per_turnover_proxy_bps')):.2f}` / `{_safe_float(selected_hybrid.get('validation_return_per_turnover_proxy_bps')):.2f}`",
        f"- gross notional: `{_safe_float(selected_hybrid.get('gross_notional_fraction')):.4f}x`",
        f"- final weights: `{json.dumps(_json_safe(selected_hybrid.get('final_weights') or {}), sort_keys=True)}`",
        f"- selection reasons: `{selected_hybrid.get('selection_reasons')}`",
        "",
        "## Repaired relaxed profiles",
        "",
        "| Profile | Sleeves | Gross | Train | Validation | Val MDD | RPT T/V bps | Relaxed share | Paper |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("profile_rows", []):
        lines.append(
            f"| `{row.get('profile_id')}` | {int(row.get('selected_sleeve_count') or 0)} | "
            f"{_safe_float(row.get('gross_notional_fraction')):.2f}x | "
            f"{_render_pct(row.get('train_return'))} | {_render_pct(row.get('validation_return'))} | "
            f"{_render_pct(row.get('validation_mdd'))} | "
            f"{_safe_float(row.get('train_return_per_turnover_proxy_bps')):.2f}/"
            f"{_safe_float(row.get('validation_return_per_turnover_proxy_bps')):.2f} | "
            f"{_safe_float(row.get('relaxed_notional_share')):.2%} | "
            f"{str(bool(row.get('ready_for_paper'))).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Strict reference",
            "",
            f"- available: `{strict_ref.get('available')}`",
            f"- path: `{strict_ref.get('path')}`",
            f"- strict selected train/validation: `{_render_pct(dict(strict_ref.get('selected_train_validation_legal_portfolio') or {}).get('train_return'))}` / `{_render_pct(dict(strict_ref.get('selected_train_validation_legal_portfolio') or {}).get('validation_return'))}`",
            f"- strict selected MDD train/validation: `{_render_pct(dict(strict_ref.get('selected_train_validation_legal_portfolio') or {}).get('train_mdd'))}` / `{_render_pct(dict(strict_ref.get('selected_train_validation_legal_portfolio') or {}).get('validation_mdd'))}`",
            "",
            "## Governance",
            "",
            f"- primary round-trip cost bps: `{payload.get('research_primary_round_trip_cost_bps')}`",
            f"- return-per-turnover threshold bps: `{payload.get('return_per_turnover_threshold_bps')}`",
            f"- ready_for_real: `{str(bool(payload.get('ready_for_real'))).lower()}`",
            f"- real_money_execution: `{str(bool(payload.get('real_money_execution'))).lower()}`",
            f"- runner peak RSS MiB: `{_safe_float(payload.get('runner_peak_rss_mib')):.2f}`",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate_summary(asset_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strict_ok = [
        row for row in asset_rows if not strict._candidate_repair_reasons(row, row["profile_spec"])
    ]
    relaxed_ok = [row for row in asset_rows if not row.get("efficiency_repair_reasons")]
    strict_ids = {str(row.get("model_id")) for row in strict_ok}
    relaxed_symbols = sorted({str(row.get("symbol")) for row in relaxed_ok})
    newly = [row for row in relaxed_ok if str(row.get("model_id")) not in strict_ids]
    return {
        "strict_gate_ok_rows": len(strict_ok),
        "relaxed_gate_ok_rows": len(relaxed_ok),
        "newly_admitted_rows": len(newly),
        "relaxed_gate_ok_symbol_count": len(relaxed_symbols),
        "relaxed_gate_ok_symbols": relaxed_symbols,
        "newly_admitted_symbols": sorted({str(row.get("symbol")) for row in newly}),
        "newly_admitted_rows_preview": sorted(
            [
                {
                    "symbol": row.get("symbol"),
                    "profile_id": row.get("profile_id"),
                    "timeframe": row.get("timeframe"),
                    "integer_leverage": row.get("integer_leverage"),
                    "train_return": row.get("train_return"),
                    "validation_return": row.get("validation_return"),
                    "train_mdd": row.get("train_mdd"),
                    "validation_mdd": row.get("validation_mdd"),
                    "train_rpt_bps": row.get("train_return_per_turnover_proxy_bps"),
                    "validation_rpt_bps": row.get("validation_return_per_turnover_proxy_bps"),
                    "train_trade_event_count": row.get("train_trade_event_count"),
                    "validation_trade_event_count": row.get("validation_trade_event_count"),
                    "relaxations_applied": row.get("relaxations_applied"),
                }
                for row in newly
            ],
            key=lambda item: _safe_float(item.get("validation_return")),
            reverse=True,
        )[:50],
    }


def _apply_relaxed_hybrid_row_fields(row: dict[str, Any]) -> dict[str, Any]:
    row.update(
        {
            "relaxed_policy_id": RELAXED_POLICY_ID,
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "low_sample_notional_share": row.get("low_sample_notional_share", 0.0),
            "unguarded_low_sample_notional_share": row.get(
                "unguarded_low_sample_notional_share", 0.0
            ),
            "relaxed_low_sample_notional_share": row.get("relaxed_low_sample_notional_share", 0.0),
            "validation_spike_notional_share": row.get("validation_spike_notional_share", 0.0),
            "relaxed_validation_spike_notional_share": row.get(
                "relaxed_validation_spike_notional_share", 0.0
            ),
            "low_efficiency_notional_share": row.get("low_efficiency_notional_share", 0.0),
            "relaxed_notional_share": row.get("relaxed_notional_share", 0.0),
        }
    )
    row["selection_reasons"] = _selection_reasons(row, max_gross=8.0)
    row["diagnostic_warnings"] = _diagnostic_warnings(row)
    row["ready_for_paper"] = not row["selection_reasons"]
    row["paper_testnet_candidate"] = row["ready_for_paper"]
    return row


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_artifact).expanduser().resolve()
    source_payload = strict._load_source_payload(source_path)
    symbols = tuple(source_payload["universe"]["symbols"])
    timeframes = tuple(source_payload["timeframes"])
    data_root = Path(source_payload["data_coverage"]["data_root"])
    bars, coverage = broad69.load_all_bars(symbols, data_root=data_root, timeframes=timeframes)
    windows = strict._split_windows_from_payload(source_payload)
    train_eligibility = broad69.build_train_eligibility_report(
        bars,
        symbols=symbols,
        timeframes=timeframes,
        windows=windows,
    )
    cache = profile69.FeatureCache(
        bars_by_symbol_tf=bars,
        symbols=symbols,
        timeframes=timeframes,
        _xsmom={},
        _anchor_returns={},
    )

    asset_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    profile_streams: list[grid_hybrid.ProfileStream] = []
    for idx, spec in enumerate(RELAXED_PROFILE_SPECS):
        candidates = _build_candidate_streams(
            source_payload=source_payload,
            spec=spec,
            cache=cache,
            windows=windows,
            train_eligibility=train_eligibility,
        )
        for stream in candidates:
            row = dict(stream.row)
            row["profile_spec"] = dict(spec)
            asset_rows.append(row)
        profile_stream, profile_row, selected = tune_relaxed_profile_allocations(
            spec=spec,
            candidate_streams=candidates,
            windows=windows,
            n_trials=int(args.profile_trials),
            seed=int(args.seed) + idx * 10_000,
        )
        profile_streams.append(profile_stream)
        profile_rows.append(profile_row)
        sleeve_rows.extend(selected)

    split_windows = strict._split_window_payload_for_hybrid(windows)
    with (
        optuna_hybrid._split_window_context(split_windows),
        _relaxed_optuna_hybrid_objective_context(),
    ):
        v35 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_5",
            n_trials=int(args.hybrid_trials),
            seed=int(args.seed) + 700_000,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        v36 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_6",
            n_trials=int(args.hybrid_trials),
            seed=int(args.seed) + 700_001,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        static_guarded = optimize_relaxed_blend(
            profile_streams, n_trials=int(args.hybrid_trials), seed=int(args.seed) + 710_000
        )
        corr_tv = grid_hybrid._profile_corr_matrix(profile_streams, split="train_validation")
        corr_val = grid_hybrid._profile_corr_matrix(profile_streams, split="validation")

    hybrid_rows = [dict(static_guarded), dict(v35.row), dict(v36.row)]
    for row in hybrid_rows[1:]:
        stress = strict._stress_metrics(
            row,
            {
                split: sum(
                    stream.turnover_by_split[split]
                    * _safe_float(row.get("weights", {}).get(stream.profile_id))
                    for stream in profile_streams
                )
                for split in broad69.SPLIT_ORDER
            },
        )
        row.update(stress)
        _apply_relaxed_hybrid_row_fields(row)
    selected_legal = _select_legal([*profile_rows, *hybrid_rows])
    selected_optuna_result = max(
        [v35, v36], key=lambda result: _relaxed_hybrid_objective_score(result.row)
    )
    selected_optuna_row = dict(selected_optuna_result.row)
    stress = strict._stress_metrics(
        selected_optuna_row,
        {
            split: sum(
                stream.turnover_by_split[split]
                * _safe_float(selected_optuna_row.get("weights", {}).get(stream.profile_id))
                for stream in profile_streams
            )
            for split in broad69.SPLIT_ORDER
        },
    )
    selected_optuna_row.update(stress)
    _apply_relaxed_hybrid_row_fields(selected_optuna_row)

    output_dir = Path(args.output_dir).expanduser().resolve()
    latest_json = output_dir / "alpha_zoo_69_asset_relaxed_efficiency_repair_optuna_latest.json"
    timestamped_json = (
        output_dir / f"alpha_zoo_69_asset_relaxed_efficiency_repair_optuna_{_timestamp()}.json"
    )
    latest_md = output_dir / "alpha_zoo_69_asset_relaxed_efficiency_repair_optuna_latest.md"
    assets_csv = output_dir / "alpha_zoo_69_asset_relaxed_efficiency_repair_candidates_latest.csv"
    sleeves_csv = output_dir / "alpha_zoo_69_asset_relaxed_efficiency_repair_sleeves_latest.csv"
    profiles_csv = output_dir / "alpha_zoo_69_asset_relaxed_efficiency_repair_profiles_latest.csv"
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_69_asset_relaxed_efficiency_repair_optuna",
        "generated_at_utc": _utc_now_iso(),
        "source_artifact": str(source_path),
        "strict_reference": _strict_reference_payload(),
        "relaxed_policy_id": RELAXED_POLICY_ID,
        "universe": {"symbol_count": len(symbols), "symbols": list(symbols)},
        "timeframes": list(timeframes),
        "split_policy": {
            **windows.as_payload(),
            "locked_oos": {
                "enabled": False,
                "start": None,
                "end": None,
                "role": "disabled_for_live_relaxed_efficiency_repair_no_test_set_reserved",
            },
        },
        "data_coverage": coverage,
        "train_eligibility": train_eligibility,
        "research_primary_round_trip_cost_bps": PRIMARY_COST_BPS,
        "avg_bbo_spread_bps_assumption": broad69.AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "return_per_turnover_threshold_bps": broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "paper_testnet_only": True,
        "ready_for_paper": bool(selected_legal.get("ready_for_paper")),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "optimization_policy": optimization_search_policy_payload(
            search_method="optuna_tpe_mdd_guarded_relaxed_efficiency_repair_from_per_asset_profile_artifact",
            objective_policy="train_validation_relaxed_efficiency_no_locked_oos_10bps_strict",
            selection_inputs=("train", "validation"),
            extra={
                "source_artifact": str(source_path),
                "profile_trials": int(args.profile_trials),
                "hybrid_trials_per_version": int(args.hybrid_trials),
                "cost_stress_bps": list(STRESS_COST_BPS),
                "policy_id": RELAXED_POLICY_ID,
                "dominance_relaxation": "train_below_validation_allowed_if_train_and_validation_ge_2pct_and_mdd_guard_passes",
                "tradfi_low_sample_relaxation": "train/validation event minima allowed as warnings for tradfi if mdd_guard_and_10bps_rpt_pass",
                "material_positive_low_sample_relaxation": "non_tradfi_low_sample_allowed_as_warning_when_both_returns_ge_2pct_and_mdd_guard_passes",
                "gross_concentration_policy": "penalty_not_hard_reject_while_validation_mdd_under_guard_and_gross_under_12x_hard_cap",
                "rpt_10bps_gate_unchanged": True,
                "train_eligible_symbol_count": train_eligibility["train_eligible_symbol_count"],
                "train_ineligible_symbol_count": train_eligibility["train_ineligible_symbol_count"],
                "train_ineligible_symbols": train_eligibility["train_ineligible_symbols"],
                "all_source_asset_profile_params_preserved": True,
                "repairs_portfolio_weights_not_asset_signal_params": True,
                "hybrid_fit_inputs": ["train"],
                "hybrid_score_inputs": ["train", "validation"],
                "hybrid_warmup_inputs": ["train"],
                "warmup_ratio_scope": "train_split_only",
                "uses_test_set": False,
                "profile_specs": list(RELAXED_PROFILE_SPECS),
            },
        ),
        "relaxed_candidate_summary": _candidate_summary(asset_rows),
        "asset_repair_rows": asset_rows,
        "profile_rows": profile_rows,
        "selected_sleeve_rows": sleeve_rows,
        "static_relaxed_efficiency_guarded_hybrid": static_guarded,
        "selected_optuna_hybrid_profile": selected_optuna_row,
        "selected_train_validation_legal_portfolio": selected_legal,
        "hybrid_v3_5_optuna": {
            "row": hybrid_rows[1],
            "optuna": v35.optuna,
            "top_trials": list(v35.top_trials),
        },
        "hybrid_v3_6_optuna": {
            "row": hybrid_rows[2],
            "optuna": v36.optuna,
            "top_trials": list(v36.top_trials),
        },
        "profile_train_validation_corr_matrix": corr_tv,
        "profile_validation_corr_matrix": corr_val,
        "runner_peak_rss_mib": peak_mib,
        "memory_summary": {"limit_mib": 8192.0, "pass_under_8gb": peak_mib < 8192.0},
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_md": str(latest_md),
            "asset_repair_csv": str(assets_csv),
            "selected_sleeves_csv": str(sleeves_csv),
            "profile_rows_csv": str(profiles_csv),
        },
    }
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.write_text(_render_markdown(payload), encoding="utf-8")
    _write_csv(assets_csv, asset_rows, CANDIDATE_FIELDS)
    _write_csv(sleeves_csv, sleeve_rows, SLEEVE_FIELDS)
    _write_csv(profiles_csv, profile_rows, PROFILE_FIELDS)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--profile-trials", type=int, default=DEFAULT_PROFILE_TRIALS)
    parser.add_argument("--hybrid-trials", type=int, default=DEFAULT_HYBRID_TRIALS)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    selected = dict(payload["selected_train_validation_legal_portfolio"])
    selected_hybrid = dict(payload["selected_optuna_hybrid_profile"])
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "relaxed_policy_id": payload["relaxed_policy_id"],
                    "relaxed_candidate_summary": payload["relaxed_candidate_summary"],
                    "selected_train_validation_legal_portfolio": {
                        key: selected.get(key)
                        for key in (
                            "profile_id",
                            "hybrid_version",
                            "train_return",
                            "validation_return",
                            "train_mdd",
                            "validation_mdd",
                            "train_return_per_turnover_proxy_bps",
                            "validation_return_per_turnover_proxy_bps",
                            "train_return_stress_20bps_proxy",
                            "validation_return_stress_20bps_proxy",
                            "gross_notional_fraction",
                            "final_weights",
                            "weights",
                            "selection_reasons",
                            "ready_for_paper",
                        )
                    },
                    "selected_optuna_hybrid_profile": {
                        key: selected_hybrid.get(key)
                        for key in (
                            "profile_id",
                            "hybrid_version",
                            "train_return",
                            "validation_return",
                            "train_mdd",
                            "validation_mdd",
                            "train_return_per_turnover_proxy_bps",
                            "validation_return_per_turnover_proxy_bps",
                            "gross_notional_fraction",
                            "selection_reasons",
                            "ready_for_paper",
                        )
                    },
                    "ready_for_paper": payload["ready_for_paper"],
                    "ready_for_real": payload["ready_for_real"],
                    "real_money_execution": payload["real_money_execution"],
                    "runner_peak_rss_mib": payload["runner_peak_rss_mib"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
