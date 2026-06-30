"""Split-aware alpha evidence and benchmark classification primitives.

The module is intentionally framework-native: it works with existing Alpha Zoo
``FactorSpec`` objects and pandas factor frames, but it does not import or copy
external registry/backtest code.  Selection evidence is computed only from the
configured research splits; locked OOS data is reported separately and never used
for ranking or classification.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

_SELECTION_SPLITS = ("train", "validation")
_LOCKED_OOS_SPLITS = ("locked_oos", "oos")


@dataclass(frozen=True, slots=True)
class AlphaDescriptor:
    """Durable metadata for a candidate alpha/factor."""

    name: str
    family: str
    market: str
    description: str
    inputs: tuple[str, ...]
    parameters: Mapping[str, Any]
    source_refs: tuple[str, ...]
    selection_policy: str = "train_validation_only_locked_oos_report_only"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_kind"] = "alpha_descriptor"
        payload["descriptor_hash"] = stable_alpha_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class AlphaEvidenceThresholds:
    """Fail-closed thresholds for IC/IR-style alpha classification."""

    min_periods: int = 8
    min_abs_ic_mean: float = 0.02
    min_positive_ratio: float = 0.55
    min_abs_t_stat: float = 2.0


@dataclass(frozen=True, slots=True)
class SplitEvidence:
    """Summary statistics for one split's per-period rank IC series."""

    n_periods: int
    n_observations: int
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_positive_ratio: float
    t_stat: float
    quantile_spread_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return str(value)


def stable_alpha_hash(payload: Mapping[str, Any]) -> str:
    """Return a deterministic short hash for alpha metadata/evidence payloads."""
    normalized = dict(payload)
    normalized.pop("created_at", None)
    normalized.pop("generated_at", None)
    normalized.pop("descriptor_hash", None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def build_alpha_descriptor(
    spec: Any,
    *,
    source_refs: Sequence[str],
    created_at: str | None = None,
    selection_policy: str = "train_validation_only_locked_oos_report_only",
) -> AlphaDescriptor:
    """Build an ``AlphaDescriptor`` from a native Alpha Zoo ``FactorSpec``-like object."""
    refs = tuple(str(ref) for ref in source_refs if str(ref).strip())
    if not refs:
        raise ValueError("alpha_descriptor_source_refs_missing")
    return AlphaDescriptor(
        name=str(getattr(spec, "name", "") or ""),
        family=str(getattr(spec, "family", "") or ""),
        market=str(getattr(spec, "market", "") or ""),
        description=str(getattr(spec, "description", "") or ""),
        inputs=tuple(str(item) for item in tuple(getattr(spec, "inputs", ()) or ())),
        parameters=dict(getattr(spec, "parameters", None) or {}),
        source_refs=refs,
        selection_policy=str(selection_policy),
        created_at=created_at or datetime.now(UTC).isoformat(),
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"alpha_evidence_missing_columns:{','.join(missing)}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except TypeError, ValueError:
        return default
    return out if math.isfinite(out) else default


def _spearman_rank_corr(left: pd.Series, right: pd.Series) -> float | None:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3:
        return None
    lhs = joined.iloc[:, 0].rank(method="average")
    rhs = joined.iloc[:, 1].rank(method="average")
    if lhs.nunique(dropna=True) < 2 or rhs.nunique(dropna=True) < 2:
        return None
    corr = float(lhs.corr(rhs))
    return corr if math.isfinite(corr) else None


def _period_quantile_spread(
    values: pd.Series,
    labels: pd.Series,
    *,
    top_quantile: float = 0.8,
    bottom_quantile: float = 0.2,
) -> float | None:
    joined = pd.concat([values, labels], axis=1).dropna()
    if len(joined) < 5:
        return None
    ranks = joined.iloc[:, 0].rank(method="average", pct=True)
    top = joined.loc[ranks >= top_quantile].iloc[:, 1]
    bottom = joined.loc[ranks <= bottom_quantile].iloc[:, 1]
    if top.empty or bottom.empty:
        return None
    spread = float(top.mean() - bottom.mean())
    return spread if math.isfinite(spread) else None


def rank_ic_series(
    frame: pd.DataFrame,
    *,
    factor: str,
    label_column: str = "forward_return",
    time_column: str = "timestamp",
    min_cross_section: int = 3,
) -> pd.DataFrame:
    """Compute causal cross-sectional Spearman rank IC per timestamp."""
    _require_columns(frame, (time_column, factor, label_column))
    rows: list[dict[str, Any]] = []
    for timestamp, part in frame.groupby(time_column, sort=True, dropna=True):
        clean = part[[factor, label_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < max(3, int(min_cross_section)):
            continue
        ic = _spearman_rank_corr(clean[factor], clean[label_column])
        if ic is None:
            continue
        spread = _period_quantile_spread(clean[factor], clean[label_column])
        rows.append(
            {
                time_column: timestamp,
                "rank_ic": ic,
                "quantile_spread": spread,
                "n": len(clean),
            }
        )
    return pd.DataFrame(rows, columns=[time_column, "rank_ic", "quantile_spread", "n"])


def summarize_split_evidence(ic_frame: pd.DataFrame) -> SplitEvidence:
    """Summarize a rank-IC frame into JSON-safe evidence statistics."""
    if ic_frame.empty or "rank_ic" not in ic_frame.columns:
        return SplitEvidence(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = pd.to_numeric(ic_frame["rank_ic"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return SplitEvidence(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n_obs = int(
        pd.to_numeric(ic_frame.get("n", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    )
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0
    if std <= 1e-12:
        if abs(mean) <= 1e-12:
            ir = 0.0
            t_stat = 0.0
        else:
            ir = math.copysign(1_000_000.0, mean)
            t_stat = math.copysign(1_000_000.0, mean)
    else:
        ir = float(mean / std)
        t_stat = float(mean / (std / math.sqrt(float(values.size))))
    spreads = pd.to_numeric(
        ic_frame.get("quantile_spread", pd.Series(dtype=float)), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    spread_mean = float(spreads.dropna().mean()) if not spreads.dropna().empty else 0.0
    return SplitEvidence(
        n_periods=int(values.size),
        n_observations=n_obs,
        ic_mean=_safe_float(mean),
        ic_std=_safe_float(std),
        ic_ir=_safe_float(ir),
        ic_positive_ratio=_safe_float(float(np.mean(values > 0.0))),
        t_stat=_safe_float(t_stat),
        quantile_spread_mean=_safe_float(spread_mean),
    )


def _classify(
    summary: SplitEvidence, thresholds: AlphaEvidenceThresholds
) -> tuple[str, bool, list[str]]:
    reasons: list[str] = []
    if summary.n_periods < int(thresholds.min_periods):
        reasons.append("insufficient_selection_periods")
        return "insufficient", False, reasons
    abs_t = abs(float(summary.t_stat))
    mean = float(summary.ic_mean)
    if not math.isfinite(mean) or not math.isfinite(abs_t):
        reasons.append("nonfinite_selection_evidence")
        return "insufficient", False, reasons
    if (
        mean >= float(thresholds.min_abs_ic_mean)
        and summary.ic_positive_ratio >= float(thresholds.min_positive_ratio)
        and abs_t >= float(thresholds.min_abs_t_stat)
    ):
        return "alive", True, reasons
    if mean <= -float(thresholds.min_abs_ic_mean) and abs_t >= float(thresholds.min_abs_t_stat):
        reasons.append("candidate_signal_direction_reversed")
        return "reversed", True, reasons
    reasons.append("weak_or_unstable_selection_evidence")
    return "dead", False, reasons


def alpha_benchmark_evidence(
    frame: pd.DataFrame,
    *,
    factor: str,
    label_column: str = "forward_return",
    time_column: str = "timestamp",
    split_column: str = "split",
    selection_splits: Sequence[str] = _SELECTION_SPLITS,
    locked_oos_splits: Sequence[str] = _LOCKED_OOS_SPLITS,
    thresholds: AlphaEvidenceThresholds | None = None,
    min_cross_section: int = 3,
) -> dict[str, Any]:
    """Classify an alpha using train/validation IC evidence and OOS report-only stats."""
    _require_columns(frame, (time_column, factor, label_column))
    limits = thresholds or AlphaEvidenceThresholds()
    split_names = tuple(str(item) for item in selection_splits if str(item).strip())
    locked_names = tuple(str(item) for item in locked_oos_splits if str(item).strip())
    locked_name_set = set(locked_names)
    forbidden_selection_splits = tuple(split for split in split_names if split in locked_name_set)
    effective_selection_splits = tuple(
        split for split in split_names if split not in locked_name_set
    )

    split_column_present = split_column in frame.columns
    if split_column_present:
        selected_mask = frame[split_column].astype(str).isin(effective_selection_splits)
        split_values = tuple(str(item) for item in frame[split_column].dropna().unique().tolist())
    else:
        selected_mask = pd.Series(False, index=frame.index)
        split_values = ()

    split_reports: dict[str, dict[str, Any]] = {}
    for split in dict.fromkeys((*split_names, *locked_names)):
        if split_column_present:
            part = frame[frame[split_column].astype(str).eq(split)]
        else:
            part = frame.iloc[0:0]
        split_reports[split] = summarize_split_evidence(
            rank_ic_series(
                part,
                factor=factor,
                label_column=label_column,
                time_column=time_column,
                min_cross_section=min_cross_section,
            )
        ).to_dict()

    selected_ic = rank_ic_series(
        frame[selected_mask],
        factor=factor,
        label_column=label_column,
        time_column=time_column,
        min_cross_section=min_cross_section,
    )
    selected_summary = summarize_split_evidence(selected_ic)
    classification, passed, reasons = _classify(selected_summary, limits)
    reasons.append("locked_oos_report_only_not_used_for_selection")
    if not split_column_present:
        classification = "insufficient"
        passed = False
        reasons.append("split_column_missing_fail_closed")
    if forbidden_selection_splits:
        classification = "insufficient"
        passed = False
        reasons.append(
            "locked_oos_selection_splits_forbidden:" + ",".join(sorted(forbidden_selection_splits))
        )
    signal_direction = "invert" if classification == "reversed" else "as_is"

    payload = {
        "artifact_kind": "alpha_benchmark_evidence",
        "factor": str(factor),
        "classification": classification,
        "pass": bool(passed),
        "signal_direction": signal_direction,
        "requires_inversion": classification == "reversed",
        "selection_score": abs(float(selected_summary.ic_mean))
        if classification == "reversed"
        else float(selected_summary.ic_mean),
        "selected_using_splits": list(effective_selection_splits),
        "requested_selection_splits": list(split_names),
        "uses_locked_oos_for_selection": bool(forbidden_selection_splits),
        "forbidden_selection_splits": list(forbidden_selection_splits),
        "locked_oos_role": "report_only_gate_not_ranked",
        "available_splits": list(split_values),
        "thresholds": asdict(limits),
        "selection_evidence": selected_summary.to_dict(),
        "split_evidence": split_reports,
        "rejection_reasons": reasons,
    }
    payload["evidence_hash"] = stable_alpha_hash(payload)
    return payload


def attach_evidence_to_screen_payload(
    screen_payload: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    label_column: str = "forward_return",
    time_column: str = "timestamp",
    split_column: str = "split",
) -> dict[str, Any]:
    """Attach benchmark evidence to an existing Alpha Zoo screen payload."""
    payload = dict(screen_payload)
    enriched: list[dict[str, Any]] = []
    for row in list(payload.get("selected_factors") or []):
        item = dict(row)
        factor = str(item.get("factor") or "")
        if factor:
            item["benchmark_evidence"] = alpha_benchmark_evidence(
                frame,
                factor=factor,
                label_column=label_column,
                time_column=time_column,
                split_column=split_column,
            )
        enriched.append(item)
    payload["selected_factors"] = enriched
    payload["evidence_policy"] = "train_validation_only_locked_oos_report_only"
    return payload


__all__ = [
    "AlphaDescriptor",
    "AlphaEvidenceThresholds",
    "SplitEvidence",
    "alpha_benchmark_evidence",
    "attach_evidence_to_screen_payload",
    "build_alpha_descriptor",
    "rank_ic_series",
    "stable_alpha_hash",
    "summarize_split_evidence",
]
