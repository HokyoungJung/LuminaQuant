"""Outcome-based edge calibration with shrinkage and fail-closed gates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    action: str
    reason: str
    lower_confidence_edge_bps: float
    tail_loss_bps: float

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


@dataclass(frozen=True, slots=True)
class BucketCalibration:
    bucket: tuple[str, ...]
    n: int
    parent_n: int
    mean_edge_bps: float
    shrinkage_mean_edge_bps: float
    lower_confidence_edge_bps: float
    tail_loss_bps: float
    decision: CalibrationDecision

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = asdict(self.decision)
        return payload


def _bucket_key(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(record.get(field, "")) for field in fields)


def _edge(record: Mapping[str, Any]) -> float | None:
    for key in ("net_pnl_bps", "realized_net_pnl_bps", "edge_bps"):
        if key in record:
            try:
                value = float(record[key])
            except Exception:
                continue
            if math.isfinite(value):
                return value
    return None


def _summary(values: Sequence[float]) -> tuple[int, float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0, 0.0, 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    tail = float(np.quantile(arr, 0.05)) if arr.size >= 2 else float(arr.min())
    return int(arr.size), mean, std, tail


def calibrate_edge_buckets(
    records: Sequence[Mapping[str, Any]],
    *,
    bucket_fields: Sequence[str],
    parent_fields: Sequence[str] = (),
    min_bucket_n: int = 30,
    confidence_z: float = 1.64,
    min_lower_edge_bps: float = 0.0,
    max_tail_loss_bps: float = 250.0,
) -> dict[tuple[str, ...], BucketCalibration]:
    """Estimate bucket edge and return block/allow decisions.

    Small buckets shrink toward their parent bucket (or global sample when no
    parent fields are supplied).  Promotion is blocked unless the lower
    confidence edge is positive and left-tail loss is within budget.
    """
    if not bucket_fields:
        raise ValueError("bucket_fields must not be empty")
    bucket_values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    parent_values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    global_values: list[float] = []
    bucket_to_parent: dict[tuple[str, ...], tuple[str, ...]] = {}
    for record in records:
        edge = _edge(record)
        if edge is None:
            continue
        bucket = _bucket_key(record, bucket_fields)
        parent = _bucket_key(record, parent_fields) if parent_fields else ("__global__",)
        bucket_values[bucket].append(edge)
        parent_values[parent].append(edge)
        global_values.append(edge)
        bucket_to_parent[bucket] = parent

    _, global_mean, global_std, global_tail = _summary(global_values)
    out: dict[tuple[str, ...], BucketCalibration] = {}
    min_n = max(1, int(min_bucket_n))
    z = max(0.0, float(confidence_z))
    for bucket, values in bucket_values.items():
        n, mean, std, tail = _summary(values)
        parent_key = bucket_to_parent.get(bucket, ("__global__",))
        parent_n, parent_mean, parent_std, parent_tail = _summary(parent_values.get(parent_key, global_values))
        if parent_n == 0:
            parent_mean = global_mean
            parent_std = global_std
            parent_tail = global_tail
        weight = min(1.0, n / float(min_n))
        shrink_mean = (weight * mean) + ((1.0 - weight) * parent_mean)
        effective_std = std if n >= 2 else parent_std
        stderr = effective_std / math.sqrt(max(1, n))
        lower = shrink_mean - (z * stderr)
        effective_tail = tail if n >= min_n else min(tail, parent_tail)
        tail_loss = max(0.0, -effective_tail)
        if lower <= float(min_lower_edge_bps):
            decision = CalibrationDecision("block", "lower_confidence_edge_not_positive", lower, tail_loss)
        elif tail_loss > float(max_tail_loss_bps):
            decision = CalibrationDecision("downsize", "tail_loss_exceeds_budget", lower, tail_loss)
        else:
            decision = CalibrationDecision("allow", "edge_positive_after_shrinkage", lower, tail_loss)
        out[bucket] = BucketCalibration(
            bucket=bucket,
            n=n,
            parent_n=parent_n,
            mean_edge_bps=mean,
            shrinkage_mean_edge_bps=shrink_mean,
            lower_confidence_edge_bps=lower,
            tail_loss_bps=tail_loss,
            decision=decision,
        )
    return out
