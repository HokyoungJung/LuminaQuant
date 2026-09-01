"""Read-only factor IC-heatmap and candidate-queue dashboard payload service.

Additive dashboard surface (Phase 6+).  This module never imports broker /
order-gateway code and never touches the flat backtest metric dict.  It renders
two read-only views on top of already-computed research artifacts:

* an **IC heatmap** — factors on the row axis, rank-decay lags on the column
  axis, cells populated from the deterministic
  :mod:`lumina_quant.research.factor_ic` batch result, plus per-factor IC mean
  and IC-IR side channels; and
* a **candidate queue** — the pending research-candidate review queue rendered
  from an optional JSON artifact.

Both inputs are optional; with no artifacts the payload is a well-formed empty
surface.  All ordering is deterministic (factors and lags sorted; candidates
ranked by a stable score then id), so the JSON output is byte-reproducible for a
given input.  The service is invoked in module mode by the Next.js bridge via
``runUvPythonModuleJson("lumina_quant.dashboard.factor_insights_service")``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCORE_KEYS: tuple[str, ...] = ("score", "sharpe", "robustness_score", "ic_ir")


def _read_json(path: str | Path | None) -> Any:
    if path is None or not str(path).strip():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except TypeError, ValueError:
        return None
    return out if math.isfinite(out) else None


def _coerce_factor_map(factor_ic: Any) -> dict[str, Mapping[str, Any]]:
    """Accept a BatchFactorICResult, its ``to_dict()`` output, or a mapping."""
    if factor_ic is None:
        return {}
    if hasattr(factor_ic, "to_dict") and not isinstance(factor_ic, Mapping):
        factor_ic = factor_ic.to_dict()
    if not isinstance(factor_ic, Mapping):
        return {}
    factors = factor_ic.get("factors")
    if not isinstance(factors, Mapping):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for name, stats in factors.items():
        if isinstance(stats, Mapping):
            out[str(name)] = stats
    return out


def _lag_count(factor_map: Mapping[str, Mapping[str, Any]], declared: Any) -> int:
    declared_int = 0
    if isinstance(declared, (int, float)) and math.isfinite(float(declared)):
        declared_int = max(0, int(declared))
    observed = 0
    for stats in factor_map.values():
        autocorr = stats.get("rank_autocorr")
        if isinstance(autocorr, Sequence) and not isinstance(autocorr, (str, bytes)):
            observed = max(observed, len(autocorr))
    return max(declared_int, observed)


def _build_ic_heatmap(
    factor_map: Mapping[str, Mapping[str, Any]],
    max_decay_lag: Any,
) -> dict[str, Any]:
    factors = sorted(factor_map)
    n_lags = _lag_count(factor_map, max_decay_lag)
    lags = [f"lag_{lag}" for lag in range(1, n_lags + 1)]
    cells: list[list[float | None]] = []
    ic_mean: dict[str, float | None] = {}
    ic_ir: dict[str, float | None] = {}
    for factor in factors:
        stats = factor_map[factor]
        autocorr = stats.get("rank_autocorr")
        row_vals: list[float | None] = []
        seq = (
            list(autocorr)
            if isinstance(autocorr, Sequence) and not isinstance(autocorr, (str, bytes))
            else []
        )
        for lag in range(n_lags):
            row_vals.append(_finite_float(seq[lag]) if lag < len(seq) else None)
        cells.append(row_vals)
        ic_mean[factor] = _finite_float(stats.get("ic_mean"))
        ic_ir[factor] = _finite_float(stats.get("ic_ir"))
    return {
        "factors": factors,
        "lags": lags,
        "cells": cells,
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
    }


def _factor_ranking(factor_map: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in sorted(factor_map):
        stats = factor_map[factor]
        rows.append(
            {
                "factor": factor,
                "ic_mean": _finite_float(stats.get("ic_mean")),
                "ic_ir": _finite_float(stats.get("ic_ir")),
                "ic_positive_ratio": _finite_float(stats.get("ic_positive_ratio")),
                "t_stat": _finite_float(stats.get("t_stat")),
                "turnover_mean": _finite_float(stats.get("turnover_mean")),
                "quantile_spread_mean": _finite_float(stats.get("quantile_spread_mean")),
                "n_periods": int(stats.get("n_periods") or 0),
            }
        )
    # Rank by IC-IR (desc, None last), then factor name for a total order.
    rows.sort(
        key=lambda r: (
            r["ic_ir"] is None,
            -(r["ic_ir"] or 0.0),
            r["factor"],
        )
    )
    return rows


def _candidate_score(row: Mapping[str, Any]) -> float | None:
    for key in _SCORE_KEYS:
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _build_candidate_queue(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        candidate_id = str(raw.get("candidate_id") or raw.get("id") or f"candidate_{index}")
        rows.append(
            {
                "candidate_id": candidate_id,
                "strategy": str(raw.get("strategy") or raw.get("strategy_name") or "unknown"),
                "status": str(raw.get("status") or "pending"),
                "score": _candidate_score(raw),
                "sharpe": _finite_float(raw.get("sharpe")),
                "robustness_score": _finite_float(raw.get("robustness_score")),
                "submitted_at": (
                    str(raw.get("submitted_at")) if raw.get("submitted_at") is not None else None
                ),
            }
        )
    # Highest score first (None last), then candidate_id for determinism.
    rows.sort(
        key=lambda r: (
            r["score"] is None,
            -(r["score"] or 0.0),
            r["candidate_id"],
        )
    )
    return rows


def build_factor_insights_payload(
    *,
    factor_ic: Any = None,
    candidate_queue: Sequence[Mapping[str, Any]] | None = None,
    max_decay_lag: int | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the read-only factor-insights dashboard payload.

    ``factor_ic`` may be a :class:`~lumina_quant.research.factor_ic.BatchFactorICResult`,
    its ``to_dict()`` mapping, or ``None``.  ``candidate_queue`` is an optional
    sequence of candidate mappings.  The result is a new top-level payload — it
    is never merged into the flat backtest metric dict.
    """
    factor_map = _coerce_factor_map(factor_ic)
    declared_lag = max_decay_lag
    if declared_lag is None and isinstance(factor_ic, Mapping):
        declared_lag = factor_ic.get("max_decay_lag")
    elif declared_lag is None and hasattr(factor_ic, "max_decay_lag"):
        declared_lag = getattr(factor_ic, "max_decay_lag", None)

    heatmap = _build_ic_heatmap(factor_map, declared_lag)
    ranking = _factor_ranking(factor_map)
    candidates = [dict(item) for item in list(candidate_queue or []) if isinstance(item, Mapping)]
    queue = _build_candidate_queue(candidates)

    top_factor = ranking[0]["factor"] if ranking else None
    top_factor_ic_ir = ranking[0]["ic_ir"] if ranking else None

    return {
        "artifact_kind": "dashboard_factor_insights_payload",
        "as_of": datetime.now(UTC).isoformat(),
        "advisory_only": True,
        "real_money_execution_enabled": False,
        "status": "ok" if factor_map or queue else "empty",
        "summary": {
            "factor_count": len(factor_map),
            "candidate_count": len(queue),
            "lag_count": len(heatmap["lags"]),
            "top_factor": top_factor,
            "top_factor_ic_ir": top_factor_ic_ir,
        },
        "ic_heatmap": heatmap,
        "factor_ranking": ranking,
        "candidate_queue": queue,
        "source": {
            "mode": "read_only_factor_insights",
            "status": "ok" if factor_map or queue else "empty",
            **dict(source or {}),
        },
    }


def load_factor_insights_payload(
    *,
    factor_ic_path: str | Path | None = None,
    candidate_queue_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load optional JSON artifacts and build the read-only payload."""
    factor_raw = _read_json(factor_ic_path)
    candidate_raw = _read_json(candidate_queue_path)
    return build_factor_insights_payload(
        factor_ic=factor_raw if isinstance(factor_raw, Mapping) else None,
        candidate_queue=[item for item in _as_list(candidate_raw) if isinstance(item, Mapping)],
        source={
            "factor_ic_path": str(factor_ic_path or ""),
            "candidate_queue_path": str(candidate_queue_path or ""),
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Module-mode entry for /api/python/dashboard/factor-insights."""
    parser = argparse.ArgumentParser(
        prog="lumina_quant.dashboard.factor_insights_service",
        description="Emit read-only factor IC-heatmap and candidate-queue dashboard payload.",
    )
    parser.add_argument("--json", action="store_true", default=True)
    parser.add_argument("--factor-ic", default="", dest="factor_ic_path")
    parser.add_argument("--candidate-queue", default="", dest="candidate_queue_path")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            load_factor_insights_payload(
                factor_ic_path=args.factor_ic_path,
                candidate_queue_path=args.candidate_queue_path,
            ),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


__all__ = [
    "build_factor_insights_payload",
    "load_factor_insights_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
