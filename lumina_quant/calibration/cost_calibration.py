"""Calibration utilities for strategy-specific impact coefficients."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def calibrate_impact_coefficients(
    fills: list[dict[str, Any]],
    current_k: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Calibrate k_strategy via realized/predicted impact ratio."""
    grouped_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for fill in fills:
        strategy = str(fill.get("strategy") or "")
        predicted = float(fill.get("impact_bps", 0.0))
        realized = float(fill.get("realized_impact_bps", fill.get("impact_bps", 0.0)))
        if strategy and predicted > 0:
            grouped_pairs[strategy].append((predicted, realized))

    output: dict[str, dict[str, float]] = {}
    for strategy, base_value in current_k.items():
        pairs = grouped_pairs.get(strategy, [])
        ratios = [max(0.0, realized / predicted) for predicted, realized in pairs if predicted > 0.0]
        ratio = sum(ratios) / len(ratios) if ratios else 1.0
        new_value = float(base_value) * float(ratio)
        mae_before = (
            sum(abs(realized - predicted) for predicted, realized in pairs) / len(pairs)
            if pairs
            else 0.0
        )
        mae_after = (
            sum(abs(realized - (predicted * ratio)) for predicted, realized in pairs) / len(pairs)
            if pairs
            else 0.0
        )
        error_reduction = float(mae_before - mae_after)
        error_reduction_pct = 0.0 if mae_before <= 0.0 else float(error_reduction / mae_before)
        output[strategy] = {
            "old_k": float(base_value),
            "new_k": float(new_value),
            "ratio": float(ratio),
            "observations": len(pairs),
            "mae_before_bps": float(mae_before),
            "mae_after_bps": float(mae_after),
            "error_reduction_bps": error_reduction,
            "error_reduction_pct": error_reduction_pct,
        }

    return output


__all__ = ["calibrate_impact_coefficients"]
