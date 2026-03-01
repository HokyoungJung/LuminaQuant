"""Compatibility wrappers for advanced research factors.

The canonical implementations live in :mod:`lumina_quant.indicators.advanced_alpha`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .advanced_alpha import (
    cross_leadlag_spillover,
    perp_crowding_score,
    pv_trend_score,
    volcomp_vwap_pressure,
)


def leadlag_spillover(
    price_by_symbol: Mapping[str, Sequence[float] | np.ndarray],
    *,
    window: int = 64,
    metals_exclusion: Sequence[str] = ("XAU/USDT", "XAG/USDT"),
    **kwargs: Any,
) -> dict[str, Any]:
    """Backwards-compatible lead/lag payload.

    Legacy callers expect `{leader, laggers, spillover_score, excluded_symbols}`.
    """

    _ = metals_exclusion
    result = cross_leadlag_spillover(
        price_by_symbol,
        window=max(64, int(window)),
        **kwargs,
    )
    preds = dict(result.get("predictions") or {})
    if not preds:
        return {
            "leader": None,
            "laggers": [],
            "spillover_score": 0.0,
            "excluded_symbols": list(result.get("excluded_symbols") or []),
            "sample_size": 0,
            "predictions": preds,
        }

    best_symbol = max(
        preds,
        key=lambda symbol: abs(float((preds.get(symbol) or {}).get("score", 0.0))),
    )
    spill = float((preds.get(best_symbol) or {}).get("score", 0.0))
    laggers = sorted(preds.keys())
    return {
        "leader": list(result.get("leaders") or [None])[0] if result.get("leaders") else None,
        "laggers": laggers,
        "spillover_score": float(spill),
        "excluded_symbols": list(result.get("excluded_symbols") or []),
        "sample_size": int(len(laggers)),
        "predictions": preds,
    }


__all__ = [
    "cross_leadlag_spillover",
    "leadlag_spillover",
    "perp_crowding_score",
    "pv_trend_score",
    "volcomp_vwap_pressure",
]
