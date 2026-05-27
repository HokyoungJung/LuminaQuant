"""Strategy-signal dispatch helpers for research runner orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

StrategySignalHandler = Callable[
    [dict[str, Any], dict[str, np.ndarray], Sequence[str], int, np.ndarray, dict[str, Any]],
    None,
]


def _returns_from_close(closes: np.ndarray) -> np.ndarray:
    if closes.size < 2:
        return np.zeros(closes.shape, dtype=float)
    return np.diff(closes, prepend=closes[0]) / np.clip(
        np.r_[closes[0], closes[:-1]], 1e-12, np.inf
    )


@dataclass(frozen=True, slots=True)
class StrategySignalDispatcher:
    """Route strategy candidates to concrete exposure builders."""

    handlers: Mapping[str, StrategySignalHandler]
    minimum_symbol_counts: Mapping[str, int] = field(default_factory=dict)

    def dispatch(
        self,
        candidate: dict[str, Any],
        *,
        aligned: dict[str, np.ndarray],
        symbols: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        strategy_class = str(candidate.get("strategy_class") or candidate.get("strategy") or "")
        params = dict(candidate.get("params") or {})

        n = len(next(iter(aligned.values()))) if aligned else 0
        if n <= 0:
            empty = np.asarray([], dtype=float)
            return empty, empty, empty, {}
        if not symbols:
            empty = np.zeros(n, dtype=float)
            return empty, empty, empty, {}

        exposures = np.zeros((len(symbols), n), dtype=float)
        returns = np.zeros((len(symbols), n), dtype=float)

        for s_idx, symbol in enumerate(symbols):
            close = aligned[f"{symbol}:close"]
            returns[s_idx] = _returns_from_close(close)

        meta: dict[str, Any] = {}
        handler = self.handlers.get(strategy_class)
        required_symbols = int(self.minimum_symbol_counts.get(strategy_class, 1))
        if handler is None or len(symbols) < required_symbols:
            self._apply_generic_fallback(aligned=aligned, symbols=symbols, exposures=exposures)
        else:
            handler(params, aligned, symbols, n, exposures, meta)

        exposure = np.nanmean(exposures, axis=0)
        portfolio_ret = np.nanmean(np.roll(exposures, 1, axis=1) * returns, axis=0)
        turnover = np.nanmean(np.abs(exposures - np.roll(exposures, 1, axis=1)), axis=0)
        return portfolio_ret, turnover, exposure, meta

    @staticmethod
    def _apply_generic_fallback(
        *,
        aligned: dict[str, np.ndarray],
        symbols: Sequence[str],
        exposures: np.ndarray,
    ) -> None:
        for s_idx, symbol in enumerate(symbols):
            close = aligned[f"{symbol}:close"]
            ret = _returns_from_close(close)
            mom = np.nan_to_num(_rolling_z(ret, 64), nan=0.0)
            exposures[s_idx] = np.where(mom >= 0.4, 1.0, np.where(mom <= -0.4, -1.0, 0.0))


# Local helper retained to keep the generic fallback self-contained.
def _rolling_z(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if window <= 1:
        return np.zeros(arr.shape, dtype=float)

    out = np.zeros(arr.shape, dtype=float)
    for idx in range(window - 1, arr.size):
        segment = arr[idx - window + 1 : idx + 1]
        mean = float(np.nanmean(segment))
        std = float(np.nanstd(segment))
        if not np.isfinite(std) or std <= 0.0:
            out[idx] = 0.0
            continue
        out[idx] = (float(arr[idx]) - mean) / std
    return out
