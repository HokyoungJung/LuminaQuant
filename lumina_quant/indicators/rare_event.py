"""Rare-event scoring indicators optimized for low-memory, latest-value use."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class RareEventScore:
    """Latest rare-event score bundle (all scores are in [0, 1])."""

    rare_return_score: float
    rare_return_lookback: int
    rare_streak_score: float
    rare_streak_value: int
    trend_break_score: float
    local_extremum_score: float
    local_extremum_side: int  # +1 near local-high extreme, -1 near local-low extreme
    composite_score: float


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _as_float_array(values: Iterable[float] | np.ndarray, *, max_points: int | None) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float64, copy=False)
    else:
        arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if max_points is not None and max_points > 0 and arr.size > max_points:
        arr = arr[-max_points:]
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
    return arr


def _one_step_change(prices: np.ndarray, *, diff: bool) -> np.ndarray:
    if prices.size < 2:
        return np.asarray([], dtype=np.float64)
    prev = prices[:-1]
    curr = prices[1:]
    if diff:
        return curr - prev
    out = np.full(curr.shape, np.nan, dtype=np.float64)
    valid = np.abs(prev) > _EPS
    out[valid] = (curr[valid] / prev[valid]) - 1.0
    return out[np.isfinite(out)]


def _lag_change(prices: np.ndarray, lag: int, *, diff: bool) -> np.ndarray:
    if lag <= 0 or prices.size <= lag:
        return np.asarray([], dtype=np.float64)
    prev = prices[:-lag]
    curr = prices[lag:]
    if diff:
        return curr - prev
    out = np.full(curr.shape, np.nan, dtype=np.float64)
    valid = np.abs(prev) > _EPS
    out[valid] = (curr[valid] / prev[valid]) - 1.0
    return out[np.isfinite(out)]


def rare_return_score_latest(
    prices: Iterable[float] | np.ndarray,
    *,
    lookbacks: Sequence[int] = (1, 2, 3, 4, 5),
    factor: float = 1.0,
    diff: bool = False,
    max_points: int = 4096,
) -> tuple[float, int]:
    """Return latest return-rarity score and selected lookback."""
    arr = _as_float_array(prices, max_points=max_points)
    if arr.size < 4:
        fallback = int(lookbacks[0]) if lookbacks else 1
        return 1.0, fallback

    candidate_lookbacks = [max(1, int(x)) for x in lookbacks if int(x) > 0]
    if not candidate_lookbacks:
        candidate_lookbacks = [1]

    best_score = 1.0
    best_lookback = candidate_lookbacks[0]

    for idx, lookback in enumerate(candidate_lookbacks):
        changes = _lag_change(arr, lookback, diff=diff)
        if changes.size < 2:
            continue
        current = float(changes[-1])
        if current >= 0.0:
            same_sign = changes[changes >= 0.0]
            if same_sign.size == 0:
                continue
            more_extreme = int(np.count_nonzero(same_sign > current + _EPS))
        else:
            same_sign = changes[changes < 0.0]
            if same_sign.size == 0:
                continue
            more_extreme = int(np.count_nonzero(same_sign < current - _EPS))

        rank_ratio = more_extreme / float(max(1, same_sign.size))
        weight = math.exp(float(factor) * idx)
        # Keep range [0,1] while preserving the "longer horizon penalty" idea.
        weighted_score = 1.0 - ((1.0 - rank_ratio) / max(1.0, weight))
        weighted_score = _clamp01(weighted_score)

        if weighted_score < best_score:
            best_score = weighted_score
            best_lookback = lookback

    return best_score, best_lookback


def rare_streak_score_latest(
    prices: Iterable[float] | np.ndarray,
    *,
    diff: bool = False,
    max_points: int = 4096,
) -> tuple[float, int]:
    """Return latest directional-streak rarity score and signed streak length."""
    arr = _as_float_array(prices, max_points=max_points)
    changes = _one_step_change(arr, diff=diff)
    if changes.size == 0:
        return 1.0, 0

    signs = np.sign(changes).astype(np.int8, copy=False)
    for idx in range(1, signs.size):
        if signs[idx] == 0:
            signs[idx] = signs[idx - 1]

    if signs.size > 0 and signs[0] == 0:
        nz = np.flatnonzero(signs != 0)
        if nz.size > 0:
            signs[: int(nz[0])] = signs[int(nz[0])]

    streak = np.zeros(signs.size, dtype=np.int32)
    for idx, sign in enumerate(signs):
        if sign == 0:
            streak[idx] = 0
            continue
        if idx > 0 and signs[idx - 1] == sign:
            streak[idx] = streak[idx - 1] + int(sign)
        else:
            streak[idx] = int(sign)

    current = int(streak[-1])
    if current == 0:
        return 1.0, 0

    if current > 0:
        same_direction = streak[streak > 0]
        if same_direction.size == 0:
            return 1.0, current
        extreme_count = int(np.count_nonzero(same_direction >= current))
    else:
        same_direction = streak[streak < 0]
        if same_direction.size == 0:
            return 1.0, current
        extreme_count = int(np.count_nonzero(same_direction <= current))

    score = (max(0, extreme_count - 1)) / float(max(1, same_direction.size))
    return _clamp01(score), current


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def trend_break_score_latest(
    prices: Iterable[float] | np.ndarray,
    *,
    rolling_window: int = 20,
    horizons: Sequence[int] = tuple(range(2, 11)),
    diff: bool = False,
    max_points: int = 4096,
) -> float:
    """Return latest trend-break rarity score in [0, 1] (lower is rarer)."""
    arr = _as_float_array(prices, max_points=max_points)
    window = max(5, int(rolling_window))
    if arr.size < window + 12:
        return 1.0

    base_changes = _one_step_change(arr, diff=diff)
    if base_changes.size == 0:
        return 1.0
    base_last = float(base_changes[-1]) if math.isfinite(float(base_changes[-1])) else 0.0

    values: list[float] = []
    for horizon in [max(2, int(x)) for x in horizons]:
        if arr.size < window + horizon:
            continue

        idx = np.arange(arr.size - window, arr.size, dtype=np.int64)
        prev_idx = idx - horizon
        curr_vals = arr[idx]
        prev_vals = arr[prev_idx]

        if diff:
            momentum = curr_vals - prev_vals
        else:
            momentum = np.full(curr_vals.shape, np.nan, dtype=np.float64)
            valid = np.abs(prev_vals) > _EPS
            momentum[valid] = (curr_vals[valid] / prev_vals[valid]) - 1.0

        finite = momentum[np.isfinite(momentum)]
        if finite.size < max(3, window // 2):
            continue

        latest_momentum = float(momentum[-1])
        if not math.isfinite(latest_momentum):
            continue

        mean_value = float(np.mean(finite))
        std_value = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        if std_value <= _EPS:
            z_value = 0.0
        else:
            z_value = (latest_momentum - mean_value) / std_value

        if abs(base_last) <= _EPS or (z_value * base_last) > 0.0:
            z_value = 0.0

        rarity = _normal_cdf(z_value)
        if rarity > 0.5:
            rarity = 1.0 - rarity
        values.append(_clamp01(rarity * 2.0))

    if not values:
        return 1.0
    return _clamp01(float(np.mean(values)))


def local_extremum_score_latest(
    prices: Iterable[float] | np.ndarray,
    *,
    rolling_window: int = 200,
    max_points: int = 4096,
) -> tuple[float, int, int, int]:
    """Return local-extremum rarity score and side/+ages.

    Returns:
      (score, side, max_age, min_age)
      - score: [0, 1], lower is rarer
      - side: +1 -> high-side extreme, -1 -> low-side extreme, 0 -> neutral
    """
    arr = _as_float_array(prices, max_points=max_points)
    window = max(10, int(rolling_window))
    if arr.size < window:
        return 1.0, 0, 0, 0

    tail = arr[-window:]
    max_value = float(np.max(tail))
    min_value = float(np.min(tail))
    span = max_value - min_value
    if span <= _EPS:
        return 1.0, 0, 0, 0

    scaled_tail = (tail - min_value) / span
    scaled_last = float(scaled_tail[-1])
    std_scaled = float(np.std(scaled_tail, ddof=1)) if scaled_tail.size > 1 else 0.0
    if not math.isfinite(std_scaled):
        std_scaled = 0.0

    max_idx = int(np.where(np.isclose(tail, max_value, atol=_EPS, rtol=0.0))[0][-1])
    min_idx = int(np.where(np.isclose(tail, min_value, atol=_EPS, rtol=0.0))[0][-1])
    max_age = int(window - max_idx)
    min_age = int(window - min_idx)

    if max_age == 1:
        max_gamma = math.exp(-1.0 / max(1, max_age))
        min_gamma = math.exp(-std_scaled * min_age)
    elif min_age == 1:
        max_gamma = math.exp(-std_scaled * max_age)
        min_gamma = math.exp(-1.0 / max(1, min_age))
    else:
        max_gamma = math.exp(-std_scaled * max_age)
        min_gamma = math.exp(-std_scaled * min_age)

    max_diff = _clamp01(1.0 - (scaled_last * max_gamma))
    min_diff = _clamp01(1.0 - ((1.0 - scaled_last) * min_gamma))

    if max_diff <= min_diff:
        return max_diff, 1, max_age, min_age
    return min_diff, -1, max_age, min_age


def rare_event_scores_latest(
    prices: Iterable[float] | np.ndarray,
    *,
    lookbacks: Sequence[int] = (1, 2, 3, 4, 5),
    return_factor: float = 1.0,
    trend_rolling_window: int = 20,
    local_extremum_window: int = 200,
    diff: bool = False,
    max_points: int = 4096,
    component_weights: Sequence[float] = (0.30, 0.25, 0.20, 0.25),
) -> RareEventScore | None:
    """Compute latest composite rare-event score from close prices.

    Lower score means rarer state. All component outputs are constrained to [0, 1].
    """
    arr = _as_float_array(prices, max_points=max_points)
    if arr.size < 8:
        return None

    return_score, return_lookback = rare_return_score_latest(
        arr,
        lookbacks=lookbacks,
        factor=return_factor,
        diff=diff,
        max_points=max_points,
    )
    streak_score, streak_value = rare_streak_score_latest(arr, diff=diff, max_points=max_points)
    trend_score = trend_break_score_latest(
        arr,
        rolling_window=trend_rolling_window,
        diff=diff,
        max_points=max_points,
    )
    local_score, local_side, _, _ = local_extremum_score_latest(
        arr,
        rolling_window=local_extremum_window,
        max_points=max_points,
    )

    weights = [max(0.0, float(w)) for w in component_weights]
    if len(weights) != 4 or sum(weights) <= _EPS:
        weights = [0.30, 0.25, 0.20, 0.25]

    total_weight = float(sum(weights))
    composite = (
        (weights[0] * return_score)
        + (weights[1] * streak_score)
        + (weights[2] * trend_score)
        + (weights[3] * local_score)
    ) / total_weight

    return RareEventScore(
        rare_return_score=_clamp01(return_score),
        rare_return_lookback=int(return_lookback),
        rare_streak_score=_clamp01(streak_score),
        rare_streak_value=int(streak_value),
        trend_break_score=_clamp01(trend_score),
        local_extremum_score=_clamp01(local_score),
        local_extremum_side=int(local_side),
        composite_score=_clamp01(composite),
    )


def load_close_tail_from_lazy(
    lazy_frame: pl.LazyFrame,
    *,
    close_column: str = "close",
    max_points: int = 4096,
    mode: str | None = None,
    device: str | int | None = None,
) -> np.ndarray:
    """Collect only the close-column tail from a LazyFrame for memory efficiency."""
    max_rows = max(1, int(max_points))
    projected = lazy_frame.select(pl.col(close_column)).tail(max_rows)

    try:
        from lumina_quant.compute_engine import collect as collect_with_engine

        frame = collect_with_engine(projected, mode=mode, device=device)
    except Exception:
        frame = projected.collect(engine="streaming")

    if close_column not in frame.columns:
        return np.asarray([], dtype=np.float64)
    return _as_float_array(frame.get_column(close_column).to_numpy(), max_points=max_rows)


def rare_event_scores_from_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    close_column: str = "close",
    max_points: int = 4096,
    mode: str | None = None,
    device: str | int | None = None,
    kwargs: dict[str, Any] | None = None,
) -> RareEventScore | None:
    """Compute rare-event scores while loading only minimal close tail data."""
    params = dict(kwargs or {})
    if isinstance(frame, pl.LazyFrame):
        closes = load_close_tail_from_lazy(
            frame,
            close_column=close_column,
            max_points=max_points,
            mode=mode,
            device=device,
        )
    else:
        if close_column not in frame.columns:
            return None
        closes = _as_float_array(
            frame.select(pl.col(close_column)).tail(max(1, int(max_points))).to_numpy().reshape(-1),
            max_points=max_points,
        )
    return rare_event_scores_latest(closes, max_points=max_points, **params)


__all__ = [
    "RareEventScore",
    "load_close_tail_from_lazy",
    "local_extremum_score_latest",
    "rare_event_scores_from_frame",
    "rare_event_scores_latest",
    "rare_return_score_latest",
    "rare_streak_score_latest",
    "trend_break_score_latest",
]
