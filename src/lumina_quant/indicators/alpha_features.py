"""Portable alpha-feature helpers for strategy prototyping.

The helpers are deliberately pure and dependency-light: they do no I/O and can
be reused by event-driven strategies, research runners, and later native-kernel
ports.  The feature set is inspired by widely-tested alpha families: time-series
momentum, volatility scaling/crash gating, liquidity shocks, order-flow
imbalance, and perpetual-futures carry/basis dislocation.
"""

from __future__ import annotations

import math
from itertools import pairwise
from collections.abc import Iterable, Sequence
from typing import Any

from lumina_quant.core.plugin_registry import register

_EPS = 1e-12


def finite_floats(values: Iterable[Any] | None) -> list[float]:
    """Return finite floats while preserving input order."""
    if values is None:
        return []
    out: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            out.append(parsed)
    return out


def clipped_tanh_score(value: float | None, *, scale: float = 1.0) -> float:
    """Map an unbounded signal to ``[-1, 1]`` with a configurable scale."""
    if value is None:
        return 0.0
    denom = max(float(scale), _EPS)
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return float(math.tanh(parsed / denom))


def log_return(values: Sequence[float] | Iterable[float], *, lookback: int = 1) -> float | None:
    """Return latest log return over ``lookback`` observations."""
    vals = finite_floats(values)
    lag = max(1, int(lookback))
    if len(vals) <= lag:
        return None
    latest = vals[-1]
    base = vals[-1 - lag]
    if latest <= 0.0 or base <= 0.0:
        return None
    return float(math.log(latest / base))


def simple_return(values: Sequence[float] | Iterable[float], *, lookback: int = 1) -> float | None:
    """Return latest simple return over ``lookback`` observations."""
    vals = finite_floats(values)
    lag = max(1, int(lookback))
    if len(vals) <= lag:
        return None
    latest = vals[-1]
    base = vals[-1 - lag]
    if abs(base) <= _EPS:
        return None
    return float(latest / base - 1.0)


def rolling_zscore(
    values: Sequence[float] | Iterable[float],
    *,
    window: int = 64,
    history_excludes_latest: bool = True,
) -> float | None:
    """Return z-score of the latest value against a trailing window."""
    vals = finite_floats(values)
    win = max(3, int(window))
    need = win + 1 if history_excludes_latest else win
    if len(vals) < need:
        return None
    tail = vals[-need:]
    latest = tail[-1]
    hist = tail[:-1] if history_excludes_latest else tail
    if len(hist) < 2:
        return None
    mean_value = sum(hist) / float(len(hist))
    variance = sum((value - mean_value) ** 2 for value in hist) / float(len(hist) - 1)
    sigma = math.sqrt(max(0.0, variance))
    if sigma <= _EPS:
        delta = latest - mean_value
        if abs(delta) <= _EPS:
            return 0.0
        return 6.0 if delta > 0.0 else -6.0
    return float((latest - mean_value) / sigma)


def realized_volatility(
    values: Sequence[float] | Iterable[float],
    *,
    window: int = 64,
    annualization: float = 1.0,
) -> float | None:
    """Return rolling close-to-close log-return volatility."""
    vals = finite_floats(values)
    win = max(2, int(window))
    if len(vals) < win + 1:
        return None
    tail = vals[-(win + 1) :]
    returns: list[float] = []
    for prev, cur in pairwise(tail):
        if prev <= 0.0 or cur <= 0.0:
            continue
        returns.append(math.log(cur / prev))
    if len(returns) < 2:
        return None
    mean_value = sum(returns) / float(len(returns))
    variance = sum((value - mean_value) ** 2 for value in returns) / float(len(returns) - 1)
    vol = math.sqrt(max(0.0, variance))
    ann = max(0.0, float(annualization))
    if ann > 0.0:
        vol *= math.sqrt(ann)
    return float(vol) if math.isfinite(vol) else None


def volatility_ratio(
    values: Sequence[float] | Iterable[float],
    *,
    fast_window: int = 16,
    slow_window: int = 96,
) -> float | None:
    """Return fast realized volatility divided by slow realized volatility."""
    slow = realized_volatility(values, window=max(3, int(slow_window)))
    fast = realized_volatility(values, window=max(2, int(fast_window)))
    if fast is None or slow is None or slow <= _EPS:
        return None
    return float(fast / slow)


def trend_efficiency(
    values: Sequence[float] | Iterable[float], *, window: int = 64
) -> float | None:
    """Return Kaufman-style trend efficiency in ``[0, 1]``."""
    vals = finite_floats(values)
    win = max(2, int(window))
    if len(vals) < win + 1:
        return None
    tail = vals[-(win + 1) :]
    net = abs(tail[-1] - tail[0])
    path = sum(abs(cur - prev) for prev, cur in pairwise(tail))
    if path <= _EPS:
        return 0.0
    return max(0.0, min(1.0, float(net / path)))


def volume_zscore(volumes: Sequence[float] | Iterable[float], *, window: int = 64) -> float | None:
    """Return z-score of latest volume against prior trailing volume."""
    return rolling_zscore(volumes, window=max(3, int(window)), history_excludes_latest=True)


def range_zscore(
    highs: Sequence[float] | Iterable[float],
    lows: Sequence[float] | Iterable[float],
    closes: Sequence[float] | Iterable[float],
    *,
    window: int = 64,
) -> float | None:
    """Return z-score of latest normalized high-low range."""
    hi = finite_floats(highs)
    lo = finite_floats(lows)
    cl = finite_floats(closes)
    n = min(len(hi), len(lo), len(cl))
    if n < max(4, int(window) + 1):
        return None
    ranges = [
        max(0.0, hi[-n + idx] - lo[-n + idx]) / max(abs(cl[-n + idx]), _EPS) for idx in range(n)
    ]
    return rolling_zscore(ranges, window=max(3, int(window)), history_excludes_latest=True)


def order_flow_imbalance(
    buy_volume: Sequence[float] | Iterable[float] | float | None,
    sell_volume: Sequence[float] | Iterable[float] | float | None,
    *,
    window: int | None = None,
) -> float | None:
    """Return normalized order-flow imbalance ``(buy - sell) / (buy + sell)``."""

    def _coerce_sum(raw: Sequence[float] | Iterable[float] | float | None) -> float:
        if raw is None:
            return 0.0
        if isinstance(raw, (int, float)):
            return max(0.0, float(raw)) if math.isfinite(float(raw)) else 0.0
        vals = finite_floats(raw)
        if window is not None:
            vals = vals[-max(1, int(window)) :]
        return float(sum(max(0.0, value) for value in vals))

    buy = _coerce_sum(buy_volume)
    sell = _coerce_sum(sell_volume)
    total = buy + sell
    if total <= _EPS:
        return None
    return max(-1.0, min(1.0, float((buy - sell) / total)))


def basis_bps(mark_price: float | None, index_price: float | None) -> float | None:
    """Return mark-vs-index basis in basis points."""
    try:
        mark = float(mark_price) if mark_price is not None else float("nan")
        index = float(index_price) if index_price is not None else float("nan")
    except Exception:
        return None
    if not math.isfinite(mark) or not math.isfinite(index) or index <= 0.0:
        return None
    return float((mark / index - 1.0) * 10_000.0)


def amihud_illiquidity(
    closes: Sequence[float] | Iterable[float],
    volumes: Sequence[float] | Iterable[float],
    *,
    window: int = 64,
    scale: float = 1_000_000.0,
) -> float | None:
    """Return latest Amihud-style ``abs(return) / dollar-volume`` liquidity proxy."""
    cls = finite_floats(closes)
    vol = finite_floats(volumes)
    n = min(len(cls), len(vol))
    win = max(2, int(window))
    if n < win + 1:
        return None
    cls = cls[-(win + 1) :]
    vol = vol[-(win + 1) :]
    values: list[float] = []
    for prev, cur, raw_volume in zip(cls[:-1], cls[1:], vol[1:], strict=True):
        if prev <= 0.0 or cur <= 0.0 or raw_volume <= 0.0:
            continue
        dollar_volume = max(cur * raw_volume, _EPS)
        values.append(abs(math.log(cur / prev)) / dollar_volume)
    if not values:
        return None
    return float(sum(values) / len(values) * max(1.0, float(scale)))


def drawdown_from_peak(
    values: Sequence[float] | Iterable[float], *, window: int = 64
) -> float | None:
    """Return latest drawdown from trailing peak as a negative or zero fraction."""
    vals = finite_floats(values)
    win = max(2, int(window))
    if len(vals) < win:
        return None
    tail = vals[-win:]
    peak = max(tail)
    if peak <= 0.0:
        return None
    return min(0.0, float(tail[-1] / peak - 1.0))


@register("indicator", "AlphaFeaturePack")
class AlphaFeaturePack:
    """Registry wrapper exposing portable alpha feature helpers."""

    finite_floats = staticmethod(finite_floats)
    clipped_tanh_score = staticmethod(clipped_tanh_score)
    log_return = staticmethod(log_return)
    simple_return = staticmethod(simple_return)
    rolling_zscore = staticmethod(rolling_zscore)
    realized_volatility = staticmethod(realized_volatility)
    volatility_ratio = staticmethod(volatility_ratio)
    trend_efficiency = staticmethod(trend_efficiency)
    volume_zscore = staticmethod(volume_zscore)
    range_zscore = staticmethod(range_zscore)
    order_flow_imbalance = staticmethod(order_flow_imbalance)
    basis_bps = staticmethod(basis_bps)
    amihud_illiquidity = staticmethod(amihud_illiquidity)
    drawdown_from_peak = staticmethod(drawdown_from_peak)


__all__ = [
    "AlphaFeaturePack",
    "amihud_illiquidity",
    "basis_bps",
    "clipped_tanh_score",
    "drawdown_from_peak",
    "finite_floats",
    "log_return",
    "order_flow_imbalance",
    "range_zscore",
    "realized_volatility",
    "rolling_zscore",
    "simple_return",
    "trend_efficiency",
    "volatility_ratio",
    "volume_zscore",
]
