"""VWAP helpers for rolling and aggregate calculations."""

from __future__ import annotations

from itertools import islice


def vwap_from_sums(value_sum: float, volume_sum: float) -> float | None:
    """Return VWAP from running sums or ``None`` when denominator is invalid."""
    denom = float(volume_sum)
    if denom <= 0.0:
        return None
    return float(value_sum) / denom


def vwap_deviation(close_price: float, vwap_value: float | None) -> float | None:
    """Return percentage deviation from VWAP."""
    if vwap_value is None or vwap_value <= 0.0:
        return None
    return (float(close_price) / float(vwap_value)) - 1.0


def rolling_vwap(prices, volumes, window: int) -> float | None:
    """Return VWAP over the latest ``window`` aligned price/volume samples."""
    window_i = int(window)
    if window_i <= 1:
        return None
    if len(prices) < window_i or len(volumes) < window_i:
        return None

    start = len(prices) - window_i
    price_tail = islice(prices, start, None)
    volume_tail = islice(volumes, start, None)
    denom = 0.0
    numerator = 0.0
    for price, volume in zip(price_tail, volume_tail, strict=False):
        weight = max(0.0, float(volume))
        denom += weight
        numerator += float(price) * weight

    if denom <= 0.0:
        return None
    return numerator / denom
