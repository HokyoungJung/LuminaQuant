"""Average True Range primitives."""

from __future__ import annotations


def true_range(high: float, low: float, prev_close: float | None) -> float:
    """Return the classic True Range for a single bar."""
    high_f = float(high)
    low_f = float(low)
    if prev_close is None:
        return max(0.0, high_f - low_f)

    prev_close_f = float(prev_close)
    return max(
        high_f - low_f,
        abs(high_f - prev_close_f),
        abs(low_f - prev_close_f),
    )


def average_true_range(tr_values, window: int) -> float | None:
    """Return ATR over the most recent ``window`` true-range samples."""
    window_i = max(1, int(window))
    if len(tr_values) < window_i:
        return None
    tail = list(tr_values)[-window_i:]
    return sum(float(value) for value in tail) / float(window_i)
