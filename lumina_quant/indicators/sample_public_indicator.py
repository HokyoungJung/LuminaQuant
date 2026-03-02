"""Public sample indicator helpers."""

from __future__ import annotations


def sample_momentum(close_now: float, close_prev: float) -> float:
    return float(close_now) - float(close_prev)
