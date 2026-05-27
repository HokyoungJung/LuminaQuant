"""Common, dependency-light helpers used by indicator modules."""

from __future__ import annotations

import math


def safe_float(value) -> float | None:
    """Return a finite float or ``None`` when parsing fails."""
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def safe_int(value, default: int = 0) -> int:
    """Return ``int(value)`` or fallback to ``default``."""
    try:
        return int(value)
    except Exception:
        return int(default)


def time_key(value) -> str:
    """Normalize timestamp-like values into a stable key string."""
    return "" if value is None else str(value)
