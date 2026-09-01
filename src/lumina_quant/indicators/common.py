"""Common, dependency-light helpers used by indicator modules."""

from __future__ import annotations

from lumina_quant.utils.numeric import safe_float as _canon_safe_float


def safe_float(value) -> float | None:
    """Return a finite float or ``None`` when parsing fails."""
    return _canon_safe_float(value, finite_only=True)


def safe_int(value, default: int = 0) -> int:
    """Return ``int(value)`` or fallback to ``default``."""
    try:
        return int(value)
    except Exception:
        return int(default)


def time_key(value) -> str:
    """Normalize timestamp-like values into a stable key string."""
    return "" if value is None else str(value)
