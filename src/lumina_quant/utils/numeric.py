"""Shared numeric coercion helpers."""

from __future__ import annotations

import math
from typing import Any


def safe_float(value: Any, default: Any = None, *, finite_only: bool = False) -> Any:
    """Coerce to float; return ``default`` on failure.

    If ``finite_only`` is set, also return ``default`` for inf/nan. Never raises.
    """
    try:
        parsed = float(value)
    except Exception:
        return default
    if finite_only and not math.isfinite(parsed):
        return default
    return parsed
