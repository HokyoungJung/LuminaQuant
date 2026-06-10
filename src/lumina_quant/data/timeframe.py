"""Single canonical timeframe parsing / normalization util.

This is the ONE home for timeframe-token parsing.  It is intentionally a leaf
module (stdlib ``re`` only, no ``lumina_quant`` imports) so that
``market_data`` and the ``data.*`` / ``storage.*`` modules can all import from
here without forming the ``market_data <-> data.*`` import cycle.
"""

from __future__ import annotations

import re

TIMEFRAME_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
    "M": 2_592_000_000,
}
_TIMEFRAME_PATTERN = re.compile(r"^([1-9][0-9]*)([smhdwM])$")


def normalize_timeframe_token(timeframe: str) -> str:
    """Normalize timeframe token while preserving month/minute semantics."""
    raw = str(timeframe or "").strip()
    if not raw:
        raise ValueError("Timeframe cannot be empty")
    if len(raw) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    value = raw[:-1].strip()
    unit_raw = raw[-1]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"Invalid timeframe value: {timeframe}")

    if unit_raw == "M":
        unit = "M"
    else:
        unit = unit_raw.lower()
    token = f"{int(value)}{unit}"
    if _TIMEFRAME_PATTERN.fullmatch(token) is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return token


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert timeframe token like 1m/1h/1d into milliseconds."""
    token = normalize_timeframe_token(timeframe)
    if len(token) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    unit = token[-1]
    value = int(token[:-1])
    if value <= 0:
        raise ValueError(f"Invalid timeframe value: {timeframe}")
    unit_ms = TIMEFRAME_UNIT_MS.get(unit)
    if unit_ms is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return value * unit_ms


__all__ = ["TIMEFRAME_UNIT_MS", "normalize_timeframe_token", "timeframe_to_milliseconds"]
