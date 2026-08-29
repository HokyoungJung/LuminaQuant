"""Host-timezone-independent time coercions.

Repo convention: a tz-naive ``datetime`` (or tz-less ISO string) is a UTC wall
time -- the data handlers, parquet repositories and engine gates all store and
compare bar times that way.  ``datetime.timestamp()`` on a naive value would
silently apply the host's local timezone and shift cursors, window bounds and
event identity by the UTC offset, so every epoch conversion should go through
these helpers instead.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any


def as_utc(value: datetime) -> datetime:
    """Return ``value`` as a tz-aware UTC datetime (naive input is taken as UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_epoch_seconds(value: datetime) -> float:
    return as_utc(value).timestamp()


def utc_epoch_ms(value: Any) -> int | None:
    """Epoch milliseconds for a datetime / ISO string / epoch number (naive == UTC).

    Numeric magnitudes below ``100_000_000_000`` are epoch seconds; larger
    magnitudes are epoch milliseconds. Fractional epoch seconds retain their
    millisecond component. Boolean and non-finite numeric inputs are rejected.
    Returns ``None`` for ``None`` or unparseable input.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not isfinite(numeric):
            return None
        if abs(numeric) < 100_000_000_000:
            try:
                return int(Decimal(str(value)) * 1000)
            except InvalidOperation:
                return None
        try:
            return int(Decimal(str(value)))
        except InvalidOperation:
            return None
    if isinstance(value, datetime):
        parsed = as_utc(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = as_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None
    delta = parsed - datetime(1970, 1, 1, tzinfo=UTC)
    total_microseconds = (
        delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
    )
    return (
        total_microseconds // 1000 if total_microseconds >= 0 else -((-total_microseconds) // 1000)
    )


__all__ = ["as_utc", "utc_epoch_ms", "utc_epoch_seconds"]
