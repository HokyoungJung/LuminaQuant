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

    Numbers below ``1e11`` are treated as epoch seconds, otherwise milliseconds.
    Returns ``None`` for ``None`` or unparseable input.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = int(float(value))
        return numeric * 1000 if abs(numeric) < 100_000_000_000 else numeric
    if isinstance(value, datetime):
        return int(utc_epoch_seconds(value) * 1000)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(utc_epoch_seconds(parsed) * 1000)


__all__ = ["as_utc", "utc_epoch_ms", "utc_epoch_seconds"]
