"""Deterministic event identity helpers (timestamp_ns + sequence)."""

from __future__ import annotations

from datetime import UTC, datetime


def _epoch_seconds(value: datetime) -> float:
    """Naive datetimes are UTC (same convention as ``core.engine._event_time_to_ms``
    and the data handlers); host-local interpretation would make event identity
    differ between hosts and break cross-host replay parity.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).timestamp()


def normalize_timestamp_ns(value) -> int:
    if value is None:
        return 0
    if isinstance(value, datetime):
        return int(_epoch_seconds(value) * 1_000_000_000)
    if isinstance(value, (int, float)):
        raw = int(value)
        if abs(raw) < 100_000_000_000:
            return raw * 1_000_000
        if abs(raw) < 100_000_000_000_000:
            return raw * 1_000
        return raw
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return int(_epoch_seconds(parsed) * 1_000_000_000)
    except Exception:
        return 0


class EventSequencer:
    def __init__(self):
        self._sequence = 0

    def next(self) -> int:
        self._sequence += 1
        return self._sequence


def assign_event_identity(event, sequencer: EventSequencer) -> None:
    if event is None:
        return
    current_ns = getattr(event, "timestamp_ns", None)
    if current_ns is None:
        time_value = getattr(event, "time", None)
        if time_value is None:
            time_value = getattr(event, "timeindex", None)
        event.timestamp_ns = normalize_timestamp_ns(time_value)
    if getattr(event, "sequence", None) is None:
        event.sequence = sequencer.next()
