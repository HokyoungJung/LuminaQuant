"""Deterministic replay helpers for ordered event streams."""

from __future__ import annotations

from typing import Any


def stable_event_sort(events: list[Any]) -> list[Any]:
    return sorted(
        events,
        key=lambda event: (
            int(getattr(event, "timestamp_ns", 0) or 0),
            int(getattr(event, "sequence", 0) or 0),
        ),
    )


def assert_monotonic_event_order(events: list[Any]) -> None:
    prev_ts = -1
    prev_seq = -1
    for event in stable_event_sort(events):
        ts = int(getattr(event, "timestamp_ns", 0) or 0)
        seq = int(getattr(event, "sequence", 0) or 0)
        if ts < prev_ts:
            raise AssertionError("Event timestamp_ns is not monotonic")
        if ts == prev_ts and seq < prev_seq:
            raise AssertionError("Event sequence is not monotonic within timestamp")
        prev_ts = ts
        prev_seq = seq
