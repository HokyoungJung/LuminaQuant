"""Shadow-live helper for side-by-side strategy evaluation on live event stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumina_quant.replay import stable_event_sort


@dataclass(slots=True)
class ShadowLiveResult:
    events_processed: int
    divergence_count: int


class ShadowLiveRunner:
    """Run deterministic replay/shadow checks using existing strategy engine contracts."""

    def __init__(self, *, divergence_tolerance: float = 1e-9) -> None:
        self.divergence_tolerance = float(divergence_tolerance)

    def run(self, *, baseline_events: list[Any], candidate_events: list[Any]) -> ShadowLiveResult:
        baseline = stable_event_sort(list(baseline_events or []))
        candidate = stable_event_sort(list(candidate_events or []))
        total = min(len(baseline), len(candidate))
        divergence = abs(len(baseline) - len(candidate))
        for idx in range(total):
            lhs = baseline[idx]
            rhs = candidate[idx]
            if int(getattr(lhs, "timestamp_ns", 0) or 0) != int(
                getattr(rhs, "timestamp_ns", 0) or 0
            ):
                divergence += 1
                continue
            if int(getattr(lhs, "sequence", 0) or 0) != int(getattr(rhs, "sequence", 0) or 0):
                divergence += 1
        return ShadowLiveResult(events_processed=total, divergence_count=int(divergence))


__all__ = ["ShadowLiveResult", "ShadowLiveRunner"]
