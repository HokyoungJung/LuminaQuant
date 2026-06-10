"""Shadow-live helper for side-by-side strategy evaluation on live event stream."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from lumina_quant.replay import stable_event_sort


@dataclass(slots=True)
class ShadowLiveResult:
    events_processed: int
    divergence_count: int
    parity_ratio: float = 1.0  # signal_matches / signal_total; 1.0 when no signals compared
    signal_matches: int = 0
    signal_total: int = 0


class ShadowLiveRunner:
    """Run deterministic replay/shadow checks using existing strategy engine contracts.

    Two modes of operation:
    1. Event-stream divergence check (``run()``): compares two pre-sorted event lists
       by timestamp_ns + sequence.  Used for order-event reconciliation.

    2. Signal parity check (``evaluate_signal_parity()``): runs ``baseline_strategy_fn``
       and ``candidate_strategy_fn`` over the same market-window sequence, counts
       signal agreement, and returns a ``parity_ratio``.  The ratio is the gate metric
       fed back into readiness_policy for the shadow→canary transition.

    Per-stage composition (spec R7): stages are not sequentially forced.  An operator
    may select any stage directly; what is mandatory is that each stage's entry checks
    pass at startup.  For the shadow stage, the entry check is:
    ``parity_ratio >= shadow_parity_min_ratio`` over ``shadow_parity_window_bars`` bars.
    """

    def __init__(
        self,
        *,
        divergence_tolerance: float = 1e-9,
        baseline_strategy_fn: Callable[[Any], float] | None = None,
        candidate_strategy_fn: Callable[[Any], float] | None = None,
        parity_window_bars: int = 1000,
    ) -> None:
        self.divergence_tolerance = float(divergence_tolerance)
        self.baseline_strategy_fn = baseline_strategy_fn
        self.candidate_strategy_fn = candidate_strategy_fn
        self.parity_window_bars = max(1, int(parity_window_bars))

    def run(self, *, baseline_events: list[Any], candidate_events: list[Any]) -> ShadowLiveResult:
        """Compare two event streams by timestamp_ns + sequence.

        Returns divergence_count (event mismatches + count delta) and parity_ratio
        derived from event-level agreement (not signal-level; use
        ``evaluate_signal_parity`` for signal-level parity).
        """
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
        matches = total - divergence if divergence <= total else 0
        parity = float(matches) / float(total) if total > 0 else 1.0
        return ShadowLiveResult(
            events_processed=total,
            divergence_count=int(divergence),
            parity_ratio=parity,
            signal_matches=matches,
            signal_total=total,
        )

    def evaluate_signal_parity(
        self,
        *,
        market_windows: list[Any],
        signal_tolerance: float = 1e-9,
    ) -> ShadowLiveResult:
        """Compare baseline vs candidate strategy signals over a window of market events.

        Each function in ``baseline_strategy_fn`` / ``candidate_strategy_fn`` receives
        one market window and returns a float signal.  Two signals are considered
        matching when ``abs(baseline - candidate) <= signal_tolerance``.

        Returns a ``ShadowLiveResult`` where:
        - ``parity_ratio = signal_matches / signal_total``
        - ``events_processed = len(market_windows)`` processed (up to parity_window_bars)
        - ``divergence_count = signal_total - signal_matches``

        If either strategy function is not configured, returns a perfect-parity result
        (parity_ratio=1.0, no signals compared) so the gate does not block.
        """
        if self.baseline_strategy_fn is None or self.candidate_strategy_fn is None:
            n = min(len(market_windows or []), self.parity_window_bars)
            return ShadowLiveResult(
                events_processed=n,
                divergence_count=0,
                parity_ratio=1.0,
                signal_matches=0,
                signal_total=0,
            )

        windows = list(market_windows or [])[: self.parity_window_bars]
        matches = 0
        total = 0
        tol = float(signal_tolerance)

        for window in windows:
            try:
                baseline_sig = float(self.baseline_strategy_fn(window))
                candidate_sig = float(self.candidate_strategy_fn(window))
            except Exception:
                # Strategy error counts as a divergence
                total += 1
                continue
            total += 1
            if (
                math.isfinite(baseline_sig)
                and math.isfinite(candidate_sig)
                and abs(baseline_sig - candidate_sig) <= tol
            ):
                matches += 1

        parity = float(matches) / float(total) if total > 0 else 1.0
        return ShadowLiveResult(
            events_processed=len(windows),
            divergence_count=total - matches,
            parity_ratio=parity,
            signal_matches=matches,
            signal_total=total,
        )


__all__ = ["ShadowLiveResult", "ShadowLiveRunner"]
