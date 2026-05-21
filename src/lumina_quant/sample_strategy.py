"""Educational sample strategy used only for public pipeline tests."""

from __future__ import annotations

from collections import deque

from lumina_quant.models import Bar, Signal, TargetPosition


class MovingAverageCrossStrategy:
    """Long/flat moving-average crossover over close prices.

    This is deliberately simple and educational. It is not investment advice,
    not a production strategy, and not connected to any exchange.
    """

    def __init__(self, fast_window: int = 3, slow_window: int = 8) -> None:
        if fast_window < 1:
            raise ValueError("fast_window must be >= 1")
        if slow_window <= fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self._closes: deque[float] = deque(maxlen=self.slow_window)
        self._target = TargetPosition.FLAT

    def on_bar(self, bar: Bar) -> Signal:
        self._closes.append(float(bar.close))
        if len(self._closes) < self.slow_window:
            return Signal(bar.timestamp, self._target, "warming_up")

        closes = list(self._closes)
        fast = sum(closes[-self.fast_window :]) / self.fast_window
        slow = sum(closes) / self.slow_window
        next_target = TargetPosition.LONG if fast > slow else TargetPosition.FLAT
        reason = (
            "fast_ma_above_slow_ma"
            if next_target is TargetPosition.LONG
            else "fast_ma_not_above_slow_ma"
        )
        self._target = next_target
        return Signal(bar.timestamp, next_target, reason)
