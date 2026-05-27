"""Incremental RSI indicator implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RsiState:
    prev_close: float | None = None
    avg_gain: float | None = None
    avg_loss: float | None = None
    gain_sum: float = 0.0
    loss_sum: float = 0.0
    samples: int = 0


class IncrementalRsi:
    """Wilder RSI with allocation-light incremental updates."""

    def __init__(self, period: int, state: RsiState | None = None):
        self.period = max(1, int(period))
        self.state = state or RsiState()

    def update(self, close_price: float) -> float | None:
        current = float(close_price)
        prev_close = self.state.prev_close
        self.state.prev_close = current
        if prev_close is None:
            return None

        delta = current - prev_close
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0

        if self.state.avg_gain is None or self.state.avg_loss is None:
            self.state.gain_sum += gain
            self.state.loss_sum += loss
            self.state.samples += 1
            if self.state.samples < self.period:
                return None

            period = float(self.period)
            self.state.avg_gain = self.state.gain_sum / period
            self.state.avg_loss = self.state.loss_sum / period
        else:
            period = float(self.period)
            keep_weight = float(self.period - 1)
            self.state.avg_gain = (self.state.avg_gain * keep_weight + gain) / period
            self.state.avg_loss = (self.state.avg_loss * keep_weight + loss) / period

        avg_gain = self.state.avg_gain
        avg_loss = self.state.avg_loss
        if avg_gain is None or avg_loss is None:
            return None

        if avg_loss <= 0.0:
            return 100.0 if avg_gain > 0.0 else 0.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def to_state(self) -> dict:
        return {
            "prev_close": self.state.prev_close,
            "avg_gain": self.state.avg_gain,
            "avg_loss": self.state.avg_loss,
            "gain_sum": self.state.gain_sum,
            "loss_sum": self.state.loss_sum,
            "samples": self.state.samples,
        }

    def load_state(self, raw_state: dict) -> None:
        if not isinstance(raw_state, dict):
            return
        prev_close = raw_state.get("prev_close")
        avg_gain = raw_state.get("avg_gain")
        avg_loss = raw_state.get("avg_loss")
        gain_sum = raw_state.get("gain_sum")
        loss_sum = raw_state.get("loss_sum")
        samples = raw_state.get("samples")

        self.state.prev_close = float(prev_close) if prev_close is not None else None
        self.state.avg_gain = float(avg_gain) if avg_gain is not None else None
        self.state.avg_loss = float(avg_loss) if avg_loss is not None else None
        self.state.gain_sum = float(gain_sum) if gain_sum is not None else 0.0
        self.state.loss_sum = float(loss_sum) if loss_sum is not None else 0.0
        self.state.samples = int(samples) if samples is not None else 0
