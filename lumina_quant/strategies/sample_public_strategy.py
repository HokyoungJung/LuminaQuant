"""Public sample strategy for open-source distribution."""

from __future__ import annotations

from lumina_quant.strategy import Strategy


class PublicSampleStrategy(Strategy):
    """No-op sample strategy used when private strategies are not published."""

    decision_cadence_seconds = 20
    required_timeframes = ("1s",)
    required_lookbacks = {"1s": 1}

    def __init__(self, bars, events, **params):
        self.bars = bars
        self.events = events
        self.params = dict(params)

    @classmethod
    def get_param_schema(cls) -> dict:
        return {}

    def calculate_signals(self, event):
        _ = event
        return None
