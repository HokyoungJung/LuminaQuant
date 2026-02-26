from abc import ABC, abstractmethod
from typing import Any

from lumina_quant.core.events import MarketEvent


class Strategy(ABC):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) Strategy handling objects.
    """

    @abstractmethod
    def calculate_signals(self, event: Any) -> None:
        """Provides the mechanisms to calculate the list of signals."""
        raise NotImplementedError

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        """Optional cadence/window-aware callback (defaults to legacy behavior)."""
        _ = aggregator
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            bars_1s = dict(getattr(event, "bars_1s", {}) or {})
            for symbol, rows in bars_1s.items():
                if not rows:
                    continue
                latest = rows[-1]
                if isinstance(latest, (tuple, list)) and len(latest) >= 6:
                    self.calculate_signals(
                        MarketEvent(
                            time=latest[0],
                            symbol=str(symbol),
                            open=float(latest[1]),
                            high=float(latest[2]),
                            low=float(latest[3]),
                            close=float(latest[4]),
                            volume=float(latest[5]),
                        )
                    )
            return
        self.calculate_signals(event)

    def get_state(self) -> dict:
        """Backward-compatible default state for strategies that are stateless."""
        return {}

    def set_state(self, state: dict) -> None:
        """Backward-compatible default state loader."""
        _ = state

    @classmethod
    def get_param_schema(cls) -> dict:
        """Optional hyper-parameter schema for registry-driven tuning."""
        return {}
    decision_cadence_seconds: int | None = None
    required_timeframes: tuple[str, ...] = ()
    required_lookbacks: dict[str, int] = {}
