from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) Strategy handling objects.
    """

    @abstractmethod
    def calculate_signals(self, event: Any) -> None:
        """Provides the mechanisms to calculate the list of signals."""
        raise NotImplementedError

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
