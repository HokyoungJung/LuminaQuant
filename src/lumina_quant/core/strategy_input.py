"""Canonical strategy-input context helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StrategyInputContext:
    """Canonical context passed to strategies that opt into richer inputs."""

    event: Any
    aggregator: Any = None
    feature_lookup: Any = None
    data_handler: Any = None
    execution_handler: Any = None
    exchange: Any = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
