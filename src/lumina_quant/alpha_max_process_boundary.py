"""Dependency-free process boundary for the frozen Alpha-Max commands."""

from __future__ import annotations

import os

__all__ = [
    "AlphaMaxRuntimeContractError",
    "AmbientLQEnvironmentError",
    "reject_ambient_lq_environment",
]


class AlphaMaxRuntimeContractError(ValueError):
    """The sealed runtime contract or one of its construction inputs is invalid."""


class AmbientLQEnvironmentError(AlphaMaxRuntimeContractError):
    """An ambient ``LQ_*`` environment key would make the replay non-hermetic."""


def reject_ambient_lq_environment() -> None:
    """Reject every ambient key beginning with ``LQ_`` without reading its value."""
    offending = tuple(sorted(key for key in os.environ if key.startswith("LQ_")))
    if offending:
        raise AmbientLQEnvironmentError(f"ambient_lq_environment:{','.join(offending)}")
