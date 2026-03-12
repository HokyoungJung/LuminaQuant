"""Shared memory-budget defaults for low-RAM execution paths."""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class ExecutionMemoryPolicy:
    """Canonical memory-policy defaults for session-safe execution."""

    total_memory_cap_gib: float = 8.0
    heavy_run_cap_gib: float = 6.5
    disk_budget_gib: float = 30.0
    heavy_run_parallelism: int = 1
    light_worker_parallelism: int = 2
    exact_window_soft_fraction: float = 0.60
    exact_window_hard_fraction: float = 0.80
    rss_limit_fraction: float = 0.90

    @property
    def total_memory_cap_bytes(self) -> int:
        return gib_to_bytes(self.total_memory_cap_gib)

    @property
    def rss_limit_gib(self) -> float:
        return float(self.total_memory_cap_gib) * float(self.rss_limit_fraction)


def gib_to_bytes(value: int | float) -> int:
    """Convert GiB to bytes with the repo's canonical base-2 multiplier."""
    return int(float(value) * BYTES_PER_GIB)


DEFAULT_EXECUTION_MEMORY_POLICY = ExecutionMemoryPolicy()

__all__ = [
    "BYTES_PER_GIB",
    "DEFAULT_EXECUTION_MEMORY_POLICY",
    "ExecutionMemoryPolicy",
    "gib_to_bytes",
]
