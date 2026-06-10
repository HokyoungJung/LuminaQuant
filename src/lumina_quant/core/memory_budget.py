"""Shared memory-budget defaults + cap enforcement for low-RAM execution paths."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any

BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0


class MemoryCapExceededError(RuntimeError):
    """Raised at engine start when process RSS exceeds the configured cap.

    This is the enforcement teeth behind ``RuntimeConfig.memory.cap_gb``:
    lowering the cap in config genuinely lowers the ceiling a run may use.
    """


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

    @classmethod
    def from_cap_gb(cls, cap_gb: float) -> ExecutionMemoryPolicy:
        """Scale the canonical policy to a new total cap, preserving ratios."""
        base = cls()
        cap = float(cap_gb)
        heavy_ratio = base.heavy_run_cap_gib / base.total_memory_cap_gib
        return replace(base, total_memory_cap_gib=cap, heavy_run_cap_gib=cap * heavy_ratio)

    @classmethod
    def from_runtime(cls, runtime: Any) -> ExecutionMemoryPolicy:
        """Build a policy from ``RuntimeConfig.memory.cap_gb`` (config-driven cap)."""
        cap = getattr(getattr(runtime, "memory", None), "cap_gb", None)
        if cap is None:
            return cls()
        return cls.from_cap_gb(float(cap))


def gib_to_bytes(value: int | float) -> int:
    """Convert GiB to bytes with the repo's canonical base-2 multiplier."""
    return int(float(value) * BYTES_PER_GIB)


def current_rss_bytes() -> int | None:
    """Best-effort current resident set size in bytes (``None`` if unmeasurable).

    Prefers psutil (accurate, cross-platform); falls back to ``getrusage`` peak
    RSS so the guard still works in minimal environments without psutil.
    """
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        import resource
        import sys

        maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports kibibytes; macOS reports bytes.
        return maxrss if sys.platform == "darwin" else maxrss * 1024
    except Exception:
        return None


def enforce_memory_cap(
    cap_gb: float | None,
    *,
    label: str = "execution",
    rss_bytes: int | None = None,
) -> int | None:
    """Engine-start guard: raise if current RSS exceeds ``cap_gb``.

    Returns the measured RSS in bytes, or ``None`` when the cap is not a
    positive finite number or RSS cannot be measured (guard becomes a no-op
    rather than failing the run on an unmeasurable host).
    """
    try:
        cap = float(cap_gb) if cap_gb is not None else None
    except TypeError, ValueError:
        return None
    if cap is None or not math.isfinite(cap) or cap <= 0.0:
        return None

    rss = rss_bytes if rss_bytes is not None else current_rss_bytes()
    if rss is None:
        return None

    if rss > gib_to_bytes(cap):
        raise MemoryCapExceededError(
            f"{label}: resident memory {rss / BYTES_PER_GIB:.2f} GiB exceeds configured "
            f"memory.cap_gb={cap:.2f} GiB. Reduce the workload (symbols/window/parallelism) "
            f"or raise memory.cap_gb in config."
        )
    return rss


def enforce_runtime_memory_cap(runtime: Any, *, label: str = "execution") -> int | None:
    """Enforce ``RuntimeConfig.memory.cap_gb`` at engine start. Config-driven."""
    cap = getattr(getattr(runtime, "memory", None), "cap_gb", None)
    return enforce_memory_cap(cap, label=label)


DEFAULT_EXECUTION_MEMORY_POLICY = ExecutionMemoryPolicy()

__all__ = [
    "BYTES_PER_GIB",
    "DEFAULT_EXECUTION_MEMORY_POLICY",
    "ExecutionMemoryPolicy",
    "MemoryCapExceededError",
    "current_rss_bytes",
    "enforce_memory_cap",
    "enforce_runtime_memory_cap",
    "gib_to_bytes",
]
