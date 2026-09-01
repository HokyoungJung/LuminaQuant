from __future__ import annotations

from pathlib import Path
from typing import Any

from lumina_quant.eval.exact_window_runtime import HeavyRunLock

ROOT = Path(__file__).resolve().parents[3]
FOLLOWUP_ROOT = ROOT / "var" / "reports" / "exact_window_backtests" / "followup_status"
DEFAULT_SESSION_MEMORY_LEASE_PATH = FOLLOWUP_ROOT / "session_memory_budget.lock"


def acquire_session_memory_lease(
    *,
    label: str,
    requested_budget_bytes: int,
    effective_budget_bytes: int,
    metadata: dict[str, Any] | None = None,
    lock_path: str | Path = DEFAULT_SESSION_MEMORY_LEASE_PATH,
) -> HeavyRunLock:
    payload = {
        "requested_budget_bytes": int(requested_budget_bytes),
        "effective_budget_bytes": int(effective_budget_bytes),
        **dict(metadata or {}),
    }
    return HeavyRunLock.acquire(
        lock_path=lock_path,
        label=str(label),
        metadata=payload,
    )


__all__ = [
    "DEFAULT_SESSION_MEMORY_LEASE_PATH",
    "FOLLOWUP_ROOT",
    "acquire_session_memory_lease",
]
