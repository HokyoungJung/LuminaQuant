"""Threading controls to avoid Polars/Numba oversubscription."""

from __future__ import annotations

import os


def configure_numba_threads(max_workers: int) -> int | None:
    """Configure Numba thread count for optimization loops.

    Priority:
    1) Explicit LQ_NUMBA_THREADS env var
    2) CPU count / worker count heuristic
    """
    try:
        from numba import get_num_threads, set_num_threads
    except Exception:
        return None

    env_override = str(os.getenv("LQ_NUMBA_THREADS", "")).strip()
    if env_override:
        try:
            target = max(1, int(env_override))
            set_num_threads(target)
            return int(get_num_threads())
        except Exception:
            return int(get_num_threads())

    cpu_total = max(1, int(os.cpu_count() or 1))
    workers = max(1, int(max_workers))
    target = max(1, cpu_total // workers)
    try:
        set_num_threads(target)
    except Exception:
        pass
    return int(get_num_threads())
