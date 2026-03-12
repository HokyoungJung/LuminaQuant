"""Runtime helpers for exact-window RSS logging and memory guards."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.core.memory_budget import (
    DEFAULT_EXECUTION_MEMORY_POLICY,
    gib_to_bytes,
)

_BYTES_PER_MIB = 1024.0 * 1024.0
_CGROUP_MAX = "max"
_ENV_BUDGET_BYTES = "LQ_EXACT_WINDOW_BUDGET_BYTES"
_ENV_BUDGET_GIB = "LQ_EXACT_WINDOW_BUDGET_GIB"


def bytes_to_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / _BYTES_PER_MIB


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_proc_status_rss_bytes(pid: int) -> int | None:
    status = _read_text(Path(f"/proc/{pid}/status"))
    if not status:
        return None
    for line in status.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        try:
            return int(fields[1]) * 1024
        except ValueError:
            return None
    return None


def current_rss_bytes(pid: int | None = None) -> int | None:
    resolved_pid = int(pid or os.getpid())
    return _read_proc_status_rss_bytes(resolved_pid)


def _read_memavailable_bytes() -> int | None:
    meminfo = _read_text(Path("/proc/meminfo"))
    if not meminfo:
        return None
    for line in meminfo.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        try:
            return int(fields[1]) * 1024
        except ValueError:
            return None
    return None


def _read_cgroup_limit_bytes() -> int | None:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        raw = _read_text(path)
        if not raw or raw == _CGROUP_MAX:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= (1 << 60):
            continue
        return value
    return None


def _read_env_budget_bytes() -> int | None:
    raw_bytes = str(os.getenv(_ENV_BUDGET_BYTES, "")).strip()
    if raw_bytes:
        try:
            value = int(float(raw_bytes))
        except ValueError:
            value = 0
        if value > 0:
            return value

    raw_gib = str(os.getenv(_ENV_BUDGET_GIB, "")).strip()
    if raw_gib:
        try:
            value = float(raw_gib)
        except ValueError:
            value = 0.0
        if value > 0.0:
            return gib_to_bytes(value)
    return None


def resolve_memory_budget_bytes() -> int | None:
    env_budget = _read_env_budget_bytes()
    cgroup_limit = _read_cgroup_limit_bytes()
    memavailable = _read_memavailable_bytes()
    candidates = [value for value in (cgroup_limit, memavailable, env_budget) if value is not None and value > 0]
    if not candidates:
        return None
    return min(candidates)


def _process_running(pid: int | None) -> bool:
    try:
        resolved = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if resolved <= 0:
        return False
    try:
        os.kill(resolved, 0)
    except OSError:
        return False
    return True


class HeavyRunActiveError(RuntimeError):
    """Raised when another heavy exact-window backtest already owns the lock."""

    def __init__(self, *, lock_path: str | Path, active_payload: dict[str, Any] | None = None) -> None:
        self.lock_path = Path(lock_path).resolve()
        self.active_payload = dict(active_payload or {})
        pid = self.active_payload.get("pid")
        run_id = self.active_payload.get("run_id") or "unknown"
        super().__init__(
            f"Another heavy exact-window run is active (pid={pid}, run_id={run_id}, lock={self.lock_path})"
        )


class HeavyRunLock:
    """Single-writer lock for heavy exact-window backtests."""

    def __init__(self, *, lock_path: str | Path, payload: dict[str, Any]) -> None:
        self.lock_path = Path(lock_path).resolve()
        self.payload = dict(payload)
        self._released = False

    @staticmethod
    def _read_payload(lock_path: Path) -> dict[str, Any]:
        try:
            raw = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    @classmethod
    def acquire(
        cls,
        *,
        lock_path: str | Path,
        label: str = "exact_window",
        metadata: dict[str, Any] | None = None,
    ) -> HeavyRunLock:
        resolved = Path(lock_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": str(label),
            "pid": int(os.getpid()),
            "started_at_utc": datetime.now(UTC).isoformat(),
            **dict(metadata or {}),
        }
        active_payload: dict[str, Any] = {}
        for _attempt in range(2):
            try:
                fd = os.open(resolved, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                active_payload = cls._read_payload(resolved)
                active_pid = active_payload.get("pid")
                if _process_running(active_pid):
                    raise HeavyRunActiveError(lock_path=resolved, active_payload=active_payload)
                try:
                    resolved.unlink()
                except OSError:
                    raise HeavyRunActiveError(lock_path=resolved, active_payload=active_payload) from None
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True))
            return cls(lock_path=resolved, payload=payload)
        raise HeavyRunActiveError(lock_path=resolved, active_payload=active_payload)

    def release(self) -> None:
        if self._released:
            return
        current = self._read_payload(self.lock_path)
        current_pid = current.get("pid")
        current_run_id = current.get("run_id")
        if (
            int(self.payload.get("pid") or 0) > 0
            and int(current_pid or 0) == int(self.payload.get("pid") or 0)
            and current_run_id == self.payload.get("run_id")
        ):
            try:
                self.lock_path.unlink()
            except OSError:
                pass
        self._released = True


class RSSLimitExceeded(RuntimeError):
    """Raised when the hard RSS limit is exceeded."""

    def __init__(self, *, rss_bytes: int, hard_limit_bytes: int, event: str) -> None:
        self.rss_bytes = int(rss_bytes)
        self.hard_limit_bytes = int(hard_limit_bytes)
        self.event = str(event)
        super().__init__(
            "RSS hard limit exceeded: "
            f"rss={bytes_to_mib(self.rss_bytes):.2f}MiB >= "
            f"hard_limit={bytes_to_mib(self.hard_limit_bytes):.2f}MiB "
            f"during {self.event}"
        )


class RSSGuard:
    """Checkpoint-based RSS logger and threshold guard."""

    def __init__(
        self,
        *,
        log_path: str | Path,
        soft_limit_bytes: int | None = None,
        hard_limit_bytes: int | None = None,
        budget_bytes: int | None = None,
        label: str = "exact_window",
    ) -> None:
        self.log_path = Path(log_path).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.label = str(label)
        self.budget_bytes = int(budget_bytes) if budget_bytes is not None else resolve_memory_budget_bytes()
        self.soft_limit_bytes = (
            int(soft_limit_bytes)
            if soft_limit_bytes is not None
            else int(self.budget_bytes * DEFAULT_EXECUTION_MEMORY_POLICY.exact_window_soft_fraction)
            if self.budget_bytes is not None
            else None
        )
        self.hard_limit_bytes = (
            int(hard_limit_bytes)
            if hard_limit_bytes is not None
            else int(self.budget_bytes * DEFAULT_EXECUTION_MEMORY_POLICY.exact_window_hard_fraction)
            if self.budget_bytes is not None
            else None
        )
        self.samples: list[dict[str, Any]] = []
        self.soft_trigger_count = 0
        self.hard_trigger_count = 0

    def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        rss_bytes = current_rss_bytes()
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "label": self.label,
            "event": str(event),
            "rss_bytes": int(rss_bytes) if rss_bytes is not None else None,
            "rss_mib": bytes_to_mib(rss_bytes),
            "budget_bytes": self.budget_bytes,
            "budget_mib": bytes_to_mib(self.budget_bytes),
            "soft_limit_bytes": self.soft_limit_bytes,
            "soft_limit_mib": bytes_to_mib(self.soft_limit_bytes),
            "hard_limit_bytes": self.hard_limit_bytes,
            "hard_limit_mib": bytes_to_mib(self.hard_limit_bytes),
            "context": dict(context or {}),
        }
        if rss_bytes is not None and self.soft_limit_bytes is not None and rss_bytes >= self.soft_limit_bytes:
            self.soft_trigger_count += 1
            record["soft_limit_exceeded"] = True
        else:
            record["soft_limit_exceeded"] = False
        if rss_bytes is not None and self.hard_limit_bytes is not None and rss_bytes >= self.hard_limit_bytes:
            self.hard_trigger_count += 1
            record["hard_limit_exceeded"] = True
        else:
            record["hard_limit_exceeded"] = False
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.samples.append(record)
        return record

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        record = self.sample(event=event, context=context)
        rss_bytes = record.get("rss_bytes")
        if (
            isinstance(rss_bytes, int)
            and self.hard_limit_bytes is not None
            and rss_bytes >= self.hard_limit_bytes
        ):
            raise RSSLimitExceeded(
                rss_bytes=rss_bytes,
                hard_limit_bytes=self.hard_limit_bytes,
                event=str(event),
            )

    def finalize(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        rss_values = [
            int(record["rss_bytes"])
            for record in self.samples
            if isinstance(record.get("rss_bytes"), int)
        ]
        peak_rss_bytes = max(rss_values) if rss_values else None
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "label": self.label,
            "status": str(status),
            "error": str(error) if error else None,
            "sample_count": len(self.samples),
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_mib": bytes_to_mib(peak_rss_bytes),
            "budget_bytes": self.budget_bytes,
            "budget_mib": bytes_to_mib(self.budget_bytes),
            "soft_limit_bytes": self.soft_limit_bytes,
            "soft_limit_mib": bytes_to_mib(self.soft_limit_bytes),
            "hard_limit_bytes": self.hard_limit_bytes,
            "hard_limit_mib": bytes_to_mib(self.hard_limit_bytes),
            "soft_trigger_count": int(self.soft_trigger_count),
            "hard_trigger_count": int(self.hard_trigger_count),
            "rss_log_path": str(self.log_path),
        }


__all__ = [
    "HeavyRunActiveError",
    "HeavyRunLock",
    "RSSGuard",
    "RSSLimitExceeded",
    "_process_running",
    "bytes_to_mib",
    "current_rss_bytes",
    "resolve_memory_budget_bytes",
]
