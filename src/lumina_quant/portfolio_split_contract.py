"""Canonical split windows and memory-guard helpers for portfolio follow-up lanes."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from lumina_quant.core.memory_budget import DEFAULT_EXECUTION_MEMORY_POLICY
from lumina_quant.core.session_memory import (
    DEFAULT_SESSION_MEMORY_LEASE_PATH,
    acquire_session_memory_lease,
)
from lumina_quant.eval.exact_window_runtime import HeavyRunLock, RSSGuard

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "var" / "reports" / "exact_window_backtests"
FOLLOWUP_ROOT = REPORT_ROOT / "followup_status"

_ENV_TRAIN_START = "LQ_PORTFOLIO_TRAIN_START"
_ENV_TRAIN_END = "LQ_PORTFOLIO_TRAIN_END"
_ENV_VAL_START = "LQ_PORTFOLIO_VAL_START"
_ENV_VAL_END = "LQ_PORTFOLIO_VAL_END"
_ENV_OOS_START = "LQ_PORTFOLIO_OOS_START"


def _parse_day_env(value: str | None, *, default: str) -> date:
    token = str(value or default).strip()
    if not token:
        token = default
    token = token.split("T", 1)[0]
    return date.fromisoformat(token)


TRAIN_START_DATE = _parse_day_env(os.getenv(_ENV_TRAIN_START), default="2025-01-01")
TRAIN_END_DATE = _parse_day_env(os.getenv(_ENV_TRAIN_END), default="2025-12-31")
VAL_START_DATE = _parse_day_env(os.getenv(_ENV_VAL_START), default="2026-01-01")
VAL_END_DATE = _parse_day_env(os.getenv(_ENV_VAL_END), default="2026-01-31")
OOS_START_DATE = _parse_day_env(os.getenv(_ENV_OOS_START), default="2026-02-01")

TRAIN_START = f"{TRAIN_START_DATE.isoformat()}T00:00:00Z"
TRAIN_END_EXCLUSIVE = f"{(TRAIN_END_DATE + timedelta(days=1)).isoformat()}T00:00:00Z"
VAL_START = f"{VAL_START_DATE.isoformat()}T00:00:00Z"
VAL_END_EXCLUSIVE = f"{(VAL_END_DATE + timedelta(days=1)).isoformat()}T00:00:00Z"
OOS_START = f"{OOS_START_DATE.isoformat()}T00:00:00Z"

PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE = (
    FOLLOWUP_ROOT / "portfolio_one_shot_incumbent_bundle_latest.json"
)
PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE = FOLLOWUP_ROOT / "portfolio_one_shot_current_bundle_latest.json"
PORTFOLIO_CURRENT_OPTIMIZATION = (
    FOLLOWUP_ROOT / "portfolio_one_shot_current_opt" / "portfolio_optimization_latest.json"
)

PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH = FOLLOWUP_ROOT / "portfolio_followup_heavy_run.lock"
PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH = DEFAULT_SESSION_MEMORY_LEASE_PATH
_ENV_PORTFOLIO_FOLLOWUP_BUDGET_BYTES = "LQ_PORTFOLIO_FOLLOWUP_BUDGET_BYTES"
_ENV_PORTFOLIO_FOLLOWUP_BUDGET_GIB = "LQ_PORTFOLIO_FOLLOWUP_BUDGET_GIB"
MEMORY_GUARD_DIRNAME = "_memory_guard"


def _read_portfolio_followup_env_budget_bytes() -> int | None:
    raw_bytes = str(os.getenv(_ENV_PORTFOLIO_FOLLOWUP_BUDGET_BYTES, "")).strip()
    if raw_bytes:
        try:
            value = int(float(raw_bytes))
        except ValueError:
            value = 0
        if value > 0:
            return value

    raw_gib = str(os.getenv(_ENV_PORTFOLIO_FOLLOWUP_BUDGET_GIB, "")).strip()
    if raw_gib:
        try:
            value = float(raw_gib)
        except ValueError:
            value = 0.0
        if value > 0.0:
            return int(value * 1024 * 1024 * 1024)
    return None


def portfolio_followup_default_budget_bytes() -> int:
    """Resolve the default follow-up budget with env overrides and heavy-run headroom."""
    env_budget = _read_portfolio_followup_env_budget_bytes()
    if env_budget is not None:
        return env_budget
    return int(DEFAULT_EXECUTION_MEMORY_POLICY.heavy_run_cap_gib * 1024 * 1024 * 1024)


PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES = portfolio_followup_default_budget_bytes()


def split_windows() -> dict[str, str]:
    """Return the canonical portfolio research split boundaries."""
    return {
        "train_start": TRAIN_START,
        "train_end_exclusive": TRAIN_END_EXCLUSIVE,
        "val_start": VAL_START,
        "val_end_exclusive": VAL_END_EXCLUSIVE,
        "oos_start": OOS_START,
    }


def split_dates() -> dict[str, date]:
    """Return the split boundaries as day-level dates."""
    return {
        "train_start": TRAIN_START_DATE,
        "val_start": VAL_START_DATE,
        "oos_start": OOS_START_DATE,
    }


def split_for_date(day_value: date) -> str:
    """Map a date into the canonical train/val/oos bucket."""
    if day_value < VAL_START_DATE:
        return "train"
    if day_value < OOS_START_DATE:
        return "val"
    return "oos"


def split_for_day_key(day_key: str) -> str:
    """Map a YYYY-MM-DD token into the canonical train/val/oos bucket."""
    return split_for_date(date.fromisoformat(day_key))


def resolve_incumbent_bundle_path(path: str | Path | None = None) -> Path:
    """Prefer the incumbent bundle and fall back to the legacy current bundle."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    if PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE not in candidates:
        candidates.append(PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE)
    if PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE not in candidates:
        candidates.append(PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE)
    for candidate in candidates:
        resolved = resolve_followup_artifact_path(candidate)
        if resolved.exists():
            return resolved.resolve()
    if path is not None:
        return resolve_followup_artifact_path(Path(path)).resolve()
    return resolve_followup_artifact_path(PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE).resolve()


def resolve_followup_artifact_path(path: str | Path) -> Path:
    """Resolve follow-up artifacts from the repo root, falling back to OMX worktrees."""
    candidate = Path(path)
    if candidate.is_absolute():
        repo_candidate = candidate
        try:
            candidate = candidate.relative_to(ROOT)
        except ValueError:
            return repo_candidate
    else:
        repo_candidate = ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    omx_root = ROOT / ".omx"
    if not omx_root.exists():
        return repo_candidate

    fallback_matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in (
        f"worktrees/*/{candidate.as_posix()}",
        f"manual-worktrees/*/{candidate.as_posix()}",
        f"team/*/worktrees/*/{candidate.as_posix()}",
    ):
        for matched in omx_root.glob(pattern):
            if not matched.exists():
                continue
            resolved = matched.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            fallback_matches.append(resolved)
    fallback_matches.sort(
        key=lambda matched: (matched.stat().st_mtime_ns, matched.as_posix()),
        reverse=True,
    )
    return fallback_matches[0] if fallback_matches else repo_candidate


def resolve_current_optimization_path(path: str | Path | None = None) -> Path:
    """Resolve the current optimization artifact, including OMX worktree fallbacks."""
    candidate = PORTFOLIO_CURRENT_OPTIMIZATION if path is None else Path(path)
    return resolve_followup_artifact_path(candidate)


def memory_policy_payload(*, budget_bytes: int | None = None) -> dict[str, Any]:
    """Return the canonical memory-budget policy for portfolio follow-up lanes."""
    policy = asdict(DEFAULT_EXECUTION_MEMORY_POLICY)
    policy["total_memory_cap_bytes"] = DEFAULT_EXECUTION_MEMORY_POLICY.total_memory_cap_bytes
    policy["rss_limit_gib"] = DEFAULT_EXECUTION_MEMORY_POLICY.rss_limit_gib
    policy["heavy_lock_path"] = str(PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH.resolve())
    policy["session_memory_lease_path"] = str(
        PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH.resolve()
    )
    if budget_bytes is not None:
        policy["explicit_budget_bytes"] = int(budget_bytes)
        policy["explicit_budget_gib"] = float(int(budget_bytes) / (1024**3))
    return policy


@dataclass(slots=True)
class PortfolioMemoryGuard:
    """Wrapper around the exact-window heavy-run lock and RSS telemetry."""

    run_name: str
    label: str
    output_dir: Path
    rss_log_path: Path
    summary_path: Path
    lock: HeavyRunLock
    guard: RSSGuard
    session_lease: HeavyRunLock | None = None

    def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.guard.sample(event=event, context=context)

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        self.guard.checkpoint(event, context)

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = {
            "artifact_kind": "portfolio_followup_memory_summary",
            "schema_version": "1.0",
            "run_name": self.run_name,
            "label": self.label,
            "output_dir": str(self.output_dir),
            "memory_policy": memory_policy_payload(budget_bytes=self.guard.budget_bytes),
            "context": dict(context or {}),
            **self.guard.finalize(status=status, error=error),
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def release(self) -> None:
        try:
            self.lock.release()
        finally:
            if self.session_lease is not None:
                self.session_lease.release()


def acquire_portfolio_memory_guard(
    *,
    run_name: str,
    output_dir: str | Path,
    input_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    budget_bytes: int | None = None,
    memory_budget_bytes: int | None = None,
    fixed_budget_bytes: int | None = None,
    rss_log_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    soft_limit_bytes: int | None = None,
    hard_limit_bytes: int | None = None,
) -> PortfolioMemoryGuard:
    """Acquire the shared heavy-run lock and RSS logger for a portfolio lane."""
    resolved_budget_bytes = next(
        (
            int(value)
            for value in (budget_bytes, memory_budget_bytes, fixed_budget_bytes)
            if value is not None
        ),
        None,
    )
    resolved_output_dir = Path(output_dir).resolve()
    memory_dir = resolved_output_dir / MEMORY_GUARD_DIRNAME
    memory_dir.mkdir(parents=True, exist_ok=True)
    label = f"portfolio_followup::{run_name}"
    effective_budget_bytes = max(
        1,
        int(
            resolved_budget_bytes
            if resolved_budget_bytes is not None
            else PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        ),
    )
    lock_metadata = {
        "run_name": run_name,
        "input_path": str(Path(input_path).resolve()) if input_path is not None else None,
        "session_memory_lease_path": str(PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH.resolve()),
        **dict(metadata or {}),
    }
    session_lease = acquire_session_memory_lease(
        label=label,
        requested_budget_bytes=effective_budget_bytes,
        effective_budget_bytes=effective_budget_bytes,
        metadata=lock_metadata,
        lock_path=PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH,
    )
    try:
        lock = HeavyRunLock.acquire(
            lock_path=PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH,
            label=label,
            metadata=lock_metadata,
        )
    except Exception:
        session_lease.release()
        raise
    try:
        resolved_rss_log_path = (
            Path(rss_log_path).resolve()
            if rss_log_path is not None
            else memory_dir / f"{run_name}_rss_latest.jsonl"
        )
        resolved_summary_path = (
            Path(summary_path).resolve()
            if summary_path is not None
            else memory_dir / f"{run_name}_memory_latest.json"
        )
        resolved_rss_log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
        guard_kwargs: dict[str, Any] = {
            "log_path": resolved_rss_log_path,
            "label": label,
            "budget_bytes": effective_budget_bytes,
        }
        if soft_limit_bytes is not None:
            guard_kwargs["soft_limit_bytes"] = max(1, int(soft_limit_bytes))
        if hard_limit_bytes is not None:
            guard_kwargs["hard_limit_bytes"] = max(1, int(hard_limit_bytes))
        guard = RSSGuard(**guard_kwargs)
        return PortfolioMemoryGuard(
            run_name=run_name,
            label=label,
            output_dir=resolved_output_dir,
            rss_log_path=resolved_rss_log_path,
            summary_path=resolved_summary_path,
            lock=lock,
            guard=guard,
            session_lease=session_lease,
        )
    except Exception:
        lock.release()
        session_lease.release()
        raise


__all__ = [
    "FOLLOWUP_ROOT",
    "MEMORY_GUARD_DIRNAME",
    "OOS_START",
    "OOS_START_DATE",
    "PORTFOLIO_CURRENT_OPTIMIZATION",
    "PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES",
    "PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH",
    "PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH",
    "PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE",
    "PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE",
    "REPORT_ROOT",
    "TRAIN_END_EXCLUSIVE",
    "TRAIN_START",
    "TRAIN_START_DATE",
    "VAL_END_EXCLUSIVE",
    "VAL_START",
    "VAL_START_DATE",
    "PortfolioMemoryGuard",
    "acquire_portfolio_memory_guard",
    "memory_policy_payload",
    "portfolio_followup_default_budget_bytes",
    "resolve_current_optimization_path",
    "resolve_followup_artifact_path",
    "resolve_incumbent_bundle_path",
    "split_dates",
    "split_for_date",
    "split_for_day_key",
    "split_windows",
]
