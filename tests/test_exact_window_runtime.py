from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from lumina_quant.core.memory_budget import DEFAULT_EXECUTION_MEMORY_POLICY
from lumina_quant.eval import exact_window_runtime as runtime
from lumina_quant import portfolio_split_contract as portfolio_contract


def test_rss_guard_logs_samples_and_raises_on_hard_limit(tmp_path: Path, monkeypatch):
    rss_values = iter([500, 700, 900])
    monkeypatch.setattr(runtime, "current_rss_bytes", lambda pid=None: next(rss_values))

    guard = runtime.RSSGuard(
        log_path=tmp_path / "rss.jsonl",
        budget_bytes=1_000,
        soft_limit_bytes=600,
        hard_limit_bytes=800,
    )

    guard.sample(event="start")
    guard.sample(event="mid")
    with pytest.raises(runtime.RSSLimitExceeded):
        guard.checkpoint("candidate_evaluated", {"candidate_index": 3})

    summary = guard.finalize(status="aborted_rss_guard", error="boom")
    assert summary["peak_rss_bytes"] == 900
    assert summary["soft_trigger_count"] == 2
    assert summary["hard_trigger_count"] == 1

    lines = [
        json.loads(line)
        for line in (tmp_path / "rss.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [line["event"] for line in lines] == ["start", "mid", "candidate_evaluated"]


def test_heavy_run_lock_reclaims_stale_file_and_blocks_live_owner(tmp_path: Path, monkeypatch):
    lock_path = tmp_path / "exact_window.lock"
    lock_path.write_text(
        json.dumps({"pid": 111, "run_id": "stale-run", "started_at_utc": "2026-03-09T00:00:00Z"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "_process_running", lambda pid: False)

    lock = runtime.HeavyRunLock.acquire(
        lock_path=lock_path,
        label="exact_window",
        metadata={"run_id": "fresh-run", "batch_id": "1h"},
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "fresh-run"
    lock.release()
    assert not lock_path.exists()

    lock_path.write_text(
        json.dumps({"pid": 222, "run_id": "busy-run", "started_at_utc": "2026-03-09T00:00:00Z"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "_process_running", lambda pid: int(pid or 0) == 222)

    with pytest.raises(runtime.HeavyRunActiveError) as exc:
        runtime.HeavyRunLock.acquire(
            lock_path=lock_path,
            label="exact_window",
            metadata={"run_id": "blocked-run", "batch_id": "30m"},
        )

    assert exc.value.active_payload["run_id"] == "busy-run"


def test_resolve_memory_budget_bytes_clamps_to_env_budget(monkeypatch):
    monkeypatch.setattr(runtime, "_read_cgroup_limit_bytes", lambda: 12_000)
    monkeypatch.setattr(runtime, "_read_memavailable_bytes", lambda: 10_000)
    monkeypatch.setenv("LQ_EXACT_WINDOW_BUDGET_BYTES", "4096")
    monkeypatch.delenv("LQ_EXACT_WINDOW_BUDGET_GIB", raising=False)

    assert runtime.resolve_memory_budget_bytes() == 4096


def test_resolve_memory_budget_bytes_uses_env_when_system_budget_missing(monkeypatch):
    monkeypatch.setattr(runtime, "_read_cgroup_limit_bytes", lambda: None)
    monkeypatch.setattr(runtime, "_read_memavailable_bytes", lambda: None)
    monkeypatch.delenv("LQ_EXACT_WINDOW_BUDGET_BYTES", raising=False)
    monkeypatch.setenv("LQ_EXACT_WINDOW_BUDGET_GIB", "1.5")

    assert runtime.resolve_memory_budget_bytes() == int(1.5 * 1024 * 1024 * 1024)


def test_rss_guard_uses_canonical_policy_fractions_by_default(tmp_path: Path):
    guard = runtime.RSSGuard(
        log_path=tmp_path / "rss.jsonl",
        budget_bytes=1_000,
    )

    assert guard.soft_limit_bytes == int(
        1_000 * DEFAULT_EXECUTION_MEMORY_POLICY.exact_window_soft_fraction
    )
    assert guard.hard_limit_bytes == int(
        1_000 * DEFAULT_EXECUTION_MEMORY_POLICY.exact_window_hard_fraction
    )


def test_acquire_portfolio_memory_guard_propagates_explicit_budget(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    class DummyLock:
        def release(self) -> None:
            return None

    class DummyGuard:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.budget_bytes = kwargs.get("budget_bytes")

    monkeypatch.setattr(
        portfolio_contract,
        "HeavyRunLock",
        type(
            "DummyHeavyRunLock",
            (),
            {"acquire": staticmethod(lambda **kwargs: DummyLock())},
        ),
    )
    monkeypatch.setattr(
        portfolio_contract,
        "acquire_session_memory_lease",
        lambda **kwargs: DummyLock(),
    )
    monkeypatch.setattr(portfolio_contract, "RSSGuard", DummyGuard)

    budget_bytes = 8 * 1024 * 1024 * 1024
    params = inspect.signature(portfolio_contract.acquire_portfolio_memory_guard).parameters
    candidate_kwargs = {
        "run_name": "portfolio_four_sleeve_search",
        "output_dir": tmp_path / "search",
        "input_path": tmp_path / "bundle.json",
        "budget_bytes": budget_bytes,
        "memory_budget_bytes": budget_bytes,
        "fixed_budget_bytes": budget_bytes,
    }
    guard = portfolio_contract.acquire_portfolio_memory_guard(
        **{name: value for name, value in candidate_kwargs.items() if name in params}
    )

    assert captured.get("budget_bytes") == budget_bytes
    assert getattr(guard.guard, "budget_bytes", None) == budget_bytes
