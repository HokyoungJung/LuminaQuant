"""Gate for AC7: RuntimeConfig.memory.cap_gb is actually ENFORCED.

The audit found cap_gb was a dead knob — defined in schema, round-tripped
through YAML, consumed nowhere. These tests prove the enforcement teeth:
changing the config cap changes the enforced ceiling, and an exceeded cap errors
clearly instead of risking an OOM kill.
"""

from __future__ import annotations

import pytest

from lumina_quant.configuration.schema import MemoryConfig, RuntimeConfig
from lumina_quant.core.memory_budget import (
    ExecutionMemoryPolicy,
    MemoryCapExceededError,
    current_rss_bytes,
    enforce_memory_cap,
    enforce_runtime_memory_cap,
)


def test_current_rss_is_measurable():
    rss = current_rss_bytes()
    assert rss is not None and rss > 0


def test_enforce_raises_when_rss_exceeds_cap():
    # A 1 MiB cap is below any live Python process RSS → must error clearly.
    with pytest.raises(MemoryCapExceededError) as exc:
        enforce_memory_cap(0.001, label="unit")
    assert "memory.cap_gb" in str(exc.value)


def test_enforce_passes_under_generous_cap():
    # A 4 TiB cap is far above any test process → returns the measured RSS.
    rss = enforce_memory_cap(4096.0, label="unit")
    assert rss is not None and rss > 0


def test_enforce_is_noop_for_invalid_cap():
    assert enforce_memory_cap(None) is None
    assert enforce_memory_cap(0.0) is None
    assert enforce_memory_cap(float("inf")) is None


def test_changing_config_changes_enforced_cap():
    """The defining behaviour the audit said was missing."""
    tiny = RuntimeConfig(memory=MemoryConfig(cap_gb=0.001))
    huge = RuntimeConfig(memory=MemoryConfig(cap_gb=4096.0))

    with pytest.raises(MemoryCapExceededError):
        enforce_runtime_memory_cap(tiny, label="engine start")
    # Same code path, larger cap from config → no error.
    assert enforce_runtime_memory_cap(huge, label="engine start") is not None


def test_policy_from_runtime_tracks_cap():
    policy = ExecutionMemoryPolicy.from_runtime(RuntimeConfig(memory=MemoryConfig(cap_gb=16.0)))
    assert policy.total_memory_cap_gib == 16.0
    # Ratio preserved (heavy cap scales with the total).
    base = ExecutionMemoryPolicy()
    assert policy.heavy_run_cap_gib == pytest.approx(
        16.0 * base.heavy_run_cap_gib / base.total_memory_cap_gib
    )
