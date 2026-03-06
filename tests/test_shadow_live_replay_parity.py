from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.live.shadow_live_runner import ShadowLiveRunner


def test_shadow_live_runner_reports_zero_divergence_for_identical_event_order():
    events = [
        SimpleNamespace(timestamp_ns=1, sequence=1),
        SimpleNamespace(timestamp_ns=2, sequence=2),
    ]
    runner = ShadowLiveRunner()
    result = runner.run(baseline_events=events, candidate_events=list(events))
    assert result.events_processed == 2
    assert result.divergence_count == 0


def test_shadow_live_runner_reports_divergence_for_sequence_mismatch():
    baseline = [SimpleNamespace(timestamp_ns=1, sequence=1)]
    candidate = [SimpleNamespace(timestamp_ns=1, sequence=2)]
    runner = ShadowLiveRunner()
    result = runner.run(baseline_events=baseline, candidate_events=candidate)
    assert result.events_processed == 1
    assert result.divergence_count == 1
