from __future__ import annotations

from types import SimpleNamespace

import pytest

from lumina_quant.core.engine import TradingEngine
from lumina_quant.core.events import MarketEvent
from lumina_quant.event_clock import EventSequencer, assign_event_identity


def test_assign_event_identity_sets_monotonic_sequence():
    seq = EventSequencer()
    e1 = MarketEvent(1700000000000, "BTC/USDT", 1.0, 1.0, 1.0, 1.0, 1.0)
    e2 = MarketEvent(1700000001000, "BTC/USDT", 1.1, 1.1, 1.1, 1.1, 1.0)

    assign_event_identity(e1, seq)
    assign_event_identity(e2, seq)

    assert isinstance(e1.timestamp_ns, int)
    assert isinstance(e2.timestamp_ns, int)
    assert e1.sequence == 1
    assert e2.sequence == 2


def test_event_sequencer_state_roundtrip_is_exact_and_rejects_coercion() -> None:
    source = EventSequencer()
    assert source.next() == 1
    assert source.next() == 2

    snapshot = source.get_state()
    restored = EventSequencer()
    restored.set_state(snapshot)

    assert restored.get_state() == {"sequence": 2}
    assert restored.next() == 3

    invalid_states = (
        None,
        {},
        {"sequence": -1},
        {"sequence": True},
        {"sequence": 2.0},
        {"sequence": "2"},
        {"sequence": 2, "extra": 0},
    )
    for invalid in invalid_states:
        with pytest.raises(ValueError, match="invalid_event_sequencer_state"):
            restored.set_state(invalid)  # type: ignore[arg-type]
    assert restored.get_state() == {"sequence": 3}


def test_trading_engine_state_carries_sequence_and_accepts_legacy_snapshot() -> None:
    strategy = SimpleNamespace(uses_timeframe_aggregator=False)
    source = TradingEngine(None, None, strategy, None, None)
    first = SimpleNamespace(type="OTHER", time=1, timestamp_ns=None, sequence=None)
    second = SimpleNamespace(type="OTHER", time=2, timestamp_ns=None, sequence=None)
    source.process_event(first)
    source.process_event(second)

    snapshot = source.get_engine_state()
    assert snapshot["event_sequencer"] == {"sequence": 2}

    restored = TradingEngine(None, None, strategy, None, None)
    restored.set_engine_state(snapshot)
    continued = SimpleNamespace(type="OTHER", time=3, timestamp_ns=None, sequence=None)
    restored.process_event(continued)
    assert continued.sequence == 3

    restored.set_engine_state({"window_decision_last_bucket": None})
    legacy_continued = SimpleNamespace(type="OTHER", time=4, timestamp_ns=None, sequence=None)
    restored.process_event(legacy_continued)
    assert legacy_continued.sequence == 4
