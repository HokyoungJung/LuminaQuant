from __future__ import annotations

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
