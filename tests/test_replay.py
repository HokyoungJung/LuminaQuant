from __future__ import annotations

from lumina_quant.core.events import MarketEvent
from lumina_quant.replay import assert_monotonic_event_order, stable_event_sort


def test_replay_stable_event_sort_and_monotonic_assertion():
    e1 = MarketEvent(1, "BTC/USDT", 1, 1, 1, 1, 1, timestamp_ns=10, sequence=2)
    e2 = MarketEvent(1, "BTC/USDT", 1, 1, 1, 1, 1, timestamp_ns=10, sequence=1)
    e3 = MarketEvent(1, "BTC/USDT", 1, 1, 1, 1, 1, timestamp_ns=11, sequence=1)

    ordered = stable_event_sort([e1, e2, e3])
    assert [event.sequence for event in ordered] == [1, 2, 1]
    assert_monotonic_event_order([e1, e2, e3])
