from __future__ import annotations

from lumina_quant.core.market_window_contract import (
    build_market_window_event,
    market_window_event_payload,
)


def test_market_window_schema_parity_contract_fields_and_types():
    event = build_market_window_event(
        time=1_700_000_000_000,
        window_seconds=20,
        bars_1s={
            "BTC/USDT": (
                (1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 100.0),
                (1_700_000_001_000, 1.1, 2.1, 0.6, 1.6, 110.0),
            )
        },
        event_time_watermark_ms=1_700_000_001_000,
        commit_id="commit-1",
        lag_ms=25,
        is_stale=False,
        timestamp_ns=1234,
        sequence=77,
        parity_v2_enabled=True,
    )
    payload = market_window_event_payload(event)

    assert set(payload.keys()) == {
        "time",
        "window_seconds",
        "bars_1s",
        "event_time_watermark_ms",
        "commit_id",
        "lag_ms",
        "is_stale",
        "timestamp_ns",
        "sequence",
        "type",
    }
    assert isinstance(payload["time"], int)
    assert isinstance(payload["window_seconds"], int)
    assert isinstance(payload["bars_1s"], dict)
    assert isinstance(payload["bars_1s"]["BTC/USDT"], tuple)
    assert isinstance(payload["event_time_watermark_ms"], int)
    assert isinstance(payload["commit_id"], str)
    assert isinstance(payload["lag_ms"], int)
    assert isinstance(payload["is_stale"], bool)
    assert isinstance(payload["timestamp_ns"], int)
    assert isinstance(payload["sequence"], int)
    assert payload["type"] == "MARKET_WINDOW"
