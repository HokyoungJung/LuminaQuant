from __future__ import annotations

from lumina_quant.dashboard.risk_health_service import empty_risk_health_payload


def test_empty_risk_health_payload_tracks_reason() -> None:
    payload = empty_risk_health_payload(reason="missing_dsn")

    assert payload["status"] == "missing_dsn"
    assert payload["risk_events"] == []
    assert payload["heartbeats"] == []
    assert payload["order_states"] == []
