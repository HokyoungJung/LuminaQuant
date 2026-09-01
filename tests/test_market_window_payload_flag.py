from __future__ import annotations

import json
from pathlib import Path

from lumina_quant.core.market_window_contract import build_market_window_event


def test_market_window_payload_flag_off_preserves_legacy_metadata_behavior(tmp_path):
    metrics_path = tmp_path / "metrics.ndjson"

    event = build_market_window_event(
        time=1_700_000_000_000,
        window_seconds=20,
        bars_1s={"BTC/USDT": ((1_700_000_000_000, 1.0, 1.0, 1.0, 1.0, 1.0),)},
        event_time_watermark_ms=1_700_000_000_000,
        commit_id="commit-1",
        lag_ms=42,
        is_stale=True,
        parity_v2_enabled=False,
        emit_metrics=True,
        metrics_log_path=str(metrics_path),
    )

    assert event.commit_id == "commit-1"
    assert event.lag_ms == 42
    assert event.is_stale is True
    assert event.event_time_watermark_ms == 1_700_000_000_000
    assert metrics_path.exists() is False


def test_market_window_payload_flag_on_emits_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.ndjson"

    event = build_market_window_event(
        time=1_700_000_000_000,
        window_seconds=20,
        bars_1s={"BTC/USDT": ((1_700_000_000_000, 1.0, 1.0, 1.0, 1.0, 1.0),)},
        event_time_watermark_ms=1_700_000_000_000,
        commit_id="commit-1",
        lag_ms=42,
        is_stale=True,
        parity_v2_enabled=True,
        emit_metrics=True,
        metrics_log_path=str(metrics_path),
    )

    assert event.commit_id == "commit-1"
    assert event.lag_ms == 42
    assert event.is_stale is True
    assert event.event_time_watermark_ms == 1_700_000_000_000

    rows = [
        json.loads(line)
        for line in Path(metrics_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["parity_v2_enabled"] is True
    assert rows[0]["payload_bytes"] > 0
