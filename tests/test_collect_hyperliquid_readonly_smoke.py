from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import collect_hyperliquid_readonly as mod

OOS_START_MS = int(datetime(2026, 5, 1, tzinfo=UTC).timestamp() * 1000)


class _DummyClient:
    def meta_and_asset_contexts(self):
        return [
            {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
            [
                {
                    "funding": "0.0001",
                    "openInterest": "100",
                    "oraclePx": "50000",
                    "markPx": "50100",
                },
                {
                    "funding": "0.0002",
                    "openInterest": "200",
                    "oraclePx": "50000",
                    "markPx": "49000",
                },
            ],
        ]

    def funding_history(self, *, coin: str, start_time_ms: int, end_time_ms: int):
        del start_time_ms, end_time_ms
        if coin == "BTC":
            return [
                {"coin": "BTC", "time": OOS_START_MS, "fundingRate": "0.001"},
                {
                    "coin": "BTC",
                    "time": OOS_START_MS,
                    "fundingRate": "0.001",
                },  # duplicated timestamp
                {"coin": "BTC", "time": OOS_START_MS + 3600_000, "fundingRate": "0.0015"},
            ]
        return [
            {"coin": "ETH", "time": OOS_START_MS + 1800_000, "fundingRate": "0.002"},
            {"coin": "ETH", "time": OOS_START_MS + 1900_000, "fundingRate": "bad"},
        ]

    def candle_snapshot(self, *, coin: str, interval: str, start_time_ms: int, end_time_ms: int):
        del interval, start_time_ms, end_time_ms
        return [
            {
                "t": 1000,
                "T": 1000 + 3600_000 - 1,
                "s": coin,
                "i": "1h",
                "o": "100",
                "h": "101",
                "l": "99",
                "c": "100.5",
                "v": "123",
            }
        ]


def test_collect_hyperliquid_builds_lightweight_artifacts_with_deterministic_payload(
    monkeypatch, tmp_path: Path
) -> None:
    """Tiny deterministic smoke for collect_hyperliquid() with fully mocked I/O."""
    monkeypatch.setattr(mod, "HyperliquidInfoClient", _DummyClient)

    upsert_calls = 0

    def _fake_upsert(*, rows: list[dict], **_: object) -> int:
        nonlocal upsert_calls
        upsert_calls += len(rows)
        return len(rows)

    monkeypatch.setattr(mod, "upsert_futures_feature_points_rows", _fake_upsert)

    result = mod.collect_hyperliquid(
        market_root=tmp_path / "market",
        output_dir=tmp_path / "out",
        symbols=["BTC/USDT", "ETH/USDT"],
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 2),
        interval="1h",
        throttle_seconds=0.0,
        write_feature_points=False,
    )

    payload = result["payload"]
    paths = result["paths"]
    assert payload["symbols"][0]["symbol"] == "BTC/USDT"
    assert payload["symbols"][1]["symbol"] == "ETH/USDT"
    assert payload["symbols"][0]["funding_rows"] == 2  # deduped duplicate timestamp
    assert payload["symbols"][1]["funding_rows"] == 1
    assert payload["symbols"][0]["funding_split_counts"]["oos"] > 0
    assert payload["symbols"][1]["funding_split_counts"]["oos"] > 0
    assert "train" in payload["symbols"][0]["candle_snapshot_coverage"]
    assert payload["upserted_feature_point_rows"] == 0
    assert upsert_calls == 0
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()


def test_collect_hyperliquid_funding_window_is_deduped() -> None:
    """_collect_funding_pages merges duplicate funding timestamps from paged API rows."""
    client = _DummyClient()
    rows, pages = mod._collect_funding_pages(
        client=client,
        coin="BTC",
        start_time_ms=1000,
        end_time_ms=1001,
        throttle_seconds=0.0,
    )
    assert [row["timestamp_ms"] for row in rows] == [OOS_START_MS, OOS_START_MS + 3600_000]
    assert [row["funding_rate"] for row in rows] == [0.001, 0.0015]
    assert pages[0]["row_count"] == 3
