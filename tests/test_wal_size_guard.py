from __future__ import annotations

import json
from pathlib import Path

from lumina_quant.storage.parquet import ParquetMarketDataRepository


def test_wal_size_guard_warns_and_marks_required_when_auto_compact_disabled(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    exchange = "binance"
    symbol = "BTC/USDT"

    wal_path = repo._wal_path(exchange=exchange, symbol=symbol)
    wal_path.parent.mkdir(parents=True, exist_ok=True)
    wal_path.write_bytes(b"0123456789")

    repo._resolve_wal_controls = lambda: (8, False, 0)
    called = {"compact": 0}
    repo.compact_wal_to_monthly_parquet = (  # type: ignore[method-assign]
        lambda **_kwargs: called.__setitem__("compact", called["compact"] + 1)
    )

    repo._enforce_wal_growth_controls(exchange=exchange, symbol=symbol)

    meta_path = repo._meta_path(exchange=exchange, symbol=symbol)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert called["compact"] == 0
    assert payload["wal_compaction_required"] is True


def test_wal_size_guard_triggers_compaction_when_enabled(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    exchange = "binance"
    symbol = "ETH/USDT"

    wal_path = repo._wal_path(exchange=exchange, symbol=symbol)
    wal_path.parent.mkdir(parents=True, exist_ok=True)
    wal_path.write_bytes(b"0123456789")

    repo._resolve_wal_controls = lambda: (8, True, 0)
    called = {"compact": 0}

    def _compact(**_kwargs):
        called["compact"] += 1

    repo.compact_wal_to_monthly_parquet = _compact  # type: ignore[method-assign]

    repo._enforce_wal_growth_controls(exchange=exchange, symbol=symbol)

    assert called["compact"] == 1
