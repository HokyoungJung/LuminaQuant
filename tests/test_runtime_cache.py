from __future__ import annotations

from lumina_quant.runtime_cache import RuntimeCache


def test_runtime_cache_snapshot_restore_roundtrip():
    cache = RuntimeCache()
    cache.update_market("BTC/USDT", {"close": 100.0, "timestamp_ns": 1})
    cache.update_positions({"BTC/USDT": 0.5})
    cache.update_position_legs({"BTC/USDT": {"LONG": 0.7, "SHORT": 0.2}})
    cache.update_account({"cash": 1000.0})
    cache.update_order_state("OID-1", {"state": "OPEN", "symbol": "BTC/USDT"})

    snap = cache.snapshot()

    restored = RuntimeCache()
    restored.restore(snap)
    assert restored.latest_market["BTC/USDT"]["close"] == 100.0
    assert restored.positions["BTC/USDT"] == 0.5
    assert restored.position_legs["BTC/USDT"]["LONG"] == 0.7
    assert restored.position_legs["BTC/USDT"]["SHORT"] == 0.2
    assert restored.account["cash"] == 1000.0
    assert "OID-1" in restored.open_orders


def test_runtime_cache_terminal_order_state_cleanup():
    cache = RuntimeCache()
    cache.update_order_state("OID-2", {"state": "OPEN"})
    assert "OID-2" in cache.open_orders
    cache.update_order_state("OID-2", {"state": "FILLED"})
    assert "OID-2" not in cache.open_orders
