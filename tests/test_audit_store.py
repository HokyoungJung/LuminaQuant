import os
import tempfile
import unittest

from lumina_quant.events import OrderEvent, FillEvent
from lumina_quant.utils.audit_store import AuditStore


class TestAuditStore(unittest.TestCase):
    def test_insert_order_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "audit.db")
            store = AuditStore(db_path)
            run_id = store.start_run("live", {"symbols": ["BTC/USDT"]})

            order = OrderEvent("BTC/USDT", "MKT", 0.01, "BUY", client_order_id="cid-1")
            fill = FillEvent(
                timeindex="2026-01-01T00:00:00",
                symbol="BTC/USDT",
                exchange="BINANCE",
                quantity=0.01,
                direction="BUY",
                fill_cost=1.0,
                commission=0.001,
                client_order_id="cid-1",
                order_id="oid-1",
                status="FILLED",
            )
            store.log_order(run_id, order, status="NEW", exchange_order_id="oid-1")
            store.log_fill(run_id, fill)
            store.log_equity(run_id, "2026-01-01T00:00:00", 10000.0, 9999.0)
            store.end_run(run_id, status="COMPLETED")
            store.close()

            self.assertTrue(os.path.exists(db_path))


if __name__ == "__main__":
    unittest.main()
