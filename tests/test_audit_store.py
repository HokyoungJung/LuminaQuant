import os
import sqlite3
import tempfile
import unittest

from lumina_quant.events import FillEvent, OrderEvent
from lumina_quant.utils.audit_store import AuditStore


class TestAuditStore(unittest.TestCase):
    def test_start_run_with_external_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "audit.db")
            store = AuditStore(db_path)
            expected_run_id = "external-run-123"
            run_id = store.start_run("backtest", {"source": "dashboard"}, run_id=expected_run_id)
            self.assertEqual(run_id, expected_run_id)
            store.end_run(run_id, status="COMPLETED")
            store.close()

            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT run_id, mode, status FROM runs WHERE run_id=?", (expected_run_id,)
                )
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], expected_run_id)
                self.assertEqual(row[1], "backtest")
                self.assertEqual(row[2], "COMPLETED")
            finally:
                conn.close()

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
            store.log_order_state(
                run_id,
                {
                    "state": "OPEN",
                    "symbol": "BTC/USDT",
                    "client_order_id": "cid-1",
                    "order_id": "oid-1",
                    "metadata": {"reason": "accepted"},
                },
            )
            store.log_order_reconciliation(
                run_id,
                {
                    "order_id": "oid-1",
                    "symbol": "BTC/USDT",
                    "client_order_id": "cid-1",
                    "local_state": "OPEN",
                    "exchange_state": "PARTIAL",
                    "local_filled": 0.0,
                    "exchange_filled": 0.005,
                    "reason": "STATE_CHANGE",
                    "metadata": {"delta": 0.005},
                },
            )
            store.log_fill(run_id, fill)
            store.log_equity(run_id, "2026-01-01T00:00:00", 10000.0, 9999.0)
            store.end_run(run_id, status="COMPLETED")
            store.close()

            self.assertTrue(os.path.exists(db_path))
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM order_state_events")
                state_rows = int(cur.fetchone()[0])
                self.assertEqual(state_rows, 1)

                cur.execute("SELECT COUNT(*) FROM order_reconciliation_events")
                reconciliation_rows = int(cur.fetchone()[0])
                self.assertEqual(reconciliation_rows, 1)

                cur.execute("SELECT status FROM orders WHERE client_order_id='cid-1'")
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "OPEN")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
