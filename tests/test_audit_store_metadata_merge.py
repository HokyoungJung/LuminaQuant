import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from lumina_quant.utils.audit_store import AuditStore


class TestAuditStoreMetadataMerge(unittest.TestCase):
    def test_end_run_merges_metadata_with_existing_run_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "audit.db"
            store = AuditStore(str(db_path))
            run_id = store.start_run(
                mode="backtest",
                metadata={"strategy": "RsiStrategy", "symbols": ["BTC/USDT"]},
                run_id="merge-test-run",
            )
            store.end_run(run_id, status="COMPLETED", metadata={"final_equity": 10042.5})
            store.close()

            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute("SELECT metadata FROM runs WHERE run_id=?", (run_id,)).fetchone()
                self.assertIsNotNone(row)
                payload = json.loads(row[0]) if row and row[0] else {}
                self.assertEqual(payload.get("strategy"), "RsiStrategy")
                self.assertEqual(payload.get("symbols"), ["BTC/USDT"])
                self.assertEqual(payload.get("final_equity"), 10042.5)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
