import os
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from lumina_quant.utils.audit_store import AuditStore
from scripts.cleanup_ghost_runs import cleanup_runs, cleanup_workflow_jobs, extract_active_run_ids


def _iso(dt):
    return dt.astimezone(UTC).isoformat()


class TestCleanupGhostRuns(unittest.TestCase):
    def test_extract_active_run_ids(self):
        rows = [
            {
                "pid": 101,
                "name": "python.exe",
                "command": "python run_live.py --run-id live-run-001 --stop-file logs/control/x.stop",
            },
            {
                "pid": 102,
                "name": "python.exe",
                "command": "python run_backtest.py --run-id=bt_run_123",
            },
            {
                "pid": 103,
                "name": "python.exe",
                "command": "python optimize.py",
            },
        ]
        run_ids = extract_active_run_ids(rows)
        self.assertEqual(run_ids, {"live-run-001", "bt_run_123"})

    def test_cleanup_runs_closes_stale_and_keeps_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "audit.db")
            store = AuditStore(db_path)
            stale_id = store.start_run("live", {"name": "stale"}, run_id="stale-run")
            active_id = store.start_run("live", {"name": "active"}, run_id="active-run")
            store.close()

            old_started = _iso(datetime.now(UTC) - timedelta(hours=2))
            conn = sqlite3.connect(db_path)
            conn.execute("UPDATE runs SET started_at=? WHERE run_id=?", (old_started, stale_id))
            conn.execute("UPDATE runs SET started_at=? WHERE run_id=?", (old_started, active_id))
            conn.commit()
            conn.row_factory = sqlite3.Row

            report = cleanup_runs(
                conn,
                now=datetime.now(UTC),
                stale_sec=300,
                startup_grace_sec=90,
                close_status="STOPPED",
                active_run_ids={active_id},
                dry_run=False,
            )
            conn.commit()

            stale_status = conn.execute(
                "SELECT status, ended_at FROM runs WHERE run_id=?", (stale_id,)
            ).fetchone()
            active_status = conn.execute(
                "SELECT status FROM runs WHERE run_id=?", (active_id,)
            ).fetchone()
            conn.close()

            self.assertEqual(stale_status[0], "STOPPED")
            self.assertIsNotNone(stale_status[1])
            self.assertEqual(active_status[0], "RUNNING")
            self.assertEqual(len(report["closed"]), 1)
            self.assertEqual(report["closed"][0]["run_id"], stale_id)

    def test_cleanup_runs_dry_run_no_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "audit.db")
            store = AuditStore(db_path)
            run_id = store.start_run("live", {}, run_id="dry-run-target")
            store.close()

            old_started = _iso(datetime.now(UTC) - timedelta(hours=1))
            conn = sqlite3.connect(db_path)
            conn.execute("UPDATE runs SET started_at=? WHERE run_id=?", (old_started, run_id))
            conn.commit()
            conn.row_factory = sqlite3.Row

            report = cleanup_runs(
                conn,
                now=datetime.now(UTC),
                stale_sec=120,
                startup_grace_sec=60,
                close_status="STOPPED",
                active_run_ids=set(),
                dry_run=True,
            )

            status = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
            conn.close()

            self.assertEqual(status, "RUNNING")
            self.assertEqual(len(report["closed"]), 1)
            self.assertEqual(report["closed"][0]["run_id"], run_id)

    def test_cleanup_workflow_jobs_marks_exited(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "audit.db")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE workflow_jobs (
                    job_id TEXT PRIMARY KEY,
                    workflow TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    started_at TEXT,
                    last_updated TEXT,
                    ended_at TEXT,
                    exit_code INTEGER,
                    run_id TEXT
                )
                """
            )
            now_iso = _iso(datetime.now(UTC) - timedelta(minutes=10))
            conn.execute(
                """
                INSERT INTO workflow_jobs(job_id, workflow, status, pid, started_at, last_updated, run_id)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                ("job-1", "live", "RUNNING", 999999, now_iso, now_iso, "run-1"),
            )
            conn.commit()

            with patch("scripts.cleanup_ghost_runs.is_process_running", return_value=False):
                report = cleanup_workflow_jobs(
                    conn,
                    now=datetime.now(UTC),
                    dry_run=False,
                    force_kill_stop_requested_after_sec=0,
                )
            conn.commit()
            status = conn.execute(
                "SELECT status FROM workflow_jobs WHERE job_id='job-1'"
            ).fetchone()[0]
            conn.close()

            self.assertEqual(status, "EXITED")
            self.assertEqual(len(report["updated"]), 1)


if __name__ == "__main__":
    unittest.main()
