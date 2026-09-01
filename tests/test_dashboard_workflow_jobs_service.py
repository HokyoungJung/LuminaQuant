from __future__ import annotations

import pandas as pd

from lumina_quant.dashboard import workflow_jobs_service


def test_normalize_workflow_jobs_frame_handles_empty_input() -> None:
    assert workflow_jobs_service.normalize_workflow_jobs_frame(pd.DataFrame()) == []


def test_normalize_workflow_jobs_frame_serializes_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "job_id": "job-1",
                "workflow": "backtest",
                "status": "RUNNING",
                "requested_mode": "paper",
                "strategy": "RsiStrategy",
                "run_id": "run-1",
                "started_at": "2026-03-01T00:00:00Z",
                "ended_at": None,
            }
        ]
    )

    rows = workflow_jobs_service.normalize_workflow_jobs_frame(frame)

    assert rows[0]["job_id"] == "job-1"
    assert rows[0]["workflow"] == "backtest"
    assert rows[0]["started_at"].startswith("2026-03-01T00:00:00")
    assert rows[0]["ended_at"] is None


def test_load_recent_workflow_jobs_reads_from_connection_cursor() -> None:
    class _Cursor:
        description = [
            ("job_id",),
            ("workflow",),
            ("status",),
            ("requested_mode",),
            ("strategy",),
            ("run_id",),
            ("started_at",),
            ("ended_at",),
        ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                (
                    "job-2",
                    "optimize",
                    "COMPLETED",
                    "backtest",
                    "Momentum",
                    "run-2",
                    "2026-03-02T00:00:00Z",
                    "2026-03-02T01:00:00Z",
                )
            ]

    class _Connection:
        def cursor(self):
            return _Cursor()

    rows = workflow_jobs_service.load_recent_workflow_jobs(_Connection(), limit=5)

    assert rows[0]["job_id"] == "job-2"
    assert rows[0]["status"] == "COMPLETED"
