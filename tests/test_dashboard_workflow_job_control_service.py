from __future__ import annotations

from lumina_quant.dashboard import workflow_jobs_service


def test_control_workflow_job_requires_dsn() -> None:
    payload = workflow_jobs_service.control_workflow_job(
        dsn="",
        job_id="job-1",
        action="stop",
    )

    assert payload["ok"] is False
    assert payload["error"] == "missing_dsn"
