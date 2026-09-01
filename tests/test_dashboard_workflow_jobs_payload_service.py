from __future__ import annotations

import pytest

from lumina_quant.dashboard import workflow_jobs_service


def test_load_recent_workflow_jobs_payload_short_circuits_without_dsn() -> None:
    payload = workflow_jobs_service.load_recent_workflow_jobs_payload(dsn="", limit=5)

    assert payload["status"] == "missing_dsn"
    assert payload["jobs"] == []


def test_load_recent_workflow_jobs_payload_includes_as_of_on_every_status() -> None:
    # Even short-circuit payloads carry the as-of provenance timestamp the
    # dashboard context bar renders.
    payload = workflow_jobs_service.load_recent_workflow_jobs_payload(dsn="", limit=5)

    assert isinstance(payload["as_of"], str)
    # ISO-8601 with an explicit UTC offset.
    assert "T" in payload["as_of"]
    assert payload["as_of"].endswith("+00:00") or payload["as_of"].endswith("Z")


def test_load_recent_workflow_jobs_payload_ok_status_includes_as_of(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Connection:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        workflow_jobs_service, "resolve_dashboard_postgres_dsn", lambda dsn=None: "postgres://x"
    )
    monkeypatch.setattr(workflow_jobs_service, "_connect_postgres", lambda dsn: _Connection())
    monkeypatch.setattr(
        workflow_jobs_service,
        "load_recent_workflow_jobs",
        lambda conn, *, limit: [{"job_id": "job-1"}],
    )

    payload = workflow_jobs_service.load_recent_workflow_jobs_payload(limit=5)

    assert payload["status"] == "ok"
    assert payload["jobs"] == [{"job_id": "job-1"}]
    assert isinstance(payload["as_of"], str)
    assert "T" in payload["as_of"]
