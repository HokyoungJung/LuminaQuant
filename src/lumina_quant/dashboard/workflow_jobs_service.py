"""Workflow job payload helpers for the Next dashboard migration."""

from __future__ import annotations

from datetime import UTC, datetime
import os
import signal
import subprocess
import sys
from typing import Any

import pandas as pd

from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.postgres_state import _connect_postgres


def resolve_dashboard_postgres_dsn(dsn: str | None = None) -> str:
    return str(
        dsn
        or os.getenv("LQ_POSTGRES_DSN")
        or get_default_runtime_config().storage.postgres_dsn
        or ""
    ).strip()


def request_job_stop(stop_file: str | None, *, timestamp: str) -> bool:
    if not stop_file:
        return False
    parent = os.path.dirname(stop_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(stop_file, "w", encoding="utf-8") as handle:
        handle.write(str(timestamp))
    return True


def terminate_process(pid: object) -> tuple[bool, str]:
    try:
        resolved_pid = int(pid)
    except TypeError, ValueError:
        return False, "invalid pid"
    if resolved_pid <= 0:
        return False, "invalid pid"
    if sys.platform.startswith("win"):
        proc = subprocess.run(
            ["taskkill", "/PID", str(resolved_pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        detail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return proc.returncode == 0, detail
    try:
        os.kill(resolved_pid, signal.SIGTERM)
    except OSError as exc:
        return False, str(exc)
    return True, "terminated"


def normalize_workflow_jobs_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    normalized: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        started_at = row.get("started_at")
        ended_at = row.get("ended_at")
        normalized.append(
            {
                "job_id": str(row.get("job_id") or ""),
                "workflow": str(row.get("workflow") or ""),
                "status": str(row.get("status") or ""),
                "requested_mode": str(row.get("requested_mode") or ""),
                "strategy": str(row.get("strategy") or ""),
                "run_id": str(row.get("run_id") or ""),
                "started_at": (
                    None
                    if pd.isna(started_at)
                    else pd.to_datetime(started_at, errors="coerce", utc=True).isoformat()
                ),
                "ended_at": (
                    None
                    if pd.isna(ended_at)
                    else pd.to_datetime(ended_at, errors="coerce", utc=True).isoformat()
                ),
            }
        )
    return normalized


def load_recent_workflow_jobs(connection: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                job_id,
                workflow,
                status,
                requested_mode,
                strategy,
                run_id,
                started_at,
                ended_at
            FROM workflow_jobs
            ORDER BY COALESCE(started_at, ended_at, last_updated) DESC
            LIMIT %s
            """,
            (int(max(1, limit)),),
        )
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description or ()]
    return normalize_workflow_jobs_frame(pd.DataFrame(rows, columns=columns))


def load_recent_workflow_jobs_payload(*, dsn: str | None = None, limit: int = 10) -> dict[str, Any]:
    as_of = datetime.now(tz=UTC).isoformat()
    if dsn is not None and not str(dsn).strip():
        return {"as_of": as_of, "jobs": [], "status": "missing_dsn"}
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {"as_of": as_of, "jobs": [], "status": "missing_dsn"}

    conn = _connect_postgres(resolved_dsn)
    try:
        return {
            "as_of": as_of,
            "jobs": load_recent_workflow_jobs(conn, limit=limit),
            "status": "ok",
        }
    finally:
        conn.close()


def control_workflow_job(
    *,
    dsn: str | None,
    job_id: str,
    action: str,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {"ok": False, "error": "missing_dsn"}
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {"ok": False, "error": "missing_dsn"}

    conn = _connect_postgres(resolved_dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT job_id, pid, stop_file, status
                FROM workflow_jobs
                WHERE job_id = %s
                LIMIT 1
                """,
                (job_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return {"ok": False, "error": "job_not_found"}
        _, pid, stop_file, status = row
        normalized_action = str(action or "").strip().lower()
        if normalized_action == "stop":
            ok = request_job_stop(stop_file, timestamp=datetime.now(tz=UTC).isoformat())
            if ok:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE workflow_jobs
                        SET status = 'STOP_REQUESTED'
                        WHERE job_id = %s
                        """,
                        (job_id,),
                    )
                conn.commit()
                return {"ok": True, "action": "stop", "job_id": job_id, "previous_status": status}
            return {"ok": False, "error": "missing_stop_file", "job_id": job_id}
        if normalized_action == "kill":
            ok, detail = terminate_process(pid)
            if ok:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE workflow_jobs
                        SET status = 'KILLED', exit_code = -9
                        WHERE job_id = %s
                        """,
                        (job_id,),
                    )
                conn.commit()
                return {"ok": True, "action": "kill", "job_id": job_id, "detail": detail}
            return {"ok": False, "error": "kill_failed", "detail": detail, "job_id": job_id}
        return {"ok": False, "error": "unsupported_action", "job_id": job_id}
    finally:
        conn.close()


__all__ = [
    "control_workflow_job",
    "load_recent_workflow_jobs",
    "load_recent_workflow_jobs_payload",
    "normalize_workflow_jobs_frame",
    "resolve_dashboard_postgres_dsn",
]


def main(argv: list[str] | None = None) -> int:
    """Module-mode entry for workflow-jobs and workflow-jobs/control routes.

    ``--fn load``    → load_recent_workflow_jobs_payload (default)
    ``--fn control`` → control_workflow_job (requires --job-id and --action)
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="lumina_quant.dashboard.workflow_jobs_service",
        description="Emit workflow-jobs payload or control a job.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default and only output mode).",
    )
    parser.add_argument(
        "--fn",
        choices=["load", "control"],
        default="load",
        help="Function to invoke (default: load).",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Max jobs to return (load only, default: 10)."
    )
    parser.add_argument("--job-id", default="", dest="job_id", help="Job ID for control action.")
    parser.add_argument(
        "--action", default="stop", help="Control action: stop | kill (default: stop)."
    )
    args = parser.parse_args(argv)

    if args.fn == "control":
        payload = control_workflow_job(dsn=None, job_id=args.job_id, action=args.action)
    else:
        payload = load_recent_workflow_jobs_payload(limit=args.limit)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
