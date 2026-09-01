"""Close stale RUNNING runs and reconcile orphan workflow jobs in PostgreSQL."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from datetime import UTC, datetime
from typing import Any

from lumina_quant.postgres_state import _connect_postgres

DEFAULT_STALE_SEC = 300
DEFAULT_STARTUP_GRACE_SEC = 90


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat()


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def is_process_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    if os.name == "nt":
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid_int}", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (proc.stdout or "").upper()
        return str(pid_int) in output and "NO TASKS" not in output
    try:
        os.kill(pid_int, 0)
    except OSError:
        return False
    return True


def kill_process(pid: Any) -> tuple[bool, str]:
    try:
        pid_int = int(pid)
    except Exception:
        return False, "invalid pid"
    if pid_int <= 0:
        return False, "invalid pid"
    if os.name == "nt":
        proc = subprocess.run(
            ["taskkill", "/PID", str(pid_int), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        detail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return proc.returncode == 0, detail
    try:
        os.kill(pid_int, signal.SIGTERM)
    except OSError as exc:
        return False, str(exc)
    return True, "terminated"


def cleanup_runs(
    conn: Any,
    *,
    now: datetime,
    stale_sec: int,
    startup_grace_sec: int,
    close_status: str,
    dry_run: bool,
) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                r.run_id,
                r.mode,
                r.status,
                r.started_at::text,
                r.metadata_json::text,
                (
                    SELECT MAX(h.heartbeat_time)::text
                    FROM heartbeats h
                    WHERE h.run_id = r.run_id
                ) AS last_heartbeat_at,
                (
                    SELECT MAX(e.timeindex)::text
                    FROM equity e
                    WHERE e.run_id = r.run_id
                ) AS last_equity_at
            FROM runs r
            WHERE UPPER(COALESCE(r.status, '')) = 'RUNNING'
            ORDER BY r.started_at DESC
            """
        )
        rows = cur.fetchall()

    report: dict[str, Any] = {"running_total": len(rows), "closed": [], "skipped": []}
    now_iso = utc_iso(now)
    for row in rows:
        run_id = str(row[0])
        mode = str(row[1] or "")
        started_at = parse_dt(row[3])
        last_hb = parse_dt(row[5])
        last_eq = parse_dt(row[6])
        latest = max([item for item in (last_hb, last_eq) if item is not None], default=started_at)
        age_sec = (now - latest).total_seconds() if latest else None
        stale = age_sec is None or (
            (last_hb is None and last_eq is None and age_sec > startup_grace_sec)
            or age_sec > stale_sec
        )

        entry = {
            "run_id": run_id,
            "mode": mode,
            "age_sec": round(age_sec, 2) if age_sec is not None else None,
            "reason": "STALE" if stale else "ACTIVE",
        }
        if not stale:
            report["skipped"].append(entry)
            continue

        report["closed"].append(entry)
        if dry_run:
            continue

        metadata = {}
        try:
            metadata = json.loads(str(row[4] or "{}"))
        except Exception:
            metadata = {}
        history = metadata.get("ghost_cleanup_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "closed_at": now_iso,
                "new_status": close_status,
                "stale_sec": stale_sec,
                "startup_grace_sec": startup_grace_sec,
                "tool": "scripts/cleanup_ghost_runs.py",
            }
        )
        metadata["ghost_cleanup_history"] = history[-20:]

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status = %s,
                    ended_at = %s,
                    metadata_json = %s::jsonb,
                    updated_at = %s
                WHERE run_id = %s AND UPPER(COALESCE(status, '')) = 'RUNNING'
                """,
                (close_status, now, json.dumps(metadata), now, run_id),
            )
    return report


def cleanup_workflow_jobs(
    conn: Any,
    *,
    now: datetime,
    dry_run: bool,
    force_kill_stop_requested_after_sec: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"updated": [], "killed": []}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT job_id, workflow, status, pid, COALESCE(started_at, last_updated)::text
            FROM workflow_jobs
            WHERE status IN ('RUNNING', 'STOP_REQUESTED')
            ORDER BY COALESCE(started_at, last_updated) DESC
            """
        )
        rows = cur.fetchall()

    for row in rows:
        job_id = str(row[0])
        workflow = str(row[1] or "")
        status = str(row[2] or "")
        pid = row[3]
        started = parse_dt(row[4])
        started_age = (now - started).total_seconds() if started else None
        running = is_process_running(pid)

        if not running:
            report["updated"].append({"job_id": job_id, "workflow": workflow, "status": "ORPHANED"})
            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE workflow_jobs
                        SET status = 'ORPHANED',
                            ended_at = %s,
                            exit_code = COALESCE(exit_code, -9),
                            last_updated = %s
                        WHERE job_id = %s
                        """,
                        (now, now, job_id),
                    )
            continue

        if (
            status == "STOP_REQUESTED"
            and force_kill_stop_requested_after_sec > 0
            and started_age is not None
            and started_age > force_kill_stop_requested_after_sec
        ):
            killed, detail = kill_process(pid)
            report["killed"].append(
                {
                    "job_id": job_id,
                    "workflow": workflow,
                    "pid": int(pid) if pid is not None else None,
                    "killed": bool(killed),
                    "detail": detail,
                }
            )
            if killed and not dry_run:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE workflow_jobs
                        SET status = 'KILLED',
                            ended_at = %s,
                            exit_code = COALESCE(exit_code, -15),
                            last_updated = %s
                        WHERE job_id = %s
                        """,
                        (now, now, job_id),
                    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cleanup stale RUNNING runs and orphan workflow jobs in PostgreSQL."
    )
    parser.add_argument(
        "--dsn",
        default=os.getenv("LQ_POSTGRES_DSN", ""),
        help="PostgreSQL DSN (defaults to LQ_POSTGRES_DSN).",
    )
    parser.add_argument("--stale-sec", type=int, default=DEFAULT_STALE_SEC)
    parser.add_argument("--startup-grace-sec", type=int, default=DEFAULT_STARTUP_GRACE_SEC)
    parser.add_argument("--close-status", default="STOPPED")
    parser.add_argument("--force-kill-stop-requested-after-sec", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dsn = str(args.dsn or "").strip() or str(os.getenv("LQ_POSTGRES_DSN", "")).strip()
    if not dsn:
        raise ValueError("PostgreSQL DSN is required (--dsn or LQ_POSTGRES_DSN).")

    conn = _connect_postgres(dsn)
    try:
        now = utc_now()
        dry_run = not bool(args.apply)
        runs_report = cleanup_runs(
            conn,
            now=now,
            stale_sec=max(1, int(args.stale_sec)),
            startup_grace_sec=max(1, int(args.startup_grace_sec)),
            close_status=str(args.close_status),
            dry_run=dry_run,
        )
        jobs_report = cleanup_workflow_jobs(
            conn,
            now=now,
            dry_run=dry_run,
            force_kill_stop_requested_after_sec=max(
                0, int(args.force_kill_stop_requested_after_sec)
            ),
        )
        if not dry_run:
            conn.commit()
        result = {
            "applied": not dry_run,
            "generated_at": utc_iso(now),
            "runs": runs_report,
            "workflow_jobs": jobs_report,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
