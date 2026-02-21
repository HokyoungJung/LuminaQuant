"""Clean stale ghost runs/workflow jobs from LuminaQuant SQLite audit DB.

This tool is designed for production operations where crashed or orphaned
processes leave `runs.status='RUNNING'` forever.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
from datetime import UTC, datetime
from typing import Any

DEFAULT_DB_PATH = "data/lq_audit.sqlite3"
DEFAULT_STALE_SEC = 300
DEFAULT_STARTUP_GRACE_SEC = 90
RUN_ID_PATTERN = re.compile(r"--run-id(?:=|\s+)([A-Za-z0-9_-]+)")


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat()


def parse_dt(value: Any) -> datetime | None:
    if not value:
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


def process_inventory() -> list[dict[str, Any]]:
    if os.name == "nt":
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$p=Get-CimInstance Win32_Process; "
                "$p | Where-Object { $_.Name -match 'python|streamlit' } "
                "| Select-Object Name,ProcessId,CommandLine | ConvertTo-Json -Depth 3"
            ),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        raw = json.loads(proc.stdout)
        rows = raw if isinstance(raw, list) else [raw]
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "pid": int(row.get("ProcessId") or 0),
                    "name": str(row.get("Name") or ""),
                    "command": str(row.get("CommandLine") or ""),
                }
            )
        return out

    proc = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    if proc.returncode != 0:
        return []
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            continue
        pid = int(parts[0])
        name = parts[1]
        cmdline = parts[2] if len(parts) > 2 else ""
        if "python" in name.lower() or "streamlit" in name.lower() or "run_live" in cmdline:
            out.append({"pid": pid, "name": name, "command": cmdline})
    return out


def extract_active_run_ids(process_rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in process_rows:
        cmd = str(row.get("command") or "")
        for match in RUN_ID_PATTERN.finditer(cmd):
            out.add(match.group(1))
    return out


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
            check=False,
        )
        output = proc.stdout.decode("utf-8", errors="ignore").upper()
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
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        detail = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, detail.strip()

    try:
        os.kill(pid_int, 15)
    except OSError as exc:
        return False, str(exc)
    return True, "terminated"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _parse_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def classify_running_run(
    row: sqlite3.Row,
    *,
    now: datetime,
    stale_sec: int,
    startup_grace_sec: int,
    active_run_ids: set[str],
) -> tuple[str, float | None]:
    run_id = str(row["run_id"])
    if run_id in active_run_ids:
        return "ACTIVE_PROCESS", None

    hb_dt = parse_dt(row["last_heartbeat_at"])
    eq_dt = parse_dt(row["last_equity_at"])
    latest = hb_dt
    if eq_dt and (latest is None or eq_dt > latest):
        latest = eq_dt

    if latest is not None:
        age_sec = (now - latest).total_seconds()
        if age_sec > stale_sec:
            return "STALE_TELEMETRY", age_sec
        return "HEALTHY", age_sec

    started_dt = parse_dt(row["started_at"])
    if started_dt is None:
        return "STALE_NO_TELEMETRY", None
    age_sec = (now - started_dt).total_seconds()
    if age_sec > startup_grace_sec:
        return "STALE_NO_TELEMETRY", age_sec
    return "STARTUP_GRACE", age_sec


def cleanup_runs(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    stale_sec: int,
    startup_grace_sec: int,
    close_status: str,
    active_run_ids: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT
            r.run_id,
            r.mode,
            r.status,
            r.started_at,
            r.metadata,
            (SELECT MAX(heartbeat_time) FROM heartbeats h WHERE h.run_id = r.run_id) AS last_heartbeat_at,
            (SELECT MAX(timeindex) FROM equity e WHERE e.run_id = r.run_id) AS last_equity_at,
            (SELECT COUNT(*) FROM heartbeats h WHERE h.run_id = r.run_id) AS hb_count,
            (SELECT COUNT(*) FROM equity e WHERE e.run_id = r.run_id) AS eq_count,
            (SELECT COUNT(*) FROM fills f WHERE f.run_id = r.run_id) AS fill_count
        FROM runs r
        WHERE UPPER(COALESCE(r.status, '')) = 'RUNNING'
        ORDER BY r.started_at DESC
        """
    ).fetchall()

    report: dict[str, Any] = {
        "running_total": len(rows),
        "closed": [],
        "skipped": [],
    }

    now_iso = utc_iso(now)
    for row in rows:
        reason, age_sec = classify_running_run(
            row,
            now=now,
            stale_sec=stale_sec,
            startup_grace_sec=startup_grace_sec,
            active_run_ids=active_run_ids,
        )
        run_id = str(row["run_id"])
        entry = {
            "run_id": run_id,
            "mode": row["mode"],
            "reason": reason,
            "age_sec": round(age_sec, 2) if age_sec is not None else None,
            "hb_count": int(row["hb_count"] or 0),
            "eq_count": int(row["eq_count"] or 0),
            "fill_count": int(row["fill_count"] or 0),
        }
        if reason not in {"STALE_TELEMETRY", "STALE_NO_TELEMETRY"}:
            report["skipped"].append(entry)
            continue

        report["closed"].append(entry)
        if dry_run:
            continue

        metadata = _parse_json(row["metadata"])
        history = metadata.get("ghost_cleanup_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "closed_at": now_iso,
                "reason": reason,
                "previous_status": row["status"],
                "new_status": close_status,
                "stale_sec": stale_sec,
                "startup_grace_sec": startup_grace_sec,
                "tool": "scripts/cleanup_ghost_runs.py",
            }
        )
        metadata["ghost_cleanup_history"] = history[-20:]

        conn.execute(
            """
            UPDATE runs
            SET status = ?, ended_at = ?, metadata = ?
            WHERE run_id = ? AND UPPER(COALESCE(status, '')) = 'RUNNING'
            """,
            (close_status, now_iso, json.dumps(metadata, ensure_ascii=False), run_id),
        )

    return report


def cleanup_workflow_jobs(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    dry_run: bool,
    force_kill_stop_requested_after_sec: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "table_exists": table_exists(conn, "workflow_jobs"),
        "updated": [],
        "killed": [],
    }
    if not report["table_exists"]:
        return report

    rows = conn.execute(
        """
        SELECT job_id, workflow, status, pid, started_at
        FROM workflow_jobs
        WHERE status IN ('RUNNING', 'STOP_REQUESTED')
        ORDER BY COALESCE(started_at, last_updated) DESC
        """
    ).fetchall()

    now_iso = utc_iso(now)
    for row in rows:
        job_id = str(row["job_id"])
        status = str(row["status"])
        pid = row["pid"]
        running = is_process_running(pid)

        if not running:
            next_status = "STOPPED" if status == "STOP_REQUESTED" else "EXITED"
            report["updated"].append(
                {
                    "job_id": job_id,
                    "previous_status": status,
                    "new_status": next_status,
                    "pid": pid,
                }
            )
            if not dry_run:
                conn.execute(
                    """
                    UPDATE workflow_jobs
                    SET status = ?, ended_at = COALESCE(ended_at, ?), last_updated = ?
                    WHERE job_id = ?
                    """,
                    (next_status, now_iso, now_iso, job_id),
                )
            continue

        if status != "STOP_REQUESTED" or force_kill_stop_requested_after_sec <= 0:
            continue

        started_dt = parse_dt(row["started_at"])
        if started_dt is None:
            continue
        age_sec = (now - started_dt).total_seconds()
        if age_sec <= force_kill_stop_requested_after_sec:
            continue

        ok, detail = kill_process(pid)
        report["killed"].append(
            {
                "job_id": job_id,
                "pid": pid,
                "ok": bool(ok),
                "detail": detail,
            }
        )
        if not dry_run and ok:
            conn.execute(
                """
                UPDATE workflow_jobs
                SET status = 'KILLED', ended_at = COALESCE(ended_at, ?), exit_code = -9, last_updated = ?
                WHERE job_id = ?
                """,
                (now_iso, now_iso, job_id),
            )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup stale ghost runs/workflow jobs.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path")
    parser.add_argument(
        "--stale-sec",
        type=int,
        default=DEFAULT_STALE_SEC,
        help="Telemetry staleness threshold in seconds.",
    )
    parser.add_argument(
        "--startup-grace-sec",
        type=int,
        default=DEFAULT_STARTUP_GRACE_SEC,
        help="Grace period for RUNNING rows with no telemetry yet.",
    )
    parser.add_argument(
        "--close-status",
        default="STOPPED",
        help="Status value used when closing stale RUNNING runs.",
    )
    parser.add_argument(
        "--force-kill-stop-requested-after-sec",
        type=int,
        default=0,
        help="When >0, force kill workflow_jobs stuck in STOP_REQUESTED older than this age.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Apply mutations (default is dry-run)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.db
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    now = utc_now()
    process_rows = process_inventory()
    active_run_ids = extract_active_run_ids(process_rows)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        runs_report = cleanup_runs(
            conn,
            now=now,
            stale_sec=max(1, int(args.stale_sec)),
            startup_grace_sec=max(1, int(args.startup_grace_sec)),
            close_status=str(args.close_status),
            active_run_ids=active_run_ids,
            dry_run=not bool(args.apply),
        )
        workflow_report = cleanup_workflow_jobs(
            conn,
            now=now,
            dry_run=not bool(args.apply),
            force_kill_stop_requested_after_sec=max(
                0,
                int(args.force_kill_stop_requested_after_sec),
            ),
        )
        if args.apply:
            conn.commit()
    finally:
        conn.close()

    summary = {
        "db_path": db_path,
        "mode": "apply" if args.apply else "dry_run",
        "now": utc_iso(now),
        "process_count": len(process_rows),
        "active_run_ids_from_processes": sorted(active_run_ids),
        "runs": {
            "running_total": runs_report["running_total"],
            "closed_count": len(runs_report["closed"]),
            "skipped_count": len(runs_report["skipped"]),
            "closed": runs_report["closed"],
            "skipped": runs_report["skipped"],
        },
        "workflow_jobs": {
            "table_exists": workflow_report["table_exists"],
            "updated_count": len(workflow_report["updated"]),
            "killed_count": len(workflow_report["killed"]),
            "updated": workflow_report["updated"],
            "killed": workflow_report["killed"],
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
