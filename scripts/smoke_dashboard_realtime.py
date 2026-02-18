"""Smoke test for dashboard realtime data progression."""

from __future__ import annotations

import argparse
import sqlite3
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for realtime smoke check."""
    parser = argparse.ArgumentParser(
        description="Validate realtime equity row growth for a dashboard-monitored run.",
    )
    parser.add_argument("--db-path", default="data/lumina_quant.db", help="SQLite DB path.")
    parser.add_argument("--run-id", default="", help="Specific run_id to monitor.")
    parser.add_argument(
        "--require-running",
        action="store_true",
        help="Require selected run status=RUNNING when run-id is not provided.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help="Maximum wait time for row growth.",
    )
    parser.add_argument(
        "--poll-sec",
        type=int,
        default=2,
        help="Polling interval while waiting for new rows.",
    )
    parser.add_argument(
        "--min-row-increase",
        type=int,
        default=1,
        help="Minimum increase in equity rows to pass.",
    )
    parser.add_argument(
        "--dashboard-url",
        default="",
        help="Optional dashboard URL to check reachability first.",
    )
    parser.add_argument(
        "--http-timeout-sec",
        type=float,
        default=5.0,
        help="HTTP timeout for dashboard URL check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve run and print baseline row count without waiting for growth.",
    )
    return parser.parse_args()


def check_dashboard_url(url: str, timeout_sec: float) -> None:
    """Ensure dashboard endpoint is reachable when URL is supplied."""
    if not url:
        return
    req = Request(url=url, method="GET")
    try:
        with urlopen(req, timeout=float(timeout_sec)) as response:
            status_code = int(getattr(response, "status", 200))
            if status_code >= 400:
                raise RuntimeError(f"Dashboard HTTP status={status_code} for {url}")
    except URLError as exc:
        raise RuntimeError(f"Dashboard reachability check failed for {url}: {exc}") from exc


def fetch_target_run_id(conn: sqlite3.Connection, run_id: str, require_running: bool) -> str:
    """Return target run id either from input or latest matching run."""
    if run_id:
        return run_id

    if require_running:
        row = conn.execute(
            """
            SELECT run_id
            FROM runs
            WHERE UPPER(COALESCE(status, '')) = 'RUNNING'
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT run_id
            FROM runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None or row[0] is None:
        mode = "RUNNING" if require_running else "latest"
        raise RuntimeError(f"No {mode} run_id found in runs table")
    return str(row[0])


def count_equity_rows(conn: sqlite3.Connection, run_id: str) -> int:
    """Return current number of equity rows for run_id."""
    row = conn.execute(
        "SELECT COUNT(*) FROM equity WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def main() -> int:
    """Execute realtime smoke validation and return process exit code."""
    args = parse_args()
    timeout_sec = max(1, int(args.timeout_sec))
    poll_sec = max(1, int(args.poll_sec))
    min_growth = max(1, int(args.min_row_increase))

    check_dashboard_url(args.dashboard_url, args.http_timeout_sec)

    conn = sqlite3.connect(args.db_path)
    try:
        run_id = fetch_target_run_id(conn, args.run_id, bool(args.require_running))
        start_rows = count_equity_rows(conn, run_id)

        print(f"Target run_id: {run_id}")
        print(f"Start equity rows: {start_rows}")
        if args.dry_run:
            print("DRY RUN: skipping growth wait loop.")
            return 0

        print(f"Waiting up to {timeout_sec}s for +{min_growth} equity row(s) (poll={poll_sec}s)...")

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            time.sleep(poll_sec)
            current_rows = count_equity_rows(conn, run_id)
            growth = current_rows - start_rows
            if growth >= min_growth:
                print(
                    f"PASS: equity rows increased by {growth} "
                    f"(from {start_rows} to {current_rows})."
                )
                return 0

        final_rows = count_equity_rows(conn, run_id)
        growth = final_rows - start_rows
        print(f"FAIL: expected +{min_growth} rows within {timeout_sec}s, observed +{growth} rows.")
        return 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
