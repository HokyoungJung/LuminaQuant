import argparse
import os
import sqlite3
from datetime import UTC, datetime, timedelta


def parse_dt(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        return None


def query_scalar(conn, query, params=()):
    cur = conn.cursor()
    cur.execute(query, params)
    row = cur.fetchone()
    return row[0] if row else 0


def query_rows(conn, query, params=()):
    cur = conn.cursor()
    cur.execute(query, params)
    return cur.fetchall()


def calculate_uptime_hours(run_rows, since, now):
    total = timedelta(0)
    for run_id, mode, started_at, ended_at, status in run_rows:
        if mode != "live":
            continue
        st = parse_dt(started_at)
        ed = parse_dt(ended_at) or now
        if st is None:
            continue
        if ed < since or st > now:
            continue
        a = max(st, since)
        b = min(ed, now)
        if b > a:
            total += b - a
    return total.total_seconds() / 3600.0


def calculate_max_heartbeat_gap_minutes(rows):
    if len(rows) < 2:
        return None
    prev = None
    max_gap = timedelta(0)
    for (ts_raw,) in rows:
        ts = parse_dt(ts_raw)
        if ts is None:
            continue
        if prev is not None:
            gap = ts - prev
            if gap > max_gap:
                max_gap = gap
        prev = ts
    return max_gap.total_seconds() / 60.0


def build_report(
    db_path,
    days=14,
    min_uptime_hours=24 * 14,
    max_heartbeat_gap_minutes=30,
    max_reconciliation_drift=0,
):
    now = datetime.now(UTC)
    since = now - timedelta(days=days)

    conn = sqlite3.connect(db_path)
    try:
        run_rows = query_rows(
            conn,
            """
            SELECT run_id, mode, started_at, ended_at, status
            FROM runs
            WHERE started_at >= ?
            ORDER BY started_at
            """,
            (since.isoformat(),),
        )

        fills = query_scalar(
            conn,
            "SELECT COUNT(*) FROM fills WHERE fill_time >= ?",
            (since.isoformat(),),
        )
        orders = query_scalar(
            conn,
            "SELECT COUNT(*) FROM orders WHERE created_at >= ?",
            (since.isoformat(),),
        )
        heartbeats = query_scalar(
            conn,
            "SELECT COUNT(*) FROM heartbeats WHERE heartbeat_time >= ?",
            (since.isoformat(),),
        )
        risk_events = query_scalar(
            conn,
            "SELECT COUNT(*) FROM risk_events WHERE event_time >= ?",
            (since.isoformat(),),
        )
        critical_errors = query_scalar(
            conn,
            """
            SELECT COUNT(*) FROM risk_events
            WHERE event_time >= ?
              AND reason IN ('MAIN_LOOP_ERROR', 'EXCHANGE_SYNC_ERROR', 'UNHANDLED_EXCEPTION')
            """,
            (since.isoformat(),),
        )
        reconciliation_drifts = query_scalar(
            conn,
            """
            SELECT COUNT(*) FROM risk_events
            WHERE event_time >= ?
              AND reason IN ('STATE_MISMATCH', 'RECONCILIATION_DRIFT')
            """,
            (since.isoformat(),),
        )

        heartbeat_rows = query_rows(
            conn,
            """
            SELECT heartbeat_time
            FROM heartbeats
            WHERE heartbeat_time >= ?
            ORDER BY heartbeat_time
            """,
            (since.isoformat(),),
        )
        max_hb_gap = calculate_max_heartbeat_gap_minutes(heartbeat_rows)
        uptime_hours = calculate_uptime_hours(run_rows, since, now)

        checks = {
            "uptime_hours": uptime_hours >= min_uptime_hours,
            "critical_errors": critical_errors == 0,
            "risk_events": risk_events == 0,
            "reconciliation_drift": reconciliation_drifts <= max_reconciliation_drift,
            "heartbeat_gap": (max_hb_gap is not None and max_hb_gap <= max_heartbeat_gap_minutes)
            if heartbeat_rows
            else False,
        }
        passed = all(checks.values())

        lines = []
        lines.append("# LuminaQuant Soak Report")
        lines.append("")
        lines.append(f"- Generated At (UTC): {now.isoformat()}")
        lines.append(f"- Window: Last {days} days")
        lines.append(f"- Database: `{db_path}`")
        lines.append(f"- Gate Result: {'PASS' if passed else 'FAIL'}")
        lines.append("")
        lines.append("## Activity")
        lines.append(f"- Live Runs: {len([r for r in run_rows if r[1] == 'live'])}")
        lines.append(f"- Orders: {orders}")
        lines.append(f"- Fills: {fills}")
        lines.append(f"- Heartbeats: {heartbeats}")
        lines.append("")
        lines.append("## Reliability")
        lines.append(f"- Uptime Hours: {uptime_hours:.2f} (>= {min_uptime_hours})")
        lines.append(
            f"- Max Heartbeat Gap (min): {max_hb_gap if max_hb_gap is not None else 'N/A'} (<= {max_heartbeat_gap_minutes})"
        )
        lines.append(f"- Risk Events: {risk_events} (must be 0)")
        lines.append(f"- Critical Errors: {critical_errors} (must be 0)")
        lines.append(
            f"- Reconciliation Drift Events: {reconciliation_drifts} (<= {max_reconciliation_drift})"
        )
        lines.append("")
        lines.append("## Gate Checks")
        for name, ok in checks.items():
            lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
        lines.append("")
        if not passed:
            lines.append("## Action")
            lines.append("- Keep running on testnet/paper.")
            lines.append("- Do not enable real trading until all checks pass.")
        else:
            lines.append("## Action")
            lines.append("- Eligible for real-trading promotion gate review.")

        return passed, "\n".join(lines)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Generate LuminaQuant soak report.")
    parser.add_argument("--db", default="logs/lumina_quant.db", help="SQLite DB path")
    parser.add_argument("--days", type=int, default=14, help="Soak window in days")
    parser.add_argument(
        "--min-uptime-hours",
        type=float,
        default=24 * 14,
        help="Minimum required uptime hours",
    )
    parser.add_argument(
        "--max-heartbeat-gap-minutes",
        type=float,
        default=30,
        help="Maximum allowed heartbeat gap in minutes",
    )
    parser.add_argument(
        "--max-reconciliation-drift",
        type=int,
        default=0,
        help="Maximum allowed reconciliation drift events",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise FileNotFoundError(f"DB not found: {args.db}")

    passed, markdown = build_report(
        args.db,
        days=args.days,
        min_uptime_hours=args.min_uptime_hours,
        max_heartbeat_gap_minutes=args.max_heartbeat_gap_minutes,
        max_reconciliation_drift=args.max_reconciliation_drift,
    )

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join(
        "reports",
        f"soak_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(markdown)
    print(f"\nSaved: {out_path}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
