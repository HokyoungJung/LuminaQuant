import argparse
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.configuration.loader import load_runtime_config


def _as_int(value, default):
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def resolve_promotion_gate_inputs(config_path="config.yaml", strategy_name=None):
    runtime = load_runtime_config(config_path=config_path)
    gate = runtime.promotion_gate
    strategy = str(strategy_name or runtime.optimization.strategy or "").strip()
    profile = dict(gate.strategy_profiles.get(strategy) or {}) if strategy else {}

    resolved: dict[str, Any] = {
        "days": _as_int(gate.days, 14),
        "max_order_rejects": _as_int(gate.max_order_rejects, 0),
        "max_order_timeouts": _as_int(gate.max_order_timeouts, 0),
        "max_reconciliation_alerts": _as_int(gate.max_reconciliation_alerts, 0),
        "max_critical_errors": _as_int(gate.max_critical_errors, 0),
        "require_alpha_card": _as_bool(gate.require_alpha_card, False),
        "alpha_card_path": None,
        "strategy_name": strategy,
        "profile_source": f"config:{config_path}",
    }

    if profile:
        resolved["days"] = _as_int(profile.get("days"), resolved["days"])
        resolved["max_order_rejects"] = _as_int(
            profile.get("max_order_rejects"),
            resolved["max_order_rejects"],
        )
        resolved["max_order_timeouts"] = _as_int(
            profile.get("max_order_timeouts"),
            resolved["max_order_timeouts"],
        )
        resolved["max_reconciliation_alerts"] = _as_int(
            profile.get("max_reconciliation_alerts"),
            resolved["max_reconciliation_alerts"],
        )
        resolved["max_critical_errors"] = _as_int(
            profile.get("max_critical_errors"),
            resolved["max_critical_errors"],
        )
        resolved["require_alpha_card"] = _as_bool(
            profile.get("require_alpha_card"),
            resolved["require_alpha_card"],
        )
        alpha_card_path = profile.get("alpha_card_path")
        if alpha_card_path:
            resolved["alpha_card_path"] = str(alpha_card_path)
        resolved["profile_source"] = f"config:{config_path}#{strategy}"

    return resolved


def query_scalar(conn, query, params=()):
    cur = conn.cursor()
    cur.execute(query, params)
    row = cur.fetchone()
    return row[0] if row else 0


def parse_soak_gate(markdown_text):
    text = str(markdown_text or "")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("- gate result:"):
            return "PASS" in line.upper()
    return None


def read_soak_report(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    parsed = parse_soak_gate(content)
    if parsed is None:
        raise ValueError(f"Could not parse gate result from soak report: {path}")
    return parsed, content


def _safe_count(conn, query, params=()):
    try:
        return int(query_scalar(conn, query, params))
    except sqlite3.OperationalError:
        return 0


def collect_metrics(db_path, days):
    now = datetime.now(UTC)
    since = now - timedelta(days=int(days))
    conn = sqlite3.connect(db_path)
    try:
        params = (since.isoformat(),)
        metrics = {
            "order_rejects": _safe_count(
                conn,
                """
                SELECT COUNT(*) FROM order_state_events
                WHERE event_time >= ? AND state = 'REJECTED'
                """,
                params,
            ),
            "order_timeouts": _safe_count(
                conn,
                """
                SELECT COUNT(*) FROM order_state_events
                WHERE event_time >= ? AND state = 'TIMEOUT'
                """,
                params,
            ),
            "reconciliation_alerts": _safe_count(
                conn,
                """
                SELECT COUNT(*) FROM order_reconciliation_events
                WHERE event_time >= ?
                  AND reason IN ('POLL_ERROR', 'MISSING_ORDER', 'STATE_CHANGE')
                """,
                params,
            ),
            "critical_errors": _safe_count(
                conn,
                """
                SELECT COUNT(*) FROM risk_events
                WHERE event_time >= ?
                  AND reason IN ('MAIN_LOOP_ERROR', 'EXCHANGE_SYNC_ERROR', 'UNHANDLED_EXCEPTION')
                """,
                params,
            ),
            "dual_leg_events": _safe_count(
                conn,
                """
                SELECT COUNT(*) FROM risk_events
                WHERE event_time >= ?
                  AND reason = 'HEDGE_DUAL_LEG_DETECTED'
                """,
                params,
            ),
        }
        return now, since, metrics
    finally:
        conn.close()


def build_promotion_gate_report(
    db_path,
    *,
    days,
    soak_passed,
    soak_source,
    strategy_name=None,
    profile_source=None,
    alpha_card_path=None,
    require_alpha_card=False,
    max_order_rejects=0,
    max_order_timeouts=0,
    max_reconciliation_alerts=0,
    max_critical_errors=0,
):
    now, since, metrics = collect_metrics(db_path=db_path, days=days)
    alpha_exists = bool(alpha_card_path and os.path.exists(alpha_card_path))

    checks = {
        "soak_gate_passed": bool(soak_passed),
        "critical_errors": int(metrics["critical_errors"]) <= int(max_critical_errors),
        "order_rejects": int(metrics["order_rejects"]) <= int(max_order_rejects),
        "order_timeouts": int(metrics["order_timeouts"]) <= int(max_order_timeouts),
        "reconciliation_alerts": int(metrics["reconciliation_alerts"])
        <= int(max_reconciliation_alerts),
        "alpha_card_present": (alpha_exists if require_alpha_card else True),
    }
    promote = all(checks.values())

    lines = [
        "# LuminaQuant Promotion Gate Report",
        "",
        f"- Generated At (UTC): {now.isoformat()}",
        f"- Window: {since.isoformat()} ~ {now.isoformat()}",
        f"- Decision: {'PROMOTE' if promote else 'HOLD'}",
        "",
        "## Inputs",
        f"- Database: `{db_path}`",
        f"- Soak Source: {soak_source}",
        f"- Soak Gate: {'PASS' if soak_passed else 'FAIL'}",
        (
            f"- Strategy Profile: {strategy_name}"
            if strategy_name
            else "- Strategy Profile: default"
        ),
        (f"- Profile Source: {profile_source}" if profile_source else "- Profile Source: cli"),
        f"- Alpha Card: `{alpha_card_path}` ({'found' if alpha_exists else 'missing'})"
        if alpha_card_path
        else "- Alpha Card: not provided",
        "",
        "## Operational Metrics",
        f"- Order Rejects: {metrics['order_rejects']} (<= {max_order_rejects})",
        f"- Order Timeouts: {metrics['order_timeouts']} (<= {max_order_timeouts})",
        "- Reconciliation Alerts: "
        f"{metrics['reconciliation_alerts']} (<= {max_reconciliation_alerts})",
        f"- Critical Errors: {metrics['critical_errors']} (<= {max_critical_errors})",
        f"- Hedge Dual-Leg Events: {metrics['dual_leg_events']} (info)",
        "",
        "## Gate Checks",
    ]
    for name, passed in checks.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")

    lines.append("")
    lines.append("## Action")
    if promote:
        lines.append("- Promotion gate passed. Eligible for real-mode enable review.")
    else:
        lines.append("- Promotion gate failed. Keep paper/testnet and remediate failed checks.")

    return promote, "\n".join(lines)


def _resolve_soak(db_path, days, soak_report_path):
    if soak_report_path:
        passed, _content = read_soak_report(soak_report_path)
        return passed, soak_report_path

    try:
        from scripts.generate_soak_report import build_report
    except Exception:
        from generate_soak_report import build_report

    passed, _content = build_report(db_path, days=days)
    return passed, "derived_from_db"


def main():
    parser = argparse.ArgumentParser(
        description="Generate promotion gate report from soak + runtime data."
    )
    parser.add_argument("--db", default="data/lq_audit.sqlite3", help="SQLite DB path")
    parser.add_argument("--config", default="config.yaml", help="Runtime config path")
    parser.add_argument("--strategy", default="", help="Strategy profile key")
    parser.add_argument("--days", type=int, default=None, help="Window size in days")
    parser.add_argument("--soak-report", default="", help="Optional soak report markdown path")
    parser.add_argument("--alpha-card", default="", help="Optional alpha card markdown path")
    parser.add_argument(
        "--require-alpha-card",
        dest="require_alpha_card",
        action="store_true",
        help="Force requiring alpha card file",
    )
    parser.add_argument(
        "--no-require-alpha-card",
        dest="require_alpha_card",
        action="store_false",
        help="Force not requiring alpha card file",
    )
    parser.set_defaults(require_alpha_card=None)
    parser.add_argument("--max-order-rejects", type=int, default=None)
    parser.add_argument("--max-order-timeouts", type=int, default=None)
    parser.add_argument("--max-reconciliation-alerts", type=int, default=None)
    parser.add_argument("--max-critical-errors", type=int, default=None)
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise FileNotFoundError(f"DB not found: {args.db}")

    resolved = resolve_promotion_gate_inputs(
        config_path=args.config,
        strategy_name=(args.strategy or None),
    )

    effective_days = _as_int(args.days, resolved["days"])
    effective_max_order_rejects = _as_int(args.max_order_rejects, resolved["max_order_rejects"])
    effective_max_order_timeouts = _as_int(args.max_order_timeouts, resolved["max_order_timeouts"])
    effective_max_reconciliation_alerts = _as_int(
        args.max_reconciliation_alerts,
        resolved["max_reconciliation_alerts"],
    )
    effective_max_critical_errors = _as_int(
        args.max_critical_errors,
        resolved["max_critical_errors"],
    )
    effective_require_alpha_card = (
        bool(args.require_alpha_card)
        if args.require_alpha_card is not None
        else bool(resolved["require_alpha_card"])
    )
    effective_alpha_card_path = (args.alpha_card or "").strip() or resolved.get("alpha_card_path")

    soak_passed, soak_source = _resolve_soak(
        db_path=args.db,
        days=effective_days,
        soak_report_path=args.soak_report,
    )
    promote, markdown = build_promotion_gate_report(
        args.db,
        days=effective_days,
        soak_passed=soak_passed,
        soak_source=soak_source,
        strategy_name=resolved.get("strategy_name") or "",
        profile_source=resolved.get("profile_source") or "",
        alpha_card_path=(effective_alpha_card_path or None),
        require_alpha_card=effective_require_alpha_card,
        max_order_rejects=effective_max_order_rejects,
        max_order_timeouts=effective_max_order_timeouts,
        max_reconciliation_alerts=effective_max_reconciliation_alerts,
        max_critical_errors=effective_max_critical_errors,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"promotion_gate_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(markdown)
    print(f"\nSaved: {out_path}")
    if not promote:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
