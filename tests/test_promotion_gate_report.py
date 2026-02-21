from __future__ import annotations

import os
import sys
import tempfile
import textwrap

from lumina_quant.utils.audit_store import AuditStore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from generate_promotion_gate_report import (
    build_promotion_gate_report,
    parse_soak_gate,
    resolve_promotion_gate_inputs,
)


def _seed_db(db_path, *, with_failures):
    store = AuditStore(db_path)
    run_id = store.start_run("live", metadata={"symbols": ["BTC/USDT"]})
    if with_failures:
        store.log_order_state(
            run_id,
            {
                "state": "REJECTED",
                "symbol": "BTC/USDT",
                "client_order_id": "cid-1",
                "order_id": "oid-1",
                "metadata": {"reason": "risk"},
            },
        )
        store.log_order_reconciliation(
            run_id,
            {
                "order_id": "oid-1",
                "symbol": "BTC/USDT",
                "client_order_id": "cid-1",
                "local_state": "OPEN",
                "exchange_state": "OPEN",
                "local_filled": 0.0,
                "exchange_filled": 0.0,
                "reason": "POLL_ERROR",
                "metadata": {"error": "timeout"},
            },
        )
        store.log_risk_event(run_id, reason="MAIN_LOOP_ERROR", details={"error": "boom"})
    store.end_run(run_id, status="COMPLETED")
    store.close()


def test_parse_soak_gate_reads_pass_fail():
    assert parse_soak_gate("- Gate Result: PASS") is True
    assert parse_soak_gate("- Gate Result: FAIL") is False
    assert parse_soak_gate("no gate line") is None


def test_build_promotion_gate_report_passes_for_clean_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "audit.db")
        _seed_db(db_path, with_failures=False)
        passed, markdown = build_promotion_gate_report(
            db_path,
            days=14,
            soak_passed=True,
            soak_source="unit-test",
        )
        assert passed is True
        assert "Decision: PROMOTE" in markdown


def test_build_promotion_gate_report_fails_when_limits_breached():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "audit.db")
        _seed_db(db_path, with_failures=True)
        passed, markdown = build_promotion_gate_report(
            db_path,
            days=14,
            soak_passed=True,
            soak_source="unit-test",
            max_order_rejects=0,
            max_reconciliation_alerts=0,
            max_critical_errors=0,
        )
        assert passed is False
        assert "Decision: HOLD" in markdown


def test_resolve_promotion_gate_inputs_uses_strategy_profile():
    yaml_text = textwrap.dedent(
        """
        trading:
          symbols: ["BTC/USDT"]
        live:
          mode: "paper"
          exchange:
            driver: "ccxt"
            name: "binance"
            market_type: "future"
            position_mode: "HEDGE"
            margin_mode: "isolated"
            leverage: 2
        promotion_gate:
          days: 14
          max_order_rejects: 0
          max_order_timeouts: 0
          max_reconciliation_alerts: 0
          max_critical_errors: 0
          strategy_profiles:
            RsiStrategy:
              days: 21
              max_order_rejects: 2
              require_alpha_card: true
              alpha_card_path: "reports/alpha_card_rsi.md"
        """
    ).strip()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
        fp.write(yaml_text)
        path = fp.name
    try:
        resolved = resolve_promotion_gate_inputs(config_path=path, strategy_name="RsiStrategy")
        assert resolved["days"] == 21
        assert resolved["max_order_rejects"] == 2
        assert resolved["require_alpha_card"] is True
        assert resolved["alpha_card_path"] == "reports/alpha_card_rsi.md"
    finally:
        os.remove(path)
