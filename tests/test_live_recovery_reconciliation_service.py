from __future__ import annotations

from lumina_quant.live.recovery_reconciliation import RecoveryReconciliationService


def test_recovery_reconciliation_service_converges_when_counts_match():
    state = {"tracked": 2, "cached": 3, "calls": 0}

    def _reconcile_positions():
        return None

    def _reconcile_orders():
        state["calls"] += 1
        if state["calls"] >= 2:
            state["cached"] = 2

    service = RecoveryReconciliationService(
        reconcile_positions=_reconcile_positions,
        reconcile_orders=_reconcile_orders,
        tracked_orders_count=lambda: state["tracked"],
        cache_open_orders_count=lambda: state["cached"],
    )

    outcome = service.startup_converge(timeout_seconds=3)
    assert outcome.converged is True
    assert outcome.tracked_orders == outcome.cache_open_orders == 2


def test_recovery_reconciliation_service_times_out_when_never_converged():
    service = RecoveryReconciliationService(
        reconcile_positions=lambda: None,
        reconcile_orders=lambda: None,
        tracked_orders_count=lambda: 5,
        cache_open_orders_count=lambda: 1,
    )

    outcome = service.startup_converge(timeout_seconds=1.2)
    assert outcome.converged is False


def test_recovery_reconciliation_service_requires_exchange_count_when_configured():
    state = {"tracked": 2, "cached": 2, "exchange": 3, "calls": 0}

    def _reconcile_orders():
        state["calls"] += 1
        if state["calls"] >= 2:
            state["exchange"] = 2

    service = RecoveryReconciliationService(
        reconcile_positions=lambda: None,
        reconcile_orders=_reconcile_orders,
        tracked_orders_count=lambda: state["tracked"],
        cache_open_orders_count=lambda: state["cached"],
        exchange_open_orders_count=lambda: state["exchange"],
    )

    outcome = service.startup_converge(timeout_seconds=3)
    assert outcome.converged is True
    assert outcome.exchange_open_orders == 2


def test_recovery_reconciliation_service_requires_snapshot_ready_when_configured():
    state = {"ready": False, "calls": 0}

    def _reconcile_orders():
        state["calls"] += 1
        if state["calls"] >= 2:
            state["ready"] = True

    service = RecoveryReconciliationService(
        reconcile_positions=lambda: None,
        reconcile_orders=_reconcile_orders,
        tracked_orders_count=lambda: 1,
        cache_open_orders_count=lambda: 1,
        exchange_open_orders_count=lambda: 1,
        exchange_snapshot_ready=lambda: state["ready"],
    )

    outcome = service.startup_converge(timeout_seconds=3)
    assert outcome.converged is True
    assert outcome.exchange_snapshot_ready is True


def test_recovery_reconciliation_service_requires_tracked_and_cache_signatures_to_match():
    state = {
        "tracked": (("ord-1", "OPEN", "0.0", "cid-1"),),
        "cached": (("ord-1", "SUBMITTED", "0.0", "cid-1"),),
        "calls": 0,
    }

    def _reconcile_orders():
        state["calls"] += 1
        if state["calls"] >= 2:
            state["cached"] = state["tracked"]

    service = RecoveryReconciliationService(
        reconcile_positions=lambda: None,
        reconcile_orders=_reconcile_orders,
        tracked_orders_count=lambda: len(state["tracked"]),
        cache_open_orders_count=lambda: len(state["cached"]),
        tracked_order_signature=lambda: state["tracked"],
        cache_open_order_signature=lambda: state["cached"],
    )

    outcome = service.startup_converge(timeout_seconds=3)
    assert outcome.converged is True
    assert outcome.tracked_cache_signature_match is True


def test_recovery_reconciliation_service_requires_exchange_signature_when_configured():
    state = {
        "tracked": (("ord-1", "OPEN", "0.0", "cid-1"),),
        "cached": (("ord-1", "OPEN", "0.0", "cid-1"),),
        "exchange": (("ord-2",),),
        "calls": 0,
    }

    def _reconcile_orders():
        state["calls"] += 1
        if state["calls"] >= 2:
            state["exchange"] = (("ord-1",),)

    service = RecoveryReconciliationService(
        reconcile_positions=lambda: None,
        reconcile_orders=_reconcile_orders,
        tracked_orders_count=lambda: len(state["tracked"]),
        cache_open_orders_count=lambda: len(state["cached"]),
        exchange_open_orders_count=lambda: len(state["exchange"]),
        tracked_order_signature=lambda: state["tracked"],
        cache_open_order_signature=lambda: state["cached"],
        exchange_open_order_signature=lambda: state["exchange"],
    )

    outcome = service.startup_converge(timeout_seconds=3)
    assert outcome.converged is True
    assert outcome.exchange_signature_match is True
