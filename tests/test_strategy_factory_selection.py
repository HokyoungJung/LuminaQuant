from __future__ import annotations

import pytest

from lumina_quant.strategy_factory.selection import (
    _has_mode_metrics,
    _mode_metric_block_keys,
    _mode_metrics,
    hurdle_score,
    passes_dsr_spa_hard_gate,
    robust_score_from_metrics,
    safe_float,
)


def test_safe_float_only_falls_back_for_coercion_errors() -> None:
    assert safe_float(None, default=1.5) == 1.5
    assert safe_float("bad", default=1.5) == 1.5

    class ExplodingFloat:
        def __float__(self) -> float:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        safe_float(ExplodingFloat(), default=1.5)


def test_robust_score_from_metrics_penalizes_sparse_fold_coverage() -> None:
    dense = robust_score_from_metrics(
        {
            "sharpe": 2.0,
            "deflated_sharpe": 0.7,
            "pbo": 0.2,
            "return": 0.04,
            "mdd": 0.05,
            "turnover": 0.3,
            "active_fold_ratio": 1.0,
            "inactive_fold_count": 0.0,
            "failed_fold_ratio": 0.0,
        }
    )
    sparse = robust_score_from_metrics(
        {
            "sharpe": 2.0,
            "deflated_sharpe": 0.7,
            "pbo": 0.2,
            "return": 0.04,
            "mdd": 0.05,
            "turnover": 0.3,
            "active_fold_ratio": 0.5,
            "inactive_fold_count": 4.0,
            "failed_fold_ratio": 0.5,
        }
    )

    assert sparse < dense


def test_hurdle_score_dominantly_penalizes_no_trade_train_candidate() -> None:
    candidate = {
        "train": {"total_return": 0.0, "trade_count": 0.0},
        "oos": {"return": 0.05, "sharpe": 2.0, "pbo": 0.2, "turnover": 0.1, "mdd": 0.05},
        "hurdle_fields": {"oos": {"score": 10.0, "excess_return": 0.05, "pass": True}},
    }

    score = hurdle_score(candidate, mode="oos")

    assert score < -100_000.0


# --------------------------------------------------------------------------- #
# Legacy mode-key semantics regression (overfit_selection_gate task 2).
#
# The validation-gate change added a ``locked_oos_report_only`` fallback. It must
# be scoped to the NEW ``val`` / ``validation`` mode ONLY -- a legacy ``mode='oos'``
# (or ``mode='live'``) read must behave EXACTLY as before: primary key only, and
# ``{}`` / ``False`` when that primary key is absent.
# --------------------------------------------------------------------------- #
def test_legacy_oos_mode_ignores_locked_oos_report_only_block() -> None:
    """A ``mode='oos'`` read of a row that carries ONLY ``locked_oos_report_only``
    (no ``oos`` key) yields the pre-change ``{}`` / ``False`` -- the fallback is
    confined to the new validation-gate mode."""
    row = {
        "symbols": ["BTC/USDT"],
        "locked_oos_report_only": {"return": 0.05, "sharpe": 2.0, "deflated_sharpe": 0.99},
    }
    assert _mode_metric_block_keys("oos") == ("oos",)
    assert _mode_metrics(row, mode="oos") == {}
    assert _has_mode_metrics(row, mode="oos") is False


def test_legacy_live_mode_ignores_validation_block() -> None:
    """A ``mode='live'`` read of a row that carries ONLY ``validation`` (no ``val``
    key) yields the pre-change ``{}`` / ``False``."""
    row = {
        "symbols": ["BTC/USDT"],
        "validation": {"return": 0.05, "sharpe": 2.0, "deflated_sharpe": 0.99},
    }
    assert _mode_metric_block_keys("live") == ("val",)
    assert _mode_metrics(row, mode="live") == {}
    assert _has_mode_metrics(row, mode="live") is False


def test_legacy_oos_hard_gate_unchanged_on_locked_oos_only_row() -> None:
    """The reject gate (``mode='oos'``, strict floors) treats a locked-OOS-only row
    as having empty metrics -> DSR missing -> fail-CLOSED, exactly as it would for
    any metric-less row before the fallback was added (no accidental read of the
    locked-OOS block)."""
    strict = {
        "enforce_selection_reject_gate": True,
        "dsr_gate_floor": 0.90,
        "spa_gate_ceiling": 0.05,
        "pbo_gate_ceiling": 0.50,
    }
    row = {
        "symbols": ["BTC/USDT"],
        "locked_oos_report_only": {"deflated_sharpe": 0.99, "spa_pvalue": 0.01, "pbo": 0.1},
    }
    # Would PASS if the locked-OOS block were (wrongly) read; empty -> rejected.
    assert passes_dsr_spa_hard_gate(row, mode="oos", robust_score_params=strict) is False


def test_new_validation_mode_reads_locked_oos_fallback() -> None:
    """The NEW ``val`` mode DOES accept ``locked_oos_report_only`` as a last-resort
    fallback (its documented, scoped home)."""
    assert _mode_metric_block_keys("val") == ("validation", "val", "locked_oos_report_only")
    row = {
        "symbols": ["BTC/USDT"],
        "locked_oos_report_only": {"return": 0.05, "sharpe": 2.0, "deflated_sharpe": 0.99},
    }
    assert _mode_metrics(row, mode="val") == row["locked_oos_report_only"]
    assert _has_mode_metrics(row, mode="val") is True
