from __future__ import annotations

import pytest

from lumina_quant.strategy_factory.selection import (
    _has_mode_metrics,
    _mode_metric_block_keys,
    _mode_metrics,
    allocate_portfolio_weights,
    hurdle_score,
    passes_dsr_spa_hard_gate,
    robust_score_from_metrics,
    safe_float,
    select_diversified_shortlist,
)


def test_safe_float_only_falls_back_for_coercion_errors() -> None:
    assert safe_float(None, default=1.5) == 1.5
    assert safe_float("bad", default=1.5) == 1.5
    assert safe_float(float("nan"), default=1.5) == 1.5
    assert safe_float(float("inf"), default=1.5) == 1.5

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
# Mode-key semantics regression (overfit_selection_gate task 2).
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


def test_strict_gate_fails_closed_when_pbo_is_missing() -> None:
    strict = {
        "enforce_selection_reject_gate": True,
        "dsr_gate_floor": 0.90,
        "spa_gate_ceiling": 0.05,
        "pbo_gate_ceiling": 0.50,
    }
    row = {
        "val": {
            "deflated_sharpe": 0.99,
            "spa_pvalue": 0.01,
        }
    }

    assert passes_dsr_spa_hard_gate(row, mode="val", robust_score_params=strict) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deflated_sharpe", float("nan")),
        ("spa_pvalue", float("inf")),
        ("pbo", float("nan")),
        ("deflated_sharpe", 1.01),
        ("spa_pvalue", -0.01),
        ("pbo", -0.01),
    ],
)
def test_strict_gate_fails_closed_on_nonfinite_statistics(field: str, value: float) -> None:
    strict = {
        "enforce_selection_reject_gate": True,
        "dsr_gate_floor": 0.90,
        "spa_gate_ceiling": 0.05,
        "pbo_gate_ceiling": 0.50,
    }
    metrics = {"deflated_sharpe": 0.99, "spa_pvalue": 0.01, "pbo": 0.10}
    metrics[field] = value

    assert (
        passes_dsr_spa_hard_gate(
            {"val": metrics},
            mode="val",
            robust_score_params=strict,
        )
        is False
    )


def test_validation_mode_never_reads_locked_oos_fallback() -> None:
    assert _mode_metric_block_keys("val") == ("validation", "val")
    row = {
        "symbols": ["BTC/USDT"],
        "locked_oos_report_only": {"return": 0.05, "sharpe": 2.0, "deflated_sharpe": 0.99},
    }
    assert _mode_metrics(row, mode="val") == {}
    assert _has_mode_metrics(row, mode="val") is False


def test_hurdle_score_validation_mode_uses_validation_hurdle_and_metrics() -> None:
    candidate = {
        "train": {"total_return": 0.01, "trade_count": 4.0},
        "val": {
            "return": 0.03,
            "sharpe": 1.2,
            "pbo": 0.2,
            "turnover": 0.1,
            "mdd": 0.04,
        },
        "oos": {
            "return": -0.20,
            "sharpe": -3.0,
            "pbo": 1.0,
            "turnover": 5.0,
            "mdd": 0.50,
        },
        "hurdle_fields": {
            "val": {"score": 7.0, "excess_return": 0.03, "pass": True},
            "oos": {"score": -9.0, "excess_return": -0.20, "pass": False},
        },
    }

    assert hurdle_score(candidate, mode="val") >= 7.0
    assert hurdle_score(candidate, mode="oos") < 0.0


def test_portfolio_weights_use_selected_split_and_respect_feasible_cap() -> None:
    rows = [
        {
            "name": f"strategy_{index}",
            "family": f"family_{index}",
            "timeframe": f"{index + 1}h",
            "shortlist_score": 20.0 if index == 0 else 0.0,
            "val": {"mdd": 0.01},
            "oos": {"mdd": 0.99 if index == 0 else 0.0},
        }
        for index in range(4)
    ]

    weighted = allocate_portfolio_weights(rows, mode="val", max_weight=0.35)
    weights = [float(row["portfolio_weight"]) for row in weighted]

    assert sum(weights) == pytest.approx(1.0)
    assert max(weights) <= 0.35 + 1e-12
    assert next(row for row in weighted if row["name"] == "strategy_0")["portfolio_weight"] == (
        pytest.approx(0.35)
    )


def test_portfolio_weight_cap_leaves_cash_when_full_investment_is_infeasible() -> None:
    rows = [
        {
            "name": f"strategy_{index}",
            "family": f"family_{index}",
            "timeframe": "1h",
            "shortlist_score": float(index),
            "val": {"mdd": 0.0},
        }
        for index in range(2)
    ]

    weighted = allocate_portfolio_weights(rows, mode="val", max_weight=0.35)

    assert [row["portfolio_weight"] for row in weighted] == pytest.approx([0.35, 0.35])
    assert all(row["unallocated_cash_weight"] == pytest.approx(0.30) for row in weighted)


def test_portfolio_weight_cap_holds_when_remaining_raw_weights_are_zero() -> None:
    rows = [
        {
            "name": "single_nonzero",
            "family": "single",
            "timeframe": "1h",
            "symbols": ["BTC/USDT"],
            "shortlist_score": 20.0,
            "val": {"mdd": 0.0},
        },
        *[
            {
                "name": f"pair_zero_{index}",
                "family": f"pair_{index}",
                "timeframe": "1h",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "shortlist_score": 0.0,
                "val": {"mdd": 0.0},
            }
            for index in range(2)
        ],
    ]

    weighted = allocate_portfolio_weights(
        rows,
        mode="val",
        max_weight=0.20,
        robust_score_params={"pair_multi_mix_bonus": 0.0},
    )
    weights = [float(row["portfolio_weight"]) for row in weighted]

    assert weights == pytest.approx([0.20, 0.20, 0.20])
    assert all(row["unallocated_cash_weight"] == pytest.approx(0.40) for row in weighted)


def test_strict_shortlist_rejects_upstream_failed_multi_asset_row() -> None:
    row = {
        "name": "failed_multi",
        "strategy_class": "PerpCrowdingCarryStrategy",
        "family": "cross_sectional",
        "timeframe": "1h",
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "train": {"total_return": 0.01, "trade_count": 10},
        "val": {
            "return": 0.05,
            "sharpe": 2.0,
            "mdd": 0.05,
            "trades": 20,
            "deflated_sharpe": 0.99,
            "spa_pvalue": 0.01,
            "pbo": 0.10,
        },
        "hurdle_fields": {"val": {"pass": True, "score": 10.0, "excess_return": 0.05}},
        "pass": False,
        "hard_reject_reasons": {"cost_stress": True},
    }

    selected = select_diversified_shortlist(
        [row],
        mode="val",
        allow_multi_asset=True,
        robust_score_params={
            "enforce_selection_reject_gate": True,
            "dsr_gate_floor": 0.90,
            "spa_gate_ceiling": 0.05,
            "pbo_gate_ceiling": 0.50,
        },
    )

    assert selected == []
