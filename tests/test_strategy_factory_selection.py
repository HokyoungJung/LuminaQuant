from __future__ import annotations

import pytest

from lumina_quant.strategy_factory.selection import (
    hurdle_score,
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
