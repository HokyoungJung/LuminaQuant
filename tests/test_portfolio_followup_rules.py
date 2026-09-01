from __future__ import annotations

from lumina_quant.portfolio_followup_rules import (
    build_correlation_aware_sparse_fold_ensemble,
    build_sparse_fold_aware_ensemble,
    build_basis_search_universes,
    build_memory_ledger_row,
    evaluate_robustness_gates,
    mean_monthly_return,
    parse_memory_ledger_row,
    serialize_memory_ledger_row,
    validate_basis_universe,
)


def _payload(
    *,
    train_total_return: float,
    val_total_return: float,
    oos_total_return: float,
    train_sharpe: float,
    oos_sharpe: float,
    oos_max_drawdown: float,
    monthly: list[float],
    train_trade_count: float = 12.0,
    train_max_drawdown: float = 0.05,
    val_max_drawdown: float = 0.05,
) -> dict[str, object]:
    return {
        "train": {
            "total_return": train_total_return,
            "sharpe": train_sharpe,
            "trade_count": train_trade_count,
            "max_drawdown": train_max_drawdown,
        },
        "val": {
            "total_return": val_total_return,
            "sharpe": 1.0,
            "max_drawdown": val_max_drawdown,
        },
        "oos": {
            "total_return": oos_total_return,
            "sharpe": oos_sharpe,
            "sortino": 2.0,
            "calmar": 4.0,
            "max_drawdown": oos_max_drawdown,
            "volatility": 0.12,
        },
        "oos_monthly_returns": [
            {"month": f"2026-0{idx}", "total_return": value, "days": 20}
            for idx, value in enumerate(monthly, start=2)
        ],
    }


def test_validate_basis_universe_rejects_raw_and_derived_same_family() -> None:
    rows = [
        {
            "candidate_id": "incumbent",
            "metadata": {
                "basis_family": "incumbent",
                "basis_variant": "incumbent",
                "basis_universe": "shared",
            },
        },
        {
            "candidate_id": "raw",
            "metadata": {
                "basis_family": "static_blend",
                "basis_variant": "raw_55_45",
                "basis_universe": "raw_basis",
            },
        },
        {
            "candidate_id": "derived",
            "metadata": {
                "basis_family": "static_blend",
                "basis_variant": "derived_80_20",
                "basis_universe": "derived_basis",
            },
        },
    ]

    validation = validate_basis_universe(rows)

    assert validation["ok"] is False
    assert validation["conflicts"][0]["family"] == "static_blend"


def test_build_basis_search_universes_splits_raw_and_derived_rows() -> None:
    rows = [
        {
            "candidate_id": "incumbent",
            "metadata": {
                "basis_family": "incumbent",
                "basis_variant": "incumbent",
                "basis_universe": "shared",
            },
        },
        {
            "candidate_id": "soft",
            "metadata": {
                "basis_family": "soft_allocator",
                "basis_variant": "soft_allocator",
                "basis_universe": "shared",
            },
        },
        {
            "candidate_id": "raw",
            "metadata": {
                "basis_family": "static_blend",
                "basis_variant": "raw_55_45",
                "basis_universe": "raw_basis",
            },
        },
        {
            "candidate_id": "derived",
            "metadata": {
                "basis_family": "static_blend",
                "basis_variant": "derived_80_20",
                "basis_universe": "derived_basis",
            },
        },
    ]

    universes = build_basis_search_universes(rows)

    assert [row["candidate_id"] for row in universes["raw_basis"]] == ["incumbent", "soft", "raw"]
    assert [row["candidate_id"] for row in universes["derived_basis"]] == [
        "incumbent",
        "soft",
        "derived",
    ]


def test_evaluate_robustness_gates_accepts_candidate_that_clears_all_rules() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.03,
        val_total_return=0.04,
        oos_total_return=0.07,
        train_sharpe=0.6,
        oos_sharpe=2.1,
        oos_max_drawdown=0.05,
        monthly=[0.025, 0.028, 0.031],
    )

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is True
    assert result["rejection_reasons"] == []
    assert result["oos_monthly_mean"] == mean_monthly_return(candidate["oos_monthly_returns"])


def test_evaluate_robustness_gates_rejects_negative_train_return() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.0,
        val_total_return=0.04,
        oos_total_return=0.07,
        train_sharpe=0.6,
        oos_sharpe=2.1,
        oos_max_drawdown=0.05,
        monthly=[0.025, 0.028, 0.031],
    )

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is False
    assert "train_total_return_non_positive" in result["rejection_reasons"]


def test_evaluate_robustness_gates_rejects_no_trade_train_candidate() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.0,
        train_trade_count=0.0,
        val_total_return=0.04,
        oos_total_return=0.08,
        train_sharpe=0.8,
        oos_sharpe=2.2,
        oos_max_drawdown=0.04,
        monthly=[0.02, 0.03, 0.04],
    )

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is False
    assert "train_no_trade" in result["rejection_reasons"]


def test_evaluate_robustness_gates_rejects_candidate_split_drawdown_above_cap() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.03,
        val_total_return=0.04,
        oos_total_return=0.08,
        train_sharpe=0.7,
        oos_sharpe=2.1,
        oos_max_drawdown=0.16,
        monthly=[0.03, 0.03, 0.03],
    )

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is False
    assert "oos_max_drawdown_above_cap" in result["rejection_reasons"]


def test_evaluate_robustness_gates_rejects_positive_strict_liquidation_evidence() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.03,
        val_total_return=0.04,
        oos_total_return=0.08,
        train_sharpe=0.8,
        oos_sharpe=2.2,
        oos_max_drawdown=0.04,
        monthly=[0.03, 0.03, 0.03],
    )
    candidate["strict_validation"] = {
        "state_leverage_validation": {
            "liquidation_counts": {
                "blend_85_15": 1,
                "incumbent": 0,
            }
        }
    }

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is False
    assert result["strict_liquidation_count"] == 1
    assert "strict_liquidation_count_positive" in result["rejection_reasons"]


def test_evaluate_robustness_gates_rejects_strict_max_leverage_above_cap() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.03,
        val_total_return=0.04,
        oos_total_return=0.08,
        train_sharpe=0.8,
        oos_sharpe=2.2,
        oos_max_drawdown=0.04,
        monthly=[0.03, 0.03, 0.03],
    )
    candidate["strict_validation"] = {
        "leverage_by_state": {
            "incumbent": 3,
            "blend_85_15": 21,
            "autoresearch_55_45": 2,
        }
    }

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is False
    assert result["strict_max_leverage"] == 21.0
    assert "strict_max_leverage_above_cap" in result["rejection_reasons"]


def test_evaluate_robustness_gates_rejects_strict_average_leverage_above_cap() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.03,
        val_total_return=0.04,
        oos_total_return=0.08,
        train_sharpe=0.8,
        oos_sharpe=2.2,
        oos_max_drawdown=0.04,
        monthly=[0.03, 0.03, 0.03],
    )
    candidate["strict_validation"] = {
        "leverage_by_state": {
            "incumbent": 3,
            "blend_85_15": 6,
            "autoresearch_55_45": 2,
        },
        "strict_allocator": {
            "state_summary": {
                "oos": {
                    "counts": {
                        "incumbent": 1,
                        "blend_85_15": 9,
                    }
                }
            }
        },
    }

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is False
    assert result["strict_average_leverage"] == 5.7
    assert "strict_average_leverage_above_cap" in result["rejection_reasons"]


def test_memory_ledger_row_round_trips() -> None:
    row = build_memory_ledger_row(
        run_name="portfolio_meta_search",
        basis_universe="raw_basis",
        candidate_count=5,
        combination_count=10626,
        budget_bytes=8 * 1024 * 1024 * 1024,
        heavy_lock_path="/tmp/heavy.lock",
        session_memory_lease_path="/tmp/session.lock",
        status="completed",
        started_at="2026-04-02T10:00:00Z",
        completed_at="2026-04-02T10:05:00Z",
        memory_summary_path="/tmp/summary.json",
    )

    parsed = parse_memory_ledger_row(serialize_memory_ledger_row(row))

    assert parsed["basis_universe"] == "raw_basis"
    assert parsed["one_heavy_lane_only"] is True
    assert parsed["combination_count"] == 10626


def test_build_sparse_fold_aware_ensemble_penalizes_sparse_components() -> None:
    dense = {
        "name": "dense_candidate",
        "train": {"total_return": 0.03, "trade_count": 20.0},
        "val": {"total_return": 0.04, "sharpe": 1.2},
        "oos": {
            "total_return": 0.05,
            "return": 0.05,
            "sharpe": 2.0,
            "pbo": 0.20,
            "active_fold_ratio": 1.0,
            "inactive_fold_count": 0.0,
            "failed_fold_ratio": 0.0,
        },
        "return_streams": {
            "train": [{"t": 1, "v": 0.01}, {"t": 2, "v": 0.0}],
            "val": [{"t": 3, "v": 0.01}, {"t": 4, "v": 0.0}],
            "oos": [{"t": 5, "v": 0.01}, {"t": 6, "v": 0.0}],
        },
    }
    sparse = {
        "name": "sparse_candidate",
        "train": {"total_return": 0.03, "trade_count": 20.0},
        "val": {"total_return": 0.04, "sharpe": 1.2},
        "oos": {
            "total_return": 0.05,
            "return": 0.05,
            "sharpe": 2.0,
            "pbo": 0.20,
            "active_fold_ratio": 0.5,
            "inactive_fold_count": 4.0,
            "failed_fold_ratio": 0.5,
        },
        "return_streams": {
            "train": [{"t": 1, "v": 0.01}, {"t": 2, "v": 0.0}],
            "val": [{"t": 3, "v": 0.01}, {"t": 4, "v": 0.0}],
            "oos": [{"t": 5, "v": 0.01}, {"t": 6, "v": 0.0}],
        },
    }

    payload = build_sparse_fold_aware_ensemble([dense, sparse], max_members=2)

    components = {row["name"]: row for row in payload["components"]}
    assert components["dense_candidate"]["weight"] > components["sparse_candidate"]["weight"]


def test_correlation_aware_sparse_fold_ensemble_prefers_orthogonal_candidate() -> None:
    benchmark = {
        "name": "benchmark",
        "train": {"total_return": 0.03, "trade_count": 20.0},
        "val": {"total_return": 0.03, "sharpe": 1.1},
        "oos": {
            "total_return": 0.06,
            "return": 0.06,
            "sharpe": 3.0,
            "pbo": 0.20,
            "turnover": 0.2,
            "active_fold_ratio": 1.0,
            "inactive_fold_count": 0.0,
            "failed_fold_ratio": 0.0,
        },
        "return_streams": {
            "train": [{"t": 1, "v": 0.01}, {"t": 2, "v": 0.0}],
            "val": [{"t": 3, "v": 0.01}, {"t": 4, "v": 0.0}],
            "oos": [
                {"datetime": "2026-02-01T00:00:00Z", "v": 0.01},
                {"datetime": "2026-02-02T00:00:00Z", "v": -0.01},
                {"datetime": "2026-02-03T00:00:00Z", "v": 0.01},
                {"datetime": "2026-02-04T00:00:00Z", "v": -0.01},
            ],
        },
    }
    highly_correlated = {
        "name": "high_corr_candidate",
        "train": {"total_return": 0.04, "trade_count": 18.0},
        "val": {"total_return": 0.03, "sharpe": 1.2},
        "oos": {
            "total_return": 0.05,
            "return": 0.05,
            "sharpe": 2.8,
            "pbo": 0.20,
            "turnover": 0.2,
            "active_fold_ratio": 0.9,
            "inactive_fold_count": 0.0,
            "failed_fold_ratio": 0.0,
        },
        "return_streams": {
            "train": [{"t": 1, "v": 0.01}, {"t": 2, "v": 0.0}],
            "val": [{"t": 3, "v": 0.01}, {"t": 4, "v": 0.0}],
            "oos": [
                {"datetime": "2026-02-01T00:00:00Z", "v": 0.009},
                {"datetime": "2026-02-02T00:00:00Z", "v": -0.011},
                {"datetime": "2026-02-03T00:00:00Z", "v": 0.009},
                {"datetime": "2026-02-04T00:00:00Z", "v": -0.011},
            ],
        },
    }
    orthogonal = {
        "name": "orthogonal_candidate",
        "train": {"total_return": 0.04, "trade_count": 18.0},
        "val": {"total_return": 0.03, "sharpe": 1.2},
        "oos": {
            "total_return": 0.05,
            "return": 0.05,
            "sharpe": 2.8,
            "pbo": 0.20,
            "turnover": 0.2,
            "active_fold_ratio": 0.9,
            "inactive_fold_count": 0.0,
            "failed_fold_ratio": 0.0,
        },
        "return_streams": {
            "train": [{"t": 1, "v": 0.01}, {"t": 2, "v": 0.0}],
            "val": [{"t": 3, "v": 0.01}, {"t": 4, "v": 0.0}],
            "oos": [
                {"datetime": "2026-02-01T00:00:00Z", "v": 0.01},
                {"datetime": "2026-02-02T00:00:00Z", "v": 0.01},
                {"datetime": "2026-02-03T00:00:00Z", "v": -0.01},
                {"datetime": "2026-02-04T00:00:00Z", "v": -0.01},
            ],
        },
    }

    payload = build_correlation_aware_sparse_fold_ensemble(
        [benchmark, highly_correlated, orthogonal],
        max_members=2,
        correlation_penalty=3.0,
        max_weight=0.75,
    )

    selected_names = {row["name"] for row in payload["components"]}
    assert "benchmark" in selected_names
    assert "orthogonal_candidate" in selected_names
    assert "high_corr_candidate" not in selected_names
    excluded = {row["name"]: row["reason"] for row in payload["excluded_candidates"]}
    assert excluded["high_corr_candidate"] in {
        "correlation_penalty_dominated",
        "max_members_reached",
    }
