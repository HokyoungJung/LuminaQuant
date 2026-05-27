from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts" / "research" / "run_alpha_zoo_top_seed_hybrid_v35_v36_cost_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_top_seed_hybrid_cost_validation", RUNNER_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _candidate_row(
    name: str,
    leverage: float,
    allocation: float,
    *,
    train: float,
    validation: float,
    oos: float,
    mdd: float,
    sharpe: float = 1.0,
    sortino: float = 1.1,
    smart: float = 1.0,
    calmar: float = 2.0,
    rank: int = 1,
    live: bool = True,
) -> dict[str, object]:
    return {
        "candidate_name": name,
        "leverage": leverage,
        "allocation_fraction": allocation,
        "frozen_train_validation_rank": rank,
        "tv_selection_score": validation * 10.0 + train,
        "train_return": train,
        "validation_return": validation,
        "locked_oos_return": oos,
        "locked_oos_mdd": mdd,
        "locked_oos_sharpe": sharpe,
        "locked_oos_sortino": sortino,
        "locked_oos_smart_sortino": smart,
        "locked_oos_calmar": calmar,
        "locked_oos_trade_count": 10,
        "locked_oos_liquidation_count": 0,
        "total_account_wipeout_count": 0,
        "locked_oos_gate_pass": live,
        "live_promotion_possible": live,
        "locked_oos_rejection_reasons": "",
    }


def test_seed_bucket_union_dedupes_top_rows_from_latest_csv_shape(tmp_path: Path) -> None:
    rows = [
        _candidate_row(
            "alpha_zoo_fast_residual",
            7,
            0.15,
            train=0.40,
            validation=0.20,
            oos=0.30,
            mdd=0.11,
            calmar=2.7,
            rank=1,
        ),
        _candidate_row(
            "alpha_zoo_fast_residual",
            6,
            0.10,
            train=0.20,
            validation=0.12,
            oos=0.16,
            mdd=0.06,
            sharpe=1.5,
            sortino=1.8,
            smart=1.7,
            calmar=2.5,
            rank=2,
        ),
        _candidate_row(
            "alpha_zoo_quality_single_pair",
            7,
            0.20,
            train=1.20,
            validation=0.35,
            oos=0.08,
            mdd=0.04,
            calmar=2.0,
            rank=3,
        ),
        _candidate_row(
            "alpha_zoo_bad_oos",
            10,
            0.20,
            train=2.0,
            validation=0.40,
            oos=-0.20,
            mdd=0.40,
            live=False,
            rank=4,
        ),
    ]
    path = tmp_path / "candidates.csv"
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frame = MODULE._load_candidate_frame(path)
    selection = MODULE.select_seed_universe(frame, current_base_oos_return=0.05, top_n=2)

    labels = [row["label"] for row in selection["deduped_seed_universe"]]
    assert len(labels) == len(set(labels))
    assert "alpha_zoo_fast_residual 7x/0.15" in labels
    assert "alpha_zoo_quality_single_pair 7x/0.2" in labels
    assert selection["live_promotion_row_count"] == 3
    assert selection["filtered_gate_row_count"] == 3
    assert any(bucket["bucket"] == "filtered_balanced_score" for bucket in selection["buckets"])


class _FakeAlpha:
    def _portfolio_trade_return(
        self,
        trade: dict[str, float],
        *,
        leverage: float,
        allocation_fraction: float,
        round_trip_slippage_bps: float = 0.0,
    ) -> float:
        return (
            float(allocation_fraction)
            * float(leverage)
            * (float(trade["gross_return"]) - float(round_trip_slippage_bps) / 10000.0)
        )


def test_round_trip_bps_cost_is_bps_over_10000_not_percent() -> None:
    trade = {"gross_return": 0.01, "_min_intratrade_adverse_return": 0.0}

    no_cost, liquidated = MODULE._cost_adjusted_trade_return(
        _FakeAlpha(),
        trade,
        leverage=2.0,
        allocation_fraction=0.5,
        round_trip_slippage_bps=0.0,
    )
    five_bps, liquidated_5 = MODULE._cost_adjusted_trade_return(
        _FakeAlpha(),
        trade,
        leverage=2.0,
        allocation_fraction=0.5,
        round_trip_slippage_bps=5.0,
    )
    ten_bps, liquidated_10 = MODULE._cost_adjusted_trade_return(
        _FakeAlpha(),
        trade,
        leverage=2.0,
        allocation_fraction=0.5,
        round_trip_slippage_bps=10.0,
    )

    assert liquidated is False
    assert liquidated_5 is False
    assert liquidated_10 is False
    assert five_bps == pytest.approx(no_cost - 0.0005)
    assert ten_bps == pytest.approx(no_cost - 0.0010)


def test_isolated_liquidation_caps_trade_loss_at_allocation() -> None:
    ret, liquidated = MODULE._cost_adjusted_trade_return(
        _FakeAlpha(),
        {"gross_return": 0.10, "_min_intratrade_adverse_return": -0.99},
        leverage=10.0,
        allocation_fraction=0.15,
        round_trip_slippage_bps=10.0,
    )

    assert liquidated is True
    assert ret == pytest.approx(-0.15)


def test_metric_rows_include_required_cost_splits_and_gate_fields() -> None:
    split_metrics = {
        "train": {
            "total_return": 0.1,
            "max_drawdown": 0.02,
            "sharpe": 1,
            "sortino": 2,
            "smart_sortino": 1.9,
            "calmar": 5,
        },
        "validation": {
            "total_return": 0.2,
            "max_drawdown": 0.03,
            "sharpe": 1,
            "sortino": 2,
            "smart_sortino": 1.9,
            "calmar": 6,
        },
        "locked_oos": {
            "total_return": 0.3,
            "max_drawdown": 0.04,
            "sharpe": 1,
            "sortino": 2,
            "smart_sortino": 1.9,
            "calmar": 7,
            "trade_count": 4,
            "liquidation_count": 0,
            "account_wipeout_count": 0,
            "minimum_margin_buffer": 123.0,
        },
    }

    rows = MODULE._metric_rows_for_model(
        model_id="seed_alpha",
        model_kind="individual_seed",
        role="seed_universe",
        cost_bps=5.0,
        split_metrics=split_metrics,
        candidate_name="alpha",
        leverage=7.0,
        allocation_fraction=0.15,
    )

    assert [row["split"] for row in rows] == ["train", "validation", "locked_oos"]
    assert {row["round_trip_slippage_fee_bps"] for row in rows} == {5.0}
    assert rows[-1]["trade_event_count"] == 4
    assert rows[-1]["liquidation_count"] == 0
    assert rows[-1]["account_wipeout_count"] == 0
    assert rows[-1]["minimum_margin_buffer"] == pytest.approx(123.0)
    assert rows[-1]["locked_oos_deployable_gate_pass"] is True


def test_hybrid_public_result_marks_locked_oos_report_only() -> None:
    returns = np.asarray(
        [
            [0.01, 0.00],
            [0.00, 0.01],
            [0.02, 0.00],
            [0.00, 0.02],
            [0.03, 0.00],
            [0.00, 0.03],
        ],
        dtype=float,
    )
    split_masks = {
        "train": np.asarray([True, True, False, False, False, False]),
        "validation": np.asarray([False, False, True, True, False, False]),
        "locked_oos": np.asarray([False, False, False, False, True, True]),
    }
    result = {
        "version": "v3_5",
        "params": MODULE.hybrid.HybridParams().__dict__,
        "learned_params": {
            "high_vol_threshold": 0.0,
            "default_idx": 0,
            "high_vol_best_idx": 1,
            "default_weight_ratio": 0.5,
            "high_vol_weight_boost": 0.1,
            "cv_score": 1.0,
        },
        "splits": {
            name: MODULE.hybrid._metrics_from_returns(returns[mask, 0])
            for name, mask in split_masks.items()
        },
        "train_val_score": 1.0,
        "train_val_gate": True,
        "allocations": [{"index": 5, "weights": [0.5, 0.5]}],
        "final_weights": [0.5, 0.5],
        "portfolio_returns": np.zeros(returns.shape[0]),
    }
    streams = [
        {
            "label": "seed_a",
            "leverage": 2.0,
            "target_allocation": 0.10,
            "sleeve_gross_weight_sum": 1.0,
        },
        {
            "label": "seed_b",
            "leverage": 3.0,
            "target_allocation": 0.10,
            "sleeve_gross_weight_sum": 1.0,
        },
    ]

    public = MODULE._public_hybrid_result(
        result,
        labels=["seed_a", "seed_b"],
        returns=returns,
        split_masks=split_masks,
        timestamps=np.arange(returns.shape[0], dtype=np.int64),
        streams=streams,
    )

    provenance = public["selection_provenance"]
    assert provenance["selection_inputs"] == ["train", "validation"]
    assert provenance["uses_locked_oos_for_objective"] is False
    assert provenance["uses_locked_oos_for_pruning"] is False
    assert provenance["uses_locked_oos_for_selection"] is False
    assert provenance["uses_locked_oos_for_parameter_fitting"] is False
    assert public["splits"]["locked_oos"]["account_wipeout_count"] == 0
