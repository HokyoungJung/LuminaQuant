"""Regression tests for frozen-weight named-quant locked-OOS reporting."""

from __future__ import annotations

import copy
import math

import pytest

from scripts.research.evaluate_named_quant_locked_oos import evaluate_locked_oos


def _lineage(*, manifest: str, universe: str, as_of: str) -> dict:
    return {
        "suite": {
            "suite_id": "same",
            "base_strategy_spec_sha256": "e" * 64,
            "manifest_sha256": manifest,
        },
        "universe": {"receipt_sha256": universe, "receipt": {"as_of": as_of}},
        "runtime_config": {"effective_sha256": "f" * 64},
        "behavioral_identity": {
            "exchange": "binance",
            "warmup_bars": 400,
            "determinism": {"random_number_generation": "none"},
            "source_commit": "a" * 40,
            "cost_profile": {"path": "/cost-profile.yaml", "sha256": "b" * 64},
            "runtime_config_sha256": "f" * 64,
            "data_inventory": {"root": "/data", "files": [], "sha256": "d" * 64},
        },
    }


def _allocation() -> dict:
    return {
        "artifact_kind": "quality_gated_allocation_manifest",
        "gross_cap": 0.8,
        "cash_weight": 0.25,
        "uses_locked_oos_for_selection": False,
        "locked_oos_evaluation": {
            "rebalance_every_observations": 1,
            "allocation_cost_bps": 0.0,
            "periods_per_year": 252,
        },
        "source_artifacts": [
            {
                "id": "selection",
                "sha256": "c" * 64,
                "frozen_at": "2024-12-31T00:00:00Z",
                "lineage": _lineage(
                    manifest="1" * 64, universe="2" * 64, as_of="2023-12-31T00:00:00Z"
                ),
            }
        ],
        "children": [
            {"candidate_id": "a", "weight": 0.25},
            {"candidate_id": "b", "weight": 0.50},
        ],
    }


def _results() -> dict:
    return {
        "purpose": "locked_oos",
        "start": "2025-01-01T00:00:00",
        "end": "2025-01-05T00:00:00",
        "period": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-05T00:00:00Z"},
        "lineage": _lineage(manifest="3" * 64, universe="4" * 64, as_of="2025-01-01T00:00:00Z"),
        "selection_artifact": {
            "sha256": "c" * 64,
            "manifest_sha256": "1" * 64,
            "universe_receipt_sha256": "2" * 64,
        },
        "results": [
            {
                "candidate_id": "a",
                "status": "pass",
                "return_timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "returns": [0.01, 0.02, 0.03],
                "returns_are_net": True,
            },
            {
                "candidate_id": "b",
                "status": "pass",
                "return_timestamps": ["2025-01-02", "2025-01-03", "2025-01-04"],
                "returns": [0.10, -0.04, 0.50],
                "returns_are_net": True,
            },
        ],
    }


def _selection() -> dict:
    return {
        "purpose": "selection",
        "period": {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T00:00:00Z"},
        "lineage": _lineage(manifest="1" * 64, universe="2" * 64, as_of="2023-12-31T00:00:00Z"),
    }


def _evaluate(
    allocation: dict | None = None,
    results: dict | None = None,
    selection: dict | None = None,
    rebalance_every_observations: int = 1,
    allocation_cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    allocation = _allocation() if allocation is None else allocation
    allocation["locked_oos_evaluation"] = {
        "rebalance_every_observations": rebalance_every_observations,
        "allocation_cost_bps": allocation_cost_bps,
        "periods_per_year": periods_per_year,
    }
    return evaluate_locked_oos(
        allocation,
        _results() if results is None else results,
        _selection() if selection is None else selection,
        allocation_sha256="a" * 64,
        suite_results_sha256="b" * 64,
        selection_sha256="c" * 64,
        rebalance_every_observations=rebalance_every_observations,
        allocation_cost_bps=allocation_cost_bps,
        periods_per_year=periods_per_year,
    )


def test_locked_oos_evaluation_must_match_frozen_contract_exactly() -> None:
    allocation = _allocation()
    with pytest.raises(ValueError, match="differ from the frozen"):
        evaluate_locked_oos(
            allocation,
            _results(),
            _selection(),
            allocation_sha256="a" * 64,
            suite_results_sha256="b" * 64,
            selection_sha256="c" * 64,
            rebalance_every_observations=5,
            allocation_cost_bps=0.0,
            periods_per_year=252,
        )


def test_exact_intersection_uses_frozen_daily_arithmetic_weights() -> None:
    report = _evaluate()
    assert report["return_timestamps"] == [
        "2025-01-02T00:00:00Z",
        "2025-01-03T00:00:00Z",
    ]
    assert report["returns"] == pytest.approx(
        [0.25 * 0.02 + 0.50 * 0.10, 0.25 * 0.03 - 0.50 * 0.04]
    )
    assert report["gross_weight"] == 0.75
    assert report["cash_return"] == 0.0
    assert report["leak_flags"]["reoptimized"] is False
    assert report["leak_flags"]["rescaled"] is False
    assert report["return_arithmetic"].startswith("frozen target weights drift")
    assert report["period"]["observations"] == 2
    assert report["input_sha256"] == {
        "allocation_manifest": "a" * 64,
        "suite_results": "b" * 64,
        "selection_artifact": "c" * 64,
    }


def test_missing_positive_weight_child_fails_closed() -> None:
    results = _results()
    results["results"] = results["results"][:1]
    with pytest.raises(ValueError, match=r"missing locked-OOS result.*'b'"):
        _evaluate(results=results)


def test_wrong_purpose_fails_closed() -> None:
    results = _results()
    results["purpose"] = "selection"
    with pytest.raises(ValueError, match="purpose must be exactly 'locked_oos'"):
        _evaluate(results=results)


@pytest.mark.parametrize("case", ["weight", "nan", "timestamp"])
def test_invalid_weight_return_or_timestamp_fails_closed(case: str) -> None:
    allocation = copy.deepcopy(_allocation())
    results = copy.deepcopy(_results())
    if case == "weight":
        allocation["children"][0]["weight"] = -0.1
    elif case == "nan":
        results["results"][0]["returns"][0] = math.nan
    else:
        results["results"][0]["return_timestamps"][1] = "not-a-timestamp"
    with pytest.raises(ValueError):
        _evaluate(allocation, results)


def test_cash_and_result_leakage_flags_must_reconcile() -> None:
    allocation = _allocation()
    allocation["cash_weight"] = 0.0
    with pytest.raises(ValueError, match="cash_weight"):
        _evaluate(allocation=allocation)

    results = _results()
    results["results"][0]["uses_locked_oos_for_selection"] = True
    with pytest.raises(ValueError, match="leakage"):
        _evaluate(results=results)


@pytest.mark.parametrize(
    "case", ["overlap", "lineage", "config", "binding", "allocation", "gross", "future_receipt"]
)
def test_locked_oos_identity_and_net_contract_fail_closed(case: str) -> None:
    allocation = copy.deepcopy(_allocation())
    results = copy.deepcopy(_results())
    selection = copy.deepcopy(_selection())
    if case == "overlap":
        selection["period"]["end"] = results["period"]["start"]
    elif case == "lineage":
        results["lineage"]["suite"]["suite_id"] = "unrelated"
    elif case == "config":
        results["lineage"]["runtime_config"]["effective_sha256"] = "0" * 64
    elif case == "binding":
        results["selection_artifact"]["sha256"] = "0" * 64
    elif case == "allocation":
        allocation["source_artifacts"][0]["sha256"] = "d" * 64
    elif case == "gross":
        results["results"][0]["returns_are_net"] = False
    else:
        results["lineage"]["universe"]["receipt"]["as_of"] = "2025-01-02T00:00:00Z"
    with pytest.raises(ValueError):
        _evaluate(allocation, results, selection)


def test_reversed_selection_period_fails_closed() -> None:
    selection = _selection()
    selection["period"] = {
        "start": "2024-12-31T00:00:00Z",
        "end": "2024-01-01T00:00:00Z",
    }
    with pytest.raises(ValueError, match="selection start"):
        _evaluate(selection=selection)


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2024-12-30", "2024-12-31"],
        ["2025-01-06", "2025-01-07"],
        ["2024-12-31", "2025-01-06"],
    ],
)
def test_aligned_timestamps_outside_locked_period_fail_closed(timestamps: list[str]) -> None:
    results = _results()
    for row in results["results"]:
        row["return_timestamps"] = timestamps
        row["returns"] = [0.01, 0.02]
    with pytest.raises(ValueError, match="within the locked-OOS period"):
        _evaluate(results=results)


def test_unweighted_locked_result_must_also_be_net() -> None:
    results = _results()
    results["results"].append(
        {"candidate_id": "unused", "status": "skip", "returns_are_net": False}
    )
    with pytest.raises(ValueError, match=r"unused.*not net"):
        _evaluate(results=results)


def test_cadence_allows_weights_to_drift_between_rebalances() -> None:
    daily = _evaluate(rebalance_every_observations=1)
    drifting = _evaluate(rebalance_every_observations=5)
    assert drifting["returns"][0] == pytest.approx(daily["returns"][0])
    assert drifting["returns"][1] != pytest.approx(daily["returns"][1])
    assert drifting["rebalance_turnover"][1] == 0.0


def test_allocation_turnover_cost_is_deducted() -> None:
    no_cost = _evaluate(allocation_cost_bps=0.0)
    cost = _evaluate(allocation_cost_bps=10.0)
    assert cost["rebalance_turnover"][0] == pytest.approx(0.75)
    assert cost["returns"][0] == pytest.approx(no_cost["returns"][0] - 0.75 * 0.001)


def test_rebalance_cost_is_included_in_next_period_drift_weights() -> None:
    report = _evaluate(rebalance_every_observations=5, allocation_cost_bps=100.0)
    first_return = 0.25 * 0.02 + 0.50 * 0.10 - 0.75 * 0.01
    expected_second = (0.25 * 1.02 * 0.03 + 0.50 * 1.10 * -0.04) / (1.0 + first_return)
    assert report["returns"] == pytest.approx([first_return, expected_second])


def test_rebalance_cost_cannot_make_nav_nonpositive() -> None:
    results = _results()
    for row in results["results"]:
        row["returns"] = [3.0, 3.0, 3.0]
    with pytest.raises(ValueError, match="nonpositive after rebalance cost"):
        _evaluate(results=results, allocation_cost_bps=20_000.0)


def test_declared_periods_per_year_is_used() -> None:
    report_252 = _evaluate(periods_per_year=252)
    report_365 = _evaluate(periods_per_year=365)
    assert report_252["periods_per_year"] == 252
    assert report_252["metrics"]["cagr"] != report_365["metrics"]["cagr"]


@pytest.mark.parametrize(
    ("cadence", "cost_bps", "periods"), [(0, 0.0, 252), (5, -1.0, 252), (5, 1.0, 0)]
)
def test_rebalance_cost_and_annualization_inputs_are_required_valid(
    cadence: int, cost_bps: float, periods: int
) -> None:
    with pytest.raises(ValueError):
        _evaluate(
            rebalance_every_observations=cadence,
            allocation_cost_bps=cost_bps,
            periods_per_year=periods,
        )
