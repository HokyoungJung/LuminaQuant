"""Apply frozen allocation weights to report-only named-quant locked-OOS returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.portfolio.optimizer_core import metrics
from lumina_quant.portfolio.quality_gated_allocation import (
    _normalize_locked_oos_evaluation,
    _prepare_return_series,
    _timestamp,
)

_LEAK_FLAGS = (
    "uses_current_fold_oos",
    "uses_locked_oos_for_selection",
    "uses_locked_oos_for_objective",
    "uses_locked_oos_for_pruning",
    "uses_locked_oos_for_parameter_fitting",
    "uses_locked_oos_for_threshold",
    "uses_locked_oos_for_tie_break",
    "uses_locked_oos_for_correlation",
    "uses_locked_oos_for_sizing",
)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _assert_no_locked_oos_leakage(row: dict[str, Any], *, label: str) -> None:
    contaminated = [flag for flag in _LEAK_FLAGS if bool(row.get(flag, False))]
    if contaminated:
        raise ValueError(f"{label} has locked-OOS leakage flags: {contaminated}")


def _finite_nonnegative(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and nonnegative") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return parsed


def evaluate_locked_oos(
    allocation: dict[str, Any],
    suite_results: dict[str, Any],
    selection: dict[str, Any],
    *,
    allocation_sha256: str,
    suite_results_sha256: str,
    selection_sha256: str,
    rebalance_every_observations: int,
    allocation_cost_bps: float,
    periods_per_year: int,
) -> dict[str, Any]:
    """Evaluate frozen weights without optimization, scaling, or cash yield."""
    if allocation.get("artifact_kind") != "quality_gated_allocation_manifest":
        raise ValueError("allocation must be a quality_gated_allocation_manifest")
    if suite_results.get("purpose") != "locked_oos":
        raise ValueError("suite_results purpose must be exactly 'locked_oos'")
    if selection.get("purpose") != "selection":
        raise ValueError("selection artifact purpose must be exactly 'selection'")
    locked_lineage = suite_results.get("lineage")
    selection_lineage = selection.get("lineage")
    if not isinstance(locked_lineage, dict) or not isinstance(selection_lineage, dict):
        raise ValueError("selection and locked-OOS lineage are required")
    for key in ("suite_id", "base_strategy_spec_sha256"):
        if locked_lineage.get("suite", {}).get(key) != selection_lineage.get("suite", {}).get(key):
            raise ValueError(f"locked-OOS {key} differs from selection")
    if locked_lineage.get("runtime_config", {}).get("effective_sha256") != selection_lineage.get(
        "runtime_config", {}
    ).get("effective_sha256"):
        raise ValueError("locked-OOS runtime-config identity differs from selection")
    binding = suite_results.get("selection_artifact")
    if not isinstance(binding, dict) or binding.get("sha256") != selection_sha256:
        raise ValueError("locked-OOS results are not bound to the supplied selection artifact")
    if binding.get("manifest_sha256") != selection_lineage.get("suite", {}).get(
        "manifest_sha256"
    ) or binding.get("universe_receipt_sha256") != selection_lineage.get("universe", {}).get(
        "receipt_sha256"
    ):
        raise ValueError("locked-OOS selection manifest/universe binding differs")
    selection_period = selection.get("period")
    locked_period = suite_results.get("period")
    if not isinstance(selection_period, dict) or not isinstance(locked_period, dict):
        raise ValueError("selection and locked-OOS periods are required")
    selection_start = _timestamp(selection_period.get("start"), sleeve_id="selection")
    selection_end = _timestamp(selection_period.get("end"), sleeve_id="selection")
    locked_start = _timestamp(locked_period.get("start"), sleeve_id="locked_oos")
    locked_end = _timestamp(locked_period.get("end"), sleeve_id="locked_oos")
    if not selection_start < selection_end < locked_start < locked_end:
        raise ValueError(
            "periods must satisfy selection start < selection end < "
            "locked-OOS start < locked-OOS end"
        )
    for key, declared in (("start", locked_start), ("end", locked_end)):
        requested = suite_results.get(key)
        if requested is not None and _timestamp(requested, sleeve_id="locked_oos") != declared:
            raise ValueError(f"locked-OOS requested {key} differs from period {key}")
    for label, lineage, split_start in (
        ("selection", selection_lineage, selection_start),
        ("locked_oos", locked_lineage, locked_start),
    ):
        as_of = _timestamp(
            lineage.get("universe", {}).get("receipt", {}).get("as_of"), sleeve_id=label
        )
        if as_of > split_start:
            raise ValueError(f"{label} universe receipt as_of is after split start")
    source_artifacts = allocation.get("source_artifacts")
    bound = [
        row
        for row in source_artifacts or []
        if isinstance(row, dict) and row.get("sha256") == selection_sha256
    ]
    if len(bound) != 1 or bound[0].get("lineage") != selection_lineage:
        raise ValueError("allocation is not bound to the supplied selection artifact lineage")
    frozen_at = _timestamp(bound[0].get("frozen_at"), sleeve_id="allocation")
    if frozen_at > selection_end or frozen_at >= locked_start:
        raise ValueError(
            "allocation must be frozen no later than selection end and before locked-OOS start"
        )
    _assert_no_locked_oos_leakage(allocation, label="allocation manifest")

    frozen_evaluation = allocation.get("locked_oos_evaluation")
    if not isinstance(frozen_evaluation, dict):
        raise ValueError("allocation manifest is missing locked_oos_evaluation")
    frozen_evaluation = _normalize_locked_oos_evaluation(frozen_evaluation)
    requested_evaluation = _normalize_locked_oos_evaluation(
        {
            "rebalance_every_observations": rebalance_every_observations,
            "allocation_cost_bps": allocation_cost_bps,
            "periods_per_year": periods_per_year,
        }
    )
    if requested_evaluation != frozen_evaluation:
        raise ValueError(
            "locked-OOS evaluation arguments differ from the frozen allocation contract"
        )

    gross_cap = _finite_nonnegative(allocation.get("gross_cap"), label="gross_cap")
    if rebalance_every_observations <= 0:
        raise ValueError("rebalance_every_observations must be positive")
    allocation_cost_rate = (
        _finite_nonnegative(allocation_cost_bps, label="allocation_cost_bps") / 10_000.0
    )
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    children = allocation.get("children")
    if not isinstance(children, list):
        raise ValueError("allocation children must be a list")

    frozen: list[tuple[str, float]] = []
    seen_children: set[str] = set()
    for child in children:
        if not isinstance(child, dict):
            raise ValueError("allocation children must be objects")
        candidate_id = str(child.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_children:
            raise ValueError("allocation child candidate_id must be nonempty and unique")
        seen_children.add(candidate_id)
        _assert_no_locked_oos_leakage(child, label=f"child {candidate_id!r}")
        weight = _finite_nonnegative(child.get("weight"), label=f"child {candidate_id!r} weight")
        if weight > 0.0:
            frozen.append((candidate_id, weight))
    if not frozen:
        raise ValueError("allocation has no positive-weight children")
    gross_weight = math.fsum(weight for _, weight in frozen)
    if gross_weight > gross_cap + 1e-12:
        raise ValueError(f"positive child weights {gross_weight} exceed gross_cap {gross_cap}")

    raw_results = suite_results.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("suite_results results must be a list")
    results: dict[str, dict[str, Any]] = {}
    for row in raw_results:
        if not isinstance(row, dict):
            raise ValueError("suite results must be objects")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in results:
            raise ValueError("suite result candidate_id must be nonempty and unique")
        if row.get("returns_are_net") is not True:
            raise ValueError(f"locked-OOS result {candidate_id!r} is not net")
        results[candidate_id] = row

    sleeve_returns: dict[str, Any] = {}
    return_timestamps: dict[str, Any] = {}
    for candidate_id, _weight in frozen:
        row = results.get(candidate_id)
        if row is None:
            raise ValueError(
                f"missing locked-OOS result for positive-weight child {candidate_id!r}"
            )
        _assert_no_locked_oos_leakage(row, label=f"locked-OOS result {candidate_id!r}")
        if row.get("status") != "pass":
            raise ValueError(
                f"locked-OOS result for positive-weight child {candidate_id!r} did not pass"
            )
        values = row.get("returns")
        timestamps = row.get("return_timestamps")
        if not isinstance(values, list) or not values:
            raise ValueError(f"positive-weight child {candidate_id!r} has no returns")
        if not isinstance(timestamps, list) or not timestamps:
            raise ValueError(f"positive-weight child {candidate_id!r} has no return timestamps")
        sleeve_returns[candidate_id] = values
        return_timestamps[candidate_id] = timestamps

    aligned, alignment, observations = _prepare_return_series(sleeve_returns, return_timestamps)
    if alignment != "exact_timestamp_intersection" or observations < 2:
        raise ValueError("locked-OOS returns require at least two exact common timestamps")
    common = sorted(
        set.intersection(
            *(
                {_timestamp(value, sleeve_id=candidate_id) for value in timestamps}
                for candidate_id, timestamps in return_timestamps.items()
            )
        )
    )
    if common[0] < locked_start or common[-1] > locked_end:
        raise ValueError("aligned return timestamps must lie within the locked-OOS period")
    timestamps = [value.isoformat().replace("+00:00", "Z") for value in common]
    target = np.asarray([weight for _, weight in frozen], dtype=np.float64)
    matrix = np.column_stack([aligned[candidate_id] for candidate_id, _ in frozen])
    current = np.zeros_like(target)
    cash = 1.0
    portfolio = np.zeros(observations, dtype=np.float64)
    turnover = np.zeros(observations, dtype=np.float64)
    target_cash = max(0.0, 1.0 - float(target.sum()))
    for index, row in enumerate(matrix):
        rebalance_cost = 0.0
        if index % rebalance_every_observations == 0:
            turnover[index] = 0.5 * (
                float(np.abs(target - current).sum()) + abs(target_cash - cash)
            )
            current = target.copy()
            cash = target_cash
            rebalance_cost = turnover[index] * allocation_cost_rate
            if rebalance_cost >= 1.0:
                raise ValueError("portfolio value became nonpositive after rebalance cost")
            cash -= rebalance_cost
        gross_return = float(current @ row)
        portfolio[index] = gross_return - rebalance_cost
        ending_total = 1.0 + portfolio[index]
        if ending_total <= 0.0:
            raise ValueError("portfolio value became nonpositive")
        current = current * (1.0 + row) / ending_total
        cash /= ending_total
    if not np.all(np.isfinite(portfolio)):
        raise ValueError("portfolio returns must be finite")

    cash_weight = _finite_nonnegative(
        allocation.get("cash_weight", max(0.0, 1.0 - gross_weight)), label="cash_weight"
    )
    expected_cash = max(0.0, 1.0 - gross_weight)
    if not math.isclose(cash_weight, expected_cash, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"cash_weight {cash_weight} does not reconcile with frozen gross weight {gross_weight}"
        )
    return {
        "artifact_kind": "named_quant_locked_oos_report",
        "purpose": "locked_oos",
        "report_only": True,
        "input_sha256": {
            "allocation_manifest": allocation_sha256,
            "suite_results": suite_results_sha256,
            "selection_artifact": selection_sha256,
        },
        "period": {
            "requested_start": suite_results.get("start"),
            "requested_end": suite_results.get("end"),
            "observed_start": timestamps[0],
            "observed_end": timestamps[-1],
            "observations": observations,
        },
        "children": [
            {"candidate_id": candidate_id, "weight": weight} for candidate_id, weight in frozen
        ],
        "gross_cap": gross_cap,
        "gross_weight": gross_weight,
        "cash_weight": cash_weight,
        "cash_return": 0.0,
        "return_arithmetic": "frozen target weights drift between declared observation-count rebalances; rebalance one-way turnover cost is deducted; cash return is zero",
        "rebalance_every_observations": rebalance_every_observations,
        "allocation_cost_bps": float(allocation_cost_bps),
        "periods_per_year": periods_per_year,
        "rebalance_turnover": [float(value) for value in turnover],
        "alignment": "exact_timestamp_intersection",
        "return_timestamps": timestamps,
        "returns": [float(value) for value in portfolio],
        "metrics": metrics(portfolio, periods_per_year=periods_per_year),
        "leak_flags": {
            "weights_frozen_before_locked_oos": True,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_optimization": False,
            "uses_locked_oos_for_sizing": False,
            "reoptimized": False,
            "rescaled": False,
        },
    }


def write_report(
    *,
    allocation_path: Path,
    suite_results_path: Path,
    selection_path: Path,
    output_path: Path,
    rebalance_every_observations: int,
    allocation_cost_bps: float,
    periods_per_year: int,
) -> Path:
    allocation, allocation_sha256 = _load(allocation_path)
    suite_results, suite_results_sha256 = _load(suite_results_path)
    selection, selection_sha256 = _load(selection_path)
    report = evaluate_locked_oos(
        allocation,
        suite_results,
        selection,
        allocation_sha256=allocation_sha256,
        suite_results_sha256=suite_results_sha256,
        selection_sha256=selection_sha256,
        rebalance_every_observations=rebalance_every_observations,
        allocation_cost_bps=allocation_cost_bps,
        periods_per_year=periods_per_year,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-manifest", type=Path, required=True)
    parser.add_argument("--suite-results", type=Path, required=True)
    parser.add_argument("--selection-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rebalance-every-observations", type=int, required=True)
    parser.add_argument("--allocation-cost-bps", type=float, required=True)
    parser.add_argument("--periods-per-year", type=int, required=True)
    args = parser.parse_args(argv)
    print(
        write_report(
            allocation_path=args.allocation_manifest.resolve(),
            suite_results_path=args.suite_results.resolve(),
            selection_path=args.selection_artifact.resolve(),
            output_path=args.output.resolve(),
            rebalance_every_observations=args.rebalance_every_observations,
            allocation_cost_bps=args.allocation_cost_bps,
            periods_per_year=args.periods_per_year,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
