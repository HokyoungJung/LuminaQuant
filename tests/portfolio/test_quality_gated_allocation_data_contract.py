"""Data-contract regression tests for offline quality-gated allocation."""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path

import numpy as np
import pytest

import lumina_quant.portfolio.quality_gated_allocation as qga
from scripts.research.build_quality_gated_allocation import build_manifest_from_input
from scripts.research.compare_hierarchical_allocators import _load_returns, run_cell_variants


class _CapturingAllocator:
    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None

    def allocate(
        self,
        ids: list[str],
        returns: np.ndarray,
        *,
        upper: dict[str, float] | None = None,
    ) -> dict[str, float]:
        del upper
        self.matrix = np.asarray(returns)
        return dict.fromkeys(ids, 1.0 / len(ids))


def _sleeves() -> dict[str, dict[str, Any]]:
    return {
        "a": {
            "returns": [0.01, 0.02, 0.04],
            "return_timestamps": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-04T00:00:00Z",
            ],
            "returns_are_net": True,
            "returns_source": "train_validation",
            "turnover": 0.1,
            "strategy_class": "MovingAverageCrossStrategy",
            "symbols": ["BTC/USDT"],
        },
        "b": {
            "returns": [0.20, 0.30, 0.40],
            "return_timestamps": [
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
                "2026-01-04T00:00:00Z",
            ],
            "returns_are_net": True,
            "returns_source": "train_validation",
            "turnover": 0.2,
            "strategy_class": "MovingAverageCrossStrategy",
            "symbols": ["ETH/USDT"],
        },
    }


def _sources() -> list[dict[str, Any]]:
    return [
        {
            "id": "source",
            "path": "/data/selection.json",
            "sha256": "a" * 64,
            "max_age_hours": 8760,
            "ready": True,
            "portfolio_ready": True,
        }
    ]


def test_net_returns_skip_cost_drag_but_keep_turnover_penalty_metadata() -> None:
    returns = [0.01, 0.02, 0.015, 0.025]

    baseline = qga.compute_sleeve_quality(returns, 0.0)
    quality = qga.compute_sleeve_quality(
        returns,
        0.5,
        returns_are_net=True,
        turnover_penalty_lambda=0.25,
    )

    assert quality["net_sharpe"] == baseline["net_sharpe"]
    assert quality["net_calmar"] == baseline["net_calmar"]
    assert quality["turnover"] == 0.5
    assert quality["quality_score"] == pytest.approx(quality["net_sharpe"] - 0.125)


def test_timestamped_returns_are_aligned_by_exact_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _CapturingAllocator()
    monkeypatch.setattr(qga, "_build_allocator", lambda *args, **kwargs: allocator)
    sleeves = _sleeves()

    qga.allocate_quality_gated(
        {sid: spec["returns"] for sid, spec in sleeves.items()},
        {sid: spec["turnover"] for sid, spec in sleeves.items()},
        returns_are_net={sid: spec["returns_are_net"] for sid, spec in sleeves.items()},
        return_timestamps={sid: spec["return_timestamps"] for sid, spec in sleeves.items()},
    )

    assert allocator.matrix is not None
    np.testing.assert_allclose(allocator.matrix, [[0.02, 0.20], [0.04, 0.40]])


def test_mixed_timestamped_and_untimestamped_returns_fail_closed() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        qga.allocate_quality_gated(
            {"a": [0.01, 0.02], "b": [0.03, 0.04]},
            returns_are_net={"a": True, "b": True},
            return_timestamps={"a": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]},
        )


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        ["2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"],
    ],
    ids=["duplicate", "unsorted"],
)
def test_duplicate_or_unsorted_timestamps_fail_closed(timestamps: list[str]) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        qga.allocate_quality_gated(
            {"a": [0.01, 0.02]},
            returns_are_net={"a": True},
            return_timestamps={"a": timestamps},
        )


def test_timestamp_and_return_length_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        qga.allocate_quality_gated(
            {"a": [0.01, 0.02]},
            returns_are_net={"a": True},
            return_timestamps={"a": ["2026-01-01T00:00:00Z"]},
        )


def test_fewer_than_two_common_timestamped_observations_fail_closed() -> None:
    with pytest.raises(ValueError, match="common"):
        qga.allocate_quality_gated(
            {"a": [0.01, 0.02], "b": [0.03, 0.04]},
            returns_are_net={"a": True, "b": True},
            return_timestamps={
                "a": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
                "b": ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
            },
        )


@pytest.mark.parametrize(
    "contamination",
    [
        {"returns_source": "locked_oos"},
        {"uses_locked_oos_for_selection": True},
        {"uses_locked_oos_for_correlation": True},
        {"uses_locked_oos_for_sizing": True},
        {"uses_current_fold_oos": True},
    ],
    ids=["source", "selection", "correlation", "sizing", "current_fold"],
)
def test_locked_oos_input_for_allocation_fails_closed(contamination: dict[str, Any]) -> None:
    sleeves = _sleeves()
    sleeves["a"].update(contamination)

    with pytest.raises(ValueError, match=r"locked.*oos|oos.*locked"):
        qga.build_allocation_manifest(sleeves, source_artifacts=[])


@pytest.mark.parametrize(
    "source",
    [
        "train_validation_plus_oos",
        {"nested": {"purpose": "LOCKED-OOS"}},
        {"nested": {"arbitrary": "oos"}},
        {"nested": ["train", {"arbitrary": "locked_oos"}]},
        {"arbitrary": "train_validation_plus_oos"},
        {"stream": "locked_oos input NEVER used"},
    ],
)
def test_nested_or_composite_oos_source_fails_closed(source: Any) -> None:
    sleeves = _sleeves()
    sleeves["a"]["returns_source"] = source
    with pytest.raises(ValueError, match=r"locked.*oos|oos.*locked"):
        qga.build_allocation_manifest(sleeves, source_artifacts=[])


def test_opt_in_contract_requires_timestamp_and_source_for_every_active_stream() -> None:
    sleeves = _sleeves()
    del sleeves["b"]["return_timestamps"]
    with pytest.raises(ValueError, match="return_timestamps"):
        qga.build_allocation_manifest(sleeves, source_artifacts=[])
    sleeves = _sleeves()
    del sleeves["b"]["returns_source"]
    with pytest.raises(ValueError, match="returns_source"):
        qga.build_allocation_manifest(sleeves, source_artifacts=[])
    sleeves = _sleeves()
    sleeves["b"]["returns_are_net"] = "false"
    with pytest.raises(ValueError, match="returns_are_net"):
        qga.build_allocation_manifest(sleeves, source_artifacts=[])


def test_variant_comparison_uses_same_provenance_gate() -> None:
    payload = {"sleeves": _sleeves(), "allocator_variants": [{"method": "erc"}]}
    payload["sleeves"]["a"]["returns_source"] = {"split": "oos"}
    with pytest.raises(ValueError, match=r"locked.*oos|oos.*locked"):
        run_cell_variants(payload)


def test_variant_comparison_records_solver_nonconvergence(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"sleeves": _sleeves(), "allocator_variants": [{"method": "wasserstein_dro"}]}

    def fail(*args: Any, **kwargs: Any) -> dict[str, float]:
        raise RuntimeError("no certificate")

    monkeypatch.setattr(qga, "allocate_quality_gated", fail)
    result = run_cell_variants(payload)
    assert result["00_wasserstein_dro"]["error"] == "no certificate"


def test_manifest_records_opt_in_data_contract_and_allocator_params() -> None:
    manifest = qga.build_allocation_manifest(
        _sleeves(),
        source_artifacts=_sources(),
        allocator_params={"radius": 1e-8},
    )

    assert manifest["allocator_params"] == {"radius": 1e-8}
    assert manifest["return_data_contract"] == {
        "alignment": "exact_timestamp_intersection",
        "common_observations": 2,
        "returns_are_net": {"a": True, "b": True},
        "returns_source": {"a": "train_validation", "b": "train_validation"},
    }
    assert manifest["optimizer_provenance"]["selection_inputs"] == ["train", "validation"]
    assert manifest["correlation_input_provenance"]["uses_locked_oos_for_correlation"] is False


def test_manifest_freezes_locked_oos_evaluation_contract() -> None:
    contract = {
        "rebalance_every_observations": 5,
        "allocation_cost_bps": 10.0,
        "periods_per_year": 252,
    }
    manifest = build_manifest_from_input(
        {
            "sleeves": _sleeves(),
            "source_artifacts": _sources(),
            "locked_oos_evaluation": contract,
        }
    )
    assert manifest["locked_oos_evaluation"] == contract


def test_unready_referenced_source_fails_before_empty_allocation_can_be_frozen() -> None:
    sleeves = _sleeves()
    for sleeve in sleeves.values():
        sleeve["returns"] = [-0.01, -0.02, -0.03]
        sleeve["return_timestamps"] = [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ]
    sources = [{**_sources()[0], "ready": False}]
    with pytest.raises(ValueError, match="not portfolio-ready"):
        qga.build_allocation_manifest(sleeves, source_artifacts=sources, min_sleeves=2)


def test_unready_explicit_source_fails_even_when_no_sleeve_references_it() -> None:
    sources = [*_sources(), {**_sources()[0], "id": "unused", "ready": False}]
    with pytest.raises(ValueError, match=r"unused.*not portfolio-ready"):
        qga.build_allocation_manifest(_sleeves(), source_artifacts=sources)


def test_materialized_research_input_cannot_freeze_when_no_sleeves_survive() -> None:
    sleeves = _sleeves()
    for sleeve in sleeves.values():
        sleeve["returns"] = [-0.01, -0.02, -0.03]
        sleeve["return_timestamps"] = [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ]
    with pytest.raises(ValueError, match=r"min_sleeves=.*all-cash"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources(), min_sleeves=2)


def test_materialized_research_input_enforces_configured_family_floor() -> None:
    sleeves = _sleeves()
    sleeves["a"]["family"] = "same"
    sleeves["b"]["family"] = "same"
    with pytest.raises(ValueError, match="min_families=2"):
        qga.build_allocation_manifest(
            sleeves,
            source_artifacts=_sources(),
            min_sleeves=2,
            min_families=2,
        )


def test_legacy_manifest_omits_opt_in_data_contract() -> None:
    manifest = qga.build_allocation_manifest(
        {
            "a": {
                "returns": [0.01, 0.02, 0.03],
                "turnover": 0.0,
                "strategy_class": "MovingAverageCrossStrategy",
                "symbols": ["BTC/USDT"],
            }
        },
        source_artifacts=_sources(),
    )

    assert "return_data_contract" not in manifest
    assert "allocator_params" not in manifest


def test_data_pc_clis_use_nested_allocator_and_exact_net_panel(tmp_path: Path) -> None:
    payload = {
        "allocator": {
            "method": "erc",
            "min_sleeves": 2,
            "upper": 0.8,
            "gross_cap": 0.9,
            "turnover_penalty_lambda": 0.25,
        },
        "sleeves": _sleeves(),
        "source_artifacts": _sources(),
    }
    manifest = build_manifest_from_input(payload)
    assert manifest["allocation_method"] == "erc"
    assert manifest["gross_cap"] == 0.9
    assert "quality_score" in manifest["sleeve_quality"]["a"]

    path = tmp_path / "allocation_input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    ids, matrix = _load_returns(path)
    assert ids == ["a", "b"]
    np.testing.assert_allclose(matrix, [[0.02, 0.20], [0.04, 0.40]])


@pytest.mark.parametrize(
    "sources",
    [
        [],
        [{**_sources()[0], "id": "other"}],
        [{**_sources()[0], "ready": False}],
        [*_sources(), *_sources()],
    ],
    ids=["missing", "dangling", "not_ready", "duplicate"],
)
def test_materialized_children_require_unique_ready_source_artifacts(
    sources: list[dict[str, Any]],
) -> None:
    sleeves = _sleeves()
    for sleeve in sleeves.values():
        sleeve["source_artifact_id"] = "source"
    with pytest.raises(ValueError, match=r"source artifact|source_artifacts"):
        qga.build_allocation_manifest(sleeves, source_artifacts=sources)
