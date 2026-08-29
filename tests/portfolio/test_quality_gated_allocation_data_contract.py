"""Data-contract regression tests for offline quality-gated allocation."""

from __future__ import annotations

from typing import Any
import json
import math
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
            "fit_start": "2026-01-01T00:00:00Z",
            "fit_end": "2026-01-04T00:00:00Z",
            "as_of": "2026-01-05T00:00:00Z",
            "apply_start": "2026-01-05T00:00:00Z",
            "turnover": 0.1,
            "strategy_class": "MovingAverageCrossStrategy",
            "symbols": ["BTC/USDT"],
            "source_artifact_id": "source",
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
            "fit_start": "2026-01-02T00:00:00Z",
            "fit_end": "2026-01-04T00:00:00Z",
            "as_of": "2026-01-05T00:00:00Z",
            "apply_start": "2026-01-05T00:00:00Z",
            "turnover": 0.2,
            "strategy_class": "MovingAverageCrossStrategy",
            "symbols": ["ETH/USDT"],
            "source_artifact_id": "source",
        },
    }


def _sources(
    sleeves: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sleeves = sleeves or _sleeves()
    return [
        {
            "id": "source",
            "path": "/data/selection.json",
            "sha256": "a" * 64,
            "max_age_hours": 8760,
            "ready": True,
            "portfolio_ready": True,
            "return_panel_sha256_by_sleeve": {
                sleeve_id: qga._materialized_return_panel_sha256(sleeve_id, spec)
                for sleeve_id, spec in sleeves.items()
            },
        }
    ]


def _risk_scaled_sleeves() -> dict[str, dict[str, Any]]:
    sleeves = _sleeves()
    timestamps = ["2026-01-02T00:00:00Z", "2026-01-04T00:00:00Z"]
    for spec in sleeves.values():
        spec["returns"] = spec["returns"][-2:]
        spec["return_timestamps"] = timestamps.copy()
        spec["fit_start"] = timestamps[0]
        spec["fit_end"] = timestamps[-1]
        spec["as_of"] = "2026-01-05T00:00:00Z"
        spec["apply_start"] = "2026-01-05T00:00:00Z"
    return sleeves


def _risk_scaling_spec() -> dict[str, Any]:
    return {
        "method": "target_vol",
        "sigma_target_annual": 0.05,
        "bars_per_year": 365,
        "min_observations": 2,
    }


def test_risk_scaling_accepts_exact_fit_window_boundaries() -> None:
    sleeves = _risk_scaled_sleeves()
    manifest = qga.build_allocation_manifest(
        sleeves,
        source_artifacts=_sources(sleeves),
        min_sleeves=1,
        risk_scaling=_risk_scaling_spec(),
    )
    assert manifest["risk_scaling"]["fit_start"] == "2026-01-02T00:00:00Z"
    assert manifest["risk_scaling"]["fit_end"] == "2026-01-04T00:00:00Z"
    boundary_sleeves = _risk_scaled_sleeves()
    for sleeve in boundary_sleeves.values():
        sleeve["as_of"] = sleeve["fit_end"]
        sleeve["apply_start"] = sleeve["fit_end"]
    boundary_manifest = qga.build_allocation_manifest(
        boundary_sleeves,
        source_artifacts=_sources(boundary_sleeves),
        min_sleeves=1,
        risk_scaling=_risk_scaling_spec(),
    )
    assert boundary_manifest["risk_scaling"]["apply_start"] == "2026-01-04T00:00:00Z"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda sleeves: sleeves["a"].update(
                return_timestamps=["2026-01-01T00:00:00Z", "2026-01-04T00:00:00Z"]
            ),
            "fit window",
        ),
        (
            lambda sleeves: sleeves["a"].update(
                return_timestamps=["2026-01-02T00:00:00Z", "2026-01-05T00:00:00Z"]
            ),
            "fit window",
        ),
        (
            lambda sleeves: sleeves["a"].update(
                return_timestamps=["2026-01-02T00:00:00Z", sleeves["a"]["apply_start"]]
            ),
            "fit window",
        ),
        (
            lambda sleeves: sleeves["b"].update(fit_end="2026-01-03T00:00:00Z"),
            "fit window",
        ),
        (
            lambda sleeves: sleeves["b"].update(
                return_timestamps=["2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z"]
            ),
            "coverage",
        ),
    ],
)
def test_risk_scaling_rejects_out_of_window_or_mixed_sleeve_evidence(
    mutation: Any, match: str
) -> None:
    sleeves = _risk_scaled_sleeves()
    mutation(sleeves)
    with pytest.raises(ValueError, match=match):
        qga.build_allocation_manifest(
            sleeves,
            source_artifacts=_sources(sleeves),
            min_sleeves=1,
            risk_scaling=_risk_scaling_spec(),
        )


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
    del sleeves["b"]["returns_are_net"]
    with pytest.raises(ValueError, match="returns_are_net"):
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
    sleeves = _sleeves()
    manifest = qga.build_allocation_manifest(
        sleeves,
        source_artifacts=_sources(sleeves),
        method="nco",
        allocator_params={"n_clusters": 2},
    )

    assert manifest["allocator_params"] == {"n_clusters": 2}
    assert manifest["return_data_contract"] == {
        "alignment": "exact_timestamp_intersection",
        "common_observations": 2,
        "returns_are_net": {"a": True, "b": True},
        "returns_source": {"a": "train_validation", "b": "train_validation"},
        "panel_sha256_by_sleeve": {
            sleeve_id: qga._materialized_return_panel_sha256(sleeve_id, spec)
            for sleeve_id, spec in sleeves.items()
        },
        "fit_apply_timestamps": {
            sleeve_id: {key: spec[key] for key in ("fit_start", "fit_end", "as_of", "apply_start")}
            for sleeve_id, spec in sleeves.items()
        },
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
        sleeve["fit_start"] = sleeve["return_timestamps"][0]
        sleeve["fit_end"] = sleeve["return_timestamps"][-1]
    with pytest.raises(ValueError, match=r"min_sleeves=.*all-cash"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources(sleeves), min_sleeves=2)


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


def test_legacy_omission_is_allowed_only_without_active_evidence() -> None:
    manifest = qga.build_allocation_manifest(
        {
            "a": {
                "returns": [],
                "strategy_class": "MovingAverageCrossStrategy",
                "symbols": ["BTC/USDT"],
            }
        },
        source_artifacts=_sources(),
    )

    assert "return_data_contract" not in manifest
    assert "optimizer_provenance" not in manifest
    assert "correlation_input_provenance" not in manifest
    assert "allocator_params" not in manifest


def test_active_evidence_cannot_use_legacy_contract_omissions() -> None:
    sleeves = _sleeves()
    del sleeves["a"]["return_timestamps"]
    del sleeves["a"]["returns_are_net"]
    del sleeves["a"]["returns_source"]
    del sleeves["a"]["turnover"]
    with pytest.raises(ValueError, match="turnover"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources())


@pytest.mark.parametrize(
    "returns",
    ["0.01,0.02", [[0.01, 0.02]], [0.01, True], [0.01, float("nan")]],
    ids=["string", "nested", "boolean", "nonfinite"],
)
def test_return_panels_reject_noncanonical_scalar_evidence(returns: Any) -> None:
    with pytest.raises(ValueError, match="returns"):
        qga.allocate_quality_gated({"a": returns}, {"a": 0.1})


@pytest.mark.parametrize(
    "source",
    [
        "train_validation_test",
        {"splits": ["train", "validation", "test"]},
        {"splits": ["train", "validation", "holdout"]},
        {"splits": ["train", "validation"], "extra": "metadata"},
    ],
)
def test_return_source_rejects_extra_or_holdout_splits(source: Any) -> None:
    sleeves = _sleeves()
    sleeves["a"]["returns_source"] = source
    with pytest.raises(ValueError, match="returns_source"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources())


def test_materialized_active_sleeve_requires_explicit_finite_turnover() -> None:
    sleeves = _sleeves()
    del sleeves["a"]["turnover"]
    with pytest.raises(ValueError, match="turnover"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources())
    sleeves = _sleeves()
    sleeves["a"]["turnover"] = float("inf")
    with pytest.raises(ValueError, match="turnover"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources())


def test_manifest_rejects_post_scaling_gross_cap_excess(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeves = _sleeves()
    monkeypatch.setattr(
        qga,
        "allocate_quality_gated",
        lambda *args, **kwargs: {"a": 0.7, "b": 0.7},
    )
    with pytest.raises(ValueError, match="gross_cap"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources(), gross_cap=0.8)


def test_relative_upper_is_not_replaced_by_gross_cap_and_two_sleeves_scale_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def allocate(*args: Any, **kwargs: Any) -> dict[str, float]:
        captured["upper"] = kwargs["upper"]
        return {"a": 0.5, "b": 0.5}

    monkeypatch.setattr(qga, "allocate_quality_gated", allocate)
    manifest = qga.build_allocation_manifest(
        _sleeves(), source_artifacts=_sources(), upper=0.75, gross_cap=0.4
    )

    assert captured["upper"] == {"a": 0.75, "b": 0.75}
    assert [child["weight"] for child in manifest["children"]] == [0.2, 0.2]
    assert math.fsum(child["weight"] for child in manifest["children"]) == 0.4
    assert (
        math.fsum(
            (math.fsum(child["weight"] for child in manifest["children"]), manifest["cash_weight"])
        )
        == 1.0
    )


def test_upper_cap_map_ignores_explicit_empty_candidate_before_survivor_gate() -> None:
    sleeves = _sleeves()
    sleeves["empty"] = {
        **sleeves["a"],
        "returns": None,
        "return_timestamps": [],
        "turnover": None,
        "symbols": ["SOL/USDT"],
    }
    manifest = qga.build_allocation_manifest(
        sleeves,
        source_artifacts=_sources({key: value for key, value in sleeves.items() if key != "empty"}),
        upper={"a": 0.75, "b": 0.75},
    )
    assert {child["candidate_id"] for child in manifest["children"]} == {"a", "b"}


def test_equal_thirds_scale_to_gross_cap_without_rounded_accounting_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeves = _sleeves()
    sleeves["c"] = {
        **sleeves["a"],
        "symbols": ["SOL/USDT"],
        "family": "c",
    }
    monkeypatch.setattr(
        qga,
        "allocate_quality_gated",
        lambda *args, **kwargs: {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0},
    )
    manifest = qga.build_allocation_manifest(
        sleeves, source_artifacts=_sources(sleeves), gross_cap=0.8
    )

    gross = math.fsum(child["weight"] for child in manifest["children"])
    assert gross == 0.8
    assert manifest["cash_weight"] == 1.0 - gross
    assert math.fsum((gross, manifest["cash_weight"])) == 1.0


def test_manifest_rejects_noncanonical_child_identities() -> None:
    sleeves = _sleeves()
    sleeves["a"]["symbols"] = [" BTC/USDT "]
    with pytest.raises(ValueError, match="stripped"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources())


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


@pytest.mark.parametrize(
    "artifact",
    [
        {"sha256": "a" * 63},
        {"sha256": "g" * 64},
        {"max_age_hours": 0},
        {"max_age_hours": float("nan")},
    ],
)
def test_materialized_sources_require_sealed_artifacts(artifact: dict[str, Any]) -> None:
    sleeves = _sleeves()
    for sleeve in sleeves.values():
        sleeve["source_artifact_id"] = "source"
    with pytest.raises(ValueError, match=r"sha256|max_age"):
        qga.build_allocation_manifest(sleeves, source_artifacts=[{**_sources()[0], **artifact}])


def test_manifest_rejects_invalid_economics_and_requires_resolved_children() -> None:
    sleeves = _sleeves()
    for sleeve in sleeves.values():
        sleeve["source_artifact_id"] = "source"
    with pytest.raises(ValueError, match="turnover"):
        qga.build_allocation_manifest(
            {**sleeves, "a": {**sleeves["a"], "turnover": float("nan")}},
            source_artifacts=_sources(),
        )
    with pytest.raises(ValueError, match="gross_cap"):
        qga.build_allocation_manifest(sleeves, source_artifacts=_sources(), gross_cap=0.0)
    with pytest.raises(ValueError, match="strategy_class"):
        qga.build_allocation_manifest(
            {**sleeves, "a": {**sleeves["a"], "strategy_class": ""}},
            source_artifacts=_sources(),
        )
    manifest = qga.build_allocation_manifest(sleeves, source_artifacts=_sources(), gross_cap=0.6)
    assert all(child["weight"] <= 0.6 for child in manifest["children"])


def test_allocator_empty_result_preserves_cash(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingAllocator:
        def allocate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
            return {}

    monkeypatch.setattr(qga, "_build_allocator", lambda *args, **kwargs: _FailingAllocator())
    sleeves = _sleeves()
    assert (
        qga.allocate_quality_gated(
            {sid: spec["returns"] for sid, spec in sleeves.items()},
            {sid: spec["turnover"] for sid, spec in sleeves.items()},
            returns_are_net=dict.fromkeys(sleeves, True),
        )
        == {}
    )


def test_allocator_contract_validates_before_empty_data_and_rejects_bad_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unsupported allocator_params"):
        qga.allocate_quality_gated({}, method="erc", allocator_params={"ignored": 1})
    with pytest.raises(ValueError, match="method"):
        qga.allocate_quality_gated({}, method=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_sleeves"):
        qga.allocate_quality_gated({}, min_sleeves=True)

    class _BadAllocator:
        def allocate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
            return {"unexpected": 1.0}

    monkeypatch.setattr(qga, "_build_allocator", lambda *args, **kwargs: _BadAllocator())
    sleeves = _sleeves()
    with pytest.raises(ValueError, match="exactly"):
        qga.allocate_quality_gated(
            {sid: spec["returns"] for sid, spec in sleeves.items()},
            {sid: spec["turnover"] for sid, spec in sleeves.items()},
            returns_are_net=dict.fromkeys(sleeves, True),
        )


@pytest.mark.parametrize("result", [None, False, 0, []])
def test_allocator_rejects_falsey_non_mapping_outputs(
    monkeypatch: pytest.MonkeyPatch,
    result: Any,
) -> None:
    class _BadAllocator:
        def allocate(self, *args: Any, **kwargs: Any) -> Any:
            return result

    monkeypatch.setattr(qga, "_build_allocator", lambda *args, **kwargs: _BadAllocator())
    sleeves = _sleeves()
    with pytest.raises(ValueError, match="exactly"):
        qga.allocate_quality_gated(
            {sid: spec["returns"] for sid, spec in sleeves.items()},
            {sid: spec["turnover"] for sid, spec in sleeves.items()},
            returns_are_net=dict.fromkeys(sleeves, True),
        )


def test_allocator_rejects_string_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadAllocator:
        def allocate(self, ids: list[str], *args: Any, **kwargs: Any) -> dict[str, str]:
            return dict.fromkeys(ids, "0.5")

    monkeypatch.setattr(qga, "_build_allocator", lambda *args, **kwargs: _BadAllocator())
    sleeves = _sleeves()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        qga.allocate_quality_gated(
            {sid: spec["returns"] for sid, spec in sleeves.items()},
            {sid: spec["turnover"] for sid, spec in sleeves.items()},
            returns_are_net=dict.fromkeys(sleeves, True),
        )


def test_manifest_identity_and_source_contracts_are_exact() -> None:
    sleeves = _sleeves()
    with pytest.raises(ValueError, match="source artifact id"):
        qga.build_allocation_manifest(sleeves, source_artifacts=[{"id": 1}])
    with pytest.raises(ValueError, match="returns_are_net"):
        qga.build_allocation_manifest(
            {**sleeves, "a": {**sleeves["a"], "returns_are_net": np.bool_(True)}},
            source_artifacts=_sources(),
        )
    with pytest.raises(ValueError, match="strategy_class"):
        qga.build_allocation_manifest(
            {**sleeves, "a": {**sleeves["a"], "symbols": "BTC/USDT"}},
            source_artifacts=_sources(),
        )
    for malformed in ("", 0, False):
        with pytest.raises(ValueError, match="source_artifact_id"):
            qga.build_allocation_manifest(
                {
                    **sleeves,
                    "a": {**sleeves["a"], "source_artifact_id": malformed},
                },
                source_artifacts=_sources(),
            )
