"""Risk-scaling layer: target-vol / gated fractional-Kelly exposure above the allocator."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lumina_quant.portfolio.quality_gated_allocation import (
    _materialized_return_panel_sha256,
    allocate_quality_gated,
    build_allocation_manifest,
)
from lumina_quant.portfolio.risk_scaling import (
    compute_risk_scaling,
    kelly_mu_sensitivity,
    resolve_risk_scaling_spec,
)

_IDS = ["a", "b", "c"]
_EQUAL = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def _panel(rows: int = 500, mean: float = 0.0005, scale: float = 0.01) -> np.ndarray:
    return np.random.default_rng(0).normal(mean, scale, size=(rows, len(_IDS)))


def _portfolio_sigma(matrix: np.ndarray) -> float:
    return float(np.std(matrix @ (np.ones(len(_IDS)) / len(_IDS)), ddof=1))


def test_target_vol_exposure_matches_hand_math_and_caps_at_max_leverage() -> None:
    matrix = _panel()
    sigma = _portfolio_sigma(matrix)
    spec = {"method": "target_vol", "sigma_target_annual": 0.10, "bars_per_year": 365}
    result = compute_risk_scaling(_EQUAL, _IDS, matrix, spec=spec)
    expected = min(1.0, (0.10 / math.sqrt(365)) / sigma)
    assert result.exposure == pytest.approx(expected, abs=1e-10)
    assert result.cash_weight == pytest.approx(1.0 - expected, abs=1e-10)
    assert result.sigma_per_bar == pytest.approx(sigma, abs=1e-12)
    assert result.sigma_annual == pytest.approx(sigma * math.sqrt(365), abs=1e-10)
    # A per-bar target equal to realized sigma pins L at exactly the 1.0 cap.
    at_cap = compute_risk_scaling(
        _EQUAL, _IDS, matrix, spec={"method": "target_vol", "sigma_target_per_bar": sigma}
    )
    assert at_cap.exposure == pytest.approx(1.0)
    with pytest.raises(ValueError, match="financing"):
        resolve_risk_scaling_spec(
            {"method": "target_vol", "sigma_target_per_bar": sigma * 2, "max_leverage": 1.5}
        )
    # target_vol consumes NO mu estimate: shifting every mean leaves L unchanged.
    shifted = matrix + 0.05
    assert compute_risk_scaling(_EQUAL, _IDS, shifted, spec=spec).exposure == pytest.approx(
        result.exposure, abs=1e-9
    )


def test_target_vol_scaling_is_the_layer_above_the_allocator_weights() -> None:
    matrix = _panel()
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    spec = {"method": "target_vol", "sigma_target_annual": 0.05, "bars_per_year": 365}
    result = compute_risk_scaling(weights, _IDS, matrix, spec=spec)
    stream = matrix @ np.asarray([0.5, 0.3, 0.2])
    assert result.sigma_per_bar == pytest.approx(float(np.std(stream, ddof=1)), abs=1e-12)


def test_fractional_kelly_is_gated_and_matches_mu_over_variance() -> None:
    matrix = _panel()
    with pytest.raises(ValueError, match="gated"):
        resolve_risk_scaling_spec({"method": "fractional_kelly", "fraction": 0.5})
    with pytest.raises(ValueError, match="gated"):
        resolve_risk_scaling_spec(
            {"method": "fractional_kelly", "mu_evidence_confirmed": "yes"}  # not literal True
        )
    spec = {
        "method": "fractional_kelly",
        "fraction": 0.5,
        "mu_evidence_confirmed": True,
    }
    result = compute_risk_scaling(_EQUAL, _IDS, matrix, spec=spec)
    stream = matrix @ (np.ones(3) / 3)
    mu = float(np.mean(stream))
    var = float(np.std(stream, ddof=1)) ** 2
    assert result.exposure == pytest.approx(min(1.0, 0.5 * mu / var), abs=1e-9)
    assert result.diagnostics["full_kelly_exposure"] == pytest.approx(mu / var, abs=1e-9)
    # Negative expected excess return -> zero risky exposure, never short.
    losing = compute_risk_scaling(
        _EQUAL,
        _IDS,
        _panel(mean=-0.001),
        spec={"method": "fractional_kelly", "fraction": 0.5, "mu_evidence_confirmed": True},
    )
    assert losing.exposure == 0.0 and losing.cash_weight == 1.0


def test_kelly_mu_sensitivity_is_linear_while_target_vol_is_invariant() -> None:
    matrix = _panel()
    with pytest.raises(ValueError, match="financing"):
        kelly_mu_sensitivity(_EQUAL, _IDS, matrix, fraction=0.5, max_leverage=50.0)
    sens = kelly_mu_sensitivity(_EQUAL, _IDS, matrix, fraction=0.01, max_leverage=1.0)
    assert sens["mu_x0.5"] == pytest.approx(0.5 * sens["mu_x1"], rel=1e-6)
    assert sens["mu_x1.5"] == pytest.approx(1.5 * sens["mu_x1"], rel=1e-6)


def test_degenerate_inputs_fail_closed() -> None:
    spec = {"method": "target_vol", "sigma_target_annual": 0.10}
    with pytest.raises(ValueError, match="matrix"):
        compute_risk_scaling(_EQUAL, _IDS, None, spec=spec)
    with pytest.raises(ValueError, match="degenerate"):
        compute_risk_scaling(_EQUAL, _IDS, np.zeros((100, 3)), spec=spec)
    with pytest.raises(ValueError, match="observations"):
        compute_risk_scaling(_EQUAL, _IDS, _panel(rows=5), spec=spec)
    with pytest.raises(ValueError, match="positive total"):
        compute_risk_scaling({"a": 0.0, "b": 0.0, "c": 0.0}, _IDS, _panel(), spec=spec)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_risk_scaling_spec({"method": "vol_target"})
    with pytest.raises(ValueError, match="sigma_target"):
        resolve_risk_scaling_spec({"method": "target_vol"})
    with pytest.raises(ValueError, match="max_leverage"):
        resolve_risk_scaling_spec(
            {"method": "target_vol", "sigma_target_annual": 0.1, "max_leverage": 0.0}
        )
    assert resolve_risk_scaling_spec(None) is None
    for malformed in ({}, [], ""):
        with pytest.raises(ValueError):
            resolve_risk_scaling_spec(malformed)


def test_scaling_contract_rejects_misaligned_ids_weights_and_targets() -> None:
    matrix = _panel()
    spec = {"method": "target_vol", "sigma_target_annual": 0.10}
    with pytest.raises(ValueError, match="unique"):
        compute_risk_scaling(_EQUAL, ["a", "a", "c"], matrix, spec=spec)
    with pytest.raises(ValueError, match="exactly cover"):
        compute_risk_scaling({"a": 0.5, "b": 0.5, "extra": 0.0}, _IDS, matrix, spec=spec)
    with pytest.raises(ValueError, match="nonnegative"):
        compute_risk_scaling({"a": 0.5, "b": -0.1, "c": 0.6}, _IDS, matrix, spec=spec)
    with pytest.raises(ValueError, match="gross-normalized"):
        compute_risk_scaling({"a": 0.4, "b": 0.3, "c": 0.2}, _IDS, matrix, spec=spec)
    with pytest.raises(ValueError, match="byte-exactly"):
        resolve_risk_scaling_spec(
            {
                "method": "target_vol",
                "sigma_target_per_bar": 0.01,
                "sigma_target_annual": 0.10,
            }
        )


def _sleeves() -> dict[str, list[float]]:
    rng = np.random.default_rng(1)
    return {f"s{i}": rng.normal(0.001, 0.01, 300).tolist() for i in range(6)}


def _materialized_risk_sleeves(sleeves: dict[str, list[float]]) -> dict[str, dict[str, object]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [
        (start + timedelta(days=index)).isoformat().replace("+00:00", "Z") for index in range(300)
    ]
    return {
        sid: {
            "returns": series,
            "return_timestamps": timestamps,
            "returns_are_net": True,
            "returns_source": "train_validation",
            "fit_start": timestamps[0],
            "fit_end": timestamps[-1],
            "as_of": "2026-11-01T00:00:00Z",
            "apply_start": "2026-11-01T00:00:00Z",
            "turnover": 0.1,
            "family": sid,
            "strategy_class": "RiskScalingTestStrategy",
            "symbols": ["BTC/USDT"],
            "source_artifact_id": "src",
        }
        for sid, series in sleeves.items()
    }


def test_allocator_weights_scale_to_l_preserving_relative_structure() -> None:
    sleeves = _sleeves()
    base = allocate_quality_gated(sleeves, method="hrp_dendrogram", min_families=1)
    out: dict[str, float] = {}
    scaled = allocate_quality_gated(
        sleeves,
        method="hrp_dendrogram",
        min_families=1,
        risk_scaling={"method": "target_vol", "sigma_target_annual": 0.05, "bars_per_year": 365},
        risk_scaling_out=out,
    )
    exposure = out["exposure"]
    assert 0.0 < exposure < 1.0
    assert sum(scaled.values()) == pytest.approx(exposure, abs=1e-8)
    for sleeve_id, weight in base.items():
        assert scaled[sleeve_id] == pytest.approx(weight * exposure, abs=1e-8)


def test_manifest_records_provenance_and_cash_weight_reflects_the_residual() -> None:
    sleeves = _sleeves()
    source = [
        {
            "id": "src",
            "path": "x",
            "sha256": "0" * 64,
            "max_age_hours": 1,
            "ready": True,
            "portfolio_ready": True,
        }
    ]
    spec = _materialized_risk_sleeves(sleeves)
    source[0]["return_panel_sha256_by_sleeve"] = {
        sleeve_id: _materialized_return_panel_sha256(sleeve_id, row)
        for sleeve_id, row in spec.items()
    }
    scaling = {"method": "target_vol", "sigma_target_annual": 0.05, "bars_per_year": 365}
    manifest = build_allocation_manifest(
        spec, source_artifacts=source, method="hrp_dendrogram", min_families=1, risk_scaling=scaling
    )
    exposure = manifest["risk_scaling"]["exposure"]
    assert manifest["risk_scaling"]["method"] == "target_vol"
    assert manifest["risk_scaling"]["spec"] == scaling
    assert manifest["cash_weight"] == pytest.approx(1.0 - exposure, abs=1e-6)
    assert sum(child["weight"] for child in manifest["children"]) == pytest.approx(
        exposure, abs=1e-6
    )
    # Absent block -> no key and full investment (byte-golden pinned elsewhere).
    plain = build_allocation_manifest(spec, source_artifacts=source, method="erc", min_families=1)
    assert "risk_scaling" not in plain
    # Ungated kelly fails closed BEFORE any allocation happens.
    with pytest.raises(ValueError, match="gated"):
        build_allocation_manifest(
            spec,
            source_artifacts=source,
            method="erc",
            min_families=1,
            risk_scaling={"method": "fractional_kelly", "fraction": 0.5},
        )


def test_scaling_applies_after_family_momentum_tilt() -> None:
    sleeves = _sleeves()
    families = {f"s{i}": f"f{i % 3}" for i in range(6)}
    kwargs = dict(
        method="erc",
        min_families=1,
        families=families,
        family_momentum_window=50,
        family_momentum_tilt_strength=0.5,
        family_momentum_tilt_cap=0.30,
    )
    tilted = allocate_quality_gated(_sleeves(), **kwargs)
    out: dict[str, float] = {}
    scaled = allocate_quality_gated(
        sleeves,
        **kwargs,
        risk_scaling={"method": "target_vol", "sigma_target_annual": 0.05, "bars_per_year": 365},
        risk_scaling_out=out,
    )
    exposure = out["exposure"]
    assert 0.0 < exposure < 1.0
    # The tilt ran FIRST: scaled weights are exactly tilted * L per sleeve.
    for sleeve_id, weight in tilted.items():
        assert scaled[sleeve_id] == pytest.approx(weight * exposure, abs=1e-8)


def test_constrained_hrp_bounds_hold_on_the_relative_layer_after_scaling() -> None:
    sleeves = _sleeves()
    params = {"lower": 0.05, "upper_bound": 0.5}
    out: dict[str, float] = {}
    scaled = allocate_quality_gated(
        sleeves,
        method="constrained_hrp",
        min_families=1,
        allocator_params=params,
        risk_scaling={"method": "target_vol", "sigma_target_annual": 0.05, "bars_per_year": 365},
        risk_scaling_out=out,
    )
    exposure = out["exposure"]
    relative = {sleeve_id: weight / exposure for sleeve_id, weight in scaled.items()}
    # Bounds are contracts on the PRE-scaling sum-1 relative weights.
    assert sum(relative.values()) == pytest.approx(1.0, abs=1e-6)
    assert min(relative.values()) >= 0.05 - 1e-9
    assert max(relative.values()) <= 0.5 + 1e-9


def test_confirmed_fractional_kelly_builds_a_manifest_end_to_end() -> None:
    """The documented unlock path must actually work through the builder (idempotent resolve)."""
    sleeves = _sleeves()
    source = [
        {
            "id": "src",
            "path": "x",
            "sha256": "0" * 64,
            "max_age_hours": 1,
            "ready": True,
            "portfolio_ready": True,
        }
    ]
    spec = _materialized_risk_sleeves(sleeves)
    source[0]["return_panel_sha256_by_sleeve"] = {
        sleeve_id: _materialized_return_panel_sha256(sleeve_id, row)
        for sleeve_id, row in spec.items()
    }
    manifest = build_allocation_manifest(
        spec,
        source_artifacts=source,
        method="erc",
        min_families=1,
        risk_scaling={
            "method": "fractional_kelly",
            "fraction": 0.25,
            "mu_evidence_confirmed": True,
        },
    )
    assert manifest["risk_scaling"]["method"] == "fractional_kelly"
    assert manifest["risk_scaling"]["exposure"] >= 0.0
    # Idempotent resolution: a resolved spec re-resolves without tripping the gate.
    once = resolve_risk_scaling_spec(
        {"method": "fractional_kelly", "fraction": 0.25, "mu_evidence_confirmed": True}
    )
    assert resolve_risk_scaling_spec(once) == once


def test_max_leverage_above_one_fails_closed_at_build_time() -> None:
    sleeves = _sleeves()
    timestamps = [
        (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index)).isoformat()
        for index in range(len(next(iter(sleeves.values()))))
    ]
    apply_start = (datetime.fromisoformat(timestamps[-1]) + timedelta(days=1)).isoformat()
    source = [
        {
            "id": "src",
            "path": "x",
            "sha256": "0" * 64,
            "max_age_hours": 1,
            "ready": True,
            "portfolio_ready": True,
        }
    ]
    spec = {
        sid: {
            "returns": series,
            "return_timestamps": timestamps,
            "returns_are_net": True,
            "returns_source": {"splits": ["train", "validation"]},
            "turnover": 0.1,
            "family": sid,
            "strategy_class": "RiskScalingTestStrategy",
            "symbols": ["BTC/USDT"],
            "source_artifact_id": "src",
            "fit_start": timestamps[0],
            "fit_end": timestamps[-1],
            "as_of": apply_start,
            "apply_start": apply_start,
        }
        for sid, series in sleeves.items()
    }
    source[0]["return_panel_sha256_by_sleeve"] = {
        sleeve_id: _materialized_return_panel_sha256(sleeve_id, row)
        for sleeve_id, row in spec.items()
    }
    with pytest.raises(ValueError, match="financing"):
        build_allocation_manifest(
            spec,
            source_artifacts=source,
            method="erc",
            min_families=1,
            risk_scaling={
                "method": "target_vol",
                "sigma_target_annual": 0.30,
                "max_leverage": 1.5,
            },
        )
