"""Tests for the config-gated MR1 extensions of the offline M2 allocator.

Lane MR1 adds two OPTIONAL, default-OFF knobs to
:mod:`lumina_quant.portfolio.quality_gated_allocation`:

* ``turnover_penalty_lambda`` -- the sleeve quality score becomes
  ``net_sharpe - lambda * turnover`` (Frazzini-Israel-Moskowitz 2012), which
  both gates and tilts the allocation against high-turnover sleeves.
* ``correlation_shrinkage`` -- a hand-rolled Ledoit-Wolf (2004) shrinkage of the
  HRP correlation linkage toward the identity (pure numpy, NO sklearn/scipy).

The load-bearing guarantee is byte-identity when BOTH knobs are OFF: the emitted
manifest must be indistinguishable from the pre-extension output. These tests
cover (a) flags-OFF byte-identity, (b) turnover-penalty down-weight/gate-out,
(c) hand-rolled correlation shrinkage (well-formed + closed-form + HRP effect),
(d) a REAL round-trip through the live ``ArtifactPortfolioModeStrategy`` consumer
(flags ON and OFF, no fail-close), and (e) run-twice determinism.

All synthetic streams come from a deterministic linear congruential generator
(no numpy RNG, no scipy) so every test is bit-for-bit reproducible. ASCII only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lumina_quant.portfolio.optimizers_extra import HRPPortfolio
from lumina_quant.portfolio.quality_gated_allocation import (
    _hand_rolled_lw_correlation_shrinkage,
    _hrp_weights_with_correlation_shrinkage,
    allocate_quality_gated,
    build_allocation_manifest,
    compute_sleeve_quality,
)

# Load the live consumer the same way the sibling M2 test does: import the
# module file directly so the round-trip exercises the REAL fail-closed gate.
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lumina_quant"
    / "strategies"
    / "artifact_portfolio_mode.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "artifact_portfolio_mode_for_turnover_tests", MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
ARTIFACT_PORTFOLIO_MODE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ARTIFACT_PORTFOLIO_MODE
_SPEC.loader.exec_module(ARTIFACT_PORTFOLIO_MODE)


def _lcg_stream(seed: int, n: int) -> list[float]:
    """Deterministic linear-congruential PRNG stream in [0, 1) (Numerical Recipes)."""
    x = seed
    values: list[float] = []
    for _ in range(n):
        x = (1664525 * x + 1013904223) % (2**32)
        values.append(x / (2**32))
    return values


def _synthetic_returns(seed: int, n: int, *, drift: float, scale: float) -> list[float]:
    return [(value - 0.5) * scale + drift for value in _lcg_stream(seed, n)]


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_artifacts(tmp_path: Path) -> list[dict[str, Any]]:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps({"ready": True}), encoding="utf-8")
    return [
        {
            "id": "src",
            "path": str(source_path),
            "sha256": _sha256_of(source_path),
            "max_age_hours": 876_000,
            "ready": True,
            "portfolio_ready": True,
        }
    ]


def _three_positive_sleeves() -> dict[str, dict[str, Any]]:
    """Three strongly-positive-edge sleeves with distinct turnovers."""
    return {
        "alpha": {
            "returns": _synthetic_returns(101, 80, drift=0.004, scale=0.01),
            "turnover": 0.01,
            "strategy_class": "MovingAverageCrossStrategy",
            "symbols": ["BTC/USDT"],
            "params": {"short_window": 4, "long_window": 12},
            "source_artifact_id": "src",
        },
        "beta": {
            "returns": _synthetic_returns(202, 80, drift=0.004, scale=0.012),
            "turnover": 0.02,
            "strategy_class": "MovingAverageCrossStrategy",
            "symbols": ["ETH/USDT"],
            "params": {"short_window": 6, "long_window": 20},
            "source_artifact_id": "src",
        },
        "gamma": {
            "returns": _synthetic_returns(303, 80, drift=0.004, scale=0.014),
            "turnover": 0.03,
            "strategy_class": "MovingAverageCrossStrategy",
            "symbols": ["SOL/USDT"],
            "params": {"short_window": 8, "long_window": 24},
            "source_artifact_id": "src",
        },
    }


# ---------------------------------------------------------------------------
# (a) Flags-OFF byte-identity.
# ---------------------------------------------------------------------------


def test_manifest_is_byte_identical_when_new_flags_default(tmp_path: Path) -> None:
    sleeves = _three_positive_sleeves()
    source_artifacts = _source_artifacts(tmp_path)

    default = build_allocation_manifest(
        sleeves, source_artifacts=source_artifacts, method="hrp", gross_cap=1.0
    )
    explicit_off = build_allocation_manifest(
        sleeves,
        source_artifacts=source_artifacts,
        method="hrp",
        gross_cap=1.0,
        turnover_penalty_lambda=0.0,
        correlation_shrinkage=None,
    )

    assert json.dumps(default, sort_keys=True) == json.dumps(explicit_off, sort_keys=True)
    # OFF must not inject the penalty provenance field anywhere.
    assert all("quality_score" not in row for row in default["sleeve_quality"].values())


def test_allocate_is_byte_identical_when_new_flags_default() -> None:
    sleeve_returns = {
        "a": _synthetic_returns(1, 80, drift=0.004, scale=0.006),
        "b": _synthetic_returns(2, 80, drift=0.004, scale=0.02),
    }
    turnovers = {"a": 0.05, "b": 0.05}

    for method in ("erc", "hrp"):
        default = allocate_quality_gated(sleeve_returns, turnovers, method=method)
        explicit_off = allocate_quality_gated(
            sleeve_returns,
            turnovers,
            method=method,
            turnover_penalty_lambda=0.0,
            correlation_shrinkage=None,
        )
        assert default == explicit_off


def test_compute_sleeve_quality_off_has_no_penalty_field() -> None:
    returns = _synthetic_returns(9, 64, drift=0.003, scale=0.01)
    default = compute_sleeve_quality(returns, 0.05)
    explicit_off = compute_sleeve_quality(returns, 0.05, turnover_penalty_lambda=0.0)
    assert default == explicit_off
    assert "quality_score" not in default


def test_compute_sleeve_quality_on_exposes_penalized_score() -> None:
    returns = _synthetic_returns(9, 64, drift=0.003, scale=0.01)
    scored = compute_sleeve_quality(returns, 0.05, turnover_penalty_lambda=3.0)
    expected = round(scored["net_sharpe"] - 3.0 * 0.05, 10)
    assert scored["quality_score"] == expected


# ---------------------------------------------------------------------------
# (b) Turnover penalty ON.
# ---------------------------------------------------------------------------


def test_turnover_penalty_down_weights_the_higher_turnover_sleeve() -> None:
    # Identical gross series -> equal gross Sharpe AND equal variance, so ERC
    # would split 50/50; only the turnover differs.
    base = _synthetic_returns(7, 80, drift=0.004, scale=0.01)
    sleeve_returns = {"lo": list(base), "hi": list(base)}
    turnovers = {"lo": 0.01, "hi": 0.20}

    off = allocate_quality_gated(sleeve_returns, turnovers, method="erc")
    on = allocate_quality_gated(
        sleeve_returns, turnovers, method="erc", turnover_penalty_lambda=2.0
    )

    assert off["hi"] == pytest.approx(off["lo"], abs=1e-9)  # OFF: symmetric split
    assert on["hi"] < on["lo"]  # ON: high-turnover sleeve strictly down-weighted
    assert sum(on.values()) == pytest.approx(1.0, abs=1e-9)


def test_turnover_penalty_gates_out_a_high_turnover_marginal_sleeve() -> None:
    base = _synthetic_returns(7, 80, drift=0.004, scale=0.01)
    sleeve_returns = {"lo": list(base), "hi": list(base)}
    # lo has zero turnover (always survives); a large lambda drives hi's
    # penalized score below zero so it is gated out entirely.
    turnovers = {"lo": 0.0, "hi": 0.20}

    on = allocate_quality_gated(
        sleeve_returns, turnovers, method="erc", turnover_penalty_lambda=200.0
    )

    assert on == {"lo": 1.0}


# ---------------------------------------------------------------------------
# (c) Hand-rolled Ledoit-Wolf correlation shrinkage.
# ---------------------------------------------------------------------------


def test_shrunk_correlation_is_well_formed() -> None:
    matrix = np.column_stack(
        [
            _synthetic_returns(11, 60, drift=0.0, scale=0.02),
            _synthetic_returns(12, 60, drift=0.0, scale=0.02),
            _synthetic_returns(13, 60, drift=0.0, scale=0.02),
        ]
    )
    shrunk, intensity, std = _hand_rolled_lw_correlation_shrinkage(matrix, True)

    assert shrunk.shape == (3, 3)
    assert np.allclose(np.diag(shrunk), 1.0)  # diagonal exactly 1
    assert np.allclose(shrunk, shrunk.T)  # symmetric
    # Convex blend of two PSD matrices -> PSD (no negative eigenvalues).
    assert np.min(np.linalg.eigvalsh(shrunk)) >= -1e-12
    assert 0.0 <= intensity <= 1.0
    assert np.all(std >= 0.0)


def test_hand_rolled_lw_matches_closed_form_on_tiny_case() -> None:
    # Two columns with a moderate positive correlation and enough observations
    # that the analytic Ledoit-Wolf intensity toward the identity is strictly
    # interior (0 < delta < 1), so the match is non-trivial.
    a = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.015, -0.005, 0.025])
    b = np.array([0.018, -0.006, 0.028, 0.004, -0.015, 0.012, 0.002, 0.02])
    matrix = np.column_stack([a, b])
    t = matrix.shape[0]

    shrunk, intensity, _std = _hand_rolled_lw_correlation_shrinkage(matrix, True)

    demeaned = matrix - matrix.mean(axis=0)
    sample_cov = (demeaned.T @ demeaned) / t
    r = sample_cov[0, 1] / np.sqrt(sample_cov[0, 0] * sample_cov[1, 1])
    expected_delta = min(1.0, max(0.0, (1.0 - r**2) ** 2 / (t * r**2)))

    assert 0.0 < intensity < 1.0  # strictly interior -> meaningful match
    assert intensity == pytest.approx(expected_delta, abs=1e-12)
    assert shrunk[0, 1] == pytest.approx((1.0 - expected_delta) * r, abs=1e-12)


def test_correlation_shrinkage_changes_hrp_weights_on_crafted_fixture() -> None:
    # Craft x, y sharing a common factor (strongly correlated) and z independent.
    def _noise(seed: int, n: int, scale: float) -> np.ndarray:
        return np.array([(v - 0.5) * scale for v in _lcg_stream(seed, n)])

    n = 80
    common = _noise(1, n, 0.02)
    x = common + _noise(2, n, 0.004) + 0.003
    y = common + _noise(3, n, 0.004) + 0.003
    z = _noise(4, n, 0.02) + 0.003
    matrix = np.column_stack([x, y, z])
    ids = ["x", "y", "z"]

    off = HRPPortfolio().allocate(ids, matrix)
    on = _hrp_weights_with_correlation_shrinkage(ids, matrix, correlation_shrinkage=0.6)

    assert set(on) == set(ids)
    assert sum(on.values()) == pytest.approx(1.0, abs=1e-9)
    # Shrinking the strong x-y linkage toward the identity materially moves the
    # HRP allocation (not a 1e-10 rounding wobble).
    assert max(abs(off[key] - on[key]) for key in ids) > 0.05


def test_allocate_hrp_shrinkage_only_active_via_flag() -> None:
    # a and b share a common factor (strongly correlated) so shrinking the HRP
    # linkage toward the identity measurably moves the allocation through the full
    # allocate_quality_gated HRP path; c is independent.
    def _factor(seed: int, scale: float) -> np.ndarray:
        return np.array([(v - 0.5) * scale for v in _lcg_stream(seed, 80)])

    common = _factor(31, 0.02)
    sleeve_returns = {
        "a": (common + _factor(32, 0.004) + 0.003).tolist(),
        "b": (common + _factor(33, 0.004) + 0.003).tolist(),
        "c": (_factor(34, 0.02) + 0.003).tolist(),
    }
    turnovers = dict.fromkeys(sleeve_returns, 0.0)

    off = allocate_quality_gated(sleeve_returns, turnovers, method="hrp")
    on = allocate_quality_gated(sleeve_returns, turnovers, method="hrp", correlation_shrinkage=0.7)
    # ERC ignores correlation_shrinkage (linkage is HRP-only) -> byte-identical.
    erc_off = allocate_quality_gated(sleeve_returns, turnovers, method="erc")
    erc_shrink = allocate_quality_gated(
        sleeve_returns, turnovers, method="erc", correlation_shrinkage=0.7
    )

    assert on != off
    assert erc_off == erc_shrink


# ---------------------------------------------------------------------------
# (d) REAL round-trip through the live consumer (flags ON and OFF).
# ---------------------------------------------------------------------------


def _resolve(manifest: dict[str, Any], tmp_path: Path, name: str) -> Any:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return ARTIFACT_PORTFOLIO_MODE.resolve_portfolio_mode_definition(f"manifest:{path}")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"turnover_penalty_lambda": 1.0},
        {"correlation_shrinkage": True},
        {"turnover_penalty_lambda": 1.0, "correlation_shrinkage": True},
    ],
)
def test_manifest_round_trips_through_real_consumer(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    sleeves = _three_positive_sleeves()
    manifest = build_allocation_manifest(
        sleeves,
        source_artifacts=_source_artifacts(tmp_path),
        method="hrp",
        gross_cap=1.0,
        **kwargs,
    )

    definition = _resolve(manifest, tmp_path, "manifest")

    assert "manifest_fail_closed_to_cash" not in definition.source_artifacts
    assert definition.cash_weight != 1.0
    assert definition.components  # at least one sleeve survived and is live


def test_manifest_on_constructs_the_real_strategy_off_cash(tmp_path: Path) -> None:
    sleeves = _three_positive_sleeves()
    manifest = build_allocation_manifest(
        sleeves,
        source_artifacts=_source_artifacts(tmp_path),
        method="hrp",
        gross_cap=1.0,
        turnover_penalty_lambda=1.0,
        correlation_shrinkage=True,
    )
    path = tmp_path / "manifest_on.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    strategy = ARTIFACT_PORTFOLIO_MODE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(
            symbol_list=["BTC/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0
        ),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode=f"manifest:{path}",
    )

    assert strategy.definition.cash_weight != 1.0
    assert "manifest_fail_closed_to_cash" not in strategy.definition.source_artifacts
    assert strategy.definition.components


# ---------------------------------------------------------------------------
# (e) Determinism.
# ---------------------------------------------------------------------------


def test_penalty_and_shrinkage_are_deterministic_run_twice(tmp_path: Path) -> None:
    sleeve_returns = {
        "a": _synthetic_returns(41, 80, drift=0.004, scale=0.01),
        "b": _synthetic_returns(42, 80, drift=0.004, scale=0.012),
        "c": _synthetic_returns(43, 80, drift=0.004, scale=0.014),
    }
    turnovers = {"a": 0.05, "b": 0.10, "c": 0.15}

    first = allocate_quality_gated(
        sleeve_returns,
        turnovers,
        method="hrp",
        turnover_penalty_lambda=1.5,
        correlation_shrinkage=True,
    )
    second = allocate_quality_gated(
        sleeve_returns,
        turnovers,
        method="hrp",
        turnover_penalty_lambda=1.5,
        correlation_shrinkage=True,
    )
    assert first == second

    sleeves = _three_positive_sleeves()
    source_artifacts = _source_artifacts(tmp_path)
    manifest_a = build_allocation_manifest(
        sleeves,
        source_artifacts=source_artifacts,
        method="hrp",
        turnover_penalty_lambda=1.5,
        correlation_shrinkage=True,
    )
    manifest_b = build_allocation_manifest(
        sleeves,
        source_artifacts=source_artifacts,
        method="hrp",
        turnover_penalty_lambda=1.5,
        correlation_shrinkage=True,
    )
    assert json.dumps(manifest_a, sort_keys=True) == json.dumps(manifest_b, sort_keys=True)
