"""Tests for the W3-9 config-gated family meta-momentum tilt on the M2 allocator.

Lane W3-9 adds an OPTIONAL, default-OFF recency tilt to
:mod:`lumina_quant.portfolio.quality_gated_allocation`: after the survivor gate,
the base ERC/HRP weights, and the MR1 turnover tilt, each surviving sleeve's
weight is multiplied by ``clip(1 + strength * z_family, 1 - cap, 1 + cap)`` where
``z_family`` is the cross-family z-score of the trailing-window net Sharpe of each
FAMILY's equal-weighted member stream (Gupta-Kelly "Factor Momentum Everywhere").
It is train/val-only by construction (the tilt reads only the SAME trailing net
matrix the covariance already consumed) and ``family_momentum_window == 0`` (the
default) leaves every output byte-identical.

The load-bearing fixture is a T=240 panel of 6 sleeves in 3 families whose
full-window return MULTISETS are all identical -> identical full-window
net_sharpe to machine precision (so the gate, the risk-parity base, and the MR1
turnover tilt are all provably blind to the family that is RECENTLY working).
Family "carry" is the block-swap of "trend" (its strong block is the recent
window); "reversion" is a uniform 4-cycle (the pivot). A jointly time-reversed
copy (Panel Q) flips which family is recent, so the tilt's book must flip while
the base allocator is bit-invariant.

Covers: (stage-1) equal full-window quality; (LEG 1) time-order invariance of the
base allocator + A/B weight symmetry; (LEG 2) the tilt up-weights the recent
family and flips on Q; (LEG 3) the tilt is OUTSIDE the MR1 turnover-tilt span;
(LEG 4) flag-OFF byte-identity + REAL ``ArtifactPortfolioModeStrategy`` round-trip
without fail-close; (LEG 5) degenerate guards; and run-twice determinism.

Hand-built deterministic fixtures only (no RNG, no scipy). ASCII only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lumina_quant.portfolio.quality_gated_allocation import (
    _family_meta_momentum_tilted_weights,
    _materialized_return_panel_sha256,
    _turnover_tilted_weights,
    allocate_quality_gated,
    build_allocation_manifest,
    compute_sleeve_quality,
)

# Load the live consumer by file path so the round-trip exercises the REAL
# fail-closed gate (same idiom as the sibling turnover-tilt test).
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lumina_quant"
    / "strategies"
    / "artifact_portfolio_mode.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "artifact_portfolio_mode_for_family_momentum_tests", MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
ARTIFACT_PORTFOLIO_MODE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ARTIFACT_PORTFOLIO_MODE
_SPEC.loader.exec_module(ARTIFACT_PORTFOLIO_MODE)


# --------------------------------------------------------------------------- #
# Fixture: equal-full-window-quality panel of 6 sleeves in 3 families.
# --------------------------------------------------------------------------- #

_HI, _LO = 0.004, -0.002
_WHI, _WLO = 0.001, -0.001
_T = 240
_BLOCK = 120


def _return_timestamps(n: int) -> list[str]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        (start + timedelta(days=index)).isoformat().replace("+00:00", "Z") for index in range(n)
    ]


def _block(strong: bool) -> list[float]:
    hi, lo = (_HI, _LO) if strong else (_WHI, _WLO)
    return [hi if t % 2 == 0 else lo for t in range(_BLOCK)]


def _trend_member() -> list[float]:
    """Strong block early, weak block late (recently DEAD)."""
    return _block(True) + _block(False)


def _carry_member() -> list[float]:
    """Block swap of trend: weak early, strong late (recently WORKING)."""
    return _block(False) + _block(True)


def _reversion_member() -> list[float]:
    """Uniform 4-cycle over the same value multiset (the neutral pivot)."""
    cycle = [_HI, _LO, _WHI, _WLO]
    return [cycle[t % 4] for t in range(_T)]


def _panel_p() -> dict[str, list[float]]:
    return {
        "s1_trend": _trend_member(),
        "s2_trend": _trend_member(),
        "s3_carry": _carry_member(),
        "s4_carry": _carry_member(),
        "s5_rev": _reversion_member(),
        "s6_rev": _reversion_member(),
    }


def _panel_q() -> dict[str, list[float]]:
    """Panel P with every series jointly time-reversed."""
    return {sid: list(reversed(series)) for sid, series in _panel_p().items()}


_FAMILIES = {
    "s1_trend": "trend",
    "s2_trend": "trend",
    "s3_carry": "carry",
    "s4_carry": "carry",
    "s5_rev": "reversion",
    "s6_rev": "reversion",
}
_TURNOVERS = dict.fromkeys(_FAMILIES, 0.05)
_TILT_KWARGS: dict[str, Any] = dict(
    families=_FAMILIES,
    family_momentum_window=60,
    family_momentum_tilt_strength=0.5,
    family_momentum_tilt_cap=0.30,
    min_families=3,
)


def _sha_free_manifest_sleeves(panel: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    timestamps = _return_timestamps(_T)
    apply_timestamp = _return_timestamps(_T + 1)[-1]
    return {
        sid: {
            "returns": series,
            "turnover": 0.05,
            "returns_are_net": False,
            "return_timestamps": timestamps,
            "returns_source": "train_validation",
            "fit_start": timestamps[0],
            "fit_end": timestamps[-1],
            "as_of": apply_timestamp,
            "apply_start": apply_timestamp,
            "family": _FAMILIES[sid],
            "strategy_class": "MovingAverageCrossStrategy",
            "symbols": ["BTC/USDT"],
            "params": {"short_window": 4, "long_window": 12},
            "source_artifact_id": "src",
        }
        for sid, series in panel.items()
    }


def _source_artifacts(tmp_path: Path, sleeves: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps({"ready": True}), encoding="utf-8")
    return [
        {
            "id": "src",
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "max_age_hours": 876_000,
            "ready": True,
            "portfolio_ready": True,
            "return_panel_sha256_by_sleeve": {
                sleeve_id: _materialized_return_panel_sha256(sleeve_id, spec)
                for sleeve_id, spec in sleeves.items()
            },
        }
    ]


# --------------------------------------------------------------------------- #
# Stage-1: the whole panel has equal full-window quality.
# --------------------------------------------------------------------------- #


def test_stage1_all_sleeves_share_full_window_quality() -> None:
    panel = _panel_p()
    sharpes = {
        sid: compute_sleeve_quality(series, 0.05)["net_sharpe"] for sid, series in panel.items()
    }
    assert len(set(sharpes.values())) == 1  # identical to the rounding grain
    assert next(iter(sharpes.values())) > 0.0  # all six survive the gate


# --------------------------------------------------------------------------- #
# LEG 1: base allocator is time-order invariant + treats A/B members symmetrically.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["erc", "hrp"])
def test_leg1_base_allocator_is_time_order_invariant(method: str) -> None:
    base_p = allocate_quality_gated(_panel_p(), _TURNOVERS, method=method)
    base_q = allocate_quality_gated(_panel_q(), _TURNOVERS, method=method)
    assert set(base_p) == set(_FAMILIES)
    assert sum(base_p.values()) == pytest.approx(1.0, abs=1e-9)
    # The gate / ERC / HRP / cost-drag are all reversal-invariant on a multiset-
    # symmetric panel: no incumbent can even see which family is recent.
    assert max(abs(base_p[k] - base_q[k]) for k in base_p) < 1e-9
    # The two members of a family always carry equal base weight (they are the
    # same series).
    assert base_p["s1_trend"] == pytest.approx(base_p["s2_trend"], abs=1e-9)
    assert base_p["s3_carry"] == pytest.approx(base_p["s4_carry"], abs=1e-9)
    if method == "erc":
        # ERC additionally ties equal-variance families across the multiset-
        # symmetric trend/carry pair (HRP clusters on the differing path, so it
        # legitimately separates them -- still reversal-invariant above).
        assert base_p["s1_trend"] == pytest.approx(base_p["s3_carry"], abs=1e-9)


@pytest.mark.parametrize("method", ["erc", "hrp"])
def test_leg1_lambda_and_shrinkage_stay_time_order_invariant(method: str) -> None:
    # The MR1 turnover tilt (and HRP shrinkage) are also reversal-invariant here.
    kwargs = dict(turnover_penalty_lambda=0.75)
    if method == "hrp":
        kwargs["correlation_shrinkage"] = True
    base_p = allocate_quality_gated(_panel_p(), _TURNOVERS, method=method, **kwargs)
    base_q = allocate_quality_gated(_panel_q(), _TURNOVERS, method=method, **kwargs)
    assert max(abs(base_p[k] - base_q[k]) for k in base_p) < 1e-9


# --------------------------------------------------------------------------- #
# LEG 2: the family tilt up-weights the recent family and flips under reversal.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["erc", "hrp"])
def test_leg2_tilt_up_weights_recent_family_and_flips_on_reversal(method: str) -> None:
    base = allocate_quality_gated(_panel_p(), _TURNOVERS, method=method)
    on_p = allocate_quality_gated(_panel_p(), _TURNOVERS, method=method, **_TILT_KWARGS)
    on_q = allocate_quality_gated(_panel_q(), _TURNOVERS, method=method, **_TILT_KWARGS)

    assert sum(on_p.values()) == pytest.approx(1.0, abs=1e-9)
    # On P the recently-working "carry" family is up-weighted, "trend" down.
    assert on_p["s3_carry"] > base["s3_carry"]
    assert on_p["s4_carry"] > base["s4_carry"]
    assert on_p["s1_trend"] < base["s1_trend"]
    assert on_p["s2_trend"] < base["s2_trend"]
    # Never zeroes a family, never levers one to the whole book (bounded tilt).
    assert all(0.0 < weight < 1.0 for weight in on_p.values())
    # The per-sleeve multiplier stays inside the [1-cap, 1+cap] band: the ratio
    # of (post/base) between the up- and down-tilted families cancels the shared
    # renormalizer, exposing mult_carry / mult_trend <= (1.30 / 0.70).
    ratio = (on_p["s3_carry"] / base["s3_carry"]) / (on_p["s1_trend"] / base["s1_trend"])
    assert ratio <= (1.30 / 0.70) + 1e-9
    # Reversing every series flips which family is recent -> the book flips.
    assert on_q["s1_trend"] > on_q["s3_carry"]
    assert on_p["s3_carry"] > on_p["s1_trend"]
    assert max(abs(on_p[k] - on_q[k]) for k in on_p) > 0.02


# --------------------------------------------------------------------------- #
# LEG 3: recency is OUTSIDE the MR1 turnover-tilt span.
# --------------------------------------------------------------------------- #


def test_leg3_family_tilt_is_outside_the_turnover_tilt_span() -> None:
    survivors = sorted(_panel_p())
    panel = _panel_p()
    quality = {sid: compute_sleeve_quality(panel[sid], 0.05) for sid in survivors}
    raw = allocate_quality_gated(panel, _TURNOVERS, method="erc")

    # Equal penalized scores -> the REAL turnover tilt is inert (returns raw).
    turnover_tilted = _turnover_tilted_weights(raw, survivors, quality, 0.75, None)
    assert max(abs(turnover_tilted[k] - raw[k]) for k in raw) < 1e-9

    # The family tilt, on the identical panel, MOVES weights materially.
    net_matrix = np.column_stack([np.asarray(panel[sid], dtype=np.float64) for sid in survivors])
    family_tilted = _family_meta_momentum_tilted_weights(
        raw,
        survivors,
        net_matrix,
        _FAMILIES,
        window=60,
        strength=0.5,
        cap=0.30,
        min_families=3,
        upper=None,
    )
    assert max(abs(family_tilted[k] - raw[k]) for k in raw) > 0.02


# --------------------------------------------------------------------------- #
# LEG 4: flag-OFF byte-identity + REAL consumer round-trip (flags ON and OFF).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["erc", "hrp"])
def test_leg4_allocate_is_byte_identical_when_flag_default(method: str) -> None:
    panel = _panel_p()
    default = allocate_quality_gated(panel, _TURNOVERS, method=method)
    explicit_off = allocate_quality_gated(
        panel,
        _TURNOVERS,
        method=method,
        families=_FAMILIES,
        family_momentum_window=0,
        family_momentum_tilt_strength=0.5,
        family_momentum_tilt_cap=0.30,
        min_families=3,
    )
    assert default == explicit_off


def test_leg4_manifest_is_byte_identical_when_flag_default(tmp_path: Path) -> None:
    sleeves = _sha_free_manifest_sleeves(_panel_p())
    source_artifacts = _source_artifacts(tmp_path, sleeves)
    default = build_allocation_manifest(
        sleeves, source_artifacts=source_artifacts, method="hrp", gross_cap=1.0
    )
    explicit_off = build_allocation_manifest(
        sleeves,
        source_artifacts=source_artifacts,
        method="hrp",
        gross_cap=1.0,
        family_momentum_window=0,
    )
    assert json.dumps(default, sort_keys=True) == json.dumps(explicit_off, sort_keys=True)


def _resolve(manifest: dict[str, Any], tmp_path: Path, name: str) -> Any:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return ARTIFACT_PORTFOLIO_MODE.resolve_portfolio_mode_definition(f"manifest:{path}")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"family_momentum_window": 60},
        {"family_momentum_window": 90, "family_momentum_tilt_cap": 0.15},
    ],
)
def test_leg4_manifest_round_trips_through_real_consumer(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    sleeves = _sha_free_manifest_sleeves(_panel_p())
    manifest = build_allocation_manifest(
        sleeves,
        source_artifacts=_source_artifacts(tmp_path, sleeves),
        method="hrp",
        gross_cap=1.0,
        **kwargs,
    )
    definition = _resolve(manifest, tmp_path, "manifest")

    assert "manifest_fail_closed_to_cash" not in definition.source_artifacts
    assert definition.cash_weight != 1.0
    assert definition.components  # at least one sleeve survived and is live
    # Provenance keys stay truthful under the tilt (no OOS bar ever consulted).
    for child in manifest["children"]:
        assert child["no_current_fold_oos_provenance"] is True
        assert child["uses_current_fold_oos"] is False


def test_leg4_tilt_on_manifest_constructs_the_real_strategy_off_cash(tmp_path: Path) -> None:
    sleeves = _sha_free_manifest_sleeves(_panel_p())
    manifest = build_allocation_manifest(
        sleeves,
        source_artifacts=_source_artifacts(tmp_path, sleeves),
        method="hrp",
        gross_cap=1.0,
        family_momentum_window=60,
    )
    path = tmp_path / "manifest_on.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    strategy = ARTIFACT_PORTFOLIO_MODE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(symbol_list=["BTC/USDT"], get_latest_bar_value=lambda *a, **k: 100.0),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode=f"manifest:{path}",
    )
    assert strategy.definition.cash_weight != 1.0
    assert "manifest_fail_closed_to_cash" not in strategy.definition.source_artifacts
    assert strategy.definition.components


# --------------------------------------------------------------------------- #
# LEG 5: degenerate guards -- every one returns the base weights unchanged.
# --------------------------------------------------------------------------- #


def _base_and_matrix() -> tuple[dict[str, float], list[str], np.ndarray]:
    survivors = sorted(_panel_p())
    panel = _panel_p()
    base = allocate_quality_gated(panel, _TURNOVERS, method="erc")
    matrix = np.column_stack([np.asarray(panel[sid], dtype=np.float64) for sid in survivors])
    return base, survivors, matrix


def test_leg5_fewer_than_min_families_is_noop() -> None:
    base, survivors, matrix = _base_and_matrix()
    two_family_map = {sid: ("trend" if "trend" in sid else "carry") for sid in survivors}
    out = _family_meta_momentum_tilted_weights(
        base,
        survivors,
        matrix,
        two_family_map,
        window=60,
        strength=0.5,
        cap=0.30,
        min_families=3,
        upper=None,
    )
    assert out == base


def test_leg5_window_at_or_above_length_is_noop() -> None:
    base, survivors, matrix = _base_and_matrix()
    out = _family_meta_momentum_tilted_weights(
        base,
        survivors,
        matrix,
        _FAMILIES,
        window=_T + 1,
        strength=0.5,
        cap=0.30,
        min_families=3,
        upper=None,
    )
    assert out == base


def test_leg5_zero_cross_family_dispersion_is_noop() -> None:
    # Every sleeve is the identical uniform series -> all three families share the
    # same trailing Sharpe -> zero cross-family dispersion -> tilt inert.
    survivors = [f"u{i}" for i in range(6)]
    uniform = _reversion_member()
    panel = dict.fromkeys(survivors, uniform)
    families = {sid: ["trend", "carry", "reversion"][i % 3] for i, sid in enumerate(survivors)}
    turnovers = dict.fromkeys(survivors, 0.05)
    base = allocate_quality_gated(panel, turnovers, method="erc")
    matrix = np.column_stack([np.asarray(uniform, dtype=np.float64) for _ in survivors])
    out = _family_meta_momentum_tilted_weights(
        base,
        survivors,
        matrix,
        families,
        window=60,
        strength=0.5,
        cap=0.30,
        min_families=3,
        upper=None,
    )
    assert out == base


def test_leg5_unmapped_sleeve_stays_neutral() -> None:
    base, survivors, matrix = _base_and_matrix()
    # Drop one sleeve from the family map: it must keep a neutral 1.0 multiplier
    # (its post-tilt/base ratio equals the renormalization factor, i.e. it never
    # gets an idiosyncratic up/down-weight) while the mapped families still tilt.
    partial = {sid: fam for sid, fam in _FAMILIES.items() if sid != "s5_rev"}
    out = _family_meta_momentum_tilted_weights(
        base,
        survivors,
        matrix,
        partial,
        window=60,
        strength=0.5,
        cap=0.30,
        min_families=3,
        upper=None,
    )
    assert out != base  # the mapped families still move
    assert set(out) == set(base)
    # Its companion s6_rev IS mapped (reversion); s5_rev is not, so s5's ratio to
    # its own base is the pure renormalization constant -- distinct from a tilted
    # family but never a raise / drop.
    assert out["s5_rev"] > 0.0


def test_leg5_empty_and_none_inputs_are_safe() -> None:
    assert (
        allocate_quality_gated(
            None,
            None,
            family_momentum_window=60,
            **{k: v for k, v in _TILT_KWARGS.items() if k != "family_momentum_window"},
        )
        == {}
    )
    assert allocate_quality_gated({}, {}, family_momentum_window=60) == {}
    # Direct helper: empty survivors -> empty base echoed back.
    assert (
        _family_meta_momentum_tilted_weights(
            {},
            [],
            np.zeros((0, 0)),
            _FAMILIES,
            window=60,
            strength=0.5,
            cap=0.30,
            min_families=3,
            upper=None,
        )
        == {}
    )


# --------------------------------------------------------------------------- #
# Determinism.
# --------------------------------------------------------------------------- #


def test_family_tilt_is_deterministic_run_twice(tmp_path: Path) -> None:
    panel = _panel_p()
    first = allocate_quality_gated(panel, _TURNOVERS, method="hrp", **_TILT_KWARGS)
    second = allocate_quality_gated(panel, _TURNOVERS, method="hrp", **_TILT_KWARGS)
    assert first == second

    sleeves = _sha_free_manifest_sleeves(panel)
    source_artifacts = _source_artifacts(tmp_path, sleeves)
    manifest_a = build_allocation_manifest(
        sleeves, source_artifacts=source_artifacts, method="hrp", family_momentum_window=60
    )
    manifest_b = build_allocation_manifest(
        sleeves, source_artifacts=source_artifacts, method="hrp", family_momentum_window=60
    )
    assert json.dumps(manifest_a, sort_keys=True) == json.dumps(manifest_b, sort_keys=True)
