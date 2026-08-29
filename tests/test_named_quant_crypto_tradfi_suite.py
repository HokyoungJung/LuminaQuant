from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import SimpleQueue

import numpy as np

from lumina_quant.configuration.loader import load_runtime_config
from lumina_quant.portfolio.optimizers_extra import HRPPortfolio
from lumina_quant.portfolio.quality_gated_allocation import (
    _materialized_return_panel_sha256,
)
from lumina_quant.research.external_source_registry import validate_source_registry
from lumina_quant.research_universe import (
    BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS,
    BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS,
    compact_to_slashed_usdt,
)
from lumina_quant.strategies.registry import resolve_strategy_class
from lumina_quant.strategy_factory import research_runner
from scripts.run_research_candidates import _score_config_scope
from scripts.research.build_quality_gated_allocation import (
    build_manifest_from_input,
    validate_cell_spec,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/research/named_quant_crypto_tradfi_suite_v1.json"


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = symbols


def _load() -> dict:
    return json.loads(SUITE.read_text(encoding="utf-8"))


def test_suite_is_runnable_research_only_and_source_bounded() -> None:
    suite = _load()
    assert suite["research_only"] is True
    assert suite["promotion_eligible"] is False
    assert suite["allow_real_money"] is False
    score_flags = suite["candidate_research"]["research"]
    assert score_flags["strict_selection_gate"] is True
    assert score_flags["route_unmapped_registered_strategies"] is True
    assert score_flags["emit_candidate_overfit_stats"] is True
    scoped = _score_config_scope(suite)
    assert (
        research_runner._candidate_cost_rate(
            {"strategy_class": "TrendGatedIbsReversionStrategy"}, scoring_config=scoped
        )
        == 0.0011
    )
    limitations = suite["screening_runner_limitations"]
    assert limitations["portfolio_input_allowed"] is False
    assert limitations["models_target_sizing"] is False
    assert limitations["models_next_open_fills"] is False
    evidence_ids = {row["source_id"] for row in suite["evidence_sources"]}
    assert len(suite["supplemental_hypotheses"]) == 3
    for hypothesis in suite["supplemental_hypotheses"]:
        assert hypothesis["status"].startswith("design_only")
        assert hypothesis["promotion_eligible"] is False
        assert set(hypothesis["public_basis_refs"]) <= evidence_ids

    recipes = {row["name"]: row for row in suite["portfolio_recipes"]}
    assert recipes["constrained_correlation_cluster_hrp"]["registry"] == "HRP"
    assert recipes["nested_clustered_optimization"]["status"] == "design_only_no_optimizer"
    assert recipes["hierarchical_equal_risk_contribution"]["status"] == (
        "design_only_no_full_dendrogram"
    )
    assert recipes["wasserstein_dro_research_protocol"]["status"] == (
        "stress_protocol_only_not_a_dro_optimizer"
    )
    assert recipes["deep_rl_portfolio_deferred"]["status"] == "deferred_overfit_risk"
    assert len(suite["candidates"]) == 15
    assert len({row["candidate_id"] for row in suite["candidates"]}) == 15

    crypto = {compact_to_slashed_usdt(s) for s in BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS}
    tradfi = {compact_to_slashed_usdt(s) for s in BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS}
    assert set(suite["universe"]["crypto_top10"]["static_smoke_symbols"]) == crypto
    assert len(crypto) == 10
    assert set(suite["universe"]["tradfi"]["static_smoke_symbols"]) == tradfi

    for row in suite["candidates"]:
        strategy = resolve_strategy_class(row["strategy_class"], strict=True)
        strategy(_Bars(row["symbols"]), SimpleQueue(), **row["params"])
        allowed = crypto if row["candidate_id"].startswith("crypto_") else tradfi
        assert set(row["symbols"]) <= allowed
        assert row["metadata"]["promotion_eligible"] is False
        if set(row["symbols"]) == crypto:
            assert row["metadata"]["universe_binding"] == "crypto_top10"
        if set(row["symbols"]) == tradfi:
            assert row["metadata"]["universe_binding"] == "tradfi_all"
        if row["candidate_id"].startswith("tradfi_"):
            assert row["metadata"]["universe_constraint"] == "tradfi_all"

    decisions = validate_source_registry(suite["evidence_sources"])
    assert decisions and all(decision.allowed for decision in decisions)
    sources = {row["source_id"]: row for row in suite["evidence_sources"]}
    assert sources["amateurquant_profile"]["allowed_usage_label"] == "evidence_only"
    candidates = {row["candidate_id"]: row for row in suite["candidates"]}
    for candidate_id in {
        "crypto_residual_momentum_weekly_v1",
        "tradfi_equity_residual_momentum_v1",
        "tradfi_gold_silver_ratio_reversion_v1",
        "tradfi_metals_relative_value_4h_v1",
    }:
        metadata = candidates[candidate_id]["metadata"]
        assert "amateurquant_profile" not in metadata["hypothesis_refs"]
        assert metadata["provenance_refs"] == ["amateurquant_profile"]
    for candidate_id in {
        "tradfi_gold_silver_ratio_reversion_v1",
        "tradfi_metals_relative_value_4h_v1",
    }:
        metadata = candidates[candidate_id]["metadata"]
        assert metadata["hypothesis_refs"] == []
        assert metadata["rule_origin"] == "independent_hypothesis"
    unresolved = {
        row["requested_label"]
        for row in suite["source_resolution"]
        if row["status"] == "unverified"
    }
    assert unresolved == {"systrader32", "부동심"}


def test_preregistered_hrp_cell_and_cost_profile_are_strict() -> None:
    suite = _load()
    ok, message = validate_cell_spec(suite)
    assert ok, message

    ids = list(suite["sleeves"])
    t = np.linspace(0.0, 12.0 * np.pi, 360)
    returns = np.column_stack(
        [0.001 + 0.002 * np.sin(t * (index + 1) / 3.0 + index) for index in range(len(ids))]
    )
    upper = dict.fromkeys(ids, suite["allocator"]["upper"])
    allocator = HRPPortfolio(corr_threshold=suite["allocator"]["corr_threshold"])
    weights = allocator.allocate(ids, returns, upper=upper)
    assert weights == allocator.allocate(ids, returns, upper=upper)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert max(weights.values()) <= suite["allocator"]["upper"] + 1e-12

    profile = load_runtime_config(ROOT / "configs/profiles/backtest_cost_realistic.yaml")
    assert profile.research.emit_candidate_overfit_stats is True
    assert profile.research.route_unmapped_registered_strategies is True
    assert profile.execution.require_funding_coverage is True
    assert profile.execution.funding_on_utc_boundary is True
    assert profile.execution.slippage_impact_model == "sqrt_impact"
    assert "full event-driven backtester" in suite["source_artifacts"][0]["note"]


def test_materialized_hrp_children_keep_runnable_strategy_definitions() -> None:
    suite = deepcopy(_load())
    t = np.linspace(0.0, 16.0 * np.pi, 480)
    for index, sleeve in enumerate(suite["sleeves"].values()):
        sleeve["returns"] = (0.001 + 0.0005 * np.sin(t * (index + 1) / 5.0 + index)).tolist()
        sleeve["return_timestamps"] = [
            (datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day)).isoformat()
            for day in range(len(t))
        ]
        sleeve["fit_start"] = sleeve["return_timestamps"][0]
        sleeve["fit_end"] = sleeve["return_timestamps"][-1]
        sleeve["as_of"] = (datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=len(t))).isoformat()
        sleeve["apply_start"] = sleeve["as_of"]
        sleeve["returns_are_net"] = True
        sleeve["turnover"] = 0.01
        sleeve["returns_source"] = {"splits": ["train", "validation"]}
    suite["source_artifacts"][0].update(
        {
            "path": "/data/named_quant_train_validation.json",
            "portfolio_ready": True,
            "ready": True,
            "sha256": "a" * 64,
            "return_panel_sha256_by_sleeve": {
                sleeve_id: _materialized_return_panel_sha256(sleeve_id, sleeve)
                for sleeve_id, sleeve in suite["sleeves"].items()
                if sleeve["returns"]
            },
        }
    )

    manifest = build_manifest_from_input(suite)
    assert len(manifest["children"]) >= suite["min_sleeves"]
    for child in manifest["children"]:
        assert child["strategy_class"]
        assert child["symbols"]
        strategy = resolve_strategy_class(child["strategy_class"], strict=True)
        strategy(_Bars(child["symbols"]), SimpleQueue(), **child["params"])
        assert child["uses_locked_oos_for_selection"] is False
        assert child["uses_locked_oos_for_sizing"] is False
