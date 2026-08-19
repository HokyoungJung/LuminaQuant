"""Wiring checks for the Claude-lane named-quant research suite (no data, no performance)."""

from __future__ import annotations

import json
from pathlib import Path
from queue import SimpleQueue

from lumina_quant.research.external_source_registry import validate_source_registry
from lumina_quant.strategies.registry import get_strategy_tier, resolve_strategy_class
from scripts.research.build_named_quant_claude_suite import build_suite
from scripts.research.build_quality_gated_allocation import validate_cell_spec

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/research/named_quant_claude_suite_v1.json"

NEW_STRATEGIES = (
    "NoiseFilteredVolatilityBreakoutStrategy",
    "MaScoreVolTargetRotationStrategy",
    "TurtleUnitPyramidingStrategy",
    "KalmanPairsStatArbStrategy",
    "PcaResidualStatArbStrategy",
    "EquityCurveKillSwitchOverlayStrategy",
    "RsiDivergenceScaleOutStrategy",
    "PrevDayBoxQuartileReversionStrategy",
    "SessionHighBreakoutScalpStrategy",
)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


def _load() -> dict:
    return json.loads(SUITE.read_text(encoding="utf-8"))


def test_committed_json_matches_generator_and_is_research_only() -> None:
    suite = _load()
    regenerated = json.loads(json.dumps(build_suite(), sort_keys=True))
    assert suite == regenerated, "run scripts/research/build_named_quant_claude_suite.py"
    assert suite["research_only"] is True
    assert suite["allow_real_money"] is False
    assert suite["promotion_eligible"] is False
    assert suite["candidate_research"]["research"]["strict_selection_gate"] is True
    assert suite["candidate_research"]["research"]["route_unmapped_registered_strategies"] is True
    ids = [row["candidate_id"] for row in suite["candidates"]]
    assert len(ids) == len(set(ids)) == 25
    assert {row["strategy_class"] for row in suite["candidates"]} == set(NEW_STRATEGIES)


def test_every_new_strategy_is_registered_research_only() -> None:
    for name in NEW_STRATEGIES:
        cls = resolve_strategy_class(name, strict=True)
        assert cls.__name__ == name
        assert get_strategy_tier(name) == "research_only", name


def test_every_candidate_constructs_from_its_params() -> None:
    for row in _load()["candidates"]:
        cls = resolve_strategy_class(row["strategy_class"], strict=True)
        strategy = cls(_Bars(row["symbols"]), SimpleQueue(), **row["params"])
        assert strategy is not None, row["candidate_id"]
        assert row["strategy_timeframe"] in {"1m", "10m", "15m", "1h", "4h", "1d"}
        assert "research_only" in row["tags"] and "claude_lane" in row["tags"]
        assert row["metadata"]["promotion_eligible"] is False
        if len(row["symbols"]) > 1:
            assert "cross_sectional" in row["tags"]


def test_cell_spec_validates_and_declares_hierarchical_variants() -> None:
    suite = _load()
    ok, message = validate_cell_spec(suite)
    assert ok, message
    assert suite["method"] == "hrp_dendrogram"
    methods = {row["method"] for row in suite["allocator_variants"]}
    assert {
        "hrp_dendrogram",
        "constrained_hrp",
        "herc",
        "nco",
        "wasserstein_dro",
        "graph_inverse_centrality",
    } <= methods
    assert {"erc", "hrp"} <= methods  # legacy baselines stay in the comparison set
    assert set(suite["sleeves"]) == {row["candidate_id"] for row in suite["candidates"]}
    for sleeve in suite["sleeves"].values():
        assert sleeve["returns"] is None and sleeve["turnover"] is None
        source = sleeve["returns_source"]
        # Lockbox hygiene: weights / quality gates only ever see train+validation
        # net returns; the locked-OOS segment is never an allocator input.
        assert source["selection_inputs"] == ["train", "validation"]
        assert source["locked_oos_used_for_weights"] is False
        assert "locked-OOS" not in source["stream"].split(";")[0]
    policy = suite["allocation_input_policy"]
    assert policy["weights_and_quality_gate_inputs"] == ["train", "validation"]
    assert policy["locked_oos_used_for_weights"] is False
    assert "compare_hierarchical_allocators.py --variants" in suite["allocator_variants_execution"]
    dro = [row for row in suite["allocator_variants"] if row["method"] == "wasserstein_dro"]
    assert dro and all(row["allocator_params"]["radius"] <= 1e-3 for row in dro)


def test_every_hypothesis_ref_resolves_to_a_registered_evidence_source() -> None:
    suite = _load()
    registry = {row["source_id"] for row in suite["evidence_sources"]}
    assert registry
    # Registry rows are the sibling lane's records verbatim (no new source minted
    # here); the boundary validator must accept every one of them.
    ok_rows = validate_source_registry(suite["evidence_sources"])
    assert all(decision.allowed for decision in ok_rows), [
        (d.source_id, d.reasons) for d in ok_rows if not d.allowed
    ]
    for row in suite["candidates"]:
        refs = row["metadata"]["hypothesis_refs"]
        assert refs, row["candidate_id"]
        unresolved = sorted(set(refs) - registry)
        assert not unresolved, (row["candidate_id"], unresolved)
    for method, refs in suite["allocator_evidence_refs"].items():
        assert set(refs) <= registry, method
    assert "pending_evidence_sources" not in suite  # every cited source is registered
    by_id = {row["source_id"]: row for row in suite["evidence_sources"]}
    # dacapogo is diagnostic/proxy provenance (pinned commit), never rule evidence.
    assert by_id["dacapogo_public_repo_633ba5d"]["allowed_usage_label"] == "diagnostic_only"
    assert (
        "633ba5d6bc0c84a20696af6b2bf807cf55d21248" in by_id["dacapogo_public_repo_633ba5d"]["url"]
    )
    assert "wasserstein_dro_esfahani_kuhn_2018" not in by_id
    assert suite["allocator_evidence_refs"]["wasserstein_dro"] == [
        "blanchet_chen_zhou_2022_wasserstein_mv"
    ]
    # The prev-day box proxy is independent: it must not cite the AOA interview.
    for row in suite["candidates"]:
        if row["strategy_class"] == "PrevDayBoxQuartileReversionStrategy":
            assert "aoa_bitmex_interview_20250611" not in row["metadata"]["hypothesis_refs"]
        if (
            row["strategy_class"] == "TurtleUnitPyramidingStrategy"
            and "public_rule" not in row["candidate_id"]
        ):
            assert "faith_way_of_turtle_2007" in row["metadata"]["hypothesis_refs"]


def test_flight_candidates_follow_the_audited_public_rule() -> None:
    rows = [
        row
        for row in _load()["candidates"]
        if row["strategy_class"] == "RsiDivergenceScaleOutStrategy"
    ]
    assert rows
    for row in rows:
        assert row["strategy_timeframe"] == "10m"
        params = row["params"]
        assert params["oversold"] == 20.0 and params["overbought"] == 80.0
        assert params["first_exit_fraction"] == 0.6
        assert params["exit_rsi_first"] == 45.0 and params["exit_rsi_second"] == 60.0
        assert params["require_htf_confirmation"] is True and params["htf_multiple"] == 6


def test_multanchanbap_public_rule_candidates_are_exact() -> None:
    rows = [
        row
        for row in _load()["candidates"]
        if "multanchanbap_20_10_public_rule" in row["candidate_id"]
    ]
    assert len(rows) == 2
    for row in rows:
        params = row["params"]
        assert params["channel_source"] == "close"
        assert params["entry_lookback"] == 20 and params["exit_lookback"] == 10
        assert params["stop_loss_pct"] == 0.035 and params["trend_ma_window"] == 120
        assert params["max_units"] == 1 and params["use_n_stop"] is False
        assert params["allow_short"] is False
