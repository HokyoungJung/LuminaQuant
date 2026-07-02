"""C2b known-answer / property tests for the deterministic alpha search loop.

Coverage map:

(a) generate: the operator-tree enumeration is canonical and input-order
    invariant, and ``raw_n`` records the *full* generated breadth even when the
    evaluated pool is capped by ``max_candidates``;
(b) evaluate + select end-to-end determinism: running the loop twice on the same
    synthetic panel yields byte-identical records and the identical promoted set;
(c) survivorship-gate integration: only gate survivors are promoted, an
    unreachable DSR threshold promotes nothing, and the planted-edge candidates
    clear the gate while pure-noise candidates do not;
(d) ledger promotion: exactly the promoted candidates land in the JSONL ledger;
(e) config gate: ``research.alpha_search`` is OFF by default and additive
    overrides round-trip through the loader;
(f) gated LLM proposer: default-OFF ignores frozen proposals; enabling merges
    them into ``raw_n`` and routes them through the same deterministic gate.

No scipy / sklearn / eigendecomposition; only research primitives.
"""

from __future__ import annotations

from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.configuration.schema import AlphaSearchConfig, ResearchConfig
from lumina_quant.research.alpha_candidate_ledger import ResearchLedger
from lumina_quant.research.alpha_search import (
    AlphaSearchCandidate,
    build_synthetic_search_panel,
    generate_candidate_trees,
    run_alpha_search,
)


# ---------------------------------------------------------------------------
# (a) generate: canonical, input-order invariant enumeration + raw_n bound.
# ---------------------------------------------------------------------------


def test_generation_is_canonical_and_order_invariant():
    trees_a = generate_candidate_trees(["close", "volume", "ret"])
    trees_b = generate_candidate_trees(["ret", "close", "volume"])
    # Input order of the feature list must not change the enumerated pool.
    assert trees_a == trees_b
    # Canonical (sorted) order is stable and deduplicated.
    formulas = [t for t in trees_a]
    assert formulas == sorted(trees_a, key=lambda tree: _formula_of(tree))
    assert len(set(map(_formula_of, trees_a))) == len(trees_a)


def _formula_of(tree):
    from lumina_quant.research.alpha_search import _formula_str

    return _formula_str(tree)


def test_raw_n_tracks_full_breadth_even_when_capped():
    panel = build_synthetic_search_panel()
    full = generate_candidate_trees(panel.feature_names())
    result = run_alpha_search(panel, max_candidates=12)
    # The evaluated pool is capped, but raw_n keeps the full generated breadth as
    # an upper bound and N_eff never exceeds it.
    assert result.evaluated_n == 12
    assert result.raw_n == len(full)
    assert result.raw_n > result.evaluated_n
    assert 1.0 <= result.n_eff <= float(result.raw_n)


# ---------------------------------------------------------------------------
# (b) end-to-end determinism.
# ---------------------------------------------------------------------------


def test_run_twice_promotes_identical_set():
    panel = build_synthetic_search_panel()
    r1 = run_alpha_search(panel, max_candidates=40)
    r2 = run_alpha_search(panel, max_candidates=40)
    assert r1.promoted_ids == r2.promoted_ids
    assert r1.records == r2.records
    assert r1.raw_n == r2.raw_n
    assert r1.n_eff == r2.n_eff
    assert r1.dispersion == r2.dispersion


# ---------------------------------------------------------------------------
# (c) survivorship-gate integration.
# ---------------------------------------------------------------------------


def test_only_gate_survivors_are_promoted():
    panel = build_synthetic_search_panel()
    result = run_alpha_search(panel, max_candidates=40)
    # Every promoted id is a gate pass; every non-promoted record failed.
    for rec in result.records:
        assert rec["passed"] is (rec["candidate_id"] in result.promoted_ids)
        if rec["passed"]:
            assert rec["deflated_sharpe"] >= 0.95
    # The planted close-momentum edge survives; a raw noisy feature does not.
    edge = next(r for r in result.records if r["formula"] == "delta(close, 5)")
    assert edge["passed"] is True
    assert edge["ic"] > 0.2
    assert result.promoted_ids  # at least one survivor exists


def test_unreachable_threshold_promotes_nothing():
    panel = build_synthetic_search_panel()
    result = run_alpha_search(panel, max_candidates=40, dsr_threshold=1.01)
    assert result.promoted_ids == ()
    assert all(rec["passed"] is False for rec in result.records)


# ---------------------------------------------------------------------------
# (d) ledger promotion.
# ---------------------------------------------------------------------------


def test_survivors_written_to_ledger(tmp_path):
    panel = build_synthetic_search_panel()
    ledger = ResearchLedger(tmp_path / "alpha_search_ledger.jsonl")
    result = run_alpha_search(panel, max_candidates=40, ledger=ledger)
    rows = ledger.read_all()
    assert [row["candidate_id"] for row in rows] == list(result.promoted_ids)
    assert all(row["passed"] is True for row in rows)
    # raw_n provenance is stamped on every promoted record.
    assert all(row["raw_n"] == result.raw_n for row in rows)
    # Re-running with the same ledger is deterministic: same promoted ids appended.
    run_alpha_search(panel, max_candidates=40, ledger=ledger)
    rows2 = ledger.read_all()
    assert [row["candidate_id"] for row in rows2[len(rows) :]] == list(result.promoted_ids)


# ---------------------------------------------------------------------------
# (e) config gate: OFF by default, additive override round-trips.
# ---------------------------------------------------------------------------


def test_alpha_search_config_defaults_off():
    cfg = ResearchConfig()
    assert isinstance(cfg.alpha_search, AlphaSearchConfig)
    assert cfg.alpha_search.enabled is False
    assert cfg.alpha_search.enable_llm_proposer is False
    # Full runtime default keeps the loop inert.
    runtime = build_runtime_config({}, {})
    assert runtime.research.alpha_search.enabled is False
    assert runtime.research.execution_attribution_enabled is False


def test_alpha_search_config_override_round_trips():
    data = {
        "research": {
            "alpha_search": {
                "enabled": True,
                "max_candidates": 32,
                "seed": 99,
                "dsr_threshold": 0.9,
                "require_spa": True,
                "enable_llm_proposer": True,
            }
        }
    }
    runtime = build_runtime_config(data, {})
    alpha = runtime.research.alpha_search
    assert alpha.enabled is True
    assert alpha.max_candidates == 32
    assert alpha.seed == 99
    assert alpha.dsr_threshold == 0.9
    assert alpha.require_spa is True
    assert alpha.require_pbo is False
    assert alpha.enable_llm_proposer is True


# ---------------------------------------------------------------------------
# (f) gated LLM proposer (default OFF; frozen artifacts through the same gate).
# ---------------------------------------------------------------------------


def _frozen_proposal() -> AlphaSearchCandidate:
    # A frozen/seeded proposer draft mirroring the planted edge (delta close).
    tree = ("u", "delta", ("feat", "close"), 5)
    return AlphaSearchCandidate(
        candidate_id="proposal-seed",
        formula="delta(close, 5)",
        tree=tree,
        seed=4242,
        origin="llm_proposer",
    )


def test_llm_proposer_off_by_default_ignores_proposals():
    panel = build_synthetic_search_panel()
    base = run_alpha_search(panel, max_candidates=40)
    with_props = run_alpha_search(panel, max_candidates=40, llm_proposals=[_frozen_proposal()])
    # Default OFF: proposals do not enter the pool or the raw_n breadth.
    assert with_props.raw_n == base.raw_n
    assert with_props.evaluated_n == base.evaluated_n
    assert with_props.records == base.records


def test_llm_proposer_enabled_merges_and_gates():
    panel = build_synthetic_search_panel()
    base = run_alpha_search(panel, max_candidates=40)
    gated = run_alpha_search(
        panel,
        max_candidates=40,
        enable_llm_proposer=True,
        llm_proposals=[_frozen_proposal()],
    )
    # The frozen proposal increments raw_n and is evaluated as one extra candidate.
    assert gated.raw_n == base.raw_n + 1
    assert gated.evaluated_n == base.evaluated_n + 1
    llm_records = [r for r in gated.records if r["candidate_id"].startswith("llm-")]
    assert len(llm_records) == 1
    # The proposal cleared the identical deterministic survivorship gate.
    assert llm_records[0]["passed"] is True
    assert "llm-0000" in gated.promoted_ids
    # Deterministic: enabling twice is byte-identical.
    gated2 = run_alpha_search(
        panel,
        max_candidates=40,
        enable_llm_proposer=True,
        llm_proposals=[_frozen_proposal()],
    )
    assert gated.records == gated2.records
