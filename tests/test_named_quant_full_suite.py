from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/build_named_quant_full_suite.py"
RUNNER = ROOT / "scripts/research/run_event_driven_candidate_evaluation.py"
MAIN = ROOT / "configs/research/named_quant_crypto_tradfi_suite_v1.json"
CLAUDE = ROOT / "configs/research/named_quant_claude_suite_v1.json"
OUTPUT = ROOT / "configs/research/named_quant_full_suite_v1.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _module("build_named_quant_full_suite", SCRIPT)
SUITE_RUNNER = _module("event_driven_candidate_evaluation_for_full_test", RUNNER)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_checked_manifest_is_exact_union_and_runner_ready() -> None:
    main = _load(MAIN)
    claude = _load(CLAUDE)
    full = _load(OUTPUT)
    assert full == BUILDER.build_full_suite(main, claude)
    assert full["research_only"] is True
    assert full["promotion_eligible"] is False
    assert full["allow_real_money"] is False
    assert full["performance_claim"].startswith("none")
    assert full["allocator"] == claude["allocator"]
    assert full["method"] == claude["allocator"]["method"]
    assert full["upper"] == claude["allocator"]["upper"]
    assert full["min_sleeves"] == claude["allocator"]["min_sleeves"]
    assert full["gross_cap"] == claude["allocator"]["gross_cap"]
    assert full["allocator_variants"] == claude["allocator_variants"]
    assert full["asset_level_allocation_study"] == claude["asset_level_allocation_study"]
    assert (
        full["portfolio_recipes"][: len(claude["portfolio_recipes"])] == claude["portfolio_recipes"]
    )
    assert full["locked_oos_evaluation"] == {
        "rebalance_every_observations": 5,
        "allocation_cost_bps": 10.0,
        "periods_per_year": 252,
    }

    candidate_ids = [row["candidate_id"] for row in full["candidates"]]
    source_ids = [row["source_id"] for row in full["evidence_sources"]]
    assert len(candidate_ids) == len(set(candidate_ids))
    assert len(source_ids) == len(set(source_ids))
    assert len(candidate_ids) == len(main["candidates"]) + len(claude["candidates"])
    assert set(full["sleeves"]) == {*main["sleeves"], *claude["sleeves"]}
    for source in (main, claude):
        for candidate in source["candidates"]:
            assert full["candidates"][candidate_ids.index(candidate["candidate_id"])] == candidate
        for sleeve_id, sleeve in source["sleeves"].items():
            assert full["sleeves"][sleeve_id] == sleeve
    for index, candidate in enumerate(full["candidates"]):
        metadata = candidate.get("metadata", {})
        refs = [
            *(candidate.get("hypothesis_refs") or metadata.get("hypothesis_refs", [])),
            *(candidate.get("provenance_refs") or metadata.get("provenance_refs", [])),
        ]
        assert set(refs) <= set(source_ids)
        assert (
            SUITE_RUNNER._candidate_spec(candidate, index)["candidate_id"]
            == candidate["candidate_id"]
        )
    amateurquant = full["evidence_sources"][source_ids.index("amateurquant_profile")]
    assert amateurquant["allowed_usage_label"] == "evidence_only"


def test_conflicting_duplicate_ids_fail_closed() -> None:
    main = {
        "candidates": [{"candidate_id": "duplicate", "strategy_class": "A"}],
        "evidence_sources": [{"source_id": "source", "url": "a"}],
    }
    with pytest.raises(ValueError, match="duplicate candidate"):
        BUILDER.build_full_suite(
            main,
            {"candidates": [{"candidate_id": "duplicate", "strategy_class": "B"}]},
        )
    with pytest.raises(ValueError, match="duplicate source"):
        BUILDER.build_full_suite(
            main,
            {"evidence_sources": [{"source_id": "source", "url": "b"}]},
        )
