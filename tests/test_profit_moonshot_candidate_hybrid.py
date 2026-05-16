from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_profit_moonshot_candidate_hybrid.py"
SPEC = importlib.util.spec_from_file_location("run_profit_moonshot_candidate_hybrid", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_candidate_hybrid_discards_fractional_or_missing_source_leverage() -> None:
    accepted, discarded = MODULE._partition_live_integer_candidate_rows(
        [
            {"name": "integer_ok", "leverage": 5.0, "_candidate_hybrid_source_kind": "liquidation"},
            {"name": "fractional_bad", "leverage": 2.3427334297703024, "_candidate_hybrid_source_kind": "candidate"},
            {"name": "missing_bad", "_candidate_hybrid_source_kind": "candidate"},
        ]
    )

    assert [row["name"] for row in accepted] == ["integer_ok"]
    assert [row["name"] for row in discarded] == ["fractional_bad", "missing_bad"]
    assert {row["reason"] for row in discarded} == {"non_integer_or_missing_live_leverage"}


def test_candidate_hybrid_live_source_gate_discards_untraceable_or_invalid_rows() -> None:
    rows = [
        {
            "name": "live_ok",
            "leverage": 5.0,
            "strategy_validity": {"pass": True},
            "research_history_refs": ["strategy_chronology:live_ok"],
            "source_search_ledger_refs": ["local_artifact:live_ok"],
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
            },
        },
        {
            "name": "calendar_bad",
            "leverage": 5.0,
            "strategy_validity": {"pass": False},
            "research_history_refs": ["strategy_chronology:calendar_bad"],
            "source_search_ledger_refs": ["local_artifact:calendar_bad"],
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
            },
        },
        {
            "name": "source_untraceable",
            "leverage": 5.0,
            "strategy_validity": {"pass": True},
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
            },
        },
        {
            "name": "liquidation_unsafe",
            "leverage": 5.0,
            "strategy_validity": {"pass": True},
            "research_history_refs": ["strategy_chronology:liquidation_unsafe"],
            "source_search_ledger_refs": ["local_artifact:liquidation_unsafe"],
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": -0.1},
            },
        },
    ]

    accepted, discarded = MODULE._partition_live_source_candidate_rows(rows)

    assert [row["name"] for row in accepted] == ["live_ok"]
    assert {row["name"] for row in discarded} == {
        "calendar_bad",
        "source_untraceable",
        "liquidation_unsafe",
    }
    reasons = {row["name"]: row["reasons"] for row in discarded}
    assert reasons["calendar_bad"] == ["strategy_validity_rejected"]
    assert reasons["source_untraceable"] == ["research_history_source_metadata_missing"]
    assert reasons["liquidation_unsafe"] == ["liquidation_source_unsafe"]


def test_candidate_hybrid_rejects_nested_hybrid_sources_only() -> None:
    rows = [
        {
            "name": "atomic_state_signal",
            "leverage": 3.0,
            "strategy_validity": {"pass": True, "primary_signal_type": "state_signal"},
            "research_history_refs": ["strategy_chronology:atomic_state_signal"],
            "source_search_ledger_refs": ["local_artifact:atomic_state_signal"],
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
            },
        },
        {
            "name": "portfolio_allocator_static_blend_plain",
            "leverage": 3.0,
            "strategy_validity": {"pass": True, "primary_signal_type": "state_signal"},
            "research_history_refs": [
                "strategy_chronology:portfolio_allocator_static_blend_plain"
            ],
            "source_search_ledger_refs": [
                "local_artifact:portfolio_allocator_static_blend_plain"
            ],
            "sleeves": [
                "prior_portfolio_governor",
                "allocator_leverage_sweep",
                "meta_portfolio_static_blend",
            ],
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
            },
        },
        {
            "name": "nested_hybrid_online_allocator",
            "leverage": 3.0,
            "strategy_validity": {"pass": True, "primary_signal_type": "state_signal"},
            "research_history_refs": ["strategy_chronology:nested_hybrid_online_allocator"],
            "source_search_ledger_refs": ["local_artifact:nested_hybrid_online_allocator"],
            "sleeves": ["prior_portfolio_governor", "state_signal_clone"],
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
            },
        },
    ]

    accepted, discarded = MODULE._partition_live_source_candidate_rows(rows)

    assert [row["name"] for row in accepted] == [
        "atomic_state_signal",
        "portfolio_allocator_static_blend_plain",
    ]
    assert [row["name"] for row in discarded] == ["nested_hybrid_online_allocator"]
    assert discarded[0]["reasons"] == ["nested_hybrid_or_same_family_source_invalid"]
