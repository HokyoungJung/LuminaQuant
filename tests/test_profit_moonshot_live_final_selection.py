from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "write_profit_moonshot_live_final_selection.py"
SPEC = importlib.util.spec_from_file_location(
    "write_profit_moonshot_live_final_selection", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TUNER_PATH = ROOT / "scripts" / "research" / "tune_profit_moonshot_fresh_portfolio.py"
TUNER_SPEC = importlib.util.spec_from_file_location(
    "tune_profit_moonshot_fresh_portfolio_for_final_tests", TUNER_PATH
)
assert TUNER_SPEC is not None and TUNER_SPEC.loader is not None
TUNER = importlib.util.module_from_spec(TUNER_SPEC)
sys.modules[TUNER_SPEC.name] = TUNER
TUNER_SPEC.loader.exec_module(TUNER)


def _metrics(
    *,
    total_return: float,
    max_drawdown: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    cagr: float = 0.5,
    smart_sortino: float | None = None,
) -> dict[str, float]:
    out = {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "volatility": 0.1,
        "downside_volatility": 0.05,
    }
    if smart_sortino is not None:
        out["smart_sortino"] = smart_sortino
    return out


def _split(
    *,
    total_return: float,
    max_drawdown: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    smart_sortino: float | None = None,
    liquidations: int = 0,
    min_buffer: float = 100.0,
) -> dict:
    return {
        "metrics": _metrics(
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            smart_sortino=smart_sortino,
        ),
        "fills": 10,
        "round_trips": 5,
        "liquidation_count": liquidations,
        "liquidation_event_count_total": liquidations,
        "minimum_margin_buffer": min_buffer,
        "minimum_margin_ratio": 5.0,
        "maximum_liquidation_event_drawdown": 0.0,
        "maximum_liquidation_equity_loss_fraction": 0.0,
        "liquidation_events": [],
    }


def _candidate(
    name: str,
    *,
    train_return: float,
    val_return: float,
    oos_return: float,
    sleeves: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "leverage": 3.0,
        "sleeves": sleeves if sleeves is not None else ["a", "b", "c"],
        "research_history_refs": [
            "strategy_chronology:fixture_dynamic_candidate",
            "docs/research_note/research_history.md#unit",
        ],
        "source_search_ledger_refs": ["local_artifact:fixture_dynamic_candidate"],
        "splits": {
            "train": _split(
                total_return=train_return,
                max_drawdown=0.04,
                sharpe=3.0,
                sortino=4.0,
                calmar=10.0,
            ),
            "val": _split(
                total_return=val_return,
                max_drawdown=0.03,
                sharpe=6.0,
                sortino=7.0,
                calmar=30.0,
            ),
            "oos": _split(
                total_return=oos_return,
                max_drawdown=0.02,
                sharpe=5.0,
                sortino=6.0,
                calmar=20.0,
                smart_sortino=5.0,
            ),
        },
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
        },
        "source_ledger_refs": ["local:unit-profit-moonshot-source"],
    }


def _candidate_with_sleeves(
    name: str, sleeve_names: list[str], *, oos_return: float = 0.10
) -> dict:
    return {
        "name": name,
        "leverage": 3.0,
        "sleeves": sleeve_names,
        "research_history_refs": [
            "strategy_chronology:fixture_dynamic_candidate",
            "docs/research_note/research_history.md#unit",
        ],
        "source_search_ledger_refs": ["local_artifact:fixture_dynamic_candidate"],
        "splits": {
            "train": _split(
                total_return=0.2, max_drawdown=0.04, sharpe=3.0, sortino=4.0, calmar=10.0
            ),
            "val": _split(
                total_return=0.2, max_drawdown=0.03, sharpe=6.0, sortino=7.0, calmar=30.0
            ),
            "oos": _split(
                total_return=oos_return,
                max_drawdown=0.02,
                sharpe=5.0,
                sortino=6.0,
                calmar=20.0,
                smart_sortino=4.5,
            ),
        },
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
        },
        "source_ledger_refs": ["local:unit-profit-moonshot-source"],
    }


def _liquidation_payload() -> dict:
    current_base = _candidate("current_base", train_return=0.10, val_return=0.08, oos_return=0.05)
    current_base["leverage"] = 2.3427334297703024
    current_base["sleeves"] = ["a", "b", "c", "d"]
    return {
        "artifact_kind": "profit_moonshot_liquidation_aware_validation",
        "oos_end_date": "2026-05-10",
        "current_base_reference_result": current_base,
        "promoted_candidate": _candidate(
            "train_val_winner", train_return=0.30, val_return=0.25, oos_return=0.09
        ),
        "selected_by_train_validation_retune": _candidate(
            "oos_poison_candidate", train_return=0.12, val_return=0.10, oos_return=0.30
        ),
        "memory_summary": {"peak_rss_bytes": 256 * 1024 * 1024, "under_8gib": True},
    }


def _candidate_portfolio_payload() -> dict:
    return {
        "artifact_kind": "profit_moonshot_fresh_portfolio_tuning",
        "oos_end_date": "2026-05-10",
        "selected_by_train_val_stability": _candidate(
            "candidate_portfolio_winner", train_return=0.25, val_return=0.20, oos_return=0.08
        ),
        "diagnostic_best_oos": _candidate(
            "diagnostic_oos_only", train_return=0.09, val_return=0.08, oos_return=0.50
        ),
        "allocator_policy": {
            "selection_basis": "train_val_stability_only",
            "uses_locked_oos_for_selection": False,
        },
        "memory_summary": {"peak_rss_bytes": 128 * 1024 * 1024, "under_8gib": True},
    }


def _candidate_hybrid_payload() -> dict:
    candidate_hybrid = _candidate(
        "candidate_hybrid_train_val", train_return=0.28, val_return=0.24, oos_return=0.12
    )
    for split_payload in candidate_hybrid["splits"].values():
        split_payload.pop("minimum_margin_buffer", None)
        split_payload.pop("minimum_margin_ratio", None)
    candidate_hybrid["liquidation_evidence_note"] = "no dynamic-weight liquidation replay"
    return {
        "artifact_kind": "profit_moonshot_candidate_hybrid",
        "oos_end_date": "2026-05-10",
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
        },
        "selected_by_train_validation": candidate_hybrid,
        "tuning_results": [
            _candidate("candidate_hybrid_alt", train_return=0.20, val_return=0.18, oos_return=0.20)
        ],
        "memory_summary": {"peak_rss_bytes": 64 * 1024 * 1024, "under_8gib": True},
    }


def _candidate_hybrid_payload_with_dynamic_replay() -> dict:
    candidate_hybrid = _candidate(
        "candidate_hybrid_dynamic_replay", train_return=0.80, val_return=0.70, oos_return=0.60
    )
    candidate_hybrid["train_val_stability_score"] = 99.0
    for split_name, replay_return in (("train", 0.11), ("val", 0.10), ("oos", 0.07)):
        split_payload = candidate_hybrid["splits"][split_name]
        split_payload["dynamic_liquidation_replay_metrics"] = _metrics(
            total_return=replay_return,
            max_drawdown=0.02,
            sharpe=4.0,
            sortino=5.0,
            calmar=20.0,
            smart_sortino=4.5,
        )
        split_payload["minimum_margin_buffer"] = 1000.0
        split_payload["minimum_margin_ratio"] = 10.0
        split_payload["liquidation_count"] = 0
        split_payload["liquidation_event_count_total"] = 0
    return {
        "artifact_kind": "profit_moonshot_candidate_hybrid",
        "oos_end_date": "2026-05-10",
        "selected_by_train_validation": candidate_hybrid,
        "source_sleeve_metrics": {
            "a": {
                "leverage": 3.0,
                "strategy_validity": {"pass": True},
                "research_history_refs": ["strategy_chronology:source_a"],
                "source_search_ledger_refs": ["local_artifact:source_a"],
                "liquidation_gate": True,
            },
            "b": {
                "leverage": 3.0,
                "strategy_validity": {"pass": True},
                "research_history_refs": ["strategy_chronology:source_b"],
                "source_search_ledger_refs": ["local_artifact:source_b"],
                "liquidation_gate": True,
            },
            "c": {
                "leverage": 3.0,
                "strategy_validity": {"pass": True},
                "research_history_refs": ["strategy_chronology:source_c"],
                "source_search_ledger_refs": ["local_artifact:source_c"],
                "liquidation_gate": True,
            },
        },
        "memory_summary": {"peak_rss_bytes": 64 * 1024 * 1024, "under_8gib": True},
    }


def test_latest_complete_oos_end_date_uses_previous_day_for_intraday_minimum() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T10:00:00Z"},
        ]
    }

    cutoff = MODULE.derive_data_cutoff(refresh_payload, required_symbols=["BTC/USDT", "ETH/USDT"])

    assert cutoff["latest_complete_oos_end_date"] == "2026-05-09"
    assert cutoff["status"] == "derived_from_refresh"


def test_train_val_score_matches_frozen_formula_and_ignores_oos() -> None:
    components = {
        "train_monthlyized_return": 0.05,
        "validation_monthlyized_return": 0.10,
        "train_sharpe": 2.0,
        "validation_sharpe": 5.0,
        "train_sortino": 3.0,
        "validation_sortino": 6.0,
        "train_calmar": 10.0,
        "validation_calmar": 50.0,
        "train_max_drawdown": 0.04,
        "validation_max_drawdown": 0.03,
        "leverage": 5.0,
        "sleeve_count": 5.0,
    }

    assert MODULE.train_val_stability_score_from_components(components) == pytest.approx(
        TUNER._train_val_stability_score_from_components(components)
    )


def test_payload_ranks_by_train_validation_not_locked_oos_and_labels_hybrid() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    legacy_hybrid = {
        "scenarios": {
            "refreshed_latest_tail": {
                "split_metrics": {
                    "oos": {
                        "total_return": 0.99,
                        "max_drawdown": 0.01,
                        "sharpe": 10.0,
                        "sortino": 11.0,
                        "calmar": 99.0,
                    }
                }
            }
        }
    }

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload=_candidate_portfolio_payload(),
        liquidation_payload=_liquidation_payload(),
        candidate_hybrid_payload=_candidate_hybrid_payload(),
        legacy_hybrid_payload=legacy_hybrid,
        source_artifacts={
            "refresh_json": "refresh.json",
            "candidate_portfolio_json": "candidate.json",
            "candidate_hybrid_json": "candidate_hybrid.json",
            "liquidation_json": "liquidation.json",
            "legacy_hybrid_json": "hybrid.json",
        },
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    assert payload["selection_policy"]["uses_locked_oos_for_selection"] is False
    assert payload["winner"]["name"] == "train_val_winner"
    assert payload["winner"]["name"] != "oos_poison_candidate"
    legacy = next(row for row in payload["rows"] if row["kind"] == "legacy_hybrid_benchmark")
    assert legacy["candidate_derived"] is False
    assert legacy["benchmark_only"] is True
    assert legacy["decision_gates"]["eligible_for_candidate_live_promotion"] is False
    candidate_hybrid = next(row for row in payload["rows"] if row["kind"] == "candidate_hybrid")
    assert candidate_hybrid["candidate_derived"] is True
    assert candidate_hybrid["benchmark_only"] is False
    assert candidate_hybrid["strategy_validity"]["pass"] is True
    assert candidate_hybrid["strategy_validity"]["primary_signal_type"] == "state_signal"
    assert candidate_hybrid["decision_gates"]["selection_firewall"] is True
    assert candidate_hybrid["decision_gates"]["liquidation_evidence_available"] is False
    assert candidate_hybrid["decision_gates"]["deployable_candidate"] is False
    assert "return_mdd" in payload["metrics_explanation"]
    assert "minimum_margin_buffer" in payload["metrics_explanation"]
    assert payload["source_artifacts"]["refresh_json"] == "refresh.json"
    assert payload["source_artifacts"]["candidate_hybrid_json"] == "candidate_hybrid.json"


def test_candidate_hybrid_uses_dynamic_replay_metrics_and_margin_evidence() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload={},
        liquidation_payload=_liquidation_payload(),
        candidate_hybrid_payload=_candidate_hybrid_payload_with_dynamic_replay(),
        legacy_hybrid_payload={},
        source_artifacts={"candidate_hybrid_json": "candidate_hybrid.json"},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "candidate_hybrid_dynamic_replay")
    assert row["splits"]["oos"]["total_return"] == pytest.approx(0.07)
    assert row["liquidation"]["evidence_available"] is True
    assert row["liquidation"]["minimum_margin_buffer"] == pytest.approx(1000.0)


def test_calendar_sleeve_rejected_by_strategy_validity_gate() -> None:
    payload = MODULE.build_final_selection_payload(
        refresh_payload={"ohlcv_results": []},
        candidate_portfolio_payload={
            "selected_by_train_val_stability": _candidate_with_sleeves(
                "calendar_candidate",
                [
                    "fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600",
                ],
                oos_return=0.12,
            )
        },
        liquidation_payload={
            "current_base_reference_result": _candidate(
                "current_base", train_return=0.10, val_return=0.08, oos_return=0.05
            )
        },
        candidate_hybrid_payload={},
        legacy_hybrid_payload={},
        source_artifacts={
            "candidate_portfolio_json": "candidate.json",
        },
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "calendar_candidate")
    assert row["decision_gates"]["strategy_validity"]["pass"] is False
    assert row["strategy_validity"]["primary_signal_type"] == "calendar_primary"
    assert (
        "calendar_fixed_month_alpha"
        in row["decision_gates"]["strategy_validity"]["rejection_reasons"]
    )
    assert row["decision_gates"]["deployable_candidate"] is False
    assert "strategy_validity_rejected" in row["rejection_reasons"]


def test_dynamic_sleeve_rows_keep_strategy_validity_metadata() -> None:
    payload = MODULE.build_final_selection_payload(
        refresh_payload={"ohlcv_results": []},
        candidate_portfolio_payload={
            "selected_by_train_val_stability": _candidate_with_sleeves(
                "dynamic_candidate",
                ["fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all"],
                oos_return=0.11,
            )
        },
        liquidation_payload={
            "current_base_reference_result": _candidate(
                "current_base", train_return=0.10, val_return=0.08, oos_return=0.05
            )
        },
        candidate_hybrid_payload={},
        legacy_hybrid_payload={},
        source_artifacts={
            "candidate_portfolio_json": "candidate.json",
        },
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )
    row = next(row for row in payload["rows"] if row["name"] == "dynamic_candidate")
    validity = row["strategy_validity"]
    required = {
        "pass",
        "primary_signal_type",
        "primary_signal_evidence",
        "rejection_reasons",
        "audited_sleeves",
        "audit_sources",
    }
    assert required.issubset(set(validity))
    assert validity["pass"] is True
    assert row["decision_gates"]["liquidation_gate"] is True
    assert row["decision_gates"]["live_integer_source_leverage"] is True
    assert row["strategy_validity"]["audit_sources"] == ["candidate.json"]
    assert row["strategy_validity"]["pass"] is True


def test_candidate_hybrid_fractional_source_leverage_blocks_promotion() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    candidate_hybrid_payload = _candidate_hybrid_payload_with_dynamic_replay()
    candidate_hybrid_payload["source_sleeve_metrics"]["b"]["leverage"] = 2.3427334297703024

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload={},
        liquidation_payload=_liquidation_payload(),
        candidate_hybrid_payload=candidate_hybrid_payload,
        legacy_hybrid_payload={},
        source_artifacts={"candidate_hybrid_json": "candidate_hybrid.json"},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "candidate_hybrid_dynamic_replay")
    assert row["decision_gates"]["live_integer_source_leverage"] is False
    assert row["decision_gates"]["deployable_candidate"] is False
    assert row["non_integer_source_leverages"][0]["source_id"] == "b"
    assert row["non_integer_source_leverages"][0]["leverage"] == pytest.approx(2.3427334297703024)
    assert "live_integer_source_leverage" in row["rejection_reasons"]


def test_strategy_validity_blocks_calendar_primary() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    candidate_portfolio = {
        "selected_by_train_val_stability": _candidate(
            "calendar_only_primary",
            train_return=0.60,
            val_return=0.45,
            oos_return=0.25,
            sleeves=[
                "fresh_calendar_rot_lbtcusdt_sethusdt_lb168_thr20_h336_sc80_st0",
                "fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all",
            ],
        )
    }

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload=candidate_portfolio,
        liquidation_payload=_liquidation_payload(),
        candidate_hybrid_payload={},
        legacy_hybrid_payload={},
        source_artifacts={"candidate_portfolio_json": "candidate_portfolio.json"},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "calendar_only_primary")
    assert row["strategy_validity"]["pass"] is False
    assert row["strategy_validity"]["primary_signal_type"] == "calendar_primary"
    assert "calendar_fixed_month_alpha" in row["strategy_validity"]["rejection_reasons"]
    assert row["strategy_validity"]["audit_sources"] == ["candidate_portfolio.json"]
    assert "strategy_validity_passed" in row["decision_gates"]
    assert row["decision_gates"]["strategy_validity_passed"] is False
    assert "calendar_fixed_month_alpha" in row["rejection_reasons"]
    assert payload["winner"] is None or payload["winner"]["name"] != "calendar_only_primary"


def test_fractional_leverage_blocks_live_candidate_promotion() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    current_base = _candidate("current_base", train_return=0.10, val_return=0.08, oos_return=0.05)
    current_base["leverage"] = 2.3427334297703024
    fractional_candidate = _candidate(
        "fractional_live_impossible", train_return=0.90, val_return=0.80, oos_return=0.70
    )
    fractional_candidate["leverage"] = 2.3427334297703024
    liquidation_payload = {
        "artifact_kind": "profit_moonshot_liquidation_aware_validation",
        "oos_end_date": "2026-05-10",
        "current_base_reference_result": current_base,
        "promoted_candidate": fractional_candidate,
        "memory_summary": {"peak_rss_bytes": 256 * 1024 * 1024, "under_8gib": True},
    }

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload={},
        liquidation_payload=liquidation_payload,
        legacy_hybrid_payload={},
        source_artifacts={},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "fractional_live_impossible")
    assert row["decision_gates"]["live_integer_leverage"] is False
    assert row["decision_gates"]["deployable_candidate"] is False
    assert "live_integer_leverage" in row["rejection_reasons"]
    assert payload["winner"] is None
    assert payload["recommendation"] == "no_live_promotion"


def test_missing_research_history_source_metadata_blocks_live_promotion() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    liquidation = _liquidation_payload()
    liquidation["promoted_candidate"].pop("source_ledger_refs", None)
    liquidation["promoted_candidate"].pop("research_history_refs", None)
    liquidation.pop("selected_by_train_validation_retune", None)

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload={},
        liquidation_payload=liquidation,
        legacy_hybrid_payload={},
        source_artifacts={"liquidation_json": "liquidation.json"},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "train_val_winner")
    assert row["source_metadata"]["present"] is False
    assert row["decision_gates"]["source_metadata_present"] is False
    assert row["decision_gates"]["deployable_candidate"] is False
    assert "source_metadata_missing" in row["rejection_reasons"]
    assert payload["winner"] is None


def test_account_wipeout_blocks_candidate_even_with_positive_return() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    liquidation = _liquidation_payload()
    liquidation["promoted_candidate"]["splits"]["oos"]["liquidation_events"] = [
        {"account_wipeout": True}
    ]

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload={},
        liquidation_payload=liquidation,
        legacy_hybrid_payload={},
        source_artifacts=MODULE.SourceArtifacts(liquidation_json="liquidation.json"),
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "train_val_winner")
    assert row["liquidation"]["account_wipeout"] is True
    assert row["decision_gates"]["no_account_wipeout"] is False
    assert row["decision_gates"]["deployable_candidate"] is False
    assert "no_account_wipeout" in row["rejection_reasons"]


def test_stale_artifact_cutoff_blocks_recommendation() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    stale_candidate = _candidate_portfolio_payload()
    stale_candidate["oos_end_date"] = "2026-05-06"

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload=stale_candidate,
        liquidation_payload=_liquidation_payload(),
        legacy_hybrid_payload={},
        source_artifacts={},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    assert payload["status"] == "failed_stale_artifact_cutoff"
    assert payload["recommendation"] == "no_live_promotion"
    assert payload["data_cutoff"]["artifact_cutoff_gate_passed"] is False


def test_hybrid_split_window_oos_end_is_in_cutoff_gate() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    hybrid_payload = {
        "split_windows": {"oos_end_inclusive": "2026-05-06"},
        "scenarios": {
            "refreshed_latest_tail": {
                "split_metrics": {
                    "oos": _metrics(
                        total_return=0.01, max_drawdown=0.01, sharpe=1.0, sortino=1.0, calmar=1.0
                    )
                }
            }
        },
    }

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload=_candidate_portfolio_payload(),
        liquidation_payload=_liquidation_payload(),
        legacy_hybrid_payload=hybrid_payload,
        source_artifacts={},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    assert payload["status"] == "failed_stale_artifact_cutoff"
    assert {
        "artifact": "legacy_hybrid",
        "path": "split_windows.oos_end_inclusive",
        "date": "2026-05-06",
    } in payload["data_cutoff"]["stale_artifact_oos_end_dates"]


def test_memory_ledger_parses_time_logs_and_flags_over_8gib(tmp_path: Path) -> None:
    ok_log = tmp_path / "ok.time"
    bad_log = tmp_path / "bad.time"
    ok_log.write_text("Maximum resident set size (kbytes): 1024\n", encoding="utf-8")
    bad_log.write_text("Maximum resident set size (kbytes): 9000000\n", encoding="utf-8")

    ledger = MODULE.build_memory_ledger(
        artifacts=[{"memory_summary": {"peak_rss_bytes": 256 * 1024 * 1024}}],
        time_logs=[ok_log, bad_log],
    )

    assert ledger["under_8gib"] is False
    assert [entry["under_8gib"] for entry in ledger["entries"]] == [True, True, False]


def test_missing_research_history_source_metadata_blocks_promotion() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    candidate = _candidate(
        "source_untraceable", train_return=0.90, val_return=0.80, oos_return=0.70
    )
    candidate.pop("research_history_refs")
    candidate.pop("source_search_ledger_refs")
    liquidation_payload = {
        "artifact_kind": "profit_moonshot_liquidation_aware_validation",
        "oos_end_date": "2026-05-10",
        "current_base_reference_result": _candidate(
            "current_base", train_return=0.10, val_return=0.08, oos_return=0.05
        ),
        "promoted_candidate": candidate,
        "memory_summary": {"peak_rss_bytes": 256 * 1024 * 1024, "under_8gib": True},
    }

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload={},
        liquidation_payload=liquidation_payload,
        legacy_hybrid_payload={},
        source_artifacts={},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "source_untraceable")
    assert row["decision_gates"]["research_history_source_metadata_present"] is False
    assert row["decision_gates"]["deployable_candidate"] is False
    assert "research_history_source_metadata_present" in row["rejection_reasons"]
    assert payload["winner"] is None


def test_candidate_hybrid_inherits_invalid_source_metadata_gate() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    candidate_hybrid_payload = _candidate_hybrid_payload_with_dynamic_replay()
    candidate_hybrid_payload["source_sleeve_metrics"]["b"]["strategy_validity"] = {
        "pass": False,
        "rejection_reasons": ["calendar_primary_alpha_unsupported"],
    }

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload={},
        liquidation_payload=_liquidation_payload(),
        candidate_hybrid_payload=candidate_hybrid_payload,
        legacy_hybrid_payload={},
        source_artifacts={"candidate_hybrid_json": "candidate_hybrid.json"},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "candidate_hybrid_dynamic_replay")
    assert row["decision_gates"]["live_source_candidate_metadata"] is False
    assert row["decision_gates"]["deployable_candidate"] is False
    assert row["source_candidate_gate_failures"] == [
        {
            "source_id": "b",
            "reasons": ["strategy_validity_rejected"],
        }
    ]
    assert "live_source_candidate_metadata" in row["rejection_reasons"]


def test_candidate_hybrid_source_gate_rejects_nested_and_fixed_month_sources() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-05-10T23:59:30Z"},
        ]
    }
    candidate_hybrid_payload = _candidate_hybrid_payload_with_dynamic_replay()
    candidate_hybrid_payload["source_sleeve_metrics"]["b"]["family"] = "meta_portfolio"
    candidate_hybrid_payload["source_sleeve_metrics"]["c"]["name"] = "fixed_month_alpha_proxy"

    payload = MODULE.build_final_selection_payload(
        refresh_payload=refresh_payload,
        candidate_portfolio_payload={},
        liquidation_payload=_liquidation_payload(),
        candidate_hybrid_payload=candidate_hybrid_payload,
        legacy_hybrid_payload={},
        source_artifacts={"candidate_hybrid_json": "candidate_hybrid.json"},
        time_logs=[],
        required_symbols=["BTC/USDT", "ETH/USDT"],
    )

    row = next(row for row in payload["rows"] if row["name"] == "candidate_hybrid_dynamic_replay")
    assert row["decision_gates"]["live_source_candidate_metadata"] is False
    assert row["decision_gates"]["deployable_candidate"] is False
    assert row["source_candidate_gate_failures"] == [
        {"source_id": "b", "reasons": ["nested_hybrid_or_same_family_source_invalid"]},
        {"source_id": "c", "reasons": ["calendar_primary_source_invalid"]},
    ]
    assert "live_source_candidate_metadata" in row["rejection_reasons"]
