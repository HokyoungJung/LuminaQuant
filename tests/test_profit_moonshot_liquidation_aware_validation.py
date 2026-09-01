from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_profit_moonshot_liquidation_aware_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "run_profit_moonshot_liquidation_aware_validation", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_intrabar_adverse_high_low_breaches_liquidation_threshold() -> None:
    model = MODULE.MarginModel(
        maintenance_margin_rate=0.01,
        taker_fee_rate=0.0,
        slippage_rate=0.0,
        funding_rate_per_8h=0.0,
        stress_buffer_rate=0.0,
        liquidation_fee_rate=0.0,
    )
    long_leg = MODULE.OpenLeg(
        sleeve="sleeve-a",
        symbol="BTC/USDT",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
    )
    short_leg = MODULE.OpenLeg(
        sleeve="sleeve-b",
        symbol="ETH/USDT",
        side="SHORT",
        qty=1.0,
        entry_price=100.0,
    )

    long_event = MODULE._intrabar_liquidation_event(
        long_leg,
        high=110.0,
        low=80.99,
        leverage=5,
        model=model,
        split_name="train",
        timestamp="2026-01-01T00:00:00Z",
    )
    short_event = MODULE._intrabar_liquidation_event(
        short_leg,
        high=119.01,
        low=90.0,
        leverage=5,
        model=model,
        split_name="val",
        timestamp="2026-01-01T01:00:00Z",
    )

    assert long_event is not None
    assert long_event["side"] == "LONG"
    assert long_event["trigger_price"] <= long_event["liquidation_price"]
    assert short_event is not None
    assert short_event["side"] == "SHORT"
    assert short_event["trigger_price"] >= short_event["liquidation_price"]


def test_split_margin_summary_records_liquidations_and_buffer_minima() -> None:
    summary = MODULE._split_margin_summary(
        split_name="train",
        snapshots=[
            {"margin_buffer": 250.0, "margin_ratio": 5.0},
            {"margin_buffer": -0.5, "margin_ratio": 0.95},
        ],
        liquidation_events=[{"split": "train"}, {"split": "oos"}],
    )

    assert summary["liquidation_count"] == 1
    assert summary["minimum_margin_buffer"] == -0.5
    assert summary["minimum_margin_ratio"] == 0.95
    assert summary["margin_buffer_positive"] is False


def test_train_validation_leverage_selection_ignores_locked_oos_poison() -> None:
    grid = [
        {
            "leverage": 4,
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 10.0},
                "val": {"liquidation_count": 0, "minimum_margin_buffer": 5.0},
                "oos": {"liquidation_count": 9, "minimum_margin_buffer": -1.0},
            },
            "train_val_score": 4.0,
        },
        {
            "leverage": 5,
            "splits": {
                "train": {"liquidation_count": 1, "minimum_margin_buffer": -1.0},
                "val": {"liquidation_count": 0, "minimum_margin_buffer": 5.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 5.0},
            },
            "train_val_score": 99.0,
        },
    ]

    selected = MODULE._select_train_validation_leverage(grid)
    gates = MODULE._liquidation_promotion_gates(selected)

    assert selected["leverage"] == 4
    assert selected["selection_policy"]["uses_locked_oos_for_selection"] is False
    assert gates["train_validation_liquidation_safe"] is True
    assert gates["all_splits_liquidation_safe"] is False


def test_liquidation_gates_block_promoted_success_when_unsafe() -> None:
    gates = MODULE._liquidation_promotion_gates(
        {
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "val": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 1, "minimum_margin_buffer": -0.01},
            }
        }
    )

    assert gates["liquidation_free"] is False
    assert gates["margin_buffer_positive"] is False
    assert MODULE._liquidation_safe_for_promotion(gates) is False


def test_liquidation_gates_accept_validation_split_name() -> None:
    gates = MODULE._liquidation_promotion_gates(
        {
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
            }
        }
    )

    assert gates["train_validation_liquidation_safe"] is True
    assert gates["all_splits_liquidation_safe"] is True
    assert MODULE._liquidation_safe_for_promotion(gates) is True


def test_cli_defaults_to_strict_zero_liquidation_promotion() -> None:
    args = MODULE.build_parser().parse_args([])

    assert args.allowed_total_liquidations == 0
    assert args.allowed_split_liquidations == 0
    assert args.retune_audit_limit == 0
    assert args.retune_csv_limit == 0
    assert args.external_state_csv == ""


def test_calendar_current_base_sleeves_are_strategy_invalid() -> None:
    validity = MODULE._strategy_validity_for_sleeves(MODULE.CURRENT_BASE_SLEEVES)

    assert validity["pass"] is False
    assert validity["primary_signal_type"] == "calendar_primary"
    assert "calendar_primary_alpha_unsupported" in validity["rejection_reasons"]


def test_calendar_month_day_hour_entry_rules_are_strategy_invalid() -> None:
    specs = [
        SimpleNamespace(
            name="state_clean",
            family="state_momentum_proxy",
            calendar_long_months=(),
            calendar_short_months=(),
            entry_days_of_month=(),
            entry_hours=(),
        ),
        SimpleNamespace(
            name="state_with_month",
            family="state_momentum_proxy",
            calendar_long_months=(3,),
            calendar_short_months=(),
            entry_days_of_month=(),
            entry_hours=(),
        ),
        SimpleNamespace(
            name="state_with_day",
            family="state_momentum_proxy",
            calendar_long_months=(),
            calendar_short_months=(),
            entry_days_of_month=(15,),
            entry_hours=(),
        ),
        SimpleNamespace(
            name="state_with_hour",
            family="state_momentum_proxy",
            calendar_long_months=(),
            calendar_short_months=(),
            entry_days_of_month=(),
            entry_hours=(2,),
        ),
    ]

    clean = MODULE._strategy_validity_for_sleeves(["state_clean"], specs=specs)
    invalid = MODULE._strategy_validity_for_sleeves(
        ["state_with_month", "state_with_day", "state_with_hour"],
        specs=specs,
    )

    assert clean["pass"] is True
    assert invalid["pass"] is False
    assert "calendar_entry_rule_unsupported" in invalid["rejection_reasons"]
    assert "calendar_month_entry_rule" in invalid["rejection_reasons"]
    assert "calendar_day_entry_rule" in invalid["rejection_reasons"]
    assert "calendar_hour_entry_rule" in invalid["rejection_reasons"]
    assert {item["primary_signal_type"] for item in invalid["audited_sleeves"]} == {
        "calendar_timed_state_signal"
    }


def test_strategy_invalid_rows_cannot_be_deployable_even_when_metrics_pass() -> None:
    class FakeTuner:
        MAX_ACCEPTABLE_OOS_MDD = 0.25
        SUCCESS_SHARPE = 1.0
        SUCCESS_SORTINO = 1.0
        SUCCESS_SMART_SORTINO = 1.0
        SUCCESS_CALMAR = 1.0

        @staticmethod
        def _return_risk_score(total_return: float, max_drawdown: float) -> float:
            return total_return / max(max_drawdown, 1e-9)

        @staticmethod
        def _smart_sortino(metrics: dict) -> float:
            return float(metrics.get("sortino") or 0.0)

    current_base = {
        "leverage": 1.0,
        "splits": {
            "oos": {
                "metrics": {
                    "total_return": 0.01,
                    "max_drawdown": 0.01,
                    "sharpe": 1.0,
                    "sortino": 1.0,
                    "calmar": 1.0,
                }
            }
        },
    }
    invalid = {
        "leverage": 5.0,
        "strategy_validity": MODULE._strategy_validity_for_sleeves(
            ["fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600"]
        ),
        "liquidation_gates": {
            "liquidation_free": True,
            "margin_buffer_positive": True,
            "all_splits_liquidation_safe": True,
        },
        "train_val_score": 5.0,
        "splits": {
            "train": {"metrics": {"total_return": 0.2}},
            "validation": {"metrics": {"total_return": 0.1}},
            "oos": {
                "metrics": {
                    "total_return": 0.2,
                    "max_drawdown": 0.02,
                    "sharpe": 5.0,
                    "sortino": 5.0,
                    "calmar": 5.0,
                }
            },
        },
    }

    MODULE._apply_reference_gates(FakeTuner(), [invalid], current_base)

    assert invalid["strategy_validity"]["pass"] is False
    assert invalid["strategy_validity_gate"] is False
    assert invalid["deployable_success"] is False


def test_current_base_calendar_tuple_is_hypothesis_only_never_selection_target() -> None:
    class FakeTuner:
        MAX_ACCEPTABLE_OOS_MDD = 0.25
        SUCCESS_SHARPE = 1.0
        SUCCESS_SORTINO = 1.0
        SUCCESS_SMART_SORTINO = 1.0
        SUCCESS_CALMAR = 1.0

        @staticmethod
        def _return_risk_score(total_return: float, max_drawdown: float) -> float:
            return total_return / max(max_drawdown, 1e-9)

        @staticmethod
        def _smart_sortino(metrics: dict) -> float:
            return float(metrics.get("sortino") or 0.0)

    current_base_reference = {
        "candidate_source": "current_base_tuple",
        "candidate_name": "current_base_tuple",
        "leverage": 5.0,
        "strategy_validity": MODULE._strategy_validity_for_sleeves(MODULE.CURRENT_BASE_SLEEVES),
        "liquidation_gates": {
            "liquidation_free": True,
            "margin_buffer_positive": True,
            "all_splits_liquidation_safe": True,
        },
        "train_val_score": 99.0,
        "splits": {
            "train": {
                "metrics": {"total_return": 0.2},
                "liquidation_count": 0,
                "minimum_margin_buffer": 5.0,
            },
            "validation": {
                "metrics": {"total_return": 0.1},
                "liquidation_count": 0,
                "minimum_margin_buffer": 5.0,
            },
            "oos": {
                "metrics": {
                    "total_return": 0.2,
                    "max_drawdown": 0.02,
                    "sharpe": 5.0,
                    "sortino": 5.0,
                    "calmar": 5.0,
                },
                "liquidation_count": 0,
                "minimum_margin_buffer": 5.0,
            },
        },
    }

    MODULE._apply_reference_gates(FakeTuner(), [current_base_reference], current_base_reference)

    assert current_base_reference["selection_policy"]["selection_target"] is False
    assert (
        current_base_reference["selection_policy"]["selection_role"] == "hypothesis_reference_only"
    )
    assert current_base_reference["deployable_success"] is False
    assert MODULE._select_train_validation_leverage([current_base_reference]) == {}
    assert (
        MODULE._select_train_validation_candidate(
            [current_base_reference],
            tolerance=MODULE.LiquidationTolerance(),
        )
        == {}
    )


def test_locked_oos_diagnostic_seed_never_becomes_selection_target() -> None:
    diagnostic_seed = {
        "candidate_source": "integer_audit_diagnostic_best_oos",
        "candidate_name": "oos_discovered_only",
        "train_val_score": 99.0,
        "splits": {
            "train": {
                "metrics": {"total_return": 0.5},
                "liquidation_count": 0,
                "minimum_margin_buffer": 10.0,
            },
            "validation": {
                "metrics": {"total_return": 0.5},
                "liquidation_count": 0,
                "minimum_margin_buffer": 10.0,
            },
            "oos": {
                "metrics": {"total_return": 0.5},
                "liquidation_count": 0,
                "minimum_margin_buffer": 10.0,
            },
        },
    }
    train_val_seed = {
        "candidate_source": "integer_audit_selected_by_train_val_stability",
        "candidate_name": "train_val_discovered",
        "train_val_score": 1.0,
        "splits": {
            "train": {
                "metrics": {"total_return": 0.01},
                "liquidation_count": 0,
                "minimum_margin_buffer": 10.0,
            },
            "validation": {
                "metrics": {"total_return": 0.01},
                "liquidation_count": 0,
                "minimum_margin_buffer": 10.0,
            },
            "oos": {
                "metrics": {"total_return": 0.01},
                "liquidation_count": 0,
                "minimum_margin_buffer": 10.0,
            },
        },
    }

    diagnostic_seed["selection_policy"] = MODULE._reference_only_selection_policy(diagnostic_seed)
    train_val_seed["selection_policy"] = MODULE._reference_only_selection_policy(train_val_seed)
    selected = MODULE._select_train_validation_candidate(
        [diagnostic_seed, train_val_seed],
        tolerance=MODULE.LiquidationTolerance(),
    )

    assert diagnostic_seed["selection_policy"]["selection_target"] is False
    assert diagnostic_seed["selection_policy"]["selection_role"] == (
        "diagnostic_locked_oos_report_only"
    )
    assert selected["candidate_name"] == "train_val_discovered"
    assert selected["selection_policy"]["uses_locked_oos_for_selection"] is False


def test_non_current_candidate_preserves_train_validation_policy_and_provenance() -> None:
    class FakeTuner:
        MAX_ACCEPTABLE_OOS_MDD = 0.25
        SUCCESS_SHARPE = 1.0
        SUCCESS_SORTINO = 1.0
        SUCCESS_SMART_SORTINO = 1.0
        SUCCESS_CALMAR = 1.0

        @staticmethod
        def _return_risk_score(total_return: float, max_drawdown: float) -> float:
            return total_return / max(max_drawdown, 1e-9)

        @staticmethod
        def _smart_sortino(metrics: dict) -> float:
            return float(metrics.get("sortino") or 0.0)

    current_base = {
        "leverage": 1.0,
        "splits": {
            "oos": {
                "metrics": {
                    "total_return": 0.01,
                    "max_drawdown": 0.02,
                    "sharpe": 1.0,
                    "sortino": 1.0,
                    "calmar": 1.0,
                }
            }
        },
    }
    candidate = {
        "candidate_source": "integer_audit_selected_by_train_val_stability",
        "candidate_name": "state_market_candidate",
        "leverage": 3.0,
        "strategy_validity": MODULE._strategy_validity_for_sleeves(["state_market_candidate"]),
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
        },
        "liquidation_gates": {
            "liquidation_free": True,
            "margin_buffer_positive": True,
            "all_splits_liquidation_safe": True,
        },
        "train_val_score": 3.0,
        "splits": {
            "train": {
                "metrics": {"total_return": 0.04},
                "liquidation_count": 0,
                "minimum_margin_buffer": 5.0,
            },
            "validation": {
                "metrics": {"total_return": 0.03},
                "liquidation_count": 0,
                "minimum_margin_buffer": 5.0,
            },
            "oos": {
                "metrics": {
                    "total_return": 0.05,
                    "max_drawdown": 0.01,
                    "sharpe": 3.0,
                    "sortino": 3.0,
                    "calmar": 3.0,
                },
                "liquidation_count": 0,
                "minimum_margin_buffer": 5.0,
            },
        },
    }

    MODULE._apply_reference_gates(FakeTuner(), [candidate], current_base)
    selected = MODULE._select_train_validation_candidate(
        [candidate],
        tolerance=MODULE.LiquidationTolerance(),
    )

    assert candidate["candidate_source"] == "integer_audit_selected_by_train_val_stability"
    assert candidate["candidate_name"] == "state_market_candidate"
    assert candidate["selection_policy"]["uses_locked_oos_for_selection"] is False
    assert candidate["selection_policy"]["selection_target"] is True
    assert selected["candidate_source"] == candidate["candidate_source"]
    assert selected["candidate_name"] == candidate["candidate_name"]
    assert selected["selection_policy"]["selection_inputs"] == ["train", "validation"]
    assert selected["selection_policy"]["uses_locked_oos_for_selection"] is False


def test_empty_retune_selection_stays_empty() -> None:
    selected = MODULE._select_train_validation_candidate(
        [],
        tolerance=MODULE.LiquidationTolerance(),
    )

    assert selected == {}


def test_retune_seed_defaults_keep_replay_current_base_only(tmp_path: Path) -> None:
    audit = {
        "selected_by_train_val_stability": {"name": "audit-selected", "sleeves": ["sleeve-a"]},
        "diagnostic_best_oos": {"name": "audit-oos", "sleeves": ["sleeve-b"]},
        "diagnostic_quarantine": [{"name": "quarantine", "sleeves": ["sleeve-c"]}],
    }

    seeds = MODULE._build_candidate_seeds(
        integer_audit=audit,
        candidate_csv=tmp_path / "missing.csv",
        audit_limit=0,
        csv_limit=0,
    )

    assert [seed["source"] for seed in seeds] == ["current_base_tuple"]
    assert [seed["sleeves"] for seed in seeds] == [list(MODULE.CURRENT_BASE_SLEEVES)]


def test_tiny_liquidation_tolerance_is_diagnostic_not_promotional() -> None:
    tolerance = MODULE.LiquidationTolerance(
        allowed_total_liquidations=1,
        allowed_split_liquidations=1,
        max_liquidation_event_drawdown=0.005,
        max_liquidation_equity_loss_fraction=0.005,
    )
    candidate = {
        "splits": {
            "train": {
                "liquidation_count": 0,
                "minimum_margin_buffer": 25.0,
                "maximum_liquidation_event_drawdown": 0.0,
                "maximum_liquidation_equity_loss_fraction": 0.0,
            },
            "validation": {
                "liquidation_count": 1,
                "minimum_margin_buffer": 20.0,
                "maximum_liquidation_event_drawdown": 0.0016,
                "maximum_liquidation_equity_loss_fraction": 0.0005,
            },
            "oos": {
                "liquidation_count": 0,
                "minimum_margin_buffer": 30.0,
                "maximum_liquidation_event_drawdown": 0.0,
                "maximum_liquidation_equity_loss_fraction": 0.0,
            },
        }
    }

    gates = MODULE._liquidation_promotion_gates(candidate, tolerance=tolerance)

    assert gates["liquidation_free"] is False
    assert gates["liquidation_within_tolerance"] is True
    assert gates["split_liquidations_within_tolerance"] is True
    assert gates["liquidation_event_drawdown_within_tolerance"] is True
    assert MODULE._liquidation_safe_for_promotion(gates) is False


def test_strict_deploy_lane_requires_zero_liquidations_and_positive_buffers() -> None:
    candidate = {
        "splits": {
            "train": {"liquidation_count": 0, "minimum_margin_buffer": 2.0},
            "validation": {"liquidation_count": 1, "minimum_margin_buffer": 3.0},
            "oos": {"liquidation_count": 0, "minimum_margin_buffer": 4.0},
        }
    }

    lane = MODULE._strict_deploy_lane(candidate)

    assert lane["lane"] == "strict_deploy"
    assert lane["promotion_allowed"] is False
    assert lane["promotion_eligible"] is False
    assert lane["total_liquidation_count"] == 1
    assert lane["split_status"]["validation"]["liquidation_free"] is False
    assert lane["split_status"]["validation"]["margin_buffer_positive"] is True


def test_diagnostic_nonfatal_lane_reports_recovery_without_promotion() -> None:
    candidate = {
        "splits": {
            "train": {
                "liquidation_count": 0,
                "minimum_margin_buffer": 25.0,
                "final_equity": 10_100.0,
                "liquidation_events": [],
            },
            "validation": {
                "liquidation_count": 1,
                "minimum_margin_buffer": 20.0,
                "maximum_liquidation_event_drawdown": 0.0016,
                "maximum_liquidation_equity_loss_fraction": 0.0005,
                "final_equity": 10_020.0,
                "liquidation_events": [
                    {
                        "pre_liquidation_equity": 10_000.0,
                        "post_liquidation_equity": 9_995.0,
                        "event_drawdown": 0.0016,
                        "equity_loss_fraction": 0.0005,
                        "account_wipeout": False,
                    }
                ],
            },
            "oos": {
                "liquidation_count": 0,
                "minimum_margin_buffer": 30.0,
                "final_equity": 10_200.0,
                "liquidation_events": [],
            },
        }
    }

    lane = MODULE._diagnostic_nonfatal_lane(
        candidate,
        tolerance=MODULE.LiquidationTolerance(
            allowed_total_liquidations=1,
            allowed_split_liquidations=1,
            max_liquidation_event_drawdown=0.005,
            max_liquidation_equity_loss_fraction=0.005,
        ),
    )

    assert lane["diagnostic_only"] is True
    assert lane["promotion_allowed"] is False
    assert lane["promotion_eligible"] is False
    assert lane["total_liquidation_count"] == 1
    assert lane["maximum_event_drawdown"] == pytest.approx(0.0016)
    assert lane["maximum_equity_loss_fraction"] == pytest.approx(0.0005)
    assert lane["split_status"]["validation"]["recovered_to_pre_liquidation_equity"] is True
    assert lane["within_nonfatal_tolerance"] is True


def test_diagnostic_liquidation_lane_reports_components_but_never_promotes() -> None:
    tolerance = MODULE.LiquidationTolerance(
        allowed_total_liquidations=1,
        allowed_split_liquidations=1,
        max_liquidation_event_drawdown=0.005,
        max_liquidation_equity_loss_fraction=0.005,
    )
    gates = MODULE._liquidation_promotion_gates(
        {
            "splits": {
                "train": {
                    "liquidation_count": 0,
                    "minimum_margin_buffer": 25.0,
                    "maximum_liquidation_event_drawdown": 0.0,
                    "maximum_liquidation_equity_loss_fraction": 0.0,
                },
                "validation": {
                    "liquidation_count": 1,
                    "minimum_margin_buffer": 20.0,
                    "maximum_liquidation_event_drawdown": 0.0016,
                    "maximum_liquidation_equity_loss_fraction": 0.0005,
                },
                "oos": {
                    "liquidation_count": 0,
                    "minimum_margin_buffer": 30.0,
                    "maximum_liquidation_event_drawdown": 0.0,
                    "maximum_liquidation_equity_loss_fraction": 0.0,
                },
            }
        },
        tolerance=tolerance,
    )

    assert gates["total_liquidations_within_tolerance"] is True
    assert gates["split_liquidations_within_tolerance"] is True
    assert gates["liquidation_event_drawdown_within_tolerance"] is True
    assert gates["liquidation_equity_loss_within_tolerance"] is True
    assert gates["liquidation_within_tolerance"] is True
    assert gates["liquidation_free"] is False
    assert MODULE._liquidation_safe_for_promotion(gates) is False


def test_tiny_liquidation_tolerance_still_blocks_excess_or_buffer_failure() -> None:
    tolerance = MODULE.LiquidationTolerance(
        allowed_total_liquidations=1,
        allowed_split_liquidations=1,
        max_liquidation_event_drawdown=0.005,
        max_liquidation_equity_loss_fraction=0.005,
    )
    gates = MODULE._liquidation_promotion_gates(
        {
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "validation": {
                    "liquidation_count": 2,
                    "minimum_margin_buffer": 1.0,
                    "maximum_liquidation_event_drawdown": 0.001,
                },
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
            }
        },
        tolerance=tolerance,
    )

    assert gates["split_liquidations_within_tolerance"] is False
    assert gates["liquidation_within_tolerance"] is False
    assert MODULE._liquidation_safe_for_promotion(gates) is False


def test_tolerant_train_validation_selection_still_ignores_locked_oos() -> None:
    tolerance = MODULE.LiquidationTolerance(
        allowed_total_liquidations=1,
        allowed_split_liquidations=1,
        max_liquidation_event_drawdown=0.005,
        max_liquidation_equity_loss_fraction=0.005,
    )
    grid = [
        {
            "leverage": 4,
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 10.0},
                "validation": {
                    "liquidation_count": 1,
                    "minimum_margin_buffer": 5.0,
                    "maximum_liquidation_event_drawdown": 0.001,
                    "maximum_liquidation_equity_loss_fraction": 0.0005,
                },
                "oos": {"liquidation_count": 9, "minimum_margin_buffer": -1.0},
            },
            "train_val_score": 4.0,
        },
        {
            "leverage": 5,
            "splits": {
                "train": {"liquidation_count": 2, "minimum_margin_buffer": 5.0},
                "validation": {"liquidation_count": 0, "minimum_margin_buffer": 5.0},
                "oos": {"liquidation_count": 0, "minimum_margin_buffer": 5.0},
            },
            "train_val_score": 99.0,
        },
    ]

    selected = MODULE._select_train_validation_leverage(grid, tolerance=tolerance)

    assert selected["leverage"] == 4
    assert selected["selection_policy"]["uses_locked_oos_for_selection"] is False


def test_validator_rejects_promoted_candidate_with_unsafe_liquidation_evidence(
    tmp_path: Path,
) -> None:
    validator_path = ROOT / "scripts" / "research" / "validate_profit_moonshot_pass_under_8gb.py"
    spec = importlib.util.spec_from_file_location(
        "validate_profit_moonshot_pass_under_8gb_for_liq", validator_path
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)

    candidate = {
        "research_success_candidate": True,
        "metrics": {
            "train_monthlyized_return": 0.030,
            "validation_monthlyized_return": 0.110,
            "raw_train_monthlyized_return": 0.012,
            "raw_val_monthlyized_return": 0.030,
            "train_max_drawdown": 0.050,
            "validation_max_drawdown": 0.040,
            "train_sharpe": 2.0,
            "train_sortino": 2.0,
            "train_calmar": 4.0,
            "validation_sharpe": 4.5,
            "validation_sortino": 5.0,
            "validation_calmar": 35.0,
            "leverage": 5.0,
            "sleeve_count": 4,
            "locked_oos_monthlyized_return": 0.060,
            "locked_oos_total_return": 0.15,
            "locked_oos_max_drawdown": 0.015,
            "locked_oos_sharpe": 5.0,
            "locked_oos_sortino": 6.0,
            "locked_oos_smart_sortino": 5.0,
            "locked_oos_calmar": 30.0,
            "liquidation_count": 1,
            "minimum_margin_buffer": -0.01,
        },
        "liquidation_aware_validation": {
            "splits": {
                "train": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "val": {"liquidation_count": 0, "minimum_margin_buffer": 1.0},
                "oos": {"liquidation_count": 1, "minimum_margin_buffer": -0.01},
            }
        },
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    rss_summary = tmp_path / "rss_summary.json"
    rss_summary.write_text(json.dumps({"peak_rss_bytes": 512 * 1024 * 1024}), encoding="utf-8")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "source_changed": True,
                "passing_candidate_artifact": "candidate.json",
                "rss_under_8gb_logs": ["rss_summary.json"],
                "test_evidence": [{"command": "pytest", "passed": True}],
                "ci_evidence": [{"workflow": "ci", "conclusion": "success"}],
                "git_evidence": [{"remote": "private/main", "pushed": True}],
            }
        ),
        encoding="utf-8",
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    quality = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    assert quality["passed"] is False
    details = json.loads(quality["detail"])
    assert details["liquidation_count"] == 1
    assert details["minimum_margin_buffer"] == -0.01

    over_limit_bytes = (8 * 1024 * 1024 * 1024) + 1
    rss_too_high = tmp_path / "rss_too_high.json"
    rss_too_high.write_text(
        json.dumps({"peak_rss_bytes": over_limit_bytes}),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "source_changed": True,
                "passing_candidate_artifact": "candidate.json",
                "rss_under_8gb_logs": ["rss_too_high.json"],
                "test_evidence": [{"command": "pytest", "passed": True}],
                "ci_evidence": [{"workflow": "ci", "conclusion": "success"}],
                "git_evidence": [{"remote": "private/main", "pushed": True}],
            }
        ),
        encoding="utf-8",
    )

    rss_payload = validator.validate(result_path, repo_root=tmp_path)

    rss_check = next(
        check for check in rss_payload["checks"] if check["name"] == "rss_under_8gib_evidence"
    )
    rss_details = json.loads(rss_check["detail"])
    assert rss_check["passed"] is False
    assert rss_details[0]["peak_rss_bytes"] == over_limit_bytes
    assert rss_details[0]["under_8gib"] is False
