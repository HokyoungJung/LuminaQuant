from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_gate_module():
    path = ROOT / "scripts" / "research" / "run_alpha_zoo_69_asset_clean_oos_gate.py"
    spec = importlib.util.spec_from_file_location("run_alpha_zoo_69_asset_clean_oos_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def _windows() -> gate.GateWindows:
    return gate.GateWindows(
        train=(pd.Timestamp("2025-01-01T00:00:00"), pd.Timestamp("2025-12-31T23:00:00")),
        validation=(
            pd.Timestamp("2026-01-01T00:00:00"),
            pd.Timestamp("2026-02-28T23:00:00"),
        ),
        locked_oos=(
            pd.Timestamp("2026-03-01T00:00:00"),
            pd.Timestamp("2026-05-06T23:00:00"),
        ),
    )


def test_artifact_oos_contamination_reasons_detects_train_and_validation_overlap() -> None:
    split_policy = {
        "train": {"start": "2025-01-01T00:00:00", "end": "2026-04-04T03:00:00"},
        "validation": {"start": "2026-04-04T04:00:00", "end": "2026-05-30T03:00:00"},
    }

    assert gate.artifact_oos_contamination_reasons(split_policy, _windows()) == [
        "artifact_train_window_overlaps_requested_locked_oos",
        "artifact_validation_window_overlaps_requested_locked_oos",
    ]


def test_artifact_oos_contamination_reasons_allows_pre_oos_freeze() -> None:
    split_policy = {
        "train": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:00:00Z"},
        "validation": {"start": "2026-01-01T00:00:00Z", "end": "2026-02-28T23:00:00Z"},
    }

    assert gate.artifact_oos_contamination_reasons(split_policy, _windows()) == []


def test_locked_oos_gate_reasons_passes_clean_profitable_oos() -> None:
    metrics = {
        "locked_oos": {
            "bar_count": 100,
            "total_return": 0.12,
            "mdd": 0.08,
            "return_per_turnover_proxy_bps": 25.0,
            "trade_event_count": 25,
            "liquidation_count": 0,
            "account_wipeout_bar_count": 0,
        }
    }

    assert gate.locked_oos_gate_reasons(metrics) == []


def test_locked_oos_gate_reasons_fails_unsafe_oos() -> None:
    metrics = {
        "locked_oos": {
            "bar_count": 100,
            "total_return": -0.03,
            "mdd": 0.25,
            "return_per_turnover_proxy_bps": -7.5,
            "trade_event_count": 3,
            "liquidation_count": 1,
            "account_wipeout_bar_count": 2,
        }
    }

    assert gate.locked_oos_gate_reasons(metrics, max_oos_mdd=0.20) == [
        "locked_oos_return_not_positive",
        "locked_oos_mdd_above_0.2000",
        "locked_oos_rpt_-7.500_not_above_10bps",
        "locked_oos_trade_event_count_3_below_20",
        "locked_oos_liquidation_count_nonzero",
        "locked_oos_account_wipeout_nonzero",
    ]


def test_render_markdown_surfaces_gate_failure() -> None:
    markdown = gate.render_markdown(
        {
            "generated_at_utc": "2026-05-31T00:00:00Z",
            "source_artifact": "/tmp/artifact.json",
            "clean_oos_gate_pass": False,
            "clean_oos_gate_reasons": ["locked_oos_return_not_positive"],
            "split_manifest": _windows().as_payload(),
            "rows": {
                "selected_optuna_hybrid_profile": {
                    "hybrid_version": "v3_5",
                    "weight_sets": {
                        "final_weights": {
                            "locked_oos_gate_pass": False,
                            "locked_oos_gate_reasons": ["locked_oos_return_not_positive"],
                            "metrics": {
                                "train": {"total_return": 1.9},
                                "validation": {"total_return": 0.58},
                                "locked_oos": {
                                    "total_return": -0.03,
                                    "mdd": 0.18,
                                    "sharpe": -0.4,
                                    "return_per_turnover_proxy_bps": -7.5,
                                    "trade_event_count": 383,
                                    "liquidation_count": 0,
                                },
                            },
                        }
                    },
                }
            },
        }
    )

    assert "Gate pass: `False`" in markdown
    assert "locked_oos_return_not_positive" in markdown
    assert "`v3_5`" in markdown
