from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_salvage_module():
    path = ROOT / "scripts" / "research" / "run_alpha_zoo_69_asset_diverse_salvage.py"
    spec = importlib.util.spec_from_file_location("run_alpha_zoo_69_asset_diverse_salvage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


salvage = _load_salvage_module()


def _source_payload() -> dict:
    return {
        "artifact_kind": "source",
        "universe": {"symbol_count": 4, "symbols": ["BTCUSDT", "ETHUSDT", "SPYUSDT", "XAUUSDT"]},
        "timeframes": ["1h"],
        "data_coverage": {"data_root": "/tmp/data"},
        "split_policy": {
            "train": {"start": "2025-01-01T00:00:00", "end": "2025-12-31T23:00:00"},
            "validation": {"start": "2026-01-01T00:00:00", "end": "2026-02-28T23:00:00"},
        },
        "train_eligibility": {
            "train_eligible_symbols": ["BTCUSDT", "ETHUSDT", "XAUUSDT"],
            "train_ineligible_symbols": ["SPYUSDT"],
        },
        "profile_rows": [
            {
                "profile_id": "growth_profile",
                "train_return": 2.0,
                "validation_return": 1.0,
                "validation_mdd": 0.20,
                "train_return_per_turnover_proxy_bps": 80.0,
                "validation_return_per_turnover_proxy_bps": 150.0,
                "ready_for_paper": False,
            },
            {
                "profile_id": "balanced_profile",
                "train_return": 1.0,
                "validation_return": 0.5,
                "validation_mdd": 0.10,
                "train_return_per_turnover_proxy_bps": 60.0,
                "validation_return_per_turnover_proxy_bps": 120.0,
                "ready_for_paper": True,
            },
        ],
        "selected_sleeve_rows": [
            {
                "symbol": "BTCUSDT",
                "profile_id": "growth_profile",
                "timeframe": "1h",
                "side": "short_only",
                "family": "trend",
                "notional_fraction": 0.5,
                "sleeve_multiplier": 2.0,
                "train_return": 0.2,
                "validation_return": 0.1,
                "validation_mdd": 0.08,
            },
            {
                "symbol": "ETHUSDT",
                "profile_id": "growth_profile",
                "timeframe": "1h",
                "side": "short_only",
                "family": "trend",
                "notional_fraction": 0.5,
                "sleeve_multiplier": 2.0,
                "train_return": 0.2,
                "validation_return": 0.1,
                "validation_mdd": 0.08,
            },
            {
                "symbol": "BTCUSDT",
                "profile_id": "balanced_profile",
                "timeframe": "1h",
                "side": "long_only",
                "family": "trend",
                "notional_fraction": 0.5,
                "sleeve_multiplier": 1.0,
                "train_return": 0.1,
                "validation_return": 0.1,
                "validation_mdd": 0.03,
            },
            {
                "symbol": "ETHUSDT",
                "profile_id": "balanced_profile",
                "timeframe": "1h",
                "side": "long_only",
                "family": "trend",
                "notional_fraction": 0.5,
                "sleeve_multiplier": 1.0,
                "train_return": 0.1,
                "validation_return": 0.1,
                "validation_mdd": 0.03,
            },
            {
                "symbol": "XAUUSDT",
                "profile_id": "balanced_profile",
                "timeframe": "1h",
                "side": "long_only",
                "family": "trend",
                "notional_fraction": 0.25,
                "sleeve_multiplier": 1.0,
                "train_return": 0.02,
                "validation_return": 0.05,
                "validation_mdd": 0.04,
            },
        ],
    }


def test_select_source_profile_prefers_ready_max_coverage_balanced_profile() -> None:
    profile_id, rows, profile_row = salvage.select_source_profile(_source_payload())

    assert profile_id == "balanced_profile"
    assert len(rows) == 3
    assert profile_row["ready_for_paper"] is True


def test_scale_rows_to_target_gross_preserves_symbols_and_caps_gross() -> None:
    _, rows, _ = salvage.select_source_profile(_source_payload())

    scaled, policy = salvage.scale_rows_to_target_gross(
        rows,
        target_gross=1.0,
        output_profile_id="diverse_profile",
    )

    assert {row["symbol"] for row in scaled} == {"BTCUSDT", "ETHUSDT", "XAUUSDT"}
    assert all(row["profile_id"] == "diverse_profile" for row in scaled)
    assert all(row["source_profile_id"] == "balanced_profile" for row in scaled)
    assert policy["source_gross_notional_fraction"] == 1.25
    assert round(policy["effective_gross_notional_fraction"], 8) == 1.0
    assert round(policy["scale_factor"], 8) == 0.8


def test_build_asset_inclusion_manifest_keeps_future_watchlist() -> None:
    _, rows, _ = salvage.select_source_profile(_source_payload())
    scaled, _ = salvage.scale_rows_to_target_gross(
        rows,
        target_gross=1.0,
        output_profile_id="diverse_profile",
    )

    manifest = salvage.build_asset_inclusion_manifest(
        universe_symbols=["BTCUSDT", "ETHUSDT", "SPYUSDT", "XAUUSDT"],
        selected_rows=scaled,
        train_eligible_symbols=["BTCUSDT", "ETHUSDT", "XAUUSDT"],
        train_ineligible_symbols=["SPYUSDT"],
    )

    status_by_symbol = {row["symbol"]: row["status"] for row in manifest}
    assert status_by_symbol == {
        "BTCUSDT": "tradable_now_train_eligible",
        "ETHUSDT": "tradable_now_train_eligible",
        "SPYUSDT": "future_watchlist_insufficient_train_history",
        "XAUUSDT": "tradable_now_train_eligible",
    }


def test_build_payload_marks_oos_non_selection_and_watchlist(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(salvage.json.dumps(_source_payload()), encoding="utf-8")

    payload = salvage.build_payload(
        Namespace(
            source_artifact=str(source),
            source_profile_id=None,
            output_profile_id="diverse_profile",
            target_gross=1.0,
            max_validation_mdd=0.12,
            allow_upscale=False,
        )
    )

    policy = payload["diversity_policy"]
    assert policy["locked_oos_used_for_selection"] is False
    assert policy["current_tradable_symbol_count"] == 3
    assert policy["future_watchlist_symbols"] == ["SPYUSDT"]
    assert payload["selected_optuna_hybrid_profile"]["weights"] == {"diverse_profile": 1.0}
    assert payload["real_execution_allowed"] is False
