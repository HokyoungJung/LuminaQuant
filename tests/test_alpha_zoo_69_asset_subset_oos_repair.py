from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "research" / "run_alpha_zoo_69_asset_subset_oos_repair.py"
    spec = importlib.util.spec_from_file_location("run_alpha_zoo_69_asset_subset_oos_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repair = _load_module()


def _row(symbol: str, *, train: float, val: float, val_mdd: float, train_rpt: float = 50.0, val_rpt: float = 50.0, gross: float = 0.5, score_boost: float = 0.0) -> dict:
    return {
        "symbol": symbol,
        "profile_id": f"profile_{symbol}_{score_boost}",
        "timeframe": "1h",
        "side": "long_only",
        "family": "trend",
        "notional_fraction": gross,
        "sleeve_multiplier": 1.0,
        "train_return": train + score_boost,
        "validation_return": val,
        "validation_mdd": val_mdd,
        "train_mdd": 0.05,
        "train_return_per_turnover_proxy_bps": train_rpt,
        "validation_return_per_turnover_proxy_bps": val_rpt,
    }


def test_row_gate_requires_train_validation_quality_without_oos() -> None:
    assert repair.row_passes_tv_quality_gate(_row("A", train=0.20, val=0.12, val_mdd=0.04))
    assert not repair.row_passes_tv_quality_gate(_row("A", train=0.20, val=0.09, val_mdd=0.04))
    assert not repair.row_passes_tv_quality_gate(_row("A", train=0.20, val=0.12, val_mdd=0.07))
    assert not repair.row_passes_tv_quality_gate(_row("A", train=0.10, val=0.12, val_mdd=0.04))
    assert repair.row_passes_tv_quality_gate(
        _row("A", train=0.10, val=0.12, val_mdd=0.04),
        selection_mode="diversified_watch_core",
    )


def test_select_keeps_one_best_per_symbol_and_caps_sleeves() -> None:
    rows = [
        _row("A", train=0.20, val=0.12, val_mdd=0.04, score_boost=0.00),
        _row("A", train=0.35, val=0.12, val_mdd=0.04, score_boost=0.05),
        _row("B", train=0.30, val=0.14, val_mdd=0.05),
        _row("C", train=0.15, val=0.11, val_mdd=0.03),
    ]

    selected = repair.select_tv_quality_core_rows(rows, max_sleeves=2)

    assert len(selected) == 2
    assert len({row["symbol"] for row in selected}) == 2
    assert selected[0]["symbol"] in {"A", "B"}
    assert {row["symbol"] for row in selected}.issubset({"A", "B", "C"})
    assert all("tv_quality_score" in row for row in selected)


def test_build_payload_preserves_69_monitor_manifest(tmp_path: Path) -> None:
    source = {
        "artifact_kind": "source",
        "universe": {"symbol_count": 3, "symbols": ["A", "B", "C"]},
        "timeframes": ["1h"],
        "data_coverage": {"data_root": "/tmp/data"},
        "split_policy": {
            "train": {"start": "2025-01-01T00:00:00", "end": "2025-12-31T23:00:00"},
            "validation": {"start": "2026-01-01T00:00:00", "end": "2026-02-28T23:00:00"},
        },
        "train_eligibility": {
            "train_eligible_symbols": ["A", "B"],
            "train_ineligible_symbols": ["C"],
        },
        "selected_sleeve_rows": [
            _row("A", train=0.20, val=0.12, val_mdd=0.04, gross=0.7),
            _row("B", train=0.30, val=0.14, val_mdd=0.05, gross=0.8),
            _row("C", train=0.08, val=0.20, val_mdd=0.03, gross=0.9),
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(repair.json.dumps(source), encoding="utf-8")

    payload = repair.build_payload(
        Namespace(
            source_artifact=str(source_path),
            output_profile_id="subset",
            target_gross=1.0,
            min_validation_return=0.10,
            max_validation_mdd=0.06,
            min_rpt_bps=10.0,
            max_sleeves=4,
            selection_mode="sparse_quality",
            allow_upscale=False,
        )
    )

    assert payload["subset_selection_policy"]["locked_oos_used_for_selection"] is False
    assert payload["candidate_pool_policy"]["candidate_pool_symbol_count"] == 3
    assert payload["candidate_pool_policy"]["all_universe_symbols_remain_candidates"] is True
    assert payload["candidate_pool_policy"]["candidate_pool_is_not_equal_to_current_positions"] is True
    assert payload["subset_selection_policy"]["current_tradable_symbol_count"] == 2
    assert len(payload["asset_inclusion_manifest"]) == 3
    status = {row["symbol"]: row["status"] for row in payload["asset_inclusion_manifest"]}
    assert status == {
        "A": "sparse_core_tradable_now",
        "B": "sparse_core_tradable_now",
        "C": "future_watchlist_insufficient_train_history",
    }
    assert payload["real_execution_allowed"] is False


def test_build_payload_can_use_explicit_source_row_indices(tmp_path: Path) -> None:
    source = {
        "artifact_kind": "source",
        "universe": {"symbol_count": 3, "symbols": ["A", "B", "C"]},
        "timeframes": ["1h"],
        "data_coverage": {"data_root": "/tmp/data"},
        "split_policy": {},
        "train_eligibility": {
            "train_eligible_symbols": ["A", "B", "C"],
            "train_ineligible_symbols": [],
        },
        "selected_sleeve_rows": [
            _row("A", train=0.20, val=0.12, val_mdd=0.04, gross=0.7),
            _row("B", train=0.30, val=0.14, val_mdd=0.05, gross=0.8),
            _row("C", train=0.08, val=0.20, val_mdd=0.03, gross=0.9),
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(repair.json.dumps(source), encoding="utf-8")

    payload = repair.build_payload(
        Namespace(
            source_artifact=str(source_path),
            output_profile_id="explicit",
            target_gross=1.0,
            min_validation_return=0.10,
            max_validation_mdd=0.06,
            min_rpt_bps=10.0,
            max_sleeves=4,
            selection_mode="explicit_indices",
            selected_row_indices="0,2",
            allow_upscale=False,
        )
    )

    assert payload["subset_selection_policy"]["selection_mode"] == "explicit_indices"
    assert payload["subset_selection_policy"]["selected_row_indices"] == [0, 2]
    assert payload["subset_selection_policy"]["current_tradable_symbols"] == ["A", "C"]
