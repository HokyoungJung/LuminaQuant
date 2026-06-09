from __future__ import annotations

import json
from pathlib import Path

from scripts.research import run_alpha_zoo_existing_candidate_reuse_selector as module


def _row(fold_id: str, model_id: str, **overrides):
    row = {
        "fold_id": fold_id,
        "model_id": model_id,
        "family": "cross_asset_lead_lag_momentum",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "train_return": 0.20,
        "validation_return": 0.10,
        "train_mdd": 0.05,
        "validation_mdd": 0.04,
        "train_trade_event_count": 50,
        "validation_trade_event_count": 20,
        "train_return_per_turnover_proxy_bps": 30.0,
        "validation_return_per_turnover_proxy_bps": 20.0,
        "locked_oos_return_report_only": -0.10,
        "locked_oos_mdd_report_only": 0.05,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
    }
    row.update(overrides)
    return row


def test_pick_variant_uses_train_validation_not_locked_oos() -> None:
    picked = module._pick_variant(
        [
            _row(
                "2026-01",
                "bad_oos_winner",
                validation_return=0.03,
                locked_oos_return_report_only=2.0,
            ),
            _row(
                "2026-01",
                "validation_winner",
                validation_return=0.20,
                locked_oos_return_report_only=-0.5,
            ),
        ],
        "robust_top1",
    )

    assert len(picked) == 1
    assert picked[0]["model_id"] == "validation_winner"


def test_diverse_variant_prefers_unique_family_and_symbol() -> None:
    picked = module._pick_variant(
        [
            _row("2026-01", "a", family="f1", symbol="BTCUSDT", validation_return=0.30),
            _row("2026-01", "b", family="f1", symbol="BTCUSDT", validation_return=0.29),
            _row("2026-01", "c", family="f2", symbol="ETHUSDT", validation_return=0.12),
            _row("2026-01", "d", family="f3", symbol="SOLUSDT", validation_return=0.11),
        ],
        "robust_diverse3_equal",
    )

    assert [row["model_id"] for row in picked] == ["a", "c", "d"]


def test_quality_variant_applies_stricter_train_validation_filters() -> None:
    picked = module._pick_variant(
        [
            _row(
                "2026-01",
                "loose_high_score",
                train_return=0.60,
                validation_return=0.20,
                validation_mdd=0.04,
            ),
            _row(
                "2026-01",
                "strict_quality",
                train_return=0.30,
                validation_return=0.18,
                validation_mdd=0.04,
            ),
            _row(
                "2026-01",
                "high_drawdown",
                train_return=0.25,
                validation_return=0.22,
                validation_mdd=0.08,
            ),
        ],
        "robust_quality_v1_top1",
    )

    assert len(picked) == 1
    assert picked[0]["model_id"] == "strict_quality"


def test_run_writes_non_promotable_reuse_artifact(tmp_path: Path) -> None:
    source_json = tmp_path / "source.json"
    source_json.write_text(
        json.dumps(
            {
                "pre_registered_search_space_sha256": "abc",
                "candidate_rows": [
                    _row(
                        "2026-01", "a", validation_return=0.20, locked_oos_return_report_only=0.05
                    ),
                    _row(
                        "2026-01", "b", validation_return=0.10, locked_oos_return_report_only=-0.10
                    ),
                    _row(
                        "2026-02", "c", validation_return=0.20, locked_oos_return_report_only=0.03
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = module.run(source_json=source_json, output_dir=tmp_path / "out")

    assert payload["selection_inputs"] == ["train", "validation"]
    assert payload["locked_oos_role"] == "report_only_after_reuse_selector_freeze"
    assert payload["decision"]["real_money_execution"] is False
    assert payload["best_variant_by_report_oos"] in module.VARIANTS
    assert "robust_quality_v1_top1" in payload["variants"]
    assert Path(payload["output_paths"]["json"]).exists()
    assert Path(payload["output_paths"]["markdown"]).exists()
