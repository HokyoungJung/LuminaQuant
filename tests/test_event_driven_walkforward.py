from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research import run_event_driven_walkforward as subject


def test_shortlist_accepts_ranked_selection(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "selected": [
                    {"strategy": "PriceVolumeCorrContinuationStrategy"},
                    {"strategy": "BitcoinBuyHoldStrategy"},
                ]
            }
        )
    )

    assert subject._shortlist(path, limit=1) == ("PriceVolumeCorrContinuationStrategy",)


def test_shortlist_ranks_positive_screen_rows_without_oos_feedback(tmp_path: Path) -> None:
    path = tmp_path / "screen.json"
    path.write_text(
        json.dumps(
            {
                "strategy_results": [
                    {
                        "strategy": "low",
                        "status": "pass",
                        "trade_count": 2,
                        "total_return": 0.1,
                        "fast_stats": {"sharpe": 1.0},
                    },
                    {
                        "strategy": "high",
                        "status": "pass",
                        "trade_count": 2,
                        "total_return": 0.05,
                        "fast_stats": {"sharpe": 2.0},
                    },
                    {
                        "strategy": "negative",
                        "status": "pass",
                        "trade_count": 2,
                        "total_return": -0.1,
                        "fast_stats": {"sharpe": 9.0},
                    },
                ]
            }
        )
    )

    assert subject._shortlist(path, limit=10) == ("high", "low")


def test_fold_plan_rejects_validation_oos_overlap(tmp_path: Path) -> None:
    path = tmp_path / "folds.json"
    path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold_id": "fold-01",
                        "validation_start": "2026-01-01T00:00:00Z",
                        "validation_end": "2026-02-01T00:00:00Z",
                        "locked_oos_start": "2026-01-15T00:00:00Z",
                        "locked_oos_end": "2026-03-01T00:00:00Z",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        subject._folds(path)
