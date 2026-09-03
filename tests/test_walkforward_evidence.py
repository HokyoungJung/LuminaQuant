from __future__ import annotations

import json
from pathlib import Path

from lumina_quant.backtesting.walkforward_evidence import (
    build_report_only_evaluation,
    select_finalists,
)


def _artifact(path: Path) -> None:
    cells = []
    for strategy, validation, oos in (("stable", 0.03, -0.01), ("weak", -0.01, 9.0)):
        cells.extend(
            [
                {
                    "strategy": strategy,
                    "phase": "validation",
                    "status": "pass",
                    "total_return": validation,
                    "fast_stats": {"sharpe": validation * 100},
                },
                {
                    "strategy": strategy,
                    "phase": "locked_oos",
                    "status": "pass",
                    "total_return": oos,
                    "fast_stats": {"sharpe": oos * 100},
                },
            ]
        )
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "lumina_quant.event_driven_walkforward.v1",
                "selection_uses_locked_oos": False,
                "cells": cells,
            }
        )
    )


def test_selection_never_uses_locked_oos(tmp_path: Path) -> None:
    walkforward = tmp_path / "walkforward.json"
    _artifact(walkforward)

    result = select_finalists(
        walkforward,
        top_n=10,
        minimum_pass_ratio=1.0,
        minimum_mean_sharpe=0.35,
    )

    assert [row["strategy"] for row in result["selected"]] == ["stable"]
    assert all(not any(key.startswith("locked_oos") for key in row) for row in result["selected"])


def test_report_attaches_oos_only_after_selection_freeze(tmp_path: Path) -> None:
    walkforward = tmp_path / "walkforward.json"
    selection = tmp_path / "selection.json"
    _artifact(walkforward)
    selection.write_text(
        json.dumps(
            select_finalists(
                walkforward,
                top_n=10,
                minimum_pass_ratio=1.0,
                minimum_mean_sharpe=0.35,
            )
        )
    )

    result = build_report_only_evaluation(walkforward, selection)

    assert result["rows"] == [
        {
            "strategy": "stable",
            "locked_oos_fold_count": 1,
            "locked_oos_pass_ratio": 1.0,
            "locked_oos_mean_return": -0.01,
            "locked_oos_mean_sharpe": -1.0,
            "locked_oos_positive_fold_ratio": 0.0,
        }
    ]
    assert result["selection_uses_locked_oos"] is False
