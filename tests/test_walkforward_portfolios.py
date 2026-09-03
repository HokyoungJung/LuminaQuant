from __future__ import annotations

import json
from pathlib import Path

from scripts.research.build_walkforward_portfolios import build_portfolios


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_portfolio_builder_skips_single_survivor(tmp_path: Path) -> None:
    walkforward = tmp_path / "walkforward.json"
    selection = tmp_path / "selection.json"
    _write(walkforward, {"cells": []})
    _write(selection, {"selected": [{"strategy": "only"}]})

    result = build_portfolios(walkforward, selection)

    assert result["status"] == "skip_insufficient_survivors"
    assert result["portfolios"] == {}


def test_portfolio_builder_runs_hrp_erc_and_nco_on_validation_only(
    tmp_path: Path,
) -> None:
    walkforward = tmp_path / "walkforward.json"
    selection = tmp_path / "selection.json"
    strategies = ["left", "right"]
    cells = []
    for index in range(6):
        for offset, strategy in enumerate(strategies):
            cells.append(
                {
                    "phase": "validation",
                    "strategy": strategy,
                    "fold_id": f"fold-{index}",
                    "total_return": (index + 1) * 0.01 * (1 if offset == 0 else -0.5),
                }
            )
            cells.append(
                {
                    "phase": "locked_oos",
                    "strategy": strategy,
                    "fold_id": f"fold-{index}",
                    "total_return": 999.0,
                }
            )
    _write(walkforward, {"cells": cells})
    _write(selection, {"selected": [{"strategy": name} for name in strategies]})

    result = build_portfolios(walkforward, selection)

    assert result["status"] == "complete"
    assert set(result["portfolios"]) == {
        "equal_weight",
        "erc",
        "hrp_threshold",
        "hrp_dendrogram",
        "herc",
        "nco",
    }
    assert result["selection_inputs"] == ["validation"]
    for cell in cells:
        if cell["phase"] == "locked_oos":
            cell["total_return"] = -999.0
    _write(walkforward, {"cells": cells})
    changed_oos = build_portfolios(walkforward, selection)
    assert changed_oos["portfolios"] == result["portfolios"]
