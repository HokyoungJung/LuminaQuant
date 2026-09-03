from __future__ import annotations

import json
from pathlib import Path

from scripts.research.validate_execution_model_ticks import symbols_from_selection


def test_selection_symbols_use_central_strategy_universe(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "selected": [
                    {"strategy": "BitcoinBuyHoldStrategy"},
                    {"strategy": "GoldSilverRatioTrendStrategy"},
                ]
            }
        )
    )

    assert symbols_from_selection(path, limit=10) == (
        "BTC/USDT",
        "XAU/USDT",
        "XAG/USDT",
    )


def test_empty_frozen_selection_has_no_tick_work(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected": []}))

    assert symbols_from_selection(path, limit=10) == ()
