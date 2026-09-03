from __future__ import annotations

import json
from pathlib import Path

from scripts.research.freeze_strategy_shortlist import freeze_shortlist


def test_freeze_shortlist_preserves_rank_and_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "frozen.json"
    source.write_text(
        json.dumps(
            {
                "artifact_kind": "screen",
                "registry_count": 2,
                "result_count": 2,
                "selected": [
                    {"strategy": "first", "validation_mean_sharpe": 2.0},
                    {"strategy": "second", "validation_mean_sharpe": 1.0},
                ],
            }
        )
    )

    result = freeze_shortlist(source, output=output, limit=1)

    assert [row["strategy"] for row in result["selected"]] == ["first"]
    assert result["selection_uses_locked_oos"] is False
    assert result["source"]["registry_count"] == 2
    assert json.loads(output.read_bytes()) == result
