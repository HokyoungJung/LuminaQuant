from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research"
    / "build_carry_trend_production_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("build_carry_trend_production_manifest", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load build_carry_trend_production_manifest module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_manifest_only_keeps_production_ready_carry_rows() -> None:
    payload = MODULE.build_manifest(
        timeframes=["1h", "4h"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )

    rows = list(payload.get("candidates") or [])
    assert payload["artifact_kind"] == MODULE.ARTIFACT_KIND
    assert rows
    assert all(row["strategy_class"] == "CarryTrendFactorRotationStrategy" for row in rows)
    assert all(bool((row.get("metadata") or {}).get("production_ready")) for row in rows)
    assert {row["strategy_timeframe"] for row in rows} == {"1h", "4h"}


def test_write_manifest_emits_latest_files(tmp_path: Path) -> None:
    result = MODULE.write_manifest(
        output_dir=tmp_path,
        timeframes=["1h", "4h"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )

    json_path = Path(result["json_path"])
    md_path = Path(result["md_path"])
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] >= 2
    assert "carry trend production manifest" in md_path.read_text(encoding="utf-8").lower()
