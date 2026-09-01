from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "pair_spread_4h_xpt_xpd_retune.py"
SPEC = importlib.util.spec_from_file_location("pair_spread_4h_xpt_xpd_retune", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load pair_spread_4h_xpt_xpd_retune module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_pair_spread_4h_xpt_xpd_retune = MODULE.build_pair_spread_4h_xpt_xpd_retune
write_pair_spread_4h_xpt_xpd_retune = MODULE.write_pair_spread_4h_xpt_xpd_retune


def _manifest(total_days: int) -> dict[str, object]:
    return {
        "adaptive_windows": {
            "profile": "metals_4h",
            "total_days": total_days,
            "common_start": "2026-01-30T10:15:00+00:00",
            "common_end": "2026-03-07T11:02:00+00:00",
        },
        "windows": {
            "train_start": "2026-01-30T10:15:00+00:00",
            "train_end_exclusive": "2026-02-17T10:15:00+00:00",
            "val_start": "2026-02-17T10:15:00+00:00",
            "val_end_exclusive": "2026-02-25T10:15:00+00:00",
            "actual_oos_end_exclusive": "2026-03-07T11:02:00+00:00",
        },
    }


def _good_row() -> dict[str, object]:
    return {
        "name": "pair_spread_4h_strict_3p2_xptusdt_xpdusdt_3.2_0.95",
        "val": {"sharpe": 1.4},
        "oos": {
            "return": 0.021,
            "sharpe": 1.3,
            "sortino": 1.8,
            "calmar": 2.5,
            "max_drawdown": 0.08,
            "pbo": 0.30,
            "trade_count": 5.0,
        },
    }


def test_pair_retune_blocks_survivor_when_coverage_is_too_short():
    payload = build_pair_spread_4h_xpt_xpd_retune(
        manifest_payload=_manifest(total_days=36),
        report_rows=[_good_row()],
    )

    assert payload["coverage_guard"]["pass"] is False
    assert payload["survives"] is False
    assert "coverage_days" in payload["blockers"]
    assert "coverage_days" in payload["top_candidates"][0]["survivor_blockers"]


def test_pair_retune_allows_survivor_when_metrics_and_coverage_pass():
    payload = build_pair_spread_4h_xpt_xpd_retune(
        manifest_payload=_manifest(total_days=90),
        report_rows=[_good_row()],
    )

    assert payload["coverage_guard"]["pass"] is True
    assert payload["survives"] is True
    assert payload["survivor"]["name"] == _good_row()["name"]


def test_write_pair_retune_writes_files(tmp_path: Path):
    result = write_pair_spread_4h_xpt_xpd_retune(
        report_root=tmp_path,
        manifest_payload=_manifest(total_days=36),
        report_rows=[_good_row()],
    )

    json_path = Path(result["json_path"])
    md_path = Path(result["md_path"])
    assert json_path.exists()
    assert md_path.exists()
