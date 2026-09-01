from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "build_portfolio_one_shot_incumbent_bundle.py"
SPEC = importlib.util.spec_from_file_location(
    "build_portfolio_one_shot_incumbent_bundle", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load build_portfolio_one_shot_incumbent_bundle module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _candidate(candidate_id: str, name: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "name": name,
        "strategy_class": "StubStrategy",
        "strategy_timeframe": "1h",
        "family": "trend",
        "symbols": ["BTC/USDT"],
        "train": {"return": 0.01, "sharpe": 1.0},
        "val": {"return": 0.02, "sharpe": 1.5},
        "oos": {"return": 0.03, "sharpe": 1.8},
        "return_streams": {
            "train": [{"t": 1, "v": 0.01}],
            "val": [{"t": 2, "v": 0.02}],
            "oos": [{"t": 3, "v": 0.03}],
        },
        "notes": "candidate note",
        "metadata": {},
    }


def test_build_portfolio_one_shot_incumbent_bundle_filters_to_saved_weights(tmp_path: Path) -> None:
    current_bundle = tmp_path / "current_bundle.json"
    current_portfolio = tmp_path / "current_portfolio.json"

    current_bundle.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_one_shot_current_bundle",
                "candidates": [
                    _candidate("a", "alpha"),
                    _candidate("b", "beta"),
                    _candidate("c", "gamma"),
                    _candidate("d", "delta"),
                ],
            }
        ),
        encoding="utf-8",
    )
    current_portfolio.write_text(
        json.dumps(
            {
                "weights": [
                    {"candidate_id": "a", "name": "alpha", "weight": 0.28, "weight_share": 0.35},
                    {"candidate_id": "c", "name": "gamma", "weight": 0.32, "weight_share": 0.40},
                    {"candidate_id": "b", "name": "beta", "weight": 0.20, "weight_share": 0.25},
                ],
                "gross_exposure": 0.80,
                "cash_weight": 0.20,
                "portfolio_metrics": {
                    "oos": {
                        "total_return": 0.05,
                        "sharpe": 1.7,
                        "sortino": 2.1,
                        "calmar": 4.0,
                        "max_drawdown": 0.06,
                        "volatility": 0.12,
                    }
                },
                "portfolio_return_streams": {"oos": [{"t": 3, "v": 0.01}]},
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_one_shot_incumbent_bundle(
        current_bundle_path=current_bundle,
        current_portfolio_path=current_portfolio,
    )

    assert payload["artifact_kind"] == "portfolio_one_shot_incumbent_bundle"
    assert payload["selection_basis"] == "incumbent_saved_one_shot_weights"
    assert payload["split_contract"]["oos_start"] == "2026-02-01T00:00:00Z"
    assert [row["candidate_id"] for row in payload["candidates"]] == ["a", "c", "b"]
    assert [row["portfolio_weight"] for row in payload["candidates"]] == [0.35, 0.40, 0.25]
    assert [row["gross_weight"] for row in payload["candidates"]] == [0.28, 0.32, 0.20]
    assert abs(payload["incumbent_summary"]["weight_total"] - 1.0) < 1e-12
    assert abs(payload["incumbent_summary"]["gross_exposure"] - 0.80) < 1e-12
    assert abs(payload["incumbent_summary"]["cash_weight"] - 0.20) < 1e-12
    assert payload["candidates"][0]["notes"][0] == "candidate note"


def test_build_portfolio_one_shot_incumbent_bundle_raises_when_weight_row_is_missing(
    tmp_path: Path,
) -> None:
    current_bundle = tmp_path / "current_bundle.json"
    current_portfolio = tmp_path / "current_portfolio.json"
    current_bundle.write_text(
        json.dumps({"candidates": [_candidate("a", "alpha")]}), encoding="utf-8"
    )
    current_portfolio.write_text(
        json.dumps({"weights": [{"candidate_id": "missing", "name": "missing", "weight": 1.0}]}),
        encoding="utf-8",
    )

    try:
        MODULE.build_portfolio_one_shot_incumbent_bundle(
            current_bundle_path=current_bundle,
            current_portfolio_path=current_portfolio,
        )
    except RuntimeError as exc:
        assert "missing from the current bundle" in str(exc)
    else:  # pragma: no cover - assertion is the point.
        raise AssertionError("expected missing incumbent row to raise RuntimeError")


def test_write_portfolio_one_shot_incumbent_bundle_writes_latest_files(tmp_path: Path) -> None:
    current_bundle = tmp_path / "current_bundle.json"
    current_portfolio = tmp_path / "current_portfolio.json"
    output_json = tmp_path / "portfolio_one_shot_incumbent_bundle_latest.json"
    output_md = tmp_path / "portfolio_one_shot_incumbent_bundle_latest.md"
    current_bundle.write_text(
        json.dumps({"candidates": [_candidate("a", "alpha")]}), encoding="utf-8"
    )
    current_portfolio.write_text(
        json.dumps(
            {
                "weights": [
                    {"candidate_id": "a", "name": "alpha", "weight": 0.7, "weight_share": 1.0}
                ],
                "gross_exposure": 0.7,
                "cash_weight": 0.3,
                "portfolio_metrics": {"oos": {"total_return": 0.01, "sharpe": 1.1}},
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.write_portfolio_one_shot_incumbent_bundle(
        current_bundle_path=current_bundle,
        current_portfolio_path=current_portfolio,
        output_json_path=output_json,
        output_md_path=output_md,
    )

    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["incumbent_summary"]["component_count"] == 1
    assert abs(written["cash_weight"] - 0.3) < 1e-12
    assert "portfolio one-shot incumbent bundle" in output_md.read_text(encoding="utf-8")
