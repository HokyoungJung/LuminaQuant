from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research"
    / "build_production_guarded_portfolio.py"
)
SPEC = importlib.util.spec_from_file_location("build_production_guarded_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load build_production_guarded_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _portfolio_payload(*, returns: dict[str, list[float]]) -> dict:
    def _stream(values: list[float], start_day: int) -> list[dict[str, float | str]]:
        return [
            {
                "datetime": f"2026-0{start_day + idx}-01T00:00:00Z",
                "t": f"2026-0{start_day + idx}-01T00:00:00Z",
                "v": value,
            }
            for idx, value in enumerate(values)
        ]

    streams = {
        "train": _stream(returns["train"], 1),
        "val": _stream(returns["val"], 2),
        "oos": _stream(returns["oos"], 3),
    }
    metrics = MODULE.evaluate_weighted_portfolio(
        [{"_saved_weight": 1.0, "return_streams": streams, "train": {}, "val": {}, "oos": {}}]
    )["portfolio_metrics"]
    return {"portfolio_metrics": metrics, "portfolio_return_streams": streams}


def _hybrid_payload(*, returns: dict[str, list[float]]) -> dict:
    daily_returns = returns["train"] + returns["val"] + returns["oos"]
    dates = [
        "2025-01-01",
        "2026-01-01",
        "2026-03-01",
    ]
    # expand one date per return in split order
    expanded_dates = []
    for base, values in zip(dates, (returns["train"], returns["val"], returns["oos"]), strict=True):
        year, month, _day = base.split("-")
        for idx, _ in enumerate(values, start=1):
            expanded_dates.append(f"{year}-{month}-{idx:02d}")
    metrics = _portfolio_payload(returns=returns)["portfolio_metrics"]
    return {
        "split_windows": {
            "train_start": "2025-01-01",
            "train_end_inclusive": f"2025-01-{len(returns['train']):02d}",
            "val_start": "2026-01-01",
            "val_end_inclusive": f"2026-01-{len(returns['val']):02d}",
            "oos_start": "2026-03-01",
            "oos_end_inclusive": f"2026-03-{len(returns['oos']):02d}",
        },
        "scenarios": {
            "refreshed_latest_tail": {
                "dates": expanded_dates,
                "daily_returns": daily_returns,
                "split_metrics": metrics,
                "final_allocation": {
                    "weights": {"balanced_overlay_80_20": 0.6},
                    "cash_weight": 0.4,
                },
            }
        },
        "readiness": {"recommended_stage": "pilot_candidate"},
    }


def _carry_report_row(name: str, *, train: list[float], val: list[float], oos: list[float]) -> dict:
    payload = _portfolio_payload(returns={"train": train, "val": val, "oos": oos})
    metrics = payload["portfolio_metrics"]
    return {
        "candidate_id": name,
        "name": name,
        "strategy_class": "CarryTrendFactorRotationStrategy",
        "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
        "train": metrics["train"],
        "val": metrics["val"],
        "oos": metrics["oos"],
        "return_streams": payload["portfolio_return_streams"],
    }


def test_build_production_guarded_portfolio_can_include_positive_carry_candidate(
    tmp_path: Path,
) -> None:
    hybrid_path = tmp_path / "hybrid.json"
    static_path = tmp_path / "static.json"
    incumbent_path = tmp_path / "incumbent.json"
    carry_report = tmp_path / "candidate_research_latest.json"

    hybrid_path.write_text(
        json.dumps(
            _hybrid_payload(
                returns={"train": [0.01, 0.01], "val": [0.015, 0.01], "oos": [0.02, 0.015]}
            )
        ),
        encoding="utf-8",
    )
    static_path.write_text(
        json.dumps(
            _portfolio_payload(
                returns={"train": [0.008, 0.009], "val": [0.01, 0.011], "oos": [0.012, 0.011]}
            )
        ),
        encoding="utf-8",
    )
    incumbent_path.write_text(
        json.dumps(
            _portfolio_payload(
                returns={"train": [0.007, 0.007], "val": [0.008, 0.008], "oos": [0.009, 0.009]}
            )
        ),
        encoding="utf-8",
    )
    carry_report.write_text(
        json.dumps(
            {
                "candidates": [
                    _carry_report_row(
                        "carry_trend_factor_rotation_1h_guarded",
                        train=[0.012, 0.012],
                        val=[0.02, 0.018],
                        oos=[0.03, 0.025],
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_production_guarded_portfolio(
        hybrid_path=hybrid_path,
        static_blend_path=static_path,
        incumbent_path=incumbent_path,
        carry_report_glob=str(carry_report),
        throttle=MODULE.DrawdownThrottleConfig(
            soft_drawdown=0.01,
            hard_drawdown=0.02,
            stop_drawdown=0.03,
            soft_scale=0.85,
            hard_scale=0.60,
            stop_scale=0.35,
        ),
    )

    assert payload["artifact_kind"] == MODULE.ARTIFACT_KIND
    assert payload["cash_weight"] > 0.0
    assert (
        payload["carry_candidate_considered"]["selected_name"]
        == "carry_trend_factor_rotation_1h_guarded"
    )
    assert payload["carry_candidate_included"] is True
    assert any(
        row["candidate_id"] == "carry_trend_factor_rotation_1h_guarded"
        for row in payload["weights"]
    )


def test_build_production_guarded_portfolio_excludes_unhealthy_carry_and_throttles_drawdown(
    tmp_path: Path,
) -> None:
    hybrid_path = tmp_path / "hybrid.json"
    static_path = tmp_path / "static.json"
    incumbent_path = tmp_path / "incumbent.json"
    carry_report = tmp_path / "candidate_research_latest.json"

    hybrid_path.write_text(
        json.dumps(
            _hybrid_payload(
                returns={
                    "train": [0.02, -0.03, 0.03],
                    "val": [0.015, -0.025, 0.02],
                    "oos": [0.01, -0.03, 0.025],
                }
            )
        ),
        encoding="utf-8",
    )
    static_path.write_text(
        json.dumps(
            _portfolio_payload(
                returns={
                    "train": [0.01, 0.0, 0.01],
                    "val": [0.008, 0.0, 0.008],
                    "oos": [0.006, 0.0, 0.006],
                }
            )
        ),
        encoding="utf-8",
    )
    incumbent_path.write_text(
        json.dumps(
            _portfolio_payload(
                returns={
                    "train": [0.009, 0.0, 0.009],
                    "val": [0.007, 0.0, 0.007],
                    "oos": [0.005, 0.0, 0.005],
                }
            )
        ),
        encoding="utf-8",
    )
    carry_report.write_text(
        json.dumps(
            {
                "candidates": [
                    _carry_report_row(
                        "carry_trend_factor_rotation_1h_bad",
                        train=[-0.01, -0.01],
                        val=[-0.02, -0.01],
                        oos=[-0.03, -0.02],
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_production_guarded_portfolio(
        hybrid_path=hybrid_path,
        static_blend_path=static_path,
        incumbent_path=incumbent_path,
        carry_report_glob=str(carry_report),
        throttle=MODULE.DrawdownThrottleConfig(
            soft_drawdown=0.01,
            hard_drawdown=0.02,
            stop_drawdown=0.03,
            soft_scale=0.85,
            hard_scale=0.60,
            stop_scale=0.35,
        ),
    )

    assert payload["carry_candidate_included"] is False
    assert (
        payload["carry_candidate_considered"]["excluded_reason"]
        == "no_carry_candidate_cleared_production_safety_filters"
    )
    throttled, schedule = MODULE._apply_drawdown_throttle(
        {
            "train": [
                {"datetime": "2025-01-01T00:00:00Z", "t": "2025-01-01T00:00:00Z", "v": 0.01},
                {"datetime": "2025-01-02T00:00:00Z", "t": "2025-01-02T00:00:00Z", "v": -0.05},
                {"datetime": "2025-01-03T00:00:00Z", "t": "2025-01-03T00:00:00Z", "v": 0.02},
            ],
            "val": [],
            "oos": [],
        },
        MODULE.DrawdownThrottleConfig(
            soft_drawdown=0.01,
            hard_drawdown=0.02,
            stop_drawdown=0.03,
            soft_scale=0.85,
            hard_scale=0.60,
            stop_scale=0.35,
        ),
    )
    assert throttled["train"]
    assert any(item["exposure_scale"] < 1.0 for item in schedule)
