from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "research" / "run_alpha_zoo_69_asset_walkforward_monitor.py"
    spec = importlib.util.spec_from_file_location("run_alpha_zoo_69_asset_walkforward_monitor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wf = _load_module()


def test_period_metrics_return_and_drawdown() -> None:
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    metrics = wf.period_metrics(
        returns,
        (wf._ts("2026-01-01T00:00:00Z"), wf._ts("2026-01-03T23:00:00Z")),
    )

    assert round(metrics["total_return"], 6) == round((1.10 * 0.95 * 1.02) - 1.0, 6)
    assert round(metrics["mdd"], 6) == 0.05
    assert metrics["bar_count"] == 3


def test_weights_for_target_gross_rejects_upscale_by_default() -> None:
    result = wf.weights_for_target_gross(
        profile_mix={"balanced": 1.0},
        profile_gross={"balanced": 2.0},
        target_gross=3.0,
        allow_upscale=False,
    )

    assert result is None


def test_weights_for_target_gross_scales_profile_mix() -> None:
    result = wf.weights_for_target_gross(
        profile_mix={"balanced": 0.75, "growth": 0.25},
        profile_gross={"balanced": 2.0, "growth": 4.0},
        target_gross=1.5,
        allow_upscale=False,
    )

    assert result is not None
    weights, source_gross = result
    assert source_gross == 2.5
    assert round(weights["balanced"], 8) == 0.45
    assert round(weights["growth"], 8) == 0.15


def test_build_monitor_manifest_keeps_all_statuses() -> None:
    source = {
        "universe": {"symbols": ["BTCUSDT", "ETHUSDT", "SPYUSDT"]},
        "train_eligibility": {
            "train_eligible_symbols": ["BTCUSDT", "ETHUSDT"],
            "train_ineligible_symbols": ["SPYUSDT"],
        },
    }
    diverse = {
        "asset_inclusion_manifest": [
            {"symbol": "BTCUSDT", "status": "tradable_now_train_eligible"},
            {"symbol": "SPYUSDT", "status": "future_watchlist_insufficient_train_history"},
        ],
        "selected_sleeve_rows": [
            {
                "symbol": "BTCUSDT",
                "weighted_notional_fraction": 0.5,
                "source_profile_id": "balanced",
                "timeframe": "1h",
                "side": "long_only",
                "family": "trend",
            }
        ],
    }

    manifest = wf.build_monitor_manifest(source_payload=source, diverse_payload=diverse)

    status_by_symbol = {row["symbol"]: row["status"] for row in manifest}
    assert status_by_symbol == {
        "BTCUSDT": "core_tradable_now",
        "ETHUSDT": "eligible_shadow_not_selected",
        "SPYUSDT": "future_watchlist_insufficient_train_history",
    }
    assert manifest[0]["core_gross_notional_fraction"] == 0.5


def test_evaluate_candidate_requires_all_validation_and_oos_positive() -> None:
    profile_returns = {
        "p": pd.Series(
            [0.01, 0.01, -0.01, 0.02],
            index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]),
        )
    }
    fold = wf.WalkForwardFold(
        "fold",
        (wf._ts("2026-01-01T00:00:00Z"), wf._ts("2026-01-01T23:00:00Z")),
        (wf._ts("2026-01-02T00:00:00Z"), wf._ts("2026-01-02T23:00:00Z")),
        (wf._ts("2026-02-01T00:00:00Z"), wf._ts("2026-02-02T23:00:00Z")),
    )
    candidate = wf.CandidateWeights(
        candidate_id="c",
        weights={"p": 1.0},
        gross_notional_fraction=1.0,
        profile_mix={"p": 1.0},
        selection_surface="unit",
        deployable_without_refit=False,
        notes=(),
    )

    evaluated = wf.evaluate_candidate(candidate, profile_returns, [fold], max_oos_mdd=0.20)

    assert evaluated["summary"]["all_validation_positive"] is True
    assert evaluated["summary"]["all_oos_positive"] is True
    assert evaluated["summary"]["return_shape_pass"] is True
