from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_causal_dynamic_portfolio.py"
SPEC = importlib.util.spec_from_file_location("run_causal_dynamic_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load run_causal_dynamic_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _stream(
    start_ts_ms: float, values: list[float], *, step_ms: float = 86_400_000.0
) -> list[dict[str, float]]:
    return [{"t": start_ts_ms + (idx * step_ms), "v": value} for idx, value in enumerate(values)]


def _row(name: str, *, train: list[float], val: list[float], oos: list[float]) -> dict[str, Any]:
    return {
        "candidate_id": name,
        "name": name,
        "strategy_class": "StubStrategy",
        "strategy_timeframe": "1d",
        "family": "trend",
        "symbols": ["BTC/USDT"],
        "return_streams": {
            "train": _stream(1_735_689_600_000.0, train),
            "val": _stream(1_767_225_600_000.0, val),
            "oos": _stream(1_769_904_000_000.0, oos),
        },
        "metadata": {"cost_rate": 0.0005},
    }


def test_daily_compound_stream_compounds_intraday_points() -> None:
    stream = [
        {"t": "2026-02-01T00:00:00Z", "v": 0.10},
        {"t": "2026-02-01T12:00:00Z", "v": 0.10},
        {"t": "2026-02-02T00:00:00Z", "v": -0.05},
    ]
    result = MODULE._daily_compound_stream(stream)
    assert set(result) == {"2026-02-01", "2026-02-02"}
    assert abs(result["2026-02-01"] - 0.21) < 1e-9
    assert abs(result["2026-02-02"] - (-0.05)) < 1e-9


def test_run_causal_dynamic_allocator_uses_only_prior_history() -> None:
    rows = [
        _row("sleeve_a", train=[0.01, 0.01, 0.01, 0.01, 0.01], val=[-0.02, -0.01], oos=[0.0]),
        _row("sleeve_b", train=[-0.01, -0.01, -0.01, -0.01, -0.01], val=[0.05, 0.04], oos=[0.0]),
    ]
    params = MODULE.AllocatorParams(
        lookback_days=3,
        rebalance_days=1,
        min_trailing_sharpe=0.0,
        min_trailing_return=0.0,
        max_trailing_drawdown=0.50,
        max_weight=1.0,
    )
    result = MODULE.run_causal_dynamic_allocator(rows, params)
    allocation_by_date = {row["date"]: row for row in result["allocations"]}
    first_val = allocation_by_date["2026-01-01"]
    # sleeve_b has a great first validation day, but allocation must be based only on train history.
    assert first_val["weights"].get("sleeve_a", 0.0) > 0.99
    assert first_val["weights"].get("sleeve_b", 0.0) == 0.0


def test_search_dynamic_allocator_selects_by_validation_not_oos() -> None:
    rows = [
        _row(
            "sleeve_a",
            train=[0.02, 0.02, 0.02, 0.02, 0.02],
            val=[0.03, 0.03, 0.03],
            oos=[-0.04, -0.03, -0.02],
        ),
        _row(
            "sleeve_b",
            train=[-0.01, -0.01, -0.01, -0.01, -0.01],
            val=[-0.01, -0.01, -0.01],
            oos=[0.03, 0.03, 0.03],
        ),
    ]
    grid = [
        MODULE.AllocatorParams(
            lookback_days=3,
            rebalance_days=1,
            min_trailing_sharpe=0.0,
            min_trailing_return=0.0,
            max_trailing_drawdown=0.50,
            max_weight=1.0,
        ),
        MODULE.AllocatorParams(
            lookback_days=3,
            rebalance_days=1,
            min_trailing_sharpe=-10.0,
            min_trailing_return=-1.0,
            max_trailing_drawdown=1.0,
            max_weight=1.0,
        ),
    ]
    best = MODULE.search_dynamic_allocator(rows, param_grid=grid)
    assert best["params"]["min_trailing_sharpe"] == 0.0
    assert best["result"]["split_metrics"]["val"]["total_return"] > 0.0


def test_run_causal_dynamic_allocator_respects_max_weight_and_leaves_cash() -> None:
    rows = [
        _row("sleeve_a", train=[0.02, 0.02, 0.02, 0.02, 0.02], val=[0.03, 0.03], oos=[0.01]),
        _row(
            "sleeve_b", train=[-0.01, -0.01, -0.01, -0.01, -0.01], val=[-0.01, -0.01], oos=[-0.01]
        ),
    ]
    params = MODULE.AllocatorParams(
        lookback_days=3,
        rebalance_days=1,
        min_trailing_sharpe=0.0,
        min_trailing_return=0.0,
        max_trailing_drawdown=0.50,
        max_weight=0.4,
    )
    result = MODULE.run_causal_dynamic_allocator(rows, params)
    allocation_by_date = {row["date"]: row for row in result["allocations"]}
    first_val = allocation_by_date["2026-01-01"]
    assert abs(first_val["weights"].get("sleeve_a", 0.0) - 0.4) < 1e-9
    assert first_val["cash_weight"] >= 0.6


def test_mean_cash_fraction_uses_requested_split_only() -> None:
    allocations = [
        {"date": "2025-12-31", "cash_weight": 0.0},
        {"date": "2026-01-01", "cash_weight": 0.2},
        {"date": "2026-01-02", "cash_weight": 0.4},
        {"date": "2026-02-01", "cash_weight": 1.0},
    ]
    assert abs(MODULE._mean_cash_fraction(allocations, split="val") - 0.3) < 1e-9


def test_search_objective_uses_more_than_sharpe() -> None:
    base = {
        "sharpe": 1.0,
        "sortino": 1.0,
        "calmar": 1.0,
        "total_return": 0.02,
        "max_drawdown": 0.05,
        "volatility": 0.10,
    }
    better_sortino = dict(base, sortino=3.0, calmar=2.0)
    assert MODULE._search_objective(better_sortino, cash_fraction=0.0) > MODULE._search_objective(
        base, cash_fraction=0.0
    )


def test_active_weighting_respects_family_cap() -> None:
    history = {
        "a": MODULE.np.asarray([0.01, 0.01, 0.01]),
        "b": MODULE.np.asarray([0.02, 0.02, 0.02]),
        "c": MODULE.np.asarray([0.03, 0.03, 0.03]),
    }
    raw_scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    meta = {
        "a": {"family": "trend"},
        "b": {"family": "trend"},
        "c": {"family": "cross_sectional"},
    }
    weights = MODULE._active_weighting(
        history,
        raw_scores,
        meta=meta,
        max_weight=0.8,
        max_family_weight=0.5,
        correlation_penalty=0.0,
    )
    trend_total = weights.get("a", 0.0) + weights.get("b", 0.0)
    assert trend_total <= 0.5000001


def test_active_weighting_respects_max_weight() -> None:
    history = {
        "a": MODULE.np.asarray([0.01, 0.01, 0.01]),
        "b": MODULE.np.asarray([0.02, 0.02, 0.02]),
    }
    raw_scores = {"a": 100.0, "b": 1.0}
    meta = {"a": {"family": "trend"}, "b": {"family": "cross_sectional"}}
    weights = MODULE._active_weighting(
        history,
        raw_scores,
        meta=meta,
        max_weight=0.4,
        max_family_weight=1.0,
        correlation_penalty=0.0,
    )
    assert weights.get("a", 0.0) <= 0.4000001


def test_write_dynamic_comparison_refreshes_current_one_shot_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "comparison_scope": ["equal_weight_diagnostic"],
                "deltas": {},
            }
        ),
        encoding="utf-8",
    )
    current_bundle = tmp_path / "current_bundle.json"
    current_bundle.write_text(
        json.dumps({"candidates": [{"name": "fresh-incumbent-123"}]}),
        encoding="utf-8",
    )
    current_portfolio = tmp_path / "current_portfolio.json"
    current_portfolio.write_text(
        json.dumps(
            {
                "weights": [
                    {"candidate_id": "probe", "name": "fresh-incumbent-123", "weight": 1.0}
                ],
                "portfolio_metrics": {
                    "val": {"total_return": 0.012345, "sharpe": 1.12345},
                    "oos": {"total_return": 0.054321, "sharpe": 2.34567},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(MODULE, "COMPARISON_INPUT", comparison)
    monkeypatch.setattr(MODULE, "FOLLOWUP_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE", current_bundle)
    monkeypatch.setattr(MODULE, "PORTFOLIO_CURRENT_OPTIMIZATION", current_portfolio)

    result = MODULE.write_dynamic_comparison(
        dynamic_payload={
            "split_metrics": {"val": {}, "oos": {"total_return": 0.04, "sharpe": 1.5}},
            "final_allocation": [],
            "best_params": {},
        }
    )
    written = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert written["current_one_shot_optimized"]["path"] == str(current_portfolio.resolve())
    assert written["current_one_shot_optimized"]["bundle_path"] == str(current_bundle.resolve())
    assert written["current_one_shot_optimized"]["oos"]["total_return"] == 0.054321
    assert "current_one_shot_optimized" in written["comparison_scope"]
    assert abs(written["deltas"]["dynamic_vs_current_one_shot_oos_return"] + 0.014321) < 1e-12


def test_regime_multiplier_can_disable_rolling_breakout_when_gate_fails() -> None:
    meta_row = {
        "strategy_class": "RollingBreakoutStrategy",
        "metadata": {"activation_rule_conditions": ["basket_vol_ratio_moderate"]},
    }
    regime_row = {"basket_vol_ratio_moderate": False}
    assert (
        MODULE._regime_multiplier(meta_row, regime_row, previous_active=False, strength=1.0) == 0.0
    )


def test_regime_multiplier_rewards_trend_when_btc_and_breadth_are_positive() -> None:
    meta_row = {"strategy_class": "CompositeTrendStrategy", "metadata": {}}
    positive = MODULE._regime_multiplier(
        meta_row,
        {"btc_above_ma192": True, "breadth_ma96_ge_60": True},
        previous_active=False,
        strength=1.0,
    )
    negative = MODULE._regime_multiplier(
        meta_row,
        {"btc_above_ma192": False, "breadth_ma96_ge_60": False},
        previous_active=False,
        strength=1.0,
    )
    assert positive > 1.0
    assert negative < 1.0


def test_portfolio_return_streams_from_daily_maps_splits() -> None:
    streams = MODULE._portfolio_return_streams_from_daily(
        ["2025-12-31", "2026-01-15", "2026-02-03"],
        [0.01, -0.02, 0.03],
    )
    assert len(streams["train"]) == 1
    assert len(streams["val"]) == 1
    assert len(streams["oos"]) == 1
    assert streams["oos"][0]["t"] == "2026-02-03T00:00:00Z"
