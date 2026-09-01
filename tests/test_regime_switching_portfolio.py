from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_regime_switching_portfolio.py"
SPEC = importlib.util.spec_from_file_location("run_regime_switching_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load run_regime_switching_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _stream(
    start_ts_ms: float, values: list[float], *, step_ms: float = 86_400_000.0
) -> list[dict[str, float]]:
    return [{"t": start_ts_ms + (idx * step_ms), "v": value} for idx, value in enumerate(values)]


def _candidate(
    name: str,
    *,
    components: list[dict[str, Any]],
    train: list[float],
    val: list[float],
    oos: list[float],
) -> dict[str, Any]:
    return {
        "candidate_id": name,
        "name": name,
        "selection_basis": "test_fixture",
        "symbols": sorted(
            {str(symbol) for row in components for symbol in list(row.get("symbols") or [])}
        ),
        "components": components,
        "return_streams": {
            "train": _stream(1_735_689_600_000.0, train),
            "val": _stream(1_767_225_600_000.0, val),
            "oos": _stream(1_769_904_000_000.0, oos),
        },
    }


def test_portfolio_regime_score_prefers_pair_portfolio_when_trend_is_weak() -> None:
    pair_candidate = {
        "components": [
            {
                "candidate_id": "pair",
                "name": "pair",
                "strategy_class": "PairSpreadZScoreStrategy",
                "family": "market_neutral",
                "symbols": ["BTC/USDT", "TRX/USDT"],
                "weight": 1.0,
            }
        ]
    }
    trend_candidate = {
        "components": [
            {
                "candidate_id": "trend",
                "name": "trend",
                "strategy_class": "CompositeTrendStrategy",
                "family": "trend",
                "symbols": ["BTC/USDT"],
                "weight": 1.0,
            }
        ]
    }
    regime_row = {
        "btc_above_ma192": False,
        "breadth_ma96_ge_60": False,
        "basket_vol_ratio_moderate": True,
    }
    pair_score, _ = MODULE._portfolio_regime_score(
        pair_candidate,
        regime_row,
        previous_active=False,
        strength=1.0,
    )
    trend_score, _ = MODULE._portfolio_regime_score(
        trend_candidate,
        regime_row,
        previous_active=False,
        strength=1.0,
    )
    assert pair_score > trend_score


def test_kelly_target_exposure_can_reduce_cash() -> None:
    history = {
        "a": MODULE.np.asarray([0.01, 0.012, 0.009, 0.011, 0.013], dtype=float),
        "b": MODULE.np.asarray([0.008, 0.007, 0.009, 0.0085, 0.010], dtype=float),
    }
    target_weights = {"a": 0.55, "b": 0.20}
    exposure = MODULE._kelly_target_exposure(
        history,
        target_weights,
        cash_buffer=0.0,
        kelly_shrinkage=1.0,
    )
    assert exposure > sum(target_weights.values())
    assert exposure <= 1.0


def test_run_regime_switch_allocator_uses_only_prior_history() -> None:
    rows = [
        _candidate(
            "incumbent",
            components=[
                {
                    "candidate_id": "trend_a",
                    "name": "trend_a",
                    "strategy_class": "CompositeTrendStrategy",
                    "family": "trend",
                    "symbols": ["BTC/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.01, 0.01, 0.01, 0.01, 0.01],
            val=[-0.02, -0.02],
            oos=[0.0],
        ),
        _candidate(
            "pair55",
            components=[
                {
                    "candidate_id": "pair_b",
                    "name": "pair_b",
                    "strategy_class": "PairSpreadZScoreStrategy",
                    "family": "market_neutral",
                    "symbols": ["BNB/USDT", "TRX/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[-0.01, -0.01, -0.01, -0.01, -0.01],
            val=[0.05, 0.05],
            oos=[0.0],
        ),
    ]
    params = MODULE.SwitchParams(
        lookback_days=3,
        rebalance_days=1,
        min_trailing_sharpe=0.0,
        min_trailing_return=0.0,
        max_trailing_drawdown=0.50,
        max_portfolio_weight=1.0,
        regime_strength=0.0,
        hysteresis_bonus=0.0,
        turnover_cost_bps=0.0,
    )
    result = MODULE.run_regime_switch_allocator(rows, params, regime_features={})
    allocation_by_date = {row["date"]: row for row in result["allocations"]}
    first_val = allocation_by_date["2026-01-01"]
    assert first_val["weights"].get("incumbent", 0.0) > 0.99
    assert first_val["weights"].get("pair55", 0.0) == 0.0


def test_search_regime_switch_allocator_uses_weekly_rebalance_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _candidate(
            "incumbent",
            components=[
                {
                    "candidate_id": "trend_a",
                    "name": "trend_a",
                    "strategy_class": "CompositeTrendStrategy",
                    "family": "trend",
                    "symbols": ["BTC/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.01] * 10,
            val=[0.01, 0.01],
            oos=[0.01],
        )
    ]
    seen: list[int] = []

    def fake_run(rows, params, *, regime_features=None):
        seen.append(params.rebalance_days)
        return {
            "dates": ["2025-01-01"],
            "daily_returns": [0.0],
            "allocations": [
                {"date": "2025-01-01", "weights": {}, "cash_weight": 1.0, "sleeve_turnover": 0.0}
            ],
            "split_metrics": {
                "train": {
                    "total_return": 0.0,
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "calmar": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                },
                "val": {
                    "total_return": 0.0,
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "calmar": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                },
                "oos": {
                    "total_return": 0.0,
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "calmar": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                },
            },
            "all_metrics": {},
        }

    monkeypatch.setattr(MODULE._helper, "_load_regime_features", lambda _rows: {})
    monkeypatch.setattr(MODULE, "run_regime_switch_allocator", fake_run)
    MODULE.search_regime_switch_allocator(rows)
    assert seen
    assert set(seen) == {7}


def test_run_regime_switch_allocator_applies_turnover_cost_using_sleeves() -> None:
    rows = [
        _candidate(
            "incumbent",
            components=[
                {
                    "candidate_id": "trend_a",
                    "name": "trend_a",
                    "strategy_class": "CompositeTrendStrategy",
                    "family": "trend",
                    "symbols": ["BTC/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.02, 0.02, 0.02, 0.02, 0.02],
            val=[0.02, -0.01],
            oos=[0.0],
        ),
        _candidate(
            "pair55",
            components=[
                {
                    "candidate_id": "pair_b",
                    "name": "pair_b",
                    "strategy_class": "PairSpreadZScoreStrategy",
                    "family": "market_neutral",
                    "symbols": ["BNB/USDT", "TRX/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[-0.01, -0.01, -0.01, -0.01, -0.01],
            val=[0.06, 0.03],
            oos=[0.0],
        ),
    ]
    params = MODULE.SwitchParams(
        lookback_days=1,
        rebalance_days=1,
        min_trailing_sharpe=-10.0,
        min_trailing_return=-1.0,
        max_trailing_drawdown=1.0,
        max_portfolio_weight=1.0,
        regime_strength=0.0,
        hysteresis_bonus=0.0,
        turnover_cost_bps=100.0,
    )
    result = MODULE.run_regime_switch_allocator(
        rows,
        params,
        regime_features={"2026-01-01": {"btc_above_ma192": True, "breadth_ma96_ge_60": True}},
    )
    allocations = {row["date"]: row for row in result["allocations"]}
    second_val = allocations["2026-01-02"]
    assert second_val["weights"].get("pair55", 0.0) > 0.0
    assert second_val["sleeve_turnover"] > 0.0
    # Day return should be the selected portfolio return minus turnover cost.
    daily_returns = list(result["daily_returns"])
    raw_blend = (
        second_val["weights"].get("pair55", 0.0) * 0.03
        + second_val["weights"].get("incumbent", 0.0) * -0.01
    )
    assert daily_returns[6] < raw_blend


def test_run_regime_switch_allocator_falls_back_to_incumbent_when_regime_missing() -> None:
    rows = [
        _candidate(
            "incumbent",
            components=[
                {
                    "candidate_id": "trend_a",
                    "name": "trend_a",
                    "strategy_class": "CompositeTrendStrategy",
                    "family": "trend",
                    "symbols": ["BTC/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.01, 0.01, 0.01, 0.01, 0.01],
            val=[0.01, 0.01],
            oos=[0.0],
        ),
        _candidate(
            "pair55",
            components=[
                {
                    "candidate_id": "pair_b",
                    "name": "pair_b",
                    "strategy_class": "PairSpreadZScoreStrategy",
                    "family": "market_neutral",
                    "symbols": ["BNB/USDT", "TRX/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.03, 0.03, 0.03, 0.03, 0.03],
            val=[0.05, 0.05],
            oos=[0.0],
        ),
    ]
    params = MODULE.SwitchParams(
        lookback_days=3,
        rebalance_days=1,
        min_trailing_sharpe=-10.0,
        min_trailing_return=-1.0,
        max_trailing_drawdown=1.0,
        max_portfolio_weight=1.0,
        regime_strength=1.0,
        hysteresis_bonus=0.0,
        turnover_cost_bps=0.0,
        incumbent_floor_weight=0.5,
    )
    result = MODULE.run_regime_switch_allocator(rows, params, regime_features={})
    first_val = {row["date"]: row for row in result["allocations"]}["2026-01-01"]
    assert first_val["weights"].get("incumbent", 0.0) > 0.99
    assert first_val["diagnostics"]["pair55"]["regime_available"] is False


def test_run_regime_switch_allocator_caps_turnover_and_respects_incumbent_floor() -> None:
    rows = [
        _candidate(
            "incumbent",
            components=[
                {
                    "candidate_id": "trend_a",
                    "name": "trend_a",
                    "strategy_class": "CompositeTrendStrategy",
                    "family": "trend",
                    "symbols": ["BTC/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.01, 0.01, 0.01, 0.01, 0.01],
            val=[0.05, -0.01],
            oos=[0.0],
        ),
        _candidate(
            "pair55",
            components=[
                {
                    "candidate_id": "pair_b",
                    "name": "pair_b",
                    "strategy_class": "PairSpreadZScoreStrategy",
                    "family": "market_neutral",
                    "symbols": ["BNB/USDT", "TRX/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[-0.01, -0.01, -0.01, -0.01, -0.01],
            val=[0.10, 0.10],
            oos=[0.0],
        ),
    ]
    params = MODULE.SwitchParams(
        lookback_days=1,
        rebalance_days=1,
        min_trailing_sharpe=-10.0,
        min_trailing_return=-1.0,
        max_trailing_drawdown=1.0,
        max_portfolio_weight=1.0,
        regime_strength=0.0,
        hysteresis_bonus=0.0,
        turnover_cost_bps=0.0,
        max_sleeve_turnover=0.25,
        incumbent_floor_weight=0.30,
    )
    result = MODULE.run_regime_switch_allocator(
        rows, params, regime_features={"2026-01-01": {"btc_above_ma192": True}}
    )
    second_val = {row["date"]: row for row in result["allocations"]}["2026-01-02"]
    assert second_val["sleeve_turnover"] <= 0.2500001
    assert second_val["weights"].get("incumbent", 0.0) >= 0.30


def test_non_incumbent_rebuild_candidates_are_blocked_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_load_runtime_risk_state",
        lambda path=None: {
            "continuity_report_path": "/tmp/continuity.json",
            "continuity_status": "passed",
            "continuity_failed": False,
            "symbols": set(),
            "timeframes": set(),
            "error": None,
        },
    )
    rows = [
        _candidate(
            "incumbent",
            components=[
                {
                    "candidate_id": "trend_a",
                    "name": "trend_a",
                    "strategy_class": "CompositeTrendStrategy",
                    "family": "trend",
                    "symbols": ["BTC/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.01, 0.01, 0.01, 0.01, 0.01],
            val=[0.01, 0.01],
            oos=[0.0],
        ),
        _candidate(
            "overlay",
            components=[
                {
                    "candidate_id": "pair_b",
                    "name": "pair_b",
                    "strategy_class": "PairSpreadZScoreStrategy",
                    "family": "market_neutral",
                    "symbols": ["BNB/USDT", "TRX/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.05, 0.05, 0.05, 0.05, 0.05],
            val=[0.05, 0.05],
            oos=[0.0],
        ),
    ]
    rows[1]["stream_mode"] = "rebuild_required"
    params = MODULE.SwitchParams(
        lookback_days=3,
        rebalance_days=1,
        min_trailing_sharpe=-10.0,
        min_trailing_return=-1.0,
        max_trailing_drawdown=1.0,
        max_portfolio_weight=1.0,
    )
    result = MODULE.run_regime_switch_allocator(
        rows, params, regime_features={"2026-01-01": {"btc_above_ma192": True}}
    )
    first_val = {row["date"]: row for row in result["allocations"]}["2026-01-01"]
    assert first_val["weights"].get("incumbent", 0.0) > 0.99
    assert first_val["diagnostics"]["overlay"]["rebuild_blocked"] is True


def test_non_incumbent_candidates_blocked_when_continuity_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_load_runtime_risk_state",
        lambda path=None: {
            "continuity_report_path": "/tmp/continuity.json",
            "continuity_status": "failed",
            "continuity_failed": True,
            "symbols": {"BNB/USDT", "TRX/USDT"},
            "timeframes": {"1h"},
            "error": "missing coverage",
        },
    )
    rows = [
        _candidate(
            "incumbent",
            components=[
                {
                    "candidate_id": "trend_a",
                    "name": "trend_a",
                    "strategy_class": "CompositeTrendStrategy",
                    "family": "trend",
                    "symbols": ["BTC/USDT"],
                    "weight": 1.0,
                }
            ],
            train=[0.01, 0.01, 0.01, 0.01, 0.01],
            val=[0.01, 0.01],
            oos=[0.0],
        ),
        _candidate(
            "pair55",
            components=[
                {
                    "candidate_id": "pair_b",
                    "name": "pair_b",
                    "strategy_class": "PairSpreadZScoreStrategy",
                    "family": "market_neutral",
                    "symbols": ["BNB/USDT", "TRX/USDT"],
                    "timeframe": "1h",
                    "weight": 1.0,
                }
            ],
            train=[0.05, 0.05, 0.05, 0.05, 0.05],
            val=[0.05, 0.05],
            oos=[0.0],
        ),
    ]
    params = MODULE.SwitchParams(
        lookback_days=3,
        rebalance_days=1,
        min_trailing_sharpe=-10.0,
        min_trailing_return=-1.0,
        max_trailing_drawdown=1.0,
        max_portfolio_weight=1.0,
        require_continuity_pass=True,
        allow_rebuilt_candidates=True,
    )
    result = MODULE.run_regime_switch_allocator(
        rows, params, regime_features={"2026-01-01": {"btc_above_ma192": True}}
    )
    first_val = {row["date"]: row for row in result["allocations"]}["2026-01-01"]
    assert first_val["weights"].get("incumbent", 0.0) > 0.99
    assert first_val["diagnostics"]["pair55"]["continuity_blocked"] is True


def test_extract_portfolio_return_streams_supports_daily_returns_payload() -> None:
    payload = {
        "dates": ["2026-01-01", "2026-02-01"],
        "daily_returns": [0.01, -0.02],
    }
    streams = MODULE._extract_portfolio_return_streams(payload)
    assert len(streams["val"]) == 1
    assert len(streams["oos"]) == 1


def test_write_regime_switch_comparison_adds_scope(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "comparison_scope": ["current_one_shot_optimized"],
                "current_one_shot_optimized": {"oos": {"total_return": 0.05, "sharpe": 2.0}},
                "deltas": {},
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "split_metrics": {"val": {}, "oos": {"total_return": 0.06, "sharpe": 3.0}},
        "final_allocation": [],
        "best_params": {},
    }
    result = MODULE.write_regime_switch_comparison(
        switch_payload=payload,
        comparison_input=comparison,
    )
    written = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert "regime_switching_portfolio" in written["comparison_scope"]
    assert abs(written["deltas"]["regime_switch_vs_current_one_shot_oos_return"] - 0.01) < 1e-12


def test_write_regime_switch_preflight_reports_blockers_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incumbent_path = tmp_path / "incumbent.json"
    overlay_path = tmp_path / "overlay.json"
    incumbent_path.write_text(
        json.dumps(
            {
                "selection_basis": "rolled_over_from_promoted_challenger",
                "weights": [
                    {
                        "candidate_id": "trend_a",
                        "name": "trend_a",
                        "family": "trend",
                        "strategy_class": "CompositeTrendStrategy",
                        "symbols": ["BTC/USDT"],
                        "weight": 1.0,
                    }
                ],
                "portfolio_return_streams": {"train": [], "val": [], "oos": []},
            }
        ),
        encoding="utf-8",
    )
    overlay_path.write_text(
        json.dumps(
            {
                "portfolio_name": "overlay",
                "artifact_kind": "causal_overlay_portfolio",
                "selection_basis": "validation_only_overlay_search_on_current_one_shot_backbone",
                "weights": [
                    {
                        "candidate_id": "pair_b",
                        "name": "pair_b",
                        "family": "market_neutral",
                        "strategy_class": "PairSpreadZScoreStrategy",
                        "symbols": ["BNB/USDT", "TRX/USDT"],
                        "timeframe": "1h",
                        "weight": 1.0,
                    }
                ],
                "best_params": {"lookback_days": 10},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "resolve_followup_artifact_path", lambda path: Path(path))
    monkeypatch.setattr(
        MODULE,
        "_load_runtime_risk_state",
        lambda path=None: {
            "continuity_report_path": "/tmp/continuity.json",
            "continuity_status": "failed",
            "continuity_failed": True,
            "symbols": {"BNB/USDT", "TRX/USDT"},
            "timeframes": {"1h"},
            "error": "missing coverage",
        },
    )
    result = MODULE.write_regime_switch_preflight(
        input_paths=[str(incumbent_path), str(overlay_path)],
        output_dir=tmp_path,
    )
    written = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert written["summary"]["blocked_candidate_count"] == 1
    status_by_id = {row["candidate_id"]: row for row in written["candidate_status"]}
    assert status_by_id["current_one_shot_incumbent"]["blocking_reasons"] == []
    assert set(status_by_id["overlay"]["blocking_reasons"]) == {
        "rebuild_blocked",
        "continuity_blocked",
    }


def test_extract_portfolio_return_streams_rebuilds_dynamic_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(MODULE, "_REBUILT_STREAM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        MODULE, "resolve_followup_artifact_path", lambda path: Path("/tmp/dynamic.json")
    )
    monkeypatch.setattr(
        MODULE._helper,
        "_load_candidates",
        lambda _path: [{"candidate_id": "stub"}],
    )
    monkeypatch.setattr(
        MODULE._helper,
        "run_causal_dynamic_allocator",
        lambda rows, params, **kwargs: {"dates": ["2026-02-01"], "daily_returns": [0.01]},
    )
    payload = {
        "artifact_kind": "causal_dynamic_portfolio",
        "input_path": "/tmp/dynamic.json",
        "best_params": {
            "lookback_days": 5,
            "rebalance_days": 1,
            "min_trailing_sharpe": 0.0,
            "min_trailing_return": 0.0,
            "max_trailing_drawdown": 0.1,
            "max_weight": 0.4,
            "max_family_weight": 1.0,
            "correlation_penalty": 0.0,
            "cash_when_no_active": True,
            "use_regime_features": False,
            "regime_strength": 1.0,
        },
    }
    source_path = tmp_path / "dynamic_payload.json"
    source_path.write_text("{}", encoding="utf-8")
    streams = MODULE._extract_portfolio_return_streams(payload, source_path=source_path)
    assert streams["oos"][0]["v"] == 0.01
    assert MODULE._stream_mode(source_path, payload) == "cached_rebuild_streams"


def test_extract_portfolio_return_streams_rebuilds_overlay_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(MODULE, "_REBUILT_STREAM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(MODULE, "resolve_followup_artifact_path", lambda path: Path(str(path)))
    monkeypatch.setattr(
        MODULE._helper, "_load_candidates", lambda _path: [{"candidate_id": "stub"}]
    )
    monkeypatch.setattr(MODULE._overlay, "_load_backbone_weights", lambda _path: {"stub": 1.0})
    monkeypatch.setattr(
        MODULE._overlay,
        "run_causal_overlay_allocator",
        lambda rows, backbone_weights, params, **kwargs: {
            "dates": ["2026-01-15"],
            "daily_returns": [-0.02],
        },
    )
    payload = {
        "artifact_kind": "causal_overlay_portfolio",
        "input_path": "/tmp/input.json",
        "backbone_path": "/tmp/backbone.json",
        "best_params": {
            "lookback_days": 5,
            "rebalance_days": 1,
            "min_trailing_sharpe": 0.0,
            "min_trailing_return": 0.0,
            "max_trailing_drawdown": 0.1,
            "overlay_strength": 1.0,
            "correlation_penalty": 0.0,
            "regime_strength": 1.0,
            "cash_buffer": 0.0,
        },
    }
    source_path = tmp_path / "overlay_payload.json"
    source_path.write_text("{}", encoding="utf-8")
    streams = MODULE._extract_portfolio_return_streams(payload, source_path=source_path)
    assert streams["val"][0]["v"] == -0.02
    assert MODULE._stream_mode(source_path, payload) == "cached_rebuild_streams"


def test_extract_portfolio_components_enriches_final_allocation_from_input_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "resolve_followup_artifact_path", lambda path: Path(str(path)))
    monkeypatch.setattr(
        MODULE,
        "_cached_candidate_rows",
        lambda _path: [
            {
                "candidate_id": "pair_b",
                "name": "pair_b",
                "family": "market_neutral",
                "strategy_class": "PairSpreadZScoreStrategy",
                "symbols": ["BNB/USDT", "TRX/USDT"],
                "strategy_timeframe": "1h",
            }
        ],
    )
    payload = {
        "input_path": "/tmp/incumbent_bundle.json",
        "final_allocation": [
            {
                "candidate_id": "pair_b",
                "name": "pair_b",
                "strategy_class": "PairSpreadZScoreStrategy",
                "weight": 0.75,
            }
        ],
    }
    components = MODULE._extract_portfolio_components(payload)
    assert components == [
        {
            "candidate_id": "pair_b",
            "name": "pair_b",
            "strategy_class": "PairSpreadZScoreStrategy",
            "weight": 0.75,
            "family": "market_neutral",
            "symbols": ["BNB/USDT", "TRX/USDT"],
            "timeframe": "1h",
        }
    ]


def test_stream_mode_marks_allocation_reconstructible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(MODULE, "_REBUILT_STREAM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(MODULE, "resolve_followup_artifact_path", lambda path: Path(str(path)))
    monkeypatch.setattr(
        MODULE,
        "_cached_candidate_rows",
        lambda _path: [
            {
                "candidate_id": "pair_b",
                "name": "pair_b",
                "return_streams": {"train": [{"t": 1.0, "v": 0.01}]},
            }
        ],
    )
    source_path = tmp_path / "overlay_payload.json"
    source_path.write_text("{}", encoding="utf-8")
    payload = {
        "input_path": "/tmp/incumbent_bundle.json",
        "allocations": [
            {
                "date": "2026-02-01",
                "weights": {"pair_b": 1.0},
            }
        ],
    }
    assert MODULE._stream_mode(source_path, payload) == "allocation_reconstructible"


def test_extract_portfolio_return_streams_reconstructs_from_saved_allocations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(MODULE, "_REBUILT_STREAM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(MODULE, "resolve_followup_artifact_path", lambda path: Path(str(path)))
    monkeypatch.setattr(
        MODULE,
        "_cached_candidate_rows",
        lambda _path: [
            {
                "candidate_id": "pair_b",
                "name": "pair_b",
                "return_streams": {
                    "train": [{"t": 1735689600000.0, "v": 0.02}],
                    "val": [{"t": 1767225600000.0, "v": -0.01}],
                    "oos": [{"t": 1769904000000.0, "v": 0.03}],
                },
            }
        ],
    )
    source_path = tmp_path / "dynamic_payload.json"
    source_path.write_text("{}", encoding="utf-8")
    payload = {
        "input_path": "/tmp/incumbent_bundle.json",
        "allocations": [
            {"date": "2025-01-01", "weights": {"pair_b": 1.0}},
            {"date": "2026-01-01", "weights": {"pair_b": 1.0}},
            {"date": "2026-02-01", "weights": {"pair_b": 1.0}},
        ],
    }
    streams = MODULE._extract_portfolio_return_streams(payload, source_path=source_path)
    assert [point["v"] for point in streams["train"]] == pytest.approx([0.02])
    assert [point["v"] for point in streams["val"]] == pytest.approx([-0.01])
    assert [point["v"] for point in streams["oos"]] == pytest.approx([0.03])
    assert MODULE._stream_mode(source_path, payload) == "cached_rebuild_streams"
