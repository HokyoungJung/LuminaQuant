from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_hybrid_online_portfolio.py"
SPEC = importlib.util.spec_from_file_location("run_hybrid_online_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load run_hybrid_online_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _stream(
    start_ts_ms: float, values: list[float], *, step_ms: float = 86_400_000.0
) -> list[dict[str, float]]:
    return [{"t": start_ts_ms + (idx * step_ms), "v": value} for idx, value in enumerate(values)]


def _row(
    name: str,
    *,
    train: list[float],
    val: list[float],
    oos: list[float],
    oos_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": name,
        "name": name,
        "strategy_class": "StubSleeve",
        "strategy_timeframe": "1d",
        "family": "portfolio",
        "symbols": ["BTC/USDT"],
        "return_streams": {
            "train": _stream(1_735_689_600_000.0, train),
            "val": _stream(1_767_225_600_000.0, val),
            "oos": _stream(1_769_904_000_000.0, oos),
        },
        "train": {
            "total_return": sum(train),
            "sharpe": 1.0 if sum(train) > 0 else -1.0,
            "trade_count": 20.0,
        },
        "val": {
            "total_return": sum(val),
            "sharpe": 1.0 if sum(val) > 0 else -1.0,
            "trade_count": 10.0,
        },
        "oos": {
            "total_return": sum(oos),
            "sharpe": 1.0 if sum(oos) > 0 else -1.0,
            "trade_count": 20.0,
            "active_fold_ratio": 1.0,
            "pbo": 0.0,
            **(oos_extra or {}),
        },
        "metadata": {"source_payload_path": "synthetic"},
    }


def _portfolio_payload(
    name: str, *, train: list[float], val: list[float], oos: list[float]
) -> dict[str, Any]:
    row = _row(name, train=train, val=val, oos=oos)
    return {
        "portfolio_return_streams": row["return_streams"],
        "portfolio_metrics": {
            "train": row["train"],
            "val": row["val"],
            "oos": row["oos"],
        },
    }


def test_payload_daily_streams_resplits_old_oos_days_into_new_val_window() -> None:
    payload = {
        "portfolio_daily_return_streams": {
            "oos": [
                {"t": "2026-02-15T00:00:00Z", "v": 0.01},
                {"t": "2026-03-05T00:00:00Z", "v": 0.02},
            ]
        }
    }
    streams = MODULE._payload_daily_streams(
        payload,
        split_config=MODULE.HybridSplitConfig(),
    )
    assert [point["t"] for point in streams["val"]] == ["2026-02-15T00:00:00Z"]
    assert [point["t"] for point in streams["oos"]] == ["2026-03-05T00:00:00Z"]


def test_resolve_warmup_days_uses_ratio_when_days_not_explicit() -> None:
    warmup_days = MODULE.resolve_warmup_days(
        config=MODULE.HybridOnlineConfig(warmup_days=None, warmup_ratio=0.60),
        split_config=MODULE.HybridSplitConfig(),
    )
    assert warmup_days == 255
    assert MODULE.HybridSplitConfig().online_start_date(warmup_days) == "2025-09-13"


def test_health_prior_shrinks_negative_sleeves() -> None:
    cfg = MODULE.HybridOnlineConfig()
    assert MODULE._health_prior({"total_return": 0.01, "sharpe": 0.5}, cfg) == 1.0
    assert (
        MODULE._health_prior({"total_return": 0.01, "sharpe": -0.5}, cfg) == cfg.mixed_health_floor
    )
    assert (
        MODULE._health_prior({"total_return": -0.01, "sharpe": -0.5}, cfg)
        == cfg.negative_health_floor
    )


def test_fragility_penalty_demotes_sparse_pair() -> None:
    cfg = MODULE.HybridOnlineConfig()
    pair = _row(
        "pair_tactical_mode",
        train=[0.01, 0.01],
        val=[0.02],
        oos=[0.03],
        oos_extra={"trade_count": 6.0, "active_fold_ratio": 0.25, "pbo": 0.625},
    )
    penalty = MODULE._fragility_penalty(pair, config=cfg)
    assert penalty > 1.0


def test_hybrid_online_allocator_uses_cash_fallback_when_all_non_cash_scores_are_bad() -> None:
    rows = [
        _row("risk_off_cash", train=[0.0, 0.0], val=[0.0], oos=[0.0]),
        _row("soft_three_way_regime", train=[-0.01] * 25, val=[-0.01] * 5, oos=[-0.01] * 5),
        _row("balanced_overlay_80_20", train=[-0.01] * 25, val=[-0.01] * 5, oos=[-0.01] * 5),
        _row(
            "pair_tactical_mode",
            train=[-0.01] * 25,
            val=[-0.01] * 5,
            oos=[-0.01] * 5,
            oos_extra={"trade_count": 4.0, "pbo": 0.625},
        ),
    ]
    result = MODULE.run_hybrid_online_allocator(
        rows,
        config=MODULE.HybridOnlineConfig(warmup_days=5, lookback_days=5),
        refreshed_health_metrics={
            "soft_three_way_regime": {"total_return": -0.02, "sharpe": -2.0},
            "balanced_overlay_80_20": {"total_return": -0.01, "sharpe": -1.0},
            "pair_tactical_mode": {"total_return": -0.01, "sharpe": -1.0},
        },
    )
    final = result["final_allocation"]
    assert final["cash_weight"] == 1.0
    assert final["weights"] == {}


def test_hybrid_online_allocator_enforces_pair_cap() -> None:
    rows = [
        _row("risk_off_cash", train=[0.0, 0.0], val=[0.0], oos=[0.0]),
        _row("soft_three_way_regime", train=[-0.005] * 30, val=[-0.005] * 5, oos=[-0.005] * 5),
        _row("balanced_overlay_80_20", train=[-0.004] * 30, val=[-0.004] * 5, oos=[-0.004] * 5),
        _row(
            "pair_tactical_mode",
            train=[0.02] * 30,
            val=[0.02] * 5,
            oos=[0.02] * 5,
            oos_extra={"trade_count": 20.0, "pbo": 0.2},
        ),
    ]
    result = MODULE.run_hybrid_online_allocator(
        rows,
        config=MODULE.HybridOnlineConfig(
            warmup_days=5, lookback_days=5, pair_weight_cap=0.3, min_positive_score=0.0
        ),
        refreshed_health_metrics={
            "soft_three_way_regime": {"total_return": -0.02, "sharpe": -2.0},
            "balanced_overlay_80_20": {"total_return": -0.01, "sharpe": -1.0},
            "pair_tactical_mode": {"total_return": 0.02, "sharpe": 1.5},
        },
    )
    final = result["final_allocation"]
    assert final["weights"]["pair_tactical_mode"] <= 0.3000001
    assert final["cash_weight"] >= 0.6999999


def test_hybrid_online_allocator_warmup_defaults_to_soft() -> None:
    rows = [
        _row("risk_off_cash", train=[0.0, 0.0], val=[0.0], oos=[0.0]),
        _row("soft_three_way_regime", train=[0.01] * 10, val=[0.01] * 2, oos=[0.01]),
        _row("balanced_overlay_80_20", train=[0.005] * 10, val=[0.005] * 2, oos=[0.005]),
        _row(
            "pair_tactical_mode",
            train=[0.02] * 10,
            val=[0.02] * 2,
            oos=[0.02],
            oos_extra={"trade_count": 8.0, "pbo": 0.625},
        ),
    ]
    result = MODULE.run_hybrid_online_allocator(
        rows,
        config=MODULE.HybridOnlineConfig(
            warmup_days=5, lookback_days=5, use_current_health_priors=False
        ),
        refreshed_health_metrics=None,
    )
    first = result["allocations"][0]
    assert first["default_sleeve"] == "soft_three_way_regime"
    assert first["weights"]["soft_three_way_regime"] == 1.0


def test_hybrid_online_allocator_uses_warmup_days_not_just_lookback() -> None:
    rows = [
        _row("risk_off_cash", train=[0.0] * 20, val=[0.0] * 2, oos=[0.0]),
        _row("soft_three_way_regime", train=[0.01] * 20, val=[0.01] * 2, oos=[0.01]),
        _row("balanced_overlay_80_20", train=[0.02] * 20, val=[0.02] * 2, oos=[0.02]),
        _row(
            "pair_tactical_mode",
            train=[0.0] * 20,
            val=[0.0] * 2,
            oos=[0.0],
            oos_extra={"trade_count": 20.0, "pbo": 0.0},
        ),
    ]
    result = MODULE.run_hybrid_online_allocator(
        rows,
        config=MODULE.HybridOnlineConfig(
            warmup_days=10, lookback_days=5, min_positive_score=0.0, use_current_health_priors=False
        ),
        refreshed_health_metrics=None,
    )
    warmup_defaults = [alloc["default_sleeve"] for alloc in result["allocations"][:10]]
    assert set(warmup_defaults) == {"soft_three_way_regime"}
    assert result["allocations"][10]["default_sleeve"] in {
        "soft_three_way_regime",
        "balanced_overlay_80_20",
    }


def test_hybrid_online_allocator_respects_sticky_default_margin() -> None:
    rows = [
        _row("risk_off_cash", train=[0.0, 0.0], val=[0.0], oos=[0.0]),
        _row("soft_three_way_regime", train=[0.01] * 30, val=[0.0105] * 5, oos=[0.01] * 5),
        _row("balanced_overlay_80_20", train=[0.0102] * 30, val=[0.0106] * 5, oos=[0.01] * 5),
        _row(
            "pair_tactical_mode",
            train=[0.0] * 30,
            val=[0.0] * 5,
            oos=[0.0] * 5,
            oos_extra={"trade_count": 20.0, "pbo": 0.0},
        ),
    ]
    result = MODULE.run_hybrid_online_allocator(
        rows,
        config=MODULE.HybridOnlineConfig(
            warmup_days=5,
            lookback_days=5,
            min_positive_score=0.0,
            sticky_default_bonus=0.2,
            switch_margin=0.2,
            use_current_health_priors=False,
        ),
        refreshed_health_metrics=None,
    )
    # After warmup, the default should remain sticky to soft_three_way_regime despite a small balanced edge.
    later_defaults = [alloc["default_sleeve"] for alloc in result["allocations"][5:10]]
    assert "soft_three_way_regime" in later_defaults


def test_fixed_default_variant_keeps_previous_default_when_positive() -> None:
    rows = [
        _row("risk_off_cash", train=[0.0, 0.0], val=[0.0], oos=[0.0]),
        _row("soft_three_way_regime", train=[0.01] * 12, val=[0.01] * 2, oos=[0.01] * 2),
        _row("balanced_overlay_80_20", train=[0.02] * 12, val=[0.02] * 2, oos=[0.02] * 2),
        _row(
            "pair_tactical_mode",
            train=[0.0] * 12,
            val=[0.0] * 2,
            oos=[0.0] * 2,
            oos_extra={"trade_count": 20.0, "pbo": 0.0},
        ),
    ]
    result = MODULE.run_hybrid_online_allocator(
        rows,
        config=MODULE.HybridOnlineConfig(
            variant="fixed_default",
            warmup_days=5,
            lookback_days=5,
            min_positive_score=0.0,
            use_current_health_priors=False,
        ),
        refreshed_health_metrics=None,
    )
    later_defaults = [alloc["default_sleeve"] for alloc in result["allocations"][5:10]]
    assert set(later_defaults) == {"soft_three_way_regime"}


def test_disagreement_switching_variant_scales_down_active_weights_when_scores_are_close() -> None:
    rows = [
        _row("risk_off_cash", train=[0.0] * 20, val=[0.0] * 2, oos=[0.0] * 2),
        _row("soft_three_way_regime", train=[0.01] * 20, val=[0.01] * 2, oos=[0.01] * 2),
        _row("balanced_overlay_80_20", train=[0.0101] * 20, val=[0.0101] * 2, oos=[0.0101] * 2),
        _row(
            "pair_tactical_mode",
            train=[0.0] * 20,
            val=[0.0] * 2,
            oos=[0.0] * 2,
            oos_extra={"trade_count": 20.0, "pbo": 0.0},
        ),
    ]
    result = MODULE.run_hybrid_online_allocator(
        rows,
        config=MODULE.HybridOnlineConfig(
            variant="disagreement_switching",
            warmup_days=5,
            lookback_days=5,
            min_positive_score=0.0,
            disagreement_threshold=0.5,
            disagreement_cash_scale=0.5,
            use_current_health_priors=False,
        ),
        refreshed_health_metrics=None,
    )
    sample = result["allocations"][6]
    assert sample["cash_weight"] > 0.0


def test_refreshed_rows_includes_production_guarded_when_artifact_exists(
    tmp_path, monkeypatch
) -> None:
    soft = tmp_path / "soft.json"
    balanced = tmp_path / "balanced.json"
    pair = tmp_path / "pair.json"
    three_way = tmp_path / "three_way.json"
    static_blend = tmp_path / "static_blend.json"
    incumbent = tmp_path / "incumbent.json"
    production = tmp_path / "production.json"

    soft.write_text(
        json.dumps(_portfolio_payload("soft", train=[0.01], val=[0.01], oos=[0.01])),
        encoding="utf-8",
    )
    balanced.write_text(
        json.dumps(_portfolio_payload("balanced", train=[0.008], val=[0.008], oos=[0.008])),
        encoding="utf-8",
    )
    three_way.write_text(
        json.dumps(_portfolio_payload("three_way", train=[0.012], val=[0.012], oos=[0.012])),
        encoding="utf-8",
    )
    static_blend.write_text(
        json.dumps(_portfolio_payload("static", train=[0.009], val=[0.009], oos=[0.009])),
        encoding="utf-8",
    )
    incumbent.write_text(
        json.dumps(_portfolio_payload("incumbent", train=[0.007], val=[0.007], oos=[0.007])),
        encoding="utf-8",
    )
    production.write_text(
        json.dumps(_portfolio_payload("production", train=[0.011], val=[0.011], oos=[0.011])),
        encoding="utf-8",
    )
    pair.write_text(
        json.dumps(
            _row(
                "pair_tactical_mode",
                train=[0.02],
                val=[0.02],
                oos=[0.02],
                oos_extra={"trade_count": 20.0, "pbo": 0.0},
            )
        ),
        encoding="utf-8",
    )

    monkeypatch.setitem(MODULE.REFRESHED_INPUTS, "soft_three_way_regime", soft)
    monkeypatch.setitem(MODULE.REFRESHED_INPUTS, "balanced_overlay_80_20", balanced)
    monkeypatch.setitem(MODULE.REFRESHED_INPUTS, "pair_tactical_mode", pair)
    monkeypatch.setitem(MODULE.REFRESHED_INPUTS, "three_way_regime", three_way)
    monkeypatch.setitem(MODULE.REFRESHED_INPUTS, "static_blend_76_24", static_blend)
    monkeypatch.setitem(MODULE.REFRESHED_INPUTS, "incumbent_only", incumbent)
    monkeypatch.setitem(MODULE.REFRESHED_INPUTS, "production_guarded_portfolio", production)

    active, benchmarks = MODULE._refreshed_rows(split_config=MODULE.HybridSplitConfig())

    active_names = [row["name"] for row in active]
    assert "production_guarded_portfolio" in active_names
    production_row = next(row for row in active if row["name"] == "production_guarded_portfolio")
    assert float(production_row["metadata"]["max_weight_cap"]) == 0.45
    assert {row["name"] for row in benchmarks} >= {
        "three_way_regime",
        "static_blend_76_24",
        "incumbent_only",
    }
