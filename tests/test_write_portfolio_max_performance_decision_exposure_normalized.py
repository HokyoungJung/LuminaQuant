from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "write_portfolio_max_performance_decision.py"
SPEC = importlib.util.spec_from_file_location(
    "write_portfolio_max_performance_decision", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load write_portfolio_max_performance_decision module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _incumbent_bundle_payload() -> dict:
    return {
        "selection_basis": "incumbent_saved_one_shot_weights",
        "candidates": [
            {
                "candidate_id": "incumbent_component",
                "name": "incumbent_component",
                "strategy_class": "StubStrategy",
                "timeframe": "1h",
                "portfolio_weight": 1.0,
            }
        ],
    }


def _incumbent_portfolio_payload() -> dict:
    return {
        "portfolio_metrics": {
            "train": {"total_return": 0.03, "sharpe": 1.0},
            "val": {"total_return": 0.025, "sharpe": 1.1},
            "oos": {
                "total_return": 0.04,
                "sharpe": 1.5,
                "sortino": 2.0,
                "calmar": 4.0,
                "max_drawdown": 0.06,
                "volatility": 0.12,
            },
        },
        "weights": [{"candidate_id": "incumbent_component", "weight": 1.0}],
    }


def _challenger_payload(
    *,
    candidate_key: str,
    label: str,
    oos_total_return: float,
    oos_sharpe: float,
    oos_max_drawdown: float,
    oos_sortino: float = 2.0,
    oos_calmar: float = 4.0,
    oos_volatility: float = 0.12,
    weights: list[dict[str, object]],
    train_total_return: float = 0.05,
    train_sharpe: float = 0.6,
    val_total_return: float = 0.04,
    monthly: list[float] | None = None,
) -> dict:
    return {
        "candidate_key": candidate_key,
        "label": label,
        "source_artifact_kind": "portfolio_followup.custom_candidate",
        "selection_basis": "manual_followup_candidate",
        "train": {
            "total_return": train_total_return,
            "sharpe": train_sharpe,
            "trade_count": 12.0,
            "max_drawdown": 0.05,
        },
        "val": {"total_return": val_total_return, "sharpe": 1.0, "max_drawdown": 0.05},
        "oos": {
            "total_return": oos_total_return,
            "sharpe": oos_sharpe,
            "sortino": oos_sortino,
            "calmar": oos_calmar,
            "max_drawdown": oos_max_drawdown,
            "volatility": oos_volatility,
        },
        "oos_monthly_returns": [
            {"month": f"2026-0{idx}", "total_return": value, "days": 20}
            for idx, value in enumerate(monthly or [0.03, 0.03, 0.03], start=2)
        ],
        "weights": weights,
    }


def _build(tmp_path: Path, *, extra_paths: tuple[Path, ...], research_config=None) -> dict:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    incumbent_bundle.write_text(json.dumps(_incumbent_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(json.dumps(_incumbent_portfolio_payload()), encoding="utf-8")
    return MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tmp_path / "missing_tuned.json",
        dynamic_comparison_path=tmp_path / "missing_dynamic.json",
        overlay_comparison_path=tmp_path / "missing_overlay.json",
        regime_switch_comparison_path=tmp_path / "missing_regime.json",
        grouped_static_blend_path=tmp_path / "missing_static_blend.json",
        grouped_strict_validation_path=None,
        backbone_triplet_path=tmp_path / "missing_triplet.json",
        anchored_comparison_path=tmp_path / "missing_anchor.json",
        extra_candidate_artifact_paths=extra_paths,
        research_config=research_config,
    )


def _write_extra(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _find(candidates: list[dict], key: str) -> dict:
    return next(entry for entry in candidates if entry.get("candidate_key") == key)


# ---------------------------------------------------------------------------
# Flag OFF byte-identity
# ---------------------------------------------------------------------------


def test_builder_flag_off_is_byte_identical(tmp_path: Path) -> None:
    levered = _write_extra(
        tmp_path,
        "levered.json",
        _challenger_payload(
            candidate_key="levered_challenger",
            label="Levered challenger",
            oos_total_return=0.09,
            oos_sharpe=2.1,
            oos_max_drawdown=0.12,
            weights=[{"candidate_id": "a", "weight": 1.0}, {"candidate_id": "b", "weight": 1.0}],
        ),
    )

    default_payload = _build(tmp_path, extra_paths=(levered,), research_config=None)
    explicit_off = _build(
        tmp_path,
        extra_paths=(levered,),
        research_config=SimpleNamespace(exposure_normalized_promotion=False),
    )

    # generated_at is a wall-clock field; everything else must be byte-identical.
    default_payload.pop("generated_at")
    explicit_off.pop("generated_at")
    assert json.dumps(default_payload, sort_keys=True) == json.dumps(explicit_off, sort_keys=True)
    assert "exposure_normalized_promotion" not in default_payload
    assert "multiple_comparison_delta_floor" not in default_payload


# ---------------------------------------------------------------------------
# Flag ON: leverage cannot buy superiority
# ---------------------------------------------------------------------------


def test_leverage_wins_off_but_incumbent_retained_on(tmp_path: Path) -> None:
    levered = _write_extra(
        tmp_path,
        "levered.json",
        _challenger_payload(
            candidate_key="levered_challenger",
            label="Levered challenger",
            oos_total_return=0.075,  # 3.75% per unit of 2x gross < 4% incumbent
            oos_sharpe=2.1,  # sharpe relief >= incumbent 1.5 + 0.5
            oos_max_drawdown=0.12,
            weights=[{"candidate_id": "a", "weight": 1.0}, {"candidate_id": "b", "weight": 1.0}],
        ),
    )

    off = _build(tmp_path, extra_paths=(levered,), research_config=None)
    on = _build(
        tmp_path,
        extra_paths=(levered,),
        research_config=SimpleNamespace(exposure_normalized_promotion=True),
    )

    # Flag OFF: the levered book buys the win.
    assert off["winner"]["status"] == "promoted_challenger"
    assert off["winner"]["candidate_key"] == "levered_challenger"

    # Flag ON: normalized superiority is negative -> incumbent retained.
    assert on["winner"]["status"] == "retained_incumbent"
    assert on["exposure_normalized_promotion"] is True
    challenger = _find(on["candidates"], "levered_challenger")
    assert challenger["promotable"] is False
    assert "oos_total_return_not_above_incumbent" in challenger["rejection_reasons"]
    assert "gross_exposure" in on["promotion_formula"]


def test_genuine_unlevered_edge_still_promotes_on(tmp_path: Path) -> None:
    genuine = _write_extra(
        tmp_path,
        "genuine.json",
        _challenger_payload(
            candidate_key="genuine_challenger",
            label="Genuine challenger",
            oos_total_return=0.09,
            oos_sharpe=2.2,
            oos_max_drawdown=0.05,  # <= incumbent 0.06 -> drawdown relief
            weights=[{"candidate_id": "a", "weight": 1.0}],
        ),
    )

    on = _build(
        tmp_path,
        extra_paths=(genuine,),
        research_config=SimpleNamespace(exposure_normalized_promotion=True),
    )

    assert on["winner"]["status"] == "promoted_challenger"
    assert on["winner"]["candidate_key"] == "genuine_challenger"
    # Single challenger -> no multiple-comparison inflation.
    assert on["multiple_comparison_delta_floor"] == 0.0


# ---------------------------------------------------------------------------
# Flag ON: multiple-comparison floor suppresses a marginal winner
# ---------------------------------------------------------------------------


def test_multiple_comparison_floor_suppresses_marginal_winner(tmp_path: Path) -> None:
    # Marginal but genuinely promotable challenger (thin edge over incumbent).
    marginal = _write_extra(
        tmp_path,
        "marginal.json",
        _challenger_payload(
            candidate_key="marginal_challenger",
            label="Marginal challenger",
            oos_total_return=0.05,
            oos_sharpe=1.6,
            oos_max_drawdown=0.05,  # <= incumbent -> relief
            weights=[{"candidate_id": "a", "weight": 1.0}],
        ),
    )
    # A high-scoring challenger that fails a hard gate (negative train return),
    # so it is not promotable but widens the cross-challenger delta spread.
    dispersed = _write_extra(
        tmp_path,
        "dispersed.json",
        _challenger_payload(
            candidate_key="dispersed_challenger",
            label="Dispersed challenger",
            oos_total_return=0.30,
            oos_sharpe=5.0,
            oos_sortino=5.0,
            oos_calmar=8.0,
            oos_max_drawdown=0.05,
            oos_volatility=0.10,
            weights=[{"candidate_id": "a", "weight": 1.0}],
            train_total_return=-0.01,  # hard-fail gate -> not promotable
        ),
    )

    off = _build(tmp_path, extra_paths=(marginal, dispersed), research_config=None)
    on = _build(
        tmp_path,
        extra_paths=(marginal, dispersed),
        research_config=SimpleNamespace(exposure_normalized_promotion=True),
    )

    # Flag OFF: the marginal challenger is promoted (no multiplicity guard).
    assert off["winner"]["status"] == "promoted_challenger"
    assert off["winner"]["candidate_key"] == "marginal_challenger"

    # Flag ON: the multiple-comparison floor (inflated by the dispersed
    # challenger) exceeds the marginal edge -> incumbent retained.
    assert on["winner"]["status"] == "retained_incumbent"
    assert on["multiple_comparison_delta_floor"] > 0.0
    marginal_entry = _find(on["candidates"], "marginal_challenger")
    # It cleared the per-candidate gates but did not clear the multiplicity floor.
    assert marginal_entry["promotable"] is True
    assert "multiple-comparison" in on["winner"]["reason"]
