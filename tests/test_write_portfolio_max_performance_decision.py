from __future__ import annotations

import inspect
import importlib.util
import json
import sys
from pathlib import Path

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


def _call_build_decision(**kwargs):
    params = inspect.signature(MODULE.build_portfolio_max_performance_decision).parameters
    kwargs.setdefault("grouped_static_blend_path", None)
    return MODULE.build_portfolio_max_performance_decision(
        **{name: value for name, value in kwargs.items() if name in params}
    )


def _bundle_payload() -> dict[str, object]:
    return {
        "artifact_kind": "portfolio_one_shot_incumbent_bundle",
        "selection_basis": "incumbent_saved_one_shot_weights",
        "candidates": [
            {
                "candidate_id": "inc-a",
                "name": "inc-a",
                "strategy_class": "StubStrategy",
                "strategy_timeframe": "1h",
                "portfolio_weight": 0.5,
            },
            {
                "candidate_id": "inc-b",
                "name": "inc-b",
                "strategy_class": "StubStrategy",
                "strategy_timeframe": "1h",
                "portfolio_weight": 0.5,
            },
        ],
    }


def _portfolio_payload(
    *,
    total_return: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
    volatility: float,
) -> dict[str, object]:
    return {
        "artifact_kind": "portfolio_optimization",
        "portfolio_metrics": {
            "train": {
                "total_return": 0.02,
                "sharpe": 0.8,
                "sortino": 1.1,
                "calmar": 2.2,
                "max_drawdown": 0.06,
                "volatility": 0.11,
            },
            "oos": {
                "total_return": total_return,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
            },
            "val": {
                "total_return": 0.03,
                "sharpe": 1.4,
                "sortino": 1.8,
                "calmar": 3.5,
                "max_drawdown": 0.05,
                "volatility": 0.10,
            },
        },
        "weights": [
            {"candidate_id": "inc-a", "weight": 0.5},
            {"candidate_id": "inc-b", "weight": 0.5},
        ],
    }


def _monthly_rows(*values: float) -> list[dict[str, float | str | int]]:
    out: list[dict[str, float | str | int]] = []
    for idx, value in enumerate(values, start=2):
        out.append({"month": f"2026-0{idx}", "total_return": value, "days": 20})
    return out


def _comparison_section(
    *,
    path: Path,
    total_return: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
    volatility: float,
    train_total_return: float = 0.01,
    train_sharpe: float = 0.2,
    oos_monthly_returns: list[float] | None = None,
) -> dict[str, object]:
    return {
        "path": str(path),
        "train": {
            "total_return": train_total_return,
            "sharpe": train_sharpe,
            "sortino": 0.9,
            "calmar": 1.5,
            "max_drawdown": 0.06,
            "volatility": 0.10,
        },
        "val": {"total_return": 0.02, "sharpe": 1.0},
        "oos": {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
        },
        "oos_monthly_returns": _monthly_rows(*(oos_monthly_returns or [max(total_return, 0.03)])),
        "weights": [{"candidate_id": "c1", "weight": 1.0}],
    }


def _anchored_candidate_entries(payload: dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for entry in list(payload.get("candidates") or []):
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("candidate_key") or "").lower()
        label = str(entry.get("label") or "").lower()
        if any(
            token in key or token in label for token in ("anchored", "four_sleeve", "four-sleeve")
        ):
            out.append(entry)
    return out


def _regime_switch_section(
    *,
    path: Path,
    total_return: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
    volatility: float,
    train_total_return: float = 0.01,
    train_sharpe: float = 0.2,
    oos_monthly_returns: list[float] | None = None,
) -> dict[str, object]:
    return {
        "path": str(path),
        "train": {
            "total_return": train_total_return,
            "sharpe": train_sharpe,
            "sortino": 1.2,
            "calmar": 2.1,
            "max_drawdown": 0.05,
            "volatility": 0.09,
        },
        "val": {
            "total_return": 0.015,
            "sharpe": 1.1,
            "sortino": 1.7,
            "calmar": 3.0,
            "max_drawdown": 0.03,
            "volatility": 0.08,
        },
        "oos": {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
        },
        "oos_monthly_returns": _monthly_rows(*(oos_monthly_returns or [max(total_return, 0.03)])),
        "weights": [
            {
                "candidate_id": "autoresearch_pair_55_45",
                "name": "autoresearch_pair_55_45",
                "weight": 0.35,
            },
            {
                "candidate_id": "current_one_shot_incumbent",
                "name": "current_one_shot_incumbent",
                "weight": 0.2,
            },
        ],
    }


def _grouped_allocator_payload(
    *,
    total_return: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
    volatility: float,
) -> dict[str, object]:
    return {
        "artifact_kind": "portfolio_three_way_market_regime_allocator",
        "selection_basis": "grouped_three_way_market_regime_allocator",
        "split_metrics": {
            "train": {"total_return": 0.02, "sharpe": 0.5},
            "val": {"total_return": 0.03, "sharpe": 1.2},
            "oos": {
                "total_return": total_return,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
            },
        },
        "oos_monthly_returns": _monthly_rows(
            max(total_return, 0.03), max(total_return, 0.03), max(total_return, 0.03)
        ),
        "current_state": {
            "state": "autoresearch_55_45",
            "raw_target_state": "incumbent",
            "confidence": 1.0,
            "weights": {"autoresearch_55_45": 1.0},
        },
        "state_summary": {
            "oos": {"counts": {"autoresearch_55_45": 10, "incumbent": 20}},
        },
    }


def _grouped_strict_validation_payload(
    *,
    train_total_return: float,
    train_sharpe: float,
    val_total_return: float,
    val_sharpe: float,
    oos_total_return: float,
    oos_sharpe: float,
    oos_max_drawdown: float,
    liquidation_count: int = 0,
) -> dict[str, object]:
    return {
        "artifact_kind": "grouped_allocator_strict_leverage_validation",
        "leverage_by_state": {"incumbent": 3, "blend_85_15": 5, "autoresearch_55_45": 2},
        "comparison_vs_promoted_challenger": {
            "oos": {
                "total_return_delta": oos_total_return - 0.05,
                "sharpe_delta": oos_sharpe - 1.60,
                "max_drawdown_delta": oos_max_drawdown - 0.065,
            }
        },
        "strict_allocator": {
            "current_state": {"state": "blend_85_15", "weights": {"blend_85_15": 1.0}},
            "state_summary": {"oos": {"counts": {"blend_85_15": 10}}},
            "split_metrics": {
                "train": {"total_return": train_total_return, "sharpe": train_sharpe},
                "val": {"total_return": val_total_return, "sharpe": val_sharpe},
                "oos": {
                    "total_return": oos_total_return,
                    "sharpe": oos_sharpe,
                    "sortino": 2.5,
                    "calmar": 5.0,
                    "max_drawdown": oos_max_drawdown,
                    "volatility": 0.12,
                },
            },
            "oos_monthly_returns": _monthly_rows(
                max(oos_total_return, 0.03),
                max(oos_total_return, 0.03),
                max(oos_total_return, 0.03),
            ),
            "state_leverage_validation": {
                "liquidation_counts": {
                    "incumbent": 0,
                    "blend_85_15": liquidation_count,
                    "autoresearch_55_45": 0,
                }
            },
        },
    }


def _static_blend_payload(
    *,
    total_return: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
    volatility: float,
) -> dict[str, object]:
    return {
        "artifact_kind": "grouped_incumbent_autoresearch_static_blend_portfolio",
        "selection_basis": "validation_only_static_group_blend",
        "best_weights": {
            "current_one_shot_incumbent": 0.85,
            "autoresearch_pair_55_45": 0.15,
        },
        "split_metrics": {
            "train": {
                "total_return": 0.03,
                "sharpe": 0.2,
                "sortino": 0.4,
                "calmar": 0.5,
                "max_drawdown": 0.06,
                "volatility": 0.08,
            },
            "val": {
                "total_return": 0.025,
                "sharpe": 4.0,
                "sortino": 6.0,
                "calmar": 20.0,
                "max_drawdown": 0.01,
                "volatility": 0.07,
            },
            "oos": {
                "total_return": total_return,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
            },
        },
        "oos_monthly_returns": _monthly_rows(
            max(total_return, 0.03), max(total_return, 0.03), max(total_return, 0.03)
        ),
    }


def _production_guarded_payload(
    *,
    total_return: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
    volatility: float,
) -> dict[str, object]:
    return {
        "artifact_kind": "production_guarded_portfolio",
        "selection_basis": "saved_sleeve_blend_with_drawdown_throttle",
        "active_exposure": 0.95,
        "cash_weight": 0.05,
        "portfolio_metrics": {
            "train": {
                "total_return": 0.03,
                "sharpe": 0.8,
                "sortino": 1.0,
                "calmar": 1.5,
                "max_drawdown": 0.06,
                "volatility": 0.09,
            },
            "val": {
                "total_return": 0.04,
                "sharpe": 1.5,
                "sortino": 2.1,
                "calmar": 3.3,
                "max_drawdown": 0.04,
                "volatility": 0.08,
            },
            "oos": {
                "total_return": total_return,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
            },
        },
        "weights": [
            {"candidate_id": "hybrid_guarded_mode", "weight": 0.4},
            {"candidate_id": "static_blend_76_24", "weight": 0.35},
            {"candidate_id": "incumbent_only", "weight": 0.2},
        ],
        "oos_monthly_returns": _monthly_rows(
            max(total_return, 0.03), max(total_return, 0.03), max(total_return, 0.03)
        ),
    }


def test_build_portfolio_max_performance_decision_retains_incumbent_when_no_challenger_clears_threshold(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.70,
                sortino=2.20,
                calmar=4.50,
                max_drawdown=0.06,
                volatility=0.12,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "exact_window_frozen_tuned": _comparison_section(
                    path=tmp_path / "tuned.json",
                    total_return=0.049,
                    sharpe=1.69,
                    sortino=2.10,
                    calmar=4.20,
                    max_drawdown=0.065,
                    volatility=0.13,
                ),
            }
        ),
        encoding="utf-8",
    )
    dynamic.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_dynamic_portfolio": _comparison_section(
                    path=tmp_path / "dynamic.json",
                    total_return=0.048,
                    sharpe=1.55,
                    sortino=2.00,
                    calmar=4.10,
                    max_drawdown=0.07,
                    volatility=0.14,
                ),
            }
        ),
        encoding="utf-8",
    )
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.051,
                    sharpe=1.69,
                    sortino=2.18,
                    calmar=4.45,
                    max_drawdown=0.061,
                    volatility=0.121,
                ),
            }
        ),
        encoding="utf-8",
    )
    triplet.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_backbone_triplet_search",
                "val": {"total_return": 0.03, "sharpe": 1.1},
                "oos": {
                    "total_return": 0.0505,
                    "sharpe": 1.71,
                    "sortino": 2.21,
                    "calmar": 4.46,
                    "max_drawdown": 0.0605,
                    "volatility": 0.121,
                },
                "weights": [{"candidate_id": "triplet", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_static_blend_path=None,
        backbone_triplet_path=triplet,
    )

    assert payload["winner"]["candidate_key"] == "current_one_shot_incumbent"
    assert payload["winner"]["status"] == "retained_incumbent"
    challengers = {entry["candidate_key"]: entry for entry in payload["candidates"][1:]}
    assert challengers["causal_overlay_portfolio"]["promotable"] is False
    assert "No challenger cleared" in payload["winner"]["reason"]


def test_build_portfolio_max_performance_decision_promotes_clear_improver(tmp_path: Path) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "exact_window_frozen_tuned": _comparison_section(
                    path=tmp_path / "tuned.json",
                    total_return=0.045,
                    sharpe=1.3,
                    sortino=1.8,
                    calmar=3.8,
                    max_drawdown=0.08,
                    volatility=0.16,
                ),
            }
        ),
        encoding="utf-8",
    )
    dynamic.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_dynamic_portfolio": _comparison_section(
                    path=tmp_path / "dynamic.json",
                    total_return=0.048,
                    sharpe=1.4,
                    sortino=1.9,
                    calmar=3.9,
                    max_drawdown=0.075,
                    volatility=0.15,
                ),
            }
        ),
        encoding="utf-8",
    )
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.072,
                    sharpe=1.95,
                    sortino=2.80,
                    calmar=5.20,
                    max_drawdown=0.055,
                    volatility=0.12,
                ),
            }
        ),
        encoding="utf-8",
    )
    triplet.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_backbone_triplet_search",
                "val": {"total_return": 0.03, "sharpe": 1.0},
                "oos": {
                    "total_return": 0.049,
                    "sharpe": 1.45,
                    "sortino": 2.0,
                    "calmar": 4.1,
                    "max_drawdown": 0.069,
                    "volatility": 0.139,
                },
                "weights": [{"candidate_id": "triplet", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_static_blend_path=None,
        backbone_triplet_path=triplet,
    )

    assert payload["winner"]["candidate_key"] == "causal_overlay_portfolio"
    assert payload["winner"]["status"] == "promoted_challenger"
    overlay_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "causal_overlay_portfolio"
    )
    assert overlay_entry["promotable"] is True
    assert overlay_entry["promotion_score_delta"] > 0.10
    assert overlay_entry["oos_total_return_delta"] > 0.0


def test_build_portfolio_max_performance_decision_includes_extra_candidate_artifact(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    extra = tmp_path / "strict_autoresearch_1x_practical_shadow_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.0036,
                sharpe=1.60,
                sortino=2.10,
                calmar=4.20,
                max_drawdown=0.0032,
                volatility=0.13,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )
    extra.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_superiority_retune_candidate",
                "candidate_key": "strict_autoresearch_1x_practical_shadow",
                "label": "Strict autoresearch practical challenger",
                "selection_basis": "production_guarded_plus_strict_autoresearch_1x_practical_cap",
                "portfolio_metrics": {
                    "train": {
                        "total_return": 0.0192,
                        "sharpe": 0.34,
                        "sortino": 0.59,
                        "calmar": 0.35,
                        "max_drawdown": 0.0542,
                        "volatility": 0.0621,
                    },
                    "val": {
                        "total_return": 0.0594,
                        "sharpe": 3.97,
                        "sortino": 11.98,
                        "calmar": 46.48,
                        "max_drawdown": 0.0092,
                        "volatility": 0.0909,
                    },
                    "oos": {
                        "total_return": 0.00526,
                        "sharpe": 2.24,
                        "sortino": 2.18,
                        "calmar": 10.02,
                        "max_drawdown": 0.00313,
                        "volatility": 0.0138,
                    },
                },
                "weights": [
                    {"candidate_id": "production_guarded_portfolio", "weight": 0.8},
                    {"candidate_id": "strict_autoresearch_1x", "weight": 0.2},
                ],
                "oos_monthly_returns": _monthly_rows(0.03, 0.03, 0.03),
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_static_blend_path=None,
        backbone_triplet_path=triplet,
        extra_candidate_artifact_paths=(extra,),
    )

    extra_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "strict_autoresearch_1x_practical_shadow"
    )
    assert extra_entry["promotable"] is True
    assert payload["winner"]["candidate_key"] == "strict_autoresearch_1x_practical_shadow"


def test_build_portfolio_max_performance_decision_requires_positive_return_even_with_drawdown_relief(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.049,
                    sharpe=2.10,
                    sortino=3.10,
                    calmar=5.50,
                    max_drawdown=0.05,
                    volatility=0.11,
                ),
            }
        ),
        encoding="utf-8",
    )
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
    )

    overlay_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "causal_overlay_portfolio"
    )
    assert overlay_entry["promotion_score_delta"] > 0.25
    assert overlay_entry["oos_max_drawdown_delta"] < 0.0
    assert overlay_entry["oos_total_return_delta"] < 0.0
    assert overlay_entry["promotable"] is False


def test_build_portfolio_max_performance_decision_includes_anchored_four_sleeve_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    anchored = tmp_path / "portfolio_four_sleeve_comparison_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.60,
                sortino=2.10,
                calmar=4.20,
                max_drawdown=0.065,
                volatility=0.13,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )

    anchored_section = _comparison_section(
        path=tmp_path / "anchored_tuned.json",
        total_return=0.049,
        sharpe=1.55,
        sortino=2.00,
        calmar=4.00,
        max_drawdown=0.067,
        volatility=0.132,
    )
    anchored.write_text(
        json.dumps(
            {
                "selection_basis": "incumbent_anchor_rolling_gate",
                "rolling_gate": {
                    "selection_basis": "train_val_only",
                    "survives_train_val": True,
                    "survives": True,
                },
                "portfolio_four_sleeve_tuned": anchored_section,
                "portfolio_four_sleeve_anchored_tuned": anchored_section,
                "anchored_four_sleeve_tuned": anchored_section,
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
        anchored_comparison_path=anchored,
        anchored_tuned_comparison_path=anchored,
        portfolio_four_sleeve_comparison_path=anchored,
        four_sleeve_comparison_path=anchored,
    )

    anchored_entries = _anchored_candidate_entries(payload)
    assert anchored_entries, "expected anchored four-sleeve candidate to be included"
    assert payload["winner"]["candidate_key"] == "current_one_shot_incumbent"
    assert anchored_entries[0]["promotable"] is False


def test_build_portfolio_max_performance_decision_includes_grouped_allocator_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    grouped_allocator = tmp_path / "three_way_market_regime_allocator_latest.json"
    grouped_strict = tmp_path / "grouped_allocator_strict_validation_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )
    grouped_allocator.write_text(
        json.dumps(
            _grouped_allocator_payload(
                total_return=0.072,
                sharpe=1.95,
                sortino=2.80,
                calmar=5.20,
                max_drawdown=0.07,
                volatility=0.12,
            )
        ),
        encoding="utf-8",
    )
    grouped_strict.write_text(
        json.dumps(
            _grouped_strict_validation_payload(
                train_total_return=0.03,
                train_sharpe=1.0,
                val_total_return=0.02,
                val_sharpe=1.1,
                oos_total_return=0.072,
                oos_sharpe=1.95,
                oos_max_drawdown=0.07,
            )
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_allocator_path=grouped_allocator,
        grouped_strict_validation_path=grouped_strict,
        grouped_static_blend_path=None,
        backbone_triplet_path=triplet,
    )

    grouped_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "grouped_three_way_market_regime_allocator"
    )
    assert grouped_entry["promotable"] is True
    assert payload["winner"]["candidate_key"] == "grouped_three_way_market_regime_allocator"


def test_grouped_allocator_strict_liquidation_evidence_blocks_promotion(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    grouped_allocator = tmp_path / "three_way_market_regime_allocator_latest.json"
    grouped_strict = tmp_path / "grouped_allocator_strict_validation_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.60,
                sortino=2.10,
                calmar=4.20,
                max_drawdown=0.065,
                volatility=0.13,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )
    grouped_allocator.write_text(
        json.dumps(
            _grouped_allocator_payload(
                total_return=0.072,
                sharpe=1.95,
                sortino=2.80,
                calmar=5.20,
                max_drawdown=0.07,
                volatility=0.12,
            )
        ),
        encoding="utf-8",
    )
    grouped_strict.write_text(
        json.dumps(
            _grouped_strict_validation_payload(
                train_total_return=0.03,
                train_sharpe=1.0,
                val_total_return=0.02,
                val_sharpe=1.1,
                oos_total_return=0.072,
                oos_sharpe=1.95,
                oos_max_drawdown=0.07,
                liquidation_count=1,
            )
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_allocator_path=grouped_allocator,
        grouped_strict_validation_path=grouped_strict,
        grouped_static_blend_path=None,
        backbone_triplet_path=triplet,
    )

    grouped_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "grouped_three_way_market_regime_allocator"
    )
    assert grouped_entry["promotable"] is False
    assert "strict_liquidation_count_positive" in grouped_entry["rejection_reasons"]
    assert payload["winner"]["candidate_key"] == "current_one_shot_incumbent"


def test_build_portfolio_max_performance_decision_includes_static_blend_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    static_blend = tmp_path / "grouped_static_blend.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )
    static_blend.write_text(
        json.dumps(
            _static_blend_payload(
                total_return=0.06,
                sharpe=1.8,
                sortino=2.4,
                calmar=5.0,
                max_drawdown=0.06,
                volatility=0.12,
            )
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_static_blend_path=static_blend,
        backbone_triplet_path=triplet,
    )

    blend_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "incumbent_autoresearch_static_blend"
    )
    assert blend_entry["label"] == "Incumbent/55-45 static blend challenger"
    assert any("Blend weights=" in note for note in blend_entry["notes"])


def test_build_portfolio_max_performance_decision_can_promote_static_blend(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    static_blend = tmp_path / "grouped_static_blend.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )
    static_blend.write_text(
        json.dumps(
            _static_blend_payload(
                total_return=0.07,
                sharpe=2.1,
                sortino=2.8,
                calmar=5.5,
                max_drawdown=0.05,
                volatility=0.12,
            )
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_static_blend_path=static_blend,
        backbone_triplet_path=triplet,
    )

    blend_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "incumbent_autoresearch_static_blend"
    )
    assert blend_entry["promotable"] is True
    assert payload["winner"]["candidate_key"] == "incumbent_autoresearch_static_blend"


def test_build_portfolio_max_performance_decision_includes_production_guarded_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    production_guarded = tmp_path / "production_guarded_portfolio_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    production_guarded.write_text(
        json.dumps(
            _production_guarded_payload(
                total_return=0.06,
                sharpe=1.8,
                sortino=2.4,
                calmar=4.8,
                max_drawdown=0.05,
                volatility=0.11,
            )
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tmp_path / "missing_tuned.json",
        dynamic_comparison_path=tmp_path / "missing_dynamic.json",
        overlay_comparison_path=tmp_path / "missing_overlay.json",
        backbone_triplet_path=tmp_path / "missing_triplet.json",
        production_guarded_path=production_guarded,
    )

    guarded_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "production_guarded_portfolio"
    )
    assert guarded_entry["selection_basis"] == "saved_sleeve_blend_with_drawdown_throttle"
    assert "Exposure=" in " ".join(guarded_entry["notes"])


def test_build_portfolio_max_performance_decision_can_promote_production_guarded(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    production_guarded = tmp_path / "production_guarded_portfolio_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    production_guarded.write_text(
        json.dumps(
            _production_guarded_payload(
                total_return=0.075,
                sharpe=2.05,
                sortino=2.8,
                calmar=5.6,
                max_drawdown=0.05,
                volatility=0.10,
            )
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tmp_path / "missing_tuned.json",
        dynamic_comparison_path=tmp_path / "missing_dynamic.json",
        overlay_comparison_path=tmp_path / "missing_overlay.json",
        backbone_triplet_path=tmp_path / "missing_triplet.json",
        production_guarded_path=production_guarded,
    )

    assert payload["winner"]["candidate_key"] == "production_guarded_portfolio"


def test_grouped_allocator_strict_gate_blocks_promotion_when_train_is_negative(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    grouped_allocator = tmp_path / "three_way_market_regime_allocator_latest.json"
    grouped_strict = tmp_path / "grouped_allocator_strict_validation_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.60,
                sortino=2.10,
                calmar=4.20,
                max_drawdown=0.065,
                volatility=0.13,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )
    grouped_allocator.write_text(
        json.dumps(
            _grouped_allocator_payload(
                total_return=0.30,
                sharpe=2.50,
                sortino=3.00,
                calmar=6.00,
                max_drawdown=0.10,
                volatility=0.20,
            )
        ),
        encoding="utf-8",
    )
    grouped_strict.write_text(
        json.dumps(
            _grouped_strict_validation_payload(
                train_total_return=-0.20,
                train_sharpe=-0.5,
                val_total_return=0.03,
                val_sharpe=1.4,
                oos_total_return=0.12,
                oos_sharpe=2.1,
                oos_max_drawdown=0.08,
            )
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_allocator_path=grouped_allocator,
        grouped_strict_validation_path=grouped_strict,
        grouped_static_blend_path=None,
        backbone_triplet_path=triplet,
    )

    grouped_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "grouped_three_way_market_regime_allocator"
    )
    assert grouped_entry["promotable"] is False
    assert "strict train total return is not positive" in grouped_entry["decision_reason"]
    assert payload["winner"]["candidate_key"] == "current_one_shot_incumbent"


def test_build_portfolio_max_performance_decision_includes_regime_switch_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    regime_switch = tmp_path / "portfolio_regime_switch_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.60,
                sortino=2.10,
                calmar=4.20,
                max_drawdown=0.065,
                volatility=0.13,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )
    regime_switch.write_text(
        json.dumps(
            {
                "selection_basis": "regime_switching_allocator",
                "regime_switching_portfolio": _regime_switch_section(
                    path=tmp_path / "regime_switching.json",
                    total_return=0.049,
                    sharpe=1.58,
                    sortino=2.05,
                    calmar=4.15,
                    max_drawdown=0.06,
                    volatility=0.11,
                ),
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        regime_switch_comparison_path=regime_switch,
        backbone_triplet_path=triplet,
    )

    regime_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "regime_switching_portfolio"
    )
    assert regime_entry["label"] == "Regime-switch allocator challenger"
    assert regime_entry["selection_basis"] == "regime_switching_allocator"
    assert any("Active final weights:" in note for note in regime_entry["notes"])
    assert regime_entry["promotable"] is False


def test_build_portfolio_max_performance_decision_can_promote_regime_switch_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    regime_switch = tmp_path / "portfolio_regime_switch_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )
    regime_switch.write_text(
        json.dumps(
            {
                "selection_basis": "regime_switching_allocator",
                "regime_switching_portfolio": _regime_switch_section(
                    path=tmp_path / "regime_switching.json",
                    total_return=0.072,
                    sharpe=1.98,
                    sortino=2.95,
                    calmar=5.50,
                    max_drawdown=0.05,
                    volatility=0.10,
                ),
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        regime_switch_comparison_path=regime_switch,
        backbone_triplet_path=triplet,
    )

    assert payload["winner"]["candidate_key"] == "regime_switching_portfolio"
    assert payload["winner"]["status"] == "promoted_challenger"


def test_build_portfolio_max_performance_decision_rejects_when_monthly_mean_is_too_low(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.072,
                    sharpe=2.05,
                    sortino=2.90,
                    calmar=5.40,
                    max_drawdown=0.05,
                    volatility=0.11,
                    oos_monthly_returns=[0.015, 0.017, 0.019],
                ),
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        overlay_comparison_path=overlay,
        tuned_comparison_path=tmp_path / "missing_tuned.json",
        dynamic_comparison_path=tmp_path / "missing_dynamic.json",
        regime_switch_comparison_path=tmp_path / "missing_regime.json",
        backbone_triplet_path=tmp_path / "missing_triplet.json",
    )

    overlay_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "causal_overlay_portfolio"
    )
    assert overlay_entry["promotable"] is False
    assert "oos_monthly_mean_below_floor" in overlay_entry["rejection_reasons"]


def test_build_portfolio_max_performance_decision_includes_portfolio_superiority_meta_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    superiority_meta = tmp_path / "portfolio_superiority_meta_portfolio_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    superiority_meta.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_superiority_meta_portfolio",
                "selection_basis": "validation_rank_then_locked_oos_robust_gates",
                "universe": "raw_basis",
                "fallback_retune_required": False,
                "validation_objective": 2.75,
                "split_metrics": {
                    "train": {
                        "total_return": 0.03,
                        "sharpe": 0.6,
                        "sortino": 1.1,
                        "calmar": 1.8,
                        "max_drawdown": 0.05,
                        "volatility": 0.09,
                    },
                    "val": {
                        "total_return": 0.04,
                        "sharpe": 1.6,
                        "sortino": 2.0,
                        "calmar": 3.2,
                        "max_drawdown": 0.04,
                        "volatility": 0.08,
                    },
                    "oos": {
                        "total_return": 0.073,
                        "sharpe": 2.02,
                        "sortino": 2.85,
                        "calmar": 5.35,
                        "max_drawdown": 0.052,
                        "volatility": 0.11,
                    },
                },
                "weights": [
                    {"candidate_id": "current_one_shot_incumbent", "weight": 0.35},
                    {"candidate_id": "autoresearch_pair_55_45", "weight": 0.25},
                    {"candidate_id": "soft_allocator", "weight": 0.20},
                    {"candidate_id": "regime_switch", "weight": 0.20},
                ],
                "oos_monthly_returns": _monthly_rows(0.024, 0.031, 0.028),
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tmp_path / "missing_tuned.json",
        dynamic_comparison_path=tmp_path / "missing_dynamic.json",
        overlay_comparison_path=tmp_path / "missing_overlay.json",
        regime_switch_comparison_path=tmp_path / "missing_regime.json",
        backbone_triplet_path=tmp_path / "missing_triplet.json",
        portfolio_superiority_meta_path=superiority_meta,
    )

    meta_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "portfolio_superiority_meta_portfolio"
    )
    assert meta_entry["promotable"] is True
    assert meta_entry["selection_basis"] == "validation_rank_then_locked_oos_robust_gates"
    assert payload["winner"]["candidate_key"] == "portfolio_superiority_meta_portfolio"
    assert payload["fallback_retune_required"] is False


def test_build_portfolio_max_performance_decision_allows_custom_anchored_metadata(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    anchored = tmp_path / "portfolio_four_sleeve_comparison_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(
        json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8"
    )

    anchored_section = _comparison_section(
        path=tmp_path / "tradecount_probe.json",
        total_return=0.08,
        sharpe=2.30,
        sortino=3.20,
        calmar=6.10,
        max_drawdown=0.04,
        volatility=0.10,
    )
    anchored_section["notes"] = ["section-level override should also be accepted"]
    anchored.write_text(
        json.dumps(
            {
                "selection_basis": "ignored_by_custom_metadata",
                "challenger_metadata": {
                    "candidate_key": "autonomous_tradecount_pair_topcap_probe",
                    "label": "Autonomous tradecount pair+topcap probe",
                    "source_artifact_kind": "autonomous_probe.tradecount_pair_topcap",
                    "selection_basis": "autonomous_tradecount_screen_pair_topcap",
                    "notes": ["Custom autonomous probe challenger."],
                },
                "anchored_four_sleeve_tuned": anchored_section,
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
        anchored_comparison_path=anchored,
        anchored_tuned_comparison_path=anchored,
        portfolio_four_sleeve_comparison_path=anchored,
        four_sleeve_comparison_path=anchored,
    )

    probe_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "autonomous_tradecount_pair_topcap_probe"
    )
    assert probe_entry["label"] == "Autonomous tradecount pair+topcap probe"
    assert probe_entry["source_artifact_kind"] == "autonomous_probe.tradecount_pair_topcap"
    assert probe_entry["selection_basis"] == "autonomous_tradecount_screen_pair_topcap"
    assert "Custom autonomous probe challenger." in probe_entry["notes"]
    assert payload["winner"]["candidate_key"] == "autonomous_tradecount_pair_topcap_probe"
    assert payload["winner"]["status"] == "promoted_challenger"


def test_write_portfolio_max_performance_decision_writes_latest_files(tmp_path: Path) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    output_json = tmp_path / "portfolio_max_performance_decision_latest.json"
    output_md = tmp_path / "portfolio_max_performance_decision_latest.md"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "exact_window_frozen_tuned": _comparison_section(
                    path=tmp_path / "tuned.json",
                    total_return=0.045,
                    sharpe=1.3,
                    sortino=1.8,
                    calmar=3.8,
                    max_drawdown=0.08,
                    volatility=0.16,
                ),
            }
        ),
        encoding="utf-8",
    )
    dynamic.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_dynamic_portfolio": _comparison_section(
                    path=tmp_path / "dynamic.json",
                    total_return=0.048,
                    sharpe=1.4,
                    sortino=1.9,
                    calmar=3.9,
                    max_drawdown=0.075,
                    volatility=0.15,
                ),
            }
        ),
        encoding="utf-8",
    )
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.072,
                    sharpe=1.95,
                    sortino=2.80,
                    calmar=5.20,
                    max_drawdown=0.055,
                    volatility=0.12,
                ),
            }
        ),
        encoding="utf-8",
    )
    triplet.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_backbone_triplet_search",
                "val": {"total_return": 0.03, "sharpe": 1.0},
                "oos": {
                    "total_return": 0.049,
                    "sharpe": 1.45,
                    "sortino": 2.0,
                    "calmar": 4.1,
                    "max_drawdown": 0.069,
                    "volatility": 0.139,
                },
                "weights": [{"candidate_id": "triplet", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.write_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        grouped_static_blend_path=None,
        backbone_triplet_path=triplet,
        output_json_path=output_json,
        output_md_path=output_md,
    )

    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["winner"]["candidate_key"] == "causal_overlay_portfolio"
    assert "portfolio max-performance decision" in output_md.read_text(encoding="utf-8")
