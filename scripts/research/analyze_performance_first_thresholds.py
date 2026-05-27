"""Analyze the performance-first hybrid switch override on current artifacts.

This is a lightweight analytical sweep. It does not rerun heavy research jobs;
it only reads the current reboot-validation artifacts and maps which threshold
combinations would still keep the hybrid guarded override active.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT

GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
DEFAULT_SWITCH_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_operating_switch_current"
    / "portfolio_operating_switch_latest.json"
)
DEFAULT_HYBRID_PATH = (
    GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
)
DEFAULT_OUTPUT_DIR = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "performance_first_threshold_frontier_current"
)

_SWITCH_SPEC = importlib.util.spec_from_file_location(
    "write_portfolio_operating_switch",
    Path(__file__).resolve().parent / "write_portfolio_operating_switch.py",
)
if _SWITCH_SPEC is None or _SWITCH_SPEC.loader is None:
    raise RuntimeError("Failed to load write_portfolio_operating_switch helpers")
_SWITCH = importlib.util.module_from_spec(_SWITCH_SPEC)
sys.modules[_SWITCH_SPEC.name] = _SWITCH
_SWITCH_SPEC.loader.exec_module(_SWITCH)


@dataclass(frozen=True, slots=True)
class ThresholdProfile:
    name: str
    min_oos_return_edge: float
    min_oos_sharpe_edge: float
    min_val_return: float
    min_val_sharpe: float


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(_SWITCH._safe_float(value, default))


def _parse_grid(value: str) -> list[float]:
    items = [token.strip() for token in str(value).split(",")]
    grid = [_safe_float(token) for token in items if token]
    if not grid:
        raise ValueError("grid must contain at least one numeric value")
    return grid


def _hybrid_health_from_payload(hybrid_payload: Mapping[str, Any]) -> dict[str, Any]:
    readiness = dict(hybrid_payload.get("readiness") or {})
    split_metrics = dict(
        dict((hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
            "split_metrics"
        )
        or {}
    )
    val = dict(split_metrics.get("val") or {})
    oos = dict(split_metrics.get("oos") or {})
    return {
        "healthy": True,
        "recommended_stage": str(readiness.get("recommended_stage") or ""),
        "beats_balanced_refreshed": bool(readiness.get("beats_balanced_refreshed")),
        "beats_pair_tactical_refreshed": bool(readiness.get("beats_pair_tactical_refreshed")),
        "val_total_return": _safe_float(val.get("total_return", val.get("return")), 0.0),
        "val_sharpe": _safe_float(val.get("sharpe"), 0.0),
        "oos_total_return": _safe_float(oos.get("total_return", oos.get("return")), 0.0),
        "oos_sharpe": _safe_float(oos.get("sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(oos.get("max_drawdown", oos.get("mdd")), 0.0),
    }


def _balanced_health_from_payload(hybrid_payload: Mapping[str, Any]) -> dict[str, Any]:
    source_metrics = dict(
        dict((hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
            "source_sleeve_metrics"
        )
        or {}
    )
    balanced = dict(source_metrics.get("balanced_overlay_80_20") or {})
    val = dict(balanced.get("val") or {})
    oos = dict(balanced.get("oos") or {})
    return {
        "healthy": _safe_float(oos.get("total_return", oos.get("return")), 0.0) > 0.0
        and _safe_float(oos.get("sharpe"), 0.0) > 0.0,
        "val_total_return": _safe_float(val.get("total_return", val.get("return")), 0.0),
        "val_sharpe": _safe_float(val.get("sharpe"), 0.0),
        "oos_total_return": _safe_float(oos.get("total_return", oos.get("return")), 0.0),
        "oos_sharpe": _safe_float(oos.get("sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(oos.get("max_drawdown", oos.get("mdd")), 0.0),
    }


def _current_profile() -> ThresholdProfile:
    return ThresholdProfile(
        name="current_override",
        min_oos_return_edge=float(_SWITCH.HYBRID_PROMOTION_STRONG_OOS_RETURN_EDGE),
        min_oos_sharpe_edge=float(_SWITCH.HYBRID_PROMOTION_STRONG_OOS_SHARPE_EDGE),
        min_val_return=float(_SWITCH.HYBRID_PROMOTION_STRONG_VAL_TOTAL_RETURN),
        min_val_sharpe=float(_SWITCH.HYBRID_PROMOTION_STRONG_VAL_SHARPE),
    )


def _profile_passes(
    *,
    profile: ThresholdProfile,
    hybrid_health: Mapping[str, Any],
    base_signal: Mapping[str, Any],
) -> bool:
    return bool(
        hybrid_health.get("healthy")
        and hybrid_health.get("beats_balanced_refreshed")
        and hybrid_health.get("beats_pair_tactical_refreshed")
        and str(hybrid_health.get("recommended_stage") or "").strip().lower() == "pilot_candidate"
        and base_signal.get("drawdown_ok")
        and _safe_float(base_signal.get("oos_return_edge"), 0.0) >= profile.min_oos_return_edge
        and _safe_float(base_signal.get("oos_sharpe_edge"), 0.0) >= profile.min_oos_sharpe_edge
        and _safe_float(hybrid_health.get("val_total_return"), 0.0) >= profile.min_val_return
        and _safe_float(hybrid_health.get("val_sharpe"), 0.0) >= profile.min_val_sharpe
    )


def _frontier_profiles(
    *,
    return_grid: Iterable[float],
    sharpe_grid: Iterable[float],
    val_return_grid: Iterable[float],
    val_sharpe_grid: Iterable[float],
    hybrid_health: Mapping[str, Any],
    base_signal: Mapping[str, Any],
) -> dict[str, Any]:
    passing: list[ThresholdProfile] = []
    for ret_edge, sharpe_edge, val_ret, val_sharpe in product(
        return_grid, sharpe_grid, val_return_grid, val_sharpe_grid
    ):
        profile = ThresholdProfile(
            name=f"grid_{ret_edge:.3f}_{sharpe_edge:.2f}_{val_ret:.3f}_{val_sharpe:.2f}",
            min_oos_return_edge=float(ret_edge),
            min_oos_sharpe_edge=float(sharpe_edge),
            min_val_return=float(val_ret),
            min_val_sharpe=float(val_sharpe),
        )
        if _profile_passes(profile=profile, hybrid_health=hybrid_health, base_signal=base_signal):
            passing.append(profile)

    if not passing:
        return {
            "passing_count": 0,
            "total_count": len(list(return_grid))
            * len(list(sharpe_grid))
            * len(list(val_return_grid))
            * len(list(val_sharpe_grid)),
            "frontier_maxima": {},
            "profiles": [],
        }

    frontier_maxima = {
        "max_return_edge_threshold_that_still_promotes": max(
            item.min_oos_return_edge for item in passing
        ),
        "max_sharpe_edge_threshold_that_still_promotes": max(
            item.min_oos_sharpe_edge for item in passing
        ),
        "max_min_val_return_that_still_promotes": max(item.min_val_return for item in passing),
        "max_min_val_sharpe_that_still_promotes": max(item.min_val_sharpe for item in passing),
    }
    profiles = [
        asdict(
            ThresholdProfile(
                name="current_override",
                min_oos_return_edge=_current_profile().min_oos_return_edge,
                min_oos_sharpe_edge=_current_profile().min_oos_sharpe_edge,
                min_val_return=_current_profile().min_val_return,
                min_val_sharpe=_current_profile().min_val_sharpe,
            )
        ),
        asdict(
            ThresholdProfile(
                name="tightest_passing_on_grid",
                min_oos_return_edge=float(
                    frontier_maxima["max_return_edge_threshold_that_still_promotes"]
                ),
                min_oos_sharpe_edge=float(
                    frontier_maxima["max_sharpe_edge_threshold_that_still_promotes"]
                ),
                min_val_return=float(frontier_maxima["max_min_val_return_that_still_promotes"]),
                min_val_sharpe=float(frontier_maxima["max_min_val_sharpe_that_still_promotes"]),
            )
        ),
    ]
    return {
        "passing_count": len(passing),
        "total_count": len(list(return_grid))
        * len(list(sharpe_grid))
        * len(list(val_return_grid))
        * len(list(val_sharpe_grid)),
        "frontier_maxima": frontier_maxima,
        "profiles": profiles,
    }


def build_report(
    *,
    switch_payload: Mapping[str, Any],
    hybrid_payload: Mapping[str, Any],
    return_grid: Iterable[float],
    sharpe_grid: Iterable[float],
    val_return_grid: Iterable[float],
    val_sharpe_grid: Iterable[float],
) -> dict[str, Any]:
    hybrid_health = _hybrid_health_from_payload(hybrid_payload)
    balanced_health = _balanced_health_from_payload(hybrid_payload)
    base_signal = _SWITCH._hybrid_balanced_promotion_signal(
        hybrid_health=hybrid_health,
        balanced_health=balanced_health,
    )
    frontier = _frontier_profiles(
        return_grid=list(return_grid),
        sharpe_grid=list(sharpe_grid),
        val_return_grid=list(val_return_grid),
        val_sharpe_grid=list(val_sharpe_grid),
        hybrid_health=hybrid_health,
        base_signal=base_signal,
    )
    return {
        "artifact_kind": "performance_first_threshold_frontier",
        "generated_at": _SWITCH._utc_now_iso(),
        "switch_mode": str(dict(switch_payload.get("recommended_mode") or {}).get("mode") or ""),
        "current_market_state": dict(switch_payload.get("current_market_state") or {}),
        "hybrid_health": hybrid_health,
        "balanced_health": balanced_health,
        "base_signal": base_signal,
        "current_thresholds": asdict(_current_profile()),
        "grid": {
            "return_edge_grid": list(return_grid),
            "sharpe_edge_grid": list(sharpe_grid),
            "val_return_grid": list(val_return_grid),
            "val_sharpe_grid": list(val_sharpe_grid),
        },
        "frontier": frontier,
    }


def _build_markdown(report: Mapping[str, Any]) -> str:
    state = dict(report.get("current_market_state") or {})
    hybrid = dict(report.get("hybrid_health") or {})
    balanced = dict(report.get("balanced_health") or {})
    signal = dict(report.get("base_signal") or {})
    frontier = dict(report.get("frontier") or {})
    maxima = dict(frontier.get("frontier_maxima") or {})
    lines = [
        "# Performance-first threshold frontier",
        "",
        f"- generated_at: `{report.get('generated_at')}`",
        f"- switch_mode: `{report.get('switch_mode')}`",
        f"- current_market_state: favored_group=`{state.get('favored_group')}`, confidence=`{_safe_float(state.get('confidence'), 0.0):.4f}`, trend=`{state.get('trend_state')}`, breadth=`{state.get('breadth_state')}`, volatility=`{state.get('volatility_state')}`, pair_liquidity=`{state.get('pair_liquidity_state')}`",
        "",
        "## Current reboot-validation metrics",
        f"- hybrid oos return / sharpe / max_dd: `{_safe_float(hybrid.get('oos_total_return'), 0.0):+.4%}` / `{_safe_float(hybrid.get('oos_sharpe'), 0.0):.4f}` / `{_safe_float(hybrid.get('oos_max_drawdown'), 0.0):.4%}`",
        f"- hybrid val return / sharpe: `{_safe_float(hybrid.get('val_total_return'), 0.0):+.4%}` / `{_safe_float(hybrid.get('val_sharpe'), 0.0):.4f}`",
        f"- balanced oos return / sharpe / max_dd: `{_safe_float(balanced.get('oos_total_return'), 0.0):+.4%}` / `{_safe_float(balanced.get('oos_sharpe'), 0.0):.4f}` / `{_safe_float(balanced.get('oos_max_drawdown'), 0.0):.4%}`",
        f"- balanced val return / sharpe: `{_safe_float(balanced.get('val_total_return'), 0.0):+.4%}` / `{_safe_float(balanced.get('val_sharpe'), 0.0):.4f}`",
        f"- oos return edge / sharpe edge: `{_safe_float(signal.get('oos_return_edge'), 0.0):+.4%}` / `{_safe_float(signal.get('oos_sharpe_edge'), 0.0):+.4f}`",
        "",
        "## Current override thresholds",
        f"- min oos return edge: `{_current_profile().min_oos_return_edge:+.4%}`",
        f"- min oos sharpe edge: `{_current_profile().min_oos_sharpe_edge:.4f}`",
        f"- min hybrid val return: `{_current_profile().min_val_return:+.4%}`",
        f"- min hybrid val sharpe: `{_current_profile().min_val_sharpe:.4f}`",
        "",
        "## Frontier summary",
        f"- passing combinations: `{frontier.get('passing_count')}` / `{frontier.get('total_count')}`",
        f"- max return-edge threshold that still promotes: `{_safe_float(maxima.get('max_return_edge_threshold_that_still_promotes'), 0.0):+.4%}`",
        f"- max sharpe-edge threshold that still promotes: `{_safe_float(maxima.get('max_sharpe_edge_threshold_that_still_promotes'), 0.0):.4f}`",
        f"- max min-val-return threshold that still promotes: `{_safe_float(maxima.get('max_min_val_return_that_still_promotes'), 0.0):+.4%}`",
        f"- max min-val-sharpe threshold that still promotes: `{_safe_float(maxima.get('max_min_val_sharpe_that_still_promotes'), 0.0):.4f}`",
    ]
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch-path", type=Path, default=DEFAULT_SWITCH_PATH)
    parser.add_argument("--hybrid-path", type=Path, default=DEFAULT_HYBRID_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--return-edge-grid", default="0.001,0.002,0.003,0.004,0.005,0.006")
    parser.add_argument("--sharpe-edge-grid", default="0.75,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--val-return-grid", default="0.03,0.04,0.05,0.06,0.07")
    parser.add_argument("--val-sharpe-grid", default="2.0,2.5,3.0,3.5,4.0")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    switch_payload = _read_json(Path(args.switch_path).resolve())
    hybrid_payload = _read_json(Path(args.hybrid_path).resolve())
    report = build_report(
        switch_payload=switch_payload,
        hybrid_payload=hybrid_payload,
        return_grid=_parse_grid(args.return_edge_grid),
        sharpe_grid=_parse_grid(args.sharpe_edge_grid),
        val_return_grid=_parse_grid(args.val_return_grid),
        val_sharpe_grid=_parse_grid(args.val_sharpe_grid),
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "performance_first_threshold_frontier_latest.json"
    md_path = output_dir / "performance_first_threshold_frontier_latest.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
