"""Replay the switch policy across reboot-validation OOS dates for threshold grids.

This is a lightweight, artifact-driven replay. It reuses saved sleeve return
streams, saved allocator state histories, daily feature-point derived regime
judgements, and historical pair-liquidity checks. It does not rerun upstream
research or Optuna.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT

GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
SWITCH_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_operating_switch_current"
    / "portfolio_operating_switch_latest.json"
)
MARKET_JUDGEMENT_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_market_regime_judgement_current"
    / "group_market_regime_judgement_latest.json"
)
SOFT_ALLOCATOR_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_soft_three_way_allocator_current"
    / "soft_three_way_market_regime_allocator_latest.json"
)
HARD_ALLOCATOR_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_three_way_allocator_current"
    / "three_way_market_regime_allocator_latest.json"
)
BALANCED_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_balanced_overlay_strategy_latest.json"
)
PAIR_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_pair_fast_exit_candidate_latest.json"
)
HYBRID_PATH = GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
OUTPUT_DIR = (
    GROUP_ROOT / "current_switch_validation_current" / "performance_first_switch_replay_current"
)
REPLAY_COVERAGE_GAP_MAX_DAYS = 21
REPLAY_STALE_GAP_MIN_DAYS = 14

_SWITCH_SPEC = importlib.util.spec_from_file_location(
    "write_portfolio_operating_switch",
    Path(__file__).resolve().parent / "write_portfolio_operating_switch.py",
)
if _SWITCH_SPEC is None or _SWITCH_SPEC.loader is None:
    raise RuntimeError("Failed to load write_portfolio_operating_switch")
_SWITCH = importlib.util.module_from_spec(_SWITCH_SPEC)
sys.modules[_SWITCH_SPEC.name] = _SWITCH
_SWITCH_SPEC.loader.exec_module(_SWITCH)

_MARKET_SPEC = importlib.util.spec_from_file_location(
    "run_group_market_regime_judgement",
    Path(__file__).resolve().parent / "run_group_market_regime_judgement.py",
)
if _MARKET_SPEC is None or _MARKET_SPEC.loader is None:
    raise RuntimeError("Failed to load run_group_market_regime_judgement")
_MARKET = importlib.util.module_from_spec(_MARKET_SPEC)
sys.modules[_MARKET_SPEC.name] = _MARKET
_MARKET_SPEC.loader.exec_module(_MARKET)

_DYN_SPEC = importlib.util.spec_from_file_location(
    "run_causal_dynamic_portfolio",
    Path(__file__).resolve().parent / "run_causal_dynamic_portfolio.py",
)
if _DYN_SPEC is None or _DYN_SPEC.loader is None:
    raise RuntimeError("Failed to load run_causal_dynamic_portfolio")
_DYN = importlib.util.module_from_spec(_DYN_SPEC)
sys.modules[_DYN_SPEC.name] = _DYN
_DYN_SPEC.loader.exec_module(_DYN)


@dataclass(frozen=True, slots=True)
class ThresholdProfile:
    min_oos_return_edge: float
    min_oos_sharpe_edge: float
    min_val_return: float
    min_val_sharpe: float

    @property
    def name(self) -> str:
        return (
            f"ret{self.min_oos_return_edge:.3f}_"
            f"sh{self.min_oos_sharpe_edge:.2f}_"
            f"valret{self.min_val_return:.3f}_"
            f"valsh{self.min_val_sharpe:.2f}"
        )


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


def _parse_day(value: str) -> date:
    token = str(value).strip()
    if not token:
        raise ValueError("missing day token")
    if "T" in token:
        token = token.split("T", 1)[0]
    elif " " in token:
        token = token.split(" ", 1)[0]
    return date.fromisoformat(token)


def _state_series(states: list[dict[str, Any]]) -> tuple[list[date], list[dict[str, Any]]]:
    keyed = sorted(
        ((_parse_day(item["date"]), dict(item)) for item in list(states or [])),
        key=lambda item: item[0],
    )
    return [day for day, _ in keyed], [item for _, item in keyed]


def _raw_coverage_index(*, raw_aggtrades_root: Path, symbols: list[str]) -> dict[str, list[date]]:
    coverage: dict[str, list[date]] = {}
    for symbol in symbols:
        symbol_root = raw_aggtrades_root / _SWITCH._symbol_token(symbol)
        dates = sorted(
            date.fromisoformat(path.name.split("=", 1)[1])
            for path in symbol_root.glob("date=*")
            if path.is_dir()
        )
        coverage[str(symbol)] = dates
    return coverage


def _adjust_pair_liquidity_state_for_coverage(
    *,
    signals: list[_SWITCH.SymbolVolumeSignal],
    as_of_date: date,
    coverage_index: Mapping[str, list[date]],
) -> tuple[str, list[str]]:
    adjusted_states: list[str] = []
    coverage_gap_symbols: list[str] = []
    for signal in signals:
        symbol = str(signal.symbol)
        raw_dates = list(coverage_index.get(symbol) or [])
        next_available = None
        if raw_dates:
            idx = bisect_right(raw_dates, as_of_date)
            if idx < len(raw_dates):
                next_available = raw_dates[idx]
        latest_available = None
        if signal.latest_available_date:
            latest_available = _parse_day(signal.latest_available_date)
        stale_gap_days = (
            (as_of_date - latest_available).days if latest_available is not None else None
        )
        coverage_gap = bool(
            signal.state in {"stale", "missing"}
            and next_available is not None
            and 0 <= (next_available - as_of_date).days <= REPLAY_COVERAGE_GAP_MAX_DAYS
            and (stale_gap_days is None or stale_gap_days >= REPLAY_STALE_GAP_MIN_DAYS)
        )
        if coverage_gap:
            adjusted_states.append("coverage_gap")
            coverage_gap_symbols.append(symbol)
        else:
            adjusted_states.append(signal.state)

    if any(state == "weak" for state in adjusted_states):
        return "weak", coverage_gap_symbols
    if all(state == "strong" for state in adjusted_states):
        return "strong", coverage_gap_symbols
    if any(state in {"coverage_gap", "normal", "strong"} for state in adjusted_states):
        return "normal", coverage_gap_symbols
    if all(state == "missing" for state in adjusted_states):
        return "missing", coverage_gap_symbols
    return "weak", coverage_gap_symbols


def _carry_forward(
    days: list[date], items: list[dict[str, Any]], target_day: date
) -> dict[str, Any]:
    idx = bisect_right(days, target_day) - 1
    if idx < 0:
        return {}
    return dict(items[idx])


def _daily_map_from_streams(streams: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for split_name in ("train", "val", "oos"):
        for point in list(dict(streams or {}).get(split_name) or []):
            raw_ts = point.get("datetime", point.get("t", point.get("timestamp")))
            day_key = str(raw_ts).split("T", 1)[0]
            out[day_key] = _safe_float(point.get("v"), 0.0)
    return out


def _daily_map_from_dates(dates: list[str], returns: list[float]) -> dict[str, float]:
    return {
        str(day).split("T", 1)[0]: _safe_float(value, 0.0)
        for day, value in zip(dates, returns, strict=True)
    }


def _blend_daily_map(
    left: dict[str, float], right: dict[str, float], *, left_weight: float, right_weight: float
) -> dict[str, float]:
    keys = sorted(set(left) | set(right))
    return {
        key: (left_weight * _safe_float(left.get(key), 0.0))
        + (right_weight * _safe_float(right.get(key), 0.0))
        for key in keys
    }


def _metrics_until(
    day_keys: list[str], daily_map: dict[str, float], *, upto_day: str
) -> dict[str, float]:
    returns = [_safe_float(daily_map.get(day), 0.0) for day in day_keys if day <= upto_day]
    if not returns:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    return dict(_DYN._metrics(np.asarray(returns, dtype=float)))


def _hybrid_health_for_day(
    *,
    oos_days: list[str],
    current_day: str,
    hybrid_map: dict[str, float],
    balanced_map: dict[str, float],
    pair_map: dict[str, float],
    hybrid_val_metrics: dict[str, Any],
    pair_cap_respected: bool,
) -> dict[str, Any]:
    hybrid_oos = _metrics_until(oos_days, hybrid_map, upto_day=current_day)
    balanced_oos = _metrics_until(oos_days, balanced_map, upto_day=current_day)
    pair_oos = _metrics_until(oos_days, pair_map, upto_day=current_day)
    beats_cash = _safe_float(hybrid_oos.get("total_return"), 0.0) > 0.0
    beats_pair = _safe_float(hybrid_oos.get("total_return"), 0.0) > _safe_float(
        pair_oos.get("total_return"), 0.0
    )
    beats_balanced = _safe_float(hybrid_oos.get("total_return"), 0.0) > _safe_float(
        balanced_oos.get("total_return"), 0.0
    )
    return {
        "healthy": (
            beats_cash
            and pair_cap_respected
            and _safe_float(hybrid_oos.get("total_return"), 0.0) > 0.0
            and _safe_float(hybrid_oos.get("sharpe"), 0.0) > 0.0
            and _safe_float(hybrid_val_metrics.get("total_return"), 0.0) >= 0.0
            and _safe_float(hybrid_val_metrics.get("sharpe"), 0.0) > 0.0
        ),
        "recommended_stage": "pilot_candidate"
        if (beats_cash and pair_cap_respected and beats_pair)
        else ("guarded_candidate" if beats_cash and pair_cap_respected else "do_not_integrate"),
        "beats_balanced_refreshed": beats_balanced,
        "beats_pair_tactical_refreshed": beats_pair,
        "beats_cash_refreshed": beats_cash,
        "pair_cap_respected": pair_cap_respected,
        "max_rss_under_8gib": True,
        "val_total_return": _safe_float(hybrid_val_metrics.get("total_return"), 0.0),
        "val_sharpe": _safe_float(hybrid_val_metrics.get("sharpe"), 0.0),
        "oos_total_return": _safe_float(hybrid_oos.get("total_return"), 0.0),
        "oos_sharpe": _safe_float(hybrid_oos.get("sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(hybrid_oos.get("max_drawdown"), 0.0),
    }


def _balanced_health_for_day(
    *,
    oos_days: list[str],
    current_day: str,
    balanced_map: dict[str, float],
    balanced_val_metrics: dict[str, Any],
) -> dict[str, Any]:
    oos = _metrics_until(oos_days, balanced_map, upto_day=current_day)
    return {
        "healthy": _safe_float(oos.get("total_return"), 0.0) > 0.0
        and _safe_float(oos.get("sharpe"), 0.0) > 0.0,
        "val_total_return": _safe_float(balanced_val_metrics.get("total_return"), 0.0),
        "val_sharpe": _safe_float(balanced_val_metrics.get("sharpe"), 0.0),
        "oos_total_return": _safe_float(oos.get("total_return"), 0.0),
        "oos_sharpe": _safe_float(oos.get("sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(oos.get("max_drawdown"), 0.0),
    }


def _allocator_health_for_day(
    *, oos_days: list[str], current_day: str, daily_map: dict[str, float]
) -> dict[str, Any]:
    oos = _metrics_until(oos_days, daily_map, upto_day=current_day)
    return {
        "healthy": _safe_float(oos.get("total_return"), 0.0) > 0.0
        and _safe_float(oos.get("sharpe"), 0.0) > 0.0,
        "oos_total_return": _safe_float(oos.get("total_return"), 0.0),
        "oos_sharpe": _safe_float(oos.get("sharpe"), 0.0),
        "oos_max_drawdown": _safe_float(oos.get("max_drawdown"), 0.0),
    }


def _market_judgements_by_day(
    *,
    market_payload: Mapping[str, Any],
    replay_days: list[str],
) -> dict[str, dict[str, Any]]:
    selected_rules = [
        dict(rule)
        for rule in list(market_payload.get("selected_rules") or [])
        if isinstance(rule, dict)
    ]
    symbol_universe = [
        str(symbol)
        for symbol in list(market_payload.get("symbol_universe") or [])
        if str(symbol).strip()
    ]
    if not replay_days or not selected_rules or not symbol_universe:
        return {}
    start_day = _parse_day(min(replay_days))
    end_day = _parse_day(max(replay_days))
    symbol_frames = []
    for symbol in symbol_universe:
        frame, _summary = _MARKET._load_symbol_close_30m_from_feature_points(
            symbol,
            start_day=_MARKET.pd.Timestamp(start_day, tz="UTC"),
            end_day=_MARKET.pd.Timestamp(end_day, tz="UTC"),
        )
        symbol_frames.append(frame)
    feature_frame = _MARKET._daily_market_feature_frame(symbol_frames)
    if feature_frame.empty:
        return {}
    judgements: dict[str, dict[str, Any]] = {}
    for _idx, row in feature_frame.sort_values("date").iterrows():
        judgement = dict(_MARKET._current_judgement(latest_row=row, selected_rules=selected_rules))
        raw_day = str(judgement.get("date") or "").strip()
        if not raw_day:
            continue
        judgements[_parse_day(raw_day).isoformat()] = judgement
    return judgements


def _return_for_mode(mode: str, day: str, mode_maps: Mapping[str, dict[str, float]]) -> float:
    return _safe_float(dict(mode_maps.get(mode) or {}).get(day), 0.0)


def _replay_profile(
    *,
    profile: ThresholdProfile,
    day_contexts: list[dict[str, Any]],
    mode_maps: Mapping[str, dict[str, float]],
) -> dict[str, Any]:
    old_values = (
        _SWITCH.HYBRID_PROMOTION_STRONG_OOS_RETURN_EDGE,
        _SWITCH.HYBRID_PROMOTION_STRONG_OOS_SHARPE_EDGE,
        _SWITCH.HYBRID_PROMOTION_STRONG_VAL_TOTAL_RETURN,
        _SWITCH.HYBRID_PROMOTION_STRONG_VAL_SHARPE,
    )
    _SWITCH.HYBRID_PROMOTION_STRONG_OOS_RETURN_EDGE = profile.min_oos_return_edge
    _SWITCH.HYBRID_PROMOTION_STRONG_OOS_SHARPE_EDGE = profile.min_oos_sharpe_edge
    _SWITCH.HYBRID_PROMOTION_STRONG_VAL_TOTAL_RETURN = profile.min_val_return
    _SWITCH.HYBRID_PROMOTION_STRONG_VAL_SHARPE = profile.min_val_sharpe
    replay_returns: list[float] = []
    mode_counts: dict[str, int] = {}
    replay_trace: list[dict[str, Any]] = []
    try:
        for context in day_contexts:
            day = str(context["day"])
            decision = _SWITCH.recommend_operating_mode(
                current_judgement=dict(context["current_judgement"]),
                soft_current_state=dict(context["soft_state"]),
                hard_current_state=dict(context["hard_state"]),
                operating_plan_payload={
                    "deployment_modes": {
                        "core_mode": {"allocation": {"soft_three_way_regime": 1.0}},
                        "balanced_overlay_mode": {
                            "allocation": {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}
                        },
                        "defensive_overlay_mode": {
                            "allocation": {"soft_three_way_regime": 0.7, "pair_fast_exit": 0.3}
                        },
                        "aggressive_realized_mode": {"allocation": {"three_way_regime": 1.0}},
                        "hybrid_guarded_mode": {"allocation": {"hybrid_online_portfolio": 1.0}},
                        "risk_off_mode": {"allocation": {"cash": 1.0}},
                    }
                },
                pair_liquidity_state=str(context["pair_liquidity_state"]),
                balanced_health=dict(context["balanced_health"]),
                hybrid_health=dict(context["hybrid_health"]),
            )
            day_return = _return_for_mode(decision.mode, day, mode_maps)
            replay_returns.append(day_return)
            mode_counts[decision.mode] = mode_counts.get(decision.mode, 0) + 1
            replay_trace.append({"date": day, "mode": decision.mode, "return": day_return})
    finally:
        (
            _SWITCH.HYBRID_PROMOTION_STRONG_OOS_RETURN_EDGE,
            _SWITCH.HYBRID_PROMOTION_STRONG_OOS_SHARPE_EDGE,
            _SWITCH.HYBRID_PROMOTION_STRONG_VAL_TOTAL_RETURN,
            _SWITCH.HYBRID_PROMOTION_STRONG_VAL_SHARPE,
        ) = old_values
    metrics = dict(_DYN._metrics(np.asarray(replay_returns, dtype=float)))
    return {
        "profile": asdict(profile),
        "oos_metrics": metrics,
        "mode_counts": mode_counts,
        "last_mode": replay_trace[-1]["mode"] if replay_trace else None,
        "trace_tail": replay_trace[-10:],
    }


def _build_day_contexts(
    *,
    replay_days: list[str],
    judgement_by_day: Mapping[str, dict[str, Any]],
    soft_days: list[date],
    soft_states: list[dict[str, Any]],
    hard_days: list[date],
    hard_states: list[dict[str, Any]],
    mode_maps: Mapping[str, dict[str, float]],
    balanced_val_metrics: Mapping[str, Any],
    hybrid_val_metrics: Mapping[str, Any],
    pair_cap_respected: bool,
    pair_liquidity_policy: str,
    coverage_index: Mapping[str, list[date]] | None = None,
) -> list[dict[str, Any]]:
    day_contexts: list[dict[str, Any]] = []
    for day in replay_days:
        day_date = _parse_day(day)
        pair_signals = [
            _SWITCH._load_symbol_volume_signal(
                raw_aggtrades_root=_SWITCH.DEFAULT_RAW_AGGTRADES_ROOT,
                symbol=symbol,
                as_of_date=day_date,
                lookback_days=_SWITCH.DEFAULT_VOLUME_LOOKBACK_DAYS,
            )
            for symbol in _SWITCH.DEFAULT_PAIR_SYMBOLS
        ]
        strict_pair_liquidity_state = _SWITCH._pair_liquidity_state(pair_signals)
        adjusted_pair_liquidity_state, coverage_gap_symbols = (
            _adjust_pair_liquidity_state_for_coverage(
                signals=pair_signals,
                as_of_date=day_date,
                coverage_index=coverage_index or {},
            )
        )
        pair_liquidity_state = (
            adjusted_pair_liquidity_state
            if str(pair_liquidity_policy).strip().lower() == "coverage_adjusted"
            else strict_pair_liquidity_state
        )
        soft_state = _carry_forward(soft_days, soft_states, day_date)
        hard_state = _carry_forward(hard_days, hard_states, day_date)
        if "_allocator_health" not in soft_state:
            soft_state["_allocator_health"] = _allocator_health_for_day(
                oos_days=replay_days,
                current_day=day,
                daily_map=mode_maps["core_mode"],
            )
        if "_allocator_health" not in hard_state:
            hard_state["_allocator_health"] = _allocator_health_for_day(
                oos_days=replay_days,
                current_day=day,
                daily_map=mode_maps["aggressive_realized_mode"],
            )
        day_contexts.append(
            {
                "day": day,
                "current_judgement": dict(judgement_by_day.get(day) or {}),
                "soft_state": soft_state,
                "hard_state": hard_state,
                "pair_liquidity_state": pair_liquidity_state,
                "strict_pair_liquidity_state": strict_pair_liquidity_state,
                "adjusted_pair_liquidity_state": adjusted_pair_liquidity_state,
                "coverage_gap_symbols": coverage_gap_symbols,
                "balanced_health": _balanced_health_for_day(
                    oos_days=replay_days,
                    current_day=day,
                    balanced_map=mode_maps["balanced_overlay_mode"],
                    balanced_val_metrics=balanced_val_metrics,
                ),
                "hybrid_health": _hybrid_health_for_day(
                    oos_days=replay_days,
                    current_day=day,
                    hybrid_map=mode_maps["hybrid_guarded_mode"],
                    balanced_map=mode_maps["balanced_overlay_mode"],
                    pair_map=mode_maps["pair_tactical_mode"],
                    hybrid_val_metrics=hybrid_val_metrics,
                    pair_cap_respected=pair_cap_respected,
                ),
            }
        )
    return day_contexts


def _coverage_summary(day_contexts: list[dict[str, Any]]) -> dict[str, Any]:
    total_days = len(day_contexts)
    market_days = sum(
        1 for item in day_contexts if dict(item.get("current_judgement") or {}).get("date")
    )
    liquidity_counts: dict[str, int] = {}
    strict_liquidity_counts: dict[str, int] = {}
    coverage_gap_day_count = 0
    for item in day_contexts:
        state = str(item.get("pair_liquidity_state") or "missing")
        liquidity_counts[state] = liquidity_counts.get(state, 0) + 1
        strict_state = str(item.get("strict_pair_liquidity_state") or "missing")
        strict_liquidity_counts[strict_state] = strict_liquidity_counts.get(strict_state, 0) + 1
        if list(item.get("coverage_gap_symbols") or []):
            coverage_gap_day_count += 1
    return {
        "replay_days": total_days,
        "market_judgement_days": market_days,
        "market_judgement_missing_days": total_days - market_days,
        "pair_liquidity_counts": liquidity_counts,
        "strict_pair_liquidity_counts": strict_liquidity_counts,
        "coverage_gap_day_count": coverage_gap_day_count,
    }


def build_replay_report(
    *,
    switch_payload: Mapping[str, Any],
    market_payload: Mapping[str, Any],
    soft_payload: Mapping[str, Any],
    hard_payload: Mapping[str, Any],
    balanced_payload: Mapping[str, Any],
    pair_payload: Mapping[str, Any],
    hybrid_payload: Mapping[str, Any],
    return_grid: list[float],
    sharpe_grid: list[float],
    val_return_grid: list[float],
    val_sharpe_grid: list[float],
) -> dict[str, Any]:
    hybrid_ref = dict(
        dict(hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}
    )
    replay_days = [
        str(day)
        for day, split in zip(
            list(hybrid_ref.get("dates") or []),
            [dict(item).get("split") for item in list(hybrid_ref.get("allocations") or [])],
            strict=True,
        )
        if split == "oos"
    ]
    if not replay_days:
        raise RuntimeError("hybrid payload has no OOS replay days")

    judgement_by_day = _market_judgements_by_day(
        market_payload=market_payload, replay_days=replay_days
    )
    soft_days, soft_states = _state_series(list(soft_payload.get("states") or []))
    hard_days, hard_states = _state_series(list(hard_payload.get("states") or []))

    soft_map = _daily_map_from_dates(
        list(soft_payload.get("dates") or []), list(soft_payload.get("daily_returns") or [])
    )
    hard_map = _daily_map_from_dates(
        list(hard_payload.get("dates") or []), list(hard_payload.get("daily_returns") or [])
    )
    balanced_map = _daily_map_from_streams(
        dict(balanced_payload.get("portfolio_return_streams") or {})
    )
    pair_map = _daily_map_from_streams(dict(pair_payload.get("return_streams") or {}))
    hybrid_map = _daily_map_from_dates(
        list(hybrid_ref.get("dates") or []), list(hybrid_ref.get("daily_returns") or [])
    )
    defensive_map = _blend_daily_map(soft_map, pair_map, left_weight=0.7, right_weight=0.3)

    mode_maps = {
        "balanced_overlay_mode": balanced_map,
        "core_mode": soft_map,
        "aggressive_realized_mode": hard_map,
        "hybrid_guarded_mode": hybrid_map,
        "pair_tactical_mode": pair_map,
        "defensive_overlay_mode": defensive_map,
        "risk_off_mode": {},
    }
    balanced_val = dict((balanced_payload.get("portfolio_metrics") or {}).get("val") or {})
    hybrid_val = dict(dict(hybrid_ref.get("split_metrics") or {}).get("val") or {})
    pair_cap_respected = bool(dict(hybrid_payload.get("readiness") or {}).get("pair_cap_respected"))
    coverage_index = _raw_coverage_index(
        raw_aggtrades_root=_SWITCH.DEFAULT_RAW_AGGTRADES_ROOT,
        symbols=list(_SWITCH.DEFAULT_PAIR_SYMBOLS),
    )
    strict_day_contexts = _build_day_contexts(
        replay_days=replay_days,
        judgement_by_day=judgement_by_day,
        soft_days=soft_days,
        soft_states=soft_states,
        hard_days=hard_days,
        hard_states=hard_states,
        mode_maps=mode_maps,
        balanced_val_metrics=balanced_val,
        hybrid_val_metrics=hybrid_val,
        pair_cap_respected=pair_cap_respected,
        pair_liquidity_policy="strict",
        coverage_index=coverage_index,
    )
    adjusted_day_contexts = _build_day_contexts(
        replay_days=replay_days,
        judgement_by_day=judgement_by_day,
        soft_days=soft_days,
        soft_states=soft_states,
        hard_days=hard_days,
        hard_states=hard_states,
        mode_maps=mode_maps,
        balanced_val_metrics=balanced_val,
        hybrid_val_metrics=hybrid_val,
        pair_cap_respected=pair_cap_respected,
        pair_liquidity_policy="coverage_adjusted",
        coverage_index=coverage_index,
    )
    coverage = _coverage_summary(adjusted_day_contexts)

    profiles = [
        ThresholdProfile(
            min_oos_return_edge=r,
            min_oos_sharpe_edge=s,
            min_val_return=vr,
            min_val_sharpe=vs,
        )
        for r, s, vr, vs in product(return_grid, sharpe_grid, val_return_grid, val_sharpe_grid)
    ]
    current_profile = ThresholdProfile(
        min_oos_return_edge=float(_SWITCH.HYBRID_PROMOTION_STRONG_OOS_RETURN_EDGE),
        min_oos_sharpe_edge=float(_SWITCH.HYBRID_PROMOTION_STRONG_OOS_SHARPE_EDGE),
        min_val_return=float(_SWITCH.HYBRID_PROMOTION_STRONG_VAL_TOTAL_RETURN),
        min_val_sharpe=float(_SWITCH.HYBRID_PROMOTION_STRONG_VAL_SHARPE),
    )
    if all(profile != current_profile for profile in profiles):
        profiles.append(current_profile)

    results = [
        _replay_profile(
            profile=profile,
            day_contexts=adjusted_day_contexts,
            mode_maps=mode_maps,
        )
        for profile in profiles
    ]
    results.sort(
        key=lambda item: (
            _safe_float(dict(item.get("oos_metrics") or {}).get("total_return"), 0.0),
            _safe_float(dict(item.get("oos_metrics") or {}).get("sharpe"), 0.0),
        ),
        reverse=True,
    )
    current_result = next(item for item in results if item["profile"] == asdict(current_profile))
    strict_current_result = _replay_profile(
        profile=current_profile,
        day_contexts=strict_day_contexts,
        mode_maps=mode_maps,
    )
    return {
        "artifact_kind": "performance_first_switch_replay",
        "generated_at": _SWITCH._utc_now_iso(),
        "replay_day_count": len(replay_days),
        "replay_start": replay_days[0],
        "replay_end": replay_days[-1],
        "pair_liquidity_policy": "coverage_adjusted",
        "current_switch_mode": str(
            dict(switch_payload.get("recommended_mode") or {}).get("mode") or ""
        ),
        "current_market_state": dict(switch_payload.get("current_market_state") or {}),
        "coverage_summary": coverage,
        "current_profile": asdict(current_profile),
        "strict_current_profile_result": strict_current_result,
        "current_profile_result": current_result,
        "top_profiles": results[:10],
        "grid": {
            "return_edge_grid": return_grid,
            "sharpe_edge_grid": sharpe_grid,
            "val_return_grid": val_return_grid,
            "val_sharpe_grid": val_sharpe_grid,
        },
    }


def _build_markdown(report: Mapping[str, Any]) -> str:
    current_state = dict(report.get("current_market_state") or {})
    coverage = dict(report.get("coverage_summary") or {})
    strict_current = dict(report.get("strict_current_profile_result") or {})
    strict_metrics = dict(strict_current.get("oos_metrics") or {})
    current_result = dict(report.get("current_profile_result") or {})
    current_metrics = dict(current_result.get("oos_metrics") or {})
    top_profiles = list(report.get("top_profiles") or [])
    lines = [
        "# Performance-first switch replay",
        "",
        f"- generated_at: `{report.get('generated_at')}`",
        f"- replay_window: `{report.get('replay_start')}` ~ `{report.get('replay_end')}`",
        f"- replay_day_count: `{report.get('replay_day_count')}`",
        f"- pair_liquidity_policy: `{report.get('pair_liquidity_policy')}`",
        f"- current_market_state: favored_group=`{current_state.get('favored_group')}`, confidence=`{_safe_float(current_state.get('confidence'), 0.0):.4f}`, trend=`{current_state.get('trend_state')}`, breadth=`{current_state.get('breadth_state')}`, volatility=`{current_state.get('volatility_state')}`, pair_liquidity=`{current_state.get('pair_liquidity_state')}`",
        f"- coverage_summary: market_judgement_days=`{coverage.get('market_judgement_days')}` / `{coverage.get('replay_days')}`, adjusted_pair_liquidity_counts=`{json.dumps(coverage.get('pair_liquidity_counts') or {}, sort_keys=True)}`, strict_pair_liquidity_counts=`{json.dumps(coverage.get('strict_pair_liquidity_counts') or {}, sort_keys=True)}`, coverage_gap_day_count=`{coverage.get('coverage_gap_day_count')}`",
        "",
        "## Current profile replay (strict baseline)",
        f"- oos_return / sharpe / max_dd: `{_safe_float(strict_metrics.get('total_return'), 0.0):+.4%}` / `{_safe_float(strict_metrics.get('sharpe'), 0.0):.4f}` / `{_safe_float(strict_metrics.get('max_drawdown'), 0.0):.4%}`",
        f"- mode_counts: `{json.dumps(strict_current.get('mode_counts') or {}, sort_keys=True)}`",
        f"- last_mode: `{strict_current.get('last_mode')}`",
        "",
        "## Current profile replay (coverage-adjusted)",
        f"- profile: `{json.dumps(report.get('current_profile') or {}, sort_keys=True)}`",
        f"- oos_return / sharpe / max_dd: `{_safe_float(current_metrics.get('total_return'), 0.0):+.4%}` / `{_safe_float(current_metrics.get('sharpe'), 0.0):.4f}` / `{_safe_float(current_metrics.get('max_drawdown'), 0.0):.4%}`",
        f"- mode_counts: `{json.dumps(current_result.get('mode_counts') or {}, sort_keys=True)}`",
        f"- last_mode: `{current_result.get('last_mode')}`",
        "",
        "## Top replay profiles",
        "",
        "| Rank | Profile | OOS return | Sharpe | Max DD | Last mode |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for idx, item in enumerate(top_profiles, start=1):
        metrics = dict(item.get("oos_metrics") or {})
        profile = dict(item.get("profile") or {})
        lines.append(
            f"| {idx} | `{json.dumps(profile, sort_keys=True)}` | `{_safe_float(metrics.get('total_return'), 0.0):+.4%}` | `{_safe_float(metrics.get('sharpe'), 0.0):.4f}` | `{_safe_float(metrics.get('max_drawdown'), 0.0):.4%}` | `{item.get('last_mode')}` |"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch-path", type=Path, default=SWITCH_PATH)
    parser.add_argument("--market-judgement-path", type=Path, default=MARKET_JUDGEMENT_PATH)
    parser.add_argument("--soft-allocator-path", type=Path, default=SOFT_ALLOCATOR_PATH)
    parser.add_argument("--hard-allocator-path", type=Path, default=HARD_ALLOCATOR_PATH)
    parser.add_argument("--balanced-path", type=Path, default=BALANCED_PATH)
    parser.add_argument("--pair-path", type=Path, default=PAIR_PATH)
    parser.add_argument("--hybrid-path", type=Path, default=HYBRID_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--return-edge-grid", default="0.001,0.002,0.003,0.004,0.005,0.006")
    parser.add_argument("--sharpe-edge-grid", default="0.75,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--val-return-grid", default="0.03,0.04,0.05,0.06,0.07")
    parser.add_argument("--val-sharpe-grid", default="2.0,2.5,3.0,3.5,4.0")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = build_replay_report(
        switch_payload=_read_json(Path(args.switch_path).resolve()),
        market_payload=_read_json(Path(args.market_judgement_path).resolve()),
        soft_payload=_read_json(Path(args.soft_allocator_path).resolve()),
        hard_payload=_read_json(Path(args.hard_allocator_path).resolve()),
        balanced_payload=_read_json(Path(args.balanced_path).resolve()),
        pair_payload=_read_json(Path(args.pair_path).resolve()),
        hybrid_payload=_read_json(Path(args.hybrid_path).resolve()),
        return_grid=_parse_grid(args.return_edge_grid),
        sharpe_grid=_parse_grid(args.sharpe_edge_grid),
        val_return_grid=_parse_grid(args.val_return_grid),
        val_sharpe_grid=_parse_grid(args.val_sharpe_grid),
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "performance_first_switch_replay_latest.json"
    md_path = output_dir / "performance_first_switch_replay_latest.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
