"""Summarize whether the current hybrid live lane is deployable with real capital.

This is a lightweight artifact-only analysis. It does not rerun any heavy
research jobs. Instead it combines:
- the latest current switch recommendation
- the coverage-adjusted switch replay
- the hybrid sleeve allocation history
- the current runtime live configuration
- refresh / readiness artifacts when present

The goal is to separate:
1) "performance looks strong on current artifacts"
2) "paper/live operational prerequisites are actually satisfied"
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.configuration.loader import load_runtime_config
from lumina_quant.live_selection import infer_strategy_class_name, supports_live_portfolio_mode
from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT, ROOT

GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
DEFAULT_SWITCH_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_operating_switch_current"
    / "portfolio_operating_switch_latest.json"
)
DEFAULT_REPLAY_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "performance_first_switch_replay_current"
    / "performance_first_switch_replay_latest.json"
)
DEFAULT_HYBRID_PATH = (
    GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
)
DEFAULT_REFRESH_PATH = FOLLOWUP_ROOT / "final_portfolio_validation_data_refresh_latest.json"
DEFAULT_DECISION_PATH = FOLLOWUP_ROOT / "portfolio_live_readiness_decision_latest.json"
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "hybrid_live_investability_current"
DEFAULT_COST_BPS_GRID = (5.0, 10.0, 20.0, 30.0)


@dataclass(frozen=True, slots=True)
class CostStressScenario:
    one_way_cost_bps: float
    total_cost_drag: float
    adjusted_total_return: float
    adjusted_sharpe: float
    adjusted_max_drawdown: float


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except TypeError, ValueError:
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return asdict(value)
    except TypeError:
        return dict(getattr(value, "__dict__", {}) or {})


def _decision_allows_live_start(decision: Mapping[str, Any]) -> tuple[bool, bool, str]:
    decision_value = str(decision.get("decision", "") or "").strip().lower()
    reference = str(
        decision.get("selected_mode")
        or decision.get("candidate_mode")
        or decision.get("candidate_key")
        or ""
    ).strip()
    allowed = decision_value == "keep_incumbent" or (
        decision_value in {"promote_candidate", "selected_live_mode"} and bool(reference)
    )
    return bool(allowed), decision_value == "keep_incumbent", reference


def _decision_runtime_compatible(
    *, decision_allowed: bool, decision_keep: bool, reference: str
) -> bool:
    if not decision_allowed:
        return False
    if decision_keep:
        return True
    return bool(infer_strategy_class_name(reference) or supports_live_portfolio_mode(reference))


def _parse_utc(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(UTC)


def _daily_metrics(
    returns: Sequence[float], *, periods_per_year: float = 365.0
) -> dict[str, float]:
    series = [float(item) for item in returns]
    if not series:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "volatility": 0.0}

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in series:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0.0:
            max_drawdown = max(max_drawdown, 1.0 - (equity / peak))

    total_return = equity - 1.0
    mean_return = sum(series) / len(series)
    variance = sum((item - mean_return) ** 2 for item in series) / len(series)
    volatility_daily = math.sqrt(max(variance, 0.0))
    sharpe = 0.0
    if volatility_daily > 0.0:
        sharpe = (mean_return / volatility_daily) * math.sqrt(periods_per_year)

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "volatility": float(volatility_daily * math.sqrt(periods_per_year)),
    }


def _scenario_payloads(
    hybrid_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    scenario = dict(dict(hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {})
    allocations = [
        dict(item) for item in list(scenario.get("allocations") or []) if isinstance(item, Mapping)
    ]
    daily_returns = [_safe_float(item, 0.0) for item in list(scenario.get("daily_returns") or [])]
    if allocations and len(allocations) != len(daily_returns):
        raise ValueError("hybrid allocations and daily_returns length mismatch")

    turnover_by_index: list[float] = []
    previous_weights: dict[str, float] = {}
    for allocation in allocations:
        current_weights = {
            str(key): _safe_float(value, 0.0)
            for key, value in dict(allocation.get("weights") or {}).items()
        }
        keys = set(previous_weights) | set(current_weights)
        turnover = 0.5 * sum(
            abs(
                _safe_float(current_weights.get(key), 0.0)
                - _safe_float(previous_weights.get(key), 0.0)
            )
            for key in keys
        )
        turnover_by_index.append(float(turnover))
        previous_weights = current_weights

    oos_allocations: list[dict[str, Any]] = []
    oos_returns: list[float] = []
    oos_turnovers: list[float] = []
    for allocation, ret, turnover in zip(
        allocations, daily_returns, turnover_by_index, strict=True
    ):
        if str(allocation.get("split") or "").strip().lower() != "oos":
            continue
        oos_allocations.append(allocation)
        oos_returns.append(float(ret))
        oos_turnovers.append(float(turnover))
    return oos_allocations, oos_returns, oos_turnovers


def _allocation_summary(hybrid_payload: Mapping[str, Any]) -> dict[str, Any]:
    oos_allocations, _oos_returns, oos_turnovers = _scenario_payloads(hybrid_payload)
    if not oos_allocations:
        return {
            "oos_days": 0,
            "default_switches": 0,
            "default_switch_rate": 0.0,
            "default_counts": {},
            "avg_daily_turnover_proxy": 0.0,
            "median_daily_turnover_proxy": 0.0,
            "p90_daily_turnover_proxy": 0.0,
            "pair_active_days": 0,
            "pair_active_ratio": 0.0,
            "avg_pair_weight": 0.0,
            "max_pair_weight": 0.0,
            "avg_cash_weight": 0.0,
            "max_cash_weight": 0.0,
            "max_non_cash_sleeve_weight": 0.0,
        }

    default_counts: dict[str, int] = {}
    default_switches = 0
    previous_default: str | None = None
    pair_weights: list[float] = []
    cash_weights: list[float] = []
    max_non_cash_sleeve_weight = 0.0
    for allocation in oos_allocations:
        default_sleeve = str(allocation.get("default_sleeve") or "")
        default_counts[default_sleeve] = default_counts.get(default_sleeve, 0) + 1
        if previous_default is not None and default_sleeve != previous_default:
            default_switches += 1
        previous_default = default_sleeve

        weights = {
            str(key): _safe_float(value, 0.0)
            for key, value in dict(allocation.get("weights") or {}).items()
        }
        pair_weight = _safe_float(weights.get("pair_tactical_mode"), 0.0)
        cash_weight = _safe_float(allocation.get("cash_weight"), 0.0)
        pair_weights.append(pair_weight)
        cash_weights.append(cash_weight)
        if weights:
            max_non_cash_sleeve_weight = max(max_non_cash_sleeve_weight, max(weights.values()))

    ordered_turnovers = sorted(float(item) for item in oos_turnovers)
    p90_index = int((len(ordered_turnovers) - 1) * 0.90) if ordered_turnovers else 0
    return {
        "oos_days": len(oos_allocations),
        "default_switches": default_switches,
        "default_switch_rate": float(default_switches / max(1, len(oos_allocations) - 1)),
        "default_counts": default_counts,
        "avg_daily_turnover_proxy": float(sum(oos_turnovers) / len(oos_turnovers)),
        "median_daily_turnover_proxy": float(ordered_turnovers[len(ordered_turnovers) // 2]),
        "p90_daily_turnover_proxy": float(ordered_turnovers[p90_index]),
        "pair_active_days": int(sum(1 for item in pair_weights if item > 0.0)),
        "pair_active_ratio": float(
            sum(1 for item in pair_weights if item > 0.0) / len(pair_weights)
        ),
        "avg_pair_weight": float(sum(pair_weights) / len(pair_weights)),
        "max_pair_weight": float(max(pair_weights)),
        "avg_cash_weight": float(sum(cash_weights) / len(cash_weights)),
        "max_cash_weight": float(max(cash_weights)),
        "max_non_cash_sleeve_weight": float(max_non_cash_sleeve_weight),
    }


def _cost_stress(
    hybrid_payload: Mapping[str, Any], *, one_way_cost_bps_grid: Sequence[float]
) -> list[dict[str, Any]]:
    _oos_allocations, oos_returns, oos_turnovers = _scenario_payloads(hybrid_payload)
    scenarios: list[dict[str, Any]] = []
    for cost_bps in one_way_cost_bps_grid:
        drag_series = [turnover * (float(cost_bps) / 10_000.0) for turnover in oos_turnovers]
        adjusted_returns = [ret - drag for ret, drag in zip(oos_returns, drag_series, strict=True)]
        metrics = _daily_metrics(adjusted_returns)
        scenario = CostStressScenario(
            one_way_cost_bps=float(cost_bps),
            total_cost_drag=float(sum(drag_series)),
            adjusted_total_return=float(metrics["total_return"]),
            adjusted_sharpe=float(metrics["sharpe"]),
            adjusted_max_drawdown=float(metrics["max_drawdown"]),
        )
        scenarios.append(asdict(scenario))
    return scenarios


def _live_readiness_summary(
    *,
    config_path: Path,
    refresh_path: Path,
    decision_path: Path,
) -> dict[str, Any]:
    runtime = load_runtime_config(config_path=str(config_path))
    live = _to_plain_dict(runtime.live)
    refresh_exists = refresh_path.exists()
    decision_exists = decision_path.exists()
    refresh = _read_json(refresh_path) if refresh_exists else {}
    decision = _read_json(decision_path) if decision_exists else {}
    refresh_cutoff = _parse_utc(refresh.get("collection_cutoff_utc"))
    stale_minutes = None
    if refresh_cutoff is not None:
        stale_minutes = (datetime.now(UTC) - refresh_cutoff).total_seconds() / 60.0

    mode = str(live.get("mode") or "").strip().lower()
    testnet = bool(live.get("testnet"))
    require_real_enable_flag = bool(live.get("require_real_enable_flag"))
    env_real_enabled = str(
        __import__("os").environ.get("LUMINA_ENABLE_LIVE_REAL", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    refresh_completed = str(refresh.get("status") or "").strip().lower() == "completed"
    decision_allowed, decision_keep, decision_reference = _decision_allows_live_start(decision)
    decision_runtime_compatible = _decision_runtime_compatible(
        decision_allowed=decision_allowed,
        decision_keep=decision_keep,
        reference=decision_reference,
    )
    stale = bool(stale_minutes is None or stale_minutes > 30.0)

    gaps: list[str] = []
    if mode != "paper":
        gaps.append("config_not_in_paper_mode")
    if not testnet:
        gaps.append("testnet_disabled_for_safe_paper_run")
    if not refresh_exists:
        gaps.append("refresh_artifact_missing")
    elif not refresh_completed:
        gaps.append("refresh_not_completed")
    if stale:
        gaps.append("refresh_stale_for_live_preflight")
    if not decision_exists:
        gaps.append("decision_artifact_missing")
    elif not decision_allowed:
        gaps.append("decision_not_live_start_compatible")
    elif not decision_runtime_compatible:
        gaps.append("portfolio_mode_executor_missing")
    if not live.get("startup_reconciliation_hard_fail", False):
        gaps.append("startup_reconciliation_hard_fail_disabled")
    if mode == "real" and require_real_enable_flag and not env_real_enabled:
        gaps.append("real_enable_env_missing")

    return {
        "config_mode": mode,
        "market_data_source": live.get("market_data_source"),
        "order_state_source": live.get("order_state_source"),
        "exchange_driver": _to_plain_dict(live.get("exchange") or {}).get("driver"),
        "exchange_name": _to_plain_dict(live.get("exchange") or {}).get("name"),
        "exchange_market_type": _to_plain_dict(live.get("exchange") or {}).get("market_type"),
        "testnet": testnet,
        "require_real_enable_flag": require_real_enable_flag,
        "env_real_enabled": env_real_enabled,
        "startup_reconciliation_hard_fail": bool(
            live.get("startup_reconciliation_hard_fail", False)
        ),
        "refresh_artifact_exists": refresh_exists,
        "refresh_status": refresh.get("status"),
        "refresh_cutoff_utc": refresh.get("collection_cutoff_utc"),
        "refresh_stale_minutes": stale_minutes,
        "decision_artifact_exists": decision_exists,
        "decision": decision.get("decision"),
        "decision_reference": decision_reference,
        "decision_allows_live_start": decision_allowed,
        "decision_keep_incumbent": decision_keep,
        "decision_runtime_compatible": decision_runtime_compatible,
        "paper_ready_now": not any(
            gap
            for gap in gaps
            if gap
            in {
                "config_not_in_paper_mode",
                "testnet_disabled_for_safe_paper_run",
                "refresh_artifact_missing",
                "refresh_not_completed",
                "refresh_stale_for_live_preflight",
                "decision_artifact_missing",
                "decision_not_live_start_compatible",
                "portfolio_mode_executor_missing",
            }
        ),
        "real_ready_now": (
            mode == "real"
            and not testnet
            and refresh_exists
            and refresh_completed
            and not stale
            and decision_exists
            and decision_allowed
            and decision_runtime_compatible
            and require_real_enable_flag
            and env_real_enabled
            and bool(live.get("startup_reconciliation_hard_fail", False))
        ),
        "gaps": gaps,
    }


def build_report(
    *,
    switch_payload: Mapping[str, Any],
    replay_payload: Mapping[str, Any],
    hybrid_payload: Mapping[str, Any],
    config_path: Path,
    refresh_path: Path,
    decision_path: Path,
    one_way_cost_bps_grid: Sequence[float] = DEFAULT_COST_BPS_GRID,
) -> dict[str, Any]:
    allocation_summary = _allocation_summary(hybrid_payload)
    cost_stress = _cost_stress(hybrid_payload, one_way_cost_bps_grid=one_way_cost_bps_grid)
    live_readiness = _live_readiness_summary(
        config_path=config_path,
        refresh_path=refresh_path,
        decision_path=decision_path,
    )
    hybrid_split_metrics = dict(
        dict(
            (dict(hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
                "split_metrics"
            )
            or {}
        ).get("oos")
        or {}
    )
    current_replay = dict(replay_payload.get("current_profile_result") or {})
    strict_replay = dict(replay_payload.get("strict_current_profile_result") or {})

    structure_ready = bool(
        allocation_summary.get("max_pair_weight", 0.0)
        <= _safe_float(dict(hybrid_payload.get("config") or {}).get("pair_weight_cap"), 1.0) + 1e-12
        and allocation_summary.get("avg_daily_turnover_proxy", 0.0) <= 0.15
        and _safe_float(hybrid_split_metrics.get("total_return"), 0.0) > 0.0
        and _safe_float(hybrid_split_metrics.get("sharpe"), 0.0) > 0.0
    )
    operationally_blocked = not bool(live_readiness.get("paper_ready_now")) or not bool(
        live_readiness.get("real_ready_now")
    )
    verdict = (
        "research_ready_but_not_real_ready"
        if structure_ready and operationally_blocked
        else "not_ready_for_live_capital"
    )
    if structure_ready and bool(live_readiness.get("real_ready_now")):
        verdict = "operationally_ready_for_real_mode"

    return {
        "artifact_kind": "hybrid_live_investability",
        "generated_at": _utc_now_iso(),
        "current_switch": {
            "mode": dict(switch_payload.get("recommended_mode") or {}).get("mode"),
            "allocation": dict(switch_payload.get("recommended_mode") or {}).get("allocation")
            or {},
            "market_state": dict(switch_payload.get("current_market_state") or {}),
        },
        "switch_replay": {
            "coverage_summary": dict(replay_payload.get("coverage_summary") or {}),
            "current_profile": dict(replay_payload.get("current_profile") or {}),
            "coverage_adjusted": {
                "oos_metrics": dict(current_replay.get("oos_metrics") or {}),
                "mode_counts": dict(current_replay.get("mode_counts") or {}),
                "last_mode": current_replay.get("last_mode"),
            },
            "strict": {
                "oos_metrics": dict(strict_replay.get("oos_metrics") or {}),
                "mode_counts": dict(strict_replay.get("mode_counts") or {}),
                "last_mode": strict_replay.get("last_mode"),
            },
        },
        "hybrid_oos": {
            "split_metrics": hybrid_split_metrics,
            "allocation_summary": allocation_summary,
            "pair_weight_cap_config": _safe_float(
                dict(hybrid_payload.get("config") or {}).get("pair_weight_cap"), 0.0
            ),
            "cash_buffer_observed_max": allocation_summary.get("max_cash_weight"),
            "cost_stress": cost_stress,
        },
        "live_readiness": live_readiness,
        "investability_verdict": {
            "strategy_structure_ready": structure_ready,
            "paper_ready_now": bool(live_readiness.get("paper_ready_now")),
            "real_ready_now": bool(live_readiness.get("real_ready_now")),
            "verdict": verdict,
        },
    }


def _build_markdown(report: Mapping[str, Any]) -> str:
    current_switch = dict(report.get("current_switch") or {})
    market_state = dict(current_switch.get("market_state") or {})
    replay = dict(report.get("switch_replay") or {})
    coverage_adjusted = dict(replay.get("coverage_adjusted") or {})
    coverage_metrics = dict(coverage_adjusted.get("oos_metrics") or {})
    strict = dict(replay.get("strict") or {})
    strict_metrics = dict(strict.get("oos_metrics") or {})
    hybrid_oos = dict(report.get("hybrid_oos") or {})
    split_metrics = dict(hybrid_oos.get("split_metrics") or {})
    allocation_summary = dict(hybrid_oos.get("allocation_summary") or {})
    readiness = dict(report.get("live_readiness") or {})
    verdict = dict(report.get("investability_verdict") or {})
    cost_stress = list(hybrid_oos.get("cost_stress") or [])

    lines = [
        "# Hybrid live investability report",
        "",
        f"- generated_at: `{report.get('generated_at')}`",
        f"- current switch mode: `{current_switch.get('mode')}`",
        f"- market_state: favored_group=`{market_state.get('favored_group')}`, confidence=`{_safe_float(market_state.get('confidence'), 0.0):.4f}`, trend=`{market_state.get('trend_state')}`, breadth=`{market_state.get('breadth_state')}`, volatility=`{market_state.get('volatility_state')}`, pair_liquidity=`{market_state.get('pair_liquidity_state')}`",
        "",
        "## Performance snapshot",
        f"- current hybrid OOS return / sharpe / max_dd: `{_safe_float(split_metrics.get('total_return'), 0.0):+.4%}` / `{_safe_float(split_metrics.get('sharpe'), 0.0):.4f}` / `{_safe_float(split_metrics.get('max_drawdown'), 0.0):.4%}`",
        f"- switch replay (coverage-adjusted) OOS return / sharpe / max_dd: `{_safe_float(coverage_metrics.get('total_return'), 0.0):+.4%}` / `{_safe_float(coverage_metrics.get('sharpe'), 0.0):.4f}` / `{_safe_float(coverage_metrics.get('max_drawdown'), 0.0):.4%}`",
        f"- switch replay (strict) OOS return / sharpe / max_dd: `{_safe_float(strict_metrics.get('total_return'), 0.0):+.4%}` / `{_safe_float(strict_metrics.get('sharpe'), 0.0):.4f}` / `{_safe_float(strict_metrics.get('max_drawdown'), 0.0):.4%}`",
        "",
        "## Execution realism proxy",
        f"- OOS days: `{allocation_summary.get('oos_days')}`",
        f"- default sleeve switches: `{allocation_summary.get('default_switches')}` (rate `{_safe_float(allocation_summary.get('default_switch_rate'), 0.0):.2%}`)",
        f"- avg / median / p90 daily turnover proxy: `{_safe_float(allocation_summary.get('avg_daily_turnover_proxy'), 0.0):.4f}` / `{_safe_float(allocation_summary.get('median_daily_turnover_proxy'), 0.0):.4f}` / `{_safe_float(allocation_summary.get('p90_daily_turnover_proxy'), 0.0):.4f}`",
        f"- pair active ratio: `{_safe_float(allocation_summary.get('pair_active_ratio'), 0.0):.2%}` | avg pair weight `{_safe_float(allocation_summary.get('avg_pair_weight'), 0.0):.2%}` | max pair weight `{_safe_float(allocation_summary.get('max_pair_weight'), 0.0):.2%}` | configured cap `{_safe_float(hybrid_oos.get('pair_weight_cap_config'), 0.0):.2%}`",
        f"- avg / max cash weight: `{_safe_float(allocation_summary.get('avg_cash_weight'), 0.0):.2%}` / `{_safe_float(allocation_summary.get('max_cash_weight'), 0.0):.2%}`",
        "",
        "## Cost stress on hybrid sleeve",
        "",
        "| One-way cost | Estimated drag | Adj. OOS return | Adj. Sharpe | Adj. Max DD |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in cost_stress:
        lines.append(
            f"| `{_safe_float(item.get('one_way_cost_bps'), 0.0):.1f} bps` | `{_safe_float(item.get('total_cost_drag'), 0.0):+.4%}` | `{_safe_float(item.get('adjusted_total_return'), 0.0):+.4%}` | `{_safe_float(item.get('adjusted_sharpe'), 0.0):.4f}` | `{_safe_float(item.get('adjusted_max_drawdown'), 0.0):.4%}` |"
        )
    lines.extend(
        [
            "",
            "## Live readiness gaps",
            f"- config mode / testnet / data source / order state: `{readiness.get('config_mode')}` / `{readiness.get('testnet')}` / `{readiness.get('market_data_source')}` / `{readiness.get('order_state_source')}`",
            f"- exchange: `{readiness.get('exchange_name')}` `{readiness.get('exchange_driver')}` `{readiness.get('exchange_market_type')}`",
            f"- refresh artifact exists? `{readiness.get('refresh_artifact_exists')}` | status `{readiness.get('refresh_status')}` | stale_minutes `{readiness.get('refresh_stale_minutes')}`",
            f"- decision artifact exists? `{readiness.get('decision_artifact_exists')}` | decision `{readiness.get('decision')}`",
            f"- startup_reconciliation_hard_fail: `{readiness.get('startup_reconciliation_hard_fail')}`",
            f"- paper_ready_now: `{readiness.get('paper_ready_now')}`",
            f"- real_ready_now: `{readiness.get('real_ready_now')}`",
            f"- gaps: `{json.dumps(list(readiness.get('gaps') or []), ensure_ascii=False)}`",
            "",
            "## Verdict",
            f"- strategy_structure_ready: `{verdict.get('strategy_structure_ready')}`",
            f"- paper_ready_now: `{verdict.get('paper_ready_now')}`",
            f"- real_ready_now: `{verdict.get('real_ready_now')}`",
            f"- verdict: `{verdict.get('verdict')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch-path", type=Path, default=DEFAULT_SWITCH_PATH)
    parser.add_argument("--replay-path", type=Path, default=DEFAULT_REPLAY_PATH)
    parser.add_argument("--hybrid-path", type=Path, default=DEFAULT_HYBRID_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--refresh-path", type=Path, default=DEFAULT_REFRESH_PATH)
    parser.add_argument("--decision-path", type=Path, default=DEFAULT_DECISION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--one-way-cost-bps-grid", default="5,10,20,30")
    return parser


def _parse_cost_grid(value: str) -> list[float]:
    grid = [_safe_float(token.strip()) for token in str(value).split(",") if token.strip()]
    if not grid:
        raise ValueError("one-way cost grid must contain at least one numeric value")
    return grid


def main() -> None:
    args = _build_parser().parse_args()
    report = build_report(
        switch_payload=_read_json(Path(args.switch_path).resolve()),
        replay_payload=_read_json(Path(args.replay_path).resolve()),
        hybrid_payload=_read_json(Path(args.hybrid_path).resolve()),
        config_path=Path(args.config_path).resolve(),
        refresh_path=Path(args.refresh_path).resolve(),
        decision_path=Path(args.decision_path).resolve(),
        one_way_cost_bps_grid=_parse_cost_grid(args.one_way_cost_bps_grid),
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hybrid_live_investability_latest.json"
    md_path = output_dir / "hybrid_live_investability_latest.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
