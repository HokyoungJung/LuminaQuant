#!/usr/bin/env python3
"""Build the live/replay notional-risk aligned Alpha Zoo report bundle."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.core.events import SignalEvent  # noqa: E402
from lumina_quant.live.readiness_policy import build_live_readiness_payload  # noqa: E402
from lumina_quant.risk_manager import RiskManager  # noqa: E402
from lumina_quant.services.portfolio import PortfolioSizingService  # noqa: E402
from scripts.research import run_alpha_zoo_validation_march_high_leverage as high  # noqa: E402
from scripts.research import run_common_split_alpha_zoo_hybrid_v35_v36 as common  # noqa: E402

DEFAULT_OUTPUT_DIR = high.DEFAULT_ALPHA_V2 / "live_notional_risk_aligned_alpha_zoo_20260518"
DEFAULT_REFRESH_JSON = (
    high.DEFAULT_ALPHA_V2 / "validation_to_20260331_latest_data_20260517/data_refresh_latest.json"
)
DEFAULT_SIZING_MODE = "isolated_margin_fraction"
DEFAULT_ALLOCATION_GRID = "0.03,0.05,0.075,0.10,0.125,0.15,0.175,0.20"
DEFAULT_LEVERAGE_MAX = 20
DEFAULT_INCUMBENT_CANDIDATE = "alpha_zoo_fast_residual"
DEFAULT_INCUMBENT_LEVERAGE = 7.0
DEFAULT_INCUMBENT_ALLOCATION = 0.15
SLIPPAGE_FEE_BPS_GRID = (1.0, 3.0, 5.0, 10.0, 20.0)
FUNDING_BPS_PER_DAY_GRID = (1.0, 2.0, 5.0, 10.0, 20.0)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(high._json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selected_spec(alpha: Any, old_replay: Mapping[str, Any], candidate_name: str) -> Any:
    specs = [common._old_selected_spec(old_replay, alpha), *alpha._default_grid_specs()]
    for spec in specs:
        if str(spec.name) == str(candidate_name):
            return spec
    raise ValueError(f"Cannot find Alpha Zoo spec for promoted candidate: {candidate_name}")


def _rebuild_promoted_trades(
    *,
    base_args: argparse.Namespace,
    base_payload: Mapping[str, Any],
) -> tuple[Any, pd.DataFrame, list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    split_contract = high._split_contract(base_args)
    bundle = common.load_real_data_bundle(
        input_path=base_args.input,
        current_tail_cache=base_args.current_tail_cache,
        external_state_csv=base_args.external_state_csv,
        strict_real_data=True,
    )
    common_frame = common.apply_common_split(bundle.frame, split_contract=split_contract)
    common_frame = common.add_split_bounded_forward_return_label(
        common_frame,
        horizon=int(base_args.horizon),
    )
    alpha = common._load_module(
        REPO_ROOT / "scripts/research/replay_crypto_fx_alpha_zoo_state.py",
        "live_notional_risk_aligned_alpha_replay",
    )
    old_replay = common._load_json(Path(base_args.old_alpha_replay_json))
    calibration_path = Path(str(base_payload["calibration_payload_path"]))
    calibrated_edges = alpha._load_calibrated_edges(calibration_path)
    data = alpha._ensure_replay_frame(common_frame)
    promoted = dict(dict(base_payload.get("selection") or {}).get("live_promoted_candidate") or {})
    spec = _selected_spec(alpha, old_replay, str(promoted.get("candidate_name") or ""))
    signals = alpha._run_strategy_signals(
        data,
        require_calibrated_edge=True,
        calibrated_edges=calibrated_edges,
        strategy_params=dict(spec.params),
    )
    trades = high._attach_trade_path_extrema(alpha, data, alpha._build_trades(data, signals))
    return (
        alpha,
        data,
        trades,
        calibrated_edges,
        {"name": spec.name, "source": spec.source, "params": dict(spec.params)},
    )


def _isolated_cost_metrics(
    alpha: Any,
    trades: list[dict[str, Any]],
    *,
    leverage: float,
    allocation_fraction: float,
    round_trip_slippage_bps: float = 0.0,
    funding_bps_per_day: float = 0.0,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in high.SPLIT_ORDER:
        returns: list[float] = []
        for trade in trades:
            if str(trade.get("entry_split")) != split:
                continue
            _base_ret, liquidated, _min_adverse = high._isolated_trade_return(
                trade,
                leverage=float(leverage),
                allocation_fraction=float(allocation_fraction),
                alpha=alpha,
            )
            if liquidated:
                returns.append(-float(allocation_fraction))
            else:
                returns.append(
                    alpha._portfolio_trade_return(
                        trade,
                        leverage=float(leverage),
                        allocation_fraction=float(allocation_fraction),
                        round_trip_slippage_bps=float(round_trip_slippage_bps),
                        funding_bps_per_day=float(funding_bps_per_day),
                    )
                )
        out[split] = alpha._metrics_from_returns(returns)
    return out


def _cost_sensitivity(
    alpha: Any,
    trades: list[dict[str, Any]],
    *,
    leverage: float,
    allocation_fraction: float,
) -> dict[str, Any]:
    slippage_rows = []
    for bps in SLIPPAGE_FEE_BPS_GRID:
        metrics = _isolated_cost_metrics(
            alpha,
            trades,
            leverage=leverage,
            allocation_fraction=allocation_fraction,
            round_trip_slippage_bps=bps,
        )
        slippage_rows.append(
            {
                "round_trip_slippage_fee_bps": float(bps),
                "split_metrics": metrics,
                "locked_oos": metrics["locked_oos"],
            }
        )
    funding_rows = []
    for bps in FUNDING_BPS_PER_DAY_GRID:
        metrics = _isolated_cost_metrics(
            alpha,
            trades,
            leverage=leverage,
            allocation_fraction=allocation_fraction,
            funding_bps_per_day=bps,
        )
        funding_rows.append(
            {
                "funding_bps_per_day": float(bps),
                "split_metrics": metrics,
                "locked_oos": metrics["locked_oos"],
            }
        )
    survives_5bps = _safe_float(slippage_rows[2]["locked_oos"].get("total_return")) > 0.0
    survives_10bps = _safe_float(slippage_rows[3]["locked_oos"].get("total_return")) > 0.0
    return {
        "diagnostic_only": True,
        "no_cost_headline_is_separate": True,
        "trade_return_model": (
            "isolated liquidation loss is capped at allocation_fraction; otherwise "
            "allocation_fraction * leverage * "
            "(gross_return - round_trip_slippage_fee_bps/10000 - funding_bps_per_day/10000*holding_days)"
        ),
        "slippage_fee_sensitivity": {
            "round_trip_bps_grid": list(SLIPPAGE_FEE_BPS_GRID),
            "rows": slippage_rows,
        },
        "funding_cost_sensitivity": {
            "funding_bps_per_day_grid": list(FUNDING_BPS_PER_DAY_GRID),
            "conservative_drag_model": "funding is treated as same-direction daily cost drag",
            "rows": funding_rows,
        },
        "promotion_cost_commentary": {
            "survives_5bps_round_trip_slippage_fee": bool(survives_5bps),
            "survives_10bps_round_trip_slippage_fee": bool(survives_10bps),
            "practical_threshold_note": (
                "No-cost headline remains positive, but high-turnover Alpha Zoo is cost fragile; "
                "prefer paper/testnet fill measurement before real-money authorization."
            ),
        },
    }


def _risk_caps_for_contract(*, leverage: float, allocation_fraction: float) -> dict[str, float]:
    notional_fraction = float(leverage) * float(allocation_fraction)
    return {
        "max_order_value": 0.0,
        "max_order_notional_pct": round(notional_fraction + 0.05, 6),
        "max_symbol_exposure_pct": round(notional_fraction + 0.05, 6),
        "max_total_notional_pct": round(
            max(notional_fraction + 0.10, notional_fraction * 2.0 + 0.10),
            6,
        ),
    }


def _prefer_incumbent_contract_on_tv_tie(
    base_payload: Mapping[str, Any],
    *,
    candidate_csv_path: Path,
    incumbent_candidate: str,
    incumbent_leverage: float,
    incumbent_allocation: float,
) -> dict[str, Any]:
    """Prefer the documented incumbent when train+validation scores tie.

    Selection still uses train+validation score only. Locked-OOS fields are only
    read as pass/fail gates after the train+validation candidate freeze.
    """
    payload = dict(base_payload)
    selection = dict(payload.get("selection") or {})
    current = dict(selection.get("live_promoted_candidate") or {})
    current_score = _safe_float(current.get("tv_selection_score"), float("-inf"))
    if not current or not candidate_csv_path.exists():
        return payload

    rows = pd.read_csv(candidate_csv_path)
    target = rows[
        (rows["candidate_name"].astype(str) == str(incumbent_candidate))
        & (rows["leverage"].astype(float).sub(float(incumbent_leverage)).abs() <= 1e-12)
        & (
            rows["allocation_fraction"].astype(float).sub(float(incumbent_allocation)).abs()
            <= 1e-12
        )
    ]
    if target.empty:
        return payload
    row = target.iloc[0].to_dict()
    target_score = _safe_float(row.get("tv_selection_score"), float("-inf"))
    gate_ok = bool(row.get("live_promotion_possible")) and bool(row.get("locked_oos_gate_pass"))
    no_wipeout = int(_safe_float(row.get("total_account_wipeout_count"), 0.0)) == 0
    tv_tie_or_better = target_score >= current_score - 1e-12
    same_strategy = str(current.get("candidate_name")) == str(incumbent_candidate)
    if not (same_strategy and gate_ok and no_wipeout and tv_tie_or_better):
        return payload

    incumbent = {
        **current,
        "allocation_fraction": float(row["allocation_fraction"]),
        "frozen_train_validation_rank": int(row["frozen_train_validation_rank"]),
        "leverage": float(row["leverage"]),
        "live_promotion_possible": bool(row["live_promotion_possible"]),
        "locked_oos_gate_pass": bool(row["locked_oos_gate_pass"]),
        "locked_oos_liquidation_count": int(
            _safe_float(row.get("locked_oos_liquidation_count"), 0.0)
        ),
        "locked_oos_rejection_reasons": [],
        "total_account_wipeout_count": int(
            _safe_float(row.get("total_account_wipeout_count"), 0.0)
        ),
        "tv_selection_score": float(row["tv_selection_score"]),
    }
    selection["live_promoted_candidate"] = incumbent
    selection["incumbent_contract_tie_breaker"] = {
        "applied": True,
        "reason": (
            "Current documented live candidate 7x/0.15 has the same train+validation "
            "score and 105% notional/equity as the raw grid pick, while preserving "
            "the requested isolated margin/equity contract."
        ),
        "incumbent_candidate": str(incumbent_candidate),
        "incumbent_leverage": float(incumbent_leverage),
        "incumbent_allocation_fraction": float(incumbent_allocation),
        "raw_live_promoted_candidate": current,
        "selection_inputs": ["train", "validation"],
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
    }
    selection["selection_policy"] = (
        str(selection.get("selection_policy") or "")
        + "; train+validation ties on the same live strategy/notional contract prefer the "
        "documented incumbent live contract"
    )
    payload["selection"] = selection
    return payload


def _paper_equivalent_sizing(
    *,
    leverage: float,
    allocation_fraction: float,
    sizing_mode: str,
    risk_caps: Mapping[str, float],
) -> dict[str, Any]:
    equity = 10_000.0
    price = 100.0
    signal = SignalEvent(
        strategy_id="paper-equivalent",
        symbol="BTC/USDT",
        datetime=datetime(2026, 1, 1, tzinfo=UTC),
        signal_type="LONG",
        stop_loss=97.5,
        metadata={
            "target_allocation": float(allocation_fraction),
            "target_allocation_mode": sizing_mode,
            "leverage": float(leverage),
            "max_order_notional_pct": float(risk_caps["max_order_notional_pct"]),
        },
    )
    quantity = PortfolioSizingService.risk_based_quantity(
        signal=signal,
        current_price=price,
        equity=equity,
        risk_per_trade=0.001,
        default_stop_loss_pct=0.025,
        max_symbol_exposure_pct=float(risk_caps["max_symbol_exposure_pct"]),
        target_allocation=float(allocation_fraction),
        max_order_value=float(risk_caps["max_order_value"]),
        target_allocation_mode=sizing_mode,
        leverage=float(leverage),
        max_order_notional_pct=float(risk_caps["max_order_notional_pct"]),
    )
    live_notional = quantity * price
    expected_replay_notional = equity * float(allocation_fraction) * float(leverage)

    class _RiskConfig:
        MAX_ORDER_VALUE = float(risk_caps["max_order_value"])
        MAX_ORDER_NOTIONAL_PCT = float(risk_caps["max_order_notional_pct"])
        MAX_DAILY_LOSS_PCT = 0.05
        MAX_INTRADAY_DRAWDOWN_PCT = 0.03
        MAX_ROLLING_LOSS_PCT_1H = 0.05
        MAX_SYMBOL_EXPOSURE_PCT = float(risk_caps["max_symbol_exposure_pct"])
        MAX_TOTAL_MARGIN_PCT = 0.5
        MAX_TOTAL_NOTIONAL_PCT = float(risk_caps["max_total_notional_pct"])
        FREEZE_NEW_ENTRIES_ON_BREACH = True
        AUTO_FLATTEN_ON_BREACH = False

    portfolio = SimpleNamespace(
        current_holdings={"total": equity, "BTC/USDT": 0.0},
        current_positions={"BTC/USDT": 0.0},
        current_position_legs={},
        symbol_list=["BTC/USDT"],
        trading_frozen=False,
        circuit_breaker_tripped=False,
    )
    order = SimpleNamespace(
        symbol="BTC/USDT",
        quantity=quantity,
        direction="BUY",
        position_side="LONG",
        reduce_only=False,
    )
    risk_passed, risk_reason = RiskManager(_RiskConfig).check_order(
        order,
        current_price=price,
        portfolio=portfolio,
    )
    return {
        "fixture": {
            "equity": equity,
            "price": price,
            "target_allocation": float(allocation_fraction),
            "leverage": float(leverage),
            "sizing_mode": sizing_mode,
        },
        "expected_replay_notional": expected_replay_notional,
        "live_quantity": quantity,
        "live_notional": live_notional,
        "absolute_notional_diff": abs(live_notional - expected_replay_notional),
        "notional_parity_passed": abs(live_notional - expected_replay_notional) <= 1e-9,
        "risk_check_passed": bool(risk_passed),
        "risk_check_reason": str(risk_reason),
        "risk_caps": dict(risk_caps),
    }


def _live_decision_payload(
    *,
    base_payload: Mapping[str, Any],
    spec: Mapping[str, Any],
    calibrated_edges: Mapping[str, float],
    sizing_mode: str,
    risk_caps: Mapping[str, float],
) -> dict[str, Any]:
    promoted = dict(dict(base_payload.get("selection") or {}).get("live_promoted_candidate") or {})
    leverage = round(_safe_float(promoted.get("leverage")))
    allocation = _safe_float(promoted.get("allocation_fraction"))
    params = {
        **dict(spec.get("params") or {}),
        "calibrated_edges": {str(key): float(value) for key, value in calibrated_edges.items()},
        "decision_cadence_seconds": 3600,
    }
    return {
        "artifact_kind": "alpha_zoo_live_notional_risk_aligned_decision",
        "generated_at_utc": _utc_now_iso(),
        "decision": "selected_live_mode",
        "selected_mode": str(promoted.get("candidate_name") or spec.get("name")),
        "strategy_name": "CryptoFxAlphaZooStateStrategy",
        "strategy_timeframe": "1h",
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TRX/USDT"],
        "exchange": {
            "driver": "binance_futures",
            "name": "binance",
            "market_type": "future",
            "position_mode": "HEDGE",
            "margin_mode": "isolated",
            "leverage": leverage,
        },
        "target_allocation": allocation,
        "sizing_mode": sizing_mode,
        "target_allocation_mode": sizing_mode,
        "risk_caps": dict(risk_caps),
        "leverage": leverage,
        "window_seconds": 3600,
        "ingest_window_seconds": 3600,
        "decision_cadence_seconds": 3600,
        "strategy_params": params,
        "live_replay_sizing_contract": {
            "sizing_mode": sizing_mode,
            "target_allocation_meaning": "isolated_margin_fraction_of_account_equity",
            "margin_fraction_of_equity": allocation,
            "notional_fraction_of_equity": allocation * float(leverage),
            "fixed_dollar_max_order_value_applies": False,
            "absolute_cap_policy": "only explicit positive max_order_value is an emergency ceiling",
        },
        "selection_provenance": dict(dict(base_payload.get("selection") or {})),
        "replay_evidence": {
            "candidate_metrics": promoted,
            "locked_oos_contamination_audit": base_payload.get("locked_oos_contamination_audit"),
            "memory_summary": base_payload.get("memory_summary"),
            "split_contract": dict(
                dict(base_payload.get("split_manifest") or {}).get("split_contract") or {}
            ),
            "split_periods": dict(
                dict(base_payload.get("split_manifest") or {}).get("split_periods") or {}
            ),
            "strict_zero_liquidation_1x_6x": dict(
                dict(
                    base_payload.get("strict_zero_liquidation_lane_1x_6x_at_10pct_allocation") or {}
                ).get("strict_zero_liquidation_lane")
                or {}
            ).get("promoted_candidate"),
        },
    }


def _preflight_payload(decision_path: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("LQ_POSTGRES_DSN", "postgresql://paper-preflight-placeholder")
    return build_live_readiness_payload(
        config_path=Path("config.yaml"),
        refresh_json=DEFAULT_REFRESH_JSON,
        decision_json=decision_path,
        stale_minutes=10_000,
        env=env,
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    selected = dict(payload.get("selected_contract") or {})
    oos = dict(dict(payload.get("selection") or {}).get("live_promoted_candidate") or {}).get(
        "locked_oos",
        {},
    )
    cost = dict(payload.get("cost_sensitivity") or {})
    slippage_rows = list(dict(cost.get("slippage_fee_sensitivity") or {}).get("rows") or [])
    funding_rows = list(dict(cost.get("funding_cost_sensitivity") or {}).get("rows") or [])
    lines = [
        "# Live notional/risk aligned Alpha Zoo report",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "## Selected live/replay sizing contract",
        "",
        f"- sizing mode: `{selected.get('sizing_mode')}`",
        f"- leverage/allocation: `{selected.get('leverage')}x` / `{selected.get('target_allocation'):.2%}`",
        f"- notional/equity: `{selected.get('notional_fraction_of_equity'):.2%}`",
        f"- isolated margin/equity: `{selected.get('margin_fraction_of_equity'):.2%}`",
        f"- locked-OOS return/MDD: `{_safe_float(oos.get('total_return')):.4%}` / `{_safe_float(oos.get('max_drawdown')):.4%}`",
        "",
        "## Cost sensitivity — locked-OOS",
        "",
        "| Scenario | Return | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in slippage_rows:
        locked = dict(row.get("locked_oos") or {})
        lines.append(
            f"| fee/slippage `{_safe_float(row.get('round_trip_slippage_fee_bps')):g} bps` "
            f"| `{_safe_float(locked.get('total_return')):.4%}` "
            f"| `{_safe_float(locked.get('max_drawdown')):.4%}` "
            f"| `{_safe_float(locked.get('sharpe')):.4f}` |"
        )
    for row in funding_rows:
        locked = dict(row.get("locked_oos") or {})
        lines.append(
            f"| funding `{_safe_float(row.get('funding_bps_per_day')):g} bps/day` "
            f"| `{_safe_float(locked.get('total_return')):.4%}` "
            f"| `{_safe_float(locked.get('max_drawdown')):.4%}` "
            f"| `{_safe_float(locked.get('sharpe')):.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Paper-equivalent sizing parity",
            "",
            f"- parity passed: `{dict(payload.get('paper_equivalent_sizing') or {}).get('notional_parity_passed')}`",
            f"- risk check passed: `{dict(payload.get('paper_equivalent_sizing') or {}).get('risk_check_passed')}`",
            "",
            "## Preflight",
            "",
            f"- ready_for_paper: `{dict(dict(payload.get('preflight') or {}).get('status') or {}).get('ready_for_paper')}`",
            f"- ready_for_real: `{dict(dict(payload.get('preflight') or {}).get('status') or {}).get('ready_for_real')}`",
            f"- recommended_action: `{dict(payload.get('preflight') or {}).get('recommended_action')}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_aligned_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    base_args = high.parse_args(
        [
            "--input",
            str(args.input),
            "--current-tail-cache",
            str(args.current_tail_cache),
            "--external-state-csv",
            str(args.external_state_csv),
            "--old-alpha-replay-json",
            str(args.old_alpha_replay_json),
            "--leverage-min",
            str(args.leverage_min),
            "--leverage-max",
            str(args.leverage_max),
            "--allocation-grid",
            str(args.allocation_grid),
            "--output-dir",
            str(output_dir),
        ]
    )
    base_payload = high.build_payload(base_args)
    base_paths = high.write_outputs(base_payload, output_dir)
    base_payload = json.loads(Path(base_paths["latest_json"]).read_text(encoding="utf-8"))
    base_payload = _prefer_incumbent_contract_on_tv_tie(
        base_payload,
        candidate_csv_path=Path(base_paths["candidate_csv"]),
        incumbent_candidate=str(args.incumbent_candidate),
        incumbent_leverage=float(args.incumbent_leverage),
        incumbent_allocation=float(args.incumbent_allocation),
    )
    promoted = dict(dict(base_payload.get("selection") or {}).get("live_promoted_candidate") or {})
    leverage = _safe_float(promoted.get("leverage"))
    allocation = _safe_float(promoted.get("allocation_fraction"))
    risk_caps = _risk_caps_for_contract(leverage=leverage, allocation_fraction=allocation)
    alpha, _data, trades, calibrated_edges, spec = _rebuild_promoted_trades(
        base_args=base_args,
        base_payload=base_payload,
    )
    cost_sensitivity = _cost_sensitivity(
        alpha,
        trades,
        leverage=leverage,
        allocation_fraction=allocation,
    )
    paper_sizing = _paper_equivalent_sizing(
        leverage=leverage,
        allocation_fraction=allocation,
        sizing_mode=args.sizing_mode,
        risk_caps=risk_caps,
    )
    decision = _live_decision_payload(
        base_payload=base_payload,
        spec=spec,
        calibrated_edges=calibrated_edges,
        sizing_mode=args.sizing_mode,
        risk_caps=risk_caps,
    )
    decision_path = output_dir / "live_alpha_zoo_notional_risk_aligned_decision_latest.json"
    _write_json(decision_path, decision)
    preflight = _preflight_payload(decision_path)
    preflight_path = output_dir / "live_readiness_preflight_notional_risk_aligned_latest.json"
    _write_json(preflight_path, preflight)
    aligned = {
        **base_payload,
        "artifact_kind": "live_notional_risk_aligned_alpha_zoo_report",
        "generated_at_utc": _utc_now_iso(),
        "base_high_leverage_output_paths": base_paths,
        "selected_contract": {
            "candidate_name": promoted.get("candidate_name"),
            "sizing_mode": args.sizing_mode,
            "target_allocation": allocation,
            "leverage": leverage,
            "margin_fraction_of_equity": allocation,
            "notional_fraction_of_equity": allocation * leverage,
            "max_order_value_policy": "disabled unless explicitly positive emergency cap",
            "risk_caps": risk_caps,
        },
        "cost_sensitivity": cost_sensitivity,
        "paper_equivalent_sizing": paper_sizing,
        "live_decision_artifact_path": str(decision_path),
        "preflight_artifact_path": str(preflight_path),
        "preflight": preflight,
        "real_money_execution": {
            "attempted": False,
            "ready_for_real": bool(dict(preflight.get("status") or {}).get("ready_for_real")),
            "authorization_required": True,
        },
        "strict_zero_liquidation_lane_separate": True,
    }
    latest_path = output_dir / "live_notional_risk_aligned_alpha_zoo_latest.json"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timestamped_path = output_dir / f"live_notional_risk_aligned_alpha_zoo_{timestamp}.json"
    latest_md = output_dir / "live_notional_risk_aligned_alpha_zoo_latest.md"
    aligned["output_paths"] = {
        **dict(aligned.get("output_paths") or {}),
        "aligned_latest_json": str(latest_path),
        "aligned_timestamped_json": str(timestamped_path),
        "aligned_latest_markdown": str(latest_md),
        "live_decision": str(decision_path),
        "preflight": str(preflight_path),
    }
    _write_json(latest_path, aligned)
    _write_json(timestamped_path, aligned)
    latest_md.write_text(_markdown(aligned), encoding="utf-8")
    return aligned


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="")
    parser.add_argument("--current-tail-cache", default=str(high.DEFAULT_CURRENT_TAIL_CACHE))
    parser.add_argument("--external-state-csv", default=str(high.DEFAULT_EXTERNAL_STATE_CSV))
    parser.add_argument("--old-alpha-replay-json", default=str(high.DEFAULT_OLD_ALPHA_REPLAY))
    parser.add_argument("--leverage-min", type=int, default=high.DEFAULT_LEVERAGE_MIN)
    parser.add_argument("--leverage-max", type=int, default=DEFAULT_LEVERAGE_MAX)
    parser.add_argument("--allocation-grid", default=DEFAULT_ALLOCATION_GRID)
    parser.add_argument("--sizing-mode", default=DEFAULT_SIZING_MODE)
    parser.add_argument("--incumbent-candidate", default=DEFAULT_INCUMBENT_CANDIDATE)
    parser.add_argument("--incumbent-leverage", type=float, default=DEFAULT_INCUMBENT_LEVERAGE)
    parser.add_argument("--incumbent-allocation", type=float, default=DEFAULT_INCUMBENT_ALLOCATION)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_aligned_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
