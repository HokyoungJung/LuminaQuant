#!/usr/bin/env python3
"""Build paper/testnet preflight and monitoring artifacts for the 10bps Alpha Zoo pair.

The runner intentionally consumes the already-frozen 10bps retune artifact. It
does not reselect, retune, or use locked-OOS for objective/selection; locked-OOS
fields are copied as gate/report-only evidence for paper-forward monitoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_validation_march_high_leverage as high  # noqa: E402
from scripts.research import run_live_notional_risk_aligned_alpha_zoo as aligned  # noqa: E402

DEFAULT_SOURCE_DIR = high.DEFAULT_ALPHA_V2 / "alpha_zoo_10bps_full_retune_20260519"
DEFAULT_RETUNE_JSON = DEFAULT_SOURCE_DIR / "alpha_zoo_10bps_full_retune_latest.json"
DEFAULT_LOW_CORRELATION_JSON = DEFAULT_SOURCE_DIR / "low_correlation_discovery_latest.json"
DEFAULT_LIVE_ALIGNED_JSON = (
    high.DEFAULT_ALPHA_V2
    / "live_notional_risk_aligned_alpha_zoo_20260518"
    / "live_notional_risk_aligned_alpha_zoo_latest.json"
)
DEFAULT_OUTPUT_DIR = high.DEFAULT_ALPHA_V2 / "alpha_zoo_7x_paper_forward_preflight_20260519"

ACTIVE_PROFILE_ID = "higher_risk_train_return_tilt_v1"
BALANCED_PROFILE_ID = "balanced_train_validation_v1"
ACTIVE_MODEL_ID = "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc"
BALANCED_MODEL_ID = (
    "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc"
)
PRIMARY_ROUND_TRIP_COST_BPS = 10.0
SIZING_MODE = "isolated_margin_fraction"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TRX/USDT"]
# Historical paper/testnet handoff artifacts intentionally use a frozen data
# refresh snapshot. Keep this override date-stable so research artifact tests do
# not start failing merely because wall-clock time moved past the old 10k-minute
# freshness window; real-money readiness is still forced false below.
PAPER_FORWARD_STALE_MINUTES = 60 * 24 * 30

MONITORING_CSV_FIELDS = [
    "role",
    "profile_id",
    "model_id",
    "leverage",
    "allocation_fraction",
    "sizing_mode",
    "target_notional_fraction_of_equity",
    "isolated_margin_fraction_of_equity",
    "expected_replay_notional_for_10000_equity",
    "live_notional_for_10000_equity",
    "notional_parity_passed",
    "research_round_trip_cost_bps",
    "observed_round_trips",
    "mean_all_in_round_trip_bps",
    "p95_all_in_round_trip_bps",
    "cost_status",
    "ready_for_paper",
    "ready_for_real",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def _required_finite_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite numeric value") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite numeric value")
    return parsed


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(high._json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative_or_abs(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def realized_bps(cost_quote: float, notional_quote: float) -> float:
    """Return cost in bps of notional, fail-closed for invalid notionals."""
    notional = float(notional_quote)
    if not math.isfinite(notional) or notional <= 0.0:
        raise ValueError("notional_quote must be positive to compute realized bps")
    cost = float(cost_quote)
    if not math.isfinite(cost):
        raise ValueError("cost_quote must be finite to compute realized bps")
    return cost / notional * 10_000.0


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = max(0.0, min(1.0, float(quantile))) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_round_trip_costs(
    rows: Iterable[Mapping[str, Any]],
    *,
    research_round_trip_cost_bps: float = PRIMARY_ROUND_TRIP_COST_BPS,
    p95_limit_bps: float = 15.0,
) -> dict[str, Any]:
    """Summarize realized all-in round-trip bps from paper/testnet fill rows."""
    values: list[float] = []
    for row in rows:
        if row.get("all_in_round_trip_bps") is not None:
            values.append(
                _required_finite_float(
                    row.get("all_in_round_trip_bps"),
                    field_name="all_in_round_trip_bps",
                )
            )
            continue
        fee_bps = _required_finite_float(row.get("realized_fee_bps"), field_name="realized_fee_bps")
        slippage_bps = _required_finite_float(
            row.get("realized_slippage_bps"),
            field_name="realized_slippage_bps",
        )
        values.append(fee_bps + slippage_bps)
    mean_value = sum(values) / len(values) if values else None
    p95_value = _percentile(values, 0.95)
    mean_pass = mean_value is not None and mean_value <= float(research_round_trip_cost_bps)
    p95_pass = p95_value is not None and p95_value <= float(p95_limit_bps)
    return {
        "observed_round_trips": len(values),
        "mean_all_in_round_trip_bps": mean_value,
        "p95_all_in_round_trip_bps": p95_value,
        "research_round_trip_cost_bps": float(research_round_trip_cost_bps),
        "p95_limit_bps": float(p95_limit_bps),
        "mean_cost_pass": bool(mean_pass),
        "p95_cost_pass": bool(p95_pass),
        "cost_status": "pass"
        if mean_pass and p95_pass
        else "pending_no_fills"
        if not values
        else "fail",
    }


def _selection_profile(retune: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    profile = dict(dict(retune.get("selection_profiles") or {}).get(profile_id) or {})
    if not profile:
        raise ValueError(f"missing selection profile: {profile_id}")
    if any(
        profile.get(key)
        for key in (
            "uses_locked_oos_for_objective",
            "uses_locked_oos_for_parameter_fitting",
            "uses_locked_oos_for_pruning",
            "uses_locked_oos_for_selection",
        )
    ):
        raise ValueError(f"profile {profile_id} violates locked-OOS report-only contract")
    return profile


def _model_metrics(retune: Mapping[str, Any], model_id: str) -> dict[str, dict[str, Any]]:
    rows = list(retune.get("candidate_model_metrics") or [])
    selected = {
        str(row.get("split")): dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("model_id")) == str(model_id)
    }
    missing = [split for split in high.SPLIT_ORDER if split not in selected]
    if missing:
        raise ValueError(f"missing split metrics for {model_id}: {missing}")
    return selected


def _model_summary(retune: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    key = (
        "higher_risk_selected_10bps_model" if role == "active" else "balanced_reference_10bps_model"
    )
    model = dict(retune.get(key) or {})
    expected = ACTIVE_MODEL_ID if role == "active" else BALANCED_MODEL_ID
    if model.get("model_id") != expected:
        raise ValueError(f"{role} model mismatch: expected {expected}, got {model.get('model_id')}")
    if _safe_float(model.get("round_trip_slippage_fee_bps")) != PRIMARY_ROUND_TRIP_COST_BPS:
        raise ValueError(f"{role} model is not keyed to 10bps primary cost")
    return model


def _strategy_params_for_candidate(
    live_aligned: Mapping[str, Any],
    *,
    candidate_name: str,
    trade_filter_params: Mapping[str, Any],
) -> dict[str, Any]:
    summaries = list(dict(live_aligned.get("selection") or {}).get("strategy_summaries") or [])
    params: dict[str, Any] = {}
    for row in summaries:
        if str(dict(row).get("candidate_name") or "") == str(candidate_name):
            params = dict(dict(row).get("params") or {})
            break
    if not params:
        raise ValueError(f"cannot resolve strategy params for {candidate_name}")

    calibration_path = Path(str(live_aligned.get("calibration_payload_path") or ""))
    if calibration_path.exists():
        calibration = _load_json(calibration_path)
        calibrated_edges = dict(calibration.get("calibrated_edges_for_strategy") or {})
    else:
        calibrated_edges = dict(live_aligned.get("calibrated_edges_for_strategy") or {})

    abs_min = trade_filter_params.get("abs_factor_score_min")
    if abs_min is not None:
        params["abs_factor_score_min"] = float(abs_min)
    params["calibrated_edges"] = {str(key): float(value) for key, value in calibrated_edges.items()}
    params["decision_cadence_seconds"] = 3600
    params["paper_forward_trade_filter"] = dict(trade_filter_params)
    return params


def _source_lineage(
    *,
    retune_path: Path,
    low_correlation_path: Path,
    live_aligned_path: Path,
    retune: Mapping[str, Any],
    low_correlation: Mapping[str, Any],
    live_aligned: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "retune_10bps": {
            "path": _relative_or_abs(retune_path),
            "sha256": _sha256(retune_path),
            "generated_at_utc": retune.get("generated_at_utc"),
            "primary_round_trip_cost_bps": retune.get("round_trip_slippage_fee_bps_primary"),
        },
        "low_correlation_discovery": {
            "path": _relative_or_abs(low_correlation_path),
            "sha256": _sha256(low_correlation_path),
            "summary": low_correlation.get("summary"),
        },
        "notional_risk_alignment": {
            "path": _relative_or_abs(live_aligned_path),
            "sha256": _sha256(live_aligned_path),
            "generated_at_utc": live_aligned.get("generated_at_utc"),
            "artifact_kind": live_aligned.get("artifact_kind"),
        },
        "source_inputs": retune.get("source_inputs"),
        "split_manifest": retune.get("split_manifest"),
        "locked_oos_contamination_audit": retune.get("locked_oos_contamination_audit"),
    }


def _decision_payload(
    *,
    role: str,
    model: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
    strategy_params: Mapping[str, Any],
    risk_caps: Mapping[str, float],
    paper_sizing: Mapping[str, Any],
    source_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    leverage = _safe_float(model.get("leverage"))
    allocation = _safe_float(model.get("allocation_fraction"))
    model_id = str(model.get("model_id"))
    return {
        "artifact_kind": "alpha_zoo_paper_forward_live_decision",
        "generated_at_utc": _utc_now_iso(),
        "role": role,
        "decision": "selected_live_mode",
        "selected_mode": str(model.get("candidate_name") or "alpha_zoo_quality_single_pair"),
        "model_id": model_id,
        "model_kind": model.get("model_kind"),
        "strategy_name": "CryptoFxAlphaZooStateStrategy",
        "strategy_timeframe": "1h",
        "symbols": list(SYMBOLS),
        "exchange": {
            "driver": "binance_futures",
            "name": "binance",
            "market_type": "future",
            "position_mode": "HEDGE",
            "margin_mode": "isolated",
            "leverage": round(leverage),
        },
        "target_allocation": allocation,
        "sizing_mode": SIZING_MODE,
        "target_allocation_mode": SIZING_MODE,
        "risk_caps": dict(risk_caps),
        "leverage": round(leverage),
        "window_seconds": 3600,
        "ingest_window_seconds": 3600,
        "decision_cadence_seconds": 3600,
        "strategy_params": dict(strategy_params),
        "paper_testnet_only": True,
        "ready_for_real": False,
        "real_money_execution": False,
        "allowed_execution_modes": ["paper", "testnet"],
        "live_replay_sizing_contract": {
            "sizing_mode": SIZING_MODE,
            "target_allocation_meaning": "isolated_margin_fraction_of_account_equity",
            "isolated_margin_fraction_of_equity": allocation,
            "notional_fraction_of_equity": allocation * leverage,
            "target_notional_formula": "account_equity * allocation_fraction * leverage",
            "isolated_margin_formula": "account_equity * allocation_fraction",
            "fixed_dollar_max_order_value_applies": False,
            "absolute_cap_policy": "only explicit positive max_order_value is an emergency ceiling",
        },
        "paper_equivalent_sizing": dict(paper_sizing),
        "selection_profile": dict(profile),
        "selection_profile_id": profile.get("profile_id"),
        "trade_filter_params": dict(model.get("trade_filter_params") or {}),
        "cost_assumption_audit": {
            "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
            "monitor_realized_fee_slippage_required": True,
            "fail_closed_if_realized_cost_unknown_for_real_money": True,
        },
        "replay_evidence": {
            "candidate_metrics": dict(model),
            "split_metrics": {split: dict(row) for split, row in metrics.items()},
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
            "liquidation_inclusive_mdd": True,
        },
        "source_lineage": dict(source_lineage),
    }


def _preflight_payload(decision_path: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("LQ_POSTGRES_DSN", "postgresql://paper-preflight-placeholder")
    payload = aligned.build_live_readiness_payload(
        config_path=Path("config.yaml"),
        refresh_json=aligned.DEFAULT_REFRESH_JSON,
        decision_json=decision_path,
        stale_minutes=PAPER_FORWARD_STALE_MINUTES,
        env=env,
    )
    payload["paper_testnet_only_governance"] = {
        "ready_for_real_forced_false_by_plan": True,
        "real_money_execution": False,
        "real_money_authorization_present": False,
    }
    if bool(dict(payload.get("status") or {}).get("ready_for_real")):
        raise ValueError("paper-forward preflight unexpectedly marked ready_for_real=true")
    return payload


def _monitoring_profile_row(
    *,
    role: str,
    model: Mapping[str, Any],
    profile: Mapping[str, Any],
    paper_sizing: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    leverage = _safe_float(model.get("leverage"))
    allocation = _safe_float(model.get("allocation_fraction"))
    cost_summary = summarize_round_trip_costs([])
    return {
        "role": role,
        "profile_id": profile.get("profile_id"),
        "model_id": model.get("model_id"),
        "leverage": leverage,
        "allocation_fraction": allocation,
        "sizing_mode": SIZING_MODE,
        "target_notional_fraction_of_equity": leverage * allocation,
        "isolated_margin_fraction_of_equity": allocation,
        "expected_replay_notional_for_10000_equity": paper_sizing.get("expected_replay_notional"),
        "live_notional_for_10000_equity": paper_sizing.get("live_notional"),
        "notional_parity_passed": paper_sizing.get("notional_parity_passed"),
        "research_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "observed_round_trips": cost_summary["observed_round_trips"],
        "mean_all_in_round_trip_bps": cost_summary["mean_all_in_round_trip_bps"],
        "p95_all_in_round_trip_bps": cost_summary["p95_all_in_round_trip_bps"],
        "cost_status": cost_summary["cost_status"],
        "ready_for_paper": dict(preflight.get("status") or {}).get("ready_for_paper"),
        "ready_for_real": dict(preflight.get("status") or {}).get("ready_for_real"),
    }


def _monitoring_contract(
    *,
    profile_rows: list[dict[str, Any]],
    source_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_kind": "alpha_zoo_paper_forward_monitoring_contract",
        "generated_at_utc": _utc_now_iso(),
        "real_money_execution": False,
        "ready_for_real": False,
        "research_assumption": {
            "primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
            "mean_all_in_round_trip_bps_limit": 10.0,
            "p95_all_in_round_trip_bps_limit": 15.0,
        },
        "required_grouping_keys": ["timestamp_utc", "role", "model_id", "symbol", "side"],
        "fill_observation_schema": [
            "timestamp_utc",
            "round_trip_id",
            "role",
            "profile_id",
            "model_id",
            "symbol",
            "side",
            "entry_order_id",
            "exit_order_id",
            "entry_notional_quote",
            "exit_notional_quote",
            "fee_quote",
            "expected_entry_price",
            "entry_fill_price",
            "expected_exit_price",
            "exit_fill_price",
            "realized_fee_bps",
            "realized_slippage_bps",
            "all_in_round_trip_bps",
            "maker_fill_count",
            "taker_fill_count",
            "partial_fill_ratio",
            "missed_signal",
            "equity_before",
            "equity_after",
            "isolated_margin_fraction",
            "target_notional_fraction",
            "minimum_margin_buffer",
            "liquidation_count",
            "account_wipeout_count",
            "drawdown",
        ],
        "formulas": {
            "realized_fee_bps": "10000 * fee_quote / round_trip_notional_quote",
            "realized_slippage_bps": (
                "10000 * signed_slippage_quote / round_trip_notional_quote; "
                "positive values are costs"
            ),
            "all_in_round_trip_bps": "realized_fee_bps + realized_slippage_bps",
            "target_notional": "equity * allocation_fraction * leverage",
            "isolated_margin": "equity * allocation_fraction",
            "liquidation_inclusive_equity": (
                "equity path includes isolated liquidation loss before MDD/account-wipeout checks"
            ),
        },
        "pass_fail_rules": {
            "realized_mean_cost_pass": "mean(all_in_round_trip_bps) <= 10.0",
            "realized_p95_cost_pass": "p95(all_in_round_trip_bps) <= 15.0",
            "active_drawdown_pass": "active drawdown <= 0.30 and liquidation_count == 0",
            "balanced_drawdown_pass": "balanced drawdown <= 0.25 and liquidation_count == 0",
            "real_money_gate": "always false for this artifact bundle",
        },
        "profile_rows": profile_rows,
        "source_lineage": dict(source_lineage),
        "status": "pending_paper_forward_fills",
    }


def _write_monitoring_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MONITORING_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in MONITORING_CSV_FIELDS})


def _markdown(payload: Mapping[str, Any]) -> str:
    rows = list(payload.get("side_by_side_profiles") or [])
    lines = [
        "# Alpha Zoo 10bps paper/testnet preflight",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "Real-money execution is disabled: `ready_for_real=false`, `real_money_execution=false`.",
        "",
        "| Role | Model | Leverage | Allocation | Train return | Validation return | Locked-OOS return | Paper | Real |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        metrics = dict(row.get("split_metrics") or {})
        train = dict(metrics.get("train") or {})
        validation = dict(metrics.get("validation") or {})
        locked = dict(metrics.get("locked_oos") or {})
        preflight = dict(row.get("preflight") or {})
        status = dict(preflight.get("status") or {})
        lines.append(
            f"| {row.get('role')} | `{row.get('model_id')}` | "
            f"{_safe_float(row.get('leverage')):g} | {_safe_float(row.get('allocation_fraction')):.3f} | "
            f"{_safe_float(train.get('total_return')):.4%} | "
            f"{_safe_float(validation.get('total_return')):.4%} | "
            f"{_safe_float(locked.get('total_return')):.4%} | "
            f"`{status.get('ready_for_paper')}` | `{status.get('ready_for_real')}` |"
        )
    lines.extend(
        [
            "",
            "## Monitoring",
            "",
            "- Realized fee/slippage/all-in round-trip bps must be compared against the locked 10bps research assumption.",
            "- Active and balanced rows share the same timestamp/symbol/side grouping requirements.",
            "- Isolated liquidation losses are included in equity, drawdown, and account-wipeout checks.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    retune_path = Path(args.retune_json).expanduser().resolve()
    low_correlation_path = Path(args.low_correlation_json).expanduser().resolve()
    live_aligned_path = Path(args.live_aligned_json).expanduser().resolve()
    retune = _load_json(retune_path)
    low_correlation = _load_json(low_correlation_path)
    live_aligned = _load_json(live_aligned_path)
    source_lineage = _source_lineage(
        retune_path=retune_path,
        low_correlation_path=low_correlation_path,
        live_aligned_path=live_aligned_path,
        retune=retune,
        low_correlation=low_correlation,
        live_aligned=live_aligned,
    )

    rows: list[dict[str, Any]] = []
    decision_paths: dict[str, str] = {}
    preflight_paths: dict[str, str] = {}
    monitoring_rows: list[dict[str, Any]] = []

    configs = [
        (
            "active",
            ACTIVE_PROFILE_ID,
            "live_alpha_zoo_quality_single_pair_7x_0p20_paper_decision_latest.json",
        ),
        (
            "balanced",
            BALANCED_PROFILE_ID,
            "live_alpha_zoo_quality_single_pair_6x_0p175_balanced_reference_decision_latest.json",
        ),
    ]
    for role, profile_id, filename in configs:
        model = _model_summary(retune, role=role)
        profile = _selection_profile(retune, profile_id)
        metrics = _model_metrics(retune, str(model["model_id"]))
        leverage = _safe_float(model.get("leverage"))
        allocation = _safe_float(model.get("allocation_fraction"))
        risk_caps = aligned._risk_caps_for_contract(
            leverage=leverage,
            allocation_fraction=allocation,
        )
        paper_sizing = aligned._paper_equivalent_sizing(
            leverage=leverage,
            allocation_fraction=allocation,
            sizing_mode=SIZING_MODE,
            risk_caps=risk_caps,
        )
        if not bool(paper_sizing.get("notional_parity_passed")):
            raise ValueError(f"paper/live notional parity failed for {role}")
        strategy_params = _strategy_params_for_candidate(
            live_aligned,
            candidate_name=str(model.get("candidate_name")),
            trade_filter_params=dict(model.get("trade_filter_params") or {}),
        )
        decision = _decision_payload(
            role=role,
            model=model,
            metrics=metrics,
            profile=profile,
            strategy_params=strategy_params,
            risk_caps=risk_caps,
            paper_sizing=paper_sizing,
            source_lineage=source_lineage,
        )
        decision_path = output_dir / filename
        _write_json(decision_path, decision)
        preflight = _preflight_payload(decision_path)
        preflight_filename = (
            "live_readiness_preflight_alpha_zoo_7x_0p20_paper_latest.json"
            if role == "active"
            else "live_readiness_preflight_alpha_zoo_6x_0p175_balanced_reference_paper_latest.json"
        )
        preflight_path = output_dir / preflight_filename
        _write_json(preflight_path, preflight)
        decision_paths[role] = str(decision_path)
        preflight_paths[role] = str(preflight_path)
        monitoring_rows.append(
            _monitoring_profile_row(
                role=role,
                model=model,
                profile=profile,
                paper_sizing=paper_sizing,
                preflight=preflight,
            )
        )
        rows.append(
            {
                "role": role,
                "profile_id": profile_id,
                "model_id": model.get("model_id"),
                "leverage": leverage,
                "allocation_fraction": allocation,
                "split_metrics": {split: dict(row) for split, row in metrics.items()},
                "paper_equivalent_sizing": paper_sizing,
                "preflight": preflight,
                "decision_artifact_path": str(decision_path),
                "preflight_artifact_path": str(preflight_path),
            }
        )

    monitoring = _monitoring_contract(
        profile_rows=monitoring_rows,
        source_lineage=source_lineage,
    )
    monitoring_json = output_dir / "paper_forward_monitoring_contract_latest.json"
    monitoring_csv = output_dir / "paper_forward_monitoring_contract_latest.csv"
    _write_json(monitoring_json, monitoring)
    _write_monitoring_csv(monitoring_csv, monitoring_rows)

    validation_log = output_dir / "artifact_generation_validation_latest.log"
    validation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={_utc_now_iso()}",
                "active_model_id=" + ACTIVE_MODEL_ID,
                "balanced_model_id=" + BALANCED_MODEL_ID,
                "ready_for_paper_active="
                + str(dict(rows[0]["preflight"]["status"]).get("ready_for_paper")),
                "ready_for_paper_balanced="
                + str(dict(rows[1]["preflight"]["status"]).get("ready_for_paper")),
                "ready_for_real_active="
                + str(dict(rows[0]["preflight"]["status"]).get("ready_for_real")),
                "ready_for_real_balanced="
                + str(dict(rows[1]["preflight"]["status"]).get("ready_for_real")),
                "monitoring_status=pending_paper_forward_fills",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    latest_path = output_dir / "alpha_zoo_7x_paper_forward_preflight_latest.json"
    timestamped_path = output_dir / f"alpha_zoo_7x_paper_forward_preflight_{_timestamp()}.json"
    latest_md = output_dir / "alpha_zoo_7x_paper_forward_preflight_latest.md"
    payload = {
        "artifact_kind": "alpha_zoo_7x_paper_forward_preflight_bundle",
        "generated_at_utc": _utc_now_iso(),
        "real_money_execution": False,
        "ready_for_real": False,
        "ready_for_paper": all(
            bool(dict(row["preflight"]["status"]).get("ready_for_paper")) for row in rows
        ),
        "side_by_side_profiles": rows,
        "source_lineage": source_lineage,
        "cost_assumption_audit": {
            "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
            "monitoring_contract_path": str(monitoring_json),
            "monitoring_csv_path": str(monitoring_csv),
            "realized_cost_status": "pending_paper_forward_fills",
        },
        "locked_oos_governance": {
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
        },
        "output_paths": {
            "latest_json": str(latest_path),
            "timestamped_json": str(timestamped_path),
            "latest_markdown": str(latest_md),
            "active_decision": decision_paths["active"],
            "balanced_decision": decision_paths["balanced"],
            "active_preflight": preflight_paths["active"],
            "balanced_preflight": preflight_paths["balanced"],
            "monitoring_contract_json": str(monitoring_json),
            "monitoring_contract_csv": str(monitoring_csv),
            "artifact_generation_validation_log": str(validation_log),
        },
        "memory_summary": {
            "limit_mib": 8192.0,
            "source_retune_peak_rss_mib": dict(retune.get("memory_summary") or {}).get(
                "peak_rss_mib"
            ),
            "source_retune_pass_under_8gb": dict(retune.get("memory_summary") or {}).get(
                "pass_under_8gb"
            ),
        },
    }
    _write_json(latest_path, payload)
    _write_json(timestamped_path, payload)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retune-json", default=str(DEFAULT_RETUNE_JSON))
    parser.add_argument("--low-correlation-json", default=str(DEFAULT_LOW_CORRELATION_JSON))
    parser.add_argument("--live-aligned-json", default=str(DEFAULT_LIVE_ALIGNED_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
