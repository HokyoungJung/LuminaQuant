#!/usr/bin/env python3
"""Blend the three integer-leverage paper profiles into one hybrid candidate.

This runner is research/paper-testnet only. It consumes the frozen
``alpha_zoo_corr_integer_leverage_portfolio`` artifact, reconstructs the
10bps-costed profile PnL streams, and selects a three-profile hybrid using
train+validation evidence only. locked-OOS remains gate/report-only after the
hybrid weights are frozen. No order execution is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_corr_integer_leverage_portfolio as ilp  # noqa: E402

DEFAULT_INTEGER_PORTFOLIO_ARTIFACT = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_corr_integer_leverage_portfolio_20260524/alpha_zoo_corr_integer_leverage_portfolio_latest.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_integer_leverage_hybrid_decision_20260524"
)

ARTIFACT_KIND = "alpha_zoo_integer_leverage_hybrid_decision"
HYBRID_PROFILE_ID = "hybrid_mdd20_three_profile_blend"
WEIGHT_STEP = 0.05
MIN_PROFILE_WEIGHT = 0.10
MAX_VALIDATION_MDD = 0.20
MAX_TRAIN_MDD = 0.45
MAX_GROSS_NOTIONAL = 8.0
MIN_VALIDATION_RETURN = 0.02

COMPARISON_FIELDS = [
    "profile_id",
    "profile_kind",
    "candidate_tier",
    "weights",
    "leverage_map",
    "gross_notional_fraction",
    "train_return",
    "validation_return",
    "locked_oos_return_report_only",
    "train_mdd",
    "validation_mdd",
    "locked_oos_mdd_report_only",
    "train_trade_event_count",
    "validation_trade_event_count",
    "locked_oos_trade_event_count_report_only",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "locked_oos_return_per_turnover_proxy_bps_report_only",
    "train_liquidation_count",
    "validation_liquidation_count",
    "locked_oos_liquidation_count_report_only",
    "train_account_wipeout_count",
    "validation_account_wipeout_count",
    "locked_oos_account_wipeout_count_report_only",
    "train_validation_score",
    "paper_testnet_candidate",
    "strict_promotion_profile",
    "promotion_gate_pass",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "selection_reasons",
    "report_only_gate_reasons",
]


@dataclass(frozen=True)
class ProfileStream:
    profile_id: str
    candidate_tier: str
    leverage_map: Mapping[str, int]
    gross_notional_fraction: float
    asset_gross_notional_fraction: Mapping[str, float]
    selected_model_ids: tuple[str, ...]
    returns: pd.Series
    turnover_by_split: Mapping[str, float]
    trade_events_by_split: Mapping[str, int]
    liquidation_count_by_split: Mapping[str, int]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    return ilp._json_safe(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return _json_safe(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _period_return(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def _split_returns(series: pd.Series, split: str) -> pd.Series:
    return series.loc[ilp._split_mask(series.index, split)]


def _metric_row_from_stream(
    *,
    profile_id: str,
    profile_kind: str,
    candidate_tier: str,
    leverage_map: Mapping[str, int] | None,
    weights: Mapping[str, float],
    gross_notional_fraction: float,
    returns: pd.Series,
    turnover_by_split: Mapping[str, float],
    trade_events_by_split: Mapping[str, int],
    liquidation_count_by_split: Mapping[str, int],
    strict_promotion_profile: bool,
    promotion_gate_pass: bool,
    paper_testnet_candidate: bool,
    selection_reasons: Sequence[str] | None = None,
    report_only_gate_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "profile_id": profile_id,
        "profile_kind": profile_kind,
        "candidate_tier": candidate_tier,
        "weights": dict(weights),
        "leverage_map": dict(leverage_map or {}),
        "gross_notional_fraction": gross_notional_fraction,
        "paper_testnet_candidate": paper_testnet_candidate,
        "strict_promotion_profile": strict_promotion_profile,
        "promotion_gate_pass": promotion_gate_pass,
        "ready_for_paper": paper_testnet_candidate,
        "ready_for_real": False,
        "real_money_execution": False,
        "selection_reasons": list(selection_reasons or []),
        "report_only_gate_reasons": list(report_only_gate_reasons or []),
    }
    for split in ilp.SPLIT_ORDER:
        split_series = _split_returns(returns, split)
        values = split_series.to_numpy(dtype=float)
        turnover = _safe_float(turnover_by_split.get(split))
        total_return = _period_return(values)
        equity = np.cumprod(1.0 + values) if values.size else np.asarray([], dtype=float)
        row[f"{split}_return" if split != "locked_oos" else "locked_oos_return_report_only"] = (
            total_return
        )
        row[f"{split}_mdd" if split != "locked_oos" else "locked_oos_mdd_report_only"] = (
            ilp.max_drawdown(values)
        )
        row[
            f"{split}_trade_event_count"
            if split != "locked_oos"
            else "locked_oos_trade_event_count_report_only"
        ] = int(trade_events_by_split.get(split, 0))
        row[
            f"{split}_return_per_turnover_proxy_bps"
            if split != "locked_oos"
            else "locked_oos_return_per_turnover_proxy_bps_report_only"
        ] = total_return * 10_000.0 / turnover if turnover > 0.0 else None
        row[
            f"{split}_liquidation_count"
            if split != "locked_oos"
            else "locked_oos_liquidation_count_report_only"
        ] = int(liquidation_count_by_split.get(split, 0))
        row[
            f"{split}_account_wipeout_count"
            if split != "locked_oos"
            else "locked_oos_account_wipeout_count_report_only"
        ] = int(np.count_nonzero(equity <= 0.0)) if equity.size else 0
    row["train_validation_score"] = _train_validation_score(row)
    return row


def _train_validation_score(row: Mapping[str, Any]) -> float:
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    validation_mdd = _safe_float(row.get("validation_mdd"))
    train_mdd = _safe_float(row.get("train_mdd"))
    validation_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"))
    gross = _safe_float(row.get("gross_notional_fraction"))
    validation_spike_penalty = max(0.0, validation - train)
    return (
        9.0 * validation
        + 1.5 * min(train, validation)
        + min(validation_rpt, 80.0) / 160.0
        + min(train_rpt, 80.0) / 260.0
        - 8.0 * validation_spike_penalty
        - 2.0 * validation_mdd
        - 0.75 * train_mdd
        - 0.02 * gross
    )


def _selection_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    if train <= 0.0:
        reasons.append("train_return_not_positive")
    if validation < MIN_VALIDATION_RETURN:
        reasons.append(f"validation_return_{validation:.4f}_below_{MIN_VALIDATION_RETURN:.4f}")
    if train < validation:
        reasons.append(f"train_return_{train:.4f}_below_validation_return_{validation:.4f}")
    if _safe_float(row.get("validation_mdd")) > MAX_VALIDATION_MDD:
        reasons.append(
            f"validation_mdd_{_safe_float(row.get('validation_mdd')):.4f}_above_{MAX_VALIDATION_MDD:.4f}"
        )
    if _safe_float(row.get("train_mdd")) > MAX_TRAIN_MDD:
        reasons.append(
            f"train_mdd_{_safe_float(row.get('train_mdd')):.4f}_above_{MAX_TRAIN_MDD:.4f}"
        )
    if _safe_float(row.get("gross_notional_fraction")) > MAX_GROSS_NOTIONAL:
        reasons.append(
            f"gross_notional_{_safe_float(row.get('gross_notional_fraction')):.4f}_above_{MAX_GROSS_NOTIONAL:.4f}"
        )
    for split in ("train", "validation"):
        if int(row.get(f"{split}_liquidation_count") or 0) != 0:
            reasons.append(f"{split}_liquidation_count_nonzero")
        if int(row.get(f"{split}_account_wipeout_count") or 0) != 0:
            reasons.append(f"{split}_account_wipeout_count_nonzero")
        rpt = row.get(f"{split}_return_per_turnover_proxy_bps")
        if rpt is None or _safe_float(rpt) <= ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS:
            rendered = "missing" if rpt is None else f"{_safe_float(rpt):.3f}"
            reasons.append(f"{split}_return_per_turnover_proxy_bps_{rendered}_not_above_10bps")
    return reasons


def _report_only_gate_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _safe_float(row.get("locked_oos_return_report_only")) <= 0.0:
        reasons.append("locked_oos_return_not_positive_report_only")
    if int(row.get("locked_oos_liquidation_count_report_only") or 0) != 0:
        reasons.append("locked_oos_liquidation_count_nonzero_report_only")
    if int(row.get("locked_oos_account_wipeout_count_report_only") or 0) != 0:
        reasons.append("locked_oos_account_wipeout_count_nonzero_report_only")
    if (
        int(row.get("locked_oos_trade_event_count_report_only") or 0)
        < ilp.MIN_LOCKED_OOS_TRADE_EVENTS
    ):
        reasons.append(
            "locked_oos_trade_event_count_"
            f"{int(row.get('locked_oos_trade_event_count_report_only') or 0)}_below_"
            f"{ilp.MIN_LOCKED_OOS_TRADE_EVENTS}_report_only"
        )
    rpt = row.get("locked_oos_return_per_turnover_proxy_bps_report_only")
    if rpt is None or _safe_float(rpt) <= ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        rendered = "missing" if rpt is None else f"{_safe_float(rpt):.3f}"
        reasons.append(
            f"locked_oos_return_per_turnover_proxy_bps_{rendered}_not_above_10bps_report_only"
        )
    return reasons


def _iter_weight_grid(
    profile_ids: Sequence[str], *, step: float = WEIGHT_STEP, min_weight: float = MIN_PROFILE_WEIGHT
):
    units = round(1.0 / step)
    min_units = round(min_weight / step)
    if len(profile_ids) != 3:
        raise ValueError("hybrid decision expects exactly three source profiles")
    for first in range(min_units, units + 1):
        for second in range(min_units, units + 1 - first):
            third = units - first - second
            if third < min_units:
                continue
            weights = (first / units, second / units, third / units)
            yield dict(zip(profile_ids, weights, strict=True))


def _weighted_hybrid_row(
    profile_streams: Sequence[ProfileStream], weights: Mapping[str, float]
) -> dict[str, Any]:
    by_id = {stream.profile_id: stream for stream in profile_streams}
    if set(weights) != set(by_id):
        raise ValueError("hybrid weights must cover exactly the source profiles")
    returns = sum(by_id[profile_id].returns * weight for profile_id, weight in weights.items())
    turnover_by_split = {
        split: float(
            sum(
                by_id[profile_id].turnover_by_split[split] * weight
                for profile_id, weight in weights.items()
            )
        )
        for split in ilp.SPLIT_ORDER
    }
    trade_events_by_split = {
        split: int(sum(by_id[profile_id].trade_events_by_split[split] for profile_id in weights))
        for split in ilp.SPLIT_ORDER
    }
    liquidation_count_by_split = {
        split: int(
            sum(by_id[profile_id].liquidation_count_by_split[split] for profile_id in weights)
        )
        for split in ilp.SPLIT_ORDER
    }
    asset_gross: dict[str, float] = defaultdict(float)
    for profile_id, weight in weights.items():
        for asset, gross in by_id[profile_id].asset_gross_notional_fraction.items():
            asset_gross[asset] += _safe_float(gross) * weight
    row = _metric_row_from_stream(
        profile_id=HYBRID_PROFILE_ID,
        profile_kind="hybrid_train_validation_selected",
        candidate_tier="hybrid_relaxed_paper_testnet_candidate",
        leverage_map={},
        weights=weights,
        gross_notional_fraction=float(
            sum(
                by_id[profile_id].gross_notional_fraction * weight
                for profile_id, weight in weights.items()
            )
        ),
        returns=returns,
        turnover_by_split=turnover_by_split,
        trade_events_by_split=trade_events_by_split,
        liquidation_count_by_split=liquidation_count_by_split,
        strict_promotion_profile=False,
        promotion_gate_pass=False,
        paper_testnet_candidate=False,
    )
    selection_reasons = _selection_reasons(row)
    report_only_reasons = _report_only_gate_reasons(row)
    row["asset_gross_notional_fraction"] = dict(sorted(asset_gross.items()))
    row["paper_testnet_candidate"] = not selection_reasons and not report_only_reasons
    row["ready_for_paper"] = row["paper_testnet_candidate"]
    row["selection_reasons"] = selection_reasons
    row["report_only_gate_reasons"] = report_only_reasons
    return row


def select_hybrid_row(
    profile_streams: Sequence[ProfileStream],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile_ids = [stream.profile_id for stream in profile_streams]
    candidates: list[dict[str, Any]] = []
    for weights in _iter_weight_grid(profile_ids):
        row = _weighted_hybrid_row(profile_streams, weights)
        candidates.append(row)
    train_validation_pass = [row for row in candidates if not row["selection_reasons"]]
    pool = train_validation_pass or candidates
    selected = max(pool, key=lambda row: _safe_float(row.get("train_validation_score"), -1e9))
    # OOS does not participate in the selection pool or score; it is attached after the frozen choice.
    return selected, sorted(
        candidates,
        key=lambda row: _safe_float(row.get("train_validation_score"), -1e9),
        reverse=True,
    )


def _profile_stream_from_row(
    profile_row: Mapping[str, Any],
    *,
    replays_by_model_id: Mapping[str, ilp.CandidateReplay],
    union_index: pd.DatetimeIndex,
) -> ProfileStream:
    leverage_map = {
        str(asset): int(value) for asset, value in dict(profile_row["leverage_map"]).items()
    }
    returns = pd.Series(0.0, index=union_index, dtype=float)
    turnover_by_split = dict.fromkeys(ilp.SPLIT_ORDER, 0.0)
    trade_events_by_split = dict.fromkeys(ilp.SPLIT_ORDER, 0)
    liquidation_count_by_split = dict.fromkeys(ilp.SPLIT_ORDER, 0)
    asset_gross: dict[str, float] = defaultdict(float)
    selected_model_ids = tuple(str(model_id) for model_id in profile_row["selected_model_ids"])
    for model_id in selected_model_ids:
        replay = replays_by_model_id[model_id]
        sim = ilp.simulate_candidate_with_integer_leverage(
            replay, integer_leverage=leverage_map[replay.symbol]
        )
        returns = returns.add(
            pd.Series(sim.returns, index=sim.datetimes, dtype=float).reindex(
                union_index, fill_value=0.0
            )
        )
        asset_gross[sim.symbol] += sim.notional_fraction
        for split in ilp.SPLIT_ORDER:
            trade_events = ilp._trade_count_for_split(sim, split)
            trade_events_by_split[split] += trade_events
            turnover_by_split[split] += trade_events * abs(sim.notional_fraction)
            split_mask = ilp._split_mask(sim.datetimes, split)
            liquidation_count_by_split[split] += int(
                np.count_nonzero(sim.liquidation_flags[split_mask])
            )
    return ProfileStream(
        profile_id=str(profile_row["profile_id"]),
        candidate_tier=str(profile_row.get("candidate_tier", "")),
        leverage_map=leverage_map,
        gross_notional_fraction=float(sum(asset_gross.values())),
        asset_gross_notional_fraction=dict(sorted(asset_gross.items())),
        selected_model_ids=selected_model_ids,
        returns=returns.sort_index(),
        turnover_by_split=turnover_by_split,
        trade_events_by_split=trade_events_by_split,
        liquidation_count_by_split=liquidation_count_by_split,
    )


def _base_comparison_row(stream: ProfileStream, source_row: Mapping[str, Any]) -> dict[str, Any]:
    return _metric_row_from_stream(
        profile_id=stream.profile_id,
        profile_kind="source_integer_leverage_profile",
        candidate_tier=stream.candidate_tier,
        leverage_map=stream.leverage_map,
        weights={stream.profile_id: 1.0},
        gross_notional_fraction=stream.gross_notional_fraction,
        returns=stream.returns,
        turnover_by_split=stream.turnover_by_split,
        trade_events_by_split=stream.trade_events_by_split,
        liquidation_count_by_split=stream.liquidation_count_by_split,
        strict_promotion_profile=bool(source_row.get("strict_promotion_profile")),
        promotion_gate_pass=bool(source_row.get("promotion_gate_pass")),
        paper_testnet_candidate=bool(source_row.get("paper_testnet_candidate")),
        selection_reasons=[],
        report_only_gate_reasons=[],
    )


def _profile_corr_matrix(
    profile_streams: Sequence[ProfileStream], *, split: str
) -> dict[str, dict[str, float | None]]:
    frame = pd.DataFrame(
        {stream.profile_id: stream.returns for stream in profile_streams}
    ).sort_index()
    if split == "train_validation":
        mask = ilp._split_mask(frame.index, "train") | ilp._split_mask(frame.index, "validation")
    else:
        mask = ilp._split_mask(frame.index, split)
    frame = frame.loc[mask]
    corr = frame.corr()
    out: dict[str, dict[str, float | None]] = {}
    for left in corr.index:
        out[str(left)] = {}
        for right in corr.columns:
            value = corr.loc[left, right]
            out[str(left)][str(right)] = None if pd.isna(value) else float(value)
    return out


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Integer-Leverage Three-Profile Hybrid Decision",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Method",
        "",
        "- Source profiles: strict balanced, relaxed growth, relaxed aggressive from the frozen integer-leverage artifact.",
        "- Reconstructs the 10bps-costed profile PnL streams; no order execution.",
        "- Searches three-profile weights on a 5% grid with at least 10% allocated to each source profile.",
        "- Selects the hybrid using train+validation score only with a 20% validation-MDD target.",
        "- locked-OOS is attached after the hybrid weights are frozen as gate/report-only evidence.",
        "",
        "## Four-profile comparison",
        "",
        "| Profile | Kind | Weights | Gross | Train | Val | OOS report-only | Val MDD | OOS MDD | RPT T/V/OOS bps | Paper candidate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["comparison_rows"]:
        weights = row.get("weights") or {}
        lines.append(
            f"| `{row['profile_id']}` | {row['profile_kind']} | `{json.dumps(weights, sort_keys=True)}` | "
            f"{_safe_float(row['gross_notional_fraction']):.2f}x | "
            f"{_safe_float(row['train_return']):.4%} | "
            f"{_safe_float(row['validation_return']):.4%} | "
            f"{_safe_float(row['locked_oos_return_report_only']):.4%} | "
            f"{_safe_float(row['validation_mdd']):.4%} | "
            f"{_safe_float(row['locked_oos_mdd_report_only']):.4%} | "
            f"{_safe_float(row['train_return_per_turnover_proxy_bps']):.2f}/"
            f"{_safe_float(row['validation_return_per_turnover_proxy_bps']):.2f}/"
            f"{_safe_float(row['locked_oos_return_per_turnover_proxy_bps_report_only']):.2f} | "
            f"{str(bool(row['paper_testnet_candidate'])).lower()} |"
        )
    selected = payload["selected_hybrid_profile"]
    lines.extend(
        [
            "",
            "## Selected hybrid",
            "",
            f"- profile: `{selected['profile_id']}`",
            f"- weights: `{json.dumps(selected['weights'], sort_keys=True)}`",
            f"- train/validation/OOS report-only: `{_safe_float(selected['train_return']):.4%}` / "
            f"`{_safe_float(selected['validation_return']):.4%}` / "
            f"`{_safe_float(selected['locked_oos_return_report_only']):.4%}`",
            f"- validation MDD / OOS MDD: `{_safe_float(selected['validation_mdd']):.4%}` / "
            f"`{_safe_float(selected['locked_oos_mdd_report_only']):.4%}`",
            f"- report-only OOS gate reasons: `{selected['report_only_gate_reasons']}`",
            "",
            "## Governance",
            "",
            f"- primary round-trip cost bps: `{payload['research_primary_round_trip_cost_bps']}`",
            f"- return-per-turnover threshold bps: `{payload['return_per_turnover_threshold_bps']}`",
            "- ready_for_real: `false`",
            "- real_money_execution: `false`",
            "- real_execution_allowed: `false`",
            f"- locked-OOS used for selection: `{payload['selection_policy']['uses_locked_oos_for_selection']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload_from_inputs(
    *,
    integer_payload: Mapping[str, Any],
    output_dir: Path,
    integer_artifact_path: Path,
    data_root: Path,
    feature_root: Path,
    write_outputs: bool = True,
) -> dict[str, Any]:
    if (
        integer_payload.get("ready_for_real") is not False
        or integer_payload.get("real_money_execution") is not False
    ):
        raise ValueError("integer portfolio artifact violates real-money disabled guard")
    if _safe_float(integer_payload.get("research_primary_round_trip_cost_bps")) != 10.0:
        raise ValueError(
            "integer portfolio artifact is not using the primary 10bps round-trip cost"
        )
    source_profile_rows = list(integer_payload.get("paper_testnet_candidate_profiles") or [])
    if len(source_profile_rows) != 3:
        raise ValueError(
            f"expected exactly three paper/testnet source profiles, found {len(source_profile_rows)}"
        )

    correlation_payload = ilp._load_json(ilp.DEFAULT_CORRELATION_ARTIFACT)
    monitoring_payload = ilp._load_json(ilp.DEFAULT_MONITORING_ARTIFACT)
    ilp._assert_governance(correlation_payload, monitoring_payload)
    selected_rows = ilp._selected_rows_from_corr_payload(correlation_payload)
    captures = ilp.corr.capture_pnl_series(
        selected_rows,
        data_root=data_root,
        feature_root=feature_root,
        monitoring_payload=monitoring_payload,
    )
    bars_by_key = ilp._load_bars_for_rows(selected_rows, data_root=data_root)
    replays = ilp.build_candidate_replays(selected_rows, captures, bars_by_key=bars_by_key)
    replays_by_model_id = {replay.model_id: replay for replay in replays}
    union_index = pd.DatetimeIndex(
        sorted(set().union(*(set(replay.datetimes) for replay in replays)))
    )
    profile_streams = [
        _profile_stream_from_row(
            row, replays_by_model_id=replays_by_model_id, union_index=union_index
        )
        for row in source_profile_rows
    ]
    base_rows = [
        _base_comparison_row(stream, source_row)
        for stream, source_row in zip(profile_streams, source_profile_rows, strict=True)
    ]
    selected_hybrid, candidate_rows = select_hybrid_row(profile_streams)
    comparison_rows = [*base_rows, selected_hybrid]
    timestamp = _timestamp()
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    latest_json = output_dir / "alpha_zoo_integer_leverage_hybrid_decision_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_integer_leverage_hybrid_decision_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_integer_leverage_hybrid_decision_latest.md"
    comparison_csv = output_dir / "integer_leverage_hybrid_comparison_latest.csv"
    candidate_csv = output_dir / "integer_leverage_hybrid_weight_candidates_latest.csv"
    methodology_md = output_dir / "integer_leverage_hybrid_methodology_latest.md"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    payload: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at_utc": _utc_now_iso(),
        "source_integer_portfolio_artifact": str(integer_artifact_path),
        "research_primary_round_trip_cost_bps": ilp.PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": ilp.AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "bbo_spread_multiplier": ilp.BBO_SPREAD_MULTIPLIER,
        "return_per_turnover_threshold_bps": ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "ready_for_paper": bool(selected_hybrid.get("paper_testnet_candidate")),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "paper_testnet_only": True,
        "hybrid_weight_policy": {
            "profile_ids": [stream.profile_id for stream in profile_streams],
            "weight_step": WEIGHT_STEP,
            "min_profile_weight": MIN_PROFILE_WEIGHT,
            "max_validation_mdd": MAX_VALIDATION_MDD,
            "max_train_mdd": MAX_TRAIN_MDD,
            "max_gross_notional": MAX_GROSS_NOTIONAL,
            "selection_inputs": ["train", "validation"],
            "locked_oos_role": "gate/report-only after train+validation hybrid weights freeze",
        },
        "selection_policy": {
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "no_calendar_date_hack": True,
            "model_id_source": "source integer profile rows derived from frozen corr decision artifact",
        },
        "selected_hybrid_profile": selected_hybrid,
        "comparison_rows": comparison_rows,
        "top_hybrid_weight_candidates": candidate_rows[:20],
        "profile_train_validation_corr_matrix": _profile_corr_matrix(
            profile_streams, split="train_validation"
        ),
        "profile_validation_corr_matrix": _profile_corr_matrix(profile_streams, split="validation"),
        "profile_locked_oos_corr_matrix_report_only": _profile_corr_matrix(
            profile_streams, split="locked_oos"
        ),
        "runner_peak_rss_mib": local_peak_mib,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "comparison_csv": str(comparison_csv),
            "weight_candidates_csv": str(candidate_csv),
            "methodology_markdown": str(methodology_md),
            "artifact_generation_validation_log": str(generation_log),
        },
    }
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(latest_json, payload)
        _write_json(timestamped_json, payload)
        latest_md.write_text(_render_markdown(payload), encoding="utf-8")
        _write_csv(comparison_csv, comparison_rows, COMPARISON_FIELDS)
        _write_csv(candidate_csv, candidate_rows, COMPARISON_FIELDS)
        methodology_md.write_text(
            "# Integer-Leverage Hybrid Methodology\n\n"
            "- Source: three paper/testnet profiles from the frozen integer-leverage artifact.\n"
            "- PnL streams: reconstructed from the same fixed candidate position states and integer asset leverage maps.\n"
            "- Cost: 10bps all-in round-trip backtest friction proxy is already embedded in each stream.\n"
            "- Weight selection: train+validation only, 5% grid, each source profile weight >=10%, validation MDD target <=20%.\n"
            "- locked-OOS: not used for discovery, objective, pruning, parameter fitting, or weight selection; report/gate only after freeze.\n"
            "- Real money: blocked. Paper/testnet monitoring must record BBO spread, all-in fee/slippage, liquidation-inclusive MDD, account wipeout, and replay/live notional parity.\n",
            encoding="utf-8",
        )
        generation_log.write_text(
            f"artifact_kind={ARTIFACT_KIND}\n"
            f"source_profile_count={len(source_profile_rows)}\n"
            f"selected_hybrid_profile={selected_hybrid['profile_id']}\n"
            f"selected_hybrid_weights={json.dumps(selected_hybrid['weights'], sort_keys=True)}\n"
            f"ready_for_paper={payload['ready_for_paper']}\n"
            f"ready_for_real={payload['ready_for_real']}\n"
            f"real_money_execution={payload['real_money_execution']}\n"
            f"locked_oos_used_for_selection={payload['selection_policy']['uses_locked_oos_for_selection']}\n"
            f"runner_peak_rss_mib={local_peak_mib:.2f}\n",
            encoding="utf-8",
        )
    return payload


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    integer_artifact = Path(args.integer_portfolio_artifact).expanduser().resolve()
    return build_payload_from_inputs(
        integer_payload=ilp._load_json(integer_artifact),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        integer_artifact_path=integer_artifact,
        data_root=Path(args.data_root).expanduser().resolve(),
        feature_root=Path(args.feature_root).expanduser().resolve(),
        write_outputs=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integer-portfolio-artifact", default=str(DEFAULT_INTEGER_PORTFOLIO_ARTIFACT)
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(ilp.DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(ilp.DEFAULT_FEATURE_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    selected = payload["selected_hybrid_profile"]
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "selected_hybrid_profile": selected,
                    "ready_for_paper": payload["ready_for_paper"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
