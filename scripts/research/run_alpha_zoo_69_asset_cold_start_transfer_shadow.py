#!/usr/bin/env python3
"""Evaluate donor-prior cold-start transfers for train-ineligible 69-asset symbols.

This runner is deliberately shadow/report-only.  It answers whether recently
listed symbols that have validation bars but no train-window bars would have
looked useful when initialized from similar, train-eligible donor profiles.

Safety contract:
* donor selection uses donor train/validation quality, static domain similarity,
  and target data coverage only;
* target validation PnL is never used to choose the donor in the primary lane;
* target train metrics are never synthesized from donor performance;
* no candidate can be promoted to live/paper/real from this artifact.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.optimization.search_policy import optimization_search_policy_payload  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as profile69  # noqa: E402

DEFAULT_SOURCE_ARTIFACT = (
    broad69.ALPHA_V2_ROOT
    / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_20260530"
    / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json"
)
DEFAULT_STRICT_REFERENCE_ARTIFACT = (
    broad69.ALPHA_V2_ROOT
    / "alpha_zoo_69_asset_efficiency_repair_optuna_20260530"
    / "alpha_zoo_69_asset_efficiency_repair_optuna_latest.json"
)
DEFAULT_OUTPUT_DIR = (
    broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_cold_start_transfer_shadow_20260531"
)

PRIMARY_MAX_GROSS = 2.0
PRIMARY_MAX_SLEEVES = 18
ORACLE_MAX_GROSS = 2.0
ORACLE_MAX_SLEEVES = 18
MIN_TARGET_VALIDATION_ROWS = 24

CANDIDATE_FIELDS = [
    "shadow_rank",
    "model_id",
    "profile_id",
    "source_profile_id",
    "symbol",
    "asset_group",
    "family",
    "timeframe",
    "side",
    "lookback_bars",
    "threshold",
    "exit_threshold",
    "min_hold_bars",
    "cooldown_bars",
    "integer_leverage",
    "notional_fraction",
    "target_train_rows",
    "target_validation_rows",
    "train_return",
    "validation_return",
    "train_mdd",
    "validation_mdd",
    "train_trade_event_count",
    "validation_trade_event_count",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "donor_symbol",
    "donor_asset_group",
    "donor_model_id",
    "donor_selection_score",
    "donor_quality_score",
    "donor_train_return",
    "donor_validation_return",
    "donor_train_return_per_turnover_proxy_bps",
    "donor_validation_return_per_turnover_proxy_bps",
    "shadow_validation_score",
    "primary_shadow_selected",
    "oracle_shadow_selected",
    "shadow_multiplier",
    "weighted_notional_fraction",
    "promotion_status",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "real_execution_allowed",
    "rejection_reasons",
]

PORTFOLIO_FIELDS = [
    "portfolio_id",
    "selection_policy",
    "sleeve_count",
    "gross_notional_fraction",
    "train_return",
    "validation_return",
    "train_mdd",
    "validation_mdd",
    "train_trade_event_count",
    "validation_trade_event_count",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "top_symbol",
    "top_symbol_share",
    "top_asset_group",
    "top_asset_group_share",
    "effective_symbol_count",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "real_execution_allowed",
]

MEGA_CAP_TECH = {
    "AAPLUSDT",
    "MSFTUSDT",
    "AVGOUSDT",
    "AMDUSDT",
    "QCOMUSDT",
    "ORCLUSDT",
    "CSCOUSDT",
    "MRVLUSDT",
    "TSMUSDT",
    "MUUSDT",
    "SNDKUSDT",
    "DRAMUSDT",
    "WDCUSDT",
    "ARMUSDT",
    "COHRUSDT",
    "NVDAUSDT",
    "GOOGLUSDT",
    "METAUSDT",
    "INTCUSDT",
}
SEMICONDUCTORS = {
    "SOXLUSDT",
    "TSMUSDT",
    "MUUSDT",
    "SNDKUSDT",
    "AMDUSDT",
    "QCOMUSDT",
    "MRVLUSDT",
    "DRAMUSDT",
    "WDCUSDT",
    "ARMUSDT",
    "COHRUSDT",
    "NVDAUSDT",
    "INTCUSDT",
    "AVGOUSDT",
}
BROAD_INDEX = {"QQQUSDT", "SPYUSDT", "EWYUSDT", "EWJUSDT"}
CONSUMER_FINANCIAL = {
    "BABAUSDT",
    "DISUSDT",
    "UBERUSDT",
    "HDUSDT",
    "WMTUSDT",
    "JPMUSDT",
    "VUSDT",
    "BRKBUSDT",
    "AMZNUSDT",
    "PAYPUSDT",
    "HOODUSDT",
}
SPECULATIVE_GROWTH = {
    "CRWVUSDT",
    "RKLBUSDT",
    "CBRSUSDT",
    "NBISUSDT",
    "BEUSDT",
    "FLNCUSDT",
    "USARUSDT",
    "LITEUSDT",
    "SPCXUSDT",
    "OPENAIUSDT",
    "QNTXUSDT",
    "PLTRUSDT",
    "TSLAUSDT",
    "MSTRUSDT",
    "COINUSDT",
}

PREFERRED_DONOR_SYMBOLS: dict[str, tuple[str, ...]] = {
    "QQQUSDT": ("NVDAUSDT", "GOOGLUSDT", "METAUSDT", "TSLAUSDT", "EWYUSDT", "EWJUSDT"),
    "SPYUSDT": ("EWYUSDT", "EWJUSDT", "AMZNUSDT", "METAUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "SOXLUSDT": ("NVDAUSDT", "INTCUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "AAPLUSDT": ("NVDAUSDT", "GOOGLUSDT", "METAUSDT", "AMZNUSDT"),
    "MSFTUSDT": ("GOOGLUSDT", "METAUSDT", "NVDAUSDT", "AMZNUSDT"),
    "AVGOUSDT": ("NVDAUSDT", "INTCUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "TSMUSDT": ("NVDAUSDT", "INTCUSDT", "TSLAUSDT", "GOOGLUSDT"),
    "MUUSDT": ("NVDAUSDT", "INTCUSDT", "TSLAUSDT", "GOOGLUSDT"),
    "SNDKUSDT": ("NVDAUSDT", "INTCUSDT", "TSLAUSDT", "GOOGLUSDT"),
    "AMDUSDT": ("NVDAUSDT", "INTCUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "QCOMUSDT": ("NVDAUSDT", "INTCUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "MRVLUSDT": ("NVDAUSDT", "INTCUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "DRAMUSDT": ("NVDAUSDT", "INTCUSDT", "TSLAUSDT", "GOOGLUSDT"),
    "WDCUSDT": ("NVDAUSDT", "INTCUSDT", "TSLAUSDT", "GOOGLUSDT"),
    "ARMUSDT": ("NVDAUSDT", "INTCUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "COHRUSDT": ("NVDAUSDT", "INTCUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "BABAUSDT": ("AMZNUSDT", "METAUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "DISUSDT": ("AMZNUSDT", "METAUSDT", "GOOGLUSDT", "PAYPUSDT"),
    "UBERUSDT": ("AMZNUSDT", "HOODUSDT", "PAYPUSDT", "METAUSDT"),
    "HDUSDT": ("AMZNUSDT", "PAYPUSDT", "METAUSDT", "GOOGLUSDT"),
    "WMTUSDT": ("AMZNUSDT", "PAYPUSDT", "METAUSDT", "GOOGLUSDT"),
    "JPMUSDT": ("HOODUSDT", "PAYPUSDT", "COINUSDT", "AMZNUSDT"),
    "VUSDT": ("PAYPUSDT", "HOODUSDT", "AMZNUSDT", "COINUSDT"),
    "BRKBUSDT": ("AMZNUSDT", "METAUSDT", "GOOGLUSDT", "TSLAUSDT"),
    "ORCLUSDT": ("GOOGLUSDT", "METAUSDT", "NVDAUSDT", "AMZNUSDT"),
    "CSCOUSDT": ("INTCUSDT", "NVDAUSDT", "GOOGLUSDT", "METAUSDT"),
    "USARUSDT": ("PLTRUSDT", "TSLAUSDT", "MSTRUSDT", "COINUSDT"),
    "LITEUSDT": ("PLTRUSDT", "NVDAUSDT", "TSLAUSDT", "COINUSDT"),
    "CRWVUSDT": ("PLTRUSDT", "NVDAUSDT", "TSLAUSDT", "COINUSDT"),
    "RKLBUSDT": ("PLTRUSDT", "TSLAUSDT", "MSTRUSDT", "COINUSDT"),
    "CBRSUSDT": ("PLTRUSDT", "TSLAUSDT", "MSTRUSDT", "COINUSDT"),
    "NBISUSDT": ("PLTRUSDT", "NVDAUSDT", "TSLAUSDT", "MSTRUSDT"),
    "BEUSDT": ("PLTRUSDT", "TSLAUSDT", "MSTRUSDT", "COINUSDT"),
    "FLNCUSDT": ("PLTRUSDT", "TSLAUSDT", "MSTRUSDT", "COINUSDT"),
    "SPCXUSDT": ("PLTRUSDT", "TSLAUSDT", "MSTRUSDT", "COINUSDT"),
    "OPENAIUSDT": ("PLTRUSDT", "NVDAUSDT", "GOOGLUSDT", "METAUSDT"),
    "QNTXUSDT": ("PLTRUSDT", "NVDAUSDT", "TSLAUSDT", "MSTRUSDT"),
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    return broad69._json_safe(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", "utf-8")


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


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if _safe_float(payload.get("research_primary_round_trip_cost_bps"), -1.0) != 10.0:
        raise ValueError("source artifact must use the 10bps primary round-trip cost model")
    if payload.get("ready_for_real") or payload.get("real_money_execution"):
        raise ValueError("source artifact unexpectedly enables real-money execution")
    return payload


def _split_windows_from_payload(payload: Mapping[str, Any]) -> broad69.SplitWindows:
    split = payload["split_policy"]
    return broad69.SplitWindows(
        train=(
            pd.Timestamp(split["train"]["start"]).tz_localize(None),
            pd.Timestamp(split["train"]["end"]).tz_localize(None),
        ),
        validation=(
            pd.Timestamp(split["validation"]["start"]).tz_localize(None),
            pd.Timestamp(split["validation"]["end"]).tz_localize(None),
        ),
    )


def _asset_subgroup(symbol: str) -> str:
    token = str(symbol).upper()
    if token in SEMICONDUCTORS:
        return "semiconductor_ai_compute"
    if token in BROAD_INDEX:
        return "tradfi_broad_index_etf"
    if token in MEGA_CAP_TECH:
        return "mega_cap_tech_platform"
    if token in CONSUMER_FINANCIAL:
        return "consumer_financial_platform"
    if token in SPECULATIVE_GROWTH:
        return "speculative_growth_ai_crypto_equity"
    return broad69._asset_group(token)


def _preferred_donors(symbol: str) -> tuple[str, ...]:
    token = str(symbol).upper()
    return PREFERRED_DONOR_SYMBOLS.get(token, ())


def _timeframe_payload(
    train_eligibility: Mapping[str, Any], symbol: str, timeframe: str
) -> dict[str, Any]:
    symbols = train_eligibility.get("symbols") or {}
    symbol_payload = dict(dict(symbols).get(str(symbol)) or {})
    timeframes = symbol_payload.get("timeframes") or {}
    return dict(dict(timeframes).get(str(timeframe)) or {})


def _row_train_eligible(row: Mapping[str, Any], train_eligibility: Mapping[str, Any]) -> bool:
    payload = _timeframe_payload(
        train_eligibility, str(row.get("symbol") or ""), str(row.get("timeframe") or "")
    )
    return bool(payload.get("train_eligible"))


def _target_validation_rows(
    train_eligibility: Mapping[str, Any], symbol: str, timeframe: str
) -> int:
    return int(_timeframe_payload(train_eligibility, symbol, timeframe).get("validation_rows") or 0)


def _donor_quality_score(row: Mapping[str, Any]) -> float:
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_mdd = _safe_float(row.get("train_mdd"))
    val_mdd = _safe_float(row.get("validation_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    train_events = int(row.get("train_trade_event_count") or 0)
    val_events = int(row.get("validation_trade_event_count") or 0)
    spike = max(0.0, validation - train)
    penalty = 0.0
    if train <= 0.0:
        penalty += 4.0 + abs(train) * 10.0
    if validation <= 0.0:
        penalty += 5.0 + abs(validation) * 12.0
    if train_rpt <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        penalty += (broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS - train_rpt) / 8.0
    if val_rpt <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        penalty += (broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS - val_rpt) / 8.0
    if train_events < 20:
        penalty += (20 - train_events) / 8.0
    if val_events < 6:
        penalty += (6 - val_events) / 4.0
    return float(
        6.0 * min(train, validation)
        + 2.5 * train
        + 4.0 * validation
        + min(max(train_rpt, -50.0), 200.0) / 110.0
        + min(max(val_rpt, -50.0), 200.0) / 90.0
        + _safe_float(row.get("profile_objective_score"))
        - 2.5 * train_mdd
        - 3.5 * val_mdd
        - 6.0 * spike
        - penalty
    )


def _donor_rows(
    source_payload: Mapping[str, Any], train_eligibility: Mapping[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in source_payload.get("asset_tuning_rows", []):
        row = dict(raw)
        params = row.get("optuna_params")
        if not isinstance(params, Mapping) or not params:
            continue
        if not _row_train_eligible(row, train_eligibility):
            continue
        row["donor_quality_score"] = _donor_quality_score(row)
        out.append(row)
    return out


def _donor_similarity_score(
    *,
    target_symbol: str,
    target_profile_id: str,
    donor_row: Mapping[str, Any],
    train_eligibility: Mapping[str, Any],
) -> float:
    params = donor_row.get("optuna_params") or {}
    timeframe = str(params.get("timeframe") or donor_row.get("timeframe") or "")
    validation_rows = _target_validation_rows(train_eligibility, target_symbol, timeframe)
    lookback = int(params.get("lookback_bars") or donor_row.get("lookback_bars") or 1)
    if validation_rows < max(MIN_TARGET_VALIDATION_ROWS, lookback + 5):
        return float("-inf")

    donor_symbol = str(donor_row.get("symbol") or "")
    target_group = broad69._asset_group(target_symbol)
    donor_group = broad69._asset_group(donor_symbol)
    target_subgroup = _asset_subgroup(target_symbol)
    donor_subgroup = _asset_subgroup(donor_symbol)
    preferred = _preferred_donors(target_symbol)
    score = _safe_float(donor_row.get("donor_quality_score"))
    if str(donor_row.get("profile_id")) == target_profile_id:
        score += 30.0
    if donor_symbol in preferred:
        score += 15.0 - min(preferred.index(donor_symbol), 5) * 1.5
    if target_group == donor_group:
        score += 8.0
    if target_subgroup == donor_subgroup:
        score += 8.0
    if target_group.startswith("tradfi") and donor_group.startswith("tradfi"):
        score += 4.0
    if donor_symbol == target_symbol:
        score -= 100.0
    score += min(validation_rows / 1000.0, 2.0)
    return float(score)


def _select_donor(
    *,
    target_symbol: str,
    source_profile_id: str,
    donor_rows: Sequence[Mapping[str, Any]],
    train_eligibility: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, float, list[str]]:
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for donor in donor_rows:
        score = _donor_similarity_score(
            target_symbol=target_symbol,
            target_profile_id=source_profile_id,
            donor_row=donor,
            train_eligibility=train_eligibility,
        )
        if math.isfinite(score):
            scored.append((score, donor))
    if not scored:
        return None, float("-inf"), ["no_train_eligible_donor_with_target_validation_coverage"]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected_score, selected = scored[0]
    donor_symbol = str(selected.get("symbol") or "")
    reasons = [
        "donor_selected_without_target_validation_pnl",
        f"source_profile={source_profile_id}",
        f"donor_symbol={donor_symbol}",
        f"target_group={broad69._asset_group(target_symbol)}",
        f"donor_group={broad69._asset_group(donor_symbol)}",
        f"target_subgroup={_asset_subgroup(target_symbol)}",
        f"donor_subgroup={_asset_subgroup(donor_symbol)}",
    ]
    if donor_symbol in _preferred_donors(target_symbol):
        reasons.append("preferred_domain_donor_match")
    return dict(selected), float(selected_score), reasons


def _shadow_profile_id(source_profile_id: str) -> str:
    return f"{source_profile_id}_cold_start_transfer_shadow"


def _shadow_validation_score(row: Mapping[str, Any]) -> float:
    validation = _safe_float(row.get("validation_return"))
    val_mdd = _safe_float(row.get("validation_mdd"))
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    events = int(row.get("validation_trade_event_count") or 0)
    score = 10.0 * validation + min(max(val_rpt, -50.0), 250.0) / 100.0 - 3.5 * val_mdd
    score += min(events, 60) / 120.0
    if validation <= 0.0:
        score -= 5.0 + abs(validation) * 5.0
    if val_rpt <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        score -= (broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS - val_rpt) / 5.0
    if events < 3:
        score -= 1.0
    return float(score)


def _build_cold_start_streams(
    *,
    source_payload: Mapping[str, Any],
    cache: profile69.FeatureCache,
    windows: broad69.SplitWindows,
    train_eligibility: Mapping[str, Any],
) -> tuple[list[broad69.CandidateStream], list[dict[str, Any]], dict[str, Any]]:
    donors = _donor_rows(source_payload, train_eligibility)
    profiles = tuple(str(spec["profile_id"]) for spec in profile69.PROFILE_SPECS)
    streams: list[broad69.CandidateStream] = []
    rejected: list[dict[str, Any]] = []
    donor_map: dict[str, Any] = {}
    for target_symbol in train_eligibility.get("train_ineligible_symbols", []):
        donor_map[str(target_symbol)] = {}
        for source_profile_id in profiles:
            donor, donor_score, donor_reasons = _select_donor(
                target_symbol=str(target_symbol),
                source_profile_id=source_profile_id,
                donor_rows=donors,
                train_eligibility=train_eligibility,
            )
            if donor is None:
                rejected.append(
                    {
                        "symbol": target_symbol,
                        "source_profile_id": source_profile_id,
                        "rejection_reasons": donor_reasons,
                    }
                )
                donor_map[str(target_symbol)][source_profile_id] = {"rejected": donor_reasons}
                continue
            params = dict(donor.get("optuna_params") or {})
            timeframe = str(params.get("timeframe") or donor.get("timeframe"))
            target_tf = _timeframe_payload(train_eligibility, str(target_symbol), timeframe)
            try:
                stream = profile69._candidate_from_params(
                    symbol=str(target_symbol),
                    profile_id=_shadow_profile_id(source_profile_id),
                    params=params,
                    cache=cache,
                    windows=windows,
                    allocation_fraction=_safe_float(
                        donor.get("allocation_fraction"), profile69.DEFAULT_ALLOCATION_FRACTION
                    ),
                )
            except Exception as exc:
                rejected.append(
                    {
                        "symbol": target_symbol,
                        "source_profile_id": source_profile_id,
                        "donor_symbol": donor.get("symbol"),
                        "timeframe": timeframe,
                        "rejection_reasons": [
                            f"candidate_simulation_failed:{type(exc).__name__}:{exc}"
                        ],
                    }
                )
                continue
            row = dict(stream.row)
            rejection_reasons = list(row.get("rejection_reasons") or [])
            rejection_reasons.extend(
                [
                    "target_symbol_has_no_train_rows",
                    "cold_start_transfer_shadow_report_only_not_promotable",
                ]
            )
            row.update(
                {
                    "source_profile_id": source_profile_id,
                    "source_profile_cold_start_shadow": _shadow_profile_id(source_profile_id),
                    "cold_start_transfer": True,
                    "transfer_mode": "donor_parameter_prior_report_only",
                    "donor_selection_inputs": (
                        "donor_train_validation_metrics_static_domain_similarity_target_coverage_only"
                    ),
                    "target_validation_pnl_used_for_donor_selection": False,
                    "target_train_metrics_synthesized_from_donor": False,
                    "donor_ohlcv_substitution": False,
                    "synthetic_target_train_metrics": False,
                    "donor_symbol": donor.get("symbol"),
                    "donor_asset_group": broad69._asset_group(str(donor.get("symbol") or "")),
                    "donor_model_id": donor.get("model_id"),
                    "donor_profile_id": donor.get("profile_id"),
                    "donor_selection_score": donor_score,
                    "donor_quality_score": _safe_float(donor.get("donor_quality_score")),
                    "donor_train_return": _safe_float(donor.get("train_return")),
                    "donor_validation_return": _safe_float(donor.get("validation_return")),
                    "donor_train_return_per_turnover_proxy_bps": donor.get(
                        "train_return_per_turnover_proxy_bps"
                    ),
                    "donor_validation_return_per_turnover_proxy_bps": donor.get(
                        "validation_return_per_turnover_proxy_bps"
                    ),
                    "donor_rejection_reasons": list(donor.get("rejection_reasons") or []),
                    "donor_selection_reasons": donor_reasons,
                    "donor_params": params,
                    "target_train_rows": int(target_tf.get("train_rows") or 0),
                    "target_validation_rows": int(target_tf.get("validation_rows") or 0),
                    "target_train_eligible": False,
                    "shadow_validation_score": _shadow_validation_score(row),
                    "primary_shadow_selected": False,
                    "oracle_shadow_selected": False,
                    "shadow_multiplier": 0.0,
                    "weighted_notional_fraction": 0.0,
                    "promotion_status": "blocked_shadow_only_no_target_train_rows",
                    "decision": "cold_start_shadow_only_no_paper_or_live_promotion",
                    "ready_for_paper": False,
                    "paper_testnet_candidate": False,
                    "ready_for_real": False,
                    "real_money_execution": False,
                    "real_execution_allowed": False,
                    "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
                }
            )
            streams.append(
                broad69.CandidateStream(row=row, returns=stream.returns, position=stream.position)
            )
            donor_map[str(target_symbol)][source_profile_id] = {
                "donor_symbol": donor.get("symbol"),
                "donor_model_id": donor.get("model_id"),
                "donor_selection_score": donor_score,
                "donor_quality_score": donor.get("donor_quality_score"),
                "timeframe": timeframe,
                "target_validation_rows": int(target_tf.get("validation_rows") or 0),
                "selection_reasons": donor_reasons,
            }
    streams.sort(
        key=lambda stream: _safe_float(stream.row.get("donor_selection_score")), reverse=True
    )
    for rank, stream in enumerate(streams, start=1):
        stream.row["shadow_rank"] = rank
    diagnostics = {
        "donor_row_count": len(donors),
        "target_symbol_count": len(train_eligibility.get("train_ineligible_symbols", [])),
        "candidate_stream_count": len(streams),
        "rejected_candidate_count": len(rejected),
    }
    return streams, rejected, {"by_target_symbol": donor_map, "diagnostics": diagnostics}


def _eligible_for_oracle_shadow(row: Mapping[str, Any]) -> bool:
    return (
        _safe_float(row.get("validation_return")) > 0.0
        and _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
        > broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS
        and _safe_float(row.get("validation_mdd")) <= 0.20
        and int(row.get("validation_trade_event_count") or 0) >= 3
    )


def _select_shadow_sleeves(
    streams: Sequence[broad69.CandidateStream],
    *,
    max_sleeves: int,
    max_gross: float,
    selection_policy: str,
    use_target_validation_score: bool,
) -> tuple[list[broad69.CandidateStream], np.ndarray]:
    candidates = list(streams)
    if use_target_validation_score:
        candidates = [stream for stream in candidates if _eligible_for_oracle_shadow(stream.row)]
        key = lambda stream: (  # noqa: E731
            _safe_float(stream.row.get("shadow_validation_score")),
            _safe_float(stream.row.get("donor_selection_score")),
            int(stream.row.get("target_validation_rows") or 0),
        )
    else:
        key = lambda stream: (  # noqa: E731
            _safe_float(stream.row.get("donor_selection_score")),
            _safe_float(stream.row.get("donor_quality_score")),
            int(stream.row.get("target_validation_rows") or 0),
        )
    candidates.sort(key=key, reverse=True)

    selected: list[broad69.CandidateStream] = []
    selected_symbols: set[str] = set()
    gross_by_group: defaultdict[str, float] = defaultdict(float)
    per_sleeve_gross = float(max_gross) / max(1, int(max_sleeves))
    multipliers: list[float] = []
    for stream in candidates:
        symbol = str(stream.row.get("symbol") or "")
        if symbol in selected_symbols:
            continue
        group = str(stream.row.get("asset_group") or broad69._asset_group(symbol))
        notional = abs(_safe_float(stream.row.get("notional_fraction")))
        if notional <= 0.0:
            continue
        multiplier = min(1.0, per_sleeve_gross / notional)
        weighted_notional = multiplier * notional
        group_cap = max(0.60, float(max_gross) * 0.70)
        if gross_by_group[group] + weighted_notional > group_cap:
            continue
        selected.append(stream)
        multipliers.append(multiplier)
        selected_symbols.add(symbol)
        gross_by_group[group] += weighted_notional
        if len(selected) >= int(max_sleeves):
            break
    mult = np.array(multipliers, dtype=float)
    for stream, multiplier in zip(selected, mult, strict=True):
        row = stream.row
        row[f"{selection_policy}_shadow_selected"] = True
        if selection_policy == "primary":
            row["primary_shadow_selected"] = True
        if selection_policy == "oracle":
            row["oracle_shadow_selected"] = True
        row["shadow_multiplier"] = max(_safe_float(row.get("shadow_multiplier")), float(multiplier))
        row["weighted_notional_fraction"] = max(
            _safe_float(row.get("weighted_notional_fraction")),
            float(multiplier) * _safe_float(row.get("notional_fraction")),
        )
    return selected, mult


def _portfolio_corr_matrix(
    streams: Sequence[broad69.CandidateStream], windows: broad69.SplitWindows, split: str
) -> dict[str, dict[str, float | None]]:
    if not streams:
        return {}
    matrix = profile69._aligned_matrix(streams)
    mask = broad69._split_mask(pd.Series(matrix.index), split, windows)
    selected = matrix.loc[mask]
    if selected.empty:
        return {}
    corr = selected.corr()
    out: dict[str, dict[str, float | None]] = {}
    for idx, row in corr.iterrows():
        out[str(idx)] = {
            str(col): (None if pd.isna(value) else float(value)) for col, value in row.items()
        }
    return out


def _portfolio_metrics(
    *,
    portfolio_id: str,
    selection_policy: str,
    streams: Sequence[broad69.CandidateStream],
    multipliers: np.ndarray,
    windows: broad69.SplitWindows,
) -> dict[str, Any]:
    if not streams:
        return {
            "portfolio_id": portfolio_id,
            "selection_policy": selection_policy,
            "sleeve_count": 0,
            "gross_notional_fraction": 0.0,
            "ready_for_paper": False,
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "rejection_reasons": ["no_shadow_sleeves_selected"],
        }
    matrix = profile69._aligned_matrix(streams)
    weights = {
        str(stream.row["model_id"]): float(multiplier)
        for stream, multiplier in zip(streams, multipliers, strict=True)
    }
    returns = matrix.mul(pd.Series(weights), axis=1).sum(axis=1).sort_index()
    turnover_by_split: dict[str, float] = {}
    events_by_split: dict[str, int] = {}
    for split in broad69.SPLIT_ORDER:
        turnover_by_split[split] = float(
            sum(
                float(multiplier)
                * _safe_float(stream.row.get("notional_fraction"))
                * int(stream.row.get(f"{split}_trade_event_count") or 0)
                for stream, multiplier in zip(streams, multipliers, strict=True)
            )
        )
        events_by_split[split] = int(
            sum(int(stream.row.get(f"{split}_trade_event_count") or 0) for stream in streams)
        )
    metrics = profile69._profile_metrics_from_returns(
        returns,
        windows=windows,
        turnover_by_split=turnover_by_split,
        events_by_split=events_by_split,
    )
    concentration = profile69._profile_concentration(streams, multipliers)
    row = {
        "portfolio_id": portfolio_id,
        "selection_policy": selection_policy,
        "sleeve_count": len(streams),
        "weights": weights,
        "gross_notional_fraction": concentration["gross_notional_fraction"],
        **metrics,
        **concentration,
        "selected_symbols": [str(stream.row.get("symbol")) for stream in streams],
        "selection_inputs": (
            "donor_metrics_similarity_coverage_only"
            if selection_policy == "donor_frozen_primary"
            else "target_validation_diagnostic_oracle_not_promotable"
        ),
        "target_validation_pnl_used_for_selection": selection_policy != "donor_frozen_primary",
        "promotion_status": "shadow_report_only_no_target_train_rows",
        "ready_for_paper": False,
        "paper_testnet_candidate": False,
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "rejection_reasons": [
            "target_symbols_have_no_train_rows",
            "cold_start_shadow_portfolio_not_promotable",
        ],
    }
    return row


def _attribution_rows(
    streams: Sequence[broad69.CandidateStream], multipliers: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stream, multiplier in zip(streams, multipliers, strict=True):
        row = stream.row
        weighted_notional = float(multiplier) * _safe_float(row.get("notional_fraction"))
        rows.append(
            {
                "symbol": row.get("symbol"),
                "asset_group": row.get("asset_group"),
                "family": row.get("family"),
                "timeframe": row.get("timeframe"),
                "donor_symbol": row.get("donor_symbol"),
                "weighted_notional_fraction": weighted_notional,
                "validation_return": row.get("validation_return"),
                "validation_mdd": row.get("validation_mdd"),
                "validation_return_per_turnover_proxy_bps": row.get(
                    "validation_return_per_turnover_proxy_bps"
                ),
                "validation_simple_pnl_contribution_proxy": weighted_notional
                * _safe_float(row.get("validation_return"))
                / max(_safe_float(row.get("notional_fraction")), 1e-12),
            }
        )
    rows.sort(
        key=lambda item: _safe_float(item["validation_simple_pnl_contribution_proxy"]), reverse=True
    )
    return rows


def _strict_reference_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False}
    payload = json.loads(path.read_text("utf-8"))
    selected = dict(payload.get("selected_train_validation_legal_portfolio") or {})
    optuna_selected = dict(payload.get("selected_optuna_hybrid_profile") or {})
    return {
        "path": str(path),
        "available": True,
        "artifact_kind": payload.get("artifact_kind"),
        "train_eligible_symbol_count": (payload.get("train_eligibility") or {}).get(
            "train_eligible_symbol_count"
        ),
        "train_ineligible_symbol_count": (payload.get("train_eligibility") or {}).get(
            "train_ineligible_symbol_count"
        ),
        "selected_train_validation_legal_portfolio": {
            key: selected.get(key)
            for key in (
                "portfolio_id",
                "train_return",
                "validation_return",
                "train_mdd",
                "validation_mdd",
                "gross_notional_fraction",
                "ready_for_paper",
                "ready_for_real",
            )
        },
        "selected_optuna_hybrid_profile": {
            key: optuna_selected.get(key)
            for key in (
                "portfolio_id",
                "train_return",
                "validation_return",
                "train_mdd",
                "validation_mdd",
                "gross_notional_fraction",
                "ready_for_paper",
                "ready_for_real",
            )
        },
    }


def _render_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    number = _safe_float(value, math.nan)
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100:.2f}%"


def _render_bps(value: Any) -> str:
    if value is None:
        return "n/a"
    number = _safe_float(value, math.nan)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.2f}bps"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    primary = dict(payload.get("donor_frozen_primary_shadow_portfolio") or {})
    oracle = dict(payload.get("validation_oracle_shadow_portfolio") or {})
    strict = dict(payload.get("strict_live_reference") or {})
    lines = [
        "# 69-Asset Cold-Start Transfer Shadow Report",
        "",
        "This artifact is report-only. It does not promote validation-only assets to live, paper, or real-money execution.",
        "",
        "## Safety Contract",
        "",
        "- `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`.",
        "- Donor selection uses donor train/validation quality, static domain similarity, and target coverage only.",
        "- Target validation PnL is not used for the primary donor-frozen selection.",
        "- Target train metrics are not synthesized from donor performance.",
        "- The validation-oracle lane is an upper-bound diagnostic only and is not promotable.",
        "",
        "## Portfolio Results",
        "",
        "| lane | sleeves | gross | train return | validation return | validation MDD | validation RPT | promotable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| donor-frozen primary | {primary.get('sleeve_count', 0)} | "
            f"{_safe_float(primary.get('gross_notional_fraction')):.2f} | "
            f"{_render_pct(primary.get('train_return'))} | "
            f"{_render_pct(primary.get('validation_return'))} | "
            f"{_render_pct(primary.get('validation_mdd'))} | "
            f"{_render_bps(primary.get('validation_return_per_turnover_proxy_bps'))} | no |"
        ),
        (
            f"| validation-oracle diagnostic | {oracle.get('sleeve_count', 0)} | "
            f"{_safe_float(oracle.get('gross_notional_fraction')):.2f} | "
            f"{_render_pct(oracle.get('train_return'))} | "
            f"{_render_pct(oracle.get('validation_return'))} | "
            f"{_render_pct(oracle.get('validation_mdd'))} | "
            f"{_render_bps(oracle.get('validation_return_per_turnover_proxy_bps'))} | no |"
        ),
    ]
    strict_selected = dict(strict.get("selected_train_validation_legal_portfolio") or {})
    if strict.get("available"):
        lines.extend(
            [
                "",
                "## Strict Live Reference",
                "",
                (
                    "Corrected strict live handoff remains the reference: "
                    f"train {_render_pct(strict_selected.get('train_return'))}, "
                    f"validation {_render_pct(strict_selected.get('validation_return'))}, "
                    f"validation MDD {_render_pct(strict_selected.get('validation_mdd'))}, "
                    f"gross {_safe_float(strict_selected.get('gross_notional_fraction')):.2f}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Primary Shadow Sleeves",
            "",
            "| symbol | donor | family | tf | weighted notional | val return | val MDD | val RPT |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("primary_shadow_attribution", [])[:25]:
        lines.append(
            f"| {row.get('symbol')} | {row.get('donor_symbol')} | {row.get('family')} | "
            f"{row.get('timeframe')} | {_safe_float(row.get('weighted_notional_fraction')):.3f} | "
            f"{_render_pct(row.get('validation_return'))} | "
            f"{_render_pct(row.get('validation_mdd'))} | "
            f"{_render_bps(row.get('validation_return_per_turnover_proxy_bps'))} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
        ]
    )
    primary_val = _safe_float(primary.get("validation_return"))
    oracle_val = _safe_float(oracle.get("validation_return"))
    if primary_val > 0.0:
        lines.append(
            "The donor-frozen cold-start transfer lane was positive on validation, but it remains "
            "shadow-only because every selected target has zero train-window rows."
        )
    else:
        lines.append(
            "The donor-frozen cold-start transfer lane did not clear a positive validation result; "
            "there is no promotion case."
        )
    if oracle_val > primary_val:
        lines.append(
            "The validation-oracle upper bound is better than the donor-frozen lane, which means "
            "future real train data may unlock useful sleeves, but current validation-only selection "
            "would be data leakage."
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_artifact).expanduser().resolve()
    strict_path = Path(args.strict_reference_artifact).expanduser().resolve()
    source_payload = _load_payload(source_path)
    symbols = tuple(source_payload["universe"]["symbols"])
    timeframes = tuple(source_payload["timeframes"])
    data_root = Path(source_payload["data_coverage"]["data_root"])
    windows = _split_windows_from_payload(source_payload)
    bars, coverage = broad69.load_all_bars(symbols, data_root=data_root, timeframes=timeframes)
    train_eligibility = broad69.build_train_eligibility_report(
        bars,
        symbols=symbols,
        timeframes=timeframes,
        windows=windows,
    )
    cache = profile69.FeatureCache(
        bars_by_symbol_tf=bars,
        symbols=symbols,
        timeframes=timeframes,
        _xsmom={},
        _anchor_returns={},
    )
    streams, rejected_rows, donor_map = _build_cold_start_streams(
        source_payload=source_payload,
        cache=cache,
        windows=windows,
        train_eligibility=train_eligibility,
    )
    primary_streams, primary_mult = _select_shadow_sleeves(
        streams,
        max_sleeves=int(args.max_sleeves),
        max_gross=float(args.max_gross),
        selection_policy="primary",
        use_target_validation_score=False,
    )
    oracle_streams, oracle_mult = _select_shadow_sleeves(
        streams,
        max_sleeves=int(args.oracle_max_sleeves),
        max_gross=float(args.oracle_max_gross),
        selection_policy="oracle",
        use_target_validation_score=True,
    )
    primary_portfolio = _portfolio_metrics(
        portfolio_id="donor_frozen_cold_start_transfer_shadow",
        selection_policy="donor_frozen_primary",
        streams=primary_streams,
        multipliers=primary_mult,
        windows=windows,
    )
    oracle_portfolio = _portfolio_metrics(
        portfolio_id="validation_oracle_cold_start_upper_bound_shadow",
        selection_policy="validation_oracle_diagnostic",
        streams=oracle_streams,
        multipliers=oracle_mult,
        windows=windows,
    )
    primary_attr = _attribution_rows(primary_streams, primary_mult)
    oracle_attr = _attribution_rows(oracle_streams, oracle_mult)
    candidate_rows = [dict(stream.row) for stream in streams]
    candidate_rows.sort(
        key=lambda row: (
            bool(row.get("primary_shadow_selected")),
            bool(row.get("oracle_shadow_selected")),
            _safe_float(row.get("shadow_validation_score")),
            _safe_float(row.get("donor_selection_score")),
        ),
        reverse=True,
    )
    for rank, row in enumerate(candidate_rows, start=1):
        row["shadow_rank"] = rank

    output_dir = Path(args.output_dir).expanduser().resolve()
    latest_json = output_dir / "alpha_zoo_69_asset_cold_start_transfer_shadow_latest.json"
    timestamped_json = (
        output_dir / f"alpha_zoo_69_asset_cold_start_transfer_shadow_{_timestamp()}.json"
    )
    latest_md = output_dir / "alpha_zoo_69_asset_cold_start_transfer_shadow_latest.md"
    candidates_csv = (
        output_dir / "alpha_zoo_69_asset_cold_start_transfer_shadow_candidates_latest.csv"
    )
    portfolios_csv = (
        output_dir / "alpha_zoo_69_asset_cold_start_transfer_shadow_portfolios_latest.csv"
    )
    primary_attr_csv = (
        output_dir / "alpha_zoo_69_asset_cold_start_transfer_shadow_primary_attr_latest.csv"
    )
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_69_asset_cold_start_transfer_shadow",
        "generated_at_utc": _utc_now_iso(),
        "source_artifact": str(source_path),
        "strict_reference_artifact": str(strict_path),
        "universe": {"symbol_count": len(symbols), "symbols": list(symbols)},
        "timeframes": list(timeframes),
        "split_policy": {
            **windows.as_payload(),
            "locked_oos": {
                "enabled": False,
                "start": None,
                "end": None,
                "role": "disabled_for_live_final_refit_no_test_set_reserved",
            },
        },
        "data_coverage": coverage,
        "train_eligibility": train_eligibility,
        "research_primary_round_trip_cost_bps": broad69.PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": broad69.AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "return_per_turnover_threshold_bps": broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "paper_testnet_only": True,
        "shadow_report_only": True,
        "ready_for_paper": False,
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "cold_start_policy": {
            "mode": "donor_parameter_prior_transfer_shadow",
            "primary_selection_uses_target_validation_pnl": False,
            "oracle_selection_uses_target_validation_pnl": True,
            "oracle_selection_purpose": "diagnostic_upper_bound_only",
            "target_train_metrics_synthesized_from_donor": False,
            "donor_ohlcv_substitution": False,
            "live_promotion_allowed": False,
            "promotion_blocker": "target_symbol_timeframes_have_zero_train_rows",
            "standard_rule": (
                "validation-only symbols may be monitored and shadow-scored, but must remain "
                "excluded from parameter fitting, sleeve allocation, hybrid selection, and live promotion "
                "until physical train-window data exists."
            ),
        },
        "optimization_policy": optimization_search_policy_payload(
            search_method="deterministic_donor_prior_transfer_shadow_no_target_fit",
            objective_policy="report_only_target_validation_after_donor_selection_freeze",
            selection_inputs=(
                "donor_train_validation_metrics",
                "static_domain_similarity",
                "coverage",
            ),
            extra={
                "source_artifact": str(source_path),
                "strict_reference_artifact": str(strict_path),
                "primary_max_gross": float(args.max_gross),
                "primary_max_sleeves": int(args.max_sleeves),
                "oracle_max_gross": float(args.oracle_max_gross),
                "oracle_max_sleeves": int(args.oracle_max_sleeves),
                "uses_test_set": False,
                "warmup_scope": "target_available_bars_only_no_donor_ohlcv_substitution",
            },
        ),
        "donor_map": donor_map,
        "cold_start_candidate_rows": candidate_rows,
        "rejected_shadow_candidate_rows": rejected_rows,
        "donor_frozen_primary_shadow_portfolio": primary_portfolio,
        "validation_oracle_shadow_portfolio": oracle_portfolio,
        "portfolio_rows": [primary_portfolio, oracle_portfolio],
        "primary_shadow_attribution": primary_attr,
        "oracle_shadow_attribution": oracle_attr,
        "primary_validation_corr_matrix": _portfolio_corr_matrix(
            primary_streams, windows, "validation"
        ),
        "oracle_validation_corr_matrix": _portfolio_corr_matrix(
            oracle_streams, windows, "validation"
        ),
        "strict_live_reference": _strict_reference_summary(strict_path),
        "runner_peak_rss_mib": peak_mib,
        "memory_summary": {"limit_mib": 8192.0, "pass_under_8gb": peak_mib < 8192.0},
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_md": str(latest_md),
            "candidates_csv": str(candidates_csv),
            "portfolios_csv": str(portfolios_csv),
            "primary_attr_csv": str(primary_attr_csv),
        },
    }
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    _write_csv(candidates_csv, candidate_rows, CANDIDATE_FIELDS)
    _write_csv(portfolios_csv, [primary_portfolio, oracle_portfolio], PORTFOLIO_FIELDS)
    _write_csv(
        primary_attr_csv,
        primary_attr,
        [
            "symbol",
            "asset_group",
            "family",
            "timeframe",
            "donor_symbol",
            "weighted_notional_fraction",
            "validation_return",
            "validation_mdd",
            "validation_return_per_turnover_proxy_bps",
            "validation_simple_pnl_contribution_proxy",
        ],
    )
    latest_md.write_text(_render_markdown(payload), "utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument(
        "--strict-reference-artifact", type=Path, default=DEFAULT_STRICT_REFERENCE_ARTIFACT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sleeves", type=int, default=PRIMARY_MAX_SLEEVES)
    parser.add_argument("--max-gross", type=float, default=PRIMARY_MAX_GROSS)
    parser.add_argument("--oracle-max-sleeves", type=int, default=ORACLE_MAX_SLEEVES)
    parser.add_argument("--oracle-max-gross", type=float, default=ORACLE_MAX_GROSS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    latest = payload["output_paths"]["latest_json"]
    primary = payload["donor_frozen_primary_shadow_portfolio"]
    oracle = payload["validation_oracle_shadow_portfolio"]
    print(f"wrote {latest}")
    print(
        "primary_shadow "
        f"validation_return={_render_pct(primary.get('validation_return'))} "
        f"validation_mdd={_render_pct(primary.get('validation_mdd'))} "
        f"gross={_safe_float(primary.get('gross_notional_fraction')):.2f}"
    )
    print(
        "oracle_shadow "
        f"validation_return={_render_pct(oracle.get('validation_return'))} "
        f"validation_mdd={_render_pct(oracle.get('validation_mdd'))} "
        f"gross={_safe_float(oracle.get('gross_notional_fraction')):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
