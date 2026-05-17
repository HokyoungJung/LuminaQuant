#!/usr/bin/env python3
"""Run common-split Alpha Zoo vs Hybrid v3.5/v3.6 comparison.

This runner is intentionally thin: it wires an explicit common split manifest
through the existing Alpha Zoo factor/calibration/replay and fixed-input hybrid
v3.5/v3.6 code paths without duplicating their core algorithms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.alpha_zoo.crypto_fx_factors import (  # noqa: E402
    assign_time_splits,
    build_crypto_fx_factor_specs,
    compute_factor_frame,
    screen_factor_frame,
)
from lumina_quant.alpha_zoo.factor_card import build_factor_card  # noqa: E402
from lumina_quant.research.candidate_outcome_ledger import CandidateOutcomeLedger  # noqa: E402
from lumina_quant.research.crypto_fx_alpha_zoo_real_data import (  # noqa: E402
    build_candidate_outcome_records,
    load_real_data_bundle,
    summarize_factor_source_coverage,
    write_candidate_outcome_ledger,
)

DEFAULT_ALPHA_V2 = REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2"
DEFAULT_OLD_ALPHA_DIR = DEFAULT_ALPHA_V2 / "crypto_fx_alpha_zoo_real_data_20260514"
DEFAULT_OLD_HYBRID_DIR = DEFAULT_ALPHA_V2 / "hybrid_v35_v36_fixed_inputs_20260517"
DEFAULT_OUTPUT_DIR = DEFAULT_ALPHA_V2 / "common_split_alpha_zoo_hybrid_v35_v36_20260517"
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market_parquet"
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT"
BASELINE_HEAD = "80a557c133930f51748ec20c4e582aa0d6f678de"
HOURLY_PERIODS_PER_YEAR = 365 * 24
COMMON_SPLIT_CONTRACT: dict[str, dict[str, str]] = {
    "train": {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-12-31T23:00:00Z",
        "role": "objective_calibration_selection",
    },
    "validation": {
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-02-28T23:00:00Z",
        "role": "objective_selection",
    },
    "locked_oos": {
        "start": "2026-03-01T00:00:00Z",
        "end": "2026-05-06T23:00:00Z",
        "role": "gate_report_only_after_candidate_freeze",
    },
}
SPLIT_ORDER = ("train", "validation", "locked_oos")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return _format_timestamp(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def _rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisoformat(str(value).replace("Z", "+00:00"))).tz_convert(UTC)


def _timestamp_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return ts.isoformat().replace("+00:00", "Z")


def apply_common_split(
    frame: pd.DataFrame,
    *,
    split_contract: Mapping[str, Mapping[str, str]] | None = None,
    timestamp_column: str = "timestamp",
    split_column: str = "split",
    drop_outside: bool = True,
) -> pd.DataFrame:
    """Apply the fixed common split labels by timestamp.

    Existing fractional split labels are overwritten.  Timestamps are normalized
    to naive UTC so existing replay code sees the same timestamp shape as prior
    artifacts while the manifest/report emits explicit `Z` timestamps.
    """
    contract = dict(split_contract or COMMON_SPLIT_CONTRACT)
    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    out = frame.copy()
    ts_utc = _timestamp_utc(out[timestamp_column])
    overall_start = min(_parse_utc(contract[name]["start"]) for name in SPLIT_ORDER)
    overall_end = max(_parse_utc(contract[name]["end"]) for name in SPLIT_ORDER)
    if drop_outside:
        keep = ts_utc.ge(overall_start) & ts_utc.le(overall_end)
        out = out.loc[keep].copy()
        ts_utc = ts_utc.loc[keep]
    labels = pd.Series("outside_common_split", index=out.index, dtype="object")
    for name in SPLIT_ORDER:
        start = _parse_utc(contract[name]["start"])
        end = _parse_utc(contract[name]["end"])
        labels.loc[ts_utc.ge(start) & ts_utc.le(end)] = name
    out[timestamp_column] = ts_utc.dt.tz_convert(UTC).dt.tz_localize(None)
    out[split_column] = labels.to_numpy(dtype=object)
    if drop_outside:
        out = out[out[split_column].isin(SPLIT_ORDER)].copy()
    return out.sort_values([timestamp_column, *(["symbol"] if "symbol" in out.columns else [])]).reset_index(drop=True)


def add_split_bounded_forward_return_label(
    frame: pd.DataFrame,
    *,
    horizon: int = 4,
    label_column: str = "forward_return",
) -> pd.DataFrame:
    """Add forward-return labels without crossing common-split boundaries."""
    required = {"symbol", "timestamp", "close", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing required columns for split-bounded labels: " + ", ".join(missing))
    h = max(1, int(horizon))
    out = frame.sort_values(["symbol", "timestamp"]).copy()
    grouped = out.groupby("symbol", sort=False)
    future_close = grouped["close"].shift(-h)
    future_split = grouped["split"].shift(-h)
    close = pd.to_numeric(out["close"], errors="coerce")
    labels = (pd.to_numeric(future_close, errors="coerce") / close.where(close.abs() > 1e-12)) - 1.0
    labels = labels.where(future_split.astype(str).eq(out["split"].astype(str)))
    out[label_column] = labels
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _timestamp_index_hash(frame: pd.DataFrame, *, split_column: str = "split") -> str:
    rows: list[str] = []
    data = frame[["timestamp", split_column]].dropna().copy()
    data["timestamp"] = _timestamp_utc(data["timestamp"])
    for split in SPLIT_ORDER:
        stamps = data.loc[data[split_column].astype(str).eq(split), "timestamp"].drop_duplicates().sort_values()
        rows.extend(f"{split}|{_format_timestamp(ts)}" for ts in stamps)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _split_periods(frame: pd.DataFrame, *, split_column: str = "split") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    data = frame.copy()
    data["timestamp"] = _timestamp_utc(data["timestamp"])
    for split in SPLIT_ORDER:
        part = data[data[split_column].astype(str).eq(split)]
        unique = part["timestamp"].drop_duplicates().sort_values()
        out[split] = {
            "start_timestamp": _format_timestamp(unique.iloc[0]) if len(unique) else None,
            "end_timestamp": _format_timestamp(unique.iloc[-1]) if len(unique) else None,
            "row_count": len(part),
            "unique_timestamp_count": len(unique),
        }
    return out


def _file_sha256(path: Path | str | None) -> str | None:
    if path is None or not str(path):
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    if not resolved.exists() or not resolved.is_file():
        return None
    h = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_common_screen_payload(
    frame: pd.DataFrame,
    *,
    source_ref: str,
    source_coverage: Mapping[str, Any],
    ledger_output: Path,
    horizon: int,
    top_n: int,
    entry_quantile: float,
    max_ledger_records_per_factor_side_split: int,
) -> dict[str, Any]:
    specs = build_crypto_fx_factor_specs()
    factors = compute_factor_frame(frame, specs=specs)
    factors = apply_common_split(factors)
    labeled = add_split_bounded_forward_return_label(factors, horizon=horizon)
    screen = screen_factor_frame(labeled, top_n=max(0, int(top_n)))
    spec_by_name = {spec.name: spec for spec in specs}
    cards = []
    for selected in screen["selected_factors"]:
        spec = spec_by_name[str(selected["factor"])]
        card = build_factor_card(
            spec,
            metrics=selected,
            selected_using_splits=("train", "validation"),
            uses_locked_oos_for_selection=False,
            source_refs=(source_ref,),
        )
        cards.append(card.to_dict())
    records = build_candidate_outcome_records(
        labeled,
        list(screen["selected_factors"]),
        entry_quantile=float(entry_quantile),
        max_records_per_factor_side_split=max(1, int(max_ledger_records_per_factor_side_split)),
    )
    ledger_summary = write_candidate_outcome_ledger(ledger_output, records)
    ledger_summary["enabled"] = True
    source_validity = dict(source_coverage.get("strategy_validity") or {"pass": True, "rejection_reasons": []})
    strategy_rejections = list(source_validity.get("rejection_reasons") or [])
    if not all(bool(card["strategy_validity"].get("pass")) for card in cards):
        strategy_rejections.append("selected_factor_card_invalid")
    return {
        "artifact_kind": "crypto_fx_alpha_zoo_common_split_screen_bundle",
        "schema_version": 3,
        "selection_policy": "train_validation_only_locked_oos_report_only_common_split",
        "uses_locked_oos_for_selection": False,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
        "calendar_primary": False,
        "split_label_authority": "explicit_common_split_manifest",
        "forward_label_policy": "split_bounded_no_cross_split_forward_return_labels",
        "factor_count": len(specs),
        "row_count": len(labeled),
        "split_periods": _split_periods(labeled),
        "timestamp_index_hash": _timestamp_index_hash(labeled),
        "screen": screen,
        "factor_cards": cards,
        "source_coverage": dict(source_coverage),
        "factor_source_coverage": summarize_factor_source_coverage(labeled),
        "candidate_outcome_ledger": ledger_summary,
        "strategy_validity": {
            "pass": not strategy_rejections,
            "calendar_primary": False,
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only",
            "primary_signal_type": "formulaic_state_factor",
            "rejection_reasons": sorted(set(strategy_rejections)),
        },
    }


def _build_calibration_payload(ledger_path: Path, calibration_module: ModuleType) -> dict[str, Any]:
    ledger = CandidateOutcomeLedger(ledger_path)
    records = ledger.read_all()
    return calibration_module.build_calibration_payload(
        records,
        ledger_summary=ledger.summary(),
        bucket_fields=("candidate_id", "side", "symbol", "regime_bucket", "volatility_bucket", "factor_bucket"),
        parent_fields=("candidate_id", "side"),
        min_bucket_n=30,
        confidence_z=1.64,
        min_lower_edge_bps=0.0,
        max_tail_loss_bps=250.0,
    )


def _old_selected_spec(old_replay: Mapping[str, Any], alpha_module: ModuleType) -> Any:
    grid = dict(old_replay.get("candidate_selection_grid") or {})
    old_name = str(grid.get("selected_candidate_name") or "alpha_zoo_conservative_exit")
    params = dict(grid.get("selected_candidate_params") or {})
    old_source = str(grid.get("selected_candidate_source") or "historical_old_split_selected")
    if not params:
        for spec in alpha_module._default_grid_specs():
            if spec.name == old_name:
                params = dict(spec.params)
                old_source = str(spec.source)
                break
    return alpha_module._GridSpec(
        f"{old_name}_carry_forward_old_split_selected",
        f"{old_source}:historical_old_split_carry_forward",
        params,
    )


def _annotate_replay_payload(
    payload: dict[str, Any],
    *,
    manifest: Mapping[str, Any],
    replay_kind: str,
    output_path: Path,
) -> dict[str, Any]:
    out = dict(payload)
    out["replay_kind"] = replay_kind
    out["common_split_manifest"] = dict(manifest)
    out["common_split_output_path"] = str(output_path)
    out["old_split_performance_role"] = "historical_only_reference_not_common_split_selection"
    return out


def _metric_row(
    *,
    candidate: str,
    family: str,
    split: str,
    metrics: Mapping[str, Any],
    split_period: Mapping[str, Any],
    selection_provenance: Mapping[str, Any],
    deployable_success: bool,
    rejection_reasons: Sequence[str],
    active_period: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active = dict(active_period or {})
    return {
        "candidate": candidate,
        "family": family,
        "split": split,
        "period_start_timestamp": split_period.get("start_timestamp"),
        "period_end_timestamp": split_period.get("end_timestamp"),
        "active_start_timestamp": active.get("start_timestamp"),
        "active_end_timestamp": active.get("end_timestamp"),
        "total_return": _safe_float(metrics.get("total_return")),
        "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        "return_mdd_diagnostic": _safe_float(metrics.get("return_mdd")),
        "sharpe": _safe_float(metrics.get("sharpe")),
        "sortino": _safe_float(metrics.get("sortino")),
        "smart_sortino": _safe_float(metrics.get("smart_sortino")),
        "calmar": _safe_float(metrics.get("calmar")),
        "trade_count": int(metrics.get("trade_count") or metrics.get("active_return_hours") or 0),
        "liquidation_count": metrics.get("liquidation_count"),
        "minimum_margin_buffer": metrics.get("minimum_margin_buffer"),
        "deployable_success": bool(deployable_success),
        "rejection_reasons": list(rejection_reasons),
        "selection_provenance": dict(selection_provenance),
    }


def _alpha_split_active_periods(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    periods = dict(payload.get("trade_split_periods") or {})
    if periods:
        return {split: dict(periods.get(split) or {}) for split in SPLIT_ORDER}
    for split in SPLIT_ORDER:
        out[split] = {"start_timestamp": None, "end_timestamp": None}
    return out


def _alpha_rows_from_replay(
    payload: Mapping[str, Any],
    *,
    candidate: str,
    family: str,
    manifest_periods: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _alpha_strict_integer_table(payload)
    row6 = next((dict(row) for row in rows if int(_safe_float(row.get("leverage"))) == 6), {})
    split_metrics = dict(row6.get("split_metrics") or {})
    split_status = dict(dict(row6.get("liquidation_audit") or {}).get("split_status") or {})
    rejection = [] if bool(row6.get("deployable_success")) else list(row6.get("rejection_reasons") or [])
    if not row6:
        rejection.append("strict_6x_row_missing")
    active_periods = _alpha_split_active_periods(payload)
    out = []
    for split in SPLIT_ORDER:
        metrics = dict(split_metrics.get(split) or {})
        status = dict(split_status.get(split) or {})
        metrics["liquidation_count"] = int(status.get("liquidation_count") or 0) if status else None
        metrics["minimum_margin_buffer"] = status.get("minimum_margin_buffer")
        out.append(
            _metric_row(
                candidate=candidate,
                family=family,
                split=split,
                metrics=metrics,
                split_period=dict(manifest_periods.get(split) or {}),
                selection_provenance=dict(payload.get("selection_provenance") or {}),
                deployable_success=bool(row6.get("deployable_success")),
                rejection_reasons=rejection,
                active_period=active_periods.get(split),
            )
        )
    return out


def _hybrid_active_periods(item: Mapping[str, Any], hybrid_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    # Public hybrid payload intentionally omits full return streams.  Its split
    # metrics are computed on the fixed hourly split periods, so expose those
    # same split min/max timestamps as the candidate's effective active window.
    periods = dict(hybrid_payload.get("split_periods") or {})
    return {
        split: {
            "start_timestamp": dict(periods.get(split) or {}).get("start_timestamp"),
            "end_timestamp": dict(periods.get(split) or {}).get("end_timestamp"),
        }
        for split in SPLIT_ORDER
    }


def _hybrid_rows(
    item: Mapping[str, Any],
    *,
    candidate: str,
    hybrid_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    splits = dict(item.get("splits") or {})
    periods = dict(hybrid_payload.get("split_periods") or {})
    provenance = dict(item.get("selection_provenance") or {})
    rejection = list(item.get("rejection_reasons") or [])
    active = _hybrid_active_periods(item, hybrid_payload)
    return [
        _metric_row(
            candidate=candidate,
            family="fixed_input_hybrid_optuna",
            split=split,
            metrics=dict(splits.get(split) or {}),
            split_period=dict(periods.get(split) or {}),
            selection_provenance=provenance,
            deployable_success=bool(item.get("deployable_success")),
            rejection_reasons=rejection,
            active_period=active.get(split),
        )
        for split in SPLIT_ORDER
    ]


def _alpha_strict_integer_table(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = list(payload.get("integer_grid_results") or [])
    if not rows:
        rows = list(dict(payload.get("strict_zero_liquidation_lane") or {}).get("integer_grid_results") or [])
    out = []
    for row in rows:
        item = dict(row)
        item["return_mdd_role"] = "diagnostic_report_only"
        out.append(item)
    return out


def _alpha_diagnostic_5x_6x(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(dict(payload.get("diagnostic_nonfatal_lane") or {}).get("high_leverage_5x_6x_report") or [])


def _locked_oos_audit(
    *,
    screen_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    reselected: Mapping[str, Any],
    carry_forward: Mapping[str, Any],
    hybrid_payload: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "screen_uses_locked_oos_for_selection": bool(screen_payload.get("uses_locked_oos_for_selection")),
        "calibration_uses_locked_oos": bool(calibration_payload.get("uses_locked_oos_for_calibration")),
        "calibration_locked_oos_record_count": int(calibration_payload.get("locked_oos_calibration_record_count") or 0),
        "reselected_uses_locked_oos_for_selection": bool(reselected.get("uses_locked_oos_for_selection")),
        "carry_forward_uses_locked_oos_for_selection": bool(carry_forward.get("uses_locked_oos_for_selection")),
        "hybrid_uses_locked_oos_for_objective": bool(dict(hybrid_payload.get("selection_policy") or {}).get("uses_locked_oos_for_objective")),
        "hybrid_uses_locked_oos_for_pruning": bool(dict(hybrid_payload.get("selection_policy") or {}).get("uses_locked_oos_for_pruning")),
        "hybrid_uses_locked_oos_for_selection": bool(dict(hybrid_payload.get("selection_policy") or {}).get("uses_locked_oos_for_selection")),
    }
    violation_reasons = [key for key, value in checks.items() if bool(value)]
    if checks["calibration_locked_oos_record_count"] != 0:
        violation_reasons.append("calibration_locked_oos_record_count_nonzero")
    for key in ("hybrid_v3_5_optuna", "hybrid_v3_6_optuna"):
        optuna = dict(dict(hybrid_payload.get(key) or {}).get("optuna") or {})
        if bool(optuna.get("uses_locked_oos_for_selection")):
            violation_reasons.append(f"{key}_optuna_uses_locked_oos_for_selection")
        if list(optuna.get("locked_oos_objective_columns_used") or []):
            violation_reasons.append(f"{key}_locked_oos_objective_columns_used")
    return {
        "checks": checks,
        "violation": bool(violation_reasons),
        "violation_reasons": sorted(set(violation_reasons)),
        "live_promotion_invalid_if_violation": True,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
    }


def _historical_fractional_split_periods(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Reconstruct the old Alpha Zoo chronological fractional split periods.

    The pre-common-split Alpha Zoo replay artifacts did not persist full split
    min/max timestamps, only counts and preview trades.  Recompute the old
    60/20/20 timestamp split from the same source frame so the historical-only
    row is visibly not on the common split.
    """
    return _split_periods(assign_time_splits(frame))


def _markdown_report(payload: Mapping[str, Any]) -> str:
    def pct(value: Any) -> str:
        return f"{_safe_float(value):+.2%}"

    def num(value: Any) -> str:
        parsed = _safe_float(value, float("nan"))
        return "" if not math.isfinite(parsed) else f"{parsed:.3f}"

    lines = [
        "# Common-split Alpha Zoo vs Hybrid v3.5/v3.6",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        f"- baseline_parent: `{payload.get('baseline_parent')}`",
        "- Alpha Zoo old split: **historical only**, not used for common-split selection.",
        "- locked-OOS: gate/report-only after candidate freeze.",
        "- return/MDD: diagnostic/report-only, not a hard promotion gate.",
        "",
        "## Common split",
        "",
    ]
    for split, item in dict(payload.get("common_split_manifest", {}).get("split_periods") or {}).items():
        lines.append(
            f"- {split}: `{item.get('start_timestamp')}` ~ `{item.get('end_timestamp')}`; "
            f"unique timestamps `{item.get('unique_timestamp_count')}`, rows `{item.get('row_count')}`"
        )
    lines.extend(
        [
            "",
            "## Candidate split performance",
            "",
            "| candidate | split | period | active | return | MDD | return/MDD diag | Sharpe | Sortino | smart Sortino | Calmar | trades | liq | min buffer | deployable | rejection reasons |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in list(payload.get("candidate_split_performance") or []):
        period = f"{row.get('period_start_timestamp')} → {row.get('period_end_timestamp')}"
        active = f"{row.get('active_start_timestamp')} → {row.get('active_end_timestamp')}"
        reasons = ", ".join(str(x) for x in list(row.get("rejection_reasons") or []))
        lines.append(
            f"| {row.get('candidate')} | {row.get('split')} | {period} | {active} | "
            f"{pct(row.get('total_return'))} | {pct(row.get('max_drawdown'))} | {num(row.get('return_mdd_diagnostic'))} | "
            f"{num(row.get('sharpe'))} | {num(row.get('sortino'))} | {num(row.get('smart_sortino'))} | {num(row.get('calmar'))} | "
            f"{row.get('trade_count')} | {row.get('liquidation_count')} | {num(row.get('minimum_margin_buffer'))} | "
            f"{row.get('deployable_success')} | {reasons} |"
        )
    audit = dict(payload.get("locked_oos_contamination_audit") or {})
    lines.extend(
        [
            "",
            "## Selection provenance and locked-OOS audit",
            "",
            f"- locked-OOS contamination violation: `{audit.get('violation')}`",
            f"- violation reasons: `{', '.join(audit.get('violation_reasons') or []) or 'none'}`",
            "- Hybrid input universe: `A0 + P0 + E0 + S1 + S2 + S3 + S4`.",
            "- Hybrid live promotion possible: `False` unless a dedicated integrated margin replay is added.",
            "",
            "## Strict zero-liquidation integer leverage lane",
            "",
            "| leverage | deployable | strict_safe | OOS return | OOS MDD | liq | min buffer |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in list(payload.get("strict_zero_liquidation_integer_leverage") or []):
        oos = dict(dict(row.get("split_metrics") or {}).get("locked_oos") or {})
        liq = dict(row.get("liquidation_audit") or {})
        lines.append(
            f"| {row.get('leverage')} | {row.get('deployable_success')} | {row.get('strict_safe')} | "
            f"{pct(oos.get('total_return'))} | {pct(oos.get('max_drawdown'))} | "
            f"{liq.get('total_liquidation_count')} | {num(liq.get('minimum_margin_buffer'))} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic nonfatal 5x/6x lane",
            "",
            "Diagnostic only; separated from live promotion.",
            "",
        ]
    )
    for row in list(payload.get("diagnostic_nonfatal_5x_6x_lane") or []):
        lines.append(
            f"- {row.get('leverage')}x: promotion_allowed `{row.get('promotion_allowed')}`, "
            f"total_liquidations `{row.get('total_liquidation_count')}`, min_buffer `{num(row.get('minimum_margin_buffer'))}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- best common-split strict candidate: `{payload.get('best_common_split_candidate')}`",
            f"- hybrid v3.5/v3.6 live promotion possible: `{payload.get('hybrid_live_promotion_possible')}`",
            f"- memory peak RSS MiB: `{num(dict(payload.get('memory_summary') or {}).get('peak_rss_mib'))}`",
            f"- research history/source ledger update: `{payload.get('research_history_update_decision')}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    alpha_module = _load_module(REPO_ROOT / "scripts/research/replay_crypto_fx_alpha_zoo_state.py", "common_split_alpha_replay")
    calibration_module = _load_module(REPO_ROOT / "scripts/research/calibrate_crypto_fx_edges.py", "common_split_edge_calibration")
    hybrid_module = _load_module(
        REPO_ROOT / "scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py",
        "common_split_hybrid_v35_v36",
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir = output_dir / "alpha_zoo_common_split"
    hybrid_dir = output_dir / "hybrid_v35_v36_common_split"
    ledger_path = alpha_dir / "candidate_outcome_ledger_common_split_latest.jsonl"
    screen_path = alpha_dir / "crypto_fx_alpha_zoo_screen_common_split_latest.json"
    calibration_path = alpha_dir / "edge_calibration_common_split_latest.json"
    carry_replay_path = alpha_dir / "crypto_fx_alpha_zoo_state_replay_carry_forward_common_split_latest.json"
    reselected_replay_path = alpha_dir / "crypto_fx_alpha_zoo_state_replay_reselected_common_split_latest.json"

    old_replay_path = Path(args.old_alpha_replay_json).expanduser().resolve()
    old_summary_path = Path(args.old_alpha_summary_json).expanduser().resolve()
    old_hybrid_path = Path(args.old_hybrid_json).expanduser().resolve()
    old_replay = _load_json(old_replay_path)
    old_summary = _load_json(old_summary_path)
    bundle = load_real_data_bundle(
        input_path=args.input,
        current_tail_cache=args.current_tail_cache,
        external_state_csv=args.external_state_csv,
        strict_real_data=bool(args.strict_real_data),
    )
    common_frame = apply_common_split(bundle.frame)
    source_path = Path(str(bundle.metadata.get("source_path") or ""))
    manifest = {
        "artifact_kind": "common_split_manifest",
        "split_contract": COMMON_SPLIT_CONTRACT,
        "split_periods": _split_periods(common_frame),
        "timestamp_index_hash": _timestamp_index_hash(common_frame),
        "baseline_parent": BASELINE_HEAD,
        "source_path": str(source_path),
        "source_sha256": _file_sha256(source_path),
        "external_state_csv": str(Path(args.external_state_csv).expanduser().resolve()) if str(args.external_state_csv).strip() else "",
        "external_state_sha256": _file_sha256(args.external_state_csv),
        "old_alpha_replay_json": str(old_replay_path),
        "old_alpha_replay_sha256": _file_sha256(old_replay_path),
        "old_alpha_summary_json": str(old_summary_path),
        "old_alpha_summary_sha256": _file_sha256(old_summary_path),
        "old_hybrid_json": str(old_hybrid_path),
        "old_hybrid_sha256": _file_sha256(old_hybrid_path),
        "commands": {
            "runner": "scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py",
            "hybrid_n_trials": int(args.n_trials),
            "hybrid_seed": int(args.seed),
        },
        "output_paths": {
            "screen": str(screen_path),
            "calibration": str(calibration_path),
            "carry_forward_replay": str(carry_replay_path),
            "reselected_replay": str(reselected_replay_path),
            "hybrid_dir": str(hybrid_dir),
        },
    }

    screen_payload = _build_common_screen_payload(
        common_frame,
        source_ref=f"input:{bundle.metadata.get('source_path')}",
        source_coverage=bundle.metadata,
        ledger_output=ledger_path,
        horizon=int(args.horizon),
        top_n=int(args.top_n),
        entry_quantile=float(args.entry_quantile),
        max_ledger_records_per_factor_side_split=int(args.max_ledger_records_per_factor_side_split),
    )
    screen_payload["common_split_manifest"] = manifest
    _write_json(screen_path, screen_payload)

    calibration_payload = _build_calibration_payload(ledger_path, calibration_module)
    calibration_payload["common_split_manifest"] = manifest
    _write_json(calibration_path, calibration_payload)

    calibrated_edges = alpha_module._load_calibrated_edges(calibration_path)
    carry_spec = _old_selected_spec(old_replay, alpha_module)
    carry_payload = alpha_module.replay_frame(
        common_frame,
        require_calibrated_edge=True,
        calibrated_edges=calibrated_edges,
        max_leverage=6,
        allocation_fraction=float(args.allocation_fraction),
        source_metadata=bundle.metadata,
        grid_specs=[carry_spec],
    )
    carry_payload = _annotate_replay_payload(
        carry_payload,
        manifest=manifest,
        replay_kind="common_split_old_split_selected_carry_forward",
        output_path=carry_replay_path,
    )
    _write_json(carry_replay_path, carry_payload)

    reselected_payload = alpha_module.replay_frame(
        common_frame,
        require_calibrated_edge=True,
        calibrated_edges=calibrated_edges,
        max_leverage=6,
        allocation_fraction=float(args.allocation_fraction),
        source_metadata=bundle.metadata,
        grid_specs=alpha_module._default_grid_specs(),
    )
    reselected_payload = _annotate_replay_payload(
        reselected_payload,
        manifest=manifest,
        replay_kind="common_split_train_validation_reselected_grid",
        output_path=reselected_replay_path,
    )
    _write_json(reselected_replay_path, reselected_payload)

    hybrid_args = hybrid_module.parse_args(
        [
            "--market-root",
            str(args.market_root),
            "--exchange",
            str(args.exchange),
            "--symbols",
            str(args.symbols),
            "--oos-end-date",
            "2026-05-06",
            "--portfolio-json",
            str(args.portfolio_json),
            "--alpha-replay-json",
            str(reselected_replay_path),
            "--alpha-calibration-json",
            str(calibration_path),
            "--external-state-csv",
            str(args.external_state_csv),
            "--n-trials",
            str(int(args.n_trials)),
            "--seed",
            str(int(args.seed)),
            "--output-dir",
            str(hybrid_dir),
        ]
    )
    hybrid_payload = hybrid_module.build_payload(hybrid_args)
    hybrid_payload["common_split_manifest"] = manifest
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    hybrid_latest = hybrid_dir / "hybrid_v35_v36_fixed_inputs_common_split_latest.json"
    _write_json(hybrid_latest, hybrid_payload)
    (hybrid_dir / "hybrid_v35_v36_fixed_inputs_common_split_latest.md").write_text(
        hybrid_module._markdown(hybrid_payload), encoding="utf-8"
    )

    candidate_rows: list[dict[str, Any]] = []
    old_historical_periods = _historical_fractional_split_periods(bundle.frame)
    candidate_rows.extend(
        _alpha_rows_from_replay(
            old_replay,
            candidate="alpha_zoo_strict_6x_old_split_historical_only",
            family="historical_reference_only_old_fractional_split",
            manifest_periods=old_historical_periods,
        )
    )
    for row in candidate_rows[-3:]:
        row["deployable_success"] = False
        row["rejection_reasons"] = sorted({*(row.get("rejection_reasons") or []), "historical_old_split_only_not_common_split_selection"})
    candidate_rows.extend(
        _alpha_rows_from_replay(
            carry_payload,
            candidate="alpha_zoo_strict_6x_common_split_carry_forward_old_selected",
            family="common_split_alpha_zoo_carry_forward",
            manifest_periods=manifest["split_periods"],
        )
    )
    selected_name = str(dict(reselected_payload.get("candidate_selection_grid") or {}).get("selected_candidate_name") or "unknown")
    candidate_rows.extend(
        _alpha_rows_from_replay(
            reselected_payload,
            candidate=f"alpha_zoo_strict_6x_common_split_reselected:{selected_name}",
            family="common_split_alpha_zoo_reselected",
            manifest_periods=manifest["split_periods"],
        )
    )
    candidate_rows.extend(
        _hybrid_rows(
            dict(hybrid_payload.get("hybrid_v3_5_optuna") or {}),
            candidate="hybrid_v3_5_optuna_common_split",
            hybrid_payload=hybrid_payload,
        )
    )
    candidate_rows.extend(
        _hybrid_rows(
            dict(hybrid_payload.get("hybrid_v3_6_optuna") or {}),
            candidate="hybrid_v3_6_optuna_common_split",
            hybrid_payload=hybrid_payload,
        )
    )

    audit = _locked_oos_audit(
        screen_payload=screen_payload,
        calibration_payload=calibration_payload,
        reselected=reselected_payload,
        carry_forward=carry_payload,
        hybrid_payload=hybrid_payload,
    )
    strict_rows = _alpha_strict_integer_table(reselected_payload)
    strict_deployables = [row for row in strict_rows if bool(row.get("deployable_success"))]
    promoted = max(strict_deployables, key=lambda row: _safe_float(row.get("leverage")), default={})

    peak_rss = max(
        _rss_mib(),
        _safe_float(dict(old_summary.get("memory_summary") or {}).get("peak_rss_mib")),
        _safe_float(dict(old_replay.get("memory_summary") or {}).get("peak_rss_mib")),
        _safe_float(dict(hybrid_payload.get("memory_summary") or {}).get("peak_rss_mib")),
        _safe_float(dict(reselected_payload.get("memory_summary") or {}).get("peak_rss_mib")),
        _safe_float(dict(carry_payload.get("memory_summary") or {}).get("peak_rss_mib")),
    )
    payload = {
        "artifact_kind": "common_split_alpha_zoo_hybrid_v35_v36_comparison",
        "generated_at_utc": _utc_now_iso(),
        "baseline_parent": BASELINE_HEAD,
        "common_split_manifest": manifest,
        "old_split_alpha_zoo_role": "historical_only_reference_not_common_split_selection",
        "current_base_calendar_tuple_role": "hypothesis_reference_only",
        "return_mdd_role": "diagnostic_report_only_not_hard_promotion_gate",
        "fixed_input_hybrid_universe": ["A0", "P0", "E0", "S1", "S2", "S3", "S4"],
        "alpha_zoo_common_split": {
            "screen_json": str(screen_path),
            "calibration_json": str(calibration_path),
            "carry_forward_replay_json": str(carry_replay_path),
            "reselected_replay_json": str(reselected_replay_path),
            "carry_forward_selected_candidate": dict(carry_payload.get("candidate_selection_grid") or {}).get("selected_candidate_name"),
            "reselected_candidate": selected_name,
        },
        "hybrid_common_split": {
            "json": str(hybrid_latest),
            "markdown": str(hybrid_dir / "hybrid_v35_v36_fixed_inputs_common_split_latest.md"),
            "selection_policy": dict(hybrid_payload.get("selection_policy") or {}),
        },
        "candidate_split_performance": candidate_rows,
        "locked_oos_contamination_audit": audit,
        "strict_zero_liquidation_integer_leverage": strict_rows,
        "diagnostic_nonfatal_5x_6x_lane": _alpha_diagnostic_5x_6x(reselected_payload),
        "best_common_split_candidate": dict(promoted).get("candidate_name") or selected_name,
        "hybrid_live_promotion_possible": False,
        "hybrid_live_promotion_rejection_reasons": sorted(
            set(
                list(dict(hybrid_payload.get("hybrid_v3_5_optuna") or {}).get("rejection_reasons") or [])
                + list(dict(hybrid_payload.get("hybrid_v3_6_optuna") or {}).get("rejection_reasons") or [])
            )
        ),
        "existing_best_vs_common_basis": {
            "old_alpha_best_split_role": "historical_only",
            "common_basis_winner": dict(promoted).get("candidate_name") or selected_name,
            "common_basis_hybrid_v3_5_deployable": bool(dict(hybrid_payload.get("hybrid_v3_5_optuna") or {}).get("deployable_success")),
            "common_basis_hybrid_v3_6_deployable": bool(dict(hybrid_payload.get("hybrid_v3_6_optuna") or {}).get("deployable_success")),
        },
        "memory_summary": {
            "peak_rss_mib": peak_rss,
            "limit_mib": 8192.0,
            "pass_under_8gb": peak_rss < 8192.0,
        },
        "research_history_update_decision": "not_regenerated_no_new_global_source_family_or_chronology_ledger_change",
        "source_payloads": {
            "old_alpha_summary": old_summary_path,
            "old_alpha_replay": old_replay_path,
            "old_hybrid_fixed_inputs": old_hybrid_path,
        },
    }
    return payload


def build_payload_from_stage_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the comparison payload from already-written stage artifacts."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    alpha_dir = output_dir / "alpha_zoo_common_split"
    hybrid_dir = output_dir / "hybrid_v35_v36_common_split"
    screen_path = alpha_dir / "crypto_fx_alpha_zoo_screen_common_split_latest.json"
    calibration_path = alpha_dir / "edge_calibration_common_split_latest.json"
    carry_replay_path = alpha_dir / "crypto_fx_alpha_zoo_state_replay_carry_forward_common_split_latest.json"
    reselected_replay_path = alpha_dir / "crypto_fx_alpha_zoo_state_replay_reselected_common_split_latest.json"
    hybrid_latest = hybrid_dir / "hybrid_v35_v36_fixed_inputs_common_split_latest.json"
    hybrid_markdown = hybrid_dir / "hybrid_v35_v36_fixed_inputs_common_split_latest.md"

    old_replay_path = Path(args.old_alpha_replay_json).expanduser().resolve()
    old_summary_path = Path(args.old_alpha_summary_json).expanduser().resolve()
    old_hybrid_path = Path(args.old_hybrid_json).expanduser().resolve()
    old_replay = _load_json(old_replay_path)
    old_summary = _load_json(old_summary_path)
    bundle = load_real_data_bundle(
        input_path=args.input,
        current_tail_cache=args.current_tail_cache,
        external_state_csv=args.external_state_csv,
        strict_real_data=bool(args.strict_real_data),
    )
    screen_payload = _load_json(screen_path)
    calibration_payload = _load_json(calibration_path)
    carry_payload = _load_json(carry_replay_path)
    reselected_payload = _load_json(reselected_replay_path)
    hybrid_payload = _load_json(hybrid_latest)
    manifest = dict(screen_payload.get("common_split_manifest") or {})
    if not manifest:
        common_frame = apply_common_split(bundle.frame)
        manifest = {
            "artifact_kind": "common_split_manifest",
            "split_contract": COMMON_SPLIT_CONTRACT,
            "split_periods": _split_periods(common_frame),
            "timestamp_index_hash": _timestamp_index_hash(common_frame),
            "baseline_parent": BASELINE_HEAD,
            "output_paths": {
                "screen": str(screen_path),
                "calibration": str(calibration_path),
                "carry_forward_replay": str(carry_replay_path),
                "reselected_replay": str(reselected_replay_path),
                "hybrid_dir": str(hybrid_dir),
            },
        }

    candidate_rows: list[dict[str, Any]] = []
    candidate_rows.extend(
        _alpha_rows_from_replay(
            old_replay,
            candidate="alpha_zoo_strict_6x_old_split_historical_only",
            family="historical_reference_only_old_fractional_split",
            manifest_periods=_historical_fractional_split_periods(bundle.frame),
        )
    )
    for row in candidate_rows[-3:]:
        row["deployable_success"] = False
        row["rejection_reasons"] = sorted(
            {*(row.get("rejection_reasons") or []), "historical_old_split_only_not_common_split_selection"}
        )
    candidate_rows.extend(
        _alpha_rows_from_replay(
            carry_payload,
            candidate="alpha_zoo_strict_6x_common_split_carry_forward_old_selected",
            family="common_split_alpha_zoo_carry_forward",
            manifest_periods=dict(manifest["split_periods"]),
        )
    )
    selected_name = str(dict(reselected_payload.get("candidate_selection_grid") or {}).get("selected_candidate_name") or "unknown")
    candidate_rows.extend(
        _alpha_rows_from_replay(
            reselected_payload,
            candidate=f"alpha_zoo_strict_6x_common_split_reselected:{selected_name}",
            family="common_split_alpha_zoo_reselected",
            manifest_periods=dict(manifest["split_periods"]),
        )
    )
    candidate_rows.extend(
        _hybrid_rows(
            dict(hybrid_payload.get("hybrid_v3_5_optuna") or {}),
            candidate="hybrid_v3_5_optuna_common_split",
            hybrid_payload=hybrid_payload,
        )
    )
    candidate_rows.extend(
        _hybrid_rows(
            dict(hybrid_payload.get("hybrid_v3_6_optuna") or {}),
            candidate="hybrid_v3_6_optuna_common_split",
            hybrid_payload=hybrid_payload,
        )
    )

    audit = _locked_oos_audit(
        screen_payload=screen_payload,
        calibration_payload=calibration_payload,
        reselected=reselected_payload,
        carry_forward=carry_payload,
        hybrid_payload=hybrid_payload,
    )
    strict_rows = _alpha_strict_integer_table(reselected_payload)
    strict_deployables = [row for row in strict_rows if bool(row.get("deployable_success"))]
    promoted = max(strict_deployables, key=lambda row: _safe_float(row.get("leverage")), default={})
    peak_rss = max(
        _rss_mib(),
        _safe_float(dict(old_summary.get("memory_summary") or {}).get("peak_rss_mib")),
        _safe_float(dict(old_replay.get("memory_summary") or {}).get("peak_rss_mib")),
        _safe_float(dict(hybrid_payload.get("memory_summary") or {}).get("peak_rss_mib")),
        _safe_float(dict(reselected_payload.get("memory_summary") or {}).get("peak_rss_mib")),
        _safe_float(dict(carry_payload.get("memory_summary") or {}).get("peak_rss_mib")),
    )
    return {
        "artifact_kind": "common_split_alpha_zoo_hybrid_v35_v36_comparison",
        "generated_at_utc": _utc_now_iso(),
        "baseline_parent": BASELINE_HEAD,
        "common_split_manifest": manifest,
        "old_split_alpha_zoo_role": "historical_only_reference_not_common_split_selection",
        "current_base_calendar_tuple_role": "hypothesis_reference_only",
        "return_mdd_role": "diagnostic_report_only_not_hard_promotion_gate",
        "fixed_input_hybrid_universe": ["A0", "P0", "E0", "S1", "S2", "S3", "S4"],
        "alpha_zoo_common_split": {
            "screen_json": str(screen_path),
            "calibration_json": str(calibration_path),
            "carry_forward_replay_json": str(carry_replay_path),
            "reselected_replay_json": str(reselected_replay_path),
            "carry_forward_selected_candidate": dict(carry_payload.get("candidate_selection_grid") or {}).get("selected_candidate_name"),
            "reselected_candidate": selected_name,
        },
        "hybrid_common_split": {
            "json": str(hybrid_latest),
            "markdown": str(hybrid_markdown),
            "selection_policy": dict(hybrid_payload.get("selection_policy") or {}),
        },
        "candidate_split_performance": candidate_rows,
        "locked_oos_contamination_audit": audit,
        "strict_zero_liquidation_integer_leverage": strict_rows,
        "diagnostic_nonfatal_5x_6x_lane": _alpha_diagnostic_5x_6x(reselected_payload),
        "best_common_split_candidate": dict(promoted).get("candidate_name") or selected_name,
        "hybrid_live_promotion_possible": False,
        "hybrid_live_promotion_rejection_reasons": sorted(
            set(
                list(dict(hybrid_payload.get("hybrid_v3_5_optuna") or {}).get("rejection_reasons") or [])
                + list(dict(hybrid_payload.get("hybrid_v3_6_optuna") or {}).get("rejection_reasons") or [])
            )
        ),
        "existing_best_vs_common_basis": {
            "old_alpha_best_split_role": "historical_only",
            "common_basis_winner": dict(promoted).get("candidate_name") or selected_name,
            "common_basis_hybrid_v3_5_deployable": bool(dict(hybrid_payload.get("hybrid_v3_5_optuna") or {}).get("deployable_success")),
            "common_basis_hybrid_v3_6_deployable": bool(dict(hybrid_payload.get("hybrid_v3_6_optuna") or {}).get("deployable_success")),
        },
        "memory_summary": {
            "peak_rss_mib": peak_rss,
            "limit_mib": 8192.0,
            "pass_under_8gb": peak_rss < 8192.0,
        },
        "research_history_update_decision": "not_regenerated_no_new_global_source_family_or_chronology_ledger_change",
        "source_payloads": {
            "old_alpha_summary": old_summary_path,
            "old_alpha_replay": old_replay_path,
            "old_hybrid_fixed_inputs": old_hybrid_path,
        },
    }


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_json = output_dir / "common_split_alpha_zoo_hybrid_v35_v36_latest.json"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timestamped_json = output_dir / f"common_split_alpha_zoo_hybrid_v35_v36_{timestamp}.json"
    latest_md = output_dir / "common_split_alpha_zoo_hybrid_v35_v36_latest.md"
    timestamped_md = output_dir / f"common_split_alpha_zoo_hybrid_v35_v36_{timestamp}.md"
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    md = _markdown_report(payload)
    latest_md.write_text(md, encoding="utf-8")
    timestamped_md.write_text(md, encoding="utf-8")
    return {
        "json": str(latest_json),
        "markdown": str(latest_md),
        "timestamped_json": str(timestamped_json),
        "timestamped_markdown": str(timestamped_md),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="")
    parser.add_argument("--current-tail-cache", default="")
    parser.add_argument("--external-state-csv", default=str(DEFAULT_ALPHA_V2 / "external_market_state_20260512/external_market_state_lagged.csv"))
    parser.add_argument("--strict-real-data", action="store_true", default=True)
    parser.add_argument("--market-root", default=str(DEFAULT_MARKET_ROOT))
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--portfolio-json", default=str(DEFAULT_ALPHA_V2 / "state_distilled_market_state_next_20260512/portfolio_tuning_leadership_unwind_top18/fresh_portfolio_tuning_latest.json"))
    parser.add_argument("--old-alpha-summary-json", default=str(DEFAULT_OLD_ALPHA_DIR / "crypto_fx_alpha_zoo_real_data_summary_latest.json"))
    parser.add_argument("--old-alpha-replay-json", default=str(DEFAULT_OLD_ALPHA_DIR / "crypto_fx_alpha_zoo_state_replay_latest.json"))
    parser.add_argument("--old-hybrid-json", default=str(DEFAULT_OLD_HYBRID_DIR / "hybrid_v35_v36_fixed_inputs_latest.json"))
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--entry-quantile", type=float, default=0.9)
    parser.add_argument("--max-ledger-records-per-factor-side-split", type=int, default=120)
    parser.add_argument("--allocation-fraction", type=float, default=0.10)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--reuse-stage-artifacts",
        action="store_true",
        help="Only reassemble the top-level comparison from already-written Alpha/Hybrid stage artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload_from_stage_artifacts(args) if args.reuse_stage_artifacts else build_payload(args)
    outputs = write_outputs(payload, Path(args.output_dir).expanduser().resolve())
    print(json.dumps({**outputs, "peak_rss_mib": payload["memory_summary"]["peak_rss_mib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
