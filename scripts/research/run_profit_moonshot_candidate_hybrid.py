#!/usr/bin/env python3
"""Build a candidate-derived hybrid strategy for the profit-moonshot sleeve set.

This runner is deliberately separate from the legacy hybrid-online benchmark:
it takes the freshly tuned/liquidation-aware profit-moonshot candidate rows,
reconstructs their sleeve return streams from the same refreshed market data,
tunes a deterministic online hybrid allocator on train/validation only, and
reports locked-OOS as report-only/gate-only evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.portfolio_split_contract import (  # noqa: E402
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
)

TUNER_PATH = REPO_ROOT / "scripts/research/tune_profit_moonshot_fresh_portfolio.py"
FRESH_PATH = REPO_ROOT / "scripts/research/replay_profit_moonshot_fresh_start.py"
HYBRID_PATH = REPO_ROOT / "scripts/research/run_hybrid_online_portfolio.py"
LIQUIDATION_PATH = REPO_ROOT / "scripts/research/run_profit_moonshot_liquidation_aware_validation.py"

DEFAULT_ROOT = REPO_ROOT / "var/reports/profit_moonshot_20260501/live_final_selection_20260510"
DEFAULT_CANDIDATE_PORTFOLIO_JSON = (
    DEFAULT_ROOT / "candidate_portfolio/fresh_portfolio_tuning_latest.json"
)
DEFAULT_LIQUIDATION_JSON = (
    DEFAULT_ROOT / "liquidation_validation/liquidation_aware_current_base_latest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "candidate_hybrid"
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market_parquet"
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT"
RUN_NAME = "profit_moonshot_candidate_hybrid"
STARTING_EQUITY = 10_000.0
LIVE_LEVERAGE_INTEGER_TOLERANCE = 1e-9
_SOURCE_LEDGER_REF_FIELDS = ("source_ledger_refs", "source_search_ledger_refs", "source_ledger_ref")
_RESEARCH_HISTORY_REF_FIELDS = ("research_history_refs", "research_history_ref", "source_history_refs")
_NESTED_HYBRID_SOURCE_TOKENS = (
    "hybrid",
    "portfolio",
    "allocator",
    "governor",
    "leverage_sweep",
    "static_blend",
    "meta_portfolio",
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path | str | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _safe_optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return float(parsed) if math.isfinite(parsed) else None


def _live_integer_leverage_supported(value: Any) -> bool:
    parsed = _safe_optional_float(value)
    if parsed is None or parsed <= 0.0:
        return False
    return math.isclose(parsed, round(parsed), rel_tol=0.0, abs_tol=LIVE_LEVERAGE_INTEGER_TOLERANCE)


def _discarded_non_integer_leverage_source(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(row.get("name") or ""),
        "source_kind": str(row.get("_candidate_hybrid_source_kind") or ""),
        "source_key": str(row.get("_candidate_hybrid_source_key") or ""),
        "source_artifact": str(row.get("_candidate_hybrid_source_artifact") or ""),
        "leverage": _safe_optional_float(row.get("leverage")),
        "reason": "non_integer_or_missing_live_leverage",
    }


def _partition_live_integer_candidate_rows(
    candidate_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for row in candidate_rows:
        if _live_integer_leverage_supported(row.get("leverage")):
            accepted.append(dict(row))
        else:
            discarded.append(_discarded_non_integer_leverage_source(row))
    return accepted, discarded


def _row_refs(row: Mapping[str, Any], fields: Iterable[str]) -> list[str]:
    refs: list[str] = []
    for field in fields:
        value = row.get(field)
        values = value if isinstance(value, list | tuple | set) else [value]
        for item in values:
            token = str(item or "").strip()
            if token and token not in refs:
                refs.append(token)
    return refs


def _source_metadata_present(row: Mapping[str, Any]) -> bool:
    return bool(_row_refs(row, _SOURCE_LEDGER_REF_FIELDS)) and bool(
        _row_refs(row, _RESEARCH_HISTORY_REF_FIELDS)
    )


def _calendar_primary_source_invalid(row: Mapping[str, Any]) -> bool:
    validity = row.get("strategy_validity")
    if isinstance(validity, Mapping):
        return not bool(validity.get("pass"))
    for sleeve in list(row.get("sleeves") or []):
        token = str(sleeve or "").lower()
        if "calendar" in token or "month" in token:
            return True
    primary_signal = str(row.get("primary_signal_type") or "").lower()
    return "calendar" in primary_signal or "seasonality" in primary_signal


def _nested_hybrid_source_invalid(row: Mapping[str, Any]) -> bool:
    """Reject hybrid/meta/portfolio sources as sleeves of a new hybrid.

    Candidate-hybrid construction must be a first-order blend of atomic,
    traceable strategy rows. Feeding an already-hybrid, allocator, portfolio,
    leverage-sweep, or same-family blend back into another hybrid double-counts
    prior selection and hides the true strategy provenance.
    """
    validity = row.get("strategy_validity")
    if isinstance(validity, Mapping):
        primary_signal = str(validity.get("primary_signal_type") or "").lower()
        if any(token in primary_signal for token in _NESTED_HYBRID_SOURCE_TOKENS):
            return True

    text_fields = (
        row.get("name"),
        row.get("mode"),
        row.get("family"),
        row.get("strategy_name"),
        row.get("candidate_family"),
        row.get("candidate_source"),
    )
    tokens = [str(value or "").lower() for value in text_fields]
    tokens.extend(str(sleeve or "").lower() for sleeve in list(row.get("sleeves") or []))
    return any(
        marker in token
        for token in tokens
        for marker in _NESTED_HYBRID_SOURCE_TOKENS
    )


def _split_like_sources(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = [row]
    splits = row.get("splits")
    if isinstance(splits, Mapping):
        for split in splits.values():
            if isinstance(split, Mapping):
                sources.append(split)
                metrics = split.get("metrics")
                if isinstance(metrics, Mapping):
                    sources.append(metrics)
    return sources


def _source_liquidation_unsafe(row: Mapping[str, Any]) -> bool:
    liquidation_count = 0
    min_buffers: list[float] = []
    account_wipeout = False
    for source in _split_like_sources(row):
        liquidation_count = max(
            liquidation_count,
            int(
                _safe_float(
                    source.get("liquidation_count", source.get("liquidation_event_count_total")),
                    0.0,
                )
            ),
        )
        buffer_value = _safe_optional_float(source.get("minimum_margin_buffer"))
        if buffer_value is not None:
            min_buffers.append(buffer_value)
        account_wipeout = account_wipeout or bool(source.get("account_wipeout"))
        for event in list(source.get("liquidation_events") or []):
            if isinstance(event, Mapping):
                account_wipeout = account_wipeout or bool(event.get("account_wipeout"))
    return bool(account_wipeout) or liquidation_count > 0 or any(value <= 0.0 for value in min_buffers)


def _source_candidate_rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _live_integer_leverage_supported(row.get("leverage")):
        reasons.append("non_integer_or_missing_live_leverage")

    validity = row.get("strategy_validity")
    if isinstance(validity, Mapping) and not bool(validity.get("pass")):
        reasons.append("strategy_validity_rejected")
    elif _calendar_primary_source_invalid(row):
        reasons.append("calendar_primary_source_invalid")

    if _nested_hybrid_source_invalid(row):
        reasons.append("nested_hybrid_or_same_family_source_invalid")

    if not _source_metadata_present(row):
        reasons.append("research_history_source_metadata_missing")
    if _source_liquidation_unsafe(row):
        reasons.append("liquidation_source_unsafe")
    return reasons


def _discarded_source_candidate_row(row: Mapping[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "name": str(row.get("name") or ""),
        "source_kind": str(row.get("_candidate_hybrid_source_kind") or ""),
        "source_key": str(row.get("_candidate_hybrid_source_key") or ""),
        "source_artifact": str(row.get("_candidate_hybrid_source_artifact") or ""),
        "leverage": _safe_optional_float(row.get("leverage")),
        "reason": reasons[0] if reasons else "source_live_gate_rejected",
        "reasons": list(reasons),
    }


def _partition_live_source_candidate_rows(
    candidate_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for row in candidate_rows:
        reasons = _source_candidate_rejection_reasons(row)
        if reasons:
            discarded.append(_discarded_source_candidate_row(row, reasons))
        else:
            accepted.append(dict(row))
    return accepted, discarded


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _slug(value: str, *, max_len: int = 120) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    while "__" in out:
        out = out.replace("__", "_")
    return (out.strip("_") or "candidate_hybrid")[:max_len]


def _display_split_name(name: str) -> str:
    return "validation" if str(name) == "val" else str(name)


def _split_metrics(payload: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    splits = dict(payload.get("splits") or {})
    key = "val" if split_name == "validation" and "val" in splits else split_name
    split = dict(splits.get(key) or {})
    metrics = dict(split.get("metrics") or split)
    return metrics


def _candidate_train_val_score(tuner: Any, payload: Mapping[str, Any]) -> float:
    existing = _safe_optional_float(payload.get("train_val_stability_score"))
    if existing is not None:
        return existing
    train = _split_metrics(payload, "train")
    val = _split_metrics(payload, "validation")
    components = {
        "train_monthlyized_return": _monthlyized(tuner, train),
        "validation_monthlyized_return": _monthlyized(tuner, val),
        "train_sharpe": _safe_float(train.get("sharpe")),
        "validation_sharpe": _safe_float(val.get("sharpe")),
        "train_sortino": _safe_float(train.get("sortino")),
        "validation_sortino": _safe_float(val.get("sortino")),
        "train_calmar": _safe_float(train.get("calmar")),
        "validation_calmar": _safe_float(val.get("calmar")),
        "train_max_drawdown": _safe_float(train.get("max_drawdown"), 1.0),
        "validation_max_drawdown": _safe_float(val.get("max_drawdown"), 1.0),
        "leverage": _safe_float(payload.get("leverage"), 1.0),
        "sleeve_count": float(len(list(payload.get("sleeves") or [])) or 1),
    }
    return float(tuner._train_val_stability_score_from_components(components))


def _monthlyized(tuner: Any, metrics: Mapping[str, Any]) -> float:
    return float(tuner._monthlyized_return(dict(metrics)))


def _smart_sortino(tuner: Any, metrics: Mapping[str, Any]) -> float:
    return float(tuner._smart_sortino(dict(metrics)))


def _with_extra_metrics(tuner: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    total_return = _safe_float(out.get("total_return", out.get("return")))
    max_drawdown = _safe_float(out.get("max_drawdown", out.get("mdd")), 0.0)
    out["total_return"] = total_return
    out["max_drawdown"] = max_drawdown
    out["return_mdd"] = total_return / max(1e-9, max_drawdown)
    out["monthlyized_return"] = _monthlyized(tuner, out)
    out["smart_sortino"] = _smart_sortino(tuner, out)
    return out


def _result_score(tuner: Any, result: Mapping[str, Any], *, sleeve_count: int) -> float:
    splits = dict(result.get("split_metrics") or {})
    train = _with_extra_metrics(tuner, dict(splits.get("train") or {}))
    val = _with_extra_metrics(tuner, dict(splits.get("val") or splits.get("validation") or {}))
    components = {
        "train_monthlyized_return": _safe_float(train.get("monthlyized_return")),
        "validation_monthlyized_return": _safe_float(val.get("monthlyized_return")),
        "train_sharpe": _safe_float(train.get("sharpe")),
        "validation_sharpe": _safe_float(val.get("sharpe")),
        "train_sortino": _safe_float(train.get("sortino")),
        "validation_sortino": _safe_float(val.get("sortino")),
        "train_calmar": _safe_float(train.get("calmar")),
        "validation_calmar": _safe_float(val.get("calmar")),
        "train_max_drawdown": _safe_float(train.get("max_drawdown"), 1.0),
        "validation_max_drawdown": _safe_float(val.get("max_drawdown"), 1.0),
        "leverage": 1.0,
        "sleeve_count": float(sleeve_count),
    }
    return float(tuner._train_val_stability_score_from_components(components))


def _iter_candidate_rows(
    *,
    candidate_payload: Mapping[str, Any],
    liquidation_payload: Mapping[str, Any],
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(kind: str, source_key: str, raw: Any, source_artifact: str) -> None:
        if not isinstance(raw, Mapping):
            return
        sleeves = [str(item) for item in list(raw.get("sleeves") or []) if str(item)]
        if not sleeves:
            return
        item = dict(raw)
        item["_candidate_hybrid_source_kind"] = kind
        item["_candidate_hybrid_source_key"] = source_key
        item["_candidate_hybrid_source_artifact"] = source_artifact
        rows.append(item)

    for key in (
        "promoted_candidate",
        "best_deployable_train_validation_retune",
        "selected_by_train_validation_retune",
        "forced_5x",
        "highest_zero_liquidation_integer",
    ):
        add("liquidation", key, liquidation_payload.get(key), "liquidation_json")
    for idx, raw in enumerate(list(liquidation_payload.get("retune_results") or [])[: max_rows]):
        add("liquidation", f"retune_results_{idx:02d}", raw, "liquidation_json")

    for key in (
        "selected_by_train_val_stability",
        "best_success_candidate",
        "selected_best_candidate",
        "selected_by_validation",
        "diagnostic_best_oos",
    ):
        add("candidate_portfolio", key, candidate_payload.get(key), "candidate_portfolio_json")
    for idx, raw in enumerate(list(candidate_payload.get("diagnostic_quarantine") or [])[: max_rows]):
        add("candidate_portfolio", f"diagnostic_quarantine_{idx:02d}", raw, "candidate_portfolio_json")

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], str, float]] = set()
    for item in rows:
        sleeves = tuple(str(name) for name in list(item.get("sleeves") or []))
        key = (sleeves, str(item.get("mode") or "train_val_monthly_return_budget"), round(_safe_float(item.get("leverage"), 1.0), 8))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[: max(1, int(max_rows))]


def _split_indices(fresh: Any, arrays: Mapping[str, Any], split: Any) -> np.ndarray:
    timestamps = arrays["timestamp"]
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = int(datetime.combine(split.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()) - 1
    return np.flatnonzero((timestamps >= start_ts) & (timestamps <= end_ts))


def _timestamp_for_idx(arrays: Mapping[str, Any], idx: int) -> str:
    raw = int(arrays["timestamp"][idx])
    return datetime.fromtimestamp(raw, tz=UTC).isoformat().replace("+00:00", "Z")


def _returns_stream_from_equity(
    *,
    equity: Iterable[float],
    indices: Iterable[int],
    arrays: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stream: list[dict[str, Any]] = []
    previous = STARTING_EQUITY
    for value, idx in zip(list(equity), list(indices), strict=False):
        current = _safe_float(value, previous)
        ret = current / previous - 1.0 if previous > 1e-12 else 0.0
        stream.append({"t": _timestamp_for_idx(arrays, int(idx)), "v": float(ret)})
        previous = max(1e-12, current)
    return stream


def _group_equity(
    *,
    tuner: Any,
    row: Mapping[str, Any],
    curves: list[list[float]],
) -> list[float]:
    mode = str(row.get("mode") or "train_val_monthly_return_budget")
    weights = list(row.get("weights") or [])
    parsed_weights = [_safe_float(item) for item in weights] if weights else None
    if mode in {"validation_return_risk_weight", "validation_drawdown_budget", "cluster_capped_validation_weight"} and not parsed_weights:
        mode = "equal_weight"
    return tuner._combine_equity(
        curves,
        mode=mode,
        weights=parsed_weights,
        leverage=_safe_float(row.get("leverage"), 1.0),
    )


def _build_candidate_hybrid_rows(
    *,
    fresh: Any,
    tuner: Any,
    arrays: Mapping[str, Any],
    splits: list[Any],
    specs_by_name: Mapping[str, Any],
    candidate_rows: list[dict[str, Any]],
    split_curves: Mapping[str, Mapping[str, list[float]]],
    split_payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    active_rows: list[dict[str, Any]] = []
    split_index_by_name = {_display_split_name(split.name): _split_indices(fresh, arrays, split) for split in splits}
    for source_idx, source_row in enumerate(candidate_rows, start=1):
        sleeves = [str(name) for name in list(source_row.get("sleeves") or []) if str(name) in specs_by_name]
        if not sleeves:
            continue
        streams: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
        split_metrics: dict[str, dict[str, Any]] = {}
        fills_by_split: dict[str, int] = {}
        trips_by_split: dict[str, int] = {}
        for split in splits:
            split_name = _display_split_name(split.name)
            raw_name = "val" if split_name == "validation" else split_name
            equity = _group_equity(
                tuner=tuner,
                row=source_row,
                curves=[split_curves[name][raw_name] for name in sleeves],
            )
            streams[raw_name] = _returns_stream_from_equity(
                equity=equity,
                indices=split_index_by_name[split_name],
                arrays=arrays,
            )
            metrics = fresh._metrics_from_equity_totals(
                equity,
                periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
            )
            split_metrics[raw_name] = _with_extra_metrics(tuner, metrics)
            fills_by_split[raw_name] = int(
                sum(int((split_payloads[name][raw_name]).get("fills") or 0) for name in sleeves)
            )
            trips_by_split[raw_name] = int(
                sum(int((split_payloads[name][raw_name]).get("round_trips") or 0) for name in sleeves)
            )
        candidate_id = f"candidate_hybrid_input_{source_idx:02d}_{_slug(str(source_row.get('name') or source_idx), max_len=80)}"
        active_rows.append(
            {
                "candidate_id": candidate_id,
                "name": candidate_id,
                "strategy_class": "ProfitMoonshotCandidatePortfolio",
                "strategy_timeframe": "1h_to_1d_compounded",
                "family": "profit_moonshot_candidate",
                "symbols": list(arrays.get("symbols") or []),
                "return_streams": streams,
                "metadata": {
                    "source_name": str(source_row.get("name") or ""),
                    "source_kind": str(source_row.get("_candidate_hybrid_source_kind") or ""),
                    "source_key": str(source_row.get("_candidate_hybrid_source_key") or ""),
                    "source_artifact": str(source_row.get("_candidate_hybrid_source_artifact") or ""),
                    "source_ledger_refs": _row_refs(source_row, _SOURCE_LEDGER_REF_FIELDS),
                    "source_search_ledger_refs": _row_refs(source_row, _SOURCE_LEDGER_REF_FIELDS),
                    "research_history_refs": _row_refs(source_row, _RESEARCH_HISTORY_REF_FIELDS),
                    "strategy_validity": dict(source_row.get("strategy_validity") or {"pass": True}),
                    "liquidation_gate": not _source_liquidation_unsafe(source_row),
                    "mode": str(source_row.get("mode") or "train_val_monthly_return_budget"),
                    "leverage": _safe_float(source_row.get("leverage"), 1.0),
                    "weights": [_safe_float(item) for item in list(source_row.get("weights") or [])],
                    "sleeves": sleeves,
                    "source_train_val_score": _candidate_train_val_score(tuner, source_row),
                    "max_weight_cap": 0.65,
                },
                "train": {**split_metrics["train"], "fills": fills_by_split["train"], "round_trips": trips_by_split["train"]},
                "val": {**split_metrics["val"], "fills": fills_by_split["val"], "round_trips": trips_by_split["val"]},
                "oos": {**split_metrics["oos"], "fills": fills_by_split["oos"], "round_trips": trips_by_split["oos"]},
            }
        )
    return active_rows


def _cash_row_from_active(hybrid: Any, active_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_days = sorted(set().union(*(hybrid._merged_daily_map(row["return_streams"]).keys() for row in active_rows)))
    return {
        "candidate_id": "risk_off_cash",
        "name": "risk_off_cash",
        "strategy_class": "CashSleeve",
        "strategy_timeframe": "1d",
        "family": "cash",
        "symbols": [],
        "return_streams": hybrid._DYN._portfolio_return_streams_from_daily(all_days, [0.0] * len(all_days)),
        "metadata": {"source_payload_path": ""},
        "train": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        "val": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        "oos": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
    }


def _source_sleeve_scale(active_row: Mapping[str, Any], sleeve_name: str) -> float:
    metadata = dict(active_row.get("metadata") or {})
    sleeves = [str(item) for item in list(metadata.get("sleeves") or [])]
    mode = str(metadata.get("mode") or "train_val_monthly_return_budget")
    weights = [_safe_float(item) for item in list(metadata.get("weights") or [])]
    if sleeve_name in sleeves and weights and len(weights) == len(sleeves):
        return max(0.0, float(weights[sleeves.index(sleeve_name)]))
    if mode == "equal_weight" and sleeves:
        return 1.0 / float(len(sleeves))
    return 1.0


def _allocation_weights_by_date(result: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for allocation in list(result.get("allocations") or []):
        if not isinstance(allocation, Mapping):
            continue
        day = str(allocation.get("date") or "")
        if not day:
            continue
        out[day] = {str(key): _safe_float(value) for key, value in dict(allocation.get("weights") or {}).items()}
    return out


def _day_key_for_idx(arrays: Mapping[str, Any], idx: int) -> str:
    raw = int(arrays["timestamp"][idx])
    return datetime.fromtimestamp(raw, tz=UTC).date().isoformat()


def _weighted_report_leverage(state_entries: list[dict[str, Any]]) -> float:
    open_leverages = [
        _safe_float(entry.get("leverage"), 1.0)
        for entry in state_entries
        if list(getattr(entry.get("state"), "legs", []) or [])
    ]
    if open_leverages:
        return float(max(open_leverages))
    return float(max((_safe_float(entry.get("leverage"), 1.0) for entry in state_entries), default=1.0))


def _dynamic_allocation_for_entry(
    *,
    arrays: Mapping[str, Any],
    idx: int,
    allocation_by_date: Mapping[str, Mapping[str, float]],
    entry: Mapping[str, Any],
) -> float:
    day = _day_key_for_idx(arrays, idx)
    candidate_id = str(entry.get("candidate_id") or "")
    candidate_weight = _safe_float(dict(allocation_by_date.get(day) or {}).get(candidate_id), 0.0)
    sleeve_scale = _safe_float(entry.get("sleeve_scale"), 1.0)
    return max(0.0, float(candidate_weight) * float(sleeve_scale))


def _run_dynamic_liquidation_split(
    *,
    liq: Any,
    fresh: Any,
    specs_by_name: Mapping[str, Any],
    active_rows: list[dict[str, Any]],
    arrays: Mapping[str, Any],
    split: Any,
    allocation_by_date: Mapping[str, Mapping[str, float]],
    model: Any,
) -> dict[str, Any]:
    split_name = _display_split_name(split.name)
    timestamps = arrays["timestamp"]
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = int(datetime.combine(split.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()) - 1
    indices = np.flatnonzero((timestamps >= start_ts) & (timestamps <= end_ts))
    if indices.size == 0:
        return {
            "metrics": {},
            "liquidation_events": [],
            "margin_snapshots": [],
            **liq._split_margin_summary(split_name=split_name, snapshots=[], liquidation_events=[]),
        }

    state_entries: list[dict[str, Any]] = []
    for row in active_rows:
        candidate_id = str(row.get("candidate_id") or row.get("name") or "")
        metadata = dict(row.get("metadata") or {})
        source_leverage = max(0.0, _safe_float(metadata.get("leverage"), 1.0))
        for sleeve_name in [str(item) for item in list(metadata.get("sleeves") or [])]:
            spec = specs_by_name.get(sleeve_name)
            if spec is None:
                continue
            state_entries.append(
                {
                    "candidate_id": candidate_id,
                    "sleeve_name": sleeve_name,
                    "sleeve_scale": _source_sleeve_scale(row, sleeve_name),
                    "leverage": source_leverage,
                    "state": liq.SleeveState(spec=spec, legs=[]),
                }
            )
    states = [entry["state"] for entry in state_entries]
    cash = STARTING_EQUITY
    equity_history: list[float] = []
    margin_snapshots: list[dict[str, Any]] = []
    liquidation_events: list[dict[str, Any]] = []

    for raw_idx in indices:
        idx = int(raw_idx)
        cash = liq._apply_funding_cost(cash=cash, states=states, arrays=arrays, idx=idx, model=model, fresh=fresh)

        for entry in state_entries:
            state = entry["state"]
            if not state.legs:
                continue
            state.bars_held += 1
            state_events: list[dict[str, Any]] = []
            override_prices: dict[str, float] = {}
            source_leverage = max(1e-9, _safe_float(entry.get("leverage"), 1.0))
            for leg in list(state.legs):
                prefix = liq._symbol_prefix(fresh, leg.symbol)
                high = liq._array_value(arrays, f"{prefix}_high", idx, leg.entry_price)
                low = liq._array_value(arrays, f"{prefix}_low", idx, leg.entry_price)
                event = liq._intrabar_liquidation_event(
                    leg,
                    high=high,
                    low=low,
                    leverage=source_leverage,
                    model=model,
                    split_name=split_name,
                    timestamp=liq._timestamp_for_idx(arrays, idx),
                )
                if event is not None:
                    event["candidate_id"] = str(entry.get("candidate_id") or "")
                    event["source_sleeve"] = str(entry.get("sleeve_name") or "")
                    state_events.append(event)
                    override_prices[leg.symbol] = _safe_float(event.get("liquidation_price"), leg.entry_price)
            if state_events:
                pre_equity = liq._portfolio_equity(cash, states, arrays, idx, fresh)
                running_peak = max([STARTING_EQUITY, *equity_history, pre_equity])
                closed_legs = len(state.legs)
                cash = liq._close_state_at_idx(
                    cash=cash,
                    state=state,
                    arrays=arrays,
                    idx=idx,
                    fresh=fresh,
                    model=model,
                    liquidation=True,
                    override_prices=override_prices,
                )
                post_equity = liq._portfolio_equity(cash, states, arrays, idx, fresh)
                liq._annotate_liquidation_events_with_equity_impact(
                    state_events,
                    pre_equity=pre_equity,
                    post_equity=post_equity,
                    running_peak=running_peak,
                    closed_legs=closed_legs,
                )
                liquidation_events.extend(state_events)
                state.cooldown = max(0, int(state.spec.cooldown_bars))

        report_leverage = _weighted_report_leverage(state_entries)
        pre_exit_snapshot = liq._margin_snapshot(
            cash=cash,
            states=states,
            arrays=arrays,
            idx=idx,
            split_name=split_name,
            leverage=report_leverage,
            model=model,
            fresh=fresh,
        )
        if liq._open_legs(states) and _safe_float(pre_exit_snapshot.get("margin_buffer"), 1.0) <= 0.0:
            cross_events = [
                {
                    "split": split_name,
                    "timestamp": liq._timestamp_for_idx(arrays, idx),
                    "reason": "candidate_hybrid_cross_margin_buffer_non_positive",
                    "margin_buffer": pre_exit_snapshot["margin_buffer"],
                    "margin_ratio": pre_exit_snapshot["margin_ratio"],
                    "leverage": report_leverage,
                }
            ]
            pre_equity = liq._portfolio_equity(cash, states, arrays, idx, fresh)
            running_peak = max([STARTING_EQUITY, *equity_history, pre_equity])
            closed_legs = len(liq._open_legs(states))
            for state in states:
                cash = liq._close_state_at_idx(
                    cash=cash,
                    state=state,
                    arrays=arrays,
                    idx=idx,
                    fresh=fresh,
                    model=model,
                    liquidation=True,
                )
            post_equity = liq._portfolio_equity(cash, states, arrays, idx, fresh)
            liq._annotate_liquidation_events_with_equity_impact(
                cross_events,
                pre_equity=pre_equity,
                post_equity=post_equity,
                running_peak=running_peak,
                closed_legs=closed_legs,
            )
            liquidation_events.extend(cross_events)

        for entry in state_entries:
            state = entry["state"]
            current_allocation = _dynamic_allocation_for_entry(
                arrays=arrays,
                idx=idx,
                allocation_by_date=allocation_by_date,
                entry=entry,
            )
            if state.legs and current_allocation <= 1e-12:
                cash = liq._close_state_at_idx(
                    cash=cash,
                    state=state,
                    arrays=arrays,
                    idx=idx,
                    fresh=fresh,
                    model=model,
                )
                state.cooldown = max(0, int(state.spec.cooldown_bars))

            if state.legs:
                exit_reason = ""
                override_prices: dict[str, float] = {}
                if liq._is_spread_state(state):
                    spread_return = liq._state_unrealized_pnl(arrays, state, idx, fresh) / max(
                        1e-9, float(state.gross_entry_notional)
                    )
                    if float(state.spec.stop_loss_pct) > 0.0 and spread_return <= -float(state.spec.stop_loss_pct):
                        exit_reason = "stop"
                    elif float(state.spec.take_profit_pct) > 0.0 and spread_return >= float(state.spec.take_profit_pct):
                        exit_reason = "take_profit"
                    elif state.bars_held >= int(state.spec.hold_bars):
                        exit_reason = "max_hold"
                else:
                    leg = state.legs[0]
                    prefix = liq._symbol_prefix(fresh, leg.symbol)
                    close = liq._array_value(arrays, f"{prefix}_close", idx, leg.entry_price)
                    high = liq._array_value(arrays, f"{prefix}_high", idx, close)
                    low = liq._array_value(arrays, f"{prefix}_low", idx, close)
                    open_ = liq._array_value(arrays, f"{prefix}_open", idx, close)
                    side = str(leg.side).upper()
                    if side == "LONG":
                        state.best_price = max(float(state.best_price or leg.entry_price), high if math.isfinite(high) else close)
                        stop_pct = float(state.position_stop_loss_pct)
                        base_stop = float(leg.entry_price) * (1.0 - stop_pct) if stop_pct > 0.0 else -math.inf
                        trail_stop = state.best_price * (1.0 - stop_pct) if stop_pct > 0.0 else -math.inf
                        stop = max(base_stop, trail_stop)
                        take = (
                            float(leg.entry_price) * (1.0 + float(state.position_take_profit_pct))
                            if float(state.position_take_profit_pct) > 0.0
                            else math.inf
                        )
                        if low <= stop:
                            exit_reason = "stop"
                            override_prices[leg.symbol] = min(open_, stop) if open_ < stop else stop
                        elif high >= take:
                            exit_reason = "take_profit"
                            override_prices[leg.symbol] = max(open_, take) if open_ > take else take
                        elif state.bars_held >= int(state.spec.hold_bars):
                            exit_reason = "max_hold"
                    elif side == "SHORT":
                        state.best_price = min(float(state.best_price or leg.entry_price), low if math.isfinite(low) else close)
                        stop_pct = float(state.position_stop_loss_pct)
                        base_stop = float(leg.entry_price) * (1.0 + stop_pct) if stop_pct > 0.0 else math.inf
                        trail_stop = state.best_price * (1.0 + stop_pct) if stop_pct > 0.0 else math.inf
                        stop = min(base_stop, trail_stop)
                        take = (
                            float(leg.entry_price) * (1.0 - float(state.position_take_profit_pct))
                            if float(state.position_take_profit_pct) > 0.0
                            else -math.inf
                        )
                        if high >= stop:
                            exit_reason = "stop"
                            override_prices[leg.symbol] = max(open_, stop) if open_ > stop else stop
                        elif low <= take:
                            exit_reason = "take_profit"
                            override_prices[leg.symbol] = min(open_, take) if open_ < take else take
                        elif state.bars_held >= int(state.spec.hold_bars):
                            exit_reason = "max_hold"
                if exit_reason:
                    cash = liq._close_state_at_idx(
                        cash=cash,
                        state=state,
                        arrays=arrays,
                        idx=idx,
                        fresh=fresh,
                        model=model,
                        override_prices=override_prices,
                    )
                    state.cooldown = max(0, int(state.spec.cooldown_bars))

            if state.legs:
                continue
            if state.cooldown > 0:
                state.cooldown -= 1
                continue
            allocation_scale = _dynamic_allocation_for_entry(
                arrays=arrays,
                idx=idx,
                allocation_by_date=allocation_by_date,
                entry=entry,
            )
            if allocation_scale <= 1e-12:
                continue
            if liq._is_spread_state(state):
                long_symbol, short_symbol, direction, _reason = liq._spread_signal_for_state(fresh, state, arrays, idx)
                if not long_symbol or not short_symbol or not direction:
                    continue
                hedge = max(0.0, float(state.spec.spread_hedge_ratio))
                if direction == "LONG_SPREAD":
                    plans = (
                        (long_symbol, "BUY", max(0.0, float(state.spec.long_allocation_scale)) * allocation_scale),
                        (short_symbol, "SELL", max(0.0, float(state.spec.short_allocation_scale)) * hedge * allocation_scale),
                    )
                else:
                    plans = (
                        (long_symbol, "SELL", max(0.0, float(state.spec.long_allocation_scale)) * allocation_scale),
                        (short_symbol, "BUY", max(0.0, float(state.spec.short_allocation_scale)) * hedge * allocation_scale),
                    )
                expected_orders = 2
            else:
                symbol, side, _reason = liq._single_leg_signal_for_state(fresh, state, arrays, idx)
                if not symbol or not side:
                    continue
                action = "BUY" if str(side).upper() == "LONG" else "SELL"
                scale = max(0.0, float(fresh._side_allocation_scale(state.spec, side))) * allocation_scale
                plans = ((symbol, action, scale),)
                expected_orders = 1
            equity = liq._portfolio_equity(cash, states, arrays, idx, fresh)
            source_leverage = max(1e-9, _safe_float(entry.get("leverage"), 1.0))
            orders = [
                order
                for symbol, action, scale in plans
                if (
                    order := liq._plan_order(
                        symbol=symbol,
                        action=action,
                        scale=scale,
                        idx=idx,
                        equity=equity,
                        leverage=source_leverage,
                        arrays=arrays,
                        fresh=fresh,
                        model=model,
                    )
                )
                is not None
            ]
            if len(orders) != expected_orders:
                continue
            state.gross_entry_notional = sum(abs(float(order["qty"]) * float(order["fill"])) for order in orders)
            for order in orders:
                cash -= abs(float(order["qty"]) * float(order["fill"])) * float(order["fee_rate"])
                state.legs.append(
                    liq.OpenLeg(
                        sleeve=f"{entry.get('candidate_id')}:{state.spec.name}",
                        symbol=str(order["symbol"]),
                        side=str(order["side"]),
                        qty=float(order["qty"]),
                        entry_price=float(order["fill"]),
                    )
                )
            state.fills += len(orders)
            state.entry_equity = liq._portfolio_equity(cash, states, arrays, idx, fresh)
            if not liq._is_spread_state(state) and state.legs:
                leg = state.legs[0]
                prefix = liq._symbol_prefix(fresh, leg.symbol)
                state.position_stop_loss_pct = float(fresh._entry_stop_pct(state.spec, arrays, prefix, idx))
                state.position_take_profit_pct = float(state.spec.take_profit_pct)
                state.best_price = float(leg.entry_price)
            state.bars_held = 0

        snapshot = liq._margin_snapshot(
            cash=cash,
            states=states,
            arrays=arrays,
            idx=idx,
            split_name=split_name,
            leverage=_weighted_report_leverage(state_entries),
            model=model,
            fresh=fresh,
        )
        margin_snapshots.append(snapshot)
        equity_history.append(float(snapshot["equity"]))

    if indices.size:
        last_idx = int(indices[-1])
        for state in states:
            cash = liq._close_state_at_idx(
                cash=cash,
                state=state,
                arrays=arrays,
                idx=last_idx,
                fresh=fresh,
                model=model,
            )
        if equity_history:
            equity_history[-1] = float(cash)

    metrics = fresh._metrics_from_equity_totals(equity_history, periods=int(fresh.HOURLY_PERIODS_PER_YEAR))
    summary = liq._split_margin_summary(
        split_name=split_name,
        snapshots=margin_snapshots,
        liquidation_events=liquidation_events,
    )
    return {
        "metrics": metrics,
        "round_trips": int(sum(state.round_trips for state in states)),
        "fills": int(sum(state.fills for state in states)),
        "final_equity": float(equity_history[-1]) if equity_history else STARTING_EQUITY,
        "liquidation_events": liquidation_events[:25],
        "liquidation_event_count_total": len(liquidation_events),
        "margin_snapshot_count": len(margin_snapshots),
        "margin_tail": margin_snapshots[-5:],
        "dynamic_allocation_replay": True,
        "replay_model": "entry_scaled_candidate_hybrid_allocations_zero_weight_forced_close",
        **summary,
    }


def _dynamic_liquidation_replay(
    *,
    liq: Any,
    fresh: Any,
    specs_by_name: Mapping[str, Any],
    active_rows: list[dict[str, Any]],
    arrays: Mapping[str, Any],
    splits: list[Any],
    result: Mapping[str, Any],
    model: Any,
) -> dict[str, Any]:
    allocation_by_date = _allocation_weights_by_date(result)
    return {
        _display_split_name(split.name): _run_dynamic_liquidation_split(
            liq=liq,
            fresh=fresh,
            specs_by_name=specs_by_name,
            active_rows=active_rows,
            arrays=arrays,
            split=split,
            allocation_by_date=allocation_by_date,
            model=model,
        )
        for split in splits
    }


def _attach_liquidation_evidence(record: dict[str, Any], replay: Mapping[str, Any], *, tuner: Any) -> dict[str, Any]:
    out = dict(record)
    splits = {key: dict(value) for key, value in dict(out.get("splits") or {}).items()}
    for split_name in ("train", "validation", "oos"):
        target = dict(splits.get(split_name) or {})
        evidence = dict(replay.get(split_name) or {})
        metrics = dict(evidence.get("metrics") or {})
        target.update(
            {
                "liquidation_count": int(_safe_float(evidence.get("liquidation_count"), 0.0)),
                "liquidation_event_count_total": int(
                    _safe_float(evidence.get("liquidation_event_count_total"), 0.0)
                ),
                "liquidation_events": list(evidence.get("liquidation_events") or []),
                "minimum_margin_buffer": _safe_optional_float(evidence.get("minimum_margin_buffer")),
                "minimum_margin_ratio": _safe_optional_float(evidence.get("minimum_margin_ratio")),
                "margin_buffer_positive": bool(evidence.get("margin_buffer_positive", False)),
                "maximum_liquidation_event_drawdown": _safe_float(
                    evidence.get("maximum_liquidation_event_drawdown"), 0.0
                ),
                "maximum_liquidation_equity_loss_fraction": _safe_float(
                    evidence.get("maximum_liquidation_equity_loss_fraction"), 0.0
                ),
                "dynamic_liquidation_replay_metrics": _with_extra_metrics(tuner, metrics),
                "dynamic_liquidation_replay_final_equity": _safe_optional_float(evidence.get("final_equity")),
                "dynamic_liquidation_replay_fills": int(_safe_float(evidence.get("fills"), 0.0)),
                "dynamic_liquidation_replay_round_trips": int(_safe_float(evidence.get("round_trips"), 0.0)),
                "dynamic_liquidation_replay_margin_snapshot_count": int(
                    _safe_float(evidence.get("margin_snapshot_count"), 0.0)
                ),
                "dynamic_liquidation_replay_margin_tail": list(evidence.get("margin_tail") or []),
            }
        )
        splits[split_name] = target
    out["splits"] = splits
    out["liquidation_evidence_note"] = (
        "candidate-hybrid allocator was replayed with dynamic daily candidate weights, "
        "source candidate leverage, conservative Binance-style fees/slippage/funding/stress "
        "buffers, and intrabar high/low liquidation checks."
    )
    out["liquidation_replay_policy"] = {
        "evidence_available": True,
        "engine": "dynamic_weight_candidate_hybrid_margin_replay_v1",
        "selection_inputs": ["train", "validation"],
        "locked_oos": "report_only_gate_only",
        "uses_locked_oos_for_selection": False,
    }
    return out


def _attach_source_metadata(record: dict[str, Any], active_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    out = dict(record)
    ledger_refs: list[str] = []
    history_refs: list[str] = []
    for row in active_rows:
        metadata = dict(row.get("metadata") or {})
        for ref in list(metadata.get("source_ledger_refs") or []):
            token = str(ref or "").strip()
            if token and token not in ledger_refs:
                ledger_refs.append(token)
        for ref in list(metadata.get("research_history_refs") or []):
            token = str(ref or "").strip()
            if token and token not in history_refs:
                history_refs.append(token)
    out["source_ledger_refs"] = ledger_refs
    out["research_history_refs"] = history_refs
    out["source_live_gate_policy"] = {
        "all_active_sources_passed_before_hybrid_construction": True,
        "integer_leverage_required": True,
        "calendar_primary_source_rejected": True,
        "research_history_source_metadata_required": True,
        "liquidation_unsafe_source_rejected": True,
    }
    return out


def _dynamic_train_val_components(tuner: Any, record: Mapping[str, Any]) -> dict[str, float]:
    splits = dict(record.get("splits") or {})
    train = dict(dict(splits.get("train") or {}).get("dynamic_liquidation_replay_metrics") or {})
    validation = dict(dict(splits.get("validation") or {}).get("dynamic_liquidation_replay_metrics") or {})
    return {
        "train_monthlyized_return": _monthlyized(tuner, train),
        "validation_monthlyized_return": _monthlyized(tuner, validation),
        "train_sharpe": _safe_float(train.get("sharpe")),
        "validation_sharpe": _safe_float(validation.get("sharpe")),
        "train_sortino": _safe_float(train.get("sortino")),
        "validation_sortino": _safe_float(validation.get("sortino")),
        "train_calmar": _safe_float(train.get("calmar")),
        "validation_calmar": _safe_float(validation.get("calmar")),
        "train_max_drawdown": _safe_float(train.get("max_drawdown"), 1.0),
        "validation_max_drawdown": _safe_float(validation.get("max_drawdown"), 1.0),
        "leverage": _safe_float(record.get("leverage"), 1.0),
        "sleeve_count": float(len(list(record.get("sleeves") or [])) or _safe_float(record.get("sleeve_count"), 1.0)),
    }


def _use_dynamic_replay_score(tuner: Any, record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    allocator_score = _safe_optional_float(out.get("train_val_stability_score"))
    if allocator_score is not None:
        out["allocator_train_val_stability_score"] = allocator_score
    components = _dynamic_train_val_components(tuner, out)
    dynamic_score = float(tuner._train_val_stability_score_from_components(components))
    out["train_val_stability"] = components
    out["train_val_stability_score"] = dynamic_score
    out["train_val_stability_formula_score"] = dynamic_score
    out["train_val_stability_formula"] = "dynamic_liquidation_replay_train_validation_score_v1"
    out["selection_score_note"] = (
        "Final candidate-hybrid comparison score uses dynamic liquidation replay "
        "train/validation metrics, not the pre-margin allocator-only score."
    )
    return out


def _with_extra_metrics_from_raw(metrics: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    total_return = _safe_float(out.get("total_return", out.get("return")))
    max_drawdown = _safe_float(out.get("max_drawdown", out.get("mdd")), 0.0)
    out["total_return"] = total_return
    out["max_drawdown"] = max_drawdown
    out["return_mdd"] = total_return / max(1e-9, max_drawdown)
    return out


def _config_grid(hybrid: Any) -> list[Any]:
    base = asdict(hybrid.HybridOnlineConfig())
    configs = []
    for variant in ("fixed_default", "disagreement_switching"):
        for lookback in (5, 13, 21):
            for min_positive in (0.0, 0.10):
                for temp in (0.90,):
                    for default_boost in (0.0, 0.12, 0.25):
                        cfg = {
                            **base,
                            "variant": variant,
                            "warmup_days": max(lookback, 7),
                            "lookback_days": lookback,
                            "min_positive_score": min_positive,
                            "score_temperature": temp,
                            "default_boost": default_boost,
                            "diversified_weight_cap": 0.75,
                            "pair_weight_cap": 0.20,
                            "use_current_health_priors": False,
                        }
                        configs.append(hybrid.HybridOnlineConfig(**cfg))
    return configs


def _row_splits_from_result(tuner: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    split_metrics = dict(result.get("split_metrics") or {})
    out: dict[str, Any] = {}
    for source_name, target_name in (("train", "train"), ("val", "validation"), ("oos", "oos")):
        out[target_name] = {
            "metrics": _with_extra_metrics(tuner, dict(split_metrics.get(source_name) or {})),
            "fills": 0,
            "round_trips": 0,
            "liquidation_count": 0,
            "liquidation_event_count_total": 0,
            "liquidation_events": [],
        }
    return out


def _candidate_hybrid_record(
    *,
    tuner: Any,
    result: Mapping[str, Any],
    score: float,
    rank: int,
    config: Any,
    default_name: str,
    selected: bool,
) -> dict[str, Any]:
    splits = _row_splits_from_result(tuner, result)
    return {
        "name": f"candidate_hybrid_online_rank_{rank:02d}_{_slug(default_name, max_len=60)}",
        "mode": "candidate_online_hybrid",
        "leverage": 1.0,
        "sleeves": sorted(
            {
                sleeve
                for allocation in list(result.get("allocations") or [])
                for sleeve in dict(allocation.get("weights") or {})
            }
        ),
        "sleeve_count": len(
            {
                sleeve
                for allocation in list(result.get("allocations") or [])
                for sleeve in dict(allocation.get("weights") or {})
            }
        ),
        "splits": splits,
        "train_val_stability_score": float(score),
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
            "selection_basis": "train_validation_only_candidate_hybrid_grid",
        },
        "hybrid_config": asdict(config),
        "default_candidate": default_name,
        "selected_by_train_validation": bool(selected),
        "locked_oos_policy": {
            "oos_is_report_only": True,
            "oos_is_gate_only": True,
            "uses_locked_oos_for_selection": False,
        },
        "liquidation_evidence_note": (
            "candidate-hybrid allocator uses reconstructed candidate portfolio return streams; "
            "no dynamic-weight liquidation replay is claimed, so final live writer must keep "
            "this row non-promotable until a dedicated margin replay exists."
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    fresh = _load_module(FRESH_PATH, "profit_moonshot_fresh_replay_for_candidate_hybrid")
    tuner = _load_module(TUNER_PATH, "profit_moonshot_fresh_tuner_for_candidate_hybrid")
    hybrid = _load_module(HYBRID_PATH, "profit_moonshot_legacy_hybrid_module_for_candidate_hybrid")
    liq = _load_module(LIQUIDATION_PATH, "profit_moonshot_liquidation_for_candidate_hybrid")

    candidate_payload = _load_json(args.candidate_portfolio_json)
    liquidation_payload = _load_json(args.liquidation_json)
    oos_end_token = str(args.oos_end_date or candidate_payload.get("oos_end_date") or liquidation_payload.get("oos_end_date") or "")
    if not oos_end_token:
        raise RuntimeError("oos_end_date is required when source artifacts do not carry one")
    oos_end = datetime.fromisoformat(oos_end_token[:10]).date()
    splits = fresh._split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    panel, data_metadata = fresh._joined_panel(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        symbols=symbols,
        start=start,
        end=end,
    )
    arrays = fresh._build_arrays(panel, symbols)
    specs_by_name = {spec.name: spec for spec in fresh._candidate_specs(arrays, symbols)}

    candidate_rows = _iter_candidate_rows(
        candidate_payload=candidate_payload,
        liquidation_payload=liquidation_payload,
        max_rows=int(args.max_candidate_rows),
    )
    raw_source_candidate_row_count = len(candidate_rows)
    candidate_rows, discarded_source_candidate_rows = _partition_live_source_candidate_rows(candidate_rows)
    discarded_non_integer_leverage_sources = [
        dict(row)
        for row in discarded_source_candidate_rows
        if "non_integer_or_missing_live_leverage" in list(row.get("reasons") or [])
    ]
    if not candidate_rows:
        split_config = hybrid.HybridSplitConfig(oos_end=oos_end.isoformat())
        return {
            "artifact_kind": "profit_moonshot_candidate_hybrid",
            "generated_at_utc": _utc_now_iso(),
            "oos_end_date": oos_end.isoformat(),
            "status": "no_live_source_candidates",
            "selection_basis": "candidate_derived_online_hybrid_train_validation_only",
            "selection_policy": {
                "selection_inputs": ["train", "validation"],
                "locked_oos": "report_only_gate_only",
                "uses_locked_oos_for_selection": False,
                "tuning_objective": "frozen_weighted_train_validation_score_v1",
            },
            "split_windows": split_config.as_payload(),
            "candidate_basis": {
                "candidate_portfolio_json": str(args.candidate_portfolio_json),
                "liquidation_json": str(args.liquidation_json),
                "source_candidate_rows": raw_source_candidate_row_count,
                "integer_leverage_source_candidate_rows": 0,
                "live_source_gate_accepted_candidate_rows": 0,
                "source_live_gate_policy": {
                    "integer_leverage_required": True,
                    "calendar_primary_source_rejected": True,
                    "research_history_source_metadata_required": True,
                    "liquidation_unsafe_source_rejected": True,
                },
                "live_integer_leverage_required": True,
                "discarded_non_integer_leverage_source_count": len(
                    discarded_non_integer_leverage_sources
                ),
                "discarded_non_integer_leverage_sources": discarded_non_integer_leverage_sources,
                "discarded_source_candidate_count": len(discarded_source_candidate_rows),
                "discarded_source_candidates": discarded_source_candidate_rows,
                "available_sleeve_count": 0,
                "missing_sleeves": [],
                "active_hybrid_input_count": 0,
                "default_candidates": [],
                "risk_pruned_active_id_prefixes": [],
                "risk_pruned_source_sleeves": [],
            },
            "liquidation_replay": {
                "engine": "dynamic_weight_candidate_hybrid_margin_replay_v1",
                "margin_model": asdict(liq.MarginModel()),
                "split_summaries": {},
            },
            "data_metadata": data_metadata,
            "source_sleeve_metrics": {},
            "selected_by_train_validation": {},
            "best_candidate_hybrid": {},
            "tuning_results": [],
            "full_tuning_result_count": 0,
            "final_allocation": {},
            "peak_rss_mib": _rss_mib(),
            "memory_policy": memory_policy_payload(
                budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
            ),
        }
    required_sleeves = sorted({name for row in candidate_rows for name in list(row.get("sleeves") or [])})
    missing_sleeves = [name for name in required_sleeves if name not in specs_by_name]
    available_sleeves = [name for name in required_sleeves if name in specs_by_name]
    split_curves: dict[str, dict[str, list[float]]] = {}
    split_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for name in available_sleeves:
        split_curves[name] = {}
        split_payloads[name] = {}
        for split in splits:
            result = fresh._run_split(
                spec=specs_by_name[name],
                arrays=arrays,
                split=split,
                include_equity=True,
            )
            raw_name = str(split.name)
            split_curves[name][raw_name] = list(result.get("equity_history") or [])
            split_payloads[name][raw_name] = dict(result)

    active = _build_candidate_hybrid_rows(
        fresh=fresh,
        tuner=tuner,
        arrays=arrays,
        splits=splits,
        specs_by_name=specs_by_name,
        candidate_rows=candidate_rows,
        split_curves=split_curves,
        split_payloads=split_payloads,
    )
    if not active:
        raise RuntimeError("no candidate-hybrid active rows could be reconstructed")
    exclude_active_ids = [str(item).strip() for item in list(args.exclude_active_id or []) if str(item).strip()]
    exclude_source_sleeves = [str(item).strip() for item in list(args.exclude_source_sleeve or []) if str(item).strip()]
    if exclude_active_ids or exclude_source_sleeves:
        filtered_active: list[dict[str, Any]] = []
        for row in active:
            candidate_id = str(row.get("candidate_id") or "")
            metadata = dict(row.get("metadata") or {})
            row_sleeves = {str(item) for item in list(metadata.get("sleeves") or [])}
            excluded_id = any(candidate_id == token or candidate_id.startswith(token) for token in exclude_active_ids)
            excluded_sleeve = bool(row_sleeves.intersection(exclude_source_sleeves))
            if not excluded_id and not excluded_sleeve:
                filtered_active.append(row)
        active = filtered_active
        if not active:
            raise RuntimeError("all candidate-hybrid active rows were removed by risk-pruning filters")
    active.sort(
        key=lambda row: (
            _safe_float(dict(row.get("metadata") or {}).get("source_train_val_score")),
            _safe_float(dict(row.get("val") or {}).get("total_return")),
        ),
        reverse=True,
    )
    active = active[: max(1, int(args.max_active_inputs))]
    rows = [_cash_row_from_active(hybrid, active), *active]
    split_config = hybrid.HybridSplitConfig(oos_end=oos_end.isoformat())

    default_candidates = [str(row.get("candidate_id")) for row in active[: max(1, int(args.default_candidate_count))]]
    tuning_results: list[dict[str, Any]] = []
    for default_name in default_candidates:
        for config in _config_grid(hybrid):
            result = hybrid.run_hybrid_online_allocator(
                rows,
                config=config,
                refreshed_health_metrics=None,
                split_config=split_config,
                default_name=default_name,
            )
            score = _result_score(tuner, result, sleeve_count=len(active))
            split_metrics = dict(result.get("split_metrics") or {})
            train = dict(split_metrics.get("train") or {})
            val = dict(split_metrics.get("val") or {})
            train_val_gate = (
                _safe_float(train.get("total_return")) > 0.0
                and _safe_float(val.get("total_return")) > 0.0
                and _safe_float(train.get("max_drawdown"), 1.0) <= 0.25
                and _safe_float(val.get("max_drawdown"), 1.0) <= 0.25
            )
            tuning_results.append(
                {
                    "score": float(score),
                    "train_val_gate": bool(train_val_gate),
                    "default_name": default_name,
                    "config": asdict(config),
                    "result": {
                        "split_metrics": {
                            name: _with_extra_metrics(tuner, dict(metrics))
                            for name, metrics in split_metrics.items()
                        },
                        "final_allocation": dict(result.get("final_allocation") or {}),
                        "resolved_warmup_days": int(result.get("resolved_warmup_days") or 0),
                    },
                }
            )
    selection_pool = [item for item in tuning_results if bool(item["train_val_gate"])] or tuning_results
    selection_pool.sort(
        key=lambda item: (
            _safe_float(item.get("score")),
            _safe_float(dict(dict(item.get("result") or {}).get("split_metrics") or {}).get("val", {}).get("total_return")),
        ),
        reverse=True,
    )
    best_tuning = selection_pool[0]
    best_config = hybrid.HybridOnlineConfig(**dict(best_tuning["config"]))
    best_result = hybrid.run_hybrid_online_allocator(
        rows,
        config=best_config,
        refreshed_health_metrics=None,
        split_config=split_config,
        default_name=str(best_tuning["default_name"]),
    )
    ranked = sorted(
        tuning_results,
        key=lambda item: (
            bool(item.get("train_val_gate")),
            _safe_float(item.get("score")),
        ),
        reverse=True,
    )
    selected_record = _candidate_hybrid_record(
        tuner=tuner,
        result=best_result,
        score=_result_score(tuner, best_result, sleeve_count=len(active)),
        rank=1,
        config=best_config,
        default_name=str(best_tuning["default_name"]),
        selected=True,
    )
    margin_model = liq.MarginModel()
    dynamic_liquidation_replay = _dynamic_liquidation_replay(
        liq=liq,
        fresh=fresh,
        specs_by_name=specs_by_name,
        active_rows=active,
        arrays=arrays,
        splits=splits,
        result=best_result,
        model=margin_model,
    )
    selected_record = _attach_liquidation_evidence(selected_record, dynamic_liquidation_replay, tuner=tuner)
    selected_record = _use_dynamic_replay_score(tuner, selected_record)
    selected_record = _attach_source_metadata(selected_record, active)
    top_records = [
        _attach_source_metadata(
            _candidate_hybrid_record(
                tuner=tuner,
                result=item["result"],
                score=_safe_float(item.get("score")),
                rank=idx,
                config=hybrid.HybridOnlineConfig(**dict(item["config"])),
                default_name=str(item["default_name"]),
                selected=idx == 1,
            ),
            active,
        )
        for idx, item in enumerate(ranked[: int(args.report_top_n)], start=1)
    ]
    top_records = [selected_record if item.get("name") == selected_record.get("name") else item for item in top_records]
    source_sleeve_metrics = {
        str(row.get("candidate_id")): {
            "source_name": dict(row.get("metadata") or {}).get("source_name"),
            "source_kind": dict(row.get("metadata") or {}).get("source_kind"),
            "mode": dict(row.get("metadata") or {}).get("mode"),
            "leverage": dict(row.get("metadata") or {}).get("leverage"),
            "sleeves": dict(row.get("metadata") or {}).get("sleeves"),
            "source_ledger_refs": dict(row.get("metadata") or {}).get("source_ledger_refs", []),
            "source_search_ledger_refs": dict(row.get("metadata") or {}).get("source_search_ledger_refs", []),
            "research_history_refs": dict(row.get("metadata") or {}).get("research_history_refs", []),
            "strategy_validity": dict(row.get("metadata") or {}).get("strategy_validity", {"pass": True}),
            "liquidation_gate": dict(row.get("metadata") or {}).get("liquidation_gate", True),
            "train": dict(row.get("train") or {}),
            "validation": dict(row.get("val") or {}),
            "oos": dict(row.get("oos") or {}),
        }
        for row in active
    }
    final_allocation = dict(best_result.get("final_allocation") or {})
    payload = {
        "artifact_kind": "profit_moonshot_candidate_hybrid",
        "generated_at_utc": _utc_now_iso(),
        "oos_end_date": oos_end.isoformat(),
        "selection_basis": "candidate_derived_online_hybrid_train_validation_only",
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
            "tuning_objective": "frozen_weighted_train_validation_score_v1",
        },
        "split_windows": split_config.as_payload(),
        "candidate_basis": {
            "candidate_portfolio_json": str(args.candidate_portfolio_json),
            "liquidation_json": str(args.liquidation_json),
            "source_candidate_rows": raw_source_candidate_row_count,
            "integer_leverage_source_candidate_rows": len(candidate_rows),
            "live_source_gate_accepted_candidate_rows": len(candidate_rows),
            "source_live_gate_policy": {
                "integer_leverage_required": True,
                "calendar_primary_source_rejected": True,
                "research_history_source_metadata_required": True,
                "liquidation_unsafe_source_rejected": True,
            },
            "live_integer_leverage_required": True,
            "discarded_non_integer_leverage_source_count": len(discarded_non_integer_leverage_sources),
            "discarded_non_integer_leverage_sources": discarded_non_integer_leverage_sources,
            "discarded_source_candidate_count": len(discarded_source_candidate_rows),
            "discarded_source_candidates": discarded_source_candidate_rows,
            "available_sleeve_count": len(available_sleeves),
            "missing_sleeves": missing_sleeves,
            "active_hybrid_input_count": len(active),
            "default_candidates": default_candidates,
            "risk_pruned_active_id_prefixes": exclude_active_ids,
            "risk_pruned_source_sleeves": exclude_source_sleeves,
        },
        "liquidation_replay": {
            "engine": "dynamic_weight_candidate_hybrid_margin_replay_v1",
            "margin_model": asdict(margin_model),
            "split_summaries": {
                name: {
                    "liquidation_count": int(_safe_float(split.get("liquidation_count"), 0.0)),
                    "minimum_margin_buffer": _safe_optional_float(split.get("minimum_margin_buffer")),
                    "minimum_margin_ratio": _safe_optional_float(split.get("minimum_margin_ratio")),
                    "margin_buffer_positive": bool(split.get("margin_buffer_positive", False)),
                    "maximum_liquidation_event_drawdown": _safe_float(
                        split.get("maximum_liquidation_event_drawdown"), 0.0
                    ),
                    "maximum_liquidation_equity_loss_fraction": _safe_float(
                        split.get("maximum_liquidation_equity_loss_fraction"), 0.0
                    ),
                    "fills": int(_safe_float(split.get("fills"), 0.0)),
                    "round_trips": int(_safe_float(split.get("round_trips"), 0.0)),
                    "final_equity": _safe_optional_float(split.get("final_equity")),
                }
                for name, split in dynamic_liquidation_replay.items()
            },
        },
        "data_metadata": data_metadata,
        "source_sleeve_metrics": source_sleeve_metrics,
        "selected_by_train_validation": selected_record,
        "best_candidate_hybrid": selected_record,
        "tuning_results": top_records,
        "full_tuning_result_count": len(tuning_results),
        "final_allocation": final_allocation,
        "peak_rss_mib": _rss_mib(),
        "memory_policy": memory_policy_payload(budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES),
    }
    return payload


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):+.4%}"


def _fmt_num(value: Any) -> str:
    parsed = _safe_optional_float(value)
    return "" if parsed is None else f"{parsed:.4f}"


def _markdown(payload: Mapping[str, Any]) -> str:
    selected = dict(payload.get("selected_by_train_validation") or {})
    splits = dict(selected.get("splits") or {})
    oos = dict(dict(splits.get("oos") or {}).get("metrics") or {})
    train = dict(dict(splits.get("train") or {}).get("metrics") or {})
    val = dict(dict(splits.get("validation") or {}).get("metrics") or {})
    replay_train = dict(dict(splits.get("train") or {}).get("dynamic_liquidation_replay_metrics") or {})
    replay_val = dict(dict(splits.get("validation") or {}).get("dynamic_liquidation_replay_metrics") or {})
    replay_oos = dict(dict(splits.get("oos") or {}).get("dynamic_liquidation_replay_metrics") or {})
    liquidation_replay = dict(payload.get("liquidation_replay") or {})
    replay_summaries = dict(liquidation_replay.get("split_summaries") or {})
    candidate_basis = dict(payload.get("candidate_basis") or {})
    lines = [
        "# Profit moonshot candidate-derived hybrid",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        f"- oos_end_date: `{payload.get('oos_end_date')}`",
        "- selection basis: train/validation-only candidate hybrid tuning",
        "- locked-OOS: report-only / gate-only",
        "- live leverage policy: integer leverage source rows only",
        "- liquidation replay: dynamic candidate weights + source leverage + conservative Binance-style margin model",
        f"- discarded non-integer/missing leverage sources: `{int(_safe_float(candidate_basis.get('discarded_non_integer_leverage_source_count'), 0.0))}`",
        "",
        "## Selected candidate hybrid",
        "",
        f"- name: `{selected.get('name')}`",
        f"- TV score: `{_fmt_num(selected.get('train_val_stability_score'))}`",
        f"- allocator train/validation/OOS return: `{_fmt_pct(train.get('total_return'))}` / `{_fmt_pct(val.get('total_return'))}` / `{_fmt_pct(oos.get('total_return'))}`",
        f"- replay train return/MDD: `{_fmt_pct(replay_train.get('total_return'))}` / `{_fmt_pct(replay_train.get('max_drawdown'))}`",
        f"- replay validation return/MDD: `{_fmt_pct(replay_val.get('total_return'))}` / `{_fmt_pct(replay_val.get('max_drawdown'))}`",
        f"- replay OOS return/MDD: `{_fmt_pct(replay_oos.get('total_return'))}` / `{_fmt_pct(replay_oos.get('max_drawdown'))}`",
        f"- replay OOS return/MDD ratio: `{_fmt_num(replay_oos.get('return_mdd'))}`",
        f"- replay OOS Sharpe/Sortino/Calmar: `{_fmt_num(replay_oos.get('sharpe'))}` / `{_fmt_num(replay_oos.get('sortino'))}` / `{_fmt_num(replay_oos.get('calmar'))}`",
        f"- liquidation counts train/validation/OOS: `{int(_safe_float(dict(replay_summaries.get('train') or {}).get('liquidation_count')))} / {int(_safe_float(dict(replay_summaries.get('validation') or {}).get('liquidation_count')))} / {int(_safe_float(dict(replay_summaries.get('oos') or {}).get('liquidation_count')))}`",
        f"- minimum margin buffer train/validation/OOS: `{_fmt_num(dict(replay_summaries.get('train') or {}).get('minimum_margin_buffer'))}` / `{_fmt_num(dict(replay_summaries.get('validation') or {}).get('minimum_margin_buffer'))}` / `{_fmt_num(dict(replay_summaries.get('oos') or {}).get('minimum_margin_buffer'))}`",
        f"- minimum margin ratio train/validation/OOS: `{_fmt_num(dict(replay_summaries.get('train') or {}).get('minimum_margin_ratio'))}` / `{_fmt_num(dict(replay_summaries.get('validation') or {}).get('minimum_margin_ratio'))}` / `{_fmt_num(dict(replay_summaries.get('oos') or {}).get('minimum_margin_ratio'))}`",
        "",
        "## Final allocation",
        "",
        f"- date: `{dict(payload.get('final_allocation') or {}).get('date')}`",
        f"- cash_weight: `{_fmt_pct(dict(payload.get('final_allocation') or {}).get('cash_weight'))}`",
    ]
    for name, weight in sorted(
        dict(dict(payload.get("final_allocation") or {}).get("weights") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"- `{name}`: `{_fmt_pct(weight)}`")
    lines.extend(
        [
            "",
            "## Top tuned candidate-hybrid rows",
            "",
            "| rank | name | TV score | train | val | OOS | OOS MDD | OOS R/MDD | Sharpe | Sortino | SmartSort | Calmar |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, item in enumerate(list(payload.get("tuning_results") or []), start=1):
        item_splits = dict(item.get("splits") or {})
        item_train = dict(dict(item_splits.get("train") or {}).get("metrics") or {})
        item_val = dict(dict(item_splits.get("validation") or {}).get("metrics") or {})
        item_oos = dict(dict(item_splits.get("oos") or {}).get("metrics") or {})
        lines.append(
            f"| {idx} | `{item.get('name')}` | {_fmt_num(item.get('train_val_stability_score'))} | "
            f"{_fmt_pct(item_train.get('total_return'))} | {_fmt_pct(item_val.get('total_return'))} | "
            f"{_fmt_pct(item_oos.get('total_return'))} | {_fmt_pct(item_oos.get('max_drawdown'))} | "
            f"{_fmt_num(item_oos.get('return_mdd'))} | {_fmt_num(item_oos.get('sharpe'))} | "
            f"{_fmt_num(item_oos.get('sortino'))} | {_fmt_num(item_oos.get('smart_sortino'))} | "
            f"{_fmt_num(item_oos.get('calmar'))} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-portfolio-json", default=str(DEFAULT_CANDIDATE_PORTFOLIO_JSON))
    parser.add_argument("--liquidation-json", default=str(DEFAULT_LIQUIDATION_JSON))
    parser.add_argument("--market-root", default=str(DEFAULT_MARKET_ROOT))
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--oos-end-date", default="")
    parser.add_argument("--max-candidate-rows", type=int, default=22)
    parser.add_argument("--max-active-inputs", type=int, default=12)
    parser.add_argument("--default-candidate-count", type=int, default=3)
    parser.add_argument("--report-top-n", type=int, default=12)
    parser.add_argument("--exclude-active-id", action="append", default=[])
    parser.add_argument("--exclude-source-sleeve", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name=RUN_NAME,
        output_dir=output_dir,
        input_path=args.candidate_portfolio_json,
        metadata={
            "script": Path(__file__).name,
            "max_candidate_rows": int(args.max_candidate_rows),
            "max_active_inputs": int(args.max_active_inputs),
            "exclude_active_id": list(args.exclude_active_id or []),
            "exclude_source_sleeve": list(args.exclude_source_sleeve or []),
            "locked_oos_policy": "report_only_gate_only",
        },
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    finalized = False
    try:
        memory_guard.checkpoint(
            "start",
            {
                "candidate_portfolio_json": str(args.candidate_portfolio_json),
                "liquidation_json": str(args.liquidation_json),
            },
        )
        payload = build_payload(args)
        memory_summary = memory_guard.finalize(
            status="completed",
            context={
                "selected": str(dict(payload.get("selected_by_train_validation") or {}).get("name") or ""),
                "full_tuning_result_count": int(payload.get("full_tuning_result_count") or 0),
            },
        )
        finalized = True
        memory_summary["summary_path"] = str(memory_guard.summary_path)
        payload["rss_log_path"] = str(memory_guard.rss_log_path)
        payload["memory_summary_path"] = str(memory_guard.summary_path)
        payload["memory_summary"] = memory_summary
        payload["peak_rss_mib"] = max(
            _safe_float(payload.get("peak_rss_mib")),
            _safe_float(memory_summary.get("peak_rss_bytes")) / (1024.0 * 1024.0),
        )
        json_path = output_dir / "candidate_hybrid_latest.json"
        md_path = output_dir / "candidate_hybrid_latest.md"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        timestamped_json = output_dir / f"candidate_hybrid_{timestamp}.json"
        timestamped_md = output_dir / f"candidate_hybrid_{timestamp}.md"
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        json_path.write_text(text, encoding="utf-8")
        timestamped_json.write_text(text, encoding="utf-8")
        markdown = _markdown(payload)
        md_path.write_text(markdown, encoding="utf-8")
        timestamped_md.write_text(markdown, encoding="utf-8")
    except Exception as exc:
        if not finalized:
            memory_guard.finalize(status="failed", error=str(exc), context={"script": Path(__file__).name})
        raise
    finally:
        memory_guard.release()
    print(
        json.dumps(
            {
                "json": str(output_dir / "candidate_hybrid_latest.json"),
                "markdown": str(output_dir / "candidate_hybrid_latest.md"),
                "peak_rss_mib": payload["peak_rss_mib"],
                "status": str(payload.get("status") or "completed"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
