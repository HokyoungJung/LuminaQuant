#!/usr/bin/env python3
"""Monthly refit walk-forward for top no-OOS 69-asset Alpha Zoo candidates.

Protocol:
* refit date is the first day of each OOS month;
* train is expanding from ``--train-start`` to the bar immediately before the
  2-month validation window;
* validation is the prior two calendar months;
* locked OOS is the next one calendar month, truncated to latest available data;
* candidate/parameter search never sees the OOS month.

This runner is deliberately report-only.  It rebuilds the high no-OOS strategy
families that were discussed for live handoff:
1. per-asset/profile Optuna source profiles + static guarded + v3.5/v3.6 hybrid;
2. individual-sleeve-first robust portfolios, then static/v3.5/v3.6 hybrid;
3. strict live-efficiency repair pass from the same source params;
4. relaxed MDD-guarded efficiency repair pass from the same source params.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.research_universe import BINANCE_EXTENDED_RESEARCH_SYMBOLS  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_efficiency_repair_optuna as strict_eff  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as profile69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna as relaxed_eff  # noqa: E402
from scripts.research import run_alpha_zoo_integer_leverage_hybrid_decision as grid_hybrid  # noqa: E402
from scripts.research import run_alpha_zoo_integer_leverage_optuna_hybrid_decision as optuna_hybrid  # noqa: E402

DEFAULT_OUTPUT_JSON = Path("/tmp/lumina_monthly_refit_walkforward_1m_oos_latest.json")
DEFAULT_OUTPUT_MD = Path("/tmp/lumina_monthly_refit_walkforward_1m_oos_latest.md")
DEFAULT_FIRST_OOS_START = "2025-09-01"
DEFAULT_TRAIN_START = "2025-01-01"
DEFAULT_ASSET_TRIALS = 6
DEFAULT_PROFILE_TRIALS = 24
DEFAULT_HYBRID_TRIALS = 48
ALLOWED_TIMEFRAMES_30M_TO_1D = ("30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")
DEFAULT_SLIPPAGE_BPS = 10.0
DEFAULT_BRIDGE_PROTOCOL_MANIFEST = (
    REPO_ROOT / ".omx" / "plans" / "bridge-protocol-manifest-oos-oracle-hybrid-v1-20260602.json"
)
CURRENT_CHALLENGER_OOS_COMP = 0.5338
CURRENT_CHALLENGER_MAX_OOS_MDD = 0.1880
ROBUST_DEFAULT_OOS_COMP = 0.2701
ROBUST_DEFAULT_MAX_OOS_MDD_LIMIT = 0.15
PERIOD_METRICS_CACHE_SIZE = max(
    0, int(os.getenv("LQ_MONTHLY_REFIT_PERIOD_METRICS_CACHE_SIZE", "200000"))
)
PREPARED_RETURNS_CACHE_SIZE = max(
    0, int(os.getenv("LQ_MONTHLY_REFIT_PREPARED_RETURNS_CACHE_SIZE", "50000"))
)
REQUIRED_BRIDGE_MANIFEST_KEYS = (
    "deployable_expert_roster",
    "allowed_pre_oos_features",
    "fixed_grids",
    "objective_utility",
    "fallback_rules",
    "negative_controls",
    "hard_no_leakage_rules",
    "promotion_thresholds",
)

INDIVIDUAL_ROBUST_PROFILE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "profile_id": "individual_robust_balanced_mdd10_gross3_core10",
        "profile_kind": "individual_sleeve_first_robust_profile",
        "max_gross_notional": 3.0,
        "max_sleeves": 18,
        "candidate_pool_size": 48,
        "min_sleeves": 10,
        "min_validation_return": 0.010,
        "min_train_return": 0.030,
        "max_validation_return": 0.18,
        "validation_spike_cap": 0.06,
        "max_validation_mdd": 0.10,
        "max_train_mdd": 0.30,
        "top_symbol_share_cap": 0.22,
        "top_asset_group_share_cap": 0.55,
    },
    {
        "profile_id": "individual_robust_growth_mdd14_gross5_core14",
        "profile_kind": "individual_sleeve_first_robust_profile",
        "max_gross_notional": 5.0,
        "max_sleeves": 28,
        "candidate_pool_size": 64,
        "min_sleeves": 14,
        "min_validation_return": 0.015,
        "min_train_return": 0.040,
        "max_validation_return": 0.24,
        "validation_spike_cap": 0.08,
        "max_validation_mdd": 0.14,
        "max_train_mdd": 0.40,
        "top_symbol_share_cap": 0.20,
        "top_asset_group_share_cap": 0.58,
    },
    {
        "profile_id": "individual_robust_opportunity_mdd18_gross7_core18",
        "profile_kind": "individual_sleeve_first_robust_profile",
        "max_gross_notional": 7.0,
        "max_sleeves": 40,
        "candidate_pool_size": 80,
        "min_sleeves": 18,
        "min_validation_return": 0.020,
        "min_train_return": 0.050,
        "max_validation_return": 0.30,
        "validation_spike_cap": 0.10,
        "max_validation_mdd": 0.18,
        "max_train_mdd": 0.50,
        "top_symbol_share_cap": 0.18,
        "top_asset_group_share_cap": 0.62,
    },
)

ASSET_TIMEFRAME_LEVERAGE_PROFILE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "profile_id": "asset_tf_leverage_balanced_mdd12_gross4_core16",
        "profile_kind": "asset_timeframe_leverage_scaled_profile",
        "max_gross_notional": 4.0,
        "max_sleeves": 32,
        "candidate_pool_size": 96,
        "min_sleeves": 16,
        "min_validation_return": 0.012,
        "min_train_return": 0.035,
        "max_validation_return": 0.22,
        "validation_spike_cap": 0.08,
        "max_validation_mdd": 0.12,
        "max_train_mdd": 0.35,
        "top_symbol_share_cap": 0.16,
        "top_asset_group_share_cap": 0.50,
        "rebalance_policy": "monthly_refit_signal_level_position_updates",
        "leverage_tuning_policy": (
            "train_validation_only_source_integer_leverage_plus_post_allocation_multiplier"
        ),
    },
    {
        "profile_id": "asset_tf_leverage_growth_mdd16_gross6_core22",
        "profile_kind": "asset_timeframe_leverage_scaled_profile",
        "max_gross_notional": 6.0,
        "max_sleeves": 44,
        "candidate_pool_size": 120,
        "min_sleeves": 22,
        "min_validation_return": 0.016,
        "min_train_return": 0.045,
        "max_validation_return": 0.28,
        "validation_spike_cap": 0.10,
        "max_validation_mdd": 0.16,
        "max_train_mdd": 0.45,
        "top_symbol_share_cap": 0.14,
        "top_asset_group_share_cap": 0.56,
        "rebalance_policy": "monthly_refit_signal_level_position_updates",
        "leverage_tuning_policy": (
            "train_validation_only_source_integer_leverage_plus_post_allocation_multiplier"
        ),
    },
)

INDIVIDUAL_PORTFOLIO_GUARD: dict[str, float] = {
    "max_validation_return": 0.35,
    "validation_spike_cap": 0.12,
    "max_train_return": 2.50,
    "max_validation_mdd": 0.18,
    "max_train_mdd": 0.48,
    "max_gross_notional": 7.0,
}


@dataclass(frozen=True)
class MonthlyFold:
    fold_id: str
    refit_at: pd.Timestamp
    train: tuple[pd.Timestamp, pd.Timestamp]
    validation: tuple[pd.Timestamp, pd.Timestamp]
    locked_oos: tuple[pd.Timestamp, pd.Timestamp]

    def windows(self) -> broad69.SplitWindows:
        return broad69.SplitWindows(train=self.train, validation=self.validation)

    def as_payload(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "refit_at": self.refit_at.isoformat(),
            "train": {"start": self.train[0].isoformat(), "end": self.train[1].isoformat()},
            "validation": {
                "start": self.validation[0].isoformat(),
                "end": self.validation[1].isoformat(),
            },
            "locked_oos": {
                "start": self.locked_oos[0].isoformat(),
                "end": self.locked_oos[1].isoformat(),
            },
        }


@dataclass(frozen=True)
class CandidateResult:
    family: str
    candidate_label: str
    source_profile_id: str
    row: Mapping[str, Any]
    returns: pd.Series


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _coerce_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _validate_timeframes_30m_to_1d(timeframes: Sequence[str]) -> tuple[str, ...]:
    allowed = set(ALLOWED_TIMEFRAMES_30M_TO_1D)
    normalized = tuple(str(item).strip().lower() for item in timeframes if str(item).strip())
    if not normalized:
        raise ValueError("at least one timeframe is required")
    unsupported = [tf for tf in normalized if tf not in allowed]
    if unsupported:
        raise ValueError(
            "unsupported timeframe(s) for this 30m-1D run: "
            f"{unsupported}; allowed={list(ALLOWED_TIMEFRAMES_30M_TO_1D)}"
        )
    return tuple(dict.fromkeys(normalized))


def _timeframe_coverage_summary(coverage: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for timeframe, tf_cov_raw in dict(coverage.get("timeframes") or {}).items():
        tf_cov = dict(tf_cov_raw or {})
        rows = [_safe_float(item.get("rows")) for item in tf_cov.values() if isinstance(item, Mapping)]
        zero_symbols = [
            symbol
            for symbol, item in tf_cov.items()
            if isinstance(item, Mapping) and int(item.get("rows") or 0) <= 0
        ]
        latest_values = [
            str(item.get("latest"))
            for item in tf_cov.values()
            if isinstance(item, Mapping) and item.get("latest")
        ]
        earliest_values = [
            str(item.get("earliest"))
            for item in tf_cov.values()
            if isinstance(item, Mapping) and item.get("earliest")
        ]
        out[str(timeframe)] = {
            "symbols_with_rows": int(sum(value > 0 for value in rows)),
            "symbols_without_rows": len(zero_symbols),
            "zero_row_symbols": zero_symbols[:30],
            "total_rows": int(sum(rows)),
            "min_rows": int(min(rows)) if rows else 0,
            "median_rows": float(np.median(rows)) if rows else 0.0,
            "max_rows": int(max(rows)) if rows else 0,
            "earliest": min(earliest_values) if earliest_values else None,
            "latest": max(latest_values) if latest_values else None,
            "complete_bucket_policy": "drop_1m_derived_buckets_with_source_1m_rows_below_timeframe_minutes",
        }
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", "utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bridge_protocol_manifest(path: Path = DEFAULT_BRIDGE_PROTOCOL_MANIFEST) -> dict[str, Any]:
    """Load the pre-registered no-OOS bridge protocol and its content hash."""
    manifest_path = path.expanduser().resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    missing = [key for key in REQUIRED_BRIDGE_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"bridge protocol manifest missing required key(s): {missing}")
    manifest["_path"] = str(manifest_path)
    manifest["_sha256"] = hashlib.sha256(raw).hexdigest()
    return manifest


def _protocol_freeze_report(manifest_ref: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(manifest_ref.get("manifest") or manifest_ref)
    return {
        "bridge_protocol_manifest_present": True,
        "bridge_protocol_manifest_sha256": str(
            manifest_ref.get("sha256") or manifest_ref.get("_sha256") or ""
        ),
        "manifest_version": manifest.get("manifest_version"),
        "frozen_before_first_oos_evaluation": True,
        "post_oos_expansion_allowed": False,
        "oos_used_for_protocol_expansion": False,
        "deployable_expert_count": len(manifest.get("deployable_expert_roster") or []),
        "allowed_pre_oos_feature_count": len(manifest.get("allowed_pre_oos_features") or []),
    }


def _online_weight_audit(weight_logs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for entry in weight_logs:
        month = str(entry.get("month") or entry.get("fold_id") or "")
        used_months = [str(item) for item in entry.get("utility_months_used") or []]
        current_utility = str(entry.get("current_month_utility") or month)
        if current_utility and current_utility in used_months:
            violations.append(month)
        if month and any(item >= month for item in used_months):
            violations.append(month)
    return {
        "fully_lagged_online_weights": not violations,
        "violating_months": sorted(set(violations)),
        "rule": "month_m_weights_use_only_completed_months_before_m",
    }


def _dynamic_self_feed_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    forbidden_tokens = ("locked_oos", "oos", "oracle", "same_fold_selected_label")
    violations: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("family") or "") != "dynamic_conviction_switch":
            continue
        fields = [
            str(item).lower()
            for key in ("selection_inputs", "feature_inputs", "target_inputs", "weighting_inputs")
            for item in (row.get(key) or [])
        ]
        bad = [item for item in fields if any(token in item for token in forbidden_tokens)]
        if bad:
            violations.append(
                {
                    "fold_id": row.get("fold_id"),
                    "candidate_label": row.get("candidate_label"),
                    "inputs": bad,
                }
            )
    return {
        "no_same_month_dynamic_self_feeding": not violations,
        "violations": violations,
        "rule": "same_fold_dynamic_switch_label_oos_utility_or_oracle_rank_not_used",
    }


def _metric_reconciliation_report(payload: Mapping[str, Any], *, tolerance: float = 1e-12) -> dict[str, Any]:
    expected = {
        row["candidate_label"]: row
        for row in _aggregate_rows(list(payload.get("fold_candidate_rows") or []))
    }
    actual = {
        row["candidate_label"]: row
        for row in list(payload.get("aggregate_rankings") or [])
    }
    mismatches: list[dict[str, Any]] = []
    for label, expected_row in expected.items():
        actual_row = actual.get(label)
        if actual_row is None:
            mismatches.append({"candidate_label": label, "field": "missing_aggregate"})
            continue
        for field in ("compounded_oos_return", "positive_oos_folds", "max_oos_mdd", "min_oos_return"):
            if abs(_safe_float(actual_row.get(field)) - _safe_float(expected_row.get(field))) > tolerance:
                mismatches.append({"candidate_label": label, "field": field})
    return {
        "metrics_reconciled": not mismatches,
        "mismatches": mismatches,
        "candidate_count": len(expected),
    }


def _promotability_decision(best: Mapping[str, Any]) -> dict[str, Any]:
    clean_candidate = bool(best.get("clean_promotion_eligible", True))
    oos_comp = _safe_float(best.get("compounded_oos_return"))
    max_mdd = _safe_float(best.get("max_oos_mdd"))
    min_oos = _safe_float(best.get("min_oos_return"))
    challenger_comp = CURRENT_CHALLENGER_OOS_COMP
    challenger_mdd = CURRENT_CHALLENGER_MAX_OOS_MDD
    robust_default_comp = ROBUST_DEFAULT_OOS_COMP
    robust_default_mdd_limit = ROBUST_DEFAULT_MAX_OOS_MDD_LIMIT
    reasons: list[str] = []
    if not clean_candidate:
        return {
            "promotable": False,
            "promotion_hard_stop_pass": False,
            "promotion_hard_stop_reasons": ["blocked_non_clean_research_variant"],
            "if_false_recommendation": "fresh_forward_shadow_required_before_promotion",
        }
    if oos_comp > challenger_comp and max_mdd <= challenger_mdd:
        reasons.append("beats_current_clean_challenger_without_worse_mdd")
    if oos_comp >= challenger_comp * 0.98 and (max_mdd < challenger_mdd or min_oos > -0.0265):
        reasons.append("comparable_to_challenger_with_material_risk_improvement")
    if oos_comp > robust_default_comp and max_mdd <= robust_default_mdd_limit:
        reasons.append("beats_robust_default_with_mdd_limit")
    return {
        "promotable": bool(reasons),
        "promotion_hard_stop_pass": bool(reasons),
        "promotion_hard_stop_reasons": reasons,
        "if_false_recommendation": (
            None
            if reasons
            else "paper_shadow_only_further_uplift_would_be_oos_mining_risk"
        ),
    }


@dataclass(frozen=True, slots=True)
class _PreparedReturns:
    source: pd.Series
    index_ns: np.ndarray
    values: np.ndarray


_PREPARED_RETURNS_CACHE: OrderedDict[tuple[Any, ...], _PreparedReturns] = OrderedDict()
_PERIOD_METRICS_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()


def _bounded_cache_get(cache: OrderedDict[tuple[Any, ...], Any], key: tuple[Any, ...]) -> Any | None:
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _bounded_cache_set(
    cache: OrderedDict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    value: Any,
    *,
    limit: int,
) -> None:
    if limit <= 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _clear_period_metric_caches() -> None:
    """Clear hot-path metric caches used by benchmarks and isolated tests."""
    _PREPARED_RETURNS_CACHE.clear()
    _PERIOD_METRICS_CACHE.clear()


def _timestamp_ns(value: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(value).value)


def _returns_signature(returns: pd.Series) -> tuple[Any, ...]:
    size = int(returns.size)
    if size == 0:
        return (id(returns), 0, 0, 0, 0.0, 0.0, True)
    index = pd.DatetimeIndex(returns.index)
    values = returns.to_numpy(dtype=float, copy=False)
    return (
        id(returns),
        size,
        int(index[0].value),
        int(index[-1].value),
        float(values[0]),
        float(values[-1]),
        bool(index.is_monotonic_increasing),
    )


def _prepared_returns(returns: pd.Series, signature: tuple[Any, ...] | None = None) -> _PreparedReturns:
    signature = _returns_signature(returns) if signature is None else signature
    cached = _bounded_cache_get(_PREPARED_RETURNS_CACHE, signature)
    if cached is not None:
        return cached

    if returns.size == 0:
        prepared = _PreparedReturns(
            source=returns,
            index_ns=np.asarray([], dtype=np.int64),
            values=np.asarray([], dtype=np.float64),
        )
    else:
        index = pd.DatetimeIndex(returns.index)
        index_ns = index.view("int64")
        values = returns.to_numpy(dtype=float, copy=False)
        if not index.is_monotonic_increasing:
            order = np.argsort(index_ns, kind="mergesort")
            index_ns = index_ns[order]
            values = values[order]
        prepared = _PreparedReturns(
            source=returns,
            index_ns=np.asarray(index_ns, dtype=np.int64),
            values=np.asarray(values, dtype=np.float64),
        )
    _bounded_cache_set(
        _PREPARED_RETURNS_CACHE,
        signature,
        prepared,
        limit=PREPARED_RETURNS_CACHE_SIZE,
    )
    return prepared


def _periods_per_year_ns(index_ns: np.ndarray) -> float:
    if index_ns.size < 2:
        return 365.0 * 24.0 * 2.0
    diffs = np.diff(index_ns) / 1_000_000_000.0
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 365.0 * 24.0 * 2.0
    seconds = float(np.median(positive))
    return 365.0 * 24.0 * 60.0 * 60.0 / seconds


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 365.0 * 24.0 * 2.0
    diffs = np.diff(index.view("int64")) / 1_000_000_000.0
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 365.0 * 24.0 * 2.0
    seconds = float(np.median(positive))
    return 365.0 * 24.0 * 60.0 * 60.0 / seconds


def _period_metrics(returns: pd.Series, window: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, Any]:
    start_ns = _timestamp_ns(window[0])
    end_ns = _timestamp_ns(window[1])
    signature = _returns_signature(returns)
    cache_key = (*signature, start_ns, end_ns)
    cached = _bounded_cache_get(_PERIOD_METRICS_CACHE, cache_key)
    if cached is not None:
        return dict(cached)

    prepared = _prepared_returns(returns, signature)
    lo = int(np.searchsorted(prepared.index_ns, start_ns, side="left"))
    hi = int(np.searchsorted(prepared.index_ns, end_ns, side="right"))
    values = prepared.values[lo:hi]
    period_index_ns = prepared.index_ns[lo:hi]
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    downside = values[values < 0.0]
    down_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    annual = _periods_per_year_ns(period_index_ns)
    total = float(np.prod(1.0 + values) - 1.0) if values.size else 0.0
    mdd = float(broad69.max_drawdown(values)) if values.size else 0.0
    out = {
        "start": window[0].isoformat(),
        "end": window[1].isoformat(),
        "bar_count": int(values.size),
        "total_return": total,
        "mdd": mdd,
        "sharpe": mean / std * math.sqrt(annual) if std > 0.0 else 0.0,
        "sortino": mean / down_std * math.sqrt(annual) if down_std > 0.0 else 0.0,
        "calmar": total / mdd if mdd > 0.0 else 0.0,
    }
    _bounded_cache_set(_PERIOD_METRICS_CACHE, cache_key, out, limit=PERIOD_METRICS_CACHE_SIZE)
    return dict(out)


def _add_month(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return _coerce_ts(ts + pd.DateOffset(months=months)).normalize()


def build_monthly_folds(
    *,
    train_start: pd.Timestamp,
    first_oos_start: pd.Timestamp,
    latest_data: pd.Timestamp,
    bar_minutes: int,
) -> list[MonthlyFold]:
    folds: list[MonthlyFold] = []
    oos_start = first_oos_start.normalize()
    step = pd.Timedelta(minutes=int(bar_minutes))
    while oos_start <= latest_data:
        validation_start = _add_month(oos_start, -2)
        validation_end = oos_start - step
        train_end = validation_start - step
        oos_end = min(_add_month(oos_start, 1) - step, latest_data)
        if train_end >= train_start and validation_end >= validation_start and oos_end >= oos_start:
            folds.append(
                MonthlyFold(
                    fold_id=f"{oos_start:%Y-%m}",
                    refit_at=oos_start,
                    train=(train_start, train_end),
                    validation=(validation_start, validation_end),
                    locked_oos=(oos_start, oos_end),
                )
            )
        oos_start = _add_month(oos_start, 1)
    return folds


def _combine_profile_returns(
    profile_streams: Sequence[grid_hybrid.ProfileStream], weights: Mapping[str, Any]
) -> pd.Series:
    active = {
        str(stream.profile_id): _safe_float(weights.get(stream.profile_id))
        for stream in profile_streams
        if _safe_float(weights.get(stream.profile_id)) != 0.0
    }
    if not active:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex(
        sorted(
            set().union(
                *(set(stream.returns.index) for stream in profile_streams if stream.profile_id in active)
            )
        )
    )
    combined = pd.Series(0.0, index=index, dtype=float)
    by_id = {stream.profile_id: stream for stream in profile_streams}
    for profile_id, weight in active.items():
        combined = combined.add(by_id[profile_id].returns.reindex(index, fill_value=0.0) * weight)
    return combined.sort_index()


def _stress_turnover_for_row(
    profile_streams: Sequence[grid_hybrid.ProfileStream], row: Mapping[str, Any]
) -> dict[str, float]:
    weights = dict(row.get("weights") or {})
    return {
        split: float(
            sum(
                stream.turnover_by_split[split] * _safe_float(weights.get(stream.profile_id))
                for stream in profile_streams
            )
        )
        for split in broad69.SPLIT_ORDER
    }


def _candidate_eval(
    *,
    family: str,
    label: str,
    row: Mapping[str, Any],
    returns: pd.Series,
) -> CandidateResult:
    return CandidateResult(
        family=family,
        candidate_label=label,
        source_profile_id=str(row.get("profile_id") or label),
        row=dict(row),
        returns=returns.sort_index(),
    )


def _candidate_validation_snapshot(candidate: CandidateResult, fold: MonthlyFold) -> dict[str, float]:
    train = _period_metrics(candidate.returns, fold.train)
    validation = _period_metrics(candidate.returns, fold.validation)
    val_return = _safe_float(validation["total_return"])
    val_mdd = _safe_float(validation["mdd"])
    train_return = _safe_float(train["total_return"])
    train_mdd = _safe_float(train["mdd"])
    return {
        "train_return": train_return,
        "validation_return": val_return,
        "train_mdd": train_mdd,
        "validation_mdd": val_mdd,
        "validation_calmar": val_return / max(val_mdd, 0.01),
        "stability_score": min(train_return, val_return)
        - 1.5 * max(0.0, val_return - max(train_return, 0.0))
        - 0.75 * val_mdd
        - 0.25 * train_mdd,
    }


def _blend_candidate_returns(
    candidates: Sequence[CandidateResult], weights: Mapping[str, float]
) -> pd.Series:
    active = [
        (candidate, float(weights.get(candidate.candidate_label, 0.0)))
        for candidate in candidates
        if float(weights.get(candidate.candidate_label, 0.0)) > 0.0
    ]
    if not active:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex(
        sorted(set().union(*(set(candidate.returns.index) for candidate, _ in active)))
    )
    blended = pd.Series(0.0, index=index, dtype=float)
    for candidate, weight in active:
        blended = blended.add(candidate.returns.reindex(index, fill_value=0.0) * weight)
    return blended.sort_index()


_NON_LEAF_PORTFOLIO_FAMILIES = {
    "cross_candidate_hybrid",
    "dynamic_aware_hybrid",
    "hybrid_oracle_bridge",
    "meta_portfolio",
    "validation_selector",
    "risk_enhanced_blend",
    "fixed_relaxed_dynamic_blend",
    "mdd30_barbell_blend",
    "mdd30_risk_scaled",
    "mdd30_high_vol_gate",
    "dynamic_conviction_switch",
}

_NON_LEAF_LABEL_TOKENS = (
    ":hybrid_",
    ":selected_optuna",
    ":selected_train_validation_legal",
    ":static_guarded",
    "hybrid_",
    "blend:",
    "meta_portfolio:",
    "validation_selector:",
    "dynamic_conviction_switch:",
)

_NON_LEAF_PROFILE_KIND_TOKENS = (
    "hybrid",
    "blend",
    "portfolio",
    "selector",
    "switch",
    "gate",
    "scaled",
    "assimilation",
)


def _leaf_strategy_material_candidate(candidate: CandidateResult) -> bool:
    """Return whether a candidate may be used as a hybrid/portfolio ingredient.

    Hybrid rows are already portfolios over lower-level strategies. Feeding a
    hybrid/blend/selector/gate row into another hybrid makes exposure look more
    diversified than it is and can double-count the same sleeve. Downstream
    portfolio builders therefore accept only leaf-like strategy/profile rows.
    """
    label = candidate.candidate_label.lower()
    family = candidate.family.lower()
    row = dict(candidate.row)
    profile_kind = str(row.get("profile_kind") or "").lower()
    profile_id = str(row.get("profile_id") or row.get("source_profile_id") or "").lower()
    if family in _NON_LEAF_PORTFOLIO_FAMILIES:
        return False
    if any(token in label for token in _NON_LEAF_LABEL_TOKENS):
        return False
    if any(token in profile_id for token in _NON_LEAF_LABEL_TOKENS):
        return False
    return not any(token in profile_kind for token in _NON_LEAF_PROFILE_KIND_TOKENS)


def _normalize_capped_weights(raw: Mapping[str, float], *, cap: float) -> dict[str, float]:
    weights = {key: max(0.0, float(value)) for key, value in raw.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    weights = {key: value / total for key, value in weights.items()}
    for _ in range(8):
        overflow = sum(max(0.0, value - cap) for value in weights.values())
        if overflow <= 1e-12:
            break
        capped = {key for key, value in weights.items() if value >= cap}
        free = [key for key in weights if key not in capped]
        for key in capped:
            weights[key] = min(cap, weights[key])
        free_total = sum(weights[key] for key in free)
        if not free or free_total <= 0.0:
            break
        for key in free:
            weights[key] += overflow * weights[key] / free_total
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items() if total > 0.0 and value > 0.0}


def _meta_portfolio_candidates(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
) -> list[CandidateResult]:
    snapshots = [
        (candidate, _candidate_validation_snapshot(candidate, fold))
        for candidate in candidates
        if candidate.returns.size
    ]
    eligible = [
        (candidate, snap)
        for candidate, snap in snapshots
        if snap["train_return"] > 0.0
        and snap["validation_return"] > 0.0
        and snap["validation_mdd"] <= 0.20
        and not dict(candidate.row).get("selection_reasons")
        and _leaf_strategy_material_candidate(candidate)
    ]
    if len(eligible) < 2:
        return []

    out: list[CandidateResult] = []

    calmar_ranked = sorted(
        eligible,
        key=lambda item: (
            item[1]["validation_calmar"],
            item[1]["validation_return"],
            -item[1]["validation_mdd"],
        ),
        reverse=True,
    )[:5]
    calmar_weights = _normalize_capped_weights(
        {
            candidate.candidate_label: max(0.0, snap["validation_calmar"])
            for candidate, snap in calmar_ranked
        },
        cap=0.35,
    )
    if calmar_weights:
        out.append(
            _candidate_eval(
                family="meta_portfolio",
                label="meta_portfolio:validation_calmar_top5_capped",
                row={
                    "profile_id": "validation_calmar_top5_capped",
                    "profile_kind": "validation_only_cross_candidate_portfolio",
                    "candidate_tier": "clean_train_validation_selected_paper_shadow",
                    "weights": calmar_weights,
                    "final_weights": calmar_weights,
                    "selection_inputs": ["train", "validation"],
                    "uses_locked_oos_for_selection": False,
                    "ready_for_paper": True,
                    "ready_for_real": False,
                    "real_money_execution": False,
                },
                returns=_blend_candidate_returns([item[0] for item in calmar_ranked], calmar_weights),
            )
        )

    stability_ranked = sorted(
        eligible,
        key=lambda item: (
            item[1]["stability_score"],
            item[1]["validation_return"],
            -item[1]["validation_mdd"],
        ),
        reverse=True,
    )[:8]
    stability_weights = {
        candidate.candidate_label: 1.0 / len(stability_ranked)
        for candidate, _ in stability_ranked
    }
    if stability_weights:
        out.append(
            _candidate_eval(
                family="meta_portfolio",
                label="meta_portfolio:validation_stability_top8_equal",
                row={
                    "profile_id": "validation_stability_top8_equal",
                    "profile_kind": "validation_only_cross_candidate_portfolio",
                    "candidate_tier": "clean_train_validation_selected_paper_shadow",
                    "weights": stability_weights,
                    "final_weights": stability_weights,
                    "selection_inputs": ["train", "validation"],
                    "uses_locked_oos_for_selection": False,
                    "ready_for_paper": True,
                    "ready_for_real": False,
                    "real_money_execution": False,
                },
                returns=_blend_candidate_returns(
                    [item[0] for item in stability_ranked], stability_weights
                ),
            )
        )

    inv_mdd_ranked = sorted(
        eligible,
        key=lambda item: (
            item[1]["validation_return"] > 0.02,
            item[1]["validation_return"] / max(item[1]["validation_mdd"], 0.02),
            item[1]["stability_score"],
        ),
        reverse=True,
    )[:10]
    inv_mdd_weights = _normalize_capped_weights(
        {
            candidate.candidate_label: 1.0 / max(snap["validation_mdd"], 0.02)
            for candidate, snap in inv_mdd_ranked
        },
        cap=0.25,
    )
    if inv_mdd_weights:
        out.append(
            _candidate_eval(
                family="meta_portfolio",
                label="meta_portfolio:validation_inverse_mdd_top10_capped",
                row={
                    "profile_id": "validation_inverse_mdd_top10_capped",
                    "profile_kind": "validation_only_cross_candidate_portfolio",
                    "candidate_tier": "clean_train_validation_selected_paper_shadow",
                    "weights": inv_mdd_weights,
                    "final_weights": inv_mdd_weights,
                    "selection_inputs": ["train", "validation"],
                    "uses_locked_oos_for_selection": False,
                    "ready_for_paper": True,
                    "ready_for_real": False,
                    "real_money_execution": False,
                },
                returns=_blend_candidate_returns([item[0] for item in inv_mdd_ranked], inv_mdd_weights),
            )
        )
    return out


def _stream_from_candidate(
    candidate: CandidateResult,
    *,
    union_index: pd.DatetimeIndex,
) -> grid_hybrid.ProfileStream:
    row = dict(candidate.row)
    gross = max(0.01, _safe_float(row.get("gross_notional_fraction"), 1.0))
    turnover: dict[str, float] = {}
    events: dict[str, int] = {}
    liquidations: dict[str, int] = {}
    for split in grid_hybrid.ilp.SPLIT_ORDER:
        event_key = (
            f"{split}_trade_event_count"
            if split != "locked_oos"
            else "locked_oos_trade_event_count_report_only"
        )
        liq_key = (
            f"{split}_liquidation_count"
            if split != "locked_oos"
            else "locked_oos_liquidation_count_report_only"
        )
        event_count = int(row.get(event_key) or 0)
        events[split] = event_count
        turnover[split] = float(max(1.0, event_count) * gross)
        liquidations[split] = int(row.get(liq_key) or 0)
    asset_gross = dict(row.get("asset_gross_notional_fraction") or {})
    return grid_hybrid.ProfileStream(
        profile_id=candidate.candidate_label,
        candidate_tier=str(row.get("candidate_tier") or "cross_candidate_hybrid_input"),
        leverage_map={},
        gross_notional_fraction=float(gross),
        asset_gross_notional_fraction=asset_gross,
        selected_model_ids=(candidate.candidate_label,),
        returns=candidate.returns.reindex(union_index, fill_value=0.0).sort_index(),
        turnover_by_split=turnover,
        trade_events_by_split=events,
        liquidation_count_by_split=liquidations,
    )


def _cross_candidate_hybrid_candidates(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
    *,
    hybrid_trials: int,
    seed: int,
) -> list[CandidateResult]:
    snapshots = [
        (candidate, _candidate_validation_snapshot(candidate, fold))
        for candidate in candidates
        if candidate.returns.size
    ]
    eligible = [
        (candidate, snap)
        for candidate, snap in snapshots
        if snap["train_return"] > 0.0
        and snap["validation_return"] > 0.0
        and snap["validation_mdd"] <= 0.18
        and _clean_downstream_candidate(candidate)
        and _leaf_strategy_material_candidate(candidate)
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            item[1]["stability_score"],
            item[1]["validation_calmar"],
            item[1]["validation_return"],
        ),
        reverse=True,
    )[:6]
    if len(ranked) < 2:
        return []
    union_index = pd.DatetimeIndex(sorted(set().union(*(set(item[0].returns.index) for item in ranked))))
    streams = [_stream_from_candidate(candidate, union_index=union_index) for candidate, _ in ranked]
    split_windows = profile69._split_windows_for_hybrid(fold.windows().as_payload())
    with optuna_hybrid._split_window_context(split_windows):
        v35 = optuna_hybrid._run_optuna(
            streams,
            version="v3_5",
            n_trials=max(4, int(hybrid_trials) // 2),
            seed=int(seed) + 810_000,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        v36 = optuna_hybrid._run_optuna(
            streams,
            version="v3_6",
            n_trials=max(4, int(hybrid_trials) // 2),
            seed=int(seed) + 810_001,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        v35_train_validation = optuna_hybrid._run_optuna(
            streams,
            version="v3_5",
            n_trials=max(4, int(hybrid_trials) // 2),
            seed=int(seed) + 810_010,
            fit_splits=("train", "validation"),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        v36_train_validation = optuna_hybrid._run_optuna(
            streams,
            version="v3_6",
            n_trials=max(4, int(hybrid_trials) // 2),
            seed=int(seed) + 810_011,
            fit_splits=("train", "validation"),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
    for result, fit_policy in (
        (v35, "train_only_after_validation_input_screen"),
        (v36, "train_only_after_validation_input_screen"),
        (v35_train_validation, "train_validation_final_refit"),
        (v36_train_validation, "train_validation_final_refit"),
    ):
        result.row.update(
            {
                "profile_kind": "validation_only_cross_candidate_optuna_hybrid",
                "candidate_tier": "clean_train_validation_selected_paper_shadow",
                "selection_inputs": ["train", "validation"],
                "uses_locked_oos_for_selection": False,
                "cross_candidate_fit_policy": fit_policy,
                "cross_candidate_inputs": [candidate.candidate_label for candidate, _ in ranked],
                "ready_for_real": False,
                "real_money_execution": False,
            }
        )
    return [
        _candidate_eval(
            family="cross_candidate_hybrid",
            label="cross_candidate_hybrid:hybrid_v3_5",
            row=v35.row,
            returns=v35.returns,
        ),
        _candidate_eval(
            family="cross_candidate_hybrid",
            label="cross_candidate_hybrid:hybrid_v3_6",
            row=v36.row,
            returns=v36.returns,
        ),
        _candidate_eval(
            family="cross_candidate_hybrid",
            label="cross_candidate_hybrid:hybrid_v3_5_train_validation_fit",
            row=v35_train_validation.row,
            returns=v35_train_validation.returns,
        ),
        _candidate_eval(
            family="cross_candidate_hybrid",
            label="cross_candidate_hybrid:hybrid_v3_6_train_validation_fit",
            row=v36_train_validation.row,
            returns=v36_train_validation.returns,
        ),
    ]


def _dynamic_conviction_switch_candidates(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
) -> list[CandidateResult]:
    """Fold-local train/validation selector over existing clean candidates.

    The switch is deliberately simple and auditable: every fold scores only
    train+validation evidence, then chooses either an aggressive leaf sleeve
    or a strict-efficiency fallback.  Locked OOS is never read by the selector.
    Multiple fixed thresholds are emitted as separate candidates so the report
    can show whether the higher-comp path is coming from a stable rule or from
    an overly specific setting.
    """
    by_label = {candidate.candidate_label: candidate for candidate in candidates}
    aggressive_labels = [
        "profile_optuna:balanced_mdd12_gross5_69_asset_profile_optuna",
        "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna",
        "profile_optuna:aggressive_mdd30_gross10_69_asset_profile_optuna",
        "individual_robust:individual_robust_balanced_mdd10_gross3_core10",
        "individual_robust:individual_robust_growth_mdd14_gross5_core14",
        "individual_robust:individual_robust_opportunity_mdd18_gross7_core18",
        "relaxed_efficiency:balanced_mdd12_gross5_69_asset_relaxed_efficiency_repair_optuna",
        "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna",
        "relaxed_efficiency:aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna",
        "strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna",
    ]
    fallback_labels = [
        "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna",
        "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna",
    ]

    def aggressive_eligible(label: str) -> CandidateResult | None:
        candidate = by_label.get(label)
        if candidate is None:
            return None
        if not _clean_downstream_candidate(candidate):
            return None
        if not _leaf_strategy_material_candidate(candidate):
            return None
        snap = _candidate_validation_snapshot(candidate, fold)
        if snap["train_return"] <= 0.0 or snap["validation_return"] <= 0.0:
            return None
        if snap["validation_mdd"] > 0.18:
            return None
        return candidate

    def fallback_eligible(label: str) -> CandidateResult | None:
        candidate = by_label.get(label)
        if candidate is None:
            return None
        if not _clean_downstream_candidate(candidate):
            return None
        if not _leaf_strategy_material_candidate(candidate):
            return None
        snap = _candidate_validation_snapshot(candidate, fold)
        # Fallback sleeves are allowed to be tiny/cash-like and can carry
        # report-only selection warnings; the switch needs a deterministic
        # low-risk branch every month rather than silently skipping a fold.
        if snap["train_return"] < -0.02 or snap["validation_return"] < -0.02:
            return None
        if snap["validation_mdd"] > 0.20:
            return None
        return candidate

    def conviction_score(candidate: CandidateResult) -> float:
        snap = _candidate_validation_snapshot(candidate, fold)
        return float(
            min(snap["validation_return"], 1.0)
            + 0.25 * min(snap["train_return"], 3.0)
            - 1.5 * snap["validation_mdd"]
            - 0.25 * snap["train_mdd"]
        )

    def fallback_score(candidate: CandidateResult) -> float:
        snap = _candidate_validation_snapshot(candidate, fold)
        return float(snap["validation_return"] / max(snap["validation_mdd"], 0.02))

    aggressive_pool = [
        candidate for label in aggressive_labels if (candidate := aggressive_eligible(label))
    ]
    fallback_pool = [
        candidate for label in fallback_labels if (candidate := fallback_eligible(label))
    ]
    if not aggressive_pool or not fallback_pool:
        return []

    best_aggressive = max(aggressive_pool, key=conviction_score)
    best_fallback = max(fallback_pool, key=fallback_score)
    balanced_fallback = by_label.get(
        "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
    )
    growth_fallback = by_label.get(
        "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna"
    )
    risk_capped_fallback = best_fallback
    if balanced_fallback is not None and growth_fallback is not None:
        balanced_snap = _candidate_validation_snapshot(balanced_fallback, fold)
        # If the higher-return strict sleeve is already drawing down >10% in
        # validation, prefer the tiny growth fallback.  This keeps the switch
        # from mistaking a stressed rebound for a safe defensive branch.
        risk_capped_fallback = (
            growth_fallback if balanced_snap["validation_mdd"] > 0.10 else balanced_fallback
        )
    best_score = conviction_score(best_aggressive)
    out: list[CandidateResult] = []
    for threshold in (0.85, 0.90, 0.95, 1.00):
        for fallback_name, fallback_candidate in (
            ("strict_fallback", best_fallback),
            ("risk_capped_fallback", risk_capped_fallback),
        ):
            selected = best_aggressive if best_score >= threshold else fallback_candidate
            selected_snap = _candidate_validation_snapshot(selected, fold)
            row = {
                "profile_id": f"dynamic_conviction_switch_t{threshold:.2f}_{fallback_name}",
                "profile_kind": "train_validation_dynamic_conviction_switch",
                "candidate_tier": "clean_train_validation_selected_paper_shadow",
                "selected_candidate_label": selected.candidate_label,
                "aggressive_candidate_label": best_aggressive.candidate_label,
                "fallback_candidate_label": fallback_candidate.candidate_label,
                "fallback_policy": fallback_name,
                "conviction_threshold": float(threshold),
                "aggressive_conviction_score": best_score,
                "selected_validation_return": selected_snap["validation_return"],
                "selected_validation_mdd": selected_snap["validation_mdd"],
                "selection_inputs": ["train", "validation"],
                "uses_locked_oos_for_selection": False,
                "ready_for_paper": True,
                "ready_for_real": False,
                "real_money_execution": False,
                "weights": {selected.candidate_label: 1.0},
                "final_weights": {selected.candidate_label: 1.0},
            }
            out.append(
                _candidate_eval(
                    family="dynamic_conviction_switch",
                    label=f"dynamic_conviction_switch:t{threshold:.2f}_{fallback_name}",
                    row=row,
                    returns=selected.returns,
                )
            )
    return out


DYNAMIC_AWARE_PRIORITY_LABELS: tuple[str, ...] = (
    "dynamic_conviction_switch:t0.85_risk_capped_fallback",
    "dynamic_conviction_switch:t0.90_risk_capped_fallback",
    "dynamic_conviction_switch:t0.95_risk_capped_fallback",
    "cross_candidate_hybrid:hybrid_v3_6_train_validation_fit",
    "cross_candidate_hybrid:hybrid_v3_5",
    "cross_candidate_hybrid:hybrid_v3_6",
    "cross_candidate_hybrid:hybrid_v3_5_train_validation_fit",
    "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna",
    "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna",
    "individual_robust:selected_train_validation_legal",
    "individual_robust:selected_optuna",
    "individual_robust:hybrid_v3_5",
    "individual_robust:hybrid_v3_6",
    "profile_optuna:selected_train_validation_legal",
    "profile_optuna:selected_optuna",
)


def _dynamic_aware_hybrid_candidates(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
    *,
    hybrid_trials: int,
    seed: int,
) -> list[CandidateResult]:
    """Deprecated nested-hybrid pass.

    The old dynamic-aware pass fed ``dynamic_conviction_switch`` and existing
    hybrid rows into another v3.5/v3.6 optimizer.  Under the no-nested-hybrid
    policy a hybrid/selector/gate is already a portfolio-level object, so it
    cannot be a material input to another hybrid.  Keep the function as an
    explicit no-op so historical callers do not silently resurrect the nested
    family.
    """
    _ = (candidates, fold, hybrid_trials, seed)
    return []


def _fixed_risk_enhanced_blend_candidates(
    candidates: Sequence[CandidateResult],
) -> list[CandidateResult]:
    """Deprecated fixed overlay over non-leaf dynamic/hybrid candidates.

    These research overlays were useful for shadow analysis, but they combine
    selector/hybrid rows rather than leaf strategy material.  The no-nested
    policy disables this family until it is rebuilt directly from leaf sleeves.
    """
    _ = candidates
    return []


def _fixed_relaxed_dynamic_blend_candidates(
    candidates: Sequence[CandidateResult],
) -> list[CandidateResult]:
    """Deprecated exact blend of two lower-level hybrid portfolios.

    ``relaxed_efficiency:hybrid_v3_5`` and
    ``dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit`` are already
    portfolio-level hybrids.  Blending them again creates hidden duplicate
    sleeve exposure, so this family is intentionally disabled by policy.
    """
    _ = candidates
    return []


def _validation_selector_candidates(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
) -> list[CandidateResult]:
    """Generic validation-only selectors over already-built clean candidates."""
    snapshots = [
        (candidate, _candidate_validation_snapshot(candidate, fold))
        for candidate in candidates
        if candidate.returns.size
    ]

    def eligible(candidate: CandidateResult, snap: Mapping[str, float], *, mdd_cap: float) -> bool:
        if not _clean_downstream_candidate(candidate):
            return False
        if not _leaf_strategy_material_candidate(candidate):
            return False
        return (
            snap["train_return"] > 0.0
            and snap["validation_return"] > 0.0
            and snap["validation_mdd"] <= mdd_cap
        )

    selector_specs: tuple[
        tuple[str, float, Callable[[CandidateResult, Mapping[str, float]], tuple[float, ...]]],
        ...,
    ] = (
        (
            "validation_selector:validation_sharpe_mdd10",
            0.10,
            lambda candidate, snap: (
                _period_metrics(candidate.returns, fold.validation)["sharpe"],
                snap["validation_return"],
                -snap["validation_mdd"],
            ),
        ),
        (
            "validation_selector:validation_calmar_mdd12",
            0.12,
            lambda _candidate, snap: (
                snap["validation_calmar"],
                snap["validation_return"],
                -snap["validation_mdd"],
            ),
        ),
        (
            "validation_selector:validation_utility_mdd15",
            0.15,
            lambda _candidate, snap: (
                min(snap["validation_return"], 1.0)
                + 0.25 * min(snap["train_return"], 3.0)
                - 2.0 * snap["validation_mdd"]
                - 0.25 * snap["train_mdd"],
                snap["validation_return"],
            ),
        ),
    )

    out: list[CandidateResult] = []
    for label, mdd_cap, score_fn in selector_specs:
        pool = [
            (candidate, snap)
            for candidate, snap in snapshots
            if eligible(candidate, snap, mdd_cap=mdd_cap)
        ]
        if not pool:
            continue
        selected, snap = max(pool, key=lambda item: score_fn(item[0], item[1]))
        row = {
            "profile_id": label,
            "profile_kind": "validation_only_cross_candidate_selector",
            "candidate_tier": "post_oos_research_forward_shadow_only",
            "selected_candidate_label": selected.candidate_label,
            "selected_validation_return": snap["validation_return"],
            "selected_validation_mdd": snap["validation_mdd"],
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "same_month_self_feeding": False,
            "current_fold_oos_used_for_weighting": False,
            "post_oos_research_variant": True,
            "requires_fresh_forward_shadow": True,
            "ready_for_paper": True,
            "ready_for_real": False,
            "real_money_execution": False,
        }
        out.append(
            _candidate_eval(
                family="validation_selector",
                label=label,
                row=row,
                returns=selected.returns,
            )
        )
    return out


def _scaled_candidate_returns(candidate: CandidateResult, scale: float) -> pd.Series:
    return candidate.returns.astype(float) * float(scale)


def _scale_candidate_row(
    *,
    label: str,
    source: CandidateResult,
    scale: float,
    mode: str,
    selection_inputs: Sequence[str],
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    source_row = dict(source.row)
    return {
        "profile_id": label,
        "profile_kind": "mdd30_high_volatility_sleeve_or_gate",
        "candidate_tier": "post_oos_research_forward_shadow_only",
        "source_candidate_label": source.candidate_label,
        "risk_scale": float(scale),
        "risk_scale_mode": mode,
        "weights": dict(weights or {source.candidate_label: float(scale)}),
        "final_weights": dict(weights or {source.candidate_label: float(scale)}),
        "selection_inputs": list(selection_inputs),
        "uses_locked_oos_for_selection": False,
        "same_month_self_feeding": False,
        "current_fold_oos_used_for_weighting": False,
        "post_oos_research_variant": True,
        "requires_fresh_forward_shadow": True,
        "ready_for_paper": True,
        "ready_for_real": False,
        "real_money_execution": False,
        "source_post_oos_research_variant": bool(source_row.get("post_oos_research_variant", False)),
    }


def _clean_source_candidate(candidate: CandidateResult) -> bool:
    row = dict(candidate.row)
    return bool(
        candidate.returns.size
        and _leaf_strategy_material_candidate(candidate)
        and not row.get("uses_locked_oos_for_selection")
        and not row.get("same_month_self_feeding")
        and not row.get("current_fold_oos_used_for_weighting")
        and not row.get("post_oos_research_variant")
        and not row.get("requires_fresh_forward_shadow")
        and not row.get("selection_reasons")
    )


def _clean_downstream_candidate(candidate: CandidateResult) -> bool:
    """Return whether a candidate may feed another clean selector/optimizer.

    A row can be no-current-OOS in isolation while still being a post-OOS
    research variant introduced after historical OOS review.  Such rows are
    valid for fresh-forward shadowing, but letting them enter validation
    selectors or bridge optimizers would contaminate those downstream rows and
    make the resulting "clean" labels misleading.
    """
    row = dict(candidate.row)
    return bool(
        candidate.returns.size
        and _leaf_strategy_material_candidate(candidate)
        and not row.get("uses_locked_oos_for_selection")
        and not row.get("same_month_self_feeding")
        and not row.get("current_fold_oos_used_for_weighting")
        and not row.get("post_oos_research_variant")
        and not row.get("requires_fresh_forward_shadow")
        and not row.get("source_post_oos_research_variant")
        and not row.get("selection_reasons")
    )


def _candidate_label_is_non_leaf_reference(label: str) -> bool:
    lower = str(label).lower()
    family = lower.split(":", 1)[0]
    return family in _NON_LEAF_PORTFOLIO_FAMILIES or any(
        token in lower for token in _NON_LEAF_LABEL_TOKENS
    )


def _row_reference_labels(row: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in (
        "selected_candidate_label",
        "aggressive_candidate_label",
        "fallback_candidate_label",
        "dynamic_expert_label",
        "source_candidate_label",
    ):
        value = row.get(key)
        if value:
            refs.add(str(value))
    for key in (
        "bridge_inputs",
        "dynamic_aware_inputs",
        "dynamic_input_labels",
        "robust_core_input_labels",
        "cross_candidate_inputs",
        "blend_components",
    ):
        refs.update(str(item) for item in (row.get(key) or []))
    for key in ("final_weights", "weights"):
        refs.update(str(item) for item in dict(row.get(key) or {}))
    return refs


def _row_references_non_leaf_material(row: Mapping[str, Any]) -> bool:
    return any(_candidate_label_is_non_leaf_reference(ref) for ref in _row_reference_labels(row))


def _mdd30_high_volatility_candidates(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
) -> list[CandidateResult]:
    """MDD-30 research sleeves/gates using only train+validation evidence.

    The user-supplied risk budget allows materially higher drawdown than the
    robust-default 15% cap.  This family therefore emits fixed-risk-budget
    overlays and a validation-only high-volatility gate, but never reads the
    current locked OOS month.  Because this family is introduced after prior
    OOS analysis, every row is explicitly research/shadow-only.
    """
    by_label = {candidate.candidate_label: candidate for candidate in candidates}
    out: list[CandidateResult] = []

    def source(label: str) -> CandidateResult | None:
        candidate = by_label.get(label)
        if candidate is None or not _clean_source_candidate(candidate):
            return None
        return candidate

    profile_growth = "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna"
    profile_aggressive = "profile_optuna:aggressive_mdd30_gross10_69_asset_profile_optuna"
    relaxed_growth = "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna"
    relaxed_aggressive = (
        "relaxed_efficiency:aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna"
    )
    strict_balanced = "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
    strict_growth = "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna"
    strict_aggressive = "strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna"

    fixed_scale_specs: tuple[tuple[str, str, float], ...] = (
        ("mdd30_risk_scaled:profile_growth_x1_50", profile_growth, 1.50),
        ("mdd30_risk_scaled:profile_aggressive_x1_50", profile_aggressive, 1.50),
        ("mdd30_risk_scaled:relaxed_growth_x1_50", relaxed_growth, 1.50),
        ("mdd30_risk_scaled:relaxed_aggressive_x1_50", relaxed_aggressive, 1.50),
        ("mdd30_risk_scaled:strict_aggressive_x1_25", strict_aggressive, 1.25),
    )
    for label, source_label, scale in fixed_scale_specs:
        candidate = source(source_label)
        if candidate is None:
            continue
        row = _scale_candidate_row(
            label=label,
            source=candidate,
            scale=scale,
            mode="fixed_user_mdd30_budget_scale",
            selection_inputs=["fixed_prior", "user_mdd_budget_30", "train", "validation"],
        )
        out.append(
            _candidate_eval(
                family="mdd30_risk_scaled",
                label=label,
                row=row,
                returns=_scaled_candidate_returns(candidate, scale),
            )
        )

    validation_scaled_specs: tuple[tuple[str, str, float, float], ...] = (
        ("mdd30_risk_scaled:profile_aggressive_val_mdd30_cap1_50", profile_aggressive, 0.30, 1.50),
        ("mdd30_risk_scaled:relaxed_aggressive_val_mdd30_cap1_75", relaxed_aggressive, 0.30, 1.75),
    )
    for label, source_label, target_mdd, max_scale in validation_scaled_specs:
        candidate = source(source_label)
        if candidate is None:
            continue
        snap = _candidate_validation_snapshot(candidate, fold)
        if snap["train_return"] <= -0.02 or snap["validation_return"] <= -0.02:
            continue
        scale = min(float(max_scale), float(target_mdd) / max(snap["validation_mdd"], 0.10))
        scale = max(0.50, scale)
        row = _scale_candidate_row(
            label=label,
            source=candidate,
            scale=scale,
            mode="validation_mdd_target30_with_scale_cap",
            selection_inputs=["train", "validation", "user_mdd_budget_30"],
        )
        row["source_validation_mdd"] = snap["validation_mdd"]
        row["source_validation_return"] = snap["validation_return"]
        out.append(
            _candidate_eval(
                family="mdd30_risk_scaled",
                label=label,
                row=row,
                returns=_scaled_candidate_returns(candidate, scale),
            )
        )

    fixed_blend_specs: tuple[tuple[str, dict[str, float], float], ...] = (
        (
            "mdd30_barbell_blend:profile_aggressive_70_strict_balanced_30_x1_50",
            {profile_aggressive: 0.70, strict_balanced: 0.30},
            1.50,
        ),
        (
            "mdd30_barbell_blend:relaxed_aggressive_70_strict_growth_30_x1_50",
            {relaxed_aggressive: 0.70, strict_growth: 0.30},
            1.50,
        ),
        (
            "mdd30_barbell_blend:profile_growth_60_strict_growth_40_x1_25",
            {profile_growth: 0.60, strict_growth: 0.40},
            1.25,
        ),
        (
            "mdd30_barbell_blend:strict_aggressive_70_strict_balanced_30_x1_25",
            {strict_aggressive: 0.70, strict_balanced: 0.30},
            1.25,
        ),
    )
    for label, weights, scale in fixed_blend_specs:
        active = [source(source_label) for source_label in weights]
        if any(candidate is None for candidate in active):
            continue
        active_candidates = [candidate for candidate in active if candidate is not None]
        scaled_weights = {key: value * scale for key, value in weights.items()}
        row = {
            "profile_id": label,
            "profile_kind": "mdd30_fixed_weight_barbell_blend",
            "candidate_tier": "post_oos_research_forward_shadow_only",
            "risk_scale": float(scale),
            "risk_scale_mode": "fixed_user_mdd30_budget_scale",
            "weights": scaled_weights,
            "final_weights": scaled_weights,
            "selection_inputs": ["fixed_prior", "user_mdd_budget_30", "train", "validation"],
            "uses_locked_oos_for_selection": False,
            "same_month_self_feeding": False,
            "current_fold_oos_used_for_weighting": False,
            "post_oos_research_variant": True,
            "requires_fresh_forward_shadow": True,
            "ready_for_paper": True,
            "ready_for_real": False,
            "real_money_execution": False,
        }
        out.append(
            _candidate_eval(
                family="mdd30_barbell_blend",
                label=label,
                row=row,
                returns=_blend_candidate_returns(active_candidates, weights) * float(scale),
            )
        )

    snapshots = [
        (candidate, _candidate_validation_snapshot(candidate, fold))
        for candidate in candidates
        if _clean_source_candidate(candidate)
    ]
    aggressive_labels = {
        profile_growth,
        profile_aggressive,
        relaxed_growth,
        relaxed_aggressive,
        strict_aggressive,
    }
    defensive_labels = {
        strict_balanced,
        strict_growth,
        "profile_optuna:balanced_mdd12_gross5_69_asset_profile_optuna",
        "relaxed_efficiency:balanced_mdd12_gross5_69_asset_relaxed_efficiency_repair_optuna",
    }

    def high_vol_score(item: tuple[CandidateResult, Mapping[str, float]]) -> tuple[float, ...]:
        candidate, snap = item
        period = _period_metrics(candidate.returns, fold.validation)
        return (
            snap["validation_return"]
            + 0.15 * min(snap["train_return"], 3.0)
            + 0.03 * max(0.0, _safe_float(period.get("sharpe")))
            - 0.45 * snap["validation_mdd"]
            - 0.10 * snap["train_mdd"],
            snap["validation_calmar"],
            -snap["validation_mdd"],
        )

    aggressive_pool = [
        (candidate, snap)
        for candidate, snap in snapshots
        if candidate.candidate_label in aggressive_labels
        and snap["train_return"] > 0.0
        and snap["validation_return"] > 0.0
        and snap["validation_mdd"] <= 0.30
        and snap["train_mdd"] <= 0.65
    ]
    defensive_pool = [
        (candidate, snap)
        for candidate, snap in snapshots
        if candidate.candidate_label in defensive_labels
        and snap["train_return"] > -0.02
        and snap["validation_return"] > -0.02
        and snap["validation_mdd"] <= 0.18
    ]
    if aggressive_pool and defensive_pool:
        aggressive, aggressive_snap = max(aggressive_pool, key=high_vol_score)
        defensive, defensive_snap = max(
            defensive_pool,
            key=lambda item: (
                item[1]["validation_calmar"],
                item[1]["validation_return"],
                -item[1]["validation_mdd"],
            ),
        )
        breakout = bool(
            aggressive_snap["validation_return"] >= 0.25
            and aggressive_snap["validation_calmar"] >= 2.0
            and aggressive_snap["validation_mdd"] <= 0.22
        )
        selected = aggressive if breakout else defensive
        selected_snap = aggressive_snap if breakout else defensive_snap
        scale = min(1.50, 0.30 / max(selected_snap["validation_mdd"], 0.10))
        scale = max(0.50, scale)
        label = "mdd30_high_vol_gate:validation_breakout_or_defensive_scaled"
        row = _scale_candidate_row(
            label=label,
            source=selected,
            scale=scale,
            mode="validation_breakout_gate_then_mdd30_scale",
            selection_inputs=["train", "validation", "user_mdd_budget_30"],
        )
        row.update(
            {
                "selected_candidate_label": selected.candidate_label,
                "aggressive_candidate_label": aggressive.candidate_label,
                "fallback_candidate_label": defensive.candidate_label,
                "high_vol_breakout": breakout,
                "selected_validation_return": selected_snap["validation_return"],
                "selected_validation_mdd": selected_snap["validation_mdd"],
            }
        )
        out.append(
            _candidate_eval(
                family="mdd30_high_vol_gate",
                label=label,
                row=row,
                returns=_scaled_candidate_returns(selected, scale),
            )
        )

        weights = (
            {aggressive.candidate_label: 0.75, defensive.candidate_label: 0.25}
            if breakout
            else {aggressive.candidate_label: 0.35, defensive.candidate_label: 0.65}
        )
        scale = 1.50 if breakout else 1.25
        label = "mdd30_high_vol_gate:breakout_barbell_blend"
        scaled_weights = {key: value * scale for key, value in weights.items()}
        row = {
            "profile_id": label,
            "profile_kind": "mdd30_high_volatility_validation_gate_barbell",
            "candidate_tier": "post_oos_research_forward_shadow_only",
            "risk_scale": float(scale),
            "risk_scale_mode": "validation_breakout_barbell_mdd30_scale",
            "weights": scaled_weights,
            "final_weights": scaled_weights,
            "selection_inputs": ["train", "validation", "user_mdd_budget_30"],
            "uses_locked_oos_for_selection": False,
            "same_month_self_feeding": False,
            "current_fold_oos_used_for_weighting": False,
            "post_oos_research_variant": True,
            "requires_fresh_forward_shadow": True,
            "ready_for_paper": True,
            "ready_for_real": False,
            "real_money_execution": False,
            "selected_candidate_label": aggressive.candidate_label if breakout else defensive.candidate_label,
            "aggressive_candidate_label": aggressive.candidate_label,
            "fallback_candidate_label": defensive.candidate_label,
            "high_vol_breakout": breakout,
        }
        out.append(
            _candidate_eval(
                family="mdd30_high_vol_gate",
                label=label,
                row=row,
                returns=_blend_candidate_returns([aggressive, defensive], weights) * float(scale),
            )
        )

    return out


def _bridge_utility(candidate: CandidateResult, fold: MonthlyFold, *, drawdown_penalty: float = 1.0) -> float:
    snap = _candidate_validation_snapshot(candidate, fold)
    return float(
        snap["validation_return"]
        + 0.20 * min(snap["train_return"], 3.0)
        - drawdown_penalty * snap["validation_mdd"]
        - 0.25 * snap["train_mdd"]
    )


def _bridge_eligible_candidates(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
) -> list[tuple[CandidateResult, dict[str, float]]]:
    eligible: list[tuple[CandidateResult, dict[str, float]]] = []
    for candidate in candidates:
        if not _clean_downstream_candidate(candidate):
            continue
        snap = _candidate_validation_snapshot(candidate, fold)
        if snap["train_return"] <= -0.02 or snap["validation_return"] <= -0.02:
            continue
        if snap["validation_mdd"] > 0.22:
            continue
        eligible.append((candidate, snap))
    return eligible


def _preferred_dynamic_expert(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
) -> tuple[CandidateResult, dict[str, float]] | None:
    preferred = "dynamic_conviction_switch:t0.90_risk_capped_fallback"
    eligible = _bridge_eligible_candidates(candidates, fold)
    by_label = {candidate.candidate_label: (candidate, snap) for candidate, snap in eligible}
    if preferred in by_label:
        return by_label[preferred]
    dynamic = [item for item in eligible if item[0].family == "dynamic_conviction_switch"]
    if not dynamic:
        return None
    return max(dynamic, key=lambda item: _bridge_utility(item[0], fold, drawdown_penalty=1.0))


def _softmax_weights(raw_scores: Mapping[str, float], *, learning_rate: float, cap: float) -> dict[str, float]:
    if not raw_scores:
        return {}
    values = np.asarray(list(raw_scores.values()), dtype=float)
    center = float(np.mean(values))
    scale = float(np.std(values)) or 1.0
    exp_scores = {
        label: math.exp(max(-12.0, min(12.0, learning_rate * ((score - center) / scale))))
        for label, score in raw_scores.items()
    }
    return _normalize_capped_weights(exp_scores, cap=cap)


def _hybrid_assimilated_dynamic_candidates(
    candidates: Sequence[CandidateResult],
    fold: MonthlyFold,
    *,
    prior_completed_utilities: Mapping[str, Sequence[float]],
    bridge_manifest: Mapping[str, Any],
) -> list[CandidateResult]:
    """Absorb dynamic switch as a normal expert stream under frozen clean rules.

    This function never reads current-fold locked OOS.  Current-month weights use
    train/validation snapshots plus optional utilities from completed prior folds.
    The dynamic switch row's same-fold selected label is not used as a feature or
    target; the switch can enter only as a candidate return stream.
    """
    dynamic_item = _preferred_dynamic_expert(candidates, fold)
    eligible = _bridge_eligible_candidates(candidates, fold)
    if dynamic_item is None or len(eligible) < 2:
        return []

    dynamic_candidate, dynamic_snap = dynamic_item
    non_dynamic_ranked = sorted(
        [item for item in eligible if item[0].candidate_label != dynamic_candidate.candidate_label],
        key=lambda item: _bridge_utility(item[0], fold, drawdown_penalty=1.0),
        reverse=True,
    )[:7]
    pool = [dynamic_item, *non_dynamic_ranked]
    if len(pool) < 2:
        return []
    pool_candidates = [item[0] for item in pool]

    def build(label: str, weights: Mapping[str, float], mode: str) -> CandidateResult | None:
        clean_weights = {key: float(value) for key, value in weights.items() if float(value) > 0.0}
        if len(clean_weights) < 2:
            return None
        row = {
            "profile_id": label.replace(":", "_"),
            "profile_kind": "frozen_manifest_dynamic_expert_assimilation",
            "candidate_tier": "clean_train_validation_selected_paper_shadow",
            "selection_inputs": ["train", "validation", "lagged_completed_oos"],
            "uses_locked_oos_for_selection": False,
            "ready_for_paper": True,
            "ready_for_real": False,
            "real_money_execution": False,
            "weights": dict(clean_weights),
            "final_weights": dict(clean_weights),
            "dynamic_expert_label": dynamic_candidate.candidate_label,
            "dynamic_expert_used_as": "return_stream_only_no_same_fold_label_feature",
            "same_month_self_feeding": False,
            "current_fold_oos_used_for_weighting": False,
            "online_update_cutoff_fold": max(prior_completed_utilities.keys(), default=None),
            "bridge_protocol_manifest_version": bridge_manifest.get("manifest_version"),
            "bridge_protocol_manifest_sha256": bridge_manifest.get("_sha256"),
            "bridge_assimilation_mode": mode,
            "bridge_inputs": [candidate.candidate_label for candidate in pool_candidates],
        }
        return _candidate_eval(
            family="hybrid_oracle_bridge",
            label=label,
            row=row,
            returns=_blend_candidate_returns(pool_candidates, clean_weights),
        )

    out: list[CandidateResult] = []

    # v1: fixed manifest blend. Dynamic contributes, but cannot dominate.
    score_weights = _softmax_weights(
        {
            candidate.candidate_label: max(0.0, _bridge_utility(candidate, fold, drawdown_penalty=1.0))
            for candidate, _ in non_dynamic_ranked[:5]
        },
        learning_rate=0.30,
        cap=0.30,
    )
    fixed_dynamic_weight = 0.20
    fixed_weights = {
        label: value * (1.0 - fixed_dynamic_weight) for label, value in score_weights.items()
    }
    fixed_weights[dynamic_candidate.candidate_label] = fixed_dynamic_weight
    if candidate := build(
        "hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1",
        _normalize_capped_weights(fixed_weights, cap=0.30),
        "fixed_manifest_validation_blend",
    ):
        out.append(candidate)

    # riskcap: dynamic has higher starting weight but shrinks on validation drawdown.
    risk_cap = 0.10
    dynamic_scale = min(1.0, risk_cap / max(dynamic_snap["validation_mdd"], 0.01))
    dynamic_weight = 0.35 * dynamic_scale
    inv_mdd_weights = _normalize_capped_weights(
        {
            candidate.candidate_label: max(0.0, snap["validation_return"])
            / max(snap["validation_mdd"], 0.02)
            for candidate, snap in non_dynamic_ranked[:7]
        },
        cap=0.25,
    )
    risk_weights = {
        label: value * (1.0 - dynamic_weight) for label, value in inv_mdd_weights.items()
    }
    risk_weights[dynamic_candidate.candidate_label] = dynamic_weight
    if candidate := build(
        "hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1_riskcap",
        _normalize_capped_weights(risk_weights, cap=0.30),
        "validation_drawdown_risk_capped_blend",
    ):
        out.append(candidate)

    # hedge: train/validation score plus completed-prior-month utility only.
    hedge_scores: dict[str, float] = {}
    for candidate, _ in pool:
        history = list(prior_completed_utilities.get(candidate.candidate_label, ()))
        lagged = float(np.mean(history)) if history else 0.0
        hedge_scores[candidate.candidate_label] = _bridge_utility(
            candidate, fold, drawdown_penalty=1.0
        ) + 0.75 * lagged
    hedge_weights = _softmax_weights(hedge_scores, learning_rate=0.30, cap=0.30)
    # Keep an entropy floor from the manifest so the bridge stays diversified.
    if hedge_weights:
        floor = 0.05
        hedge_weights = _normalize_capped_weights(
            {label: weight + floor for label, weight in hedge_weights.items()}, cap=0.30
        )
    if candidate := build(
        "hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1_hedge",
        hedge_weights,
        "fully_lagged_hedge_validation_blend",
    ):
        out.append(candidate)
    return out


def _update_bridge_prior_utilities(
    prior_completed_utilities: dict[str, list[float]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for row in rows:
        label = str(row.get("candidate_label"))
        utility = _safe_float(row.get("locked_oos", {}).get("total_return")) - _safe_float(
            row.get("locked_oos", {}).get("mdd")
        )
        prior_completed_utilities.setdefault(label, []).append(float(utility))


def _append_profile_family_candidates(
    *,
    out: list[CandidateResult],
    family: str,
    profile_streams: Sequence[grid_hybrid.ProfileStream],
    profile_rows: Sequence[Mapping[str, Any]],
    static_row: Mapping[str, Any],
    v35: optuna_hybrid.OptunaModelResult,
    v36: optuna_hybrid.OptunaModelResult,
    selected_legal: Mapping[str, Any],
    selected_optuna: optuna_hybrid.OptunaModelResult,
) -> None:
    stream_by_id = {stream.profile_id: stream for stream in profile_streams}
    for row in profile_rows:
        profile_id = str(row["profile_id"])
        out.append(
            _candidate_eval(
                family=family,
                label=f"{family}:{profile_id}",
                row=row,
                returns=stream_by_id[profile_id].returns,
            )
        )
    out.append(
        _candidate_eval(
            family=family,
            label=f"{family}:static_guarded",
            row=static_row,
            returns=_combine_profile_returns(profile_streams, dict(static_row.get("final_weights") or {})),
        )
    )
    out.append(
        _candidate_eval(
            family=family,
            label=f"{family}:hybrid_v3_5",
            row=v35.row,
            returns=v35.returns,
        )
    )
    out.append(
        _candidate_eval(
            family=family,
            label=f"{family}:hybrid_v3_6",
            row=v36.row,
            returns=v36.returns,
        )
    )
    out.append(
        _candidate_eval(
            family=family,
            label=f"{family}:selected_optuna",
            row=selected_optuna.row,
            returns=selected_optuna.returns,
        )
    )
    selected_id = str(selected_legal.get("profile_id"))
    if selected_id == str(static_row.get("profile_id")):
        legal_returns = _combine_profile_returns(
            profile_streams, dict(static_row.get("final_weights") or {})
        )
    elif selected_id == str(v35.row.get("profile_id")):
        legal_returns = v35.returns
    elif selected_id == str(v36.row.get("profile_id")):
        legal_returns = v36.returns
    else:
        legal_returns = stream_by_id[selected_id].returns
    out.append(
        _candidate_eval(
            family=family,
            label=f"{family}:selected_train_validation_legal",
            row=selected_legal,
            returns=legal_returns,
        )
    )


def _individual_candidate_score(
    stream: broad69.CandidateStream, spec: Mapping[str, Any]
) -> float:
    row = stream.row
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_mdd = _safe_float(row.get("train_mdd"))
    val_mdd = _safe_float(row.get("validation_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    train_events = int(row.get("train_trade_event_count") or 0)
    val_events = int(row.get("validation_trade_event_count") or 0)
    spike = max(0.0, validation - max(train, 0.0))
    validation_cap = _safe_float(spec["max_validation_return"])
    stable_validation = min(validation, validation_cap)
    penalty = 0.0
    penalty += max(0.0, _safe_float(spec["min_train_return"]) - train) * 8.0
    penalty += max(0.0, _safe_float(spec["min_validation_return"]) - validation) * 16.0
    penalty += max(0.0, validation - validation_cap) * 10.0
    penalty += max(0.0, spike - _safe_float(spec["validation_spike_cap"])) * 8.0
    penalty += max(0.0, train_mdd - _safe_float(spec["max_train_mdd"])) * 3.0
    penalty += max(0.0, val_mdd - _safe_float(spec["max_validation_mdd"])) * 8.0
    penalty += max(0.0, 10.0 - train_rpt) / 12.0
    penalty += max(0.0, 10.0 - val_rpt) / 8.0
    penalty += max(0, 20 - train_events) / 12.0
    penalty += max(0, 5 - val_events) / 4.0
    penalty += spike * 3.0
    return float(
        4.0 * stable_validation
        + 1.5 * min(train, stable_validation)
        + min(max(train_rpt, -20.0), 160.0) / 240.0
        + min(max(val_rpt, -20.0), 160.0) / 180.0
        - 1.6 * val_mdd
        - 0.4 * train_mdd
        - penalty
    )


def _allocatable_individual_streams(
    streams: Sequence[broad69.CandidateStream], spec: Mapping[str, Any]
) -> list[broad69.CandidateStream]:
    def soft_pass(stream: broad69.CandidateStream) -> bool:
        row = stream.row
        return (
            _safe_float(row.get("train_return")) > 0.0
            and _safe_float(row.get("validation_return")) > 0.0
            and _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0) > 0.0
            and _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0) > 0.0
            and int(row.get("train_trade_event_count") or 0) >= 8
            and int(row.get("validation_trade_event_count") or 0) >= 2
        )

    preferred = [
        stream
        for stream in streams
        if soft_pass(stream)
        and _safe_float(stream.row.get("validation_return"))
        <= _safe_float(spec["max_validation_return"]) * 1.8
        and _safe_float(stream.row.get("validation_return"))
        <= _safe_float(stream.row.get("train_return")) + _safe_float(spec["validation_spike_cap"]) * 2.5
        and _safe_float(stream.row.get("validation_mdd"))
        <= _safe_float(spec["max_validation_mdd"]) * 1.35
        and _safe_float(stream.row.get("train_mdd")) <= _safe_float(spec["max_train_mdd"]) * 1.35
    ]
    pool = preferred or [stream for stream in streams if soft_pass(stream)] or list(streams)
    return sorted(pool, key=lambda stream: _individual_candidate_score(stream, spec), reverse=True)[
        : int(spec["candidate_pool_size"])
    ]


def _individual_profile_reasons(row: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _safe_float(row.get("train_return")) <= 0.0:
        reasons.append("train_return_not_positive")
    if _safe_float(row.get("validation_return")) < _safe_float(spec["min_validation_return"]):
        reasons.append("validation_return_below_profile_min")
    if _safe_float(row.get("validation_return")) > _safe_float(spec["max_validation_return"]) * 1.5:
        reasons.append("validation_return_above_antispike_cap")
    if _safe_float(row.get("validation_mdd")) > _safe_float(spec["max_validation_mdd"]) * 1.35:
        reasons.append("validation_mdd_above_relaxed_profile_cap")
    if int(row.get("selected_sleeve_count") or 0) < int(spec["min_sleeves"]):
        reasons.append("selected_sleeves_below_core_target")
    concentration = dict(row.get("concentration") or {})
    if _safe_float(concentration.get("top_symbol_share")) > _safe_float(spec["top_symbol_share_cap"]):
        reasons.append("top_symbol_share_above_cap")
    if _safe_float(concentration.get("top_asset_group_share")) > _safe_float(
        spec["top_asset_group_share_cap"]
    ):
        reasons.append("top_asset_group_share_above_cap")
    return reasons


def _individual_portfolio_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons = list(row.get("selection_reasons") or [])
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    validation_cap = _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["max_validation_return"])
    spike_cap = _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["validation_spike_cap"])
    if validation > validation_cap:
        reasons.append("portfolio_validation_return_above_antispike_cap")
    if validation > train + spike_cap:
        reasons.append("portfolio_validation_spike_above_cap")
    if train > _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["max_train_return"]):
        reasons.append("portfolio_train_return_above_stability_cap")
    if _safe_float(row.get("validation_mdd")) > _safe_float(
        INDIVIDUAL_PORTFOLIO_GUARD["max_validation_mdd"]
    ):
        reasons.append("portfolio_validation_mdd_above_cap")
    if _safe_float(row.get("train_mdd")) > _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["max_train_mdd"]):
        reasons.append("portfolio_train_mdd_above_cap")
    if _safe_float(row.get("gross_notional_fraction")) > _safe_float(
        INDIVIDUAL_PORTFOLIO_GUARD["max_gross_notional"]
    ):
        reasons.append("portfolio_gross_above_cap")
    return sorted(set(reasons))


def _apply_individual_portfolio_guard(row: dict[str, Any]) -> dict[str, Any]:
    row["selection_reasons"] = _individual_portfolio_reasons(row)
    row["paper_testnet_candidate"] = not row["selection_reasons"]
    row["ready_for_paper"] = row["paper_testnet_candidate"]
    row["individual_portfolio_guard"] = dict(INDIVIDUAL_PORTFOLIO_GUARD)
    return row


def _compact_individual_profile_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": row.get("profile_id"),
        "profile_kind": row.get("profile_kind"),
        "gross_notional_fraction": row.get("gross_notional_fraction"),
        "selected_sleeve_count": row.get("selected_sleeve_count"),
        "leverage_map": dict(row.get("leverage_map") or {}),
        "asset_gross_notional_fraction": dict(row.get("asset_gross_notional_fraction") or {}),
        "final_weights": dict(row.get("final_weights") or row.get("weights") or {}),
        "concentration": dict(row.get("concentration") or {}),
        "train_return": row.get("train_return"),
        "validation_return": row.get("validation_return"),
        "train_mdd": row.get("train_mdd"),
        "validation_mdd": row.get("validation_mdd"),
        "train_return_per_turnover_proxy_bps": row.get("train_return_per_turnover_proxy_bps"),
        "validation_return_per_turnover_proxy_bps": row.get(
            "validation_return_per_turnover_proxy_bps"
        ),
        "rebalance_policy": row.get("rebalance_policy"),
        "leverage_tuning_policy": row.get("leverage_tuning_policy"),
        "selection_reasons": list(row.get("selection_reasons") or []),
        "ready_for_paper": bool(row.get("ready_for_paper") or row.get("paper_testnet_candidate")),
    }


def _compact_individual_sleeve_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_profile_id": row.get("parent_profile_id"),
        "profile_weight_rank": row.get("profile_weight_rank"),
        "symbol": row.get("symbol"),
        "timeframe": row.get("timeframe"),
        "family": row.get("family"),
        "model_id": row.get("model_id"),
        "integer_leverage": row.get("integer_leverage"),
        "notional_fraction": row.get("notional_fraction"),
        "sleeve_multiplier": row.get("sleeve_multiplier"),
        "weighted_notional_fraction": row.get("weighted_notional_fraction"),
        "train_return": row.get("train_return"),
        "validation_return": row.get("validation_return"),
        "train_mdd": row.get("train_mdd"),
        "validation_mdd": row.get("validation_mdd"),
        "train_return_per_turnover_proxy_bps": row.get("train_return_per_turnover_proxy_bps"),
        "validation_return_per_turnover_proxy_bps": row.get(
            "validation_return_per_turnover_proxy_bps"
        ),
        "individual_candidate_score": row.get("individual_candidate_score"),
    }


def _individual_portfolio_guard_score(row: Mapping[str, Any]) -> float:
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_mdd = _safe_float(row.get("train_mdd"))
    val_mdd = _safe_float(row.get("validation_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    gross = _safe_float(row.get("gross_notional_fraction"))
    validation_cap = _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["max_validation_return"])
    stable_validation = min(validation, validation_cap)
    stable_train = min(train, _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["max_train_return"]))
    reasons = _individual_portfolio_reasons(row)
    penalty = 18.0 * len(reasons)
    penalty += max(0.0, validation - validation_cap) * 35.0
    penalty += max(
        0.0,
        validation - train - _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["validation_spike_cap"]),
    ) * 20.0
    penalty += max(0.0, train - _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["max_train_return"])) * 6.0
    penalty += max(0.0, val_mdd - _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["max_validation_mdd"])) * 20.0
    penalty += max(0.0, train_mdd - _safe_float(INDIVIDUAL_PORTFOLIO_GUARD["max_train_mdd"])) * 8.0
    return float(
        8.0 * stable_validation
        + 1.6 * min(stable_train, stable_validation)
        + min(max(val_rpt, -20.0), 160.0) / 140.0
        + min(max(train_rpt, -20.0), 160.0) / 220.0
        - 2.5 * val_mdd
        - 0.7 * train_mdd
        - 0.015 * gross
        - penalty
    )


def tune_individual_robust_profile(
    *,
    spec: Mapping[str, Any],
    candidate_streams: Sequence[broad69.CandidateStream],
    windows: broad69.SplitWindows,
    n_trials: int,
    seed: int,
) -> tuple[grid_hybrid.ProfileStream, dict[str, Any], list[dict[str, Any]]]:
    if profile69.optuna is None:
        raise RuntimeError("Optuna is required for individual robust profile tuning")
    ranked = _allocatable_individual_streams(candidate_streams, spec)[: int(spec["max_sleeves"])]
    if not ranked:
        raise ValueError(f"no candidate streams for {spec['profile_id']}")
    matrix = profile69._aligned_matrix(ranked)
    values = matrix.to_numpy(dtype=float)
    notionals = np.array(
        [_safe_float(stream.row.get("notional_fraction")) for stream in ranked], dtype=float
    )
    profile_id = str(spec["profile_id"])

    def multipliers_from_trial(trial: Any) -> np.ndarray:
        raw = np.array(
            [trial.suggest_float(f"m_{idx}", 0.0, 1.0) for idx in range(len(ranked))],
            dtype=float,
        )
        raw = np.where(raw < 0.04, 0.0, raw)
        if float(raw.sum()) <= 0.0:
            scores = [_individual_candidate_score(stream, spec) for stream in ranked]
            for idx in np.argsort(scores)[-min(int(spec["min_sleeves"]), len(ranked)) :]:
                raw[int(idx)] = 1.0
        gross = float(np.dot(raw, notionals))
        max_gross = _safe_float(spec["max_gross_notional"])
        if gross > max_gross and gross > 0.0:
            raw *= max_gross / gross
        return raw

    def score_multipliers(
        mult: np.ndarray,
    ) -> tuple[float, dict[str, Any], pd.Series, dict[str, float], dict[str, int]]:
        returns = pd.Series(values @ mult, index=matrix.index)
        turnover: dict[str, float] = {}
        events: dict[str, int] = {}
        for split in broad69.SPLIT_ORDER:
            turnover[split] = float(
                sum(
                    float(mult[idx])
                    * _safe_float(stream.row.get("notional_fraction"))
                    * int(stream.row.get(f"{split}_trade_event_count") or 0)
                    for idx, stream in enumerate(ranked)
                )
            )
            events[split] = int(
                sum(
                    int(stream.row.get(f"{split}_trade_event_count") or 0)
                    for idx, stream in enumerate(ranked)
                    if float(mult[idx]) > 1e-6
                )
            )
        metrics = profile69._profile_metrics_from_returns(
            returns, windows=windows, turnover_by_split=turnover, events_by_split=events
        )
        conc = profile69._profile_concentration(ranked, mult)
        train = _safe_float(metrics.get("train_return"))
        validation = _safe_float(metrics.get("validation_return"))
        train_mdd = _safe_float(metrics.get("train_mdd"))
        val_mdd = _safe_float(metrics.get("validation_mdd"))
        train_rpt = _safe_float(metrics.get("train_return_per_turnover_proxy_bps"), -100.0)
        val_rpt = _safe_float(metrics.get("validation_return_per_turnover_proxy_bps"), -100.0)
        active_count = int(np.sum(mult > 1e-6))
        spike = max(0.0, validation - max(train, 0.0))
        validation_cap = _safe_float(spec["max_validation_return"])
        stable_validation = min(validation, validation_cap)
        stable_train = min(train, max(validation_cap * 4.0, _safe_float(spec["min_train_return"])))
        penalty = 0.0
        penalty += max(0.0, _safe_float(spec["min_validation_return"]) - validation) * 30.0
        penalty += max(0.0, _safe_float(spec["min_train_return"]) - train) * 10.0
        penalty += max(0.0, validation - validation_cap) * 36.0
        penalty += max(
            0.0, spike - _safe_float(spec["validation_spike_cap"]) * 1.5
        ) * 16.0
        penalty += max(0.0, val_mdd - _safe_float(spec["max_validation_mdd"])) * 26.0
        penalty += max(0.0, train_mdd - _safe_float(spec["max_train_mdd"])) * 6.0
        penalty += max(0.0, 10.0 - train_rpt) / 4.0
        penalty += max(0.0, 10.0 - val_rpt) / 3.0
        penalty += max(0, int(spec["min_sleeves"]) - active_count) / 2.0
        penalty += max(
            0.0,
            _safe_float(conc.get("top_symbol_share")) - _safe_float(spec["top_symbol_share_cap"]),
        ) * 6.0
        penalty += max(
            0.0,
            _safe_float(conc.get("top_asset_group_share"))
            - _safe_float(spec["top_asset_group_share_cap"]),
        ) * 3.0
        penalty += spike * 5.0
        metrics["concentration"] = conc
        for split in grid_hybrid.ilp.SPLIT_ORDER:
            turnover.setdefault(split, 0.0)
            events.setdefault(split, 0)
        score = (
            8.0 * stable_validation
            + 1.8 * min(stable_train, stable_validation)
            + min(max(train_rpt, -20.0), 160.0) / 160.0
            + min(max(val_rpt, -20.0), 160.0) / 120.0
            - 3.2 * val_mdd
            - 0.7 * train_mdd
            - penalty
        )
        return float(score), metrics, returns, turnover, events

    equal_count = min(int(spec["min_sleeves"]), len(ranked))
    equal = {f"m_{idx}": 1.0 if idx < equal_count else 0.0 for idx in range(len(ranked))}
    study = profile69.run_optuna_study(
        optuna_module=profile69.optuna,
        objective=lambda trial: score_multipliers(multipliers_from_trial(trial))[0],
        n_trials=max(1, int(n_trials)),
        direction="maximize",
        seed=int(seed),
        enqueue_trials=[equal],
        n_jobs=1,
        show_progress_bar=False,
    )
    best_raw = np.array(
        [float(study.best_params.get(f"m_{idx}", 0.0)) for idx in range(len(ranked))],
        dtype=float,
    )
    best_raw = np.where(best_raw < 0.04, 0.0, best_raw)
    gross = float(np.dot(best_raw, notionals))
    if gross > _safe_float(spec["max_gross_notional"]) and gross > 0.0:
        best_raw *= _safe_float(spec["max_gross_notional"]) / gross
    if float(best_raw.sum()) <= 0.0:
        best_raw[:equal_count] = 1.0
    score, metrics, returns, turnover, events = score_multipliers(best_raw)

    selected_rows: list[dict[str, Any]] = []
    asset_gross: dict[str, float] = defaultdict(float)
    leverage_map: dict[str, int] = {}
    model_ids: list[str] = []
    for rank, (mult, stream) in enumerate(
        sorted(
            [(float(m), s) for m, s in zip(best_raw, ranked, strict=True) if float(m) > 1e-6],
            key=lambda item: item[0] * _safe_float(item[1].row.get("notional_fraction")),
            reverse=True,
        ),
        start=1,
    ):
        row = dict(stream.row)
        notional = mult * _safe_float(row.get("notional_fraction"))
        row.update(
            {
                "parent_profile_id": profile_id,
                "sleeve_multiplier": mult,
                "weighted_notional_fraction": notional,
                "profile_weight_rank": rank,
                "individual_candidate_score": _individual_candidate_score(stream, spec),
            }
        )
        selected_rows.append(row)
        asset_gross[str(row["symbol"])] += notional
        leverage_map[str(row["symbol"])] = max(
            int(leverage_map.get(str(row["symbol"]), 0)), int(row.get("integer_leverage") or 0)
        )
        model_ids.append(str(row["model_id"]))

    stream = grid_hybrid.ProfileStream(
        profile_id=profile_id,
        candidate_tier="individual_sleeve_first_robust_profile",
        leverage_map=leverage_map,
        gross_notional_fraction=float(metrics["concentration"]["gross_notional_fraction"]),
        asset_gross_notional_fraction=dict(sorted(asset_gross.items())),
        selected_model_ids=tuple(model_ids),
        returns=returns.sort_index(),
        turnover_by_split=turnover,
        trade_events_by_split=events,
        liquidation_count_by_split=dict.fromkeys(grid_hybrid.ilp.SPLIT_ORDER, 0),
    )
    row: dict[str, Any] = {
        "profile_id": profile_id,
        "profile_kind": str(spec["profile_kind"]),
        "candidate_tier": "paper_testnet_individual_sleeve_first_candidate",
        "leverage_map": leverage_map,
        "weights": {profile_id: 1.0},
        "gross_notional_fraction": stream.gross_notional_fraction,
        "optimizer": "optuna_tpe_individual_sleeve_first_allocation",
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "profile_spec": dict(spec),
        "concentration": metrics["concentration"],
        "asset_gross_notional_fraction": dict(sorted(asset_gross.items())),
        "selected_sleeve_count": len(selected_rows),
        "rebalance_policy": spec.get(
            "rebalance_policy", "monthly_refit_signal_level_position_updates"
        ),
        "leverage_tuning_policy": spec.get(
            "leverage_tuning_policy",
            "train_validation_only_source_integer_leverage_plus_post_allocation_multiplier",
        ),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
    }
    for split in broad69.SPLIT_ORDER:
        row[f"{split}_return"] = metrics[f"{split}_return"]
        row[f"{split}_mdd"] = metrics[f"{split}_mdd"]
        row[f"{split}_trade_event_count"] = metrics[f"{split}_trade_event_count"]
        row[f"{split}_return_per_turnover_proxy_bps"] = metrics[
            f"{split}_return_per_turnover_proxy_bps"
        ]
        row[f"{split}_liquidation_count"] = 0
        row[f"{split}_account_wipeout_count"] = 0
    row["train_validation_score"] = grid_hybrid._train_validation_score(row)
    row["selection_reasons"] = _individual_profile_reasons(row, spec)
    row["paper_testnet_candidate"] = not row["selection_reasons"]
    row["ready_for_paper"] = row["paper_testnet_candidate"]
    row["profile_objective_score"] = score
    return stream, row, selected_rows


def _run_individual_robust_family(
    *,
    fold: MonthlyFold,
    candidate_streams: Sequence[broad69.CandidateStream],
    hybrid_trials: int,
    seed: int,
) -> tuple[list[CandidateResult], dict[str, Any]]:
    windows = fold.windows()
    profile_streams: list[grid_hybrid.ProfileStream] = []
    profile_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(INDIVIDUAL_ROBUST_PROFILE_SPECS):
        try:
            profile_stream, profile_row, selected = tune_individual_robust_profile(
                spec=spec,
                candidate_streams=candidate_streams,
                windows=windows,
                n_trials=int(hybrid_trials),
                seed=int(seed) + idx * 10_000,
            )
        except ValueError:
            continue
        profile_streams.append(profile_stream)
        profile_rows.append(profile_row)
        sleeve_rows.extend(selected)

    if not profile_streams:
        return [], {
            "profile_row_count": 0,
            "selected_sleeve_row_count": 0,
            "skip_reason": "no_individual_robust_profile_streams",
        }

    split_windows = profile69._split_windows_for_hybrid(windows.as_payload())
    with optuna_hybrid._split_window_context(split_windows):
        v35 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_5",
            n_trials=int(hybrid_trials),
            seed=int(seed) + 700_000,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        v36 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_6",
            n_trials=int(hybrid_trials),
            seed=int(seed) + 700_001,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        static_guarded = profile69.optimize_static_profile_blend(
            profile_streams, n_trials=int(hybrid_trials), seed=int(seed) + 710_000
        )
        selected_optuna = optuna_hybrid._choose_selected_optuna_result([v35, v36])

    static_guarded = _apply_individual_portfolio_guard(dict(static_guarded))
    for row in profile_rows:
        _apply_individual_portfolio_guard(row)
    v35.row.update(_apply_individual_portfolio_guard(dict(v35.row)))
    v36.row.update(_apply_individual_portfolio_guard(dict(v36.row)))
    selected_optuna = max([v35, v36], key=lambda result: _individual_portfolio_guard_score(result.row))

    legal_pool = [dict(static_guarded), *profile_rows, dict(v35.row), dict(v36.row)]
    legal_pass = [row for row in legal_pool if not row.get("selection_reasons")]
    selected_legal = max(legal_pass or legal_pool, key=_individual_portfolio_guard_score)
    out: list[CandidateResult] = []
    _append_profile_family_candidates(
        out=out,
        family="individual_robust",
        profile_streams=profile_streams,
        profile_rows=profile_rows,
        static_row=static_guarded,
        v35=v35,
        v36=v36,
        selected_legal=selected_legal,
        selected_optuna=selected_optuna,
    )
    return out, {
        "profile_row_count": len(profile_rows),
        "selected_sleeve_row_count": len(sleeve_rows),
        "profile_ids": [row["profile_id"] for row in profile_rows],
        "profile_rows": [_compact_individual_profile_row(row) for row in profile_rows],
        "selected_sleeve_rows": [_compact_individual_sleeve_row(row) for row in sleeve_rows],
    }


def _mark_asset_timeframe_leverage_row(row: dict[str, Any]) -> dict[str, Any]:
    row.setdefault("profile_kind", "asset_timeframe_leverage_scaled_profile")
    row["rebalance_policy"] = "monthly_refit_signal_level_position_updates"
    row["leverage_tuning_policy"] = (
        "train_validation_only_source_integer_leverage_plus_post_allocation_multiplier"
    )
    row["candidate_tier"] = row.get("candidate_tier") or "clean_train_validation_selected_paper_shadow"
    row["ready_for_real"] = False
    row["real_money_execution"] = False
    row["real_execution_allowed"] = False
    return row


def _run_asset_timeframe_leverage_family(
    *,
    fold: MonthlyFold,
    candidate_streams: Sequence[broad69.CandidateStream],
    hybrid_trials: int,
    seed: int,
) -> tuple[list[CandidateResult], dict[str, Any]]:
    """Clean train/validation-only post-allocation leverage scaling.

    Source sleeves already choose ``symbol x timeframe x integer_leverage`` by
    train/validation Optuna.  This family adds a second, portfolio-level
    multiplier layer over those source sleeves.  It is not a new signal and it
    does not read locked OOS; it tests whether practical exposure should be
    rebalanced across asset/timeframe sleeves instead of only across the
    higher-level hybrid candidates.
    """
    windows = fold.windows()
    profile_streams: list[grid_hybrid.ProfileStream] = []
    profile_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(ASSET_TIMEFRAME_LEVERAGE_PROFILE_SPECS):
        try:
            profile_stream, profile_row, selected = tune_individual_robust_profile(
                spec=spec,
                candidate_streams=candidate_streams,
                windows=windows,
                n_trials=int(hybrid_trials),
                seed=int(seed) + idx * 10_000,
            )
        except ValueError:
            continue
        _mark_asset_timeframe_leverage_row(profile_row)
        profile_streams.append(profile_stream)
        profile_rows.append(profile_row)
        sleeve_rows.extend(selected)

    if not profile_streams:
        return [], {
            "profile_row_count": 0,
            "selected_sleeve_row_count": 0,
            "skip_reason": "no_asset_timeframe_leverage_profile_streams",
        }

    split_windows = profile69._split_windows_for_hybrid(windows.as_payload())
    with optuna_hybrid._split_window_context(split_windows):
        v35 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_5",
            n_trials=int(hybrid_trials),
            seed=int(seed) + 700_000,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        v36 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_6",
            n_trials=int(hybrid_trials),
            seed=int(seed) + 700_001,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        static_guarded = profile69.optimize_static_profile_blend(
            profile_streams, n_trials=int(hybrid_trials), seed=int(seed) + 710_000
        )
        selected_optuna = optuna_hybrid._choose_selected_optuna_result([v35, v36])

    static_guarded = _mark_asset_timeframe_leverage_row(
        _apply_individual_portfolio_guard(dict(static_guarded))
    )
    for row in profile_rows:
        _apply_individual_portfolio_guard(row)
    v35.row.update(_mark_asset_timeframe_leverage_row(_apply_individual_portfolio_guard(dict(v35.row))))
    v36.row.update(_mark_asset_timeframe_leverage_row(_apply_individual_portfolio_guard(dict(v36.row))))
    selected_optuna = max([v35, v36], key=lambda result: _individual_portfolio_guard_score(result.row))

    legal_pool = [dict(static_guarded), *profile_rows, dict(v35.row), dict(v36.row)]
    legal_pass = [row for row in legal_pool if not row.get("selection_reasons")]
    selected_legal = _mark_asset_timeframe_leverage_row(
        max(legal_pass or legal_pool, key=_individual_portfolio_guard_score)
    )
    out: list[CandidateResult] = []
    _append_profile_family_candidates(
        out=out,
        family="asset_timeframe_leverage",
        profile_streams=profile_streams,
        profile_rows=profile_rows,
        static_row=static_guarded,
        v35=v35,
        v36=v36,
        selected_legal=selected_legal,
        selected_optuna=selected_optuna,
    )
    return out, {
        "profile_row_count": len(profile_rows),
        "selected_sleeve_row_count": len(sleeve_rows),
        "profile_ids": [row["profile_id"] for row in profile_rows],
        "profile_rows": [_compact_individual_profile_row(row) for row in profile_rows],
        "selected_sleeve_rows": [_compact_individual_sleeve_row(row) for row in sleeve_rows],
    }


def _run_source_profile_family(
    *,
    fold: MonthlyFold,
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
    bars: Mapping[tuple[str, str], pd.DataFrame],
    cache: profile69.FeatureCache,
    data_coverage: Mapping[str, Any],
    asset_trials: int,
    profile_trials: int,
    hybrid_trials: int,
    seed: int,
    allocation_fraction: float,
) -> tuple[list[CandidateResult], dict[str, Any]]:
    windows = fold.windows()
    train_eligibility = broad69.build_train_eligibility_report(
        bars,
        symbols=symbols,
        timeframes=timeframes,
        windows=windows,
    )
    asset_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    individual_streams: list[broad69.CandidateStream] = []
    profile_streams: list[grid_hybrid.ProfileStream] = []
    for profile_idx, base_spec in enumerate(profile69.PROFILE_SPECS):
        profile_spec = {**base_spec, "_timeframes": timeframes}
        streams: list[broad69.CandidateStream] = []
        for symbol_idx, symbol in enumerate(symbols):
            eligible_timeframes = profile69._eligible_timeframes_for_symbol(
                train_eligibility, symbol, timeframes
            )
            if not eligible_timeframes:
                continue
            symbol_spec = {**base_spec, "_timeframes": eligible_timeframes}
            stream = profile69.tune_symbol_profile(
                symbol=symbol,
                spec=symbol_spec,
                cache=cache,
                windows=windows,
                n_trials=int(asset_trials),
                seed=int(seed) + profile_idx * 100_000 + symbol_idx,
                allocation_fraction=float(allocation_fraction),
            )
            if stream is None:
                continue
            asset_rows.append(dict(stream.row))
            individual_streams.append(stream)
            if _safe_float(stream.row.get("profile_objective_score"), -1e18) > -1e8:
                streams.append(stream)
        profile_stream, profile_row, selected_sleeves = profile69.tune_profile_allocations(
            spec=profile_spec,
            candidate_streams=streams,
            windows=windows,
            n_trials=int(profile_trials),
            seed=int(seed) + profile_idx * 10_000 + 500,
        )
        profile_streams.append(profile_stream)
        profile_rows.append(profile_row)
        sleeve_rows.extend(selected_sleeves)

    split_windows = profile69._split_windows_for_hybrid(windows.as_payload())
    with optuna_hybrid._split_window_context(split_windows):
        v35 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_5",
            n_trials=int(hybrid_trials),
            seed=int(seed) + 700_000,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        v36 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_6",
            n_trials=int(hybrid_trials),
            seed=int(seed) + 700_001,
            fit_splits=("train",),
            warmup_splits=("train",),
            require_locked_oos_gate=False,
        )
        static_guarded = profile69.optimize_static_profile_blend(
            profile_streams, n_trials=int(hybrid_trials), seed=int(seed) + 710_000
        )
        selected_optuna = optuna_hybrid._choose_selected_optuna_result([v35, v36])
    legal_pool = [dict(static_guarded), *profile_rows, dict(v35.row), dict(v36.row)]
    legal_pass = [row for row in legal_pool if not row.get("selection_reasons")]
    selected_legal = max(legal_pass or legal_pool, key=lambda row: grid_hybrid._train_validation_score(row))

    source_payload = {
        "artifact_kind": "alpha_zoo_69_asset_profile_optuna_hybrid_refit_monthly_fold_source",
        "generated_at_utc": _utc_now_iso(),
        "universe": {"symbol_count": len(symbols), "symbols": list(symbols)},
        "timeframes": list(timeframes),
        "split_policy": windows.as_payload(),
        "data_coverage": dict(data_coverage),
        "train_eligibility": train_eligibility,
        "research_primary_round_trip_cost_bps": broad69.PRIMARY_ROUND_TRIP_COST_BPS,
        "asset_tuning_rows": asset_rows,
        "profile_rows": profile_rows,
        "selected_sleeve_rows": sleeve_rows,
        "selected_optuna_hybrid_profile": dict(selected_optuna.row),
        "static_train_dominance_guarded_hybrid": dict(static_guarded),
        "selected_train_validation_legal_portfolio": dict(selected_legal),
        "hybrid_v3_5_optuna": {"row": dict(v35.row)},
        "hybrid_v3_6_optuna": {"row": dict(v36.row)},
    }
    candidates: list[CandidateResult] = []
    _append_profile_family_candidates(
        out=candidates,
        family="profile_optuna",
        profile_streams=profile_streams,
        profile_rows=profile_rows,
        static_row=static_guarded,
        v35=v35,
        v36=v36,
        selected_legal=selected_legal,
        selected_optuna=selected_optuna,
    )
    aux = {
        "source_payload": source_payload,
        "profile_streams": profile_streams,
        "individual_streams": individual_streams,
        "asset_tuning_row_count": len(asset_rows),
        "selected_sleeve_row_count": len(sleeve_rows),
        "train_eligible_symbol_count": train_eligibility["train_eligible_symbol_count"],
        "train_ineligible_symbols": train_eligibility["train_ineligible_symbols"],
    }
    return candidates, aux


def _run_efficiency_family(
    *,
    fold: MonthlyFold,
    family: str,
    source_payload: Mapping[str, Any],
    bars: Mapping[tuple[str, str], pd.DataFrame],
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
    cache: profile69.FeatureCache,
    profile_trials: int,
    hybrid_trials: int,
    seed: int,
    relaxed: bool,
) -> tuple[list[CandidateResult], dict[str, Any]]:
    windows = fold.windows()
    train_eligibility = broad69.build_train_eligibility_report(
        bars,
        symbols=symbols,
        timeframes=timeframes,
        windows=windows,
    )
    candidates: list[CandidateResult] = []
    profile_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    profile_streams: list[grid_hybrid.ProfileStream] = []
    specs = relaxed_eff.RELAXED_PROFILE_SPECS if relaxed else strict_eff.EFFICIENCY_PROFILE_SPECS
    for idx, spec in enumerate(specs):
        if relaxed:
            candidate_streams = relaxed_eff._build_candidate_streams(
                source_payload=source_payload,
                spec=spec,
                cache=cache,
                windows=windows,
                train_eligibility=train_eligibility,
            )
            if not candidate_streams:
                continue
            try:
                profile_stream, profile_row, selected = relaxed_eff.tune_relaxed_profile_allocations(
                    spec=spec,
                    candidate_streams=candidate_streams,
                    windows=windows,
                    n_trials=int(profile_trials),
                    seed=int(seed) + idx * 10_000,
                )
            except ValueError:
                continue
        else:
            candidate_streams = strict_eff._build_candidate_streams(
                source_payload=source_payload,
                spec=spec,
                cache=cache,
                windows=windows,
                train_eligibility=train_eligibility,
            )
            if not candidate_streams:
                continue
            try:
                profile_stream, profile_row, selected = strict_eff.tune_efficiency_profile_allocations(
                    spec=spec,
                    candidate_streams=candidate_streams,
                    windows=windows,
                    n_trials=int(profile_trials),
                    seed=int(seed) + idx * 10_000,
                )
            except ValueError:
                continue
        profile_streams.append(profile_stream)
        profile_rows.append(profile_row)
        sleeve_rows.extend(selected)

    if not profile_streams:
        return [], {
            "profile_row_count": 0,
            "selected_sleeve_row_count": 0,
            "train_eligible_symbol_count": train_eligibility["train_eligible_symbol_count"],
            "skip_reason": "no_allocatable_efficiency_profile_streams",
        }

    split_windows = strict_eff._split_window_payload_for_hybrid(windows)
    if relaxed:
        context = relaxed_eff._relaxed_optuna_hybrid_objective_context()
    else:
        context = None
    with optuna_hybrid._split_window_context(split_windows):
        if context is not None:
            context.__enter__()
        try:
            v35 = optuna_hybrid._run_optuna(
                profile_streams,
                version="v3_5",
                n_trials=int(hybrid_trials),
                seed=int(seed) + 700_000,
                fit_splits=("train",),
                warmup_splits=("train",),
                require_locked_oos_gate=False,
            )
            v36 = optuna_hybrid._run_optuna(
                profile_streams,
                version="v3_6",
                n_trials=int(hybrid_trials),
                seed=int(seed) + 700_001,
                fit_splits=("train",),
                warmup_splits=("train",),
                require_locked_oos_gate=False,
            )
            if relaxed:
                static_guarded = relaxed_eff.optimize_relaxed_blend(
                    profile_streams,
                    n_trials=int(hybrid_trials),
                    seed=int(seed) + 710_000,
                )
            else:
                static_guarded = strict_eff.optimize_efficiency_blend(
                    profile_streams,
                    n_trials=int(hybrid_trials),
                    seed=int(seed) + 710_000,
                )
        finally:
            if context is not None:
                context.__exit__(None, None, None)

    hybrid_rows = [dict(static_guarded), dict(v35.row), dict(v36.row)]
    for row in hybrid_rows[1:]:
        row.update(strict_eff._stress_metrics(row, _stress_turnover_for_row(profile_streams, row)))
        if relaxed:
            relaxed_eff._apply_relaxed_hybrid_row_fields(row)
        else:
            row["selection_reasons"] = strict_eff._selection_reasons(row, max_gross=8.0)
            row["diagnostic_warnings"] = strict_eff._diagnostic_warnings(row)
            row["ready_for_paper"] = not row["selection_reasons"]
            row["paper_testnet_candidate"] = row["ready_for_paper"]
            row["ready_for_real"] = False
            row["real_money_execution"] = False
            row["real_execution_allowed"] = False

    if relaxed:
        selected_legal = relaxed_eff._select_legal([*profile_rows, *hybrid_rows])
        selected_optuna = max([v35, v36], key=lambda result: relaxed_eff._relaxed_hybrid_objective_score(result.row))
        selected_optuna_row = dict(selected_optuna.row)
        selected_optuna_row.update(
            strict_eff._stress_metrics(
                selected_optuna_row,
                _stress_turnover_for_row(profile_streams, selected_optuna_row),
            )
        )
        relaxed_eff._apply_relaxed_hybrid_row_fields(selected_optuna_row)
    else:
        selected_legal = strict_eff._select_legal([*profile_rows, *hybrid_rows])
        selected_optuna = optuna_hybrid._choose_selected_optuna_result([v35, v36])
        selected_optuna_row = dict(selected_optuna.row)
        selected_optuna_row.update(
            strict_eff._stress_metrics(
                selected_optuna_row,
                _stress_turnover_for_row(profile_streams, selected_optuna_row),
            )
        )
        selected_optuna_row["selection_reasons"] = strict_eff._selection_reasons(
            selected_optuna_row, max_gross=8.0
        )
    # Override selected Optuna row in a shallow result copy for reporting labels.
    selected_optuna_for_eval = optuna_hybrid.OptunaModelResult(
        row=selected_optuna_row,
        returns=selected_optuna.returns,
        weights=selected_optuna.weights,
        allocations=selected_optuna.allocations,
        learned_params=selected_optuna.learned_params,
        params=selected_optuna.params,
        optuna=selected_optuna.optuna,
        top_trials=selected_optuna.top_trials,
    )
    v35_for_eval = optuna_hybrid.OptunaModelResult(
        row=hybrid_rows[1],
        returns=v35.returns,
        weights=v35.weights,
        allocations=v35.allocations,
        learned_params=v35.learned_params,
        params=v35.params,
        optuna=v35.optuna,
        top_trials=v35.top_trials,
    )
    v36_for_eval = optuna_hybrid.OptunaModelResult(
        row=hybrid_rows[2],
        returns=v36.returns,
        weights=v36.weights,
        allocations=v36.allocations,
        learned_params=v36.learned_params,
        params=v36.params,
        optuna=v36.optuna,
        top_trials=v36.top_trials,
    )
    _append_profile_family_candidates(
        out=candidates,
        family=family,
        profile_streams=profile_streams,
        profile_rows=profile_rows,
        static_row=static_guarded,
        v35=v35_for_eval,
        v36=v36_for_eval,
        selected_legal=selected_legal,
        selected_optuna=selected_optuna_for_eval,
    )
    aux = {
        "profile_row_count": len(profile_rows),
        "selected_sleeve_row_count": len(sleeve_rows),
        "train_eligible_symbol_count": train_eligibility["train_eligible_symbol_count"],
    }
    return candidates, aux


def _evaluate_candidate(candidate: CandidateResult, fold: MonthlyFold) -> dict[str, Any]:
    train = _period_metrics(candidate.returns, fold.train)
    validation = _period_metrics(candidate.returns, fold.validation)
    locked_oos = _period_metrics(candidate.returns, fold.locked_oos)
    row = dict(candidate.row)
    post_oos_research_variant = bool(row.get("post_oos_research_variant", False))
    requires_fresh_forward_shadow = bool(row.get("requires_fresh_forward_shadow", False))
    source_post_oos_research_variant = bool(row.get("source_post_oos_research_variant", False))
    uses_locked_oos_for_selection = bool(row.get("uses_locked_oos_for_selection", False))
    nested_hybrid_dependency = _row_references_non_leaf_material(row)
    return {
        "fold_id": fold.fold_id,
        "family": candidate.family,
        "candidate_label": candidate.candidate_label,
        "source_profile_id": candidate.source_profile_id,
        "train": train,
        "validation": validation,
        "locked_oos": locked_oos,
        "row_train_return": row.get("train_return"),
        "row_validation_return": row.get("validation_return"),
        "gross_notional_fraction": row.get("gross_notional_fraction"),
        "final_weights": row.get("final_weights") or row.get("weights") or {},
        "asset_gross_notional_fraction": row.get("asset_gross_notional_fraction") or {},
        "timeframe": row.get("timeframe"),
        "profile_kind": row.get("profile_kind"),
        "rebalance_policy": row.get("rebalance_policy"),
        "leverage_tuning_policy": row.get("leverage_tuning_policy"),
        "candidate_tier": row.get("candidate_tier"),
        "selection_reasons": row.get("selection_reasons") or [],
        "selection_inputs": row.get("selection_inputs") or ["train", "validation"],
        "uses_locked_oos_for_selection": uses_locked_oos_for_selection,
        "post_oos_research_variant": post_oos_research_variant,
        "requires_fresh_forward_shadow": requires_fresh_forward_shadow,
        "nested_hybrid_dependency": nested_hybrid_dependency,
        "clean_promotion_eligible": not uses_locked_oos_for_selection
        and not post_oos_research_variant
        and not requires_fresh_forward_shadow
        and not source_post_oos_research_variant
        and not nested_hybrid_dependency,
        "ready_for_paper": bool(row.get("ready_for_paper") or row.get("paper_testnet_candidate")),
        "selected_candidate_label": row.get("selected_candidate_label"),
        "aggressive_candidate_label": row.get("aggressive_candidate_label"),
        "fallback_candidate_label": row.get("fallback_candidate_label"),
        "dynamic_expert_label": row.get("dynamic_expert_label"),
        "dynamic_expert_used_as": row.get("dynamic_expert_used_as"),
        "same_month_self_feeding": bool(row.get("same_month_self_feeding", False)),
        "current_fold_oos_used_for_weighting": bool(
            row.get("current_fold_oos_used_for_weighting", False)
        ),
        "online_update_cutoff_fold": row.get("online_update_cutoff_fold"),
        "bridge_protocol_manifest_version": row.get("bridge_protocol_manifest_version"),
        "bridge_protocol_manifest_sha256": row.get("bridge_protocol_manifest_sha256"),
        "bridge_assimilation_mode": row.get("bridge_assimilation_mode"),
        "bridge_inputs": row.get("bridge_inputs") or [],
        "dynamic_aware_fit_policy": row.get("dynamic_aware_fit_policy"),
        "dynamic_aware_inputs": row.get("dynamic_aware_inputs") or [],
        "dynamic_input_labels": row.get("dynamic_input_labels") or [],
        "robust_core_input_labels": row.get("robust_core_input_labels") or [],
    }


def _aggregate_rows(fold_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        by_label[str(row["candidate_label"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for label, rows in sorted(by_label.items()):
        oos_returns = [_safe_float(row["locked_oos"]["total_return"]) for row in rows]
        val_returns = [_safe_float(row["validation"]["total_return"]) for row in rows]
        train_returns = [_safe_float(row["train"]["total_return"]) for row in rows]
        oos_mdds = [_safe_float(row["locked_oos"]["mdd"]) for row in rows]
        val_mdds = [_safe_float(row["validation"]["mdd"]) for row in rows]
        compounded = float(np.prod([1.0 + value for value in oos_returns]) - 1.0)
        oos_array = np.asarray(oos_returns, dtype=float)
        positive = oos_array[oos_array > 0.0]
        negative = oos_array[oos_array < 0.0]
        monthly_std = float(np.std(oos_array, ddof=1)) if oos_array.size > 1 else 0.0
        downside_std = float(np.std(negative, ddof=1)) if negative.size > 1 else 0.0
        fold_count = int(oos_array.size)
        annualized_comp = (
            float((1.0 + compounded) ** (12.0 / fold_count) - 1.0)
            if fold_count > 0 and compounded > -1.0
            else 0.0
        )
        monthly_equity = np.cumprod(1.0 + oos_array) if oos_array.size else np.asarray([])
        monthly_equity_mdd = 0.0
        if monthly_equity.size:
            running_peak = np.maximum.accumulate(monthly_equity)
            monthly_equity_mdd = float(np.max(1.0 - monthly_equity / np.maximum(running_peak, 1e-12)))
        centered = oos_array - float(np.mean(oos_array)) if oos_array.size else oos_array
        population_std = float(np.std(oos_array, ddof=0)) if oos_array.size else 0.0
        monthly_skew = (
            float(np.mean(centered**3) / (population_std**3))
            if population_std > 0.0 and oos_array.size
            else 0.0
        )
        monthly_excess_kurtosis = (
            float(np.mean(centered**4) / (population_std**4) - 3.0)
            if population_std > 0.0 and oos_array.size
            else 0.0
        )
        loss_streak = 0
        max_loss_streak = 0
        for value in oos_returns:
            if value < 0.0:
                loss_streak += 1
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:
                loss_streak = 0
        var_05 = float(np.quantile(oos_array, 0.05)) if oos_array.size else 0.0
        q95 = float(np.quantile(oos_array, 0.95)) if oos_array.size else 0.0
        cvar_25 = (
            float(np.mean(oos_array[oos_array <= np.quantile(oos_array, 0.25)]))
            if oos_array.size
            else 0.0
        )
        clean_candidate = all(bool(row.get("clean_promotion_eligible", True)) for row in rows)
        beats_challenger = (
            compounded > CURRENT_CHALLENGER_OOS_COMP and max(oos_mdds, default=0.0) <= CURRENT_CHALLENGER_MAX_OOS_MDD
        )
        material_risk_improvement = (
            compounded >= CURRENT_CHALLENGER_OOS_COMP * 0.85
            and max(oos_mdds, default=0.0) <= ROBUST_DEFAULT_MAX_OOS_MDD_LIMIT
            and min(oos_returns, default=0.0) >= -0.02
        )
        robust_default_improvement = (
            compounded > ROBUST_DEFAULT_OOS_COMP
            and max(oos_mdds, default=0.0) <= ROBUST_DEFAULT_MAX_OOS_MDD_LIMIT
        )
        nested_hybrid_dependency = any(bool(row.get("nested_hybrid_dependency")) for row in rows)
        post_oos_research_variant = any(bool(row.get("post_oos_research_variant")) for row in rows)
        requires_fresh_forward_shadow = any(
            bool(row.get("requires_fresh_forward_shadow")) for row in rows
        )
        uses_locked_oos_for_selection = any(
            bool(row.get("uses_locked_oos_for_selection")) for row in rows
        )
        non_clean_reasons = [
            reason
            for reason, active in (
                ("nested_hybrid_dependency", nested_hybrid_dependency),
                ("post_oos_research_variant", post_oos_research_variant),
                ("requires_fresh_forward_shadow", requires_fresh_forward_shadow),
                ("uses_locked_oos_for_selection", uses_locked_oos_for_selection),
            )
            if active
        ]
        if not clean_candidate and not non_clean_reasons:
            non_clean_reasons.append("non_clean_fold_flag")
        hard_stop_promotable = clean_candidate and (
            beats_challenger or material_risk_improvement or robust_default_improvement
        )
        aggregate.append(
            {
                "candidate_label": label,
                "family": rows[0].get("family"),
                "fold_count": len(rows),
                "clean_promotion_eligible": clean_candidate,
                "nested_hybrid_dependency": nested_hybrid_dependency,
                "post_oos_research_variant": post_oos_research_variant,
                "requires_fresh_forward_shadow": requires_fresh_forward_shadow,
                "uses_locked_oos_for_selection": uses_locked_oos_for_selection,
                "non_clean_reasons": non_clean_reasons,
                "compounded_oos_return": compounded,
                "annualized_oos_return_approx": annualized_comp,
                "mean_oos_return": float(np.mean(oos_returns)) if oos_returns else 0.0,
                "median_oos_return": float(np.median(oos_returns)) if oos_returns else 0.0,
                "min_oos_return": min(oos_returns) if oos_returns else 0.0,
                "latest_oos_return": oos_returns[-1] if oos_returns else 0.0,
                "positive_oos_folds": sum(value > 0.0 for value in oos_returns),
                "oos_hit_rate": (
                    float(sum(value > 0.0 for value in oos_returns) / len(oos_returns))
                    if oos_returns
                    else 0.0
                ),
                "positive_validation_folds": sum(value > 0.0 for value in val_returns),
                "min_validation_return": min(val_returns) if val_returns else 0.0,
                "mean_validation_return": float(np.mean(val_returns)) if val_returns else 0.0,
                "mean_train_return": float(np.mean(train_returns)) if train_returns else 0.0,
                "max_oos_mdd": max(oos_mdds) if oos_mdds else 0.0,
                "monthly_equity_mdd": monthly_equity_mdd,
                "max_validation_mdd": max(val_mdds) if val_mdds else 0.0,
                "monthly_volatility": monthly_std,
                "monthly_downside_volatility": downside_std,
                "monthly_sharpe_approx": (
                    float(np.mean(oos_array) / monthly_std * math.sqrt(12.0))
                    if monthly_std > 0.0
                    else 0.0
                ),
                "monthly_sortino_approx": (
                    float(np.mean(oos_array) / downside_std * math.sqrt(12.0))
                    if downside_std > 0.0
                    else 0.0
                ),
                "monthly_var_05": var_05,
                "monthly_quantile_95": q95,
                "monthly_cvar_25": cvar_25,
                "tail_ratio_95_05": (
                    float(q95 / abs(var_05)) if abs(var_05) > 0.0 else 0.0
                ),
                "monthly_skew": monthly_skew,
                "monthly_excess_kurtosis": monthly_excess_kurtosis,
                "avg_gain": float(np.mean(positive)) if positive.size else 0.0,
                "avg_loss": float(np.mean(negative)) if negative.size else 0.0,
                "gain_loss_ratio": (
                    float(np.mean(positive) / abs(np.mean(negative)))
                    if positive.size and negative.size and abs(float(np.mean(negative))) > 0.0
                    else 0.0
                ),
                "profit_factor": (
                    float(np.sum(positive) / abs(np.sum(negative)))
                    if positive.size and negative.size and abs(float(np.sum(negative))) > 0.0
                    else 0.0
                ),
                "omega_0": (
                    float(np.sum(np.maximum(oos_array, 0.0)) / np.sum(np.maximum(-oos_array, 0.0)))
                    if oos_array.size and np.sum(np.maximum(-oos_array, 0.0)) > 0.0
                    else 0.0
                ),
                "oos_return_to_max_mdd": (
                    float(compounded / max(oos_mdds)) if oos_mdds and max(oos_mdds) > 0.0 else 0.0
                ),
                "annualized_return_to_monthly_equity_mdd": (
                    float(annualized_comp / monthly_equity_mdd)
                    if monthly_equity_mdd > 0.0
                    else 0.0
                ),
                "max_loss_streak": int(max_loss_streak),
                "ready_for_paper_folds": sum(bool(row.get("ready_for_paper")) for row in rows),
                "hard_stop_promotable": hard_stop_promotable,
                "hard_stop_reasons": {
                    "beats_challenger": beats_challenger,
                    "material_risk_improvement": material_risk_improvement,
                    "robust_default_improvement": robust_default_improvement,
                    "clean_candidate": clean_candidate,
                    "current_challenger_oos_comp": CURRENT_CHALLENGER_OOS_COMP,
                    "current_challenger_max_oos_mdd": CURRENT_CHALLENGER_MAX_OOS_MDD,
                    "robust_default_oos_comp": ROBUST_DEFAULT_OOS_COMP,
                    "robust_default_max_oos_mdd_limit": ROBUST_DEFAULT_MAX_OOS_MDD_LIMIT,
                },
            }
        )
    return sorted(
        aggregate,
        key=lambda row: (
            int(row["positive_oos_folds"]),
            _safe_float(row["compounded_oos_return"]),
            _safe_float(row["min_oos_return"]),
            -_safe_float(row["max_oos_mdd"]),
        ),
        reverse=True,
    )


def _clean_promotion_rankings(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(row) for row in aggregate_rows if bool(row.get("clean_promotion_eligible"))]


def _demoted_nested_or_historical_rankings(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in aggregate_rows
        if not bool(row.get("clean_promotion_eligible"))
        and (
            bool(row.get("nested_hybrid_dependency"))
            or bool(row.get("post_oos_research_variant"))
            or bool(row.get("requires_fresh_forward_shadow"))
            or bool(row.get("uses_locked_oos_for_selection"))
            or bool(row.get("non_clean_reasons"))
        )
    ]


def _refresh_payload_derived_reports(payload: dict[str, Any]) -> None:
    """Refresh aggregate reports that depend on fold candidate rows."""
    rows = list(payload.get("fold_candidate_rows") or [])
    aggregate = _aggregate_rows(rows)
    clean = _clean_promotion_rankings(aggregate)
    demoted = _demoted_nested_or_historical_rankings(aggregate)
    payload["aggregate_rankings"] = aggregate
    payload["clean_promotion_rankings"] = clean
    payload["demoted_nested_or_historical_rankings"] = demoted
    payload["dynamic_self_feed_audit"] = _dynamic_self_feed_audit(rows)
    payload["metric_reconciliation"] = _metric_reconciliation_report(payload)
    payload["promotability"] = (
        _promotability_decision(clean[0])
        if clean
        else (
            _promotability_decision(aggregate[0])
            if aggregate
            else _promotability_decision({})
        )
    )


def _sanitize_research_dependency_flags(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Mark rows contaminated by post-OOS research inputs as non-clean.

    This is a fast report-only repair path.  It lets us fix clean/promotability
    labeling without rerunning the expensive per-fold Optuna search whenever a
    downstream audit rule changes.
    """
    mutable_rows = [dict(row) for row in rows]
    post_oos_labels = {
        str(row.get("candidate_label"))
        for row in mutable_rows
        if row.get("post_oos_research_variant")
        or row.get("requires_fresh_forward_shadow")
        or row.get("source_post_oos_research_variant")
    }
    post_oos_prefixes = ("mdd30_", "risk_enhanced_blend:")

    changed = True
    while changed:
        changed = False
        for row in mutable_rows:
            label = str(row.get("candidate_label"))
            refs = _row_reference_labels(row)
            contaminated = any(ref in post_oos_labels for ref in refs) or any(
                ref.startswith(post_oos_prefixes) for ref in refs
            )
            if contaminated and label not in post_oos_labels:
                post_oos_labels.add(label)
                changed = True

    for row in mutable_rows:
        label = str(row.get("candidate_label"))
        nested_hybrid_dependency = _row_references_non_leaf_material(row)
        contaminated = label in post_oos_labels or any(
            label.startswith(prefix) for prefix in post_oos_prefixes
        )
        if contaminated:
            row["post_oos_research_variant"] = True
            row["requires_fresh_forward_shadow"] = True
        uses_oos = bool(row.get("uses_locked_oos_for_selection", False))
        row["nested_hybrid_dependency"] = nested_hybrid_dependency
        row["clean_promotion_eligible"] = (
            not uses_oos
            and not bool(row.get("post_oos_research_variant", False))
            and not bool(row.get("requires_fresh_forward_shadow", False))
            and not nested_hybrid_dependency
        )
    return mutable_rows


def _recompute_payload_from_existing(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict[str, Any]:
    """Fast path: recompute clean flags, rankings, audits, and markdown inputs."""
    out = dict(payload)
    rows = _sanitize_research_dependency_flags(list(payload.get("fold_candidate_rows") or []))
    out["fold_candidate_rows"] = rows
    _refresh_payload_derived_reports(out)
    out["recomputed_from_existing_rows"] = True
    out["recompute_note"] = (
        "no strategy optimization rerun; clean/research dependency flags and aggregates recomputed"
    )
    if source_path is not None:
        resolved_source = source_path.expanduser().resolve()
        out["recompute_provenance"] = {
            "source_json_path": str(resolved_source),
            "source_json_sha256": _file_sha256(resolved_source),
            "recomputed_from_existing_rows": True,
            "fresh_optuna_rerun": False,
            "generated_at_utc": _utc_now_iso(),
            "output_paths": {
                "json": str(output_json.expanduser().resolve()) if output_json is not None else None,
                "markdown": str(output_md.expanduser().resolve()) if output_md is not None else None,
            },
            "interpretation": (
                "governance/ranking repair only; not a fresh no-nested Optuna search"
            ),
        }
    out["generated_at_utc"] = _utc_now_iso()
    return out


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):.2%}"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    aggregates = list(payload.get("aggregate_rankings") or [])
    clean_rankings = list(payload.get("clean_promotion_rankings") or []) or _clean_promotion_rankings(
        aggregates
    )
    demoted_rankings = list(
        payload.get("demoted_nested_or_historical_rankings") or []
    ) or _demoted_nested_or_historical_rankings(aggregates)
    fold_rows = list(payload.get("fold_candidate_rows") or [])
    folds = list(payload.get("folds") or [])
    top = aggregates[:12]
    clean_top = clean_rankings[:12]
    demoted_top = demoted_rankings[:12]
    provenance = dict(payload.get("recompute_provenance") or {})

    def append_ranking_table(rows: Sequence[Mapping[str, Any]]) -> None:
        lines.extend(
            [
                "| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |",
                "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for idx, row in enumerate(rows, start=1):
            reasons = ",".join(str(item) for item in (row.get("non_clean_reasons") or []))
            lines.append(
                f"| {idx} | `{row['candidate_label']}` | `{row['family']}` | "
                f"`{bool(row.get('clean_promotion_eligible'))}` | "
                f"`{reasons}` | "
                f"`{bool(row.get('hard_stop_promotable'))}` | "
                f"{_fmt_pct(row['compounded_oos_return'])} | "
                f"{row['positive_oos_folds']}/{row['fold_count']} | "
                f"{_fmt_pct(row['min_oos_return'])} | "
                f"{_fmt_pct(row['latest_oos_return'])} | "
                f"{_safe_float(row.get('monthly_sharpe_approx')):.2f} | "
                f"{_safe_float(row.get('monthly_sortino_approx')):.2f} | "
                f"{_fmt_pct(row['max_oos_mdd'])} |"
            )

    lines = [
        "# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS",
        "",
        f"- generated: `{payload.get('generated_at_utc')}`",
        f"- latest available data: `{payload.get('data_coverage', {}).get('global_latest_utc')}`",
        f"- allowed timeframes: `{', '.join(payload.get('timeframes') or [])}`",
        f"- slippage/cost proxy: `{payload.get('cost_model', {}).get('slippage_bps')}` bps",
        f"- folds: `{len(folds)}` (`{folds[0]['fold_id'] if folds else ''}` → `{folds[-1]['fold_id'] if folds else ''}`)",
        f"- trials: asset/profile/hybrid = `{payload.get('trial_policy', {}).get('asset_trials')}` / `{payload.get('trial_policy', {}).get('profile_trials')}` / `{payload.get('trial_policy', {}).get('hybrid_trials')}`",
        "- selection/refit input: train + 2M validation only; OOS month is evaluated after frozen fold params.",
    ]
    if provenance:
        lines.extend(
            [
                f"- recomputed from existing rows: `{bool(provenance.get('recomputed_from_existing_rows'))}`",
                f"- source JSON: `{provenance.get('source_json_path')}`",
                f"- source sha256: `{provenance.get('source_json_sha256')}`",
                f"- recompute interpretation: `{provenance.get('interpretation')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Fold schedule",
            "",
            "| Fold | Refit | Train | Validation | OOS |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for fold in folds:
        lines.append(
            f"| `{fold['fold_id']}` | `{fold['refit_at']}` | "
            f"`{fold['train']['start']} → {fold['train']['end']}` | "
            f"`{fold['validation']['start']} → {fold['validation']['end']}` | "
            f"`{fold['locked_oos']['start']} → {fold['locked_oos']['end']}` |"
        )
    lines.extend(
        [
            "",
            "## Raw aggregate ranking (diagnostic only)",
            "",
        ]
    )
    append_ranking_table(top)
    lines.extend(
        [
            "",
            "## Clean-promotion ranking (current recommendation set)",
            "",
        ]
    )
    append_ranking_table(clean_top)
    lines.extend(
        [
            "",
            "## Demoted nested/historical ranking",
            "",
            "These rows may remain useful diagnostics, but they are not current clean-promotion evidence.",
            "",
        ]
    )
    append_ranking_table(demoted_top)
    if clean_top or top:
        best_label = str((clean_top or top)[0]["candidate_label"])
        best_rows = [row for row in fold_rows if row.get("candidate_label") == best_label]
        lines.extend(
            [
                "",
                f"## Best clean candidate monthly OOS detail: `{best_label}`",
                "",
                "| Fold | Val | OOS | OOS MDD | Weights/source |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in best_rows:
            weights = json.dumps(_json_safe(row.get("final_weights") or {}), sort_keys=True)
            if len(weights) > 140:
                weights = weights[:137] + "..."
            lines.append(
                f"| `{row['fold_id']}` | "
                f"{_fmt_pct(row['validation']['total_return'])} | "
                f"{_fmt_pct(row['locked_oos']['total_return'])} | "
                f"{_fmt_pct(row['locked_oos']['mdd'])} | "
                f"`{row['source_profile_id']}` / `{weights}` |"
            )
        best_agg = next((row for row in aggregates if row.get("candidate_label") == best_label), {})
        lines.extend(
            [
                "",
                "### Best candidate extended metrics",
                "",
                f"- OOS comp: `{_fmt_pct(best_agg.get('compounded_oos_return'))}`",
                f"- hit rate: `{_safe_float(best_agg.get('positive_oos_folds')):.0f}/{_safe_float(best_agg.get('fold_count')):.0f}`",
                f"- monthly Sharpe / Sortino approx: `{_safe_float(best_agg.get('monthly_sharpe_approx')):.2f}` / `{_safe_float(best_agg.get('monthly_sortino_approx')):.2f}`",
                f"- 5% monthly VaR / 25% CVaR: `{_fmt_pct(best_agg.get('monthly_var_05'))}` / `{_fmt_pct(best_agg.get('monthly_cvar_25'))}`",
                f"- avg gain / avg loss: `{_fmt_pct(best_agg.get('avg_gain'))}` / `{_fmt_pct(best_agg.get('avg_loss'))}`",
                f"- gain/loss ratio: `{_safe_float(best_agg.get('gain_loss_ratio')):.2f}`",
                f"- max loss streak: `{int(_safe_float(best_agg.get('max_loss_streak'))):d}`",
                f"- mean/min validation: `{_fmt_pct(best_agg.get('mean_validation_return'))}` / `{_fmt_pct(best_agg.get('min_validation_return'))}`",
            ]
        )
    tf_summary = dict(payload.get("timeframe_coverage") or {})
    if tf_summary:
        lines.extend(
            [
                "",
                "## Timeframe coverage",
                "",
                "| Timeframe | Symbols with rows | Symbols skipped | Median rows | Latest |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for timeframe in payload.get("timeframes") or sorted(tf_summary):
            row = dict(tf_summary.get(str(timeframe)) or {})
            lines.append(
                f"| `{timeframe}` | {int(row.get('symbols_with_rows') or 0)} | "
                f"{int(row.get('symbols_without_rows') or 0)} | "
                f"{_safe_float(row.get('median_rows')):.1f} | `{row.get('latest')}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- This is still research/paper-testnet evidence, not real-money approval.",
            "- The latest OOS month can be partial when the data feed ends before month-end.",
            "- If a candidate has a negative validation fold or low OOS consistency, prefer shadow monitoring over allocation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _checkpoint_due(fold_idx: int, fold_count: int, interval: int) -> bool:
    if fold_idx >= fold_count:
        return False
    return int(interval) > 0 and fold_idx % int(interval) == 0


def run_walkforward(args: argparse.Namespace) -> dict[str, Any]:
    symbols = tuple(str(item).strip().upper() for item in str(args.symbols).split(",") if item.strip())
    timeframes = _validate_timeframes_30m_to_1d(
        str(item).strip() for item in str(args.timeframes).split(",") if item.strip()
    )
    slippage_bps = float(args.slippage_bps)
    if not math.isclose(slippage_bps, broad69.PRIMARY_ROUND_TRIP_COST_BPS, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"this runner is pinned to {broad69.PRIMARY_ROUND_TRIP_COST_BPS:g} bps; "
            f"got --slippage-bps={slippage_bps:g}"
        )
    bridge_manifest = _load_bridge_protocol_manifest(Path(args.bridge_protocol_manifest))
    data_root = Path(args.data_root).expanduser().resolve()
    print(f"[load] symbols={len(symbols)} timeframes={timeframes} data_root={data_root}", flush=True)
    bars, coverage = broad69.load_all_bars(symbols, data_root=data_root, timeframes=timeframes)
    timeframe_coverage = _timeframe_coverage_summary(coverage)
    latest_data = _coerce_ts(coverage["global_latest_utc"])
    folds = build_monthly_folds(
        train_start=_coerce_ts(args.train_start),
        first_oos_start=_coerce_ts(args.first_oos_start),
        latest_data=latest_data,
        bar_minutes=int(args.bar_minutes),
    )
    if args.max_folds is not None:
        folds = folds[-int(args.max_folds) :]
    if not folds:
        raise ValueError("no monthly folds generated")
    cache = profile69.FeatureCache(
        bars_by_symbol_tf=bars,
        symbols=symbols,
        timeframes=timeframes,
        _xsmom={},
        _anchor_returns={},
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    fold_candidate_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_69_asset_monthly_refit_walkforward",
        "generated_at_utc": _utc_now_iso(),
        "protocol": {
            "refit_frequency": "monthly",
            "refit_day": "calendar_month_day_1_utc",
            "validation_window": "previous_two_calendar_months",
            "locked_oos_window": "next_one_calendar_month",
            "train_window": "expanding_from_train_start_to_validation_start_minus_one_bar",
            "oos_used_for_selection": False,
        },
        "bridge_protocol_manifest": {
            "path": bridge_manifest.get("_path"),
            "sha256": bridge_manifest.get("_sha256"),
            "manifest_version": bridge_manifest.get("manifest_version"),
            "hard_no_leakage_rules": bridge_manifest.get("hard_no_leakage_rules", []),
            "promotion_thresholds": bridge_manifest.get("promotion_thresholds", {}),
        },
        "bridge_protocol_audit": {
            "manifest_frozen_before_bridge_evaluation": True,
            "post_oos_expansion_for_same_protocol": False,
            "current_fold_oos_used_for_bridge_weighting": False,
            "same_month_dynamic_self_feeding": False,
        },
        "protocol_freeze_report": _protocol_freeze_report(bridge_manifest),
        "online_weight_audit": _online_weight_audit([]),
        "dynamic_self_feed_audit": _dynamic_self_feed_audit([]),
        "metric_reconciliation": {"metrics_reconciled": True, "mismatches": [], "candidate_count": 0},
        "promotability": _promotability_decision({}),
        "trial_policy": {
            "asset_trials": int(args.asset_trials),
            "profile_trials": int(args.profile_trials),
            "hybrid_trials": int(args.hybrid_trials),
            "seed": int(args.seed),
        },
        "cost_model": {
            "slippage_bps": slippage_bps,
            "research_primary_round_trip_cost_bps": broad69.PRIMARY_ROUND_TRIP_COST_BPS,
            "return_per_turnover_threshold_bps": broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        },
        "universe": {"symbol_count": len(symbols), "symbols": list(symbols)},
        "timeframes": list(timeframes),
        "data_coverage": coverage,
        "timeframe_coverage": timeframe_coverage,
        "folds": [fold.as_payload() for fold in folds],
        "fold_summaries": fold_summaries,
        "fold_candidate_rows": fold_candidate_rows,
        "aggregate_rankings": [],
        "clean_promotion_rankings": [],
        "demoted_nested_or_historical_rankings": [],
        "recomputed_from_existing_rows": False,
        "output_paths": {"json": str(output_json), "markdown": str(output_md)},
    }
    _write_json(output_json, payload)
    output_md.write_text(_render_markdown(payload), "utf-8")
    bridge_prior_completed_utilities: dict[str, list[float]] = {}

    for fold_idx, fold in enumerate(folds, start=1):
        fold_seed = int(args.seed) + fold_idx * 1_000_000
        print(
            f"[fold {fold_idx}/{len(folds)}] {fold.fold_id} train={fold.train[0]}..{fold.train[1]} "
            f"val={fold.validation[0]}..{fold.validation[1]} oos={fold.locked_oos[0]}..{fold.locked_oos[1]}",
            flush=True,
        )
        source_candidates, source_aux = _run_source_profile_family(
            fold=fold,
            symbols=symbols,
            timeframes=timeframes,
            bars=bars,
            cache=cache,
            data_coverage=coverage,
            asset_trials=int(args.asset_trials),
            profile_trials=int(args.profile_trials),
            hybrid_trials=int(args.hybrid_trials),
            seed=fold_seed,
            allocation_fraction=float(args.allocation_fraction),
        )
        all_candidates = list(source_candidates)
        source_payload = source_aux["source_payload"]
        if "individual_robust" in args.families:
            individual_candidates, individual_aux = _run_individual_robust_family(
                fold=fold,
                candidate_streams=source_aux["individual_streams"],
                hybrid_trials=int(args.hybrid_trials),
                seed=fold_seed + 100_000,
            )
            all_candidates.extend(individual_candidates)
        else:
            individual_aux = {}
        if "asset_timeframe_leverage" in args.families:
            asset_tf_leverage_candidates, asset_tf_leverage_aux = (
                _run_asset_timeframe_leverage_family(
                    fold=fold,
                    candidate_streams=source_aux["individual_streams"],
                    hybrid_trials=int(args.hybrid_trials),
                    seed=fold_seed + 150_000,
                )
            )
            all_candidates.extend(asset_tf_leverage_candidates)
        else:
            asset_tf_leverage_candidates = []
            asset_tf_leverage_aux = {}
        if "strict_efficiency" in args.families:
            strict_candidates, strict_aux = _run_efficiency_family(
                fold=fold,
                family="strict_efficiency",
                source_payload=source_payload,
                bars=bars,
                symbols=symbols,
                timeframes=timeframes,
                cache=cache,
                profile_trials=int(args.profile_trials),
                hybrid_trials=int(args.hybrid_trials),
                seed=fold_seed + 200_000,
                relaxed=False,
            )
            all_candidates.extend(strict_candidates)
        else:
            strict_aux = {}
        if "relaxed_efficiency" in args.families:
            relaxed_candidates, relaxed_aux = _run_efficiency_family(
                fold=fold,
                family="relaxed_efficiency",
                source_payload=source_payload,
                bars=bars,
                symbols=symbols,
                timeframes=timeframes,
                cache=cache,
                profile_trials=int(args.profile_trials),
                hybrid_trials=int(args.hybrid_trials),
                seed=fold_seed + 400_000,
                relaxed=True,
            )
            all_candidates.extend(relaxed_candidates)
        else:
            relaxed_aux = {}
        meta_candidates = _meta_portfolio_candidates(all_candidates, fold)
        all_candidates.extend(meta_candidates)
        cross_hybrid_candidates = _cross_candidate_hybrid_candidates(
            all_candidates,
            fold,
            hybrid_trials=int(args.hybrid_trials),
            seed=fold_seed + 800_000,
        )
        all_candidates.extend(cross_hybrid_candidates)
        dynamic_switch_candidates = _dynamic_conviction_switch_candidates(all_candidates, fold)
        all_candidates.extend(dynamic_switch_candidates)
        dynamic_aware_hybrid_candidates = _dynamic_aware_hybrid_candidates(
            all_candidates,
            fold,
            hybrid_trials=int(args.hybrid_trials),
            seed=fold_seed + 900_000,
        )
        all_candidates.extend(dynamic_aware_hybrid_candidates)
        fixed_relaxed_dynamic_blend_candidates = _fixed_relaxed_dynamic_blend_candidates(
            all_candidates
        )
        all_candidates.extend(fixed_relaxed_dynamic_blend_candidates)
        risk_enhanced_blend_candidates = _fixed_risk_enhanced_blend_candidates(all_candidates)
        all_candidates.extend(risk_enhanced_blend_candidates)
        validation_selector_candidates = _validation_selector_candidates(all_candidates, fold)
        all_candidates.extend(validation_selector_candidates)
        mdd30_high_volatility_candidates = _mdd30_high_volatility_candidates(all_candidates, fold)
        all_candidates.extend(mdd30_high_volatility_candidates)
        bridge_candidates = _hybrid_assimilated_dynamic_candidates(
            all_candidates,
            fold,
            prior_completed_utilities=bridge_prior_completed_utilities,
            bridge_manifest=bridge_manifest,
        )
        all_candidates.extend(bridge_candidates)
        rows = [_evaluate_candidate(candidate, fold) for candidate in all_candidates]
        fold_candidate_rows.extend(rows)
        _update_bridge_prior_utilities(bridge_prior_completed_utilities, rows)
        best_fold = max(rows, key=lambda row: _safe_float(row["locked_oos"]["total_return"]))
        fold_summary = {
            "fold_id": fold.fold_id,
            "candidate_count": len(rows),
            "best_oos_candidate": best_fold["candidate_label"],
            "best_oos_return": best_fold["locked_oos"]["total_return"],
            "best_oos_mdd": best_fold["locked_oos"]["mdd"],
            "source_aux": {
                key: value
                for key, value in source_aux.items()
                if key not in {"source_payload", "profile_streams", "individual_streams"}
            },
            "individual_robust_aux": individual_aux,
            "asset_timeframe_leverage_aux": asset_tf_leverage_aux,
            "strict_efficiency_aux": strict_aux,
            "relaxed_efficiency_aux": relaxed_aux,
            "asset_timeframe_leverage_count": len(asset_tf_leverage_candidates),
            "meta_portfolio_candidate_count": len(meta_candidates),
            "cross_candidate_hybrid_count": len(cross_hybrid_candidates),
            "dynamic_conviction_switch_count": len(dynamic_switch_candidates),
            "dynamic_aware_hybrid_count": len(dynamic_aware_hybrid_candidates),
            "fixed_relaxed_dynamic_blend_count": len(fixed_relaxed_dynamic_blend_candidates),
            "risk_enhanced_blend_count": len(risk_enhanced_blend_candidates),
            "validation_selector_count": len(validation_selector_candidates),
            "mdd30_high_volatility_count": len(mdd30_high_volatility_candidates),
            "hybrid_oracle_bridge_count": len(bridge_candidates),
            "bridge_prior_utility_labels": sorted(bridge_prior_completed_utilities)[:50],
            "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        fold_summaries.append(fold_summary)
        _refresh_payload_derived_reports(payload)
        payload["generated_at_utc"] = _utc_now_iso()
        if _checkpoint_due(fold_idx, len(folds), int(args.checkpoint_interval)):
            _write_json(output_json, payload)
        if _checkpoint_due(fold_idx, len(folds), int(args.checkpoint_markdown_interval)):
            output_md.write_text(_render_markdown(payload), "utf-8")
        print(
            f"[fold {fold.fold_id}] best={best_fold['candidate_label']} "
            f"oos={_fmt_pct(best_fold['locked_oos']['total_return'])} "
            f"mdd={_fmt_pct(best_fold['locked_oos']['mdd'])}",
            flush=True,
        )

    payload["completed_at_utc"] = _utc_now_iso()
    payload["runner_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    _refresh_payload_derived_reports(payload)
    _write_json(output_json, payload)
    output_md.write_text(_render_markdown(payload), "utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(broad69.DEFAULT_DATA_ROOT))
    parser.add_argument("--symbols", default=",".join(BINANCE_EXTENDED_RESEARCH_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(ALLOWED_TIMEFRAMES_30M_TO_1D))
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--first-oos-start", default=DEFAULT_FIRST_OOS_START)
    parser.add_argument("--bar-minutes", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--allocation-fraction", type=float, default=profile69.DEFAULT_ALLOCATION_FRACTION)
    parser.add_argument("--asset-trials", type=int, default=DEFAULT_ASSET_TRIALS)
    parser.add_argument("--profile-trials", type=int, default=DEFAULT_PROFILE_TRIALS)
    parser.add_argument("--hybrid-trials", type=int, default=DEFAULT_HYBRID_TRIALS)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument(
        "--families",
        default=(
            "profile_optuna,individual_robust,asset_timeframe_leverage,"
            "strict_efficiency,relaxed_efficiency"
        ),
        help=(
            "Comma-separated families. profile_optuna is always included; optional: "
            "individual_robust, asset_timeframe_leverage, strict_efficiency, relaxed_efficiency."
        ),
    )
    parser.add_argument("--bridge-protocol-manifest", default=str(DEFAULT_BRIDGE_PROTOCOL_MANIFEST))
    parser.add_argument(
        "--recompute-from-json",
        default=None,
        help=(
            "Fast path: load an existing walk-forward JSON and recompute clean/research "
            "dependency flags plus aggregate reports without rerunning Optuna."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help=(
            "Write JSON checkpoint every N folds during full reruns. Use 0 to write only "
            "initial/final artifacts; default preserves fold-level recovery."
        ),
    )
    parser.add_argument(
        "--checkpoint-markdown-interval",
        type=int,
        default=0,
        help=(
            "Render markdown checkpoint every N folds during full reruns. Default 0 skips "
            "expensive growing markdown renders until final output."
        ),
    )
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args(argv)
    families = {item.strip() for item in str(args.families).split(",") if item.strip()}
    families.add("profile_optuna")
    args.families = tuple(sorted(families))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    if args.recompute_from_json:
        source_path = Path(args.recompute_from_json).expanduser().resolve()
        payload = _recompute_payload_from_existing(
            json.loads(source_path.read_text("utf-8")),
            source_path=source_path,
            output_json=output_json,
            output_md=output_md,
        )
        payload["output_paths"] = {
            "json": str(output_json),
            "markdown": str(output_md),
        }
        if "recompute_provenance" in payload:
            payload["recompute_provenance"]["output_paths"] = dict(payload["output_paths"])
        _write_json(output_json, payload)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(payload), "utf-8")
    else:
        payload = run_walkforward(args)
    top = list(payload.get("aggregate_rankings") or [])[:5]
    clean_top = list(payload.get("clean_promotion_rankings") or [])[:5]
    demoted_top = list(payload.get("demoted_nested_or_historical_rankings") or [])[:5]
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "fold_count": len(payload["folds"]),
                    "latest_available_data": payload["data_coverage"]["global_latest_utc"],
                    "top_5": top,
                    "clean_top_5": clean_top,
                    "demoted_top_5": demoted_top,
                    "runner_peak_rss_mib": payload.get("runner_peak_rss_mib"),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
