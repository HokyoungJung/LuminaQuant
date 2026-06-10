#!/usr/bin/env python3
"""Build strategy-PnL correlation matrices and a correlation-aware paper slate.

This runner is research/paper-testnet only. It replays already discovered paper
candidates to capture per-bar PnL return streams, computes correlation matrices
from train/validation data for portfolio de-duplication, and keeps locked-OOS as
report-only evidence after the train+validation correlation decision is frozen.
It never executes orders and never enables real money.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import resource
import sys
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_alpha_zoo_htf_momentum_crowding_discovery import (  # noqa: E402
    AVG_BBO_SPREAD_BPS_ASSUMPTION,
    BBO_SPREAD_MULTIPLIER,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEATURE_ROOT,
    PRIMARY_ROUND_TRIP_COST_BPS,
    RETURN_PER_TURNOVER_THRESHOLD_BPS,
    SPLIT_ORDER,
    SimResult,
    _json_safe,
    _split_mask,
    max_drawdown,
)

DEFAULT_MONITORING_ARTIFACT = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_multi_asset_monitoring_slate_20260524/multi_asset_monitoring_slate_latest.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_pnl_correlation_decision_20260524"
)

ARTIFACT_KIND = "alpha_zoo_pnl_correlation_decision"
PAPER_STATUS = "paper_testnet_monitor"
STRICT_SHADOW_STATUSES = {"shadow_watchlist_no_promotion", "coverage_blocked_shadow"}
CORRELATION_METHOD = "pearson_per_bar_pnl_returns_aligned_by_datetime_missing_bars_filled_zero"
PRIMARY_CORR_SURFACE = "train_validation"
MAX_TRAIN_VALIDATION_ABS_CORR = 0.70
MAX_VALIDATION_ABS_CORR = 0.75
HIGH_CORR_CLUSTER_THRESHOLD = 0.85
REPRESENTATIVE_MATRIX_LIMIT = 30

SOURCE_KIND_DEBOUNCED_REPAIR = "alpha_zoo_debounced_efficiency_repair_discovery"
SOURCE_KIND_FEEDBACK = "alpha_zoo_30m_plus_alpha_feedback_discovery"
SOURCE_KIND_BOOSTER = "alpha_zoo_30m_plus_alpha_booster_discovery"
SOURCE_KIND_ASSET_DIVERSE = "alpha_zoo_asset_diverse_strategy_discovery"
SOURCE_KINDS = (
    SOURCE_KIND_DEBOUNCED_REPAIR,
    SOURCE_KIND_FEEDBACK,
    SOURCE_KIND_BOOSTER,
    SOURCE_KIND_ASSET_DIVERSE,
)

EXTERNAL_METHOD_REFERENCES = [
    {
        "label": "Markowitz (1952), Portfolio Selection",
        "url": "https://ideas.repec.org/a/bla/jfinan/v7y1952i1p77-91.html",
        "usage": "Mean-variance portfolio framing: covariance/correlation of returns matters for diversification risk.",
    },
    {
        "label": "Markowitz Portfolio Construction at Seventy",
        "url": "https://stanford.edu/~boyd/papers/pdf/markowitz.pdf",
        "usage": "Modern discussion of portfolio construction and covariance-aware diversification; not used as a trading signal.",
    },
]

CSV_FIELDS = [
    "selection_rank",
    "model_id",
    "source_artifact_kind",
    "symbol",
    "timeframe",
    "family",
    "side",
    "notional_fraction",
    "monitoring_score_train_validation_only",
    "train_return",
    "validation_return",
    "locked_oos_return",
    "validation_mdd",
    "train_trade_event_count",
    "validation_trade_event_count",
    "locked_oos_trade_event_count",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "locked_oos_return_per_turnover_proxy_bps",
    "max_abs_corr_to_prior_train_validation",
    "max_abs_corr_to_prior_validation",
    "nearest_prior_model_id_train_validation",
    "correlation_decision",
    "correlation_rejection_reasons",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
]


@dataclass(frozen=True)
class CapturedPnl:
    model_id: str
    source_artifact_kind: str
    datetimes: tuple[pd.Timestamp, ...]
    returns: np.ndarray
    position: np.ndarray
    timeframe: str


@dataclass(frozen=True)
class SourceReplayRequest:
    source_kind: str
    rows: tuple[dict[str, Any], ...]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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
        return ";".join(str(v) for v in value)
    return _json_safe(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_matrix_csv(path: Path, matrix: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, index=True, lineterminator="\n")


def _period_return(returns: np.ndarray | pd.Series) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except TypeError, ValueError:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _monitoring_score_train_validation_only(row: Mapping[str, Any]) -> float:
    """Train/validation-only ordering score; locked-OOS is intentionally absent."""
    explicit = row.get("monitoring_score_train_validation_only")
    if explicit is not None:
        return _safe_float(explicit)
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    validation_mdd = _safe_float(row.get("validation_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"))
    validation_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"))
    validation_spike_penalty = max(0.0, validation - train)
    return (
        7.0 * validation
        + 2.0 * min(train, validation)
        + min(train_rpt, 70.0) / 260.0
        + min(validation_rpt, 70.0) / 180.0
        - 8.0 * validation_spike_penalty
        - 2.25 * validation_mdd
    )


def _load_monitoring_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("ready_for_real") is not False
        or payload.get("real_money_execution") is not False
    ):
        raise ValueError(f"real-money guard violation in monitoring artifact: {path}")
    if payload.get("selection_policy", {}).get("uses_locked_oos_for_discovery") is not False:
        raise ValueError("monitoring artifact does not fail closed on locked-OOS discovery usage")
    return payload


def _dedupe_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_id = str(row.get("model_id") or "")
        if not model_id:
            continue
        if model_id not in by_id:
            by_id[model_id] = dict(row)
    return list(by_id.values())


def _paper_rows_from_monitoring(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _dedupe_rows(
        row
        for row in payload.get("monitoring_rows", [])
        if row.get("monitoring_status") == PAPER_STATUS
    )
    return sorted(rows, key=_monitoring_score_train_validation_only, reverse=True)


def _strict_shadow_rows_from_monitoring(
    payload: Mapping[str, Any], *, limit: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    shadows: list[dict[str, Any]] = []
    for row in payload.get("monitoring_rows", []):
        if row.get("monitoring_status") not in STRICT_SHADOW_STATUSES:
            continue
        if _safe_float(row.get("train_return")) < _safe_float(row.get("validation_return")):
            continue
        if _safe_float(row.get("validation_return")) < 0.05:
            continue
        if _safe_float(row.get("locked_oos_return")) < 0.02:
            continue
        if (
            _safe_float(row.get("train_return_per_turnover_proxy_bps"))
            <= RETURN_PER_TURNOVER_THRESHOLD_BPS
        ):
            continue
        if (
            _safe_float(row.get("validation_return_per_turnover_proxy_bps"))
            <= RETURN_PER_TURNOVER_THRESHOLD_BPS
        ):
            continue
        if (
            _safe_float(row.get("locked_oos_return_per_turnover_proxy_bps"))
            <= RETURN_PER_TURNOVER_THRESHOLD_BPS
        ):
            continue
        shadows.append(dict(row))
    return sorted(_dedupe_rows(shadows), key=_monitoring_score_train_validation_only, reverse=True)[
        :limit
    ]


def _capture_finalizer(
    *,
    source_kind: str,
    target_ids: set[str],
    captures: dict[str, CapturedPnl],
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def wrapped(
        base: dict[str, Any], sim: SimResult, datetimes: pd.Series, *, timeframe: str
    ) -> dict[str, Any]:
        row = original(base, sim, datetimes, timeframe=timeframe)
        model_id = str(row.get("model_id") or base.get("model_id") or "")
        if model_id in target_ids and model_id not in captures:
            captures[model_id] = CapturedPnl(
                model_id=model_id,
                source_artifact_kind=source_kind,
                datetimes=tuple(pd.to_datetime(pd.Series(datetimes)).tolist()),
                returns=np.asarray(sim.returns, dtype=float).copy(),
                position=np.asarray(sim.position, dtype=float).copy(),
                timeframe=timeframe,
            )
        return row

    return wrapped


def _group_requests(rows: Sequence[Mapping[str, Any]]) -> dict[str, SourceReplayRequest]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_kind = str(row.get("source_artifact_kind") or "")
        if source_kind in SOURCE_KINDS:
            grouped[source_kind].append(dict(row))
    return {
        kind: SourceReplayRequest(source_kind=kind, rows=tuple(rows_))
        for kind, rows_ in grouped.items()
    }


def _target_symbols(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted({str(row.get("symbol") or "").upper() for row in rows if row.get("symbol")})
    )


def _target_timeframes(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted({str(row.get("timeframe") or "").lower() for row in rows if row.get("timeframe")})
    )


def _capture_debounced_repair(
    request: SourceReplayRequest,
    *,
    data_root: Path,
    captures: dict[str, CapturedPnl],
) -> None:
    from scripts.research import run_alpha_zoo_debounced_efficiency_repair_discovery as repair

    target_ids = {str(row["model_id"]) for row in request.rows}
    symbols = _target_symbols(request.rows)
    timeframes = _target_timeframes(request.rows)
    data_symbols = tuple(dict.fromkeys((repair.BTC_REGIME_SYMBOL, *symbols)))
    hourly = repair.load_hourly_bars(data_symbols, data_root=data_root)
    bars_by_symbol_tf = {
        (symbol, timeframe): repair.resample_bars(hourly[symbol], timeframe)
        for timeframe in timeframes
        for symbol in data_symbols
    }
    original = repair._finalize_candidate
    repair._finalize_candidate = _capture_finalizer(
        source_kind=request.source_kind,
        target_ids=target_ids,
        captures=captures,
        original=original,
    )
    try:
        repair.discover_repair_candidates(bars_by_symbol_tf, symbols=symbols, timeframes=timeframes)
    finally:
        repair._finalize_candidate = original
    del hourly, bars_by_symbol_tf
    gc.collect()


def _capture_feedback(
    request: SourceReplayRequest,
    *,
    data_root: Path,
    feature_root: Path,
    captures: dict[str, CapturedPnl],
) -> None:
    from scripts.research import run_alpha_zoo_30m_plus_alpha_feedback_discovery as feedback

    target_ids = {str(row["model_id"]) for row in request.rows}
    symbols = _target_symbols(request.rows)
    timeframes = feedback._validate_timeframes(_target_timeframes(request.rows))
    data_symbols = tuple(dict.fromkeys((feedback.BTC_REGIME_SYMBOL, *symbols)))
    bars_by_symbol_tf = feedback.load_requested_bars(
        data_symbols, timeframes=timeframes, data_root=data_root
    )
    for symbol in data_symbols:
        features = feedback.load_feature_points(symbol, feature_root=feature_root)
        for timeframe in timeframes:
            bars_by_symbol_tf[(symbol, timeframe)] = feedback._attach_features_with_age(
                bars_by_symbol_tf[(symbol, timeframe)],
                features,
                timeframe=timeframe,
            )
    original = feedback._finalize_candidate
    feedback._finalize_candidate = _capture_finalizer(
        source_kind=request.source_kind,
        target_ids=target_ids,
        captures=captures,
        original=original,
    )
    try:
        feedback.discover_candidates(bars_by_symbol_tf, symbols=symbols, timeframes=timeframes)
    finally:
        feedback._finalize_candidate = original
    del bars_by_symbol_tf
    gc.collect()


def _capture_booster(
    request: SourceReplayRequest,
    *,
    data_root: Path,
    feature_root: Path,
    captures: dict[str, CapturedPnl],
) -> None:
    from scripts.research import run_alpha_zoo_30m_plus_alpha_booster_discovery as booster
    from scripts.research import run_alpha_zoo_30m_plus_alpha_feedback_discovery as feedback

    target_ids = {str(row["model_id"]) for row in request.rows}
    symbols = _target_symbols(request.rows)
    timeframes = feedback._validate_timeframes(_target_timeframes(request.rows))
    data_symbols = tuple(dict.fromkeys((feedback.BTC_REGIME_SYMBOL, *symbols)))
    bars_by_symbol_tf = feedback.load_requested_bars(
        data_symbols, timeframes=timeframes, data_root=data_root
    )
    for symbol in data_symbols:
        features = feedback.load_feature_points(symbol, feature_root=feature_root)
        for timeframe in timeframes:
            bars_by_symbol_tf[(symbol, timeframe)] = feedback._attach_features_with_age(
                bars_by_symbol_tf[(symbol, timeframe)],
                features,
                timeframe=timeframe,
            )
    original = booster._finalize_booster_candidate
    booster._finalize_booster_candidate = _capture_finalizer(
        source_kind=request.source_kind,
        target_ids=target_ids,
        captures=captures,
        original=original,
    )
    try:
        booster.discover_booster_candidates(
            bars_by_symbol_tf, symbols=symbols, timeframes=timeframes
        )
    finally:
        booster._finalize_booster_candidate = original
    del bars_by_symbol_tf
    gc.collect()


def _source_symbols_from_payload(
    payload: Mapping[str, Any], source_kind: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    for source in payload.get("source_artifacts", []):
        if source.get("artifact_kind") != source_kind:
            continue
        source_path = Path(str(source.get("path"))).expanduser()
        if source_path.exists():
            source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            source_data = source_payload.get("source_data") or {}
            promotion = tuple(
                source_data.get("promotion_symbols")
                or source_data.get("symbols")
                or source.get("source_symbols")
                or ()
            )
            shadow = tuple(source_data.get("shadow_symbols") or ())
            return tuple(str(item).upper() for item in promotion), tuple(
                str(item).upper() for item in shadow
            )
    return (), ()


def _capture_asset_diverse(
    request: SourceReplayRequest,
    *,
    data_root: Path,
    monitoring_payload: Mapping[str, Any],
    captures: dict[str, CapturedPnl],
) -> None:
    from scripts.research import run_alpha_zoo_asset_diverse_strategy_discovery as asset
    from scripts.research import run_alpha_zoo_30m_plus_alpha_feedback_discovery as feedback

    target_ids = {str(row["model_id"]) for row in request.rows}
    target_timeframes = feedback._validate_timeframes(_target_timeframes(request.rows))
    promotion_symbols, shadow_symbols = _source_symbols_from_payload(
        monitoring_payload, request.source_kind
    )
    if not promotion_symbols:
        promotion_symbols = _target_symbols(request.rows)
    all_symbols = tuple(dict.fromkeys((*promotion_symbols, *shadow_symbols)))
    bars_by_symbol_tf = feedback.load_requested_bars(
        all_symbols, timeframes=target_timeframes, data_root=data_root
    )
    original = asset._finalize_asset_candidate
    asset._finalize_asset_candidate = _capture_finalizer(
        source_kind=request.source_kind,
        target_ids=target_ids,
        captures=captures,
        original=original,
    )
    try:
        asset.discover_asset_diverse_candidates(
            bars_by_symbol_tf,
            symbols=promotion_symbols,
            shadow_symbols=shadow_symbols,
            timeframes=target_timeframes,
        )
    finally:
        asset._finalize_asset_candidate = original
    del bars_by_symbol_tf
    gc.collect()


def capture_pnl_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    feature_root: Path,
    monitoring_payload: Mapping[str, Any],
) -> dict[str, CapturedPnl]:
    captures: dict[str, CapturedPnl] = {}
    requests = _group_requests(rows)
    if SOURCE_KIND_DEBOUNCED_REPAIR in requests:
        _capture_debounced_repair(
            requests[SOURCE_KIND_DEBOUNCED_REPAIR], data_root=data_root, captures=captures
        )
    if SOURCE_KIND_FEEDBACK in requests:
        _capture_feedback(
            requests[SOURCE_KIND_FEEDBACK],
            data_root=data_root,
            feature_root=feature_root,
            captures=captures,
        )
    if SOURCE_KIND_BOOSTER in requests:
        _capture_booster(
            requests[SOURCE_KIND_BOOSTER],
            data_root=data_root,
            feature_root=feature_root,
            captures=captures,
        )
    if SOURCE_KIND_ASSET_DIVERSE in requests:
        _capture_asset_diverse(
            requests[SOURCE_KIND_ASSET_DIVERSE],
            data_root=data_root,
            monitoring_payload=monitoring_payload,
            captures=captures,
        )
    return captures


def _series_for_split(capture: CapturedPnl, split: str) -> pd.Series:
    dt = pd.DatetimeIndex(capture.datetimes)
    if split == "train_validation":
        train_mask = _split_mask(dt, "train")
        validation_mask = _split_mask(dt, "validation")
        mask = train_mask | validation_mask
    else:
        mask = _split_mask(dt, split)
    return pd.Series(capture.returns[mask], index=dt[mask], name=capture.model_id, dtype=float)


def _aligned_pnl_frame(
    captures: Mapping[str, CapturedPnl], model_ids: Sequence[str], *, split: str
) -> pd.DataFrame:
    series = [
        _series_for_split(captures[model_id], split)
        for model_id in model_ids
        if model_id in captures
    ]
    if not series:
        return pd.DataFrame()
    frame = pd.concat(series, axis=1).sort_index().fillna(0.0)
    return frame.loc[:, list(dict.fromkeys(s.name for s in series if s.name is not None))]


def _corr_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    corr = frame.corr(method="pearson").fillna(0.0)
    for idx in corr.index:
        corr.loc[idx, idx] = 1.0
    return corr.clip(lower=-1.0, upper=1.0)


def _matrix_pair_stats(corr: pd.DataFrame) -> dict[str, Any]:
    if corr.empty or len(corr.columns) < 2:
        return {
            "candidate_count": len(corr.columns),
            "pair_count": 0,
            "mean_pairwise_corr": 0.0,
            "mean_pairwise_abs_corr": 0.0,
            "max_pairwise_abs_corr": 0.0,
            "high_corr_pair_count_ge_0p85_abs": 0,
        }
    arr = corr.to_numpy(dtype=float)
    upper = arr[np.triu_indices_from(arr, k=1)]
    return {
        "candidate_count": len(corr.columns),
        "pair_count": int(upper.size),
        "mean_pairwise_corr": float(np.mean(upper)) if upper.size else 0.0,
        "mean_pairwise_abs_corr": float(np.mean(np.abs(upper))) if upper.size else 0.0,
        "max_pairwise_abs_corr": float(np.max(np.abs(upper))) if upper.size else 0.0,
        "high_corr_pair_count_ge_0p85_abs": int(
            np.count_nonzero(np.abs(upper) >= HIGH_CORR_CLUSTER_THRESHOLD)
        ),
    }


def _top_abs_corr_pairs(corr: pd.DataFrame, *, limit: int = 25) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = float(corr.loc[left, right])
            pairs.append(
                {
                    "left_model_id": left,
                    "right_model_id": right,
                    "corr": value,
                    "abs_corr": abs(value),
                }
            )
    return sorted(pairs, key=lambda row: row["abs_corr"], reverse=True)[:limit]


def _correlation_clusters(corr: pd.DataFrame, *, threshold: float) -> list[list[str]]:
    if corr.empty:
        return []
    columns = list(corr.columns)
    adjacency: dict[str, set[str]] = {col: set() for col in columns}
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            if abs(float(corr.loc[left, right])) >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    visited: set[str] = set()
    clusters: list[list[str]] = []
    for start in columns:
        if start in visited:
            continue
        queue: deque[str] = deque([start])
        visited.add(start)
        cluster: list[str] = []
        while queue:
            node = queue.popleft()
            cluster.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append(cluster)
    return sorted(clusters, key=lambda item: (-len(item), item[0]))


def _nearest_selected_corr(
    candidate_id: str, selected_ids: Sequence[str], corr: pd.DataFrame
) -> tuple[float, str | None]:
    if not selected_ids or candidate_id not in corr.index:
        return 0.0, None
    best_abs = 0.0
    best_id: str | None = None
    for selected_id in selected_ids:
        if selected_id not in corr.columns:
            continue
        value = abs(float(corr.loc[candidate_id, selected_id]))
        if value > best_abs:
            best_abs = value
            best_id = selected_id
    return best_abs, best_id


def greedy_correlation_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_validation_corr: pd.DataFrame,
    validation_corr: pd.DataFrame,
    max_train_validation_abs_corr: float = MAX_TRAIN_VALIDATION_ABS_CORR,
    max_validation_abs_corr: float = MAX_VALIDATION_ABS_CORR,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows), key=_monitoring_score_train_validation_only, reverse=True
    )
    selected_ids: list[str] = []
    decision_rows: list[dict[str, Any]] = []
    for row in ordered:
        model_id = str(row["model_id"])
        tv_abs, tv_nearest = _nearest_selected_corr(model_id, selected_ids, train_validation_corr)
        val_abs, _ = _nearest_selected_corr(model_id, selected_ids, validation_corr)
        reasons: list[str] = []
        if tv_abs > max_train_validation_abs_corr:
            reasons.append(
                f"train_validation_abs_corr_{tv_abs:.4f}_above_{max_train_validation_abs_corr:.2f}_to_{tv_nearest}"
            )
        if val_abs > max_validation_abs_corr:
            reasons.append(f"validation_abs_corr_{val_abs:.4f}_above_{max_validation_abs_corr:.2f}")
        accepted = not reasons
        if accepted:
            selected_ids.append(model_id)
        row.update(
            {
                "selection_rank": len(selected_ids) if accepted else None,
                "max_abs_corr_to_prior_train_validation": tv_abs,
                "max_abs_corr_to_prior_validation": val_abs,
                "nearest_prior_model_id_train_validation": tv_nearest,
                "correlation_decision": "selected_corr_diversified_paper_monitor"
                if accepted
                else "rejected_high_pnl_correlation_duplicate",
                "correlation_rejection_reasons": reasons,
                "ready_for_paper": bool(row.get("ready_for_paper")) and accepted,
                "ready_for_real": False,
                "real_money_execution": False,
            }
        )
        decision_rows.append(row)
    selected_rows = [
        row
        for row in decision_rows
        if row["correlation_decision"] == "selected_corr_diversified_paper_monitor"
    ]
    for idx, row in enumerate(selected_rows, start=1):
        row["selection_rank"] = idx
    return decision_rows


def _portfolio_series(frame: pd.DataFrame, model_ids: Sequence[str], *, mode: str) -> pd.Series:
    ids = [model_id for model_id in model_ids if model_id in frame.columns]
    if not ids:
        return pd.Series(dtype=float)
    sub = frame[ids]
    if mode == "equal_weight_mean":
        return sub.mean(axis=1)
    if mode == "unscaled_sum":
        return sub.sum(axis=1)
    raise ValueError(f"unsupported portfolio mode {mode!r}")


def _portfolio_stats(
    frames_by_split: Mapping[str, pd.DataFrame],
    model_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gross_notional = sum(
        _safe_float(rows_by_id[model_id].get("notional_fraction"))
        for model_id in model_ids
        if model_id in rows_by_id
    )
    out: dict[str, Any] = {
        "candidate_count": len(model_ids),
        "gross_notional_fraction_unscaled_sum": gross_notional,
        "mode_notes": {
            "equal_weight_mean": "mean of strategy PnL return streams; comparison-normalized and not an execution allocation",
            "unscaled_sum": "sum of each candidate's native notional PnL stream; unsafe if gross notional is large",
        },
        "splits": {},
    }
    for split, frame in frames_by_split.items():
        split_stats: dict[str, Any] = {}
        for mode in ("equal_weight_mean", "unscaled_sum"):
            series = _portfolio_series(frame, model_ids, mode=mode)
            split_stats[mode] = {
                "total_return": _period_return(series),
                "max_drawdown": max_drawdown(series.to_numpy(dtype=float))
                if not series.empty
                else 0.0,
                "bar_count": int(series.size),
            }
        out["splits"][split] = split_stats
    return out


def _candidate_label(row: Mapping[str, Any]) -> str:
    return f"{row.get('symbol')} {row.get('timeframe')} {row.get('family')}"


def _selected_summary_rows(decision_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in decision_rows
        if row.get("correlation_decision") == "selected_corr_diversified_paper_monitor"
    ]
    return [
        {
            "selection_rank": row.get("selection_rank"),
            "model_id": row.get("model_id"),
            "label": _candidate_label(row),
            "source_artifact_kind": row.get("source_artifact_kind"),
            "monitoring_score_train_validation_only": _monitoring_score_train_validation_only(row),
            "train_return": row.get("train_return"),
            "validation_return": row.get("validation_return"),
            "locked_oos_return_report_only": row.get("locked_oos_return"),
            "max_abs_corr_to_prior_train_validation": row.get(
                "max_abs_corr_to_prior_train_validation"
            ),
            "max_abs_corr_to_prior_validation": row.get("max_abs_corr_to_prior_validation"),
            "ready_for_paper": row.get("ready_for_paper"),
            "ready_for_real": False,
            "real_money_execution": False,
        }
        for row in selected
    ]


def _matrix_subset(corr: pd.DataFrame, model_ids: Sequence[str], *, limit: int) -> pd.DataFrame:
    ids = [model_id for model_id in model_ids if model_id in corr.columns][:limit]
    if not ids:
        return pd.DataFrame()
    return corr.loc[ids, ids]


def _small_matrix_markdown(
    corr: pd.DataFrame, rows_by_id: Mapping[str, Mapping[str, Any]], *, limit: int = 12
) -> str:
    if corr.empty:
        return "_No matrix available._\n"
    ids = list(corr.columns)[:limit]
    aliases = {model_id: f"S{idx + 1}" for idx, model_id in enumerate(ids)}
    lines = ["| ID | Strategy |", "| --- | --- |"]
    for model_id in ids:
        lines.append(
            f"| {aliases[model_id]} | `{model_id}` {_candidate_label(rows_by_id.get(model_id, {}))} |"
        )
    lines.extend(
        [
            "",
            "| | " + " | ".join(aliases[model_id] for model_id in ids) + " |",
            "| --- | " + " | ".join("---:" for _ in ids) + " |",
        ]
    )
    for left in ids:
        values = " | ".join(f"{float(corr.loc[left, right]):.3f}" for right in ids)
        lines.append(f"| {aliases[left]} | {values} |")
    if len(corr.columns) > limit:
        lines.append(
            f"\n_Showing first {limit} of {len(corr.columns)} IDs; full matrix is in CSV artifacts._"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(
    payload: Mapping[str, Any],
    *,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    selected_corr: pd.DataFrame,
) -> str:
    summary = payload["correlation_decision_summary"]
    selected = payload["selected_corr_diversified_candidates"]
    portfolio = payload["portfolio_comparison"]
    lines = [
        "# Alpha Zoo PnL Correlation Decision",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Decision method",
        "",
        "- Compute per-strategy PnL return streams from replayed paper/testnet candidates.",
        "- Align timestamps across 30m/1h/2h/4h/6h strategies; missing bars are filled with 0 PnL before Pearson correlation.",
        "- Rank only by train+validation monitoring score; locked-OOS is report-only after selection freeze.",
        f"- Greedy-select candidates if max abs train+validation corr <= {MAX_TRAIN_VALIDATION_ABS_CORR:.2f} and max abs validation corr <= {MAX_VALIDATION_ABS_CORR:.2f} versus already selected candidates.",
        "- Keep `ready_for_real=false` and `real_money_execution=false` for every artifact and candidate.",
        "",
        "## Summary",
        "",
        f"- Paper universe replayed: {summary['paper_universe_candidate_count']} candidates",
        f"- PnL capture count: {summary['captured_pnl_candidate_count']} candidates; missing: {len(summary['missing_pnl_model_ids'])}",
        f"- Corr-diversified selected candidates: {summary['selected_candidate_count']}",
        f"- High-correlation clusters at |corr| >= {HIGH_CORR_CLUSTER_THRESHOLD:.2f}: {summary['high_corr_cluster_count']}",
        f"- Decision: **{summary['decision']}**",
        "",
        "## Selected corr-diversified paper/testnet-only slate",
        "",
        "| Rank | Strategy | Train | Val | OOS report-only | Max train+val corr to prior | Max val corr to prior |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not selected:
        lines.append("| - | - | - | - | - | - | - |")
    for row in selected:
        model_id = row["model_id"]
        label = f"`{model_id}` {row['label']}"
        lines.append(
            f"| {row['selection_rank']} | {label} | "
            f"{_safe_float(row.get('train_return')):.4%} | "
            f"{_safe_float(row.get('validation_return')):.4%} | "
            f"{_safe_float(row.get('locked_oos_return_report_only')):.4%} | "
            f"{_safe_float(row.get('max_abs_corr_to_prior_train_validation')):.3f} | "
            f"{_safe_float(row.get('max_abs_corr_to_prior_validation')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Portfolio comparison",
            "",
            "| Portfolio | Count | Gross notional unscaled | Val equal-weight return | OOS equal-weight return | Val unscaled return | OOS unscaled return |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, stats in portfolio.items():
        val = stats["splits"].get("validation", {})
        oos = stats["splits"].get("locked_oos", {})
        lines.append(
            f"| {name} | {stats['candidate_count']} | {stats['gross_notional_fraction_unscaled_sum']:.2f}x | "
            f"{_safe_float(val.get('equal_weight_mean', {}).get('total_return')):.4%} | "
            f"{_safe_float(oos.get('equal_weight_mean', {}).get('total_return')):.4%} | "
            f"{_safe_float(val.get('unscaled_sum', {}).get('total_return')):.4%} | "
            f"{_safe_float(oos.get('unscaled_sum', {}).get('total_return')):.4%} |"
        )
    lines.extend(
        [
            "",
            "## Selected train+validation correlation matrix (excerpt)",
            "",
            _small_matrix_markdown(selected_corr, rows_by_id=rows_by_id),
            "## Top absolute train+validation correlation pairs",
            "",
            "| Abs corr | Corr | Left | Right |",
            "| ---: | ---: | --- | --- |",
        ]
    )
    for pair in payload["correlation_diagnostics"]["top_abs_corr_pairs_train_validation"][:15]:
        lines.append(
            f"| {pair['abs_corr']:.4f} | {pair['corr']:.4f} | `{pair['left_model_id']}` | `{pair['right_model_id']}` |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            f"- ready_for_real={str(payload['ready_for_real']).lower()}",
            f"- real_money_execution={str(payload['real_money_execution']).lower()}",
            f"- locked-OOS used for selection={str(payload['selection_policy']['uses_locked_oos_for_selection']).lower()}",
            f"- locked-OOS role: {payload['selection_policy']['locked_oos_role']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload_from_monitoring(
    monitoring_payload: Mapping[str, Any],
    *,
    output_dir: Path,
    monitoring_artifact_path: Path,
    data_root: Path,
    feature_root: Path,
    shadow_matrix_limit: int = REPRESENTATIVE_MATRIX_LIMIT,
    write_outputs: bool = True,
) -> dict[str, Any]:
    paper_rows = _paper_rows_from_monitoring(monitoring_payload)
    strict_shadow_rows = _strict_shadow_rows_from_monitoring(
        monitoring_payload, limit=shadow_matrix_limit
    )
    replay_rows = _dedupe_rows([*paper_rows, *strict_shadow_rows])
    captures = capture_pnl_series(
        replay_rows,
        data_root=data_root,
        feature_root=feature_root,
        monitoring_payload=monitoring_payload,
    )
    rows_by_id = {str(row["model_id"]): row for row in replay_rows}
    paper_ids = [str(row["model_id"]) for row in paper_rows if str(row.get("model_id")) in captures]
    shadow_ids = [
        str(row["model_id"]) for row in strict_shadow_rows if str(row.get("model_id")) in captures
    ]
    split_frames = {
        split: _aligned_pnl_frame(captures, paper_ids, split=split)
        for split in (*SPLIT_ORDER, "train_validation")
    }
    corr_by_split = {split: _corr_matrix(frame) for split, frame in split_frames.items()}
    decision_rows = greedy_correlation_selection(
        [rows_by_id[model_id] for model_id in paper_ids],
        train_validation_corr=corr_by_split["train_validation"],
        validation_corr=corr_by_split["validation"],
    )
    selected_ids = [
        str(row["model_id"])
        for row in decision_rows
        if row.get("correlation_decision") == "selected_corr_diversified_paper_monitor"
    ]
    selected_corr = _matrix_subset(
        corr_by_split["train_validation"], selected_ids, limit=REPRESENTATIVE_MATRIX_LIMIT
    )
    all_clusters = _correlation_clusters(
        corr_by_split["train_validation"], threshold=HIGH_CORR_CLUSTER_THRESHOLD
    )
    multi_member_clusters = [cluster for cluster in all_clusters if len(cluster) > 1]
    shadow_frames = {
        split: _aligned_pnl_frame(captures, shadow_ids, split=split)
        for split in ("validation", "locked_oos", "train_validation")
    }
    shadow_corr = _corr_matrix(shadow_frames["train_validation"])
    frames_for_portfolio = {
        split: frame for split, frame in split_frames.items() if split in SPLIT_ORDER
    }
    selected_stats = _portfolio_stats(frames_for_portfolio, selected_ids, rows_by_id)
    all_paper_stats = _portfolio_stats(frames_for_portfolio, paper_ids, rows_by_id)
    selected_rows = _selected_summary_rows(decision_rows)
    missing = sorted(
        str(row["model_id"]) for row in replay_rows if str(row.get("model_id")) not in captures
    )
    timestamp = _timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_json = output_dir / "alpha_zoo_pnl_correlation_decision_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_pnl_correlation_decision_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_pnl_correlation_decision_latest.md"
    decision_csv = output_dir / "correlation_decision_rows_latest.csv"
    selected_csv = output_dir / "selected_corr_diversified_candidates_latest.csv"
    paper_corr_tv_csv = output_dir / "paper_pnl_corr_train_validation_latest.csv"
    paper_corr_validation_csv = output_dir / "paper_pnl_corr_validation_latest.csv"
    paper_corr_oos_csv = output_dir / "paper_pnl_corr_locked_oos_report_only_latest.csv"
    selected_corr_csv = output_dir / "selected_pnl_corr_train_validation_latest.csv"
    shadow_corr_csv = output_dir / "shadow_representative_pnl_corr_train_validation_latest.csv"
    methodology_md = output_dir / "pnl_correlation_decision_methodology_latest.md"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at_utc": _utc_now_iso(),
        "source_monitoring_artifact": str(monitoring_artifact_path),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "bbo_spread_multiplier": BBO_SPREAD_MULTIPLIER,
        "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "paper_testnet_only": True,
        "ready_for_paper": bool(selected_rows),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": bool(selected_rows),
        "real_execution_allowed": False,
        "correlation_method": {
            "method": CORRELATION_METHOD,
            "primary_surface": PRIMARY_CORR_SURFACE,
            "align_by": "datetime",
            "missing_bar_policy": "fill_missing_strategy_bars_with_zero_pnl_before_corr",
            "pnl_stream": "per-bar fractional strategy return after 10bps round-trip cost and native notional_fraction",
            "train_validation_corr_threshold_abs": MAX_TRAIN_VALIDATION_ABS_CORR,
            "validation_corr_threshold_abs": MAX_VALIDATION_ABS_CORR,
            "high_corr_cluster_threshold_abs": HIGH_CORR_CLUSTER_THRESHOLD,
        },
        "selection_policy": {
            "ranking_inputs": [
                "monitoring_score_train_validation_only",
                "train_return",
                "validation_return",
                "validation_mdd",
                "train_return_per_turnover_proxy_bps",
                "validation_return_per_turnover_proxy_bps",
            ],
            "correlation_inputs": ["train_pnl_stream", "validation_pnl_stream"],
            "locked_oos_role": "gate/report-only after train+validation correlation ranking freeze",
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "no_calendar_date_hack": True,
            "decision_rule": (
                "greedy train+validation score order; accept only if abs corr to selected candidates is <= "
                f"{MAX_TRAIN_VALIDATION_ABS_CORR:.2f} on train+validation and <= {MAX_VALIDATION_ABS_CORR:.2f} on validation"
            ),
        },
        "external_method_references": EXTERNAL_METHOD_REFERENCES,
        "correlation_decision_summary": {
            "paper_universe_candidate_count": len(paper_rows),
            "strict_shadow_representative_count": len(strict_shadow_rows),
            "captured_pnl_candidate_count": len(captures),
            "captured_paper_pnl_candidate_count": len(paper_ids),
            "missing_pnl_model_ids": missing,
            "selected_candidate_count": len(selected_rows),
            "high_corr_cluster_count": len(multi_member_clusters),
            "largest_high_corr_cluster_size": max(
                (len(cluster) for cluster in multi_member_clusters), default=0
            ),
            "decision": (
                "do_not_adopt_all_paper_candidates; use corr-diversified paper/testnet-only subset"
                if selected_rows and len(selected_rows) < len(paper_ids)
                else "paper_universe_not_reduced_by_correlation"
            ),
            "all_paper_adoption_judgement": "reject_unscaled_all_in_due_to_duplicate_pnl_clusters_and_excess_gross_notional",
        },
        "correlation_diagnostics": {
            "paper_train_validation": _matrix_pair_stats(corr_by_split["train_validation"]),
            "paper_validation": _matrix_pair_stats(corr_by_split["validation"]),
            "paper_locked_oos_report_only": _matrix_pair_stats(corr_by_split["locked_oos"]),
            "selected_train_validation": _matrix_pair_stats(selected_corr),
            "shadow_representative_train_validation": _matrix_pair_stats(shadow_corr),
            "top_abs_corr_pairs_train_validation": _top_abs_corr_pairs(
                corr_by_split["train_validation"], limit=25
            ),
            "high_corr_clusters_train_validation_abs_ge_0p85": [
                {"cluster_size": len(cluster), "model_ids": cluster[:25]}
                for cluster in multi_member_clusters[:25]
            ],
        },
        "portfolio_comparison": {
            "all_paper_candidates": all_paper_stats,
            "corr_diversified_selected": selected_stats,
        },
        "selected_corr_diversified_candidates": selected_rows,
        "correlation_decision_rows": decision_rows,
        "strict_shadow_representative_rows_report_only": strict_shadow_rows,
        "runner_peak_rss_mib": local_peak_mib,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "methodology_markdown": str(methodology_md),
            "decision_rows_csv": str(decision_csv),
            "selected_candidates_csv": str(selected_csv),
            "paper_corr_train_validation_csv": str(paper_corr_tv_csv),
            "paper_corr_validation_csv": str(paper_corr_validation_csv),
            "paper_corr_locked_oos_report_only_csv": str(paper_corr_oos_csv),
            "selected_corr_train_validation_csv": str(selected_corr_csv),
            "shadow_corr_train_validation_csv": str(shadow_corr_csv),
            "artifact_generation_validation_log": str(generation_log),
        },
    }
    if write_outputs:
        _write_json(latest_json, payload)
        _write_json(timestamped_json, payload)
        markdown = _render_markdown(payload, rows_by_id=rows_by_id, selected_corr=selected_corr)
        latest_md.write_text(markdown, encoding="utf-8")
        methodology_md.write_text(_methodology_markdown(payload), encoding="utf-8")
        _write_csv(decision_csv, decision_rows, CSV_FIELDS)
        _write_csv(
            selected_csv, [row for row in decision_rows if row.get("ready_for_paper")], CSV_FIELDS
        )
        _write_matrix_csv(paper_corr_tv_csv, corr_by_split["train_validation"])
        _write_matrix_csv(paper_corr_validation_csv, corr_by_split["validation"])
        _write_matrix_csv(paper_corr_oos_csv, corr_by_split["locked_oos"])
        _write_matrix_csv(selected_corr_csv, selected_corr)
        _write_matrix_csv(shadow_corr_csv, shadow_corr)
        generation_log.write_text(
            "artifact_kind=alpha_zoo_pnl_correlation_decision\n"
            f"paper_universe_candidate_count={len(paper_rows)}\n"
            f"captured_paper_pnl_candidate_count={len(paper_ids)}\n"
            f"selected_candidate_count={len(selected_rows)}\n"
            f"ready_for_real={payload['ready_for_real']}\n"
            f"real_money_execution={payload['real_money_execution']}\n"
            f"locked_oos_used_for_selection={payload['selection_policy']['uses_locked_oos_for_selection']}\n"
            f"runner_peak_rss_mib={local_peak_mib:.2f}\n",
            encoding="utf-8",
        )
    return payload


def _methodology_markdown(payload: Mapping[str, Any]) -> str:
    method = payload["correlation_method"]
    policy = payload["selection_policy"]
    lines = [
        "# PnL Correlation Decision Methodology",
        "",
        "This record defines how the paper/testnet Alpha Zoo portfolio is reduced after individual candidate gates pass.",
        "",
        "## Inputs",
        "",
        "- Candidate universe: `paper_testnet_monitor` rows from the multi-asset monitoring slate.",
        "- PnL stream: per-bar fractional strategy return after the 10bps round-trip cost assumption and native notional fraction.",
        "- Ranking: train+validation monitoring score only.",
        "- locked-OOS: report-only after selection freeze; never used for discovery, fitting, pruning, objective, or selection.",
        "",
        "## Correlation construction",
        "",
        f"- Method: `{method['method']}`.",
        "- Each strategy is replayed and indexed by bar datetime.",
        "- Different timeframes are aligned on the union of timestamps.",
        "- A missing timestamp for a strategy means the strategy has no bar/position update there, so PnL is filled with zero before correlation.",
        "- Primary matrix: combined train+validation PnL streams.",
        "- Validation-only matrix is a guardrail to avoid selecting candidates that diversify only because of train behavior.",
        "- locked-OOS correlation matrix is saved for report-only monitoring diagnostics.",
        "",
        "## Selection rule",
        "",
        f"- Sort by: {', '.join(policy['ranking_inputs'])}.",
        f"- Accept greedily if max abs train+validation corr to selected <= {method['train_validation_corr_threshold_abs']:.2f}.",
        f"- Also require max abs validation corr to selected <= {method['validation_corr_threshold_abs']:.2f}.",
        "- Reject otherwise as a high-PnL-correlation duplicate, not as a bad standalone alpha.",
        "- Any accepted strategy remains paper/testnet-only and inherits original risk/efficiency gates.",
        "",
        "## Interpretation",
        "",
        "- Correlation is a de-duplication and diversification diagnostic; it is not a real-money approval.",
        "- High-correlation clusters should be monitored as one alpha sleeve, not many independent strategies.",
        "- Unscaled adoption of all candidates is rejected when gross notional and high-correlation clusters are large.",
        "",
        "## Guardrails",
        "",
        f"- ready_for_real={str(payload['ready_for_real']).lower()}.",
        f"- real_money_execution={str(payload['real_money_execution']).lower()}.",
        f"- uses_locked_oos_for_selection={str(policy['uses_locked_oos_for_selection']).lower()}.",
        "",
    ]
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    monitoring_artifact = Path(args.monitoring_artifact).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    payload = _load_monitoring_payload(monitoring_artifact)
    return build_payload_from_monitoring(
        payload,
        output_dir=output_dir,
        monitoring_artifact_path=monitoring_artifact,
        data_root=data_root,
        feature_root=feature_root,
        shadow_matrix_limit=int(args.shadow_matrix_limit),
        write_outputs=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitoring-artifact", default=str(DEFAULT_MONITORING_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--shadow-matrix-limit", type=int, default=REPRESENTATIVE_MATRIX_LIMIT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "summary": payload["correlation_decision_summary"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
