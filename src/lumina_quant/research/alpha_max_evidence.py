"""Fail-closed evidence contracts for the alpha-max research experiment.

This module deliberately contains only deterministic validation, allocation, and
materialization helpers.  It does not discover data, inspect the environment, or
construct a backtest.  Callers must supply every root, universe, and output path
explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import polars as pl

from lumina_quant.backtesting.data_windowed_parquet import (
    HistoricParquetWindowedDataHandler,
    RawPoint,
)
from lumina_quant.backtesting.execution_model import (
    ExecutionModel,
    ExecutionPricingTrace,
    execution_pricing_trace_sha256,
)
from lumina_quant.backtesting.execution_sim import NoFillAttempt
from lumina_quant.backtesting.portfolio_backtest import FillApplicationAttribution
from lumina_quant.data.feature_points import (
    FEATURE_POINT_MAX_STALE_MS,
    FeaturePoint,
    FeaturePointLookup,
)
from lumina_quant.portfolio.optimizer_core import project_simplex_with_upper_bounds
from lumina_quant.portfolio import optimizer_core
from lumina_quant.portfolio.optimizers_extra import ERCPortfolio
from lumina_quant.portfolio.quality_gated_allocation import (
    _hrp_weights_with_correlation_shrinkage,
    _round,
)
from lumina_quant.utils.artifact_read_receipt import ArtifactReadReceipt, read_artifact_bytes
from lumina_quant.strategy_factory import research_metrics

__all__ = [
    "ALPHA_MAX_CANDIDATE_SYMBOLS",
    "ALPHA_MAX_DSR_NUM_TRIALS",
    "ALPHA_MAX_MANIFEST_CHILD_KEYS",
    "ALPHA_MAX_MANIFEST_TOP_LEVEL_KEYS",
    "ALPHA_MAX_PERIODS_PER_YEAR",
    "AlphaMaxActualEngineRunReceipt",
    "AlphaMaxAdmissionArtifact",
    "AlphaMaxAdmissionCandidateInput",
    "AlphaMaxAdmissionComputation",
    "AlphaMaxAdmissionDailyCandidateInput",
    "AlphaMaxCapacityDiagnostics",
    "AlphaMaxCapsuleReceipt",
    "AlphaMaxCombinedStreamingEquityEvidence",
    "AlphaMaxContractManifestSeal",
    "AlphaMaxContractRecord",
    "AlphaMaxCostCellEvidence",
    "AlphaMaxCostCellPreGateEvidence",
    "AlphaMaxDailyQuoteNotional",
    "AlphaMaxEquityEndpoint",
    "AlphaMaxFoldRunEvidence",
    "AlphaMaxFundingBoundaryLedgerRow",
    "AlphaMaxFundingBoundaryRequest",
    "AlphaMaxFundingBoundaryResolver",
    "AlphaMaxGateDecision",
    "AlphaMaxGateInput",
    "AlphaMaxLiquidationEventEvidence",
    "AlphaMaxManifestMaterialization",
    "AlphaMaxManifestReceipt",
    "AlphaMaxMetricStatistics",
    "AlphaMaxNormalizedFoldSegmentEvidence",
    "AlphaMaxOrderedFundingLookup",
    "AlphaMaxPreGateSharpeEvidence",
    "AlphaMaxPrelockArtifact",
    "AlphaMaxPrelockSeal",
    "AlphaMaxPrimaryReturnStream",
    "AlphaMaxRawObservation",
    "AlphaMaxReconciliationEvidence",
    "AlphaMaxRootReceipt",
    "AlphaMaxRootSeal",
    "AlphaMaxRowEvidence",
    "AlphaMaxSelectionResult",
    "AlphaMaxStatisticalEvidence",
    "AlphaMaxStreamingEquityEvidence",
    "AlphaMaxStreamingEquityTracker",
    "AlphaMaxTerminalGateEvidence",
    "AlphaMaxTerminalState",
    "AlphaMaxTreeEntry",
    "AlphaMaxTrialLedger",
    "AlphaMaxTurnoverRPTDiagnostics",
    "FeatureRootSpec",
    "allocate_alpha_max_equal_risk",
    "allocate_alpha_max_equal_weight",
    "allocate_alpha_max_shrunk_hrp",
    "alpha_max_common_rng_seed",
    "alpha_max_common_rng_seed_payload",
    "alpha_max_drawdown_duration",
    "alpha_max_equal_risk_weights",
    "alpha_max_equal_weight_weights",
    "alpha_max_full_event_mdd",
    "alpha_max_pre_gate_sharpe_variance",
    "alpha_max_seed_schedule_sha256",
    "alpha_max_shrunk_hrp_weights",
    "alpha_max_terminal_outcome",
    "alpha_max_trial_key",
    "alpha_max_trial_key_set_lf_bytes",
    "alpha_max_type7_quantile",
    "build_alpha_max_actual_engine_run_receipt",
    "build_alpha_max_cost_cell_evidence",
    "build_alpha_max_cost_cell_pre_gate_evidence",
    "build_alpha_max_daily_quote_notional",
    "build_alpha_max_fold_run_evidence",
    "build_alpha_max_normalized_fold_segment_evidence",
    "build_alpha_max_prelock_seal",
    "build_alpha_max_primary_return_stream",
    "build_alpha_max_statistical_evidence",
    "build_alpha_max_terminal_state",
    "build_alpha_max_trial_ledger",
    "canonical_alpha_max_cost_cell_bytes",
    "canonical_alpha_max_row_bytes",
    "compute_alpha_max_capacity_diagnostics",
    "compute_alpha_max_metric_statistics",
    "compute_alpha_max_train_admission",
    "compute_alpha_max_train_admission_from_daily_summaries",
    "compute_alpha_max_turnover_rpt",
    "materialize_alpha_max_manifest",
    "normalize_alpha_max_prior_trial_node",
    "rank_alpha_max_historical_report",
    "read_alpha_max_prior_trial_blob",
    "reconcile_alpha_max_cost_attribution",
    "seal_alpha_max_contract_manifest",
    "seal_alpha_max_root_tree",
    "select_alpha_max_prelock_champion",
    "validate_alpha_max_admission_artifact",
    "validate_alpha_max_admitted_symbols",
]


ALPHA_MAX_CANDIDATE_SYMBOLS: Final[tuple[str, ...]] = (
    "ADAUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "TONUSDT",
    "TRXUSDT",
    "XRPUSDT",
)
ALPHA_MAX_PERIODS_PER_YEAR: Final[int] = 2190
ALPHA_MAX_DSR_NUM_TRIALS: Final[int] = 1487

_ADMISSION_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_kind",
        "phase",
        "selection_inputs",
        "input_root_hashes",
        "candidate_symbols",
        "candidate_symbols_sha256",
        "admitted_symbols",
        "admitted_symbols_sha256",
        "per_candidate",
    }
)
_ADMISSION_CANDIDATE_KEYS: Final[frozenset[str]] = frozenset({"admitted", "reasons", "statistics"})
_ADMISSION_STATISTIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "daily_quote_notional_day_count",
        "median_quote_notional_usdt",
        "p10_quote_notional_usdt",
        "consecutive_completed_daily_bars_before_train",
        "readable_monotone_unique_finite_partitions",
        "complete_train_daily_keys",
        "complete_train_4h_keys",
        "causal_funding_coverage_complete",
        "unresolved_daily_cross_section_count",
    }
)
_ADMISSION_INPUT_ROOT_IDS: Final[tuple[str, str]] = ("warmup", "train")
_ADMISSION_DAILY_QUOTE_NOTIONAL_DAYS: Final[int] = 517
_ADMISSION_PRETRAIN_DAILY_BARS: Final[int] = 366
_ADMISSION_MEDIAN_MINIMUM: Final[float] = 20_000_000.0
_ADMISSION_P10_MINIMUM: Final[float] = 2_000_000.0

_FUNDING_INTERVAL_MS: Final[int] = 8 * 60 * 60 * 1000
_FEATURE_CAUSAL_EDGE_MS: Final[int] = 1000
_RAW_CLOSE_MAX_STALE_MS: Final[int] = 1000
_MANIFEST_PHASES: Final[tuple[str, str]] = (
    "validation_train_fit",
    "prelock_final_refit",
)
_FORBIDDEN_OOS_KEYS: Final[tuple[str, ...]] = (
    "uses_current_fold_oos",
    "uses_locked_oos_for_selection",
    "uses_locked_oos_for_objective",
    "uses_locked_oos_for_pruning",
    "uses_locked_oos_for_parameter_fitting",
    "uses_locked_oos_for_threshold",
    "uses_locked_oos_for_tie_break",
    "uses_locked_oos_for_correlation",
    "uses_locked_oos_for_sizing",
)
_REAL_MONEY_KEYS: Final[tuple[str, str, str]] = (
    "real_money_execution",
    "allow_real_money",
    "ready_for_real",
)

ALPHA_MAX_MANIFEST_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_kind",
        "candidate_symbols",
        "admitted_symbols",
        "admission_manifest_sha256",
        *_REAL_MONEY_KEYS,
        *_FORBIDDEN_OOS_KEYS,
        "gross_cap",
        "cash_weight",
        "allocation_method",
        "optimizer_provenance",
        "correlation_input_provenance",
        "source_artifacts",
        "children",
    }
)
ALPHA_MAX_MANIFEST_CHILD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "candidate_id",
        "name",
        "strategy_class",
        "candidate_symbols",
        "symbols",
        "params",
        "weight",
        "leaf_gross",
        "leaf_gross_cap",
        "netting_group",
        "netting_group_gross_cap",
        "source_artifact_id",
        "ready",
        "portfolio_ready",
        *_REAL_MONEY_KEYS,
        "no_current_fold_oos_provenance",
        "train_validation_optimizer_provenance",
        "lagged_completed_shadow_optimizer_provenance",
        *_FORBIDDEN_OOS_KEYS,
        "optimizer_provenance",
        "correlation_input_provenance",
    }
)


def _utc(value: datetime | str, *, field: str) -> datetime:
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        token = value.strip()
        if token.endswith("Z"):
            token = f"{token[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(token)
        except ValueError as exc:
            raise ValueError(f"{field}_invalid") from exc
    else:
        raise TypeError(f"{field} must be a UTC datetime or RFC3339 string")
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field}_must_be_utc")
    return parsed.astimezone(UTC)


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _canonical_json_bytes(value: Any, *, newline: bool) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _symbol_sequence_sha256(symbols: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(symbols), newline=False))


def _require_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field}_invalid")
    token = value
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError(f"{field}_invalid")
    return token


def _require_explicit_canonical_path(path: str | os.PathLike[str], *, field: str) -> str:
    raw = os.fspath(path)
    if not raw or not os.path.isabs(raw):
        raise ValueError(f"{field}_must_be_absolute")
    lexical = os.path.abspath(raw)
    target = Path(lexical)
    if target.is_symlink():
        raise ValueError(f"{field}_symlink_rejected")
    try:
        canonical = str(target.resolve(strict=True))
    except FileNotFoundError as exc:
        raise ValueError(f"{field}_missing") from exc
    if canonical != lexical:
        raise ValueError(f"{field}_noncanonical")
    return canonical


_ROOT_INTERVALS: Final[dict[str, tuple[datetime, datetime]]] = {
    "warmup": (
        datetime(2022, 12, 31, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
    ),
    "train": (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2025, 6, 1, tzinfo=UTC),
    ),
    "purge": (
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2025, 6, 8, tzinfo=UTC),
    ),
    "validation": (
        datetime(2025, 6, 8, tzinfo=UTC),
        datetime(2025, 8, 31, tzinfo=UTC),
    ),
    "embargo": (
        datetime(2025, 8, 31, tzinfo=UTC),
        datetime(2025, 9, 7, tzinfo=UTC),
    ),
    "historical_exposed_evaluation": (
        datetime(2025, 9, 7, tzinfo=UTC),
        datetime(2026, 7, 1, tzinfo=UTC),
    ),
}
_ALLOWED_ROOT_SEQUENCES: Final[frozenset[tuple[str, ...]]] = frozenset(
    {
        ("warmup",),
        ("warmup", "train"),
        ("train", "purge"),
        ("purge", "validation"),
        ("validation", "embargo"),
        ("embargo", "historical_exposed_evaluation"),
    }
)
_ALPHA_MAX_VALIDATION_FOLD_IDS: Final[tuple[str, ...]] = tuple(
    f"validation_w{index:02d}" for index in range(1, 13)
)
_ALPHA_MAX_HISTORICAL_FOLD_IDS: Final[tuple[str, ...]] = (
    "historical_2025_09_partial",
    "historical_2025_10",
    "historical_2025_11",
    "historical_2025_12",
    "historical_2026_01",
    "historical_2026_02",
    "historical_2026_03",
    "historical_2026_04",
    "historical_2026_05",
    "historical_2026_06",
)
_ALPHA_MAX_VALIDATION_FOLD_INTERVALS: Final[tuple[tuple[datetime, datetime], ...]] = tuple(
    (
        _ROOT_INTERVALS["validation"][0] + timedelta(days=7 * index),
        _ROOT_INTERVALS["validation"][0] + timedelta(days=7 * (index + 1)),
    )
    for index in range(12)
)
_ALPHA_MAX_HISTORICAL_FOLD_INTERVALS: Final[tuple[tuple[datetime, datetime], ...]] = (
    (
        datetime(2025, 9, 7, tzinfo=UTC),
        datetime(2025, 10, 1, tzinfo=UTC),
    ),
    *tuple(
        (
            datetime(year, month, 1, tzinfo=UTC),
            (
                datetime(year + 1, 1, 1, tzinfo=UTC)
                if month == 12
                else datetime(year, month + 1, 1, tzinfo=UTC)
            ),
        )
        for year, month in (
            (2025, 10),
            (2025, 11),
            (2025, 12),
            (2026, 1),
            (2026, 2),
            (2026, 3),
            (2026, 4),
            (2026, 5),
            (2026, 6),
        )
    ),
)
_ALPHA_MAX_DOMAIN_FOLD_IDS: Final[dict[str, tuple[str, ...]]] = {
    "validation": _ALPHA_MAX_VALIDATION_FOLD_IDS,
    "historical_exposed_evaluation": _ALPHA_MAX_HISTORICAL_FOLD_IDS,
}
_ALPHA_MAX_FOLD_INTERVALS: Final[dict[str, tuple[datetime, datetime]]] = dict(
    zip(_ALPHA_MAX_VALIDATION_FOLD_IDS, _ALPHA_MAX_VALIDATION_FOLD_INTERVALS, strict=True)
) | dict(zip(_ALPHA_MAX_HISTORICAL_FOLD_IDS, _ALPHA_MAX_HISTORICAL_FOLD_INTERVALS, strict=True))
_ALPHA_MAX_LOGICAL_ACTUAL_ENGINE_CELL_COUNT: Final[int] = 68
_ALPHA_MAX_DOMAIN_ENGINE_RUN_COUNT: Final[dict[str, int]] = {
    domain: _ALPHA_MAX_LOGICAL_ACTUAL_ENGINE_CELL_COUNT * len(fold_ids)
    for domain, fold_ids in _ALPHA_MAX_DOMAIN_FOLD_IDS.items()
}
_ALPHA_MAX_DOMAIN_ROOT_IDS: Final[dict[str, tuple[str, str]]] = {
    "validation": ("purge", "validation"),
    "historical_exposed_evaluation": ("embargo", "historical_exposed_evaluation"),
}


@dataclass(frozen=True, slots=True)
class FeatureRootSpec:
    """Explicit half-open ownership contract for one feature root."""

    root_id: str
    path: str
    exchange: str
    start_utc: datetime | str
    end_utc: datetime | str
    inventory_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        root_id = str(self.root_id or "")
        if root_id not in _ROOT_INTERVALS:
            raise ValueError("feature_root_id_invalid")
        if self.exchange != "binance":
            raise ValueError("feature_root_exchange_invalid")
        start = _utc(self.start_utc, field="feature_root_start")
        end = _utc(self.end_utc, field="feature_root_end")
        if start >= end:
            raise ValueError("feature_root_bounds_invalid")
        expected_start, expected_end = _ROOT_INTERVALS[root_id]
        if (start, end) != (expected_start, expected_end):
            raise ValueError("feature_root_frozen_bounds_mismatch")
        canonical_path = _require_explicit_canonical_path(self.path, field="feature_root_path")
        object.__setattr__(self, "root_id", root_id)
        object.__setattr__(self, "path", canonical_path)
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)
        object.__setattr__(
            self,
            "inventory_sha256",
            _require_sha256(self.inventory_sha256, field="feature_root_inventory_sha256"),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _require_sha256(self.content_sha256, field="feature_root_content_sha256"),
        )

    @property
    def start_timestamp_ms(self) -> int:
        return _epoch_ms(self.start_utc)  # type: ignore[arg-type]

    @property
    def end_timestamp_ms(self) -> int:
        return _epoch_ms(self.end_utc)  # type: ignore[arg-type]


class AlphaMaxOrderedFundingLookup:
    """Strict newest-point composite over one frozen adjacent root sequence."""

    __slots__ = ("_locked", "_lookups", "_root_specs")

    def __init__(self, root_specs: Sequence[FeatureRootSpec]) -> None:
        specs = tuple(root_specs)
        if not specs or any(type(spec) is not FeatureRootSpec for spec in specs):
            raise TypeError("feature_root_specs_must_be_exact")
        root_ids = tuple(spec.root_id for spec in specs)
        if root_ids not in _ALLOWED_ROOT_SEQUENCES:
            raise ValueError("feature_root_order_not_immediately_adjacent")
        if len({spec.path for spec in specs}) != len(specs):
            raise ValueError("feature_root_path_duplicate")
        if len(specs) == 2 and specs[0].end_utc != specs[1].start_utc:
            raise ValueError("feature_root_gap_or_overlap")

        lookups = tuple(
            FeaturePointLookup(
                db_path=spec.path,
                exchange=spec.exchange,
                start_date=spec.start_utc,
                end_date=spec.end_utc,
            )
            for spec in specs
        )
        for spec, lookup in zip(specs, lookups, strict=True):
            if getattr(lookup, "db_path", spec.path) != spec.path:
                raise ValueError("feature_lookup_path_identity_mismatch")
            if getattr(lookup, "exchange", spec.exchange) != spec.exchange:
                raise ValueError("feature_lookup_exchange_identity_mismatch")
        object.__setattr__(self, "_root_specs", specs)
        object.__setattr__(self, "_lookups", lookups)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("AlphaMaxOrderedFundingLookup is immutable")
        object.__setattr__(self, name, value)

    @property
    def root_specs(self) -> tuple[FeatureRootSpec, ...]:
        return self._root_specs

    @property
    def ordered_root_ids(self) -> tuple[str, ...]:
        return tuple(spec.root_id for spec in self._root_specs)

    @property
    def current_root(self) -> FeatureRootSpec:
        return self._root_specs[-1]

    @property
    def db_path(self) -> str:
        """Expose the exact current root as the engine's immutable capability token."""
        return self.current_root.path

    def get_latest_point(
        self,
        symbol: str,
        field: str,
        *,
        timestamp_ms: int | None,
    ) -> FeaturePoint | None:
        if field != "funding_rate":
            raise ValueError("alpha_max_funding_lookup_field_forbidden")
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
            raise TypeError("alpha_max_funding_query_timestamp_must_be_int")
        query_ms = timestamp_ms
        current = self.current_root
        if not (current.start_timestamp_ms <= query_ms <= current.end_timestamp_ms):
            raise ValueError("alpha_max_funding_query_outside_current_root")

        candidates: list[tuple[FeatureRootSpec, FeaturePoint]] = []
        for spec, lookup in zip(self._root_specs, self._lookups, strict=True):
            point = lookup.get_latest_point(symbol, field, timestamp_ms=query_ms)
            if point is None:
                continue
            if type(point) is not FeaturePoint:
                try:
                    point = FeaturePoint(
                        value=float(point.value),
                        source_timestamp_ms=int(point.source_timestamp_ms),
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError("alpha_max_funding_point_invalid") from exc
            candidates.append((spec, point))

        timestamps = [point.source_timestamp_ms for _, point in candidates]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("alpha_max_funding_equal_timestamp_conflict")
        eligible: list[FeaturePoint] = []
        for spec, point in candidates:
            source_ms = point.source_timestamp_ms
            if not (spec.start_timestamp_ms <= source_ms < spec.end_timestamp_ms):
                raise ValueError("alpha_max_funding_point_outside_owned_root")
            if source_ms > query_ms:
                raise ValueError("alpha_max_funding_point_from_future")
            if query_ms - source_ms > FEATURE_POINT_MAX_STALE_MS:
                raise ValueError("alpha_max_funding_point_stale")
            if not math.isfinite(point.value):
                raise ValueError("alpha_max_funding_point_nonfinite")
            eligible.append(point)
        if not eligible:
            return None
        return max(eligible, key=lambda point: point.source_timestamp_ms)

    def get_latest(
        self,
        symbol: str,
        field: str,
        *,
        timestamp_ms: int | None,
    ) -> float | None:
        point = self.get_latest_point(symbol, field, timestamp_ms=timestamp_ms)
        return None if point is None else point.value


def validate_alpha_max_admitted_symbols(
    candidate_symbols: Sequence[str], admitted_symbols: Sequence[str]
) -> tuple[str, ...]:
    """Validate the exact candidate identity and one frozen execution subset."""
    candidates = tuple(candidate_symbols)
    if candidates != ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_candidate_symbols_mismatch")
    admitted = tuple(admitted_symbols)
    if not 5 <= len(admitted) <= 10:
        raise ValueError("alpha_max_admitted_symbol_count_invalid")
    if admitted != tuple(sorted(admitted)) or len(admitted) != len(set(admitted)):
        raise ValueError("alpha_max_admitted_symbols_not_lexicographic_unique")
    if any(symbol not in candidates for symbol in admitted):
        raise ValueError("alpha_max_admitted_symbol_outside_candidates")
    return admitted


@dataclass(frozen=True, slots=True)
class AlphaMaxAdmissionArtifact:
    """Canonical, train-only candidate-to-admitted membership seal."""

    candidate_symbols: tuple[str, ...]
    admitted_symbols: tuple[str, ...]
    sha256: str
    canonical_bytes: bytes


def _admission_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"alpha_max_admission_{field}_invalid")
    return value


def _admission_nonnegative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"alpha_max_admission_{field}_invalid")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"alpha_max_admission_{field}_invalid")
    return parsed


def _expected_admission_reasons(statistics: Mapping[str, Any]) -> tuple[str, ...]:
    day_count = _admission_nonnegative_int(
        statistics["daily_quote_notional_day_count"],
        field="daily_quote_notional_day_count",
    )
    median = _admission_nonnegative_number(
        statistics["median_quote_notional_usdt"],
        field="median_quote_notional_usdt",
    )
    p10 = _admission_nonnegative_number(
        statistics["p10_quote_notional_usdt"],
        field="p10_quote_notional_usdt",
    )
    if p10 > median:
        raise ValueError("alpha_max_admission_quote_notional_statistics_invalid")
    pretrain_bars = _admission_nonnegative_int(
        statistics["consecutive_completed_daily_bars_before_train"],
        field="consecutive_completed_daily_bars_before_train",
    )
    unresolved_count = _admission_nonnegative_int(
        statistics["unresolved_daily_cross_section_count"],
        field="unresolved_daily_cross_section_count",
    )
    boolean_fields = (
        "readable_monotone_unique_finite_partitions",
        "complete_train_daily_keys",
        "complete_train_4h_keys",
        "causal_funding_coverage_complete",
    )
    if any(type(statistics[field]) is not bool for field in boolean_fields):
        raise ValueError("alpha_max_admission_coverage_statistic_invalid")

    reasons: list[str] = []
    if day_count != _ADMISSION_DAILY_QUOTE_NOTIONAL_DAYS:
        reasons.append("daily_quote_notional_day_count_mismatch")
    if median < _ADMISSION_MEDIAN_MINIMUM:
        reasons.append("median_quote_notional_below_minimum")
    if p10 < _ADMISSION_P10_MINIMUM:
        reasons.append("p10_quote_notional_below_minimum")
    if pretrain_bars != _ADMISSION_PRETRAIN_DAILY_BARS:
        reasons.append("pretrain_daily_history_incomplete")
    if not statistics["readable_monotone_unique_finite_partitions"]:
        reasons.append("partition_integrity_failure")
    if not statistics["complete_train_daily_keys"]:
        reasons.append("incomplete_train_daily_keys")
    if not statistics["complete_train_4h_keys"]:
        reasons.append("incomplete_train_4h_keys")
    if not statistics["causal_funding_coverage_complete"]:
        reasons.append("incomplete_causal_funding_coverage")
    if unresolved_count != 0:
        reasons.append("unresolved_daily_cross_section")
    return tuple(sorted(reasons))


def validate_alpha_max_admission_artifact(
    artifact: Mapping[str, Any] | bytes,
    *,
    expected_sha256: str | None = None,
) -> AlphaMaxAdmissionArtifact:
    """Validate the exact canonical train-only admission evidence schema.

    The runner owns observation aggregation.  This boundary independently binds
    the candidate/admitted sequence hashes, the exact warmup/train root hashes,
    and one complete reasons/statistics row per candidate.  Membership must agree
    with the frozen numeric and coverage gates; validation or historical inputs
    cannot be represented by the schema.
    """
    if isinstance(artifact, bytes):
        try:
            payload = json.loads(artifact.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("alpha_max_admission_json_invalid") from exc
        if not isinstance(payload, dict):
            raise TypeError("alpha_max_admission_artifact_not_object")
        canonical = _canonical_json_bytes(payload, newline=True)
        if artifact != canonical:
            raise ValueError("alpha_max_admission_bytes_not_canonical")
    elif isinstance(artifact, Mapping):
        payload = dict(artifact)
        canonical = _canonical_json_bytes(payload, newline=True)
    else:
        raise TypeError("alpha_max_admission_artifact_invalid")

    if set(payload) != _ADMISSION_ARTIFACT_KEYS:
        raise ValueError("alpha_max_admission_key_set_mismatch")
    if payload["artifact_kind"] != "alpha_max_train_admission.v1":
        raise ValueError("alpha_max_admission_artifact_kind_invalid")
    if payload["phase"] != "train_admission":
        raise ValueError("alpha_max_admission_not_train_only")
    if payload["selection_inputs"] != ["warmup", "train"]:
        raise ValueError("alpha_max_admission_selection_inputs_invalid")

    root_hashes = payload["input_root_hashes"]
    if type(root_hashes) is not dict or tuple(sorted(root_hashes)) != tuple(
        sorted(_ADMISSION_INPUT_ROOT_IDS)
    ):
        raise ValueError("alpha_max_admission_input_roots_not_warmup_train")
    for root_id in _ADMISSION_INPUT_ROOT_IDS:
        _require_sha256(
            root_hashes[root_id],
            field=f"alpha_max_admission_{root_id}_root_sha256",
        )

    candidates_raw = payload.get("candidate_symbols")
    admitted_raw = payload.get("admitted_symbols")
    if type(candidates_raw) is not list or type(admitted_raw) is not list:
        raise TypeError("alpha_max_admission_symbol_lists_required")
    admitted = validate_alpha_max_admitted_symbols(candidates_raw, admitted_raw)
    expected_candidate_sha = _symbol_sequence_sha256(ALPHA_MAX_CANDIDATE_SYMBOLS)
    candidate_sha = _require_sha256(
        payload["candidate_symbols_sha256"],
        field="alpha_max_admission_candidate_symbols_sha256",
    )
    if candidate_sha != expected_candidate_sha:
        raise ValueError("alpha_max_admission_candidate_symbols_sha256_mismatch")
    expected_admitted_sha = _symbol_sequence_sha256(admitted)
    admitted_sha = _require_sha256(
        payload["admitted_symbols_sha256"],
        field="alpha_max_admission_admitted_symbols_sha256",
    )
    if admitted_sha != expected_admitted_sha:
        raise ValueError("alpha_max_admission_admitted_symbols_sha256_mismatch")

    per_candidate = payload["per_candidate"]
    if type(per_candidate) is not dict or set(per_candidate) != set(ALPHA_MAX_CANDIDATE_SYMBOLS):
        raise ValueError("alpha_max_admission_per_candidate_coverage_mismatch")
    admitted_set = set(admitted)
    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        row = per_candidate[symbol]
        if type(row) is not dict or set(row) != _ADMISSION_CANDIDATE_KEYS:
            raise ValueError("alpha_max_admission_candidate_row_key_set_mismatch")
        if type(row["admitted"]) is not bool:
            raise ValueError("alpha_max_admission_candidate_status_invalid")
        statistics = row["statistics"]
        if type(statistics) is not dict or set(statistics) != _ADMISSION_STATISTIC_KEYS:
            raise ValueError("alpha_max_admission_statistics_key_set_mismatch")
        expected_reasons = _expected_admission_reasons(statistics)
        reasons = row["reasons"]
        if (
            type(reasons) is not list
            or any(type(reason) is not str or not reason for reason in reasons)
            or reasons != sorted(set(reasons))
            or tuple(reasons) != expected_reasons
        ):
            raise ValueError("alpha_max_admission_candidate_reasons_mismatch")
        expected_status = not expected_reasons
        if row["admitted"] is not expected_status or expected_status != (symbol in admitted_set):
            raise ValueError("alpha_max_admission_candidate_membership_mismatch")

    actual_sha256 = _sha256_bytes(canonical)
    if expected_sha256 is not None and actual_sha256 != _require_sha256(
        expected_sha256, field="alpha_max_admission_expected_sha256"
    ):
        raise ValueError("alpha_max_admission_sha256_mismatch")
    return AlphaMaxAdmissionArtifact(
        candidate_symbols=ALPHA_MAX_CANDIDATE_SYMBOLS,
        admitted_symbols=admitted,
        sha256=actual_sha256,
        canonical_bytes=canonical,
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxFundingBoundaryRequest:
    symbol: str
    boundary_ms: int
    qty: float
    latest_datetime: datetime


@dataclass(frozen=True, slots=True)
class AlphaMaxFundingBoundaryLedgerRow:
    symbol: str
    boundary_ms: int
    rate_source_timestamp_ms: int
    price_row_timestamp_ms: int
    price_close_timestamp_ms: int
    qty: float
    rate: float
    price: float
    payment: float | None = None


class AlphaMaxFundingBoundaryResolver:
    """Resolve and atomically seal causal per-boundary funding inputs."""

    __slots__ = (
        "_admitted_symbols",
        "_bound_raw_accessor_owner",
        "_ledger",
        "_locked",
        "_ordered_lookup",
    )

    def __init__(
        self,
        ordered_lookup: AlphaMaxOrderedFundingLookup,
        admitted_symbols: tuple[str, ...],
    ) -> None:
        if type(ordered_lookup) is not AlphaMaxOrderedFundingLookup:
            raise TypeError("alpha_max_ordered_lookup_identity_invalid")
        if type(admitted_symbols) is not tuple:
            raise TypeError("alpha_max_admitted_symbols_must_be_tuple")
        validated = validate_alpha_max_admitted_symbols(
            ALPHA_MAX_CANDIDATE_SYMBOLS, admitted_symbols
        )
        object.__setattr__(self, "_ordered_lookup", ordered_lookup)
        object.__setattr__(self, "_admitted_symbols", admitted_symbols)
        if validated is not admitted_symbols:
            raise AssertionError("admitted tuple identity was not preserved")
        object.__setattr__(self, "_bound_raw_accessor_owner", None)
        object.__setattr__(self, "_ledger", ())
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False) and name in {
            "_admitted_symbols",
            "_bound_raw_accessor_owner",
            "_ledger",
            "_locked",
            "_ordered_lookup",
        }:
            raise AttributeError("funding resolver contracts are constructor-bound")
        object.__setattr__(self, name, value)

    @property
    def ordered_lookup(self) -> AlphaMaxOrderedFundingLookup:
        return self._ordered_lookup

    @property
    def admitted_symbols(self) -> tuple[str, ...]:
        return self._admitted_symbols

    @property
    def ledger(self) -> tuple[AlphaMaxFundingBoundaryLedgerRow, ...]:
        return self._ledger

    @staticmethod
    def _validate_prior_ledger(
        ledger: tuple[AlphaMaxFundingBoundaryLedgerRow, ...],
        *,
        admitted_symbols: tuple[str, ...],
    ) -> None:
        if type(ledger) is not tuple or any(
            type(value) is not AlphaMaxFundingBoundaryLedgerRow for value in ledger
        ):
            raise TypeError("alpha_max_funding_prior_ledger_identity_invalid")
        keys: list[tuple[int, str]] = []
        for row in ledger:
            if row.symbol not in admitted_symbols:
                raise ValueError("alpha_max_funding_prior_ledger_symbol_invalid")
            if (
                type(row.boundary_ms) is not int
                or row.boundary_ms <= 0
                or row.boundary_ms % _FUNDING_INTERVAL_MS != 0
                or type(row.rate_source_timestamp_ms) is not int
                or type(row.price_row_timestamp_ms) is not int
                or type(row.price_close_timestamp_ms) is not int
                or row.rate_source_timestamp_ms > row.boundary_ms
                or row.boundary_ms - row.rate_source_timestamp_ms > _FUNDING_INTERVAL_MS
                or row.price_close_timestamp_ms != row.price_row_timestamp_ms + 1000
                or row.price_close_timestamp_ms > row.boundary_ms
                or row.boundary_ms - row.price_close_timestamp_ms > _RAW_CLOSE_MAX_STALE_MS
            ):
                raise ValueError("alpha_max_funding_prior_ledger_timestamp_invalid")
            if (
                row.payment is None
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in (row.qty, row.rate, row.price, row.payment)
                )
                or abs(row.qty) < 1e-12
                or row.price <= 0.0
            ):
                raise ValueError("alpha_max_funding_prior_ledger_value_invalid")
            keys.append((row.boundary_ms, row.symbol))
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("alpha_max_funding_prior_ledger_order_invalid")

    def carry_forward(self) -> AlphaMaxFundingBoundaryResolver:
        """Return a fresh handler-owner capability with the exact paid ledger prefix."""
        self._validate_prior_ledger(
            self._ledger,
            admitted_symbols=self._admitted_symbols,
        )
        carried = AlphaMaxFundingBoundaryResolver(
            self._ordered_lookup,
            self._admitted_symbols,
        )
        object.__setattr__(carried, "_ledger", self._ledger)
        return carried

    def _validate_symbols_before_lookup(
        self, requests: Sequence[AlphaMaxFundingBoundaryRequest]
    ) -> None:
        if not requests:
            raise ValueError("funding_boundary_batch_empty")
        for request in requests:
            if request.symbol not in self._admitted_symbols:
                raise ValueError("funding_boundary_symbol_outside_admitted_domain")

    def _validate_raw_accessor(self, raw_point_accessor: Any) -> object:
        owner = getattr(raw_point_accessor, "__self__", None)
        function = getattr(raw_point_accessor, "__func__", None)
        if type(owner) is not HistoricParquetWindowedDataHandler:
            raise ValueError("funding_boundary_raw_accessor_owner_mismatch")
        if function is not HistoricParquetWindowedDataHandler.get_latest_raw_point:
            raise ValueError("funding_boundary_raw_accessor_function_mismatch")
        if getattr(owner, "_feature_lookup", None) is not self._ordered_lookup:
            raise ValueError("funding_boundary_feature_lookup_identity_mismatch")
        bound_owner = self._bound_raw_accessor_owner
        if bound_owner is not None and owner is not bound_owner:
            raise ValueError("funding_boundary_raw_accessor_replaced")
        return owner

    @staticmethod
    def _coerce_request(
        value: AlphaMaxFundingBoundaryRequest | Mapping[str, Any],
    ) -> AlphaMaxFundingBoundaryRequest:
        if isinstance(value, AlphaMaxFundingBoundaryRequest):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("funding_boundary_request_invalid")
        return AlphaMaxFundingBoundaryRequest(
            symbol=str(value.get("symbol") or ""),
            boundary_ms=value.get("boundary_ms"),
            qty=value.get("qty"),
            latest_datetime=value.get("latest_datetime"),
        )

    def _resolve_one(
        self,
        request: AlphaMaxFundingBoundaryRequest,
        *,
        raw_point_accessor: Any,
    ) -> AlphaMaxFundingBoundaryLedgerRow:
        boundary_ms = request.boundary_ms
        if isinstance(boundary_ms, bool) or not isinstance(boundary_ms, int):
            raise ValueError("funding_boundary_timestamp_invalid")
        if boundary_ms <= 0 or boundary_ms % _FUNDING_INTERVAL_MS != 0:
            raise ValueError("funding_boundary_not_utc_00_08_16")
        qty = float(request.qty)
        if not math.isfinite(qty) or abs(qty) < 1e-12:
            raise ValueError("funding_boundary_quantity_invalid")
        latest_datetime = _utc(request.latest_datetime, field="funding_boundary_latest_datetime")
        if _epoch_ms(latest_datetime) < boundary_ms:
            raise ValueError("funding_boundary_after_raw_watermark")

        rate_point = self._ordered_lookup.get_latest_point(
            request.symbol,
            "funding_rate",
            timestamp_ms=boundary_ms,
        )
        price_point = raw_point_accessor(
            request.symbol,
            "close",
            timestamp_ms=boundary_ms,
        )
        if type(rate_point) is not FeaturePoint or type(price_point) is not RawPoint:
            raise ValueError("funding_boundary_coverage")
        rate = float(rate_point.value)
        price = float(price_point.value)
        if (
            not math.isfinite(rate)
            or rate_point.source_timestamp_ms > boundary_ms
            or boundary_ms - rate_point.source_timestamp_ms > _FUNDING_INTERVAL_MS
        ):
            raise ValueError("funding_boundary_coverage")
        if (
            not math.isfinite(price)
            or price <= 0.0
            or price_point.close_timestamp_ms != price_point.row_timestamp_ms + 1000
            or price_point.close_timestamp_ms > boundary_ms
            or not 0 <= boundary_ms - price_point.close_timestamp_ms <= _RAW_CLOSE_MAX_STALE_MS
        ):
            raise ValueError("funding_boundary_coverage")
        return AlphaMaxFundingBoundaryLedgerRow(
            symbol=request.symbol,
            boundary_ms=boundary_ms,
            rate_source_timestamp_ms=rate_point.source_timestamp_ms,
            price_row_timestamp_ms=price_point.row_timestamp_ms,
            price_close_timestamp_ms=price_point.close_timestamp_ms,
            qty=qty,
            rate=rate,
            price=price,
        )

    def _prevalidate_batch(
        self,
        requests: Sequence[AlphaMaxFundingBoundaryRequest | Mapping[str, Any]],
        *,
        raw_point_accessor: Any,
    ) -> tuple[tuple[AlphaMaxFundingBoundaryLedgerRow, ...], object]:
        normalized = tuple(self._coerce_request(request) for request in requests)
        self._validate_symbols_before_lookup(normalized)
        owner = self._validate_raw_accessor(raw_point_accessor)
        keys = [(request.symbol, request.boundary_ms) for request in normalized]
        if len(keys) != len(set(keys)):
            raise ValueError("funding_boundary_duplicate")
        existing = {(row.symbol, row.boundary_ms) for row in self._ledger}
        if existing.intersection(keys):
            raise ValueError("funding_boundary_duplicate")
        rows = tuple(
            self._resolve_one(request, raw_point_accessor=raw_point_accessor)
            for request in normalized
        )
        ordered = tuple(sorted(rows, key=lambda row: (row.boundary_ms, row.symbol)))
        last_by_symbol: dict[str, int] = {}
        for row in self._ledger:
            last_by_symbol[row.symbol] = row.boundary_ms
        for row in ordered:
            if row.boundary_ms <= last_by_symbol.get(row.symbol, -1):
                raise ValueError("funding_boundary_not_ascending")
            last_by_symbol[row.symbol] = row.boundary_ms
        return ordered, owner

    def resolve(
        self,
        *,
        symbol: str,
        boundary_ms: int,
        qty: float,
        latest_datetime: datetime,
        raw_point_accessor: Any,
    ) -> AlphaMaxFundingBoundaryLedgerRow:
        """Resolve one immutable row without mutating the committed batch ledger."""
        rows, owner = self._prevalidate_batch(
            (
                AlphaMaxFundingBoundaryRequest(
                    symbol=symbol,
                    boundary_ms=boundary_ms,
                    qty=qty,
                    latest_datetime=latest_datetime,
                ),
            ),
            raw_point_accessor=raw_point_accessor,
        )
        if self._bound_raw_accessor_owner is None:
            object.__setattr__(self, "_bound_raw_accessor_owner", owner)
        return rows[0]

    def resolve_batch(
        self,
        requests: Sequence[AlphaMaxFundingBoundaryRequest | Mapping[str, Any]],
        *,
        raw_point_accessor: Any,
        execution_model: ExecutionModel,
    ) -> tuple[AlphaMaxFundingBoundaryLedgerRow, ...]:
        """Validate a complete batch, then atomically append immutable paid rows."""
        if type(execution_model) is not ExecutionModel:
            raise TypeError("funding_boundary_execution_model_identity_invalid")
        rows, owner = self._prevalidate_batch(
            requests,
            raw_point_accessor=raw_point_accessor,
        )
        paid_rows: list[AlphaMaxFundingBoundaryLedgerRow] = []
        for row in rows:
            payment = execution_model.compute_funding_payment(
                signed_qty=row.qty,
                price=row.price,
                periods=1,
                rate=row.rate,
            )
            if not math.isfinite(payment):
                raise ValueError("funding_boundary_payment_nonfinite")
            paid_rows.append(replace(row, payment=float(payment)))
        committed = tuple(paid_rows)
        if self._bound_raw_accessor_owner is None:
            object.__setattr__(self, "_bound_raw_accessor_owner", owner)
        object.__setattr__(self, "_ledger", (*self._ledger, *committed))
        return committed


def _canonicalize_allocator_inputs(
    component_ids: Sequence[str],
    returns_matrix: Any,
    *,
    per_component_cap: float,
) -> tuple[tuple[str, ...], np.ndarray, float]:
    original_ids = tuple(str(component_id) for component_id in component_ids)
    if len(original_ids) not in {2, 3} or len(original_ids) != len(set(original_ids)):
        raise ValueError("allocator_component_ids_invalid")
    expected_cap = 0.50 if len(original_ids) == 3 else 0.70
    cap = float(per_component_cap)
    if cap != expected_cap:
        raise ValueError("allocator_component_cap_mismatch")
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(original_ids) or matrix.shape[0] < 252:
        raise ValueError("allocator_fit_invalid")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("allocator_fit_invalid")
    order = tuple(sorted(range(len(original_ids)), key=lambda index: original_ids[index]))
    sorted_ids = tuple(original_ids[index] for index in order)
    canonical_matrix = matrix[:, order]
    std = np.std(canonical_matrix, axis=0, ddof=0)
    if not np.all(np.isfinite(std)) or np.any(std <= 1e-12):
        raise ValueError("allocator_fit_invalid")
    return sorted_ids, canonical_matrix, cap


def _round_and_validate_allocation(
    raw_weights: Mapping[str, float],
    ids: tuple[str, ...],
    *,
    cap: float,
) -> dict[str, float]:
    if set(raw_weights) != set(ids):
        raise ValueError("allocator_fit_invalid")
    normalized = {component_id: float(raw_weights[component_id]) for component_id in ids}
    if any(
        not math.isfinite(weight) or weight < 0.0 or weight > cap + 1e-12
        for weight in normalized.values()
    ):
        raise ValueError("allocator_fit_invalid")
    if abs(math.fsum(normalized.values()) - 1.0) > 1e-9:
        raise ValueError("allocator_fit_invalid")
    rounded = {component_id: _round(normalized[component_id], ndigits=10) for component_id in ids}
    residual = 1.0 - math.fsum(rounded.values())
    if not 0.0 <= residual < 1e-9:
        raise ValueError("allocator_rounding_invalid")
    if any(weight > cap + 1e-12 for weight in rounded.values()):
        raise ValueError("allocator_rounding_invalid")
    return rounded


def allocate_alpha_max_equal_weight(
    component_ids: Sequence[str], *, per_component_cap: float
) -> dict[str, float]:
    original_ids = tuple(str(component_id) for component_id in component_ids)
    if len(original_ids) not in {2, 3} or len(original_ids) != len(set(original_ids)):
        raise ValueError("allocator_component_ids_invalid")
    ids = tuple(sorted(original_ids))
    expected_cap = 0.50 if len(ids) == 3 else 0.70
    cap = float(per_component_cap)
    if cap != expected_cap:
        raise ValueError("allocator_component_cap_mismatch")
    raw = dict.fromkeys(ids, 1.0 / float(len(ids)))
    projected = project_simplex_with_upper_bounds(
        raw,
        upper=dict.fromkeys(ids, cap),
        target_sum=1.0,
    )
    return _round_and_validate_allocation(projected, ids, cap=cap)


def allocate_alpha_max_equal_risk(
    component_ids: Sequence[str],
    returns_matrix: Any,
    *,
    per_component_cap: float,
) -> dict[str, float]:
    ids, matrix, cap = _canonicalize_allocator_inputs(
        component_ids,
        returns_matrix,
        per_component_cap=per_component_cap,
    )
    raw = ERCPortfolio(max_iter=10000, tol=1e-10, cov_window=None).allocate(
        list(ids),
        matrix,
        upper=dict.fromkeys(ids, cap),
    )
    return _round_and_validate_allocation(raw, ids, cap=cap)


def allocate_alpha_max_shrunk_hrp(
    component_ids: Sequence[str],
    returns_matrix: Any,
    *,
    per_component_cap: float,
) -> dict[str, float]:
    ids, matrix, cap = _canonicalize_allocator_inputs(
        component_ids,
        returns_matrix,
        per_component_cap=per_component_cap,
    )
    raw = _hrp_weights_with_correlation_shrinkage(
        ids,
        matrix,
        correlation_shrinkage=True,
        corr_threshold=0.60,
    )
    projected = project_simplex_with_upper_bounds(
        raw,
        upper=dict.fromkeys(ids, cap),
        target_sum=1.0,
    )
    return _round_and_validate_allocation(projected, ids, cap=cap)


alpha_max_equal_weight_weights = allocate_alpha_max_equal_weight
alpha_max_equal_risk_weights = allocate_alpha_max_equal_risk
alpha_max_shrunk_hrp_weights = allocate_alpha_max_shrunk_hrp


_COMPONENT_NODES: Final[dict[str, dict[str, Any]]] = {
    "component_carry_1x": {
        "strategy_class": "ResearchOnlyFourHourFundingHarvestCarryStrategy",
        "params": {
            "add_alloc_fraction": 0.5,
            "add_step_atr": 1.0,
            "allow_short": True,
            "atr_period": 14,
            "entry_funding": 0.00005,
            "exit_funding": 0.0,
            "funding_scale": 0.0003,
            "funding_window": 6,
            "max_adds": 2,
            "max_hold_bars": 180,
            "max_order_value": 5000.0,
            "min_price": 0.1,
            "no_fight_roc": 0.06,
            "no_fight_roc_period": 4,
            "target_allocation": 0.3,
            "target_vol": 0.03,
            "trail_atr_mult": 4.0,
            "vol_window": 36,
        },
    },
    "component_near_high_1x": {
        "strategy_class": "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy",
        "params": {
            "allow_short": True,
            "base_allocation": 0.2,
            "high_lookback_bars": 364,
            "max_hold_bars": 0,
            "max_order_value": 400.0,
            "max_symbol_exposure_pct": 0.4,
            "min_history_bars": 60,
            "min_hold_bars": 7,
            "min_price": 0.1,
            "min_symbols": 5,
            "quantile_pct": 0.25,
            "rebalance_bars": 7,
            "stop_loss_pct": 0.1,
            "target_gross_exposure": 1.0,
            "target_vol": 0.2,
            "vol_window": 20,
        },
    },
    "component_trend_1x": {
        "strategy_class": "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
        "params": {
            "adx_min": 20.0,
            "adx_period": 14,
            "allow_short": True,
            "cooldown_bars": 4,
            "efficiency_period": 20,
            "max_hold_bars": 2000,
            "max_order_value": 400.0,
            "min_efficiency": 0.3,
            "min_hold_bars": 36,
            "min_price": 0.1,
            "target_allocation": 0.2,
            "target_vol": 0.2,
            "tsmom_long": 84,
            "tsmom_mid": 56,
            "tsmom_short": 28,
            "vol_persist_fast": 16,
            "vol_persist_max": 1.5,
            "vol_persist_slow": 64,
            "vol_window": 56,
        },
    },
}

_RESOLVABLE_ROW_SHA256: Final[dict[str, str]] = {
    "component_carry_1x": "22f52ed7093fa792dafbac47a0261e54e62257bc5704f0b6f3637deac30d3dbc",
    "component_near_high_1x": "f7a2177a7e0f60003b1690af7ec862589dcef483d9fe27ba29f26c77da9ae866",
    "component_trend_1x": "455cac5ea1ac87ba26bac93d743c3b3ef63a67db886c26c08ca19ec087143108",
    "full_equal_risk_1x": "11e3813b95be95208f8efb8b26abb1b9c6f86a6944828a3314fce10e333e8a8d",
    "full_equal_risk_scaled": "c8c1d112c9c0a3a8fbd78de76f39989ba4acc40d82c68180a4b2fc4d853404fa",
    "full_equal_weight_1x": "90a822e37b2755a06445baef676adb0f850927860976fa9799e2207e7f4450a5",
    "full_shrunk_hrp_1x": "26f0903ac9508b39b46bfc43cec59325bc3192983c8ade7d5d72287070f3309f",
    "full_shrunk_hrp_scaled": "57bfd3c4dec985b5c5d1fc4bb738bd31a408e166e181afbcf6473e86020e59c0",
    "loo_equal_risk_omit_carry_1x": "604677396167d9edcffd14cbbea23049bdd302b836f5e00a90386f65af9a5b6c",
    "loo_equal_risk_omit_near_high_1x": "a49997a58f14336d757417a490e1705247da07d2f2eb7a8a747ebbaca30aa863",
    "loo_equal_risk_omit_trend_1x": "01e67b0f90917e1c775b32022a494657fa16f1de3f21e4fbb2deb152c4dd1fe5",
    "loo_equal_weight_omit_carry_1x": "df09ad7ae77095db7185f179e455384d2e0f6c3cac1a18e73e020358936d86bb",
    "loo_equal_weight_omit_near_high_1x": "3b46711f2cdfaa6ce7a31692b6623dc758df9a08cb243234c04995b92d2b9a14",
    "loo_equal_weight_omit_trend_1x": "d75edb200de748f50788d15081a99ae3e67951c2a8041fe2f5f017fc625f71e7",
    "loo_shrunk_hrp_omit_carry_1x": "481b4bde118a639905e77a79745e19d35157d9d8ace0e2a2c5dfe6aca33b5953",
    "loo_shrunk_hrp_omit_near_high_1x": "f5119cd99d13a780a9a3dc96c02e6bc6027a7b86995a93c05ba8b327f0a0ee13",
    "loo_shrunk_hrp_omit_trend_1x": "5717eca1c6cbf7f73800288532744b63b5dfd3964f0014f6e064a21d547ae7ab",
}
_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "kind",
        "row_id",
        "implementation",
        "source_id",
        "timeframe",
        "symbols",
        "params",
        "members",
        "allocation",
        "gross",
        "omission",
    }
)


@dataclass(frozen=True, slots=True)
class AlphaMaxManifestMaterialization:
    """Immutable path/hash/byte seal returned by the sole materializer."""

    path: str
    sha256: str
    canonical_bytes: bytes
    strategy_params: Mapping[str, Any]

    @property
    def payload(self) -> dict[str, Any]:
        parsed = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(parsed, dict):  # pragma: no cover - construction invariant
            raise AssertionError("materialized manifest was not an object")
        return parsed

    @property
    def manifest_path(self) -> str:
        return self.path

    @property
    def manifest_sha256(self) -> str:
        return self.sha256

    @property
    def manifest_bytes(self) -> bytes:
        return self.canonical_bytes

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "path": self.path,
            "manifest_path": self.path,
            "sha256": self.sha256,
            "manifest_sha256": self.sha256,
            "bytes": self.canonical_bytes,
            "canonical_bytes": self.canonical_bytes,
            "manifest_bytes": self.canonical_bytes,
            "payload": self.payload,
            "strategy_params": self.strategy_params,
        }
        return aliases[key]


def _validate_frozen_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    materialized_row = dict(row)
    if set(materialized_row) != _ROW_KEYS:
        raise ValueError("alpha_max_row_key_set_mismatch")
    row_id = str(materialized_row.get("row_id") or "")
    expected_sha = _RESOLVABLE_ROW_SHA256.get(row_id)
    if expected_sha is None:
        raise ValueError("alpha_max_row_not_materializable")
    actual_sha = _sha256_bytes(_canonical_json_bytes(materialized_row, newline=False))
    if actual_sha != expected_sha:
        raise ValueError("alpha_max_row_registry_mismatch")
    if row_id in _COMPONENT_NODES:
        members = (row_id,)
    else:
        raw_members = materialized_row["members"]
        if not isinstance(raw_members, list) or raw_members != sorted(raw_members):
            raise ValueError("alpha_max_row_members_invalid")
        members = tuple(raw_members)
    if not members or any(member not in _COMPONENT_NODES for member in members):
        raise ValueError("alpha_max_row_member_not_component")
    return materialized_row, members


def _validate_alpha_max_resolved_gross(row: Mapping[str, Any], value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("alpha_max_resolved_gross_invalid")
    try:
        gross = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha_max_resolved_gross_invalid") from exc
    if not math.isfinite(gross) or gross <= 0.0:
        raise ValueError("alpha_max_resolved_gross_invalid")

    rule = row.get("gross")
    if not isinstance(rule, dict):  # frozen row hash makes this defensive
        raise ValueError("alpha_max_resolved_gross_rule_invalid")
    method = rule.get("method")
    if method == "fixed":
        expected = float(rule.get("value"))
        if gross != expected:
            raise ValueError("alpha_max_resolved_gross_invalid_fixed_mismatch")
        return gross
    if method == "validation_mdd_target":
        # The canonical row hash has already frozen target_mdd, epsilon, and the
        # positive 1x sibling requirement; this argument can only prove the
        # resolved scalar remains inside that rule's immutable clip interval.
        clip_min = float(rule.get("clip_min"))
        clip_max = float(rule.get("clip_max"))
        if not clip_min <= gross <= clip_max:
            raise ValueError("alpha_max_resolved_gross_invalid_scaled_clip")
        return gross
    raise ValueError("alpha_max_resolved_gross_rule_invalid")


def _validate_run_owned_phase(output_root: str | os.PathLike[str], phase: str) -> Path:
    if phase not in _MANIFEST_PHASES:
        raise ValueError("alpha_max_manifest_phase_invalid")
    root_path = Path(_require_explicit_canonical_path(output_root, field="alpha_max_output_root"))
    manifests_path = root_path / "manifests"
    phase_paths = {name: manifests_path / name for name in _MANIFEST_PHASES}
    for field, path in (
        ("alpha_max_output_root", root_path),
        ("alpha_max_manifests_dir", manifests_path),
        *((f"alpha_max_phase_dir_{name}", path) for name, path in phase_paths.items()),
    ):
        try:
            status = path.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"{field}_missing") from exc
        if path.is_symlink() or not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"{field}_not_owned_directory")
        if status.st_uid != os.geteuid() or str(path.resolve(strict=True)) != str(path):
            raise ValueError(f"{field}_not_owned_directory")
    if {entry.name for entry in manifests_path.iterdir()} != set(_MANIFEST_PHASES):
        raise ValueError("alpha_max_manifests_parent_not_run_owned")
    return phase_paths[phase]


def _write_new_manifest(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    created = False
    try:
        fd = os.open(path, flags, 0o600)
        created = True
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                written += os.write(fd, view[written:])
            os.fsync(fd)
        finally:
            os.close(fd)
    except FileExistsError as exc:
        raise ValueError("alpha_max_manifest_target_exists") from exc
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def materialize_alpha_max_manifest(
    row: Mapping[str, Any],
    resolved_weights: Mapping[str, float],
    resolved_gross: float,
    phase: str,
    config_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    candidate_symbols: Sequence[str],
    admitted_symbols: tuple[str, ...],
    admission_manifest_sha256: str,
) -> AlphaMaxManifestMaterialization:
    """Materialize one immutable canonical actual-engine manifest.

    Incumbent and diagnostic rows are absent from the sealed resolvable-row hash
    table and therefore fail before any config read or target creation.
    """
    frozen_row, members = _validate_frozen_row(row)
    candidates = tuple(candidate_symbols)
    admitted = validate_alpha_max_admitted_symbols(candidates, admitted_symbols)
    if type(admitted_symbols) is not tuple or admitted is not admitted_symbols:
        raise TypeError("alpha_max_admitted_symbols_must_be_frozen_tuple")
    admission_sha = _require_sha256(
        admission_manifest_sha256,
        field="alpha_max_admission_manifest_sha256",
    )

    weights = {str(key): float(value) for key, value in resolved_weights.items()}
    if set(weights) != set(members) or tuple(sorted(weights)) != members:
        raise ValueError("alpha_max_resolved_weight_coverage_mismatch")
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("alpha_max_resolved_weight_invalid")
    allocation_residual = 1.0 - math.fsum(weights.values())
    if not 0.0 <= allocation_residual < 1e-9:
        raise ValueError("allocator_rounding_invalid")

    gross = _validate_alpha_max_resolved_gross(frozen_row, resolved_gross)
    allocation = frozen_row["allocation"]
    if not isinstance(allocation, dict):  # frozen hash makes this defensive
        raise ValueError("alpha_max_row_allocation_invalid")
    cap = float(allocation["per_component_cap"])
    if any(weight > cap + 1e-12 for weight in weights.values()):
        raise ValueError("alpha_max_resolved_weight_cap_breach")
    fixed_weights = allocation.get("fixed_weights")
    if fixed_weights is not None:
        if type(fixed_weights) is not dict:  # frozen hash makes this defensive
            raise ValueError("alpha_max_row_fixed_weights_invalid")
        expected_weights = {
            member: _round(float(fixed_weights[member]), ndigits=10) for member in members
        }
        if weights != expected_weights:
            raise ValueError("alpha_max_resolved_weight_fixed_mismatch")

    config_canonical = _require_explicit_canonical_path(config_path, field="alpha_max_config_path")
    config_receipt, _config_bytes = read_artifact_bytes(
        config_canonical,
        artifact_id="alpha_max_config",
    )
    if config_receipt.requested_path != config_receipt.canonical_path:
        raise ValueError("alpha_max_config_path_identity_mismatch")

    phase_dir = _validate_run_owned_phase(output_root, phase)
    row_id = str(frozen_row["row_id"])
    if not row_id or Path(row_id).name != row_id or row_id in {".", ".."}:
        raise ValueError("alpha_max_row_id_path_invalid")
    manifest_path = phase_dir / f"{row_id}.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("alpha_max_manifest_target_exists")
    if manifest_path.parent != phase_dir or manifest_path.resolve(strict=False).parent != phase_dir:
        raise ValueError("alpha_max_manifest_path_escape")

    use_train_validation = phase == "prelock_final_refit" and allocation["method"] in {
        "equal_risk",
        "shrunk_hrp",
    }
    selection_inputs = ["train", "validation"] if use_train_validation else ["train"]
    correlation_source = (
        "alpha_max_train_validation_daily_net_returns"
        if use_train_validation
        else "alpha_max_train_daily_net_returns"
    )
    optimizer_provenance = {"selection_inputs": selection_inputs}
    correlation_provenance = {
        "selection_inputs": selection_inputs,
        "ready": True,
        "source": correlation_source,
    }
    false_flags = dict.fromkeys((*_REAL_MONEY_KEYS, *_FORBIDDEN_OOS_KEYS), False)

    children: list[dict[str, Any]] = []
    for member in members:
        component = _COMPONENT_NODES[member]
        component_candidates = list(ALPHA_MAX_CANDIDATE_SYMBOLS)
        if tuple(component_candidates) != candidates:
            raise ValueError("alpha_max_component_candidate_symbols_mismatch")
        leaf_gross = weights[member] * gross
        leaf_cap = cap * gross
        child: dict[str, Any] = {
            "candidate_id": member,
            "name": member,
            "strategy_class": component["strategy_class"],
            "candidate_symbols": component_candidates,
            "symbols": list(admitted),
            "params": component["params"],
            "weight": leaf_gross,
            "leaf_gross": leaf_gross,
            "leaf_gross_cap": leaf_cap,
            "netting_group": member,
            "netting_group_gross_cap": leaf_cap,
            "source_artifact_id": "alpha_max_config",
            "ready": True,
            "portfolio_ready": True,
            **false_flags,
            "no_current_fold_oos_provenance": True,
            "train_validation_optimizer_provenance": True,
            "lagged_completed_shadow_optimizer_provenance": False,
            "optimizer_provenance": optimizer_provenance,
            "correlation_input_provenance": correlation_provenance,
        }
        if set(child) != ALPHA_MAX_MANIFEST_CHILD_KEYS:
            raise AssertionError("alpha-max child schema drift")
        if child["weight"] != child["leaf_gross"] or child["leaf_gross"] > leaf_cap + 1e-12:
            raise ValueError("alpha_max_leaf_gross_identity_invalid")
        children.append(child)
    children.sort(key=lambda child: child["candidate_id"])

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_max_engine_portfolio_manifest.v1",
        "candidate_symbols": list(candidates),
        "admitted_symbols": list(admitted),
        "admission_manifest_sha256": admission_sha,
        **false_flags,
        "gross_cap": gross,
        "cash_weight": max(0.0, 1.0 - gross * math.fsum(weights.values())),
        "allocation_method": allocation["method"],
        "optimizer_provenance": optimizer_provenance,
        "correlation_input_provenance": correlation_provenance,
        "source_artifacts": [
            {
                "id": "alpha_max_config",
                "path": config_receipt.canonical_path,
                "sha256": config_receipt.sha256,
                "max_age_hours": 876000,
                "ready": True,
                "portfolio_ready": True,
            }
        ],
        "children": children,
    }
    if set(payload) != ALPHA_MAX_MANIFEST_TOP_LEVEL_KEYS:
        raise AssertionError("alpha-max manifest schema drift")
    canonical_bytes = _canonical_json_bytes(payload, newline=True)
    manifest_sha = _sha256_bytes(canonical_bytes)
    _write_new_manifest(manifest_path, canonical_bytes)
    return AlphaMaxManifestMaterialization(
        path=str(manifest_path),
        sha256=manifest_sha,
        canonical_bytes=canonical_bytes,
        strategy_params=MappingProxyType(
            {
                "portfolio_mode": f"manifest:{manifest_path}",
                "decision_cadence_seconds": 1,
            }
        ),
    )


_ALPHA_MAX_COST_CELLS: Final[frozenset[int]] = frozenset({10, 15, 20, 30})
_ALPHA_MAX_SLIPPAGE_BY_COST: Final[dict[int, float]] = {
    10: 0.0005,
    15: 0.001,
    20: 0.0015,
    30: 0.0025,
}
_ALPHA_MAX_INITIAL_CAPITAL: Final[float] = 10_000.0
_ALPHA_MAX_PRIOR_COMMIT: Final[str] = "252910e54e280cc593365484cbc99d6ca87893f9"
_ALPHA_MAX_PRIOR_PATH: Final[str] = (
    "var/reports/ultragoal_full_pool_strategy/g004_frozen_candidate_manifest.json"
)
_ALPHA_MAX_PRIOR_BLOB_OID: Final[str] = "1bb06b6e9d4ca5a82af4686001b880db9709d9b8"
_ALPHA_MAX_PRIOR_FILE_SHA256: Final[str] = (
    "f2c86ae7bb9f9719143fa0b11e73c68ad021160aeac03a0aa5c6fa93636d57b6"
)
_ALPHA_MAX_PRIOR_KEY_SET_SHA256: Final[str] = (
    "3b078011040f89e8d788b2cef9214c58f687221104381e26a688a7f8cdbddd78"
)
_ALPHA_MAX_CURRENT_REGISTRY_SHA256: Final[str] = (
    "cfe3a04620c52cc235d6f1cda1cac617ba30cd7327c753fc2f620d8250d51a4e"
)
_ALPHA_MAX_CURRENT_KEY_SET_SHA256: Final[str] = (
    "3a4791cf353abcb82f9717ce89ee16b9d73d84f431d5b058135046c2ba8e332b"
)
_ALPHA_MAX_PRIOR_CANDIDATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "candidate_id",
        "family",
        "metadata",
        "name",
        "notes",
        "params",
        "strategy",
        "strategy_class",
        "strategy_timeframe",
        "symbols",
        "tags",
        "timeframe",
    }
)
_ALPHA_MAX_PRIOR_COSMETIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "availability",
        "candidate_id",
        "costs",
        "family",
        "hash",
        "hashes",
        "name",
        "notes",
        "pass",
        "path",
        "paths",
        "rank",
        "returns",
        "status",
        "tags",
        "timestamp",
        "timestamps",
    }
)
_ALPHA_MAX_PRIOR_BEHAVIOR_KEYS: Final[frozenset[str]] = _ALPHA_MAX_PRIOR_CANDIDATE_KEYS.difference(
    _ALPHA_MAX_PRIOR_COSMETIC_KEYS
)
_ALPHA_MAX_PRIOR_NODE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "allocation",
        "behavior_metadata",
        "gross",
        "implementation",
        "kind",
        "members",
        "omission",
        "params",
        "schema",
        "symbols",
        "timeframe",
    }
)
_ALPHA_MAX_CURRENT_REGISTRY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "candidate_symbols",
        "canonicalization",
        "current_key_set_sha256",
        "current_node_count",
        "nodes",
        "schema",
    }
)
_ALPHA_MAX_CANONICALIZATION: Final[str] = (
    'json.dumps(node,sort_keys=True,separators=(",",":"),ensure_ascii=False,'
    'allow_nan=False).encode("utf-8")'
)


def _alpha_max_finite_number(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"alpha_max_{field}_invalid")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"alpha_max_{field}_invalid")
    if positive and parsed <= 0.0:
        raise ValueError(f"alpha_max_{field}_invalid")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"alpha_max_{field}_invalid")
    return parsed


def alpha_max_common_rng_seed_payload(split_or_fold_id: str, nominal_cost_bps: int) -> bytes:
    """Return the exact common-random-number payload for one split/cost replay."""
    if (
        type(split_or_fold_id) is not str
        or not split_or_fold_id
        or split_or_fold_id != split_or_fold_id.strip()
        or "\0" in split_or_fold_id
    ):
        raise ValueError("alpha_max_rng_split_or_fold_id_invalid")
    if type(nominal_cost_bps) is not int or nominal_cost_bps not in _ALPHA_MAX_COST_CELLS:
        raise ValueError("alpha_max_rng_nominal_cost_bps_invalid")
    try:
        split_bytes = split_or_fold_id.encode("utf-8")
        cost_bytes = str(nominal_cost_bps).encode("ascii")
    except UnicodeError as exc:  # pragma: no cover - guarded by Python strings
        raise ValueError("alpha_max_rng_payload_encoding_invalid") from exc
    return b"alpha_max_20260710\0" + split_bytes + b"\0" + cost_bytes


def alpha_max_common_rng_seed(split_or_fold_id: str, nominal_cost_bps: int) -> int:
    """Derive the exact positive 31-bit common seed; row identity is not an input."""
    payload = alpha_max_common_rng_seed_payload(split_or_fold_id, nominal_cost_bps)
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647
    return 1 if seed == 0 else seed


@dataclass(frozen=True, slots=True)
class AlphaMaxEquityEndpoint:
    """One observed portfolio-equity value at an exact UTC four-hour endpoint."""

    timestamp: datetime
    equity: float


@dataclass(frozen=True, slots=True)
class AlphaMaxPrimaryReturnStream:
    """Exact complete UTC 4h endpoint/equity/arithmetic-return evidence."""

    endpoint_timestamps: tuple[datetime, ...]
    endpoint_equities: tuple[float, ...]
    returns: tuple[float, ...]
    initial_capital: float
    periods_per_year: int
    calendar_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_primary_return_stream.v1",
            "calendar_sha256": self.calendar_sha256,
            "endpoint_equities": list(self.endpoint_equities),
            "endpoint_timestamps": [
                value.isoformat().replace("+00:00", "Z") for value in self.endpoint_timestamps
            ],
            "initial_capital": self.initial_capital,
            "periods_per_year": self.periods_per_year,
            "returns": list(self.returns),
        }


def _alpha_max_utc_4h_timestamp(value: Any, *, field: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"alpha_max_{field}_not_utc")
    normalized = value.astimezone(UTC)
    if (
        normalized.hour % 4 != 0
        or normalized.minute != 0
        or normalized.second != 0
        or normalized.microsecond != 0
    ):
        raise ValueError(f"alpha_max_{field}_not_4h_endpoint")
    return normalized


def _alpha_max_validate_expected_calendar(values: Sequence[datetime]) -> tuple[datetime, ...]:
    expected = tuple(
        _alpha_max_utc_4h_timestamp(value, field="expected_endpoint_timestamp") for value in values
    )
    if not expected:
        raise ValueError("alpha_max_expected_endpoint_calendar_empty")
    for previous, current in pairwise(expected):
        if current <= previous:
            raise ValueError("alpha_max_expected_endpoint_calendar_not_strict")
        if current - previous != timedelta(hours=4):
            raise ValueError("alpha_max_expected_endpoint_calendar_incomplete")
    return expected


def _alpha_max_calendar_sha256(timestamps: Sequence[datetime]) -> str:
    payload = [value.isoformat().replace("+00:00", "Z") for value in timestamps]
    return _sha256_bytes(_canonical_json_bytes(payload, newline=False))


def build_alpha_max_primary_return_stream(
    endpoints: Sequence[AlphaMaxEquityEndpoint],
    expected_endpoints: Sequence[datetime],
    *,
    initial_capital: float = _ALPHA_MAX_INITIAL_CAPITAL,
) -> AlphaMaxPrimaryReturnStream:
    """Build the complete primary stream without fill, interpolation, or truncation."""
    capital = _alpha_max_finite_number(
        initial_capital,
        field="primary_initial_capital",
        positive=True,
    )
    if capital != _ALPHA_MAX_INITIAL_CAPITAL:
        raise ValueError("alpha_max_primary_initial_capital_mismatch")
    expected = _alpha_max_validate_expected_calendar(expected_endpoints)
    observed = tuple(endpoints)
    if len(observed) != len(expected):
        raise ValueError("alpha_max_primary_endpoint_count_mismatch")

    timestamps: list[datetime] = []
    equities: list[float] = []
    for index, endpoint in enumerate(observed):
        if type(endpoint) is not AlphaMaxEquityEndpoint:
            raise TypeError("alpha_max_primary_endpoint_schema_invalid")
        timestamp = _alpha_max_utc_4h_timestamp(
            endpoint.timestamp,
            field="primary_endpoint_timestamp",
        )
        if timestamp != expected[index]:
            raise ValueError("alpha_max_primary_endpoint_calendar_mismatch")
        equity = _alpha_max_finite_number(
            endpoint.equity,
            field="primary_endpoint_equity",
            positive=True,
        )
        timestamps.append(timestamp)
        equities.append(equity)

    returns: list[float] = []
    prior = capital
    for equity in equities:
        value = (equity / prior) - 1.0
        if not math.isfinite(value):  # pragma: no cover - finite positive quotient
            raise ValueError("alpha_max_primary_return_nonfinite")
        returns.append(value)
        prior = equity
    return AlphaMaxPrimaryReturnStream(
        endpoint_timestamps=tuple(timestamps),
        endpoint_equities=tuple(equities),
        returns=tuple(returns),
        initial_capital=capital,
        periods_per_year=ALPHA_MAX_PERIODS_PER_YEAR,
        calendar_sha256=_alpha_max_calendar_sha256(timestamps),
    )


def _validate_alpha_max_primary_stream(
    stream: AlphaMaxPrimaryReturnStream,
) -> AlphaMaxPrimaryReturnStream:
    if type(stream) is not AlphaMaxPrimaryReturnStream:
        raise TypeError("alpha_max_primary_return_stream_identity_invalid")
    if stream.initial_capital != _ALPHA_MAX_INITIAL_CAPITAL:
        raise ValueError("alpha_max_primary_initial_capital_mismatch")
    if stream.periods_per_year != ALPHA_MAX_PERIODS_PER_YEAR:
        raise ValueError("alpha_max_primary_annualization_mismatch")
    expected = _alpha_max_validate_expected_calendar(stream.endpoint_timestamps)
    if len(expected) != len(stream.endpoint_equities) or len(expected) != len(stream.returns):
        raise ValueError("alpha_max_primary_stream_length_mismatch")
    equities = tuple(
        _alpha_max_finite_number(value, field="primary_endpoint_equity", positive=True)
        for value in stream.endpoint_equities
    )
    recalculated: list[float] = []
    prior = stream.initial_capital
    for equity in equities:
        recalculated.append((equity / prior) - 1.0)
        prior = equity
    if tuple(recalculated) != stream.returns:
        raise ValueError("alpha_max_primary_return_identity_mismatch")
    if stream.calendar_sha256 != _alpha_max_calendar_sha256(expected):
        raise ValueError("alpha_max_primary_calendar_sha256_mismatch")
    return stream


def alpha_max_type7_quantile(values: Sequence[float], probability: float) -> float:
    """Exact Hyndman-Fan type-7 quantile over a finite nonempty sample."""
    p = _alpha_max_finite_number(probability, field="type7_probability", nonnegative=True)
    if p > 1.0:
        raise ValueError("alpha_max_type7_probability_invalid")
    ordered = sorted(_alpha_max_finite_number(value, field="type7_observation") for value in values)
    if not ordered:
        raise ValueError("alpha_max_type7_observations_empty")
    h = (len(ordered) - 1) * p
    lower = math.floor(h)
    fraction = h - lower
    upper = min(lower + 1, len(ordered) - 1)
    return ((1.0 - fraction) * ordered[lower]) + (fraction * ordered[upper])


def alpha_max_full_event_mdd(
    event_equities: Sequence[float],
    *,
    initial_capital: float = _ALPHA_MAX_INITIAL_CAPITAL,
) -> float:
    """Full-event maximum drawdown from the flat initial-capital peak."""
    capital = _alpha_max_finite_number(
        initial_capital,
        field="full_event_initial_capital",
        positive=True,
    )
    if capital != _ALPHA_MAX_INITIAL_CAPITAL:
        raise ValueError("alpha_max_full_event_initial_capital_mismatch")
    values = tuple(event_equities)
    if not values:
        raise ValueError("alpha_max_full_event_equities_empty")
    peak = capital
    maximum = 0.0
    for raw in values:
        equity = _alpha_max_finite_number(raw, field="full_event_equity", positive=True)
        peak = max(peak, equity)
        maximum = max(maximum, 1.0 - (equity / peak))
    return maximum


def alpha_max_drawdown_duration(
    stream: AlphaMaxPrimaryReturnStream,
) -> tuple[int, int]:
    """Return maximum below-peak endpoint count and the exact four-hour duration."""
    validated = _validate_alpha_max_primary_stream(stream)
    peak = validated.initial_capital
    current = 0
    maximum = 0
    for equity in validated.endpoint_equities:
        if equity >= peak:
            peak = equity
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum, maximum * 4


@dataclass(frozen=True, slots=True)
class AlphaMaxMetricStatistics:
    """Canonical optimizer metrics plus strictly separate full-event diagnostics."""

    canonical_metrics: Mapping[str, float]
    primary_return_stream_sha256: str
    streaming_equity_sha256: str
    full_event_event_count: int
    uncapped_full_event_drawdown: float
    full_event_mdd: float
    reporting_4h_mdd: float
    gate_mdd: float
    ruin_detected: bool
    drawdown_duration_endpoints: int
    drawdown_duration_hours: int
    value_at_risk_5pct_type7: float
    expected_shortfall_5pct: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_metric_statistics.v1",
            "canonical_metrics": dict(self.canonical_metrics),
            "drawdown_duration_endpoints": self.drawdown_duration_endpoints,
            "drawdown_duration_hours": self.drawdown_duration_hours,
            "expected_shortfall_5pct": self.expected_shortfall_5pct,
            "full_event_event_count": self.full_event_event_count,
            "full_event_mdd": self.full_event_mdd,
            "gate_mdd": self.gate_mdd,
            "primary_return_stream_sha256": self.primary_return_stream_sha256,
            "reporting_4h_mdd": self.reporting_4h_mdd,
            "ruin_detected": self.ruin_detected,
            "streaming_equity_sha256": self.streaming_equity_sha256,
            "uncapped_full_event_drawdown": self.uncapped_full_event_drawdown,
            "value_at_risk_5pct_type7": self.value_at_risk_5pct_type7,
        }


def compute_alpha_max_metric_statistics(
    stream: AlphaMaxPrimaryReturnStream,
    full_event_equity: AlphaMaxStreamingEquityEvidence,
) -> AlphaMaxMetricStatistics:
    """Call the sole canonical metric primitive and add non-overlapping diagnostics."""
    validated = _validate_alpha_max_primary_stream(stream)
    if type(full_event_equity) is not AlphaMaxStreamingEquityEvidence:
        raise TypeError("alpha_max_streaming_equity_evidence_identity_invalid")
    _validate_alpha_max_streaming_equity_evidence(full_event_equity)
    if full_event_equity.initial_capital != validated.initial_capital or not math.isclose(
        full_event_equity.ending_equity,
        validated.endpoint_equities[-1],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("alpha_max_metric_streaming_equity_binding_mismatch")
    returns = np.asarray(validated.returns, dtype=np.float64)
    raw_metrics = optimizer_core.metrics(
        returns,
        periods_per_year=ALPHA_MAX_PERIODS_PER_YEAR,
    )
    expected_keys = {
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "volatility",
    }
    if type(raw_metrics) is not dict or set(raw_metrics) != expected_keys:
        raise ValueError("alpha_max_canonical_metrics_schema_mismatch")
    canonical = {
        key: _alpha_max_finite_number(value, field=f"canonical_metric_{key}")
        for key, value in raw_metrics.items()
    }
    reporting_mdd = canonical["max_drawdown"]
    if reporting_mdd < 0.0:
        raise ValueError("alpha_max_canonical_metric_max_drawdown_invalid")
    full_mdd = full_event_equity.full_event_mdd
    duration_count, duration_hours = alpha_max_drawdown_duration(validated)
    var_5pct = alpha_max_type7_quantile(validated.returns, 0.05)
    tail_count = max(1, math.ceil(0.05 * len(validated.returns)))
    worst = sorted(enumerate(validated.returns), key=lambda item: (item[1], item[0]))[:tail_count]
    expected_shortfall = math.fsum(value for _, value in worst) / tail_count
    return AlphaMaxMetricStatistics(
        canonical_metrics=MappingProxyType(canonical),
        primary_return_stream_sha256=_sha256_bytes(
            _canonical_json_bytes(validated.to_payload(), newline=True)
        ),
        streaming_equity_sha256=full_event_equity.sha256,
        full_event_event_count=full_event_equity.event_count,
        uncapped_full_event_drawdown=full_event_equity.uncapped_full_event_drawdown,
        full_event_mdd=full_mdd,
        reporting_4h_mdd=reporting_mdd,
        gate_mdd=max(full_mdd, reporting_mdd),
        ruin_detected=full_event_equity.ruin_detected,
        drawdown_duration_endpoints=duration_count,
        drawdown_duration_hours=duration_hours,
        value_at_risk_5pct_type7=var_5pct,
        expected_shortfall_5pct=expected_shortfall,
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxTurnoverRPTDiagnostics:
    """Report-only turnover and return-per-turnover evidence."""

    turnover_notional: float
    turnover_multiple: float
    rpt_bps: float | None
    undefined_reason: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_turnover_rpt.v1",
            "report_only": True,
            "rpt_bps": self.rpt_bps,
            "turnover_multiple": self.turnover_multiple,
            "turnover_notional": self.turnover_notional,
            "undefined_reason": self.undefined_reason,
        }


def compute_alpha_max_turnover_rpt(
    applied_records: Sequence[Mapping[str, Any]],
    *,
    initial_capital: float,
    ending_equity: float,
) -> AlphaMaxTurnoverRPTDiagnostics:
    """Compute report-only turnover/RPT from exact positive applied-fill records."""
    capital = _alpha_max_finite_number(
        initial_capital,
        field="turnover_initial_capital",
        positive=True,
    )
    ending = _alpha_max_finite_number(ending_equity, field="turnover_ending_equity")
    notionals: list[float] = []
    for record in applied_records:
        if type(record) is not dict or set(record) != {"applied_qty", "fill_price"}:
            raise ValueError("alpha_max_turnover_record_schema_mismatch")
        quantity = _alpha_max_finite_number(
            record["applied_qty"],
            field="turnover_applied_qty",
            nonnegative=True,
        )
        price = _alpha_max_finite_number(
            record["fill_price"],
            field="turnover_fill_price",
            positive=True,
        )
        if quantity > 0.0:
            notionals.append(abs(quantity * price))
    turnover = math.fsum(notionals)
    multiple = turnover / capital
    if turnover == 0.0:
        return AlphaMaxTurnoverRPTDiagnostics(
            turnover_notional=0.0,
            turnover_multiple=0.0,
            rpt_bps=None,
            undefined_reason="undefined_zero_turnover",
        )
    rpt = 10_000.0 * (ending - capital) / turnover
    if not math.isfinite(rpt):  # pragma: no cover - finite inputs and positive denominator
        raise ValueError("alpha_max_turnover_rpt_nonfinite")
    return AlphaMaxTurnoverRPTDiagnostics(
        turnover_notional=turnover,
        turnover_multiple=multiple,
        rpt_bps=rpt,
        undefined_reason=None,
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxCapacityDiagnostics:
    """Report-only finite-positive order-capacity proxy summary."""

    observation_count: int
    capacity_proxy_equity_usdt: Mapping[str, float] | None
    undefined_reason: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_capacity_diagnostics.v1",
            "capacity_proxy_equity_usdt": (
                None
                if self.capacity_proxy_equity_usdt is None
                else dict(self.capacity_proxy_equity_usdt)
            ),
            "observation_count": self.observation_count,
            "report_only": True,
            "undefined_reason": self.undefined_reason,
        }


def compute_alpha_max_capacity_diagnostics(
    requested_orders: Sequence[Mapping[str, Any]],
) -> AlphaMaxCapacityDiagnostics:
    """Compute the frozen 10%-participation capacity proxy for positive requests."""
    observations: list[float] = []
    required_keys = {"bar_volume", "raw_price", "equity_before", "requested_qty"}
    for record in requested_orders:
        if type(record) is not dict or set(record) != required_keys:
            raise ValueError("alpha_max_capacity_record_schema_mismatch")
        quantity = _alpha_max_finite_number(
            record["requested_qty"],
            field="capacity_requested_qty",
            nonnegative=True,
        )
        raw_price = _alpha_max_finite_number(
            record["raw_price"],
            field="capacity_raw_price",
            positive=True,
        )
        bar_volume = _alpha_max_finite_number(
            record["bar_volume"],
            field="capacity_bar_volume",
            nonnegative=True,
        )
        equity_before = _alpha_max_finite_number(
            record["equity_before"],
            field="capacity_equity_before",
            positive=True,
        )
        if quantity <= 0.0:
            continue
        capacity = 0.10 * (bar_volume * raw_price) * equity_before / abs(quantity * raw_price)
        if not math.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("alpha_max_capacity_observation_not_finite_positive")
        observations.append(capacity)
    if not observations:
        return AlphaMaxCapacityDiagnostics(
            observation_count=0,
            capacity_proxy_equity_usdt=None,
            undefined_reason="undefined_no_positive_order",
        )
    summary = MappingProxyType(
        {
            "minimum": min(observations),
            "p10_type7": alpha_max_type7_quantile(observations, 0.10),
            "median_type7": alpha_max_type7_quantile(observations, 0.50),
        }
    )
    return AlphaMaxCapacityDiagnostics(
        observation_count=len(observations),
        capacity_proxy_equity_usdt=summary,
        undefined_reason=None,
    )


def _alpha_max_duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("alpha_max_json_duplicate_key")
        parsed[key] = value
    return parsed


def _alpha_max_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"alpha_max_json_nonfinite_constant:{value}")


def _alpha_max_strict_json_object(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_alpha_max_duplicate_rejecting_object,
            parse_constant=_alpha_max_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"alpha_max_{field}_json_invalid") from exc
    if type(parsed) is not dict:
        raise ValueError(f"alpha_max_{field}_json_not_object")
    return parsed


def _alpha_max_git_command(repo_root: str, *args: str) -> bytes:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", repo_root, *args],
            check=True,
            capture_output=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("alpha_max_prior_trial_inventory_mismatch") from exc
    return completed.stdout


def read_alpha_max_prior_trial_blob(repo_root: str | os.PathLike[str]) -> bytes:
    """Read the sole frozen G004 prior inventory by immutable Git object identity."""
    root = _require_explicit_canonical_path(repo_root, field="alpha_max_repo_root")
    top_level = _alpha_max_git_command(root, "rev-parse", "--show-toplevel").decode().strip()
    if str(Path(top_level).resolve(strict=True)) != root:
        raise ValueError("alpha_max_prior_trial_inventory_mismatch")
    resolved_oid = (
        _alpha_max_git_command(
            root,
            "rev-parse",
            f"{_ALPHA_MAX_PRIOR_COMMIT}:{_ALPHA_MAX_PRIOR_PATH}",
        )
        .decode("ascii")
        .strip()
    )
    if resolved_oid != _ALPHA_MAX_PRIOR_BLOB_OID:
        raise ValueError("alpha_max_prior_trial_inventory_mismatch")
    payload = _alpha_max_git_command(root, "cat-file", "blob", _ALPHA_MAX_PRIOR_BLOB_OID)
    git_oid = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
    if (
        git_oid != _ALPHA_MAX_PRIOR_BLOB_OID
        or _sha256_bytes(payload) != _ALPHA_MAX_PRIOR_FILE_SHA256
    ):
        raise ValueError("alpha_max_prior_trial_inventory_mismatch")
    return payload


def normalize_alpha_max_prior_trial_node(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one exact G004 candidate into the frozen behavioral trial schema."""
    if type(candidate) is not dict or not _ALPHA_MAX_PRIOR_BEHAVIOR_KEYS.issubset(candidate):
        raise ValueError("alpha_max_prior_trial_candidate_schema_mismatch")
    if (
        set(candidate)
        .difference(_ALPHA_MAX_PRIOR_CANDIDATE_KEYS)
        .difference(_ALPHA_MAX_PRIOR_COSMETIC_KEYS)
    ):
        raise ValueError("alpha_max_prior_trial_candidate_schema_mismatch")
    implementation = candidate["strategy_class"] or candidate["strategy"]
    timeframe = candidate["strategy_timeframe"] or candidate["timeframe"]
    if type(implementation) is not str or not implementation:
        raise ValueError("alpha_max_prior_trial_implementation_invalid")
    if type(timeframe) is not str or not timeframe:
        raise ValueError("alpha_max_prior_trial_timeframe_invalid")
    if type(candidate["symbols"]) is not list or any(
        type(symbol) is not str or not symbol for symbol in candidate["symbols"]
    ):
        raise ValueError("alpha_max_prior_trial_symbols_invalid")
    symbols = sorted(str(symbol).upper().replace("/", "") for symbol in candidate["symbols"])
    if any(not symbol for symbol in symbols) or len(symbols) != len(set(symbols)):
        raise ValueError("alpha_max_prior_trial_symbols_invalid")
    if type(candidate["params"]) is not dict or type(candidate["metadata"]) is not dict:
        raise ValueError("alpha_max_prior_trial_behavior_invalid")
    params = json.loads(_canonical_json_bytes(candidate["params"], newline=False))
    metadata = json.loads(_canonical_json_bytes(candidate["metadata"], newline=False))
    node = {
        "allocation": {},
        "behavior_metadata": metadata,
        "gross": None,
        "implementation": implementation,
        "kind": "prior_strategy_leaf",
        "members": [],
        "omission": None,
        "params": params,
        "schema": "alpha_max_trial_node.v1",
        "symbols": symbols,
        "timeframe": timeframe,
    }
    if set(node) != _ALPHA_MAX_PRIOR_NODE_KEYS:
        raise AssertionError("alpha-max prior trial node schema drift")
    return node


def alpha_max_trial_key(node: Mapping[str, Any]) -> str:
    """Hash one exact canonical behavioral trial node."""
    if type(node) is not dict:
        raise TypeError("alpha_max_trial_node_must_be_exact_dict")
    return _sha256_bytes(_canonical_json_bytes(node, newline=False))


def alpha_max_trial_key_set_lf_bytes(keys: Sequence[str]) -> bytes:
    """Serialize exact-deduplicated sorted keys with actual LF separator/trailer."""
    normalized: set[str] = set()
    for key in keys:
        normalized.add(_require_sha256(key, field="alpha_max_trial_key"))
    if not normalized:
        raise ValueError("alpha_max_trial_key_set_empty")
    return b"\n".join(key.encode("ascii") for key in sorted(normalized)) + b"\n"


@dataclass(frozen=True, slots=True)
class AlphaMaxTrialLedger:
    """Closed immutable prior/current trial-key ledger bound to 1487 hypotheses."""

    prior_trial_keys: tuple[str, ...]
    current_trial_keys: tuple[str, ...]
    union_trial_keys: tuple[str, ...]
    prior_key_set_lf_bytes: bytes
    current_key_set_lf_bytes: bytes
    prior_key_set_sha256: str
    current_key_set_sha256: str
    current_registry_sha256: str
    num_trials: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_trial_ledger.v1",
            "current_node_count": len(self.current_trial_keys),
            "current_registry_sha256": self.current_registry_sha256,
            "current_trial_key_set_sha256": self.current_key_set_sha256,
            "num_trials": self.num_trials,
            "prior_git_blob_oid": _ALPHA_MAX_PRIOR_BLOB_OID,
            "prior_node_count": len(self.prior_trial_keys),
            "prior_trial_key_set_sha256": self.prior_key_set_sha256,
        }


def _validate_alpha_max_trial_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if type(config) is not dict:
        raise ValueError("alpha_max_current_trial_inventory_mismatch")
    registry = config.get("current_trial_registry")
    contract = config.get("trial_ledger")
    normative = config.get("normative_sources")
    if (
        type(registry) is not dict
        or set(registry) != _ALPHA_MAX_CURRENT_REGISTRY_KEYS
        or type(contract) is not dict
        or type(normative) is not dict
    ):
        raise ValueError("alpha_max_current_trial_inventory_mismatch")
    expected_prior = {
        "artifact_kind": "g004_frozen_candidate_manifest",
        "baseline_git_blob_oid": _ALPHA_MAX_PRIOR_BLOB_OID,
        "baseline_path": _ALPHA_MAX_PRIOR_PATH,
        "candidate_count": 1466,
        "candidate_manifest_sha256": (
            "1292498b3b729038c74932175a12d910fc4351b2feb3bbfc95f827517e423efe"
        ),
        "candidate_set_sha256": (
            "01ca7a5c04b490b5472a62b49d0fcc7d432f0e2045c0e6fae9b1bfcb079a0564"
        ),
        "file_sha256": _ALPHA_MAX_PRIOR_FILE_SHA256,
        "prior_key_set_actual_lf_sha256": _ALPHA_MAX_PRIOR_KEY_SET_SHA256,
        "prior_node_count": 1466,
        "source_commit": _ALPHA_MAX_PRIOR_COMMIT,
    }
    expected_contract = {
        "canonical_node_bytes": _ALPHA_MAX_CANONICALIZATION,
        "cost_cells_are_trials": False,
        "current_node_count": 21,
        "current_registry_sha256": _ALPHA_MAX_CURRENT_REGISTRY_SHA256,
        "current_set_sha256": _ALPHA_MAX_CURRENT_KEY_SET_SHA256,
        "dsr_num_trials": ALPHA_MAX_DSR_NUM_TRIALS,
        "prior_inventory": expected_prior,
        "set_hash_serialization": {
            "actual_lf_byte_hex": "0a",
            "formula": 'sha256(("\\n".join(sorted(keys))+"\\n").encode("utf-8"))',
            "literal_backslash_n_bytes_forbidden": True,
            "literal_backslash_n_hex": "5c6e",
        },
        "status_or_availability_may_change_trial_count": False,
        "union_formula": "1466+21=1487",
    }
    if (
        contract != expected_contract
        or normative.get("baseline_commit") != _ALPHA_MAX_PRIOR_COMMIT
        or normative.get("current_trial_registry_sha256") != _ALPHA_MAX_CURRENT_REGISTRY_SHA256
        or normative.get("current_trial_key_set_sha256") != _ALPHA_MAX_CURRENT_KEY_SET_SHA256
        or normative.get("prior_trial_key_set_actual_lf_sha256") != _ALPHA_MAX_PRIOR_KEY_SET_SHA256
    ):
        raise ValueError("alpha_max_current_trial_inventory_mismatch")
    return registry


def build_alpha_max_trial_ledger(
    prior_blob_bytes: bytes,
    config: Mapping[str, Any],
) -> AlphaMaxTrialLedger:
    """Build the closed 1466+21 trial ledger without reading ambient registries."""
    if type(prior_blob_bytes) is not bytes or _sha256_bytes(prior_blob_bytes) != (
        _ALPHA_MAX_PRIOR_FILE_SHA256
    ):
        raise ValueError("alpha_max_prior_trial_inventory_mismatch")
    prior_manifest = _alpha_max_strict_json_object(
        prior_blob_bytes,
        field="prior_trial_inventory",
    )
    if (
        prior_manifest.get("artifact_kind") != "g004_frozen_candidate_manifest"
        or prior_manifest.get("candidate_count") != 1466
        or prior_manifest.get("candidate_manifest_sha256")
        != "1292498b3b729038c74932175a12d910fc4351b2feb3bbfc95f827517e423efe"
        or prior_manifest.get("candidate_set_sha256")
        != "01ca7a5c04b490b5472a62b49d0fcc7d432f0e2045c0e6fae9b1bfcb079a0564"
        or type(prior_manifest.get("candidates")) is not list
        or len(prior_manifest["candidates"]) != 1466
    ):
        raise ValueError("alpha_max_prior_trial_inventory_mismatch")
    try:
        prior_keys_unsorted = [
            alpha_max_trial_key(normalize_alpha_max_prior_trial_node(candidate))
            for candidate in prior_manifest["candidates"]
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha_max_prior_trial_inventory_mismatch") from exc
    prior_keys = tuple(sorted(set(prior_keys_unsorted)))
    prior_lf = alpha_max_trial_key_set_lf_bytes(prior_keys)
    if (
        len(prior_keys_unsorted) != 1466
        or len(prior_keys) != 1466
        or b"\\n" in prior_lf
        or not prior_lf.endswith(b"\x0a")
        or _sha256_bytes(prior_lf) != _ALPHA_MAX_PRIOR_KEY_SET_SHA256
    ):
        raise ValueError("alpha_max_prior_trial_inventory_mismatch")

    registry = _validate_alpha_max_trial_config(config)
    if (
        registry["schema"] != "alpha_max_current_trial_registry.v1"
        or registry["candidate_symbols"] != list(ALPHA_MAX_CANDIDATE_SYMBOLS)
        or registry["canonicalization"] != _ALPHA_MAX_CANONICALIZATION
        or registry["current_node_count"] != 21
        or registry["current_key_set_sha256"] != _ALPHA_MAX_CURRENT_KEY_SET_SHA256
        or type(registry["nodes"]) is not list
        or len(registry["nodes"]) != 21
    ):
        raise ValueError("alpha_max_current_trial_inventory_mismatch")
    current_registry_bytes = (
        json.dumps(
            registry,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if _sha256_bytes(current_registry_bytes) != _ALPHA_MAX_CURRENT_REGISTRY_SHA256:
        raise ValueError("alpha_max_current_trial_inventory_mismatch")
    current_nodes = registry["nodes"]
    if any(type(node) is not dict or set(node) != _ROW_KEYS for node in current_nodes):
        raise ValueError("alpha_max_current_trial_inventory_mismatch")
    if [node["row_id"] for node in current_nodes] != sorted(
        node["row_id"] for node in current_nodes
    ) or any(
        node.get("schema") != "alpha_max_trial_node.v1" or node.get("kind") != "current_matrix_row"
        for node in current_nodes
    ):
        raise ValueError("alpha_max_current_trial_inventory_mismatch")
    try:
        current_keys_unsorted = [alpha_max_trial_key(node) for node in current_nodes]
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha_max_current_trial_inventory_mismatch") from exc
    current_keys = tuple(sorted(set(current_keys_unsorted)))
    current_lf = alpha_max_trial_key_set_lf_bytes(current_keys)
    if (
        len(current_keys) != 21
        or b"\\n" in current_lf
        or not current_lf.endswith(b"\x0a")
        or _sha256_bytes(current_lf) != _ALPHA_MAX_CURRENT_KEY_SET_SHA256
    ):
        raise ValueError("alpha_max_current_trial_inventory_mismatch")
    if set(prior_keys).intersection(current_keys):
        raise ValueError("alpha_max_trial_key_collision")
    union_keys = tuple(sorted((*prior_keys, *current_keys)))
    if len(union_keys) != ALPHA_MAX_DSR_NUM_TRIALS:
        raise ValueError("alpha_max_trial_count_mismatch")
    return AlphaMaxTrialLedger(
        prior_trial_keys=prior_keys,
        current_trial_keys=current_keys,
        union_trial_keys=union_keys,
        prior_key_set_lf_bytes=prior_lf,
        current_key_set_lf_bytes=current_lf,
        prior_key_set_sha256=_ALPHA_MAX_PRIOR_KEY_SET_SHA256,
        current_key_set_sha256=_ALPHA_MAX_CURRENT_KEY_SET_SHA256,
        current_registry_sha256=_ALPHA_MAX_CURRENT_REGISTRY_SHA256,
        num_trials=ALPHA_MAX_DSR_NUM_TRIALS,
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxPreGateSharpeEvidence:
    """Matched-calendar finite nonannualized Sharpes and their sample variance."""

    candidate_ids: tuple[str, ...]
    calendar_sha256: str
    finite_sharpes: Mapping[str, float]
    degenerate_candidate_ids: tuple[str, ...]
    variance_across_trials: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_pre_gate_sharpe_evidence.v1",
            "calendar_sha256": self.calendar_sha256,
            "candidate_ids": list(self.candidate_ids),
            "degenerate_candidate_ids": list(self.degenerate_candidate_ids),
            "finite_nonannualized_sharpes": dict(self.finite_sharpes),
            "variance_across_trials": self.variance_across_trials,
        }


def _alpha_max_matched_primary_streams(
    streams: Mapping[str, AlphaMaxPrimaryReturnStream],
) -> tuple[tuple[str, ...], tuple[AlphaMaxPrimaryReturnStream, ...]]:
    if not isinstance(streams, Mapping) or not streams:
        raise ValueError("alpha_max_statistical_streams_empty")
    if any(type(candidate_id) is not str or not candidate_id for candidate_id in streams):
        raise ValueError("alpha_max_statistical_candidate_id_invalid")
    candidate_ids = tuple(sorted(streams))
    ordered = tuple(_validate_alpha_max_primary_stream(streams[key]) for key in candidate_ids)
    reference_calendar = ordered[0].endpoint_timestamps
    if any(stream.endpoint_timestamps != reference_calendar for stream in ordered[1:]):
        raise ValueError("alpha_max_statistical_calendar_mismatch")
    return candidate_ids, ordered


def alpha_max_pre_gate_sharpe_variance(
    streams: Mapping[str, AlphaMaxPrimaryReturnStream],
) -> AlphaMaxPreGateSharpeEvidence:
    """Compute exact finite pre-gate nonannualized Sharpe sample variance."""
    candidate_ids, ordered = _alpha_max_matched_primary_streams(streams)
    finite: dict[str, float] = {}
    degenerate: list[str] = []
    for candidate_id, stream in zip(candidate_ids, ordered, strict=True):
        values = np.asarray(stream.returns, dtype=np.float64)
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        if not math.isfinite(sigma) or sigma <= 1e-12:
            degenerate.append(candidate_id)
            continue
        sharpe = float(np.mean(values)) / sigma
        if math.isfinite(sharpe):
            finite[candidate_id] = sharpe
        else:
            degenerate.append(candidate_id)
    sharpe_values = tuple(
        finite[candidate_id] for candidate_id in candidate_ids if candidate_id in finite
    )
    variance = float(np.var(sharpe_values, ddof=1)) if len(sharpe_values) >= 2 else 0.0
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("alpha_max_pre_gate_sharpe_variance_invalid")
    return AlphaMaxPreGateSharpeEvidence(
        candidate_ids=candidate_ids,
        calendar_sha256=ordered[0].calendar_sha256,
        finite_sharpes=MappingProxyType(finite),
        degenerate_candidate_ids=tuple(degenerate),
        variance_across_trials=variance,
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxStatisticalEvidence:
    """Canonical DSR/SPA candidate evidence and one separate family-wise PBO."""

    candidate_ids: tuple[str, ...]
    input_role: str
    nominal_cost_bps: int
    calendar_sha256: str
    variance_across_trials: float
    finite_nonannualized_sharpes: Mapping[str, float]
    degenerate_candidate_ids: tuple[str, ...]
    dsr_by_candidate: Mapping[str, float]
    spa_pvalue_by_candidate: Mapping[str, float]
    pbo: float
    dsr_num_trials: int
    dsr_hac_inference: bool
    spa_bootstrap_rounds: int
    spa_block_size: int
    spa_seed: int
    pbo_n_splits: int
    prior_trial_key_set_sha256: str
    current_trial_key_set_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_statistical_evidence.v1",
            "calendar_sha256": self.calendar_sha256,
            "candidate_ids": list(self.candidate_ids),
            "current_trial_key_set_sha256": self.current_trial_key_set_sha256,
            "degenerate_candidate_ids": list(self.degenerate_candidate_ids),
            "dsr_by_candidate": dict(self.dsr_by_candidate),
            "dsr_hac_inference": self.dsr_hac_inference,
            "dsr_num_trials": self.dsr_num_trials,
            "finite_nonannualized_sharpes": dict(self.finite_nonannualized_sharpes),
            "input_role": self.input_role,
            "nominal_cost_bps": self.nominal_cost_bps,
            "pbo": self.pbo,
            "pbo_n_splits": self.pbo_n_splits,
            "prior_trial_key_set_sha256": self.prior_trial_key_set_sha256,
            "spa_block_size": self.spa_block_size,
            "spa_bootstrap_rounds": self.spa_bootstrap_rounds,
            "spa_pvalue_by_candidate": dict(self.spa_pvalue_by_candidate),
            "spa_seed": self.spa_seed,
            "variance_across_trials": self.variance_across_trials,
        }


def _alpha_max_probability(value: Any, *, field: str) -> float:
    parsed = _alpha_max_finite_number(value, field=field)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"alpha_max_{field}_invalid")
    return parsed


def _validate_alpha_max_trial_ledger_binding(ledger: AlphaMaxTrialLedger) -> None:
    if type(ledger) is not AlphaMaxTrialLedger:
        raise TypeError("alpha_max_trial_ledger_identity_invalid")
    prior = ledger.prior_trial_keys
    current = ledger.current_trial_keys
    union = ledger.union_trial_keys
    if (
        type(prior) is not tuple
        or type(current) is not tuple
        or type(union) is not tuple
        or len(prior) != 1466
        or len(current) != 21
        or len(union) != ALPHA_MAX_DSR_NUM_TRIALS
        or prior != tuple(sorted(set(prior)))
        or current != tuple(sorted(set(current)))
        or set(prior).intersection(current)
        or union != tuple(sorted((*prior, *current)))
    ):
        raise ValueError("alpha_max_trial_ledger_binding_invalid")
    try:
        prior_lf = alpha_max_trial_key_set_lf_bytes(prior)
        current_lf = alpha_max_trial_key_set_lf_bytes(current)
    except ValueError as exc:
        raise ValueError("alpha_max_trial_ledger_binding_invalid") from exc
    if (
        ledger.prior_key_set_lf_bytes != prior_lf
        or ledger.current_key_set_lf_bytes != current_lf
        or ledger.prior_key_set_sha256 != _ALPHA_MAX_PRIOR_KEY_SET_SHA256
        or ledger.current_key_set_sha256 != _ALPHA_MAX_CURRENT_KEY_SET_SHA256
        or ledger.current_registry_sha256 != _ALPHA_MAX_CURRENT_REGISTRY_SHA256
        or _sha256_bytes(prior_lf) != ledger.prior_key_set_sha256
        or _sha256_bytes(current_lf) != ledger.current_key_set_sha256
        or ledger.num_trials != ALPHA_MAX_DSR_NUM_TRIALS
    ):
        raise ValueError("alpha_max_trial_ledger_binding_invalid")


def build_alpha_max_statistical_evidence(
    streams: Mapping[str, AlphaMaxPrimaryReturnStream],
    trial_ledger: AlphaMaxTrialLedger,
) -> AlphaMaxStatisticalEvidence:
    """Call only the canonical DSR, SPA, and CSCV-PBO primitives with frozen inputs."""
    _validate_alpha_max_trial_ledger_binding(trial_ledger)
    candidate_ids, ordered = _alpha_max_matched_primary_streams(streams)
    sharpe_evidence = alpha_max_pre_gate_sharpe_variance(streams)
    if sharpe_evidence.degenerate_candidate_ids:
        raise ValueError("alpha_max_statistical_stream_degenerate")
    block_size = max(1, round(len(ordered[0].returns) ** (1.0 / 3.0)))
    dsr: dict[str, float] = {}
    spa: dict[str, float] = {}
    matrix_rows: list[np.ndarray] = []
    for candidate_id, stream in zip(candidate_ids, ordered, strict=True):
        values = np.asarray(stream.returns, dtype=np.float64)
        matrix_rows.append(values)
        dsr[candidate_id] = _alpha_max_probability(
            research_metrics.deflated_sharpe_ratio(
                values,
                num_trials=ALPHA_MAX_DSR_NUM_TRIALS,
                variance_across_trials=sharpe_evidence.variance_across_trials,
                hac_inference=True,
            ),
            field="dsr_output",
        )
        spa[candidate_id] = _alpha_max_probability(
            research_metrics.spa_like_pvalue(
                values,
                bootstrap_rounds=2000,
                block_size=block_size,
                seed=12345,
            ),
            field="spa_output",
        )
    matrix = np.vstack(matrix_rows)
    pbo = _alpha_max_probability(
        research_metrics.cscv_pbo(matrix, n_splits=8),
        field="pbo_output",
    )
    return AlphaMaxStatisticalEvidence(
        candidate_ids=candidate_ids,
        input_role="pre_gate_matched_selection_eligible",
        nominal_cost_bps=30,
        calendar_sha256=sharpe_evidence.calendar_sha256,
        variance_across_trials=sharpe_evidence.variance_across_trials,
        finite_nonannualized_sharpes=sharpe_evidence.finite_sharpes,
        degenerate_candidate_ids=sharpe_evidence.degenerate_candidate_ids,
        dsr_by_candidate=MappingProxyType(dsr),
        spa_pvalue_by_candidate=MappingProxyType(spa),
        pbo=pbo,
        dsr_num_trials=ALPHA_MAX_DSR_NUM_TRIALS,
        dsr_hac_inference=True,
        spa_bootstrap_rounds=2000,
        spa_block_size=block_size,
        spa_seed=12345,
        pbo_n_splits=8,
        prior_trial_key_set_sha256=trial_ledger.prior_key_set_sha256,
        current_trial_key_set_sha256=trial_ledger.current_key_set_sha256,
    )


# ---------------------------------------------------------------------------
# Revision 5.14 pure data, evidence, selection, and sealing contracts.
# These helpers intentionally remain below the canonical statistical primitives
# so orchestration code has one deterministic evidence module and no parallel
# implementation of admission, selection, reconciliation, or serialization.


def _alpha_max_safe_relative_path(value: str, *, field: str) -> str:
    if type(value) is not str or not value or "\0" in value or "\\" in value:
        raise ValueError(f"alpha_max_{field}_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"alpha_max_{field}_invalid")
    return value


def _alpha_max_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _alpha_max_partition_contract(
    relative_path: str,
    *,
    root_kind: str,
    exchange: str,
) -> tuple[str, datetime, datetime]:
    parts = PurePosixPath(relative_path).parts
    if root_kind == "raw":
        if len(parts) == 4 and parts[:2] == ("market_ohlcv_1s", exchange):
            _, _, symbol, filename = parts
        elif len(parts) == 3 and parts[0] == exchange:
            _, symbol, filename = parts
        elif len(parts) == 2:
            symbol, filename = parts
        else:
            raise ValueError("alpha_max_raw_root_partition_layout_invalid")
        if symbol not in ALPHA_MAX_CANDIDATE_SYMBOLS or not filename.endswith(".parquet"):
            raise ValueError("alpha_max_raw_root_partition_layout_invalid")
        month_token = filename.removesuffix(".parquet")
        try:
            partition_start = datetime.strptime(month_token, "%Y-%m").replace(tzinfo=UTC)
        except ValueError as exc:
            raise ValueError("alpha_max_raw_root_partition_layout_invalid") from exc
        if partition_start.month == 12:
            partition_end = partition_start.replace(
                year=partition_start.year + 1,
                month=1,
            )
        else:
            partition_end = partition_start.replace(month=partition_start.month + 1)
        return symbol, partition_start, partition_end

    offset = 0
    if len(parts) >= 2 and parts[:2] == ("feature_points", f"exchange={exchange}"):
        offset = 2
    elif parts and parts[0] == f"exchange={exchange}":
        offset = 1
    scoped = parts[offset:]
    if (
        len(scoped) != 3
        or not scoped[0].startswith("symbol=")
        or not scoped[1].startswith("date=")
        or not scoped[2].endswith(".parquet")
    ):
        raise ValueError("alpha_max_feature_root_partition_layout_invalid")
    symbol = scoped[0].removeprefix("symbol=")
    if symbol not in ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_feature_root_partition_layout_invalid")
    try:
        partition_start = datetime.strptime(
            scoped[1].removeprefix("date="),
            "%Y-%m-%d",
        ).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("alpha_max_feature_root_partition_layout_invalid") from exc
    return symbol, partition_start, partition_start + timedelta(days=1)


def _alpha_max_validate_root_inventory_coverage(
    entries: tuple[AlphaMaxTreeEntry, ...],
    *,
    root_kind: str,
    exchange: str,
    start: datetime,
    end: datetime,
    expected_symbols: tuple[str, ...],
) -> None:
    observed: dict[str, list[tuple[datetime, AlphaMaxTreeEntry]]] = {
        symbol: [] for symbol in expected_symbols
    }
    for entry in entries:
        symbol, partition_start, _ = _alpha_max_partition_contract(
            entry.relative_path,
            root_kind=root_kind,
            exchange=exchange,
        )
        if symbol not in observed:
            raise ValueError("alpha_max_root_symbol_scope_mismatch")
        if (
            root_kind == "feature"
            and entry.maximum_gap_ms
            > _FUNDING_INTERVAL_MS + _FEATURE_CAUSAL_EDGE_MS
        ):
            raise ValueError("alpha_max_feature_root_timestamp_cadence_invalid")
        observed[symbol].append((partition_start, entry))
    if any(not values for values in observed.values()):
        raise ValueError("alpha_max_root_symbol_coverage_incomplete")

    if root_kind == "feature":
        expected_partitions = tuple(
            start + timedelta(days=index) for index in range((end - start).days)
        )
        for values in observed.values():
            ordered = tuple(sorted(values, key=lambda value: value[0]))
            actual = tuple(partition for partition, _ in ordered)
            if actual != expected_partitions:
                raise ValueError("alpha_max_feature_root_interval_coverage_incomplete")
            if (
                ordered[0][1].minimum_timestamp_ms
                > _epoch_ms(start) + _FUNDING_INTERVAL_MS
                or ordered[-1][1].maximum_timestamp_ms
                < _epoch_ms(end) - _FUNDING_INTERVAL_MS
                or any(
                    right.minimum_timestamp_ms - left.maximum_timestamp_ms
                    > _FUNDING_INTERVAL_MS + _FEATURE_CAUSAL_EDGE_MS
                    for (_, left), (_, right) in pairwise(ordered)
                )
            ):
                raise ValueError("alpha_max_feature_root_funding_coverage_incomplete")
        return

    expected_months: list[datetime] = []
    cursor = start.replace(day=1)
    while cursor < end:
        expected_months.append(cursor)
        cursor = (
            cursor.replace(year=cursor.year + 1, month=1)
            if cursor.month == 12
            else cursor.replace(month=cursor.month + 1)
        )
    for values in observed.values():
        ordered = tuple(sorted(values, key=lambda value: value[0]))
        if tuple(partition for partition, _ in ordered) != tuple(expected_months):
            raise ValueError("alpha_max_raw_root_interval_coverage_incomplete")
def _alpha_max_parquet_timestamp_bounds(
    descriptor: int,
    *,
    root_kind: str,
    expected_symbol: str,
    expected_exchange: str,
    owned_start_ms: int,
    owned_end_ms: int,
) -> tuple[int, int, int, int]:
    column = "datetime" if root_kind == "raw" else "timestamp_ms"
    try:
        with os.fdopen(os.dup(descriptor), "rb") as parquet_file:
            schema = pl.read_parquet_schema(parquet_file)
            parquet_file.seek(0)
            columns = [column]
            if root_kind == "feature":
                if "funding_rate" not in schema:
                    raise ValueError("alpha_max_feature_root_funding_column_missing")
                columns.append("funding_rate")
            columns.extend(
                value for value in ("symbol", "exchange") if value in schema
            )
            frame = pl.read_parquet(parquet_file, columns=columns)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("alpha_max_root_parquet_timestamp_read_failed") from exc
    if frame.height <= 0 or column not in frame.columns:
        raise ValueError("alpha_max_root_parquet_timestamp_empty")
    if "symbol" in frame.columns and (
        frame.get_column("symbol").null_count()
        or set(frame.get_column("symbol").cast(pl.String).to_list())
        != {expected_symbol}
    ):
        raise ValueError("alpha_max_root_content_symbol_mismatch")
    if "exchange" in frame.columns and (
        frame.get_column("exchange").null_count()
        or {value.lower() for value in frame.get_column("exchange").cast(pl.String).to_list()}
        != {expected_exchange}
    ):
        raise ValueError("alpha_max_root_content_exchange_mismatch")
    series = frame.get_column(column)
    minimum = series.min()
    maximum = series.max()
    if root_kind == "raw":
        if type(minimum) is not datetime or type(maximum) is not datetime:
            raise ValueError("alpha_max_raw_root_timestamp_schema_invalid")
        if minimum.tzinfo is None:
            minimum = minimum.replace(tzinfo=UTC)
        if maximum.tzinfo is None:
            maximum = maximum.replace(tzinfo=UTC)
        minimum_ms = _epoch_ms(_utc(minimum, field="raw_root_minimum_timestamp"))
        maximum_ms = _epoch_ms(_utc(maximum, field="raw_root_maximum_timestamp"))
        timestamp_series = series.dt.epoch("ms")
    else:
        if type(minimum) is not int or type(maximum) is not int:
            raise ValueError("alpha_max_feature_root_timestamp_schema_invalid")
        minimum_ms = minimum
        maximum_ms = maximum
        timestamp_series = series
    if minimum_ms < 0 or maximum_ms < minimum_ms:
        raise ValueError("alpha_max_root_timestamp_bounds_invalid")
    if timestamp_series.null_count() or timestamp_series.n_unique() != timestamp_series.len():
        raise ValueError("alpha_max_root_timestamp_duplicate_or_null")
    diffs = timestamp_series.diff().drop_nulls()
    maximum_gap_ms = 0 if diffs.is_empty() else int(diffs.max())
    minimum_gap_ms = 0 if diffs.is_empty() else int(diffs.min())
    if timestamp_series.len() > 1 and minimum_gap_ms <= 0:
        raise ValueError("alpha_max_root_timestamp_not_strictly_increasing")
    if root_kind == "raw" and any(int(value) % 1000 != 0 for value in timestamp_series):
        raise ValueError("alpha_max_raw_root_timestamp_alignment_invalid")
    if (
        root_kind == "feature"
        and maximum_gap_ms > _FUNDING_INTERVAL_MS + _FEATURE_CAUSAL_EDGE_MS
    ):
        raise ValueError("alpha_max_feature_root_timestamp_cadence_invalid")
    if root_kind == "feature":
        funding = frame.get_column("funding_rate")
        rates = funding.to_list()
        if any(
            type(rate) not in {int, float} or not math.isfinite(float(rate))
            for rate in rates
        ):
            raise ValueError("alpha_max_feature_root_funding_value_invalid")
    return minimum_ms, maximum_ms, int(timestamp_series.len()), maximum_gap_ms


def _alpha_max_stream_regular_file(
    path: Path,
    *,
    root_kind: str,
    expected_symbol: str,
    expected_exchange: str,
    owned_start_ms: int,
    owned_end_ms: int,
) -> tuple[str, os.stat_result, int, int, int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("alpha_max_root_file_open_failed") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("alpha_max_root_entry_not_regular")
        if int(before.st_nlink) != 1:
            raise ValueError("alpha_max_root_hardlink_rejected")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        (
            minimum_timestamp_ms,
            maximum_timestamp_ms,
            row_count,
            maximum_gap_ms,
        ) = _alpha_max_parquet_timestamp_bounds(
            descriptor,
            root_kind=root_kind,
            expected_symbol=expected_symbol,
            expected_exchange=expected_exchange,
            owned_start_ms=owned_start_ms,
            owned_end_ms=owned_end_ms,
        )
        after = os.fstat(descriptor)
        if _alpha_max_file_identity(before) != _alpha_max_file_identity(after):
            raise ValueError("alpha_max_root_file_changed_during_seal")
        return (
            digest.hexdigest(),
            before,
            minimum_timestamp_ms,
            maximum_timestamp_ms,
            row_count,
            maximum_gap_ms,
        )
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class AlphaMaxTreeEntry:
    """One safe regular file in a canonical raw/feature root inventory."""

    relative_path: str
    byte_count: int
    mode: int
    mtime_ns: int
    minimum_timestamp_ms: int
    maximum_timestamp_ms: int
    row_count: int
    maximum_gap_ms: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _alpha_max_safe_relative_path(self.relative_path, field="tree_relative_path"),
        )
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("alpha_max_tree_byte_count_invalid")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o7777:
            raise ValueError("alpha_max_tree_mode_invalid")
        if type(self.mtime_ns) is not int or self.mtime_ns < 0:
            raise ValueError("alpha_max_tree_mtime_invalid")
        if (
            type(self.minimum_timestamp_ms) is not int
            or type(self.maximum_timestamp_ms) is not int
            or self.minimum_timestamp_ms < 0
            or self.maximum_timestamp_ms < self.minimum_timestamp_ms
        ):
            raise ValueError("alpha_max_tree_timestamp_bounds_invalid")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("alpha_max_tree_row_count_invalid")
        if type(self.maximum_gap_ms) is not int or self.maximum_gap_ms < 0:
            raise ValueError("alpha_max_tree_maximum_gap_invalid")
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, field="alpha_max_tree_entry_sha256"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
            "maximum_timestamp_ms": self.maximum_timestamp_ms,
            "maximum_gap_ms": self.maximum_gap_ms,
            "minimum_timestamp_ms": self.minimum_timestamp_ms,
            "relative_path": self.relative_path,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AlphaMaxRootReceipt:
    """Immutable root identity carried by every row/cost-cell bundle."""

    root_id: str
    root_kind: str
    path: str
    exchange: str
    symbols: tuple[str, ...]
    start_utc: datetime
    end_utc: datetime
    inventory_sha256: str
    content_sha256: str
    file_count: int

    def __post_init__(self) -> None:
        if self.root_id not in _ROOT_INTERVALS:
            raise ValueError("alpha_max_root_receipt_id_invalid")
        if self.root_kind not in {"raw", "feature"}:
            raise ValueError("alpha_max_root_receipt_kind_invalid")
        if self.exchange != "binance":
            raise ValueError("alpha_max_root_receipt_exchange_invalid")
        if self.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS:
            raise ValueError("alpha_max_root_receipt_symbols_invalid")
        expected_start, expected_end = _ROOT_INTERVALS[self.root_id]
        start = _utc(self.start_utc, field="root_receipt_start")
        end = _utc(self.end_utc, field="root_receipt_end")
        if start != expected_start or end != expected_end:
            raise ValueError("alpha_max_root_receipt_bounds_invalid")
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)
        object.__setattr__(
            self,
            "path",
            _require_explicit_canonical_path(self.path, field="alpha_max_root_receipt_path"),
        )
        object.__setattr__(
            self,
            "inventory_sha256",
            _require_sha256(
                self.inventory_sha256,
                field="alpha_max_root_receipt_inventory_sha256",
            ),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _require_sha256(
                self.content_sha256,
                field="alpha_max_root_receipt_content_sha256",
            ),
        )
        if type(self.file_count) is not int or self.file_count <= 0:
            raise ValueError("alpha_max_root_receipt_file_count_invalid")

    def to_payload(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "end_utc": self.end_utc.isoformat().replace("+00:00", "Z"),
            "exchange": self.exchange,
            "file_count": self.file_count,
            "inventory_sha256": self.inventory_sha256,
            "path": self.path,
            "root_id": self.root_id,
            "root_kind": self.root_kind,
            "start_utc": self.start_utc.isoformat().replace("+00:00", "Z"),
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True, slots=True)
class AlphaMaxRootSeal:
    """Canonical metadata inventory and path-independent file-content seal."""

    root_id: str
    root_kind: str
    path: str
    exchange: str
    symbols: tuple[str, ...]
    start_utc: datetime
    end_utc: datetime
    entries: tuple[AlphaMaxTreeEntry, ...]
    inventory_sha256: str
    content_sha256: str
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        if self.root_id not in _ROOT_INTERVALS or self.root_kind not in {"raw", "feature"}:
            raise ValueError("alpha_max_root_seal_identity_invalid")
        if self.exchange != "binance":
            raise ValueError("alpha_max_root_seal_exchange_invalid")
        if self.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS:
            raise ValueError("alpha_max_root_seal_symbols_invalid")
        expected_start, expected_end = _ROOT_INTERVALS[self.root_id]
        if (
            _utc(self.start_utc, field="root_seal_start") != expected_start
            or _utc(self.end_utc, field="root_seal_end") != expected_end
        ):
            raise ValueError("alpha_max_root_seal_bounds_invalid")
        canonical_path = _require_explicit_canonical_path(
            self.path,
            field="alpha_max_root_seal_path",
        )
        object.__setattr__(self, "path", canonical_path)
        if (
            type(self.entries) is not tuple
            or not self.entries
            or any(type(value) is not AlphaMaxTreeEntry for value in self.entries)
            or tuple(sorted(self.entries, key=lambda value: value.relative_path)) != self.entries
            or len({value.relative_path for value in self.entries}) != len(self.entries)
        ):
            raise ValueError("alpha_max_root_seal_entries_invalid")
        root_start_ms = _epoch_ms(expected_start)
        root_end_ms = _epoch_ms(expected_end)
        for entry in self.entries:
            _, partition_start, partition_end = _alpha_max_partition_contract(
                entry.relative_path,
                root_kind=self.root_kind,
                exchange=self.exchange,
            )
            if not (
                _epoch_ms(partition_start)
                <= entry.minimum_timestamp_ms
                <= entry.maximum_timestamp_ms
                < _epoch_ms(partition_end)
                and root_start_ms
                <= entry.minimum_timestamp_ms
                <= entry.maximum_timestamp_ms
                < root_end_ms
            ):
                raise ValueError("alpha_max_root_seal_partition_content_mismatch")
        _alpha_max_validate_root_inventory_coverage(
            self.entries,
            root_kind=self.root_kind,
            exchange=self.exchange,
            start=expected_start,
            end=expected_end,
            expected_symbols=self.symbols,
        )
        inventory_payload = [
            {
                "byte_count": entry.byte_count,
                "maximum_timestamp_ms": entry.maximum_timestamp_ms,
                "maximum_gap_ms": entry.maximum_gap_ms,
                "minimum_timestamp_ms": entry.minimum_timestamp_ms,
                "mode": entry.mode,
                "mtime_ns": entry.mtime_ns,
                "relative_path": entry.relative_path,
                "row_count": entry.row_count,
            }
            for entry in self.entries
        ]
        content_payload = [
            {
                "byte_count": entry.byte_count,
                "maximum_timestamp_ms": entry.maximum_timestamp_ms,
                "maximum_gap_ms": entry.maximum_gap_ms,
                "minimum_timestamp_ms": entry.minimum_timestamp_ms,
                "relative_path": entry.relative_path,
                "row_count": entry.row_count,
                "sha256": entry.sha256,
            }
            for entry in self.entries
        ]
        expected_inventory = _sha256_bytes(
            _canonical_json_bytes(inventory_payload, newline=True)
        )
        expected_content = _sha256_bytes(_canonical_json_bytes(content_payload, newline=True))
        if self.inventory_sha256 != expected_inventory or self.content_sha256 != expected_content:
            raise ValueError("alpha_max_root_seal_digest_mismatch")
        payload = {
            "artifact_kind": "alpha_max_root_seal.v1",
            "content_sha256": expected_content,
            "end_utc": expected_end.isoformat().replace("+00:00", "Z"),
            "entries": [entry.to_payload() for entry in self.entries],
            "exchange": self.exchange,
            "file_count": len(self.entries),
            "inventory_sha256": expected_inventory,
            "path": canonical_path,
            "root_id": self.root_id,
            "root_kind": self.root_kind,
            "start_utc": expected_start.isoformat().replace("+00:00", "Z"),
            "symbols": list(self.symbols),
        }
        expected_canonical = _canonical_json_bytes(payload, newline=True)
        if (
            type(self.canonical_bytes) is not bytes
            or self.canonical_bytes != expected_canonical
            or self.sha256 != _sha256_bytes(expected_canonical)
        ):
            raise ValueError("alpha_max_root_seal_canonical_mismatch")

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)

    def to_receipt(self) -> AlphaMaxRootReceipt:
        return AlphaMaxRootReceipt(
            root_id=self.root_id,
            root_kind=self.root_kind,
            path=self.path,
            exchange=self.exchange,
            symbols=self.symbols,
            start_utc=self.start_utc,
            end_utc=self.end_utc,
            inventory_sha256=self.inventory_sha256,
            content_sha256=self.content_sha256,
            file_count=len(self.entries),
        )


def seal_alpha_max_root_tree(
    root_id: str,
    root_kind: str,
    root_path: str | os.PathLike[str],
    *,
    exchange: str = "binance",
    expected_symbols: tuple[str, ...] = ALPHA_MAX_CANDIDATE_SYMBOLS,
) -> AlphaMaxRootSeal:
    """Stream-hash one explicit root without following links or retaining file bytes."""
    if root_id not in _ROOT_INTERVALS:
        raise ValueError("alpha_max_root_id_invalid")
    if root_kind not in {"raw", "feature"}:
        raise ValueError("alpha_max_root_kind_invalid")
    if exchange != "binance":
        raise ValueError("alpha_max_root_exchange_invalid")
    if expected_symbols != ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_root_expected_symbols_invalid")
    canonical = _require_explicit_canonical_path(root_path, field="alpha_max_root_path")
    root = Path(canonical)
    if not root.is_dir():
        raise ValueError("alpha_max_root_not_directory")
    start, end = _ROOT_INTERVALS[root_id]
    root_start_ms = _epoch_ms(start)
    root_end_ms = _epoch_ms(end)

    entries: list[AlphaMaxTreeEntry] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in sorted(directory_names):
            child = current / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError("alpha_max_root_symlink_rejected")
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError("alpha_max_root_entry_type_rejected")
            resolved = child.resolve(strict=True)
            if root not in resolved.parents:
                raise ValueError("alpha_max_root_path_escape")
        for name in sorted(file_names):
            child = current / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError("alpha_max_root_symlink_rejected")
            if not stat.S_ISREG(child_stat.st_mode):
                raise ValueError("alpha_max_root_entry_type_rejected")
            relative = child.relative_to(root).as_posix()
            symbol, partition_start, partition_end = _alpha_max_partition_contract(
                relative,
                root_kind=root_kind,
                exchange=exchange,
            )
            owned_partition_start = max(start, partition_start)
            owned_partition_end = min(end, partition_end)
            (
                file_sha256,
                sealed_stat,
                minimum_timestamp_ms,
                maximum_timestamp_ms,
                row_count,
                maximum_gap_ms,
            ) = _alpha_max_stream_regular_file(
                child,
                root_kind=root_kind,
                expected_symbol=symbol,
                expected_exchange=exchange,
                owned_start_ms=_epoch_ms(owned_partition_start),
                owned_end_ms=_epoch_ms(owned_partition_end),
            )
            partition_start_ms = _epoch_ms(partition_start)
            partition_end_ms = _epoch_ms(partition_end)
            if not (
                partition_start_ms <= minimum_timestamp_ms <= maximum_timestamp_ms
                < partition_end_ms
            ):
                raise ValueError("alpha_max_root_partition_content_mismatch")
            if not (
                root_start_ms <= minimum_timestamp_ms <= maximum_timestamp_ms < root_end_ms
            ):
                raise ValueError("alpha_max_root_content_outside_interval")
            entries.append(
                AlphaMaxTreeEntry(
                    relative_path=relative,
                    byte_count=int(sealed_stat.st_size),
                    mode=stat.S_IMODE(sealed_stat.st_mode),
                    mtime_ns=int(sealed_stat.st_mtime_ns),
                    minimum_timestamp_ms=minimum_timestamp_ms,
                    maximum_timestamp_ms=maximum_timestamp_ms,
                    row_count=row_count,
                    maximum_gap_ms=maximum_gap_ms,
                    sha256=file_sha256,
                )
            )
    ordered = tuple(sorted(entries, key=lambda entry: entry.relative_path))
    if not ordered:
        raise ValueError("alpha_max_root_inventory_empty")
    _alpha_max_validate_root_inventory_coverage(
        ordered,
        root_kind=root_kind,
        exchange=exchange,
        start=start,
        end=end,
        expected_symbols=ALPHA_MAX_CANDIDATE_SYMBOLS,
    )
    inventory_payload = [
        {
            "byte_count": entry.byte_count,
            "mode": entry.mode,
            "mtime_ns": entry.mtime_ns,
            "maximum_timestamp_ms": entry.maximum_timestamp_ms,
            "maximum_gap_ms": entry.maximum_gap_ms,
            "minimum_timestamp_ms": entry.minimum_timestamp_ms,
            "relative_path": entry.relative_path,
            "row_count": entry.row_count,
        }
        for entry in ordered
    ]
    content_payload = [
        {
            "byte_count": entry.byte_count,
            "maximum_timestamp_ms": entry.maximum_timestamp_ms,
            "maximum_gap_ms": entry.maximum_gap_ms,
            "minimum_timestamp_ms": entry.minimum_timestamp_ms,
            "relative_path": entry.relative_path,
            "row_count": entry.row_count,
            "sha256": entry.sha256,
        }
        for entry in ordered
    ]
    inventory_sha256 = _sha256_bytes(_canonical_json_bytes(inventory_payload, newline=True))
    content_sha256 = _sha256_bytes(_canonical_json_bytes(content_payload, newline=True))
    payload = {
        "artifact_kind": "alpha_max_root_seal.v1",
        "content_sha256": content_sha256,
        "end_utc": end.isoformat().replace("+00:00", "Z"),
        "entries": [entry.to_payload() for entry in ordered],
        "exchange": exchange,
        "file_count": len(ordered),
        "inventory_sha256": inventory_sha256,
        "path": canonical,
        "root_id": root_id,
        "root_kind": root_kind,
        "start_utc": start.isoformat().replace("+00:00", "Z"),
        "symbols": list(ALPHA_MAX_CANDIDATE_SYMBOLS),
    }
    canonical_bytes = _canonical_json_bytes(payload, newline=True)
    return AlphaMaxRootSeal(
        root_id=root_id,
        root_kind=root_kind,
        path=canonical,
        exchange=exchange,
        symbols=ALPHA_MAX_CANDIDATE_SYMBOLS,
        start_utc=start,
        end_utc=end,
        entries=ordered,
        inventory_sha256=inventory_sha256,
        content_sha256=content_sha256,
        canonical_bytes=canonical_bytes,
        sha256=_sha256_bytes(canonical_bytes),
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxContractRecord:
    symbol: str
    market_type: str
    linear: bool
    inverse: bool
    quote_asset: str
    margin_asset: str
    settle_asset: str
    volume_unit: str
    contract_multiplier: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract_multiplier": self.contract_multiplier,
            "inverse": self.inverse,
            "linear": self.linear,
            "margin_asset": self.margin_asset,
            "market_type": self.market_type,
            "quote_asset": self.quote_asset,
            "settle_asset": self.settle_asset,
            "symbol": self.symbol,
            "volume_unit": self.volume_unit,
        }


@dataclass(frozen=True, slots=True)
class AlphaMaxContractManifestSeal:
    path: str
    records: tuple[AlphaMaxContractRecord, ...]
    byte_count: int
    canonical_bytes: bytes
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _alpha_max_expected_contract_record(symbol: str) -> AlphaMaxContractRecord:
    return AlphaMaxContractRecord(
        symbol=symbol,
        market_type="perpetual",
        linear=True,
        inverse=False,
        quote_asset="USDT",
        margin_asset="USDT",
        settle_asset="USDT",
        volume_unit="base_asset",
        contract_multiplier=1.0,
    )


def seal_alpha_max_contract_manifest(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> AlphaMaxContractManifestSeal:
    """Read once and validate the sole exact ten-contract metadata assertion."""
    receipt, raw = read_artifact_bytes(path, artifact_id="alpha_max_contract_manifest")
    payload = _alpha_max_strict_json_object(raw, field="contract_manifest")
    canonical = _canonical_json_bytes(payload, newline=True)
    if raw != canonical:
        raise ValueError("alpha_max_contract_manifest_not_canonical")
    expected_records = tuple(
        _alpha_max_expected_contract_record(symbol) for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS
    )
    expected = {
        "exchange": "binance",
        "records": [record.to_payload() for record in expected_records],
        "schema_version": "alpha_max_contract_manifest.v1",
    }
    if payload != expected:
        raise ValueError("alpha_max_contract_manifest_mismatch")
    if expected_sha256 is not None and receipt.sha256 != _require_sha256(
        expected_sha256,
        field="alpha_max_contract_manifest_expected_sha256",
    ):
        raise ValueError("alpha_max_contract_manifest_sha256_mismatch")
    return AlphaMaxContractManifestSeal(
        path=receipt.canonical_path,
        records=expected_records,
        byte_count=receipt.byte_count,
        canonical_bytes=canonical,
        sha256=receipt.sha256,
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxRawObservation:
    """One exact train-owned raw observation used by the admission computation."""

    timestamp: datetime
    close: float
    volume: float

    def __post_init__(self) -> None:
        timestamp = _utc(self.timestamp, field="admission_observation_timestamp")
        train_start, train_end = _ROOT_INTERVALS["train"]
        if not train_start <= timestamp < train_end:
            raise ValueError("alpha_max_admission_observation_outside_train")
        close = _alpha_max_finite_number(
            self.close,
            field="admission_observation_close",
            positive=True,
        )
        volume = _alpha_max_finite_number(
            self.volume,
            field="admission_observation_volume",
            nonnegative=True,
        )
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "volume", volume)


@dataclass(frozen=True, slots=True)
class AlphaMaxAdmissionCandidateInput:
    symbol: str
    train_observations: tuple[AlphaMaxRawObservation, ...]
    consecutive_completed_daily_bars_before_train: int
    causal_funding_coverage_complete: bool
    unresolved_daily_cross_section_count: int
    partition_integrity_complete: bool = True

    def __post_init__(self) -> None:
        if self.symbol not in ALPHA_MAX_CANDIDATE_SYMBOLS:
            raise ValueError("alpha_max_admission_input_symbol_invalid")
        if type(self.train_observations) is not tuple or any(
            type(value) is not AlphaMaxRawObservation for value in self.train_observations
        ):
            raise TypeError("alpha_max_admission_observations_must_be_exact_tuple")
        _admission_nonnegative_int(
            self.consecutive_completed_daily_bars_before_train,
            field="consecutive_completed_daily_bars_before_train",
        )
        _admission_nonnegative_int(
            self.unresolved_daily_cross_section_count,
            field="unresolved_daily_cross_section_count",
        )
        if (
            type(self.causal_funding_coverage_complete) is not bool
            or type(self.partition_integrity_complete) is not bool
        ):
            raise TypeError("alpha_max_admission_input_coverage_must_be_bool")


@dataclass(frozen=True, slots=True)
class AlphaMaxDailyQuoteNotional:
    day: date
    quote_notional_usdt: float
    completed_4h_bucket_hours: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.day) is not date:
            raise TypeError("alpha_max_daily_quote_notional_day_invalid")
        value = _alpha_max_finite_number(
            self.quote_notional_usdt,
            field="daily_quote_notional_usdt",
            nonnegative=True,
        )
        object.__setattr__(self, "quote_notional_usdt", value)
        if type(self.completed_4h_bucket_hours) is not tuple or any(
            type(hour) is not int for hour in self.completed_4h_bucket_hours
        ):
            raise TypeError("alpha_max_daily_quote_notional_buckets_invalid")
        if self.completed_4h_bucket_hours != tuple(sorted(set(self.completed_4h_bucket_hours))):
            raise ValueError("alpha_max_daily_quote_notional_buckets_invalid")
        if any(hour not in {0, 4, 8, 12, 16, 20} for hour in self.completed_4h_bucket_hours):
            raise ValueError("alpha_max_daily_quote_notional_buckets_invalid")

    def to_payload(self) -> dict[str, Any]:
        return {
            "completed_4h_bucket_hours": list(self.completed_4h_bucket_hours),
            "day": self.day.isoformat(),
            "quote_notional_usdt": self.quote_notional_usdt,
        }


def build_alpha_max_daily_quote_notional(
    timestamps: tuple[datetime, ...],
    closes: tuple[float, ...],
    volumes: tuple[float, ...],
) -> AlphaMaxDailyQuoteNotional:
    """Collapse one UTC train day with exactly one row-level ``math.fsum``.

    The caller may discard the raw vectors immediately after this function
    returns.  In particular, no 4h subtotal is produced or re-summed here.
    """
    if type(timestamps) is not tuple or type(closes) is not tuple or type(volumes) is not tuple:
        raise TypeError("alpha_max_daily_quote_notional_vectors_must_be_exact_tuples")
    if not timestamps or len(timestamps) != len(closes) or len(timestamps) != len(volumes):
        raise ValueError("alpha_max_daily_quote_notional_vector_lengths_invalid")

    normalized_timestamps: list[datetime] = []
    products: list[float] = []
    completed_buckets: set[int] = set()
    observed_day: date | None = None
    train_start, train_end = _ROOT_INTERVALS["train"]
    for timestamp_value, close_value, volume_value in zip(
        timestamps,
        closes,
        volumes,
        strict=True,
    ):
        timestamp = _utc(timestamp_value, field="daily_quote_notional_timestamp")
        if not train_start <= timestamp < train_end:
            raise ValueError("alpha_max_admission_observation_outside_train")
        if observed_day is None:
            observed_day = timestamp.date()
        elif timestamp.date() != observed_day:
            raise ValueError("alpha_max_daily_quote_notional_multiple_days")
        close = _alpha_max_finite_number(
            close_value,
            field="daily_quote_notional_close",
            positive=True,
        )
        volume = _alpha_max_finite_number(
            volume_value,
            field="daily_quote_notional_volume",
            nonnegative=True,
        )
        product = float(np.float64(close) * np.float64(volume))
        if not math.isfinite(product) or product < 0.0:
            raise ValueError("alpha_max_admission_quote_notional_nonfinite")
        normalized_timestamps.append(timestamp)
        products.append(product)
        completed_buckets.add((timestamp.hour // 4) * 4)

    if any(left >= right for left, right in pairwise(normalized_timestamps)):
        raise ValueError("alpha_max_admission_observations_not_strictly_increasing")
    assert observed_day is not None  # non-empty vectors are required above
    return AlphaMaxDailyQuoteNotional(
        day=observed_day,
        quote_notional_usdt=math.fsum(products),
        completed_4h_bucket_hours=tuple(sorted(completed_buckets)),
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxAdmissionDailyCandidateInput:
    """Bounded admission input after raw rows have been discarded day-by-day."""

    symbol: str
    daily_quote_notional: tuple[AlphaMaxDailyQuoteNotional, ...]
    consecutive_completed_daily_bars_before_train: int
    causal_funding_coverage_complete: bool
    unresolved_daily_cross_section_count: int
    partition_integrity_complete: bool = True

    def __post_init__(self) -> None:
        if self.symbol not in ALPHA_MAX_CANDIDATE_SYMBOLS:
            raise ValueError("alpha_max_admission_input_symbol_invalid")
        if type(self.daily_quote_notional) is not tuple or any(
            type(value) is not AlphaMaxDailyQuoteNotional
            for value in self.daily_quote_notional
        ):
            raise TypeError("alpha_max_admission_daily_summaries_must_be_exact_tuple")
        days = tuple(value.day for value in self.daily_quote_notional)
        if any(left >= right for left, right in pairwise(days)):
            raise ValueError("alpha_max_admission_daily_summaries_not_strictly_increasing")
        train_start, train_end = _ROOT_INTERVALS["train"]
        if any(
            not train_start.date() <= current_day < train_end.date()
            for current_day in days
        ):
            raise ValueError("alpha_max_admission_observation_outside_train")
        _admission_nonnegative_int(
            self.consecutive_completed_daily_bars_before_train,
            field="consecutive_completed_daily_bars_before_train",
        )
        _admission_nonnegative_int(
            self.unresolved_daily_cross_section_count,
            field="unresolved_daily_cross_section_count",
        )
        if (
            type(self.causal_funding_coverage_complete) is not bool
            or type(self.partition_integrity_complete) is not bool
        ):
            raise TypeError("alpha_max_admission_input_coverage_must_be_bool")


@dataclass(frozen=True, slots=True)
class AlphaMaxAdmissionComputation:
    artifact: AlphaMaxAdmissionArtifact
    daily_quote_notional_by_symbol: Mapping[str, tuple[AlphaMaxDailyQuoteNotional, ...]]
    daily_quote_notional_sha256_by_symbol: Mapping[str, str]
    canonical_bytes: bytes
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def compute_alpha_max_train_admission(
    inputs: Mapping[str, AlphaMaxAdmissionCandidateInput],
    *,
    input_root_hashes: Mapping[str, str],
    candidate_symbols: Sequence[str] = ALPHA_MAX_CANDIDATE_SYMBOLS,
) -> AlphaMaxAdmissionComputation:
    """Compatibility raw-row API that collapses and discards one day at a time."""
    if tuple(candidate_symbols) != ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_candidate_symbols_mismatch")
    if type(inputs) is not dict or tuple(sorted(inputs)) != ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_admission_input_coverage_mismatch")

    summarized: dict[str, AlphaMaxAdmissionDailyCandidateInput] = {}
    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        candidate = inputs[symbol]
        if type(candidate) is not AlphaMaxAdmissionCandidateInput or candidate.symbol != symbol:
            raise ValueError("alpha_max_admission_input_identity_mismatch")
        observations = candidate.train_observations
        timestamps = tuple(value.timestamp for value in observations)
        if any(left >= right for left, right in pairwise(timestamps)):
            raise ValueError("alpha_max_admission_observations_not_strictly_increasing")

        daily_rows: list[AlphaMaxDailyQuoteNotional] = []
        current_timestamps: list[datetime] = []
        current_closes: list[float] = []
        current_volumes: list[float] = []
        current_day: date | None = None
        for observation in observations:
            observation_day = observation.timestamp.date()
            if current_day is not None and observation_day != current_day:
                daily_rows.append(
                    build_alpha_max_daily_quote_notional(
                        tuple(current_timestamps),
                        tuple(current_closes),
                        tuple(current_volumes),
                    )
                )
                current_timestamps.clear()
                current_closes.clear()
                current_volumes.clear()
            current_day = observation_day
            current_timestamps.append(observation.timestamp)
            current_closes.append(observation.close)
            current_volumes.append(observation.volume)
        if current_timestamps:
            daily_rows.append(
                build_alpha_max_daily_quote_notional(
                    tuple(current_timestamps),
                    tuple(current_closes),
                    tuple(current_volumes),
                )
            )
        summarized[symbol] = AlphaMaxAdmissionDailyCandidateInput(
            symbol=symbol,
            daily_quote_notional=tuple(daily_rows),
            consecutive_completed_daily_bars_before_train=(
                candidate.consecutive_completed_daily_bars_before_train
            ),
            causal_funding_coverage_complete=candidate.causal_funding_coverage_complete,
            unresolved_daily_cross_section_count=(
                candidate.unresolved_daily_cross_section_count
            ),
            partition_integrity_complete=candidate.partition_integrity_complete,
        )
    return compute_alpha_max_train_admission_from_daily_summaries(
        summarized,
        input_root_hashes=input_root_hashes,
        candidate_symbols=candidate_symbols,
    )


def compute_alpha_max_train_admission_from_daily_summaries(
    inputs: Mapping[str, AlphaMaxAdmissionDailyCandidateInput],
    *,
    input_root_hashes: Mapping[str, str],
    candidate_symbols: Sequence[str] = ALPHA_MAX_CANDIDATE_SYMBOLS,
) -> AlphaMaxAdmissionComputation:
    """Compute admission from strict daily summaries without re-summing them."""
    if tuple(candidate_symbols) != ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_candidate_symbols_mismatch")
    if type(inputs) is not dict or tuple(sorted(inputs)) != ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_admission_input_coverage_mismatch")
    if type(input_root_hashes) is not dict or set(input_root_hashes) != set(
        _ADMISSION_INPUT_ROOT_IDS
    ):
        raise ValueError("alpha_max_admission_input_roots_not_warmup_train")
    normalized_root_hashes = {
        root_id: _require_sha256(
            input_root_hashes[root_id],
            field=f"alpha_max_admission_{root_id}_root_sha256",
        )
        for root_id in _ADMISSION_INPUT_ROOT_IDS
    }
    expected_days = tuple(
        (_ROOT_INTERVALS["train"][0] + timedelta(days=index)).date()
        for index in range(_ADMISSION_DAILY_QUOTE_NOTIONAL_DAYS)
    )
    expected_day_set = set(expected_days)
    expected_buckets = frozenset({0, 4, 8, 12, 16, 20})
    vectors: dict[str, tuple[AlphaMaxDailyQuoteNotional, ...]] = {}
    vector_hashes: dict[str, str] = {}
    per_candidate: dict[str, Any] = {}
    admitted: list[str] = []

    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        candidate = inputs[symbol]
        if (
            type(candidate) is not AlphaMaxAdmissionDailyCandidateInput
            or candidate.symbol != symbol
        ):
            raise ValueError("alpha_max_admission_input_identity_mismatch")
        daily_rows = candidate.daily_quote_notional
        if any(value.day not in expected_day_set for value in daily_rows):
            raise ValueError("alpha_max_admission_observation_outside_train")
        vectors[symbol] = daily_rows
        vector_payload = [value.to_payload() for value in daily_rows]
        vector_hashes[symbol] = _sha256_bytes(_canonical_json_bytes(vector_payload, newline=True))
        quote_values = tuple(value.quote_notional_usdt for value in daily_rows)
        median = alpha_max_type7_quantile(quote_values, 0.50) if quote_values else 0.0
        p10 = alpha_max_type7_quantile(quote_values, 0.10) if quote_values else 0.0
        complete_daily = tuple(value.day for value in daily_rows) == expected_days
        complete_4h = complete_daily and all(
            set(value.completed_4h_bucket_hours) == expected_buckets for value in daily_rows
        )
        statistics = {
            "daily_quote_notional_day_count": len(daily_rows),
            "median_quote_notional_usdt": median,
            "p10_quote_notional_usdt": p10,
            "consecutive_completed_daily_bars_before_train": (
                candidate.consecutive_completed_daily_bars_before_train
            ),
            "readable_monotone_unique_finite_partitions": (
                candidate.partition_integrity_complete
            ),
            "complete_train_daily_keys": complete_daily,
            "complete_train_4h_keys": complete_4h,
            "causal_funding_coverage_complete": candidate.causal_funding_coverage_complete,
            "unresolved_daily_cross_section_count": (
                candidate.unresolved_daily_cross_section_count
            ),
        }
        reasons = _expected_admission_reasons(statistics)
        is_admitted = not reasons
        if is_admitted:
            admitted.append(symbol)
        per_candidate[symbol] = {
            "admitted": is_admitted,
            "reasons": list(reasons),
            "statistics": statistics,
        }

    if len(admitted) < 5:
        raise ValueError("alpha_max_insufficient_train_universe")
    admitted_tuple = tuple(admitted)
    payload = {
        "artifact_kind": "alpha_max_train_admission.v1",
        "phase": "train_admission",
        "selection_inputs": ["warmup", "train"],
        "input_root_hashes": normalized_root_hashes,
        "candidate_symbols": list(ALPHA_MAX_CANDIDATE_SYMBOLS),
        "candidate_symbols_sha256": _symbol_sequence_sha256(ALPHA_MAX_CANDIDATE_SYMBOLS),
        "admitted_symbols": list(admitted_tuple),
        "admitted_symbols_sha256": _symbol_sequence_sha256(admitted_tuple),
        "per_candidate": per_candidate,
    }
    artifact = validate_alpha_max_admission_artifact(payload)
    computation_payload = {
        "artifact_kind": "alpha_max_train_admission_computation.v1",
        "admission_artifact_sha256": artifact.sha256,
        "daily_quote_notional_by_symbol": {
            symbol: [value.to_payload() for value in vectors[symbol]]
            for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS
        },
        "daily_quote_notional_sha256_by_symbol": vector_hashes,
        "input_root_hashes": normalized_root_hashes,
    }
    canonical = _canonical_json_bytes(computation_payload, newline=True)
    return AlphaMaxAdmissionComputation(
        artifact=artifact,
        daily_quote_notional_by_symbol=MappingProxyType(vectors),
        daily_quote_notional_sha256_by_symbol=MappingProxyType(vector_hashes),
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


def _alpha_max_nonempty_token(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\0" in value:
        raise ValueError(f"alpha_max_{field}_invalid")
    return value


def _alpha_max_validate_activation_receipt(
    receipt: ArtifactReadReceipt,
    *,
    artifact_id: str,
    expected_sha256: str,
    expected_byte_count: int,
) -> None:
    if type(receipt) is not ArtifactReadReceipt:
        raise TypeError("alpha_max_activation_receipt_identity_invalid")
    if (
        receipt.artifact_id != artifact_id
        or receipt.requested_path != receipt.canonical_path
        or receipt.pre_fstat_identity != receipt.post_fstat_identity
        or receipt.sha256 != expected_sha256
        or receipt.byte_count != expected_byte_count
    ):
        raise ValueError("alpha_max_activation_receipt_binding_mismatch")


def _alpha_max_utc_text(value: datetime) -> str:
    return _utc(value, field="utc_text").isoformat().replace("+00:00", "Z")


def _alpha_max_capsule_scope(prefix_id: str) -> tuple[str, datetime]:
    _alpha_max_nonempty_token(prefix_id, field="capsule_prefix_id")
    for fold_ids, initial_predecessor in (
        (_ALPHA_MAX_VALIDATION_FOLD_IDS, "purge"),
        (_ALPHA_MAX_HISTORICAL_FOLD_IDS, "embargo"),
    ):
        if prefix_id in fold_ids:
            index = fold_ids.index(prefix_id)
            predecessor = initial_predecessor if index == 0 else fold_ids[index - 1]
            return predecessor, _ALPHA_MAX_FOLD_INTERVALS[prefix_id][0]
    raise ValueError("alpha_max_capsule_prefix_id_invalid")


def _alpha_max_thaw_json_tree(value: Any, *, field: str) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"alpha_max_{field}_key_invalid")
        return {
            key: _alpha_max_thaw_json_tree(child, field=f"{field}_{key}")
            for key, child in value.items()
        }
    if type(value) in {tuple, list}:
        return [
            _alpha_max_thaw_json_tree(child, field=f"{field}_item")
            for child in value
        ]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"alpha_max_{field}_value_invalid")


_ALPHA_MAX_CAPSULE_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "capsule",
        "capsule_sha256",
        "discarded_signal_count",
        "fill_event_count",
        "finalized_children",
        "funding_event_count",
        "manifest_sha256",
        "market_event_count",
        "order_event_count",
        "phase_id",
        "portfolio_mode",
        "trade_count",
        "windows_processed",
    }
)


def _alpha_max_validate_capsule_state_payload(
    state_payload: Mapping[str, Any],
    *,
    capsule_phase_id: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    if not isinstance(state_payload, Mapping):
        raise TypeError("alpha_max_capsule_state_payload_identity_invalid")
    normalized = _alpha_max_thaw_json_tree(
        state_payload,
        field="capsule_state_payload",
    )
    if type(normalized) is not dict or set(normalized) != _ALPHA_MAX_CAPSULE_STATE_KEYS:
        raise ValueError("alpha_max_capsule_state_payload_schema_invalid")
    if (
        normalized["phase_id"] != capsule_phase_id
        or normalized["manifest_sha256"] != manifest_sha256
        or type(normalized["portfolio_mode"]) is not str
        or not normalized["portfolio_mode"]
    ):
        raise ValueError("alpha_max_capsule_state_payload_scope_mismatch")
    _require_sha256(
        normalized["capsule_sha256"],
        field="alpha_max_capsule_state_capsule_sha256",
    )
    capsule = normalized["capsule"]
    finalized = normalized["finalized_children"]
    if type(capsule) is not dict or type(finalized) is not dict:
        raise ValueError("alpha_max_capsule_state_payload_schema_invalid")
    retained_sha256 = capsule.get("sha256")
    capsule_scope = {key: value for key, value in capsule.items() if key != "sha256"}
    if (
        retained_sha256 != normalized["capsule_sha256"]
        or retained_sha256
        != _sha256_bytes(_canonical_json_bytes(capsule_scope, newline=False))
    ):
        raise ValueError("alpha_max_capsule_inner_state_hash_mismatch")
    count_fields = (
        "discarded_signal_count",
        "fill_event_count",
        "funding_event_count",
        "market_event_count",
        "order_event_count",
        "trade_count",
        "windows_processed",
    )
    if any(
        type(normalized[field]) is not int or normalized[field] < 0
        for field in count_fields
    ) or any(
        normalized[field] != 0
        for field in ("fill_event_count", "order_event_count", "trade_count")
    ):
        raise ValueError("alpha_max_capsule_state_count_invalid")
    return normalized


_ALPHA_MAX_CAPSULE_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_kind",
        "boundary_utc",
        "capsule_phase_id",
        "manifest_phase",
        "manifest_sha256",
        "prefix_id",
        "row_id",
        "state_payload",
        "state_sha256",
    }
)


def _alpha_max_parse_capsule_envelope(raw_bytes: bytes) -> dict[str, Any]:
    if type(raw_bytes) is not bytes or not raw_bytes:
        raise TypeError("alpha_max_capsule_envelope_bytes_invalid")
    try:
        envelope = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("alpha_max_capsule_envelope_invalid") from exc
    if (
        type(envelope) is not dict
        or set(envelope) != _ALPHA_MAX_CAPSULE_ENVELOPE_KEYS
        or envelope["artifact_kind"] != "alpha_max_indicator_capsule_envelope.v1"
    ):
        raise ValueError("alpha_max_capsule_envelope_schema_invalid")
    try:
        canonical = _canonical_json_bytes(envelope, newline=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha_max_capsule_envelope_invalid") from exc
    if raw_bytes != canonical:
        raise ValueError("alpha_max_capsule_envelope_not_canonical")
    return envelope


@dataclass(frozen=True, slots=True)
class AlphaMaxCapsuleReceipt:
    row_id: str
    phase: str
    prefix_id: str
    capsule_phase_id: str
    boundary_utc: datetime
    manifest_sha256: str
    state_payload: Mapping[str, Any]
    state_sha256: str
    relative_path: str
    sha256: str
    byte_count: int
    activation_receipt: ArtifactReadReceipt

    def __post_init__(self) -> None:
        _alpha_max_nonempty_token(self.row_id, field="capsule_row_id")
        if self.phase not in _MANIFEST_PHASES:
            raise ValueError("alpha_max_capsule_manifest_phase_invalid")
        _alpha_max_nonempty_token(self.prefix_id, field="capsule_prefix_id")
        expected_predecessor, expected_boundary = _alpha_max_capsule_scope(
            self.prefix_id
        )
        if (
            self.capsule_phase_id != expected_predecessor
            or _utc(self.boundary_utc, field="capsule_boundary_utc")
            != expected_boundary
        ):
            raise ValueError("alpha_max_capsule_causal_scope_mismatch")
        object.__setattr__(self, "boundary_utc", expected_boundary)
        object.__setattr__(
            self,
            "manifest_sha256",
            _require_sha256(
                self.manifest_sha256,
                field="alpha_max_capsule_manifest_sha256",
            ),
        )
        normalized_state = _alpha_max_validate_capsule_state_payload(
            self.state_payload,
            capsule_phase_id=self.capsule_phase_id,
            manifest_sha256=self.manifest_sha256,
        )
        object.__setattr__(self, "state_payload", MappingProxyType(normalized_state))
        expected_state_sha256 = _sha256_bytes(
            _canonical_json_bytes(normalized_state, newline=False)
        )
        if self.state_sha256 != expected_state_sha256:
            raise ValueError("alpha_max_capsule_state_hash_mismatch")
        object.__setattr__(
            self,
            "relative_path",
            _alpha_max_safe_relative_path(
                self.relative_path,
                field="capsule_relative_path",
            ),
        )
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, field="alpha_max_capsule_sha256"),
        )
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("alpha_max_capsule_byte_count_invalid")
        _alpha_max_validate_activation_receipt(
            self.activation_receipt,
            artifact_id="alpha_max_indicator_capsule",
            expected_sha256=self.sha256,
            expected_byte_count=self.byte_count,
        )
        expected_envelope = self.canonical_envelope_bytes(
            row_id=self.row_id,
            phase=self.phase,
            prefix_id=self.prefix_id,
            manifest_sha256=self.manifest_sha256,
            state_payload=self.state_payload,
        )
        if (
            self.sha256 != _sha256_bytes(expected_envelope)
            or self.byte_count != len(expected_envelope)
        ):
            raise ValueError("alpha_max_capsule_envelope_binding_mismatch")

    @property
    def path(self) -> str:
        """Ephemeral activation path; canonical payloads use ``relative_path``."""
        return self.activation_receipt.canonical_path

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike[str],
        *,
        row_id: str,
        phase: str,
        prefix_id: str,
        manifest_sha256: str,
        relative_path: str,
    ) -> AlphaMaxCapsuleReceipt:
        receipt, raw_bytes = read_artifact_bytes(
            path,
            artifact_id="alpha_max_indicator_capsule",
        )
        envelope = _alpha_max_parse_capsule_envelope(raw_bytes)
        expected_predecessor, expected_boundary = _alpha_max_capsule_scope(prefix_id)
        expected_boundary_text = _alpha_max_utc_text(expected_boundary)
        if (
            envelope["row_id"] != row_id
            or envelope["manifest_phase"] != phase
            or envelope["prefix_id"] != prefix_id
            or envelope["capsule_phase_id"] != expected_predecessor
            or envelope["boundary_utc"] != expected_boundary_text
            or envelope["manifest_sha256"] != manifest_sha256
        ):
            raise ValueError("alpha_max_capsule_envelope_scope_mismatch")
        return cls(
            row_id=row_id,
            phase=phase,
            prefix_id=prefix_id,
            capsule_phase_id=expected_predecessor,
            boundary_utc=expected_boundary,
            manifest_sha256=manifest_sha256,
            state_payload=envelope["state_payload"],
            state_sha256=envelope["state_sha256"],
            relative_path=relative_path,
            sha256=receipt.sha256,
            byte_count=receipt.byte_count,
            activation_receipt=receipt,
        )

    @classmethod
    def canonical_envelope_bytes(
        cls,
        *,
        row_id: str,
        phase: str,
        prefix_id: str,
        manifest_sha256: str,
        state_payload: Mapping[str, Any],
    ) -> bytes:
        """Serialize the only accepted capsule envelope for one scored fold."""
        _alpha_max_nonempty_token(row_id, field="capsule_row_id")
        if phase not in _MANIFEST_PHASES:
            raise ValueError("alpha_max_capsule_manifest_phase_invalid")
        predecessor, boundary = _alpha_max_capsule_scope(prefix_id)
        manifest_hash = _require_sha256(
            manifest_sha256,
            field="alpha_max_capsule_manifest_sha256",
        )
        normalized_state = _alpha_max_validate_capsule_state_payload(
            state_payload,
            capsule_phase_id=predecessor,
            manifest_sha256=manifest_hash,
        )
        state_sha256 = _sha256_bytes(
            _canonical_json_bytes(normalized_state, newline=False)
        )
        envelope = {
            "artifact_kind": "alpha_max_indicator_capsule_envelope.v1",
            "boundary_utc": _alpha_max_utc_text(boundary),
            "capsule_phase_id": predecessor,
            "manifest_phase": phase,
            "manifest_sha256": manifest_hash,
            "prefix_id": prefix_id,
            "row_id": row_id,
            "state_payload": normalized_state,
            "state_sha256": state_sha256,
        }
        return _canonical_json_bytes(envelope, newline=True)

    def to_payload(self) -> dict[str, Any]:
        return {
            "boundary_utc": _alpha_max_utc_text(self.boundary_utc),
            "byte_count": self.byte_count,
            "capsule_phase_id": self.capsule_phase_id,
            "manifest_sha256": self.manifest_sha256,
            "phase": self.phase,
            "prefix_id": self.prefix_id,
            "relative_path": self.relative_path,
            "row_id": self.row_id,
            "sha256": self.sha256,
            "state_payload": dict(self.state_payload),
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True)
class AlphaMaxManifestReceipt:
    row_id: str
    phase: str
    relative_path: str
    sha256: str
    byte_count: int
    activation_receipt: ArtifactReadReceipt

    def __post_init__(self) -> None:
        _alpha_max_nonempty_token(self.row_id, field="manifest_receipt_row_id")
        if self.phase not in _MANIFEST_PHASES:
            raise ValueError("alpha_max_manifest_receipt_phase_invalid")
        object.__setattr__(
            self,
            "relative_path",
            _alpha_max_safe_relative_path(
                self.relative_path,
                field="manifest_receipt_relative_path",
            ),
        )
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, field="alpha_max_manifest_receipt_sha256"),
        )
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("alpha_max_manifest_receipt_byte_count_invalid")
        _alpha_max_validate_activation_receipt(
            self.activation_receipt,
            artifact_id="alpha_max_engine_portfolio_manifest",
            expected_sha256=self.sha256,
            expected_byte_count=self.byte_count,
        )

    @property
    def path(self) -> str:
        """Ephemeral activation path; canonical payloads use ``relative_path``."""
        return self.activation_receipt.canonical_path

    @classmethod
    def from_materialization(
        cls,
        materialization: AlphaMaxManifestMaterialization,
        *,
        phase: str,
        relative_path: str,
    ) -> AlphaMaxManifestReceipt:
        if type(materialization) is not AlphaMaxManifestMaterialization:
            raise TypeError("alpha_max_manifest_materialization_identity_invalid")
        payload = materialization.payload
        children = payload.get("children")
        if type(children) is not list or not children:
            raise ValueError("alpha_max_manifest_receipt_children_invalid")
        row_ids = {str(child.get("candidate_id") or "") for child in children}
        row_id = Path(materialization.path).stem
        if not row_id or any(not value for value in row_ids):
            raise ValueError("alpha_max_manifest_receipt_row_id_invalid")
        receipt, _ = read_artifact_bytes(
            materialization.path,
            artifact_id="alpha_max_engine_portfolio_manifest",
        )
        if receipt.sha256 != materialization.sha256:
            raise ValueError("alpha_max_manifest_receipt_sha256_mismatch")
        return cls(
            row_id=row_id,
            phase=phase,
            relative_path=relative_path,
            sha256=receipt.sha256,
            byte_count=receipt.byte_count,
            activation_receipt=receipt,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "phase": self.phase,
            "relative_path": self.relative_path,
            "row_id": self.row_id,
            "sha256": self.sha256,
        }


def _alpha_max_funding_row_payload(row: AlphaMaxFundingBoundaryLedgerRow) -> dict[str, Any]:
    if type(row) is not AlphaMaxFundingBoundaryLedgerRow or row.payment is None:
        raise ValueError("alpha_max_funding_reconciliation_row_invalid")
    numeric = (
        row.qty,
        row.rate,
        row.price,
        row.payment,
    )
    if any(type(value) is not float or not math.isfinite(value) for value in numeric):
        raise ValueError("alpha_max_funding_reconciliation_row_invalid")
    return {
        "boundary_ms": row.boundary_ms,
        "payment": row.payment,
        "price": row.price,
        "price_close_timestamp_ms": row.price_close_timestamp_ms,
        "price_row_timestamp_ms": row.price_row_timestamp_ms,
        "qty": row.qty,
        "rate": row.rate,
        "rate_source_timestamp_ms": row.rate_source_timestamp_ms,
        "symbol": row.symbol,
    }


@dataclass(frozen=True, slots=True)
class AlphaMaxReconciliationEvidence:
    pricing_trace_count: int
    application_count: int
    no_fill_attempt_count: int
    zero_applied_application_count: int
    pricing_trace_hashes: tuple[str, ...]
    application_trace_hashes: tuple[str, ...]
    model_commission_total: float
    applied_commission_total: float
    portfolio_fee_total: float
    funding_payment_total: float
    portfolio_funding_total: float
    liquidation_cost_total: float
    portfolio_liquidation_total: float
    pricing_application_bijection: bool
    no_fill_excluded_from_bijection: bool
    fee_reconciled: bool
    funding_reconciled: bool
    liquidation_reconciled: bool
    complete: bool
    canonical_bytes: bytes
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def reconcile_alpha_max_cost_attribution(
    pricing_traces: Sequence[ExecutionPricingTrace],
    applications: Sequence[FillApplicationAttribution],
    no_fill_attempts: Sequence[NoFillAttempt],
    funding_ledger: Sequence[AlphaMaxFundingBoundaryLedgerRow],
    *,
    portfolio_fee_total: float,
    portfolio_funding_total: float,
    liquidation_cost_total: float = 0.0,
    portfolio_liquidation_total: float = 0.0,
) -> AlphaMaxReconciliationEvidence:
    """Prove the positive-trace/application bijection and each separate cost layer."""
    traces = tuple(pricing_traces)
    application_rows = tuple(applications)
    no_fills = tuple(no_fill_attempts)
    funding_rows = tuple(funding_ledger)
    if any(type(value) is not ExecutionPricingTrace for value in traces):
        raise TypeError("alpha_max_pricing_trace_identity_invalid")
    if any(type(value) is not FillApplicationAttribution for value in application_rows):
        raise TypeError("alpha_max_application_identity_invalid")
    if any(type(value) is not NoFillAttempt for value in no_fills):
        raise TypeError("alpha_max_no_fill_identity_invalid")
    trace_payloads = tuple(value.to_payload() for value in traces)
    application_payloads = tuple(value.to_payload() for value in application_rows)
    no_fill_payloads = tuple(value.to_payload() for value in no_fills)
    funding_payloads = tuple(_alpha_max_funding_row_payload(value) for value in funding_rows)
    funding_keys = tuple((value.boundary_ms, value.symbol) for value in funding_rows)
    if len(funding_keys) != len(set(funding_keys)) or funding_keys != tuple(sorted(funding_keys)):
        raise ValueError("alpha_max_funding_reconciliation_ledger_order")
    trace_id_counts = Counter(id(value) for value in traces)
    application_trace_id_counts = Counter(id(value.pricing_trace) for value in application_rows)
    if (
        any(count != 1 for count in trace_id_counts.values())
        or any(count != 1 for count in application_trace_id_counts.values())
        or trace_id_counts != application_trace_id_counts
    ):
        raise ValueError("alpha_max_pricing_application_bijection")
    trace_hashes = tuple(execution_pricing_trace_sha256(value) for value in traces)
    application_hashes = tuple(value.pricing_trace_hash for value in application_rows)
    if Counter(trace_hashes) != Counter(application_hashes):
        raise ValueError("alpha_max_pricing_application_bijection")
    if any(value.executed_qty != 0.0 for value in no_fills):
        raise ValueError("alpha_max_no_fill_exclusion_invalid")

    model_commission_total = math.fsum(value.commission for value in traces)
    applied_commission_total = math.fsum(value.applied_commission for value in application_rows)
    fee_total = _alpha_max_finite_number(
        portfolio_fee_total,
        field="portfolio_fee_total",
        nonnegative=True,
    )
    funding_total = _alpha_max_finite_number(
        portfolio_funding_total,
        field="portfolio_funding_total",
    )
    liquidation_total = _alpha_max_finite_number(
        liquidation_cost_total,
        field="liquidation_cost_total",
        nonnegative=True,
    )
    portfolio_liquidation = _alpha_max_finite_number(
        portfolio_liquidation_total,
        field="portfolio_liquidation_total",
        nonnegative=True,
    )
    funding_payment_total = math.fsum(float(value.payment) for value in funding_rows)
    fee_reconciled = math.isclose(
        applied_commission_total + portfolio_liquidation,
        fee_total,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    funding_reconciled = math.isclose(
        funding_payment_total,
        funding_total,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    liquidation_reconciled = math.isclose(
        liquidation_total,
        portfolio_liquidation,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if not fee_reconciled:
        raise ValueError("alpha_max_fee_reconciliation")
    if not funding_reconciled:
        raise ValueError("alpha_max_funding_reconciliation")
    if not liquidation_reconciled:
        raise ValueError("alpha_max_liquidation_reconciliation")
    payload = {
        "application_count": len(application_rows),
        "application_trace_hashes": list(application_hashes),
        "applications": list(application_payloads),
        "applied_commission_total": applied_commission_total,
        "artifact_kind": "alpha_max_cost_reconciliation.v1",
        "complete": True,
        "fee_reconciled": fee_reconciled,
        "funding_ledger": list(funding_payloads),
        "funding_payment_total": funding_payment_total,
        "funding_reconciled": funding_reconciled,
        "liquidation_cost_total": liquidation_total,
        "liquidation_reconciled": liquidation_reconciled,
        "model_commission_total": model_commission_total,
        "no_fill_attempt_count": len(no_fills),
        "no_fill_attempts": list(no_fill_payloads),
        "no_fill_excluded_from_bijection": True,
        "portfolio_fee_total": fee_total,
        "portfolio_funding_total": funding_total,
        "portfolio_liquidation_total": portfolio_liquidation,
        "pricing_application_bijection": True,
        "pricing_trace_count": len(traces),
        "pricing_trace_hashes": list(trace_hashes),
        "pricing_traces": list(trace_payloads),
        "zero_applied_application_count": sum(
            value.application_status == "rejected" for value in application_rows
        ),
    }
    canonical = _canonical_json_bytes(payload, newline=True)
    return AlphaMaxReconciliationEvidence(
        pricing_trace_count=len(traces),
        application_count=len(application_rows),
        no_fill_attempt_count=len(no_fills),
        zero_applied_application_count=payload["zero_applied_application_count"],
        pricing_trace_hashes=trace_hashes,
        application_trace_hashes=application_hashes,
        model_commission_total=model_commission_total,
        applied_commission_total=applied_commission_total,
        portfolio_fee_total=fee_total,
        funding_payment_total=funding_payment_total,
        portfolio_funding_total=funding_total,
        liquidation_cost_total=liquidation_total,
        portfolio_liquidation_total=portfolio_liquidation,
        pricing_application_bijection=True,
        no_fill_excluded_from_bijection=True,
        fee_reconciled=True,
        funding_reconciled=True,
        liquidation_reconciled=True,
        complete=True,
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


_ALPHA_MAX_COMPARISON_ROLES: Final[frozenset[str]] = frozenset(
    {"prelock_selection", "historical_report"}
)
_ALPHA_MAX_EARLY_GATE_ORDER: Final[tuple[str, ...]] = (
    "dsr",
    "spa",
    "pbo",
    "positive_metrics",
    "native_data_coverage",
    "hash_validity",
    "funding_coverage",
    "manifest_validity",
    "reconciliation",
    "ruin",
)


@dataclass(frozen=True, slots=True)
class AlphaMaxGateInput:
    """Nominal-30-bps matched-domain input to the sole fixed gate pipeline."""

    row_id: str
    comparison_role: str
    evidence_tier: str
    comparison_valid: bool
    nominal_cost_bps: int
    cumulative_return: float
    cagr: float
    calmar: float
    net_sharpe: float
    full_event_mdd: float
    reporting_4h_mdd: float
    dsr: float
    spa_pvalue: float
    pbo: float
    native_data_coverage_complete: bool
    funding_coverage_complete: bool
    hash_valid: bool
    manifest_valid: bool
    reconciliation_complete: bool
    ruin: bool
    raw_root_set_sha256: str
    feature_root_set_sha256: str
    universe_sha256: str
    calendar_sha256: str
    seed_schedule_sha256: str

    def __post_init__(self) -> None:
        _alpha_max_nonempty_token(self.row_id, field="gate_row_id")
        if self.comparison_role not in _ALPHA_MAX_COMPARISON_ROLES:
            raise ValueError("alpha_max_gate_comparison_role_invalid")
        if self.evidence_tier not in {"actual_engine", "diagnostic", "identity"}:
            raise ValueError("alpha_max_gate_evidence_tier_invalid")
        boolean_fields = (
            "comparison_valid",
            "native_data_coverage_complete",
            "funding_coverage_complete",
            "hash_valid",
            "manifest_valid",
            "reconciliation_complete",
            "ruin",
        )
        if any(type(getattr(self, field)) is not bool for field in boolean_fields):
            raise TypeError("alpha_max_gate_boolean_invalid")
        if self.comparison_valid and self.evidence_tier != "actual_engine":
            raise ValueError("alpha_max_gate_nonengine_selection_forbidden")
        if type(self.nominal_cost_bps) is not int or self.nominal_cost_bps != 30:
            raise ValueError("alpha_max_gate_not_nominal_30_bps")
        for field in (
            "cumulative_return",
            "cagr",
            "calmar",
            "net_sharpe",
            "full_event_mdd",
            "reporting_4h_mdd",
            "dsr",
            "spa_pvalue",
            "pbo",
        ):
            value = _alpha_max_finite_number(getattr(self, field), field=f"gate_{field}")
            if field in {"full_event_mdd", "reporting_4h_mdd"} and not 0.0 <= value <= 1.0:
                raise ValueError(f"alpha_max_gate_{field}_invalid")
            object.__setattr__(self, field, value)
        for field in ("dsr", "spa_pvalue", "pbo"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(f"alpha_max_gate_{field}_invalid")
        for field in (
            "raw_root_set_sha256",
            "feature_root_set_sha256",
            "universe_sha256",
            "calendar_sha256",
            "seed_schedule_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), field=f"alpha_max_gate_{field}"),
            )

    @property
    def gate_mdd(self) -> float:
        return max(self.full_event_mdd, self.reporting_4h_mdd)

    @property
    def rank_key(self) -> tuple[float, float, float, float, float, str]:
        return (
            -self.cumulative_return,
            -self.cagr,
            -self.calmar,
            -self.net_sharpe,
            self.gate_mdd,
            self.row_id,
        )

    @property
    def comparison_domain(self) -> tuple[Any, ...]:
        return (
            self.comparison_role,
            self.nominal_cost_bps,
            self.raw_root_set_sha256,
            self.feature_root_set_sha256,
            self.universe_sha256,
            self.calendar_sha256,
            self.seed_schedule_sha256,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "calendar_sha256": self.calendar_sha256,
            "cagr": self.cagr,
            "calmar": self.calmar,
            "comparison_role": self.comparison_role,
            "comparison_valid": self.comparison_valid,
            "cumulative_return": self.cumulative_return,
            "dsr": self.dsr,
            "evidence_tier": self.evidence_tier,
            "feature_root_set_sha256": self.feature_root_set_sha256,
            "full_event_mdd": self.full_event_mdd,
            "funding_coverage_complete": self.funding_coverage_complete,
            "gate_mdd": self.gate_mdd,
            "hash_valid": self.hash_valid,
            "manifest_valid": self.manifest_valid,
            "native_data_coverage_complete": self.native_data_coverage_complete,
            "net_sharpe": self.net_sharpe,
            "nominal_cost_bps": self.nominal_cost_bps,
            "pbo": self.pbo,
            "raw_root_set_sha256": self.raw_root_set_sha256,
            "reconciliation_complete": self.reconciliation_complete,
            "reporting_4h_mdd": self.reporting_4h_mdd,
            "row_id": self.row_id,
            "ruin": self.ruin,
            "seed_schedule_sha256": self.seed_schedule_sha256,
            "spa_pvalue": self.spa_pvalue,
            "universe_sha256": self.universe_sha256,
        }


@dataclass(frozen=True, slots=True)
class AlphaMaxGateDecision:
    row_id: str
    eligible: bool
    evaluated_gates: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    gate_mdd: float | None
    mdd_band: str
    comparator_row_id: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "comparator_row_id": self.comparator_row_id,
            "eligible": self.eligible,
            "evaluated_gates": list(self.evaluated_gates),
            "gate_mdd": self.gate_mdd,
            "mdd_band": self.mdd_band,
            "rejection_reasons": list(self.rejection_reasons),
            "row_id": self.row_id,
        }


@dataclass(frozen=True, slots=True)
class AlphaMaxSelectionResult:
    role: str
    decisions: tuple[AlphaMaxGateDecision, ...]
    ranked_candidate_ids: tuple[str, ...]
    prelock_champion: str | None
    selected_candidate_id: str | None
    historical_evaluation_leader: str | None
    canonical_bytes: bytes
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _alpha_max_early_gate_failure(candidate: AlphaMaxGateInput) -> tuple[str, str] | None:
    if candidate.dsr < 0.90:
        return "dsr", "dsr_below_threshold"
    if candidate.spa_pvalue > 0.05:
        return "spa", "spa_above_threshold"
    if candidate.pbo > 0.50:
        return "pbo", "pbo_above_threshold"
    for field in ("cumulative_return", "cagr", "calmar", "net_sharpe"):
        if getattr(candidate, field) <= 0.0:
            return "positive_metrics", f"nonpositive_{field}"
    if not candidate.native_data_coverage_complete:
        return "native_data_coverage", "native_data_coverage_incomplete"
    if not candidate.hash_valid:
        return "hash_validity", "hash_validity_failed"
    if not candidate.funding_coverage_complete:
        return "funding_coverage", "funding_coverage_incomplete"
    if not candidate.manifest_valid:
        return "manifest_validity", "manifest_validity_failed"
    if not candidate.reconciliation_complete:
        return "reconciliation", "reconciliation_incomplete"
    if candidate.ruin:
        return "ruin", "ruin_detected"
    return None


def _alpha_max_gate_prefix(failed_gate: str | None) -> tuple[str, ...]:
    if failed_gate is None:
        return _ALPHA_MAX_EARLY_GATE_ORDER
    index = _ALPHA_MAX_EARLY_GATE_ORDER.index(failed_gate)
    return _ALPHA_MAX_EARLY_GATE_ORDER[: index + 1]


def _alpha_max_select_gate_inputs(
    candidates: Sequence[AlphaMaxGateInput | AlphaMaxTerminalGateEvidence],
    *,
    role: str,
) -> AlphaMaxSelectionResult:
    values = tuple(candidates)
    if any(
        type(value) not in {AlphaMaxGateInput, AlphaMaxTerminalGateEvidence}
        for value in values
    ):
        raise TypeError("alpha_max_gate_input_identity_invalid")
    if any(value.comparison_role != role for value in values):
        raise ValueError("alpha_max_gate_role_mismatch")
    row_ids = tuple(value.row_id for value in values)
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("alpha_max_gate_duplicate_row_id")
    normal_values = tuple(
        value
        for value in values
        if type(value) is AlphaMaxGateInput
    )
    terminal_values = tuple(
        value
        for value in values
        if type(value) is AlphaMaxTerminalGateEvidence
    )
    common_domains = tuple(
        (
            value.comparison_role,
            value.nominal_cost_bps,
            value.raw_root_set_sha256,
            value.feature_root_set_sha256,
            value.universe_sha256,
            value.seed_schedule_sha256,
        )
        for value in values
    )
    if common_domains and any(value != common_domains[0] for value in common_domains[1:]):
        raise ValueError("alpha_max_comparison_domain_mismatch")
    valid = tuple(
        value
        for value in normal_values
        if value.comparison_valid and value.evidence_tier == "actual_engine"
    )
    if valid and any(value.comparison_domain != valid[0].comparison_domain for value in valid[1:]):
        raise ValueError("alpha_max_comparison_domain_mismatch")

    decisions: dict[str, AlphaMaxGateDecision] = {
        value.row_id: AlphaMaxGateDecision(
            row_id=value.row_id,
            eligible=False,
            evaluated_gates=("ruin",),
            rejection_reasons=("ruin_detected",),
            gate_mdd=None,
            mdd_band="terminal",
            comparator_row_id=None,
        )
        for value in terminal_values
    }
    early_pass: list[AlphaMaxGateInput] = []
    for candidate in sorted(normal_values, key=lambda value: value.row_id):
        if not candidate.comparison_valid or candidate.evidence_tier != "actual_engine":
            decisions[candidate.row_id] = AlphaMaxGateDecision(
                row_id=candidate.row_id,
                eligible=False,
                evaluated_gates=("evidence",),
                rejection_reasons=("incomplete_engine_evidence",),
                gate_mdd=candidate.gate_mdd,
                mdd_band="not_evaluated",
                comparator_row_id=None,
            )
            continue
        failure = _alpha_max_early_gate_failure(candidate)
        if failure is not None:
            failed_gate, reason = failure
            decisions[candidate.row_id] = AlphaMaxGateDecision(
                row_id=candidate.row_id,
                eligible=False,
                evaluated_gates=_alpha_max_gate_prefix(failed_gate),
                rejection_reasons=(reason,),
                gate_mdd=candidate.gate_mdd,
                mdd_band="not_evaluated",
                comparator_row_id=None,
            )
            continue
        early_pass.append(candidate)

    normal = tuple(value for value in early_pass if value.gate_mdd <= 0.30)
    best_normal = min(normal, key=lambda value: value.rank_key) if normal else None
    eligible: list[AlphaMaxGateInput] = []
    for candidate in early_pass:
        evaluated = (*_ALPHA_MAX_EARLY_GATE_ORDER, "mdd")
        if candidate.gate_mdd <= 0.30:
            decision = AlphaMaxGateDecision(
                row_id=candidate.row_id,
                eligible=True,
                evaluated_gates=evaluated,
                rejection_reasons=(),
                gate_mdd=candidate.gate_mdd,
                mdd_band="normal",
                comparator_row_id=None,
            )
            eligible.append(candidate)
        elif candidate.gate_mdd > 0.35:
            decision = AlphaMaxGateDecision(
                row_id=candidate.row_id,
                eligible=False,
                evaluated_gates=evaluated,
                rejection_reasons=("mdd_above_hard_limit",),
                gate_mdd=candidate.gate_mdd,
                mdd_band="hard_reject",
                comparator_row_id=None,
            )
        elif best_normal is None:
            decision = AlphaMaxGateDecision(
                row_id=candidate.row_id,
                eligible=False,
                evaluated_gates=evaluated,
                rejection_reasons=("soft_mdd_requires_normal_comparator",),
                gate_mdd=candidate.gate_mdd,
                mdd_band="soft",
                comparator_row_id=None,
            )
        elif candidate.cagr > best_normal.cagr and candidate.calmar > best_normal.calmar:
            decision = AlphaMaxGateDecision(
                row_id=candidate.row_id,
                eligible=True,
                evaluated_gates=evaluated,
                rejection_reasons=(),
                gate_mdd=candidate.gate_mdd,
                mdd_band="soft",
                comparator_row_id=best_normal.row_id,
            )
            eligible.append(candidate)
        else:
            decision = AlphaMaxGateDecision(
                row_id=candidate.row_id,
                eligible=False,
                evaluated_gates=evaluated,
                rejection_reasons=("soft_mdd_not_strictly_superior_to_best_normal",),
                gate_mdd=candidate.gate_mdd,
                mdd_band="soft",
                comparator_row_id=best_normal.row_id,
            )
        decisions[candidate.row_id] = decision

    ranked = tuple(value.row_id for value in sorted(eligible, key=lambda value: value.rank_key))
    leader = ranked[0] if ranked else None
    prelock = leader if role == "prelock_selection" else None
    historical = leader if role == "historical_report" else None
    payload = {
        "artifact_kind": (
            "alpha_max_prelock_selection.v1"
            if role == "prelock_selection"
            else "alpha_max_historical_report_ranking.v1"
        ),
        "decisions": [decisions[key].to_payload() for key in sorted(decisions)],
        "historical_evaluation_leader": historical,
        "prelock_champion": prelock,
        "ranked_candidate_ids": list(ranked),
        "role": role,
        "selected_candidate_id": prelock,
    }
    canonical = _canonical_json_bytes(payload, newline=True)
    return AlphaMaxSelectionResult(
        role=role,
        decisions=tuple(decisions[key] for key in sorted(decisions)),
        ranked_candidate_ids=ranked,
        prelock_champion=prelock,
        selected_candidate_id=prelock,
        historical_evaluation_leader=historical,
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


def _alpha_max_validate_domain_fold(domain: str, split_or_fold_id: str) -> None:
    expected_folds = (
        _ALPHA_MAX_VALIDATION_FOLD_IDS
        if domain == "validation"
        else _ALPHA_MAX_HISTORICAL_FOLD_IDS
        if domain == "historical_exposed_evaluation"
        else ()
    )
    if not expected_folds:
        raise ValueError("alpha_max_cost_cell_domain_invalid")
    if split_or_fold_id not in expected_folds:
        raise ValueError("alpha_max_cost_cell_fold_id_invalid")


def _alpha_max_validate_domain_root_seals(
    seals: tuple[AlphaMaxRootSeal, ...],
    *,
    domain: str,
    root_kind: str,
) -> tuple[AlphaMaxRootReceipt, ...]:
    if type(seals) is not tuple or any(type(value) is not AlphaMaxRootSeal for value in seals):
        raise TypeError("alpha_max_actual_run_root_seals_invalid")
    expected_ids = _ALPHA_MAX_DOMAIN_ROOT_IDS.get(domain)
    if expected_ids is None or tuple(value.root_id for value in seals) != expected_ids:
        raise ValueError("alpha_max_cost_cell_root_domain_mismatch")
    if any(value.root_kind != root_kind for value in seals):
        raise ValueError("alpha_max_cost_cell_root_receipt_kind_mismatch")
    if seals[0].end_utc != seals[1].start_utc:
        raise ValueError("alpha_max_cost_cell_root_sequence_not_adjacent")
    return tuple(value.to_receipt() for value in seals)


def _alpha_max_payload_sequence_sha256(payloads: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(payloads), newline=True))


def _alpha_max_root_set_sha256(receipts: tuple[AlphaMaxRootReceipt, ...]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes([value.to_payload() for value in receipts], newline=True)
    )


def alpha_max_seed_schedule_sha256(domain: str) -> str:
    fold_ids = _ALPHA_MAX_DOMAIN_FOLD_IDS.get(domain)
    if fold_ids is None:
        raise ValueError("alpha_max_seed_schedule_domain_invalid")
    payload = [
        {
            "fold_id": fold_id,
            "nominal_cost_bps": cost,
            "seed": alpha_max_common_rng_seed(fold_id, cost),
        }
        for fold_id in fold_ids
        for cost in sorted(_ALPHA_MAX_COST_CELLS)
    ]
    return _sha256_bytes(_canonical_json_bytes(payload, newline=True))


def _alpha_max_artifact_receipt_payload(receipt: ArtifactReadReceipt) -> dict[str, Any]:
    if type(receipt) is not ArtifactReadReceipt:
        raise TypeError("alpha_max_config_receipt_identity_invalid")
    if (
        receipt.artifact_id != "alpha_max_config"
        or receipt.requested_path != receipt.canonical_path
        or receipt.pre_fstat_identity != receipt.post_fstat_identity
        or type(receipt.byte_count) is not int
        or receipt.byte_count <= 0
    ):
        raise ValueError("alpha_max_config_receipt_invalid")
    _require_sha256(receipt.sha256, field="alpha_max_config_receipt_sha256")
    return {
        "artifact_id": receipt.artifact_id,
        "byte_count": receipt.byte_count,
        "canonical_path": receipt.canonical_path,
        "post_fstat_identity": list(receipt.post_fstat_identity),
        "pre_fstat_identity": list(receipt.pre_fstat_identity),
        "requested_path": receipt.requested_path,
        "sha256": receipt.sha256,
    }


def _alpha_max_validate_effective_config_bytes(
    effective_config_bytes: bytes,
    effective_config_sha256: str,
    *,
    split_or_fold_id: str,
    nominal_cost_bps: int,
    admitted_symbols: tuple[str, ...],
    runtime_contract_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(effective_config_bytes) is not bytes or not effective_config_bytes:
        raise TypeError("alpha_max_effective_config_bytes_invalid")
    _require_sha256(
        effective_config_sha256,
        field="alpha_max_effective_config_sha256",
    )
    if _sha256_bytes(effective_config_bytes) != effective_config_sha256:
        raise ValueError("alpha_max_effective_config_hash_mismatch")
    try:
        payload = json.loads(effective_config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("alpha_max_effective_config_invalid") from exc
    if (
        type(payload) is not dict
        or not payload
        or _canonical_json_bytes(payload, newline=False) != effective_config_bytes
    ):
        raise ValueError("alpha_max_effective_config_not_canonical")
    required_derived = {"END_DATE", "RANDOM_SEED", "SLIPPAGE_RATE", "START_DATE", "SYMBOLS"}
    required_static = {"DECISION_CADENCE_SECONDS", "INITIAL_CAPITAL"}
    if not required_derived.union(required_static).issubset(payload):
        raise ValueError("alpha_max_effective_config_schema_invalid")
    fold_start, fold_end = _ALPHA_MAX_FOLD_INTERVALS[split_or_fold_id]
    expected_values: dict[str, Any] = {
        "DECISION_CADENCE_SECONDS": 1,
        "END_DATE": _alpha_max_utc_text(fold_end),
        "INITIAL_CAPITAL": _ALPHA_MAX_INITIAL_CAPITAL,
        "RANDOM_SEED": alpha_max_common_rng_seed(split_or_fold_id, nominal_cost_bps),
        "SLIPPAGE_RATE": _ALPHA_MAX_SLIPPAGE_BY_COST[nominal_cost_bps],
        "START_DATE": _alpha_max_utc_text(fold_start),
        "SYMBOLS": list(admitted_symbols),
    }
    if any(payload[key] != value for key, value in expected_values.items()):
        raise ValueError("alpha_max_effective_config_runtime_binding_mismatch")
    if runtime_contract_payload is not None:
        if not isinstance(runtime_contract_payload, Mapping):
            raise TypeError("alpha_max_runtime_contract_payload_identity_invalid")
        allowlist = runtime_contract_payload.get("attribute_allowlist")
        static_attributes = runtime_contract_payload.get("static_attributes")
        if (
            type(allowlist) is not list
            or any(type(value) is not str for value in allowlist)
            or allowlist != sorted(allowlist)
            or len(allowlist) != len(set(allowlist))
            or type(static_attributes) is not dict
            or set(payload) != set(allowlist)
            or set(payload).difference(static_attributes) != required_derived
            or any(payload.get(key) != value for key, value in static_attributes.items())
        ):
            raise ValueError("alpha_max_effective_config_contract_binding_mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class AlphaMaxLiquidationEventEvidence:
    timestamp_ms: int
    symbol: str
    position_qty: float
    entry_price: float
    liquidation_price: float
    close_price: float
    fill_cost: float
    commission: float
    configured_margin_mode: str
    modeled_margin_mode: str

    def __post_init__(self) -> None:
        if type(self.timestamp_ms) is not int or self.timestamp_ms < 0:
            raise ValueError("alpha_max_liquidation_timestamp_invalid")
        if self.symbol not in ALPHA_MAX_CANDIDATE_SYMBOLS:
            raise ValueError("alpha_max_liquidation_symbol_invalid")
        position_qty = _alpha_max_finite_number(
            self.position_qty,
            field="liquidation_position_qty",
        )
        if position_qty == 0.0:
            raise ValueError("alpha_max_liquidation_position_qty_invalid")
        for field in ("entry_price", "liquidation_price", "close_price", "fill_cost"):
            _alpha_max_finite_number(
                getattr(self, field),
                field=f"liquidation_{field}",
                positive=True,
            )
        _alpha_max_finite_number(
            self.commission,
            field="liquidation_commission",
            nonnegative=True,
        )
        _alpha_max_nonempty_token(
            self.configured_margin_mode,
            field="liquidation_configured_margin_mode",
        )
        _alpha_max_nonempty_token(
            self.modeled_margin_mode,
            field="liquidation_modeled_margin_mode",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "close_price": self.close_price,
            "configured_margin_mode": self.configured_margin_mode,
            "commission": self.commission,
            "entry_price": self.entry_price,
            "fill_cost": self.fill_cost,
            "liquidation_price": self.liquidation_price,
            "modeled_margin_mode": self.modeled_margin_mode,
            "position_qty": self.position_qty,
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
        }


def _alpha_max_normalize_liquidation_events(
    values: Sequence[Mapping[str, object]],
) -> tuple[AlphaMaxLiquidationEventEvidence, ...]:
    expected_keys = {
        "time",
        "symbol",
        "position_qty",
        "entry_price",
        "liquidation_price",
        "close_price",
        "fill_cost",
        "commission",
        "configured_margin_mode",
        "modeled_margin_mode",
    }
    events: list[AlphaMaxLiquidationEventEvidence] = []
    for raw in values:
        if type(raw) is not dict or set(raw) != expected_keys:
            raise ValueError("alpha_max_liquidation_event_schema_invalid")
        timestamp = _utc(raw["time"], field="liquidation_event_time")  # type: ignore[arg-type]
        events.append(
            AlphaMaxLiquidationEventEvidence(
                timestamp_ms=_epoch_ms(timestamp),
                symbol=str(raw["symbol"]),
                position_qty=_alpha_max_finite_number(
                    raw["position_qty"],
                    field="liquidation_position_qty",
                ),
                entry_price=_alpha_max_finite_number(
                    raw["entry_price"],
                    field="liquidation_entry_price",
                    positive=True,
                ),
                liquidation_price=_alpha_max_finite_number(
                    raw["liquidation_price"],
                    field="liquidation_liquidation_price",
                    positive=True,
                ),
                close_price=_alpha_max_finite_number(
                    raw["close_price"],
                    field="liquidation_close_price",
                    positive=True,
                ),
                fill_cost=_alpha_max_finite_number(
                    raw["fill_cost"],
                    field="liquidation_fill_cost",
                    positive=True,
                ),
                commission=_alpha_max_finite_number(
                    raw["commission"],
                    field="liquidation_commission",
                    nonnegative=True,
                ),
                configured_margin_mode=str(raw["configured_margin_mode"]),
                modeled_margin_mode=str(raw["modeled_margin_mode"]),
            )
        )
    result = tuple(events)
    order = tuple((value.timestamp_ms, value.symbol) for value in result)
    if order != tuple(sorted(order)) or len(order) != len(set(order)):
        raise ValueError("alpha_max_liquidation_event_order_invalid")
    return result


def _alpha_max_liquidation_cost_totals(
    liquidations: tuple[AlphaMaxLiquidationEventEvidence, ...],
    applications: tuple[FillApplicationAttribution, ...],
    portfolio_fee_total: float,
) -> tuple[float, float]:
    """Separate event-sealed liquidation commission from attributed fill fees."""
    if any(type(value) is not AlphaMaxLiquidationEventEvidence for value in liquidations):
        raise TypeError("alpha_max_liquidation_identity_invalid")
    if any(type(value) is not FillApplicationAttribution for value in applications):
        raise TypeError("alpha_max_application_identity_invalid")
    fee_total = _alpha_max_finite_number(
        portfolio_fee_total,
        field="portfolio_fee_total",
        nonnegative=True,
    )
    liquidation_cost_total = math.fsum(value.commission for value in liquidations)
    applied_commission_total = math.fsum(
        value.applied_commission for value in applications
    )
    portfolio_liquidation_total = fee_total - applied_commission_total
    if abs(portfolio_liquidation_total) <= 1e-12:
        portfolio_liquidation_total = 0.0
    return liquidation_cost_total, _alpha_max_finite_number(
        portfolio_liquidation_total,
        field="portfolio_liquidation_total",
        nonnegative=True,
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxActualEngineRunReceipt:
    row_id: str
    domain: str
    split_or_fold_id: str
    nominal_cost_bps: int
    seed: int
    raw_root_receipts: tuple[AlphaMaxRootReceipt, ...]
    feature_root_receipts: tuple[AlphaMaxRootReceipt, ...]
    raw_root_set_sha256: str
    feature_root_set_sha256: str
    capsule_receipt: AlphaMaxCapsuleReceipt
    manifest_receipt: AlphaMaxManifestReceipt
    config_receipt: ArtifactReadReceipt
    config_sha256: str
    runtime_contract_sha256: str
    effective_config_bytes: bytes
    effective_config_sha256: str
    runtime_read_audit: tuple[str, ...]
    runtime_read_audit_sha256: str
    admitted_symbols: tuple[str, ...]
    universe_sha256: str
    market_event_count: int
    equity_observation_count: int
    signal_event_count: int
    order_event_count: int
    fill_event_count: int
    trade_count: int
    starting_cash: float
    starting_equity: float
    starting_open_position_count: int
    starting_open_order_count: int
    starting_used_margin: float
    ending_cash: float
    ending_equity: float
    full_event_equity: AlphaMaxStreamingEquityEvidence
    pricing_trace_count: int
    pricing_trace_set_sha256: str
    application_count: int
    application_set_sha256: str
    no_fill_attempt_count: int
    no_fill_attempt_set_sha256: str
    funding_ledger_count: int
    funding_ledger_set_sha256: str
    liquidation_event_count: int
    liquidation_event_set_sha256: str
    liquidation_events: tuple[AlphaMaxLiquidationEventEvidence, ...]
    reconciliation: AlphaMaxReconciliationEvidence
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        _alpha_max_nonempty_token(self.row_id, field="actual_run_row_id")
        _alpha_max_validate_domain_fold(self.domain, self.split_or_fold_id)
        if self.nominal_cost_bps not in _ALPHA_MAX_COST_CELLS:
            raise ValueError("alpha_max_actual_run_nominal_cost_invalid")
        if self.seed != alpha_max_common_rng_seed(
            self.split_or_fold_id,
            self.nominal_cost_bps,
        ):
            raise ValueError("alpha_max_actual_run_seed_mismatch")
        expected_roots = _ALPHA_MAX_DOMAIN_ROOT_IDS[self.domain]
        for receipts, kind in (
            (self.raw_root_receipts, "raw"),
            (self.feature_root_receipts, "feature"),
        ):
            if (
                type(receipts) is not tuple
                or any(type(value) is not AlphaMaxRootReceipt for value in receipts)
                or tuple(value.root_id for value in receipts) != expected_roots
                or any(value.root_kind != kind for value in receipts)
            ):
                raise ValueError("alpha_max_cost_cell_root_domain_mismatch")
        if self.raw_root_set_sha256 != _alpha_max_root_set_sha256(
            self.raw_root_receipts
        ) or self.feature_root_set_sha256 != _alpha_max_root_set_sha256(
            self.feature_root_receipts
        ):
            raise ValueError("alpha_max_actual_run_root_hash_mismatch")
        expected_phase = (
            "validation_train_fit"
            if self.domain == "validation"
            else "prelock_final_refit"
        )
        if (
            type(self.capsule_receipt) is not AlphaMaxCapsuleReceipt
            or type(self.manifest_receipt) is not AlphaMaxManifestReceipt
            or self.capsule_receipt.row_id != self.row_id
            or self.manifest_receipt.row_id != self.row_id
            or self.capsule_receipt.phase != expected_phase
            or self.manifest_receipt.phase != expected_phase
            or self.capsule_receipt.prefix_id != self.split_or_fold_id
            or self.capsule_receipt.manifest_sha256 != self.manifest_receipt.sha256
        ):
            raise ValueError("alpha_max_actual_run_artifact_binding_mismatch")
        _alpha_max_artifact_receipt_payload(self.config_receipt)
        if self.config_sha256 != self.config_receipt.sha256:
            raise ValueError("alpha_max_actual_run_config_binding_mismatch")
        _require_sha256(
            self.runtime_contract_sha256,
            field="alpha_max_actual_run_runtime_contract_sha256",
        )
        admitted = validate_alpha_max_admitted_symbols(
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            self.admitted_symbols,
        )
        if admitted != self.admitted_symbols or self.universe_sha256 != _symbol_sequence_sha256(
            admitted
        ):
            raise ValueError("alpha_max_actual_run_universe_binding_mismatch")
        effective_config = _alpha_max_validate_effective_config_bytes(
            self.effective_config_bytes,
            self.effective_config_sha256,
            split_or_fold_id=self.split_or_fold_id,
            nominal_cost_bps=self.nominal_cost_bps,
            admitted_symbols=admitted,
        )
        if (
            type(self.runtime_read_audit) is not tuple
            or not self.runtime_read_audit
            or any(
                type(value) is not str or value not in effective_config
                for value in self.runtime_read_audit
            )
        ):
            raise ValueError("alpha_max_actual_run_runtime_read_audit_invalid")
        expected_audit_hash = _sha256_bytes(
            _canonical_json_bytes(list(self.runtime_read_audit), newline=False)
        )
        if self.runtime_read_audit_sha256 != expected_audit_hash:
            raise ValueError("alpha_max_actual_run_runtime_read_audit_hash_mismatch")
        if any(
            receipt.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS
            for receipt in (*self.raw_root_receipts, *self.feature_root_receipts)
        ):
            raise ValueError("alpha_max_actual_run_root_universe_mismatch")
        for field in (
            "market_event_count",
            "equity_observation_count",
            "signal_event_count",
            "order_event_count",
            "fill_event_count",
            "trade_count",
        ):
            count = getattr(self, field)
            if type(count) is not int or count < 0:
                raise ValueError(f"alpha_max_actual_run_{field}_invalid")
        if self.market_event_count <= 0:
            raise ValueError("alpha_max_actual_run_market_event_count_invalid")
        starting_cash = _alpha_max_finite_number(
            self.starting_cash,
            field="actual_run_starting_cash",
        )
        starting_equity = _alpha_max_finite_number(
            self.starting_equity,
            field="actual_run_starting_equity",
        )
        starting_margin = _alpha_max_finite_number(
            self.starting_used_margin,
            field="actual_run_starting_used_margin",
            nonnegative=True,
        )
        if (
            starting_cash != _ALPHA_MAX_INITIAL_CAPITAL
            or starting_equity != _ALPHA_MAX_INITIAL_CAPITAL
            or type(self.starting_open_position_count) is not int
            or self.starting_open_position_count != 0
            or type(self.starting_open_order_count) is not int
            or self.starting_open_order_count != 0
            or starting_margin != 0.0
        ):
            raise ValueError("alpha_max_actual_run_flat_start_invalid")
        ending_equity = _alpha_max_finite_number(
            self.ending_equity,
            field="actual_run_ending_equity",
        )
        _alpha_max_finite_number(self.ending_cash, field="actual_run_ending_cash")
        _validate_alpha_max_streaming_equity_evidence(self.full_event_equity)
        fold_start, fold_end = _ALPHA_MAX_FOLD_INTERVALS[self.split_or_fold_id]
        if (
            self.full_event_equity.first_timestamp_ms is None
            or self.full_event_equity.last_timestamp_ms is None
            or not _epoch_ms(fold_start)
            <= self.full_event_equity.first_timestamp_ms
            <= self.full_event_equity.last_timestamp_ms
            <= _epoch_ms(fold_end)
        ):
            raise ValueError("alpha_max_actual_run_streaming_fold_bounds_invalid")
        if self.equity_observation_count != self.full_event_equity.event_count:
            raise ValueError("alpha_max_actual_run_equity_observation_count_mismatch")
        if not math.isclose(
            ending_equity,
            self.full_event_equity.ending_equity,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("alpha_max_actual_run_streaming_equity_mismatch")
        if type(self.reconciliation) is not AlphaMaxReconciliationEvidence or not (
            self.reconciliation.complete
        ):
            raise ValueError("alpha_max_actual_run_reconciliation_incomplete")
        count_bindings = (
            (self.pricing_trace_count, self.reconciliation.pricing_trace_count),
            (self.application_count, self.reconciliation.application_count),
            (self.no_fill_attempt_count, self.reconciliation.no_fill_attempt_count),
            (self.liquidation_event_count, len(self.liquidation_events)),
        )
        if any(
            type(actual) is not int or actual < 0 or actual != expected
            for actual, expected in count_bindings
        ):
            raise ValueError("alpha_max_actual_run_attribution_count_mismatch")
        _validate_alpha_max_engine_event_counts(
            fill_event_count=self.fill_event_count,
            pricing_trace_count=self.pricing_trace_count,
            application_count=self.application_count,
            liquidation_event_count=self.liquidation_event_count,
            trade_count=self.trade_count,
        )
        if type(self.funding_ledger_count) is not int or self.funding_ledger_count < 0:
            raise ValueError("alpha_max_actual_run_funding_count_invalid")
        if (
            type(self.liquidation_events) is not tuple
            or any(
                type(value) is not AlphaMaxLiquidationEventEvidence
                for value in self.liquidation_events
            )
        ):
            raise TypeError("alpha_max_actual_run_liquidation_identity_invalid")
        for field in (
            "pricing_trace_set_sha256",
            "application_set_sha256",
            "no_fill_attempt_set_sha256",
            "funding_ledger_set_sha256",
            "liquidation_event_set_sha256",
        ):
            _require_sha256(getattr(self, field), field=f"alpha_max_actual_run_{field}")
        liquidation_hash = _alpha_max_payload_sequence_sha256(
            [value.to_payload() for value in self.liquidation_events]
        )
        reconciliation_payload = self.reconciliation.to_payload()
        derived_bindings = (
            (
                self.pricing_trace_set_sha256,
                _alpha_max_payload_sequence_sha256(reconciliation_payload["pricing_traces"]),
            ),
            (
                self.application_set_sha256,
                _alpha_max_payload_sequence_sha256(reconciliation_payload["applications"]),
            ),
            (
                self.no_fill_attempt_set_sha256,
                _alpha_max_payload_sequence_sha256(
                    reconciliation_payload["no_fill_attempts"]
                ),
            ),
            (
                self.funding_ledger_set_sha256,
                _alpha_max_payload_sequence_sha256(reconciliation_payload["funding_ledger"]),
            ),
            (self.liquidation_event_set_sha256, liquidation_hash),
        )
        if (
            any(actual != expected for actual, expected in derived_bindings)
            or self.funding_ledger_count != len(reconciliation_payload["funding_ledger"])
        ):
            raise ValueError("alpha_max_actual_run_attribution_hash_mismatch")
        expected_canonical = _canonical_json_bytes(
            _alpha_max_actual_engine_run_payload(self),
            newline=True,
        )
        if (
            type(self.canonical_bytes) is not bytes
            or self.canonical_bytes != expected_canonical
            or self.sha256 != _sha256_bytes(expected_canonical)
        ):
            raise ValueError("alpha_max_actual_run_canonical_mismatch")

    @property
    def ruin_detected(self) -> bool:
        return self.full_event_equity.ruin_detected or self.liquidation_event_count > 0

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _alpha_max_actual_engine_run_payload(
    receipt: AlphaMaxActualEngineRunReceipt,
) -> dict[str, Any]:
    return {
        "admitted_symbols": list(receipt.admitted_symbols),
        "application_count": receipt.application_count,
        "application_set_sha256": receipt.application_set_sha256,
        "artifact_kind": "alpha_max_actual_engine_run_receipt.v1",
        "capsule_receipt": receipt.capsule_receipt.to_payload(),
        "config_receipt": _alpha_max_artifact_receipt_payload(receipt.config_receipt),
        "config_sha256": receipt.config_sha256,
        "domain": receipt.domain,
        "ending_cash": receipt.ending_cash,
        "ending_equity": receipt.ending_equity,
        "effective_config": json.loads(receipt.effective_config_bytes),
        "effective_config_sha256": receipt.effective_config_sha256,
        "equity_observation_count": receipt.equity_observation_count,
        "feature_root_receipts": [value.to_payload() for value in receipt.feature_root_receipts],
        "feature_root_set_sha256": receipt.feature_root_set_sha256,
        "fill_event_count": receipt.fill_event_count,
        "fold_end_utc": _ALPHA_MAX_FOLD_INTERVALS[receipt.split_or_fold_id][1]
        .isoformat()
        .replace("+00:00", "Z"),
        "fold_start_utc": _ALPHA_MAX_FOLD_INTERVALS[receipt.split_or_fold_id][0]
        .isoformat()
        .replace("+00:00", "Z"),
        "full_event_equity": receipt.full_event_equity.to_payload(),
        "funding_ledger_count": receipt.funding_ledger_count,
        "funding_ledger_set_sha256": receipt.funding_ledger_set_sha256,
        "liquidation_event_count": receipt.liquidation_event_count,
        "liquidation_event_set_sha256": receipt.liquidation_event_set_sha256,
        "liquidation_events": [value.to_payload() for value in receipt.liquidation_events],
        "manifest_receipt": receipt.manifest_receipt.to_payload(),
        "market_event_count": receipt.market_event_count,
        "no_fill_attempt_count": receipt.no_fill_attempt_count,
        "no_fill_attempt_set_sha256": receipt.no_fill_attempt_set_sha256,
        "nominal_cost_bps": receipt.nominal_cost_bps,
        "order_event_count": receipt.order_event_count,
        "pricing_trace_count": receipt.pricing_trace_count,
        "pricing_trace_set_sha256": receipt.pricing_trace_set_sha256,
        "raw_root_receipts": [value.to_payload() for value in receipt.raw_root_receipts],
        "raw_root_set_sha256": receipt.raw_root_set_sha256,
        "reconciliation": receipt.reconciliation.to_payload(),
        "row_id": receipt.row_id,
        "ruin_detected": receipt.ruin_detected,
        "runtime_contract_sha256": receipt.runtime_contract_sha256,
        "runtime_read_audit": list(receipt.runtime_read_audit),
        "runtime_read_audit_sha256": receipt.runtime_read_audit_sha256,
        "seed": receipt.seed,
        "signal_event_count": receipt.signal_event_count,
        "split_or_fold_id": receipt.split_or_fold_id,
        "starting_cash": receipt.starting_cash,
        "starting_equity": receipt.starting_equity,
        "starting_open_order_count": receipt.starting_open_order_count,
        "starting_open_position_count": receipt.starting_open_position_count,
        "starting_used_margin": receipt.starting_used_margin,
        "trade_count": receipt.trade_count,
        "universe_sha256": receipt.universe_sha256,
    }


def _validate_alpha_max_engine_event_counts(
    *,
    fill_event_count: int,
    pricing_trace_count: int,
    application_count: int,
    liquidation_event_count: int,
    trade_count: int,
) -> None:
    """Bind ordinary attributed fills and synthetic liquidation fills separately."""
    counts = (
        fill_event_count,
        pricing_trace_count,
        application_count,
        liquidation_event_count,
        trade_count,
    )
    if any(type(value) is not int or value < 0 for value in counts) or (
        pricing_trace_count != application_count
        or fill_event_count != pricing_trace_count + liquidation_event_count
        or trade_count > fill_event_count
        or trade_count < liquidation_event_count
    ):
        raise ValueError("alpha_max_actual_run_engine_count_mismatch")


def _alpha_max_verify_file_receipt(
    *,
    activation_receipt: ArtifactReadReceipt,
    expected_sha256: str,
    expected_byte_count: int,
    artifact_id: str,
) -> None:
    _alpha_max_validate_activation_receipt(
        activation_receipt,
        artifact_id=artifact_id,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
    )
    receipt, _ = read_artifact_bytes(
        activation_receipt.canonical_path,
        artifact_id=artifact_id,
    )
    if receipt != activation_receipt:
        raise ValueError("alpha_max_actual_run_artifact_bytes_mismatch")


def build_alpha_max_actual_engine_run_receipt(
    *,
    row_id: str,
    domain: str,
    split_or_fold_id: str,
    nominal_cost_bps: int,
    raw_root_seals: tuple[AlphaMaxRootSeal, ...],
    feature_root_seals: tuple[AlphaMaxRootSeal, ...],
    capsule_receipt: AlphaMaxCapsuleReceipt,
    manifest_receipt: AlphaMaxManifestReceipt,
    config_receipt: ArtifactReadReceipt,
    config_bytes: bytes,
    runtime_contract_bytes: bytes,
    effective_config_bytes: bytes,
    effective_config_sha256: str,
    runtime_read_audit: tuple[str, ...],
    runtime_read_audit_sha256: str,
    admitted_symbols: tuple[str, ...],
    market_event_count: int,
    signal_event_count: int,
    order_event_count: int,
    fill_event_count: int,
    trade_count: int,
    starting_cash: float,
    starting_equity: float,
    starting_open_position_count: int,
    starting_open_order_count: int,
    starting_used_margin: float,
    ending_cash: float,
    ending_equity: float,
    full_event_equity: AlphaMaxStreamingEquityEvidence,
    pricing_traces: tuple[ExecutionPricingTrace, ...],
    fill_applications: tuple[FillApplicationAttribution, ...],
    no_fill_attempts: tuple[NoFillAttempt, ...],
    funding_ledger: tuple[AlphaMaxFundingBoundaryLedgerRow, ...],
    liquidation_events: tuple[Mapping[str, object], ...],
    portfolio_fee_total: float,
    portfolio_funding_total: float,
) -> AlphaMaxActualEngineRunReceipt:
    """Build the sole typed complete-engine receipt from observed replay evidence."""
    _alpha_max_validate_domain_fold(domain, split_or_fold_id)
    raw_receipts = _alpha_max_validate_domain_root_seals(
        raw_root_seals,
        domain=domain,
        root_kind="raw",
    )
    feature_receipts = _alpha_max_validate_domain_root_seals(
        feature_root_seals,
        domain=domain,
        root_kind="feature",
    )
    if type(config_bytes) is not bytes or type(runtime_contract_bytes) is not bytes:
        raise TypeError("alpha_max_actual_run_contract_bytes_invalid")
    config_payload = _alpha_max_artifact_receipt_payload(config_receipt)
    if (
        len(config_bytes) != config_payload["byte_count"]
        or _sha256_bytes(config_bytes) != config_payload["sha256"]
    ):
        raise ValueError("alpha_max_actual_run_config_bytes_mismatch")
    fresh_config_receipt, fresh_config_bytes = read_artifact_bytes(
        config_receipt.canonical_path,
        artifact_id="alpha_max_config",
    )
    if fresh_config_receipt != config_receipt or fresh_config_bytes != config_bytes:
        raise ValueError("alpha_max_actual_run_config_receipt_stale")
    try:
        runtime_payload = json.loads(runtime_contract_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("alpha_max_actual_run_runtime_contract_invalid") from exc
    if (
        type(runtime_payload) is not dict
        or not runtime_payload
        or _canonical_json_bytes(runtime_payload, newline=False) != runtime_contract_bytes
    ):
        raise ValueError("alpha_max_actual_run_runtime_contract_invalid")
    _alpha_max_verify_file_receipt(
        activation_receipt=capsule_receipt.activation_receipt,
        expected_sha256=capsule_receipt.sha256,
        expected_byte_count=capsule_receipt.byte_count,
        artifact_id="alpha_max_indicator_capsule",
    )
    _alpha_max_verify_file_receipt(
        activation_receipt=manifest_receipt.activation_receipt,
        expected_sha256=manifest_receipt.sha256,
        expected_byte_count=manifest_receipt.byte_count,
        artifact_id="alpha_max_engine_portfolio_manifest",
    )
    if (
        type(pricing_traces) is not tuple
        or type(fill_applications) is not tuple
        or type(no_fill_attempts) is not tuple
        or type(funding_ledger) is not tuple
        or type(liquidation_events) is not tuple
    ):
        raise TypeError("alpha_max_actual_run_evidence_must_be_tuple")
    normalized_liquidations = _alpha_max_normalize_liquidation_events(liquidation_events)
    liquidation_cost_total, portfolio_liquidation_total = (
        _alpha_max_liquidation_cost_totals(
            normalized_liquidations,
            fill_applications,
            portfolio_fee_total,
        )
    )
    reconciliation = reconcile_alpha_max_cost_attribution(
        pricing_traces,
        fill_applications,
        no_fill_attempts,
        funding_ledger,
        portfolio_fee_total=portfolio_fee_total,
        portfolio_funding_total=portfolio_funding_total,
        liquidation_cost_total=liquidation_cost_total,
        portfolio_liquidation_total=portfolio_liquidation_total,
    )
    trace_payloads = tuple(value.to_payload() for value in pricing_traces)
    application_payloads = tuple(value.to_payload() for value in fill_applications)
    no_fill_payloads = tuple(value.to_payload() for value in no_fill_attempts)
    funding_payloads = tuple(_alpha_max_funding_row_payload(value) for value in funding_ledger)
    liquidation_payloads = tuple(value.to_payload() for value in normalized_liquidations)
    admitted = validate_alpha_max_admitted_symbols(
        ALPHA_MAX_CANDIDATE_SYMBOLS,
        admitted_symbols,
    )
    effective_config = _alpha_max_validate_effective_config_bytes(
        effective_config_bytes,
        effective_config_sha256,
        split_or_fold_id=split_or_fold_id,
        nominal_cost_bps=nominal_cost_bps,
        admitted_symbols=admitted,
        runtime_contract_payload=runtime_payload,
    )
    if (
        type(runtime_read_audit) is not tuple
        or not runtime_read_audit
        or any(
            type(value) is not str or value not in effective_config
            for value in runtime_read_audit
        )
        or runtime_read_audit_sha256
        != _sha256_bytes(
            _canonical_json_bytes(list(runtime_read_audit), newline=False)
        )
    ):
        raise ValueError("alpha_max_actual_run_runtime_read_audit_invalid")
    if any(
        seal.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS
        for seal in (*raw_root_seals, *feature_root_seals)
    ):
        raise ValueError("alpha_max_actual_run_root_universe_mismatch")
    values = {
        "row_id": row_id,
        "domain": domain,
        "split_or_fold_id": split_or_fold_id,
        "nominal_cost_bps": nominal_cost_bps,
        "seed": alpha_max_common_rng_seed(split_or_fold_id, nominal_cost_bps),
        "raw_root_receipts": raw_receipts,
        "feature_root_receipts": feature_receipts,
        "raw_root_set_sha256": _alpha_max_root_set_sha256(raw_receipts),
        "feature_root_set_sha256": _alpha_max_root_set_sha256(feature_receipts),
        "capsule_receipt": capsule_receipt,
        "manifest_receipt": manifest_receipt,
        "config_receipt": config_receipt,
        "config_sha256": config_receipt.sha256,
        "runtime_contract_sha256": _sha256_bytes(runtime_contract_bytes),
        "effective_config_bytes": effective_config_bytes,
        "effective_config_sha256": effective_config_sha256,
        "runtime_read_audit": runtime_read_audit,
        "runtime_read_audit_sha256": runtime_read_audit_sha256,
        "admitted_symbols": admitted,
        "universe_sha256": _symbol_sequence_sha256(admitted),
        "market_event_count": market_event_count,
        "equity_observation_count": full_event_equity.event_count,
        "signal_event_count": signal_event_count,
        "order_event_count": order_event_count,
        "fill_event_count": fill_event_count,
        "trade_count": trade_count,
        "starting_cash": starting_cash,
        "starting_equity": starting_equity,
        "starting_open_position_count": starting_open_position_count,
        "starting_open_order_count": starting_open_order_count,
        "starting_used_margin": starting_used_margin,
        "ending_cash": ending_cash,
        "ending_equity": ending_equity,
        "full_event_equity": full_event_equity,
        "pricing_trace_count": len(pricing_traces),
        "pricing_trace_set_sha256": _alpha_max_payload_sequence_sha256(trace_payloads),
        "application_count": len(fill_applications),
        "application_set_sha256": _alpha_max_payload_sequence_sha256(application_payloads),
        "no_fill_attempt_count": len(no_fill_attempts),
        "no_fill_attempt_set_sha256": _alpha_max_payload_sequence_sha256(no_fill_payloads),
        "funding_ledger_count": len(funding_ledger),
        "funding_ledger_set_sha256": _alpha_max_payload_sequence_sha256(funding_payloads),
        "liquidation_event_count": len(normalized_liquidations),
        "liquidation_event_set_sha256": _alpha_max_payload_sequence_sha256(
            liquidation_payloads
        ),
        "liquidation_events": normalized_liquidations,
        "reconciliation": reconciliation,
    }
    temporary = object.__new__(AlphaMaxActualEngineRunReceipt)
    for field, value in values.items():
        object.__setattr__(temporary, field, value)
    canonical = _canonical_json_bytes(_alpha_max_actual_engine_run_payload(temporary), newline=True)
    return AlphaMaxActualEngineRunReceipt(
        **values,
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


def _alpha_max_fold_reporting_calendar(fold_id: str) -> tuple[datetime, ...]:
    try:
        start, end = _ALPHA_MAX_FOLD_INTERVALS[fold_id]
    except KeyError as exc:
        raise ValueError("alpha_max_fold_id_invalid") from exc
    count = int((end - start) / timedelta(hours=4))
    if start + timedelta(hours=4 * count) != end:
        raise AssertionError("alpha-max fold is not four-hour aligned")
    return tuple(start + timedelta(hours=4 * index) for index in range(1, count + 1))


def _alpha_max_primary_stream_sha256(stream: AlphaMaxPrimaryReturnStream) -> str:
    return _sha256_bytes(_canonical_json_bytes(stream.to_payload(), newline=True))


@dataclass(frozen=True, slots=True)
class AlphaMaxNormalizedFoldSegmentEvidence:
    """Same-sink binding for one normalized fold segment and aggregate prefix."""

    fold_id: str
    source_streaming_equity_sha256: str
    source_event_stream_sha256: str
    normalization_scale: float
    normalized_starting_equity: float
    normalized_ending_equity: float
    normalized_segment_event_stream_sha256: str
    event_count: int
    first_timestamp_ms: int
    last_timestamp_ms: int
    aggregate_prefix_event_count: int
    aggregate_prefix_event_stream_sha256: str
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        if self.fold_id not in _ALPHA_MAX_FOLD_INTERVALS:
            raise ValueError("alpha_max_normalized_segment_fold_id_invalid")
        for field in (
            "source_streaming_equity_sha256",
            "source_event_stream_sha256",
            "normalized_segment_event_stream_sha256",
            "aggregate_prefix_event_stream_sha256",
        ):
            _require_sha256(
                getattr(self, field),
                field=f"alpha_max_normalized_segment_{field}",
            )
        for field in (
            "normalization_scale",
            "normalized_starting_equity",
            "normalized_ending_equity",
        ):
            _alpha_max_finite_number(
                getattr(self, field),
                field=f"normalized_segment_{field}",
                positive=True,
            )
        if (
            type(self.event_count) is not int
            or self.event_count <= 0
            or type(self.aggregate_prefix_event_count) is not int
            or self.aggregate_prefix_event_count < self.event_count
            or type(self.first_timestamp_ms) is not int
            or type(self.last_timestamp_ms) is not int
            or self.first_timestamp_ms < 0
            or self.last_timestamp_ms < self.first_timestamp_ms
        ):
            raise ValueError("alpha_max_normalized_segment_counts_invalid")
        canonical = _canonical_json_bytes(
            _alpha_max_normalized_fold_segment_payload(self),
            newline=True,
        )
        if (
            type(self.canonical_bytes) is not bytes
            or self.canonical_bytes != canonical
            or self.sha256 != _sha256_bytes(canonical)
        ):
            raise ValueError("alpha_max_normalized_segment_canonical_mismatch")

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _alpha_max_normalized_fold_segment_payload(
    value: AlphaMaxNormalizedFoldSegmentEvidence,
) -> dict[str, Any]:
    return {
        "aggregate_prefix_event_count": value.aggregate_prefix_event_count,
        "aggregate_prefix_event_stream_sha256": (
            value.aggregate_prefix_event_stream_sha256
        ),
        "artifact_kind": "alpha_max_normalized_fold_segment_evidence.v1",
        "event_count": value.event_count,
        "first_timestamp_ms": value.first_timestamp_ms,
        "fold_id": value.fold_id,
        "last_timestamp_ms": value.last_timestamp_ms,
        "normalization_scale": value.normalization_scale,
        "normalized_ending_equity": value.normalized_ending_equity,
        "normalized_segment_event_stream_sha256": (
            value.normalized_segment_event_stream_sha256
        ),
        "normalized_starting_equity": value.normalized_starting_equity,
        "source_event_stream_sha256": value.source_event_stream_sha256,
        "source_streaming_equity_sha256": value.source_streaming_equity_sha256,
    }


def build_alpha_max_normalized_fold_segment_evidence(
    *,
    fold_id: str,
    source_streaming_equity_sha256: str,
    source_event_stream_sha256: str,
    normalization_scale: float,
    normalized_starting_equity: float,
    normalized_ending_equity: float,
    normalized_segment_event_stream_sha256: str,
    event_count: int,
    first_timestamp_ms: int,
    last_timestamp_ms: int,
    aggregate_prefix_event_count: int,
    aggregate_prefix_event_stream_sha256: str,
) -> AlphaMaxNormalizedFoldSegmentEvidence:
    values = {
        "fold_id": fold_id,
        "source_streaming_equity_sha256": source_streaming_equity_sha256,
        "source_event_stream_sha256": source_event_stream_sha256,
        "normalization_scale": normalization_scale,
        "normalized_starting_equity": normalized_starting_equity,
        "normalized_ending_equity": normalized_ending_equity,
        "normalized_segment_event_stream_sha256": normalized_segment_event_stream_sha256,
        "event_count": event_count,
        "first_timestamp_ms": first_timestamp_ms,
        "last_timestamp_ms": last_timestamp_ms,
        "aggregate_prefix_event_count": aggregate_prefix_event_count,
        "aggregate_prefix_event_stream_sha256": aggregate_prefix_event_stream_sha256,
    }
    temporary = object.__new__(AlphaMaxNormalizedFoldSegmentEvidence)
    for field, value in values.items():
        object.__setattr__(temporary, field, value)
    canonical = _canonical_json_bytes(
        _alpha_max_normalized_fold_segment_payload(temporary),
        newline=True,
    )
    return AlphaMaxNormalizedFoldSegmentEvidence(
        **values,
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxCombinedStreamingEquityEvidence:
    """Binding wrapper for the live normalized full-event fan-out receipt."""

    domain: str
    fold_ids: tuple[str, ...]
    fold_run_sha256s: tuple[str, ...]
    fold_streaming_equity_sha256s: tuple[str, ...]
    fold_event_stream_set_sha256: str
    normalized_fold_segments: tuple[AlphaMaxNormalizedFoldSegmentEvidence, ...]
    streaming_equity: AlphaMaxStreamingEquityEvidence
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        expected_ids = _ALPHA_MAX_DOMAIN_FOLD_IDS.get(self.domain)
        if self.fold_ids != expected_ids:
            raise ValueError("alpha_max_combined_streaming_fold_sequence_invalid")
        if (
            type(self.fold_run_sha256s) is not tuple
            or type(self.fold_streaming_equity_sha256s) is not tuple
            or type(self.normalized_fold_segments) is not tuple
            or len(self.fold_run_sha256s) != len(self.fold_ids)
            or len(self.fold_streaming_equity_sha256s) != len(self.fold_ids)
            or tuple(value.fold_id for value in self.normalized_fold_segments)
            != self.fold_ids
            or any(
                type(value) is not AlphaMaxNormalizedFoldSegmentEvidence
                for value in self.normalized_fold_segments
            )
        ):
            raise ValueError("alpha_max_combined_streaming_fold_hashes_invalid")
        for field in (
            "fold_run_sha256s",
            "fold_streaming_equity_sha256s",
        ):
            for value in getattr(self, field):
                _require_sha256(value, field=f"alpha_max_combined_streaming_{field}")
        _require_sha256(
            self.fold_event_stream_set_sha256,
            field="alpha_max_combined_streaming_event_set_sha256",
        )
        stream = _validate_alpha_max_streaming_equity_evidence(self.streaming_equity)
        if stream.ruin_detected:
            raise ValueError("alpha_max_combined_streaming_terminal_stream_forbidden")
        payload = _alpha_max_combined_streaming_payload(self)
        canonical = _canonical_json_bytes(payload, newline=True)
        if (
            type(self.canonical_bytes) is not bytes
            or self.canonical_bytes != canonical
            or self.sha256 != _sha256_bytes(canonical)
        ):
            raise ValueError("alpha_max_combined_streaming_canonical_mismatch")

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)

    @property
    def event_count(self) -> int:
        return self.streaming_equity.event_count

    @property
    def ending_equity(self) -> float:
        return self.streaming_equity.ending_equity

    @property
    def full_event_mdd(self) -> float:
        return self.streaming_equity.full_event_mdd

    @property
    def ruin_detected(self) -> bool:
        return self.streaming_equity.ruin_detected


def _alpha_max_combined_streaming_payload(
    value: AlphaMaxCombinedStreamingEquityEvidence,
) -> dict[str, Any]:
    return {
        "artifact_kind": "alpha_max_combined_streaming_equity.v1",
        "domain": value.domain,
        "fold_event_stream_set_sha256": value.fold_event_stream_set_sha256,
        "fold_ids": list(value.fold_ids),
        "fold_run_sha256s": list(value.fold_run_sha256s),
        "fold_streaming_equity_sha256s": list(value.fold_streaming_equity_sha256s),
        "normalized_fold_segments": [
            segment.to_payload() for segment in value.normalized_fold_segments
        ],
        "streaming_equity": value.streaming_equity.to_payload(),
    }


def _build_alpha_max_combined_streaming_equity(
    fold_runs: tuple[AlphaMaxFoldRunEvidence, ...],
    live_streaming_equity: AlphaMaxStreamingEquityEvidence,
    normalized_fold_segments: tuple[AlphaMaxNormalizedFoldSegmentEvidence, ...],
) -> AlphaMaxCombinedStreamingEquityEvidence:
    if not fold_runs or any(value.status != "complete" for value in fold_runs):
        raise ValueError("alpha_max_combined_streaming_terminal_fold_forbidden")
    live_stream = _validate_alpha_max_streaming_equity_evidence(live_streaming_equity)
    if live_stream.ruin_detected:
        raise ValueError("alpha_max_combined_streaming_terminal_stream_forbidden")
    actual_runs = tuple(value.actual_engine_run for value in fold_runs)
    domain = actual_runs[0].domain
    fold_ids = tuple(value.split_or_fold_id for value in actual_runs)
    if fold_ids != _ALPHA_MAX_DOMAIN_FOLD_IDS[domain]:
        raise ValueError("alpha_max_combined_streaming_fold_sequence_invalid")
    if (
        type(normalized_fold_segments) is not tuple
        or tuple(value.fold_id for value in normalized_fold_segments) != fold_ids
        or any(
            type(value) is not AlphaMaxNormalizedFoldSegmentEvidence
            for value in normalized_fold_segments
        )
    ):
        raise ValueError("alpha_max_combined_streaming_segment_sequence_invalid")

    expected_event_count = 0
    normalized_ending_equity = _ALPHA_MAX_INITIAL_CAPITAL
    event_bindings: list[dict[str, str]] = []
    for fold_evidence, fold, segment in zip(
        fold_runs,
        actual_runs,
        normalized_fold_segments,
        strict=True,
    ):
        stream = _validate_alpha_max_streaming_equity_evidence(fold.full_event_equity)
        if fold.ruin_detected:
            raise ValueError("alpha_max_combined_streaming_terminal_fold_forbidden")
        expected_scale = normalized_ending_equity / stream.initial_capital
        expected_normalized_end = expected_scale * stream.ending_equity
        expected_event_count += stream.event_count
        if (
            segment.source_streaming_equity_sha256 != stream.sha256
            or segment.source_event_stream_sha256 != stream.event_stream_sha256
            or segment.event_count != stream.event_count
            or segment.first_timestamp_ms != stream.first_timestamp_ms
            or segment.last_timestamp_ms != stream.last_timestamp_ms
            or segment.aggregate_prefix_event_count != expected_event_count
            or not math.isclose(
                segment.normalization_scale,
                expected_scale,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                segment.normalized_starting_equity,
                normalized_ending_equity,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                segment.normalized_ending_equity,
                expected_normalized_end,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("alpha_max_combined_streaming_segment_binding_mismatch")
        normalized_ending_equity = expected_normalized_end
        event_bindings.append(
            {
                "event_stream_sha256": stream.event_stream_sha256,
                "fold_id": fold.split_or_fold_id,
                "fold_run_sha256": fold_evidence.sha256,
                "normalized_segment_sha256": segment.sha256,
                "run_sha256": fold.sha256,
                "streaming_equity_sha256": stream.sha256,
            }
        )
    first_stream = actual_runs[0].full_event_equity
    last_stream = actual_runs[-1].full_event_equity
    if (
        live_stream.event_count != expected_event_count
        or live_stream.first_timestamp_ms != first_stream.first_timestamp_ms
        or live_stream.last_timestamp_ms != last_stream.last_timestamp_ms
        or normalized_fold_segments[-1].aggregate_prefix_event_stream_sha256
        != live_stream.event_stream_sha256
        or not math.isclose(
            live_stream.ending_equity,
            normalized_ending_equity,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("alpha_max_combined_streaming_live_binding_mismatch")
    values = {
        "domain": domain,
        "fold_ids": fold_ids,
        "fold_run_sha256s": tuple(value.sha256 for value in fold_runs),
        "fold_streaming_equity_sha256s": tuple(
            value.full_event_equity.sha256 for value in actual_runs
        ),
        "fold_event_stream_set_sha256": _sha256_bytes(
            _canonical_json_bytes(event_bindings, newline=True)
        ),
        "normalized_fold_segments": normalized_fold_segments,
        "streaming_equity": live_stream,
    }
    temporary = object.__new__(AlphaMaxCombinedStreamingEquityEvidence)
    for field, value in values.items():
        object.__setattr__(temporary, field, value)
    canonical = _canonical_json_bytes(
        _alpha_max_combined_streaming_payload(temporary),
        newline=True,
    )
    return AlphaMaxCombinedStreamingEquityEvidence(
        **values,
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxFoldRunEvidence:
    actual_engine_run: AlphaMaxActualEngineRunReceipt
    primary_return_stream: AlphaMaxPrimaryReturnStream | None
    status: str
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        if type(self.actual_engine_run) is not AlphaMaxActualEngineRunReceipt:
            raise TypeError("alpha_max_fold_run_receipt_identity_invalid")
        run = self.actual_engine_run
        terminal = run.ruin_detected
        expected_status = "ruin_detected" if terminal else "complete"
        if self.status != expected_status:
            raise ValueError("alpha_max_fold_run_status_mismatch")
        if terminal:
            if self.primary_return_stream is not None:
                raise ValueError("alpha_max_fold_run_terminal_stream_forbidden")
        else:
            if type(self.primary_return_stream) is not AlphaMaxPrimaryReturnStream:
                raise ValueError("alpha_max_fold_run_primary_stream_required")
            stream = _validate_alpha_max_primary_stream(self.primary_return_stream)
            if stream.endpoint_timestamps != _alpha_max_fold_reporting_calendar(
                run.split_or_fold_id
            ):
                raise ValueError("alpha_max_fold_run_reporting_calendar_mismatch")
            if not math.isclose(
                stream.endpoint_equities[-1],
                run.ending_equity,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("alpha_max_fold_run_equity_binding_mismatch")
        canonical = _canonical_json_bytes(_alpha_max_fold_run_payload(self), newline=True)
        if (
            type(self.canonical_bytes) is not bytes
            or self.canonical_bytes != canonical
            or self.sha256 != _sha256_bytes(canonical)
        ):
            raise ValueError("alpha_max_fold_run_canonical_mismatch")

    @property
    def split_or_fold_id(self) -> str:
        return self.actual_engine_run.split_or_fold_id

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _alpha_max_fold_run_payload(value: AlphaMaxFoldRunEvidence) -> dict[str, Any]:
    return {
        "actual_engine_run": value.actual_engine_run.to_payload(),
        "artifact_kind": "alpha_max_fold_run_evidence.v1",
        "primary_return_stream": (
            None
            if value.primary_return_stream is None
            else value.primary_return_stream.to_payload()
        ),
        "status": value.status,
    }


def build_alpha_max_fold_run_evidence(
    actual_engine_run: AlphaMaxActualEngineRunReceipt,
    primary_return_stream: AlphaMaxPrimaryReturnStream | None,
) -> AlphaMaxFoldRunEvidence:
    if type(actual_engine_run) is not AlphaMaxActualEngineRunReceipt:
        raise TypeError("alpha_max_actual_run_receipt_identity_invalid")
    status = "ruin_detected" if actual_engine_run.ruin_detected else "complete"
    temporary = object.__new__(AlphaMaxFoldRunEvidence)
    object.__setattr__(temporary, "actual_engine_run", actual_engine_run)
    object.__setattr__(temporary, "primary_return_stream", primary_return_stream)
    object.__setattr__(temporary, "status", status)
    canonical = _canonical_json_bytes(_alpha_max_fold_run_payload(temporary), newline=True)
    return AlphaMaxFoldRunEvidence(
        actual_engine_run=actual_engine_run,
        primary_return_stream=primary_return_stream,
        status=status,
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


def _build_alpha_max_combined_primary_return_stream(
    fold_runs: tuple[AlphaMaxFoldRunEvidence, ...],
) -> AlphaMaxPrimaryReturnStream:
    if not fold_runs or any(value.status != "complete" for value in fold_runs):
        raise ValueError("alpha_max_combined_primary_terminal_fold_forbidden")
    timestamps: list[datetime] = []
    equities: list[float] = []
    source_returns: list[float] = []
    current_equity = _ALPHA_MAX_INITIAL_CAPITAL
    for fold_run in fold_runs:
        stream = fold_run.primary_return_stream
        if type(stream) is not AlphaMaxPrimaryReturnStream:
            raise ValueError("alpha_max_fold_run_primary_stream_required")
        scale = current_equity / stream.initial_capital
        timestamps.extend(stream.endpoint_timestamps)
        equities.extend(scale * value for value in stream.endpoint_equities)
        source_returns.extend(stream.returns)
        current_equity = equities[-1]
    expected_calendar = tuple(
        timestamp
        for fold_id in _ALPHA_MAX_DOMAIN_FOLD_IDS[
            fold_runs[0].actual_engine_run.domain
        ]
        for timestamp in _alpha_max_fold_reporting_calendar(fold_id)
    )
    if tuple(timestamps) != expected_calendar:
        raise ValueError("alpha_max_combined_primary_calendar_mismatch")
    returns: list[float] = []
    prior = _ALPHA_MAX_INITIAL_CAPITAL
    for equity in equities:
        returns.append((equity / prior) - 1.0)
        prior = equity
    if any(
        not math.isclose(actual, source, rel_tol=0.0, abs_tol=1e-12)
        for actual, source in zip(returns, source_returns, strict=True)
    ):
        raise ValueError("alpha_max_combined_primary_return_binding_mismatch")
    stream = AlphaMaxPrimaryReturnStream(
        endpoint_timestamps=tuple(timestamps),
        endpoint_equities=tuple(equities),
        returns=tuple(returns),
        initial_capital=_ALPHA_MAX_INITIAL_CAPITAL,
        periods_per_year=ALPHA_MAX_PERIODS_PER_YEAR,
        calendar_sha256=_alpha_max_calendar_sha256(timestamps),
    )
    return _validate_alpha_max_primary_stream(stream)


@dataclass(frozen=True, slots=True)
class AlphaMaxCostCellPreGateEvidence:
    """Stage-one logical row/cost cell with every fresh fold run bound."""

    row_id: str
    domain: str
    nominal_cost_bps: int
    status: str
    fold_runs: tuple[AlphaMaxFoldRunEvidence, ...]
    fold_run_set_sha256: str
    source_return_stream_set_sha256: str | None
    combined_primary_return_stream: AlphaMaxPrimaryReturnStream | None
    combined_streaming_equity: AlphaMaxCombinedStreamingEquityEvidence | None
    metric_statistics: AlphaMaxMetricStatistics | None
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        _alpha_max_nonempty_token(self.row_id, field="pre_gate_row_id")
        expected_ids = _ALPHA_MAX_DOMAIN_FOLD_IDS.get(self.domain)
        if expected_ids is None or self.nominal_cost_bps not in _ALPHA_MAX_COST_CELLS:
            raise ValueError("alpha_max_pre_gate_identity_invalid")
        if (
            type(self.fold_runs) is not tuple
            or tuple(value.split_or_fold_id for value in self.fold_runs) != expected_ids
            or any(type(value) is not AlphaMaxFoldRunEvidence for value in self.fold_runs)
        ):
            raise ValueError("alpha_max_pre_gate_fold_sequence_invalid")
        actual_runs = tuple(value.actual_engine_run for value in self.fold_runs)
        if any(
            run.row_id != self.row_id
            or run.domain != self.domain
            or run.nominal_cost_bps != self.nominal_cost_bps
            for run in actual_runs
        ):
            raise ValueError("alpha_max_pre_gate_fold_identity_mismatch")
        baseline = actual_runs[0]
        if any(
            run.raw_root_receipts != baseline.raw_root_receipts
            or run.feature_root_receipts != baseline.feature_root_receipts
            or run.manifest_receipt.to_payload() != baseline.manifest_receipt.to_payload()
            or run.config_sha256 != baseline.config_sha256
            or run.runtime_contract_sha256 != baseline.runtime_contract_sha256
            or run.admitted_symbols != baseline.admitted_symbols
            or run.universe_sha256 != baseline.universe_sha256
            for run in actual_runs[1:]
        ):
            raise ValueError("alpha_max_pre_gate_fold_common_binding_mismatch")
        expected_set_hash = _sha256_bytes(
            _canonical_json_bytes(
                [
                    {
                        "fold_evidence_sha256": value.sha256,
                        "fold_id": value.split_or_fold_id,
                        "run_sha256": value.actual_engine_run.sha256,
                    }
                    for value in self.fold_runs
                ],
                newline=True,
            )
        )
        if self.fold_run_set_sha256 != expected_set_hash:
            raise ValueError("alpha_max_pre_gate_fold_run_set_hash_mismatch")
        terminal = any(value.status == "ruin_detected" for value in self.fold_runs)
        if self.status != ("ruin_detected" if terminal else "complete"):
            raise ValueError("alpha_max_pre_gate_status_mismatch")
        if terminal:
            if any(
                value is not None
                for value in (
                    self.source_return_stream_set_sha256,
                    self.combined_primary_return_stream,
                    self.combined_streaming_equity,
                    self.metric_statistics,
                )
            ):
                raise ValueError("alpha_max_pre_gate_terminal_metrics_forbidden")
        else:
            if type(self.source_return_stream_set_sha256) is not str:
                raise ValueError("alpha_max_pre_gate_source_stream_hash_required")
            _require_sha256(
                self.source_return_stream_set_sha256,
                field="alpha_max_pre_gate_source_stream_set_sha256",
            )
            expected_primary = _build_alpha_max_combined_primary_return_stream(self.fold_runs)
            if type(self.combined_streaming_equity) is not AlphaMaxCombinedStreamingEquityEvidence:
                raise ValueError("alpha_max_pre_gate_combined_evidence_mismatch")
            expected_streaming = _build_alpha_max_combined_streaming_equity(
                self.fold_runs,
                self.combined_streaming_equity.streaming_equity,
                self.combined_streaming_equity.normalized_fold_segments,
            )
            if (
                type(self.combined_primary_return_stream) is not AlphaMaxPrimaryReturnStream
                or self.combined_primary_return_stream.to_payload()
                != expected_primary.to_payload()
                or self.combined_streaming_equity.to_payload()
                != expected_streaming.to_payload()
                or type(self.metric_statistics) is not AlphaMaxMetricStatistics
            ):
                raise ValueError("alpha_max_pre_gate_combined_evidence_mismatch")
            expected_metrics = compute_alpha_max_metric_statistics(
                expected_primary,
                expected_streaming.streaming_equity,
            )
            if self.metric_statistics.to_payload() != expected_metrics.to_payload():
                raise ValueError("alpha_max_pre_gate_metric_binding_mismatch")
        canonical = _canonical_json_bytes(_alpha_max_pre_gate_payload(self), newline=True)
        if (
            type(self.canonical_bytes) is not bytes
            or self.canonical_bytes != canonical
            or self.sha256 != _sha256_bytes(canonical)
        ):
            raise ValueError("alpha_max_pre_gate_canonical_mismatch")

    @property
    def raw_root_receipts(self) -> tuple[AlphaMaxRootReceipt, ...]:
        return self.fold_runs[0].actual_engine_run.raw_root_receipts

    @property
    def feature_root_receipts(self) -> tuple[AlphaMaxRootReceipt, ...]:
        return self.fold_runs[0].actual_engine_run.feature_root_receipts

    @property
    def admitted_symbols(self) -> tuple[str, ...]:
        return self.fold_runs[0].actual_engine_run.admitted_symbols

    @property
    def ruin_detected(self) -> bool:
        return self.status == "ruin_detected"

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _alpha_max_pre_gate_payload(value: AlphaMaxCostCellPreGateEvidence) -> dict[str, Any]:
    return {
        "artifact_kind": "alpha_max_cost_cell_pre_gate_evidence.v1",
        "combined_primary_return_stream": (
            None
            if value.combined_primary_return_stream is None
            else value.combined_primary_return_stream.to_payload()
        ),
        "combined_streaming_equity": (
            None
            if value.combined_streaming_equity is None
            else value.combined_streaming_equity.to_payload()
        ),
        "domain": value.domain,
        "domain_engine_run_count": _ALPHA_MAX_DOMAIN_ENGINE_RUN_COUNT[value.domain],
        "fold_run_count": len(value.fold_runs),
        "fold_run_set_sha256": value.fold_run_set_sha256,
        "fold_runs": [fold.to_payload() for fold in value.fold_runs],
        "logical_actual_engine_cell_count": _ALPHA_MAX_LOGICAL_ACTUAL_ENGINE_CELL_COUNT,
        "metric_statistics": (
            None if value.metric_statistics is None else value.metric_statistics.to_payload()
        ),
        "nominal_cost_bps": value.nominal_cost_bps,
        "row_id": value.row_id,
        "source_return_stream_set_sha256": value.source_return_stream_set_sha256,
        "status": value.status,
    }


def build_alpha_max_cost_cell_pre_gate_evidence(
    fold_runs: tuple[AlphaMaxFoldRunEvidence, ...],
    combined_full_event_equity: AlphaMaxStreamingEquityEvidence | None = None,
    normalized_fold_segments: tuple[AlphaMaxNormalizedFoldSegmentEvidence, ...]
    | None = None,
) -> AlphaMaxCostCellPreGateEvidence:
    if type(fold_runs) is not tuple or not fold_runs or any(
        type(value) is not AlphaMaxFoldRunEvidence for value in fold_runs
    ):
        raise TypeError("alpha_max_pre_gate_fold_runs_must_be_exact_tuple")
    actual_runs = tuple(value.actual_engine_run for value in fold_runs)
    values: dict[str, Any] = {
        "row_id": actual_runs[0].row_id,
        "domain": actual_runs[0].domain,
        "nominal_cost_bps": actual_runs[0].nominal_cost_bps,
        "status": (
            "ruin_detected"
            if any(value.status == "ruin_detected" for value in fold_runs)
            else "complete"
        ),
        "fold_runs": fold_runs,
        "fold_run_set_sha256": _sha256_bytes(
            _canonical_json_bytes(
                [
                    {
                        "fold_evidence_sha256": value.sha256,
                        "fold_id": value.split_or_fold_id,
                        "run_sha256": value.actual_engine_run.sha256,
                    }
                    for value in fold_runs
                ],
                newline=True,
            )
        ),
    }
    if values["status"] == "ruin_detected":
        if combined_full_event_equity is not None or normalized_fold_segments is not None:
            raise ValueError("alpha_max_pre_gate_terminal_combined_stream_forbidden")
        values.update(
            source_return_stream_set_sha256=None,
            combined_primary_return_stream=None,
            combined_streaming_equity=None,
            metric_statistics=None,
        )
    else:
        if type(combined_full_event_equity) is not AlphaMaxStreamingEquityEvidence:
            raise TypeError("alpha_max_pre_gate_live_combined_stream_required")
        if type(normalized_fold_segments) is not tuple:
            raise TypeError("alpha_max_pre_gate_normalized_segments_required")
        combined_primary = _build_alpha_max_combined_primary_return_stream(fold_runs)
        combined_streaming = _build_alpha_max_combined_streaming_equity(
            fold_runs,
            combined_full_event_equity,
            normalized_fold_segments,
        )
        values.update(
            source_return_stream_set_sha256=_sha256_bytes(
                _canonical_json_bytes(
                    [
                        {
                            "fold_id": value.split_or_fold_id,
                            "primary_return_stream_sha256": _alpha_max_primary_stream_sha256(
                                value.primary_return_stream  # type: ignore[arg-type]
                            ),
                        }
                        for value in fold_runs
                    ],
                    newline=True,
                )
            ),
            combined_primary_return_stream=combined_primary,
            combined_streaming_equity=combined_streaming,
            metric_statistics=compute_alpha_max_metric_statistics(
                combined_primary,
                combined_streaming.streaming_equity,
            ),
        )
    temporary = object.__new__(AlphaMaxCostCellPreGateEvidence)
    for field, value in values.items():
        object.__setattr__(temporary, field, value)
    canonical = _canonical_json_bytes(_alpha_max_pre_gate_payload(temporary), newline=True)
    return AlphaMaxCostCellPreGateEvidence(
        **values,
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxTerminalGateEvidence:
    row_id: str
    comparison_role: str
    domain: str
    nominal_cost_bps: int
    pre_gate_evidence_sha256: str
    fold_run_set_sha256: str
    ruined_fold_ids: tuple[str, ...]
    streaming_ruin_fold_ids: tuple[str, ...]
    liquidation_fold_ids: tuple[str, ...]
    raw_root_set_sha256: str
    feature_root_set_sha256: str
    universe_sha256: str
    seed_schedule_sha256: str

    def __post_init__(self) -> None:
        expected_role = (
            "prelock_selection" if self.domain == "validation" else "historical_report"
        )
        if (
            self.comparison_role != expected_role
            or self.nominal_cost_bps != 30
            or not self.ruined_fold_ids
            or self.ruined_fold_ids
            != tuple(
                fold_id
                for fold_id in _ALPHA_MAX_DOMAIN_FOLD_IDS[self.domain]
                if fold_id in set(self.ruined_fold_ids)
            )
            or set(self.ruined_fold_ids)
            != set(self.streaming_ruin_fold_ids).union(self.liquidation_fold_ids)
        ):
            raise ValueError("alpha_max_terminal_gate_evidence_invalid")
        for field in (
            "pre_gate_evidence_sha256",
            "fold_run_set_sha256",
            "raw_root_set_sha256",
            "feature_root_set_sha256",
            "universe_sha256",
            "seed_schedule_sha256",
        ):
            _require_sha256(getattr(self, field), field=f"alpha_max_terminal_gate_{field}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_terminal_gate_evidence.v1",
            "comparison_role": self.comparison_role,
            "domain": self.domain,
            "feature_root_set_sha256": self.feature_root_set_sha256,
            "fold_run_set_sha256": self.fold_run_set_sha256,
            "liquidation_fold_ids": list(self.liquidation_fold_ids),
            "nominal_cost_bps": self.nominal_cost_bps,
            "pre_gate_evidence_sha256": self.pre_gate_evidence_sha256,
            "raw_root_set_sha256": self.raw_root_set_sha256,
            "row_id": self.row_id,
            "ruined_fold_ids": list(self.ruined_fold_ids),
            "seed_schedule_sha256": self.seed_schedule_sha256,
            "streaming_ruin_fold_ids": list(self.streaming_ruin_fold_ids),
            "universe_sha256": self.universe_sha256,
        }


@dataclass(frozen=True, slots=True)
class AlphaMaxCostCellEvidence:
    """Final logical row/cost evidence after optional cross-row statistics."""

    row_id: str
    domain: str
    nominal_cost_bps: int
    status: str
    evidence_tier: str
    selection_valid: bool
    pre_gate_evidence: AlphaMaxCostCellPreGateEvidence | None = None
    gate_input: AlphaMaxGateInput | None = None
    terminal_gate_evidence: AlphaMaxTerminalGateEvidence | None = None

    def __post_init__(self) -> None:
        _alpha_max_nonempty_token(self.row_id, field="cost_cell_row_id")
        _alpha_max_nonempty_token(self.status, field="cost_cell_status")
        if self.domain not in _ALPHA_MAX_DOMAIN_FOLD_IDS:
            raise ValueError("alpha_max_cost_cell_domain_invalid")
        if self.nominal_cost_bps not in _ALPHA_MAX_COST_CELLS:
            raise ValueError("alpha_max_cost_cell_nominal_bps_invalid")
        if self.evidence_tier not in {"actual_engine", "diagnostic", "identity"}:
            raise ValueError("alpha_max_cost_cell_evidence_tier_invalid")
        if type(self.selection_valid) is not bool:
            raise TypeError("alpha_max_cost_cell_selection_valid_invalid")
        if self.pre_gate_evidence is None:
            if self.selection_valid or any(
                value is not None for value in (self.gate_input, self.terminal_gate_evidence)
            ) or self.evidence_tier == "actual_engine":
                raise ValueError("alpha_max_cost_cell_selection_evidence_invalid")
            return
        if type(self.pre_gate_evidence) is not AlphaMaxCostCellPreGateEvidence:
            raise TypeError("alpha_max_pre_gate_evidence_identity_invalid")
        pre_gate = self.pre_gate_evidence
        if (
            pre_gate.row_id != self.row_id
            or pre_gate.domain != self.domain
            or pre_gate.nominal_cost_bps != self.nominal_cost_bps
            or self.evidence_tier != "actual_engine"
        ):
            raise ValueError("alpha_max_cost_cell_pre_gate_binding_mismatch")
        if pre_gate.status == "ruin_detected":
            if (
                self.status != "ruin_detected"
                or self.selection_valid
                or self.gate_input is not None
                or (
                    self.nominal_cost_bps == 30
                    and type(self.terminal_gate_evidence) is not AlphaMaxTerminalGateEvidence
                )
                or (self.nominal_cost_bps != 30 and self.terminal_gate_evidence is not None)
            ):
                raise ValueError("alpha_max_cost_cell_terminal_binding_mismatch")
            if self.terminal_gate_evidence is not None:
                terminal = self.terminal_gate_evidence
                baseline = pre_gate.fold_runs[0].actual_engine_run
                if (
                    terminal.row_id != self.row_id
                    or terminal.domain != self.domain
                    or terminal.pre_gate_evidence_sha256 != pre_gate.sha256
                    or terminal.fold_run_set_sha256 != pre_gate.fold_run_set_sha256
                    or terminal.raw_root_set_sha256 != baseline.raw_root_set_sha256
                    or terminal.feature_root_set_sha256 != baseline.feature_root_set_sha256
                    or terminal.universe_sha256 != baseline.universe_sha256
                    or terminal.seed_schedule_sha256
                    != alpha_max_seed_schedule_sha256(self.domain)
                ):
                    raise ValueError("alpha_max_cost_cell_terminal_binding_mismatch")
            return
        if self.status != "complete" or not self.selection_valid:
            raise ValueError("alpha_max_cost_cell_selection_evidence_invalid")
        if self.terminal_gate_evidence is not None:
            raise ValueError("alpha_max_cost_cell_terminal_binding_mismatch")
        if self.nominal_cost_bps != 30:
            if self.gate_input is not None:
                raise ValueError("alpha_max_cost_cell_gate_only_nominal_30_bps")
            return
        if type(self.gate_input) is not AlphaMaxGateInput:
            raise ValueError("alpha_max_incomplete_engine_evidence")
        metrics = pre_gate.metric_statistics
        stream = pre_gate.combined_primary_return_stream
        if type(metrics) is not AlphaMaxMetricStatistics or type(stream) is not AlphaMaxPrimaryReturnStream:
            raise ValueError("alpha_max_incomplete_engine_evidence")
        baseline = pre_gate.fold_runs[0].actual_engine_run
        expected_role = (
            "prelock_selection" if self.domain == "validation" else "historical_report"
        )
        required_true = (
            self.gate_input.comparison_valid,
            self.gate_input.native_data_coverage_complete,
            self.gate_input.funding_coverage_complete,
            self.gate_input.hash_valid,
            self.gate_input.manifest_valid,
            self.gate_input.reconciliation_complete,
        )
        if (
            self.gate_input.row_id != self.row_id
            or self.gate_input.comparison_role != expected_role
            or self.gate_input.evidence_tier != "actual_engine"
            or not all(required_true)
            or self.gate_input.calendar_sha256 != stream.calendar_sha256
            or self.gate_input.ruin
            or self.gate_input.raw_root_set_sha256 != baseline.raw_root_set_sha256
            or self.gate_input.feature_root_set_sha256 != baseline.feature_root_set_sha256
            or self.gate_input.universe_sha256 != baseline.universe_sha256
            or self.gate_input.seed_schedule_sha256
            != alpha_max_seed_schedule_sha256(self.domain)
            or not math.isclose(
                self.gate_input.gate_mdd,
                metrics.gate_mdd,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("alpha_max_cost_cell_gate_binding_mismatch")
        metric_bindings = (
            ("total_return", self.gate_input.cumulative_return),
            ("cagr", self.gate_input.cagr),
            ("calmar", self.gate_input.calmar),
            ("sharpe", self.gate_input.net_sharpe),
        )
        if any(
            not math.isclose(
                metrics.canonical_metrics[name],
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name, value in metric_bindings
        ):
            raise ValueError("alpha_max_cost_cell_gate_binding_mismatch")

    @property
    def raw_root_receipts(self) -> tuple[AlphaMaxRootReceipt, ...]:
        return () if self.pre_gate_evidence is None else self.pre_gate_evidence.raw_root_receipts

    @property
    def feature_root_receipts(self) -> tuple[AlphaMaxRootReceipt, ...]:
        return (
            ()
            if self.pre_gate_evidence is None
            else self.pre_gate_evidence.feature_root_receipts
        )

    @property
    def capsule_receipts(self) -> tuple[AlphaMaxCapsuleReceipt, ...]:
        return (
            ()
            if self.pre_gate_evidence is None
            else tuple(
                value.actual_engine_run.capsule_receipt
                for value in self.pre_gate_evidence.fold_runs
            )
        )

    @property
    def manifest_receipts(self) -> tuple[AlphaMaxManifestReceipt, ...]:
        return (
            ()
            if self.pre_gate_evidence is None
            else tuple(
                value.actual_engine_run.manifest_receipt
                for value in self.pre_gate_evidence.fold_runs
            )
        )

    @property
    def reconciliations(self) -> tuple[AlphaMaxReconciliationEvidence, ...]:
        return (
            ()
            if self.pre_gate_evidence is None
            else tuple(
                value.actual_engine_run.reconciliation
                for value in self.pre_gate_evidence.fold_runs
            )
        )

    @property
    def streaming_equity(self) -> AlphaMaxCombinedStreamingEquityEvidence | None:
        return (
            None
            if self.pre_gate_evidence is None
            else self.pre_gate_evidence.combined_streaming_equity
        )

    @property
    def runtime_contract_sha256(self) -> str | None:
        return (
            None
            if self.pre_gate_evidence is None
            else self.pre_gate_evidence.fold_runs[0].actual_engine_run.runtime_contract_sha256
        )

    @property
    def config_sha256(self) -> str | None:
        return (
            None
            if self.pre_gate_evidence is None
            else self.pre_gate_evidence.fold_runs[0].actual_engine_run.config_sha256
        )

    @property
    def ruin(self) -> bool | None:
        return None if self.pre_gate_evidence is None else self.pre_gate_evidence.ruin_detected

    @classmethod
    def unavailable(
        cls,
        *,
        row_id: str,
        domain: str,
        nominal_cost_bps: int,
        status: str,
        evidence_tier: str = "identity",
    ) -> AlphaMaxCostCellEvidence:
        return cls(
            row_id=row_id,
            domain=domain,
            nominal_cost_bps=nominal_cost_bps,
            status=status,
            evidence_tier=evidence_tier,
            selection_valid=False,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_cost_cell_evidence.v3",
            "domain": self.domain,
            "evidence_tier": self.evidence_tier,
            "gate_input": None if self.gate_input is None else self.gate_input.to_payload(),
            "nominal_cost_bps": self.nominal_cost_bps,
            "pre_gate_evidence": (
                None if self.pre_gate_evidence is None else self.pre_gate_evidence.to_payload()
            ),
            "row_id": self.row_id,
            "selection_valid": self.selection_valid,
            "status": self.status,
            "terminal_gate_evidence": (
                None
                if self.terminal_gate_evidence is None
                else self.terminal_gate_evidence.to_payload()
            ),
        }


def build_alpha_max_cost_cell_evidence(
    pre_gate_evidence: AlphaMaxCostCellPreGateEvidence,
    *,
    statistical_evidence: AlphaMaxStatisticalEvidence | None = None,
) -> AlphaMaxCostCellEvidence:
    """Finalize stage-one evidence; only normal 30bps cells consume cross-row stats."""
    if type(pre_gate_evidence) is not AlphaMaxCostCellPreGateEvidence:
        raise TypeError("alpha_max_pre_gate_evidence_identity_invalid")
    baseline = pre_gate_evidence.fold_runs[0].actual_engine_run
    if pre_gate_evidence.status == "ruin_detected":
        if statistical_evidence is not None:
            raise ValueError("alpha_max_terminal_cell_statistics_forbidden")
        terminal: AlphaMaxTerminalGateEvidence | None = None
        if pre_gate_evidence.nominal_cost_bps == 30:
            streaming_ids = tuple(
                fold.split_or_fold_id
                for fold in pre_gate_evidence.fold_runs
                if fold.actual_engine_run.full_event_equity.ruin_detected
            )
            liquidation_ids = tuple(
                fold.split_or_fold_id
                for fold in pre_gate_evidence.fold_runs
                if fold.actual_engine_run.liquidation_event_count > 0
            )
            ruined_ids = tuple(
                fold_id
                for fold_id in _ALPHA_MAX_DOMAIN_FOLD_IDS[pre_gate_evidence.domain]
                if fold_id in set(streaming_ids).union(liquidation_ids)
            )
            terminal = AlphaMaxTerminalGateEvidence(
                row_id=pre_gate_evidence.row_id,
                comparison_role=(
                    "prelock_selection"
                    if pre_gate_evidence.domain == "validation"
                    else "historical_report"
                ),
                domain=pre_gate_evidence.domain,
                nominal_cost_bps=30,
                pre_gate_evidence_sha256=pre_gate_evidence.sha256,
                fold_run_set_sha256=pre_gate_evidence.fold_run_set_sha256,
                ruined_fold_ids=ruined_ids,
                streaming_ruin_fold_ids=streaming_ids,
                liquidation_fold_ids=liquidation_ids,
                raw_root_set_sha256=baseline.raw_root_set_sha256,
                feature_root_set_sha256=baseline.feature_root_set_sha256,
                universe_sha256=baseline.universe_sha256,
                seed_schedule_sha256=alpha_max_seed_schedule_sha256(
                    pre_gate_evidence.domain
                ),
            )
        return AlphaMaxCostCellEvidence(
            row_id=pre_gate_evidence.row_id,
            domain=pre_gate_evidence.domain,
            nominal_cost_bps=pre_gate_evidence.nominal_cost_bps,
            status="ruin_detected",
            evidence_tier="actual_engine",
            selection_valid=False,
            pre_gate_evidence=pre_gate_evidence,
            terminal_gate_evidence=terminal,
        )
    gate_input: AlphaMaxGateInput | None = None
    if pre_gate_evidence.nominal_cost_bps == 30:
        if type(statistical_evidence) is not AlphaMaxStatisticalEvidence:
            raise TypeError("alpha_max_statistical_evidence_identity_invalid")
        metrics = pre_gate_evidence.metric_statistics
        stream = pre_gate_evidence.combined_primary_return_stream
        if type(metrics) is not AlphaMaxMetricStatistics or type(stream) is not AlphaMaxPrimaryReturnStream:
            raise ValueError("alpha_max_incomplete_engine_evidence")
        if (
            statistical_evidence.nominal_cost_bps != 30
            or statistical_evidence.calendar_sha256 != stream.calendar_sha256
            or pre_gate_evidence.row_id not in statistical_evidence.candidate_ids
            or pre_gate_evidence.row_id not in statistical_evidence.dsr_by_candidate
            or pre_gate_evidence.row_id not in statistical_evidence.spa_pvalue_by_candidate
        ):
            raise ValueError("alpha_max_statistical_evidence_binding_mismatch")
        gate_input = AlphaMaxGateInput(
            row_id=pre_gate_evidence.row_id,
            comparison_role=(
                "prelock_selection"
                if pre_gate_evidence.domain == "validation"
                else "historical_report"
            ),
            evidence_tier="actual_engine",
            comparison_valid=True,
            nominal_cost_bps=30,
            cumulative_return=metrics.canonical_metrics["total_return"],
            cagr=metrics.canonical_metrics["cagr"],
            calmar=metrics.canonical_metrics["calmar"],
            net_sharpe=metrics.canonical_metrics["sharpe"],
            full_event_mdd=metrics.full_event_mdd,
            reporting_4h_mdd=metrics.reporting_4h_mdd,
            dsr=statistical_evidence.dsr_by_candidate[pre_gate_evidence.row_id],
            spa_pvalue=statistical_evidence.spa_pvalue_by_candidate[
                pre_gate_evidence.row_id
            ],
            pbo=statistical_evidence.pbo,
            native_data_coverage_complete=True,
            funding_coverage_complete=True,
            hash_valid=True,
            manifest_valid=True,
            reconciliation_complete=True,
            ruin=False,
            raw_root_set_sha256=baseline.raw_root_set_sha256,
            feature_root_set_sha256=baseline.feature_root_set_sha256,
            universe_sha256=baseline.universe_sha256,
            calendar_sha256=stream.calendar_sha256,
            seed_schedule_sha256=alpha_max_seed_schedule_sha256(
                pre_gate_evidence.domain
            ),
        )
    elif statistical_evidence is not None:
        raise ValueError("alpha_max_statistics_only_nominal_30_bps")
    return AlphaMaxCostCellEvidence(
        row_id=pre_gate_evidence.row_id,
        domain=pre_gate_evidence.domain,
        nominal_cost_bps=pre_gate_evidence.nominal_cost_bps,
        status="complete",
        evidence_tier="actual_engine",
        selection_valid=True,
        pre_gate_evidence=pre_gate_evidence,
        gate_input=gate_input,
    )


def canonical_alpha_max_cost_cell_bytes(cell: AlphaMaxCostCellEvidence) -> bytes:
    if type(cell) is not AlphaMaxCostCellEvidence:
        raise TypeError("alpha_max_cost_cell_identity_invalid")
    return _canonical_json_bytes(cell.to_payload(), newline=True)


@dataclass(frozen=True, slots=True)
class AlphaMaxRowEvidence:
    """One frozen matrix row containing exactly four explicit cost-cell statuses."""

    row_id: str
    matrix_role: str
    status: str
    evidence_tier: str
    selection_valid: bool
    cost_cells: tuple[AlphaMaxCostCellEvidence, ...]

    def __post_init__(self) -> None:
        _alpha_max_nonempty_token(self.row_id, field="row_evidence_id")
        _alpha_max_nonempty_token(self.matrix_role, field="row_evidence_matrix_role")
        _alpha_max_nonempty_token(self.status, field="row_evidence_status")
        if self.evidence_tier not in {"actual_engine", "diagnostic", "identity"}:
            raise ValueError("alpha_max_row_evidence_tier_invalid")
        if type(self.selection_valid) is not bool:
            raise TypeError("alpha_max_row_selection_valid_invalid")
        if type(self.cost_cells) is not tuple or any(
            type(value) is not AlphaMaxCostCellEvidence for value in self.cost_cells
        ):
            raise TypeError("alpha_max_row_cost_cells_must_be_exact_tuple")
        ordered = tuple(sorted(self.cost_cells, key=lambda value: value.nominal_cost_bps))
        object.__setattr__(self, "cost_cells", ordered)
        if tuple(value.nominal_cost_bps for value in ordered) != (10, 15, 20, 30):
            raise ValueError("alpha_max_row_cost_cell_matrix_incomplete")
        if any(value.row_id != self.row_id for value in ordered):
            raise ValueError("alpha_max_row_cost_cell_identity_mismatch")
        if len({value.domain for value in ordered}) != 1:
            raise ValueError("alpha_max_row_cost_cell_domain_mismatch")
        actual_cells = tuple(
            value for value in ordered if value.pre_gate_evidence is not None
        )
        if actual_cells:
            if len(actual_cells) != 4 or self.evidence_tier != "actual_engine":
                raise ValueError("alpha_max_row_cost_cell_evidence_mismatch")
            baseline = actual_cells[0]
            expected_folds = _ALPHA_MAX_DOMAIN_FOLD_IDS[baseline.domain]
            if any(
                value.raw_root_receipts != baseline.raw_root_receipts
                or value.feature_root_receipts != baseline.feature_root_receipts
                or tuple(receipt.prefix_id for receipt in value.capsule_receipts)
                != expected_folds
                or len(value.manifest_receipts) != len(expected_folds)
                for value in actual_cells
            ):
                raise ValueError("alpha_max_row_cost_cell_evidence_mismatch")
            baseline_capsules = baseline.capsule_receipts
            baseline_manifests = baseline.manifest_receipts
            for value in actual_cells[1:]:
                if any(
                    left.to_payload() != right.to_payload()
                    for left, right in zip(
                        value.capsule_receipts,
                        baseline_capsules,
                        strict=True,
                    )
                ) or any(
                    left.to_payload() != right.to_payload()
                    for left, right in zip(
                        value.manifest_receipts,
                        baseline_manifests,
                        strict=True,
                    )
                ):
                    raise ValueError("alpha_max_row_cost_cell_evidence_mismatch")
        if self.selection_valid:
            if (
                self.status != "complete"
                or self.evidence_tier != "actual_engine"
                or not all(value.selection_valid for value in ordered)
                or type(ordered[-1].gate_input) is not AlphaMaxGateInput
            ):
                raise ValueError("alpha_max_row_selection_evidence_invalid")
        elif any(value.status == "ruin_detected" for value in ordered):
            if (
                self.status != "ruin_detected"
                or self.evidence_tier != "actual_engine"
                or ordered[-1].status != "ruin_detected"
                or type(ordered[-1].terminal_gate_evidence)
                is not AlphaMaxTerminalGateEvidence
            ):
                raise ValueError("alpha_max_row_terminal_evidence_invalid")
        elif any(value.selection_valid for value in ordered):
            raise ValueError("alpha_max_row_selection_evidence_invalid")

    @property
    def gate_input(self) -> AlphaMaxGateInput | None:
        return self.cost_cells[-1].gate_input

    @property
    def terminal_gate_evidence(self) -> AlphaMaxTerminalGateEvidence | None:
        return self.cost_cells[-1].terminal_gate_evidence

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "alpha_max_row_evidence.v1",
            "cost_cells": [value.to_payload() for value in self.cost_cells],
            "evidence_tier": self.evidence_tier,
            "matrix_role": self.matrix_role,
            "row_id": self.row_id,
            "selection_valid": self.selection_valid,
            "status": self.status,
        }


def canonical_alpha_max_row_bytes(row: AlphaMaxRowEvidence) -> bytes:
    if type(row) is not AlphaMaxRowEvidence:
        raise TypeError("alpha_max_row_evidence_identity_invalid")
    return _canonical_json_bytes(row.to_payload(), newline=True)


def _alpha_max_selection_inputs(
    rows: Sequence[
        AlphaMaxRowEvidence | AlphaMaxGateInput | AlphaMaxTerminalGateEvidence
    ],
) -> tuple[AlphaMaxGateInput | AlphaMaxTerminalGateEvidence, ...]:
    values = tuple(rows)
    row_ids: list[str] = []
    gate_inputs: list[AlphaMaxGateInput | AlphaMaxTerminalGateEvidence] = []
    for value in values:
        if type(value) in {AlphaMaxGateInput, AlphaMaxTerminalGateEvidence}:
            row_ids.append(value.row_id)
            gate_inputs.append(value)
        elif type(value) is AlphaMaxRowEvidence:
            row_ids.append(value.row_id)
            if value.selection_valid:
                gate = value.gate_input
                if type(gate) is not AlphaMaxGateInput:
                    raise ValueError("alpha_max_incomplete_engine_evidence")
                gate_inputs.append(gate)
            elif value.status == "ruin_detected":
                terminal = value.terminal_gate_evidence
                if type(terminal) is not AlphaMaxTerminalGateEvidence:
                    raise ValueError("alpha_max_terminal_evidence_missing")
                gate_inputs.append(terminal)
        else:
            raise TypeError("alpha_max_selection_row_identity_invalid")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("alpha_max_gate_duplicate_row_id")
    return tuple(gate_inputs)


def select_alpha_max_prelock_champion(
    rows: Sequence[
        AlphaMaxRowEvidence | AlphaMaxGateInput | AlphaMaxTerminalGateEvidence
    ],
) -> AlphaMaxSelectionResult:
    """Apply frozen validation gates and fix at most one selection identity."""
    return _alpha_max_select_gate_inputs(
        _alpha_max_selection_inputs(rows),
        role="prelock_selection",
    )


def rank_alpha_max_historical_report(
    rows: Sequence[
        AlphaMaxRowEvidence | AlphaMaxGateInput | AlphaMaxTerminalGateEvidence
    ],
) -> AlphaMaxSelectionResult:
    """Apply identical gates in the exposed report domain without selecting."""
    return _alpha_max_select_gate_inputs(
        _alpha_max_selection_inputs(rows),
        role="historical_report",
    )


def _alpha_max_stream_timestamp_ms(value: Any) -> int:
    if type(value) is int and value >= 0:
        return value
    if type(value) is datetime:
        normalized = _utc(value, field="streaming_equity_timestamp")
        return _epoch_ms(normalized)
    raise ValueError("alpha_max_streaming_equity_timestamp_invalid")


@dataclass(frozen=True, slots=True)
class AlphaMaxStreamingEquityEvidence:
    event_count: int
    initial_capital: float
    ending_equity: float
    peak_equity: float
    minimum_equity: float
    uncapped_full_event_drawdown: float
    full_event_mdd: float
    ruin_detected: bool
    max_drawdown_duration_events: int
    max_drawdown_duration_ms: int | None
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    event_stream_sha256: str
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        _validate_alpha_max_streaming_equity_evidence(self)

    @property
    def ruin(self) -> bool:
        """Compatibility alias; canonical evidence uses ``ruin_detected``."""
        return self.ruin_detected

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _validate_alpha_max_streaming_equity_evidence(
    value: AlphaMaxStreamingEquityEvidence,
) -> AlphaMaxStreamingEquityEvidence:
    if type(value) is not AlphaMaxStreamingEquityEvidence:
        raise TypeError("alpha_max_streaming_equity_evidence_identity_invalid")
    if type(value.event_count) is not int or value.event_count <= 0:
        raise ValueError("alpha_max_streaming_equity_event_count_invalid")
    initial = _alpha_max_finite_number(
        value.initial_capital,
        field="streaming_equity_initial_capital",
        positive=True,
    )
    if initial != _ALPHA_MAX_INITIAL_CAPITAL:
        raise ValueError("alpha_max_full_event_initial_capital_mismatch")
    ending = _alpha_max_finite_number(
        value.ending_equity,
        field="streaming_equity_ending_equity",
    )
    peak = _alpha_max_finite_number(
        value.peak_equity,
        field="streaming_equity_peak_equity",
        positive=True,
    )
    minimum = _alpha_max_finite_number(
        value.minimum_equity,
        field="streaming_equity_minimum_equity",
    )
    uncapped = _alpha_max_finite_number(
        value.uncapped_full_event_drawdown,
        field="streaming_equity_uncapped_drawdown",
        nonnegative=True,
    )
    gate_mdd = _alpha_max_finite_number(
        value.full_event_mdd,
        field="streaming_equity_gate_mdd",
        nonnegative=True,
    )
    if (
        peak < initial
        or minimum > initial
        or minimum > ending
        or gate_mdd != min(uncapped, 1.0)
        or gate_mdd > 1.0
    ):
        raise ValueError("alpha_max_streaming_equity_drawdown_invalid")
    if type(value.ruin_detected) is not bool or value.ruin_detected != (uncapped >= 1.0):
        raise ValueError("alpha_max_streaming_equity_ruin_invalid")
    if ending <= 0.0 and not value.ruin_detected:
        raise ValueError("alpha_max_streaming_equity_ruin_invalid")
    if (
        type(value.max_drawdown_duration_events) is not int
        or value.max_drawdown_duration_events < 0
        or value.max_drawdown_duration_events > value.event_count
    ):
        raise ValueError("alpha_max_streaming_equity_duration_invalid")
    timestamp_pair = (value.first_timestamp_ms, value.last_timestamp_ms)
    if (timestamp_pair[0] is None) is not (timestamp_pair[1] is None):
        raise ValueError("alpha_max_streaming_equity_timestamp_invalid")
    if timestamp_pair[0] is None:
        if value.max_drawdown_duration_ms is not None:
            raise ValueError("alpha_max_streaming_equity_timestamp_invalid")
    elif (
        type(timestamp_pair[0]) is not int
        or type(timestamp_pair[1]) is not int
        or timestamp_pair[0] < 0
        or timestamp_pair[1] < timestamp_pair[0]
        or type(value.max_drawdown_duration_ms) is not int
        or value.max_drawdown_duration_ms < 0
    ):
        raise ValueError("alpha_max_streaming_equity_timestamp_invalid")
    event_stream_sha256 = _require_sha256(
        value.event_stream_sha256,
        field="alpha_max_streaming_equity_event_stream_sha256",
    )
    payload = {
        "artifact_kind": "alpha_max_streaming_full_event_equity.v2",
        "ending_equity": ending,
        "event_count": value.event_count,
        "event_stream_sha256": event_stream_sha256,
        "first_timestamp_ms": value.first_timestamp_ms,
        "full_event_mdd": gate_mdd,
        "initial_capital": initial,
        "last_timestamp_ms": value.last_timestamp_ms,
        "max_drawdown_duration_events": value.max_drawdown_duration_events,
        "max_drawdown_duration_ms": value.max_drawdown_duration_ms,
        "minimum_equity": minimum,
        "peak_equity": peak,
        "ruin_detected": value.ruin_detected,
        "uncapped_full_event_drawdown": uncapped,
    }
    canonical = _canonical_json_bytes(payload, newline=True)
    if (
        type(value.canonical_bytes) is not bytes
        or value.canonical_bytes != canonical
        or value.sha256 != _sha256_bytes(canonical)
    ):
        raise ValueError("alpha_max_streaming_equity_canonical_mismatch")
    return value


class AlphaMaxStreamingEquityTracker:
    """Constant-memory exact full-event equity/MDD/drawdown-duration tracker."""

    __slots__ = (
        "_current_duration",
        "_digest",
        "_ending_equity",
        "_event_count",
        "_first_timestamp_ms",
        "_initial_capital",
        "_last_peak_timestamp_ms",
        "_last_timestamp_ms",
        "_max_drawdown_duration_events",
        "_max_drawdown_duration_ms",
        "_maximum_drawdown",
        "_minimum_equity",
        "_peak_equity",
        "_ruin_observed",
        "_timestamp_mode",
    )

    def __init__(self, *, initial_capital: float = _ALPHA_MAX_INITIAL_CAPITAL) -> None:
        capital = _alpha_max_finite_number(
            initial_capital,
            field="streaming_equity_initial_capital",
            positive=True,
        )
        if capital != _ALPHA_MAX_INITIAL_CAPITAL:
            raise ValueError("alpha_max_full_event_initial_capital_mismatch")
        self._initial_capital = capital
        self._peak_equity = capital
        self._ending_equity = capital
        self._maximum_drawdown = 0.0
        self._minimum_equity = capital
        self._ruin_observed = False
        self._current_duration = 0
        self._max_drawdown_duration_events = 0
        self._max_drawdown_duration_ms = 0
        self._event_count = 0
        self._first_timestamp_ms = None
        self._last_timestamp_ms = None
        self._last_peak_timestamp_ms = None
        self._timestamp_mode = None
        self._digest = hashlib.sha256()

    @property
    def retained_point_count(self) -> int:
        return 0

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def ending_equity(self) -> float:
        return self._ending_equity

    @property
    def event_stream_sha256(self) -> str:
        return self._digest.copy().hexdigest()

    @property
    def state_size_bytes(self) -> int:
        return object.__sizeof__(self) + self._digest.__sizeof__()

    def observe(self, point: tuple[float, float]) -> None:
        """Consume the Portfolio full-event sink's ``(unix_seconds, equity)`` tuple."""
        if type(point) is not tuple or len(point) != 2:
            raise TypeError("alpha_max_streaming_equity_point_invalid")
        unix_seconds = _alpha_max_finite_number(
            point[0],
            field="streaming_equity_unix_seconds",
            nonnegative=True,
        )
        milliseconds = unix_seconds * 1000.0
        if not math.isfinite(milliseconds):
            raise ValueError("alpha_max_streaming_equity_unix_seconds_invalid")
        timestamp_ms = int(milliseconds)
        self.update(point[1], timestamp_ms)

    def update(self, equity: float, timestamp: datetime | int | None = None) -> None:
        parsed = _alpha_max_finite_number(
            equity,
            field="streaming_full_event_equity",
        )
        mode = timestamp is not None
        if self._timestamp_mode is None:
            self._timestamp_mode = mode
        elif self._timestamp_mode is not mode:
            raise ValueError("alpha_max_streaming_timestamp_mode_changed")
        timestamp_ms = None if timestamp is None else _alpha_max_stream_timestamp_ms(timestamp)
        if (
            timestamp_ms is not None
            and self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            raise ValueError("alpha_max_streaming_timestamp_not_monotone")
        event_index = self._event_count
        self._digest.update(
            _canonical_json_bytes(
                {
                    "equity": parsed,
                    "event_index": event_index,
                    "timestamp_ms": timestamp_ms,
                },
                newline=True,
            )
        )
        self._event_count += 1
        self._ending_equity = parsed
        self._minimum_equity = min(self._minimum_equity, parsed)
        self._ruin_observed = self._ruin_observed or parsed <= 0.0
        if self._first_timestamp_ms is None:
            self._first_timestamp_ms = timestamp_ms
        self._last_timestamp_ms = timestamp_ms
        if parsed >= self._peak_equity:
            self._peak_equity = parsed
            self._current_duration = 0
            self._last_peak_timestamp_ms = timestamp_ms
        else:
            self._current_duration += 1
            self._max_drawdown_duration_events = max(
                self._max_drawdown_duration_events,
                self._current_duration,
            )
            if timestamp_ms is not None and self._last_peak_timestamp_ms is not None:
                self._max_drawdown_duration_ms = max(
                    self._max_drawdown_duration_ms,
                    timestamp_ms - self._last_peak_timestamp_ms,
                )
        self._maximum_drawdown = max(
            self._maximum_drawdown,
            1.0 - (parsed / self._peak_equity),
        )

    def finalize(self) -> AlphaMaxStreamingEquityEvidence:
        if self._event_count == 0:
            raise ValueError("alpha_max_full_event_equities_empty")
        payload = {
            "artifact_kind": "alpha_max_streaming_full_event_equity.v2",
            "ending_equity": self._ending_equity,
            "event_count": self._event_count,
            "event_stream_sha256": self._digest.copy().hexdigest(),
            "first_timestamp_ms": self._first_timestamp_ms,
            "full_event_mdd": min(self._maximum_drawdown, 1.0),
            "initial_capital": self._initial_capital,
            "last_timestamp_ms": self._last_timestamp_ms,
            "max_drawdown_duration_events": self._max_drawdown_duration_events,
            "max_drawdown_duration_ms": (
                self._max_drawdown_duration_ms if self._timestamp_mode else None
            ),
            "minimum_equity": self._minimum_equity,
            "peak_equity": self._peak_equity,
            "ruin_detected": self._ruin_observed,
            "uncapped_full_event_drawdown": self._maximum_drawdown,
        }
        canonical = _canonical_json_bytes(payload, newline=True)
        return AlphaMaxStreamingEquityEvidence(
            event_count=self._event_count,
            initial_capital=self._initial_capital,
            ending_equity=self._ending_equity,
            peak_equity=self._peak_equity,
            minimum_equity=self._minimum_equity,
            uncapped_full_event_drawdown=self._maximum_drawdown,
            full_event_mdd=min(self._maximum_drawdown, 1.0),
            ruin_detected=self._ruin_observed,
            max_drawdown_duration_events=self._max_drawdown_duration_events,
            max_drawdown_duration_ms=(
                self._max_drawdown_duration_ms if self._timestamp_mode else None
            ),
            first_timestamp_ms=self._first_timestamp_ms,
            last_timestamp_ms=self._last_timestamp_ms,
            event_stream_sha256=payload["event_stream_sha256"],
            canonical_bytes=canonical,
            sha256=_sha256_bytes(canonical),
        )


@dataclass(frozen=True, slots=True)
class AlphaMaxPrelockArtifact:
    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _alpha_max_safe_relative_path(
                self.relative_path,
                field="prelock_artifact_path",
            ),
        )
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("alpha_max_prelock_artifact_byte_count_invalid")
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, field="alpha_max_prelock_artifact_sha256"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AlphaMaxPrelockSeal:
    artifacts: tuple[AlphaMaxPrelockArtifact, ...]
    inventory_sha256: str
    prelock_champion: str | None
    selected_candidate_id: str | None
    canonical_bytes: bytes
    sha256: str

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def build_alpha_max_prelock_seal(
    stable_artifacts: Mapping[str, bytes],
    *,
    prelock_champion: str | None,
    selected_candidate_id: str | None,
) -> AlphaMaxPrelockSeal:
    """Seal immutable pre-historical artifact bytes without opening later roots."""
    if not isinstance(stable_artifacts, Mapping) or not stable_artifacts:
        raise ValueError("alpha_max_prelock_inventory_empty")
    if prelock_champion is not None:
        _alpha_max_nonempty_token(prelock_champion, field="prelock_champion")
    if selected_candidate_id is not None:
        _alpha_max_nonempty_token(selected_candidate_id, field="selected_candidate_id")
    if prelock_champion != selected_candidate_id:
        raise ValueError("alpha_max_prelock_selection_identity_mismatch")
    artifacts: list[AlphaMaxPrelockArtifact] = []
    for raw_path, raw_bytes in stable_artifacts.items():
        relative_path = _alpha_max_safe_relative_path(
            raw_path,
            field="prelock_artifact_path",
        )
        if any(
            marker in part and "boundary" not in part
            for part in PurePosixPath(relative_path).parts
            for marker in ("historical_evaluation", "historical_exposed_evaluation")
        ):
            raise ValueError("alpha_max_prelock_historical_input_forbidden")
        if type(raw_bytes) is not bytes:
            raise TypeError("alpha_max_prelock_artifact_bytes_required")
        artifacts.append(
            AlphaMaxPrelockArtifact(
                relative_path=relative_path,
                byte_count=len(raw_bytes),
                sha256=_sha256_bytes(raw_bytes),
            )
        )
    ordered = tuple(sorted(artifacts, key=lambda value: value.relative_path))
    if len({value.relative_path for value in ordered}) != len(ordered):
        raise ValueError("alpha_max_prelock_artifact_path_duplicate")
    inventory_payload = [value.to_payload() for value in ordered]
    inventory_sha256 = _sha256_bytes(_canonical_json_bytes(inventory_payload, newline=True))
    payload = {
        "artifact_count": len(ordered),
        "artifact_kind": "alpha_max_immutable_prelock_seal.v1",
        "artifacts": inventory_payload,
        "historical_evaluation_inputs_included": False,
        "immutable": True,
        "inventory_sha256": inventory_sha256,
        "prelock_champion": prelock_champion,
        "selected_candidate_id": selected_candidate_id,
    }
    canonical = _canonical_json_bytes(payload, newline=True)
    return AlphaMaxPrelockSeal(
        artifacts=ordered,
        inventory_sha256=inventory_sha256,
        prelock_champion=prelock_champion,
        selected_candidate_id=selected_candidate_id,
        canonical_bytes=canonical,
        sha256=_sha256_bytes(canonical),
    )


def alpha_max_terminal_outcome(
    prelock_champion: str | None,
    *,
    champion_historical_complete: bool | None,
    champion_historical_passed: bool | None,
) -> str:
    """Return the singular first-match Revision 5.14 terminal outcome."""
    if prelock_champion is None:
        return "no_demonstrated_alpha"
    _alpha_max_nonempty_token(prelock_champion, field="terminal_prelock_champion")
    if champion_historical_complete is not True:
        return "historical_evaluation_incomplete"
    if type(champion_historical_passed) is not bool:
        return "historical_evaluation_incomplete"
    if not champion_historical_passed:
        return "prelock_champion_historical_robustness_failed"
    return "prelock_champion_historical_robustness_passed"


@dataclass(frozen=True, slots=True)
class AlphaMaxTerminalState:
    terminal_outcome: str
    prelock_champion: str | None
    selected_candidate_id: str | None
    historical_evaluation_leader: str | None
    leader_differs_from_prelock_champion: bool
    incumbent_comparison_status: str
    historical_exposure_status: str
    requires_fresh_confirmation: bool
    confirmation_status: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "confirmation_status": self.confirmation_status,
            "historical_evaluation_leader": self.historical_evaluation_leader,
            "historical_exposure_status": self.historical_exposure_status,
            "incumbent_comparison_status": self.incumbent_comparison_status,
            "leader_differs_from_prelock_champion": (self.leader_differs_from_prelock_champion),
            "prelock_champion": self.prelock_champion,
            "requires_fresh_confirmation": self.requires_fresh_confirmation,
            "selected_candidate_id": self.selected_candidate_id,
            "terminal_outcome": self.terminal_outcome,
        }


def _alpha_max_validate_selection_result(
    value: AlphaMaxSelectionResult,
    *,
    role: str,
) -> AlphaMaxSelectionResult:
    if type(value) is not AlphaMaxSelectionResult or value.role != role:
        raise TypeError("alpha_max_selection_result_identity_invalid")
    canonical = _canonical_json_bytes(value.to_payload(), newline=True)
    if (
        type(value.canonical_bytes) is not bytes
        or value.canonical_bytes != canonical
        or value.sha256 != _sha256_bytes(canonical)
        or len({decision.row_id for decision in value.decisions}) != len(value.decisions)
    ):
        raise ValueError("alpha_max_selection_result_canonical_invalid")
    return value


def build_alpha_max_terminal_state(
    *,
    prelock_selection: AlphaMaxSelectionResult,
    champion_historical_nominal_30_cell: AlphaMaxCostCellEvidence | None,
    historical_ranking: AlphaMaxSelectionResult | None,
    incumbent_comparison_status: str,
) -> AlphaMaxTerminalState:
    """Derive the terminal outcome only from typed selection and engine evidence."""
    prelock = _alpha_max_validate_selection_result(
        prelock_selection,
        role="prelock_selection",
    )
    prelock_champion = prelock.prelock_champion
    if (
        prelock.selected_candidate_id != prelock_champion
        or (prelock_champion is None) != (prelock.ranked_candidate_ids == ())
        or (
            prelock_champion is not None
            and (
                not prelock.ranked_candidate_ids
                or prelock.ranked_candidate_ids[0] != prelock_champion
            )
        )
    ):
        raise ValueError("alpha_max_terminal_prelock_selection_invalid")
    if incumbent_comparison_status not in {
        "matched_outperformed",
        "matched_not_outperformed",
        "unavailable",
        "not_applicable",
    }:
        raise ValueError("alpha_max_incumbent_comparison_status_invalid")
    historical_evaluation_leader: str | None = None
    historical_result: AlphaMaxSelectionResult | None = None
    if historical_ranking is not None:
        historical_result = _alpha_max_validate_selection_result(
            historical_ranking,
            role="historical_report",
        )
        historical_evaluation_leader = historical_result.historical_evaluation_leader

    historical_complete: bool | None = None
    historical_passed: bool | None = None
    if prelock_champion is None:
        if champion_historical_nominal_30_cell is not None:
            raise ValueError("alpha_max_terminal_historical_without_champion")
    elif champion_historical_nominal_30_cell is not None:
        cell = champion_historical_nominal_30_cell
        if (
            type(cell) is not AlphaMaxCostCellEvidence
            or cell.row_id != prelock_champion
            or cell.domain != "historical_exposed_evaluation"
            or cell.nominal_cost_bps != 30
        ):
            raise ValueError("alpha_max_terminal_historical_cell_binding_mismatch")
        if cell.pre_gate_evidence is not None and historical_result is not None:
            decisions = tuple(
                decision
                for decision in historical_result.decisions
                if decision.row_id == prelock_champion
            )
            if len(decisions) != 1:
                raise ValueError("alpha_max_terminal_historical_decision_missing")
            decision = decisions[0]
            if cell.status == "complete":
                gate = cell.gate_input
                if (
                    type(gate) is not AlphaMaxGateInput
                    or gate.comparison_role != "historical_report"
                    or decision.gate_mdd is None
                    or not math.isclose(
                        decision.gate_mdd,
                        gate.gate_mdd,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError("alpha_max_terminal_historical_gate_mismatch")
                historical_complete = True
                historical_passed = decision.eligible
            elif cell.status == "ruin_detected":
                if (
                    type(cell.terminal_gate_evidence)
                    is not AlphaMaxTerminalGateEvidence
                    or decision.eligible
                    or decision.rejection_reasons != ("ruin_detected",)
                    or decision.gate_mdd is not None
                ):
                    raise ValueError("alpha_max_terminal_ruin_decision_mismatch")
                historical_complete = True
                historical_passed = False
    outcome = alpha_max_terminal_outcome(
        prelock_champion,
        champion_historical_complete=historical_complete,
        champion_historical_passed=historical_passed,
    )
    return AlphaMaxTerminalState(
        terminal_outcome=outcome,
        prelock_champion=prelock_champion,
        selected_candidate_id=prelock_champion,
        historical_evaluation_leader=historical_evaluation_leader,
        leader_differs_from_prelock_champion=(
            historical_evaluation_leader is not None
            and historical_evaluation_leader != prelock_champion
        ),
        incumbent_comparison_status=incumbent_comparison_status,
        historical_exposure_status=(
            "committed_period_outcomes_observed"
            if historical_complete is True
            or (prelock_champion is None and historical_result is not None)
            else "not_applicable"
            if prelock_champion is None
            else "historical_evaluation_incomplete"
        ),
        requires_fresh_confirmation=True,
        confirmation_status="not_run",
    )
