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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from lumina_quant.backtesting.data_windowed_parquet import (
    HistoricParquetWindowedDataHandler,
    RawPoint,
)
from lumina_quant.backtesting.execution_model import ExecutionModel
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
from lumina_quant.utils.artifact_read_receipt import read_artifact_bytes
from lumina_quant.strategy_factory import research_metrics

__all__ = [
    "ALPHA_MAX_CANDIDATE_SYMBOLS",
    "ALPHA_MAX_DSR_NUM_TRIALS",
    "ALPHA_MAX_MANIFEST_CHILD_KEYS",
    "ALPHA_MAX_MANIFEST_TOP_LEVEL_KEYS",
    "ALPHA_MAX_PERIODS_PER_YEAR",
    "AlphaMaxAdmissionArtifact",
    "AlphaMaxCapacityDiagnostics",
    "AlphaMaxEquityEndpoint",
    "AlphaMaxFundingBoundaryLedgerRow",
    "AlphaMaxFundingBoundaryRequest",
    "AlphaMaxFundingBoundaryResolver",
    "AlphaMaxManifestMaterialization",
    "AlphaMaxMetricStatistics",
    "AlphaMaxOrderedFundingLookup",
    "AlphaMaxPreGateSharpeEvidence",
    "AlphaMaxPrimaryReturnStream",
    "AlphaMaxStatisticalEvidence",
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
    "alpha_max_shrunk_hrp_weights",
    "alpha_max_trial_key",
    "alpha_max_trial_key_set_lf_bytes",
    "alpha_max_type7_quantile",
    "build_alpha_max_primary_return_stream",
    "build_alpha_max_statistical_evidence",
    "build_alpha_max_trial_ledger",
    "compute_alpha_max_capacity_diagnostics",
    "compute_alpha_max_metric_statistics",
    "compute_alpha_max_turnover_rpt",
    "materialize_alpha_max_manifest",
    "normalize_alpha_max_prior_trial_node",
    "read_alpha_max_prior_trial_blob",
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
    full_event_mdd: float
    reporting_4h_mdd: float
    gate_mdd: float
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
            "full_event_mdd": self.full_event_mdd,
            "gate_mdd": self.gate_mdd,
            "reporting_4h_mdd": self.reporting_4h_mdd,
            "value_at_risk_5pct_type7": self.value_at_risk_5pct_type7,
        }


def compute_alpha_max_metric_statistics(
    stream: AlphaMaxPrimaryReturnStream,
    full_event_equities: Sequence[float],
) -> AlphaMaxMetricStatistics:
    """Call the sole canonical metric primitive and add non-overlapping diagnostics."""
    validated = _validate_alpha_max_primary_stream(stream)
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
    full_mdd = alpha_max_full_event_mdd(full_event_equities)
    duration_count, duration_hours = alpha_max_drawdown_duration(validated)
    var_5pct = alpha_max_type7_quantile(validated.returns, 0.05)
    tail_count = max(1, math.ceil(0.05 * len(validated.returns)))
    worst = sorted(enumerate(validated.returns), key=lambda item: (item[1], item[0]))[:tail_count]
    expected_shortfall = math.fsum(value for _, value in worst) / tail_count
    return AlphaMaxMetricStatistics(
        canonical_metrics=MappingProxyType(canonical),
        full_event_mdd=full_mdd,
        reporting_4h_mdd=reporting_mdd,
        gate_mdd=max(full_mdd, reporting_mdd),
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
