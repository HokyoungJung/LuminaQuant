"""Fail-closed engine orchestration for the alpha-max experiment.

The descriptor-bound frozen-config foundation in this module is also the sole
runtime boundary for actual alpha-max engines.  Manifest activation, identity
checks, indicator-only warmup, raw-first cell replay, matrix status accounting,
and the two physical CLI bundles all stay here so that the pure evidence module
never needs to construct or mutate a backtest.

The experiment config is a sealed Revision 5.15 artifact.  No profile, ambient
``LQ_*`` value, default runtime config, YAML file, or merge layer participates in
construction.
"""

from __future__ import annotations

import base64
import copy
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import math
import multiprocessing
import os
import queue
import re
import stat
import sys
import tempfile
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol

import numpy as np
from lumina_quant.alpha_max_process_boundary import (
    AlphaMaxRuntimeContractError,
    AmbientLQEnvironmentError,
    reject_ambient_lq_environment,
)
from lumina_quant.backtesting._config_view import register_alpha_max_backtest_config_type
from lumina_quant.backtesting.backtest import Backtest, FastQueue
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_model import (
    ExecutionPricingTrace,
    execution_pricing_trace_sha256,
)
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import FillApplicationAttribution, Portfolio
from lumina_quant.core.events import SignalEvent
from lumina_quant.core.market_window_contract import build_market_window_event
from lumina_quant.core.strategy_input import StrategyInputContext
from lumina_quant.market_data import timeframe_to_milliseconds
from lumina_quant.portfolio.quality_gated_allocation import _round
from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_MANIFEST_CHILD_KEYS,
    ALPHA_MAX_MANIFEST_TOP_LEVEL_KEYS,
    AlphaMaxAdmissionComputation,
    AlphaMaxAdmissionDailyCandidateInput,
    AlphaMaxActualEngineRunReceipt,
    AlphaMaxCapsuleReceipt,
    AlphaMaxContractManifestSeal,
    AlphaMaxCostCellEvidence,
    AlphaMaxCostCellPreGateEvidence,
    AlphaMaxDailyQuoteNotional,
    AlphaMaxEquityEndpoint,
    AlphaMaxFoldRunEvidence,
    AlphaMaxFundingBoundaryLedgerRow,
    AlphaMaxFundingBoundaryResolver,
    AlphaMaxGateDecision,
    AlphaMaxManifestReceipt,
    AlphaMaxNativeFinalizationReceipt,
    AlphaMaxNormalizedFoldSegmentEvidence,
    AlphaMaxOrderedFundingLookup,
    AlphaMaxPrimaryReturnStream,
    AlphaMaxRowEvidence,
    AlphaMaxRootSeal,
    AlphaMaxRootReceipt,
    AlphaMaxSelectionResult,
    AlphaMaxScalingAttribution,
    AlphaMaxStatisticalEvidence,
    AlphaMaxTrialLedger,
    AlphaMaxStreamingEquityTracker,
    AlphaMaxTrainLiquidityBuckets,
    AlphaMaxTreeEntry,
    FeatureRootSpec,
    build_alpha_max_actual_engine_run_receipt,
    build_alpha_max_cost_cell_evidence,
    build_alpha_max_cost_cell_pre_gate_evidence,
    build_alpha_max_daily_quote_notional,
    build_alpha_max_fold_run_evidence,
    build_alpha_max_normalized_fold_segment_evidence,
    build_alpha_max_native_finalization_receipt,
    build_alpha_max_primary_return_stream,
    build_alpha_max_statistical_evidence,
    build_alpha_max_terminal_state,
    build_alpha_max_train_liquidity_buckets,
    build_alpha_max_trend_liquidity_falsifier,
    build_alpha_max_trial_ledger,
    canonical_alpha_max_cost_cell_bytes,
    canonical_alpha_max_row_bytes,
    compute_alpha_max_train_admission_from_daily_summaries,
    parse_alpha_max_cost_cell_pre_gate_evidence,
    rank_alpha_max_historical_report,
    read_alpha_max_prior_trial_blob_input,
    seal_alpha_max_contract_manifest,
    seal_alpha_max_root_tree,
    select_alpha_max_prelock_champion,
    validate_alpha_max_train_liquidity_buckets,
)
from lumina_quant.strategies.artifact_portfolio_mode import (
    ArtifactPortfolioModeStrategy,
    PortfolioModeComponent,
)
from lumina_quant.timeframe_aggregator import NativeBarRelease, TimeframeAggregator

from lumina_quant.utils.artifact_read_receipt import (
    ArtifactReadReceipt,
    read_artifact_bytes,
)

__all__ = [
    "ALPHA_MAX_CANDIDATE_SYMBOLS",
    "ALPHA_MAX_CONFIG_FILE_SHA256",
    "ALPHA_MAX_CONFIG_PAYLOAD_SHA256",
    "ALPHA_MAX_COST_CELL_BPS",
    "ALPHA_MAX_INCUMBENT_RESOLUTION_AUDIT_SHA256",
    "ALPHA_MAX_RUNTIME_CONTRACT_SHA256",
    "AlphaMaxAllocatorFitEvidence",
    "AlphaMaxAncestorIdentity",
    "AlphaMaxArtifactSeal",
    "AlphaMaxAttributionCollector",
    "AlphaMaxBacktestConfig",
    "AlphaMaxCommandResult",
    "AlphaMaxCostCell",
    "AlphaMaxEngineActivation",
    "AlphaMaxEngineConstructorPlan",
    "AlphaMaxExpectedDefinition",
    "AlphaMaxIndicatorCapsule",
    "AlphaMaxIndicatorPhaseInput",
    "AlphaMaxMatrixCellStatus",
    "AlphaMaxMatrixResult",
    "AlphaMaxPhaseWindow",
    "AlphaMaxRuntimeContractError",
    "AlphaMaxRuntimePreflight",
    "AlphaMaxSealedBundle",
    "AmbientLQEnvironmentError",
    "FrozenRuntimeMutationError",
    "UnfrozenRuntimeFieldError",
    "alpha_max_common_rng_seed",
    "alpha_max_common_rng_seed_payload",
    "build_alpha_max_backtest_config",
    "build_alpha_max_cost_cell_configs",
    "build_alpha_max_engine_constructor_plan",
    "build_alpha_max_final_refit_indicator_capsule",
    "build_alpha_max_indicator_capsule",
    "construct_alpha_max_engine",
    "create_alpha_max_historical_package",
    "create_alpha_max_indicator_day_checkpoint_store",
    "create_alpha_max_prelock_bundle",
    "fit_alpha_max_nominal_20_allocators",
    "orchestrate_alpha_max_status_matrix",
    "preflight_alpha_max_runtime_contract",
    "reject_ambient_lq_environment",
    "replay_alpha_max_cost_cell",
    "run_alpha_max_historical_process",
    "run_alpha_max_prelock_process",
    "seal_alpha_max_manifest_activation",
    "validate_alpha_max_cost_cell_config_matrix",
    "validate_alpha_max_engine_activation",
]


ALPHA_MAX_RUNTIME_CONTRACT_SHA256: Final[str] = (
    "b3859443c842cf8b04d04ed32923e6c6a8207af18e26f68a717ba623b4edfef9"
)
ALPHA_MAX_CONFIG_PAYLOAD_SHA256: Final[str] = (
    "b062e3805d94087cc18cd22634918815503f94dd73f8fa8ac1979e7aef535f85"
)
ALPHA_MAX_CONFIG_CANONICAL_SHA256: Final[str] = (
    "691bf756519be1984ebb331142dce1b8787783d8da1d3e9911fbdd8d7d0d4ac3"
)
ALPHA_MAX_CONFIG_FILE_SHA256: Final[str] = (
    "2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c"
)
ALPHA_MAX_INCUMBENT_RESOLUTION_AUDIT_SHA256: Final[str] = (
    "5133bc40116399fe7af32e75a1ecc52a4f385dc8a0b5d3a4a9585e2437615ed8"
)
_ALPHA_MAX_INCUMBENT_RESOLUTION_CANONICAL_SHA256: Final[str] = (
    "c8419138ba62bf504e87202a17dc977dc97f90a39930d241b446b7756f88417d"
)
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
ALPHA_MAX_COST_CELL_BPS: Final[tuple[int, ...]] = (10, 15, 20, 30)
_ALPHA_MAX_MAX_PARALLEL_WORKERS: Final[int] = 4
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
_ALPHA_MAX_DOMAIN_FOLD_IDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "validation": _ALPHA_MAX_VALIDATION_FOLD_IDS,
        "historical_exposed_evaluation": _ALPHA_MAX_HISTORICAL_FOLD_IDS,
    }
)

_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "admission",
        "allocation_and_refit",
        "candidate_symbols",
        "chronology",
        "contract_manifest_contract",
        "cost_cells",
        "current_trial_registry",
        "experiment_id",
        "funding_sidecar_and_settlement",
        "incumbent_resolution",
        "integrity",
        "metrics_and_statistics",
        "native_timeframes_and_warmup",
        "normative_sources",
        "process_boundaries",
        "revision",
        "runtime_contract",
        "safety_and_claims",
        "schema_version",
        "selection_and_reporting",
        "trial_ledger",
    }
)
_RUNTIME_KEYS: Final[frozenset[str]] = frozenset(
    {
        "attribute_allowlist",
        "backtest_constructor",
        "class_name",
        "construction",
        "derived_attributes",
        "override_policy",
        "portfolio_strategy_constructor",
        "schema_version",
        "seed_schedule",
        "static_attributes",
    }
)
_EXPECTED_CANONICALIZATION: Final[str] = (
    'json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,'
    'allow_nan=False).encode("utf-8")'
)
_EXPECTED_CONSTRUCTION: Final[dict[str, object]] = {
    "allow_private_rt_attribute": False,
    "allow_runtime_config_fallback": False,
    "allow_unknown_attribute_read": False,
    "final_after_construction": True,
    "runtime_field_missing_policy": "reject_unfrozen_runtime_field",
    "runtime_field_unknown_policy": "reject_unfrozen_runtime_field",
}
_EXPECTED_OVERRIDE_POLICY: Final[dict[str, object]] = {
    "any_environment_key_prefix_rejected": "LQ_",
    "config_yaml_loaded": False,
    "default_runtime_config_loaded": False,
    "environment_values_loaded": False,
    "merge_layer_count": 0,
    "profile_loaded": False,
    "runtime_override_loaded": False,
    "unknown_cli_arguments_rejected": True,
}
_EXPECTED_DERIVED_ATTRIBUTES: Final[dict[str, object]] = {
    "END_DATE": {
        "derivation": "exact_current_phase_end_utc",
        "source": "chronology.splits[split_id].end_utc",
    },
    "RANDOM_SEED": {
        "derivation": "common_random_number_seed_schedule",
        "row_id_in_payload": False,
        "zero_result_replacement": 1,
    },
    "SLIPPAGE_RATE": {
        "allowed_values": [0.0005, 0.001, 0.0015, 0.0025],
        "derivation": "exact_nominal_cost_cell_slippage_rate",
        "source": "cost_cells[].slippage_rate",
    },
    "START_DATE": {
        "derivation": "exact_current_phase_start_utc",
        "source": "chronology.splits[split_id].start_utc",
    },
    "SYMBOLS": {
        "derivation": "exact_sealed_train_admission_tuple_by_identity",
        "maximum_count": 10,
        "minimum_count": 5,
        "source": "sealed_admission_manifest.admitted_symbols",
        "subset_of": "candidate_symbols",
    },
}
_EXPECTED_SEED_SCHEDULE: Final[dict[str, object]] = {
    "algorithm": "sha256_first_8_bytes_big_endian_mod_2147483647",
    "cost_encoding": 'str(nominal_cost_bps).encode("ascii")',
    "payload_prefix_hex": "616c7068615f6d61785f323032363037313000",
    "payload_text": (
        'b"alpha_max_20260710\\0" + split_or_fold_id.encode("utf-8") + b"\\0" + '
        'str(nominal_cost_bps).encode("ascii")'
    ),
    "row_id_in_payload": False,
    "zero_result_replacement": 1,
}
_EXPECTED_BACKTEST_CONSTRUCTOR: Final[dict[str, object]] = {
    "config": "final_AlphaMaxBacktestConfig",
    "data_handler_kwargs": {
        "backtest_poll_seconds": 1,
        "backtest_window_seconds": 1,
        "feature_db_path": None,
        "feature_exchange": "binance",
        "feature_lookup": "phase_owned_AlphaMaxOrderedFundingLookup_by_identity",
        "market_window_parity_v2_enabled": True,
    },
    "execution_handler_kwargs": {"record_cost_attribution": True},
    "portfolio_kwargs": {
        "fill_application_attribution_sink": "collector.record_application",
        "funding_boundary_resolver": "phase_owned_AlphaMaxFundingBoundaryResolver_by_identity",
    },
    "record_history": True,
    "record_trades": True,
    "strategy_timeframe": "1s",
    "strict_data_handler_construction": True,
    "track_metrics": True,
    "warmup_bars": 0,
}
_EXPECTED_PORTFOLIO_STRATEGY_CONSTRUCTOR: Final[dict[str, object]] = {
    "class_name": "ArtifactPortfolioModeStrategy",
    "decision_cadence_seconds": 1,
    "portfolio_mode": "manifest:<immutable_absolute_row_path>",
    "strategy_params_exact_keys": ["decision_cadence_seconds", "portfolio_mode"],
}

_SYMBOL_LIMIT: Final[dict[str, float]] = {
    "min_notional": 5.0,
    "min_qty": 0.001,
    "price_tick_size": 1e-8,
    "qty_step": 0.001,
}
_EXPECTED_STATIC_ATTRIBUTES: Final[dict[str, object]] = {
    "ALLOW_MARKET_ORDERS": True,
    "ALLOW_METADATA_RISK_OVERRIDE": False,
    "ANNUAL_PERIODS": 2190,
    "APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS": True,
    "ATTACH_DEFAULT_PROTECTIVE_STOP": False,
    "AUTO_FLATTEN_ON_BREACH": False,
    "BACKTEST_DECISION_SECONDS": 1,
    "BACKTEST_POLL_SECONDS": 1,
    "BACKTEST_WINDOW_SECONDS": 1,
    "CHUNK_DAYS": 1,
    "CHUNK_WARMUP_BARS": 0,
    "COMMISSION_RATE": 0.0004,
    "COMPUTE_BACKEND": "cpu",
    "CONSECUTIVE_LOSS_HALT_COUNT": 0,
    "DATA_SOURCE": "explicit_raw_parquet_root",
    "DECISION_CADENCE_SECONDS": 1,
    "DEFAULT_ORDER_TYPE": "MKT",
    "DEFAULT_STOP_LOSS_PCT": 0.01,
    "EFFECTIVE_POSITION_FRACTION": 1.0,
    "ENFORCE_ORDER_RISK_GATE_IN_BACKTEST": False,
    "ENFORCE_REDUCE_ONLY": True,
    "FREEZE_NEW_ENTRIES_ON_BREACH": True,
    "FUNDING_INTERVAL_HOURS": 8,
    "FUNDING_ON_UTC_BOUNDARY": True,
    "FUNDING_RATE_PER_8H": 0.0,
    "GPU_MODE": "cpu",
    "GPU_VRAM_GB": 0.0,
    "HARD_DRAWDOWN_FLATTEN_PCT": 0.0,
    "INITIAL_CAPITAL": 10000.0,
    "LEVERAGE": 3,
    "LIMIT_PRICE_MODE": "one_tick_worse",
    "LIMIT_PRICE_OFFSET_TICKS": 1,
    "LIMIT_TIME_IN_FORCE": "GTC",
    "LIQUIDATION_BUFFER_RATE": 0.0005,
    "MAINTENANCE_MARGIN_RATE": 0.005,
    "MAKER_FEE_RATE": 0.0002,
    "MARGIN_MODE": "isolated",
    "MARKET_WINDOW_PARITY_V2_ENABLED": True,
    "MAX_DAILY_LOSS_PCT": 0.03,
    "MAX_INTRADAY_DRAWDOWN_PCT": 0.03,
    "MAX_LEVERAGE": 3,
    "MAX_ORDER_NOTIONAL_PCT": 0.5,
    "MAX_ORDER_VALUE": 5000.0,
    "MAX_POSITION_SIZE_PCT": 0.5,
    "MAX_ROLLING_LOSS_PCT_1H": 0.05,
    "MAX_SYMBOL_EXPOSURE_PCT": 0.5,
    "MAX_TOTAL_MARGIN_PCT": 0.75,
    "MAX_TOTAL_NOTIONAL_PCT": 2.25,
    "MIN_TRADE_QTY": 0.001,
    "MODE": "windowed",
    "PERSIST_OUTPUT": False,
    "POLL_SECONDS": 1,
    "REQUIRE_FUNDING_COVERAGE": True,
    "RISK_FREE_ANNUAL": 0.0,
    "RISK_FREE_MODE": "zero",
    "RISK_FREE_RATE": 0.0,
    "RISK_FREE_SERIES_PATH": "",
    "RISK_PER_TRADE": 0.005,
    "SIM_LATENCY_MAX_BARS": 1,
    "SIM_LATENCY_MIN_BARS": 1,
    "SIM_MAX_BAR_VOLUME_RATIO": 0.1,
    "SKIP_AHEAD_ENABLED": False,
    "SLIPPAGE_ADV_QUOTE": 0.0,
    "SLIPPAGE_IMPACT_COEFFICIENT": 0.1,
    "SLIPPAGE_IMPACT_MODEL": "sqrt_impact",
    "SORTINO_TARGET_ANNUAL": 0.0,
    "SORTINO_TARGET_MODE": "zero",
    "SPREAD_RATE": 0.0002,
    "STRATEGY_QUALITY_ENABLED": False,
    "SYMBOL_LIMITS": {symbol: dict(_SYMBOL_LIMIT) for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS},
    "TAKER_FEE_RATE": 0.0004,
    "TARGET_ALLOCATION": 0.1,
    "TARGET_ALLOCATION_MODE": "notional_fraction",
    "TIMEFRAME": "1s",
    "TIMEFRAMES": ["1s", "4h", "1d"],
    "WINDOW_SECONDS": 1,
}
_EXPECTED_ATTRIBUTE_ALLOWLIST: Final[tuple[str, ...]] = tuple(
    sorted({*_EXPECTED_STATIC_ATTRIBUTES, *_EXPECTED_DERIVED_ATTRIBUTES})
)
_RUNTIME_ATTRIBUTE_SET: Final[frozenset[str]] = frozenset(_EXPECTED_ATTRIBUTE_ALLOWLIST)
_INTERNAL_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {"_contract_sha256", "_read_audit", "_runtime_instance_sha256"}
)
_CONFIG_CONSTRUCTION_TOKEN: Final[object] = object()

_EXPECTED_COST_CELLS: Final[tuple[dict[str, object], ...]] = (
    {
        "additional_modeled_costs": ["sqrt_impact", "funding", "financing", "liquidation"],
        "expected_base_slippage_bps": 5,
        "half_spread_bps": 1,
        "maker_fee_bps": 2,
        "nominal_one_way_bps": 10,
        "selection_reference": False,
        "slippage_rate": 0.0005,
        "taker_fee_bps": 4,
    },
    {
        "additional_modeled_costs": ["sqrt_impact", "funding", "financing", "liquidation"],
        "expected_base_slippage_bps": 10,
        "half_spread_bps": 1,
        "maker_fee_bps": 2,
        "nominal_one_way_bps": 15,
        "selection_reference": False,
        "slippage_rate": 0.001,
        "taker_fee_bps": 4,
    },
    {
        "additional_modeled_costs": ["sqrt_impact", "funding", "financing", "liquidation"],
        "expected_base_slippage_bps": 15,
        "half_spread_bps": 1,
        "maker_fee_bps": 2,
        "nominal_one_way_bps": 20,
        "selection_reference": False,
        "slippage_rate": 0.0015,
        "taker_fee_bps": 4,
    },
    {
        "additional_modeled_costs": ["sqrt_impact", "funding", "financing", "liquidation"],
        "expected_base_slippage_bps": 25,
        "half_spread_bps": 1,
        "maker_fee_bps": 2,
        "nominal_one_way_bps": 30,
        "selection_reference": True,
        "slippage_rate": 0.0025,
        "taker_fee_bps": 4,
    },
)


class UnfrozenRuntimeFieldError(RuntimeError):
    """The engine attempted to read a field absent from the versioned allowlist."""


class FrozenRuntimeMutationError(TypeError):
    """A caller attempted to mutate the constructor-bound runtime config."""


@dataclass(frozen=True, slots=True)
class AlphaMaxCostCell:
    """One exact modeled one-way cost cell from the frozen experiment."""

    nominal_one_way_bps: int
    slippage_rate: float
    expected_base_slippage_bps: int
    selection_reference: bool


@dataclass(frozen=True, slots=True)
class AlphaMaxPhaseWindow:
    """An exact UTC split/fold interval from the descriptor-bound config."""

    phase_id: str
    start_utc: str
    end_utc: str


@dataclass(frozen=True, slots=True)
class AlphaMaxRuntimePreflight:
    """Immutable result of validating the sealed runtime artifact."""

    config_receipt: ArtifactReadReceipt
    config_bytes: bytes
    config_payload_sha256: str
    runtime_contract_sha256: str
    runtime_contract_bytes: bytes
    common_runtime_bytes: bytes
    attribute_allowlist: tuple[str, ...]
    static_attributes: Mapping[str, object]
    cost_cells: tuple[AlphaMaxCostCell, ...]
    phase_windows: Mapping[str, AlphaMaxPhaseWindow]
    candidate_symbols: tuple[str, ...]
    backtest_constructor: Mapping[str, object]
    portfolio_strategy_constructor: Mapping[str, object]
    incumbent_resolution_bytes: bytes
    incumbent_resolution_audit_sha256: str


@dataclass(frozen=True, slots=True)
class AlphaMaxEngineConstructorPlan:
    """Pure, explicit kwargs for later engine orchestration; no engine is created."""

    config: AlphaMaxBacktestConfig
    strategy_timeframe: str
    warmup_bars: int
    record_history: bool
    track_metrics: bool
    record_trades: bool
    strict_data_handler_construction: bool
    data_handler_kwargs: Mapping[str, object]
    portfolio_kwargs: Mapping[str, object]
    execution_handler_kwargs: Mapping[str, object]

    def as_kwargs(self) -> Mapping[str, object]:
        """Expose a read-only top-level constructor mapping."""
        return MappingProxyType(
            {
                "config": self.config,
                "data_handler_kwargs": self.data_handler_kwargs,
                "execution_handler_kwargs": self.execution_handler_kwargs,
                "portfolio_kwargs": self.portfolio_kwargs,
                "record_history": self.record_history,
                "record_trades": self.record_trades,
                "strategy_timeframe": self.strategy_timeframe,
                "strict_data_handler_construction": self.strict_data_handler_construction,
                "track_metrics": self.track_metrics,
                "warmup_bars": self.warmup_bars,
            }
        )


@dataclass(frozen=True, slots=True)
class AlphaMaxAncestorIdentity:
    """Lexical identity of one activation path component."""

    path: str
    device: int
    inode: int
    file_type: int
    link_count: int
    owner_uid: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class AlphaMaxExpectedComponent:
    """Fields retained by the real manifest consumer for one positive child."""

    component_id: str
    strategy_class: str
    symbols: tuple[str, ...]
    params_bytes: bytes
    weight: float
    source_artifact_id: str


@dataclass(frozen=True, slots=True)
class AlphaMaxExpectedDefinition:
    """Descriptor-parsed immutable definition expected from the real consumer."""

    portfolio_mode: str
    artifact_kind: str
    candidate_symbols: tuple[str, ...]
    admitted_symbols: tuple[str, ...]
    admission_manifest_sha256: str
    gross_cap: float
    cash_weight: float
    allocation_method: str
    source_path: str
    source_sha256: str
    components: tuple[AlphaMaxExpectedComponent, ...]
    native_timeframes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlphaMaxArtifactSeal:
    """Pre-construction manifest/config byte seal plus lexical path identities."""

    output_root: str
    phase: str
    manifest_path: str
    ancestor_identities: tuple[AlphaMaxAncestorIdentity, ...]
    manifest_receipt: ArtifactReadReceipt
    config_receipt: ArtifactReadReceipt
    manifest_bytes: bytes
    config_bytes: bytes
    expected_definition: AlphaMaxExpectedDefinition

    @property
    def consumer_receipts(self) -> tuple[ArtifactReadReceipt, ArtifactReadReceipt]:
        return (self.manifest_receipt, self.config_receipt)


class AlphaMaxAttributionCollector:
    """Constructor-owned append-only sink for post-clamp fill applications."""

    __slots__ = ("_applications",)

    def __init__(self) -> None:
        self._applications: list[FillApplicationAttribution] = []

    def record_application(self, application: FillApplicationAttribution) -> None:
        if type(application) is not FillApplicationAttribution:
            raise TypeError("alpha_max_fill_application_identity_invalid")
        self._applications.append(application)

    @property
    def applications(self) -> tuple[FillApplicationAttribution, ...]:
        return tuple(self._applications)


class _AlphaMaxFoldEquityFanout(AlphaMaxStreamingEquityTracker):
    """Fan out full-event equity and retain causal completed-4h endpoints.

    Sparse raw roots are valid, so a reporting boundary need not have a raw row
    at that exact second.  At the first event after a boundary (or the explicit
    day-end settlement), the engine aggregator already owns the completed native
    4h bars and Portfolio has settled boundary funding on the pre-fill position.
    We mark positions to those completed native closes rather than interpolating
    or substituting the later raw close.
    """

    __slots__ = (
        "_aggregate_scale",
        "_aggregate_tracker",
        "_backtest",
        "_end_ms",
        "_next_boundary_ms",
        "_normalized_segment_tracker",
        "_reporting_endpoints",
    )

    def __init__(
        self,
        aggregate_tracker: AlphaMaxStreamingEquityTracker,
        *,
        aggregate_scale: float,
        reporting_start: datetime,
        reporting_end: datetime,
    ) -> None:
        super().__init__()
        if type(aggregate_tracker) is not AlphaMaxStreamingEquityTracker:
            raise TypeError("alpha_max_aggregate_equity_tracker_identity_invalid")
        if (
            type(aggregate_scale) is not float
            or not math.isfinite(aggregate_scale)
            or aggregate_scale <= 0.0
        ):
            raise ValueError("alpha_max_aggregate_equity_scale_invalid")
        self._aggregate_tracker = aggregate_tracker
        self._aggregate_scale = aggregate_scale
        self._normalized_segment_tracker = AlphaMaxStreamingEquityTracker()
        start_ms = int(reporting_start.timestamp() * 1000)
        end_ms = int(reporting_end.timestamp() * 1000)
        if start_ms % 14_400_000 or end_ms % 14_400_000 or end_ms <= start_ms:
            raise ValueError("alpha_max_reporting_boundary_invalid")
        self._next_boundary_ms = start_ms + 14_400_000
        self._end_ms = end_ms
        self._reporting_endpoints: list[AlphaMaxEquityEndpoint] = []
        self._backtest: Backtest | None = None

    def bind_backtest(self, backtest: Backtest) -> None:
        if type(backtest) is not Backtest:
            raise TypeError("alpha_max_reporting_backtest_identity_invalid")
        self._backtest = backtest

    @staticmethod
    def _bar_timestamp_ms(value: object) -> int:
        if isinstance(value, datetime):
            timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return int(timestamp.astimezone(UTC).timestamp() * 1000)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value)
            if math.isfinite(parsed):
                return int(parsed if abs(parsed) >= 1e11 else parsed * 1000.0)
        raise AlphaMaxRuntimeContractError("alpha_max_reporting_native_bar_timestamp_invalid")

    def _completed_native_close(self, symbol: str, boundary_ms: int) -> float:
        if self._backtest is None or self._backtest.timeframe_aggregator is None:
            raise AlphaMaxRuntimeContractError("alpha_max_reporting_aggregator_missing")
        state = self._backtest.timeframe_aggregator.get_state()
        target_bucket_ms = boundary_ms - 14_400_000
        candidates: list[object] = []
        history = state.get("history", {})
        if isinstance(history, Mapping):
            symbol_history = history.get(symbol, {})
            if isinstance(symbol_history, Mapping):
                values = symbol_history.get("4h", ())
                if isinstance(values, (list, tuple)):
                    candidates.extend(values)
        working = state.get("working", {})
        if isinstance(working, Mapping):
            symbol_working = working.get(symbol, {})
            if isinstance(symbol_working, Mapping):
                value = symbol_working.get("4h")
                if isinstance(value, Mapping):
                    candidates.append(
                        (
                            value.get("time"),
                            value.get("open"),
                            value.get("high"),
                            value.get("low"),
                            value.get("close"),
                            value.get("volume"),
                        )
                    )
        matches = [
            value
            for value in candidates
            if isinstance(value, (list, tuple))
            and len(value) >= 5
            and self._bar_timestamp_ms(value[0]) == target_bucket_ms
        ]
        if len(matches) != 1:
            raise AlphaMaxRuntimeContractError("alpha_max_reporting_native_bar_incomplete")
        close = float(matches[0][4])
        if not math.isfinite(close) or close <= 0.0:
            raise AlphaMaxRuntimeContractError("alpha_max_reporting_native_close_invalid")
        return close

    def _report_boundary(self, boundary_ms: int, *, commit_holdings: bool = False) -> None:
        if self._backtest is None:
            raise AlphaMaxRuntimeContractError("alpha_max_reporting_backtest_missing")
        portfolio = self._backtest.portfolio
        cash = float(portfolio.current_holdings["cash"])
        market_values = {
            symbol: float(portfolio.current_positions[symbol])
            * self._completed_native_close(symbol, boundary_ms)
            for symbol in portfolio.symbol_list
        }
        total = cash + math.fsum(market_values.values())
        if not math.isfinite(total):
            raise AlphaMaxRuntimeContractError("alpha_max_reporting_endpoint_nonfinite")
        if commit_holdings:
            portfolio.current_holdings.update(market_values)
            portfolio.current_holdings["total"] = total
        self._reporting_endpoints.append(
            AlphaMaxEquityEndpoint(
                timestamp=datetime.fromtimestamp(boundary_ms / 1000.0, tz=UTC),
                equity=total,
            )
        )
        self._next_boundary_ms += 14_400_000

    def _observe_full_event(self, point: tuple[float, float]) -> None:
        super().observe(point)
        normalized = (point[0], self._aggregate_scale * point[1])
        self._normalized_segment_tracker.observe(normalized)
        self._aggregate_tracker.observe(normalized)

    def observe(self, point: tuple[float, float]) -> None:
        timestamp_ms = int(float(point[0]) * 1000.0)
        while self._next_boundary_ms < self._end_ms and timestamp_ms >= self._next_boundary_ms:
            self._report_boundary(self._next_boundary_ms)
        self._observe_full_event(point)

    def update_batch(self, points: Any) -> None:
        """Fan out a boundary-free exact equity batch."""
        if (
            type(points) is not np.ndarray
            or points.dtype != np.dtype(np.float64)
            or points.ndim != 2
            or points.shape[1:] != (2,)
            or points.shape[0] == 0
        ):
            raise TypeError("alpha_max_equity_fanout_batch_invalid")
        last_timestamp_ms = int(float(points[-1, 0]) * 1000.0)
        if last_timestamp_ms >= self._next_boundary_ms:
            raise AlphaMaxRuntimeContractError("alpha_max_equity_fanout_batch_crossed_boundary")
        super().update_batch(points)
        normalized = points.copy()
        normalized[:, 1] = normalized[:, 1] * self._aggregate_scale
        self._normalized_segment_tracker.update_batch(normalized)
        self._aggregate_tracker.update_batch(normalized)

    def settle_day_end(self, boundary: datetime, *, settle_funding: bool) -> None:
        """Emit a native-close endpoint after the crossed boundary is settled."""
        boundary_ms = int(boundary.timestamp() * 1000)
        if boundary_ms != self._next_boundary_ms or boundary_ms > self._end_ms:
            raise AlphaMaxRuntimeContractError("alpha_max_reporting_boundary_sequence_invalid")
        if self._backtest is None:
            raise AlphaMaxRuntimeContractError("alpha_max_reporting_backtest_missing")
        portfolio = self._backtest.portfolio
        if settle_funding:
            portfolio._apply_funding(boundary)
        self._report_boundary(boundary_ms, commit_holdings=True)
        endpoint = self._reporting_endpoints[-1]
        self._observe_full_event((boundary.timestamp(), endpoint.equity))

    @property
    def reporting_endpoints(self) -> tuple[AlphaMaxEquityEndpoint, ...]:
        if self._next_boundary_ms != self._end_ms + 14_400_000:
            raise AlphaMaxRuntimeContractError("alpha_max_reporting_endpoint_incomplete")
        return tuple(self._reporting_endpoints)

    @property
    def normalized_segment_tracker(self) -> AlphaMaxStreamingEquityTracker:
        return self._normalized_segment_tracker


def _alpha_max_signal_time(value: object) -> object:
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")
    if value is None or type(value) in {str, int, float}:
        return value
    raise AlphaMaxRuntimeContractError("alpha_max_boundary_signal_time_invalid")


def _alpha_max_boundary_signal_payload(
    component_id: str,
    signal: SignalEvent,
) -> dict[str, object]:
    if type(signal) is not SignalEvent:
        raise AlphaMaxRuntimeContractError("alpha_max_boundary_event_invalid")
    payload: dict[str, object] = {
        "client_order_id": signal.client_order_id,
        "component_id": component_id,
        "datetime": _alpha_max_signal_time(signal.datetime),
        "metadata": signal.metadata,
        "position_side": signal.position_side,
        "price": signal.price,
        "sequence": signal.sequence,
        "signal_type": signal.signal_type,
        "stop_loss": signal.stop_loss,
        "strategy_id": signal.strategy_id,
        "strength": signal.strength,
        "symbol": signal.symbol,
        "take_profit": signal.take_profit,
        "time_in_force": signal.time_in_force,
        "timestamp_ns": signal.timestamp_ns,
        "trailing_percent": signal.trailing_percent,
        "type": signal.type,
    }
    try:
        _canonical_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_boundary_signal_payload_invalid") from exc
    return payload


def _alpha_max_assert_parent_queue_empty(events: FastQueue) -> None:
    if type(events) is not FastQueue:
        raise AlphaMaxRuntimeContractError("alpha_max_boundary_parent_queue_invalid")
    try:
        event = events.get(False)
    except queue.Empty:
        return
    raise AlphaMaxRuntimeContractError(
        f"alpha_max_boundary_parent_queue_not_empty:{getattr(event, 'type', 'UNKNOWN')!s}"
    )


_ALPHA_MAX_NATIVE_SNAPSHOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "adapter_class",
        "native_timeframe",
        "barrier_mode",
        "completed_native_keys",
        "completed_native_count_by_symbol",
        "last_completed_native_key_by_symbol",
        "barrier_pending_keys",
        "barrier_closed_keys",
        "barrier_symbol_coverage",
        "failed_native_keys",
        "partial_bucket_error",
    }
)


def _alpha_max_native_coverage_snapshot(child: object) -> dict[str, Any]:
    getter = getattr(child, "get_native_finalization_evidence", None)
    if not callable(getter):
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_native_finalization_coverage_missing")
    try:
        raw = getter()
    except Exception as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_scoring_native_finalization_coverage_invalid"
        ) from exc
    if type(raw) is not dict or set(raw) != _ALPHA_MAX_NATIVE_SNAPSHOT_KEYS:
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_native_finalization_coverage_invalid")
    try:
        snapshot = copy.deepcopy(raw)
        _canonical_bytes(snapshot)
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_scoring_native_finalization_coverage_invalid"
        ) from exc
    return snapshot


def _alpha_max_native_completed_key_set(snapshot: Mapping[str, Any]) -> set[tuple[str, str]]:
    raw = snapshot.get("completed_native_keys")
    if not isinstance(raw, (list, tuple)):
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_native_finalization_coverage_invalid")
    result: set[tuple[str, str]] = set()
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not str
            or not item[1]
        ):
            raise AlphaMaxRuntimeContractError(
                "alpha_max_scoring_native_finalization_coverage_invalid"
            )
        result.add((item[0], item[1]))
    if len(result) != len(raw):
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_native_finalization_coverage_invalid")
    return result


def _alpha_max_native_barrier_key_set(
    snapshot: Mapping[str, Any],
    *,
    field: str,
) -> set[str]:
    raw = snapshot.get(field)
    if not isinstance(raw, (list, tuple)) or any(type(item) is not str or not item for item in raw):
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_native_finalization_coverage_invalid")
    result = set(raw)
    if len(result) != len(raw):
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_native_finalization_coverage_invalid")
    return result


def _alpha_max_assert_native_coverage_binding(
    snapshot: Mapping[str, Any],
    *,
    component: AlphaMaxExpectedComponent,
    admitted_symbols: tuple[str, ...],
) -> None:
    expected_timeframe = _ALPHA_MAX_NATIVE_TIMEFRAME_BY_CLASS.get(component.strategy_class)
    expected_barrier_mode = (
        "atomic_cross_section"
        if component.strategy_class == "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy"
        else "none"
    )
    completed_symbols = {symbol for symbol, _key in _alpha_max_native_completed_key_set(snapshot)}
    raw_counts = snapshot.get("completed_native_count_by_symbol")
    raw_last = snapshot.get("last_completed_native_key_by_symbol")
    raw_barrier_coverage = snapshot.get("barrier_symbol_coverage")
    admitted_set = set(admitted_symbols)
    if (
        expected_timeframe is None
        or tuple(component.symbols) != admitted_symbols
        or snapshot.get("adapter_class") != component.strategy_class
        or snapshot.get("native_timeframe") != expected_timeframe
        or snapshot.get("barrier_mode") != expected_barrier_mode
        or completed_symbols != admitted_set
        or not isinstance(raw_counts, Mapping)
        or set(raw_counts) != admitted_set
        or not isinstance(raw_last, Mapping)
        or set(raw_last) != admitted_set
        or not isinstance(raw_barrier_coverage, Mapping)
        or (
            expected_barrier_mode == "atomic_cross_section"
            and any(set(symbols) != admitted_set for symbols in raw_barrier_coverage.values())
        )
        or (expected_barrier_mode == "none" and bool(raw_barrier_coverage))
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_scoring_native_finalization_coverage_binding_mismatch"
        )


def _finalize_alpha_max_native_boundary(
    strategy: ArtifactPortfolioModeStrategy,
    expected_definition: AlphaMaxExpectedDefinition,
    boundary: datetime,
    *,
    admitted_symbol_count: int,
    require_exact_counts: bool,
) -> AlphaMaxNativeFinalizationReceipt:
    """Finalize each child once and seal the deliberately discarded boundary signals."""
    if type(strategy) is not ArtifactPortfolioModeStrategy:
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_strategy_identity_invalid")
    if type(expected_definition) is not AlphaMaxExpectedDefinition:
        raise TypeError("alpha_max_expected_definition_identity_invalid")
    if type(admitted_symbol_count) is not int or admitted_symbol_count <= 0:
        raise ValueError("alpha_max_admitted_symbol_count_invalid")
    expected_components = expected_definition.components
    expected_ids = tuple(component.component_id for component in expected_components)
    children = getattr(strategy, "_children", None)
    if (
        type(children) is not list
        or tuple(component.component_id for component, _child, _child_queue in children)
        != expected_ids
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_child_queue_coverage_invalid")
    _alpha_max_assert_parent_queue_empty(strategy.events)
    before_coverage: dict[str, dict[str, Any]] = {}
    for component, child, _child_queue in children:
        snapshot = _alpha_max_native_coverage_snapshot(child)
        _alpha_max_assert_native_coverage_binding(
            snapshot,
            component=component,
            admitted_symbols=expected_definition.admitted_symbols,
        )
        before_coverage[component.component_id] = snapshot
    finalized = strategy.finalize_completed_native_buckets(boundary)
    if type(finalized) is not dict or tuple(finalized) != expected_ids:
        raise AlphaMaxRuntimeContractError("alpha_max_scoring_native_finalization_invalid")
    normalized: dict[str, int] = {}
    native_coverage: dict[str, dict[str, Any]] = {}
    signal_payloads: list[dict[str, object]] = []
    expected_by_id = {component.component_id: component for component in expected_components}
    for component_id, value in finalized.items():
        if type(value) is not int or value <= 0:
            raise AlphaMaxRuntimeContractError("alpha_max_scoring_native_finalization_invalid")
        component = expected_by_id[component_id]
        if require_exact_counts:
            expected_count = (
                1
                if component.strategy_class
                == "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy"
                else admitted_symbol_count
            )
            if value != expected_count:
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_scoring_native_finalization_count_mismatch"
                )
        normalized[component_id] = value
    for component, child, _child_queue in children:
        component_id = component.component_id
        before = before_coverage[component_id]
        after = _alpha_max_native_coverage_snapshot(child)
        _alpha_max_assert_native_coverage_binding(
            after,
            component=component,
            admitted_symbols=expected_definition.admitted_symbols,
        )
        if any(
            after[field] != before[field]
            for field in ("adapter_class", "native_timeframe", "barrier_mode")
        ):
            raise AlphaMaxRuntimeContractError(
                "alpha_max_scoring_native_finalization_coverage_invalid"
            )
        before_completed = _alpha_max_native_completed_key_set(before)
        after_completed = _alpha_max_native_completed_key_set(after)
        before_pending = _alpha_max_native_barrier_key_set(before, field="barrier_pending_keys")
        after_pending = _alpha_max_native_barrier_key_set(after, field="barrier_pending_keys")
        before_closed = _alpha_max_native_barrier_key_set(before, field="barrier_closed_keys")
        after_closed = _alpha_max_native_barrier_key_set(after, field="barrier_closed_keys")
        if (
            not before_completed.issubset(after_completed)
            or not before_pending.issubset(after_pending)
            or not before_closed.issubset(after_closed)
        ):
            raise AlphaMaxRuntimeContractError(
                "alpha_max_scoring_native_finalization_coverage_regressed"
            )
        coverage = copy.deepcopy(after)
        coverage["finalization_completed_native_keys"] = sorted(after_completed - before_completed)
        coverage["finalization_barrier_keys"] = sorted(after_closed - before_closed)
        native_coverage[component_id] = coverage
    for component, _child, child_queue in children:
        drain = getattr(child_queue, "drain", None)
        if not callable(drain):
            raise AlphaMaxRuntimeContractError("alpha_max_scoring_child_queue_invalid")
        drained = drain()
        if type(drained) is not list:
            raise AlphaMaxRuntimeContractError("alpha_max_scoring_child_queue_invalid")
        signal_payloads.extend(
            _alpha_max_boundary_signal_payload(component.component_id, event) for event in drained
        )
    _alpha_max_assert_parent_queue_empty(strategy.events)
    signal_bytes = b"".join(_canonical_bytes(payload) + b"\n" for payload in signal_payloads)
    try:
        return build_alpha_max_native_finalization_receipt(
            boundary_utc=boundary,
            finalized_children=normalized,
            native_coverage_by_child=native_coverage,
            discarded_signal_count=len(signal_payloads),
            discarded_signal_sha256=_sha256(signal_bytes),
        )
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_scoring_native_finalization_coverage_invalid"
        ) from exc


def _settle_alpha_max_day_boundary(
    activation: AlphaMaxEngineActivation,
    tracker: _AlphaMaxFoldEquityFanout,
    boundary: datetime,
    *,
    scoring_boundary: bool,
) -> AlphaMaxNativeFinalizationReceipt | None:
    """Settle causal economics, then finalize native state at a score handoff."""
    if type(activation) is not AlphaMaxEngineActivation:
        raise TypeError("alpha_max_engine_activation_required")
    if type(tracker) is not _AlphaMaxFoldEquityFanout:
        raise TypeError("alpha_max_fold_equity_tracker_required")
    tracker.settle_day_end(boundary, settle_funding=True)
    if not scoring_boundary:
        return None
    return _finalize_alpha_max_native_boundary(
        activation.backtest.strategy,
        activation.artifact_seal.expected_definition,
        boundary,
        admitted_symbol_count=len(activation.admitted_symbols),
        require_exact_counts=True,
    )


@dataclass(frozen=True, slots=True)
class AlphaMaxEngineActivation:
    """A constructed real engine that has passed the first activation assertion."""

    backtest: Backtest
    preflight: AlphaMaxRuntimePreflight
    constructor_plan: AlphaMaxEngineConstructorPlan
    artifact_seal: AlphaMaxArtifactSeal
    phase_id: str
    raw_root: str
    admitted_symbols: tuple[str, ...]
    ordered_lookup: AlphaMaxOrderedFundingLookup
    funding_resolver: AlphaMaxFundingBoundaryResolver
    attribution_collector: AlphaMaxAttributionCollector
    full_event_equity_tracker: AlphaMaxStreamingEquityTracker
    strategy_params: Mapping[str, object]
    indicator_capsule: AlphaMaxIndicatorCapsule | None
    restored_capsule_sha256: str | None
    raw_root_seals: tuple[AlphaMaxRootSeal, ...]
    feature_root_seals: tuple[AlphaMaxRootSeal, ...]
    repeat_root_hash_on_activation: bool
    chunk_start_utc: datetime
    chunk_end_utc: datetime


@dataclass(frozen=True, slots=True)
class _AlphaMaxDailyCarry:
    """Compact exact state transferred between fresh daily Backtests."""

    strategy_state: dict[str, object]
    portfolio_state: dict[str, object]
    execution_state: dict[str, object]
    engine_state: dict[str, object]
    handler_rows: tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]
    handler_timestamps_ms: tuple[tuple[str, tuple[int | None, ...]], ...]
    funding_ledger: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _AlphaMaxFoldReplayInput:
    """Causal artifacts and bounded roots used by one exact scored fold."""

    fold_id: str
    raw_root: str
    ordered_lookup: AlphaMaxOrderedFundingLookup
    indicator_capsule: AlphaMaxIndicatorCapsule
    capsule_receipt: AlphaMaxCapsuleReceipt
    raw_root_seals: tuple[AlphaMaxRootSeal, ...]
    feature_root_seals: tuple[AlphaMaxRootSeal, ...]
    bounded_raw_loader: _AlphaMaxBoundedRawLoader

    def __post_init__(self) -> None:
        if self.fold_id not in {
            *_ALPHA_MAX_VALIDATION_FOLD_IDS,
            *_ALPHA_MAX_HISTORICAL_FOLD_IDS,
        }:
            raise AlphaMaxRuntimeContractError("alpha_max_fold_replay_id_invalid")
        if type(self.indicator_capsule) is not AlphaMaxIndicatorCapsule:
            raise TypeError("alpha_max_indicator_capsule_identity_invalid")
        if self.indicator_capsule.phase_id != _alpha_max_capsule_predecessor(self.fold_id):
            raise AlphaMaxRuntimeContractError("alpha_max_fold_capsule_predecessor_invalid")
        if type(self.capsule_receipt) is not AlphaMaxCapsuleReceipt:
            raise TypeError("alpha_max_capsule_receipt_identity_invalid")
        if self.capsule_receipt.prefix_id != self.fold_id:
            raise AlphaMaxRuntimeContractError("alpha_max_fold_capsule_receipt_scope_invalid")
        if (
            self.capsule_receipt.capsule_phase_id != self.indicator_capsule.phase_id
            or _canonical_bytes(dict(self.capsule_receipt.state_payload))
            != _canonical_bytes(_alpha_max_capsule_state_payload(self.indicator_capsule))
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_fold_capsule_receipt_state_mismatch")
        if type(self.ordered_lookup) is not AlphaMaxOrderedFundingLookup:
            raise TypeError("alpha_max_ordered_lookup_identity_invalid")
        if type(self.bounded_raw_loader) is not _AlphaMaxBoundedRawLoader:
            raise TypeError("alpha_max_bounded_raw_loader_identity_invalid")
        if self.bounded_raw_loader.seal.path != self.raw_root:
            raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_loader_scope_invalid")


@dataclass(frozen=True, slots=True)
class AlphaMaxIndicatorCapsule:
    """Research-only state produced without Portfolio or Execution economics."""

    portfolio_mode: str
    phase_id: str
    manifest_sha256: str
    capsule_sha256: str
    capsule: Mapping[str, object]
    finalized_children: Mapping[str, object]
    native_finalization_sha256: str
    windows_processed: int
    discarded_signal_count: int
    market_event_count: int = 0
    funding_event_count: int = 0
    order_event_count: int = 0
    fill_event_count: int = 0
    trade_count: int = 0


def _alpha_max_capsule_state_payload(capsule: AlphaMaxIndicatorCapsule) -> dict[str, object]:
    if type(capsule) is not AlphaMaxIndicatorCapsule:
        raise TypeError("alpha_max_indicator_capsule_identity_invalid")
    return {
        "portfolio_mode": capsule.portfolio_mode,
        "phase_id": capsule.phase_id,
        "manifest_sha256": capsule.manifest_sha256,
        "capsule_sha256": capsule.capsule_sha256,
        "capsule": _thaw_json(capsule.capsule),
        "finalized_children": _thaw_json(capsule.finalized_children),
        "native_finalization_sha256": capsule.native_finalization_sha256,
        "windows_processed": capsule.windows_processed,
        "discarded_signal_count": capsule.discarded_signal_count,
        "market_event_count": capsule.market_event_count,
        "funding_event_count": capsule.funding_event_count,
        "order_event_count": capsule.order_event_count,
        "fill_event_count": capsule.fill_event_count,
        "trade_count": capsule.trade_count,
    }


@dataclass(frozen=True, slots=True)
class AlphaMaxIndicatorPhaseInput:
    """One exact phase in a causal indicator-only prefix replay."""

    phase_id: str
    raw_root: str
    ordered_lookup: AlphaMaxOrderedFundingLookup
    watermark: object
    data_dict: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _alpha_max_current_root_id(self.phase_id)
        if type(self.raw_root) is not str:
            raise TypeError("alpha_max_indicator_phase_raw_root_invalid")
        _require_exact_explicit_path(self.raw_root)
        if type(self.ordered_lookup) is not AlphaMaxOrderedFundingLookup:
            raise TypeError("alpha_max_ordered_lookup_identity_invalid")
        if self.data_dict is not None and not isinstance(self.data_dict, Mapping):
            raise TypeError("alpha_max_indicator_phase_data_dict_invalid")


@dataclass(frozen=True, slots=True)
class AlphaMaxMatrixCellStatus:
    """One of the frozen 84 row/cost statuses."""

    row_id: str
    row_role: str
    nominal_cost_bps: int
    status: str
    engine_constructed: bool
    selection_eligible: bool
    capsule_sha256: str | None
    manifest_sha256: str | None
    evidence: object | None = None


@dataclass(frozen=True, slots=True)
class AlphaMaxMatrixResult:
    """The complete 21-row by four-cost prelock status surface."""

    statuses: tuple[AlphaMaxMatrixCellStatus, ...]
    resolvable_row_ids: tuple[str, ...]
    unavailable_row_ids: tuple[str, ...]
    diagnostic_row_ids: tuple[str, ...]

    @property
    def engine_cell_count(self) -> int:
        return sum(1 for status in self.statuses if status.engine_constructed)


@dataclass(frozen=True, slots=True)
class AlphaMaxSealedBundle:
    """Physical append-only bundle written beneath one newly-created root."""

    output_root: str
    stable_paths: tuple[str, ...]
    seal_path: str
    seal_sha256: str


@dataclass(frozen=True, slots=True)
class AlphaMaxCommandResult:
    """Stable outcome of one physical alpha-max process boundary."""

    exit_code: int
    terminal_outcome: str
    bundle: AlphaMaxSealedBundle
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlphaMaxAllocatorFitEvidence:
    """Typed, separately sealed nominal-20 daily-return allocator input."""

    phase: str
    component_ids: tuple[str, ...]
    calendar: tuple[str, ...]
    returns_by_component: Mapping[str, tuple[float, ...]]
    weights_by_row: Mapping[str, Mapping[str, float]]
    input_sha256: str

    def __post_init__(self) -> None:
        if self.phase not in {"train", "train_validation"}:
            raise ValueError("alpha_max_allocator_fit_phase_invalid")
        if (
            self.component_ids != tuple(sorted(self.component_ids))
            or len(self.component_ids) != 3
            or set(self.returns_by_component) != set(self.component_ids)
            or len(self.calendar) < 252
            or len(self.calendar) != len(set(self.calendar))
        ):
            raise ValueError("alpha_max_allocator_fit_input_invalid")
        if any(
            type(values) is not tuple
            or len(values) != len(self.calendar)
            or any(not math.isfinite(float(value)) for value in values)
            for values in self.returns_by_component.values()
        ):
            raise ValueError("alpha_max_allocator_fit_input_invalid")
        expected = _sha256(
            _canonical_bytes(
                {
                    "calendar": list(self.calendar),
                    "component_ids": list(self.component_ids),
                    "nominal_cost_bps": 20,
                    "returns_by_component": {
                        key: list(self.returns_by_component[key]) for key in self.component_ids
                    },
                }
            )
        )
        if self.input_sha256 != expected:
            raise ValueError("alpha_max_allocator_fit_hash_mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "alpha_max_allocator_fit_evidence.v1",
            "calendar": list(self.calendar),
            "component_ids": list(self.component_ids),
            "input_sha256": self.input_sha256,
            "nominal_cost_bps": 20,
            "phase": self.phase,
            "returns_by_component": {
                key: list(self.returns_by_component[key]) for key in self.component_ids
            },
            "weights_by_row": {
                row_id: dict(weights) for row_id, weights in sorted(self.weights_by_row.items())
            },
        }


@dataclass(frozen=True, slots=True)
class _AlphaMaxCompletedMatrix:
    domain: str
    rows: tuple[AlphaMaxRowEvidence, ...]
    cells: Mapping[tuple[str, int], AlphaMaxCostCellEvidence]
    status_payload: bytes
    physical_fold_run_count: int
    prepared_rows: Mapping[str, _AlphaMaxPreparedReplayRow]
    gross_by_row: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _AlphaMaxPreparedReplayRow:
    manifest_receipt: AlphaMaxManifestReceipt
    fold_inputs: tuple[_AlphaMaxFoldReplayInput, ...]
    gross: float


class _AlphaMaxRowExecutor(Protocol):
    def __call__(
        self,
        row: Mapping[str, object],
        nominal_cost_bps: int,
    ) -> AlphaMaxCostCellEvidence: ...


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AlphaMaxRuntimeContractError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise AlphaMaxRuntimeContractError(f"nonfinite_json_constant:{value}")


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_config_json_invalid") from exc
    if type(value) is not dict:
        raise AlphaMaxRuntimeContractError("alpha_max_config_root_not_object")
    _assert_finite_json_tree(value)
    return value


def _assert_finite_json_tree(value: object, path: str = "$") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise AlphaMaxRuntimeContractError(f"nonfinite_json_number:{path}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_finite_json_tree(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            _assert_finite_json_tree(child, f"{path}.{key}")
        return
    raise AlphaMaxRuntimeContractError(f"unsupported_json_value:{path}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exact_state_equal(left: object, right: object) -> bool:
    """Compare checkpoint trees without coercion or JSON serialization."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(  # type: ignore[arg-type]
            _exact_state_equal(left[key], right[key])  # type: ignore[index]
            for key in left
        )
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _exact_state_equal(a, b)
            for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    if type(left) in {set, frozenset}:
        return left == right
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(child) for key, child in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if type(value) is tuple:
        return [_thaw_json(child) for child in value]
    return value


def _require_exact_explicit_path(path: str | os.PathLike[str]) -> str:
    raw = os.fspath(path)
    if not raw or not os.path.isabs(raw) or os.path.abspath(raw) != raw:
        raise AlphaMaxRuntimeContractError("alpha_max_config_path_not_explicit_canonical")
    return raw


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise AlphaMaxRuntimeContractError(f"alpha_max_runtime_schema_invalid:{field}")
    return value


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if value != expected:
        raise AlphaMaxRuntimeContractError(f"alpha_max_runtime_contract_mismatch:{field}")


def _validate_incumbent_resolution_contract(config: Mapping[str, object]) -> bytes:
    """Validate the embedded audit without consulting planning/worktree files."""
    incumbent = _require_mapping(
        config.get("incumbent_resolution"),
        field="incumbent_resolution",
    )
    canonical = _canonical_bytes(incumbent)
    if _sha256(canonical) != _ALPHA_MAX_INCUMBENT_RESOLUTION_CANONICAL_SHA256:
        raise AlphaMaxRuntimeContractError("alpha_max_incumbent_resolution_mismatch")
    normative = _require_mapping(config.get("normative_sources"), field="normative_sources")
    if (
        normative.get("incumbent_resolution_audit_sha256")
        != ALPHA_MAX_INCUMBENT_RESOLUTION_AUDIT_SHA256
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_incumbent_resolution_audit_hash_mismatch")
    return canonical


def _validate_runtime_contract(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != _TOP_LEVEL_KEYS:
        raise AlphaMaxRuntimeContractError("alpha_max_config_top_level_schema_mismatch")
    _require_exact(
        config.get("schema_version"), "alpha_max_portfolio_experiment.v1", field="schema"
    )
    _require_exact(
        config.get("experiment_id"),
        "alpha_max_portfolio_20260711_listing_aware",
        field="experiment",
    )
    _require_exact(config.get("revision"), "5.15", field="revision")
    _require_exact(
        config.get("candidate_symbols"), list(ALPHA_MAX_CANDIDATE_SYMBOLS), field="symbols"
    )

    runtime = _require_mapping(config.get("runtime_contract"), field="runtime_contract")
    if set(runtime) != _RUNTIME_KEYS:
        raise AlphaMaxRuntimeContractError("alpha_max_runtime_schema_key_mismatch")
    _require_exact(runtime.get("schema_version"), "alpha_max_runtime_contract.v1", field="schema")
    _require_exact(runtime.get("class_name"), "AlphaMaxBacktestConfig", field="class_name")
    _require_exact(runtime.get("construction"), _EXPECTED_CONSTRUCTION, field="construction")
    _require_exact(
        runtime.get("override_policy"), _EXPECTED_OVERRIDE_POLICY, field="override_policy"
    )
    _require_exact(
        runtime.get("derived_attributes"),
        _EXPECTED_DERIVED_ATTRIBUTES,
        field="derived_attributes",
    )
    _require_exact(runtime.get("seed_schedule"), _EXPECTED_SEED_SCHEDULE, field="seed_schedule")
    _require_exact(
        runtime.get("backtest_constructor"),
        _EXPECTED_BACKTEST_CONSTRUCTOR,
        field="backtest_constructor",
    )
    _require_exact(
        runtime.get("portfolio_strategy_constructor"),
        _EXPECTED_PORTFOLIO_STRATEGY_CONSTRUCTOR,
        field="portfolio_strategy_constructor",
    )
    _require_exact(
        runtime.get("attribute_allowlist"),
        list(_EXPECTED_ATTRIBUTE_ALLOWLIST),
        field="attribute_allowlist",
    )
    _require_exact(
        runtime.get("static_attributes"),
        _EXPECTED_STATIC_ATTRIBUTES,
        field="static_attributes",
    )
    _require_exact(config.get("cost_cells"), list(_EXPECTED_COST_CELLS), field="cost_cells")
    _validate_incumbent_resolution_contract(config)

    integrity = _require_mapping(config.get("integrity"), field="integrity")
    if set(integrity) != {
        "canonicalization",
        "config_payload_sha256",
        "config_payload_sha256_scope",
        "runtime_contract_sha256",
    }:
        raise AlphaMaxRuntimeContractError("alpha_max_integrity_schema_mismatch")
    _require_exact(
        integrity.get("canonicalization"),
        _EXPECTED_CANONICALIZATION,
        field="canonicalization",
    )
    _require_exact(
        integrity.get("config_payload_sha256_scope"),
        "entire_document_with_integrity.config_payload_sha256_omitted",
        field="config_payload_scope",
    )
    runtime_sha = _sha256(_canonical_bytes(runtime))
    if runtime_sha != ALPHA_MAX_RUNTIME_CONTRACT_SHA256:
        raise AlphaMaxRuntimeContractError("alpha_max_runtime_contract_hash_mismatch")
    if integrity.get("runtime_contract_sha256") != runtime_sha:
        raise AlphaMaxRuntimeContractError("alpha_max_embedded_runtime_hash_mismatch")

    hash_scope = copy.deepcopy(config)
    scope_integrity = _require_mapping(hash_scope["integrity"], field="integrity")
    embedded_payload_sha = scope_integrity.pop("config_payload_sha256", None)
    if embedded_payload_sha != ALPHA_MAX_CONFIG_PAYLOAD_SHA256:
        raise AlphaMaxRuntimeContractError("alpha_max_embedded_payload_hash_mismatch")
    if _sha256(_canonical_bytes(hash_scope)) != ALPHA_MAX_CONFIG_PAYLOAD_SHA256:
        raise AlphaMaxRuntimeContractError("alpha_max_config_payload_hash_mismatch")
    if _sha256(_canonical_bytes(config)) != ALPHA_MAX_CONFIG_CANONICAL_SHA256:
        raise AlphaMaxRuntimeContractError("alpha_max_config_canonical_hash_mismatch")

    return runtime


def _parse_phase_window(phase_id: object, start: object, end: object) -> AlphaMaxPhaseWindow:
    if type(phase_id) is not str or not phase_id:
        raise AlphaMaxRuntimeContractError("alpha_max_phase_id_invalid")
    if type(start) is not str or type(end) is not str:
        raise AlphaMaxRuntimeContractError(f"alpha_max_phase_window_invalid:{phase_id}")
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlphaMaxRuntimeContractError(f"alpha_max_phase_window_invalid:{phase_id}") from exc
    if (
        not start.endswith("Z")
        or not end.endswith("Z")
        or start_dt.tzinfo != UTC
        or end_dt.tzinfo != UTC
        or end_dt <= start_dt
    ):
        raise AlphaMaxRuntimeContractError(f"alpha_max_phase_window_invalid:{phase_id}")
    return AlphaMaxPhaseWindow(phase_id=phase_id, start_utc=start, end_utc=end)


def _phase_windows(config: dict[str, Any]) -> Mapping[str, AlphaMaxPhaseWindow]:
    chronology = _require_mapping(config.get("chronology"), field="chronology")
    windows: dict[str, AlphaMaxPhaseWindow] = {}
    for collection_name, id_key in (
        ("splits", "split_id"),
        ("validation_folds", "fold_id"),
        ("historical_evaluation_folds", "fold_id"),
    ):
        rows = chronology.get(collection_name)
        if type(rows) is not list:
            raise AlphaMaxRuntimeContractError(
                f"alpha_max_runtime_schema_invalid:chronology.{collection_name}"
            )
        for row in rows:
            row_mapping = _require_mapping(row, field=f"chronology.{collection_name}[]")
            window = _parse_phase_window(
                row_mapping.get(id_key),
                row_mapping.get("start_utc"),
                row_mapping.get("end_utc"),
            )
            if window.phase_id in windows:
                raise AlphaMaxRuntimeContractError(
                    f"alpha_max_duplicate_phase_window:{window.phase_id}"
                )
            windows[window.phase_id] = window
    validation_ids = tuple(
        str(row.get("fold_id") or "")
        for row in chronology["validation_folds"]
        if isinstance(row, Mapping)
    )
    historical_ids = tuple(
        str(row.get("fold_id") or "")
        for row in chronology["historical_evaluation_folds"]
        if isinstance(row, Mapping)
    )
    if validation_ids != _ALPHA_MAX_VALIDATION_FOLD_IDS:
        raise AlphaMaxRuntimeContractError("alpha_max_validation_fold_sequence_invalid")
    if historical_ids != _ALPHA_MAX_HISTORICAL_FOLD_IDS:
        raise AlphaMaxRuntimeContractError("alpha_max_historical_fold_sequence_invalid")
    for domain, fold_ids in _ALPHA_MAX_DOMAIN_FOLD_IDS.items():
        parent = windows[domain]
        folds = tuple(windows[fold_id] for fold_id in fold_ids)
        if (
            folds[0].start_utc != parent.start_utc
            or folds[-1].end_utc != parent.end_utc
            or any(left.end_utc != right.start_utc for left, right in pairwise(folds))
        ):
            raise AlphaMaxRuntimeContractError(f"alpha_max_{domain}_fold_coverage_invalid")
    return MappingProxyType(windows)


def preflight_alpha_max_runtime_contract(
    config_path: str | os.PathLike[str],
) -> AlphaMaxRuntimePreflight:
    """Read and validate the exact frozen config through one descriptor.

    The ambient environment gate runs before path resolution and before the
    descriptor helper is called.
    """
    reject_ambient_lq_environment()
    explicit_path = _require_exact_explicit_path(config_path)
    receipt, payload = read_artifact_bytes(explicit_path, artifact_id="alpha_max_config")
    if receipt.requested_path != receipt.canonical_path and not _is_proc_fd_anchored_path(
        Path(explicit_path)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_config_path_identity_mismatch")

    config = _strict_json_object(payload)
    runtime = _validate_runtime_contract(config)
    if receipt.sha256 != ALPHA_MAX_CONFIG_FILE_SHA256:
        raise AlphaMaxRuntimeContractError("alpha_max_config_file_hash_mismatch")

    cost_cells = tuple(
        AlphaMaxCostCell(
            nominal_one_way_bps=int(row["nominal_one_way_bps"]),
            slippage_rate=float(row["slippage_rate"]),
            expected_base_slippage_bps=int(row["expected_base_slippage_bps"]),
            selection_reference=bool(row["selection_reference"]),
        )
        for row in _EXPECTED_COST_CELLS
    )
    static_attributes = _freeze_json(runtime["static_attributes"])
    if not isinstance(static_attributes, Mapping):  # defensive, sealed above
        raise AlphaMaxRuntimeContractError("alpha_max_static_attributes_invalid")
    backtest_constructor = _freeze_json(runtime["backtest_constructor"])
    portfolio_constructor = _freeze_json(runtime["portfolio_strategy_constructor"])
    if not isinstance(backtest_constructor, Mapping) or not isinstance(
        portfolio_constructor, Mapping
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_constructor_contract_invalid")

    return AlphaMaxRuntimePreflight(
        config_receipt=receipt,
        config_bytes=payload,
        config_payload_sha256=ALPHA_MAX_CONFIG_PAYLOAD_SHA256,
        runtime_contract_sha256=ALPHA_MAX_RUNTIME_CONTRACT_SHA256,
        runtime_contract_bytes=_canonical_bytes(runtime),
        common_runtime_bytes=_canonical_bytes(runtime["static_attributes"]),
        attribute_allowlist=_EXPECTED_ATTRIBUTE_ALLOWLIST,
        static_attributes=static_attributes,
        cost_cells=cost_cells,
        phase_windows=_phase_windows(config),
        candidate_symbols=ALPHA_MAX_CANDIDATE_SYMBOLS,
        backtest_constructor=backtest_constructor,
        portfolio_strategy_constructor=portfolio_constructor,
        incumbent_resolution_bytes=_validate_incumbent_resolution_contract(config),
        incumbent_resolution_audit_sha256=ALPHA_MAX_INCUMBENT_RESOLUTION_AUDIT_SHA256,
    )


class AlphaMaxBacktestConfig:
    """Constructor-bound immutable uppercase runtime surface with read auditing."""

    __slots__ = (
        *_EXPECTED_ATTRIBUTE_ALLOWLIST,
        "_contract_sha256",
        "_read_audit",
        "_runtime_instance_sha256",
    )

    def __init__(
        self,
        *,
        attributes: Mapping[str, object],
        contract_sha256: str,
        construction_token: object,
    ) -> None:
        reject_ambient_lq_environment()
        if construction_token is not _CONFIG_CONSTRUCTION_TOKEN:
            raise AlphaMaxRuntimeContractError("alpha_max_config_constructor_private")
        if set(attributes) != _RUNTIME_ATTRIBUTE_SET:
            raise AlphaMaxRuntimeContractError("alpha_max_runtime_attribute_set_mismatch")
        if contract_sha256 != ALPHA_MAX_RUNTIME_CONTRACT_SHA256:
            raise AlphaMaxRuntimeContractError("alpha_max_runtime_contract_hash_mismatch")
        object.__setattr__(self, "_read_audit", [])
        object.__setattr__(self, "_contract_sha256", contract_sha256)
        for name in _EXPECTED_ATTRIBUTE_ALLOWLIST:
            object.__setattr__(self, name, attributes[name])
        instance_bytes = _canonical_bytes(
            {name: _thaw_json(attributes[name]) for name in _EXPECTED_ATTRIBUTE_ALLOWLIST}
        )
        object.__setattr__(self, "_runtime_instance_sha256", _sha256(instance_bytes))

    def __getattribute__(self, name: str) -> object:
        if name in _INTERNAL_CONFIG_FIELDS:
            raise AttributeError(name)
        if name in _RUNTIME_ATTRIBUTE_SET:
            audit = object.__getattribute__(self, "_read_audit")
            audit.append(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> object:
        if name in _INTERNAL_CONFIG_FIELDS or name.startswith("__"):
            raise AttributeError(name)
        raise UnfrozenRuntimeFieldError(f"unfrozen_runtime_field:{name}")

    def __setattr__(self, name: str, value: object) -> None:
        del value
        raise FrozenRuntimeMutationError(f"frozen_runtime_field:{name}")

    def __delattr__(self, name: str) -> None:
        raise FrozenRuntimeMutationError(f"frozen_runtime_field:{name}")

    @property
    def runtime_contract_sha256(self) -> str:
        """Return the sealed source-contract digest without adding an engine read."""
        return object.__getattribute__(self, "_contract_sha256")

    @property
    def runtime_instance_sha256(self) -> str:
        """Return the digest of this fully derived runtime instance."""
        return object.__getattribute__(self, "_runtime_instance_sha256")

    @property
    def runtime_read_audit(self) -> tuple[str, ...]:
        """Return the deterministic ordered sequence of allowlisted reads."""
        return tuple(object.__getattribute__(self, "_read_audit"))

    @property
    def runtime_read_audit_sha256(self) -> str:
        """Return a deterministic digest of the ordered read sequence."""
        return _sha256(_canonical_bytes(list(self.runtime_read_audit)))

    def runtime_attribute_snapshot(self) -> Mapping[str, object]:
        """Expose immutable values without manufacturing engine-read events."""
        return MappingProxyType(
            {name: object.__getattribute__(self, name) for name in _EXPECTED_ATTRIBUTE_ALLOWLIST}
        )

    def runtime_attribute_bytes(self) -> bytes:
        """Return canonical bytes for every constructor-bound runtime attribute."""
        snapshot = self.runtime_attribute_snapshot()
        return _canonical_bytes({name: _thaw_json(value) for name, value in snapshot.items()})


register_alpha_max_backtest_config_type(AlphaMaxBacktestConfig)


def alpha_max_common_rng_seed_payload(split_or_fold_id: str, nominal_cost_bps: int) -> bytes:
    """Build the exact Revision 5.15 common-random-number seed payload."""
    if type(split_or_fold_id) is not str or not split_or_fold_id:
        raise AlphaMaxRuntimeContractError("alpha_max_seed_phase_id_invalid")
    if type(nominal_cost_bps) is not int or nominal_cost_bps not in ALPHA_MAX_COST_CELL_BPS:
        raise AlphaMaxRuntimeContractError("alpha_max_nominal_cost_cell_invalid")
    return (
        b"alpha_max_20260710\0"
        + split_or_fold_id.encode("utf-8")
        + b"\0"
        + str(nominal_cost_bps).encode("ascii")
    )


def alpha_max_common_rng_seed(split_or_fold_id: str, nominal_cost_bps: int) -> int:
    """Derive the exact non-zero 31-bit seed for one phase and cost cell."""
    payload = alpha_max_common_rng_seed_payload(split_or_fold_id, nominal_cost_bps)
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647
    return seed or 1


def _alpha_max_receipt_path_identity_valid(receipt: ArtifactReadReceipt) -> bool:
    if receipt.requested_path == receipt.canonical_path:
        return True
    requested = Path(receipt.requested_path)
    if not _is_proc_fd_anchored_path(requested):
        return False
    try:
        return str(requested.resolve(strict=True)) == receipt.canonical_path
    except OSError:
        return False


def _validate_preflight(preflight: AlphaMaxRuntimePreflight) -> None:
    if type(preflight) is not AlphaMaxRuntimePreflight:
        raise TypeError("alpha_max_runtime_preflight_required")
    if (
        preflight.runtime_contract_sha256 != ALPHA_MAX_RUNTIME_CONTRACT_SHA256
        or preflight.attribute_allowlist != _EXPECTED_ATTRIBUTE_ALLOWLIST
        or preflight.common_runtime_bytes != _canonical_bytes(_EXPECTED_STATIC_ATTRIBUTES)
        or _sha256(preflight.config_bytes) != ALPHA_MAX_CONFIG_FILE_SHA256
        or len(preflight.config_bytes) != preflight.config_receipt.byte_count
        or preflight.config_receipt.artifact_id != "alpha_max_config"
        or not _alpha_max_receipt_path_identity_valid(preflight.config_receipt)
        or preflight.config_receipt.sha256 != ALPHA_MAX_CONFIG_FILE_SHA256
        or preflight.config_receipt.pre_fstat_identity
        != preflight.config_receipt.post_fstat_identity
        or _sha256(preflight.incumbent_resolution_bytes)
        != _ALPHA_MAX_INCUMBENT_RESOLUTION_CANONICAL_SHA256
        or preflight.incumbent_resolution_audit_sha256
        != ALPHA_MAX_INCUMBENT_RESOLUTION_AUDIT_SHA256
        or preflight.candidate_symbols != ALPHA_MAX_CANDIDATE_SYMBOLS
        or preflight.backtest_constructor != _freeze_json(_EXPECTED_BACKTEST_CONSTRUCTOR)
        or preflight.portfolio_strategy_constructor
        != _freeze_json(_EXPECTED_PORTFOLIO_STRATEGY_CONSTRUCTOR)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_runtime_preflight_invalid")


def _validate_admitted_symbols(
    preflight: AlphaMaxRuntimePreflight,
    admitted_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    if type(admitted_symbols) is not tuple:
        raise TypeError("alpha_max_admitted_symbols_must_be_frozen_tuple")
    if not 5 <= len(admitted_symbols) <= 10:
        raise AlphaMaxRuntimeContractError("alpha_max_admitted_symbol_count_invalid")
    if len(set(admitted_symbols)) != len(admitted_symbols):
        raise AlphaMaxRuntimeContractError("alpha_max_admitted_symbols_duplicate")
    candidate_order = tuple(
        symbol for symbol in preflight.candidate_symbols if symbol in admitted_symbols
    )
    if admitted_symbols != candidate_order:
        raise AlphaMaxRuntimeContractError("alpha_max_admitted_symbols_order_or_membership_invalid")
    return admitted_symbols


def _build_alpha_max_backtest_config(
    preflight: AlphaMaxRuntimePreflight,
    *,
    phase_id: str,
    admitted_symbols: tuple[str, ...],
    nominal_cost_bps: int,
) -> AlphaMaxBacktestConfig:
    _validate_preflight(preflight)
    symbols = _validate_admitted_symbols(preflight, admitted_symbols)
    if type(phase_id) is not str or phase_id not in preflight.phase_windows:
        raise AlphaMaxRuntimeContractError(f"alpha_max_phase_window_unknown:{phase_id}")
    if type(nominal_cost_bps) is not int:
        raise AlphaMaxRuntimeContractError("alpha_max_nominal_cost_cell_invalid")
    cost_by_bps = {cell.nominal_one_way_bps: cell for cell in preflight.cost_cells}
    try:
        cost_cell = cost_by_bps[nominal_cost_bps]
    except KeyError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_nominal_cost_cell_invalid") from exc
    window = preflight.phase_windows[phase_id]

    attributes = dict(preflight.static_attributes)
    attributes.update(
        {
            "END_DATE": window.end_utc,
            "RANDOM_SEED": alpha_max_common_rng_seed(phase_id, nominal_cost_bps),
            "SLIPPAGE_RATE": cost_cell.slippage_rate,
            "START_DATE": window.start_utc,
            "SYMBOLS": symbols,
        }
    )
    return AlphaMaxBacktestConfig(
        attributes=attributes,
        contract_sha256=preflight.runtime_contract_sha256,
        construction_token=_CONFIG_CONSTRUCTION_TOKEN,
    )


def build_alpha_max_backtest_config(
    preflight: AlphaMaxRuntimePreflight,
    *,
    phase_id: str,
    admitted_symbols: tuple[str, ...],
    nominal_cost_bps: int,
) -> AlphaMaxBacktestConfig:
    """Construct one immutable runtime instance without a default/config merge."""
    reject_ambient_lq_environment()
    return _build_alpha_max_backtest_config(
        preflight,
        phase_id=phase_id,
        admitted_symbols=admitted_symbols,
        nominal_cost_bps=nominal_cost_bps,
    )


def validate_alpha_max_cost_cell_config_matrix(
    configs: tuple[AlphaMaxBacktestConfig, ...],
) -> None:
    """Validate four exact cost configs and byte-identical non-cost runtime state.

    ``RANDOM_SEED`` is the sole additional derived difference required by the
    sealed per-cost seed schedule; it is not a mutable cost-cell override.
    """
    if type(configs) is not tuple or len(configs) != 4:
        raise AlphaMaxRuntimeContractError("alpha_max_cost_config_matrix_invalid")
    snapshots = [config.runtime_attribute_snapshot() for config in configs]
    slippages = tuple(snapshot["SLIPPAGE_RATE"] for snapshot in snapshots)
    if slippages != (0.0005, 0.001, 0.0015, 0.0025):
        raise AlphaMaxRuntimeContractError("alpha_max_cost_config_slippage_mismatch")
    symbols = snapshots[0]["SYMBOLS"]
    if any(snapshot["SYMBOLS"] is not symbols for snapshot in snapshots[1:]):
        raise AlphaMaxRuntimeContractError("alpha_max_cost_config_symbol_identity_mismatch")
    excluded = {"RANDOM_SEED", "SLIPPAGE_RATE"}
    common_bytes = tuple(
        _canonical_bytes(
            {name: _thaw_json(value) for name, value in snapshot.items() if name not in excluded}
        )
        for snapshot in snapshots
    )
    if len(set(common_bytes)) != 1:
        raise AlphaMaxRuntimeContractError("alpha_max_cost_config_common_bytes_mismatch")
    if any(
        config.runtime_contract_sha256 != ALPHA_MAX_RUNTIME_CONTRACT_SHA256 for config in configs
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_cost_config_contract_hash_mismatch")


def build_alpha_max_cost_cell_configs(
    preflight: AlphaMaxRuntimePreflight,
    *,
    phase_id: str,
    admitted_symbols: tuple[str, ...],
) -> tuple[AlphaMaxBacktestConfig, ...]:
    """Construct and cross-check the exact four-cell runtime matrix."""
    reject_ambient_lq_environment()
    configs = tuple(
        _build_alpha_max_backtest_config(
            preflight,
            phase_id=phase_id,
            admitted_symbols=admitted_symbols,
            nominal_cost_bps=nominal_cost_bps,
        )
        for nominal_cost_bps in ALPHA_MAX_COST_CELL_BPS
    )
    validate_alpha_max_cost_cell_config_matrix(configs)
    return configs


def build_alpha_max_engine_constructor_plan(
    preflight: AlphaMaxRuntimePreflight,
    *,
    config: AlphaMaxBacktestConfig,
    feature_lookup: object,
    funding_boundary_resolver: object,
    fill_application_attribution_sink: object,
    full_event_equity_sink: object | None = None,
    reporting_sampling_timeframe: str | None = None,
) -> AlphaMaxEngineConstructorPlan:
    """Bind explicit phase-owned identities without constructing ``Backtest``."""
    reject_ambient_lq_environment()
    _validate_preflight(preflight)
    if type(config) is not AlphaMaxBacktestConfig:
        raise TypeError("alpha_max_backtest_config_required")
    if config.runtime_contract_sha256 != preflight.runtime_contract_sha256:
        raise AlphaMaxRuntimeContractError("alpha_max_config_preflight_contract_mismatch")
    if feature_lookup is None:
        raise AlphaMaxRuntimeContractError("alpha_max_feature_lookup_required")
    if funding_boundary_resolver is None:
        raise AlphaMaxRuntimeContractError("alpha_max_funding_boundary_resolver_required")
    if fill_application_attribution_sink is None:
        raise AlphaMaxRuntimeContractError("alpha_max_fill_application_sink_required")
    if full_event_equity_sink is not None and not callable(full_event_equity_sink):
        raise AlphaMaxRuntimeContractError("alpha_max_full_event_equity_sink_invalid")
    if reporting_sampling_timeframe not in {None, "4h"}:
        raise AlphaMaxRuntimeContractError("alpha_max_reporting_sampling_timeframe_invalid")

    portfolio_kwargs: dict[str, object] = {
        "fill_application_attribution_sink": fill_application_attribution_sink,
        "funding_boundary_resolver": funding_boundary_resolver,
    }
    if full_event_equity_sink is not None:
        portfolio_kwargs["full_event_equity_sink"] = full_event_equity_sink
    if reporting_sampling_timeframe is not None:
        portfolio_kwargs["reporting_sampling_timeframe"] = reporting_sampling_timeframe

    return AlphaMaxEngineConstructorPlan(
        config=config,
        strategy_timeframe="1s",
        warmup_bars=0,
        record_history=True,
        track_metrics=True,
        record_trades=True,
        strict_data_handler_construction=True,
        data_handler_kwargs=MappingProxyType(
            {
                "backtest_poll_seconds": 1,
                "backtest_window_seconds": 1,
                "feature_db_path": None,
                "feature_exchange": "binance",
                "feature_lookup": feature_lookup,
                "market_window_parity_v2_enabled": True,
            }
        ),
        portfolio_kwargs=MappingProxyType(portfolio_kwargs),
        execution_handler_kwargs=MappingProxyType({"record_cost_attribution": True}),
    )


_ALPHA_MAX_NATIVE_TIMEFRAME_BY_CLASS: Final[dict[str, str]] = {
    "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy": "1d",
    "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy": "1d",
    "ResearchOnlyFourHourFundingHarvestCarryStrategy": "4h",
}
_ALPHA_MAX_MANIFEST_PHASES: Final[frozenset[str]] = frozenset(
    {"validation_train_fit", "prelock_final_refit"}
)
_ALPHA_MAX_FALSE_MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "allow_real_money",
    "ready_for_real",
    "real_money_execution",
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


def _activation_identity(path: Path, *, expected_directory: bool) -> AlphaMaxAncestorIdentity:
    try:
        status = path.lstat()
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch") from exc
    if stat.S_ISLNK(status.st_mode):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if expected_directory:
        if not stat.S_ISDIR(status.st_mode):
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    elif not stat.S_ISREG(status.st_mode) or int(status.st_nlink) != 1:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if int(status.st_uid) != os.geteuid():
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    return AlphaMaxAncestorIdentity(
        path=str(path),
        device=int(status.st_dev),
        inode=int(status.st_ino),
        file_type=int(stat.S_IFMT(status.st_mode)),
        link_count=int(status.st_nlink),
        owner_uid=int(status.st_uid),
        size=int(status.st_size),
        mtime_ns=int(status.st_mtime_ns),
        ctime_ns=int(status.st_ctime_ns),
    )


def _activation_paths(
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_path: str | os.PathLike[str],
) -> tuple[Path, Path, Path, Path]:
    root_raw = os.fspath(output_root)
    manifest_raw = os.fspath(manifest_path)
    if (
        phase not in _ALPHA_MAX_MANIFEST_PHASES
        or not root_raw
        or not manifest_raw
        or not os.path.isabs(root_raw)
        or not os.path.isabs(manifest_raw)
        or os.path.abspath(root_raw) != root_raw
        or os.path.abspath(manifest_raw) != manifest_raw
    ):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    root = Path(root_raw)
    manifests = root / "manifests"
    phase_path = manifests / phase
    target = Path(manifest_raw)
    expected_parent = phase_path
    if (
        target.parent != expected_parent
        and _is_proc_fd_anchored_path(expected_parent)
        and target.parent == expected_parent.resolve(strict=True)
    ):
        target = expected_parent / target.name
    if (
        target.parent != expected_parent
        or target.name in {"", ".", ".."}
        or target.suffix != ".json"
    ):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    try:
        if any(
            str(path.resolve(strict=True)) != str(path) and not _is_proc_fd_anchored_path(path)
            for path in (root, manifests, phase_path, target)
        ):
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch") from exc
    return root, manifests, phase_path, target


def _snapshot_activation_paths(paths: Sequence[Path]) -> tuple[AlphaMaxAncestorIdentity, ...]:
    return tuple(
        _activation_identity(path, expected_directory=index < len(paths) - 1)
        for index, path in enumerate(paths)
    )


def _finite_float(value: object, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0.0) or (nonnegative and parsed < 0.0):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    return parsed


def _validate_manifest_bytes(
    payload_bytes: bytes,
    *,
    preflight: AlphaMaxRuntimePreflight,
    admitted_symbols: tuple[str, ...],
    manifest_path: str,
    config_receipt: ArtifactReadReceipt,
    config_payload: Mapping[str, object],
    phase: str,
) -> AlphaMaxExpectedDefinition:
    payload = _strict_json_object(payload_bytes)
    if payload_bytes != _canonical_bytes(payload) + b"\n":
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if set(payload) != ALPHA_MAX_MANIFEST_TOP_LEVEL_KEYS:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if payload.get("artifact_kind") != "alpha_max_engine_portfolio_manifest.v1":
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if tuple(payload.get("candidate_symbols") or ()) != preflight.candidate_symbols:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if tuple(payload.get("admitted_symbols") or ()) != admitted_symbols:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if any(payload.get(key) is not False for key in _ALPHA_MAX_FALSE_MANIFEST_KEYS):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    admission_sha = str(payload.get("admission_manifest_sha256") or "")
    if len(admission_sha) != 64 or any(ch not in "0123456789abcdef" for ch in admission_sha):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    gross_cap = _finite_float(payload.get("gross_cap"), positive=True)
    if gross_cap > 2.25:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    cash_weight = _finite_float(payload.get("cash_weight"), nonnegative=True)
    if cash_weight > 1.0:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    allocation_method = str(payload.get("allocation_method") or "")
    if allocation_method not in {
        "single_component",
        "equal_weight",
        "equal_risk",
        "shrunk_hrp",
    }:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")

    registry = config_payload.get("current_trial_registry")
    if not isinstance(registry, Mapping) or type(registry.get("nodes")) is not list:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    row_id = Path(manifest_path).stem
    registry_rows = {
        row.get("row_id"): row
        for row in registry["nodes"]
        if type(row) is dict and type(row.get("row_id")) is str
    }
    row = registry_rows.get(row_id)
    if type(row) is not dict or row_id not in _ALPHA_MAX_RESOLVABLE_ROWS:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    allocation = row.get("allocation")
    gross_rule = row.get("gross")
    if (
        type(allocation) is not dict
        or type(gross_rule) is not dict
        or allocation.get("method") != allocation_method
    ):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    expected_members = tuple(row.get("members") or (row_id,))
    if not expected_members or tuple(sorted(expected_members)) != expected_members:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    component_rows = {
        candidate.get("row_id"): candidate
        for candidate in registry["nodes"]
        if type(candidate) is dict and str(candidate.get("row_id") or "").startswith("component_")
    }
    if any(member not in component_rows for member in expected_members):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    gross_method = gross_rule.get("method")
    if gross_method == "fixed":
        if gross_cap != _finite_float(gross_rule.get("value"), positive=True):
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    elif gross_method == "validation_mdd_target":
        clip_min = _finite_float(gross_rule.get("clip_min"), positive=True)
        clip_max = _finite_float(gross_rule.get("clip_max"), positive=True)
        if not clip_min <= gross_cap <= clip_max:
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    else:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    cap = _finite_float(allocation.get("per_component_cap"), positive=True)

    use_train_validation = phase == "prelock_final_refit" and allocation_method in {
        "equal_risk",
        "shrunk_hrp",
    }
    selection_inputs = ["train", "validation"] if use_train_validation else ["train"]
    correlation_source = (
        "alpha_max_train_validation_daily_net_returns"
        if use_train_validation
        else "alpha_max_train_daily_net_returns"
    )
    expected_optimizer = {"selection_inputs": selection_inputs}
    expected_correlation = {
        "selection_inputs": selection_inputs,
        "ready": True,
        "source": correlation_source,
    }
    if (
        payload.get("optimizer_provenance") != expected_optimizer
        or payload.get("correlation_input_provenance") != expected_correlation
    ):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")

    source_artifacts = payload.get("source_artifacts")
    if type(source_artifacts) is not list or len(source_artifacts) != 1:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    source = source_artifacts[0]
    if type(source) is not dict or set(source) != {
        "id",
        "max_age_hours",
        "path",
        "portfolio_ready",
        "ready",
        "sha256",
    }:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if (
        source.get("id") != "alpha_max_config"
        or source.get("path") != config_receipt.canonical_path
        or source.get("sha256") != config_receipt.sha256
        or source.get("ready") is not True
        or source.get("portfolio_ready") is not True
        or source.get("max_age_hours") != 876000
    ):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")

    raw_children = payload.get("children")
    if type(raw_children) is not list or not raw_children:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    expected_components: list[AlphaMaxExpectedComponent] = []
    native_timeframes: set[str] = set()
    seen: set[str] = set()
    for child in raw_children:
        if type(child) is not dict or set(child) != ALPHA_MAX_MANIFEST_CHILD_KEYS:
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
        child_id = str(child.get("candidate_id") or "")
        strategy_class = str(child.get("strategy_class") or "")
        if (
            not child_id
            or child_id in seen
            or child.get("name") != child_id
            or tuple(child.get("candidate_symbols") or ()) != preflight.candidate_symbols
            or tuple(child.get("symbols") or ()) != admitted_symbols
            or child.get("source_artifact_id") != "alpha_max_config"
            or child.get("ready") is not True
            or child.get("portfolio_ready") is not True
            or child.get("no_current_fold_oos_provenance") is not True
            or any(child.get(key) is not False for key in _ALPHA_MAX_FALSE_MANIFEST_KEYS)
            or strategy_class not in _ALPHA_MAX_NATIVE_TIMEFRAME_BY_CLASS
            or type(child.get("params")) is not dict
            or child.get("train_validation_optimizer_provenance") is not True
            or child.get("lagged_completed_shadow_optimizer_provenance") is not False
            or child.get("optimizer_provenance") != expected_optimizer
            or child.get("correlation_input_provenance") != expected_correlation
        ):
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
        seen.add(child_id)
        weight = _finite_float(child.get("weight"), positive=True)
        leaf_gross = _finite_float(child.get("leaf_gross"), positive=True)
        leaf_cap = _finite_float(child.get("leaf_gross_cap"), positive=True)
        netting_cap = _finite_float(child.get("netting_group_gross_cap"), positive=True)
        if (
            weight != leaf_gross
            or leaf_cap != cap * gross_cap
            or leaf_gross > leaf_cap + 1e-12
            or child.get("netting_group") != child_id
            or netting_cap != leaf_cap
        ):
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
        component_row = component_rows.get(child_id)
        if (
            type(component_row) is not dict
            or component_row.get("implementation") != strategy_class
            or _canonical_bytes(component_row.get("params"))
            != _canonical_bytes(child.get("params"))
        ):
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
        native_timeframes.add(_ALPHA_MAX_NATIVE_TIMEFRAME_BY_CLASS[strategy_class])
        expected_components.append(
            AlphaMaxExpectedComponent(
                component_id=child_id,
                strategy_class=strategy_class,
                symbols=admitted_symbols,
                params_bytes=_canonical_bytes(child["params"]),
                weight=weight,
                source_artifact_id="alpha_max_config",
            )
        )
    if tuple(component.component_id for component in expected_components) != tuple(sorted(seen)):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if tuple(component.component_id for component in expected_components) != expected_members:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    exact_gross_total = math.fsum(component.weight for component in expected_components)
    if abs(exact_gross_total - gross_cap) >= 1e-9 or cash_weight != max(
        0.0, 1.0 - exact_gross_total
    ):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    fixed_weights = allocation.get("fixed_weights")
    if fixed_weights is not None:
        if type(fixed_weights) is not dict or set(fixed_weights) != set(expected_members):
            raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
        for component in expected_components:
            expected_weight = (
                _round(float(fixed_weights[component.component_id]), ndigits=10) * gross_cap
            )
            if component.weight != expected_weight:
                raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    return AlphaMaxExpectedDefinition(
        portfolio_mode=f"manifest:{manifest_path}",
        artifact_kind="alpha_max_engine_portfolio_manifest.v1",
        candidate_symbols=preflight.candidate_symbols,
        admitted_symbols=admitted_symbols,
        admission_manifest_sha256=admission_sha,
        gross_cap=gross_cap,
        cash_weight=cash_weight,
        allocation_method=allocation_method,
        source_path=config_receipt.canonical_path,
        source_sha256=config_receipt.sha256,
        components=tuple(expected_components),
        native_timeframes=tuple(sorted(native_timeframes)),
    )


def seal_alpha_max_manifest_activation(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_path: str | os.PathLike[str],
    admitted_symbols: tuple[str, ...],
) -> AlphaMaxArtifactSeal:
    """Seal exact manifest/config bytes after lexical ancestor validation."""
    reject_ambient_lq_environment()
    _validate_preflight(preflight)
    admitted = _validate_admitted_symbols(preflight, admitted_symbols)
    paths = _activation_paths(output_root, phase, manifest_path)
    before = _snapshot_activation_paths(paths)
    try:
        manifest_receipt, manifest_bytes = read_artifact_bytes(
            paths[-1], artifact_id="artifact_portfolio_manifest"
        )
        config_path = preflight.config_receipt.canonical_path
        config_receipt, config_bytes = read_artifact_bytes(
            config_path, artifact_id="source:alpha_max_config"
        )
    except (OSError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch") from exc
    after = _snapshot_activation_paths(paths)
    if before != after:
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if (
        manifest_receipt.requested_path != str(paths[-1])
        or manifest_receipt.canonical_path != str(paths[-1].resolve(strict=True))
        or config_receipt.requested_path != preflight.config_receipt.canonical_path
        or config_receipt.canonical_path != preflight.config_receipt.canonical_path
        or config_receipt.sha256 != preflight.config_receipt.sha256
    ):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    # Both documents are parsed only from the descriptor-returned bytes.  The
    # final pre-event seal repeats the same descriptor-bound check.
    config_payload = _strict_json_object(config_bytes)
    _validate_runtime_contract(config_payload)
    expected = _validate_manifest_bytes(
        manifest_bytes,
        preflight=preflight,
        admitted_symbols=admitted,
        manifest_path=str(paths[-1]),
        config_receipt=config_receipt,
        config_payload=config_payload,
        phase=phase,
    )
    return AlphaMaxArtifactSeal(
        output_root=str(paths[0]),
        phase=phase,
        manifest_path=str(paths[-1]),
        ancestor_identities=after,
        manifest_receipt=manifest_receipt,
        config_receipt=config_receipt,
        manifest_bytes=manifest_bytes,
        config_bytes=config_bytes,
        expected_definition=expected,
    )


def _bound_method_matches(candidate: object, owner: object, function: object) -> bool:
    return (
        callable(candidate)
        and getattr(candidate, "__self__", None) is owner
        and getattr(candidate, "__func__", None) is function
    )


def _alpha_max_current_root_id(phase_id: str) -> str:
    if phase_id.startswith("validation_w"):
        return "validation"
    if phase_id.startswith("historical_20"):
        return "historical_exposed_evaluation"
    if phase_id in {
        "warmup",
        "train",
        "purge",
        "validation",
        "embargo",
        "historical_exposed_evaluation",
    }:
        return phase_id
    raise AlphaMaxRuntimeContractError(f"alpha_max_phase_root_unknown:{phase_id}")


def _alpha_max_fold_ids(domain: str) -> tuple[str, ...]:
    try:
        return _ALPHA_MAX_DOMAIN_FOLD_IDS[domain]
    except KeyError as exc:
        raise AlphaMaxRuntimeContractError(f"alpha_max_domain_unknown:{domain}") from exc


def _alpha_max_physical_fold_schedule(
    domain: str,
) -> tuple[tuple[str, int, str], ...]:
    """Return the immutable row/cost/fold schedule executed by a full matrix."""
    return tuple(
        (row_id, nominal, fold_id)
        for row_id in _ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
        for fold_id in _alpha_max_fold_ids(domain)
    )


def _validate_alpha_max_physical_fold_schedule(
    observed: tuple[tuple[str, int, str], ...],
    *,
    domain: str,
) -> None:
    if type(observed) is not tuple or observed != _alpha_max_physical_fold_schedule(domain):
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_physical_fold_cardinality_mismatch")


def _alpha_max_capsule_predecessor(phase_id: str) -> str:
    for fold_ids, initial in (
        (_ALPHA_MAX_VALIDATION_FOLD_IDS, "purge"),
        (_ALPHA_MAX_HISTORICAL_FOLD_IDS, "embargo"),
    ):
        if phase_id in fold_ids:
            index = fold_ids.index(phase_id)
            return initial if index == 0 else fold_ids[index - 1]
    current = _alpha_max_current_root_id(phase_id)
    predecessor = {
        "train": "warmup",
        "purge": "train",
        "validation": "purge",
        "embargo": "validation",
        "historical_exposed_evaluation": "embargo",
    }.get(current)
    if predecessor is None:
        raise AlphaMaxRuntimeContractError("alpha_max_scored_warmup_forbidden")
    return predecessor


def _validate_alpha_max_indicator_capsule(
    capsule: AlphaMaxIndicatorCapsule,
    *,
    seal: AlphaMaxArtifactSeal,
    expected_phase_id: str | None = None,
) -> dict[str, object]:
    if type(capsule) is not AlphaMaxIndicatorCapsule:
        raise TypeError("alpha_max_indicator_capsule_identity_invalid")
    count_values = (
        capsule.windows_processed,
        capsule.discarded_signal_count,
        capsule.market_event_count,
        capsule.funding_event_count,
        capsule.order_event_count,
        capsule.fill_event_count,
        capsule.trade_count,
    )
    hashes = (
        capsule.manifest_sha256,
        capsule.capsule_sha256,
        capsule.native_finalization_sha256,
    )
    if (
        type(capsule.portfolio_mode) is not str
        or not capsule.portfolio_mode
        or type(capsule.phase_id) is not str
        or not capsule.phase_id
        or any(type(value) is not int or value < 0 for value in count_values)
        or capsule.windows_processed == 0
        or any(value != 0 for value in count_values[2:])
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_invalid")
    if (
        capsule.portfolio_mode != seal.expected_definition.portfolio_mode
        or capsule.manifest_sha256 != seal.manifest_receipt.sha256
        or (expected_phase_id is not None and capsule.phase_id != expected_phase_id)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_scope_mismatch")
    restored = _thaw_json(capsule.capsule)
    if type(restored) is not dict:
        raise TypeError("alpha_max_indicator_capsule_invalid")
    retained_sha = restored.get("sha256")
    scope = {key: value for key, value in restored.items() if key != "sha256"}
    if (
        type(retained_sha) is not str
        or retained_sha != capsule.capsule_sha256
        or retained_sha != _sha256(_canonical_bytes(scope))
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_hash_mismatch")
    finalized = _thaw_json(capsule.finalized_children)
    expected_components = {
        component.component_id: component for component in seal.expected_definition.components
    }
    if type(finalized) is not dict or set(finalized) != set(expected_components):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_finalization_invalid")
    for component_id, component in expected_components.items():
        coverage = finalized[component_id]
        if (
            type(coverage) is not dict
            or set(coverage)
            != _ALPHA_MAX_NATIVE_SNAPSHOT_KEYS
            | {"finalization_completed_native_keys", "finalization_barrier_keys"}
            or not isinstance(coverage["finalization_completed_native_keys"], list)
            or not isinstance(coverage["finalization_barrier_keys"], list)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_finalization_invalid")
        snapshot = {key: coverage[key] for key in _ALPHA_MAX_NATIVE_SNAPSHOT_KEYS}
        _alpha_max_assert_native_coverage_binding(
            snapshot,
            component=component,
            admitted_symbols=seal.expected_definition.admitted_symbols,
        )
        completed = _alpha_max_native_completed_key_set(snapshot)
        final_completed = _alpha_max_native_completed_key_set(
            {"completed_native_keys": coverage["finalization_completed_native_keys"]}
        )
        barrier_closed = _alpha_max_native_barrier_key_set(
            snapshot,
            field="barrier_closed_keys",
        )
        final_barriers = _alpha_max_native_barrier_key_set(
            coverage,
            field="finalization_barrier_keys",
        )
        if not final_completed.issubset(completed) or not final_barriers.issubset(barrier_closed):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_finalization_invalid")
    return restored


def _validate_alpha_max_root_seals(
    *,
    raw_root: str | os.PathLike[str],
    phase_id: str,
    ordered_lookup: AlphaMaxOrderedFundingLookup,
    raw_root_seals: tuple[AlphaMaxRootSeal, ...],
    feature_root_seals: tuple[AlphaMaxRootSeal, ...],
    required: bool,
    repeat_hash: bool,
) -> None:
    if not raw_root_seals and not feature_root_seals and not required:
        return
    if type(raw_root_seals) is not tuple or any(
        type(value) is not AlphaMaxRootSeal for value in raw_root_seals
    ):
        raise TypeError("alpha_max_raw_root_seals_identity_invalid")
    if type(feature_root_seals) is not tuple or any(
        type(value) is not AlphaMaxRootSeal for value in feature_root_seals
    ):
        raise TypeError("alpha_max_feature_root_seals_identity_invalid")
    raw_path = _require_exact_explicit_path(raw_root)
    expected_feature_root_ids = _alpha_max_expected_root_sequence(phase_id)
    expected_raw_root_ids = (_alpha_max_current_root_id(phase_id),)
    if (
        tuple(value.root_id for value in raw_root_seals) != expected_raw_root_ids
        or any(value.root_kind != "raw" for value in raw_root_seals)
        or any(value.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS for value in raw_root_seals)
        or any(value.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS for value in feature_root_seals)
        or raw_root_seals[-1].path != raw_path
        or len(feature_root_seals) != len(ordered_lookup.root_specs)
        or tuple(value.root_id for value in feature_root_seals) != expected_feature_root_ids
        or ordered_lookup.root_seals != feature_root_seals
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_root_seal_scope_mismatch")
    for retained, spec in zip(feature_root_seals, ordered_lookup.root_specs, strict=True):
        if (
            retained.root_kind != "feature"
            or retained.root_id != spec.root_id
            or retained.path != spec.path
            or retained.exchange != spec.exchange
            or retained.inventory_sha256 != spec.inventory_sha256
            or retained.content_sha256 != spec.content_sha256
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_root_seal_scope_mismatch")
    if not repeat_hash:
        return
    retained_roots = (*raw_root_seals, *feature_root_seals)

    def reseal(retained: AlphaMaxRootSeal) -> AlphaMaxRootSeal:
        return seal_alpha_max_root_tree(
            retained.root_id,
            retained.root_kind,
            retained.path,
            exchange=retained.exchange,
            availability_start_by_symbol=retained.availability_start_by_symbol,
            availability_end_by_symbol=retained.availability_end_by_symbol,
        )

    if len(retained_roots) == 1:
        repeated = (reseal(retained_roots[0]),)
    else:
        with ThreadPoolExecutor(
            max_workers=min(_ALPHA_MAX_MAX_PARALLEL_WORKERS, len(retained_roots)),
            thread_name_prefix="alpha-max-root-seal",
        ) as executor:
            repeated = tuple(executor.map(reseal, retained_roots))
    repeated_raw = repeated[: len(raw_root_seals)]
    repeated_features = repeated[len(raw_root_seals) :]
    if repeated_raw != raw_root_seals or repeated_features != feature_root_seals:
        raise AlphaMaxRuntimeContractError("alpha_max_root_seal_changed")


def _activation_mismatch(exc: BaseException | None = None) -> AlphaMaxRuntimeContractError:
    error = AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    if exc is not None:
        error.__cause__ = exc
    return error


def _assert_definition_matches(
    strategy: ArtifactPortfolioModeStrategy,
    seal: AlphaMaxArtifactSeal,
) -> None:
    expected = seal.expected_definition
    definition = strategy.definition
    if (
        definition.portfolio_mode != expected.portfolio_mode
        or definition.cash_weight != expected.cash_weight
        or definition.source_artifacts.get("artifact_portfolio_manifest_path")
        != seal.manifest_receipt.canonical_path
        or definition.source_artifacts.get("manifest_source_artifact:alpha_max_config")
        != seal.config_receipt.canonical_path
        or "manifest_fail_closed_reason" in definition.source_artifacts
        or tuple(receipt.artifact_id for receipt in definition.artifact_read_receipts)
        != ("artifact_portfolio_manifest", "source:alpha_max_config")
        or definition.artifact_read_receipts != seal.consumer_receipts
        or len(definition.components) != len(expected.components)
    ):
        raise _activation_mismatch()

    for actual, retained in zip(definition.components, expected.components, strict=True):
        if (
            type(actual) is not PortfolioModeComponent
            or actual.component_id != retained.component_id
            or actual.label != retained.component_id
            or actual.strategy_class != retained.strategy_class
            or actual.symbols != retained.symbols
            or _canonical_bytes(actual.params) != retained.params_bytes
            or actual.weight != retained.weight
            or actual.source != f"{expected.portfolio_mode}:manifest:{retained.source_artifact_id}"
        ):
            raise _activation_mismatch()


def _assert_child_identities(
    strategy: ArtifactPortfolioModeStrategy,
    admitted_symbols: tuple[str, ...],
    expected: AlphaMaxExpectedDefinition,
) -> None:
    children = tuple(getattr(strategy, "_children", ()))
    if len(children) != len(expected.components):
        raise _activation_mismatch()
    for entry, retained in zip(children, expected.components, strict=True):
        if type(entry) is not tuple or len(entry) != 3:
            raise _activation_mismatch()
        component, child, child_queue = entry
        child_bars = getattr(child, "bars", None)
        if (
            type(component) is not PortfolioModeComponent
            or component.component_id != retained.component_id
            or component.strategy_class != retained.strategy_class
            or tuple(component.symbols) != admitted_symbols
            or tuple(getattr(child_bars, "symbol_list", ())) != admitted_symbols
            or tuple(getattr(child, "symbol_list", ())) != admitted_symbols
            or not callable(getattr(child_queue, "drain", None))
        ):
            raise _activation_mismatch()
        admitted_barrier = getattr(child, "_alpha_max_admitted_symbols", None)
        if admitted_barrier is not None and admitted_barrier != admitted_symbols:
            raise _activation_mismatch()
        # Carry uses feature points only while handling a context; retaining a
        # separate long-lived lookup would create an unsealed second capability.
        for name in ("_funding_lookup", "funding_lookup"):
            if getattr(child, name, None) is not None:
                raise _activation_mismatch()


def _alpha_max_handler_carry_rows(
    handler: HistoricParquetWindowedDataHandler,
    admitted_symbols: tuple[str, ...],
) -> tuple[
    tuple[tuple[str, tuple[tuple[object, ...], ...]], ...],
    tuple[tuple[str, tuple[int | None, ...]], ...],
]:
    rows: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
    timestamps: list[tuple[str, tuple[int | None, ...]]] = []
    for symbol in admitted_symbols:
        raw_rows = getattr(handler, "_window_rows", {}).get(symbol)
        raw_timestamps = getattr(handler, "_window_row_timestamps_ms", {}).get(symbol)
        if raw_rows is None or raw_timestamps is None or len(raw_rows) != len(raw_timestamps):
            raise AlphaMaxRuntimeContractError("alpha_max_daily_handler_state_invalid")
        rows.append((symbol, tuple(copy.deepcopy(tuple(raw_rows)))))
        timestamps.append((symbol, tuple(raw_timestamps)))
    return tuple(rows), tuple(timestamps)


def _capture_alpha_max_daily_carry(
    activation: AlphaMaxEngineActivation,
) -> _AlphaMaxDailyCarry:
    execution = activation.backtest.execution_handler
    handler_rows, handler_timestamps = _alpha_max_handler_carry_rows(
        activation.backtest.data_handler,
        activation.admitted_symbols,
    )
    return _AlphaMaxDailyCarry(
        strategy_state=copy.deepcopy(activation.backtest.strategy.get_state()),
        portfolio_state=copy.deepcopy(activation.backtest.portfolio.get_state()),
        execution_state=copy.deepcopy(execution.get_state()),
        engine_state=copy.deepcopy(activation.backtest.get_engine_state()),
        handler_rows=handler_rows,
        handler_timestamps_ms=handler_timestamps,
        funding_ledger=activation.funding_resolver.ledger,
    )


def _restore_alpha_max_daily_carry(
    activation: AlphaMaxEngineActivation,
    carry: _AlphaMaxDailyCarry,
) -> None:
    if type(carry) is not _AlphaMaxDailyCarry:
        raise TypeError("alpha_max_daily_carry_required")
    backtest = activation.backtest
    handler = backtest.data_handler
    rows_by_symbol = dict(carry.handler_rows)
    timestamps_by_symbol = dict(carry.handler_timestamps_ms)
    if tuple(rows_by_symbol) != activation.admitted_symbols or tuple(timestamps_by_symbol) != (
        activation.admitted_symbols
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_daily_handler_state_invalid")
    for symbol in activation.admitted_symbols:
        rows = handler._window_rows[symbol]
        timestamps = handler._window_row_timestamps_ms[symbol]
        if rows or timestamps:
            raise AlphaMaxRuntimeContractError("alpha_max_daily_handler_not_fresh")
        retained_rows = copy.deepcopy(rows_by_symbol[symbol])
        retained_timestamps = timestamps_by_symbol[symbol]
        if len(retained_rows) != len(retained_timestamps):
            raise AlphaMaxRuntimeContractError("alpha_max_daily_handler_state_invalid")
        rows.extend(retained_rows)
        timestamps.extend(retained_timestamps)
    backtest.strategy.set_state(copy.deepcopy(carry.strategy_state))
    backtest.portfolio.set_state(copy.deepcopy(carry.portfolio_state))
    backtest.execution_handler.set_state(copy.deepcopy(carry.execution_state))
    backtest.set_engine_state(copy.deepcopy(carry.engine_state))


def validate_alpha_max_engine_activation(
    activation: AlphaMaxEngineActivation,
    *,
    _expected_daily_carry: _AlphaMaxDailyCarry | None = None,
) -> None:
    """Repeat the descriptor seal and assert every constructor-bound identity.

    Callers invoke this once immediately after construction and once more as the
    final operation before ``Backtest._run_backtest``.  The function performs no
    event processing and deliberately collapses every disagreement to the one
    fail-closed activation error required by the experiment contract.
    """
    reject_ambient_lq_environment()
    if type(activation) is not AlphaMaxEngineActivation:
        raise TypeError("alpha_max_engine_activation_required")
    try:
        backtest = activation.backtest
        plan = activation.constructor_plan
        seal = activation.artifact_seal
        admitted = activation.admitted_symbols
        lookup = activation.ordered_lookup
        resolver = activation.funding_resolver
        collector = activation.attribution_collector
        equity_tracker = activation.full_event_equity_tracker
        capsule = activation.indicator_capsule
        raw_root_seals = activation.raw_root_seals
        feature_root_seals = activation.feature_root_seals

        if (
            type(backtest) is not Backtest
            or type(plan) is not AlphaMaxEngineConstructorPlan
            or type(seal) is not AlphaMaxArtifactSeal
            or type(admitted) is not tuple
            or type(lookup) is not AlphaMaxOrderedFundingLookup
            or type(resolver) is not AlphaMaxFundingBoundaryResolver
            or type(collector) is not AlphaMaxAttributionCollector
            or not isinstance(equity_tracker, AlphaMaxStreamingEquityTracker)
            or (capsule is not None and type(capsule) is not AlphaMaxIndicatorCapsule)
            or type(feature_root_seals) is not tuple
            or type(raw_root_seals) is not tuple
            or type(activation.phase_id) is not str
            or not activation.phase_id
            or type(activation.raw_root) is not str
            or not activation.raw_root
            or type(activation.repeat_root_hash_on_activation) is not bool
            or type(activation.chunk_start_utc) is not datetime
            or type(activation.chunk_end_utc) is not datetime
            or activation.chunk_start_utc.tzinfo != UTC
            or activation.chunk_end_utc.tzinfo != UTC
            or activation.chunk_end_utc - activation.chunk_start_utc != timedelta(days=1)
            or plan.strategy_timeframe != "1s"
            or plan.warmup_bars != 0
            or plan.record_history is not True
            or plan.track_metrics is not True
            or plan.record_trades is not True
            or plan.strict_data_handler_construction is not True
            or lookup.ordered_root_ids != _alpha_max_expected_root_sequence(activation.phase_id)
        ):
            raise _activation_mismatch()

        repeated = seal_alpha_max_manifest_activation(
            activation.preflight,
            output_root=seal.output_root,
            phase=seal.phase,
            manifest_path=seal.manifest_path,
            admitted_symbols=admitted,
        )
        if repeated != seal:
            raise _activation_mismatch()

        if (
            backtest.symbol_list is not admitted
            or backtest.csv_dir != activation.raw_root
            or backtest.config is not plan.config
            or backtest.data_handler_cls is not HistoricParquetWindowedDataHandler
            or backtest.strategy_cls is not ArtifactPortfolioModeStrategy
            or backtest.portfolio_cls is not Portfolio
            or backtest.execution_handler_cls is not SimulatedExecutionHandler
            or backtest.strict_data_handler_construction is not True
            or backtest.record_history is not True
            or backtest.track_metrics is not True
            or backtest.record_trades is not True
            or backtest.strategy_timeframe != "1s"
            or backtest.warmup_bars != 0
            or type(backtest.data_handler) is not HistoricParquetWindowedDataHandler
            or backtest.data_handler.symbol_list is not admitted
            or type(backtest.strategy) is not ArtifactPortfolioModeStrategy
            or type(backtest.portfolio) is not Portfolio
            or type(backtest.execution_handler) is not SimulatedExecutionHandler
            or backtest.portfolio.symbol_list is not backtest.data_handler.symbol_list
            or backtest.portfolio.bars is not backtest.data_handler
            or getattr(backtest.data_handler, "_feature_lookup", None) is not lookup
            or resolver.ordered_lookup is not lookup
            or resolver.admitted_symbols is not admitted
            or backtest.strategy.decision_cadence_seconds != 1
            or plan.config.DECISION_CADENCE_SECONDS != 1
            or plan.config.SYMBOLS is not admitted
            or backtest.strategy.required_timeframes != seal.expected_definition.native_timeframes
            or dict(activation.strategy_params)
            != {
                "portfolio_mode": seal.expected_definition.portfolio_mode,
                "decision_cadence_seconds": 1,
            }
            or backtest.execution_handler.record_cost_attribution is not True
            or backtest.portfolio.reporting_sampling_timeframe != "4h"
            or backtest.start_date != activation.chunk_start_utc
            or backtest.end_date != activation.chunk_end_utc
        ):
            raise _activation_mismatch()

        if (
            set(backtest.data_handler_kwargs) != set(plan.data_handler_kwargs)
            or any(
                backtest.data_handler_kwargs[key] is not value
                if key == "feature_lookup"
                else backtest.data_handler_kwargs[key] != value
                for key, value in plan.data_handler_kwargs.items()
            )
            or set(backtest.execution_handler_kwargs) != {"record_cost_attribution"}
            or backtest.execution_handler_kwargs.get("record_cost_attribution") is not True
            or set(backtest.portfolio_kwargs)
            != {
                "fill_application_attribution_sink",
                "full_event_equity_sink",
                "funding_boundary_resolver",
                "reporting_sampling_timeframe",
            }
            or backtest.portfolio_kwargs.get("funding_boundary_resolver") is not resolver
            or backtest.portfolio_kwargs.get("reporting_sampling_timeframe") != "4h"
        ):
            raise _activation_mismatch()

        application_sink = backtest.portfolio.fill_application_attribution_sink
        equity_sink = backtest.portfolio.full_event_equity_sink
        pricing_sink = backtest.execution_handler.pricing_attribution_sink
        raw_accessor = backtest.data_handler.get_latest_raw_point
        if (
            not _bound_method_matches(
                application_sink,
                collector,
                AlphaMaxAttributionCollector.record_application,
            )
            or not _bound_method_matches(
                pricing_sink,
                backtest.execution_handler,
                SimulatedExecutionHandler._capture_pricing_trace,
            )
            or not _bound_method_matches(
                raw_accessor,
                backtest.data_handler,
                HistoricParquetWindowedDataHandler.get_latest_raw_point,
            )
            or backtest.portfolio.funding_boundary_resolver is not resolver
            or not _bound_method_matches(
                backtest.portfolio_kwargs.get("fill_application_attribution_sink"),
                collector,
                AlphaMaxAttributionCollector.record_application,
            )
            or not any(
                _bound_method_matches(equity_sink, equity_tracker, function)
                for function in (
                    AlphaMaxStreamingEquityTracker.observe,
                    _AlphaMaxFoldEquityFanout.observe,
                )
            )
            or not any(
                _bound_method_matches(
                    backtest.portfolio_kwargs.get("full_event_equity_sink"),
                    equity_tracker,
                    function,
                )
                for function in (
                    AlphaMaxStreamingEquityTracker.observe,
                    _AlphaMaxFoldEquityFanout.observe,
                )
            )
        ):
            raise _activation_mismatch()
        # Bind the exact raw-accessor capability before the first replay event.
        # This performs no data lookup and binds no ledger row.
        owner = resolver.bind_raw_accessor(raw_accessor)
        if owner is not backtest.data_handler:
            raise _activation_mismatch()

        _validate_alpha_max_root_seals(
            raw_root=activation.raw_root,
            phase_id=activation.phase_id,
            ordered_lookup=lookup,
            raw_root_seals=raw_root_seals,
            feature_root_seals=feature_root_seals,
            required=capsule is not None,
            repeat_hash=capsule is not None and activation.repeat_root_hash_on_activation,
        )
        if capsule is None:
            if activation.restored_capsule_sha256 is not None:
                raise _activation_mismatch()
        else:
            restored = _validate_alpha_max_indicator_capsule(
                capsule,
                seal=seal,
                expected_phase_id=_alpha_max_capsule_predecessor(activation.phase_id),
            )
            if activation.restored_capsule_sha256 != capsule.capsule_sha256:
                raise _activation_mismatch()
            if _expected_daily_carry is None:
                actual_state = backtest.strategy.get_research_indicator_state()
                if type(actual_state) is not dict or _canonical_bytes(
                    actual_state
                ) != _canonical_bytes(restored):
                    raise _activation_mismatch()
            else:
                carry = _expected_daily_carry
                handler_rows, handler_timestamps = _alpha_max_handler_carry_rows(
                    backtest.data_handler,
                    admitted,
                )
                if (
                    type(carry) is not _AlphaMaxDailyCarry
                    or not _exact_state_equal(
                        backtest.strategy.get_state(),
                        carry.strategy_state,
                    )
                    or not _exact_state_equal(
                        backtest.portfolio.get_state(),
                        carry.portfolio_state,
                    )
                    or not _exact_state_equal(
                        backtest.execution_handler.get_state(),
                        carry.execution_state,
                    )
                    or not _exact_state_equal(
                        backtest.get_engine_state(),
                        carry.engine_state,
                    )
                    or handler_rows != carry.handler_rows
                    or handler_timestamps != carry.handler_timestamps_ms
                    or resolver.ledger != carry.funding_ledger
                ):
                    raise _activation_mismatch()

        _assert_definition_matches(backtest.strategy, seal)
        _assert_child_identities(backtest.strategy, admitted, seal.expected_definition)
    except AlphaMaxRuntimeContractError as exc:
        if str(exc) == "portfolio_manifest_activation_mismatch":
            raise
        raise _activation_mismatch(exc) from exc
    except Exception as exc:
        raise _activation_mismatch(exc) from exc


def construct_alpha_max_engine(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_path: str | os.PathLike[str],
    admitted_symbols: tuple[str, ...],
    phase_id: str,
    nominal_cost_bps: int,
    raw_root: str | os.PathLike[str],
    ordered_lookup: AlphaMaxOrderedFundingLookup,
    funding_resolver: AlphaMaxFundingBoundaryResolver,
    data_dict: Mapping[str, object] | None = None,
    attribution_collector: AlphaMaxAttributionCollector | None = None,
    full_event_equity_tracker: AlphaMaxStreamingEquityTracker | None = None,
    indicator_capsule: AlphaMaxIndicatorCapsule | None = None,
    raw_root_seals: tuple[AlphaMaxRootSeal, ...] = (),
    feature_root_seals: tuple[AlphaMaxRootSeal, ...] = (),
    _repeat_root_hash_on_activation: bool = True,
    _chunk_start_utc: datetime | None = None,
    _chunk_end_utc: datetime | None = None,
) -> AlphaMaxEngineActivation:
    """Construct one actual, manifest-bound, independent cost-cell engine."""
    reject_ambient_lq_environment()
    _validate_preflight(preflight)
    admitted = _validate_admitted_symbols(preflight, admitted_symbols)
    if type(ordered_lookup) is not AlphaMaxOrderedFundingLookup:
        raise TypeError("alpha_max_ordered_lookup_identity_invalid")
    if ordered_lookup.ordered_root_ids != _alpha_max_expected_root_sequence(phase_id):
        raise AlphaMaxRuntimeContractError("alpha_max_feature_root_sequence_mismatch")
    if type(funding_resolver) is not AlphaMaxFundingBoundaryResolver:
        raise TypeError("alpha_max_funding_resolver_identity_invalid")
    if type(_repeat_root_hash_on_activation) is not bool:
        raise TypeError("alpha_max_root_repeat_flag_invalid")
    if (
        funding_resolver.ordered_lookup is not ordered_lookup
        or funding_resolver.admitted_symbols is not admitted
    ):
        raise AlphaMaxRuntimeContractError("portfolio_manifest_activation_mismatch")
    seal = seal_alpha_max_manifest_activation(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted,
    )
    raw_root_path = _require_exact_explicit_path(raw_root)
    if indicator_capsule is not None:
        _validate_alpha_max_indicator_capsule(
            indicator_capsule,
            seal=seal,
            expected_phase_id=_alpha_max_capsule_predecessor(phase_id),
        )
    _validate_alpha_max_root_seals(
        raw_root=raw_root_path,
        phase_id=phase_id,
        ordered_lookup=ordered_lookup,
        raw_root_seals=raw_root_seals,
        feature_root_seals=feature_root_seals,
        required=indicator_capsule is not None,
        repeat_hash=False,
    )
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id=phase_id,
        admitted_symbols=admitted,
        nominal_cost_bps=nominal_cost_bps,
    )
    fold_start = datetime.fromisoformat(config.START_DATE.replace("Z", "+00:00")).astimezone(UTC)
    fold_end = datetime.fromisoformat(config.END_DATE.replace("Z", "+00:00")).astimezone(UTC)
    chunk_start = fold_start if _chunk_start_utc is None else _chunk_start_utc
    chunk_end = (
        min(fold_start + timedelta(days=1), fold_end) if _chunk_end_utc is None else _chunk_end_utc
    )
    if (
        type(chunk_start) is not datetime
        or type(chunk_end) is not datetime
        or chunk_start.tzinfo != UTC
        or chunk_end.tzinfo != UTC
        or chunk_end - chunk_start != timedelta(days=1)
        or not fold_start <= chunk_start < chunk_end <= fold_end
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_daily_chunk_window_invalid")
    collector = attribution_collector or AlphaMaxAttributionCollector()
    if type(collector) is not AlphaMaxAttributionCollector:
        raise TypeError("alpha_max_attribution_collector_identity_invalid")
    equity_tracker = full_event_equity_tracker or AlphaMaxStreamingEquityTracker()
    if not isinstance(equity_tracker, AlphaMaxStreamingEquityTracker):
        raise TypeError("alpha_max_full_event_equity_tracker_identity_invalid")
    plan = build_alpha_max_engine_constructor_plan(
        preflight,
        config=config,
        feature_lookup=ordered_lookup,
        funding_boundary_resolver=funding_resolver,
        fill_application_attribution_sink=collector.record_application,
        full_event_equity_sink=equity_tracker.observe,
        reporting_sampling_timeframe="4h",
    )
    strategy_params = MappingProxyType(
        {
            "portfolio_mode": seal.expected_definition.portfolio_mode,
            "decision_cadence_seconds": 1,
        }
    )
    backtest = Backtest(
        csv_dir=raw_root_path,
        symbol_list=admitted,
        start_date=chunk_start,
        end_date=chunk_end,
        data_handler_cls=HistoricParquetWindowedDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=ArtifactPortfolioModeStrategy,
        strategy_params=dict(strategy_params),
        data_dict=data_dict,
        **dict(plan.as_kwargs()),
    )
    restored_capsule_sha256: str | None = None
    if indicator_capsule is not None:
        restored = _validate_alpha_max_indicator_capsule(
            indicator_capsule,
            seal=seal,
            expected_phase_id=_alpha_max_capsule_predecessor(phase_id),
        )
        backtest.strategy.set_research_indicator_state(copy.deepcopy(restored))
        backtest.strategy.validate_research_warmup_ready()
        actual_state = backtest.strategy.get_research_indicator_state()
        if type(actual_state) is not dict or _canonical_bytes(actual_state) != _canonical_bytes(
            restored
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_restore_mismatch")
        restored_capsule_sha256 = indicator_capsule.capsule_sha256
    activation = AlphaMaxEngineActivation(
        backtest=backtest,
        preflight=preflight,
        constructor_plan=plan,
        artifact_seal=seal,
        phase_id=phase_id,
        raw_root=raw_root_path,
        admitted_symbols=admitted,
        ordered_lookup=ordered_lookup,
        funding_resolver=funding_resolver,
        attribution_collector=collector,
        full_event_equity_tracker=equity_tracker,
        strategy_params=strategy_params,
        indicator_capsule=indicator_capsule,
        restored_capsule_sha256=restored_capsule_sha256,
        raw_root_seals=raw_root_seals,
        feature_root_seals=feature_root_seals,
        repeat_root_hash_on_activation=_repeat_root_hash_on_activation,
        chunk_start_utc=chunk_start,
        chunk_end_utc=chunk_end,
    )
    validate_alpha_max_engine_activation(activation)
    return activation


def _drain_indicator_events(events: FastQueue) -> int:
    discarded = 0
    while True:
        try:
            event = events.get(False)
        except queue.Empty:
            return discarded
        event_type = str(getattr(event, "type", "")).upper()
        if event_type != "SIGNAL":
            raise AlphaMaxRuntimeContractError(
                f"alpha_max_warmup_economic_event_forbidden:{event_type or 'UNKNOWN'}"
            )
        discarded += 1


def _alpha_max_expected_root_sequence(phase_id: str) -> tuple[str, ...]:
    if phase_id == "warmup":
        return ("warmup",)
    if phase_id == "train":
        return ("warmup", "train")
    if phase_id == "purge":
        return ("train", "purge")
    if phase_id == "validation" or phase_id.startswith("validation_w"):
        return ("purge", "validation")
    if phase_id == "embargo":
        return ("validation", "embargo")
    if phase_id == "historical_exposed_evaluation" or phase_id.startswith("historical_20"):
        return ("embargo", "historical_exposed_evaluation")
    raise AlphaMaxRuntimeContractError(f"alpha_max_phase_root_sequence_unknown:{phase_id}")


def _alpha_max_watermark_ms(value: object) -> int:
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        parsed = int(value)
        return parsed if abs(parsed) >= 100_000_000_000 else parsed * 1000
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise AlphaMaxRuntimeContractError("alpha_max_warmup_watermark_invalid")
        return int(value.astimezone(UTC).timestamp() * 1000)
    if type(value) is str:
        try:
            parsed_dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_warmup_watermark_invalid") from exc
        if parsed_dt.tzinfo is None:
            raise AlphaMaxRuntimeContractError("alpha_max_warmup_watermark_invalid")
        return int(parsed_dt.astimezone(UTC).timestamp() * 1000)
    raise AlphaMaxRuntimeContractError("alpha_max_warmup_watermark_invalid")


@dataclass(frozen=True, slots=True)
class _AlphaMaxIndicatorDayCarry:
    """Exact non-economic continuation state for one completed UTC day."""

    next_day_start_utc: datetime
    strategy_state: object
    aggregator_state: object
    windows_processed: int
    discarded_signal_count: int
    market_event_count: int = 0
    funding_event_count: int = 0
    order_event_count: int = 0
    fill_event_count: int = 0
    trade_count: int = 0


def _alpha_max_indicator_checkpoint_encode(value: object) -> object:
    """Encode only the state types deliberately admitted by the day journal."""
    if value is None:
        return {"t": "none"}
    if type(value) is bool:
        return {"t": "bool", "v": value}
    if type(value) is int:
        return {"t": "int", "v": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_nonfinite")
        return {"t": "float", "v": value.hex()}
    if type(value) is str:
        return {"t": "str", "v": value}
    if type(value) is datetime:
        return {"t": "datetime", "v": value.isoformat(), "fold": value.fold}
    if type(value) is list:
        return {"t": "list", "v": [_alpha_max_indicator_checkpoint_encode(item) for item in value]}
    if type(value) is tuple:
        return {"t": "tuple", "v": [_alpha_max_indicator_checkpoint_encode(item) for item in value]}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_dict_key_invalid")
        return {
            "t": "dict",
            "v": [
                [key, _alpha_max_indicator_checkpoint_encode(value[key])] for key in sorted(value)
            ],
        }
    if type(value) in (set, frozenset):
        try:
            encoded = [_alpha_max_indicator_checkpoint_encode(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_set_invalid"
            ) from exc
        encoded.sort(key=_canonical_bytes)
        if any(encoded[index] == encoded[index - 1] for index in range(1, len(encoded))):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_duplicate_item")
        return {"t": "frozenset" if type(value) is frozenset else "set", "v": encoded}
    raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_type_invalid")


def _alpha_max_indicator_checkpoint_decode(value: object, *, depth: int = 0) -> object:
    if depth > 128 or type(value) is not dict or type(value.get("t")) is not str:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_node_invalid")
    tag = value["t"]
    allowed = {
        "none": {"t"},
        "bool": {"t", "v"},
        "int": {"t", "v"},
        "float": {"t", "v"},
        "str": {"t", "v"},
        "datetime": {"t", "v", "fold"},
        "list": {"t", "v"},
        "tuple": {"t", "v"},
        "dict": {"t", "v"},
        "set": {"t", "v"},
        "frozenset": {"t", "v"},
    }
    if tag not in allowed or set(value) != allowed[tag]:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_node_keys_invalid")
    if tag == "none":
        return None
    if tag == "bool" and type(value["v"]) is bool:
        return value["v"]
    if tag == "int" and type(value["v"]) is str:
        try:
            parsed_int = int(value["v"])
        except ValueError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_int_invalid"
            ) from exc
        if not value["v"] or value["v"] in ("-0", "+0") or str(parsed_int) != value["v"]:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_int_invalid")
        return parsed_int
    if tag == "float" and type(value["v"]) is str:
        try:
            result = float.fromhex(value["v"])
        except ValueError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_float_invalid"
            ) from exc
        if not math.isfinite(result) or result.hex() != value["v"]:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_float_invalid")
        return result
    if tag == "str" and type(value["v"]) is str:
        return value["v"]
    if tag == "datetime" and type(value["v"]) is str and type(value["fold"]) is int:
        try:
            result = datetime.fromisoformat(value["v"]).replace(fold=value["fold"])
        except ValueError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_datetime_invalid"
            ) from exc
        if value["fold"] not in (0, 1) or result.isoformat() != value["v"]:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_datetime_invalid")
        return result
    if tag in ("list", "tuple", "set", "frozenset") and type(value["v"]) is list:
        items = [
            _alpha_max_indicator_checkpoint_decode(item, depth=depth + 1) for item in value["v"]
        ]
        if tag in ("set", "frozenset"):
            encoded = [
                _canonical_bytes(_alpha_max_indicator_checkpoint_encode(item)) for item in items
            ]
            if encoded != sorted(encoded) or len(set(encoded)) != len(encoded):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_set_invalid")
            try:
                result = set(items) if tag == "set" else frozenset(items)
            except TypeError as exc:
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_indicator_checkpoint_set_invalid"
                ) from exc
            if len(result) != len(items):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_bool_int_alias")
            return result
        return items if tag == "list" else tuple(items)
    if tag == "dict" and type(value["v"]) is list:
        result: dict[str, object] = {}
        previous = ""
        for pair in value["v"]:
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or (result and pair[0] <= previous)
                or pair[0] in result
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_dict_invalid")
            previous = pair[0]
            result[pair[0]] = _alpha_max_indicator_checkpoint_decode(pair[1], depth=depth + 1)
        return result
    raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_value_invalid")


def _alpha_max_indicator_checkpoint_bytes(value: object) -> bytes:
    return _canonical_bytes(_alpha_max_indicator_checkpoint_encode(value)) + b"\n"


def _parse_alpha_max_indicator_checkpoint_bytes(payload: bytes) -> object:
    if len(payload) > 16 * 1024 * 1024:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_size_invalid")
    parsed = _strict_json_object(payload)
    if payload != _canonical_bytes(parsed) + b"\n":
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_noncanonical")
    pending = [parsed]
    nodes = 0
    while pending:
        item = pending.pop()
        nodes += 1
        if nodes > 1_000_000:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_node_limit")
        if type(item) is dict:
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
    return _alpha_max_indicator_checkpoint_decode(parsed)


_ALPHA_MAX_INDICATOR_DESCRIPTOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_kind",
        "phase",
        "phase_id",
        "checkpoint_unit",
        "start_utc",
        "end_utc",
        "watermark_utc",
        "window_seconds",
        "windows_per_day",
        "terminal_windows",
        "config",
        "contract_manifest",
        "manifest",
        "admitted_symbols",
        "raw_roots",
        "feature_roots",
        "implementation_identity",
        "runtime_identity",
        "python_identity",
        "thread_identity",
        "candidate_identity",
        "checkpoint",
        "order_routing_enabled",
        "partial_output_reusable",
    }
)


def _alpha_max_indicator_descriptor_datetime(value: object, field: str) -> datetime:
    if type(value) is not str:
        raise AlphaMaxRuntimeContractError(
            f"alpha_max_indicator_checkpoint_descriptor_{field}_invalid"
        )
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlphaMaxRuntimeContractError(
            f"alpha_max_indicator_checkpoint_descriptor_{field}_invalid"
        ) from exc
    if (
        result.tzinfo is None
        or result.utcoffset() != timedelta(0)
        or result.isoformat().replace("+00:00", "Z") != value
    ):
        raise AlphaMaxRuntimeContractError(
            f"alpha_max_indicator_checkpoint_descriptor_{field}_invalid"
        )
    return result


def _alpha_max_indicator_sha256(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _alpha_max_indicator_absolute_path(value: object) -> bool:
    if type(value) is not str or not os.path.isabs(value):
        return False
    path = Path(value)
    return str(path) == value and ".." not in path.parts


def _alpha_max_paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _alpha_max_indicator_receipt_binding(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"byte_count", "path", "sha256"}
        and type(value["byte_count"]) is int
        and value["byte_count"] >= 0
        and _alpha_max_indicator_absolute_path(value["path"])
        and _alpha_max_indicator_sha256(value["sha256"])
    )


def _validate_alpha_max_indicator_day_descriptor(
    descriptor: Mapping[str, object], *, root: Path, parent_identity: tuple[int, int]
) -> tuple[datetime, datetime]:
    if type(descriptor) is not dict or set(descriptor) != _ALPHA_MAX_INDICATOR_DESCRIPTOR_KEYS:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_schema_invalid"
        )
    if (
        descriptor["artifact_kind"] != "alpha_max_indicator_day_checkpoint_attempt.v1"
        or type(descriptor["phase"]) is not str
        or not descriptor["phase"]
        or type(descriptor["phase_id"]) is not str
        or not descriptor["phase_id"]
        or descriptor["checkpoint_unit"] != "whole_utc_day_pre_finalization"
        or descriptor["window_seconds"] != 1
        or descriptor["windows_per_day"] != 86_400
        or descriptor["terminal_windows"] != 31_622_400
        or descriptor["order_routing_enabled"] is not False
        or descriptor["partial_output_reusable"] is not False
        or descriptor["phase_id"] != "warmup"
        or type(descriptor["admitted_symbols"]) is not list
        or not 5 <= len(descriptor["admitted_symbols"]) <= len(ALPHA_MAX_CANDIDATE_SYMBOLS)
        or tuple(descriptor["admitted_symbols"])
        != tuple(
            symbol
            for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS
            if symbol in descriptor["admitted_symbols"]
        )
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_value_invalid"
        )
    start = _alpha_max_indicator_descriptor_datetime(descriptor["start_utc"], "start")
    end = _alpha_max_indicator_descriptor_datetime(descriptor["end_utc"], "end")
    watermark = _alpha_max_indicator_descriptor_datetime(descriptor["watermark_utc"], "watermark")
    if (
        end <= start
        or (end - start).total_seconds() != 31_622_400
        or watermark != end
        or start.hour
        or end.hour
        or start.minute
        or end.minute
        or start.second
        or end.second
        or start.microsecond
        or end.microsecond
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_schedule_invalid"
        )
    config = descriptor["config"]
    contract = descriptor["contract_manifest"]
    manifest = descriptor["manifest"]
    implementation = descriptor["implementation_identity"]
    runtime = descriptor["runtime_identity"]
    python = descriptor["python_identity"]
    if (
        not _alpha_max_indicator_receipt_binding(config)
        or not _alpha_max_indicator_receipt_binding(manifest)
        or type(contract) is not dict
        or set(contract) != {"byte_count", "sha256"}
        or type(contract["byte_count"]) is not int
        or contract["byte_count"] <= 0
        or not _alpha_max_indicator_sha256(contract["sha256"])
        or type(implementation) is not dict
        or set(implementation) != {"inventory"}
        or type(implementation["inventory"]) is not list
        or not implementation["inventory"]
        or any(
            type(row) is not dict
            or set(row) != {"byte_count", "relative_path", "sha256"}
            or type(row["byte_count"]) is not int
            or row["byte_count"] < 0
            or type(row["relative_path"]) is not str
            or not row["relative_path"]
            or Path(row["relative_path"]).is_absolute()
            or ".." in Path(row["relative_path"]).parts
            or not _alpha_max_indicator_sha256(row["sha256"])
            for row in implementation["inventory"]
        )
        or len({row["relative_path"] for row in implementation["inventory"]})
        != len(implementation["inventory"])
        or type(runtime) is not dict
        or set(runtime)
        != {
            "extension_byte_count",
            "extension_module",
            "extension_path",
            "extension_sha256",
            "extension_source_hash",
            "extension_version",
            "runtime_contract_sha256",
        }
        or type(runtime["extension_byte_count"]) is not int
        or runtime["extension_byte_count"] <= 0
        or runtime["extension_module"] != "lumina_quant._compute"
        or not _alpha_max_indicator_absolute_path(runtime["extension_path"])
        or not _alpha_max_indicator_sha256(runtime["extension_sha256"])
        or type(runtime["extension_source_hash"]) is not str
        or re.fullmatch(r"[0-9a-f]{16}", runtime["extension_source_hash"]) is None
        or type(runtime["extension_version"]) is not str
        or not runtime["extension_version"]
        or not _alpha_max_indicator_sha256(runtime["runtime_contract_sha256"])
        or type(python) is not dict
        or set(python)
        != {
            "cache_tag",
            "executable",
            "executable_byte_count",
            "executable_sha256",
            "version",
        }
        or type(python["cache_tag"]) is not str
        or not python["cache_tag"]
        or not _alpha_max_indicator_absolute_path(python["executable"])
        or type(python["executable_byte_count"]) is not int
        or python["executable_byte_count"] <= 0
        or not _alpha_max_indicator_sha256(python["executable_sha256"])
        or type(python["version"]) is not list
        or len(python["version"]) != 3
        or any(type(part) is not int or part < 0 for part in python["version"])
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_identity_invalid"
        )
    if (
        set(descriptor["thread_identity"])
        != {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"}
        or any(value != "1" for value in descriptor["thread_identity"].values())
        or set(descriptor["candidate_identity"])
        != {"path", "candidate_seal_sha256", "capsule_sha256", "finalization_sha256"}
        or type(descriptor["candidate_identity"]["path"]) is not str
        or not _alpha_max_indicator_absolute_path(descriptor["candidate_identity"]["path"])
        or any(
            type(descriptor["candidate_identity"][field]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", descriptor["candidate_identity"][field]) is None
            for field in ("candidate_seal_sha256", "capsule_sha256", "finalization_sha256")
        )
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_identity_invalid"
        )
    for field in ("raw_roots", "feature_roots"):
        roots = descriptor[field]
        if (
            type(roots) is not list
            or not roots
            or any(
                type(item) is not dict
                or set(item) != _ALPHA_MAX_INDICATOR_ROOT_BINDING_KEYS
                or not _alpha_max_indicator_absolute_path(item["path"])
                or type(item["root_id"]) is not str
                or not item["root_id"]
                or item["root_kind"] != ("raw" if field == "raw_roots" else "feature")
                or any(
                    not _alpha_max_indicator_sha256(item[key])
                    for key in (
                        "availability_sha256",
                        "content_sha256",
                        "inventory_sha256",
                        "seal_sha256",
                    )
                )
                for item in roots
            )
        ):
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_descriptor_roots_invalid"
            )
    if (
        len(descriptor["raw_roots"]) != 1
        or descriptor["raw_roots"][0]["root_id"] != "warmup"
        or [item["root_id"] for item in descriptor["feature_roots"]] != ["warmup"]
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_roots_invalid"
        )
    checkpoint = descriptor["checkpoint"]
    try:
        resolved_parent = root.parent.resolve(strict=True)
    except OSError as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_parent_mismatch"
        ) from exc
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != {"root", "parent", "parent_identity"}
        or checkpoint["root"] != str(root)
        or checkpoint["parent"] != str(root.parent)
        or not _alpha_max_indicator_absolute_path(checkpoint["root"])
        or not _alpha_max_indicator_absolute_path(checkpoint["parent"])
        or checkpoint["parent_identity"] != list(parent_identity)
        or any(type(item) is not int or item < 0 for item in checkpoint["parent_identity"])
        or str(resolved_parent) != str(root.parent)
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_parent_mismatch"
        )
    manifest_path = Path(manifest["path"])
    if (
        manifest_path.parent.name != descriptor["phase"]
        or manifest_path.parent.parent.name != "manifests"
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_descriptor_identity_invalid"
        )
    output_root = manifest_path.parent.parent.parent
    protected_paths = (
        output_root,
        Path(config["path"]),
        manifest_path,
        Path(descriptor["candidate_identity"]["path"]),
        Path(runtime["extension_path"]),
        Path(python["executable"]),
        *(Path(item["path"]) for item in descriptor["raw_roots"]),
        *(Path(item["path"]) for item in descriptor["feature_roots"]),
    )
    if any(_alpha_max_paths_overlap(root, path) for path in protected_paths):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_path_overlap")
    return start, end


def _alpha_max_read_regular_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_open_invalid") from exc
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_file_invalid")
        payload = b""
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                return payload
            payload += block
            if len(payload) > 16 * 1024 * 1024:
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_size_invalid")
    finally:
        os.close(fd)


def _alpha_max_fsync_regular_nofollow(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_file_invalid")
        os.fsync(fd)
    finally:
        os.close(fd)


def _alpha_max_read_regular_at(
    directory_fd: int,
    name: str,
    *,
    expected_mode: int,
) -> bytes:
    if (
        type(directory_fd) is not int
        or type(name) is not str
        or "/" in name
        or name in {"", ".", ".."}
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_file_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_open_invalid") from exc
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or stat.S_IMODE(status.st_mode) != expected_mode
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_file_invalid")
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
            size += len(block)
            if size > 16 * 1024 * 1024:
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_size_invalid")
    finally:
        os.close(descriptor)


def _alpha_max_open_checkpoint_directory_at(parent_fd: int, name: str) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_directory_invalid"
        ) from exc
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        os.close(descriptor)
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_directory_invalid")
    return descriptor


class _AlphaMaxIndicatorDayCheckpointStore:
    """Descriptor-bound append-only operational carry journal."""

    def __init__(self, root: str | os.PathLike[str], *, descriptor: Mapping[str, object]) -> None:
        self.root = Path(_require_exact_explicit_path(root))
        self._descriptor = copy.deepcopy(dict(descriptor))
        self._lock_fd: int | None = None
        try:
            if str(self.root.parent.resolve(strict=True)) != str(self.root.parent):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_parent_invalid")
        except OSError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_parent_invalid"
            ) from exc
        self._parent_fd = _alpha_max_open_directory_at(self.root.parent)
        parent_status = os.fstat(self._parent_fd)
        if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(parent_status.st_mode):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_parent_invalid")
        self._parent_identity = (int(parent_status.st_dev), int(parent_status.st_ino))
        self._start_utc, self._end_utc = _validate_alpha_max_indicator_day_descriptor(
            descriptor, root=self.root, parent_identity=self._parent_identity
        )
        self._descriptor_bytes = _canonical_bytes(dict(descriptor)) + b"\n"
        self.descriptor_sha256 = _sha256(self._descriptor_bytes)
        parent_path = Path(f"/proc/self/fd/{self._parent_fd}")
        lock_name = f".{self.root.name}.alpha-max-indicator.lock"
        self._lock_name = lock_name
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        lock_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            self._lock_fd = os.open(lock_name, lock_flags, 0o600, dir_fd=self._parent_fd)
            lock_status = os.fstat(self._lock_fd)
            if not stat.S_ISREG(lock_status.st_mode) or int(lock_status.st_nlink) != 1:
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_lock_invalid")
            self._lock_identity = (int(lock_status.st_dev), int(lock_status.st_ino))
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_lock_unavailable"
            ) from exc
        try:
            os.stat(self.root.name, dir_fd=self._parent_fd, follow_symlinks=False)
            root_exists = True
        except FileNotFoundError:
            root_exists = False
        if not root_exists:
            init_prefix = f".{self.root.name}.alpha-max-indicator-init.staging-"
            for name in os.listdir(self._parent_fd):
                if not name.startswith(init_prefix):
                    continue
                if re.fullmatch(re.escape(init_prefix) + r"[a-z0-9_]{8}", name) is None:
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_indicator_checkpoint_unknown_entry"
                    )
                _alpha_max_cleanup_recognized_staging_bundle(
                    parent_path / name,
                    allowed_files=frozenset({"ATTEMPT.json"}),
                    allowed_directories=frozenset({"days"}),
                    error_token="alpha_max_indicator_checkpoint_unknown_entry",
                )
                os.fsync(self._parent_fd)
            stage = Path(
                tempfile.mkdtemp(
                    prefix=init_prefix,
                    dir=parent_path,
                )
            )
            try:
                _write_bundle_file(stage, "ATTEMPT.json", self._descriptor_bytes)
                os.chmod(stage / "ATTEMPT.json", 0o400)
                _alpha_max_fsync_regular_nofollow(stage / "ATTEMPT.json")
                # The journal directory must remain writable for the next
                # atomic day publication; sealed children are immutable.
                (stage / "days").mkdir(mode=0o700)
                _fsync_directory(stage / "days")
                os.chmod(stage, 0o500)
                _fsync_directory(stage)
                _rename_bundle_noreplace(stage, parent_path / self.root.name)
                os.fsync(self._parent_fd)
            except Exception:
                _cleanup_partial_bundle(stage)
                raise
        self._root_fd = _alpha_max_open_checkpoint_directory_at(self._parent_fd, self.root.name)
        self._days_fd = _alpha_max_open_checkpoint_directory_at(self._root_fd, "days")
        try:
            fcntl.flock(self._root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_lock_unavailable"
            ) from exc
        root_status = os.fstat(self._root_fd)
        days_status = os.fstat(self._days_fd)
        self._root_identity = (int(root_status.st_dev), int(root_status.st_ino))
        self._days_identity = (int(days_status.st_dev), int(days_status.st_ino))
        self._journal_loaded = False
        self._latest_carry: _AlphaMaxIndicatorDayCarry | None = None
        self._latest_seal_sha256 = ""
        self._day_identities: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._validate_root()

    def __del__(self) -> None:
        for name in ("_days_fd", "_root_fd", "_lock_fd", "_parent_fd"):
            descriptor = getattr(self, name, None)
            if type(descriptor) is int:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, name, None)

    def _validate_root(self) -> None:
        try:
            if str(self.root.parent.resolve(strict=True)) != str(self.root.parent):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_parent_replaced")
            lock_path = os.stat(self._lock_name, dir_fd=self._parent_fd, follow_symlinks=False)
            parent = self.root.parent.lstat()
            root = self.root.lstat()
            days = (self.root / "days").lstat()
        except OSError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_parent_replaced"
            ) from exc
        open_parent = os.fstat(self._parent_fd)
        open_root = os.fstat(self._root_fd)
        open_days = os.fstat(self._days_fd)
        open_lock = os.fstat(self._lock_fd)
        if (
            (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or (int(open_parent.st_dev), int(open_parent.st_ino)) != self._parent_identity
            or (int(root.st_dev), int(root.st_ino)) != self._root_identity
            or (int(open_root.st_dev), int(open_root.st_ino)) != self._root_identity
            or (int(days.st_dev), int(days.st_ino)) != self._days_identity
            or (int(open_days.st_dev), int(open_days.st_ino)) != self._days_identity
            or (int(lock_path.st_dev), int(lock_path.st_ino)) != self._lock_identity
            or (int(open_lock.st_dev), int(open_lock.st_ino)) != self._lock_identity
            or not stat.S_ISREG(lock_path.st_mode)
            or stat.S_ISLNK(lock_path.st_mode)
            or int(lock_path.st_nlink) != 1
            or stat.S_ISLNK(root.st_mode)
            or stat.S_ISLNK(days.st_mode)
            or stat.S_IMODE(root.st_mode) != 0o500
            or stat.S_IMODE(days.st_mode) != 0o700
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_parent_replaced")
        if set(os.listdir(self._root_fd)) != {"ATTEMPT.json", "days"}:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_inventory_invalid")
        payload = _alpha_max_read_regular_at(self._root_fd, "ATTEMPT.json", expected_mode=0o400)
        if payload != self._descriptor_bytes:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_descriptor_mismatch")

    def _cleanup_staging(self, name: str) -> None:
        stage_fd = _alpha_max_open_checkpoint_directory_at(self._days_fd, name)
        try:
            names = set(os.listdir(stage_fd))
            if not names <= {"STATE.json", "SEALED.json"}:
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_unknown_entry")
            for filename in names:
                status = os.stat(filename, dir_fd=stage_fd, follow_symlinks=False)
                if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_indicator_checkpoint_unknown_entry"
                    )
                os.unlink(filename, dir_fd=stage_fd)
            os.fsync(stage_fd)
            os.fchmod(stage_fd, 0o700)
        finally:
            os.close(stage_fd)
        os.rmdir(name, dir_fd=self._days_fd)
        os.fsync(self._days_fd)

    @staticmethod
    def _day_name(value: datetime) -> str:
        if (
            type(value) is not datetime
            or value.tzinfo is None
            or value.utcoffset() != timedelta(0)
            or (value.hour, value.minute, value.second, value.microsecond) != (0, 0, 0, 0)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_day_invalid")
        return value.strftime("%Y%m%d")

    @staticmethod
    def _entry_identity(status: os.stat_result) -> tuple[int, ...]:
        return (
            int(status.st_dev),
            int(status.st_ino),
            int(status.st_mode),
            int(status.st_nlink),
            int(status.st_size),
            int(status.st_mtime_ns),
            int(status.st_ctime_ns),
        )

    def _validate_cached_journal(self) -> None:
        self._validate_root()
        if not self._journal_loaded:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_journal_not_loaded")
        if set(os.listdir(self._days_fd)) != set(self._day_identities):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_inventory_invalid")
        for name, expected in self._day_identities.items():
            target_fd = _alpha_max_open_checkpoint_directory_at(self._days_fd, name)
            try:
                if (
                    set(os.listdir(target_fd)) != {"STATE.json", "SEALED.json"}
                    or self._entry_identity(os.fstat(target_fd)) != expected[0]
                    or self._entry_identity(
                        os.stat("STATE.json", dir_fd=target_fd, follow_symlinks=False)
                    )
                    != expected[1]
                    or self._entry_identity(
                        os.stat("SEALED.json", dir_fd=target_fd, follow_symlinks=False)
                    )
                    != expected[2]
                ):
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_indicator_checkpoint_identity_changed"
                    )
            finally:
                os.close(target_fd)

    def load_latest(
        self, *, start_utc: datetime, end_utc: datetime
    ) -> _AlphaMaxIndicatorDayCarry | None:
        self._validate_root()
        if start_utc != self._start_utc or end_utc != self._end_utc:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_schedule_mismatch")
        for name in os.listdir(self._days_fd):
            staging_match = re.fullmatch(
                r"\.alpha-max-indicator-day-(\d{8})\.staging-[a-z0-9_]{8}",
                name,
            )
            if staging_match is not None:
                try:
                    staging_day = datetime.strptime(staging_match.group(1), "%Y%m%d").replace(
                        tzinfo=UTC
                    )
                except ValueError as exc:
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_indicator_checkpoint_unknown_entry"
                    ) from exc
                if not self._start_utc <= staging_day < self._end_utc:
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_indicator_checkpoint_unknown_entry"
                    )
                self._cleanup_staging(name)
            elif not name.isdigit() or len(name) != 8:
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_unknown_entry")
        carry: _AlphaMaxIndicatorDayCarry | None = None
        expected = start_utc
        previous_seal = ""
        identities: dict[str, tuple[tuple[int, ...], ...]] = {}
        for name in sorted(os.listdir(self._days_fd)):
            target_fd = _alpha_max_open_checkpoint_directory_at(self._days_fd, name)
            try:
                target_status = os.fstat(target_fd)
                if (
                    name != self._day_name(expected)
                    or stat.S_IMODE(target_status.st_mode) != 0o500
                    or set(os.listdir(target_fd)) != {"STATE.json", "SEALED.json"}
                ):
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_indicator_checkpoint_gap_or_invalid"
                    )
                state_status = os.stat("STATE.json", dir_fd=target_fd, follow_symlinks=False)
                seal_status = os.stat("SEALED.json", dir_fd=target_fd, follow_symlinks=False)
                state_bytes = _alpha_max_read_regular_at(
                    target_fd, "STATE.json", expected_mode=0o400
                )
                seal_bytes = _alpha_max_read_regular_at(
                    target_fd, "SEALED.json", expected_mode=0o400
                )
                identity = (
                    self._entry_identity(target_status),
                    self._entry_identity(state_status),
                    self._entry_identity(seal_status),
                )
                if identity != (
                    self._entry_identity(os.fstat(target_fd)),
                    self._entry_identity(
                        os.stat(
                            "STATE.json",
                            dir_fd=target_fd,
                            follow_symlinks=False,
                        )
                    ),
                    self._entry_identity(
                        os.stat(
                            "SEALED.json",
                            dir_fd=target_fd,
                            follow_symlinks=False,
                        )
                    ),
                ):
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_indicator_checkpoint_identity_changed"
                    )
            finally:
                os.close(target_fd)
            if not state_bytes or not seal_bytes:
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_file_invalid")
            seal_payload = _strict_json_object(seal_bytes)
            ordinal = (expected - start_utc).days + 1
            expected_seal = {
                "artifact_kind": "alpha_max_indicator_day_checkpoint_seal.v1",
                "attempt_descriptor_sha256": self.descriptor_sha256,
                "byte_count": len(state_bytes),
                "day": name,
                "next_day_start_utc": (expected + timedelta(days=1))
                .isoformat()
                .replace("+00:00", "Z"),
                "ordinal": ordinal,
                "previous_seal_sha256": previous_seal,
                "state_sha256": _sha256(state_bytes),
                "success": True,
            }
            if (
                seal_bytes != _canonical_bytes(seal_payload) + b"\n"
                or seal_payload != expected_seal
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_seal_invalid")
            parsed = _parse_alpha_max_indicator_checkpoint_bytes(state_bytes)
            if type(parsed) is not dict or set(parsed) != {
                "aggregator_state",
                "discarded_signal_count",
                "market_event_count",
                "funding_event_count",
                "order_event_count",
                "fill_event_count",
                "trade_count",
                "next_day_start_utc",
                "strategy_state",
                "windows_processed",
            }:
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_state_invalid")
            if (
                type(parsed["next_day_start_utc"]) is not datetime
                or type(parsed["strategy_state"]) is not dict
                or type(parsed["aggregator_state"]) is not dict
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_state_invalid")
            carry = _AlphaMaxIndicatorDayCarry(**parsed)
            counts = (
                carry.windows_processed,
                carry.discarded_signal_count,
                carry.market_event_count,
                carry.funding_event_count,
                carry.order_event_count,
                carry.fill_event_count,
                carry.trade_count,
            )
            if (
                any(type(count) is not int or count < 0 for count in counts)
                or any(count != 0 for count in counts[2:])
                or carry.windows_processed != ordinal * 86_400
                or carry.next_day_start_utc != expected + timedelta(days=1)
                or carry.next_day_start_utc > end_utc
            ):
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_indicator_checkpoint_next_day_invalid"
                )
            identities[name] = identity
            expected, previous_seal = carry.next_day_start_utc, _sha256(seal_bytes)
        self._day_identities = identities
        self._latest_carry = copy.deepcopy(carry)
        self._latest_seal_sha256 = previous_seal
        self._journal_loaded = True
        return copy.deepcopy(carry)

    def seal(self, carry: _AlphaMaxIndicatorDayCarry) -> None:
        self._validate_root()
        if type(carry) is not _AlphaMaxIndicatorDayCarry:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_carry_invalid")
        if type(carry.strategy_state) is not dict or type(carry.aggregator_state) is not dict:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_carry_invalid")
        counts = (
            carry.windows_processed,
            carry.discarded_signal_count,
            carry.market_event_count,
            carry.funding_event_count,
            carry.order_event_count,
            carry.fill_event_count,
            carry.trade_count,
        )
        if any(type(count) is not int or count < 0 for count in counts) or any(
            count != 0 for count in counts[2:]
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_carry_invalid")
        if self._journal_loaded:
            self._validate_cached_journal()
            latest = self._latest_carry
        else:
            latest = self.load_latest(
                start_utc=self._start_utc,
                end_utc=self._end_utc,
            )
        expected_next = (
            self._start_utc + timedelta(days=1)
            if latest is None
            else latest.next_day_start_utc + timedelta(days=1)
        )
        if carry.next_day_start_utc != expected_next or (
            latest is not None and carry.discarded_signal_count < latest.discarded_signal_count
        ):
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_carry_sequence_invalid"
            )
        name = self._day_name(carry.next_day_start_utc - timedelta(days=1))
        try:
            os.stat(name, dir_fd=self._days_fd, follow_symlinks=False)
            target_exists = True
        except FileNotFoundError:
            target_exists = False
        if target_exists:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_day_exists")
        ordinal = (carry.next_day_start_utc - self._start_utc).days
        if carry.next_day_start_utc >= self._end_utc or carry.windows_processed != ordinal * 86_400:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_carry_schedule_invalid"
            )
        state_value = {
            "next_day_start_utc": carry.next_day_start_utc,
            "strategy_state": carry.strategy_state,
            "aggregator_state": carry.aggregator_state,
            "windows_processed": carry.windows_processed,
            "discarded_signal_count": carry.discarded_signal_count,
            "market_event_count": 0,
            "funding_event_count": 0,
            "order_event_count": 0,
            "fill_event_count": 0,
            "trade_count": 0,
        }
        state = _alpha_max_indicator_checkpoint_bytes(state_value)
        if not _exact_state_equal(_parse_alpha_max_indicator_checkpoint_bytes(state), state_value):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_roundtrip_invalid")
        previous_seal_sha256 = self._latest_seal_sha256
        seal = (
            _canonical_bytes(
                {
                    "artifact_kind": "alpha_max_indicator_day_checkpoint_seal.v1",
                    "attempt_descriptor_sha256": self.descriptor_sha256,
                    "byte_count": len(state),
                    "day": name,
                    "next_day_start_utc": carry.next_day_start_utc.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "ordinal": ordinal,
                    "previous_seal_sha256": previous_seal_sha256,
                    "state_sha256": _sha256(state),
                    "success": True,
                }
            )
            + b"\n"
        )
        days_path = Path(f"/proc/self/fd/{self._days_fd}")
        target = days_path / name
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".alpha-max-indicator-day-{name}.staging-",
                dir=days_path,
            )
        )
        published = False
        try:
            _write_bundle_file(stage, "STATE.json", state)
            _write_bundle_file(stage, "SEALED.json", seal)
            os.chmod(stage / "STATE.json", 0o400)
            _alpha_max_fsync_regular_nofollow(stage / "STATE.json")
            os.chmod(stage / "SEALED.json", 0o400)
            _alpha_max_fsync_regular_nofollow(stage / "SEALED.json")
            os.chmod(stage, 0o500)
            _fsync_directory(stage)
            _rename_bundle_noreplace(stage, target)
            published = True
            os.fsync(self._days_fd)
            target_fd = _alpha_max_open_checkpoint_directory_at(self._days_fd, name)
            try:
                if (
                    _alpha_max_read_regular_at(
                        target_fd,
                        "STATE.json",
                        expected_mode=0o400,
                    )
                    != state
                    or _alpha_max_read_regular_at(
                        target_fd,
                        "SEALED.json",
                        expected_mode=0o400,
                    )
                    != seal
                ):
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_indicator_checkpoint_publication_invalid"
                    )
                identity = (
                    self._entry_identity(os.fstat(target_fd)),
                    self._entry_identity(
                        os.stat(
                            "STATE.json",
                            dir_fd=target_fd,
                            follow_symlinks=False,
                        )
                    ),
                    self._entry_identity(
                        os.stat(
                            "SEALED.json",
                            dir_fd=target_fd,
                            follow_symlinks=False,
                        )
                    ),
                )
            finally:
                os.close(target_fd)
        except Exception:
            try:
                if published:
                    rollback_fd = _alpha_max_open_checkpoint_directory_at(self._days_fd, name)
                    try:
                        _alpha_max_cleanup_directory_fd(rollback_fd)
                    finally:
                        os.close(rollback_fd)
                    os.rmdir(name, dir_fd=self._days_fd)
                    os.fsync(self._days_fd)
                else:
                    _cleanup_partial_bundle(stage)
            except Exception as rollback_exc:
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_indicator_checkpoint_rollback_failed"
                ) from rollback_exc
            raise
        self._day_identities[name] = identity
        self._latest_carry = copy.deepcopy(carry)
        self._latest_seal_sha256 = _sha256(seal)
        self._journal_loaded = True


_ALPHA_MAX_INDICATOR_THREAD_KEYS: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
)
_ALPHA_MAX_INDICATOR_ROOT_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "availability_sha256",
        "content_sha256",
        "inventory_sha256",
        "path",
        "root_id",
        "root_kind",
        "seal_sha256",
    }
)


def _alpha_max_indicator_root_binding(seal: AlphaMaxRootSeal) -> dict[str, object]:
    if type(seal) is not AlphaMaxRootSeal:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_root_identity_invalid")
    seal.__post_init__()
    return {
        "availability_sha256": seal.availability_sha256,
        "content_sha256": seal.content_sha256,
        "inventory_sha256": seal.inventory_sha256,
        "path": _require_exact_explicit_path(seal.path),
        "root_id": seal.root_id,
        "root_kind": seal.root_kind,
        "seal_sha256": seal.sha256,
    }


def _alpha_max_indicator_candidate_binding(
    candidate_identity: Mapping[str, object],
) -> dict[str, object]:
    expected_keys = {
        "path",
        "candidate_seal_sha256",
        "capsule_sha256",
        "finalization_sha256",
    }
    if (
        type(candidate_identity) is not dict
        or set(candidate_identity) != expected_keys
        or type(candidate_identity["path"]) is not str
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_candidate_identity_invalid"
        )
    path = Path(_require_exact_explicit_path(candidate_identity["path"]))
    try:
        status = path.lstat()
    except OSError as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_candidate_identity_invalid"
        ) from exc
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or status.st_nlink != 1
        or status.st_mode & 0o222
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_candidate_identity_invalid"
        )
    receipt, payload = read_artifact_bytes(path, artifact_id="alpha-max-indicator-candidate-seal")
    hashes = {
        name: candidate_identity[name]
        for name in (
            "candidate_seal_sha256",
            "capsule_sha256",
            "finalization_sha256",
        )
    }
    if (
        any(
            type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in hashes.values()
        )
        or hashes["candidate_seal_sha256"] != receipt.sha256
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_candidate_identity_invalid"
        )
    parsed = _strict_json_object(payload)
    candidate = parsed.get("candidate")
    if (
        type(candidate) is not dict
        or candidate.get("capsule_sha256") != hashes["capsule_sha256"]
        or candidate.get("native_finalization_sha256") != hashes["finalization_sha256"]
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_candidate_identity_invalid"
        )
    return {
        "path": receipt.canonical_path,
        **hashes,
    }


def _alpha_max_loaded_mapping_identity(
    path: Path,
    builtins: tuple[types.BuiltinFunctionType, ...] = (),
) -> tuple[int, int]:
    """Bind a loaded native object's mapped inode to its current pathname."""
    try:
        status = path.lstat()
        if (
            not stat.S_ISREG(status.st_mode)
            or stat.S_ISLNK(status.st_mode)
            or int(status.st_nlink) != 1
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid")
        target = str(path)
        identities: set[tuple[int, int]] = set()
        mappings_by_address: list[tuple[int, int, tuple[int, int], str]] = []
        with Path("/proc/self/maps").open(encoding="utf-8") as mappings:
            for line in mappings:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) != 6:
                    continue
                mapped_path = fields[5].removesuffix(" (deleted)")
                major_text, minor_text = fields[3].split(":", 1)
                identity = (
                    int(os.makedev(int(major_text, 16), int(minor_text, 16))),
                    int(fields[4]),
                )
                start_text, end_text = fields[0].split("-", 1)
                mappings_by_address.append(
                    (int(start_text, 16), int(end_text, 16), identity, mapped_path)
                )
                if mapped_path == target:
                    identities.add(identity)
    except (OSError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid") from exc
    expected = (int(status.st_dev), int(status.st_ino))
    if identities != {expected}:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid")
    try:
        get_function = ctypes.pythonapi.PyCFunction_GetFunction
        get_function.argtypes = (ctypes.py_object,)
        get_function.restype = ctypes.c_void_p
        addresses = tuple(int(get_function(function) or 0) for function in builtins)
    except (AttributeError, TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid") from exc
    for address in addresses:
        owners = {
            (identity, mapped_path)
            for start, end, identity, mapped_path in mappings_by_address
            if start <= address < end
        }
        if owners != {(expected, target)}:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid")
    return expected


def _alpha_max_indicator_runtime_binding() -> dict[str, object]:
    try:
        from lumina_quant import _compute
        from lumina_quant._native_kernel_version import compute_src_hash

        extension_path = Path(_compute.__file__).resolve(strict=True)
        expected_parent = Path(__file__).resolve().parents[1]
        fold_native = getattr(_compute, "fold_alpha_max_native_bars", None)
        build_info = getattr(_compute, "build_info", None)
        kernel_src_hash = getattr(_compute, "kernel_src_hash", None)
        if (
            type(_compute) is not types.ModuleType
            or _compute.__name__ != "lumina_quant._compute"
            or extension_path.parent != expected_parent
            or not extension_path.name.startswith("_compute.")
            or type(fold_native) is not types.BuiltinFunctionType
            or type(build_info) is not types.BuiltinFunctionType
            or type(kernel_src_hash) is not types.BuiltinFunctionType
            or any(
                function.__module__ != _compute.__name__
                for function in (fold_native, build_info, kernel_src_hash)
            )
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid")
        mapped_identity = _alpha_max_loaded_mapping_identity(
            extension_path,
            (fold_native, build_info, kernel_src_hash),
        )
        receipt, _payload = read_artifact_bytes(
            extension_path, artifact_id="alpha-max-indicator-native-extension"
        )
        version = build_info()
        source_hash = kernel_src_hash()
        cargo_toml = (
            Path(__file__).resolve().parents[3] / "native" / "lumina_compute" / "Cargo.toml"
        )
        expected_version = next(
            (
                line.split('"')[1]
                for line in cargo_toml.read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("version = ")
            ),
            None,
        )
        if (
            type(version) is not str
            or not version
            or version != expected_version
            or type(source_hash) is not str
            or re.fullmatch(r"[0-9a-f]{16}", source_hash) is None
            or source_hash != compute_src_hash()
            or tuple(receipt.pre_fstat_identity[:2]) != mapped_identity
            or tuple(receipt.post_fstat_identity[:2]) != mapped_identity
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid")
    except AlphaMaxRuntimeContractError:
        raise
    except Exception as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid") from exc
    return {
        "extension_byte_count": receipt.byte_count,
        "extension_module": _compute.__name__,
        "extension_path": receipt.canonical_path,
        "extension_sha256": receipt.sha256,
        "extension_source_hash": source_hash,
        "extension_version": version,
    }


def _alpha_max_indicator_day_checkpoint_descriptor(
    preflight: AlphaMaxRuntimePreflight,
    *,
    checkpoint_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_path: str | os.PathLike[str],
    admitted_symbols: tuple[str, ...],
    phase_id: str,
    raw_root: str | os.PathLike[str],
    ordered_lookup: AlphaMaxOrderedFundingLookup,
    watermark: object,
    bounded_raw_loader: _AlphaMaxBoundedRawLoader,
    checkpoint_candidate_identity: Mapping[str, object],
) -> dict[str, object]:
    """Derive the exact active identity for one checkpointed warmup replay."""
    _validate_preflight(preflight)
    if (
        type(bounded_raw_loader) is not _AlphaMaxBoundedRawLoader
        or type(ordered_lookup) is not AlphaMaxOrderedFundingLookup
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_candidate_identity_invalid"
        )
    admitted = _validate_admitted_symbols(preflight, admitted_symbols)
    seal = seal_alpha_max_manifest_activation(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted,
    )
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id=phase_id,
        admitted_symbols=admitted,
        nominal_cost_bps=20,
    )
    if _alpha_max_watermark_ms(watermark) != _alpha_max_watermark_ms(config.END_DATE):
        raise AlphaMaxRuntimeContractError("alpha_max_warmup_watermark_mismatch")
    if (
        phase_id != "warmup"
        or config.START_DATE != "2022-12-31T00:00:00Z"
        or config.END_DATE != "2024-01-01T00:00:00Z"
        or any(os.environ.get(name) != "1" for name in _ALPHA_MAX_INDICATOR_THREAD_KEYS)
    ):
        raise AlphaMaxRuntimeContractError(
            "alpha_max_indicator_checkpoint_runtime_contract_invalid"
        )
    root = Path(_require_exact_explicit_path(checkpoint_root))
    parent = root.parent
    parent_status = parent.lstat()
    raw_path = _require_exact_explicit_path(raw_root)
    if (
        bounded_raw_loader.seal.path != raw_path
        or bounded_raw_loader.seal.root_id != "warmup"
        or bounded_raw_loader.seal.root_kind != "raw"
        or tuple(seal.root_id for seal in ordered_lookup.root_seals)
        != ordered_lookup.ordered_root_ids
        or not ordered_lookup.root_seals
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_root_identity_invalid")
    manifest_receipt = seal.manifest_receipt
    config_receipt = preflight.config_receipt
    python_receipt, _ = read_artifact_bytes(
        Path(sys.executable).resolve(strict=True),
        artifact_id="alpha-max-indicator-python-executable",
    )
    descriptor: dict[str, object] = {
        "artifact_kind": "alpha_max_indicator_day_checkpoint_attempt.v1",
        "phase": phase,
        "phase_id": phase_id,
        "checkpoint_unit": "whole_utc_day_pre_finalization",
        "start_utc": config.START_DATE,
        "end_utc": config.END_DATE,
        "watermark_utc": config.END_DATE,
        "window_seconds": 1,
        "windows_per_day": 86_400,
        "terminal_windows": 31_622_400,
        "config": {
            "path": config_receipt.canonical_path,
            "sha256": config_receipt.sha256,
            "byte_count": config_receipt.byte_count,
        },
        "contract_manifest": {
            "sha256": preflight.runtime_contract_sha256,
            "byte_count": len(preflight.runtime_contract_bytes),
        },
        "manifest": {
            "path": manifest_receipt.canonical_path,
            "sha256": manifest_receipt.sha256,
            "byte_count": manifest_receipt.byte_count,
        },
        "admitted_symbols": list(admitted),
        "raw_roots": [_alpha_max_indicator_root_binding(bounded_raw_loader.seal)],
        "feature_roots": [
            _alpha_max_indicator_root_binding(root_seal) for root_seal in ordered_lookup.root_seals
        ],
        "implementation_identity": {"inventory": _alpha_max_checkpoint_implementation_inventory()},
        "runtime_identity": {
            "runtime_contract_sha256": preflight.runtime_contract_sha256,
            **_alpha_max_indicator_runtime_binding(),
        },
        "python_identity": {
            "cache_tag": sys.implementation.cache_tag,
            "executable": python_receipt.canonical_path,
            "executable_byte_count": python_receipt.byte_count,
            "executable_sha256": python_receipt.sha256,
            "version": list(sys.version_info[:3]),
        },
        "thread_identity": {name: os.environ[name] for name in _ALPHA_MAX_INDICATOR_THREAD_KEYS},
        "candidate_identity": _alpha_max_indicator_candidate_binding(checkpoint_candidate_identity),
        "checkpoint": {
            "root": str(root),
            "parent": str(parent),
            "parent_identity": [int(parent_status.st_dev), int(parent_status.st_ino)],
        },
        "order_routing_enabled": False,
        "partial_output_reusable": False,
    }
    _validate_alpha_max_indicator_day_descriptor(
        descriptor,
        root=root,
        parent_identity=(int(parent_status.st_dev), int(parent_status.st_ino)),
    )
    return descriptor


def create_alpha_max_indicator_day_checkpoint_store(
    preflight: AlphaMaxRuntimePreflight,
    *,
    checkpoint_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_path: str | os.PathLike[str],
    admitted_symbols: tuple[str, ...],
    phase_id: str,
    raw_root: str | os.PathLike[str],
    ordered_lookup: AlphaMaxOrderedFundingLookup,
    watermark: object,
    bounded_raw_loader: _AlphaMaxBoundedRawLoader,
    checkpoint_candidate_identity: Mapping[str, object],
) -> _AlphaMaxIndicatorDayCheckpointStore:
    """Create the exact store used by production whole-day native replay."""
    descriptor = _alpha_max_indicator_day_checkpoint_descriptor(
        preflight,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted_symbols,
        phase_id=phase_id,
        raw_root=raw_root,
        ordered_lookup=ordered_lookup,
        watermark=watermark,
        bounded_raw_loader=bounded_raw_loader,
        checkpoint_candidate_identity=checkpoint_candidate_identity,
    )
    return _AlphaMaxIndicatorDayCheckpointStore(checkpoint_root, descriptor=descriptor)


def _build_alpha_max_indicator_capsule_incremental(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_path: str | os.PathLike[str],
    admitted_symbols: tuple[str, ...],
    phase_id: str,
    raw_root: str | os.PathLike[str],
    ordered_lookup: AlphaMaxOrderedFundingLookup,
    watermark: object,
    data_dict: Mapping[str, object] | None = None,
    prior_indicator_capsule: AlphaMaxIndicatorCapsule | None = None,
    bounded_raw_loader: _AlphaMaxBoundedRawLoader | None = None,
    checkpoint_store: _AlphaMaxIndicatorDayCheckpointStore | None = None,
) -> AlphaMaxIndicatorCapsule:
    """Prime research indicators with the real windowed handler and no economics."""
    reject_ambient_lq_environment()
    _validate_preflight(preflight)
    admitted = _validate_admitted_symbols(preflight, admitted_symbols)
    if type(ordered_lookup) is not AlphaMaxOrderedFundingLookup:
        raise TypeError("alpha_max_ordered_lookup_identity_invalid")
    if ordered_lookup.ordered_root_ids != _alpha_max_expected_root_sequence(phase_id):
        raise AlphaMaxRuntimeContractError("alpha_max_feature_root_sequence_mismatch")
    seal = seal_alpha_max_manifest_activation(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted,
    )
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id=phase_id,
        admitted_symbols=admitted,
        nominal_cost_bps=20,
    )
    if _alpha_max_watermark_ms(watermark) != _alpha_max_watermark_ms(config.END_DATE):
        raise AlphaMaxRuntimeContractError("alpha_max_warmup_watermark_mismatch")
    start_utc = datetime.fromisoformat(config.START_DATE.replace("Z", "+00:00")).astimezone(UTC)
    end_utc = datetime.fromisoformat(config.END_DATE.replace("Z", "+00:00")).astimezone(UTC)
    carry = None
    if checkpoint_store is not None:
        if type(checkpoint_store) is not _AlphaMaxIndicatorDayCheckpointStore:
            raise TypeError("alpha_max_indicator_checkpoint_store_invalid")
        carry = checkpoint_store.load_latest(start_utc=start_utc, end_utc=end_utc)
    initial_day_start = start_utc if carry is None else carry.next_day_start_utc
    if initial_day_start >= end_utc:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_terminal_carry_invalid")
    if bounded_raw_loader is not None:
        if type(bounded_raw_loader) is not _AlphaMaxBoundedRawLoader:
            raise TypeError("alpha_max_bounded_raw_loader_identity_invalid")
        if data_dict is not None:
            raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_loader_data_conflict")
        if (
            bounded_raw_loader.seal.path != _require_exact_explicit_path(raw_root)
            or bounded_raw_loader.seal.root_id != _alpha_max_current_root_id(phase_id)
            or bounded_raw_loader.seal.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_loader_scope_invalid")
        data_dict = bounded_raw_loader.load_day(
            initial_day_start, min(initial_day_start + timedelta(days=1), end_utc)
        )
    events = FastQueue()
    handler = HistoricParquetWindowedDataHandler(
        events,
        _require_exact_explicit_path(raw_root),
        admitted,
        initial_day_start,
        min(initial_day_start + timedelta(days=1), end_utc),
        data_dict,
        backtest_poll_seconds=1,
        backtest_window_seconds=1,
        feature_db_path=None,
        feature_exchange="binance",
        feature_lookup=ordered_lookup,
        market_window_parity_v2_enabled=True,
    )
    strategy = ArtifactPortfolioModeStrategy(
        handler,
        events,
        portfolio_mode=seal.expected_definition.portfolio_mode,
        decision_cadence_seconds=1,
    )
    repeated = seal_alpha_max_manifest_activation(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted,
    )
    if (
        repeated != seal
        or handler.symbol_list is not admitted
        or getattr(handler, "_feature_lookup", None) is not ordered_lookup
        or strategy.required_timeframes != seal.expected_definition.native_timeframes
    ):
        raise _activation_mismatch()
    _assert_definition_matches(strategy, seal)
    _assert_child_identities(strategy, admitted, seal.expected_definition)

    if phase_id == "warmup":
        if prior_indicator_capsule is not None:
            raise AlphaMaxRuntimeContractError("alpha_max_warmup_prior_capsule_forbidden")
    else:
        if type(prior_indicator_capsule) is not AlphaMaxIndicatorCapsule:
            raise TypeError("alpha_max_prior_indicator_capsule_required")
        restored = _validate_alpha_max_indicator_capsule(
            prior_indicator_capsule,
            seal=seal,
            expected_phase_id=_alpha_max_capsule_predecessor(phase_id),
        )
        strategy.set_research_indicator_state(copy.deepcopy(restored))
        strategy.validate_research_warmup_ready()
        actual_state = strategy.get_research_indicator_state()
        if type(actual_state) is not dict or _canonical_bytes(actual_state) != _canonical_bytes(
            restored
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_restore_mismatch")

    aggregator = TimeframeAggregator(timeframes=list(strategy.required_timeframes))
    windows_processed = 0 if carry is None else carry.windows_processed
    discarded_signals = 0 if carry is None else carry.discarded_signal_count
    if carry is not None:
        strategy.set_state(copy.deepcopy(carry.strategy_state))
        aggregator.set_state(copy.deepcopy(carry.aggregator_state))
        if not _exact_state_equal(
            strategy.get_state(), carry.strategy_state
        ) or not _exact_state_equal(aggregator.get_state(), carry.aggregator_state):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_restore_mismatch")
    day_start = initial_day_start
    while day_start < end_utc:
        if day_start != initial_day_start:
            if bounded_raw_loader is None:
                break
            strategy_state = copy.deepcopy(strategy.get_state())
            aggregator_state = copy.deepcopy(aggregator.get_state())
            previous_handler = handler
            previous_strategy = strategy
            day_end = min(day_start + timedelta(days=1), end_utc)
            next_data = bounded_raw_loader.load_day(day_start, day_end)
            events = FastQueue()
            handler = HistoricParquetWindowedDataHandler(
                events,
                _require_exact_explicit_path(raw_root),
                admitted,
                day_start,
                day_end,
                next_data,
                backtest_poll_seconds=1,
                backtest_window_seconds=1,
                feature_db_path=None,
                feature_exchange="binance",
                feature_lookup=ordered_lookup,
                market_window_parity_v2_enabled=True,
            )
            strategy = ArtifactPortfolioModeStrategy(
                handler,
                events,
                portfolio_mode=seal.expected_definition.portfolio_mode,
                decision_cadence_seconds=1,
            )
            strategy.set_state(copy.deepcopy(strategy_state))
            aggregator = TimeframeAggregator(timeframes=list(strategy.required_timeframes))
            aggregator.set_state(copy.deepcopy(aggregator_state))
            repeated = seal_alpha_max_manifest_activation(
                preflight,
                output_root=output_root,
                phase=phase,
                manifest_path=manifest_path,
                admitted_symbols=admitted,
            )
            if (
                repeated != seal
                or handler is previous_handler
                or strategy is previous_strategy
                or handler.symbol_list is not admitted
                or getattr(handler, "_feature_lookup", None) is not ordered_lookup
                or not _exact_state_equal(strategy.get_state(), strategy_state)
                or not _exact_state_equal(aggregator.get_state(), aggregator_state)
            ):
                raise _activation_mismatch()
            _assert_definition_matches(strategy, seal)
            _assert_child_identities(strategy, admitted, seal.expected_definition)
        while handler.continue_backtest:
            handler.update_bars()
            while True:
                try:
                    event = events.get(False)
                except queue.Empty:
                    break
                if str(getattr(event, "type", "")).upper() != "MARKET_WINDOW":
                    raise AlphaMaxRuntimeContractError("alpha_max_warmup_handler_event_invalid")
                bars_1s = getattr(event, "bars_1s", {}) or {}
                aggregator.update_from_1s_batch(bars_1s)
                context = StrategyInputContext(
                    event=event,
                    aggregator=aggregator,
                    feature_lookup=ordered_lookup,
                    data_handler=handler,
                    execution_handler=None,
                    exchange=None,
                    provider_metadata={
                        "data_handler_class": type(handler).__name__,
                        "execution_handler_class": None,
                        "market_data_source": "alpha_max_indicator_only",
                    },
                )
                strategy.calculate_signals_context(context)
                windows_processed += 1
                discarded_signals += _drain_indicator_events(events)
        day_end = min(day_start + timedelta(days=1), end_utc)
        if checkpoint_store is not None and day_end < end_utc:
            if handler.continue_backtest or not events.empty():
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_queue_not_empty")
            checkpoint_store.seal(
                _AlphaMaxIndicatorDayCarry(
                    next_day_start_utc=day_end,
                    strategy_state=copy.deepcopy(strategy.get_state()),
                    aggregator_state=copy.deepcopy(aggregator.get_state()),
                    windows_processed=windows_processed,
                    discarded_signal_count=discarded_signals,
                )
            )
        day_start += timedelta(days=1)

    finalization = _finalize_alpha_max_native_boundary(
        strategy,
        seal.expected_definition,
        watermark,
        admitted_symbol_count=len(admitted_symbols),
        require_exact_counts=True,
    )
    finalized = dict(finalization.native_coverage_by_child)
    discarded_signals += finalization.discarded_signal_count
    strategy.validate_research_warmup_ready()
    discarded_signals += _drain_indicator_events(events)
    raw_capsule = strategy.get_research_indicator_state()
    if type(raw_capsule) is not dict:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_invalid")
    capsule_sha = str(raw_capsule.get("sha256") or "")
    capsule_scope = {key: value for key, value in raw_capsule.items() if key != "sha256"}
    if capsule_sha != _sha256(_canonical_bytes(capsule_scope)):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_hash_mismatch")
    frozen_capsule = _freeze_json(raw_capsule)
    frozen_finalized = _freeze_json(finalized)
    if not isinstance(frozen_capsule, Mapping) or not isinstance(frozen_finalized, Mapping):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_invalid")
    return AlphaMaxIndicatorCapsule(
        portfolio_mode=seal.expected_definition.portfolio_mode,
        phase_id=phase_id,
        manifest_sha256=seal.manifest_receipt.sha256,
        capsule_sha256=capsule_sha,
        capsule=frozen_capsule,
        finalized_children=frozen_finalized,
        native_finalization_sha256=finalization.sha256,
        windows_processed=(
            windows_processed
            + (0 if prior_indicator_capsule is None else prior_indicator_capsule.windows_processed)
        ),
        discarded_signal_count=(
            discarded_signals
            + (
                0
                if prior_indicator_capsule is None
                else prior_indicator_capsule.discarded_signal_count
            )
        ),
    )


class _AlphaMaxIndicatorBarsProxy:
    """Minimal bars capability for exact completed-native indicator replay."""

    __slots__ = ("_feature_lookup", "symbol_list")

    def __init__(
        self,
        admitted_symbols: tuple[str, ...],
        ordered_lookup: AlphaMaxOrderedFundingLookup,
    ) -> None:
        self.symbol_list = admitted_symbols
        self._feature_lookup = ordered_lookup


def _alpha_max_assert_indicator_checkpoint_queues_empty(
    strategy: ArtifactPortfolioModeStrategy, events: FastQueue
) -> None:
    if not events.empty():
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_queue_not_empty")
    children = getattr(strategy, "_children", None)
    if type(children) is not list:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_child_queue_invalid")
    for entry in children:
        if type(entry) is not tuple or len(entry) != 3:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_child_queue_invalid")
        empty = getattr(entry[2], "empty", None)
        if not callable(empty) or empty() is not True:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_queue_not_empty")


def _alpha_max_indicator_checkpoint_aggregator_state(
    aggregator: TimeframeAggregator,
) -> dict[str, Any]:
    state = copy.deepcopy(aggregator.get_state())
    history = state.get("history")
    if not isinstance(history, dict):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_aggregator_invalid")
    for by_timeframe in history.values():
        if not isinstance(by_timeframe, dict):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_aggregator_invalid")
        # The exact-native fold repopulates the complete rolling 1s tail from
        # every full UTC day before Python replays native releases. Only the
        # completed required-timeframe histories and working bars cross days.
        by_timeframe.pop("1s", None)
    return state


def _alpha_max_native_release_groups(
    releases: tuple[NativeBarRelease, ...],
    *,
    admitted_symbols: tuple[str, ...],
    required_timeframes: tuple[str, ...],
    start_ms: int,
    end_ms: int,
    carried_working_state: bool = False,
) -> tuple[
    tuple[
        int,
        Mapping[str, Mapping[str, tuple[Any, float, float, float, float, float]]],
    ],
    ...,
]:
    grouped: dict[
        int,
        dict[str, dict[str, tuple[Any, float, float, float, float, float]]],
    ] = {}
    expected_counts = {
        timeframe: ((end_ms - start_ms) // int(timeframe_to_milliseconds(timeframe)))
        - (0 if carried_working_state else 1)
        for timeframe in required_timeframes
    }
    if any(count < 0 for count in expected_counts.values()):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_release_count_invalid")
    observed_counts = {
        (symbol, timeframe): 0 for symbol in admitted_symbols for timeframe in required_timeframes
    }
    for release in releases:
        if (
            type(release) is not NativeBarRelease
            or release.symbol not in admitted_symbols
            or release.timeframe not in required_timeframes
            or not (
                (start_ms <= release.release_timestamp_ms < end_ms)
                if carried_working_state
                else (start_ms < release.release_timestamp_ms < end_ms)
            )
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_release_invalid")
        by_timeframe = grouped.setdefault(release.release_timestamp_ms, {})
        by_symbol = by_timeframe.setdefault(release.timeframe, {})
        if release.symbol in by_symbol:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_release_duplicate")
        by_symbol[release.symbol] = release.bar
        observed_counts[(release.symbol, release.timeframe)] += 1

    output: list[
        tuple[
            int,
            Mapping[str, Mapping[str, tuple[Any, float, float, float, float, float]]],
        ]
    ] = []
    for release_timestamp_ms in sorted(grouped):
        ordered_timeframes: dict[
            str,
            Mapping[str, tuple[Any, float, float, float, float, float]],
        ] = {}
        for timeframe in required_timeframes:
            raw_bars = grouped[release_timestamp_ms].get(timeframe)
            if raw_bars is None:
                continue
            if set(raw_bars) != set(admitted_symbols):
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_indicator_native_release_cross_section_invalid"
                )
            ordered_timeframes[timeframe] = MappingProxyType(
                {symbol: raw_bars[symbol] for symbol in admitted_symbols}
            )
        if set(ordered_timeframes) != set(grouped[release_timestamp_ms]):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_release_invalid")
        output.append((release_timestamp_ms, MappingProxyType(ordered_timeframes)))
    if any(
        observed_counts[(symbol, timeframe)] != expected_counts[timeframe]
        for symbol in admitted_symbols
        for timeframe in required_timeframes
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_release_count_invalid")
    return tuple(output)


def _alpha_max_validate_final_native_working_bars(
    aggregator: TimeframeAggregator,
    *,
    admitted_symbols: tuple[str, ...],
    required_timeframes: tuple[str, ...],
    end_ms: int,
) -> None:
    state = aggregator.get_state()
    working = state.get("working")
    if not isinstance(working, Mapping):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_working_invalid")
    for symbol in admitted_symbols:
        by_timeframe = working.get(symbol)
        if not isinstance(by_timeframe, Mapping):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_working_invalid")
        for timeframe in required_timeframes:
            value = by_timeframe.get(timeframe)
            expected_start_ms = end_ms - int(timeframe_to_milliseconds(timeframe))
            raw_time = value.get("time") if isinstance(value, Mapping) else None
            observed_start_ms = (
                int(
                    (
                        raw_time.replace(tzinfo=UTC)
                        if isinstance(raw_time, datetime) and raw_time.tzinfo is None
                        else raw_time.astimezone(UTC)
                    ).timestamp()
                    * 1000
                )
                if isinstance(raw_time, datetime)
                else -1
            )
            if (
                not isinstance(value, Mapping)
                or int(value.get("bucket_ms", -1)) != expected_start_ms
                or observed_start_ms != expected_start_ms
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_working_invalid")


def _build_alpha_max_indicator_capsule_exact_native(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_path: str | os.PathLike[str],
    admitted_symbols: tuple[str, ...],
    phase_id: str,
    raw_root: str | os.PathLike[str],
    ordered_lookup: AlphaMaxOrderedFundingLookup,
    watermark: object,
    prior_indicator_capsule: AlphaMaxIndicatorCapsule | None,
    bounded_raw_loader: _AlphaMaxBoundedRawLoader,
    checkpoint_store: _AlphaMaxIndicatorDayCheckpointStore | None = None,
    checkpoint_candidate_identity: Mapping[str, object] | None = None,
) -> AlphaMaxIndicatorCapsule:
    """Replay only exact completed-native releases from authenticated raw rows."""
    reject_ambient_lq_environment()
    _validate_preflight(preflight)
    admitted = _validate_admitted_symbols(preflight, admitted_symbols)
    if type(ordered_lookup) is not AlphaMaxOrderedFundingLookup:
        raise TypeError("alpha_max_ordered_lookup_identity_invalid")
    if type(bounded_raw_loader) is not _AlphaMaxBoundedRawLoader:
        raise TypeError("alpha_max_bounded_raw_loader_identity_invalid")
    if ordered_lookup.ordered_root_ids != _alpha_max_expected_root_sequence(phase_id):
        raise AlphaMaxRuntimeContractError("alpha_max_feature_root_sequence_mismatch")
    if (
        bounded_raw_loader.seal.path != _require_exact_explicit_path(raw_root)
        or bounded_raw_loader.seal.root_id != _alpha_max_current_root_id(phase_id)
        or bounded_raw_loader.seal.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_loader_scope_invalid")
    seal = seal_alpha_max_manifest_activation(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted,
    )
    config = build_alpha_max_backtest_config(
        preflight,
        phase_id=phase_id,
        admitted_symbols=admitted,
        nominal_cost_bps=20,
    )
    if _alpha_max_watermark_ms(watermark) != _alpha_max_watermark_ms(config.END_DATE):
        raise AlphaMaxRuntimeContractError("alpha_max_warmup_watermark_mismatch")
    start_utc = datetime.fromisoformat(config.START_DATE.replace("Z", "+00:00")).astimezone(UTC)
    end_utc = datetime.fromisoformat(config.END_DATE.replace("Z", "+00:00")).astimezone(UTC)
    end_ms = int(end_utc.timestamp() * 1000)
    required_timeframes = seal.expected_definition.native_timeframes
    if not required_timeframes or any(
        _ALPHA_MAX_NATIVE_TIMEFRAME_BY_CLASS.get(component.strategy_class)
        not in required_timeframes
        for component in seal.expected_definition.components
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_mode_unsupported")

    events = FastQueue()
    bars = _AlphaMaxIndicatorBarsProxy(admitted, ordered_lookup)
    strategy = ArtifactPortfolioModeStrategy(
        bars,
        events,
        portfolio_mode=seal.expected_definition.portfolio_mode,
        decision_cadence_seconds=1,
    )
    repeated = seal_alpha_max_manifest_activation(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted,
    )
    if (
        repeated != seal
        or bars.symbol_list is not admitted
        or bars._feature_lookup is not ordered_lookup
        or strategy.required_timeframes != required_timeframes
    ):
        raise _activation_mismatch()
    _assert_definition_matches(strategy, seal)
    _assert_child_identities(strategy, admitted, seal.expected_definition)

    if phase_id == "warmup":
        if prior_indicator_capsule is not None:
            raise AlphaMaxRuntimeContractError("alpha_max_warmup_prior_capsule_forbidden")
    else:
        if type(prior_indicator_capsule) is not AlphaMaxIndicatorCapsule:
            raise TypeError("alpha_max_prior_indicator_capsule_required")
        restored = _validate_alpha_max_indicator_capsule(
            prior_indicator_capsule,
            seal=seal,
            expected_phase_id=_alpha_max_capsule_predecessor(phase_id),
        )
        strategy.set_research_indicator_state(copy.deepcopy(restored))
        strategy.validate_research_warmup_ready()
        actual_state = strategy.get_research_indicator_state()
        if type(actual_state) is not dict or _canonical_bytes(actual_state) != _canonical_bytes(
            restored
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_restore_mismatch")

    carry = None
    if checkpoint_store is not None:
        if type(checkpoint_store) is not _AlphaMaxIndicatorDayCheckpointStore:
            raise TypeError("alpha_max_indicator_checkpoint_store_invalid")
        if type(checkpoint_candidate_identity) is not dict:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_candidate_identity_mismatch"
            )
        expected_descriptor = _alpha_max_indicator_day_checkpoint_descriptor(
            preflight,
            checkpoint_root=checkpoint_store.root,
            output_root=output_root,
            phase=phase,
            manifest_path=manifest_path,
            admitted_symbols=admitted,
            phase_id=phase_id,
            raw_root=raw_root,
            ordered_lookup=ordered_lookup,
            watermark=watermark,
            bounded_raw_loader=bounded_raw_loader,
            checkpoint_candidate_identity=checkpoint_candidate_identity,
        )
        if _canonical_bytes(expected_descriptor) + b"\n" != checkpoint_store._descriptor_bytes:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_descriptor_mismatch")
        # The journal is consulted before any raw partition is opened.
        carry = checkpoint_store.load_latest(start_utc=start_utc, end_utc=end_utc)
    aggregator = TimeframeAggregator(timeframes=list(required_timeframes))
    windows_processed = 0 if carry is None else carry.windows_processed
    discarded_signals = 0 if carry is None else carry.discarded_signal_count
    if carry is not None:
        strategy.set_state(copy.deepcopy(carry.strategy_state))
        aggregator.set_state(copy.deepcopy(carry.aggregator_state))
        if not _exact_state_equal(
            strategy.get_state(), carry.strategy_state
        ) or not _exact_state_equal(aggregator.get_state(), carry.aggregator_state):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_restore_mismatch")

    day_start = start_utc if carry is None else carry.next_day_start_utc
    if day_start >= end_utc:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_checkpoint_terminal_carry_invalid")
    while day_start < end_utc:
        day_end = day_start + timedelta(days=1)
        releases, day_windows = bounded_raw_loader.fold_exact_indicator_phase(
            aggregator, start=day_start, end=day_end
        )
        release_groups = _alpha_max_native_release_groups(
            releases,
            admitted_symbols=admitted,
            required_timeframes=required_timeframes,
            start_ms=int(day_start.timestamp() * 1000),
            end_ms=int(day_end.timestamp() * 1000),
            carried_working_state=day_start != start_utc or carry is not None,
        )
        if day_windows != 86_400:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_window_count_invalid")
        windows_processed += day_windows
        for release_timestamp_ms, bars_by_timeframe in release_groups:
            strategy.calculate_signals_completed_native_release(
                release_timestamp_ms=release_timestamp_ms,
                bars_by_timeframe=bars_by_timeframe,
                feature_lookup=ordered_lookup,
            )
            discarded_signals += _drain_indicator_events(events)
        if checkpoint_store is not None and day_end < end_utc:
            _alpha_max_assert_indicator_checkpoint_queues_empty(strategy, events)
            checkpoint_store.seal(
                _AlphaMaxIndicatorDayCarry(
                    next_day_start_utc=day_end,
                    strategy_state=copy.deepcopy(strategy.get_state()),
                    aggregator_state=_alpha_max_indicator_checkpoint_aggregator_state(aggregator),
                    windows_processed=windows_processed,
                    discarded_signal_count=discarded_signals,
                )
            )
        day_start = day_end
    _alpha_max_validate_final_native_working_bars(
        aggregator,
        admitted_symbols=admitted,
        required_timeframes=required_timeframes,
        end_ms=end_ms,
    )

    # The one-second oracle hands the final forming native bars and feature
    # lookup to every child at the timestamp of the last real raw row.  Reuse
    # that exact timestamp here: an artificial end-boundary strategy decision
    # would change causality, while omitting the handoff leaves finalizers
    # detached from the authenticated aggregator.
    last_raw_timestamp_ms = end_ms - 1000
    final_event = build_market_window_event(
        time=last_raw_timestamp_ms,
        window_seconds=1,
        bars_1s={},
        event_time_watermark_ms=last_raw_timestamp_ms,
        parity_v2_enabled=True,
        emit_metrics=False,
    )
    strategy.calculate_signals_context(
        StrategyInputContext(
            event=final_event,
            aggregator=aggregator,
            feature_lookup=ordered_lookup,
            data_handler=bars,
            execution_handler=None,
            exchange=None,
            provider_metadata={
                "data_handler_class": type(bars).__name__,
                "execution_handler_class": None,
                "market_data_source": "alpha_max_exact_native_final_handoff",
            },
        )
    )
    discarded_signals += _drain_indicator_events(events)

    repeated = seal_alpha_max_manifest_activation(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted,
    )
    if repeated != seal:
        raise _activation_mismatch()

    finalization = _finalize_alpha_max_native_boundary(
        strategy,
        seal.expected_definition,
        watermark,
        admitted_symbol_count=len(admitted),
        require_exact_counts=True,
    )
    finalized = dict(finalization.native_coverage_by_child)
    discarded_signals += finalization.discarded_signal_count
    strategy.validate_research_warmup_ready()
    discarded_signals += _drain_indicator_events(events)
    raw_capsule = strategy.get_research_indicator_state()
    if type(raw_capsule) is not dict:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_invalid")
    capsule_sha = str(raw_capsule.get("sha256") or "")
    capsule_scope = {key: value for key, value in raw_capsule.items() if key != "sha256"}
    if capsule_sha != _sha256(_canonical_bytes(capsule_scope)):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_hash_mismatch")
    frozen_capsule = _freeze_json(raw_capsule)
    frozen_finalized = _freeze_json(finalized)
    if not isinstance(frozen_capsule, Mapping) or not isinstance(frozen_finalized, Mapping):
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_capsule_invalid")
    repeated = seal_alpha_max_manifest_activation(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted,
    )
    if repeated != seal:
        raise _activation_mismatch()
    return AlphaMaxIndicatorCapsule(
        portfolio_mode=seal.expected_definition.portfolio_mode,
        phase_id=phase_id,
        manifest_sha256=seal.manifest_receipt.sha256,
        capsule_sha256=capsule_sha,
        capsule=frozen_capsule,
        finalized_children=frozen_finalized,
        native_finalization_sha256=finalization.sha256,
        windows_processed=(
            windows_processed
            + (0 if prior_indicator_capsule is None else prior_indicator_capsule.windows_processed)
        ),
        discarded_signal_count=(
            discarded_signals
            + (
                0
                if prior_indicator_capsule is None
                else prior_indicator_capsule.discarded_signal_count
            )
        ),
    )


def build_alpha_max_indicator_capsule(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_path: str | os.PathLike[str],
    admitted_symbols: tuple[str, ...],
    phase_id: str,
    raw_root: str | os.PathLike[str],
    ordered_lookup: AlphaMaxOrderedFundingLookup,
    watermark: object,
    data_dict: Mapping[str, object] | None = None,
    prior_indicator_capsule: AlphaMaxIndicatorCapsule | None = None,
    bounded_raw_loader: _AlphaMaxBoundedRawLoader | None = None,
    checkpoint_store: _AlphaMaxIndicatorDayCheckpointStore | None = None,
    checkpoint_candidate_identity: Mapping[str, object] | None = None,
) -> AlphaMaxIndicatorCapsule:
    """Prime exact native indicators without constructing economic events."""
    if bounded_raw_loader is None:
        if checkpoint_store is not None:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_checkpoint_requires_bounded_native_loader"
            )
        return _build_alpha_max_indicator_capsule_incremental(
            preflight,
            output_root=output_root,
            phase=phase,
            manifest_path=manifest_path,
            admitted_symbols=admitted_symbols,
            phase_id=phase_id,
            raw_root=raw_root,
            ordered_lookup=ordered_lookup,
            watermark=watermark,
            data_dict=data_dict,
            prior_indicator_capsule=prior_indicator_capsule,
            bounded_raw_loader=None,
            checkpoint_store=checkpoint_store,
        )
    if data_dict is not None:
        raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_loader_data_conflict")
    return _build_alpha_max_indicator_capsule_exact_native(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_path=manifest_path,
        admitted_symbols=admitted_symbols,
        phase_id=phase_id,
        raw_root=raw_root,
        ordered_lookup=ordered_lookup,
        watermark=watermark,
        prior_indicator_capsule=prior_indicator_capsule,
        bounded_raw_loader=bounded_raw_loader,
        checkpoint_store=checkpoint_store,
        checkpoint_candidate_identity=checkpoint_candidate_identity,
    )


def _alpha_max_fold_calendar(
    preflight: AlphaMaxRuntimePreflight, fold_id: str
) -> tuple[datetime, ...]:
    try:
        window = preflight.phase_windows[fold_id]
    except KeyError as exc:
        raise AlphaMaxRuntimeContractError(f"alpha_max_phase_window_unknown:{fold_id}") from exc
    start = datetime.fromisoformat(window.start_utc.replace("Z", "+00:00")).astimezone(UTC)
    end = datetime.fromisoformat(window.end_utc.replace("Z", "+00:00")).astimezone(UTC)
    count = int((end - start) / timedelta(hours=4))
    calendar = tuple(start + timedelta(hours=4 * index) for index in range(1, count + 1))
    if not calendar or calendar[-1] != end:
        raise AlphaMaxRuntimeContractError("alpha_max_fold_reporting_calendar_invalid")
    return calendar


def _alpha_max_first_true_index(mask: np.ndarray, *, offset: int) -> int | None:
    matches = np.flatnonzero(mask)
    return None if not matches.size else offset + int(matches[0])


def _alpha_max_next_tick_action_index(
    activation: AlphaMaxEngineActivation,
    view: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    start_index: int,
    end_index: int,
) -> int | None:
    """Return the earliest boundary, conditional fill, or liquidation second."""
    timestamps = view[activation.admitted_symbols[0]][0]
    scoped_timestamps = timestamps[start_index : end_index + 1]
    boundary = _alpha_max_first_true_index(
        scoped_timestamps % 14_400_000 == 0,
        offset=start_index,
    )
    candidates = [] if boundary is None else [boundary]
    execution = activation.backtest.execution_handler
    for order in execution.active_orders:
        if not isinstance(order, Mapping):
            return start_index
        symbol = str(order.get("symbol") or "")
        if symbol not in view:
            raise AlphaMaxRuntimeContractError("alpha_max_tick_order_symbol_invalid")
        order_type = str(order.get("type") or "").upper()
        direction = str(order.get("direction") or "").upper()
        if direction not in {"BUY", "SELL"}:
            candidates.append(start_index)
            continue
        numeric = view[symbol][1][start_index : end_index + 1]
        if order_type in {"MKT", "TRAIL_STOP"}:
            candidates.append(start_index)
            continue
        if order_type == "LMT":
            level = order.get("limit_price")
            if not isinstance(level, (int, float)) or isinstance(level, bool):
                candidates.append(start_index)
                continue
            mask = (
                numeric[:, 2] < float(level) if direction == "BUY" else numeric[:, 1] > float(level)
            )
        elif order_type == "STOP":
            level = order.get("stop_price")
            if not isinstance(level, (int, float)) or isinstance(level, bool):
                candidates.append(start_index)
                continue
            mask = (
                numeric[:, 2] <= float(level)
                if direction == "SELL"
                else numeric[:, 1] >= float(level)
            )
        elif order_type == "TAKE_PROFIT":
            level = order.get("stop_price")
            if not isinstance(level, (int, float)) or isinstance(level, bool):
                candidates.append(start_index)
                continue
            mask = (
                numeric[:, 1] >= float(level)
                if direction == "SELL"
                else numeric[:, 2] <= float(level)
            )
        else:
            candidates.append(start_index)
            continue
        trigger = _alpha_max_first_true_index(mask, offset=start_index)
        if trigger is not None:
            candidates.append(trigger)

    portfolio = activation.backtest.portfolio
    for symbol in activation.admitted_symbols:
        quantity = float(portfolio.current_positions.get(symbol, 0.0))
        entry_price = portfolio.entry_prices.get(symbol)
        if (
            abs(quantity) < 1e-12
            or symbol in portfolio._pending_liquidation
            or not isinstance(entry_price, (int, float))
            or isinstance(entry_price, bool)
            or float(entry_price) <= 0.0
        ):
            continue
        liquidation_price = portfolio._cached_liquidation_price(
            symbol,
            quantity,
            float(entry_price),
        )
        if liquidation_price is None:
            continue
        numeric = view[symbol][1][start_index : end_index + 1]
        mask = (
            numeric[:, 2] <= liquidation_price
            if quantity > 0.0
            else numeric[:, 1] >= liquidation_price
        )
        trigger = _alpha_max_first_true_index(mask, offset=start_index)
        if trigger is not None:
            candidates.append(trigger)
    return min(candidates) if candidates else None


def _alpha_max_feed_exact_tick_rows(
    activation: AlphaMaxEngineActivation,
    view: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    start_index: int,
    end_index: int,
) -> tuple[NativeBarRelease, ...]:
    aggregator = activation.backtest._ensure_timeframe_aggregator()
    releases: list[NativeBarRelease] = []
    for symbol in activation.admitted_symbols:
        timestamps, numeric = view[symbol]
        rows = (
            (
                int(timestamps[index]),
                float(numeric[index, 0]),
                float(numeric[index, 1]),
                float(numeric[index, 2]),
                float(numeric[index, 3]),
                float(numeric[index, 4]),
            )
            for index in range(start_index, end_index + 1)
        )
        releases.extend(aggregator.update_from_canonical_1s_rows_exact(symbol, rows))
    return tuple(releases)


def _alpha_max_replay_tick_releases(
    activation: AlphaMaxEngineActivation,
    releases: tuple[NativeBarRelease, ...],
    *,
    release_timestamp_ms: int,
) -> None:
    if not releases:
        return
    grouped: dict[
        str,
        dict[str, tuple[Any, float, float, float, float, float]],
    ] = {}
    for release in releases:
        if release.release_timestamp_ms != release_timestamp_ms:
            raise AlphaMaxRuntimeContractError("alpha_max_tick_native_release_timing_invalid")
        by_symbol = grouped.setdefault(release.timeframe, {})
        if release.symbol in by_symbol:
            raise AlphaMaxRuntimeContractError("alpha_max_tick_native_release_duplicate")
        by_symbol[release.symbol] = release.bar
    ordered: dict[
        str,
        Mapping[str, tuple[Any, float, float, float, float, float]],
    ] = {}
    for timeframe in activation.backtest.strategy.required_timeframes:
        bars = grouped.get(timeframe)
        if bars is None:
            continue
        if set(bars) != set(activation.admitted_symbols):
            raise AlphaMaxRuntimeContractError("alpha_max_tick_native_release_incomplete")
        ordered[timeframe] = MappingProxyType(
            {symbol: bars[symbol] for symbol in activation.admitted_symbols}
        )
    if set(ordered) != set(grouped):
        raise AlphaMaxRuntimeContractError("alpha_max_tick_native_release_timeframe_invalid")
    activation.backtest.strategy.calculate_signals_completed_native_release(
        release_timestamp_ms=release_timestamp_ms,
        bars_by_timeframe=MappingProxyType(ordered),
        feature_lookup=activation.ordered_lookup,
    )


def _alpha_max_advance_inert_ticks(
    activation: AlphaMaxEngineActivation,
    view: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    start_index: int,
    end_index: int,
) -> None:
    if end_index < start_index:
        return
    timestamps = np.asarray(
        view[activation.admitted_symbols[0]][0][start_index : end_index + 1],
        dtype=np.int64,
    )
    closes = {
        symbol: np.asarray(view[symbol][1][start_index : end_index + 1, 3], dtype=np.float64)
        for symbol in activation.admitted_symbols
    }
    activation.backtest.portfolio.update_timeindex_inert_batch(timestamps, closes)
    count = end_index - start_index + 1
    activation.backtest.market_events += count * len(activation.admitted_symbols)
    sequence = activation.backtest._event_sequencer.get_state()["sequence"]
    activation.backtest._event_sequencer.set_state({"sequence": sequence + count})
    activation.backtest._window_decision_last_bucket = int(timestamps[-1]) // 1000


def _alpha_max_process_tick_action(
    activation: AlphaMaxEngineActivation,
    view: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    action_index: int,
    releases: tuple[NativeBarRelease, ...],
) -> None:
    handler = activation.backtest.data_handler
    timestamp_ms = int(view[activation.admitted_symbols[0]][0][action_index])
    _alpha_max_replay_tick_releases(
        activation,
        releases,
        release_timestamp_ms=timestamp_ms,
    )
    bars_1s = {
        symbol: (
            (
                timestamp_ms,
                *(float(value) for value in view[symbol][1][action_index]),
            ),
        )
        for symbol in activation.admitted_symbols
    }
    event = build_market_window_event(
        time=timestamp_ms,
        window_seconds=1,
        bars_1s=bars_1s,
        event_time_watermark_ms=timestamp_ms,
        commit_id=None,
        lag_ms=0,
        is_stale=False,
        parity_v2_enabled=True,
        metrics_log_path=handler._metrics_log_path,
        emit_metrics=False,
    )
    activation.backtest.process_event(event)
    while True:
        try:
            queued = activation.backtest.events.get(False)
        except queue.Empty:
            break
        activation.backtest.process_event(queued)


def _run_alpha_max_exact_tick_reducer(activation: AlphaMaxEngineActivation) -> None:
    """Run one validated day with exact event semantics and inert mark batches."""
    handler = activation.backtest.data_handler
    if type(handler) is not HistoricParquetWindowedDataHandler:
        raise TypeError("alpha_max_tick_handler_identity_invalid")
    if (
        activation.backtest.record_history
        or activation.backtest.track_metrics
        or activation.backtest.record_trades
        or activation.backtest.portfolio.strategy_quality.enabled
    ):
        activation.backtest._run_backtest()
        return
    retained_aggregator = activation.backtest.timeframe_aggregator
    retained_aggregator_state = (
        None if retained_aggregator is None else copy.deepcopy(retained_aggregator.get_state())
    )
    activation.backtest.timeframe_aggregator = TimeframeAggregator(
        timeframes=list(activation.backtest.strategy.required_timeframes),
        lookbacks=activation.backtest._resolve_required_lookbacks(),
    )
    if retained_aggregator_state is not None:
        activation.backtest.timeframe_aggregator.set_state(retained_aggregator_state)
    try:
        view = handler.alpha_max_exact_columnar_view()
    except (TypeError, ValueError) as exc:
        if str(exc) not in {
            "alpha_max_columnar_view_state_invalid",
            "alpha_max_columnar_rows_identity_invalid",
            "alpha_max_columnar_rows_invalid",
            "alpha_max_columnar_timeline_mismatch",
        }:
            raise
        activation.backtest.timeframe_aggregator = retained_aggregator
        activation.backtest._run_backtest()
        return
    if tuple(view) != activation.admitted_symbols:
        raise AlphaMaxRuntimeContractError("alpha_max_tick_symbol_order_invalid")
    row_count = len(view[activation.admitted_symbols[0]][0])
    cursor = 0
    while cursor < row_count:
        action = _alpha_max_next_tick_action_index(
            activation,
            view,
            start_index=cursor,
            end_index=row_count - 1,
        )
        segment_end = row_count - 1 if action is None else action
        releases = _alpha_max_feed_exact_tick_rows(
            activation,
            view,
            start_index=cursor,
            end_index=segment_end,
        )
        if action is None:
            if releases:
                raise AlphaMaxRuntimeContractError("alpha_max_tick_unhandled_native_release")
            _alpha_max_advance_inert_ticks(
                activation,
                view,
                start_index=cursor,
                end_index=segment_end,
            )
            handler.alpha_max_advance_without_event(cursor, segment_end)
            cursor = row_count
            continue
        _alpha_max_advance_inert_ticks(
            activation,
            view,
            start_index=cursor,
            end_index=action - 1,
        )
        handler.alpha_max_advance_without_event(cursor, action)
        _alpha_max_process_tick_action(
            activation,
            view,
            action_index=action,
            releases=releases,
        )
        cursor = action + 1
    if handler.continue_backtest or not activation.backtest.events.empty():
        raise AlphaMaxRuntimeContractError("alpha_max_tick_reducer_incomplete")


def _replay_alpha_max_fold(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str,
    phase: str,
    manifest_receipt: AlphaMaxManifestReceipt,
    admitted_symbols: tuple[str, ...],
    row_id: str,
    domain: str,
    nominal_cost_bps: int,
    fold_input: _AlphaMaxFoldReplayInput,
    aggregate_tracker: AlphaMaxStreamingEquityTracker,
    aggregate_scale: float,
) -> tuple[AlphaMaxFoldRunEvidence, AlphaMaxNormalizedFoldSegmentEvidence | None]:
    """Replay one flat-start fold through fresh daily engines and compact it."""
    fold_id = fold_input.fold_id
    if fold_id not in _alpha_max_fold_ids(domain):
        raise AlphaMaxRuntimeContractError("alpha_max_fold_replay_domain_mismatch")
    if manifest_receipt.row_id != row_id or fold_input.capsule_receipt.row_id != row_id:
        raise AlphaMaxRuntimeContractError("alpha_max_fold_replay_artifact_row_mismatch")
    if (
        manifest_receipt.phase != phase
        or fold_input.capsule_receipt.phase != phase
        or fold_input.capsule_receipt.manifest_sha256 != manifest_receipt.sha256
        or fold_input.indicator_capsule.manifest_sha256 != manifest_receipt.sha256
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_fold_replay_artifact_scope_mismatch")
    if nominal_cost_bps not in ALPHA_MAX_COST_CELL_BPS:
        raise AlphaMaxRuntimeContractError("alpha_max_nominal_cost_cell_invalid")
    window = preflight.phase_windows[fold_id]
    fold_start = datetime.fromisoformat(window.start_utc.replace("Z", "+00:00")).astimezone(UTC)
    fold_end = datetime.fromisoformat(window.end_utc.replace("Z", "+00:00")).astimezone(UTC)
    if (fold_end - fold_start) % timedelta(days=1):
        raise AlphaMaxRuntimeContractError("alpha_max_fold_daily_partition_invalid")

    _validate_alpha_max_root_seals(
        raw_root=fold_input.raw_root,
        phase_id=fold_id,
        ordered_lookup=fold_input.ordered_lookup,
        raw_root_seals=fold_input.raw_root_seals,
        feature_root_seals=fold_input.feature_root_seals,
        required=True,
        repeat_hash=True,
    )
    collector = AlphaMaxAttributionCollector()
    fold_tracker = _AlphaMaxFoldEquityFanout(
        aggregate_tracker,
        aggregate_scale=aggregate_scale,
        reporting_start=fold_start,
        reporting_end=fold_end,
    )
    resolver = AlphaMaxFundingBoundaryResolver(
        fold_input.ordered_lookup,
        admitted_symbols,
    )
    carry: _AlphaMaxDailyCarry | None = None
    pricing_traces: list[ExecutionPricingTrace] = []
    market_event_count = 0
    signal_event_count = 0
    order_event_count = 0
    fill_event_count = 0
    starting_cash = 0.0
    starting_equity = 0.0
    starting_open_position_count = 0
    starting_open_order_count = 0
    final_portfolio: Portfolio | None = None
    final_execution: SimulatedExecutionHandler | None = None
    final_resolver: AlphaMaxFundingBoundaryResolver | None = None
    effective_config_bytes: bytes | None = None
    effective_config_sha256: str | None = None
    runtime_read_audit: tuple[str, ...] | None = None
    runtime_read_audit_sha256: str | None = None
    native_finalization: AlphaMaxNativeFinalizationReceipt | None = None
    target_gross_exposure: float | None = None

    day_start = fold_start
    while day_start < fold_end:
        day_end = day_start + timedelta(days=1)
        data_dict = fold_input.bounded_raw_loader.load_day(day_start, day_end)
        if carry is not None:
            resolver = resolver.carry_forward()
        activation = construct_alpha_max_engine(
            preflight,
            output_root=output_root,
            phase=phase,
            manifest_path=manifest_receipt.path,
            admitted_symbols=admitted_symbols,
            phase_id=fold_id,
            nominal_cost_bps=nominal_cost_bps,
            raw_root=fold_input.raw_root,
            ordered_lookup=fold_input.ordered_lookup,
            funding_resolver=resolver,
            data_dict=data_dict,
            attribution_collector=collector,
            full_event_equity_tracker=fold_tracker,
            indicator_capsule=fold_input.indicator_capsule,
            raw_root_seals=fold_input.raw_root_seals,
            feature_root_seals=fold_input.feature_root_seals,
            _repeat_root_hash_on_activation=False,
            _chunk_start_utc=day_start,
            _chunk_end_utc=day_end,
        )
        config = activation.constructor_plan.config
        current_target_gross = float(activation.artifact_seal.expected_definition.gross_cap)
        if target_gross_exposure is None:
            target_gross_exposure = current_target_gross
        elif not math.isclose(
            target_gross_exposure,
            current_target_gross,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_fold_target_gross_changed")
        current_effective_bytes = config.runtime_attribute_bytes()
        current_effective_sha = config.runtime_instance_sha256
        if effective_config_bytes is None:
            effective_config_bytes = current_effective_bytes
            effective_config_sha256 = current_effective_sha
        elif (
            current_effective_bytes != effective_config_bytes
            or current_effective_sha != effective_config_sha256
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_fold_runtime_config_changed")
        if carry is not None:
            _restore_alpha_max_daily_carry(activation, carry)
        fold_tracker.bind_backtest(activation.backtest)
        portfolio = activation.backtest.portfolio
        execution = activation.backtest.execution_handler
        if day_start == fold_start:
            starting_cash = float(portfolio.current_holdings["cash"])
            starting_equity = float(portfolio.current_holdings["total"])
            starting_open_position_count = sum(
                abs(float(value)) > 1e-12 for value in portfolio.current_positions.values()
            )
            starting_open_order_count = len(execution.active_orders)

        # This validation remains the final operation before event one for every day.
        validate_alpha_max_engine_activation(
            activation,
            _expected_daily_carry=carry,
        )
        _run_alpha_max_exact_tick_reducer(activation)

        market_event_count += int(activation.backtest.market_events)
        signal_event_count += int(activation.backtest.signals)
        order_event_count += int(activation.backtest.orders)
        fill_event_count += int(activation.backtest.fills)
        daily_pricing = execution.pricing_trace_evidence
        if type(daily_pricing) is not tuple or any(
            type(value) is not ExecutionPricingTrace for value in daily_pricing
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_pricing_trace_identity_invalid")
        pricing_traces.extend(daily_pricing)
        day_finalization = _settle_alpha_max_day_boundary(
            activation,
            fold_tracker,
            day_end,
            scoring_boundary=day_end == fold_end,
        )
        if day_finalization is not None:
            if native_finalization is not None:
                raise AlphaMaxRuntimeContractError("alpha_max_fold_native_finalization_duplicate")
            native_finalization = day_finalization
        # Bind the receipt to every attribute the fully validated engine and
        # explicit causal day-boundary settlement actually read.  Different
        # days may take different order/fill paths, so concatenate the post-run
        # audits in chronological day order rather than requiring equality.
        runtime_read_audit, runtime_read_audit_sha256 = _alpha_max_append_runtime_read_audit(
            runtime_read_audit,
            config.runtime_read_audit,
        )
        carry = _capture_alpha_max_daily_carry(activation)
        final_portfolio = portfolio
        final_execution = execution
        final_resolver = resolver
        day_start = day_end
        del activation, data_dict

    if (
        carry is None
        or final_portfolio is None
        or final_execution is None
        or final_resolver is None
        or effective_config_bytes is None
        or effective_config_sha256 is None
        or runtime_read_audit is None
        or runtime_read_audit_sha256 is None
        or native_finalization is None
        or target_gross_exposure is None
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_fold_replay_empty")
    applications = collector.applications
    pricing_tuple = tuple(pricing_traces)
    if len(pricing_tuple) != len(applications) or any(
        application.pricing_trace_hash != execution_pricing_trace_sha256(trace)
        for trace, application in zip(pricing_tuple, applications, strict=True)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_pricing_application_bijection_failed")
    _validate_alpha_max_root_seals(
        raw_root=fold_input.raw_root,
        phase_id=fold_id,
        ordered_lookup=fold_input.ordered_lookup,
        raw_root_seals=fold_input.raw_root_seals,
        feature_root_seals=fold_input.feature_root_seals,
        required=True,
        repeat_hash=True,
    )
    holdings = final_portfolio.current_holdings
    ending_cash = float(holdings["cash"])
    ending_equity = float(holdings["total"])
    if not math.isfinite(ending_cash) or not math.isfinite(ending_equity):
        raise AlphaMaxRuntimeContractError("alpha_max_replay_nonfinite_economics")
    full_event_equity = fold_tracker.finalize()
    receipt = build_alpha_max_actual_engine_run_receipt(
        row_id=row_id,
        domain=domain,
        split_or_fold_id=fold_id,
        nominal_cost_bps=nominal_cost_bps,
        raw_root_seals=fold_input.raw_root_seals,
        feature_root_seals=fold_input.feature_root_seals,
        capsule_receipt=fold_input.capsule_receipt,
        manifest_receipt=manifest_receipt,
        config_receipt=preflight.config_receipt,
        config_bytes=preflight.config_bytes,
        runtime_contract_bytes=preflight.runtime_contract_bytes,
        effective_config_bytes=effective_config_bytes,
        effective_config_sha256=effective_config_sha256,
        runtime_read_audit=runtime_read_audit,
        runtime_read_audit_sha256=runtime_read_audit_sha256,
        admitted_symbols=admitted_symbols,
        market_event_count=market_event_count,
        signal_event_count=signal_event_count,
        order_event_count=order_event_count,
        fill_event_count=fill_event_count,
        trade_count=int(final_portfolio.trade_count),
        starting_cash=starting_cash,
        starting_equity=starting_equity,
        starting_open_position_count=starting_open_position_count,
        starting_open_order_count=starting_open_order_count,
        starting_used_margin=0.0,
        ending_cash=ending_cash,
        ending_equity=ending_equity,
        full_event_equity=full_event_equity,
        native_finalization=native_finalization,
        pricing_traces=pricing_tuple,
        fill_applications=applications,
        no_fill_attempts=final_execution.no_fill_attempt_evidence,
        funding_ledger=final_resolver.ledger,
        liquidation_events=tuple(final_portfolio.liquidation_events),
        portfolio_fee_total=float(holdings.get("commission", 0.0)),
        portfolio_funding_total=float(holdings.get("funding", 0.0)),
        capacity_observations=tuple(
            {
                "bar_volume": value.bar_volume,
                "equity_before": value.equity_before,
                "raw_price": value.raw_price,
                "requested_qty": value.requested_qty,
            }
            for value in final_execution.capacity_observation_evidence
        ),
        ending_market_values={symbol: float(holdings[symbol]) for symbol in admitted_symbols},
        target_gross_exposure=target_gross_exposure,
    )
    primary_stream = (
        None
        if receipt.ruin_detected
        else build_alpha_max_primary_return_stream(
            fold_tracker.reporting_endpoints,
            _alpha_max_fold_calendar(preflight, fold_id),
        )
    )
    fold_run = build_alpha_max_fold_run_evidence(receipt, primary_stream)
    segment_tracker = fold_tracker.normalized_segment_tracker
    segment = (
        None
        if receipt.ruin_detected
        else build_alpha_max_normalized_fold_segment_evidence(
            fold_id=fold_id,
            source_streaming_equity_sha256=receipt.full_event_equity.sha256,
            source_event_stream_sha256=receipt.full_event_equity.event_stream_sha256,
            normalization_scale=aggregate_scale,
            normalized_starting_equity=aggregate_scale * 10_000.0,
            normalized_ending_equity=aggregate_scale * receipt.ending_equity,
            normalized_segment_event_stream_sha256=(segment_tracker.event_stream_sha256),
            event_count=receipt.full_event_equity.event_count,
            first_timestamp_ms=receipt.full_event_equity.first_timestamp_ms,  # type: ignore[arg-type]
            last_timestamp_ms=receipt.full_event_equity.last_timestamp_ms,  # type: ignore[arg-type]
            aggregate_prefix_event_count=aggregate_tracker.event_count,
            aggregate_prefix_event_stream_sha256=aggregate_tracker.event_stream_sha256,
        )
    )
    return fold_run, segment


def _alpha_max_append_runtime_read_audit(
    retained: tuple[str, ...] | None,
    current: tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    """Append one daily post-run audit without assuming identical control flow."""
    if type(current) is not tuple or any(
        type(name) is not str or name not in _RUNTIME_ATTRIBUTE_SET for name in current
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_runtime_read_audit_invalid")
    if retained is not None and (
        type(retained) is not tuple
        or any(type(name) is not str or name not in _RUNTIME_ATTRIBUTE_SET for name in retained)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_runtime_read_audit_invalid")
    combined = (*(() if retained is None else retained), *current)
    return combined, _sha256(_canonical_bytes(list(combined)))


def _replay_alpha_max_cost_cell_pre_gate(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_receipt: AlphaMaxManifestReceipt,
    admitted_symbols: tuple[str, ...],
    row_id: str,
    domain: str,
    nominal_cost_bps: int,
    fold_inputs: tuple[_AlphaMaxFoldReplayInput, ...],
) -> AlphaMaxCostCellPreGateEvidence:
    expected_fold_ids = _alpha_max_fold_ids(domain)
    if (
        type(fold_inputs) is not tuple
        or tuple(value.fold_id for value in fold_inputs) != expected_fold_ids
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_cost_cell_fold_sequence_invalid")
    aggregate_tracker = AlphaMaxStreamingEquityTracker()
    aggregate_endpoint = 10_000.0
    aggregate_terminal = False
    fold_runs: list[AlphaMaxFoldRunEvidence] = []
    normalized_segments: list[AlphaMaxNormalizedFoldSegmentEvidence] = []
    for fold_input in fold_inputs:
        target_aggregate_tracker = (
            AlphaMaxStreamingEquityTracker() if aggregate_terminal else aggregate_tracker
        )
        fold_run, segment = _replay_alpha_max_fold(
            preflight,
            output_root=_require_exact_explicit_path(output_root),
            phase=phase,
            manifest_receipt=manifest_receipt,
            admitted_symbols=admitted_symbols,
            row_id=row_id,
            domain=domain,
            nominal_cost_bps=nominal_cost_bps,
            fold_input=fold_input,
            aggregate_tracker=target_aggregate_tracker,
            aggregate_scale=(1.0 if aggregate_terminal else aggregate_endpoint / 10_000.0),
        )
        fold_runs.append(fold_run)
        if segment is not None:
            normalized_segments.append(segment)
        if fold_run.status == "ruin_detected":
            aggregate_terminal = True
        elif not aggregate_terminal:
            aggregate_endpoint = (
                aggregate_endpoint * fold_run.actual_engine_run.ending_equity / 10_000.0
            )
    aggregate_equity = None if aggregate_terminal else aggregate_tracker.finalize()
    if aggregate_equity is not None and aggregate_equity.event_count != sum(
        value.actual_engine_run.full_event_equity.event_count for value in fold_runs
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_combined_equity_event_count_mismatch")
    return build_alpha_max_cost_cell_pre_gate_evidence(
        tuple(fold_runs),
        aggregate_equity,
        None if aggregate_terminal else tuple(normalized_segments),
    )


def replay_alpha_max_cost_cell(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str | os.PathLike[str],
    phase: str,
    manifest_receipt: AlphaMaxManifestReceipt,
    admitted_symbols: tuple[str, ...],
    row_id: str,
    domain: str,
    nominal_cost_bps: int,
    fold_inputs: tuple[_AlphaMaxFoldReplayInput, ...],
    statistical_evidence: AlphaMaxStatisticalEvidence | None = None,
) -> AlphaMaxCostCellEvidence:
    """Return one typed logical cell backed by every fresh flat-start fold."""
    reject_ambient_lq_environment()
    pre_gate = _replay_alpha_max_cost_cell_pre_gate(
        preflight,
        output_root=output_root,
        phase=phase,
        manifest_receipt=manifest_receipt,
        admitted_symbols=admitted_symbols,
        row_id=row_id,
        domain=domain,
        nominal_cost_bps=nominal_cost_bps,
        fold_inputs=fold_inputs,
    )
    return build_alpha_max_cost_cell_evidence(
        pre_gate,
        statistical_evidence=statistical_evidence,
    )


_ALPHA_MAX_CURRENT_ROW_IDS: Final[tuple[str, ...]] = (
    "component_carry_1x",
    "component_near_high_1x",
    "component_trend_1x",
    "diagnostic_track_b_codex_lagged_leaf_router_grid",
    "full_equal_risk_1x",
    "full_equal_risk_scaled",
    "full_equal_weight_1x",
    "full_shrunk_hrp_1x",
    "full_shrunk_hrp_scaled",
    "incumbent_cross_asset_lead_lag_momentum",
    "incumbent_cross_candidate_hybrid_v3_5",
    "incumbent_track_a_dynamic_conviction_switch",
    "loo_equal_risk_omit_carry_1x",
    "loo_equal_risk_omit_near_high_1x",
    "loo_equal_risk_omit_trend_1x",
    "loo_equal_weight_omit_carry_1x",
    "loo_equal_weight_omit_near_high_1x",
    "loo_equal_weight_omit_trend_1x",
    "loo_shrunk_hrp_omit_carry_1x",
    "loo_shrunk_hrp_omit_near_high_1x",
    "loo_shrunk_hrp_omit_trend_1x",
)
_ALPHA_MAX_UNAVAILABLE_ROWS: Final[tuple[str, ...]] = (
    "incumbent_cross_asset_lead_lag_momentum",
    "incumbent_cross_candidate_hybrid_v3_5",
    "incumbent_track_a_dynamic_conviction_switch",
)
_ALPHA_MAX_DIAGNOSTIC_ROWS: Final[tuple[str, ...]] = (
    "diagnostic_track_b_codex_lagged_leaf_router_grid",
)
_ALPHA_MAX_RESOLVABLE_ROWS: Final[tuple[str, ...]] = tuple(
    row_id
    for row_id in _ALPHA_MAX_CURRENT_ROW_IDS
    if row_id not in {*_ALPHA_MAX_UNAVAILABLE_ROWS, *_ALPHA_MAX_DIAGNOSTIC_ROWS}
)


def _validate_alpha_max_replay_evidence(
    evidence: AlphaMaxCostCellEvidence,
    *,
    row_id: str,
    nominal_cost_bps: int,
) -> None:
    if type(evidence) is not AlphaMaxCostCellEvidence:
        raise TypeError("alpha_max_matrix_replay_evidence_identity_invalid")
    pre_gate = evidence.pre_gate_evidence
    if (
        evidence.row_id != row_id
        or evidence.domain != "validation"
        or evidence.nominal_cost_bps != nominal_cost_bps
        or evidence.status not in {"complete", "ruin_detected"}
        or evidence.evidence_tier != "actual_engine"
        or type(pre_gate) is not AlphaMaxCostCellPreGateEvidence
        or tuple(value.split_or_fold_id for value in pre_gate.fold_runs)
        != _ALPHA_MAX_VALIDATION_FOLD_IDS
        or any(
            type(value.actual_engine_run) is not AlphaMaxActualEngineRunReceipt
            for value in pre_gate.fold_runs
        )
        or (evidence.status == "complete" and evidence.selection_valid is not True)
        or (evidence.status == "ruin_detected" and evidence.selection_valid is not False)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_replay_evidence_incomplete")
    try:
        canonical = canonical_alpha_max_cost_cell_bytes(evidence)
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_replay_evidence_incomplete") from exc
    if not canonical:
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_replay_evidence_incomplete")


def orchestrate_alpha_max_status_matrix(
    current_nodes: Sequence[Mapping[str, object]],
    row_executor: _AlphaMaxRowExecutor,
) -> AlphaMaxMatrixResult:
    """Produce all 84 statuses while constructing only the 68 resolvable cells."""
    reject_ambient_lq_environment()
    if not callable(row_executor):
        raise TypeError("alpha_max_row_executor_required")
    nodes: dict[str, Mapping[str, object]] = {}
    for row in current_nodes:
        if not isinstance(row, Mapping):
            raise TypeError("alpha_max_current_trial_node_invalid")
        row_id = row.get("row_id")
        if type(row_id) is not str or not row_id or row_id in nodes:
            raise AlphaMaxRuntimeContractError("alpha_max_current_trial_registry_invalid")
        nodes[row_id] = row
    if tuple(sorted(nodes)) != _ALPHA_MAX_CURRENT_ROW_IDS:
        raise AlphaMaxRuntimeContractError("alpha_max_current_trial_registry_mismatch")

    statuses: list[AlphaMaxMatrixCellStatus] = []
    retained_evidence: list[AlphaMaxCostCellEvidence] = []
    retained_runs: list[AlphaMaxActualEngineRunReceipt] = []
    capsule_by_row_fold: dict[tuple[str, str], AlphaMaxCapsuleReceipt] = {}
    manifest_by_row: dict[str, AlphaMaxManifestReceipt] = {}
    for row_id in _ALPHA_MAX_CURRENT_ROW_IDS:
        if row_id in _ALPHA_MAX_UNAVAILABLE_ROWS:
            for nominal in ALPHA_MAX_COST_CELL_BPS:
                statuses.append(
                    AlphaMaxMatrixCellStatus(
                        row_id=row_id,
                        row_role="incumbent_unavailable",
                        nominal_cost_bps=nominal,
                        status="incumbent_replay_unavailable",
                        engine_constructed=False,
                        selection_eligible=False,
                        capsule_sha256=None,
                        manifest_sha256=None,
                    )
                )
            continue
        if row_id in _ALPHA_MAX_DIAGNOSTIC_ROWS:
            for nominal in ALPHA_MAX_COST_CELL_BPS:
                statuses.append(
                    AlphaMaxMatrixCellStatus(
                        row_id=row_id,
                        row_role="track_b_diagnostic",
                        nominal_cost_bps=nominal,
                        status="diagnostic_report_only",
                        engine_constructed=False,
                        selection_eligible=False,
                        capsule_sha256=None,
                        manifest_sha256=None,
                    )
                )
            continue

        for nominal in ALPHA_MAX_COST_CELL_BPS:
            evidence = row_executor(nodes[row_id], nominal)
            _validate_alpha_max_replay_evidence(
                evidence,
                row_id=row_id,
                nominal_cost_bps=nominal,
            )
            if any(evidence is retained for retained in retained_evidence):
                raise AlphaMaxRuntimeContractError("alpha_max_matrix_cell_evidence_reused")
            retained_evidence.append(evidence)
            pre_gate = evidence.pre_gate_evidence
            assert type(pre_gate) is AlphaMaxCostCellPreGateEvidence
            for fold_run in pre_gate.fold_runs:
                actual_run = fold_run.actual_engine_run
                if any(actual_run is retained for retained in retained_runs):
                    raise AlphaMaxRuntimeContractError("alpha_max_matrix_engine_reused")
                retained_runs.append(actual_run)
                capsule_key = (row_id, fold_run.split_or_fold_id)
                retained_capsule = capsule_by_row_fold.get(capsule_key)
                if (
                    retained_capsule is not None
                    and actual_run.capsule_receipt is not retained_capsule
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_row_fold_capsule_not_reused")
                capsule_by_row_fold[capsule_key] = actual_run.capsule_receipt
                retained_manifest = manifest_by_row.get(row_id)
                if (
                    retained_manifest is not None
                    and actual_run.manifest_receipt is not retained_manifest
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_row_manifest_not_reused")
                manifest_by_row[row_id] = actual_run.manifest_receipt
            capsule_sha = _sha256(
                _canonical_bytes(
                    [
                        {
                            "prefix_id": receipt.prefix_id,
                            "sha256": receipt.sha256,
                        }
                        for receipt in evidence.capsule_receipts
                    ]
                )
            )
            manifest_sha = evidence.manifest_receipts[0].sha256
            statuses.append(
                AlphaMaxMatrixCellStatus(
                    row_id=row_id,
                    row_role="resolvable_candidate",
                    nominal_cost_bps=nominal,
                    status="resolved_engine_cell_complete",
                    engine_constructed=True,
                    selection_eligible=evidence.selection_valid,
                    capsule_sha256=capsule_sha,
                    manifest_sha256=manifest_sha,
                    evidence=evidence,
                )
            )
    result = AlphaMaxMatrixResult(
        statuses=tuple(statuses),
        resolvable_row_ids=_ALPHA_MAX_RESOLVABLE_ROWS,
        unavailable_row_ids=_ALPHA_MAX_UNAVAILABLE_ROWS,
        diagnostic_row_ids=_ALPHA_MAX_DIAGNOSTIC_ROWS,
    )
    if len(result.statuses) != 84 or result.engine_cell_count != 68:
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_cardinality_mismatch")
    if len(retained_runs) != 816:
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_physical_fold_cardinality_mismatch")
    return result


def fit_alpha_max_nominal_20_allocators(
    component_ids: Sequence[str],
    returns_matrix: object,
    *,
    nominal_cost_bps: int,
    per_component_cap: float,
) -> Mapping[str, Mapping[str, float]]:
    """Fit the only data-derived allocators at the frozen nominal 20-bps cell."""
    reject_ambient_lq_environment()
    if nominal_cost_bps != 20:
        raise AlphaMaxRuntimeContractError("alpha_max_allocator_fit_not_nominal_20")
    from lumina_quant.research.alpha_max_evidence import (
        allocate_alpha_max_equal_risk,
        allocate_alpha_max_shrunk_hrp,
    )

    erc = allocate_alpha_max_equal_risk(
        component_ids,
        returns_matrix,
        per_component_cap=per_component_cap,
    )
    hrp = allocate_alpha_max_shrunk_hrp(
        component_ids,
        returns_matrix,
        per_component_cap=per_component_cap,
    )
    return MappingProxyType(
        {
            "equal_risk": MappingProxyType(dict(erc)),
            "shrunk_hrp": MappingProxyType(dict(hrp)),
        }
    )


def _alpha_max_current_nodes(preflight: AlphaMaxRuntimePreflight) -> tuple[dict[str, object], ...]:
    config_payload = _strict_json_object(preflight.config_bytes)
    registry = config_payload.get("current_trial_registry")
    if type(registry) is not dict or type(registry.get("nodes")) is not list:
        raise AlphaMaxRuntimeContractError("alpha_max_current_trial_registry_invalid")
    nodes = tuple(registry["nodes"])
    if any(type(row) is not dict for row in nodes):
        raise AlphaMaxRuntimeContractError("alpha_max_current_trial_registry_invalid")
    if tuple(sorted(str(row.get("row_id") or "") for row in nodes)) != _ALPHA_MAX_CURRENT_ROW_IDS:
        raise AlphaMaxRuntimeContractError("alpha_max_current_trial_registry_mismatch")
    return nodes


def _alpha_max_row_members(row: Mapping[str, object]) -> tuple[str, ...]:
    row_id = str(row.get("row_id") or "")
    raw = row.get("members")
    members = tuple(raw) if type(raw) is list and raw else (row_id,)
    if not members or members != tuple(sorted(members)):
        raise AlphaMaxRuntimeContractError("alpha_max_row_members_invalid")
    return members


def _alpha_max_fit_weights(
    nodes: Sequence[Mapping[str, object]],
    *,
    phase: str,
    calendar: tuple[str, ...],
    component_returns: Mapping[str, tuple[float, ...]],
) -> AlphaMaxAllocatorFitEvidence:
    component_ids = tuple(sorted(component_returns))
    input_sha256 = _sha256(
        _canonical_bytes(
            {
                "calendar": list(calendar),
                "component_ids": list(component_ids),
                "nominal_cost_bps": 20,
                "returns_by_component": {
                    key: list(component_returns[key]) for key in component_ids
                },
            }
        )
    )
    weights_by_row: dict[str, Mapping[str, float]] = {}
    for row in nodes:
        row_id = str(row.get("row_id") or "")
        if row_id not in _ALPHA_MAX_RESOLVABLE_ROWS:
            continue
        allocation = row.get("allocation")
        if type(allocation) is not dict:
            raise AlphaMaxRuntimeContractError("alpha_max_row_allocation_invalid")
        members = _alpha_max_row_members(row)
        fixed = allocation.get("fixed_weights")
        if type(fixed) is dict:
            weights_by_row[row_id] = MappingProxyType(
                {member: _round(float(fixed[member]), ndigits=10) for member in members}
            )
            continue
        method = str(allocation.get("method") or "")
        matrix = tuple(
            tuple(component_returns[member][index] for member in members)
            for index in range(len(calendar))
        )
        fitted = fit_alpha_max_nominal_20_allocators(
            members,
            matrix,
            nominal_cost_bps=20,
            per_component_cap=float(allocation["per_component_cap"]),
        )
        if method not in fitted:
            raise AlphaMaxRuntimeContractError("alpha_max_row_allocator_method_invalid")
        weights_by_row[row_id] = fitted[method]
    return AlphaMaxAllocatorFitEvidence(
        phase=phase,
        component_ids=component_ids,
        calendar=calendar,
        returns_by_component=MappingProxyType(dict(component_returns)),
        weights_by_row=MappingProxyType(weights_by_row),
        input_sha256=input_sha256,
    )


def _alpha_max_ordered_lookup(
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    root_ids: tuple[str, ...],
) -> AlphaMaxOrderedFundingLookup:
    try:
        feature_seals = tuple(root_seals[(root_id, "feature")] for root_id in root_ids)
    except KeyError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_feature_root_sequence_incomplete") from exc
    specs = tuple(_alpha_max_feature_spec(seal) for seal in feature_seals)
    return AlphaMaxOrderedFundingLookup(specs, root_seals=feature_seals)


def _alpha_max_phase_lookup(
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    phase_id: str,
) -> AlphaMaxOrderedFundingLookup:
    return _alpha_max_ordered_lookup(
        root_seals,
        _alpha_max_expected_root_sequence(phase_id),
    )


def _alpha_max_materialize_manifest_receipt(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: Path,
    phase: str,
    row: Mapping[str, object],
    weights: Mapping[str, float],
    gross: float,
    admitted_symbols: tuple[str, ...],
    admission_sha256: str,
) -> AlphaMaxManifestReceipt:
    from lumina_quant.research.alpha_max_evidence import materialize_alpha_max_manifest

    def materialize(root: Path):
        return materialize_alpha_max_manifest(
            row,
            weights,
            gross,
            phase,
            preflight.config_receipt.canonical_path,
            str(root),
            preflight.candidate_symbols,
            admitted_symbols,
            admission_sha256,
        )

    row_id = str(row.get("row_id") or "")
    expected_path = output_root / "manifests" / phase / f"{row_id}.json"
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.manifest-check-",
        dir=output_root.parent,
    ) as temporary:
        check_root = _create_alpha_max_run_owned_root(Path(temporary) / "run")
        expected = materialize(check_root)
    if expected_path.exists() or expected_path.is_symlink():
        if expected_path.is_symlink():
            raise AlphaMaxRuntimeContractError("alpha_max_manifest_resume_symlink")
        receipt, observed = read_artifact_bytes(
            expected_path,
            artifact_id="alpha_max_resumed_portfolio_manifest",
        )
        if (
            observed != expected.canonical_bytes
            or receipt.sha256 != expected.sha256
            or receipt.canonical_path != str(expected_path.resolve(strict=True))
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_manifest_resume_mismatch")
        return _alpha_max_manifest_receipt_from_path(
            expected_path,
            root=output_root,
            phase=phase,
        )
    relative = expected_path.relative_to(output_root).as_posix()
    _write_bundle_file_atomic(output_root, relative, expected.canonical_bytes)
    return _alpha_max_manifest_receipt_from_path(
        expected_path,
        root=output_root,
        phase=phase,
    )


def _alpha_max_manifest_receipt_from_path(
    path: Path,
    *,
    root: Path,
    phase: str,
) -> AlphaMaxManifestReceipt:
    receipt, payload = read_artifact_bytes(
        path,
        artifact_id="alpha_max_engine_portfolio_manifest",
    )
    parsed = _strict_json_object(payload)
    if payload != _canonical_bytes(parsed) + b"\n":
        raise AlphaMaxRuntimeContractError("alpha_max_manifest_not_canonical")
    return AlphaMaxManifestReceipt(
        row_id=path.stem,
        phase=phase,
        relative_path=path.relative_to(root).as_posix(),
        sha256=receipt.sha256,
        byte_count=receipt.byte_count,
        activation_receipt=receipt,
    )


def _alpha_max_materialize_capsule_receipt(
    capsule_root: Path,
    *,
    row_id: str,
    phase: str,
    prefix_id: str,
    manifest_sha256: str,
    capsule: AlphaMaxIndicatorCapsule,
) -> AlphaMaxCapsuleReceipt:
    relative = f"capsules/{phase}/{row_id}/{prefix_id}.json"
    envelope = AlphaMaxCapsuleReceipt.canonical_envelope_bytes(
        row_id=row_id,
        phase=phase,
        prefix_id=prefix_id,
        manifest_sha256=manifest_sha256,
        state_payload=_alpha_max_capsule_state_payload(capsule),
    )
    path = capsule_root / relative
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise AlphaMaxRuntimeContractError("alpha_max_capsule_resume_symlink")
        receipt, observed = read_artifact_bytes(
            path,
            artifact_id="alpha_max_resumed_indicator_capsule",
        )
        if (
            observed != envelope
            or receipt.sha256 != _sha256(envelope)
            or receipt.canonical_path != str(path.resolve(strict=True))
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_capsule_resume_mismatch")
    else:
        path = _write_bundle_file_atomic(capsule_root, relative, envelope)
    return AlphaMaxCapsuleReceipt.from_path(
        path,
        row_id=row_id,
        phase=phase,
        prefix_id=prefix_id,
        manifest_sha256=manifest_sha256,
        relative_path=relative,
    )


def _alpha_max_build_indicator_prefix(
    preflight: AlphaMaxRuntimePreflight,
    *,
    manifest_output_root: Path,
    phase: str,
    manifest_receipt: AlphaMaxManifestReceipt,
    admitted_symbols: tuple[str, ...],
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    phase_ids: tuple[str, ...],
    initial_capsule: AlphaMaxIndicatorCapsule | None = None,
) -> AlphaMaxIndicatorCapsule:
    capsule = initial_capsule
    for phase_id in phase_ids:
        root_id = _alpha_max_current_root_id(phase_id)
        raw_seal = root_seals[(root_id, "raw")]
        capsule = build_alpha_max_indicator_capsule(
            preflight,
            output_root=str(manifest_output_root),
            phase=phase,
            manifest_path=manifest_receipt.path,
            admitted_symbols=admitted_symbols,
            phase_id=phase_id,
            raw_root=raw_seal.path,
            ordered_lookup=_alpha_max_phase_lookup(root_seals, phase_id),
            watermark=preflight.phase_windows[phase_id].end_utc,
            prior_indicator_capsule=capsule,
            bounded_raw_loader=_AlphaMaxBoundedRawLoader(raw_seal, admitted_symbols),
        )
    if type(capsule) is not AlphaMaxIndicatorCapsule:
        raise AlphaMaxRuntimeContractError("alpha_max_indicator_prefix_empty")
    return capsule


def _alpha_max_build_fold_inputs(
    preflight: AlphaMaxRuntimePreflight,
    *,
    manifest_output_root: Path,
    capsule_output_root: Path,
    phase: str,
    manifest_receipt: AlphaMaxManifestReceipt,
    admitted_symbols: tuple[str, ...],
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    domain: str,
    initial_capsule: AlphaMaxIndicatorCapsule,
    initial_receipt: AlphaMaxCapsuleReceipt | None = None,
) -> tuple[_AlphaMaxFoldReplayInput, ...]:
    fold_ids = _alpha_max_fold_ids(domain)
    root_id = _alpha_max_current_root_id(fold_ids[0])
    raw_seal = root_seals[(root_id, "raw")]
    lookup = _alpha_max_phase_lookup(root_seals, fold_ids[0])
    loader = _AlphaMaxBoundedRawLoader(raw_seal, admitted_symbols)
    current = initial_capsule
    inputs: list[_AlphaMaxFoldReplayInput] = []
    for index, fold_id in enumerate(fold_ids):
        receipt = (
            initial_receipt
            if index == 0 and initial_receipt is not None
            else _alpha_max_materialize_capsule_receipt(
                capsule_output_root,
                row_id=manifest_receipt.row_id,
                phase=phase,
                prefix_id=fold_id,
                manifest_sha256=manifest_receipt.sha256,
                capsule=current,
            )
        )
        assert receipt is not None
        inputs.append(
            _AlphaMaxFoldReplayInput(
                fold_id=fold_id,
                raw_root=raw_seal.path,
                ordered_lookup=lookup,
                indicator_capsule=current,
                capsule_receipt=receipt,
                raw_root_seals=(raw_seal,),
                feature_root_seals=tuple(
                    root_seals[(feature_id, "feature")]
                    for feature_id in _alpha_max_expected_root_sequence(fold_id)
                ),
                bounded_raw_loader=loader,
            )
        )
        if index + 1 < len(fold_ids):
            current = build_alpha_max_indicator_capsule(
                preflight,
                output_root=str(manifest_output_root),
                phase=phase,
                manifest_path=manifest_receipt.path,
                admitted_symbols=admitted_symbols,
                phase_id=fold_id,
                raw_root=raw_seal.path,
                ordered_lookup=lookup,
                watermark=preflight.phase_windows[fold_id].end_utc,
                prior_indicator_capsule=current,
                bounded_raw_loader=loader,
            )
    return tuple(inputs)


def _alpha_max_daily_returns_from_primary_stream(
    stream: AlphaMaxPrimaryReturnStream,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Collapse the frozen six 4h observations per UTC day into net daily returns."""
    if type(stream) is not AlphaMaxPrimaryReturnStream or len(stream.returns) % 6:
        raise AlphaMaxRuntimeContractError("alpha_max_component_daily_stream_invalid")
    calendar: list[str] = []
    returns: list[float] = []
    for offset in range(0, len(stream.returns), 6):
        endpoints = stream.endpoint_timestamps[offset : offset + 6]
        if (
            len(endpoints) != 6
            or endpoints[-1].hour != 0
            or endpoints[-1].minute != 0
            or any(right - left != timedelta(hours=4) for left, right in pairwise(endpoints))
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_component_daily_calendar_invalid")
        calendar.append(endpoints[-1].date().isoformat())
        returns.append(
            math.prod(1.0 + value for value in stream.returns[offset : offset + 6]) - 1.0
        )
    if len(calendar) != len(set(calendar)) or any(not math.isfinite(value) for value in returns):
        raise AlphaMaxRuntimeContractError("alpha_max_component_daily_stream_invalid")
    return tuple(calendar), tuple(returns)


def _alpha_max_training_day_checkpoint_bytes(
    *,
    component_id: str,
    manifest: AlphaMaxManifestReceipt,
    prefix_sha256: str,
    day_start: datetime,
    carry: _AlphaMaxDailyCarry,
    calendar_day: str,
    endpoint_equity: float,
    daily_return: float,
    ordinal: int,
    previous_data_sha256: str,
) -> bytes:
    """Encode the complete, typed continuation state for one settled UTC day."""
    if (
        type(carry) is not _AlphaMaxDailyCarry
        or not math.isfinite(endpoint_equity)
        or endpoint_equity <= 0.0
        or not math.isfinite(daily_return)
        or ordinal <= 0
        or re.fullmatch(r"[0-9a-f]{64}|", previous_data_sha256) is None
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_day_checkpoint_invalid")
    ledger = [
        {
            "boundary_ms": value.boundary_ms,
            "payment": value.payment,
            "price": value.price,
            "price_close_timestamp_ms": value.price_close_timestamp_ms,
            "price_row_timestamp_ms": value.price_row_timestamp_ms,
            "qty": value.qty,
            "rate": value.rate,
            "rate_source_timestamp_ms": value.rate_source_timestamp_ms,
            "symbol": value.symbol,
        }
        for value in carry.funding_ledger
    ]
    state = {
        "strategy_state": carry.strategy_state,
        "portfolio_state": carry.portfolio_state,
        "execution_state": carry.execution_state,
        "engine_state": carry.engine_state,
        "handler_rows": carry.handler_rows,
        "handler_timestamps_ms": carry.handler_timestamps_ms,
        "funding_ledger": ledger,
    }
    value = {
        "artifact_kind": "alpha_max_training_component_day_checkpoint.v1",
        "calendar_day": calendar_day,
        "carry": _alpha_max_indicator_checkpoint_encode(state),
        "component_id": component_id,
        "day_start_utc": day_start.isoformat().replace("+00:00", "Z"),
        "daily_return_hex": daily_return.hex(),
        "endpoint_equity_hex": endpoint_equity.hex(),
        "manifest": _alpha_max_manifest_checkpoint_identity(manifest),
        "next_day_start_utc": (day_start + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "ordinal": ordinal,
        "prefix_sha256": prefix_sha256,
        "previous_data_sha256": previous_data_sha256,
    }
    return _canonical_bytes(value) + b"\n"


def _alpha_max_training_prefix_from_checkpoint(
    payload: bytes,
    *,
    component_id: str,
    manifest: AlphaMaxManifestReceipt,
) -> AlphaMaxIndicatorCapsule:
    value = _strict_json_object(payload)
    if (
        payload != _canonical_bytes(value) + b"\n"
        or set(value) != {"artifact_kind", "component_id", "manifest", "state"}
        or value["artifact_kind"] != "alpha_max_training_component_prefix_checkpoint.v1"
        or value["component_id"] != component_id
        or value["manifest"] != _alpha_max_manifest_checkpoint_identity(manifest)
        or type(value["state"]) is not dict
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_prefix_checkpoint_invalid")
    state = value["state"]
    required = set(
        _alpha_max_capsule_state_payload(
            AlphaMaxIndicatorCapsule(
                portfolio_mode="x",
                phase_id="x",
                manifest_sha256="x",
                capsule_sha256="x",
                capsule=MappingProxyType({}),
                finalized_children=MappingProxyType({}),
                native_finalization_sha256="x",
                windows_processed=0,
                discarded_signal_count=0,
            )
        )
    )
    if set(state) != required:
        raise AlphaMaxRuntimeContractError("alpha_max_training_prefix_checkpoint_invalid")
    string_fields = (
        "portfolio_mode",
        "phase_id",
        "manifest_sha256",
        "capsule_sha256",
        "native_finalization_sha256",
    )
    count_fields = (
        "windows_processed",
        "discarded_signal_count",
        "market_event_count",
        "funding_event_count",
        "order_event_count",
        "fill_event_count",
        "trade_count",
    )
    if (
        any(type(state[field]) is not str or not state[field] for field in string_fields)
        or any(type(state[field]) is not int or state[field] < 0 for field in count_fields)
        or any(state[field] != 0 for field in count_fields[2:])
        or type(state["capsule"]) is not dict
        or type(state["finalized_children"]) is not dict
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_prefix_checkpoint_invalid")
    capsule = AlphaMaxIndicatorCapsule(
        portfolio_mode=state["portfolio_mode"],
        phase_id=state["phase_id"],
        manifest_sha256=state["manifest_sha256"],
        capsule_sha256=state["capsule_sha256"],
        capsule=_freeze_json(state["capsule"]),
        finalized_children=_freeze_json(state["finalized_children"]),
        native_finalization_sha256=state["native_finalization_sha256"],
        windows_processed=state["windows_processed"],
        discarded_signal_count=state["discarded_signal_count"],
        market_event_count=state["market_event_count"],
        funding_event_count=state["funding_event_count"],
        order_event_count=state["order_event_count"],
        fill_event_count=state["fill_event_count"],
        trade_count=state["trade_count"],
    )
    if (
        capsule.manifest_sha256 != manifest.sha256
        or capsule.phase_id != "warmup"
        or _canonical_bytes(
            {
                "artifact_kind": value["artifact_kind"],
                "component_id": component_id,
                "manifest": value["manifest"],
                "state": _alpha_max_capsule_state_payload(capsule),
            }
        )
        + b"\n"
        != payload
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_prefix_checkpoint_invalid")
    return capsule


def _alpha_max_training_day_from_checkpoint(
    payload: bytes,
    *,
    component_id: str,
    manifest: AlphaMaxManifestReceipt,
    prefix_sha256: str,
    expected_day_start: datetime,
    ordinal: int,
    previous_data_sha256: str,
) -> tuple[_AlphaMaxDailyCarry, float, float]:
    value = _strict_json_object(payload)
    if payload != _canonical_bytes(value) + b"\n" or set(value) != {
        "artifact_kind",
        "calendar_day",
        "carry",
        "component_id",
        "day_start_utc",
        "daily_return_hex",
        "endpoint_equity_hex",
        "manifest",
        "next_day_start_utc",
        "ordinal",
        "prefix_sha256",
        "previous_data_sha256",
    }:
        raise AlphaMaxRuntimeContractError("alpha_max_training_day_checkpoint_invalid")
    if (
        value["artifact_kind"] != "alpha_max_training_component_day_checkpoint.v1"
        or value["component_id"] != component_id
        or value["manifest"] != _alpha_max_manifest_checkpoint_identity(manifest)
        or value["prefix_sha256"] != prefix_sha256
        or value["ordinal"] != ordinal
        or value["previous_data_sha256"] != previous_data_sha256
        or value["day_start_utc"] != expected_day_start.isoformat().replace("+00:00", "Z")
        or value["next_day_start_utc"]
        != (expected_day_start + timedelta(days=1)).isoformat().replace("+00:00", "Z")
        or value["calendar_day"] != (expected_day_start + timedelta(days=1)).date().isoformat()
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_day_checkpoint_identity_invalid")
    try:
        endpoint_equity = float.fromhex(value["endpoint_equity_hex"])
        daily_return = float.fromhex(value["daily_return_hex"])
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_training_day_checkpoint_float_invalid"
        ) from exc
    if (
        not math.isfinite(endpoint_equity)
        or endpoint_equity <= 0.0
        or not math.isfinite(daily_return)
        or endpoint_equity.hex() != value["endpoint_equity_hex"]
        or daily_return.hex() != value["daily_return_hex"]
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_day_checkpoint_float_invalid")
    state = _alpha_max_indicator_checkpoint_decode(value["carry"])
    if type(state) is not dict or set(state) != {
        "strategy_state",
        "portfolio_state",
        "execution_state",
        "engine_state",
        "handler_rows",
        "handler_timestamps_ms",
        "funding_ledger",
    }:
        raise AlphaMaxRuntimeContractError("alpha_max_training_day_checkpoint_carry_invalid")
    try:
        ledger = tuple(AlphaMaxFundingBoundaryLedgerRow(**row) for row in state["funding_ledger"])
        carry = _AlphaMaxDailyCarry(
            strategy_state=state["strategy_state"],
            portfolio_state=state["portfolio_state"],
            execution_state=state["execution_state"],
            engine_state=state["engine_state"],
            handler_rows=state["handler_rows"],
            handler_timestamps_ms=state["handler_timestamps_ms"],
            funding_ledger=ledger,
        )
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_training_day_checkpoint_carry_invalid"
        ) from exc
    if (
        _alpha_max_training_day_checkpoint_bytes(
            component_id=component_id,
            manifest=manifest,
            prefix_sha256=prefix_sha256,
            day_start=expected_day_start,
            carry=carry,
            calendar_day=value["calendar_day"],
            endpoint_equity=endpoint_equity,
            daily_return=daily_return,
            ordinal=ordinal,
            previous_data_sha256=previous_data_sha256,
        )
        != payload
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_day_checkpoint_roundtrip_invalid")
    return carry, endpoint_equity, daily_return


def _alpha_max_replay_training_component_returns(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: Path,
    manifest_receipt: AlphaMaxManifestReceipt,
    admitted_symbols: tuple[str, ...],
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    checkpoint_store: _AlphaMaxPrecomputeCheckpointStore,
) -> tuple[
    tuple[str, ...],
    tuple[float, ...],
    AlphaMaxNativeFinalizationReceipt,
]:
    """Replay one component at nominal 20 bps over train with fresh daily engines."""
    train_window = preflight.phase_windows["train"]
    train_start = datetime.fromisoformat(train_window.start_utc.replace("Z", "+00:00")).astimezone(
        UTC
    )
    train_end = datetime.fromisoformat(train_window.end_utc.replace("Z", "+00:00")).astimezone(UTC)
    raw_seal = root_seals[("train", "raw")]
    feature_seals = tuple(
        root_seals[(root_id, "feature")] for root_id in _alpha_max_expected_root_sequence("train")
    )
    lookup = _alpha_max_phase_lookup(root_seals, "train")
    loader = _AlphaMaxBoundedRawLoader(raw_seal, admitted_symbols)
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted_symbols)
    component_id = manifest_receipt.row_id
    sealed_prefix = checkpoint_store.load(unit_kind="training_prefix", unit_id=component_id)
    if sealed_prefix is None:
        warmup_capsule = _alpha_max_build_indicator_prefix(
            preflight,
            manifest_output_root=output_root,
            phase="validation_train_fit",
            manifest_receipt=manifest_receipt,
            admitted_symbols=admitted_symbols,
            root_seals=root_seals,
            phase_ids=("warmup",),
        )
        prefix_bytes = (
            _canonical_bytes(
                {
                    "artifact_kind": "alpha_max_training_component_prefix_checkpoint.v1",
                    "component_id": component_id,
                    "manifest": _alpha_max_manifest_checkpoint_identity(manifest_receipt),
                    "state": _alpha_max_capsule_state_payload(warmup_capsule),
                }
            )
            + b"\n"
        )
        sealed_prefix = checkpoint_store.seal(
            unit_kind="training_prefix", unit_id=component_id, data_bytes=prefix_bytes
        )
    warmup_capsule = _alpha_max_training_prefix_from_checkpoint(
        sealed_prefix, component_id=component_id, manifest=manifest_receipt
    )
    prefix_sha256 = _sha256(sealed_prefix)
    carry: _AlphaMaxDailyCarry | None = None
    native_finalization: AlphaMaxNativeFinalizationReceipt | None = None
    day_start = train_start
    calendar: list[str] = []
    returns: list[float] = []
    prior_equity = 10_000.0
    previous_data_sha256 = ""
    ordinal = 1
    while day_start + timedelta(days=1) < train_end:
        unit_id = f"{component_id}--{day_start:%Y%m%d}"
        payload = checkpoint_store.load(unit_kind="training_day", unit_id=unit_id)
        if payload is None:
            break
        carry, endpoint_equity, daily_return = _alpha_max_training_day_from_checkpoint(
            payload,
            component_id=component_id,
            manifest=manifest_receipt,
            prefix_sha256=prefix_sha256,
            expected_day_start=day_start,
            ordinal=ordinal,
            previous_data_sha256=previous_data_sha256,
        )
        if daily_return != endpoint_equity / prior_equity - 1.0:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_training_day_checkpoint_economics_invalid"
            )
        calendar.append((day_start + timedelta(days=1)).date().isoformat())
        returns.append(daily_return)
        prior_equity = endpoint_equity
        previous_data_sha256 = _sha256(payload)
        day_start += timedelta(days=1)
        ordinal += 1
    probe_day = day_start + timedelta(days=1)
    while probe_day + timedelta(days=1) < train_end:
        if (
            checkpoint_store.load(
                unit_kind="training_day",
                unit_id=f"{component_id}--{probe_day:%Y%m%d}",
            )
            is not None
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_training_day_checkpoint_gap")
        probe_day += timedelta(days=1)
    while day_start < train_end:
        day_end = day_start + timedelta(days=1)
        if carry is not None:
            resolver = AlphaMaxFundingBoundaryResolver.from_checkpoint(
                lookup, admitted_symbols, ledger=carry.funding_ledger
            )
        collector = AlphaMaxAttributionCollector()
        aggregate = AlphaMaxStreamingEquityTracker()
        tracker = _AlphaMaxFoldEquityFanout(
            aggregate,
            aggregate_scale=1.0,
            reporting_start=day_start,
            reporting_end=day_end,
        )
        data_dict = loader.load_day(day_start, day_end)
        activation = construct_alpha_max_engine(
            preflight,
            output_root=str(output_root),
            phase="validation_train_fit",
            manifest_path=manifest_receipt.path,
            admitted_symbols=admitted_symbols,
            phase_id="train",
            nominal_cost_bps=20,
            raw_root=raw_seal.path,
            ordered_lookup=lookup,
            funding_resolver=resolver,
            data_dict=data_dict,
            attribution_collector=collector,
            full_event_equity_tracker=tracker,
            indicator_capsule=warmup_capsule,
            raw_root_seals=(raw_seal,),
            feature_root_seals=feature_seals,
            _repeat_root_hash_on_activation=False,
            _chunk_start_utc=day_start,
            _chunk_end_utc=day_end,
        )
        if carry is not None:
            _restore_alpha_max_daily_carry(activation, carry)
        tracker.bind_backtest(activation.backtest)
        validate_alpha_max_engine_activation(activation, _expected_daily_carry=carry)
        _run_alpha_max_exact_tick_reducer(activation)
        traces = activation.backtest.execution_handler.pricing_trace_evidence
        day_finalization = _settle_alpha_max_day_boundary(
            activation,
            tracker,
            day_end,
            scoring_boundary=day_end == train_end,
        )
        if day_finalization is not None:
            if native_finalization is not None:
                raise AlphaMaxRuntimeContractError("alpha_max_train_native_finalization_duplicate")
            native_finalization = day_finalization
        carry = _capture_alpha_max_daily_carry(activation)
        applications = collector.applications
        if len(traces) != len(applications) or any(
            application.pricing_trace_hash != execution_pricing_trace_sha256(trace)
            for trace, application in zip(traces, applications, strict=True)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_pricing_application_bijection_failed")
        aggregate.finalize()
        endpoints = tracker.reporting_endpoints
        if len(endpoints) != 6 or endpoints[-1].timestamp != day_end:
            raise AlphaMaxRuntimeContractError("alpha_max_train_component_calendar_invalid")
        endpoint_equity = endpoints[-1].equity
        daily_return = endpoint_equity / prior_equity - 1.0
        if not math.isfinite(daily_return):
            raise AlphaMaxRuntimeContractError("alpha_max_train_component_return_nonfinite")
        calendar.append(day_end.date().isoformat())
        returns.append(daily_return)
        if day_end < train_end:
            day_payload = _alpha_max_training_day_checkpoint_bytes(
                component_id=component_id,
                manifest=manifest_receipt,
                prefix_sha256=prefix_sha256,
                day_start=day_start,
                carry=carry,
                calendar_day=day_end.date().isoformat(),
                endpoint_equity=endpoint_equity,
                daily_return=daily_return,
                ordinal=ordinal,
                previous_data_sha256=previous_data_sha256,
            )
            restored_carry, restored_equity, restored_return = (
                _alpha_max_training_day_from_checkpoint(
                    day_payload,
                    component_id=component_id,
                    manifest=manifest_receipt,
                    prefix_sha256=prefix_sha256,
                    expected_day_start=day_start,
                    ordinal=ordinal,
                    previous_data_sha256=previous_data_sha256,
                )
            )
            if (
                not _exact_state_equal(restored_carry.strategy_state, carry.strategy_state)
                or not _exact_state_equal(restored_carry.portfolio_state, carry.portfolio_state)
                or not _exact_state_equal(restored_carry.execution_state, carry.execution_state)
                or not _exact_state_equal(restored_carry.engine_state, carry.engine_state)
                or restored_carry.handler_rows != carry.handler_rows
                or restored_carry.handler_timestamps_ms != carry.handler_timestamps_ms
                or restored_carry.funding_ledger != carry.funding_ledger
                or restored_equity != endpoint_equity
                or restored_return != daily_return
            ):
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_training_day_checkpoint_roundtrip_invalid"
                )
            checkpoint_store.seal(
                unit_kind="training_day",
                unit_id=f"{component_id}--{day_start:%Y%m%d}",
                data_bytes=day_payload,
            )
            previous_data_sha256 = _sha256(day_payload)
        prior_equity = endpoint_equity
        ordinal += 1
        day_start = day_end
    if carry is None or native_finalization is None:
        raise AlphaMaxRuntimeContractError("alpha_max_train_component_replay_empty")
    _validate_alpha_max_root_seals(
        raw_root=raw_seal.path,
        phase_id="train",
        ordered_lookup=lookup,
        raw_root_seals=(raw_seal,),
        feature_root_seals=feature_seals,
        required=True,
        repeat_hash=False,
    )
    if len(calendar) < 252 or len(calendar) != len(set(calendar)):
        raise AlphaMaxRuntimeContractError("alpha_max_train_component_calendar_invalid")
    return tuple(calendar), tuple(returns), native_finalization


def _alpha_max_scaled_gross(
    sibling: AlphaMaxCostCellPreGateEvidence,
) -> float:
    """Resolve one scaled gross from the actual validation nominal-30 sibling."""
    if sibling.domain != "validation" or sibling.nominal_cost_bps != 30:
        raise AlphaMaxRuntimeContractError("alpha_max_scaled_sibling_invalid")
    if sibling.metric_statistics is not None:
        mdd = sibling.metric_statistics.gate_mdd
    else:
        mdd = max(
            fold.actual_engine_run.full_event_equity.full_event_mdd for fold in sibling.fold_runs
        )
    gross = min(2.25, max(0.25, 0.27 / max(float(mdd), 1e-12)))
    if not math.isfinite(gross):
        raise AlphaMaxRuntimeContractError("alpha_max_scaled_gross_invalid")
    return gross


def _alpha_max_prepare_validation_row(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: Path,
    row: Mapping[str, object],
    weights: Mapping[str, float],
    gross: float,
    admitted_symbols: tuple[str, ...],
    admission_sha256: str,
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    retained_manifest: AlphaMaxManifestReceipt | None = None,
) -> _AlphaMaxPreparedReplayRow:
    manifest = retained_manifest or _alpha_max_materialize_manifest_receipt(
        preflight,
        output_root=output_root,
        phase="validation_train_fit",
        row=row,
        weights=weights,
        gross=gross,
        admitted_symbols=admitted_symbols,
        admission_sha256=admission_sha256,
    )
    prefix = _alpha_max_build_indicator_prefix(
        preflight,
        manifest_output_root=output_root,
        phase="validation_train_fit",
        manifest_receipt=manifest,
        admitted_symbols=admitted_symbols,
        root_seals=root_seals,
        phase_ids=("warmup", "train", "purge"),
    )
    return _AlphaMaxPreparedReplayRow(
        manifest_receipt=manifest,
        fold_inputs=_alpha_max_build_fold_inputs(
            preflight,
            manifest_output_root=output_root,
            capsule_output_root=output_root,
            phase="validation_train_fit",
            manifest_receipt=manifest,
            admitted_symbols=admitted_symbols,
            root_seals=root_seals,
            domain="validation",
            initial_capsule=prefix,
        ),
        gross=gross,
    )


def _alpha_max_checkpoint_implementation_inventory() -> list[dict[str, object]]:
    repository = Path(__file__).resolve().parents[3]
    paths = sorted((repository / "src" / "lumina_quant").rglob("*.py"))
    paths.extend(
        sorted(
            path
            for path in (repository / "scripts" / "research").glob("run_alpha_max_*.py")
            if path.is_file()
        )
    )
    native_compute = repository / "native" / "lumina_compute"
    paths.extend(sorted((native_compute / "src").rglob("*.rs")))
    for name in ("Cargo.toml", "Cargo.lock", "build.rs"):
        path = native_compute / name
        if path.is_file():
            paths.append(path)
    for relative in ("pyproject.toml", "scripts/build_native_backends.py"):
        path = repository / relative
        if path.is_file():
            paths.append(path)
    lock = repository / "uv.lock"
    if lock.is_file():
        paths.append(lock)
    inventory: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in paths:
        relative = path.relative_to(repository).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        receipt, _payload = read_artifact_bytes(
            path,
            artifact_id=f"checkpoint-implementation:{relative}",
        )
        inventory.append(
            {
                "byte_count": receipt.byte_count,
                "relative_path": relative,
                "sha256": receipt.sha256,
            }
        )
    return inventory


def _verify_alpha_max_checkpoint_implementation_inventory(
    expected: object,
) -> list[dict[str, object]]:
    if type(expected) is not list or not all(type(row) is dict for row in expected):
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_implementation_inventory_invalid")
    current = _alpha_max_checkpoint_implementation_inventory()
    if current != expected:
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_implementation_inventory_mismatch")
    return current


def _alpha_max_checkpoint_runtime_identity_sha256(value: object) -> str:
    if (
        type(value) is not dict
        or set(value)
        != {
            "extension_byte_count",
            "extension_module",
            "extension_path",
            "extension_sha256",
            "extension_source_hash",
            "extension_version",
        }
        or type(value["extension_byte_count"]) is not int
        or value["extension_byte_count"] <= 0
        or value["extension_module"] != "lumina_quant._compute"
        or not _alpha_max_indicator_absolute_path(value["extension_path"])
        or not _alpha_max_indicator_sha256(value["extension_sha256"])
        or type(value["extension_source_hash"]) is not str
        or re.fullmatch(r"[0-9a-f]{16}", value["extension_source_hash"]) is None
        or type(value["extension_version"]) is not str
        or not value["extension_version"]
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_runtime_identity_invalid")
    return _sha256(_canonical_bytes(value))


def _alpha_max_prelock_checkpoint_descriptor(
    *,
    preflight: AlphaMaxRuntimePreflight,
    contract_seal: AlphaMaxContractManifestSeal,
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    admitted_symbols: tuple[str, ...],
    output_root: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    implementation_inventory: list[dict[str, object]],
    prior_trial_binding: Mapping[str, object] | None = None,
    _include_v2_bindings: bool = False,
) -> dict[str, object]:
    target, parent = _validated_output_target(output_root)
    checkpoint = Path(_require_exact_explicit_path(checkpoint_root))
    checkpoint_parent = checkpoint.parent
    try:
        checkpoint_parent_resolved = checkpoint_parent.resolve(strict=True)
        checkpoint_parent_status = checkpoint_parent.lstat()
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_parent_invalid") from exc
    if (
        str(checkpoint_parent_resolved) != str(checkpoint_parent)
        or not stat.S_ISDIR(checkpoint_parent_status.st_mode)
        or stat.S_ISLNK(checkpoint_parent_status.st_mode)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_parent_invalid")
    if prior_trial_binding is not None and (
        type(prior_trial_binding) is not dict
        or set(prior_trial_binding) != {"byte_count", "path", "sha256"}
        or type(prior_trial_binding.get("path")) is not str
        or type(prior_trial_binding.get("byte_count")) is not int
        or type(prior_trial_binding.get("sha256")) is not str
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_prior_trial_binding_invalid")
    runtime_identity = _alpha_max_indicator_runtime_binding()
    protected_paths = (
        Path(preflight.config_receipt.canonical_path),
        Path(contract_seal.path),
        Path(runtime_identity["extension_path"]),
        Path(sys.executable).resolve(strict=True),
        *(Path(seal.path) for seal in root_seals.values()),
        *((Path(str(prior_trial_binding["path"])),) if prior_trial_binding is not None else ()),
    )
    for left, right in (
        (checkpoint, target),
        *((checkpoint, value) for value in protected_paths),
        *((target, value) for value in protected_paths),
    ):
        if left == right or left in right.parents or right in left.parents:
            raise AlphaMaxRuntimeContractError("alpha_max_restart_path_overlap")
    schedule = [
        {
            "fold_id": fold_id,
            "nominal_cost_bps": nominal,
            "row_id": row_id,
            "seed": alpha_max_common_rng_seed(fold_id, nominal),
        }
        for row_id, nominal, fold_id in _alpha_max_physical_fold_schedule("validation")
    ]
    roots = [
        {
            "availability_sha256": seal.availability_sha256,
            "content_sha256": seal.content_sha256,
            "inventory_sha256": seal.inventory_sha256,
            "path": seal.path,
            "root_id": root_id,
            "root_kind": root_kind,
            "seal_sha256": seal.sha256,
        }
        for (root_id, root_kind), seal in sorted(root_seals.items())
    ]
    executable_receipt, _executable = read_artifact_bytes(
        Path(sys.executable).resolve(strict=True),
        artifact_id="checkpoint-python-executable",
    )
    descriptor = {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v2",
        "attempt_role": "prelock",
        "domain": "validation",
        "checkpoint_unit": "whole_row_cost_cell",
        "checkpoint": {
            "parent": str(checkpoint_parent),
            "parent_identity": [
                int(checkpoint_parent_status.st_dev),
                int(checkpoint_parent_status.st_ino),
            ],
            "root": str(checkpoint),
        },
        "config": {
            "byte_count": preflight.config_receipt.byte_count,
            "sha256": preflight.config_receipt.sha256,
        },
        "contract_manifest": {
            "byte_count": contract_seal.byte_count,
            "path": contract_seal.path,
            "sha256": contract_seal.sha256,
        },
        "cost_cells_bps": list(ALPHA_MAX_COST_CELL_BPS),
        "logical_cell_count": 68,
        "implementation_inventory": implementation_inventory,
        "immutable": True,
        "order_routing_enabled": False,
        "output": {
            "parent": str(parent),
            "parent_identity": [
                int(parent.stat().st_dev),
                int(parent.stat().st_ino),
            ],
            "target": str(target),
        },
        "phase_windows": {
            phase_id: {
                "end_utc": window.end_utc,
                "start_utc": window.start_utc,
            }
            for phase_id, window in sorted(preflight.phase_windows.items())
        },
        "physical_fold_run_count": len(schedule),
        "physical_schedule": schedule,
        "physical_schedule_sha256": _sha256(_canonical_bytes(schedule)),
        "python": {
            "byte_count": executable_receipt.byte_count,
            "cache_tag": sys.implementation.cache_tag,
            "executable": executable_receipt.canonical_path,
            "sha256": executable_receipt.sha256,
            "version": list(sys.version_info[:3]),
        },
        "root_seals": roots,
        "runtime_identity": runtime_identity,
        "runtime_contract_sha256": preflight.runtime_contract_sha256,
        "thread_contract": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "POLARS_MAX_THREADS",
                "RAYON_NUM_THREADS",
            )
        },
        "universe": {
            "admitted_symbols": list(admitted_symbols),
            "candidate_symbols": list(preflight.candidate_symbols),
            "sha256": _sha256(_canonical_bytes(list(admitted_symbols))),
        },
    }
    if prior_trial_binding is not None:
        descriptor["prior_trial_blob"] = dict(prior_trial_binding)
    if _include_v2_bindings:
        return descriptor
    return {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v1",
        "attempt_role": "prelock",
        "implementation_inventory": implementation_inventory,
        "checkpoint": descriptor["checkpoint"],
        "output": descriptor["output"],
    }


def _alpha_max_historical_checkpoint_descriptor(
    *,
    preflight: AlphaMaxRuntimePreflight,
    contract_seal: AlphaMaxContractManifestSeal,
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    admitted_symbols: tuple[str, ...],
    output_root: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    implementation_inventory: list[dict[str, object]],
    prelock_seal_bytes: bytes,
    prelock_snapshot_sha256: str,
) -> dict[str, object]:
    """Build the distinct, immutable historical checkpoint attempt identity."""
    descriptor = _alpha_max_prelock_checkpoint_descriptor(
        preflight=preflight,
        contract_seal=contract_seal,
        root_seals=root_seals,
        admitted_symbols=admitted_symbols,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        implementation_inventory=implementation_inventory,
        _include_v2_bindings=True,
    )
    domain = "historical_exposed_evaluation"
    schedule = [
        {
            "fold_id": fold_id,
            "nominal_cost_bps": nominal,
            "row_id": row_id,
            "seed": alpha_max_common_rng_seed(fold_id, nominal),
        }
        for row_id, nominal, fold_id in _alpha_max_physical_fold_schedule(domain)
    ]
    descriptor.update(
        {
            "artifact_kind": "alpha_max_restartable_attempt_descriptor.v2",
            "attempt_role": "historical",
            "domain": domain,
            "physical_fold_run_count": len(schedule),
            "physical_schedule": schedule,
            "physical_schedule_sha256": _sha256(_canonical_bytes(schedule)),
            "prelock_binding": {
                "immutable_prelock_seal_sha256": _sha256(prelock_seal_bytes),
                "validated_snapshot_sha256": prelock_snapshot_sha256,
            },
        }
    )
    return descriptor


def _alpha_max_validate_checkpoint_descriptor(descriptor: Mapping[str, object]) -> tuple[str, str]:
    """Versioned descriptor parser; v1 remains validation/prelock-only."""
    kind = descriptor.get("artifact_kind")
    if kind == "alpha_max_restartable_attempt_descriptor.v1":
        if (
            set(descriptor)
            != {
                "artifact_kind",
                "attempt_role",
                "implementation_inventory",
                "checkpoint",
                "output",
            }
            or descriptor.get("attempt_role") != "prelock"
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_descriptor_role_invalid")
        for field in ("implementation_inventory", "checkpoint", "output"):
            if type(descriptor.get(field)) is not (
                list if field == "implementation_inventory" else dict
            ):
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_checkpoint_descriptor_binding_invalid"
                )
        return "prelock", "validation"
    if kind != "alpha_max_restartable_attempt_descriptor.v2":
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_descriptor_version_invalid")
    required = {
        "artifact_kind",
        "attempt_role",
        "domain",
        "checkpoint_unit",
        "checkpoint",
        "config",
        "contract_manifest",
        "cost_cells_bps",
        "logical_cell_count",
        "implementation_inventory",
        "immutable",
        "order_routing_enabled",
        "output",
        "phase_windows",
        "physical_fold_run_count",
        "physical_schedule",
        "physical_schedule_sha256",
        "python",
        "root_seals",
        "runtime_identity",
        "runtime_contract_sha256",
        "thread_contract",
        "universe",
    }
    role = descriptor.get("attempt_role")
    domain = descriptor.get("domain")
    if (role, domain) not in {
        ("prelock", "validation"),
        ("historical", "historical_exposed_evaluation"),
    }:
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_descriptor_role_invalid")
    schedule = descriptor.get("physical_schedule")
    expected_schedule = [
        {
            "fold_id": fold_id,
            "nominal_cost_bps": nominal,
            "row_id": row_id,
            "seed": alpha_max_common_rng_seed(fold_id, nominal),
        }
        for row_id, nominal, fold_id in _alpha_max_physical_fold_schedule(domain)
    ]
    if (
        set(descriptor)
        != required | ({"prelock_binding"} if role == "historical" else {"prior_trial_blob"})
        or descriptor.get("immutable") is not True
        or descriptor.get("checkpoint_unit") != "whole_row_cost_cell"
        or descriptor.get("cost_cells_bps") != list(ALPHA_MAX_COST_CELL_BPS)
        or descriptor.get("logical_cell_count") != 68
        or descriptor.get("physical_fold_run_count") != len(expected_schedule)
        or schedule != expected_schedule
        or descriptor.get("physical_schedule_sha256")
        != _sha256(_canonical_bytes(expected_schedule))
        or descriptor.get("order_routing_enabled") is not False
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_descriptor_schedule_invalid")
    for field in ("config", "contract_manifest", "checkpoint", "output", "python", "universe"):
        if type(descriptor.get(field)) is not dict:
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_descriptor_binding_invalid")
    if (
        type(descriptor.get("root_seals")) is not list
        or type(descriptor.get("phase_windows")) is not dict
        or type(descriptor.get("thread_contract")) is not dict
        or type(descriptor.get("implementation_inventory")) is not list
        or type(descriptor.get("runtime_contract_sha256")) is not str
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_descriptor_binding_invalid")
    _alpha_max_checkpoint_runtime_identity_sha256(descriptor.get("runtime_identity"))
    prelock_binding = descriptor.get("prelock_binding")
    if role == "historical":
        if (
            type(prelock_binding) is not dict
            or set(prelock_binding)
            != {
                "immutable_prelock_seal_sha256",
                "validated_snapshot_sha256",
            }
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in prelock_binding.values()
            )
        ):
            raise AlphaMaxRuntimeContractError(
                "alpha_max_checkpoint_descriptor_prelock_binding_invalid"
            )
    else:
        if prelock_binding is not None:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_checkpoint_descriptor_prelock_binding_invalid"
            )
        prior_trial_blob = descriptor.get("prior_trial_blob")
        if (
            type(prior_trial_blob) is not dict
            or set(prior_trial_blob) != {"byte_count", "path", "sha256"}
            or type(prior_trial_blob.get("byte_count")) is not int
            or prior_trial_blob["byte_count"] <= 0
            or type(prior_trial_blob.get("path")) is not str
            or type(prior_trial_blob.get("sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", prior_trial_blob["sha256"]) is None
        ):
            raise AlphaMaxRuntimeContractError(
                "alpha_max_checkpoint_descriptor_prior_trial_binding_invalid"
            )
    return role, domain


def _alpha_max_stat_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_mode),
        int(status.st_nlink),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _alpha_max_cleanup_recognized_staging_bundle(
    root: Path,
    *,
    allowed_files: frozenset[str],
    allowed_directories: frozenset[str],
    error_token: str,
) -> None:
    """Delete only an exact writer staging tree with no links or unknown entries."""
    parent_fd = -1
    opened_directories: list[int] = []
    try:
        parent_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        if not _is_proc_fd_parent(root.parent):
            parent_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(root.parent, parent_flags)
        root_status = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
            raise AlphaMaxRuntimeContractError(error_token)
        root_fd = _alpha_max_open_directory_at(root.name, dir_fd=parent_fd)
        opened_directories.append(root_fd)
        _alpha_max_require_open_identity(root_fd, root_status, directory=True)
        directory_nodes: list[tuple[int, str, int, os.stat_result, str]] = []
        file_nodes: list[tuple[int, str, os.stat_result, str]] = []
        inventories: dict[int, frozenset[str]] = {}

        def inspect(directory_fd: int, prefix: str) -> None:
            names = frozenset(os.listdir(directory_fd))
            inventories[directory_fd] = names
            for name in sorted(names):
                relative = f"{prefix}/{name}" if prefix else name
                observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(observed.st_mode):
                    if relative not in allowed_directories:
                        raise AlphaMaxRuntimeContractError(error_token)
                    child_fd = _alpha_max_open_directory_at(name, dir_fd=directory_fd)
                    opened_directories.append(child_fd)
                    _alpha_max_require_open_identity(child_fd, observed, directory=True)
                    directory_nodes.append((directory_fd, name, child_fd, observed, relative))
                    inspect(child_fd, relative)
                elif (
                    not stat.S_ISREG(observed.st_mode)
                    or int(observed.st_nlink) != 1
                    or relative not in allowed_files
                ):
                    raise AlphaMaxRuntimeContractError(error_token)
                else:
                    file_nodes.append((directory_fd, name, observed, relative))

        inspect(root_fd, "")
        if any(frozenset(os.listdir(fd)) != names for fd, names in inventories.items()):
            raise AlphaMaxRuntimeContractError(error_token)
        for directory_fd, name, observed, _relative in file_nodes:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _alpha_max_stat_identity(current) != _alpha_max_stat_identity(observed):
                raise AlphaMaxRuntimeContractError(error_token)
            os.unlink(name, dir_fd=directory_fd)
        for directory_fd, name, child_fd, observed, _relative in reversed(directory_nodes):
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (int(current.st_dev), int(current.st_ino)) != (
                int(observed.st_dev),
                int(observed.st_ino),
            ) or os.listdir(child_fd):
                raise AlphaMaxRuntimeContractError(error_token)
            os.fchmod(child_fd, 0o700)
            os.rmdir(name, dir_fd=directory_fd)
        current_root = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if (int(current_root.st_dev), int(current_root.st_ino)) != (
            int(root_status.st_dev),
            int(root_status.st_ino),
        ) or os.listdir(root_fd):
            raise AlphaMaxRuntimeContractError(error_token)
        os.fchmod(root_fd, 0o700)
        os.rmdir(root.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise AlphaMaxRuntimeContractError(error_token) from exc
    finally:
        for fd in reversed(opened_directories):
            os.close(fd)
        if parent_fd >= 0:
            os.close(parent_fd)


class _AlphaMaxPrecomputeCheckpointStore:
    """Atomic immutable whole-precompute-unit journal for one cell attempt."""

    __slots__ = (
        "_attempt_descriptor_sha256",
        "_attempt_role",
        "_descriptor_bytes",
        "_descriptor_sha256",
        "_display_root",
        "_domain",
        "_root",
        "_root_fd",
        "_root_identity",
        "_runtime_identity_sha256",
        "_training_day_ids",
        "_transaction_lock_identity",
        "_units",
        "_units_fd",
        "_units_identity",
    )

    def __init__(
        self,
        root: Path,
        *,
        attempt_descriptor_sha256: str,
        attempt_role: str,
        domain: str,
        runtime_identity_sha256: str,
        training_day_ids: tuple[str, ...] = (),
        transaction_lock_identity: tuple[int, int] | None = None,
    ) -> None:
        if (
            not isinstance(root, Path)
            or not root.is_absolute()
            or type(attempt_descriptor_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", attempt_descriptor_sha256) is None
            or type(runtime_identity_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", runtime_identity_sha256) is None
            or (attempt_role, domain)
            not in {
                ("prelock", "validation"),
                ("historical", "historical_exposed_evaluation"),
            }
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_descriptor_invalid")
        if (
            type(training_day_ids) is not tuple
            or len(training_day_ids) != len(set(training_day_ids))
            or any(
                re.fullmatch(r"component_(?:carry|near_high|trend)_1x--\d{8}", value) is None
                for value in training_day_ids
            )
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_training_schedule_invalid")
        self._display_root = root
        self._root = root
        self._units = root / "units"
        self._root_fd = -1
        self._units_fd = -1
        self._attempt_descriptor_sha256 = attempt_descriptor_sha256
        self._attempt_role = attempt_role
        self._domain = domain
        self._runtime_identity_sha256 = runtime_identity_sha256
        self._training_day_ids = training_day_ids
        descriptor = {
            "artifact_kind": "alpha_max_precompute_checkpoint_attempt.v1",
            "attempt_descriptor_sha256": attempt_descriptor_sha256,
            "attempt_role": attempt_role,
            "checkpoint_unit": "whole_authenticated_precompute_unit",
            "domain": domain,
            "order_routing_enabled": False,
            "runtime_identity_sha256": runtime_identity_sha256,
            "unit_schedule": [
                {"unit_id": unit_id, "unit_kind": unit_kind}
                for unit_kind, unit_id in self._allowed_units()
            ],
        }
        self._descriptor_bytes = _canonical_bytes(descriptor) + b"\n"
        self._descriptor_sha256 = _sha256(self._descriptor_bytes)
        if not root.exists() and not root.is_symlink():
            self._initialize()
        try:
            root_status = self._display_root.lstat()
            self._root_fd = _alpha_max_open_directory_at(self._display_root)
            _alpha_max_require_open_identity(self._root_fd, root_status, directory=True)
            units_status = os.stat("units", dir_fd=self._root_fd, follow_symlinks=False)
            self._units_fd = _alpha_max_open_directory_at("units", dir_fd=self._root_fd)
            _alpha_max_require_open_identity(self._units_fd, units_status, directory=True)
            self._root_identity = (int(root_status.st_dev), int(root_status.st_ino))
            self._units_identity = (int(units_status.st_dev), int(units_status.st_ino))
            lock_status = os.stat(".transaction.lock", dir_fd=self._units_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(lock_status.st_mode)
                or stat.S_ISLNK(lock_status.st_mode)
                or stat.S_IMODE(lock_status.st_mode) != 0o600
                or lock_status.st_nlink != 1
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_precompute_transaction_lock_invalid")
            self._transaction_lock_identity = (
                int(lock_status.st_dev),
                int(lock_status.st_ino),
            )
            if (
                transaction_lock_identity is not None
                and self._transaction_lock_identity != transaction_lock_identity
            ):
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_precompute_transaction_lock_identity_mismatch"
                )
            self._root = Path(f"/proc/self/fd/{self._root_fd}")
            self._units = Path(f"/proc/self/fd/{self._units_fd}")
            self._validate_root()
        except Exception:
            self.__del__()
            raise

    def __del__(self) -> None:
        for field in ("_units_fd", "_root_fd"):
            fd = getattr(self, field, -1)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, field, -1)

    def _allowed_units(self) -> tuple[tuple[str, str], ...]:
        component_ids = (
            "component_carry_1x",
            "component_near_high_1x",
            "component_trend_1x",
        )
        if self._attempt_role == "historical":
            return tuple(("historical_row", row_id) for row_id in _ALPHA_MAX_RESOLVABLE_ROWS)
        return (
            *(("training_prefix", component_id) for component_id in component_ids),
            *(("training_day", day_id) for day_id in self._training_day_ids),
            *(("training_component", component_id) for component_id in component_ids),
            *(("validation_row", row_id) for row_id in _ALPHA_MAX_RESOLVABLE_ROWS),
            *(("final_refit_row", row_id) for row_id in _ALPHA_MAX_RESOLVABLE_ROWS),
        )

    @staticmethod
    def _is_training_day_unit(unit_kind: str, unit_id: str) -> bool:
        return unit_kind == "training_day" and type(unit_id) is str

    def _unit_name(self, unit_kind: str, unit_id: str) -> str:
        if type(unit_kind) is not str or type(unit_id) is not str:
            raise TypeError("alpha_max_precompute_unit_identity_invalid")
        try:
            index = self._allowed_units().index((unit_kind, unit_id))
        except ValueError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_precompute_unit_identity_invalid"
            ) from exc
        return f"{index:02d}-{unit_kind}-{unit_id}"

    def _initialize(self) -> None:
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{self._root.name}.staging-",
                dir=self._root.parent,
            )
        )
        try:
            (stage / "units").mkdir(mode=0o700)
            _write_bundle_file(stage / "units", ".transaction.lock", b"")
            os.chmod(stage / "units/.transaction.lock", 0o600)
            _write_bundle_file(stage, "ATTEMPT.json", self._descriptor_bytes)
            os.chmod(stage / "ATTEMPT.json", 0o444)
            _fsync_directory(stage / "units")
            _fsync_directory(stage)
            _rename_bundle_noreplace(stage, self._root)
            _fsync_directory(self._root.parent)
        except Exception:
            _cleanup_partial_bundle(stage)
            raise

    @contextmanager
    def _transaction(self) -> Any:
        """Serialize short cross-process journal operations, never replay."""
        try:
            fd = os.open(
                ".transaction.lock",
                os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._units_fd,
            )
            lock_status = os.stat(".transaction.lock", dir_fd=self._units_fd, follow_symlinks=False)
            open_status = os.fstat(fd)
            if (
                (int(lock_status.st_dev), int(lock_status.st_ino))
                != self._transaction_lock_identity
                or (int(open_status.st_dev), int(open_status.st_ino))
                != self._transaction_lock_identity
                or lock_status.st_nlink != 1
                or stat.S_ISLNK(lock_status.st_mode)
                or stat.S_IMODE(lock_status.st_mode) != 0o600
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_precompute_transaction_lock_invalid")
            fcntl.flock(fd, fcntl.LOCK_EX)
            lock_after = os.stat(".transaction.lock", dir_fd=self._units_fd, follow_symlinks=False)
            if (
                (int(lock_after.st_dev), int(lock_after.st_ino)) != self._transaction_lock_identity
                or (int(os.fstat(fd).st_dev), int(os.fstat(fd).st_ino))
                != self._transaction_lock_identity
                or lock_after.st_nlink != 1
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_precompute_transaction_lock_invalid")
        except OSError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_precompute_transaction_lock_invalid"
            ) from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _validate_root(self) -> None:
        try:
            display_root_status = self._display_root.lstat()
            display_units_status = (self._display_root / "units").lstat()
            open_root_status = os.fstat(self._root_fd)
            open_units_status = os.fstat(self._units_fd)
            status = open_root_status
            if (
                (int(display_root_status.st_dev), int(display_root_status.st_ino))
                != self._root_identity
                or (int(open_root_status.st_dev), int(open_root_status.st_ino))
                != self._root_identity
                or (int(display_units_status.st_dev), int(display_units_status.st_ino))
                != self._units_identity
                or (int(open_units_status.st_dev), int(open_units_status.st_ino))
                != self._units_identity
                or not stat.S_ISDIR(status.st_mode)
                or not stat.S_ISDIR(display_root_status.st_mode)
                or not stat.S_ISDIR(display_units_status.st_mode)
                or stat.S_ISLNK(status.st_mode)
                or stat.S_ISLNK(display_root_status.st_mode)
                or stat.S_ISLNK(display_units_status.st_mode)
                or not {path.name for path in self._root.iterdir()}
                <= {"ATTEMPT.json", "FAILED.json", "units"}
                or not stat.S_ISDIR(open_units_status.st_mode)
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_precompute_root_invalid")
            attempt_receipt, attempt_bytes = read_artifact_bytes(
                self._root / "ATTEMPT.json",
                artifact_id="precompute-attempt-descriptor",
            )
            attempt_status = (self._root / "ATTEMPT.json").lstat()
            if (
                attempt_bytes != self._descriptor_bytes
                or attempt_receipt.sha256 != self._descriptor_sha256
                or attempt_status.st_nlink != 1
                or attempt_status.st_mode & 0o222
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_precompute_descriptor_mismatch")
            failed = self._root / "FAILED.json"
            if failed.exists() or failed.is_symlink():
                payload = _strict_json_object(
                    _alpha_max_read_regular_at(self._root_fd, "FAILED.json", expected_mode=0o400)
                )
                if payload != {
                    "artifact_kind": "alpha_max_precompute_attempt_failed.v1",
                    "attempt_descriptor_sha256": self._attempt_descriptor_sha256,
                    "success": False,
                }:
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_precompute_failure_marker_invalid"
                    )
                raise AlphaMaxRuntimeContractError("alpha_max_precompute_attempt_poisoned")
            allowed_names = {
                self._unit_name(unit_kind, unit_id) for unit_kind, unit_id in self._allowed_units()
            }
            for path in self._units.iterdir():
                if path.name == ".transaction.lock":
                    status = path.lstat()
                    if (
                        not stat.S_ISREG(status.st_mode)
                        or stat.S_ISLNK(status.st_mode)
                        or stat.S_IMODE(status.st_mode) != 0o600
                        or status.st_nlink != 1
                    ):
                        raise AlphaMaxRuntimeContractError("alpha_max_precompute_inventory_invalid")
                    continue
                if path.name.startswith(".") and ".staging-" in path.name:
                    base = path.name[1:].split(".staging-", 1)[0]
                    if (base not in allowed_names) or re.fullmatch(
                        rf"\.{re.escape(base)}\.staging-[a-z0-9_]{{8}}",
                        path.name,
                    ) is None:
                        raise AlphaMaxRuntimeContractError("alpha_max_precompute_inventory_invalid")
                    _alpha_max_cleanup_recognized_staging_bundle(
                        path,
                        allowed_files=frozenset({"DATA.json", "SEALED.json"}),
                        allowed_directories=frozenset(),
                        error_token="alpha_max_precompute_inventory_invalid",
                    )
                    continue
                if path.name not in allowed_names:
                    raise AlphaMaxRuntimeContractError("alpha_max_precompute_inventory_invalid")
        except OSError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_root_invalid") from exc

    def _expected_seal(
        self,
        *,
        data_bytes: bytes,
        unit_kind: str,
        unit_id: str,
    ) -> dict[str, object]:
        return {
            "artifact_kind": "alpha_max_precompute_checkpoint_seal.v1",
            "attempt_descriptor_sha256": self._attempt_descriptor_sha256,
            "byte_count": len(data_bytes),
            "data_sha256": _sha256(data_bytes),
            "precompute_descriptor_sha256": self._descriptor_sha256,
            "runtime_identity_sha256": self._runtime_identity_sha256,
            "success": True,
            "unit_id": unit_id,
            "unit_kind": unit_kind,
            "unit_name": self._unit_name(unit_kind, unit_id),
        }

    def _load_unlocked(self, *, unit_kind: str, unit_id: str) -> bytes | None:
        self._validate_root()
        target = self._units / self._unit_name(unit_kind, unit_id)
        if not target.exists() and not target.is_symlink():
            return None
        try:
            status = target.lstat()
            if (
                not stat.S_ISDIR(status.st_mode)
                or stat.S_ISLNK(status.st_mode)
                or status.st_mode & 0o222
                or {path.name for path in target.iterdir()} != {"DATA.json", "SEALED.json"}
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_precompute_unit_invalid")
            data_receipt, data_bytes = read_artifact_bytes(
                target / "DATA.json",
                artifact_id=f"precompute:{unit_kind}:{unit_id}:data",
            )
            seal_receipt, seal_bytes = read_artifact_bytes(
                target / "SEALED.json",
                artifact_id=f"precompute:{unit_kind}:{unit_id}:seal",
            )
            data_status = (target / "DATA.json").lstat()
            seal_status = (target / "SEALED.json").lstat()
        except OSError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_unit_invalid") from exc
        data_payload = _strict_json_object(data_bytes)
        seal_payload = _strict_json_object(seal_bytes)
        expected_seal = self._expected_seal(
            data_bytes=data_bytes,
            unit_kind=unit_kind,
            unit_id=unit_id,
        )
        if (
            data_bytes != _canonical_bytes(data_payload) + b"\n"
            or seal_bytes != _canonical_bytes(seal_payload) + b"\n"
            or seal_payload != expected_seal
            or data_receipt.byte_count != expected_seal["byte_count"]
            or data_receipt.sha256 != expected_seal["data_sha256"]
            or seal_receipt.byte_count != len(seal_bytes)
            or data_status.st_nlink != 1
            or seal_status.st_nlink != 1
            or data_status.st_mode & 0o222
            or seal_status.st_mode & 0o222
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_unit_seal_invalid")
        return data_bytes

    def _seal_unlocked(self, *, unit_kind: str, unit_id: str, data_bytes: bytes) -> bytes:
        self._validate_root()
        if type(data_bytes) is not bytes:
            raise TypeError("alpha_max_precompute_data_bytes_invalid")
        data_payload = _strict_json_object(data_bytes)
        if data_bytes != _canonical_bytes(data_payload) + b"\n":
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_data_noncanonical")
        target = self._units / self._unit_name(unit_kind, unit_id)
        if target.exists() or target.is_symlink():
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_unit_exists")
        seal_bytes = (
            _canonical_bytes(
                self._expected_seal(
                    data_bytes=data_bytes,
                    unit_kind=unit_kind,
                    unit_id=unit_id,
                )
            )
            + b"\n"
        )
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.staging-",
                dir=target.parent,
            )
        )
        try:
            _write_bundle_file(stage, "DATA.json", data_bytes)
            _write_bundle_file(stage, "SEALED.json", seal_bytes)
            os.chmod(stage / "DATA.json", 0o444)
            os.chmod(stage / "SEALED.json", 0o444)
            os.chmod(stage, 0o555)
            _fsync_directory(stage)
            _rename_bundle_noreplace(stage, target)
            _fsync_directory(target.parent)
        except Exception:
            _cleanup_partial_bundle(stage)
            raise
        loaded = self._load_unlocked(unit_kind=unit_kind, unit_id=unit_id)
        if loaded != data_bytes:
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_publish_invalid")
        return loaded

    def load(self, *, unit_kind: str, unit_id: str) -> bytes | None:
        with self._transaction():
            return self._load_unlocked(unit_kind=unit_kind, unit_id=unit_id)

    def seal(self, *, unit_kind: str, unit_id: str, data_bytes: bytes) -> bytes:
        with self._transaction():
            return self._seal_unlocked(
                unit_kind=unit_kind,
                unit_id=unit_id,
                data_bytes=data_bytes,
            )

    def poison(self) -> None:
        """Durably reject semantic-failure progress; SIGKILL cannot call this."""
        with self._transaction():
            self._validate_root()
            marker = self._root / "FAILED.json"
            payload = (
                _canonical_bytes(
                    {
                        "artifact_kind": "alpha_max_precompute_attempt_failed.v1",
                        "attempt_descriptor_sha256": self._attempt_descriptor_sha256,
                        "success": False,
                    }
                )
                + b"\n"
            )
            if marker.exists() or marker.is_symlink():
                raise AlphaMaxRuntimeContractError("alpha_max_precompute_attempt_poisoned")
            _write_bundle_file(marker.parent, marker.name, payload)
            os.chmod(marker, 0o400)
            _alpha_max_fsync_regular_nofollow(marker)
            _fsync_directory(marker.parent)


class _AlphaMaxCellCheckpointStore:
    """Atomic immutable whole-cell store bound to one exact attempt descriptor."""

    __slots__ = (
        "_attempt_role",
        "_bound_output_fd",
        "_bound_output_identity",
        "_cells",
        "_cells_fd",
        "_cells_identity",
        "_checkpoint_parent_fd",
        "_config_path",
        "_descriptor_sha256",
        "_descriptor_v2",
        "_display_output_root",
        "_display_root",
        "_domain",
        "_implementation_inventory",
        "_lock_fd",
        "_lock_identity",
        "_lock_name",
        "_output_parent_fd",
        "_output_root",
        "_physical_schedule_sha256",
        "_precompute_store",
        "_prelock_binding",
        "_root",
        "_root_fd",
        "_root_identity",
        "_runtime_identity",
        "_runtime_identity_sha256",
    )

    def __init__(
        self,
        root_value: str | os.PathLike[str],
        *,
        output_root: str | os.PathLike[str],
        descriptor: Mapping[str, object],
        config_bytes: bytes,
    ) -> None:
        raw_root = _require_exact_explicit_path(root_value)
        self._display_root = Path(raw_root)
        self._root_fd = -1
        self._cells_fd = -1
        self._bound_output_fd = -1
        parent = self._display_root.parent
        if str(parent.resolve(strict=True)) != str(parent):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_parent_invalid")
        descriptor_bytes = _canonical_bytes(dict(descriptor)) + b"\n"
        self._descriptor_sha256 = _sha256(descriptor_bytes)
        self._descriptor_v2 = (
            descriptor.get("artifact_kind") == "alpha_max_restartable_attempt_descriptor.v2"
        )
        self._attempt_role, self._domain = _alpha_max_validate_checkpoint_descriptor(descriptor)
        self._runtime_identity = (
            copy.deepcopy(descriptor["runtime_identity"]) if self._descriptor_v2 else None
        )
        self._runtime_identity_sha256 = (
            _alpha_max_checkpoint_runtime_identity_sha256(self._runtime_identity)
            if self._descriptor_v2
            else ""
        )
        if self._descriptor_v2 and self._runtime_identity != _alpha_max_indicator_runtime_binding():
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_runtime_identity_mismatch")
        self._physical_schedule_sha256 = (
            descriptor["physical_schedule_sha256"]
            if self._descriptor_v2
            else _sha256(
                _canonical_bytes(
                    [
                        {
                            "fold_id": fold_id,
                            "nominal_cost_bps": nominal,
                            "row_id": row_id,
                            "seed": alpha_max_common_rng_seed(fold_id, nominal),
                        }
                        for row_id, nominal, fold_id in _alpha_max_physical_fold_schedule(
                            self._domain
                        )
                    ]
                )
            )
        )
        self._prelock_binding = descriptor.get("prelock_binding")
        self._precompute_store: _AlphaMaxPrecomputeCheckpointStore | None = None
        self._implementation_inventory = _verify_alpha_max_checkpoint_implementation_inventory(
            descriptor.get("implementation_inventory")
        )
        output_target, output_parent = _validated_output_target(output_root)
        self._display_output_root = output_target
        output_descriptor = descriptor.get("output")
        checkpoint_descriptor = descriptor.get("checkpoint")
        if (
            type(output_descriptor) is not dict
            or output_descriptor.get("target") != str(output_target)
            or type(checkpoint_descriptor) is not dict
            or checkpoint_descriptor.get("root") != str(self._display_root)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_descriptor_path_mismatch")
        parent_flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        self._checkpoint_parent_fd = os.open(parent, parent_flags)
        self._output_parent_fd = os.open(output_parent, parent_flags)
        try:
            for field, path, fd, payload in (
                ("checkpoint", parent, self._checkpoint_parent_fd, checkpoint_descriptor),
                ("output", output_parent, self._output_parent_fd, output_descriptor),
            ):
                expected_identity = payload.get("parent_identity")
                if (
                    payload.get("parent") != str(path)
                    or type(expected_identity) is not list
                    or len(expected_identity) != 2
                    or any(type(value) is not int for value in expected_identity)
                ):
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_checkpoint_descriptor_parent_mismatch"
                    )
                status = os.fstat(fd)
                if (
                    not stat.S_ISDIR(status.st_mode)
                    or [int(status.st_dev), int(status.st_ino)] != expected_identity
                ):
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_checkpoint_descriptor_parent_mismatch"
                    )
        except Exception:
            os.close(self._checkpoint_parent_fd)
            os.close(self._output_parent_fd)
            self._checkpoint_parent_fd = -1
            self._output_parent_fd = -1
            raise
        self._root = Path(f"/proc/self/fd/{self._checkpoint_parent_fd}") / self._display_root.name
        anchored_output_parent = Path(f"/proc/self/fd/{self._output_parent_fd}")
        self._output_root = anchored_output_parent / output_target.name
        self._lock_name = f".{output_target.name}.alpha-max-restart.lock"
        lock_path = anchored_output_parent / self._lock_name
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        self._lock_fd = os.open(lock_path, flags, 0o600)
        try:
            status = os.fstat(self._lock_fd)
            if not stat.S_ISREG(status.st_mode) or int(status.st_nlink) != 1:
                raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_lock_invalid")
            self._lock_identity = (int(status.st_dev), int(status.st_ino))
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not self._root.exists() and not self._root.is_symlink():
                for path in self._root.parent.iterdir():
                    if re.fullmatch(
                        rf"\.{re.escape(self._root.name)}\.staging-[a-z0-9_]{{8}}",
                        path.name,
                    ):
                        _alpha_max_cleanup_recognized_staging_bundle(
                            path,
                            allowed_files=frozenset({"ATTEMPT.json", "inputs/config.json"}),
                            allowed_directories=frozenset({"cells", "inputs"}),
                            error_token="alpha_max_checkpoint_inventory_invalid",
                        )
                self._initialize(
                    descriptor_bytes=descriptor_bytes,
                    config_bytes=config_bytes,
                )
            root_status = os.stat(
                self._display_root.name,
                dir_fd=self._checkpoint_parent_fd,
                follow_symlinks=False,
            )
            self._root_fd = _alpha_max_open_directory_at(
                self._display_root.name,
                dir_fd=self._checkpoint_parent_fd,
            )
            _alpha_max_require_open_identity(self._root_fd, root_status, directory=True)
            try:
                fcntl.flock(self._root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_lock_unavailable") from exc
            cells_status = os.stat("cells", dir_fd=self._root_fd, follow_symlinks=False)
            self._cells_fd = _alpha_max_open_directory_at("cells", dir_fd=self._root_fd)
            _alpha_max_require_open_identity(self._cells_fd, cells_status, directory=True)
            self._root_identity = (int(root_status.st_dev), int(root_status.st_ino))
            self._cells_identity = (int(cells_status.st_dev), int(cells_status.st_ino))
            self._root = Path(f"/proc/self/fd/{self._root_fd}")
            self._cells = Path(f"/proc/self/fd/{self._cells_fd}")
            self._validate_root(
                descriptor_bytes=descriptor_bytes,
                config_bytes=config_bytes,
                allow_missing_precompute=True,
            )
            if self._descriptor_v2:
                nonterminal_days: tuple[str, ...] = ()
                if self._attempt_role == "prelock":
                    train_window = descriptor["phase_windows"]["train"]
                    train_start = datetime.fromisoformat(
                        str(train_window["start_utc"]).replace("Z", "+00:00")
                    ).astimezone(UTC)
                    train_end = datetime.fromisoformat(
                        str(train_window["end_utc"]).replace("Z", "+00:00")
                    ).astimezone(UTC)
                    if train_start >= train_end or any(
                        value != 0
                        for value in (
                            train_start.hour,
                            train_start.minute,
                            train_start.second,
                            train_start.microsecond,
                            train_end.hour,
                            train_end.minute,
                            train_end.second,
                            train_end.microsecond,
                        )
                    ):
                        raise AlphaMaxRuntimeContractError(
                            "alpha_max_precompute_training_schedule_invalid"
                        )
                    nonterminal_days = tuple(
                        f"{component_id}--{day:%Y%m%d}"
                        for component_id in (
                            "component_carry_1x",
                            "component_near_high_1x",
                            "component_trend_1x",
                        )
                        for day in (
                            train_start + timedelta(days=index)
                            for index in range((train_end - train_start).days - 1)
                        )
                    )
                self._precompute_store = _AlphaMaxPrecomputeCheckpointStore(
                    self._root / "precompute",
                    attempt_descriptor_sha256=self._descriptor_sha256,
                    attempt_role=self._attempt_role,
                    domain=self._domain,
                    runtime_identity_sha256=self._runtime_identity_sha256,
                    training_day_ids=nonterminal_days,
                )
                self._validate_root(
                    descriptor_bytes=descriptor_bytes,
                    config_bytes=config_bytes,
                    allow_missing_precompute=False,
                )
        except Exception:
            for field in ("_cells_fd", "_root_fd"):
                fd = getattr(self, field, -1)
                if fd >= 0:
                    os.close(fd)
                    setattr(self, field, -1)
            os.close(self._lock_fd)
            self._lock_fd = -1
            os.close(self._checkpoint_parent_fd)
            os.close(self._output_parent_fd)
            self._checkpoint_parent_fd = -1
            self._output_parent_fd = -1
            raise
        self._config_path = self._display_root / "inputs" / "config.json"

    def __del__(self) -> None:
        lock_fd = getattr(self, "_lock_fd", -1)
        if lock_fd >= 0:
            os.close(lock_fd)
            self._lock_fd = -1
        for field in (
            "_cells_fd",
            "_root_fd",
            "_bound_output_fd",
            "_checkpoint_parent_fd",
            "_output_parent_fd",
        ):
            fd = getattr(self, field, -1)
            if fd >= 0:
                os.close(fd)
                setattr(self, field, -1)

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def display_output_root(self) -> Path:
        return self._display_output_root

    def bind_output_root(self) -> Path:
        try:
            observed = os.stat(
                self._display_output_root.name,
                dir_fd=self._output_parent_fd,
                follow_symlinks=False,
            )
            if self._bound_output_fd < 0:
                self._bound_output_fd = _alpha_max_open_directory_at(
                    self._display_output_root.name,
                    dir_fd=self._output_parent_fd,
                )
                _alpha_max_require_open_identity(
                    self._bound_output_fd,
                    observed,
                    directory=True,
                )
                self._bound_output_identity = (int(observed.st_dev), int(observed.st_ino))
            opened = os.fstat(self._bound_output_fd)
            if (
                (int(observed.st_dev), int(observed.st_ino)) != self._bound_output_identity
                or (int(opened.st_dev), int(opened.st_ino)) != self._bound_output_identity
                or str(self._display_output_root.resolve(strict=True))
                != str(self._display_output_root)
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_output_root_identity_changed")
        except OSError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_output_root_identity_changed") from exc
        return self._output_root

    def _initialize(
        self,
        *,
        descriptor_bytes: bytes,
        config_bytes: bytes,
    ) -> None:
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{self._root.name}.staging-",
                dir=self._root.parent,
            )
        )
        try:
            (stage / "inputs").mkdir(mode=0o755)
            (stage / "cells").mkdir(mode=0o755)
            _write_bundle_file(stage, "inputs/config.json", config_bytes)
            _write_bundle_file(stage, "ATTEMPT.json", descriptor_bytes)
            os.chmod(stage / "inputs" / "config.json", 0o444)
            os.chmod(stage / "ATTEMPT.json", 0o444)
            _fsync_directory(stage / "inputs")
            _fsync_directory(stage)
            _rename_bundle_noreplace(stage, self._root)
            os.fsync(self._checkpoint_parent_fd)
        except Exception:
            _cleanup_partial_bundle(stage)
            raise

    def _validate_root(
        self,
        *,
        descriptor_bytes: bytes,
        config_bytes: bytes,
        allow_missing_precompute: bool,
    ) -> None:
        self._validate_open_checkpoint_identity()
        try:
            display_root_status = self._display_root.lstat()
            display_cells_status = (self._display_root / "cells").lstat()
            open_root_status = os.fstat(self._root_fd)
            open_cells_status = os.fstat(self._cells_fd)
            if (
                (int(display_root_status.st_dev), int(display_root_status.st_ino))
                != self._root_identity
                or (int(open_root_status.st_dev), int(open_root_status.st_ino))
                != self._root_identity
                or (int(display_cells_status.st_dev), int(display_cells_status.st_ino))
                != self._cells_identity
                or (int(open_cells_status.st_dev), int(open_cells_status.st_ino))
                != self._cells_identity
                or not stat.S_ISDIR(display_root_status.st_mode)
                or not stat.S_ISDIR(display_cells_status.st_mode)
                or not stat.S_ISDIR(open_root_status.st_mode)
                or not stat.S_ISDIR(open_cells_status.st_mode)
                or stat.S_ISLNK(display_root_status.st_mode)
                or stat.S_ISLNK(display_cells_status.st_mode)
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_root_invalid")
            if self._descriptor_v2 and allow_missing_precompute:
                for path in self._root.iterdir():
                    if re.fullmatch(r"\.precompute\.staging-[a-z0-9_]{8}", path.name):
                        _alpha_max_cleanup_recognized_staging_bundle(
                            path,
                            allowed_files=frozenset({"ATTEMPT.json"}),
                            allowed_directories=frozenset({"units"}),
                            error_token="alpha_max_checkpoint_inventory_invalid",
                        )
            names = {path.name for path in self._root.iterdir()}
            base_names = {"ATTEMPT.json", "cells", "inputs"}
            expected_names = (
                (base_names, base_names | {"precompute"})
                if self._descriptor_v2 and allow_missing_precompute
                else (base_names | ({"precompute"} if self._descriptor_v2 else set()),)
            )
            if names not in expected_names:
                raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_inventory_invalid")
            inputs = self._root / "inputs"
            cells = self._cells
            if (
                not inputs.is_dir()
                or inputs.is_symlink()
                or not cells.is_dir()
                or {path.name for path in inputs.iterdir()} != {"config.json"}
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_inventory_invalid")
            attempt_receipt, observed_descriptor = read_artifact_bytes(
                self._root / "ATTEMPT.json",
                artifact_id="checkpoint-attempt-descriptor",
            )
            config_receipt, observed_config = read_artifact_bytes(
                inputs / "config.json",
                artifact_id="checkpoint-config",
            )
        except OSError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_root_invalid") from exc
        if (
            observed_descriptor != descriptor_bytes
            or _sha256(observed_descriptor) != self._descriptor_sha256
            or observed_config != config_bytes
            or attempt_receipt.byte_count != len(descriptor_bytes)
            or config_receipt.byte_count != len(config_bytes)
            or (self._root / "ATTEMPT.json").stat().st_mode & 0o222
            or (inputs / "config.json").stat().st_mode & 0o222
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_descriptor_mismatch")
        allowed_cells = {
            self._cell_name(row_id, nominal)
            for row_id in _ALPHA_MAX_RESOLVABLE_ROWS
            for nominal in ALPHA_MAX_COST_CELL_BPS
        }
        for path in cells.iterdir():
            if path.name.startswith(".") and ".staging-" in path.name:
                base = path.name[1:].split(".staging-", 1)[0]
                if (
                    base not in allowed_cells
                    or re.fullmatch(
                        rf"\.{re.escape(base)}\.staging-[a-z0-9_]{{8}}",
                        path.name,
                    )
                    is None
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_inventory_invalid")
                _alpha_max_cleanup_recognized_staging_bundle(
                    path,
                    allowed_files=frozenset({"EVIDENCE.json", "SEALED.json"}),
                    allowed_directories=frozenset(),
                    error_token="alpha_max_checkpoint_inventory_invalid",
                )
                continue
            try:
                cell_status = path.lstat()
            except OSError as exc:
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_checkpoint_inventory_invalid"
                ) from exc
            if (
                path.name not in allowed_cells
                or not stat.S_ISDIR(cell_status.st_mode)
                or stat.S_ISLNK(cell_status.st_mode)
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_inventory_invalid")

    def load_precompute(self, *, unit_kind: str, unit_id: str) -> bytes | None:
        if self._precompute_store is None:
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_store_unavailable")
        _verify_alpha_max_checkpoint_implementation_inventory(self._implementation_inventory)
        self._verify_runtime_identity()
        self._validate_open_checkpoint_identity()
        return self._precompute_store.load(unit_kind=unit_kind, unit_id=unit_id)

    def training_precompute_store(self) -> _AlphaMaxPrecomputeCheckpointStore:
        """Expose the descriptor-bound journal to daily training replay only."""
        if self._precompute_store is None:
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_store_unavailable")
        _verify_alpha_max_checkpoint_implementation_inventory(self._implementation_inventory)
        self._verify_runtime_identity()
        self._validate_open_checkpoint_identity()
        return self._precompute_store

    def seal_precompute(
        self,
        *,
        unit_kind: str,
        unit_id: str,
        data_bytes: bytes,
    ) -> bytes:
        if self._precompute_store is None:
            raise AlphaMaxRuntimeContractError("alpha_max_precompute_store_unavailable")
        _verify_alpha_max_checkpoint_implementation_inventory(self._implementation_inventory)
        self._verify_runtime_identity()
        self._validate_open_checkpoint_identity()
        return self._precompute_store.seal(
            unit_kind=unit_kind,
            unit_id=unit_id,
            data_bytes=data_bytes,
        )

    def _verify_runtime_identity(self) -> None:
        if self._descriptor_v2 and self._runtime_identity != _alpha_max_indicator_runtime_binding():
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_runtime_identity_mismatch")

    def _validate_open_checkpoint_identity(self) -> None:
        try:
            if str(self._display_root.resolve(strict=True)) != str(self._display_root) or str(
                self._display_output_root.parent.resolve(strict=True)
            ) != str(self._display_output_root.parent):
                raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_root_invalid")
            lock_path = os.stat(
                self._lock_name,
                dir_fd=self._output_parent_fd,
                follow_symlinks=False,
            )
            open_lock = os.fstat(self._lock_fd)
            display_root = self._display_root.lstat()
            display_cells = (self._display_root / "cells").lstat()
            open_root = os.fstat(self._root_fd)
            open_cells = os.fstat(self._cells_fd)
        except OSError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_root_invalid") from exc
        if (
            (int(display_root.st_dev), int(display_root.st_ino)) != self._root_identity
            or (int(open_root.st_dev), int(open_root.st_ino)) != self._root_identity
            or (int(display_cells.st_dev), int(display_cells.st_ino)) != self._cells_identity
            or (int(open_cells.st_dev), int(open_cells.st_ino)) != self._cells_identity
            or (int(lock_path.st_dev), int(lock_path.st_ino)) != self._lock_identity
            or (int(open_lock.st_dev), int(open_lock.st_ino)) != self._lock_identity
            or not stat.S_ISREG(lock_path.st_mode)
            or stat.S_ISLNK(lock_path.st_mode)
            or int(lock_path.st_nlink) != 1
            or not stat.S_ISDIR(display_root.st_mode)
            or not stat.S_ISDIR(display_cells.st_mode)
            or stat.S_ISLNK(display_root.st_mode)
            or stat.S_ISLNK(display_cells.st_mode)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_root_invalid")

    @staticmethod
    def _cell_name(row_id: str, nominal_cost_bps: int) -> str:
        if (
            row_id not in _ALPHA_MAX_RESOLVABLE_ROWS
            or nominal_cost_bps not in ALPHA_MAX_COST_CELL_BPS
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_cell_identity_invalid")
        index = _ALPHA_MAX_RESOLVABLE_ROWS.index(row_id) * len(
            ALPHA_MAX_COST_CELL_BPS
        ) + ALPHA_MAX_COST_CELL_BPS.index(nominal_cost_bps)
        return f"{index:02d}-{row_id}-{nominal_cost_bps:02d}"

    @staticmethod
    def _current_receipts(
        prepared: _AlphaMaxPreparedReplayRow,
    ) -> tuple[
        dict[str, AlphaMaxCapsuleReceipt],
        dict[str, AlphaMaxRootReceipt],
    ]:
        capsules = {
            value.capsule_receipt.sha256: value.capsule_receipt for value in prepared.fold_inputs
        }
        root_seals = {
            (seal.root_id, seal.root_kind, seal.content_sha256): seal
            for value in prepared.fold_inputs
            for seal in (*value.raw_root_seals, *value.feature_root_seals)
        }
        roots = {
            f"{root_id}:{root_kind}:{content_sha256}": seal.to_receipt()
            for (root_id, root_kind, content_sha256), seal in root_seals.items()
        }
        return capsules, roots

    def _expected_cell_seal(
        self,
        *,
        evidence_bytes: bytes,
        row_id: str,
        nominal_cost_bps: int,
        preflight: AlphaMaxRuntimePreflight,
        prepared: _AlphaMaxPreparedReplayRow,
    ) -> dict[str, object]:
        capsules, roots = self._current_receipts(prepared)
        fold_ids = list(_alpha_max_fold_ids(self._domain))
        if not self._descriptor_v2:
            return {
                "artifact_kind": "alpha_max_restartable_cost_cell_seal.v1",
                "attempt_descriptor_sha256": self._descriptor_sha256,
                "byte_count": len(evidence_bytes),
                "capsule_receipt_sha256s": sorted(capsules),
                "cell_name": self._cell_name(row_id, nominal_cost_bps),
                "config_sha256": preflight.config_receipt.sha256,
                "domain": self._domain,
                "evidence_sha256": _sha256(evidence_bytes),
                "manifest_sha256": prepared.manifest_receipt.sha256,
                "nominal_cost_bps": nominal_cost_bps,
                "root_receipt_identities": sorted(roots),
                "row_id": row_id,
                "runtime_contract_sha256": preflight.runtime_contract_sha256,
                "success": True,
            }
        return {
            "artifact_kind": "alpha_max_restartable_cost_cell_seal.v2",
            "attempt_descriptor_sha256": self._descriptor_sha256,
            "byte_count": len(evidence_bytes),
            "capsule_receipt_sha256s": sorted(capsules),
            "cell_name": self._cell_name(row_id, nominal_cost_bps),
            "config_sha256": preflight.config_receipt.sha256,
            "domain": self._domain,
            "evidence_sha256": _sha256(evidence_bytes),
            "fold_count": len(fold_ids),
            "fold_ids": fold_ids,
            "fold_run_set_sha256": _sha256(_canonical_bytes(fold_ids)),
            "manifest_sha256": prepared.manifest_receipt.sha256,
            "nominal_cost_bps": nominal_cost_bps,
            "physical_schedule_sha256": self._physical_schedule_sha256,
            "prelock_binding": self._prelock_binding,
            "root_receipt_identities": sorted(roots),
            "row_id": row_id,
            "runtime_identity_sha256": self._runtime_identity_sha256,
            "runtime_contract_sha256": preflight.runtime_contract_sha256,
            "success": True,
        }

    def load(
        self,
        *,
        row_id: str,
        nominal_cost_bps: int,
        preflight: AlphaMaxRuntimePreflight,
        prepared: _AlphaMaxPreparedReplayRow,
    ) -> AlphaMaxCostCellPreGateEvidence | None:
        _verify_alpha_max_checkpoint_implementation_inventory(self._implementation_inventory)
        self._verify_runtime_identity()
        self._validate_open_checkpoint_identity()
        target = self._cells / self._cell_name(row_id, nominal_cost_bps)
        if not target.exists() and not target.is_symlink():
            return None
        status = target.lstat()
        if (
            not stat.S_ISDIR(status.st_mode)
            or stat.S_ISLNK(status.st_mode)
            or status.st_mode & 0o222
            or {path.name for path in target.iterdir()} != {"EVIDENCE.json", "SEALED.json"}
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_cell_invalid")
        evidence_receipt, evidence_bytes = read_artifact_bytes(
            target / "EVIDENCE.json",
            artifact_id=f"checkpoint-cell:{row_id}:{nominal_cost_bps}:evidence",
        )
        seal_receipt, seal_bytes = read_artifact_bytes(
            target / "SEALED.json",
            artifact_id=f"checkpoint-cell:{row_id}:{nominal_cost_bps}:seal",
        )
        seal_payload = _strict_json_object(seal_bytes)
        expected_seal = self._expected_cell_seal(
            evidence_bytes=evidence_bytes,
            row_id=row_id,
            nominal_cost_bps=nominal_cost_bps,
            preflight=preflight,
            prepared=prepared,
        )
        if (
            seal_bytes != _canonical_bytes(seal_payload) + b"\n"
            or seal_payload != expected_seal
            or evidence_receipt.byte_count != expected_seal["byte_count"]
            or evidence_receipt.sha256 != expected_seal["evidence_sha256"]
            or seal_receipt.byte_count != len(seal_bytes)
            or (target / "EVIDENCE.json").stat().st_mode & 0o222
            or (target / "SEALED.json").stat().st_mode & 0o222
            or int((target / "EVIDENCE.json").stat().st_nlink) != 1
            or int((target / "SEALED.json").stat().st_nlink) != 1
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_cell_seal_invalid")
        evidence_payload = _strict_json_object(evidence_bytes)
        capsules, roots = self._current_receipts(prepared)
        try:
            evidence = parse_alpha_max_cost_cell_pre_gate_evidence(
                evidence_payload,
                manifest_receipt=prepared.manifest_receipt,
                config_receipt=preflight.config_receipt,
                capsule_receipts_by_sha256=capsules,
                root_receipts_by_identity=roots,
                runtime_contract_sha256=preflight.runtime_contract_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_cell_parse_invalid") from exc
        if (
            evidence.canonical_bytes != evidence_bytes
            or evidence.row_id != row_id
            or evidence.domain != self._domain
            or evidence.nominal_cost_bps != nominal_cost_bps
            or evidence.sha256 != expected_seal["evidence_sha256"]
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_cell_binding_invalid")
        return evidence

    def seal(
        self,
        evidence: AlphaMaxCostCellPreGateEvidence,
        *,
        preflight: AlphaMaxRuntimePreflight,
        prepared: _AlphaMaxPreparedReplayRow,
    ) -> AlphaMaxCostCellPreGateEvidence:
        _verify_alpha_max_checkpoint_implementation_inventory(self._implementation_inventory)
        self._verify_runtime_identity()
        if type(evidence) is not AlphaMaxCostCellPreGateEvidence:
            raise TypeError("alpha_max_checkpoint_cell_evidence_invalid")
        row_id = evidence.row_id
        nominal_cost_bps = evidence.nominal_cost_bps
        self._validate_open_checkpoint_identity()
        target = self._cells / self._cell_name(row_id, nominal_cost_bps)
        if target.exists() or target.is_symlink():
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_cell_exists")
        evidence_bytes = evidence.canonical_bytes
        seal_bytes = (
            _canonical_bytes(
                self._expected_cell_seal(
                    evidence_bytes=evidence_bytes,
                    row_id=row_id,
                    nominal_cost_bps=nominal_cost_bps,
                    preflight=preflight,
                    prepared=prepared,
                )
            )
            + b"\n"
        )
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.staging-",
                dir=target.parent,
            )
        )
        try:
            _write_bundle_file(stage, "EVIDENCE.json", evidence_bytes)
            _write_bundle_file(stage, "SEALED.json", seal_bytes)
            os.chmod(stage / "EVIDENCE.json", 0o444)
            os.chmod(stage / "SEALED.json", 0o444)
            os.chmod(stage, 0o555)
            _fsync_directory(stage)
            _rename_bundle_noreplace(stage, target)
            _fsync_directory(target.parent)
        except Exception:
            _cleanup_partial_bundle(stage)
            raise
        loaded = self.load(
            row_id=row_id,
            nominal_cost_bps=nominal_cost_bps,
            preflight=preflight,
            prepared=prepared,
        )
        if loaded is None:
            raise AlphaMaxRuntimeContractError("alpha_max_checkpoint_cell_publish_missing")
        return loaded


def _alpha_max_manifest_checkpoint_identity(
    manifest: AlphaMaxManifestReceipt,
) -> dict[str, object]:
    if type(manifest) is not AlphaMaxManifestReceipt:
        raise TypeError("alpha_max_precompute_manifest_identity_invalid")
    return {
        "byte_count": manifest.byte_count,
        "phase": manifest.phase,
        "relative_path": manifest.relative_path,
        "row_id": manifest.row_id,
        "sha256": manifest.sha256,
    }


def _alpha_max_validate_manifest_checkpoint_identity(
    value: object,
    manifest: AlphaMaxManifestReceipt,
) -> None:
    if value != _alpha_max_manifest_checkpoint_identity(manifest):
        raise AlphaMaxRuntimeContractError("alpha_max_precompute_manifest_identity_mismatch")


def _alpha_max_training_component_checkpoint_bytes(
    *,
    component_id: str,
    manifest: AlphaMaxManifestReceipt,
    calendar: tuple[str, ...],
    returns: tuple[float, ...],
    native_finalization: AlphaMaxNativeFinalizationReceipt,
) -> bytes:
    if (
        component_id
        not in {
            "component_carry_1x",
            "component_near_high_1x",
            "component_trend_1x",
        }
        or manifest.row_id != component_id
        or type(calendar) is not tuple
        or type(returns) is not tuple
        or len(calendar) != len(returns)
        or len(calendar) < 252
        or len(calendar) != len(set(calendar))
        or any(type(value) is not str or not value for value in calendar)
        or any(type(value) is not float or not math.isfinite(value) for value in returns)
        or type(native_finalization) is not AlphaMaxNativeFinalizationReceipt
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_checkpoint_value_invalid")
    return (
        _canonical_bytes(
            {
                "artifact_kind": "alpha_max_training_component_checkpoint.v1",
                "calendar": list(calendar),
                "component_id": component_id,
                "manifest": _alpha_max_manifest_checkpoint_identity(manifest),
                "native_finalization": native_finalization.to_payload(),
                "returns_hex": [value.hex() for value in returns],
            }
        )
        + b"\n"
    )


def _alpha_max_training_component_from_checkpoint(
    payload: bytes,
    *,
    preflight: AlphaMaxRuntimePreflight,
    component_id: str,
    manifest: AlphaMaxManifestReceipt,
) -> tuple[tuple[str, ...], tuple[float, ...], AlphaMaxNativeFinalizationReceipt]:
    value = _strict_json_object(payload)
    if (
        payload != _canonical_bytes(value) + b"\n"
        or set(value)
        != {
            "artifact_kind",
            "calendar",
            "component_id",
            "manifest",
            "native_finalization",
            "returns_hex",
        }
        or value["artifact_kind"] != "alpha_max_training_component_checkpoint.v1"
        or value["component_id"] != component_id
        or type(value["calendar"]) is not list
        or type(value["returns_hex"]) is not list
        or len(value["calendar"]) != len(value["returns_hex"])
        or len(value["calendar"]) < 252
        or len(value["calendar"]) != len(set(value["calendar"]))
        or any(type(item) is not str or not item for item in value["calendar"])
        or any(type(item) is not str for item in value["returns_hex"])
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_checkpoint_parse_invalid")
    _alpha_max_validate_manifest_checkpoint_identity(value["manifest"], manifest)
    train_window = preflight.phase_windows["train"]
    train_start = datetime.fromisoformat(train_window.start_utc.replace("Z", "+00:00")).astimezone(
        UTC
    )
    train_end = datetime.fromisoformat(train_window.end_utc.replace("Z", "+00:00")).astimezone(UTC)
    expected_calendar = tuple(
        (train_start + timedelta(days=offset + 1)).date().isoformat()
        for offset in range((train_end - train_start).days)
    )
    if tuple(value["calendar"]) != expected_calendar:
        raise AlphaMaxRuntimeContractError("alpha_max_training_checkpoint_parse_invalid")
    try:
        returns = tuple(float.fromhex(item) for item in value["returns_hex"])
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_training_checkpoint_parse_invalid") from exc
    if any(
        type(item) is not float or not math.isfinite(item) or item.hex() != encoded
        for item, encoded in zip(returns, value["returns_hex"], strict=True)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_checkpoint_parse_invalid")
    native = value["native_finalization"]
    if (
        type(native) is not dict
        or set(native)
        != {
            "artifact_kind",
            "boundary_utc",
            "discarded_signal_count",
            "discarded_signal_sha256",
            "finalized_children",
            "native_coverage_by_child",
        }
        or native["artifact_kind"] != "alpha_max_native_finalization_receipt.v1"
        or type(native["boundary_utc"]) is not str
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_checkpoint_parse_invalid")
    try:
        boundary = datetime.fromisoformat(native["boundary_utc"].replace("Z", "+00:00"))
        finalization = build_alpha_max_native_finalization_receipt(
            boundary_utc=boundary,
            finalized_children=native["finalized_children"],
            native_coverage_by_child=native["native_coverage_by_child"],
            discarded_signal_count=native["discarded_signal_count"],
            discarded_signal_sha256=native["discarded_signal_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_training_checkpoint_parse_invalid") from exc
    if (
        finalization.canonical_bytes != _canonical_bytes(native) + b"\n"
        or finalization.boundary_utc != train_end
        or set(finalization.finalized_children) != {component_id}
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_checkpoint_parse_invalid")
    return tuple(value["calendar"]), returns, finalization


def _alpha_max_prepared_row_checkpoint_bytes(
    prepared: _AlphaMaxPreparedReplayRow,
    *,
    domain: str,
) -> bytes:
    if (
        type(prepared) is not _AlphaMaxPreparedReplayRow
        or tuple(value.fold_id for value in prepared.fold_inputs) != _alpha_max_fold_ids(domain)
        or type(prepared.gross) is not float
        or not math.isfinite(prepared.gross)
        or prepared.gross <= 0.0
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_prepared_checkpoint_value_invalid")
    capsule_identities: list[dict[str, object]] = []
    for value in prepared.fold_inputs:
        receipt, envelope = read_artifact_bytes(
            value.capsule_receipt.path,
            artifact_id=f"precompute-capsule:{prepared.manifest_receipt.row_id}:{value.fold_id}",
        )
        if (
            receipt.sha256 != value.capsule_receipt.sha256
            or receipt.byte_count != value.capsule_receipt.byte_count
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prepared_checkpoint_capsule_mismatch")
        capsule_identities.append(
            {
                "byte_count": value.capsule_receipt.byte_count,
                "envelope_base64": base64.b64encode(envelope).decode("ascii"),
                "prefix_id": value.fold_id,
                "relative_path": value.capsule_receipt.relative_path,
                "sha256": value.capsule_receipt.sha256,
            }
        )
    return (
        _canonical_bytes(
            {
                "artifact_kind": "alpha_max_prepared_replay_row_checkpoint.v1",
                "domain": domain,
                "fold_capsules": capsule_identities,
                "gross_hex": prepared.gross.hex(),
                "manifest": _alpha_max_manifest_checkpoint_identity(prepared.manifest_receipt),
                "row_id": prepared.manifest_receipt.row_id,
            }
        )
        + b"\n"
    )


def _alpha_max_restore_prepared_row_checkpoint(
    payload: bytes,
    *,
    preflight: AlphaMaxRuntimePreflight,
    manifest: AlphaMaxManifestReceipt,
    admitted_symbols: tuple[str, ...],
    root_seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    domain: str,
    gross: float,
    capsule_output_root: Path,
    initial_receipt: AlphaMaxCapsuleReceipt | None = None,
) -> _AlphaMaxPreparedReplayRow:
    value = _strict_json_object(payload)
    expected_fold_ids = _alpha_max_fold_ids(domain)
    if (
        payload != _canonical_bytes(value) + b"\n"
        or set(value)
        != {
            "artifact_kind",
            "domain",
            "fold_capsules",
            "gross_hex",
            "manifest",
            "row_id",
        }
        or value["artifact_kind"] != "alpha_max_prepared_replay_row_checkpoint.v1"
        or value["domain"] != domain
        or value["row_id"] != manifest.row_id
        or type(value["gross_hex"]) is not str
        or type(gross) is not float
        or not math.isfinite(gross)
        or value["gross_hex"] != gross.hex()
        or type(value["fold_capsules"]) is not list
        or len(value["fold_capsules"]) != len(expected_fold_ids)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_prepared_checkpoint_parse_invalid")
    _alpha_max_validate_manifest_checkpoint_identity(value["manifest"], manifest)
    root_id = _alpha_max_current_root_id(expected_fold_ids[0])
    raw_seal = root_seals[(root_id, "raw")]
    lookup = _alpha_max_phase_lookup(root_seals, expected_fold_ids[0])
    loader = _AlphaMaxBoundedRawLoader(raw_seal, admitted_symbols)
    fold_inputs: list[_AlphaMaxFoldReplayInput] = []
    for index, (fold_id, identity) in enumerate(
        zip(expected_fold_ids, value["fold_capsules"], strict=True)
    ):
        if (
            type(identity) is not dict
            or set(identity)
            != {
                "byte_count",
                "envelope_base64",
                "prefix_id",
                "relative_path",
                "sha256",
            }
            or identity["prefix_id"] != fold_id
            or type(identity["envelope_base64"]) is not str
            or type(identity["relative_path"]) is not str
            or type(identity["byte_count"]) is not int
            or identity["byte_count"] <= 0
            or type(identity["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prepared_checkpoint_parse_invalid")
        try:
            envelope = base64.b64decode(identity["envelope_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_prepared_checkpoint_parse_invalid"
            ) from exc
        if (
            base64.b64encode(envelope).decode("ascii") != identity["envelope_base64"]
            or len(envelope) != identity["byte_count"]
            or _sha256(envelope) != identity["sha256"]
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prepared_checkpoint_capsule_mismatch")
        if index == 0 and initial_receipt is not None:
            receipt = initial_receipt
            _receipt, observed_envelope = read_artifact_bytes(
                receipt.path,
                artifact_id=f"precompute-initial-capsule:{manifest.row_id}:{fold_id}",
            )
            if observed_envelope != envelope:
                raise AlphaMaxRuntimeContractError("alpha_max_prepared_checkpoint_capsule_mismatch")
        else:
            relative_path = _safe_bundle_relative_path(identity["relative_path"])
            capsule_path = capsule_output_root / relative_path
            if not capsule_path.exists() and not capsule_path.is_symlink():
                _write_bundle_file_atomic(
                    capsule_output_root,
                    relative_path,
                    envelope,
                )
            receipt = AlphaMaxCapsuleReceipt.from_path(
                capsule_path,
                row_id=manifest.row_id,
                phase=manifest.phase,
                prefix_id=fold_id,
                manifest_sha256=manifest.sha256,
                relative_path=identity["relative_path"],
            )
        if (
            receipt.prefix_id != fold_id
            or receipt.relative_path != identity["relative_path"]
            or receipt.sha256 != identity["sha256"]
            or receipt.byte_count != identity["byte_count"]
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prepared_checkpoint_capsule_mismatch")
        fold_inputs.append(
            _AlphaMaxFoldReplayInput(
                fold_id=fold_id,
                raw_root=raw_seal.path,
                ordered_lookup=lookup,
                indicator_capsule=_alpha_max_capsule_from_receipt(receipt),
                capsule_receipt=receipt,
                raw_root_seals=(raw_seal,),
                feature_root_seals=tuple(
                    root_seals[(feature_id, "feature")]
                    for feature_id in _alpha_max_expected_root_sequence(fold_id)
                ),
                bounded_raw_loader=loader,
            )
        )
    return _AlphaMaxPreparedReplayRow(
        manifest_receipt=manifest,
        fold_inputs=tuple(fold_inputs),
        gross=gross,
    )


def _alpha_max_final_refit_checkpoint_bytes(
    *,
    manifest: AlphaMaxManifestReceipt,
    receipt: AlphaMaxCapsuleReceipt,
    gross: float,
) -> bytes:
    if (
        manifest.row_id != receipt.row_id
        or manifest.phase != "prelock_final_refit"
        or receipt.phase != manifest.phase
        or receipt.manifest_sha256 != manifest.sha256
        or receipt.prefix_id != _ALPHA_MAX_HISTORICAL_FOLD_IDS[0]
        or type(gross) is not float
        or not math.isfinite(gross)
        or gross <= 0.0
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_final_refit_checkpoint_value_invalid")
    activation_receipt, envelope = read_artifact_bytes(
        receipt.path,
        artifact_id=f"precompute-final-refit-capsule:{manifest.row_id}",
    )
    if (
        activation_receipt.sha256 != receipt.sha256
        or activation_receipt.byte_count != receipt.byte_count
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_final_refit_checkpoint_capsule_mismatch")
    return (
        _canonical_bytes(
            {
                "artifact_kind": "alpha_max_final_refit_row_checkpoint.v1",
                "capsule": {
                    "byte_count": receipt.byte_count,
                    "envelope_base64": base64.b64encode(envelope).decode("ascii"),
                    "prefix_id": receipt.prefix_id,
                    "relative_path": receipt.relative_path,
                    "sha256": receipt.sha256,
                },
                "gross_hex": gross.hex(),
                "manifest": _alpha_max_manifest_checkpoint_identity(manifest),
                "row_id": manifest.row_id,
            }
        )
        + b"\n"
    )


def _alpha_max_restore_final_refit_checkpoint(
    payload: bytes,
    *,
    manifest: AlphaMaxManifestReceipt,
    capsule_root: Path,
    gross: float,
) -> AlphaMaxCapsuleReceipt:
    value = _strict_json_object(payload)
    capsule = value.get("capsule")
    if (
        payload != _canonical_bytes(value) + b"\n"
        or set(value)
        != {
            "artifact_kind",
            "capsule",
            "gross_hex",
            "manifest",
            "row_id",
        }
        or value["artifact_kind"] != "alpha_max_final_refit_row_checkpoint.v1"
        or value["row_id"] != manifest.row_id
        or type(value["gross_hex"]) is not str
        or type(gross) is not float
        or value["gross_hex"] != gross.hex()
        or type(capsule) is not dict
        or set(capsule)
        != {
            "byte_count",
            "envelope_base64",
            "prefix_id",
            "relative_path",
            "sha256",
        }
        or capsule["prefix_id"] != _ALPHA_MAX_HISTORICAL_FOLD_IDS[0]
        or type(capsule["envelope_base64"]) is not str
        or type(capsule["relative_path"]) is not str
        or type(capsule["byte_count"]) is not int
        or capsule["byte_count"] <= 0
        or type(capsule["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", capsule["sha256"]) is None
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_final_refit_checkpoint_parse_invalid")
    _alpha_max_validate_manifest_checkpoint_identity(value["manifest"], manifest)
    try:
        envelope = base64.b64decode(capsule["envelope_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_final_refit_checkpoint_parse_invalid"
        ) from exc
    if (
        base64.b64encode(envelope).decode("ascii") != capsule["envelope_base64"]
        or len(envelope) != capsule["byte_count"]
        or _sha256(envelope) != capsule["sha256"]
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_final_refit_checkpoint_capsule_mismatch")
    relative_path = _safe_bundle_relative_path(capsule["relative_path"])
    capsule_path = capsule_root / relative_path
    if not capsule_path.exists() and not capsule_path.is_symlink():
        _write_bundle_file_atomic(capsule_root, relative_path, envelope)
    receipt = AlphaMaxCapsuleReceipt.from_path(
        capsule_path,
        row_id=manifest.row_id,
        phase=manifest.phase,
        prefix_id=capsule["prefix_id"],
        manifest_sha256=manifest.sha256,
        relative_path=capsule["relative_path"],
    )
    if receipt.sha256 != capsule["sha256"] or receipt.byte_count != capsule["byte_count"]:
        raise AlphaMaxRuntimeContractError("alpha_max_final_refit_checkpoint_capsule_mismatch")
    _alpha_max_capsule_from_receipt(receipt)
    return receipt


def _alpha_max_complete_domain_matrix(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: Path,
    phase: str,
    nodes: Sequence[Mapping[str, object]],
    admitted_symbols: tuple[str, ...],
    domain: str,
    trial_ledger: AlphaMaxTrialLedger,
    prepared_rows: dict[str, _AlphaMaxPreparedReplayRow],
    scaled_row_factory: object | None = None,
    checkpoint_store: _AlphaMaxCellCheckpointStore | None = None,
) -> _AlphaMaxCompletedMatrix:
    """Execute every physical fold, then add cross-row statistics exactly once."""
    row_by_id = {str(row["row_id"]): row for row in nodes}
    if tuple(sorted(row_by_id)) != _ALPHA_MAX_CURRENT_ROW_IDS:
        raise AlphaMaxRuntimeContractError("alpha_max_current_trial_registry_mismatch")
    scaled_ids = ("full_equal_risk_scaled", "full_shrunk_hrp_scaled")
    non_scaled_ids = tuple(
        row_id for row_id in _ALPHA_MAX_RESOLVABLE_ROWS if row_id not in scaled_ids
    )
    pre_gates: dict[tuple[str, int], AlphaMaxCostCellPreGateEvidence] = {}

    def replay_rows(row_ids: Sequence[str]) -> None:
        for row_id in row_ids:
            try:
                prepared = prepared_rows[row_id]
            except KeyError as exc:
                raise AlphaMaxRuntimeContractError("alpha_max_matrix_row_not_prepared") from exc
            for nominal in ALPHA_MAX_COST_CELL_BPS:
                existing = (
                    None
                    if checkpoint_store is None
                    else checkpoint_store.load(
                        row_id=row_id,
                        nominal_cost_bps=nominal,
                        preflight=preflight,
                        prepared=prepared,
                    )
                )
                if existing is None:
                    existing = _replay_alpha_max_cost_cell_pre_gate(
                        preflight,
                        output_root=output_root,
                        phase=phase,
                        manifest_receipt=prepared.manifest_receipt,
                        admitted_symbols=admitted_symbols,
                        row_id=row_id,
                        domain=domain,
                        nominal_cost_bps=nominal,
                        fold_inputs=prepared.fold_inputs,
                    )
                    if checkpoint_store is not None:
                        existing = checkpoint_store.seal(
                            existing,
                            preflight=preflight,
                            prepared=prepared,
                        )
                pre_gates[(row_id, nominal)] = existing

    replay_rows(non_scaled_ids)
    if scaled_row_factory is not None:
        if not callable(scaled_row_factory):
            raise TypeError("alpha_max_scaled_row_factory_required")
        siblings = {
            "full_equal_risk_scaled": "full_equal_risk_1x",
            "full_shrunk_hrp_scaled": "full_shrunk_hrp_1x",
        }
        for row_id in scaled_ids:
            prepared_rows[row_id] = scaled_row_factory(
                row_by_id[row_id],
                _alpha_max_scaled_gross(pre_gates[(siblings[row_id], 30)]),
            )
    replay_rows(scaled_ids)
    if set(pre_gates) != {
        (row_id, nominal)
        for row_id in _ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
    }:
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_cardinality_mismatch")

    statistical_streams = {
        row_id: stream
        for row_id in _ALPHA_MAX_RESOLVABLE_ROWS
        if (stream := pre_gates[(row_id, 30)].combined_primary_return_stream) is not None
    }
    statistics = (
        build_alpha_max_statistical_evidence(statistical_streams, trial_ledger)
        if statistical_streams
        else None
    )
    cells: dict[tuple[str, int], AlphaMaxCostCellEvidence] = {}
    rows: list[AlphaMaxRowEvidence] = []
    for row_id in _ALPHA_MAX_RESOLVABLE_ROWS:
        row_cells: list[AlphaMaxCostCellEvidence] = []
        for nominal in ALPHA_MAX_COST_CELL_BPS:
            pre_gate = pre_gates[(row_id, nominal)]
            cell = build_alpha_max_cost_cell_evidence(
                pre_gate,
                statistical_evidence=(
                    statistics if nominal == 30 and pre_gate.status != "ruin_detected" else None
                ),
            )
            cells[(row_id, nominal)] = cell
            row_cells.append(cell)
        if any(cell.status == "ruin_detected" for cell in row_cells) and row_cells[-1].status != (
            "ruin_detected"
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_row_nonmonotone_ruin")
        rows.append(
            AlphaMaxRowEvidence(
                row_id=row_id,
                matrix_role="resolvable_candidate",
                status=("ruin_detected" if row_cells[-1].status == "ruin_detected" else "complete"),
                evidence_tier="actual_engine",
                selection_valid=all(cell.selection_valid for cell in row_cells),
                cost_cells=tuple(row_cells),
            )
        )

    status_rows: list[dict[str, object]] = []
    for row_id in _ALPHA_MAX_CURRENT_ROW_IDS:
        for nominal in ALPHA_MAX_COST_CELL_BPS:
            if row_id in _ALPHA_MAX_UNAVAILABLE_ROWS:
                status_rows.append(
                    {
                        "capsule_sha256": None,
                        "engine_constructed": False,
                        "manifest_sha256": None,
                        "nominal_cost_bps": nominal,
                        "row_id": row_id,
                        "row_role": "incumbent_unavailable",
                        "selection_eligible": False,
                        "status": "incumbent_replay_unavailable",
                    }
                )
                continue
            if row_id in _ALPHA_MAX_DIAGNOSTIC_ROWS:
                status_rows.append(
                    {
                        "capsule_sha256": None,
                        "engine_constructed": False,
                        "manifest_sha256": None,
                        "nominal_cost_bps": nominal,
                        "row_id": row_id,
                        "row_role": "track_b_diagnostic",
                        "selection_eligible": False,
                        "status": "diagnostic_report_only",
                    }
                )
                continue
            prepared = prepared_rows[row_id]
            cell = cells[(row_id, nominal)]
            capsule_sha = _sha256(
                _canonical_bytes(
                    [
                        {
                            "prefix_id": receipt.prefix_id,
                            "sha256": receipt.sha256,
                        }
                        for receipt in cell.capsule_receipts
                    ]
                )
            )
            status_rows.append(
                {
                    "capsule_sha256": capsule_sha,
                    "cell_sha256": _sha256(canonical_alpha_max_cost_cell_bytes(cell)),
                    "engine_constructed": True,
                    "manifest_sha256": prepared.manifest_receipt.sha256,
                    "nominal_cost_bps": nominal,
                    "row_id": row_id,
                    "row_role": "resolvable_candidate",
                    "selection_eligible": cell.selection_valid,
                    "status": "resolved_engine_cell_complete",
                }
            )
    physical = sum(
        len(cell.pre_gate_evidence.fold_runs)  # type: ignore[union-attr]
        for cell in cells.values()
    )
    expected_schedule = _alpha_max_physical_fold_schedule(domain)
    observed_schedule = tuple(
        (row_id, nominal, fold.split_or_fold_id)
        for row_id in _ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
        for fold in pre_gates[(row_id, nominal)].fold_runs
    )
    _validate_alpha_max_physical_fold_schedule(observed_schedule, domain=domain)
    if len(status_rows) != 84 or len(cells) != 68 or physical != len(expected_schedule):
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_physical_fold_cardinality_mismatch")
    status_payload = (
        _canonical_bytes(
            {
                "artifact_kind": "alpha_max_matrix_statuses.v1",
                "domain": domain,
                "engine_cell_count": 68,
                "physical_fold_run_count": physical,
                "status_count": 84,
                "statuses": status_rows,
            }
        )
        + b"\n"
    )
    return _AlphaMaxCompletedMatrix(
        domain=domain,
        rows=tuple(rows),
        cells=MappingProxyType(cells),
        status_payload=status_payload,
        physical_fold_run_count=physical,
        prepared_rows=MappingProxyType(dict(prepared_rows)),
        gross_by_row=MappingProxyType(
            {row_id: prepared_rows[row_id].gross for row_id in _ALPHA_MAX_RESOLVABLE_ROWS}
        ),
    )


def build_alpha_max_final_refit_indicator_capsule(
    preflight: AlphaMaxRuntimePreflight,
    *,
    output_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    admitted_symbols: tuple[str, ...],
    phase_inputs: tuple[AlphaMaxIndicatorPhaseInput, ...],
) -> AlphaMaxIndicatorCapsule:
    """Replay the full allowed prefix through embargo under the final-refit manifest."""
    if type(phase_inputs) is not tuple or any(
        type(value) is not AlphaMaxIndicatorPhaseInput for value in phase_inputs
    ):
        raise TypeError("alpha_max_final_refit_phase_inputs_must_be_exact_tuple")
    expected = ("warmup", "train", "purge", "validation", "embargo")
    if tuple(value.phase_id for value in phase_inputs) != expected:
        raise AlphaMaxRuntimeContractError("alpha_max_final_refit_prefix_incomplete")
    capsule: AlphaMaxIndicatorCapsule | None = None
    for phase_input in phase_inputs:
        capsule = build_alpha_max_indicator_capsule(
            preflight,
            output_root=output_root,
            phase="prelock_final_refit",
            manifest_path=manifest_path,
            admitted_symbols=admitted_symbols,
            phase_id=phase_input.phase_id,
            raw_root=phase_input.raw_root,
            ordered_lookup=phase_input.ordered_lookup,
            watermark=phase_input.watermark,
            data_dict=phase_input.data_dict,
            prior_indicator_capsule=capsule,
        )
    if type(capsule) is not AlphaMaxIndicatorCapsule or capsule.phase_id != "embargo":
        raise AlphaMaxRuntimeContractError("alpha_max_final_refit_prefix_incomplete")
    return capsule


def _safe_bundle_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_relative_path_invalid")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_relative_path_invalid")
    normalized = path.as_posix()
    if normalized in {"SEALED.json", "DURATION.json", "RSS.json"}:
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_reserved_path")
    return normalized


def _fsync_directory(path: Path) -> None:
    if _is_proc_fd_parent(path):
        os.fsync(int(path.name))
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _prepare_new_output_root(output_root: str | os.PathLike[str]) -> Path:
    raw = _require_exact_explicit_path(output_root)
    root = Path(raw)
    parent = root.parent
    try:
        if str(parent.resolve(strict=True)) != str(parent):
            raise AlphaMaxRuntimeContractError("alpha_max_output_parent_identity_invalid")
        parent_status = parent.lstat()
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_output_parent_identity_invalid") from exc
    if not stat.S_ISDIR(parent_status.st_mode) or stat.S_ISLNK(parent_status.st_mode):
        raise AlphaMaxRuntimeContractError("alpha_max_output_parent_identity_invalid")
    try:
        os.mkdir(root, mode=0o700)
    except FileExistsError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_exists") from exc
    _fsync_directory(parent)
    return root


def _alpha_max_raw_entry_symbol(
    relative_path: str,
    *,
    exchange: str,
) -> str:
    parts = Path(relative_path).parts
    if len(parts) == 4 and parts[:2] == ("market_ohlcv_1s", exchange):
        return parts[2]
    if len(parts) == 3 and parts[0] == exchange:
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    raise AlphaMaxRuntimeContractError("alpha_max_raw_root_partition_layout_invalid")


def _alpha_max_feature_entry_symbol(relative_path: str, *, exchange: str) -> str:
    parts = Path(relative_path).parts
    offset = 0
    if len(parts) >= 2 and parts[:2] == ("feature_points", f"exchange={exchange}"):
        offset = 2
    elif parts and parts[0] == f"exchange={exchange}":
        offset = 1
    scoped = parts[offset:]
    if len(scoped) != 3 or not scoped[0].startswith("symbol="):
        raise AlphaMaxRuntimeContractError("alpha_max_feature_root_partition_layout_invalid")
    symbol = scoped[0].removeprefix("symbol=")
    if symbol not in ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise AlphaMaxRuntimeContractError("alpha_max_feature_root_partition_layout_invalid")
    return symbol


def _validate_alpha_max_adjacent_feature_roots(
    seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
    adjacent_pairs: Sequence[tuple[str, str]],
) -> None:
    """Close the per-root edge gap by checking every adjacent funding pair."""
    maximum_gap_ms = 8 * 60 * 60 * 1000 + 1000
    for predecessor_id, current_id in adjacent_pairs:
        try:
            predecessor = seals[(predecessor_id, "feature")]
            current = seals[(current_id, "feature")]
        except KeyError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_adjacent_feature_root_missing") from exc
        if predecessor.end_utc != current.start_utc:
            raise AlphaMaxRuntimeContractError("alpha_max_adjacent_feature_root_bounds_invalid")
        predecessor_last: dict[str, int] = {}
        current_first: dict[str, int] = {}
        for entry in predecessor.entries:
            symbol = _alpha_max_feature_entry_symbol(
                entry.relative_path,
                exchange=predecessor.exchange,
            )
            predecessor_last[symbol] = max(
                predecessor_last.get(symbol, entry.maximum_timestamp_ms),
                entry.maximum_timestamp_ms,
            )
        for entry in current.entries:
            symbol = _alpha_max_feature_entry_symbol(
                entry.relative_path,
                exchange=current.exchange,
            )
            current_first[symbol] = min(
                current_first.get(symbol, entry.minimum_timestamp_ms),
                entry.minimum_timestamp_ms,
            )
        predecessor_availability_start = getattr(
            predecessor,
            "availability_start_by_symbol",
            None,
        )
        current_availability_start = getattr(current, "availability_start_by_symbol", None)
        predecessor_availability_end = getattr(
            predecessor,
            "availability_end_by_symbol",
            None,
        )
        current_availability_end = getattr(current, "availability_end_by_symbol", None)
        if (
            predecessor_availability_start is None
            or current_availability_start is None
            or predecessor_availability_end is None
            or current_availability_end is None
            or predecessor_availability_start != current_availability_start
            or predecessor_availability_end != current_availability_end
            or tuple(predecessor_availability_start) != ALPHA_MAX_CANDIDATE_SYMBOLS
            or tuple(predecessor_availability_end) != ALPHA_MAX_CANDIDATE_SYMBOLS
        ):
            complete = False
        else:
            complete = True
            predecessor_start_ms = int(predecessor.start_utc.timestamp() * 1000)
            predecessor_end_ms = int(predecessor.end_utc.timestamp() * 1000)
            current_start_ms = int(current.start_utc.timestamp() * 1000)
            current_end_ms = int(current.end_utc.timestamp() * 1000)
            for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
                availability_start = predecessor_availability_start[symbol]
                availability_end = predecessor_availability_end[symbol]
                if (
                    not isinstance(availability_start, datetime)
                    or availability_start.tzinfo != UTC
                    or not isinstance(availability_end, datetime)
                    or availability_end.tzinfo != UTC
                    or availability_end <= availability_start
                ):
                    complete = False
                    break
                availability_start_ms = int(availability_start.timestamp() * 1000)
                availability_end_ms = int(availability_end.timestamp() * 1000)
                predecessor_owned_start_ms = max(
                    predecessor_start_ms,
                    availability_start_ms,
                )
                predecessor_owned_end_ms = min(
                    predecessor_end_ms,
                    availability_end_ms,
                )
                current_owned_start_ms = max(current_start_ms, availability_start_ms)
                current_owned_end_ms = min(current_end_ms, availability_end_ms)
                predecessor_active = predecessor_owned_start_ms < predecessor_owned_end_ms
                current_active = current_owned_start_ms < current_owned_end_ms

                if not predecessor_active and not current_active:
                    if symbol in predecessor_last or symbol in current_first:
                        complete = False
                        break
                    continue
                if not predecessor_active:
                    if (
                        symbol in predecessor_last
                        or symbol not in current_first
                        or not 0 <= current_first[symbol] - current_owned_start_ms <= maximum_gap_ms
                    ):
                        complete = False
                        break
                    continue
                if not current_active:
                    if (
                        symbol not in predecessor_last
                        or symbol in current_first
                        or not 0
                        < predecessor_owned_end_ms - predecessor_last[symbol]
                        <= maximum_gap_ms
                    ):
                        complete = False
                        break
                    continue
                if (
                    symbol not in predecessor_last
                    or symbol not in current_first
                    or not 0 < current_first[symbol] - predecessor_last[symbol] <= maximum_gap_ms
                ):
                    complete = False
                    break
        if not complete:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_adjacent_feature_root_funding_coverage_incomplete"
            )


def _alpha_max_raw_directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(stat.S_IFMT(value.st_mode)))


def _alpha_max_raw_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


class _AlphaMaxSealedRawReader:
    """Read exact sealed raw partitions through one retained root capability."""

    __slots__ = ("_entry_ids", "_root_fd", "_root_identity", "_seal")

    def __init__(self, seal: AlphaMaxRootSeal) -> None:
        if type(seal) is not AlphaMaxRootSeal or seal.root_kind != "raw":
            raise TypeError("alpha_max_sealed_raw_root_required")
        root = Path(seal.path)
        try:
            observed = root.lstat()
        except OSError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_root_invalid") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_root_invalid")
        root_fd: int | None = None
        try:
            root_fd = _alpha_max_open_directory_at(root)
            opened = os.fstat(root_fd)
        except OSError as exc:
            if root_fd is not None:
                os.close(root_fd)
            raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_root_invalid") from exc
        root_identity = _alpha_max_raw_directory_identity(opened)
        if root_identity != _alpha_max_raw_directory_identity(observed):
            os.close(root_fd)
            raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_root_changed_during_open")
        self._seal = seal
        self._entry_ids = frozenset(id(entry) for entry in seal.entries)
        self._root_fd: int | None = root_fd
        self._root_identity = root_identity

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> _AlphaMaxSealedRawReader:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        descriptor = getattr(self, "_root_fd", None)
        if descriptor is None:
            return
        self._root_fd = None
        try:
            os.close(descriptor)
        except OSError:
            pass

    def read_entry(self, entry: AlphaMaxTreeEntry) -> Any:
        if type(entry) is not AlphaMaxTreeEntry or id(entry) not in self._entry_ids:
            raise TypeError("alpha_max_sealed_raw_entry_not_owned")
        root_fd = self._root_fd
        if root_fd is None:
            raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_reader_closed")
        try:
            if _alpha_max_raw_directory_identity(os.fstat(root_fd)) != self._root_identity:
                raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_root_identity_changed")
        except OSError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_sealed_raw_root_identity_changed"
            ) from exc

        relative = Path(entry.relative_path)
        opened_directories: list[int] = []
        descriptor: int | None = None
        try:
            parent_fd = root_fd
            for part in relative.parts[:-1]:
                observed_directory = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if stat.S_ISLNK(observed_directory.st_mode) or not stat.S_ISDIR(
                    observed_directory.st_mode
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_identity_invalid")
                child_fd = _alpha_max_open_directory_at(part, dir_fd=parent_fd)
                try:
                    opened_directory = os.fstat(child_fd)
                except OSError:
                    os.close(child_fd)
                    raise
                if _alpha_max_raw_directory_identity(
                    opened_directory
                ) != _alpha_max_raw_directory_identity(observed_directory):
                    os.close(child_fd)
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_sealed_raw_path_changed_during_open"
                    )
                opened_directories.append(child_fd)
                parent_fd = child_fd

            observed_file = os.stat(
                relative.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                stat.S_ISLNK(observed_file.st_mode)
                or not stat.S_ISREG(observed_file.st_mode)
                or int(observed_file.st_nlink) != 1
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_identity_invalid")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(relative.name, flags, dir_fd=parent_fd)
            opened_file = os.fstat(descriptor)
            if (
                _alpha_max_raw_file_identity(opened_file)
                != _alpha_max_raw_file_identity(observed_file)
                or int(opened_file.st_size) != entry.byte_count
                or stat.S_IMODE(opened_file.st_mode) != entry.mode
                or int(opened_file.st_mtime_ns) != entry.mtime_ns
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_identity_invalid")

            digest = hashlib.sha256()
            payload = bytearray()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                payload.extend(chunk)
                digest.update(chunk)
                if len(payload) > entry.byte_count:
                    raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_size_mismatch")
            after = os.fstat(descriptor)
            if _alpha_max_raw_file_identity(after) != _alpha_max_raw_file_identity(opened_file):
                raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_changed_during_read")
            if len(payload) != entry.byte_count or digest.hexdigest() != entry.sha256:
                raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_hash_mismatch")
        except OSError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_read_failed") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            for directory_fd in reversed(opened_directories):
                os.close(directory_fd)

        try:
            import polars as pl

            return pl.read_parquet(io.BytesIO(payload))
        except Exception as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_sealed_raw_parquet_invalid") from exc


def _alpha_max_load_raw_admission_summary(
    seal: AlphaMaxRootSeal,
    *,
    symbol: str,
    include_quote_notional: bool,
) -> tuple[
    tuple[AlphaMaxDailyQuoteNotional, ...],
    frozenset[tuple[datetime, int]],
    bool,
]:
    """Stream one symbol/month and collapse each train day exactly once."""
    if type(seal) is not AlphaMaxRootSeal or seal.root_kind != "raw":
        raise TypeError("alpha_max_raw_root_seal_required")
    if symbol not in ALPHA_MAX_CANDIDATE_SYMBOLS:
        raise AlphaMaxRuntimeContractError("alpha_max_admission_symbol_invalid")

    import polars as pl

    entries = tuple(
        entry
        for entry in seal.entries
        if _alpha_max_raw_entry_symbol(entry.relative_path, exchange=seal.exchange) == symbol
    )
    if not entries:
        return (), frozenset(), False

    daily_summaries: list[AlphaMaxDailyQuoteNotional] = []
    completed_keys: set[tuple[datetime, int]] = set()
    previous_timestamp_ms: int | None = None
    current_day = None
    current_timestamps: list[datetime] = []
    current_closes: list[float] = []
    current_volumes: list[float] = []

    def flush_day() -> None:
        if not current_timestamps:
            return
        daily_summaries.append(
            build_alpha_max_daily_quote_notional(
                tuple(current_timestamps),
                tuple(current_closes),
                tuple(current_volumes),
            )
        )
        current_timestamps.clear()
        current_closes.clear()
        current_volumes.clear()

    try:
        reader = _AlphaMaxSealedRawReader(seal)
    except AlphaMaxRuntimeContractError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_admission_raw_partition_read_failed") from exc
    try:
        for entry in sorted(entries, key=lambda value: value.minimum_timestamp_ms):
            try:
                frame = (
                    reader.read_entry(entry)
                    .lazy()
                    .select(
                        [
                            pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"),
                            pl.col("close").cast(pl.Float64),
                            pl.col("volume").cast(pl.Float64),
                        ]
                    )
                    .collect(engine="streaming")
                    .sort("timestamp_ms")
                )
            except Exception as exc:
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_admission_raw_partition_read_failed"
                ) from exc
            if frame.is_empty():
                raise AlphaMaxRuntimeContractError("alpha_max_admission_raw_partition_empty")
            if frame.null_count().row(0) != (0, 0, 0):
                raise AlphaMaxRuntimeContractError("alpha_max_admission_raw_partition_null")
            timestamps = frame.get_column("timestamp_ms")
            if timestamps.n_unique() != frame.height:
                raise AlphaMaxRuntimeContractError("alpha_max_admission_raw_timestamp_duplicate")
            first_timestamp_ms = int(timestamps[0])
            if previous_timestamp_ms is not None and first_timestamp_ms <= previous_timestamp_ms:
                raise AlphaMaxRuntimeContractError("alpha_max_admission_raw_timestamp_order")

            for timestamp_ms, close, volume in frame.iter_rows(named=False):
                parsed_timestamp_ms = int(timestamp_ms)
                parsed_close = float(close)
                parsed_volume = float(volume)
                if (
                    not math.isfinite(parsed_close)
                    or parsed_close <= 0.0
                    or not math.isfinite(parsed_volume)
                    or parsed_volume < 0.0
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_admission_raw_value_invalid")
                timestamp = datetime.fromtimestamp(parsed_timestamp_ms / 1000.0, tz=UTC)
                bucket_hour = (timestamp.hour // 4) * 4
                day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                key = (day_start, bucket_hour)
                completed_keys.add(key)
                if include_quote_notional:
                    if current_day is not None and timestamp.date() != current_day:
                        flush_day()
                    current_day = timestamp.date()
                    current_timestamps.append(timestamp)
                    current_closes.append(parsed_close)
                    current_volumes.append(parsed_volume)
            previous_timestamp_ms = int(timestamps[-1])
    finally:
        reader.close()

    if include_quote_notional:
        flush_day()
    return tuple(daily_summaries), frozenset(completed_keys), True


def _alpha_max_feature_spec(seal: AlphaMaxRootSeal) -> FeatureRootSpec:
    if type(seal) is not AlphaMaxRootSeal or seal.root_kind != "feature":
        raise TypeError("alpha_max_feature_root_seal_required")
    return FeatureRootSpec(
        root_id=seal.root_id,
        path=seal.path,
        exchange=seal.exchange,
        start_utc=seal.start_utc,
        end_utc=seal.end_utc,
        inventory_sha256=seal.inventory_sha256,
        content_sha256=seal.content_sha256,
    )


def _compute_alpha_max_admission_from_seals(
    *,
    warmup_raw: AlphaMaxRootSeal,
    warmup_feature: AlphaMaxRootSeal,
    train_raw: AlphaMaxRootSeal,
    train_feature: AlphaMaxRootSeal,
) -> AlphaMaxAdmissionComputation:
    """Derive the train-only admission artifact from bounded sealed partitions."""
    ordered_lookup = AlphaMaxOrderedFundingLookup(
        (_alpha_max_feature_spec(warmup_feature), _alpha_max_feature_spec(train_feature)),
        root_seals=(warmup_feature, train_feature),
    )
    expected_hours = frozenset({0, 4, 8, 12, 16, 20})
    train_start = train_raw.start_utc
    train_end = train_raw.end_utc
    inputs: dict[str, AlphaMaxAdmissionDailyCandidateInput] = {}
    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        _warmup_observations, warmup_keys, warmup_integrity = _alpha_max_load_raw_admission_summary(
            warmup_raw,
            symbol=symbol,
            include_quote_notional=False,
        )
        train_daily, train_keys, train_integrity = _alpha_max_load_raw_admission_summary(
            train_raw,
            symbol=symbol,
            include_quote_notional=True,
        )
        warmup_days = {
            day_start
            for day_start, _hour in warmup_keys
            if {hour for candidate_day, hour in warmup_keys if candidate_day == day_start}
            == expected_hours
        }
        train_days = {
            day_start
            for day_start, _hour in train_keys
            if {hour for candidate_day, hour in train_keys if candidate_day == day_start}
            == expected_hours
        }
        funding_complete = True
        boundary = train_start + timedelta(hours=4)
        while boundary <= train_end:
            point = ordered_lookup.get_latest_point(
                symbol,
                "funding_rate",
                timestamp_ms=int(boundary.timestamp() * 1000),
            )
            if point is None:
                funding_complete = False
                break
            boundary += timedelta(hours=4)
        inputs[symbol] = AlphaMaxAdmissionDailyCandidateInput(
            symbol=symbol,
            daily_quote_notional=train_daily,
            consecutive_completed_daily_bars_before_train=(
                len(warmup_days) if warmup_integrity else 0
            ),
            causal_funding_coverage_complete=funding_complete,
            unresolved_daily_cross_section_count=max(0, 517 - len(train_days)),
            partition_integrity_complete=warmup_integrity and train_integrity,
        )

    return compute_alpha_max_train_admission_from_daily_summaries(
        inputs,
        input_root_hashes={
            "warmup": _sha256(
                _canonical_bytes(
                    {
                        "feature": warmup_feature.sha256,
                        "raw": warmup_raw.sha256,
                    }
                )
            ),
            "train": _sha256(
                _canonical_bytes(
                    {
                        "feature": train_feature.sha256,
                        "raw": train_raw.sha256,
                    }
                )
            ),
        },
    )


class _AlphaMaxBoundedRawLoader:
    """One-day sealed Parquet repository reader for one logical engine."""

    __slots__ = (
        "_admitted_symbols",
        "_entries",
        "_frame_cache",
        "_reader",
        "_seal",
    )

    def __init__(
        self,
        seal: AlphaMaxRootSeal,
        admitted_symbols: tuple[str, ...],
    ) -> None:
        if type(seal) is not AlphaMaxRootSeal or seal.root_kind != "raw":
            raise TypeError("alpha_max_bounded_raw_seal_required")
        if seal.symbols != ALPHA_MAX_CANDIDATE_SYMBOLS:
            raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_inventory_mismatch")
        if (
            type(admitted_symbols) is not tuple
            or not admitted_symbols
            or tuple(symbol for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS if symbol in admitted_symbols)
            != admitted_symbols
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_universe_mismatch")
        sealed_root = Path(seal.path)
        layouts = {len(Path(entry.relative_path).parts) for entry in seal.entries}
        if layouts == {4}:
            if any(
                Path(entry.relative_path).parts[:2] != ("market_ohlcv_1s", seal.exchange)
                for entry in seal.entries
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_raw_repository_root_unprovable")
            repository_root = sealed_root
        elif layouts == {3}:
            if sealed_root.name != "market_ohlcv_1s":
                raise AlphaMaxRuntimeContractError("alpha_max_raw_repository_root_unprovable")
            repository_root = sealed_root.parent
        elif layouts == {2}:
            if sealed_root.name != seal.exchange or sealed_root.parent.name != "market_ohlcv_1s":
                raise AlphaMaxRuntimeContractError("alpha_max_raw_repository_root_unprovable")
            repository_root = sealed_root.parent.parent
        else:
            raise AlphaMaxRuntimeContractError("alpha_max_raw_repository_layout_mixed")
        entries: dict[tuple[str, str], object] = {}
        for entry in seal.entries:
            symbol = _alpha_max_raw_entry_symbol(
                entry.relative_path,
                exchange=seal.exchange,
            )
            expected = (
                repository_root
                / "market_ohlcv_1s"
                / seal.exchange
                / symbol
                / Path(entry.relative_path).name
            )
            observed = sealed_root / entry.relative_path
            key = (symbol, Path(entry.relative_path).name)
            if expected != observed or key in entries:
                raise AlphaMaxRuntimeContractError("alpha_max_raw_repository_root_unprovable")
            entries[key] = entry
        self._seal = seal
        self._admitted_symbols = admitted_symbols
        self._entries = MappingProxyType(entries)
        self._frame_cache: dict[str, tuple[str, object]] = {}
        self._reader = _AlphaMaxSealedRawReader(seal)

    def __del__(self) -> None:
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.close()

    @property
    def seal(self) -> AlphaMaxRootSeal:
        return self._seal

    def _read_entry_cached(self, symbol: str, entry: AlphaMaxTreeEntry) -> object:
        relative_path = entry.relative_path
        cached = self._frame_cache.get(symbol)
        if cached is not None and cached[0] == relative_path:
            return cached[1]
        try:
            frame = self._reader.read_entry(entry)
        except AlphaMaxRuntimeContractError as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_read_failed") from exc
        self._frame_cache[symbol] = (relative_path, frame)
        return frame

    def load_day(self, start: datetime, end: datetime) -> dict[str, object]:
        if (
            type(start) is not datetime
            or type(end) is not datetime
            or start.tzinfo != UTC
            or end.tzinfo != UTC
            or end - start != timedelta(days=1)
            or not self._seal.start_utc <= start < end <= self._seal.end_utc
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_day_invalid")
        loaded: dict[str, object] = {}
        required_columns = ("open", "high", "low", "close", "volume")
        import polars as pl

        for symbol in self._admitted_symbols:
            entry = self._entries.get((symbol, f"{start:%Y-%m}.parquet"))
            if entry is None:
                raise AlphaMaxRuntimeContractError(
                    f"alpha_max_bounded_raw_day_empty:{symbol}:{start.date().isoformat()}"
                )
            frame = self._read_entry_cached(symbol, entry)
            if "datetime" not in frame.columns or any(
                column not in frame.columns for column in required_columns
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_ohlcv_schema_invalid")
            frame = (
                frame.with_columns(pl.col("datetime").dt.epoch("ms").alias("__timestamp_ms"))
                .filter(
                    (pl.col("__timestamp_ms") >= int(start.timestamp() * 1000))
                    & (pl.col("__timestamp_ms") < int(end.timestamp() * 1000))
                )
                .sort("__timestamp_ms")
                .select(("datetime", *required_columns))
            )
            if frame.is_empty():
                raise AlphaMaxRuntimeContractError(
                    f"alpha_max_bounded_raw_day_empty:{symbol}:{start.date().isoformat()}"
                )
            timestamps_ms = frame.get_column("datetime").dt.epoch("ms")
            if (
                int(timestamps_ms[0]) < int(start.timestamp() * 1000)
                or int(timestamps_ms[-1]) >= int(end.timestamp() * 1000)
                or timestamps_ms.n_unique() != frame.height
                or not bool((timestamps_ms.diff().drop_nulls() > 0).all())
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_day_scope_invalid")
            if any(column not in frame.columns for column in required_columns):
                raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_ohlcv_schema_invalid")
            for column in required_columns:
                series = frame.get_column(column)
                if series.null_count() or not bool(series.is_finite().all()):
                    raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_ohlcv_nonfinite")
            if any(
                not bool((frame.get_column(column) > 0.0).all())
                for column in ("open", "high", "low", "close")
            ) or not bool((frame.get_column("volume") >= 0.0).all()):
                raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_ohlcv_value_invalid")
            bucket_ids = (timestamps_ms - int(start.timestamp() * 1000)) // 14_400_000
            if tuple(sorted({int(value) for value in bucket_ids.to_list()})) != tuple(range(6)):
                raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_native_bucket_missing")
            loaded[symbol] = frame
        return loaded

    def fold_exact_indicator_phase(
        self,
        aggregator: TimeframeAggregator,
        *,
        start: datetime,
        end: datetime,
    ) -> tuple[tuple[NativeBarRelease, ...], int]:
        """Authenticate each partition once and exact-fold the complete phase."""
        if type(aggregator) is not TimeframeAggregator:
            raise TypeError("alpha_max_indicator_aggregator_identity_invalid")
        if (
            type(start) is not datetime
            or type(end) is not datetime
            or start.tzinfo != UTC
            or end.tzinfo != UTC
            or not self._seal.start_utc <= start < end <= self._seal.end_utc
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_phase_bounds_invalid")
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        if start_ms % 1000 or end_ms % 1000 or end_ms <= start_ms:
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_phase_bounds_invalid")
        expected_windows = (end_ms - start_ms) // 1000
        releases: list[NativeBarRelease] = []
        required_columns = ("open", "high", "low", "close", "volume")
        import polars as pl

        try:
            from lumina_quant import _compute
            from lumina_quant._native_kernel_version import compute_src_hash

            runtime_binding = _alpha_max_indicator_runtime_binding()
            fold_native = getattr(_compute, "fold_alpha_max_native_bars", None)
            build_info = getattr(_compute, "build_info", None)
            kernel_src_hash = getattr(_compute, "kernel_src_hash", None)
            cargo_toml = (
                Path(__file__).resolve().parents[3] / "native" / "lumina_compute" / "Cargo.toml"
            )
            expected_version = next(
                (
                    line.split('"')[1]
                    for line in cargo_toml.read_text(encoding="utf-8").splitlines()
                    if line.strip().startswith("version = ")
                ),
                None,
            )
            expected_hash = compute_src_hash()
            if (
                type(_compute) is not types.ModuleType
                or not isinstance(getattr(_compute, "__file__", None), str)
                or type(fold_native) is not types.BuiltinFunctionType
                or type(build_info) is not types.BuiltinFunctionType
                or type(kernel_src_hash) is not types.BuiltinFunctionType
                or fold_native.__module__ != _compute.__name__
                or build_info.__module__ != _compute.__name__
                or kernel_src_hash.__module__ != _compute.__name__
                or type(expected_version) is not str
                or not expected_version
                or type(expected_hash) is not str
                or not expected_hash
                or build_info() != expected_version
                or kernel_src_hash() != expected_hash
                or runtime_binding["extension_version"] != expected_version
                or runtime_binding["extension_source_hash"] != expected_hash
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_identity_invalid")
        except AlphaMaxRuntimeContractError:
            raise
        except Exception as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_native_identity_invalid"
            ) from exc
        timeframes = tuple(timeframe for timeframe in aggregator._timeframes if timeframe != "1s")
        timeframe_values = tuple(
            int(timeframe_to_milliseconds(timeframe)) for timeframe in timeframes
        )
        if (
            not timeframes
            or len(timeframes) != len(set(timeframes))
            or any(value not in (14_400_000, 86_400_000) for value in timeframe_values)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_indicator_native_timeframe_invalid")
        native_timeframes = np.ascontiguousarray(timeframe_values, dtype=np.int64)

        for symbol in self._admitted_symbols:
            if (
                self._seal.availability_start_by_symbol[symbol] > start
                or self._seal.availability_end_by_symbol[symbol] < end
            ):
                raise AlphaMaxRuntimeContractError(
                    f"alpha_max_indicator_symbol_coverage_invalid:{symbol}"
                )
            entries = sorted(
                (
                    entry
                    for (entry_symbol, _name), entry in self._entries.items()
                    if entry_symbol == symbol
                    and self._partition_intersects(
                        entry.relative_path,
                        start=start,
                        end=end,
                    )
                ),
                key=lambda value: value.relative_path,
            )
            if not entries:
                raise AlphaMaxRuntimeContractError(
                    f"alpha_max_bounded_raw_day_empty:{symbol}:{start.date().isoformat()}"
                )
            previous_timestamp_ms: int | None = None
            observed_rows = 0
            for entry in entries:
                if type(entry) is not AlphaMaxTreeEntry:
                    raise TypeError("alpha_max_sealed_raw_entry_not_owned")
                frame = self._read_entry_cached(symbol, entry)
                if "datetime" not in frame.columns or any(
                    column not in frame.columns for column in required_columns
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_ohlcv_schema_invalid")
                frame = (
                    frame.with_columns(pl.col("datetime").dt.epoch("ms").alias("__timestamp_ms"))
                    .filter(
                        (pl.col("__timestamp_ms") >= start_ms) & (pl.col("__timestamp_ms") < end_ms)
                    )
                    .select(("__timestamp_ms", *required_columns))
                )
                if frame.is_empty():
                    continue
                timestamps = frame.get_column("__timestamp_ms")
                if (
                    (
                        previous_timestamp_ms is not None
                        and int(timestamps[0]) <= previous_timestamp_ms
                    )
                    or timestamps.n_unique() != frame.height
                    or not bool((timestamps.diff().drop_nulls() > 0).all())
                ):
                    raise AlphaMaxRuntimeContractError(
                        f"alpha_max_indicator_symbol_timeline_invalid:{symbol}"
                    )
                for column in required_columns:
                    values = frame.get_column(column)
                    if values.null_count() or not bool(values.is_finite().all()):
                        raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_ohlcv_nonfinite")
                if (
                    any(
                        not bool((frame.get_column(column) > 0.0).all())
                        for column in ("open", "high", "low", "close")
                    )
                    or not bool((frame.get_column("volume") >= 0.0).all())
                    or not all(
                        frame.select(
                            (pl.col("high") >= pl.max_horizontal("open", "close"))
                            .all()
                            .alias("high_valid"),
                            (pl.col("low") <= pl.min_horizontal("open", "close"))
                            .all()
                            .alias("low_valid"),
                        ).row(0)
                    )
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_bounded_raw_ohlcv_value_invalid")
                try:
                    previous_timestamp_ms = aggregator._last_seen_ms.get(symbol, -1)
                    if type(previous_timestamp_ms) is not int:
                        raise ValueError("last_seen_invalid")
                    active: list[int] = []
                    buckets: list[int] = []
                    work_open: list[float] = []
                    work_high: list[float] = []
                    work_low: list[float] = []
                    work_close: list[float] = []
                    work_volume: list[float] = []
                    for timeframe in timeframes:
                        working = aggregator._working[symbol].get(timeframe)
                        if working is None:
                            active.append(0)
                            buckets.append(0)
                            work_open.append(0.0)
                            work_high.append(0.0)
                            work_low.append(0.0)
                            work_close.append(0.0)
                            work_volume.append(0.0)
                            continue
                        if set(working) != {
                            "bucket_ms",
                            "time",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        }:
                            raise ValueError("working_state_invalid")
                        active.append(1)
                        buckets.append(int(working["bucket_ms"]))
                        work_open.append(float(working["open"]))
                        work_high.append(float(working["high"]))
                        work_low.append(float(working["low"]))
                        work_close.append(float(working["close"]))
                        work_volume.append(float(working["volume"]))
                    native_result = fold_native(
                        np.ascontiguousarray(timestamps.to_numpy(), dtype=np.int64),
                        np.ascontiguousarray(frame.get_column("open").to_numpy(), dtype=np.float64),
                        np.ascontiguousarray(frame.get_column("high").to_numpy(), dtype=np.float64),
                        np.ascontiguousarray(frame.get_column("low").to_numpy(), dtype=np.float64),
                        np.ascontiguousarray(
                            frame.get_column("close").to_numpy(), dtype=np.float64
                        ),
                        np.ascontiguousarray(
                            frame.get_column("volume").to_numpy(), dtype=np.float64
                        ),
                        native_timeframes,
                        previous_timestamp_ms,
                        np.ascontiguousarray(active, dtype=np.uint8),
                        np.ascontiguousarray(buckets, dtype=np.int64),
                        np.ascontiguousarray(work_open, dtype=np.float64),
                        np.ascontiguousarray(work_high, dtype=np.float64),
                        np.ascontiguousarray(work_low, dtype=np.float64),
                        np.ascontiguousarray(work_close, dtype=np.float64),
                        np.ascontiguousarray(work_volume, dtype=np.float64),
                    )
                    if type(native_result) is not tuple or len(native_result) != 2:
                        raise ValueError("native_result_invalid")
                    native_releases, native_state = native_result
                    if (
                        type(native_releases) is not tuple
                        or len(native_releases) != 8
                        or type(native_state) is not tuple
                        or len(native_state) != 8
                        or type(native_state[0]) is not int
                    ):
                        raise ValueError("native_result_invalid")
                    release_arrays = native_releases
                    state_arrays = native_state[1:]
                    expected_dtypes = (
                        np.dtype(np.int64),
                        np.dtype(np.int64),
                        np.dtype(np.int64),
                        np.dtype(np.float64),
                        np.dtype(np.float64),
                        np.dtype(np.float64),
                        np.dtype(np.float64),
                        np.dtype(np.float64),
                    )
                    if any(
                        type(value) is not np.ndarray
                        or value.ndim != 1
                        or not value.flags.c_contiguous
                        or value.dtype != dtype
                        for value, dtype in zip(release_arrays, expected_dtypes, strict=True)
                    ):
                        raise ValueError("native_release_arrays_invalid")
                    release_count = len(release_arrays[0])
                    if any(len(value) != release_count for value in release_arrays):
                        raise ValueError("native_release_lengths_invalid")
                    state_dtypes = (
                        np.dtype(np.uint8),
                        np.dtype(np.int64),
                        np.dtype(np.float64),
                        np.dtype(np.float64),
                        np.dtype(np.float64),
                        np.dtype(np.float64),
                        np.dtype(np.float64),
                    )
                    if any(
                        type(value) is not np.ndarray
                        or value.ndim != 1
                        or not value.flags.c_contiguous
                        or value.dtype != dtype
                        or len(value) != len(timeframes)
                        for value, dtype in zip(state_arrays, state_dtypes, strict=True)
                    ):
                        raise ValueError("native_state_arrays_invalid")
                    if native_state[0] != int(timestamps[-1]):
                        raise ValueError("native_cursor_invalid")
                    for row in zip(*release_arrays, strict=True):
                        (
                            release_timestamp,
                            timeframe_index,
                            bucket_ms,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            volume,
                        ) = row
                        if (
                            not 0 <= int(timeframe_index) < len(timeframes)
                            or int(release_timestamp) % 1000
                            or int(bucket_ms) % timeframe_values[int(timeframe_index)]
                            or not all(math.isfinite(float(value)) for value in row[3:])
                            or float(open_price) <= 0.0
                            or float(low_price) <= 0.0
                            or float(high_price) < max(float(open_price), float(close_price))
                            or float(low_price) > min(float(open_price), float(close_price))
                            or float(volume) < 0.0
                        ):
                            raise ValueError("native_release_invalid")
                        timeframe = timeframes[int(timeframe_index)]
                        bar = (
                            aggregator._bucket_time(int(bucket_ms)),
                            float(open_price),
                            float(high_price),
                            float(low_price),
                            float(close_price),
                            float(volume),
                        )
                        aggregator._ensure_history(symbol, timeframe).append(bar)
                        releases.append(
                            NativeBarRelease(
                                release_timestamp_ms=int(release_timestamp),
                                symbol=symbol,
                                timeframe=timeframe,
                                bar=bar,
                            )
                        )
                    history_1s = aggregator._ensure_history(symbol, "1s")
                    retained_1s_count = min(frame.height, history_1s.maxlen or frame.height)
                    if frame.height >= (history_1s.maxlen or frame.height):
                        history_1s.clear()
                    for row in frame.tail(retained_1s_count).iter_rows():
                        history_1s.append(
                            (
                                int(row[0]),
                                float(row[1]),
                                float(row[2]),
                                float(row[3]),
                                float(row[4]),
                                float(row[5]),
                            )
                        )
                    aggregator._last_seen_ms[symbol] = native_state[0]
                    for index, timeframe in enumerate(timeframes):
                        (
                            next_active,
                            next_bucket,
                            next_open,
                            next_high,
                            next_low,
                            next_close,
                            next_volume,
                        ) = (value[index] for value in state_arrays)
                        if int(next_active) == 0:
                            if any(
                                float(value) != 0.0
                                for value in (
                                    next_bucket,
                                    next_open,
                                    next_high,
                                    next_low,
                                    next_close,
                                    next_volume,
                                )
                            ):
                                raise ValueError("native_inactive_state_invalid")
                            aggregator._working[symbol].pop(timeframe, None)
                        elif (
                            int(next_active) == 1
                            and all(
                                math.isfinite(float(value))
                                for value in (
                                    next_open,
                                    next_high,
                                    next_low,
                                    next_close,
                                    next_volume,
                                )
                            )
                            and int(next_bucket) % timeframe_values[index] == 0
                            and float(next_open) > 0.0
                            and float(next_low) > 0.0
                            and float(next_high) >= max(float(next_open), float(next_close))
                            and float(next_low) <= min(float(next_open), float(next_close))
                            and float(next_volume) >= 0.0
                        ):
                            aggregator._working[symbol][timeframe] = {
                                "bucket_ms": int(next_bucket),
                                "time": aggregator._bucket_time(int(next_bucket)),
                                "open": float(next_open),
                                "high": float(next_high),
                                "low": float(next_low),
                                "close": float(next_close),
                                "volume": float(next_volume),
                            }
                        else:
                            raise ValueError("native_working_state_invalid")
                except Exception as exc:
                    raise AlphaMaxRuntimeContractError(
                        f"alpha_max_indicator_native_fold_invalid:{symbol}"
                    ) from exc
                observed_rows += frame.height
                previous_timestamp_ms = int(timestamps[-1])
            if not observed_rows:
                raise AlphaMaxRuntimeContractError(
                    f"alpha_max_indicator_symbol_coverage_invalid:{symbol}"
                )
        return tuple(releases), expected_windows

    @staticmethod
    def _partition_intersects(
        relative_path: str,
        *,
        start: datetime,
        end: datetime,
    ) -> bool:
        try:
            partition_start = datetime.strptime(
                Path(relative_path).name,
                "%Y-%m.parquet",
            ).replace(tzinfo=UTC)
        except ValueError as exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_indicator_partition_name_invalid"
            ) from exc
        partition_end = (
            datetime(partition_start.year + 1, 1, 1, tzinfo=UTC)
            if partition_start.month == 12
            else datetime(partition_start.year, partition_start.month + 1, 1, tzinfo=UTC)
        )
        return partition_start < end and start < partition_end


def _is_proc_fd_anchored_path(path: Path) -> bool:
    return (
        path.parts[:4] == ("/", "proc", "self", "fd")
        and len(path.parts) >= 5
        and path.parts[4].isdigit()
    )


def _is_proc_fd_parent(path: Path) -> bool:
    return _is_proc_fd_anchored_path(path) and len(path.parts) == 5


def _validated_output_target(output_root: str | os.PathLike[str]) -> tuple[Path, Path]:
    target = Path(_require_exact_explicit_path(output_root))
    parent = target.parent
    if target == parent or target.name in {"", ".", ".."}:
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_invalid")
    proc_fd_parent = _is_proc_fd_anchored_path(parent)
    try:
        if not proc_fd_parent and str(parent.resolve(strict=True)) != str(parent):
            raise AlphaMaxRuntimeContractError("alpha_max_output_parent_identity_invalid")
        status = parent.stat() if proc_fd_parent else parent.lstat()
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_output_parent_identity_invalid") from exc
    if not stat.S_ISDIR(status.st_mode) or (not proc_fd_parent and stat.S_ISLNK(status.st_mode)):
        raise AlphaMaxRuntimeContractError("alpha_max_output_parent_identity_invalid")
    return target, parent


def _rename_bundle_noreplace(staging: Path, target: Path) -> None:
    """Atomically publish a complete stage without replacing any extant path."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AlphaMaxRuntimeContractError("alpha_max_atomic_publish_unsupported")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_exists")
    raise OSError(error_number, os.strerror(error_number), str(target))


def _write_bundle_file(root: Path, relative_path: str, payload: bytes) -> Path:
    if type(payload) is not bytes:
        raise TypeError("alpha_max_bundle_artifact_bytes_required")
    if type(relative_path) is not str or "\\" in relative_path:
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_relative_path_invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_relative_path_invalid")
    try:
        root_observed = root.lstat()
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_parent_invalid") from exc
    if not stat.S_ISDIR(root_observed.st_mode) or stat.S_ISLNK(root_observed.st_mode):
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_parent_invalid")
    opened_directories: list[int] = []
    try:
        root_fd = _alpha_max_open_directory_at(root)
        opened_directories.append(root_fd)
        _alpha_max_require_open_identity(root_fd, root_observed, directory=True)
        parent_fd = root_fd
        for part in relative.parts[:-1]:
            try:
                observed = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except FileExistsError:
                    pass
                observed = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                raise AlphaMaxRuntimeContractError("alpha_max_bundle_parent_invalid")
            child_fd = _alpha_max_open_directory_at(part, dir_fd=parent_fd)
            opened_directories.append(child_fd)
            _alpha_max_require_open_identity(child_fd, observed, directory=True)
            parent_fd = child_fd
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(relative.name, flags, 0o600, dir_fd=parent_fd)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(errno.EIO, "alpha_max_bundle_short_write")
                view = view[written:]
            os.fsync(fd)
            os.fchmod(fd, 0o444)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(parent_fd)
    finally:
        for directory_fd in reversed(opened_directories):
            os.close(directory_fd)
    return root / relative


def _write_bundle_file_atomic(root: Path, relative_path: str, payload: bytes) -> Path:
    """Publish a fully fsynced recognized temporary file by rename-no-replace."""
    if type(payload) is not bytes:
        raise TypeError("alpha_max_bundle_artifact_bytes_required")
    if type(relative_path) is not str or "\\" in relative_path:
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_relative_path_invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_relative_path_invalid")
    try:
        root_observed = root.lstat()
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_parent_invalid") from exc
    if not stat.S_ISDIR(root_observed.st_mode) or stat.S_ISLNK(root_observed.st_mode):
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_parent_invalid")
    temporary_name = f".{relative.name}.atomic-{os.getpid()}-{os.urandom(12).hex()}"
    temporary_relative = relative.with_name(temporary_name)
    temporary_path = _write_bundle_file(
        root,
        temporary_relative.as_posix(),
        payload,
    )
    opened_directories: list[int] = []
    try:
        root_fd = _alpha_max_open_directory_at(root)
        opened_directories.append(root_fd)
        _alpha_max_require_open_identity(root_fd, root_observed, directory=True)
        parent_fd = root_fd
        for part in relative.parts[:-1]:
            try:
                observed = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except FileExistsError:
                    pass
                observed = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                raise AlphaMaxRuntimeContractError("alpha_max_bundle_parent_invalid")
            child_fd = _alpha_max_open_directory_at(part, dir_fd=parent_fd)
            opened_directories.append(child_fd)
            _alpha_max_require_open_identity(child_fd, observed, directory=True)
            parent_fd = child_fd
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise AlphaMaxRuntimeContractError("alpha_max_atomic_file_publish_unsupported")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        if (
            renameat2(
                parent_fd,
                os.fsencode(temporary_name),
                parent_fd,
                os.fsencode(relative.name),
                1,
            )
            != 0
        ):
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise AlphaMaxRuntimeContractError("alpha_max_bundle_artifact_exists")
            raise OSError(error_number, os.strerror(error_number), str(root / relative))
        os.fsync(parent_fd)
    except Exception:
        try:
            observed = temporary_path.lstat()
            if stat.S_ISREG(observed.st_mode) and int(observed.st_nlink) == 1:
                temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        for directory_fd in reversed(opened_directories):
            os.close(directory_fd)
    return root / relative


def _alpha_max_cleanup_atomic_bundle_temps(root: Path) -> None:
    """Remove only writer-owned crash remnants identified by an exact name shape."""
    for path in sorted(root.rglob("*"), key=lambda value: str(value)):
        name = path.name
        if ".atomic-" not in name or not name.startswith("."):
            continue
        prefix, marker = name.rsplit(".atomic-", 1)
        parts = marker.split("-", 1)
        recognized = (
            len(prefix) > 1
            and len(parts) == 2
            and parts[0].isdigit()
            and len(parts[1]) == 24
            and all(character in "0123456789abcdef" for character in parts[1])
        )
        if not recognized:
            continue
        observed = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(observed.st_mode) or int(observed.st_nlink) != 1:
            raise AlphaMaxRuntimeContractError("alpha_max_atomic_temp_identity_invalid")
        path.unlink()
        _fsync_directory(path.parent)


def _alpha_max_open_directory_at(name: str | Path, *, dir_fd: int | None = None) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(name, flags, dir_fd=dir_fd)


def _alpha_max_require_open_identity(
    opened_fd: int,
    observed: os.stat_result,
    *,
    directory: bool,
) -> None:
    opened = os.fstat(opened_fd)
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        int(opened.st_dev) != int(observed.st_dev)
        or int(opened.st_ino) != int(observed.st_ino)
        or not expected_type(opened.st_mode)
        or (not directory and int(opened.st_nlink) != 1)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_tree_invalid")


def _alpha_max_seal_directory_fd(directory_fd: int) -> None:
    with os.scandir(directory_fd) as entries:
        ordered = sorted(entries, key=lambda value: value.name)
    for entry in ordered:
        observed = entry.stat(follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            child_fd = _alpha_max_open_directory_at(entry.name, dir_fd=directory_fd)
            try:
                _alpha_max_require_open_identity(child_fd, observed, directory=True)
                _alpha_max_seal_directory_fd(child_fd)
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise AlphaMaxRuntimeContractError("alpha_max_bundle_tree_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(entry.name, flags, dir_fd=directory_fd)
        try:
            _alpha_max_require_open_identity(file_fd, observed, directory=False)
            os.fchmod(file_fd, 0o444)
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
    os.fchmod(directory_fd, 0o555)
    os.fsync(directory_fd)


def _make_bundle_immutable(root: Path) -> None:
    observed = root.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_tree_invalid")
    root_fd = _alpha_max_open_directory_at(root)
    try:
        _alpha_max_require_open_identity(root_fd, observed, directory=True)
        _alpha_max_seal_directory_fd(root_fd)
    finally:
        os.close(root_fd)
    _fsync_directory(root.parent)


def _alpha_max_cleanup_directory_fd(directory_fd: int) -> None:
    os.fchmod(directory_fd, 0o700)
    with os.scandir(directory_fd) as entries:
        ordered = sorted(entries, key=lambda value: value.name)
    for entry in ordered:
        observed = entry.stat(follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            child_fd = _alpha_max_open_directory_at(entry.name, dir_fd=directory_fd)
            try:
                _alpha_max_require_open_identity(child_fd, observed, directory=True)
                _alpha_max_cleanup_directory_fd(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(entry.name, dir_fd=directory_fd)
        else:
            os.unlink(entry.name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _cleanup_partial_bundle(root: Path) -> None:
    try:
        observed = root.lstat()
    except FileNotFoundError:
        return
    try:
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            os.unlink(root)
        else:
            root_fd = _alpha_max_open_directory_at(root)
            try:
                _alpha_max_require_open_identity(root_fd, observed, directory=True)
                _alpha_max_cleanup_directory_fd(root_fd)
            finally:
                os.close(root_fd)
            os.rmdir(root)
    except FileNotFoundError, OSError:
        return
    try:
        _fsync_directory(root.parent)
    except OSError:
        pass


def _create_alpha_max_run_owned_root(output_root: str | os.PathLike[str]) -> Path:
    """Create the final canonical root; receipts must never name a staging path."""
    target, parent = _validated_output_target(output_root)
    if target.exists() or target.is_symlink():
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_exists")
    created = False
    try:
        os.mkdir(target, mode=0o700)
        created = True
        _fsync_directory(parent)
        for relative in (
            "manifests",
            "manifests/validation_train_fit",
            "manifests/prelock_final_refit",
            "capsules",
            "capsules/validation_train_fit",
            "capsules/prelock_final_refit",
        ):
            path = target / relative
            os.mkdir(path, mode=0o700)
            _fsync_directory(path.parent)
    except FileExistsError as exc:
        if created:
            _cleanup_partial_bundle(target)
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_exists") from exc
    except Exception:
        if created:
            _cleanup_partial_bundle(target)
        raise
    return target


def _alpha_max_create_or_resume_run_root(
    output_root: str | os.PathLike[str],
    *,
    config_bytes: bytes,
    attempt_descriptor_sha256: str,
    sealed_role: str = "prelock",
) -> Path:
    if type(config_bytes) is not bytes:
        raise TypeError("alpha_max_run_config_bytes_invalid")
    if (
        type(attempt_descriptor_sha256) is not str
        or len(attempt_descriptor_sha256) != 64
        or any(character not in "0123456789abcdef" for character in attempt_descriptor_sha256)
        or sealed_role not in {"prelock", "historical"}
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_attempt_descriptor_sha256_invalid")
    target, _parent = _validated_output_target(output_root)
    binding_bytes = (
        _canonical_bytes(
            {
                "artifact_kind": "alpha_max_restartable_attempt_binding.v1",
                "attempt_descriptor_sha256": attempt_descriptor_sha256,
            }
        )
        + b"\n"
    )
    if not target.exists() and not target.is_symlink():
        root = _create_alpha_max_run_owned_root(target)
        _write_bundle_file_atomic(root, "inputs/config.json", config_bytes)
        _write_bundle_file_atomic(root, "inputs/restart_attempt.json", binding_bytes)
        return root
    try:
        status = target.lstat()
        if (
            not stat.S_ISDIR(status.st_mode)
            or stat.S_ISLNK(status.st_mode)
            or (
                not _is_proc_fd_parent(target.parent)
                and str(target.resolve(strict=True)) != str(target)
            )
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_output_root_not_resumable")
        if (target / "SEALED.json").exists():
            snapshot = _snapshot_bundle_tree(target, require_immutable=False)
            if sealed_role == "prelock":
                _validate_prelock_snapshot(snapshot)
            else:
                _validate_historical_output_snapshot(snapshot)
            _config_receipt, sealed_config = read_artifact_bytes(
                target / "inputs/config.json",
                artifact_id="recovered-prelock-config",
            )
            _binding_receipt, sealed_binding = read_artifact_bytes(
                target / "inputs/restart_attempt.json",
                artifact_id="recovered-prelock-attempt-binding",
            )
            if sealed_config != config_bytes or sealed_binding != binding_bytes:
                raise AlphaMaxRuntimeContractError("alpha_max_output_root_resume_binding_mismatch")
            _make_bundle_immutable(target)
            _snapshot_bundle_tree(target)
            raise AlphaMaxRuntimeContractError("alpha_max_output_root_recovered_sealed")
        _alpha_max_cleanup_atomic_bundle_temps(target)
        config_path = target / "inputs/config.json"
        binding_path = target / "inputs/restart_attempt.json"
        observed_files = {
            path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
        }
        if not config_path.exists():
            if observed_files:
                raise AlphaMaxRuntimeContractError("alpha_max_output_root_not_resumable")
            _write_bundle_file_atomic(target, "inputs/config.json", config_bytes)
            observed_files.add("inputs/config.json")
        if not binding_path.exists():
            if observed_files != {"inputs/config.json"}:
                raise AlphaMaxRuntimeContractError("alpha_max_output_root_not_resumable")
            _write_bundle_file_atomic(
                target,
                "inputs/restart_attempt.json",
                binding_bytes,
            )
        config_receipt, observed_config = read_artifact_bytes(
            config_path,
            artifact_id="resumed-prelock-config",
        )
        binding_receipt, observed_binding = read_artifact_bytes(
            binding_path,
            artifact_id="resumed-prelock-attempt-binding",
        )
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_not_resumable") from exc
    if (
        observed_config != config_bytes
        or observed_binding != binding_bytes
        or config_receipt.canonical_path
        != str((target / "inputs/config.json").resolve(strict=True))
        or binding_receipt.canonical_path
        != str((target / "inputs/restart_attempt.json").resolve(strict=True))
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_resume_binding_mismatch")
    return target


def _finalize_alpha_max_run_owned_root(
    root: Path,
    artifacts: Mapping[str, bytes],
    *,
    seal_bytes: bytes,
) -> AlphaMaxSealedBundle:
    """Write/verify final-path artifacts and publish only by writing SEALED last."""
    if not isinstance(root, Path) or not root.is_absolute():
        raise TypeError("alpha_max_run_owned_root_required")
    if (root / "SEALED.json").exists():
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_already_sealed")
    written: list[str] = []
    seal_fd = -1
    try:
        for raw_path, payload in sorted(artifacts.items()):
            relative_path = _safe_bundle_relative_path(raw_path)
            if relative_path == "SEALED.json" or type(payload) is not bytes:
                raise AlphaMaxRuntimeContractError("alpha_max_bundle_inventory_invalid")
            target = root / relative_path
            if target.exists():
                receipt, observed = read_artifact_bytes(
                    target,
                    artifact_id=f"run_owned:{relative_path}",
                )
                if observed != payload or receipt.canonical_path != str(
                    target.resolve(strict=True)
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_run_owned_artifact_mismatch")
            else:
                _write_bundle_file_atomic(root, relative_path, payload)
            written.append(str(target))
        seal_path, seal_fd = _alpha_max_create_empty_seal(root)
        _make_bundle_immutable(root)
        _alpha_max_write_final_seal(seal_fd, seal_bytes)
        closing_fd = seal_fd
        seal_fd = -1
        os.close(closing_fd)
    except Exception:
        close_failure: OSError | None = None
        if seal_fd >= 0:
            closing_fd = seal_fd
            seal_fd = -1
            try:
                os.close(closing_fd)
            except OSError as exc:
                close_failure = exc
        try:
            _alpha_max_rollback_run_owned_root(root)
        except Exception as cleanup_exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_run_owned_root_rollback_failed"
            ) from cleanup_exc
        if close_failure is not None:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_final_seal_close_failed"
            ) from close_failure
        raise
    return AlphaMaxSealedBundle(
        output_root=str(root),
        stable_paths=tuple(written),
        seal_path=str(seal_path),
        seal_sha256=_sha256(seal_bytes),
    )


def _alpha_max_rollback_run_owned_root(root: Path) -> None:
    """Remove a failed command-owned root through retained no-follow identities."""
    parent_fd = _alpha_max_open_directory_at(root.parent)
    try:
        observed = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise AlphaMaxRuntimeContractError("alpha_max_run_owned_root_rollback_invalid")
        root_fd = _alpha_max_open_directory_at(root.name, dir_fd=parent_fd)
        try:
            _alpha_max_require_open_identity(root_fd, observed, directory=True)
            _alpha_max_cleanup_directory_fd(root_fd)
        finally:
            os.close(root_fd)
        os.rmdir(root.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _alpha_max_create_empty_seal(root: Path) -> tuple[Path, int]:
    """Create an invalid empty seal and retain its descriptor across tree chmod."""
    observed = root.lstat()
    root_fd = _alpha_max_open_directory_at(root)
    try:
        _alpha_max_require_open_identity(root_fd, observed, directory=True)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        seal_fd = os.open("SEALED.json", flags, 0o400, dir_fd=root_fd)
        status = os.fstat(seal_fd)
        if not stat.S_ISREG(status.st_mode) or int(status.st_nlink) != 1:
            os.close(seal_fd)
            raise AlphaMaxRuntimeContractError("alpha_max_final_seal_invalid")
        os.fsync(seal_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    return root / "SEALED.json", seal_fd


def _alpha_max_write_final_seal(seal_fd: int, seal_bytes: bytes) -> None:
    if type(seal_bytes) is not bytes or not seal_bytes:
        raise AlphaMaxRuntimeContractError("alpha_max_final_seal_invalid")
    view = memoryview(seal_bytes)
    written = 0
    while written < len(view):
        written += os.write(seal_fd, view[written:])
    os.fsync(seal_fd)


def _alpha_max_display_bundle(
    bundle: AlphaMaxSealedBundle,
    *,
    anchored_root: Path,
    display_root: Path,
) -> AlphaMaxSealedBundle:
    def display(path: str) -> str:
        return str(display_root / Path(path).relative_to(anchored_root))

    return AlphaMaxSealedBundle(
        output_root=str(display_root),
        stable_paths=tuple(display(path) for path in bundle.stable_paths),
        seal_path=display(bundle.seal_path),
        seal_sha256=bundle.seal_sha256,
    )


def _write_sealed_bundle(
    output_root: str | os.PathLike[str],
    artifacts: Mapping[str, bytes],
    *,
    seal_bytes: bytes,
) -> AlphaMaxSealedBundle:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise AlphaMaxRuntimeContractError("alpha_max_bundle_inventory_empty")
    normalized: dict[str, bytes] = {}
    for raw_path, payload in artifacts.items():
        relative_path = _safe_bundle_relative_path(raw_path)
        if relative_path in normalized or type(payload) is not bytes:
            raise AlphaMaxRuntimeContractError("alpha_max_bundle_inventory_invalid")
        normalized[relative_path] = payload
    target, parent = _validated_output_target(output_root)
    if target.exists() or target.is_symlink():
        raise AlphaMaxRuntimeContractError("alpha_max_output_root_exists")
    root = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=parent,
        )
    )
    published = False
    try:
        _fsync_directory(parent)
        relative_paths = tuple(sorted(normalized))
        for relative_path in relative_paths:
            _write_bundle_file(root, relative_path, normalized[relative_path])
        _write_bundle_file(root, "SEALED.json", seal_bytes)
        _make_bundle_immutable(root)
        _rename_bundle_noreplace(root, target)
        published = True
        _fsync_directory(parent)
    except Exception:
        cleanup_target = target if published else root
        try:
            _alpha_max_rollback_run_owned_root(cleanup_target)
        except Exception as cleanup_exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_sealed_bundle_rollback_failed"
            ) from cleanup_exc
        raise
    written_paths = tuple(str(target / relative_path) for relative_path in relative_paths)
    seal_path = target / "SEALED.json"
    return AlphaMaxSealedBundle(
        output_root=str(target),
        stable_paths=written_paths,
        seal_path=str(seal_path),
        seal_sha256=_sha256(seal_bytes),
    )


def create_alpha_max_prelock_bundle(
    output_root: str | os.PathLike[str],
    stable_artifacts: Mapping[str, bytes],
    *,
    prelock_champion: str | None,
    selected_candidate_id: str | None,
) -> AlphaMaxSealedBundle:
    """Write an immutable evidence-owned prelock inventory into a new root."""
    reject_ambient_lq_environment()
    from lumina_quant.research.alpha_max_evidence import build_alpha_max_prelock_seal

    seal = build_alpha_max_prelock_seal(
        stable_artifacts,
        prelock_champion=prelock_champion,
        selected_candidate_id=selected_candidate_id,
    )
    return _write_sealed_bundle(
        output_root,
        stable_artifacts,
        seal_bytes=seal.canonical_bytes,
    )


@dataclass(frozen=True, slots=True)
class _AlphaMaxBundleSnapshot:
    root_path: str
    rows: tuple[tuple[object, ...], ...]
    seal_bytes: bytes


def _stream_bundle_file_sha256(path: Path, expected: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            int(opened.st_dev) != int(expected.st_dev)
            or int(opened.st_ino) != int(expected.st_ino)
            or int(opened.st_nlink) != 1
            or not stat.S_ISREG(opened.st_mode)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_identity_invalid")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        sealed = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        int(sealed.st_dev) != int(opened.st_dev)
        or int(sealed.st_ino) != int(opened.st_ino)
        or int(sealed.st_size) != int(opened.st_size)
        or int(sealed.st_mtime_ns) != int(opened.st_mtime_ns)
        or int(sealed.st_ctime_ns) != int(opened.st_ctime_ns)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_snapshot")
    return digest.hexdigest()


def _snapshot_bundle_tree(
    root_value: str | os.PathLike[str],
    *,
    require_immutable: bool = True,
) -> _AlphaMaxBundleSnapshot:
    raw = _require_exact_explicit_path(root_value)
    root = Path(raw)
    try:
        if str(root.resolve(strict=True)) != str(root) and not _is_proc_fd_anchored_path(root):
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_identity_invalid")
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_identity_invalid") from exc
    rows: list[tuple[object, ...]] = []
    seal_bytes: bytes | None = None
    root_fd = -1
    retained_directories: list[tuple[int, os.stat_result]] = []
    retained_files: list[tuple[int, os.stat_result]] = []
    try:
        root_status = root.lstat()
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or stat.S_ISLNK(root_status.st_mode)
            or (require_immutable and root_status.st_mode & 0o222)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_identity_invalid")
        root_fd = _alpha_max_open_directory_at(root)
        _alpha_max_require_open_identity(root_fd, root_status, directory=True)
        retained_directories.append((root_fd, root_status))
        rows.append(
            (
                ".",
                "directory",
                int(root_status.st_dev),
                int(root_status.st_ino),
                int(root_status.st_mtime_ns),
                int(root_status.st_ctime_ns),
            )
        )

        def snapshot_directory(
            directory_fd: int,
            prefix: str,
            directory_status: os.stat_result,
        ) -> None:
            nonlocal seal_bytes
            names = tuple(sorted(os.listdir(directory_fd)))
            for name in names:
                relative = f"{prefix}/{name}" if prefix else name
                observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISLNK(observed.st_mode) or (
                    require_immutable and observed.st_mode & 0o222
                ):
                    raise AlphaMaxRuntimeContractError("alpha_max_prelock_identity_invalid")
                if stat.S_ISDIR(observed.st_mode):
                    child_fd = _alpha_max_open_directory_at(name, dir_fd=directory_fd)
                    _alpha_max_require_open_identity(child_fd, observed, directory=True)
                    retained_directories.append((child_fd, observed))
                    rows.append(
                        (
                            relative,
                            "directory",
                            int(observed.st_dev),
                            int(observed.st_ino),
                            int(observed.st_mtime_ns),
                            int(observed.st_ctime_ns),
                        )
                    )
                    snapshot_directory(child_fd, relative, observed)
                    continue
                if not stat.S_ISREG(observed.st_mode) or int(observed.st_nlink) != 1:
                    raise AlphaMaxRuntimeContractError("alpha_max_prelock_identity_invalid")
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                file_fd = os.open(name, flags, dir_fd=directory_fd)
                opened = os.fstat(file_fd)
                if _alpha_max_stat_identity(opened) != _alpha_max_stat_identity(observed):
                    os.close(file_fd)
                    raise AlphaMaxRuntimeContractError("alpha_max_prelock_identity_invalid")
                retained_files.append((file_fd, opened))
                digest = hashlib.sha256()
                chunks: list[bytes] | None = [] if relative == "SEALED.json" else None
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    if chunks is not None:
                        chunks.append(chunk)
                sealed = os.fstat(file_fd)
                if _alpha_max_stat_identity(sealed) != _alpha_max_stat_identity(opened):
                    raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_snapshot")
                if chunks is not None:
                    seal_bytes = b"".join(chunks)
                rows.append(
                    (
                        relative,
                        "file",
                        int(observed.st_dev),
                        int(observed.st_ino),
                        int(observed.st_size),
                        int(observed.st_mtime_ns),
                        int(observed.st_ctime_ns),
                        digest.hexdigest(),
                    )
                )
            if tuple(sorted(os.listdir(directory_fd))) != names:
                raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_snapshot")
            final_directory = os.fstat(directory_fd)
            if _alpha_max_stat_identity(final_directory) != _alpha_max_stat_identity(
                directory_status
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_snapshot")

        snapshot_directory(root_fd, "", root_status)
        for file_fd, expected in retained_files:
            if _alpha_max_stat_identity(os.fstat(file_fd)) != _alpha_max_stat_identity(expected):
                raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_snapshot")
        for directory_fd, expected in retained_directories:
            if _alpha_max_stat_identity(os.fstat(directory_fd)) != _alpha_max_stat_identity(
                expected
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_snapshot")
        current_root = root.lstat()
        if _alpha_max_stat_identity(current_root) != _alpha_max_stat_identity(root_status) or (
            not _is_proc_fd_anchored_path(root) and str(root.resolve(strict=True)) != str(root)
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_snapshot")
    except OSError as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_identity_invalid") from exc
    finally:
        for file_fd, _expected in reversed(retained_files):
            os.close(file_fd)
        for directory_fd, _expected in reversed(retained_directories):
            os.close(directory_fd)
    if seal_bytes is None:
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_seal_missing")
    return _AlphaMaxBundleSnapshot(
        root_path=str(root),
        rows=tuple(rows),
        seal_bytes=seal_bytes,
    )


def _validate_prelock_snapshot(snapshot: _AlphaMaxBundleSnapshot) -> tuple[str, bytes]:
    if type(snapshot) is not _AlphaMaxBundleSnapshot:
        raise TypeError("alpha_max_prelock_snapshot_identity_invalid")
    files = {str(row[0]): row for row in snapshot.rows if row[1] == "file"}
    sealed = files.get("SEALED.json")
    if sealed is None:
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_seal_missing")
    seal_bytes = snapshot.seal_bytes
    payload = _strict_json_object(seal_bytes)
    if (
        seal_bytes != _canonical_bytes(payload) + b"\n"
        or payload.get("artifact_kind") != "alpha_max_immutable_prelock_seal.v1"
        or payload.get("immutable") is not True
        or payload.get("historical_evaluation_inputs_included") is not False
        or type(payload.get("artifacts")) is not list
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_seal_invalid")
    inventory: dict[str, tuple[int, str]] = {}
    for entry in payload["artifacts"]:
        if type(entry) is not dict or set(entry) != {"byte_count", "relative_path", "sha256"}:
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_seal_invalid")
        relative = _safe_bundle_relative_path(entry["relative_path"])
        if relative in inventory:
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_seal_invalid")
        inventory[relative] = (int(entry["byte_count"]), str(entry["sha256"]))
    if set(inventory) != set(files) - {"SEALED.json"}:
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_inventory_mismatch")
    for relative, (byte_count, digest) in inventory.items():
        row = files[relative]
        if row[4] != byte_count or row[-1] != digest:
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_inventory_mismatch")
    snapshot_bytes = _canonical_bytes([list(row) for row in snapshot.rows])
    return _sha256(snapshot_bytes), seal_bytes


def _validate_historical_output_snapshot(
    snapshot: _AlphaMaxBundleSnapshot,
) -> tuple[str, bytes]:
    if type(snapshot) is not _AlphaMaxBundleSnapshot:
        raise TypeError("alpha_max_historical_snapshot_identity_invalid")
    files = {str(row[0]): row for row in snapshot.rows if row[1] == "file"}
    if "SEALED.json" not in files:
        raise AlphaMaxRuntimeContractError("alpha_max_historical_seal_missing")
    seal_bytes = snapshot.seal_bytes
    payload = _strict_json_object(seal_bytes)
    if (
        seal_bytes != _canonical_bytes(payload) + b"\n"
        or set(payload)
        != {
            "artifact_kind",
            "completion_id",
            "historical_artifacts",
            "immutable",
            "prelock_seal_sha256",
            "prelock_snapshot_sha256",
        }
        or payload["artifact_kind"] != "alpha_max_append_only_historical_package.v1"
        or type(payload["completion_id"]) is not str
        or not payload["completion_id"]
        or payload["immutable"] is not True
        or type(payload["historical_artifacts"]) is not list
        or any(
            type(payload[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", payload[field]) is None
            for field in ("prelock_seal_sha256", "prelock_snapshot_sha256")
        )
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_historical_seal_invalid")
    inventory: dict[str, tuple[int, str]] = {}
    for entry in payload["historical_artifacts"]:
        if (
            type(entry) is not dict
            or set(entry) != {"byte_count", "relative_path", "sha256"}
            or type(entry["byte_count"]) is not int
            or entry["byte_count"] <= 0
            or type(entry["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_historical_seal_invalid")
        relative = _safe_bundle_relative_path(entry["relative_path"])
        if relative in inventory:
            raise AlphaMaxRuntimeContractError("alpha_max_historical_seal_invalid")
        inventory[relative] = (entry["byte_count"], entry["sha256"])
    if set(inventory) != set(files) - {"SEALED.json"}:
        raise AlphaMaxRuntimeContractError("alpha_max_historical_inventory_mismatch")
    for relative, (byte_count, digest) in inventory.items():
        row = files[relative]
        if row[4] != byte_count or row[-1] != digest:
            raise AlphaMaxRuntimeContractError("alpha_max_historical_inventory_mismatch")
    snapshot_bytes = _canonical_bytes([list(row) for row in snapshot.rows])
    return _sha256(snapshot_bytes), seal_bytes


def _acquire_historical_completion_claim(
    output_parent: Path,
    *,
    completion_id: str,
    prelock_seal_sha256: str,
    attempt_descriptor_sha256: str,
    output_root: Path,
) -> Path:
    claim_id = _sha256(f"{prelock_seal_sha256}\0{completion_id}".encode())
    claim = output_parent / f".alpha-max-completion-{claim_id}.claim"
    payload = (
        _canonical_bytes(
            {
                "artifact_kind": "alpha_max_historical_completion_claim.v2",
                "attempt_descriptor_sha256": attempt_descriptor_sha256,
                "completion_id": completion_id,
                "output_root": str(output_root),
                "prelock_seal_sha256": prelock_seal_sha256,
            }
        )
        + b"\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(claim, flags, 0o600)
    except FileExistsError as exc:
        try:
            receipt, observed = read_artifact_bytes(
                claim,
                artifact_id="alpha-max-historical-completion-claim",
            )
            status = claim.lstat()
        except OSError as read_exc:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_historical_completion_duplicate"
            ) from read_exc
        if (
            observed != payload
            or receipt.byte_count != len(payload)
            or status.st_nlink != 1
            or status.st_mode & 0o222
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_historical_completion_duplicate") from exc
        return claim
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(errno.EIO, "alpha_max_completion_claim_short_write")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    except Exception:
        os.close(fd)
        claim.unlink(missing_ok=True)
        _fsync_directory(output_parent)
        raise
    os.close(fd)
    _fsync_directory(output_parent)
    return claim


def _release_historical_completion_claim(claim: Path) -> None:
    try:
        claim.unlink()
        _fsync_directory(claim.parent)
    except FileNotFoundError:
        return


def create_alpha_max_historical_package(
    sealed_prelock_directory: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    historical_artifacts: Mapping[str, bytes],
    *,
    completion_id: str,
) -> AlphaMaxSealedBundle:
    """Append one immutable historical package without mutating the prelock tree."""
    reject_ambient_lq_environment()
    if (
        type(completion_id) is not str
        or not completion_id
        or Path(completion_id).name != completion_id
        or completion_id in {".", ".."}
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_historical_completion_id_invalid")
    before = _snapshot_bundle_tree(sealed_prelock_directory)
    prelock_snapshot_sha, prelock_seal_bytes = _validate_prelock_snapshot(before)
    output_target = Path(_require_exact_explicit_path(output_root))
    output_parent = output_target.parent
    standalone_attempt_sha256 = _sha256(
        (
            "alpha_max_standalone_historical_package\0"
            + completion_id
            + "\0"
            + _sha256(prelock_seal_bytes)
            + "\0"
            + str(output_target)
        ).encode()
    )
    claim = _acquire_historical_completion_claim(
        output_parent,
        completion_id=completion_id,
        prelock_seal_sha256=_sha256(prelock_seal_bytes),
        attempt_descriptor_sha256=standalone_attempt_sha256,
        output_root=output_target,
    )
    completion_probe_errors = (OSError, ValueError, AlphaMaxRuntimeContractError)
    try:
        for sibling in output_parent.iterdir():
            candidate = sibling / "SEALED.json"
            try:
                status = candidate.lstat()
                if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
                    continue
                _receipt, candidate_bytes = read_artifact_bytes(
                    candidate,
                    artifact_id="historical_completion_probe",
                )
                candidate_payload = _strict_json_object(candidate_bytes)
            except completion_probe_errors:
                continue
            if (
                candidate_payload.get("artifact_kind")
                == "alpha_max_append_only_historical_package.v1"
                and candidate_payload.get("completion_id") == completion_id
                and candidate_payload.get("prelock_seal_sha256") == _sha256(prelock_seal_bytes)
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_historical_completion_duplicate")
        normalized = {
            _safe_bundle_relative_path(path): payload
            for path, payload in historical_artifacts.items()
        }
        if not normalized or len(normalized) != len(historical_artifacts):
            raise AlphaMaxRuntimeContractError("alpha_max_historical_inventory_invalid")
        inventory = [
            {
                "byte_count": len(payload),
                "relative_path": path,
                "sha256": _sha256(payload),
            }
            for path, payload in sorted(normalized.items())
        ]
        seal_payload = {
            "artifact_kind": "alpha_max_append_only_historical_package.v1",
            "completion_id": completion_id,
            "historical_artifacts": inventory,
            "immutable": True,
            "prelock_seal_sha256": _sha256(prelock_seal_bytes),
            "prelock_snapshot_sha256": prelock_snapshot_sha,
        }
        seal_bytes = _canonical_bytes(seal_payload) + b"\n"
        after = _snapshot_bundle_tree(sealed_prelock_directory)
        if after != before:
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_historical_run")
        return _write_sealed_bundle(output_root, normalized, seal_bytes=seal_bytes)
    except Exception:
        _release_historical_completion_claim(claim)
        raise


def _alpha_max_failure_reason(stage: str, exc: BaseException) -> str:
    token = str(exc).strip() or type(exc).__name__
    return f"{stage}:{token}"


def _alpha_max_blocked_matrix_payload(
    *,
    domain: str,
    blocking_reasons: Sequence[str],
) -> bytes:
    reasons = tuple(str(value) for value in blocking_reasons)
    statuses: list[dict[str, object]] = []
    for row_id in _ALPHA_MAX_CURRENT_ROW_IDS:
        if row_id in _ALPHA_MAX_UNAVAILABLE_ROWS:
            row_role = "incumbent_unavailable"
            status = "incumbent_replay_unavailable"
        elif row_id in _ALPHA_MAX_DIAGNOSTIC_ROWS:
            row_role = "track_b_diagnostic"
            status = "diagnostic_report_only"
        else:
            row_role = "resolvable_candidate"
            status = "blocked_before_engine"
        for nominal_cost_bps in ALPHA_MAX_COST_CELL_BPS:
            statuses.append(
                {
                    "blocking_reasons": list(reasons) if status == "blocked_before_engine" else [],
                    "capsule_sha256": None,
                    "engine_constructed": False,
                    "manifest_sha256": None,
                    "nominal_cost_bps": nominal_cost_bps,
                    "row_id": row_id,
                    "row_role": row_role,
                    "selection_eligible": False,
                    "status": status,
                }
            )
    payload = {
        "artifact_kind": "alpha_max_matrix_statuses.v1",
        "domain": domain,
        "engine_cell_count": 0,
        "status_count": len(statuses),
        "statuses": statuses,
    }
    if len(statuses) != 84:
        raise AlphaMaxRuntimeContractError("alpha_max_matrix_cardinality_mismatch")
    return _canonical_bytes(payload) + b"\n"


def _alpha_max_root_validation(
    roots: Sequence[tuple[str, str, str]],
    *,
    exchange: str,
    availability_start_by_kind: Mapping[str, Mapping[str, datetime]],
    availability_end_by_kind: Mapping[str, Mapping[str, datetime]],
    max_workers: int = _ALPHA_MAX_MAX_PARALLEL_WORKERS,
) -> tuple[dict[tuple[str, str], AlphaMaxRootSeal], tuple[str, ...]]:
    if type(max_workers) is not int or not 1 <= max_workers <= _ALPHA_MAX_MAX_PARALLEL_WORKERS:
        raise AlphaMaxRuntimeContractError("alpha_max_root_worker_count_invalid")
    seals: dict[tuple[str, str], AlphaMaxRootSeal] = {}
    failures: list[str] = []
    root_specs = tuple(roots)

    def seal_root(
        spec: tuple[str, str, str],
    ) -> tuple[tuple[str, str], AlphaMaxRootSeal | None, str | None]:
        root_id, root_kind, root_path = spec
        try:
            availability_start_by_symbol = availability_start_by_kind[root_kind]
            availability_end_by_symbol = availability_end_by_kind[root_kind]
            seal = seal_alpha_max_root_tree(
                root_id,
                root_kind,
                root_path,
                exchange=exchange,
                availability_start_by_symbol=availability_start_by_symbol,
                availability_end_by_symbol=availability_end_by_symbol,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return (
                (root_id, root_kind),
                None,
                _alpha_max_failure_reason(f"{root_id}_{root_kind}_root", exc),
            )
        return (root_id, root_kind), seal, None

    if len(root_specs) <= 1 or max_workers == 1:
        results = tuple(seal_root(spec) for spec in root_specs)
    else:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(root_specs)),
            thread_name_prefix="alpha-max-root-seal",
        ) as executor:
            results = tuple(executor.map(seal_root, root_specs))
    for key, seal, failure in results:
        if failure is not None:
            failures.append(failure)
        elif seal is not None:
            seals[key] = seal
    return seals, tuple(failures)


def _alpha_max_root_artifacts(
    seals: Mapping[tuple[str, str], AlphaMaxRootSeal],
) -> dict[str, bytes]:
    return {
        f"roots/{root_kind}/{root_id}.json": seal.canonical_bytes
        for (root_id, root_kind), seal in sorted(seals.items())
    }


def _alpha_max_collect_existing_artifacts(
    root: Path,
    *,
    allowed_paths: set[str],
    required_paths: set[str],
) -> dict[str, bytes]:
    if (
        type(allowed_paths) is not set
        or type(required_paths) is not set
        or not required_paths <= allowed_paths
        or any(type(path) is not str for path in allowed_paths)
    ):
        raise TypeError("alpha_max_resumable_inventory_contract_invalid")
    allowed_directories = {
        parent.as_posix()
        for relative in allowed_paths
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    allowed_directories.update(
        {
            "capsules",
            "capsules/prelock_final_refit",
            "capsules/validation_train_fit",
            "manifests",
            "manifests/prelock_final_refit",
            "manifests/validation_train_fit",
        }
    )
    artifacts: dict[str, bytes] = {}
    for path in sorted(root.rglob("*"), key=lambda value: str(value.relative_to(root))):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            if path.is_symlink() or relative not in allowed_directories:
                raise AlphaMaxRuntimeContractError(
                    f"alpha_max_resumable_inventory_unknown:{relative}"
                )
            continue
        if not path.is_file():
            raise AlphaMaxRuntimeContractError(f"alpha_max_resumable_inventory_unknown:{relative}")
        if relative not in allowed_paths or relative == "SEALED.json":
            raise AlphaMaxRuntimeContractError(f"alpha_max_resumable_inventory_unknown:{relative}")
        _receipt, payload = read_artifact_bytes(
            path,
            artifact_id=f"alpha_max_run_owned:{relative}",
        )
        artifacts[relative] = payload
    if not required_paths <= set(artifacts):
        raise AlphaMaxRuntimeContractError("alpha_max_resumable_inventory_required_missing")
    return artifacts


def _alpha_max_collect_fresh_run_artifacts(root: Path) -> dict[str, bytes]:
    """Collect a non-resumable run root created and owned by this process."""
    artifacts: dict[str, bytes] = {}
    for path in sorted(root.rglob("*"), key=lambda value: str(value.relative_to(root))):
        if path.is_dir():
            if path.is_symlink():
                raise AlphaMaxRuntimeContractError("alpha_max_run_owned_inventory_invalid")
            continue
        if not path.is_file() or path.is_symlink():
            raise AlphaMaxRuntimeContractError("alpha_max_run_owned_inventory_invalid")
        relative = path.relative_to(root).as_posix()
        if relative == "SEALED.json":
            raise AlphaMaxRuntimeContractError("alpha_max_output_root_already_sealed")
        _receipt, payload = read_artifact_bytes(
            path,
            artifact_id=f"alpha_max_run_owned:{relative}",
        )
        artifacts[relative] = payload
    return artifacts


def _alpha_max_historical_activation_paths() -> set[str]:
    return {
        "inputs/config.json",
        "inputs/restart_attempt.json",
        *(
            f"capsules/prelock_final_refit/{row_id}/{fold_id}.json"
            for row_id in _ALPHA_MAX_RESOLVABLE_ROWS
            for fold_id in _ALPHA_MAX_HISTORICAL_FOLD_IDS[1:]
        ),
    }


def _alpha_max_matrix_artifacts(matrix: _AlphaMaxCompletedMatrix) -> dict[str, bytes]:
    domain_path = "validation" if matrix.domain == "validation" else "historical_exposed_evaluation"
    artifacts = {"status/matrix.json": matrix.status_payload}
    for row in matrix.rows:
        artifacts[f"evidence/{domain_path}/rows/{row.row_id}.json"] = canonical_alpha_max_row_bytes(
            row
        )
        for cell in row.cost_cells:
            artifacts[f"evidence/{domain_path}/cells/{row.row_id}/{cell.nominal_cost_bps}.json"] = (
                canonical_alpha_max_cost_cell_bytes(cell)
            )
    return artifacts


def _alpha_max_trend_liquidity_falsifier_artifact(
    matrix: _AlphaMaxCompletedMatrix,
    train_liquidity_buckets: AlphaMaxTrainLiquidityBuckets,
) -> bytes:
    """Seal E20 from the exact nominal-30 trend fold receipts, report-only."""
    try:
        cell = matrix.cells[("component_trend_1x", 30)]
    except KeyError as exc:  # pragma: no cover - complete matrix invariant
        raise AlphaMaxRuntimeContractError("alpha_max_trend_nominal_30_cell_missing") from exc
    pre_gate = cell.pre_gate_evidence
    if type(pre_gate) is not AlphaMaxCostCellPreGateEvidence:
        raise AlphaMaxRuntimeContractError("alpha_max_trend_nominal_30_fold_evidence_missing")
    admitted = train_liquidity_buckets.admitted_symbols
    fold_hashes: list[str] = []
    contributions: list[Mapping[str, float]] = []
    for fold in pre_gate.fold_runs:
        diagnostics = fold.actual_engine_run.report_only_diagnostics
        fold_hashes.append(fold.sha256)
        contributions.append(
            MappingProxyType(
                {symbol: diagnostics.symbol_contribution_usdt[symbol] for symbol in admitted}
            )
        )
    falsifier = build_alpha_max_trend_liquidity_falsifier(
        domain=matrix.domain,
        train_liquidity_buckets=train_liquidity_buckets,
        fold_run_sha256s=tuple(fold_hashes),
        symbol_contribution_usdt_by_fold=tuple(contributions),
    )
    return falsifier.canonical_bytes


_ALPHA_MAX_TRAINING_WORKER_CONTEXT: (
    tuple[
        AlphaMaxRuntimePreflight,
        Path,
        Mapping[tuple[str, str], AlphaMaxRootSeal],
        tuple[str, ...],
        tuple[
            Path,
            str,
            str,
            str,
            str,
            tuple[str, ...],
            tuple[int, int],
            tuple[dict[str, object], ...],
            Mapping[str, object],
            int,
            int,
            tuple[int, int],
            str,
        ],
        Mapping[str, AlphaMaxManifestReceipt],
    ]
    | None
) = None


def _alpha_max_replay_training_component_worker(component_id: str) -> tuple[str, bytes]:
    """Fork-only replay; the parent remains the canonical component publisher."""
    context = _ALPHA_MAX_TRAINING_WORKER_CONTEXT
    if context is None or type(component_id) is not str:
        raise AlphaMaxRuntimeContractError("alpha_max_training_worker_context_invalid")
    preflight, _output_root, root_seals, admitted_symbols, store_binding, manifests = context
    manifest = manifests.get(component_id)
    if type(manifest) is not AlphaMaxManifestReceipt:
        raise AlphaMaxRuntimeContractError("alpha_max_training_worker_component_invalid")
    reject_ambient_lq_environment()
    store = _AlphaMaxPrecomputeCheckpointStore(
        store_binding[0],
        attempt_descriptor_sha256=store_binding[1],
        attempt_role=store_binding[2],
        domain=store_binding[3],
        runtime_identity_sha256=store_binding[4],
        training_day_ids=store_binding[5],
        transaction_lock_identity=store_binding[6],
    )
    _verify_alpha_max_checkpoint_implementation_inventory(list(store_binding[7]))
    if _alpha_max_indicator_runtime_binding() != store_binding[8]:
        raise AlphaMaxRuntimeContractError("alpha_max_training_worker_runtime_identity_invalid")
    output_parent_fd, output_fd, output_identity, output_name = store_binding[9:]
    try:
        output_entry = os.stat(output_name, dir_fd=output_parent_fd, follow_symlinks=False)
        output_opened = os.fstat(output_fd)
    except OSError as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_training_worker_output_authority_invalid"
        ) from exc
    if (
        (int(output_entry.st_dev), int(output_entry.st_ino)) != output_identity
        or (int(output_opened.st_dev), int(output_opened.st_ino)) != output_identity
        or not stat.S_ISDIR(output_entry.st_mode)
        or stat.S_ISLNK(output_entry.st_mode)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_worker_output_authority_invalid")
    calendar, returns, native_finalization = _alpha_max_replay_training_component_returns(
        preflight,
        output_root=Path(f"/proc/self/fd/{output_fd}"),
        manifest_receipt=manifest,
        admitted_symbols=admitted_symbols,
        root_seals=root_seals,
        checkpoint_store=store,
    )
    try:
        output_entry = os.stat(output_name, dir_fd=output_parent_fd, follow_symlinks=False)
        output_opened = os.fstat(output_fd)
    except OSError as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_training_worker_output_authority_invalid"
        ) from exc
    if (int(output_entry.st_dev), int(output_entry.st_ino)) != output_identity or (
        int(output_opened.st_dev),
        int(output_opened.st_ino),
    ) != output_identity:
        raise AlphaMaxRuntimeContractError("alpha_max_training_worker_output_authority_invalid")
    return component_id, _alpha_max_training_component_checkpoint_bytes(
        component_id=component_id,
        manifest=manifest,
        calendar=calendar,
        returns=returns,
        native_finalization=native_finalization,
    )


def run_alpha_max_prelock_process(
    *,
    config: str | os.PathLike[str],
    contract_manifest: str | os.PathLike[str],
    prior_trial_blob: str | os.PathLike[str],
    exchange: str,
    output_root: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    warmup_raw_root: str,
    warmup_feature_root: str,
    train_raw_root: str,
    train_feature_root: str,
    purge_raw_root: str,
    purge_feature_root: str,
    validation_raw_root: str,
    validation_feature_root: str,
    embargo_raw_root: str,
    embargo_feature_root: str,
    bootstrap_implementation_inventory: list[dict[str, object]] | None = None,
    max_training_workers: int = _ALPHA_MAX_MAX_PARALLEL_WORKERS,
) -> AlphaMaxCommandResult:
    """Run the physical prelock boundary from explicit frozen inputs only.

    Invalid or incomplete inputs fail before the output root is created.  A
    no-champion bundle is reserved for a structurally complete actual-engine
    matrix whose gates found no survivor.
    """
    reject_ambient_lq_environment()
    if (
        type(max_training_workers) is not int
        or not 1 <= max_training_workers <= _ALPHA_MAX_MAX_PARALLEL_WORKERS
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_training_worker_count_invalid")
    implementation_inventory = _verify_alpha_max_checkpoint_implementation_inventory(
        _alpha_max_checkpoint_implementation_inventory()
        if bootstrap_implementation_inventory is None
        else bootstrap_implementation_inventory
    )
    if exchange != "binance":
        raise AlphaMaxRuntimeContractError("alpha_max_exchange_invalid")
    preflight = preflight_alpha_max_runtime_contract(config)

    failures: list[str] = []
    contract_seal: AlphaMaxContractManifestSeal | None = None
    try:
        contract_seal = seal_alpha_max_contract_manifest(contract_manifest)
    except (OSError, TypeError, ValueError) as exc:
        failures.append(_alpha_max_failure_reason("contract_manifest", exc))

    roots = (
        ("warmup", "raw", warmup_raw_root),
        ("warmup", "feature", warmup_feature_root),
        ("train", "raw", train_raw_root),
        ("train", "feature", train_feature_root),
        ("purge", "raw", purge_raw_root),
        ("purge", "feature", purge_feature_root),
        ("validation", "raw", validation_raw_root),
        ("validation", "feature", validation_feature_root),
        ("embargo", "raw", embargo_raw_root),
        ("embargo", "feature", embargo_feature_root),
    )
    if contract_seal is None:
        root_seals = {}
        root_failures = ("root_validation:contract_manifest_required",)
    else:
        root_seals, root_failures = _alpha_max_root_validation(
            roots,
            exchange=exchange,
            availability_start_by_kind={
                "feature": contract_seal.feature_availability_start_by_symbol,
                "raw": contract_seal.raw_availability_start_by_symbol,
            },
            availability_end_by_kind={
                "feature": contract_seal.feature_availability_end_by_symbol,
                "raw": contract_seal.raw_availability_end_by_symbol,
            },
        )
    failures.extend(root_failures)
    required_feature_roots = {
        (root_id, "feature") for root_id in ("warmup", "train", "purge", "validation", "embargo")
    }
    if required_feature_roots <= set(root_seals):
        try:
            _validate_alpha_max_adjacent_feature_roots(
                root_seals,
                (
                    ("warmup", "train"),
                    ("train", "purge"),
                    ("purge", "validation"),
                    ("validation", "embargo"),
                ),
            )
        except (TypeError, ValueError) as exc:
            failures.append(_alpha_max_failure_reason("adjacent_feature_roots", exc))

    admission: AlphaMaxAdmissionComputation | None = None
    required_admission_roots = {
        ("warmup", "raw"),
        ("warmup", "feature"),
        ("train", "raw"),
        ("train", "feature"),
    }
    if required_admission_roots <= set(root_seals):
        try:
            admission = _compute_alpha_max_admission_from_seals(
                warmup_raw=root_seals[("warmup", "raw")],
                warmup_feature=root_seals[("warmup", "feature")],
                train_raw=root_seals[("train", "raw")],
                train_feature=root_seals[("train", "feature")],
            )
        except (OSError, TypeError, ValueError) as exc:
            failures.append(_alpha_max_failure_reason("train_admission", exc))
    else:
        failures.append("train_admission:required_warmup_train_root_validation_failed")

    if failures:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_prelock_input_invalid:" + "|".join(sorted(failures))
        )
    if contract_seal is None or admission is None:
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_input_incomplete")

    nodes = _alpha_max_current_nodes(preflight)
    admitted_symbols = _validate_admitted_symbols(
        preflight,
        admission.artifact.admitted_symbols,
    )
    train_liquidity_buckets = build_alpha_max_train_liquidity_buckets(admission)
    try:
        prior_trial_path = _require_exact_explicit_path(prior_trial_blob)
        prior_bytes = read_alpha_max_prior_trial_blob_input(prior_trial_path)
        trial_ledger = build_alpha_max_trial_ledger(
            prior_bytes,
            _strict_json_object(preflight.config_bytes),
        )
        _validated_output_target(output_root)
    except (OSError, TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_prelock_input_invalid:"
            + _alpha_max_failure_reason("trial_ledger_or_output", exc)
        ) from exc

    attempt_descriptor = _alpha_max_prelock_checkpoint_descriptor(
        preflight=preflight,
        contract_seal=contract_seal,
        root_seals=root_seals,
        admitted_symbols=admitted_symbols,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        implementation_inventory=implementation_inventory,
        prior_trial_binding={
            "byte_count": len(prior_bytes),
            "path": prior_trial_path,
            "sha256": _sha256(prior_bytes),
        },
        _include_v2_bindings=True,
    )
    checkpoint_store = _AlphaMaxCellCheckpointStore(
        checkpoint_root,
        output_root=output_root,
        descriptor=attempt_descriptor,
        config_bytes=preflight.config_bytes,
    )
    root = _alpha_max_create_or_resume_run_root(
        checkpoint_store.output_root,
        config_bytes=preflight.config_bytes,
        attempt_descriptor_sha256=checkpoint_store.descriptor_sha256,
    )
    root = checkpoint_store.bind_output_root()
    run_preflight = preflight_alpha_max_runtime_contract(root / "inputs/config.json")
    try:
        component_ids = (
            "component_carry_1x",
            "component_near_high_1x",
            "component_trend_1x",
        )
        rows_by_id = {str(row["row_id"]): row for row in nodes}
        component_manifests: dict[str, AlphaMaxManifestReceipt] = {}
        train_calendar: tuple[str, ...] | None = None
        train_returns: dict[str, tuple[float, ...]] = {}
        train_native_finalizations: dict[str, AlphaMaxNativeFinalizationReceipt] = {}
        whole_component_bytes: dict[str, bytes] = {}
        for component_id in component_ids:
            row = rows_by_id[component_id]
            manifest = _alpha_max_materialize_manifest_receipt(
                run_preflight,
                output_root=root,
                phase="validation_train_fit",
                row=row,
                weights={component_id: 1.0},
                gross=1.0,
                admitted_symbols=admitted_symbols,
                admission_sha256=admission.artifact.sha256,
            )
            component_manifests[component_id] = manifest
            checkpoint_bytes = checkpoint_store.load_precompute(
                unit_kind="training_component",
                unit_id=component_id,
            )
            if checkpoint_bytes is not None:
                whole_component_bytes[component_id] = checkpoint_bytes
        missing_component_ids = tuple(
            component_id
            for component_id in component_ids
            if component_id not in whole_component_bytes
        )
        if missing_component_ids:
            global _ALPHA_MAX_TRAINING_WORKER_CONTEXT
            _ALPHA_MAX_TRAINING_WORKER_CONTEXT = (
                run_preflight,
                root,
                root_seals,
                admitted_symbols,
                (
                    checkpoint_store.training_precompute_store()._display_root,
                    checkpoint_store.training_precompute_store()._attempt_descriptor_sha256,
                    checkpoint_store.training_precompute_store()._attempt_role,
                    checkpoint_store.training_precompute_store()._domain,
                    checkpoint_store.training_precompute_store()._runtime_identity_sha256,
                    checkpoint_store.training_precompute_store()._training_day_ids,
                    checkpoint_store.training_precompute_store()._transaction_lock_identity,
                    tuple(implementation_inventory),
                    checkpoint_store._runtime_identity,
                    checkpoint_store._output_parent_fd,
                    checkpoint_store._bound_output_fd,
                    checkpoint_store._bound_output_identity,
                    checkpoint_store._display_output_root.name,
                ),
                MappingProxyType(component_manifests),
            )
            try:
                with ProcessPoolExecutor(
                    max_workers=min(max_training_workers, len(missing_component_ids)),
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    completed = dict(
                        executor.map(
                            _alpha_max_replay_training_component_worker,
                            missing_component_ids,
                        )
                    )
            except BrokenProcessPool:
                # A killed worker cannot have made a semantic claim; immutable
                # completed day units remain the only resumable progress.
                raise
            except AlphaMaxRuntimeContractError, TypeError, ValueError:
                checkpoint_store.training_precompute_store().poison()
                raise
            finally:
                _ALPHA_MAX_TRAINING_WORKER_CONTEXT = None
            try:
                if tuple(completed) != missing_component_ids:
                    raise AlphaMaxRuntimeContractError("alpha_max_training_worker_result_invalid")
                train_raw_seal = root_seals[("train", "raw")]
                _validate_alpha_max_root_seals(
                    raw_root=train_raw_seal.path,
                    phase_id="train",
                    ordered_lookup=_alpha_max_phase_lookup(root_seals, "train"),
                    raw_root_seals=(train_raw_seal,),
                    feature_root_seals=tuple(
                        root_seals[(root_id, "feature")]
                        for root_id in _alpha_max_expected_root_sequence("train")
                    ),
                    required=True,
                    repeat_hash=True,
                )
                for component_id in missing_component_ids:
                    whole_component_bytes[component_id] = checkpoint_store.seal_precompute(
                        unit_kind="training_component",
                        unit_id=component_id,
                        data_bytes=completed[component_id],
                    )
                    _alpha_max_training_component_from_checkpoint(
                        whole_component_bytes[component_id],
                        preflight=run_preflight,
                        component_id=component_id,
                        manifest=component_manifests[component_id],
                    )
            except AlphaMaxRuntimeContractError, TypeError, ValueError:
                checkpoint_store.training_precompute_store().poison()
                raise
        try:
            for component_id in component_ids:
                manifest = component_manifests[component_id]
                checkpoint_bytes = whole_component_bytes[component_id]
                calendar, values, native_finalization = (
                    _alpha_max_training_component_from_checkpoint(
                        checkpoint_bytes,
                        preflight=run_preflight,
                        component_id=component_id,
                        manifest=manifest,
                    )
                )
                if train_calendar is None:
                    train_calendar = calendar
                elif calendar != train_calendar:
                    raise AlphaMaxRuntimeContractError(
                        "alpha_max_train_component_calendar_mismatch"
                    )
                train_returns[component_id] = values
                train_native_finalizations[component_id] = native_finalization
            if train_calendar is None:
                raise AlphaMaxRuntimeContractError("alpha_max_train_component_replay_empty")
            train_fit = _alpha_max_fit_weights(
                nodes,
                phase="train",
                calendar=train_calendar,
                component_returns=MappingProxyType(train_returns),
            )
        except AlphaMaxRuntimeContractError, KeyError, TypeError, ValueError:
            checkpoint_store.training_precompute_store().poison()
            raise

        scaled_ids = {"full_equal_risk_scaled", "full_shrunk_hrp_scaled"}
        prepared: dict[str, _AlphaMaxPreparedReplayRow] = {}
        for row_id in _ALPHA_MAX_RESOLVABLE_ROWS:
            if row_id in scaled_ids:
                continue
            row = rows_by_id[row_id]
            gross_rule = row.get("gross")
            if type(gross_rule) is not dict or gross_rule.get("method") != "fixed":
                raise AlphaMaxRuntimeContractError("alpha_max_row_gross_rule_invalid")
            gross = float(gross_rule["value"])
            manifest = component_manifests.get(row_id) or _alpha_max_materialize_manifest_receipt(
                run_preflight,
                output_root=root,
                phase="validation_train_fit",
                row=row,
                weights=train_fit.weights_by_row[row_id],
                gross=gross,
                admitted_symbols=admitted_symbols,
                admission_sha256=admission.artifact.sha256,
            )
            checkpoint_bytes = checkpoint_store.load_precompute(
                unit_kind="validation_row",
                unit_id=row_id,
            )
            if checkpoint_bytes is None:
                current = _alpha_max_prepare_validation_row(
                    run_preflight,
                    output_root=root,
                    row=row,
                    weights=train_fit.weights_by_row[row_id],
                    gross=gross,
                    admitted_symbols=admitted_symbols,
                    admission_sha256=admission.artifact.sha256,
                    root_seals=root_seals,
                    retained_manifest=manifest,
                )
                checkpoint_bytes = checkpoint_store.seal_precompute(
                    unit_kind="validation_row",
                    unit_id=row_id,
                    data_bytes=_alpha_max_prepared_row_checkpoint_bytes(
                        current,
                        domain="validation",
                    ),
                )
            prepared[row_id] = _alpha_max_restore_prepared_row_checkpoint(
                checkpoint_bytes,
                preflight=run_preflight,
                manifest=manifest,
                admitted_symbols=admitted_symbols,
                root_seals=root_seals,
                domain="validation",
                gross=gross,
                capsule_output_root=root,
            )

        def prepare_scaled(
            row: Mapping[str, object],
            gross: float,
        ) -> _AlphaMaxPreparedReplayRow:
            row_id = str(row["row_id"])
            manifest = _alpha_max_materialize_manifest_receipt(
                run_preflight,
                output_root=root,
                phase="validation_train_fit",
                row=row,
                weights=train_fit.weights_by_row[row_id],
                gross=gross,
                admitted_symbols=admitted_symbols,
                admission_sha256=admission.artifact.sha256,
            )
            checkpoint_bytes = checkpoint_store.load_precompute(
                unit_kind="validation_row",
                unit_id=row_id,
            )
            if checkpoint_bytes is None:
                current = _alpha_max_prepare_validation_row(
                    run_preflight,
                    output_root=root,
                    row=row,
                    weights=train_fit.weights_by_row[row_id],
                    gross=gross,
                    admitted_symbols=admitted_symbols,
                    admission_sha256=admission.artifact.sha256,
                    root_seals=root_seals,
                    retained_manifest=manifest,
                )
                checkpoint_bytes = checkpoint_store.seal_precompute(
                    unit_kind="validation_row",
                    unit_id=row_id,
                    data_bytes=_alpha_max_prepared_row_checkpoint_bytes(
                        current,
                        domain="validation",
                    ),
                )
            return _alpha_max_restore_prepared_row_checkpoint(
                checkpoint_bytes,
                preflight=run_preflight,
                manifest=manifest,
                admitted_symbols=admitted_symbols,
                root_seals=root_seals,
                domain="validation",
                gross=gross,
                capsule_output_root=root,
            )

        validation_matrix = _alpha_max_complete_domain_matrix(
            run_preflight,
            output_root=root,
            phase="validation_train_fit",
            nodes=nodes,
            admitted_symbols=admitted_symbols,
            domain="validation",
            trial_ledger=trial_ledger,
            prepared_rows=prepared,
            scaled_row_factory=prepare_scaled,
            checkpoint_store=checkpoint_store,
        )
        validation_trend_liquidity_falsifier = _alpha_max_trend_liquidity_falsifier_artifact(
            validation_matrix,
            train_liquidity_buckets,
        )
        prelock_selection = select_alpha_max_prelock_champion(validation_matrix.rows)

        validation_calendar: tuple[str, ...] | None = None
        validation_returns: dict[str, tuple[float, ...]] = {}
        for component_id in component_ids:
            pre_gate = validation_matrix.cells[(component_id, 20)].pre_gate_evidence
            if (
                type(pre_gate) is not AlphaMaxCostCellPreGateEvidence
                or type(pre_gate.combined_primary_return_stream) is not AlphaMaxPrimaryReturnStream
            ):
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_validation_component_daily_stream_missing"
                )
            calendar, values = _alpha_max_daily_returns_from_primary_stream(
                pre_gate.combined_primary_return_stream
            )
            if validation_calendar is None:
                validation_calendar = calendar
            elif calendar != validation_calendar:
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_validation_component_calendar_mismatch"
                )
            validation_returns[component_id] = values
        if validation_calendar is None:
            raise AlphaMaxRuntimeContractError(
                "alpha_max_validation_component_daily_stream_missing"
            )
        train_validation_calendar = (*train_calendar, *validation_calendar)
        train_validation_returns = {
            component_id: (*train_returns[component_id], *validation_returns[component_id])
            for component_id in component_ids
        }
        refit = _alpha_max_fit_weights(
            nodes,
            phase="train_validation",
            calendar=train_validation_calendar,
            component_returns=MappingProxyType(train_validation_returns),
        )

        final_manifests: dict[str, AlphaMaxManifestReceipt] = {}
        first_historical_receipts: dict[str, AlphaMaxCapsuleReceipt] = {}
        for row_id in _ALPHA_MAX_RESOLVABLE_ROWS:
            row = rows_by_id[row_id]
            manifest = _alpha_max_materialize_manifest_receipt(
                run_preflight,
                output_root=root,
                phase="prelock_final_refit",
                row=row,
                weights=refit.weights_by_row[row_id],
                gross=validation_matrix.gross_by_row[row_id],
                admitted_symbols=admitted_symbols,
                admission_sha256=admission.artifact.sha256,
            )
            final_manifests[row_id] = manifest
            checkpoint_bytes = checkpoint_store.load_precompute(
                unit_kind="final_refit_row",
                unit_id=row_id,
            )
            if checkpoint_bytes is None:
                prefix = _alpha_max_build_indicator_prefix(
                    run_preflight,
                    manifest_output_root=root,
                    phase="prelock_final_refit",
                    manifest_receipt=manifest,
                    admitted_symbols=admitted_symbols,
                    root_seals=root_seals,
                    phase_ids=("warmup", "train", "purge", "validation", "embargo"),
                )
                receipt = _alpha_max_materialize_capsule_receipt(
                    root,
                    row_id=row_id,
                    phase="prelock_final_refit",
                    prefix_id=_ALPHA_MAX_HISTORICAL_FOLD_IDS[0],
                    manifest_sha256=manifest.sha256,
                    capsule=prefix,
                )
                checkpoint_bytes = checkpoint_store.seal_precompute(
                    unit_kind="final_refit_row",
                    unit_id=row_id,
                    data_bytes=_alpha_max_final_refit_checkpoint_bytes(
                        manifest=manifest,
                        receipt=receipt,
                        gross=validation_matrix.gross_by_row[row_id],
                    ),
                )
            first_historical_receipts[row_id] = _alpha_max_restore_final_refit_checkpoint(
                checkpoint_bytes,
                manifest=manifest,
                capsule_root=root,
                gross=validation_matrix.gross_by_row[row_id],
            )
        if set(final_manifests) != set(_ALPHA_MAX_RESOLVABLE_ROWS) or set(
            first_historical_receipts
        ) != set(_ALPHA_MAX_RESOLVABLE_ROWS):
            raise AlphaMaxRuntimeContractError("alpha_max_final_refit_inventory_incomplete")

        terminal = build_alpha_max_terminal_state(
            prelock_selection=prelock_selection,
            champion_historical_nominal_30_cell=None,
            historical_ranking=None,
            incumbent_comparison_status="unavailable",
        )
        run_payload = {
            "artifact_kind": "alpha_max_prelock_process_result.v1",
            "engine_cell_count": 68,
            "failure_reasons": [],
            "physical_fold_run_count": validation_matrix.physical_fold_run_count,
            "prelock_champion": prelock_selection.prelock_champion,
            "selected_candidate_id": prelock_selection.selected_candidate_id,
            "status": "complete",
            "terminal_outcome": terminal.terminal_outcome,
        }
        generated_artifacts = {
            **_alpha_max_root_artifacts(root_seals),
            **_alpha_max_matrix_artifacts(validation_matrix),
            **{
                f"native_finalization/train/{component_id}.json": receipt.canonical_bytes
                for component_id, receipt in sorted(train_native_finalizations.items())
            },
            "admission/train.json": admission.artifact.canonical_bytes,
            "admission/train_computation.json": admission.canonical_bytes,
            "admission/train_liquidity_buckets.json": (train_liquidity_buckets.canonical_bytes),
            "allocation/train_fit.json": _canonical_bytes(train_fit.to_payload()) + b"\n",
            "allocation/train_validation_refit.json": (
                _canonical_bytes(refit.to_payload()) + b"\n"
            ),
            "inputs/contract_manifest.json": contract_seal.canonical_bytes,
            "inputs/prior_trial_inventory.json": prior_bytes,
            "run/prelock_result.json": _canonical_bytes(run_payload) + b"\n",
            "selection/prelock.json": prelock_selection.canonical_bytes,
            "diagnostics/validation/trend_liquidity_falsifier.json": (
                validation_trend_liquidity_falsifier
            ),
            "terminal/prelock.json": _canonical_bytes(terminal.to_payload()) + b"\n",
            "trial/ledger.json": _canonical_bytes(trial_ledger.to_payload()) + b"\n",
        }
        manifest_receipts = (
            *component_manifests.values(),
            *(value.manifest_receipt for value in prepared.values()),
            *final_manifests.values(),
        )
        capsule_receipts = (
            *(
                fold_input.capsule_receipt
                for value in prepared.values()
                for fold_input in value.fold_inputs
            ),
            *first_historical_receipts.values(),
        )
        activation_paths = {
            "inputs/config.json",
            "inputs/restart_attempt.json",
            *(receipt.relative_path for receipt in manifest_receipts),
            *(
                receipt.relative_path
                for receipt in capsule_receipts
                if type(receipt) is AlphaMaxCapsuleReceipt
            ),
        }
        existing_artifacts = _alpha_max_collect_existing_artifacts(
            root,
            allowed_paths=activation_paths | set(generated_artifacts),
            required_paths=activation_paths,
        )
        for relative, expected_bytes in generated_artifacts.items():
            observed = existing_artifacts.get(relative)
            if observed is not None and observed != expected_bytes:
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_resumable_generated_artifact_mismatch"
                )
        artifacts = {
            **{relative: existing_artifacts[relative] for relative in sorted(activation_paths)},
            **generated_artifacts,
        }
        from lumina_quant.research.alpha_max_evidence import build_alpha_max_prelock_seal

        seal = build_alpha_max_prelock_seal(
            artifacts,
            prelock_champion=prelock_selection.prelock_champion,
            selected_candidate_id=prelock_selection.selected_candidate_id,
        )
        _verify_alpha_max_checkpoint_implementation_inventory(implementation_inventory)
        prospective_bundle = AlphaMaxSealedBundle(
            output_root=str(root),
            stable_paths=tuple(str(root / relative) for relative in sorted(artifacts)),
            seal_path=str(root / "SEALED.json"),
            seal_sha256=_sha256(seal.canonical_bytes),
        )
        display_bundle = _alpha_max_display_bundle(
            prospective_bundle,
            anchored_root=root,
            display_root=checkpoint_store.display_output_root,
        )
        command_result = AlphaMaxCommandResult(
            exit_code=0,
            terminal_outcome=terminal.terminal_outcome,
            bundle=display_bundle,
            failure_reasons=(),
        )
        _finalize_alpha_max_run_owned_root(
            root,
            artifacts,
            seal_bytes=seal.canonical_bytes,
        )
        return command_result
    except Exception:
        raise


def _read_alpha_max_prelock_artifact(
    snapshot: _AlphaMaxBundleSnapshot,
    relative_path: str,
) -> bytes:
    safe = _safe_bundle_relative_path(relative_path)
    files = {str(row[0]): row for row in snapshot.rows if row[1] == "file"}
    expected = files.get(safe)
    if expected is None:
        raise AlphaMaxRuntimeContractError(f"alpha_max_prelock_artifact_missing:{safe}")
    path = Path(snapshot.root_path) / safe
    receipt, payload = read_artifact_bytes(path, artifact_id=f"prelock:{safe}")
    if receipt.byte_count != expected[4] or receipt.sha256 != expected[-1]:
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_artifact_identity_invalid")
    return payload


def _validate_complete_alpha_max_prelock_matrix(
    snapshot: _AlphaMaxBundleSnapshot,
    prelock_payload: Mapping[str, object],
) -> None:
    """Reject historical access unless all 68 selectable cells completed."""
    matrix_bytes = _read_alpha_max_prelock_artifact(snapshot, "status/matrix.json")
    matrix = _strict_json_object(matrix_bytes)
    statuses = matrix.get("statuses")
    top_level_keys = {
        "artifact_kind",
        "domain",
        "engine_cell_count",
        "physical_fold_run_count",
        "status_count",
        "statuses",
    }
    if (
        matrix_bytes != _canonical_bytes(matrix) + b"\n"
        or set(matrix) != top_level_keys
        or matrix.get("artifact_kind") != "alpha_max_matrix_statuses.v1"
        or matrix.get("domain") != "validation"
        or type(matrix.get("status_count")) is not int
        or matrix.get("status_count") != 84
        or type(matrix.get("engine_cell_count")) is not int
        or matrix.get("engine_cell_count") != 68
        or type(matrix.get("physical_fold_run_count")) is not int
        or matrix.get("physical_fold_run_count") != 816
        or type(prelock_payload.get("engine_cell_count")) is not int
        or prelock_payload.get("engine_cell_count") != 68
        or type(prelock_payload.get("physical_fold_run_count")) is not int
        or prelock_payload.get("physical_fold_run_count") != 816
        or type(statuses) is not list
        or len(statuses) != 84
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_matrix_incomplete")
    common_keys = {
        "capsule_sha256",
        "engine_constructed",
        "manifest_sha256",
        "nominal_cost_bps",
        "row_id",
        "row_role",
        "selection_eligible",
        "status",
    }

    def valid_sha256(value: object) -> bool:
        return (
            type(value) is str
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    observed: set[tuple[str, int]] = set()
    expected_order = tuple(
        (row_id, nominal)
        for row_id in _ALPHA_MAX_CURRENT_ROW_IDS
        for nominal in ALPHA_MAX_COST_CELL_BPS
    )
    for index, raw in enumerate(statuses):
        if type(raw) is not dict:
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_matrix_incomplete")
        row_id = raw.get("row_id")
        nominal = raw.get("nominal_cost_bps")
        if (
            type(row_id) is not str
            or type(nominal) is not int
            or nominal not in ALPHA_MAX_COST_CELL_BPS
            or (row_id, nominal) in observed
            or (row_id, nominal) != expected_order[index]
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_matrix_incomplete")
        observed.add((row_id, nominal))
        resolvable = row_id in _ALPHA_MAX_RESOLVABLE_ROWS
        if resolvable:
            if (
                set(raw) != common_keys | {"cell_sha256"}
                or raw.get("engine_constructed") is not True
                or raw.get("status") != "resolved_engine_cell_complete"
                or raw.get("row_role") != "resolvable_candidate"
                or type(raw.get("selection_eligible")) is not bool
                or not valid_sha256(raw.get("capsule_sha256"))
                or not valid_sha256(raw.get("cell_sha256"))
                or not valid_sha256(raw.get("manifest_sha256"))
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_prelock_matrix_incomplete")
        elif row_id in _ALPHA_MAX_UNAVAILABLE_ROWS:
            if (
                set(raw) != common_keys
                or raw.get("engine_constructed") is not False
                or raw.get("status") != "incumbent_replay_unavailable"
                or raw.get("row_role") != "incumbent_unavailable"
                or raw.get("selection_eligible") is not False
                or raw.get("capsule_sha256") is not None
                or raw.get("manifest_sha256") is not None
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_prelock_matrix_incomplete")
        elif row_id in _ALPHA_MAX_DIAGNOSTIC_ROWS:
            if (
                set(raw) != common_keys
                or raw.get("engine_constructed") is not False
                or raw.get("status") != "diagnostic_report_only"
                or raw.get("row_role") != "track_b_diagnostic"
                or raw.get("selection_eligible") is not False
                or raw.get("capsule_sha256") is not None
                or raw.get("manifest_sha256") is not None
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_prelock_matrix_incomplete")
        else:
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_matrix_incomplete")
    if observed != set(expected_order):
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_matrix_incomplete")


def _alpha_max_selection_from_bytes(
    raw: bytes,
    *,
    role: str,
) -> AlphaMaxSelectionResult:
    payload = _strict_json_object(raw)
    if role not in {"prelock_selection", "historical_report"}:
        raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
    expected_kind = (
        "alpha_max_prelock_selection.v2"
        if role == "prelock_selection"
        else "alpha_max_historical_report_ranking.v2"
    )
    top_level_keys = {
        "artifact_kind",
        "decisions",
        "historical_evaluation_leader",
        "prelock_champion",
        "ranked_candidate_ids",
        "role",
        "scaling_attributions",
        "selected_candidate_id",
    }
    if (
        raw != _canonical_bytes(payload) + b"\n"
        or set(payload) != top_level_keys
        or payload.get("artifact_kind") != expected_kind
        or payload.get("role") != role
        or type(payload.get("decisions")) is not list
        or type(payload.get("ranked_candidate_ids")) is not list
        or type(payload.get("scaling_attributions")) is not list
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
    allowed_gates = {
        "dsr",
        "evidence",
        "funding_coverage",
        "hash_validity",
        "manifest_validity",
        "mdd",
        "native_data_coverage",
        "pbo",
        "positive_metrics",
        "reconciliation",
        "ruin",
        "scaled_1x_sibling",
        "spa",
    }
    allowed_bands = {"hard_reject", "normal", "not_evaluated", "soft", "terminal"}
    expected_decision_ids = tuple(sorted(_ALPHA_MAX_RESOLVABLE_ROWS))
    decisions: list[AlphaMaxGateDecision] = []
    observed_decision_ids: list[str] = []
    for value in payload["decisions"]:
        if type(value) is not dict or set(value) != {
            "comparator_row_id",
            "eligible",
            "evaluated_gates",
            "gate_mdd",
            "mdd_band",
            "rejection_reasons",
            "row_id",
        }:
            raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
        row_id = value["row_id"]
        eligible = value["eligible"]
        evaluated = value["evaluated_gates"]
        reasons = value["rejection_reasons"]
        gate_mdd = value["gate_mdd"]
        mdd_band = value["mdd_band"]
        comparator = value["comparator_row_id"]
        if (
            type(row_id) is not str
            or row_id not in _ALPHA_MAX_RESOLVABLE_ROWS
            or type(eligible) is not bool
            or type(evaluated) is not list
            or not evaluated
            or any(type(item) is not str or item not in allowed_gates for item in evaluated)
            or len(evaluated) != len(set(evaluated))
            or type(reasons) is not list
            or any(type(item) is not str or not item for item in reasons)
            or (eligible and reasons)
            or (not eligible and not reasons)
            or type(mdd_band) is not str
            or mdd_band not in allowed_bands
            or (
                gate_mdd is not None
                and (
                    type(gate_mdd) not in {int, float}
                    or not math.isfinite(gate_mdd)
                    or not 0.0 <= gate_mdd <= 1.0
                )
            )
            or (
                comparator is not None
                and (
                    type(comparator) is not str
                    or comparator not in _ALPHA_MAX_RESOLVABLE_ROWS
                    or comparator == row_id
                )
            )
            or (eligible and (gate_mdd is None or mdd_band not in {"normal", "soft"}))
            or (mdd_band == "normal" and comparator is not None)
            or (mdd_band == "hard_reject" and reasons != ["mdd_above_hard_limit"])
            or (
                mdd_band == "terminal"
                and (
                    eligible
                    or gate_mdd is not None
                    or evaluated != ["ruin"]
                    or reasons != ["ruin_detected"]
                    or comparator is not None
                )
            )
            or (mdd_band == "not_evaluated" and comparator is not None)
            or (
                mdd_band == "soft"
                and comparator is None
                and reasons != ["soft_mdd_requires_normal_comparator"]
            )
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
        observed_decision_ids.append(row_id)
        decisions.append(
            AlphaMaxGateDecision(
                row_id=row_id,
                eligible=eligible,
                evaluated_gates=tuple(evaluated),
                rejection_reasons=tuple(reasons),
                gate_mdd=(None if gate_mdd is None else float(gate_mdd)),
                mdd_band=mdd_band,
                comparator_row_id=comparator,
            )
        )
    if tuple(observed_decision_ids) != expected_decision_ids:
        raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
    ranked_values = payload["ranked_candidate_ids"]
    if (
        any(type(value) is not str for value in ranked_values)
        or len(ranked_values) != len(set(ranked_values))
        or any(value not in _ALPHA_MAX_RESOLVABLE_ROWS for value in ranked_values)
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
    ranked = tuple(ranked_values)
    decisions_by_id = {value.row_id: value for value in decisions}
    eligible_ids = {value.row_id for value in decisions if value.eligible}
    if set(ranked) != eligible_ids:
        raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
    scaling_attributions: list[AlphaMaxScalingAttribution] = []
    scaling_keys = {
        "attribution_label",
        "comparison_role",
        "dependency_rejection_reason",
        "exposure_normalization",
        "matched_domain_sha256",
        "nominal_cost_bps",
        "passive_scaled_counterfactual",
        "scaled_minus_sibling_cagr",
        "scaled_minus_sibling_calmar",
        "scaled_minus_sibling_net_sharpe",
        "scaled_minus_sibling_total_return",
        "scaled_row_id",
        "sibling_dependency_satisfied",
        "sibling_exposure_normalized_return",
        "sibling_gate_eligible",
        "sibling_gross_exposure",
        "sibling_positive_exposure_normalized",
        "sibling_row_id",
    }
    expected_scaled_ids = ("full_equal_risk_scaled", "full_shrunk_hrp_scaled")
    observed_scaled_ids: list[str] = []

    def optional_finite_number(value: object) -> float | None:
        if value is None:
            return None
        if type(value) not in {int, float} or not math.isfinite(value):
            raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
        return float(value)

    for value in payload["scaling_attributions"]:
        if type(value) is not dict or set(value) != scaling_keys:
            raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
        scaled_row_id = value["scaled_row_id"]
        sibling_row_id = value["sibling_row_id"]
        dependency_reason = value["dependency_rejection_reason"]
        if (
            type(scaled_row_id) is not str
            or type(sibling_row_id) is not str
            or type(value["comparison_role"]) is not str
            or value["comparison_role"] != role
            or type(value["nominal_cost_bps"]) is not int
            or value["nominal_cost_bps"] != 30
            or type(value["matched_domain_sha256"]) is not str
            or type(value["sibling_gate_eligible"]) is not bool
            or type(value["sibling_gross_exposure"]) not in {int, float}
            or value["sibling_gross_exposure"] != 1.0
            or type(value["exposure_normalization"]) is not str
            or type(value["sibling_positive_exposure_normalized"]) is not bool
            or type(value["sibling_dependency_satisfied"]) is not bool
            or (
                dependency_reason is not None
                and (
                    type(dependency_reason) is not str
                    or dependency_reason
                    not in {
                        "scaled_1x_exposure_normalized_nonpositive",
                        "scaled_1x_sibling_not_eligible",
                    }
                )
            )
            or type(value["attribution_label"]) is not str
            or type(value["passive_scaled_counterfactual"]) is not str
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
        try:
            attribution = AlphaMaxScalingAttribution(
                scaled_row_id=scaled_row_id,
                sibling_row_id=sibling_row_id,
                comparison_role=value["comparison_role"],
                nominal_cost_bps=value["nominal_cost_bps"],
                matched_domain_sha256=value["matched_domain_sha256"],
                sibling_gate_eligible=value["sibling_gate_eligible"],
                sibling_gross_exposure=float(value["sibling_gross_exposure"]),
                exposure_normalization=value["exposure_normalization"],
                sibling_exposure_normalized_return=optional_finite_number(
                    value["sibling_exposure_normalized_return"]
                ),
                sibling_positive_exposure_normalized=value["sibling_positive_exposure_normalized"],
                sibling_dependency_satisfied=value["sibling_dependency_satisfied"],
                dependency_rejection_reason=dependency_reason,
                scaled_minus_sibling_total_return=optional_finite_number(
                    value["scaled_minus_sibling_total_return"]
                ),
                scaled_minus_sibling_cagr=optional_finite_number(
                    value["scaled_minus_sibling_cagr"]
                ),
                scaled_minus_sibling_calmar=optional_finite_number(
                    value["scaled_minus_sibling_calmar"]
                ),
                scaled_minus_sibling_net_sharpe=optional_finite_number(
                    value["scaled_minus_sibling_net_sharpe"]
                ),
                attribution_label=value["attribution_label"],
                passive_scaled_counterfactual=value["passive_scaled_counterfactual"],
            )
        except (TypeError, ValueError) as exc:
            raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid") from exc
        observed_scaled_ids.append(scaled_row_id)
        scaling_attributions.append(attribution)
    if tuple(observed_scaled_ids) != expected_scaled_ids:
        raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
    prelock_champion = payload.get("prelock_champion")
    selected_candidate_id = payload.get("selected_candidate_id")
    historical_leader = payload.get("historical_evaluation_leader")
    leader = ranked[0] if ranked else None
    if (
        any(
            value is not None and (type(value) is not str or not value)
            for value in (
                prelock_champion,
                selected_candidate_id,
                historical_leader,
            )
        )
        or (
            role == "prelock_selection"
            and (
                prelock_champion != leader
                or selected_candidate_id != leader
                or historical_leader is not None
            )
        )
        or (
            role == "historical_report"
            and (
                prelock_champion is not None
                or selected_candidate_id is not None
                or historical_leader != leader
            )
        )
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
    for attribution in scaling_attributions:
        sibling_decision = decisions_by_id[attribution.sibling_row_id]
        scaled_decision = decisions_by_id[attribution.scaled_row_id]
        if (
            attribution.sibling_gate_eligible is not sibling_decision.eligible
            or (scaled_decision.eligible and not attribution.sibling_dependency_satisfied)
            or (
                "scaled_1x_sibling" in scaled_decision.evaluated_gates
                and scaled_decision.rejection_reasons != (attribution.dependency_rejection_reason,)
            )
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_selection_artifact_invalid")
    return AlphaMaxSelectionResult(
        role=role,
        decisions=tuple(decisions),
        ranked_candidate_ids=ranked,
        prelock_champion=prelock_champion,
        selected_candidate_id=selected_candidate_id,
        historical_evaluation_leader=historical_leader,
        scaling_attributions=tuple(scaling_attributions),
        canonical_bytes=raw,
        sha256=_sha256(raw),
    )


def _alpha_max_capsule_from_receipt(
    receipt: AlphaMaxCapsuleReceipt,
) -> AlphaMaxIndicatorCapsule:
    state = dict(receipt.state_payload)
    expected = {
        "capsule",
        "capsule_sha256",
        "discarded_signal_count",
        "fill_event_count",
        "finalized_children",
        "funding_event_count",
        "manifest_sha256",
        "market_event_count",
        "native_finalization_sha256",
        "order_event_count",
        "phase_id",
        "portfolio_mode",
        "trade_count",
        "windows_processed",
    }
    if set(state) != expected:
        raise AlphaMaxRuntimeContractError("alpha_max_capsule_state_payload_invalid")
    string_fields = (
        "portfolio_mode",
        "phase_id",
        "manifest_sha256",
        "capsule_sha256",
        "native_finalization_sha256",
    )
    count_fields = (
        "windows_processed",
        "discarded_signal_count",
        "market_event_count",
        "funding_event_count",
        "order_event_count",
        "fill_event_count",
        "trade_count",
    )
    if (
        any(type(state[field]) is not str or not state[field] for field in string_fields)
        or any(type(state[field]) is not int or state[field] < 0 for field in count_fields)
        or any(state[field] != 0 for field in count_fields[2:])
        or type(state["capsule"]) is not dict
        or type(state["finalized_children"]) is not dict
    ):
        raise AlphaMaxRuntimeContractError("alpha_max_capsule_state_payload_invalid")
    capsule = _freeze_json(state["capsule"])
    finalized = _freeze_json(state["finalized_children"])
    if not isinstance(capsule, Mapping) or not isinstance(finalized, Mapping):
        raise AlphaMaxRuntimeContractError("alpha_max_capsule_state_payload_invalid")
    return AlphaMaxIndicatorCapsule(
        portfolio_mode=state["portfolio_mode"],
        phase_id=state["phase_id"],
        manifest_sha256=state["manifest_sha256"],
        capsule_sha256=state["capsule_sha256"],
        capsule=capsule,
        finalized_children=finalized,
        native_finalization_sha256=state["native_finalization_sha256"],
        windows_processed=state["windows_processed"],
        discarded_signal_count=state["discarded_signal_count"],
        market_event_count=state["market_event_count"],
        funding_event_count=state["funding_event_count"],
        order_event_count=state["order_event_count"],
        fill_event_count=state["fill_event_count"],
        trade_count=state["trade_count"],
    )


def _alpha_max_prelock_final_row_artifacts(
    snapshot: _AlphaMaxBundleSnapshot,
    *,
    row_id: str,
) -> tuple[
    AlphaMaxManifestReceipt,
    AlphaMaxCapsuleReceipt,
    AlphaMaxIndicatorCapsule,
    float,
]:
    root = Path(snapshot.root_path)
    manifest_path = root / f"manifests/prelock_final_refit/{row_id}.json"
    manifest = _alpha_max_manifest_receipt_from_path(
        manifest_path,
        root=root,
        phase="prelock_final_refit",
    )
    manifest_payload = _strict_json_object(
        _read_alpha_max_prelock_artifact(
            snapshot,
            f"manifests/prelock_final_refit/{row_id}.json",
        )
    )
    try:
        gross = float(manifest_payload["gross_cap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError("alpha_max_final_manifest_gross_invalid") from exc
    if not math.isfinite(gross) or gross <= 0.0:
        raise AlphaMaxRuntimeContractError("alpha_max_final_manifest_gross_invalid")
    relative = f"capsules/prelock_final_refit/{row_id}/{_ALPHA_MAX_HISTORICAL_FOLD_IDS[0]}.json"
    capsule_receipt = AlphaMaxCapsuleReceipt.from_path(
        root / relative,
        row_id=row_id,
        phase="prelock_final_refit",
        prefix_id=_ALPHA_MAX_HISTORICAL_FOLD_IDS[0],
        manifest_sha256=manifest.sha256,
        relative_path=relative,
    )
    capsule = _alpha_max_capsule_from_receipt(capsule_receipt)
    if capsule.manifest_sha256 != manifest.sha256 or capsule.phase_id != "embargo":
        raise AlphaMaxRuntimeContractError("alpha_max_final_capsule_scope_invalid")
    return manifest, capsule_receipt, capsule, gross


def run_alpha_max_historical_process(
    *,
    sealed_prelock_directory: str | os.PathLike[str],
    embargo_feature_root: str,
    historical_evaluation_raw_root: str,
    historical_evaluation_feature_root: str,
    exchange: str,
    output_root: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    bootstrap_implementation_inventory: list[dict[str, object]] | None = None,
) -> AlphaMaxCommandResult:
    """Run one append-only, report-only exposed historical boundary."""
    reject_ambient_lq_environment()
    if exchange != "binance":
        raise AlphaMaxRuntimeContractError("alpha_max_exchange_invalid")
    before = _snapshot_bundle_tree(sealed_prelock_directory)
    prelock_snapshot_sha256, prelock_seal_bytes = _validate_prelock_snapshot(before)
    prelock_payload = _strict_json_object(
        _read_alpha_max_prelock_artifact(before, "run/prelock_result.json")
    )
    _validate_complete_alpha_max_prelock_matrix(before, prelock_payload)
    champion = prelock_payload.get("prelock_champion")
    if champion is not None and (type(champion) is not str or not champion):
        raise AlphaMaxRuntimeContractError("alpha_max_prelock_champion_invalid")

    failure_list: list[str] = []
    contract_seal: AlphaMaxContractManifestSeal | None = None
    try:
        retained_contract_bytes = _read_alpha_max_prelock_artifact(
            before,
            "inputs/contract_manifest.json",
        )
        contract_seal = seal_alpha_max_contract_manifest(
            Path(before.root_path) / "inputs/contract_manifest.json"
        )
        if contract_seal.canonical_bytes != retained_contract_bytes:
            raise AlphaMaxRuntimeContractError("alpha_max_contract_manifest_snapshot_mismatch")
    except (AlphaMaxRuntimeContractError, OSError, TypeError, ValueError) as exc:
        failure_list.append(_alpha_max_failure_reason("contract_manifest", exc))

    historical_roots = (
        ("embargo", "feature", embargo_feature_root),
        (
            "historical_exposed_evaluation",
            "raw",
            historical_evaluation_raw_root,
        ),
        (
            "historical_exposed_evaluation",
            "feature",
            historical_evaluation_feature_root,
        ),
    )
    if contract_seal is None:
        root_seals: dict[tuple[str, str], AlphaMaxRootSeal] = {}
        failure_list.append("root_validation:contract_manifest_required")
    else:
        root_seals, failures = _alpha_max_root_validation(
            historical_roots,
            exchange=exchange,
            availability_start_by_kind={
                "feature": contract_seal.feature_availability_start_by_symbol,
                "raw": contract_seal.raw_availability_start_by_symbol,
            },
            availability_end_by_kind={
                "feature": contract_seal.feature_availability_end_by_symbol,
                "raw": contract_seal.raw_availability_end_by_symbol,
            },
        )
        failure_list.extend(failures)
    if {
        ("embargo", "feature"),
        ("historical_exposed_evaluation", "feature"),
    } <= set(root_seals):
        try:
            _validate_alpha_max_adjacent_feature_roots(
                root_seals,
                (("embargo", "historical_exposed_evaluation"),),
            )
        except (TypeError, ValueError) as exc:
            failure_list.append(_alpha_max_failure_reason("adjacent_feature_roots", exc))
    try:
        expected_embargo = _read_alpha_max_prelock_artifact(
            before,
            "roots/feature/embargo.json",
        )
        observed_embargo = root_seals.get(("embargo", "feature"))
        if observed_embargo is None or observed_embargo.canonical_bytes != expected_embargo:
            raise AlphaMaxRuntimeContractError("alpha_max_embargo_feature_root_hash_mismatch")
    except (OSError, TypeError, ValueError) as exc:
        failure_list.append(_alpha_max_failure_reason("embargo_feature_rehash", exc))

    if failure_list:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_historical_input_invalid:" + "|".join(sorted(failure_list))
        )
    try:
        prelock_root = Path(before.root_path)
        config_path = prelock_root / "inputs/config.json"
        preflight = preflight_alpha_max_runtime_contract(config_path)
        nodes = _alpha_max_current_nodes(preflight)
        prior_bytes = _read_alpha_max_prelock_artifact(
            before,
            "inputs/prior_trial_inventory.json",
        )
        trial_ledger = build_alpha_max_trial_ledger(
            prior_bytes,
            _strict_json_object(preflight.config_bytes),
        )
        from lumina_quant.research.alpha_max_evidence import (
            validate_alpha_max_admission_artifact,
        )

        admission_bytes = _read_alpha_max_prelock_artifact(before, "admission/train.json")
        admission = validate_alpha_max_admission_artifact(admission_bytes)
        admitted_symbols = _validate_admitted_symbols(
            preflight,
            admission.admitted_symbols,
        )
        admission_computation_bytes = _read_alpha_max_prelock_artifact(
            before,
            "admission/train_computation.json",
        )
        admission_computation_payload = _strict_json_object(admission_computation_bytes)
        train_liquidity_buckets = validate_alpha_max_train_liquidity_buckets(
            _read_alpha_max_prelock_artifact(
                before,
                "admission/train_liquidity_buckets.json",
            )
        )
        if (
            admission_computation_payload.get("artifact_kind")
            != "alpha_max_train_admission_computation.v1"
            or admission_computation_payload.get("admission_artifact_sha256") != admission.sha256
            or train_liquidity_buckets.admitted_symbols != admitted_symbols
            or train_liquidity_buckets.admission_computation_sha256
            != _sha256(admission_computation_bytes)
        ):
            raise AlphaMaxRuntimeContractError(
                "alpha_max_train_liquidity_bucket_prelock_binding_mismatch"
            )
        prelock_selection = _alpha_max_selection_from_bytes(
            _read_alpha_max_prelock_artifact(before, "selection/prelock.json"),
            role="prelock_selection",
        )
        prelock_seal_payload = _strict_json_object(prelock_seal_bytes)
        if (
            prelock_selection.prelock_champion != champion
            or prelock_selection.selected_candidate_id != champion
            or prelock_seal_payload.get("prelock_champion") != champion
            or prelock_seal_payload.get("selected_candidate_id") != champion
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_selection_identity_mismatch")
        retained: dict[
            str,
            tuple[
                AlphaMaxManifestReceipt,
                AlphaMaxCapsuleReceipt,
                AlphaMaxIndicatorCapsule,
                float,
            ],
        ] = {
            row_id: _alpha_max_prelock_final_row_artifacts(before, row_id=row_id)
            for row_id in _ALPHA_MAX_RESOLVABLE_ROWS
        }
        for row_id in _ALPHA_MAX_RESOLVABLE_ROWS:
            manifest = retained[row_id][0]
            activation = seal_alpha_max_manifest_activation(
                preflight,
                output_root=prelock_root,
                phase="prelock_final_refit",
                manifest_path=manifest.path,
                admitted_symbols=admitted_symbols,
            )
            if (
                activation.manifest_receipt.sha256 != manifest.sha256
                or activation.manifest_receipt.byte_count != manifest.byte_count
            ):
                raise AlphaMaxRuntimeContractError("alpha_max_final_manifest_activation_mismatch")
        _AlphaMaxBoundedRawLoader(
            root_seals[("historical_exposed_evaluation", "raw")],
            admitted_symbols,
        )
        _validated_output_target(output_root)
    except (OSError, TypeError, ValueError) as exc:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_historical_input_invalid:"
            + _alpha_max_failure_reason("prelock_execution_inputs", exc)
        ) from exc

    if contract_seal is None:
        raise AlphaMaxRuntimeContractError(
            "alpha_max_historical_input_invalid:contract_manifest_required"
        )
    implementation_inventory = _verify_alpha_max_checkpoint_implementation_inventory(
        _alpha_max_checkpoint_implementation_inventory()
        if bootstrap_implementation_inventory is None
        else bootstrap_implementation_inventory
    )
    attempt_descriptor = _alpha_max_historical_checkpoint_descriptor(
        preflight=preflight,
        contract_seal=contract_seal,
        root_seals=root_seals,
        admitted_symbols=admitted_symbols,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        implementation_inventory=implementation_inventory,
        prelock_seal_bytes=prelock_seal_bytes,
        prelock_snapshot_sha256=prelock_snapshot_sha256,
    )
    checkpoint_store = _AlphaMaxCellCheckpointStore(
        checkpoint_root,
        output_root=output_root,
        descriptor=attempt_descriptor,
        config_bytes=preflight.config_bytes,
    )
    completion_id = "historical_exposed_evaluation"
    _claim = _acquire_historical_completion_claim(
        checkpoint_store.output_root.parent,
        completion_id=completion_id,
        prelock_seal_sha256=_sha256(prelock_seal_bytes),
        attempt_descriptor_sha256=checkpoint_store.descriptor_sha256,
        output_root=checkpoint_store.display_output_root,
    )
    root: Path | None = None
    try:
        root = _alpha_max_create_or_resume_run_root(
            checkpoint_store.output_root,
            config_bytes=preflight.config_bytes,
            attempt_descriptor_sha256=checkpoint_store.descriptor_sha256,
            sealed_role="historical",
        )
        root = checkpoint_store.bind_output_root()
        prepared: dict[str, _AlphaMaxPreparedReplayRow] = {}
        for row_id in _ALPHA_MAX_RESOLVABLE_ROWS:
            manifest, first_receipt, initial_capsule, gross = retained[row_id]
            checkpoint_bytes = checkpoint_store.load_precompute(
                unit_kind="historical_row",
                unit_id=row_id,
            )
            if checkpoint_bytes is None:
                current = _AlphaMaxPreparedReplayRow(
                    manifest_receipt=manifest,
                    fold_inputs=_alpha_max_build_fold_inputs(
                        preflight,
                        manifest_output_root=prelock_root,
                        capsule_output_root=root,
                        phase="prelock_final_refit",
                        manifest_receipt=manifest,
                        admitted_symbols=admitted_symbols,
                        root_seals=root_seals,
                        domain="historical_exposed_evaluation",
                        initial_capsule=initial_capsule,
                        initial_receipt=first_receipt,
                    ),
                    gross=gross,
                )
                checkpoint_bytes = checkpoint_store.seal_precompute(
                    unit_kind="historical_row",
                    unit_id=row_id,
                    data_bytes=_alpha_max_prepared_row_checkpoint_bytes(
                        current,
                        domain="historical_exposed_evaluation",
                    ),
                )
            prepared[row_id] = _alpha_max_restore_prepared_row_checkpoint(
                checkpoint_bytes,
                preflight=preflight,
                manifest=manifest,
                admitted_symbols=admitted_symbols,
                root_seals=root_seals,
                domain="historical_exposed_evaluation",
                gross=gross,
                capsule_output_root=root,
                initial_receipt=first_receipt,
            )
        historical_matrix = _alpha_max_complete_domain_matrix(
            preflight,
            output_root=prelock_root,
            phase="prelock_final_refit",
            nodes=nodes,
            admitted_symbols=admitted_symbols,
            domain="historical_exposed_evaluation",
            trial_ledger=trial_ledger,
            prepared_rows=prepared,
            checkpoint_store=checkpoint_store,
        )
        historical_trend_liquidity_falsifier = _alpha_max_trend_liquidity_falsifier_artifact(
            historical_matrix,
            train_liquidity_buckets,
        )
        historical_ranking = rank_alpha_max_historical_report(historical_matrix.rows)
        champion_cell = None if champion is None else historical_matrix.cells[(champion, 30)]
        terminal = build_alpha_max_terminal_state(
            prelock_selection=prelock_selection,
            champion_historical_nominal_30_cell=champion_cell,
            historical_ranking=historical_ranking,
            incumbent_comparison_status="unavailable",
        )
        report_payload = {
            "artifact_kind": "alpha_max_historical_process_result.v1",
            "confirmation_status": terminal.confirmation_status,
            "engine_cell_count": 68,
            "failure_reasons": [],
            "historical_evaluation_leader": terminal.historical_evaluation_leader,
            "historical_exposure_status": terminal.historical_exposure_status,
            "physical_fold_run_count": historical_matrix.physical_fold_run_count,
            "prelock_champion": champion,
            "requires_fresh_confirmation": terminal.requires_fresh_confirmation,
            "selected_candidate_id": champion,
            "status": "complete_report_only",
            "terminal_outcome": terminal.terminal_outcome,
        }
        generated_artifacts = {
            **_alpha_max_root_artifacts(root_seals),
            **_alpha_max_matrix_artifacts(historical_matrix),
            "admission/train_liquidity_buckets.json": (train_liquidity_buckets.canonical_bytes),
            "binding/prelock_seal.json": prelock_seal_bytes,
            "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json": (
                historical_trend_liquidity_falsifier
            ),
            "report/historical_result.json": _canonical_bytes(report_payload) + b"\n",
            "selection/historical_ranking.json": historical_ranking.canonical_bytes,
            "terminal/historical.json": _canonical_bytes(terminal.to_payload()) + b"\n",
        }
        activation_paths = _alpha_max_historical_activation_paths()
        existing_artifacts = _alpha_max_collect_existing_artifacts(
            root,
            allowed_paths=activation_paths | set(generated_artifacts),
            required_paths=activation_paths,
        )
        for relative, expected_bytes in generated_artifacts.items():
            observed = existing_artifacts.get(relative)
            if observed is not None and observed != expected_bytes:
                raise AlphaMaxRuntimeContractError(
                    "alpha_max_resumable_generated_artifact_mismatch"
                )
        artifacts = {
            **{relative: existing_artifacts[relative] for relative in sorted(activation_paths)},
            **generated_artifacts,
        }
        inventory = [
            {
                "byte_count": len(payload),
                "relative_path": relative,
                "sha256": _sha256(payload),
            }
            for relative, payload in sorted(artifacts.items())
        ]
        seal_bytes = (
            _canonical_bytes(
                {
                    "artifact_kind": "alpha_max_append_only_historical_package.v1",
                    "completion_id": completion_id,
                    "historical_artifacts": inventory,
                    "immutable": True,
                    "prelock_seal_sha256": _sha256(prelock_seal_bytes),
                    "prelock_snapshot_sha256": prelock_snapshot_sha256,
                }
            )
            + b"\n"
        )
        after = _snapshot_bundle_tree(sealed_prelock_directory)
        if after != before:
            raise AlphaMaxRuntimeContractError("alpha_max_prelock_mutated_during_historical_run")
        _verify_alpha_max_checkpoint_implementation_inventory(implementation_inventory)
        expected_schedule = _alpha_max_physical_fold_schedule("historical_exposed_evaluation")
        matrix_status = _strict_json_object(historical_matrix.status_payload)
        if (
            historical_matrix.domain != "historical_exposed_evaluation"
            or len(historical_matrix.cells) != 68
            or historical_matrix.physical_fold_run_count != len(expected_schedule)
            or checkpoint_store._physical_schedule_sha256
            != attempt_descriptor["physical_schedule_sha256"]
            or matrix_status
            != {
                "artifact_kind": "alpha_max_matrix_statuses.v1",
                "domain": "historical_exposed_evaluation",
                "engine_cell_count": 68,
                "physical_fold_run_count": len(expected_schedule),
                "status_count": 84,
                "statuses": matrix_status.get("statuses"),
            }
            or type(matrix_status.get("statuses")) is not list
            or len(matrix_status["statuses"]) != 84
            or set(historical_matrix.cells)
            != {
                (row_id, nominal)
                for row_id in _ALPHA_MAX_RESOLVABLE_ROWS
                for nominal in ALPHA_MAX_COST_CELL_BPS
            }
        ):
            raise AlphaMaxRuntimeContractError("alpha_max_historical_matrix_finalization_invalid")
        prospective_bundle = AlphaMaxSealedBundle(
            output_root=str(root),
            stable_paths=tuple(
                str(root / _safe_bundle_relative_path(relative)) for relative in sorted(artifacts)
            ),
            seal_path=str(root / "SEALED.json"),
            seal_sha256=_sha256(seal_bytes),
        )
        display_bundle = _alpha_max_display_bundle(
            prospective_bundle,
            anchored_root=root,
            display_root=checkpoint_store.display_output_root,
        )
        command_result = AlphaMaxCommandResult(
            exit_code=0,
            terminal_outcome=terminal.terminal_outcome,
            bundle=display_bundle,
            failure_reasons=(),
        )
        _finalize_alpha_max_run_owned_root(root, artifacts, seal_bytes=seal_bytes)
        return command_result
    except Exception:
        raise
