"""Fail-closed runtime-contract foundation for the alpha-max experiment.

This module intentionally stops before constructing or running ``Backtest``.  It
owns only the descriptor-bound frozen-config preflight, immutable uppercase
runtime surface, deterministic runtime-read audit, and pure constructor plans
needed by a later orchestration change.

The experiment config is a sealed Revision 5.14 artifact.  No profile, ambient
``LQ_*`` value, default runtime config, YAML file, or merge layer participates in
construction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Final

from lumina_quant.utils.artifact_read_receipt import (
    ArtifactReadReceipt,
    read_artifact_bytes,
)

__all__ = [
    "ALPHA_MAX_CANDIDATE_SYMBOLS",
    "ALPHA_MAX_CONFIG_FILE_SHA256",
    "ALPHA_MAX_CONFIG_PAYLOAD_SHA256",
    "ALPHA_MAX_COST_CELL_BPS",
    "ALPHA_MAX_RUNTIME_CONTRACT_SHA256",
    "AlphaMaxBacktestConfig",
    "AlphaMaxCostCell",
    "AlphaMaxEngineConstructorPlan",
    "AlphaMaxPhaseWindow",
    "AlphaMaxRuntimeContractError",
    "AlphaMaxRuntimePreflight",
    "AmbientLQEnvironmentError",
    "FrozenRuntimeMutationError",
    "UnfrozenRuntimeFieldError",
    "alpha_max_common_rng_seed",
    "alpha_max_common_rng_seed_payload",
    "build_alpha_max_backtest_config",
    "build_alpha_max_cost_cell_configs",
    "build_alpha_max_engine_constructor_plan",
    "preflight_alpha_max_runtime_contract",
    "reject_ambient_lq_environment",
    "validate_alpha_max_cost_cell_config_matrix",
]


ALPHA_MAX_RUNTIME_CONTRACT_SHA256: Final[str] = (
    "b3859443c842cf8b04d04ed32923e6c6a8207af18e26f68a717ba623b4edfef9"
)
ALPHA_MAX_CONFIG_PAYLOAD_SHA256: Final[str] = (
    "b53c2274624fe4bc017ead59975efc805d166038f841773337bb48d55ee9692d"
)
ALPHA_MAX_CONFIG_CANONICAL_SHA256: Final[str] = (
    "85ab64360d77265441d2eeaaa7a41a4df12589667bccdfec75b62572bfcf5e62"
)
ALPHA_MAX_CONFIG_FILE_SHA256: Final[str] = (
    "34f1ea894b0af984d4f76348f52fbca09fab45b9e3d5d963f257ec9d128ee356"
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


class AlphaMaxRuntimeContractError(ValueError):
    """The sealed runtime contract or one of its construction inputs is invalid."""


class AmbientLQEnvironmentError(AlphaMaxRuntimeContractError):
    """An ambient ``LQ_*`` environment key would make the replay non-hermetic."""


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


def reject_ambient_lq_environment() -> None:
    """Reject every ambient key beginning with ``LQ_`` without reading its value."""
    offending = tuple(sorted(key for key in os.environ if key.startswith("LQ_")))
    if offending:
        joined = ",".join(offending)
        raise AmbientLQEnvironmentError(f"ambient_lq_environment:{joined}")


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


def _validate_runtime_contract(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != _TOP_LEVEL_KEYS:
        raise AlphaMaxRuntimeContractError("alpha_max_config_top_level_schema_mismatch")
    _require_exact(
        config.get("schema_version"), "alpha_max_portfolio_experiment.v1", field="schema"
    )
    _require_exact(config.get("experiment_id"), "alpha_max_portfolio_20260710", field="experiment")
    _require_exact(config.get("revision"), "5.14", field="revision")
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
    if receipt.requested_path != receipt.canonical_path:
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


def alpha_max_common_rng_seed_payload(split_or_fold_id: str, nominal_cost_bps: int) -> bytes:
    """Build the exact Revision 5.14 common-random-number seed payload."""
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


def _validate_preflight(preflight: AlphaMaxRuntimePreflight) -> None:
    if type(preflight) is not AlphaMaxRuntimePreflight:
        raise TypeError("alpha_max_runtime_preflight_required")
    if (
        preflight.runtime_contract_sha256 != ALPHA_MAX_RUNTIME_CONTRACT_SHA256
        or preflight.attribute_allowlist != _EXPECTED_ATTRIBUTE_ALLOWLIST
        or preflight.common_runtime_bytes != _canonical_bytes(_EXPECTED_STATIC_ATTRIBUTES)
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
        portfolio_kwargs=MappingProxyType(
            {
                "fill_application_attribution_sink": fill_application_attribution_sink,
                "funding_boundary_resolver": funding_boundary_resolver,
            }
        ),
        execution_handler_kwargs=MappingProxyType({"record_cost_attribution": True}),
    )
