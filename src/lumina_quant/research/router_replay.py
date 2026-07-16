"""Read-only, fail-closed verifier for the frozen router replay evidence bundle.

The source freeze is authoritative router input.  The manifest is only a claim about
that freeze; the commit receipt, supplied through an out-of-band trusted digest,
binds every file and canonical receipt artifact consumed below.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from lumina_quant.data.symbol_lifecycle import (
    load_symbol_lifecycle_registry,
    validate_fold_membership_manifest,
)
from lumina_quant.strategies.registry import resolve_strategy_class

SCHEMA = "router_replay_v2"
SOURCE_SCHEMA = "router_source_v2"
COMMIT_SCHEMA = "router_replay_commit_v1"
PROFILE_HANDLER = (
    "scripts.research.run_alpha_zoo_69_asset_profile_optuna_hybrid_refit._candidate_from_params"
)
CANDIDATE_IDS = (
    "codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled",
    "codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2",
)
CANDIDATE_IDS_SHA256 = "ddc8996136e70d3847e8270f6165a26992ec8def8439ba6f56e3bcdbdee239b9"
BALANCED_LABEL = "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
GROWTH_LABEL = "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna"
PROFILE_SHA256 = "a3e9572365d39e3388c97b8b6c094c0bb9d63a3b1fd6d38c918342b435716950"
ROUTER_SOURCE_PREDICATE = (
    "scripts.research.run_alpha_zoo_69_asset_monthly_refit_walkforward._lagged_shadow_leaf_source"
)
RUNNER_SOURCE_PATH = "lumina_quant.strategy_factory.research_runner"
SCALE_GRID_PPM = (
    500_000,
    750_000,
    1_000_000,
    1_100_000,
    1_250_000,
    1_400_000,
    1_500_000,
    1_750_000,
    2_000_000,
    2_250_000,
    2_500_000,
    2_750_000,
    3_000_000,
)
_HASH = set("0123456789abcdef")
_BASE_SIGNAL_PPM = (-1_000_000, 1_000_000)
_BASE_RETURN_PPM = (-1_000_000, 100_000_000)
_SCALED_POSITION_PPM = (-3_000_000, 3_000_000)
_SCALED_RETURN_PPM = (-3_000_000, 300_000_000)
_INPUT_ERRORS = (
    OSError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    yaml.YAMLError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    ImportError,
    RecursionError,
    OverflowError,
)


@dataclass(frozen=True, slots=True)
class _EligibilityReturns:
    size: int


@dataclass(frozen=True, slots=True)
class _EligibilityCandidate:
    """The only candidate surface consumed by the frozen source predicate."""

    candidate_label: str
    family: str
    returns: _EligibilityReturns
    row: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RouterReplayReport:
    status: str
    reasons: tuple[str, ...]
    candidate_ids: tuple[str, str]
    fold_count: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HASH:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON value: {value}")


def _finite(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        if any(type(key) is not str or not key for key in value):
            raise ValueError(f"{name} keys are invalid")
        for key, item in value.items():
            _finite(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{name}[{index}]")
    elif value is not None and not isinstance(value, (str, bool)):
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} type is invalid")
        try:
            finite = math.isfinite(float(value))
        except OverflowError, ValueError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must be finite")


def _canonical_json(path: str | Path) -> tuple[Any, str]:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    _finite(value, "JSON")
    if raw != _canonical_bytes(value):
        raise ValueError("artifact JSON is not canonical")
    return value, _raw_sha(raw)


def _exact(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} keys are invalid")
    return value


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} is invalid")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z") or "+" in value:
        raise ValueError(f"{name} must be canonical UTC Z")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} invalid") from exc
    if result.tzinfo != UTC or result.isoformat().replace("+00:00", "Z") != value:
        raise ValueError(f"{name} must be canonical UTC Z")
    return result


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def _profile(path: str | Path) -> None:
    raw = Path(path).read_bytes()
    if _raw_sha(raw) != PROFILE_SHA256:
        raise ValueError("combined profile bytes drift")
    root = yaml.load(raw.decode("utf-8"), Loader=_UniqueSafeLoader)
    _finite(root, "profile")
    if not isinstance(root, Mapping) or root.get("profile") != "backtest_cost_realistic":
        raise ValueError("combined profile identity is invalid")
    for name in ("research", "execution", "risk", "live", "data"):
        if not isinstance(root.get(name), Mapping):
            raise ValueError("combined profile sections are invalid")
    research, execution, risk, live, data = (
        root[name] for name in ("research", "execution", "risk", "live", "data")
    )
    required_bools = (
        (research, "strict_selection_gate", True),
        (research, "use_lockbox_split", True),
        (research, "single_correlation_discount", True),
        (research, "hac_inference", True),
        (research, "cscv_pbo", True),
        (research, "exposure_normalized_promotion", True),
        (research, "route_unmapped_registered_strategies", True),
        (research, "require_actual_engine_routing", True),
        (execution, "require_funding_coverage", True),
        (execution, "funding_on_utc_boundary", True),
        (risk, "attach_default_protective_stop", True),
        (risk, "enforce_order_risk_gate_in_backtest", True),
        (live, "testnet", True),
        (live, "require_real_enable_flag", True),
        (live, "allow_market_orders", False),
        (live, "shadow_live_enabled", False),
    )
    if any(section.get(key) is not expected for section, key, expected in required_bools):
        raise ValueError("profile boolean contract is invalid")
    if type(research.get("purge_embargo_bars")) is not int or research["purge_embargo_bars"] != 1:
        raise ValueError("profile purge embargo is invalid")
    if (
        type(execution.get("funding_interval_hours")) is not int
        or execution["funding_interval_hours"] != 8
    ):
        raise ValueError("profile funding interval is invalid")
    if execution.get("slippage_impact_model") != "sqrt_impact" or live.get("mode") != "paper":
        raise ValueError("profile safety contract is invalid")
    coefficient = execution.get("slippage_impact_coefficient")
    if type(coefficient) is not float or not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("profile coefficient is invalid")
    kinds = data.get("kinds")
    if (
        not isinstance(kinds, list)
        or any(type(item) is not str for item in kinds)
        or len(set(kinds)) != len(kinds)
        or "funding" not in kinds
    ):
        raise ValueError("profile funding data is invalid")


def _object_source(qualified: str) -> Path:
    if not isinstance(qualified, str) or "." not in qualified:
        raise ValueError("qualified object is invalid")
    parts = qualified.split(".")
    obj: Any | None = None
    for boundary in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:boundary])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise
            continue
        for attribute in parts[boundary:]:
            obj = getattr(obj, attribute)
        break
    if obj is None:
        raise ValueError("qualified object is unavailable")
    source = inspect.getsourcefile(obj)
    if source is None:
        raise ValueError("qualified object source unavailable")
    return Path(source)


def _source_eligible(value: Any, name: str) -> None:
    row = _exact(value, {"candidate_label", "family", "return_count", "row"}, name)
    if (
        type(row["candidate_label"]) is not str
        or not row["candidate_label"]
        or type(row["family"]) is not str
        or not row["family"]
        or not isinstance(row["row"], Mapping)
    ):
        raise ValueError(f"{name} identity is invalid")
    count = _integer(row["return_count"], f"{name} return count", 0, 100_000_000)
    predicate = importlib.import_module(
        "scripts.research.run_alpha_zoo_69_asset_monthly_refit_walkforward"
    )._lagged_shadow_leaf_source
    candidate = _EligibilityCandidate(
        row["candidate_label"], row["family"], _EligibilityReturns(count), dict(row["row"])
    )
    if not predicate(candidate):
        raise ValueError(f"{name} is ineligible under frozen source predicate")


class _Artifacts:
    """Canonical externally-produced artifacts keyed by their raw-byte digest.

    A digest may legitimately be referenced by multiple leaves/candidates.  `used`
    records attribution, not destructive consumption; commit-index exhaustiveness is
    checked after all folds have been replayed.
    """

    def __init__(self, index: Mapping[str, str], paths: Mapping[str, str | Path]) -> None:
        if set(index) != set(paths):
            raise ValueError("artifact paths must exactly match the commit index")
        self.index = dict(index)
        self.paths = paths
        self.used: set[str] = set()

    def get(self, digest: Any, kind: str, schema: str) -> Mapping[str, Any]:
        key = _digest(digest, "artifact digest")
        if self.index.get(key) != kind:
            raise ValueError("artifact kind is missing or wrong")
        value, actual = _canonical_json(self.paths[key])
        if actual != key:
            raise ValueError("artifact bytes do not match digest")
        if not isinstance(value, Mapping) or value.get("schema") != schema:
            raise ValueError("artifact payload schema is invalid")
        self.used.add(key)
        return value


def _engine_dependency(
    digest: Any, artifacts: _Artifacts, mode: str, handler: str, strategy_class: str
) -> None:
    row = artifacts.get(digest, "engine_dependency", "router_engine_dependency_v2")
    row = _exact(
        row,
        {"schema", "evaluation_mode", "engine_handler", "strategy_class", "components"},
        "engine dependency",
    )
    if (
        row["evaluation_mode"] != mode
        or row["engine_handler"] != handler
        or row["strategy_class"] != strategy_class
    ):
        raise ValueError("engine dependency identity drift")
    handler_roles = {
        "entrypoint": PROFILE_HANDLER,
        "feature_cache": "scripts.research.run_alpha_zoo_69_asset_profile_optuna_hybrid_refit.FeatureCache",
        "signal": "scripts.research.run_alpha_zoo_69_asset_optuna_hybrid_refit._debounced_state_signal",
        "simulator": "scripts.research.run_alpha_zoo_69_asset_optuna_hybrid_refit.simulate_symbol",
        "candidate_builder": "scripts.research.run_alpha_zoo_69_asset_optuna_hybrid_refit._candidate_base",
        "finalizer": "scripts.research.run_alpha_zoo_69_asset_optuna_hybrid_refit.finalize_candidate",
    }
    registry_roles = {
        "entrypoint": "lumina_quant.strategy_factory.research_runner._strict_registry_simulator_router",
        "bar_store": "lumina_quant.strategy_factory.research_runner._AlignedStrategyBarStore",
        "simulator": "lumina_quant.strategy_factory.research_runner._simulate_event_driven_strategy_exposures",
        "dispatch_validator": "lumina_quant.strategy_factory.strategy_signal_dispatch.StrategySignalDispatcher._validate_actual_engine_outputs",
    }
    handler_roles["source_eligibility"] = ROUTER_SOURCE_PREDICATE
    registry_roles["source_eligibility"] = ROUTER_SOURCE_PREDICATE
    if mode == "handler":
        expected = handler_roles
        if handler != PROFILE_HANDLER:
            raise ValueError("handler is not the frozen profile handler")
    elif mode == "registry_simulator":
        if handler != "registry_simulator":
            raise ValueError("registry simulator identifier is invalid")
        klass = resolve_strategy_class(strategy_class, strict=True)
        if inspect.getsourcefile(klass) is None:
            raise ValueError("registry strategy source unavailable")
        expected = {**registry_roles, "strategy_class": f"{klass.__module__}.{klass.__qualname__}"}
    else:
        raise ValueError("leaf evaluation mode is invalid")
    components = row["components"]
    if not isinstance(components, list) or len(components) != len(expected):
        raise ValueError("engine dependency component count is invalid")
    found: set[str] = set()
    for component in components:
        component = _exact(
            component, {"role", "qualified_name", "source_sha256"}, "engine component"
        )
        role = component["role"]
        if role not in expected or role in found or component["qualified_name"] != expected[role]:
            raise ValueError("engine dependency role/name drift")
        if _raw_sha(_object_source(component["qualified_name"]).read_bytes()) != _digest(
            component["source_sha256"], "component source"
        ):
            raise ValueError("engine dependency source drift")
        found.add(role)
    if found != set(expected):
        raise ValueError("engine dependency closure drift")


def _leaf(value: Any, eligible_symbols: set[str], artifacts: _Artifacts) -> str:
    row = _exact(
        value,
        {
            "leaf_id",
            "profile_id",
            "strategy_class",
            "engine_handler",
            "params",
            "params_sha256",
            "traded_symbols",
            "dependency_symbols",
            "native_timeframe",
            "allocation_fraction_ppm",
            "source_weight_ppm",
            "native_gross_ppm",
            "evaluation_mode",
            "engine_source_sha256",
            "engine_dependency_receipt_sha256",
            "source_row_sha256",
        },
        "leaf",
    )
    for name in ("leaf_id", "profile_id", "strategy_class", "engine_handler", "native_timeframe"):
        if not isinstance(row[name], str) or not row[name] or row[name] != row[name].strip():
            raise ValueError("leaf identity is invalid")
    if not isinstance(row["params"], Mapping) or _sha(row["params"]) != _digest(
        row["params_sha256"], "params"
    ):
        raise ValueError("leaf params drift")
    timeframe = row["params"].get("timeframe")
    if not isinstance(timeframe, str) or timeframe != row["native_timeframe"]:
        raise ValueError("leaf timeframe drift")
    for name in ("traded_symbols", "dependency_symbols"):
        symbols = row[name]
        if (
            not isinstance(symbols, list)
            or not symbols
            or any(type(x) is not str or not x for x in symbols)
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError("leaf symbols are invalid")
    if not set(row["traded_symbols"]).issubset(row["dependency_symbols"]) or not set(
        row["dependency_symbols"]
    ).issubset(eligible_symbols):
        raise ValueError("leaf dependencies are invalid")
    for name in ("allocation_fraction_ppm", "source_weight_ppm", "native_gross_ppm"):
        _integer(row[name], name, 1, 100_000_000)
    mode = row["evaluation_mode"]
    if mode == "handler":
        source = _object_source(row["engine_handler"])
    elif mode == "registry_simulator":
        if row["engine_handler"] != "registry_simulator":
            raise ValueError("registry engine handler is invalid")
        source = Path(
            inspect.getsourcefile(resolve_strategy_class(row["strategy_class"], strict=True)) or ""
        )
        if not source:
            raise ValueError("registry strategy source unavailable")
    else:
        raise ValueError("leaf evaluation mode is invalid")
    if _raw_sha(source.read_bytes()) != _digest(row["engine_source_sha256"], "engine source"):
        raise ValueError("engine source drift")
    _engine_dependency(
        row["engine_dependency_receipt_sha256"],
        artifacts,
        mode,
        row["engine_handler"],
        row["strategy_class"],
    )
    if _sha({key: item for key, item in row.items() if key != "source_row_sha256"}) != _digest(
        row["source_row_sha256"], "source row"
    ):
        raise ValueError("source row drift")
    return row["leaf_id"]


def _period_metrics(returns_ppm: list[int], scale_ppm: int) -> tuple[Fraction, Fraction]:
    equity = Fraction(1)
    cumulative: list[Fraction] = []
    for value in returns_ppm:
        scaled = Fraction(
            _integer(value, "period return", *_SCALED_RETURN_PPM) * scale_ppm, 1_000_000**2
        )
        equity *= 1 + scaled
        cumulative.append(equity)
    if not cumulative:
        raise ValueError("period returns are empty")
    peak = Fraction(1)
    mdd = Fraction(0)
    for value in cumulative:
        peak = max(peak, value)
        mdd = max(mdd, (peak - value) / max(peak, Fraction(1, 1_000_000_000_000)))
    return equity - 1, mdd


def _validation_score(candidate: Mapping[str, Any]) -> Fraction:
    validation_return = Fraction(candidate["validation_return_ppm"], 1_000_000)
    validation_mdd = Fraction(candidate["validation_mdd_ppm"], 1_000_000)
    train_return = Fraction(candidate["train_return_ppm"], 1_000_000)
    train_mdd = Fraction(candidate["train_mdd_ppm"], 1_000_000)
    return (
        validation_return / max(validation_mdd, Fraction(2, 100))
        + Fraction(3, 100) * min(train_return, Fraction(150, 100))
        - Fraction(10, 100) * train_mdd
    )


def _receipt_identity(
    row: Mapping[str, Any],
    fold: Mapping[str, Any],
    leaf: Mapping[str, Any],
    data: Mapping[str, Any],
    window: Mapping[str, Any],
    scale: int,
) -> None:
    identity = {
        "fold_id": fold["fold_id"],
        "locked_oos": fold["locked_oos"],
        "membership_fold_sha256": fold["membership_fold_sha256"],
        "leaf_id": leaf["leaf_id"],
        "source_row_sha256": leaf["source_row_sha256"],
        "params_sha256": leaf["params_sha256"],
        "dependency_symbols": leaf["dependency_symbols"],
        "evaluation_mode": leaf["evaluation_mode"],
        "engine_handler": leaf["engine_handler"],
        "strategy_class": leaf["strategy_class"],
        "native_timeframe": leaf["native_timeframe"],
        "engine_dependency_receipt_sha256": leaf["engine_dependency_receipt_sha256"],
        "data_receipt_sha256": _sha(data),
        "window_receipt_sha256": _sha(window),
        "applied_scale_ppm": scale,
        "generic_fallback_proxy_count": 0,
        "current_fold_oos_input_count": 0,
    }
    if (
        type(row["applied_scale_ppm"]) is not int
        or type(row["generic_fallback_proxy_count"]) is not int
        or type(row["current_fold_oos_input_count"]) is not int
    ):
        raise ValueError("receipt integer controls are invalid")
    if any(row[name] != expected for name, expected in identity.items()):
        raise ValueError("receipt identity drift")


def _strings(value: Any, name: str, *, nonempty: bool = True) -> list[str]:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(type(item) is not str or not item or item != item.strip() for item in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _data_source(digest: Any, artifacts: _Artifacts, data: Mapping[str, Any]) -> None:
    row = artifacts.get(digest, "data_source_receipt", "router_data_source_receipt_v2")
    row = _exact(
        row,
        {
            "schema",
            "dataset_id",
            "content_sha256",
            "symbols",
            "start_utc",
            "end_utc",
            "input_cutoff_utc",
            "native_timeframe",
            "source_artifact_sha256",
            "tape_sha256",
        },
        "data source receipt",
    )
    if row != {key: data[key] for key in row if key != "schema"} | {"schema": row["schema"]}:
        raise ValueError("data source receipt binding drift")
    for name in ("content_sha256", "source_artifact_sha256", "tape_sha256"):
        _digest(row[name], f"data source {name}")


def _data_window(
    digest: Any,
    schema: str,
    artifacts: _Artifacts,
    fold: Mapping[str, Any],
    symbols: list[str] | None = None,
    timeframe: str | None = None,
    role: str | None = None,
) -> Mapping[str, Any]:
    kind = "data_receipt" if schema == "router_data_receipt_v2" else "window_receipt"
    row = artifacts.get(digest, kind, schema)
    if schema == "router_data_receipt_v2":
        row = _exact(
            row,
            {
                "schema",
                "dataset_id",
                "content_sha256",
                "symbols",
                "start_utc",
                "end_utc",
                "input_cutoff_utc",
                "native_timeframe",
                "source_artifact_sha256",
                "strict_receipt_sha256",
                "tape_sha256",
            },
            "data receipt",
        )
        if (
            type(row["dataset_id"]) is not str
            or not row["dataset_id"]
            or row["dataset_id"] != row["dataset_id"].strip()
        ):
            raise ValueError("dataset identity is invalid")
        _strings(row["symbols"], "data symbols")
        if len(set(row["symbols"])) != len(row["symbols"]):
            raise ValueError("data symbol order is not unique")
        if symbols is not None and row["symbols"] != symbols:
            raise ValueError("data symbol coverage drift")
        if timeframe is not None and row["native_timeframe"] != timeframe:
            raise ValueError("data timeframe drift")
        start, end, cutoff = (
            _timestamp(row[name], f"data {name}")
            for name in ("start_utc", "end_utc", "input_cutoff_utc")
        )
        if not start < end or cutoff != _timestamp(fold["input_cutoff_utc"], "fold cutoff"):
            raise ValueError("data range/cutoff drift")
        for name in (
            "content_sha256",
            "source_artifact_sha256",
            "strict_receipt_sha256",
            "tape_sha256",
        ):
            _digest(row[name], f"data {name}")
        _data_source(row["strict_receipt_sha256"], artifacts, row)
        return row
    row = _exact(
        row,
        {
            "schema",
            "data_receipt_sha256",
            "fold_id",
            "role",
            "start_utc",
            "end_utc",
            "input_cutoff_utc",
            "native_timeframe",
            "membership_fold_sha256",
            "output_timestamps_utc",
        },
        "window receipt",
    )
    data = _data_window(
        row["data_receipt_sha256"], "router_data_receipt_v2", artifacts, fold, symbols, timeframe
    )
    if (
        row["fold_id"] != fold["fold_id"]
        or row["membership_fold_sha256"] != fold["membership_fold_sha256"]
        or row["input_cutoff_utc"] != fold["input_cutoff_utc"]
        or row["native_timeframe"] != data["native_timeframe"]
        or (role is not None and row["role"] != role)
    ):
        raise ValueError("window fold binding drift")
    if row["role"] not in {"train", "validation", "history", "locked_oos"}:
        raise ValueError("window role is invalid")
    start, end, cutoff = (
        _timestamp(row[name], f"window {name}")
        for name in ("start_utc", "end_utc", "input_cutoff_utc")
    )
    if not start < end:
        raise ValueError("window range drift")
    if start < _timestamp(data["start_utc"], "data start") or end > _timestamp(
        data["end_utc"], "data end"
    ):
        raise ValueError("window exceeds data coverage")
    timestamps = _strings(row["output_timestamps_utc"], "window output timestamps")
    if (
        timestamps != sorted(timestamps)
        or len(set(timestamps)) != len(timestamps)
        or any(
            not start <= _timestamp(stamp, "window output timestamp") < end for stamp in timestamps
        )
    ):
        raise ValueError("window output sequence drift")
    if row["role"] == "locked_oos":
        if {"start_utc": row["start_utc"], "end_utc": row["end_utc"]} != fold[
            "locked_oos"
        ] or cutoff >= start:
            raise ValueError("locked-OOS window drift")
    elif end > cutoff:
        raise ValueError("decision-input window exceeds cutoff")
    return row


def _payload(rows: Any, digest: Any, name: str, keys: set[str]) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} rows are invalid")
    for item in rows:
        item = _exact(item, keys, f"{name} row")
        _timestamp(item["timestamp_utc"], f"{name} timestamp")
        for key, value in item.items():
            if key == "timestamp_utc":
                continue
            if key == "event_index":
                _integer(value, f"{name} {key}", 0, 100_000_000)
            elif key.endswith("_sha256"):
                _digest(value, f"{name} {key}")
            elif key.endswith("_ppm"):
                continue
            elif type(value) is not str or not value:
                raise ValueError(f"{name} {key} is invalid")
    if _sha(rows) != _digest(digest, f"{name} digest"):
        raise ValueError(f"{name} payload drift")


def _row_keys(
    rows: list[Mapping[str, Any]], symbols: list[str], window: Mapping[str, Any], name: str
) -> list[tuple[str, str]]:
    expected = [
        (timestamp, symbol) for timestamp in window["output_timestamps_utc"] for symbol in symbols
    ]
    keys = [(item["timestamp_utc"], item["symbol"]) for item in rows]
    if keys != expected:
        raise ValueError(f"{name} row coverage/order drift")
    return keys


def _event_rows(rows: list[Mapping[str, Any]], window: Mapping[str, Any]) -> None:
    start = _timestamp(window["start_utc"], "event window start")
    end = _timestamp(window["end_utc"], "event window end")
    keys = [(item["timestamp_utc"], item["event_index"]) for item in rows]
    if (
        keys != sorted(keys)
        or len(set(keys)) != len(keys)
        or {item["timestamp_utc"] for item in rows} != set(window["output_timestamps_utc"])
        or any(
            not start <= _timestamp(item["timestamp_utc"], "event timestamp") < end for item in rows
        )
    ):
        raise ValueError("event chronology/window drift")


def _execution_receipt(
    digest: Any,
    schema: str,
    fold: Mapping[str, Any],
    leaf: Mapping[str, Any],
    data: Mapping[str, Any],
    window: Mapping[str, Any],
    scale: int,
    artifacts: _Artifacts,
) -> Mapping[str, Any]:
    kind = schema.removeprefix("router_").removesuffix("_v2")
    row = artifacts.get(digest, kind, schema)
    common = {
        "schema",
        "fold_id",
        "locked_oos",
        "membership_fold_sha256",
        "leaf_id",
        "source_row_sha256",
        "params_sha256",
        "dependency_symbols",
        "evaluation_mode",
        "engine_handler",
        "strategy_class",
        "native_timeframe",
        "engine_dependency_receipt_sha256",
        "data_receipt_sha256",
        "window_receipt_sha256",
        "applied_scale_ppm",
        "generic_fallback_proxy_count",
        "current_fold_oos_input_count",
    }
    extra = {
        "router_signal_receipt_v2": {"signal_rows", "signal_rows_sha256"},
        "router_position_receipt_v2": {
            "signal_receipt_sha256",
            "position_rows",
            "position_rows_sha256",
        },
        "router_engine_receipt_v2": {
            "signal_receipt_sha256",
            "position_receipt_sha256",
            "execution_rows",
            "execution_rows_sha256",
            "event_rows",
            "event_rows_sha256",
        },
    }[schema]
    row = _exact(row, common | extra, schema)
    _receipt_identity(row, fold, leaf, data, window, scale)
    _engine_dependency(
        row["engine_dependency_receipt_sha256"],
        artifacts,
        leaf["evaluation_mode"],
        leaf["engine_handler"],
        leaf["strategy_class"],
    )
    if schema == "router_signal_receipt_v2":
        _payload(
            row["signal_rows"],
            row["signal_rows_sha256"],
            "signal",
            {"timestamp_utc", "symbol", "signal_ppm"},
        )
        _row_keys(row["signal_rows"], leaf["traded_symbols"], window, "signal")
        for signal_row in row["signal_rows"]:
            _integer(signal_row["signal_ppm"], "signal", *_BASE_SIGNAL_PPM)
    elif schema == "router_position_receipt_v2":
        signal = _execution_receipt(
            row["signal_receipt_sha256"],
            "router_signal_receipt_v2",
            fold,
            leaf,
            data,
            window,
            scale,
            artifacts,
        )
        _payload(
            row["position_rows"],
            row["position_rows_sha256"],
            "position",
            {"timestamp_utc", "symbol", "position_ppm"},
        )
        position_keys = _row_keys(row["position_rows"], leaf["traded_symbols"], window, "position")
        signal_keys = _row_keys(signal["signal_rows"], leaf["traded_symbols"], window, "signal")
        if position_keys != signal_keys:
            raise ValueError("position/signal key drift")
        for position_row in row["position_rows"]:
            _integer(position_row["position_ppm"], "position", *_SCALED_POSITION_PPM)
        for position, signal_row in zip(row["position_rows"], signal["signal_rows"], strict=True):
            expected = _round_fraction_half_up(
                Fraction(signal_row["signal_ppm"] * scale, 1_000_000)
            )
            if position["position_ppm"] != expected:
                raise ValueError("position scale derivation drift")
    else:
        signal = _execution_receipt(
            row["signal_receipt_sha256"],
            "router_signal_receipt_v2",
            fold,
            leaf,
            data,
            window,
            scale,
            artifacts,
        )
        position = _execution_receipt(
            row["position_receipt_sha256"],
            "router_position_receipt_v2",
            fold,
            leaf,
            data,
            window,
            scale,
            artifacts,
        )
        _payload(
            row["execution_rows"],
            row["execution_rows_sha256"],
            "execution",
            {"timestamp_utc", "symbol", "base_return_ppm", "return_ppm"},
        )
        execution_keys = _row_keys(
            row["execution_rows"], leaf["traded_symbols"], window, "execution"
        )
        if execution_keys != _row_keys(
            position["position_rows"], leaf["traded_symbols"], window, "position"
        ) or execution_keys != _row_keys(
            signal["signal_rows"], leaf["traded_symbols"], window, "signal"
        ):
            raise ValueError("execution/signal/position key drift")
        for execution_row in row["execution_rows"]:
            _integer(execution_row["base_return_ppm"], "base return", *_BASE_RETURN_PPM)
            _integer(execution_row["return_ppm"], "return", *_SCALED_RETURN_PPM)
        for execution_row in row["execution_rows"]:
            expected_return = _round_fraction_half_up(
                Fraction(execution_row["base_return_ppm"] * scale, 1_000_000)
            )
            if execution_row["return_ppm"] != expected_return:
                raise ValueError("execution scale derivation drift")
        _payload(
            row["event_rows"],
            row["event_rows_sha256"],
            "event",
            {"timestamp_utc", "event_index", "event_type", "event_sha256"},
        )
        _event_rows(row["event_rows"], window)
    return row


def _round_fraction_half_up(value: Fraction) -> int:
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude.numerator, magnitude.denominator)
    rounded = quotient + (1 if 2 * remainder >= magnitude.denominator else 0)
    return rounded if value >= 0 else -rounded


def _aggregate_candidate_return_rows(
    engines: list[Mapping[str, Any]], leaves: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if len(engines) != len(leaves) or not engines:
        raise ValueError("candidate aggregation inputs are invalid")
    per_engine: list[list[tuple[str, int]]] = []
    for engine in engines:
        grouped: dict[str, int] = {}
        for row in engine["execution_rows"]:
            grouped[row["timestamp_utc"]] = grouped.get(row["timestamp_utc"], 0) + row["return_ppm"]
        if not grouped:
            raise ValueError("candidate engine chronology drift")
        per_engine.append(list(grouped.items()))
    expected_timestamps = [stamp for stamp, _ in per_engine[0]]
    if any([stamp for stamp, _ in rows] != expected_timestamps for rows in per_engine[1:]):
        raise ValueError("candidate engine chronology drift")
    aggregated: list[dict[str, Any]] = []
    for index, timestamp in enumerate(expected_timestamps):
        weighted = sum(
            (
                Fraction(per_engine[engine_index][index][1] * leaf["source_weight_ppm"], 1_000_000)
                for engine_index, leaf in enumerate(leaves)
            ),
            Fraction(0),
        )
        aggregated.append(
            {"timestamp_utc": timestamp, "return_ppm": _round_fraction_half_up(weighted)}
        )
    return aggregated


def _history(
    digest: Any,
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    prior: Mapping[str, Any],
    prior_source_sha: str,
    prior_eligible_symbols: set[str],
    artifacts: _Artifacts,
) -> int:
    row = artifacts.get(digest, "history_receipt", "router_history_receipt_v2")
    row = _exact(
        row,
        {
            "schema",
            "fold_id",
            "candidate_label",
            "locked_oos",
            "input_cutoff_utc",
            "completed_at_utc",
            "membership_fold_sha256",
            "prior_source_artifact_sha256",
            "leaf_list_sha256",
            "data_receipt_sha256",
            "window_receipt_sha256",
            "engine_receipt_sha256s",
            "candidate_aggregation_receipt_sha256",
            "candidate_return_rows",
            "candidate_return_rows_sha256",
            "return_ppm",
        },
        "history receipt",
    )
    prior_candidates = prior["candidates"]
    prior_candidate = next(
        (
            item
            for item in prior_candidates
            if item["candidate_label"] == candidate["candidate_label"]
        ),
        None,
    )
    if prior_candidate is None:
        raise ValueError("history candidate is absent from prior source fold")
    completed = _timestamp(row["completed_at_utc"], "history completed")
    if (
        row["fold_id"] != prior["fold_id"]
        or row["candidate_label"] != candidate["candidate_label"]
        or row["locked_oos"] != prior["locked_oos"]
        or row["input_cutoff_utc"] != prior["input_cutoff_utc"]
        or row["membership_fold_sha256"] != prior["membership_fold_sha256"]
        or row["prior_source_artifact_sha256"] != prior_source_sha
        or row["leaf_list_sha256"] != prior_candidate["leaf_list_sha256"]
        or not _timestamp(prior["locked_oos"]["end_utc"], "prior OOS end")
        <= completed
        < _timestamp(current["input_cutoff_utc"], "current cutoff")
    ):
        raise ValueError("history provenance chronology drift")
    leaves = prior_candidate["leaves"]
    if _sha(leaves) != row["leaf_list_sha256"]:
        raise ValueError("history leaf list drift")
    if (
        not leaves
        or len({leaf.get("leaf_id") for leaf in leaves if isinstance(leaf, Mapping)}) != len(leaves)
        or len({leaf.get("native_timeframe") for leaf in leaves if isinstance(leaf, Mapping)}) != 1
    ):
        raise ValueError("history leaf identity/timeframe drift")
    timeframe = next(iter({leaf["native_timeframe"] for leaf in leaves}))
    if type(timeframe) is not str or not timeframe:
        raise ValueError("history leaf timeframe is invalid")
    symbols = list(
        dict.fromkeys(symbol for leaf in leaves for symbol in leaf["dependency_symbols"])
    )
    data = _data_window(
        row["data_receipt_sha256"], "router_data_receipt_v2", artifacts, prior, symbols, timeframe
    )
    window = _data_window(
        row["window_receipt_sha256"],
        "router_window_receipt_v2",
        artifacts,
        prior,
        symbols,
        timeframe,
        role="locked_oos",
    )
    engines = row["engine_receipt_sha256s"]
    if (
        not isinstance(engines, list)
        or len(engines) != len(leaves)
        or len(set(engines)) != len(engines)
    ):
        raise ValueError("history engine sequence is invalid")
    engine_rows: list[Mapping[str, Any]] = []
    for engine_digest, leaf in zip(engines, leaves, strict=True):
        _leaf(leaf, prior_eligible_symbols, artifacts)
        engine_rows.append(
            _execution_receipt(
                engine_digest,
                "router_engine_receipt_v2",
                prior,
                leaf,
                data,
                window,
                1_000_000,
                artifacts,
            )
        )
    aggregation = artifacts.get(
        row["candidate_aggregation_receipt_sha256"],
        "candidate_aggregation",
        "router_candidate_aggregation_receipt_v2",
    )
    aggregation = _exact(
        aggregation,
        {"schema", "engine_receipt_sha256s", "weights_ppm", "candidate_return_rows_sha256"},
        "candidate aggregation receipt",
    )
    if (
        not isinstance(aggregation["weights_ppm"], list)
        or any(type(weight) is not int for weight in aggregation["weights_ppm"])
        or aggregation["engine_receipt_sha256s"] != engines
        or aggregation["weights_ppm"] != [leaf["source_weight_ppm"] for leaf in leaves]
        or aggregation["candidate_return_rows_sha256"] != row["candidate_return_rows_sha256"]
    ):
        raise ValueError("history candidate aggregation binding drift")
    rows = row["candidate_return_rows"]
    if (
        not isinstance(rows, list)
        or not rows
        or _sha(rows) != _digest(row["candidate_return_rows_sha256"], "history candidate returns")
    ):
        raise ValueError("history candidate return commitment drift")
    if rows != _aggregate_candidate_return_rows(engine_rows, leaves):
        raise ValueError("history candidate return aggregation drift")
    returns: list[int] = []
    previous: datetime | None = None
    for item in rows:
        item = _exact(item, {"timestamp_utc", "return_ppm"}, "history candidate return row")
        timestamp = _timestamp(item["timestamp_utc"], "history candidate return timestamp")
        if previous is not None and timestamp <= previous:
            raise ValueError("history candidate return chronology drift")
        previous = timestamp
        returns.append(
            _integer(item["return_ppm"], "history candidate return", -1_000_000, 100_000_000)
        )
    equity, _ = _period_metrics(returns, 1_000_000)
    derived = _round_fraction_half_up(equity * 1_000_000)
    if row["return_ppm"] != derived:
        raise ValueError("history compounded return derivation drift")
    return _integer(row["return_ppm"], "history return", -1_000_000, 100_000_000)


def _fallback_scale(shared: Mapping[str, Any], target_mdd: Fraction, cap: Fraction) -> int:
    train, validation = shared["train_returns_ppm"], shared["validation_returns_ppm"]
    if (
        not isinstance(train, list)
        or not isinstance(validation, list)
        or not train
        or not validation
    ):
        raise ValueError("fallback return evidence is invalid")
    for value in [*train, *validation]:
        _integer(value, "fallback return", *_SCALED_RETURN_PPM)
    best_scale, best_score = 0, None
    for scale in SCALE_GRID_PPM:
        factor = Fraction(scale, 1_000_000)
        if factor > cap:
            continue
        train_return, _ = _period_metrics(train, scale)
        validation_return, validation_mdd = _period_metrics(validation, scale)
        if (
            train_return <= Fraction(-2, 100)
            or validation_return < 0
            or validation_mdd > target_mdd
        ):
            continue
        score = (
            validation_return / max(validation_mdd, Fraction(2, 100))
            + Fraction(10, 100) * min(train_return, Fraction(2))
            - Fraction(3, 100) * factor
        )
        if best_score is None or score > best_score:
            best_scale, best_score = scale, score
    return best_scale


def _strict_core(
    source_fold: Mapping[str, Any],
) -> tuple[str, Any, list[Any], Mapping[str, Any] | None]:
    strict = _exact(
        source_fold["strict_core"],
        {"candidates", "leaves", "leaf_list_sha256", "shared_mdd_receipt_sha256"},
        "strict core",
    )
    if (
        not isinstance(strict["candidates"], list)
        or _sha(strict["leaves"]) != strict["leaf_list_sha256"]
    ):
        raise ValueError("strict core leaf list drift")
    candidates: dict[str, Mapping[str, Any]] = {}
    eligible: list[tuple[Fraction, Fraction, Mapping[str, Any]]] = []
    expected_labels = (BALANCED_LABEL, GROWTH_LABEL)
    if len(strict["candidates"]) > len(expected_labels):
        raise ValueError("strict candidate count is invalid")
    for candidate in strict["candidates"]:
        candidate = _exact(
            candidate,
            {
                "candidate_label",
                "source_kind",
                "source_candidate",
                "source_eligibility",
                "train_return_ppm",
                "validation_return_ppm",
                "validation_mdd_ppm",
                "leaves",
                "leaf_list_sha256",
            },
            "strict candidate",
        )
        flags = _exact(
            candidate["source_eligibility"],
            {
                "post_oos_augment",
                "generic_fallback_proxy",
                "current_fold_oos_input",
                "recomputed_from_json",
            },
            "strict source controls",
        )
        expected_flags = {
            "post_oos_augment": False,
            "generic_fallback_proxy": False,
            "current_fold_oos_input": False,
            "recomputed_from_json": False,
        }
        _source_eligible(candidate["source_candidate"], "strict source candidate")
        if (
            candidate["source_candidate"]["candidate_label"] != candidate["candidate_label"]
            or candidate["candidate_label"] not in expected_labels
            or candidate["candidate_label"] in candidates
            or candidate["source_kind"] != "lagged_shadow_leaf"
            or flags != expected_flags
            or any(type(value) is not bool for value in flags.values())
            or _sha(candidate["leaves"]) != candidate["leaf_list_sha256"]
        ):
            raise ValueError("strict candidate identity/eligibility drift")
        candidates[candidate["candidate_label"]] = candidate
        train = Fraction(
            _integer(candidate["train_return_ppm"], "strict train", -1_000_000, 100_000_000),
            1_000_000,
        )
        validation = Fraction(
            _integer(
                candidate["validation_return_ppm"], "strict validation", -1_000_000, 100_000_000
            ),
            1_000_000,
        )
        mdd = Fraction(
            _integer(candidate["validation_mdd_ppm"], "strict MDD", 0, 1_000_000), 1_000_000
        )
        if (
            candidate["leaves"]
            and train >= Fraction(-2, 100)
            and validation >= Fraction(-2, 100)
            and mdd <= Fraction(20, 100)
        ):
            eligible.append((validation / max(mdd, Fraction(2, 100)), validation, candidate))
    expected_order = tuple(label for label in expected_labels if label in candidates)
    if tuple(candidates) != expected_order:
        raise ValueError("strict candidate order drift")
    selected = None
    balanced = candidates.get(BALANCED_LABEL)
    growth = candidates.get(GROWTH_LABEL)
    if balanced is not None and growth is not None:
        b_mdd = Fraction(balanced["validation_mdd_ppm"], 1_000_000)
        if (
            balanced["leaves"]
            and Fraction(balanced["train_return_ppm"], 1_000_000) >= Fraction(-2, 100)
            and Fraction(balanced["validation_return_ppm"], 1_000_000) >= Fraction(-2, 100)
            and b_mdd <= Fraction(20, 100)
        ):
            selected = growth if b_mdd > Fraction(10, 100) else balanced
    if selected is None and eligible:
        selected = max(eligible, key=lambda item: item[:2])[2]
    if (
        selected is None
        or Fraction(selected["validation_return_ppm"], 1_000_000) < Fraction(2, 100)
        or Fraction(selected["validation_return_ppm"], 1_000_000)
        / max(Fraction(selected["validation_mdd_ppm"], 1_000_000), Fraction(1, 100))
        < Fraction(80, 100)
    ):
        return "strict_core_cash", None, [], None
    if selected["leaves"] != strict["leaves"]:
        raise ValueError("strict selected leaves drift")
    return "strict_core_scaled", selected["candidate_label"], strict["leaves"], strict


def _decision(
    source_fold: Mapping[str, Any],
    prior: list[tuple[Mapping[str, Any], str, set[str]]],
    artifacts: _Artifacts,
) -> tuple[str, Any, list[Any], str | None, list[str]]:
    if not isinstance(source_fold["candidates"], list) or not source_fold["candidates"]:
        raise ValueError("source candidates are invalid")
    eligible, union = [], []
    for order, candidate in enumerate(source_fold["candidates"]):
        candidate = _exact(
            candidate,
            {
                "candidate_label",
                "source_candidate",
                "train_return_ppm",
                "train_mdd_ppm",
                "validation_return_ppm",
                "validation_mdd_ppm",
                "history_receipt_sha256s",
                "leaves",
                "leaf_list_sha256",
            },
            "source candidate",
        )
        _source_eligible(candidate["source_candidate"], "source candidate")
        if candidate["source_candidate"]["candidate_label"] != candidate["candidate_label"]:
            raise ValueError("source candidate label drift")
        for name, low, high in (
            ("train_return_ppm", -1_000_000, 100_000_000),
            ("train_mdd_ppm", 0, 1_000_000),
            ("validation_return_ppm", -1_000_000, 100_000_000),
            ("validation_mdd_ppm", 0, 1_000_000),
        ):
            _integer(candidate[name], name, low, high)
        history, tail = candidate["history_receipt_sha256s"], prior[-4:]
        if (
            not isinstance(history, list)
            or len(history) != len(tail)
            or len(set(history)) != len(history)
        ):
            raise ValueError("history tail is incomplete or duplicate")
        returns = [
            _history(
                item,
                source_fold,
                candidate,
                previous,
                source_sha,
                prior_eligible_symbols,
                artifacts,
            )
            for item, (previous, source_sha, prior_eligible_symbols) in zip(
                history, tail, strict=True
            )
        ]
        union.extend(history)
        if (
            len(returns) >= 4
            and candidate["train_return_ppm"] >= -20_000
            and candidate["train_mdd_ppm"] <= 500_000
            and candidate["validation_return_ppm"] >= 0
            and candidate["validation_mdd_ppm"] <= 250_000
        ):
            lagged, score = Fraction(returns[-1], 1_000_000), _validation_score(candidate)
            eligible.append(
                (
                    lagged + Fraction(1, 4) * score,
                    lagged,
                    score,
                    Fraction(candidate["validation_return_ppm"], 1_000_000),
                    -order,
                    candidate,
                )
            )
    if eligible:
        selected = max(eligible, key=lambda item: item[:5])[-1]
        if _sha(selected["leaves"]) != selected["leaf_list_sha256"]:
            raise ValueError("candidate leaf list drift")
        return (
            "pre_registered_lagged_plus_validation_leaf",
            selected["candidate_label"],
            selected["leaves"],
            None,
            union,
        )
    branch, label, leaves, strict = _strict_core(source_fold)
    return (
        branch,
        label,
        leaves,
        None if strict is None else strict["shared_mdd_receipt_sha256"],
        union,
    )


def _cost_tape(
    digest: Any,
    fold: Mapping[str, Any],
    variant_id: str,
    selected_label: str,
    leaf: Mapping[str, Any],
    execution: Mapping[str, Any],
    artifacts: _Artifacts,
    downstream_row_digests: set[str],
) -> None:
    row = artifacts.get(digest, "cost_tape_receipt", "router_cost_tape_receipt_v1")
    row = _exact(
        row,
        {
            "schema",
            "fold_id",
            "variant_id",
            "selected_label",
            "leaf_id",
            "source_row_sha256",
            "params_sha256",
            "engine_receipt_sha256",
            "signal_receipt_sha256",
            "position_receipt_sha256",
            "tapes",
        },
        "cost tape receipt",
    )
    expected = {
        "fold_id": fold["fold_id"],
        "variant_id": variant_id,
        "selected_label": selected_label,
        "leaf_id": leaf["leaf_id"],
        "source_row_sha256": leaf["source_row_sha256"],
        "params_sha256": leaf["params_sha256"],
        "engine_receipt_sha256": execution["engine_receipt_sha256"],
        "signal_receipt_sha256": execution["signal_receipt_sha256"],
        "position_receipt_sha256": execution["position_receipt_sha256"],
    }
    if (
        any(row[name] != value for name, value in expected.items())
        or not isinstance(row["tapes"], list)
        or len(row["tapes"]) != 4
    ):
        raise ValueError("cost tape identity drift")
    for tape, bps in zip(row["tapes"], (10, 15, 20, 30), strict=True):
        tape = _exact(
            tape,
            {
                "cost_bps",
                "signal_position_sha256",
                "order_tape_sha256",
                "fill_tape_sha256",
                "event_tape_sha256",
            },
            "cost tape commitment",
        )
        if type(tape["cost_bps"]) is not int or tape["cost_bps"] != bps:
            raise ValueError("cost tape order drift")
        for name, kind, schema in (
            (
                "signal_position_sha256",
                "cost_signal_position_tape",
                "router_cost_signal_position_tape_v1",
            ),
            ("order_tape_sha256", "cost_order_tape", "router_cost_order_tape_v1"),
            ("fill_tape_sha256", "cost_fill_tape", "router_cost_fill_tape_v1"),
            ("event_tape_sha256", "cost_event_tape", "router_cost_event_tape_v1"),
        ):
            artifact = artifacts.get(tape[name], kind, schema)
            artifact = _exact(
                artifact,
                {
                    "schema",
                    "cost_cell",
                    "cost_bps",
                    "fold_id",
                    "variant_id",
                    "leaf_id",
                    "engine_receipt_sha256",
                    "sequence",
                    "sequence_sha256",
                    "rows",
                    "rows_sha256",
                },
                "cost tape artifact",
            )
            if (
                type(artifact["cost_bps"]) is not int
                or artifact["cost_cell"] != f"{bps}bps"
                or artifact["cost_bps"] != bps
                or artifact["fold_id"] != fold["fold_id"]
                or artifact["variant_id"] != variant_id
                or artifact["leaf_id"] != leaf["leaf_id"]
                or artifact["engine_receipt_sha256"] != execution["engine_receipt_sha256"]
                or _sha(artifact["sequence"])
                != _digest(artifact["sequence_sha256"], "cost sequence")
                or _sha(artifact["rows"]) != _digest(artifact["rows_sha256"], "cost rows")
            ):
                raise ValueError("cost tape artifact binding drift")
            if (
                not isinstance(artifact["sequence"], list)
                or not artifact["sequence"]
                or any(type(item) is not str or not item for item in artifact["sequence"])
                or len(set(artifact["sequence"])) != len(artifact["sequence"])
                or not isinstance(artifact["rows"], list)
                or len(artifact["rows"]) != len(artifact["sequence"])
            ):
                raise ValueError("cost tape artifact payload is invalid")
            for sequence, item in zip(artifact["sequence"], artifact["rows"], strict=True):
                item = _exact(
                    item,
                    {
                        "sequence_id",
                        "fold_id",
                        "variant_id",
                        "leaf_id",
                        "engine_receipt_sha256",
                        "row_sha256",
                    },
                    "cost tape row",
                )
                _digest(item["row_sha256"], "cost tape row")
                digest = item["row_sha256"]
                if digest in downstream_row_digests:
                    raise ValueError("cost tape row digest is reused")
                downstream_row_digests.add(digest)
                if item["sequence_id"] != sequence or any(
                    item[name] != artifact[name]
                    for name in ("fold_id", "variant_id", "leaf_id", "engine_receipt_sha256")
                ):
                    raise ValueError("cost tape row identity drift")


def _variants(
    value: Any,
    fold: Mapping[str, Any],
    branch: str,
    label: Any,
    leaves: list[Any],
    shared_digest: str | None,
    artifacts: _Artifacts,
    source_fold: Mapping[str, Any],
    downstream_row_digests: set[str],
) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("variant count is invalid")
    shared = None
    parity: list[list[tuple[Mapping[str, Any], str, str, str, str]]] = []
    if branch == "strict_core_scaled":
        shared = artifacts.get(shared_digest, "shared_mdd", "router_shared_mdd_receipt_v2")
        shared = _exact(
            shared,
            {
                "schema",
                "fold_id",
                "membership_fold_sha256",
                "leaf_list_sha256",
                "candidate_label",
                "measurement_end_utc",
                "input_cutoff_utc",
                "data_receipt_sha256",
                "train_window_receipt_sha256",
                "validation_window_receipt_sha256",
                "engine_dependency_receipt_sha256s",
                "train_engine_receipt_sha256s",
                "validation_engine_receipt_sha256s",
                "native_timeframe",
                "train_return_rows",
                "validation_return_rows",
                "train_return_rows_sha256",
                "validation_return_rows_sha256",
            },
            "shared MDD receipt",
        )
        if (
            shared["fold_id"] != fold["fold_id"]
            or shared["membership_fold_sha256"] != fold["membership_fold_sha256"]
            or shared["leaf_list_sha256"] != _sha(leaves)
            or shared["candidate_label"] != label
            or shared["input_cutoff_utc"] != fold["input_cutoff_utc"]
            or _timestamp(shared["measurement_end_utc"], "MDD measurement")
            > _timestamp(fold["input_cutoff_utc"], "cutoff")
        ):
            raise ValueError("shared MDD provenance drift")
        symbols = list(
            dict.fromkeys(symbol for leaf in leaves for symbol in leaf["dependency_symbols"])
        )
        if {leaf["native_timeframe"] for leaf in leaves} != {shared["native_timeframe"]} or shared[
            "engine_dependency_receipt_sha256s"
        ] != [leaf["engine_dependency_receipt_sha256"] for leaf in leaves]:
            raise ValueError("shared MDD engine/timeframe drift")
        data = _data_window(
            shared["data_receipt_sha256"],
            "router_data_receipt_v2",
            artifacts,
            fold,
            symbols,
            shared["native_timeframe"],
        )
        train_window = _data_window(
            shared["train_window_receipt_sha256"],
            "router_window_receipt_v2",
            artifacts,
            fold,
            symbols,
            shared["native_timeframe"],
            "train",
        )
        validation_window = _data_window(
            shared["validation_window_receipt_sha256"],
            "router_window_receipt_v2",
            artifacts,
            fold,
            symbols,
            shared["native_timeframe"],
            "validation",
        )
        if (
            train_window["data_receipt_sha256"] != _sha(data)
            or validation_window["data_receipt_sha256"] != _sha(data)
            or train_window == validation_window
        ):
            raise ValueError("shared MDD data/window drift")
        expected_train = _exact(
            source_fold["train_window"], {"start_utc", "end_utc"}, "source train window"
        )
        expected_validation = _exact(
            source_fold["validation_window"], {"start_utc", "end_utc"}, "source validation window"
        )
        if (
            {name: train_window[name] for name in expected_train} != expected_train
            or {name: validation_window[name] for name in expected_validation}
            != expected_validation
            or not _timestamp(train_window["start_utc"], "train start")
            < _timestamp(train_window["end_utc"], "train end")
            <= _timestamp(validation_window["start_utc"], "validation start")
            < _timestamp(validation_window["end_utc"], "validation end")
            <= _timestamp(fold["input_cutoff_utc"], "cutoff")
        ):
            raise ValueError("shared MDD authoritative window drift")
        for dependency, leaf in zip(
            shared["engine_dependency_receipt_sha256s"], leaves, strict=True
        ):
            _engine_dependency(
                dependency,
                artifacts,
                leaf["evaluation_mode"],
                leaf["engine_handler"],
                leaf["strategy_class"],
            )
        derived_rows: dict[str, list[dict[str, Any]]] = {}
        for name, engines, window in (
            ("train_return_rows", shared["train_engine_receipt_sha256s"], train_window),
            (
                "validation_return_rows",
                shared["validation_engine_receipt_sha256s"],
                validation_window,
            ),
        ):
            if not isinstance(engines, list) or len(engines) != len(leaves):
                raise ValueError("shared MDD engine sequence drift")
            engine_rows = [
                _execution_receipt(
                    engine,
                    "router_engine_receipt_v2",
                    fold,
                    leaf,
                    data,
                    window,
                    1_000_000,
                    artifacts,
                )
                for engine, leaf in zip(engines, leaves, strict=True)
            ]
            derived_rows[name] = _aggregate_candidate_return_rows(engine_rows, leaves)
        for name, window in (
            ("train_return_rows", train_window),
            ("validation_return_rows", validation_window),
        ):
            rows = shared[name]
            if (
                not isinstance(rows, list)
                or not rows
                or _sha(rows) != _digest(shared[f"{name}_sha256"], name)
                or rows != derived_rows[name]
            ):
                raise ValueError("shared MDD return commitment drift")
            returns = []
            previous: datetime | None = None
            for item in rows:
                item = _exact(item, {"timestamp_utc", "return_ppm"}, "shared MDD return row")
                stamp = _timestamp(item["timestamp_utc"], "shared MDD return timestamp")
                if (previous is not None and stamp <= previous) or not _timestamp(
                    window["start_utc"], "window start"
                ) <= stamp < _timestamp(window["end_utc"], "window end"):
                    raise ValueError("shared MDD return chronology/window drift")
                previous = stamp
                returns.append(
                    _integer(item["return_ppm"], "shared MDD return", *_SCALED_RETURN_PPM)
                )
            shared[f"{name[:-5]}s_ppm"] = returns
    for variant, identifier, target, cap in zip(
        value,
        CANDIDATE_IDS,
        (Fraction(30, 100), Fraction(20, 100)),
        (Fraction(3), Fraction(2)),
        strict=True,
    ):
        variant = _exact(
            variant,
            {
                "variant_id",
                "selected_label",
                "base_leaf_list_sha256",
                "policy",
                "applied_scale_ppm",
                "leaves",
                "execution_receipts",
            },
            "variant",
        )
        if (
            not isinstance(variant["policy"], Mapping)
            or set(variant["policy"]) != {"fallback_mdd_ppm", "fallback_cap_ppm"}
            or any(type(item) is not int for item in variant["policy"].values())
        ):
            raise ValueError("variant policy integer drift")
        if (
            variant["variant_id"] != identifier
            or variant["selected_label"] != label
            or variant["base_leaf_list_sha256"] != _sha(leaves)
            or variant["policy"]
            != {
                "fallback_mdd_ppm": int(target * 1_000_000),
                "fallback_cap_ppm": int(cap * 1_000_000),
            }
        ):
            raise ValueError("variant selection binding drift")
        scale = _integer(variant["applied_scale_ppm"], "applied scale", 0, 3_000_000)
        expected_scale = (
            0
            if branch == "strict_core_cash"
            else 1_000_000
            if branch == "pre_registered_lagged_plus_validation_leaf"
            else _fallback_scale(shared, target, cap)
        )
        if (
            scale != expected_scale
            or len(variant["leaves"]) != len(leaves)
            or len(variant["execution_receipts"]) != len(leaves)
        ):
            raise ValueError("variant scale/rows drift")
        variant_parity: list[tuple[Mapping[str, Any], str, str, str, str]] = []
        for base, effective, execution in zip(
            leaves, variant["leaves"], variant["execution_receipts"], strict=True
        ):
            if (
                type(effective.get("effective_weight_ppm")) is not int
                or type(effective.get("effective_gross_ppm")) is not int
            ):
                raise ValueError("effective leaf integer drift")
            if effective != {
                "leaf_id": base["leaf_id"],
                "effective_weight_ppm": base["source_weight_ppm"] * scale // 1_000_000,
                "effective_gross_ppm": base["native_gross_ppm"] * scale // 1_000_000,
            }:
                raise ValueError("effective leaf scale drift")
            execution = _exact(
                execution,
                {
                    "leaf_id",
                    "evaluation_mode",
                    "engine_source_sha256",
                    "engine_dependency_receipt_sha256",
                    "data_receipt_sha256",
                    "window_receipt_sha256",
                    "signal_receipt_sha256",
                    "position_receipt_sha256",
                    "engine_receipt_sha256",
                    "cost_tape_receipt_sha256",
                    "generic_fallback_proxy_count",
                    "current_fold_oos_input_count",
                },
                "execution row",
            )
            if any(
                execution[name] != base[name]
                for name in (
                    "leaf_id",
                    "evaluation_mode",
                    "engine_source_sha256",
                    "engine_dependency_receipt_sha256",
                )
            ):
                raise ValueError("execution leaf binding drift")
            data = _data_window(
                execution["data_receipt_sha256"],
                "router_data_receipt_v2",
                artifacts,
                fold,
                base["dependency_symbols"],
                base["native_timeframe"],
            )
            window = _data_window(
                execution["window_receipt_sha256"],
                "router_window_receipt_v2",
                artifacts,
                fold,
                base["dependency_symbols"],
                base["native_timeframe"],
                "locked_oos",
            )
            signal = _execution_receipt(
                execution["signal_receipt_sha256"],
                "router_signal_receipt_v2",
                fold,
                base,
                data,
                window,
                scale,
                artifacts,
            )
            position = _execution_receipt(
                execution["position_receipt_sha256"],
                "router_position_receipt_v2",
                fold,
                base,
                data,
                window,
                scale,
                artifacts,
            )
            engine = _execution_receipt(
                execution["engine_receipt_sha256"],
                "router_engine_receipt_v2",
                fold,
                base,
                data,
                window,
                scale,
                artifacts,
            )
            if (
                engine["signal_receipt_sha256"] != execution["signal_receipt_sha256"]
                or engine["position_receipt_sha256"] != execution["position_receipt_sha256"]
                or position["signal_receipt_sha256"] != execution["signal_receipt_sha256"]
            ):
                raise ValueError("execution chain drift")
            _cost_tape(
                execution["cost_tape_receipt_sha256"],
                fold,
                identifier,
                label,
                base,
                execution,
                artifacts,
                downstream_row_digests,
            )
            if (
                type(execution["generic_fallback_proxy_count"]) is not int
                or type(execution["current_fold_oos_input_count"]) is not int
            ):
                raise ValueError("execution integer controls drift")
            if (
                execution["generic_fallback_proxy_count"] != 0
                or execution["current_fold_oos_input_count"] != 0
            ):
                raise ValueError("execution zero-control drift")
            variant_parity.append(
                (
                    execution,
                    signal["signal_rows_sha256"],
                    position["position_rows_sha256"],
                    engine["execution_rows_sha256"],
                    engine["event_rows_sha256"],
                    [
                        (item["timestamp_utc"], item["symbol"], item["base_return_ppm"])
                        for item in engine["execution_rows"]
                    ],
                    engine["event_rows"],
                )
            )
        parity.append(variant_parity)
    if len(parity) == 2:
        for left, right in zip(parity[0], parity[1], strict=True):
            for name in (
                "leaf_id",
                "evaluation_mode",
                "engine_source_sha256",
                "engine_dependency_receipt_sha256",
                "data_receipt_sha256",
                "window_receipt_sha256",
            ):
                if left[0][name] != right[0][name]:
                    raise ValueError("variant base execution parity drift")
            if left[1] != right[1]:
                raise ValueError("variant signal payload parity drift")
            if branch == "strict_core_scaled" and (left[5] != right[5] or left[6] != right[6]):
                raise ValueError("fallback base return/event parity drift")
            if branch == "pre_registered_lagged_plus_validation_leaf" and left[2:] != right[2:]:
                raise ValueError("mature position/engine payload parity drift")


def _source(source: Any) -> list[Mapping[str, Any]]:
    root = _exact(
        source,
        {
            "schema",
            "candidate_ids",
            "candidate_ids_sha256",
            "controls",
            "frozen_at_utc",
            "policy",
            "candidate_order",
            "folds",
        },
        "source",
    )
    if (
        root["schema"] != SOURCE_SCHEMA
        or tuple(root["candidate_ids"]) != CANDIDATE_IDS
        or root["candidate_ids_sha256"] != CANDIDATE_IDS_SHA256
        or root["candidate_order"] != list(CANDIDATE_IDS)
    ):
        raise ValueError("source candidate identity drift")
    controls = {
        "new_grid_search": False,
        "recompute_from_json": False,
        "post_oos_augment": False,
        "post_oos_research_variant": True,
    }
    if root["controls"] != controls or any(
        type(value) is not bool for value in root["controls"].values()
    ):
        raise ValueError("source controls are invalid")
    policy = {
        "min_history": 4,
        "avg_window": 1,
        "min_train_return_ppm": -20_000,
        "max_train_mdd_ppm": 500_000,
        "min_validation_return_ppm": 0,
        "max_validation_mdd_ppm": 250_000,
        "validation_weight_ppm": 250_000,
        "tie_break": "combined,lagged,validation_score,validation_return,source_order",
    }
    if root["policy"] != policy or any(
        type(root["policy"][name]) is not int
        for name in (
            "min_history",
            "avg_window",
            "min_train_return_ppm",
            "max_train_mdd_ppm",
            "min_validation_return_ppm",
            "max_validation_mdd_ppm",
            "validation_weight_ppm",
        )
    ):
        raise ValueError("source policy drift")
    frozen = _timestamp(root["frozen_at_utc"], "source frozen")
    if not isinstance(root["folds"], list) or not root["folds"]:
        raise ValueError("source folds are invalid")
    if any(not isinstance(fold, Mapping) or "fold_id" not in fold for fold in root["folds"]) or len(
        {fold["fold_id"] for fold in root["folds"]}
    ) != len(root["folds"]):
        raise ValueError("source fold IDs are invalid")
    for fold in root["folds"]:
        locked = _exact(fold["locked_oos"], {"start_utc", "end_utc"}, "source locked OOS")
        if frozen < _timestamp(locked["end_utc"], "source locked OOS end"):
            raise ValueError("source freeze predates represented OOS")
    return root["folds"]


def _source_windows(fold: Mapping[str, Any]) -> None:
    train = _exact(fold["train_window"], {"start_utc", "end_utc"}, "source train window")
    validation = _exact(
        fold["validation_window"], {"start_utc", "end_utc"}, "source validation window"
    )
    train_start = _timestamp(train["start_utc"], "train start_utc")
    train_end = _timestamp(train["end_utc"], "train end_utc")
    validation_start = _timestamp(validation["start_utc"], "validation start_utc")
    validation_end = _timestamp(validation["end_utc"], "validation end_utc")
    if (
        not train_start
        < train_end
        <= validation_start
        < validation_end
        <= _timestamp(fold["input_cutoff_utc"], "fold cutoff")
    ):
        raise ValueError("source train/validation chronology drift")


def _fold(
    manifest_fold: Any,
    source_fold: Mapping[str, Any],
    member: Mapping[str, Any],
    prior: list[tuple[Mapping[str, Any], str, set[str]]],
    artifacts: _Artifacts,
    downstream_row_digests: set[str],
) -> None:
    row = _exact(
        manifest_fold,
        {
            "fold_id",
            "locked_oos",
            "input_cutoff_utc",
            "decision_timestamp_utc",
            "membership_fold_sha256",
            "selection",
            "variants",
        },
        "manifest fold",
    )
    required_source = {
        "fold_id",
        "locked_oos",
        "input_cutoff_utc",
        "decision_timestamp_utc",
        "membership_fold_sha256",
        "candidates",
        "strict_core",
        "train_window",
        "validation_window",
    }
    if set(source_fold) != required_source:
        raise ValueError("source fold schema is invalid")
    if (
        any(
            row[name] != source_fold[name]
            for name in (
                "fold_id",
                "input_cutoff_utc",
                "decision_timestamp_utc",
                "membership_fold_sha256",
            )
        )
        or row["fold_id"] != member["fold_id"]
        or _sha(member) != row["membership_fold_sha256"]
    ):
        raise ValueError("fold source/membership drift")
    locked_oos = _exact(row["locked_oos"], {"start_utc", "end_utc"}, "manifest locked OOS")
    if locked_oos != _exact(
        source_fold["locked_oos"], {"start_utc", "end_utc"}, "source locked OOS"
    ):
        raise ValueError("locked OOS source drift")
    start = datetime.fromtimestamp(member["start_ms"] / 1000, UTC)
    end = datetime.fromtimestamp(member["end_ms"] / 1000, UTC)
    if (
        _timestamp(locked_oos["start_utc"], "OOS start") != start
        or _timestamp(locked_oos["end_utc"], "OOS end") != end
        or _timestamp(row["decision_timestamp_utc"], "decision") != start
        or not _timestamp(row["input_cutoff_utc"], "cutoff") < start
    ):
        raise ValueError("fold timing drift")
    _source_windows(source_fold)
    branch, label, leaves, shared, history = _decision(source_fold, prior, artifacts)
    selection = _exact(
        row["selection"],
        {
            "branch",
            "selected_label",
            "source_fold_sha256",
            "selection_inputs_sha256",
            "current_fold_oos_input_count",
            "decision_receipt_sha256s",
            "fallback_mdd_receipt_sha256",
            "leaves",
            "leaf_list_sha256",
        },
        "selection",
    )
    expected_inputs = {
        "branch": branch,
        "selected_label": label,
        "decision_receipt_sha256s": history,
        "leaves": leaves,
    }
    if (
        selection["branch"] != branch
        or selection["selected_label"] != label
        or selection["source_fold_sha256"] != _sha(source_fold)
        or selection["selection_inputs_sha256"] != _sha(expected_inputs)
        or selection["current_fold_oos_input_count"] != 0
        or type(selection["current_fold_oos_input_count"]) is not int
        or selection["decision_receipt_sha256s"] != history
        or selection["fallback_mdd_receipt_sha256"] != shared
        or selection["leaves"] != leaves
        or selection["leaf_list_sha256"] != _sha(leaves)
    ):
        raise ValueError("selection replay drift")
    eligible = set(member["eligible_symbols"])
    if len({_leaf(leaf, eligible, artifacts) for leaf in leaves}) != len(leaves):
        raise ValueError("leaf identity is duplicated")
    _variants(
        row["variants"],
        row,
        branch,
        label,
        leaves,
        shared,
        artifacts,
        source_fold,
        downstream_row_digests,
    )


def _validate(
    manifest_path: str | Path,
    source_path: str | Path,
    lifecycle_path: str | Path,
    membership_path: str | Path,
    profile_path: str | Path,
    producer_path: str | Path,
    commit_path: str | Path,
    trusted_source: str,
    trusted_commit: str,
    artifact_paths: Mapping[str, str | Path],
) -> int:
    manifest, manifest_sha = _canonical_json(manifest_path)
    source, source_sha = _canonical_json(source_path)
    commit, commit_sha = _canonical_json(commit_path)
    if source_sha != _digest(trusted_source, "trusted source") or commit_sha != _digest(
        trusted_commit, "trusted commit"
    ):
        raise ValueError("trusted root mismatch")
    root = _exact(
        manifest,
        {"schema", "candidate_ids", "candidate_ids_sha256", "controls", "provenance", "folds"},
        "manifest",
    )
    if (
        root["schema"] != SCHEMA
        or tuple(root["candidate_ids"]) != CANDIDATE_IDS
        or root["candidate_ids_sha256"] != CANDIDATE_IDS_SHA256
        or _sha(list(CANDIDATE_IDS)) != CANDIDATE_IDS_SHA256
    ):
        raise ValueError("manifest candidate identity drift")
    controls = {
        "new_grid_search": False,
        "recompute_from_json": False,
        "post_oos_augment": False,
        "real_money_enabled": False,
        "orders_submitted": 0,
        "capital_allocated": 0,
    }
    if (
        root["controls"] != controls
        or any(
            type(root["controls"][name]) is not bool
            for name in (
                "new_grid_search",
                "recompute_from_json",
                "post_oos_augment",
                "real_money_enabled",
            )
        )
        or any(
            type(root["controls"][name]) is not int
            for name in ("orders_submitted", "capital_allocated")
        )
    ):
        raise ValueError("manifest controls are invalid")
    commit = _exact(
        commit,
        {
            "schema",
            "repository_commit",
            "candidate_ids",
            "candidate_ids_sha256",
            "manifest_sha256",
            "source_artifact_sha256",
            "producer_source_sha256",
            "verifier_source_sha256",
            "lifecycle_registry_sha256",
            "membership_manifest_sha256",
            "combined_profile_sha256",
            "runner_source_sha256",
            "research_runner_source_sha256",
            "artifact_index",
        },
        "commit root",
    )
    if (
        commit["schema"] != COMMIT_SCHEMA
        or tuple(commit["candidate_ids"]) != CANDIDATE_IDS
        or commit["candidate_ids_sha256"] != CANDIDATE_IDS_SHA256
    ):
        raise ValueError("commit candidate identity drift")
    expected_files = {
        "manifest_sha256": manifest_sha,
        "source_artifact_sha256": source_sha,
        "producer_source_sha256": _raw_sha(Path(producer_path).read_bytes()),
        "verifier_source_sha256": _raw_sha(Path(__file__).read_bytes()),
        "lifecycle_registry_sha256": _raw_sha(Path(lifecycle_path).read_bytes()),
        "membership_manifest_sha256": _raw_sha(Path(membership_path).read_bytes()),
        "combined_profile_sha256": _raw_sha(Path(profile_path).read_bytes()),
        "runner_source_sha256": _raw_sha(_object_source(ROUTER_SOURCE_PREDICATE).read_bytes()),
        "research_runner_source_sha256": _raw_sha(
            Path(
                inspect.getsourcefile(importlib.import_module(RUNNER_SOURCE_PATH)) or ""
            ).read_bytes()
        ),
    }
    if (
        any(commit[name] != value for name, value in expected_files.items())
        or not isinstance(commit["repository_commit"], str)
        or len(commit["repository_commit"]) != 40
        or set(commit["repository_commit"]) - _HASH
    ):
        raise ValueError("commit root binding drift")
    index_rows = commit["artifact_index"]
    if not isinstance(index_rows, list):
        raise ValueError("artifact index is invalid")
    index: dict[str, str] = {}
    previous = ""
    for item in index_rows:
        item = _exact(item, {"kind", "sha256"}, "artifact index row")
        digest = _digest(item["sha256"], "artifact index digest")
        if (
            type(item["kind"]) is not str
            or not item["kind"]
            or digest <= previous
            or digest in index
        ):
            raise ValueError("artifact index is not unique digest-sorted")
        previous = digest
        index[digest] = item["kind"]
    artifacts = _Artifacts(index, artifact_paths)
    provenance = _exact(
        root["provenance"],
        {
            "repository_commit",
            "producer_source_sha256",
            "verifier_source_sha256",
            "source_artifact_sha256",
            "lifecycle_registry_sha256",
            "membership_manifest_sha256",
            "combined_profile_sha256",
        },
        "manifest provenance",
    )
    if provenance != {name: commit[name] for name in provenance}:
        raise ValueError("manifest provenance drift")
    _profile(profile_path)
    source_folds = _source(source)
    registry = load_symbol_lifecycle_registry(lifecycle_path)
    membership_value, _ = _canonical_json(membership_path)
    membership = validate_fold_membership_manifest(registry, membership_value)
    if (
        not isinstance(root["folds"], list)
        or len(root["folds"]) != len(source_folds)
        or len(source_folds) != len(membership["folds"])
    ):
        raise ValueError("fold count drift")
    prior: list[tuple[Mapping[str, Any], str, set[str]]] = []
    downstream_row_digests: set[str] = set()
    last_cutoff: datetime | None = None
    for manifest_fold, source_fold, member in zip(
        root["folds"], source_folds, membership["folds"], strict=True
    ):
        cutoff = _timestamp(source_fold["input_cutoff_utc"], "source fold cutoff")
        if last_cutoff is not None and cutoff <= last_cutoff:
            raise ValueError("fold cutoff chronology drift")
        _fold(manifest_fold, source_fold, member, prior, artifacts, downstream_row_digests)
        prior.append((source_fold, _sha(source_fold), set(member["eligible_symbols"])))
        last_cutoff = cutoff
    if artifacts.used != set(index):
        raise ValueError("artifact index is not exhaustively attributable")
    return len(prior)


def evaluate_router_replay(
    manifest_path: str | Path,
    *,
    source_artifact_path: str | Path,
    lifecycle_registry_path: str | Path,
    membership_manifest_path: str | Path,
    combined_profile_path: str | Path,
    producer_source_path: str | Path,
    commit_receipt_path: str | Path,
    trusted_source_artifact_sha256: str,
    trusted_commit_receipt_sha256: str,
    artifact_paths: Mapping[str, str | Path],
) -> RouterReplayReport:
    """Verify v2 evidence using required out-of-band source and commit digests."""
    try:
        count = _validate(
            manifest_path,
            source_artifact_path,
            lifecycle_registry_path,
            membership_manifest_path,
            combined_profile_path,
            producer_source_path,
            commit_receipt_path,
            trusted_source_artifact_sha256,
            trusted_commit_receipt_sha256,
            artifact_paths,
        )
        return RouterReplayReport("PASS", (), CANDIDATE_IDS, count)
    except _INPUT_ERRORS as exc:
        return RouterReplayReport(
            "STOP", tuple(sorted({str(exc) or type(exc).__name__})), CANDIDATE_IDS, 0
        )


validate_router_replay = evaluate_router_replay
