"""Read-only, fail-closed validation seam for the frozen G004 router replay."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lumina_quant.data.symbol_lifecycle import (
    load_symbol_lifecycle_registry,
    validate_fold_membership_manifest,
)
from lumina_quant.strategies.registry import resolve_strategy_class

SCHEMA = "router_replay_v1"
SOURCE_SCHEMA = "router_source_v1"
PROFILE_HANDLER = (
    "scripts.research.run_alpha_zoo_69_asset_profile_optuna_hybrid_refit._candidate_from_params"
)
_PROFILE_LEVERAGE_CAPS = {
    "balanced_mdd12_gross5_69_asset_profile_optuna": 6,
    "growth_mdd20_gross8_69_asset_profile_optuna": 10,
    "aggressive_mdd30_gross10_69_asset_profile_optuna": 12,
}
CANDIDATE_IDS = (
    "codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled",
    "codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2",
)
CANDIDATE_IDS_SHA256 = "ddc8996136e70d3847e8270f6165a26992ec8def8439ba6f56e3bcdbdee239b9"
SCALE_GRID = (0.0, 0.5, 0.75, 1.0, 1.1, 1.25, 1.4, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
_HASH = set("0123456789abcdef")
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
)


@dataclass(frozen=True, slots=True)
class RouterReplayReport:
    status: str
    reasons: tuple[str, ...]
    candidate_ids: tuple[str, str]
    fold_count: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON value: {value}")


def _load(path: str | Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
    )


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    output: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise ValueError(f"duplicate YAML key: {key}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _profile(path: str | Path) -> None:
    root = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    if not isinstance(root, Mapping) or root.get("profile") != "backtest_cost_realistic":
        raise ValueError("combined profile identity is invalid")
    research = root.get("research")
    execution = root.get("execution")
    risk = root.get("risk")
    live = root.get("live")
    data = root.get("data")
    if not all(isinstance(item, Mapping) for item in (research, execution, risk, live, data)):
        raise ValueError("combined profile sections are invalid")
    assert isinstance(research, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(risk, Mapping)
    assert isinstance(live, Mapping)
    assert isinstance(data, Mapping)
    required_research = {
        "strict_selection_gate": True,
        "use_lockbox_split": True,
        "purge_embargo_bars": 1,
        "single_correlation_discount": True,
        "hac_inference": True,
        "cscv_pbo": True,
        "exposure_normalized_promotion": True,
        "route_unmapped_registered_strategies": True,
        "require_actual_engine_routing": True,
    }
    if any(research.get(key) != expected for key, expected in required_research.items()):
        raise ValueError("combined profile research contract is invalid")
    if (
        execution.get("slippage_impact_model") != "sqrt_impact"
        or _number(
            execution.get("slippage_impact_coefficient"),
            "slippage impact coefficient",
            positive=True,
        )
        <= 0.0
        or execution.get("funding_interval_hours") != 8
        or execution.get("require_funding_coverage") is not True
        or execution.get("funding_on_utc_boundary") is not True
        or risk.get("attach_default_protective_stop") is not True
        or risk.get("enforce_order_risk_gate_in_backtest") is not True
        or live.get("mode") != "paper"
        or live.get("testnet") is not True
        or live.get("require_real_enable_flag") is not True
        or live.get("allow_market_orders") is not False
        or live.get("shadow_live_enabled") is not False
        or not isinstance(data.get("kinds"), list)
        or "funding" not in data["kinds"]
    ):
        raise ValueError("combined profile safety contract is invalid")


def _exact(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} keys are invalid")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HASH:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _number(value: Any, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _json_finite(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError(f"{name} keys are invalid")
        for key, item in value.items():
            _json_finite(item, f"{name}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_finite(item, f"{name}[{index}]")
        return
    if value is None or isinstance(value, (str, bool)):
        return
    _number(value, name)


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


def _symbols(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(x, str) or not x or x != x.strip() for x in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must contain unique exact symbols")
    return value


def _handler(dotted: Any) -> Path:
    if not isinstance(dotted, str) or "." not in dotted:
        raise ValueError("invalid handler")
    module, _, name = dotted.rpartition(".")
    obj = getattr(importlib.import_module(module), name)
    if not callable(obj):
        raise ValueError("handler is not callable")
    source = inspect.getsourcefile(obj)
    if source is None:
        raise ValueError("handler source unavailable")
    return Path(source)


def _engine_source(row: Mapping[str, Any]) -> Path:
    if row["evaluation_mode"] == "handler":
        return _handler(row["engine_handler"])
    if row["evaluation_mode"] != "registry_simulator":
        raise ValueError("invalid evaluation mode")
    if row["engine_handler"] != "registry_simulator":
        raise ValueError("registry simulator handler invalid")
    klass = resolve_strategy_class(row["strategy_class"], strict=True)
    source = inspect.getsourcefile(klass)
    if source is None:
        raise ValueError("registry class source unavailable")
    return Path(source)


def _profile_params(params: Any, profile_id: str) -> None:
    if not isinstance(params, Mapping):
        raise ValueError("params must be an object")
    common = {"family", "timeframe", "side", "integer_leverage", "min_hold_bars", "cooldown_bars"}
    fields = {
        "cross_sectional_momentum_rank": common
        | {"lookback_bars", "threshold", "exit_threshold", "market_guard", "breadth_guard"},
        "volatility_adjusted_trend_persistence": common
        | {"lookback_bars", "threshold", "exit_threshold", "adx_min", "market_abs_max"},
        "trend_pullback_reclaim": common
        | {
            "lookback_bars",
            "fast_divisor",
            "threshold",
            "exit_threshold",
            "trend_slope_min",
            "market_guard",
        },
    }
    family = params.get("family")
    if family not in fields or set(params) != fields[family]:
        raise ValueError("unknown family or parameter keys")
    if not isinstance(params["timeframe"], str) or params["timeframe"] not in {
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    }:
        raise ValueError("invalid timeframe")
    if params["side"] not in {"long_short", "long_only", "short_only"}:
        raise ValueError("invalid side")
    leverage_cap = _PROFILE_LEVERAGE_CAPS.get(profile_id)
    if leverage_cap is None:
        raise ValueError("unknown source profile")
    for key, allowed in (
        ("integer_leverage", range(1, leverage_cap + 1)),
        ("min_hold_bars", range(6, 73, 6)),
        ("cooldown_bars", range(0, 19, 3)),
    ):
        if type(params[key]) is not int or params[key] not in allowed:
            raise ValueError(f"invalid {key}")
    if type(params["lookback_bars"]) is not int or params["lookback_bars"] not in (
        {6, 12, 24, 48, 72} if family != "trend_pullback_reclaim" else {24, 36, 48, 72, 96, 144}
    ):
        raise ValueError("invalid lookback")
    numeric_ranges = {
        "cross_sectional_momentum_rank": {
            "threshold": (0.05, 0.30),
            "exit_threshold": (0.35, 0.65),
            "market_guard": (0.0, 0.08),
            "breadth_guard": (0.20, 0.50),
        },
        "volatility_adjusted_trend_persistence": {
            "threshold": (0.40, 2.00),
            "exit_threshold": (0.05, 0.60),
            "adx_min": (5.0, 30.0),
            "market_abs_max": (0.08, 0.40),
        },
        "trend_pullback_reclaim": {
            "threshold": (-2.00, -0.25),
            "exit_threshold": (0.0, 0.0),
            "trend_slope_min": (0.0, 0.03),
            "market_guard": (0.0, 0.08),
        },
    }
    for key, (low, high) in numeric_ranges[family].items():
        if type(params[key]) is not float or not low <= params[key] <= high:
            raise ValueError(f"invalid {key}")
    allowed_numbers = {
        "cross_sectional_momentum_rank": {
            "threshold": {0.05, 0.10, 0.15, 0.20, 0.25, 0.30},
            "exit_threshold": {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65},
            "market_guard": {0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08},
            "breadth_guard": {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50},
        },
        "volatility_adjusted_trend_persistence": {
            "threshold": {round(0.40 + 0.10 * i, 2) for i in range(17)},
            "exit_threshold": {round(0.05 + 0.05 * i, 2) for i in range(12)},
            "adx_min": {5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0},
            "market_abs_max": {0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40},
        },
        "trend_pullback_reclaim": {
            "threshold": {round(-2.00 + 0.25 * i, 2) for i in range(8)},
            "exit_threshold": {0.0},
            "trend_slope_min": {0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03},
            "market_guard": {0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08},
        },
    }
    if any(params[key] not in allowed_numbers[family][key] for key in allowed_numbers[family]):
        raise ValueError("invalid parameter step")
    if family == "trend_pullback_reclaim" and (
        type(params["fast_divisor"]) is not int or params["fast_divisor"] not in {3, 4, 6}
    ):
        raise ValueError("invalid fast_divisor")


def _leaf(value: Any, eligible: set[str]) -> str:
    row = _exact(
        value,
        {
            "leaf_id",
            "profile_id",
            "strategy_class",
            "engine_handler",
            "params",
            "traded_symbols",
            "dependency_symbols",
            "native_timeframe",
            "allocation_fraction",
            "source_weight",
            "native_gross",
            "evaluation_mode",
            "engine_source_sha256",
            "source_row_sha256",
        },
        "leaf",
    )
    if any(
        not isinstance(row[k], str) or not row[k] or row[k] != row[k].strip()
        for k in ("leaf_id", "profile_id", "strategy_class", "engine_handler", "native_timeframe")
    ):
        raise ValueError("leaf identity invalid")
    if not isinstance(row["params"], Mapping):
        raise ValueError("leaf params must be an object")
    _json_finite(row["params"], "params")
    if row["evaluation_mode"] == "handler":
        if row["engine_handler"] != PROFILE_HANDLER:
            raise ValueError("unregistered actual-engine handler")
        _profile_params(row["params"], row["profile_id"])
        if row["strategy_class"] != row["params"]["family"]:
            raise ValueError("leaf strategy class does not bind params")
    elif row["evaluation_mode"] != "registry_simulator":
        raise ValueError("invalid evaluation mode")
    if row["params"].get("timeframe") is not None and (
        row["native_timeframe"] != row["params"]["timeframe"]
    ):
        raise ValueError("leaf timeframe does not bind params")
    traded, deps = (
        _symbols(row["traded_symbols"], "traded_symbols"),
        _symbols(row["dependency_symbols"], "dependency_symbols"),
    )
    if not set(traded).issubset(deps) or not set(deps).issubset(eligible):
        raise ValueError("dependency is partial or inactive")
    for key in ("allocation_fraction", "source_weight", "native_gross"):
        _number(row[key], key, True)
    _digest(row["engine_source_sha256"], "engine source")
    _digest(row["source_row_sha256"], "source row")
    if _file(_engine_source(row)) != row["engine_source_sha256"]:
        raise ValueError("engine source hash drift")
    if _sha({k: v for k, v in row.items() if k != "source_row_sha256"}) != row["source_row_sha256"]:
        raise ValueError("row hash drift")
    return row["leaf_id"]


def _receipts(value: Any, leaves: list[Any]) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != len(leaves):
        raise ValueError("execution receipt count invalid")
    rows = []
    for base, receipt in zip(leaves, value, strict=True):
        row = _exact(
            receipt,
            {
                "leaf_id",
                "evaluation_mode",
                "engine_source_sha256",
                "signal_receipt_sha256",
                "position_receipt_sha256",
                "engine_receipt_sha256",
                "generic_fallback_proxy_count",
                "current_fold_oos_input_count",
            },
            "execution receipt",
        )
        if (
            row["leaf_id"] != base["leaf_id"]
            or row["evaluation_mode"] != base["evaluation_mode"]
            or row["engine_source_sha256"] != base["engine_source_sha256"]
            or row["generic_fallback_proxy_count"] != 0
            or row["current_fold_oos_input_count"] != 0
            or type(row["generic_fallback_proxy_count"]) is not int
            or type(row["current_fold_oos_input_count"]) is not int
        ):
            raise ValueError("execution receipt unsafe")
        for key in ("signal_receipt_sha256", "position_receipt_sha256", "engine_receipt_sha256"):
            _digest(row[key], key)
        rows.append(row)
    return rows


def _variants(value: Any, selection: Mapping[str, Any]) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("requires two variants")
    parsed: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for row, candidate, mdd, cap in zip(
        value, CANDIDATE_IDS, (0.30, 0.20), (3.0, 2.0), strict=True
    ):
        v = _exact(
            row,
            {
                "variant_id",
                "selected_label",
                "base_leaf_list_sha256",
                "policy",
                "applied_scale",
                "leaves",
                "execution_receipts",
            },
            "variant",
        )
        if (
            v["variant_id"] != candidate
            or v["selected_label"] != selection["selected_label"]
            or v["base_leaf_list_sha256"] != selection["leaf_list_sha256"]
            or _exact(v["policy"], {"fallback_mdd", "fallback_cap"}, "policy")
            != {"fallback_mdd": mdd, "fallback_cap": cap}
        ):
            raise ValueError("variant parity drift")
        scale = _number(v["applied_scale"], "applied_scale")
        cash = selection["branch"] == "strict_core_cash"
        if cash:
            if scale != 0 or v["leaves"] or v["execution_receipts"]:
                raise ValueError("cash variant unsafe")
            parsed.append((v, []))
            continue
        if (
            scale not in SCALE_GRID
            or scale > cap
            or (selection["branch"] == "pre_registered_lagged_plus_validation_leaf" and scale != 1)
        ):
            raise ValueError("invalid applied scale")
        if not isinstance(v["leaves"], list) or len(v["leaves"]) != len(selection["leaves"]):
            raise ValueError("variant leaves invalid")
        for base, effective in zip(selection["leaves"], v["leaves"], strict=True):
            e = _exact(
                effective, {"leaf_id", "effective_weight", "effective_gross"}, "effective leaf"
            )
            if (
                e["leaf_id"] != base["leaf_id"]
                or _number(e["effective_weight"], "effective_weight")
                != _number(base["source_weight"], "source_weight") * scale
                or _number(e["effective_gross"], "effective_gross")
                != _number(base["native_gross"], "native_gross") * scale
            ):
                raise ValueError("effective leaf drift")
        parsed.append((v, _receipts(v["execution_receipts"], selection["leaves"])))
    if selection["branch"] == "pre_registered_lagged_plus_validation_leaf":
        left, right = parsed
        if left[0]["leaves"] != right[0]["leaves"] or any(
            a["signal_receipt_sha256"] != b["signal_receipt_sha256"]
            or a["position_receipt_sha256"] != b["position_receipt_sha256"]
            or a["engine_receipt_sha256"] != b["engine_receipt_sha256"]
            for a, b in zip(left[1], right[1], strict=True)
        ):
            raise ValueError("mature variant receipts differ")
    elif selection["branch"] == "strict_core_scaled":
        left, right = parsed
        if any(
            a["signal_receipt_sha256"] != b["signal_receipt_sha256"]
            for a, b in zip(left[1], right[1], strict=True)
        ):
            raise ValueError("fallback invariant signal receipts differ")


def _fold(value: Any, member: Mapping[str, Any], prior_ids: set[str]) -> None:
    row = _exact(
        value,
        {
            "fold_id",
            "locked_oos",
            "input_cutoff_utc",
            "decision_timestamp_utc",
            "membership_fold_sha256",
            "selection",
            "variants",
        },
        "fold",
    )
    if row["fold_id"] != member["fold_id"] or _sha(member) != _digest(
        row["membership_fold_sha256"], "membership fold"
    ):
        raise ValueError("membership fold drift")
    start, end = (
        datetime.fromtimestamp(member["start_ms"] / 1000, UTC),
        datetime.fromtimestamp(member["end_ms"] / 1000, UTC),
    )
    oos = _exact(row["locked_oos"], {"start_utc", "end_utc"}, "locked_oos")
    cutoff, decision = (
        _timestamp(row["input_cutoff_utc"], "cutoff"),
        _timestamp(row["decision_timestamp_utc"], "decision"),
    )
    if (
        _timestamp(oos["start_utc"], "oos start") != start
        or _timestamp(oos["end_utc"], "oos end") != end
        or not cutoff < decision
        or decision != start
    ):
        raise ValueError("fold interval drift")
    selection = _exact(
        row["selection"],
        {
            "branch",
            "selected_label",
            "selection_inputs_sha256",
            "current_fold_oos_input_count",
            "history_receipts",
            "leaves",
            "leaf_list_sha256",
        },
        "selection",
    )
    if (
        selection["branch"]
        not in {
            "pre_registered_lagged_plus_validation_leaf",
            "strict_core_scaled",
            "strict_core_cash",
        }
        or selection["current_fold_oos_input_count"] != 0
        or type(selection["current_fold_oos_input_count"]) is not int
    ):
        raise ValueError("selection unsafe")
    leaves, history = selection["leaves"], selection["history_receipts"]
    cash = selection["branch"] == "strict_core_cash"
    if cash and (selection["selected_label"] is not None or leaves):
        raise ValueError("cash selection unsafe")
    if not cash and (
        not isinstance(selection["selected_label"], str)
        or not selection["selected_label"]
        or not isinstance(leaves, list)
        or not leaves
    ):
        raise ValueError("noncash selection unsafe")
    if (
        not isinstance(leaves, list)
        or len({_leaf(x, set(member["eligible_symbols"])) for x in leaves}) != len(leaves)
        or _sha(leaves) != _digest(selection["leaf_list_sha256"], "leaf list")
    ):
        raise ValueError("leaf list drift")
    if not isinstance(history, list):
        raise ValueError("history invalid")
    seen: set[tuple[str, str]] = set()
    history_order: list[tuple[datetime, str, str]] = []
    for item in history:
        h = _exact(
            item,
            {
                "fold_id",
                "candidate_label",
                "completed_at_utc",
                "input_cutoff_utc",
                "return",
                "source_sha256",
            },
            "history",
        )
        identity = (h["fold_id"], h["candidate_label"])
        completed_at = _timestamp(h["completed_at_utc"], "completed_at_utc")
        history_cutoff = _timestamp(h["input_cutoff_utc"], "history input_cutoff_utc")
        if (
            h["fold_id"] not in prior_ids
            or identity in seen
            or not isinstance(h["candidate_label"], str)
            or not h["candidate_label"]
            or completed_at >= cutoff
            or history_cutoff >= cutoff
            or history_cutoff > completed_at
        ):
            raise ValueError("history is current or future")
        seen.add(identity)
        history_order.append((completed_at, h["fold_id"], h["candidate_label"]))
        _number(h["return"], "history return")
        _digest(h["source_sha256"], "history source")
    if history_order != sorted(history_order):
        raise ValueError("history receipts are unordered")
    if _sha(
        {
            "branch": selection["branch"],
            "selected_label": selection["selected_label"],
            "history_receipts": history,
            "leaves": leaves,
        }
    ) != _digest(selection["selection_inputs_sha256"], "selection inputs"):
        raise ValueError("selection digest drift")
    _variants(row["variants"], selection)


def _source_fold(fold: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: fold[key]
        for key in (
            "fold_id",
            "locked_oos",
            "input_cutoff_utc",
            "decision_timestamp_utc",
            "membership_fold_sha256",
            "selection",
        )
    }


def _source_contract(source: Any, manifest_folds: list[Any]) -> None:
    root = _exact(
        source,
        {
            "schema",
            "candidate_ids",
            "candidate_ids_sha256",
            "controls",
            "frozen_at_utc",
            "folds",
        },
        "router source",
    )
    if (
        root["schema"] != SOURCE_SCHEMA
        or tuple(root["candidate_ids"]) != CANDIDATE_IDS
        or root["candidate_ids_sha256"] != CANDIDATE_IDS_SHA256
    ):
        raise ValueError("router source candidate identity drift")
    controls = _exact(
        root["controls"],
        {
            "new_grid_search",
            "recompute_from_json",
            "post_oos_augment",
            "post_oos_research_variant",
        },
        "router source controls",
    )
    if controls != {
        "new_grid_search": False,
        "recompute_from_json": False,
        "post_oos_augment": False,
        "post_oos_research_variant": True,
    }:
        raise ValueError("router source controls are unsafe")
    if any(
        controls[key] is not expected
        for key, expected in (
            ("new_grid_search", False),
            ("recompute_from_json", False),
            ("post_oos_augment", False),
            ("post_oos_research_variant", True),
        )
    ):
        raise ValueError("router source control types are unsafe")
    _timestamp(root["frozen_at_utc"], "router source frozen_at_utc")
    source_folds = root["folds"]
    if not isinstance(source_folds, list) or not all(
        isinstance(fold, Mapping) for fold in source_folds
    ):
        raise ValueError("router source folds are invalid")
    expected = [_source_fold(fold) for fold in manifest_folds]
    if _bytes(source_folds) != _bytes(expected):
        raise ValueError("router source fold content drift")


def _validate(manifest: Mapping[str, Any], files: Mapping[str, str | Path]) -> int:
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
        raise ValueError("candidate list drift")
    if _exact(
        root["controls"],
        {
            "new_grid_search",
            "recompute_from_json",
            "post_oos_augment",
            "real_money_enabled",
            "orders_submitted",
            "capital_allocated",
        },
        "controls",
    ) != {
        "new_grid_search": False,
        "recompute_from_json": False,
        "post_oos_augment": False,
        "real_money_enabled": False,
        "orders_submitted": 0,
        "capital_allocated": 0,
    }:
        raise ValueError("unsafe controls")
    controls = root["controls"]
    if any(
        controls[key] is not False
        for key in (
            "new_grid_search",
            "recompute_from_json",
            "post_oos_augment",
            "real_money_enabled",
        )
    ) or any(type(controls[key]) is not int for key in ("orders_submitted", "capital_allocated")):
        raise ValueError("unsafe control types")
    prov = _exact(
        root["provenance"],
        {
            "producer_sha256",
            "verifier_version_sha256",
            "commit_receipt_sha256",
            "source_artifact_sha256",
            "lifecycle_registry_sha256",
            "membership_manifest_sha256",
            "combined_profile_sha256",
        },
        "provenance",
    )
    for k in prov:
        _digest(prov[k], k)
    for key, path in (
        ("producer_sha256", "producer_source"),
        ("commit_receipt_sha256", "commit_receipt"),
        ("source_artifact_sha256", "source_artifact"),
        ("lifecycle_registry_sha256", "lifecycle_registry"),
        ("membership_manifest_sha256", "membership_manifest"),
        ("combined_profile_sha256", "combined_profile"),
    ):
        if _file(files[path]) != prov[key]:
            raise ValueError(f"{key} drift")
    if _file(__file__) != prov["verifier_version_sha256"]:
        raise ValueError("verifier source drift")
    _profile(files["combined_profile"])
    if not isinstance(root["folds"], list):
        raise ValueError("manifest folds are invalid")
    _source_contract(_load(files["source_artifact"]), root["folds"])
    registry = load_symbol_lifecycle_registry(files["lifecycle_registry"])
    membership = validate_fold_membership_manifest(registry, _load(files["membership_manifest"]))
    if len(root["folds"]) != len(membership["folds"]):
        raise ValueError("fold count drift")
    prior: set[str] = set()
    for fold, member in zip(root["folds"], membership["folds"], strict=True):
        _fold(fold, member, prior)
        prior.add(member["fold_id"])
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
) -> RouterReplayReport:
    try:
        count = _validate(
            _load(manifest_path),
            {
                "source_artifact": source_artifact_path,
                "lifecycle_registry": lifecycle_registry_path,
                "membership_manifest": membership_manifest_path,
                "combined_profile": combined_profile_path,
                "producer_source": producer_source_path,
                "commit_receipt": commit_receipt_path,
            },
        )
        return RouterReplayReport("PASS", (), CANDIDATE_IDS, count)
    except _INPUT_ERRORS as exc:
        return RouterReplayReport(
            "STOP", tuple(sorted({str(exc) or type(exc).__name__})), CANDIDATE_IDS, 0
        )


validate_router_replay = evaluate_router_replay
