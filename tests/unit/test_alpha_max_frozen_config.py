from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
CONTRACT_MANIFEST_PATH = (
    REPO_ROOT / "configs/research/alpha_max_contract_manifest_20260711_listing_aware.json"
)
AVAILABILITY_EVIDENCE_PATH = (
    REPO_ROOT / "configs/research/alpha_max_official_availability_evidence_20260711.json"
)
PLAN_ROOT = REPO_ROOT / ".omx/plans"
CURRENT_REGISTRY_PATH = PLAN_ROOT / "alpha-max-current-trial-nodes-v1.json"
INCUMBENT_AUDIT_PATH = PLAN_ROOT / "alpha-max-incumbent-resolution-v1.json"
PRIOR_MANIFEST_PATH = (
    REPO_ROOT / "var/reports/ultragoal_full_pool_strategy/g004_frozen_candidate_manifest.json"
)

BASELINE_COMMIT = "252910e54e280cc593365484cbc99d6ca87893f9"
CURRENT_REGISTRY_SHA256 = "cfe3a04620c52cc235d6f1cda1cac617ba30cd7327c753fc2f620d8250d51a4e"
CURRENT_KEY_SET_SHA256 = "3a4791cf353abcb82f9717ce89ee16b9d73d84f431d5b058135046c2ba8e332b"
PRIOR_KEY_SET_ACTUAL_LF_SHA256 = "3b078011040f89e8d788b2cef9214c58f687221104381e26a688a7f8cdbddd78"
INCUMBENT_AUDIT_SHA256 = "5133bc40116399fe7af32e75a1ecc52a4f385dc8a0b5d3a4a9585e2437615ed8"
RUNTIME_CONTRACT_SHA256 = "a6c945e43870c3d45e0f5f745e689eb75d68d82e0da07d22918df29e14ada753"
CONFIG_PAYLOAD_SHA256 = "44cc454556f11d5bf66f8992e41343b0ada0ca8803a771342756057138e3cd44"
CONFIG_CANONICAL_SHA256 = "3f54ec513402204602fc14233e59f3be0dcbef4f2e89f7cd065133305d023f1c"
CONFIG_FILE_SHA256 = "01d1c783d8393966d356024ab41349b540339184384eb72ef2952ae11f4dad04"

CANDIDATE_SYMBOLS = [
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
]

TOP_LEVEL_KEYS = {
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

RUNTIME_KEYS = {
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

EXPECTED_OVERRIDE_POLICY = {
    "any_environment_key_prefix_rejected": "LQ_",
    "config_yaml_loaded": False,
    "default_runtime_config_loaded": False,
    "environment_values_loaded": False,
    "merge_layer_count": 0,
    "profile_loaded": False,
    "runtime_override_loaded": False,
    "unknown_cli_arguments_rejected": True,
}


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite_constant,
    )


def _strict_load(path: Path) -> Any:
    return _strict_loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _assert_finite_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite numeric value at {path}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value)!r}")


def _validate_declared_config_surface(config: dict[str, Any]) -> None:
    if set(config) != TOP_LEVEL_KEYS:
        raise ValueError("undeclared top-level config surface")
    runtime = config["runtime_contract"]
    if set(runtime) != RUNTIME_KEYS:
        raise ValueError("undeclared runtime-contract surface")
    if runtime["override_policy"] != EXPECTED_OVERRIDE_POLICY:
        raise ValueError("runtime override policy is not frozen")

    static_keys = set(runtime["static_attributes"])
    derived_keys = set(runtime["derived_attributes"])
    allowlist = runtime["attribute_allowlist"]
    if allowlist != sorted(allowlist):
        raise ValueError("runtime attribute allowlist is not canonical")
    if set(allowlist) != static_keys | derived_keys:
        raise ValueError("undeclared or missing runtime attribute")
    if static_keys & derived_keys:
        raise ValueError("runtime attribute has two sources")
    if any(not key.isupper() for key in allowlist):
        raise ValueError("runtime attribute is not uppercase")

    _assert_finite_tree(config)


def _trial_key(node: dict[str, Any]) -> str:
    return _sha256(_canonical_bytes(node))


def _actual_lf_set_bytes(keys: list[str]) -> bytes:
    return ("\n".join(sorted(keys)) + "\n").encode("ascii")


def _normalize_prior_node(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "allocation": {},
        "behavior_metadata": candidate.get("metadata", {}),
        "gross": None,
        "implementation": candidate.get("strategy_class") or candidate.get("strategy"),
        "kind": "prior_strategy_leaf",
        "members": [],
        "omission": None,
        "params": candidate.get("params", {}),
        "schema": "alpha_max_trial_node.v1",
        "symbols": sorted(
            str(symbol).upper().replace("/", "") for symbol in candidate.get("symbols", [])
        ),
        "timeframe": candidate.get("strategy_timeframe") or candidate.get("timeframe"),
    }


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo == UTC
    return parsed


def test_strict_json_loader_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    config = _strict_load(CONFIG_PATH)
    assert config["schema_version"] == "alpha_max_portfolio_experiment.v1"

    with pytest.raises(ValueError, match="duplicate JSON key"):
        _strict_loads('{"runtime_contract":{},"runtime_contract":{}}')

    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="non-finite JSON constant"):
            _strict_loads(f'{{"value":{token}}}')


def test_config_schema_and_canonical_hash_are_frozen() -> None:
    config = _strict_load(CONFIG_PATH)
    _validate_declared_config_surface(config)

    assert config["experiment_id"] == "alpha_max_portfolio_20260711_listing_aware"
    assert config["revision"] == "5.15"
    assert _sha256(CONFIG_PATH.read_bytes()) == CONFIG_FILE_SHA256
    assert _sha256(_canonical_bytes(config)) == CONFIG_CANONICAL_SHA256

    hash_scope = copy.deepcopy(config)
    embedded_payload_hash = hash_scope["integrity"].pop("config_payload_sha256")
    assert embedded_payload_hash == CONFIG_PAYLOAD_SHA256
    assert _sha256(_canonical_bytes(hash_scope)) == CONFIG_PAYLOAD_SHA256

    runtime = config["runtime_contract"]
    assert config["integrity"]["runtime_contract_sha256"] == RUNTIME_CONTRACT_SHA256
    assert _sha256(_canonical_bytes(runtime)) == RUNTIME_CONTRACT_SHA256


def test_normative_artifacts_and_embedded_values_match_exactly() -> None:
    config = _strict_load(CONFIG_PATH)
    normative = config["normative_sources"]
    expected_sources = {
        "availability_source_evidence_sha256": AVAILABILITY_EVIDENCE_PATH,
        "architect_review_sha256": (
            PLAN_ROOT / "architect-review-alpha-max-independent-20260710-revision5.14.md"
        ),
        "consensus_sha256": (PLAN_ROOT / "ralplan-consensus-alpha-max-independent-20260710.json"),
        "critic_review_sha256": (
            PLAN_ROOT / "critic-review-alpha-max-independent-20260710-revision5.14.md"
        ),
        "current_trial_registry_sha256": CURRENT_REGISTRY_PATH,
        "incumbent_resolution_audit_sha256": INCUMBENT_AUDIT_PATH,
        "plan_sha256": PLAN_ROOT / "ralplan-alpha-max-independent-20260710.md",
        "prd_sha256": PLAN_ROOT / "prd-alpha-max-independent-20260710.md",
        "test_spec_sha256": PLAN_ROOT / "test-spec-alpha-max-independent-20260710.md",
    }
    for key, path in expected_sources.items():
        assert normative[key] == _sha256(path.read_bytes())
    assert normative["availability_correction_plan_sha256"] == _sha256(
        (PLAN_ROOT / "alpha-max-listing-aware-correction-20260711.md").read_bytes()
    )
    assert normative["availability_correction_test_spec_sha256"] == _sha256(
        (PLAN_ROOT / "test-spec-alpha-max-listing-aware-correction-20260711.md").read_bytes()
    )

    assert normative["baseline_commit"] == BASELINE_COMMIT
    assert normative["current_trial_key_set_sha256"] == CURRENT_KEY_SET_SHA256
    assert normative["prior_trial_key_set_actual_lf_sha256"] == (PRIOR_KEY_SET_ACTUAL_LF_SHA256)
    assert config["current_trial_registry"] == _strict_load(CURRENT_REGISTRY_PATH)
    assert config["incumbent_resolution"] == _strict_load(INCUMBENT_AUDIT_PATH)


def test_closed_trial_ledger_recomputes_actual_lf_and_current_key_sets() -> None:
    config = _strict_load(CONFIG_PATH)
    prior_manifest = _strict_load(PRIOR_MANIFEST_PATH)
    assert (
        _sha256(PRIOR_MANIFEST_PATH.read_bytes())
        == (config["trial_ledger"]["prior_inventory"]["file_sha256"])
    )
    assert prior_manifest["candidate_count"] == len(prior_manifest["candidates"]) == 1466
    assert prior_manifest["candidate_manifest_sha256"] == (
        "1292498b3b729038c74932175a12d910fc4351b2feb3bbfc95f827517e423efe"
    )
    assert prior_manifest["candidate_set_sha256"] == (
        "01ca7a5c04b490b5472a62b49d0fcc7d432f0e2045c0e6fae9b1bfcb079a0564"
    )

    prior_keys = [_trial_key(_normalize_prior_node(row)) for row in prior_manifest["candidates"]]
    assert len(prior_keys) == len(set(prior_keys)) == 1466
    prior_lf_bytes = _actual_lf_set_bytes(prior_keys)
    assert prior_lf_bytes.endswith(b"\x0a")
    assert b"\\n" not in prior_lf_bytes
    assert _sha256(prior_lf_bytes) == PRIOR_KEY_SET_ACTUAL_LF_SHA256

    registry = config["current_trial_registry"]
    current_keys = [_trial_key(node) for node in registry["nodes"]]
    assert len(current_keys) == len(set(current_keys)) == 21
    assert _sha256(_actual_lf_set_bytes(current_keys)) == CURRENT_KEY_SET_SHA256
    assert registry["current_key_set_sha256"] == CURRENT_KEY_SET_SHA256
    assert _sha256(CURRENT_REGISTRY_PATH.read_bytes()) == CURRENT_REGISTRY_SHA256

    assert config["trial_ledger"]["union_formula"] == "1466+21=1487"
    assert config["trial_ledger"]["dsr_num_trials"] == 1487
    assert config["metrics_and_statistics"]["statistical_gates"]["dsr"]["num_trials"] == 1487
    assert config["trial_ledger"]["cost_cells_are_trials"] is False


def test_candidate_admission_calendars_and_native_contract_are_exact() -> None:
    config = _strict_load(CONFIG_PATH)
    assert config["candidate_symbols"] == CANDIDATE_SYMBOLS
    assert all(
        node["symbols"] == CANDIDATE_SYMBOLS for node in config["current_trial_registry"]["nodes"]
    )

    manifest = config["contract_manifest_contract"]
    assert manifest["schema_version"] == "alpha_max_contract_manifest.v2"
    assert manifest["exchange"] == "binance"
    assert manifest["availability_interval_granularity"] == "exact_utc_millisecond"
    assert manifest["availability_interval_semantics"] == (
        "kind_specific_half_open_official_source_interval;root_observed_inference_forbidden"
    )
    assert [record["symbol"] for record in manifest["records"]] == CANDIDATE_SYMBOLS
    for record in manifest["records"]:
        assert record == {
            "contract_multiplier": 1.0,
            "feature_availability_end_utc": (
                "2026-06-23T09:00:00Z" if record["symbol"] == "TONUSDT" else "2026-07-01T00:00:00Z"
            ),
            "feature_availability_start_utc": (
                "2024-03-01T16:00:00Z" if record["symbol"] == "TONUSDT" else "2022-12-31T00:00:00Z"
            ),
            "inverse": False,
            "linear": True,
            "margin_asset": "USDT",
            "market_type": "perpetual",
            "quote_asset": "USDT",
            "raw_availability_end_utc": (
                "2026-06-23T09:00:00Z" if record["symbol"] == "TONUSDT" else "2026-07-01T00:00:00Z"
            ),
            "raw_availability_start_utc": (
                "2024-03-01T12:31:10Z" if record["symbol"] == "TONUSDT" else "2022-12-31T00:00:00Z"
            ),
            "settle_asset": "USDT",
            "symbol": record["symbol"],
            "volume_unit": "base_asset",
        }

    external_manifest = _strict_load(CONTRACT_MANIFEST_PATH)
    embedded_manifest = {
        "exchange": manifest["exchange"],
        "records": manifest["records"],
        "schema_version": manifest["schema_version"],
    }
    assert external_manifest == embedded_manifest
    assert CONTRACT_MANIFEST_PATH.read_bytes() == _canonical_bytes(embedded_manifest) + b"\n"

    evidence = _strict_load(AVAILABILITY_EVIDENCE_PATH)
    ton_contract = next(
        record for record in external_manifest["records"] if record["symbol"] == "TONUSDT"
    )
    for kind in ("raw", "feature"):
        owned = evidence["owned_intervals"][kind]
        assert _parse_utc(owned["start_utc"]) == _parse_utc(
            ton_contract[f"{kind}_availability_start_utc"]
        )
        assert _parse_utc(owned["end_utc"]) == _parse_utc(
            ton_contract[f"{kind}_availability_end_utc"]
        )
    assert evidence["owned_intervals"]["raw"]["start_utc"] == "2024-03-01T12:31:10Z"
    assert evidence["owned_intervals"]["feature"]["start_utc"] == "2024-03-01T16:00:00Z"
    assert evidence["tonusdt_funding"]["first"]["funding_time_utc"] == ("2024-03-01T08:00:00.000Z")
    transition = evidence["tonusdt_funding"]["listing_transition"]
    assert transition["official_onboard_time_utc"] == "2024-03-01T12:30:00.000Z"
    assert transition["missing_nominal_settlement"] == {
        "funding_time_ms": 1_709_294_400_000,
        "funding_time_utc": "2024-03-01T12:00:00.000Z",
        "present_in_official_response": False,
        "synthesis_forbidden": True,
    }
    assert transition["first_post_onboard_continuous_point"]["funding_time_utc"] == (
        "2024-03-01T16:00:00.000Z"
    )
    returned_times = {
        row["funding_time_ms"] for row in transition["official_query_returned_points"]
    }
    assert 1_709_280_000_000 in returned_times
    assert 1_709_294_400_000 not in returned_times
    assert 1_709_308_800_000 in returned_times

    funding = config["funding_sidecar_and_settlement"]
    assert funding["sealed_parquet_contract"] == {
        "canonical_settlement_collision_policy": "reject",
        "canonical_settlement_timestamp_column": "timestamp_ms",
        "canonical_settlement_timestamp_semantics": "utc_nominal_funding_grid_epoch_milliseconds",
        "official_source_timestamp_column": "source_timestamp_ms",
        "official_source_timestamp_semantics": (
            "official_binance_fundingTime_utc_epoch_milliseconds"
        ),
        "required_columns": [
            "timestamp_ms",
            "source_timestamp_ms",
            "exchange",
            "symbol",
            "funding_rate",
        ],
        "source_minus_settlement_jitter_milliseconds": {
            "maximum_inclusive": 1000,
            "minimum_inclusive": 0,
        },
        "timestamp_duplicate_policy": "reject",
    }
    expected_cadence = {
        symbol: 14_400_000 if symbol == "TONUSDT" else 28_800_000 for symbol in CANDIDATE_SYMBOLS
    }
    assert funding["funding_cadence_milliseconds_by_symbol"] == expected_cadence
    assert funding["eight_hour_resolver_admission"] == {
        "cadence_milliseconds": 28_800_000,
        "exact_cadence_required": True,
        "tonusdt_forbidden": True,
    }

    admission = config["admission"]
    assert admission["daily_quote_notional"]["day_count"] == 517
    assert admission["thresholds"] == {
        "median_quote_notional_usdt_minimum": 20_000_000.0,
        "p10_quote_notional_usdt_minimum": 2_000_000.0,
    }
    assert admission["type7_quantile"]["probabilities"] == [0.1, 0.5]
    assert admission["admitted_symbols_artifact"]["minimum_count"] == 5
    assert admission["admitted_symbols_artifact"]["maximum_count"] == 10
    assert admission["admitted_symbols_artifact"]["trial_node_mutation_forbidden"] is True

    chronology = config["chronology"]
    splits = chronology["splits"]
    assert [(row["split_id"], row["start_utc"], row["end_utc"]) for row in splits] == [
        ("warmup", "2022-12-31T00:00:00Z", "2024-01-01T00:00:00Z"),
        ("train", "2024-01-01T00:00:00Z", "2025-06-01T00:00:00Z"),
        ("purge", "2025-06-01T00:00:00Z", "2025-06-08T00:00:00Z"),
        ("validation", "2025-06-08T00:00:00Z", "2025-08-31T00:00:00Z"),
        ("embargo", "2025-08-31T00:00:00Z", "2025-09-07T00:00:00Z"),
        (
            "historical_exposed_evaluation",
            "2025-09-07T00:00:00Z",
            "2026-07-01T00:00:00Z",
        ),
    ]

    validation_folds = chronology["validation_folds"]
    assert len(validation_folds) == 12
    assert [row["fold_id"] for row in validation_folds] == [
        f"validation_w{index:02d}" for index in range(1, 13)
    ]
    for prior, current in pairwise(validation_folds):
        assert prior["end_utc"] == current["start_utc"]
    assert all(
        _parse_utc(row["end_utc"]) - _parse_utc(row["start_utc"]) == timedelta(days=7)
        for row in validation_folds
    )

    historical_folds = chronology["historical_evaluation_folds"]
    assert len(historical_folds) == 10
    assert historical_folds[0]["start_utc"] == "2025-09-07T00:00:00Z"
    assert historical_folds[-1]["end_utc"] == "2026-07-01T00:00:00Z"
    for prior, current in pairwise(historical_folds):
        assert prior["end_utc"] == current["start_utc"]

    native = config["native_timeframes_and_warmup"]
    assert native["minimum_completed_bars"] == {"1d": 366, "4h": 64}
    assert native["base_replay_timeframe"] == "1s"
    assert native["reporting_timeframe"] == "4h"
    assert native["two_engine_protocol"] is True
    assert native["indicator_only_capsule"] is True
    assert native["scoring_engine_starts_flat"] is True
    assert native["warmup_economic_events_allowed"] is False


def test_cost_allocator_gross_and_refit_matrix_are_frozen() -> None:
    config = _strict_load(CONFIG_PATH)
    cells = config["cost_cells"]
    assert [cell["nominal_one_way_bps"] for cell in cells] == [10, 15, 20, 30]
    assert [cell["slippage_rate"] for cell in cells] == [0.0005, 0.001, 0.0015, 0.0025]
    assert all(cell["taker_fee_bps"] == 4 for cell in cells)
    assert all(cell["maker_fee_bps"] == 2 for cell in cells)
    assert all(cell["half_spread_bps"] == 1 for cell in cells)
    assert [cell["selection_reference"] for cell in cells] == [False, False, False, True]

    allocation = config["allocation_and_refit"]
    assert allocation["allocation_input"] == {
        "calendar_rule": "exact_inner_equality_no_imputation",
        "component_id_order": "lexicographic",
        "cost_cell_bps": 20,
        "minimum_observations": 252,
        "return_frequency": "1d",
        "return_kind": "actual_engine_arithmetic_net_equity",
    }
    assert allocation["equal_risk"] == {
        "constructor": "ERCPortfolio",
        "cov_window": None,
        "covariance": "ledoit_wolf_shrunk_covariance",
        "max_iter": 10000,
        "tol": 1e-10,
    }
    assert allocation["shrunk_hrp"]["corr_threshold"] == 0.6
    assert allocation["shrunk_hrp"]["correlation_shrinkage"] is True
    assert allocation["caps"] == {"full": 0.5, "loo": 0.7}
    assert allocation["rounding"]["ndigits"] == 10
    assert allocation["rounding"]["cash_residual_upper_exclusive"] == 1e-9
    assert allocation["final_weight_refit"]["enabled"] is True
    assert allocation["final_weight_refit"]["fit_inputs"] == ["train", "validation"]
    assert allocation["final_weight_refit"]["no_post_refit_validation_rescore"] is True
    assert allocation["gross_scaling"] == {
        "clip_max": 2.25,
        "clip_min": 0.25,
        "epsilon": 1e-12,
        "formula": "clip(0.25,2.25,0.27/max(validation_1x_mdd,1e-12))",
        "requires_positive_exposure_normalized_1x_sibling": True,
        "target_validation_mdd": 0.27,
    }

    nodes = config["current_trial_registry"]["nodes"]
    row_ids = {row["row_id"] for row in nodes}
    assert len(row_ids) == 21
    assert len([row_id for row_id in row_ids if row_id.startswith("component_")]) == 3
    assert len([row_id for row_id in row_ids if row_id.startswith("full_")]) == 5
    assert len([row_id for row_id in row_ids if row_id.startswith("loo_")]) == 9
    assert len([row_id for row_id in row_ids if row_id.startswith("incumbent_")]) == 3
    assert row_ids & {"diagnostic_track_b_codex_lagged_leaf_router_grid"}

    scaled = [row for row in nodes if row["gross"]["method"] == "validation_mdd_target"]
    assert [row["row_id"] for row in scaled] == [
        "full_equal_risk_scaled",
        "full_shrunk_hrp_scaled",
    ]
    assert all(row["gross"]["target_mdd"] == 0.27 for row in scaled)
    assert all(row["gross"]["clip_min"] == 0.25 for row in scaled)
    assert all(row["gross"]["clip_max"] == 2.25 for row in scaled)

    portfolio_rows = [
        row
        for row in nodes
        if row["row_id"].startswith("full_") or row["row_id"].startswith("loo_")
    ]
    assert len(portfolio_rows) == 14
    assert all(
        row["params"]
        == {
            "decision_cadence_seconds": 1,
            "final_weight_refit": True,
            "score_from_flat": True,
        }
        for row in portfolio_rows
    )


def test_runtime_contract_is_exhaustive_and_has_no_override_surface() -> None:
    config = _strict_load(CONFIG_PATH)
    runtime = config["runtime_contract"]
    _validate_declared_config_surface(config)

    assert runtime["class_name"] == "AlphaMaxBacktestConfig"
    assert runtime["construction"] == {
        "allow_private_rt_attribute": False,
        "allow_runtime_config_fallback": False,
        "allow_unknown_attribute_read": False,
        "final_after_construction": True,
        "runtime_field_missing_policy": "reject_unfrozen_runtime_field",
        "runtime_field_unknown_policy": "reject_unfrozen_runtime_field",
    }
    assert runtime["override_policy"] == EXPECTED_OVERRIDE_POLICY
    assert runtime["derived_attributes"] == {
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

    static = runtime["static_attributes"]
    required_exact = {
        "ALLOW_MARKET_ORDERS": True,
        "ANNUAL_PERIODS": 2190,
        "APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS": True,
        "BACKTEST_DECISION_SECONDS": 1,
        "BACKTEST_POLL_SECONDS": 1,
        "BACKTEST_WINDOW_SECONDS": 1,
        "COMMISSION_RATE": 0.0004,
        "COMPUTE_BACKEND": "cpu",
        "DECISION_CADENCE_SECONDS": 1,
        "DEFAULT_ORDER_TYPE": "MKT",
        "EFFECTIVE_POSITION_FRACTION": 1.0,
        "ENFORCE_REDUCE_ONLY": True,
        "FUNDING_INTERVAL_HOURS": 8,
        "FUNDING_ON_UTC_BOUNDARY": True,
        "FUNDING_RATE_PER_8H": 0.0,
        "INITIAL_CAPITAL": 10000.0,
        "LEVERAGE": 3,
        "LIMIT_PRICE_MODE": "one_tick_worse",
        "LIMIT_PRICE_OFFSET_TICKS": 1,
        "LIMIT_TIME_IN_FORCE": "GTC",
        "MAKER_FEE_RATE": 0.0002,
        "MARKET_WINDOW_PARITY_V2_ENABLED": True,
        "REQUIRE_FUNDING_COVERAGE": True,
        "SIM_LATENCY_MAX_BARS": 1,
        "SIM_LATENCY_MIN_BARS": 1,
        "SIM_MAX_BAR_VOLUME_RATIO": 0.1,
        "SKIP_AHEAD_ENABLED": False,
        "SLIPPAGE_IMPACT_COEFFICIENT": 0.1,
        "SLIPPAGE_IMPACT_MODEL": "sqrt_impact",
        "SPREAD_RATE": 0.0002,
        "STRATEGY_QUALITY_ENABLED": False,
        "TAKER_FEE_RATE": 0.0004,
        "TARGET_ALLOCATION": 0.1,
        "TARGET_ALLOCATION_MODE": "notional_fraction",
        "TIMEFRAME": "1s",
        "TIMEFRAMES": ["1s", "4h", "1d"],
        "WINDOW_SECONDS": 1,
    }
    assert {key: static[key] for key in required_exact} == required_exact
    assert all(value is not None for value in static.values())
    assert static["SYMBOL_LIMITS"] == {
        symbol: {
            "min_notional": 5.0,
            "min_qty": 0.001,
            "price_tick_size": 1e-8,
            "qty_step": 0.001,
        }
        for symbol in CANDIDATE_SYMBOLS
    }

    constructor = runtime["backtest_constructor"]
    assert constructor["strategy_timeframe"] == "1s"
    assert constructor["warmup_bars"] == 0
    assert constructor["record_history"] is False
    assert constructor["track_metrics"] is False
    assert constructor["record_trades"] is False
    assert constructor["strict_data_handler_construction"] is True
    assert constructor["data_handler_kwargs"] == {
        "backtest_poll_seconds": 1,
        "backtest_window_seconds": 1,
        "feature_db_path": None,
        "feature_exchange": "binance",
        "feature_lookup": "phase_owned_AlphaMaxOrderedFundingLookup_by_identity",
        "market_window_parity_v2_enabled": True,
    }
    assert constructor["portfolio_kwargs"] == {
        "fill_application_attribution_sink": "collector.record_application",
        "funding_boundary_resolver": ("phase_owned_AlphaMaxFundingBoundaryResolver_by_identity"),
    }
    assert constructor["execution_handler_kwargs"] == {
        "record_cost_attribution": True,
    }
    assert runtime["portfolio_strategy_constructor"] == {
        "class_name": "ArtifactPortfolioModeStrategy",
        "decision_cadence_seconds": 1,
        "portfolio_mode": "manifest:<immutable_absolute_row_path>",
        "strategy_params_exact_keys": ["decision_cadence_seconds", "portfolio_mode"],
    }

    unknown_top_level = copy.deepcopy(config)
    unknown_top_level["runtime_override"] = {}
    with pytest.raises(ValueError, match="undeclared top-level"):
        _validate_declared_config_surface(unknown_top_level)

    unknown_runtime = copy.deepcopy(config)
    unknown_runtime["runtime_contract"]["static_attributes"]["UNDECLARED"] = 1
    with pytest.raises(ValueError, match="undeclared or missing runtime attribute"):
        _validate_declared_config_surface(unknown_runtime)

    enabled_override = copy.deepcopy(config)
    enabled_override["runtime_contract"]["override_policy"]["runtime_override_loaded"] = True
    with pytest.raises(ValueError, match="override policy"):
        _validate_declared_config_surface(enabled_override)

    nonfinite = copy.deepcopy(config)
    nonfinite["runtime_contract"]["static_attributes"]["INITIAL_CAPITAL"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        _validate_declared_config_surface(nonfinite)


def test_return_first_mdd_and_historical_report_only_policy_is_frozen() -> None:
    config = _strict_load(CONFIG_PATH)
    selection = config["selection_and_reporting"]
    assert selection["objective"] == "return_first_after_eligibility"
    assert selection["eligibility_reference_cost_bps"] == 30
    assert selection["ranking"] == [
        {"direction": "descending", "field": "cumulative_return"},
        {"direction": "descending", "field": "cagr"},
        {"direction": "descending", "field": "calmar"},
        {"direction": "descending", "field": "net_sharpe"},
        {"direction": "ascending", "field": "mdd"},
        {"direction": "ascending", "field": "row_id"},
    ]
    assert selection["mdd_policy"] == {
        "hard_reject_above": 0.35,
        "normal_upper_inclusive": 0.3,
        "soft_comparator": "deterministic_return_first_best_matched_normal_row",
        "soft_lower_exclusive": 0.3,
        "soft_requires_nonempty_normal_set": True,
        "soft_requires_strict_calmar_superiority": True,
        "soft_requires_strict_cagr_superiority": True,
        "soft_upper_inclusive": 0.35,
    }

    historical = selection["historical_evaluation"]
    assert historical == {
        "confirmation_status": "not_run",
        "historical_evaluation_leader_is_selection": False,
        "historical_exposure_status": "committed_period_outcomes_observed",
        "may_mutate_prelock": False,
        "requires_fresh_confirmation": True,
    }
    assert selection["terminal_outcome_precedence"] == [
        "no_demonstrated_alpha",
        "historical_evaluation_incomplete",
        "prelock_champion_historical_robustness_failed",
        "prelock_champion_historical_robustness_passed",
    ]

    process = config["process_boundaries"]
    assert process["prelock_historical_evaluation_inputs_forbidden"] is True
    assert process["historical_evaluation_process_is_separate"] is True
    assert process["runtime_omx_reads_forbidden"] is True
    assert not any("historical" in argument for argument in process["prelock_cli_allowed_inputs"])

    safety = config["safety_and_claims"]
    assert safety == {
        "allow_live_allocation": False,
        "allow_paper_allocation": False,
        "allow_promotion": False,
        "allow_real_allocation": False,
        "external_data_collection_required_for_local_delivery": False,
        "historical_period_is_untouched": False,
        "local_alpha_performance_claim_allowed": False,
        "requires_fresh_future_or_genuinely_withheld_confirmation": True,
        "research_only": True,
    }


def test_three_incumbents_are_explicit_replay_unavailable() -> None:
    config = _strict_load(CONFIG_PATH)
    audit = config["incumbent_resolution"]
    assert _sha256(INCUMBENT_AUDIT_PATH.read_bytes()) == INCUMBENT_AUDIT_SHA256
    assert audit["baseline_commit"] == BASELINE_COMMIT
    assert audit["aggregate_resolution"] == {
        "all_named_incumbents_unavailable": True,
        "resolved_count": 0,
        "unavailable_count": 3,
    }
    assert [row["row_id"] for row in audit["rows"]] == [
        "incumbent_track_a_dynamic_conviction_switch",
        "incumbent_cross_asset_lead_lag_momentum",
        "incumbent_cross_candidate_hybrid_v3_5",
    ]
    assert all(row["resolution_status"] == "incumbent_replay_unavailable" for row in audit["rows"])
    assert all(row["selection_eligible"] is False for row in audit["rows"])
    assert all(row["mdd_comparator_eligible"] is False for row in audit["rows"])
    assert audit["runtime_contract"] == {
        "all_three_rows_are_attempted_as_explicit_unavailable_rows_on_every_cost_cell": True,
        "dynamic_incumbent_resolution_forbidden": True,
        "embed_json_value_identically": True,
        "incumbent_comparison_status": "unavailable",
        "later_research_may_create_new_trial_but_may_not_mutate_this_audit": True,
        "nearby_proxy_forbidden": True,
        "selection_and_mdd_comparator_membership": "excluded",
    }
