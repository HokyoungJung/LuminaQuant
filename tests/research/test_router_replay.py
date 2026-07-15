"""Synthetic artifacts exercise the read-only router replay validator only."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from lumina_quant.data.symbol_lifecycle import (
    build_fold_membership_manifest,
    build_symbol_lifecycle_registry,
)
from lumina_quant.research import router_replay as replay


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class _RegistryStrategy:
    pass


def _source_payload(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "schema": replay.SOURCE_SCHEMA,
        "candidate_ids": list(replay.CANDIDATE_IDS),
        "candidate_ids_sha256": replay.CANDIDATE_IDS_SHA256,
        "controls": {
            "new_grid_search": False,
            "recompute_from_json": False,
            "post_oos_augment": False,
            "post_oos_research_variant": True,
        },
        "frozen_at_utc": "2026-07-15T00:00:00Z",
        "folds": [
            {
                key: deepcopy(fold[key])
                for key in (
                    "fold_id",
                    "locked_oos",
                    "input_cutoff_utc",
                    "decision_timestamp_utc",
                    "membership_fold_sha256",
                    "selection",
                )
            }
            for fold in manifest["folds"]
        ],
    }


def _sync_source(manifest: dict[str, object], paths: dict[str, Path]) -> None:
    _write(paths["source"], _source_payload(manifest))
    manifest["provenance"]["source_artifact_sha256"] = _file(paths["source"])


def _rehash_selection(fold: dict[str, object]) -> None:
    selection = fold["selection"]
    selection["leaf_list_sha256"] = _sha(selection["leaves"])
    selection["selection_inputs_sha256"] = _sha(
        {
            "branch": selection["branch"],
            "selected_label": selection["selected_label"],
            "history_receipts": selection["history_receipts"],
            "leaves": selection["leaves"],
        }
    )
    for variant in fold["variants"]:
        variant["selected_label"] = selection["selected_label"]
        variant["base_leaf_list_sha256"] = selection["leaf_list_sha256"]


def _set_cash(fold: dict[str, object]) -> None:
    selection = fold["selection"]
    selection.update(branch="strict_core_cash", selected_label=None, leaves=[])
    _rehash_selection(fold)
    for variant in fold["variants"]:
        variant.update(applied_scale=0.0, leaves=[], execution_receipts=[])


def _set_fallback(fold: dict[str, object], scales: tuple[float, float]) -> None:
    fold["selection"]["branch"] = "strict_core_scaled"
    _rehash_selection(fold)
    for variant, scale in zip(fold["variants"], scales, strict=True):
        variant["applied_scale"] = scale
        for base, effective in zip(fold["selection"]["leaves"], variant["leaves"], strict=True):
            effective["effective_weight"] = base["source_weight"] * scale
            effective["effective_gross"] = base["native_gross"] * scale
        variant["execution_receipts"][0]["position_receipt_sha256"] = f"{int(scale * 100):064x}"
        variant["execution_receipts"][0]["engine_receipt_sha256"] = f"{int(scale * 1000):064x}"


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    engine = replay._handler(replay.PROFILE_HANDLER)
    producer = tmp_path / "producer.py"
    producer.write_text("# frozen\n", encoding="utf-8")
    commit = tmp_path / "commit-receipt.json"
    _write(commit, {"commit": "e74658e16421d6e5380780aa1d6b37c5fd11ce55"})
    source = tmp_path / "source.json"
    profile = tmp_path / "profile.json"
    _write(
        profile,
        {
            "profile": "backtest_cost_realistic",
            "research": {
                "strict_selection_gate": True,
                "use_lockbox_split": True,
                "purge_embargo_bars": 1,
                "single_correlation_discount": True,
                "hac_inference": True,
                "cscv_pbo": True,
                "exposure_normalized_promotion": True,
                "route_unmapped_registered_strategies": True,
                "require_actual_engine_routing": True,
            },
            "execution": {
                "slippage_impact_model": "sqrt_impact",
                "slippage_impact_coefficient": 0.1,
                "funding_interval_hours": 8,
                "require_funding_coverage": True,
                "funding_on_utc_boundary": True,
            },
            "risk": {
                "attach_default_protective_stop": True,
                "enforce_order_risk_gate_in_backtest": True,
            },
            "live": {
                "mode": "paper",
                "testnet": True,
                "require_real_enable_flag": True,
                "allow_market_orders": False,
                "shadow_live_enabled": False,
            },
            "data": {"kinds": ["funding"]},
        },
    )
    registry = build_symbol_lifecycle_registry(
        {"symbols": [{"symbol": "BTCUSDT", "onboardDate": 0, "deliveryDate": None}]},
        ["BTCUSDT"],
        {"uri": "frozen://registry", "retrieved_at_ms": 0, "payload_sha256": "a" * 64},
    )
    registry_path = tmp_path / "registry.json"
    _write(registry_path, registry)
    membership = build_fold_membership_manifest(
        registry,
        [
            {"fold_id": "f0", "start_ms": 1735689600000, "end_ms": 1735776000000},
            {"fold_id": "f1", "start_ms": 1735776000000, "end_ms": 1735862400000},
        ],
    )
    membership_path = tmp_path / "membership.json"
    _write(membership_path, membership)
    params = {
        "family": "trend_pullback_reclaim",
        "timeframe": "1h",
        "side": "long_short",
        "integer_leverage": 1,
        "min_hold_bars": 6,
        "cooldown_bars": 0,
        "lookback_bars": 24,
        "fast_divisor": 3,
        "threshold": -1.0,
        "exit_threshold": 0.0,
        "trend_slope_min": 0.0,
        "market_guard": 0.0,
    }
    leaf = {
        "leaf_id": "leaf",
        "profile_id": "balanced_mdd12_gross5_69_asset_profile_optuna",
        "strategy_class": "trend_pullback_reclaim",
        "engine_handler": replay.PROFILE_HANDLER,
        "params": params,
        "traded_symbols": ["BTCUSDT"],
        "dependency_symbols": ["BTCUSDT"],
        "native_timeframe": "1h",
        "allocation_fraction": 1.0,
        "source_weight": 1.0,
        "native_gross": 1.0,
        "evaluation_mode": "handler",
        "engine_source_sha256": _file(engine),
    }
    leaf["source_row_sha256"] = _sha(leaf)

    def selection(
        branch: str, label: object, leaves: list[object], history: list[object]
    ) -> dict[str, object]:
        result = {
            "branch": branch,
            "selected_label": label,
            "current_fold_oos_input_count": 0,
            "history_receipts": history,
            "leaves": leaves,
            "leaf_list_sha256": _sha(leaves),
        }
        result["selection_inputs_sha256"] = _sha(
            {
                "branch": branch,
                "selected_label": label,
                "history_receipts": history,
                "leaves": leaves,
            }
        )
        return result

    def fold(
        index: int,
        branch: str = "pre_registered_lagged_plus_validation_leaf",
        leaves: list[object] | None = None,
    ) -> dict[str, object]:
        member = membership["folds"][index]
        leaves = [deepcopy(leaf)] if leaves is None else leaves
        if branch == "strict_core_cash":
            leaves = []
        selected = None if branch == "strict_core_cash" else "router-label"
        history = (
            []
            if index == 0
            else [
                {
                    "fold_id": "f0",
                    "candidate_label": "router-label",
                    "completed_at_utc": "2024-12-30T00:00:00Z",
                    "input_cutoff_utc": "2024-12-30T00:00:00Z",
                    "return": 0.1,
                    "source_sha256": "b" * 64,
                }
            ]
        )
        s = selection(branch, selected, leaves, history)
        variants = []
        for candidate, mdd, cap in zip(replay.CANDIDATE_IDS, (0.30, 0.20), (3.0, 2.0), strict=True):
            effective = [
                {
                    "leaf_id": x["leaf_id"],
                    "effective_weight": x["source_weight"],
                    "effective_gross": x["native_gross"],
                }
                for x in leaves
            ]
            receipts = [
                {
                    "leaf_id": x["leaf_id"],
                    "evaluation_mode": x["evaluation_mode"],
                    "engine_source_sha256": x["engine_source_sha256"],
                    "signal_receipt_sha256": "c" * 64,
                    "position_receipt_sha256": "d" * 64,
                    "engine_receipt_sha256": "e" * 64,
                    "generic_fallback_proxy_count": 0,
                    "current_fold_oos_input_count": 0,
                }
                for x in leaves
            ]
            variants.append(
                {
                    "variant_id": candidate,
                    "selected_label": selected,
                    "base_leaf_list_sha256": s["leaf_list_sha256"],
                    "policy": {"fallback_mdd": mdd, "fallback_cap": cap},
                    "applied_scale": 0.0 if branch == "strict_core_cash" else 1.0,
                    "leaves": effective,
                    "execution_receipts": receipts,
                }
            )
        return {
            "fold_id": member["fold_id"],
            "locked_oos": {
                "start_utc": "2025-01-%02dT00:00:00Z" % (1 + index),
                "end_utc": "2025-01-%02dT00:00:00Z" % (2 + index),
            },
            "input_cutoff_utc": ("2024-12-31T00:00:00Z", "2025-01-01T00:00:00Z")[index],
            "decision_timestamp_utc": "2025-01-%02dT00:00:00Z" % (1 + index),
            "membership_fold_sha256": _sha(member),
            "selection": s,
            "variants": variants,
        }

    manifest = {
        "schema": replay.SCHEMA,
        "candidate_ids": list(replay.CANDIDATE_IDS),
        "candidate_ids_sha256": replay.CANDIDATE_IDS_SHA256,
        "controls": {
            "new_grid_search": False,
            "recompute_from_json": False,
            "post_oos_augment": False,
            "real_money_enabled": False,
            "orders_submitted": 0,
            "capital_allocated": 0,
        },
        "provenance": {
            "producer_sha256": _file(producer),
            "verifier_version_sha256": _file(Path(replay.__file__)),
            "commit_receipt_sha256": _file(commit),
            "source_artifact_sha256": "0" * 64,
            "lifecycle_registry_sha256": _file(registry_path),
            "membership_manifest_sha256": _file(membership_path),
            "combined_profile_sha256": _file(profile),
        },
        "folds": [fold(0), fold(1)],
    }
    paths = {
        "manifest": tmp_path / "manifest.json",
        "source": source,
        "registry": registry_path,
        "membership": membership_path,
        "profile": profile,
        "producer": producer,
        "commit": commit,
    }
    _sync_source(manifest, paths)
    return manifest, paths


def _report(manifest: dict[str, object], paths: dict[str, Path], *, sync_source: bool = True):
    if sync_source:
        _sync_source(manifest, paths)
    _write(paths["manifest"], manifest)
    return replay.evaluate_router_replay(
        paths["manifest"],
        source_artifact_path=paths["source"],
        lifecycle_registry_path=paths["registry"],
        membership_manifest_path=paths["membership"],
        combined_profile_path=paths["profile"],
        producer_source_path=paths["producer"],
        commit_receipt_path=paths["commit"],
    )


def _stop(manifest: dict[str, object], paths: dict[str, Path]) -> None:
    assert _report(manifest, paths).status == "STOP"


def test_handler_manifest_passes(artifacts):
    manifest, paths = artifacts
    report = _report(manifest, paths)
    assert report.status == "PASS" and report.fold_count == 2 and "NaN" not in report.to_json()


def test_registry_simulator_manifest_passes(artifacts, monkeypatch):
    manifest, paths = artifacts
    monkeypatch.setattr(replay, "resolve_strategy_class", lambda *args, **kwargs: _RegistryStrategy)
    registry_source_sha = _file(Path(__file__))
    for fold in manifest["folds"]:
        for leaf in fold["selection"]["leaves"]:
            leaf["evaluation_mode"] = "registry_simulator"
            leaf["engine_handler"] = "registry_simulator"
            leaf["engine_source_sha256"] = registry_source_sha
            leaf["source_row_sha256"] = _sha(
                {k: v for k, v in leaf.items() if k != "source_row_sha256"}
            )
            for variant in fold["variants"]:
                variant["execution_receipts"][0]["evaluation_mode"] = "registry_simulator"
                variant["execution_receipts"][0]["engine_source_sha256"] = registry_source_sha
        selection = fold["selection"]
        selection["leaf_list_sha256"] = _sha(selection["leaves"])
        selection["selection_inputs_sha256"] = _sha(
            {
                "branch": selection["branch"],
                "selected_label": selection["selected_label"],
                "history_receipts": selection["history_receipts"],
                "leaves": selection["leaves"],
            }
        )
        for variant in fold["variants"]:
            variant["base_leaf_list_sha256"] = selection["leaf_list_sha256"]
    assert _report(manifest, paths).status == "PASS"


def test_cash_and_strict_fallback_manifests_pass(artifacts):
    original, paths = artifacts
    cash = deepcopy(original)
    _set_cash(cash["folds"][0])
    assert _report(cash, paths).status == "PASS"

    fallback = deepcopy(original)
    _set_fallback(fallback["folds"][0], (2.25, 1.75))
    assert _report(fallback, paths).status == "PASS"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m, p: m.update({"extra": 1}),
        lambda m, p: m["controls"].update({"orders_submitted": 1}),
        lambda m, p: m["controls"].update({"orders_submitted": False}),
        lambda m, p: m["folds"][0]["selection"].update({"current_fold_oos_input_count": 1}),
        lambda m, p: m["folds"][0]["selection"].update({"current_fold_oos_input_count": False}),
        lambda m, p: m["folds"][0]["selection"]["leaves"][0].update({"dependency_symbols": ["X"]}),
        lambda m, p: m["folds"][0]["selection"]["leaves"][0].update(
            {"engine_handler": "missing.handler"}
        ),
        lambda m, p: m["folds"][0]["selection"]["leaves"][0]["params"].update(
            {"family": "unknown"}
        ),
        lambda m, p: m["folds"][0]["selection"]["leaves"][0].update(
            {"source_row_sha256": "0" * 64}
        ),
        lambda m, p: m["folds"][0]["selection"].update({"leaf_list_sha256": "0" * 64}),
        lambda m, p: m["folds"][0]["variants"][0]["leaves"][0].update({"effective_gross": 9.0}),
        lambda m, p: m["folds"][0]["variants"][0].update({"applied_scale": 1.1}),
        lambda m, p: m["folds"][0]["variants"][0]["execution_receipts"][0].update(
            {"generic_fallback_proxy_count": 1}
        ),
        lambda m, p: m["folds"][0]["variants"][0]["execution_receipts"][0].update(
            {"generic_fallback_proxy_count": False}
        ),
    ],
)
def test_adversarial_manifest_rows_stop(artifacts, mutate):
    manifest, paths = artifacts
    mutate(manifest, paths)
    _stop(manifest, paths)


def test_external_hash_and_duplicate_nonfinite_json_stop(artifacts):
    manifest, paths = artifacts
    paths["source"].write_text("{}", encoding="utf-8")
    assert _report(manifest, paths, sync_source=False).status == "STOP"

    manifest, paths = artifacts
    _write(paths["source"], _source_payload(manifest))
    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    source["controls"]["post_oos_research_variant"] = False
    _write(paths["source"], source)
    manifest["provenance"]["source_artifact_sha256"] = _file(paths["source"])
    assert _report(manifest, paths, sync_source=False).status == "STOP"

    manifest, paths = artifacts
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    profile["live"]["mode"] = "real"
    _write(paths["profile"], profile)
    manifest["provenance"]["combined_profile_sha256"] = _file(paths["profile"])
    assert _report(manifest, paths).status == "STOP"

    manifest, paths = artifacts
    paths["commit"].write_text("drift", encoding="utf-8")
    assert _report(manifest, paths).status == "STOP"

    manifest, paths = artifacts
    paths["manifest"].write_text(
        '{"schema":"router_replay_v1","schema":"router_replay_v1"}', encoding="utf-8"
    )
    assert (
        replay.evaluate_router_replay(
            paths["manifest"],
            source_artifact_path=paths["source"],
            lifecycle_registry_path=paths["registry"],
            membership_manifest_path=paths["membership"],
            combined_profile_path=paths["profile"],
            producer_source_path=paths["producer"],
            commit_receipt_path=paths["commit"],
        ).status
        == "STOP"
    )
    paths["manifest"].write_text('{"schema":NaN}', encoding="utf-8")
    assert (
        replay.evaluate_router_replay(
            paths["manifest"],
            source_artifact_path=paths["source"],
            lifecycle_registry_path=paths["registry"],
            membership_manifest_path=paths["membership"],
            combined_profile_path=paths["profile"],
            producer_source_path=paths["producer"],
            commit_receipt_path=paths["commit"],
        ).status
        == "STOP"
    )


def test_future_history_interval_cash_and_receipt_parity_stop(artifacts):
    manifest, paths = artifacts
    manifest["folds"][1]["selection"]["history_receipts"][0]["fold_id"] = "f1"
    _stop(manifest, paths)
    manifest, paths = artifacts
    manifest["folds"][0]["locked_oos"]["end_utc"] = "2025-01-03T00:00:00Z"
    _stop(manifest, paths)
    manifest, paths = artifacts
    manifest["folds"][0]["selection"]["branch"] = "strict_core_cash"
    _stop(manifest, paths)
    manifest, paths = artifacts
    manifest["folds"][0]["variants"][1]["execution_receipts"][0]["position_receipt_sha256"] = (
        "9" * 64
    )
    _stop(manifest, paths)
