from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.research import export_alpha_max_observability as exporter


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(exporter._canonical_bytes(payload))


def _seal_prelock(
    root: Path,
    *,
    prelock_champion: str | None = None,
) -> dict[str, object]:
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name == "SEALED.json":
            continue
        raw = path.read_bytes()
        inventory.append(
            {
                "byte_count": len(raw),
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": exporter._sha256(raw),
            }
        )
    seal = {
        "artifact_count": len(inventory),
        "artifact_kind": "alpha_max_immutable_prelock_seal.v1",
        "artifacts": inventory,
        "historical_evaluation_inputs_included": False,
        "immutable": True,
        "inventory_sha256": exporter._sha256(exporter._canonical_bytes(inventory)),
        "prelock_champion": prelock_champion,
        "selected_candidate_id": prelock_champion,
    }
    _write_json(root / "SEALED.json", seal)
    return seal


def _manifest() -> dict[str, object]:
    return {
        "admission_manifest_sha256": "a" * 64,
        "admitted_symbols": ["BTCUSDT"],
        "allocation_method": "fixed",
        "candidate_symbols": ["BTCUSDT"],
        "cash_weight": 0.0,
        "children": [
            {
                "candidate_id": "component_trend_1x",
                "candidate_symbols": ["BTCUSDT"],
                "leaf_gross": 1.0,
                "leaf_gross_cap": 1.0,
                "netting_group": "component_trend_1x",
                "netting_group_gross_cap": 1.0,
                "params": {"lookback": 20},
                "strategy_class": "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
                "symbols": ["BTCUSDT"],
                "weight": 1.0,
            }
        ],
        "correlation_input_provenance": {"ready": True},
        "gross_cap": 1.0,
        "optimizer_provenance": {"selection_inputs": ["train"]},
    }


def _actual_run() -> dict[str, object]:
    observations = [
        {
            "bar_volume": 100.0,
            "equity_before": 10_000.0,
            "raw_price": 50.0,
            "requested_qty": 2.0,
        }
    ]
    reconciliation = {
        "application_count": 0,
        "application_trace_hashes": [],
        "applications": [],
        "applied_commission_total": 0.0,
        "artifact_kind": "alpha_max_cost_reconciliation.v1",
        "complete": True,
        "fee_reconciled": True,
        "funding_ledger": [],
        "funding_payment_total": 0.0,
        "funding_reconciled": True,
        "liquidation_cost_total": 0.0,
        "liquidation_reconciled": True,
        "model_commission_total": 0.0,
        "no_fill_attempt_count": 0,
        "no_fill_attempts": [],
        "no_fill_excluded_from_bijection": True,
        "portfolio_fee_total": 0.0,
        "portfolio_funding_total": 0.0,
        "portfolio_liquidation_total": 0.0,
        "pricing_application_bijection": True,
        "pricing_trace_count": 0,
        "pricing_trace_hashes": [],
        "pricing_traces": [],
        "zero_applied_application_count": 0,
    }
    diagnostics = {
        "capacity": {
            "capacity_proxy_equity_usdt": {"median_type7": 500_000.0},
            "observation_count": 1,
            "report_only": True,
            "undefined_reason": None,
        },
        "capacity_observation_set_sha256": exporter._sha256(
            exporter._canonical_bytes(observations)
        ),
        "capacity_observations": observations,
        "contribution_total_usdt": 5.0,
        "ending_market_value_usdt": {"BTCUSDT": 105.0},
        "ending_realized_gross_exposure": 0.01,
        "ending_realized_gross_undefined_reason": None,
        "fold_pnl_usdt": 5.0,
        "liquidity_clip_count": 0,
        "no_fill_attempt_count": 0,
        "reconciliation_residual_usdt": 0.0,
        "reduce_only_clip_count": 0,
        "report_only": True,
        "selection_influence": False,
        "symbol_contribution_usdt": {"BTCUSDT": 5.0},
        "target_gross_exposure": 1.0,
        "turnover_rpt": {"rpt_bps": 5.0, "turnover_multiple": 1.0},
    }
    manifest_raw = exporter._canonical_bytes(_manifest())
    return {
        "admitted_symbols": ["BTCUSDT"],
        "application_count": 0,
        "application_set_sha256": "1" * 64,
        "capsule_receipt": {"sha256": "2" * 64},
        "config_sha256": "3" * 64,
        "domain": "validation",
        "effective_config": {"slippage": 0.0025},
        "effective_config_sha256": "4" * 64,
        "ending_cash": 9900.0,
        "ending_equity": 10005.0,
        "equity_observation_count": 1,
        "feature_root_receipts": [{"content_sha256": "5" * 64}],
        "feature_root_set_sha256": "6" * 64,
        "fill_event_count": 0,
        "full_event_equity": {
            "ending_equity": 10005.0,
            "event_count": 1,
            "full_event_mdd": 0.0,
            "ruin_detected": False,
        },
        "funding_ledger_count": 0,
        "funding_ledger_set_sha256": "7" * 64,
        "liquidation_event_count": 0,
        "liquidation_event_set_sha256": "8" * 64,
        "manifest_receipt": {
            "byte_count": len(manifest_raw),
            "phase": "validation_train_fit",
            "relative_path": "manifests/validation_train_fit/component_trend_1x.json",
            "row_id": "component_trend_1x",
            "sha256": exporter._sha256(manifest_raw),
        },
        "market_event_count": 1,
        "native_finalization": {"boundary_utc": "2025-06-15T00:00:00Z"},
        "no_fill_attempt_count": 0,
        "no_fill_attempt_set_sha256": "b" * 64,
        "nominal_cost_bps": 30,
        "order_event_count": 0,
        "pricing_trace_count": 0,
        "pricing_trace_set_sha256": "c" * 64,
        "raw_root_receipts": [{"content_sha256": "d" * 64}],
        "raw_root_set_sha256": "e" * 64,
        "reconciliation": reconciliation,
        "report_only_diagnostics": diagnostics,
        "row_id": "component_trend_1x",
        "ruin_detected": False,
        "runtime_contract_sha256": "f" * 64,
        "runtime_read_audit": ["slippage"],
        "runtime_read_audit_sha256": "0" * 64,
        "seed": 42,
        "signal_event_count": 0,
        "split_or_fold_id": "validation_w01",
        "starting_cash": 10_000.0,
        "starting_equity": 10_000.0,
        "starting_open_order_count": 0,
        "starting_open_position_count": 0,
        "starting_used_margin": 0.0,
        "trade_count": 0,
        "universe_sha256": "a" * 64,
    }


def test_export_preserves_params_costs_capacity_and_reconciliation_without_raw_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    for phase in exporter._MANIFEST_PHASES:
        _write_json(root / "manifests" / phase / "component_trend_1x.json", _manifest())
    run = _actual_run()
    pre_gate = {
        "fold_run_set_sha256": "a" * 64,
        "fold_runs": [{"actual_engine_run": run}],
    }
    cell = {
        "domain": "validation",
        "evidence_tier": "actual_engine",
        "nominal_cost_bps": 30,
        "pre_gate_evidence": pre_gate,
        "row_id": "component_trend_1x",
        "selection_valid": True,
        "status": "complete",
        "terminal_gate_evidence": None,
    }
    _write_json(
        root / "evidence/validation/cells/component_trend_1x/30.json",
        cell,
    )
    monkeypatch.setattr(exporter, "_EXPECTED_MANIFESTS_PER_PHASE", 1)
    monkeypatch.setattr(exporter, "_EXPECTED_ACTUAL_CELLS", 1)
    monkeypatch.setitem(exporter._EXPECTED_FOLD_RUNS, "validation", 1)
    _seal_prelock(root)

    payload = exporter.build_export(root, root, "validation")

    assert payload["physical_fold_run_count"] == 1
    assert payload["manifests"][0]["children"][0]["params"] == {"lookback": 20}
    observed = payload["actual_engine_cells"][0]["fold_runs"][0]
    assert observed["nominal_cost_bps"] == 30
    assert observed["reconciliation"]["complete"] is True
    diagnostics = observed["report_only_diagnostics"]
    assert diagnostics["capacity_observation_count"] == 1
    assert "capacity_observations" not in diagnostics


def test_canonical_reader_rejects_intermediate_symlink(tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    outside = (tmp_path / "outside").resolve()
    root.mkdir()
    outside.mkdir()
    _write_json(outside / "payload.json", {"safe": False})
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe JSON parent"):
        exporter._read_canonical_json(root, "linked/payload.json")


def test_canonical_reader_rejects_atomic_swap_after_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    target = root / "payload.json"
    replacement = root / "replacement.json"
    _write_json(target, {"version": "original"})
    _write_json(replacement, {"version": "replacement"})
    original_open = os.open
    swapped = False

    def hostile_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "payload.json" and dir_fd is not None and not swapped:
            swapped = True
            replacement.replace(target)
        return descriptor

    monkeypatch.setattr(exporter.os, "open", hostile_open)

    with pytest.raises(ValueError, match="JSON artifact changed while opening"):
        exporter._read_canonical_json(root, "payload.json")

    assert swapped is True


def test_canonical_reader_rejects_nonfinite_json_constant(tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    (root / "payload.json").write_bytes(b'{"value":NaN}\n')

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        exporter._read_canonical_json(root, "payload.json")


def test_export_rejects_manifest_root_unrelated_to_fold_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = (tmp_path / "bundle").resolve()
    manifests = (tmp_path / "manifests").resolve()
    for phase in exporter._MANIFEST_PHASES:
        payload = _manifest()
        payload["gross_cap"] = 1.5
        _write_json(manifests / "manifests" / phase / "component_trend_1x.json", payload)
    run = _actual_run()
    cell = {
        "domain": "validation",
        "evidence_tier": "actual_engine",
        "nominal_cost_bps": 30,
        "pre_gate_evidence": {
            "fold_run_set_sha256": "a" * 64,
            "fold_runs": [{"actual_engine_run": run}],
        },
        "row_id": "component_trend_1x",
        "selection_valid": True,
        "status": "complete",
        "terminal_gate_evidence": None,
    }
    _write_json(
        bundle / "evidence/validation/cells/component_trend_1x/30.json",
        cell,
    )
    monkeypatch.setattr(exporter, "_EXPECTED_MANIFESTS_PER_PHASE", 1)
    monkeypatch.setattr(exporter, "_EXPECTED_ACTUAL_CELLS", 1)
    monkeypatch.setitem(exporter._EXPECTED_FOLD_RUNS, "validation", 1)
    _seal_prelock(bundle)
    _seal_prelock(manifests)

    with pytest.raises(ValueError, match="validation bundle/manifest seal mismatch"):
        exporter.build_export(bundle, manifests, "validation")


def test_seal_inventory_rejects_unsealed_artifact(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_json(root / "payload.json", {"value": "sealed"})
    _seal_prelock(root)
    _write_json(root / "unsealed.json", {"value": "not-in-inventory"})
    seal_payload, seal_bytes = exporter._read_canonical_json(root, "SEALED.json")

    with pytest.raises(ValueError, match="sealed artifact inventory path mismatch"):
        exporter._verify_seal_inventory(root, seal_payload, seal_bytes)


def test_manifest_summary_rechecks_bytes_after_seal_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    for phase in exporter._MANIFEST_PHASES:
        _write_json(root / "manifests" / phase / "component_trend_1x.json", _manifest())
    _seal_prelock(root)
    seal_payload, seal_bytes = exporter._read_canonical_json(root, "SEALED.json")
    _seal_sha256, inventory = exporter._verify_seal_inventory(
        root,
        seal_payload,
        seal_bytes,
    )
    hostile = _manifest()
    hostile["gross_cap"] = 1.5
    _write_json(
        root / "manifests/validation_train_fit/component_trend_1x.json",
        hostile,
    )
    monkeypatch.setattr(exporter, "_EXPECTED_MANIFESTS_PER_PHASE", 1)

    with pytest.raises(ValueError, match="changed after inventory verification"):
        exporter._manifest_summary(root, inventory)


def test_historical_bundle_must_bind_the_selected_prelock_seal() -> None:
    with pytest.raises(ValueError, match="historical bundle/prelock seal mismatch"):
        exporter._verify_bundle_pair(
            domain="historical_exposed_evaluation",
            bundle_seal_payload={
                "artifact_kind": "alpha_max_append_only_historical_package.v1",
                "prelock_seal_sha256": "a" * 64,
            },
            bundle_seal_sha256="b" * 64,
            manifest_seal_payload={
                "artifact_kind": "alpha_max_immutable_prelock_seal.v1",
            },
            manifest_seal_sha256="c" * 64,
        )


def test_historical_binding_file_must_equal_selected_prelock_seal(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    relative = "binding/prelock_seal.json"
    _write_json(root / relative, {"seal": "unrelated"})
    raw = (root / relative).read_bytes()
    inventory = {relative: (len(raw), exporter._sha256(raw))}

    with pytest.raises(ValueError, match="historical prelock seal binding mismatch"):
        exporter._verify_historical_prelock_binding(
            root,
            inventory,
            exporter._canonical_bytes({"seal": "selected"}),
        )
