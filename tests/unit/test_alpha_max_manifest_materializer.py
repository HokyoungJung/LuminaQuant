from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import pytest

from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_CANDIDATE_SYMBOLS,
    ALPHA_MAX_MANIFEST_CHILD_KEYS,
    ALPHA_MAX_MANIFEST_TOP_LEVEL_KEYS,
    allocate_alpha_max_equal_weight,
    materialize_alpha_max_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / ".omx" / "plans" / "alpha-max-current-trial-nodes-v1.json"
ADMISSION_SHA = "d" * 64


@pytest.fixture(scope="module")
def registry_nodes():
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {node["row_id"]: node for node in payload["nodes"]}


def _owned_output_root(tmp_path: Path) -> Path:
    output_root = tmp_path / "alpha-max-run"
    (output_root / "manifests" / "validation_train_fit").mkdir(parents=True)
    (output_root / "manifests" / "prelock_final_refit").mkdir()
    return output_root.resolve()


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "alpha-max-config.json"
    path.write_bytes(b'{"schema":"alpha_max_config.v1"}\n')
    return path.resolve()


def _materialize(
    tmp_path,
    registry_nodes,
    row_id,
    weights,
    *,
    gross=1.0,
    phase="validation_train_fit",
    admitted=ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
    output_root=None,
    config_path=None,
):
    output_root = output_root or _owned_output_root(tmp_path)
    config_path = config_path or _config(tmp_path)
    result = materialize_alpha_max_manifest(
        registry_nodes[row_id],
        weights,
        gross,
        phase,
        config_path,
        output_root,
        ALPHA_MAX_CANDIDATE_SYMBOLS,
        admitted,
        ADMISSION_SHA,
    )
    return result, output_root, config_path


@pytest.mark.parametrize(
    ("row_id", "weights", "gross", "expected_count", "expected_method"),
    [
        ("component_carry_1x", {"component_carry_1x": 1.0}, 1.0, 1, "single_component"),
        (
            "full_equal_weight_1x",
            {
                "component_carry_1x": 0.3333333333,
                "component_near_high_1x": 0.3333333333,
                "component_trend_1x": 0.3333333333,
            },
            1.0,
            3,
            "equal_weight",
        ),
        (
            "loo_equal_risk_omit_carry_1x",
            {"component_near_high_1x": 0.55, "component_trend_1x": 0.45},
            1.0,
            2,
            "equal_risk",
        ),
        (
            "full_shrunk_hrp_scaled",
            {
                "component_carry_1x": 0.40,
                "component_near_high_1x": 0.30,
                "component_trend_1x": 0.30,
            },
            1.75,
            3,
            "shrunk_hrp",
        ),
    ],
)
def test_materializer_emits_exact_component_full_loo_and_scaled_schema(
    tmp_path,
    registry_nodes,
    row_id,
    weights,
    gross,
    expected_count,
    expected_method,
):
    result, _, config_path = _materialize(tmp_path, registry_nodes, row_id, weights, gross=gross)
    payload = result.payload

    assert set(payload) == ALPHA_MAX_MANIFEST_TOP_LEVEL_KEYS
    assert payload["artifact_kind"] == "alpha_max_engine_portfolio_manifest.v1"
    assert payload["candidate_symbols"] == list(ALPHA_MAX_CANDIDATE_SYMBOLS)
    assert payload["admitted_symbols"] == list(ALPHA_MAX_CANDIDATE_SYMBOLS[:5])
    assert payload["admission_manifest_sha256"] == ADMISSION_SHA
    assert payload["gross_cap"] == gross
    assert payload["cash_weight"] == max(0.0, 1.0 - gross * math.fsum(weights.values()))
    assert payload["allocation_method"] == expected_method
    assert len(payload["children"]) == expected_count
    assert [child["candidate_id"] for child in payload["children"]] == sorted(weights)
    assert all(set(child) == ALPHA_MAX_MANIFEST_CHILD_KEYS for child in payload["children"])
    assert all(
        child["candidate_symbols"] == list(ALPHA_MAX_CANDIDATE_SYMBOLS)
        for child in payload["children"]
    )
    assert all(
        child["symbols"] == list(ALPHA_MAX_CANDIDATE_SYMBOLS[:5]) for child in payload["children"]
    )
    assert all(child["weight"] == child["leaf_gross"] for child in payload["children"])
    assert all(child["source_artifact_id"] == "alpha_max_config" for child in payload["children"])
    assert payload["source_artifacts"] == [
        {
            "id": "alpha_max_config",
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "max_age_hours": 876000,
            "ready": True,
            "portfolio_ready": True,
        }
    ]
    assert all(
        payload[key] is False
        for key in (
            "real_money_execution",
            "allow_real_money",
            "ready_for_real",
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
    )


def test_materializer_bytes_path_hash_and_strategy_params_are_exact(tmp_path, registry_nodes):
    result, output_root, _ = _materialize(
        tmp_path,
        registry_nodes,
        "component_trend_1x",
        {"component_trend_1x": 1.0},
    )
    expected_path = output_root / "manifests" / "validation_train_fit" / "component_trend_1x.json"
    expected_bytes = (
        json.dumps(
            result.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )

    assert result.path == str(expected_path)
    assert result.manifest_path == result.path
    assert result.canonical_bytes == expected_bytes == expected_path.read_bytes()
    assert result.sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert result.manifest_sha256 == result.sha256
    assert dict(result.strategy_params) == {
        "portfolio_mode": f"manifest:{expected_path}",
        "decision_cadence_seconds": 1,
    }
    assert result["payload"] == result.payload
    assert result["manifest_bytes"] == result.canonical_bytes

    with pytest.raises(ValueError, match="target_exists"):
        materialize_alpha_max_manifest(
            registry_nodes["component_trend_1x"],
            {"component_trend_1x": 1.0},
            1.0,
            "validation_train_fit",
            _config(tmp_path),
            output_root,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )


def test_fixed_component_and_equal_weight_payloads_are_phase_identical(tmp_path, registry_nodes):
    output_root = _owned_output_root(tmp_path)
    config_path = _config(tmp_path)
    weights = allocate_alpha_max_equal_weight(
        (
            "component_carry_1x",
            "component_near_high_1x",
            "component_trend_1x",
        ),
        per_component_cap=0.50,
    )
    validation, _, _ = _materialize(
        tmp_path,
        registry_nodes,
        "full_equal_weight_1x",
        weights,
        output_root=output_root,
        config_path=config_path,
    )
    final, _, _ = _materialize(
        tmp_path,
        registry_nodes,
        "full_equal_weight_1x",
        weights,
        phase="prelock_final_refit",
        output_root=output_root,
        config_path=config_path,
    )

    assert validation.path != final.path
    assert validation.canonical_bytes == final.canonical_bytes
    assert validation.sha256 == final.sha256
    assert validation.payload["optimizer_provenance"] == {"selection_inputs": ["train"]}


def test_final_refit_provenance_changes_only_for_equal_risk_and_hrp(tmp_path, registry_nodes):
    result, _, _ = _materialize(
        tmp_path,
        registry_nodes,
        "full_equal_risk_1x",
        {
            "component_carry_1x": 0.40,
            "component_near_high_1x": 0.30,
            "component_trend_1x": 0.30,
        },
        phase="prelock_final_refit",
    )
    payload = result.payload

    assert payload["optimizer_provenance"] == {"selection_inputs": ["train", "validation"]}
    assert payload["correlation_input_provenance"] == {
        "selection_inputs": ["train", "validation"],
        "ready": True,
        "source": "alpha_max_train_validation_daily_net_returns",
    }
    assert all(
        child["optimizer_provenance"] == payload["optimizer_provenance"]
        and child["correlation_input_provenance"] == payload["correlation_input_provenance"]
        for child in payload["children"]
    )


def test_component_manifest_uses_exact_component_class_params_and_active_admission(
    tmp_path, registry_nodes
):
    admitted = ALPHA_MAX_CANDIDATE_SYMBOLS
    result, _, _ = _materialize(
        tmp_path,
        registry_nodes,
        "component_near_high_1x",
        {"component_near_high_1x": 1.0},
        admitted=admitted,
    )
    child = result.payload["children"][0]

    assert child["strategy_class"] == registry_nodes["component_near_high_1x"]["implementation"]
    assert child["params"] == registry_nodes["component_near_high_1x"]["params"]
    assert child["candidate_symbols"] == registry_nodes["component_near_high_1x"]["symbols"]
    assert child["symbols"] == list(admitted)
    assert "decision_cadence_seconds" not in child["params"]
    assert "final_weight_refit" not in child["params"]
    assert "score_from_flat" not in child["params"]


@pytest.mark.parametrize(
    "row_id",
    [
        "incumbent_track_a_dynamic_conviction_switch",
        "incumbent_cross_asset_lead_lag_momentum",
        "incumbent_cross_candidate_hybrid_v3_5",
        "diagnostic_track_b_codex_lagged_leaf_router_grid",
    ],
)
def test_materializer_never_materializes_incumbents_or_diagnostics(
    tmp_path, registry_nodes, row_id
):
    output_root = _owned_output_root(tmp_path)
    config_path = _config(tmp_path)
    with pytest.raises(ValueError, match="not_materializable"):
        materialize_alpha_max_manifest(
            registry_nodes[row_id],
            {},
            1.0,
            "validation_train_fit",
            config_path,
            output_root,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )
    assert not list((output_root / "manifests" / "validation_train_fit").iterdir())


def test_materializer_rejects_registry_mutation_before_writing(tmp_path, registry_nodes):
    output_root = _owned_output_root(tmp_path)
    config_path = _config(tmp_path)
    row = deepcopy(registry_nodes["full_equal_weight_1x"])
    row["params"]["decision_cadence_seconds"] = 60

    with pytest.raises(ValueError, match="registry_mismatch"):
        materialize_alpha_max_manifest(
            row,
            {
                "component_carry_1x": 0.3333333333,
                "component_near_high_1x": 0.3333333333,
                "component_trend_1x": 0.3333333333,
            },
            1.0,
            "validation_train_fit",
            config_path,
            output_root,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )
    assert not list((output_root / "manifests" / "validation_train_fit").iterdir())


@pytest.mark.parametrize(
    ("weights", "gross", "candidate_symbols", "admitted", "admission_sha", "reason"),
    [
        (
            {},
            1.0,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
            "coverage",
        ),
        (
            {"component_carry_1x": 0.8, "component_near_high_1x": 0.1, "component_trend_1x": 0.1},
            1.0,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
            "cap_breach",
        ),
        (
            {"component_carry_1x": 0.3, "component_near_high_1x": 0.3, "component_trend_1x": 0.3},
            1.0,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
            "rounding_invalid",
        ),
        (
            {"component_carry_1x": 0.4, "component_near_high_1x": 0.3, "component_trend_1x": 0.3},
            2.26,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
            "gross_invalid",
        ),
        (
            {"component_carry_1x": 0.4, "component_near_high_1x": 0.3, "component_trend_1x": 0.3},
            1.0,
            ALPHA_MAX_CANDIDATE_SYMBOLS[::-1],
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
            "candidate_symbols_mismatch",
        ),
        (
            {"component_carry_1x": 0.4, "component_near_high_1x": 0.3, "component_trend_1x": 0.3},
            1.0,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:4],
            ADMISSION_SHA,
            "count_invalid",
        ),
        (
            {"component_carry_1x": 0.4, "component_near_high_1x": 0.3, "component_trend_1x": 0.3},
            1.0,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            "NOT-A-HASH",
            "sha256_invalid",
        ),
    ],
)
def test_manifest_preconditions_fail_closed(
    tmp_path,
    registry_nodes,
    weights,
    gross,
    candidate_symbols,
    admitted,
    admission_sha,
    reason,
):
    output_root = _owned_output_root(tmp_path)
    config_path = _config(tmp_path)
    with pytest.raises((TypeError, ValueError), match=reason):
        materialize_alpha_max_manifest(
            registry_nodes["full_equal_risk_1x"],
            weights,
            gross,
            "validation_train_fit",
            config_path,
            output_root,
            candidate_symbols,
            admitted,
            admission_sha,
        )
    assert not list((output_root / "manifests" / "validation_train_fit").iterdir())


@pytest.mark.parametrize(
    ("row_id", "weights"),
    [
        ("component_carry_1x", {"component_carry_1x": 1.0}),
        (
            "full_equal_risk_1x",
            {
                "component_carry_1x": 0.4,
                "component_near_high_1x": 0.3,
                "component_trend_1x": 0.3,
            },
        ),
        (
            "loo_shrunk_hrp_omit_trend_1x",
            {"component_carry_1x": 0.5, "component_near_high_1x": 0.5},
        ),
    ],
)
def test_fixed_rows_reject_non_1x_gross_before_artifact_access(
    tmp_path, registry_nodes, row_id, weights
):
    missing_config = (tmp_path / "missing-config.json").resolve()
    missing_output = (tmp_path / "missing-output").resolve()

    with pytest.raises(ValueError, match="gross_invalid_fixed_mismatch"):
        materialize_alpha_max_manifest(
            registry_nodes[row_id],
            weights,
            2.0,
            "validation_train_fit",
            missing_config,
            missing_output,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )


@pytest.mark.parametrize(
    ("row_id", "weights"),
    [
        (
            "full_equal_weight_1x",
            {
                "component_carry_1x": 0.50,
                "component_near_high_1x": 0.25,
                "component_trend_1x": 0.25,
            },
        ),
        (
            "loo_equal_weight_omit_trend_1x",
            {"component_carry_1x": 0.70, "component_near_high_1x": 0.30},
        ),
    ],
)
def test_fixed_weight_rows_require_exact_registry_weights_before_artifact_access(
    tmp_path, registry_nodes, row_id, weights
):
    missing_config = (tmp_path / "missing-config.json").resolve()
    missing_output = (tmp_path / "missing-output").resolve()

    with pytest.raises(ValueError, match="resolved_weight_fixed_mismatch"):
        materialize_alpha_max_manifest(
            registry_nodes[row_id],
            weights,
            1.0,
            "validation_train_fit",
            missing_config,
            missing_output,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )


@pytest.mark.parametrize("gross", [0.1, 2.26])
def test_scaled_rows_enforce_frozen_validation_mdd_clip_before_writing(
    tmp_path, registry_nodes, gross
):
    output_root = _owned_output_root(tmp_path)
    config_path = _config(tmp_path)

    with pytest.raises(ValueError, match="gross_invalid_scaled_clip"):
        materialize_alpha_max_manifest(
            registry_nodes["full_equal_risk_scaled"],
            {
                "component_carry_1x": 0.4,
                "component_near_high_1x": 0.3,
                "component_trend_1x": 0.3,
            },
            gross,
            "validation_train_fit",
            config_path,
            output_root,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )
    assert not list((output_root / "manifests" / "validation_train_fit").iterdir())


def test_materializer_rejects_ambient_relative_and_nonowned_output_paths(tmp_path, registry_nodes):
    output_root = _owned_output_root(tmp_path)
    config_path = _config(tmp_path)
    weights = {"component_carry_1x": 1.0}

    with pytest.raises(ValueError, match="config_path_must_be_absolute"):
        materialize_alpha_max_manifest(
            registry_nodes["component_carry_1x"],
            weights,
            1.0,
            "validation_train_fit",
            config_path.name,
            output_root,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )

    extra = output_root / "manifests" / "ambient.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not_run_owned"):
        materialize_alpha_max_manifest(
            registry_nodes["component_carry_1x"],
            weights,
            1.0,
            "validation_train_fit",
            config_path,
            output_root,
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )


def test_materializer_rejects_symlink_phase_and_existing_target(tmp_path, registry_nodes):
    config_path = _config(tmp_path)
    output_root = tmp_path / "run"
    manifests = output_root / "manifests"
    real_validation = tmp_path / "real-validation"
    real_validation.mkdir(parents=True)
    manifests.mkdir(parents=True)
    os.symlink(real_validation, manifests / "validation_train_fit")
    (manifests / "prelock_final_refit").mkdir()

    with pytest.raises(ValueError, match="not_owned_directory"):
        materialize_alpha_max_manifest(
            registry_nodes["component_carry_1x"],
            {"component_carry_1x": 1.0},
            1.0,
            "validation_train_fit",
            config_path,
            output_root.resolve(),
            ALPHA_MAX_CANDIDATE_SYMBOLS,
            ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
            ADMISSION_SHA,
        )


def test_scaled_leaf_and_cap_identities_are_exact(tmp_path, registry_nodes):
    gross = 2.0
    weights = {
        "component_carry_1x": 0.50,
        "component_near_high_1x": 0.25,
        "component_trend_1x": 0.25,
    }
    result, _, _ = _materialize(
        tmp_path,
        registry_nodes,
        "full_equal_risk_scaled",
        weights,
        gross=gross,
    )
    payload = result.payload
    by_id = {child["candidate_id"]: child for child in payload["children"]}

    for component_id, weight in weights.items():
        child = by_id[component_id]
        assert child["weight"] == weight * gross
        assert child["leaf_gross"] == weight * gross
        assert child["leaf_gross_cap"] == 0.50 * gross
        assert child["netting_group_gross_cap"] == 0.50 * gross
    assert math.fsum(child["weight"] for child in payload["children"]) == gross
    assert payload["cash_weight"] == 0.0
