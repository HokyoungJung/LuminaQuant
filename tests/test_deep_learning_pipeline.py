from __future__ import annotations

import json
from datetime import UTC, datetime

from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.workflows.deep_learning_pipeline import (
    PIPELINE_MANIFEST_VERSION,
    build_deep_learning_pipeline_manifest,
    evaluate_deep_learning_chronological_sanity,
    recommend_deep_learning_gpu_parallelism,
    select_stable_deep_learning_hpo_trial,
    build_deep_learning_strategy_profiles,
    validate_deep_learning_forecast_artifacts,
)


def _runtime(overrides: dict | None = None):
    data = {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m"},
        "deep_learning": {
            "enabled": True,
            "repo_path": "/home/hoky/DeepLearning",
            "feature_matrix_path": "var/data/deeplearning/features",
            "prediction_path": "var/data/deeplearning/predictions",
            "models": ["FITS", "CycleNet", "CMamba", "PatchTST"],
            "seq_len": 120,
            "pred_len": 30,
            "total_pred": 120,
        },
    }
    if overrides:
        data["deep_learning"].update(overrides)
    return build_runtime_config(data, env={})


def test_deep_learning_pipeline_manifest_includes_training_and_strategy_format():
    runtime = _runtime(
        {
            "models": "FITS,CMamba,PatchTST,Unknown",
            "hpo_enabled": True,
            "hpo_trials": 7,
        }
    )

    manifest = build_deep_learning_pipeline_manifest(
        runtime,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert manifest["version"] == PIPELINE_MANIFEST_VERSION
    assert manifest["artifact_contract"]["models"] == ["FITS", "CMamba", "PatchTST"]
    assert manifest["stages"] == [
        "export_feature_matrices",
        "tune_models",
        "select_stable_hpo_params",
        "chronological_sanity_check",
        "train_predict_models",
        "validate_forecast_artifacts",
        "materialize_strategy_params",
        "backtest_strategy",
        "paper_or_shadow_live_gate",
    ]
    assert len(manifest["jobs"]) == 3
    first_job = manifest["jobs"][0]
    assert first_job["target"] == "BTC_close"
    assert first_job["stage"] == "tune_train_predict"
    assert first_job["cwd"] == "/home/hoky/DeepLearning"
    assert "--hpo" in first_job["argv"]
    assert first_job["argv"][first_job["argv"].index("--n-trials") + 1] == "7"
    assert first_job["feature_path"].endswith("BTC_close_1m.parquet")
    guardrails = manifest["operational_guardrails"]
    assert guardrails["split_contract"]["pre_test_train_validation"] == (
        "window_shuffle_split_for_candidate_hpo_search"
    )
    assert guardrails["hpo_selection"]["top_fraction"] == 0.10
    assert guardrails["drift_response"]["sequence"][2] == "retrain_latest_data_with_existing_params"

    ensemble_params = manifest["strategy_profiles"]["ensemble"]["params"]
    assert ensemble_params["models"] == "FITS,CMamba,PatchTST"
    assert ensemble_params["horizon_seconds"] == 30 * 60
    assert ensemble_params["forecast_path"] == "var/data/deeplearning/predictions"


def test_deep_learning_strategy_profiles_cover_single_models_and_ensemble():
    runtime = _runtime({"models": ["FITS", "PatchTST"], "pred_len": 4})

    profiles = build_deep_learning_strategy_profiles(runtime)

    assert sorted(profiles) == ["FITS", "PatchTST", "ensemble"]
    assert profiles["FITS"]["strategy"] == "DeepLearningForecastGateStrategy"
    assert profiles["FITS"]["params"]["models"] == "FITS"
    assert profiles["PatchTST"]["params"]["models"] == "PatchTST"
    assert profiles["ensemble"]["params"]["min_models"] == 2


def test_deep_learning_artifact_validation_reports_model_coverage(tmp_path):
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "forecasts.csv").write_text(
        "model_code,dbcode,pred_date,Date,pred_return,confidence\n"
        "FITS,BTC_close,2026-01-01T00:00:00Z,2026-01-01T00:30:00Z,0.01,0.9\n"
        "CMamba,BTC_close,2026-01-01T00:00:00Z,2026-01-01T00:30:00Z,0.02,0.8\n",
        encoding="utf-8",
    )
    runtime = _runtime(
        {
            "prediction_path": str(prediction_dir),
            "models": ["FITS", "CMamba", "PatchTST"],
            "min_model_coverage": 2,
        }
    )

    result = validate_deep_learning_forecast_artifacts(runtime)

    assert result["ok"] is True
    assert result["record_count"] == 2
    btc = result["symbols"]["BTC/USDT"]
    assert btc["present_models"] == ["FITS", "CMamba"]
    assert btc["missing_models"] == ["PatchTST"]
    assert btc["meets_min_model_coverage"] is True


def test_deep_learning_manifest_json_round_trip(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    runtime = _runtime({"manifest_path": str(manifest_path), "models": ["FITS"]})

    from lumina_quant.workflows.deep_learning_pipeline import write_deep_learning_pipeline_manifest

    written = write_deep_learning_pipeline_manifest(runtime)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written == manifest_path
    assert payload["training_state"]["job_count"] == 1
    assert payload["jobs"][0]["model"] == "FITS"


def test_deep_learning_gpu_parallelism_uses_utilization_not_model_size():
    saturated = recommend_deep_learning_gpu_parallelism(
        requested_jobs=3,
        free_vram_gb=6.0,
        gpu_utilization_pct=96.0,
    )
    assert saturated["decision"] == "parallel_not_recommended"
    assert saturated["parallel_recommended"] is False

    underused = recommend_deep_learning_gpu_parallelism(
        requested_jobs=3,
        free_vram_gb=6.0,
        gpu_utilization_pct=35.0,
    )
    assert underused["decision"] == "parallel_ok"
    assert underused["parallel_recommended"] is True

    low_vram = recommend_deep_learning_gpu_parallelism(
        requested_jobs=2,
        free_vram_gb=0.5,
        gpu_utilization_pct=20.0,
    )
    assert low_vram["decision"] == "reduce_batch_or_enable_amp"


def test_deep_learning_hpo_selection_prefers_stable_top_trial_over_spike():
    trials = [
        {
            "trial_id": "best-spike",
            "val_score": 0.940,
            "train_score": 0.995,
            "params": {
                "learning_rate": 0.01,
                "dropout": 0.0,
                "weight_decay": 0.0,
                "hidden_dim": 512,
            },
        },
        {
            "trial_id": "stable-top",
            "val_score": 0.932,
            "train_score": 0.941,
            "params": {
                "learning_rate": 0.003,
                "dropout": 0.20,
                "weight_decay": 0.001,
                "hidden_dim": 256,
            },
        },
        {
            "trial_id": "weak",
            "val_score": 0.800,
            "train_score": 0.801,
            "params": {
                "learning_rate": 0.003,
                "dropout": 0.20,
                "weight_decay": 0.001,
                "hidden_dim": 128,
            },
        },
    ]

    result = select_stable_deep_learning_hpo_trial(
        trials,
        search_space={
            "learning_rate": {"low": 0.0001, "high": 0.01},
            "hidden_dim": {"low": 64, "high": 512},
        },
        min_top_trials=2,
        max_train_val_gap=0.02,
    )

    assert result["top_candidate_count"] == 2
    assert result["selected"]["trial_id"] == "stable-top"
    assert "learning_rate_search_boundary:learning_rate" in result["ranked"][0]["flags"]


def test_deep_learning_chronological_sanity_bounds_relative_drop():
    ok = evaluate_deep_learning_chronological_sanity(
        reference_validation_score=1.0,
        chronological_validation_score=0.82,
        max_relative_drop=0.25,
    )
    assert ok["ok"] is True
    assert ok["relative_drop"] == 0.18000000000000005

    failed = evaluate_deep_learning_chronological_sanity(
        reference_validation_score=1.0,
        chronological_validation_score=0.70,
        max_relative_drop=0.25,
    )
    assert failed["ok"] is False
    assert failed["interpretation"] == "selected_params_need_alternative_or_partial_hpo"


def test_deep_learning_cli_hpo_select_reads_csv_trials(tmp_path, capsys):
    trials_path = tmp_path / "trials.csv"
    trials_path.write_text(
        "number,value,train_score,params_learning_rate,params_dropout,params_weight_decay\n"
        "0,0.940,0.995,0.01,0.0,0.0\n"
        "1,0.932,0.941,0.003,0.2,0.001\n",
        encoding="utf-8",
    )
    search_space_path = tmp_path / "search_space.json"
    search_space_path.write_text(
        json.dumps({"learning_rate": {"low": 0.0001, "high": 0.01}}),
        encoding="utf-8",
    )

    from lumina_quant.cli.deep_learning import main as deep_learning_main

    rc = deep_learning_main(
        [
            "--config",
            str(tmp_path / "missing.yaml"),
            "hpo-select",
            str(trials_path),
            "--search-space",
            str(search_space_path),
            "--min-top-trials",
            "2",
            "--max-train-val-gap",
            "0.02",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["selected"]["trial_id"] == 1
