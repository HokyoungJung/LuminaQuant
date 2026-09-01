"""DeepLearning sidecar pipeline planning for forecast-driven strategies.

This module does not train models. It builds a stable manifest that connects
LuminaQuant feature artifacts, external DeepLearning train/predict commands,
forecast artifact validation, and LuminaQuant strategy parameter profiles.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from lumina_quant.configuration.schema import DeepLearningRuntimeConfig, RuntimeConfig
from lumina_quant.data.deep_learning_forecasts import (
    DeepLearningForecastStore,
    normalize_deep_learning_models,
    normalize_forecast_symbol,
)

PIPELINE_MANIFEST_VERSION = "deep_learning_forecast_pipeline_v1"
_VALIDATION_SCORE_KEYS = (
    "validation_score",
    "val_score",
    "validation_return",
    "val_return",
    "score",
    "value",
)
_TRAIN_SCORE_KEYS = ("train_score", "training_score", "train_return", "training_return")
_LOWER_IS_BETTER_HINTS = ("loss", "mse", "mae", "rmse", "mape", "error")
_REGULARIZATION_PARAM_NAMES = ("dropout", "weight_decay")
_MODEL_SIZE_PARAM_NAMES = ("hidden_dim", "d_model", "layers", "num_layers", "n_layers")
_LR_PARAM_NAMES = ("learning_rate", "lr")


@dataclass(frozen=True, slots=True)
class DeepLearningPipelineJob:
    """Serializable command plan for one external DeepLearning job."""

    job_id: str
    stage: str
    model: str
    symbol: str
    target: str
    cwd: str
    argv: tuple[str, ...]
    feature_path: str
    prediction_path: str
    state_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "model": self.model,
            "symbol": self.symbol,
            "target": self.target,
            "cwd": self.cwd,
            "argv": list(self.argv),
            "feature_path": self.feature_path,
            "prediction_path": self.prediction_path,
            "state_path": self.state_path,
        }


def _tokenize_symbol(symbol: str) -> str:
    return normalize_forecast_symbol(symbol).replace("/", "_").replace("-", "_")


def deep_learning_target_name(symbol: str, *, suffix: str = "close") -> str:
    """Map a LuminaQuant symbol to the DeepLearning target naming convention."""
    token = _tokenize_symbol(symbol)
    base = token.split("_", 1)[0]
    clean_suffix = str(suffix or "close").strip().lower()
    return f"{base}_{clean_suffix}"


def _artifact_path(base: str, stem: str, extension: str) -> str:
    root = Path(str(base or ""))
    if root.suffix:
        return str(root)
    suffix = extension.lstrip(".") or "parquet"
    return str(root / f"{stem}.{suffix}")


def _state_path(base: str, job_id: str) -> str:
    path = Path(str(base or ""))
    if path.suffix:
        return str(path.with_name(f"{path.stem}.{job_id}{path.suffix}"))
    return str(path / f"{job_id}.json")


def _timeframe_seconds(value: str) -> int:
    token = str(value or "").strip().lower()
    if not token:
        return 60
    digits = "".join(ch for ch in token if ch.isdigit())
    unit = token[len(digits) :]
    count = int(digits or "1")
    if unit == "s":
        return count
    if unit == "m":
        return count * 60
    if unit == "h":
        return count * 3600
    if unit == "d":
        return count * 86_400
    if unit == "w":
        return count * 7 * 86_400
    return 60


def _append_flag(argv: list[str], enabled: bool, positive: str, negative: str = "") -> None:
    if enabled:
        argv.append(positive)
    elif negative:
        argv.append(negative)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _first_numeric(
    mapping: Mapping[str, Any], keys: Sequence[str]
) -> tuple[str, float] | tuple[str, None]:
    for key in keys:
        if key in mapping:
            parsed = _finite_float(mapping[key])
            if parsed is not None:
                return key, parsed
    return "", None


def _trial_params(trial: Mapping[str, Any]) -> dict[str, Any]:
    params = trial.get("params")
    if isinstance(params, Mapping):
        return dict(params)
    return {}


def _search_bounds(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, Mapping):
        low = _finite_float(raw.get("low"))
        high = _finite_float(raw.get("high"))
    elif isinstance(raw, Sequence) and not isinstance(raw, str) and len(raw) >= 2:
        low = _finite_float(raw[0])
        high = _finite_float(raw[-1])
    else:
        return None
    if low is None or high is None or high <= low:
        return None
    return low, high


def _is_search_boundary(value: Any, bounds: tuple[float, float], tolerance: float) -> bool:
    parsed = _finite_float(value)
    if parsed is None:
        return False
    low, high = bounds
    span = high - low
    edge = max(0.0, min(0.50, float(tolerance))) * span
    return parsed <= low + edge or parsed >= high - edge


def _metric_defaults_to_maximize(metric_key: str) -> bool:
    lowered = metric_key.lower()
    return not any(hint in lowered for hint in _LOWER_IS_BETTER_HINTS)


def recommend_deep_learning_gpu_parallelism(
    *,
    free_vram_gb: float | None = None,
    gpu_utilization_pct: float | None = None,
    dataloader_bottleneck: bool = False,
    requested_jobs: int = 1,
    min_free_vram_gb: float = 2.0,
    max_utilization_pct: float = 85.0,
) -> dict[str, Any]:
    """Decide whether small DeepLearning jobs should share one GPU."""
    jobs = max(1, int(requested_jobs))
    min_free = max(0.0, float(min_free_vram_gb))
    max_util = max(1.0, min(100.0, float(max_utilization_pct)))
    if jobs <= 1:
        return {
            "decision": "single_job",
            "parallel_recommended": False,
            "reason": "Only one training job is requested.",
        }

    if free_vram_gb is None or gpu_utilization_pct is None:
        return {
            "decision": "measure_gpu_first",
            "parallel_recommended": False,
            "reason": "Parallelism depends on free VRAM, GPU utilization, and DataLoader pressure.",
            "required_inputs": ["free_vram_gb", "gpu_utilization_pct", "dataloader_bottleneck"],
        }

    free_vram = float(free_vram_gb)
    utilization = float(gpu_utilization_pct)
    if free_vram < min_free:
        return {
            "decision": "reduce_batch_or_enable_amp",
            "parallel_recommended": False,
            "reason": "Free VRAM is below the configured safety floor.",
            "free_vram_gb": free_vram,
            "min_free_vram_gb": min_free,
        }
    if utilization >= max_util:
        return {
            "decision": "parallel_not_recommended",
            "parallel_recommended": False,
            "reason": "GPU utilization is already high; parallel jobs would mostly share saturated compute.",
            "gpu_utilization_pct": utilization,
            "max_utilization_pct": max_util,
        }
    if dataloader_bottleneck:
        return {
            "decision": "fix_dataloader_before_parallelism",
            "parallel_recommended": False,
            "reason": "Low utilization caused by input loading should be fixed before adding GPU jobs.",
        }
    return {
        "decision": "parallel_ok",
        "parallel_recommended": True,
        "reason": "VRAM has headroom and GPU utilization is below the configured saturation threshold.",
        "requested_jobs": jobs,
    }


def evaluate_deep_learning_chronological_sanity(
    *,
    reference_validation_score: float,
    chronological_validation_score: float,
    maximize: bool = True,
    max_relative_drop: float = 0.25,
) -> dict[str, Any]:
    """Check that selected HPO params do not collapse on a later pre-test split."""
    reference = _finite_float(reference_validation_score)
    chronological = _finite_float(chronological_validation_score)
    if reference is None or chronological is None:
        return {
            "ok": False,
            "reason": "Scores must be finite numbers.",
            "relative_drop": None,
        }
    denominator = max(abs(reference), 1e-12)
    if maximize:
        relative_drop = max(0.0, (reference - chronological) / denominator)
    else:
        relative_drop = max(0.0, (chronological - reference) / denominator)
    allowed_drop = max(0.0, float(max_relative_drop))
    return {
        "ok": relative_drop <= allowed_drop,
        "reference_validation_score": reference,
        "chronological_validation_score": chronological,
        "relative_drop": relative_drop,
        "max_relative_drop": allowed_drop,
        "interpretation": (
            "pass"
            if relative_drop <= allowed_drop
            else "selected_params_need_alternative_or_partial_hpo"
        ),
    }


def select_stable_deep_learning_hpo_trial(
    trials: Sequence[Mapping[str, Any]],
    *,
    metric_key: str = "",
    train_metric_key: str = "",
    maximize: bool | None = None,
    search_space: Mapping[str, Any] | None = None,
    top_fraction: float = 0.10,
    min_top_trials: int = 5,
    max_train_val_gap: float = 0.15,
    boundary_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Select a stable top HPO trial instead of blindly taking the best score."""
    diagnostics: list[dict[str, Any]] = []
    bounds_by_name = {
        str(name): bounds
        for name, spec in dict(search_space or {}).items()
        if (bounds := _search_bounds(spec)) is not None
    }
    for index, trial in enumerate(trials):
        val_key, validation_score = _first_numeric(
            trial,
            (metric_key,) if metric_key else _VALIDATION_SCORE_KEYS,
        )
        if validation_score is None:
            continue
        train_key, train_score = _first_numeric(
            trial,
            (train_metric_key,) if train_metric_key else _TRAIN_SCORE_KEYS,
        )
        resolved_maximize = _metric_defaults_to_maximize(val_key)
        if maximize is not None:
            resolved_maximize = bool(maximize)
        params = _trial_params(trial)
        gap = None if train_score is None else abs(float(train_score) - float(validation_score))
        flags: list[str] = []
        penalty = 0.0
        if gap is not None and gap > max_train_val_gap:
            flags.append("wide_train_validation_gap")
            penalty += gap / max(max_train_val_gap, 1e-12)

        for name, value in params.items():
            lowered = str(name).lower()
            numeric_value = _finite_float(value)
            if (
                numeric_value is not None
                and lowered in _REGULARIZATION_PARAM_NAMES
                and numeric_value <= 0
            ):
                flags.append(f"missing_regularization:{name}")
                penalty += 1.0
            bounds = bounds_by_name.get(str(name))
            if bounds is not None and _is_search_boundary(value, bounds, boundary_tolerance):
                if lowered in _LR_PARAM_NAMES:
                    flags.append(f"learning_rate_search_boundary:{name}")
                    penalty += 2.0
                elif lowered in _MODEL_SIZE_PARAM_NAMES:
                    flags.append(f"model_size_search_boundary:{name}")
                    penalty += 1.0
                else:
                    flags.append(f"search_boundary:{name}")
                    penalty += 0.5

        objective_score = validation_score if resolved_maximize else -validation_score
        diagnostics.append(
            {
                "trial_id": trial.get("trial_id", trial.get("number", trial.get("id", index))),
                "validation_metric": val_key,
                "validation_score": validation_score,
                "train_metric": train_key,
                "train_score": train_score,
                "train_validation_gap": gap,
                "objective_score": objective_score,
                "flags": flags,
                "stability_penalty": penalty,
                "stable": not flags,
                "params": params,
                "trial": dict(trial),
            }
        )

    ranked = sorted(diagnostics, key=lambda item: item["objective_score"], reverse=True)
    if not ranked:
        return {
            "selected": None,
            "ranked": [],
            "candidate_count": 0,
            "top_candidate_count": 0,
            "policy": {
                "top_fraction": top_fraction,
                "min_top_trials": min_top_trials,
                "max_train_val_gap": max_train_val_gap,
                "boundary_tolerance": boundary_tolerance,
            },
        }

    top_count = min(
        len(ranked),
        max(1, math.ceil(len(ranked) * max(0.01, min(1.0, top_fraction))), int(min_top_trials)),
    )
    top_trials = ranked[:top_count]
    stable_top = [item for item in top_trials if item["stable"]]
    if stable_top:
        selected = max(stable_top, key=lambda item: item["objective_score"])
    else:
        selected = min(
            top_trials,
            key=lambda item: (item["stability_penalty"], -item["objective_score"]),
        )

    return {
        "selected": selected,
        "ranked": ranked,
        "candidate_count": len(ranked),
        "top_candidate_count": top_count,
        "policy": {
            "top_fraction": top_fraction,
            "min_top_trials": min_top_trials,
            "max_train_val_gap": max_train_val_gap,
            "boundary_tolerance": boundary_tolerance,
            "selection_rule": "best stable trial inside top validation band; otherwise lowest-risk top trial",
        },
    }


def build_deep_learning_operational_guardrails(cfg: DeepLearningRuntimeConfig) -> dict[str, Any]:
    """Build auditable DL training/HPO/drift guardrails for the pipeline manifest."""
    return {
        "gpu_parallel_training": {
            "basis": "Model size alone is not a parallelism signal.",
            "decision_inputs": ["free_vram_gb", "gpu_utilization_pct", "dataloader_bottleneck"],
            "requested_parallel_jobs": int(cfg.gpu_parallel_jobs),
            "thresholds": {
                "min_free_vram_gb": float(cfg.gpu_parallel_min_free_vram_gb),
                "max_utilization_pct": float(cfg.gpu_parallel_max_utilization_pct),
            },
            "rules": [
                "free_vram_ok + low_gpu_utilization + no_dataloader_bottleneck => parallel_ok",
                "free_vram_ok + high_gpu_utilization => parallel_not_recommended",
                "low_free_vram => reduce_batch_or_enable_amp",
            ],
            "default_decision": recommend_deep_learning_gpu_parallelism(
                requested_jobs=int(cfg.gpu_parallel_jobs),
                min_free_vram_gb=float(cfg.gpu_parallel_min_free_vram_gb),
                max_utilization_pct=float(cfg.gpu_parallel_max_utilization_pct),
            ),
        },
        "split_contract": {
            "future_test": "time_based_holdout_used_once_for_final_generalization_check",
            "pre_test_train_validation": "window_shuffle_split_for_candidate_hpo_search",
            "interpretation": (
                "Shuffle validation is data-efficient for weight learning and HPO candidate search, "
                "but its score may be optimistic because overlapping windows can leak regime similarity."
            ),
            "final_refit": "After params are selected, refit on train+validation before the single future test.",
        },
        "hpo_selection": {
            "objective": "prefer stable top trials over one aggressive best-score spike",
            "top_fraction": float(cfg.hpo_top_trial_fraction),
            "min_top_trials": int(cfg.hpo_min_top_trials),
            "max_train_val_gap": float(cfg.hpo_max_train_val_gap),
            "boundary_tolerance": float(cfg.hpo_boundary_tolerance),
            "prefer": [
                "validation score inside top band",
                "small train/validation gap",
                "stable validation curve",
                "learning rate away from search boundary",
                "nonzero dropout or weight_decay when those params exist",
                "model size not pinned to maximum search boundary",
                "nearby trials with similar scores",
            ],
            "avoid": [
                "single isolated best trial",
                "dropout=0 and weight_decay=0",
                "learning rate on search boundary",
                "hidden_dim/layers at search maximum",
                "train score much stronger than validation score",
                "one-epoch validation spike",
            ],
        },
        "chronological_sanity_check": {
            "enabled": bool(cfg.chronological_sanity_check),
            "purpose": "Check selected params on the later pre-test window without rerunning full HPO.",
            "pre_test_split": {
                "train_fraction": 1.0 - float(cfg.chronological_sanity_val_fraction),
                "chrono_val_fraction": float(cfg.chronological_sanity_val_fraction),
            },
            "max_relative_drop": float(cfg.chronological_sanity_max_relative_drop),
            "action_on_fail": "Inspect top stable alternatives or run partial HPO before final refit.",
        },
        "drift_response": {
            "principle": "Drift alone does not trigger HPO.",
            "sequence": [
                "detect_drift",
                "confirm_performance_degradation",
                "retrain_latest_data_with_existing_params",
                "run_partial_hpo_if_retrain_fails",
                "run_full_hpo_if_partial_hpo_fails_or_param_ranking_changes",
            ],
            "partial_hpo_trials": int(cfg.partial_hpo_trials),
            "decision_table": [
                {"state": "no_drift_and_performance_ok", "action": "keep"},
                {"state": "drift_and_performance_ok", "action": "monitor"},
                {"state": "drift_and_performance_down", "action": "retrain_existing_params"},
                {"state": "retrain_recovers", "action": "no_hpo"},
                {"state": "retrain_fails", "action": "partial_hpo"},
                {"state": "partial_hpo_fails_or_ranking_changes", "action": "full_hpo_candidate"},
            ],
        },
    }


def _deep_learning_command(
    cfg: DeepLearningRuntimeConfig,
    *,
    model: str,
    target: str,
    feature_path: str,
    freq: str,
    tuning: bool,
) -> tuple[str, ...]:
    argv = [
        "uv",
        "run",
        "python",
        "main.py",
        "-p",
        cfg.config_path,
        "--series-path",
        feature_path,
        "--target",
        target,
        "--model",
        model,
        "--seq-len",
        str(cfg.seq_len),
        "--pred-len",
        str(cfg.pred_len),
        "--total-pred",
        str(cfg.total_pred),
        "--freq",
        freq,
        "--gpus",
        cfg.gpus,
    ]
    if cfg.dataset_path:
        argv.extend(["--dataset-path", cfg.dataset_path])
    if cfg.epochs > 0:
        argv.extend(["--epochs", str(cfg.epochs)])
    if tuning:
        argv.extend(["--hpo", "--n-trials", str(cfg.hpo_trials)])
        if cfg.epochs_per_trial > 0:
            argv.extend(["--epochs-per-trial", str(cfg.epochs_per_trial)])
    else:
        argv.append("--no-hpo")
    _append_flag(argv, cfg.force_train, "--force_train")
    _append_flag(argv, cfg.resume, "--resume", "--no-resume")
    _append_flag(
        argv,
        cfg.run_test_after_fit,
        "--run-test-after-fit",
        "--no-run-test-after-fit",
    )
    _append_flag(argv, cfg.no_upload, "--no-upload", "--upload")
    argv.append("--pred")
    return tuple(argv)


def _strategy_params(
    cfg: DeepLearningRuntimeConfig,
    models: tuple[str, ...],
    *,
    horizon_seconds: int,
) -> dict[str, Any]:
    return {
        "forecast_path": cfg.prediction_path,
        "models": ",".join(models),
        "horizon_seconds": int(horizon_seconds),
        "max_forecast_age_seconds": 86_400,
        "entry_threshold_bps": float(cfg.entry_threshold_bps),
        "exit_threshold_bps": float(cfg.exit_threshold_bps),
        "min_model_agreement": float(cfg.min_model_agreement),
        "max_dispersion_bps": float(cfg.max_dispersion_bps),
        "min_confidence": float(cfg.min_confidence),
        "min_models": min(max(1, int(cfg.min_model_coverage)), len(models)),
        "target_allocation": float(cfg.target_allocation),
        "stop_loss_pct": float(cfg.stop_loss_pct),
        "take_profit_pct": float(cfg.take_profit_pct),
    }


def build_deep_learning_strategy_profiles(runtime: RuntimeConfig) -> dict[str, dict[str, Any]]:
    """Build ready-to-materialize strategy parameter profiles."""
    cfg = runtime.deep_learning
    models = normalize_deep_learning_models(cfg.models)
    ensemble_models = normalize_deep_learning_models(cfg.ensemble_models, default=models)
    freq = str(cfg.freq or runtime.trading.timeframe or "1m")
    horizon_seconds = _timeframe_seconds(freq) * int(cfg.pred_len)
    profiles: dict[str, dict[str, Any]] = {
        "ensemble": {
            "strategy": "DeepLearningForecastGateStrategy",
            "params": _strategy_params(cfg, ensemble_models, horizon_seconds=horizon_seconds),
            "best_params_path": (
                "best_optimized_parameters/DeepLearningForecastGateStrategy/best_params.json"
            ),
        }
    }
    for model in models:
        profiles[model] = {
            "strategy": "DeepLearningForecastGateStrategy",
            "params": _strategy_params(cfg, (model,), horizon_seconds=horizon_seconds),
            "best_params_path": (
                "best_optimized_parameters/DeepLearningForecastGateStrategy/"
                f"best_params.{model}.json"
            ),
        }
    return profiles


def build_deep_learning_jobs(
    runtime: RuntimeConfig, *, tuning: bool | None = None
) -> list[DeepLearningPipelineJob]:
    """Build external DeepLearning command jobs without executing them."""
    cfg = runtime.deep_learning
    models = normalize_deep_learning_models(cfg.models)
    freq = str(cfg.freq or runtime.trading.timeframe or "1m")
    use_tuning = cfg.hpo_enabled if tuning is None else bool(tuning)
    jobs: list[DeepLearningPipelineJob] = []
    for symbol in runtime.trading.symbols:
        target = deep_learning_target_name(symbol, suffix=cfg.target_suffix)
        feature_stem = f"{target}_{freq}"
        feature_path = _artifact_path(cfg.feature_matrix_path, feature_stem, cfg.feature_format)
        for model in models:
            job_id = f"{target}.{model}.{freq}"
            state_path = _state_path(cfg.training_state_path, job_id)
            jobs.append(
                DeepLearningPipelineJob(
                    job_id=job_id,
                    stage="tune_train_predict" if use_tuning else "train_predict",
                    model=model,
                    symbol=str(symbol),
                    target=target,
                    cwd=cfg.repo_path,
                    argv=_deep_learning_command(
                        cfg,
                        model=model,
                        target=target,
                        feature_path=feature_path,
                        freq=freq,
                        tuning=use_tuning,
                    ),
                    feature_path=feature_path,
                    prediction_path=cfg.prediction_path,
                    state_path=state_path,
                )
            )
    return jobs


def _deep_learning_manifest_stages(cfg: DeepLearningRuntimeConfig) -> list[str]:
    stages = ["export_feature_matrices"]
    if cfg.hpo_enabled:
        stages.extend(["tune_models", "select_stable_hpo_params"])
        if cfg.chronological_sanity_check:
            stages.append("chronological_sanity_check")
    else:
        stages.append("skip_tuning")
    stages.extend(
        [
            "train_predict_models",
            "validate_forecast_artifacts",
            "materialize_strategy_params",
            "backtest_strategy",
            "paper_or_shadow_live_gate",
        ]
    )
    return stages


def build_deep_learning_pipeline_manifest(
    runtime: RuntimeConfig,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the integrated DL sidecar -> forecast artifact -> strategy manifest."""
    cfg = runtime.deep_learning
    generated = generated_at or datetime.now(UTC)
    models = normalize_deep_learning_models(cfg.models)
    jobs = build_deep_learning_jobs(runtime)
    return {
        "version": PIPELINE_MANIFEST_VERSION,
        "generated_at": generated.isoformat(),
        "enabled": bool(cfg.enabled),
        "external_repo": {
            "path": cfg.repo_path,
            "entrypoint": "main.py",
            "config_path": cfg.config_path,
            "test_config_path": cfg.test_config_path,
        },
        "artifact_contract": {
            "feature_matrix_path": cfg.feature_matrix_path,
            "feature_format": cfg.feature_format,
            "prediction_path": cfg.prediction_path,
            "metrics_path": cfg.metrics_path,
            "accepted_prediction_formats": ["csv", "json", "jsonl", "parquet"],
            "canonical_prediction_columns": [
                "model_code",
                "dbcode",
                "pred_date",
                "Date",
                "value",
                "pred_return",
                "confidence",
            ],
            "models": list(models),
            "min_model_coverage": int(cfg.min_model_coverage),
        },
        "stages": _deep_learning_manifest_stages(cfg),
        "operational_guardrails": build_deep_learning_operational_guardrails(cfg),
        "training_state": {
            "path": cfg.training_state_path,
            "statuses": ["planned", "running", "completed", "failed", "skipped"],
            "job_count": len(jobs),
        },
        "jobs": [job.to_dict() for job in jobs],
        "strategy_profiles": build_deep_learning_strategy_profiles(runtime),
        "lumina_commands": {
            "validate_artifacts": ["uv", "run", "lq", "deep-learning", "validate-artifacts"],
            "write_manifest": ["uv", "run", "lq", "deep-learning", "plan", "--write"],
        },
    }


def write_deep_learning_pipeline_manifest(
    runtime: RuntimeConfig,
    output_path: str | Path | None = None,
) -> Path:
    """Write the pipeline manifest JSON and return its path."""
    cfg = runtime.deep_learning
    path = Path(output_path or cfg.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_deep_learning_pipeline_manifest(runtime)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_deep_learning_forecast_artifacts(runtime: RuntimeConfig) -> dict[str, Any]:
    """Validate forecast artifact coverage for the configured symbols/models."""
    cfg = runtime.deep_learning
    models = normalize_deep_learning_models(cfg.models)
    store = DeepLearningForecastStore(cfg.prediction_path, models=models)
    symbols: dict[str, Any] = {}
    for symbol in runtime.trading.symbols:
        normalized = normalize_forecast_symbol(symbol)
        coverage = store.model_coverage(normalized)
        present = [model for model, count in coverage.items() if count > 0]
        missing = [model for model in models if model not in present]
        symbols[str(symbol)] = {
            "normalized_symbol": normalized,
            "records_by_model": coverage,
            "present_models": present,
            "missing_models": missing,
            "meets_min_model_coverage": len(present) >= int(cfg.min_model_coverage),
        }
    all_ok = all(item["meets_min_model_coverage"] for item in symbols.values())
    return {
        "prediction_path": cfg.prediction_path,
        "record_count": store.record_count,
        "models": list(models),
        "min_model_coverage": int(cfg.min_model_coverage),
        "symbols": symbols,
        "ok": bool(all_ok),
    }


__all__ = [
    "PIPELINE_MANIFEST_VERSION",
    "DeepLearningPipelineJob",
    "build_deep_learning_jobs",
    "build_deep_learning_operational_guardrails",
    "build_deep_learning_pipeline_manifest",
    "build_deep_learning_strategy_profiles",
    "deep_learning_target_name",
    "evaluate_deep_learning_chronological_sanity",
    "recommend_deep_learning_gpu_parallelism",
    "select_stable_deep_learning_hpo_trial",
    "validate_deep_learning_forecast_artifacts",
    "write_deep_learning_pipeline_manifest",
]
