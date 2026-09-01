"""Optuna tuning for the hybrid online portfolio governor."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.portfolio.hybrid_objective import (
    HYBRID_LOCKED_OBJECTIVE_PROFILE,
    HYBRID_OBJECTIVE_PROFILES,
    hybrid_online_objective_from_payload,
    hybrid_online_objective_policy,
)
from lumina_quant.portfolio.optimizer_core import safe_float as _safe_float
from lumina_quant.optimization.search_policy import (
    optimization_search_policy_payload,
    run_optuna_study,
)

try:
    import optuna
except ModuleNotFoundError:
    optuna = None

ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "run_hybrid_online_portfolio",
    ROOT / "run_hybrid_online_portfolio.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load run_hybrid_online_portfolio")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

OUTPUT_DIR = _MOD.GROUP_ROOT / "portfolio_hybrid_online_optuna_current"
LOCKED_OBJECTIVE_PROFILE = HYBRID_LOCKED_OBJECTIVE_PROFILE
OBJECTIVE_PROFILES = HYBRID_OBJECTIVE_PROFILES


def _require_optuna():
    if optuna is None:
        raise RuntimeError("Optuna is required; run with `uv run --extra optimize ...`")
    return optuna


def _objective_policy_for_profile(profile: str) -> dict:
    return hybrid_online_objective_policy(profile)


def _objective_from_payload(payload: dict, *, profile: str) -> float:
    return hybrid_online_objective_from_payload(payload, profile=profile)


def _evaluate_config(
    config: _MOD.HybridOnlineConfig,
    *,
    split_config: _MOD.HybridSplitConfig,
) -> dict:
    historical_active = _MOD._historical_active_rows(split_config=split_config)
    refreshed_active, refreshed_benchmarks = _MOD._refreshed_rows(split_config=split_config)
    refreshed_health_metrics = {
        row["name"]: dict(row.get("oos") or {}) for row in refreshed_active + refreshed_benchmarks
    }
    historical_config = _MOD.HybridOnlineConfig(
        **({**asdict(config), "use_current_health_priors": False})
    )
    historical_result = _MOD.run_hybrid_online_allocator(
        historical_active,
        config=historical_config,
        refreshed_health_metrics=None,
        split_config=split_config,
    )
    refreshed_result = _MOD.run_hybrid_online_allocator(
        refreshed_active,
        config=config,
        refreshed_health_metrics=refreshed_health_metrics,
        split_config=split_config,
    )
    refreshed_rows = _MOD._comparison_rows(
        hybrid_result=refreshed_result,
        benchmarks=refreshed_benchmarks,
        active_rows=refreshed_active,
    )
    refreshed_by_name = {row["name"]: row for row in refreshed_rows}
    pair_alloc_weights = [
        _safe_float((alloc.get("weights") or {}).get("pair_tactical_mode"), 0.0)
        for alloc in list(refreshed_result.get("allocations") or [])
    ]
    payload = {
        "config": asdict(config),
        "scenarios": {
            "historical_saved_baseline": {"split_metrics": historical_result["split_metrics"]},
            "refreshed_latest_tail": {"split_metrics": refreshed_result["split_metrics"]},
        },
        "readiness": {
            "beats_cash_refreshed": bool(
                _safe_float(
                    refreshed_by_name["hybrid_online_portfolio"].get(
                        "total_return", refreshed_by_name["hybrid_online_portfolio"].get("return")
                    ),
                    0.0,
                )
                > _safe_float(
                    refreshed_by_name["risk_off_cash"].get(
                        "total_return", refreshed_by_name["risk_off_cash"].get("return")
                    ),
                    0.0,
                )
            ),
            "beats_pair_tactical_refreshed": bool(
                _safe_float(
                    refreshed_by_name["hybrid_online_portfolio"].get(
                        "total_return", refreshed_by_name["hybrid_online_portfolio"].get("return")
                    ),
                    0.0,
                )
                > _safe_float(
                    refreshed_by_name["pair_tactical_mode"].get(
                        "total_return", refreshed_by_name["pair_tactical_mode"].get("return")
                    ),
                    0.0,
                )
            ),
            "pair_cap_respected": bool(
                max(pair_alloc_weights or [0.0]) <= config.pair_weight_cap + 1e-9
            ),
        },
    }
    return payload


def _build_config(
    trial: optuna.Trial,
    *,
    warmup_ratio: float,
    warmup_days: int | None,
) -> _MOD.HybridOnlineConfig:
    return _MOD.HybridOnlineConfig(
        variant=trial.suggest_categorical(
            "variant", ["dynamic_default", "fixed_default", "disagreement_switching"]
        ),
        warmup_ratio=float(warmup_ratio),
        warmup_days=None if warmup_days is None else int(warmup_days),
        lookback_days=trial.suggest_int("lookback_days", 10, 28),
        default_boost=trial.suggest_float("default_boost", 0.15, 0.5),
        sticky_default_bonus=trial.suggest_float("sticky_default_bonus", 0.0, 0.25),
        switch_margin=trial.suggest_float("switch_margin", 0.0, 0.2),
        score_temperature=trial.suggest_float("score_temperature", 0.7, 1.3),
        min_positive_score=trial.suggest_float("min_positive_score", 0.0, 0.2),
        pair_score_boost=trial.suggest_float("pair_score_boost", 0.0, 0.25),
        disagreement_threshold=trial.suggest_float("disagreement_threshold", 0.05, 0.5),
        disagreement_cash_scale=trial.suggest_float("disagreement_cash_scale", 0.4, 1.0),
        pair_weight_cap=trial.suggest_float("pair_weight_cap", 0.15, 0.3),
        diversified_weight_cap=trial.suggest_float("diversified_weight_cap", 0.7, 0.9),
        pair_pbo_penalty_scale=trial.suggest_float("pair_pbo_penalty_scale", 1.5, 4.0),
        pair_sparsity_penalty_scale=trial.suggest_float("pair_sparsity_penalty_scale", 1.0, 4.0),
        negative_health_floor=trial.suggest_float("negative_health_floor", 0.05, 0.3),
        mixed_health_floor=trial.suggest_float("mixed_health_floor", 0.4, 0.75),
        use_current_health_priors=True,
    )


def _search_policy_for_profile(profile: str, *, n_trials: int) -> dict:
    return optimization_search_policy_payload(
        search_method="optuna",
        objective_policy=_objective_policy_for_profile(profile),
        selection_inputs=("train", "validation"),
        extra={
            "requested_trials": max(1, int(n_trials)),
            "direction": "maximize",
            "sampler": "TPESampler",
            "seed": 42,
            "selection_policy": "best_objective_desc",
            "search_space_source": "scripts/research/optuna_tune_hybrid_online_portfolio.py::_build_config",
        },
    )


def main() -> None:
    optuna_mod = _require_optuna()
    parser = argparse.ArgumentParser(description=__doc__)
    _MOD.add_split_config_arguments(parser)
    parser.add_argument(
        "--warmup-ratio", type=float, default=_MOD.HybridOnlineConfig().warmup_ratio
    )
    parser.add_argument("--warmup-days", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--objective-profile",
        choices=OBJECTIVE_PROFILES,
        default=LOCKED_OBJECTIVE_PROFILE,
    )
    parser.add_argument("--n-trials", type=int, default=24)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_config = _MOD.split_config_from_args(args)

    best_payload: dict | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_payload
        config = _build_config(
            trial,
            warmup_ratio=float(args.warmup_ratio),
            warmup_days=None if args.warmup_days is None else int(args.warmup_days),
        )
        payload = _evaluate_config(config, split_config=split_config)
        payload["objective_profile"] = args.objective_profile
        payload["objective_policy"] = _objective_policy_for_profile(args.objective_profile)
        trial_score = float(_objective_from_payload(payload, profile=args.objective_profile))
        payload["objective"] = trial_score
        trial.set_user_attr("payload", payload)
        if best_payload is None or trial_score > best_payload["objective"]:
            best_payload = payload
        return trial_score

    study = run_optuna_study(
        optuna_module=optuna_mod,
        objective=objective,
        n_trials=max(1, int(args.n_trials)),
        seed=42,
        show_progress_bar=False,
    )

    trials = []
    for trial in study.trials:
        payload = trial.user_attrs["payload"]
        payload["objective"] = float(
            _objective_from_payload(payload, profile=args.objective_profile)
        )
        payload = {
            "trial_number": trial.number,
            "objective": trial.value,
            **payload,
        }
        trials.append(payload)
    trials.sort(key=lambda row: float(row["objective"]), reverse=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"hybrid_online_optuna_{stamp}.json"
    latest_json = output_dir / "hybrid_online_optuna_latest.json"
    md_path = output_dir / f"hybrid_online_optuna_{stamp}.md"
    payload = {
        "artifact_kind": "hybrid_online_portfolio_optuna",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "objective_profile": args.objective_profile,
        "objective_policy": _objective_policy_for_profile(args.objective_profile),
        "search_policy": _search_policy_for_profile(
            args.objective_profile,
            n_trials=int(args.n_trials),
        ),
        "split_windows": split_config.as_payload(),
        "warmup_ratio": float(args.warmup_ratio),
        "fixed_warmup_days": None if args.warmup_days is None else int(args.warmup_days),
        "best_trial": trials[0],
        "top_trials": trials[:10],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    json_path.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    lines = [
        "# Hybrid online portfolio optuna tuning",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- trials: `{len(trials)}`",
        f"- objective_profile: `{args.objective_profile}`",
        f"- objective_policy: `{payload['objective_policy']['objective_policy']}`",
        f"- locked_oos_label: `{payload['objective_policy']['locked_oos_label']}`",
        "",
        "## Best trial",
        "",
        "```json",
        json.dumps(trials[0], indent=2, sort_keys=True),
        "```",
        "",
        "## Top trials",
        "",
        "| Trial | Objective | Variant | Refresh OOS | Refresh Sharpe | Refresh Val | Refresh Train | Beats cash | Beats pair |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in trials[:10]:
        ref = row["scenarios"]["refreshed_latest_tail"]["split_metrics"]
        cfg = row["config"]
        lines.append(
            f"| {row['trial_number']} | {float(row['objective']):.4f} | `{cfg['variant']}` | {_safe_float(ref['oos'].get('total_return', ref['oos'].get('return')), 0.0):+.4%} | {_safe_float(ref['oos'].get('sharpe'), 0.0):.4f} | {_safe_float(ref['val'].get('total_return', ref['val'].get('return')), 0.0):+.4%} | {_safe_float(ref['train'].get('total_return', ref['train'].get('return')), 0.0):+.4%} | `{row['readiness']['beats_cash_refreshed']}` | `{row['readiness']['beats_pair_tactical_refreshed']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(latest_json.resolve()))
    print(str(md_path.resolve()))


if __name__ == "__main__":
    main()
