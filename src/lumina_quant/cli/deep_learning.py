"""lq deep-learning — DeepLearning sidecar pipeline planning commands."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _resolve_config_path(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "config", "") or "").strip()
    return explicit or os.environ.get("LQ_CONFIG_PATH", "config.yaml")


def _load_runtime(args: argparse.Namespace):
    from lumina_quant.configuration import RuntimeConfig, load_runtime_config

    try:
        return load_runtime_config(_resolve_config_path(args))
    except FileNotFoundError:
        return RuntimeConfig()


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    token = value.strip()
    if not token:
        return ""
    try:
        return json.loads(token)
    except Exception:
        pass
    try:
        return float(token)
    except Exception:
        return token


def _load_json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_trial_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("trials", "study_trials", "results", "records"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(item) for item in rows if isinstance(item, dict)]
        return [dict(payload)]
    return []


def _load_hpo_trials(path_value: str) -> list[dict[str, Any]]:
    path = Path(path_value)
    if path.suffix.lower() == ".csv":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = {}
                params: dict[str, Any] = {}
                for key, value in raw.items():
                    parsed = _parse_scalar(value)
                    if key.startswith("params_"):
                        params[key.removeprefix("params_")] = parsed
                    elif key.startswith("params."):
                        params[key.removeprefix("params.")] = parsed
                    else:
                        row[key] = parsed
                existing_params = row.get("params")
                if isinstance(existing_params, dict):
                    params = {**existing_params, **params}
                if params:
                    row["params"] = params
                rows.append(row)
        return rows
    return _extract_trial_rows(_load_json_payload(path))


def _load_search_space(path_value: str) -> dict[str, Any]:
    if not path_value:
        return {}
    payload = _load_json_payload(Path(path_value))
    if isinstance(payload, dict):
        return payload
    return {}


def cmd_plan(args: argparse.Namespace) -> int:
    from lumina_quant.workflows.deep_learning_pipeline import (
        build_deep_learning_pipeline_manifest,
        write_deep_learning_pipeline_manifest,
    )

    runtime = _load_runtime(args)
    output_arg = getattr(args, "write", "")
    output_path = None if output_arg == "__default__" else str(output_arg or "").strip()
    if output_arg:
        path = write_deep_learning_pipeline_manifest(runtime, output_path)
        print(f"[lq deep-learning plan] wrote {path}")
        if not bool(getattr(args, "json", False)):
            return 0
    manifest = build_deep_learning_pipeline_manifest(runtime)
    _print_payload(manifest, as_json=bool(getattr(args, "json", False)))
    return 0


def cmd_validate_artifacts(args: argparse.Namespace) -> int:
    from lumina_quant.workflows.deep_learning_pipeline import (
        validate_deep_learning_forecast_artifacts,
    )

    runtime = _load_runtime(args)
    result = validate_deep_learning_forecast_artifacts(runtime)
    _print_payload(result, as_json=bool(getattr(args, "json", False)))
    return 0 if result.get("ok") or not bool(getattr(args, "strict", False)) else 1


def cmd_strategy_config(args: argparse.Namespace) -> int:
    from lumina_quant.workflows.deep_learning_pipeline import build_deep_learning_strategy_profiles

    runtime = _load_runtime(args)
    profiles = build_deep_learning_strategy_profiles(runtime)
    profile_name = str(getattr(args, "profile", "ensemble") or "ensemble")
    if profile_name == "all":
        payload = profiles
    else:
        payload = profiles.get(profile_name)
        if payload is None:
            print(
                f"[error] unknown profile={profile_name!r}; available={sorted(profiles)}",
                file=sys.stderr,
            )
            return 1
    _print_payload(dict(payload), as_json=bool(getattr(args, "json", False)))
    return 0


def cmd_hpo_select(args: argparse.Namespace) -> int:
    from lumina_quant.workflows.deep_learning_pipeline import select_stable_deep_learning_hpo_trial

    runtime = _load_runtime(args)
    cfg = runtime.deep_learning
    trials = _load_hpo_trials(str(getattr(args, "trials", "")))
    top_fraction = (
        float(args.top_fraction)
        if getattr(args, "top_fraction", None) is not None
        else float(cfg.hpo_top_trial_fraction)
    )
    min_top_trials = (
        int(args.min_top_trials)
        if getattr(args, "min_top_trials", None) is not None
        else int(cfg.hpo_min_top_trials)
    )
    max_train_val_gap = (
        float(args.max_train_val_gap)
        if getattr(args, "max_train_val_gap", None) is not None
        else float(cfg.hpo_max_train_val_gap)
    )
    boundary_tolerance = (
        float(args.boundary_tolerance)
        if getattr(args, "boundary_tolerance", None) is not None
        else float(cfg.hpo_boundary_tolerance)
    )
    result = select_stable_deep_learning_hpo_trial(
        trials,
        metric_key=str(getattr(args, "metric", "") or ""),
        train_metric_key=str(getattr(args, "train_metric", "") or ""),
        maximize=False if bool(getattr(args, "minimize", False)) else None,
        search_space=_load_search_space(str(getattr(args, "search_space", "") or "")),
        top_fraction=top_fraction,
        min_top_trials=min_top_trials,
        max_train_val_gap=max_train_val_gap,
        boundary_tolerance=boundary_tolerance,
    )
    _print_payload(result, as_json=bool(getattr(args, "json", False)))
    return 0 if result.get("selected") is not None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lq deep-learning",
        description="Plan and validate the DeepLearning sidecar forecast pipeline.",
    )
    parser.add_argument(
        "--config", default="", help="Runtime config path; defaults to LQ_CONFIG_PATH/config.yaml."
    )
    sub = parser.add_subparsers(dest="subcommand")

    plan = sub.add_parser(
        "plan", help="Build the integrated pipeline manifest without running training."
    )
    plan.add_argument(
        "--write",
        nargs="?",
        const="__default__",
        default="",
        help="Write manifest JSON to this path or deep_learning.manifest_path.",
    )
    plan.add_argument("--json", action="store_true", help="Print manifest as JSON instead of YAML.")

    validate = sub.add_parser("validate-artifacts", help="Check saved forecast artifact coverage.")
    validate.add_argument(
        "--strict", action="store_true", help="Exit non-zero when coverage is incomplete."
    )
    validate.add_argument(
        "--json", action="store_true", help="Print validation as JSON instead of YAML."
    )

    strategy = sub.add_parser("strategy-config", help="Print a materializable strategy profile.")
    strategy.add_argument("--profile", default="ensemble", help="ensemble, all, or one model name.")
    strategy.add_argument(
        "--json", action="store_true", help="Print profile as JSON instead of YAML."
    )

    hpo_select = sub.add_parser(
        "hpo-select",
        help="Select stable top HPO params from JSON/CSV trial results.",
    )
    hpo_select.add_argument(
        "trials",
        help="JSON or CSV trial table. JSON may be a list or contain a trials/results key.",
    )
    hpo_select.add_argument(
        "--search-space", default="", help="Optional JSON mapping param names to low/high bounds."
    )
    hpo_select.add_argument(
        "--metric", default="", help="Validation metric key; auto-detected when omitted."
    )
    hpo_select.add_argument(
        "--train-metric", default="", help="Train metric key; auto-detected when omitted."
    )
    hpo_select.add_argument(
        "--minimize", action="store_true", help="Treat lower metric values as better."
    )
    hpo_select.add_argument(
        "--top-fraction", type=float, default=None, help="Validation top fraction to inspect."
    )
    hpo_select.add_argument(
        "--min-top-trials", type=int, default=None, help="Minimum top trials to inspect."
    )
    hpo_select.add_argument(
        "--max-train-val-gap", type=float, default=None, help="Maximum stable train/validation gap."
    )
    hpo_select.add_argument(
        "--boundary-tolerance", type=float, default=None, help="Search-boundary flag tolerance."
    )
    hpo_select.add_argument(
        "--json", action="store_true", help="Print selection as JSON instead of YAML."
    )

    args = parser.parse_args(argv)
    if args.subcommand == "plan":
        return cmd_plan(args)
    if args.subcommand == "validate-artifacts":
        return cmd_validate_artifacts(args)
    if args.subcommand == "strategy-config":
        return cmd_strategy_config(args)
    if args.subcommand == "hpo-select":
        return cmd_hpo_select(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
