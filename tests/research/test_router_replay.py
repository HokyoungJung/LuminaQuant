"""Source-first canonical Router v2 replay fixtures."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import inspect
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from lumina_quant.data.symbol_lifecycle import (
    build_fold_membership_manifest,
    build_symbol_lifecycle_registry,
)
from lumina_quant.research import router_replay as replay
from lumina_quant.strategies.registry import resolve_strategy_class


def _raw(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_raw(value)).hexdigest()


def _write(path: Path, value: object) -> str:
    path.write_bytes(_raw(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rebuild(paths: dict[str, Path], trusted: dict[str, str]) -> None:
    """Re-root producer-controlled manifest claims; never change out-of-band roots by accident."""
    manifest = json.loads(paths["manifest"].read_text())
    commit = json.loads(paths["commit"].read_text())
    commit["manifest_sha256"] = _write(paths["manifest"], manifest)
    trusted["commit"] = _write(paths["commit"], commit)


def _reroot_candidate_identity_layer(
    paths: dict[str, Path],
    trusted: dict[str, str],
    layer: str,
    candidate_ids: list[str],
) -> None:
    """Mutate one identity layer while preserving earlier independent bindings."""
    candidate_ids_sha256 = _sha(candidate_ids)
    if layer == "source":
        source = json.loads(paths["source"].read_text(encoding="utf-8"))
        source["candidate_ids"] = candidate_ids
        source["candidate_ids_sha256"] = candidate_ids_sha256
        source["candidate_order"] = candidate_ids
        trusted["source"] = _write(paths["source"], source)

        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["provenance"]["source_artifact_sha256"] = trusted["source"]
        _write(paths["manifest"], manifest)

        commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
        commit["source_artifact_sha256"] = trusted["source"]
        commit["manifest_sha256"] = _write(paths["manifest"], manifest)
        trusted["commit"] = _write(paths["commit"], commit)
    elif layer == "manifest":
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["candidate_ids"] = candidate_ids
        manifest["candidate_ids_sha256"] = candidate_ids_sha256

        commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
        commit["manifest_sha256"] = _write(paths["manifest"], manifest)
        trusted["commit"] = _write(paths["commit"], commit)
    elif layer == "commit":
        commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
        commit["candidate_ids"] = candidate_ids
        commit["candidate_ids_sha256"] = candidate_ids_sha256
        trusted["commit"] = _write(paths["commit"], commit)
    else:
        raise ValueError(f"unknown identity layer: {layer}")


def _reroot_manifest_selection(
    paths: dict[str, Path],
    trusted: dict[str, str],
    fold_index: int,
    selected_label: str | None,
) -> None:
    """Re-root an authenticated manifest after changing only its claimed selection."""
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    selection = manifest["folds"][fold_index]["selection"]
    selection["selected_label"] = selected_label
    selection["selection_inputs_sha256"] = _sha(
        {
            "branch": selection["branch"],
            "selected_label": selected_label,
            "decision_receipt_sha256s": selection["decision_receipt_sha256s"],
            "leaves": selection["leaves"],
        }
    )
    _write(paths["manifest"], manifest)
    _rebuild(paths, trusted)


CostRowFactory = Callable[[str, str, str, int, str, str], list[dict[str, Any]] | None]


def _bundle(
    tmp_path: Path,
    mode: str = "handler",
    mutation: str | None = None,
    cost_rows: CostRowFactory | None = None,
    *,
    all_folds_scaled: bool = False,
    traded_symbols: list[str] | None = None,
    strict_candidate_eligibility: tuple[bool, bool] | None = None,
) -> tuple[dict[str, object], dict[str, Path], dict[str, str]]:
    """Build authoritative source, artifacts, manifest, then its trusted commit receipt."""
    producer = tmp_path / "producer.py"
    producer.write_text("# frozen producer\n", encoding="utf-8")
    profile = Path(replay.__file__).parents[3] / "configs/profiles/backtest_cost_realistic.yaml"
    if mutation == "profile_unknown_key":
        profile = tmp_path / "profile.yaml"
        profile.write_bytes(
            (
                Path(replay.__file__).parents[3] / "configs/profiles/backtest_cost_realistic.yaml"
            ).read_bytes()
            + b"\nignored_router_key: true\n"
        )
    traded_symbols = traded_symbols or ["BTCUSDT"]
    symbols = (
        ["ETHUSDT", "BTCUSDT"]
        if mutation == "weighted_nonlexicographic_history"
        else traded_symbols
    )
    registry = build_symbol_lifecycle_registry(
        {
            "symbols": [
                {"symbol": symbol, "onboardDate": 0, "deliveryDate": None} for symbol in symbols
            ]
        },
        symbols,
        {"uri": "frozen://registry", "retrieved_at_ms": 0, "payload_sha256": "a" * 64},
    )
    lifecycle, membership = tmp_path / "lifecycle.json", tmp_path / "membership.json"
    _write(lifecycle, registry)
    members = build_fold_membership_manifest(
        registry,
        [
            {
                "fold_id": f"f{i}",
                "start_ms": 1735689600000 + i * 3 * 86400000,
                "end_ms": 1735776000000 + i * 3 * 86400000,
            }
            for i in range(5)
        ],
    )
    _write(membership, members)
    paths: dict[str, Path] = {}
    kinds: dict[str, str] = {}

    def artifact(kind: str, schema: str, row: dict[str, object]) -> str:
        row = {"schema": schema, **row}
        if mutation == "extra" and schema == "router_signal_receipt_v2":
            row["unexpected"] = True
        digest = _sha(row)
        path = tmp_path / f"{digest}.json"
        _write(path, row)
        paths[digest], kinds[digest] = path, kind
        return digest

    def grid(start: str, end: str) -> list[str]:
        cursor = datetime.fromisoformat(start[:-1] + "+00:00")
        finish = datetime.fromisoformat(end[:-1] + "+00:00")
        result: list[str] = []
        while cursor < finish:
            result.append(cursor.astimezone(UTC).isoformat().replace("+00:00", "Z"))
            cursor += timedelta(hours=1)
        return result

    registry_mode = mode == "registry"
    strategy = "RsiStrategy" if registry_mode else "trend_pullback_reclaim"
    handler = "registry_simulator" if registry_mode else replay.PROFILE_HANDLER
    evaluation = "registry_simulator" if registry_mode else "handler"
    if registry_mode:
        klass = resolve_strategy_class(strategy, strict=True)
        source_path = Path(inspect.getsourcefile(klass) or "")
        names = {
            "entrypoint": "lumina_quant.strategy_factory.research_runner._strict_registry_simulator_router",
            "bar_store": "lumina_quant.strategy_factory.research_runner._AlignedStrategyBarStore",
            "simulator": "lumina_quant.strategy_factory.research_runner._simulate_event_driven_strategy_exposures",
            "dispatch_validator": "lumina_quant.strategy_factory.strategy_signal_dispatch.StrategySignalDispatcher._validate_actual_engine_outputs",
            "strategy_class": f"{klass.__module__}.{klass.__qualname__}",
            "source_eligibility": replay.ROUTER_SOURCE_PREDICATE,
        }
    else:
        source_path = replay._object_source(handler)
        names = {
            "entrypoint": handler,
            "feature_cache": "scripts.research.run_alpha_zoo_69_asset_profile_optuna_hybrid_refit.FeatureCache",
            "signal": "scripts.research.run_alpha_zoo_69_asset_optuna_hybrid_refit._debounced_state_signal",
            "simulator": "scripts.research.run_alpha_zoo_69_asset_optuna_hybrid_refit.simulate_symbol",
            "candidate_builder": "scripts.research.run_alpha_zoo_69_asset_optuna_hybrid_refit._candidate_base",
            "finalizer": "scripts.research.run_alpha_zoo_69_asset_optuna_hybrid_refit.finalize_candidate",
            "source_eligibility": replay.ROUTER_SOURCE_PREDICATE,
        }
    components = [
        {
            "role": role,
            "qualified_name": name,
            "source_sha256": hashlib.sha256(replay._object_source(name).read_bytes()).hexdigest(),
        }
        for role, name in names.items()
    ]
    dep = artifact(
        "engine_dependency",
        "router_engine_dependency_v2",
        {
            "evaluation_mode": evaluation,
            "engine_handler": handler,
            "strategy_class": strategy,
            "components": components,
        },
    )
    params = {"family": "frozen", "timeframe": "1h"}
    leaf = {
        "leaf_id": "leaf",
        "profile_id": "balanced",
        "strategy_class": strategy,
        "engine_handler": handler,
        "params": params,
        "params_sha256": _sha(params),
        "traded_symbols": (
            ["ETHUSDT", "BTCUSDT"]
            if mutation == "weighted_nonlexicographic_history"
            else traded_symbols
        ),
        "dependency_symbols": (
            ["ETHUSDT", "BTCUSDT"]
            if mutation == "weighted_nonlexicographic_history"
            else traded_symbols
        ),
        "native_timeframe": "1h",
        "allocation_fraction_ppm": 1_000_000,
        "source_weight_ppm": (
            250_000 if mutation == "weighted_nonlexicographic_history" else 1_000_000
        ),
        "native_gross_ppm": 1_000_000,
        "evaluation_mode": evaluation,
        "engine_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "engine_dependency_receipt_sha256": dep,
    }
    leaf["source_row_sha256"] = _sha(leaf)
    history_leaves = [leaf]
    if mutation == "history_ineligible_symbol":
        history_leaf = deepcopy(leaf)
        history_leaf["traded_symbols"] = ["ETHUSDT"]
        history_leaf["dependency_symbols"] = ["ETHUSDT"]
        history_leaf["source_row_sha256"] = _sha(
            {key: value for key, value in history_leaf.items() if key != "source_row_sha256"}
        )
        history_leaves = [history_leaf]
    if mutation in {"history_duplicate_leaf_id", "weighted_nonlexicographic_history"}:
        secondary = deepcopy(leaf)
        secondary["leaf_id"] = (
            "leaf" if mutation == "history_duplicate_leaf_id" else "leaf-secondary"
        )
        secondary["profile_id"] = "balanced-secondary"
        secondary["source_weight_ppm"] = (
            750_000 if mutation == "weighted_nonlexicographic_history" else 1_000_000
        )
        secondary["engine_dependency_receipt_sha256"] = artifact(
            "engine_dependency",
            "router_engine_dependency_v2",
            {
                "evaluation_mode": evaluation,
                "engine_handler": handler,
                "strategy_class": strategy,
                "components": list(reversed(components)),
            },
        )
        secondary["source_row_sha256"] = _sha(
            {key: value for key, value in secondary.items() if key != "source_row_sha256"}
        )
        history_leaves.append(secondary)

    def fold(index: int, member: dict[str, object]) -> dict[str, object]:
        day = 1 + index * 3
        return {
            "fold_id": member["fold_id"],
            "locked_oos": {
                "start_utc": f"2025-01-{day:02d}T00:00:00Z",
                "end_utc": f"2025-01-{day + 1:02d}T00:00:00Z",
            },
            "input_cutoff_utc": (
                "2024-12-31T00:00:00Z" if index == 0 else f"2025-01-{day - 1:02d}T00:00:00Z"
            ),
            "decision_timestamp_utc": f"2025-01-{day:02d}T00:00:00Z",
            "train_window": {
                "start_utc": "2024-01-02T00:00:00Z",
                "end_utc": "2024-01-02T01:00:00Z",
            },
            "validation_window": {
                "start_utc": "2024-06-02T00:00:00Z",
                "end_utc": (
                    "2024-06-02T02:00:00Z"
                    if mutation in {"negative_scaled_short", "shared_initial_loss_rerooted"}
                    else "2024-06-02T01:00:00Z"
                ),
            },
            "membership_fold_sha256": _sha(member),
            "candidates": [],
            "strict_core": {
                "candidates": [],
                "leaves": [],
                "leaf_list_sha256": _sha([]),
                "shared_mdd_receipt_sha256": None,
            },
        }

    folds = [fold(i, member) for i, member in enumerate(members["folds"])]

    def source_candidate(label: str) -> dict[str, object]:
        return {
            "candidate_label": label,
            "family": "strict_efficiency",
            "return_count": 1,
            "row": {
                "profile_kind": "strategy",
                "uses_locked_oos_for_selection": False,
                "same_month_self_feeding": False,
                "current_fold_oos_used_for_weighting": False,
                "post_oos_research_variant": False,
                "requires_fresh_forward_shadow": False,
            },
        }

    def candidate(
        label: str,
        histories: list[str],
        *,
        good: bool = True,
        leaves: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        candidate_leaves = history_leaves if leaves is None else leaves
        return {
            "candidate_label": label,
            "source_candidate": source_candidate(label),
            "train_return_ppm": 100_000 if good else -30_000,
            "train_mdd_ppm": 100_000,
            "validation_return_ppm": (
                -10_000
                if mutation == "weighted_nonlexicographic_history"
                else 100_000
                if good
                else -10_000
            ),
            "validation_mdd_ppm": 100_000,
            "history_receipt_sha256s": histories,
            "leaves": [deepcopy(item) for item in candidate_leaves],
            "leaf_list_sha256": _sha(candidate_leaves),
        }

    for item in folds[:4]:
        item["candidates"] = [candidate("candidate", [])]

    def data_window(
        item: dict[str, object], role: str, receipt_symbols: list[str] | None = None
    ) -> tuple[str, str]:
        data = {
            "dataset_id": "frozen",
            "content_sha256": "b" * 64,
            "symbols": symbols if receipt_symbols is None else receipt_symbols,
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": item["locked_oos"]["end_utc"],
            "input_cutoff_utc": item["input_cutoff_utc"],
            "native_timeframe": (
                "4h"
                if mutation == "history_timeframe"
                and item["fold_id"] == "f0"
                and role == "locked_oos"
                else "1h"
            ),
            "source_artifact_sha256": "c" * 64,
            "tape_sha256": "d" * 64,
        }
        data["strict_receipt_sha256"] = artifact(
            "data_source_receipt", "router_data_source_receipt_v2", data
        )
        digest = artifact("data_receipt", "router_data_receipt_v2", data)
        if role == "locked_oos":
            start, end = item["locked_oos"]["start_utc"], item["locked_oos"]["end_utc"]
            grid_end = end
            if mutation == "window":
                end = start
        elif role == "train":
            start, end = "2024-01-02T00:00:00Z", "2024-01-02T01:00:00Z"
            grid_end = end
        else:
            start, end = (
                "2024-06-02T00:00:00Z",
                (
                    "2024-06-02T02:00:00Z"
                    if mutation in {"negative_scaled_short", "shared_initial_loss_rerooted"}
                    else "2024-06-02T01:00:00Z"
                ),
            )
            grid_end = end
        window_role = "history" if mutation == "role" and role == "locked_oos" else role
        output_timestamps = grid(start, grid_end)
        if mutation == "native_grid_cadence_drift" and role == "locked_oos":
            output_timestamps[2] = start.replace("00:00:00Z", "02:30:00Z")
        elif mutation == "native_grid_sparse" and role == "locked_oos":
            del output_timestamps[1]
        elif mutation == "native_grid_incomplete" and role == "locked_oos":
            output_timestamps.pop()
        if mutation == "same_key_omission" and item["fold_id"] == "f4" and role == "locked_oos":
            output_timestamps.append(start.replace("00:00:00Z", "12:00:00Z"))
        window = artifact(
            "window_receipt",
            "router_window_receipt_v2",
            {
                "data_receipt_sha256": digest,
                "fold_id": item["fold_id"],
                "role": window_role,
                "start_utc": start,
                "end_utc": end,
                "input_cutoff_utc": item["input_cutoff_utc"],
                "native_timeframe": (
                    "4h"
                    if mutation == "history_timeframe"
                    and item["fold_id"] == "f0"
                    and role == "locked_oos"
                    else "1h"
                ),
                "membership_fold_sha256": item["membership_fold_sha256"],
                "output_timestamps_utc": output_timestamps,
            },
        )
        return digest, window

    def execute(
        item: dict[str, object],
        data: str,
        window: str,
        scale: int,
        stamp: str,
        ret: int = 100_000,
        position_offset: int = 0,
        leaf_record: dict[str, object] = leaf,
    ) -> tuple[str, str, str]:
        common = {
            "fold_id": item["fold_id"],
            "locked_oos": item["locked_oos"],
            "membership_fold_sha256": item["membership_fold_sha256"],
            "leaf_id": leaf_record["leaf_id"],
            "source_row_sha256": leaf_record["source_row_sha256"],
            "params_sha256": leaf_record["params_sha256"],
            "dependency_symbols": leaf_record["dependency_symbols"],
            "evaluation_mode": evaluation,
            "engine_handler": handler,
            "strategy_class": strategy,
            "native_timeframe": "1h",
            "engine_dependency_receipt_sha256": leaf_record["engine_dependency_receipt_sha256"],
            "data_receipt_sha256": data,
            "window_receipt_sha256": window,
            "applied_scale_ppm": scale,
            "generic_fallback_proxy_count": False if mutation == "receipt_count_bool" else 0,
            "current_fold_oos_input_count": 0,
        }
        row_specs = [
            (timestamp, ret)
            for timestamp in json.loads(paths[window].read_text(encoding="utf-8"))[
                "output_timestamps_utc"
            ]
        ]
        if mutation == "execution_outside_window":
            row_specs = [(item["locked_oos"]["end_utc"], ret)]
        if mutation in {"negative_scaled_short", "shared_initial_loss_rerooted"} and stamp == (
            "2024-06-02T00:00:00Z"
        ):
            row_specs = [
                (
                    "2024-06-02T00:00:00Z",
                    -18_181 if mutation == "negative_scaled_short" else -200_000,
                ),
                ("2024-06-02T01:00:00Z", 1_000_000),
            ]
        execution_symbols = (
            ["ETHUSDT", "BTCUSDT"]
            if mutation == "weighted_nonlexicographic_history"
            else ["ETHUSDT"]
            if mutation == "wrong_symbol"
            else traded_symbols
        )
        signal_rows = [
            {
                "timestamp_utc": timestamp,
                "symbol": symbol,
                "signal_ppm": (
                    -999_999
                    if mutation == "negative_scaled_short"
                    else -1_000_000
                    if mutation == "negative_history"
                    else 1_000_000
                ),
            }
            for timestamp, _ in row_specs
            for symbol in execution_symbols
        ]
        if mutation == "weighted_nonlexicographic_history":
            signal_rows = [
                {"timestamp_utc": timestamp, "symbol": symbol, "signal_ppm": signal_ppm}
                for timestamp, _ in row_specs
                for symbol, signal_ppm in (("ETHUSDT", 1_000_000), ("BTCUSDT", -1_000_000))
            ]
        base_signal_rows = signal_rows
        if mutation == "missing_row":
            signal_rows = []
        elif mutation == "extra_row":
            signal_rows.append(dict(signal_rows[0]))
        signal = artifact(
            "signal_receipt",
            "router_signal_receipt_v2",
            common | {"signal_rows": signal_rows, "signal_rows_sha256": _sha(signal_rows)},
        )
        position_rows = [
            {
                "timestamp_utc": signal_row["timestamp_utc"],
                "symbol": signal_row["symbol"],
                "position_ppm": replay._round_fraction_half_up(
                    replay.Fraction(signal_row["signal_ppm"] * scale, 1_000_000)
                )
                + position_offset
                + (1 if mutation == "fallback_position" else 0),
            }
            for signal_row in (base_signal_rows if mutation == "missing_row" else signal_rows)
        ]
        position = artifact(
            "position_receipt",
            "router_position_receipt_v2",
            common
            | {
                "signal_receipt_sha256": signal,
                "position_rows": position_rows,
                "position_rows_sha256": _sha(position_rows),
            },
        )
        weighted_returns = {"ETHUSDT": 80, "BTCUSDT": -20}
        execution_rows = [
            {
                "timestamp_utc": signal_row["timestamp_utc"],
                "symbol": signal_row["symbol"],
                "base_return_ppm": (
                    weighted_returns[signal_row["symbol"]]
                    if mutation == "weighted_nonlexicographic_history"
                    else dict(row_specs)[signal_row["timestamp_utc"]]
                ),
                "return_ppm": replay._round_fraction_half_up(
                    replay.Fraction(
                        (
                            weighted_returns[signal_row["symbol"]]
                            if mutation == "weighted_nonlexicographic_history"
                            else dict(row_specs)[signal_row["timestamp_utc"]]
                        )
                        * scale,
                        1_000_000,
                    )
                )
                + (1 if mutation == "fallback_return" else 0),
            }
            for signal_row in (base_signal_rows if mutation == "missing_row" else signal_rows)
        ]
        event_rows = [
            {
                "timestamp_utc": timestamp,
                "event_index": 0,
                "event_type": "fill",
                "event_sha256": "e" * 64,
            }
            for timestamp, _ in row_specs
        ]
        engine = artifact(
            "engine_receipt",
            "router_engine_receipt_v2",
            common
            | {
                "signal_receipt_sha256": signal,
                "position_receipt_sha256": position,
                "execution_rows": execution_rows,
                "execution_rows_sha256": _sha(execution_rows),
                "event_rows": event_rows,
                "event_rows_sha256": _sha(event_rows),
            },
        )
        return signal, position, engine

    def strict_core(target: dict[str, object]) -> dict[str, object]:
        flags = {
            "post_oos_augment": False,
            "generic_fallback_proxy": False,
            "current_fold_oos_input": False,
            "recomputed_from_json": False,
        }
        strict_leaves = [deepcopy(leaf)] if mode == "scaled" else []
        balanced = {
            "candidate_label": replay.BALANCED_LABEL,
            "source_kind": "lagged_shadow_leaf",
            "source_candidate": source_candidate(replay.BALANCED_LABEL),
            "source_eligibility": flags,
            "train_return_ppm": 100_000,
            "validation_return_ppm": 100_000 if mode == "scaled" else 10_000,
            "validation_mdd_ppm": 120_000 if mode == "scaled" else 200_000,
            "leaves": strict_leaves,
            "leaf_list_sha256": _sha(strict_leaves),
        }
        growth = {
            "candidate_label": replay.GROWTH_LABEL,
            "source_kind": "lagged_shadow_leaf",
            "source_candidate": source_candidate(replay.GROWTH_LABEL),
            "source_eligibility": flags,
            "train_return_ppm": 100_000,
            "validation_return_ppm": 120_000 if mode == "scaled" else 10_000,
            "validation_mdd_ppm": 100_000,
            "leaves": strict_leaves,
            "leaf_list_sha256": _sha(strict_leaves),
        }
        if strict_candidate_eligibility is not None:
            for candidate, eligible in zip(
                (balanced, growth), strict_candidate_eligibility, strict=True
            ):
                if not eligible:
                    candidate["train_return_ppm"] = -20_001
        strict: dict[str, object] = {
            "candidates": [balanced, growth],
            "leaves": strict_leaves,
            "leaf_list_sha256": _sha(strict_leaves),
            "shared_mdd_receipt_sha256": None,
        }
        if mode != "scaled" or strict_candidate_eligibility == (False, False):
            return strict
        data, train = data_window(target, "train")
        validation = artifact(
            "window_receipt",
            "router_window_receipt_v2",
            {
                "data_receipt_sha256": data,
                "fold_id": target["fold_id"],
                "role": "validation",
                "start_utc": (
                    "2024-01-02T00:30:00Z"
                    if mutation == "overlapping_windows"
                    else "2024-06-02T00:00:00Z"
                ),
                "end_utc": (
                    "2024-06-02T02:00:00Z"
                    if mutation in {"negative_scaled_short", "shared_initial_loss_rerooted"}
                    else "2024-06-02T01:00:00Z"
                ),
                "input_cutoff_utc": target["input_cutoff_utc"],
                "native_timeframe": "1h",
                "membership_fold_sha256": target["membership_fold_sha256"],
                "output_timestamps_utc": (
                    ["2024-06-02T00:00:00Z", "2024-06-02T01:00:00Z"]
                    if mutation in {"negative_scaled_short", "shared_initial_loss_rerooted"}
                    else ["2024-01-02T00:30:00Z"]
                    if mutation == "overlapping_windows"
                    else ["2024-06-02T00:00:00Z"]
                ),
            },
        )
        _, _, train_engine = execute(target, data, train, 1_000_000, "2024-01-02T00:00:00Z")
        _, _, validation_engine = execute(
            target, data, validation, 1_000_000, "2024-06-02T00:00:00Z"
        )
        train_rows = replay._aggregate_candidate_return_rows(
            [json.loads(paths[train_engine].read_text(encoding="utf-8"))], strict_leaves
        )
        validation_rows = replay._aggregate_candidate_return_rows(
            [json.loads(paths[validation_engine].read_text(encoding="utf-8"))], strict_leaves
        )
        if mutation == "shared":
            train_rows = [{"timestamp_utc": "2024-01-02T00:00:00Z", "return_ppm": 99_000}]
        strict["shared_mdd_receipt_sha256"] = artifact(
            "shared_mdd",
            "router_shared_mdd_receipt_v2",
            {
                "fold_id": target["fold_id"],
                "membership_fold_sha256": target["membership_fold_sha256"],
                "leaf_list_sha256": _sha(strict_leaves),
                "candidate_label": (
                    replay.BALANCED_LABEL
                    if strict_candidate_eligibility is not None and strict_candidate_eligibility[0]
                    else replay.GROWTH_LABEL
                ),
                "measurement_end_utc": target["input_cutoff_utc"],
                "input_cutoff_utc": target["input_cutoff_utc"],
                "data_receipt_sha256": data,
                "train_window_receipt_sha256": train,
                "validation_window_receipt_sha256": validation,
                "engine_dependency_receipt_sha256s": [dep],
                "train_engine_receipt_sha256s": [train_engine],
                "validation_engine_receipt_sha256s": [validation_engine],
                "native_timeframe": "1h",
                "train_return_rows": train_rows,
                "validation_return_rows": validation_rows,
                "train_return_rows_sha256": _sha(train_rows),
                "validation_return_rows_sha256": _sha(validation_rows),
            },
        )
        return strict

    if mode == "scaled" and all_folds_scaled:
        for target in folds:
            target["strict_core"] = strict_core(target)

    histories: list[str] = []
    for i, prior in enumerate(folds[:4]):
        prior["candidates"] = [candidate("candidate", list(histories))]
        history_symbols = list(
            dict.fromkeys(
                symbol
                for history_leaf in history_leaves
                for symbol in history_leaf["dependency_symbols"]
            )
        )
        data, window = data_window(prior, "locked_oos", history_symbols)
        baseline_return = -100_000 if mutation == "negative_history" else 100_000
        engines = [
            execute(
                prior,
                data,
                window,
                1_000_000,
                prior["locked_oos"]["start_utc"],
                ret=baseline_return,
                leaf_record=leaf_record,
            )[2]
            for leaf_record in history_leaves
        ]
        rows = replay._aggregate_candidate_return_rows(
            [json.loads(paths[engine].read_text(encoding="utf-8")) for engine in engines],
            prior["candidates"][0]["leaves"],
        )
        history_return = replay._round_fraction_half_up(
            replay._period_metrics([row["return_ppm"] for row in rows], 1_000_000)[0] * 1_000_000
        )
        weights = [item["source_weight_ppm"] for item in history_leaves]
        if mutation == "history_weights_bool" and i == 0:
            weights[0] = False
        elif mutation == "history_weights_float" and i == 0:
            weights[0] = 1_000_000.0
        aggregation = artifact(
            "candidate_aggregation",
            "router_candidate_aggregation_receipt_v2",
            {
                "engine_receipt_sha256s": engines,
                "weights_ppm": weights,
                "candidate_return_rows_sha256": _sha(rows),
            },
        )
        histories.append(
            artifact(
                "history_receipt",
                "router_history_receipt_v2",
                {
                    "fold_id": prior["fold_id"],
                    "candidate_label": "candidate",
                    "locked_oos": prior["locked_oos"],
                    "input_cutoff_utc": prior["input_cutoff_utc"],
                    "completed_at_utc": (
                        prior["locked_oos"]["start_utc"]
                        if mutation == "history_completion" and i == 0
                        else "2025-12-31T00:00:00Z"
                        if mutation == "history_future" and i == 0
                        else prior["locked_oos"]["end_utc"]
                    ),
                    "membership_fold_sha256": prior["membership_fold_sha256"],
                    "prior_source_artifact_sha256": (
                        "0" * 64 if mutation == "history" and i == 0 else _sha(prior)
                    ),
                    "leaf_list_sha256": prior["candidates"][0]["leaf_list_sha256"],
                    "data_receipt_sha256": data,
                    "window_receipt_sha256": window,
                    "engine_receipt_sha256s": engines,
                    "candidate_aggregation_receipt_sha256": aggregation,
                    "candidate_return_rows": rows,
                    "candidate_return_rows_sha256": _sha(rows),
                    "return_ppm": history_return,
                },
            )
        )
    final = folds[4]
    if mode in {"handler", "registry"}:
        final["candidates"] = [
            candidate(
                "candidate",
                histories,
                leaves=[leaf] if mutation == "history_ineligible_symbol" else None,
            )
        ]
    else:
        final["candidates"] = [candidate("candidate", histories, good=False)]
        if not (mode == "scaled" and all_folds_scaled):
            final["strict_core"] = strict_core(final)
    source = {
        "schema": replay.SOURCE_SCHEMA,
        "candidate_ids": list(replay.CANDIDATE_IDS),
        "candidate_ids_sha256": replay.CANDIDATE_IDS_SHA256,
        "controls": {
            "post_oos_augment": mutation == "source_post_oos_augment_rerooted",
            "new_grid_search": mutation == "source_new_grid_search_rerooted",
            "recompute_from_json": mutation == "source_recompute_from_json_rerooted",
            "post_oos_research_variant": True,
        },
        "frozen_at_utc": "2025-01-14T00:00:00Z",
        "policy": {
            "min_history": 4,
            "avg_window": 1,
            "min_train_return_ppm": -20_000,
            "max_train_mdd_ppm": 500_000,
            "min_validation_return_ppm": 0,
            "max_validation_mdd_ppm": 250_000,
            "validation_weight_ppm": 250_000,
            "tie_break": "combined,lagged,validation_score,validation_return,source_order",
        },
        "candidate_order": list(replay.CANDIDATE_IDS),
        "folds": folds,
    }
    source_path = tmp_path / "source.json"
    if mutation == "source_eligibility":
        source["folds"][4]["candidates"][0]["source_candidate"]["family"] = "excluded_family"
    elif mutation == "policy_bool":
        source["policy"]["min_history"] = False
    elif mutation == "source_window_mismatch":
        source["folds"][4]["validation_window"]["start_utc"] = (
            "2024-01-02T00:30:00Z" if mode == "handler" else "2024-06-03T00:00:00Z"
        )
    elif mutation == "overlapping_windows" and mode != "scaled":
        source["folds"][4]["validation_window"]["start_utc"] = "2024-01-02T00:30:00Z"
    source_hash = _write(source_path, source)

    manifest_folds = []
    for i, item in enumerate(folds):
        if mode == "scaled":
            branch, label, leaves, strict = replay._strict_core(item)
        else:
            branch = (
                "strict_core_cash"
                if (i < 4 and not all_folds_scaled)
                or mode == "cash"
                or mutation == "weighted_nonlexicographic_history"
                else "pre_registered_lagged_plus_validation_leaf"
            )
            label = "candidate" if branch.startswith("pre_") else None
            leaves = [deepcopy(leaf)] if branch != "strict_core_cash" else []
            strict = None
        shared = strict["shared_mdd_receipt_sha256"] if strict is not None else None
        history = item["candidates"][0]["history_receipt_sha256s"]
        selection = {
            "branch": branch,
            "selected_label": label,
            "source_fold_sha256": _sha(item),
            "selection_inputs_sha256": _sha(
                {
                    "branch": branch,
                    "selected_label": label,
                    "decision_receipt_sha256s": history,
                    "leaves": leaves,
                }
            ),
            "current_fold_oos_input_count": (
                1 if mutation == "selection_current_fold_oos_input_count_rerooted" and i == 4 else 0
            ),
            "decision_receipt_sha256s": history,
            "fallback_mdd_receipt_sha256": shared,
            "leaves": leaves,
            "leaf_list_sha256": _sha(leaves),
        }
        variants = []
        for variant_index, (variant, target, cap) in enumerate(
            zip(
                replay.CANDIDATE_IDS,
                (300_000, 200_000),
                (3_000_000, 2_000_000),
                strict=True,
            )
        ):
            scale = (
                0
                if branch == "strict_core_cash"
                else (
                    replay._fallback_scale(
                        {
                            "train_returns_ppm": [100_000],
                            "validation_returns_ppm": [-18_181, 1_000_000],
                        },
                        replay.Fraction(target, 1_000_000),
                        replay.Fraction(cap, 1_000_000),
                    )
                    if mutation == "negative_scaled_short"
                    else cap
                )
                if branch == "strict_core_scaled"
                else 1_000_000
            )
            if mutation == "scale" and i == 4 and variant_index == 0 and scale:
                scale += 1
            rows: list[object] = []
            effective: list[object] = []
            if leaves:
                data, window = data_window(item, "locked_oos")
                signal, position, engine = execute(
                    item,
                    data,
                    window,
                    scale,
                    item["locked_oos"]["start_utc"],
                    ret=(
                        -999_999
                        if mutation == "negative_scaled_short"
                        else 100_001
                        if mutation == "fallback_base_return" and variant_index == 1
                        else 100_000
                    ),
                    position_offset=(
                        1
                        if mutation == "position_engine"
                        and branch == "pre_registered_lagged_plus_validation_leaf"
                        and variant_index == 1
                        else 0
                    ),
                )
                tape_types = (
                    ("cost_signal_position_tape", "router_cost_signal_position_tape_v1"),
                    ("cost_order_tape", "router_cost_order_tape_v1"),
                    ("cost_fill_tape", "router_cost_fill_tape_v1"),
                    ("cost_event_tape", "router_cost_event_tape_v1"),
                )
                base_tape_commitments: dict[str, str] = {}
                tapes = []
                for bps in (10, 15, 20, 30):
                    hashes = []
                    for kind, schema in tape_types:
                        provided_rows = (
                            cost_rows(item["fold_id"], variant, "leaf", bps, kind, engine)
                            if callable(cost_rows)
                            else None
                        )
                        if provided_rows is None:
                            sequence = [f"{variant}:{kind}"]
                            downstream_rows = [
                                {
                                    "sequence_id": sequence[0],
                                    "cost_bps": bps,
                                    "tape_kind": kind,
                                    "notional_ppm": 1_000_000,
                                }
                            ]
                        else:
                            if not isinstance(provided_rows, list) or not provided_rows:
                                raise ValueError("cost row factory returned no rows")
                            downstream_rows = provided_rows
                            sequence = [str(row["sequence_id"]) for row in downstream_rows]
                        if kind not in base_tape_commitments:
                            base_tape_commitments[kind] = artifact(
                                "cost_base_tape_projection",
                                "router_cost_base_tape_projection_v1",
                                {
                                    "fold_id": item["fold_id"],
                                    "variant_id": variant,
                                    "leaf_id": "leaf",
                                    "engine_receipt_sha256": engine,
                                    "tape_kind": kind,
                                    "projection": sequence,
                                    "projection_sha256": _sha(sequence),
                                },
                            )
                        base_tape_commitment = base_tape_commitments[kind]
                        if (
                            mutation == "cost_base_commitment_drift"
                            and i == 4
                            and variant_index == 0
                            and bps == 10
                            and kind == "cost_order_tape"
                        ):
                            downstream_rows = deepcopy(downstream_rows)
                            downstream_rows[0]["sequence_id"] = f"{sequence[0]}:drift"
                            sequence = [str(row["sequence_id"]) for row in downstream_rows]
                            base_tape_commitment = artifact(
                                "cost_base_tape_projection",
                                "router_cost_base_tape_projection_v1",
                                {
                                    "fold_id": item["fold_id"],
                                    "variant_id": variant,
                                    "leaf_id": "leaf",
                                    "engine_receipt_sha256": engine,
                                    "tape_kind": kind,
                                    "projection": sequence,
                                    "projection_sha256": _sha(sequence),
                                },
                            )
                        tape_rows = [
                            {
                                "sequence_id": sequence_id,
                                "fold_id": item["fold_id"],
                                "variant_id": variant,
                                "leaf_id": "leaf",
                                "engine_receipt_sha256": engine,
                                "row_sha256": "a" * 64
                                if mutation == "duplicate_row_digest"
                                else _sha(row),
                            }
                            for sequence_id, row in zip(sequence, downstream_rows, strict=True)
                        ]
                        hashes.append(
                            artifact(
                                kind,
                                schema,
                                {
                                    "cost_cell": f"{bps}bps",
                                    "cost_bps": bps,
                                    "fold_id": item["fold_id"],
                                    "variant_id": variant,
                                    "leaf_id": "leaf",
                                    "engine_receipt_sha256": engine,
                                    "sequence": sequence,
                                    "sequence_sha256": _sha(sequence),
                                    "rows": tape_rows,
                                    "rows_sha256": _sha(tape_rows),
                                    "base_tape_projection_sha256": base_tape_commitment,
                                },
                            )
                        )
                    reported_bps = (
                        10.0
                        if mutation == "cost_bps_float"
                        and i == 4
                        and variant_index == 0
                        and bps == 10
                        else 15
                        if mutation == "cost" and i == 4 and variant_index == 0 and bps == 10
                        else bps
                    )
                    tapes.append(
                        {
                            "cost_bps": reported_bps,
                            "signal_position_sha256": hashes[0],
                            "order_tape_sha256": hashes[1],
                            "fill_tape_sha256": hashes[2],
                            "event_tape_sha256": hashes[3],
                        }
                    )
                cost = artifact(
                    "cost_tape_receipt",
                    "router_cost_tape_receipt_v1",
                    {
                        "fold_id": item["fold_id"],
                        "variant_id": variant,
                        "selected_label": label,
                        "leaf_id": "leaf",
                        "source_row_sha256": leaf["source_row_sha256"],
                        "params_sha256": leaf["params_sha256"],
                        "engine_receipt_sha256": engine,
                        "signal_receipt_sha256": signal,
                        "position_receipt_sha256": position,
                        "tapes": tapes,
                    },
                )
                rows = [
                    {
                        "leaf_id": "leaf",
                        "evaluation_mode": evaluation,
                        "engine_source_sha256": leaf["engine_source_sha256"],
                        "engine_dependency_receipt_sha256": dep,
                        "data_receipt_sha256": data,
                        "window_receipt_sha256": window,
                        "signal_receipt_sha256": signal,
                        "position_receipt_sha256": position,
                        "engine_receipt_sha256": engine,
                        "cost_tape_receipt_sha256": cost,
                        "generic_fallback_proxy_count": 0,
                        "current_fold_oos_input_count": (
                            1
                            if mutation == "execution_current_fold_oos_input_count_rerooted"
                            and i == 4
                            and variant_index == 0
                            else 0
                        ),
                    }
                ]
                effective = [
                    {"leaf_id": "leaf", "effective_weight_ppm": scale, "effective_gross_ppm": scale}
                ]
                if mutation == "effective_bool" and variant_index == 0:
                    effective[0]["effective_weight_ppm"] = False
            variants.append(
                {
                    "variant_id": variant,
                    "selected_label": label,
                    "base_leaf_list_sha256": _sha(leaves),
                    "policy": {"fallback_mdd_ppm": target, "fallback_cap_ppm": cap},
                    "applied_scale_ppm": scale,
                    "leaves": effective,
                    "execution_receipts": rows,
                }
            )
        manifest_folds.append(
            {
                key: item[key]
                for key in (
                    "fold_id",
                    "locked_oos",
                    "input_cutoff_utc",
                    "decision_timestamp_utc",
                    "membership_fold_sha256",
                )
            }
            | {"selection": selection, "variants": variants}
        )
    manifest = {
        "schema": replay.SCHEMA,
        "candidate_ids": list(replay.CANDIDATE_IDS),
        "candidate_ids_sha256": replay.CANDIDATE_IDS_SHA256,
        "controls": {
            "new_grid_search": False,
            "recompute_from_json": False,
            "post_oos_augment": False,
            "real_money_enabled": False,
            "orders_submitted": 1 if mutation == "manifest_orders_submitted_rerooted" else 0,
            "capital_allocated": 1 if mutation == "manifest_capital_allocated_rerooted" else 0,
        },
        "provenance": {},
        "folds": manifest_folds,
    }
    if mutation == "bool":
        manifest["controls"]["real_money_enabled"] = 0
    elif mutation == "overflow":
        manifest["controls"]["orders_submitted"] = int("9" * 400)
    elif mutation == "manifest_post_oos_augment_rerooted":
        manifest["controls"]["post_oos_augment"] = True
    elif mutation == "manifest_real_money_enabled_rerooted":
        manifest["controls"]["real_money_enabled"] = True
    manifest_path = tmp_path / "manifest.json"
    artifact_index = [{"kind": kinds[digest], "sha256": digest} for digest in sorted(kinds)]
    if mutation == "kind":
        artifact_index[0]["kind"] = "wrong_kind"
    commit = {
        "schema": replay.COMMIT_SCHEMA,
        "repository_commit": "0" * 40,
        "candidate_ids": list(replay.CANDIDATE_IDS),
        "candidate_ids_sha256": replay.CANDIDATE_IDS_SHA256,
        "manifest_sha256": "",
        "source_artifact_sha256": source_hash,
        "producer_source_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
        "verifier_source_sha256": hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest(),
        "lifecycle_registry_sha256": hashlib.sha256(lifecycle.read_bytes()).hexdigest(),
        "membership_manifest_sha256": hashlib.sha256(membership.read_bytes()).hexdigest(),
        "combined_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "runner_source_sha256": hashlib.sha256(
            replay._object_source(replay.ROUTER_SOURCE_PREDICATE).read_bytes()
        ).hexdigest(),
        "research_runner_source_sha256": hashlib.sha256(
            Path(
                inspect.getsourcefile(
                    __import__("lumina_quant.strategy_factory.research_runner", fromlist=["x"])
                )
                or ""
            ).read_bytes()
        ).hexdigest(),
        "artifact_index": artifact_index,
    }
    manifest["provenance"] = {
        name: commit[name]
        for name in (
            "repository_commit",
            "producer_source_sha256",
            "verifier_source_sha256",
            "source_artifact_sha256",
            "lifecycle_registry_sha256",
            "membership_manifest_sha256",
            "combined_profile_sha256",
        )
    }
    if mutation == "runner_source_drift":
        commit["runner_source_sha256"] = "f" * 64
    commit["manifest_sha256"] = _write(manifest_path, manifest)
    commit_path = tmp_path / "commit.json"
    commit_hash = _write(commit_path, commit)
    paths.update(
        {
            "manifest": manifest_path,
            "source": source_path,
            "commit": commit_path,
            "lifecycle": lifecycle,
            "membership": membership,
            "profile": profile,
            "producer": producer,
        }
    )
    return manifest, paths, {"source": source_hash, "commit": commit_hash}


def _report(paths: dict[str, Path], trusted: dict[str, str]):
    return replay.evaluate_router_replay(
        paths["manifest"],
        source_artifact_path=paths["source"],
        lifecycle_registry_path=paths["lifecycle"],
        membership_manifest_path=paths["membership"],
        combined_profile_path=paths["profile"],
        producer_source_path=paths["producer"],
        commit_receipt_path=paths["commit"],
        trusted_source_artifact_sha256=trusted["source"],
        trusted_commit_receipt_sha256=trusted["commit"],
        artifact_paths={key: value for key, value in paths.items() if len(key) == 64},
    )


@pytest.mark.parametrize(
    ("mode", "branch"),
    [
        ("handler", "pre_registered_lagged_plus_validation_leaf"),
        ("registry", "pre_registered_lagged_plus_validation_leaf"),
        ("scaled", "strict_core_scaled"),
        ("cash", "strict_core_cash"),
    ],
)
def test_router_v2_source_first_positive_paths(tmp_path: Path, mode: str, branch: str) -> None:
    manifest, paths, trusted = _bundle(tmp_path, mode)
    assert manifest["folds"][4]["selection"]["branch"] == branch
    assert _report(paths, trusted).status == "PASS"
    assert _report(paths, trusted).fold_count == 5


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("selection_current_fold_oos_input_count_rerooted", "selection replay drift"),
        ("execution_current_fold_oos_input_count_rerooted", "execution zero-control drift"),
        ("source_post_oos_augment_rerooted", "source controls are invalid"),
        ("source_new_grid_search_rerooted", "source controls are invalid"),
        ("source_recompute_from_json_rerooted", "source controls are invalid"),
        ("manifest_orders_submitted_rerooted", "manifest controls are invalid"),
        ("manifest_capital_allocated_rerooted", "manifest controls are invalid"),
        ("manifest_post_oos_augment_rerooted", "manifest controls are invalid"),
        ("manifest_real_money_enabled_rerooted", "manifest controls are invalid"),
    ],
)
def test_router_v2_rerooted_nonzero_locked_oos_report_only_controls_stop(
    tmp_path: Path, mutation: str, expected_reason: str
) -> None:
    manifest, paths, trusted = _bundle(tmp_path, "handler", mutation)
    _rebuild(paths, trusted)

    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    selection = manifest["folds"][4]["selection"]
    execution = manifest["folds"][4]["variants"][0]["execution_receipts"][0]
    assert type(selection["current_fold_oos_input_count"]) is int
    assert type(execution["current_fold_oos_input_count"]) is int
    assert type(source["controls"]["post_oos_augment"]) is bool
    assert type(source["controls"]["new_grid_search"]) is bool
    assert type(source["controls"]["recompute_from_json"]) is bool
    assert type(manifest["controls"]["orders_submitted"]) is int
    assert type(manifest["controls"]["capital_allocated"]) is int
    assert type(manifest["controls"]["post_oos_augment"]) is bool
    assert type(manifest["controls"]["real_money_enabled"]) is bool
    assert selection["current_fold_oos_input_count"] == int(
        mutation == "selection_current_fold_oos_input_count_rerooted"
    )
    assert execution["current_fold_oos_input_count"] == int(
        mutation == "execution_current_fold_oos_input_count_rerooted"
    )
    assert source["controls"]["post_oos_augment"] is (
        mutation == "source_post_oos_augment_rerooted"
    )
    assert source["controls"]["new_grid_search"] is (mutation == "source_new_grid_search_rerooted")
    assert source["controls"]["recompute_from_json"] is (
        mutation == "source_recompute_from_json_rerooted"
    )
    assert manifest["controls"]["orders_submitted"] == int(
        mutation == "manifest_orders_submitted_rerooted"
    )
    assert manifest["controls"]["capital_allocated"] == int(
        mutation == "manifest_capital_allocated_rerooted"
    )
    assert manifest["controls"]["post_oos_augment"] is (
        mutation == "manifest_post_oos_augment_rerooted"
    )
    assert manifest["controls"]["real_money_enabled"] is (
        mutation == "manifest_real_money_enabled_rerooted"
    )

    report = _report(paths, trusted)
    assert report.status == "STOP"
    assert report.fold_count == 0
    assert report.reasons == (expected_reason,)


@pytest.mark.parametrize(
    ("layer", "expected_reason"),
    [
        ("source", "source candidate identity drift"),
        ("manifest", "manifest candidate identity drift"),
        ("commit", "commit candidate identity drift"),
    ],
)
@pytest.mark.parametrize(
    "expected_ids",
    [
        pytest.param(lambda r1, r2: [r1], id="missing"),
        pytest.param(lambda r1, r2: [r1, r2, f"{r2}:unapproved"], id="extra"),
        pytest.param(lambda r1, r2: [r2, r1], id="reordered"),
        pytest.param(lambda r1, r2: [r1, f"{r2}:unapproved"], id="substituted"),
    ],
)
def test_router_v2_public_rerooted_candidate_identity_layer_boundaries_stop(
    tmp_path: Path,
    layer: str,
    expected_reason: str,
    expected_ids: Callable[[str, str], list[str]],
) -> None:
    _, paths, trusted = _bundle(tmp_path, "handler")
    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    r1, r2 = source["candidate_ids"]

    assert tuple(source["candidate_ids"]) == replay.CANDIDATE_IDS
    assert source["candidate_order"] == list(replay.CANDIDATE_IDS)
    assert len(source["candidate_ids"]) == 2
    assert source["controls"] == {
        "post_oos_augment": False,
        "new_grid_search": False,
        "recompute_from_json": False,
        "post_oos_research_variant": True,
    }
    assert _report(paths, trusted).status == "PASS"

    candidate_ids = expected_ids(r1, r2)
    _reroot_candidate_identity_layer(paths, trusted, layer, candidate_ids)

    rerooted_source = json.loads(paths["source"].read_text(encoding="utf-8"))
    rerooted_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    rerooted_commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
    layers = {
        "source": rerooted_source,
        "manifest": rerooted_manifest,
        "commit": rerooted_commit,
    }
    for name, rerooted in layers.items():
        if name == layer:
            assert rerooted["candidate_ids"] == candidate_ids
            assert rerooted["candidate_ids_sha256"] == _sha(candidate_ids)
        else:
            assert rerooted["candidate_ids"] == list(replay.CANDIDATE_IDS)
            assert rerooted["candidate_ids_sha256"] == replay.CANDIDATE_IDS_SHA256
    assert rerooted_source["candidate_order"] == (
        candidate_ids if layer == "source" else list(replay.CANDIDATE_IDS)
    )
    assert rerooted_source["controls"] == source["controls"]
    source_sha256 = hashlib.sha256(paths["source"].read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()
    commit_sha256 = hashlib.sha256(paths["commit"].read_bytes()).hexdigest()
    assert trusted["source"] == source_sha256
    assert trusted["commit"] == commit_sha256
    assert rerooted_manifest["provenance"]["source_artifact_sha256"] == source_sha256
    assert rerooted_commit["source_artifact_sha256"] == source_sha256
    assert rerooted_commit["manifest_sha256"] == manifest_sha256
    assert rerooted_manifest["controls"] == {
        "new_grid_search": False,
        "recompute_from_json": False,
        "post_oos_augment": False,
        "real_money_enabled": False,
        "orders_submitted": 0,
        "capital_allocated": 0,
    }

    report = _report(paths, trusted)
    assert report.status == "STOP"
    assert report.fold_count == 0
    assert report.reasons == (expected_reason,)


def test_router_v2_native_grid_rejects_cadence_drift_sparse_and_incomplete_sequences() -> None:
    start = replay._timestamp("2025-01-01T00:00:00Z", "test start")
    end = replay._timestamp("2025-01-01T03:00:00Z", "test end")

    replay._window_grid(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00Z",
            "2025-01-01T02:00:00Z",
        ],
        start,
        end,
        "1h",
    )
    for timestamps in (
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00Z",
            "2025-01-01T02:30:00Z",
        ],
        ["2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z"],
        ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
    ):
        with pytest.raises(ValueError, match="window native-grid drift"):
            replay._window_grid(timestamps, start, end, "1h")


def test_router_v2_public_manifest_lone_surrogate_stops(tmp_path: Path) -> None:
    _, paths, trusted = _bundle(tmp_path, "handler")
    paths["manifest"].write_bytes(b'{"malformed":"\\ud800"}')

    report = _report(paths, trusted)

    assert report.status == "STOP"
    assert report.fold_count == 0
    assert report.reasons == ("JSON contains invalid Unicode",)


@pytest.mark.parametrize(
    "mutation",
    ("native_grid_cadence_drift", "native_grid_sparse", "native_grid_incomplete"),
)
def test_router_v2_public_native_grid_exploits_stop_exactly(tmp_path: Path, mutation: str) -> None:
    _, paths, trusted = _bundle(tmp_path, "handler", mutation)

    report = _report(paths, trusted)

    assert report.status == "STOP"
    assert report.fold_count == 0
    assert report.reasons == ("window native-grid drift",)


def test_router_v2_frozen_no_current_oos_no_augment_no_order_no_capital_lineage_passes(
    tmp_path: Path,
) -> None:
    manifest, paths, trusted = _bundle(tmp_path, "handler")
    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    selection = manifest["folds"][4]["selection"]
    execution = manifest["folds"][4]["variants"][0]["execution_receipts"][0]

    assert selection["current_fold_oos_input_count"] == 0
    assert execution["current_fold_oos_input_count"] == 0
    assert source["controls"]["post_oos_augment"] is False
    assert manifest["controls"]["orders_submitted"] == 0
    assert manifest["controls"]["capital_allocated"] == 0
    assert _report(paths, trusted).status == "PASS"


@pytest.mark.parametrize(
    "mutation, mode, reason",
    [
        ("trusted", "handler", "trusted root mismatch"),
        ("kind", "handler", "artifact kind is missing or wrong"),
        ("history", "handler", "history provenance chronology drift"),
        ("history_completion", "handler", "history provenance chronology drift"),
        ("history_future", "handler", "history provenance chronology drift"),
        ("history_timeframe", "handler", "data timeframe drift"),
        ("history_weights_float", "handler", "history candidate aggregation binding drift"),
        ("history_weights_bool", "handler", "history candidate aggregation binding drift"),
        ("history_duplicate_leaf_id", "handler", "history leaf identity/timeframe drift"),
        ("history_ineligible_symbol", "handler", "leaf dependencies are invalid"),
        (
            "source_eligibility",
            "handler",
            "source candidate is ineligible under frozen source predicate",
        ),
        ("wrong_symbol", "handler", "signal row coverage/order drift"),
        ("missing_row", "handler", "signal rows are invalid"),
        ("same_key_omission", "handler", "window output sequence drift"),
        ("extra_row", "handler", "signal row coverage/order drift"),
        ("execution_outside_window", "handler", "signal row coverage/order drift"),
        ("source_window_mismatch", "scaled", "source train/validation chronology drift"),
        ("overlapping_windows", "scaled", "window native-grid drift"),
        ("source_window_mismatch", "handler", "source train/validation chronology drift"),
        ("overlapping_windows", "cash", "source train/validation chronology drift"),
        ("fallback_position", "scaled", "position scale derivation drift"),
        ("fallback_return", "scaled", "execution scale derivation drift"),
        ("fallback_base_return", "scaled", "fallback base return/event parity drift"),
        ("profile_unknown_key", "handler", "combined profile bytes drift"),
        ("bool", "handler", "manifest controls are invalid"),
        ("policy_bool", "handler", "source policy drift"),
        ("effective_bool", "handler", "effective leaf integer drift"),
        ("receipt_count_bool", "handler", "receipt integer controls are invalid"),
        ("scale", "handler", "variant scale/rows drift"),
        ("cost", "handler", "cost tape order drift"),
        ("overflow", "handler", "JSON.controls.orders_submitted must be finite"),
        ("extra", "handler", "router_signal_receipt_v2 keys are invalid"),
        ("window", "handler", "window range drift"),
        ("role", "handler", "window fold binding drift"),
        ("position_engine", "handler", "position scale derivation drift"),
        ("shared", "scaled", "shared MDD return commitment drift"),
        ("runner_source_drift", "cash", "commit root binding drift"),
        ("cost_bps_float", "handler", "cost tape order drift"),
        ("duplicate_row_digest", "handler", "cost tape row digest is reused"),
    ],
)
def test_router_v2_exploit_boundaries_stop(
    tmp_path: Path, mutation: str, mode: str, reason: str
) -> None:
    _, paths, trusted = _bundle(tmp_path, mode, mutation)
    if mutation == "trusted":
        trusted["source"] = "f" * 64
    report = _report(paths, trusted)
    assert report.status == "STOP"
    assert report.reasons == (reason,)


def test_router_v2_fallback_scale_counts_initial_loss_drawdown() -> None:
    scale = replay._fallback_scale(
        {
            "train_returns_ppm": [100_000],
            "validation_returns_ppm": [-200_000, 1_000_000],
        },
        replay.Fraction(10, 100),
        replay.Fraction(3),
    )

    assert scale == 500_000


@pytest.mark.parametrize(
    ("eligibility", "expected_branch", "expected_label"),
    [
        ((True, False), "strict_core_scaled", replay.BALANCED_LABEL),
        ((False, True), "strict_core_scaled", replay.GROWTH_LABEL),
        ((False, False), "strict_core_cash", None),
    ],
)
def test_router_v2_strict_core_requires_selected_candidate_eligibility(
    tmp_path: Path,
    eligibility: tuple[bool, bool],
    expected_branch: str,
    expected_label: str | None,
) -> None:
    _, paths, trusted = _bundle(
        tmp_path,
        "scaled",
        strict_candidate_eligibility=eligibility,
    )
    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
    final_strict = source["folds"][-1]["strict_core"]
    candidates = final_strict["candidates"]

    assert trusted["source"] == hashlib.sha256(paths["source"].read_bytes()).hexdigest()
    assert trusted["commit"] == hashlib.sha256(paths["commit"].read_bytes()).hexdigest()
    assert manifest["provenance"]["source_artifact_sha256"] == trusted["source"]
    assert commit["source_artifact_sha256"] == trusted["source"]
    assert commit["manifest_sha256"] == hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()
    assert [candidate["candidate_label"] for candidate in candidates] == [
        replay.BALANCED_LABEL,
        replay.GROWTH_LABEL,
    ]
    assert [candidate["train_return_ppm"] >= -20_000 for candidate in candidates] == list(
        eligibility
    )
    assert _report(paths, trusted).status == "PASS"

    selection = manifest["folds"][-1]["selection"]
    assert selection["branch"] == expected_branch
    assert selection["selected_label"] == expected_label
    assert selection["leaves"] == (final_strict["leaves"] if expected_label is not None else [])

    if eligibility == (True, False):
        _reroot_manifest_selection(paths, trusted, -1, replay.GROWTH_LABEL)
        rerooted_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        rerooted_commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
        assert rerooted_manifest["folds"][-1]["selection"]["selected_label"] == replay.GROWTH_LABEL
        assert (
            rerooted_commit["manifest_sha256"]
            == hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()
        )
        assert trusted["commit"] == hashlib.sha256(paths["commit"].read_bytes()).hexdigest()

        report = _report(paths, trusted)
        assert report.status == "STOP"
        assert report.reasons == ("selection replay drift",)


def test_router_v2_public_cost_base_tape_commitment_control_and_drift(
    tmp_path: Path,
) -> None:
    _, paths, trusted = _bundle(tmp_path, "handler")
    assert _report(paths, trusted).status == "PASS"

    _, paths, trusted = _bundle(tmp_path, "handler", "cost_base_commitment_drift")
    report = _report(paths, trusted)
    assert report.status == "STOP"
    assert report.reasons == ("cost tape base commitment drift",)


def test_router_v2_aggregate_multi_symbol_leaf_returns() -> None:
    rows = replay._aggregate_candidate_return_rows(
        [
            {
                "execution_rows": [
                    {
                        "timestamp_utc": "2025-01-01T00:00:00Z",
                        "symbol": "BTCUSDT",
                        "return_ppm": 10,
                    },
                    {
                        "timestamp_utc": "2025-01-01T00:00:00Z",
                        "symbol": "ETHUSDT",
                        "return_ppm": 20,
                    },
                ]
            }
        ],
        [{"source_weight_ppm": 1_000_000}],
    )
    assert rows == [{"timestamp_utc": "2025-01-01T00:00:00Z", "return_ppm": 30}]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (replay.Fraction(1, 2), 1),
        (replay.Fraction(-1, 2), -1),
        (replay.Fraction(2, 5), 0),
        (replay.Fraction(-2, 5), 0),
        (replay.Fraction(3, 2), 2),
        (replay.Fraction(-3, 2), -2),
    ],
)
def test_router_v2_half_up_signed_fraction_boundaries(
    value: replay.Fraction, expected: int
) -> None:
    assert replay._round_fraction_half_up(value) == expected


def test_router_v2_semantic_tie_preserves_source_order(monkeypatch: pytest.MonkeyPatch) -> None:
    history = [f"{index:064x}" for index in range(4)]
    leaves: list[object] = []

    def candidate(label: str) -> dict[str, object]:
        return {
            "candidate_label": label,
            "source_candidate": {
                "candidate_label": label,
                "family": "strict_efficiency",
                "return_count": 1,
                "row": {
                    "profile_kind": "strategy",
                    "uses_locked_oos_for_selection": False,
                    "same_month_self_feeding": False,
                    "current_fold_oos_used_for_weighting": False,
                    "post_oos_research_variant": False,
                    "requires_fresh_forward_shadow": False,
                },
            },
            "train_return_ppm": 100_000,
            "train_mdd_ppm": 100_000,
            "validation_return_ppm": 100_000,
            "validation_mdd_ppm": 100_000,
            "history_receipt_sha256s": history,
            "leaves": leaves,
            "leaf_list_sha256": _sha(leaves),
        }

    source_fold = {
        "candidates": [candidate("first"), candidate("second")],
        "strict_core": {
            "candidates": [],
            "leaves": [],
            "leaf_list_sha256": _sha([]),
            "shared_mdd_receipt_sha256": None,
        },
    }
    prior = [({"fold_id": f"f{index}"}, "a" * 64, {"BTCUSDT"}) for index in range(4)]
    monkeypatch.setattr(replay, "_history", lambda *args: 100_000)
    branch, label, _, _, _ = replay._decision(source_fold, prior, object())
    assert branch == "pre_registered_lagged_plus_validation_leaf"
    assert label == "first"


def test_router_v2_strict_core_scaled_negative_short_outputs(tmp_path: Path) -> None:
    manifest, paths, trusted = _bundle(tmp_path, "scaled", "negative_scaled_short")

    assert _report(paths, trusted).status == "PASS"
    execution = manifest["folds"][4]["variants"][0]["execution_receipts"][0]
    signal = json.loads(paths[execution["signal_receipt_sha256"]].read_text(encoding="utf-8"))
    position = json.loads(paths[execution["position_receipt_sha256"]].read_text(encoding="utf-8"))
    engine = json.loads(paths[execution["engine_receipt_sha256"]].read_text(encoding="utf-8"))
    assert signal["signal_rows"][0]["signal_ppm"] == -999_999
    assert position["position_rows"][0]["position_ppm"] == -1_099_999
    assert engine["execution_rows"][0]["base_return_ppm"] == -999_999
    assert engine["execution_rows"][0]["return_ppm"] == -1_099_999


def test_router_v2_rerooted_initial_loss_shared_mdd_stops(tmp_path: Path) -> None:
    _, paths, trusted = _bundle(tmp_path, "scaled", "shared_initial_loss_rerooted")
    _rebuild(paths, trusted)

    report = _report(paths, trusted)
    assert report.status == "STOP"
    assert report.reasons == ("variant scale/rows drift",)


def test_router_v2_negative_position_return_history_replay(tmp_path: Path) -> None:
    _, paths, trusted = _bundle(tmp_path, "handler", "negative_history")
    assert _report(paths, trusted).status == "PASS"


def test_router_v2_weighted_nonlexicographic_multi_symbol_aggregation() -> None:
    rows = replay._aggregate_candidate_return_rows(
        [
            {
                "execution_rows": [
                    {"timestamp_utc": "2025-01-01T00:00:00Z", "symbol": "Z", "return_ppm": 20},
                    {"timestamp_utc": "2025-01-01T00:00:00Z", "symbol": "A", "return_ppm": 10},
                ]
            },
            {
                "execution_rows": [
                    {"timestamp_utc": "2025-01-01T00:00:00Z", "symbol": "Z", "return_ppm": 40},
                    {"timestamp_utc": "2025-01-01T00:00:00Z", "symbol": "A", "return_ppm": 30},
                ]
            },
        ],
        [{"source_weight_ppm": 250_000}, {"source_weight_ppm": 750_000}],
    )
    assert rows == [{"timestamp_utc": "2025-01-01T00:00:00Z", "return_ppm": 60}]


def test_router_v2_weighted_nonlexicographic_history_bundle(tmp_path: Path) -> None:
    manifest, paths, trusted = _bundle(tmp_path, "handler", "weighted_nonlexicographic_history")
    history = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths.values()
        if path.suffix == ".json"
        and json.loads(path.read_text(encoding="utf-8")).get("schema")
        == "router_history_receipt_v2"
    ]

    assert len(history) == 4
    assert all(
        len(row["candidate_return_rows"]) == 24
        and row["candidate_return_rows"][0]["timestamp_utc"] == row["locked_oos"]["start_utc"]
        and all(item["return_ppm"] == 60 for item in row["candidate_return_rows"])
        for row in history
    )
    assert all(
        json.loads(paths[row["candidate_aggregation_receipt_sha256"]].read_text(encoding="utf-8"))[
            "weights_ppm"
        ]
        == [250_000, 750_000]
        for row in history
    )
    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    engines = [
        json.loads(paths[digest].read_text(encoding="utf-8"))
        for digest in history[0]["engine_receipt_sha256s"]
    ]
    assert (
        replay._aggregate_candidate_return_rows(
            engines, source["folds"][0]["candidates"][0]["leaves"]
        )
        == history[0]["candidate_return_rows"]
    )
    assert manifest["folds"][4]["selection"]["branch"] == "strict_core_cash"
    assert _report(paths, trusted).status == "PASS"
