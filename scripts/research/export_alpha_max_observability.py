#!/usr/bin/env python3
"""Export compact, audit-ready Alpha-Max manifests and actual-engine evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any


_MANIFEST_PHASES = ("validation_train_fit", "prelock_final_refit")
_EXPECTED_MANIFESTS_PER_PHASE = 17
_EXPECTED_ACTUAL_CELLS = 68
_EXPECTED_FOLD_RUNS = {
    "validation": 816,
    "historical_exposed_evaluation": 680,
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _safe_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return relative


def _validated_root(value: str, *, field: str) -> Path:
    requested = Path(value)
    if not requested.is_absolute() or requested.is_symlink():
        raise ValueError(f"{field} must be an absolute nonsymlink directory")
    root = requested.resolve(strict=True)
    status = root.stat()
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{field} is not a directory")
    return root


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _open_directory_chain(path: Path) -> tuple[int, list[int]]:
    if not path.is_absolute():
        raise ValueError("directory chain must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    opened = [os.open("/", flags)]
    try:
        for part in path.parts[1:]:
            parent_fd = opened[-1]
            observed = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ValueError(f"unsafe directory component: {part!r}")
            child_fd = os.open(part, flags, dir_fd=parent_fd)
            opened_child = os.fstat(child_fd)
            if (
                int(opened_child.st_dev) != int(observed.st_dev)
                or int(opened_child.st_ino) != int(observed.st_ino)
                or not stat.S_ISDIR(opened_child.st_mode)
            ):
                os.close(child_fd)
                raise ValueError(f"directory component changed while opening: {part!r}")
            opened.append(child_fd)
        return opened[-1], opened
    except Exception:
        for descriptor in reversed(opened):
            os.close(descriptor)
        raise


def _read_canonical_json(root: Path, relative: str) -> tuple[dict[str, Any], bytes]:
    pure = _safe_relative_path(relative)
    _root_fd, opened = _open_directory_chain(root)
    descriptor: int | None = None
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        parent_fd = opened[-1]
        for part in pure.parts[:-1]:
            observed = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ValueError(f"unsafe JSON parent: {relative}")
            child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            opened_child = os.fstat(child_fd)
            if (
                int(opened_child.st_dev) != int(observed.st_dev)
                or int(opened_child.st_ino) != int(observed.st_ino)
                or not stat.S_ISDIR(opened_child.st_mode)
            ):
                os.close(child_fd)
                raise ValueError(f"JSON parent changed while opening: {relative}")
            opened.append(child_fd)
            parent_fd = child_fd

        observed_file = os.stat(pure.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(observed_file.st_mode)
            or not stat.S_ISREG(observed_file.st_mode)
            or int(observed_file.st_nlink) != 1
        ):
            raise ValueError(f"unsafe JSON artifact: {relative}")
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(pure.name, file_flags, dir_fd=parent_fd)
        opened_file = os.fstat(descriptor)
        if _stat_identity(opened_file) != _stat_identity(observed_file):
            raise ValueError(f"JSON artifact changed while opening: {relative}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        sealed_file = os.fstat(descriptor)
        if _stat_identity(sealed_file) != _stat_identity(opened_file):
            raise ValueError(f"JSON artifact changed while reading: {relative}")
        raw = b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for opened_fd in reversed(opened):
            os.close(opened_fd)
    payload = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )
    if type(payload) is not dict or raw != _canonical_bytes(payload):
        raise ValueError(f"noncanonical JSON artifact: {relative}")
    return payload, raw


def _required(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"missing required keys: {missing}")
    return {key: payload[key] for key in keys}


def _inventory_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"symlink in sealed inventory: {relative}")
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode) or int(status.st_nlink) != 1:
            raise ValueError(f"unsafe sealed inventory entry: {relative}")
        paths.add(relative)
    return paths


def _verify_seal_inventory(
    root: Path,
    seal_payload: dict[str, Any],
    seal_bytes: bytes,
) -> tuple[str, dict[str, tuple[int, str]]]:
    kind = seal_payload.get("artifact_kind")
    if kind == "alpha_max_immutable_prelock_seal.v1":
        inventory_key = "artifacts"
        if (
            seal_payload.get("immutable") is not True
            or seal_payload.get("historical_evaluation_inputs_included") is not False
        ):
            raise ValueError("prelock seal policy invalid")
    elif kind == "alpha_max_append_only_historical_package.v1":
        inventory_key = "historical_artifacts"
        if seal_payload.get("immutable") is not True:
            raise ValueError("historical seal policy invalid")
    else:
        raise ValueError("unsupported Alpha-Max seal kind")
    raw_inventory = seal_payload.get(inventory_key)
    if type(raw_inventory) is not list:
        raise ValueError("sealed artifact inventory missing")
    expected: dict[str, tuple[int, str]] = {}
    for entry in raw_inventory:
        if type(entry) is not dict or set(entry) != {"byte_count", "relative_path", "sha256"}:
            raise ValueError("sealed artifact inventory entry invalid")
        relative = _safe_relative_path(str(entry["relative_path"])).as_posix()
        byte_count = entry["byte_count"]
        sha256 = entry["sha256"]
        if (
            relative == "SEALED.json"
            or relative in expected
            or type(byte_count) is not int
            or byte_count < 0
            or type(sha256) is not str
            or len(sha256) != 64
            or any(value not in "0123456789abcdef" for value in sha256)
        ):
            raise ValueError("sealed artifact inventory entry invalid")
        expected[relative] = (byte_count, sha256)
    observed = _inventory_paths(root)
    if observed != set(expected) | {"SEALED.json"}:
        raise ValueError("sealed artifact inventory path mismatch")
    for relative, (byte_count, sha256) in expected.items():
        _payload, raw = _read_canonical_json(root, relative)
        if len(raw) != byte_count or _sha256(raw) != sha256:
            raise ValueError(f"sealed artifact hash mismatch: {relative}")
    if kind == "alpha_max_immutable_prelock_seal.v1" and (
        seal_payload.get("artifact_count") != len(expected)
        or seal_payload.get("inventory_sha256") != _sha256(_canonical_bytes(raw_inventory))
    ):
        raise ValueError("prelock seal inventory summary mismatch")
    return _sha256(seal_bytes), expected


def _read_verified_json(
    root: Path,
    relative: str,
    inventory: dict[str, tuple[int, str]],
) -> tuple[dict[str, Any], bytes]:
    expected = inventory.get(relative)
    if expected is None:
        raise ValueError(f"artifact absent from sealed inventory: {relative}")
    payload, raw = _read_canonical_json(root, relative)
    if (len(raw), _sha256(raw)) != expected:
        raise ValueError(f"sealed artifact changed after inventory verification: {relative}")
    return payload, raw


def _verify_bundle_pair(
    *,
    domain: str,
    bundle_seal_payload: dict[str, Any],
    bundle_seal_sha256: str,
    manifest_seal_payload: dict[str, Any],
    manifest_seal_sha256: str,
) -> None:
    if manifest_seal_payload.get("artifact_kind") != "alpha_max_immutable_prelock_seal.v1":
        raise ValueError("manifest root is not an immutable prelock bundle")
    if domain == "validation":
        if (
            bundle_seal_payload.get("artifact_kind") != "alpha_max_immutable_prelock_seal.v1"
            or bundle_seal_sha256 != manifest_seal_sha256
        ):
            raise ValueError("validation bundle/manifest seal mismatch")
        return
    if (
        bundle_seal_payload.get("artifact_kind") != "alpha_max_append_only_historical_package.v1"
        or bundle_seal_payload.get("prelock_seal_sha256") != manifest_seal_sha256
    ):
        raise ValueError("historical bundle/prelock seal mismatch")


def _verify_historical_prelock_binding(
    bundle_root: Path,
    bundle_inventory: dict[str, tuple[int, str]],
    manifest_seal_bytes: bytes,
) -> None:
    _binding_payload, binding_bytes = _read_verified_json(
        bundle_root,
        "binding/prelock_seal.json",
        bundle_inventory,
    )
    if binding_bytes != manifest_seal_bytes:
        raise ValueError("historical prelock seal binding mismatch")


def _manifest_summary(
    root: Path,
    inventory: dict[str, tuple[int, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    summaries: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for phase in _MANIFEST_PHASES:
        prefix = f"manifests/{phase}/"
        relative_paths = sorted(
            relative
            for relative in inventory
            if relative.startswith(prefix)
            and len(PurePosixPath(relative).parts) == 3
            and PurePosixPath(relative).suffix == ".json"
        )
        counts[phase] = len(relative_paths)
        if len(relative_paths) != _EXPECTED_MANIFESTS_PER_PHASE:
            raise ValueError(f"manifest count mismatch for {phase}: {len(relative_paths)}")
        for relative in relative_paths:
            payload, raw = _read_verified_json(root, relative, inventory)
            children = payload.get("children")
            if type(children) is not list or not children:
                raise ValueError(f"manifest children missing: {relative}")
            child_summaries = []
            for child in children:
                if type(child) is not dict:
                    raise ValueError(f"manifest child invalid: {relative}")
                child_summaries.append(
                    _required(
                        child,
                        "candidate_id",
                        "strategy_class",
                        "candidate_symbols",
                        "symbols",
                        "params",
                        "weight",
                        "leaf_gross",
                        "leaf_gross_cap",
                        "netting_group",
                        "netting_group_gross_cap",
                    )
                )
            summaries.append(
                {
                    **_required(
                        payload,
                        "allocation_method",
                        "candidate_symbols",
                        "admitted_symbols",
                        "admission_manifest_sha256",
                        "gross_cap",
                        "cash_weight",
                        "optimizer_provenance",
                        "correlation_input_provenance",
                    ),
                    "children": child_summaries,
                    "manifest_path": relative,
                    "manifest_byte_count": len(raw),
                    "manifest_sha256": _sha256(raw),
                    "phase": phase,
                    "row_id": PurePosixPath(relative).stem,
                }
            )
    return summaries, counts


def _reconciliation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = _required(
        payload,
        "application_count",
        "application_trace_hashes",
        "applied_commission_total",
        "complete",
        "fee_reconciled",
        "funding_payment_total",
        "funding_reconciled",
        "liquidation_cost_total",
        "liquidation_reconciled",
        "model_commission_total",
        "no_fill_attempt_count",
        "no_fill_excluded_from_bijection",
        "portfolio_fee_total",
        "portfolio_funding_total",
        "portfolio_liquidation_total",
        "pricing_application_bijection",
        "pricing_trace_count",
        "pricing_trace_hashes",
        "zero_applied_application_count",
    )
    if not all(
        result[key]
        for key in (
            "complete",
            "fee_reconciled",
            "funding_reconciled",
            "liquidation_reconciled",
            "no_fill_excluded_from_bijection",
            "pricing_application_bijection",
        )
    ):
        raise ValueError("incomplete cost reconciliation")
    if len(result["pricing_trace_hashes"]) != result["pricing_trace_count"]:
        raise ValueError("pricing trace count/hash mismatch")
    if len(result["application_trace_hashes"]) != result["application_count"]:
        raise ValueError("application count/hash mismatch")
    result["reconciliation_sha256"] = _sha256(_canonical_bytes(payload))
    return result


def _diagnostics_summary(payload: dict[str, Any]) -> dict[str, Any]:
    observations = payload.get("capacity_observations")
    capacity = payload.get("capacity")
    if type(observations) is not list or type(capacity) is not dict:
        raise ValueError("capacity diagnostics missing")
    if len(observations) != capacity.get("observation_count"):
        raise ValueError("capacity observation count mismatch")
    observation_sha256 = _sha256(_canonical_bytes(observations))
    if observation_sha256 != payload.get("capacity_observation_set_sha256"):
        raise ValueError("capacity observation hash mismatch")
    result = _required(
        payload,
        "turnover_rpt",
        "target_gross_exposure",
        "ending_realized_gross_exposure",
        "ending_realized_gross_undefined_reason",
        "liquidity_clip_count",
        "reduce_only_clip_count",
        "no_fill_attempt_count",
        "ending_market_value_usdt",
        "symbol_contribution_usdt",
        "contribution_total_usdt",
        "fold_pnl_usdt",
        "reconciliation_residual_usdt",
        "report_only",
        "selection_influence",
    )
    if result["report_only"] is not True or result["selection_influence"] is not False:
        raise ValueError("diagnostics influenced selection")
    if not math.isclose(
        float(result["fold_pnl_usdt"]) - float(result["contribution_total_usdt"]),
        float(result["reconciliation_residual_usdt"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(result["reconciliation_residual_usdt"]), 0.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("symbol contribution reconciliation failed")
    result["capacity"] = capacity
    result["capacity_observation_count"] = len(observations)
    result["capacity_observation_set_sha256"] = observation_sha256
    return result


def _run_summary(
    payload: dict[str, Any],
    *,
    expected_domain: str,
    expected_row_id: str,
    expected_manifest: dict[str, Any],
) -> dict[str, Any]:
    reconciliation = payload.get("reconciliation")
    diagnostics = payload.get("report_only_diagnostics")
    if type(reconciliation) is not dict or type(diagnostics) is not dict:
        raise ValueError("actual-engine attribution evidence missing")
    result = _required(
        payload,
        "row_id",
        "domain",
        "split_or_fold_id",
        "nominal_cost_bps",
        "seed",
        "admitted_symbols",
        "universe_sha256",
        "config_sha256",
        "runtime_contract_sha256",
        "effective_config",
        "effective_config_sha256",
        "runtime_read_audit",
        "runtime_read_audit_sha256",
        "raw_root_receipts",
        "raw_root_set_sha256",
        "feature_root_receipts",
        "feature_root_set_sha256",
        "manifest_receipt",
        "capsule_receipt",
        "market_event_count",
        "equity_observation_count",
        "signal_event_count",
        "order_event_count",
        "fill_event_count",
        "trade_count",
        "starting_cash",
        "starting_equity",
        "starting_open_position_count",
        "starting_open_order_count",
        "starting_used_margin",
        "ending_cash",
        "ending_equity",
        "ruin_detected",
        "full_event_equity",
        "native_finalization",
        "pricing_trace_count",
        "pricing_trace_set_sha256",
        "application_count",
        "application_set_sha256",
        "no_fill_attempt_count",
        "no_fill_attempt_set_sha256",
        "funding_ledger_count",
        "funding_ledger_set_sha256",
        "liquidation_event_count",
        "liquidation_event_set_sha256",
    )
    manifest_receipt = result["manifest_receipt"]
    expected_receipt = {
        "byte_count": expected_manifest["manifest_byte_count"],
        "phase": expected_manifest["phase"],
        "relative_path": expected_manifest["manifest_path"],
        "row_id": expected_row_id,
        "sha256": expected_manifest["manifest_sha256"],
    }
    if (
        result["domain"] != expected_domain
        or result["row_id"] != expected_row_id
        or type(manifest_receipt) is not dict
        or manifest_receipt != expected_receipt
    ):
        raise ValueError("actual-engine manifest receipt mismatch")
    result["actual_engine_run_sha256"] = _sha256(_canonical_bytes(payload))
    result["reconciliation"] = _reconciliation_summary(reconciliation)
    result["report_only_diagnostics"] = _diagnostics_summary(diagnostics)
    return result


def _cell_summaries(
    root: Path,
    domain: str,
    manifest_index: dict[tuple[str, str], dict[str, Any]],
    inventory: dict[str, tuple[int, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actual: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    prefix = f"evidence/{domain}/cells/"
    relative_paths = sorted(
        relative
        for relative in inventory
        if relative.startswith(prefix)
        and len(PurePosixPath(relative).parts) == 5
        and PurePosixPath(relative).suffix == ".json"
    )
    if not relative_paths:
        raise ValueError(f"evidence cells missing for domain: {domain}")
    for relative in relative_paths:
        payload, raw = _read_verified_json(root, relative, inventory)
        pre_gate = payload.get("pre_gate_evidence")
        identity = {
            **_required(
                payload,
                "row_id",
                "domain",
                "nominal_cost_bps",
                "status",
                "evidence_tier",
                "selection_valid",
            ),
            "cell_path": relative,
            "cell_sha256": _sha256(raw),
        }
        if pre_gate is None:
            other.append(identity)
            continue
        if type(pre_gate) is not dict:
            raise ValueError(f"pre-gate evidence invalid: {relative}")
        expected_phase = "validation_train_fit" if domain == "validation" else "prelock_final_refit"
        expected_manifest = manifest_index.get((expected_phase, str(payload["row_id"])))
        if expected_manifest is None:
            raise ValueError(f"manifest missing for actual-engine cell: {relative}")
        folds = pre_gate.get("fold_runs")
        if type(folds) is not list or not folds:
            raise ValueError(f"fold evidence missing: {relative}")
        runs = []
        for fold in folds:
            if type(fold) is not dict or type(fold.get("actual_engine_run")) is not dict:
                raise ValueError(f"actual-engine run missing: {relative}")
            runs.append(
                _run_summary(
                    fold["actual_engine_run"],
                    expected_domain=domain,
                    expected_row_id=str(payload["row_id"]),
                    expected_manifest=expected_manifest,
                )
            )
        actual.append(
            {
                **identity,
                "fold_run_count": len(runs),
                "fold_run_set_sha256": pre_gate.get("fold_run_set_sha256"),
                "fold_runs": runs,
                "pre_gate_evidence_sha256": _sha256(_canonical_bytes(pre_gate)),
                "terminal_gate_evidence": payload.get("terminal_gate_evidence"),
            }
        )
    fold_count = sum(cell["fold_run_count"] for cell in actual)
    if len(actual) != _EXPECTED_ACTUAL_CELLS:
        raise ValueError(f"actual-engine cell count mismatch: {len(actual)}")
    if fold_count != _EXPECTED_FOLD_RUNS[domain]:
        raise ValueError(f"physical fold-run count mismatch: {fold_count}")
    return actual, other


def _optional_artifact(
    root: Path,
    relative: str,
    inventory: dict[str, tuple[int, str]],
) -> dict[str, Any] | None:
    if relative not in inventory:
        return None
    payload, raw = _read_verified_json(root, relative, inventory)
    return {"path": relative, "sha256": _sha256(raw), "payload": payload}


def build_export(bundle_root: Path, manifest_root: Path, domain: str) -> dict[str, Any]:
    seal_payload, seal_bytes = _read_canonical_json(bundle_root, "SEALED.json")
    bundle_seal_sha256, bundle_inventory = _verify_seal_inventory(
        bundle_root,
        seal_payload,
        seal_bytes,
    )
    manifest_seal_payload, manifest_seal_bytes = _read_canonical_json(manifest_root, "SEALED.json")
    manifest_seal_sha256, manifest_inventory = _verify_seal_inventory(
        manifest_root,
        manifest_seal_payload,
        manifest_seal_bytes,
    )
    _verify_bundle_pair(
        domain=domain,
        bundle_seal_payload=seal_payload,
        bundle_seal_sha256=bundle_seal_sha256,
        manifest_seal_payload=manifest_seal_payload,
        manifest_seal_sha256=manifest_seal_sha256,
    )
    if domain == "historical_exposed_evaluation":
        _verify_historical_prelock_binding(
            bundle_root,
            bundle_inventory,
            manifest_seal_bytes,
        )
    manifests, manifest_counts = _manifest_summary(manifest_root, manifest_inventory)
    manifest_index = {(value["phase"], value["row_id"]): value for value in manifests}
    if len(manifest_index) != len(manifests):
        raise ValueError("duplicate manifest phase/row identity")
    actual_cells, non_actual_cells = _cell_summaries(
        bundle_root,
        domain,
        manifest_index,
        bundle_inventory,
    )
    terminal_name = "prelock" if domain == "validation" else "historical"
    falsifier = (
        "diagnostics/validation/trend_liquidity_falsifier.json"
        if domain == "validation"
        else "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json"
    )
    return {
        "actual_engine_cell_count": len(actual_cells),
        "actual_engine_cells": actual_cells,
        "artifact_kind": "alpha_max_operator_observability_export.v1",
        "bundle_root": str(bundle_root),
        "bundle_seal": {
            "payload": seal_payload,
            "sha256": bundle_seal_sha256,
        },
        "domain": domain,
        "manifest_counts": manifest_counts,
        "manifest_root": str(manifest_root),
        "manifest_root_seal": {
            "payload": manifest_seal_payload,
            "sha256": manifest_seal_sha256,
        },
        "manifests": manifests,
        "non_actual_cells": non_actual_cells,
        "physical_fold_run_count": sum(cell["fold_run_count"] for cell in actual_cells),
        "supporting_artifacts": [
            value
            for value in (
                _optional_artifact(
                    manifest_root,
                    "admission/train.json",
                    manifest_inventory,
                ),
                _optional_artifact(
                    manifest_root,
                    "admission/train_liquidity_buckets.json",
                    manifest_inventory,
                ),
                _optional_artifact(bundle_root, falsifier, bundle_inventory),
                _optional_artifact(
                    bundle_root,
                    f"terminal/{terminal_name}.json",
                    bundle_inventory,
                ),
            )
            if value is not None
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument(
        "--domain",
        choices=tuple(_EXPECTED_FOLD_RUNS),
        required=True,
    )
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    bundle_root = _validated_root(args.bundle_root, field="bundle_root")
    manifest_root = _validated_root(args.manifest_root, field="manifest_root")
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("output must be a new absolute path")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(build_export(bundle_root, manifest_root, args.domain))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(output, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    print(
        json.dumps(
            {"byte_count": len(payload), "output": str(output), "sha256": _sha256(payload)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
