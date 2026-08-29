from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
for _key in tuple(os.environ):
    if _key.startswith("LQ_"):
        del os.environ[_key]

import lumina_quant.research.alpha_max_engine_runner as runner
import lumina_quant.research.alpha_max_evidence as evidence
import polars as pl


_COMPONENT_IDS = (
    "component_carry_1x",
    "component_near_high_1x",
    "component_trend_1x",
)
_REQUIRED_SEALS = (("train", "raw"), ("warmup", "feature"), ("train", "feature"))


def _deterministic_replay(preflight, *, output_root, manifest_receipt, **_kwargs):
    """Importable test replay; children still perform every worker authority check."""
    telemetry = Path(output_root).resolve().parent / "worker-pids"
    telemetry.mkdir(exist_ok=True)
    (telemetry / f"{os.getpid()}-{manifest_receipt.row_id}").write_text(
        str(os.getpid()), encoding="ascii"
    )
    # Keep all three work items alive long enough for the pool to fill its cap.
    time.sleep(0.35)
    train = preflight.phase_windows["train"]
    start = datetime.fromisoformat(train.start_utc.replace("Z", "+00:00")).astimezone(UTC)
    end = datetime.fromisoformat(train.end_utc.replace("Z", "+00:00")).astimezone(UTC)
    calendar = tuple(
        (start + timedelta(days=index + 1)).date().isoformat()
        for index in range((end - start).days)
    )
    completed = [["BTCUSDT", end.date().isoformat()]]
    coverage = {
        "adapter_class": "ValidSpawnFixtureAdapter",
        "barrier_mode": "none",
        "barrier_closed_keys": [],
        "barrier_pending_keys": [],
        "barrier_symbol_coverage": {},
        "completed_native_count_by_symbol": {"BTCUSDT": 1},
        "completed_native_keys": completed,
        "failed_native_keys": {},
        "finalization_barrier_keys": [],
        "finalization_completed_native_keys": completed,
        "last_completed_native_key_by_symbol": {"BTCUSDT": end.date().isoformat()},
        "native_timeframe": "1d",
        "partial_bucket_error": None,
    }
    native = evidence.build_alpha_max_native_finalization_receipt(
        boundary_utc=end,
        finalized_children={manifest_receipt.row_id: 1},
        native_coverage_by_child={manifest_receipt.row_id: coverage},
        discarded_signal_count=0,
        discarded_signal_sha256=hashlib.sha256(b"").hexdigest(),
    )
    returns = tuple(0.0001 + index * 0.0 for index in range(len(calendar)))
    return calendar, returns, native


# Spawn imports this file as __mp_main__; this is deliberately the only production seam patched.
runner._alpha_max_replay_training_component_returns = _deterministic_replay


def _month_after(value: datetime) -> datetime:
    return (
        value.replace(year=value.year + 1, month=1)
        if value.month == 12
        else value.replace(month=value.month + 1)
    )


def _root_seal(root: Path, root_id: str, root_kind: str):
    root.mkdir(parents=True)
    start, end = evidence._ROOT_INTERVALS[root_id]
    starts = MappingProxyType(dict.fromkeys(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS, start))
    ends = MappingProxyType(dict.fromkeys(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS, end))
    entries = []
    for symbol in evidence.ALPHA_MAX_CANDIDATE_SYMBOLS:
        if root_kind == "raw":
            cursor = start.replace(day=1)
            while cursor < end:
                partition_end = _month_after(cursor)
                owned_start, owned_end = max(start, cursor), min(end, partition_end)
                entries.append(
                    evidence.AlphaMaxTreeEntry(
                        relative_path=f"market_ohlcv_1s/binance/{symbol}/{cursor:%Y-%m}.parquet",
                        byte_count=1,
                        mode=0o444,
                        mtime_ns=0,
                        minimum_timestamp_ms=evidence._epoch_ms(owned_start),
                        maximum_timestamp_ms=evidence._epoch_ms(owned_end) - 1000,
                        row_count=int((owned_end - owned_start).total_seconds()),
                        maximum_gap_ms=1000,
                        sha256=hashlib.sha256(f"{symbol}:{cursor:%Y-%m}".encode()).hexdigest(),
                    )
                )
                cursor = partition_end
        else:
            cursor = start
            while cursor < end:
                partition_end = cursor + timedelta(days=1)
                interval = evidence._alpha_max_funding_interval_ms(symbol)
                grid = evidence._alpha_max_expected_grid_timestamps(
                    evidence._epoch_ms(max(start, cursor)),
                    evidence._epoch_ms(min(end, partition_end)),
                    interval,
                )
                entries.append(
                    evidence.AlphaMaxTreeEntry(
                        relative_path=(
                            f"feature_points/exchange=binance/symbol={symbol}/date={cursor:%Y-%m-%d}/part.parquet"
                        ),
                        byte_count=1,
                        mode=0o444,
                        mtime_ns=0,
                        minimum_timestamp_ms=grid[0],
                        maximum_timestamp_ms=grid[-1],
                        row_count=len(grid),
                        maximum_gap_ms=interval if len(grid) > 1 else 0,
                        sha256=hashlib.sha256(f"{symbol}:{cursor:%F}".encode()).hexdigest(),
                    )
                )
                cursor = partition_end
    ordered = tuple(sorted(entries, key=lambda entry: entry.relative_path))
    inventory = [
        {
            key: getattr(entry, key)
            for key in (
                "byte_count",
                "maximum_timestamp_ms",
                "maximum_gap_ms",
                "minimum_timestamp_ms",
                "mode",
                "mtime_ns",
                "relative_path",
                "row_count",
            )
        }
        for entry in ordered
    ]
    content = [
        {
            key: getattr(entry, key)
            for key in (
                "byte_count",
                "maximum_timestamp_ms",
                "maximum_gap_ms",
                "minimum_timestamp_ms",
                "relative_path",
                "row_count",
                "sha256",
            )
        }
        for entry in ordered
    ]
    availability = evidence._alpha_max_availability_sha256(starts, ends)
    inventory_sha = evidence._sha256_bytes(evidence._canonical_json_bytes(inventory, newline=True))
    content_sha = evidence._sha256_bytes(evidence._canonical_json_bytes(content, newline=True))
    payload = {
        "artifact_kind": "alpha_max_root_seal.v2",
        "availability_end_by_symbol": {
            key: value.isoformat().replace("+00:00", "Z") for key, value in ends.items()
        },
        "availability_sha256": availability,
        "availability_start_by_symbol": {
            key: value.isoformat().replace("+00:00", "Z") for key, value in starts.items()
        },
        "content_sha256": content_sha,
        "end_utc": end.isoformat().replace("+00:00", "Z"),
        "entries": [entry.to_payload() for entry in ordered],
        "exchange": "binance",
        "file_count": len(ordered),
        "inventory_sha256": inventory_sha,
        "path": str(root.resolve()),
        "root_id": root_id,
        "root_kind": root_kind,
        "start_utc": start.isoformat().replace("+00:00", "Z"),
        "symbols": list(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS),
    }
    raw = runner._canonical_bytes(payload) + b"\n"
    return evidence.parse_alpha_max_root_seal(
        raw,
        expected_root_id=root_id,
        expected_root_kind=root_kind,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _snapshot(root: Path) -> tuple[tuple[str, int, int, str], ...]:
    rows = []
    for path in sorted(root.rglob("*")):
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        rows.append((str(path.relative_to(root)), stat.st_ino, stat.st_mode, digest))
    return tuple(rows)


def _run_cases(base: Path) -> dict[int, dict[str, object]]:
    config = (
        Path(__file__).resolve().parents[2]
        / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
    ).resolve()
    preflight = runner.preflight_alpha_max_runtime_contract(config)
    output, checkpoint = (base / "output").resolve(), (base / "checkpoint").resolve()
    roots = {key: _root_seal(base / f"{key[0]}-{key[1]}", *key) for key in _REQUIRED_SEALS}
    contract = runner.seal_alpha_max_contract_manifest(
        Path(__file__).resolve().parents[2]
        / "configs/research/alpha_max_contract_manifest_20260711_listing_aware.json"
    )
    admitted = tuple(preflight.candidate_symbols[:5])
    prior = (base / "prior-trials.json").resolve()
    prior.write_bytes(b"[]\n")
    descriptor = runner._alpha_max_prelock_checkpoint_descriptor(
        preflight=preflight,
        contract_seal=contract,
        root_seals=roots,
        admitted_symbols=admitted,
        output_root=output,
        checkpoint_root=checkpoint,
        implementation_inventory=runner._alpha_max_checkpoint_implementation_inventory(),
        prior_trial_binding={
            "path": str(prior),
            "byte_count": 3,
            "sha256": hashlib.sha256(b"[]\n").hexdigest(),
        },
    )
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint, output_root=output, descriptor=descriptor, config_bytes=preflight.config_bytes
    )
    try:
        runner._alpha_max_create_or_resume_run_root(
            store.output_root,
            config_bytes=preflight.config_bytes,
            attempt_descriptor_sha256=store.descriptor_sha256,
        )
        root = store.bind_output_root()
        active = runner.preflight_alpha_max_runtime_contract(root / "inputs/config.json")
        nodes = {row["row_id"]: row for row in runner._alpha_max_current_nodes(active)}
        manifests = {
            component: runner._alpha_max_materialize_manifest_receipt(
                active,
                output_root=root,
                phase="validation_train_fit",
                row=nodes[component],
                weights={component: 1.0},
                gross=1.0,
                admitted_symbols=admitted,
                admission_sha256="a" * 64,
            )
            for component in _COMPONENT_IDS
        }
        bindings = tuple(
            runner._alpha_max_training_worker_binding_bytes(
                component_id=component,
                attempt_descriptor=descriptor,
                checkpoint_store=store,
                manifest=manifests[component],
                root_seals=roots,
            )
            for component in _COMPONENT_IDS
        )
        items = tuple(
            (component, binding, hashlib.sha256(binding).hexdigest())
            for component, binding in zip(_COMPONENT_IDS, bindings, strict=True)
        )
        runner._parse_alpha_max_training_worker_binding(
            items[0][1], expected_component_id=items[0][0], expected_sha256=items[0][2]
        )
        before = _snapshot(root)
        telemetry = base / "worker-pids"
        cases: dict[int, dict[str, object]] = {}
        for cap in (1, 2, 3, 4):
            telemetry.mkdir(exist_ok=True)
            for path in telemetry.iterdir():
                path.unlink()
            results = runner._run_alpha_max_training_component_workers(
                items, max_training_workers=cap
            )
            after = _snapshot(root)
            parsed = [
                runner._alpha_max_training_component_from_checkpoint(
                    payload,
                    preflight=active,
                    component_id=component,
                    manifest=manifests[component],
                )
                for component, status, payload, reason in results
                if status == "complete" and not reason
            ]
            if before != after:
                raise AssertionError("spawn_worker_mutated_output")
            pids = sorted({int(path.read_text(encoding="ascii")) for path in telemetry.iterdir()})
            cases[cap] = {"results": results, "parsed": len(parsed), "pids": pids}
        return cases
    finally:
        del store


def run(root: Path) -> dict[str, object]:
    parent_pid = os.getpid()
    if pl.DataFrame({"value": [1]}).select(pl.col("value").sum()).item() != 1:
        raise AssertionError("native_runtime_initialization_failed")
    ready = threading.Event()
    release = threading.Event()
    native_thread = threading.Thread(target=lambda: (ready.set(), release.wait(180)))
    native_thread.start()
    if not ready.wait(2):
        raise AssertionError("threaded_parent_initialization_failed")
    try:
        cases = _run_cases(root / "shared")
    finally:
        release.set()
        native_thread.join(2)
    if native_thread.is_alive():
        raise AssertionError("threaded_parent_cleanup_failed")
    baseline = tuple(result[2] for result in cases[1]["results"])
    assert all(
        tuple(item[0] for item in case["results"]) == _COMPONENT_IDS for case in cases.values()
    )
    assert all(
        all(result[1] == "complete" and result[3] == "" for result in case["results"])
        for case in cases.values()
    )
    assert all(case["parsed"] == 3 for case in cases.values())
    assert all(
        tuple(result[2] for result in case["results"]) == baseline for case in cases.values()
    )
    assert [len(cases[cap]["pids"]) for cap in (1, 2, 3, 4)] == [1, 2, 3, 3]
    assert all(parent_pid not in case["pids"] for case in cases.values())
    return {
        "pid_counts": [len(cases[cap]["pids"]) for cap in (1, 2, 3, 4)],
        "parent_pid": parent_pid,
    }


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]).resolve()), sort_keys=True))
