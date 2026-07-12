from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, redirect_stderr
from datetime import UTC, datetime, timedelta
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from types import MappingProxyType, SimpleNamespace
from typing import Any

import polars as pl

import lumina_quant.research.alpha_max_engine_runner as runner
import lumina_quant.research.alpha_max_evidence as evidence
import lumina_quant.strategies.artifact_portfolio_mode as artifact_mode


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
).resolve()
CONTRACT_PATH = (
    REPO_ROOT / "configs/research/alpha_max_contract_manifest_20260711_listing_aware.json"
).resolve()
PRELOCK_PATH = REPO_ROOT / "scripts/research/run_alpha_max_prelock.py"
HISTORICAL_PATH = REPO_ROOT / "scripts/research/run_alpha_max_historical_evaluation.py"
ROOT_IDS = (
    "warmup",
    "train",
    "purge",
    "validation",
    "embargo",
    "historical_exposed_evaluation",
)
PRELOCK_ROOT_IDS = ROOT_IDS[:-1]
_PRODUCTION_ROOT_INTERVALS = MappingProxyType(dict(evidence._ROOT_INTERVALS))
_FIXTURE_ROOT_INTERVALS = MappingProxyType(
    {
        root_id: (
            datetime(2024, 3, 1, 16, tzinfo=UTC) + timedelta(hours=8 * index),
            datetime(2024, 3, 1, 16, tzinfo=UTC) + timedelta(hours=8 * (index + 1)),
        )
        for index, root_id in enumerate(ROOT_IDS)
    }
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable_to_load:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _root_interval_scope(
    intervals: Mapping[str, tuple[datetime, datetime]],
):
    previous = dict(evidence._ROOT_INTERVALS)
    evidence._ROOT_INTERVALS.clear()
    evidence._ROOT_INTERVALS.update(intervals)
    try:
        yield
    finally:
        evidence._ROOT_INTERVALS.clear()
        evidence._ROOT_INTERVALS.update(previous)


def _write_raw_root(root: Path, root_id: str) -> None:
    start, end = evidence._ROOT_INTERVALS[root_id]
    month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months: list[datetime] = []
    while month < end:
        months.append(month)
        month = (
            month.replace(year=month.year + 1, month=1)
            if month.month == 12
            else month.replace(month=month.month + 1)
        )
    for symbol in evidence.ALPHA_MAX_CANDIDATE_SYMBOLS:
        directory = root / "market_ohlcv_1s" / "binance" / symbol
        directory.mkdir(parents=True, exist_ok=True)
        availability_start = evidence._ALPHA_MAX_RAW_AVAILABILITY_START_BY_SYMBOL[symbol]
        availability_end = evidence._ALPHA_MAX_RAW_AVAILABILITY_END_BY_SYMBOL[symbol]
        for partition_start in months:
            partition_end = (
                partition_start.replace(year=partition_start.year + 1, month=1)
                if partition_start.month == 12
                else partition_start.replace(month=partition_start.month + 1)
            )
            owned_start = max(start, availability_start, partition_start)
            owned_end = min(end, availability_end, partition_end)
            if owned_start >= owned_end:
                continue
            timestamps = pl.datetime_range(
                owned_start,
                owned_end - timedelta(seconds=1),
                interval="1s",
                eager=True,
            )
            pl.DataFrame({"datetime": timestamps}).with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit("binance").alias("exchange"),
                pl.lit(100.0).alias("open"),
                pl.lit(101.0).alias("high"),
                pl.lit(99.0).alias("low"),
                pl.lit(100.0).alias("close"),
                pl.lit(1.0).alias("volume"),
            ).write_parquet(directory / f"{partition_start:%Y-%m}.parquet")


def _write_feature_root(root: Path, root_id: str) -> None:
    start, end = evidence._ROOT_INTERVALS[root_id]
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        day_end = day + timedelta(days=1)
        for symbol in evidence.ALPHA_MAX_CANDIDATE_SYMBOLS:
            owned_start = max(
                start,
                evidence._ALPHA_MAX_FEATURE_AVAILABILITY_START_BY_SYMBOL[symbol],
                day,
            )
            owned_end = min(
                end,
                evidence._ALPHA_MAX_FEATURE_AVAILABILITY_END_BY_SYMBOL[symbol],
                day_end,
            )
            if owned_start >= owned_end:
                continue
            directory = (
                root
                / "feature_points"
                / "exchange=binance"
                / f"symbol={symbol}"
                / f"date={day:%Y-%m-%d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            timestamps = [
                int(boundary.timestamp() * 1000)
                for hour in ((0, 4, 8, 12, 16, 20) if symbol == "TONUSDT" else (0, 8, 16))
                for boundary in (day + timedelta(hours=hour),)
                if owned_start <= boundary < owned_end
            ]
            if not timestamps:
                raise AssertionError("listing-aware feature fixture has no owned boundary")
            pl.DataFrame(
                {
                    "timestamp_ms": timestamps,
                    "source_timestamp_ms": [value + 500 for value in timestamps],
                    "funding_rate": [0.0001, -0.0002, 0.0003][: len(timestamps)],
                    "symbol": [symbol] * len(timestamps),
                    "exchange": ["binance"] * len(timestamps),
                }
            ).write_parquet(directory / "part-0.parquet")
        day = day_end


def _snapshot_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _snapshot_modes(root: Path) -> dict[str, int]:
    values = {".": stat.S_IMODE(root.stat().st_mode)}
    values.update(
        {
            path.relative_to(root).as_posix(): stat.S_IMODE(path.stat().st_mode)
            for path in sorted(root.rglob("*"))
        }
    )
    return values


def _make_writable(root: Path) -> None:
    os.chmod(root, 0o755)
    for path in root.rglob("*"):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def _mutable_clone(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    _make_writable(target)
    return target


def _restore_file(path: Path, payload: bytes, mode: int, times_ns: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.chmod(path, mode)
    os.utime(path, ns=times_ns)


class _CapsuleReceiptStub:
    def __init__(self, prefix_id: str, row_id: str) -> None:
        self.prefix_id = prefix_id
        self.sha256 = _sha256(f"capsule:{row_id}:{prefix_id}".encode())
        self._payload = {
            "prefix_id": prefix_id,
            "row_id": row_id,
            "sha256": self.sha256,
        }

    def to_payload(self) -> dict[str, object]:
        return dict(self._payload)


class AlphaMaxCliProcessHarness:
    def __init__(self, temp_root: Path) -> None:
        evidence._ROOT_INTERVALS.clear()
        evidence._ROOT_INTERVALS.update(_FIXTURE_ROOT_INTERVALS)
        self.temp_root = temp_root.resolve()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.roots: dict[tuple[str, str], Path] = {}
        self.root_seal_cache: dict[tuple[str, str, str], runner.AlphaMaxRootSeal] = {}
        self.force_root_paths: set[str] = set()
        self.admission_cache: dict[
            tuple[str, str, str, str], evidence.AlphaMaxAdmissionComputation
        ] = {}
        self.admission_override: evidence.AlphaMaxAdmissionComputation | None = None
        self.validation_mode = "no_champion"
        self.historical_mode = "all_reject"
        self.run_label = "setup"
        self.replay_calls: list[tuple[str, str, str, int, str]] = []
        self.matrix_invocations: list[tuple[str, str]] = []
        self.constructor_rows: list[tuple[str, int]] = []
        self.real_root_validation = runner._alpha_max_root_validation
        self.real_complete_matrix = runner._alpha_max_complete_domain_matrix
        self.real_backtest = runner.Backtest
        self._build_roots()
        self._install_replay_stubs()
        self.prelock_cli = _load("alpha_max_process_fixture_prelock", PRELOCK_PATH)
        self.historical_cli = _load("alpha_max_process_fixture_historical", HISTORICAL_PATH)

    def _build_roots(self) -> None:
        root_parent = self.temp_root / "owned-roots"
        root_parent.mkdir()
        for root_id in ROOT_IDS:
            raw = (root_parent / f"{root_id}-raw").resolve()
            feature = (root_parent / f"{root_id}-feature").resolve()
            _write_raw_root(raw, root_id)
            _write_feature_root(feature, root_id)
            self.roots[(root_id, "raw")] = raw
            self.roots[(root_id, "feature")] = feature

    def _install_replay_stubs(self) -> None:
        real_root_validation = self.real_root_validation

        def cached_root_validation(
            roots: tuple[tuple[str, str, str], ...],
            *,
            exchange: str,
            availability_start_by_kind: Mapping[str, Mapping[str, datetime]] | None = None,
            availability_end_by_kind: Mapping[str, Mapping[str, datetime]] | None = None,
        ) -> tuple[dict[tuple[str, str], runner.AlphaMaxRootSeal], tuple[str, ...]]:
            resolved: dict[tuple[str, str], runner.AlphaMaxRootSeal] = {}
            failures: list[str] = []
            missing: list[tuple[str, str, str]] = []
            for entry in roots:
                key = (entry[0], entry[1], entry[2])
                if entry[2] not in self.force_root_paths and key in self.root_seal_cache:
                    resolved[(entry[0], entry[1])] = self.root_seal_cache[key]
                else:
                    missing.append(entry)
            if missing:
                observed, observed_failures = real_root_validation(
                    tuple(missing),
                    exchange=exchange,
                    availability_start_by_kind=availability_start_by_kind,
                    availability_end_by_kind=availability_end_by_kind,
                )
                resolved.update(observed)
                failures.extend(observed_failures)
                for root_id, root_kind, root_path in missing:
                    seal = observed.get((root_id, root_kind))
                    if seal is not None and root_path not in self.force_root_paths:
                        self.root_seal_cache[(root_id, root_kind, root_path)] = seal
            return resolved, tuple(failures)

        runner._alpha_max_root_validation = cached_root_validation

        def admission_stub(
            *,
            warmup_raw: runner.AlphaMaxRootSeal,
            warmup_feature: runner.AlphaMaxRootSeal,
            train_raw: runner.AlphaMaxRootSeal,
            train_feature: runner.AlphaMaxRootSeal,
        ) -> evidence.AlphaMaxAdmissionComputation:
            if self.admission_override is not None:
                return self.admission_override
            key = (
                warmup_raw.sha256,
                warmup_feature.sha256,
                train_raw.sha256,
                train_feature.sha256,
            )
            cached = self.admission_cache.get(key)
            if cached is not None:
                return cached
            with _root_interval_scope(_PRODUCTION_ROOT_INTERVALS):
                train_start = evidence._ROOT_INTERVALS["train"][0]
                daily = tuple(
                    evidence.AlphaMaxDailyQuoteNotional(
                        day=(train_start + timedelta(days=index)).date(),
                        quote_notional_usdt=100_000_000.0,
                        completed_4h_bucket_hours=(0, 4, 8, 12, 16, 20),
                    )
                    for index in range(517)
                )
                inputs = {
                    symbol: evidence.AlphaMaxAdmissionDailyCandidateInput(
                        symbol=symbol,
                        daily_quote_notional=(
                            daily
                            if index < 5
                            else tuple(
                                evidence.AlphaMaxDailyQuoteNotional(
                                    day=value.day,
                                    quote_notional_usdt=1.0,
                                    completed_4h_bucket_hours=value.completed_4h_bucket_hours,
                                )
                                for value in daily
                            )
                        ),
                        consecutive_completed_daily_bars_before_train=366,
                        causal_funding_coverage_complete=True,
                        unresolved_daily_cross_section_count=0,
                        partition_integrity_complete=True,
                    )
                    for index, symbol in enumerate(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS)
                }
                computed = evidence.compute_alpha_max_train_admission_from_daily_summaries(
                    inputs,
                    input_root_hashes={
                        "warmup": _sha256(
                            _canonical_bytes(
                                {"feature": warmup_feature.sha256, "raw": warmup_raw.sha256}
                            )
                        ),
                        "train": _sha256(
                            _canonical_bytes(
                                {"feature": train_feature.sha256, "raw": train_raw.sha256}
                            )
                        ),
                    },
                )
            self.admission_cache[key] = computed
            return computed

        runner._compute_alpha_max_admission_from_seals = admission_stub

        def training_stub(
            _preflight,
            *,
            manifest_receipt: evidence.AlphaMaxManifestReceipt,
            **_kwargs,
        ) -> tuple[tuple[str, ...], tuple[float, ...], evidence.AlphaMaxNativeFinalizationReceipt]:
            calendar = tuple(
                (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index)).date().isoformat()
                for index in range(300)
            )
            component_index = (
                "component_carry_1x",
                "component_near_high_1x",
                "component_trend_1x",
            ).index(manifest_receipt.row_id) + 1
            values = tuple(
                0.0005
                + component_index * 0.0001
                + math.sin((index + 1) * (component_index + 1)) * 0.0002
                for index in range(len(calendar))
            )
            completed_native_keys = [["BTCUSDT", "2025-05-31"]]
            native_coverage = {
                "adapter_class": "StubNativeAdapter",
                "native_timeframe": "1d",
                "barrier_mode": "none",
                "completed_native_keys": completed_native_keys,
                "completed_native_count_by_symbol": {"BTCUSDT": 1},
                "last_completed_native_key_by_symbol": {"BTCUSDT": "2025-05-31"},
                "barrier_pending_keys": [],
                "barrier_closed_keys": [],
                "barrier_symbol_coverage": {},
                "failed_native_keys": {},
                "partial_bucket_error": None,
                "finalization_completed_native_keys": completed_native_keys,
                "finalization_barrier_keys": [],
            }
            native = evidence.build_alpha_max_native_finalization_receipt(
                boundary_utc=datetime(2025, 6, 1, tzinfo=UTC),
                finalized_children={manifest_receipt.row_id: 1},
                native_coverage_by_child={manifest_receipt.row_id: native_coverage},
                discarded_signal_count=0,
                discarded_signal_sha256=_sha256(b""),
            )
            return calendar, values, native

        runner._alpha_max_replay_training_component_returns = training_stub

        def capsule_for(
            *, manifest_sha256: str, phase_id: str, row_id: str
        ) -> runner.AlphaMaxIndicatorCapsule:
            scope = {"phase_id": phase_id, "row_id": row_id, "stubbed_replay_data": True}
            capsule_sha = _sha256(_canonical_bytes(scope))
            capsule = {**scope, "sha256": capsule_sha}
            return runner.AlphaMaxIndicatorCapsule(
                portfolio_mode="alpha_max_process_fixture",
                phase_id=phase_id,
                manifest_sha256=manifest_sha256,
                capsule_sha256=capsule_sha,
                capsule=MappingProxyType(capsule),
                finalized_children=MappingProxyType({row_id: 1}),
                native_finalization_sha256=_sha256(f"native:{row_id}:{phase_id}".encode()),
                windows_processed=1,
                discarded_signal_count=0,
            )

        def indicator_prefix_stub(
            preflight,
            *,
            manifest_output_root: Path,
            phase: str,
            manifest_receipt: evidence.AlphaMaxManifestReceipt,
            admitted_symbols: tuple[str, ...],
            phase_ids: tuple[str, ...],
            **_kwargs,
        ) -> runner.AlphaMaxIndicatorCapsule:
            try:
                runner.seal_alpha_max_manifest_activation(
                    preflight,
                    output_root=str(manifest_output_root),
                    phase=phase,
                    manifest_path=manifest_receipt.path,
                    admitted_symbols=admitted_symbols,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"indicator_prefix_manifest_activation:{manifest_receipt.row_id}:{phase}"
                ) from exc
            return capsule_for(
                manifest_sha256=manifest_receipt.sha256,
                phase_id=phase_ids[-1],
                row_id=manifest_receipt.row_id,
            )

        runner._alpha_max_build_indicator_prefix = indicator_prefix_stub

        def fold_inputs_stub(
            preflight,
            *,
            manifest_output_root: Path,
            phase: str,
            manifest_receipt: evidence.AlphaMaxManifestReceipt,
            admitted_symbols: tuple[str, ...],
            domain: str,
            **_kwargs,
        ) -> tuple[SimpleNamespace, ...]:
            try:
                runner.seal_alpha_max_manifest_activation(
                    preflight,
                    output_root=str(manifest_output_root),
                    phase=phase,
                    manifest_path=manifest_receipt.path,
                    admitted_symbols=admitted_symbols,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"fold_input_manifest_activation:{manifest_receipt.row_id}:{phase}"
                ) from exc
            return tuple(
                SimpleNamespace(
                    fold_id=fold_id,
                    capsule_receipt=_CapsuleReceiptStub(fold_id, manifest_receipt.row_id),
                )
                for fold_id in runner._alpha_max_fold_ids(domain)
            )

        runner._alpha_max_build_fold_inputs = fold_inputs_stub

        def primary_stream(row_id: str) -> evidence.AlphaMaxPrimaryReturnStream:
            value = object.__new__(evidence.AlphaMaxPrimaryReturnStream)
            object.__setattr__(value, "endpoint_timestamps", ())
            object.__setattr__(value, "endpoint_equities", ())
            object.__setattr__(value, "returns", ())
            object.__setattr__(value, "initial_capital", 10_000.0)
            object.__setattr__(value, "periods_per_year", 2190)
            object.__setattr__(value, "calendar_sha256", _sha256(row_id.encode()))
            return value

        def replay_stub(
            _preflight,
            *,
            manifest_receipt: evidence.AlphaMaxManifestReceipt,
            row_id: str,
            domain: str,
            nominal_cost_bps: int,
            fold_inputs: tuple[SimpleNamespace, ...],
            **_kwargs,
        ) -> evidence.AlphaMaxCostCellPreGateEvidence:
            folds: list[SimpleNamespace] = []
            for fold_input in fold_inputs:
                self.replay_calls.append(
                    (self.run_label, domain, row_id, nominal_cost_bps, fold_input.fold_id)
                )
                actual = SimpleNamespace(
                    capsule_receipt=fold_input.capsule_receipt,
                    feature_root_receipts=(),
                    full_event_equity=SimpleNamespace(full_event_mdd=0.20),
                    manifest_receipt=manifest_receipt,
                    raw_root_receipts=(),
                    report_only_diagnostics=SimpleNamespace(
                        symbol_contribution_usdt=dict.fromkeys(
                            evidence.ALPHA_MAX_CANDIDATE_SYMBOLS[:5], 1.0
                        )
                    ),
                )
                folds.append(
                    SimpleNamespace(
                        actual_engine_run=actual,
                        sha256=_sha256(
                            f"{domain}:{row_id}:{nominal_cost_bps}:{fold_input.fold_id}".encode()
                        ),
                        split_or_fold_id=fold_input.fold_id,
                    )
                )
            pre_gate = object.__new__(evidence.AlphaMaxCostCellPreGateEvidence)
            values: dict[str, object] = {
                "row_id": row_id,
                "domain": domain,
                "nominal_cost_bps": nominal_cost_bps,
                "status": "complete",
                "fold_runs": tuple(folds),
                "fold_run_set_sha256": _sha256(
                    f"fold-set:{domain}:{row_id}:{nominal_cost_bps}".encode()
                ),
                "source_return_stream_set_sha256": _sha256(
                    f"stream-set:{domain}:{row_id}:{nominal_cost_bps}".encode()
                ),
                "combined_primary_return_stream": (
                    primary_stream(row_id)
                    if nominal_cost_bps == 20
                    and row_id
                    in {
                        "component_carry_1x",
                        "component_near_high_1x",
                        "component_trend_1x",
                    }
                    else None
                ),
                "combined_streaming_equity": None,
                "metric_statistics": (
                    SimpleNamespace(gate_mdd=0.20) if nominal_cost_bps == 30 else None
                ),
                "canonical_bytes": b"stubbed-replay-data\n",
                "sha256": _sha256(f"pre-gate:{domain}:{row_id}:{nominal_cost_bps}".encode()),
            }
            for key, value in values.items():
                object.__setattr__(pre_gate, key, value)
            return pre_gate

        runner._replay_alpha_max_cost_cell_pre_gate = replay_stub

        def gate_for(
            pre_gate: evidence.AlphaMaxCostCellPreGateEvidence,
        ) -> evidence.AlphaMaxGateInput:
            mode = self.validation_mode if pre_gate.domain == "validation" else self.historical_mode
            role = "prelock_selection" if pre_gate.domain == "validation" else "historical_report"
            total = 0.20
            dsr = 0.96
            if mode in {"no_champion", "all_reject"}:
                dsr = 0.10
            elif mode == "champion":
                total = 0.80 if pre_gate.row_id == "component_carry_1x" else 0.20
            elif mode in {"champion_pass_disagreement", "champion_fail_disagreement"}:
                total = 0.90 if pre_gate.row_id == "component_near_high_1x" else 0.25
                if pre_gate.row_id == "component_carry_1x":
                    total = 0.60
                    if mode == "champion_fail_disagreement":
                        dsr = 0.10
            return evidence.AlphaMaxGateInput(
                row_id=pre_gate.row_id,
                comparison_role=role,
                evidence_tier="actual_engine",
                comparison_valid=True,
                nominal_cost_bps=30,
                cumulative_return=total,
                cagr=max(0.01, total / 2.0),
                calmar=max(0.01, total / 0.20),
                net_sharpe=max(0.01, total * 2.0),
                full_event_mdd=0.20,
                reporting_4h_mdd=0.20,
                dsr=dsr,
                spa_pvalue=0.01,
                pbo=0.10,
                native_data_coverage_complete=True,
                funding_coverage_complete=True,
                hash_valid=True,
                manifest_valid=True,
                reconciliation_complete=True,
                ruin=False,
                raw_root_set_sha256="a" * 64,
                feature_root_set_sha256="b" * 64,
                universe_sha256="c" * 64,
                calendar_sha256="d" * 64,
                seed_schedule_sha256=evidence.alpha_max_seed_schedule_sha256(pre_gate.domain),
            )

        def cell_stub(
            pre_gate: evidence.AlphaMaxCostCellPreGateEvidence,
            *,
            statistical_evidence=None,
        ) -> evidence.AlphaMaxCostCellEvidence:
            if statistical_evidence is not None:
                raise AssertionError("statistical replay data must remain stubbed")
            cell = object.__new__(evidence.AlphaMaxCostCellEvidence)
            values = {
                "row_id": pre_gate.row_id,
                "domain": pre_gate.domain,
                "nominal_cost_bps": pre_gate.nominal_cost_bps,
                "status": "complete",
                "evidence_tier": "actual_engine",
                "selection_valid": True,
                "pre_gate_evidence": pre_gate,
                "gate_input": gate_for(pre_gate) if pre_gate.nominal_cost_bps == 30 else None,
                "terminal_gate_evidence": None,
            }
            for key, value in values.items():
                object.__setattr__(cell, key, value)
            return cell

        runner.build_alpha_max_cost_cell_evidence = cell_stub
        runner.canonical_alpha_max_cost_cell_bytes = lambda cell: (
            _canonical_bytes(
                {
                    "artifact_kind": "alpha_max_cost_cell_fixture_evidence.v1",
                    "domain": cell.domain,
                    "gate_input": (
                        None if cell.gate_input is None else cell.gate_input.to_payload()
                    ),
                    "nominal_cost_bps": cell.nominal_cost_bps,
                    "row_id": cell.row_id,
                    "stubbed_expensive_replay_data": True,
                }
            )
            + b"\n"
        )

        def matrix_artifacts(matrix: runner._AlphaMaxCompletedMatrix) -> dict[str, bytes]:
            domain_path = (
                "validation" if matrix.domain == "validation" else "historical_exposed_evaluation"
            )
            artifacts = {"status/matrix.json": matrix.status_payload}
            for row in matrix.rows:
                artifacts[f"evidence/{domain_path}/rows/{row.row_id}.json"] = (
                    _canonical_bytes(
                        {
                            "artifact_kind": "alpha_max_row_fixture_evidence.v1",
                            "row_id": row.row_id,
                            "stubbed_expensive_replay_data": True,
                        }
                    )
                    + b"\n"
                )
                for cell in row.cost_cells:
                    artifacts[
                        f"evidence/{domain_path}/cells/{row.row_id}/{cell.nominal_cost_bps}.json"
                    ] = runner.canonical_alpha_max_cost_cell_bytes(cell)
            return artifacts

        runner._alpha_max_matrix_artifacts = matrix_artifacts
        runner._alpha_max_trend_liquidity_falsifier_artifact = lambda matrix, _buckets: (
            _canonical_bytes(
                {
                    "artifact_kind": "alpha_max_trend_liquidity_fixture.v1",
                    "domain": matrix.domain,
                    "report_only": True,
                    "stubbed_expensive_replay_data": True,
                }
            )
            + b"\n"
        )
        validation_stream_ids = {
            _sha256(row_id.encode()): index
            for index, row_id in enumerate(
                (
                    "component_carry_1x",
                    "component_near_high_1x",
                    "component_trend_1x",
                ),
                1,
            )
        }

        def daily_returns_stub(
            stream: evidence.AlphaMaxPrimaryReturnStream,
        ) -> tuple[tuple[str, ...], tuple[float, ...]]:
            component_index = validation_stream_ids[stream.calendar_sha256]
            return (
                tuple(
                    (datetime(2025, 6, 2, tzinfo=UTC) + timedelta(days=index)).date().isoformat()
                    for index in range(84)
                ),
                tuple(
                    0.0005
                    + component_index * 0.0001
                    + math.sin((300 + index + 1) * (component_index + 1)) * 0.0002
                    for index in range(84)
                ),
            )

        runner._alpha_max_daily_returns_from_primary_stream = daily_returns_stub

        def complete_matrix_wrapper(*args, **kwargs):
            domain = kwargs["domain"]
            self.matrix_invocations.append((self.run_label, domain))
            return self.real_complete_matrix(*args, **kwargs)

        runner._alpha_max_complete_domain_matrix = complete_matrix_wrapper

    def prelock_argv(
        self,
        output: Path,
        *,
        config: Path = CONFIG_PATH,
        contract: Path = CONTRACT_PATH,
        root_overrides: dict[tuple[str, str], Path] | None = None,
    ) -> list[str]:
        roots = dict(self.roots)
        roots.update(root_overrides or {})
        argv = [
            "--config",
            str(config.resolve()),
            "--contract-manifest",
            str(contract.resolve()),
            "--exchange",
            "binance",
            "--output-root",
            str(output.resolve()),
        ]
        for root_id in PRELOCK_ROOT_IDS:
            for kind in ("raw", "feature"):
                argv.extend((f"--{root_id}-{kind}-root", str(roots[(root_id, kind)].resolve())))
        return argv

    def historical_argv(
        self,
        prelock: Path,
        output: Path,
        *,
        root_overrides: dict[tuple[str, str], Path] | None = None,
    ) -> list[str]:
        roots = dict(self.roots)
        roots.update(root_overrides or {})
        return [
            "--sealed-prelock-directory",
            str(prelock.resolve()),
            "--embargo-feature-root",
            str(roots[("embargo", "feature")].resolve()),
            "--historical-evaluation-raw-root",
            str(roots[("historical_exposed_evaluation", "raw")].resolve()),
            "--historical-evaluation-feature-root",
            str(roots[("historical_exposed_evaluation", "feature")].resolve()),
            "--exchange",
            "binance",
            "--output-root",
            str(output.resolve()),
        ]

    def run_prelock(
        self,
        output: Path,
        *,
        label: str,
        mode: str,
        config: Path = CONFIG_PATH,
        contract: Path = CONTRACT_PATH,
        root_overrides: dict[tuple[str, str], Path] | None = None,
    ) -> int:
        self.run_label = label
        self.validation_mode = mode
        return self.prelock_cli.main(
            self.prelock_argv(
                output,
                config=config,
                contract=contract,
                root_overrides=root_overrides,
            )
        )

    def run_historical(
        self,
        prelock: Path,
        output: Path,
        *,
        label: str,
        mode: str,
        root_overrides: dict[tuple[str, str], Path] | None = None,
    ) -> int:
        self.run_label = label
        self.historical_mode = mode
        return self.historical_cli.main(
            self.historical_argv(prelock, output, root_overrides=root_overrides)
        )

    def expect_error(self, operation: Callable[[], object]) -> str:
        try:
            operation()
        except Exception as exc:
            return f"{type(exc).__name__}:{exc}"
        raise AssertionError("expected_alpha_max_boundary_error")

    def cleanup_bundle(self, root: Path) -> None:
        runner._cleanup_partial_bundle(root)

    def reseal_prelock(self, root: Path) -> None:
        seal_path = root / "SEALED.json"
        seal_path.unlink(missing_ok=True)
        artifacts = _snapshot_bytes(root)
        run_payload = json.loads(artifacts["run/prelock_result.json"])
        seal = evidence.build_alpha_max_prelock_seal(
            artifacts,
            prelock_champion=run_payload["prelock_champion"],
            selected_candidate_id=run_payload["selected_candidate_id"],
        )
        seal_path.write_bytes(seal.canonical_bytes)
        runner._make_bundle_immutable(root)

    def _rebind_first_capsule(self, root: Path, row_id: str, manifest_sha256: str) -> None:
        relative = (
            f"capsules/prelock_final_refit/{row_id}/{runner._ALPHA_MAX_HISTORICAL_FOLD_IDS[0]}.json"
        )
        path = root / relative
        envelope = json.loads(path.read_bytes())
        state = dict(envelope["state_payload"])
        state["manifest_sha256"] = manifest_sha256
        path.write_bytes(
            evidence.AlphaMaxCapsuleReceipt.canonical_envelope_bytes(
                row_id=row_id,
                phase="prelock_final_refit",
                prefix_id=runner._ALPHA_MAX_HISTORICAL_FOLD_IDS[0],
                manifest_sha256=manifest_sha256,
                state_payload=state,
            )
        )


def _bundle_inventory_is_exact(root: Path, *, seal_key: str) -> bool:
    seal = json.loads((root / "SEALED.json").read_bytes())
    inventory = seal[seal_key]
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SEALED.json"
    }
    if {entry["relative_path"] for entry in inventory} != expected:
        return False
    return all(
        len(payload := (root / entry["relative_path"]).read_bytes()) == entry["byte_count"]
        and _sha256(payload) == entry["sha256"]
        for entry in inventory
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise AssertionError(f"json_object_required:{path}")
    return value


def _mutate_feature_values(path: Path, delta: float) -> None:
    frame = pl.read_parquet(path)
    frame.with_columns((pl.col("funding_rate") + delta).alias("funding_rate")).write_parquet(path)


def run_alpha_max_cli_process_fixture(temp_root: Path) -> dict[str, object]:
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            del os.environ[key]
    harness = AlphaMaxCliProcessHarness(temp_root)
    payload: dict[str, object] = {}

    # P12/P08/P06 baseline: actual roots, manifests, capsules, selection, terminal,
    # immutable bundle, and production matrix control flow. Only expensive data
    # replay and its bulky evidence encoding are deterministic stubs.
    prelock_no = (harness.temp_root / "prelock-no-champion").resolve()
    assert (
        harness.run_prelock(
            prelock_no,
            label="prelock-no-champion",
            mode="no_champion",
        )
        == 0
    )
    no_run = _read_json(prelock_no / "run/prelock_result.json")
    no_terminal = _read_json(prelock_no / "terminal/prelock.json")
    no_selection = _read_json(prelock_no / "selection/prelock.json")
    no_ledger = _read_json(prelock_no / "trial/ledger.json")
    matrix = _read_json(prelock_no / "status/matrix.json")
    statuses = matrix["statuses"]
    inventory = _read_json(prelock_no / "SEALED.json")["artifacts"]
    inventory_by_path = {entry["relative_path"]: entry for entry in inventory}
    capsule_paths = tuple(
        sorted(
            f"capsules/prelock_final_refit/{row_id}/{runner._ALPHA_MAX_HISTORICAL_FOLD_IDS[0]}.json"
            for row_id in runner._ALPHA_MAX_RESOLVABLE_ROWS
        )
    )
    capsule_snapshot = {
        relative: (prelock_no / relative).read_bytes() for relative in capsule_paths
    }
    payload["p06"] = {
        "capsule_count": len(capsule_paths),
        "all_inventory_bound": all(
            relative in inventory_by_path
            and inventory_by_path[relative]["byte_count"] == len(capsule_snapshot[relative])
            and inventory_by_path[relative]["sha256"] == _sha256(capsule_snapshot[relative])
            for relative in capsule_paths
        ),
        "all_before_historical": all(
            (prelock_no / relative).is_file() for relative in capsule_paths
        ),
    }
    payload["p08"] = {
        "status_count": matrix["status_count"],
        "engine_cell_count": matrix["engine_cell_count"],
        "physical_fold_run_count": matrix["physical_fold_run_count"],
        "row_ids": sorted({row["row_id"] for row in statuses}),
        "unavailable": sorted(
            {row["row_id"] for row in statuses if row["status"] == "incumbent_replay_unavailable"}
        ),
        "diagnostic": sorted(
            {row["row_id"] for row in statuses if row["status"] == "diagnostic_report_only"}
        ),
        "sealed_inventory_exact": _bundle_inventory_is_exact(prelock_no, seal_key="artifacts"),
    }
    payload["p12"] = {
        "champion": no_run["prelock_champion"],
        "selected": no_run["selected_candidate_id"],
        "terminal": no_run["terminal_outcome"],
        "terminal_artifact": no_terminal["terminal_outcome"],
        "ranked": no_selection["ranked_candidate_ids"],
        "num_trials": no_ledger["num_trials"],
        "complete_matrix": matrix["engine_cell_count"] == 68
        and matrix["physical_fold_run_count"] == 816,
    }

    # P04 prelock overwrite is refused and a successful historical run proves
    # the production before/after snapshot invariant.
    before_historical = _snapshot_bytes(prelock_no)
    overwrite_matrix_count = len(harness.matrix_invocations)
    prelock_overwrite_error = harness.expect_error(
        lambda: harness.run_prelock(
            prelock_no,
            label="prelock-overwrite",
            mode="no_champion",
        )
    )
    overwrite_before_replay = len(harness.matrix_invocations) == overwrite_matrix_count
    no_history_parent = (harness.temp_root / "historical-no-parent").resolve()
    no_history_parent.mkdir()
    historical_no = no_history_parent / "result"
    assert (
        harness.run_historical(
            prelock_no,
            historical_no,
            label="historical-no-champion",
            mode="all_reject",
        )
        == 0
    )
    payload["p04"] = {
        "overwrite_error": prelock_overwrite_error,
        "overwrite_before_replay": overwrite_before_replay,
        "prelock_bytes_unchanged": _snapshot_bytes(prelock_no) == before_historical,
        "prelock_modes_unchanged": all(
            mode == (0o555 if relative == "." or not (prelock_no / relative).is_file() else 0o444)
            for relative, mode in _snapshot_modes(prelock_no).items()
        ),
    }

    # P09 append-only completion claim and immutable exact inventory.
    duplicate_output = no_history_parent / "duplicate"
    overwrite_history_error = harness.expect_error(
        lambda: harness.run_historical(
            prelock_no,
            historical_no,
            label="historical-overwrite",
            mode="all_reject",
        )
    )
    duplicate_error = harness.expect_error(
        lambda: harness.run_historical(
            prelock_no,
            duplicate_output,
            label="historical-duplicate-completion",
            mode="all_reject",
        )
    )
    payload["p09"] = {
        "overwrite_error": overwrite_history_error,
        "duplicate_error": duplicate_error,
        "duplicate_output_absent": not duplicate_output.exists(),
        "immutable": stat.S_IMODE(historical_no.stat().st_mode) == 0o555
        and all(
            stat.S_IMODE(path.stat().st_mode) == 0o444
            for path in historical_no.rglob("*")
            if path.is_file()
        ),
        "inventory_exact": _bundle_inventory_is_exact(
            historical_no, seal_key="historical_artifacts"
        ),
    }

    # P03: every listed poison operation targets only the later historical raw/
    # feature trees. The same public prelock target is removed and recreated so
    # every byte (including absolute-path-bound manifests) is comparable.
    p03_output = (harness.temp_root / "p03-prelock-stable").resolve()
    assert (
        harness.run_prelock(
            p03_output,
            label="p03-reference",
            mode="no_champion",
        )
        == 0
    )
    p03_reference = _snapshot_bytes(p03_output)
    p03_capsules = {relative: p03_reference[relative] for relative in capsule_paths}
    p03_results: dict[str, bool] = {}
    historical_raw_file = next(
        path for path in harness.roots[("historical_exposed_evaluation", "raw")].rglob("*.parquet")
    )
    historical_feature_file = next(
        path
        for path in harness.roots[("historical_exposed_evaluation", "feature")].rglob("*.parquet")
    )

    def run_p03(name: str, mutate: Callable[[], Callable[[], None]]) -> None:
        harness.cleanup_bundle(p03_output)
        restore = mutate()
        try:
            assert (
                harness.run_prelock(
                    p03_output,
                    label=f"p03-{name}",
                    mode="no_champion",
                )
                == 0
            )
            observed = _snapshot_bytes(p03_output)
            p03_results[name] = observed == p03_reference and all(
                observed[relative] == p03_capsules[relative] for relative in capsule_paths
            )
        finally:
            restore()

    def add_poison() -> Callable[[], None]:
        path = harness.roots[("historical_exposed_evaluation", "raw")] / "attacker-added.txt"
        path.write_text("poison", encoding="utf-8")
        return lambda: path.unlink()

    def remove_poison() -> Callable[[], None]:
        path = historical_raw_file
        status = path.stat()
        data = path.read_bytes()
        path.unlink()
        return lambda: _restore_file(
            path,
            data,
            stat.S_IMODE(status.st_mode),
            (status.st_atime_ns, status.st_mtime_ns),
        )

    def rename_poison() -> Callable[[], None]:
        renamed = historical_feature_file.with_name("attacker-renamed.parquet")
        historical_feature_file.rename(renamed)
        return lambda: renamed.rename(historical_feature_file)

    def chmod_poison() -> Callable[[], None]:
        path = historical_feature_file
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, 0o600)
        return lambda: os.chmod(path, mode)

    def touch_poison() -> Callable[[], None]:
        path = historical_feature_file
        status = path.stat()
        os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 2_000_000_000))
        return lambda: os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))

    def content_poison() -> Callable[[], None]:
        path = historical_feature_file
        status = path.stat()
        data = path.read_bytes()
        _mutate_feature_values(path, 0.001)
        return lambda: _restore_file(
            path,
            data,
            stat.S_IMODE(status.st_mode),
            (status.st_atime_ns, status.st_mtime_ns),
        )

    for poison_name, poison in (
        ("add", add_poison),
        ("remove", remove_poison),
        ("rename", rename_poison),
        ("chmod", chmod_poison),
        ("touch", touch_poison),
        ("content", content_poison),
    ):
        run_p03(poison_name, poison)
    payload["p03"] = p03_results
    payload["p06"]["inventory_independent"] = all(p03_results.values())  # type: ignore[index]

    # Champion prelock drives disagreement and terminal-collision cases.
    prelock_champion = (harness.temp_root / "prelock-champion").resolve()
    assert (
        harness.run_prelock(
            prelock_champion,
            label="prelock-champion",
            mode="champion",
        )
        == 0
    )
    champion_run = _read_json(prelock_champion / "run/prelock_result.json")
    champion_id = champion_run["prelock_champion"]
    assert champion_id == "component_carry_1x"
    champion_snapshot = _snapshot_bytes(prelock_champion)

    def run_collision(name: str, prelock: Path, mode: str) -> tuple[Path, dict[str, Any]]:
        parent = (harness.temp_root / f"historical-{name}-parent").resolve()
        parent.mkdir()
        output = parent / "result"
        assert (
            harness.run_historical(
                prelock,
                output,
                label=f"historical-{name}",
                mode=mode,
            )
            == 0
        )
        return output, _read_json(output / "terminal/historical.json")

    historical_pass, terminal_pass = run_collision(
        "pass-disagreement",
        prelock_champion,
        "champion_pass_disagreement",
    )
    historical_fail, terminal_fail = run_collision(
        "fail-disagreement",
        prelock_champion,
        "champion_fail_disagreement",
    )
    terminal_no = _read_json(historical_no / "terminal/historical.json")
    pass_report = _read_json(historical_pass / "report/historical_result.json")
    fail_report = _read_json(historical_fail / "report/historical_result.json")
    payload["p13"] = {
        "prelock_champion": champion_id,
        "historical_leader": terminal_pass["historical_evaluation_leader"],
        "selected_candidate_id": terminal_pass["selected_candidate_id"],
        "leader_differs": terminal_pass["leader_differs_from_prelock_champion"],
        "requires_fresh_confirmation": terminal_pass["requires_fresh_confirmation"],
        "report_selected_candidate_id": pass_report["selected_candidate_id"],
    }
    payload["p16"] = {
        "pass": terminal_pass,
        "fail": terminal_fail,
        "no_survivor": terminal_no,
        "pass_report": pass_report,
        "fail_report": fail_report,
    }

    # P10: actual re-seal of changed historical values, with immutable prelock
    # bytes and selection identity unchanged. Only the historical package owns
    # the new root seal/report fields.
    poison_path = historical_feature_file
    poison_status = poison_path.stat()
    poison_bytes = poison_path.read_bytes()
    harness.force_root_paths.add(str(harness.roots[("historical_exposed_evaluation", "feature")]))
    try:
        _mutate_feature_values(poison_path, 0.002)
        poisoned_history, poisoned_terminal = run_collision(
            "poisoned-values",
            prelock_champion,
            "champion_pass_disagreement",
        )
    finally:
        _restore_file(
            poison_path,
            poison_bytes,
            stat.S_IMODE(poison_status.st_mode),
            (poison_status.st_atime_ns, poison_status.st_mtime_ns),
        )
        harness.force_root_paths.discard(
            str(harness.roots[("historical_exposed_evaluation", "feature")])
        )
    payload["p10"] = {
        "prelock_bytes_identical": _snapshot_bytes(prelock_champion) == champion_snapshot,
        "selected_identity_preserved": poisoned_terminal["selected_candidate_id"] == champion_id,
        "binding_identical": (poisoned_history / "binding/prelock_seal.json").read_bytes()
        == (historical_pass / "binding/prelock_seal.json").read_bytes(),
        "historical_feature_root_changed": (
            poisoned_history / "roots/feature/historical_exposed_evaluation.json"
        ).read_bytes()
        != (historical_pass / "roots/feature/historical_exposed_evaluation.json").read_bytes(),
    }

    # P11: production matrix loops/schedule validation executed exact physical
    # cardinalities and never admitted the three incumbent/one diagnostic ids.
    labels_to_check = {
        "prelock-no-champion": ("validation", 816),
        "historical-no-champion": ("historical_exposed_evaluation", 680),
    }
    p11: dict[str, object] = {}
    for label, (domain, expected_count) in labels_to_check.items():
        calls = [value for value in harness.replay_calls if value[0] == label]
        schedule = {(value[2], value[3], value[4]) for value in calls}
        p11[label] = {
            "domain": domain,
            "count": len(calls),
            "unique": len(schedule),
            "expected": expected_count,
            "forbidden_rows_absent": not any(
                value[2]
                in {*runner._ALPHA_MAX_UNAVAILABLE_ROWS, *runner._ALPHA_MAX_DIAGNOSTIC_ROWS}
                for value in calls
            ),
        }
    payload["p11"] = p11

    # P14 chronology is read from the production preflight/config, while each
    # hostile config/endpoint mutation crosses the public parser/process gate.
    preflight = runner.preflight_alpha_max_runtime_contract(CONFIG_PATH)
    config_payload = _read_json(CONFIG_PATH)
    chronology = config_payload["chronology"]
    validation_start = datetime(2025, 6, 8, tzinfo=UTC)
    validation_fold_windows = tuple(
        (
            f"validation_w{index:02d}",
            (validation_start + timedelta(days=7 * (index - 1))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            (validation_start + timedelta(days=7 * index)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        for index in range(1, 13)
    )
    historical_boundaries = (
        "2025-09-07T00:00:00Z",
        "2025-10-01T00:00:00Z",
        "2025-11-01T00:00:00Z",
        "2025-12-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
        "2026-03-01T00:00:00Z",
        "2026-04-01T00:00:00Z",
        "2026-05-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
        "2026-07-01T00:00:00Z",
    )
    chronology_exact = (
        [(row["split_id"], row["start_utc"], row["end_utc"]) for row in chronology["splits"]]
        == [
            ("warmup", "2022-12-31T00:00:00Z", "2024-01-01T00:00:00Z"),
            ("train", "2024-01-01T00:00:00Z", "2025-06-01T00:00:00Z"),
            ("purge", "2025-06-01T00:00:00Z", "2025-06-08T00:00:00Z"),
            ("validation", "2025-06-08T00:00:00Z", "2025-08-31T00:00:00Z"),
            ("embargo", "2025-08-31T00:00:00Z", "2025-09-07T00:00:00Z"),
            (
                "historical_exposed_evaluation",
                "2025-09-07T00:00:00Z",
                "2026-07-01T00:00:00Z",
            ),
        ]
        and chronology["purge_days"] == chronology["embargo_days"] == 7
        and tuple(row["fold_id"] for row in chronology["validation_folds"])
        == runner._ALPHA_MAX_VALIDATION_FOLD_IDS
        and tuple(
            (row["fold_id"], row["start_utc"], row["end_utc"])
            for row in chronology["validation_folds"]
        )
        == validation_fold_windows
        and tuple(
            (row["regime_id"], row["start_utc"], row["end_utc"])
            for row in chronology["validation_regimes"]
        )
        == (
            ("validation_r01", "2025-06-08T00:00:00Z", "2025-07-06T00:00:00Z"),
            ("validation_r02", "2025-07-06T00:00:00Z", "2025-08-03T00:00:00Z"),
            ("validation_r03", "2025-08-03T00:00:00Z", "2025-08-31T00:00:00Z"),
        )
        and tuple(row["fold_id"] for row in chronology["historical_evaluation_folds"])
        == runner._ALPHA_MAX_HISTORICAL_FOLD_IDS
        and tuple(
            (row["fold_id"], row["start_utc"], row["end_utc"])
            for row in chronology["historical_evaluation_folds"]
        )
        == tuple(
            (fold_id, historical_boundaries[index], historical_boundaries[index + 1])
            for index, fold_id in enumerate(runner._ALPHA_MAX_HISTORICAL_FOLD_IDS)
        )
        and tuple(chronology["historical_evaluation_regimes"])
        == runner._ALPHA_MAX_HISTORICAL_FOLD_IDS
        and tuple(preflight.phase_windows)
        == (
            ROOT_IDS + runner._ALPHA_MAX_VALIDATION_FOLD_IDS + runner._ALPHA_MAX_HISTORICAL_FOLD_IDS
        )
    )
    config_errors: dict[str, str] = {}
    for name, mutate in (
        (
            "overlap",
            lambda value: value["chronology"]["splits"][2].__setitem__(
                "start_utc", "2025-05-31T00:00:00Z"
            ),
        ),
        (
            "shortening",
            lambda value: value["chronology"]["splits"][3].__setitem__(
                "end_utc", "2025-08-30T00:00:00Z"
            ),
        ),
        (
            "shifted_boundary",
            lambda value: value["chronology"]["validation_folds"][0].__setitem__(
                "start_utc", "2025-06-09T00:00:00Z"
            ),
        ),
        (
            "regime",
            lambda value: value["chronology"]["validation_regimes"][0].__setitem__(
                "end_utc", "2025-07-05T00:00:00Z"
            ),
        ),
    ):
        mutated = json.loads(CONFIG_PATH.read_bytes())
        mutate(mutated)
        path = (harness.temp_root / f"p14-{name}.json").resolve()
        path.write_bytes(_canonical_bytes(mutated) + b"\n")
        output = (harness.temp_root / f"p14-{name}-output").resolve()
        before = len(harness.matrix_invocations)
        config_errors[name] = harness.expect_error(
            lambda path=path, output=output, name=name: harness.run_prelock(
                output,
                label=f"p14-{name}",
                mode="no_champion",
                config=path,
            )
        )
        assert len(harness.matrix_invocations) == before and not output.exists()
    parser_stderr = io.StringIO()
    with redirect_stderr(parser_stderr):
        try:
            harness.prelock_cli.main(
                [*harness.prelock_argv(harness.temp_root / "p14-cli-override"), "--start-date", "x"]
            )
        except SystemExit as exc:
            cli_override_code = exc.code
        else:
            cli_override_code = 0
    final_feature = next(
        (
            harness.roots[("historical_exposed_evaluation", "feature")]
            / "feature_points/exchange=binance/symbol=TONUSDT"
        ).rglob("*.parquet")
    )
    final_status = final_feature.stat()
    final_bytes = final_feature.read_bytes()
    harness.force_root_paths.add(str(harness.roots[("historical_exposed_evaluation", "feature")]))
    before = len(harness.matrix_invocations)
    partial_endpoint_parent = (harness.temp_root / "p14-partial-endpoint-parent").resolve()
    partial_endpoint_parent.mkdir()
    try:
        frame = pl.read_parquet(final_feature)
        if frame.height < 2:
            raise AssertionError("partial endpoint fixture requires at least two funding rows")
        frame.head(frame.height - 1).write_parquet(final_feature)
        partial_endpoint_error = harness.expect_error(
            lambda: harness.run_historical(
                prelock_champion,
                partial_endpoint_parent / "result",
                label="p14-partial-endpoint",
                mode="champion_pass_disagreement",
            )
        )
    finally:
        _restore_file(
            final_feature,
            final_bytes,
            stat.S_IMODE(final_status.st_mode),
            (final_status.st_atime_ns, final_status.st_mtime_ns),
        )
        harness.force_root_paths.discard(
            str(harness.roots[("historical_exposed_evaluation", "feature")])
        )
    payload["p14"] = {
        "chronology_exact": chronology_exact,
        "config_errors": config_errors,
        "cli_override_code": cli_override_code,
        "cli_override_message": parser_stderr.getvalue(),
        "partial_endpoint_error": partial_endpoint_error,
        "partial_endpoint_before_replay": len(harness.matrix_invocations) == before,
    }

    # P07/P17 owned-root attacks. Cached baseline seals are reused only for
    # untouched roots; each attacked/wrong-identity root is sealed by production.
    root_errors: dict[str, str] = {}

    def root_attack(
        name: str,
        operation: Callable[[], object],
        *,
        output: Path,
    ) -> None:
        before = len(harness.matrix_invocations)
        root_errors[name] = harness.expect_error(operation)
        root_errors[f"{name}_before_replay"] = str(
            len(harness.matrix_invocations) == before and not output.exists()
        )

    for name, overrides in (
        (
            "historical_preboundary_raw",
            {("historical_exposed_evaluation", "raw"): harness.roots[("validation", "raw")]},
        ),
        (
            "historical_preboundary_feature",
            {
                ("historical_exposed_evaluation", "feature"): harness.roots[
                    ("validation", "feature")
                ]
            },
        ),
        (
            "historical_overlap_embargo",
            {("historical_exposed_evaluation", "feature"): harness.roots[("embargo", "feature")]},
        ),
    ):
        output = (harness.temp_root / f"p07-{name}-parent" / "result").resolve()
        output.parent.mkdir()
        root_attack(
            name,
            lambda output=output, overrides=overrides, name=name: harness.run_historical(
                prelock_champion,
                output,
                label=f"p07-{name}",
                mode="champion_pass_disagreement",
                root_overrides=overrides,
            ),
            output=output,
        )
    payload["p07"] = {
        key: value for key, value in root_errors.items() if key.startswith("historical_")
    }

    warmup_feature = harness.roots[("warmup", "feature")]
    attacked_feature = next(warmup_feature.rglob("*.parquet"))

    def prelock_root_failure(
        name: str,
        *,
        overrides: dict[tuple[str, str], Path] | None = None,
        prepare: Callable[[], Callable[[], None]] | None = None,
    ) -> None:
        output = (harness.temp_root / f"p17-{name}-output").resolve()
        restore = (prepare or (lambda: lambda: None))()
        try:
            root_attack(
                name,
                lambda: harness.run_prelock(
                    output,
                    label=f"p17-{name}",
                    mode="no_champion",
                    root_overrides=overrides,
                ),
                output=output,
            )
        finally:
            restore()

    def gap_prepare() -> Callable[[], None]:
        status = attacked_feature.stat()
        data = attacked_feature.read_bytes()
        harness.force_root_paths.add(str(warmup_feature))
        attacked_feature.unlink()

        def restore() -> None:
            _restore_file(
                attacked_feature,
                data,
                stat.S_IMODE(status.st_mode),
                (status.st_atime_ns, status.st_mtime_ns),
            )
            harness.force_root_paths.discard(str(warmup_feature))

        return restore

    def duplicate_prepare() -> Callable[[], None]:
        status = attacked_feature.stat()
        data = attacked_feature.read_bytes()
        harness.force_root_paths.add(str(warmup_feature))
        frame = pl.read_parquet(attacked_feature)
        pl.concat((frame, frame.head(1))).write_parquet(attacked_feature)

        def restore() -> None:
            _restore_file(
                attacked_feature,
                data,
                stat.S_IMODE(status.st_mode),
                (status.st_atime_ns, status.st_mtime_ns),
            )
            harness.force_root_paths.discard(str(warmup_feature))

        return restore

    def outside_prepare() -> Callable[[], None]:
        target = (
            warmup_feature
            / "feature_points/exchange=binance/symbol=ADAUSDT/date=2022-12-30/part-0.parquet"
        )
        target.parent.mkdir(parents=True)
        shutil.copy2(attacked_feature, target)
        harness.force_root_paths.add(str(warmup_feature))

        def restore() -> None:
            target.unlink()
            target.parent.rmdir()
            harness.force_root_paths.discard(str(warmup_feature))

        return restore

    prelock_root_failure("gap", prepare=gap_prepare)
    prelock_root_failure("duplicate_timestamps", prepare=duplicate_prepare)
    prelock_root_failure("outside_interval", prepare=outside_prepare)
    prelock_root_failure(
        "nonadjacent_later_root",
        overrides={("warmup", "feature"): harness.roots[("validation", "feature")]},
    )
    prelock_root_failure(
        "purge_hidden_in_adjacent",
        overrides={("purge", "feature"): harness.roots[("embargo", "feature")]},
    )

    embargo_feature = harness.roots[("embargo", "feature")]
    embargo_file = next(embargo_feature.rglob("*.parquet"))
    embargo_status = embargo_file.stat()
    embargo_bytes = embargo_file.read_bytes()
    harness.force_root_paths.add(str(embargo_feature))
    embargo_output = (harness.temp_root / "p17-embargo-hash-parent" / "result").resolve()
    embargo_output.parent.mkdir()
    before = len(harness.matrix_invocations)
    try:
        _mutate_feature_values(embargo_file, 0.003)
        embargo_hash_error = harness.expect_error(
            lambda: harness.run_historical(
                prelock_champion,
                embargo_output,
                label="p17-embargo-hash",
                mode="champion_pass_disagreement",
            )
        )
    finally:
        _restore_file(
            embargo_file,
            embargo_bytes,
            stat.S_IMODE(embargo_status.st_mode),
            (embargo_status.st_atime_ns, embargo_status.st_mtime_ns),
        )
        harness.force_root_paths.discard(str(embargo_feature))
    root_errors["embargo_hash_mutation"] = embargo_hash_error
    root_errors["embargo_hash_mutation_before_replay"] = str(
        len(harness.matrix_invocations) == before and not embargo_output.exists()
    )

    contract_mutation = (harness.temp_root / "p17-contract-mutated.json").resolve()
    contract_payload = json.loads(CONTRACT_PATH.read_bytes())
    contract_payload["attacker"] = True
    contract_mutation.write_bytes(_canonical_bytes(contract_payload) + b"\n")
    contract_output = (harness.temp_root / "p17-contract-output").resolve()
    before = len(harness.matrix_invocations)
    contract_error = harness.expect_error(
        lambda: harness.run_prelock(
            contract_output,
            label="p17-contract",
            mode="no_champion",
            contract=contract_mutation,
        )
    )
    root_errors["contract_manifest"] = contract_error
    root_errors["contract_manifest_before_replay"] = str(
        len(harness.matrix_invocations) == before and not contract_output.exists()
    )
    root_errors["historical_poison_prelock_stable"] = str(all(p03_results.values()))
    payload["p17"] = root_errors

    # P05 outer-seal-preserving mutations prove the inner bindings. Manifest
    # semantic mutations also rebind the boundary capsule to the new manifest
    # hash so production manifest activation, not an earlier capsule mismatch,
    # is the rejecting seam.
    p05_errors: dict[str, str] = {}

    def historical_clone_failure(
        name: str,
        mutate: Callable[[Path], None],
    ) -> None:
        clone = _mutable_clone(
            prelock_champion,
            (harness.temp_root / f"p05-{name}-prelock").resolve(),
        )
        mutate(clone)
        harness.reseal_prelock(clone)
        parent = (harness.temp_root / f"p05-{name}-history-parent").resolve()
        parent.mkdir()
        output = parent / "result"
        before = len(harness.matrix_invocations)
        p05_errors[name] = harness.expect_error(
            lambda: harness.run_historical(
                clone,
                output,
                label=f"p05-{name}",
                mode="champion_pass_disagreement",
            )
        )
        p05_errors[f"{name}_before_replay"] = str(
            len(harness.matrix_invocations) == before and not output.exists()
        )

    def mutate_config(clone: Path) -> None:
        config = _read_json(clone / "inputs/config.json")
        config["revision"] = "attacker"
        (clone / "inputs/config.json").write_bytes(_canonical_bytes(config) + b"\n")

    historical_clone_failure("config", mutate_config)

    def mutate_runtime_contract(clone: Path) -> None:
        config = _read_json(clone / "inputs/config.json")
        config["runtime_contract"]["static_attributes"]["INITIAL_CAPITAL"] = 1.0
        (clone / "inputs/config.json").write_bytes(_canonical_bytes(config) + b"\n")

    historical_clone_failure("runtime_contract", mutate_runtime_contract)

    def mutate_capsule(clone: Path) -> None:
        path = clone / capsule_paths[0]
        envelope = _read_json(path)
        envelope["state_payload"]["capsule_sha256"] = "0" * 64
        path.write_bytes(_canonical_bytes(envelope) + b"\n")

    historical_clone_failure("capsule", mutate_capsule)

    manifest_mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "source": lambda value: value["source_artifacts"][0].__setitem__("sha256", "0" * 64),
        "membership": lambda value: value["children"][0].__setitem__(
            "candidate_id", "attacker_child"
        ),
        "policy": lambda value: value.__setitem__("allocation_method", "attacker_policy"),
        "gross": lambda value: value.__setitem__("gross_cap", 1.25),
        "manifest": lambda value: value.__setitem__("artifact_kind", "attacker_manifest.v1"),
    }
    for name, mutate in manifest_mutations.items():

        def mutate_manifest(
            clone: Path,
            *,
            mutate: Callable[[dict[str, Any]], None] = mutate,
        ) -> None:
            row_id = "component_carry_1x"
            path = clone / f"manifests/prelock_final_refit/{row_id}.json"
            manifest = _read_json(path)
            mutate(manifest)
            manifest_bytes = _canonical_bytes(manifest) + b"\n"
            path.write_bytes(manifest_bytes)
            harness._rebind_first_capsule(clone, row_id, _sha256(manifest_bytes))

        historical_clone_failure(name, mutate_manifest)

    p05_wrong_root_output = (harness.temp_root / "p05-wrong-root-parent" / "result").resolve()
    p05_wrong_root_output.parent.mkdir()
    before = len(harness.matrix_invocations)
    p05_errors["raw_feature"] = harness.expect_error(
        lambda: harness.run_historical(
            prelock_champion,
            p05_wrong_root_output,
            label="p05-raw-feature",
            mode="champion_pass_disagreement",
            root_overrides={
                ("historical_exposed_evaluation", "raw"): harness.roots[("validation", "raw")],
                ("historical_exposed_evaluation", "feature"): harness.roots[
                    ("validation", "feature")
                ],
            },
        )
    )
    p05_errors["raw_feature_before_replay"] = str(
        len(harness.matrix_invocations) == before and not p05_wrong_root_output.exists()
    )
    payload["p05"] = p05_errors

    # P19 exact Git-blob trial inventory is invariant to every ambient candidate
    # location. Same output target preserves path-bound manifest bytes.
    p19_output = (harness.temp_root / "p19-prelock").resolve()
    assert (
        harness.run_prelock(
            p19_output,
            label="p19-reference",
            mode="no_champion",
        )
        == 0
    )
    p19_prior = (p19_output / "inputs/prior_trial_inventory.json").read_bytes()
    p19_ledger = (p19_output / "trial/ledger.json").read_bytes()
    harness.cleanup_bundle(p19_output)
    poison_paths = (
        REPO_ROOT / ".omc" / f"alpha-max-poison-{os.getpid()}.json",
        REPO_ROOT / ".omx" / f"alpha-max-poison-{os.getpid()}.json",
        REPO_ROOT
        / "var/reports/ultragoal_full_pool_strategy"
        / f"alpha-max-newer-{os.getpid()}.json",
        harness.temp_root / f"alpha-max-output-poison-{os.getpid()}.json",
    )
    for path in poison_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"attacker_candidate":true}\n')
    try:
        assert (
            harness.run_prelock(
                p19_output,
                label="p19-poisoned",
                mode="no_champion",
            )
            == 0
        )
    finally:
        for path in poison_paths:
            path.unlink(missing_ok=True)
    p19_ledger_payload = _read_json(p19_output / "trial/ledger.json")
    payload["p19"] = {
        "prior_identical": (p19_output / "inputs/prior_trial_inventory.json").read_bytes()
        == p19_prior,
        "ledger_identical": (p19_output / "trial/ledger.json").read_bytes() == p19_ledger,
        "num_trials": p19_ledger_payload["num_trials"],
        "prior_key_set_sha256": p19_ledger_payload["prior_trial_key_set_sha256"],
        "current_key_set_sha256": p19_ledger_payload["current_trial_key_set_sha256"],
    }

    # P20 uses every actual retained manifest and actual Backtest constructor;
    # only event execution is omitted. Rejecting/ignoring doubles prove both
    # portfolio sinks are supplied and independently validated before start.
    preflight_champion = runner.preflight_alpha_max_runtime_contract(
        prelock_champion / "inputs/config.json"
    )
    admitted = tuple(_read_json(prelock_champion / "admission/train.json")["admitted_symbols"])
    validation_seals, validation_failures = harness.real_root_validation(
        (
            (
                "purge",
                "feature",
                str(harness.roots[("purge", "feature")]),
            ),
            (
                "validation",
                "feature",
                str(harness.roots[("validation", "feature")]),
            ),
        ),
        exchange="binance",
        availability_start_by_kind={
            "feature": evidence._ALPHA_MAX_FEATURE_AVAILABILITY_START_BY_SYMBOL,
        },
        availability_end_by_kind={
            "feature": evidence._ALPHA_MAX_FEATURE_AVAILABILITY_END_BY_SYMBOL,
        },
    )
    assert not validation_failures
    lookup = runner._alpha_max_ordered_lookup(
        {
            ("purge", "feature"): validation_seals[("purge", "feature")],
            ("validation", "feature"): validation_seals[("validation", "feature")],
        },
        ("purge", "validation"),
    )
    timestamp_ms = int(datetime(2025, 6, 8, tzinfo=UTC).timestamp() * 1000)
    data_dict = dict.fromkeys(admitted, ((timestamp_ms, 100.0, 101.0, 99.0, 100.0, 10.0),))
    plan_keys: set[str] | None = None
    sink_identities = True
    for row_id in runner._ALPHA_MAX_RESOLVABLE_ROWS:
        manifest_path = prelock_champion / f"manifests/validation_train_fit/{row_id}.json"
        for nominal in runner.ALPHA_MAX_COST_CELL_BPS:
            resolver = runner.AlphaMaxFundingBoundaryResolver(lookup, admitted)
            activation = runner.construct_alpha_max_engine(
                preflight_champion,
                output_root=str(prelock_champion),
                phase="validation_train_fit",
                manifest_path=str(manifest_path),
                admitted_symbols=admitted,
                phase_id="validation",
                nominal_cost_bps=nominal,
                raw_root=str(harness.roots[("validation", "raw")]),
                ordered_lookup=lookup,
                funding_resolver=resolver,
                data_dict=data_dict,
            )
            harness.constructor_rows.append((row_id, nominal))
            keys = set(activation.constructor_plan.portfolio_kwargs)
            plan_keys = keys if plan_keys is None else plan_keys
            sink_identities = sink_identities and (
                keys == plan_keys
                and activation.backtest.portfolio.fill_application_attribution_sink.__self__
                is activation.attribution_collector
                and activation.backtest.portfolio.full_event_equity_sink.__self__
                is activation.full_event_equity_tracker
                and activation.backtest.portfolio.funding_boundary_resolver is resolver
                and activation.backtest.portfolio.reporting_sampling_timeframe == "4h"
            )

    legacy_handler_calls: list[dict[str, object]] = []

    class LegacyHandler:
        def __init__(self, events, csv_dir, symbols, start, end, data_dict, **kwargs) -> None:
            legacy_handler_calls.append(dict(kwargs))
            if kwargs:
                raise TypeError("legacy handler rejects kwargs")
            self.symbol_list = symbols
            self.continue_backtest = False

    class LegacyStrategy:
        def __init__(self, bars, events) -> None:
            self.bars = bars
            self.events = events
            self.decision_cadence_seconds = 60

    class LegacyPortfolio:
        def __init__(self, bars, events, start, config, **kwargs) -> None:
            self.bars = bars
            self.kwargs = kwargs

    class LegacyExecution:
        def __init__(self, events, bars, config) -> None:
            self.bars = bars

    legacy = harness.real_backtest(
        csv_dir="/tmp/legacy",
        symbol_list=["BTCUSDT"],
        start_date=datetime(2025, 1, 1, tzinfo=UTC),
        end_date=datetime(2025, 1, 2, tzinfo=UTC),
        data_handler_cls=LegacyHandler,
        execution_handler_cls=LegacyExecution,
        portfolio_cls=LegacyPortfolio,
        strategy_cls=LegacyStrategy,
        data_handler_kwargs={"legacy_optional": True},
        config=SimpleNamespace(
            TIMEFRAME="1s",
            DECISION_CADENCE_SECONDS=60,
            SKIP_AHEAD_ENABLED=False,
        ),
    )
    alpha_portfolio_kwargs = {
        "fill_application_attribution_sink",
        "full_event_equity_sink",
        "funding_boundary_resolver",
        "reporting_sampling_timeframe",
    }

    class SinkRejected(RuntimeError):
        pass

    class RejectingBacktest:
        def __init__(self, *args, **kwargs) -> None:
            required = {
                "fill_application_attribution_sink",
                "full_event_equity_sink",
            }
            portfolio_kwargs = kwargs.get("portfolio_kwargs")
            if type(portfolio_kwargs) is not MappingProxyType and not isinstance(
                portfolio_kwargs, dict
            ):
                raise AssertionError("alpha_portfolio_kwargs_missing")
            if not required <= set(portfolio_kwargs):
                raise AssertionError("alpha_sink_kwargs_missing")
            raise SinkRejected("alpha_sink_kwargs_rejected")

    runner.Backtest = RejectingBacktest
    rejecting_error = harness.expect_error(
        lambda: runner.construct_alpha_max_engine(
            preflight_champion,
            output_root=str(prelock_champion),
            phase="validation_train_fit",
            manifest_path=str(
                prelock_champion / "manifests/validation_train_fit/component_carry_1x.json"
            ),
            admitted_symbols=admitted,
            phase_id="validation",
            nominal_cost_bps=30,
            raw_root=str(harness.roots[("validation", "raw")]),
            ordered_lookup=lookup,
            funding_resolver=runner.AlphaMaxFundingBoundaryResolver(lookup, admitted),
            data_dict=data_dict,
        )
    )

    class IgnoringBacktest(harness.real_backtest):
        def __init__(self, *args, **kwargs) -> None:
            portfolio_kwargs = dict(kwargs["portfolio_kwargs"])
            portfolio_kwargs.pop("full_event_equity_sink")
            kwargs["portfolio_kwargs"] = portfolio_kwargs
            super().__init__(*args, **kwargs)

    runner.Backtest = IgnoringBacktest
    ignoring_error = harness.expect_error(
        lambda: runner.construct_alpha_max_engine(
            preflight_champion,
            output_root=str(prelock_champion),
            phase="validation_train_fit",
            manifest_path=str(
                prelock_champion / "manifests/validation_train_fit/component_carry_1x.json"
            ),
            admitted_symbols=admitted,
            phase_id="validation",
            nominal_cost_bps=30,
            raw_root=str(harness.roots[("validation", "raw")]),
            ordered_lookup=lookup,
            funding_resolver=runner.AlphaMaxFundingBoundaryResolver(lookup, admitted),
            data_dict=data_dict,
        )
    )
    runner.Backtest = harness.real_backtest
    payload["p20"] = {
        "constructor_count": len(harness.constructor_rows),
        "constructor_pairs_unique": len(set(harness.constructor_rows)),
        "forbidden_rows_absent": not any(
            row_id in {*runner._ALPHA_MAX_UNAVAILABLE_ROWS, *runner._ALPHA_MAX_DIAGNOSTIC_ROWS}
            for row_id, _nominal in harness.constructor_rows
        ),
        "portfolio_kwargs": sorted(plan_keys or ()),
        "sink_identities": sink_identities,
        "legacy_retry_calls": legacy_handler_calls,
        "legacy_alpha_kwargs_absent": not (
            alpha_portfolio_kwargs.intersection(legacy.portfolio.kwargs)
        ),
        "rejecting_error": rejecting_error,
        "ignoring_error": ignoring_error,
    }

    # P23-P25 run inside this public child and use actual engine construction.
    # They never call ``simulate_trading``: every hostile mutation is rejected
    # by the production descriptor/activation validator before an event or
    # funding-ledger row can exist.
    activation_manifest = (
        prelock_champion / "manifests/validation_train_fit/component_carry_1x.json"
    )
    base_feature_specs = tuple(lookup.root_specs)

    def fresh_activation_inputs() -> tuple[
        runner.AlphaMaxOrderedFundingLookup,
        runner.AlphaMaxFundingBoundaryResolver,
    ]:
        local_lookup = runner.AlphaMaxOrderedFundingLookup(base_feature_specs)
        return local_lookup, runner.AlphaMaxFundingBoundaryResolver(local_lookup, admitted)

    def construct_actual_activation(
        local_lookup: runner.AlphaMaxOrderedFundingLookup,
        local_resolver: runner.AlphaMaxFundingBoundaryResolver,
    ) -> runner.AlphaMaxEngineActivation:
        return runner.construct_alpha_max_engine(
            preflight_champion,
            output_root=str(prelock_champion),
            phase="validation_train_fit",
            manifest_path=str(activation_manifest),
            admitted_symbols=admitted,
            phase_id="validation",
            nominal_cost_bps=30,
            raw_root=str(harness.roots[("validation", "raw")]),
            ordered_lookup=local_lookup,
            funding_resolver=local_resolver,
            data_dict=data_dict,
        )

    economic_events: list[str] = []
    original_fast_queue_put = runner.FastQueue.put

    def recording_fast_queue_put(self, item, block=True, timeout=None):
        economic_events.append(str(getattr(item, "type", type(item).__name__)).upper())
        return original_fast_queue_put(self, item, block=block, timeout=timeout)

    runner.FastQueue.put = recording_fast_queue_put
    try:
        # P23 swaps hostile-but-readable bytes into the exact manifest/config
        # path only while the real consumer opens its descriptor. The producer
        # seal has already retained A, so consuming transient B must fail.
        manifest_a = activation_manifest.read_bytes()
        config_target = prelock_champion / "inputs/config.json"
        config_a = config_target.read_bytes()
        manifest_b_payload = json.loads(manifest_a)
        manifest_b_payload["candidate_symbols"] = list(
            reversed(manifest_b_payload["candidate_symbols"])
        )
        manifest_b_payload["children"][0]["candidate_symbols"] = list(
            reversed(manifest_b_payload["children"][0]["candidate_symbols"])
        )
        manifest_b_payload["children"][0]["netting_group_gross_cap"] = 1.5
        manifest_b = (harness.temp_root / "p23-manifest-b.json").resolve()
        manifest_b.write_bytes(_canonical_bytes(manifest_b_payload) + b"\n")
        config_b_payload = json.loads(config_a)
        config_b_payload["incumbent_resolution"]["rows"][0]["resolution_reason"] = (
            "hostile transient incumbent audit B"
        )
        config_b = (harness.temp_root / "p23-config-b.json").resolve()
        config_b.write_bytes(_canonical_bytes(config_b_payload) + b"\n")

        def atomic_descriptor_swap(target: Path, replacement: Path, read: Callable[[], Any]):
            retained = replacement.with_name(f"{replacement.stem}-retained-a.json")
            os.replace(target, retained)
            os.replace(replacement, target)
            try:
                return read()
            finally:
                os.replace(target, replacement)
                os.replace(retained, target)

        p23_errors: dict[str, str | bool] = {}
        manifest_parent_mode = stat.S_IMODE(activation_manifest.parent.stat().st_mode)
        config_parent_mode = stat.S_IMODE(config_target.parent.stat().st_mode)
        os.chmod(activation_manifest.parent, 0o755)
        os.chmod(config_target.parent, 0o755)
        try:
            original_artifact_json = artifact_mode.read_artifact_json

            def transient_manifest(path, *, artifact_id):
                return atomic_descriptor_swap(
                    Path(path),
                    manifest_b,
                    lambda: original_artifact_json(path, artifact_id=artifact_id),
                )

            local_lookup, local_resolver = fresh_activation_inputs()
            before_events = len(economic_events)
            artifact_mode.read_artifact_json = transient_manifest
            try:
                p23_errors["manifest_descriptor_swap"] = harness.expect_error(
                    lambda: construct_actual_activation(local_lookup, local_resolver)
                )
            finally:
                artifact_mode.read_artifact_json = original_artifact_json
            p23_errors["manifest_descriptor_swap_before_events"] = (
                len(economic_events) == before_events and local_resolver.ledger == ()
            )

            original_artifact_bytes = artifact_mode.read_artifact_bytes

            def transient_config(path, *, artifact_id):
                return atomic_descriptor_swap(
                    Path(path),
                    config_b,
                    lambda: original_artifact_bytes(path, artifact_id=artifact_id),
                )

            local_lookup, local_resolver = fresh_activation_inputs()
            before_events = len(economic_events)
            artifact_mode.read_artifact_bytes = transient_config
            try:
                p23_errors["config_descriptor_swap"] = harness.expect_error(
                    lambda: construct_actual_activation(local_lookup, local_resolver)
                )
            finally:
                artifact_mode.read_artifact_bytes = original_artifact_bytes
            p23_errors["config_descriptor_swap_before_events"] = (
                len(economic_events) == before_events and local_resolver.ledger == ()
            )
        finally:
            os.chmod(activation_manifest.parent, manifest_parent_mode)
            os.chmod(config_target.parent, config_parent_mode)
        p23_errors["targets_restored"] = (
            activation_manifest.read_bytes() == manifest_a
            and config_target.read_bytes() == config_a
        )
        payload["p23"] = p23_errors

        def activation_mutation_result(
            mutation: Callable[[runner.AlphaMaxEngineActivation], None],
        ) -> tuple[str, bool]:
            before_events = len(economic_events)
            local_lookup, local_resolver = fresh_activation_inputs()
            activation = construct_actual_activation(local_lookup, local_resolver)
            mutation(activation)
            error = harness.expect_error(
                lambda: runner.validate_alpha_max_engine_activation(activation)
            )
            return error, (
                len(economic_events) == before_events
                and activation.funding_resolver.ledger == ()
                and local_resolver.ledger == ()
            )

        # P24 mutates each identity-bearing lookup location independently.
        p24_errors: dict[str, str | bool] = {}

        def mutate_handler_lookup(activation: runner.AlphaMaxEngineActivation) -> None:
            activation.backtest.data_handler._feature_lookup = runner.AlphaMaxOrderedFundingLookup(
                base_feature_specs
            )

        def mutate_activation_lookup(activation: runner.AlphaMaxEngineActivation) -> None:
            object.__setattr__(
                activation,
                "ordered_lookup",
                runner.AlphaMaxOrderedFundingLookup(base_feature_specs),
            )

        def mutate_lookup_sequence(activation: runner.AlphaMaxEngineActivation) -> None:
            object.__setattr__(
                activation.ordered_lookup,
                "_root_specs",
                (activation.ordered_lookup.root_specs[-1],),
            )

        for name, mutate in (
            ("handler_lookup_copy", mutate_handler_lookup),
            ("activation_lookup_copy", mutate_activation_lookup),
            ("lookup_root_sequence", mutate_lookup_sequence),
        ):
            error, before_events = activation_mutation_result(mutate)
            p24_errors[name] = error
            p24_errors[f"{name}_before_events"] = before_events
        payload["p24"] = p24_errors

        # P25 mutates resolver, admitted-tuple, raw-accessor, and owning bars
        # identities only after a successful actual activation.
        p25_errors: dict[str, str | bool] = {}

        def mutate_activation_resolver(activation: runner.AlphaMaxEngineActivation) -> None:
            object.__setattr__(
                activation,
                "funding_resolver",
                runner.AlphaMaxFundingBoundaryResolver(
                    activation.ordered_lookup, activation.admitted_symbols
                ),
            )

        def mutate_resolver_lookup(activation: runner.AlphaMaxEngineActivation) -> None:
            object.__setattr__(
                activation.funding_resolver,
                "_ordered_lookup",
                runner.AlphaMaxOrderedFundingLookup(base_feature_specs),
            )

        def mutate_resolver_admitted(activation: runner.AlphaMaxEngineActivation) -> None:
            object.__setattr__(
                activation.funding_resolver,
                "_admitted_symbols",
                (*activation.admitted_symbols,),
            )

        def mutate_accessor_owner(activation: runner.AlphaMaxEngineActivation) -> None:
            object.__setattr__(
                activation.funding_resolver,
                "_bound_raw_accessor_owner",
                object(),
            )

        def mutate_accessor_function(activation: runner.AlphaMaxEngineActivation) -> None:
            activation.backtest.data_handler.get_latest_raw_point = lambda *args, **kwargs: None

        def mutate_portfolio_bars(activation: runner.AlphaMaxEngineActivation) -> None:
            activation.backtest.portfolio.bars = object()

        def mutate_missing_resolver(activation: runner.AlphaMaxEngineActivation) -> None:
            object.__setattr__(
                activation.backtest.portfolio,
                "_funding_boundary_resolver",
                None,
            )

        for name, mutate in (
            ("activation_resolver_copy", mutate_activation_resolver),
            ("resolver_lookup_copy", mutate_resolver_lookup),
            ("resolver_admitted_copy", mutate_resolver_admitted),
            ("bound_accessor_owner", mutate_accessor_owner),
            ("raw_accessor_function", mutate_accessor_function),
            ("portfolio_bars", mutate_portfolio_bars),
            ("missing_resolver", mutate_missing_resolver),
        ):
            error, before_events = activation_mutation_result(mutate)
            p25_errors[name] = error
            p25_errors[f"{name}_before_events"] = before_events
        payload["p25"] = p25_errors
    finally:
        runner.FastQueue.put = original_fast_queue_put

    # P21 candidate identity remains 10 symbols in config/manifests; only the
    # exact admission tuple activates engines. Behavioral/candidate/source and
    # admitted-order mutations all fail before the matrix loop.
    config_nodes = config_payload["current_trial_registry"]["nodes"]
    final_manifests = [
        _read_json(prelock_champion / f"manifests/prelock_final_refit/{row_id}.json")
        for row_id in runner._ALPHA_MAX_RESOLVABLE_ROWS
    ]
    p21_errors: dict[str, str] = {}
    for name, mutate in (
        (
            "behavioral_node",
            lambda value: value["current_trial_registry"]["nodes"][0]["params"].__setitem__(
                "entry_funding", 0.123
            ),
        ),
        (
            "candidate_mapping",
            lambda value: value.__setitem__(
                "candidate_symbols", list(reversed(value["candidate_symbols"]))
            ),
        ),
        (
            "alternate_source_id",
            lambda value: value["current_trial_registry"]["nodes"][10].__setitem__(
                "source_id", "attacker_source"
            ),
        ),
        (
            "registry_reorder",
            lambda value: value["current_trial_registry"]["nodes"].reverse(),
        ),
    ):
        mutated = json.loads(CONFIG_PATH.read_bytes())
        mutate(mutated)
        path = (harness.temp_root / f"p21-{name}.json").resolve()
        path.write_bytes(_canonical_bytes(mutated) + b"\n")
        output = (harness.temp_root / f"p21-{name}-output").resolve()
        before = len(harness.matrix_invocations)
        p21_errors[name] = harness.expect_error(
            lambda path=path, output=output, name=name: harness.run_prelock(
                output,
                label=f"p21-{name}",
                mode="no_champion",
                config=path,
            )
        )
        p21_errors[f"{name}_before_replay"] = str(
            len(harness.matrix_invocations) == before and not output.exists()
        )

    baseline_admission = next(iter(harness.admission_cache.values()))
    reversed_artifact = object.__new__(evidence.AlphaMaxAdmissionArtifact)
    for field in baseline_admission.artifact.__dataclass_fields__:
        value = getattr(baseline_admission.artifact, field)
        if field == "admitted_symbols":
            value = tuple(reversed(value))
        object.__setattr__(reversed_artifact, field, value)
    reversed_computation = object.__new__(evidence.AlphaMaxAdmissionComputation)
    for field in baseline_admission.__dataclass_fields__:
        value = getattr(baseline_admission, field)
        if field == "artifact":
            value = reversed_artifact
        object.__setattr__(reversed_computation, field, value)
    harness.admission_override = reversed_computation
    reversed_output = (harness.temp_root / "p21-reversed-admission-output").resolve()
    before = len(harness.matrix_invocations)
    try:
        p21_errors["admitted_mapping"] = harness.expect_error(
            lambda: harness.run_prelock(
                reversed_output,
                label="p21-admitted-mapping",
                mode="no_champion",
            )
        )
    finally:
        harness.admission_override = None
    p21_errors["admitted_mapping_before_replay"] = str(
        len(harness.matrix_invocations) == before and not reversed_output.exists()
    )
    prior_candidate = _read_json(p19_output / "inputs/prior_trial_inventory.json")["candidates"][0]
    prior_node = evidence.normalize_alpha_max_prior_trial_node(prior_candidate)
    prior_key = evidence.alpha_max_trial_key(prior_node)
    cosmetic_candidate = json.loads(_canonical_bytes(prior_candidate))
    cosmetic_candidate.pop("name", None)
    cosmetic_candidate["status"] = "renamed-cosmetic-status"
    cosmetic_candidate["tags"] = ["renamed"]
    reordered_candidate = json.loads(_canonical_bytes(prior_candidate))
    reordered_candidate["symbols"].reverse()
    behavioral_candidate = json.loads(_canonical_bytes(prior_candidate))
    behavioral_candidate["metadata"]["alpha_max_process_fixture_mutation"] = True
    payload["p21"] = {
        "config_registry_sha256": config_payload["trial_ledger"]["current_registry_sha256"],
        "all_nodes_ten_candidates": all(
            tuple(node["symbols"]) == evidence.ALPHA_MAX_CANDIDATE_SYMBOLS for node in config_nodes
        ),
        "admitted_symbols": list(admitted),
        "all_manifests_candidate_ten_active_admitted": all(
            tuple(manifest["candidate_symbols"]) == evidence.ALPHA_MAX_CANDIDATE_SYMBOLS
            and tuple(manifest["admitted_symbols"]) == admitted
            and all(
                tuple(child["candidate_symbols"]) == evidence.ALPHA_MAX_CANDIDATE_SYMBOLS
                for child in manifest["children"]
            )
            and all(tuple(child["symbols"]) == admitted for child in manifest["children"])
            for manifest in final_manifests
        ),
        "cosmetic_prior_key_stable": evidence.alpha_max_trial_key(
            evidence.normalize_alpha_max_prior_trial_node(cosmetic_candidate)
        )
        == prior_key,
        "symbol_reorder_key_stable": evidence.alpha_max_trial_key(
            evidence.normalize_alpha_max_prior_trial_node(reordered_candidate)
        )
        == prior_key,
        "behavioral_prior_key_changed": evidence.alpha_max_trial_key(
            evidence.normalize_alpha_max_prior_trial_node(behavioral_candidate)
        )
        != prior_key,
        "errors": p21_errors,
    }

    # P26 crosses the public prelock parser into the actual runtime preflight.
    # Each embedded incumbent-audit mutation is rejected before the production
    # matrix/replay function can be entered or an output root can be created.
    p26_errors: dict[str, str | bool] = {}
    for name, field, value in (
        ("audit_path", "path", "hostile/report-latest.json"),
        ("audit_git_blob", "git_blob_oid", "0" * 40),
        ("audit_content_sha", "content_sha256", "0" * 64),
        ("resolution_status", "resolution_status", "resolved"),
        ("resolution_reason", "resolution_reason", "hostile reason"),
        ("normative_audit_sha", "audit_sha256", "0" * 64),
    ):
        mutated = json.loads(CONFIG_PATH.read_bytes())
        if field == "audit_sha256":
            mutated["normative_sources"]["incumbent_resolution_audit_sha256"] = value
        elif field in {"path", "git_blob_oid", "content_sha256"}:
            mutated["incumbent_resolution"]["rows"][0]["frozen_audit_files"][0][field] = value
        else:
            mutated["incumbent_resolution"]["rows"][0][field] = value
        config_path = (harness.temp_root / f"p26-{name}.json").resolve()
        config_path.write_bytes(_canonical_bytes(mutated) + b"\n")
        output = (harness.temp_root / f"p26-{name}-output").resolve()
        before = len(harness.matrix_invocations)
        p26_errors[name] = harness.expect_error(
            lambda config_path=config_path, output=output, name=name: harness.run_prelock(
                output,
                label=f"p26-{name}",
                mode="no_champion",
                config=config_path,
            )
        )
        p26_errors[f"{name}_before_replay"] = (
            len(harness.matrix_invocations) == before and not output.exists()
        )
    p26_errors["embedded_audit_bytes_exact"] = (
        preflight_champion.incumbent_resolution_bytes
        == _canonical_bytes(config_payload["incumbent_resolution"])
    )
    p26_errors["embedded_audit_sha_exact"] = (
        preflight_champion.incumbent_resolution_audit_sha256
        == config_payload["normative_sources"]["incumbent_resolution_audit_sha256"]
    )
    payload["p26"] = p26_errors

    # P15 scans actual sealed process artifacts, not only help text.
    forbidden_pattern = re.compile(r"\b(?:untouched|locked|prospective|confirmatory)\b", re.I)
    text_artifacts: list[tuple[str, bytes]] = []
    for root in (prelock_no, prelock_champion, historical_no, historical_pass, historical_fail):
        text_artifacts.extend(
            (f"{root.name}/{relative}", value)
            for relative, value in _snapshot_bytes(root).items()
            if relative.endswith((".json", ".txt", ".md"))
        )
    forbidden_hits = [
        relative
        for relative, value in text_artifacts
        if forbidden_pattern.search(value.decode("utf-8", errors="ignore"))
    ]
    payload["p15"] = {
        "forbidden_hits": forbidden_hits,
        "historical_exposure_status": terminal_pass["historical_exposure_status"],
        "requires_fresh_confirmation": terminal_pass["requires_fresh_confirmation"],
        "report_status": pass_report["status"],
    }

    return payload


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("alpha_max_cli_process_fixture_arguments_invalid")
    print(json.dumps(run_alpha_max_cli_process_fixture(Path(sys.argv[1])), sort_keys=True))
