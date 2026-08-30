from __future__ import annotations

import hashlib
import inspect
import json
import os
import struct
import subprocess
import sys
import threading
from concurrent.futures import Future
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import polars as pl
import pytest

import lumina_quant.research.alpha_max_evidence as alpha_max_evidence
import lumina_quant.research.alpha_max_engine_runner as alpha_max_runner

from lumina_quant.core.events import OrderEvent, SignalEvent
from lumina_quant.research.alpha_max_engine_runner import (
    ALPHA_MAX_COST_CELL_BPS,
    AlphaMaxAttributionCollector,
    AlphaMaxRuntimeContractError,
    construct_alpha_max_engine,
    create_alpha_max_historical_package,
    create_alpha_max_prelock_bundle,
    orchestrate_alpha_max_status_matrix,
    preflight_alpha_max_runtime_contract,
    seal_alpha_max_manifest_activation,
    validate_alpha_max_engine_activation,
)
from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_CANDIDATE_SYMBOLS,
    AlphaMaxActualEngineRunReceipt,
    AlphaMaxCapsuleReceipt,
    AlphaMaxCostCellEvidence,
    AlphaMaxCostCellPreGateEvidence,
    AlphaMaxFoldRunEvidence,
    AlphaMaxFundingBoundaryResolver,
    AlphaMaxGateInput,
    AlphaMaxManifestReceipt,
    AlphaMaxOrderedFundingLookup,
    AlphaMaxRunReportOnlyDiagnostics,
    AlphaMaxRootSeal,
    AlphaMaxStreamingEquityTracker,
    AlphaMaxTreeEntry,
    FeatureRootSpec,
    materialize_alpha_max_manifest,
    select_alpha_max_prelock_champion,
    validate_alpha_max_train_liquidity_buckets,
)
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.strategies.artifact_portfolio_mode import ArtifactPortfolioModeStrategy


def test_prelock_training_worker_cap_is_bounded_and_uses_processes() -> None:
    parameter = inspect.signature(alpha_max_runner.run_alpha_max_prelock_process).parameters[
        "max_training_workers"
    ]
    assert parameter.default == 4
    assert alpha_max_runner.ProcessPoolExecutor is not alpha_max_runner.ThreadPoolExecutor


def test_parent_prepares_only_missing_training_prefixes_in_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component_ids = (
        "component_carry_1x",
        "component_near_high_1x",
        "component_trend_1x",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"{}\n")
    receipt, _payload = alpha_max_evidence.read_artifact_bytes(
        manifest_path, artifact_id="alpha_max_engine_portfolio_manifest"
    )
    manifests = {
        component_id: AlphaMaxManifestReceipt(
            row_id=component_id,
            phase="validation_train_fit",
            relative_path=f"manifests/validation_train_fit/{component_id}.json",
            sha256=receipt.sha256,
            byte_count=receipt.byte_count,
            activation_receipt=receipt,
        )
        for component_id in component_ids
    }
    existing = b'{"sealed":"resume"}\n'
    stored = {"component_near_high_1x": existing}
    built: list[str] = []
    parsed: list[tuple[str, bytes]] = []

    class Store:
        def load(self, *, unit_id: str, **_kwargs: object) -> bytes | None:
            return stored.get(unit_id)

        def seal(self, *, unit_id: str, data_bytes: bytes, **_kwargs: object) -> bytes:
            assert unit_id not in stored
            stored[unit_id] = data_bytes
            return data_bytes

    def build(
        *_args: object, manifest_receipt: AlphaMaxManifestReceipt, **_kwargs: object
    ) -> object:
        built.append(manifest_receipt.row_id)
        return object()

    monkeypatch.setattr(alpha_max_runner, "_alpha_max_build_indicator_prefix", build)
    monkeypatch.setattr(alpha_max_runner, "_alpha_max_capsule_state_payload", lambda _capsule: {})
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_training_prefix_from_checkpoint",
        lambda payload, *, component_id, **_kwargs: parsed.append((component_id, payload)),
    )
    alpha_max_runner._alpha_max_prepare_training_indicator_prefixes(
        SimpleNamespace(),
        output_root=tmp_path,
        component_ids=component_ids,
        manifests=manifests,
        admitted_symbols=("BTCUSDT",),
        root_seals={},
        checkpoint_store=Store(),
    )
    assert built == ["component_carry_1x", "component_trend_1x"]
    assert [component_id for component_id, _payload in parsed] == list(component_ids)
    assert parsed[1] == ("component_near_high_1x", existing)


def test_worker_rejects_missing_training_prefix_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"{}\n")
    receipt, _payload = alpha_max_evidence.read_artifact_bytes(
        manifest_path, artifact_id="alpha_max_engine_portfolio_manifest"
    )
    manifest = AlphaMaxManifestReceipt(
        row_id="component_carry_1x",
        phase="validation_train_fit",
        relative_path="manifests/validation_train_fit/component_carry_1x.json",
        sha256=receipt.sha256,
        byte_count=receipt.byte_count,
        activation_receipt=receipt,
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_build_indicator_prefix",
        lambda *_args, **_kwargs: pytest.fail("worker attempted prefix construction"),
    )

    class Store:
        def load(self, **_kwargs: object) -> None:
            return None

    start = datetime(2024, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(alpha_max_runner, "_AlphaMaxBoundedRawLoader", lambda *_args: object())
    monkeypatch.setattr(
        alpha_max_runner, "AlphaMaxFundingBoundaryResolver", lambda *_args: object()
    )
    monkeypatch.setattr(alpha_max_runner, "_alpha_max_phase_lookup", lambda *_args: object())
    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_expected_root_sequence", lambda _phase: ("train",)
    )
    with pytest.raises(AlphaMaxRuntimeContractError, match="alpha_max_training_prefix_missing"):
        alpha_max_runner._alpha_max_replay_training_component_returns(
            SimpleNamespace(
                phase_windows={
                    "train": SimpleNamespace(
                        start_utc=start.isoformat().replace("+00:00", "Z"),
                        end_utc=(start + timedelta(days=2)).isoformat().replace("+00:00", "Z"),
                    )
                }
            ),
            output_root=tmp_path,
            manifest_receipt=manifest,
            admitted_symbols=("BTCUSDT",),
            root_seals={
                ("train", "raw"): SimpleNamespace(path=str(tmp_path / "raw")),
                ("train", "feature"): SimpleNamespace(path=str(tmp_path / "feature")),
            },
            checkpoint_store=Store(),
        )


def test_spawn_worker_runner_completes_from_a_threaded_parent() -> None:
    """The production runner uses a fresh spawn pool, never forked state."""
    errors: list[BaseException] = []
    assert pl.DataFrame({"value": [1]}).select(pl.col("value").sum()).item() == 1

    def invoke() -> None:
        try:
            alpha_max_runner._run_alpha_max_training_component_workers(
                (("invalid", b"{}\n", "0" * 64),),
                max_training_workers=1,
            )
        except BaseException as exc:
            errors.append(exc)

    # Keep a Python thread live while native libraries have already been imported.
    ready = threading.Event()
    release = threading.Event()
    native_thread = threading.Thread(target=lambda: (ready.set(), release.wait(10)))
    native_thread.start()
    assert ready.wait(2)
    parent = threading.Thread(target=invoke)
    parent.start()
    parent.join(30)
    release.set()
    native_thread.join(2)
    assert not parent.is_alive(), "spawn worker runner deadlocked"
    assert len(errors) == 1
    assert isinstance(errors[0], AlphaMaxRuntimeContractError)
    assert str(errors[0]) == (
        "alpha_max_training_worker_result_invalid:invalid:alpha_max_training_worker_binding_invalid"
    )


def test_spawn_workers_reconstruct_real_authorities_in_a_clean_subprocess(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).with_name("_alpha_max_valid_spawn_fixture.py")
    completed = subprocess.run(
        [sys.executable, str(fixture), str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
        timeout=180,
    )
    report = json.loads(completed.stdout)
    assert report["pid_counts"] == [1, 2, 3, 3]


def test_training_component_replay_resumes_sealed_days_with_funding_carry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real precompute journal resumes immutable daily replay units exactly."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=253)
    component_id = "component_carry_1x"
    manifest_path = (tmp_path / "manifest.json").resolve()
    manifest_path.write_bytes(b"{}\n")
    manifest_read, _payload = alpha_max_evidence.read_artifact_bytes(
        manifest_path,
        artifact_id="alpha_max_engine_portfolio_manifest",
    )
    manifest = AlphaMaxManifestReceipt(
        row_id=component_id,
        phase="validation_train_fit",
        relative_path="manifests/validation_train_fit/component_carry_1x.json",
        sha256=manifest_read.sha256,
        byte_count=manifest_read.byte_count,
        activation_receipt=manifest_read,
    )
    day_ids = tuple(
        f"{component_id}--{(start + timedelta(days=index)):%Y%m%d}" for index in range(252)
    )
    ledger_row = alpha_max_evidence.AlphaMaxFundingBoundaryLedgerRow(
        symbol="BTCUSDT",
        boundary_ms=1,
        rate_source_timestamp_ms=1,
        price_row_timestamp_ms=1,
        price_close_timestamp_ms=1,
        qty=1.0,
        rate=0.001,
        price=100.0,
        payment=-0.1,
    )
    restored_ledgers: list[tuple[object, ...]] = []

    class Loader:
        def __init__(self, *_args: object) -> None:
            pass

        def load_day(self, day_start: datetime, day_end: datetime) -> dict[str, object]:
            return {"day": day_start.date().isoformat(), "end": day_end}

    class Resolver:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @classmethod
        def from_checkpoint(cls, *_args: object, ledger: tuple[object, ...]) -> Resolver:
            restored_ledgers.append(ledger)
            return cls()

    class Tracker:
        def __init__(self, *_args: object, reporting_end: datetime, **_kwargs: object) -> None:
            self.reporting_endpoints = tuple(
                SimpleNamespace(timestamp=reporting_end, equity=10_000.0 + offset)
                for offset in range(1, 7)
            )

        def bind_backtest(self, _backtest: object) -> None:
            pass

        def finalize(self) -> None:
            pass

    class InterruptingStore(alpha_max_runner._AlphaMaxPrecomputeCheckpointStore):
        crash_once = False

        def seal(self, *, unit_kind: str, unit_id: str, data_bytes: bytes) -> bytes:
            sealed = super().seal(unit_kind=unit_kind, unit_id=unit_id, data_bytes=data_bytes)
            if self.crash_once and unit_kind == "training_day":
                self.crash_once = False
                raise RuntimeError("crash-after-sealed-day")
            return sealed

    def carry() -> object:
        return alpha_max_runner._AlphaMaxDailyCarry(
            strategy_state={"strategy": 1},
            portfolio_state={"portfolio": 1},
            execution_state={"execution": 1},
            engine_state={"engine": 1},
            handler_rows=(),
            handler_timestamps_ms=(),
            funding_ledger=(ledger_row,),
        )

    def activation(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            backtest=SimpleNamespace(execution_handler=SimpleNamespace(pricing_trace_evidence=()))
        )

    monkeypatch.setattr(alpha_max_runner, "_AlphaMaxBoundedRawLoader", Loader)
    monkeypatch.setattr(alpha_max_runner, "AlphaMaxFundingBoundaryResolver", Resolver)
    monkeypatch.setattr(
        alpha_max_runner, "AlphaMaxAttributionCollector", lambda: SimpleNamespace(applications=())
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "AlphaMaxStreamingEquityTracker",
        lambda: SimpleNamespace(finalize=lambda: None),
    )
    monkeypatch.setattr(alpha_max_runner, "_AlphaMaxFoldEquityFanout", Tracker)
    monkeypatch.setattr(alpha_max_runner, "construct_alpha_max_engine", activation)
    monkeypatch.setattr(
        alpha_max_runner, "_run_alpha_max_exact_tick_reducer", lambda _activation: None
    )
    monkeypatch.setattr(
        alpha_max_runner, "validate_alpha_max_engine_activation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        alpha_max_runner, "_capture_alpha_max_daily_carry", lambda _activation: carry()
    )
    monkeypatch.setattr(alpha_max_runner, "_restore_alpha_max_daily_carry", lambda *_args: None)
    monkeypatch.setattr(
        alpha_max_runner,
        "_settle_alpha_max_day_boundary",
        lambda _activation, _tracker, day_end, *, scoring_boundary: (
            {"finalization": day_end.isoformat()} if scoring_boundary else None
        ),
    )
    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_build_indicator_prefix", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_training_prefix_from_checkpoint",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_capsule_state_payload", lambda _capsule: {"prefix": True}
    )
    monkeypatch.setattr(alpha_max_runner, "_alpha_max_phase_lookup", lambda *_args: object())
    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_expected_root_sequence", lambda _phase: ("train",)
    )
    monkeypatch.setattr(alpha_max_runner, "_validate_alpha_max_root_seals", lambda **_kwargs: None)

    preflight = SimpleNamespace(
        phase_windows={
            "train": SimpleNamespace(
                start_utc=start.isoformat().replace("+00:00", "Z"),
                end_utc=end.isoformat().replace("+00:00", "Z"),
            )
        }
    )
    roots = {
        ("train", "raw"): SimpleNamespace(path=str(tmp_path / "raw")),
        ("train", "feature"): SimpleNamespace(path=str(tmp_path / "feature")),
    }

    def store(root: Path) -> InterruptingStore:
        result = InterruptingStore(
            root,
            attempt_descriptor_sha256="a" * 64,
            attempt_role="prelock",
            domain="validation",
            runtime_identity_sha256="b" * 64,
            training_day_ids=day_ids,
        )
        result.seal(
            unit_kind="training_prefix",
            unit_id=component_id,
            data_bytes=b'{"prefix":true}\n',
        )
        return result

    uninterrupted = store((tmp_path / "uninterrupted").resolve())
    expected = alpha_max_runner._alpha_max_replay_training_component_returns(
        preflight,
        output_root=tmp_path,
        manifest_receipt=manifest,
        admitted_symbols=("BTCUSDT",),
        root_seals=roots,
        checkpoint_store=uninterrupted,
    )
    interrupted = store((tmp_path / "interrupted").resolve())
    interrupted.crash_once = True
    with pytest.raises(RuntimeError, match="crash-after-sealed-day"):
        alpha_max_runner._alpha_max_replay_training_component_returns(
            preflight,
            output_root=tmp_path,
            manifest_receipt=manifest,
            admitted_symbols=("BTCUSDT",),
            root_seals=roots,
            checkpoint_store=interrupted,
        )
    actual = alpha_max_runner._alpha_max_replay_training_component_returns(
        preflight,
        output_root=tmp_path,
        manifest_receipt=manifest,
        admitted_symbols=("BTCUSDT",),
        root_seals=roots,
        checkpoint_store=interrupted,
    )
    assert actual == expected
    assert interrupted.load(unit_kind="training_day", unit_id=day_ids[0]) is not None
    assert restored_ledgers and all(ledger == (ledger_row,) for ledger in restored_ledgers)


def test_prelock_worker_caps_order_publication_and_failure_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The parent alone publishes ordered results; only semantic failures poison."""
    component_ids = ("component_carry_1x", "component_near_high_1x", "component_trend_1x")
    worker_counts: list[int] = []
    published: list[str] = []
    poisoned: list[bool] = []
    worker_output_snapshots: list[tuple[str, ...]] = []
    prepared_prefixes: list[tuple[str, ...]] = []

    class Journal:
        _display_root = tmp_path / "journal"
        _descriptor_sha256 = "a" * 64
        _attempt_descriptor_sha256 = "a" * 64
        _attempt_role = "prelock"
        _domain = "validation"
        _runtime_identity_sha256 = "b" * 64
        _training_day_ids: tuple[str, ...] = ()
        _transaction_lock_identity = (1, 1)
        _root_identity = (1, 1)
        _units_identity = (1, 1)

        def poison(self) -> None:
            poisoned.append(True)

    class Store:
        descriptor_sha256 = "c" * 64
        _runtime_identity = {}

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._display_root = (tmp_path / "checkpoint").resolve()
            self._display_root.mkdir(exist_ok=True)
            precompute_root = self._display_root / "precompute"
            precompute_root.mkdir(exist_ok=True)
            self.output_root = (tmp_path / "output").resolve()
            self.output_root.mkdir(exist_ok=True)
            self._display_output_root = self.output_root
            self._output_parent_fd = os.open(self.output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
            self._bound_output_fd = os.open(self.output_root, os.O_RDONLY | os.O_DIRECTORY)
            status = os.fstat(self._bound_output_fd)
            self._bound_output_identity = (status.st_dev, status.st_ino)
            self.journal = Journal()
            precompute_status = precompute_root.stat()
            self.journal._root_identity = (
                precompute_status.st_dev,
                precompute_status.st_ino,
            )

        def bind_output_root(self) -> Path:
            return self.output_root

        def training_precompute_store(self) -> Journal:
            return self.journal

        def load_precompute(self, **_kwargs: object) -> None:
            return None

        def seal_precompute(self, *, unit_id: str, data_bytes: bytes, **_kwargs: object) -> bytes:
            published.append(unit_id)
            assert data_bytes == f"worker:{unit_id}".encode()
            return data_bytes

    class Executor:
        mode: Exception | None = None

        def __init__(self, *, max_workers: int, **_kwargs: object) -> None:
            worker_counts.append(max_workers)
            self.submitted: list[str] = []

        def submit(
            self,
            _worker: object,
            item: tuple[str, bytes, str],
        ) -> Future[tuple[str, str, bytes, str]]:
            if not self.submitted:
                worker_output_snapshots.append(
                    tuple(sorted(path.name for path in (tmp_path / "output").iterdir()))
                )
            component_id = item[0]
            self.submitted.append(component_id)
            future: Future[tuple[str, str, bytes, str]] = Future()
            if self.mode is not None:
                future.set_exception(self.mode)
            else:
                future.set_result(
                    (
                        component_id,
                        "complete",
                        f"worker:{component_id}".encode(),
                        "",
                    )
                )
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            _ = wait, cancel_futures
            assert prepared_prefixes == [component_ids]
            assert tuple(self.submitted) == component_ids

    preflight = SimpleNamespace(
        config_bytes=b"{}\n",
        phase_windows={},
    )
    admission = SimpleNamespace(
        artifact=SimpleNamespace(
            admitted_symbols=("BTCUSDT",),
            sha256="d" * 64,
        )
    )

    class RootSeal:
        def __init__(self, phase: str, kind: str) -> None:
            self.path = str(tmp_path / f"{phase}-{kind}")
            self._phase = phase
            self._kind = kind

        def to_payload(self) -> dict[str, object]:
            return {"root_id": self._phase, "root_kind": self._kind}

    roots = {
        (phase, kind): RootSeal(phase, kind)
        for phase in ("warmup", "train", "purge", "validation", "embargo")
        for kind in ("raw", "feature")
    }
    sentinel = RuntimeError("stop-after-parent-publication")

    monkeypatch.setattr(alpha_max_runner, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_checkpoint_implementation_inventory", lambda: []
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "_verify_alpha_max_checkpoint_implementation_inventory",
        lambda value: value,
    )
    monkeypatch.setattr(
        alpha_max_runner, "preflight_alpha_max_runtime_contract", lambda _config: preflight
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "seal_alpha_max_contract_manifest",
        lambda _path: SimpleNamespace(
            feature_availability_start_by_symbol={},
            raw_availability_start_by_symbol={},
            feature_availability_end_by_symbol={},
            raw_availability_end_by_symbol={},
        ),
    )
    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_root_validation", lambda *_args, **_kwargs: (roots, ())
    )
    monkeypatch.setattr(
        alpha_max_runner, "_validate_alpha_max_adjacent_feature_roots", lambda *_args: None
    )
    monkeypatch.setattr(
        alpha_max_runner, "_compute_alpha_max_admission_from_seals", lambda **_kwargs: admission
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_current_nodes",
        lambda _preflight: [{"row_id": component_id} for component_id in component_ids],
    )
    monkeypatch.setattr(
        alpha_max_runner, "_validate_admitted_symbols", lambda _preflight, symbols: symbols
    )
    monkeypatch.setattr(
        alpha_max_runner, "build_alpha_max_train_liquidity_buckets", lambda _admission: object()
    )
    monkeypatch.setattr(
        alpha_max_runner, "_require_exact_explicit_path", lambda _path: str(tmp_path / "prior.json")
    )
    monkeypatch.setattr(
        alpha_max_runner, "read_alpha_max_prior_trial_blob_input", lambda _path: b"{}"
    )
    monkeypatch.setattr(alpha_max_runner, "build_alpha_max_trial_ledger", lambda *_args: object())
    monkeypatch.setattr(alpha_max_runner, "_strict_json_object", lambda _payload: {})
    monkeypatch.setattr(alpha_max_runner, "_validated_output_target", lambda _path: None)
    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_prelock_checkpoint_descriptor", lambda **_kwargs: {}
    )
    monkeypatch.setattr(alpha_max_runner, "_AlphaMaxCellCheckpointStore", Store)
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_create_or_resume_run_root",
        lambda output_root, **_kwargs: output_root,
    )

    def manifest_receipt(
        *_args: object, row: dict[str, str], **_kwargs: object
    ) -> AlphaMaxManifestReceipt:
        path = tmp_path / f"{row['row_id']}.json"
        path.write_bytes(b"{}\n")
        receipt, _payload = alpha_max_evidence.read_artifact_bytes(
            path,
            artifact_id="alpha_max_engine_portfolio_manifest",
        )
        return AlphaMaxManifestReceipt(
            row_id=row["row_id"],
            phase="validation_train_fit",
            relative_path=path.name,
            sha256=receipt.sha256,
            byte_count=receipt.byte_count,
            activation_receipt=receipt,
        )

    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_materialize_manifest_receipt",
        manifest_receipt,
    )
    monkeypatch.setattr(alpha_max_runner, "_validate_alpha_max_root_seals", lambda **_kwargs: None)
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_prepare_training_indicator_prefixes",
        lambda *_args, component_ids, **_kwargs: prepared_prefixes.append(component_ids),
    )
    monkeypatch.setattr(alpha_max_runner, "_alpha_max_phase_lookup", lambda *_args: object())
    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_expected_root_sequence", lambda _phase: ("train",)
    )
    parse_calls = 0

    def parse_component(
        *_args: object, **_kwargs: object
    ) -> tuple[tuple[str, ...], tuple[float, ...], object]:
        nonlocal parse_calls
        parse_calls += 1
        if parse_calls % 7 == 0:
            raise sentinel
        return (("2024-01-02",), (0.0,), {"finalization": "exact"})

    monkeypatch.setattr(
        alpha_max_runner, "_alpha_max_training_component_from_checkpoint", parse_component
    )

    kwargs = dict(
        config="config.json",
        contract_manifest="contract.json",
        prior_trial_blob="prior.json",
        exchange="binance",
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoint",
        warmup_raw_root="warmup-raw",
        warmup_feature_root="warmup-feature",
        train_raw_root="train-raw",
        train_feature_root="train-feature",
        purge_raw_root="purge-raw",
        purge_feature_root="purge-feature",
        validation_raw_root="validation-raw",
        validation_feature_root="validation-feature",
        embargo_raw_root="embargo-raw",
        embargo_feature_root="embargo-feature",
    )
    for cap in (1, 2, 3, 4):
        published.clear()
        prepared_prefixes.clear()
        with pytest.raises(RuntimeError, match="stop-after-parent-publication"):
            alpha_max_runner.run_alpha_max_prelock_process(**kwargs, max_training_workers=cap)
        assert published == list(component_ids)
    assert worker_counts == [1, 2, 3, 3]
    assert worker_output_snapshots == [(), (), (), ()]

    prepared_prefixes.clear()
    Executor.mode = AlphaMaxRuntimeContractError("semantic-worker-failure")
    with pytest.raises(AlphaMaxRuntimeContractError, match="semantic-worker-failure"):
        alpha_max_runner.run_alpha_max_prelock_process(**kwargs, max_training_workers=3)
    assert poisoned
    poisoned.clear()
    prepared_prefixes.clear()
    Executor.mode = alpha_max_runner.BrokenProcessPool("abrupt-worker-exit")
    with pytest.raises(alpha_max_runner.BrokenProcessPool, match="abrupt-worker-exit"):
        alpha_max_runner.run_alpha_max_prelock_process(**kwargs, max_training_workers=3)
    assert not poisoned
    Executor.mode = None
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_prepare_training_indicator_prefixes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AlphaMaxRuntimeContractError("semantic-prefix-failure")
        ),
    )
    with pytest.raises(AlphaMaxRuntimeContractError, match="semantic-prefix-failure"):
        alpha_max_runner.run_alpha_max_prelock_process(**kwargs, max_training_workers=3)
    assert poisoned


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
).resolve()


@pytest.fixture(autouse=True)
def _clean_lq_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)


def _nodes() -> list[dict[str, object]]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return payload["current_trial_registry"]["nodes"]


def _owned_root(tmp_path: Path) -> Path:
    root = (tmp_path / "run").resolve()
    (root / "manifests/validation_train_fit").mkdir(parents=True)
    (root / "manifests/prelock_final_refit").mkdir()
    return root


def test_fitted_fixed_weights_use_the_manifest_canonical_precision() -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    nodes = alpha_max_runner._alpha_max_current_nodes(preflight)
    observation_count = 252
    component_returns = MappingProxyType(
        {
            "component_carry_1x": tuple(
                0.001 + (index % 7) * 0.0001 for index in range(observation_count)
            ),
            "component_near_high_1x": tuple(
                -0.0005 + (index % 11) * 0.00015 for index in range(observation_count)
            ),
            "component_trend_1x": tuple(
                0.0002 + ((index * 3) % 13) * 0.00012 for index in range(observation_count)
            ),
        }
    )

    fit = alpha_max_runner._alpha_max_fit_weights(
        nodes,
        phase="train",
        calendar=tuple(f"day-{index:03d}" for index in range(observation_count)),
        component_returns=component_returns,
    )

    assert fit.weights_by_row["full_equal_weight_1x"] == {
        "component_carry_1x": 0.3333333333,
        "component_near_high_1x": 0.3333333333,
        "component_trend_1x": 0.3333333333,
    }


def test_status_matrix_has_84_statuses_68_logical_cells_and_816_fold_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    manifests: dict[str, AlphaMaxManifestReceipt] = {}
    capsules: dict[tuple[str, str], AlphaMaxCapsuleReceipt] = {}

    monkeypatch.setattr(
        "lumina_quant.research.alpha_max_engine_runner.canonical_alpha_max_cost_cell_bytes",
        lambda _value: b"typed-cell\n",
    )

    def bare(cls, **fields):
        value = object.__new__(cls)
        for name, field_value in fields.items():
            object.__setattr__(value, name, field_value)
        return value

    def execute(row, nominal_cost_bps):
        row_id = row["row_id"]
        calls.append((row_id, nominal_cost_bps))
        manifest = manifests.setdefault(
            row_id,
            bare(
                AlphaMaxManifestReceipt,
                sha256=f"{abs(hash(('manifest', row_id))):064x}"[-64:],
            ),
        )
        fold_runs = []
        for index in range(1, 13):
            fold_id = f"validation_w{index:02d}"
            capsule = capsules.setdefault(
                (row_id, fold_id),
                bare(
                    AlphaMaxCapsuleReceipt,
                    prefix_id=fold_id,
                    sha256=f"{abs(hash((row_id, fold_id))):064x}"[-64:],
                ),
            )
            actual = bare(
                AlphaMaxActualEngineRunReceipt,
                split_or_fold_id=fold_id,
                capsule_receipt=capsule,
                manifest_receipt=manifest,
            )
            fold_runs.append(
                bare(
                    AlphaMaxFoldRunEvidence,
                    actual_engine_run=actual,
                )
            )
        pre_gate = bare(
            AlphaMaxCostCellPreGateEvidence,
            fold_runs=tuple(fold_runs),
        )
        return bare(
            AlphaMaxCostCellEvidence,
            row_id=row_id,
            domain="validation",
            nominal_cost_bps=nominal_cost_bps,
            status="complete",
            evidence_tier="actual_engine",
            selection_valid=True,
            pre_gate_evidence=pre_gate,
        )

    result = orchestrate_alpha_max_status_matrix(_nodes(), execute)

    assert len(result.statuses) == 21 * 4
    assert result.engine_cell_count == 17 * 4
    assert len(calls) == 17 * 4
    assert (
        sum(
            len(status.evidence.pre_gate_evidence.fold_runs)
            for status in result.statuses
            if status.engine_constructed
        )
        == 816
    )
    assert {cost for _, cost in calls} == set(ALPHA_MAX_COST_CELL_BPS)
    assert not any(row_id.startswith("incumbent_") for row_id, _ in calls)
    assert not any(row_id.startswith("diagnostic_") for row_id, _ in calls)


def test_production_physical_fold_schedules_are_exact() -> None:
    validation = alpha_max_runner._alpha_max_physical_fold_schedule("validation")
    historical = alpha_max_runner._alpha_max_physical_fold_schedule("historical_exposed_evaluation")

    assert len(validation) == 816
    assert len(historical) == 680
    assert len(set(validation)) == 816
    assert len(set(historical)) == 680
    assert all(
        [fold_id for row, cost, fold_id in validation if row == row_id and cost == nominal]
        == [f"validation_w{index:02d}" for index in range(1, 13)]
        for row_id in alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
    )
    alpha_max_runner._validate_alpha_max_physical_fold_schedule(
        validation,
        domain="validation",
    )
    alpha_max_runner._validate_alpha_max_physical_fold_schedule(
        historical,
        domain="historical_exposed_evaluation",
    )
    for invalid, domain in (
        (validation[:-1], "validation"),
        ((*validation, validation[-1]), "validation"),
        (historical[:-1], "historical_exposed_evaluation"),
        ((*historical, historical[-1]), "historical_exposed_evaluation"),
    ):
        with pytest.raises(
            AlphaMaxRuntimeContractError,
            match="physical_fold_cardinality_mismatch",
        ):
            alpha_max_runner._validate_alpha_max_physical_fold_schedule(
                invalid,
                domain=domain,
            )
    assert all(
        len([fold_id for row, cost, fold_id in historical if row == row_id and cost == nominal])
        == 10
        for row_id in alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS
        for nominal in ALPHA_MAX_COST_CELL_BPS
    )


def test_adjacent_feature_root_gap_is_closed_across_split() -> None:
    boundary = datetime(2025, 1, 1, tzinfo=UTC)
    availability_start = MappingProxyType(
        dict.fromkeys(
            alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS,
            boundary.replace(year=2024),
        )
    )
    availability_end = MappingProxyType(
        dict.fromkeys(
            alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS,
            boundary.replace(year=2026),
        )
    )

    def seal(root_id: str, *, first: int, last: int, contracted: bool = True):
        entries = tuple(
            SimpleNamespace(
                relative_path=f"symbol={symbol}/date=2025-01-01/part.parquet",
                minimum_timestamp_ms=first,
                maximum_timestamp_ms=last,
            )
            for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS
        )
        value = SimpleNamespace(
            root_id=root_id,
            exchange="binance",
            entries=entries,
            start_utc=boundary if root_id == "right" else boundary.replace(year=2024),
            end_utc=boundary.replace(year=2026) if root_id == "right" else boundary,
        )
        if contracted:
            value.availability_start_by_symbol = availability_start
            value.availability_end_by_symbol = availability_end
        return value

    uncontracted = {
        ("left", "feature"): seal("left", first=1, last=10_000, contracted=False),
        ("right", "feature"): seal(
            "right",
            first=10_000 + 28_801_000,
            last=40_000_000,
            contracted=False,
        ),
    }
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="adjacent_feature_root_funding_coverage_incomplete",
    ):
        alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
            uncontracted,
            (("left", "right"),),
        )

    predecessor = seal("left", first=1, last=10_000)
    current = seal("right", first=10_000 + 28_801_000, last=40_000_000)
    seals = {("left", "feature"): predecessor, ("right", "feature"): current}
    alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
        seals,
        (("left", "right"),),
    )

    current.entries = tuple(
        SimpleNamespace(
            relative_path=entry.relative_path,
            minimum_timestamp_ms=10_000 + 28_801_001,
            maximum_timestamp_ms=entry.maximum_timestamp_ms,
        )
        for entry in current.entries
    )
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="adjacent_feature_root_funding_coverage_incomplete",
    ):
        alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
            seals,
            (("left", "right"),),
        )


def test_adjacent_feature_roots_accept_only_a_fresh_declared_availability_end() -> None:
    predecessor_start = datetime(2026, 6, 1, tzinfo=UTC)
    boundary = datetime(2026, 6, 24, tzinfo=UTC)
    current_end = datetime(2026, 7, 1, tzinfo=UTC)
    ton_end = datetime(2026, 6, 23, 9, tzinfo=UTC)
    ton_last_ms = int(datetime(2026, 6, 23, 8, tzinfo=UTC).timestamp() * 1000)
    boundary_ms = int(boundary.timestamp() * 1000)
    availability_start = {
        symbol: (
            datetime(2024, 3, 1, 16, tzinfo=UTC)
            if symbol == "TONUSDT"
            else datetime(2022, 12, 31, tzinfo=UTC)
        )
        for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS
    }
    availability_end = {
        symbol: ton_end if symbol == "TONUSDT" else current_end
        for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS
    }

    def entries(*, predecessor: bool, stale_ton: bool = False) -> tuple[SimpleNamespace, ...]:
        rows: list[SimpleNamespace] = []
        for symbol in alpha_max_runner.ALPHA_MAX_CANDIDATE_SYMBOLS:
            if symbol == "TONUSDT" and not predecessor:
                continue
            timestamp_ms = (
                ton_last_ms - (28_801_001 if stale_ton else 0)
                if symbol == "TONUSDT"
                else boundary_ms - 1
                if predecessor
                else boundary_ms
            )
            rows.append(
                SimpleNamespace(
                    relative_path=f"symbol={symbol}/date=2026-06-23/part.parquet",
                    minimum_timestamp_ms=timestamp_ms,
                    maximum_timestamp_ms=timestamp_ms,
                )
            )
        return tuple(rows)

    predecessor = SimpleNamespace(
        exchange="binance",
        entries=entries(predecessor=True),
        start_utc=predecessor_start,
        end_utc=boundary,
        availability_start_by_symbol=availability_start,
        availability_end_by_symbol=availability_end,
    )
    current = SimpleNamespace(
        exchange="binance",
        entries=entries(predecessor=False),
        start_utc=boundary,
        end_utc=current_end,
        availability_start_by_symbol=availability_start,
        availability_end_by_symbol=availability_end,
    )
    seals = {("left", "feature"): predecessor, ("right", "feature"): current}
    alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
        seals,
        (("left", "right"),),
    )

    predecessor.entries = entries(predecessor=True, stale_ton=True)
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="adjacent_feature_root_funding_coverage_incomplete",
    ):
        alpha_max_runner._validate_alpha_max_adjacent_feature_roots(
            seals,
            (("left", "right"),),
        )


def test_runtime_read_audit_keeps_different_daily_paths_in_chronological_order() -> None:
    first = ("START_DATE", "END_DATE", "SYMBOLS")
    second = ("START_DATE", "SLIPPAGE_RATE", "END_DATE")

    retained, first_sha = alpha_max_runner._alpha_max_append_runtime_read_audit(
        None,
        first,
    )
    combined, combined_sha = alpha_max_runner._alpha_max_append_runtime_read_audit(
        retained,
        second,
    )

    assert retained == first
    assert combined == (*first, *second)
    assert first_sha != combined_sha
    assert combined_sha == alpha_max_runner._sha256(
        alpha_max_runner._canonical_bytes(list(combined))
    )
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="runtime_read_audit_invalid",
    ):
        alpha_max_runner._alpha_max_append_runtime_read_audit(
            combined,
            ("UNSEALED_FIELD",),
        )


def test_run_owned_root_writes_sealed_last_and_never_seals_mismatched_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "sealed-last").resolve()
    root = alpha_max_runner._create_alpha_max_run_owned_root(output)
    writes: list[str] = []
    original_write = alpha_max_runner._write_bundle_file
    original_atomic_write = alpha_max_runner._write_bundle_file_atomic
    original_final_seal = alpha_max_runner._alpha_max_write_final_seal

    def recording_atomic_write(bundle_root, relative_path, payload):
        written = original_atomic_write(bundle_root, relative_path, payload)
        writes.append(relative_path)
        return written

    monkeypatch.setattr(
        alpha_max_runner,
        "_write_bundle_file_atomic",
        recording_atomic_write,
    )

    def recording_final_seal(seal_fd, payload):
        result = original_final_seal(seal_fd, payload)
        writes.append("SEALED.json")
        return result

    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_write_final_seal",
        recording_final_seal,
    )
    bundle = alpha_max_runner._finalize_alpha_max_run_owned_root(
        root,
        {"report/result.json": b"{}\n"},
        seal_bytes=b'{"sealed":true}\n',
    )
    assert writes[-1] == "SEALED.json"
    assert Path(bundle.seal_path).read_bytes() == b'{"sealed":true}\n'

    mismatch_root = alpha_max_runner._create_alpha_max_run_owned_root(
        (tmp_path / "mismatch").resolve()
    )
    original_write(mismatch_root, "report/result.json", b"first\n")
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="run_owned_artifact_mismatch",
    ):
        alpha_max_runner._finalize_alpha_max_run_owned_root(
            mismatch_root,
            {"report/result.json": b"second\n"},
            seal_bytes=b'{"sealed":true}\n',
        )
    assert not (mismatch_root / "SEALED.json").exists()
    alpha_max_runner._cleanup_partial_bundle(mismatch_root)
    assert not mismatch_root.exists()


def test_bundle_immutability_and_cleanup_never_follow_external_symlinks(
    tmp_path: Path,
) -> None:
    external = (tmp_path / "external").resolve()
    external.mkdir(mode=0o700)
    external_file = external / "evidence.txt"
    external_file.write_bytes(b"must remain untouched\n")
    external_file.chmod(0o640)
    external_mode = external.stat().st_mode & 0o777
    external_file_mode = external_file.stat().st_mode & 0o777

    root = (tmp_path / "bundle").resolve()
    root.mkdir(mode=0o700)
    (root / "artifact.json").write_bytes(b"{}\n")
    (root / "escape").symlink_to(external, target_is_directory=True)

    with pytest.raises(AlphaMaxRuntimeContractError, match="bundle_tree_invalid"):
        alpha_max_runner._make_bundle_immutable(root)
    assert external.stat().st_mode & 0o777 == external_mode
    assert external_file.stat().st_mode & 0o777 == external_file_mode
    assert external_file.read_bytes() == b"must remain untouched\n"

    alpha_max_runner._cleanup_partial_bundle(root)
    assert not root.exists()
    assert external.stat().st_mode & 0o777 == external_mode
    assert external_file.stat().st_mode & 0o777 == external_file_mode
    assert external_file.read_bytes() == b"must remain untouched\n"


def test_cleanup_and_claim_release_unlink_hostile_links_without_chmodding_target(
    tmp_path: Path,
) -> None:
    external = (tmp_path / "external-claim.txt").resolve()
    external.write_bytes(b"external claim target\n")
    external.chmod(0o640)
    original_mode = external.stat().st_mode & 0o777

    partial = (tmp_path / "partial").resolve()
    partial.mkdir()
    os.link(external, partial / "hard-linked-artifact")
    (partial / "symlinked-artifact").symlink_to(external)
    alpha_max_runner._cleanup_partial_bundle(partial)
    assert not partial.exists()
    assert external.read_bytes() == b"external claim target\n"
    assert external.stat().st_mode & 0o777 == original_mode

    claim = (tmp_path / ".completion.claim").resolve()
    claim.symlink_to(external)
    alpha_max_runner._release_historical_completion_claim(claim)
    assert not claim.exists()
    assert external.read_bytes() == b"external claim target\n"
    assert external.stat().st_mode & 0o777 == original_mode


def test_bundle_write_is_bound_to_opened_root_when_parent_path_is_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "owned").resolve()
    nested = root / "nested"
    nested.mkdir(parents=True)
    moved = root / "nested-opened"
    external = (tmp_path / "external").resolve()
    external.mkdir()
    original_open = os.open
    swapped = False

    def hostile_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "artifact.json" and dir_fd is not None and not swapped:
            swapped = True
            nested.rename(moved)
            nested.symlink_to(external, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", hostile_open)
    alpha_max_runner._write_bundle_file(
        root,
        "nested/artifact.json",
        b'{"owned":true}\n',
    )

    assert swapped is True
    assert not (external / "artifact.json").exists()
    assert (moved / "artifact.json").read_bytes() == b'{"owned":true}\n'


def test_manifest_seal_rejects_hard_linked_target(tmp_path: Path) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    seal = seal_alpha_max_manifest_activation(
        preflight,
        output_root=str(root),
        phase="validation_train_fit",
        manifest_path=materialized.path,
        admitted_symbols=admitted,
    )
    assert seal.manifest_receipt.sha256 == materialized.sha256

    target = Path(materialized.path)
    linked = target.with_suffix(".linked")
    os.link(target, linked)
    with pytest.raises(
        AlphaMaxRuntimeContractError, match="portfolio_manifest_activation_mismatch"
    ):
        seal_alpha_max_manifest_activation(
            preflight,
            output_root=str(root),
            phase="validation_train_fit",
            manifest_path=materialized.path,
            admitted_symbols=admitted,
        )


def test_actual_backtest_construction_binds_all_alpha_runtime_seams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    purge_feature_root = (tmp_path / "features-purge").resolve()
    validation_feature_root = (tmp_path / "features-validation").resolve()
    purge_feature_root.mkdir()
    validation_feature_root.mkdir()
    purge_window = preflight.phase_windows["purge"]
    validation_window = preflight.phase_windows["validation"]
    lookup = AlphaMaxOrderedFundingLookup(
        (
            FeatureRootSpec(
                "purge",
                str(purge_feature_root),
                "binance",
                purge_window.start_utc,
                purge_window.end_utc,
                "a" * 64,
                "b" * 64,
            ),
            FeatureRootSpec(
                "validation",
                str(validation_feature_root),
                "binance",
                validation_window.start_utc,
                validation_window.end_utc,
                "c" * 64,
                "d" * 64,
            ),
        )
    )
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted)
    timestamp_ms = int(datetime(2025, 6, 8, tzinfo=UTC).timestamp() * 1000)
    data_dict = dict.fromkeys(admitted, ((timestamp_ms, 100.0, 101.0, 99.0, 100.0, 10.0),))

    activation = construct_alpha_max_engine(
        preflight,
        output_root=str(root),
        phase="validation_train_fit",
        manifest_path=materialized.path,
        admitted_symbols=admitted,
        phase_id="validation",
        nominal_cost_bps=30,
        raw_root=str((tmp_path / "raw-validation").resolve()),
        ordered_lookup=lookup,
        funding_resolver=resolver,
        data_dict=data_dict,
    )

    validate_alpha_max_engine_activation(activation)
    assert type(activation.backtest) is Backtest
    assert type(activation.backtest.data_handler) is HistoricParquetWindowedDataHandler
    assert type(activation.backtest.strategy) is ArtifactPortfolioModeStrategy
    assert type(activation.backtest.portfolio) is Portfolio
    assert type(activation.backtest.execution_handler) is SimulatedExecutionHandler
    assert activation.backtest.symbol_list is admitted
    assert activation.backtest.data_handler.symbol_list is admitted
    assert activation.backtest.portfolio.symbol_list is admitted
    assert set(activation.constructor_plan.portfolio_kwargs) == {
        "fill_application_attribution_sink",
        "funding_boundary_resolver",
        "full_event_equity_sink",
        "reporting_sampling_timeframe",
    }
    fill_sink = activation.backtest.portfolio.fill_application_attribution_sink
    equity_sink = activation.backtest.portfolio.full_event_equity_sink
    assert fill_sink.__self__ is activation.attribution_collector
    assert fill_sink.__func__ is AlphaMaxAttributionCollector.record_application
    assert equity_sink.__self__ is activation.full_event_equity_tracker
    assert activation.backtest.portfolio.reporting_sampling_timeframe == "4h"
    assert activation.backtest.execution_handler.pricing_trace_evidence == ()

    # A crossed reporting boundary is priced from the completed native 4h
    # bucket even when sparse raw data has no row at the exact boundary and a
    # later row carries a hostile close.  The inclusive final endpoint settles
    # funding exactly once before it is emitted.
    reporting_start = datetime(2025, 6, 8, tzinfo=UTC)
    reporting_end = reporting_start + timedelta(hours=8)
    completed_close = 101.0
    hostile_later_close = 999.0
    aggregator = alpha_max_runner.TimeframeAggregator(timeframes=["4h"])
    for symbol in admitted:
        aggregator.update_from_1s_batch(
            symbol,
            (
                (
                    int((reporting_start + timedelta(seconds=5)).timestamp() * 1000),
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    1.0,
                ),
                (
                    int(
                        (reporting_start + timedelta(hours=3, minutes=59, seconds=55)).timestamp()
                        * 1000
                    ),
                    completed_close,
                    completed_close,
                    completed_close,
                    completed_close,
                    1.0,
                ),
                (
                    int((reporting_start + timedelta(hours=4, seconds=5)).timestamp() * 1000),
                    hostile_later_close,
                    hostile_later_close,
                    hostile_later_close,
                    hostile_later_close,
                    1.0,
                ),
            ),
        )
    # Symbols with no position do not require a price.  A newly listed symbol
    # can legitimately have no completed native bar at an early fold boundary.
    aggregator_state = aggregator.get_state()
    aggregator_state["history"].pop(admitted[-1])
    aggregator_state["working"].pop(admitted[-1])
    aggregator.set_state(aggregator_state)
    activation.backtest.timeframe_aggregator = aggregator
    activation.backtest.portfolio.current_holdings["cash"] = 1_000.0
    activation.backtest.portfolio.current_positions[admitted[0]] = 2.0
    funding_boundaries: list[datetime] = []

    def record_funding(_portfolio: Portfolio, boundary: datetime) -> None:
        funding_boundaries.append(boundary)

    monkeypatch.setattr(Portfolio, "_apply_funding", record_funding)
    fanout = alpha_max_runner._AlphaMaxFoldEquityFanout(
        AlphaMaxStreamingEquityTracker(),
        aggregate_scale=1.0,
        reporting_start=reporting_start,
        reporting_end=reporting_end,
    )
    fanout.bind_backtest(activation.backtest)
    fanout.observe(
        (
            (reporting_start + timedelta(hours=4, seconds=5)).timestamp(),
            12_345.0,
        )
    )
    fanout.settle_day_end(reporting_end, settle_funding=True)

    endpoints = fanout.reporting_endpoints
    assert endpoints[0].equity == pytest.approx(1_000.0 + (2.0 * completed_close))
    assert endpoints[0].equity != pytest.approx(1_000.0 + (2.0 * hostile_later_close))
    assert funding_boundaries == [reporting_end]
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="reporting_boundary_sequence_invalid",
    ):
        fanout.settle_day_end(reporting_end, settle_funding=True)
    assert funding_boundaries == [reporting_end]


def test_reporting_boundary_carries_last_completed_native_close_across_sparse_bucket() -> None:
    symbol = "BTCUSDT"
    reporting_start = datetime(2025, 6, 8, tzinfo=UTC)
    completed_close = 101.0
    hostile_later_close = 999.0
    aggregator = alpha_max_runner.TimeframeAggregator(timeframes=["4h"])
    aggregator.update_from_1s_batch(
        symbol,
        (
            (
                int(
                    (reporting_start - timedelta(hours=4) + timedelta(seconds=5)).timestamp() * 1000
                ),
                completed_close,
                completed_close,
                completed_close,
                completed_close,
                1.0,
            ),
            (
                int((reporting_start + timedelta(hours=4, seconds=5)).timestamp() * 1000),
                hostile_later_close,
                hostile_later_close,
                hostile_later_close,
                hostile_later_close,
                1.0,
            ),
        ),
    )
    tracker = object.__new__(alpha_max_runner._AlphaMaxFoldEquityFanout)
    tracker._backtest = SimpleNamespace(timeframe_aggregator=aggregator)

    assert tracker._completed_native_close(
        symbol,
        int((reporting_start + timedelta(hours=4)).timestamp() * 1000),
    ) == pytest.approx(completed_close)


def test_exact_tick_reducer_includes_reporting_timeframe() -> None:
    backtest = SimpleNamespace(
        strategy=SimpleNamespace(required_timeframes=("20s", "1m")),
        portfolio=SimpleNamespace(reporting_sampling_timeframe="4h"),
    )

    assert alpha_max_runner._alpha_max_exact_tick_timeframes(backtest) == (
        "20s",
        "1m",
        "4h",
    )


def test_alpha_max_fold_equity_fanout_batch_matches_pointwise_reference() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=8)
    points = np.array(
        [
            [start.timestamp() + 1.0, 10_000.0],
            [start.timestamp() + 2.0, 9_500.0],
            [start.timestamp() + 3.0, 10_250.0],
        ],
        dtype=np.float64,
    )
    aggregate_reference = AlphaMaxStreamingEquityTracker()
    pointwise = alpha_max_runner._AlphaMaxFoldEquityFanout(
        aggregate_reference,
        aggregate_scale=1.25,
        reporting_start=start,
        reporting_end=end,
    )
    for point in points:
        pointwise.observe((float(point[0]), float(point[1])))

    aggregate_batch = AlphaMaxStreamingEquityTracker()
    batched = alpha_max_runner._AlphaMaxFoldEquityFanout(
        aggregate_batch,
        aggregate_scale=1.25,
        reporting_start=start,
        reporting_end=end,
    )
    batched.update_batch(points)

    assert batched.finalize().to_payload() == pointwise.finalize().to_payload()
    assert (
        batched.normalized_segment_tracker.finalize().to_payload()
        == pointwise.normalized_segment_tracker.finalize().to_payload()
    )
    assert aggregate_batch.finalize().to_payload() == aggregate_reference.finalize().to_payload()


def test_exact_indicator_loader_consumes_only_requested_subwindow() -> None:
    symbol = "ADAUSDT"
    parent_start = datetime(2025, 6, 1, tzinfo=UTC)
    parent_end = datetime(2025, 7, 1, tzinfo=UTC)
    start = datetime(2025, 6, 8, tzinfo=UTC)
    end = start + timedelta(hours=8)
    timestamps = pl.datetime_range(
        start,
        end - timedelta(seconds=1),
        interval="1s",
        eager=True,
    )
    frame = pl.DataFrame(
        {
            "datetime": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.0] * len(timestamps),
            "volume": [0.1] * len(timestamps),
        }
    )

    def entry(relative_path: str) -> AlphaMaxTreeEntry:
        return AlphaMaxTreeEntry(
            relative_path=relative_path,
            byte_count=1,
            mode=0o444,
            mtime_ns=1,
            minimum_timestamp_ms=int(parent_start.timestamp() * 1000),
            maximum_timestamp_ms=int((parent_end - timedelta(seconds=1)).timestamp() * 1000),
            row_count=1,
            maximum_gap_ms=0,
            sha256="a" * 64,
        )

    june = entry("raw/ADAUSDT/2025-06.parquet")
    may = entry("raw/ADAUSDT/2025-05.parquet")

    class Reader:
        def __init__(self):
            self.paths = []

        def read_entry(self, value):
            self.paths.append(value.relative_path)
            if value is not june:
                raise AssertionError("out-of-window partition was read")
            return frame

        def close(self):
            return None

    loader = object.__new__(alpha_max_runner._AlphaMaxBoundedRawLoader)
    loader._seal = SimpleNamespace(
        start_utc=parent_start,
        end_utc=parent_end,
        availability_start_by_symbol={symbol: parent_start},
        availability_end_by_symbol={symbol: parent_end},
    )
    loader._admitted_symbols = (symbol,)
    loader._entries = MappingProxyType(
        {
            (symbol, "2025-05.parquet"): may,
            (symbol, "2025-06.parquet"): june,
        }
    )
    loader._frame_cache = {}
    loader._reader = Reader()
    releases, windows = loader.fold_exact_indicator_phase(
        alpha_max_runner.TimeframeAggregator(timeframes=["4h"]),
        start=start,
        end=end,
    )

    assert windows == 28_800
    assert len(releases) == 1
    assert loader._reader.paths == [june.relative_path]
    assert loader._read_entry_cached(symbol, june) is frame
    assert loader._reader.paths == [june.relative_path]


def test_bounded_raw_loader_reuses_authenticated_month_frame_across_days() -> None:
    symbol = "ADAUSDT"
    start = datetime(2025, 6, 1, tzinfo=UTC)
    end = start + timedelta(days=2)
    timestamps = [start + timedelta(hours=4 * offset) for offset in range(12)]
    frame = pl.DataFrame(
        {
            "datetime": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.0] * len(timestamps),
            "volume": [0.1] * len(timestamps),
        }
    )
    entry = AlphaMaxTreeEntry(
        relative_path="raw/ADAUSDT/2025-06.parquet",
        byte_count=1,
        mode=0o444,
        mtime_ns=1,
        minimum_timestamp_ms=int(start.timestamp() * 1000),
        maximum_timestamp_ms=int((end - timedelta(seconds=1)).timestamp() * 1000),
        row_count=frame.height,
        maximum_gap_ms=14_400_000,
        sha256="a" * 64,
    )

    class Reader:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def read_entry(self, value: AlphaMaxTreeEntry) -> pl.DataFrame:
            self.paths.append(value.relative_path)
            return frame

        def close(self) -> None:
            return None

    loader = object.__new__(alpha_max_runner._AlphaMaxBoundedRawLoader)
    loader._seal = SimpleNamespace(start_utc=start, end_utc=end)
    loader._admitted_symbols = (symbol,)
    loader._entries = MappingProxyType({(symbol, "2025-06.parquet"): entry})
    loader._frame_cache = {}
    loader._reader = Reader()

    first = loader.load_day(start, start + timedelta(days=1))[symbol]
    second = loader.load_day(start + timedelta(days=1), end)[symbol]

    assert first.height == second.height == 6
    assert first.get_column("datetime")[0] == start
    assert second.get_column("datetime")[0] == start + timedelta(days=1)
    assert loader._reader.paths == [entry.relative_path]


def test_bounded_raw_loader_close_releases_cached_month_frames() -> None:
    class Reader:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    loader = object.__new__(alpha_max_runner._AlphaMaxBoundedRawLoader)
    loader._frame_cache = {"ADAUSDT": ("2025-06.parquet", pl.DataFrame({"close": [100.0]}))}
    reader = Reader()
    loader._reader = reader

    loader.close()

    assert reader.closed is True
    assert loader._frame_cache == {}
    assert loader._reader is None
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_bounded_raw_loader_closed",
    ):
        loader._read_entry_cached("ADAUSDT", object())


def _assert_bit_exact(actual: object, expected: object) -> None:
    if isinstance(expected, float):
        assert type(actual) is float
        assert struct.pack("!d", actual) == struct.pack("!d", expected)
    elif isinstance(expected, dict):
        assert type(actual) is dict
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_bit_exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_bit_exact(actual_item, expected_item)
    else:
        assert actual == expected


def _exact_loader(
    *,
    admitted: tuple[str, ...],
    start: datetime,
    end: datetime,
    frames: dict[str, pl.DataFrame],
    entry_order: tuple[str, ...],
) -> alpha_max_runner._AlphaMaxBoundedRawLoader:
    entries = {
        path: AlphaMaxTreeEntry(
            relative_path=path,
            byte_count=1,
            mode=0o444,
            mtime_ns=1,
            minimum_timestamp_ms=int(start.timestamp() * 1000),
            maximum_timestamp_ms=int((end - timedelta(seconds=1)).timestamp() * 1000),
            row_count=1,
            maximum_gap_ms=0,
            sha256="a" * 64,
        )
        for path in entry_order
    }

    class Reader:
        def read_entry(self, entry: AlphaMaxTreeEntry) -> pl.DataFrame:
            return frames[entry.relative_path]

        def close(self) -> None:
            return None

    loader = object.__new__(alpha_max_runner._AlphaMaxBoundedRawLoader)
    loader._seal = SimpleNamespace(
        start_utc=start - timedelta(days=31),
        end_utc=end + timedelta(days=31),
        availability_start_by_symbol=dict.fromkeys(admitted, start - timedelta(days=31)),
        availability_end_by_symbol=dict.fromkeys(admitted, end + timedelta(days=31)),
    )
    loader._admitted_symbols = admitted
    loader._entries = MappingProxyType(
        {(path.split("/")[1], Path(path).name): entries[path] for path in entry_order}
    )
    loader._frame_cache = {}
    loader._reader = Reader()
    return loader


def _canonical_rows(frame: pl.DataFrame, start: datetime, end: datetime):
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    scoped = (
        frame.with_columns(pl.col("datetime").dt.epoch("ms").alias("timestamp"))
        .filter((pl.col("timestamp") >= start_ms) & (pl.col("timestamp") < end_ms))
        .sort("timestamp")
    )
    return tuple(
        (int(timestamp), float(open_), float(high), float(low), float(close), float(volume))
        for timestamp, open_, high, low, close, volume in scoped.select(
            ("timestamp", "open", "high", "low", "close", "volume")
        ).iter_rows()
    )


def test_exact_indicator_loader_is_bit_exact_for_dense_multisymbol_4h_and_1d_releases() -> None:
    start = datetime(2025, 6, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    admitted = ("ADAUSDT", "BTCUSDT")
    timestamps = pl.datetime_range(start, end - timedelta(seconds=1), "1s", eager=True)
    frames: dict[str, pl.DataFrame] = {}
    paths: list[str] = []
    for symbol_index, symbol in enumerate(admitted):
        values = np.arange(len(timestamps), dtype=np.float64)
        close = 100.0 + symbol_index + (values % 997.0) / 997.0
        path = f"raw/{symbol}/2025-06.parquet"
        paths.append(path)
        frames[path] = pl.DataFrame(
            {
                "datetime": timestamps,
                "open": close - 0.125,
                "high": close + 0.25,
                "low": close - 0.5,
                "close": close,
                "volume": (values % 13.0) / 10.0,
            }
        )

    seeded = alpha_max_runner.TimeframeAggregator(timeframes=["4h", "1d"])
    for symbol in admitted:
        seeded.update_from_canonical_1s_rows_exact(
            symbol,
            (
                (
                    int((start - timedelta(seconds=1)).timestamp() * 1000),
                    99.0,
                    100.0,
                    98.0,
                    99.5,
                    0.0,
                ),
            ),
        )
    initial_state = seeded.get_state()
    expected = alpha_max_runner.TimeframeAggregator(timeframes=["4h", "1d"])
    expected.set_state(initial_state)
    expected_releases = tuple(
        release
        for symbol in admitted
        for release in expected.update_from_canonical_1s_rows_exact(
            symbol, _canonical_rows(frames[f"raw/{symbol}/2025-06.parquet"], start, end)
        )
    )
    actual = alpha_max_runner.TimeframeAggregator(timeframes=["4h", "1d"])
    actual.set_state(initial_state)
    releases, windows = _exact_loader(
        admitted=admitted, start=start, end=end, frames=frames, entry_order=tuple(reversed(paths))
    ).fold_exact_indicator_phase(actual, start=start, end=end)

    assert windows == 86_400
    _assert_bit_exact(releases, expected_releases)
    _assert_bit_exact(actual.get_state(), expected.get_state())
    _assert_bit_exact(
        actual.get_state()["history"]["ADAUSDT"]["1s"],
        expected.get_state()["history"]["ADAUSDT"]["1s"],
    )
    alpha_max_runner._alpha_max_validate_final_native_working_bars(
        actual,
        admitted_symbols=admitted,
        required_timeframes=("4h", "1d"),
        end_ms=int(end.timestamp() * 1000),
    )


def test_exact_indicator_loader_matches_sparse_partitioned_subwindows_and_prior_state() -> None:
    start = datetime(2025, 6, 30, 20, tzinfo=UTC)
    end = datetime(2025, 7, 1, 8, tzinfo=UTC)
    admitted = ("ADAUSDT", "BTCUSDT")
    offsets = (-1, 60, 14_399, 14_401, 28_799, 28_801, 43_199, 43_201, 57_599, 57_601)
    frames: dict[str, pl.DataFrame] = {}
    paths: list[str] = []
    for symbol_index, symbol in enumerate(admitted):
        rows = [start + timedelta(seconds=offset) for offset in offsets]
        close = [100.0 + symbol_index + index / 10.0 for index in range(len(rows))]
        for month in ("2025-06", "2025-07"):
            path = f"raw/{symbol}/{month}.parquet"
            paths.append(path)
            frames[path] = pl.DataFrame(
                {
                    "datetime": rows,
                    "open": [value - 0.1 for value in close],
                    "high": [value + 0.2 for value in close],
                    "low": [value - 0.3 for value in close],
                    "close": close,
                    "volume": [0.0 if index % 2 else 0.25 for index in range(len(rows))],
                }
            ).filter(pl.col("datetime").dt.strftime("%Y-%m") == month)

    prior_time = start - timedelta(seconds=1)
    prior = (int(prior_time.timestamp() * 1000), 90.0, 91.0, 89.0, 90.5, 0.0)
    expected = alpha_max_runner.TimeframeAggregator(timeframes=["4h", "1d"])
    actual = alpha_max_runner.TimeframeAggregator(timeframes=["4h", "1d"])
    for symbol in admitted:
        expected.update_from_canonical_1s_rows_exact(symbol, (prior,))
    actual.set_state(expected.get_state())
    expected_releases = tuple(
        release
        for symbol in admitted
        for month in ("2025-06", "2025-07")
        for release in expected.update_from_canonical_1s_rows_exact(
            symbol, _canonical_rows(frames[f"raw/{symbol}/{month}.parquet"], start, end)
        )
    )
    loader = _exact_loader(
        admitted=admitted, start=start, end=end, frames=frames, entry_order=tuple(reversed(paths))
    )
    releases, windows = loader.fold_exact_indicator_phase(actual, start=start, end=end)

    assert windows == 43_200
    _assert_bit_exact(releases, expected_releases)
    _assert_bit_exact(actual.get_state(), expected.get_state())
    for symbol in admitted:
        _assert_bit_exact(
            actual.get_state()["history"][symbol]["1s"],
            expected.get_state()["history"][symbol]["1s"],
        )


def test_native_indicator_fold_releases_simultaneous_boundaries_in_trigger_order() -> None:
    start = datetime(2025, 6, 30, 23, 59, 59, tzinfo=UTC)
    end = start + timedelta(seconds=3)
    symbol = "ADAUSDT"
    rows = [start + timedelta(seconds=offset) for offset in range(3)]
    frames = {
        "raw/ADAUSDT/2025-06.parquet": pl.DataFrame(
            {
                "datetime": rows[:1],
                "open": [10.0],
                "high": [10.0],
                "low": [10.0],
                "close": [10.0],
                "volume": [1.0e16],
            }
        ),
        "raw/ADAUSDT/2025-07.parquet": pl.DataFrame(
            {
                "datetime": rows[1:],
                "open": [11.0, 12.0],
                "high": [11.0, 12.0],
                "low": [11.0, 12.0],
                "close": [11.0, 12.0],
                "volume": [1.0, 1.0e-16],
            }
        ),
    }
    prior = (
        int((start - timedelta(seconds=1)).timestamp() * 1000),
        9.0,
        9.0,
        9.0,
        9.0,
        1.0e16,
    )
    expected = alpha_max_runner.TimeframeAggregator(timeframes=["4h", "1d"])
    actual = alpha_max_runner.TimeframeAggregator(timeframes=["4h", "1d"])
    expected.update_from_canonical_1s_rows_exact(symbol, (prior,))
    actual.set_state(expected.get_state())
    expected_releases = tuple(
        release
        for path in ("raw/ADAUSDT/2025-06.parquet", "raw/ADAUSDT/2025-07.parquet")
        for release in expected.update_from_canonical_1s_rows_exact(
            symbol, _canonical_rows(frames[path], start, end)
        )
    )
    releases, windows = _exact_loader(
        admitted=(symbol,),
        start=start,
        end=end,
        frames=frames,
        entry_order=tuple(reversed(tuple(frames))),
    ).fold_exact_indicator_phase(actual, start=start, end=end)

    assert windows == 3
    assert tuple(release.release_timestamp_ms for release in releases) == (
        int(rows[1].timestamp() * 1000),
        int(rows[1].timestamp() * 1000),
    )
    assert tuple(release.timeframe for release in releases) == ("4h", "1d")
    _assert_bit_exact(releases, expected_releases)
    _assert_bit_exact(actual.get_state(), expected.get_state())
    _assert_bit_exact(
        actual.get_state()["history"][symbol]["1s"],
        expected.get_state()["history"][symbol]["1s"],
    )


def test_native_indicator_fold_rejects_malformed_arrays_and_state() -> None:
    from lumina_quant import _compute

    fold = _compute.fold_alpha_max_native_bars
    common = (
        np.array([1_000], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([14_400_000], dtype=np.int64),
        -1,
        np.array([0], dtype=np.uint8),
        np.array([0], dtype=np.int64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="timestamps_ms"):
        fold(np.array([1_001], dtype=np.int64), *common[1:])
    with pytest.raises(ValueError, match="inactive working state"):
        fold(
            *common[:8], np.array([0], dtype=np.uint8), np.array([1], dtype=np.int64), *common[10:]
        )
    with pytest.raises(ValueError, match="timeframe_ms"):
        fold(
            *common[:6],
            np.array([14_400_000, 14_400_000], dtype=np.int64),
            common[7],
            np.array([0, 0], dtype=np.uint8),
            np.array([0, 0], dtype=np.int64),
            *[np.array([0.0, 0.0], dtype=np.float64) for _ in range(5)],
        )
    with pytest.raises(ValueError, match="realistic"):
        fold(
            np.array([100_000_001_000], dtype=np.int64),
            *common[1:7],
            0,
            np.array([1], dtype=np.uint8),
            np.array([0], dtype=np.int64),
            *[np.array([1.0], dtype=np.float64) for _ in range(5)],
        )


def test_exact_native_day_restart_matches_uninterrupted_capsule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=2)
    admitted = ("BTCUSDT",)
    descriptor = {"artifact_kind": "instrumented-day-checkpoint"}
    descriptor_bytes = alpha_max_runner._canonical_bytes(descriptor) + b"\n"
    finalizations: list[int] = []

    class FakeLookup:
        ordered_root_ids = ("warmup",)

    class FakeAggregator:
        def __init__(self, *, timeframes: list[str]) -> None:
            assert timeframes == ["4h", "1d"]
            self.state = {"day_count": 0, "history": {}}

        def get_state(self) -> dict[str, object]:
            return dict(self.state)

        def set_state(self, value: dict[str, object]) -> None:
            self.state = dict(value)

    class FakeLoader:
        seal = SimpleNamespace(
            path=str((tmp_path / "raw").resolve()),
            root_id="warmup",
            symbols=ALPHA_MAX_CANDIDATE_SYMBOLS,
        )

        def fold_exact_indicator_phase(
            self,
            aggregator: FakeAggregator,
            *,
            start: datetime,
            end: datetime,
        ) -> tuple[tuple[object, ...], int]:
            assert end - start == timedelta(days=1)
            day_index = (start - globals_start).days
            releases: list[object] = []
            four_hour_offsets = range(4, 24, 4) if day_index == 0 else range(0, 24, 4)
            for hour in four_hour_offsets:
                timestamp = start + timedelta(hours=hour)
                releases.append(
                    alpha_max_runner.NativeBarRelease(
                        release_timestamp_ms=int(timestamp.timestamp() * 1000),
                        symbol=admitted[0],
                        timeframe="4h",
                        bar=(timestamp, 1.0, 1.0, 1.0, 1.0, 1.0),
                    )
                )
            if day_index:
                releases.append(
                    alpha_max_runner.NativeBarRelease(
                        release_timestamp_ms=int(start.timestamp() * 1000),
                        symbol=admitted[0],
                        timeframe="1d",
                        bar=(start, 1.0, 1.0, 1.0, 1.0, 1.0),
                    )
                )
            aggregator.state["day_count"] = day_index + 1
            return tuple(releases), 86_400

    class FakeStrategy:
        def __init__(
            self,
            bars: object,
            events: object,
            *,
            portfolio_mode: str,
            decision_cadence_seconds: int,
        ) -> None:
            assert portfolio_mode == "instrumented"
            assert decision_cadence_seconds == 1
            self.required_timeframes = ("4h", "1d")
            self._children: list[tuple[object, object, object]] = []
            self.state = {"handoffs": 0, "release_groups": 0}

        def get_state(self) -> dict[str, int]:
            return dict(self.state)

        def set_state(self, value: dict[str, int]) -> None:
            self.state = dict(value)

        def calculate_signals_completed_native_release(self, **_kwargs: object) -> None:
            self.state["release_groups"] += 1

        def calculate_signals_context(self, _context: object) -> None:
            self.state["handoffs"] += 1

        def validate_research_warmup_ready(self) -> None:
            return None

        def get_research_indicator_state(self) -> dict[str, object]:
            value: dict[str, object] = {"state": dict(self.state)}
            value["sha256"] = hashlib.sha256(alpha_max_runner._canonical_bytes(value)).hexdigest()
            return value

    class FakeStore:
        def __init__(
            self,
            journal: dict[str, object],
            *,
            crash_after_seal: bool,
        ) -> None:
            self.root = (tmp_path / "checkpoint").resolve()
            self._descriptor_bytes = descriptor_bytes
            self.journal = journal
            self.crash_after_seal = crash_after_seal

        def load_latest(self, **_kwargs: object) -> object | None:
            return self.journal.get("carry")

        def seal(self, carry: object) -> None:
            self.journal["carry"] = carry
            if self.crash_after_seal:
                raise RuntimeError("instrumented-crash-after-day-rename")

    globals_start = start
    component = SimpleNamespace(
        component_id="component",
        strategy_class="ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
        symbols=admitted,
    )
    seal = SimpleNamespace(
        expected_definition=SimpleNamespace(
            admitted_symbols=admitted,
            components=(component,),
            native_timeframes=("4h", "1d"),
            portfolio_mode="instrumented",
        ),
        manifest_receipt=SimpleNamespace(sha256="a" * 64),
    )
    preflight = SimpleNamespace(
        phase_windows={"warmup": SimpleNamespace()},
    )
    config = SimpleNamespace(
        START_DATE=start.isoformat().replace("+00:00", "Z"),
        END_DATE=end.isoformat().replace("+00:00", "Z"),
    )
    candidate_identity = {"candidate": "instrumented"}

    monkeypatch.setattr(alpha_max_runner, "AlphaMaxOrderedFundingLookup", FakeLookup)
    monkeypatch.setattr(alpha_max_runner, "_AlphaMaxBoundedRawLoader", FakeLoader)
    monkeypatch.setattr(alpha_max_runner, "_AlphaMaxIndicatorDayCheckpointStore", FakeStore)
    monkeypatch.setattr(alpha_max_runner, "TimeframeAggregator", FakeAggregator)
    monkeypatch.setattr(alpha_max_runner, "ArtifactPortfolioModeStrategy", FakeStrategy)
    monkeypatch.setattr(alpha_max_runner, "_validate_preflight", lambda _value: None)
    monkeypatch.setattr(
        alpha_max_runner,
        "_validate_admitted_symbols",
        lambda _preflight, symbols: symbols,
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_expected_root_sequence",
        lambda _phase_id: ("warmup",),
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "seal_alpha_max_manifest_activation",
        lambda *_args, **_kwargs: seal,
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "build_alpha_max_backtest_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(alpha_max_runner, "_assert_definition_matches", lambda *_args: None)
    monkeypatch.setattr(alpha_max_runner, "_assert_child_identities", lambda *_args: None)
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_indicator_day_checkpoint_descriptor",
        lambda *_args, **_kwargs: descriptor,
    )
    monkeypatch.setattr(
        alpha_max_runner,
        "_alpha_max_validate_final_native_working_bars",
        lambda *_args, **_kwargs: None,
    )

    def finalize(*_args: object, **_kwargs: object) -> SimpleNamespace:
        finalizations.append(1)
        coverage = _native_finalization_coverage_stub([("BTCUSDT", "2024-01-02")])
        coverage["finalization_completed_native_keys"] = [["BTCUSDT", "2024-01-02"]]
        coverage["finalization_barrier_keys"] = []
        return SimpleNamespace(
            discarded_signal_count=0,
            finalized_children={"component": 1},
            native_coverage_by_child={"component": coverage},
            sha256="b" * 64,
        )

    monkeypatch.setattr(alpha_max_runner, "_finalize_alpha_max_native_boundary", finalize)

    def run(store: FakeStore) -> object:
        return alpha_max_runner._build_alpha_max_indicator_capsule_exact_native(
            preflight,
            output_root=tmp_path,
            phase="validation_train_fit",
            manifest_path=tmp_path / "manifest.json",
            admitted_symbols=admitted,
            phase_id="warmup",
            raw_root=FakeLoader.seal.path,
            ordered_lookup=FakeLookup(),
            watermark=config.END_DATE,
            prior_indicator_capsule=None,
            bounded_raw_loader=FakeLoader(),
            checkpoint_store=store,
            checkpoint_candidate_identity=candidate_identity,
        )

    uninterrupted = run(FakeStore({}, crash_after_seal=False))
    journal: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="crash-after-day-rename"):
        run(FakeStore(journal, crash_after_seal=True))
    resumed = run(FakeStore(journal, crash_after_seal=False))

    assert uninterrupted == resumed
    assert uninterrupted.windows_processed == 172_800
    restored = alpha_max_runner._validate_alpha_max_indicator_capsule(
        uninterrupted,
        seal=seal,
        expected_phase_id="warmup",
    )
    assert restored == uninterrupted.capsule
    assert uninterrupted.finalized_children["component"]["finalization_completed_native_keys"] == (
        ("BTCUSDT", "2024-01-02"),
    )
    assert uninterrupted.capsule["state"] == {
        "handoffs": 1,
        "release_groups": 11,
    }
    assert len(finalizations) == 2


def test_indicator_checkpoint_accepts_drained_real_child_queue() -> None:
    from lumina_quant.strategies.artifact_portfolio_mode import _SignalCaptureQueue

    child_queue = _SignalCaptureQueue()
    child_queue.put(object())
    assert child_queue.drain()
    assert child_queue.empty()

    strategy = SimpleNamespace(_children=[(object(), object(), child_queue)])
    alpha_max_runner._alpha_max_assert_indicator_checkpoint_queues_empty(
        strategy,
        alpha_max_runner.FastQueue(),
    )

    child_queue.put(object())
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="checkpoint_queue_not_empty",
    ):
        alpha_max_runner._alpha_max_assert_indicator_checkpoint_queues_empty(
            strategy,
            alpha_max_runner.FastQueue(),
        )


def test_indicator_checkpoint_omits_reconstructible_one_second_history() -> None:
    aggregator = alpha_max_runner.TimeframeAggregator(timeframes=["4h", "1d"])
    one_second = (1_700_000_000_000, 10.0, 11.0, 9.0, 10.5, 1.0)
    four_hour = (
        datetime(2023, 11, 14, tzinfo=UTC).replace(tzinfo=None),
        10.0,
        12.0,
        9.0,
        11.0,
        2.0,
    )
    aggregator._ensure_history("ADAUSDT", "1s").append(one_second)
    aggregator._ensure_history("ADAUSDT", "4h").append(four_hour)

    state = alpha_max_runner._alpha_max_indicator_checkpoint_aggregator_state(aggregator)

    assert "1s" not in state["history"]["ADAUSDT"]
    assert state["history"]["ADAUSDT"]["4h"] == [four_hour]
    assert aggregator.get_state()["history"]["ADAUSDT"]["1s"] == [one_second]
    assert (
        alpha_max_runner._parse_alpha_max_indicator_checkpoint_bytes(
            alpha_max_runner._alpha_max_indicator_checkpoint_bytes(state)
        )
        == state
    )


def test_indicator_capsule_validation_rejects_economic_or_finalization_tamper() -> None:
    state = {"child_states": {}, "sha256": ""}
    state["sha256"] = hashlib.sha256(
        alpha_max_runner._canonical_bytes({"child_states": {}})
    ).hexdigest()
    seal = SimpleNamespace(
        manifest_receipt=SimpleNamespace(sha256="a" * 64),
        expected_definition=SimpleNamespace(
            portfolio_mode="mode",
            admitted_symbols=("ADAUSDT",),
            components=(SimpleNamespace(component_id="component"),),
        ),
    )
    valid_fields = {
        "portfolio_mode": "mode",
        "phase_id": "warmup",
        "manifest_sha256": "a" * 64,
        "capsule_sha256": state["sha256"],
        "capsule": MappingProxyType(state),
        "finalized_children": MappingProxyType({}),
        "native_finalization_sha256": "b" * 64,
        "windows_processed": 1,
        "discarded_signal_count": 0,
    }
    with pytest.raises(AlphaMaxRuntimeContractError, match="finalization_invalid"):
        alpha_max_runner._validate_alpha_max_indicator_capsule(
            alpha_max_runner.AlphaMaxIndicatorCapsule(**valid_fields),
            seal=seal,
            expected_phase_id="warmup",
        )
    with pytest.raises(AlphaMaxRuntimeContractError, match="capsule_invalid"):
        alpha_max_runner._validate_alpha_max_indicator_capsule(
            alpha_max_runner.AlphaMaxIndicatorCapsule(
                **valid_fields,
                market_event_count=1,
            ),
            seal=seal,
            expected_phase_id="warmup",
        )


def test_root_validation_parallelizes_independent_roots_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendezvous = threading.Barrier(2)

    def seal(
        root_id: str,
        root_kind: str,
        _root_path: str,
        **_kwargs: object,
    ) -> object:
        rendezvous.wait(timeout=2)
        if root_id == "train":
            raise ValueError("poisoned")
        return (root_id, root_kind)

    monkeypatch.setattr(alpha_max_runner, "seal_alpha_max_root_tree", seal)
    seals, failures = alpha_max_runner._alpha_max_root_validation(
        (
            ("warmup", "raw", "/sealed/warmup"),
            ("train", "feature", "/sealed/train"),
        ),
        exchange="binance",
        availability_start_by_kind={"raw": {}, "feature": {}},
        availability_end_by_kind={"raw": {}, "feature": {}},
    )

    assert seals == {("warmup", "raw"): ("warmup", "raw")}
    assert failures == ("train_feature_root:poisoned",)


def test_root_validation_caps_parallelism_and_matches_serial_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_worker_counts: list[int] = []
    real_executor = alpha_max_runner.ThreadPoolExecutor

    def recording_executor(*, max_workers: int, thread_name_prefix: str):
        observed_worker_counts.append(max_workers)
        return real_executor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

    def seal(
        root_id: str,
        root_kind: str,
        root_path: str,
        **_kwargs: object,
    ) -> tuple[str, str, str]:
        return root_id, root_kind, root_path

    monkeypatch.setattr(alpha_max_runner, "ThreadPoolExecutor", recording_executor)
    monkeypatch.setattr(alpha_max_runner, "seal_alpha_max_root_tree", seal)
    roots = tuple(
        (f"root-{index}", "raw" if index % 2 == 0 else "feature", f"/sealed/{index}")
        for index in range(6)
    )
    availability = {"raw": {}, "feature": {}}

    serial = alpha_max_runner._alpha_max_root_validation(
        roots,
        exchange="binance",
        availability_start_by_kind=availability,
        availability_end_by_kind=availability,
        max_workers=1,
    )
    parallel = alpha_max_runner._alpha_max_root_validation(
        roots,
        exchange="binance",
        availability_start_by_kind=availability,
        availability_end_by_kind=availability,
    )

    def result_bytes(result: object) -> bytes:
        seals, failures = result
        return alpha_max_runner._canonical_bytes(
            {
                "failures": list(failures),
                "seals": [[list(key), list(value)] for key, value in sorted(seals.items())],
            }
        )

    assert observed_worker_counts == [4]
    assert result_bytes(serial) == result_bytes(parallel)
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_root_worker_count_invalid",
    ):
        alpha_max_runner._alpha_max_root_validation(
            roots,
            exchange="binance",
            availability_start_by_kind=availability,
            availability_end_by_kind=availability,
            max_workers=5,
        )


def test_exact_tick_reducer_matches_one_second_engine_state(
    tmp_path: Path,
) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    purge_feature_root = (tmp_path / "features-purge").resolve()
    validation_feature_root = (tmp_path / "features-validation").resolve()
    raw_root = (tmp_path / "raw-validation").resolve()
    purge_feature_root.mkdir()
    validation_feature_root.mkdir()
    raw_root.mkdir()
    purge_window = preflight.phase_windows["purge"]
    validation_window = preflight.phase_windows["validation"]
    lookup = AlphaMaxOrderedFundingLookup(
        (
            FeatureRootSpec(
                "purge",
                str(purge_feature_root),
                "binance",
                purge_window.start_utc,
                purge_window.end_utc,
                "a" * 64,
                "b" * 64,
            ),
            FeatureRootSpec(
                "validation",
                str(validation_feature_root),
                "binance",
                validation_window.start_utc,
                validation_window.end_utc,
                "c" * 64,
                "d" * 64,
            ),
        )
    )
    start = datetime(2025, 6, 8, tzinfo=UTC)
    timestamps = pl.datetime_range(
        start,
        start + timedelta(hours=1) - timedelta(seconds=1),
        interval="1s",
        eager=True,
    )
    data = {
        symbol: pl.DataFrame(
            {
                "datetime": timestamps,
                "open": [100.0 + index] * len(timestamps),
                "high": [101.0 + index] * len(timestamps),
                "low": [99.0 + index] * len(timestamps),
                "close": [100.0 + index] * len(timestamps),
                "volume": [10.0] * len(timestamps),
            }
        )
        for index, symbol in enumerate(admitted)
    }

    def activation(tracker: AlphaMaxStreamingEquityTracker):
        result = construct_alpha_max_engine(
            preflight,
            output_root=str(root),
            phase="validation_train_fit",
            manifest_path=materialized.path,
            admitted_symbols=admitted,
            phase_id="validation",
            nominal_cost_bps=30,
            raw_root=str(raw_root),
            ordered_lookup=lookup,
            funding_resolver=AlphaMaxFundingBoundaryResolver(lookup, admitted),
            data_dict=data,
            full_event_equity_tracker=tracker,
            _chunk_start_utc=start,
            _chunk_end_utc=start + timedelta(days=1),
        )
        return result

    reference_tracker = AlphaMaxStreamingEquityTracker()
    reference = activation(reference_tracker)
    validate_alpha_max_engine_activation(reference)
    reference.backtest._run_backtest()

    batch_tracker = AlphaMaxStreamingEquityTracker()
    batched = activation(batch_tracker)
    validate_alpha_max_engine_activation(batched)
    alpha_max_runner._run_alpha_max_exact_tick_reducer(batched)

    assert batch_tracker.finalize().to_payload() == reference_tracker.finalize().to_payload()
    assert batched.backtest.market_events == reference.backtest.market_events
    assert batched.backtest.signals == reference.backtest.signals
    assert batched.backtest.orders == reference.backtest.orders
    assert batched.backtest.fills == reference.backtest.fills
    assert batched.backtest.strategy.get_state() == reference.backtest.strategy.get_state()
    assert batched.backtest.portfolio.get_state() == reference.backtest.portfolio.get_state()
    assert (
        batched.backtest.execution_handler.get_state()
        == reference.backtest.execution_handler.get_state()
    )
    assert (
        batched.backtest._event_sequencer.get_state()
        == reference.backtest._event_sequencer.get_state()
    )
    assert {
        symbol: tuple(rows) for symbol, rows in batched.backtest.data_handler._window_rows.items()
    } == {
        symbol: tuple(rows) for symbol, rows in reference.backtest.data_handler._window_rows.items()
    }


def test_actual_engine_intrabar_liquidation_wipeout_is_terminal_before_open_order_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = preflight_alpha_max_runtime_contract(str(CONFIG_PATH))
    admitted = preflight.candidate_symbols[:5]
    root = _owned_root(tmp_path)
    node = next(row for row in _nodes() if row["row_id"] == "component_trend_1x")
    materialized = materialize_alpha_max_manifest(
        node,
        {"component_trend_1x": 1.0},
        1.0,
        "validation_train_fit",
        str(CONFIG_PATH),
        str(root),
        preflight.candidate_symbols,
        admitted,
        "d" * 64,
    )
    purge_feature_root = (tmp_path / "features-purge").resolve()
    validation_feature_root = (tmp_path / "features-validation").resolve()
    purge_feature_root.mkdir()
    validation_feature_root.mkdir()
    purge_window = preflight.phase_windows["purge"]
    validation_window = preflight.phase_windows["validation"]
    lookup = AlphaMaxOrderedFundingLookup(
        (
            FeatureRootSpec(
                "purge",
                str(purge_feature_root),
                "binance",
                purge_window.start_utc,
                purge_window.end_utc,
                "a" * 64,
                "b" * 64,
            ),
            FeatureRootSpec(
                "validation",
                str(validation_feature_root),
                "binance",
                validation_window.start_utc,
                validation_window.end_utc,
                "c" * 64,
                "d" * 64,
            ),
        )
    )
    resolver = AlphaMaxFundingBoundaryResolver(lookup, admitted)
    start = datetime(2025, 6, 8, 1, 0, tzinfo=UTC)
    timestamp_ms = int(start.timestamp() * 1000)
    rows: dict[str, tuple[tuple[object, ...], ...]] = {}
    for index, symbol in enumerate(admitted):
        second_low, second_close = 99.0, 100.0
        if index == 0:
            second_low, second_close = 50.0, 100.0
        elif index == 1:
            second_low, second_close = 40.0, 50.0
        elif index == 2:
            second_low, second_close = 80.0, 100.0
        rows[symbol] = (
            (timestamp_ms, 100.0, 101.0, 99.0, 100.0, 1_000_000.0),
            (timestamp_ms + 1000, 100.0, 101.0, second_low, second_close, 1_000_000.0),
        )

    activation = construct_alpha_max_engine(
        preflight,
        output_root=str(root),
        phase="validation_train_fit",
        manifest_path=materialized.path,
        admitted_symbols=admitted,
        phase_id="validation",
        nominal_cost_bps=30,
        raw_root=str((tmp_path / "raw-validation").resolve()),
        ordered_lookup=lookup,
        funding_resolver=resolver,
        data_dict=rows,
    )
    validate_alpha_max_engine_activation(activation)
    backtest = activation.backtest
    execution = backtest.execution_handler
    portfolio = backtest.portfolio
    raw_calls: list[tuple[str, object]] = []
    raw_check = execution.check_open_orders

    def check_open_orders(event) -> None:
        raw_calls.append((event.symbol, event.time))
        raw_check(event)

    seen_fills = []
    monkeypatch.setattr(execution, "check_open_orders", check_open_orders)
    monkeypatch.setattr(backtest, "on_fill", seen_fills.append)

    backtest.process_event(OrderEvent(admitted[0], "MKT", 1.0, "BUY"))
    backtest.process_event(OrderEvent(admitted[1], "MKT", 1000.0, "BUY"))
    backtest.process_event(OrderEvent(admitted[2], "STOP", 1.0, "SELL", stop_price=90.0))
    backtest._run_backtest()

    assert raw_calls == [(symbol, timestamp_ms) for symbol in admitted]
    assert backtest.market_events == 10
    assert backtest.orders == 3
    assert backtest.fills == 4
    assert portfolio.trade_count == 4
    assert len(seen_fills) == 4
    assert [fill.exchange for fill in seen_fills[:2]] == ["BINANCE_SIM", "BINANCE_SIM"]
    assert [fill.exchange for fill in seen_fills[2:]] == [
        "SIM_LIQUIDATION",
        "SIM_LIQUIDATION",
    ]
    assert all(fill.status == "LIQUIDATED" for fill in seen_fills[2:])
    liquidation_metadata = {
        "reason",
        "entry_price",
        "liquidation_price",
        "trigger_price",
        "bar_high",
        "bar_low",
        "close_price",
        "leverage",
        "configured_margin_mode",
        "modeled_margin_mode",
    }
    assert all(liquidation_metadata <= set(fill.metadata or {}) for fill in seen_fills[2:])
    assert all(
        (fill.metadata or {})["reason"] == "maintenance_margin_breach" for fill in seen_fills[2:]
    )

    assert len(portfolio.liquidation_events) == 2
    liquidation_by_symbol = {str(event["symbol"]): event for event in portfolio.liquidation_events}
    intrabar = liquidation_by_symbol[admitted[0]]
    assert set(intrabar) == {
        "bar_high",
        "bar_low",
        "close_price",
        "commission",
        "configured_margin_mode",
        "entry_price",
        "fill_cost",
        "leverage",
        "liquidation_price",
        "modeled_margin_mode",
        "position_qty",
        "reason",
        "symbol",
        "time",
        "trigger_price",
    }
    normalized_liquidations = alpha_max_evidence._alpha_max_normalize_liquidation_events(
        tuple(portfolio.liquidation_events)
    )
    persisted_intrabar = next(
        value.to_payload() for value in normalized_liquidations if value.symbol == admitted[0]
    )
    assert persisted_intrabar["reason"] == "maintenance_margin_breach"
    assert persisted_intrabar["trigger_price"] == pytest.approx(50.0)
    assert persisted_intrabar["bar_high"] == pytest.approx(101.0)
    assert persisted_intrabar["bar_low"] == pytest.approx(50.0)
    assert persisted_intrabar["leverage"] == pytest.approx(3.0)
    assert float(intrabar["close_price"]) > float(intrabar["liquidation_price"])
    assert 50.0 < float(intrabar["liquidation_price"]) < 100.0
    assert portfolio.current_positions[admitted[0]] == pytest.approx(0.0)
    assert portfolio.current_positions[admitted[1]] == pytest.approx(0.0)
    assert admitted[0] not in portfolio._pending_liquidation
    assert admitted[1] not in portfolio._pending_liquidation

    remaining = execution.active_orders
    assert len(remaining) == 1
    assert remaining[0]["symbol"] == admitted[2]
    assert remaining[0]["type"] == "STOP"
    assert all(fill.symbol != admitted[2] for fill in seen_fills)

    traces = execution.pricing_trace_evidence
    applications = activation.attribution_collector.applications
    capacity = execution.capacity_observation_evidence
    assert len(traces) == len(applications) == len(capacity) == 2
    assert {value.equity_before for value in capacity} == {10_000.0}
    assert backtest.fills == len(traces) + len(portfolio.liquidation_events)
    assert portfolio.current_holdings["commission"] == pytest.approx(
        sum(float(fill.commission or 0.0) for fill in seen_fills)
    )

    full_event = activation.full_event_equity_tracker.finalize()
    assert full_event.event_count == 2
    assert full_event.minimum_equity <= 0.0
    assert full_event.ruin_detected is True
    assert full_event.full_event_mdd == 1.0
    assert full_event.uncapped_full_event_drawdown > 1.0


def _native_finalization_coverage_stub(
    completed_keys: list[tuple[str, str]],
    *,
    strategy_class: str = "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
) -> dict[str, object]:
    counts: dict[str, int] = {}
    last: dict[str, str] = {}
    for symbol, key in sorted(completed_keys):
        counts[symbol] = counts.get(symbol, 0) + 1
        last[symbol] = max(last.get(symbol, key), key)
    atomic = strategy_class == "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy"
    barrier_keys = sorted({key for _symbol, key in completed_keys}) if atomic else []
    return {
        "adapter_class": strategy_class,
        "native_timeframe": "1d",
        "barrier_mode": "atomic_cross_section" if atomic else "none",
        "completed_native_keys": sorted(completed_keys),
        "completed_native_count_by_symbol": dict(sorted(counts.items())),
        "last_completed_native_key_by_symbol": dict(sorted(last.items())),
        "barrier_pending_keys": barrier_keys,
        "barrier_closed_keys": barrier_keys,
        "barrier_symbol_coverage": {
            key: sorted(symbol for symbol, completed_key in completed_keys if completed_key == key)
            for key in barrier_keys
        },
        "failed_native_keys": {},
        "partial_bucket_error": None,
    }


def test_native_finalization_is_exact_once_and_seals_discarded_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CaptureQueue:
        def __init__(self) -> None:
            self.items: list[object] = []

        def put(self, value: object) -> None:
            self.items.append(value)

        def drain(self) -> list[object]:
            values = list(self.items)
            self.items.clear()
            return values

    component = alpha_max_runner.AlphaMaxExpectedComponent(
        component_id="component_trend_1x",
        strategy_class="ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
        symbols=("BTCUSDT", "ETHUSDT"),
        params_bytes=b"{}",
        weight=1.0,
        source_artifact_id="current_trial_registry",
    )
    expected = alpha_max_runner.AlphaMaxExpectedDefinition(
        portfolio_mode="manifest:/sealed/component_trend_1x.json",
        artifact_kind="alpha_max_portfolio_manifest.v1",
        candidate_symbols=("BTCUSDT", "ETHUSDT"),
        admitted_symbols=("BTCUSDT", "ETHUSDT"),
        admission_manifest_sha256="a" * 64,
        gross_cap=1.0,
        cash_weight=0.0,
        allocation_method="static",
        source_path="/sealed/component_trend_1x.json",
        source_sha256="b" * 64,
        components=(component,),
        native_timeframes=("1d",),
    )
    strategy = object.__new__(ArtifactPortfolioModeStrategy)
    strategy.events = alpha_max_runner.FastQueue()
    child_queue = CaptureQueue()
    completed_keys: list[tuple[str, str]] = [
        ("BTCUSDT", "2025-06-07"),
        ("ETHUSDT", "2025-06-07"),
    ]
    child = SimpleNamespace(
        get_native_finalization_evidence=lambda: _native_finalization_coverage_stub(completed_keys)
    )
    strategy._children = [(component, child, child_queue)]
    boundary = datetime(2025, 6, 9, tzinfo=UTC)
    signal = SignalEvent(
        strategy_id="trend",
        symbol="BTCUSDT",
        datetime=datetime(2025, 6, 8, 23, 59, 59),
        signal_type="LONG",
        strength=0.25,
        metadata={"boundary": "final"},
    )
    calls: list[datetime] = []

    def finalize(_strategy: ArtifactPortfolioModeStrategy, watermark: datetime):
        calls.append(watermark)
        completed_keys.extend((("BTCUSDT", "2025-06-08"), ("ETHUSDT", "2025-06-08")))
        child_queue.put(signal)
        return {component.component_id: 2}

    monkeypatch.setattr(
        ArtifactPortfolioModeStrategy,
        "finalize_completed_native_buckets",
        finalize,
    )
    receipt = alpha_max_runner._finalize_alpha_max_native_boundary(
        strategy,
        expected,
        boundary,
        admitted_symbol_count=2,
        require_exact_counts=True,
    )

    expected_payload = alpha_max_runner._alpha_max_boundary_signal_payload(
        component.component_id,
        signal,
    )
    expected_signal_bytes = alpha_max_runner._canonical_bytes(expected_payload) + b"\n"
    assert calls == [boundary]
    assert dict(receipt.finalized_children) == {component.component_id: 2}
    assert receipt.discarded_signal_count == 1
    assert receipt.discarded_signal_sha256 == alpha_max_runner._sha256(expected_signal_bytes)
    assert expected_payload["datetime"] == "2025-06-08T23:59:59Z"
    assert child_queue.items == []
    assert strategy.events.empty()
    coverage = receipt.to_payload()["native_coverage_by_child"][component.component_id]
    assert coverage["finalization_completed_native_keys"] == [
        ["BTCUSDT", "2025-06-08"],
        ["ETHUSDT", "2025-06-08"],
    ]
    assert coverage["finalization_barrier_keys"] == []


def test_native_finalization_receipt_seals_atomic_barrier_coverage_fail_closed() -> None:
    child_id = "component_near_high_1x"
    key = "2025-06-08"
    coverage = {
        "adapter_class": "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy",
        "native_timeframe": "1d",
        "barrier_mode": "atomic_cross_section",
        "completed_native_keys": [["BTCUSDT", key], ["ETHUSDT", key]],
        "completed_native_count_by_symbol": {"BTCUSDT": 1, "ETHUSDT": 1},
        "last_completed_native_key_by_symbol": {"BTCUSDT": key, "ETHUSDT": key},
        "barrier_pending_keys": [key],
        "barrier_closed_keys": [key],
        "barrier_symbol_coverage": {key: ["BTCUSDT", "ETHUSDT"]},
        "failed_native_keys": {},
        "partial_bucket_error": None,
        "finalization_completed_native_keys": [["BTCUSDT", key], ["ETHUSDT", key]],
        "finalization_barrier_keys": [key],
    }
    receipt = alpha_max_evidence.build_alpha_max_native_finalization_receipt(
        boundary_utc=datetime(2025, 6, 9, tzinfo=UTC),
        finalized_children={child_id: 1},
        native_coverage_by_child={child_id: coverage},
        discarded_signal_count=0,
        discarded_signal_sha256=hashlib.sha256(b"").hexdigest(),
    )
    assert receipt.to_payload()["native_coverage_by_child"][child_id] == coverage
    historical_key = "2025-06-07"
    historical_coverage = {
        **coverage,
        "completed_native_keys": [
            ["BTCUSDT", historical_key],
            ["BTCUSDT", key],
            ["ETHUSDT", historical_key],
            ["ETHUSDT", key],
        ],
        "completed_native_count_by_symbol": {"BTCUSDT": 2, "ETHUSDT": 2},
    }
    historical_receipt = alpha_max_evidence.build_alpha_max_native_finalization_receipt(
        boundary_utc=datetime(2025, 6, 9, tzinfo=UTC),
        finalized_children={child_id: 1},
        native_coverage_by_child={child_id: historical_coverage},
        discarded_signal_count=0,
        discarded_signal_sha256=hashlib.sha256(b"").hexdigest(),
    )
    assert (
        historical_receipt.to_payload()["native_coverage_by_child"][child_id] == historical_coverage
    )

    for poison in (
        {"barrier_closed_keys": []},
        {"failed_native_keys": {key: "poisoned_barrier"}},
        {"finalization_barrier_keys": []},
    ):
        with pytest.raises(ValueError, match="native_finalization_coverage_invalid"):
            alpha_max_evidence.build_alpha_max_native_finalization_receipt(
                boundary_utc=datetime(2025, 6, 9, tzinfo=UTC),
                finalized_children={child_id: 1},
                native_coverage_by_child={child_id: {**coverage, **poison}},
                discarded_signal_count=0,
                discarded_signal_sha256=hashlib.sha256(b"").hexdigest(),
            )


@pytest.mark.parametrize("component_id", ("bad\ncomponent", "x" * 1024))
def test_training_worker_failure_diagnostic_is_bounded(
    component_id: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = alpha_max_runner._alpha_max_replay_training_component_worker((component_id, b"", ""))
    assert result == (
        component_id,
        "semantic_failure",
        b"",
        "alpha_max_training_worker_start_method_invalid",
    )
    assert capsys.readouterr().err == (
        "alpha_max_training_worker_failure:unknown:alpha_max_training_worker_start_method_invalid\n"
    )


@pytest.mark.parametrize(
    "poison",
    (
        {"adapter_class": "StubNativeAdapter"},
        {"native_timeframe": "4h"},
        {"barrier_mode": "atomic_cross_section"},
        {
            "completed_native_keys": [("ETHUSDT", "2025-06-07")],
            "completed_native_count_by_symbol": {"ETHUSDT": 1},
            "last_completed_native_key_by_symbol": {"ETHUSDT": "2025-06-07"},
        },
    ),
)
def test_native_finalization_rejects_unbound_adapter_timeframe_mode_and_symbols(
    monkeypatch: pytest.MonkeyPatch,
    poison: dict[str, object],
) -> None:
    component = alpha_max_runner.AlphaMaxExpectedComponent(
        component_id="component_trend_1x",
        strategy_class="ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
        symbols=("BTCUSDT",),
        params_bytes=b"{}",
        weight=1.0,
        source_artifact_id="current_trial_registry",
    )
    expected = alpha_max_runner.AlphaMaxExpectedDefinition(
        portfolio_mode="manifest:/sealed/component.json",
        artifact_kind="alpha_max_portfolio_manifest.v1",
        candidate_symbols=("BTCUSDT",),
        admitted_symbols=("BTCUSDT",),
        admission_manifest_sha256="a" * 64,
        gross_cap=1.0,
        cash_weight=0.0,
        allocation_method="static",
        source_path="/sealed/component.json",
        source_sha256="b" * 64,
        components=(component,),
        native_timeframes=("1d",),
    )
    snapshot = {
        **_native_finalization_coverage_stub([("BTCUSDT", "2025-06-07")]),
        **poison,
    }
    strategy = object.__new__(ArtifactPortfolioModeStrategy)
    strategy.events = alpha_max_runner.FastQueue()
    strategy._children = [
        (
            component,
            SimpleNamespace(get_native_finalization_evidence=lambda: snapshot),
            SimpleNamespace(drain=lambda: []),
        )
    ]
    monkeypatch.setattr(
        ArtifactPortfolioModeStrategy,
        "finalize_completed_native_buckets",
        lambda _strategy, _watermark: pytest.fail("unbound snapshot reached finalizer"),
    )

    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="native_finalization_coverage_binding_mismatch",
    ):
        alpha_max_runner._finalize_alpha_max_native_boundary(
            strategy,
            expected,
            datetime(2025, 6, 9, tzinfo=UTC),
            admitted_symbol_count=1,
            require_exact_counts=True,
        )


@pytest.mark.parametrize(
    ("strategy_class", "finalized_count", "expected_error"),
    (
        (
            "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
            0,
            "native_finalization_invalid",
        ),
        (
            "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
            3,
            "native_finalization_count_mismatch",
        ),
        (
            "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy",
            2,
            "native_finalization_count_mismatch",
        ),
    ),
)
def test_native_finalization_rejects_zero_duplicate_and_near_high_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    strategy_class: str,
    finalized_count: int,
    expected_error: str,
) -> None:
    component = alpha_max_runner.AlphaMaxExpectedComponent(
        component_id="component",
        strategy_class=strategy_class,
        symbols=("BTCUSDT", "ETHUSDT"),
        params_bytes=b"{}",
        weight=1.0,
        source_artifact_id="current_trial_registry",
    )
    expected = alpha_max_runner.AlphaMaxExpectedDefinition(
        portfolio_mode="manifest:/sealed/component.json",
        artifact_kind="alpha_max_portfolio_manifest.v1",
        candidate_symbols=("BTCUSDT", "ETHUSDT"),
        admitted_symbols=("BTCUSDT", "ETHUSDT"),
        admission_manifest_sha256="a" * 64,
        gross_cap=1.0,
        cash_weight=0.0,
        allocation_method="static",
        source_path="/sealed/component.json",
        source_sha256="b" * 64,
        components=(component,),
        native_timeframes=("1d",),
    )
    strategy = object.__new__(ArtifactPortfolioModeStrategy)
    strategy.events = alpha_max_runner.FastQueue()
    strategy._children = [
        (
            component,
            SimpleNamespace(
                get_native_finalization_evidence=lambda: _native_finalization_coverage_stub(
                    [("BTCUSDT", "2025-06-07"), ("ETHUSDT", "2025-06-07")],
                    strategy_class=strategy_class,
                )
            ),
            SimpleNamespace(drain=lambda: []),
        )
    ]
    monkeypatch.setattr(
        ArtifactPortfolioModeStrategy,
        "finalize_completed_native_buckets",
        lambda _strategy, _watermark: {component.component_id: finalized_count},
    )

    with pytest.raises(AlphaMaxRuntimeContractError, match=expected_error):
        alpha_max_runner._finalize_alpha_max_native_boundary(
            strategy,
            expected,
            datetime(2025, 6, 9, tzinfo=UTC),
            admitted_symbol_count=2,
            require_exact_counts=True,
        )


def test_native_finalization_rejects_non_signal_child_queue_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = alpha_max_runner.AlphaMaxExpectedComponent(
        component_id="component_trend_1x",
        strategy_class="ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
        symbols=("BTCUSDT",),
        params_bytes=b"{}",
        weight=1.0,
        source_artifact_id="current_trial_registry",
    )
    expected = alpha_max_runner.AlphaMaxExpectedDefinition(
        portfolio_mode="manifest:/sealed/component.json",
        artifact_kind="alpha_max_portfolio_manifest.v1",
        candidate_symbols=("BTCUSDT",),
        admitted_symbols=("BTCUSDT",),
        admission_manifest_sha256="a" * 64,
        gross_cap=1.0,
        cash_weight=0.0,
        allocation_method="static",
        source_path="/sealed/component.json",
        source_sha256="b" * 64,
        components=(component,),
        native_timeframes=("1d",),
    )
    strategy = object.__new__(ArtifactPortfolioModeStrategy)
    strategy.events = alpha_max_runner.FastQueue()
    strategy._children = [
        (
            component,
            SimpleNamespace(
                get_native_finalization_evidence=lambda: _native_finalization_coverage_stub(
                    [("BTCUSDT", "2025-06-07")]
                )
            ),
            SimpleNamespace(drain=lambda: [SimpleNamespace(type="ORDER")]),
        )
    ]
    monkeypatch.setattr(
        ArtifactPortfolioModeStrategy,
        "finalize_completed_native_buckets",
        lambda _strategy, _watermark: {component.component_id: 1},
    )

    with pytest.raises(AlphaMaxRuntimeContractError, match="boundary_event_invalid"):
        alpha_max_runner._finalize_alpha_max_native_boundary(
            strategy,
            expected,
            datetime(2025, 6, 9, tzinfo=UTC),
            admitted_symbol_count=1,
            require_exact_counts=True,
        )


def test_day_boundary_settlement_finalizes_only_the_scoring_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = object.__new__(alpha_max_runner.AlphaMaxEngineActivation)
    object.__setattr__(activation, "backtest", SimpleNamespace(strategy="strategy"))
    object.__setattr__(
        activation,
        "artifact_seal",
        SimpleNamespace(expected_definition="definition"),
    )
    object.__setattr__(activation, "admitted_symbols", ("BTCUSDT",))
    tracker = object.__new__(alpha_max_runner._AlphaMaxFoldEquityFanout)
    settled: list[tuple[datetime, bool]] = []
    finalized: list[tuple[object, ...]] = []
    sentinel = object()

    monkeypatch.setattr(
        alpha_max_runner._AlphaMaxFoldEquityFanout,
        "settle_day_end",
        lambda _tracker, boundary, *, settle_funding: settled.append((boundary, settle_funding)),
    )

    def finalize(*args, **kwargs):
        finalized.append((*args, kwargs))
        return sentinel

    monkeypatch.setattr(alpha_max_runner, "_finalize_alpha_max_native_boundary", finalize)
    internal = datetime(2025, 6, 9, tzinfo=UTC)
    final = internal + timedelta(days=1)

    assert (
        alpha_max_runner._settle_alpha_max_day_boundary(
            activation,
            tracker,
            internal,
            scoring_boundary=False,
        )
        is None
    )
    assert finalized == []
    assert (
        alpha_max_runner._settle_alpha_max_day_boundary(
            activation,
            tracker,
            final,
            scoring_boundary=True,
        )
        is sentinel
    )
    assert settled == [(internal, True), (final, True)]
    assert len(finalized) == 1
    assert finalized[0][0:3] == ("strategy", "definition", final)
    assert finalized[0][3] == {
        "admitted_symbol_count": 1,
        "require_exact_counts": True,
    }


def test_prelock_and_historical_bundles_are_exclusive_immutable_and_duplicate_safe(
    tmp_path: Path,
) -> None:
    prelock_root = (tmp_path / "prelock").resolve()
    stable = {"status/matrix.json": b'{"status":"complete"}\n'}
    prelock = create_alpha_max_prelock_bundle(
        str(prelock_root),
        stable,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    before = {
        path.relative_to(prelock_root): path.read_bytes()
        for path in prelock_root.rglob("*")
        if path.is_file()
    }

    assert Path(prelock.seal_path).is_file()
    assert prelock_root.stat().st_mode & 0o222 == 0
    with pytest.raises(AlphaMaxRuntimeContractError, match="output_root_exists"):
        create_alpha_max_prelock_bundle(
            str(prelock_root),
            stable,
            prelock_champion=None,
            selected_candidate_id=None,
        )

    historical = create_alpha_max_historical_package(
        str(prelock_root),
        str((tmp_path / "historical-1").resolve()),
        {"report.json": b'{"report_only":true}\n'},
        completion_id="evaluation-001",
    )
    after = {
        path.relative_to(prelock_root): path.read_bytes()
        for path in prelock_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert Path(historical.seal_path).is_file()
    with pytest.raises(AlphaMaxRuntimeContractError, match="completion_duplicate"):
        create_alpha_max_historical_package(
            str(prelock_root),
            str((tmp_path / "historical-2").resolve()),
            {"report.json": b'{"report_only":true}\n'},
            completion_id="evaluation-001",
        )


def test_historical_gate_requires_exact_complete_prelock_matrix_schema(
    tmp_path: Path,
) -> None:
    statuses: list[dict[str, object]] = []
    for row_id in alpha_max_runner._ALPHA_MAX_CURRENT_ROW_IDS:
        for nominal in ALPHA_MAX_COST_CELL_BPS:
            if row_id in alpha_max_runner._ALPHA_MAX_UNAVAILABLE_ROWS:
                statuses.append(
                    {
                        "capsule_sha256": None,
                        "engine_constructed": False,
                        "manifest_sha256": None,
                        "nominal_cost_bps": nominal,
                        "row_id": row_id,
                        "row_role": "incumbent_unavailable",
                        "selection_eligible": False,
                        "status": "incumbent_replay_unavailable",
                    }
                )
            elif row_id in alpha_max_runner._ALPHA_MAX_DIAGNOSTIC_ROWS:
                statuses.append(
                    {
                        "capsule_sha256": None,
                        "engine_constructed": False,
                        "manifest_sha256": None,
                        "nominal_cost_bps": nominal,
                        "row_id": row_id,
                        "row_role": "track_b_diagnostic",
                        "selection_eligible": False,
                        "status": "diagnostic_report_only",
                    }
                )
            else:
                statuses.append(
                    {
                        "capsule_sha256": "a" * 64,
                        "cell_sha256": "b" * 64,
                        "engine_constructed": True,
                        "manifest_sha256": "c" * 64,
                        "nominal_cost_bps": nominal,
                        "row_id": row_id,
                        "row_role": "resolvable_candidate",
                        "selection_eligible": True,
                        "status": "resolved_engine_cell_complete",
                    }
                )
    valid = {
        "artifact_kind": "alpha_max_matrix_statuses.v1",
        "domain": "validation",
        "engine_cell_count": 68,
        "physical_fold_run_count": 816,
        "status_count": 84,
        "statuses": statuses,
    }
    prelock_payload = {
        "engine_cell_count": 68,
        "physical_fold_run_count": 816,
    }

    def validate(payload: dict[str, object], name: str) -> None:
        matrix_bytes = alpha_max_runner._canonical_bytes(payload) + b"\n"
        root = (tmp_path / name).resolve()
        create_alpha_max_prelock_bundle(
            root,
            {"status/matrix.json": matrix_bytes},
            prelock_champion=None,
            selected_candidate_id=None,
        )
        snapshot = alpha_max_runner._snapshot_bundle_tree(root)
        alpha_max_runner._validate_complete_alpha_max_prelock_matrix(
            snapshot,
            prelock_payload,
        )

    validate(valid, "valid")
    mutations: list[dict[str, object]] = []

    def mutated() -> dict[str, object]:
        return json.loads(json.dumps(valid))

    resolved_index = next(
        index
        for index, value in enumerate(statuses)
        if value["row_id"] in alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS
    )
    unavailable_index = next(
        index
        for index, value in enumerate(statuses)
        if value["row_id"] in alpha_max_runner._ALPHA_MAX_UNAVAILABLE_ROWS
    )
    diagnostic_index = next(
        index
        for index, value in enumerate(statuses)
        if value["row_id"] in alpha_max_runner._ALPHA_MAX_DIAGNOSTIC_ROWS
    )
    extra_key = mutated()
    extra_key["statuses"][resolved_index]["attacker"] = True  # type: ignore[index]
    mutations.append(extra_key)
    invalid_hash = mutated()
    invalid_hash["statuses"][resolved_index]["capsule_sha256"] = "not-a-sha"  # type: ignore[index]
    mutations.append(invalid_hash)
    invalid_bool = mutated()
    invalid_bool["statuses"][resolved_index]["selection_eligible"] = "true"  # type: ignore[index]
    mutations.append(invalid_bool)
    invalid_unavailable = mutated()
    invalid_unavailable["statuses"][unavailable_index]["status"] = "attacker_status"  # type: ignore[index]
    mutations.append(invalid_unavailable)
    invalid_diagnostic = mutated()
    invalid_diagnostic["statuses"][diagnostic_index]["row_role"] = "attacker_role"  # type: ignore[index]
    mutations.append(invalid_diagnostic)
    reordered = mutated()
    reordered["statuses"][0], reordered["statuses"][1] = (  # type: ignore[index]
        reordered["statuses"][1],  # type: ignore[index]
        reordered["statuses"][0],  # type: ignore[index]
    )
    mutations.append(reordered)
    top_level_extra = mutated()
    top_level_extra["attacker"] = True
    mutations.append(top_level_extra)

    for index, payload in enumerate(mutations):
        with pytest.raises(
            AlphaMaxRuntimeContractError,
            match="prelock_matrix_incomplete",
        ):
            validate(payload, f"invalid-{index}")


def test_sealed_selection_parser_rejects_coercion_missing_attribution_and_relabeling() -> None:
    gate_inputs = tuple(
        AlphaMaxGateInput(
            row_id=row_id,
            comparison_role="prelock_selection",
            evidence_tier="actual_engine",
            comparison_valid=True,
            nominal_cost_bps=30,
            cumulative_return=0.20 + index / 100.0,
            cagr=0.10 + index / 1000.0,
            calmar=0.50 + index / 1000.0,
            net_sharpe=1.0 + index / 1000.0,
            full_event_mdd=0.20,
            reporting_4h_mdd=0.20,
            dsr=0.95,
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
            seed_schedule_sha256="e" * 64,
        )
        for index, row_id in enumerate(alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS)
    )
    selection = select_alpha_max_prelock_champion(gate_inputs)
    parsed = alpha_max_runner._alpha_max_selection_from_bytes(
        selection.canonical_bytes,
        role="prelock_selection",
    )
    assert parsed.prelock_champion == selection.prelock_champion
    assert len(parsed.decisions) == len(alpha_max_runner._ALPHA_MAX_RESOLVABLE_ROWS)
    assert len(parsed.scaling_attributions) == 2

    def mutated() -> dict[str, object]:
        return json.loads(selection.canonical_bytes)

    invalid_payloads: list[dict[str, object]] = []
    extra = mutated()
    extra["attacker"] = True
    invalid_payloads.append(extra)
    string_bool = mutated()
    string_bool["decisions"][0]["eligible"] = "false"  # type: ignore[index]
    invalid_payloads.append(string_bool)
    empty_decisions = mutated()
    empty_decisions["decisions"] = []
    empty_decisions["ranked_candidate_ids"] = []
    empty_decisions["prelock_champion"] = None
    empty_decisions["selected_candidate_id"] = None
    invalid_payloads.append(empty_decisions)
    no_scaling = mutated()
    no_scaling["scaling_attributions"] = []
    invalid_payloads.append(no_scaling)
    wrong_champion = mutated()
    wrong_champion["prelock_champion"] = None
    wrong_champion["selected_candidate_id"] = None
    wrong_champion["historical_evaluation_leader"] = selection.prelock_champion
    invalid_payloads.append(wrong_champion)
    coerced_attribution = mutated()
    coerced_attribution["scaling_attributions"][0]["sibling_gate_eligible"] = "true"  # type: ignore[index]
    invalid_payloads.append(coerced_attribution)

    for payload in invalid_payloads:
        raw = alpha_max_runner._canonical_bytes(payload) + b"\n"
        with pytest.raises(
            AlphaMaxRuntimeContractError,
            match="selection_artifact_invalid",
        ):
            alpha_max_runner._alpha_max_selection_from_bytes(
                raw,
                role="prelock_selection",
            )


def _trend_liquidity_buckets_fixture():
    admitted = ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    medians = {symbol: float(index + 1) for index, symbol in enumerate(admitted)}
    bucket_by_symbol = {
        symbol: ("weakest" if index < 2 else "middle" if index < 4 else "liquid")
        for index, symbol in enumerate(admitted)
    }
    payload = {
        "admission_computation_sha256": "a" * 64,
        "admitted_symbols": list(admitted),
        "artifact_kind": "alpha_max_train_liquidity_buckets.v1",
        "bucket_by_symbol": bucket_by_symbol,
        "bucket_order": ["weakest", "middle", "liquid"],
        "bucket_rule": "floor(3*ascending_rank_index/admitted_symbol_count)",
        "median_quote_notional_usdt": medians,
        "phase": "train_frozen_report_only",
        "report_only": True,
        "selection_influence": False,
        "symbols_by_bucket": {
            bucket: [symbol for symbol in admitted if bucket_by_symbol[symbol] == bucket]
            for bucket in ("weakest", "middle", "liquid")
        },
        "tie_break": "median_quote_notional_usdt_ascending_then_symbol_ascending",
    }
    return validate_alpha_max_train_liquidity_buckets(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _exact_trend_liquidity_matrix_fixture(domain: str):
    buckets = _trend_liquidity_buckets_fixture()
    fold_runs = []
    fold_hashes = []
    contributions_by_fold = []
    for fold_index, fold_id in enumerate(alpha_max_runner._alpha_max_fold_ids(domain)):
        fold_hash = hashlib.sha256(f"{domain}:{fold_id}".encode()).hexdigest()
        diagnostics = object.__new__(AlphaMaxRunReportOnlyDiagnostics)
        contributions = {
            symbol: (
                float(fold_index + symbol_index + 1)
                if symbol in buckets.admitted_symbols
                else -1_000_000.0
            )
            for symbol_index, symbol in enumerate(ALPHA_MAX_CANDIDATE_SYMBOLS)
        }
        object.__setattr__(diagnostics, "symbol_contribution_usdt", contributions)
        actual_run = object.__new__(AlphaMaxActualEngineRunReceipt)
        object.__setattr__(actual_run, "report_only_diagnostics", diagnostics)
        fold = object.__new__(AlphaMaxFoldRunEvidence)
        object.__setattr__(fold, "actual_engine_run", actual_run)
        object.__setattr__(fold, "sha256", fold_hash)
        fold_runs.append(fold)
        fold_hashes.append(fold_hash)
        contributions_by_fold.append(contributions)

    pre_gate = object.__new__(AlphaMaxCostCellPreGateEvidence)
    object.__setattr__(pre_gate, "fold_runs", tuple(fold_runs))
    target_cell = object.__new__(AlphaMaxCostCellEvidence)
    object.__setattr__(target_cell, "pre_gate_evidence", pre_gate)
    decoy = SimpleNamespace(pre_gate_evidence=SimpleNamespace(fold_runs=()))
    matrix = alpha_max_runner._AlphaMaxCompletedMatrix(
        domain=domain,
        rows=(),
        cells=MappingProxyType(
            {
                ("component_trend_1x", 20): decoy,
                ("component_trend_1x", 30): target_cell,
                ("full_equal_weight_1x", 30): decoy,
            }
        ),
        status_payload=b"{}\n",
        physical_fold_run_count=len(fold_runs),
        prepared_rows=MappingProxyType({}),
        gross_by_row=MappingProxyType({}),
    )
    return matrix, buckets, tuple(fold_hashes), tuple(contributions_by_fold)


def test_trend_liquidity_falsifier_uses_nominal_30_trend_receipts_and_admitted_symbols() -> None:
    matrix, buckets, fold_hashes, contributions = _exact_trend_liquidity_matrix_fixture(
        "validation"
    )

    payload = json.loads(
        alpha_max_runner._alpha_max_trend_liquidity_falsifier_artifact(matrix, buckets)
    )

    assert payload["row_id"] == "component_trend_1x"
    assert payload["nominal_cost_bps"] == 30
    assert payload["fold_run_sha256s"] == list(fold_hashes)
    assert tuple(payload["symbol_contribution_usdt"]) == buckets.admitted_symbols
    assert payload["symbol_contribution_usdt"] == {
        symbol: sum(fold[symbol] for fold in contributions) for symbol in buckets.admitted_symbols
    }
    assert not set(ALPHA_MAX_CANDIDATE_SYMBOLS[5:]) & set(payload["symbol_contribution_usdt"])


@pytest.mark.parametrize(
    ("domain", "fold_count"),
    (("validation", 12), ("historical_exposed_evaluation", 10)),
)
def test_trend_liquidity_falsifier_preserves_domain_fold_order_and_hashes(
    domain: str,
    fold_count: int,
) -> None:
    matrix, buckets, fold_hashes, _contributions = _exact_trend_liquidity_matrix_fixture(domain)

    payload = json.loads(
        alpha_max_runner._alpha_max_trend_liquidity_falsifier_artifact(matrix, buckets)
    )

    assert payload["domain"] == domain
    assert payload["fold_run_sha256s"] == list(fold_hashes)
    assert len(payload["fold_run_sha256s"]) == fold_count


def test_positive_trend_liquidity_falsifier_is_report_only_and_noncausal() -> None:
    matrix, buckets, _fold_hashes, _contributions = _exact_trend_liquidity_matrix_fixture(
        "validation"
    )

    payload = json.loads(
        alpha_max_runner._alpha_max_trend_liquidity_falsifier_artifact(matrix, buckets)
    )

    assert payload["status"] == "liquidity_falsifier_not_triggered"
    assert payload["rejection_reasons"] == []
    assert payload["report_only"] is True
    assert payload["selection_influence"] is False
    assert "causal" not in payload["status"]


def test_trend_liquidity_falsifier_is_wired_to_prelock_and_historical_artifact_paths() -> None:
    prelock_source = inspect.getsource(alpha_max_runner.run_alpha_max_prelock_process)
    historical_source = inspect.getsource(alpha_max_runner.run_alpha_max_historical_process)

    assert "diagnostics/validation/trend_liquidity_falsifier.json" in prelock_source
    assert "validation_trend_liquidity_falsifier" in prelock_source
    assert (
        "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json"
        in historical_source
    )
    assert "historical_trend_liquidity_falsifier" in historical_source


def test_indicator_capsule_requires_exact_bounded_loader_and_keeps_checkpoint_external() -> None:
    """Indicator replay has one exact-native path; the journal stays out of capsules."""
    parameters = inspect.signature(alpha_max_runner.build_alpha_max_indicator_capsule).parameters
    assert parameters["bounded_raw_loader"].default is inspect.Parameter.empty
    assert "data_dict" not in parameters
    assert parameters["checkpoint_store"].default is None
    assert "checkpoint" not in alpha_max_runner.AlphaMaxIndicatorCapsule.__dataclass_fields__


@pytest.mark.parametrize(
    ("mask", "offset", "expected"),
    (
        (np.array([], dtype=bool), 0, None),
        (np.array([False, False, False]), 17, None),
        (np.array([True, False, True]), 3, 3),
        (np.array([False, False, True, True]), 9, 11),
        (np.array([[False, False], [True, False]]), 5, 7),
    ),
)
def test_first_true_index_preserves_flattened_first_match_without_match_array(
    mask: np.ndarray, offset: int, expected: int | None
) -> None:
    assert alpha_max_runner._alpha_max_first_true_index(mask, offset=offset) == expected


def test_exact_tick_reducer_fails_closed_when_columnar_contract_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidColumnarHandler:
        def alpha_max_exact_columnar_view(self) -> object:
            raise ValueError("alpha_max_columnar_rows_invalid")

    generic_replay_called = False

    def generic_replay() -> None:
        nonlocal generic_replay_called
        generic_replay_called = True

    handler = InvalidColumnarHandler()
    backtest = SimpleNamespace(
        data_handler=handler,
        record_history=False,
        track_metrics=False,
        record_trades=False,
        portfolio=SimpleNamespace(strategy_quality=SimpleNamespace(enabled=False)),
        timeframe_aggregator=None,
        strategy=SimpleNamespace(required_timeframes=()),
        _resolve_required_lookbacks=lambda: {},
        _run_backtest=generic_replay,
    )
    activation = SimpleNamespace(backtest=backtest)
    monkeypatch.setattr(
        alpha_max_runner, "HistoricParquetWindowedDataHandler", InvalidColumnarHandler
    )

    with pytest.raises(AlphaMaxRuntimeContractError, match="alpha_max_tick_columnar_view_invalid"):
        alpha_max_runner._run_alpha_max_exact_tick_reducer(activation)

    assert generic_replay_called is False


def test_checkpoint_schema_rejects_legacy_descriptor_versions() -> None:
    assert (
        "_include_v2_bindings"
        not in inspect.signature(
            alpha_max_runner._alpha_max_prelock_checkpoint_descriptor
        ).parameters
    )
    with pytest.raises(
        AlphaMaxRuntimeContractError,
        match="alpha_max_checkpoint_descriptor_version_invalid",
    ):
        alpha_max_runner._alpha_max_validate_checkpoint_descriptor(
            {"artifact_kind": "alpha_max_restartable_attempt_descriptor.v2"}
        )
