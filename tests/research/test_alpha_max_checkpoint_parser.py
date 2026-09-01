from __future__ import annotations

import copy
import gc
import hashlib
import mmap
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from lumina_quant.backtesting.execution_model import ExecutionModel, ExecutionModelConfig
from lumina_quant.backtesting.portfolio_backtest import FillApplicationAttribution
from lumina_quant.research import alpha_max_evidence as evidence
from lumina_quant.research import alpha_max_engine_runner as runner
from lumina_quant.utils.artifact_read_receipt import read_artifact_bytes

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
).resolve()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_actual_engine_config_receipt_accepts_pinned_proc_fd_path() -> None:
    descriptor = os.open(_CONFIG_PATH.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    requested_path = f"/proc/self/fd/{descriptor}/{_CONFIG_PATH.name}"
    try:
        receipt, payload = read_artifact_bytes(
            requested_path,
            artifact_id="alpha_max_config",
        )
        result = evidence._alpha_max_artifact_receipt_payload(receipt)
        fresh_receipt, fresh_payload = read_artifact_bytes(
            receipt.canonical_path,
            artifact_id="alpha_max_config",
        )
    finally:
        os.close(descriptor)

    assert result["requested_path"] == requested_path
    assert result["canonical_path"] == str(_CONFIG_PATH)
    assert result["byte_count"] == len(payload)
    assert result["sha256"] == _sha256(payload)
    assert fresh_payload == payload
    assert evidence._alpha_max_artifact_receipts_match(receipt, fresh_receipt)


def _indicator_day_descriptor(root: Path) -> dict[str, object]:
    parent = root.parent.stat()
    config_receipt = {
        "byte_count": 1,
        "path": str((root.parent / "identity.json").resolve()),
        "sha256": "a" * 64,
    }
    manifest_receipt = {
        "byte_count": 1,
        "path": str(
            (
                root.parent / "output/manifests/validation_train_fit/component_carry_1x.json"
            ).resolve()
        ),
        "sha256": "a" * 64,
    }
    root_binding = {
        "availability_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "inventory_sha256": "c" * 64,
        "path": str((root.parent / "root").resolve()),
        "root_id": "warmup",
        "root_kind": "raw",
        "seal_sha256": "d" * 64,
    }
    return {
        "artifact_kind": "alpha_max_indicator_day_checkpoint_attempt.v1",
        "phase": "validation_train_fit",
        "phase_id": "warmup",
        "checkpoint_unit": "whole_utc_day_pre_finalization",
        "start_utc": "2024-01-01T00:00:00Z",
        "end_utc": "2025-01-01T00:00:00Z",
        "watermark_utc": "2025-01-01T00:00:00Z",
        "window_seconds": 1,
        "windows_per_day": 86_400,
        "terminal_windows": 31_622_400,
        "config": config_receipt,
        "contract_manifest": {"byte_count": 1, "sha256": "a" * 64},
        "manifest": manifest_receipt,
        "admitted_symbols": list(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS),
        "raw_roots": [root_binding],
        "feature_roots": [{**root_binding, "root_kind": "feature"}],
        "implementation_identity": {
            "inventory": [
                {
                    "byte_count": 1,
                    "relative_path": "src/example.py",
                    "sha256": "a" * 64,
                }
            ]
        },
        "runtime_identity": {
            "extension_byte_count": 1,
            "extension_module": "lumina_quant._compute",
            "extension_path": str((root.parent / "_compute.so").resolve()),
            "extension_sha256": "a" * 64,
            "extension_source_hash": "a" * 16,
            "extension_version": "0.2.0",
            "runtime_contract_sha256": "a" * 64,
        },
        "python_identity": {
            "cache_tag": "cpython-test",
            "executable": str((root.parent / "python").resolve()),
            "executable_byte_count": 1,
            "executable_sha256": "a" * 64,
            "version": [3, 14, 0],
        },
        "thread_identity": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
        },
        "candidate_identity": {
            "path": str((root.parent / "candidate.json").resolve()),
            "candidate_seal_sha256": "a" * 64,
            "capsule_sha256": "b" * 64,
            "finalization_sha256": "c" * 64,
        },
        "checkpoint": {
            "root": str(root),
            "parent": str(root.parent),
            "parent_identity": [parent.st_dev, parent.st_ino],
        },
        "order_routing_enabled": False,
        "partial_output_reusable": False,
    }


def test_indicator_day_checkpoint_typed_codec_rejects_tamper_and_aliases() -> None:
    value = {
        "none": None,
        "bool": True,
        "int": 1,
        "float": -0.0,
        "day": __import__("datetime").date(2024, 1, 1),
        "when": __import__("datetime").datetime(2024, 1, 1),
        "items": [("x",), {"a", 2}, frozenset({"z"})],
    }
    payload = runner._alpha_max_indicator_checkpoint_bytes(value)
    assert runner._parse_alpha_max_indicator_checkpoint_bytes(payload) == value
    with pytest.raises(runner.AlphaMaxRuntimeContractError):
        runner._parse_alpha_max_indicator_checkpoint_bytes(payload[:-1])
    with pytest.raises(runner.AlphaMaxRuntimeContractError):
        runner._parse_alpha_max_indicator_checkpoint_bytes(
            b'{"t":"set","v":[{"t":"bool","v":true},{"t":"int","v":"1"}]}\n'
        )


def test_indicator_day_checkpoint_date_codec_is_exact_and_canonical() -> None:
    datetime_module = __import__("datetime")
    day = datetime_module.date(2024, 1, 1)
    moment = datetime_module.datetime(2024, 1, 1)
    assert runner._alpha_max_indicator_checkpoint_encode(day) == {
        "t": "date",
        "v": "2024-01-01",
    }
    assert runner._alpha_max_indicator_checkpoint_encode(moment)["t"] == "datetime"
    decoded = runner._parse_alpha_max_indicator_checkpoint_bytes(b'{"t":"date","v":"2024-01-01"}\n')
    assert type(decoded) is datetime_module.date
    assert decoded == day

    class DateSubclass(datetime_module.date):
        pass

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_indicator_checkpoint_type_invalid:datesubclass",
    ):
        runner._alpha_max_indicator_checkpoint_encode(DateSubclass(2024, 1, 1))
    for payload in (
        b'{"t":"date","v":"20240101"}\n',
        b'{"t":"date","v":"2024-W01-1"}\n',
        b'{"extra":0,"t":"date","v":"2024-01-01"}\n',
        b'{"t":"date"}\n',
        b'{"t":"date","v":20240101}\n',
    ):
        with pytest.raises(runner.AlphaMaxRuntimeContractError):
            runner._parse_alpha_max_indicator_checkpoint_bytes(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("thread_identity", {"OMP_NUM_THREADS": "2"}),
        ("candidate_identity", {"path": "/tmp/candidate"}),
        ("runtime_identity", {"extension_module": "substitute"}),
        (
            "raw_roots",
            [
                {
                    "path": "/tmp/root",
                    "root_id": "warmup",
                    "root_kind": "raw",
                }
            ],
        ),
    ),
)
def test_indicator_day_checkpoint_descriptor_rejects_partial_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / field
    descriptor = _indicator_day_descriptor(root)
    descriptor[field] = value
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match=r"descriptor_(identity|roots)_invalid",
    ):
        runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)


def test_indicator_day_checkpoint_store_rejects_gap_tamper_and_parent_swap(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    store = runner._AlphaMaxIndicatorDayCheckpointStore(
        root, descriptor=_indicator_day_descriptor(root)
    )
    start = __import__("datetime").datetime(2024, 1, 1, tzinfo=__import__("datetime").UTC)
    carry = runner._AlphaMaxIndicatorDayCarry(
        next_day_start_utc=start + __import__("datetime").timedelta(days=1),
        strategy_state={"bucket": (1, 2)},
        aggregator_state={"working": {"4h": [1]}},
        windows_processed=86_400,
        discarded_signal_count=7,
    )
    store.seal(carry)
    assert (
        store.load_latest(
            start_utc=start,
            end_utc=__import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").UTC),
        )
        == carry
    )
    (root / "days" / "20240103").mkdir()
    with pytest.raises(runner.AlphaMaxRuntimeContractError):
        store.load_latest(
            start_utc=start,
            end_utc=__import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").UTC),
        )


def test_indicator_day_checkpoint_seal_revalidates_prior_days_by_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dt = __import__("datetime")
    root = tmp_path / "checkpoint"
    store = runner._AlphaMaxIndicatorDayCheckpointStore(
        root, descriptor=_indicator_day_descriptor(root)
    )
    start = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    store.seal(
        runner._AlphaMaxIndicatorDayCarry(
            next_day_start_utc=start + dt.timedelta(days=1),
            strategy_state={"day": 1},
            aggregator_state={"history": {}},
            windows_processed=86_400,
            discarded_signal_count=1,
        )
    )
    original = runner._alpha_max_read_regular_at
    reads: list[str] = []

    def instrumented(
        directory_fd: int,
        name: str,
        *,
        expected_mode: int,
    ) -> bytes:
        reads.append(name)
        return original(directory_fd, name, expected_mode=expected_mode)

    monkeypatch.setattr(runner, "_alpha_max_read_regular_at", instrumented)
    store.seal(
        runner._AlphaMaxIndicatorDayCarry(
            next_day_start_utc=start + dt.timedelta(days=2),
            strategy_state={"day": 2},
            aggregator_state={"history": {}},
            windows_processed=172_800,
            discarded_signal_count=2,
        )
    )

    assert reads.count("STATE.json") == 1
    assert reads.count("SEALED.json") == 1
    assert (
        store.load_latest(
            start_utc=start,
            end_utc=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        ).windows_processed
        == 172_800
    )


def test_indicator_day_checkpoint_cached_identity_rejects_prior_tamper(
    tmp_path: Path,
) -> None:
    dt = __import__("datetime")
    root = tmp_path / "checkpoint"
    store = runner._AlphaMaxIndicatorDayCheckpointStore(
        root, descriptor=_indicator_day_descriptor(root)
    )
    start = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    store.seal(
        runner._AlphaMaxIndicatorDayCarry(
            next_day_start_utc=start + dt.timedelta(days=1),
            strategy_state={},
            aggregator_state={},
            windows_processed=86_400,
            discarded_signal_count=0,
        )
    )
    state = root / "days" / "20240101" / "STATE.json"
    state_stat = state.stat()
    os.utime(
        state,
        ns=(state_stat.st_atime_ns, state_stat.st_mtime_ns + 1_000_000_000),
    )

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="checkpoint_identity_changed",
    ):
        store.seal(
            runner._AlphaMaxIndicatorDayCarry(
                next_day_start_utc=start + dt.timedelta(days=2),
                strategy_state={},
                aggregator_state={},
                windows_processed=172_800,
                discarded_signal_count=0,
            )
        )


def test_indicator_day_checkpoint_post_rename_fsync_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dt = __import__("datetime")
    root = tmp_path / "checkpoint"
    store = runner._AlphaMaxIndicatorDayCheckpointStore(
        root, descriptor=_indicator_day_descriptor(root)
    )
    start = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    published = root / "days" / "20240101"
    original_fsync = runner.os.fsync

    def fail_after_rename(descriptor: int) -> None:
        if descriptor == store._days_fd and published.exists():
            raise OSError("injected day parent fsync")
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", fail_after_rename)
    with pytest.raises(OSError, match="injected day parent fsync"):
        store.seal(
            runner._AlphaMaxIndicatorDayCarry(
                next_day_start_utc=start + dt.timedelta(days=1),
                strategy_state={},
                aggregator_state={},
                windows_processed=86_400,
                discarded_signal_count=0,
            )
        )

    assert not published.exists()
    assert (
        store.load_latest(
            start_utc=start,
            end_utc=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        )
        is None
    )


def test_indicator_day_checkpoint_initialization_failure_leaves_no_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"

    def fail_publish(_stage: Path, _target: Path) -> None:
        raise OSError("instrumented-init-crash")

    monkeypatch.setattr(runner, "_rename_bundle_noreplace", fail_publish)
    with pytest.raises(OSError, match="instrumented-init-crash"):
        runner._AlphaMaxIndicatorDayCheckpointStore(
            root, descriptor=_indicator_day_descriptor(root)
        )

    assert not root.exists()
    assert not tuple(
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(".checkpoint.alpha-max-indicator-init.staging-")
    )


def test_indicator_day_checkpoint_store_cleans_only_exact_staging_name(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    store = runner._AlphaMaxIndicatorDayCheckpointStore(
        root, descriptor=_indicator_day_descriptor(root)
    )
    staging = root / "days" / ".alpha-max-indicator-day-20240101.staging-deadbeef"
    staging.mkdir()
    start = __import__("datetime").datetime(2024, 1, 1, tzinfo=__import__("datetime").UTC)
    end = __import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").UTC)

    assert store.load_latest(start_utc=start, end_utc=end) is None
    assert not staging.exists()

    unknown = root / "days" / ".alpha-max-indicator-day-20240101.staging"
    unknown.mkdir()
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="checkpoint_unknown_entry",
    ):
        store.load_latest(start_utc=start, end_utc=end)

    unknown.rmdir()
    out_of_range = root / "days" / ".alpha-max-indicator-day-20250101.staging-deadbeef"
    out_of_range.mkdir()
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="checkpoint_unknown_entry",
    ):
        store.load_latest(start_utc=start, end_utc=end)


def test_indicator_day_checkpoint_store_rejects_out_of_sequence_and_parent_swap(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "checkpoint-parent"
    parent.mkdir()
    root = parent / "checkpoint"
    store = runner._AlphaMaxIndicatorDayCheckpointStore(
        root, descriptor=_indicator_day_descriptor(root)
    )
    start = __import__("datetime").datetime(2024, 1, 1, tzinfo=__import__("datetime").UTC)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="carry_sequence_invalid",
    ):
        store.seal(
            runner._AlphaMaxIndicatorDayCarry(
                next_day_start_utc=start + __import__("datetime").timedelta(days=2),
                strategy_state={},
                aggregator_state={},
                windows_processed=172_800,
                discarded_signal_count=0,
            )
        )
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="checkpoint_carry_invalid",
    ):
        store.seal(
            runner._AlphaMaxIndicatorDayCarry(
                next_day_start_utc=start + __import__("datetime").timedelta(days=1),
                strategy_state=(),
                aggregator_state={},
                windows_processed=86_400,
                discarded_signal_count=0,
            )
        )

    moved = tmp_path / "checkpoint-parent-moved"
    parent.rename(moved)
    parent.mkdir()
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="checkpoint_parent_replaced",
    ):
        store.load_latest(
            start_utc=start,
            end_utc=__import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").UTC),
        )


def test_indicator_day_checkpoint_descriptor_accepts_ordered_admitted_subset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "subset"
    descriptor = _indicator_day_descriptor(root)
    descriptor["admitted_symbols"] = list(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS[:5])

    store = runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)

    assert store._descriptor["admitted_symbols"] == list(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS[:5])


def test_indicator_day_checkpoint_rejects_protected_path_overlap(tmp_path: Path) -> None:
    root = tmp_path / "root"
    descriptor = _indicator_day_descriptor(root)

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_indicator_checkpoint_path_overlap",
    ):
        runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)
    assert not root.exists()


def test_indicator_day_checkpoint_rejects_symlinked_parent_alias(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "subdir").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)
    root = alias / "subdir/checkpoint"
    descriptor = _indicator_day_descriptor(root)
    descriptor["raw_roots"][0]["path"] = str(protected.resolve())

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_indicator_checkpoint_parent_invalid",
    ):
        runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)
    assert not root.exists()


def test_indicator_day_checkpoint_rejects_postconstruction_ancestor_alias(
    tmp_path: Path,
) -> None:
    container = tmp_path / "container"
    parent = container / "subdir"
    parent.mkdir(parents=True)
    root = parent / "checkpoint"
    store = runner._AlphaMaxIndicatorDayCheckpointStore(
        root,
        descriptor=_indicator_day_descriptor(root),
    )
    protected = tmp_path / "protected"
    protected.mkdir()
    moved = protected / "container"
    container.rename(moved)
    container.symlink_to(moved, target_is_directory=True)

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_indicator_checkpoint_parent_replaced",
    ):
        store.load_latest(
            start_utc=store._start_utc,
            end_utc=store._end_utc,
        )


def test_indicator_day_checkpoint_holds_exclusive_writer_lock(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    descriptor = _indicator_day_descriptor(root)
    first = runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_indicator_checkpoint_lock_unavailable",
    ):
        runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)

    lock_path = tmp_path / ".checkpoint.alpha-max-indicator.lock"
    displaced_lock = tmp_path / ".checkpoint.alpha-max-indicator.lock.displaced"
    lock_path.rename(displaced_lock)
    lock_path.write_bytes(b"")
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_indicator_checkpoint_parent_replaced",
    ):
        first.load_latest(
            start_utc=first._start_utc,
            end_utc=first._end_utc,
        )
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_indicator_checkpoint_lock_unavailable",
    ):
        runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)
    lock_path.unlink()
    displaced_lock.rename(lock_path)

    first.__del__()
    del first
    gc.collect()
    restarted = runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)
    assert (
        restarted.load_latest(
            start_utc=restarted._start_utc,
            end_utc=restarted._end_utc,
        )
        is None
    )


def test_indicator_day_checkpoint_store_rejects_root_and_days_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    store = runner._AlphaMaxIndicatorDayCheckpointStore(
        root, descriptor=_indicator_day_descriptor(root)
    )
    start = __import__("datetime").datetime(2024, 1, 1, tzinfo=__import__("datetime").UTC)
    end = __import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").UTC)
    moved = tmp_path / "checkpoint-moved"
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="checkpoint_parent_replaced",
    ):
        store.load_latest(start_utc=start, end_utc=end)

    second_root = tmp_path / "checkpoint-two"
    second = runner._AlphaMaxIndicatorDayCheckpointStore(
        second_root, descriptor=_indicator_day_descriptor(second_root)
    )
    second_root.chmod(0o700)
    (second_root / "days").rename(second_root / "days-moved")
    (second_root / "days").mkdir(mode=0o700)
    second_root.chmod(0o500)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="checkpoint_parent_replaced",
    ):
        second.load_latest(start_utc=start, end_utc=end)


def test_indicator_day_checkpoint_resume_preserves_partial_4h_1d_state_and_hashes(
    tmp_path: Path,
) -> None:
    dt = __import__("datetime")
    start = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    day_one_rows = (
        (int((start + dt.timedelta(hours=20)).timestamp() * 1000), 10.0, 12.0, 9.0, 11.0, 1.0e16),
        (
            int((start + dt.timedelta(hours=23, minutes=59, seconds=59)).timestamp() * 1000),
            11.0,
            13.0,
            10.0,
            12.0,
            1.0,
        ),
    )
    day_two_rows = (
        (int((start + dt.timedelta(days=1)).timestamp() * 1000), 12.0, 14.0, 11.0, 13.0, 1.0e-16),
        (
            int((start + dt.timedelta(days=1, seconds=1)).timestamp() * 1000),
            13.0,
            15.0,
            12.0,
            14.0,
            2.0,
        ),
    )

    uninterrupted = runner.TimeframeAggregator(timeframes=["4h", "1d"])
    assert uninterrupted.update_from_canonical_1s_rows_exact("BTCUSDT", day_one_rows) == ()
    uninterrupted_releases = uninterrupted.update_from_canonical_1s_rows_exact(
        "BTCUSDT", day_two_rows
    )

    before_restart = runner.TimeframeAggregator(timeframes=["4h", "1d"])
    assert before_restart.update_from_canonical_1s_rows_exact("BTCUSDT", day_one_rows) == ()
    strategy_state = {
        "child": ("partial", {1, "x"}, frozenset({2, "y"})),
        "when": start + dt.timedelta(hours=23, minutes=59, seconds=59),
    }
    root = tmp_path / "indicator-checkpoint"
    descriptor = _indicator_day_descriptor(root)
    store = runner._AlphaMaxIndicatorDayCheckpointStore(root, descriptor=descriptor)
    store.seal(
        runner._AlphaMaxIndicatorDayCarry(
            next_day_start_utc=start + dt.timedelta(days=1),
            strategy_state=strategy_state,
            aggregator_state=before_restart.get_state(),
            windows_processed=86_400,
            discarded_signal_count=7,
        )
    )
    carry = store.load_latest(start_utc=start, end_utc=end)
    assert carry is not None
    resumed = runner.TimeframeAggregator(timeframes=["4h", "1d"])
    resumed.set_state(copy.deepcopy(carry.aggregator_state))
    assert runner._exact_state_equal(resumed.get_state(), carry.aggregator_state)
    resumed_releases = resumed.update_from_canonical_1s_rows_exact("BTCUSDT", day_two_rows)
    restored_strategy_state = copy.deepcopy(carry.strategy_state)
    finalization_count = 0

    def finalize_once(aggregator, strategy):
        nonlocal finalization_count
        finalization_count += 1
        return _sha256(
            runner._alpha_max_indicator_checkpoint_bytes(
                {
                    "aggregator": aggregator.get_state(),
                    "strategy": strategy,
                }
            )
        )

    resumed_hash = finalize_once(resumed, restored_strategy_state)
    uninterrupted_hash = _sha256(
        runner._alpha_max_indicator_checkpoint_bytes(
            {
                "aggregator": uninterrupted.get_state(),
                "strategy": strategy_state,
            }
        )
    )
    assert finalization_count == 1
    assert resumed_hash == uninterrupted_hash
    assert runner._exact_state_equal(resumed.get_state(), uninterrupted.get_state())
    assert resumed_releases == uninterrupted_releases
    assert carry.windows_processed == 86_400
    assert carry.discarded_signal_count == 7


def _fake_root_seal(tmp_path: Path, root_id: str, root_kind: str):
    start, end = evidence._ROOT_INTERVALS[root_id]
    path = (tmp_path / f"{root_id}-{root_kind}").resolve()
    path.mkdir()
    starts = MappingProxyType(dict.fromkeys(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS, start))
    ends = MappingProxyType(dict.fromkeys(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS, end))
    seal = object.__new__(evidence.AlphaMaxRootSeal)
    values = {
        "root_id": root_id,
        "root_kind": root_kind,
        "path": str(path),
        "exchange": "binance",
        "symbols": evidence.ALPHA_MAX_CANDIDATE_SYMBOLS,
        "start_utc": start,
        "end_utc": end,
        "availability_start_by_symbol": starts,
        "availability_end_by_symbol": ends,
        "availability_sha256": evidence._alpha_max_availability_sha256(starts, ends),
        "entries": (),
        "inventory_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "canonical_bytes": b"",
        "sha256": "3" * 64,
    }
    for field, value in values.items():
        object.__setattr__(seal, field, value)
    return seal


def test_root_seal_parser_requires_exact_canonical_spawn_transport_bytes(
    tmp_path: Path,
) -> None:
    start, end = evidence._ROOT_INTERVALS["train"]
    root = (tmp_path / "train-raw").resolve()
    root.mkdir()
    availability_start = MappingProxyType(
        dict.fromkeys(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS, start)
    )
    availability_end = MappingProxyType(dict.fromkeys(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS, end))
    entries = []
    for symbol in evidence.ALPHA_MAX_CANDIDATE_SYMBOLS:
        partition_start = start.replace(day=1)
        while partition_start < end:
            partition_end = (
                partition_start.replace(year=partition_start.year + 1, month=1)
                if partition_start.month == 12
                else partition_start.replace(month=partition_start.month + 1)
            )
            owned_start = max(start, partition_start)
            owned_end = min(end, partition_end)
            row_count = int((owned_end - owned_start).total_seconds())
            entries.append(
                evidence.AlphaMaxTreeEntry(
                    relative_path=(
                        f"market_ohlcv_1s/binance/{symbol}/{partition_start:%Y-%m}.parquet"
                    ),
                    byte_count=1,
                    mode=0o444,
                    mtime_ns=0,
                    minimum_timestamp_ms=evidence._epoch_ms(owned_start),
                    maximum_timestamp_ms=evidence._epoch_ms(owned_end) - 1000,
                    row_count=row_count,
                    maximum_gap_ms=1000,
                    sha256=_sha256(f"{symbol}:{partition_start:%Y-%m}".encode()),
                )
            )
            partition_start = partition_end
    ordered_entries = tuple(sorted(entries, key=lambda entry: entry.relative_path))
    inventory_payload = [
        {
            "byte_count": entry.byte_count,
            "maximum_timestamp_ms": entry.maximum_timestamp_ms,
            "maximum_gap_ms": entry.maximum_gap_ms,
            "minimum_timestamp_ms": entry.minimum_timestamp_ms,
            "mode": entry.mode,
            "mtime_ns": entry.mtime_ns,
            "relative_path": entry.relative_path,
            "row_count": entry.row_count,
        }
        for entry in ordered_entries
    ]
    content_payload = [
        {
            "byte_count": entry.byte_count,
            "maximum_timestamp_ms": entry.maximum_timestamp_ms,
            "maximum_gap_ms": entry.maximum_gap_ms,
            "minimum_timestamp_ms": entry.minimum_timestamp_ms,
            "relative_path": entry.relative_path,
            "row_count": entry.row_count,
            "sha256": entry.sha256,
        }
        for entry in ordered_entries
    ]
    inventory_sha256 = evidence._sha256_bytes(
        evidence._canonical_json_bytes(inventory_payload, newline=True)
    )
    content_sha256 = evidence._sha256_bytes(
        evidence._canonical_json_bytes(content_payload, newline=True)
    )
    availability_sha256 = evidence._alpha_max_availability_sha256(
        availability_start, availability_end
    )
    payload = {
        "artifact_kind": "alpha_max_root_seal.v2",
        "availability_end_by_symbol": {
            symbol: end.isoformat().replace("+00:00", "Z")
            for symbol in evidence.ALPHA_MAX_CANDIDATE_SYMBOLS
        },
        "availability_sha256": availability_sha256,
        "availability_start_by_symbol": {
            symbol: start.isoformat().replace("+00:00", "Z")
            for symbol in evidence.ALPHA_MAX_CANDIDATE_SYMBOLS
        },
        "content_sha256": content_sha256,
        "end_utc": end.isoformat().replace("+00:00", "Z"),
        "entries": [entry.to_payload() for entry in ordered_entries],
        "exchange": "binance",
        "file_count": len(ordered_entries),
        "inventory_sha256": inventory_sha256,
        "path": str(root),
        "root_id": "train",
        "root_kind": "raw",
        "start_utc": start.isoformat().replace("+00:00", "Z"),
        "symbols": list(evidence.ALPHA_MAX_CANDIDATE_SYMBOLS),
    }
    raw = runner._canonical_bytes(payload) + b"\n"
    parsed = evidence.parse_alpha_max_root_seal(
        raw,
        expected_root_id="train",
        expected_root_kind="raw",
        expected_sha256=_sha256(raw),
    )
    assert parsed.canonical_bytes == raw
    for tampered in (
        raw.replace(b'"root_id":"train"', b'"root_id":"warmup"'),
        raw.replace(b"\n", b" \n"),
        raw[:-1],
    ):
        with pytest.raises(ValueError, match="alpha_max_root_seal_parse_invalid"):
            evidence.parse_alpha_max_root_seal(
                tampered,
                expected_root_id="train",
                expected_root_kind="raw",
                expected_sha256=_sha256(tampered),
            )


def _native_finalization(fold_id: str, row_id: str):
    _start, end = evidence._ALPHA_MAX_FOLD_INTERVALS[fold_id]
    completed = [["BTCUSDT", end.date().isoformat()]]
    coverage = {
        "adapter_class": "CheckpointParserTestAdapter",
        "native_timeframe": "1d",
        "barrier_mode": "none",
        "completed_native_keys": completed,
        "completed_native_count_by_symbol": {"BTCUSDT": 1},
        "last_completed_native_key_by_symbol": {"BTCUSDT": end.date().isoformat()},
        "barrier_pending_keys": [],
        "barrier_closed_keys": [],
        "barrier_symbol_coverage": {},
        "failed_native_keys": {},
        "partial_bucket_error": None,
        "finalization_completed_native_keys": completed,
        "finalization_barrier_keys": [],
    }
    return evidence.build_alpha_max_native_finalization_receipt(
        boundary_utc=end,
        finalized_children={row_id: 1},
        native_coverage_by_child={row_id: coverage},
        discarded_signal_count=0,
        discarded_signal_sha256=_sha256(b""),
    )


def _pricing_pair(timeindex: str):
    traces = []
    model = ExecutionModel(
        ExecutionModelConfig(
            taker_fee_rate=0.0004,
            maker_fee_rate=0.0002,
            slippage_rate=0.0005,
            spread_rate=0.0002,
            leverage=3,
            margin_mode="isolated",
            maintenance_margin_rate=0.005,
            liquidation_buffer_rate=0.0005,
            funding_rate_per_8h=0.0,
            funding_interval_hours=8,
            random_seed=7,
            max_bar_volume_ratio=0.1,
            slippage_impact_model="sqrt_impact",
            slippage_impact_coefficient=0.1,
        )
    )
    model.compute_fill(
        raw_price=100.0,
        qty=2.0,
        direction="BUY",
        bar_volume=100.0,
        order_kind="MKT",
        order_id="O-1",
        attribution_sink=traces.append,
    )
    trace = traces[0]
    application = FillApplicationAttribution(
        record_type="fill_application_attribution",
        pricing_trace_hash=trace.sha256,
        pricing_trace=trace,
        timeindex=timeindex,
        symbol="BTCUSDT",
        direction="BUY",
        order_id="O-1",
        client_order_id=None,
        position_side=None,
        status=None,
        reduce_only=False,
        model_quantity=trace.executed_qty,
        model_fill_cost=trace.fill_price * trace.executed_qty,
        model_commission=trace.commission,
        applied_quantity=trace.executed_qty,
        applied_fill_cost=trace.fill_price * trace.executed_qty,
        applied_commission=trace.commission,
        reduce_only_scale=1.0,
        application_status="applied_unchanged",
        zero_applied_reason=None,
    )
    return trace, application


def _build_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ruin: bool = False,
    domain: str = "validation",
):
    for key in tuple(os.environ):
        if key.startswith("LQ_"):
            monkeypatch.delenv(key, raising=False)
    preflight = runner.preflight_alpha_max_runtime_contract(str(_CONFIG_PATH))
    admitted = evidence.ALPHA_MAX_CANDIDATE_SYMBOLS[:9]
    row_id = "component_carry_1x"
    manifest_phase = "validation_train_fit" if domain == "validation" else "prelock_final_refit"
    raw = _fake_root_seal(tmp_path, domain, "raw")
    features = (
        (
            _fake_root_seal(tmp_path, "purge", "feature"),
            _fake_root_seal(tmp_path, "validation", "feature"),
        )
        if domain == "validation"
        else (
            _fake_root_seal(tmp_path, "embargo", "feature"),
            _fake_root_seal(tmp_path, "historical_exposed_evaluation", "feature"),
        )
    )

    manifest_path = (tmp_path / "manifest.json").resolve()
    manifest_path.write_bytes(b"{}\n")
    manifest_read, _payload = evidence.read_artifact_bytes(
        manifest_path,
        artifact_id="alpha_max_engine_portfolio_manifest",
    )
    manifest = evidence.AlphaMaxManifestReceipt(
        row_id=row_id,
        phase=manifest_phase,
        relative_path=f"manifests/{manifest_phase}/component_carry_1x.json",
        sha256=manifest_read.sha256,
        byte_count=manifest_read.byte_count,
        activation_receipt=manifest_read,
    )

    def capsule_receipt(fold_id: str):
        predecessor, _boundary = evidence._alpha_max_capsule_scope(fold_id)
        capsule_scope = {"child_states": {}}
        capsule_scope["sha256"] = _sha256(
            evidence._canonical_json_bytes({"child_states": {}}, newline=False)
        )
        state = {
            "capsule": capsule_scope,
            "capsule_sha256": capsule_scope["sha256"],
            "discarded_signal_count": 0,
            "fill_event_count": 0,
            "finalized_children": {},
            "funding_event_count": 0,
            "manifest_sha256": manifest.sha256,
            "market_event_count": 0,
            "native_finalization_sha256": "4" * 64,
            "order_event_count": 0,
            "phase_id": predecessor,
            "portfolio_mode": "alpha_max_checkpoint_parser_test",
            "trade_count": 0,
            "windows_processed": 1,
        }
        envelope = evidence.AlphaMaxCapsuleReceipt.canonical_envelope_bytes(
            row_id=row_id,
            phase=manifest_phase,
            prefix_id=fold_id,
            manifest_sha256=manifest.sha256,
            state_payload=state,
        )
        path = (tmp_path / f"{fold_id}.json").resolve()
        path.write_bytes(envelope)
        return evidence.AlphaMaxCapsuleReceipt.from_path(
            path,
            row_id=row_id,
            phase=manifest_phase,
            prefix_id=fold_id,
            manifest_sha256=manifest.sha256,
            relative_path=f"capsules/{manifest_phase}/{row_id}/{fold_id}.json",
        )

    fold_runs = []
    fold_inputs = []
    aggregate = evidence.AlphaMaxStreamingEquityTracker()
    segments = []
    normalized_start = 10_000.0
    fold_ids = (
        evidence._ALPHA_MAX_VALIDATION_FOLD_IDS
        if domain == "validation"
        else evidence._ALPHA_MAX_HISTORICAL_FOLD_IDS
    )
    for fold_index, fold_id in enumerate(fold_ids):
        _start, end = evidence._ALPHA_MAX_FOLD_INTERVALS[fold_id]
        capsule = capsule_receipt(fold_id)
        fold_inputs.append(
            SimpleNamespace(
                fold_id=fold_id,
                capsule_receipt=capsule,
                raw_root_seals=(raw,),
                feature_root_seals=features,
            )
        )
        traces = ()
        applications = ()
        capacity = ()
        ending_cash = 10_000.0
        ending_equity = 10_000.0
        ending_market_values = {}
        portfolio_fee_total = 0.0
        terminal = ruin and fold_index == 0
        if terminal:
            ending_equity = -1.0
            ending_market_values = {"BTCUSDT": -10_001.0}
        elif fold_index == 0:
            trace, application = _pricing_pair(end.isoformat())
            traces = (trace,)
            applications = (application,)
            ending_equity -= trace.commission
            ending_cash -= application.applied_fill_cost + trace.commission
            ending_market_values = {"BTCUSDT": application.applied_fill_cost}
            portfolio_fee_total = trace.commission
            capacity = (
                {
                    "bar_volume": 100.0,
                    "equity_before": 10_000.0,
                    "raw_price": 100.0,
                    "requested_qty": 2.0,
                },
            )

        stream_tracker = evidence.AlphaMaxStreamingEquityTracker()
        stream_tracker.update(ending_equity, end)
        full_event_equity = stream_tracker.finalize()
        config = runner.build_alpha_max_backtest_config(
            preflight,
            phase_id=fold_id,
            admitted_symbols=admitted,
            nominal_cost_bps=20,
        )
        _ = config.START_DATE
        effective_config = config.runtime_attribute_bytes()
        runtime_audit = config.runtime_read_audit
        actual = evidence.build_alpha_max_actual_engine_run_receipt(
            row_id=row_id,
            domain=domain,
            split_or_fold_id=fold_id,
            nominal_cost_bps=20,
            raw_root_seals=(raw,),
            feature_root_seals=features,
            capsule_receipt=capsule,
            manifest_receipt=manifest,
            config_receipt=preflight.config_receipt,
            config_bytes=preflight.config_bytes,
            runtime_contract_bytes=preflight.runtime_contract_bytes,
            effective_config_bytes=effective_config,
            effective_config_sha256=config.runtime_instance_sha256,
            runtime_read_audit=runtime_audit,
            runtime_read_audit_sha256=_sha256(
                evidence._canonical_json_bytes(list(runtime_audit), newline=False)
            ),
            admitted_symbols=admitted,
            market_event_count=1,
            signal_event_count=len(traces),
            order_event_count=len(traces),
            fill_event_count=len(traces),
            trade_count=len(traces),
            starting_cash=10_000.0,
            starting_equity=10_000.0,
            starting_open_position_count=0,
            starting_open_order_count=0,
            starting_used_margin=0.0,
            ending_cash=ending_cash,
            ending_equity=ending_equity,
            full_event_equity=full_event_equity,
            native_finalization=_native_finalization(fold_id, row_id),
            pricing_traces=traces,
            fill_applications=applications,
            no_fill_attempts=(),
            funding_ledger=(),
            liquidation_events=(),
            portfolio_fee_total=portfolio_fee_total,
            portfolio_funding_total=0.0,
            capacity_observations=capacity,
            ending_market_values=ending_market_values,
            target_gross_exposure=1.0,
        )
        primary = None
        if not terminal:
            calendar = evidence._alpha_max_fold_reporting_calendar(fold_id)
            primary = evidence.build_alpha_max_primary_return_stream(
                tuple(
                    evidence.AlphaMaxEquityEndpoint(
                        timestamp=timestamp,
                        equity=(ending_equity if timestamp == calendar[-1] else 10_000.0),
                    )
                    for timestamp in calendar
                ),
                calendar,
            )
        fold_runs.append(evidence.build_alpha_max_fold_run_evidence(actual, primary))
        if not ruin:
            scale = normalized_start / 10_000.0
            normalized_end = scale * ending_equity
            aggregate.update(normalized_end, end)
            segment_tracker = evidence.AlphaMaxStreamingEquityTracker()
            segment_tracker.update(normalized_end, end)
            segments.append(
                evidence.build_alpha_max_normalized_fold_segment_evidence(
                    fold_id=fold_id,
                    source_streaming_equity_sha256=full_event_equity.sha256,
                    source_event_stream_sha256=full_event_equity.event_stream_sha256,
                    normalization_scale=scale,
                    normalized_starting_equity=normalized_start,
                    normalized_ending_equity=normalized_end,
                    normalized_segment_event_stream_sha256=(segment_tracker.event_stream_sha256),
                    event_count=1,
                    first_timestamp_ms=full_event_equity.first_timestamp_ms,
                    last_timestamp_ms=full_event_equity.last_timestamp_ms,
                    aggregate_prefix_event_count=aggregate.event_count,
                    aggregate_prefix_event_stream_sha256=aggregate.event_stream_sha256,
                )
            )
            normalized_start = normalized_end

    if ruin:
        cell = evidence.build_alpha_max_cost_cell_pre_gate_evidence(tuple(fold_runs))
    else:
        cell = evidence.build_alpha_max_cost_cell_pre_gate_evidence(
            tuple(fold_runs),
            aggregate.finalize(),
            tuple(segments),
        )
    capsules = {
        fold.actual_engine_run.capsule_receipt.sha256: (fold.actual_engine_run.capsule_receipt)
        for fold in fold_runs
    }
    roots = {
        f"{receipt.root_id}:{receipt.root_kind}:{receipt.content_sha256}": receipt
        for receipt in (raw.to_receipt(), *(value.to_receipt() for value in features))
    }
    prepared = runner._AlphaMaxPreparedReplayRow(
        manifest_receipt=manifest,
        fold_inputs=tuple(fold_inputs),
        gross=1.0,
    )
    return SimpleNamespace(
        cell=cell,
        preflight=preflight,
        manifest=manifest,
        capsules=capsules,
        roots=roots,
        prepared=prepared,
    )


def _parse(context, payload=None, **overrides):
    arguments = {
        "manifest_receipt": context.manifest,
        "config_receipt": context.preflight.config_receipt,
        "capsule_receipts_by_sha256": context.capsules,
        "root_receipts_by_identity": context.roots,
        "runtime_contract_sha256": _sha256(context.preflight.runtime_contract_bytes),
    }
    arguments.update(overrides)
    return evidence.parse_alpha_max_cost_cell_pre_gate_evidence(
        context.cell.to_payload() if payload is None else payload,
        **arguments,
    )


def test_training_day_checkpoint_rejects_tamper_and_out_of_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    day = datetime(2024, 1, 1, tzinfo=UTC)
    carry = runner._AlphaMaxDailyCarry(
        strategy_state={},
        portfolio_state={},
        execution_state={},
        engine_state={},
        handler_rows=(),
        handler_timestamps_ms=(),
        funding_ledger=(),
    )
    payload = runner._alpha_max_training_day_checkpoint_bytes(
        component_id=context.manifest.row_id,
        manifest=context.manifest,
        prefix_sha256="a" * 64,
        day_start=day,
        carry=carry,
        calendar_day="2024-01-02",
        endpoint_equity=10_100.0,
        daily_return=0.01,
        ordinal=1,
        previous_data_sha256="",
    )
    restored, endpoint, daily_return = runner._alpha_max_training_day_from_checkpoint(
        payload,
        component_id=context.manifest.row_id,
        manifest=context.manifest,
        prefix_sha256="a" * 64,
        expected_day_start=day,
        ordinal=1,
        previous_data_sha256="",
    )
    assert restored == carry
    assert (endpoint, daily_return) == (10_100.0, 0.01)
    negative_payload = runner._alpha_max_training_day_checkpoint_bytes(
        component_id=context.manifest.row_id,
        manifest=context.manifest,
        prefix_sha256="a" * 64,
        day_start=day,
        carry=carry,
        calendar_day="2024-01-02",
        endpoint_equity=-100.0,
        daily_return=-1.01,
        ordinal=1,
        previous_data_sha256="",
    )
    _, negative_endpoint, negative_return = runner._alpha_max_training_day_from_checkpoint(
        negative_payload,
        component_id=context.manifest.row_id,
        manifest=context.manifest,
        prefix_sha256="a" * 64,
        expected_day_start=day,
        ordinal=1,
        previous_data_sha256="",
    )
    assert (negative_endpoint, negative_return) == (-100.0, -1.01)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_training_day_endpoint_zero",
    ):
        runner._alpha_max_training_day_checkpoint_bytes(
            component_id=context.manifest.row_id,
            manifest=context.manifest,
            prefix_sha256="a" * 64,
            day_start=day,
            carry=carry,
            calendar_day="2024-01-02",
            endpoint_equity=0.0,
            daily_return=-1.0,
            ordinal=1,
            previous_data_sha256="",
        )
    with pytest.raises(runner.AlphaMaxRuntimeContractError):
        runner._alpha_max_training_day_from_checkpoint(
            payload,
            component_id=context.manifest.row_id,
            manifest=context.manifest,
            prefix_sha256="a" * 64,
            expected_day_start=day,
            ordinal=2,
            previous_data_sha256="",
        )


def test_precompute_transaction_lock_replacement_is_rejected(tmp_path: Path) -> None:
    output = (tmp_path / "output").resolve()
    checkpoint = (tmp_path / "checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=b'{"config":"test"}\n',
    )
    journal = store.training_precompute_store()
    lock = checkpoint / "precompute/units/.transaction.lock"
    replacement = lock.with_suffix(".replacement")
    replacement.write_bytes(b"")
    os.chmod(replacement, 0o600)
    replacement.replace(lock)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="transaction_lock",
    ):
        journal.load(unit_kind="training_prefix", unit_id="component_carry_1x")


def _v3_cell_descriptor(
    checkpoint: Path,
    output: Path,
    *,
    domain: str,
) -> dict[str, object]:
    role = "prelock" if domain == "validation" else "historical"
    schedule = [
        {
            "fold_id": fold_id,
            "nominal_cost_bps": nominal,
            "row_id": row_id,
            "seed": runner.alpha_max_common_rng_seed(fold_id, nominal),
        }
        for row_id, nominal, fold_id in runner._alpha_max_physical_fold_schedule(domain)
    ]
    descriptor: dict[str, object] = {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v3",
        "attempt_role": role,
        "domain": domain,
        "checkpoint_unit": "whole_row_cost_cell",
        "checkpoint": {
            "parent": str(checkpoint.parent),
            "parent_identity": [
                checkpoint.parent.stat().st_dev,
                checkpoint.parent.stat().st_ino,
            ],
            "root": str(checkpoint),
        },
        "config": {},
        "contract_manifest": {},
        "cost_cells_bps": list(runner.ALPHA_MAX_COST_CELL_BPS),
        "logical_cell_count": 68,
        "implementation_inventory": runner._alpha_max_checkpoint_implementation_inventory(),
        "immutable": True,
        "order_routing_enabled": False,
        "output": {
            "parent": str(output.parent),
            "parent_identity": [
                output.parent.stat().st_dev,
                output.parent.stat().st_ino,
            ],
            "target": str(output),
        },
        "phase_windows": (
            {}
            if role == "historical"
            else {
                "train": {
                    "start_utc": "2024-01-01T00:00:00Z",
                    "end_utc": "2024-01-03T00:00:00Z",
                }
            }
        ),
        "physical_fold_run_count": len(schedule),
        "physical_schedule": schedule,
        "physical_schedule_sha256": _sha256(runner._canonical_bytes(schedule)),
        "python": {},
        "root_seals": [],
        "runtime_identity": runner._alpha_max_indicator_runtime_binding(),
        "runtime_contract_sha256": "a" * 64,
        "thread_contract": {},
        "training_worker_transport": {
            "binding_schema": "alpha_max_training_component_worker_binding.v1",
            "maximum_component_processes": 3,
            "result_statuses": ["complete", "semantic_failure"],
            "start_method": "spawn",
        },
        "universe": {},
    }
    if role == "historical":
        descriptor["prelock_binding"] = {
            "immutable_prelock_seal_sha256": "b" * 64,
            "validated_snapshot_sha256": "c" * 64,
        }
    else:
        descriptor["prior_trial_blob"] = {
            "byte_count": 1,
            "path": str((checkpoint.parent / "prior-trials.json").resolve()),
            "sha256": "d" * 64,
        }
    return descriptor


def test_complete_parser_round_trip_preserves_nonempty_pricing_bijection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    original = evidence.reconcile_alpha_max_cost_attribution
    observed_identity = []

    def checking_reconciliation(pricing_traces, applications, *args, **kwargs):
        observed_identity.append(
            len(pricing_traces) == len(applications) == 1
            and applications[0].pricing_trace is pricing_traces[0]
        )
        return original(pricing_traces, applications, *args, **kwargs)

    monkeypatch.setattr(
        evidence,
        "reconcile_alpha_max_cost_attribution",
        checking_reconciliation,
    )
    parsed = _parse(context)
    assert parsed == context.cell
    assert parsed.canonical_bytes == context.cell.canonical_bytes
    assert parsed.sha256 == context.cell.sha256
    assert observed_identity[0] is True
    assert observed_identity[1:] == [False] * 11


def test_ruin_parser_round_trip_retains_every_fold_and_no_combined_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch, ruin=True)
    parsed = _parse(context)
    assert parsed == context.cell
    assert parsed.status == "ruin_detected"
    assert len(parsed.fold_runs) == 12
    assert parsed.combined_primary_return_stream is None
    assert parsed.combined_streaming_equity is None
    assert parsed.metric_statistics is None


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update(extra_field=True),
        lambda payload: payload.update(fold_run_count=False),
        lambda payload: payload.update(nominal_cost_bps=20.0),
        lambda payload: payload["fold_runs"][0]["actual_engine_run"].update(market_event_count=2),
        lambda payload: payload["fold_runs"][0]["actual_engine_run"]["full_event_equity"].update(
            event_count=2
        ),
    ),
)
def test_parser_rejects_schema_numeric_alias_and_nested_derived_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    payload = copy.deepcopy(context.cell.to_payload())
    mutation(payload)
    with pytest.raises((TypeError, ValueError)):
        _parse(context, payload)


def test_parser_rejects_missing_unknown_or_mismatched_live_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    missing_capsules = dict(context.capsules)
    missing_capsules.pop(next(iter(missing_capsules)))
    with pytest.raises(ValueError):
        _parse(context, capsule_receipts_by_sha256=missing_capsules)

    missing_roots = dict(context.roots)
    missing_roots.pop(next(iter(missing_roots)))
    with pytest.raises(ValueError):
        _parse(context, root_receipts_by_identity=missing_roots)

    extra_capsules = dict(context.capsules)
    extra_capsules["f" * 64] = next(iter(context.capsules.values()))
    with pytest.raises(ValueError):
        _parse(context, capsule_receipts_by_sha256=extra_capsules)

    with pytest.raises(ValueError):
        _parse(context, runtime_contract_sha256="f" * 64)

    with pytest.raises(ValueError):
        _parse(
            context,
            config_receipt=context.manifest.activation_receipt,
        )


def test_checkpoint_store_seals_and_reloads_exact_typed_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    output = (tmp_path / "official-output").resolve()
    checkpoint = (tmp_path / "checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )
    sealed = store.seal(
        context.cell,
        preflight=context.preflight,
        prepared=context.prepared,
    )
    descriptor_sha256 = store.descriptor_sha256
    cell_path = checkpoint / "cells" / store._cell_name(context.cell.row_id, 20)
    inode_before = cell_path.stat().st_ino
    assert sealed == context.cell
    assert tuple(sorted(path.name for path in cell_path.iterdir())) == (
        "EVIDENCE.json",
        "SEALED.json",
    )
    seal = __import__("json").loads((cell_path / "SEALED.json").read_bytes())
    assert seal["artifact_kind"] == "alpha_max_restartable_cost_cell_seal.v2"
    assert set(seal) == {
        "artifact_kind",
        "attempt_descriptor_sha256",
        "byte_count",
        "capsule_receipt_sha256s",
        "cell_name",
        "config_sha256",
        "domain",
        "evidence_sha256",
        "fold_count",
        "fold_ids",
        "fold_run_set_sha256",
        "manifest_sha256",
        "nominal_cost_bps",
        "physical_schedule_sha256",
        "prelock_binding",
        "root_receipt_identities",
        "row_id",
        "runtime_contract_sha256",
        "runtime_identity_sha256",
        "success",
    }
    store.__del__()
    del store
    gc.collect()

    restarted = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )
    loaded = restarted.load(
        row_id=context.cell.row_id,
        nominal_cost_bps=20,
        preflight=context.preflight,
        prepared=context.prepared,
    )
    assert restarted.descriptor_sha256 == descriptor_sha256
    assert cell_path.stat().st_ino == inode_before
    assert loaded == context.cell


def test_v3_prelock_descriptor_requires_prior_trial_blob_binding(
    tmp_path: Path,
) -> None:
    descriptor = _v3_cell_descriptor(
        (tmp_path / "prelock-checkpoint").resolve(),
        (tmp_path / "prelock-output").resolve(),
        domain="validation",
    )

    assert runner._alpha_max_validate_checkpoint_descriptor(descriptor) == (
        "prelock",
        "validation",
    )
    descriptor.pop("prior_trial_blob")
    with pytest.raises(runner.AlphaMaxRuntimeContractError):
        runner._alpha_max_validate_checkpoint_descriptor(descriptor)


def test_v3_checkpoint_rejects_loaded_native_identity_mismatch(tmp_path: Path) -> None:
    output = (tmp_path / "prelock-output").resolve()
    checkpoint = (tmp_path / "prelock-checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    descriptor["runtime_identity"]["extension_sha256"] = "f" * 64

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_checkpoint_runtime_identity_mismatch",
    ):
        runner._AlphaMaxCellCheckpointStore(
            checkpoint,
            output_root=output,
            descriptor=descriptor,
            config_bytes=b'{"config":"test"}\n',
        )
    assert not checkpoint.exists()


def test_loaded_mapping_identity_rejects_path_replacement(tmp_path: Path) -> None:
    mapped_path = (tmp_path / "mapped-native.so").resolve()
    mapped_path.write_bytes(b"x" * 4096)
    fd = os.open(mapped_path, os.O_RDONLY)
    mapping = mmap.mmap(fd, 4096, access=mmap.ACCESS_READ)
    os.close(fd)
    try:
        status = mapped_path.stat()
        assert runner._alpha_max_loaded_mapping_identity(mapped_path) == (
            status.st_dev,
            status.st_ino,
        )
        displaced = tmp_path / "mapped-native-original.so"
        mapped_path.rename(displaced)
        mapped_path.write_bytes(b"y" * 4096)
        with pytest.raises(
            runner.AlphaMaxRuntimeContractError,
            match="alpha_max_indicator_native_identity_invalid",
        ):
            runner._alpha_max_loaded_mapping_identity(mapped_path)
        assert displaced.read_bytes() == b"x" * 4096
    finally:
        mapping.close()


def test_loaded_mapping_identity_binds_builtin_function_address() -> None:
    from lumina_quant import _compute

    extension_path = Path(_compute.__file__).resolve(strict=True)
    status = extension_path.stat()
    assert runner._alpha_max_loaded_mapping_identity(
        extension_path,
        (
            _compute.fold_alpha_max_native_bars,
            _compute.build_info,
            _compute.kernel_src_hash,
        ),
    ) == (status.st_dev, status.st_ino)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_indicator_native_identity_invalid",
    ):
        runner._alpha_max_loaded_mapping_identity(extension_path, (len,))


def test_precompute_checkpoint_store_seals_reloads_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "prelock-output").resolve()
    checkpoint = (tmp_path / "prelock-checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    config_bytes = b'{"config":"test"}\n'
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=config_bytes,
    )
    payload = (
        runner._canonical_bytes(
            {
                "artifact_kind": "test_precompute_payload.v1",
                "component_id": "component_carry_1x",
                "values": ["0x0.0p+0"],
            }
        )
        + b"\n"
    )

    assert (
        store.load_precompute(
            unit_kind="training_component",
            unit_id="component_carry_1x",
        )
        is None
    )
    assert (
        store.seal_precompute(
            unit_kind="training_component",
            unit_id="component_carry_1x",
            data_bytes=payload,
        )
        == payload
    )
    unit = (
        checkpoint
        / "precompute/units"
        / store._precompute_store._unit_name(
            "training_component",
            "component_carry_1x",
        )
    )
    assert stat.S_IMODE(unit.stat().st_mode) == 0o555
    assert stat.S_IMODE((unit / "DATA.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((unit / "SEALED.json").stat().st_mode) == 0o444
    unit_seal = runner._strict_json_object((unit / "SEALED.json").read_bytes())
    assert unit_seal["runtime_identity_sha256"] == store._runtime_identity_sha256
    store.__del__()
    del store
    gc.collect()

    restarted = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=config_bytes,
    )
    assert (
        restarted.load_precompute(
            unit_kind="training_component",
            unit_id="component_carry_1x",
        )
        == payload
    )
    os.chmod(unit / "DATA.json", 0o644)
    (unit / "DATA.json").write_bytes(payload.replace(b"carry", b"trend"))
    os.chmod(unit / "DATA.json", 0o444)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_precompute_unit_seal_invalid",
    ):
        restarted.load_precompute(
            unit_kind="training_component",
            unit_id="component_carry_1x",
        )


def test_precompute_poison_is_immutable_and_fails_closed(tmp_path: Path) -> None:
    journal = runner._AlphaMaxPrecomputeCheckpointStore(
        (tmp_path / "journal").resolve(),
        attempt_descriptor_sha256="a" * 64,
        attempt_role="prelock",
        domain="validation",
        runtime_identity_sha256="b" * 64,
    )
    journal.poison()
    marker = tmp_path / "journal/FAILED.json"
    assert stat.S_IMODE((tmp_path / "journal").stat().st_mode) == 0o500
    assert stat.S_IMODE(marker.stat().st_mode) == 0o400
    assert marker.stat().st_nlink == 1
    assert runner._strict_json_object(marker.read_bytes()) == {
        "artifact_kind": "alpha_max_precompute_attempt_failed.v1",
        "attempt_descriptor_sha256": "a" * 64,
        "success": False,
    }
    for operation in (
        lambda: journal.load(unit_kind="training_prefix", unit_id="component_carry_1x"),
        lambda: journal.seal(
            unit_kind="training_prefix",
            unit_id="component_carry_1x",
            data_bytes=b"{}\n",
        ),
        journal.poison,
    ):
        with pytest.raises(
            runner.AlphaMaxRuntimeContractError,
            match="alpha_max_precompute_attempt_poisoned",
        ):
            operation()
    with pytest.raises(PermissionError):
        os.unlink(marker)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    with pytest.raises(PermissionError):
        os.replace(replacement, marker)
    symlink = tmp_path / "replacement-link"
    os.symlink("ATTEMPT.json", symlink)
    with pytest.raises(PermissionError):
        os.replace(symlink, marker)


def test_precompute_poison_rejects_hardlink_marker_replacement(tmp_path: Path) -> None:
    root = (tmp_path / "journal").resolve()
    journal = runner._AlphaMaxPrecomputeCheckpointStore(
        root,
        attempt_descriptor_sha256="a" * 64,
        attempt_role="prelock",
        domain="validation",
        runtime_identity_sha256="b" * 64,
    )
    victim = tmp_path / "unclassified-failure-marker.json"
    victim.write_bytes(b"{}\n")
    os.link(victim, root / "FAILED.json")
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_precompute_root_invalid",
    ):
        journal.load(unit_kind="training_prefix", unit_id="component_carry_1x")


@pytest.mark.parametrize(
    ("root_mode", "marker"),
    (
        (0o500, False),
        (0o700, True),
    ),
)
def test_precompute_rejects_tampered_failure_mode_combinations(
    tmp_path: Path,
    root_mode: int,
    marker: bool,
) -> None:
    root = (tmp_path / "journal").resolve()
    journal = runner._AlphaMaxPrecomputeCheckpointStore(
        root,
        attempt_descriptor_sha256="a" * 64,
        attempt_role="prelock",
        domain="validation",
        runtime_identity_sha256="b" * 64,
    )
    if marker:
        (root / "FAILED.json").write_bytes(
            runner._canonical_bytes(
                {
                    "artifact_kind": "alpha_max_precompute_attempt_failed.v1",
                    "attempt_descriptor_sha256": "a" * 64,
                    "success": False,
                }
            )
            + b"\n"
        )
        os.chmod(root / "FAILED.json", 0o400)
    os.chmod(root, root_mode)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError, match="alpha_max_precompute_root_invalid"
    ):
        journal.load(unit_kind="training_prefix", unit_id="component_carry_1x")


def test_precompute_checkpoint_rejects_units_directory_replacement(tmp_path: Path) -> None:
    output = (tmp_path / "prelock-output").resolve()
    checkpoint = (tmp_path / "prelock-checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=b'{"config":"test"}\n',
    )
    precompute = checkpoint / "precompute"
    units = precompute / "units"
    displaced = precompute / "units-displaced"
    victim = tmp_path / "unclassified-units"
    victim.mkdir()
    units.rename(displaced)
    units.symlink_to(victim, target_is_directory=True)

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_precompute_root_invalid",
    ):
        store.load_precompute(
            unit_kind="training_component",
            unit_id="component_carry_1x",
        )
    assert not tuple(victim.iterdir())

    units.unlink()
    displaced.rename(units)


def test_cell_checkpoint_rejects_cells_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    output = (tmp_path / "prelock-output").resolve()
    checkpoint = (tmp_path / "prelock-checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )
    cells = checkpoint / "cells"
    displaced = checkpoint / "cells-displaced"
    victim = tmp_path / "unclassified-cells"
    victim.mkdir()
    cells.rename(displaced)
    cells.symlink_to(victim, target_is_directory=True)

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_checkpoint_root_invalid",
    ):
        store.load(
            row_id=context.cell.row_id,
            nominal_cost_bps=20,
            preflight=context.preflight,
            prepared=context.prepared,
        )
    assert not tuple(victim.iterdir())

    cells.unlink()
    displaced.rename(cells)


def test_cell_checkpoint_root_lock_survives_lock_path_replacement(tmp_path: Path) -> None:
    output = (tmp_path / "prelock-output").resolve()
    checkpoint = (tmp_path / "prelock-checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    config_bytes = b'{"config":"test"}\n'
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=config_bytes,
    )
    lock_path = tmp_path / ".prelock-output.alpha-max-restart.lock"
    displaced_lock = tmp_path / ".prelock-output.alpha-max-restart.lock.displaced"
    lock_path.rename(displaced_lock)
    lock_path.write_bytes(b"")

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_checkpoint_root_invalid",
    ):
        store.load_precompute(
            unit_kind="training_component",
            unit_id="component_carry_1x",
        )
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_checkpoint_lock_unavailable",
    ):
        runner._AlphaMaxCellCheckpointStore(
            checkpoint,
            output_root=output,
            descriptor=descriptor,
            config_bytes=config_bytes,
        )

    lock_path.unlink()
    displaced_lock.rename(lock_path)


def test_precompute_checkpoint_rejects_unknown_and_cleans_exact_staging(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "prelock-output").resolve()
    checkpoint = (tmp_path / "prelock-checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    config_bytes = b'{"config":"test"}\n'
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=config_bytes,
    )
    precompute = checkpoint / "precompute"
    valid_name = store._precompute_store._unit_name(
        "training_component",
        "component_carry_1x",
    )
    store.__del__()
    del store
    gc.collect()

    staging = precompute / "units" / f".{valid_name}.staging-abcdefgh"
    staging.mkdir()
    (staging / "DATA.json").write_bytes(b"partial")
    restarted = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=config_bytes,
    )
    assert not staging.exists()
    restarted.__del__()
    del restarted
    gc.collect()

    victim = tmp_path / "unclassified-work"
    victim.mkdir()
    symlink_stage = precompute / "units" / f".{valid_name}.staging-ijklmnop"
    symlink_stage.symlink_to(victim, target_is_directory=True)
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_precompute_inventory_invalid",
    ):
        runner._AlphaMaxCellCheckpointStore(
            checkpoint,
            output_root=output,
            descriptor=descriptor,
            config_bytes=config_bytes,
        )
    assert symlink_stage.is_symlink()
    assert victim.is_dir()
    symlink_stage.unlink()

    hardlink_stage = precompute / "units" / f".{valid_name}.staging-qrstuvwx"
    hardlink_stage.mkdir()
    outside_file = victim / "outside.json"
    outside_file.write_bytes(b"unclassified")
    os.link(outside_file, hardlink_stage / "DATA.json")
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_precompute_inventory_invalid",
    ):
        runner._AlphaMaxCellCheckpointStore(
            checkpoint,
            output_root=output,
            descriptor=descriptor,
            config_bytes=config_bytes,
        )
    assert outside_file.read_bytes() == b"unclassified"
    assert (hardlink_stage / "DATA.json").exists()
    (hardlink_stage / "DATA.json").unlink()
    hardlink_stage.rmdir()

    (precompute / "units" / "unknown-unit").mkdir()
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_precompute_inventory_invalid",
    ):
        runner._AlphaMaxCellCheckpointStore(
            checkpoint,
            output_root=output,
            descriptor=descriptor,
            config_bytes=config_bytes,
        )


def test_prepared_checkpoint_restart_republishes_exact_capsule_bytes_and_rejects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    checkpoint_bytes = runner._alpha_max_prepared_row_checkpoint_bytes(
        context.prepared,
        domain="validation",
    )
    original_envelopes = {
        value.fold_id: Path(value.capsule_receipt.path).read_bytes()
        for value in context.prepared.fold_inputs
    }
    root_seals = {
        (seal.root_id, seal.root_kind): seal
        for value in context.prepared.fold_inputs
        for seal in (*value.raw_root_seals, *value.feature_root_seals)
    }
    restored_root = (tmp_path / "restored-output").resolve()
    restored_root.mkdir()

    class RestoredFoldInput:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)

    monkeypatch.setattr(runner, "_AlphaMaxFoldReplayInput", RestoredFoldInput)
    monkeypatch.setattr(
        runner,
        "_AlphaMaxBoundedRawLoader",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runner,
        "_alpha_max_phase_lookup",
        lambda *_args, **_kwargs: object(),
    )

    restored = runner._alpha_max_restore_prepared_row_checkpoint(
        checkpoint_bytes,
        preflight=context.preflight,
        manifest=context.manifest,
        admitted_symbols=evidence.ALPHA_MAX_CANDIDATE_SYMBOLS[:9],
        root_seals=root_seals,
        domain="validation",
        gross=1.0,
        capsule_output_root=restored_root,
    )

    assert tuple(value.fold_id for value in restored.fold_inputs) == tuple(original_envelopes)
    for value in restored.fold_inputs:
        assert Path(value.capsule_receipt.path).read_bytes() == original_envelopes[value.fold_id]
        assert value.capsule_receipt.sha256 == _sha256(original_envelopes[value.fold_id])

    corrupted = runner._strict_json_object(checkpoint_bytes)
    corrupted["fold_capsules"][0]["envelope_base64"] = "Y29ycnVwdA=="
    corrupted_bytes = runner._canonical_bytes(corrupted) + b"\n"
    rejected_root = (tmp_path / "rejected-output").resolve()
    rejected_root.mkdir()
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_prepared_checkpoint_capsule_mismatch",
    ):
        runner._alpha_max_restore_prepared_row_checkpoint(
            corrupted_bytes,
            preflight=context.preflight,
            manifest=context.manifest,
            admitted_symbols=evidence.ALPHA_MAX_CANDIDATE_SYMBOLS[:9],
            root_seals=root_seals,
            domain="validation",
            gross=1.0,
            capsule_output_root=rejected_root,
        )
    assert not tuple(rejected_root.rglob("*.json"))


def test_historical_checkpoint_store_round_trip_binds_ten_fold_cell_and_prelock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(
        tmp_path,
        monkeypatch,
        domain="historical_exposed_evaluation",
    )
    output = (tmp_path / "historical-output").resolve()
    checkpoint = (tmp_path / "historical-checkpoint").resolve()
    descriptor = _v3_cell_descriptor(
        checkpoint,
        output,
        domain="historical_exposed_evaluation",
    )
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )

    sealed = store.seal(
        context.cell,
        preflight=context.preflight,
        prepared=context.prepared,
    )
    assert sealed == context.cell
    assert len(sealed.fold_runs) == 10
    assert sealed.domain == "historical_exposed_evaluation"
    cell_path = checkpoint / "cells" / store._cell_name(context.cell.row_id, 20)
    seal = __import__("json").loads((cell_path / "SEALED.json").read_bytes())
    assert seal["artifact_kind"] == "alpha_max_restartable_cost_cell_seal.v2"
    assert seal["domain"] == "historical_exposed_evaluation"
    assert seal["fold_count"] == 10
    assert seal["fold_ids"] == list(runner._ALPHA_MAX_HISTORICAL_FOLD_IDS)
    assert seal["runtime_identity_sha256"] == store._runtime_identity_sha256
    assert seal["physical_schedule_sha256"] == descriptor["physical_schedule_sha256"]
    assert seal["prelock_binding"] == descriptor["prelock_binding"]
    store.__del__()
    del store
    gc.collect()

    restarted = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )
    assert (
        restarted.load(
            row_id=context.cell.row_id,
            nominal_cost_bps=20,
            preflight=context.preflight,
            prepared=context.prepared,
        )
        == context.cell
    )
    restarted.__del__()
    del restarted
    gc.collect()

    mismatched = copy.deepcopy(descriptor)
    mismatched["prelock_binding"]["validated_snapshot_sha256"] = "d" * 64
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="checkpoint_descriptor_mismatch",
    ):
        runner._AlphaMaxCellCheckpointStore(
            checkpoint,
            output_root=output,
            descriptor=mismatched,
            config_bytes=context.preflight.config_bytes,
        )


def test_historical_checkpoint_descriptor_rejects_role_and_schedule_aliases(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "historical-output").resolve()
    checkpoint = (tmp_path / "historical-checkpoint").resolve()
    descriptor = _v3_cell_descriptor(
        checkpoint,
        output,
        domain="historical_exposed_evaluation",
    )
    wrong_role = copy.deepcopy(descriptor)
    wrong_role["attempt_role"] = "prelock"
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="descriptor_role_invalid",
    ):
        runner._alpha_max_validate_checkpoint_descriptor(wrong_role)
    wrong_schedule = copy.deepcopy(descriptor)
    wrong_schedule["physical_fold_run_count"] = 816
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="descriptor_schedule_invalid",
    ):
        runner._alpha_max_validate_checkpoint_descriptor(wrong_schedule)
    wrong_domain = copy.deepcopy(descriptor)
    wrong_domain["domain"] = "validation"
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="descriptor_role_invalid",
    ):
        runner._alpha_max_validate_checkpoint_descriptor(wrong_domain)
    wrong_prelock = copy.deepcopy(descriptor)
    wrong_prelock["prelock_binding"] = {
        "immutable_prelock_seal_sha256": "b" * 64,
    }
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="descriptor_prelock_binding_invalid",
    ):
        runner._alpha_max_validate_checkpoint_descriptor(wrong_prelock)


def test_legacy_v1_descriptor_is_version_invalid(tmp_path: Path) -> None:
    output = (tmp_path / "output").resolve()
    checkpoint = (tmp_path / "checkpoint").resolve()
    descriptor = {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v1",
        "attempt_role": "prelock",
        "implementation_inventory": [],
        "checkpoint": {
            "parent": str(checkpoint.parent),
            "parent_identity": [checkpoint.parent.stat().st_dev, checkpoint.parent.stat().st_ino],
            "root": str(checkpoint),
        },
        "output": {
            "parent": str(output.parent),
            "parent_identity": [output.parent.stat().st_dev, output.parent.stat().st_ino],
            "target": str(output),
        },
    }
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_checkpoint_descriptor_version_invalid",
    ):
        runner._alpha_max_validate_checkpoint_descriptor(descriptor)


def test_checkpoint_store_rejects_duplicate_key_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    output = (tmp_path / "official-output").resolve()
    checkpoint = (tmp_path / "checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )
    store.seal(
        context.cell,
        preflight=context.preflight,
        prepared=context.prepared,
    )
    cell = checkpoint / "cells" / store._cell_name(context.cell.row_id, 20)
    evidence_path = cell / "EVIDENCE.json"
    os.chmod(evidence_path, 0o644)
    evidence_path.write_bytes(b'{"value":1,"value":2}\n')
    os.chmod(evidence_path, 0o444)
    with pytest.raises(runner.AlphaMaxRuntimeContractError):
        store.load(
            row_id=context.cell.row_id,
            nominal_cost_bps=20,
            preflight=context.preflight,
            prepared=context.prepared,
        )


def test_resumable_inventory_rejects_unknown_file_and_cleans_only_named_temp(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "run").resolve()
    root.mkdir()
    (root / "inputs").mkdir()
    (root / "inputs/config.json").write_bytes(b"{}\n")
    unknown = root / "unknown.json"
    unknown.write_bytes(b"{}\n")
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match=r"alpha_max_resumable_inventory_unknown:unknown\.json",
    ):
        runner._alpha_max_collect_existing_artifacts(
            root,
            allowed_paths={"inputs/config.json"},
            required_paths={"inputs/config.json"},
        )

    unknown.unlink()
    temporary = root / ".SEALED.json.atomic-123-0123456789abcdef01234567"
    temporary.write_bytes(b"partial")
    runner._alpha_max_cleanup_atomic_bundle_temps(root)
    assert not temporary.exists()
    assert runner._alpha_max_collect_existing_artifacts(
        root,
        allowed_paths={"inputs/config.json"},
        required_paths={"inputs/config.json"},
    ) == {"inputs/config.json": b"{}\n"}


def test_historical_resumable_inventory_rejects_unknown_regular_file(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "historical-run").resolve()
    root.mkdir()
    activation_paths = runner._alpha_max_historical_activation_paths()
    assert len(activation_paths) == 155
    for relative in sorted(activation_paths):
        runner._write_bundle_file(root, relative, b"{}\n")
    unknown = root / "stale-output.json"
    unknown.write_bytes(b"{}\n")

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match=r"alpha_max_resumable_inventory_unknown:stale-output\.json",
    ):
        runner._alpha_max_collect_existing_artifacts(
            root,
            allowed_paths=activation_paths,
            required_paths=activation_paths,
        )
    assert unknown.read_bytes() == b"{}\n"


def test_checkpoint_store_rejects_replaced_descriptor_parent(
    tmp_path: Path,
) -> None:
    checkpoint_parent = (tmp_path / "checkpoint-parent").resolve()
    output_parent = (tmp_path / "output-parent").resolve()
    checkpoint_parent.mkdir()
    output_parent.mkdir()
    checkpoint = checkpoint_parent / "checkpoint"
    output = output_parent / "output"
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    displaced = tmp_path / "checkpoint-parent-original"
    checkpoint_parent.rename(displaced)
    checkpoint_parent.mkdir()
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_checkpoint_descriptor_parent_mismatch",
    ):
        runner._AlphaMaxCellCheckpointStore(
            checkpoint,
            output_root=output,
            descriptor=descriptor,
            config_bytes=b"{}\n",
        )
    assert list(checkpoint_parent.iterdir()) == []
    assert list(output_parent.iterdir()) == []


def test_run_root_creation_stays_in_bound_output_parent_after_path_replacement(
    tmp_path: Path,
) -> None:
    checkpoint_parent = (tmp_path / "checkpoint-parent").resolve()
    output_parent = (tmp_path / "output-parent").resolve()
    checkpoint_parent.mkdir()
    output_parent.mkdir()
    checkpoint = checkpoint_parent / "checkpoint"
    output = output_parent / "output"
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=b"{}\n",
    )
    displaced = tmp_path / "output-parent-original"
    output_parent.rename(displaced)
    output_parent.mkdir()

    root = runner._alpha_max_create_or_resume_run_root(
        store.output_root,
        config_bytes=b"{}\n",
        attempt_descriptor_sha256=store.descriptor_sha256,
    )

    assert root == store.output_root
    assert (displaced / "output" / "inputs/config.json").read_bytes() == b"{}\n"
    assert list(output_parent.iterdir()) == []


def test_cell_is_loadable_after_crash_immediately_after_atomic_directory_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    output = (tmp_path / "official-output").resolve()
    checkpoint = (tmp_path / "checkpoint").resolve()
    descriptor = _v3_cell_descriptor(checkpoint, output, domain="validation")
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )
    cell_path = checkpoint / "cells" / store._cell_name(context.cell.row_id, 20)
    original_fsync = runner._fsync_directory

    def crash_after_rename(path: Path) -> None:
        if Path(path).resolve() == (checkpoint / "cells").resolve() and cell_path.exists():
            raise RuntimeError("injected crash after directory rename")
        original_fsync(path)

    monkeypatch.setattr(runner, "_fsync_directory", crash_after_rename)
    with pytest.raises(RuntimeError, match="injected crash"):
        store.seal(
            context.cell,
            preflight=context.preflight,
            prepared=context.prepared,
        )
    assert cell_path.stat().st_mode & 0o222 == 0
    store.__del__()
    monkeypatch.setattr(runner, "_fsync_directory", original_fsync)

    restarted = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )
    assert (
        restarted.load(
            row_id=context.cell.row_id,
            nominal_cost_bps=20,
            preflight=context.preflight,
            prepared=context.prepared,
        )
        == context.cell
    )


def test_writable_final_seal_is_recovered_to_immutable_or_rejected(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "prelock").resolve()
    config_bytes = b'{"config":true}\n'
    attempt_sha = "a" * 64
    root = runner._alpha_max_create_or_resume_run_root(
        output,
        config_bytes=config_bytes,
        attempt_descriptor_sha256=attempt_sha,
    )
    binding_bytes = (root / "inputs/restart_attempt.json").read_bytes()
    extra = b'{"complete":true}\n'
    artifacts = {
        "inputs/config.json": config_bytes,
        "inputs/restart_attempt.json": binding_bytes,
        "status/complete.json": extra,
    }
    seal = evidence.build_alpha_max_prelock_seal(
        artifacts,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    runner._write_bundle_file_atomic(root, "status/complete.json", extra)
    runner._write_bundle_file_atomic(root, "SEALED.json", seal.canonical_bytes)
    assert root.stat().st_mode & 0o200
    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_output_root_recovered_sealed",
    ):
        runner._alpha_max_create_or_resume_run_root(
            output,
            config_bytes=config_bytes,
            attempt_descriptor_sha256=attempt_sha,
        )


def test_canonical_final_seal_becomes_valid_only_after_tree_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "prelock").resolve()
    config_bytes = b'{"config":true}\n'
    root = runner._alpha_max_create_or_resume_run_root(
        output,
        config_bytes=config_bytes,
        attempt_descriptor_sha256="b" * 64,
    )
    artifacts = {
        "inputs/config.json": config_bytes,
        "inputs/restart_attempt.json": (root / "inputs/restart_attempt.json").read_bytes(),
        "status/complete.json": b'{"complete":true}\n',
    }
    seal = evidence.build_alpha_max_prelock_seal(
        artifacts,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    observed: list[bool] = []
    original = runner._alpha_max_write_final_seal

    def write_and_observe(fd: int, payload: bytes) -> None:
        original(fd, payload)
        snapshot = runner._snapshot_bundle_tree(root)
        runner._validate_prelock_snapshot(snapshot)
        observed.append(True)

    monkeypatch.setattr(runner, "_alpha_max_write_final_seal", write_and_observe)
    runner._finalize_alpha_max_run_owned_root(
        root,
        artifacts,
        seal_bytes=seal.canonical_bytes,
    )
    assert observed == [True]
    assert all(path.stat().st_mode & 0o222 == 0 for path in (root, *root.rglob("*")))
    runner._validate_prelock_snapshot(runner._snapshot_bundle_tree(root))


@pytest.mark.parametrize("after_seal", (False, True))
def test_finalization_failure_removes_command_owned_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_seal: bool,
) -> None:
    output = (tmp_path / "failed-finalization").resolve()
    root = runner._create_alpha_max_run_owned_root(output)
    artifacts = {"status/complete.json": b'{"complete":true}\n'}
    if after_seal:
        original = runner._alpha_max_write_final_seal

        def fail_after_seal(fd: int, payload: bytes) -> None:
            original(fd, payload)
            raise OSError("injected after seal fsync")

        monkeypatch.setattr(runner, "_alpha_max_write_final_seal", fail_after_seal)
    else:

        def fail_before_seal(_root: Path) -> None:
            raise OSError("injected before seal creation")

        monkeypatch.setattr(runner, "_make_bundle_immutable", fail_before_seal)
    with pytest.raises(OSError, match="injected"):
        runner._finalize_alpha_max_run_owned_root(
            root,
            artifacts,
            seal_bytes=b'{"artifact_kind":"test"}\n',
        )
    assert not output.exists()


def test_delayed_final_seal_close_error_cannot_bypass_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "failed-close").resolve()
    root = runner._create_alpha_max_run_owned_root(output)
    original_create = runner._alpha_max_create_empty_seal
    original_close = runner.os.close
    seal_descriptors: set[int] = set()
    close_attempts: list[int] = []

    def create(root_value: Path) -> tuple[Path, int]:
        path, descriptor = original_create(root_value)
        seal_descriptors.add(descriptor)
        return path, descriptor

    def delayed_close(descriptor: int) -> None:
        if descriptor in seal_descriptors:
            close_attempts.append(descriptor)
            seal_descriptors.remove(descriptor)
            original_close(descriptor)
            raise OSError("injected delayed seal close")
        original_close(descriptor)

    monkeypatch.setattr(runner, "_alpha_max_create_empty_seal", create)
    monkeypatch.setattr(runner.os, "close", delayed_close)
    with pytest.raises(OSError, match="delayed seal close"):
        runner._finalize_alpha_max_run_owned_root(
            root,
            {"status/complete.json": b'{"complete":true}\n'},
            seal_bytes=b'{"artifact_kind":"test"}\n',
        )

    assert len(close_attempts) == 1
    assert not output.exists()


def test_staged_bundle_parent_fsync_failure_rolls_back_published_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "standalone-historical").resolve()
    original_fsync = runner._fsync_directory

    def fail_after_rename(path: Path) -> None:
        if path == output.parent and output.exists():
            raise OSError("injected post-rename parent fsync")
        original_fsync(path)

    monkeypatch.setattr(runner, "_fsync_directory", fail_after_rename)
    with pytest.raises(OSError, match="post-rename parent fsync"):
        runner._write_sealed_bundle(
            output,
            {"result.json": b'{"complete":true}\n'},
            seal_bytes=b'{"artifact_kind":"test"}\n',
        )

    assert not output.exists()
    assert not tuple(
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(".standalone-historical.staging-")
    )
