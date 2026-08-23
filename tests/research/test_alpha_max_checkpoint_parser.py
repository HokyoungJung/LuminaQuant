from __future__ import annotations

import copy
import gc
import hashlib
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from lumina_quant.backtesting.execution_model import ExecutionModel, ExecutionModelConfig
from lumina_quant.backtesting.portfolio_backtest import FillApplicationAttribution
from lumina_quant.research import alpha_max_evidence as evidence
from lumina_quant.research import alpha_max_engine_runner as runner

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/research/alpha_max_portfolio_20260711_listing_aware.json"
).resolve()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _indicator_day_descriptor(root: Path) -> dict[str, object]:
    parent = root.parent.stat()
    receipt = {
        "byte_count": 1,
        "path": str((root.parent / "identity.json").resolve()),
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
        "config": receipt,
        "contract_manifest": {"byte_count": 1, "sha256": "a" * 64},
        "manifest": receipt,
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
    state.chmod(0o600)
    state.chmod(0o400)

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


def _v2_cell_descriptor(
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
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v2",
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
        "phase_windows": {},
        "physical_fold_run_count": len(schedule),
        "physical_schedule": schedule,
        "physical_schedule_sha256": _sha256(runner._canonical_bytes(schedule)),
        "python": {},
        "root_seals": [],
        "runtime_contract_sha256": "a" * 64,
        "thread_contract": {},
        "universe": {},
    }
    if role == "historical":
        descriptor["prelock_binding"] = {
            "immutable_prelock_seal_sha256": "b" * 64,
            "validated_snapshot_sha256": "c" * 64,
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
    descriptor = {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v1",
        "attempt_role": "prelock",
        "implementation_inventory": runner._alpha_max_checkpoint_implementation_inventory(),
        "checkpoint": {
            "parent": str(checkpoint.parent),
            "parent_identity": [
                checkpoint.parent.stat().st_dev,
                checkpoint.parent.stat().st_ino,
            ],
            "root": str(checkpoint),
        },
        "output": {
            "parent": str(output.parent),
            "parent_identity": [
                output.parent.stat().st_dev,
                output.parent.stat().st_ino,
            ],
            "target": str(output),
        },
    }
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
    assert seal["artifact_kind"] == "alpha_max_restartable_cost_cell_seal.v1"
    assert set(seal) == {
        "artifact_kind",
        "attempt_descriptor_sha256",
        "byte_count",
        "capsule_receipt_sha256s",
        "cell_name",
        "config_sha256",
        "domain",
        "evidence_sha256",
        "manifest_sha256",
        "nominal_cost_bps",
        "root_receipt_identities",
        "row_id",
        "runtime_contract_sha256",
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
    descriptor = _v2_cell_descriptor(
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
    descriptor = _v2_cell_descriptor(
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


def test_prelock_v1_descriptor_rejects_historical_v2_fields(tmp_path: Path) -> None:
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
    assert runner._alpha_max_validate_checkpoint_descriptor(descriptor) == (
        "prelock",
        "validation",
    )
    descriptor["domain"] = "historical_exposed_evaluation"
    with pytest.raises(runner.AlphaMaxRuntimeContractError):
        runner._alpha_max_validate_checkpoint_descriptor(descriptor)


def test_checkpoint_store_rejects_duplicate_key_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tmp_path, monkeypatch)
    output = (tmp_path / "official-output").resolve()
    checkpoint = (tmp_path / "checkpoint").resolve()
    descriptor = {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v1",
        "attempt_role": "prelock",
        "implementation_inventory": runner._alpha_max_checkpoint_implementation_inventory(),
        "checkpoint": {
            "parent": str(checkpoint.parent),
            "parent_identity": [
                checkpoint.parent.stat().st_dev,
                checkpoint.parent.stat().st_ino,
            ],
            "root": str(checkpoint),
        },
        "output": {
            "parent": str(output.parent),
            "parent_identity": [
                output.parent.stat().st_dev,
                output.parent.stat().st_ino,
            ],
            "target": str(output),
        },
    }
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


def test_checkpoint_store_rejects_replaced_descriptor_parent(
    tmp_path: Path,
) -> None:
    checkpoint_parent = (tmp_path / "checkpoint-parent").resolve()
    output_parent = (tmp_path / "output-parent").resolve()
    checkpoint_parent.mkdir()
    output_parent.mkdir()
    checkpoint = checkpoint_parent / "checkpoint"
    output = output_parent / "output"
    descriptor = {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v1",
        "attempt_role": "prelock",
        "implementation_inventory": runner._alpha_max_checkpoint_implementation_inventory(),
        "checkpoint": {
            "parent": str(checkpoint_parent),
            "parent_identity": [
                checkpoint_parent.stat().st_dev,
                checkpoint_parent.stat().st_ino,
            ],
            "root": str(checkpoint),
        },
        "output": {
            "parent": str(output_parent),
            "parent_identity": [
                output_parent.stat().st_dev,
                output_parent.stat().st_ino,
            ],
            "target": str(output),
        },
    }
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
    descriptor = {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v1",
        "attempt_role": "prelock",
        "implementation_inventory": runner._alpha_max_checkpoint_implementation_inventory(),
        "checkpoint": {
            "parent": str(checkpoint_parent),
            "parent_identity": [
                checkpoint_parent.stat().st_dev,
                checkpoint_parent.stat().st_ino,
            ],
            "root": str(checkpoint),
        },
        "output": {
            "parent": str(output_parent),
            "parent_identity": [
                output_parent.stat().st_dev,
                output_parent.stat().st_ino,
            ],
            "target": str(output),
        },
    }
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
    descriptor = {
        "artifact_kind": "alpha_max_restartable_attempt_descriptor.v1",
        "attempt_role": "prelock",
        "implementation_inventory": runner._alpha_max_checkpoint_implementation_inventory(),
        "checkpoint": {
            "parent": str(checkpoint.parent),
            "parent_identity": [
                checkpoint.parent.stat().st_dev,
                checkpoint.parent.stat().st_ino,
            ],
            "root": str(checkpoint),
        },
        "output": {
            "parent": str(output.parent),
            "parent_identity": [
                output.parent.stat().st_dev,
                output.parent.stat().st_ino,
            ],
            "target": str(output),
        },
    }
    store = runner._AlphaMaxCellCheckpointStore(
        checkpoint,
        output_root=output,
        descriptor=descriptor,
        config_bytes=context.preflight.config_bytes,
    )
    cell_path = checkpoint / "cells" / store._cell_name(context.cell.row_id, 20)
    original_fsync = runner._fsync_directory

    def crash_after_rename(path: Path) -> None:
        if Path(path).name == "cells" and cell_path.exists():
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
