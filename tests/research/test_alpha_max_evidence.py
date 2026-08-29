from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import lumina_quant.research.alpha_max_evidence as evidence
from lumina_quant.core.engine import TradingEngine
from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_CANDIDATE_SYMBOLS,
    AlphaMaxAdmissionDailyCandidateInput,
    AlphaMaxCapsuleReceipt,
    AlphaMaxCostCellEvidence,
    AlphaMaxDailyQuoteNotional,
    AlphaMaxEquityEndpoint,
    AlphaMaxGateInput,
    AlphaMaxFundingBoundaryResolver,
    AlphaMaxManifestMaterialization,
    AlphaMaxManifestReceipt,
    AlphaMaxTerminalGateEvidence,
    AlphaMaxOrderedFundingLookup,
    AlphaMaxRootReceipt,
    AlphaMaxRowEvidence,
    AlphaMaxStreamingEquityTracker,
    FeatureRootSpec,
    alpha_max_common_rng_seed,
    alpha_max_full_event_mdd,
    alpha_max_terminal_outcome,
    build_alpha_max_prelock_seal,
    build_alpha_max_normalized_fold_segment_evidence,
    build_alpha_max_primary_return_stream,
    build_alpha_max_terminal_state,
    build_alpha_max_train_liquidity_buckets,
    build_alpha_max_trend_liquidity_falsifier,
    canonical_alpha_max_cost_cell_bytes,
    canonical_alpha_max_row_bytes,
    compute_alpha_max_metric_statistics,
    compute_alpha_max_train_admission_from_daily_summaries,
    rank_alpha_max_historical_report,
    reconcile_alpha_max_cost_attribution,
    seal_alpha_max_contract_manifest,
    seal_alpha_max_root_tree,
    select_alpha_max_prelock_champion,
    validate_alpha_max_admitted_symbols,
    validate_alpha_max_train_liquidity_buckets,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_AVAILABILITY_FLOOR = datetime(2022, 12, 31, tzinfo=UTC)
_AVAILABILITY_CEILING = datetime(2026, 7, 1, tzinfo=UTC)
_TONUSDT_RAW_AVAILABILITY_START = datetime(2024, 3, 1, 12, 31, 10, tzinfo=UTC)
_TONUSDT_FEATURE_AVAILABILITY_START = datetime(2024, 3, 1, 16, tzinfo=UTC)
_TONUSDT_AVAILABILITY_END = datetime(2026, 6, 23, 9, tzinfo=UTC)


class _FeatureStrategy:
    required_inputs = ()
    required_features = ("funding_rate",)


def _feature_spec(tmp_path: Path, root_id: str) -> FeatureRootSpec:
    path = tmp_path / root_id
    path.mkdir()
    start, end = evidence._ROOT_INTERVALS[root_id]
    return FeatureRootSpec(root_id, str(path), "binance", start, end, _HASH_A, _HASH_B)


def _write_sparse_raw_root(root: Path, root_id: str) -> None:
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
    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        directory = root / "market_ohlcv_1s" / "binance" / symbol
        directory.mkdir(parents=True, exist_ok=True)
        for partition_start in months:
            partition_end = (
                partition_start.replace(year=partition_start.year + 1, month=1)
                if partition_start.month == 12
                else partition_start.replace(month=partition_start.month + 1)
            )
            owned_start = max(start, partition_start)
            owned_end = min(end, partition_end)
            pl = pytest.importorskip("polars")
            timestamps = pl.datetime_range(
                owned_start,
                owned_end,
                interval="1s",
                closed="left",
                time_zone="UTC",
                eager=True,
            )
            pl.DataFrame({"datetime": timestamps}).with_columns(
                pl.lit(100.0).alias("open"),
                pl.lit(101.0).alias("high"),
                pl.lit(99.0).alias("low"),
                pl.lit(100.0).alias("close"),
                pl.lit(1.0).alias("volume"),
            ).write_parquet(directory / f"{partition_start:%Y-%m}.parquet")


def _write_feature_root(root: Path, root_id: str) -> None:
    pl = pytest.importorskip("polars")
    start, end = evidence._ROOT_INTERVALS[root_id]
    day = start
    while day < end:
        for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
            directory = (
                root
                / "feature_points"
                / "exchange=binance"
                / f"symbol={symbol}"
                / f"date={day:%Y-%m-%d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            hours = (0, 4, 8, 12, 16, 20) if symbol == "TONUSDT" else (0, 8, 16)
            timestamps = [int((day + timedelta(hours=hour)).timestamp() * 1000) for hour in hours]
            pl.DataFrame(
                {
                    "timestamp_ms": timestamps,
                    "source_timestamp_ms": [
                        value + (index % 2) * 1000 for index, value in enumerate(timestamps)
                    ],
                    "funding_rate": [0.0001 * (index + 1) for index in range(len(timestamps))],
                    "symbol": [symbol] * len(timestamps),
                    "exchange": ["binance"] * len(timestamps),
                }
            ).write_parquet(directory / "part-0.parquet")
        day += timedelta(days=1)


def _ton_listing_availability_start_contract(root_kind: str) -> MappingProxyType:
    if root_kind not in {"raw", "feature"}:
        raise ValueError("root_kind_invalid")
    ton_start = (
        _TONUSDT_RAW_AVAILABILITY_START
        if root_kind == "raw"
        else _TONUSDT_FEATURE_AVAILABILITY_START
    )
    return MappingProxyType(
        {
            symbol: ton_start if symbol == "TONUSDT" else _AVAILABILITY_FLOOR
            for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS
        }
    )


def _ton_listing_availability_end_contract() -> MappingProxyType:
    return MappingProxyType(
        {
            symbol: (_TONUSDT_AVAILABILITY_END if symbol == "TONUSDT" else _AVAILABILITY_CEILING)
            for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS
        }
    )


def _seal_test_root_tree(root_id: str, root_kind: str, root_path: Path, **kwargs):
    """Seal a synthetic fixture under an explicit immutable availability interval."""
    kwargs.setdefault(
        "availability_start_by_symbol",
        MappingProxyType(dict.fromkeys(ALPHA_MAX_CANDIDATE_SYMBOLS, _AVAILABILITY_FLOOR)),
    )
    kwargs.setdefault(
        "availability_end_by_symbol",
        MappingProxyType(dict.fromkeys(ALPHA_MAX_CANDIDATE_SYMBOLS, _AVAILABILITY_CEILING)),
    )
    return seal_alpha_max_root_tree(root_id, root_kind, root_path, **kwargs)


def _use_short_listing_interval(
    monkeypatch: pytest.MonkeyPatch,
    root_id: str,
    root_kind: str,
) -> None:
    intervals = {
        "warmup": (
            datetime(2023, 12, 29, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
        "train": (
            (
                datetime(2024, 3, 1, 12, 31, 10, tzinfo=UTC)
                if root_kind == "raw"
                else datetime(2024, 2, 28, tzinfo=UTC)
            ),
            (
                datetime(2024, 3, 1, 12, 31, 20, tzinfo=UTC)
                if root_kind == "raw"
                else datetime(2024, 3, 4, tzinfo=UTC)
            ),
        ),
    }
    monkeypatch.setitem(
        evidence._ROOT_INTERVALS,
        root_id,
        intervals[root_id],
    )


def _use_short_raw_interval(monkeypatch: pytest.MonkeyPatch, root_id: str) -> None:
    start = evidence._ROOT_INTERVALS[root_id][0]
    monkeypatch.setitem(
        evidence._ROOT_INTERVALS,
        root_id,
        (start, start + timedelta(seconds=20)),
    )


def _use_short_delivery_interval(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root_kind: str,
    start: datetime = datetime(2026, 6, 22, tzinfo=UTC),
    end: datetime = datetime(2026, 6, 25, tzinfo=UTC),
) -> str:
    root_id = "historical_exposed_evaluation"
    if root_kind == "raw":
        if start >= _TONUSDT_AVAILABILITY_END:
            end = start + timedelta(seconds=10)
        else:
            start = _TONUSDT_AVAILABILITY_END - timedelta(seconds=10)
            end = _TONUSDT_AVAILABILITY_END + timedelta(seconds=10)
    monkeypatch.setitem(evidence._ROOT_INTERVALS, root_id, (start, end))
    return root_id


def _write_listing_aware_raw_root(
    root: Path,
    availability_start_by_symbol: MappingProxyType,
    availability_end_by_symbol: MappingProxyType,
    *,
    root_id: str = "train",
) -> None:
    pl = pytest.importorskip("polars")
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
    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        directory = root / "market_ohlcv_1s" / "binance" / symbol
        directory.mkdir(parents=True, exist_ok=True)
        available = availability_start_by_symbol[symbol]
        for partition_start in months:
            partition_end = (
                partition_start.replace(year=partition_start.year + 1, month=1)
                if partition_start.month == 12
                else partition_start.replace(month=partition_start.month + 1)
            )
            owned_start = max(start, available, partition_start)
            owned_end = min(end, availability_end_by_symbol[symbol], partition_end)
            if owned_start >= owned_end:
                continue
            timestamps = pl.datetime_range(
                owned_start,
                owned_end,
                interval="1s",
                closed="left",
                time_zone="UTC",
                eager=True,
            )
            pl.DataFrame({"datetime": timestamps}).with_columns(
                pl.lit(100.0).alias("open"),
                pl.lit(101.0).alias("high"),
                pl.lit(99.0).alias("low"),
                pl.lit(100.0).alias("close"),
                pl.lit(1.0).alias("volume"),
            ).write_parquet(directory / f"{partition_start:%Y-%m}.parquet")


def _write_listing_aware_feature_root(
    root: Path,
    availability_start_by_symbol: MappingProxyType,
    availability_end_by_symbol: MappingProxyType,
    *,
    root_id: str = "train",
) -> None:
    pl = pytest.importorskip("polars")
    start, end = evidence._ROOT_INTERVALS[root_id]
    day = start
    while day < end:
        day_end = day + timedelta(days=1)
        for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
            owned_start = max(start, availability_start_by_symbol[symbol], day)
            owned_end = min(end, availability_end_by_symbol[symbol], day_end)
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
            hours = (0, 4, 8, 12, 16, 20) if symbol == "TONUSDT" else (0, 8, 16)
            timestamps = [
                int(boundary.timestamp() * 1000)
                for hour in hours
                for boundary in (day + timedelta(hours=hour),)
                if owned_start <= boundary < owned_end
            ]
            if not timestamps:
                raise AssertionError("listing-aware feature fixture has no owned funding boundary")
            pl.DataFrame(
                {
                    "timestamp_ms": timestamps,
                    "source_timestamp_ms": [value + 5 for value in timestamps],
                    "funding_rate": [0.0001 * (index + 1) for index in range(len(timestamps))],
                    "symbol": [symbol] * len(timestamps),
                    "exchange": ["binance"] * len(timestamps),
                }
            ).write_parquet(directory / "part-0.parquet")
        day += timedelta(days=1)


def _write_ton_prelisting_partition(root: Path, root_kind: str, *, synthetic: bool) -> None:
    pl = pytest.importorskip("polars")
    day = datetime(2024, 2, 29, tzinfo=UTC)
    if root_kind == "raw":
        target = root / "market_ohlcv_1s" / "binance" / "TONUSDT" / "2024-02.parquet"
        pl.DataFrame(
            {
                "datetime": [day + timedelta(seconds=7), day + timedelta(hours=23)],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.0, 101.0],
                "volume": [1.0, 2.0],
            }
        ).write_parquet(target)
        return
    directory = root / "feature_points" / "exchange=binance" / "symbol=TONUSDT" / "date=2024-02-29"
    directory.mkdir(parents=True, exist_ok=True)
    timestamps = [int((day + timedelta(hours=hour)).timestamp() * 1000) for hour in (0, 8, 16)]
    filename = "synthetic.parquet" if synthetic else "part-0.parquet"
    pl.DataFrame(
        {
            "timestamp_ms": timestamps,
            "source_timestamp_ms": timestamps,
            "funding_rate": [0.0001, -0.0002, 0.0003],
            "symbol": ["TONUSDT"] * 3,
            "exchange": ["binance"] * 3,
        }
    ).write_parquet(directory / filename)


def test_b2_ordered_lookup_exposes_immutable_current_root_capability_to_real_engine_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Lookup:
        def __init__(self, *, db_path, exchange, start_date, end_date):
            self.db_path = db_path
            self.exchange = exchange

    monkeypatch.setattr(evidence, "FeaturePointLookup", _Lookup)
    warmup = _feature_spec(tmp_path, "warmup")
    train = _feature_spec(tmp_path, "train")
    lookup = AlphaMaxOrderedFundingLookup((warmup, train))

    assert lookup.db_path == train.path
    engine = object.__new__(TradingEngine)
    engine.strategy = _FeatureStrategy()
    TradingEngine._assert_strategy_requirements(
        engine,
        available_inputs=set(),
        feature_lookup=lookup,
    )
    with pytest.raises(AttributeError, match="immutable"):
        lookup.db_path = warmup.path


def test_prior_trial_inventory_reads_the_frozen_git_blob_not_mutable_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "alpha-max@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Alpha Max Test"],
        check=True,
    )
    relative = Path("frozen/prior.json")
    path = repo / relative
    path.parent.mkdir(parents=True)
    frozen = b'{"frozen":true}\n'
    path.write_bytes(frozen)
    subprocess.run(["git", "-C", str(repo), "add", str(relative)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "freeze prior"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    blob_oid = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"HEAD:{relative.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(evidence, "_ALPHA_MAX_PRIOR_COMMIT", commit)
    monkeypatch.setattr(evidence, "_ALPHA_MAX_PRIOR_PATH", relative.as_posix())
    monkeypatch.setattr(evidence, "_ALPHA_MAX_PRIOR_BLOB_OID", blob_oid)
    monkeypatch.setattr(
        evidence,
        "_ALPHA_MAX_PRIOR_FILE_SHA256",
        hashlib.sha256(frozen).hexdigest(),
    )

    path.write_bytes(b'{"poisoned_worktree":true}\n')
    assert evidence.read_alpha_max_prior_trial_blob(repo) == frozen

    monkeypatch.setattr(evidence, "_ALPHA_MAX_PRIOR_BLOB_OID", "0" * 40)
    with pytest.raises(ValueError, match="prior_trial_inventory_mismatch"):
        evidence.read_alpha_max_prior_trial_blob(repo)


def test_prior_trial_runtime_input_requires_exact_read_only_blob(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "var/reports/ultragoal_full_pool_strategy/g004_frozen_candidate_manifest.json"
    )
    frozen = evidence.read_alpha_max_prior_trial_blob(Path(__file__).resolve().parents[2])
    runtime_input = (tmp_path / "prior-trials.json").resolve()
    runtime_input.write_bytes(frozen)
    runtime_input.chmod(0o400)

    assert evidence.read_alpha_max_prior_trial_blob_input(runtime_input) == frozen

    runtime_input.chmod(0o600)
    with pytest.raises(ValueError, match="prior_trial_inventory_mismatch"):
        evidence.read_alpha_max_prior_trial_blob_input(runtime_input)
    runtime_input.chmod(0o400)
    alias = tmp_path / "prior-trials-alias.json"
    alias.symlink_to(runtime_input)
    with pytest.raises(ValueError, match="prior_trial_inventory_mismatch"):
        evidence.read_alpha_max_prior_trial_blob_input(alias)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == hashlib.sha256(frozen).hexdigest()


def test_manifest_write_is_bound_to_opened_phase_when_ancestor_is_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "owned").resolve()
    manifests = root / "manifests"
    validation = manifests / "validation_train_fit"
    prelock = manifests / "prelock_final_refit"
    validation.mkdir(parents=True)
    prelock.mkdir()
    validated = evidence._validate_run_owned_phase(root, "validation_train_fit")
    moved = manifests / "validation-opened"
    external = (tmp_path / "external").resolve()
    external.mkdir()
    original_open = os.open
    swapped = False

    def hostile_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "row.json" and dir_fd is not None and not swapped:
            swapped = True
            validation.rename(moved)
            validation.symlink_to(external, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", hostile_open)
    evidence._write_new_manifest(validated / "row.json", b'{"owned":true}\n')

    assert swapped is True
    assert not (external / "row.json").exists()
    assert (moved / "row.json").read_bytes() == b'{"owned":true}\n'


def test_root_tree_seal_requires_explicit_availability_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")
    availability_start = MappingProxyType(
        dict.fromkeys(ALPHA_MAX_CANDIDATE_SYMBOLS, _AVAILABILITY_FLOOR)
    )
    availability_end = MappingProxyType(
        dict.fromkeys(ALPHA_MAX_CANDIDATE_SYMBOLS, _AVAILABILITY_CEILING)
    )

    with pytest.raises(TypeError, match="availability_start_by_symbol"):
        seal_alpha_max_root_tree("purge", "raw", root)
    with pytest.raises(TypeError, match="must_supply_start_and_end"):
        seal_alpha_max_root_tree(
            "purge",
            "raw",
            root,
            availability_start_by_symbol=None,
            availability_end_by_symbol=availability_end,
        )
    with pytest.raises(TypeError, match="must_supply_start_and_end"):
        seal_alpha_max_root_tree(
            "purge",
            "raw",
            root,
            availability_start_by_symbol=availability_start,
            availability_end_by_symbol=None,
        )


def test_root_tree_seal_is_canonical_streaming_and_rejects_unsafe_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")

    first = _seal_test_root_tree("purge", "raw", root)
    second = _seal_test_root_tree("purge", "raw", root)
    assert first == second
    assert first.symbols == ALPHA_MAX_CANDIDATE_SYMBOLS
    assert len(first.entries) == len(ALPHA_MAX_CANDIDATE_SYMBOLS)
    start, end = evidence._ROOT_INTERVALS["purge"]
    assert min(entry.minimum_timestamp_ms for entry in first.entries) == int(
        start.timestamp() * 1000
    )
    assert max(entry.maximum_timestamp_ms for entry in first.entries) == int(
        (end - timedelta(seconds=1)).timestamp() * 1000
    )
    assert first.inventory_sha256 == second.inventory_sha256
    assert first.content_sha256 == second.content_sha256
    assert first.canonical_bytes == second.canonical_bytes
    assert first.to_receipt().content_sha256 == first.content_sha256

    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    frame.with_columns((pl.col("close") + 1.0).alias("close")).write_parquet(target)
    mutated = _seal_test_root_tree("purge", "raw", root)
    assert mutated.content_sha256 != first.content_sha256
    assert mutated.inventory_sha256 != first.inventory_sha256

    (root / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        _seal_test_root_tree("purge", "raw", root)
    with pytest.raises(ValueError, match="must_be_absolute"):
        _seal_test_root_tree("train", "raw", Path("relative"))


@pytest.mark.parametrize("root_kind", ("raw", "feature"))
def test_train_root_seal_allows_only_ton_prelisting_gap_from_immutable_availability_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    _use_short_listing_interval(monkeypatch, "train", root_kind)
    availability_start = _ton_listing_availability_start_contract(root_kind)
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / root_kind
    if root_kind == "raw":
        _write_listing_aware_raw_root(root, availability_start, availability_end)
    else:
        _write_listing_aware_feature_root(root, availability_start, availability_end)

    seal = _seal_test_root_tree(
        "train",
        root_kind,
        root,
        availability_start_by_symbol=availability_start,
        availability_end_by_symbol=availability_end,
    )

    ton_partitions = tuple(
        partition_start
        for entry in seal.entries
        for symbol, partition_start, _ in (
            evidence._alpha_max_partition_contract(
                entry.relative_path,
                root_kind=root_kind,
                exchange="binance",
            ),
        )
        if symbol == "TONUSDT"
    )
    assert ton_partitions
    assert min(ton_partitions) == datetime(2024, 3, 1, tzinfo=UTC)
    assert all(
        entry.minimum_timestamp_ms >= int(availability_start["TONUSDT"].timestamp() * 1000)
        for entry in seal.entries
        if "TONUSDT" in entry.relative_path
    )
    assert seal.to_payload()["availability_start_by_symbol"] == {
        symbol: start.isoformat().replace("+00:00", "Z")
        for symbol, start in availability_start.items()
    }
    assert seal.to_payload()["availability_end_by_symbol"] == {
        symbol: end.isoformat().replace("+00:00", "Z") for symbol, end in availability_end.items()
    }
    assert seal.to_receipt().availability_sha256 == seal.availability_sha256


def test_train_root_seal_rejects_mutable_symbol_availability_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_listing_interval(monkeypatch, "train", "raw")
    availability_start = _ton_listing_availability_start_contract("raw")
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / "raw"
    _write_listing_aware_raw_root(root, availability_start, availability_end)

    with pytest.raises(TypeError, match=r"availability.*immutable"):
        _seal_test_root_tree(
            "train",
            "raw",
            root,
            availability_start_by_symbol=dict(availability_start),
            availability_end_by_symbol=availability_end,
        )
    with pytest.raises(TypeError, match=r"availability.*immutable"):
        _seal_test_root_tree(
            "train",
            "raw",
            root,
            availability_start_by_symbol=availability_start,
            availability_end_by_symbol=dict(availability_end),
        )


@pytest.mark.parametrize("root_kind", ("raw", "feature"))
def test_train_root_seal_rejects_ton_partition_before_canonical_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    _use_short_listing_interval(monkeypatch, "train", root_kind)
    availability_start = _ton_listing_availability_start_contract(root_kind)
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / root_kind
    if root_kind == "raw":
        _write_listing_aware_raw_root(root, availability_start, availability_end)
    else:
        _write_listing_aware_feature_root(root, availability_start, availability_end)
    _write_ton_prelisting_partition(root, root_kind, synthetic=False)

    with pytest.raises(ValueError, match="availability"):
        _seal_test_root_tree(
            "train",
            root_kind,
            root,
            availability_start_by_symbol=availability_start,
            availability_end_by_symbol=availability_end,
        )


@pytest.mark.parametrize("root_kind", ("raw", "feature"))
def test_train_root_seal_rejects_missing_ton_partition_after_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    _use_short_listing_interval(monkeypatch, "train", root_kind)
    availability_start = _ton_listing_availability_start_contract(root_kind)
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / root_kind
    if root_kind == "raw":
        _write_listing_aware_raw_root(root, availability_start, availability_end)
        target = root / "market_ohlcv_1s" / "binance" / "TONUSDT" / "2024-03.parquet"
    else:
        _write_listing_aware_feature_root(root, availability_start, availability_end)
        target = (
            root
            / "feature_points"
            / "exchange=binance"
            / "symbol=TONUSDT"
            / "date=2024-03-02"
            / "part-0.parquet"
        )
    target.unlink()

    with pytest.raises(ValueError, match=r"symbol_coverage|interval_coverage"):
        _seal_test_root_tree(
            "train",
            root_kind,
            root,
            availability_start_by_symbol=availability_start,
            availability_end_by_symbol=availability_end,
        )


def test_train_feature_root_rejects_missing_ton_funding_cadence_after_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_listing_interval(monkeypatch, "train", "feature")
    availability_start = _ton_listing_availability_start_contract("feature")
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / "feature"
    _write_listing_aware_feature_root(root, availability_start, availability_end)
    target = (
        root
        / "feature_points"
        / "exchange=binance"
        / "symbol=TONUSDT"
        / "date=2024-03-01"
        / "part-0.parquet"
    )
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    frame.filter(pl.col("timestamp_ms") != frame["timestamp_ms"][1]).write_parquet(target)

    with pytest.raises(
        ValueError,
        match=r"funding_boundary_missing|timestamp_cadence|funding_(canonical_)?coverage",
    ):
        _seal_test_root_tree(
            "train",
            "feature",
            root,
            availability_start_by_symbol=availability_start,
            availability_end_by_symbol=availability_end,
        )


def test_train_feature_root_rejects_synthetic_ton_file_before_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_listing_interval(monkeypatch, "train", "feature")
    availability_start = _ton_listing_availability_start_contract("feature")
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / "feature"
    _write_listing_aware_feature_root(root, availability_start, availability_end)
    _write_ton_prelisting_partition(root, "feature", synthetic=True)

    with pytest.raises(ValueError, match="availability"):
        _seal_test_root_tree(
            "train",
            "feature",
            root,
            availability_start_by_symbol=availability_start,
            availability_end_by_symbol=availability_end,
        )


@pytest.mark.parametrize("root_kind", ("raw", "feature"))
def test_root_seal_allows_zero_ton_partitions_when_phase_is_after_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    root_id = _use_short_delivery_interval(
        monkeypatch,
        root_kind=root_kind,
        start=datetime(2026, 6, 24, tzinfo=UTC),
        end=datetime(2026, 6, 25, tzinfo=UTC),
    )
    availability_start = _ton_listing_availability_start_contract(root_kind)
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / root_kind
    if root_kind == "raw":
        _write_listing_aware_raw_root(
            root,
            availability_start,
            availability_end,
            root_id=root_id,
        )
    else:
        _write_listing_aware_feature_root(
            root,
            availability_start,
            availability_end,
            root_id=root_id,
        )

    seal = _seal_test_root_tree(
        root_id,
        root_kind,
        root,
        availability_start_by_symbol=availability_start,
        availability_end_by_symbol=availability_end,
    )

    assert seal.entries
    assert not any("TONUSDT" in entry.relative_path for entry in seal.entries)
    assert seal.to_payload()["availability_end_by_symbol"]["TONUSDT"] == ("2026-06-23T09:00:00Z")


@pytest.mark.parametrize("root_kind", ("raw", "feature"))
def test_root_seal_rejects_ton_content_or_partition_at_or_after_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    root_id = _use_short_delivery_interval(monkeypatch, root_kind=root_kind)
    availability_start = _ton_listing_availability_start_contract(root_kind)
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / root_kind
    pl = pytest.importorskip("polars")
    if root_kind == "raw":
        _write_listing_aware_raw_root(
            root,
            availability_start,
            availability_end,
            root_id=root_id,
        )
        target = root / "market_ohlcv_1s" / "binance" / "TONUSDT" / "2026-06.parquet"
        frame = pl.read_parquet(target)
        pl.concat(
            [
                frame,
                pl.DataFrame(
                    {
                        "datetime": [_TONUSDT_AVAILABILITY_END],
                        "open": [102.0],
                        "high": [103.0],
                        "low": [101.0],
                        "close": [102.0],
                        "volume": [3.0],
                    }
                ),
            ]
        ).write_parquet(target)
    else:
        _write_listing_aware_feature_root(
            root,
            availability_start,
            availability_end,
            root_id=root_id,
        )
        day = datetime(2026, 6, 24, tzinfo=UTC)
        directory = (
            root / "feature_points" / "exchange=binance" / "symbol=TONUSDT" / "date=2026-06-24"
        )
        directory.mkdir(parents=True)
        pl.DataFrame(
            {
                "timestamp_ms": [int(day.timestamp() * 1000)],
                "source_timestamp_ms": [int(day.timestamp() * 1000)],
                "funding_rate": [0.0001],
                "symbol": ["TONUSDT"],
                "exchange": ["binance"],
            }
        ).write_parquet(directory / "part-0.parquet")

    with pytest.raises(ValueError, match="after_availability"):
        _seal_test_root_tree(
            root_id,
            root_kind,
            root,
            availability_start_by_symbol=availability_start,
            availability_end_by_symbol=availability_end,
        )


def test_feature_root_requires_right_edge_funding_cadence_before_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_id = _use_short_delivery_interval(monkeypatch, root_kind="feature")
    availability_start = _ton_listing_availability_start_contract("feature")
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / "feature"
    _write_listing_aware_feature_root(
        root,
        availability_start,
        availability_end,
        root_id=root_id,
    )
    target = (
        root
        / "feature_points"
        / "exchange=binance"
        / "symbol=TONUSDT"
        / "date=2026-06-23"
        / "part-0.parquet"
    )
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    frame.filter(pl.col("timestamp_ms") != frame.get_column("timestamp_ms").max()).write_parquet(
        target
    )

    with pytest.raises(ValueError, match=r"funding_(canonical_)?coverage"):
        _seal_test_root_tree(
            root_id,
            "feature",
            root,
            availability_start_by_symbol=availability_start,
            availability_end_by_symbol=availability_end,
        )


def test_root_seal_combined_availability_hash_rejects_end_map_or_digest_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_id = _use_short_delivery_interval(monkeypatch, root_kind="raw")
    availability_start = _ton_listing_availability_start_contract("raw")
    availability_end = _ton_listing_availability_end_contract()
    root = tmp_path / "raw"
    _write_listing_aware_raw_root(
        root,
        availability_start,
        availability_end,
        root_id=root_id,
    )
    seal = _seal_test_root_tree(
        root_id,
        "raw",
        root,
        availability_start_by_symbol=availability_start,
        availability_end_by_symbol=availability_end,
    )

    with pytest.raises(ValueError, match="availability_sha256_mismatch"):
        replace(seal, availability_sha256=_HASH_A)
    tampered_end = MappingProxyType(
        dict(availability_end) | {"TONUSDT": _TONUSDT_AVAILABILITY_END + timedelta(hours=1)}
    )
    with pytest.raises(ValueError, match="availability_sha256_mismatch"):
        replace(seal, availability_end_by_symbol=tampered_end)


def test_root_tree_seal_rejects_hardlinked_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")
    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    os.link(target, tmp_path / "outside-alias.parquet")

    with pytest.raises(ValueError, match="hardlink"):
        _seal_test_root_tree("purge", "raw", root)


def test_root_tree_seal_stays_on_opened_ancestor_when_path_is_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = (tmp_path / "raw").resolve()
    outside_root = (tmp_path / "outside-raw").resolve()
    _write_sparse_raw_root(root, "purge")
    _write_sparse_raw_root(outside_root, "purge")
    owned_binance = root / "market_ohlcv_1s" / "binance"
    opened_binance = tmp_path / "opened-binance"
    outside_binance = outside_root / "market_ohlcv_1s" / "binance"
    outside_file_identities = {
        (int(value.stat().st_dev), int(value.stat().st_ino))
        for value in outside_binance.rglob("*.parquet")
    }
    streamed_identities: list[tuple[int, int]] = []
    original_stream = evidence._alpha_max_stream_regular_descriptor
    original_open = os.open
    swapped = False
    opened_ancestor_identity: tuple[int, int] | None = None

    def capture_stream(descriptor: int, **kwargs):
        opened = os.fstat(descriptor)
        streamed_identities.append((int(opened.st_dev), int(opened.st_ino)))
        return original_stream(descriptor, **kwargs)

    def hostile_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal opened_ancestor_identity, swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if (
            path == "binance"
            and dir_fd is not None
            and flags & getattr(os, "O_DIRECTORY", 0)
            and not swapped
        ):
            swapped = True
            opened = os.fstat(descriptor)
            opened_ancestor_identity = (int(opened.st_dev), int(opened.st_ino))
            owned_binance.rename(opened_binance)
            owned_binance.symlink_to(outside_binance, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(evidence, "_alpha_max_stream_regular_descriptor", capture_stream)
    monkeypatch.setattr(os, "open", hostile_open)

    with pytest.raises(ValueError, match=r"directory_changed|path_changed"):
        _seal_test_root_tree("purge", "raw", root)

    assert swapped is True
    assert opened_ancestor_identity == (
        int(opened_binance.stat().st_dev),
        int(opened_binance.stat().st_ino),
    )
    assert opened_ancestor_identity != (
        int(outside_binance.stat().st_dev),
        int(outside_binance.stat().st_ino),
    )
    assert outside_file_identities.isdisjoint(streamed_identities)


def test_root_tree_seal_hash_and_parquet_metadata_share_one_open_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = (tmp_path / "raw").resolve()
    _write_sparse_raw_root(root, "purge")
    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    opened_target = tmp_path / "opened-target.parquet"
    replacement = tmp_path / "replacement.parquet"
    replacement.write_bytes(target.read_bytes())
    original_identity = (int(target.stat().st_dev), int(target.stat().st_ino))
    parsed_identity: tuple[int, int] | None = None
    original_bounds = evidence._alpha_max_parquet_timestamp_bounds
    swapped = False

    def hostile_bounds(descriptor: int, **kwargs):
        nonlocal parsed_identity, swapped
        opened = os.fstat(descriptor)
        identity = (int(opened.st_dev), int(opened.st_ino))
        if identity == original_identity and not swapped:
            swapped = True
            target.rename(opened_target)
            replacement.rename(target)
        parsed_identity = (
            int(os.fstat(descriptor).st_dev),
            int(os.fstat(descriptor).st_ino),
        )
        return original_bounds(descriptor, **kwargs)

    monkeypatch.setattr(evidence, "_alpha_max_parquet_timestamp_bounds", hostile_bounds)

    with pytest.raises(ValueError, match=r"file_changed|directory_changed"):
        _seal_test_root_tree("purge", "raw", root)

    assert swapped is True
    assert parsed_identity == original_identity
    assert (int(target.stat().st_dev), int(target.stat().st_ino)) != original_identity


def test_actual_run_domain_seals_bind_current_raw_and_adjacent_features(
    tmp_path: Path,
) -> None:
    def seal_stub(root_id: str, root_kind: str):
        path = (tmp_path / f"{root_id}-{root_kind}").resolve()
        path.mkdir()
        start, end = evidence._ROOT_INTERVALS[root_id]
        availability_start = _ton_listing_availability_start_contract(root_kind)
        availability_end = _ton_listing_availability_end_contract()
        value = object.__new__(evidence.AlphaMaxRootSeal)
        fields = {
            "root_id": root_id,
            "root_kind": root_kind,
            "path": str(path),
            "exchange": "binance",
            "symbols": ALPHA_MAX_CANDIDATE_SYMBOLS,
            "start_utc": start,
            "end_utc": end,
            "availability_start_by_symbol": availability_start,
            "availability_end_by_symbol": availability_end,
            "availability_sha256": evidence._alpha_max_availability_sha256(
                availability_start,
                availability_end,
            ),
            "entries": (object(),),
            "inventory_sha256": _HASH_A,
            "content_sha256": _HASH_B,
            "canonical_bytes": b"unused-in-domain-seal-test\n",
            "sha256": "c" * 64,
        }
        for name, field_value in fields.items():
            object.__setattr__(value, name, field_value)
        return value

    purge_raw = seal_stub("purge", "raw")
    validation_raw = seal_stub("validation", "raw")
    purge_feature = seal_stub("purge", "feature")
    validation_feature = seal_stub("validation", "feature")

    raw_receipts = evidence._alpha_max_validate_domain_root_seals(
        (validation_raw,),
        domain="validation",
        root_kind="raw",
    )
    feature_receipts = evidence._alpha_max_validate_domain_root_seals(
        (purge_feature, validation_feature),
        domain="validation",
        root_kind="feature",
    )

    assert tuple(receipt.root_id for receipt in raw_receipts) == ("validation",)
    assert tuple(receipt.root_id for receipt in feature_receipts) == (
        "purge",
        "validation",
    )
    with pytest.raises(ValueError, match="root_domain_mismatch"):
        evidence._alpha_max_validate_domain_root_seals(
            (purge_raw, validation_raw),
            domain="validation",
            root_kind="raw",
        )
    with pytest.raises(ValueError, match="root_domain_mismatch"):
        evidence._alpha_max_validate_domain_root_seals(
            (validation_feature,),
            domain="validation",
            root_kind="feature",
        )


@pytest.mark.parametrize(
    "timestamps,match",
    [
        (
            lambda start, end: [start + timedelta(seconds=7)] * 2,
            "duplicate_or_null",
        ),
        (
            lambda start, end: [end - timedelta(seconds=11), start + timedelta(seconds=7)],
            "not_strictly_increasing",
        ),
        (
            lambda start, end: [start - timedelta(seconds=1), end - timedelta(seconds=11)],
            "outside_interval|partition_content|before_availability",
        ),
    ],
)
def test_raw_root_rejects_duplicate_nonmonotone_and_out_of_range_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timestamps,
    match: str,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")
    start, end = evidence._ROOT_INTERVALS["purge"]
    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    pl = pytest.importorskip("polars")
    pl.DataFrame(
        {
            "datetime": timestamps(start, end),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1.0, 2.0],
        }
    ).write_parquet(target)
    with pytest.raises(ValueError, match=match):
        _seal_test_root_tree("purge", "raw", root)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_open", "ohlcv_schema_invalid"),
        ("string_close", "ohlcv_schema_invalid"),
        ("nan_open", "ohlcv_value_invalid"),
        ("infinite_close", "ohlcv_value_invalid"),
        ("zero_low", "ohlc_nonpositive"),
        ("negative_open", "ohlc_nonpositive"),
        ("negative_volume", "volume_negative"),
        ("high_below_open", "ohlcv_relation_invalid"),
        ("low_above_close", "ohlcv_relation_invalid"),
    ],
)
def test_raw_root_seal_rejects_invalid_ohlcv_values_and_relations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")
    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    if mutation == "missing_open":
        frame = frame.drop("open")
    elif mutation == "string_close":
        frame = frame.with_columns(pl.lit("100.0").alias("close"))
    elif mutation == "nan_open":
        frame = frame.with_columns(pl.lit(float("nan")).alias("open"))
    elif mutation == "infinite_close":
        frame = frame.with_columns(pl.lit(float("inf")).alias("close"))
    elif mutation == "zero_low":
        frame = frame.with_columns(pl.lit(0.0).alias("low"))
    elif mutation == "negative_open":
        frame = frame.with_columns(pl.lit(-1.0).alias("open"))
    elif mutation == "negative_volume":
        frame = frame.with_columns(pl.lit(-1.0).alias("volume"))
    elif mutation == "high_below_open":
        frame = frame.with_columns(pl.lit(99.5).alias("high"))
    elif mutation == "low_above_close":
        frame = frame.with_columns(pl.lit(100.5).alias("low"))
    else:  # pragma: no cover - exhaustive table guard
        raise AssertionError(f"unsupported OHLCV mutation: {mutation}")
    frame.write_parquet(target)

    with pytest.raises(ValueError, match=match):
        _seal_test_root_tree("purge", "raw", root)


def test_raw_root_seal_accepts_real_schema_without_symbol_or_exchange_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")
    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    pl = pytest.importorskip("polars")

    assert set(pl.read_parquet_schema(target)) == {
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    seal = _seal_test_root_tree("purge", "raw", root)

    assert seal.entries


@pytest.mark.parametrize("time_unit", ("us", "ns"))
def test_raw_root_seal_rejects_subsecond_source_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    time_unit: str,
) -> None:
    _use_short_raw_interval(monkeypatch, "purge")
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")
    target = root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2025-06.parquet"
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target).with_columns(
        pl.col("datetime").cast(pl.Datetime(time_unit, "UTC"))
    )
    offset = pl.duration(microseconds=500) if time_unit == "us" else pl.duration(nanoseconds=500)
    frame.with_columns((pl.col("datetime") + offset).alias("datetime")).write_parquet(target)

    with pytest.raises(ValueError, match="timestamp_subsecond_invalid"):
        _seal_test_root_tree("purge", "raw", root)


def test_raw_root_seal_enforces_exact_cross_month_partition_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2024, 2, 29, 23, 59, 58, tzinfo=UTC)
    end = datetime(2024, 3, 1, 0, 0, 2, tzinfo=UTC)
    monkeypatch.setitem(evidence._ROOT_INTERVALS, "purge", (start, end))
    root = tmp_path / "raw"
    _write_sparse_raw_root(root, "purge")

    seal = _seal_test_root_tree("purge", "raw", root)

    assert len(seal.entries) == 2 * len(ALPHA_MAX_CANDIDATE_SYMBOLS)
    btc_march = root / "market_ohlcv_1s/binance/BTCUSDT/2024-03.parquet"
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(btc_march)
    frame.filter(pl.col("datetime") != datetime(2024, 3, 1, tzinfo=UTC)).write_parquet(btc_march)
    with pytest.raises(ValueError, match="exact_1s_coverage"):
        _seal_test_root_tree("purge", "raw", root)


def test_feature_root_binds_content_ownership_and_every_funding_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _write_feature_root(root, "purge")
    seal = _seal_test_root_tree("purge", "feature", root)
    assert seal.symbols == ALPHA_MAX_CANDIDATE_SYMBOLS
    assert len(seal.entries) == 7 * len(ALPHA_MAX_CANDIDATE_SYMBOLS)

    target = (
        root
        / "feature_points"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "date=2025-06-01"
        / "part-0.parquet"
    )
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    frame.with_columns(pl.lit("ETHUSDT").alias("symbol")).write_parquet(target)
    with pytest.raises(ValueError, match="content_symbol_mismatch"):
        _seal_test_root_tree("purge", "feature", root)

    frame.filter(pl.col("timestamp_ms") != frame["timestamp_ms"][1]).write_parquet(target)
    with pytest.raises(
        ValueError,
        match=r"funding_boundary_missing|timestamp_cadence|funding_canonical_coverage",
    ):
        _seal_test_root_tree("purge", "feature", root)


def test_feature_root_accepts_verified_source_jitter_after_canonical_boundaries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _write_feature_root(root, "purge")
    target = (
        root
        / "feature_points"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "date=2025-06-01"
        / "part-0.parquet"
    )
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    timestamps = frame.get_column("timestamp_ms").to_list()
    frame.with_columns(
        pl.Series(
            "source_timestamp_ms",
            [timestamps[0], timestamps[1] + 1000, timestamps[2] + 999],
        )
    ).write_parquet(target)

    seal = _seal_test_root_tree("purge", "feature", root)

    entry = next(
        row for row in seal.entries if "symbol=BTCUSDT/date=2025-06-01" in row.relative_path
    )
    assert entry.maximum_gap_ms == evidence._FUNDING_INTERVAL_MS


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("jitter", "source_timestamp_jitter"),
        ("source_duplicate", "source_timestamp_duplicate"),
        ("canonical_collision", "timestamp_duplicate"),
    ),
)
def test_feature_root_rejects_unverified_jitter_and_timestamp_collisions(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = tmp_path / "features"
    _write_feature_root(root, "purge")
    target = root / "feature_points/exchange=binance/symbol=BTCUSDT/date=2025-06-01/part-0.parquet"
    pl = pytest.importorskip("polars")
    frame = pl.read_parquet(target)
    timestamps = frame.get_column("timestamp_ms").to_list()
    if mutation == "jitter":
        frame = frame.with_columns(
            pl.Series(
                "source_timestamp_ms",
                [timestamps[0], timestamps[1] + 1001, timestamps[2]],
            )
        )
    elif mutation == "source_duplicate":
        frame = frame.with_columns(
            pl.Series(
                "source_timestamp_ms",
                [timestamps[0], timestamps[0], timestamps[2]],
            )
        )
    else:
        frame = frame.with_columns(
            pl.Series(
                "timestamp_ms",
                [timestamps[0], timestamps[0], timestamps[2]],
            )
        )
    frame.write_parquet(target)

    with pytest.raises(ValueError, match=match):
        _seal_test_root_tree("purge", "feature", root)


def test_eight_hour_funding_resolver_forbids_ton_admission_fail_closed() -> None:
    ordered_lookup = object.__new__(AlphaMaxOrderedFundingLookup)
    admitted = ("ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "TONUSDT")

    with pytest.raises(ValueError, match="ton_4h_funding_forbidden_in_8h_resolver"):
        AlphaMaxFundingBoundaryResolver(ordered_lookup, admitted)


def _contract_manifest() -> dict[str, object]:
    return {
        "schema_version": "alpha_max_contract_manifest.v2",
        "exchange": "binance",
        "records": [
            {
                "symbol": symbol,
                "market_type": "perpetual",
                "linear": True,
                "inverse": False,
                "quote_asset": "USDT",
                "margin_asset": "USDT",
                "settle_asset": "USDT",
                "volume_unit": "base_asset",
                "contract_multiplier": 1.0,
                "raw_availability_start_utc": (
                    _TONUSDT_RAW_AVAILABILITY_START if symbol == "TONUSDT" else _AVAILABILITY_FLOOR
                )
                .isoformat()
                .replace("+00:00", "Z"),
                "raw_availability_end_utc": (
                    _TONUSDT_AVAILABILITY_END if symbol == "TONUSDT" else _AVAILABILITY_CEILING
                )
                .isoformat()
                .replace("+00:00", "Z"),
                "feature_availability_start_utc": (
                    _TONUSDT_FEATURE_AVAILABILITY_START
                    if symbol == "TONUSDT"
                    else _AVAILABILITY_FLOOR
                )
                .isoformat()
                .replace("+00:00", "Z"),
                "feature_availability_end_utc": (
                    _TONUSDT_AVAILABILITY_END if symbol == "TONUSDT" else _AVAILABILITY_CEILING
                )
                .isoformat()
                .replace("+00:00", "Z"),
            }
            for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS
        ],
    }


def test_contract_manifest_seal_accepts_only_exact_canonical_ten_symbol_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contracts.json"
    canonical = (
        json.dumps(
            _contract_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    path.write_bytes(canonical)
    seal = seal_alpha_max_contract_manifest(path)
    assert seal.sha256 == hashlib.sha256(canonical).hexdigest()
    assert tuple(record.symbol for record in seal.records) == ALPHA_MAX_CANDIDATE_SYMBOLS
    assert type(seal.raw_availability_start_by_symbol) is type(MappingProxyType({}))
    assert seal.raw_availability_start_by_symbol["TONUSDT"] == (_TONUSDT_RAW_AVAILABILITY_START)
    assert seal.feature_availability_start_by_symbol["TONUSDT"] == (
        _TONUSDT_FEATURE_AVAILABILITY_START
    )
    assert seal.raw_availability_end_by_symbol["TONUSDT"] == _TONUSDT_AVAILABILITY_END
    assert seal.feature_availability_end_by_symbol["TONUSDT"] == _TONUSDT_AVAILABILITY_END
    with pytest.raises(FrozenInstanceError):
        seal.sha256 = _HASH_A

    poisoned = _contract_manifest()
    poisoned["records"][0]["volume_unit"] = "quote_asset"  # type: ignore[index]
    path.write_bytes(json.dumps(poisoned, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    with pytest.raises(ValueError, match="contract_manifest_mismatch"):
        seal_alpha_max_contract_manifest(path)


def _daily_candidate_input(
    symbol: str,
    *,
    passes: bool,
    missing_last_bucket: bool = False,
    first_day_index: int = 0,
):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    volume = 40_000.0 if passes else 20_000.0
    daily_summaries: list[AlphaMaxDailyQuoteNotional] = []
    for day_index in range(first_day_index, 517):
        completed_4h_bucket_hours = (0, 4, 8, 12, 16, 20)
        if missing_last_bucket and day_index == 516:
            completed_4h_bucket_hours = (0, 4, 8, 12, 16)
        daily_summaries.append(
            AlphaMaxDailyQuoteNotional(
                day=(start + timedelta(days=day_index)).date(),
                quote_notional_usdt=100.0 * volume * len(completed_4h_bucket_hours),
                completed_4h_bucket_hours=completed_4h_bucket_hours,
            )
        )
    return AlphaMaxAdmissionDailyCandidateInput(
        symbol=symbol,
        daily_quote_notional=tuple(daily_summaries),
        consecutive_completed_daily_bars_before_train=366,
        causal_funding_coverage_complete=True,
        unresolved_daily_cross_section_count=0,
    )


def test_actual_517_day_train_admission_uses_daily_summaries_and_type7_vectors() -> None:
    inputs = {
        symbol: _daily_candidate_input(symbol, passes=index < 5)
        for index, symbol in enumerate(ALPHA_MAX_CANDIDATE_SYMBOLS)
    }
    result = compute_alpha_max_train_admission_from_daily_summaries(
        inputs,
        input_root_hashes={"warmup": _HASH_A, "train": _HASH_B},
    )
    assert result.artifact.admitted_symbols == ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    assert len(result.daily_quote_notional_by_symbol["ADAUSDT"]) == 517
    assert result.daily_quote_notional_by_symbol["ADAUSDT"][0].quote_notional_usdt == (24_000_000.0)
    payload = json.loads(result.artifact.canonical_bytes)
    assert payload["per_candidate"]["ADAUSDT"]["statistics"] == {
        "causal_funding_coverage_complete": True,
        "complete_train_4h_keys": True,
        "complete_train_daily_keys": True,
        "consecutive_completed_daily_bars_before_train": 366,
        "daily_quote_notional_day_count": 517,
        "median_quote_notional_usdt": 24_000_000.0,
        "p10_quote_notional_usdt": 24_000_000.0,
        "readable_monotone_unique_finite_partitions": True,
        "unresolved_daily_cross_section_count": 0,
    }
    assert result.sha256 == hashlib.sha256(result.canonical_bytes).hexdigest()


def test_train_admission_missing_bucket_is_not_synthetic_zero_and_fails_membership() -> None:
    inputs = {
        symbol: _daily_candidate_input(
            symbol,
            passes=index < 5,
            missing_last_bucket=index == 0,
        )
        for index, symbol in enumerate(ALPHA_MAX_CANDIDATE_SYMBOLS)
    }
    with pytest.raises(ValueError, match="insufficient_train_universe"):
        compute_alpha_max_train_admission_from_daily_summaries(
            inputs,
            input_root_hashes={"warmup": _HASH_A, "train": _HASH_B},
        )


def test_train_admission_rejects_ton_listing_gap_instead_of_fabricating_coverage() -> None:
    inputs = {
        symbol: _daily_candidate_input(
            symbol,
            passes=index < 5 or symbol == "TONUSDT",
            first_day_index=60 if symbol == "TONUSDT" else 0,
        )
        for index, symbol in enumerate(ALPHA_MAX_CANDIDATE_SYMBOLS)
    }

    result = compute_alpha_max_train_admission_from_daily_summaries(
        inputs,
        input_root_hashes={"warmup": _HASH_A, "train": _HASH_B},
    )

    assert result.artifact.admitted_symbols == ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    ton_days = result.daily_quote_notional_by_symbol["TONUSDT"]
    assert len(ton_days) == 457
    assert ton_days[0].day.isoformat() == "2024-03-01"
    payload = json.loads(result.artifact.canonical_bytes)
    assert payload["per_candidate"]["TONUSDT"] == {
        "admitted": False,
        "reasons": [
            "daily_quote_notional_day_count_mismatch",
            "incomplete_train_4h_keys",
            "incomplete_train_daily_keys",
        ],
        "statistics": {
            "causal_funding_coverage_complete": True,
            "complete_train_4h_keys": False,
            "complete_train_daily_keys": False,
            "consecutive_completed_daily_bars_before_train": 366,
            "daily_quote_notional_day_count": 457,
            "median_quote_notional_usdt": 24_000_000.0,
            "p10_quote_notional_usdt": 24_000_000.0,
            "readable_monotone_unique_finite_partitions": True,
            "unresolved_daily_cross_section_count": 0,
        },
    }


def _train_liquidity_bucket_fixture():
    admission = compute_alpha_max_train_admission_from_daily_summaries(
        {
            symbol: _daily_candidate_input(symbol, passes=index < 5)
            for index, symbol in enumerate(ALPHA_MAX_CANDIDATE_SYMBOLS)
        },
        input_root_hashes={"warmup": _HASH_A, "train": _HASH_B},
    )
    return build_alpha_max_train_liquidity_buckets(admission)


def test_e20_train_liquidity_buckets_are_tie_deterministic_and_canonical() -> None:
    buckets = _train_liquidity_bucket_fixture()

    assert buckets.admitted_symbols == ALPHA_MAX_CANDIDATE_SYMBOLS[:5]
    assert dict(buckets.symbols_by_bucket) == {
        "weakest": ("ADAUSDT", "AVAXUSDT"),
        "middle": ("BNBUSDT", "BTCUSDT"),
        "liquid": ("DOGEUSDT",),
    }
    assert validate_alpha_max_train_liquidity_buckets(buckets.canonical_bytes) == buckets
    payload = buckets.to_payload()
    payload["bucket_by_symbol"]["ADAUSDT"] = "liquid"
    with pytest.raises(ValueError, match="assignment_mismatch"):
        validate_alpha_max_train_liquidity_buckets(payload)


def _build_e20_falsifier(per_symbol: dict[str, float]):
    buckets = _train_liquidity_bucket_fixture()
    fold_count = 12
    return build_alpha_max_trend_liquidity_falsifier(
        domain="validation",
        train_liquidity_buckets=buckets,
        fold_run_sha256s=tuple(
            hashlib.sha256(f"validation-fold-{index}".encode()).hexdigest()
            for index in range(fold_count)
        ),
        symbol_contribution_usdt_by_fold=tuple(dict(per_symbol) for _ in range(fold_count)),
    )


def test_e20_falsifier_is_report_only_and_never_makes_a_positive_causal_claim() -> None:
    falsifier = _build_e20_falsifier(dict.fromkeys(ALPHA_MAX_CANDIDATE_SYMBOLS[:5], 1.0))

    assert falsifier.status == "liquidity_falsifier_not_triggered"
    assert falsifier.rejection_reasons == ()
    assert falsifier.to_payload()["selection_influence"] is False
    assert falsifier.to_payload()["report_only"] is True


def test_e20_offsetting_positive_middle_bucket_is_not_mislabeled_weakest_only() -> None:
    falsifier = _build_e20_falsifier(
        {
            "ADAUSDT": 2.0,
            "AVAXUSDT": 2.0,
            "BNBUSDT": 0.5,
            "BTCUSDT": 0.5,
            "DOGEUSDT": -2.0,
        }
    )

    assert falsifier.status == "trend_mechanism_not_supported"
    assert falsifier.rejection_reasons == ("liquid_bucket_nonpositive",)


def test_e20_rejects_edge_confined_to_weakest_train_liquidity_bucket() -> None:
    falsifier = _build_e20_falsifier(
        {
            "ADAUSDT": 3.0,
            "AVAXUSDT": 3.0,
            "BNBUSDT": -0.5,
            "BTCUSDT": -0.5,
            "DOGEUSDT": -1.0,
        }
    )

    assert falsifier.status == "trend_mechanism_not_supported"
    assert falsifier.rejection_reasons == (
        "liquid_bucket_nonpositive",
        "positive_edge_confined_to_weakest",
    )


def _gate(
    row_id: str,
    *,
    role: str = "prelock_selection",
    total: float = 0.20,
    cagr: float = 0.15,
    calmar: float = 0.50,
    sharpe: float = 1.0,
    full_mdd: float = 0.20,
    report_mdd: float = 0.20,
    dsr: float = 0.95,
) -> AlphaMaxGateInput:
    return AlphaMaxGateInput(
        row_id=row_id,
        comparison_role=role,
        evidence_tier="actual_engine",
        comparison_valid=True,
        nominal_cost_bps=30,
        cumulative_return=total,
        cagr=cagr,
        calmar=calmar,
        net_sharpe=sharpe,
        full_event_mdd=full_mdd,
        reporting_4h_mdd=report_mdd,
        dsr=dsr,
        spa_pvalue=0.01,
        pbo=0.10,
        native_data_coverage_complete=True,
        funding_coverage_complete=True,
        hash_valid=True,
        manifest_valid=True,
        reconciliation_complete=True,
        ruin=False,
        raw_root_set_sha256=_HASH_A,
        feature_root_set_sha256=_HASH_B,
        universe_sha256="c" * 64,
        calendar_sha256="d" * 64,
        seed_schedule_sha256="e" * 64,
    )


def test_gate_order_soft_mdd_comparator_and_return_first_selection_are_exact() -> None:
    normal = _gate("normal", total=0.25, cagr=0.20, calmar=0.60, full_mdd=0.30)
    soft = _gate("soft", total=0.40, cagr=0.21, calmar=0.61, full_mdd=0.35)
    soft_equal = _gate("soft_equal", total=0.50, cagr=0.20, calmar=0.70, full_mdd=0.31)
    hard = _gate("hard", total=0.90, cagr=0.50, calmar=1.0, full_mdd=0.3500001)
    early_fail = _gate("early", total=1.0, full_mdd=0.9, dsr=0.89)

    result = select_alpha_max_prelock_champion([hard, soft_equal, early_fail, soft, normal])
    assert result.prelock_champion == "soft"
    assert result.selected_candidate_id == "soft"
    assert result.ranked_candidate_ids == ("soft", "normal")
    by_id = {decision.row_id: decision for decision in result.decisions}
    assert by_id["normal"].mdd_band == "normal"
    assert by_id["soft"].mdd_band == "soft"
    assert by_id["soft_equal"].rejection_reasons == (
        "soft_mdd_not_strictly_superior_to_best_normal",
    )
    assert by_id["hard"].rejection_reasons == ("mdd_above_hard_limit",)
    assert by_id["early"].evaluated_gates == ("dsr",)
    assert by_id["early"].rejection_reasons == ("dsr_below_threshold",)


def test_scaled_selection_requires_passing_positive_frozen_1x_sibling() -> None:
    baseline = _gate("baseline", total=0.30)
    failing_sibling = _gate("full_equal_risk_1x", total=0.40, dsr=0.89)
    scaled = _gate("full_equal_risk_scaled", total=2.0, cagr=0.9, calmar=3.0)

    result = select_alpha_max_prelock_champion((scaled, failing_sibling, baseline))

    assert result.prelock_champion == "baseline"
    decisions = {value.row_id: value for value in result.decisions}
    assert decisions["full_equal_risk_1x"].rejection_reasons == ("dsr_below_threshold",)
    assert decisions["full_equal_risk_scaled"].rejection_reasons == (
        "scaled_1x_sibling_not_eligible",
    )
    attribution = result.scaling_attributions[0]
    assert attribution.scaled_row_id == "full_equal_risk_scaled"
    assert attribution.sibling_row_id == "full_equal_risk_1x"
    assert attribution.sibling_gross_exposure == 1.0
    assert attribution.exposure_normalization == "total_return / frozen_1x_gross"
    assert attribution.sibling_exposure_normalized_return == pytest.approx(0.40)
    assert attribution.sibling_dependency_satisfied is False
    assert attribution.attribution_label == "risk_transform_not_alpha"
    assert attribution.passive_scaled_counterfactual == "absent"
    assert attribution.scaled_minus_sibling_total_return == pytest.approx(1.60)
    assert len(attribution.matched_domain_sha256) == 64
    assert result.to_payload()["artifact_kind"] == "alpha_max_prelock_selection.v2"


def test_scaled_selection_rejects_nonpositive_1x_and_preserves_own_earlier_gate() -> None:
    nonpositive = _gate("full_shrunk_hrp_1x", total=-0.01)
    scaled = _gate("full_shrunk_hrp_scaled", total=1.0)
    result = select_alpha_max_prelock_champion((nonpositive, scaled))
    decisions = {value.row_id: value for value in result.decisions}
    assert decisions["full_shrunk_hrp_scaled"].rejection_reasons == (
        "scaled_1x_exposure_normalized_nonpositive",
    )
    assert result.scaling_attributions[0].sibling_positive_exposure_normalized is False

    both_fail = select_alpha_max_prelock_champion(
        (
            _gate("full_equal_risk_1x", dsr=0.80),
            _gate("full_equal_risk_scaled", dsr=0.70, total=9.0),
        )
    )
    both_decisions = {value.row_id: value for value in both_fail.decisions}
    assert both_decisions["full_equal_risk_scaled"].evaluated_gates == ("dsr",)
    assert both_decisions["full_equal_risk_scaled"].rejection_reasons == ("dsr_below_threshold",)


def test_passing_1x_allows_scaled_ranking_and_missing_sibling_rejects() -> None:
    sibling = _gate("full_equal_risk_1x", total=0.30)
    scaled = _gate("full_equal_risk_scaled", total=0.70, cagr=0.50, calmar=1.0)
    result = select_alpha_max_prelock_champion((sibling, scaled))
    assert result.prelock_champion == "full_equal_risk_scaled"
    assert result.scaling_attributions[0].sibling_dependency_satisfied is True
    assert result.scaling_attributions[0].dependency_rejection_reason is None

    with pytest.raises(ValueError, match="scaled_sibling_missing"):
        select_alpha_max_prelock_champion((scaled,))


def test_scaled_dependency_is_resolved_before_scaled_row_enters_comparator_universe() -> None:
    baseline = _gate("baseline", total=0.20, cagr=0.15, calmar=0.50, full_mdd=0.20)
    sibling = _gate(
        "full_equal_risk_1x",
        total=0.40,
        cagr=0.16,
        calmar=0.51,
        full_mdd=0.32,
    )
    scaled = _gate(
        "full_equal_risk_scaled",
        total=2.00,
        cagr=0.90,
        calmar=3.00,
        full_mdd=0.25,
    )

    result = select_alpha_max_prelock_champion((scaled, sibling, baseline))
    decisions = {value.row_id: value for value in result.decisions}
    attribution = result.scaling_attributions[0]

    assert decisions["full_equal_risk_1x"].eligible is True
    assert decisions["full_equal_risk_1x"].mdd_band == "soft"
    assert decisions["full_equal_risk_1x"].comparator_row_id == "baseline"
    assert attribution.sibling_gate_eligible is decisions["full_equal_risk_1x"].eligible
    assert attribution.sibling_dependency_satisfied is True
    assert attribution.dependency_rejection_reason is None
    assert decisions["full_equal_risk_scaled"].eligible is True
    assert decisions["full_equal_risk_scaled"].rejection_reasons == ()
    assert result.prelock_champion == "full_equal_risk_scaled"


def test_failed_1x_cannot_make_scaled_row_a_soft_mdd_comparator() -> None:
    baseline = _gate("baseline", total=0.20, cagr=0.15, calmar=0.50, full_mdd=0.20)
    sibling = _gate("full_equal_risk_1x", total=0.40, dsr=0.89)
    scaled = _gate(
        "full_equal_risk_scaled",
        total=2.00,
        cagr=0.90,
        calmar=3.00,
        full_mdd=0.25,
    )
    unrelated_soft = _gate(
        "unrelated_soft",
        total=0.80,
        cagr=0.50,
        calmar=2.00,
        full_mdd=0.32,
    )

    result = select_alpha_max_prelock_champion((unrelated_soft, scaled, sibling, baseline))
    decisions = {value.row_id: value for value in result.decisions}

    assert decisions["full_equal_risk_scaled"].eligible is False
    assert decisions["full_equal_risk_scaled"].rejection_reasons == (
        "scaled_1x_sibling_not_eligible",
    )
    assert decisions["unrelated_soft"].eligible is True
    assert decisions["unrelated_soft"].comparator_row_id == "baseline"


def test_authorized_scaled_normal_constrains_unrelated_soft_mdd_candidate() -> None:
    baseline = _gate("baseline", total=0.20, cagr=0.15, calmar=0.50, full_mdd=0.20)
    sibling = _gate("full_equal_risk_1x", total=0.30, cagr=0.18, calmar=0.60)
    scaled = _gate(
        "full_equal_risk_scaled",
        total=2.00,
        cagr=0.90,
        calmar=3.00,
        full_mdd=0.25,
    )
    unrelated_soft = _gate(
        "unrelated_soft",
        total=0.80,
        cagr=0.50,
        calmar=2.00,
        full_mdd=0.32,
    )

    result = select_alpha_max_prelock_champion((unrelated_soft, scaled, sibling, baseline))
    decisions = {value.row_id: value for value in result.decisions}

    assert decisions["full_equal_risk_1x"].eligible is True
    assert decisions["full_equal_risk_scaled"].eligible is True
    assert decisions["unrelated_soft"].eligible is False
    assert decisions["unrelated_soft"].comparator_row_id == "full_equal_risk_scaled"
    assert decisions["unrelated_soft"].rejection_reasons == (
        "soft_mdd_not_strictly_superior_to_best_normal",
    )


def test_historical_ranking_is_report_only_and_terminal_precedence_is_singular() -> None:
    rows = [
        _gate("champion", role="historical_report", total=0.2),
        _gate("other", role="historical_report", total=0.3),
    ]
    result = rank_alpha_max_historical_report(rows)
    assert result.historical_evaluation_leader == "other"
    assert result.selected_candidate_id is None
    assert result.prelock_champion is None

    assert (
        alpha_max_terminal_outcome(
            None,
            champion_historical_complete=True,
            champion_historical_passed=True,
        )
        == "no_demonstrated_alpha"
    )
    assert (
        alpha_max_terminal_outcome(
            "champion",
            champion_historical_complete=False,
            champion_historical_passed=True,
        )
        == "historical_evaluation_incomplete"
    )
    assert (
        alpha_max_terminal_outcome(
            "champion",
            champion_historical_complete=True,
            champion_historical_passed=False,
        )
        == "prelock_champion_historical_robustness_failed"
    )
    assert (
        alpha_max_terminal_outcome(
            "champion",
            champion_historical_complete=True,
            champion_historical_passed=True,
        )
        == "prelock_champion_historical_robustness_passed"
    )
    prelock = select_alpha_max_prelock_champion(
        [_gate("champion", role="prelock_selection", total=0.2)]
    )
    terminal = build_alpha_max_terminal_state(
        prelock_selection=prelock,
        champion_historical_nominal_30_cell=None,
        historical_ranking=result,
        incumbent_comparison_status="unavailable",
    )
    assert terminal.selected_candidate_id == "champion"
    assert terminal.leader_differs_from_prelock_champion is True
    assert terminal.terminal_outcome == "historical_evaluation_incomplete"
    assert terminal.historical_exposure_status == "historical_evaluation_incomplete"
    assert terminal.requires_fresh_confirmation is True
    assert terminal.confirmation_status == "not_run"

    no_champion = select_alpha_max_prelock_champion(
        [_gate("rejected", role="prelock_selection", dsr=0.10)]
    )
    report_only_terminal = build_alpha_max_terminal_state(
        prelock_selection=no_champion,
        champion_historical_nominal_30_cell=None,
        historical_ranking=result,
        incumbent_comparison_status="unavailable",
    )
    assert report_only_terminal.prelock_champion is None
    assert report_only_terminal.selected_candidate_id is None
    assert report_only_terminal.historical_evaluation_leader == "other"
    assert report_only_terminal.terminal_outcome == "no_demonstrated_alpha"
    assert report_only_terminal.historical_exposure_status == "committed_period_outcomes_observed"


def test_unavailable_row_cost_cells_and_prelock_inventory_are_immutable_canonical() -> None:
    cells = tuple(
        AlphaMaxCostCellEvidence.unavailable(
            row_id="incumbent",
            domain="validation",
            nominal_cost_bps=cost,
            status="incumbent_replay_unavailable",
        )
        for cost in (30, 10, 20, 15)
    )
    row = AlphaMaxRowEvidence(
        row_id="incumbent",
        matrix_role="incumbent",
        status="incumbent_replay_unavailable",
        evidence_tier="identity",
        selection_valid=False,
        cost_cells=cells,
    )
    assert tuple(cell.nominal_cost_bps for cell in row.cost_cells) == (10, 15, 20, 30)
    assert canonical_alpha_max_row_bytes(row) == canonical_alpha_max_row_bytes(row)

    artifacts_a = {
        "rows/incumbent.json": canonical_alpha_max_row_bytes(row),
        "config.json": b"{}\n",
    }
    artifacts_b = dict(reversed(tuple(artifacts_a.items())))
    first = build_alpha_max_prelock_seal(
        artifacts_a,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    second = build_alpha_max_prelock_seal(
        artifacts_b,
        prelock_champion=None,
        selected_candidate_id=None,
    )
    assert first == second
    assert first.sha256 == hashlib.sha256(first.canonical_bytes).hexdigest()
    with pytest.raises(FrozenInstanceError):
        first.sha256 = _HASH_A
    with pytest.raises(ValueError, match="prelock_artifact_path_invalid"):
        build_alpha_max_prelock_seal(
            {"../escape": b"x"},
            prelock_champion=None,
            selected_candidate_id=None,
        )
    with pytest.raises(ValueError, match="selection_identity_mismatch"):
        build_alpha_max_prelock_seal(
            artifacts_a,
            prelock_champion="a",
            selected_candidate_id="b",
        )
    with pytest.raises(ValueError, match="historical_input_forbidden"):
        build_alpha_max_prelock_seal(
            {"historical_evaluation/root-seal.json": b"poison"},
            prelock_champion=None,
            selected_candidate_id=None,
        )


def test_capsule_receipt_parses_causal_envelope_and_rejects_fold_relabel(
    tmp_path: Path,
) -> None:
    manifest_sha256 = "c" * 64
    capsule_scope = {"ready": True}
    capsule_sha256 = hashlib.sha256(
        json.dumps(capsule_scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state_payload = {
        "capsule": {**capsule_scope, "sha256": capsule_sha256},
        "capsule_sha256": capsule_sha256,
        "discarded_signal_count": 3,
        "fill_event_count": 0,
        "finalized_children": {"component": {"ready": True}},
        "funding_event_count": 0,
        "manifest_sha256": manifest_sha256,
        "market_event_count": 0,
        "native_finalization_sha256": "d" * 64,
        "order_event_count": 0,
        "phase_id": "purge",
        "portfolio_mode": "manifest:/stable/row-a.json",
        "trade_count": 0,
        "windows_processed": 10,
    }
    raw = AlphaMaxCapsuleReceipt.canonical_envelope_bytes(
        row_id="row-a",
        phase="validation_train_fit",
        prefix_id="validation_w01",
        manifest_sha256=manifest_sha256,
        state_payload=state_payload,
    )
    path = tmp_path / "capsule.json"
    path.write_bytes(raw)
    receipt = AlphaMaxCapsuleReceipt.from_path(
        path,
        row_id="row-a",
        phase="validation_train_fit",
        prefix_id="validation_w01",
        manifest_sha256=manifest_sha256,
        relative_path="capsules/row-a/validation_w01.json",
    )
    assert receipt.capsule_phase_id == "purge"
    assert receipt.state_payload["native_finalization_sha256"] == "d" * 64
    assert receipt.boundary_utc == evidence._ALPHA_MAX_FOLD_INTERVALS["validation_w01"][0]
    with pytest.raises(ValueError, match="envelope_scope_mismatch"):
        AlphaMaxCapsuleReceipt.from_path(
            path,
            row_id="row-a",
            phase="validation_train_fit",
            prefix_id="validation_w02",
            manifest_sha256=manifest_sha256,
            relative_path="capsules/row-a/validation_w02.json",
        )


def test_manifest_receipt_binds_materialized_file_stem_and_exact_bytes(tmp_path: Path) -> None:
    raw = b'{"children":[{"candidate_id":"component"}]}\n'
    path = tmp_path / "row-a.json"
    path.write_bytes(raw)
    materialization = AlphaMaxManifestMaterialization(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        canonical_bytes=raw,
        strategy_params={},
    )

    receipt = AlphaMaxManifestReceipt.from_materialization(
        materialization,
        phase="validation_train_fit",
        relative_path="manifests/validation_train_fit/row-a.json",
    )

    assert receipt.row_id == "row-a"
    assert receipt.sha256 == materialization.sha256
    assert receipt.byte_count == len(raw)


def test_effective_runtime_config_binds_all_attributes_and_rejects_seed_forge() -> None:
    admitted = validate_alpha_max_admitted_symbols(
        ALPHA_MAX_CANDIDATE_SYMBOLS,
        ALPHA_MAX_CANDIDATE_SYMBOLS[:5],
    )
    fold_id = "validation_w01"
    start, end = evidence._ALPHA_MAX_FOLD_INTERVALS[fold_id]
    static = {
        "DECISION_CADENCE_SECONDS": 1,
        "INITIAL_CAPITAL": 10_000.0,
        "STATIC_SENTINEL": "sealed",
    }
    payload = {
        **static,
        "END_DATE": end.isoformat().replace("+00:00", "Z"),
        "RANDOM_SEED": alpha_max_common_rng_seed(fold_id, 30),
        "SLIPPAGE_RATE": 0.0025,
        "START_DATE": start.isoformat().replace("+00:00", "Z"),
        "SYMBOLS": list(admitted),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    runtime = {
        "attribute_allowlist": sorted(payload),
        "static_attributes": static,
    }
    assert (
        evidence._alpha_max_validate_effective_config_bytes(
            raw,
            hashlib.sha256(raw).hexdigest(),
            split_or_fold_id=fold_id,
            nominal_cost_bps=30,
            admitted_symbols=admitted,
            runtime_contract_payload=runtime,
        )
        == payload
    )
    poisoned = {**payload, "RANDOM_SEED": payload["RANDOM_SEED"] + 1}
    poisoned_raw = json.dumps(poisoned, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="runtime_binding_mismatch"):
        evidence._alpha_max_validate_effective_config_bytes(
            poisoned_raw,
            hashlib.sha256(poisoned_raw).hexdigest(),
            split_or_fold_id=fold_id,
            nominal_cost_bps=30,
            admitted_symbols=admitted,
            runtime_contract_payload=runtime,
        )


def _fake_fold_runs_and_live_segments(domain: str):
    fold_runs = []
    segment_inputs = []
    aggregate = AlphaMaxStreamingEquityTracker()
    current = 10_000.0
    for index, fold_id in enumerate(evidence._ALPHA_MAX_DOMAIN_FOLD_IDS[domain]):
        start, _ = evidence._ALPHA_MAX_FOLD_INTERVALS[fold_id]
        timestamps = (
            int((start + timedelta(seconds=1)).timestamp() * 1000),
            int((start + timedelta(seconds=2)).timestamp() * 1000),
        )
        source_values = (8_000.0, 20_000.0) if index == 0 else (10_000.0, 10_000.0)
        source_tracker = AlphaMaxStreamingEquityTracker()
        for timestamp, value in zip(timestamps, source_values, strict=True):
            source_tracker.update(value, timestamp)
        source = source_tracker.finalize()
        run = object.__new__(evidence.AlphaMaxActualEngineRunReceipt)
        object.__setattr__(run, "domain", domain)
        object.__setattr__(run, "split_or_fold_id", fold_id)
        object.__setattr__(run, "full_event_equity", source)
        object.__setattr__(run, "liquidation_event_count", 0)
        object.__setattr__(run, "sha256", hashlib.sha256(f"run:{fold_id}".encode()).hexdigest())
        fold = object.__new__(evidence.AlphaMaxFoldRunEvidence)
        object.__setattr__(fold, "actual_engine_run", run)
        object.__setattr__(fold, "status", "complete")
        object.__setattr__(fold, "sha256", hashlib.sha256(f"fold:{fold_id}".encode()).hexdigest())
        fold_runs.append(fold)

        scale = current / 10_000.0
        normalized_values = tuple(scale * value for value in source_values)
        for timestamp, value in zip(timestamps, normalized_values, strict=True):
            aggregate.update(value, timestamp)
        normalized_end = normalized_values[-1]
        segment_inputs.append(
            build_alpha_max_normalized_fold_segment_evidence(
                fold_id=fold_id,
                source_streaming_equity_sha256=source.sha256,
                source_event_stream_sha256=source.event_stream_sha256,
                normalization_scale=scale,
                normalized_starting_equity=current,
                normalized_ending_equity=normalized_end,
                normalized_segment_event_stream_sha256=hashlib.sha256(
                    f"normalized:{fold_id}".encode()
                ).hexdigest(),
                event_count=source.event_count,
                first_timestamp_ms=source.first_timestamp_ms,
                last_timestamp_ms=source.last_timestamp_ms,
                aggregate_prefix_event_count=aggregate.event_count,
                aggregate_prefix_event_stream_sha256=aggregate.event_stream_sha256,
            )
        )
        current = normalized_end
    return tuple(fold_runs), aggregate.finalize(), tuple(segment_inputs)


def test_live_combined_stream_preserves_peak_after_min_chronology_and_hash_binding() -> None:
    fold_runs, live, segments = _fake_fold_runs_and_live_segments("validation")
    combined = evidence._build_alpha_max_combined_streaming_equity(
        fold_runs,
        live,
        segments,
    )
    assert len(combined.fold_ids) == 12
    assert combined.full_event_mdd == pytest.approx(0.20)
    assert evidence._ALPHA_MAX_DOMAIN_ENGINE_RUN_COUNT == {
        "validation": 816,
        "historical_exposed_evaluation": 680,
    }
    assert len(evidence._ALPHA_MAX_DOMAIN_FOLD_IDS["historical_exposed_evaluation"]) == 10

    fake = AlphaMaxStreamingEquityTracker()
    for index, segment in enumerate(segments):
        fake.update(5_000.0 if index == 0 else 20_000.0, segment.first_timestamp_ms)
        fake.update(20_000.0, segment.last_timestamp_ms)
    forged = fake.finalize()
    assert forged.event_count == live.event_count
    assert forged.ending_equity == live.ending_equity
    with pytest.raises(ValueError, match="live_binding_mismatch"):
        evidence._build_alpha_max_combined_streaming_equity(
            fold_runs,
            forged,
            segments,
        )


def test_typed_ruin_gate_is_rejected_without_fabricated_metrics() -> None:
    terminal = AlphaMaxTerminalGateEvidence(
        row_id="ruined",
        comparison_role="prelock_selection",
        domain="validation",
        nominal_cost_bps=30,
        pre_gate_evidence_sha256="1" * 64,
        fold_run_set_sha256="2" * 64,
        ruined_fold_ids=("validation_w01", "validation_w02"),
        streaming_ruin_fold_ids=("validation_w01",),
        liquidation_fold_ids=("validation_w02",),
        raw_root_set_sha256="3" * 64,
        feature_root_set_sha256="4" * 64,
        universe_sha256="5" * 64,
        seed_schedule_sha256="6" * 64,
    )
    result = select_alpha_max_prelock_champion((terminal,))
    assert result.prelock_champion is None
    assert result.decisions[0].rejection_reasons == ("ruin_detected",)
    assert result.decisions[0].gate_mdd is None


def test_streaming_full_event_tracker_is_exact_and_constant_memory_for_large_stream() -> None:
    tracker = AlphaMaxStreamingEquityTracker(initial_capital=10_000.0)
    values: list[float] = []
    peak = 10_000.0
    current_duration = 0
    expected_duration = 0
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for index in range(120_000):
        value = 10_000.0 + 750.0 * math.sin(index / 137.0) + (index % 11)
        values.append(value)
        timestamp = start + timedelta(seconds=index)
        tracker.observe((timestamp.timestamp(), value))
        if value >= peak:
            peak = value
            current_duration = 0
        else:
            current_duration += 1
            expected_duration = max(expected_duration, current_duration)

    snapshot = tracker.finalize()
    assert snapshot.full_event_mdd == pytest.approx(alpha_max_full_event_mdd(values))
    assert snapshot.max_drawdown_duration_events == expected_duration
    assert snapshot.event_count == len(values)
    assert snapshot.last_timestamp_ms == int(
        (start + timedelta(seconds=119_999)).timestamp() * 1000
    )
    assert tracker.retained_point_count == 0
    assert tracker.state_size_bytes < 4096
    with pytest.raises(FrozenInstanceError):
        snapshot.event_count = 1


@pytest.mark.parametrize(
    "value",
    (0.0, -0.0, 1.0, -25.125, 1e-10, 1e20, 9_999.999999999998),
)
def test_streaming_equity_record_fast_encoding_is_canonical(value: float) -> None:
    assert evidence._alpha_max_streaming_equity_record_bytes(value, 7, 1234) == (
        evidence._canonical_json_bytes(
            {"equity": value, "event_index": 7, "timestamp_ms": 1234},
            newline=True,
        )
    )


def test_streaming_full_event_batch_is_byte_exact_across_continuations() -> None:
    seconds = np.arange(1_700_000_000.0, 1_700_000_010.0, dtype=np.float64)
    equities = np.array(
        [
            10_000.0,
            10_000.0,
            9_500.0,
            9_000.0,
            10_000.0,
            10_500.0,
            10_500.0,
            0.0,
            -25.125,
            10_600.0,
        ],
        dtype=np.float64,
    )
    points = np.column_stack((seconds, equities))
    reference = AlphaMaxStreamingEquityTracker()
    for second, equity in points:
        reference.observe((float(second), float(equity)))
    batched = AlphaMaxStreamingEquityTracker()
    batched.update_batch(points[:4])
    batched.update_batch(points[4:])

    assert batched.finalize().to_payload() == reference.finalize().to_payload()
    assert batched.event_stream_sha256 == reference.event_stream_sha256


def test_streaming_full_event_tracker_rejects_malformed_portfolio_sink_points() -> None:
    tracker = AlphaMaxStreamingEquityTracker()

    with pytest.raises(TypeError, match="streaming_equity_point_invalid"):
        tracker.observe([1.0, 10_000.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="streaming_equity_unix_seconds_invalid"):
        tracker.observe((-1.0, 10_000.0))
    with pytest.raises(ValueError, match="streaming_equity_unix_seconds_invalid"):
        tracker.observe((1e308, 10_000.0))


def test_streaming_full_event_tracker_records_zero_and_negative_equity_as_ruin() -> None:
    tracker = AlphaMaxStreamingEquityTracker()
    tracker.observe((1.0, 10_000.0))
    tracker.observe((2.0, 0.0))
    tracker.observe((3.0, -25.0))

    snapshot = tracker.finalize()

    assert snapshot.ruin_detected is True
    assert snapshot.ending_equity == -25.0
    assert snapshot.full_event_mdd == 1.0
    assert snapshot.uncapped_full_event_drawdown == pytest.approx(1.0025)
