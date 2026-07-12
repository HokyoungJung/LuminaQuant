from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from scripts.research import prepare_alpha_max_phase_roots as subject


_MINI_PHASE_INTERVALS = {
    "warmup": (
        datetime(2024, 2, 29, tzinfo=UTC),
        datetime(2024, 2, 29, 8, tzinfo=UTC),
    ),
    "train": (
        datetime(2024, 2, 29, 8, tzinfo=UTC),
        datetime(2024, 2, 29, 16, tzinfo=UTC),
    ),
    "purge": (
        datetime(2024, 2, 29, 16, tzinfo=UTC),
        datetime(2024, 3, 1, tzinfo=UTC),
    ),
    "validation": (
        datetime(2024, 3, 1, tzinfo=UTC),
        datetime(2024, 3, 1, 8, tzinfo=UTC),
    ),
    "embargo": (
        datetime(2024, 3, 1, 8, tzinfo=UTC),
        datetime(2024, 3, 1, 16, tzinfo=UTC),
    ),
    "historical_exposed_evaluation": (
        datetime(2024, 3, 1, 16, tzinfo=UTC),
        datetime(2024, 3, 2, tzinfo=UTC),
    ),
}


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _write_contract_manifest(path: Path) -> None:
    inactive_start = datetime(2024, 2, 1, tzinfo=UTC)
    inactive_end = datetime(2024, 2, 2, tzinfo=UTC)
    protocol_start = datetime(2024, 2, 29, tzinfo=UTC)
    protocol_end = datetime(2024, 3, 2, tzinfo=UTC)
    ton_raw_start = datetime(2024, 3, 1, 12, 31, 10, tzinfo=UTC)
    ton_feature_start = datetime(2024, 3, 1, 16, tzinfo=UTC)
    ton_end = protocol_end
    records = []
    for symbol in subject.CANDIDATE_SYMBOLS:
        active = symbol == "BTCUSDT"
        is_ton = symbol == "TONUSDT"
        records.append(
            {
                "contract_multiplier": 1.0,
                "feature_availability_end_utc": _iso(
                    ton_end if is_ton else protocol_end if active else inactive_end
                ),
                "feature_availability_start_utc": _iso(
                    ton_feature_start if is_ton else protocol_start if active else inactive_start
                ),
                "inverse": False,
                "linear": True,
                "margin_asset": "USDT",
                "market_type": "perpetual",
                "quote_asset": "USDT",
                "raw_availability_end_utc": _iso(
                    ton_end if is_ton else protocol_end if active else inactive_end
                ),
                "raw_availability_start_utc": _iso(
                    ton_raw_start if is_ton else protocol_start if active else inactive_start
                ),
                "settle_asset": "USDT",
                "symbol": symbol,
                "volume_unit": "base_asset",
            }
        )
    payload = {
        "exchange": "binance",
        "records": records,
        "schema_version": "alpha_max_contract_manifest.v2",
    }
    path.write_bytes(subject._canonical_json_bytes(payload))


def _write_raw_month(root: Path, *, symbol: str, month: datetime) -> None:
    if month.month == 2:
        start = datetime(2024, 2, 29, tzinfo=UTC)
        end = datetime(2024, 3, 1, tzinfo=UTC)
    else:
        start = (
            datetime(2024, 3, 1, 12, 31, 10, tzinfo=UTC)
            if symbol == "TONUSDT"
            else datetime(2024, 3, 1, tzinfo=UTC)
        )
        end = datetime(2024, 3, 2, tzinfo=UTC)
    timestamps = pl.datetime_range(
        start,
        end,
        interval="1s",
        closed="left",
        time_zone="UTC",
        eager=True,
    )
    target = root / "binance" / symbol / f"{month:%Y-%m}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"datetime": timestamps}).with_columns(
        pl.lit(100.0).alias("open"),
        pl.lit(101.0).alias("high"),
        pl.lit(99.0).alias("low"),
        pl.lit(100.5).alias("close"),
        pl.lit(2.0).alias("volume"),
    ).write_parquet(target)


def _write_feature_day(root: Path, *, symbol: str, day: datetime) -> None:
    hours = (8, 16, 20) if symbol == "TONUSDT" else (0, 8, 16)
    canonical = [int((day + timedelta(hours=hour)).timestamp() * 1000) for hour in hours]
    jitters = [0, 0, 0] if symbol == "TONUSDT" else [0, 1000, 5]
    timestamps = [value + jitter for value, jitter in zip(canonical, jitters, strict=True)]
    target = (
        root
        / "exchange=binance"
        / f"symbol={symbol}"
        / f"date={day:%Y-%m-%d}"
        / f"compact-{day:%Y-%m-%d}.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp_ms": timestamps,
            "exchange": ["binance"] * len(timestamps),
            "symbol": [symbol] * len(timestamps),
            "funding_rate": [0.0001 * (index + 1) for index in range(len(timestamps))],
            "unused_feature": [float(index) for index in range(len(timestamps))],
        }
    ).write_parquet(target)


@pytest.fixture
def phase_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(subject, "PHASE_INTERVALS", _MINI_PHASE_INTERVALS)
    raw_root = tmp_path / "market_ohlcv_1s"
    feature_root = tmp_path / "feature_points"
    raw_root.mkdir()
    feature_root.mkdir()
    manifest = tmp_path / "contract-v2.json"
    _write_contract_manifest(manifest)
    monkeypatch.setattr(
        subject,
        "_APPROVED_CONTRACT_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )

    for month in (
        datetime(2024, 2, 1, tzinfo=UTC),
        datetime(2024, 3, 1, tzinfo=UTC),
    ):
        _write_raw_month(raw_root, symbol="BTCUSDT", month=month)
    _write_raw_month(
        raw_root,
        symbol="TONUSDT",
        month=datetime(2024, 3, 1, tzinfo=UTC),
    )

    day = datetime(2024, 2, 29, tzinfo=UTC)
    while day < datetime(2024, 3, 2, tzinfo=UTC):
        _write_feature_day(feature_root, symbol="BTCUSDT", day=day)
        day += timedelta(days=1)
    _write_feature_day(
        feature_root,
        symbol="TONUSDT",
        day=datetime(2024, 3, 1, tzinfo=UTC),
    )

    return {
        "raw": raw_root,
        "feature": feature_root,
        "manifest": manifest,
        "parent": tmp_path,
    }


def _prepare(sources: dict[str, Path], output: Path) -> dict[str, object]:
    return subject.prepare_alpha_max_phase_roots(
        raw_root=sources["raw"],
        feature_root=sources["feature"],
        contract_manifest=sources["manifest"],
        output_root=output,
    )


def _retained_temporary_root(parent: Path, output_name: str) -> Path:
    retained = list(parent.glob(f".{output_name}.prepare-*"))
    assert len(retained) == 1
    return retained[0]


def test_approved_contract_digest_matches_repository_manifest() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = (
        repository_root / "configs/research/alpha_max_contract_manifest_20260711_listing_aware.json"
    )
    payload = manifest.read_bytes()

    assert hashlib.sha256(payload).hexdigest() == subject._APPROVED_CONTRACT_MANIFEST_SHA256
    availability, observed_sha256 = subject._read_contract_manifest(manifest)
    assert tuple(value.symbol for value in availability) == subject.CANDIDATE_SYMBOLS
    assert observed_sha256 == subject._APPROVED_CONTRACT_MANIFEST_SHA256


@pytest.mark.parametrize("mutation", ("contract_multiplier", "ton_raw_start"))
def test_rejects_canonical_but_unapproved_contract_manifest(
    phase_sources: dict[str, Path],
    mutation: str,
) -> None:
    manifest = phase_sources["manifest"]
    document = json.loads(manifest.read_bytes())
    if mutation == "contract_multiplier":
        document["records"][0]["contract_multiplier"] = 999.0
    else:
        ton = next(record for record in document["records"] if record["symbol"] == "TONUSDT")
        ton["raw_availability_start_utc"] = "2024-03-01T00:00:00Z"
    manifest.write_bytes(subject._canonical_json_bytes(document))
    output = phase_sources["parent"] / f"failed-unapproved-{mutation}"

    with pytest.raises(ValueError, match="contract_manifest_unapproved"):
        _prepare(phase_sources, output)

    assert not output.exists()
    assert not list(phase_sources["parent"].glob(f".{output.name}.prepare-*"))


def test_materializes_deterministic_half_open_availability_owned_roots(
    phase_sources: dict[str, Path],
) -> None:
    output = phase_sources["parent"] / "phase-roots-a"
    result = _prepare(phase_sources, output)

    for phase_id in _MINI_PHASE_INTERVALS:
        assert (output / phase_id / "raw").is_dir()
        assert (output / phase_id / "feature").is_dir()

    assert not list((output / "validation" / "raw").rglob("*TONUSDT*"))
    assert not list((output / "validation" / "feature").rglob("*TONUSDT*"))

    ton_embargo_raw = pl.read_parquet(
        output / "embargo/raw/market_ohlcv_1s/binance/TONUSDT/2024-03.parquet"
    )
    assert ton_embargo_raw.get_column("datetime").min() >= datetime(
        2024,
        3,
        1,
        12,
        31,
        10,
        tzinfo=UTC,
    )
    assert ton_embargo_raw.get_column("datetime").max() == datetime(
        2024, 3, 1, 15, 59, 59, tzinfo=UTC
    )

    ton_historical_raw = pl.read_parquet(
        output / "historical_exposed_evaluation/raw/market_ohlcv_1s/binance/TONUSDT/2024-03.parquet"
    )
    assert ton_historical_raw.get_column("datetime").max() == datetime(
        2024, 3, 1, 23, 59, 59, tzinfo=UTC
    )

    ton_embargo_feature = (
        output
        / "embargo/feature/feature_points/exchange=binance/symbol=TONUSDT"
        / "date=2024-03-01/part-0.parquet"
    )
    assert not ton_embargo_feature.exists()

    ton_historical_feature = pl.read_parquet(
        output
        / "historical_exposed_evaluation/feature/feature_points/exchange=binance"
        / "symbol=TONUSDT/date=2024-03-01/part-0.parquet"
    )
    assert ton_historical_feature.get_column("timestamp_ms").to_list() == [
        int(datetime(2024, 3, 1, hour, tzinfo=UTC).timestamp() * 1000) for hour in (16, 20)
    ]
    assert ton_historical_feature.get_column("source_timestamp_ms").to_list() == [
        int(datetime(2024, 3, 1, hour, tzinfo=UTC).timestamp() * 1000) for hour in (16, 20)
    ]
    assert not (
        output
        / "historical_exposed_evaluation/feature/feature_points/exchange=binance"
        / "symbol=TONUSDT/date=2024-03-02"
    ).exists()

    btc_warmup_raw = pl.read_parquet(
        output / "warmup/raw/market_ohlcv_1s/binance/BTCUSDT/2024-02.parquet"
    )
    assert btc_warmup_raw.get_column("datetime").min() == datetime(2024, 2, 29, tzinfo=UTC)
    assert btc_warmup_raw.get_column("datetime").max() == datetime(
        2024, 2, 29, 7, 59, 59, tzinfo=UTC
    )
    assert btc_warmup_raw.height == 8 * 60 * 60

    preparation_bytes = (output / "preparation_manifest.json").read_bytes()
    preparation = json.loads(preparation_bytes)
    assert preparation_bytes == subject._canonical_json_bytes(preparation)
    assert result["preparation_manifest_sha256"] == subject._sha256_bytes(preparation_bytes)
    output_files = {
        path.relative_to(output).as_posix()
        for phase_id in _MINI_PHASE_INTERVALS
        for kind in ("raw", "feature")
        for path in (output / phase_id / kind).rglob("*.parquet")
    }
    manifest_files = {entry["output_relative_path"] for entry in preparation["files"]}
    assert output_files == manifest_files
    assert preparation["file_count"] == len(output_files) == result["file_count"]
    assert [entry["output_relative_path"] for entry in preparation["files"]] == sorted(output_files)
    entries_by_path = {entry["output_relative_path"]: entry for entry in preparation["files"]}
    for relative in output_files:
        published_path = output / relative
        payload = published_path.read_bytes()
        entry = entries_by_path[relative]
        assert len(payload) == entry["output_byte_count"]
        assert hashlib.sha256(payload).hexdigest() == entry["output_sha256"]
        observed = published_path.lstat()
        assert stat.S_ISREG(observed.st_mode)
        assert not stat.S_ISLNK(observed.st_mode)
        assert observed.st_nlink == 1
    for path in (output, *output.rglob("*")):
        observed = path.lstat()
        if stat.S_ISDIR(observed.st_mode):
            assert stat.S_IMODE(observed.st_mode) == 0o555
        else:
            assert stat.S_ISREG(observed.st_mode)
            assert stat.S_IMODE(observed.st_mode) == 0o444

    second_output = phase_sources["parent"] / "phase-roots-b"
    _prepare(phase_sources, second_output)
    assert (second_output / "preparation_manifest.json").read_bytes() == preparation_bytes


def test_postavailability_missing_partition_rejects_without_publishing(
    phase_sources: dict[str, Path],
) -> None:
    missing = phase_sources["feature"] / "exchange=binance/symbol=TONUSDT/date=2024-03-01"
    shutil.rmtree(missing)
    output = phase_sources["parent"] / "failed-roots"

    with pytest.raises(ValueError, match="feature_partition_missing"):
        _prepare(phase_sources, output)

    assert not output.exists()
    assert _retained_temporary_root(phase_sources["parent"], output.name).is_dir()


@pytest.mark.parametrize("mutation", ("raw_hardlink", "feature_symlink"))
def test_rejects_linked_source_without_publishing(
    phase_sources: dict[str, Path],
    mutation: str,
) -> None:
    if mutation == "raw_hardlink":
        source = phase_sources["raw"] / "binance/BTCUSDT/2024-02.parquet"
        os.link(source, phase_sources["parent"] / "raw-hardlink.parquet")
        expected = "hardlink_rejected"
    else:
        source = (
            phase_sources["feature"]
            / "exchange=binance/symbol=BTCUSDT/date=2024-02-29"
            / "compact-2024-02-29.parquet"
        )
        backing = phase_sources["parent"] / "feature-backing.parquet"
        source.rename(backing)
        source.symlink_to(backing)
        expected = "symlink_rejected"
    output = phase_sources["parent"] / f"failed-{mutation}"

    with pytest.raises(ValueError, match=expected):
        _prepare(phase_sources, output)

    assert not output.exists()


def test_existing_target_is_never_replaced(phase_sources: dict[str, Path]) -> None:
    output = phase_sources["parent"] / "existing-roots"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="output_root_must_be_absent"):
        _prepare(phase_sources, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_frozen_temporary_tree_is_identity_safely_retained_when_publish_fails(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = phase_sources["parent"] / "failed-after-freeze"
    observed_frozen = False

    def failing_publish(
        temporary_name: str,
        *,
        parent,
        temporary_fd: int,
        **_kwargs,
    ) -> None:
        nonlocal observed_frozen
        temporary_root = Path(f"/proc/self/fd/{parent.descriptor}") / temporary_name
        assert stat.S_IMODE(os.fstat(temporary_fd).st_mode) == 0o555
        for path in temporary_root.rglob("*"):
            observed = path.lstat()
            assert stat.S_IMODE(observed.st_mode) == (
                0o555 if stat.S_ISDIR(observed.st_mode) else 0o444
            )
        observed_frozen = True
        raise ValueError("forced_publish_failure")

    monkeypatch.setattr(subject, "_publish", failing_publish)

    with pytest.raises(ValueError, match="forced_publish_failure"):
        _prepare(phase_sources, output)

    assert observed_frozen is True
    assert not output.exists()
    retained = _retained_temporary_root(phase_sources["parent"], output.name)
    assert stat.S_IMODE(retained.lstat().st_mode) == 0o555


def test_publish_revalidates_frozen_file_mode_after_output_parent_check(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = phase_sources["parent"] / "failed-frozen-mode"
    original_revalidate = subject._revalidate_pinned_directory
    mutated = False

    def hostile_revalidate(capability, *, field: str):
        nonlocal mutated
        result = original_revalidate(capability, field=field)
        if field == "output_parent" and not mutated:
            parent = Path(f"/proc/self/fd/{capability.descriptor}")
            temporary_root = next(parent.glob(f".{output.name}.prepare-*"))
            next(temporary_root.rglob("*.parquet")).chmod(0o644)
            mutated = True
        return result

    monkeypatch.setattr(subject, "_revalidate_pinned_directory", hostile_revalidate)

    with pytest.raises(ValueError, match="output_frozen_file_mode_invalid"):
        _prepare(phase_sources, output)

    assert mutated is True
    assert not output.exists()
    assert _retained_temporary_root(phase_sources["parent"], output.name).is_dir()


@pytest.mark.parametrize("directory_kind", ("root", "nested"))
def test_publish_revalidates_frozen_directory_mode_after_output_parent_check(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    directory_kind: str,
) -> None:
    output = phase_sources["parent"] / f"failed-frozen-directory-mode-{directory_kind}"
    original_revalidate = subject._revalidate_pinned_directory
    mutated = False

    def hostile_revalidate(capability, *, field: str):
        nonlocal mutated
        result = original_revalidate(capability, field=field)
        if field == "output_parent" and not mutated:
            parent = Path(f"/proc/self/fd/{capability.descriptor}")
            temporary_root = next(parent.glob(f".{output.name}.prepare-*"))
            target = (
                temporary_root
                if directory_kind == "root"
                else next(path for path in temporary_root.rglob("*") if path.is_dir())
            )
            target.chmod(0o755)
            mutated = True
        return result

    monkeypatch.setattr(subject, "_revalidate_pinned_directory", hostile_revalidate)

    with pytest.raises(ValueError, match="output_frozen_directory_mode_invalid"):
        _prepare(phase_sources, output)

    assert mutated is True
    assert not output.exists()
    assert _retained_temporary_root(phase_sources["parent"], output.name).is_dir()


@pytest.mark.parametrize("relative_kind", ("parquet", "preparation_manifest"))
def test_publish_revalidates_frozen_file_sha256_after_output_parent_check(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    relative_kind: str,
) -> None:
    output = phase_sources["parent"] / f"failed-frozen-hash-{relative_kind}"
    original_revalidate = subject._revalidate_pinned_directory
    mutated = False

    def hostile_revalidate(capability, *, field: str):
        nonlocal mutated
        result = original_revalidate(capability, field=field)
        if field == "output_parent" and not mutated:
            parent = Path(f"/proc/self/fd/{capability.descriptor}")
            temporary_root = next(parent.glob(f".{output.name}.prepare-*"))
            target = (
                next(temporary_root.rglob("*.parquet"))
                if relative_kind == "parquet"
                else temporary_root / "preparation_manifest.json"
            )
            payload = bytearray(target.read_bytes())
            payload[len(payload) // 2] ^= 1
            target.chmod(0o644)
            target.write_bytes(payload)
            target.chmod(0o444)
            mutated = True
        return result

    monkeypatch.setattr(subject, "_revalidate_pinned_directory", hostile_revalidate)

    with pytest.raises(ValueError, match="output_frozen_file_sha256_mismatch"):
        _prepare(phase_sources, output)

    assert mutated is True
    assert not output.exists()
    assert _retained_temporary_root(phase_sources["parent"], output.name).is_dir()


def test_publish_revalidates_frozen_file_byte_count_after_output_parent_check(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = phase_sources["parent"] / "failed-frozen-byte-count"
    original_revalidate = subject._revalidate_pinned_directory
    mutated = False

    def hostile_revalidate(capability, *, field: str):
        nonlocal mutated
        result = original_revalidate(capability, field=field)
        if field == "output_parent" and not mutated:
            parent = Path(f"/proc/self/fd/{capability.descriptor}")
            temporary_root = next(parent.glob(f".{output.name}.prepare-*"))
            target = next(temporary_root.rglob("*.parquet"))
            payload = target.read_bytes()
            target.chmod(0o644)
            target.write_bytes(payload[:-1])
            target.chmod(0o444)
            mutated = True
        return result

    monkeypatch.setattr(subject, "_revalidate_pinned_directory", hostile_revalidate)

    with pytest.raises(ValueError, match="output_frozen_file_byte_count_mismatch"):
        _prepare(phase_sources, output)

    assert mutated is True
    assert not output.exists()
    assert _retained_temporary_root(phase_sources["parent"], output.name).is_dir()


@pytest.mark.parametrize("replacement_kind", ("file", "directory"))
def test_freeze_rejects_preverified_owned_object_replacement_without_chmod(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    output = phase_sources["parent"] / f"failed-pre-freeze-{replacement_kind}"
    original_verify = subject._verify_cleanup_tree_descriptor
    marker = b"attacker-owned-before-freeze\n"
    replacement_relative: Path | None = None
    replacement_mode = 0o640 if replacement_kind == "file" else 0o711
    swapped = False

    def hostile_verify(directory_fd: int, **kwargs) -> None:
        nonlocal replacement_relative, swapped
        original_verify(directory_fd, **kwargs)
        if swapped:
            return
        root = Path(f"/proc/self/fd/{directory_fd}")
        target = (
            next(root.rglob("*.parquet"))
            if replacement_kind == "file"
            else next(path for path in root.rglob("*") if path.is_dir())
        )
        replacement_relative = target.relative_to(root)
        displaced = target.with_name(f".{target.name}.retained-original")
        target.rename(displaced)
        if replacement_kind == "file":
            target.write_bytes(marker)
            target.chmod(replacement_mode)
        else:
            target.mkdir(mode=replacement_mode)
            (target / "marker").write_bytes(marker)
        swapped = True

    monkeypatch.setattr(subject, "_verify_cleanup_tree_descriptor", hostile_verify)

    with pytest.raises(ValueError, match=f"freeze_{replacement_kind}_identity_mismatch"):
        _prepare(phase_sources, output)

    assert swapped is True
    retained = _retained_temporary_root(phase_sources["parent"], output.name)
    assert replacement_relative is not None
    replacement = retained / replacement_relative
    assert stat.S_IMODE(replacement.lstat().st_mode) == replacement_mode
    if replacement_kind == "file":
        assert replacement.read_bytes() == marker
    else:
        assert (replacement / "marker").read_bytes() == marker
    assert replacement.with_name(f".{replacement.name}.retained-original").exists()
    assert not output.exists()


def test_postfreeze_baseline_stays_bound_to_pre_freeze_file_identity(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = phase_sources["parent"] / "failed-post-freeze-baseline-replacement"
    original_freeze = subject._freeze_tree_descriptor
    replacement_relative: Path | None = None
    replacement_identity: tuple[int, int] | None = None
    displaced = phase_sources["parent"] / "post-freeze-retained-original.parquet"
    replaced = False

    def hostile_freeze(directory_fd: int, **kwargs) -> None:
        nonlocal replaced, replacement_identity, replacement_relative
        original_freeze(directory_fd, **kwargs)
        root = Path(f"/proc/self/fd/{directory_fd}")
        target = next(root.rglob("*.parquet"))
        payload = target.read_bytes()
        replacement_relative = target.relative_to(root)
        target.parent.chmod(0o755)
        target.rename(displaced)
        target.write_bytes(payload)
        target.chmod(0o444)
        target.parent.chmod(0o555)
        observed = target.lstat()
        replacement_identity = (observed.st_dev, observed.st_ino)
        replaced = True

    monkeypatch.setattr(subject, "_freeze_tree_descriptor", hostile_freeze)

    with pytest.raises(ValueError, match="output_frozen_file_identity_mismatch"):
        _prepare(phase_sources, output)

    assert replaced is True
    retained = _retained_temporary_root(phase_sources["parent"], output.name)
    assert replacement_relative is not None
    replacement = retained / replacement_relative
    observed = replacement.lstat()
    assert replacement_identity == (observed.st_dev, observed.st_ino)
    assert stat.S_IMODE(observed.st_mode) == 0o444
    assert displaced.exists()
    assert not output.exists()


@pytest.mark.parametrize("relative_kind", ("parquet", "preparation_manifest"))
def test_publish_rejects_same_content_frozen_file_identity_replacement(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    relative_kind: str,
) -> None:
    output = phase_sources["parent"] / f"failed-frozen-identity-{relative_kind}"
    original_revalidate = subject._revalidate_pinned_directory
    replacement_identity: tuple[int, int] | None = None
    replacement_relative: Path | None = None
    replaced = False

    def hostile_revalidate(capability, *, field: str):
        nonlocal replaced, replacement_identity, replacement_relative
        result = original_revalidate(capability, field=field)
        if field == "output_parent" and not replaced:
            parent = Path(f"/proc/self/fd/{capability.descriptor}")
            temporary_root = next(parent.glob(f".{output.name}.prepare-*"))
            target = (
                next(temporary_root.rglob("*.parquet"))
                if relative_kind == "parquet"
                else temporary_root / "preparation_manifest.json"
            )
            payload = target.read_bytes()
            displaced = target.with_name(f".{target.name}.displaced")
            target.parent.chmod(0o755)
            target.rename(displaced)
            target.write_bytes(payload)
            target.chmod(0o444)
            displaced.unlink()
            target.parent.chmod(0o555)
            observed = target.lstat()
            replacement_identity = (observed.st_dev, observed.st_ino)
            replacement_relative = target.relative_to(temporary_root)
            replaced = True
        return result

    monkeypatch.setattr(subject, "_revalidate_pinned_directory", hostile_revalidate)

    with pytest.raises(ValueError, match="output_frozen_file_identity_mismatch"):
        _prepare(phase_sources, output)

    assert replaced is True
    assert not output.exists()
    remaining_roots = list(phase_sources["parent"].glob(f".{output.name}.prepare-*"))
    assert len(remaining_roots) == 1
    assert replacement_identity is not None
    assert replacement_relative is not None
    preserved = (remaining_roots[0] / replacement_relative).lstat()
    assert (preserved.st_dev, preserved.st_ino) == replacement_identity


def test_cleanup_retains_owned_temporary_identity_and_preserves_replacement(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = phase_sources["parent"] / "failed-temp-replacement"
    moved_original: Path | None = None
    attacker_root: Path | None = None
    marker = b"attacker-owned\n"

    def failing_publish(
        temporary_name: str,
        *,
        parent,
        **_kwargs,
    ) -> None:
        nonlocal attacker_root, moved_original
        parent_path = Path(f"/proc/self/fd/{parent.descriptor}")
        temporary_root = parent_path / temporary_name
        moved_original = phase_sources["parent"] / f"{temporary_name}.retained-original"
        temporary_root.rename(moved_original)
        attacker_root = phase_sources["parent"] / temporary_name
        attacker_root.mkdir()
        (attacker_root / "marker").write_bytes(marker)
        raise ValueError("forced_publish_failure")

    monkeypatch.setattr(subject, "_publish", failing_publish)

    with pytest.raises(ValueError, match="forced_publish_failure"):
        _prepare(phase_sources, output)

    assert moved_original is not None
    assert attacker_root is not None
    assert moved_original.is_dir()
    assert (attacker_root / "marker").read_bytes() == marker
    assert not output.exists()


@pytest.mark.parametrize("replacement_kind", ("symlink", "directory"))
def test_output_creation_stays_inside_retained_root_fd_when_ancestor_is_replaced(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    output = phase_sources["parent"] / f"failed-output-ancestor-{replacement_kind}"
    external = phase_sources["parent"] / f"external-attacker-{replacement_kind}"
    external.mkdir()
    marker = b"external-attacker-owned\n"
    marker_path = external / "marker"
    marker_path.write_bytes(marker)
    external.chmod(0o711)
    original_open_parent = subject._open_owned_output_parent
    attacked_relative: Path | None = None
    attacked = False

    def hostile_open_parent(directory_fd: int, relative_parts: tuple[str, ...], **kwargs):
        nonlocal attacked, attacked_relative
        parent_fd = original_open_parent(directory_fd, relative_parts, **kwargs)
        if not attacked and "market_ohlcv_1s" in relative_parts:
            root = Path(f"/proc/self/fd/{directory_fd}")
            boundary_index = relative_parts.index("market_ohlcv_1s") + 1
            attacked_relative = Path(*relative_parts[:boundary_index])
            ancestor = root / attacked_relative
            displaced = ancestor.with_name(f".{ancestor.name}.retained-original")
            ancestor.rename(displaced)
            if replacement_kind == "symlink":
                ancestor.symlink_to(external, target_is_directory=True)
            else:
                ancestor.mkdir(mode=0o711)
                (ancestor / "marker").write_bytes(b"in-tree-attacker-owned\n")
            attacked = True
        return parent_fd

    monkeypatch.setattr(subject, "_open_owned_output_parent", hostile_open_parent)

    with pytest.raises(ValueError, match="output_owned_directory"):
        _prepare(phase_sources, output)

    assert attacked is True
    assert marker_path.read_bytes() == marker
    assert stat.S_IMODE(external.lstat().st_mode) == 0o711
    assert not list(external.rglob("*.parquet"))
    retained = _retained_temporary_root(phase_sources["parent"], output.name)
    assert attacked_relative is not None
    attacker_entry = retained / attacked_relative
    if replacement_kind == "symlink":
        assert attacker_entry.is_symlink()
    else:
        assert (attacker_entry / "marker").read_bytes() == b"in-tree-attacker-owned\n"
        assert stat.S_IMODE(attacker_entry.lstat().st_mode) == 0o711
    assert not output.exists()


@pytest.mark.parametrize("replacement_kind", ("file", "directory"))
def test_failure_retention_never_deletes_last_moment_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    root = tmp_path / "owned-root"
    root.mkdir()
    marker = b"attacker-last-moment-replacement\n"
    if replacement_kind == "file":
        victim = root / "victim"
        victim.write_bytes(b"owned\n")
        descriptor = os.open(victim, os.O_RDONLY)
        try:
            fingerprint = subject._fingerprint_owned_output(
                descriptor,
                relative_path="victim",
            )
        finally:
            os.close(descriptor)
        snapshot = subject.FrozenTreeSnapshot(
            directory_identities={"": subject._directory_identity(root.lstat())},
            file_fingerprints={"victim": fingerprint},
        )
        victim.rename(root / ".victim.retained-original")
        victim.write_bytes(marker)
        victim.chmod(0o640)
    else:
        victim = root / "victim"
        victim.mkdir()
        snapshot = subject.FrozenTreeSnapshot(
            directory_identities={
                "": subject._directory_identity(root.lstat()),
                "victim": subject._directory_identity(victim.lstat()),
            },
            file_fingerprints={},
        )
        victim.rename(root / ".victim.retained-original")
        victim.mkdir(mode=0o711)
        (victim / "marker").write_bytes(marker)

    def destructive_call_forbidden(*_args, **_kwargs):
        raise AssertionError("failure retention must remain read-only")

    monkeypatch.setattr(subject.os, "unlink", destructive_call_forbidden)
    monkeypatch.setattr(subject.os, "rmdir", destructive_call_forbidden)
    monkeypatch.setattr(subject.os, "fchmod", destructive_call_forbidden)
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(ValueError, match=r"rollback_.*identity_mismatch"):
            subject._rollback_directory_by_identity(
                parent_fd,
                expected_identity=subject._directory_identity(root.lstat()),
                snapshot=snapshot,
            )
    finally:
        os.close(parent_fd)

    assert (root / ".victim.retained-original").exists()
    if replacement_kind == "file":
        assert victim.read_bytes() == marker
        assert stat.S_IMODE(victim.lstat().st_mode) == 0o640
    else:
        assert (victim / "marker").read_bytes() == marker
        assert stat.S_IMODE(victim.lstat().st_mode) == 0o711


@pytest.mark.parametrize("relative_kind", ("parquet", "preparation_manifest"))
def test_creation_descriptor_fingerprint_preserves_file_replaced_before_snapshot_record(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    relative_kind: str,
) -> None:
    output = phase_sources["parent"] / f"failed-pre-freeze-{relative_kind}-replacement"
    original_record = subject._record_owned_file_descriptor
    attacker_identity: tuple[int, int] | None = None
    creation_identity: tuple[int, int] | None = None
    attacker_relative: Path | None = None
    marker = b"attacker-owned-before-freeze\n"
    replaced = False

    def hostile_record(
        directory_fd: int,
        fingerprint: subject.SourceFingerprint,
        **kwargs,
    ) -> None:
        nonlocal attacker_identity, attacker_relative, creation_identity, replaced
        should_replace = (
            relative_kind == "preparation_manifest"
            and fingerprint.relative_path == "preparation_manifest.json"
        ) or (relative_kind == "parquet" and fingerprint.relative_path.endswith(".parquet"))
        if replaced or not should_replace:
            original_record(directory_fd, fingerprint, **kwargs)
            return
        temporary_root = Path(f"/proc/self/fd/{directory_fd}")
        target = temporary_root / fingerprint.relative_path
        displaced = target.with_name(f".{target.name}.displaced")
        target.rename(displaced)
        target.write_bytes(marker)
        displaced.unlink()
        observed = target.lstat()
        attacker_identity = (observed.st_dev, observed.st_ino)
        creation_identity = (fingerprint.identity[0], fingerprint.identity[1])
        attacker_relative = Path(fingerprint.relative_path)
        replaced = True
        original_record(directory_fd, fingerprint, **kwargs)
        raise ValueError("forced_pre_freeze_failure")

    monkeypatch.setattr(subject, "_record_owned_file_descriptor", hostile_record)

    with pytest.raises(ValueError, match="forced_pre_freeze_failure"):
        _prepare(phase_sources, output)

    assert replaced is True
    remaining_roots = list(phase_sources["parent"].glob(f".{output.name}.prepare-*"))
    assert len(remaining_roots) == 1
    assert attacker_identity is not None
    assert creation_identity is not None
    assert creation_identity != attacker_identity
    assert attacker_relative is not None
    preserved_path = remaining_roots[0] / attacker_relative
    preserved = preserved_path.lstat()
    assert (preserved.st_dev, preserved.st_ino) == attacker_identity
    assert preserved_path.read_bytes() == marker
    assert not output.exists()


def test_publish_final_open_rejects_stat_to_open_symlink_swap(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = phase_sources["parent"] / "failed-final-open-symlink-swap"
    original_revalidate = subject._revalidate_pinned_directory
    original_open = os.open
    armed = False
    swapped = False

    def arm_after_parent_check(capability, *, field: str):
        nonlocal armed
        result = original_revalidate(capability, field=field)
        if field == "output_parent" and not armed:
            armed = True
        return result

    def hostile_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            armed
            and isinstance(path, str)
            and path.endswith(".parquet")
            and dir_fd is not None
            and not swapped
        ):
            displaced = f".{path}.displaced"
            os.fchmod(dir_fd, 0o755)
            os.rename(
                path,
                displaced,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.symlink(displaced, path, dir_fd=dir_fd)
            os.fchmod(dir_fd, 0o555)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(subject, "_revalidate_pinned_directory", arm_after_parent_check)
    monkeypatch.setattr(subject.os, "open", hostile_open)

    with pytest.raises(ValueError, match="output_frozen_file_open_failed"):
        _prepare(phase_sources, output)

    assert swapped is True
    remaining_roots = list(phase_sources["parent"].glob(f".{output.name}.prepare-*"))
    assert len(remaining_roots) == 1
    assert any(path.is_symlink() for path in remaining_roots[0].rglob("*.parquet"))
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    ("internal_gap", "left_edge_gap", "cross_month_edge_gap"),
)
def test_rejects_any_raw_1s_gap_or_partition_edge_gap_without_publishing(
    phase_sources: dict[str, Path],
    mutation: str,
) -> None:
    if mutation == "cross_month_edge_gap":
        target = phase_sources["raw"] / "binance/BTCUSDT/2024-03.parquet"
        rejected = datetime(2024, 3, 1, tzinfo=UTC)
    else:
        target = phase_sources["raw"] / "binance/BTCUSDT/2024-02.parquet"
        rejected = datetime(2024, 2, 29, tzinfo=UTC)
        if mutation == "internal_gap":
            rejected += timedelta(seconds=10)
    frame = pl.read_parquet(target)
    frame.filter(pl.col("datetime") != rejected).write_parquet(target)
    output = phase_sources["parent"] / f"failed-{mutation}"

    with pytest.raises(ValueError, match=r"exact_1s|1s_coverage"):
        _prepare(phase_sources, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "source_timestamps,match",
    (
        ((0, 1, 8 * 60 * 60 * 1000), "canonical.*duplicate|timestamp_duplicate"),
        ((1001, 8 * 60 * 60 * 1000), "jitter_invalid"),
        ((-1, 8 * 60 * 60 * 1000), "partition_bounds"),
    ),
)
def test_funding_normalization_rejects_collision_jitter_and_out_of_range(
    source_timestamps: tuple[int, ...],
    match: str,
) -> None:
    day = datetime(1970, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "timestamp_ms": source_timestamps,
            "exchange": ["binance"] * len(source_timestamps),
            "symbol": ["BTCUSDT"] * len(source_timestamps),
            "funding_rate": [0.0001] * len(source_timestamps),
        }
    )

    with pytest.raises(ValueError, match=match):
        subject._prepare_feature_frame(
            frame,
            symbol="BTCUSDT",
            partition_start=day,
            partition_end=day + timedelta(days=1),
            owned_start=day,
            owned_end=day + timedelta(hours=16),
        )


def test_funding_normalization_accepts_slash_symbol_and_emits_compact_symbol() -> None:
    day = datetime(1970, 1, 1, tzinfo=UTC)
    interval_ms = 8 * 60 * 60 * 1000
    frame = pl.DataFrame(
        {
            "timestamp_ms": [0, interval_ms],
            "exchange": ["binance", "binance"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "funding_rate": [0.0001, 0.0002],
        }
    )

    normalized = subject._prepare_feature_frame(
        frame,
        symbol="BTCUSDT",
        partition_start=day,
        partition_end=day + timedelta(days=1),
        owned_start=day,
        owned_end=day + timedelta(hours=16),
    )

    assert normalized.get_column("symbol").to_list() == ["BTCUSDT", "BTCUSDT"]

    with pytest.raises(ValueError, match="feature_source_symbol_mismatch"):
        subject._prepare_feature_frame(
            frame.with_columns(pl.lit("ETH/USDT").alias("symbol")),
            symbol="BTCUSDT",
            partition_start=day,
            partition_end=day + timedelta(days=1),
            owned_start=day,
            owned_end=day + timedelta(hours=16),
        )


def test_funding_normalization_rejects_nonmonotonic_source_before_normalization() -> None:
    day = datetime(1970, 1, 1, tzinfo=UTC)
    interval_ms = 8 * 60 * 60 * 1000
    frame = pl.DataFrame(
        {
            "timestamp_ms": [interval_ms, 0],
            "funding_rate": [0.0002, 0.0001],
        }
    )

    with pytest.raises(ValueError, match="feature_source_timestamp_not_strictly_increasing"):
        subject._prepare_feature_frame(
            frame,
            symbol="BTCUSDT",
            partition_start=day,
            partition_end=day + timedelta(days=1),
            owned_start=day,
            owned_end=day + timedelta(hours=16),
        )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("missing", "ohlcv_schema_invalid"),
        ("nan", "ohlcv_value_invalid"),
        ("infinite", "ohlcv_value_invalid"),
        ("zero_price", "ohlc_nonpositive"),
        ("negative_price", "ohlc_nonpositive"),
        ("negative_volume", "volume_negative"),
        ("inconsistent", "ohlcv_relation_invalid"),
    ),
)
def test_raw_normalization_rejects_invalid_ohlcv(
    mutation: str,
    match: str,
) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "datetime": [start, start + timedelta(seconds=1)],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 100.5],
            "volume": [1.0, 1.0],
        }
    )
    if mutation == "missing":
        frame = frame.drop("open")
    elif mutation == "nan":
        frame = frame.with_columns(pl.lit(float("nan")).alias("open"))
    elif mutation == "infinite":
        frame = frame.with_columns(pl.lit(float("inf")).alias("high"))
    elif mutation == "zero_price":
        frame = frame.with_columns(pl.lit(0.0).alias("low"))
    elif mutation == "negative_price":
        frame = frame.with_columns(pl.lit(-1.0).alias("open"))
    elif mutation == "negative_volume":
        frame = frame.with_columns(pl.lit(-1.0).alias("volume"))
    else:
        frame = frame.with_columns(pl.lit(99.5).alias("high"))

    with pytest.raises(ValueError, match=match):
        subject._prepare_raw_frame(
            frame,
            symbol="BTCUSDT",
            partition_start=start,
            partition_end=start + timedelta(days=1),
            owned_start=start,
            owned_end=start + timedelta(seconds=2),
        )


@pytest.mark.parametrize("time_unit,offset", (("us", 500), ("ns", 500)))
def test_raw_normalization_rejects_subsecond_source_timestamps(
    time_unit: str,
    offset: int,
) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    units_per_second = {"us": 1_000_000, "ns": 1_000_000_000}[time_unit]
    epoch_units = int(start.timestamp()) * units_per_second
    frame = pl.DataFrame(
        {
            "datetime": pl.Series(
                "datetime",
                [epoch_units + offset, epoch_units + units_per_second + offset],
                dtype=pl.Datetime(time_unit, "UTC"),
            ),
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 100.5],
            "volume": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="timestamp_subsecond_invalid"):
        subject._prepare_raw_frame(
            frame,
            symbol="BTCUSDT",
            partition_start=start,
            partition_end=start + timedelta(days=1),
            owned_start=start,
            owned_end=start + timedelta(seconds=2),
        )


def test_source_root_swap_reads_only_pinned_descriptor_then_rejects_rebind(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = subject._read_parquet
    source = phase_sources["raw"]
    moved = phase_sources["parent"] / "raw-opened"
    swapped = False

    def hostile_read(relative_path: str, **kwargs):
        nonlocal swapped
        if not swapped and kwargs["field"] == "raw_partition":
            swapped = True
            source.rename(moved)
            source.mkdir()
        return original_read(relative_path, **kwargs)

    monkeypatch.setattr(subject, "_read_parquet", hostile_read)
    output = phase_sources["parent"] / "failed-source-rebind"

    with pytest.raises(ValueError, match="raw_source_root_path_changed"):
        _prepare(phase_sources, output)

    assert swapped is True
    assert not output.exists()
    assert _retained_temporary_root(phase_sources["parent"], output.name).is_dir()


def test_output_parent_swap_cannot_redirect_atomic_publish(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_parent = phase_sources["parent"] / "publish-parent"
    publish_parent.mkdir()
    moved_parent = phase_sources["parent"] / "publish-parent-opened"
    output = publish_parent / "phase-roots"
    original_publish = subject._publish
    swapped = False

    def hostile_publish(temporary_name: str, **kwargs):
        nonlocal swapped
        swapped = True
        publish_parent.rename(moved_parent)
        publish_parent.mkdir()
        return original_publish(temporary_name, **kwargs)

    monkeypatch.setattr(subject, "_publish", hostile_publish)

    with pytest.raises(ValueError, match="output_parent_path_changed"):
        _prepare(phase_sources, output)

    assert swapped is True
    assert not output.exists()
    assert not (moved_parent / output.name).exists()
    assert _retained_temporary_root(moved_parent, output.name).is_dir()


def test_output_parent_swap_after_rename_retains_descriptor_published_target(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_parent = phase_sources["parent"] / "post-rename-parent"
    publish_parent.mkdir()
    moved_parent = phase_sources["parent"] / "post-rename-parent-opened"
    output = publish_parent / "phase-roots"
    original_revalidate = subject._revalidate_pinned_directory
    output_parent_revalidations = 0

    def hostile_revalidate(capability, *, field: str):
        nonlocal output_parent_revalidations
        if field == "output_parent":
            output_parent_revalidations += 1
            if output_parent_revalidations == 2:
                publish_parent.rename(moved_parent)
                publish_parent.mkdir()
        return original_revalidate(capability, field=field)

    monkeypatch.setattr(subject, "_revalidate_pinned_directory", hostile_revalidate)

    with pytest.raises(ValueError, match="output_parent_path_changed"):
        _prepare(phase_sources, output)

    assert output_parent_revalidations == 2
    assert not output.exists()
    assert (moved_parent / output.name).is_dir()
    assert not list(publish_parent.glob(".phase-roots.prepare-*"))
    assert not list(moved_parent.glob(".phase-roots.prepare-*"))


def test_target_replacement_after_rename_never_returns_success_or_deletes_attacker(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_parent = phase_sources["parent"] / "target-replacement-parent"
    publish_parent.mkdir()
    output = publish_parent / "phase-roots"
    moved_output = publish_parent / "moved-phase-roots"
    original_revalidate = subject._revalidate_pinned_directory
    output_parent_revalidations = 0
    marker = b"attacker-owned\n"

    def hostile_revalidate(capability, *, field: str):
        nonlocal output_parent_revalidations
        if field == "output_parent":
            output_parent_revalidations += 1
            if output_parent_revalidations == 2:
                output.rename(moved_output)
                output.mkdir()
                (output / "marker").write_bytes(marker)
        return original_revalidate(capability, field=field)

    monkeypatch.setattr(subject, "_revalidate_pinned_directory", hostile_revalidate)

    with pytest.raises(ValueError, match="published_identity_mismatch"):
        _prepare(phase_sources, output)

    assert output_parent_revalidations == 2
    assert (output / "marker").read_bytes() == marker
    assert moved_output.is_dir()
    assert not list(publish_parent.glob(".phase-roots.prepare-*"))


def test_target_creation_race_before_rename_preserves_attacker_target(
    phase_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_parent = phase_sources["parent"] / "target-creation-race-parent"
    publish_parent.mkdir()
    output = publish_parent / "phase-roots"
    original_target_is_absent = subject._target_is_absent
    marker = b"attacker-created-before-rename\n"
    attacker_identity: tuple[int, int] | None = None
    created = False

    def hostile_target_is_absent(parent, target_name: str) -> bool:
        nonlocal attacker_identity, created
        absent = original_target_is_absent(parent, target_name)
        parent_path = Path(f"/proc/self/fd/{parent.descriptor}")
        if absent and not created and list(parent_path.glob(f".{target_name}.prepare-*")):
            output.mkdir()
            marker_path = output / "marker"
            marker_path.write_bytes(marker)
            marker_path.chmod(0o444)
            output.chmod(0o555)
            observed = output.lstat()
            attacker_identity = (observed.st_dev, observed.st_ino)
            created = True
        return absent

    monkeypatch.setattr(subject, "_target_is_absent", hostile_target_is_absent)

    with pytest.raises(ValueError, match="output_root_must_remain_absent"):
        _prepare(phase_sources, output)

    assert created is True
    assert attacker_identity is not None
    observed = output.lstat()
    assert (observed.st_dev, observed.st_ino) == attacker_identity
    assert stat.S_IMODE(observed.st_mode) == 0o555
    assert (output / "marker").read_bytes() == marker
    assert _retained_temporary_root(publish_parent, output.name).is_dir()
