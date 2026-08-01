from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE = Path(__file__).parents[1] / "scripts/research/acquire_alpha_max_official_source.py"
_SPEC = importlib.util.spec_from_file_location("official_acquirer", _MODULE)
assert _SPEC and _SPEC.loader
subject = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = subject
_SPEC.loader.exec_module(subject)


def archive(
    tmp_path: Path,
    rows: list[list[str]],
    header: str | None = None,
    member: str = "BTCUSDT-aggTrades-2024-01.csv",
    extra_members: tuple[str, ...] = (),
    mode: int = stat.S_IFREG | 0o600,
    content: str | None = None,
) -> Path:
    path = tmp_path / "trades.zip"
    with zipfile.ZipFile(path, "w") as output:
        body = (
            content
            if content is not None
            else "\n".join(([] if header is None else [header]) + [",".join(row) for row in rows])
        )
        info = zipfile.ZipInfo(member)
        info.external_attr = mode << 16
        output.writestr(info, body)
        for extra_member in extra_members:
            extra = zipfile.ZipInfo(extra_member)
            extra.external_attr = (stat.S_IFREG | 0o600) << 16
            output.writestr(extra, "")
    return path


def may_2023_archive(tmp_path: Path, rows: list[list[str]], name: str = "may-2023.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as output:
        info = zipfile.ZipInfo("BTCUSDT-aggTrades-2023-05.csv")
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        output.writestr(info, "\n".join(",".join(row) for row in rows))
    return path


def parquet_bytes(frame: subject.pl.DataFrame) -> bytes:
    destination = io.BytesIO()
    frame.write_parquet(destination)
    return destination.getvalue()


def aggtrades_scratch_entries(root: Path) -> list[Path]:
    scratch = subject.scratch_path(root)
    return sorted(scratch.glob(".aggtrades-*")) if scratch.exists() else []


MAY_2023_START_MS = 1_682_899_200_000


def allow_may_2023_order(monkeypatch: pytest.MonkeyPatch, source: Path) -> tuple[str, int]:
    identity = (subject.file_sha256(source), source.stat().st_size)
    monkeypatch.setitem(subject.CANONICAL_ORDER_ARCHIVES, ("BTCUSDT", "2023-05"), identity)
    return identity


SOL_NOVEMBER_2023_START_MS = 1_698_796_800_000


def sol_november_2023_archive(tmp_path: Path, rows: list[list[str]]) -> Path:
    return archive(
        tmp_path,
        rows,
        member="SOLUSDT-aggTrades-2023-11.csv",
    )


def allow_sol_november_2023_order(monkeypatch: pytest.MonkeyPatch, source: Path) -> tuple[str, int]:
    identity = (subject.file_sha256(source), source.stat().st_size)
    monkeypatch.setitem(subject.CANONICAL_ORDER_ARCHIVES, ("SOLUSDT", "2023-11"), identity)
    return identity


SOL_DECEMBER_2023_START_MS = 1_701_388_800_000


def sol_december_2023_archive(tmp_path: Path, rows: list[list[str]]) -> Path:
    return archive(
        tmp_path,
        rows,
        member="SOLUSDT-aggTrades-2023-12.csv",
    )


def allow_sol_december_2023_order(monkeypatch: pytest.MonkeyPatch, source: Path) -> tuple[str, int]:
    identity = (subject.file_sha256(source), source.stat().st_size)
    monkeypatch.setitem(subject.CANONICAL_ORDER_ARCHIVES, ("SOLUSDT", "2023-12"), identity)
    return identity


SOL_JULY_2025_START_MS = 1_751_328_000_000


def sol_july_2025_archive(tmp_path: Path, rows: list[list[str]]) -> Path:
    return archive(
        tmp_path,
        rows,
        member="SOLUSDT-aggTrades-2025-07.csv",
    )


def allow_sol_july_2025_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    duplicate_ids: frozenset[int] = frozenset({2}),
) -> tuple[str, int, frozenset[int]]:
    identity = (subject.file_sha256(source), source.stat().st_size, duplicate_ids)
    monkeypatch.setitem(
        subject.AUTHENTICATED_DUPLICATE_AGGREGATE_ARCHIVES,
        ("SOLUSDT", "2025-07"),
        identity,
    )
    return identity


def authenticated_archive_receipt(source: Path) -> dict[str, int | str]:
    return {"sha256": subject.file_sha256(source), "byte_count": source.stat().st_size}


class _Height:
    def __init__(self, physical: int, aggregate: int) -> None:
        self.physical = physical
        self.aggregate = aggregate

    def __eq__(self, other: object) -> bool:
        return self.physical == other

    def __radd__(self, other: object) -> int:
        return self.aggregate


class _RawFrame:
    def __init__(self, frame: subject.pl.DataFrame, aggregate_rows: int) -> None:
        self.frame = frame
        self.aggregate_rows = aggregate_rows

    @property
    def height(self) -> _Height:
        return _Height(self.frame.height, self.aggregate_rows)

    def equals(self, other: object) -> bool:
        return isinstance(other, _RawFrame) and self.frame.equals(other.frame)

    def __getitem__(self, key: str) -> object:
        return self.frame[key]

    def __getattr__(self, name: str) -> object:
        return getattr(self.frame, name)


def _tree_snapshot(root: Path) -> list[tuple[str, str, str | None, int]]:
    snapshot = []
    for path in sorted(root.rglob("*")):
        item = path.lstat()
        relative = str(path.relative_to(root))
        if path.is_dir():
            snapshot.append((relative, "directory", None, item.st_mode & 0o777))
        else:
            snapshot.append(
                (
                    relative,
                    "file",
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    item.st_mode & 0o777,
                )
            )
    return snapshot


def _lstat_tree_snapshot(root: Path) -> list[tuple[str, int, int, int, str | None]]:
    snapshot = []
    for base, directories, filenames in os.walk(root, followlinks=False):
        for name in sorted([*directories, *filenames]):
            path = Path(base) / name
            item = path.lstat()
            relative = str(path.relative_to(root))
            if stat.S_ISLNK(item.st_mode):
                snapshot.append(
                    (relative, item.st_mode, item.st_size, item.st_nlink, os.readlink(path))
                )
            elif stat.S_ISREG(item.st_mode):
                snapshot.append(
                    (
                        relative,
                        item.st_mode,
                        item.st_size,
                        item.st_nlink,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
            else:
                snapshot.append((relative, item.st_mode, item.st_size, item.st_nlink, None))
    return snapshot


def _write_request_fixture(
    path: Path, url: str, data: bytes, scratch_root: Path | None = None
) -> None:
    if not path.exists():
        subject.atomic_write(path, data, scratch_root=scratch_root)
    requested, query = subject._request_fields(url, None)
    subject.immutable_json(
        path.with_suffix(path.suffix + ".receipt.json"),
        {
            "schema": "official_request_receipt.v1",
            "requested_url": requested,
            "final_url": requested,
            "final_host": subject.urllib.parse.urlsplit(requested).hostname,
            "query": query,
            "retrieved_at_utc": "2024-01-01T00:00:00Z",
            "byte_count": len(data),
            "sha256": subject.sha256(data),
        },
        scratch_root,
    )


def _eligible_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path, Path]:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    contract_path, evidence_path = tmp_path / "contract.json", tmp_path / "evidence.json"
    contract_path.write_bytes(b"contract")
    evidence_path.write_bytes(b"evidence")
    monkeypatch.setattr(subject, "CONTRACT_SHA256", subject.sha256(b"contract"))
    monkeypatch.setattr(subject, "EVIDENCE_SHA256", subject.sha256(b"evidence"))
    monkeypatch.setattr(subject, "assert_roots", lambda *args: None)
    contract = subject.Contract(
        "BTCUSDT", 1704067200000, 1704067201000, 1704067200000, 1704067200001
    )
    monkeypatch.setattr(subject, "load_contract", lambda _: [contract])
    monkeypatch.setattr(subject, "load_evidence", lambda _: {})
    raw_relative = "market_ohlcv_1s/binance/BTCUSDT/2024-01.parquet"
    archive_path = report / "provenance/archives/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive_file:
        info = zipfile.ZipInfo("BTCUSDT-aggTrades-2024-01.csv")
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        archive_file.writestr(info, "1,1,1,1,1,1704067200000,False")
    archive_data = archive_path.read_bytes()
    archive_url = f"{subject.ARCHIVE_BASE}/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip"
    _write_request_fixture(archive_path, archive_url, archive_data, report)
    checksum_path = archive_path.with_name("BTCUSDT-aggTrades-2024-01.zip.CHECKSUM")
    _write_request_fixture(
        checksum_path,
        archive_url + ".CHECKSUM",
        f"{subject.sha256(archive_data)} BTCUSDT-aggTrades-2024-01.zip\n".encode(),
        report,
    )
    exchange = report / "provenance/exchangeInfo.json"
    _write_request_fixture(exchange, f"{subject.API_BASE}/exchangeInfo", b'{"symbols":[]}', report)
    subject.atomic_write(
        report / "provenance/contract_manifest.json", b"contract", scratch_root=report
    )
    subject.atomic_write(
        report / "provenance/availability_evidence.json", b"evidence", scratch_root=report
    )
    frame, _ = subject.frame_from_archive(
        archive_path,
        contract.symbol,
        contract.raw_start_ms,
        contract.raw_end_ms,
        None,
        "2024-01",
        report,
    )
    target = output / raw_relative
    target.parent.mkdir(parents=True)
    frame.write_parquet(target)
    raw_frame = _RawFrame(frame, 1_066_681_730)
    funding_relative = (
        "feature_points/exchange=binance/symbol=BTCUSDT/date=2024-01-01/funding.parquet"
    )
    funding_rows = [
        {
            "timestamp_ms": 1704067200000,
            "source_timestamp_ms": 1704067200000,
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "funding_rate": 0.1,
        }
    ]
    funding_frame = (
        subject.pl.DataFrame(funding_rows)
        .select(subject.FUNDING_COLUMNS)
        .with_columns(
            subject.pl.col("timestamp_ms").cast(subject.pl.Int64),
            subject.pl.col("source_timestamp_ms").cast(subject.pl.Int64),
            subject.pl.col("funding_rate").cast(subject.pl.Float64),
        )
    )
    funding_target = output / funding_relative
    funding_target.parent.mkdir(parents=True)
    funding_frame.write_parquet(funding_target)
    funding_proxy = _RawFrame(funding_frame, 39_569)
    page_path = report / "provenance/funding_pages/BTCUSDT/000001.json"
    page = [{"symbol": "BTCUSDT", "fundingTime": 1704067200000, "fundingRate": "0.1"}]
    page_data = subject.canonical_bytes(page)
    _write_request_fixture(
        page_path,
        f"{subject.API_BASE}/fundingRate?symbol=BTCUSDT&startTime=1704067200000&endTime=1704067200000&limit=1000",
        page_data,
        report,
    )
    page_hash = subject.sha256(page_data)
    monkeypatch.setattr(subject, "frame_from_archive", lambda *args: (raw_frame, {}))

    def read_fixture_frame(source: object):
        target_name = Path(os.readlink(f"/proc/self/fd/{source.fileno()}"))
        return raw_frame if target_name == target else funding_proxy

    monkeypatch.setattr(subject.pl, "read_parquet", read_fixture_frame)
    monkeypatch.setattr(
        subject,
        "funding_pages_from_provenance",
        lambda *args: (page, [page_hash], {page_path, page_path.with_suffix(".json.receipt.json")}),
    )
    receipt = {
        "schema": "alpha_max_partition_receipt.v2",
        "path": raw_relative,
        "source_sha256": subject.sha256(archive_data),
        "output_sha256": subject.file_sha256(target),
        "rows": 1,
        "start_ms": contract.raw_start_ms,
        "end_ms": contract.raw_end_ms,
        "input_carry_close": None,
        "output_carry_close": 1.0,
        "derivation_version": subject.DERIVATION_VERSION,
        "code_sha256": subject.code_hash(),
        "page_hashes": [
            subject.file_sha256(checksum_path),
            subject.file_sha256(archive_path),
        ],
    }
    subject.immutable_json(subject.partition_path(report, raw_relative), receipt, report)
    evidence_paths = subject.archive_evidence_paths(report, contract.symbol, "2024-01")
    archive_receipt = subject.request_receipt(archive_path, archive_url)
    derivation = subject.expected_archive_derivation(
        archive_path,
        archive_path.with_suffix(".zip.receipt.json"),
        archive_url,
        target,
        receipt,
        None,
        archive_receipt,
    )
    subject.immutable_json(evidence_paths["derivation"], derivation, report)
    intent = subject.expected_archive_intent(archive_path, derivation)
    subject.immutable_json(evidence_paths["retirement-intent"], intent, report)
    archive_path.unlink()
    subject.immutable_json(
        evidence_paths["deletion"],
        {
            "schema": "alpha_max_archive_deletion_receipt.v1",
            "retirement_intent_sha256": subject.sha256(subject.canonical_bytes(intent)),
            "derivation_receipt_sha256": subject.sha256(subject.canonical_bytes(derivation)),
            "archive_relative_path": str(archive_path.relative_to(report)),
            "archive_sha256": archive_receipt["sha256"],
            "archive_byte_count": archive_receipt["byte_count"],
            "archive_absent": True,
        },
        report,
    )
    funding_receipt = {
        "schema": "alpha_max_partition_receipt.v2",
        "path": funding_relative,
        "source_sha256": subject.sha256(subject.canonical_bytes(funding_rows)),
        "output_sha256": subject.file_sha256(funding_target),
        "rows": 1,
        "start_ms": 1704067200000,
        "end_ms": 1704153600000,
        "input_carry_close": None,
        "output_carry_close": None,
        "derivation_version": subject.DERIVATION_VERSION,
        "code_sha256": subject.code_hash(),
        "page_hashes": [page_hash],
    }
    subject.immutable_json(
        subject.partition_path(report, funding_relative), funding_receipt, report
    )
    subject.atomic_write(
        report / "acquisition.journal.jsonl", b'{"event":"complete"}\n', scratch_root=report
    )
    plan = subject.full_plan()
    plan_data = subject.canonical_bytes(plan)
    subject.atomic_write(report / "plan.json", plan_data, scratch_root=report)
    subject.ownership(output, report, subject.sha256(plan_data))
    subject.rebuild_manifest(output, report)
    eligible_receipt = subject.validate_complete(output, [contract], report / "provenance", report)
    eligible_receipt["source_manifest_sha256"] = subject.file_sha256(
        report / "source_manifest.json"
    )
    eligible_receipt["acquisition_journal_sha256"] = subject.file_sha256(
        report / "acquisition.journal.jsonl"
    )
    subject.immutable_json(report / "source_eligible_receipt.json", eligible_receipt, report)
    return (
        [
            "--contract-manifest",
            str(contract_path),
            "--availability-evidence",
            str(evidence_path),
            "--output-root",
            str(output),
            "--report-dir",
            str(report),
            "--verify-eligible",
        ],
        output,
        report,
    )


def test_production_parser_streams_without_filesystem_scratch(tmp_path: Path) -> None:
    path = archive(
        tmp_path,
        [
            ["1", "10", "2", "900", "901", "1704067200000", "FALSE"],
            ["2", "11", "3", "10", "11", "1704067200999", "True"],
            ["3", "12", "4", "1", "2", "1704067202000", "false"],
        ],
    )
    before = _tree_snapshot(tmp_path)
    frame, facts = subject.frame_from_archive(
        path, "BTCUSDT", 1704067200000, 1704067203000, None, "2024-01"
    )
    assert frame["open"].to_list() == [10.0, 11.0, 12.0]
    assert frame["close"].to_list() == [11.0, 11.0, 12.0]
    assert frame["volume"].to_list() == [5.0, 0.0, 4.0]
    assert facts["first_trade"] == (1704067200000, 1)
    assert _tree_snapshot(tmp_path) == before


def test_allowlisted_may_2023_interleaving_matches_ordered_frame_and_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ordered_rows = [
        ["1", "10", "2", "1", "1", str(MAY_2023_START_MS), "FALSE"],
        ["2", "11", "3", "2", "2", str(MAY_2023_START_MS + 100), "TRUE"],
        ["3", "12", "4", "3", "3", str(MAY_2023_START_MS + 1000), "FALSE"],
    ]
    interleaved = may_2023_archive(tmp_path, [ordered_rows[1], ordered_rows[0], ordered_rows[2]])
    ordered = may_2023_archive(tmp_path, ordered_rows, "ordered.zip")
    digest, byte_count = allow_may_2023_order(monkeypatch, interleaved)
    assert (digest, byte_count) == (subject.file_sha256(interleaved), interleaved.stat().st_size)
    receipt = authenticated_archive_receipt(interleaved)
    original_stable_file = subject.stable_file
    archive_opens = 0

    def count_archive_open(path: Path):
        nonlocal archive_opens
        if path == interleaved:
            archive_opens += 1
        return original_stable_file(path)

    monkeypatch.setattr(subject, "stable_file", count_archive_open)
    frame, facts = subject.frame_from_archive(
        interleaved,
        "BTCUSDT",
        MAY_2023_START_MS,
        MAY_2023_START_MS + 2000,
        None,
        "2023-05",
        scratch,
        receipt,
    )
    reference, reference_facts = subject.frame_from_archive(
        ordered,
        "BTCUSDT",
        MAY_2023_START_MS,
        MAY_2023_START_MS + 2000,
        None,
        "2023-05",
        scratch,
    )
    assert frame.equals(reference)
    assert parquet_bytes(frame) == parquet_bytes(reference)
    assert facts == reference_facts
    assert aggtrades_scratch_entries(scratch) == []
    assert archive_opens == 1


def test_sol_november_2023_production_identity_is_exact() -> None:
    assert subject.CANONICAL_ORDER_ARCHIVES[("SOLUSDT", "2023-11")] == (
        "188c3145ecaab1cf546318c293fb4fef0e320a6dc05b14eea013a46209ebbd73",
        535864305,
    )


def test_authenticated_sol_november_2023_identity_canonicalizes_interleaving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = sol_november_2023_archive(
        tmp_path,
        [
            ["2", "11", "2", "2", "2", str(SOL_NOVEMBER_2023_START_MS + 100), "TRUE"],
            ["1", "10", "1", "1", "1", str(SOL_NOVEMBER_2023_START_MS), "FALSE"],
            ["3", "12", "3", "3", "3", str(SOL_NOVEMBER_2023_START_MS + 1000), "FALSE"],
        ],
    )
    allow_sol_november_2023_order(monkeypatch, source)
    frame, _ = subject.frame_from_archive(
        source,
        "SOLUSDT",
        SOL_NOVEMBER_2023_START_MS,
        SOL_NOVEMBER_2023_START_MS + 2000,
        None,
        "2023-11",
        scratch,
        authenticated_archive_receipt(source),
    )
    assert frame["open"].to_list() == [10.0, 12.0]
    assert frame["high"].to_list() == [11.0, 12.0]
    assert frame["low"].to_list() == [10.0, 12.0]
    assert frame["close"].to_list() == [11.0, 12.0]
    assert frame["volume"].to_list() == [3.0, 3.0]


def test_sol_december_2023_production_identity_is_exact() -> None:
    assert subject.CANONICAL_ORDER_ARCHIVES[("SOLUSDT", "2023-12")] == (
        "c12bc6707c8fb6ab5f3fe712ad6c8b816053a27c6c6ead4f7dc98df7a098b70c",
        558456557,
    )


def test_authenticated_sol_december_2023_identity_canonicalizes_interleaving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = sol_december_2023_archive(
        tmp_path,
        [
            ["2", "11", "2", "2", "2", str(SOL_DECEMBER_2023_START_MS + 100), "TRUE"],
            ["1", "10", "1", "1", "1", str(SOL_DECEMBER_2023_START_MS), "FALSE"],
            ["3", "12", "3", "3", "3", str(SOL_DECEMBER_2023_START_MS + 1000), "FALSE"],
        ],
    )
    allow_sol_december_2023_order(monkeypatch, source)
    frame, _ = subject.frame_from_archive(
        source,
        "SOLUSDT",
        SOL_DECEMBER_2023_START_MS,
        SOL_DECEMBER_2023_START_MS + 2000,
        None,
        "2023-12",
        scratch,
        authenticated_archive_receipt(source),
    )
    assert frame["open"].to_list() == [10.0, 12.0]
    assert frame["high"].to_list() == [11.0, 12.0]
    assert frame["low"].to_list() == [10.0, 12.0]
    assert frame["close"].to_list() == [11.0, 12.0]
    assert frame["volume"].to_list() == [3.0, 3.0]


def test_sol_july_2025_production_identity_and_duplicate_are_exact() -> None:
    assert subject.AUTHENTICATED_DUPLICATE_AGGREGATE_ARCHIVES[("SOLUSDT", "2025-07")] == (
        "07842c476aab159f008ffc4e95e421e181f75348610c23baeae1dc3799d4e89b",
        208568498,
        frozenset({926014272}),
    )


def test_authenticated_sol_july_2025_preserves_pinned_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = sol_july_2025_archive(
        tmp_path,
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 100), "TRUE"],
            ["2", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 200), "FALSE"],
            ["3", "13", "4", "4", "4", str(SOL_JULY_2025_START_MS + 1000), "TRUE"],
        ],
    )
    allow_sol_july_2025_duplicate(monkeypatch, source)

    frame, facts = subject.frame_from_archive(
        source,
        "SOLUSDT",
        SOL_JULY_2025_START_MS,
        SOL_JULY_2025_START_MS + 2000,
        None,
        "2025-07",
        archive_receipt=authenticated_archive_receipt(source),
    )

    assert frame["open"].to_list() == [10.0, 13.0]
    assert frame["high"].to_list() == [12.0, 13.0]
    assert frame["low"].to_list() == [10.0, 13.0]
    assert frame["close"].to_list() == [12.0, 13.0]
    assert frame["volume"].to_list() == [6.0, 4.0]
    assert facts["trade_count"] == 4


@pytest.mark.parametrize(
    ("symbol", "month", "receipt"),
    [
        ("SOLUSDT", "2025-07", None),
        ("SOLUSDT", "2025-07", {"sha256": "0" * 64, "byte_count": 0}),
        ("BTCUSDT", "2025-07", None),
        ("SOLUSDT", "2025-06", None),
    ],
)
def test_sol_july_2025_unscoped_duplicate_stays_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    month: str,
    receipt: dict[str, int | str] | None,
) -> None:
    source = sol_july_2025_archive(
        tmp_path,
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 100), "TRUE"],
            ["2", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 200), "FALSE"],
        ],
    )
    allow_sol_july_2025_duplicate(monkeypatch, source)

    with pytest.raises(subject.AcquisitionError):
        subject.frame_from_archive(
            source,
            symbol,
            SOL_JULY_2025_START_MS,
            SOL_JULY_2025_START_MS + 1000,
            None,
            month,
            archive_receipt=receipt,
        )


def test_sol_july_2025_self_authenticated_wrong_production_identity_stays_strict(
    tmp_path: Path,
) -> None:
    source = sol_july_2025_archive(
        tmp_path,
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 100), "TRUE"],
            ["2", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 200), "FALSE"],
        ],
    )

    with pytest.raises(subject.AcquisitionError, match=r"^archive_trade_order_invalid$"):
        subject.frame_from_archive(
            source,
            "SOLUSDT",
            SOL_JULY_2025_START_MS,
            SOL_JULY_2025_START_MS + 1000,
            None,
            "2025-07",
            archive_receipt=authenticated_archive_receipt(source),
        )


@pytest.mark.parametrize(
    "rows",
    [
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 100), "TRUE"],
            ["2", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 100), "FALSE"],
        ],
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 100), "TRUE"],
            ["2", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 200), "FALSE"],
            ["2", "13", "4", "4", "4", str(SOL_JULY_2025_START_MS + 300), "TRUE"],
        ],
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 100), "TRUE"],
            ["3", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 200), "FALSE"],
        ],
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 200), "TRUE"],
            ["2", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 100), "FALSE"],
        ],
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 100), "TRUE"],
            ["2", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 200), "FALSE"],
            ["3", "13", "4", "4", "4", str(SOL_JULY_2025_START_MS + 300), "TRUE"],
            ["3", "14", "5", "5", "5", str(SOL_JULY_2025_START_MS + 400), "FALSE"],
        ],
        [
            ["1", "10", "1", "1", "1", str(SOL_JULY_2025_START_MS), "FALSE"],
            ["2", "11", "2", "2", "2", str(SOL_JULY_2025_START_MS + 100), "TRUE"],
            ["2", "12", "3", "3", "3", str(SOL_JULY_2025_START_MS + 200), "FALSE"],
            ["1", "13", "4", "4", "4", str(SOL_JULY_2025_START_MS + 300), "TRUE"],
        ],
    ],
)
def test_sol_july_2025_pin_rejects_unmatched_duplicate_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[list[str]]
) -> None:
    source = sol_july_2025_archive(tmp_path, rows)
    allow_sol_july_2025_duplicate(monkeypatch, source)

    with pytest.raises(subject.AcquisitionError, match=r"^archive_trade_order_invalid$"):
        subject.frame_from_archive(
            source,
            "SOLUSDT",
            SOL_JULY_2025_START_MS,
            SOL_JULY_2025_START_MS + 1000,
            None,
            "2025-07",
            archive_receipt=authenticated_archive_receipt(source),
        )


@pytest.mark.parametrize(
    ("symbol", "month", "receipt", "identity"),
    [
        ("SOLUSDT", "2023-11", None, None),
        ("SOLUSDT", "2023-11", {"sha256": "0" * 64, "byte_count": 0}, None),
        ("SOLUSDT", "2023-11", {"sha256": "0" * 64}, None),
        ("BTCUSDT", "2023-11", None, None),
        ("SOLUSDT", "2023-10", None, None),
        ("SOLUSDT", "2023-11", None, ("0" * 64, 0)),
    ],
)
def test_sol_november_2023_unauthenticated_or_wrong_scope_cannot_canonicalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    month: str,
    receipt: dict[str, int | str] | None,
    identity: tuple[str, int] | None,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = sol_november_2023_archive(
        tmp_path,
        [
            ["2", "11", "2", "2", "2", str(SOL_NOVEMBER_2023_START_MS + 1), "FALSE"],
            ["1", "10", "1", "1", "1", str(SOL_NOVEMBER_2023_START_MS), "FALSE"],
        ],
    )
    allow_sol_november_2023_order(monkeypatch, source)
    if identity is not None:
        monkeypatch.setitem(subject.CANONICAL_ORDER_ARCHIVES, ("SOLUSDT", "2023-11"), identity)
    with pytest.raises(subject.AcquisitionError):
        subject.frame_from_archive(
            source,
            symbol,
            SOL_NOVEMBER_2023_START_MS,
            SOL_NOVEMBER_2023_START_MS + 1000,
            None,
            month,
            scratch,
            receipt,
        )


def test_sol_november_2023_wrong_byte_count_receipt_cannot_canonicalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = sol_november_2023_archive(
        tmp_path,
        [
            ["2", "11", "11", "11", "2", str(SOL_NOVEMBER_2023_START_MS + 1), "FALSE"],
            ["1", "10", "10", "10", "1", str(SOL_NOVEMBER_2023_START_MS), "FALSE"],
        ],
    )
    allow_sol_november_2023_order(monkeypatch, source)
    receipt = authenticated_archive_receipt(source)
    receipt["byte_count"] += 1
    with pytest.raises(subject.AcquisitionError, match=r"^archive_receipt_invalid$"):
        subject.frame_from_archive(
            source,
            "SOLUSDT",
            SOL_NOVEMBER_2023_START_MS,
            SOL_NOVEMBER_2023_START_MS + 1000,
            None,
            "2023-11",
            scratch,
            receipt,
        )


@pytest.mark.parametrize(
    "rows",
    [
        [
            ["2", "11", "2", "2", "2", str(SOL_NOVEMBER_2023_START_MS + 1), "FALSE"],
            ["1", "10", "1", "1", "1", str(SOL_NOVEMBER_2023_START_MS), "FALSE"],
            ["1", "12", "3", "3", "3", str(SOL_NOVEMBER_2023_START_MS + 2), "FALSE"],
        ],
        [
            ["2", "11", "2", "2", "2", str(SOL_NOVEMBER_2023_START_MS), "FALSE"],
            ["1", "10", "1", "1", "1", str(SOL_NOVEMBER_2023_START_MS + 1), "FALSE"],
        ],
    ],
)
def test_sol_november_2023_canonicalization_rejects_duplicate_ids_and_timestamp_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[list[str]]
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = sol_november_2023_archive(tmp_path, rows)
    allow_sol_november_2023_order(monkeypatch, source)
    with pytest.raises(subject.AcquisitionError, match=r"^archive_trade_order_invalid$"):
        subject.frame_from_archive(
            source,
            "SOLUSDT",
            SOL_NOVEMBER_2023_START_MS,
            SOL_NOVEMBER_2023_START_MS + 1000,
            None,
            "2023-11",
            scratch,
            authenticated_archive_receipt(source),
        )


def test_standard_archive_parsing_never_uses_path_replacement_after_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = archive(
        tmp_path,
        [["1", "99", "1", "1", "1", "1704067200000", "False"]],
    ).rename(tmp_path / "replacement.zip")
    source = archive(tmp_path, [["1", "10", "1", "1", "1", "1704067200000", "False"]])
    receipt = authenticated_archive_receipt(source)
    backup = tmp_path / "authenticated.zip"
    original_identity = subject.opened_file_identity

    def replace_after_hash(handle: object) -> tuple[str, int]:
        identity = original_identity(handle)
        os.replace(source, backup)
        os.replace(replacement, source)
        return identity

    monkeypatch.setattr(subject, "opened_file_identity", replace_after_hash)
    try:
        frame, _ = subject.frame_from_archive(
            source, "BTCUSDT", 1704067200000, 1704067201000, None, "2024-01", None, receipt
        )
    except subject.AcquisitionError as exc:
        assert str(exc) == "unsafe_file"
    else:
        assert frame["open"].to_list() == [10.0]
    assert subject.file_sha256(backup) == receipt["sha256"]
    assert subject.file_sha256(source) != receipt["sha256"]


def test_invalid_supplied_archive_receipt_fails_closed(tmp_path: Path) -> None:
    source = archive(tmp_path, [["1", "1", "1", "1", "1", "1704067200000", "False"]])
    with pytest.raises(subject.AcquisitionError, match=r"^archive_receipt_invalid$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-01",
            None,
            {"sha256": "0" * 64, "byte_count": 0},
        )


def test_may_2023_interleaving_without_exact_identity_rejects_without_scratch(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = may_2023_archive(
        tmp_path,
        [
            ["2", "11", "1", "2", "2", str(MAY_2023_START_MS + 1), "FALSE"],
            ["1", "10", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"],
        ],
    )
    before = _tree_snapshot(tmp_path)
    with pytest.raises(subject.AcquisitionError, match=r"^archive_trade_order_invalid$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            scratch,
            authenticated_archive_receipt(source),
        )
    assert _tree_snapshot(tmp_path) == before


def test_allowlisted_identity_without_receipt_stays_strict_without_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = may_2023_archive(
        tmp_path,
        [
            ["2", "11", "1", "2", "2", str(MAY_2023_START_MS + 1), "FALSE"],
            ["1", "10", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"],
        ],
    )
    allow_may_2023_order(monkeypatch, source)
    before = _tree_snapshot(tmp_path)
    with pytest.raises(subject.AcquisitionError, match=r"^archive_trade_order_invalid$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
        )
    assert _tree_snapshot(tmp_path) == before


def test_allowlisted_identity_with_receipt_requires_scratch_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = may_2023_archive(
        tmp_path,
        [["1", "10", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"]],
    )
    allow_may_2023_order(monkeypatch, source)
    with pytest.raises(subject.AcquisitionError, match=r"^archive_order_scratch_root_required$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            archive_receipt=authenticated_archive_receipt(source),
        )


def test_ordered_archive_with_explicit_scratch_root_stays_on_zero_scratch_streaming_path(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subject.scratch_directory(scratch)
    source = may_2023_archive(
        tmp_path,
        [
            ["1", "10", "2", "1", "1", str(MAY_2023_START_MS), "FALSE"],
            ["2", "11", "3", "2", "2", str(MAY_2023_START_MS + 1), "FALSE"],
        ],
    )
    before = _tree_snapshot(tmp_path)
    frame, _ = subject.frame_from_archive(
        source,
        "BTCUSDT",
        MAY_2023_START_MS,
        MAY_2023_START_MS + 1000,
        None,
        "2023-05",
        scratch,
        authenticated_archive_receipt(source),
    )
    assert frame["open"].to_list() == [10.0]
    assert frame["close"].to_list() == [11.0]
    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "rows",
    [
        [
            ["2", "11", "1", "2", "2", str(MAY_2023_START_MS + 1), "FALSE"],
            ["1", "10", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"],
            ["1", "12", "1", "3", "3", str(MAY_2023_START_MS + 2), "FALSE"],
        ],
        [
            ["2", "11", "1", "2", "2", str(MAY_2023_START_MS), "FALSE"],
            ["1", "10", "1", "1", "1", str(MAY_2023_START_MS + 1), "FALSE"],
        ],
    ],
)
def test_allowlisted_may_2023_rejects_duplicate_ids_and_post_sort_timestamp_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[list[str]]
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subject.scratch_directory(scratch)
    source = may_2023_archive(tmp_path, rows)
    allow_may_2023_order(monkeypatch, source)
    before = _tree_snapshot(tmp_path)
    with pytest.raises(subject.AcquisitionError, match=r"^archive_trade_order_invalid$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            scratch,
            authenticated_archive_receipt(source),
        )
    assert _tree_snapshot(tmp_path) == before


def test_allowlisted_external_merge_uses_packed_bounded_chunks_and_fan_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    rows = [
        [
            str(index),
            str(index + 10),
            "1",
            str(index),
            str(index),
            str(MAY_2023_START_MS + index),
            "FALSE",
        ]
        for index in (5, 4, 3, 2, 1)
    ]
    source = may_2023_archive(tmp_path, rows)
    allow_may_2023_order(monkeypatch, source)
    monkeypatch.setattr(subject, "ORDER_CHUNK_RECORDS", 2)
    monkeypatch.setattr(subject, "ORDER_MERGE_FAN_IN", 2)
    observed_chunk_sizes: list[int] = []
    observed_fan_in: list[int] = []
    original_fsync = subject.fsync_directory
    original_heapify = subject.heapq.heapify

    def observe_fsync(path: Path) -> None:
        original_fsync(path)
        observed_chunk_sizes.extend(
            item.stat().st_size for item in subject.scratch_path(scratch).glob(".aggtrades-chunk-*")
        )

    def observe_heapify(values: list[object]) -> None:
        observed_fan_in.append(len(values))
        original_heapify(values)

    monkeypatch.setattr(subject, "fsync_directory", observe_fsync)
    monkeypatch.setattr(subject.heapq, "heapify", observe_heapify)
    frame, _ = subject.frame_from_archive(
        source,
        "BTCUSDT",
        MAY_2023_START_MS,
        MAY_2023_START_MS + 1000,
        None,
        "2023-05",
        scratch,
        authenticated_archive_receipt(source),
    )
    assert subject.ORDER_RECORD.size == 32
    assert observed_chunk_sizes
    assert max(observed_chunk_sizes) <= 2 * subject.ORDER_RECORD.size
    assert observed_fan_in and max(observed_fan_in) <= 2
    assert frame["open"].to_list() == [11.0]
    assert aggtrades_scratch_entries(scratch) == []


def test_canonical_merge_failure_and_stale_direct_file_leave_no_owned_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    owned_scratch = subject.scratch_directory(scratch)
    stale = owned_scratch / ".aggtrades-crash.bin"
    stale.write_bytes(b"stale")
    stale.chmod(0o600)
    rows = [
        [str(index), "1", "1", str(index), str(index), str(MAY_2023_START_MS), "FALSE"]
        for index in (3, 2, 1)
    ]
    source = may_2023_archive(tmp_path, rows)
    allow_may_2023_order(monkeypatch, source)
    monkeypatch.setattr(subject, "ORDER_CHUNK_RECORDS", 1)
    monkeypatch.setattr(subject, "ORDER_MERGE_FAN_IN", 2)

    def merge_failure(_heap: list[object]) -> object:
        raise OSError("injected merge failure")

    monkeypatch.setattr(subject.heapq, "heappop", merge_failure)
    with pytest.raises(subject.AcquisitionError, match=r"^archive_order_canonicalization_failed$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            scratch,
            authenticated_archive_receipt(source),
        )
    assert aggtrades_scratch_entries(scratch) == []


def test_canonical_primary_failure_keeps_cleanup_failure_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = may_2023_archive(
        tmp_path,
        [
            [str(index), "1", "1", str(index), str(index), str(MAY_2023_START_MS), "FALSE"]
            for index in (3, 2, 1)
        ],
    )
    allow_may_2023_order(monkeypatch, source)
    monkeypatch.setattr(subject, "ORDER_CHUNK_RECORDS", 1)
    monkeypatch.setattr(subject, "ORDER_MERGE_FAN_IN", 2)
    monkeypatch.setattr(
        subject.heapq,
        "heappop",
        lambda _: (_ for _ in ()).throw(OSError("injected merge failure")),
    )
    monkeypatch.setattr(
        subject,
        "remove_aggtrades_file",
        lambda *_: (_ for _ in ()).throw(OSError("injected cleanup failure")),
    )
    with pytest.raises(subject.AcquisitionError) as raised:
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            scratch,
            authenticated_archive_receipt(source),
        )
    assert str(raised.value) == "archive_order_canonicalization_failed"
    assert any(
        note.startswith("archive_order_cleanup_failed:")
        for note in getattr(raised.value, "__notes__", ())
    )


def test_canonical_merge_removal_failure_is_retried_by_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = may_2023_archive(
        tmp_path,
        [
            [str(index), "1", "1", str(index), str(index), str(MAY_2023_START_MS), "FALSE"]
            for index in (3, 2, 1)
        ],
    )
    allow_may_2023_order(monkeypatch, source)
    monkeypatch.setattr(subject, "ORDER_CHUNK_RECORDS", 1)
    monkeypatch.setattr(subject, "ORDER_MERGE_FAN_IN", 2)
    original_identity = subject.scratch_file_identity
    original_remove = subject.remove_aggtrades_file
    tracked: set[Path] = set()
    attempts: dict[Path, int] = {}
    failed = False

    def track_identity(path: Path) -> tuple[int, int]:
        identity = original_identity(path)
        if path.name.startswith(".aggtrades-"):
            tracked.add(path)
        return identity

    def fail_first_removal(path: Path, identity: tuple[int, int] | None = None) -> None:
        nonlocal failed
        attempts[path] = attempts.get(path, 0) + 1
        if not failed:
            failed = True
            raise OSError("injected removal failure")
        original_remove(path, identity)

    monkeypatch.setattr(subject, "scratch_file_identity", track_identity)
    monkeypatch.setattr(subject, "remove_aggtrades_file", fail_first_removal)
    with pytest.raises(subject.AcquisitionError, match=r"^archive_order_canonicalization_failed$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            scratch,
            authenticated_archive_receipt(source),
        )
    assert failed
    assert set(attempts) == tracked
    assert max(attempts.values()) == 2
    assert aggtrades_scratch_entries(scratch) == []


def test_pre_sink_identity_failure_closes_descriptor_and_is_stale_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = may_2023_archive(
        tmp_path,
        [["1", "10", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"]],
    )
    allow_may_2023_order(monkeypatch, source)
    original_identity = subject.scratch_file_identity

    def pre_sink_identity(path: Path) -> tuple[int, int]:
        if path.name.startswith(".aggtrades-chunk-"):
            raise subject.AcquisitionError("injected_identity_failure")
        return original_identity(path)

    def descriptor_count() -> int:
        return len(list(Path("/proc/self/fd").iterdir()))

    before = descriptor_count()
    monkeypatch.setattr(subject, "scratch_file_identity", pre_sink_identity)
    with pytest.raises(subject.AcquisitionError, match=r"^injected_identity_failure$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            scratch,
            authenticated_archive_receipt(source),
        )
    assert descriptor_count() == before
    assert aggtrades_scratch_entries(scratch)
    monkeypatch.setattr(subject, "scratch_file_identity", original_identity)
    subject.frame_from_archive(
        source,
        "BTCUSDT",
        MAY_2023_START_MS,
        MAY_2023_START_MS + 1000,
        None,
        "2023-05",
        scratch,
        authenticated_archive_receipt(source),
    )
    assert aggtrades_scratch_entries(scratch) == []


def test_cleanup_failure_is_recovered_before_the_next_authenticated_canonicalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = may_2023_archive(
        tmp_path,
        [
            ["2", "11", "1", "2", "2", str(MAY_2023_START_MS + 1), "FALSE"],
            ["1", "10", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"],
        ],
    )
    allow_may_2023_order(monkeypatch, source)
    original_remove = subject.remove_aggtrades_file
    monkeypatch.setattr(
        subject, "remove_aggtrades_file", lambda *_: (_ for _ in ()).throw(OSError("cleanup"))
    )
    with pytest.raises(subject.AcquisitionError, match=r"^archive_order_canonicalization_failed$"):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            scratch,
            authenticated_archive_receipt(source),
        )
    assert aggtrades_scratch_entries(scratch)
    monkeypatch.setattr(subject, "remove_aggtrades_file", original_remove)
    subject.frame_from_archive(
        source,
        "BTCUSDT",
        MAY_2023_START_MS,
        MAY_2023_START_MS + 1000,
        None,
        "2023-05",
        scratch,
        authenticated_archive_receipt(source),
    )
    assert aggtrades_scratch_entries(scratch) == []


@pytest.mark.parametrize(
    "kind",
    ("directory", "wrong_mode", "special_mode", "hardlink"),
)
def test_unsafe_aggtrades_scratch_entry_rejects_without_mutation(tmp_path: Path, kind: str) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    owned_scratch = subject.scratch_directory(scratch)
    unsafe = owned_scratch / ".aggtrades-foreign"
    if kind == "directory":
        unsafe.mkdir(mode=0o700)
    else:
        unsafe.write_bytes(b"unsafe")
        mode = 0o600
        if kind == "wrong_mode":
            mode = 0o644
        elif kind == "special_mode":
            mode = 0o4600
        unsafe.chmod(mode)
        if kind == "hardlink":
            os.link(unsafe, owned_scratch / ".acquire-linked")
    before = _lstat_tree_snapshot(tmp_path)
    with pytest.raises(subject.AcquisitionError, match=r"^unsafe_scratch_entry$"):
        subject.scratch_directory(scratch)
    assert _lstat_tree_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("source_factory", "error"),
    [
        (
            lambda root: archive(
                root,
                [["1", "1", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"]],
                "id,wrong",
                "BTCUSDT-aggTrades-2023-05.csv",
            ),
            r"^archive_csv_header_invalid$",
        ),
        (
            lambda root: may_2023_archive(
                root, [["1", "", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"]]
            ),
            r"^archive_csv_null_or_schema_invalid$",
        ),
        (
            lambda root: may_2023_archive(
                root, [["1", "0", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"]]
            ),
            r"^archive_trade_value_or_month_bounds_invalid$",
        ),
        (
            lambda root: archive(
                root,
                [["1", "1", "1", "1", "1", str(MAY_2023_START_MS), "FALSE"]],
                member="not-the-official-member.csv",
            ),
            r"^archive_zip_schema_invalid$",
        ),
    ],
)
def test_allowlisted_branch_preserves_parse_precedence_and_cleans_order_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_factory: object,
    error: str,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subject.scratch_directory(scratch)
    source = source_factory(tmp_path)
    allow_may_2023_order(monkeypatch, source)
    before = _tree_snapshot(tmp_path)
    with pytest.raises(subject.AcquisitionError, match=error):
        subject.frame_from_archive(
            source,
            "BTCUSDT",
            MAY_2023_START_MS,
            MAY_2023_START_MS + 1000,
            None,
            "2023-05",
            scratch,
            authenticated_archive_receipt(source),
        )
    assert aggtrades_scratch_entries(scratch) == []
    assert _tree_snapshot(tmp_path) == before


def test_validate_complete_never_reopens_a_retired_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        subject,
        "frame_from_archive",
        lambda *args: pytest.fail("offline validation reopened a retired ZIP"),
    )
    contract = subject.Contract(
        "BTCUSDT", 1704067200000, 1704067201000, 1704067200000, 1704067200001
    )
    subject.validate_complete(output, [contract], report / "provenance", report)
    assert not list((report / "provenance" / "archives").glob("**/*.zip"))


@pytest.mark.parametrize(
    "member,extra_members,mode",
    [
        ("ETHUSDT-aggTrades-2024-01.csv", (), stat.S_IFREG | 0o600),
        ("BTCUSDT-aggTrades-2024-02.csv", (), stat.S_IFREG | 0o600),
        ("BTCUSDT-aggTrades-2024-01.csv.bak", (), stat.S_IFREG | 0o600),
        ("nested/BTCUSDT-aggTrades-2024-01.csv", (), stat.S_IFREG | 0o600),
        ("BTCUSDT-aggTrades-2024-01.csv", ("extra.csv",), stat.S_IFREG | 0o600),
        ("BTCUSDT-aggTrades-2024-01.csv", (), stat.S_IFDIR | 0o700),
        ("BTCUSDT-aggTrades-2024-01.csv", (), stat.S_IFLNK | 0o777),
        ("BTCUSDT-aggTrades-2024-01.csv", (), stat.S_IFIFO | 0o600),
    ],
)
def test_production_parser_requires_exact_single_regular_official_member(
    tmp_path: Path, member: str, extra_members: tuple[str, ...], mode: int
) -> None:
    with pytest.raises(subject.AcquisitionError, match="archive_zip_schema_invalid"):
        subject.frame_from_archive(
            archive(
                tmp_path,
                [["1", "1", "1", "1", "1", "1704067200000", "False"]],
                member=member,
                extra_members=extra_members,
                mode=mode,
            ),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-01",
        )


def test_production_parser_rejects_noncanonical_month_and_late_malformed_quoting(
    tmp_path: Path,
) -> None:
    with pytest.raises(subject.AcquisitionError, match="archive_month_invalid"):
        subject.frame_from_archive(
            archive(tmp_path, [["1", "1", "1", "1", "1", "1704067200000", "False"]]),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-1",
        )
    with pytest.raises(subject.AcquisitionError, match="archive_csv_schema_invalid"):
        subject.frame_from_archive(
            archive(
                tmp_path,
                [],
                content='1,1,1,1,1,1704067200000,False\n2,"unterminated',
            ),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-01",
        )


@pytest.mark.parametrize(
    "rows,error",
    [
        (
            [
                ["1", "1", "1", "1", "1", "1704067200000", "False"],
                ["1", "1", "1", "1", "1", "1704067200001", "False"],
            ],
            "order",
        ),
        ([["1", "1", "1", "2", "1", "1704067200000", "False"]], "value"),
        ([["1", "1", "1", "1", "1", "1704067200000", "maybe"]], "value"),
    ],
)
def test_production_parser_rejects_only_actual_domain_violations(
    tmp_path: Path, rows: list[list[str]], error: str
) -> None:
    with pytest.raises(subject.AcquisitionError, match=error):
        subject.frame_from_archive(
            archive(tmp_path, rows), "BTCUSDT", 1704067200000, 1704067201000, 1.0, "2024-01"
        )


def test_production_parser_rejects_unknown_header_null_and_month_bounds(tmp_path: Path) -> None:
    with pytest.raises(subject.AcquisitionError, match="header"):
        subject.frame_from_archive(
            archive(tmp_path, [["1", "1", "1", "1", "1", "1704067200000", "False"]], "id,wrong"),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-01",
        )
    with pytest.raises(subject.AcquisitionError):
        subject.frame_from_archive(
            archive(tmp_path, [["1", "", "1", "1", "1", "1704067200000", "False"]]),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-01",
        )
    with pytest.raises(subject.AcquisitionError, match="month"):
        subject.frame_from_archive(
            archive(tmp_path, [["1", "1", "1", "1", "1", "1706745600000", "False"]]),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            1.0,
            "2024-01",
        )


@pytest.mark.parametrize(
    "index,lexeme",
    [
        (0, " 1"),
        (6, " false "),
        (0, "+1"),
        (3, "-1"),
        (4, "+1"),
        (5, "+1704067200000"),
        (1, ".1"),
        (1, "1."),
        (2, "1_0"),
        (1, "NaN"),
        (2, "NaN"),
        (1, "Inf"),
        (2, "-Inf"),
        (2, "Inf"),
        (1, "0"),
        (1, "-1"),
        (2, "-1"),
    ],
)
def test_production_parser_rejects_noncanonical_field_lexemes_before_coercion(
    tmp_path: Path, index: int, lexeme: str
) -> None:
    row = ["1", "1", "1", "1", "1", "1704067200000", "False"]
    row[index] = lexeme
    with pytest.raises(subject.AcquisitionError, match="value_or_month_bounds"):
        subject.frame_from_archive(
            archive(tmp_path, [row]),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-01",
        )


def test_production_parser_accepts_canonical_positive_exponent_lexemes(tmp_path: Path) -> None:
    frame, _ = subject.frame_from_archive(
        archive(
            tmp_path,
            [["1", "1.25e+1", "2.5E-1", "1", "1", "1704067200000", "TRUE"]],
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
        ),
        "BTCUSDT",
        1704067200000,
        1704067201000,
        None,
        "2024-01",
    )
    assert frame["open"].to_list() == [12.5]
    assert frame["volume"].to_list() == [0.25]


def _old_semantics_reference(
    start_ms: int, size: int, carry: float | None, rows: list[tuple[int, float, float]]
) -> tuple[subject.pl.DataFrame, float | None]:
    open_values = [float("nan")] * size
    high_values = [float("nan")] * size
    low_values = [float("nan")] * size
    close_values = [float("nan")] * size
    volume_values = [0.0] * size
    for timestamp, price, quantity in rows:
        if timestamp < start_ms:
            carry = price
        else:
            second = (timestamp - start_ms) // 1000
            if second >= size:
                continue
            if subject.math.isnan(open_values[second]):
                open_values[second] = high_values[second] = low_values[second] = price
            else:
                high_values[second] = max(high_values[second], price)
                low_values[second] = min(low_values[second], price)
            close_values[second] = price
            volume_values[second] += quantity
    output = {"datetime": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for index in range(size):
        if subject.math.isnan(open_values[index]):
            assert carry is not None
            values = (carry, carry, carry, carry, 0.0)
        else:
            carry = close_values[index]
            values = (
                open_values[index],
                high_values[index],
                low_values[index],
                carry,
                volume_values[index],
            )
        output["datetime"].append(
            datetime.fromtimestamp((start_ms + index * 1000) / 1000, UTC).replace(tzinfo=None)
        )
        for name, value in zip(subject.RAW_COLUMNS[1:], values, strict=True):
            output[name].append(value)
    return subject.pl.DataFrame(output).with_columns(
        subject.pl.col("datetime").cast(subject.pl.Datetime("ms")),
        *[subject.pl.col(name).cast(subject.pl.Float64) for name in subject.RAW_COLUMNS[1:]],
    ), carry


@pytest.mark.parametrize("index", (0, 3, 4, 5))
def test_production_parser_rejects_unicode_integer_tokens(tmp_path: Path, index: int) -> None:
    row = ["1", "1", "1", "1", "1", "1704067200000", "FALSE"]
    row[index] = (
        "\u0661"
        if index != 5
        else ("\u0661\u0667\u0660\u0664\u0660\u0666\u0667\u0662\u0660\u0660\u0660\u0660\u0660")
    )
    with pytest.raises(subject.AcquisitionError, match="value_or_month_bounds"):
        subject.frame_from_archive(
            archive(tmp_path, [row]),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-01",
        )


@pytest.mark.parametrize("index", range(7))
def test_production_parser_empty_fields_take_null_schema_precedence(
    tmp_path: Path, index: int
) -> None:
    row = ["1", "1", "1", "1", "1", "1704067200000", "FALSE"]
    row[index] = ""
    with pytest.raises(subject.AcquisitionError, match=r"^archive_csv_null_or_schema_invalid$"):
        subject.frame_from_archive(
            archive(tmp_path, [row]),
            "BTCUSDT",
            1704067200000,
            1704067201000,
            None,
            "2024-01",
        )


@pytest.mark.parametrize(
    "price,quantity,accepted",
    [
        ("1.25e+1", "2.5E-1", True),
        ("1e", "1", False),
        ("1", "2e-", False),
        ("1.", "1", False),
        ("+1", "1", False),
        ("1", "NaN", False),
    ],
)
def test_production_parser_decimal_lexemes_through_archive(
    tmp_path: Path, price: str, quantity: str, accepted: bool
) -> None:
    source = archive(
        tmp_path,
        [["1", price, quantity, "1", "1", "1704067200000", "TrUe"]],
    )
    if not accepted:
        with pytest.raises(subject.AcquisitionError, match="value_or_month_bounds"):
            subject.frame_from_archive(
                source, "BTCUSDT", 1704067200000, 1704067201000, None, "2024-01"
            )
        return
    frame, facts = subject.frame_from_archive(
        source, "BTCUSDT", 1704067200000, 1704067201000, None, "2024-01"
    )
    assert frame["open"].to_list() == [12.5]
    assert frame["volume"].to_list() == [0.25]
    assert facts["carry_close"] == 12.5


def test_production_parser_normalizes_buyer_once_per_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original_reader = subject.csv.reader

    class CountingBuyer(str):
        def lower(self) -> str:
            nonlocal calls
            calls += 1
            return super().lower()

    def reader_with_count(*args: object, **kwargs: object):
        for fields in original_reader(*args, **kwargs):
            fields[6] = CountingBuyer(fields[6])
            yield fields

    monkeypatch.setattr(subject.csv, "reader", reader_with_count)
    frame, _ = subject.frame_from_archive(
        archive(
            tmp_path,
            [["1", "1", "1", "1", "1", "1704067200000", "TrUe"]],
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
        ),
        "BTCUSDT",
        1704067200000,
        1704067201000,
        None,
        "2024-01",
    )
    assert frame["close"].to_list() == [1.0]
    assert calls == 1


def test_production_parser_forward_fills_gaps_with_carry_and_preserves_parquet_bytes(
    tmp_path: Path,
) -> None:
    start_ms = 1704067200000
    rows = [
        ["2", "10", "2", "2", "2", str(start_ms + 100), "TRUE"],
        ["3", "12", "3", "3", "3", str(start_ms + 2100), "False"],
        ["4", "11", "4", "4", "4", str(start_ms + 2500), "fAlSe"],
    ]
    frame, facts = subject.frame_from_archive(
        archive(tmp_path, rows), "BTCUSDT", start_ms, start_ms + 4000, 9.0, "2024-01"
    )
    reference, reference_carry = _old_semantics_reference(
        start_ms,
        4,
        9.0,
        [
            (start_ms + 100, 10.0, 2.0),
            (start_ms + 2100, 12.0, 3.0),
            (start_ms + 2500, 11.0, 4.0),
        ],
    )
    assert frame.columns == subject.RAW_COLUMNS
    assert frame.dtypes == [subject.pl.Datetime("ms"), *([subject.pl.Float64] * 5)]
    assert frame.to_dict(as_series=False) == reference.to_dict(as_series=False)
    assert facts["carry_close"] == reference_carry == 11.0
    optimized_bytes, reference_bytes = io.BytesIO(), io.BytesIO()
    frame.write_parquet(optimized_bytes)
    reference.write_parquet(reference_bytes)
    assert optimized_bytes.getvalue() == reference_bytes.getvalue()


def test_production_parser_rejects_initial_gap_without_carry_and_fills_with_carry(
    tmp_path: Path,
) -> None:
    source = archive(
        tmp_path,
        [["1", "12", "2", "1", "1", "1704067201000", "FALSE"]],
    )
    with pytest.raises(
        subject.AcquisitionError, match="raw_first_owned_second_has_no_official_close"
    ):
        subject.frame_from_archive(source, "BTCUSDT", 1704067200000, 1704067202000, None, "2024-01")
    frame, facts = subject.frame_from_archive(
        source, "BTCUSDT", 1704067200000, 1704067202000, 10.0, "2024-01"
    )
    assert frame.to_dict(as_series=False) == {
        "datetime": [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 1)],
        "open": [10.0, 12.0],
        "high": [10.0, 12.0],
        "low": [10.0, 12.0],
        "close": [10.0, 12.0],
        "volume": [0.0, 2.0],
    }
    assert facts["carry_close"] == 12.0


def test_production_parser_accepts_zero_quantity_lexemes_with_ohlc_and_order(
    tmp_path: Path,
) -> None:
    start = 1704067200000
    frame, facts = subject.frame_from_archive(
        archive(
            tmp_path,
            [
                ["1", "10", "0", "1", "1", str(start), "False"],
                ["2", "12", "0.0", "2", "2", str(start), "False"],
                ["3", "11", "0e0", "3", "3", str(start), "False"],
            ],
        ),
        "BTCUSDT",
        start,
        start + 1000,
        None,
        "2024-01",
    )

    assert frame["open"].to_list() == [10.0]
    assert frame["high"].to_list() == [12.0]
    assert frame["low"].to_list() == [10.0]
    assert frame["close"].to_list() == [11.0]
    assert frame["volume"].to_list() == [0.0]
    assert facts["first_trade"] == (start, 1)
    assert facts["last_trade"] == (start, 3)


@pytest.mark.parametrize(
    "payload",
    [
        b'[{"symbol":"BTCUSDT","fundingTime":0,"fundingTime":1,"fundingRate":"0"}]',
        b"[NaN]",
    ],
)
def test_funding_payload_parse_failures_are_rejected_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    contract = subject.Contract("BTCUSDT", 0, 1, 0, 1)
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    destination = report / "provenance/funding_pages/BTCUSDT/000001.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    monkeypatch.setattr(
        subject, "fetch_receipt", lambda *_args, **_kwargs: {"sha256": subject.sha256(payload)}
    )

    with pytest.raises(subject.AcquisitionError, match="funding_api_json_invalid") as error:
        subject.acquire_funding(contract, output, report)

    assert isinstance(error.value.__cause__, subject.AcquisitionError)
    assert error.value.__cause__.__cause__ is not None
    assert _tree_snapshot(output) == []


def test_funding_payload_allows_noncanonical_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = subject.Contract("BTCUSDT", 0, 1, 0, 1)
    report = tmp_path / "report"
    payload = b'[\n  { "fundingRate" : "0", "fundingTime" : 0, "symbol" : "BTCUSDT" }\n]\n'
    destination = report / "provenance/funding_pages/BTCUSDT/000001.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    monkeypatch.setattr(
        subject, "fetch_receipt", lambda *_args, **_kwargs: {"sha256": subject.sha256(payload)}
    )

    rows, hashes = subject.funding_pages(contract, report)

    assert rows == [{"fundingRate": "0", "fundingTime": 0, "symbol": "BTCUSDT"}]
    assert hashes == [subject.sha256(payload)]


def test_exchange_info_rejects_duplicate_nested_symbol_keys_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, report = _eligible_fixture(tmp_path, monkeypatch)
    exchange = report / "provenance/exchangeInfo.json"
    receipt = exchange.with_suffix(".json.receipt.json")
    exchange.unlink()
    receipt.unlink()
    payload = b'{"symbols":[{"symbol":"BTCUSDT","symbol":"ETHUSDT"}]}'
    _write_request_fixture(exchange, f"{subject.API_BASE}/exchangeInfo", payload, report)
    before_output = _tree_snapshot(output)
    before_report = _tree_snapshot(report)
    contract = subject.load_contract(tmp_path / "ignored")[0]

    with pytest.raises(subject.AcquisitionError, match="exchange_info_schema_invalid") as error:
        subject.validate_complete(output, [contract], report / "provenance", report)

    assert isinstance(error.value.__cause__, subject.AcquisitionError)
    assert error.value.__cause__.__cause__ is not None
    assert _tree_snapshot(output) == before_output
    assert _tree_snapshot(report) == before_report


def test_ton_08_preowned_12_absent_and_16_owned_timestamp_is_preserved() -> None:
    march_08 = 1709280000000
    march_16 = 1709308800000
    march_20 = 1709323200000
    contract = subject.Contract("TONUSDT", 0, 1, march_16, march_20 + 1)
    rows = subject.normalize_funding(
        contract,
        [
            {"symbol": "TONUSDT", "fundingTime": march_08, "fundingRate": "0"},
            {"symbol": "TONUSDT", "fundingTime": march_16 + 500, "fundingRate": "0"},
            {"symbol": "TONUSDT", "fundingTime": march_20, "fundingRate": "0"},
        ],
    )
    assert [x["timestamp_ms"] for x in rows] == [march_16 + 500, march_20]
    assert [x["source_timestamp_ms"] for x in rows] == [march_16 + 500, march_20]
    assert all(x["timestamp_ms"] != march_08 for x in rows)


def test_ton_12_unavailable_settlement_is_rejected() -> None:
    march_08 = 1709280000000
    march_12 = 1709294400000
    march_16 = 1709308800000
    contract = subject.Contract("TONUSDT", 0, 1, march_16, march_16 + 1)
    with pytest.raises(subject.AcquisitionError, match="unavailable"):
        subject.normalize_funding(
            contract,
            [
                {"symbol": "TONUSDT", "fundingTime": march_08, "fundingRate": "0"},
                {"symbol": "TONUSDT", "fundingTime": march_12, "fundingRate": "0"},
                {"symbol": "TONUSDT", "fundingTime": march_16, "fundingRate": "0"},
            ],
        )


class _StreamingResponse:
    def __init__(self, data: bytes) -> None:
        self.data, self.offset = data, 0

    def __enter__(self) -> _StreamingResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return "https://fapi.binance.com/fapi/v1/exchangeInfo"

    def read(self, size: int = -1) -> bytes:
        assert size != -1, "transport must never use unbounded response.read()"
        result = self.data[self.offset : self.offset + size]
        self.offset += len(result)
        return result


def test_streaming_transport_receipt_and_cache_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subject.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _StreamingResponse(b'{"symbols":[]}'),
    )
    destination = tmp_path / "exchangeInfo.json"
    receipt = subject.fetch_receipt("https://fapi.binance.com/fapi/v1/exchangeInfo", destination)
    assert receipt["byte_count"] == len(b'{"symbols":[]}')
    assert (
        subject.fetch_receipt("https://fapi.binance.com/fapi/v1/exchangeInfo", destination)[
            "sha256"
        ]
        == hashlib.sha256(b'{"symbols":[]}').hexdigest()
    )
    with pytest.raises(subject.AcquisitionError, match="receipt"):
        subject.fetch_receipt(
            "https://fapi.binance.com/fapi/v1/exchangeInfo?x=1", destination, {"x": 1}
        )


def test_cached_receipt_rejects_query_and_hash_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "page.json"
    destination.write_bytes(b"[]")
    receipt = {
        "schema": "official_request_receipt.v1",
        "requested_url": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT",
        "final_url": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT",
        "final_host": "fapi.binance.com",
        "query": {"symbol": "BTCUSDT"},
        "byte_count": 2,
        "sha256": hashlib.sha256(b"[]").hexdigest(),
    }
    destination.with_suffix(".json.receipt.json").write_bytes(subject.canonical_bytes(receipt))
    with pytest.raises(subject.AcquisitionError, match="receipt"):
        subject.fetch_receipt(
            "https://fapi.binance.com/fapi/v1/fundingRate?symbol=ETHUSDT",
            destination,
            {"symbol": "ETHUSDT"},
        )
    destination.write_bytes(b"[1]")
    with pytest.raises(subject.AcquisitionError, match="receipt"):
        subject.fetch_receipt(
            "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT",
            destination,
            {"symbol": "BTCUSDT"},
        )


def test_request_receipt_requires_exact_canonical_producer_fields(tmp_path: Path) -> None:
    destination = tmp_path / "page.json"
    url = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT"
    _write_request_fixture(destination, url, b"[]")
    receipt_path = destination.with_suffix(".json.receipt.json")
    receipt = subject.read_json(receipt_path)
    receipt["unexpected"] = True
    receipt_path.write_bytes(subject.canonical_bytes(receipt))
    with pytest.raises(subject.AcquisitionError, match="receipt"):
        subject.request_receipt(destination, url)
    receipt_path.write_bytes(
        b'{"byte_count":2,"byte_count":2,"final_host":"fapi.binance.com","final_url":"'
        b'https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT","query":{"symbol":'
        b'"BTCUSDT"},"retrieved_at_utc":"2024-01-01T00:00:00Z","requested_url":"'
        b'https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT","schema":'
        b'"official_request_receipt.v1","sha256":"'
        + hashlib.sha256(b"[]").hexdigest().encode()
        + b'"}\n'
    )
    with pytest.raises(subject.AcquisitionError, match="immutable_json"):
        subject.request_receipt(destination, url)


def test_plan_report_can_be_securely_adopted_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = subject.Contract(
        "BTCUSDT", 1704067200000, 1706745600000, 1704067200000, 1706745600000
    )
    monkeypatch.setattr(subject, "load_contract", lambda _: [contract])
    monkeypatch.setattr(subject, "load_evidence", lambda _: {})
    monkeypatch.setattr(subject, "bind_input_provenance", lambda *args: None)
    monkeypatch.setattr(subject, "safe_existing_directory", lambda _: [1, 1])

    def cached_exchange(_: str, destination: Path, **__: object) -> dict[str, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'{"symbols":[{"symbol":"BTCUSDT"}]}')
        return {"sha256": "x"}

    monkeypatch.setattr(subject, "fetch_receipt", cached_exchange)
    monkeypatch.setattr(subject, "rebuild_manifest", lambda *_: None)
    monkeypatch.setattr(subject, "acquire_archive", lambda *args, **kwargs: None)
    report, output = tmp_path / "report", tmp_path / "output"
    common = [
        "--contract-manifest",
        str(tmp_path / "contract"),
        "--availability-evidence",
        str(tmp_path / "evidence"),
        "--output-root",
        str(output),
        "--report-dir",
        str(report),
        "--forbidden-root",
        "/forbidden",
        "--symbols",
        "BTCUSDT",
        "--months",
        "2024-01",
    ]
    assert subject.main(common) == 0
    assert subject.main([*common, "--execute"]) == 0
    assert (output / ".alpha_max_owner.json").exists()


def test_noncontiguous_month_selectors_are_forwarded_as_an_exact_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = subject.Contract(
        "BTCUSDT", 1704067200000, 1711929600000, 1704067200000, 1711929600000
    )
    monkeypatch.setattr(subject, "load_contract", lambda _: [contract])
    monkeypatch.setattr(subject, "load_evidence", lambda _: {})
    monkeypatch.setattr(subject, "bind_input_provenance", lambda *args: None)
    monkeypatch.setattr(subject, "safe_existing_directory", lambda _: [1, 1])

    def cached_exchange(_: str, destination: Path, **__: object) -> dict[str, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'{"symbols":[{"symbol":"BTCUSDT"}]}')
        return {"sha256": "x"}

    chosen: list[set[str] | None] = []
    monkeypatch.setattr(subject, "fetch_receipt", cached_exchange)
    monkeypatch.setattr(
        subject, "acquire_archive", lambda _c, _o, _r, months=None: chosen.append(months)
    )
    monkeypatch.setattr(subject, "rebuild_manifest", lambda *_: None)
    report, output = tmp_path / "report", tmp_path / "output"
    assert (
        subject.main(
            [
                "--contract-manifest",
                str(tmp_path / "contract.json"),
                "--availability-evidence",
                str(tmp_path / "evidence.json"),
                "--output-root",
                str(output),
                "--report-dir",
                str(report),
                "--forbidden-root",
                "/forbidden",
                "--symbols",
                "BTCUSDT",
                "--months",
                "2024-01",
                "2024-03",
                "--execute",
            ]
        )
        == 0
    )
    assert chosen == [{"2024-01", "2024-03"}]
    assert subject.read_json(report / "plan.json") == {
        **subject.full_plan(),
        "symbols": ["BTCUSDT"],
        "months": ["2024-01", "2024-03"],
    }
    assert (output / ".alpha_max_owner.json").exists()


def test_execute_recovers_empty_report_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = subject.Contract(
        "BTCUSDT", 1704067200000, 1706745600000, 1704067200000, 1706745600000
    )
    monkeypatch.setattr(subject, "load_contract", lambda _: [contract])
    monkeypatch.setattr(subject, "load_evidence", lambda _: {})
    monkeypatch.setattr(subject, "bind_input_provenance", lambda *_: None)
    monkeypatch.setattr(subject, "safe_existing_directory", lambda _: [1, 1])

    def cached(_: str, destination: Path, **__: object) -> dict[str, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'{"symbols":[{"symbol":"BTCUSDT"}]}')
        return {"sha256": "x"}

    monkeypatch.setattr(subject, "fetch_receipt", cached)
    monkeypatch.setattr(subject, "acquire_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "rebuild_manifest", lambda *_: None)
    output, report = tmp_path / "output", tmp_path / "report"
    report.mkdir()
    arguments = [
        "--contract-manifest",
        str(tmp_path / "contract.json"),
        "--availability-evidence",
        str(tmp_path / "evidence.json"),
        "--output-root",
        str(output),
        "--report-dir",
        str(report),
        "--forbidden-root",
        "/forbidden",
        "--symbols",
        "BTCUSDT",
        "--months",
        "2024-01",
        "--execute",
    ]
    assert subject.main(arguments) == 0
    assert (report / "plan.json").exists()
    assert (output / ".alpha_max_owner.json").exists()


def test_empty_report_prefix_rejects_unowned_content(tmp_path: Path) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    report.mkdir()
    (report / "unexpected").write_bytes(b"x")
    with pytest.raises(subject.AcquisitionError, match="roots_resume_pair_invalid"):
        subject.assert_roots(output, report, [Path("/forbidden")], True)


def test_checked_tree_requires_transitive_directories_and_rejects_unsafe_objects(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    nested = root / "one" / "two"
    nested.mkdir(parents=True)
    (nested / "file").write_bytes(b"x")
    subject.checked_tree(root, {"one/two/file"})

    (root / "one" / "extra").mkdir()
    with pytest.raises(subject.AcquisitionError, match="inventory"):
        subject.checked_tree(root, {"one/two/file"})
    (root / "one" / "extra").rmdir()
    (root / "extra").write_bytes(b"x")
    with pytest.raises(subject.AcquisitionError, match="inventory"):
        subject.checked_tree(root, {"one/two/file"})
    (root / "extra").unlink()
    (root / "link").symlink_to(nested, target_is_directory=True)
    with pytest.raises(subject.AcquisitionError, match="unsafe"):
        subject.checked_tree(root, {"one/two/file"})
    (root / "link").unlink()
    os.link(nested / "file", root / "hardlink")
    with pytest.raises(subject.AcquisitionError, match="unsafe"):
        subject.checked_tree(root, {"one/two/file"})


def test_verified_partition_rejects_stale_raw_and_funding_receipts(tmp_path: Path) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    raw = subject.pl.DataFrame(
        {
            "datetime": [subject.datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).with_columns(subject.pl.col("datetime").cast(subject.pl.Datetime("ms")))
    funding = subject.pl.DataFrame(
        {
            "timestamp_ms": [1],
            "source_timestamp_ms": [1],
            "exchange": ["binance"],
            "symbol": ["BTCUSDT"],
            "funding_rate": [0.1],
        }
    )
    for relative, frame, carry, pages in (
        ("raw.parquet", raw, 1.0, ["raw-page"]),
        ("funding.parquet", funding, None, ["funding-page"]),
    ):
        target = output / relative
        subject.publish_frame(target, frame, output)
        receipt = subject.expected_partition_receipt(
            relative, "source", target, frame.height, 0, 1, carry, carry, pages
        )
        subject.immutable_json(subject.partition_path(report, relative), receipt, report)
        assert subject.verified_partition(report, relative, target, receipt, frame)
        stale = receipt | {"page_hashes": ["stale"]}
        with pytest.raises(subject.AcquisitionError, match="partition"):
            subject.verified_partition(report, relative, target, stale, frame)
        stale = receipt | {"input_carry_close": 9.0 if carry is not None else 1.0}
        with pytest.raises(subject.AcquisitionError, match="partition"):
            subject.verified_partition(report, relative, target, stale, frame)


def test_acquire_archive_resets_carry_across_selected_month_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = 1704067200000
    contract = subject.Contract("BTCUSDT", start, 1711929600000, start, 1711929600000)
    carries: list[float | None] = []

    def frame(
        _: Path,
        _symbol: str,
        frame_start: int,
        frame_end: int,
        carry: float | None,
        __: str,
        ___: Path | None,
        ____: dict[str, object] | None,
    ):
        carries.append(carry)
        return (
            subject.pl.DataFrame(
                {
                    "datetime": [subject.datetime.fromtimestamp(frame_start / 1000)],
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1.0],
                }
            ).with_columns(subject.pl.col("datetime").cast(subject.pl.Datetime("ms"))),
            {},
        )

    def cached(url: str, destination: Path, **__: object) -> dict[str, object]:
        payload = b"source"
        digest = subject.sha256(payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        destination.chmod(0o600)
        receipt = {
            "schema": "official_request_receipt.v1",
            "requested_url": url,
            "final_url": url,
            "final_host": subject.urllib.parse.urlsplit(url).hostname,
            "query": dict(subject.urllib.parse.parse_qsl(subject.urllib.parse.urlsplit(url).query)),
            "retrieved_at_utc": "2026-01-01T00:00:00Z",
            "byte_count": len(payload),
            "sha256": digest,
        }
        destination.with_suffix(destination.suffix + ".receipt.json").write_bytes(
            subject.canonical_bytes(receipt)
        )
        return receipt

    monkeypatch.setattr(subject, "fetch_receipt", cached)
    monkeypatch.setattr(subject, "parse_checksum", lambda *_: subject.sha256(b"source"))
    monkeypatch.setattr(subject, "frame_from_archive", frame)
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    subject.acquire_archive(contract, output, report, {"2024-01", "2024-03"})
    assert carries == [None, None, None, None]


def test_roots_reject_symlink_leaves_before_target_content_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "plan.json").write_bytes(b"sentinel")
    report_link = tmp_path / "report-link"
    report_link.symlink_to(sentinel, target_is_directory=True)
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == sentinel:
            raise AssertionError("symlink target must not be traversed")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    with pytest.raises(subject.AcquisitionError, match="unsafe_root"):
        subject.assert_roots(tmp_path / "output", report_link, [Path("/forbidden")], True)

    output_link = tmp_path / "output-link"
    output_link.symlink_to(sentinel, target_is_directory=True)
    with pytest.raises(subject.AcquisitionError, match="unsafe_root"):
        subject.assert_roots(output_link, tmp_path / "report", [Path("/forbidden")], True)

    dangling = tmp_path / "dangling-report"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(subject.AcquisitionError, match="unsafe_root"):
        subject.assert_roots(tmp_path / "output", dangling, [Path("/forbidden")], True)


def test_roots_are_lexically_guarded_and_disjoint(tmp_path: Path) -> None:
    with pytest.raises(subject.AcquisitionError, match="forbidden"):
        subject.assert_roots(tmp_path / "out", tmp_path / "report", [], False)
    with pytest.raises(subject.AcquisitionError, match="overlap"):
        subject.assert_roots(tmp_path / "out", tmp_path / "out/report", [Path("/forbidden")], False)


def test_inputs_are_lexically_rejected_before_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    source = forbidden / "contract.json"
    source.write_bytes(b"must-not-be-read")
    monkeypatch.setattr(
        subject,
        "safe_existing_directory",
        lambda _path: (_ for _ in ()).throw(AssertionError("accessed")),
    )
    with pytest.raises(subject.AcquisitionError, match="input_is_forbidden"):
        subject.assert_input_paths(source, tmp_path / "evidence.json", [forbidden])


def test_provenance_binding_reopens_approved_inputs_safely_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    contract = inputs / "contract.json"
    evidence = inputs / "evidence.json"
    contract_data, evidence_data = b'{"contract":true}\n', b'{"evidence":true}\n'
    contract.write_bytes(contract_data)
    evidence.write_bytes(evidence_data)
    monkeypatch.setattr(subject, "CONTRACT_SHA256", subject.sha256(contract_data))
    monkeypatch.setattr(subject, "EVIDENCE_SHA256", subject.sha256(evidence_data))
    assert subject.safe_file_bytes(contract) == contract_data
    assert subject.safe_file_bytes(evidence) == evidence_data
    replacement = tmp_path / "replacement"
    replacement.write_bytes(contract_data)
    contract.unlink()
    contract.symlink_to(replacement)
    report = tmp_path / "report"
    report.mkdir()
    with pytest.raises(subject.AcquisitionError):
        subject.bind_input_provenance(contract, evidence, report)
    assert not (report / "provenance").exists()


def test_roots_reserve_sibling_scratch_namespaces(tmp_path: Path) -> None:
    report = tmp_path / "report"
    output = subject.scratch_path(report)
    with pytest.raises(subject.AcquisitionError, match="scratch_roots_overlap"):
        subject.assert_roots(output, report, [Path("/forbidden")], False)
    with pytest.raises(subject.AcquisitionError, match="scratch_root_is_forbidden"):
        subject.assert_roots(
            tmp_path / "output",
            report,
            [subject.scratch_path(tmp_path / "output")],
            False,
        )


def test_root_parent_rejects_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(subject.AcquisitionError, match="unsafe_root_parent"):
        subject.safe_existing_directory(link / "child")


def test_trusted_parent_allows_root_owned_and_sticky_ancestors(
    tmp_path: Path,
) -> None:
    def item(mode: int, uid: int) -> subject.os.stat_result:
        return subject.os.stat_result((mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))

    subject.trusted_parent_directory(item(subject.stat.S_IFDIR | 0o755, 0), False)
    subject.trusted_parent_directory(
        item(subject.stat.S_IFDIR | subject.stat.S_ISVTX | 0o777, 0), False
    )
    subject.trusted_parent_directory(
        item(subject.stat.S_IFDIR | 0o755, 65534), False, namespace_root=True
    )
    subject.trusted_parent_directory(
        item(subject.stat.S_IFDIR | 0o755, 65534),
        False,
        namespace_root_uid=65534,
    )
    assert subject.safe_existing_directory(tmp_path) == [
        tmp_path.stat().st_dev,
        tmp_path.stat().st_ino,
    ]


def test_trusted_parent_rejects_writable_or_untrusted_paths() -> None:
    def item(mode: int, uid: int) -> subject.os.stat_result:
        return subject.os.stat_result((mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))

    with pytest.raises(subject.AcquisitionError, match="unsafe_root_parent"):
        subject.trusted_parent_directory(item(subject.stat.S_IFDIR | 0o777, 0), False)
    with pytest.raises(subject.AcquisitionError, match="unsafe_root_parent"):
        subject.trusted_parent_directory(item(subject.stat.S_IFDIR | 0o777, os.getuid()), True)
    with pytest.raises(subject.AcquisitionError, match="unsafe_root_parent"):
        subject.trusted_parent_directory(item(subject.stat.S_IFDIR | 0o700, os.getuid() + 1), False)
    with pytest.raises(subject.AcquisitionError, match="unsafe_root_parent"):
        subject.trusted_parent_directory(
            item(subject.stat.S_IFDIR | 0o777, 65534), False, namespace_root=True
        )
    with pytest.raises(subject.AcquisitionError, match="unsafe_root_parent"):
        subject.trusted_parent_directory(
            item(subject.stat.S_IFDIR | 0o755, 65534),
            True,
            namespace_root_uid=65534,
        )


@pytest.mark.parametrize("prefix", ["empty_output", "output_owner", "report_owner"])
def test_first_execute_prefixes_are_recovered(tmp_path: Path, prefix: str) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    report.mkdir()
    plan_data = subject.canonical_bytes({"plan": "exact"})
    subject.atomic_write(report / "plan.json", plan_data)
    run_id = subject.sha256(plan_data)
    if prefix != "empty_output":
        output.mkdir()
    if prefix == "output_owner":
        marker = subject.canonical_bytes(subject.ownership_marker(output, report, run_id))
        subject.atomic_write(output / ".alpha_max_owner.json", marker)
    elif prefix == "report_owner":
        marker = subject.canonical_bytes(subject.ownership_marker(output, report, run_id))
        subject.atomic_write(report / ".alpha_max_owner.json", marker)

    assert subject.recover_first_execute_prefix(output, report, plan_data, run_id)
    subject.verify_ownership(output, report, run_id)


def test_first_execute_prefix_rejects_unowned_or_mismatched_content(tmp_path: Path) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    plan_data = subject.canonical_bytes({"plan": "exact"})
    subject.atomic_write(report / "plan.json", plan_data)
    subject.atomic_write(output / "unexpected", b"x")
    assert not subject.recover_first_execute_prefix(
        output, report, plan_data, subject.sha256(plan_data)
    )

    (output / "unexpected").unlink()
    subject.atomic_write(output / ".alpha_max_owner.json", b'{"wrong":true}\n')
    with pytest.raises(subject.AcquisitionError, match="ownership"):
        subject.recover_first_execute_prefix(output, report, plan_data, subject.sha256(plan_data))


def test_verify_eligible_rejects_selectors_and_validation() -> None:
    with pytest.raises(SystemExit):
        subject.parse_args(
            [
                "--contract-manifest",
                "/contract",
                "--availability-evidence",
                "/evidence",
                "--output-root",
                "/output",
                "--report-dir",
                "/report",
                "--verify-eligible",
                "--symbols",
                "BTCUSDT",
            ]
        )
    with pytest.raises(SystemExit):
        subject.parse_args(
            [
                "--contract-manifest",
                "/contract",
                "--availability-evidence",
                "/evidence",
                "--output-root",
                "/output",
                "--report-dir",
                "/report",
                "--verify-eligible",
                "--validate-complete",
            ]
        )


def test_verify_eligible_accepts_immutable_fixture_without_network_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    before = (_tree_snapshot(output), _tree_snapshot(report))
    monkeypatch.setattr(
        subject,
        "fetch_receipt",
        lambda *args, **kwargs: pytest.fail("verify-eligible attempted network acquisition"),
    )
    monkeypatch.setattr(
        subject.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("verify-eligible attempted network acquisition"),
    )

    assert subject.main(args) == 0
    assert (_tree_snapshot(output), _tree_snapshot(report)) == before


@pytest.mark.parametrize(
    ("branch", "error"),
    [
        ("contract_leaf", "required_file_missing"),
        ("evidence_leaf", "required_file_missing"),
        ("provenance_parent", "unsafe_root_parent"),
    ],
)
def test_verify_eligible_rejects_unsafe_provenance_before_sentinel_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, branch: str, error: str
) -> None:
    args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    provenance = report / "provenance"
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    unsafe_path = provenance / (
        "contract_manifest.json" if branch != "evidence_leaf" else "availability_evidence.json"
    )
    (sentinel / unsafe_path.name).write_bytes(b"sentinel-must-not-be-read")
    if branch == "provenance_parent":
        provenance.rename(report / "provenance-real")
        provenance.symlink_to(sentinel, target_is_directory=True)
    else:
        unsafe_path.unlink()
        unsafe_path.symlink_to(sentinel / unsafe_path.name)

    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path == unsafe_path:
            pytest.fail("sentinel target must not be opened or read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    before = (
        _lstat_tree_snapshot(output),
        _lstat_tree_snapshot(report),
        _lstat_tree_snapshot(sentinel),
    )
    with pytest.raises(subject.AcquisitionError, match=f"^{error}$"):
        subject.main(args)
    assert (
        _lstat_tree_snapshot(output),
        _lstat_tree_snapshot(report),
        _lstat_tree_snapshot(sentinel),
    ) == before


@pytest.mark.parametrize(
    ("branch", "validate_directly"),
    [
        ("exchange_metadata", False),
        ("archive_checksum", True),
        ("funding_receipt", False),
        ("partition_parquet", True),
        ("output_inventory", False),
    ],
)
def test_eligible_validation_rejects_intermediate_symlink_before_sentinel_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, branch: str, validate_directly: bool
) -> None:
    args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    paths = {
        "exchange_metadata": report / "provenance",
        "archive_checksum": report / "provenance" / "archives" / "BTCUSDT",
        "funding_receipt": report / "provenance" / "funding_pages" / "BTCUSDT",
        "partition_parquet": output / "market_ohlcv_1s" / "binance" / "BTCUSDT",
        "output_inventory": output / "feature_points",
    }
    unsafe_path = paths[branch]
    sentinel = tmp_path / f"sentinel-{branch}"
    sentinel.mkdir()
    (sentinel / "must-not-read").write_bytes(b"sentinel-must-not-be-read")
    unsafe_path.rename(tmp_path / f"{branch}-real")
    unsafe_path.symlink_to(sentinel, target_is_directory=True)
    before = (
        _lstat_tree_snapshot(output),
        _lstat_tree_snapshot(report),
        _lstat_tree_snapshot(sentinel),
    )
    original_open = os.open
    original_scandir = os.scandir

    def sentinel_descriptor(value: object) -> bool:
        return isinstance(value, int) and os.path.realpath(f"/proc/self/fd/{value}") == str(
            sentinel
        )

    def guarded_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if sentinel_descriptor(dir_fd):
            pytest.fail("symlink sentinel must not be traversed")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def guarded_scandir(path: object):
        if sentinel_descriptor(path):
            pytest.fail("symlink sentinel must not be scanned")
        return original_scandir(path)

    monkeypatch.setattr(subject.os, "open", guarded_open)
    monkeypatch.setattr(subject.os, "scandir", guarded_scandir)
    with pytest.raises(subject.AcquisitionError):
        if validate_directly:
            subject.validate_complete(
                output,
                [
                    subject.Contract(
                        "BTCUSDT", 1704067200000, 1704067201000, 1704067200000, 1704067200001
                    )
                ],
                report / "provenance",
                report,
            )
        else:
            subject.main(args)
    assert (
        _lstat_tree_snapshot(output),
        _lstat_tree_snapshot(report),
        _lstat_tree_snapshot(sentinel),
    ) == before


def test_completed_eligible_original_argv_replays_offline_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    execute_args = [item for item in args if item != "--verify-eligible"] + [
        "--execute",
        "--validate-complete",
    ]
    before = (_tree_snapshot(output), _tree_snapshot(report))
    monkeypatch.setattr(
        subject, "fetch_receipt", lambda *args, **kwargs: pytest.fail("replay attempted network")
    )
    assert subject.main(execute_args) == 0
    assert (_tree_snapshot(output), _tree_snapshot(report)) == before


def test_streaming_and_scratch_publication_crashes_never_mutate_authenticated_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    source = archive(
        tmp_path,
        [["1", "1", "1", "1", "1", "1704067200000", "False"]],
    )
    frame = subject.pl.DataFrame(
        {
            "datetime": [subject.datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).with_columns(subject.pl.col("datetime").cast(subject.pl.Datetime("ms")))
    before = (_tree_snapshot(output), _tree_snapshot(report))
    subject.frame_from_archive(
        source, "BTCUSDT", 1704067200000, 1704067201000, None, "2024-01", report
    )
    assert (_tree_snapshot(output), _tree_snapshot(report)) == before

    created: list[Path] = []
    original = subject.scratch_file

    class SimulatedCrash(BaseException):
        pass

    def crash_after_create(root: Path, prefix: str, suffix: str = "") -> tuple[int, str]:
        fd, temporary = original(root, prefix, suffix)
        os.close(fd)
        created.append(Path(temporary))
        raise SimulatedCrash()

    monkeypatch.setattr(subject, "scratch_file", crash_after_create)
    for call in (
        lambda: subject.atomic_write(report / "receipt.json", b"{}", scratch_root=report),
        lambda: subject.fetch_receipt(
            "https://fapi.binance.com/fapi/v1/exchangeInfo",
            report / "exchangeInfo.json",
            scratch_root=report,
        ),
        lambda: subject.publish_frame(output / "frame.parquet", frame, output),
        lambda: subject.frame_sha256(frame, output, output),
    ):
        with pytest.raises(SimulatedCrash):
            call()
        temporary = created.pop()
        assert output not in temporary.parents
        assert report not in temporary.parents
        temporary.unlink()
        assert (_tree_snapshot(output), _tree_snapshot(report)) == before


def test_linked_scratch_crash_prefixes_are_recovered_without_content_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    frame = subject.pl.DataFrame(
        {
            "datetime": [subject.datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).with_columns(subject.pl.col("datetime").cast(subject.pl.Datetime("ms")))

    class SimulatedCrash(BaseException):
        pass

    original_unlink = subject.os.unlink

    def crash_before_unlink(prefix: str) -> None:
        def unlink(path: str | Path, *args: object, **kwargs: object) -> None:
            if Path(path).name.startswith(prefix):
                raise SimulatedCrash()
            original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(subject.os, "unlink", unlink)

    atomic_target = report / "atomic.json"
    crash_before_unlink(".acquire-")
    with pytest.raises(SimulatedCrash):
        subject.atomic_write(atomic_target, b'{"published":true}', scratch_root=report)
    assert atomic_target.read_bytes() == b'{"published":true}'
    assert atomic_target.stat().st_nlink == 2
    monkeypatch.setattr(subject.os, "unlink", original_unlink)
    subject.cleanup_scratch(report)
    assert atomic_target.stat().st_nlink == 1
    assert atomic_target.read_bytes() == b'{"published":true}'

    receipt_target = report / "exchangeInfo.json"
    monkeypatch.setattr(
        subject.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _StreamingResponse(b'{"symbols":[]}'),
    )
    crash_before_unlink(".partial-")
    with pytest.raises(SimulatedCrash):
        subject.fetch_receipt(
            "https://fapi.binance.com/fapi/v1/exchangeInfo", receipt_target, scratch_root=report
        )
    assert receipt_target.stat().st_nlink == 2
    assert not receipt_target.with_suffix(".json.receipt.json").exists()
    monkeypatch.setattr(subject.os, "unlink", original_unlink)
    subject.cleanup_scratch(report)
    assert receipt_target.stat().st_nlink == 1
    subject.fetch_receipt(
        "https://fapi.binance.com/fapi/v1/exchangeInfo", receipt_target, scratch_root=report
    )
    assert subject.request_receipt(receipt_target, "https://fapi.binance.com/fapi/v1/exchangeInfo")[
        "sha256"
    ] == subject.sha256(b'{"symbols":[]}')

    frame_target = output / "frame.parquet"
    crash_before_unlink(".acquire-")
    with pytest.raises(SimulatedCrash):
        subject.publish_frame(frame_target, frame, output)
    assert frame_target.stat().st_nlink == 2
    monkeypatch.setattr(subject.os, "unlink", original_unlink)
    subject.cleanup_scratch(output)
    assert frame_target.stat().st_nlink == 1
    assert subject.pl.read_parquet(frame_target).equals(frame)


def test_journal_recovers_every_pending_journal_fragment_prefix(tmp_path: Path) -> None:
    event = {"event": "write"}
    target = subject.canonical_bytes(event)
    for prefix in range(1, len(target)):
        report = tmp_path / f"report-{prefix}"
        report.mkdir()
        subject.immutable_json(report / "acquisition.journal.pending.json", event, report)
        (report / "acquisition.journal.jsonl").write_bytes(target[:prefix])
        subject.recover_pending_event(report)
        assert (report / "acquisition.journal.jsonl").read_bytes() == target
        assert not (report / "acquisition.journal.pending.json").exists()


def test_pending_event_recovers_its_matching_journal_fragment(tmp_path: Path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    event = {"event": "started", "plan_sha256": "a" * 64, "source_eligible": False}
    subject.immutable_json(report / "acquisition.journal.pending.json", event, report)
    target = subject.canonical_bytes(event)
    (report / "acquisition.journal.jsonl").write_bytes(target[:11])
    subject.recover_pending_event(report)
    assert (report / "acquisition.journal.jsonl").read_bytes() == target
    assert not (report / "acquisition.journal.pending.json").exists()


def test_pending_event_rejects_unrelated_fragment(tmp_path: Path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    subject.immutable_json(
        report / "acquisition.journal.pending.json", {"event": "started"}, report
    )
    (report / "acquisition.journal.jsonl").write_bytes(b'{"event":"other"')
    with pytest.raises(subject.AcquisitionError, match="fragment"):
        subject.recover_pending_event(report)


def test_hardlink_prefix_recovery_retires_only_owned_plan_temp(tmp_path: Path) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    plan = subject.canonical_bytes(subject.full_plan())
    run_id = subject.sha256(plan)
    subject.atomic_write(report / "plan.json", plan, scratch_root=report)
    subject.ownership(output, report, run_id)
    scratch = subject.scratch_directory(report)
    temp = scratch / ".acquire-plan"
    os.link(report / "plan.json", temp)
    subject.recover_owned_hardlink_prefixes(output, report, plan)
    assert not temp.exists()
    assert (report / "plan.json").stat().st_nlink == 1


def test_owned_run_lock_is_descriptor_backed(tmp_path: Path) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    plan = subject.canonical_bytes(subject.full_plan())
    subject.atomic_write(report / "plan.json", plan, scratch_root=report)
    subject.ownership(output, report, subject.sha256(plan))
    lock = subject.OwnedRunLock(report, exclusive=True)
    try:
        assert lock.fd >= 0
    finally:
        lock.close()


def test_journal_full_line_fsync_and_pending_retirement_are_retry_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report"
    report.mkdir()
    event = {"event": "write"}
    target = subject.canonical_bytes(event)
    subject.immutable_json(report / "acquisition.journal.pending.json", event, report)
    (report / "acquisition.journal.jsonl").write_bytes(target)

    class SimulatedCrash(BaseException):
        pass

    original_unlink = Path.unlink

    def crash_pending_retirement(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "acquisition.journal.pending.json":
            raise SimulatedCrash()
        original_unlink(path, *args, **kwargs)

    fsync_calls: list[int] = []
    original_fsync = subject.os.fsync

    def crash_before_journal_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        raise SimulatedCrash()

    monkeypatch.setattr(subject.os, "fsync", crash_before_journal_fsync)
    with pytest.raises(SimulatedCrash):
        subject.recover_pending_event(report)
    assert (report / "acquisition.journal.jsonl").read_bytes() == target
    assert (report / "acquisition.journal.pending.json").exists()
    monkeypatch.setattr(subject.os, "fsync", original_fsync)
    monkeypatch.setattr(Path, "unlink", crash_pending_retirement)
    with pytest.raises(SimulatedCrash):
        subject.recover_pending_event(report)
    monkeypatch.setattr(Path, "unlink", original_unlink)
    subject.journal(report, event)
    assert (report / "acquisition.journal.jsonl").read_bytes() == target
    assert not (report / "acquisition.journal.pending.json").exists()
    assert fsync_calls

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    subject.immutable_json(corrupt / "acquisition.journal.pending.json", event, corrupt)
    (corrupt / "acquisition.journal.jsonl").write_bytes(b'{"event":"other"')
    with pytest.raises(subject.AcquisitionError, match="fragment"):
        subject.journal(corrupt, event)


def test_verify_eligible_rejects_stale_source_manifest_after_complete_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _output, report = _eligible_fixture(tmp_path, monkeypatch)
    subject.atomic_write(report / "source_manifest.json", b"{}\n", replace=True)

    with pytest.raises(subject.AcquisitionError, match="source_manifest_stale"):
        subject.main(args)


def test_verify_eligible_rejects_tampered_journal_at_receipt_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    subject.atomic_write(
        report / "acquisition.journal.jsonl", b'{"event":"tampered"}\n', replace=True
    )
    subject.rebuild_manifest(output, report)

    with pytest.raises(subject.AcquisitionError, match="source_eligible_receipt_invalid"):
        subject.main(args)


def test_verify_eligible_rejects_tampered_eligible_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _output, report = _eligible_fixture(tmp_path, monkeypatch)
    subject.atomic_write(report / "source_eligible_receipt.json", b"{}\n", replace=True)

    with pytest.raises(subject.AcquisitionError, match="source_eligible_receipt_invalid"):
        subject.main(args)


def test_ownership_binds_the_exact_plan_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "safe_existing_directory", lambda _: [1, 1])
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    run_id = "a" * 64
    subject.ownership(output, report, run_id)
    subject.verify_ownership(output, report, run_id)
    with pytest.raises(subject.AcquisitionError, match="ownership"):
        subject.verify_ownership(output, report, "b" * 64)


def test_validation_helpers_reject_nonfinite_discontinuous_ohlcv() -> None:
    start = 1704067200000
    good = subject.pl.DataFrame(
        {
            "datetime": [
                subject.datetime.fromtimestamp(start / 1000, subject.UTC).replace(tzinfo=None)
            ],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [0.0],
        }
    ).with_columns(subject.pl.col("datetime").cast(subject.pl.Datetime("ms")))
    subject.assert_raw_frame(good, start, start + 1000)
    with pytest.raises(subject.AcquisitionError, match="continuity"):
        subject.assert_raw_frame(
            good.with_columns(
                (subject.pl.col("datetime") + subject.pl.duration(seconds=1)).alias("datetime")
            ),
            start,
            start + 1000,
        )
    with pytest.raises(subject.AcquisitionError, match="ohlcv"):
        subject.assert_raw_frame(
            good.with_columns(subject.pl.lit(float("nan")).alias("close")), start, start + 1000
        )


@pytest.mark.parametrize(("tree_name", "prefix"), [("output", "output"), ("report", "report")])
def test_manifest_hashes_nested_tree_through_inventory_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tree_name: str, prefix: str
) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    root = output if tree_name == "output" else report
    nested = root / "nested"
    nested.mkdir()
    (nested / "artifact").write_bytes(b"original")
    original_file_digest = subject.hashlib.file_digest
    replaced = False

    def replace_nested_tree(source: io.BufferedReader, algorithm: str) -> object:
        nonlocal replaced
        if not replaced:
            replaced = True
            nested.rename(root / "nested-original")
            nested.mkdir()
            (nested / "artifact").write_bytes(b"replacement")
            (nested / "replacement-extra").write_bytes(b"extra")
        return original_file_digest(source, algorithm)

    monkeypatch.setattr(subject.hashlib, "file_digest", replace_nested_tree)

    manifest = subject.manifest_value(output, report)

    assert manifest["artifacts"] == [
        {
            "path": f"{prefix}/nested/artifact",
            "sha256": hashlib.sha256(b"original").hexdigest(),
        }
    ]


def test_manifest_rejects_same_inode_same_size_mutation_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, report = tmp_path / "output", tmp_path / "report"
    output.mkdir()
    report.mkdir()
    artifact = output / "artifact"
    artifact.write_bytes(b"before")
    original_file_digest = subject.hashlib.file_digest

    def mutate_during_hash(source: io.BufferedReader, algorithm: str) -> object:
        before = artifact.stat()
        artifact.write_bytes(b"after!")
        os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        after = artifact.stat()
        assert (after.st_dev, after.st_ino, after.st_size) == (
            before.st_dev,
            before.st_ino,
            before.st_size,
        )
        return original_file_digest(source, algorithm)

    monkeypatch.setattr(subject.hashlib, "file_digest", mutate_during_hash)

    with pytest.raises(subject.AcquisitionError, match=r"^complete_inventory_unsafe_object$"):
        subject.manifest_value(output, report)


def test_main_preserves_ambient_plan_and_verify_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Lock:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    class NoopRunLock:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(subject, "assert_input_paths", lambda *_: None)
    monkeypatch.setattr(subject, "assert_roots", lambda *_: None)
    monkeypatch.setattr(subject, "load_contract", lambda _: [])
    monkeypatch.setattr(subject, "load_evidence", lambda _: None)
    monkeypatch.setattr(subject, "owned_run_lock", lambda *_, **__: NoopRunLock())
    monkeypatch.setattr(subject, "verify_eligible", lambda *_: None)
    ambient = Lock()
    token = subject._ACTIVE_EXECUTE_LOCK.set(ambient)
    try:
        common = [
            "--contract-manifest",
            str(tmp_path / "contract.json"),
            "--availability-evidence",
            str(tmp_path / "evidence.json"),
            "--output-root",
            str(tmp_path / "output"),
        ]
        assert subject.main([*common, "--report-dir", str(tmp_path / "plan-report")]) == 0
        assert subject._ACTIVE_EXECUTE_LOCK.get() is ambient
        assert (
            subject.main(
                [*common, "--report-dir", str(tmp_path / "verify-report"), "--verify-eligible"]
            )
            == 0
        )
        assert subject._ACTIVE_EXECUTE_LOCK.get() is ambient
        assert ambient.close_count == 0
    finally:
        subject._ACTIVE_EXECUTE_LOCK.reset(token)


def test_release_execute_lock_closes_nested_execute_lock_once() -> None:
    class Lock:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    ambient = Lock()
    local = Lock()
    token = subject._ACTIVE_EXECUTE_LOCK.set(ambient)
    try:

        @subject.release_execute_lock
        def nested_execute() -> None:
            assert subject._ACTIVE_EXECUTE_LOCK.get() is None
            subject._ACTIVE_EXECUTE_LOCK.set(local)

        @subject.release_execute_lock
        def execute() -> None:
            assert subject._ACTIVE_EXECUTE_LOCK.get() is None
            nested_execute()
            assert subject._ACTIVE_EXECUTE_LOCK.get() is None

        execute()
        assert local.close_count == 1
        assert subject._ACTIVE_EXECUTE_LOCK.get() is ambient
        assert ambient.close_count == 0
    finally:
        subject._ACTIVE_EXECUTE_LOCK.reset(token)


def test_main_isolates_verifier_context_through_deep_execute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "assert_input_paths", lambda *_: None)
    monkeypatch.setattr(subject, "assert_roots", lambda *_: None)
    monkeypatch.setattr(subject, "load_contract", lambda _: [])
    monkeypatch.setattr(subject, "load_evidence", lambda _: None)
    observed: list[int | None] = []

    def observe_context(*_: object) -> None:
        observed.append(subject._VERIFIER_CODE_FD.get())
        raise subject.AcquisitionError("deep_execute_context")

    monkeypatch.setattr(subject, "recover_owned_hardlink_prefixes", observe_context)
    descriptor = os.open(_MODULE, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    token = subject._VERIFIER_CODE_FD.set(descriptor)
    try:
        with pytest.raises(subject.AcquisitionError, match=r"^deep_execute_context$"):
            subject.main(
                [
                    "--contract-manifest",
                    str(tmp_path / "contract.json"),
                    "--availability-evidence",
                    str(tmp_path / "evidence.json"),
                    "--output-root",
                    str(tmp_path / "output"),
                    "--report-dir",
                    str(tmp_path / "report"),
                    "--execute",
                ]
            )
        assert observed == [None]
        assert subject._VERIFIER_CODE_FD.get() == descriptor
    finally:
        subject._VERIFIER_CODE_FD.reset(token)
        os.close(descriptor)


def test_code_hash_uses_inherited_verifier_descriptor() -> None:
    descriptor = os.open(_MODULE, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    token = subject._VERIFIER_CODE_FD.set(descriptor)
    try:
        assert subject.code_hash() == subject.file_sha256(_MODULE)
    finally:
        subject._VERIFIER_CODE_FD.reset(token)
        os.close(descriptor)


def test_stable_file_rejects_same_size_in_place_mutation(tmp_path: Path) -> None:
    target = tmp_path / "source"
    target.write_bytes(b"before")
    original_stat = target.stat()

    with (
        pytest.raises(subject.AcquisitionError, match=r"^unsafe_file$"),
        subject.stable_file(target) as source,
    ):
        target.write_bytes(b"after!")
        os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1))
        assert source.read() == b"after!"


@pytest.mark.parametrize("action", [[], ["--execute"]])
def test_main_restores_verifier_context_after_non_verifier_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: list[str]
) -> None:
    def reject_input_paths(*_: object) -> None:
        raise subject.AcquisitionError("test_input_failure")

    monkeypatch.setattr(subject, "assert_input_paths", reject_input_paths)
    descriptor = os.open(_MODULE, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    token = subject._VERIFIER_CODE_FD.set(descriptor)
    try:
        with pytest.raises(subject.AcquisitionError, match=r"^test_input_failure$"):
            subject.main(
                [
                    "--contract-manifest",
                    str(tmp_path / "contract.json"),
                    "--availability-evidence",
                    str(tmp_path / "evidence.json"),
                    "--output-root",
                    str(tmp_path / "output"),
                    "--report-dir",
                    str(tmp_path / "report"),
                    *action,
                ]
            )
        assert subject._VERIFIER_CODE_FD.get() == descriptor
    finally:
        subject._VERIFIER_CODE_FD.reset(token)
        os.close(descriptor)


def test_compact_plan_binds_host_storage_contract() -> None:
    plan = subject.full_plan()
    assert plan["schema"] == "alpha_max_official_acquisition_plan.v4"
    assert plan["storage_contract"] == {
        "host_reserve_path": "/mnt/c",
        "host_reserve_bytes": 21_474_836_480,
        "max_live_archives": 1,
        "archive_retention": "retired_after_double_derivation",
    }


def test_completed_evidence_rejects_noncanonical_retained_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    receipt = report / "provenance/archives/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip.receipt.json"
    receipt.write_text(json.dumps(subject.read_json(receipt), indent=2) + "\n")
    contract = subject.Contract(
        "BTCUSDT", 1704067200000, 1704067201000, 1704067200000, 1704067200001
    )
    with pytest.raises(subject.AcquisitionError, match="retired_archive_request_receipt_invalid"):
        subject.validate_complete(output, [contract], report / "provenance", report)


def test_completed_evidence_rejects_retired_body_reappearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    archive_path = report / "provenance/archives/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip"
    archive_path.write_bytes(b"body-must-remain-retired")
    contract = subject.Contract(
        "BTCUSDT", 1704067200000, 1704067201000, 1704067200000, 1704067200001
    )
    with pytest.raises(subject.AcquisitionError, match="retired_archive_body_present"):
        subject.validate_complete(output, [contract], report / "provenance", report)


def test_completed_evidence_rejects_derivation_chain_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _args, output, report = _eligible_fixture(tmp_path, monkeypatch)
    derivation = report / "provenance/archive-evidence/BTCUSDT/2024-01.derivation.json"
    value = subject.read_json(derivation)
    value["prior_derivation_receipt_sha256"] = "0" * 64
    subject.atomic_write(
        derivation, subject.canonical_bytes(value), replace=True, scratch_root=report
    )
    contract = subject.Contract(
        "BTCUSDT", 1704067200000, 1704067201000, 1704067200000, 1704067200001
    )
    with pytest.raises(subject.AcquisitionError, match="archive_derivation_receipt_invalid"):
        subject.validate_complete(output, [contract], report / "provenance", report)


@pytest.mark.parametrize("required_bytes", (-1, True))
def test_host_reserve_rejects_invalid_requirement(required_bytes: object) -> None:
    with pytest.raises(subject.AcquisitionError, match="host_reserve_requirement_invalid"):
        subject.assert_host_reserve(required_bytes)


def test_host_reserve_accounts_for_the_pending_write(monkeypatch: pytest.MonkeyPatch) -> None:
    required = 4096
    monkeypatch.setattr(
        subject.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=subject.HOST_RESERVE_BYTES + required),
    )
    with pytest.raises(subject.AcquisitionError, match="host_reserve_exhausted"):
        subject.assert_host_reserve(required)

    monkeypatch.setattr(
        subject.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=subject.HOST_RESERVE_BYTES + required + 1),
    )
    subject.assert_host_reserve(required)


def test_atomic_write_charges_exact_payload_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    charged: list[int] = []
    monkeypatch.setattr(subject, "assert_host_reserve", charged.append)
    target = tmp_path / "receipt.json"
    subject.atomic_write(target, b"payload", scratch_root=tmp_path)
    assert charged == [7]
    assert target.read_bytes() == b"payload"
