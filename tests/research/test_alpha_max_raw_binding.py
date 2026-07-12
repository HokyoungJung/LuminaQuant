from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType

import polars as pl
import pytest

import lumina_quant.research.alpha_max_engine_runner as runner
import lumina_quant.research.alpha_max_evidence as evidence
from lumina_quant.research.alpha_max_evidence import (
    ALPHA_MAX_CANDIDATE_SYMBOLS,
    seal_alpha_max_root_tree,
)


def _write_raw_root(root: Path, root_id: str = "purge") -> None:
    start, end = evidence._ROOT_INTERVALS[root_id]
    timestamps = pl.datetime_range(
        start,
        end,
        interval="1s",
        closed="left",
        time_zone="UTC",
        eager=True,
    )
    row_count = len(timestamps)
    for symbol in ALPHA_MAX_CANDIDATE_SYMBOLS:
        target = root / "market_ohlcv_1s" / "binance" / symbol / f"{start:%Y-%m}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "datetime": timestamps,
                "symbol": [symbol] * row_count,
                "exchange": ["binance"] * row_count,
                "open": [100.0] * row_count,
                "high": [101.0] * row_count,
                "low": [99.0] * row_count,
                "close": [100.0] * row_count,
                "volume": [1.0] * row_count,
            }
        ).write_parquet(target)


def _sealed_raw_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    start = evidence._ROOT_INTERVALS["purge"][0]
    monkeypatch.setitem(
        evidence._ROOT_INTERVALS,
        "purge",
        (start, start + timedelta(seconds=20)),
    )
    root = (tmp_path / "raw").resolve()
    _write_raw_root(root)
    seal = seal_alpha_max_root_tree(
        "purge",
        "raw",
        root,
        availability_start_by_symbol=MappingProxyType(
            dict.fromkeys(ALPHA_MAX_CANDIDATE_SYMBOLS, evidence._ROOT_INTERVALS["warmup"][0])
        ),
        availability_end_by_symbol=MappingProxyType(
            dict.fromkeys(
                ALPHA_MAX_CANDIDATE_SYMBOLS,
                evidence._ROOT_INTERVALS["historical_exposed_evaluation"][1],
            )
        ),
    )
    entry = next(
        candidate
        for candidate in seal.entries
        if runner._alpha_max_raw_entry_symbol(
            candidate.relative_path,
            exchange=seal.exchange,
        )
        == "BTCUSDT"
    )
    target = root / entry.relative_path
    return root, seal, entry, target


def _write_replacement(target: Path, *, close: float = 999.0) -> None:
    start, end = evidence._ROOT_INTERVALS["purge"]
    timestamps = pl.datetime_range(
        start,
        end,
        interval="1s",
        closed="left",
        time_zone="UTC",
        eager=True,
    )
    row_count = len(timestamps)
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "datetime": timestamps,
            "symbol": ["BTCUSDT"] * row_count,
            "exchange": ["binance"] * row_count,
            "open": [close] * row_count,
            "high": [close] * row_count,
            "low": [close] * row_count,
            "close": [close] * row_count,
            "volume": [1.0] * row_count,
        }
    ).write_parquet(target)


def test_raw_admission_reads_verified_bytes_without_path_scan(tmp_path, monkeypatch) -> None:
    _root, seal, _entry, _target = _sealed_raw_fixture(tmp_path, monkeypatch)

    def forbid_path_scan(*_args, **_kwargs):
        raise AssertionError("path scan must not be used after sealing")

    monkeypatch.setattr(pl, "scan_parquet", forbid_path_scan)

    daily, completed, integrity = runner._alpha_max_load_raw_admission_summary(
        seal,
        symbol="BTCUSDT",
        include_quote_notional=False,
    )

    assert daily == ()
    assert integrity is True
    assert len(completed) == 1


def test_raw_admission_rejects_post_seal_content_replacement(tmp_path, monkeypatch) -> None:
    _root, seal, _entry, target = _sealed_raw_fixture(tmp_path, monkeypatch)
    replacement = target.with_name("replacement.parquet")
    _write_replacement(replacement)
    os.replace(replacement, target)

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_admission_raw_partition_read_failed",
    ):
        runner._alpha_max_load_raw_admission_summary(
            seal,
            symbol="BTCUSDT",
            include_quote_notional=False,
        )


def test_raw_admission_rejects_post_seal_symlink(tmp_path, monkeypatch) -> None:
    _root, seal, _entry, target = _sealed_raw_fixture(tmp_path, monkeypatch)
    original = target.with_name("original.parquet")
    target.rename(original)
    target.symlink_to(original.name)

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_admission_raw_partition_read_failed",
    ):
        runner._alpha_max_load_raw_admission_summary(
            seal,
            symbol="BTCUSDT",
            include_quote_notional=False,
        )


def test_raw_admission_rejects_post_seal_hardlink(tmp_path, monkeypatch) -> None:
    _root, seal, _entry, target = _sealed_raw_fixture(tmp_path, monkeypatch)
    os.link(target, target.with_name("hardlink.parquet"))

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_admission_raw_partition_read_failed",
    ):
        runner._alpha_max_load_raw_admission_summary(
            seal,
            symbol="BTCUSDT",
            include_quote_notional=False,
        )


def test_raw_admission_rejects_root_path_swap_before_binding(tmp_path, monkeypatch) -> None:
    root, seal, entry, _target = _sealed_raw_fixture(tmp_path, monkeypatch)
    original_root = tmp_path / "original-root"
    root.rename(original_root)
    _write_replacement(root / entry.relative_path)

    with pytest.raises(
        runner.AlphaMaxRuntimeContractError,
        match="alpha_max_admission_raw_partition_read_failed",
    ):
        runner._alpha_max_load_raw_admission_summary(
            seal,
            symbol="BTCUSDT",
            include_quote_notional=False,
        )


def test_sealed_raw_reader_retains_original_root_across_path_swap(tmp_path, monkeypatch) -> None:
    root, seal, entry, _target = _sealed_raw_fixture(tmp_path, monkeypatch)
    reader = runner._AlphaMaxSealedRawReader(seal)
    original_root = tmp_path / "original-root"
    root.rename(original_root)
    _write_replacement(root / entry.relative_path)
    try:
        frame = reader.read_entry(entry)
    finally:
        reader.close()

    assert frame.get_column("close").to_list() == [100.0] * 20
