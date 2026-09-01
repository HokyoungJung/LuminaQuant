#!/usr/bin/env python3
"""Fail-closed validation receipt for a frozen research-data contract."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import polars as pl

from lumina_quant.compute.ohlcv_validation import validate_ohlcv_frame
from lumina_quant.data.symbol_lifecycle import (
    registry_sha256,
    validate_fold_membership_manifest,
    validate_symbol_lifecycle_registry,
)
from lumina_quant.market_data import (
    load_futures_feature_points_from_db,
    load_strict_ohlcv_route,
    normalize_timeframe_token,
    timeframe_to_milliseconds,
)

_SHA = set("0123456789abcdef")
_OHLCV_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")
_FUNDING_COLUMNS = ("timestamp_ms", "funding_rate", "source")
_MAX_ISSUES = 100
_BINANCE_FUNDING_EXCHANGES = {
    "fapi.binance.com": "binance",
    "dapi.binance.com": "binance",
}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _issue(receipt: dict[str, Any], code: str, **detail: Any) -> None:
    if len(receipt["issues"]) < _MAX_ISSUES:
        receipt["issues"].append({"code": code, **detail})


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty exact string")
    return value


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (value <= 0 if positive else value < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _sha(value: Any, field: str) -> str:
    value = _string(value, field)
    if len(value) != 64 or any(character not in _SHA for character in value):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _timestamp_ms(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(normalized.astimezone(UTC).timestamp() * 1000)
    return None


def _frame_digest(frame: Any, columns: tuple[str, ...]) -> str:
    if not isinstance(frame, pl.DataFrame) or tuple(frame.columns) != columns:
        raise ValueError("frame schema does not exactly match contract")
    digest = hashlib.sha256()
    for row in frame.iter_rows():
        encoded: list[Any] = []
        for value in row:
            if isinstance(value, datetime):
                timestamp = _timestamp_ms(value)
                if timestamp is None:
                    raise ValueError("datetime must be timezone-aware")
                encoded.append({"timestamp_ms": timestamp})
            elif isinstance(value, float):
                encoded.append(
                    {"float": value.hex()}
                    if math.isfinite(value)
                    else {"nonfinite_float": repr(value)}
                )
            elif isinstance(value, bool):
                encoded.append({"bool": value})
            elif value is None or isinstance(value, (int, str)):
                encoded.append(value)
            else:
                raise ValueError("unsupported frame value")
        digest.update(_canonical_bytes(encoded))
        digest.update(b"\n")
    return digest.hexdigest()


def _sealed_prefix_matches(
    seal: Any,
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    expected_id: str,
) -> bool:
    if not isinstance(seal, Mapping) or set(seal) != {
        "id",
        "row_count",
        "row_value_sha256",
    }:
        return False
    count = seal["row_count"]
    return (
        isinstance(count, int)
        and not isinstance(count, bool)
        and 0 < count <= frame.height
        and seal["id"] == expected_id
        and isinstance(seal["row_value_sha256"], str)
        and len(seal["row_value_sha256"]) == 64
        and all(character in _SHA for character in seal["row_value_sha256"])
        and seal["row_value_sha256"] == _frame_digest(frame.slice(0, count), columns)
    )


def _direct_physical_loader(
    db_path: str | Path,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_exclusive_ms: int,
    **_: Any,
) -> pl.DataFrame:
    """Read physical partitions without normalizing source order or duplicate rows."""
    base = (
        Path(db_path)
        / f"exchange={exchange}"
        / f"symbol={symbol.replace('/', '')}"
        / f"timeframe={timeframe}"
    )
    paths = sorted(base.glob("date=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"physical OHLCV partitions are missing: {base}")
    first_day = datetime.fromtimestamp(start_ms / 1000, UTC).date()
    last_day = datetime.fromtimestamp((end_exclusive_ms - 1) / 1000, UTC).date()
    expected_dates = {
        (first_day.fromordinal(day)).isoformat()
        for day in range(first_day.toordinal(), last_day.toordinal() + 1)
    }
    paths = [path for path in paths if path.parent.name.removeprefix("date=") in expected_dates]
    present_dates = {path.parent.name.removeprefix("date=") for path in paths}
    missing_dates = sorted(expected_dates - present_dates)
    if missing_dates:
        raise FileNotFoundError(
            f"physical OHLCV partitions are missing dates: {', '.join(missing_dates)}"
        )
    frames = [pl.read_parquet(path) for path in paths]
    frame = pl.concat(frames, how="diagonal_relaxed")
    if "datetime" not in frame.columns:
        return frame
    timestamps = [_timestamp_ms(value) for value in frame.get_column("datetime").to_list()]
    if any(timestamp is None for timestamp in timestamps):
        return frame
    owned = pl.Series([start_ms <= timestamp < end_exclusive_ms for timestamp in timestamps])
    return frame.filter(owned)


def _missing_counts(expected: list[int], actual: set[int]) -> tuple[int, int, int]:
    missing = [point not in actual for point in expected]
    prefix = next((index for index, absent in enumerate(missing) if not absent), len(missing))
    suffix = next(
        (index for index, absent in enumerate(reversed(missing[prefix:])) if not absent),
        len(missing) - prefix,
    )
    return prefix, sum(missing) - prefix - suffix, suffix


def _canonical_mode(mode: str) -> str:
    aliases = {
        "pre-append": "pre_append",
        "pre_append": "pre_append",
        "post-append-strict": "post_append_strict",
        "post_append_strict": "post_append_strict",
    }
    if mode not in aliases:
        raise ValueError("mode is invalid")
    return aliases[mode]


def _load_json(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], bytes, Path]:
    if isinstance(value, Mapping):
        return dict(value), _canonical_bytes(value), Path.cwd()
    path = Path(value)
    raw = path.read_bytes()
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("JSON artifact must be an object")
    return loaded, raw, path.parent


def _contract_core_sha256(contract: Mapping[str, Any]) -> str:
    core = dict(contract)
    core["pre_append_receipt_sha256"] = None
    return _digest(_canonical_bytes(core))


def _local_path(base: Path, value: Any, field: str) -> Path:
    token = _string(value, field)
    if "://" in token:
        raise ValueError(f"{field} must be a local path")
    path = Path(token)
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise ValueError(f"{field} is missing")
    return path


def _valid_origin(source_uri: Any, data_kind: str, exchange: str) -> bool:
    parsed = urlparse(_string(source_uri, "receipt.source_uri"))
    if data_kind == "canonical_inventory":
        return parsed.scheme == "file" and bool(parsed.path) and exchange == "binance"
    return (
        parsed.scheme == "https"
        and _BINANCE_FUNDING_EXCHANGES.get(parsed.hostname or "") == exchange
    )


def _validate_inventory(payload: Any, root: Path, symbols: set[tuple[str, str]]) -> None:
    required = {
        "artifact_kind",
        "schemaVersion",
        "market_root",
        "pairs",
        "snapshot_inventory",
        "source_inventory_sha256",
        "source_root",
        "synthetic_source_contract",
    }
    if not isinstance(payload, Mapping) or not required <= set(payload):
        raise ValueError("canonical inventory payload has no recovery inventory decision shape")
    if payload["artifact_kind"] != "recovery_inventory_decision" or payload["schemaVersion"] != 1:
        raise ValueError("canonical inventory payload schema is invalid")
    market_root = Path(_string(payload["market_root"], "inventory.market_root")).resolve()
    source_root = Path(_string(payload["source_root"], "inventory.source_root")).resolve()
    if market_root != root.resolve() or not source_root.is_dir() or source_root == market_root:
        raise ValueError("canonical inventory payload does not bind separated data roots")
    snapshot = payload["snapshot_inventory"]
    if not isinstance(snapshot, Mapping):
        raise ValueError("canonical inventory snapshot proof is invalid")
    before = _sha(snapshot.get("before_scans_sha256"), "inventory.before_sha256")
    after = _sha(snapshot.get("after_scans_sha256"), "inventory.after_sha256")
    source_hash = _sha(payload["source_inventory_sha256"], "inventory.source_inventory_sha256")
    if snapshot.get("stable_across_scans") is not True or before != after or before != source_hash:
        raise ValueError("canonical inventory snapshot is not stable or source-identical")
    synthetic = payload["synthetic_source_contract"]
    if (
        not isinstance(synthetic, Mapping)
        or synthetic.get("passed") is not True
        or synthetic.get("selected_root_csv_count") != 0
    ):
        raise ValueError("canonical inventory synthetic-source exclusion failed")
    pairs = payload["pairs"]
    if not isinstance(pairs, list):
        raise ValueError("canonical inventory pairs are invalid")
    observed = {
        (item.get("symbol"), item.get("timeframe")) for item in pairs if isinstance(item, Mapping)
    }
    if not symbols <= observed:
        raise ValueError("canonical inventory omits requested symbol/timeframe")


def _payload_funding_row(item: Mapping[str, Any]) -> tuple[str, int, float]:
    symbol = _string(item.get("symbol"), "funding.symbol")
    timestamps: list[int] = []
    for field in ("timestamp_ms", "fundingTime"):
        if field not in item:
            continue
        timestamp = _timestamp_ms(item[field])
        if timestamp is None:
            raise ValueError("funding timestamp is invalid")
        timestamps.append(timestamp)
    if not timestamps:
        raise ValueError("funding timestamp is invalid")
    if len(set(timestamps)) != 1:
        raise ValueError("funding timestamp aliases conflict")

    rates: list[float] = []
    for field in ("funding_rate", "fundingRate"):
        if field not in item:
            continue
        rate = item[field]
        if isinstance(rate, str):
            try:
                rate = float(rate)
            except ValueError as exc:
                raise ValueError("funding rate is invalid") from exc
        rates.append(_finite_number(rate, "funding rate"))
    if not rates:
        raise ValueError("funding rate is invalid")
    if len(set(rates)) != 1:
        raise ValueError("funding rate aliases conflict")
    return symbol, timestamps[0], rates[0]


def _validate_provenance(
    contract: Mapping[str, Any], base: Path, root: Path
) -> dict[str, dict[str, Any]]:
    rows = contract.get("provenance")
    if not isinstance(rows, list) or not rows:
        raise ValueError("provenance must be non-empty")
    requested_pairs = {
        (item.get("symbol"), item.get("timeframe"))
        for item in contract.get("ohlcv_series", [])
        if isinstance(item, Mapping)
    }
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "id",
            "kind",
            "exchange",
            "receipt_file",
            "sha256",
        }:
            raise ValueError("provenance fields are invalid")
        identifier = _string(row["id"], "provenance.id")
        kind = _string(row["kind"], "provenance.kind")
        exchange = _string(row["exchange"], "provenance.exchange")
        if identifier in result:
            raise ValueError("duplicate provenance id")
        receipt_path = _local_path(base, row["receipt_file"], "provenance.receipt_file")
        if _digest(receipt_path.read_bytes()) != _sha(row["sha256"], "provenance.sha256"):
            raise ValueError("provenance receipt hash mismatch")
        envelope = json.loads(receipt_path.read_bytes())
        required = {
            "artifact_kind",
            "schema_version",
            "provenance_id",
            "provenance_kind",
            "exchange",
            "data_kind",
            "source_uri",
            "symbols",
            "start_ms",
            "end_exclusive_ms",
            "captured_at",
            "payload",
        }
        allowed = required | {"timeframes"}
        if not isinstance(envelope, Mapping) or set(envelope) not in (required, allowed):
            raise ValueError("provenance receipt envelope is invalid")
        if (
            envelope["artifact_kind"] != "research_data_provenance_receipt"
            or envelope["schema_version"] != 1
        ):
            raise ValueError("provenance receipt envelope is invalid")
        if (envelope["provenance_id"], envelope["provenance_kind"], envelope["exchange"]) != (
            identifier,
            kind,
            exchange,
        ):
            raise ValueError("provenance receipt identity is not bound")
        data_kind = envelope["data_kind"]
        if data_kind not in {"canonical_inventory", "official_funding"} or not _valid_origin(
            envelope["source_uri"], data_kind, exchange
        ):
            raise ValueError("provenance source URI is not an approved origin")
        symbols = envelope["symbols"]
        if (
            not isinstance(symbols, list)
            or not symbols
            or any(not isinstance(value, str) for value in symbols)
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError("provenance receipt symbols are invalid")
        timeframes = envelope.get("timeframes", [])
        if (
            not isinstance(timeframes, list)
            or any(not isinstance(value, str) for value in timeframes)
            or len(set(timeframes)) != len(timeframes)
        ):
            raise ValueError("provenance receipt timeframes are invalid")
        start = _integer(envelope["start_ms"], "receipt.start_ms")
        end = _integer(envelope["end_exclusive_ms"], "receipt.end_exclusive_ms")
        if end <= start:
            raise ValueError("provenance receipt interval is invalid")
        captured = _string(envelope["captured_at"], "receipt.captured_at")
        if datetime.fromisoformat(captured.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("provenance receipt capture time must be timezone-aware")
        payload = envelope["payload"]
        if not isinstance(payload, Mapping) or set(payload) != {"path", "sha256"}:
            raise ValueError("provenance payload is invalid")
        payload_path = _local_path(receipt_path.parent, payload["path"], "provenance.payload.path")
        if _digest(payload_path.read_bytes()) != _sha(
            payload["sha256"], "provenance.payload.sha256"
        ):
            raise ValueError("provenance payload hash mismatch")
        parsed = json.loads(payload_path.read_bytes())
        official_funding_rows: list[tuple[int, str, int, float]] = []
        if data_kind == "canonical_inventory":
            if kind != "canonical_local_copy":
                raise ValueError("canonical inventory provenance kind is invalid")
            _validate_inventory(parsed, root, requested_pairs)
        else:
            if (
                kind != "official_exchange_api_receipt"
                or not isinstance(parsed, list)
                or not parsed
            ):
                raise ValueError("official funding payload is invalid")
            for payload_index, item in enumerate(parsed):
                if not isinstance(item, Mapping):
                    raise ValueError("official funding payload semantics are invalid")
                funding_row = _payload_funding_row(item)
                symbol, timestamp, _ = funding_row
                if symbol not in symbols or not start <= timestamp < end:
                    raise ValueError("official funding payload semantics are invalid")
                official_funding_rows.append((payload_index, *funding_row))
        result[identifier] = {
            **dict(row),
            "envelope": dict(envelope),
            "official_funding_rows": official_funding_rows,
        }
    return result


def _validate_lifecycle(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = validate_symbol_lifecycle_registry(value["registry"])
    membership = validate_fold_membership_manifest(registry, value["fold_membership"])
    return (
        {row["symbol"]: row for row in registry["symbols"]},
        {row["fold_id"]: row for row in membership["folds"]},
    )


def _manifest_locator(value: Any) -> str:
    return "<inline>" if isinstance(value, Mapping) else str(value)


def _base_receipt(
    db_path: str | Path,
    contract_manifest: Any,
    lifecycle_manifest: Any,
    mode: str,
) -> dict[str, Any]:
    return {
        "artifact_kind": "research_data_contract_validation",
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "contract_id": None,
        "mode": mode,
        "decision_class": "admission" if mode == "post_append_strict" else "triage",
        "decision": "STOP",
        "passed": False,
        "admission_eligible": False,
        "validation_layer": ["physical", "repository_view"],
        "input": {
            "db_path": str(db_path),
            "contract_manifest": _manifest_locator(contract_manifest),
            "lifecycle_manifest": _manifest_locator(lifecycle_manifest),
        },
        "issues": [],
        "d04_status": {"passed": False},
        "series": [],
        "funding": [],
        "seals": {},
    }


def _check_series_common(
    raw: Mapping[str, Any],
    common: Mapping[str, Any],
    folds: Mapping[str, Any],
    symbols: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    funding: bool = False,
) -> tuple[str, str, int, int]:
    exchange = _string(raw["exchange"], "series.exchange")
    symbol = _string(raw["symbol"], "series.symbol")
    start = _integer(raw["start_ms"], "series.start_ms")
    end = _integer(raw["end_exclusive_ms"], "series.end_exclusive_ms")
    if (start, end) != (common["start_ms"], common["end_exclusive_ms"]):
        raise ValueError("series must use the exact common owned interval")
    lifecycle = symbols.get(symbol)
    if (
        lifecycle is None
        or start < lifecycle["onboard_ms"]
        or (lifecycle["delivery_ms"] is not None and end > lifecycle["delivery_ms"])
    ):
        raise ValueError("series is outside lifecycle")
    identifiers = raw["provenance_ids"]
    if (
        not isinstance(identifiers, list)
        or not identifiers
        or len(identifiers) != len(set(identifiers))
    ):
        raise ValueError("provenance references are invalid")
    for provenance_id in identifiers:
        item = provenance.get(provenance_id)
        if item is None or item["exchange"] != exchange:
            raise ValueError("provenance references are invalid")
        envelope = item["envelope"]
        if symbol not in envelope["symbols"] or not (
            envelope["start_ms"] <= start and end <= envelope["end_exclusive_ms"]
        ):
            raise ValueError("provenance does not cover the owned interval")
        if not funding and (
            envelope["data_kind"] != "canonical_inventory"
            or raw["timeframe"] not in envelope.get("timeframes", [])
        ):
            raise ValueError("canonical inventory does not identify the requested timeframe")
    fold_ids = raw["fold_ids"]
    if not isinstance(fold_ids, list) or not fold_ids or len(fold_ids) != len(set(fold_ids)):
        raise ValueError("fold references are invalid")
    for fold_id in fold_ids:
        fold = folds.get(fold_id)
        if (
            fold is None
            or symbol not in fold["eligible_symbols"]
            or not (fold["start_ms"] <= start and end <= fold["end_ms"])
        ):
            raise ValueError("fold does not own the complete series interval")
    return exchange, symbol, start, end


def _seal_entry(identifier: str, frame: pl.DataFrame, columns: tuple[str, ...]) -> dict[str, Any]:
    return {
        "id": identifier,
        "row_count": frame.height,
        "row_value_sha256": _frame_digest(frame, columns),
    }


def _funding_settlement_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("funding_rate").is_not_null())


def _require_zero_metrics(row: Mapping[str, Any], fields: tuple[str, ...], scope: str) -> None:
    for field in fields:
        if _integer(row.get(field), f"{scope}.{field}") != 0:
            raise ValueError(f"{scope} contains nonzero invalid metrics")


def _validate_pre_ohlcv_metric(
    row: Mapping[str, Any], declared: Mapping[str, Any]
) -> tuple[int, str]:
    if set(row) != {
        "id",
        "passed",
        "safe_tail_append",
        "rows",
        "invalid_timestamp_count",
        "duplicate_count",
        "nonmonotone_count",
        "nonfinite_count",
        "off_grid_count",
        "out_of_range_count",
        "expected_count",
        "expected_grid_gap_count",
        "missing_prefix_count",
        "interior_gap_count",
        "missing_tail_count",
        "repository_rows",
        "repository_invalid_timestamp_count",
        "repository_duplicate_count",
        "repository_nonmonotone_count",
        "repository_nonfinite_count",
        "repository_off_grid_count",
        "repository_out_of_range_count",
        "repository_expected_count",
        "repository_expected_grid_gap_count",
        "repository_missing_prefix_count",
        "repository_interior_gap_count",
        "repository_missing_tail_count",
        "physical_row_value_sha256",
        "repository_row_value_sha256",
        "physical_repository_mismatch_count",
        "physical_ohlcv_validation",
        "repository_ohlcv_validation",
    }:
        raise ValueError("pre-append OHLCV metric schema is invalid")
    scope = "pre.ohlcv"
    if row.get("safe_tail_append") is not True:
        raise ValueError("pre-append OHLCV metric is not safe-tail")
    start = _integer(declared.get("start_ms"), "contract.ohlcv.start_ms")
    end = _integer(declared.get("end_exclusive_ms"), "contract.ohlcv.end_exclusive_ms")
    step = _integer(declared.get("step_ms"), "contract.ohlcv.step_ms", positive=True)
    if end <= start or (end - start) % step:
        raise ValueError("contract OHLCV interval is invalid")
    expected = (end - start) // step
    rows = _integer(row.get("rows"), f"{scope}.rows", positive=True)
    tail = _integer(row.get("missing_tail_count"), f"{scope}.missing_tail_count")
    repository_rows = _integer(
        row.get("repository_rows"), f"{scope}.repository_rows", positive=True
    )
    repository_tail = _integer(
        row.get("repository_missing_tail_count"),
        f"{scope}.repository_missing_tail_count",
    )
    if (
        _integer(row.get("expected_count"), f"{scope}.expected_count", positive=True) != expected
        or _integer(
            row.get("repository_expected_count"),
            f"{scope}.repository_expected_count",
            positive=True,
        )
        != expected
        or rows + tail != expected
        or repository_rows != rows
        or repository_tail != tail
        or _integer(
            row.get("expected_grid_gap_count"),
            f"{scope}.expected_grid_gap_count",
        )
        != tail
        or _integer(
            row.get("repository_expected_grid_gap_count"),
            f"{scope}.repository_expected_grid_gap_count",
        )
        != tail
        or row.get("passed") is not (tail == 0)
    ):
        raise ValueError("pre-append OHLCV counts are internally inconsistent")
    _require_zero_metrics(
        row,
        (
            "invalid_timestamp_count",
            "duplicate_count",
            "nonmonotone_count",
            "nonfinite_count",
            "off_grid_count",
            "out_of_range_count",
            "missing_prefix_count",
            "interior_gap_count",
            "repository_invalid_timestamp_count",
            "repository_duplicate_count",
            "repository_nonmonotone_count",
            "repository_nonfinite_count",
            "repository_off_grid_count",
            "repository_out_of_range_count",
            "repository_missing_prefix_count",
            "repository_interior_gap_count",
            "physical_repository_mismatch_count",
        ),
        scope,
    )
    physical_digest = _sha(
        row.get("physical_row_value_sha256"),
        f"{scope}.physical_row_value_sha256",
    )
    repository_digest = _sha(
        row.get("repository_row_value_sha256"),
        f"{scope}.repository_row_value_sha256",
    )
    if physical_digest != repository_digest:
        raise ValueError("pre-append OHLCV validation or digest parity is invalid")
    expected_report_metrics = {
        "required_columns": list(_OHLCV_COLUMNS),
        "datetime_column": "datetime",
        "symbol_column": None,
        "price_columns": ["open", "high", "low", "close"],
        "require_unique_timestamp": True,
        "require_monotonic": True,
    }
    for report_key, report_rows in (
        ("physical_ohlcv_validation", rows),
        ("repository_ohlcv_validation", repository_rows),
    ):
        report = row[report_key]
        metrics = report.get("metrics") if isinstance(report, Mapping) else None
        if (
            not isinstance(report, Mapping)
            or set(report) != {"artifact_kind", "passed", "rows", "issues", "metrics"}
            or report["artifact_kind"] != "ohlcv_validation_report"
            or report["passed"] is not True
            or _integer(report["rows"], f"{scope}.{report_key}.rows", positive=True) != report_rows
            or report["issues"] != []
            or not isinstance(metrics, Mapping)
            or set(metrics) != set(expected_report_metrics)
            or metrics["required_columns"] != expected_report_metrics["required_columns"]
            or metrics["datetime_column"] != "datetime"
            or metrics["symbol_column"] is not None
            or metrics["price_columns"] != expected_report_metrics["price_columns"]
            or metrics["require_unique_timestamp"] is not True
            or metrics["require_monotonic"] is not True
        ):
            raise ValueError("pre-append OHLCV validation report is inconsistent")
    return rows, physical_digest


def _validate_pre_funding_metric(
    row: Mapping[str, Any],
    declared: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[int, str]:
    if set(row) != {
        "id",
        "passed",
        "safe_tail_append",
        "rows",
        "actual_settlement_count",
        "expected_settlement_count",
        "duplicate_count",
        "nonmonotone_count",
        "nonfinite_count",
        "invalid_source_count",
        "funding_rate_mismatch_count",
        "unexpected_timestamp_count",
        "unexpected_settlement_gap_count",
        "missing_prefix_count",
        "interior_gap_count",
        "missing_tail_count",
        "settlement_row_value_sha256",
    }:
        raise ValueError("pre-append funding metric schema is invalid")
    scope = "pre.funding"
    if row.get("safe_tail_append") is not True:
        raise ValueError("pre-append funding metric is not safe-tail")
    start = _integer(declared.get("start_ms"), "contract.funding.start_ms")
    end = _integer(declared.get("end_exclusive_ms"), "contract.funding.end_exclusive_ms")
    expected = len(_funding_schedule(declared, start, end, provenance)[0])
    actual = _integer(
        row.get("actual_settlement_count"),
        f"{scope}.actual_settlement_count",
        positive=True,
    )
    rows = _integer(row.get("rows"), f"{scope}.rows", positive=True)
    tail = _integer(row.get("missing_tail_count"), f"{scope}.missing_tail_count")
    if (
        _integer(
            row.get("expected_settlement_count"),
            f"{scope}.expected_settlement_count",
            positive=True,
        )
        != expected
        or rows < actual
        or actual + tail != expected
        or _integer(
            row.get("unexpected_settlement_gap_count"),
            f"{scope}.unexpected_settlement_gap_count",
        )
        != tail
        or row.get("passed") is not (tail == 0)
    ):
        raise ValueError("pre-append funding counts are internally inconsistent")
    _require_zero_metrics(
        row,
        (
            "duplicate_count",
            "nonmonotone_count",
            "nonfinite_count",
            "invalid_source_count",
            "funding_rate_mismatch_count",
            "unexpected_timestamp_count",
            "missing_prefix_count",
            "interior_gap_count",
        ),
        scope,
    )
    digest = _sha(
        row.get("settlement_row_value_sha256"),
        f"{scope}.settlement_row_value_sha256",
    )
    return actual, digest


def _validate_pre_seal(
    pre: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if set(pre) != {
        "artifact_kind",
        "schema_version",
        "generated_at_utc",
        "contract_id",
        "mode",
        "decision_class",
        "decision",
        "passed",
        "admission_eligible",
        "validation_layer",
        "input",
        "issues",
        "d04_status",
        "series",
        "funding",
        "seals",
    }:
        raise ValueError("pre-append receipt schema is invalid")
    schema_version = _integer(pre["schema_version"], "pre.schema_version", positive=True)
    d04_status = pre["d04_status"]
    if (
        not isinstance(d04_status, Mapping)
        or set(d04_status) != {"passed", "registry_sha256"}
        or d04_status["passed"] is not True
        or _sha(d04_status["registry_sha256"], "pre.d04_status.registry_sha256")
        != receipt["d04_status"]["registry_sha256"]
    ):
        raise ValueError("pre-append receipt D-04 status is invalid")
    if (
        pre["artifact_kind"] != "research_data_contract_validation"
        or schema_version != 1
        or pre["mode"] != "pre_append"
        or pre["decision_class"] != "triage"
        or pre["admission_eligible"] is not False
        or pre["validation_layer"] != ["physical", "repository_view"]
    ):
        raise ValueError("pre-append receipt identity or mode is invalid")
    if (
        pre.get("passed") is not True
        or pre.get("decision") != "SAFE_TAIL_APPEND"
        or pre.get("issues") != []
    ):
        raise ValueError("pre-append receipt is not a clean safe-tail decision")
    generated_at = _string(pre["generated_at_utc"], "pre.generated_at_utc")
    try:
        generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("pre-append receipt generation time is invalid") from exc
    if generated.tzinfo is None:
        raise ValueError("pre-append receipt generation time is invalid")
    current_generated_at = _string(receipt["generated_at_utc"], "receipt.generated_at_utc")
    current_generated = datetime.fromisoformat(current_generated_at.replace("Z", "+00:00"))
    # ponytail: allow five minutes of host-clock drift; signed monotonic receipts remove this window.
    if generated > current_generated + timedelta(minutes=5):
        raise ValueError("pre-append receipt generation time is invalid")
    if _string(pre["contract_id"], "pre.contract_id") != receipt["contract_id"]:
        raise ValueError("pre-append receipt contract id is invalid")
    expected = receipt["input"]
    inputs = pre["input"]
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "db_path",
        "contract_manifest",
        "lifecycle_manifest",
        "contract_manifest_sha256",
        "lifecycle_manifest_sha256",
        "contract_core_sha256",
    }:
        raise ValueError("pre-append receipt input schema is invalid")
    for field in ("db_path", "contract_manifest", "lifecycle_manifest"):
        _string(inputs[field], f"pre.input.{field}")
    for field in (
        "contract_manifest_sha256",
        "lifecycle_manifest_sha256",
        "contract_core_sha256",
    ):
        _sha(inputs[field], f"pre.input.{field}")
    if (
        inputs["db_path"] != expected["db_path"]
        or inputs["contract_manifest"] != expected["contract_manifest"]
        or inputs["lifecycle_manifest"] != expected["lifecycle_manifest"]
        or inputs["contract_manifest_sha256"] != expected["contract_core_sha256"]
        or inputs["contract_core_sha256"] != expected["contract_core_sha256"]
        or inputs["lifecycle_manifest_sha256"] != expected["lifecycle_manifest_sha256"]
    ):
        raise ValueError("pre-append receipt manifest chain is invalid")

    series = pre.get("series")
    funding = pre.get("funding")
    seals = pre.get("seals")
    if (
        not isinstance(series, list)
        or not series
        or not isinstance(funding, list)
        or not funding
        or not isinstance(seals, Mapping)
    ):
        raise ValueError("pre-append receipt seals or metrics are missing")
    declared_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for prefix, contract_key in (
        ("ohlcv", "ohlcv_series"),
        ("funding", "funding_series"),
    ):
        contract_rows = contract.get(contract_key)
        if not isinstance(contract_rows, list) or not contract_rows:
            raise ValueError("contract series declarations are invalid")
        rows_by_id: dict[str, Mapping[str, Any]] = {}
        for row in contract_rows:
            if not isinstance(row, Mapping):
                raise ValueError("contract series declarations are invalid")
            identifier = _string(row.get("id"), f"contract.{prefix}.id")
            if identifier in rows_by_id:
                raise ValueError("contract contains duplicate series ids")
            rows_by_id[identifier] = row
        declared_rows[prefix] = rows_by_id

    expected_seals: dict[str, tuple[str, int, str]] = {}
    for prefix, rows in (("ohlcv", series), ("funding", funding)):
        metric_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("pre-append receipt metrics are invalid")
            identifier = _string(row.get("id"), f"pre.{prefix}.id")
            declared = declared_rows[prefix].get(identifier)
            if declared is None:
                raise ValueError("pre-append receipt metrics do not match contract series")
            if prefix == "ohlcv":
                count, metric_digest = _validate_pre_ohlcv_metric(row, declared)
            else:
                count, metric_digest = _validate_pre_funding_metric(row, declared, provenance)
            key = f"{prefix}:{identifier}"
            if key in expected_seals:
                raise ValueError("pre-append receipt contains duplicate series ids")
            expected_seals[key] = (identifier, count, metric_digest)
            metric_ids.add(identifier)
        if metric_ids != set(declared_rows[prefix]):
            raise ValueError("pre-append receipt metrics do not match contract series")

    if set(seals) != set(expected_seals):
        raise ValueError("pre-append receipt seal keys do not match validated series")
    for key, (identifier, expected_count, metric_digest) in expected_seals.items():
        seal = seals[key]
        if not isinstance(seal, Mapping) or set(seal) != {
            "id",
            "row_count",
            "row_value_sha256",
        }:
            raise ValueError("pre-append receipt seal identity, count, or digest is invalid")
        seal_id = _string(seal["id"], f"pre.seals.{key}.id")
        seal_count = _integer(seal["row_count"], f"pre.seals.{key}.row_count", positive=True)
        seal_digest = _sha(seal["row_value_sha256"], f"pre.seals.{key}.row_value_sha256")
        if seal_id != identifier or seal_count != expected_count or seal_digest != metric_digest:
            raise ValueError("pre-append receipt seal identity, count, or digest is invalid")


def _ohlcv_accounting(
    frame: pl.DataFrame, expected: list[int], start: int, end: int, step: int
) -> dict[str, Any]:
    _frame_digest(frame, _OHLCV_COLUMNS)
    timestamps = [_timestamp_ms(value) for value in frame.get_column("datetime").to_list()]
    invalid = sum(timestamp is None for timestamp in timestamps)
    values = [timestamp for timestamp in timestamps if timestamp is not None]
    duplicates = len(values) - len(set(values))
    nonmonotone = sum(current < previous for previous, current in pairwise(values))
    nonfinite = 0
    for column in _OHLCV_COLUMNS[1:]:
        for value in frame.get_column(column).to_list():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                nonfinite += 1
    out_of_range = sum(timestamp < start or timestamp >= end for timestamp in values)
    off_grid = sum(
        start <= timestamp < end and (timestamp - start) % step != 0 for timestamp in values
    )
    owned_grid = {
        timestamp
        for timestamp in values
        if start <= timestamp < end and (timestamp - start) % step == 0
    }
    prefix, interior, tail = _missing_counts(expected, owned_grid)
    return {
        "rows": frame.height,
        "invalid_timestamp_count": invalid,
        "duplicate_count": duplicates,
        "nonmonotone_count": nonmonotone,
        "nonfinite_count": nonfinite,
        "off_grid_count": off_grid,
        "out_of_range_count": out_of_range,
        "expected_count": len(expected),
        "expected_grid_gap_count": prefix + interior + tail,
        "missing_prefix_count": prefix,
        "interior_gap_count": interior,
        "missing_tail_count": tail,
    }


def _funding_schedule(
    raw: Mapping[str, Any],
    start: int,
    end: int,
    provenance: Mapping[str, Any],
) -> tuple[list[int], list[int], list[str], list[float], list[tuple[str, int]]]:
    schedule = raw["schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("funding schedule is invalid")
    expected: list[int] = []
    tolerances: list[int] = []
    sources: list[str] = []
    cursor = start
    for segment in schedule:
        fields = {
            "start_ms",
            "end_exclusive_ms",
            "cadence_ms",
            "first_settlement_ms",
            "tolerance_ms",
            "provenance_id",
        }
        if not isinstance(segment, Mapping) or set(segment) != fields:
            raise ValueError("funding segment schema is invalid")
        segment_start = _integer(segment["start_ms"], "funding.start_ms")
        segment_end = _integer(segment["end_exclusive_ms"], "funding.end_exclusive_ms")
        cadence = _integer(segment["cadence_ms"], "funding.cadence_ms", positive=True)
        first = _integer(segment["first_settlement_ms"], "funding.first_settlement_ms")
        tolerance = _integer(segment["tolerance_ms"], "funding.tolerance_ms")
        provenance_id = segment["provenance_id"]
        if (
            segment_start != cursor
            or segment_end <= segment_start
            or not segment_start <= first < segment_end
            or tolerance * 2 >= cadence
            or provenance_id not in raw["provenance_ids"]
            or provenance[provenance_id]["envelope"]["data_kind"] != "official_funding"
        ):
            raise ValueError("funding schedule is invalid")
        points = list(range(first, segment_end, cadence))
        expected.extend(points)
        tolerances.extend([tolerance] * len(points))
        sources.extend([provenance_id] * len(points))
        cursor = segment_end
    if cursor != end or len(expected) != len(set(expected)):
        raise ValueError("funding schedule does not tile interval uniquely")
    for left, right, left_tolerance, right_tolerance in zip(
        expected, expected[1:], tolerances, tolerances[1:]
    ):
        if left + left_tolerance >= right - right_tolerance:
            raise ValueError("funding tolerance windows overlap")
    symbol = _string(raw["symbol"], "funding.symbol")
    rates: list[float] = []
    evidence_keys: list[tuple[str, int]] = []
    for point, tolerance, provenance_id in zip(expected, tolerances, sources, strict=True):
        matching_rows = [
            (payload_index, rate)
            for payload_index, evidence_symbol, timestamp, rate in provenance[provenance_id][
                "official_funding_rows"
            ]
            if evidence_symbol == symbol and abs(timestamp - point) <= tolerance
        ]
        if len(matching_rows) != 1:
            raise ValueError("official funding evidence does not own each schedule slot")
        payload_index, rate = matching_rows[0]
        evidence_keys.append((provenance_id, payload_index))
        rates.append(rate)
    official_ids = {
        provenance_id
        for provenance_id in raw["provenance_ids"]
        if provenance[provenance_id]["envelope"]["data_kind"] == "official_funding"
    }
    if set(sources) != official_ids:
        raise ValueError("every official funding provenance must own schedule slots")
    for provenance_id in official_ids:
        for _, evidence_symbol, timestamp, _ in provenance[provenance_id]["official_funding_rows"]:
            if evidence_symbol != symbol or not start <= timestamp < end:
                continue
            matching_slots = [
                index
                for index, (point, tolerance, source) in enumerate(
                    zip(expected, tolerances, sources, strict=True)
                )
                if source == provenance_id and abs(timestamp - point) <= tolerance
            ]
            if len(matching_slots) != 1:
                raise ValueError("official funding evidence and schedule are not bijective")
    return expected, tolerances, sources, rates, evidence_keys


def _funding_accounting(
    frame: pl.DataFrame,
    expected: list[int],
    tolerances: list[int],
    sources: list[str],
    expected_rates: list[float],
    source_map: Mapping[str, str],
) -> dict[str, Any]:
    _frame_digest(frame, _FUNDING_COLUMNS)
    nonmonotone = nonfinite = invalid_source = unexpected = rate_mismatch = 0
    accepted_counts = [0] * len(expected)
    slot_counts = [0] * len(expected)
    settlements = 0
    previous: int | None = None
    for row in frame.to_dicts():
        rate = row["funding_rate"]
        if rate is None:
            continue
        settlements += 1
        timestamp = _timestamp_ms(row["timestamp_ms"])
        if timestamp is None:
            unexpected += 1
            continue
        if previous is not None and timestamp < previous:
            nonmonotone += 1
        previous = timestamp
        index = bisect.bisect_left(expected, timestamp)
        candidates = [
            candidate for candidate in (index - 1, index) if 0 <= candidate < len(expected)
        ]
        matching = next(
            (
                candidate
                for candidate in candidates
                if abs(timestamp - expected[candidate]) <= tolerances[candidate]
            ),
            None,
        )
        if matching is None:
            unexpected += 1
            continue
        slot_counts[matching] += 1
        valid_rate = (
            not isinstance(rate, bool)
            and isinstance(rate, (int, float))
            and math.isfinite(float(rate))
        )
        if not valid_rate:
            nonfinite += 1
            continue
        source = row["source"]
        if source not in source_map or source_map[source] != sources[matching]:
            invalid_source += 1
            continue
        if not math.isclose(float(rate), expected_rates[matching], rel_tol=1e-12, abs_tol=1e-15):
            rate_mismatch += 1
            continue
        accepted_counts[matching] += 1
    duplicate = sum(max(0, count - 1) for count in slot_counts)
    mapped = {expected[index] for index, count in enumerate(accepted_counts) if count}
    prefix, interior, tail = _missing_counts(expected, mapped)
    return {
        "rows": frame.height,
        "actual_settlement_count": settlements,
        "expected_settlement_count": len(expected),
        "duplicate_count": duplicate,
        "nonmonotone_count": nonmonotone,
        "nonfinite_count": nonfinite,
        "invalid_source_count": invalid_source,
        "funding_rate_mismatch_count": rate_mismatch,
        "unexpected_timestamp_count": unexpected,
        "unexpected_settlement_gap_count": prefix + interior + tail,
        "missing_prefix_count": prefix,
        "interior_gap_count": interior,
        "missing_tail_count": tail,
    }


def validate_research_data_contract(
    db_path: str | Path,
    contract_manifest: str | Path | Mapping[str, Any],
    lifecycle_manifest: str | Path | Mapping[str, Any],
    mode: str,
    *,
    pre_append_receipt: str | Path | Mapping[str, Any] | None = None,
    physical_loader: Callable[..., Any] = _direct_physical_loader,
    ohlcv_loader: Callable[..., Any] = load_strict_ohlcv_route,
    funding_loader: Callable[..., Any] = load_futures_feature_points_from_db,
) -> dict[str, Any]:
    """Validate local data only; pre-append is intentionally never admitting."""
    try:
        mode = _canonical_mode(mode)
    except ValueError:
        receipt = _base_receipt(db_path, contract_manifest, lifecycle_manifest, str(mode))
        _issue(receipt, "mode_invalid")
        return receipt
    receipt = _base_receipt(db_path, contract_manifest, lifecycle_manifest, mode)
    root = Path(db_path)
    if not root.is_dir():
        _issue(receipt, "db_root_invalid")
        return receipt
    try:
        contract, _contract_bytes, base = _load_json(contract_manifest)
        lifecycle, lifecycle_bytes, _ = _load_json(lifecycle_manifest)
        receipt["input"].update(
            {
                "contract_manifest_sha256": _digest(_canonical_bytes(contract)),
                "lifecycle_manifest_sha256": _digest(lifecycle_bytes),
            }
        )
        required = {
            "artifact_kind",
            "schema_version",
            "contract_id",
            "lifecycle_manifest_sha256",
            "common_owned_interval",
            "session_calendar",
            "provenance",
            "ohlcv_series",
            "funding_series",
            "pre_append_receipt_sha256",
        }
        if (
            set(contract) != required
            or contract["artifact_kind"] != "research_data_contract_manifest"
            or contract["schema_version"] != 1
            or contract["session_calendar"] != "utc_24x7"
            or _sha(contract["lifecycle_manifest_sha256"], "lifecycle_manifest_sha256")
            != _digest(lifecycle_bytes)
        ):
            raise ValueError("contract manifest schema is invalid")
        receipt["input"]["contract_core_sha256"] = _contract_core_sha256(contract)
        pre_append_sha = contract["pre_append_receipt_sha256"]
        if mode == "pre_append":
            if pre_append_sha is not None:
                raise ValueError("pre-append contract must not bind a pre-append receipt")
            expected_pre_append_sha = None
        else:
            expected_pre_append_sha = _sha(pre_append_sha, "pre_append_receipt_sha256")
        common = contract["common_owned_interval"]
        if not isinstance(common, Mapping) or set(common) != {"start_ms", "end_exclusive_ms"}:
            raise ValueError("common owned interval is invalid")
        common_start = _integer(common["start_ms"], "common.start")
        common_end = _integer(common["end_exclusive_ms"], "common.end", positive=True)
        if common_end <= common_start:
            raise ValueError("common owned interval is invalid")
        receipt["contract_id"] = _string(contract["contract_id"], "contract_id")
        provenance = _validate_provenance(contract, base, root.resolve())
        symbols, folds = _validate_lifecycle(lifecycle)
        receipt["d04_status"] = {
            "passed": True,
            "registry_sha256": registry_sha256(lifecycle["registry"]),
        }
        if (
            not isinstance(contract["ohlcv_series"], list)
            or not contract["ohlcv_series"]
            or not isinstance(contract["funding_series"], list)
            or not contract["funding_series"]
        ):
            raise ValueError("series lists must be non-empty")
        pre = None
        if mode == "post_append_strict":
            if pre_append_receipt is None:
                raise ValueError("post-append strict requires a pre-append receipt")
            pre, pre_bytes, _ = _load_json(pre_append_receipt)
            actual_pre_append_sha = _digest(pre_bytes)
            if actual_pre_append_sha != expected_pre_append_sha:
                raise ValueError("pre-append receipt hash does not match contract")
            receipt["input"]["pre_append_receipt_sha256"] = actual_pre_append_sha
            _validate_pre_seal(pre, receipt, contract, provenance)
    except Exception as exc:
        _issue(receipt, "manifest_or_lifecycle_invalid", detail=str(exc))
        return receipt

    seen_ids: set[str] = set()
    claimed_ohlcv_series: set[tuple[str, str, str, int, int]] = set()
    for raw in contract["ohlcv_series"]:
        result: dict[str, Any] = {
            "id": raw.get("id") if isinstance(raw, Mapping) else None,
            "passed": False,
            "safe_tail_append": False,
        }
        try:
            required = {
                "id",
                "exchange",
                "symbol",
                "timeframe",
                "start_ms",
                "end_exclusive_ms",
                "step_ms",
                "anchor_ms",
                "physical_layout",
                "provenance_ids",
                "fold_ids",
            }
            if (
                not isinstance(raw, Mapping)
                or set(raw) != required
                or raw["physical_layout"] != "partitioned_ohlcv"
            ):
                raise ValueError("ohlcv schema/layout is invalid")
            identifier = _string(raw["id"], "ohlcv.id")
            if identifier in seen_ids:
                raise ValueError("duplicate series id")
            seen_ids.add(identifier)
            exchange, symbol, start, end = _check_series_common(
                raw, common, folds, symbols, provenance
            )
            timeframe = normalize_timeframe_token(_string(raw["timeframe"], "ohlcv.timeframe"))
            semantic_key = (exchange, symbol, timeframe, start, end)
            if semantic_key in claimed_ohlcv_series:
                raise ValueError("duplicate semantic OHLCV series")
            claimed_ohlcv_series.add(semantic_key)
            step = _integer(raw["step_ms"], "ohlcv.step_ms", positive=True)
            anchor = _integer(raw["anchor_ms"], "ohlcv.anchor_ms")
            if (
                step != timeframe_to_milliseconds(timeframe)
                or anchor != 0
                or (start - anchor) % step
                or (end - start) % step
            ):
                raise ValueError("timeframe step or UTC anchor is invalid")
            physical = physical_loader(
                root,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_ms=start,
                end_exclusive_ms=end,
            )
            repository = ohlcv_loader(
                str(root),
                storage_route="partitioned_ohlcv",
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start,
                end_date=end - 1,
            )
            expected = list(range(start, end, step))
            physical_metrics = _ohlcv_accounting(physical, expected, start, end, step)
            repository_metrics = _ohlcv_accounting(repository, expected, start, end, step)
            physical_digest = _frame_digest(physical, _OHLCV_COLUMNS)
            repository_digest = _frame_digest(repository, _OHLCV_COLUMNS)
            physical_report = validate_ohlcv_frame(physical).to_dict()
            repository_report = validate_ohlcv_frame(repository).to_dict()
            bad_metrics = (
                "invalid_timestamp_count",
                "duplicate_count",
                "nonmonotone_count",
                "nonfinite_count",
                "off_grid_count",
                "out_of_range_count",
                "missing_prefix_count",
                "interior_gap_count",
            )
            safe = (
                not any(physical_metrics[key] or repository_metrics[key] for key in bad_metrics)
                and physical_digest == repository_digest
                and physical_report.get("passed") is True
                and repository_report.get("passed") is True
                and bool(physical_metrics["rows"])
            )
            result.update(physical_metrics)
            result.update({f"repository_{key}": value for key, value in repository_metrics.items()})
            result.update(
                {
                    "physical_row_value_sha256": physical_digest,
                    "repository_row_value_sha256": repository_digest,
                    "physical_repository_mismatch_count": int(physical_digest != repository_digest),
                    "physical_ohlcv_validation": physical_report,
                    "repository_ohlcv_validation": repository_report,
                    "safe_tail_append": safe,
                }
            )
            result["passed"] = (
                safe
                and not physical_metrics["missing_tail_count"]
                and not repository_metrics["missing_tail_count"]
            )
            receipt["seals"]["ohlcv:" + identifier] = _seal_entry(
                identifier, physical, _OHLCV_COLUMNS
            )
            if mode == "post_append_strict" and not _sealed_prefix_matches(
                pre["seals"].get("ohlcv:" + identifier),
                physical,
                _OHLCV_COLUMNS,
                identifier,
            ):
                raise ValueError("sealed OHLCV rows or values changed")
        except Exception as exc:
            result["error"] = str(exc)
            _issue(receipt, "ohlcv_validation_exception", id=result["id"], detail=str(exc))
        receipt["series"].append(result)

    official_funding_catalog: dict[tuple[str, int], tuple[str, str, int, float]] = {}
    for provenance_id, item in provenance.items():
        if item["envelope"]["data_kind"] != "official_funding":
            continue
        for payload_index, event_symbol, timestamp, rate in item["official_funding_rows"]:
            official_funding_catalog[(provenance_id, payload_index)] = (
                item["exchange"],
                event_symbol,
                timestamp,
                rate,
            )

    claimed_funding_evidence: dict[tuple[str, int], str] = {}
    claimed_funding_slots: dict[tuple[str, str, int], str] = {}
    claimed_funding_events: dict[tuple[str, str, int], str] = {}
    claimed_funding_series: set[tuple[str, str, str, int, int]] = set()

    for raw in contract["funding_series"]:
        result = {
            "id": raw.get("id") if isinstance(raw, Mapping) else None,
            "passed": False,
            "safe_tail_append": False,
        }
        try:
            required = {
                "id",
                "exchange",
                "symbol",
                "interval",
                "start_ms",
                "end_exclusive_ms",
                "provenance_ids",
                "fold_ids",
                "row_sources",
                "schedule",
            }
            if not isinstance(raw, Mapping) or set(raw) != required:
                raise ValueError("funding schema is invalid")
            identifier = _string(raw["id"], "funding.id")
            if identifier in seen_ids:
                raise ValueError("duplicate series id")
            seen_ids.add(identifier)
            exchange, symbol, start, end = _check_series_common(
                raw, common, folds, symbols, provenance, funding=True
            )
            interval = _string(raw["interval"], "funding.interval")
            if interval != "perpetual":
                raise ValueError("funding interval must be 'perpetual'")
            semantic_series = (exchange, symbol, interval, start, end)
            if semantic_series in claimed_funding_series:
                raise ValueError("duplicate semantic funding series")
            claimed_funding_series.add(semantic_series)
            source_map = raw["row_sources"]
            if (
                not isinstance(source_map, Mapping)
                or not source_map
                or any(
                    not isinstance(source, str)
                    or not source
                    or source != source.strip()
                    or provenance_id not in raw["provenance_ids"]
                    or provenance[provenance_id]["envelope"]["data_kind"] != "official_funding"
                    for source, provenance_id in source_map.items()
                )
            ):
                raise ValueError("funding sources must map exactly to official funding provenance")
            (
                expected,
                tolerances,
                expected_sources,
                expected_rates,
                evidence_keys,
            ) = _funding_schedule(raw, start, end, provenance)
            for evidence_key in evidence_keys:
                owner = claimed_funding_evidence.get(evidence_key)
                if owner is not None:
                    raise ValueError(
                        f"official funding evidence is reused by {owner} and {identifier}"
                    )
                claimed_funding_evidence[evidence_key] = identifier
                event = official_funding_catalog[evidence_key]
                semantic_event = (event[0], event[1], event[2])
                event_owner = claimed_funding_events.get(semantic_event)
                if event_owner is not None:
                    raise ValueError(
                        f"semantic funding event is reused by {event_owner} and {identifier}"
                    )
                claimed_funding_events[semantic_event] = identifier
            for point in expected:
                slot_key = (exchange, symbol, point)
                owner = claimed_funding_slots.get(slot_key)
                if owner is not None:
                    raise ValueError(f"funding schedule slot is reused by {owner} and {identifier}")
                claimed_funding_slots[slot_key] = identifier
            frame = funding_loader(
                str(root), exchange=exchange, symbol=symbol, start_date=start, end_date=end - 1
            )
            if not isinstance(frame, pl.DataFrame) or not set(_FUNDING_COLUMNS) <= set(
                frame.columns
            ):
                raise ValueError("funding loader frame schema invalid")
            frame = frame.select(_FUNDING_COLUMNS)
            metrics = _funding_accounting(
                frame,
                expected,
                tolerances,
                expected_sources,
                expected_rates,
                source_map,
            )
            bad_metrics = (
                "duplicate_count",
                "nonmonotone_count",
                "nonfinite_count",
                "funding_rate_mismatch_count",
                "invalid_source_count",
                "unexpected_timestamp_count",
                "missing_prefix_count",
                "interior_gap_count",
            )
            safe = not any(metrics[key] for key in bad_metrics) and bool(
                metrics["actual_settlement_count"]
            )
            result.update(metrics)
            result["safe_tail_append"] = safe
            result["passed"] = safe and not metrics["missing_tail_count"]
            settlement_frame = _funding_settlement_frame(frame)
            settlement_digest = _frame_digest(settlement_frame, _FUNDING_COLUMNS)
            result["settlement_row_value_sha256"] = settlement_digest
            receipt["seals"]["funding:" + identifier] = _seal_entry(
                identifier, settlement_frame, _FUNDING_COLUMNS
            )
            if mode == "post_append_strict" and not _sealed_prefix_matches(
                pre["seals"].get("funding:" + identifier),
                settlement_frame,
                _FUNDING_COLUMNS,
                identifier,
            ):
                raise ValueError("sealed funding rows or values changed")
        except Exception as exc:
            result["error"] = str(exc)
            _issue(receipt, "funding_validation_exception", id=result["id"], detail=str(exc))
        receipt["funding"].append(result)

    unclaimed_funding_evidence = set(official_funding_catalog) - set(claimed_funding_evidence)
    if unclaimed_funding_evidence:
        _issue(
            receipt,
            "unowned_official_funding_evidence",
            count=len(unclaimed_funding_evidence),
        )
    rows = receipt["series"] + receipt["funding"]
    receipt["passed"] = (
        bool(rows)
        and all(row["passed"] or (mode == "pre_append" and row["safe_tail_append"]) for row in rows)
        and not receipt["issues"]
    )
    receipt["admission_eligible"] = mode == "post_append_strict" and receipt["passed"]
    receipt["decision"] = (
        "ADMIT"
        if receipt["admission_eligible"]
        else ("SAFE_TAIL_APPEND" if receipt["passed"] else "STOP")
    )
    return receipt


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _output_is_under_root(output: Path, root: Path) -> bool:
    try:
        output.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--contract-manifest", required=True, type=Path)
    parser.add_argument("--lifecycle-manifest", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("pre-append", "pre_append", "post-append-strict", "post_append_strict"),
    )
    parser.add_argument("--pre-append-receipt", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    try:
        mode = _canonical_mode(args.mode)
        if not args.db_path.is_dir():
            raise ValueError("db root must exist")
        if mode == "post_append_strict" and args.pre_append_receipt is None:
            raise ValueError("post-append strict requires --pre-append-receipt")
        if args.output_json is not None and _output_is_under_root(args.output_json, args.db_path):
            raise ValueError("output JSON must be outside db root")
        receipt = validate_research_data_contract(
            args.db_path,
            args.contract_manifest,
            args.lifecycle_manifest,
            mode,
            pre_append_receipt=args.pre_append_receipt,
        )
        if args.output_json is not None:
            _atomic_json(args.output_json, receipt)
    except Exception as exc:
        receipt = _base_receipt(
            args.db_path,
            args.contract_manifest,
            args.lifecycle_manifest,
            locals().get("mode", args.mode),
        )
        _issue(receipt, "internal_or_malformed_failure", detail=str(exc))
        print(json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 1
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False))
    if any(issue["code"] == "manifest_or_lifecycle_invalid" for issue in receipt["issues"]):
        return 1
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
