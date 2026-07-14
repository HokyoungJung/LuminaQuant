"""Fail-closed point-in-time Binance symbol lifecycle helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

REGISTRY_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


def _require_timestamp(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative UTC epoch millisecond integer")
    return value


def _require_symbol(value: Any, field: str = "symbol") -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty exact symbol string")
    return value


def _require_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("source provenance must be an object")
    expected = {"uri", "retrieved_at_ms", "payload_sha256"}
    if set(provenance) != expected:
        raise ValueError("source provenance fields are invalid")
    uri = provenance["uri"]
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("source URI is required")
    payload_sha256 = provenance["payload_sha256"]
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or any(char not in "0123456789abcdef" for char in payload_sha256)
    ):
        raise ValueError("source payload SHA-256 must be a lowercase hexadecimal digest")
    return {
        "uri": uri,
        "retrieved_at_ms": _require_timestamp(provenance["retrieved_at_ms"], "retrieved_at_ms"),
        "payload_sha256": payload_sha256,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def registry_sha256(registry: Mapping[str, Any]) -> str:
    """Return a deterministic hash of a validated registry artifact."""
    return hashlib.sha256(
        _canonical_json_bytes(validate_symbol_lifecycle_registry(registry))
    ).hexdigest()


def build_symbol_lifecycle_registry(
    exchange_info: Mapping[str, Any],
    symbols: Sequence[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a sorted registry from an already-frozen Binance exchangeInfo payload."""
    if not isinstance(exchange_info, Mapping):
        raise ValueError("exchangeInfo payload must be an object")
    raw_rows = exchange_info.get("symbols")
    if not isinstance(raw_rows, list):
        raise ValueError("exchangeInfo symbols must be a list")

    if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
        raise ValueError("symbols must be a sequence of exact symbol strings")
    requested = [_require_symbol(symbol, "requested symbol") for symbol in symbols]
    if not requested:
        raise ValueError("at least one requested symbol is required")
    if len(set(requested)) != len(requested):
        raise ValueError("requested symbols must not contain duplicates")

    by_symbol: dict[str, Mapping[str, Any]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("exchangeInfo symbol entries must be objects")
        symbol = _require_symbol(raw_row.get("symbol"), "exchangeInfo symbol")
        if "deliveryDate" not in raw_row:
            raise ValueError(f"{symbol}.deliveryDate is required")
        if symbol in by_symbol:
            raise ValueError(f"exchangeInfo contains duplicate symbol: {symbol}")
        by_symbol[symbol] = raw_row

    rows: list[dict[str, Any]] = []
    for symbol in requested:
        raw_row = by_symbol.get(symbol)
        if raw_row is None:
            raise ValueError(f"requested symbol is absent from exchangeInfo: {symbol}")
        onboard_ms = _require_timestamp(raw_row.get("onboardDate"), f"{symbol}.onboardDate")
        delivery_raw = raw_row["deliveryDate"]
        delivery_ms = (
            None
            if delivery_raw is None
            else _require_timestamp(delivery_raw, f"{symbol}.deliveryDate")
        )
        if delivery_ms is not None and delivery_ms <= onboard_ms:
            raise ValueError(f"{symbol} has an ambiguous lifecycle interval")
        rows.append({"symbol": symbol, "onboard_ms": onboard_ms, "delivery_ms": delivery_ms})

    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "source": _require_provenance(source),
        "symbols": sorted(rows, key=lambda row: row["symbol"]),
    }


def validate_symbol_lifecycle_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a lifecycle registry and return its canonical JSON-safe form."""
    if not isinstance(registry, Mapping) or set(registry) != {
        "schema_version",
        "source",
        "symbols",
    }:
        raise ValueError("symbol lifecycle registry fields are invalid")
    if registry["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported symbol lifecycle registry schema version")
    source = _require_provenance(registry["source"])
    raw_rows = registry["symbols"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("registry symbols must be a non-empty list")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "symbol",
            "onboard_ms",
            "delivery_ms",
        }:
            raise ValueError("registry symbol fields are invalid")
        symbol = _require_symbol(raw_row["symbol"])
        if symbol in seen:
            raise ValueError(f"registry contains duplicate symbol: {symbol}")
        seen.add(symbol)
        onboard_ms = _require_timestamp(raw_row["onboard_ms"], f"{symbol}.onboard_ms")
        delivery_raw = raw_row["delivery_ms"]
        delivery_ms = (
            None
            if delivery_raw is None
            else _require_timestamp(delivery_raw, f"{symbol}.delivery_ms")
        )
        if delivery_ms is not None and delivery_ms <= onboard_ms:
            raise ValueError(f"{symbol} has an ambiguous lifecycle interval")
        rows.append({"symbol": symbol, "onboard_ms": onboard_ms, "delivery_ms": delivery_ms})

    rows.sort(key=lambda row: row["symbol"])
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "source": source, "symbols": rows}


def load_symbol_lifecycle_registry(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on an invalid lifecycle registry JSON artifact."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load symbol lifecycle registry: {path}") from exc
    return validate_symbol_lifecycle_registry(payload)


def is_symbol_active(symbol: Mapping[str, Any], timestamp_ms: int) -> bool:
    """Return whether a symbol is active at a UTC epoch-ms instant."""
    timestamp_ms = _require_timestamp(timestamp_ms, "timestamp_ms")
    if not isinstance(symbol, Mapping):
        raise ValueError("symbol lifecycle entry must be an object")
    onboard_ms = _require_timestamp(symbol.get("onboard_ms"), "onboard_ms")
    if "delivery_ms" not in symbol:
        raise ValueError("delivery_ms is required")
    delivery_raw = symbol["delivery_ms"]
    delivery_ms = None if delivery_raw is None else _require_timestamp(delivery_raw, "delivery_ms")
    if delivery_ms is not None and delivery_ms <= onboard_ms:
        raise ValueError("symbol has an ambiguous lifecycle interval")
    return onboard_ms <= timestamp_ms and (delivery_ms is None or timestamp_ms < delivery_ms)


def build_fold_membership_manifest(
    registry: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic full/partial/inactive lifecycle membership for each fold."""
    normalized_registry = validate_symbol_lifecycle_registry(registry)
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)) or not folds:
        raise ValueError("folds must be a non-empty sequence")

    normalized_folds: list[dict[str, Any]] = []
    fold_ids: set[str] = set()
    for raw_fold in folds:
        if not isinstance(raw_fold, Mapping) or set(raw_fold) != {"fold_id", "start_ms", "end_ms"}:
            raise ValueError("fold fields are invalid")
        fold_id = _require_symbol(raw_fold["fold_id"], "fold_id")
        if fold_id in fold_ids:
            raise ValueError(f"duplicate fold_id: {fold_id}")
        fold_ids.add(fold_id)
        start_ms = _require_timestamp(raw_fold["start_ms"], f"{fold_id}.start_ms")
        end_ms = _require_timestamp(raw_fold["end_ms"], f"{fold_id}.end_ms")
        if end_ms <= start_ms:
            raise ValueError(f"{fold_id} has an ambiguous fold interval")
        normalized_folds.append({"fold_id": fold_id, "start_ms": start_ms, "end_ms": end_ms})

    memberships: list[dict[str, Any]] = []
    for fold in sorted(
        normalized_folds, key=lambda item: (item["start_ms"], item["end_ms"], item["fold_id"])
    ):
        eligible_symbols: list[str] = []
        partial_symbols: list[str] = []
        inactive_symbols: list[str] = []
        for symbol in normalized_registry["symbols"]:
            onboard_ms = symbol["onboard_ms"]
            delivery_ms = symbol["delivery_ms"]
            full_fold = onboard_ms <= fold["start_ms"] and (
                delivery_ms is None or delivery_ms >= fold["end_ms"]
            )
            overlaps = onboard_ms < fold["end_ms"] and (
                delivery_ms is None or delivery_ms > fold["start_ms"]
            )
            if full_fold:
                eligible_symbols.append(symbol["symbol"])
            elif overlaps:
                partial_symbols.append(symbol["symbol"])
            else:
                inactive_symbols.append(symbol["symbol"])
        memberships.append(
            {
                **fold,
                "eligible_symbols": eligible_symbols,
                "partial_symbols": partial_symbols,
                "inactive_symbols": inactive_symbols,
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "registry_sha256": registry_sha256(normalized_registry),
        "source": normalized_registry["source"],
        "folds": memberships,
    }


def validate_fold_membership_manifest(
    registry: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and canonically reconstruct a fold membership manifest."""
    normalized_registry = validate_symbol_lifecycle_registry(registry)
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_version",
        "registry_sha256",
        "source",
        "folds",
    }:
        raise ValueError("fold membership manifest fields are invalid")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported fold membership manifest schema version")
    if not isinstance(manifest["folds"], list) or not manifest["folds"]:
        raise ValueError("manifest folds must be a non-empty list")

    folds: list[dict[str, Any]] = []
    for raw_fold in manifest["folds"]:
        if not isinstance(raw_fold, Mapping) or set(raw_fold) != {
            "fold_id",
            "start_ms",
            "end_ms",
            "eligible_symbols",
            "partial_symbols",
            "inactive_symbols",
        }:
            raise ValueError("manifest fold fields are invalid")
        folds.append(
            {
                "fold_id": raw_fold["fold_id"],
                "start_ms": raw_fold["start_ms"],
                "end_ms": raw_fold["end_ms"],
            }
        )

    expected = build_fold_membership_manifest(normalized_registry, folds)
    try:
        matches = _canonical_json_bytes(manifest) == _canonical_json_bytes(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError("fold membership manifest is not JSON-safe") from exc
    if not matches:
        raise ValueError("fold membership manifest does not match registry")
    return expected
