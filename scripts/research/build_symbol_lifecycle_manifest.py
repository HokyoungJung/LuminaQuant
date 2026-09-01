#!/usr/bin/env python3
"""Build a source-provenanced symbol lifecycle registry without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from lumina_quant.data.symbol_lifecycle import (
    build_fold_membership_manifest,
    build_symbol_lifecycle_registry,
    validate_fold_membership_manifest,
    validate_symbol_lifecycle_registry,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-info-json", required=True, type=Path)
    parser.add_argument("--symbols", required=True, nargs="+", help="Exact Binance symbols.")
    parser.add_argument(
        "--source-uri", required=True, help="URI identifying the frozen source payload."
    )
    parser.add_argument(
        "--retrieved-at-ms", required=True, type=int, help="UTC epoch milliseconds."
    )
    parser.add_argument(
        "--folds-json", required=True, type=Path, help="JSON array of fold objects."
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load JSON: {path}") from exc


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = _parse_args()
    try:
        raw_exchange_info = args.exchange_info_json.read_bytes()
        exchange_info = json.loads(raw_exchange_info)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load JSON: {args.exchange_info_json}") from exc

    registry = build_symbol_lifecycle_registry(
        exchange_info,
        args.symbols,
        {
            "uri": args.source_uri,
            "retrieved_at_ms": args.retrieved_at_ms,
            "payload_sha256": hashlib.sha256(raw_exchange_info).hexdigest(),
        },
    )
    registry = validate_symbol_lifecycle_registry(registry)
    folds = _load_json(args.folds_json)
    fold_membership = build_fold_membership_manifest(registry, folds)
    payload: dict[str, Any] = {
        "registry": registry,
        "fold_membership": validate_fold_membership_manifest(registry, fold_membership),
    }
    _write_json_atomically(args.output_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
