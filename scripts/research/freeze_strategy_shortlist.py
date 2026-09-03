#!/usr/bin/env python3
"""Freeze an existing registry screen/selection as a content-addressed shortlist."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.research.run_card import atomic_write_text, stable_json_dumps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_shortlist(source: Path, *, output: Path, limit: int) -> dict:
    payload = json.loads(source.read_bytes())
    rows = payload.get("selected")
    if type(rows) is not list:
        raise ValueError("source selection must contain selected")
    selected = [
        dict(row) for row in rows[:limit] if type(row) is dict and type(row.get("strategy")) is str
    ]
    if not selected:
        raise ValueError("source selection contains no strategies")
    result = {
        "artifact_kind": "lumina_quant.frozen_strategy_shortlist.v1",
        "status": "complete",
        "source": {
            "path": str(source.resolve()),
            "sha256": _sha256(source),
            "artifact_kind": payload.get("artifact_kind"),
            "registry_count": payload.get("registry_count"),
            "result_count": payload.get("result_count"),
        },
        "selected": selected,
        "selection_uses_locked_oos": False,
        "promotion_authority": False,
        "order_routing_enabled": False,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_text(output, stable_json_dumps(result) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = freeze_shortlist(
        args.source.resolve(),
        output=args.output.resolve(),
        limit=max(1, args.limit),
    )
    print(json.dumps({"status": result["status"], "selected": len(result["selected"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
