"""Read-only alpha evidence and live-readiness payload service."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read_json(path: str | Path | None) -> Any:
    if path is None or not str(path).strip():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _gate_summary(run_cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_gates: dict[str, list[bool]] = {}
    for card in run_cards:
        for gate, passed in dict(card.get("reality_gates") or {}).items():
            all_gates.setdefault(str(gate), []).append(bool(passed))
    return {
        gate: {
            "passed": bool(values) and all(values),
            "observations": len(values),
        }
        for gate, values in sorted(all_gates.items())
    }


def build_alpha_evidence_payload(
    *,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    run_cards: Sequence[Mapping[str, Any]] | None = None,
    live_readiness: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a dashboard payload without importing broker/order-gateway code."""
    rows = [dict(item) for item in list(evidence or [])]
    cards = [dict(item) for item in list(run_cards or [])]
    live = dict(live_readiness or {})
    classifications: dict[str, int] = {}
    for row in rows:
        key = str(row.get("classification") or "unknown")
        classifications[key] = classifications.get(key, 0) + 1
    return {
        "artifact_kind": "dashboard_alpha_evidence_payload",
        "as_of": datetime.now(UTC).isoformat(),
        "advisory_only": True,
        "real_money_execution_enabled": False,
        "summary": {
            "alpha_count": len(rows),
            "run_card_count": len(cards),
            "classifications": classifications,
            "reality_gates": _gate_summary(cards),
            "live_readiness_action": live.get("recommended_action"),
        },
        "evidence": rows,
        "run_cards": cards,
        "live_readiness": live,
        "source": {
            "mode": "read_only_alpha_evidence",
            "status": "ok" if rows or cards or live else "empty",
            **dict(source or {}),
        },
    }


def load_alpha_evidence_payload(
    *,
    evidence_path: str | Path | None = None,
    run_card_path: str | Path | None = None,
    live_readiness_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load optional JSON artifacts and build the read-only dashboard payload."""
    evidence_raw = _read_json(evidence_path)
    run_card_raw = _read_json(run_card_path)
    live_raw = _read_json(live_readiness_path)
    return build_alpha_evidence_payload(
        evidence=[item for item in _as_list(evidence_raw) if isinstance(item, Mapping)],
        run_cards=[item for item in _as_list(run_card_raw) if isinstance(item, Mapping)],
        live_readiness=live_raw if isinstance(live_raw, Mapping) else None,
        source={
            "evidence_path": str(evidence_path or ""),
            "run_card_path": str(run_card_path or ""),
            "live_readiness_path": str(live_readiness_path or ""),
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Module-mode entry for /api/python/dashboard/alpha-evidence."""
    parser = argparse.ArgumentParser(
        prog="lumina_quant.dashboard.alpha_evidence_service",
        description="Emit read-only alpha evidence dashboard payload.",
    )
    parser.add_argument("--json", action="store_true", default=True)
    parser.add_argument("--evidence", default="", dest="evidence_path")
    parser.add_argument("--run-card", default="", dest="run_card_path")
    parser.add_argument("--live-readiness", default="", dest="live_readiness_path")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            load_alpha_evidence_payload(
                evidence_path=args.evidence_path,
                run_card_path=args.run_card_path,
                live_readiness_path=args.live_readiness_path,
            ),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


__all__ = [
    "build_alpha_evidence_payload",
    "load_alpha_evidence_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
