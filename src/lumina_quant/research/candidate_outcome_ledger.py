"""Append-only candidate outcome ledger for Alpha Zoo calibration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateOutcomeRecord:
    candidate_id: str
    split: str
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    barrier_type: str
    net_pnl_bps: float
    mae_bps: float
    mfe_bps: float
    feature_snapshot: Mapping[str, Any] = field(default_factory=dict)
    costs: Mapping[str, float] = field(default_factory=dict)
    strategy_validity: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateOutcomeLedger:
    """Small JSONL ledger used by calibration scripts and tests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: CandidateOutcomeRecord | Mapping[str, Any]) -> None:
        payload = record.to_dict() if isinstance(record, CandidateOutcomeRecord) else dict(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                token = line.strip()
                if token:
                    out.append(json.loads(token))
        return out

    def summary(self) -> dict[str, Any]:
        records = self.read_all()
        total = len(records)
        train_val = sum(1 for row in records if str(row.get("split")) in {"train", "validation"})
        locked_oos = sum(1 for row in records if str(row.get("split")) in {"locked_oos", "oos"})
        pnl = [float(row.get("net_pnl_bps", 0.0)) for row in records]
        return {
            "artifact_kind": "candidate_outcome_ledger_summary",
            "path": str(self.path),
            "record_count": total,
            "train_validation_record_count": train_val,
            "locked_oos_record_count": locked_oos,
            "uses_locked_oos_for_selection": False,
            "mean_net_pnl_bps": sum(pnl) / total if total else 0.0,
        }
