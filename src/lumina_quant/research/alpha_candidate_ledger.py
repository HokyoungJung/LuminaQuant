"""Provenance ledger for alpha research candidates.

This ledger is deliberately *separate* from
:class:`lumina_quant.research.candidate_outcome_ledger.CandidateOutcomeRecord`
(which records realized triple-barrier trade outcomes). ``AlphaCandidateRecord``
captures the *research provenance* of a candidate formula: its seed, information
coefficient, turnover, Sharpe, and the full survivorship-gate audit trail (DSR,
effective / raw trial counts, cross-trial dispersion, SPA p-value, PBO). It must
never be merged into or mutate any existing metric dictionary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AlphaCandidateRecord",
    "ResearchLedger",
]


@dataclass(frozen=True, slots=True)
class AlphaCandidateRecord:
    """Immutable research-provenance record for a single alpha candidate.

    Frozen + slots so it can neither be mutated in place nor grow ad-hoc
    attributes. ``cost_ab`` is a forward-looking placeholder for cost-sensitivity
    A/B provenance and defaults to an empty mapping.
    """

    candidate_id: str
    formula: str
    seed: int
    ic: float
    icir: float
    turnover: float
    sharpe: float
    deflated_sharpe: float
    n_eff: float
    dispersion: float
    spa_pvalue: float
    pbo: float
    raw_n: int
    passed: bool = False
    cost_ab: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchLedger:
    """Append-only JSONL ledger of :class:`AlphaCandidateRecord` rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: AlphaCandidateRecord | Mapping[str, Any]) -> None:
        payload = (
            record.to_dict()
            if isinstance(record, AlphaCandidateRecord)
            else dict(record)
        )
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

    def query(
        self,
        *,
        candidate_id: str | None = None,
        passed: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return records filtered by candidate id and/or pass flag (in-order)."""
        rows = self.read_all()
        if candidate_id is not None:
            rows = [row for row in rows if str(row.get("candidate_id")) == candidate_id]
        if passed is not None:
            rows = [row for row in rows if bool(row.get("passed", False)) == passed]
        return rows

    def summary(self) -> dict[str, Any]:
        rows = self.read_all()
        total = len(rows)
        survived = sum(1 for row in rows if bool(row.get("passed", False)))
        return {
            "artifact_kind": "alpha_candidate_ledger_summary",
            "path": str(self.path),
            "record_count": total,
            "survived_count": survived,
        }
