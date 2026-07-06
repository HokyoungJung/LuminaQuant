#!/usr/bin/env python3
"""Emit a referenced real-money attestation artifact for the live-readiness gate.

Audit P1 closes the "self-attestation" hole: the strategy-agnostic real-money veto
in ``lumina_quant.live.readiness_policy`` now honors the positive flags
(``ready_for_real`` / ``real_execution_allowed`` / ``real_money_execution`` and the
canary flags) ONLY when they come from a *referenced* artifact — never from three
booleans hand-typed into the decision JSON.

This tool produces exactly that referenced artifact.  It is fail-closed by
construction: it **refuses** to set any positive flag unless the required evidence
references are supplied AND verified on disk (each file exists, is readable, and is
recorded with its sha256 + byte size).  Point the live-readiness decision at the
output via, e.g., ``strategy_params.real_money_attestation_artifact_path``.

Evidence contract:
  * ready_for_real / real_execution_allowed / real_money_execution require an
    operator id, a paper-stats artifact, a fill-slippage summary, and the
    live-readiness decision this attests to (for a lineage hash);
  * canary_execution_allowed requires the same real-money evidence (canary is a real
    prod stage) plus an explicit --assert-canary-execution-allowed;
  * canary_execution_recorded (which unlocks the ``full`` stage) additionally
    requires a verified --canary-run evidence artifact.

Nothing is written when the request cannot be satisfied; the process exits non-zero
with a structured reason on stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_KIND = "real_money_attestation"


class AttestationRefused(RuntimeError):
    """Raised when a positive flag is requested without verified evidence."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("attestation refused: " + "; ".join(self.reasons))


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _verify_reference(path: Path | None, *, require_json: bool = False) -> dict[str, Any] | None:
    """Verify a referenced evidence file and return an embeddable descriptor.

    Returns ``None`` when ``path`` is falsy.  Raises ``FileNotFoundError`` /
    ``ValueError`` (surfaced by the caller as a refusal reason) when the reference
    cannot be verified.
    """
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"evidence reference not found: {resolved}")
    raw = resolved.read_bytes()
    descriptor: dict[str, Any] = {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "verified": True,
    }
    if require_json:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"evidence reference is not valid JSON: {resolved} ({exc})") from exc
        if isinstance(parsed, dict):
            decision_value = str(parsed.get("decision") or "").strip()
            if decision_value:
                descriptor["decision"] = decision_value
    return descriptor


def build_attestation(
    *,
    operator_id: str = "",
    paper_stats: Path | None = None,
    fill_slippage_summary: Path | None = None,
    decision: Path | None = None,
    canary_run: Path | None = None,
    assert_ready_for_real: bool = False,
    assert_real_execution_allowed: bool = False,
    assert_real_money_execution: bool = False,
    assert_canary_execution_allowed: bool = False,
    record_canary_evidence: bool = False,
    clean_promotion_eligible: bool = True,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the attestation payload, refusing positive flags without evidence.

    A flag is only emitted as ``True`` when every piece of evidence it depends on is
    present and verifiable; otherwise :class:`AttestationRefused` is raised naming the
    missing/invalid references, and nothing should be written.
    """
    reasons: list[str] = []
    evidence: dict[str, Any] = {}

    wants_real = bool(
        assert_ready_for_real
        or assert_real_execution_allowed
        or assert_real_money_execution
        or assert_canary_execution_allowed
    )
    wants_any = wants_real or record_canary_evidence

    if wants_any and not str(operator_id or "").strip():
        reasons.append("operator_id is required to assert any positive flag")

    def _try_verify(name: str, path: Path | None, *, require_json: bool) -> None:
        if path is None:
            return
        try:
            descriptor = _verify_reference(path, require_json=require_json)
        except (FileNotFoundError, ValueError) as exc:
            reasons.append(str(exc))
            return
        if descriptor is not None:
            evidence[name] = descriptor

    _try_verify("paper_stats", paper_stats, require_json=False)
    _try_verify("fill_slippage_summary", fill_slippage_summary, require_json=False)
    _try_verify("decision_lineage", decision, require_json=True)
    _try_verify("canary_run", canary_run, require_json=False)

    if wants_real:
        for name in ("paper_stats", "fill_slippage_summary", "decision_lineage"):
            if name not in evidence:
                reasons.append(f"{name} evidence is required to assert real-money flags")

    if record_canary_evidence and "canary_run" not in evidence:
        reasons.append("canary_run evidence is required to record canary execution")

    if reasons:
        raise AttestationRefused(reasons)

    ready_for_real = bool(assert_ready_for_real)
    real_execution_allowed = bool(assert_real_execution_allowed)
    real_money_execution = bool(assert_real_money_execution)
    canary_execution_allowed = bool(assert_canary_execution_allowed)
    canary_execution_recorded = bool(record_canary_evidence)

    return {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": generated_at_utc or _utc_now_iso(),
        "operator_id": str(operator_id).strip(),
        "ready_for_real": ready_for_real,
        "real_execution_allowed": real_execution_allowed,
        "real_money_execution": real_money_execution,
        "canary_execution_allowed": canary_execution_allowed,
        "canary_execution_recorded": canary_execution_recorded,
        "clean_promotion_eligible": bool(clean_promotion_eligible),
        "paper_testnet_only": False,
        "evidence": evidence,
    }


def write_attestation(payload: dict[str, Any], *, output_json: Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-id", default="")
    parser.add_argument("--paper-stats", type=Path, default=None)
    parser.add_argument("--fill-slippage-summary", type=Path, default=None)
    parser.add_argument("--decision", type=Path, default=None)
    parser.add_argument("--canary-run", type=Path, default=None)
    parser.add_argument("--assert-ready-for-real", action="store_true")
    parser.add_argument("--assert-real-execution-allowed", action="store_true")
    parser.add_argument("--assert-real-money-execution", action="store_true")
    parser.add_argument("--assert-canary-execution-allowed", action="store_true")
    parser.add_argument("--record-canary-evidence", action="store_true")
    parser.add_argument(
        "--not-clean-promotion",
        action="store_true",
        help="mark clean_promotion_eligible=false (governance blocker)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        payload = build_attestation(
            operator_id=args.operator_id,
            paper_stats=args.paper_stats,
            fill_slippage_summary=args.fill_slippage_summary,
            decision=args.decision,
            canary_run=args.canary_run,
            assert_ready_for_real=args.assert_ready_for_real,
            assert_real_execution_allowed=args.assert_real_execution_allowed,
            assert_real_money_execution=args.assert_real_money_execution,
            assert_canary_execution_allowed=args.assert_canary_execution_allowed,
            record_canary_evidence=args.record_canary_evidence,
            clean_promotion_eligible=not args.not_clean_promotion,
        )
    except AttestationRefused as exc:
        print("REFUSED: real-money attestation not written.", file=sys.stderr)
        for reason in exc.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 2

    out = write_attestation(payload, output_json=args.output)
    print(json.dumps({"written": str(out), **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
