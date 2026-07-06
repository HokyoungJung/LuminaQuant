#!/usr/bin/env python3
"""Bridge a strategy-factory / G005 candidate into a live-readiness decision.

Audit P2: G005's top families (e.g. ``abnormal_return_continuation``,
``last_day_liquidity_regime``, ``funding_liquidation_crowding_fade``) register their
class as ``live_opt_in`` but there was no tool to turn a
``candidate_research_*.json`` winner into a ``promote_candidate`` decision, and the
readiness gate ignored an explicit ``strategy_name``.  With the gate now honoring an
explicit ``strategy_name``/``strategy_class`` validated against
``registry.get_live_strategy_map(include_opt_in=True)``, this script emits that
decision so a new family becomes promotable **without editing** ``live_selection.py``.

Fail-closed: the referenced candidate must exist, must not be a hard-rejected /
non-passing row (unless ``--allow-unqualified``), and its strategy class must be in
the live (opt-in) registry map — otherwise nothing is written.

This decision does NOT assert real-money readiness on its own.  Real execution still
requires a *referenced* attestation artifact (see
``scripts/ops/write_real_money_attestation.py``); pass it via ``--attestation`` to
wire the reference into ``strategy_params`` so the readiness veto can read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_KIND = "portfolio_live_readiness_decision"


class DecisionRefused(RuntimeError):
    """Raised when a candidate cannot be promoted to a live decision."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _live_strategy_names() -> set[str]:
    """Registry names + class names in the live (opt-in) map; empty on failure."""
    try:
        from lumina_quant.strategies.registry import get_live_strategy_map

        live_map = get_live_strategy_map(include_opt_in=True)
    except Exception:  # pragma: no cover - registry import must never open a path
        return set()
    names: set[str] = set(live_map.keys())
    names.update(getattr(cls, "__name__", "") for cls in live_map.values())
    names.discard("")
    return names


def _find_candidate(
    research: Mapping[str, Any], *, candidate_id: str, candidate_name: str
) -> dict[str, Any]:
    candidates = research.get("candidates")
    if not isinstance(candidates, list):
        raise DecisionRefused("candidate_research payload has no 'candidates' list")
    wanted_id = str(candidate_id or "").strip()
    wanted_name = str(candidate_name or "").strip()
    if not wanted_id and not wanted_name:
        raise DecisionRefused("one of --candidate-id / --candidate-name is required")
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        if wanted_id and str(item.get("candidate_id") or "").strip() == wanted_id:
            return dict(item)
        if wanted_name and str(item.get("name") or "").strip() == wanted_name:
            return dict(item)
    target = wanted_id or wanted_name
    raise DecisionRefused(f"candidate not found in research artifact: {target}")


def build_strategy_factory_decision(
    *,
    research: Mapping[str, Any],
    research_path: Path,
    candidate_id: str = "",
    candidate_name: str = "",
    strategy_name_override: str = "",
    attestation_path: Path | None = None,
    allow_unqualified: bool = False,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    candidate = _find_candidate(research, candidate_id=candidate_id, candidate_name=candidate_name)

    resolved_id = str(candidate.get("candidate_id") or "").strip()
    resolved_name = str(candidate.get("name") or "").strip()
    strategy_class = str(
        strategy_name_override or candidate.get("strategy_class") or candidate.get("strategy") or ""
    ).strip()
    if not strategy_class:
        raise DecisionRefused(f"candidate {resolved_id or resolved_name} has no strategy_class")

    if not allow_unqualified:
        if bool(candidate.get("hard_reject")):
            reasons = candidate.get("hard_reject_reasons") or []
            raise DecisionRefused(
                f"candidate {resolved_id or resolved_name} is hard-rejected: {reasons} "
                "(pass --allow-unqualified to override)"
            )
        if candidate.get("pass") is False:
            raise DecisionRefused(
                f"candidate {resolved_id or resolved_name} did not pass selection "
                "(pass --allow-unqualified to override)"
            )

    live_names = _live_strategy_names()
    if strategy_class not in live_names:
        raise DecisionRefused(
            f"strategy '{strategy_class}' is not in the live (opt-in) registry map; "
            "register it as live_default/live_opt_in before promoting"
        )

    symbols = [
        str(sym).strip().upper() for sym in (candidate.get("symbols") or []) if str(sym).strip()
    ]
    params = candidate.get("params")
    strategy_params: dict[str, Any] = dict(params) if isinstance(params, Mapping) else {}
    if attestation_path is not None:
        strategy_params["real_money_attestation_artifact_path"] = str(
            Path(attestation_path).expanduser().resolve()
        )
    strategy_timeframe = str(
        candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""
    ).strip()

    reference = resolved_name or resolved_id or strategy_class
    decision: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": generated_at_utc or _utc_now_iso(),
        "decision": "promote_candidate",
        "selected_mode": reference,
        "candidate_mode": reference,
        "candidate_key": resolved_id or reference,
        "strategy_name": strategy_class,
        "strategy_class": strategy_class,
        "strategy_params": strategy_params,
        "symbols": symbols,
        "selection_basis": "strategy_factory_candidate_research",
        "decision_reason": (
            f"Promote strategy-factory candidate {reference} ({strategy_class}) "
            "from candidate research."
        ),
        "source_artifacts": {
            "candidate_research_path": str(Path(research_path).resolve()),
            "candidate_id": resolved_id,
        },
    }
    if strategy_timeframe:
        decision["strategy_timeframe"] = strategy_timeframe
    return decision


def write_decision(payload: dict[str, Any], *, output_json: Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-research", type=Path, required=True)
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--candidate-name", default="")
    parser.add_argument("--strategy-name", default="", help="override strategy_class")
    parser.add_argument("--attestation", type=Path, default=None)
    parser.add_argument("--allow-unqualified", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        research = _read_json(args.candidate_research)
        payload = build_strategy_factory_decision(
            research=research,
            research_path=args.candidate_research,
            candidate_id=args.candidate_id,
            candidate_name=args.candidate_name,
            strategy_name_override=args.strategy_name,
            attestation_path=args.attestation,
            allow_unqualified=args.allow_unqualified,
        )
    except (DecisionRefused, OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"REFUSED: live decision not written: {exc}", file=sys.stderr)
        return 2

    out = write_decision(payload, output_json=args.output)
    print(json.dumps({"written": str(out), **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
