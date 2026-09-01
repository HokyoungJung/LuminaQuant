"""Validate the profit-moonshot continuation autoresearch gate.

The validator is intentionally small and artifact-driven.  It reads the latest
profit-moonshot summary, checks that the promoted live-equivalent candidate beats
our previous best validation return, and writes the autoresearch completion
artifact consumed by OMX hooks.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASELINE_MODE = "profit_moonshot_adaptive_momentum_mode"
BASELINE_VAL_RETURN = 0.0026493262
DEFAULT_SUMMARY_PATH = Path(
    "var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.json"
)
DEFAULT_RESULT_PATH = Path(".omx/specs/autoresearch-profit-moonshot-continuation/result.json")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if parsed == parsed and abs(parsed) != float("inf") else default


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def validate(summary_path: Path = DEFAULT_SUMMARY_PATH) -> dict[str, Any]:
    summary = _load_json(summary_path)
    promoted = dict(summary.get("promoted_candidate") or {})
    override = dict(summary.get("operator_oos_override") or {})
    override_candidate = dict(override.get("candidate") or {})
    ranked = [
        dict(item)
        for item in list(summary.get("ranked_candidates") or [])
        if isinstance(item, dict)
    ]
    improved_candidates = [
        item
        for item in ranked
        if bool(item.get("promotion_eligible"))
        and _safe_float(item.get("total_return"), 0.0) > BASELINE_VAL_RETURN
        and _safe_float(item.get("sharpe"), 0.0) > 0.0
        and _safe_float(item.get("sortino"), 0.0) > 0.0
        and int(_safe_float(item.get("trades"), 0.0)) >= 3
        and int(_safe_float(item.get("liquidations"), 0.0)) == 0
        and not list(item.get("blockers") or [])
    ]
    override_active = bool(override.get("user_gate_pass") and override_candidate)
    if override_active:
        override_candidate.setdefault("primary_split", "oos")
        override_candidate.setdefault("promotion_eligible", True)
    best = (
        override_candidate
        if override_active
        else (
            max(improved_candidates, key=lambda item: _safe_float(item.get("total_return"), 0.0))
            if improved_candidates
            else promoted or (ranked[0] if ranked else {})
        )
    )
    mode = str(best.get("mode") or best.get("candidate_id") or "")
    val_return = _safe_float(best.get("total_return"), 0.0)
    sharpe = _safe_float(best.get("sharpe"), 0.0)
    sortino = _safe_float(best.get("sortino"), 0.0)
    trades = int(_safe_float(best.get("trades"), 0.0))
    liquidations = int(_safe_float(best.get("liquidations"), 0.0))
    blockers = list(best.get("blockers") or [])
    improved = val_return > BASELINE_VAL_RETURN
    decision_ok = summary.get("decision") in {
        "promoted_candidate_found",
        "operator_oos_override_candidate_found",
    }
    passed = bool(
        decision_ok
        and improved
        and sharpe > 0.0
        and sortino > 0.0
        and trades >= 3
        and liquidations == 0
        and not blockers
    )
    primary_split = str(best.get("primary_split") or "")
    return {
        "status": "passed" if passed else "running",
        "passed": passed,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "summary_path": str(summary_path),
        "baseline_mode": BASELINE_MODE,
        "baseline_val_return": BASELINE_VAL_RETURN,
        "candidate_mode": mode,
        "candidate_val_return": val_return,
        "candidate_sharpe": sharpe,
        "candidate_sortino": sortino,
        "candidate_trades": trades,
        "candidate_liquidations": liquidations,
        "candidate_primary_split": primary_split,
        "improved_over_baseline": improved,
        "blockers": blockers,
        "promoted_by_summary": str(promoted.get("mode") or promoted.get("candidate_id") or ""),
        "operator_oos_override_active": override_active,
        "improved_candidate_count": len(improved_candidates),
        "summary": (
            f"{mode} {primary_split or 'candidate'}_return={val_return:.6f} vs baseline "
            f"{BASELINE_VAL_RETURN:.6f}; passed={passed}"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--result-path", default=str(DEFAULT_RESULT_PATH))
    args = parser.parse_args(argv)
    result = validate(Path(args.summary_path))
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
