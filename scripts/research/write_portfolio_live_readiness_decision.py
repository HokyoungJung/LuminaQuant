"""Write the canonical live-readiness decision artifact.

This bridges the gap between portfolio research selection artifacts and the
live-readiness preflight gate. It intentionally produces a small, explicit
decision bundle that can say either:
- keep_incumbent
- promote_candidate / selected_live_mode
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT

DEFAULT_SWITCH_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "current_switch_validation_current"
    / "refreshed_operating_switch_current"
    / "portfolio_operating_switch_latest.json"
)
DEFAULT_MAX_PERF_PATH = FOLLOWUP_ROOT / "portfolio_max_performance_decision_latest.json"
DEFAULT_REVIEW_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "portfolio_superiority_retune_current"
    / "strict_autoresearch_1x_practical_promotion_review_latest.json"
)
DEFAULT_OUTPUT_JSON = FOLLOWUP_ROOT / "portfolio_live_readiness_decision_latest.json"
DEFAULT_OUTPUT_MD = FOLLOWUP_ROOT / "portfolio_live_readiness_decision_latest.md"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _build_from_switch(switch_payload: dict[str, Any], *, switch_path: Path) -> dict[str, Any]:
    recommended_mode = dict(switch_payload.get("recommended_mode") or {})
    selected_mode = str(recommended_mode.get("mode") or "").strip()
    if not selected_mode:
        raise ValueError("switch artifact is missing recommended_mode.mode")
    rationale = [
        str(item) for item in list(switch_payload.get("rationale") or []) if str(item).strip()
    ]
    current_market_state = dict(switch_payload.get("current_market_state") or {})
    return {
        "artifact_kind": "portfolio_live_readiness_decision",
        "generated_at": _utc_now_iso(),
        "decision": "selected_live_mode",
        "selected_mode": selected_mode,
        "candidate_mode": selected_mode,
        "candidate_key": selected_mode,
        "selection_basis": "current_operating_switch",
        "decision_reason": rationale[-1]
        if rationale
        else f"Use the current operating switch recommendation: {selected_mode}.",
        "source_artifacts": {
            "switch_path": str(switch_path.resolve()),
        },
        "current_market_state": current_market_state,
    }


def _build_from_max_perf(
    max_perf_payload: dict[str, Any], *, max_perf_path: Path
) -> dict[str, Any]:
    winner = dict(max_perf_payload.get("winner") or {})
    winner_status = str(winner.get("status") or "").strip().lower()
    winner_key = str(winner.get("candidate_key") or "").strip()
    winner_label = str(winner.get("label") or "").strip()
    reason = str(winner.get("reason") or "").strip()

    if winner_status == "promoted_challenger":
        decision = "promote_candidate"
        candidate_reference = winner_key or winner_label
    else:
        decision = "keep_incumbent"
        candidate_reference = ""

    return {
        "artifact_kind": "portfolio_live_readiness_decision",
        "generated_at": _utc_now_iso(),
        "decision": decision,
        "candidate_key": candidate_reference,
        "candidate_mode": "",
        "selected_mode": "",
        "selection_basis": str(
            max_perf_payload.get("selection_basis") or "portfolio_max_performance_decision"
        ),
        "decision_reason": reason
        or (
            "No challenger cleared the locked-OOS robustness gates; keep the incumbent."
            if decision == "keep_incumbent"
            else f"Promote challenger {candidate_reference}."
        ),
        "source_artifacts": {
            "portfolio_max_performance_path": str(max_perf_path.resolve()),
        },
        "winner_status": winner_status,
        "winner_label": winner_label,
    }


def _build_from_review(review_payload: dict[str, Any], *, review_path: Path) -> dict[str, Any]:
    status = str(review_payload.get("status") or "").strip().lower()
    recommendation = str(review_payload.get("recommendation") or "").strip()
    review_target = str(review_payload.get("review_target") or "").strip()
    candidate_key = str(review_payload.get("candidate_key") or "").strip()
    if not candidate_key and review_target:
        candidate_key = Path(review_target).stem
    selected_mode = str(
        review_payload.get("selected_mode") or review_payload.get("candidate_mode") or ""
    ).strip()
    if not selected_mode and candidate_key == "strict_autoresearch_1x_practical_shadow_latest":
        selected_mode = "strict_autoresearch_practical_mode"

    if status == "promotion_ready_with_review":
        decision = "promote_candidate"
        reason = recommendation or f"Promote candidate {candidate_key} after manual review."
    else:
        decision = "keep_incumbent"
        reason = recommendation or "Promotion review did not clear the candidate."
        candidate_key = ""
        selected_mode = ""

    return {
        "artifact_kind": "portfolio_live_readiness_decision",
        "generated_at": _utc_now_iso(),
        "decision": decision,
        "candidate_key": candidate_key,
        "candidate_mode": selected_mode,
        "selected_mode": selected_mode,
        "selection_basis": str(
            review_payload.get("selection_basis") or "portfolio_promotion_review"
        ),
        "decision_reason": reason,
        "source_artifacts": {
            "promotion_review_path": str(review_path.resolve()),
        },
        "review_status": status,
        "current_live_default": str(review_payload.get("current_live_default") or ""),
    }


def build_live_readiness_decision(
    *,
    review_path: Path = DEFAULT_REVIEW_PATH,
    switch_path: Path = DEFAULT_SWITCH_PATH,
    max_perf_path: Path = DEFAULT_MAX_PERF_PATH,
    prefer_review: bool = False,
) -> dict[str, Any]:
    resolved_review_path = Path(review_path).resolve()
    if prefer_review and resolved_review_path.exists():
        return _build_from_review(
            _read_json(resolved_review_path), review_path=resolved_review_path
        )

    resolved_switch_path = Path(switch_path).resolve()
    if resolved_switch_path.exists():
        return _build_from_switch(
            _read_json(resolved_switch_path), switch_path=resolved_switch_path
        )

    resolved_max_perf_path = Path(max_perf_path).resolve()
    if resolved_max_perf_path.exists():
        return _build_from_max_perf(
            _read_json(resolved_max_perf_path), max_perf_path=resolved_max_perf_path
        )

    if resolved_review_path.exists():
        return _build_from_review(
            _read_json(resolved_review_path), review_path=resolved_review_path
        )

    raise FileNotFoundError(
        "No live-readiness source artifact found. "
        f"Missing review path={resolved_review_path}, switch path={resolved_switch_path}, "
        f"and max-performance path={resolved_max_perf_path}."
    )


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Portfolio live readiness decision",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- decision: `{payload.get('decision')}`",
        f"- selected_mode: `{payload.get('selected_mode') or ''}`",
        f"- candidate_key: `{payload.get('candidate_key') or ''}`",
        f"- selection_basis: `{payload.get('selection_basis')}`",
        f"- decision_reason: {payload.get('decision_reason')}",
    ]
    market_state = dict(payload.get("current_market_state") or {})
    if market_state:
        lines.extend(
            [
                "",
                "## Current market state",
                f"- favored_group: `{market_state.get('favored_group')}`",
                f"- confidence: `{market_state.get('confidence')}`",
                f"- trend_state: `{market_state.get('trend_state')}`",
                f"- breadth_state: `{market_state.get('breadth_state')}`",
                f"- volatility_state: `{market_state.get('volatility_state')}`",
                f"- pair_liquidity_state: `{market_state.get('pair_liquidity_state')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_live_readiness_decision(
    *,
    review_path: Path = DEFAULT_REVIEW_PATH,
    switch_path: Path = DEFAULT_SWITCH_PATH,
    max_perf_path: Path = DEFAULT_MAX_PERF_PATH,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    prefer_review: bool = False,
) -> dict[str, str]:
    payload = build_live_readiness_decision(
        review_path=review_path,
        switch_path=switch_path,
        max_perf_path=max_perf_path,
        prefer_review=prefer_review,
    )
    resolved_output_json = Path(output_json).resolve()
    resolved_output_md = Path(output_md).resolve()
    resolved_output_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    resolved_output_md.write_text(_build_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(resolved_output_json),
        "md_path": str(resolved_output_md),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--switch-path", type=Path, default=DEFAULT_SWITCH_PATH)
    parser.add_argument("--max-perf-path", type=Path, default=DEFAULT_MAX_PERF_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--prefer-review", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = write_live_readiness_decision(
        review_path=Path(args.review_path),
        switch_path=Path(args.switch_path),
        max_perf_path=Path(args.max_perf_path),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        prefer_review=bool(args.prefer_review),
    )
    print(result["json_path"])
    print(result["md_path"])


if __name__ == "__main__":
    main()
