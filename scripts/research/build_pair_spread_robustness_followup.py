"""Build a focused follow-up manifest for pair-spread robustness tightening."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(
    "var/reports/exact_window_backtests/followup_status/"
    "portfolio_incumbent_autoresearch_grouped/pair_spread_robustness_followup_current"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "var/reports/exact_window_backtests/followup_status/"
    "portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/"
    "article_pipeline_candidate_manifest_latest.json"
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _candidate_id(*, name: str, timeframe: str, params: dict[str, Any], symbols: list[str]) -> str:
    payload = {
        "name": name,
        "timeframe": str(timeframe),
        "params": params,
        "symbols": list(symbols),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    return parser


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _derive_candidate(
    template: dict[str, Any],
    *,
    name: str,
    params_override: dict[str, Any],
    notes_suffix: str,
) -> dict[str, Any]:
    candidate = deepcopy(template)
    params = dict(candidate.get("params") or {})
    params.update(params_override)
    candidate["name"] = name
    candidate["params"] = params
    candidate["notes"] = f"{str(candidate.get('notes') or '').rstrip()} {notes_suffix}".strip()
    candidate["candidate_id"] = _candidate_id(
        name=name,
        timeframe=str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""),
        params=params,
        symbols=list(candidate.get("symbols") or []),
    )
    metadata = dict(candidate.get("metadata") or {})
    metadata["followup_origin"] = "pair_spread_robustness_tightening"
    candidate["metadata"] = metadata
    tags = list(
        dict.fromkeys([*list(candidate.get("tags") or []), "followup", "pair_spread_robustness"])
    )
    candidate["tags"] = tags
    return candidate


def main() -> None:
    args = _build_parser().parse_args()
    source_manifest = Path(args.source_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_payload = _load_payload(source_manifest)
    by_name = {
        str(candidate.get("name")): dict(candidate)
        for candidate in list(source_payload.get("candidates") or [])
        if isinstance(candidate, dict)
    }

    followup_specs = [
        (
            "pair_spread_1h_robust_core_bnbusdt_trxusdt_2.4_0.60",
            "pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70",
            {
                "entry_z": 2.4,
                "exit_z": 0.60,
                "stop_z": 4.0,
                "min_correlation": 0.25,
                "cooldown_bars": 10,
                "reentry_z_buffer": 0.30,
                "max_hold_bars": 168,
                "min_z_turn": 0.05,
            },
            "Robustness follow-up: slightly lower entry with stronger re-entry / cooldown discipline.",
        ),
        (
            "pair_spread_1h_robust_exec_tp_bnbusdt_trxusdt_2.4_0.60",
            "pair_spread_1h_exec_takeprofit_bnbusdt_trxusdt_2.6_0.70",
            {
                "entry_z": 2.4,
                "exit_z": 0.60,
                "stop_z": 4.0,
                "min_correlation": 0.25,
                "cooldown_bars": 10,
                "reentry_z_buffer": 0.30,
                "max_hold_bars": 144,
                "take_profit_pct": 0.08,
                "min_z_turn": 0.05,
            },
            "Robustness follow-up: take-profit kept, but with shorter holds and stronger re-entry hysteresis.",
        ),
        (
            "pair_spread_1h_robust_exec_tight_bnbusdt_trxusdt_2.4_0.60",
            "pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.6_0.70",
            {
                "entry_z": 2.4,
                "exit_z": 0.60,
                "stop_z": 4.0,
                "min_correlation": 0.25,
                "cooldown_bars": 10,
                "reentry_z_buffer": 0.30,
                "max_hold_bars": 144,
                "take_profit_pct": 0.06,
                "min_z_turn": 0.05,
            },
            "Robustness follow-up: tighter stop/TP pair with less aggressive profit clipping.",
        ),
        (
            "pair_spread_1h_robust_state_vwap_bnbusdt_trxusdt_2.4_0.60",
            "pair_spread_1h_state_vwap_bnbusdt_trxusdt_2.6_0.70",
            {
                "entry_z": 2.4,
                "exit_z": 0.60,
                "stop_z": 4.0,
                "min_correlation": 0.30,
                "cooldown_bars": 10,
                "reentry_z_buffer": 0.30,
                "max_hold_bars": 144,
                "vwap_window": 72,
                "min_volume_window": 24,
                "min_volume_ratio": 0.25,
                "min_z_turn": 0.05,
            },
            "Robustness follow-up: VWAP + volume confirmation with stricter correlation floor.",
        ),
        (
            "pair_spread_1h_robust_state_atr_bnbusdt_trxusdt_2.4_0.60",
            "pair_spread_1h_state_atr_bnbusdt_trxusdt_2.6_0.70",
            {
                "entry_z": 2.4,
                "exit_z": 0.60,
                "stop_z": 4.0,
                "min_correlation": 0.30,
                "cooldown_bars": 10,
                "reentry_z_buffer": 0.30,
                "max_hold_bars": 144,
                "atr_window": 14,
                "atr_max_pct": 0.035,
                "min_z_turn": 0.05,
            },
            "Robustness follow-up: ATR gating tightened to suppress unstable spread entries.",
        ),
        (
            "pair_spread_1h_robust_core_strict_bnbusdt_trxusdt_2.6_0.70",
            "pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70",
            {
                "min_correlation": 0.30,
                "cooldown_bars": 12,
                "reentry_z_buffer": 0.35,
                "max_hold_bars": 144,
                "min_z_turn": 0.10,
            },
            "Robustness follow-up: stricter correlation/cooldown gating around the current positive-split core setup.",
        ),
        (
            "pair_spread_1h_robust_vwap_strict_bnbusdt_trxusdt_2.6_0.70",
            "pair_spread_1h_state_vwap_bnbusdt_trxusdt_2.6_0.70",
            {
                "min_correlation": 0.35,
                "cooldown_bars": 12,
                "reentry_z_buffer": 0.35,
                "max_hold_bars": 120,
                "vwap_window": 96,
                "min_volume_window": 24,
                "min_volume_ratio": 0.30,
                "min_z_turn": 0.10,
            },
            "Robustness follow-up: stricter VWAP confirmation plus shorter holding window.",
        ),
        (
            "pair_spread_1h_robust_hybrid_bnbusdt_trxusdt_2.4_0.60",
            "pair_spread_1h_exec_takeprofit_bnbusdt_trxusdt_2.6_0.70",
            {
                "entry_z": 2.4,
                "exit_z": 0.60,
                "stop_z": 4.0,
                "min_correlation": 0.30,
                "cooldown_bars": 10,
                "reentry_z_buffer": 0.30,
                "max_hold_bars": 144,
                "take_profit_pct": 0.08,
                "vwap_window": 72,
                "min_volume_window": 24,
                "min_volume_ratio": 0.25,
                "min_z_turn": 0.05,
            },
            "Robustness follow-up: hybrid take-profit + VWAP/volume confirmation around the best BNB/TRX lane.",
        ),
    ]

    candidates: list[dict[str, Any]] = []
    for name, template_name, params_override, notes_suffix in followup_specs:
        template = by_name.get(template_name)
        if template is None:
            raise KeyError(f"missing template candidate: {template_name}")
        candidates.append(
            _derive_candidate(
                template,
                name=name,
                params_override=params_override,
                notes_suffix=notes_suffix,
            )
        )

    manifest_payload = {
        "artifact_kind": "pair_spread_robustness_followup_manifest",
        "generated_at": _utc_now_iso(),
        "source_manifest": str(source_manifest),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    manifest_path = output_dir / "pair_spread_robustness_candidate_manifest_latest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    batch_payload = {
        "artifact_kind": "article_pipeline_research_batches",
        "generated_at": _utc_now_iso(),
        "source_manifest": str(manifest_path),
        "candidate_count": len(candidates),
        "batch_count": 1,
        "max_batch_candidates": len(candidates),
        "notes": [
            "Focused follow-up around BNB/TRX 1h pair-spread robustness tightening.",
        ],
        "batches": [
            {
                "batch_id": "followup_batch_01",
                "family": "market_neutral_followup",
                "strategy_class": "PairSpreadZScoreStrategy",
                "timeframe": "1h",
                "candidate_count": len(candidates),
                "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
                "candidate_names": [candidate["name"] for candidate in candidates],
                "start_index": 0,
                "end_index": len(candidates) - 1,
            }
        ],
    }
    batches_path = output_dir / "pair_spread_robustness_batches_latest.json"
    batches_path.write_text(json.dumps(batch_payload, indent=2, sort_keys=True), encoding="utf-8")

    md_path = output_dir / "pair_spread_robustness_followup_latest.md"
    lines = [
        "# pair spread robustness follow-up",
        "",
        f"- generated_at: `{manifest_payload['generated_at']}`",
        f"- source_manifest: `{source_manifest}`",
        f"- candidate_count: `{len(candidates)}`",
        "",
        "## candidates",
        *[f"- {candidate['name']}" for candidate in candidates],
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(manifest_path)
    print(batches_path)
    print(md_path)


if __name__ == "__main__":
    main()
