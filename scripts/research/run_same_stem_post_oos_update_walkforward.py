#!/usr/bin/env python3
"""Replay same-strategy post-OOS update rules as a monthly walk-forward.

The rule evaluated here is intentionally narrow: within a normalized strategy stem
(e.g. ``hybrid_v3_5``), month ``t`` may rank variants using only completed
post-OOS returns from folds before ``t``. The current fold's locked OOS is copied
only after the variant is selected, then becomes available for later folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_JSON = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_69_asset_no_nested_clean_recompute_20260604/"
    "no_nested_clean_recompute_latest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "var/reports/strategy_research/same_stem_post_oos_update_walkforward_20260619"
)
DEFAULT_OUTPUT_STEM = "same_stem_post_oos_update_walkforward_latest"
DEFAULT_TARGET_STEMS = ("hybrid_v3_5", "hybrid_v3_6", "fixed_relaxed_dynamic_blend")
FAMILY = "same_stem_post_oos_update"
POLICY = "lagged_top1_calmar"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object JSON at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _split_label(label: str) -> tuple[str, str]:
    family, sep, name = str(label).partition(":")
    return family if sep else "", name if sep else str(label)


def normalized_strategy_stem(label: str) -> str | None:
    """Return the same-strategy replay stem for a candidate label."""
    family, name = _split_label(label)
    lower = name.lower()
    if "hybrid_v3_5" in lower:
        return "hybrid_v3_5"
    if "hybrid_v3_6" in lower:
        return "hybrid_v3_6"
    if family == "fixed_relaxed_dynamic_blend":
        return "fixed_relaxed_dynamic_blend"
    if "balanced_mdd12_gross5" in lower:
        return "balanced_mdd12_gross5"
    if "growth_mdd20_gross8" in lower:
        return "growth_mdd20_gross8"
    if "aggressive_mdd30_gross10" in lower:
        return "aggressive_mdd30_gross10"
    if lower in {"selected_optuna", "selected_train_validation_legal", "static_guarded"}:
        return lower
    return None


def _locked_oos_return(row: Mapping[str, Any]) -> float:
    return _safe_float(dict(row.get("locked_oos") or {}).get("total_return"))


def _validation_calmar(row: Mapping[str, Any]) -> float:
    validation = dict(row.get("validation") or {})
    if validation.get("calmar") is not None:
        return _safe_float(validation.get("calmar"), default=-1e9)
    total_return = _safe_float(validation.get("total_return"), default=-1e9)
    mdd = max(_safe_float(validation.get("mdd"), default=0.0), 0.01)
    return total_return / mdd


def _compounded_metrics(returns: Sequence[float]) -> dict[str, Any]:
    equity = 1.0
    peak = 1.0
    monthly_equity_mdd = 0.0
    for item in returns:
        equity *= 1.0 + float(item)
        peak = max(peak, equity)
        if peak > 0.0:
            monthly_equity_mdd = max(monthly_equity_mdd, (peak - equity) / peak)
    compounded = equity - 1.0
    mean = statistics.mean(returns) if returns else 0.0
    volatility = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    sharpe = mean / volatility * math.sqrt(12.0) if volatility > 1e-12 else 0.0
    losses = [float(item) for item in returns if float(item) < 0.0]
    gains = [float(item) for item in returns if float(item) > 0.0]
    return {
        "compounded_return": compounded,
        "monthly_equity_mdd": monthly_equity_mdd,
        "monthly_sharpe_approx": sharpe,
        "positive_folds": len(gains),
        "fold_count": len(returns),
        "latest_return": float(returns[-1]) if returns else None,
        "min_return": min(returns) if returns else None,
        "avg_gain": statistics.mean(gains) if gains else 0.0,
        "avg_loss": statistics.mean(losses) if losses else 0.0,
    }


def _lagged_calmar_score(row: Mapping[str, Any], history: Sequence[float]) -> tuple[float, str]:
    if history:
        metrics = _compounded_metrics(history)
        mdd = max(float(metrics["monthly_equity_mdd"]), 0.01)
        return float(metrics["compounded_return"]) / mdd, "prior_completed_post_oos_calmar"
    return _validation_calmar(row), "bootstrap_validation_calmar"


def _candidate_rows_by_fold(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fold_id = str(row.get("fold_id") or "")
        if fold_id:
            by_fold[fold_id].append(dict(row))
    return dict(by_fold)


def replay_same_stem_post_oos_updates(
    rows: Sequence[Mapping[str, Any]], *, target_stems: Sequence[str]
) -> list[dict[str, Any]]:
    """Build replay rows using only prior completed post-OOS history per stem."""
    target_stem_set = {str(item) for item in target_stems}
    by_fold = _candidate_rows_by_fold(rows)
    histories: dict[str, list[float]] = defaultdict(list)
    completed_folds: list[str] = []
    replay_rows: list[dict[str, Any]] = []

    for fold_id in sorted(by_fold):
        fold_rows = by_fold[fold_id]
        for stem in sorted(target_stem_set):
            candidates = [
                row
                for row in fold_rows
                if normalized_strategy_stem(str(row.get("candidate_label") or "")) == stem
            ]
            if not candidates:
                continue
            scored: list[tuple[float, float, str, str, dict[str, Any], list[float]]] = []
            for row in candidates:
                label = str(row.get("candidate_label") or "")
                history = list(histories.get(label) or [])
                score, score_mode = _lagged_calmar_score(row, history)
                scored.append((score, _validation_calmar(row), label, score_mode, row, history))
            score, validation_score, selected_label, score_mode, selected, history = max(
                scored,
                key=lambda item: (item[0], item[1], item[2]),
            )
            replay_label = f"{FAMILY}:{stem}_{POLICY}"
            replay_row = dict(selected)
            replay_row.update(
                {
                    "family": FAMILY,
                    "candidate_label": replay_label,
                    "source_profile_id": replay_label,
                    "profile_id": replay_label,
                    "profile_kind": "same_stem_post_oos_update_walkforward_replay",
                    "candidate_tier": "post_oos_research_forward_shadow_only",
                    "strategy_stem": stem,
                    "selected_candidate_label": selected_label,
                    "selected_strategy_stem": stem,
                    "selection_policy": (
                        "same_stem_lagged_top1_calmar_current_fold_hidden_"
                        "bootstrap_validation_calmar"
                    ),
                    "selection_inputs": ["train", "validation", "lagged_completed_post_oos"],
                    "selection_score": float(score),
                    "selection_score_mode": score_mode,
                    "selected_validation_calmar": float(validation_score),
                    "lagged_completed_oos_history_count": len(history),
                    "lagged_completed_oos_history_tail": [float(item) for item in history[-4:]],
                    "online_update_cutoff_fold": completed_folds[-1] if completed_folds else None,
                    "weights": {selected_label: 1.0},
                    "final_weights": {selected_label: 1.0},
                    "uses_locked_oos_for_selection": False,
                    "current_fold_oos_used_for_weighting": False,
                    "same_month_self_feeding": False,
                    "post_oos_research_variant": True,
                    "requires_fresh_forward_shadow": True,
                    "nested_hybrid_dependency": bool(selected.get("nested_hybrid_dependency")),
                    "clean_promotion_eligible": False,
                    "ready_for_paper": True,
                    "ready_for_real": False,
                    "real_money_execution": False,
                    "paper_order_execution": False,
                    "non_clean_reasons": [
                        "post_oos_research_variant",
                        "requires_fresh_forward_shadow",
                    ],
                    "selection_reasons": [],
                }
            )
            replay_rows.append(replay_row)

        for row in fold_rows:
            label = str(row.get("candidate_label") or "")
            if normalized_strategy_stem(label) in target_stem_set:
                histories[label].append(_locked_oos_return(row))
        completed_folds.append(fold_id)

    return replay_rows


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("candidate_label") or "")].append(row)

    out: list[dict[str, Any]] = []
    for label, label_rows in grouped.items():
        ordered = sorted(label_rows, key=lambda row: str(row.get("fold_id") or ""))
        returns = [_locked_oos_return(row) for row in ordered]
        metrics = _compounded_metrics(returns)
        latest = ordered[-1]
        out.append(
            {
                "candidate_label": label,
                "family": str(latest.get("family") or ""),
                "strategy_stem": latest.get("strategy_stem") or normalized_strategy_stem(label),
                "selected_candidate_label_latest": latest.get("selected_candidate_label"),
                "selection_policy": latest.get("selection_policy"),
                "selection_inputs": list(latest.get("selection_inputs") or []),
                "fold_count": metrics["fold_count"],
                "compounded_return": metrics["compounded_return"],
                "monthly_equity_mdd": metrics["monthly_equity_mdd"],
                "monthly_sharpe_approx": metrics["monthly_sharpe_approx"],
                "positive_folds": metrics["positive_folds"],
                "latest_return": metrics["latest_return"],
                "min_return": metrics["min_return"],
                "mdd_gate_pass": float(metrics["monthly_equity_mdd"]) <= 0.30,
                "uses_locked_oos_for_selection": any(
                    bool(row.get("uses_locked_oos_for_selection")) for row in ordered
                ),
                "current_fold_oos_used_for_weighting": any(
                    bool(row.get("current_fold_oos_used_for_weighting")) for row in ordered
                ),
                "same_month_self_feeding": any(
                    bool(row.get("same_month_self_feeding")) for row in ordered
                ),
                "post_oos_research_variant": any(
                    bool(row.get("post_oos_research_variant")) for row in ordered
                ),
                "requires_fresh_forward_shadow": any(
                    bool(row.get("requires_fresh_forward_shadow")) for row in ordered
                ),
                "ready_for_real": any(bool(row.get("ready_for_real")) for row in ordered),
                "real_money_execution": any(
                    bool(row.get("real_money_execution")) for row in ordered
                ),
            }
        )
    return sorted(
        out,
        key=lambda item: (
            float(item["compounded_return"]),
            -float(item["monthly_equity_mdd"]),
            int(item["positive_folds"]),
        ),
        reverse=True,
    )


def _static_same_stem_diagnostics(
    rows: Sequence[Mapping[str, Any]], *, target_stems: Sequence[str]
) -> list[dict[str, Any]]:
    target_stem_set = set(target_stems)
    source_rows = [
        dict(row)
        for row in rows
        if normalized_strategy_stem(str(row.get("candidate_label") or "")) in target_stem_set
    ]
    return _aggregate_rows(source_rows)


def _leak_checks(
    source_rows: Sequence[Mapping[str, Any]], replay_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "source_fold_rows": len(source_rows),
        "replay_rows": len(replay_rows),
        "source_current_fold_oos_used_for_weighting_true": sum(
            1 for row in source_rows if bool(row.get("current_fold_oos_used_for_weighting"))
        ),
        "source_same_month_self_feeding_true": sum(
            1 for row in source_rows if bool(row.get("same_month_self_feeding"))
        ),
        "source_real_money_execution_true": sum(
            1 for row in source_rows if bool(row.get("real_money_execution"))
        ),
        "replay_current_fold_oos_used_for_weighting_true": sum(
            1 for row in replay_rows if bool(row.get("current_fold_oos_used_for_weighting"))
        ),
        "replay_uses_locked_oos_for_current_fold_selection_true": sum(
            1 for row in replay_rows if bool(row.get("uses_locked_oos_for_selection"))
        ),
        "replay_same_month_self_feeding_true": sum(
            1 for row in replay_rows if bool(row.get("same_month_self_feeding"))
        ),
        "replay_real_money_execution_true": sum(
            1 for row in replay_rows if bool(row.get("real_money_execution"))
        ),
        "replay_paper_order_execution_true": sum(
            1 for row in replay_rows if bool(row.get("paper_order_execution"))
        ),
    }


def build_payload(
    source_payload: Mapping[str, Any], *, source_path: Path, target_stems: Sequence[str]
) -> dict[str, Any]:
    source_rows = list(source_payload.get("fold_candidate_rows") or [])
    replay_rows = replay_same_stem_post_oos_updates(source_rows, target_stems=target_stems)
    aggregate_rankings = _aggregate_rows(replay_rows)
    static_diagnostics = _static_same_stem_diagnostics(source_rows, target_stems=target_stems)
    leak_checks = _leak_checks(source_rows, replay_rows)

    hard_fail_keys = (
        "source_current_fold_oos_used_for_weighting_true",
        "source_same_month_self_feeding_true",
        "source_real_money_execution_true",
        "replay_current_fold_oos_used_for_weighting_true",
        "replay_uses_locked_oos_for_current_fold_selection_true",
        "replay_same_month_self_feeding_true",
        "replay_real_money_execution_true",
        "replay_paper_order_execution_true",
    )
    if any(leak_checks[key] for key in hard_fail_keys):
        raise ValueError(f"same-stem post-OOS replay gate failed: {leak_checks}")

    folds = sorted({str(row.get("fold_id") or "") for row in source_rows if row.get("fold_id")})
    return {
        "artifact_kind": "same_stem_post_oos_update_walkforward_replay",
        "generated_at_utc": _utc_now_iso(),
        "status": "research_shadow_only_no_execution",
        "real_money_execution": False,
        "paper_order_execution": False,
        "method": {
            "target_stems": list(target_stems),
            "policy": POLICY,
            "policy_description": (
                "Within each strategy stem, fold t selects the candidate variant with the "
                "best Calmar from completed folds < t; if no completed post-OOS exists, "
                "it bootstraps from current train/validation Calmar only. Current-fold OOS "
                "is copied after selection and can only affect later folds."
            ),
            "global_strategy_rotation": False,
            "current_fold_oos_hidden_for_selection": True,
        },
        "source_artifact": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "folds": folds,
        "leak_checks": leak_checks,
        "aggregate_rankings": aggregate_rankings,
        "static_same_stem_diagnostics": static_diagnostics,
        "replay_rows": replay_rows,
        "hard_gates": {
            "mdd_limit": 0.30,
            "locked_oos_current_fold_selection": False,
            "same_month_self_feeding": False,
            "real_money_execution": False,
            "paper_order_execution": False,
            "promotion_eligible": False,
        },
    }


def _fmt_pct(value: Any) -> str:
    parsed = _safe_float(value)
    return f"{parsed:.2%}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    method = dict(payload.get("method") or {})
    leak_checks = dict(payload.get("leak_checks") or {})
    rankings = list(payload.get("aggregate_rankings") or [])
    static = list(payload.get("static_same_stem_diagnostics") or [])
    rows = list(payload.get("replay_rows") or [])
    lines = [
        "# Same-stem post-OOS update walk-forward replay",
        "",
        f"- generated: `{payload.get('generated_at_utc')}`",
        f"- status: `{payload.get('status')}`",
        "- real-money: `false`",
        "- paper order execution: `false`",
        f"- target stems: `{', '.join(method.get('target_stems') or [])}`",
        f"- policy: `{method.get('policy')}`",
        "- rule: current fold OOS is hidden for selection; completed prior post-OOS can update later folds within the same stem only.",
        "",
        "## Leak / gate checks",
        "",
    ]
    for key in [
        "source_fold_rows",
        "replay_rows",
        "source_current_fold_oos_used_for_weighting_true",
        "source_same_month_self_feeding_true",
        "source_real_money_execution_true",
        "replay_current_fold_oos_used_for_weighting_true",
        "replay_uses_locked_oos_for_current_fold_selection_true",
        "replay_same_month_self_feeding_true",
        "replay_real_money_execution_true",
        "replay_paper_order_execution_true",
    ]:
        lines.append(f"- {key}: `{leak_checks.get(key)}`")
    lines.extend(
        [
            "",
            "## Walk-forward replay ranking",
            "",
            "| Rank | Candidate | Stem | Comp | Monthly eq MDD | Sharpe | Hit | Latest | Gate |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, row in enumerate(rankings, start=1):
        lines.append(
            "| "
            f"{idx} | `{row.get('candidate_label')}` | `{row.get('strategy_stem')}` | "
            f"{_fmt_pct(row.get('compounded_return'))} | "
            f"{_fmt_pct(row.get('monthly_equity_mdd'))} | "
            f"{_safe_float(row.get('monthly_sharpe_approx')):.2f} | "
            f"{row.get('positive_folds')}/{row.get('fold_count')} | "
            f"{_fmt_pct(row.get('latest_return'))} | "
            f"{'MDD<=30 pass' if row.get('mdd_gate_pass') else 'MDD fail'} |"
        )
    h35_rows = [row for row in rows if row.get("strategy_stem") == "hybrid_v3_5"]
    if h35_rows:
        lines.extend(
            [
                "",
                "## H3.5 monthly selected variants",
                "",
                "| Fold | Selected variant | Score mode | History | OOS return |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in sorted(h35_rows, key=lambda item: str(item.get("fold_id") or "")):
            lines.append(
                "| "
                f"`{row.get('fold_id')}` | `{row.get('selected_candidate_label')}` | "
                f"`{row.get('selection_score_mode')}` | "
                f"{row.get('lagged_completed_oos_history_count')} | "
                f"{_fmt_pct(dict(row.get('locked_oos') or {}).get('total_return'))} |"
            )
    lines.extend(
        [
            "",
            "## Static same-stem diagnostics",
            "",
            "| Rank | Label | Stem | Comp | Monthly eq MDD | Sharpe | Hit | Read |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, row in enumerate(static[:20], start=1):
        lines.append(
            "| "
            f"{idx} | `{row.get('candidate_label')}` | `{row.get('strategy_stem')}` | "
            f"{_fmt_pct(row.get('compounded_return'))} | "
            f"{_fmt_pct(row.get('monthly_equity_mdd'))} | "
            f"{_safe_float(row.get('monthly_sharpe_approx')):.2f} | "
            f"{row.get('positive_folds')}/{row.get('fold_count')} | "
            "diagnostic only, not selector approval |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the requested walk-forward application of completed post-OOS history inside the same strategy stem.",
            "- The replay does not rotate globally across unrelated strategies.",
            "- Current-fold OOS is never used for same-fold selection or weighting.",
            "- All replay candidates remain post-OOS research / fresh-forward shadow only; no real-money or paper orders are enabled.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(payload: Mapping[str, Any], *, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    _write_json(json_path, payload)
    md_path.write_text(render_markdown(payload), "utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--target-stems", default=",".join(DEFAULT_TARGET_STEMS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = Path(args.source_json).expanduser().resolve()
    target_stems = tuple(item.strip() for item in str(args.target_stems).split(",") if item.strip())
    payload = build_payload(
        _read_json(source_path),
        source_path=source_path,
        target_stems=target_stems,
    )
    payload["output_paths"] = write_outputs(
        payload,
        output_dir=Path(args.output_dir).expanduser(),
        stem=str(args.output_stem),
    )
    _write_json(Path(payload["output_paths"]["json"]), payload)
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
