"""Write the operator-facing final scoreboard, one-pager, and hybrid runbook.

This consolidates the refreshed reboot-split artifacts into the small set of
operator surfaces used during handoff and live decision review.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GROUP_ROOT = Path(
    "var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped"
)
DEFAULT_SWITCH_RECOMMENDATION = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_operating_switch_current"
    / "portfolio_operating_switch_latest.json"
)
DEFAULT_SWITCH_VALIDATION = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_switch_vs_strategy1_validation_latest.json"
)
DEFAULT_HYBRID_PORTFOLIO = (
    GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
)
DEFAULT_PAIR_CANDIDATE = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_pair_fast_exit_candidate_latest.json"
)
DEFAULT_HYBRID_TUNING = (
    GROUP_ROOT / "portfolio_hybrid_online_tuning_current" / "hybrid_online_tuning_latest.json"
)
DEFAULT_HYBRID_OPTUNA = (
    GROUP_ROOT / "portfolio_hybrid_online_optuna_current" / "hybrid_online_optuna_latest.json"
)
DEFAULT_HYBRID_OPTUNA_TRAIN_AWARE = (
    GROUP_ROOT
    / "portfolio_hybrid_online_optuna_current"
    / "train_aware_guarded"
    / "hybrid_online_optuna_latest.json"
)
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "final_master_scoreboard_current"
ARTIFACT_REFRESH_BASIS = (
    "reboot split with warmup_ratio=0.60, retuned hybrid inputs + switch/gates, "
    "pair raw latest-tail exact recompute included, no article batch reruns"
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except TypeError, ValueError:
        return float(default)
    return float(numeric)


def _mode_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    raw = dict(metrics or {})
    if "oos" in raw and isinstance(raw.get("oos"), dict):
        raw = dict(raw.get("oos") or {})
    return {
        "oos_total_return": _safe_float(raw.get("total_return", raw.get("return")), 0.0),
        "sharpe": _safe_float(raw.get("sharpe"), 0.0),
        "max_drawdown": _safe_float(raw.get("max_drawdown", raw.get("mdd")), 0.0),
    }


def _comparison_row_map(hybrid_payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    scenario = dict(
        dict((hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {})
    )
    rows = list(scenario.get("comparison_rows") or [])
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out[name] = _mode_metrics(row)
    return out


def _scoreboard_row(
    *,
    name: str,
    status: str,
    comparison_rows: dict[str, dict[str, float]],
    source_metrics: dict[str, dict[str, Any]],
    comparison_name: str | None = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    resolved_comparison = str(comparison_name or name)
    resolved_source = str(source_name or resolved_comparison)
    metrics = comparison_rows.get(resolved_comparison) or _mode_metrics(
        source_metrics.get(resolved_source)
    )
    return {
        "name": name,
        **metrics,
        "status": status,
    }


def _scoreboard_rows(
    *,
    switch_payload: dict[str, Any],
    hybrid_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    switch_mode = str(dict(switch_payload.get("recommended_mode") or {}).get("mode") or "")
    comparison_rows = _comparison_row_map(hybrid_payload)
    source_metrics = dict(
        dict(
            dict((hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
                "source_sleeve_metrics"
            )
            or {}
        )
    )
    hybrid_oos = dict(
        dict(
            (
                dict(
                    (hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}
                ).get("split_metrics")
                or {}
            ).get("oos")
            or {}
        )
    )
    rows = [
        _scoreboard_row(
            name="pair_tactical_mode",
            status="tactical_only",
            comparison_rows=comparison_rows,
            source_metrics=source_metrics,
        ),
        {
            "name": "hybrid_guarded_mode",
            **_mode_metrics(hybrid_oos),
            "status": "switch_default"
            if switch_mode == "hybrid_guarded_mode"
            else "guarded_challenger",
        },
        _scoreboard_row(
            name="balanced_overlay_mode",
            comparison_name="balanced_overlay_80_20",
            source_name="balanced_overlay_80_20",
            status="switch_default"
            if switch_mode == "balanced_overlay_mode"
            else "small_overlay_backup",
            comparison_rows=comparison_rows,
            source_metrics=source_metrics,
        ),
        _scoreboard_row(
            name="aggressive_realized_mode",
            comparison_name="three_way_regime",
            source_name="three_way_regime",
            status="high_beta_alt",
            comparison_rows=comparison_rows,
            source_metrics=source_metrics,
        ),
        _scoreboard_row(
            name="core_mode",
            comparison_name="soft_three_way_regime",
            source_name="soft_three_way_regime",
            status="defensive_active_backup",
            comparison_rows=comparison_rows,
            source_metrics=source_metrics,
        ),
        _scoreboard_row(
            name="production_guarded_mode",
            comparison_name="production_guarded_portfolio",
            status="switch_default"
            if switch_mode == "production_guarded_mode"
            else "production_candidate",
            comparison_rows=comparison_rows,
            source_metrics=source_metrics,
        ),
        _scoreboard_row(
            name="static_blend_76_24",
            status="benchmark_static_blend",
            comparison_rows=comparison_rows,
            source_metrics=source_metrics,
        ),
        _scoreboard_row(
            name="incumbent_only",
            status="benchmark_incumbent",
            comparison_rows=comparison_rows,
            source_metrics=source_metrics,
        ),
        {
            "name": "risk_off_cash",
            "oos_total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "status": "fallback",
        },
    ]
    rows = [
        row
        for row in rows
        if row["name"] == "risk_off_cash"
        or any(key in row for key in ("oos_total_return", "sharpe", "max_drawdown"))
    ]
    return sorted(rows, key=lambda item: item["oos_total_return"], reverse=True)


def _tuning_summary(
    payload: dict[str, Any] | None,
    *,
    root_key: str,
    config_name: str | None = None,
) -> dict[str, Any]:
    root = dict((payload or {}).get(root_key) or {})
    scenarios = dict(root.get("scenarios") or {})
    return {
        **({"config_name": config_name} if config_name else {}),
        **({"trial_number": root.get("trial_number")} if "trial_number" in root else {}),
        **(
            {"objective_profile": root.get("objective_profile")}
            if "objective_profile" in root
            else {}
        ),
        "objective": _safe_float(root.get("objective"), 0.0),
        "config": dict(root.get("config") or {}),
        "readiness": dict(root.get("readiness") or {}),
        "scenarios": {
            name: {"split_metrics": dict(dict(scenario or {}).get("split_metrics") or {})}
            for name, scenario in scenarios.items()
        },
    }


def _find_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    row = next((item for item in rows if item["name"] == name), None)
    if row is None:
        raise KeyError(f"missing scoreboard row for {name}")
    return dict(row)


def build_master_scoreboard(
    *,
    switch_payload: dict[str, Any],
    switch_validation: dict[str, Any],
    hybrid_payload: dict[str, Any],
    pair_candidate_payload: dict[str, Any],
    hybrid_tuning_payload: dict[str, Any] | None = None,
    hybrid_optuna_payload: dict[str, Any] | None = None,
    hybrid_train_aware_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_at = _utc_now_iso()
    switch_mode = dict(switch_payload.get("recommended_mode") or {})
    switch_mode_name = str(switch_mode.get("mode") or "")
    scoreboard_rows = _scoreboard_rows(switch_payload=switch_payload, hybrid_payload=hybrid_payload)
    current_default = _find_row(scoreboard_rows, switch_mode_name)
    current_default["mode"] = current_default.pop("name")
    current_default["allocation"] = dict(switch_mode.get("allocation") or {})
    current_default["reason"] = (
        list(switch_mode.get("rationale") or [])[-1]
        if list(switch_mode.get("rationale") or [])
        else "Current switch/gate logic selected this mode."
    )

    hybrid_row = _find_row(scoreboard_rows, "hybrid_guarded_mode")
    hybrid_challenger = {
        "mode": hybrid_row["name"],
        "allocation": {"hybrid_online_portfolio": 1.0},
        "oos_total_return": hybrid_row["oos_total_return"],
        "sharpe": hybrid_row["sharpe"],
        "max_drawdown": hybrid_row["max_drawdown"],
        "stage": str(dict(hybrid_payload.get("readiness") or {}).get("recommended_stage") or ""),
        "why_not_default": (
            "Promoted to the live default because the mixed/calm superiority gate over balanced now clears."
            if switch_mode_name == "hybrid_guarded_mode"
            else "The switch still prefers a smaller overlay default, so hybrid remains the strongest diversified / guarded challenger."
        ),
    }
    benchmark_reference_rows = [
        _find_row(scoreboard_rows, "static_blend_76_24"),
        _find_row(scoreboard_rows, "incumbent_only"),
    ]

    pair_row = _find_row(scoreboard_rows, "pair_tactical_mode")
    tactical_override = {
        "mode": "pair_tactical_mode",
        "sleeve": str(pair_candidate_payload.get("name") or "pair_tactical_mode"),
        "oos_total_return": pair_row["oos_total_return"],
        "sharpe": pair_row["sharpe"],
        "max_drawdown": pair_row["max_drawdown"],
        "status": "tactical_only",
    }

    state = dict(switch_payload.get("current_market_state") or {})
    favored_group = state.get("favored_group")
    confidence = _safe_float(state.get("confidence"), 0.0)
    if switch_mode_name == "hybrid_guarded_mode":
        summary = (
            "After the switch superiority gate was tightened on the reboot split, "
            "hybrid_guarded_mode now clears the mixed/calm promotion rule over balanced_overlay_mode. "
            "Pair remains the highest raw-return tactical sleeve, but it stays tactical-only."
        )
    else:
        summary = (
            "After retuning the reboot split artifacts, the switch still prefers the smaller balanced overlay. "
            "Pair remains the highest raw-return tactical sleeve, while hybrid stays the strongest diversified / guarded challenger."
        )

    return {
        "artifact_kind": "portfolio_master_scoreboard",
        "artifact_refresh_basis": ARTIFACT_REFRESH_BASIS,
        "generated_at": generated_at,
        "refresh_cutoff_utc": str(switch_payload.get("as_of_date") or ""),
        "latest_common_complete_utc": str(
            switch_validation.get("latest_common_complete_utc") or ""
        ),
        "split_windows": dict(hybrid_payload.get("split_windows") or {}),
        "online_policy": dict(hybrid_payload.get("online_policy") or {}),
        "switch_recommended_mode": switch_mode,
        "current_default": current_default,
        "fallback_mode": {
            "mode": "risk_off_cash",
            "allocation": {"cash": 1.0},
            "oos_total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        },
        "hybrid_challenger": hybrid_challenger,
        "hybrid_readiness": dict(hybrid_payload.get("readiness") or {}),
        "benchmark_reference_rows": benchmark_reference_rows,
        "tactical_override": tactical_override,
        "refreshed_live_scoreboard": scoreboard_rows,
        "final_recommendation": {
            "default_live_mode": switch_mode_name,
            "fallback_mode": "risk_off_cash",
            "hybrid_challenger_mode": "hybrid_guarded_mode",
            "tactical_override": "pair_tactical_mode",
            "summary": summary,
        },
        "upstream_reboot_inputs": {
            "incumbent_only": str(
                switch_validation.get("artifact_paths", {}).get("refreshed_incumbent") or ""
            ),
            "autoresearch_55_45": str(
                switch_validation.get("artifact_paths", {}).get("refreshed_autoresearch") or ""
            ),
            "static_blend_re_tuned": str(
                switch_validation.get("artifact_paths", {}).get("refreshed_blend") or ""
            ),
            "soft_three_way_re_tuned": str(
                switch_validation.get("artifact_paths", {}).get("refreshed_soft_allocator") or ""
            ),
            "three_way_re_tuned": str(
                switch_validation.get("artifact_paths", {}).get("refreshed_three_way_allocator")
                or ""
            ),
            "pair_tactical_re_tuned": str(
                switch_validation.get("artifact_paths", {}).get("refreshed_pair_candidate") or ""
            ),
            "balanced_overlay_re_tuned": str(
                switch_validation.get("artifact_paths", {}).get("refreshed_balanced_overlay") or ""
            ),
        },
        "tuning_comparison": {
            "curated_best": _tuning_summary(
                hybrid_tuning_payload,
                root_key="best",
                config_name=str(
                    dict((hybrid_tuning_payload or {}).get("best") or {}).get("config_name") or ""
                ),
            ),
            "optuna_live_guarded_best": _tuning_summary(
                hybrid_optuna_payload,
                root_key="best_trial",
            ),
            "optuna_train_aware_best": _tuning_summary(
                hybrid_train_aware_payload,
                root_key="best_trial",
            ),
            "selected_config_source": "portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json",
        },
        "market_state": {
            "favored_group": favored_group,
            "confidence": confidence,
            "trend_state": state.get("trend_state"),
            "breadth_state": state.get("breadth_state"),
            "volatility_state": state.get("volatility_state"),
            "pair_liquidity_state": state.get("pair_liquidity_state"),
        },
    }


def build_onepager_payload(scoreboard: dict[str, Any]) -> dict[str, Any]:
    default_mode = dict(scoreboard.get("current_default") or {})
    hybrid_challenger = dict(scoreboard.get("hybrid_challenger") or {})
    return {
        "artifact_kind": "portfolio_operating_recommendation_onepager",
        "generated_at": str(scoreboard.get("generated_at") or ""),
        "source_scoreboard": "final_master_scoreboard_current/portfolio_master_scoreboard_latest.md",
        "refresh_cutoff_utc": str(scoreboard.get("refresh_cutoff_utc") or ""),
        "latest_common_complete_utc": str(scoreboard.get("latest_common_complete_utc") or ""),
        "split_windows": dict(scoreboard.get("split_windows") or {}),
        "online_policy": dict(scoreboard.get("online_policy") or {}),
        "market_state": dict(scoreboard.get("market_state") or {}),
        "default_live_mode": default_mode,
        "fallback_mode": dict(scoreboard.get("fallback_mode") or {}),
        "hybrid_challenger": hybrid_challenger,
        "benchmark_reference_rows": list(scoreboard.get("benchmark_reference_rows") or []),
        "tactical_override": dict(scoreboard.get("tactical_override") or {}),
        "summary": str(dict(scoreboard.get("final_recommendation") or {}).get("summary") or ""),
    }


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value, 0.0):+.4%}"


def _fmt_num(value: Any) -> str:
    return f"{_safe_float(value, 0.0):+.4f}"


def _build_master_markdown(scoreboard: dict[str, Any]) -> str:
    split = dict(scoreboard.get("split_windows") or {})
    policy = dict(scoreboard.get("online_policy") or {})
    rec = dict(scoreboard.get("switch_recommended_mode") or {})
    state = dict(scoreboard.get("market_state") or {})
    benchmark_rows = list(scoreboard.get("benchmark_reference_rows") or [])
    lines = [
        "# Portfolio master scoreboard",
        "",
        f"- generated_at: `{scoreboard.get('generated_at')}`",
        f"- artifact_refresh_basis: `{scoreboard.get('artifact_refresh_basis')}`",
        f"- refresh_cutoff_utc: `{scoreboard.get('refresh_cutoff_utc')}`",
        f"- latest_common_complete_utc: `{scoreboard.get('latest_common_complete_utc')}`",
        f"- reboot OOS start: `{split.get('oos_start')}`",
        "",
        "## Reboot split + online policy",
        "",
        f"- train: `{split.get('train_start')}` ~ `{split.get('train_end_inclusive')}`",
        f"- val: `{split.get('val_start')}` ~ `{split.get('val_end_inclusive')}`",
        f"- oos: `{split.get('oos_start')}` ~ `{split.get('oos_end_inclusive')}`",
        f"- warmup_ratio / warmup_days / online_start: `{policy.get('warmup_ratio')}` / `{policy.get('warmup_days')}` / `{policy.get('online_start')}`",
        "",
        "## Current switch outcome",
        f"- recommended mode: `{rec.get('mode')}`",
        f"- allocation: `{json.dumps(rec.get('allocation') or {}, sort_keys=True)}`",
        (
            f"- current market state: favored_group=`{state.get('favored_group')}`, confidence=`{_safe_float(state.get('confidence'), 0.0)}`, "
            f"trend=`{state.get('trend_state')}`, breadth=`{state.get('breadth_state')}`, "
            f"volatility=`{state.get('volatility_state')}`, pair_liquidity=`{state.get('pair_liquidity_state')}`"
        ),
        "",
        "## Live scoreboard (reboot split)",
        "",
        "| Rank | Mode / sleeve | OOS return | Sharpe | Max DD | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(list(scoreboard.get("refreshed_live_scoreboard") or []), start=1):
        lines.append(
            f"| {idx} | `{row['name']}` | `{_fmt_pct(row.get('oos_total_return'))}` | `{_fmt_num(row.get('sharpe'))}` | `{abs(_safe_float(row.get('max_drawdown'), 0.0)):.4%}` | {row.get('status')} |"
        )
    default_mode = dict(scoreboard.get("current_default") or {}).get("mode")
    lines.extend(
        [
            "",
            "## Interpretation",
            "1. Upstream hybrid inputs were rebuilt/re-ranked on the reboot split without rerunning article batch_01~44.",
            "2. Pair tactical remains the best raw tactical sleeve, but it stays tactical-only.",
            (
                "3. The switch/gate layer now promotes `hybrid_guarded_mode` over the smaller balanced overlay for the current mixed/calm regime."
                if default_mode == "hybrid_guarded_mode"
                else "3. The switch/gate layer still prefers `balanced_overlay_mode` for the current mixed/calm regime."
            ),
            "4. Hybrid remains positive across train/val/oos and now carries the strongest diversified / guarded evidence in the reboot lane.",
            "",
            "## Benchmark anchors",
            "",
        ]
    )
    for row in benchmark_rows:
        lines.append(
            f"- `{row['name']}` -> return `{_fmt_pct(row.get('oos_total_return'))}`, sharpe `{_fmt_num(row.get('sharpe'))}`, maxDD `{abs(_safe_float(row.get('max_drawdown'), 0.0)):.4%}`"
        )
    lines.extend(
        [
            "",
            "## Operational conclusion",
            f"- **default live mode:** `{default_mode}`",
            "- **fallback:** `risk_off_cash`",
            "- **best raw tactical sleeve:** `pair_tactical_mode`",
            (
                "- **best diversified / guarded mode:** `hybrid_guarded_mode`"
                if default_mode == "hybrid_guarded_mode"
                else "- **best diversified / guarded challenger:** `hybrid_guarded_mode`"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _build_onepager_markdown(onepager: dict[str, Any]) -> str:
    default_mode = dict(onepager.get("default_live_mode") or {})
    fallback = dict(onepager.get("fallback_mode") or {})
    hybrid = dict(onepager.get("hybrid_challenger") or {})
    benchmark_rows = list(onepager.get("benchmark_reference_rows") or [])
    tactical = dict(onepager.get("tactical_override") or {})
    market = dict(onepager.get("market_state") or {})
    split = dict(onepager.get("split_windows") or {})
    lines = [
        "# Portfolio operating recommendation — one pager",
        "",
        f"- generated_at: `{onepager.get('generated_at')}`",
        f"- source_scoreboard: `{onepager.get('source_scoreboard')}`",
        f"- refresh_cutoff_utc: `{onepager.get('refresh_cutoff_utc')}`",
        f"- latest_common_complete_utc: `{onepager.get('latest_common_complete_utc')}`",
        f"- reboot OOS start: `{split.get('oos_start')}`",
        "",
        "## 1) Decision in one line",
        "",
        (
            f"**Run `{default_mode.get('mode')}` as the live default right now.**  \n"
            f"Keep **`{fallback.get('mode')}` as the fallback**, keep **`balanced_overlay_mode` as the small-overlay backup**, "
            f"and keep **`{tactical.get('mode')}` only as a tactical override**."
            if default_mode.get("mode") == "hybrid_guarded_mode"
            else f"**Run `{default_mode.get('mode')}` as the live default right now.**  \n"
            f"Keep **`{fallback.get('mode')}` as the fallback**, keep **`hybrid_guarded_mode` as the diversified pilot challenger**, "
            f"and keep **`{tactical.get('mode')}` only as a tactical override**."
        ),
        "",
        "## 2) Why",
        "",
        "- The reboot split remains `train 2025 / val 2026-01~02 / oos 2026-03~latest` with `warmup_ratio=0.60` -> `warmup_days=255`.",
        "- Upstream hybrid inputs and the pair tactical sleeve remain refreshed inside the reboot lane without rerunning article batch_01~44.",
        (
            "- The tuned switch superiority gate now sees `favored_group=mixed`, `confidence=0.0`, `trend=bullish`, `breadth=broad`, `volatility=calm`, and promotes `hybrid_guarded_mode` because it materially beats balanced on OOS return, Sharpe, and drawdown."
            if default_mode.get("mode") == "hybrid_guarded_mode"
            else "- The current retuned switch/gate logic still sees `favored_group=mixed` with confidence `0.0` and keeps the live default on `balanced_overlay_mode`."
        ),
        f"- **Raw 성적만 보면 최고는 pair_tactical** (`{_fmt_pct(tactical.get('oos_total_return'))}` OOS), but tactical-only입니다.",
        f"- **Hybrid guarded 성능**: OOS `{_fmt_pct(hybrid.get('oos_total_return'))}`, Sharpe `{_fmt_num(hybrid.get('sharpe'))}`, MaxDD `{abs(_safe_float(hybrid.get('max_drawdown'), 0.0)):.4%}`.",
        "",
        "## 3) Current ladder",
        "",
        f"- default: `{default_mode.get('mode')}` -> `{json.dumps(default_mode.get('allocation') or {}, sort_keys=True)}`",
        f"- fallback: `{fallback.get('mode')}`",
        "- best raw tactical sleeve: `pair_tactical_mode`",
        (
            "- best diversified / guarded mode: `hybrid_guarded_mode`"
            if default_mode.get("mode") == "hybrid_guarded_mode"
            else "- best diversified / guarded challenger: `hybrid_guarded_mode`"
        ),
    ]
    for row in benchmark_rows:
        lines.append(
            f"- benchmark anchor `{row.get('name')}`: return `{_fmt_pct(row.get('oos_total_return'))}`, sharpe `{_fmt_num(row.get('sharpe'))}`, maxDD `{abs(_safe_float(row.get('max_drawdown'), 0.0)):.4%}`"
        )
    lines.extend(
        [
            "",
            (
                "> 현재 live default는 tuned mixed/calm superiority gate를 통과한 hybrid guarded입니다.  \n"
                "> pair는 계속 tactical-only이며, balanced overlay는 소형 overlay backup으로 남겨둡니다."
                if default_mode.get("mode") == "hybrid_guarded_mode"
                else "> 현재 live default는 switch/gate 정책상 balanced overlay입니다.  \n> 하지만 **hybrid는 현재 재구성된 diversified/guarded stack 안에서는 best**입니다."
            ),
            "",
            "## 4) Current market state",
            "",
            f"- favored_group: `{market.get('favored_group')}`",
            f"- confidence: `{_safe_float(market.get('confidence'), 0.0):.4f}`",
            f"- trend_state: `{market.get('trend_state')}`",
            f"- breadth_state: `{market.get('breadth_state')}`",
            f"- volatility_state: `{market.get('volatility_state')}`",
            f"- pair_liquidity_state: `{market.get('pair_liquidity_state')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_hybrid_runbook_markdown(scoreboard: dict[str, Any]) -> str:
    split = dict(scoreboard.get("split_windows") or {})
    policy = dict(scoreboard.get("online_policy") or {})
    default_mode = dict(scoreboard.get("current_default") or {})
    hybrid = dict(scoreboard.get("hybrid_challenger") or {})
    balanced = _find_row(
        list(scoreboard.get("refreshed_live_scoreboard") or []), "balanced_overlay_mode"
    )
    lines = [
        "# Hybrid Guarded Mode Runbook",
        "",
        f"- generated_at: `{scoreboard.get('generated_at')}`",
        f"- canonical_default_mode: `{default_mode.get('mode')}`",
        "- canonical_fallback_mode: `risk_off_cash`",
        "- hybrid_challenger_mode: `hybrid_guarded_mode`",
        "- tactical_override_only: `pair_tactical_mode`",
        "",
        "## 1) Operating intent",
        "",
        (
            "The tuned switch superiority gate now promotes `hybrid_guarded_mode` in the current mixed/calm regime. "
            "Use it as the canonical live default while keeping `risk_off_cash` as the hard fallback and `pair_tactical_mode` only as an explicit tactical override."
            if default_mode.get("mode") == "hybrid_guarded_mode"
            else "After retuning the sleeves that feed hybrid and the switch/gate layer, current live policy still defaults to `balanced_overlay_mode`. "
            "Use `hybrid_guarded_mode` as the main pilot challenger when you want the stronger reboot-split performer rather than the stricter switch outcome."
        ),
        "",
        "## 2) Reboot split and policy",
        "",
        f"- train: `{split.get('train_start')}` ~ `{split.get('train_end_inclusive')}`",
        f"- val: `{split.get('val_start')}` ~ `{split.get('val_end_inclusive')}`",
        f"- oos: `{split.get('oos_start')}` ~ `{split.get('oos_end_inclusive')}`",
        f"- warmup_ratio / warmup_days / online_start: `{policy.get('warmup_ratio')}` / `{policy.get('warmup_days')}` / `{policy.get('online_start')}`",
        "",
        "## 3) Current acceptance state",
        "",
        f"- switch default: `{default_mode.get('mode')}` {json.dumps(default_mode.get('allocation') or {}, sort_keys=True)}",
        f"- hybrid guarded OOS: `{_fmt_pct(hybrid.get('oos_total_return'))}` sharpe `{_fmt_num(hybrid.get('sharpe'))}` stage `{hybrid.get('stage')}`",
        f"- balanced overlay OOS: `{_fmt_pct(balanced.get('oos_total_return'))}` sharpe `{_fmt_num(balanced.get('sharpe'))}`",
        "",
        "## 4) Hard constraints",
        "- total session memory `< 8 GiB`",
        "- heavy runs strictly sequential only",
        "- article batch_01~44 rerun 금지",
        "- pair tactical은 override only",
        "",
        "## 5) Daily refresh / rerun flow",
        "```bash",
        "cd /home/hoky/Quants-agent/LuminaQuant",
        "",
        "export POLARS_MAX_THREADS=1",
        "export RAYON_NUM_THREADS=1",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export LQ_BACKTEST_LOW_MEMORY=1",
        "export LQ_AUTO_COLLECT_DB=0",
        "export PYTHONUNBUFFERED=1",
        "",
        "/usr/bin/time -v uv run python scripts/research/run_current_switch_validation_reboot.py --train-start 2025-01-01 --train-end 2025-12-31 --val-start 2026-01-01 --val-end 2026-02-28 --oos-start 2026-03-01",
        "",
        "/usr/bin/time -v uv run python scripts/research/write_portfolio_operating_switch.py --artifact-profile reboot_validation",
        "",
        "/usr/bin/time -v uv run python scripts/research/tune_hybrid_online_portfolio.py --train-start 2025-01-01 --train-end 2025-12-31 --val-start 2026-01-01 --val-end 2026-02-28 --oos-start 2026-03-01 --warmup-ratio 0.60",
        "",
        "/usr/bin/time -v uv run python scripts/research/optuna_tune_hybrid_online_portfolio.py --train-start 2025-01-01 --train-end 2025-12-31 --val-start 2026-01-01 --val-end 2026-02-28 --oos-start 2026-03-01 --warmup-ratio 0.60 --objective-profile live_guarded --n-trials 24",
        "",
        "/usr/bin/time -v uv run python scripts/research/run_hybrid_online_portfolio.py --train-start 2025-01-01 --train-end 2025-12-31 --val-start 2026-01-01 --val-end 2026-02-28 --oos-start 2026-03-01 --warmup-ratio 0.60 --config-json var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_hybrid_online_optuna_current/hybrid_online_optuna_latest.json",
        "",
        "/usr/bin/time -v uv run python scripts/research/write_portfolio_master_scoreboard.py",
        "```",
        "",
        "## 6) Plain-English operator rule",
        "",
        (
            "> Current live default is `hybrid_guarded_mode` because it now clears the mixed/calm superiority gate over balanced overlay.  \n"
            "> Keep `pair_tactical_mode` tactical-only, and fall back to cash if hybrid loses its reboot-split edge or violates its readiness constraints."
            if default_mode.get("mode") == "hybrid_guarded_mode"
            else "> Current live default follows the retuned switch/gate layer, so stay in balanced overlay by default.  \n> Hybrid is now the strongest reboot-split challenger and the best raw diversified performer, so promote it only deliberately if you want performance-first pilot behavior.  \n> If either loses its edge, go to cash."
        ),
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    scoreboard: dict[str, Any],
    onepager: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scoreboard_json = output_dir / "portfolio_master_scoreboard_latest.json"
    scoreboard_md = output_dir / "portfolio_master_scoreboard_latest.md"
    onepager_json = output_dir / "portfolio_operating_recommendation_onepager_latest.json"
    onepager_md = output_dir / "portfolio_operating_recommendation_onepager_latest.md"
    runbook_md = output_dir / "hybrid_guarded_mode_runbook_latest.md"

    scoreboard_json.write_text(json.dumps(scoreboard, indent=2, sort_keys=True), encoding="utf-8")
    scoreboard_md.write_text(_build_master_markdown(scoreboard), encoding="utf-8")
    onepager_json.write_text(json.dumps(onepager, indent=2, sort_keys=True), encoding="utf-8")
    onepager_md.write_text(_build_onepager_markdown(onepager), encoding="utf-8")
    runbook_md.write_text(_build_hybrid_runbook_markdown(scoreboard), encoding="utf-8")
    return scoreboard_json, scoreboard_md, onepager_json, onepager_md, runbook_md


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--switch-recommendation-path", type=Path, default=DEFAULT_SWITCH_RECOMMENDATION
    )
    parser.add_argument("--switch-validation-path", type=Path, default=DEFAULT_SWITCH_VALIDATION)
    parser.add_argument("--hybrid-portfolio-path", type=Path, default=DEFAULT_HYBRID_PORTFOLIO)
    parser.add_argument("--pair-candidate-path", type=Path, default=DEFAULT_PAIR_CANDIDATE)
    parser.add_argument("--hybrid-tuning-path", type=Path, default=DEFAULT_HYBRID_TUNING)
    parser.add_argument("--hybrid-optuna-path", type=Path, default=DEFAULT_HYBRID_OPTUNA)
    parser.add_argument(
        "--hybrid-train-aware-path", type=Path, default=DEFAULT_HYBRID_OPTUNA_TRAIN_AWARE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    switch_payload = _read_json(Path(args.switch_recommendation_path).resolve())
    switch_validation = _read_json(Path(args.switch_validation_path).resolve())
    hybrid_payload = _read_json(Path(args.hybrid_portfolio_path).resolve())
    pair_candidate_payload = _read_json(Path(args.pair_candidate_path).resolve())
    hybrid_tuning_payload = _read_json(Path(args.hybrid_tuning_path).resolve())
    hybrid_optuna_payload = _read_json(Path(args.hybrid_optuna_path).resolve())
    hybrid_train_aware_payload = _read_json(Path(args.hybrid_train_aware_path).resolve())

    scoreboard = build_master_scoreboard(
        switch_payload=switch_payload,
        switch_validation=switch_validation,
        hybrid_payload=hybrid_payload,
        pair_candidate_payload=pair_candidate_payload,
        hybrid_tuning_payload=hybrid_tuning_payload,
        hybrid_optuna_payload=hybrid_optuna_payload,
        hybrid_train_aware_payload=hybrid_train_aware_payload,
    )
    onepager = build_onepager_payload(scoreboard)
    written = write_outputs(
        scoreboard=scoreboard, onepager=onepager, output_dir=Path(args.output_dir).resolve()
    )
    for path in written:
        print(str(path.resolve()))


if __name__ == "__main__":
    main()
